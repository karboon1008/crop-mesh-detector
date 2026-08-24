"""Converts a trained checkpoint's FP32 ONNX export (from
scripts/export_for_pi.py) into a .kmodel for the Kendryte K210 (e.g. the
Yahboom K210 developer kit), via Kendryte's nncase 1.0 compiler.

IMPORTANT — separate Python environment required: nncase's K210-targeting
1.0 release only ships wheels for Python <=3.10, so this script must be run
with a dedicated venv, not the repo's main .venv/ (which is on a newer
Python for everything else). One-time setup:

    py -3.10 -m venv .venv-nncase
    .venv-nncase\\Scripts\\python -m pip install nncase==1.0.0.20211029 Pillow numpy

Run (after 'python scripts/export_for_pi.py --all' in the MAIN .venv has
produced model_fp32.onnx bundles under outputs/pi_export/<arch>/):

    .venv-nncase\\Scripts\\python scripts\\export_for_k210.py --arch efficientnet_lite0
    .venv-nncase\\Scripts\\python scripts\\export_for_k210.py --all

Note: mobilevit_xxs's self-attention ops are not supported by the K210 KPU
or nncase's ONNX importer and are expected to fail here — that's a hardware
limitation, not a bug in this script. mobilenet_v3_small and
efficientnet_lite0 (plain conv/depthwise-conv architectures) are the
realistic K210 targets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# nncase's own package never calls os.add_dll_directory() for its native
# DLLs (it predates Python 3.8's stricter Windows DLL search-path rules) --
# without this, `import nncase` fails with "DLL load failed".
os.add_dll_directory(os.path.join(sys.prefix, "Lib", "site-packages"))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import shape_inference  # noqa: E402
from onnxsim import simplify  # noqa: E402
from PIL import Image  # noqa: E402
import nncase  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.model_selection import list_architectures, pick_best_arch_node, pick_best_node_for_arch  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_calibration_batch(
    data_root: Path, image_size: int, mean: list[float], std: list[float], num_samples: int
) -> tuple[np.ndarray, int]:
    """Loads a handful of real images spread across classes, preprocessed
    identically to training (resize -> [0,1] -> normalize -> CHW), for
    nncase's post-training quantization (PTQ) calibration.
    """
    class_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under {data_root}")

    step = max(1, len(class_dirs) // num_samples)
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)

    images = []
    for class_dir in class_dirs[::step][:num_samples]:
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        if not files:
            continue
        img = Image.open(files[0]).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - mean_arr) / std_arr
        images.append(arr.transpose(2, 0, 1))  # HWC -> CHW

    if not images:
        raise RuntimeError(f"Could not find any calibration images under {data_root}")

    batch = np.stack(images, axis=0).astype(np.float32)  # (N, 3, H, W)
    return batch, batch.shape[0]


def export_one(
    arch: str,
    node_id: str,
    pi_export_dir: Path,
    output_dir: Path,
    data_root: Path,
    num_calibration_samples: int,
) -> None:
    bundle_dir = pi_export_dir / arch
    fp32_path = bundle_dir / "model_fp32_k210.onnx"
    manifest_path = bundle_dir / "manifest.json"
    if not fp32_path.exists():
        raise FileNotFoundError(
            f"{fp32_path} not found. In the MAIN .venv (not this nncase one), run:\n"
            f"    python scripts/export_for_pi.py --arch {arch}\n"
            f"    python scripts/export_onnx_k210.py --arch {arch} --node {node_id}"
        )
    manifest = json.loads(manifest_path.read_text())
    image_size = manifest["image_size"]
    mean, std = manifest["mean"], manifest["std"]

    print(f"[{arch}] Loading calibration images from {data_root}...")
    calib_batch, n_samples = load_calibration_batch(data_root, image_size, mean, std, num_calibration_samples)
    print(f"[{arch}] {n_samples} calibration samples, shape {calib_batch.shape}")

    compile_options = nncase.CompileOptions()
    compile_options.target = "k210"
    compile_options.input_type = "float32"
    compile_options.output_type = "float32"
    # quant_type / w_quant_type default to 'uint8' already, which is what
    # the K210 KPU requires -- left as default rather than restated.

    compiler = nncase.Compiler(compile_options)
    # Two fixups nncase's 2021-era importer needs that torch.onnx.export()
    # doesn't provide on its own:
    # 1. onnx-simplifier folds shape-dependent arithmetic (e.g. TF-style
    #    "same"-padding computed from Shape/Gather/Sub ops) into constants,
    #    now that the input shape is fixed -- without this, nncase errors
    #    with "only constant initialization is supported".
    # 2. ONNX's own shape inference populates a value_info entry for every
    #    intermediate tensor -- without this, nncase errors with
    #    "Can't find value info ... to parse its shape".
    onnx_model = onnx.load(str(fp32_path))
    onnx_model, _ = simplify(onnx_model)
    onnx_model = shape_inference.infer_shapes(onnx_model)
    compiler.import_onnx(onnx_model.SerializeToString(), nncase.ImportOptions())

    ptq_options = nncase.PTQTensorOptions()
    ptq_options.samples_count = n_samples
    ptq_options.set_tensor_data(calib_batch.tobytes())
    compiler.use_ptq(ptq_options)

    print(f"[{arch}] Compiling to kmodel (PTQ calibration + K210 codegen)...")
    compiler.compile()
    kmodel_bytes = compiler.gencode_tobytes()

    out_dir = output_dir / arch
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.kmodel").write_bytes(kmodel_bytes)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{arch}] Wrote {out_dir / 'model.kmodel'} ({len(kmodel_bytes) / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi-export-dir", default="outputs/pi_export", help="Where scripts/export_for_pi.py wrote model_fp32.onnx bundles")
    parser.add_argument("--results", default="outputs/results_summary.json")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--node", default=None)
    parser.add_argument("--all", action="store_true", help="Attempt every architecture in results_summary.json")
    parser.add_argument("--data-root", default="data/PlantVillage", help="Used for PTQ calibration images")
    parser.add_argument("--num-calibration-samples", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/k210_export")
    args = parser.parse_args()

    pi_export_dir = Path(args.pi_export_dir)
    output_dir = Path(args.output_dir)
    data_root = Path(args.data_root)

    if args.all:
        archs = list_architectures(Path(args.results))
        print(f"Attempting all {len(archs)} architecture(s): {archs}")
        succeeded, failed = [], []
        for arch in archs:
            node_id, score = pick_best_node_for_arch(Path(args.results), arch)
            print(f"\n=== {arch} (best node: {node_id}, avg test accuracy {score:.4f}) ===")
            try:
                export_one(arch, node_id, pi_export_dir, output_dir, data_root, args.num_calibration_samples)
                succeeded.append(arch)
            except Exception as e:
                print(f"[{arch}] FAILED: {e}")
                print(
                    f"[{arch}] (some architectures -- e.g. mobilevit_xxs's self-attention -- are not "
                    f"supported by the K210 KPU; this is a hardware limitation, not necessarily a bug here)"
                )
                failed.append(arch)
        print(f"\nDone. Succeeded: {succeeded}. Failed: {failed}.")
        return

    if args.arch and args.node:
        arch, node_id = args.arch, args.node
    else:
        arch, node_id = pick_best_arch_node(Path(args.results))
    export_one(arch, node_id, pi_export_dir, output_dir, data_root, args.num_calibration_samples)
    print(f"\nDone. Copy {output_dir / arch}/ to the K210's flash/SD card.")


if __name__ == "__main__":
    main()
