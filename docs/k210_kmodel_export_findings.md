# K210 kmodel export: pipeline, environment setup, and results

This documents the actual, verified PyTorch → ONNX → `.kmodel` conversion
pipeline for deploying a mesh-trained checkpoint to the Yahboom K210
developer kit, and the concrete result per architecture. Everything here was
run end-to-end on a Windows dev machine, not just planned — every error
below is a real error hit and fixed, not a hypothetical.

(This is a different, complementary track to
[k210_riscv_c_sdk_benchmark.md](k210_riscv_c_sdk_benchmark.md), which covers
measuring real on-device energy via the bare-metal C SDK. This doc is about
the standalone Python/nncase conversion pipeline and which architectures
actually fit on the chip.)

## TL;DR

| Architecture | K210 kmodel result |
|---|---|
| `mobilenet_v3_small` | ✅ **Success** — compiles to a working `model.kmodel` (4.88MB total, fits the K210's memory budget) |
| `efficientnet_lite0` | ❌ Compiles all the way through PTQ calibration and code generation, but the final buffer allocator runs out of on-chip SRAM — its activation maps are too large for the K210, even though the weight count alone would fit |
| `mobilevit_xxs` | ❌ Can't even export to a K210-compatible ONNX opset — its attention op needs opset ≥14, and the K210 KPU has no attention hardware regardless (a hard ceiling, not a bug) |

**Use `mobilenet_v3_small` for the K210.** Note this *reverses* the
Raspberry Pi ranking, where `efficientnet_lite0` scored higher on accuracy
(see `outputs/results_summary.json`) — on the K210, memory footprint wins
over raw accuracy because the bigger model simply doesn't fit.

---

## 1. The pipeline (3 scripts, 2 Python environments)

```
PyTorch checkpoint (outputs/checkpoints/<arch>/<node>.pt)
        │
        ├─► scripts/export_for_pi.py           [MAIN .venv]
        │     produces: outputs/pi_export/<arch>/model.onnx        (opset 17, int8, for the Pi)
        │           and outputs/pi_export/<arch>/model_fp32.onnx   (opset 17, plain float)
        │
        ├─► scripts/export_onnx_k210.py         [MAIN .venv]
        │     re-exports at a LOWER opset (11) -- opset 17 isn't parseable
        │     by K210's 2021-era nncase compiler
        │     produces: outputs/pi_export/<arch>/model_fp32_k210.onnx
        │
        └─► scripts/export_for_k210.py          [SEPARATE .venv-nncase]
              onnx-simplify + shape-inference fixups, then nncase PTQ
              calibration + K210 code generation
              produces: outputs/k210_export/<arch>/model.kmodel
```

Why two environments: nncase's K210-targeting 1.0 release only ships wheels
for Python ≤3.10 (confirmed via PyPI's file listing for
`nncase==1.0.0.20211029`), while this repo's main `.venv` is on a newer
Python. K210 support lives on nncase's separate `release/1.0` branch — the
current `pip install nncase` (2.x line) targets the newer K230 chip instead
and won't work for K210 at all.

## 2. Setting up `.venv-nncase` (one-time)

```bash
# 1. Get a Python <=3.10 (this machine only had 3.13 installed)
winget install Python.Python.3.10 --scope user

# 2. Create a separate venv -- do NOT install nncase into the main .venv
py -3.10 -m venv .venv-nncase
.venv-nncase\Scripts\python -m pip install --upgrade pip

# 3. If on a network with a TLS-inspecting proxy (common on corporate
#    laptops), fix cert trust before anything else -- see step 3 below
.venv-nncase\Scripts\python -m pip install pip-system-certs

# 4. Install the K210-targeting nncase release specifically (not latest)
.venv-nncase\Scripts\python -m pip install nncase==1.0.0.20211029 Pillow numpy onnx onnxsim
```

### Environment issues hit and fixed along the way

1. **Corporate TLS-inspection proxy → SSL errors.** Same fix as the main
   `.venv` needed for the dataset download (`pip-system-certs`, which patches
   Python to trust the same certificate store Windows already does). Needs
   installing separately per-venv.

2. **`ImportError: DLL load failed while importing _nncase`.** This 2021
   package never calls `os.add_dll_directory()` for its own native DLLs —
   a pre-Python-3.8 assumption about Windows' DLL search path that no longer
   holds. Fixed by calling it ourselves before `import nncase`:
   ```python
   os.add_dll_directory(os.path.join(sys.prefix, "Lib", "site-packages"))
   ```
   (This is already baked into `scripts/export_for_k210.py` — not something
   you need to do by hand.)

3. **Still failing after that: `nncase.dll` itself couldn't find one of its
   dependencies.** Inspecting its import table with `pefile` (`pip install
   pefile`) pinpointed the exact missing library: `libomp140.x86_64.dll`
   (LLVM's OpenMP runtime), which isn't part of the standard Visual C++
   Redistributable. Fixed by copying an existing, functionally-equivalent
   copy already on this machine and renaming it — Microsoft's
   `libomp140.x86_64.dll` is a renamed redistribution of LLVM's own
   `libomp.dll`, a well-documented interchangeable substitution:
   ```bash
   cp "C:\Program Files\LLVM\bin\libomp.dll" ".venv-nncase\Lib\site-packages\libomp140.x86_64.dll"
   ```
   (Only needed if your machine doesn't already have `libomp140.x86_64.dll`
   somewhere on its DLL search path.)

## 3. Running the export

```bash
# In the MAIN .venv (needs torch/timm):
python scripts/export_for_pi.py --arch mobilenet_v3_small --node node_2
python scripts/export_onnx_k210.py --arch mobilenet_v3_small --node node_2

# In .venv-nncase (needs nncase, NOT torch):
.venv-nncase\Scripts\python scripts\export_for_k210.py --arch mobilenet_v3_small --node node_2
```

`export_for_k210.py` prints nncase's own stage-by-stage progress
(`1. Import graph... 2. Optimize target independent... ... 8. Generate
code...`), then a memory-usage summary, then writes
`outputs/k210_export/<arch>/model.kmodel` + a copy of `manifest.json`.

### ONNX-side fixes needed inside `export_for_k210.py`

Two graph-level issues had to be fixed before nncase's importer would accept
the model at all — both are now handled automatically by the script, not
manual steps:

1. **`RuntimeError: Bad optional access`** on `import_onnx()` — turned out to
   be an ONNX opset-17-specific construct (a `Shape` op behavior introduced
   between opset 15–17) that this 2021-era importer can't parse. Confirmed by
   trying to downgrade the opset after the fact with `onnx.version_converter`,
   which itself failed with "No Adapter From Version 15 for Shape" — proving
   it's a genuine graph-construct incompatibility, not something fixable
   after export. Fixed by re-exporting directly from PyTorch at **opset 11**
   instead (`scripts/export_onnx_k210.py`), which predates the problem opset
   version entirely. (Only viable because `mobilenet_v3_small` and
   `efficientnet_lite0` don't need `mobilevit_xxs`'s opset≥14 attention op.)

2. **`RuntimeError: Can't find value info for .../Gather_output_0 to parse
   its shape`** — `torch.onnx.export()` doesn't populate shape metadata
   (`value_info`) for every intermediate tensor by default; nncase's importer
   requires it for all of them. Fixed with ONNX's own shape-inference pass:
   ```python
   onnx_model = onnx.shape_inference.infer_shapes(onnx.load(fp32_path))
   ```

3. **`RuntimeError: Can't pull input data for .../Sub_6_output_0: only
   constant initialization is supported`** — `tf_efficientnet_lite0`'s
   TensorFlow-style "same" padding computes its padding amount dynamically
   from the input shape (`Shape`/`Gather`/`Sub` ops), which PyTorch's
   exporter leaves as live graph nodes even though, with a fixed input size,
   they're always the same value and could be folded to constants. Fixed
   with `onnx-simplifier`:
   ```python
   from onnxsim import simplify
   onnx_model, _ = simplify(onnx_model)
   ```
   This is a well-known standard step in the K210/nncase community
   specifically for this class of TF-ported-model padding issue.

## 4. Why `efficientnet_lite0` still fails (and it's not a bug)

With all three fixes above in place, `efficientnet_lite0` gets all the way
through import, optimization, PTQ calibration, quantization, and code
generation — every stage prints successfully — and only fails at the very
last step, `gencode_tobytes()`:

```
RuntimeError: Allocator has ran out of memory
```

This is nncase allocating the K210's actual on-chip SRAM working set for
weights + activation buffers during code generation, not a host-machine
memory issue. `efficientnet_lite0` (3.42M params) has larger intermediate
activation maps than `mobilenet_v3_small` (1.55M params) does, and those
don't fit in the K210's limited SRAM even though the model's raw weight size
alone would be storable in the board's flash. This matches the memory
constraint flagged early in K210 planning discussions (~2MB usable KPU
working memory on most K210 dev boards) — now confirmed empirically rather
than just estimated.

`mobilenet_v3_small`'s successful compile reports:
```
MEMORY USAGES
.input   300.00 KB
.output  140.00 B
.data    321.88 KB
MODEL    4.88 MB
TOTAL    5.49 MB
```

## 5. Why `mobilevit_xxs` fails, definitively

Attempting `scripts/export_onnx_k210.py --arch mobilevit_xxs` fails before
nncase is even involved:

```
torch.onnx.errors.UnsupportedOperatorError: Exporting the operator
'aten::scaled_dot_product_attention' to ONNX opset version 11 is not
supported. Support for this operator was added in version 14
```

Using opset 14+ would re-introduce the opset-17-class importer
incompatibilities described above, but it's moot regardless: the K210 KPU
is a fixed-function CNN accelerator with no self-attention support in
silicon. No amount of ONNX-side fixing changes that. `mobilevit_xxs` is not
a realistic K210 target — this was anticipated before conversion was even
attempted, and this run confirms it concretely rather than just in theory.

## 6. Files this pipeline touches

- `scripts/export_for_pi.py` — modified to stop deleting `model_fp32.onnx`
  after quantization (it's now a kept output, both for general reuse and
  because `export_onnx_k210.py` depends on this bundle existing)
- `scripts/export_onnx_k210.py` — new; re-exports at opset 11 for K210
- `scripts/export_for_k210.py` — new; the actual nncase-based ONNX→kmodel
  converter, including the onnx-simplifier + shape-inference fixups
- `.venv-nncase/` — new, separate Python 3.10 environment (not committed to
  the repo; gitignore-equivalent to `.venv/`)
