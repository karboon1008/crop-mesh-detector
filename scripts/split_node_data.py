"""One-time pre-split of PlantVillage into physically isolated per-node
folders plus a shared public probe folder, mirroring exactly the
probe/shard split src/train.py's build_dataloaders performs in-memory —
but writing files to disk so each Docker node container can be given a
read-only bind mount of only its own folder.

Run from the repo root:
    python scripts/split_node_data.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.plantvillage import (
    build_global_label_map,
    carve_public_probe_set,
    load_full_dataset,
    partition_nodes,
    save_global_label_map,
)


def _copy_indices(dataset, indices: list[int], dest_root: Path) -> None:
    for idx in indices:
        src_path, class_idx = dataset.base.samples[idx]
        class_name = dataset.base.classes[class_idx]
        dest_dir = dest_root / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_dir / Path(src_path).name)


def split(cfg: Config, output_root: Path) -> None:
    # Clear stale output_root to ensure clean split every run (prevents old files
    # from previous runs undermining the per-node data isolation)
    shutil.rmtree(output_root, ignore_errors=True)

    dataset = load_full_dataset(cfg.get("data.root"), cfg.get("data.image_size", 160))
    global_map = build_global_label_map(dataset)
    output_root.mkdir(parents=True, exist_ok=True)
    save_global_label_map(global_map, output_root / "classes.json")

    probe_idx, remaining_idx = carve_public_probe_set(
        dataset, cfg.get("data.probe_set_fraction", 0.05), cfg.get("data.seed", 42)
    )
    _copy_indices(dataset, probe_idx, output_root / "probe")

    shards = partition_nodes(
        dataset,
        remaining_idx,
        cfg.get("data.num_nodes", 3),
        cfg.get("data.non_iid_strategy", "manual"),
        cfg.get("data.dirichlet_alpha", 0.3),
        cfg.get("data.seed", 42),
        manual_node_crops=cfg.get("data.manual_node_crops", None),
    )
    for i, shard in enumerate(shards):
        _copy_indices(dataset, shard, output_root / f"node_{i}")


def main() -> None:
    cfg = Config.load()
    output_root = Path(cfg.get("docker_mesh.data_dir", "data/docker_mesh"))
    split(cfg, output_root)
    print(f"Wrote split dataset + classes.json to {output_root}")


if __name__ == "__main__":
    main()
