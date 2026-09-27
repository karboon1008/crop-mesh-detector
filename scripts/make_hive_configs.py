"""Writes the hive-experiment configs from configs/n4.yaml and configs/n2.yaml,
and prints the per-node class counts of the n4 split (to pick the withheld disease).

usage: python scripts/make_hive_configs.py [withheld_class]
"""

import copy
import sys
from collections import Counter
from pathlib import Path

import yaml

N4 = yaml.safe_load(Path("configs/n4.yaml").read_text())
N2 = yaml.safe_load(Path("configs/n2.yaml").read_text())


def write(name, base, **updates):
    cfg = copy.deepcopy(base)
    for dotted, value in updates.items():
        section, key = dotted.split(".")
        cfg.setdefault(section, {})[key] = value
    Path(f"configs/{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("wrote", name)


if len(sys.argv) == 1:
    from src.config import Config
    from src.data.multi_source import load_dataset
    from src.data.stream import DataStream

    cfg = Config(N4)
    dataset = load_dataset(cfg)
    stream = DataStream.create(cfg, dataset, Path("/tmp/n4_stream_probe.json"))
    for node_i, shard in enumerate(stream.node_shards):
        counts = Counter(dataset.base.classes[dataset.targets[i]] for i in shard)
        print(f"node_{node_i}", dict(sorted(counts.items())))
else:
    offline = {"node_1": [6, 7, 8, 9]}
    write("n4_q4_withhold", N4, **{"continual.withhold": {"node": "node_0", "classes": [sys.argv[1]], "until_batch": 8}})
    write("n4_q5_offline_keep", N4, **{"continual.offline": offline})
    write("n4_q5_offline_purge", N4, **{"continual.offline": offline, "continual.offline_purge": True})
    write("src2", N2, **{
        "data.num_nodes": 2, "data.non_iid_strategy": "by_source", "data.included_crops": ["Tomato"],
        "data.source_nodes": {"plantvillage": "node_0", "plantdoc": "node_1", "plantwild": "node_1"},
    })
