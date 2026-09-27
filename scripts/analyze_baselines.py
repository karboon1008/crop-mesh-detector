"""Compare the continual mesh against the FedAvg / D-PSGD control groups.

    python scripts/analyze_baselines.py <runs dir> --setting n2 [--out report.md]

Reads <runs dir>/<setting>_<strategy>_s<seed>/continual/<arch>/{batch_logs,pooled_log}.json.

Performance: each node's pair accuracy right after local training (pre) vs right after
the exchange (post), on its own per-batch test set and on the pooled test set. Three
lenses, because our mesh skips exchanges by design and the textbook baselines never do:
  per exchange   mean gain over node-batches that actually exchanged
  per node-batch mean gain over every node-batch after batch 0 (a skipped exchange = 0)
  batch 0        the first exchange, which every strategy performs
Sustainability: exchange-phase compute (upload+exchange for weights, knowledge_extraction
+distill for ours), local-training compute (should be equal across strategies), bytes, and
total energy = compute + communication under three radios. Evaluation energy is excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

METRIC = "pair_accuracy"
EXCHANGE_PHASES = ("knowledge_extraction", "distill", "upload", "exchange")
RADIO_J_PER_BYTE = {"Wi-Fi": 0.00003, "LTE": 0.0003, "LoRa": 0.00012}  # config.yaml energy.radio_energy_j_per_byte
J_PER_KWH = 3.6e6
ORDER = ["continual", "fedavg", "fedavg_gated", "dpsgd", "dpsgd_gated"]


def mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return st.stdev(xs) if len(xs) > 1 else float("nan")


def lenses(rows):
    """rows: (batch, pre, post, exchanged) -> the three gain lenses (fractions, not pp)."""
    later = [(pre, post, ex) for b, pre, post, ex in rows if b > 0]
    exchanged = [post - pre for pre, post, ex in later if ex]
    return {
        "boot": mean(post - pre for b, pre, post, ex in rows if b == 0),
        "per_exchange": mean(exchanged),
        "per_nodebatch": mean(post - pre for pre, post, ex in later),
        "pct_up": mean(g > 0 for g in exchanged) if exchanged else float("nan"),
        "n_exchange": len(exchanged),
        "sum_exchange_gain": sum(exchanged) + sum(post - pre for b, pre, post, ex in rows if b == 0),
        "final": mean(post for b, pre, post, ex in rows if b >= max(b2 for b2, *_ in rows) - 2),
    }


def summarize_run(logs, pooled):
    own_rows, pool_rows, energy, seconds = [], [], defaultdict(float), defaultdict(float)
    up = down = 0
    pool = {(e["batch"], e["node"]): e for e in pooled}
    for log in logs:
        for r in log["per_node"].values():
            b = r["batch_idx"]
            own_rows.append((b, r["pre_distill_eval"][METRIC], r["post_distill_eval"][METRIC], r["distilled"]))
            p = pool.get((b, r["node_id"]))
            if p:
                pool_rows.append((b, p["pre"][METRIC], p["post"][METRIC], r["distilled"]))
            for phase, v in r["energy_kwh"].items():
                energy[phase] += v
            for phase, v in r["duration_s"].items():
                seconds[phase] += v
            up += r["bytes_uploaded"]
            down += r["bytes_downloaded"]
    ex_kwh = sum(energy[p] for p in EXCHANGE_PHASES)
    total_bytes = up + down
    return {
        "own": lenses(own_rows), "pooled": lenses(pool_rows) if pool_rows else None,
        "local_kwh": energy["local_train"], "exchange_kwh": ex_kwh,
        "local_s": seconds["local_train"], "exchange_s": sum(seconds[p] for p in EXCHANGE_PHASES),
        "bytes": total_bytes,
        "total_kwh": {radio: energy["local_train"] + ex_kwh + total_bytes * j / J_PER_KWH for radio, j in RADIO_J_PER_BYTE.items()},
    }


def load_runs(root, setting):
    runs = {}
    for d in sorted(Path(root).glob(f"{setting}_*_s*")):
        m = re.fullmatch(rf"{setting}_(.+)_s(\d+)", d.name)
        if not m or not (d / "continual").is_dir():
            continue
        for arch_dir in sorted((d / "continual").iterdir()):
            logs_path = arch_dir / "batch_logs.json"
            if not logs_path.exists():
                continue
            pooled_path = arch_dir / "pooled_log.json"
            runs[(m.group(1), int(m.group(2)), arch_dir.name)] = summarize_run(
                json.loads(logs_path.read_text()), json.loads(pooled_path.read_text()) if pooled_path.exists() else [],
            )
    return runs


def pm(xs, scale=100.0, nd=1):
    xs = [x * scale for x in xs]
    return f"{mean(xs):.{nd}f} ± {sd(xs):.{nd}f}" if len(xs) > 1 else f"{mean(xs):.{nd}f}"


def build_report(runs, setting):
    strategies = [s for s in ORDER if any(k[0] == s for k in runs)]
    seeds = sorted({k[1] for k in runs})
    archs = sorted({k[2] for k in runs})
    R = lambda s: [v for k, v in runs.items() if k[0] == s]
    L = [f"# {setting}: continual mesh vs FedAvg / D-PSGD(ring) — seeds {seeds}, archs {archs}", "",
         "Mean ± sd over arch×seed runs. pp = percentage points of pair accuracy. "
         "'gated' variants use the continual mesh's own EMA upload/retrieve rules.", ""]

    for scope, title in (("own", "own per-batch test set"), ("pooled", "pooled test set (all nodes' classes)")):
        L += [f"## Performance across the exchange — {title}", "",
              "| strategy | batch-0 gain (pp) | gain / exchange (pp) | gain / node-batch (pp) | % exchanges that helped | exchanges / run | final acc (%) |",
              "|---|---|---|---|---|---|---|"]
        for s in strategies:
            v = [r[scope] for r in R(s) if r[scope]]
            if not v:
                continue
            L.append(f"| {s} | {pm([x['boot'] for x in v])} | {pm([x['per_exchange'] for x in v])} | "
                     f"{pm([x['per_nodebatch'] for x in v])} | {pm([x['pct_up'] for x in v], 100, 0)} | "
                     f"{mean(x['n_exchange'] for x in v):.0f} | {pm([x['final'] for x in v])} |")
        L.append("")

    L += ["## Sustainability — exchange step only, plus totals", "",
          "| strategy | exchange compute (Wh) | exchange time (s) | local-train compute (Wh) | bytes (MB) | total energy Wi-Fi / LTE / LoRa (Wh) | bytes vs continual |",
          "|---|---|---|---|---|---|---|"]
    base_bytes = mean(r["bytes"] for r in R("continual")) if "continual" in strategies else float("nan")
    for s in strategies:
        v = R(s)
        tot = " / ".join(f"{mean(r['total_kwh'][radio] for r in v) * 1000:.1f}" for radio in RADIO_J_PER_BYTE)
        ratio = mean(r["bytes"] for r in v) / base_bytes if base_bytes and not math.isnan(base_bytes) else float("nan")
        L.append(f"| {s} | {mean(r['exchange_kwh'] for r in v) * 1000:.2f} | {mean(r['exchange_s'] for r in v):.1f} | "
                 f"{mean(r['local_kwh'] for r in v) * 1000:.1f} | {mean(r['bytes'] for r in v) / 1e6:.2f} | {tot} | {ratio:.1f}× |")
    L += ["", "Efficiency: total pooled-test gain (pp, summed over every exchange) per MB and per Wh of exchange energy:", "",
          "| strategy | pp per MB | pp per Wh (exchange compute) |", "|---|---|---|"]
    for s in strategies:
        v = [r for r in R(s) if r["pooled"]]
        if not v:
            continue
        per_mb = mean(100 * r["pooled"]["sum_exchange_gain"] / (r["bytes"] / 1e6) for r in v if r["bytes"])
        per_wh = mean(100 * r["pooled"]["sum_exchange_gain"] / (r["exchange_kwh"] * 1000) for r in v if r["exchange_kwh"])
        L.append(f"| {s} | {per_mb:.2f} | {per_wh:.1f} |")
    L += ["", "Sanity: local-train compute should match across strategies (same stream, same epochs); "
          "a gap means the comparison is not like-for-like."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--setting", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report = build_report(load_runs(a.runs_dir, a.setting), a.setting)
    print(report)
    if a.out:
        Path(a.out).write_text(report, encoding="utf-8")
