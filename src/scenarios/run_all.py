"""Runs all three challenge scenarios end to end and consolidates them —
the single command a reviewer can trigger to see the mesh handle node
disconnection, a class appearing at run time, and a local distribution
shift, then read the results as one table.

Each scenario runs in its own subprocess so that one failure (a missing
dataset, an out-of-memory kill) neither takes down the others nor leaves a
half-mutated in-process state behind; the exit status of every scenario is
reported, and the consolidation step still runs over whatever reports did
get written.

Run:
    python -m src.scenarios.run_all                       # all three, then summarise
    python -m src.scenarios.run_all --only disconnection  # one scenario
    python -m src.scenarios.run_all --summarise-only      # re-consolidate existing reports
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.config import Config
from src.scenarios.summarise import SCENARIOS, write_summary


def run_scenario_module(name: str, config: str | None, arch: str | None) -> tuple[int, float]:
    cmd = [sys.executable, "-m", f"src.scenarios.{name}"]
    if config:
        cmd += ["--config", config]
    if arch:
        cmd += ["--arch", arch]
    print(f"\n=== {name}: {' '.join(cmd)} ===", flush=True)
    start = time.time()
    exit_code = subprocess.call(cmd)
    return exit_code, time.time() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="config.yaml path passed through to each scenario")
    parser.add_argument("--arch", default=None, help="architecture passed through to each scenario")
    parser.add_argument(
        "--only", action="append", choices=SCENARIOS,
        help="run only this scenario (repeatable); defaults to all three",
    )
    parser.add_argument(
        "--summarise-only", action="store_true",
        help="skip the runs and only re-consolidate the reports already in outputs/scenarios/",
    )
    args = parser.parse_args()

    output_dir = Path(Config.load(args.config).get("output.dir", "outputs"))
    names = tuple(args.only) if args.only else SCENARIOS

    results: list[tuple[str, int, float]] = []
    if not args.summarise_only:
        for name in names:
            exit_code, elapsed = run_scenario_module(name, args.config, args.arch)
            results.append((name, exit_code, elapsed))

        print("\n=== scenario run summary ===")
        for name, exit_code, elapsed in results:
            print(f"{'OK  ' if exit_code == 0 else 'FAIL'} {name:<20} {elapsed / 60:6.1f} min  (exit {exit_code})")

    print("\n=== consolidating ===")
    try:
        json_path = write_summary(output_dir, names)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    print(f"Wrote {json_path}, {json_path.with_name('summary.md')}, {json_path.with_name('summary.csv')}")

    return 1 if any(exit_code != 0 for _, exit_code, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
