"""Bayesian hyperparameter search over the mesh training config, using Optuna.

Each trial samples a set of `training.*` values, writes them into a scratch
copy of config.yaml, and runs `python -m src.train --arch <arch>` as a
subprocess against a dedicated trial output directory. The trial's score
comes from that run's own results: each node's final post-distill accuracy
on its last continual batch's private test set, averaged over nodes, to
maximize, and the run's total compute energy (from the existing CodeCarbon
tracking) to minimize — a two-objective Optuna study whose result is a Pareto front
rather than one "best" config.

Note: the default search space below (lr, distill_lr, proto_weight,
kd_weight, kd_temperature) doesn't change epoch/round counts, so compute
energy will vary only by measurement noise across trials — the energy
objective only becomes meaningful once the search space also covers
something that changes compute (e.g. local_epochs_per_round, rounds,
batch_size, continual.num_batches, or architecture).

Usage:
    python -m scripts.tune_hyperparams --n-trials 30 --arch efficientnet_lite0

Resume an interrupted study (same --study-name/--storage db):
    python -m scripts.tune_hyperparams --n-trials 20 --study-name mesh-hpo
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import optuna
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "config.yaml"

# dotted config key -> (kind, low, high). "loguniform" for scale-free params
# (learning rates), "uniform" for weights/temperatures.
SEARCH_SPACE = {
    "training.lr": ("loguniform", 1e-4, 1e-2),
    "training.distill_lr": ("loguniform", 1e-5, 5e-3),
    "training.proto_weight": ("uniform", 0.0, 1.0),
    "training.kd_weight": ("uniform", 0.0, 1.0),
    "training.kd_temperature": ("uniform", 1.0, 4.0),
}

# Artifacts safe to discard once a trial's two objective values are read out;
# config.yaml and metrics.json are kept so every trial stays reproducible.
_CLEANUP_DIRS = ("checkpoints", "continual")
_CLEANUP_GLOBS = ("emissions.csv", "sustainability_report.*")


def _set_nested(data: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    node = data
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def _suggest(trial: optuna.Trial, name: str, spec: tuple) -> float:
    kind, low, high = spec
    if kind == "loguniform":
        return trial.suggest_float(name, low, high, log=True)
    if kind == "uniform":
        return trial.suggest_float(name, low, high)
    raise ValueError(f"Unknown search-space kind '{kind}' for '{name}'")


def _accuracy_score(arch_result: dict) -> float:
    evals = arch_result["mesh_eval"].values()
    return sum((e["crop_accuracy"] + e["disease_accuracy"]) / 2 for e in evals) / len(evals)


def _cleanup_trial_dir(trial_dir: Path) -> None:
    for name in _CLEANUP_DIRS:
        shutil.rmtree(trial_dir / name, ignore_errors=True)
    for pattern in _CLEANUP_GLOBS:
        for f in trial_dir.glob(pattern):
            f.unlink()


def run_trial(trial: optuna.Trial, arch: str, base_config: dict, trials_dir: Path) -> tuple[float, float]:
    trial_config = copy.deepcopy(base_config)
    for dotted_key, spec in SEARCH_SPACE.items():
        _set_nested(trial_config, dotted_key, _suggest(trial, dotted_key, spec))

    trial_dir = trials_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_config["output"]["dir"] = str(trial_dir)
    trial_config_path = trial_dir / "config.yaml"
    trial_config_path.write_text(yaml.safe_dump(trial_config, sort_keys=False))

    result = subprocess.run(
        [sys.executable, "-m", "src.train", "--config", str(trial_config_path), "--arch", arch],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise optuna.TrialPruned(f"training subprocess failed for trial {trial.number}")

    arch_result = json.loads((trial_dir / f"results_{arch}.json").read_text())
    return _accuracy_score(arch_result), arch_result["continual"]["total_compute_energy_kwh"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="efficientnet_lite0",
                         help="Architecture to tune (config.yaml's models.architectures entries)")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--config", default=str(BASE_CONFIG_PATH), help="Base config.yaml to start each trial from")
    parser.add_argument("--study-name", default="mesh-hpo")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs_tuning"))
    parser.add_argument("--storage", default=None,
                         help="Optuna storage URL (default: sqlite db under --output-root, enabling resume)")
    parser.add_argument("--keep-trial-outputs", action="store_true",
                         help="Keep each trial's full outputs/ (checkpoints, plots, csvs) instead of "
                              "pruning them down to config.yaml + metrics.json after scoring")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    trials_dir = output_root / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{output_root / 'study.db'}"

    base_config = yaml.safe_load(Path(args.config).read_text())

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        directions=["maximize", "minimize"],  # (accuracy, compute_energy_kwh)
    )

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        trial_dir = trials_dir / f"trial_{trial.number:04d}"
        accuracy, energy_kwh = run_trial(trial, args.arch, base_config, trials_dir)
        (trial_dir / "metrics.json").write_text(
            json.dumps({"accuracy": accuracy, "energy_kwh": energy_kwh, "params": trial.params}, indent=2)
        )
        if not args.keep_trial_outputs:
            _cleanup_trial_dir(trial_dir)
        print(f"trial {trial.number}: accuracy={accuracy:.4f} energy_kwh={energy_kwh:.5f} params={trial.params}")
        return accuracy, energy_kwh

    study.optimize(objective, n_trials=args.n_trials)

    print(f"\nPareto-optimal trials ({len(study.best_trials)}):")
    for t in study.best_trials:
        print(f"  trial {t.number}: accuracy={t.values[0]:.4f} energy_kwh={t.values[1]:.5f} params={t.params}")

    pareto_path = output_root / "pareto_front.json"
    pareto_path.write_text(json.dumps(
        [{"trial": t.number, "accuracy": t.values[0], "energy_kwh": t.values[1], "params": t.params}
         for t in study.best_trials],
        indent=2,
    ))
    print(f"\nPareto front written to {pareto_path}")


if __name__ == "__main__":
    main()
