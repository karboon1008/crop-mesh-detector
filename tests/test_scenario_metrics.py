"""Tests for the derived scenario metrics (adaptation / retention /
gain-per-cost) and the cross-scenario summariser that publishes them.

These run on hand-built report dicts rather than on a real simulation, so
each metric's arithmetic is pinned to a case whose expected answer can be
read off by eye — which is what makes the figures quoted in the research
document checkable.
"""

from __future__ import annotations

import json

import pytest

from src.scenarios import metrics, summarise


def _round(idx, mesh, baseline, *, events=None, bytes_=1000, comm_j=0.03, base_kwh=0.0, mesh_kwh=0.0,
           macro_gain=None, active=("node_0", "node_1")):
    """One round record in the shape write_scenario_report emits."""
    return {
        "round_idx": idx,
        "baseline_eval": {"node_0": {"crop_accuracy": baseline}},
        "mesh_eval": {"node_0": {"crop_accuracy": mesh}},
        "collaboration_gain": {"macro_gain": macro_gain if macro_gain is not None else {"crop_accuracy": mesh - baseline}},
        "events": events or [],
        "active_nodes": list(active),
        "total_bytes_exchanged": bytes_,
        "baseline_compute_energy_kwh": base_kwh,
        "mesh_compute_energy_kwh": mesh_kwh,
        "communication_energy_j": comm_j,
    }


def _report(rounds, target="node_0", scenario="unit_test", **extra):
    return {"scenario": scenario, "target_node": target, "rounds": rounds, "summary": {}, **extra}


def test_adaptation_gain_averages_the_target_advantage_after_the_disruption():
    rounds = [
        _round(0, mesh=0.90, baseline=0.80),  # pre-disruption: +10pp
        _round(1, mesh=0.92, baseline=0.80),  # pre-disruption: +12pp
        _round(2, mesh=0.80, baseline=0.60),  # post: +20pp
        _round(3, mesh=0.90, baseline=0.60),  # post: +30pp
    ]
    result = metrics.adaptation_gain(rounds, "node_0", disruption_start_round=2)

    assert result["adaptation_gain"]["crop_accuracy"] == pytest.approx(0.25)
    assert result["pre_disruption_advantage"]["crop_accuracy"] == pytest.approx(0.11)
    assert result["advantage_delta"]["crop_accuracy"] == pytest.approx(0.14)


def test_adaptation_gain_has_no_delta_when_the_disruption_lands_on_round_zero():
    rounds = [_round(0, mesh=0.9, baseline=0.8), _round(1, mesh=0.9, baseline=0.7)]
    result = metrics.adaptation_gain(rounds, "node_0", disruption_start_round=0)

    assert result["adaptation_gain"]["crop_accuracy"] == pytest.approx(0.15)
    assert result["pre_disruption_advantage"] == {}
    assert result["advantage_delta"] is None


def test_retention_gain_credits_whichever_arm_gave_up_less():
    # pre-disruption reference is round 1. The mesh dips to 0.88 (a 4pp drop
    # from 0.92); local-only dips to 0.60 (a 20pp drop from 0.80), so the
    # mesh retained 16pp more of its prior accuracy.
    rounds = [
        _round(0, mesh=0.90, baseline=0.80),
        _round(1, mesh=0.92, baseline=0.80),
        _round(2, mesh=0.88, baseline=0.60),
        _round(3, mesh=0.95, baseline=0.85),
    ]
    result = metrics.retention_gain(rounds, "node_0", disruption_start_round=2)

    assert result["mesh_drop"]["crop_accuracy"] == pytest.approx(0.04)
    assert result["baseline_drop"]["crop_accuracy"] == pytest.approx(0.20)
    assert result["retention_gain"]["crop_accuracy"] == pytest.approx(0.16)


def test_retention_gain_reports_a_zero_drop_rather_than_a_negative_one():
    # an arm that only ever improves after the disruption has retained
    # everything; a "negative drop" would otherwise read as a bonus.
    rounds = [_round(0, mesh=0.80, baseline=0.80), _round(1, mesh=0.95, baseline=0.90)]
    result = metrics.retention_gain(rounds, "node_0", disruption_start_round=1)

    assert result["mesh_drop"]["crop_accuracy"] == pytest.approx(0.0)
    assert result["baseline_drop"]["crop_accuracy"] == pytest.approx(0.0)


def test_retention_gain_is_none_without_a_pre_disruption_round():
    rounds = [_round(0, mesh=0.9, baseline=0.8)]
    assert metrics.retention_gain(rounds, "node_0", disruption_start_round=0)["retention_gain"] is None


def test_energy_totals_charge_collaboration_only_for_what_it_adds():
    rounds = [
        _round(0, 0.9, 0.8, base_kwh=0.001, mesh_kwh=0.003, comm_j=50.0),
        _round(1, 0.9, 0.8, base_kwh=0.001, mesh_kwh=0.003, comm_j=50.0),
    ]
    totals = metrics.energy_totals(rounds)

    assert totals["total_baseline_compute_energy_kwh"] == pytest.approx(0.002)
    assert totals["total_mesh_compute_energy_kwh"] == pytest.approx(0.006)
    assert totals["total_communication_energy_j"] == pytest.approx(100.0)
    assert totals["total_mesh_energy_j"] == pytest.approx(0.006 * 3_600_000 + 100.0)
    # the extra energy collaboration cost over local-only: the mesh's compute
    # premium plus all of the communication, which local-only never pays.
    assert totals["additional_energy_j"] == pytest.approx(0.004 * 3_600_000 + 100.0)


def test_efficiency_metrics_divide_the_final_gain_by_each_cost():
    rounds = [
        _round(0, 0.9, 0.8, base_kwh=0.001, mesh_kwh=0.003, comm_j=50.0, bytes_=1000),
        _round(1, 0.9, 0.8, base_kwh=0.001, mesh_kwh=0.003, comm_j=50.0, bytes_=1000,
               macro_gain={"crop_accuracy": 0.12}),
    ]
    eff = metrics.efficiency_metrics(rounds)

    assert eff["final_round_macro_gain"]["crop_accuracy"] == pytest.approx(0.12)
    assert eff["gain_per_joule"]["crop_accuracy"] == pytest.approx(0.12 / (0.006 * 3_600_000 + 100.0))
    assert eff["gain_per_additional_joule"]["crop_accuracy"] == pytest.approx(0.12 / (0.004 * 3_600_000 + 100.0))
    assert eff["gain_per_byte"]["crop_accuracy"] == pytest.approx(0.12 / 2000)


def test_efficiency_metrics_are_none_when_nothing_was_measured():
    # an un-instrumented run must read as "not measured", never as a
    # measured zero, so a null can't be mistaken for a real figure.
    rounds = [_round(0, 0.9, 0.8, comm_j=0.0, bytes_=0)]
    eff = metrics.efficiency_metrics(rounds)

    assert eff["gain_per_joule"] is None
    assert eff["gain_per_additional_joule"] is None
    assert eff["gain_per_byte"] is None


def test_disruption_rounds_falls_back_to_config_then_to_the_first_event():
    explicit = _report([], disruption_start_round=3, disruption_end_round=5)
    assert summarise.disruption_rounds(explicit) == (3, 5)

    from_config = _report([], config={"disconnect_round": 2, "reconnect_round": 4})
    assert summarise.disruption_rounds(from_config) == (2, 4)

    from_event = _report(
        [_round(0, 0.9, 0.8), _round(1, 0.9, 0.8, events=[{"event_type": "shift_applied", "node_id": "node_0"}])],
        config={},
    )
    assert summarise.disruption_rounds(from_event) == (1, 1)


def test_summarise_writes_json_csv_and_markdown_for_the_reports_present(tmp_path):
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    rounds = [
        _round(0, 0.90, 0.80, base_kwh=0.001, mesh_kwh=0.003),
        _round(1, 0.92, 0.82, base_kwh=0.001, mesh_kwh=0.003,
               events=[{"event_type": "disconnect", "node_id": "node_0", "details": {}}]),
        _round(2, 0.94, 0.84, base_kwh=0.001, mesh_kwh=0.003, macro_gain={"crop_accuracy": 0.10}),
    ]
    (scenarios_dir / "disconnection.json").write_text(json.dumps(
        _report(rounds, scenario="disconnection", config={"disconnect_round": 1, "reconnect_round": 2})
    ))

    json_path = summarise.write_summary(tmp_path)
    summary = json.loads(json_path.read_text())

    assert summary["scenarios_summarised"] == ["disconnection"]
    assert sorted(summary["scenarios_missing"]) == ["class_addition", "distribution_shift"]
    assert len(summary["per_round_table"]) == 3
    row = summary["cross_scenario_table"][0]
    assert row["scenario"] == "disconnection"
    assert row["final_macro_gain_crop_accuracy"] == pytest.approx(0.10)
    assert row["adaptation_gain_crop_accuracy"] == pytest.approx(0.10)

    for name in ("summary.md", "summary.csv", "summary_per_round.csv"):
        assert (scenarios_dir / name).read_text(encoding="utf-8").strip()


def test_summarise_raises_a_pointed_error_when_no_reports_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_all"):
        summarise.write_summary(tmp_path)
