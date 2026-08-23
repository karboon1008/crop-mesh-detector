"""Sustainability accounting: measured compute energy (CodeCarbon) plus
an estimated communication energy budget for the prototype/logit
exchange, both converted to CO2e with a stated grid-carbon-intensity
factor. Mirrors the three-part inventory (compute, memory access,
communication) the report's methodology calls for — communication is
estimated rather than measured because there is no real radio in a
pure-simulation run.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_CODECARBON_MEASURED_CPU_MARKERS = ("rapl", "power gadget", "powermetrics")


class ComputeEnergyTracker:
    """Wraps CodeCarbon around a labelled block of code. Falls back to a
    simple wall-clock * assumed-average-power proxy estimate when
    CodeCarbon is unavailable or disabled, so the pipeline always
    produces *some* energy figure — clearly tagged with which method
    produced it.
    """

    def __init__(
        self,
        enabled: bool = True,
        output_dir: str | Path = "outputs",
        country_iso_code: str = "GBR",
        fallback_power_watts: float = 15.0,
    ):
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.country_iso_code = country_iso_code
        self.fallback_power_watts = fallback_power_watts
        self.log: list[dict] = []

        self._codecarbon_available = False
        if enabled:
            try:
                import codecarbon  # noqa: F401

                # CodeCarbon's EmissionsTracker does not take a country
                # code as a constructor argument; it otherwise falls back
                # to IP-based geolocation to pick a grid factor for its
                # OWN co2_kg figure. This env var is CodeCarbon's
                # documented override for that. We don't actually rely on
                # its co2_kg downstream, though — write_sustainability_report()
                # takes only the measured, location-independent
                # energy_consumed (kWh) and applies config.yaml's own
                # stated grid_carbon_intensity_gco2_per_kwh, so the
                # reported carbon figure stays reproducible regardless of
                # where this is actually run.
                import os

                os.environ["CODECARBON_COUNTRY_ISO_CODE"] = country_iso_code
                self._codecarbon_available = True
            except Exception:
                self._codecarbon_available = False

    @contextmanager
    def track(self, label: str):
        """Usage:
            with tracker.track("node_0_local_train") as record:
                ... do work ...
            record now holds duration_s, energy_kwh, co2_kg (may be None), method.
        """
        start = time.time()
        record: dict = {"label": label}
        codecarbon_tracker = None

        if self._codecarbon_available:
            try:
                from codecarbon import EmissionsTracker

                codecarbon_tracker = EmissionsTracker(
                    project_name=label,
                    output_dir=str(self.output_dir),
                    output_file="emissions.csv",
                    log_level="error",
                    save_to_file=True,
                )
                codecarbon_tracker.start()
                record["method"] = "codecarbon"
                hardware = codecarbon_tracker._conf.get("hardware", [])
                record["hardware_sources"] = hardware
                record["cpu_measured"] = any(
                    marker in desc.lower()
                    for desc in hardware
                    if desc.upper().startswith("CPU")
                    for marker in _CODECARBON_MEASURED_CPU_MARKERS
                )
                if not record["cpu_measured"]:
                    print(
                        f"[energy] WARNING: no RAPL/Power-Gadget/powermetrics access detected "
                        f"for '{label}' — CPU energy is a constant-TDP estimate, not a measurement "
                        f"(hardware: {hardware}).",
                        file=sys.stderr,
                    )
            except Exception:
                self._codecarbon_available = False
                codecarbon_tracker = None

        if codecarbon_tracker is None:
            record["method"] = "proxy_wall_power"
            record["cpu_measured"] = False

        try:
            yield record
        finally:
            duration = time.time() - start
            record["duration_s"] = duration
            if codecarbon_tracker is not None:
                co2_kg = codecarbon_tracker.stop() or 0.0
                energy_kwh = 0.0
                final_data = getattr(codecarbon_tracker, "final_emissions_data", None)
                if final_data is not None:
                    energy_kwh = getattr(final_data, "energy_consumed", 0.0) or 0.0
                record["energy_kwh"] = energy_kwh
                record["co2_kg"] = co2_kg
            else:
                energy_kwh = (self.fallback_power_watts * duration) / 3600 / 1000
                record["energy_kwh"] = energy_kwh
                record["co2_kg"] = None  # filled in later once a grid factor is applied
            self.log.append(record)

    def summary(self) -> dict:
        total_energy = sum(r.get("energy_kwh", 0.0) for r in self.log)
        total_duration = sum(r.get("duration_s", 0.0) for r in self.log)
        return {
            "num_tracked_blocks": len(self.log),
            "total_compute_energy_kwh": total_energy,
            "total_duration_s": total_duration,
            "all_blocks_cpu_measured": bool(self.log) and all(r.get("cpu_measured") for r in self.log),
        }


def sweep_totals_from_emissions_csv(output_dir: str | Path) -> dict | None:
    """Summing total to give the real sweep cost regardless of how the
    run was split across processes/jobs.
    """
    path = Path(output_dir) / "emissions.csv"
    if not path.exists():
        return None
    total_duration_s = 0.0
    total_energy_kwh = 0.0
    total_co2_kg = 0.0
    num_rows = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            total_duration_s += float(row["duration"])
            total_energy_kwh += float(row["energy_consumed"])
            total_co2_kg += float(row["emissions"])
            num_rows += 1
    return {
        "num_rows": num_rows,
        "total_duration_s": total_duration_s,
        "total_compute_energy_kwh": total_energy_kwh,
        "total_co2_kg": total_co2_kg,
    }


class CommunicationCostEstimator:
    """Converts a byte count (from KnowledgePayload.size_bytes()) into an
    estimated transmit energy and CO2e, using published per-byte radio
    energy figures. This is intentionally an *estimate*, not a
    measurement — there is no physical radio in a simulation — but it
    is what lets the report's "communication energy must stay below
    compute savings" claim be checked quantitatively (§2.6 of the
    accompanying report).
    """

    def __init__(self, radio_energy_j_per_byte: dict[str, float], grid_carbon_intensity_gco2_per_kwh: float):
        self.radio_energy_j_per_byte = radio_energy_j_per_byte
        self.grid_carbon_intensity_gco2_per_kwh = grid_carbon_intensity_gco2_per_kwh

    def estimate(self, total_bytes: int, radio: str) -> dict:
        joules = total_bytes * self.radio_energy_j_per_byte[radio]
        kwh = joules / 3_600_000
        co2_kg = kwh * self.grid_carbon_intensity_gco2_per_kwh / 1000
        return {"radio": radio, "bytes": total_bytes, "energy_kwh": kwh, "co2_kg": co2_kg}

    def estimate_all_radios(self, total_bytes: int) -> dict[str, dict]:
        return {radio: self.estimate(total_bytes, radio) for radio in self.radio_energy_j_per_byte}


def grid_co2_kg(energy_kwh: float, grid_carbon_intensity_gco2_per_kwh: float) -> float:
    return energy_kwh * grid_carbon_intensity_gco2_per_kwh / 1000


def write_sustainability_report(
    output_path: str | Path,
    compute_summary: dict,
    communication_estimate: dict,
    collaboration_gain: dict,
    grid_carbon_intensity_gco2_per_kwh: float,
    emissions_csv_totals: dict | None = None,
) -> None:
    """Writes both a machine-readable JSON and a short human-readable
    markdown narrative summarising the energy/carbon picture and whether
    the collaboration gain was "worth" the communication overhead.
    """
    output_path = Path(output_path)
    total_compute_kwh = compute_summary.get("total_compute_energy_kwh", 0.0)
    compute_co2_kg = grid_co2_kg(total_compute_kwh, grid_carbon_intensity_gco2_per_kwh)

    payload = {
        "compute": {**compute_summary, "co2_kg": compute_co2_kg},
        "compute_from_emissions_csv": emissions_csv_totals,
        "communication": communication_estimate,
        "collaboration_gain": collaboration_gain,
        "grid_carbon_intensity_gco2_per_kwh": grid_carbon_intensity_gco2_per_kwh,
    }
    output_path.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str))

    wifi = communication_estimate.get("wifi", {})
    comm_kwh = wifi.get("energy_kwh", 0.0)
    comm_co2 = wifi.get("co2_kg", 0.0)
    total_kwh = total_compute_kwh + comm_kwh
    comm_share_pct = (comm_kwh / total_kwh * 100) if total_kwh > 0 else 0.0

    if compute_summary.get("num_tracked_blocks", 0) == 0:
        measurement_note = "no compute blocks were tracked"
    elif compute_summary.get("all_blocks_cpu_measured"):
        measurement_note = "hardware-measured (RAPL/Power Gadget/powermetrics) for every tracked block"
    else:
        measurement_note = (
            "**at least one block used a constant-TDP estimate, not a hardware measurement** "
            "— see per-block `hardware_sources` in the JSON report"
        )

    lines = [
        "# Sustainability report",
        "",
        f"- Compute energy measurement basis: {measurement_note}",
        f"- Measured/estimated **compute energy**: {total_compute_kwh:.6f} kWh "
        f"({compute_co2_kg * 1000:.3f} g CO2e)",
        f"- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): "
        f"{comm_kwh:.8f} kWh ({comm_co2 * 1000:.5f} g CO2e)",
        f"- Communication is **{comm_share_pct:.2f}%** of total energy — "
        f"{'well within' if comm_share_pct < 10 else 'a significant share of'} "
        f"the 'communication should not erase compute savings' target.",
    ]

    if emissions_csv_totals is not None:
        csv_kwh = emissions_csv_totals["total_compute_energy_kwh"]
        csv_co2_kg = emissions_csv_totals["total_co2_kg"]
        csv_rows = emissions_csv_totals["num_rows"]
        discrepancy_pct = (
            abs(csv_kwh - total_compute_kwh) / csv_kwh * 100 if csv_kwh > 0 else 0.0
        )
        lines.append(
            f"- **Real sweep total from emissions.csv** ({csv_rows} tracked blocks, all "
            f"invocations): {csv_kwh:.6f} kWh ({csv_co2_kg * 1000:.3f} g CO2e), "
            f"{emissions_csv_totals['total_duration_s'] / 60:.1f} min."
        )
        if discrepancy_pct > 1.0:
            lines.append(
                f"  - **WARNING**: this is {discrepancy_pct:.1f}% different from the "
                f"`total_compute_energy_kwh` figure above ({compute_summary.get('num_tracked_blocks', 0)} "
                f"blocks) — run_state.json's cross-invocation accumulator likely missed one or more "
                f"earlier invocations (e.g. a separate job per --arch, or a run with --fresh). "
                f"Treat the emissions.csv-derived total as the authoritative one."
            )

    lines += [
        "",
        "## Collaboration gain vs. energy spent",
        "",
        json.dumps(collaboration_gain, indent=2),
        "",
        f"Grid carbon intensity used: {grid_carbon_intensity_gco2_per_kwh} gCO2/kWh.",
    ]
    output_path.with_suffix(".md").write_text("\n".join(lines))
