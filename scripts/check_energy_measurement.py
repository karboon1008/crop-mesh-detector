#!/usr/bin/env python3
"""Pre-flight check for src/energy/tracker.py's ComputeEnergyTracker on a
new machine (e.g. an allocated Isambard compute node) -- run this INSIDE
the actual job allocation (salloc/srun/sbatch), not on the login node,
since power counters are per-node and login nodes often expose different
(or no) hardware access than the compute nodes a job actually lands on.

CodeCarbon silently degrades to a constant-TDP-times-load estimate when it
can't reach real hardware counters (RAPL for CPU, NVML for GPU) -- the
result is still tagged "codecarbon" in the pipeline's own output, so this
is the only way to tell, before a real run, whether the numbers you'll get
are measured or guessed.

Run:
    python scripts/check_energy_measurement.py
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from codecarbon import EmissionsTracker
    except Exception as exc:
        print(f"codecarbon is not importable: {exc}")
        print("-> energy tracking will use the proxy_wall_power fallback only.")
        return 1

    tracker = EmissionsTracker(
        project_name="energy_preflight_check",
        log_level="error",
        save_to_file=False,
    )
    tracker.start()
    tracker.stop()

    hardware = tracker._conf.get("hardware", [])
    print("Hardware CodeCarbon detected on this node:")
    for desc in hardware:
        print(f"  - {desc}")

    cpu_desc = next((d for d in hardware if d.upper().startswith("CPU")), None)
    gpu_desc = next((d for d in hardware if d.upper().startswith("GPU")), None)

    measured_markers = ("rapl", "power gadget", "powermetrics")
    cpu_measured = bool(cpu_desc) and any(m in cpu_desc.lower() for m in measured_markers)

    print()
    if cpu_measured:
        print(f"CPU: hardware-measured ({cpu_desc}).")
    elif cpu_desc:
        print(
            f"CPU: NOT hardware-measured -- '{cpu_desc}' is a constant-TDP estimate "
            "(no RAPL access on this node/allocation)."
        )
    else:
        print("CPU: no CPU hardware entry detected at all.")

    gpu_missing_despite_cuda = False
    if gpu_desc:
        print(f"GPU: detected via NVML ({gpu_desc}).")
    else:
        try:
            import torch

            cuda_visible = torch.cuda.is_available()
        except Exception:
            cuda_visible = None
        if cuda_visible:
            gpu_missing_despite_cuda = True
            print(
                "GPU: torch.cuda.is_available() is True but CodeCarbon found no GPU -- "
                "NVML is likely blocked in this allocation; GPU energy will be missing "
                "from the report entirely, not just estimated."
            )
        else:
            print("GPU: none detected (expected on a CPU-only allocation).")

    if not cpu_measured or gpu_missing_despite_cuda:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
