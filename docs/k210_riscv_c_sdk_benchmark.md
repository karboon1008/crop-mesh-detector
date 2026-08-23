# Measuring real compute energy on a RISC-V K210 (Yahboom AI-Motion kit), C SDK path

Goal: get a **real, disclosed hardware measurement** of per-inference compute
energy on the one physical RISC-V device we have, using the bare-metal C SDK
(not MaixPy), then feed that measured figure into the mesh simulation's
sustainability report as the compute-energy source for every simulated node
(justified because all simulated nodes run the same trained architecture —
see the "Disclosure" section at the end). This is what the grading rubric's
E.3 table calls "High" credibility: *"Measures power on a real / simulated
device or via system tools, disclosing the measurement conditions and the
voltage/frequency settings."* One board is enough for that — it's about
method and disclosure, not device count.

Board: Yahboom AI-Motion K210 Developer Kit. Chip: Kendryte K210 — dual-core
RV64GC @ default 400 MHz, with a dedicated KPU (CNN accelerator, not a
general CPU-bound inference path). No OS — this doc uses the bare-metal
`kendryte-standalone-sdk` (C), not MaixPy/MicroPython.

**Hard constraint to check before anything else**: the KPU only accelerates
classic conv-net layers (1×1 / 3×3 kernels, ≤ ~5.9 MiB of parameters/working
set for the real-time path). Of this repo's three architectures
(`config.yaml` → `models.architectures`), `mobilenet_v3_small` and
`efficientnet_lite0` are plain CNNs and are the realistic targets.
`mobilevit_xxs` uses transformer/attention blocks that the K210's nncase
backend cannot map to the KPU — **don't spend conversion time on it**; pick
`mobilenet_v3_small` first.

---

## 1. Model conversion walkthrough: PyTorch → ONNX → `.kmodel`

The K210's KPU doesn't run ONNX or PyTorch directly. It runs a `.kmodel` —
a format produced by Kendryte's **nncase** compiler, which also does the
int8 quantization.

### 1.1 Export ONNX from the trained checkpoint

Reuse the existing export path rather than writing a new one —
`scripts/export_for_pi.py` already does PyTorch → ONNX (+ a parity check
against real images) for this repo's models:

```bash
python scripts/export_for_pi.py --arch mobilenet_v3_small --node node_0
```

This gives you `outputs/pi_export/mobilenet_v3_small/model.onnx` and a
`manifest.json` with the exact image size and normalization constants —
you'll need both again in step 3 (input buffers must match this exactly, or
the accuracy — and therefore the "same model" claim behind the energy
number — doesn't hold).

### 1.2 Install nncase — version gotcha

**Use a legacy nncase release, not the current one.** Current nncase
versions target the newer K230 (which runs Linux); K210 (`-t k210`) support
only exists in the old `v0.1.x`/`v0.2.x` line of the compiler. Get the `ncc`
binary from a `v0.1.0-rc5`/`v0.2.0`-era release of
[kendryte/nncase](https://github.com/kendryte/nncase) (or a community Linux
build of it, e.g. `kvant6ubl/nncase`). Confirm with `./ncc --help` that
`k210` is listed as a valid `-t` target before going further.

### 1.3 Build a calibration set

nncase does post-training quantization at compile time and needs
representative images to calibrate int8 ranges. Reuse a small slice
(20–50 images) of the same data used for the export parity check, saved as
plain image files in one folder, e.g. `calib_images/`.

### 1.4 Compile

```bash
./ncc compile outputs/pi_export/mobilenet_v3_small/model.onnx \
    model.kmodel \
    -i onnx -t k210 \
    --dataset calib_images \
    --input-layout NHWC \
    --input-mean 0.485,0.456,0.406 \
    --input-std 0.229,0.224,0.225
```

Use the *exact* mean/std from `manifest.json` (`IMAGENET_MEAN`/`IMAGENET_STD`
in `src/data/plantvillage.py` — standard ImageNet stats, shown above), not
defaults — this is the same normalization the model was trained with, and
it's one of the "estimation assumptions" you'll want to state verbatim in
the disclosure block later.

`--input-layout NHWC` matters beyond just matching the ONNX graph's native
NCHW: it tells nncase to accept plain interleaved-RGB pixel buffers (the
natural image-sensor/KPU format), so the raw `.bin` files from Step 2 are
just resized RGB bytes with no channel transpose needed. If you compile
without this flag, nncase keeps the ONNX-default NCHW layout and Step 2's
conversion script needs `--layout chw` instead.

**Record the compiler's console output** — it reports the chosen input
layout and quantization scale/zero-point. Save this log; it's your
"quantisation precision" disclosure line for E.3's Medium/High bar.

### 1.5 Check the output size

`model.kmodel` must comfortably fit the KPU's real-time working-set ceiling
(~5.9 MiB). `mobilenet_v3_small` at `image_size: 160` (per `config.yaml`)
should be well under this; if you sized up the input resolution, re-check.

---

## Step 1 — Toolchain + SDK project skeleton

```bash
git clone https://github.com/kendryte/kendryte-standalone-sdk
git clone https://github.com/kendryte/kendryte-gnu-toolchain   # or a prebuilt riscv64-unknown-elf-gcc
```

Yahboom also publishes their own fork/examples for this exact kit —
[YahboomTechnology/K210-Developer-Kit](https://github.com/YahboomTechnology/K210-Developer-Kit)
— worth checking for board-specific pin mappings and known-good CMake
settings before assuming vanilla `kendryte-standalone-sdk` defaults match
your dock.

Project layout (mirrors every other demo in the SDK):

```
kendryte-standalone-sdk/
  src/
    energy_bench/          # <- new, your project
      main.c
      CMakeLists.txt
```

Minimal `src/energy_bench/CMakeLists.txt`:

```cmake
set(SOURCE_FILES main.c)
```

Build from the SDK root:

```bash
mkdir build && cd build
cmake .. -DPROJ=energy_bench -DTOOLCHAIN=/path/to/riscv-toolchain/bin
make
```

Produces `build/energy_bench.bin` (and a debug `.elf`).

---

## Step 2 — Input buffers: no camera needed for the benchmark

Live camera capture adds sensor/ISP power and frame-timing jitter on top of
the number you actually want (compute energy per inference), so don't use
it for the measurement run. Instead, pre-bake a handful of already-quantized
input tensors into the firmware image itself:

1. On your dev machine, run `scripts/k210_prepare_inputs.py` — it picks a
   handful of real PlantVillage images, resizes each to the trained
   `image_size` (read straight from `manifest.json`, so it can't drift out
   of sync with the model), and writes each as a raw `uint8` HWC pixel
   buffer (`input_0.bin`, `input_1.bin`, …) plus a generated
   `inputs_incbin.h` with the `INCBIN` declarations already written out:

   ```bash
   python scripts/k210_prepare_inputs.py \
       --manifest outputs/pi_export/mobilenet_v3_small/manifest.json \
       --num-images 20 --layout hwc --output-dir k210_inputs
   ```

   **Important:** these buffers are plain resized pixels (0–255), *not*
   normalized floats — the `--input-mean`/`--input-std` you passed to `ncc
   compile` in Step "1.4 Compile" already baked that normalization into the
   `.kmodel` itself. Feeding already-normalized data here would double-apply
   it. Keep `--layout hwc` matched to `--input-layout NHWC` at compile time
   (the default in this doc); switch both to `chw`/omit the flag together
   if you compiled without it.
2. In firmware, embed the compiled model and each input buffer directly into
   the binary with `INCBIN` (already used throughout the SDK's own demos) —
   copy the whole `k210_inputs/` folder next to `main.c` and `#include
   "inputs_incbin.h"` instead of hand-typing the block below:

```c
#include "incbin.h"

INCBIN(model, "model.kmodel");
INCBIN(input0, "input_0.bin");
INCBIN(input1, "input_1.bin");
/* ... one INCBIN per calibration image you want to cycle through ... */
```

No SD card or filesystem needed — everything ships inside the flashed
firmware, and there's no camera-driver power draw mixed into your timing.

(A live camera+LCD demo is still worth wiring up separately for judges to
*watch* the board classify something in real time — just don't use that run
for the numbers you cite in the sustainability report.)

---

## Step 3 — Benchmark firmware (C)

Uses the nncase-model KPU API (`kpu_model_context_t` + friends), a fixed
CPU frequency you record and print, and `sysctl_get_time_us()` for timing —
all standard `kendryte-standalone-sdk` calls:

```c
#include <stdio.h>
#include "kpu.h"
#include "sysctl.h"
#include "plic.h"
#include "incbin.h"

INCBIN(model, "model.kmodel");
#include "inputs_incbin.h"   /* generated by scripts/k210_prepare_inputs.py:
                                 declares input0_data..inputN_data plus
                                 k210_bench_inputs[]/k210_bench_num_inputs */

#define NUM_ITERATIONS 100

static kpu_model_context_t ctx;
static volatile int g_done = 0;

static void ai_done_cb(void *userdata) {
    g_done = 1;
}

int main(void) {
    /* 1. Fix and disclose the clock — don't let it float mid-measurement */
    uint32_t actual_freq = sysctl_cpu_set_freq(400000000);   /* 400 MHz */
    printf("cpu_freq_hz=%u\n", actual_freq);
    printf("supply_rail=board_5V_input (metered externally, not silicon-level)\n");

    if (kpu_load_kmodel(&ctx, model_data) != 0) {
        printf("kpu_load_kmodel failed\n");
        return -1;
    }

    uint64_t durations_us[NUM_ITERATIONS];

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        const uint8_t *src = k210_bench_inputs[i % k210_bench_num_inputs];
        g_done = 0;

        uint64_t t0 = sysctl_get_time_us();
        kpu_run_kmodel(&ctx, src, DMAC_CHANNEL5, ai_done_cb, NULL);
        while (!g_done) { /* busy-wait for the completion callback */ }
        uint64_t t1 = sysctl_get_time_us();

        durations_us[i] = t1 - t0;
        printf("iter=%d duration_us=%llu\n", i, durations_us[i]);
    }

    /* mean/std over the run — printed once at the end for the record */
    uint64_t sum = 0;
    for (int i = 0; i < NUM_ITERATIONS; i++) sum += durations_us[i];
    double mean = (double)sum / NUM_ITERATIONS;
    double var = 0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        double d = durations_us[i] - mean;
        var += d * d;
    }
    var /= NUM_ITERATIONS;
    printf("mean_us=%.2f std_us=%.2f n=%d\n", mean, sqrt(var), NUM_ITERATIONS);

    kpu_model_free(&ctx);
    return 0;
}
```

Notes:
- `kpu_run_kmodel` is asynchronous — the busy-wait-on-flag pattern above is
  the standard simple synchronization used across the SDK's own demos; a
  real interrupt-driven wait is possible via PLIC but isn't necessary for
  benchmarking accuracy since the flag write happens right as the KPU
  finishes.
- Sanity-check correctness once outside the timing loop with
  `kpu_get_output(&ctx, 0, &data, &size)` and compare against the PyTorch
  prediction for the same image (mirrors the parity check
  `scripts/export_for_pi.py` already does for the Pi path) — you want to know
  the on-device model is actually predicting correctly, not just running fast.

Build exactly as in Step 1 (`cmake -DPROJ=energy_bench ... && make`).

---

## Step 4 — Power measurement while it runs

The K210 has no onboard power-monitor IC exposed to software, so this has
to be external and board-level (disclose this explicitly — it's honest, and
still satisfies the rubric's "disclosing measurement conditions"):

1. Put a USB power meter (or an inline INA219/INA226 breakout) on the 5V
   supply feeding the board, sampling continuously.
2. Flash the firmware (Step "Flashing" below) and let it run the full
   `NUM_ITERATIONS` loop while the meter logs.
3. Capture the UART log (per-iteration `duration_us` lines) alongside the
   power trace's timestamps.
4. Compute: `mean_power_W = mean(logged current) * 5V`;
   `energy_per_inference_J = mean_power_W * (mean_duration_us / 1e6)`.
   Report both the timing mean/std (from firmware) and the power-meter
   mean/std (from the external log) — two independent uncertainty sources,
   both worth stating.

---

## Flashing and capturing output

```bash
git clone https://github.com/kendryte/kflash.py
sudo cp kflash.py/kflash.py /usr/bin/kflash && sudo chmod +x /usr/bin/kflash

kflash -p /dev/ttyUSB0 -b 1500000 -B goE -s -t build/energy_bench.bin
```

`-B` is the board-reset-circuit type (`kd233`, `dan`, `goE`, `trainer`, …) —
it varies by which dock/carrier the K210 module sits in. If `goE` doesn't
reset the board into ISP mode, check
[YahboomTechnology/K210-Developer-Kit](https://github.com/YahboomTechnology/K210-Developer-Kit)
for the value their own flashing scripts use for this exact kit.

Capture the run's UART output to a file for the record (e.g. via `screen`,
`minicom`, or `putty` at 115200 baud) — that log *is* your raw evidence for
the disclosure block below.

---

## Disclosure block — paste into the sustainability report

This is the text that turns "we measured something" into E.3's High-tier
disclosure. Fill in the blanks from your actual run:

```
Device: Kendryte K210 (Yahboom AI-Motion Developer Kit), dual-core RV64GC + KPU
CPU frequency: 400 MHz (fixed via sysctl_cpu_set_freq, logged at firmware start)
Supply rail metered: board-level 5V USB input (external power meter; not
  silicon-level — no on-chip power monitor is exposed on this SoC)
Model: mobilenet_v3_small, exported via scripts/export_for_pi.py,
  quantized to int8 via nncase <version> for the k210 target
Input: <image_size>x<image_size>, normalization mean/std <from manifest.json>
Measured: mean inference latency <X> us (std <Y> us, n=100 iterations),
  mean board power draw <P> W, mean energy per inference <E> J
Assumption: this per-inference compute-energy figure is applied uniformly
  to every simulated node's local_train/distill steps in the mesh
  simulation, scaled by each step's approximate FLOPs multiple relative to
  a forward pass (justified because all simulated nodes run this same
  trained architecture) — training/distillation energy on this figure is
  therefore an *estimate* extrapolated from a *measured* inference number,
  not a direct on-device training measurement, because the K210's KPU is
  inference-only and cannot run backpropagation.
Communication energy: separately estimated (not measured) via
  CommunicationCostEstimator's published per-byte radio figures — no radio
  hardware in the simulation.
```

---

## Known limits worth stating up front (not hiding)

- **KPU is inference-only.** There's no on-device backward pass, so
  `local_train`/`distill` energy per round can't be measured directly on
  this board — only inference can. State the FLOPs-ratio extrapolation
  method explicitly (see disclosure block) rather than implying it was
  measured.
- **Board-level, not silicon-level, power.** The meter sees the whole
  board's 5V draw, not an isolated KPU/CPU rail. Say so.
- **Architecture constraint.** Only `mobilenet_v3_small` /
  `efficientnet_lite0` are realistic KPU targets; `mobilevit_xxs`'s
  attention blocks aren't representable on this chip's nncase backend.
- **nncase version pinning.** K210 support only exists in the old
  `v0.1.x`/`v0.2.x` nncase line — using a current release will simply fail
  to offer `-t k210` at all.
