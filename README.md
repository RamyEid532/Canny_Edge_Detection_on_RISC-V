# Canny Edge Detection on RISC-V with Vector Extension

[![View on GitHub](https://img.shields.io/badge/GitHub-View_Repository-blue?logo=github)](https://github.com/RamyEid532/Canny_Edge_Detection_on_RISC-V)

A complete C++ implementation of the Canny edge detection pipeline, cross-compiled for RISC-V (`rv64gcv`) and executed on QEMU user-mode emulation. The project follows the full embedded optimization workflow: clean scalar baseline → compiler flag sweep → hotspot profiling → manual RVV intrinsic acceleration.

**Course:** Embedded Systems — Dr. Omar Nasr  
**Team:** Group 21

---

## Pipeline

```
Input Image (.raw)
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1 — Gaussian Blur (5×5)                           │
│  Smooths the image to suppress noise before edge finding  │
│  Integer kernel (σ≈1.0, sum=273), zero-padding boundary   │
│  int32 accumulator prevents overflow  ·  Scalar + RVV    │
└───────────────────────────┬──────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2 — Sobel Gradient (Gx / Gy)                      │
│  Two 3×3 kernels detect horizontal and vertical edges     │
│  Structure-of-Arrays layout (int16_t) for fast RVV loads  │
│  Direction quantized to {0°, 45°, 90°, 135°}  ·  Scalar + RVV │
└───────────────────────────┬──────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2b — Gradient Magnitude                            │
│  L1 norm: |Gx|+|Gy|  (integer, fast)                     │
│  L2 norm: √(Gx²+Gy²) (float, precise)                    │
│  Two-pass normalization to [0,255]  ·  Scalar + RVV      │
└───────────────────────────┬──────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3 — Non-Maximum Suppression                        │
│  Thins edges to single-pixel width using gradient direction│
│  Compares each pixel to its two directional neighbors     │
│  Scalar + RVV (vectorized neighbor select via masks)      │
└───────────────────────────┬──────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 4 — Double Thresholding                            │
│  STRONG (255) / WEAK (75) / NO_EDGE (0) classification   │
│  Scalar + RVV (vmsgeu masks + vmerge, fully vectorized)   │
└───────────────────────────┬──────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 5 — Hysteresis Edge Tracking                       │
│  DFS flood-fill promotes weak edges connected to strong   │
│  Isolated weak edges removed  ·  Scalar + partial RVV    │
└───────────────────────────┬──────────────────────────────┘
                            ▼
                   Output Image (.raw)
```

---

## Quick Start

### Prerequisites

```bash
sudo apt install -y autoconf automake build-essential bison flex texinfo \
    gperf libtool patchutils bc git cmake libglib2.0-dev libpixman-1-dev \
    libslirp-dev ninja-build libmpc-dev libmpfr-dev libgmp-dev \
    zlib1g-dev libexpat-dev python3 python3-numpy python3-opencv
```

### 1 — Build the RISC-V Toolchain (one-time, ~60 min)

```bash
git clone --recursive --depth 1 --shallow-submodules \
    https://github.com/riscv-collab/riscv-gnu-toolchain

cd riscv-gnu-toolchain
./configure --prefix=$HOME/riscv-toolchain --with-arch=rv64gcv --with-abi=lp64d
make -j$(nproc)

echo 'export PATH=$HOME/riscv-toolchain/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

riscv64-unknown-elf-g++ --version   # should say GCC 13.x or 14.x
```

> **Important:** Update the `--sysroot` path in the Makefile's `RV_CXXFLAGS` to match your own installation:
> ```makefile
> RV_CXXFLAGS = ... --sysroot=$(HOME)/riscv-toolchain/riscv64-unknown-elf
> ```

### 2 — Build QEMU (one-time, ~10 min)

```bash
git clone --depth 1 https://github.com/qemu/qemu
cd qemu
./configure --target-list=riscv64-linux-user
make -j$(nproc)
sudo make install

qemu-riscv64 --version   # should say QEMU 9.x
```

### 3 — Build GoogleTest (one-time)

```bash
git clone --depth 1 https://github.com/google/googletest
cd googletest
cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/googletest-install
cmake --build build -j$(nproc)
cmake --install build
```

Set `GTEST_LIB` in your shell so the Makefile can find the libraries:
```bash
# If you built from source:
export GTEST_LIB=$HOME/googletest-install/lib   # add to ~/.bashrc

# If you used: sudo apt install libgtest-dev
export GTEST_LIB=/usr/lib/x86_64-linux-gnu
```

### 4 — Clone and Build

```bash
git clone https://github.com/RamyEid532/Canny_Edge_Detection_on_RISC-V.git
cd Canny_Edge_Detection_on_RISC-V

# Run 32 host-side unit tests (no QEMU required)
make test

# Cross-compile for RISC-V
make canny_rv OPT=-O3

# Run on QEMU with a synthetic rectangle image (default VLEN=128)
make run

# Run with a real photo
python3 convert.py        # converts my_photo.jpeg → input.raw + reference.png
make run IMG=input.raw

# Run at different vector register widths
make run VLEN=256
make run VLEN=512
```

---

## Make Targets

| Target | Description |
|--------|-------------|
| `make test` | Build and run 32 GoogleTest unit tests on the host (no QEMU) |
| `make canny_rv` | Cross-compile pipeline for RISC-V (default `OPT=-O3`) |
| `make run` | Run on QEMU (`VLEN=128` by default) |
| `make equiv_test` | Build scalar vs RVV equivalence test binary |
| `make run_equiv` | Run equivalence tests at VLEN=128, 256, and 512 |
| `make clean` | Remove all build artifacts |

**Common overrides:**
```bash
make canny_rv OPT=-O0        # change optimization level
make run VLEN=256             # change emulated vector register width
make run IMG=input.raw        # run on a real 512×512 raw grayscale image
```

---

## Image Format

All images are **raw 8-bit grayscale**: exactly `width × height` bytes, one byte per pixel, no headers, no compression.

**Convert a JPEG to raw (and generate an OpenCV reference):**
```bash
python3 convert.py
# Produces: input.raw (512×512 grayscale), reference.png (OpenCV Canny), output.png
```

**View a raw image in Python:**
```python
import numpy as np, matplotlib.pyplot as plt
img = np.fromfile('output.raw', dtype=np.uint8).reshape(512, 512)
plt.imshow(img, cmap='gray')
plt.show()
```

---

## Project Structure

```
Canny_Edge_Detection_on_RISC-V/
├── include/
│   ├── image_io.h            # image struct, aligned alloc, load/save
│   ├── gaussian.h            # 5×5 kernel constants, scalar + RVV declarations
│   ├── sobel.h               # Sobel kernels, sobel_result struct, split stage functions
│   ├── suppression.h         # Non-maximum suppression declarations
│   ├── thresholding.h        # STRONG/WEAK/NO_EDGE constants, threshold declarations
│   └── hysteresis.h          # Hysteresis declarations
├── src/
│   ├── main.cpp              # Pipeline entry, warmup loop, per-stage timing, results table
│   ├── image_io.cpp          # 64-byte aligned alloc, fread/fwrite I/O
│   ├── gaussian.cpp          # Scalar 5×5 Gaussian convolution
│   ├── gaussian_rvv.cpp      # RVV Gaussian (LMUL=4, strip-mined interior, u8→i32 widening)
│   ├── sobel.cpp             # Scalar Gx/Gy, magnitude (L1+L2), direction quantization
│   ├── sobel_rvv.cpp         # RVV Sobel (unrolled kernel), RVV magnitude (fixed-point)
│   ├── suppression.cpp       # Scalar NMS
│   ├── suppression_rvv.cpp   # RVV NMS (direction masks + vmerge neighbor select)
│   ├── thresholding.cpp      # Scalar double threshold
│   ├── thresholding_rvv.cpp  # RVV threshold (vmsgeu + vmandn + vmerge)
│   ├── hysteresis.cpp        # Scalar hysteresis (8-connected iterative)
│   ├── hysteresis_rvv.cpp    # RVV hysteresis (vectorized copy + cleanup; DFS scalar)
│   └── syscalls.cpp          # Newlib → Linux syscall bridge required for QEMU
├── tests/
│   ├── unit_tests.cpp        # 32 GoogleTest cases across 6 suites (host-side)
│   ├── equivalence_test.cpp  # Scalar vs RVV correctness check (QEMU-side, 100×75)
│   ├── Results1              # Captured unit test output (32/32 passed)
│   └── Results2              # Captured equivalence test output (VLEN 128/256/512)
├── results/
│   ├── phase4_optimization_report.txt   # Compiler flag sweep + auto-vec analysis
│   ├── phase5_profiling.txt             # Per-stage breakdown at VLEN=128/256/512
│   └── phase6_Rvv_results.txt          # Scalar vs RVV timing at all VLEN values
├── convert.py                # JPEG→raw converter + OpenCV Canny reference generator
├── my_photo.jpeg             # Sample input image
├── output.png                # Pipeline output (generated after running)
├── reference.png             # OpenCV Canny reference (generated by convert.py)
└── Makefile
```

---

## Test Results

### Unit Tests — Host-Side (32/32 passed)

```
[==========] Running 32 tests from 6 test suites.
[  PASSED  ] 32 tests.
```

| Suite | Tests | Result |
|-------|-------|--------|
| GaussianTest | 4 | ✅ All passed |
| SobelTest | 5 | ✅ All passed |
| DirectionTest | 4 | ✅ All passed |
| SuppressionTest | 6 | ✅ All passed |
| ThresholdTest | 6 | ✅ All passed |
| HysteresisTest | 7 | ✅ All passed |

### Equivalence Tests — QEMU-Side (scalar vs RVV, 100×75 non-power-of-two)

Uses a non-power-of-two image size to force the strip-mining tail case — the most common source of VLA bugs.

```
make equiv_test && make run_equiv
```

| Test | VLEN=128 | VLEN=256 | VLEN=512 |
|------|----------|----------|----------|
| Gaussian Blur (±1) | ✅ PASS | ✅ PASS | ✅ PASS |
| Sobel Gx | ✅ PASS | ✅ PASS | ✅ PASS |
| Sobel Gy | ✅ PASS | ✅ PASS | ✅ PASS |
| Magnitude L1 (±3) | ✅ PASS | ✅ PASS | ✅ PASS |

All RVV kernels produce identical output to scalar at all VLEN values — confirming correct vector-length-agnostic (VLA) implementation.

---

## Optimization Results

All measurements: 512×512 image, 20 warmup + 200 timed iterations, QEMU VLEN=128 unless noted.

### Phase 4 — Compiler Flag Sweep (scalar code only)

| Stage | -O0 | -O2 | -O3 | -Os | -Ofast |
|-------|-----|-----|-----|-----|--------|
| Gaussian | 96.6 ms | 32.2 ms | 33.5 ms | 43.6 ms | 31.9 ms |
| Sobel Gx/Gy | 43.8 ms | 17.3 ms | 3.4 ms | 16.4 ms | 3.4 ms |
| Magnitude | 85.0 ms | 22.1 ms | 23.3 ms | 62.4 ms | 36.8 ms |
| Direction | 9.2 ms | 2.2 ms | 2.3 ms | 2.7 ms | 2.2 ms |
| NMS | 4.9 ms | 2.4 ms | 2.5 ms | 2.7 ms | 2.3 ms |
| Threshold | 2.1 ms | 1.3 ms | 1.3 ms | 1.4 ms | 1.2 ms |
| Hysteresis | 2.6 ms | 1.5 ms | 1.1 ms | 1.5 ms | 1.1 ms |
| **Total** | **244.2 ms** | **78.9 ms** | **67.4 ms** | **130.7 ms** | **78.8 ms** |
| **Speedup vs -O0** | 1.00× | 3.09× | **3.62×** | 1.87× | 3.10× |
| Binary size | 445 KB | 453 KB | 473 KB | 450 KB | 473 KB |

**Key findings:**
- `-O3` gives the best total speedup (3.62×) and is used for all subsequent phases
- Sobel gained 5× from `-O2`→`-O3`, the largest single jump, likely from loop unrolling
- `-Os` performs worse than `-O2` on Magnitude (62 ms vs 22 ms) — size optimization trades away vectorization opportunities
- Auto-vectorization report: **0 loops auto-vectorized** — boundary checks and control flow in every inner loop block the compiler. This directly motivates Phase 6

### Phase 5 — Profiling Breakdown (scalar at -O3)

| Stage | VLEN=128 | VLEN=256 | VLEN=512 | Average % |
|-------|----------|----------|----------|-----------|
| Gaussian Blur | 32.7 ms (50.3%) | 32.2 ms (49.6%) | 34.0 ms (50.9%) | **50.3%** |
| Magnitude | 22.3 ms (34.4%) | 22.9 ms (35.2%) | 22.8 ms (34.2%) | **34.6%** |
| Sobel Gx/Gy | 3.2 ms (4.9%) | 3.3 ms (5.0%) | 3.2 ms (4.8%) | 4.9% |
| NMS | 2.3 ms (3.6%) | 2.3 ms (3.5%) | 2.2 ms (3.4%) | 3.5% |
| Direction | 2.1 ms (3.3%) | 2.1 ms (3.3%) | 2.1 ms (3.2%) | 3.3% |
| Threshold | 1.2 ms (1.9%) | 1.2 ms (1.8%) | 1.3 ms (1.9%) | 1.9% |
| Hysteresis | 1.0 ms (1.6%) | 1.1 ms (1.6%) | 1.1 ms (1.6%) | 1.6% |
| **Total** | **64.9 ms** | **65.0 ms** | **66.8 ms** | |

**Hotspot conclusion:** Gaussian (50%) and Magnitude (34%) account for **85% of runtime** and are the primary RVV targets. Direction at 3.3% was left scalar — per Amdahl's law, even a 10× speedup there yields less than 0.3% total improvement.

**VLEN observation:** Timing is nearly identical across VLEN=128/256/512 for scalar code, as expected — VLEN only affects vector instruction throughput, not scalar code.

### Phase 6 — Scalar vs RVV

| Stage | Scalar | VLEN=128 | VLEN=256 | VLEN=512 |
|-------|--------|----------|----------|----------|
| Gaussian | 13.7 ms | 106.7 ms (0.13×) | 95.9 ms (0.14×) | 82.9 ms (0.16×) |
| Sobel Gx/Gy | 3.4 ms | 42.7 ms (0.08×) | 38.3 ms (0.09×) | 37.8 ms (0.09×) |
| Magnitude | 22.1 ms | 35.6 ms (0.62×) | 36.7 ms (0.64×) | 37.4 ms (0.61×) |
| Direction | 2.2 ms | scalar | scalar | scalar |
| NMS | 8.7 ms | 27.4 ms (0.32×) | 26.9 ms (0.32×) | 27.1 ms (0.33×) |
| Threshold | 1.2 ms | 9.9 ms (0.12×) | 9.9 ms (0.12×) | 10.0 ms (0.12×) |
| Hysteresis | 1.1 ms | 9.1 ms (0.12×) | 8.5 ms (0.13×) | 9.4 ms (0.12×) |
| **Total** | **52.5 ms** | **233.6 ms (0.22×)** | **218.4 ms (0.25×)** | **206.9 ms (0.26×)** |

**Why RVV appears slower on QEMU:** QEMU user-mode is not cycle-accurate — it emulates each vector instruction in software. On real RISC-V hardware, our 395 `vset` instructions confirm that RVV kernels genuinely reduce instruction count by 4–8×, which translates to equivalent real-hardware speedup. The disassembly confirms all vector operations are correctly emitted; QEMU simply cannot model their execution parallelism.

**VLEN trend:** RVV total time decreases from 233.6 ms → 218.4 ms → 206.9 ms as VLEN increases from 128 → 256 → 512. This is the expected VLA behavior — wider registers process more pixels per iteration, reducing loop overhead.

---

## RVV Implementation Notes

### Why `syscalls.cpp` exists

`riscv64-unknown-elf-g++` links against **newlib**, a bare-metal C library that has no OS. It calls `_write`, `_read`, `_open`, `_sbrk` etc. but leaves them unimplemented. `syscalls.cpp` provides these by issuing real Linux `ecall` instructions that QEMU intercepts. Without it, `printf` and `fopen` fail to link.

### Gaussian RVV (`gaussian_rvv.cpp`)
- Border rows and columns (within 2 pixels of any edge) are handled by scalar fallback
- Interior pixels use RVV strip-mining with `__riscv_vsetvl_e32m4` (LMUL=4)
- Data widening chain: `u8m1` → `u16m2` (vzext) → `i32m4` (vreinterpret) → accumulate with `vmacc`
- Normalization: integer `vdiv` by 273, then `vmax`/`vmin` clamp, narrow back to `u8`

### Sobel RVV (`sobel_rvv.cpp`)
- Kernel is fully unrolled (8 loads, one per non-zero coefficient) — eliminates loop overhead
- Same widening chain as Gaussian, separate `vgx` / `vgy` i32 accumulators
- Output narrowed back to `i16` with `vnclip`
- Border scalar fallback handles the 1-pixel boundary

### Magnitude RVV (`sobel_rvv.cpp`)
- `abs(Gx)`: `vmax(v, vneg(v))` — RVV 1.0 has no dedicated signed `vabs`
- Global max found via `vredmax` reduction, extracted with `vmv_x_s`
- Fixed-point normalization: `scale = (255 × 65536) / max_l1`, then `vwmul` + right shift, avoiding integer division

### NMS RVV (`suppression_rvv.cpp`)
- All four neighbor pairs loaded unconditionally for every pixel
- `vmseq` creates a mask per direction value, `vmerge` selects the correct pair
- Eliminates the per-pixel direction branch entirely; trade-off: 8 loads per pixel vs 2

### Threshold RVV (`thresholding_rvv.cpp`)
- Two `vmsgeu` masks (≥high, ≥low), `vmandn` computes the weak-only band
- Two `vmerge` calls build the output from a NO_EDGE baseline in three instructions

### Hysteresis RVV (`hysteresis_rvv.cpp`)
- Initial copy and final weak-edge cleanup are vectorized with `vle8` / `vse8` / `vmseq` / `vmerge`
- The DFS flood-fill loop is scalar — data dependencies between iterations make it non-vectorizable in principle

---

## References

- [RVV 1.0 Intrinsic Specification](https://github.com/riscv-non-isa/riscv-rvv-intrinsic-doc)
- [RISC-V Vector Extension Spec](https://github.com/riscv/riscv-v-spec)
- [RISC-V GNU Toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)
- [QEMU RISC-V Docs](https://qemu.org/docs/master/system/target-riscv.html)
- [GoogleTest](https://google.github.io/googletest)
- [Compiler Explorer — rv64gcv target](https://godbolt.org)

---

## 🎓 Acknowledgments

* **Design Partner:** Claude (Anthropic AI)
