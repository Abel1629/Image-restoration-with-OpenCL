# Restoration of Digitised Historical Film using OpenCL

A GPU-accelerated multi-stage restoration pipeline for digitised historical film sequences, implemented in Python and OpenCL (`pyopencl`). The system addresses mechanical camera jitter, global and spatially varying luminance flicker, impulsive dust blotches, persistent vertical roller scratches, and film grain while preserving genuine object motion.

---

## Table of Contents
1. [Pipeline Architecture](#pipeline-architecture)
2. [Project Structure](#project-structure)
3. [Setup & Installation](#setup--installation)
4. [Usage & CLI Options](#usage--cli-options)
5. [Experimental Results & Performance Analysis](#experimental-results--performance-analysis)
6. [Key Architectural Insights](#key-architectural-insights)
7. [Version Control & Repository Finalization](#version-control--repository-finalization)

---

## Pipeline Architecture

The restoration pipeline processes grayscale frame sequences sequentially across three heterogeneous CPU/GPU stages, followed by an interactive synchronised comparison viewer:

1. **Stage 1 — Frame Jitter Stabilisation (`src/stabilizer.py`, `kernels/stabilize.cl`)**
   * Normalizes frame luminance prior to feature extraction so global brightness flicker does not bias optical flow tracking.
   * Tracks Shi-Tomasi keypoints via Lucas-Kanade pyramidal optical flow and estimates frame-to-frame rigid affine transforms using RANSAC (`cv2.estimateAffinePartial2D`).
   * Smooths cumulative camera trajectories via 1D median + Gaussian filtering (`smoothing_radius = 20`) to remove high-frequency mechanical gate jitter while preserving intentional camera motion.
   * Executes inverse affine warping with bilinear interpolation in parallel across all pixels on the OpenCL device.

2. **Stage 2 — Spatio-Temporal Deflicker (`src/deflicker.py`, `kernels/deflicker.cl`)**
   * Divides each frame into spatial tiles ($48 \times 48$ pixels) and computes per-tile mean and standard deviation using OpenCL local-memory (`__local`) parallel reductions.
   * Applies temporal median + Gaussian smoothing across a sliding frame window ($\pm 8$ frames) to establish stable target exposure statistics.
   * Applies bilinearly interpolated per-pixel gain and offset compensation on the GPU to eliminate both global exposure swings and uneven spatial illumination without block seams.

3. **Stage 3 — Motion-Protected Dust, Scratch & Grain Removal (`src/dust_scratch_remover.py`, `kernels/inpaint_temporal.cl`)**
   * **Motion Protection (`compute_motion_mask`):** Evaluates temporal differences between $I_{t-1}$ and $I_{t+1}$ (`|prev - next|`) to build a morphological protection zone around moving foreground objects (e.g., moving vehicles and swinging pendulums), preventing false erasure or ghosting.
   * **Impulse Inpainting (`inpaint_defects`):** Detects single-frame dust blotches outside the moving-object mask via temporal outlier consensus and suppresses persistent vertical roller scratches via 21-pixel vertical contrast profiling, reconstructing damaged regions via temporal averaging or horizontal flank interpolation.
   * **Spatio-Temporal Bilateral Denoising (`spatiotemporal_bilateral`):** Combines $5 \times 5$ spatial bilateral filtering on $I_t$ with $3 \times 3$ temporal bilateral support from $I_{t-1}$ and $I_{t+1}$, reducing background film grain while preserving sharp object boundaries.

4. **Interactive Synchronised Preview (`src/viewer.py`)**
   * Provides an OpenCV GUI supporting side-by-side and split-screen wipe comparison modes, frame scrubbing, playback controls (`Space` to play/pause, `A`/`D` to step frames, `M` to switch view mode, `Q` to quit), and real-time visual inspection.

---

## Project Structure

```text
ImageRestorationWithOpenCL/
├── kernels/
│   ├── stabilize.cl              # Bilinear inverse affine warping kernel
│   ├── deflicker.cl              # Local-memory tile reduction & bilinear gain/offset kernels
│   └── inpaint_temporal.cl      # Motion mask, dust/scratch inpainting & spatio-temporal bilateral kernels
├── src/
│   ├── cl_manager.py             # OpenCL platform/device selection, program caching & profiling
│   ├── generate_test_sequence.py # Synthetic historical film ground-truth & degradation generator
│   ├── stabilizer.py             # Stage 1: Trajectory estimation & OpenCL frame stabilization
│   ├── deflicker.py              # Stage 2: Tile luminance profiling & OpenCL deflicker
│   ├── dust_scratch_remover.py   # Stage 3: OpenCL dust/scratch removal & bilateral denoising
│   ├── evaluator.py              # PSNR/SSIM quality metrics & CPU baseline implementation
│   └── viewer.py                 # Synchronised before-and-after OpenCV preview player
├── data/
│   ├── clean/                    # Ground-truth synthetic frames (90 frames, 640x360)
│   ├── degraded/                 # Degraded input frames (jitter, flicker, dust, scratches, grain)
│   └── restored/                 # Restored output frames
├── main.py                       # Main CLI entry point & multi-device benchmark runner
└── README.md                     # Project documentation and experimental results
```

---

## Setup & Installation

### Prerequisites
* Python 3.10+ (tested on Windows 11 with Python 3.14)
* OpenCL-capable GPU drivers (NVIDIA CUDA/OpenCL runtime and/or Intel Graphics Compute Runtime)

### Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy opencv-python pyopencl scikit-image
```

---

## Usage & CLI Options

### 1. Run the Full Restoration Pipeline & Interactive Preview
```bash
python main.py
```
* Automatically generates the 90-frame ($640 \times 360$) synthetic test sequence in `data/clean/` and `data/degraded/` if not already present.
* Runs all three OpenCL restoration stages on the preferred GPU, writes restored frames to `data/restored/`, prints PSNR/SSIM metrics, and opens the synchronised before-and-after viewer.

### 2. Run the Multi-Device Parallel Performance Benchmark
```bash
python main.py --benchmark
```
* Executes the full pipeline across three compute targets—**CPU Baseline (NumPy/OpenCV)**, **Intel UHD Integrated GPU**, and **NVIDIA GeForce RTX 4050 Discrete GPU**—and outputs kernel timings, end-to-end wall times, and speedup factors.

### 3. Additional CLI Flags
* `--vendor Intel` — Select Intel integrated graphics instead of the NVIDIA discrete GPU.
* `--no-gui` — Run processing and metrics evaluation headlessly without launching the OpenCV preview window.

---

## Experimental Results & Performance Analysis

All benchmarks were recorded on a **90-frame sequence at $640 \times 360$ resolution** (`20,736,000` total pixels processed per stage).

### 1. Quantitative Restoration Quality (vs. Ground Truth)
| Metric | Degraded Input | Restored Output | Net Improvement |
| :--- | :---: | :---: | :---: |
| **PSNR (dB)** | 20.01 dB | **22.32 dB** | **+2.32 dB** |
| **SSIM** | 0.6311 | **0.8629** | **+0.2319** |

### 2. Multi-Device Parallel Performance Benchmark
| Execution Target | OpenCL Kernel Time | Total Pipeline Wall Time | End-to-End Speedup |
| :--- | :---: | :---: | :---: |
| **CPU Baseline (NumPy / OpenCV)** | — | 7,214.02 ms | 1.00× |
| **Intel(R) UHD Graphics (14,819 MB VRAM, 80 CUs)** | 510.80 ms | 1,396.15 ms | **5.17×** |
| **NVIDIA GeForce RTX 4050 Laptop GPU (6,140 MB VRAM, 20 CUs)** | **18.17 ms** | **747.36 ms** | **9.65×** |

### 3. Stage-by-Stage Execution Breakdown (NVIDIA RTX 4050)
| Pipeline Stage | OpenCL Kernel Time | Total Stage Wall Time | Effective Throughput |
| :--- | :---: | :---: | :---: |
| **Stage 1 — Jitter Stabilisation** | 1.00 ms | 637.15 ms | ~141 FPS |
| **Stage 2 — Spatio-Temporal Deflicker** | 3.69 ms | 57.28 ms | ~1,571 FPS |
| **Stage 3 — Dust, Scratch & Grain Removal** | 15.44 ms | 64.03 ms | ~1,406 FPS |
| **Total 3-Stage Pipeline** | **20.13 ms** | **758.46 ms** | **~119 FPS** |

---

## Key Architectural Insights

* **Sub-Millisecond GPU Compute & Hardware Scaling:** Across all 90 frames, combined OpenCL kernel execution on the **NVIDIA RTX 4050** completes in **18.17–20.13 ms** (~**0.22 ms per frame**, or >4,000 FPS of pure GPU compute). Compared to the integrated **Intel UHD Graphics** (`510.80 ms` kernel time), the discrete GPU achieves **28× faster kernel execution** and nearly **2× faster end-to-end wall-clock runtime** (`747.36 ms` vs. `1,396.15 ms`).
* **Amdahl's Law in Heterogeneous Pipelines:** In Stage 1, the GPU affine warping kernel executes in just **1.00 ms**, while the overall stage takes **637.15 ms** due to sequential CPU optical-flow feature tracking (`cv2.goodFeaturesToTrack` + `cv2.calcOpticalFlowPyrLK`) and host-to-device buffer transfers. Once pixel-parallel operations are offloaded to OpenCL, sequential host preprocessing becomes the primary runtime bottleneck.
* **PSNR vs. SSIM Trade-off in Trajectory Smoothing:** Structural similarity (**SSIM**) improves substantially from **0.6311 to 0.8629 (+0.2319)**, confirming effective flicker removal, impulse inpainting, and noise suppression. Pixel-wise **PSNR** reaches **22.32 dB (+2.32 dB)** because trajectory smoothing intentionally preserves smooth camera paths rather than snapping frames back to the exact unjittered ground-truth pixel coordinates.

---

## Version Control & Repository Finalization

To commit the finalized project and documentation to Git:

```bash
git status
git add .
git commit -m "docs: add complete project README with architecture and benchmark results"
git push
```