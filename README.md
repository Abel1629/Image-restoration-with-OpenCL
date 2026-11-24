# Restoration of Digitised Historical Film (OpenCL)

MSc Parallel Image Processing project implementing GPU-accelerated restoration of digitised historical film sequences using Python and OpenCL.

## Features
- **Frame Jitter Stabilisation:** Global motion estimation and GPU bilinear warping.
- **Deflicker:** Sliding-window luminance equalisation via OpenCL reduction kernels.
- **Motion-Compensated Dust & Scratch Removal:** Optical-flow-guided spatio-temporal filtering that removes impulsive defects while preserving genuine motion.
- **Synchronised Preview:** Interactive OpenCV split-screen and side-by-side playback viewer.

## Setup
1. Create and activate a virtual environment (Python 3.10+):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
