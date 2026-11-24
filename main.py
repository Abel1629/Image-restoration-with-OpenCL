import argparse
import time
from pathlib import Path
import cv2
import numpy as np

from src.cl_manager import OpenCLManager
from src.stabilizer import FilmStabilizer
from src.deflicker import FilmDeflicker
from src.dust_scratch_remover import DustAndScratchRemover
from src.evaluator import evaluate_sequence_quality, run_cpu_baseline_pipeline
from src.generate_test_sequence import generate_clean_sequence, degrade_sequence
from src.viewer import SynchronisedPreviewViewer, load_sequence

PROJECT_ROOT = Path(__file__).resolve().parent


def save_sequence(frames: list[np.ndarray], output_dir: Path) -> None:
    """Saves restored PNG frames to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(output_dir / f"frame_{idx:04d}.png"), frame)


def run_opencl_pipeline(
    degraded_frames: list[np.ndarray],
    prefer_vendor: str = "NVIDIA",
) -> tuple[list[np.ndarray], dict[str, float], str]:
    """Executes the full 3-stage OpenCL restoration pipeline on the specified GPU vendor."""
    mgr = OpenCLManager(prefer_vendor=prefer_vendor, enable_profiling=True)

    stabilizer = FilmStabilizer(mgr, smoothing_radius=20)
    deflicker = FilmDeflicker(mgr, tile_size=48, temporal_radius=8)
    remover = DustAndScratchRemover(mgr)

    t0 = time.perf_counter()
    stabilized_frames, gpu_stab_ms = stabilizer.stabilize(degraded_frames)
    wall_stab_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    deflickered_frames, gpu_defl_ms = deflicker.deflicker(stabilized_frames)
    wall_defl_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    restored_frames, gpu_clean_ms = remover.clean(deflickered_frames)
    wall_clean_ms = (time.perf_counter() - t2) * 1000.0

    timings = {
        "gpu_stab_ms": gpu_stab_ms,
        "gpu_defl_ms": gpu_defl_ms,
        "gpu_clean_ms": gpu_clean_ms,
        "gpu_total_ms": gpu_stab_ms + gpu_defl_ms + gpu_clean_ms,
        "wall_stab_ms": wall_stab_ms,
        "wall_defl_ms": wall_defl_ms,
        "wall_clean_ms": wall_clean_ms,
        "wall_total_ms": wall_stab_ms + wall_defl_ms + wall_clean_ms,
    }
    return restored_frames, timings, mgr.info()


def main() -> None:
    parser = argparse.ArgumentParser(description="MSc OpenCL Historical Film Restoration Pipeline")
    parser.add_argument("--vendor", type=str, default="NVIDIA", help="Preferred OpenCL GPU vendor (NVIDIA or Intel)")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark NVIDIA GPU vs Intel GPU vs CPU baseline")
    parser.add_argument("--no-gui", action="store_true", help="Skip launching the interactive OpenCV preview window")
    args = parser.parse_args()

    clean_dir = PROJECT_ROOT / "data" / "clean"
    degraded_dir = PROJECT_ROOT / "data" / "degraded"
    restored_dir = PROJECT_ROOT / "data" / "restored"

    # Ensure test dataset exists
    if not list(degraded_dir.glob("*.png")):
        print("Test sequences not found. Generating synthetic historical film dataset...")
        clean_frames = generate_clean_sequence(clean_dir, num_frames=90)
        degraded_frames = degrade_sequence(clean_frames, degraded_dir)
    else:
        degraded_frames = load_sequence(degraded_dir)
        clean_frames = load_sequence(clean_dir) if list(clean_dir.glob("*.png")) else []

    print("=" * 72)
    print("RESTORATION OF DIGITISED HISTORICAL FILM (OPENCL PIPELINE)")
    print("=" * 72)
    print(f"Loaded {len(degraded_frames)} frames ({degraded_frames[0].shape[1]}x{degraded_frames[0].shape[0]})")

    # Run primary OpenCL restoration pipeline
    restored_frames, gpu_timings, device_info = run_opencl_pipeline(degraded_frames, prefer_vendor=args.vendor)
    save_sequence(restored_frames, restored_dir)

    print(f"\nActive OpenCL Device: {device_info}")
    print("-" * 72)
    print(f"Stage 1 - Jitter Stabilisation : Kernel {gpu_timings['gpu_stab_ms']:7.2f} ms | Total {gpu_timings['wall_stab_ms']:7.2f} ms")
    print(f"Stage 2 - Deflicker            : Kernel {gpu_timings['gpu_defl_ms']:7.2f} ms | Total {gpu_timings['wall_defl_ms']:7.2f} ms")
    print(f"Stage 3 - Dust & Scratch Clean : Kernel {gpu_timings['gpu_clean_ms']:7.2f} ms | Total {gpu_timings['wall_clean_ms']:7.2f} ms")
    print("-" * 72)
    print(f"TOTAL PIPELINE                 : Kernel {gpu_timings['gpu_total_ms']:7.2f} ms | Total {gpu_timings['wall_total_ms']:7.2f} ms")
    print(f"Saved {len(restored_frames)} restored frames to '{restored_dir}'.")

    # Quantitative quality evaluation if ground-truth clean frames are available
    if clean_frames:
        metrics = evaluate_sequence_quality(clean_frames, degraded_frames, restored_frames)
        print("\n" + "=" * 72)
        print("QUANTITATIVE RESTORATION QUALITY (vs. Ground Truth)")
        print("=" * 72)
        print(f"PSNR : {metrics['psnr_before']:6.2f} dB (Degraded)  -->  {metrics['psnr_after']:6.2f} dB (Restored)  [+{metrics['psnr_after'] - metrics['psnr_before']:.2f} dB]")
        print(f"SSIM : {metrics['ssim_before']:6.4f}    (Degraded)  -->  {metrics['ssim_after']:6.4f}    (Restored)  [+{metrics['ssim_after'] - metrics['ssim_before']:.4f}]")

    # Optional multi-device benchmark (CPU vs Intel UHD vs NVIDIA RTX 4050)
    if args.benchmark:
        print("\n" + "=" * 72)
        print("PARALLEL PERFORMANCE BENCHMARK (CPU vs. Integrated GPU vs. Discrete GPU)")
        print("=" * 72)
        _, cpu_timings = run_cpu_baseline_pipeline(degraded_frames)
        _, intel_timings, intel_info = run_opencl_pipeline(degraded_frames, prefer_vendor="Intel")
        _, nvidia_timings, nvidia_info = run_opencl_pipeline(degraded_frames, prefer_vendor="NVIDIA")

        print(f"1. CPU Baseline (NumPy/OpenCV) : {cpu_timings['total_ms']:8.2f} ms  (1.00x)")
        print(f"2. {intel_info:<28}: {intel_timings['wall_total_ms']:8.2f} ms  ({cpu_timings['total_ms'] / intel_timings['wall_total_ms']:.2f}x speedup | kernel: {intel_timings['gpu_total_ms']:.2f} ms)")
        print(f"3. {nvidia_info:<28}: {nvidia_timings['wall_total_ms']:8.2f} ms  ({cpu_timings['total_ms'] / nvidia_timings['wall_total_ms']:.2f}x speedup | kernel: {nvidia_timings['gpu_total_ms']:.2f} ms)")

    if not args.no_gui:
        print("\nLaunching Synchronised Before-and-After Preview Window...")
        viewer = SynchronisedPreviewViewer(
            degraded_frames,
            restored_frames,
            window_name="Historical Film Restoration (OpenCL) - Before vs After",
        )
        viewer.run()


if __name__ == "__main__":
    main()