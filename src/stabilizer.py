import time
from pathlib import Path
import cv2
import numpy as np
import pyopencl as cl

from src.cl_manager import OpenCLManager
from src.viewer import SynchronisedPreviewViewer, load_sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FilmStabilizer:
    """Estimates camera jitter via flicker-invariant RANSAC feature tracking and warps frames using OpenCL."""

    def __init__(self, cl_mgr: OpenCLManager, smoothing_radius: int = 20):
        self.cl_mgr = cl_mgr
        self.program = cl_mgr.load_program("stabilize.cl")
        self.smoothing_radius = smoothing_radius

    @staticmethod
    def _normalize_for_tracking(frame: np.ndarray) -> np.ndarray:
        """Removes global brightness/contrast flicker before optical flow tracking."""
        f = frame.astype(np.float32)
        mean, std = cv2.meanStdDev(f)
        norm = (f - mean[0, 0]) / max(float(std[0, 0]), 1.0)
        norm = np.clip(norm * 50.0 + 128.0, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(norm, (5, 5), 1.0)

    @staticmethod
    def _median_filter_1d(signal: np.ndarray, kernel_size: int = 9) -> np.ndarray:
        """1D sliding-window median filter for float64 trajectory arrays."""
        pad = kernel_size // 2
        padded = np.pad(signal, (pad, pad), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
        return np.median(windows, axis=-1)

    def _estimate_pairwise_matrices(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Estimates 3x3 rigid affine transforms mapping frame[i-1] -> frame[i]."""
        num_frames = len(frames)
        matrices = []

        prev_norm = self._normalize_for_tracking(frames[0])
        for i in range(1, num_frames):
            curr_norm = self._normalize_for_tracking(frames[i])

            prev_pts = cv2.goodFeaturesToTrack(
                prev_norm, maxCorners=400, qualityLevel=0.01, minDistance=18, blockSize=7
            )

            H = np.eye(3, dtype=np.float64)
            if prev_pts is not None and len(prev_pts) >= 10:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_norm, curr_norm, prev_pts, None, winSize=(21, 21), maxLevel=3
                )
                valid = status.ravel() == 1
                if np.count_nonzero(valid) >= 6:
                    mat, _ = cv2.estimateAffinePartial2D(
                        prev_pts[valid], curr_pts[valid], method=cv2.RANSAC, ransacReprojThreshold=1.5
                    )
                    if mat is not None:
                        # Enforce pure rigid transform (rotation + translation, scale = 1)
                        da = np.arctan2(mat[1, 0], mat[0, 0])
                        H[0, 0] = np.cos(da)
                        H[0, 1] = -np.sin(da)
                        H[1, 0] = np.sin(da)
                        H[1, 1] = np.cos(da)
                        H[0, 2] = mat[0, 2]
                        H[1, 2] = mat[1, 2]

            matrices.append(H)
            prev_norm = curr_norm

        return matrices

    def _smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Removes impulsive jitter spikes and smooths camera path while preserving slow pans."""
        smoothed = np.copy(trajectory)
        r = self.smoothing_radius
        x = np.arange(-r, r + 1, dtype=np.float64)
        sigma = max(2.0, r / 2.0)
        weights = np.exp(-0.5 * (x / sigma) ** 2)
        weights /= np.sum(weights)

        for dim in range(3):
            med = self._median_filter_1d(trajectory[:, dim], kernel_size=9)
            padded = np.pad(med, (r, r), mode="edge")
            smoothed[:, dim] = np.convolve(padded, weights, mode="valid")

        return smoothed

    def stabilize(self, frames: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
        """Stabilizes a sequence of grayscale frames on the OpenCL device. Returns (frames, gpu_ms)."""
        if len(frames) < 2:
            return [f.copy() for f in frames], 0.0

        height, width = frames[0].shape
        pairwise = self._estimate_pairwise_matrices(frames)

        # Accumulate global camera poses C_i relative to frame 0 (around origin (0, 0))
        poses = [np.eye(3, dtype=np.float64)]
        for H in pairwise:
            poses.append(H @ poses[-1])

        # Extract (tx, ty, angle) around origin (0, 0)
        trajectory = np.zeros((len(frames), 3), dtype=np.float64)
        for idx, C in enumerate(poses):
            trajectory[idx, 0] = C[0, 2]
            trajectory[idx, 1] = C[1, 2]
            trajectory[idx, 2] = np.arctan2(C[1, 0], C[0, 0])

        smoothed_traj = self._smooth_trajectory(trajectory)
        print(f"Max detected jitter amplitude: dx={np.ptp(trajectory[:, 0]):.2f}px, dy={np.ptp(trajectory[:, 1]):.2f}px")

        ctx = self.cl_mgr.ctx
        queue = self.cl_mgr.queue
        mf = cl.mem_flags
        frame_bytes = width * height

        src_buf = cl.Buffer(ctx, mf.READ_ONLY, size=frame_bytes)
        dst_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=frame_bytes)

        stabilized_frames = []
        total_gpu_ns = 0

        for idx, frame in enumerate(frames):
            # Build smoothed pose C_smooth around (0, 0)
            stx, sty, sa = smoothed_traj[idx]
            C_smooth = np.array([
                [np.cos(sa), -np.sin(sa), stx],
                [np.sin(sa),  np.cos(sa), sty],
                [0.0,         0.0,        1.0],
            ], dtype=np.float64)

            # Forward stabilizing transform S = C_smooth @ inv(C_raw)
            # Kernel expects inverse mapping inv(S) = C_raw @ inv(C_smooth)
            inv_S = (poses[idx] @ np.linalg.inv(C_smooth)).astype(np.float32)

            cl.enqueue_copy(queue, src_buf, np.ascontiguousarray(frame))
            event = self.program.warp_affine_bilinear(
                queue,
                (width, height),
                None,
                src_buf,
                dst_buf,
                np.int32(width),
                np.int32(height),
                inv_S[0, 0], inv_S[0, 1], inv_S[0, 2],
                inv_S[1, 0], inv_S[1, 1], inv_S[1, 2],
            )
            event.wait()
            total_gpu_ns += event.profile.end - event.profile.start

            out_frame = np.empty((height, width), dtype=np.uint8)
            cl.enqueue_copy(queue, out_frame, dst_buf).wait()
            stabilized_frames.append(out_frame)

        gpu_ms = total_gpu_ns / 1e6
        return stabilized_frames, gpu_ms


if __name__ == "__main__":
    mgr = OpenCLManager(prefer_vendor="NVIDIA")
    degraded_seq = load_sequence(PROJECT_ROOT / "data" / "degraded")

    stabilizer = FilmStabilizer(mgr, smoothing_radius=20)
    t0 = time.perf_counter()
    stabilized_seq, gpu_time_ms = stabilizer.stabilize(degraded_seq)
    total_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Stabilized {len(stabilized_seq)} frames.")
    print(f"OpenCL Kernel Time: {gpu_time_ms:.2f} ms | Total Pipeline Time: {total_ms:.2f} ms")

    viewer = SynchronisedPreviewViewer(
        degraded_seq,
        stabilized_seq,
        window_name="Stage 1: Degraded (Jittery) vs OpenCL Stabilised",
    )
    viewer.run()