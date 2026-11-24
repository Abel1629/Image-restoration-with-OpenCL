import math
import time
from pathlib import Path
import cv2
import numpy as np
import pyopencl as cl

from src.cl_manager import OpenCLManager
from src.stabilizer import FilmStabilizer
from src.viewer import SynchronisedPreviewViewer, load_sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FilmDeflicker:
    """Removes global and spatially varying film luminance flicker using OpenCL tile reduction and bilinear gain/offset."""

    def __init__(self, cl_mgr: OpenCLManager, tile_size: int = 32, temporal_radius: int = 8):
        self.cl_mgr = cl_mgr
        self.program = cl_mgr.load_program("deflicker.cl")
        self.stats_kernel = cl.Kernel(self.program, "compute_tile_stats")
        self.apply_kernel = cl.Kernel(self.program, "apply_deflicker")
        self.tile_size = tile_size
        self.temporal_radius = temporal_radius

    @staticmethod
    def _temporal_smooth_3d(data: np.ndarray, radius: int) -> np.ndarray:
        """Applies temporal median + Gaussian smoothing along axis 0 (frame index)."""
        pad_med = 3
        padded_med = np.pad(data, ((pad_med, pad_med), (0, 0), (0, 0)), mode="edge")
        med = np.median(
            np.lib.stride_tricks.sliding_window_view(padded_med, 2 * pad_med + 1, axis=0),
            axis=-1,
        )

        x = np.arange(-radius, radius + 1, dtype=np.float32)
        sigma = max(1.5, radius / 2.0)
        weights = np.exp(-0.5 * (x / sigma) ** 2)
        weights /= np.sum(weights)

        padded_g = np.pad(med, ((radius, radius), (0, 0), (0, 0)), mode="edge")
        smoothed = np.tensordot(
            np.lib.stride_tricks.sliding_window_view(padded_g, 2 * radius + 1, axis=0),
            weights,
            axes=([-1], [0]),
        )
        return smoothed.astype(np.float32)

    def deflicker(self, frames: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
        """Equalizes luminance fluctuations across frames on the OpenCL device. Returns (frames, gpu_ms)."""
        if len(frames) < 2:
            return [f.copy() for f in frames], 0.0

        num_frames = len(frames)
        height, width = frames[0].shape
        grid_w = math.ceil(width / self.tile_size)
        grid_h = math.ceil(height / self.tile_size)
        num_tiles = grid_w * grid_h

        ctx = self.cl_mgr.ctx
        queue = self.cl_mgr.queue
        mf = cl.mem_flags

        img_buf = cl.Buffer(ctx, mf.READ_ONLY, size=width * height)
        means_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_tiles * 4)
        stds_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_tiles * 4)

        all_means = np.empty((num_frames, grid_h, grid_w), dtype=np.float32)
        all_stds = np.empty((num_frames, grid_h, grid_w), dtype=np.float32)
        total_gpu_ns = 0

        local_size = (16, 16)
        global_size_stats = (grid_w * local_size[0], grid_h * local_size[1])

        for idx, frame in enumerate(frames):
            cl.enqueue_copy(queue, img_buf, np.ascontiguousarray(frame))
            ev = self.stats_kernel(
                queue,
                global_size_stats,
                local_size,
                img_buf,
                means_buf,
                stds_buf,
                np.int32(width),
                np.int32(height),
                np.int32(self.tile_size),
                np.int32(grid_w),
                np.int32(grid_h),
            )
            ev.wait()
            total_gpu_ns += ev.profile.end - ev.profile.start

            cl.enqueue_copy(queue, all_means[idx], means_buf).wait()
            cl.enqueue_copy(queue, all_stds[idx], stds_buf).wait()

        target_means = self._temporal_smooth_3d(all_means, self.temporal_radius)
        target_stds = self._temporal_smooth_3d(all_stds, self.temporal_radius)

        for idx in range(num_frames):
            target_means[idx] = cv2.GaussianBlur(target_means[idx], (3, 3), 0.8)
            target_stds[idx] = cv2.GaussianBlur(target_stds[idx], (3, 3), 0.8)

        gains = np.clip(target_stds / np.maximum(all_stds, 6.0), 0.72, 1.38).astype(np.float32)
        offsets = (target_means - gains * all_means).astype(np.float32)

        dst_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=width * height)
        gain_buf = cl.Buffer(ctx, mf.READ_ONLY, size=num_tiles * 4)
        offset_buf = cl.Buffer(ctx, mf.READ_ONLY, size=num_tiles * 4)

        deflickered_frames = []
        for idx, frame in enumerate(frames):
            cl.enqueue_copy(queue, img_buf, np.ascontiguousarray(frame))
            cl.enqueue_copy(queue, gain_buf, np.ascontiguousarray(gains[idx]))
            cl.enqueue_copy(queue, offset_buf, np.ascontiguousarray(offsets[idx]))

            ev = self.apply_kernel(
                queue,
                (width, height),
                None,
                img_buf,
                dst_buf,
                gain_buf,
                offset_buf,
                np.int32(width),
                np.int32(height),
                np.int32(self.tile_size),
                np.int32(grid_w),
                np.int32(grid_h),
            )
            ev.wait()
            total_gpu_ns += ev.profile.end - ev.profile.start

            out_frame = np.empty((height, width), dtype=np.uint8)
            cl.enqueue_copy(queue, out_frame, dst_buf).wait()
            deflickered_frames.append(out_frame)

        gpu_ms = total_gpu_ns / 1e6
        return deflickered_frames, gpu_ms


if __name__ == "__main__":
    mgr = OpenCLManager(prefer_vendor="NVIDIA")
    degraded_seq = load_sequence(PROJECT_ROOT / "data" / "degraded")

    stabilizer = FilmStabilizer(mgr, smoothing_radius=20)
    stabilized_seq, stab_ms = stabilizer.stabilize(degraded_seq)

    deflicker_filter = FilmDeflicker(mgr, tile_size=32, temporal_radius=8)
    t0 = time.perf_counter()
    deflickered_seq, deflicker_gpu_ms = deflicker_filter.deflicker(stabilized_seq)
    total_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Deflickered {len(deflickered_seq)} frames.")
    print(f"OpenCL Kernel Time: {deflicker_gpu_ms:.2f} ms | Total Stage Time: {total_ms:.2f} ms")

    viewer = SynchronisedPreviewViewer(
        stabilized_seq,
        deflickered_seq,
        window_name="Stage 2: Stabilised (Flickering) vs OpenCL Deflickered",
    )
    viewer.run()