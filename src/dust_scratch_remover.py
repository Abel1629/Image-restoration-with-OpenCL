import time
from pathlib import Path
import numpy as np
import pyopencl as cl

from src.cl_manager import OpenCLManager
from src.stabilizer import FilmStabilizer
from src.deflicker import FilmDeflicker
from src.viewer import SynchronisedPreviewViewer, load_sequence

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "src" else _HERE


class DustAndScratchRemover:
    """GPU-accelerated motion-protected dust/scratch inpainter with spatio-temporal bilateral denoising."""

    def __init__(
        self,
        cl_mgr: OpenCLManager,
        motion_thresh: float = 18.0,
        dust_thresh: float = 16.0,
        scratch_thresh: float = 24.0,
        sigma_spatial: float = 1.5,
        sigma_range: float = 13.5,
    ):
        self.cl_mgr = cl_mgr
        self.program = cl_mgr.load_program("inpaint_temporal.cl")

        self.k_motion = cl.Kernel(self.program, "compute_motion_mask")
        self.k_inpaint = cl.Kernel(self.program, "inpaint_defects")
        self.k_bilateral = cl.Kernel(self.program, "spatiotemporal_bilateral")

        self.motion_thresh = motion_thresh
        self.dust_thresh = dust_thresh
        self.scratch_thresh = scratch_thresh
        self.sigma_spatial = sigma_spatial
        self.sigma_range = sigma_range

    def clean(self, frames: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
        """Protects moving objects via |prev - next|, inpaints dust/scratches, and applies bilateral denoising."""
        num_frames = len(frames)
        if num_frames < 3:
            return [f.copy() for f in frames], 0.0

        height, width = frames[0].shape
        num_pixels = width * height

        ctx = self.cl_mgr.ctx
        queue = self.cl_mgr.queue
        mf = cl.mem_flags

        raw_bufs = [
            cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(f))
            for f in frames
        ]
        inpainted_bufs = [cl.Buffer(ctx, mf.READ_WRITE, size=num_pixels) for _ in range(num_frames)]
        motion_buf = cl.Buffer(ctx, mf.READ_WRITE, size=num_pixels)
        dst_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_pixels)

        total_gpu_ns = 0

        # Pass A: Build moving-object mask & inpaint dust + vertical scratches
        for idx in range(num_frames):
            prev_idx = 1 if idx == 0 else idx - 1
            next_idx = num_frames - 2 if idx == num_frames - 1 else idx + 1

            ev1 = self.k_motion(
                queue,
                (width, height),
                None,
                raw_bufs[prev_idx],
                raw_bufs[next_idx],
                motion_buf,
                np.int32(width),
                np.int32(height),
                np.float32(self.motion_thresh),
            )

            ev2 = self.k_inpaint(
                queue,
                (width, height),
                None,
                raw_bufs[prev_idx],
                raw_bufs[idx],
                raw_bufs[next_idx],
                motion_buf,
                inpainted_bufs[idx],
                np.int32(width),
                np.int32(height),
                np.float32(self.dust_thresh),
                np.float32(self.scratch_thresh),
            )

            ev2.wait()
            total_gpu_ns += (ev1.profile.end - ev1.profile.start) + (ev2.profile.end - ev2.profile.start)

        # Pass B: Spatio-temporal bilateral denoising across all frames
        cleaned_frames = []
        for idx in range(num_frames):
            prev_idx = 1 if idx == 0 else idx - 1
            next_idx = num_frames - 2 if idx == num_frames - 1 else idx + 1

            ev3 = self.k_bilateral(
                queue,
                (width, height),
                None,
                inpainted_bufs[prev_idx],
                inpainted_bufs[idx],
                inpainted_bufs[next_idx],
                dst_buf,
                np.int32(width),
                np.int32(height),
                np.float32(self.sigma_spatial),
                np.float32(self.sigma_range),
            )
            ev3.wait()
            total_gpu_ns += ev3.profile.end - ev3.profile.start

            out_frame = np.empty((height, width), dtype=np.uint8)
            cl.enqueue_copy(queue, out_frame, dst_buf).wait()
            cleaned_frames.append(out_frame)

        gpu_ms = total_gpu_ns / 1e6
        return cleaned_frames, gpu_ms


if __name__ == "__main__":
    mgr = OpenCLManager(prefer_vendor="NVIDIA")
    degraded_seq = load_sequence(PROJECT_ROOT / "data" / "degraded")

    stabilizer = FilmStabilizer(mgr, smoothing_radius=20)
    stabilized_seq, stab_ms = stabilizer.stabilize(degraded_seq)

    deflicker = FilmDeflicker(mgr, tile_size=48, temporal_radius=8)
    deflickered_seq, deflicker_ms = deflicker.deflicker(stabilized_seq)

    remover = DustAndScratchRemover(mgr)
    t0 = time.perf_counter()
    restored_seq, clean_gpu_ms = remover.clean(deflickered_seq)
    total_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Restored {len(restored_seq)} frames.")
    print(f"OpenCL Kernel Time: {clean_gpu_ms:.2f} ms | Total Stage Time: {total_ms:.2f} ms")

    viewer = SynchronisedPreviewViewer(
        degraded_seq,
        restored_seq,
        window_name="Full Pipeline: Degraded vs Restored (Motion-Protected OpenCL)",
    )
    viewer.run()