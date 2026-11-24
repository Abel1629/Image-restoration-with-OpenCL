import time
from pathlib import Path
import numpy as np
import pyopencl as cl

from src.cl_manager import OpenCLManager
from src.stabilizer import FilmStabilizer
from src.deflicker import FilmDeflicker
from src.viewer import SynchronisedPreviewViewer, load_sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DustAndScratchRemover:
    """GPU-accelerated motion-compensated dust blotch and vertical scratch remover."""

    def __init__(
        self,
        cl_mgr: OpenCLManager,
        search_radius: int = 22,
        patch_radius: int = 3,
        dust_thresh: float = 18.0,
        neighbor_agree_thresh: float = 18.0,
        scratch_thresh: float = 28.0,
    ):
        self.cl_mgr = cl_mgr
        self.program = cl_mgr.load_program("inpaint_temporal.cl")

        # Cache kernel instances once to avoid RepeatedKernelRetrieval overhead
        self.k_block_motion = cl.Kernel(self.program, "estimate_block_motion")
        self.k_sym_motion = cl.Kernel(self.program, "estimate_symmetric_motion")
        self.k_detect_defects = cl.Kernel(self.program, "detect_temporal_defects")
        self.k_reconstruct = cl.Kernel(self.program, "reconstruct_frame")

        self.search_radius = search_radius
        self.patch_radius = patch_radius
        self.dust_thresh = dust_thresh
        self.neighbor_agree_thresh = neighbor_agree_thresh
        self.scratch_thresh = scratch_thresh

    def clean(self, frames: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
        """Removes impulsive dust blotches and persistent vertical scratches while preserving genuine motion."""
        num_frames = len(frames)
        if num_frames < 3:
            return [f.copy() for f in frames], 0.0

        height, width = frames[0].shape
        num_pixels = width * height

        ctx = self.cl_mgr.ctx
        queue = self.cl_mgr.queue
        mf = cl.mem_flags

        frame_bufs = [
            cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(f))
            for f in frames
        ]

        flow_prev_buf = cl.Buffer(ctx, mf.READ_WRITE, size=num_pixels * 4)
        flow_next_buf = cl.Buffer(ctx, mf.READ_WRITE, size=num_pixels * 4)
        flow_sym_buf = cl.Buffer(ctx, mf.READ_WRITE, size=num_pixels * 4)
        temporal_ref_buf = cl.Buffer(ctx, mf.READ_WRITE, size=num_pixels)
        defect_mask_buf = cl.Buffer(ctx, mf.READ_WRITE, size=num_pixels)
        dst_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=num_pixels)

        cleaned_frames = []
        total_gpu_ns = 0

        for idx in range(num_frames):
            prev_idx = 1 if idx == 0 else idx - 1
            next_idx = num_frames - 2 if idx == num_frames - 1 else idx + 1

            ev1 = self.k_block_motion(
                queue,
                (width, height),
                None,
                frame_bufs[idx],
                frame_bufs[prev_idx],
                flow_prev_buf,
                np.int32(width),
                np.int32(height),
                np.int32(self.search_radius),
                np.int32(self.patch_radius),
            )

            ev2 = self.k_block_motion(
                queue,
                (width, height),
                None,
                frame_bufs[idx],
                frame_bufs[next_idx],
                flow_next_buf,
                np.int32(width),
                np.int32(height),
                np.int32(self.search_radius),
                np.int32(self.patch_radius),
            )

            ev3 = self.k_sym_motion(
                queue,
                (width, height),
                None,
                frame_bufs[prev_idx],
                frame_bufs[next_idx],
                flow_sym_buf,
                np.int32(width),
                np.int32(height),
                np.int32(self.search_radius),
                np.int32(self.patch_radius),
            )

            ev4 = self.k_detect_defects(
                queue,
                (width, height),
                None,
                frame_bufs[prev_idx],
                frame_bufs[idx],
                frame_bufs[next_idx],
                flow_prev_buf,
                flow_next_buf,
                flow_sym_buf,
                temporal_ref_buf,
                defect_mask_buf,
                np.int32(width),
                np.int32(height),
                np.float32(self.dust_thresh),
                np.float32(self.neighbor_agree_thresh),
            )

            ev5 = self.k_reconstruct(
                queue,
                (width, height),
                None,
                frame_bufs[idx],
                temporal_ref_buf,
                defect_mask_buf,
                dst_buf,
                np.int32(width),
                np.int32(height),
                np.float32(self.scratch_thresh),
            )

            ev5.wait()
            for ev in (ev1, ev2, ev3, ev4, ev5):
                total_gpu_ns += ev.profile.end - ev.profile.start

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

    deflicker = FilmDeflicker(mgr, tile_size=32, temporal_radius=8)
    deflickered_seq, deflicker_ms = deflicker.deflicker(stabilized_seq)

    remover = DustAndScratchRemover(mgr)
    t0 = time.perf_counter()
    restored_seq, clean_gpu_ms = remover.clean(deflickered_seq)
    total_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Cleaned dust & scratches on {len(restored_seq)} frames.")
    print(f"OpenCL Kernel Time: {clean_gpu_ms:.2f} ms | Total Stage Time: {total_ms:.2f} ms")

    viewer = SynchronisedPreviewViewer(
        degraded_seq,
        restored_seq,
        window_name="Full Pipeline: Degraded vs Restored (Stabilised + Deflickered + Denoised)",
    )
    viewer.run()