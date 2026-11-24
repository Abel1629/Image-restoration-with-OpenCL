import time
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def evaluate_sequence_quality(
    clean_frames: list[np.ndarray],
    degraded_frames: list[np.ndarray],
    restored_frames: list[np.ndarray],
) -> dict[str, float]:
    """Computes average PSNR (dB) and SSIM across all frames before and after restoration."""
    n = min(len(clean_frames), len(degraded_frames), len(restored_frames))
    psnr_before, psnr_after = [], []
    ssim_before, ssim_after = [], []

    for i in range(n):
        c = clean_frames[i]
        d = degraded_frames[i]
        r = restored_frames[i]

        psnr_before.append(compute_psnr(c, d, data_range=255))
        psnr_after.append(compute_psnr(c, r, data_range=255))
        ssim_before.append(compute_ssim(c, d, data_range=255))
        ssim_after.append(compute_ssim(c, r, data_range=255))

    return {
        "psnr_before": float(np.mean(psnr_before)),
        "psnr_after": float(np.mean(psnr_after)),
        "ssim_before": float(np.mean(ssim_before)),
        "ssim_after": float(np.mean(ssim_after)),
    }


def run_cpu_baseline_pipeline(frames: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, float]]:
    """
    Pure CPU reference implementation (OpenCV/NumPy) of the 3-stage restoration pipeline
    used to measure parallel GPU speedup on identical workloads.
    """
    num_frames = len(frames)
    height, width = frames[0].shape

    # Stage 1 (CPU): Feature tracking + OpenCV warpAffine
    t0 = time.perf_counter()
    matrices = []
    prev_gray = cv2.GaussianBlur(frames[0], (5, 5), 1.0)
    for i in range(1, num_frames):
        curr_gray = cv2.GaussianBlur(frames[i], (5, 5), 1.0)
        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=18, blockSize=7)
        H = np.eye(3, dtype=np.float64)
        if pts is not None and len(pts) >= 10:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)
            valid = status.ravel() == 1
            if np.count_nonzero(valid) >= 6:
                mat, _ = cv2.estimateAffinePartial2D(pts[valid], curr_pts[valid], method=cv2.RANSAC)
                if mat is not None:
                    da = np.arctan2(mat[1, 0], mat[0, 0])
                    H[:2, :2] = [[np.cos(da), -np.sin(da)], [np.sin(da), np.cos(da)]]
                    H[0, 2], H[1, 2] = mat[0, 2], mat[1, 2]
        matrices.append(H)
        prev_gray = curr_gray

    poses = [np.eye(3, dtype=np.float64)]
    for H in matrices:
        poses.append(H @ poses[-1])

    stab_cpu = []
    for idx, frame in enumerate(frames):
        S = np.linalg.inv(poses[idx])[:2, :]
        stab_cpu.append(cv2.warpAffine(frame, S, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101))
    cpu_stab_ms = (time.perf_counter() - t0) * 1000.0

    # Stage 2 (CPU): Frame luminance equalization
    t1 = time.perf_counter()
    means = np.array([float(np.mean(f)) for f in stab_cpu], dtype=np.float32)
    target_mean = float(np.median(means))
    defl_cpu = []
    for idx, frame in enumerate(stab_cpu):
        gain = target_mean / max(means[idx], 1.0)
        defl_cpu.append(np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8))
    cpu_defl_ms = (time.perf_counter() - t1) * 1000.0

    # Stage 3 (CPU): Farneback dense optical flow + temporal outlier replacement
    t2 = time.perf_counter()
    clean_cpu = []
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    for idx in range(num_frames):
        prev_idx = 1 if idx == 0 else idx - 1
        next_idx = num_frames - 2 if idx == num_frames - 1 else idx + 1

        flow_p = cv2.calcOpticalFlowFarneback(defl_cpu[idx], defl_cpu[prev_idx], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_n = cv2.calcOpticalFlowFarneback(defl_cpu[idx], defl_cpu[next_idx], None, 0.5, 3, 15, 3, 5, 1.2, 0)

        p_warp = cv2.remap(defl_cpu[prev_idx], grid_x + flow_p[..., 0], grid_y + flow_p[..., 1], cv2.INTER_LINEAR).astype(np.float32)
        n_warp = cv2.remap(defl_cpu[next_idx], grid_x + flow_n[..., 0], grid_y + flow_n[..., 1], cv2.INTER_LINEAR).astype(np.float32)
        c_val = defl_cpu[idx].astype(np.float32)

        t_avg = 0.5 * (p_warp + n_warp)
        mask = (np.abs(p_warp - n_warp) < 18.0) & (((c_val - p_warp) * (c_val - n_warp)) > 0) & (np.minimum(np.abs(c_val - p_warp), np.abs(c_val - n_warp)) > 18.0)
        out = np.where(mask, t_avg, c_val)
        clean_cpu.append(np.clip(out, 0, 255).astype(np.uint8))
    cpu_clean_ms = (time.perf_counter() - t2) * 1000.0

    timings = {
        "stabilize_ms": cpu_stab_ms,
        "deflicker_ms": cpu_defl_ms,
        "denoise_ms": cpu_clean_ms,
        "total_ms": cpu_stab_ms + cpu_defl_ms + cpu_clean_ms,
    }
    return clean_cpu, timings