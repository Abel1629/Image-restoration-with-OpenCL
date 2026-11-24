import os
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_clean_sequence(output_dir: Path, num_frames: int = 90, width: int = 640, height: int = 360) -> list[np.ndarray]:
    """Generates a clean synthetic scene with stationary background detail and moving foreground objects."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    bg = 110.0 + 35.0 * np.sin(x_grid * 0.03) * np.cos(y_grid * 0.03)

    bg[::24, :] = 65.0
    bg[:, ::48] = 75.0
    bg = cv2.GaussianBlur(bg, (3, 3), 0.6)

    bg_u8 = np.clip(bg, 0, 255).astype(np.uint8)
    cv2.rectangle(bg_u8, (60, 140), (190, 320), 90, -1)
    cv2.rectangle(bg_u8, (85, 165), (115, 205), 175, -1)
    cv2.rectangle(bg_u8, (135, 165), (165, 205), 175, -1)
    cv2.rectangle(bg_u8, (430, 110), (570, 320), 80, -1)
    cv2.line(bg_u8, (0, 320), (width, 320), 45, 3)

    for idx in range(num_frames):
        frame = bg_u8.copy()

        # Moving Object 1: Rolling cart
        car_x = int(-80 + (width + 160) * (idx / max(1, num_frames - 1)))
        car_y = 275
        cv2.rectangle(frame, (car_x, car_y), (car_x + 75, car_y + 35), 210, -1)
        cv2.rectangle(frame, (car_x + 12, car_y - 18), (car_x + 58, car_y), 185, -1)
        cv2.circle(frame, (car_x + 18, car_y + 38), 9, 30, -1)
        cv2.circle(frame, (car_x + 57, car_y + 38), 9, 30, -1)

        # Moving Object 2: Swinging pendulum
        angle = 0.45 * np.sin(idx * 0.18)
        pivot = (310, 60)
        bob_x = int(pivot[0] + 110 * np.sin(angle))
        bob_y = int(pivot[1] + 110 * np.cos(angle))
        cv2.line(frame, pivot, (bob_x, bob_y), 40, 2)
        cv2.circle(frame, (bob_x, bob_y), 16, 230, -1)
        cv2.circle(frame, (bob_x, bob_y), 7, 50, -1)

        filename = output_dir / f"frame_{idx:04d}.png"
        cv2.imwrite(str(filename), frame)
        frames.append(frame)

    return frames


def degrade_sequence(clean_frames: list[np.ndarray], output_dir: Path, seed: int = 42) -> list[np.ndarray]:
    """Applies realistic historical film artifacts: jitter, flicker, dust blotches, and scratches."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    degraded_frames = []

    height, width = clean_frames[0].shape
    y_norm, x_norm = np.mgrid[0:height, 0:width].astype(np.float32)
    x_norm = (x_norm - width / 2.0) / (width / 2.0)
    y_norm = (y_norm - height / 2.0) / (height / 2.0)
    vignette_base = 1.0 - 0.18 * (x_norm**2 + y_norm**2)

    scratch_positions = [width * 0.28, width * 0.71]

    for idx, clean in enumerate(clean_frames):
        img = clean.astype(np.float32)

        # 1. Global & spatial luminance flicker
        global_gain = 1.0 + 0.18 * np.sin(idx * 0.75) + rng.normal(0.0, 0.08)
        global_offset = rng.normal(0.0, 10.0)
        flicker_gradient = 1.0 + rng.normal(0.0, 0.06) * x_norm
        img = img * global_gain * vignette_base * flicker_gradient + global_offset

        # 2. Dust & dirt blotches
        num_dust = rng.integers(12, 28)
        for _ in range(num_dust):
            cx = int(rng.integers(10, width - 10))
            cy = int(rng.integers(10, height - 10))
            rx = int(rng.integers(2, 7))
            ry = int(rng.integers(1, 5))
            angle = float(rng.uniform(0, 180))
            intensity = float(rng.choice([rng.uniform(5, 35), rng.uniform(225, 255)], p=[0.75, 0.25]))
            cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, intensity, -1)

        # 3. Vertical scratches
        for s_idx in range(len(scratch_positions)):
            scratch_positions[s_idx] += rng.normal(0.0, 0.8)
            sx = int(np.clip(scratch_positions[s_idx], 5, width - 5))
            if rng.random() < 0.85:
                scratch_val = 235.0 if s_idx == 0 else 25.0
                thickness = int(rng.choice([1, 2]))
                y_start = int(rng.integers(0, height // 4))
                y_end = int(rng.integers(3 * height // 4, height))
                cv2.line(img, (sx, y_start), (sx, y_end), scratch_val, thickness)

        # 4. Subtle film grain
        img += rng.normal(0.0, 4.5, size=img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)

        # 5. Frame jitter
        dx = float(rng.normal(0.0, 3.2))
        dy = float(rng.normal(0.0, 2.5))
        d_angle = float(rng.normal(0.0, 0.35))
        rot_mat = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), d_angle, 1.0)
        rot_mat[0, 2] += dx
        rot_mat[1, 2] += dy
        jittered = cv2.warpAffine(
            img,
            rot_mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        filename = output_dir / f"frame_{idx:04d}.png"
        cv2.imwrite(str(filename), jittered)
        degraded_frames.append(jittered)

    return degraded_frames


def main():
    clean_dir = PROJECT_ROOT / "data" / "clean"
    degraded_dir = PROJECT_ROOT / "data" / "degraded"

    print(f"Generating clean reference sequence in '{clean_dir}'...")
    clean_frames = generate_clean_sequence(clean_dir, num_frames=90)
    print(f"Saved {len(clean_frames)} clean frames.")

    print(f"Applying synthetic film degradation in '{degraded_dir}'...")
    degraded_frames = degrade_sequence(clean_frames, degraded_dir)
    print(f"Saved {len(degraded_frames)} degraded frames.")


if __name__ == "__main__":
    main()