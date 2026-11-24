from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_sequence(folder: Path) -> list[np.ndarray]:
    """Loads sorted PNG frames from a directory as uint8 grayscale images."""
    paths = sorted(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in '{folder}'.")
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    return frames


class SynchronisedPreviewViewer:
    """Lightweight OpenCV window for synchronised before-and-after film comparison."""

    def __init__(
        self,
        before_frames: list[np.ndarray],
        after_frames: list[np.ndarray],
        window_name: str = "Historical Film Restoration - Before vs After",
        fps: float = 24.0,
    ):
        if len(before_frames) != len(after_frames):
            raise ValueError("Before and after sequences must have the same frame count.")

        self.before = before_frames
        self.after = after_frames
        self.num_frames = len(before_frames)
        self.height, self.width = before_frames[0].shape[:2]

        self.window_name = window_name
        self.delay_ms = max(1, int(round(1000.0 / fps)))
        self.current_idx = 0
        self.split_ratio = 0.5
        self.mode = "split"  # "split" or "side_by_side"
        self.paused = False

    def _on_frame_trackbar(self, val: int) -> None:
        self.current_idx = int(np.clip(val, 0, self.num_frames - 1))

    def _on_split_trackbar(self, val: int) -> None:
        self.split_ratio = float(np.clip(val / 100.0, 0.0, 1.0))

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if self.mode == "split" and (event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON)):
            self.split_ratio = float(np.clip(x / max(1, self.width), 0.0, 1.0))
            cv2.setTrackbarPos("Split %", self.window_name, int(round(self.split_ratio * 100)))

    def _render_frame(self) -> np.ndarray:
        bef = cv2.cvtColor(self.before[self.current_idx], cv2.COLOR_GRAY2BGR)
        aft = cv2.cvtColor(self.after[self.current_idx], cv2.COLOR_GRAY2BGR)

        if self.mode == "split":
            canvas = aft.copy()
            split_x = int(round(self.width * self.split_ratio))
            canvas[:, :split_x] = bef[:, :split_x]

            # Draw divider line & labels
            cv2.line(canvas, (split_x, 0), (split_x, self.height - 1), (0, 220, 255), 2)
            cv2.putText(canvas, "BEFORE (Degraded)", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            cv2.putText(canvas, "AFTER (Restored)", (self.width - 185, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 120), 2)
        else:
            cv2.putText(bef, "BEFORE (Degraded)", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            cv2.putText(aft, "AFTER (Restored)", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 120), 2)
            canvas = np.hstack([bef, aft])

        # Status bar overlay at bottom
        status = f"Frame: {self.current_idx + 1}/{self.num_frames} | Mode: {self.mode} [M] | {'PAUSED' if self.paused else 'PLAY'} [Space] | Step [A/D] | Quit [Q]"
        cv2.rectangle(canvas, (0, self.height - 26), (canvas.shape[1], self.height), (20, 20, 20), -1)
        cv2.putText(canvas, status, (10, self.height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        return canvas

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Frame", self.window_name, 0, self.num_frames - 1, self._on_frame_trackbar)
        cv2.createTrackbar("Split %", self.window_name, 50, 100, self._on_split_trackbar)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        while True:
            display = self._render_frame()
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(self.delay_ms if not self.paused else 30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q or Esc
                break
            elif key == ord(" "):
                self.paused = not self.paused
            elif key in (ord("m"), ord("M")):
                self.mode = "side_by_side" if self.mode == "split" else "split"
            elif key in (ord("a"), ord("A")):
                self.paused = True
                self.current_idx = (self.current_idx - 1) % self.num_frames
                cv2.setTrackbarPos("Frame", self.window_name, self.current_idx)
            elif key in (ord("d"), ord("D")):
                self.paused = True
                self.current_idx = (self.current_idx + 1) % self.num_frames
                cv2.setTrackbarPos("Frame", self.window_name, self.current_idx)

            if not self.paused:
                self.current_idx = (self.current_idx + 1) % self.num_frames
                cv2.setTrackbarPos("Frame", self.window_name, self.current_idx)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Smoke-test the viewer by comparing data/degraded against data/clean
    degraded_seq = load_sequence(PROJECT_ROOT / "data" / "degraded")
    clean_seq = load_sequence(PROJECT_ROOT / "data" / "clean")
    viewer = SynchronisedPreviewViewer(degraded_seq, clean_seq)
    viewer.run()