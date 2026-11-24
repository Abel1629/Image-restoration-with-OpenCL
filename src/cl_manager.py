import os
import warnings
from pathlib import Path
import pyopencl as cl

# Suppress harmless PTX compiler info warnings from NVIDIA drivers
warnings.filterwarnings("ignore", category=cl.CompilerWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = PROJECT_ROOT / "kernels"


class OpenCLManager:
    """Manages OpenCL device selection, context, profiling command queue, and kernel compilation."""

    def __init__(self, prefer_vendor: str = "NVIDIA", enable_profiling: bool = True):
        self.device = self._select_device(prefer_vendor)
        self.ctx = cl.Context([self.device])

        queue_props = cl.command_queue_properties.PROFILING_ENABLE if enable_profiling else 0
        self.queue = cl.CommandQueue(self.ctx, properties=queue_props)

    @staticmethod
    def _select_device(prefer_vendor: str) -> cl.Device:
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms available.")

        prefer_upper = prefer_vendor.upper()
        fallback_gpu = None

        for platform in platforms:
            for device in platform.get_devices(device_type=cl.device_type.GPU):
                if fallback_gpu is None:
                    fallback_gpu = device
                if prefer_upper in platform.name.upper() or prefer_upper in device.name.upper():
                    return device

        if fallback_gpu is not None:
            return fallback_gpu

        return platforms[0].get_devices()[0]

    def load_program(self, kernel_filename: str, build_options: list[str] | None = None) -> cl.Program:
        """Reads a .cl file from the kernels/ directory and compiles it for the active device."""
        kernel_path = KERNELS_DIR / kernel_filename
        if not kernel_path.exists():
            raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

        source_code = kernel_path.read_text(encoding="utf-8")
        options = build_options if build_options is not None else ["-cl-mad-enable"]
        return cl.Program(self.ctx, source_code).build(options=options)

    def info(self) -> str:
        mem_mb = self.device.global_mem_size // (1024 * 1024)
        return f"{self.device.name.strip()} ({mem_mb} MB VRAM, {self.device.max_compute_units} CUs)"


if __name__ == "__main__":
    mgr = OpenCLManager(prefer_vendor="NVIDIA")
    print("Initialized OpenCLManager on:", mgr.info())