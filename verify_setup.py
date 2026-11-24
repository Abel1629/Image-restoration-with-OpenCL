import pyopencl as cl
import numpy as np
import cv2


def check_environment():
    print(f"OpenCV version: {cv2.__version__}")
    print(f"PyOpenCL version: {cl.VERSION_TEXT}\n")

    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("No OpenCL platforms found. Check your NVIDIA/Intel GPU drivers.")

    for p_idx, platform in enumerate(platforms):
        print(f"Platform [{p_idx}]: {platform.name} ({platform.vendor})")
        for d_idx, device in enumerate(platform.get_devices()):
            dev_type = cl.device_type.to_string(device.type)
            mem_mb = device.global_mem_size // (1024 * 1024)
            print(f"  Device [{d_idx}]: {device.name.strip()} | Type: {dev_type} | VRAM: {mem_mb} MB")

    # Quick kernel execution test on the first GPU device found (preferring NVIDIA)
    selected_device = None
    for platform in platforms:
        for device in platform.get_devices(device_type=cl.device_type.GPU):
            if "NVIDIA" in platform.name.upper() or "NVIDIA" in device.name.upper():
                selected_device = device
                break
        if selected_device:
            break

    if selected_device is None:
        selected_device = platforms[0].get_devices()[0]

    print(f"\nRunning smoke test on: {selected_device.name.strip()}...")
    ctx = cl.Context([selected_device])
    queue = cl.CommandQueue(ctx)

    a = np.arange(1024, dtype=np.float32)
    b = np.ones(1024, dtype=np.float32) * 2.0
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)

    prg = cl.Program(ctx, """
        __kernel void add_vec(__global const float *a,
                              __global const float *b,
                              __global float *out) {
            int gid = get_global_id(0);
            out[gid] = a[gid] + b[gid];
        }
    """).build()

    prg.add_vec(queue, a.shape, None, a_buf, b_buf, out_buf)
    result = np.empty_like(a)
    cl.enqueue_copy(queue, result, out_buf)

    assert np.allclose(result, a + b), "OpenCL kernel result mismatch!"
    print("Success! OpenCL kernel executed and verified.")


if __name__ == "__main__":
    check_environment()