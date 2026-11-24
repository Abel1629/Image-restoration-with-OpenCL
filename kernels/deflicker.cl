// Kernel 1: Compute per-tile sum and sum-of-squares using local memory reduction
__kernel void compute_tile_stats(
    __global const uchar* img,
    __global float* tile_means,
    __global float* tile_stds,
    const int width,
    const int height,
    const int tile_size,
    const int grid_w,
    const int grid_h
) {
    int tx = get_group_id(0);
    int ty = get_group_id(1);
    if (tx >= grid_w || ty >= grid_h) return;

    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int tid = ly * get_local_size(0) + lx;
    int group_threads = get_local_size(0) * get_local_size(1);

    int start_x = tx * tile_size;
    int start_y = ty * tile_size;
    int end_x = min(start_x + tile_size, width);
    int end_y = min(start_y + tile_size, height);

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    int local_count = 0;

    for (int y = start_y + ly; y < end_y; y += get_local_size(1)) {
        for (int x = start_x + lx; x < end_x; x += get_local_size(0)) {
            float val = (float)img[y * width + x];
            local_sum += val;
            local_sq_sum += val * val;
            local_count++;
        }
    }

    __local float s_sum[256];
    __local float s_sq[256];
    __local int s_cnt[256];

    s_sum[tid] = local_sum;
    s_sq[tid] = local_sq_sum;
    s_cnt[tid] = local_count;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq[tid] += s_sq[tid + stride];
            s_cnt[tid] += s_cnt[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        int tile_idx = ty * grid_w + tx;
        float n = max((float)s_cnt[0], 1.0f);
        float mean = s_sum[0] / n;
        float var = max((s_sq[0] / n) - (mean * mean), 0.0f);
        tile_means[tile_idx] = mean;
        tile_stds[tile_idx] = sqrt(var);
    }
}

// Kernel 2: Apply spatially smooth bilinear gain and offset correction per pixel
__kernel void apply_deflicker(
    __global const uchar* src,
    __global uchar* dst,
    __global const float* gain_grid,
    __global const float* offset_grid,
    const int width,
    const int height,
    const int tile_size,
    const int grid_w,
    const int grid_h
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    // Map pixel coordinate to continuous tile-center grid coordinates
    float gx = ((float)x + 0.5f) / (float)tile_size - 0.5f;
    float gy = ((float)y + 0.5f) / (float)tile_size - 0.5f;

    gx = clamp(gx, 0.0f, (float)(grid_w - 1));
    gy = clamp(gy, 0.0f, (float)(grid_h - 1));

    int x0 = (int)floor(gx);
    int y0 = (int)floor(gy);
    int x1 = min(x0 + 1, grid_w - 1);
    int y1 = min(y0 + 1, grid_h - 1);

    float fx = gx - (float)x0;
    float fy = gy - (float)y0;

    int idx00 = y0 * grid_w + x0;
    int idx10 = y0 * grid_w + x1;
    int idx01 = y1 * grid_w + x0;
    int idx11 = y1 * grid_w + x1;

    float g = mix(mix(gain_grid[idx00], gain_grid[idx10], fx),
                  mix(gain_grid[idx01], gain_grid[idx11], fx), fy);
    float o = mix(mix(offset_grid[idx00], offset_grid[idx10], fx),
                  mix(offset_grid[idx01], offset_grid[idx11], fx), fy);

    float val = (float)src[y * width + x] * g + o;
    dst[y * width + x] = (uchar)clamp(round(val), 0.0f, 255.0f);
}