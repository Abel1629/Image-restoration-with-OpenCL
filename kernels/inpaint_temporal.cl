// Kernel 1: Identify true moving-object regions using |prev - next|
// Single-frame dust blotches are absent in both prev and next, so |prev - next| is ~0 on dust
// and high ONLY where genuine objects (car, wheels, pendulum) are moving!
__kernel void compute_motion_mask(
    __global const uchar* prev_img,
    __global const uchar* next_img,
    __global uchar* motion_mask,
    const int width,
    const int height,
    const float motion_thresh
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int diff_count = 0;
    for (int dy = -3; dy <= 3; dy++) {
        int ny = clamp(y + dy, 0, height - 1);
        for (int dx = -3; dx <= 3; dx++) {
            int nx = clamp(x + dx, 0, width - 1);
            int n_idx = ny * width + nx;
            if (fabs((float)prev_img[n_idx] - (float)next_img[n_idx]) > motion_thresh) {
                diff_count++;
            }
        }
    }

    // Require at least 4 differing pixels in 7x7 so isolated random noise spikes don't trigger motion protection
    motion_mask[y * width + x] = (diff_count >= 4) ? 255 : 0;
}

// Kernel 2: Remove ALL dust blotches (outside motion_mask) + ALL vertical roller scratches
__kernel void inpaint_defects(
    __global const uchar* prev_img,
    __global const uchar* curr_img,
    __global const uchar* next_img,
    __global const uchar* motion_mask,
    __global uchar* dst_img,
    const int width,
    const int height,
    const float dust_thresh,
    const float scratch_thresh
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float c = (float)curr_img[idx];
    float p = (float)prev_img[idx];
    float n = (float)next_img[idx];
    float t_avg = 0.5f * (p + n);

    // 1. Vertical roller scratch detection (21px vertical span, checks x-1, x, x+1 for 2px scratches)
    bool is_scratch = false;
    for (int offset = -1; offset <= 1; offset++) {
        int cx = x + offset;
        if (cx < 4 || cx >= width - 4) continue;

        int consistent_rows = 0;
        float contrast_sum = 0.0f;
        for (int ky = -10; ky <= 10; ky += 2) {
            int sy = clamp(y + ky, 0, height - 1);
            float cy = (float)curr_img[sy * width + cx];
            float l4 = (float)curr_img[sy * width + (cx - 4)];
            float r4 = (float)curr_img[sy * width + (cx + 4)];
            float diff = cy - 0.5f * (l4 + r4);
            if (fabs(diff) > scratch_thresh && fabs(l4 - r4) < 22.0f) {
                consistent_rows++;
                contrast_sum += diff;
            }
        }

        float l4_now = (float)curr_img[y * width + (cx - 4)];
        float r4_now = (float)curr_img[y * width + (cx + 4)];
        float c_col = (float)curr_img[y * width + cx];
        float diff_now = c_col - 0.5f * (l4_now + r4_now);

        if (consistent_rows >= 8 && (diff_now * contrast_sum) > 0.0f && fabs(diff_now) > scratch_thresh * 0.65f) {
            is_scratch = true;
        }
    }

    if (is_scratch) {
        float l4_here = (float)curr_img[y * width + max(x - 4, 0)];
        float r4_here = (float)curr_img[y * width + min(x + 4, width - 1)];
        float spatial_fill = 0.5f * (l4_here + r4_here);
        if (motion_mask[idx] == 0 && fabs(t_avg - spatial_fill) < 16.0f) {
            dst_img[idx] = (uchar)clamp(round(0.5f * (t_avg + spatial_fill)), 0.0f, 255.0f);
        } else {
            dst_img[idx] = (uchar)clamp(round(spatial_fill), 0.0f, 255.0f);
        }
        return;
    }

    // 2. Dust blotch removal (including 1px dilated rim around blotches)
    // Protected wherever motion_mask > 0 so car, wheels, and pendulum are NEVER touched
    if (motion_mask[idx] == 0) {
        bool near_dust = false;
        for (int dy = -1; dy <= 1; dy++) {
            int ny = clamp(y + dy, 0, height - 1);
            for (int dx = -1; dx <= 1; dx++) {
                int nx = clamp(x + dx, 0, width - 1);
                int n_idx = ny * width + nx;
                if (motion_mask[n_idx] == 0) {
                    float nc = (float)curr_img[n_idx];
                    float np = (float)prev_img[n_idx];
                    float nn = (float)next_img[n_idx];
                    if (fabs(np - nn) < 18.0f && ((nc - np) * (nc - nn) > 0.0f) &&
                        fmin(fabs(nc - np), fabs(nc - nn)) > dust_thresh) {
                        near_dust = true;
                    }
                }
            }
        }

        if (near_dust && fabs(p - n) < 20.0f) {
            dst_img[idx] = (uchar)clamp(round(t_avg), 0.0f, 255.0f);
            return;
        }
    }

    dst_img[idx] = curr_img[idx];
}

// Kernel 3: Spatio-Temporal Bilateral Denoiser
__kernel void spatiotemporal_bilateral(
    __global const uchar* prev_img,
    __global const uchar* curr_img,
    __global const uchar* next_img,
    __global uchar* dst_img,
    const int width,
    const int height,
    const float sigma_spatial,
    const float sigma_range
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float center = (float)curr_img[idx];
    float inv_2s2 = -0.5f / (sigma_spatial * sigma_spatial);
    float inv_2r2 = -0.5f / (sigma_range * sigma_range);

    float sum_w = 0.0f;
    float sum_v = 0.0f;

    // 1. Spatial 5x5 bilateral support on current frame
    for (int dy = -2; dy <= 2; dy++) {
        int ny = clamp(y + dy, 0, height - 1);
        for (int dx = -2; dx <= 2; dx++) {
            int nx = clamp(x + dx, 0, width - 1);
            float nb = (float)curr_img[ny * width + nx];
            float d2 = (float)(dx * dx + dy * dy);
            float r = nb - center;
            float w = exp(d2 * inv_2s2 + (r * r) * inv_2r2);
            sum_w += w;
            sum_v += w * nb;
        }
    }

    // 2. Temporal 3x3 bilateral support from prev & next frames
    for (int dy = -1; dy <= 1; dy++) {
        int ny = clamp(y + dy, 0, height - 1);
        for (int dx = -1; dx <= 1; dx++) {
            int nx = clamp(x + dx, 0, width - 1);
            int n_idx = ny * width + nx;
            float d2 = (float)(dx * dx + dy * dy) + 1.0f;

            float vp = (float)prev_img[n_idx];
            float rp = vp - center;
            float wp = 1.6f * exp(d2 * inv_2s2 + (rp * rp) * inv_2r2);
            sum_w += wp;
            sum_v += wp * vp;

            float vn = (float)next_img[n_idx];
            float rn = vn - center;
            float wn = 1.6f * exp(d2 * inv_2s2 + (rn * rn) * inv_2r2);
            sum_w += wn;
            sum_v += wn * vn;
        }
    }

    dst_img[idx] = (uchar)clamp(round(sum_v / max(sum_w, 1e-6f)), 0.0f, 255.0f);
}