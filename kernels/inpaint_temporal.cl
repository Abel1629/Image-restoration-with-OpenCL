inline float sample_bilinear(__global const uchar* img, float fx, float fy, int width, int height) {
    fx = clamp(fx, 0.0f, (float)(width - 1));
    fy = clamp(fy, 0.0f, (float)(height - 1));

    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);

    float wx = fx - (float)x0;
    float wy = fy - (float)y0;

    float p00 = (float)img[y0 * width + x0];
    float p10 = (float)img[y0 * width + x1];
    float p01 = (float)img[y1 * width + x0];
    float p11 = (float)img[y1 * width + x1];

    return mix(mix(p00, p10, wx), mix(p01, p11, wx), wy);
}

inline float patch_sad(
    __global const uchar* img_a, int ax, int ay,
    __global const uchar* img_b, int bx, int by,
    int width, int height, int patch_radius
) {
    float sad = 0.0f;
    for (int py = -patch_radius; py <= patch_radius; py++) {
        int ya = clamp(ay + py, 0, height - 1);
        int yb = clamp(by + py, 0, height - 1);
        for (int px = -patch_radius; px <= patch_radius; px++) {
            int xa = clamp(ax + px, 0, width - 1);
            int xb = clamp(bx + px, 0, width - 1);
            sad += fabs((float)img_a[ya * width + xa] - (float)img_b[yb * width + xb]);
        }
    }
    return sad;
}

__kernel void estimate_block_motion(
    __global const uchar* curr,
    __global const uchar* ref,
    __global short2* flow_out,
    const int width,
    const int height,
    const int search_radius,
    const int patch_radius
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float zero_cost = patch_sad(curr, x, y, ref, x, y, width, height, patch_radius);
    int best_dx = 0;
    int best_dy = 0;
    float best_cost = zero_cost * 0.72f;

    for (int dy = -search_radius; dy <= search_radius; dy += 2) {
        int ry = y + dy;
        if (ry < patch_radius || ry >= height - patch_radius) continue;
        for (int dx = -search_radius; dx <= search_radius; dx += 2) {
            if (dx == 0 && dy == 0) continue;
            int rx = x + dx;
            if (rx < patch_radius || rx >= width - patch_radius) continue;

            float sad = patch_sad(curr, x, y, ref, rx, ry, width, height, patch_radius)
                      + 0.40f * (float)(abs(dx) + abs(dy));
            if (sad < best_cost) {
                best_cost = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    if (best_dx != 0 || best_dy != 0) {
        int c_dx = best_dx;
        int c_dy = best_dy;
        for (int dy = c_dy - 1; dy <= c_dy + 1; dy++) {
            int ry = y + dy;
            if (ry < patch_radius || ry >= height - patch_radius) continue;
            for (int dx = c_dx - 1; dx <= c_dx + 1; dx++) {
                int rx = x + dx;
                if (rx < patch_radius || rx >= width - patch_radius) continue;

                float sad = patch_sad(curr, x, y, ref, rx, ry, width, height, patch_radius)
                          + 0.40f * (float)(abs(dx) + abs(dy));
                if (sad < best_cost) {
                    best_cost = sad;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }
    }

    flow_out[y * width + x] = (short2)((short)best_dx, (short)best_dy);
}

__kernel void estimate_symmetric_motion(
    __global const uchar* prev_img,
    __global const uchar* next_img,
    __global short2* sym_flow_out,
    const int width,
    const int height,
    const int search_radius,
    const int patch_radius
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float zero_cost = patch_sad(prev_img, x, y, next_img, x, y, width, height, patch_radius);
    int best_dx = 0;
    int best_dy = 0;
    float best_cost = zero_cost * 0.72f;

    for (int dy = -search_radius; dy <= search_radius; dy += 2) {
        if (y - dy < patch_radius || y - dy >= height - patch_radius) continue;
        if (y + dy < patch_radius || y + dy >= height - patch_radius) continue;
        for (int dx = -search_radius; dx <= search_radius; dx += 2) {
            if (dx == 0 && dy == 0) continue;
            if (x - dx < patch_radius || x - dx >= width - patch_radius) continue;
            if (x + dx < patch_radius || x + dx >= width - patch_radius) continue;

            float sad = patch_sad(prev_img, x - dx, y - dy, next_img, x + dx, y + dy, width, height, patch_radius)
                      + 0.40f * (float)(abs(dx) + abs(dy));
            if (sad < best_cost) {
                best_cost = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    if (best_dx != 0 || best_dy != 0) {
        int c_dx = best_dx;
        int c_dy = best_dy;
        for (int dy = c_dy - 1; dy <= c_dy + 1; dy++) {
            if (y - dy < patch_radius || y - dy >= height - patch_radius) continue;
            if (y + dy < patch_radius || y + dy >= height - patch_radius) continue;
            for (int dx = c_dx - 1; dx <= c_dx + 1; dx++) {
                if (x - dx < patch_radius || x - dx >= width - patch_radius) continue;
                if (x + dx < patch_radius || x + dx >= width - patch_radius) continue;

                float sad = patch_sad(prev_img, x - dx, y - dy, next_img, x + dx, y + dy, width, height, patch_radius)
                          + 0.40f * (float)(abs(dx) + abs(dy));
                if (sad < best_cost) {
                    best_cost = sad;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }
    }

    sym_flow_out[y * width + x] = (short2)((short)best_dx, (short)best_dy);
}

__kernel void detect_temporal_defects(
    __global const uchar* prev_img,
    __global const uchar* curr_img,
    __global const uchar* next_img,
    __global const short2* flow_prev,
    __global const short2* flow_next,
    __global const short2* flow_sym,
    __global uchar* temporal_ref,
    __global uchar* defect_mask,
    const int width,
    const int height,
    const float dust_thresh,
    const float neighbor_agree_thresh
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float c_val = (float)curr_img[idx];

    short2 fp = flow_prev[idx];
    short2 fn = flow_next[idx];
    float p_dir = sample_bilinear(prev_img, (float)(x + fp.x), (float)(y + fp.y), width, height);
    float n_dir = sample_bilinear(next_img, (float)(x + fn.x), (float)(y + fn.y), width, height);

    short2 fs = flow_sym[idx];
    float p_sym = sample_bilinear(prev_img, (float)(x - fs.x), (float)(y - fs.y), width, height);
    float n_sym = sample_bilinear(next_img, (float)(x + fs.x), (float)(y + fs.y), width, height);

    float diff_dir = fabs(p_dir - n_dir);
    float diff_sym = fabs(p_sym - n_sym);

    float p_best = (diff_sym < diff_dir) ? p_sym : p_dir;
    float n_best = (diff_sym < diff_dir) ? n_sym : n_dir;
    float pair_diff = fmin(diff_dir, diff_sym);
    float t_avg = 0.5f * (p_best + n_best);

    temporal_ref[idx] = (uchar)clamp(round(t_avg), 0.0f, 255.0f);

    float dev_p = c_val - p_best;
    float dev_n = c_val - n_best;
    bool same_sign = (dev_p * dev_n) > 0.0f;
    float min_dev = fmin(fabs(dev_p), fabs(dev_n));

    if (pair_diff < neighbor_agree_thresh && same_sign && min_dev > dust_thresh) {
        defect_mask[idx] = 255;
    } else {
        defect_mask[idx] = 0;
    }
}

__kernel void reconstruct_frame(
    __global const uchar* curr_img,
    __global const uchar* temporal_ref,
    __global const uchar* defect_mask,
    __global uchar* dst_img,
    const int width,
    const int height,
    const float scratch_thresh
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    bool is_dust_3x3 = false;
    for (int dy = -1; dy <= 1; dy++) {
        int ny = clamp(y + dy, 0, height - 1);
        for (int dx = -1; dx <= 1; dx++) {
            int nx = clamp(x + dx, 0, width - 1);
            if (defect_mask[ny * width + nx] > 0) {
                is_dust_3x3 = true;
            }
        }
    }

    if (is_dust_3x3) {
        int blob_area = 0;
        for (int dy = -8; dy <= 8; dy++) {
            int ny = clamp(y + dy, 0, height - 1);
            for (int dx = -8; dx <= 8; dx++) {
                int nx = clamp(x + dx, 0, width - 1);
                if (defect_mask[ny * width + nx] > 0) {
                    blob_area++;
                }
            }
        }

        if (blob_area <= 115) {
            dst_img[idx] = temporal_ref[idx];
            return;
        }
    }

    int consistent_rows = 0;
    float contrast_sum = 0.0f;

    for (int ky = -8; ky <= 8; ky += 2) {
        int sy = clamp(y + ky, 0, height - 1);
        float c_y = (float)curr_img[sy * width + x];
        float l3 = (float)curr_img[sy * width + max(x - 3, 0)];
        float r3 = (float)curr_img[sy * width + min(x + 3, width - 1)];
        float flank = 0.5f * (l3 + r3);
        float diff = c_y - flank;

        if (fabs(diff) > scratch_thresh && fabs(l3 - r3) < 20.0f) {
            consistent_rows++;
            contrast_sum += diff;
        }
    }

    float l3_now = (float)curr_img[y * width + max(x - 3, 0)];
    float r3_now = (float)curr_img[y * width + min(x + 3, width - 1)];
    float c_now = (float)curr_img[idx];
    float diff_now = c_now - 0.5f * (l3_now + r3_now);

    if (consistent_rows >= 7 && (diff_now * contrast_sum) > 0.0f && fabs(diff_now) > scratch_thresh * 0.75f) {
        float spatial_fill = 0.5f * (l3_now + r3_now);
        float t_val = (float)temporal_ref[idx];
        if (fabs(t_val - spatial_fill) < 16.0f) {
            dst_img[idx] = (uchar)clamp(round(0.5f * (t_val + spatial_fill)), 0.0f, 255.0f);
        } else {
            dst_img[idx] = (uchar)clamp(round(spatial_fill), 0.0f, 255.0f);
        }
        return;
    }

    dst_img[idx] = curr_img[idx];
}