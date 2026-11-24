__kernel void warp_affine_bilinear(
    __global const uchar* src,
    __global uchar* dst,
    const int width,
    const int height,
    const float m00, const float m01, const float m02,
    const float m10, const float m11, const float m12
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) {
        return;
    }

    // Inverse mapping from destination pixel (x, y) to source coordinates (sx, sy)
    float sx = m00 * (float)x + m01 * (float)y + m02;
    float sy = m10 * (float)x + m11 * (float)y + m12;

    // Reflect-101 boundary handling to avoid black borders on small shifts
    if (sx < 0.0f) sx = -sx;
    if (sy < 0.0f) sy = -sy;
    if (sx > (float)(width - 1))  sx = 2.0f * (float)(width - 1) - sx;
    if (sy > (float)(height - 1)) sy = 2.0f * (float)(height - 1) - sy;

    sx = clamp(sx, 0.0f, (float)(width - 1));
    sy = clamp(sy, 0.0f, (float)(height - 1));

    int x0 = (int)floor(sx);
    int y0 = (int)floor(sy);
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);

    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    float p00 = (float)src[y0 * width + x0];
    float p10 = (float)src[y0 * width + x1];
    float p01 = (float)src[y1 * width + x0];
    float p11 = (float)src[y1 * width + x1];

    float top = mix(p00, p10, fx);
    float bot = mix(p01, p11, fx);
    float val = mix(top, bot, fy);

    dst[y * width + x] = (uchar)clamp(round(val), 0.0f, 255.0f);
}