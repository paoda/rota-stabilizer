# version 330 core

layout(location = 0) out vec4 angle;

uniform sampler2D u_y_tex;
uniform sampler2D u_uv_tex;
uniform vec2 u_resolution;

// TODO: select colorspace based on AVFrame
vec3 nv12ToRgb(float y_norm, vec2 uv_norm) {
    const mat3 bt601 = mat3(
            1.0, 1.0, 1.0,
            0.0, -0.39465, 2.03211,
            1.13983, -0.58060, 0.0
        );

    const mat3 bt709 = mat3(
            1.0, 1.0, 1.0,
            0.0, -0.1873, 1.8556,
            1.5748, -0.4681, 0.0
        );

    const float offset = 16.0 / 255.0;
    float y = (y_norm - offset) * (255.0 / (235.0 - 16.0));
    vec2 uv = (uv_norm - offset) * (255.0 / (240.0 - 16.0));

    y = clamp(y, 0.0, 1.0);
    uv = clamp(uv, 0.0, 1.0);

    float u = uv.r - 0.5;
    float v = uv.g - 0.5;

    return clamp(bt601 * vec3(y, u, v), 0.0, 1.0);
}

vec3 sampleTexture(int size, ivec2 start_pos) {
    float y_sum = 0.0;
    vec2 uv_sum = vec2(0.0);

    for (int dy = 0; dy < size; dy++) {
        for (int dx = 0; dx < size; dx++) {
            ivec2 tex_pos = start_pos + ivec2(dx, dy);
            // NB: y and uv buffer are not the same size per the image format

            y_sum += texelFetch(u_y_tex, tex_pos, 0).r;
            uv_sum += texelFetch(u_uv_tex, tex_pos / 2, 0).rg;
        }
    }

    float count = float(size * size);
    return nv12ToRgb(y_sum / count, uv_sum / count);
}

void main() {
    const int ofs = 1;
    const int box = 3;
    const float threshold = 0.5;

    ivec2 resolution = textureSize(u_y_tex, 0);
    int W = resolution.x;
    int H = resolution.y;

    vec3 btm_left = sampleTexture(box, ivec2(ofs, H - ofs - box));
    vec3 top_left = sampleTexture(box, ivec2(ofs, ofs));
    vec3 btm_right = sampleTexture(box, ivec2(W - ofs - box, H - ofs - box));
    vec3 top_right = sampleTexture(box, ivec2(W - ofs - box, ofs));

    float value = 0.0;
    value += dot(step(threshold, top_left), vec3(2048.0, 1024.0, 512.0));
    value += dot(step(threshold, top_right), vec3(256.0, 128.0, 64.0));
    value += dot(step(threshold, btm_left), vec3(32.0, 16.0, 8.0));
    value += dot(step(threshold, btm_right), vec3(4.0, 2.0, 1.0));

    float deg = (360.0 * value / 4096.0);
    angle = vec4(radians(-deg), 0.0, 0.0, 1.0);
}
