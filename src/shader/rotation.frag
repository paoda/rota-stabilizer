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

vec3 sampleTexture(int size, vec2 start_pos) {
    float y_sum = 0.0;
    vec2 uv_sum = vec2(0.0);

    for (int dy = 0; dy < size; dy++) {
        for (int dx = 0; dx < size; dx++) {
            vec2 tex_pos = start_pos + vec2(dx, dy);
            vec2 pos_norm = (tex_pos + 0.5) / u_resolution;

            y_sum += texture(u_y_tex, pos_norm).r;
            uv_sum += texture(u_uv_tex, pos_norm).rg;
        }
    }

    float count = float(size * size);
    return nv12ToRgb(y_sum / count, uv_sum / count);
}

void main() {
    const int ofs = 5;
    const int size = 3;
    const float threshold = 0.5;

    // Sample regions at the same positions as the CPU code
    vec3 btm_left = sampleTexture(size, vec2(ofs, u_resolution.y - ofs - size));
    vec3 top_left = sampleTexture(size, vec2(ofs, ofs));
    vec3 btm_right = sampleTexture(size, vec2(u_resolution.x - ofs - size, u_resolution.y - ofs - size));
    vec3 top_right = sampleTexture(size, vec2(u_resolution.x - ofs - size, ofs));

    uint value = 0u;
    value |= uint(top_left.r >= threshold) << 11;
    value |= uint(top_left.g >= threshold) << 10;
    value |= uint(top_left.b >= threshold) << 9;
    value |= uint(top_right.r >= threshold) << 8;
    value |= uint(top_right.g >= threshold) << 7;
    value |= uint(top_right.b >= threshold) << 6;
    value |= uint(btm_left.r >= threshold) << 5;
    value |= uint(btm_left.g >= threshold) << 4;
    value |= uint(btm_left.b >= threshold) << 3;
    value |= uint(btm_right.r >= threshold) << 2;
    value |= uint(btm_right.g >= threshold) << 1;
    value |= uint(btm_right.b >= threshold) << 0;

    float deg = (360.0 * float(value) / 4096.0);
    angle = vec4(radians(-deg), 0.0, 0.0, 1.0);
}
