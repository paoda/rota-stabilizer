# version 330 core

layout(location = 0) out vec4 angle;
in vec2 uv;

uniform sampler2D u_screen;
uniform ivec2 u_dimension;

vec3 sampleTexture(int size, ivec2 start_pos) {
    vec3 sum = vec3(0.0);

    for (int dy = 0; dy < size; dy++) {
        for (int dx = 0; dx < size; dx++) {
            ivec2 tex_pos = start_pos + ivec2(dx, dy);
            sum += texelFetch(u_screen, tex_pos, 0).rgb;
        }
    }

    return sum / float(size * size);
}

void main() {
    const int ofs = 5;
    const int size = 3;
    const float threshold = (255.0 / 2.0) / 255.0;

    // Sample regions at the same positions as the CPU code
    vec3 btm_left = sampleTexture(size, ivec2(ofs, u_dimension.y - ofs - size));
    vec3 top_left = sampleTexture(size, ivec2(ofs, ofs));
    vec3 btm_right = sampleTexture(size, ivec2(u_dimension.x - ofs - size, u_dimension.y - ofs - size));
    vec3 top_right = sampleTexture(size, ivec2(u_dimension.x - ofs - size, ofs));

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
