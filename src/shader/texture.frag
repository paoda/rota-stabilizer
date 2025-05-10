#version 330 core
out vec4 frag_color;

in vec2 uv;

uniform sampler2D u_y_tex;
uniform sampler2D u_uv_tex;

float border = 0.0075;

vec3 nv12ToRgb(float normalized_y, vec2 normalized_uv) {
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
    float y = (normalized_y - offset) * (255.0 / (235.0 - 16.0));
    vec2 uv = (normalized_uv - offset) * (255.0 / (240.0 - 16.0));

    y = clamp(y, 0.0, 1.0);
    uv = clamp(uv, 0.0, 1.0);

    float u = uv.r - 0.5;
    float v = uv.g - 0.5;

    return clamp(bt601 * vec3(y, u, v), 0.0, 1.0);
}

void main() {
    if (uv.x >= (1.0 - border) || uv.x <= border || uv.y >= (1.0 - border * 2.0) || uv.y <= border * 2.0) {
        frag_color = vec4(vec3(1.0), 0.4); // TODO: make alpha channel runtime available?
        return;
    }

    vec3 rgb = nv12ToRgb(texture(u_y_tex, uv).r, texture(u_uv_tex, uv).rg);
    frag_color = vec4(rgb, 1.0);
}
