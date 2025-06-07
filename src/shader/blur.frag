#version 330 core

out vec4 frag_colour;
in vec2 uv;

const float weight[5] = float[](
    0.2041636887,
    0.1801738229,
    0.1238315368,
    0.0662822453,
    0.0276305506
);

uniform vec2 u_resolution;
uniform bool u_horizontal;
uniform bool u_use_nv12;

uniform sampler2D u_screen;
uniform sampler2D u_y_tex;
uniform sampler2D u_uv_tex;

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

vec3 getTexture(vec2 uv) {
    if (u_use_nv12) {
        float y = texture(u_y_tex, uv).r;
        vec2 uv = texture(u_uv_tex, uv).rg;

        return nv12ToRgb(y, uv);
    }

    return texture(u_screen, uv).rgb;
}

void main() {
    // Calculate texel size based on screen dimensions
    vec2 tex_offset = 1.0 / u_resolution;

    vec3 result = getTexture(uv) * weight[0];

    if (u_horizontal) { // Horizontal pass - sample along X-axis
        for (int i = 1; i < 5; ++i) {
            result += getTexture(uv + vec2(tex_offset.x * i, 0.0)) * weight[i];
            result += getTexture(uv - vec2(tex_offset.x * i, 0.0)) * weight[i];
        }
    } else { // Vertical pass - sample along Y-axis
        for (int i = 1; i < 5; ++i) {
            result += getTexture(uv + vec2(0.0, tex_offset.y * i)) * weight[i];
            result += getTexture(uv - vec2(0.0, tex_offset.y * i)) * weight[i];
        }
    }

    frag_colour = vec4(result, 1.0);
}
