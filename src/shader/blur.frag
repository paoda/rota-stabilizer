#version 330 core

out vec4 frag_colour;
in vec2 uv;

// -- ORIGINAL WEIGHTS --
// const float weight[5] = float[](
//         0.2041636887,
//         0.1801738229,
//         0.1238315368,
//         0.0662822453,
//         0.0276305506
//     );

// -- GL.LINEAR NECESSARY --
const float weight[3] = float[](
    0.2041636887,
    0.3040053597,
    0.0939127959
);

// sub-pixel offsets (gl.LINEAR will do the adding for us)
const float offset[3] = float[](
    0.0,
    1.4073333,
    3.2942149
);

uniform vec2 u_texel_size;
uniform vec2 u_direction;
uniform bool u_use_nv12;
uniform mat3 u_colour_space;

uniform sampler2D u_screen;
uniform sampler2D u_y_tex;
uniform sampler2D u_uv_tex;

const float Y_OFFSET = 16.0 / 255.0;
const float Y_SCALE = 255.0 / (235.0 - 16.0);
const float UV_SCALE = 255.0 / (240.0 - 16.0);

vec3 sampleTex(vec2 pos) {
    if (!u_use_nv12) return texture(u_screen, pos).rgb;

    float y = texture(u_y_tex, pos).r;
    vec2 _uv = texture(u_uv_tex, pos).rg;

    return vec3(y, _uv);
}

vec3 process(vec3 colour) {
    if (!u_use_nv12) return colour;

    float y = (colour.r - Y_OFFSET) * Y_SCALE;
    vec2 _uv = (colour.gb - Y_OFFSET) * UV_SCALE;

    y = clamp(y, 0.0, 1.0);
    _uv = clamp(_uv, 0.0, 1.0);

    return clamp(u_colour_space * vec3(y, _uv.r - 0.5, _uv.g - 0.5), 0.0, 1.0);
}

void main() {
    vec3 result = sampleTex(uv) * weight[0];

    // for (int i = 1; i < 5; ++i) {
    //     vec2 offset = u_direction * u_texel_size * float(i);
    //     result += sampleTex(uv + offset) * weight[i];
    //     result += sampleTex(uv - offset) * weight[i];
    // }

    for (int i = 1; i < 3; ++i) {
        vec2 ofs = u_direction * u_texel_size * offset[i];

        result += sampleTex(uv + ofs) * weight[i];
        result += sampleTex(uv - ofs) * weight[i];
    }

    frag_colour = vec4(process(result), 1.0);
}
