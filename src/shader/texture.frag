#version 330 core

out vec4 frag_color;
in vec2 uv;

uniform sampler2D u_y_tex;
uniform sampler2D u_uv_tex;

uniform float u_ratio;
uniform float u_border_radius;
uniform mat3 u_colour_space;
uniform ivec2 u_resolution;

const float Y_OFFSET = 16.0 / 255.0;
const float Y_SCALE = 255.0 / (235.0 - 16.0);
const float UV_SCALE = 255.0 / (240.0 - 16.0);

vec3 sampleTex(vec2 pos) {
    float y = (texture(u_y_tex, pos).r - Y_OFFSET) * Y_SCALE;
    vec2 _uv = (texture(u_uv_tex, pos).rg - Y_OFFSET) * UV_SCALE;

    y = clamp(y, 0.0, 1.0);
    _uv = clamp(_uv, 0.0, 1.0);

    return clamp(u_colour_space * vec3(y, _uv.r - 0.5, _uv.g - 0.5), 0.0, 1.0);
}

float roundedBoxSDF(vec2 pos, vec2 half_size, float radius) {
    vec2 q = abs(pos) - half_size + radius;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - radius;
}

void main() {
    float W = float(u_resolution.x);
    float H = float(u_resolution.y);

    float border_thickness = W * 0.005;

    float gameplay_height = W / u_ratio;
    float height_diff = H - gameplay_height;
    float threshold = (height_diff / 2.0) / H;

    if (uv.y < threshold || uv.y > (1.0 - threshold)) discard;

    vec2 uv_norm = uv;
    if (height_diff > 0.0) uv_norm.y = (uv.y - threshold) / (gameplay_height / H);

    vec2 half_size = vec2(W, gameplay_height) / 2.0;
    vec2 pos = (uv_norm * 2.0 - 1.0) * half_size;

    float dist = roundedBoxSDF(pos, half_size, u_border_radius);

    float softness = fwidth(dist);

    // clip corners
    float outer_alpha = 1.0 - smoothstep(-softness, softness, dist);
    if (outer_alpha <= 0.0) discard;

    // transition to border smoothly
    float border_mix = smoothstep(-border_thickness - softness, -border_thickness + softness, dist);

    vec4 border_colour = vec4(vec3(1.0), 0.7);

    // Blend the video and border based on distance
    vec4 final_colour = mix(vec4(sampleTex(uv), 1.0), border_colour, border_mix);
    final_colour.a *= outer_alpha; // apply rounded corner cutoff

    frag_color = final_colour;
}
