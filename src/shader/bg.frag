#version 330 core

in vec2 uv;
out vec4 frag_colour;

uniform mat2 u_world_transform;
uniform mat2 u_view_transform;

uniform sampler2D u_blur;
uniform sampler2D u_y_tex;
uniform sampler2D u_uv_tex;

uniform mat3 u_colour_space;
uniform float u_intensity;
uniform float u_radius;
uniform float u_zoom;

uniform vec3 u_tint;

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

void main() {
    vec2 zoom_uv = (uv - 0.5) / u_zoom + 0.5;

    vec2 pos = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    vec2 world_pos = u_world_transform * pos;
    vec2 view_pos = u_view_transform * world_pos;

    float d = length(view_pos);
    float mask = smoothstep(u_radius - fwidth(d), u_radius, d);

    vec3 outer = mix(texture(u_blur, uv).rgb, u_tint, u_intensity);
    // vec3 inner = sampleTex(zoom_uv); // zoom_uv is only for inside u_radius
    vec3 inner = texture(u_blur, zoom_uv).rgb; // zoom_uv is only for inside u_radius

    frag_colour = vec4(mix(inner, outer, mask), 1.0);
}
