#version 330 core

in vec2 uv;
out vec4 frag_colour;

uniform mat2 u_world_transform;
uniform mat2 u_view_transform;

uniform sampler2D u_blur;

uniform float u_darkness = 0.0;
uniform float u_radius = 0.5;

uniform vec3 u_tint = vec3(0.0, 1.0, 1.0);

void main() {
    vec2 pos = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    vec2 world_pos = u_world_transform * pos;
    vec2 view_pos = u_view_transform * world_pos;

    float d = length(view_pos);
    float mask = smoothstep(u_radius - fwidth(d), u_radius, d);

    vec3 normal = texture(u_blur, uv).rgb;
    // vec3 tinted = mix(normal, vec3(0.0), u_darkness) * u_tint;
    vec3 tinted = normal;

    frag_colour = vec4(mix(normal, tinted, mask), 1.0);
}
