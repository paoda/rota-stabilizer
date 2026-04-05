#version 330 core

in vec2 uv;
out vec4 frag_colour;

uniform float u_radius;

void main() {
    vec2 pos = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float d = length(pos);

    float mask = 1.0 - smoothstep(u_radius - fwidth(d), u_radius, d);
    if (mask <= 0.0) discard;

    frag_colour = vec4(0.0, 0.0, 0.0, 0.1 * mask);
}
