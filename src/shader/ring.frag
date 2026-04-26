#version 330 core

in vec2 uv;
out vec4 frag_colour;

uniform float u_radius;
uniform float u_thickness;

void main() {
    vec2 pos = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float d = length(pos);

    float half_thick = u_thickness * 0.5;
    float softness = fwidth(d) * 1;

    float mask = 1.0 - smoothstep(half_thick - softness, half_thick, abs(d - u_radius));
    if (mask <= 0.0) discard;

    frag_colour = vec4(vec3(1.0), 0.3 * mask);
}
