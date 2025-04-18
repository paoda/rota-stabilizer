#version 330 core

uniform sampler2D u_screen; 

in vec2 uv;
out vec4 frag_colour;

uniform float u_darkness = 0.1;

void main() {
    vec3 tinted = mix(texture(u_screen, uv).rgb, vec3(0), u_darkness);
    
    frag_colour = vec4(tinted, 1.0);
}
