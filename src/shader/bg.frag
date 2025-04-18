#version 330 core

uniform sampler2D u_screen; 

in vec2 uv;
out vec4 frag_colour;

void main() {
    frag_colour = texture(u_screen, uv);
}
