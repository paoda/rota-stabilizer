#version 330 core

layout (location = 0) in vec2 pos;

uniform float u_scale;
    
void main() {
	gl_Position = vec4(pos * u_scale, 0.0, 1.0);
}
