#version 330 core
out vec2 uv;

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 _uv; 

uniform mat2 u_transform;

void main() {
	gl_Position = vec4(u_transform * pos, 0.0, 1.0);
	uv = _uv;
}
