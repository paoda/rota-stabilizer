#version 330 core
out vec2 uv;

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 _uv; 

uniform vec2 u_scale;
uniform vec2 u_rotation;


void main() {
	vec2 _pos = pos * u_scale * 0.9;

	gl_Position = vec4(_pos, 0.0, 1.0);
	uv = _uv;
}
