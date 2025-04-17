#version 330 core
out vec2 uv;

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 _uv; 

uniform mat2 u_scale;
uniform mat2 u_aspect;
uniform mat2 u_rotation;


void main() {
	mat2 transform =  u_rotation * u_scale * u_aspect;
	
	gl_Position = vec4(transform * pos, 0.0, 1.0);
	uv = _uv;
}
