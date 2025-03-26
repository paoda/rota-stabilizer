#version 330 core
out vec2 uv;

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 _uv; 

uniform vec2 u_aspect;
uniform vec2 u_rotation;
uniform float u_scale;

void main() {
	vec2 scaled = pos * u_aspect * u_scale; 
	vec2 rotated = vec2(
		scaled.x * u_rotation.y + scaled.y * u_rotation.x,
		scaled.y * u_rotation.y - scaled.x * u_rotation.x
	);	

	gl_Position = vec4(rotated, 0.0, 1.0);
	uv = _uv;
}
