#version 330 core

layout (location = 0) in vec2 pos;

uniform mat2 u_world_transform;
uniform mat2 u_view_transform;
uniform mat2 u_clip_transform;

void main() {

	vec2 view_pos  = u_view_transform * pos;
    vec2 world_pos =  u_world_transform * view_pos;
    vec2 clip_pos  = u_clip_transform * world_pos;

    gl_Position = vec4(clip_pos, 0.0, 1.0);
}
