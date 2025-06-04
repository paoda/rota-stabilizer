#version 330 core
out vec2 uv;

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 _uv;

uniform mat2 u_world_transform;
uniform mat2 u_view_transform;
uniform mat2 u_clip_transform;

uniform sampler2D u_angle;

void main() {
    float angle = texelFetch(u_angle, ivec2(0), 0).r;
    float cosTheta = cos(angle);
    float sinTheta = sin(angle);

    // FIXME: issues passing matrix as texture, so we do this 4x
    mat2 rotation = mat2(cosTheta, -sinTheta, sinTheta, cosTheta); 

    vec2 view_pos  = u_view_transform * pos;
    vec2 world_pos = rotation * (u_world_transform * view_pos);
    vec2 clip_pos  = u_clip_transform * world_pos;

    gl_Position = vec4(clip_pos, 0.0, 1.0);
    uv = _uv;
}
