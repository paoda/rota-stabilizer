#version 330 core
out vec2 uv;

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 _uv;

uniform mat2 u_transform;
uniform sampler2D u_angle;

void main() {
    float angle = texelFetch(u_angle, ivec2(0), 0).r;
    float cosTheta = cos(angle);
    float sinTheta = sin(angle);

    // FIXME: issues passing matrix as texture, so we do this 4x
    mat2 rotation = mat2(cosTheta, -sinTheta, sinTheta, cosTheta); 

    gl_Position = vec4(rotation * (u_transform * pos), 0.0, 1.0);
    uv = _uv;
}
