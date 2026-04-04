#version 330 core
in vec2 uv;
out vec4 frag_color;

uniform sampler2D u_rgb_tex;
uniform mat3 u_colour_space;
uniform int u_is_y;

const float Y_OFFSET = 16.0 / 255.0;
const float INV_Y_SCALE = (235.0 - 16.0) / 255.0;
const float INV_UV_SCALE = (240.0 - 16.0) / 255.0;

void main() {
    // libav expects top-down by default. glReadPixels gives bottom-up.
    // Flip vertically here so glReadPixels gives us top-down data.
    vec3 rgb = clamp(texture(u_rgb_tex, vec2(uv.x, 1.0 - uv.y)).rgb, 0.0, 1.0);
    
    vec3 yuv = clamp(inverse(u_colour_space) * rgb, -1.0, 1.0);
    
    float y = clamp(yuv.x, 0.0, 1.0);
    vec2 _uv = clamp(vec2(yuv.y, yuv.z) + 0.5, 0.0, 1.0);
    
    if (u_is_y == 1) {
        frag_color = vec4(y * INV_Y_SCALE + Y_OFFSET, 0.0, 0.0, 1.0);
    } else {
        frag_color = vec4(_uv * INV_UV_SCALE + Y_OFFSET, 0.0, 1.0);
    }
}
