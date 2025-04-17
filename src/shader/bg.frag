#version 330 core

uniform sampler2D u_screen;  // Input texture
uniform vec2 u_dimension;    // (1.0 / textureWidth, 1.0 / textureHeight)

in vec2 uv;
out vec4 frag_color;

void main() {
    // Larger Gaussian weights for a 7-tap kernel
    float weights[7] = float[](0.204164, 0.176033, 0.120985, 0.064759, 0.034457, 0.016216, 0.007383);

    // Start with the center texel
    vec4 color = texture(u_screen, uv) * weights[0];
    
    // Accumulate contributions from neighboring texels
    for (int i = 1; i < 7; ++i) {
        vec2 offset = u_dimension * float(i * 2); // Increased sampling offset
        color += texture(u_screen, uv + vec2(offset.x, 0.0)) * weights[i];  // Horizontal +
        color += texture(u_screen, uv - vec2(offset.x, 0.0)) * weights[i];  // Horizontal -
        color += texture(u_screen, uv + vec2(0.0, offset.y)) * weights[i];  // Vertical +
        color += texture(u_screen, uv - vec2(0.0, offset.y)) * weights[i];  // Vertical -
    }

    frag_color = color;
}