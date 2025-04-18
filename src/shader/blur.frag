#version 330 core

out vec4 frag_colour;
in vec2 uv;

uniform float weight[15] = float[] (
    0.031225, 0.033324, 0.035206, 0.036826, 0.038138, 
    0.039104, 0.039695, 0.039894, 0.039695, 
    0.039104, 0.038138, 0.036826, 0.035206, 0.033324, 0.031225
);

uniform vec2 u_resolution;
uniform bool u_horizontal;
uniform sampler2D u_screen;

void main() {
     // Calculate texel size based on screen dimensions
     vec2 tex_offset = 1.0 / u_resolution;
     vec3 result = texture(u_screen, uv).rgb * weight[0]; // current fragment's contribution
    
     if (u_horizontal) { // Horizontal pass - sample along X-axis
         for (int i = 1; i < 15; ++i) {
             result += texture(u_screen, uv + vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
             result += texture(u_screen, uv - vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
         }
     } else { // Vertical pass - sample along Y-axis
         for (int i = 1; i < 15; ++i) {
             result += texture(u_screen, uv + vec2(0.0, tex_offset.y * i)).rgb * weight[i];
             result += texture(u_screen, uv - vec2(0.0, tex_offset.y * i)).rgb * weight[i];
         }
     }
    
     frag_colour = vec4(result, 1.0);
}

