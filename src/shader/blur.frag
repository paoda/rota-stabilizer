#version 330 core

out vec4 frag_colour;
in vec2 uv;

uniform float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
// TODO: use length here so we can make this variable

uniform vec2 u_resolution;
uniform bool u_horizontal;
uniform sampler2D u_screen;

void main() {
     // Calculate texel size based on screen dimensions
     vec2 tex_offset = 1.0 / u_resolution;
     vec3 result = texture(u_screen, uv).rgb * weight[0]; // current fragment's contribution
    
     if (u_horizontal) { // Horizontal pass - sample along X-axis
         for (int i = 1; i < 5; ++i) {
             result += texture(u_screen, uv + vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
             result += texture(u_screen, uv - vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
         }
     } else { // Vertical pass - sample along Y-axis
         for (int i = 1; i < 5; ++i) {
             result += texture(u_screen, uv + vec2(0.0, tex_offset.y * i)).rgb * weight[i];
             result += texture(u_screen, uv - vec2(0.0, tex_offset.y * i)).rgb * weight[i];
         }
     }
    
     frag_colour = vec4(result, 1.0);
}

