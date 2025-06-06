#version 330 core
    
in vec2 uv;
out vec4 frag_colour;

uniform sampler2D u_blur; 

uniform float u_darkness = 0.0;
// uniform vec3 u_tint = vec3(1, 1, 1);

void main() {    
    vec3 tinted = mix(texture(u_blur, uv).rgb, vec3(0), u_darkness);
        
    // float luminance = dot(tinted.rgb, vec3(0.2126, 0.7152, 0.0722)); // ITU BT.709
    // tinted = luminance * u_tint.rgb;

    frag_colour = vec4(tinted, 1);    
}
