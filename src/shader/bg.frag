#version 330 core
    
in vec2 uv;
out vec4 frag_colour;

uniform sampler2D u_screen; 
uniform sampler2D u_blurred; 

uniform vec2 u_viewport; 
uniform float u_radius;
uniform float u_darkness = 0.0;

// uniform vec3 u_tint = vec3(0, 1, 0.980);

vec2 center = vec2(0.5, 0.5);
void main() {
    vec2 normalized = gl_FragCoord.xy / u_viewport;
    float dist = distance(normalized, center) * 2;
    
    vec3 tinted;
    if (dist < u_radius) {
        tinted = mix(texture(u_screen, uv).rgb, vec3(0), u_darkness);
    } else {
        tinted = mix(texture(u_blurred, uv).rgb, vec3(0), u_darkness);
        // vec3 tmp = mix(texture(u_blurred, uv).rgb, vec3(0), u_darkness);
        
        // float luminance = dot(tmp.rgb, vec3(0.299, 0.587, 0.114));
        // tinted = luminance * u_tint.rgb;
    }

    frag_colour = vec4(tinted, 1);
    
}
