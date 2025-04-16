#version 330 core
out vec4 frag_color;

in vec2 uv;

uniform sampler2D u_screen;
uniform vec2 u_viewport;
// TODO: Properly determine the radius

float border = 0.0075;

void main() {
	vec2 normalized_coord = gl_FragCoord.xy / u_viewport;
	float len = length(normalized_coord - vec2(0.5, 0.5));

	if (
		uv.x < (1 - border) &&
		uv.x > border &&
		uv.y < (1 - border * 2) &&
		uv.y > border * 2
	) {
		frag_color = texture(u_screen, uv);
		return;
	}

	frag_color = vec4(vec3(0.5), 1.0);
}

