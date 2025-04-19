#version 330 core
out vec4 frag_color;

in vec2 uv;

uniform sampler2D u_screen;
float border = 0.0075;

void main() {
	if (
		uv.x < (1 - border) &&
		uv.x > border &&
		uv.y < (1 - border * 2) &&
		uv.y > border * 2
	) {
		frag_color = texture(u_screen, uv);
		return;
	}

	frag_color = vec4(vec3(1.0), 0.4); // TODO: make alpha channel runtime available?
}

