#version 330 core
out vec4 frag_colour;

void main() {
	vec3 colour = vec3(0.6);
	frag_colour = vec4(colour, 0.7);
}
