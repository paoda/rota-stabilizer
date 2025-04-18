#version 330 core
out vec2 uv;

const vec2 pos[3] = vec2[3](vec2(-1.0f, -1.0f), vec2(3.0f, -1.0f), vec2(-1.0f, 3.0f));

void main() {
	uv = pos[gl_VertexID] * 0.5 + 0.5;
	gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
}

