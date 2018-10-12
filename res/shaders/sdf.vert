#version 410 core

in vec3 position;
out vec3 nearPlanePos;

uniform mat4 VPI;

void main() {
    gl_Position = vec4(position, 1.0);
    vec4 tmp = vec4(VPI * gl_Position);
    nearPlanePos = vec3(tmp) / tmp.w;
}
