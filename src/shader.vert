#version 450

layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec2 a_texUV;

layout (location = 0) out vec2 texUV;

layout(binding = 0) uniform TransformationData {
    mat4 proj_view;
    mat4 model;
    float t;
} ubo;

void main() {
    texUV = a_texUV;
    gl_Position = ubo.proj_view * ubo.model * vec4(a_pos * (sin(ubo.t) + 2), 1.0);
}