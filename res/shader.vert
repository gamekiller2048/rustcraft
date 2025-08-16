#version 450

layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec2 a_texUV;

layout (location = 0) out vec2 texUV;
layout (location = 1) out uint index;

layout(std140, binding = 0) uniform TransformationData {
    mat4 proj_view;
    mat4 model;
    float time;
} ubo;

struct Particle {
  vec3 pos;
  vec3 vel;
  vec3 color;
};

layout(std430, binding = 2) readonly buffer ParticleSSBOIn {
   Particle particlesIn[];
};


void main() {
    texUV = a_texUV;
    index = uint(gl_InstanceIndex);
    gl_Position = ubo.proj_view * (ubo.model * vec4(a_pos, 1.0) + vec4(particlesIn[index].pos, 0.0));
}