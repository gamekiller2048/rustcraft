#version 450

layout (location = 0) out vec4 fragColor;

layout (location = 0) in vec2 texUV;
layout (location = 1) in flat uint index;

layout (binding = 1) uniform sampler2D texSampler;

struct Particle {
  vec3 pos;
  vec3 vel;
  vec3 color;
};

layout(std430, binding = 2) readonly buffer ParticleSSBOIn {
   Particle particlesIn[];
};

void main() {    
    fragColor = vec4(particlesIn[index].color, 1.0);
}