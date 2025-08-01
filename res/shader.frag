#version 450

layout (location = 0) out vec4 fragColor;
layout (location = 0) in vec2 texUV;

layout (binding = 1) uniform sampler2D texSampler;

void main() {    
    fragColor = texture(texSampler, texUV);
}