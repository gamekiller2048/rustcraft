#version 450

layout (location = 0) out vec4 fragColor;
layout (location = 0) in vec2 texUV;

// layout (binding = 1) uniform sampler2D texSampler;

void main() {    
    fragColor = vec4(gl_FragCoord.x / 500.0f, gl_FragCoord.y / 5000.0f, 1, 1); // texture(texSampler, texUV);
}