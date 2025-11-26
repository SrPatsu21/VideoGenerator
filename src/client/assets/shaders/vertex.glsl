#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 fragColor;

// Uniform Buffer Object
layout(set = 0, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

void main() {
    fragColor = inColor;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
}
