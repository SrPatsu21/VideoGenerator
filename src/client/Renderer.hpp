#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>

class VulkanContext;

class Renderer {
public:
    explicit Renderer(VulkanContext* ctx);
    ~Renderer();

    bool init();
    void cleanup();

    VkImage uploadImageRGBA(const uint8_t* data, uint32_t width, uint32_t height,
                            VkDeviceMemory* outMemory);

    void drawFrame(VkImage image);

private:
    VulkanContext* ctx_;
};
