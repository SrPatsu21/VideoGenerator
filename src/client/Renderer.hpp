#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>


class VulkanContext;


// Minimal renderer: uploads RGBA images and presents
class Renderer {
public:
    Renderer(VulkanContext* ctx);
    ~Renderer();


    bool init();
    void cleanup();


    // Uploads a raw RGBA8 CPU buffer into a GPU image
    // and returns a VkImage handle
    VkImage uploadImageRGBA(const uint8_t* data, uint32_t width, uint32_t height);


    // Draws a frame using an image
    void drawFrame(VkImage image);


private:
    VulkanContext* ctx_ = nullptr;


    // Swapchain (placeholder)
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> swapImages_;
    std::vector<VkImageView> swapViews_;


    bool createSwapchain();
    bool createImageViews();
};