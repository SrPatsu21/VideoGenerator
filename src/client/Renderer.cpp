#include "Renderer.hpp"
#include "VulkanContext.hpp"
#include <iostream>

Renderer::Renderer(VulkanContext* ctx) : ctx_(ctx) {}

Renderer::~Renderer() {
    cleanup();
}

bool Renderer::init() {
    return true;
}

void Renderer::cleanup() {}

VkImage Renderer::uploadImageRGBA(const uint8_t* data, uint32_t width, uint32_t height,
                                  VkDeviceMemory* outMemory)
{
    VkDevice device = ctx_->device();

    VkImageCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent.width = width;
    ci.extent.height = height;
    ci.extent.depth = 1;
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    VkImage image;
    if (vkCreateImage(device, &ci, nullptr, &image) != VK_SUCCESS)
        return VK_NULL_HANDLE;

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, image, &memReq);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = memReq.size;
    ai.memoryTypeIndex = 0;

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(ctx_->physicalDevice(), &memProps);

    bool found = false;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            ai.memoryTypeIndex = i;
            found = true;
            break;
        }
    }

    if (!found)
        return VK_NULL_HANDLE;

    VkDeviceMemory memory;
    if (vkAllocateMemory(device, &ai, nullptr, &memory) != VK_SUCCESS)
        return VK_NULL_HANDLE;

    vkBindImageMemory(device, image, memory, 0);
    *outMemory = memory;

    std::cout << "Image uploaded (placeholder, no staging yet)\n";
    return image;
}

void Renderer::drawFrame(VkImage image) {
    std::cout << "Drawing frame (stub). Vulkan swapchain not implemented yet.\n";
}
