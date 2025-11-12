#pragma once

#include "VulkanContext.hpp"
#include <vulkan/vulkan.h>
#include <string>

struct ComputePipeline {
    VkDevice device = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    // storage image + memory + view (device local)
    VkImage storageImage = VK_NULL_HANDLE;
    VkDeviceMemory storageMemory = VK_NULL_HANDLE;
    VkImageView storageView = VK_NULL_HANDLE;

    int width = 0;
    int height = 0;

    void init(VkDevice dev, VkPhysicalDevice phys, VulkanContext &ctx, const std::string& spvPath);
    void cleanup(VulkanContext &ctx);

    // record dispatch into provided command buffer (pushes float time)
    void recordDispatch(VkCommandBuffer cmd, float t) const;
};
