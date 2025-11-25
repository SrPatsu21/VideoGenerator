#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>


// Minimal Vulkan context skeleton
class VulkanContext {
public:
VulkanContext();
~VulkanContext();


bool init();
void cleanup();


VkDevice device() const { return device_; }
VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
VkQueue graphicsQueue() const { return graphicsQueue_; }
VkCommandPool commandPool() const { return commandPool_; }
VkInstance instance() const { return instance_; }


// Helpers
VkCommandBuffer beginSingleTimeCommands();
void endSingleTimeCommands(VkCommandBuffer cmd);


private:
VkInstance instance_ = VK_NULL_HANDLE;
VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
VkDevice device_ = VK_NULL_HANDLE;
VkQueue graphicsQueue_ = VK_NULL_HANDLE;
uint32_t graphicsQueueFamily_ = 0;
VkCommandPool commandPool_ = VK_NULL_HANDLE;


bool createInstance();
bool pickPhysicalDevice();
bool createLogicalDevice();
bool createCommandPool();
};