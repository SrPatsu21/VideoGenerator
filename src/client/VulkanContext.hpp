#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <optional>

struct VulkanContext {
    // GLFW window
    GLFWwindow* window = nullptr;
    int width = 512;
    int height = 512;

    // Core Vulkan objects
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    // Swapchain
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapchainFormat = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D swapchainExtent{};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    // Command pool
    VkCommandPool cmdPool = VK_NULL_HANDLE;

    // Synchronization (multi-frame-in-flight)
    uint32_t maxFramesInFlight = 2;

    VulkanContext() = default;
    ~VulkanContext();

    void init(GLFWwindow* win);
    void cleanup();

    // helpers
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer cmd);

    // find suitable memory type
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) const;
};
