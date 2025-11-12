#pragma once
#include "VulkanContext.hpp"
#include "ComputePipeline.hpp"

struct Renderer {
    VulkanContext* ctx = nullptr;
    ComputePipeline* compute = nullptr;

    // sync objects per-frame
    struct FrameObjects {
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        VkFence inFlightFence = VK_NULL_HANDLE;
    };
    std::vector<FrameObjects> frames;
    size_t currentFrame = 0;

    // staging buffer for GPU->CPU readback
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkDeviceSize stagingSize = 0;

    void init(VulkanContext* c, ComputePipeline* comp);
    void cleanup();

    // Render one frame: dispatch compute, copy to swapchain image, present.
    // Also copy into the staging buffer (host-visible) so caller can read and send to FFmpeg.
    // Returns index of swapchain image used and whether the staging buffer contains the frame (true if ready to read).
    bool renderFrame(float t, void** outMappedPtr, VkDeviceSize* outSize);
};
