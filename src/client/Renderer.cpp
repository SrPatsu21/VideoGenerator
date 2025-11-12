#include "Renderer.hpp"
#include <stdexcept>
#include <array>
#include <iostream>
#include <cstring>

void Renderer::init(VulkanContext* c, ComputePipeline* comp) {
    ctx = c;
    compute = comp;
    frames.resize(ctx->maxFramesInFlight);

    // Create sync objects
    for (size_t i = 0; i < frames.size(); ++i) {
        VkSemaphoreCreateInfo sem{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VkFenceCreateInfo fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT; // start signaled so first frame can run

        if (vkCreateSemaphore(ctx->device, &sem, nullptr, &frames[i].imageAvailable) != VK_SUCCESS ||
            vkCreateSemaphore(ctx->device, &sem, nullptr, &frames[i].renderFinished) != VK_SUCCESS ||
            vkCreateFence(ctx->device, &fi, nullptr, &frames[i].inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create sync objects");
        }
    }

    // create staging buffer (host visible) for readback: allocate enough for one RGBA8 frame
    stagingSize = (VkDeviceSize)ctx->swapchainExtent.width * ctx->swapchainExtent.height * 4;
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = stagingSize;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx->device, &bci, nullptr, &stagingBuffer) != VK_SUCCESS)
        throw std::runtime_error("Failed to create staging buffer");

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(ctx->device, stagingBuffer, &mr);
    uint32_t memType = ctx->findMemoryType(mr.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = memType;
    if (vkAllocateMemory(ctx->device, &mai, nullptr, &stagingMemory) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate staging memory");

    vkBindBufferMemory(ctx->device, stagingBuffer, stagingMemory, 0);

    // map persistently
    void* mapped = nullptr;
    vkMapMemory(ctx->device, stagingMemory, 0, stagingSize, 0, &mapped);
    // keep mapped pointer for reads; we will hand it out to caller
}

void Renderer::cleanup() {
    if (stagingMemory != VK_NULL_HANDLE) {
        vkUnmapMemory(ctx->device, stagingMemory);
        vkDestroyBuffer(ctx->device, stagingBuffer, nullptr);
        vkFreeMemory(ctx->device, stagingMemory, nullptr);
    }

    for (auto &f : frames) {
        if (f.imageAvailable) vkDestroySemaphore(ctx->device, f.imageAvailable, nullptr);
        if (f.renderFinished) vkDestroySemaphore(ctx->device, f.renderFinished, nullptr);
        if (f.inFlightFence) vkDestroyFence(ctx->device, f.inFlightFence, nullptr);
    }
}

bool Renderer::renderFrame(float t, void** outMappedPtr, VkDeviceSize* outSize) {
    if (!ctx || !compute) return false;

    FrameObjects &fo = frames[currentFrame];

    // wait for fence for this frame
    vkWaitForFences(ctx->device, 1, &fo.inFlightFence, VK_TRUE, UINT64_MAX);
    vkResetFences(ctx->device, 1, &fo.inFlightFence);

    // acquire swapchain image
    uint32_t imageIndex;
    VkResult r = vkAcquireNextImageKHR(ctx->device, ctx->swapchain, UINT64_MAX, fo.imageAvailable, VK_NULL_HANDLE, &imageIndex);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        // swapchain recreation needed — not handled here
        std::cerr << "Swapchain out of date\n";
        return false;
    } else if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image");
    }

    // Build a single command buffer that:
    // 1) dispatch compute writing into compute->storageImage (GENERAL)
    // 2) transition storageImage -> TRANSFER_SRC
    // 3) transition swapchainImages[imageIndex] -> TRANSFER_DST
    // 4) copy storageImage -> swapchainImage
    // 5) copy storageImage -> stagingBuffer (so host can read)
    // 6) transition images back to present/general as needed
    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandPool = ctx->cmdPool;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx->device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    // (1) compute dispatch
    compute->recordDispatch(cmd, t);

    // (2) storageImage -> TRANSFER_SRC
    VkImageMemoryBarrier toTransferSrc{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    toTransferSrc.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toTransferSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransferSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferSrc.image = compute->storageImage;
    toTransferSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransferSrc.subresourceRange.levelCount = 1;
    toTransferSrc.subresourceRange.layerCount = 1;
    toTransferSrc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toTransferSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toTransferSrc);

    // (3) swapchain image -> TRANSFER_DST
    VkImage dstImage = ctx->swapchainImages[imageIndex];
    VkImageMemoryBarrier swapToDst{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    swapToDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; // assume undefined
    swapToDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapToDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapToDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapToDst.image = dstImage;
    swapToDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    swapToDst.subresourceRange.levelCount = 1;
    swapToDst.subresourceRange.layerCount = 1;
    swapToDst.srcAccessMask = 0;
    swapToDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &swapToDst);

    // (4) copy storageImage -> swapchain image
    VkImageCopy copyRegion{};
    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.srcSubresource.mipLevel = 0;
    copyRegion.srcSubresource.baseArrayLayer = 0;
    copyRegion.srcSubresource.layerCount = 1;
    copyRegion.srcOffset = {0,0,0};
    copyRegion.dstSubresource = copyRegion.srcSubresource;
    copyRegion.dstOffset = {0,0,0};
    copyRegion.extent = { ctx->swapchainExtent.width, ctx->swapchainExtent.height, 1 };
    vkCmdCopyImage(cmd, compute->storageImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    // (5) copy storageImage -> stagingBuffer
    VkBufferImageCopy bic{};
    bic.bufferOffset = 0;
    bic.bufferRowLength = 0; // tightly packed
    bic.bufferImageHeight = 0;
    bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bic.imageSubresource.mipLevel = 0;
    bic.imageSubresource.baseArrayLayer = 0;
    bic.imageSubresource.layerCount = 1;
    bic.imageOffset = {0,0,0};
    bic.imageExtent = { ctx->swapchainExtent.width, ctx->swapchainExtent.height, 1 };
    vkCmdCopyImageToBuffer(cmd, compute->storageImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer, 1, &bic);

    // (6) transition swapchain image -> PRESENT
    VkImageMemoryBarrier swapToPresent = swapToDst;
    swapToPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapToPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    swapToPresent.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    swapToPresent.dstAccessMask = 0;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &swapToPresent);

    // transition storageImage back to GENERAL for next compute iteration
    VkImageMemoryBarrier srcToGeneral = toTransferSrc;
    srcToGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    srcToGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    srcToGeneral.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    srcToGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &srcToGeneral);

    vkEndCommandBuffer(cmd);

    // submit
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    VkSemaphore waitSemaphores[] = { frames[currentFrame].imageAvailable };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT }; // compute will run after acquire
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = waitSemaphores;
    si.pWaitDstStageMask = waitStages;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VkSemaphore signalSemaphores[] = { frames[currentFrame].renderFinished };
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(ctx->queue, 1, &si, frames[currentFrame].inFlightFence) != VK_SUCCESS)
        throw std::runtime_error("Failed to submit draw command buffer");

    // present
    VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = signalSemaphores;
    pi.swapchainCount = 1;
    pi.pSwapchains = &ctx->swapchain;
    pi.pImageIndices = &imageIndex;
    VkResult pres = vkQueuePresentKHR(ctx->queue, &pi);
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
        std::cerr << "Swapchain out of date / suboptimal on present\n";
    } else if (pres != VK_SUCCESS) {
        throw std::runtime_error("Failed to present swapchain image");
    }

    // Wait until GPU finished transfer to staging buffer (we waited on fence earlier)
    // Map and return pointer to mapped memory
    void* mapped = nullptr;
    vkMapMemory(ctx->device, stagingMemory, 0, stagingSize, 0, &mapped);
    if (!mapped) {
        throw std::runtime_error("Failed to map staging memory");
    }

    *outMappedPtr = mapped;
    *outSize = stagingSize;

    // cleanup local cmd buffer
    vkFreeCommandBuffers(ctx->device, ctx->cmdPool, 1, &cmd);

    // advance frame index
    currentFrame = (currentFrame + 1) % frames.size();
    return true;
}
