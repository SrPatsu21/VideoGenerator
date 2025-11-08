#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <algorithm>   // for std::clamp
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// -----------------------------------------------------------------------------
// Helper: read SPIR-V binary
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open file!");
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

// -----------------------------------------------------------------------------
// Create Vulkan image + view
VkImage createStorageImage(VkDevice device, VkPhysicalDevice phys, int width, int height,
                           VkDeviceMemory& outMemory, VkImageView& outView) {
    VkImageCreateInfo imgInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imgInfo.extent = { (uint32_t)width, (uint32_t)height, 1 };
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image;
    if (vkCreateImage(device, &imgInfo, nullptr, &image) != VK_SUCCESS)
        throw std::runtime_error("failed to create image!");

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, image, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);

    uint32_t memType = 0;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        if (memReq.memoryTypeBits & (1 << i))
            if (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                memType = i;
                break;
            }

    VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memType;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &outMemory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate image memory!");
    vkBindImageMemory(device, image, outMemory, 0);

    VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &outView) != VK_SUCCESS)
        throw std::runtime_error("failed to create image view!");

    return image;
}

// -----------------------------------------------------------------------------
// Transition image layout one-shot
void transitionImageLayoutOneShot(VkDevice device, VkQueue queue, VkCommandPool pool,
                                  VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = pool;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier
    );

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, pool, 1, &cmd);
}

// -----------------------------------------------------------------------------
// Create compute pipeline
VkPipeline createComputePipeline(VkDevice device, VkPipelineLayout layout, const std::string& path) {
    auto code = readFile(path);
    VkShaderModuleCreateInfo modInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    modInfo.codeSize = code.size();
    modInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &modInfo, nullptr, &shaderModule) != VK_SUCCESS)
        throw std::runtime_error("failed to create shader module");

    VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shaderModule;
    stage.pName = "main";

    VkComputePipelineCreateInfo pipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipeInfo.stage = stage;
    pipeInfo.layout = layout;

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute pipeline!");

    vkDestroyShaderModule(device, shaderModule, nullptr);
    return pipeline;
}

// -----------------------------------------------------------------------------
// Dispatch and wait
void dispatchComputeAndWait(VkDevice device, VkQueue queue, VkCommandPool pool, VkPipeline pipeline,
                            VkPipelineLayout layout, VkDescriptorSet descSet, int w, int h, float t) {
    VkCommandBufferAllocateInfo alloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandPool = pool;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &alloc, &cmd);

    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descSet, 0, nullptr);
    vkCmdDispatch(cmd, (uint32_t)std::ceil(w / 16.0), (uint32_t)std::ceil(h / 16.0), 1);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, pool, 1, &cmd);
}

// -----------------------------------------------------------------------------
// Copy image to CPU and save
void copyImageToBufferAndSave(VkDevice device, VkPhysicalDevice phys, VkQueue queue,
                              VkCommandPool pool, VkImage srcImage, int width, int height,
                              const char* filename) {
    // For brevity, this assumes the image was transitioned to TRANSFER_SRC_OPTIMAL
    VkDeviceSize imgSize = width * height * 4;

    VkBuffer buffer;
    VkDeviceMemory mem;
    VkBufferCreateInfo bufInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufInfo.size = imgSize;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vkCreateBuffer(device, &bufInfo, nullptr, &buffer);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    uint32_t memType = 0;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        if (memReq.memoryTypeBits & (1 << i))
            if (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                memType = i;

    VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memType;
    vkAllocateMemory(device, &allocInfo, nullptr, &mem);
    vkBindBufferMemory(device, buffer, mem, 0);

    // record command to copy image → buffer
    VkCommandBufferAllocateInfo alloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandPool = pool;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &alloc, &cmd);

    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { (uint32_t)width, (uint32_t)height, 1 };
    vkCmdCopyImageToBuffer(cmd, srcImage, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &region);

    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    // map memory to CPU
    void* data;
    vkMapMemory(device, mem, 0, imgSize, 0, &data);
    unsigned char* pixels = reinterpret_cast<unsigned char*>(data);

    std::vector<unsigned char> img(width * height * 4);
    for (int i = 0; i < width * height; i++) {
        img[i * 4 + 0] = (unsigned char)(255.0f * std::clamp(pixels[i * 4 + 0] / 255.0f, 0.0f, 1.0f));
        img[i * 4 + 1] = (unsigned char)(255.0f * std::clamp(pixels[i * 4 + 1] / 255.0f, 0.0f, 1.0f));
        img[i * 4 + 2] = (unsigned char)(255.0f * std::clamp(pixels[i * 4 + 2] / 255.0f, 0.0f, 1.0f));
        img[i * 4 + 3] = 255;
    }

    stbi_write_png(filename, width, height, 4, img.data(), width * 4);

    vkUnmapMemory(device, mem);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, mem, nullptr);
    vkFreeCommandBuffers(device, pool, 1, &cmd);
}

// -----------------------------------------------------------------------------
// main()
int main() {
    // Init Vulkan instance/device minimal setup
    VkInstance instance;
    VkInstanceCreateInfo ci{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    vkCreateInstance(&ci, nullptr, &instance);

    uint32_t deviceCount = 1;
    VkPhysicalDevice phys;
    vkEnumeratePhysicalDevices(instance, &deviceCount, &phys);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDevice device;
    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    vkCreateDevice(phys, &dci, nullptr, &device);

    VkQueue queue;
    vkGetDeviceQueue(device, 0, 0, &queue);

    VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.queueFamilyIndex = 0;
    VkCommandPool pool;
    vkCreateCommandPool(device, &poolInfo, nullptr, &pool);

    int width = 512, height = 512;

    VkDeviceMemory imgMem;
    VkImageView imgView;
    VkImage image = createStorageImage(device, phys, width, height, imgMem, imgView);

    // FIX: Transition image to GENERAL
    transitionImageLayoutOneShot(device, queue, pool, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // Descriptor set layout
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    VkDescriptorSetLayout descLayout;
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descLayout);

    VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &descLayout;

    VkPipelineLayout pipelineLayout;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &pipelineLayout);

    VkPipeline pipeline = createComputePipeline(device, pipelineLayout, "./shaders/compute.glsl.spv");

    // Descriptor pool
    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 };
    VkDescriptorPoolCreateInfo poolCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolCI.maxSets = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes = &poolSize;
    VkDescriptorPool descPool;
    vkCreateDescriptorPool(device, &poolCI, nullptr, &descPool);

    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = descPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descLayout;

    VkDescriptorSet descSet;
    vkAllocateDescriptorSets(device, &allocInfo, &descSet);

    VkDescriptorImageInfo imgDesc{};
    imgDesc.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgDesc.imageView = imgView;

    VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = descSet;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &imgDesc;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    // Dispatch compute
    dispatchComputeAndWait(device, queue, pool, pipeline, pipelineLayout, descSet, width, height, 0.0f);

    // Copy and save
    copyImageToBufferAndSave(device, phys, queue, pool, image, width, height, "out.png");

    std::cout << "Saved out.png\n";

    return 0;
}
