// main.cpp
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <vector>
#include <chrono>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Helper: read SPIR-V
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file: " + filename);
    size_t size = (size_t) file.tellg();
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), size);
    return buffer;
}

uint32_t findComputeQueueFamily(VkPhysicalDevice physicalDevice) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            return i;
    }

    throw std::runtime_error("No compute queue family found!");
}

// --- Forward declare functions we'll implement below ---
VkImage createStorageImage(VkDevice device, VkPhysicalDevice phys, int width, int height, VkDeviceMemory &outMemory, VkImageView &outView);
VkPipeline createComputePipeline(VkDevice device, VkPipelineLayout layout, const std::string &spvPath);
VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);
void dispatchComputeAndWait(VkDevice device, VkQueue computeQueue, VkCommandPool cmdPool, VkPipeline pipeline, VkPipelineLayout layout, VkDescriptorSet descSet, int w, int h, float time);
void copyImageToBufferAndSave(VkDevice device, VkPhysicalDevice phys, VkQueue queue, VkCommandPool pool, VkImage srcImage, int width, int height, const char* filename);

// ---- simplified (pseudo-)main: assumes you have init code for Vulkan/GLFW/ImGui ----
int main() {
    // ---- 1) init GLFW, Vulkan instance, device, queues, swapchain, ImGui ----
    // User already said they have libs configured; assume you have:
    // VkInstance instance; VkDevice device; VkPhysicalDevice phys; VkQueue computeQueue, graphicsQueue; VkCommandPool cmdPool;
    // GLFWwindow* window; Swapchain, render pass, framebuffer, ImGui Vulkan init done...
    // For brevity we do not re-implement initialization here.

    // ---------- USER-SUPPLIED/EXISTING OBJECTS ----------
    VkInstance instance = VK_NULL_HANDLE;        // <--- fill
    VkDevice device = VK_NULL_HANDLE;            // <--- fill
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; // <--- fill
    VkQueue computeQueue = VK_NULL_HANDLE;       // <--- fill (queue that supports compute)
    VkQueue graphicsQueue = VK_NULL_HANDLE;      // <--- fill
    VkCommandPool cmdPool = VK_NULL_HANDLE;      // <--- fill (compute-capable)
    // ---------------------------------------------------

    VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.pApplicationName = "GenAI";
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instInfo.enabledLayerCount = 1;
    const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
    instInfo.ppEnabledLayerNames = layers;

    instInfo.pApplicationInfo = &appInfo;
    vkCreateInstance(&instInfo, nullptr, &instance);

    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr);
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());
    physicalDevice = gpus[0]; // pick the first one for now

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queueInfo.queueFamilyIndex = 0; // choose a family supporting compute (check below)
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo devInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &queueInfo;
    vkCreateDevice(physicalDevice, &devInfo, nullptr, &device);

    vkGetDeviceQueue(device, 0, 0, &computeQueue);

    uint32_t computeQueueFamily = findComputeQueueFamily(physicalDevice);

    // Create the command pool
    VkCommandPoolCreateInfo cmdPoolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cmdPoolInfo.queueFamilyIndex = computeQueueFamily;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool commandPool;
    if (vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool!");
    }


    // IMAGE size
    const int IMG_W = 1024;
    const int IMG_H = 1024;

    // 2) Create storage image (we'll write RGBA32F for best flexibility)
    VkDeviceMemory imgMemory;
    VkImage storageImage;
    VkImageView storageImageView;
    storageImage = createStorageImage(device, physicalDevice, IMG_W, IMG_H, imgMemory, storageImageView);

    // 3) Descriptor set / layout for compute shader (binding 0 = storage image)
    VkDescriptorSetLayoutBinding imgBinding{};
    imgBinding.binding = 0;
    imgBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    imgBinding.descriptorCount = 1;
    imgBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslInfo{};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = 1;
    dslInfo.pBindings = &imgBinding;
    VkDescriptorSetLayout descriptorSetLayout;
    vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &descriptorSetLayout);

    // pipeline layout: push constants for width/height/time
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(int)*2 + sizeof(float);

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &descriptorSetLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;
    VkPipelineLayout pipelineLayout;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &pipelineLayout);

    // 4) Create compute pipeline from compiled compute.spv
    VkPipeline computePipeline = createComputePipeline(device, pipelineLayout, "./shaders/compute.glsl.spv");

    // 5) Allocate descriptor set and update it with storage image view
    // (Assume a descriptor pool exists; create one if needed.)
    // For brevity: create a very simple pool capable of 1 storage image descriptor here:
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo descPoolInfo{};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.poolSizeCount = 1;
    descPoolInfo.pPoolSizes = &poolSize;
    descPoolInfo.maxSets = 1;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE; // <--- fill/create with storage image support
    if (vkCreateDescriptorPool(device, &descPoolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool!");
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;
    VkDescriptorSet descriptorSet;
    vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);

    VkDescriptorImageInfo imgInfo{};
    imgInfo.imageView = storageImageView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // our compute shader will write in GENERAL

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &imgInfo;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    // Main loop variables
    bool running = true;
    auto startTime = std::chrono::high_resolution_clock::now();

    // You will want to expose the storage image to ImGui as a sampled texture.
    // Using ImGui_ImplVulkan_AddTexture is the typical path: it creates a descriptor set for a combined image sampler.
    // For simplicity we assume you have a sampler created:
    VkSampler linearSampler = VK_NULL_HANDLE; // <--- create VkSampler (vkCreateSampler) with normalized coords
    // Create a combined image sampler descriptor for ImGui
    // ImGui_ImplVulkan_AddTexture(sampler, imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) -> returns ImTextureID
    // But you must ensure image is transitioned to SHADER_READ_ONLY layout after compute writes (we'll handle transitions on dispatch)

    // Main loop
    while (running) {
        // compute time
        auto now = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(now - startTime).count();

        // 6) Dispatch compute that writes into storageImage
        dispatchComputeAndWait(device, computeQueue, cmdPool, computePipeline, pipelineLayout, descriptorSet, IMG_W, IMG_H, time);

        // 7) Transition storageImage to SHADER_READ_ONLY_OPTIMAL so the graphics shader / ImGui can sample it
        // (This is done inside dispatchComputeAndWait or you can add a barrier here.)

        // 8) Render ImGui frame and show image using ImGui image widget
        // ImGui::Begin("Generated");
        // ImGui::Image((void*)myImGuiTextureId, ImVec2(IMG_W/2.0f, IMG_H/2.0f));
        // ImGui::End();

        // 9) Poll events, present swapchain, etc. (assume your existing code)

        // For demonstration, let's save an image once (or on user action)
        static bool saved = false;
        if (!saved) {
            copyImageToBufferAndSave(device, physicalDevice, computeQueue, commandPool, storageImage, IMG_W, IMG_H, "out.png");
            std::cout << "Saved output.png\n";
            saved = true;
        }

        // exit condition - integrate with your window closing logic
        // running = !glfwWindowShouldClose(window);
        running = false; // run once for simplified example
    }

    // cleanup (destroy pipelines, images, descriptor pools, layout, etc.)
    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyImageView(device, storageImageView, nullptr);
    vkDestroyImage(device, storageImage, nullptr);
    vkFreeMemory(device, imgMemory, nullptr);

    // ... plus your other cleanup (ImGui, GLFW, device, instance)

    return 0;
}

VkFormat STORAGE_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;

VkImage createStorageImage(VkDevice device, VkPhysicalDevice phys, int width, int height, VkDeviceMemory &outMemory, VkImageView &outView) {
    VkImage img;
    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = STORAGE_FORMAT;
    imgInfo.extent = { (uint32_t)width, (uint32_t)height, 1 };
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vkCreateImage(device, &imgInfo, nullptr, &img);

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, img, &memReq);

    // choose memory type - helper needed; here we assume function exists selectMemoryType
    auto selectMemoryType = [&](uint32_t typeFilter, VkMemoryPropertyFlags props) -> uint32_t {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((typeFilter & (1u<<i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) return i;
        }
        throw std::runtime_error("Failed to find memory type");
    };

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = selectMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vkAllocateMemory(device, &allocInfo, nullptr, &outMemory);
    vkBindImageMemory(device, img, outMemory, 0);

    // create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = img;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = STORAGE_FORMAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(device, &viewInfo, nullptr, &outView);
    return img;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule module;
    if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS)
        throw std::runtime_error("createShaderModule failed");
    return module;
}

VkPipeline createComputePipeline(VkDevice device, VkPipelineLayout layout, const std::string &spvPath) {
    auto bytes = readFile(spvPath);
    VkShaderModule module = createShaderModule(device, bytes);

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";

    VkComputePipelineCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage = stage;
    cp.layout = layout;

    VkPipeline pipeline;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp, nullptr, &pipeline);

    vkDestroyShaderModule(device, module, nullptr);
    return pipeline;
}

void dispatchComputeAndWait(VkDevice device, VkQueue computeQueue, VkCommandPool cmdPool, VkPipeline pipeline, VkPipelineLayout layout, VkDescriptorSet descSet, int w, int h, float time) {
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandPool = cmdPool;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &alloc, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &bi);

    // transition image to GENERAL if needed - assume descriptor image uses GENERAL
    // bind pipeline + descriptor
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descSet, 0, nullptr);

    // push constants: ivec2 size; float time;
    struct Push { int sx, sy; float t; } p;
    p.sx = w; p.sy = h; p.t = time;
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);

    uint32_t groupX = (w + 15) / 16;
    uint32_t groupY = (h + 15) / 16;
    vkCmdDispatch(cmd, groupX, groupY, 1);

    // Insert memory barrier: ensure write finishes before we sample or copy the image
    VkMemoryBarrier memBar{};
    memBar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memBar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0,
                         1, &memBar,
                         0, nullptr,
                         0, nullptr);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    VkFence fence;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device, &fci, nullptr, &fence);

    vkQueueSubmit(computeQueue, 1, &si, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
}

void copyImageToBufferAndSave(VkDevice device, VkPhysicalDevice phys, VkQueue queue, VkCommandPool pool, VkImage srcImage, int width, int height, const char* filename) {
    // Create buffer with VK_BUFFER_USAGE_TRANSFER_DST_BIT and host-visible memory
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    size_t pixelBytes = sizeof(float) * 4;
    VkDeviceSize bufSize = (VkDeviceSize)width * height * pixelBytes;
    bci.size = bufSize;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer dstBuffer;
    vkCreateBuffer(device, &bci, nullptr, &dstBuffer);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, dstBuffer, &req);
    VkMemoryAllocateInfo ainfo{};
    ainfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ainfo.allocationSize = req.size;

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    uint32_t memoryTypeIndex = 0;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memoryTypeIndex = i;
            break;
        }
    }
    ainfo.memoryTypeIndex = memoryTypeIndex;
    VkDeviceMemory bufMemory;
    vkAllocateMemory(device, &ainfo, nullptr, &bufMemory);
    vkBindBufferMemory(device, dstBuffer, bufMemory, 0);

    // Record command buffer to copy image -> buffer (image layout must be TRANSFER_SRC_OPTIMAL)
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandPool = pool;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &alloc, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &bi);

    // Transition srcImage to TRANSFER_SRC_OPTIMAL
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL; // after compute we used GENERAL
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.image = srcImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &barrier);

    VkBufferImageCopy copyRegion{};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = { (uint32_t)width, (uint32_t)height, 1 };
    copyRegion.imageOffset = {0,0,0};
    vkCmdCopyImageToBuffer(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstBuffer, 1, &copyRegion);

    // Transition back to GENERAL (optional)
    VkImageMemoryBarrier barrier2 = barrier;
    barrier2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier2.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier2.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &barrier2);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VkFence fence;
    VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    vkCreateFence(device, &fci, nullptr, &fence);
    vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    // Map buffer and save as PNG (convert float RGBA -> uint8 RGBA)
    void* data;
    vkMapMemory(device, bufMemory, 0, bufSize, 0, &data);
    float* floatPixels = reinterpret_cast<float*>(data);

    std::vector<unsigned char> out;
    out.resize(width * height * 4);
    for (int i = 0; i < width * height; ++i) {
        float r = floatPixels[4*i+0];
        float g = floatPixels[4*i+1];
        float b = floatPixels[4*i+2];
        float a = floatPixels[4*i+3];
        auto conv = [](float v)->unsigned char {
            float x = v;
            x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
            return (unsigned char)(x * 255.0f);
        };
        out[4*i+0] = conv(r);
        out[4*i+1] = conv(g);
        out[4*i+2] = conv(b);
        out[4*i+3] = conv(a);
    }
    vkUnmapMemory(device, bufMemory);

    // Save using stb
    stbi_write_png(filename, width, height, 4, out.data(), width * 4);

    // cleanup
    vkDestroyBuffer(device, dstBuffer, nullptr);
    vkFreeMemory(device, bufMemory, nullptr);
    vkFreeCommandBuffers(device, pool, 1, &cmd);
    vkDestroyFence(device, fence, nullptr);
}

