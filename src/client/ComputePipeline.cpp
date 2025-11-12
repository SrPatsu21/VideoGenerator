#include "ComputePipeline.hpp"
#include "VulkanContext.hpp"
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstring>

static std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Failed to open shader file: " + path);
    size_t s = (size_t)f.tellg();
    std::vector<char> buf(s);
    f.seekg(0);
    f.read(buf.data(), s);
    return buf;
}

void ComputePipeline::init(VkDevice dev, VkPhysicalDevice phys, VulkanContext &ctx, const std::string& spvPath) {
    device = dev;
    width = ctx.width;
    height = ctx.height;

    // Create storage image (device local, optimal)
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = { (uint32_t)width, (uint32_t)height, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &ici, nullptr, &storageImage) != VK_SUCCESS)
        throw std::runtime_error("Failed to create storage image");

    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(device, storageImage, &mr);
    uint32_t memType = ctx.findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = memType;
    if (vkAllocateMemory(device, &mai, nullptr, &storageMemory) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate storage image memory");
    vkBindImageMemory(device, storageImage, storageMemory, 0);

    // view
    VkImageViewCreateInfo iv{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    iv.image = storageImage;
    iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    iv.format = VK_FORMAT_R8G8B8A8_UNORM;
    iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    iv.subresourceRange.levelCount = 1;
    iv.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &iv, nullptr, &storageView) != VK_SUCCESS)
        throw std::runtime_error("Failed to create storage image view");

    // descriptor layout
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dsl{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dsl.bindingCount = 1;
    dsl.pBindings = &b;
    if (vkCreateDescriptorSetLayout(device, &dsl, nullptr, &descLayout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create descriptor set layout");

    // pipeline layout with push constant (float time)
    VkPushConstantRange pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) };
    VkPipelineLayoutCreateInfo plc{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plc.setLayoutCount = 1;
    plc.pSetLayouts = &descLayout;
    plc.pushConstantRangeCount = 1;
    plc.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(device, &plc, nullptr, &layout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create pipeline layout");

    // shader module
    auto code = readFile(spvPath);
    VkShaderModuleCreateInfo sm{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    sm.codeSize = code.size();
    sm.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule module;
    if (vkCreateShaderModule(device, &sm, nullptr, &module) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module");

    VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";

    VkComputePipelineCreateInfo cpi{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpi.stage = stage;
    cpi.layout = layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create compute pipeline");
    vkDestroyShaderModule(device, module, nullptr);

    // descriptor pool + set
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps.descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpc{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpc.poolSizeCount = 1;
    dpc.pPoolSizes = &ps;
    dpc.maxSets = 1;
    if (vkCreateDescriptorPool(device, &dpc, nullptr, &descPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create descriptor pool");

    VkDescriptorSetAllocateInfo dsa{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsa.descriptorPool = descPool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &descLayout;
    if (vkAllocateDescriptorSets(device, &dsa, &descSet) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate descriptor set");

    VkDescriptorImageInfo di{};
    di.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    di.imageView = storageView;
    VkWriteDescriptorSet ws{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    ws.dstSet = descSet;
    ws.dstBinding = 0;
    ws.descriptorCount = 1;
    ws.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws.pImageInfo = &di;
    vkUpdateDescriptorSets(device, 1, &ws, 0, nullptr);

    // transition storage image to GENERAL so compute shader can access it
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = storageImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
    ctx.endSingleTimeCommands(cmd);
}

void ComputePipeline::recordDispatch(VkCommandBuffer cmd, float t) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descSet, 0, nullptr);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &t);
    uint32_t gx = (width + 15) / 16;
    uint32_t gy = (height + 15) / 16;
    vkCmdDispatch(cmd, gx, gy, 1);
}

void ComputePipeline::cleanup(VulkanContext &ctx) {
    if (device == VK_NULL_HANDLE) return;
    if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
    if (descPool) vkDestroyDescriptorPool(device, descPool, nullptr);
    if (descLayout) vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
    if (storageView) vkDestroyImageView(device, storageView, nullptr);
    if (storageImage) vkDestroyImage(device, storageImage, nullptr);
    if (storageMemory) vkFreeMemory(device, storageMemory, nullptr);
    device = VK_NULL_HANDLE;
}
