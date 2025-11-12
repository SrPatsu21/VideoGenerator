#include <iostream>
#include <stdexcept>
#include <cstring>
#include <chrono>
#include <thread>

#include <GLFW/glfw3.h>

#include "client/VulkanContext.hpp"
#include "client/ComputePipeline.hpp"
#include "client/Renderer.hpp"
#include "client/VideoWriter.hpp"

int main() {
    try {
        if (!glfwInit()) throw std::runtime_error("Failed to init GLFW");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow* window = glfwCreateWindow(512, 512, "VideoGenerator", nullptr, nullptr);
        if (!window) throw std::runtime_error("Failed to create GLFW window");

        VulkanContext ctx;
        ctx.init(window);

        ComputePipeline compute;
        // path produced by your CMake shader compile target
        compute.init(ctx.device, ctx.physicalDevice, ctx, "./shaders/compute.glsl.spv");

        Renderer renderer;
        renderer.init(&ctx, &compute);

        VideoWriter vw;
        vw.open("output.mp4", ctx.swapchainExtent.width, ctx.swapchainExtent.height, 30);

        std::cout << "Starting render loop. Close the window to finish and finalize the mp4 file.\n";

        auto start = std::chrono::steady_clock::now();
        const double targetFps = 30.0;
        const std::chrono::duration<double> frameDuration(1.0 / targetFps);

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            auto now = std::chrono::steady_clock::now();
            float t = std::chrono::duration<float>(now - start).count();

            void* mapped = nullptr;
            VkDeviceSize mappedSize = 0;
            bool ok = renderer.renderFrame(t, &mapped, &mappedSize);
            if (!ok) {
                std::cerr << "Frame render failed; breaking\n";
                break;
            }

            // mapped points to staging memory (RGBA8)
            // NOTE: For some GPUs/driver combos you might need to respect row pitch; here we use tightly packed copy (bufferRowLength=0)
            if (mapped && mappedSize > 0) {
                vw.writeFrameRGBA(mapped, (size_t)mappedSize);
            }

            // frame pacing
            std::this_thread::sleep_for(frameDuration);
        }

        std::cout << "Shutting down, finalizing video (this may take a moment)...\n";
        vw.close();

        renderer.cleanup();
        compute.cleanup(ctx);
        ctx.cleanup();

        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }
}
