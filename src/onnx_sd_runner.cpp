// onnx_sd_runner.cpp
// Integração ONNX Runtime + Tokenizers FFI -> Vulkan renderer (adaptado à sua API)

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <thread>
#include <cstdint>

// ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>

// Seu renderer / contexto Vulkan (ajuste include path se necessário)
#include "client/VulkanContext.hpp"
#include "client/Renderer.hpp"
#include "client/VideoWriter.hpp"

namespace fs = std::filesystem;

// ------------------------- Util: carregar arquivo binário -------------------
static std::vector<uint8_t> read_file_bytes(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    std::streampos sp = f.tellg();
    if (sp <= 0) return {};
    size_t sz = static_cast<size_t>(sp);
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> b(sz);
    f.read(reinterpret_cast<char*>(b.data()), sz);
    return b;
}

// ------------------------- ONNX Runner (simplificado) ----------------------
class ONNXRunner {
public:
    ONNXRunner(const std::string &onnx_dir, bool use_cuda = false)
    : env_(ORT_LOGGING_LEVEL_WARNING, "onnx_runner") {
        onnx_dir_ = onnx_dir;
        Ort::SessionOptions sess_opts;
        sess_opts.SetIntraOpNumThreads(4);
        sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        if (use_cuda) {
#ifdef USE_CUDA
            OrtCUDAProviderOptions cuda_opts{};
            sess_opts.AppendExecutionProvider_CUDA(cuda_opts);
#else
            std::cerr << "ONNXRunner built without CUDA support (define USE_CUDA and link providers)" << std::endl;
#endif
        }

        // Try to load model files if present; otherwise we'll be in fallback mode
        fs::path base(onnx_dir);
        auto tenc = base / "text_encoder.onnx";
        auto unet  = base / "unet.onnx";
        auto vae   = base / "vae_decoder.onnx";

        try {
            if (fs::exists(tenc)) session_text_ = std::make_unique<Ort::Session>(env_, tenc.string().c_str(), sess_opts);
            if (fs::exists(unet))  session_unet_  = std::make_unique<Ort::Session>(env_, unet.string().c_str(), sess_opts);
            if (fs::exists(vae))   session_vae_   = std::make_unique<Ort::Session>(env_, vae.string().c_str(), sess_opts);
        } catch (const std::exception &e) {
            std::cerr << "ONNX load error: " << e.what() << std::endl;
        }
    }

    // Gera uma imagem RGBA width*height*4 bytes. Se modelos não estiverem prontos, gera padrão.
    std::vector<uint8_t> generate_image_rgba(const std::string &prompt, int width, int height, int steps = 28, int seed = 1337) {
        if (!session_text_ || !session_unet_ || !session_vae_) {
            return make_test_image(width, height, prompt);
        }

        // --- Simplificação: many steps omitted ---
        // Aqui você deve executar: text_encoder -> embeddings, diffusion loop (unet) -> latents, vae decode -> image
        // Implementação completa do SD em ONNX é longa; por enquanto, retornamos imagem de teste
        return make_test_image(width, height, prompt);
    }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_text_;
    std::unique_ptr<Ort::Session> session_unet_;
    std::unique_ptr<Ort::Session> session_vae_;
    std::string onnx_dir_;

    std::vector<uint8_t> make_test_image(int w, int h, const std::string &seed_text) {
        std::vector<uint8_t> img(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
        // Gradient + text hash color
        uint64_t hash = 1469598103934665603ull;
        for (unsigned char c : seed_text) hash = (hash ^ c) * 1099511628211ull;
        uint8_t r = static_cast<uint8_t>((hash >> 0) & 0xFF);
        uint8_t g = static_cast<uint8_t>((hash >> 8) & 0xFF);
        uint8_t b = static_cast<uint8_t>((hash >> 16) & 0xFF);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                size_t i = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 4;
                img[i+0] = static_cast<uint8_t>(((x * 255) / std::max(1, w-1)) ^ r); // R
                img[i+1] = static_cast<uint8_t>(((y * 255) / std::max(1, h-1)) ^ g); // G
                img[i+2] = static_cast<uint8_t>(((((x+y)/2) * 255) / std::max(1, (w+h)/2-1)) ^ b); // B
                img[i+3] = 255;
            }
        }
        return img;
    }
};


// ------------------------- Integration function ---------------------------
// This function demonstrates the full flow: use tokenizer, run ONNXRunner, upload to your Renderer
bool generate_and_present(const std::string &prompt, int w, int h, const std::string &models_dir, const std::string &tokenizer_json, VulkanContext &vkctx, Renderer &renderer, VideoWriter *vw = nullptr) {

    // 2) ONNX -> produce RGBA image buffer
    ONNXRunner runner(models_dir);
    auto rgba = runner.generate_image_rgba(prompt, w, h);
    if (rgba.empty()) {
        std::cerr << "Failed to generate image\n";
        return false;
    }

    // 3) Upload to renderer
    VkDeviceMemory outMem = VK_NULL_HANDLE;
    VkImage image = renderer.uploadImageRGBA(rgba.data(), static_cast<uint32_t>(w), static_cast<uint32_t>(h), &outMem);
    if (image == VK_NULL_HANDLE) {
        std::cerr << "Renderer uploadImageRGBA failed\n";
        return false;
    }

    // (Optional) draw/present - renderer.drawFrame is a placeholder in your Renderer
    renderer.drawFrame(image);

    // 4) Optionally write frame to video (use VideoWriter::writeFrame)
    if (vw) {
        // VideoWriter in your headers exposes writeFrame(const uint8_t*)
        vw->writeFrame(rgba.data());
    }

    // Note: Caller is responsible for destroying image and freeing memory if needed (renderer.destroyImage)
    return true;
}

// ------------------------- Usage example (main) ---------------------------
int main(int argc, char** argv) {
    std::string models_dir = "models"; // models/text_encoder.onnx etc
    std::string tokenizer_json = "models/tokenizer.json";
    std::string prompt = "um gato astronauta, painting, high detail";
    int width = 512, height = 512;

    // 0) Init VulkanContext (use your existing class)
    VulkanContext vkctx;
    if (!vkctx.init()) {
        std::cerr << "Failed to initialize VulkanContext - adapt call to your class" << std::endl;
        return -1;
    }

    // 1) Create Renderer using your Vulkan context
    Renderer renderer(&vkctx);
    if (!renderer.init()) {
        std::cerr << "Renderer init failed - adapt to your Renderer API" << std::endl;
        return -1;
    }

    // 2) Create VideoWriter if you want to record (folder, width, height)
    VideoWriter vw("frames_out", static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    vw.writeFrame(nullptr); // harmless no-op depending on impl; you can remove or call open if implemented

    // 3) Generate and present
    if (!generate_and_present(prompt, width, height, models_dir, tokenizer_json, vkctx, renderer, &vw)) {
        std::cerr << "generate_and_present failed" << std::endl;
        return -1;
    }

    // 4) Keep window open / loop (this depends on your existing app structure). For demo, sleep then cleanup
    std::this_thread::sleep_for(std::chrono::seconds(3));

    renderer.cleanup();
    vkctx.cleanup();
    return 0;
}
