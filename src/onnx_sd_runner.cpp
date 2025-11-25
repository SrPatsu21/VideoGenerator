// onnx_vulkan_integration.cpp
// INTEGRAÇÃO: ONNX Runtime + Tokenizers FFI -> seu motor Vulkan existente
// ------------------------------------------------------------
// Objetivo: fornecer um módulo C++ pronto para integrar a geração ONNX (Stable Diffusion
// via ONNX Runtime) e o tokenizer (Rust FFI) ao seu renderer Vulkan já existente.
// Isso NÃO reescreve toda a stack Vulkan — ele usa as classes que você já tem
// (VulkanContext, Renderer, VideoWriter). Os paths para os seus arquivos que você
// carregou no projeto estão listados abaixo (use esses arquivos no seu build):
//
//   /mnt/data/main.cpp
//   /mnt/data/VulkanContext.cpp
//   /mnt/data/Renderer.cpp
//   /mnt/data/ComputePipeline.cpp
//   /mnt/data/VideoWriter.cpp
//
// Coloque este arquivo em src/client/ (ou onde preferir) e adicione ao seu CMake.
// Ele implementa:
//  - wrapper simples que chama o tokenizer (via tokenizers.h)
//  - um runner ONNX (usa ONNX Runtime C++ API) que carrega sessões e produz uma
//    imagem RGBA em memória (vector<uint8_t>)
//  - função de integração que converte essa imagem para a textura do seu Renderer
//
// NOTAS IMPORTANTES (leia antes de compilar):
//  - Este módulo exige que os modelos ONNX existam em models/: text_encoder.onnx,
//    unet.onnx, vae_decoder.onnx. Se não existir, o runner gera uma imagem de teste.
//  - Você já compilou tokenizers como cdylib e colocou include/tokenizers.h e lib/libtokenizers.so
//  - ONNX Runtime já deve estar disponível em external/onnxruntime (include + lib)
//  - Adapte nomes de métodos do Renderer/VulkanContext/VideoWriter conforme sua API real.
//
// Build: certifique-se que seu CMake inclui ONNX Runtime includes/libs e o include
// external/tokenizers/include. Example CMake additions shown no final do arquivo.
// ------------------------------------------------------------

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <chrono>

// ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>

// Tokenizers C header (lib criada com cdylib + ffi.rs)
#include "tokenizers.h" // deve estar em external/tokenizers/include

#include "client/VulkanContext.hpp"
#include "client/Renderer.hpp"
#include "client/VideoWriter.hpp"

namespace fs = std::filesystem;

// ------------------------- Util: carregar arquivo binário -------------------
static std::vector<uint8_t> read_file_bytes(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    size_t sz = (size_t)f.tellg();
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
        std::vector<uint8_t> img((size_t)w*h*4);
        // Gradient + text hash color
        uint32_t hash = 1469598103934665603u;
        for (char c : seed_text) hash = (hash ^ (unsigned char)c) * 1099511628211u;
        uint8_t r = (hash >> 0) & 0xFF;
        uint8_t g = (hash >> 8) & 0xFF;
        uint8_t b = (hash >> 16) & 0xFF;
        for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
            size_t i = (y*w + x) * 4;
            img[i+0] = (uint8_t)((x * 255) / std::max(1, w-1)) ^ r; // R
            img[i+1] = (uint8_t)((y * 255) / std::max(1, h-1)) ^ g; // G
            img[i+2] = (uint8_t)((((x+y)/2) * 255) / std::max(1, (w+h)/2-1)) ^ b; // B
            img[i+3] = 255;
        }
        return img;
    }
};

// ------------------------- Tokenizer thin wrapper --------------------------
class TokenizerHandle {
public:
    TokenizerHandle(const std::string &tokenizer_json_path) {
        handle_ = tokenizer_load(tokenizer_json_path.c_str());
        if (!handle_) std::cerr << "Warning: tokenizer_load returned null" << std::endl;
    }
    ~TokenizerHandle() {
        // no destroy function in minimal header? we added tokenizer_destroy earlier
        tokenizer_destroy(handle_);
    }
    std::vector<int> encode_ids(const std::string &text) {
        char *out = tokenizer_encode(handle_, text.c_str());
        if (!out) return {};
        std::string s(out);
        tokenizer_free_string(out);
        // parse comma separated ids
        std::vector<int> ids;
        size_t start = 0;
        while (start < s.size()) {
            size_t pos = s.find(',', start);
            std::string token = (pos==std::string::npos) ? s.substr(start) : s.substr(start, pos-start);
            if (!token.empty()) ids.push_back(std::stoi(token));
            if (pos==std::string::npos) break;
            start = pos+1;
        }
        return ids;
    }
private:
    TokenizerHandle()=default;
    void* handle_{nullptr};
};

// ------------------------- Integration function ---------------------------
// This function demonstrates the full flow: use tokenizer, run ONNXRunner, upload to your Renderer
bool generate_and_present(const std::string &prompt, int w, int h, const std::string &models_dir, const std::string &tokenizer_json, VulkanContext &vkctx, Renderer &renderer, VideoWriter *vw = nullptr) {
    // 1) Tokenize (optional, shown for pipeline completeness)
    TokenizerHandle tokenizer(tokenizer_json);
    auto ids = tokenizer.encode_ids(prompt);
    std::cerr << "Token count: " << ids.size() << std::endl;

    // 2) ONNX -> produce RGBA image buffer
    ONNXRunner runner(models_dir);
    auto rgba = runner.generate_image_rgba(prompt, w, h);
    if (rgba.empty()) {
        std::cerr << "Failed to generate image\n";
        return false;
    }

    // 3) Upload to renderer
    // The Renderer API in your project may be different. Below are suggested calls you must adapt.
    // Try to find methods like: renderer.uploadTextureFromCPU(data, width, height) or createTextureFromHost

    if (renderer.uploadImageRGBA) {
        // If your Renderer class exposes a helper (pseudo-code). Replace with real API.
        renderer.uploadImageRGBA(rgba.data(), w, h);
    } else {
        // Fallback: try to use a staging buffer + copy to texture. We'll call a generic method if present.
        // If not present, see notes under "Adapting to your Renderer" below.
        std::cerr << "Warning: renderer.uploadImageRGBA not available in your Renderer API. You must adapt this call to your renderer.\n";
    }

    // 4) Optionally write frame to video
    if (vw) {
        // assume VideoWriter has addFrameRGBA(const uint8_t* data, int w, int h)
        if (vw->addFrameRGBA) {
            vw->addFrameRGBA(rgba.data(), w, h);
        } else {
            std::cerr << "VideoWriter::addFrameRGBA not found; adapt call to your VideoWriter implementation" << std::endl;
        }
    }

    return true;
}

// ------------------------- Usage example (main) ---------------------------
// NOTE: this main shows how to call generate_and_present using your existing VulkanContext/Renderer
// Replace initialization with your app's flow (GLFW window, ImGui setup etc.)

int main(int argc, char** argv) {
    std::string models_dir = "models"; // models/text_encoder.onnx etc
    std::string tokenizer_json = "models/tokenizer.json";
    std::string prompt = "um gato astronauta, painting, high detail";
    int width = 512, height = 512;

    // 0) Init VulkanContext (use your existing class)
    VulkanContext vkctx;
    if (!vkctx.initialize()) {
        std::cerr << "Failed to initialize VulkanContext - adapt call to your class" << std::endl;
        return -1;
    }

    // 1) Create Renderer using your Vulkan context
    Renderer renderer(&vkctx);
    if (!renderer.init()) {
        std::cerr << "Renderer init failed - adapt to your Renderer API" << std::endl;
        return -1;
    }

    // 2) Create VideoWriter if you want to record
    VideoWriter vw(&vkctx);
    // adapt open params as needed

    // 3) Generate and present
    if (!generate_and_present(prompt, width, height, models_dir, tokenizer_json, vkctx, renderer, &vw)) {
        std::cerr << "generate_and_present failed" << std::endl;
        return -1;
    }

    // 4) Keep window open / loop (this depends on your existing app structure). For demo, sleep then cleanup
    std::this_thread::sleep_for(std::chrono::seconds(3));

    renderer.shutdown();
    vkctx.shutdown();
    return 0;
}


/*
CMake snippets (add to your CMakeLists.txt):

# ONNX Runtime
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/external/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Tokenizers
set(TOKENIZERS_DIR "${CMAKE_SOURCE_DIR}/external/tokenizers")
include_directories(${TOKENIZERS_DIR}/include)
link_directories(${TOKENIZERS_DIR}/lib)

# link libs to target
target_link_libraries(VideoGenerator
    onnxruntime
    tokenizers
    glfw
    Vulkan::Vulkan
    pthread
    dl
    X11
    glm::glm
    assimp
)

Notes:
- Ensure external/tokenizers/lib/libtokenizers.so exists (from cargo build --release)
- Ensure external/onnxruntime/lib/libonnxruntime.so exists
- Replace renderer.uploadImageRGBA, renderer.init, VulkanContext::initialize, VideoWriter calls with your actual APIs

Adapting to your Renderer (where most work is):
- If your Renderer has a method to create a texture and upload host data, call it with the rgba pointer
- Otherwise implement an upload helper:
  1) create a staging VkBuffer with VK_BUFFER_USAGE_TRANSFER_SRC
  2) map and memcpy the RGBA data into it
  3) record command buffer that transitions destination image layout and copies buffer->image
  4) submit and wait
  5) present via your swapchain or draw a quad sampling that texture

If you want, I can now:
  - A) Generate a concrete `uploadImageRGBA` helper that uses raw Vulkan calls (staging buffer + copy) so you can paste into Renderer.cpp
  - B) Implement a full ONNX inference pipeline (text encoder + UNet + VAE) in C++ calling ONNX Runtime (long, but doable) — note: needs correct ONNX exports
  - C) Produce a concrete CMakeLists patch to add the tokenizers and onnxruntime libs and copy .so to build dir

Escolha A, B ou C e eu executo a opção imediatamente.
*/
