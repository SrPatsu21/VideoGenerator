// onnx_sd_runner.cpp
// Módulo C++ para rodar inference de Stable Diffusion via ONNX Runtime.
// Este arquivo fornece uma classe ONNXSDRunner que carrega os modelos ONNX do
// text-encoder, unet e vae-decoder, tokeniza prompts usando HuggingFace Tokenizers (C++),
// roda a inferência e devolve imagens RGBA prontas para upload para o seu renderer.
//
// Observações importantes:
// - Este código foi escrito como um módulo integrado: adaptações ao seu projeto
//   (nomes de funções do renderer / video writer) podem ser necessárias.
// - Requer as bibliotecas: ONNX Runtime C++ (ORT), HuggingFace Tokenizers (C++),
//   e uma implementação de linear algebra suportada pela ORT provider (CUDA/CPU).
// - O scheduler de difusão está implementado aqui de forma simplificada (DDIM-like)
//   como exemplo. Para produção, substitua por um scheduler preciso (DDPM/PLMS/Euler/etc.)
//
// Uso (exemplo):
//   ONNXSDRunner runner("models/text_encoder.onnx","models/unet.onnx","models/vae_decoder.onnx", true /*use_cuda*/);
//   std::vector<uint8_t> rgba = runner.generate("um gato astronauta", 512, 512, 28, 1);
//   // agora rgba contém width*height*4 bytes

#include <onnxruntime_cxx_api.h>
#include <tokenizers_cpp/tokenizers_cpp.h> // suposto path para Tokenizers C++ (ajuste conforme instalação)

#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <random>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace fs = std::filesystem;

// Helper simples para verificar caminhos
static void ensure_exists(const std::string &path) {
    if (!fs::exists(path)) {
        throw std::runtime_error("Arquivo não encontrado: " + path);
    }
}

class ONNXSDRunner {
public:
    ONNXSDRunner(const std::string& text_encoder_path,
                 const std::string& unet_path,
                 const std::string& vae_decoder_path,
                 bool use_cuda = true)
        : env_(ORT_LOGGING_LEVEL_WARNING, "onnx_sd")
    {
        ensure_exists(text_encoder_path);
        ensure_exists(unet_path);
        ensure_exists(vae_decoder_path);

        Ort::SessionOptions sess_opts;
        sess_opts.SetIntraOpNumThreads(4);
        sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (use_cuda) {
#ifdef USE_CUDA
            OrtCUDAProviderOptions cuda_opts;
            sess_opts.AppendExecutionProvider_CUDA(cuda_opts);
#else
            std::cerr << "Aviso: build sem CUDA. Recompile com ORT CUDA provider e defina -DUSE_CUDA." << std::endl;
#endif
        }

        // Criar sessões ONNX
        session_text_ = std::make_unique<Ort::Session>(env_, text_encoder_path.c_str(), sess_opts);
        session_unet_ = std::make_unique<Ort::Session>(env_, unet_path.c_str(), sess_opts);
        session_vae_  = std::make_unique<Ort::Session>(env_, vae_decoder_path.c_str(), sess_opts);

        // Inicializar tokenizer (assume um tokenizer salvo em JSON)
        // Você precisa de um tokenizer compatível com o text-encoder (ex: CLIP/RoBERTa BPE)
        // Coloque o arquivo tokenizer.json em `models/tokenizer.json` ou ajuste o caminho.
        std::string tokenizer_path = "models/tokenizer.json";
        if (fs::exists(tokenizer_path)) {
            tokenizer_ = tokenizers::Tokenizer::from_file(tokenizer_path);
        } else {
            std::cerr << "Tokenizer não encontrado em 'models/tokenizer.json'. Tokenização poderá falhar." << std::endl;
            tokenizer_ = nullptr;
        }

        // Capturar alocador e env
        allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();
    }

    // Gera `num_images` imagens (em batch seqencial) a partir do prompt
    // Retorna vetor concatenado de bytes RGBA (cada imagem contigua). Cada pixel 0..255.
    std::vector<uint8_t> generate(const std::string& prompt, int width, int height, int steps = 28, int num_images = 1, int seed = -1) {
        if (!session_text_ || !session_unet_ || !session_vae_) {
            throw std::runtime_error("Sessões ONNX não inicializadas");
        }

        // Tokenize prompt
        std::vector<int64_t> input_ids = tokenize_prompt(prompt);
        if (input_ids.empty()) {
            throw std::runtime_error("Tokenização retornou vazia. Verifique tokenizer.json e compatibilidade.");
        }

        // Executar text encoder -> embeddings
        std::vector<float> text_embeddings = run_text_encoder(input_ids);

        // Para cada imagem no batch (aqui rodamos sequencialmente para simplicidade)
        std::vector<uint8_t> out_all;
        for (int img = 0; img < num_images; ++img) {
            // Gerar latents aleatórios
            std::vector<float> latents = make_initial_latents(width, height, seed < 0 ? (int)std::random_device{}() : seed + img);

            // Loop do scheduler simplificado
            std::vector<float> final_latents = run_diffusion_loop(latents, text_embeddings, steps);

            // Decodificar latents com VAE
            std::vector<float> image_f = run_vae_decoder(final_latents, width, height);

            // Converter float imagem (-1..1 ou 0..1) para uint8 RGBA
            std::vector<uint8_t> rgba = convert_image_to_rgba(image_f, width, height);
            out_all.insert(out_all.end(), rgba.begin(), rgba.end());
        }

        return out_all;
    }

private:
    // Componentes ONNX Runtime
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_text_;
    std::unique_ptr<Ort::Session> session_unet_;
    std::unique_ptr<Ort::Session> session_vae_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;

    // Tokenizer (HuggingFace Tokenizers C++)
    std::shared_ptr<tokenizers::Tokenizer> tokenizer_;

    // ============================ Tokenização ==================================
    std::vector<int64_t> tokenize_prompt(const std::string& prompt) {
        if (!tokenizer_) {
            std::cerr << "Tokenizer não inicializado — retornando token fall-back (zeros)." << std::endl;
            return std::vector<int64_t>(77, 0); // fallback: 77 tokens zeros (tamanho usual para SD)
        }

        auto encoding = tokenizer_->encode(prompt);
        std::vector<int64_t> ids;
        ids.reserve(encoding.get_ids().size());
        for (auto id : encoding.get_ids()) ids.push_back((int64_t)id);

        // Ajustar comprimento para 77 tokens (padding/trunc)
        if (ids.size() < 77) ids.resize(77, 0);
        else if (ids.size() > 77) ids.resize(77);

        return ids;
    }

    // ====================== Text Encoder (ONNX) ===============================
    std::vector<float> run_text_encoder(const std::vector<int64_t>& input_ids) {
        // Supondo que o text_encoder espera entrada [1, 77] int64
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t,2> shape{1, (int64_t)input_ids.size()};

        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(mem_info, const_cast<int64_t*>(input_ids.data()), input_ids.size(), shape.data(), shape.size());

        const char* input_names[] = { session_text_->GetInputName(0, *allocator_) };
        const char* output_names[] = { session_text_->GetOutputName(0, *allocator_) };

        auto output_tensors = session_text_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        Ort::Value &emb = output_tensors.front();

        // Copiar dados do tensor float
        float* emb_ptr = emb.GetTensorMutableData<float>();
        size_t total = 1;
        auto tinfo = emb.GetTensorTypeAndShapeInfo();
        auto shapes = tinfo.GetShape();
        for (auto s : shapes) total *= s;

        std::vector<float> embeddings(total);
        std::memcpy(embeddings.data(), emb_ptr, total * sizeof(float));

        // liberar nomes alocados pelo API
        allocator_->Free(const_cast<char*>(input_names[0]));
        allocator_->Free(const_cast<char*>(output_names[0]));

        return embeddings;
    }

    // ====================== UNet single-step =================================
    // O UNet ONNX normalmente espera: latents, timestep, text_embeddings
    // Aqui implementamos uma chamada simples; os nomes de inputs/outputs variam entre ONNX export
    std::vector<float> run_unet(const std::vector<float>& latents, int64_t timestep, const std::vector<float>& text_embeddings) {
        // Obter shapes esperados a partir da sessão
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Exemplo de shapes: [1, 4, H/8, W/8] para latents. Precisamos saber H/8 e W/8.
        // Para manter o exemplo simples, vamos inferir o shape dos inputs através da API.

        // Preparar input tensors (este trecho deve ser adaptado ao seu ONNX export)
        // Nome dos inputs pode ser algo como: "sample", "timestep", "encoder_hidden_states"
        const char* input_names[3];
        input_names[0] = session_unet_->GetInputName(0, *allocator_);
        input_names[1] = session_unet_->GetInputName(1, *allocator_);
        input_names[2] = session_unet_->GetInputName(2, *allocator_);

        // latents tensor
        // pegar shape do input 0 para preparar shape dinamicamente
        auto in0_type = session_unet_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape0 = in0_type.GetShape();
        size_t latents_elems = 1;
        for (auto s : shape0) {
            if (s <= 0) s = 1; // fallback se dimensionamento dinâmico
            latents_elems *= s;
        }

        // criar tensor dos latents (float32)
        Ort::Value lat_tensor = Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(latents.data()), latents.size(), shape0.data(), shape0.size());

        // timestep tensor (float or int, depende do export) - vamos usar float scalar
        float t_f = static_cast<float>(timestep);
        int64_t tshape[1] = {1};
        Ort::Value timestep_tensor = Ort::Value::CreateTensor<float>(mem_info, &t_f, 1, tshape, 1);

        // text embeddings tensor
        // inferir shape output do text encoder para construir tensor correto
        // assumimos shape [1, seq, dim]
        std::vector<int64_t> emb_shape = {1, (int64_t)(text_embeddings.size()/768), 768};
        Ort::Value emb_tensor = Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(text_embeddings.data()), text_embeddings.size(), emb_shape.data(), emb_shape.size());

        Ort::Value* input_tensors[3];
        input_tensors[0] = &lat_tensor;
        input_tensors[1] = &timestep_tensor;
        input_tensors[2] = &emb_tensor;

        const char* output_names[] = { session_unet_->GetOutputName(0, *allocator_) };

        auto outputs = session_unet_->Run(Ort::RunOptions{nullptr}, input_names, input_tensors, 3, output_names, 1);
        Ort::Value &out = outputs.front();

        float* out_ptr = out.GetTensorMutableData<float>();
        size_t out_elems = 1;
        auto out_info = out.GetTensorTypeAndShapeInfo();
        auto out_shape = out_info.GetShape();
        for (auto s : out_shape) out_elems *= s;

        std::vector<float> result(out_elems);
        std::memcpy(result.data(), out_ptr, out_elems * sizeof(float));

        // liberar nomes
        allocator_->Free(const_cast<char*>(input_names[0]));
        allocator_->Free(const_cast<char*>(input_names[1]));
        allocator_->Free(const_cast<char*>(input_names[2]));
        allocator_->Free(const_cast<char*>(output_names[0]));

        return result;
    }

    // ====================== Scheduler simplificado ===========================
    // Implementa um scheduler de difusão muito simples — EULER/PLMS/DPM- adaptáveis.
    // Aqui usamos um loop de passos decrescentes que atualiza latents baseado em predicted noise.
    std::vector<float> run_diffusion_loop(std::vector<float> latents, const std::vector<float>& text_emb, int steps) {
        // Número de timesteps simples: steps -> create a linear schedule
        for (int i = 0; i < steps; ++i) {
            int t = steps - 1 - i; // timestep decrescente
            // Chamar UNet para estimar ruído
            std::vector<float> eps = run_unet(latents, t, text_emb);

            // Atualizar latents com passo de Euler simples (placeholder)
            // latents = latents - lr * eps;  (essa é uma simplificação grosseira)
            float lr = 1.0f / (steps);
            for (size_t k = 0; k < latents.size(); ++k) {
                latents[k] = latents[k] - lr * eps[k];
            }
        }
        return latents;
    }

    // ====================== VAE Decoder ======================================
    std::vector<float> run_vae_decoder(const std::vector<float>& latents, int width, int height) {
        // Converter latents para tensor esperado pelo VAE e executar sessão
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Inferir shape de input do VAE
        const char* in_name = session_vae_->GetInputName(0, *allocator_);
        auto in_info = session_vae_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto in_shape = in_info.GetShape();
        // Ajustar se necessário
        size_t lat_elems = 1;
        for (auto s : in_shape) {
            if (s <= 0) s = 1;
            lat_elems *= s;
        }

        Ort::Value lat_tensor = Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(latents.data()), latents.size(), in_shape.data(), in_shape.size());

        const char* input_names[] = { in_name };
        const char* output_names[] = { session_vae_->GetOutputName(0, *allocator_) };

        auto outputs = session_vae_->Run(Ort::RunOptions{nullptr}, input_names, &lat_tensor, 1, output_names, 1);
        Ort::Value &out = outputs.front();

        float* out_ptr = out.GetTensorMutableData<float>();
        size_t out_elems = 1;
        auto out_info = out.GetTensorTypeAndShapeInfo();
        auto out_shape = out_info.GetShape();
        for (auto s : out_shape) out_elems *= s;

        std::vector<float> img(out_elems);
        std::memcpy(img.data(), out_ptr, out_elems * sizeof(float));

        allocator_->Free(const_cast<char*>(in_name));
        allocator_->Free(const_cast<char*>(output_names[0]));

        return img;
    }

    // ====================== Utils ============================================
    std::vector<float> make_initial_latents(int width, int height, int seed) {
        // Latents typicamente tem shape [1, 4, H/8, W/8]
        int h8 = height / 8;
        int w8 = width / 8;
        size_t elems = (size_t)1 * 4 * h8 * w8;
        std::vector<float> lat(elems);

        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (size_t i = 0; i < elems; ++i) lat[i] = nd(rng);
        return lat;
    }

    std::vector<uint8_t> convert_image_to_rgba(const std::vector<float>& img, int width, int height) {
        // img assumed shape [1, C, H, W] or [H, W, C] — detectar pelo tamanho
        size_t expected_hw3 = (size_t)width * height * 3;
        std::vector<uint8_t> out((size_t)width * height * 4, 0);

        // Tentativa simples: se tamanho corresponde a 3 canais
        if (img.size() >= expected_hw3) {
            // assumir formato CHW (1, 3, H, W) com ordenação RGB
            bool is_chw = (img.size() == (size_t)1 * 3 * height * width);
            if (is_chw) {
                size_t plane = (size_t)height * width;
                const float* r = img.data() + 0 * plane;
                const float* g = img.data() + 1 * plane;
                const float* b = img.data() + 2 * plane;
                for (size_t i = 0; i < plane; ++i) {
                    auto conv = [](float v)->uint8_t {
                        // esperar v em [-1,1] ou [0,1]
                        float x = v;
                        if (x < -1.0f) x = -1.0f;
                        if (x > 1.0f) x = 1.0f;
                        x = (x + 1.0f) * 0.5f; // map -1..1 -> 0..1
                        int iv = static_cast<int>(x * 255.0f + 0.5f);
                        if (iv < 0) iv = 0; if (iv > 255) iv = 255;
                        return (uint8_t)iv;
                    };
                    out[i*4 + 0] = conv(r[i]);
                    out[i*4 + 1] = conv(g[i]);
                    out[i*4 + 2] = conv(b[i]);
                    out[i*4 + 3] = 255;
                }
                return out;
            }
        }

        // fallback: preencher com magenta para evidenciar erro
        for (size_t i = 0; i < out.size(); i += 4) {
            out[i+0] = 255; out[i+1] = 0; out[i+2] = 255; out[i+3] = 255;
        }
        return out;
    }
};

// Fim do arquivo
