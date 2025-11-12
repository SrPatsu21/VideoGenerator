#pragma once
#include <cstdio>
#include <string>
#include <cstdint>

struct VideoWriter {
    FILE* ffmpeg = nullptr;
    int width = 0, height = 0, fps = 30;
    void open(const std::string& filename, int w, int h, int fps_ = 30);
    void writeFrameRGBA(const void* rgbaData, size_t bytes);
    void close();
};
