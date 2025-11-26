#pragma once
#include <string>
#include <vector>
#include <cstdint>

class VideoWriter {
public:
    VideoWriter(const std::string& folder, uint32_t w, uint32_t h);
    ~VideoWriter();

    std::string writeFrame(const uint8_t* rgbaData);

private:
    std::string folder_;
    uint32_t width_;
    uint32_t height_;
    int frameIndex_;
};
