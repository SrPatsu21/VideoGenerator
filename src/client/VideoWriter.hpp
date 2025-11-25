#pragma once

#include <string>
#include <cstdint>
#include <vector>


// Simple raw-to-file video writer placeholder
class VideoWriter {
public:
    VideoWriter();
    ~VideoWriter();


    bool open(const std::string& path, uint32_t w, uint32_t h, uint32_t fps);
    void addFrame(const uint8_t* rgbaData); // CPU buffer
    void close();


private:
    std::string path_;
    uint32_t w_ = 0;
    uint32_t h_ = 0;
    uint32_t fps_ = 0;


    // You can wire this into ffmpeg later
    std::vector<std::vector<uint8_t>> frames_;
};