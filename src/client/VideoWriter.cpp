#include "VideoWriter.hpp"
#include "stb_image_write.h"
#include <sstream>
#include <filesystem>

VideoWriter::VideoWriter(const std::string& folder, uint32_t w, uint32_t h)
    : folder_(folder), width_(w), height_(h), frameIndex_(0)
{
    std::filesystem::create_directories(folder_);
}

VideoWriter::~VideoWriter() {}

std::string VideoWriter::writeFrame(const uint8_t* rgbaData) {
    std::ostringstream ss;
    ss << folder_ << "/frame_" << frameIndex_++ << ".png";
    std::string path = ss.str();

    stbi_write_png(path.c_str(), width_, height_, 4, rgbaData, width_ * 4);
    return path;
}
