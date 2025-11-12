#include "VideoWriter.hpp"
#include <stdexcept>
#include <iostream>

void VideoWriter::open(const std::string& filename, int w, int h, int fps_) {
    width = w; height = h; fps = fps_;
    // ffmpeg reads raw RGBA frames from stdin and encodes to H264 mp4 (yuv420p)
    // -preset ultrafast/veryfast trades compression for speed
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -y -f rawvideo -pixel_format rgba -video_size %dx%d -framerate %d -i - "
             "-c:v libx264 -preset veryfast -pix_fmt yuv420p %s",
             width, height, fps, filename.c_str());
    ffmpeg = popen(cmd, "w");
    if (!ffmpeg) throw std::runtime_error("Failed to open ffmpeg pipe (make sure ffmpeg is on PATH)");
}

void VideoWriter::writeFrameRGBA(const void* rgbaData, size_t bytes) {
    if (!ffmpeg) return;
    size_t wrote = fwrite(rgbaData, 1, bytes, ffmpeg);
    if (wrote != bytes) {
        std::cerr << "Warning: ffmpeg short write " << wrote << " / " << bytes << std::endl;
    }
}

void VideoWriter::close() {
    if (ffmpeg) {
        fflush(ffmpeg);
        pclose(ffmpeg);
        ffmpeg = nullptr;
    }
}
