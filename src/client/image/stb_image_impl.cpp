#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

unsigned char* loadTexture(const char* filename, int* width, int* height, int* channels) {
    return stbi_load(filename, width, height, channels, STBI_rgb_alpha);
}