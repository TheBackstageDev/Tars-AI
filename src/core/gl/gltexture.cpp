#include "gltexture.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <deps/stb_image.h>

#include <iostream>

namespace gl
{
    GLTexture::GLTexture(const std::string& path)
    {
        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);

        uint8_t* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
        if (!data)
        {
            std::cerr << "Failed to load image! " << path;
            return;
        }

        GLenum format = channels == 4 ? GL_RGBA : GL_RGB;
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        stbi_image_free(data);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    GLTexture::~GLTexture()
    {
        glDeleteTextures(1, &textureId);
    }
} // namespace gl
