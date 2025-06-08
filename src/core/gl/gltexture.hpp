#ifndef CORE_GL_TEXTURE
#define CORE_GL_TEXTURE

#include <glad/glad.h>
#include <stdint.h>
#include <string>

namespace gl
{
    class GLTexture
    {
    public:
        GLTexture(const std::string& path);
        ~GLTexture();

        inline void bind() const { glBindTexture(GL_TEXTURE_2D, textureId); }
        
        inline int32_t getWidth() const { return width; }
        inline int32_t getHeight() const { return height; }
        inline int32_t getChannels() const { return channels; }
        inline int32_t getId() const { return textureId; }
    private:
        GLuint textureId;
        int32_t width, height, channels;
    };
} // namespace gl

#endif // CORE_GL_TEXTURE_CORE