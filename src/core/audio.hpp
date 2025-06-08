#ifndef CORE_AUDIO_HPP
#define CORE_AUDIO_HPP

#include <string>
#include <unordered_map>
#include <miniaudio.h>

#include <fstream>

namespace core
{
    class SoundHandle
    {
    public:
        SoundHandle();
        ~SoundHandle();

        static void play(const char* name);
        static void stop(const char* name);

        static void add(const char* name, const char* path);
        static void remove(const char* name);
    private:
        ma_engine_config engineConfig;

        static ma_engine engine;
        static std::unordered_map<std::string, ma_sound> sounds;
    };
} // namespace core

#endif