#include "audio.hpp"

#include <stdexcept>
#include <iostream>

#include <filesystem>

namespace core
{
    std::unordered_map<std::string, ma_sound> SoundHandle::sounds;
    ma_engine SoundHandle::engine;

    SoundHandle::SoundHandle()
    {
        engineConfig = ma_engine_config_init();

        ma_result result = ma_engine_init(&engineConfig, &engine);
        if (result != MA_SUCCESS)
            throw std::runtime_error("Audio didn't init successfully!");
    }
    
    SoundHandle::~SoundHandle()
    {
        for (auto& sound : sounds)
        {
            ma_sound_uninit(&sound.second);
        }

        ma_engine_uninit(&engine);
    }

    void SoundHandle::play(const char* name)
    {
        if (sounds.find(name) == sounds.end()) 
        {
            std::cerr << "Error: Sound '" << name << "' not found!\n";
            return;
        }
        ma_sound_start(&sounds[name]);
    }

    void SoundHandle::stop(const char* name)
    {
        if (sounds.find(name) == sounds.end()) 
        {
            std::cerr << "Error: Sound '" << name << "' not found!\n";
            return;
        }
        ma_sound_stop(&sounds[name]);
    }

    void SoundHandle::add(const char* name, const char* path)
    {
        std::filesystem::path pathToFile = path;

        if(!std::filesystem::exists(path))
        {
            std::cerr << "path " << path << " is invalid.";
            return;
        }

        ma_result result = ma_sound_init_from_file(&engine, path, 0, NULL, NULL, &sounds[name]);
        if (result != MA_SUCCESS) {
            throw std::runtime_error("Failed to add sound!");
        }
    }
} // namespace core
