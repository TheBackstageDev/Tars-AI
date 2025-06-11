#ifndef CHECKERS_BOT_HPP
#define CHECKERS_BOT_HPP

#include <string>
#include <vector>

#include <glad/glad.h>
#include <imgui.h>

#include <memory>

#include "core/gl/gltexture.hpp"

enum class SpeechType
{
    Capture,       // When a piece is captured
    MultiCapture,  // When multiple pieces are captured in one move
    GoodMove,      // Praise for a strong strategic move
    BadMove,       // Commentary on a poor move
    Win,           // Bot wins the game
    Lose,          // Bot loses the game
    Taunt,         // Playful or competitive remarks
    Encouragement, // Supportive statements (if losing)
    Surprise,      // Unexpected or shocking move
    Neutral        // General remarks without bias
};

struct BotSpeech
{
    std::string text;
    std::string key; // For the audio

    SpeechType type;
};

struct BotInfo
{
    std::string name = "default";
    std::vector<BotSpeech> speeches;

    uint32_t currentSpeech = 0; // the index

    float blunderChance;
};

class Bot
{
public:
    Bot(BotInfo info, const std::string& path);
    ~Bot() = default;

    Bot(const Bot&) = delete;
    Bot& operator=(const Bot&) = delete;
    Bot(Bot&&) = default;
    Bot& operator=(Bot&&) = default;

    inline std::string getName() { return info.name; }
    inline uint32_t getCurrentSpeech() { return info.currentSpeech; }
    inline float getBlunderChance() { return info.blunderChance; }

    void chooseSpeech(SpeechType type);
    void handleSpeech(int32_t boardScore);
    bool shouldBlunder();

    void drawBot(bool showSpeech = false);

private:

    BotInfo info;
    std::unique_ptr<gl::GLTexture> image;
};

#endif // CHECKERS_BOT_HPP