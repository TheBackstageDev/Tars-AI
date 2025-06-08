#include "checkersbot.hpp"

#include <cstdlib>
#include <ctime>

Bot::Bot(BotInfo info, const std::string& path)
    : info(info)
{
    image = std::make_unique<gl::GLTexture>(path);
}

bool Bot::shouldBlunder()
{
    return (std::rand() % 100) < static_cast<int32_t>(info.blunderChance * 100);
}

void Bot::chooseSpeech(SpeechType type)
{
    std::vector<std::string> matchingSpeeches;

    for (const auto& speech : info.speeches)
    {
        if (speech.type == type)
            matchingSpeeches.push_back(speech.text);
    }

    if (matchingSpeeches.empty())
        info.currentSpeech = 0;

    static bool seeded = false;
    if (!seeded)
    {
        std::srand(std::time(nullptr));
        seeded = true;
    }

    int32_t randomIndex = std::rand() % matchingSpeeches.size();
    info.currentSpeech = randomIndex;
}

void Bot::drawBot(bool showSpeech)
{
    ImGui::Begin("Bot", nullptr, ImGuiWindowFlags_None);

    ImGui::Image(image->getId(), ImVec2(100, 100));

    ImGui::SameLine();

    ImGui::Text("%s", info.name.c_str());

    if (showSpeech)
    {
        ImGui::Separator();
        char speechBuffer[512] = {0};
        strncpy_s(speechBuffer, info.speeches[info.currentSpeech].text.c_str(), sizeof(speechBuffer) - 1);

        ImGui::InputTextMultiline("##botSpeech", speechBuffer, sizeof(speechBuffer), ImVec2(300, 100), ImGuiInputTextFlags_ReadOnly);
    }

    ImGui::End();
}