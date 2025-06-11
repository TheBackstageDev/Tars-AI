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

void Bot::handleSpeech(int32_t boardScore)
{
    SpeechType type{SpeechType::Neutral};
    bool tendsToDraw = (boardScore >= -10 && boardScore <= 10);

    if (std::rand() % 5 == 0) {
        while (type == SpeechType::Lose || type == SpeechType::Win)
            type = static_cast<SpeechType>(std::rand() % static_cast<int32_t>(SpeechType::Neutral));
    }
    else {
        if (boardScore > 20) {
            type = SpeechType::GoodMove; 
        }
        else if (boardScore < -20) {
            type = SpeechType::BadMove;
        }
        else if (tendsToDraw) {
            type = SpeechType::Encouragement; 
        }
        else {
            type = SpeechType::Neutral; 
        }
    }

    chooseSpeech(type);
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
    auto it = std::find_if(info.speeches.begin(), info.speeches.end(),
        [&](const BotSpeech& s) { return s.text == matchingSpeeches[randomIndex]; });
    info.currentSpeech = static_cast<int32_t>(std::distance(info.speeches.begin(), it));
}

void Bot::drawBot(bool showSpeech)
{
    ImGui::Begin("Bot", nullptr, ImGuiWindowFlags_None);

    ImGui::Image(image->getId(), ImVec2(120, 120)); 

    ImGui::SameLine();
    ImGui::AlignTextToFramePadding(); 

    ImGui::Text("%s", info.name.c_str());

    if (showSpeech)
    {
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.2f, 0.2f, 0.2f, 1.0f)); 
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f)); 

        static char speechBuffer[512] = {0};
        strncpy_s(speechBuffer, info.speeches[info.currentSpeech].text.c_str(), sizeof(speechBuffer) - 1);

        ImGui::InputTextMultiline("##botSpeech", speechBuffer, sizeof(speechBuffer), ImVec2(350, 120), ImGuiInputTextFlags_ReadOnly);

        ImGui::PopStyleColor(2);
    }

    ImGui::End();
}
