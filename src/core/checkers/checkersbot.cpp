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

    int roll = std::rand() % 10;

    if (roll < 5)
    {
        type = static_cast<SpeechType>((std::rand() % 2 == 0) ? SpeechType::Taunt : SpeechType::Surprise);
    }
    else if (roll < 15)
    {
        do {
            type = static_cast<SpeechType>(std::rand() % static_cast<int32_t>(SpeechType::Neutral));
        } while (type == SpeechType::Win || type == SpeechType::Lose);
    }
    else
    {
        if (boardScore > 30)
        {
            type = SpeechType::GoodMove;
        }
        else if (boardScore < -30)
        {
            type = SpeechType::BadMove;
        }
        else if (tendsToDraw)
        {
            type = SpeechType::Encouragement;
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
    ImGui::Begin("Bot", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    const float imageSize = 120.0f;
    ImGui::Image(image->getId(), ImVec2(imageSize, imageSize));
    ImGui::SameLine();
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + imageSize * 0.3f);
    ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.3f, 1.0f), "%s", info.name.c_str());

    if (showSpeech && !info.speeches.empty() && info.currentSpeech >= 0 && info.currentSpeech < static_cast<int32_t>(info.speeches.size()))
    {
        ImGui::Spacing();
        ImGui::Dummy(ImVec2(0.0f, 4.0f));

        const char* bubbleText = info.speeches[info.currentSpeech].text.c_str();
        const float bubbleWidth = 360.0f;
        const float bubbleTextPadding = 10.0f;

        ImVec2 textSize = ImGui::CalcTextSize(bubbleText, nullptr, true, bubbleWidth - 2 * bubbleTextPadding);
        ImVec2 bubbleSize = ImVec2(bubbleWidth, textSize.y + 2 * bubbleTextPadding);
        ImVec2 bubblePos = ImGui::GetCursorScreenPos();
        ImVec2 bubbleEnd = ImVec2(bubblePos.x + bubbleSize.x, bubblePos.y + bubbleSize.y);

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImU32 bubbleColor = ImGui::GetColorU32(ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
        ImU32 borderColor = ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, 0.05f));

        drawList->AddRectFilled(bubblePos, bubbleEnd, bubbleColor, 12.0f);
        drawList->AddRect(bubblePos, bubbleEnd, borderColor, 12.0f, 0, 1.0f);

        ImVec2 tailTop[3] = {
            ImVec2(bubblePos.x + 20, bubblePos.y - 10),          // touches bubble top edge
            ImVec2(bubblePos.x + 25, bubblePos.y),     // upper left
            ImVec2(bubblePos.x + 40, bubblePos.y)      // upper right
        };
        drawList->AddConvexPolyFilled(tailTop, 3, bubbleColor);

        ImGui::SetCursorScreenPos(ImVec2(bubblePos.x + bubbleTextPadding, bubblePos.y + bubbleTextPadding));
        ImGui::PushTextWrapPos(bubblePos.x + bubbleWidth - bubbleTextPadding);
        ImGui::TextWrapped(bubbleText);
        ImGui::PopTextWrapPos();

        ImGui::Dummy(ImVec2(0, bubbleSize.y + 5));
    }

    ImGui::End();
}

