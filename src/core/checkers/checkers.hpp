#ifndef CHECKERS_GAME_HPP
#define CHECKERS_GAME_HPP

#include <stdint.h>

#include "board.hpp"
#include <imgui/imgui/backends/imgui_impl_opengl3.h>
#include "checkersminmax.hpp"
#include "checkersbot.hpp"

#include <vector>
#include <map>

class Checkers 
{
public:
    Checkers(BitBoard& board, const float tile_size);

    void drawBoard(Bot& bot);
    void drawInfo(int32_t boardScore, Bot& bot);

    void handleNetworkAction(std::vector<float>& activations, NETWORK::CheckersMinMax& algorithm);
    void handleAction(uint64_t pieceIndex, uint64_t moveIndex);

    inline const int32_t getCellMouseAt(const ImVec2 boardStart) const
    {
        const ImVec2 mousePos = ImGui::GetMousePos();

        if (mousePos.x < boardStart.x || mousePos.x >= boardStart.x + board.getSize() * tile_size ||
            mousePos.y < boardStart.y || mousePos.y >= boardStart.y + board.getSize() * tile_size)
            return -1; 

        uint32_t x = (mousePos.x - boardStart.x) / tile_size;
        uint32_t y = (mousePos.y - boardStart.y) / tile_size;

        return y * board.getSize() + x;
    }

    inline void setNewTileSize(float newSize) { tile_size = newSize; }
private:
    void drawGameOverScreen(Bot& bot);

    void drawPiece(ImDrawList* drawlist, const ImU32 color, ImVec2 center, const uint32_t id);
    void drawCrown(ImDrawList* drawlist, ImVec2 center);

    void drawLeaderboard(Bot& bot);
    void incrementLeaderboard(const std::string name, Bot& bot);

    BitBoard& board;

    // 3 leaderboards for 3 different difficulties
    std::unordered_map<std::string, std::vector<std::pair<std::string, int32_t>>> leaderboard;

    float tile_size{0};
    const float margin = 20.f;
};

#endif // CHECKERS_GAME_HPP