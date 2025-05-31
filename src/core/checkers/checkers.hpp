#ifndef CHECKERS_GAME_HPP
#define CHECKERS_GAME_HPP

#include <stdint.h>

#include "board.hpp"
#include <imgui/imgui/backends/imgui_impl_opengl3.h>
#include "checkersminmax.hpp"

class Checkers 
{
public:
    Checkers(Board& board, const float tile_size);

    void drawBoard();
    void drawInfo(int32_t boardScore);

    void handleNetworkAction(std::vector<float>& activations, NETWORK::CheckersMinMax& algorithm);
    void handleAction(int32_t pieceIndex, int32_t moveIndex);

    inline const int32_t getCellMouseAt(const ImVec2 boardStart) const
    {
        const ImVec2 mousePos = ImGui::GetMousePos();

        if (mousePos.x < boardStart.x || mousePos.x >= boardStart.x + board.getSize() * tile_size ||
            mousePos.y < boardStart.y || mousePos.y >= boardStart.y + board.getSize() * tile_size)
            return -1; 

        uint32_t x = (mousePos.x - boardStart.x) / tile_size;
        uint32_t y = (mousePos.y - boardStart.y) / tile_size;

        return x * board.getSize() + y;
    }
private:
    void drawGameOverScreen();

    void drawPiece(ImDrawList* drawlist, const ImU32 color, const ImVec2 center, const uint32_t id);
    void drawCrown(ImDrawList* drawlist, ImVec2 center);

    Board& board;

    float tile_size{0};
    const float margin = 20.f;
};

#endif // CHECKERS_GAME_HPP