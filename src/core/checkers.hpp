#ifndef CHECKERS_GAME_HPP
#define CHECKERS_GAME_HPP

#include <stdint.h>
#include <vector>

#include <imgui/imgui/backends/imgui_impl_opengl3.h>

// 0.0 -> empty spot, 0.5 -> normal piece, 1.0 -> queen
// negative for player/opponent, positive for bot.

class Checkers 
{
public:
    Checkers(const uint32_t board_size, const float tile_size);

    void drawBoard();

    const std::vector<float>& getCurrentBoard() const { return board_state; }
private:
    void initiateBoard();
    void drawPiece(ImDrawList* drawlist, const ImU32 color, const ImVec2 center, const uint32_t id);

    uint32_t board_size{0};
    float tile_size{0};

    std::vector<float> board_state;
    uint32_t pieces_left{24};
};

#endif // CHECKERS_GAME_HPP