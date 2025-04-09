#ifndef CHECKERS_GAME_HPP
#define CHECKERS_GAME_HPP

#include <stdint.h>
#include <vector>
#include <tuple>

#include <imgui/imgui/backends/imgui_impl_opengl3.h>
// 0.0 -> empty spot, 0.5 -> normal piece, 1.0 -> queen
// negative for player/opponent, positive for bot.

struct Move
{
    uint32_t moveIndex{0};
    bool player;
};

class Checkers 
{
public:
    Checkers(const uint32_t board_size, const float tile_size);

    void drawBoard();

    const std::vector<float>& getCurrentBoard() const { return board_state; }

    void handleAction(int32_t pieceIndex, int32_t moveIndex);
    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> getPossibleMoves(bool player = false); // player has to be inverted for some reason (don't ask me why);
    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> getPossiblePieceMoves(uint32_t pieceIndex);

    inline const int32_t getCellMouseAt(const ImVec2 boardStart) const
    {
        const ImVec2 mousePos = ImGui::GetMousePos();

        if (mousePos.x < boardStart.x || mousePos.x >= boardStart.x + board_size * tile_size ||
            mousePos.y < boardStart.y || mousePos.y >= boardStart.y + board_size * tile_size)
            return -1; 

        uint32_t x = (mousePos.x - boardStart.x) / tile_size;
        uint32_t y = (mousePos.y - boardStart.y) / tile_size;

        return x * board_size + y;
    }
private:
    void initiateBoard();
    bool isQueen(uint32_t pieceIndex) { return std::abs(board_state[pieceIndex]) == 1; }
    bool isMoveLegal(uint32_t x, uint32_t y) { return x < board_size && y < board_size && board_state[x * board_size + y] == 0; }
    bool canCapture(uint32_t moveIndex, uint32_t currentIndex);

    void drawPiece(ImDrawList* drawlist, const ImU32 color, const ImVec2 center, const uint32_t id);
    void drawCrown(ImDrawList* drawlist, const ImVec2 center);

    uint32_t board_size{0};
    float tile_size{0};

    std::vector<float> board_state;
    uint32_t pieces_left{24};
};

#endif // CHECKERS_GAME_HPP