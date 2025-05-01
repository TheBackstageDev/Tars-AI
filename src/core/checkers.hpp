#ifndef CHECKERS_GAME_HPP
#define CHECKERS_GAME_HPP

#include <stdint.h>
#include <vector>
#include <tuple>
#include <set>

#include <imgui/imgui/backends/imgui_impl_opengl3.h>
// 0.0 -> empty spot, 0.5 -> normal piece, 1.0 -> queen
// negative for player/opponent, positive for bot.

#define PLR false
#define BOT true

class Checkers 
{
public:
    Checkers(const uint32_t board_size, const float tile_size);

    void drawBoard();
    void drawInfo();

    const std::vector<float>& getCurrentBoard() const { return board_state; }

    bool handleNetworkAction(std::vector<float>& activations);
    bool handleAction(int32_t pieceIndex, int32_t moveIndex);

    std::vector<uint32_t> getPieces(bool player);
    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> getPossibleMoves(bool player); // player has to be inverted for some reason (don't ask me why);
    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> getPossiblePieceMoves(uint32_t pieceIndex);
    std::vector<uint32_t> getPossiblePieceMovesVector(uint32_t pieceIndex);
    std::vector<uint32_t> getPossibleAllMoves();

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
    inline uint32_t getBoardSize() const { return board_size; }

    inline const bool getTurn() { return currentTurn; }
    inline const uint32_t getAmmountMoves() { return getPossibleAllMoves().size(); }

    bool isGameOver(bool player)
    {
        const std::pair<std::vector<uint32_t>, std::vector<uint32_t>> possibleMoves = getPossibleMoves(player);
        return possibleMoves.first.size() == 0 && possibleMoves.second.size() == 0;
    }
private:
    void initiateBoard();
    void drawGameOverScreen();
    
    bool isQueen(uint32_t pieceIndex) { return std::abs(board_state[pieceIndex]) == 1; }
    bool isMoveLegal(uint32_t x, uint32_t y) { return isWithinBounds(x, y) && board_state[x * board_size + y] == 0; }
    bool isWithinBounds(uint32_t x, uint32_t y) { return x < board_size && y < board_size; }

    bool canCapture(uint32_t pieceIndex, uint32_t moveIndex, bool pieceOwner);
    void checkCaptures(uint32_t pieceIndex, std::vector<uint32_t>& captures, int dir = -2, int32_t pieceOwner = -1);
    void checkMoves(uint32_t pieceIndex, std::vector<uint32_t>& moves, int dir = -2);

    uint32_t getMiddle(uint32_t index1, uint32_t index2) { return (index1 + index2) / (board_size * 2) * board_size + ((index1 % board_size + index2 % board_size) / 2); }

    void drawPiece(ImDrawList* drawlist, const ImU32 color, const ImVec2 center, const uint32_t id);
    void drawCrown(ImDrawList* drawlist, ImVec2 center);

    uint32_t board_size{0};
    float tile_size{0};

    bool currentTurn = PLR;
    const float margin = 20.f;

    std::set<std::pair<uint32_t, uint32_t>> moveHistory;
    std::vector<float> board_state;
};

#endif // CHECKERS_GAME_HPP