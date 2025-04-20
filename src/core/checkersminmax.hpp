#ifndef CHECKERS_MINMAX
#define CHECKERS_MINMAX

#include <vector>
#include <stdint.h>
#include <functional>
#include <cstdlib>

namespace NETWORK
{
    class CheckersMinMax
    {
    public:
        CheckersMinMax(uint32_t depth, uint32_t board_size);
        
        std::tuple<float, int32_t, int32_t> findBestMove(const std::vector<float>& board, bool max = true, uint32_t currentDepth = 0);
    private:
        float evaluatePosition(const std::vector<float> currentBoard, bool max);

        std::vector<uint32_t> getMovesByPiece(const std::vector<float> board, uint32_t pieceIndex, bool max);
        std::vector<uint32_t> getAllMoves(const std::vector<float> board, bool max);

        // Imported From Checkers.hpp
        bool isMoveLegal(uint32_t x, uint32_t y, const std::vector<float> board) { return isWithinBounds(x, y) && board[x * board_size + y] == 0; }
        bool isWithinBounds(uint32_t x, uint32_t y) { return x < board_size && y < board_size; }    
        bool isQueen(uint32_t pieceIndex, const std::vector<float> board) { return std::abs(board[pieceIndex]) == 1; }

        bool canCapture(const std::vector<float> board, uint32_t pieceIndex, uint32_t moveIndex, bool pieceOwner);
        void checkCaptures(const std::vector<float> board, uint32_t pieceIndex, std::vector<uint32_t>& captures, int dir = -2, int32_t pieceOwner = -1);
        void checkMoves(const std::vector<float> board, uint32_t pieceIndex, std::vector<uint32_t>& moves, int dir = -2);

        uint32_t getMiddle(uint32_t index1, uint32_t index2) { return (index1 + index2) / (board_size * 2) * board_size + ((index1 % board_size + index2 % board_size) / 2); }

        std::vector<uint32_t> getPieces(const std::vector<float> board, bool max);

        uint32_t board_size{1};
        uint32_t depth{1};
    };
} // namespace NETWORK

#endif // CHECKERS_MINMAX;