#ifndef CHECKERS_MINMAX
#define CHECKERS_MINMAX

#include <vector>
#include <stdint.h>
#include <functional>
#include <cstdlib>
#include "ntars/base/data.hpp"

namespace NETWORK
{
    class CheckersMinMax
    {
    public:
        CheckersMinMax(uint32_t depth, uint32_t board_size);
        
        std::tuple<float, int32_t, int32_t> findBestMove(const std::vector<float>& board, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData, bool max = false, uint32_t currentDepth = 0, float alpha = -1, float beta = -1);

        inline void setNewDepth(uint32_t depth) { this->depth = depth; }
        inline void incrementMoveCount() { moveNumber++; }
        inline void resetMoveCount() { moveNumber = 0; }
    private:
        float evaluatePosition(const std::vector<float>& currentBoard, bool max);

        std::vector<uint32_t> getMovesByPiece(const std::vector<float> board, uint32_t pieceIndex);
        std::pair<std::vector<uint32_t>, std::vector<uint32_t>> getMovesByPieceWithCaptures(const std::vector<float> board, uint32_t pieceIndex);
        std::vector<uint32_t> getCapturesByPiece(const std::vector<float> board, uint32_t pieceIndex);
        std::vector<uint32_t> getAllMoves(const std::vector<float> board, bool max);
        std::pair<std::vector<uint32_t>, std::vector<uint32_t>> getAllMovesWithCaptures(const std::vector<float> board, bool max);

        std::vector<float> getTrainingLabel(uint32_t moveIndex);

        // Imported From Checkers.hpp
        bool isMoveLegal(uint32_t x, uint32_t y, const std::vector<float> board) { return isWithinBounds(x, y) && board[x * board_size + y] == 0; }
        bool isWithinBounds(uint32_t x, uint32_t y) { return x < board_size && y < board_size; }    
        bool isQueen(uint32_t pieceIndex, const std::vector<float> board) { return std::abs(board[pieceIndex]) == 1; }

        bool canCapture(const std::vector<float> board, uint32_t pieceIndex, uint32_t moveIndex, bool pieceOwner);
        void checkCaptures(const std::vector<float> board, uint32_t pieceIndex, std::vector<uint32_t>& captures, int dir = -2, int32_t pieceOwner = -1);
        void checkMoves(const std::vector<float> board, uint32_t pieceIndex, std::vector<uint32_t>& moves, int dir = -2);

        uint32_t getMiddle(uint32_t index1, uint32_t index2) { return (index1 + index2) / (board_size * 2) * board_size + ((index1 % board_size + index2 % board_size) / 2); }

        std::vector<uint32_t> getPieces(const std::vector<float> board, bool max);
        void handleBoardCaptures(const uint32_t pieceIndex, const uint32_t moveIndex, std::vector<float>& board, std::vector<std::pair<uint32_t, float>>& undoList);

        bool isGameOver(const std::vector<float>& board, bool player)
        {
            if (moveNumber <= 12) return false;

            const std::vector<uint32_t> possibleMoves = getAllMoves(board, player);
            return possibleMoves.size() == 0;
        }    

        inline void doBoardMove(std::vector<float>& board, uint32_t& pieceIndex, uint32_t& moveIndex)
        {
            board[moveIndex] = board[pieceIndex];
            board[pieceIndex] = 0;
        }

        inline void undoBoardMove(std::vector<float>& board, uint32_t& pieceIndex, uint32_t& moveIndex)
        {
            board[pieceIndex] = board[moveIndex];
            board[moveIndex] = 0;
        }

        inline void undoBoardCaptures(std::vector<float> &board, std::vector<std::pair<uint32_t, float>> &undoStack)
        {
            for (const auto &[index, pieceValue] : undoStack)
                board[index] = pieceValue; 
              
            undoStack.clear(); 
        }

        uint32_t moveNumber{0};
        uint32_t board_size{1};
        uint32_t depth;
    };
} // namespace NETWORK

#endif // CHECKERS_MINMAX;