#ifndef CHECKERS_MINMAX
#define CHECKERS_MINMAX

#include <vector>
#include <stdint.h>
#include <functional>
#include <cstdlib>
#include "ntars/base/data.hpp"
#include "board.hpp"

namespace NETWORK
{
    class CheckersMinMax
    {
    public:
        CheckersMinMax(uint32_t depth, Board& board);
        
        std::pair<float, Move> findBestMove(std::vector<float>& board_state, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData, bool max = false, uint32_t currentDepth = 0, float alpha = -1, float beta = -1);

        inline void setNewDepth(uint32_t depth) { this->depth = depth; }
        inline void incrementMoveCount() { moveNumber++; }
        inline void resetMoveCount() { moveNumber = 0; }
    private:
        float evaluatePosition(std::vector<float>& currentBoard, bool max);

        std::vector<float> getTrainingLabel(uint32_t moveIndex);
  
        bool isGameOver(std::vector<float>& board_state, bool max)
        {
            if (moveNumber <= 12) return false;

            const std::vector<Move> possibleMoves = board.getMoves(max, board_state);
            return possibleMoves.size() == 0;
        }    

        Board& board;

        uint32_t moveNumber{0};
        uint32_t depth;
    };
} // namespace NETWORK

#endif // CHECKERS_MINMAX;