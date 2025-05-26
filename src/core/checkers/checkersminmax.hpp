#ifndef CHECKERS_MINMAX
#define CHECKERS_MINMAX

#include <vector>
#include <stdint.h>
#include <functional>
#include <cstdlib>
#include <limits>
#include "ntars/base/data.hpp"
#include "board.hpp"

namespace NETWORK
{
    class CheckersMinMax
    {
    public:
        CheckersMinMax(uint32_t depth, Board& board);

        inline void setNewDepth(uint32_t depth) { this->depth = depth; }
        inline void incrementMoveCount() { moveNumber++; }
        inline void resetMoveCount() { moveNumber = 0; }
        inline uint32_t getCheckedMoveCount() { return checkedMoves; }

        Move getBestMove(std::vector<float>& board_state, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData, bool max);
    private:

        std::pair<int32_t, Move> minimax(std::vector<float>& board_state, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData,
             bool max = true, uint32_t currentDepth = 0, int32_t alpha = -std::numeric_limits<int32_t>::max(), int32_t beta = std::numeric_limits<int32_t>::max());
        int32_t evaluatePosition(std::vector<float>& currentBoard, bool max);
        std::vector<float> getTrainingLabel(uint32_t moveIndex);

        int32_t valueMove(std::vector<float>& board_state, const Move& move);
        void sortMoves(std::vector<float>& board_state, std::vector<Move>& moves);
  
        bool isGameOver(std::vector<float>& board_state, bool max)
        {
            if (moveNumber <= 12) return false;

            const std::vector<Move> possibleMoves = board.getMoves(max, board_state);
            return possibleMoves.size() == 0;
        }    

        Board& board;

        uint32_t moveNumber{0};
        uint32_t depth;
        uint32_t checkedMoves{0};
    };
} // namespace NETWORK

#endif // CHECKERS_MINMAX;