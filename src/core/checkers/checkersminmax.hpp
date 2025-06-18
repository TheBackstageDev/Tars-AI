#ifndef CHECKERS_MINMAX
#define CHECKERS_MINMAX

#include <vector>
#include <stdint.h>
#include <functional>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include "ntars/base/data.hpp"
#include "board.hpp"

struct BoardHash {
    std::size_t operator()(const BoardStruct& board) const {
        return std::hash<uint64_t>()(board.occupiedBoard) ^
               (std::hash<uint64_t>()(board.board_state[0]) << 1) ^
               (std::hash<uint64_t>()(board.board_state[1]) << 2) ^
               (std::hash<uint64_t>()(board.queenBoard) << 3);
    }
};

namespace NETWORK
{
    class CheckersMinMax
    {
    public:
        CheckersMinMax(uint32_t depth, BitBoard& board);

        inline void setNewDepth(uint32_t depth) { this->depth = depth; }
        inline void incrementMoveCount() { moveNumber++; }
        inline void resetMoveCount() { moveNumber = 0; }
        inline uint32_t getCheckedMoveCount() { return checkedMoves; }
        inline int32_t getCurrentBoardScore() { return boardScore; }

        void sortMoves(BoardStruct& board_state, std::vector<BitMove>& moves);
        BitMove getBestMove(BoardStruct& board_state, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData, bool max);
    private:

        std::pair<int32_t, BitMove> minimax(BoardStruct& board_state, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData,
             bool max = true, uint32_t currentDepth = 0, uint32_t maxDepth = 1, int32_t alpha = -std::numeric_limits<int32_t>::max(), int32_t beta = std::numeric_limits<int32_t>::max());
        int32_t evaluatePosition(BoardStruct& currentBoard, bool max);
        std::vector<float> getTrainingLabel(uint32_t moveIndex);
        
        int32_t valueMove(BoardStruct& board_state, const BitMove& move, const bool max);
  
        bool isGameOver(BoardStruct& board_state, bool max)
        {
            if (moveNumber <= 12) return false;

            const std::vector<BitMove> possibleMoves = board.getMoves(board_state, max);
            return possibleMoves.size() == 0;
        }    

        // Searching Variables
        std::unordered_map<BoardStruct, std::pair<int32_t, int32_t>, BoardHash> permutations;
        std::vector<BitMove> previousBestMoves;

        BitBoard& board;

        int32_t boardScore{0};
        uint32_t moveNumber{0};
        uint32_t depth;
        uint32_t checkedMoves{0};
    };
} // namespace NETWORK

#endif // CHECKERS_MINMAX;