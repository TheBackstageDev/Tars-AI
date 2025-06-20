#include "checkersminmax.hpp"
#include <tarsmath/linear_algebra/vector_component.hpp>

#include <string>
#include <iostream>
#include <algorithm>

namespace NETWORK
{
    CheckersMinMax::CheckersMinMax(uint32_t depth, BitBoard& board)
        : depth(depth), board(board)
    {
    }

    float distance(TMATH::Vector2 x0, TMATH::Vector2 x1)
    {
        return sqrtf(powf(x1.x - x0.x, 2) + powf(x1.y - x0.y, 2));
    }

    std::vector<float> CheckersMinMax::getTrainingLabel(uint32_t moveIndex)
    {
        std::vector<float> emptyBoard(board.getSize() * board.getSize(), 0);
        emptyBoard[moveIndex] = 1.0;

        return emptyBoard;
    }

    BitMove CheckersMinMax::getBestMove(BoardStruct& board_state, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData, bool max)
    {
        checkedMoves = 0;

        for (int32_t iteration = 1; iteration < depth - 1; ++iteration)
        {
            previousBestMoves.clear();
            minimax(board_state, trainingData, max, 0, iteration);
        }

        return minimax(board_state, trainingData, max, 0, depth).second;
    }

    std::pair<int32_t, BitMove> CheckersMinMax::minimax(
        BoardStruct& board_state, 
        std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData, 
        bool max, uint32_t currentDepth, uint32_t maxDepth, int32_t alpha, int32_t beta)
    {
        if (currentDepth == maxDepth)
            return {evaluatePosition(board_state, max), BitMove()};

        int32_t bestValue = max ? std::numeric_limits<int32_t>::min() : std::numeric_limits<int32_t>::max();
        std::vector<BitMove> moves = board.getMoves(board_state, max);

        if (moves.size() == 0)
            return {bestValue, BitMove()};

        BitMove chosenMove{};
        sortMoves(board_state, moves);

        bool _inserted{true};
        for (BitMove& move : moves)
        {
            BoardStruct tempBoard = board_state; 
            board.makeMove(move, tempBoard, true);
            checkedMoves++;

            auto [iter, inserted] = permutations.try_emplace(tempBoard, std::make_pair(0, maxDepth));

            int32_t value = iter->second.first;
            if (!inserted && iter->second.second > currentDepth)
            {
                _inserted = false;

                if ((max && value > bestValue) || (!max && value < bestValue))
                {
                    bestValue = value;
                    chosenMove = move;
                }

                alpha = max ? std::max(alpha, value) : alpha;
                beta = !max ? std::min(beta, value) : beta;

                if (alpha >= beta) break;
            }

            value = minimax(tempBoard, trainingData, !max, currentDepth + 1, maxDepth, alpha, beta).first;
            iter->second = {value, maxDepth};

            if ((max && value > bestValue) || (!max && value < bestValue))
            {
                bestValue = value;
                chosenMove = move;

                if (currentDepth == 0 && value > 0) 
                {
                    previousBestMoves.push_back(chosenMove);
                }
            }

            alpha = max ? std::max(alpha, value) : alpha;
            beta = !max ? std::min(beta, value) : beta;

            if (alpha >= beta) break;
        }
        
        boardScore = bestValue;

/*         if ((chosenMove.moveMask != 0 || chosenMove.indexMask != 0) && _inserted == true) {
            NTARS::DATA::TrainingData<std::vector<float>> moveData;
            moveData.data = board_state;
            moveData.label = getTrainingLabel(chosenMove.moveMask);

            auto exists = std::find_if(trainingData.begin(), trainingData.end(), 
            [&](const NTARS::DATA::TrainingData<std::vector<float>>& existingData)
            {
                return moveData.data == existingData.data;
            });

            if (exists == trainingData.end())
                trainingData.push_back(std::move(moveData));
        } */
        
        return {bestValue, chosenMove};
    }

    const uint32_t captureMultiplier = 5;    
    const uint32_t queenValue = 15;       
    const uint32_t pieceValue = 6;          
    const uint32_t endGameCount = 8;       

    int32_t CheckersMinMax::valueMove(BoardStruct& board_state, const BitMove& move, bool max)
    {
        int32_t score = 0;
        int32_t moveCaptureCount = __popcnt64(move.captureMask); 

        score += (max ? 1 : -1) * moveCaptureCount * captureMultiplier;

        uint64_t promotionColumnMask = max ? 0x0101010101010101ULL : 0x8080808080808080ULL;
        bool promotesToQueen = (move.moveMask & promotionColumnMask) && !(board_state.queenBoard & move.indexMask);

        if (promotesToQueen)
            score += (max ? 1 : -1) * queenValue * 2;

        uint64_t opponentAttackMask = 0;

        std::vector<BitMove> oppMoves = board.getMoves(board_state, !max);
        for (const BitMove& oppMove : oppMoves) {
            opponentAttackMask |= oppMove.moveMask; 
        }

        if (opponentAttackMask & move.moveMask)
            score -= (max ? 1 : -1) * captureMultiplier;

        return score;
    }

    void CheckersMinMax::sortMoves(BoardStruct& board_state, std::vector<BitMove>& moves)
    {
        std::sort(moves.begin(), moves.end(), [&](const BitMove& a, const BitMove& b) {
            bool aWasBest = std::find(previousBestMoves.begin(), previousBestMoves.end(), a) != previousBestMoves.end();
            bool bWasBest = std::find(previousBestMoves.begin(), previousBestMoves.end(), b) != previousBestMoves.end();

            if (aWasBest != bWasBest)
                return aWasBest && !bWasBest; 

            bool isMaxPieceA = (board_state.board_state[MAX] & a.indexMask) != 0;
            bool isMaxPieceB = (board_state.board_state[MAX] & b.indexMask) != 0;

            uint32_t scoreA = valueMove(board_state, a, isMaxPieceA);
            uint32_t scoreB = valueMove(board_state, b, isMaxPieceB);

            if (scoreA != scoreB)
                return scoreA > scoreB;

            return false; 
        });
    }

    int32_t CheckersMinMax::evaluatePosition(BoardStruct& board_state, bool max)
    {
        if (isGameOver(board_state, !max))
            return max ? std::numeric_limits<int32_t>::max() : std::numeric_limits<int32_t>::min();

        int32_t score = 0;

        // Material evaluation
        const int32_t maxMen = __popcnt64(board_state.board_state[max] & ~board_state.queenBoard);
        const int32_t minMen = __popcnt64(board_state.board_state[!max] & ~board_state.queenBoard);
        const int32_t maxQueens = __popcnt64(board_state.board_state[max] & board_state.queenBoard);
        const int32_t minQueens = __popcnt64(board_state.board_state[!max] & board_state.queenBoard);

        score += (maxMen - minMen) * pieceValue;
        score += (maxQueens - minQueens) * queenValue;

        // Mobility
        std::vector<BitMove> maxMoves = board.getMoves(board_state, true);
        std::vector<BitMove> minMoves = board.getMoves(board_state, false);
        score += static_cast<int32_t>(maxMoves.size() - minMoves.size()) * (pieceValue / 3);

        // Potential king promotions
        const uint64_t maxPromoMask = max ? 0x0101010101010101ULL : 0x8080808080808080ULL;
        const uint64_t minPromoMask = max ? 0x8080808080808080ULL : 0x0101010101010101ULL;
        const int32_t maxNearPromotion = __popcnt64(board_state.board_state[max] & maxPromoMask);
        const int32_t minNearPromotion = __popcnt64(board_state.board_state[!max] & minPromoMask);
        score += (maxNearPromotion - minNearPromotion) * (queenValue / 2);

        // Threats and captures
        uint64_t opponentAttackSquares = 0;
        for (const BitMove& move : minMoves)
            opponentAttackSquares |= move.moveMask;

        for (const BitMove& move : maxMoves)
        {
            if (move.flag == MoveFlag::CAPTURE || move.flag == MoveFlag::MULTICAPTURE)
            {
                int32_t captured = __popcnt64(move.captureMask);
                score += captured * captureMultiplier;

                if (move.flag == MoveFlag::MULTICAPTURE)
                    score += static_cast<int32_t>(captured * 0.2f * captureMultiplier);
            }

            // Penalize risky squares
            if (opponentAttackSquares & move.moveMask)
                score -= pieceValue / 2;
        }

        // Endgame heuristics
        const int32_t totalPieces = maxMen + minMen + maxQueens + minQueens;
        if (totalPieces <= endGameCount)
        {
            const bool lowMobility = maxMoves.size() < 4;
            score += (max ? 1 : -1) * (lowMobility ? -queenValue : queenValue / 2);
        }

        return score;
    }

} // namespace NETWORK
