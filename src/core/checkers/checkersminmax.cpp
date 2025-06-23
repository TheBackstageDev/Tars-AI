#include "checkersminmax.hpp"
#include <tarsmath/linear_algebra/vector_component.hpp>

#include <string>
#include <iostream>
#include <algorithm>
#include <random>

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

    std::vector<float> CheckersMinMax::getTrainingLabel(uint64_t moveIndex)
    {
        std::vector<float> emptyBoard(board.getSize() * board.getSize(), 0);

        unsigned long rawIndex;
        _BitScanForward64(&rawIndex, moveIndex);

        emptyBoard[rawIndex] = 1.0;

        return emptyBoard;
    }

    BitMove CheckersMinMax::getBestMove(BoardStruct& board_state, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData, bool max, float blunderChance)
    {
        checkedMoves = 0;

        for (int32_t iteration = 1; iteration < depth - 1; ++iteration)
        {
            previousBestMoves.clear();
            minimax(board_state, trainingData, max, 0, iteration);
        }

        auto finalResult = minimax(board_state, trainingData, max, 0, depth);
        BitMove bestMove = finalResult.second;

        if (blunderChance > 0.0f)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            if (dist(gen) < blunderChance)
            {
                std::vector<BitMove> legalMoves = board.getMoves(board_state, max);

                std::vector<BitMove> badChoices;
                for (const auto& move : legalMoves)
                {
                    if (!(move == bestMove))
                        badChoices.push_back(move);
                }

                if (!badChoices.empty())
                {
                    return badChoices[rand() % badChoices.size()];
                }
            }
        }

        return bestMove;
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

        if ((chosenMove.moveMask != 0 || chosenMove.indexMask != 0) && _inserted == true) {
            NTARS::DATA::TrainingData<std::vector<float>> moveData;
            moveData.data = board.vectorBoard(board_state);
            moveData.label = getTrainingLabel(chosenMove.moveMask);

            auto exists = std::find_if(trainingData.begin(), trainingData.end(), 
            [&](const NTARS::DATA::TrainingData<std::vector<float>>& existingData)
            {
                return moveData.data == existingData.data;
            });

            if (exists == trainingData.end())
                trainingData.push_back(std::move(moveData));
        }
        
        return {bestValue, chosenMove};
    }

    const uint32_t captureMultiplier = 5;    
    const uint32_t queenValue = 15;       
    const uint32_t pieceValue = 6;          

    int32_t CheckersMinMax::valueMove(BoardStruct& board_state, const BitMove& move, bool max)
    {
        int32_t score = 0;
        int32_t moveCaptureCount = __popcnt64(move.captureMask); 

        score += (max ? 1 : -1) * moveCaptureCount * captureMultiplier;

        uint64_t promotionColumnMask = !max ? 0xFF00000000000000ULL : 0x00000000000000FFULL;
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

    int32_t CheckersMinMax::evaluateCaptures(BoardStruct& board_state, bool max)
    {
        const auto selfMoves = board.getMoves(board_state, max);
        const auto opponentMoves = board.getMoves(board_state, !max);

        int32_t selfScore = 0;
        int32_t opponentScore = 0;

        auto evaluateMove = [&](const BitMove& move, bool isSelfSide) -> int32_t {
            int32_t score = 0;

            if (move.flag & MoveFlag::CAPTURE || move.flag & MoveFlag::MULTICAPTURE)
            {
                int32_t captured = __popcnt64(move.captureMask);
                score += captured * captureMultiplier;

                if (move.flag == MoveFlag::MULTICAPTURE)
                    score += static_cast<int32_t>(captured * 0.3f * captureMultiplier);
            }

            if (move.flag & MoveFlag::PROMOTION)
            {
                bool wasNotQueen = !(board_state.queenBoard & move.indexMask);
                if (wasNotQueen)
                    score += queenValue / 2;
            }

            return score;
        };

        for (const BitMove& move : selfMoves)
            selfScore += evaluateMove(move, true);

        for (const BitMove& move : opponentMoves)
            opponentScore += evaluateMove(move, false);

        return selfScore - opponentScore;
    }

    const uint32_t endGameCount = 10;

    int32_t CheckersMinMax::evaluateEndGame(BoardStruct& board_state, bool max)
    {
        const int32_t maxMen = __popcnt64(board_state.board_state[max] & ~board_state.queenBoard);
        const int32_t minMen = __popcnt64(board_state.board_state[!max] & ~board_state.queenBoard);
        const int32_t maxQueens = __popcnt64(board_state.board_state[max] & board_state.queenBoard);
        const int32_t minQueens = __popcnt64(board_state.board_state[!max] & board_state.queenBoard);

        const int32_t totalPieces = maxMen + minMen;
        if (totalPieces > endGameCount) return 0 ;

        int32_t endgameScore = 0;

        endgameScore += (maxQueens - minQueens) * (queenValue / 2);

        if (maxQueens > 0 && minQueens == 0 && minMen == 1)
            endgameScore += queenValue;

        if (minQueens > 0 && maxQueens == 0 && maxMen == 1)
            endgameScore -= queenValue;

        const int maxMobility = static_cast<int>(board.getMoves(board_state, max).size());
        const int minMobility = static_cast<int>(board.getMoves(board_state, !max).size());

        if (maxMobility < 3)
            endgameScore -= pieceValue / 2;

        if (minMobility < 3)
            endgameScore += pieceValue / 2;

        return static_cast<int32_t>(endgameScore * moveNumber / (totalPieces + 1));
    }

    int32_t CheckersMinMax::evaluatePosition(BoardStruct& board_state, bool max)
    {
        if (isGameOver(board_state, !max))
            return max ? std::numeric_limits<int32_t>::max() : std::numeric_limits<int32_t>::min();

        int32_t score = 0;

        // Material count
        const int32_t maxMen = __popcnt64(board_state.board_state[max] & ~board_state.queenBoard);
        const int32_t minMen = __popcnt64(board_state.board_state[!max] & ~board_state.queenBoard);
        const int32_t maxQueens = __popcnt64(board_state.board_state[max] & board_state.queenBoard);
        const int32_t minQueens = __popcnt64(board_state.board_state[!max] & board_state.queenBoard);

        score += (maxMen - minMen) * pieceValue;
        score += (maxQueens - minQueens) * queenValue;

        const std::vector<BitMove> maxMoves = board.getMoves(board_state, true);
        const std::vector<BitMove> minMoves = board.getMoves(board_state, false);

        score += static_cast<int32_t>(maxMoves.size() - minMoves.size()) * (pieceValue / 3);

        const uint64_t maxPromoMask = max ? 0xFF00000000000000ULL : 0x00000000000000FFULL;
        const uint64_t minPromoMask = max ? 0x00000000000000FFULL : 0xFF00000000000000ULL;
        const int32_t maxNearPromotion = __popcnt64(board_state.board_state[max] & maxPromoMask);
        const int32_t minNearPromotion = __popcnt64(board_state.board_state[!max] & minPromoMask);

        score += (maxNearPromotion - minNearPromotion) * (queenValue / 2);

        return (max ? 1 : -1) * (score + evaluateCaptures(board_state, max) + evaluateEndGame(board_state, max));
    }

} // namespace NETWORK
