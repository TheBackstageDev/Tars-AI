#include "checkersminmax.hpp"
#include <tarsmath/linear_algebra/vector_component.hpp>

#include <string>
#include <iostream>
#include <algorithm>

namespace NETWORK
{
    CheckersMinMax::CheckersMinMax(uint32_t depth, Board& board)
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

    Move CheckersMinMax::getBestMove(std::vector<float>& board_state, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData, bool max)
    {
        checkedMoves = 0;

        for (int32_t iteration = 1; iteration < depth - 1; ++iteration)
        {
            previousBestMoves.clear();
            minimax(board_state, trainingData, max, 0, iteration);
        }

        return minimax(board_state, trainingData, max, 0, depth).second;
    }

    std::pair<int32_t, Move> CheckersMinMax::minimax(
        std::vector<float>& board_state, 
        std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData, 
        bool max, uint32_t currentDepth, uint32_t maxDepth, int32_t alpha, int32_t beta)
    {
        if (currentDepth == maxDepth)
            return {evaluatePosition(board_state, max), Move()};

        int32_t bestValue = max ? std::numeric_limits<int32_t>::min() : std::numeric_limits<int32_t>::max();
        std::vector<Move> moves = board.getMoves(max, board_state);

        if (moves.size() == 0)
            return {bestValue, Move()};

        Move chosenMove{};
        sortMoves(board_state, moves);

        bool _inserted{true};
        for (Move& move : moves)
        {
            std::vector<float> tempBoard = board_state; 
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

        if ((chosenMove.endPos != 0 || chosenMove.startPos != 0) && _inserted == true) {
            NTARS::DATA::TrainingData<std::vector<float>> moveData;
            moveData.data = board_state;
            moveData.label = getTrainingLabel(chosenMove.endPos);

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
    const uint32_t endGameCount = 8;       

    int32_t CheckersMinMax::valueMove(std::vector<float>& board_state, const Move& move, bool max)
    {
        int32_t score = 0;
        int32_t moveCaptureCount = move.middlePositions.size();

        score += (max ? 1 : -1) * moveCaptureCount * captureMultiplier;

        float pieceType = board_state[move.startPos];
        bool promotesToQueen = (pieceType == (max ? 0.5f : -0.5f)) && (board.getY(move.endPos) == (max ? board.getSize() - 1 : 0));
        
        if (promotesToQueen)
            score += (max ? 1 : -1) * queenValue * 2;

        std::vector<Move> opponentMoves = board.getMoves(!max, board_state);
        for (const Move& oppMove : opponentMoves)
        {
            if (oppMove.endPos == move.endPos)
                score -= (max ? 1 : -1) * captureMultiplier;
        }

        return score;
    }


    void CheckersMinMax::sortMoves(std::vector<float>& board_state, std::vector<Move>& moves)
    {
        std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
            bool aWasBest = std::find(previousBestMoves.begin(), previousBestMoves.end(), a) != previousBestMoves.end();
            bool bWasBest = std::find(previousBestMoves.begin(), previousBestMoves.end(), b) != previousBestMoves.end();

            if (aWasBest != bWasBest)
                return aWasBest;  

            uint32_t scoreA = valueMove(board_state, a, board_state[a.startPos] > 0);
            uint32_t scoreB = valueMove(board_state, b, board_state[b.startPos] > 0);

            return scoreA > scoreB;
        });
    }

    int32_t CheckersMinMax::evaluatePosition(std::vector<float>& board_state, bool max)
    {
        if (isGameOver(board_state, !max))
            return max ? std::numeric_limits<int32_t>::max() : std::numeric_limits<int32_t>::min();

        std::vector<Move> maxMoves = board.getMoves(true, board_state);
        std::vector<Move> minMoves = board.getMoves(false, board_state);

        int32_t score = 0;
        int32_t maxThreatenedPieces = 0, minThreatenedPieces = 0;
        int32_t maxPieceCount = 0, minPieceCount = 0;
        int32_t maxQueenCount = 0, minQueenCount = 0;
        int32_t maxPotentialQueens = 0, minPotentialQueens = 0;

        for (int32_t x = 0; x < board.getSize() * board.getSize(); ++x)
        {
            float piece = board_state[x];
            if (piece == 0) continue;

            bool isMaxPiece = (piece > 0);
            bool isQueen = std::abs(piece) == 1;
            bool isNearPromotion = (isMaxPiece && board.getY(x) >= board.getSize() - 2) || (!isMaxPiece && board.getY(x) <= 1);

            int32_t pieceValueAdjusted = isQueen ? queenValue * 2 : pieceValue;
            score += (isMaxPiece == max ? pieceValueAdjusted : -pieceValueAdjusted);
            if (isNearPromotion) score += (isMaxPiece == max ? queenValue : -queenValue);

            if (isMaxPiece) {
                maxPieceCount++;
                if (isQueen) maxQueenCount++;
                if (isNearPromotion) maxPotentialQueens++;
            } else {
                minPieceCount++;
                if (isQueen) minQueenCount++;
                if (isNearPromotion) minPotentialQueens++;
            }
        }

        score += (maxPieceCount - minPieceCount) * pieceValue * 2;
        score += (maxQueenCount - minQueenCount) * queenValue * 2;
        score += (maxMoves.size() - minMoves.size()) * pieceValue / 2; 
        
        for (const Move& move : maxMoves)
        {
            int32_t captureScore = move.middlePositions.size() * captureMultiplier;

            if (move.flag == MoveFlag::MULTICAPTURE)
                captureScore *= 1.2;

            score += captureScore;
        }

        for (const Move& move : minMoves)
        {
            int32_t captureScore = move.middlePositions.size() * captureMultiplier;

            if (move.flag == MoveFlag::MULTICAPTURE)
                captureScore *= 1.2;

            score -= captureScore;
        }

        int32_t remainingPieces = maxPieceCount + minPieceCount;
        if (remainingPieces <= endGameCount)
        {
            score += (max ? queenValue : -queenValue) * (maxMoves.size() < 5);
            score += (max ? pieceValue : -pieceValue) * !(maxMoves.size() < 5);
        }

        return score;
    }

} // namespace NETWORK
