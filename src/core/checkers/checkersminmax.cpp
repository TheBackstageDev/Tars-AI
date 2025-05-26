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
        return minimax(board_state, trainingData).second;
    }

    std::pair<int32_t, Move> CheckersMinMax::minimax(std::vector<float>& board_state, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData, bool max, uint32_t currentDepth, int32_t alpha, int32_t beta)
    {
        if (currentDepth == depth - 1)
            return {evaluatePosition(board_state, max), Move()};

        if (currentDepth == 0)
            checkedMoves = 0;

        int32_t bestValue = max ? -std::numeric_limits<int32_t>::max() : std::numeric_limits<int32_t>::max();
        std::vector<Move> moves = board.getMoves(max, board_state);

        if (moves.size() == 0)
            return {bestValue, Move()};

        Move chosenMove{};
        sortMoves(board_state, moves);

        for (Move move : moves)
        {
            std::vector<float> tempBoard = board_state; 
            board.makeMove(move, tempBoard);

            checkedMoves++;

            int32_t value = minimax(tempBoard, trainingData, !max, currentDepth + 1, alpha, beta).first;

            if ((max && value > bestValue) || (!max && value < bestValue))
            {
                bestValue = value;
                chosenMove = move;
            }

            if (max)
                alpha = std::max(alpha, value);
            else
                beta = std::min(beta, value);

            if (alpha >= beta)
                break;
        }

        if (chosenMove.endPos != 0 || chosenMove.startPos != 0) {
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
    const uint32_t multiCaptureBonus = 3;
    const uint32_t moveCountMultiplier = 3;
    const uint32_t positionMultiplier = 2;
    const uint32_t queenValue = 7;
    const uint32_t pieceValue = 1;
    const uint32_t endGameCount = 10;

    int32_t CheckersMinMax::valueMove(std::vector<float>& board_state, const Move& move)
    {
        uint32_t score{0};

        score += move.middlePositions.size() * captureMultiplier;
        if (move.middlePositions.size() > 1) 
            score += multiCaptureBonus * move.middlePositions.size(); 

        if ((board_state[move.endPos] == -0.5 && board.getY(move.endPos) == 0)) // PLR Reaches Top Row
        {
            score += queenValue;
        } // Move forms a queen, probably good.
        
        return score;    
    }

    void CheckersMinMax::sortMoves(std::vector<float>& board_state, std::vector<Move>& moves)
    {
        std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
            uint32_t scoreA = valueMove(board_state, a);
            uint32_t scoreB = valueMove(board_state, b);

            return scoreA > scoreB;
        });
    }

    int32_t CheckersMinMax::evaluatePosition(std::vector<float>& board_state, bool max)
    {
        std::vector<Move> maxMoves = board.getMoves(true, board_state);
        std::vector<Move> minMoves = board.getMoves(false, board_state);

        int32_t score = (static_cast<int32_t>(maxMoves.size()) - static_cast<int32_t>(minMoves.size())) * moveCountMultiplier;

        int32_t maxPieceCount = 0, minPieceCount = 0;
        int32_t maxQueenCount = 0, minQueenCount = 0;
        int32_t maxCapturedPieces = 0, minCapturedPieces = 0;

        for (int32_t x = 0; x < board.getSize() * board.getSize(); ++x)
        {
            const float currentCell = board_state[x];

            if (currentCell == 0) continue;

            bool isMaxPiece = currentCell > 0;
            bool isCurrentQueen = std::abs(currentCell) == 1;

            if (isMaxPiece) 
            {
                maxPieceCount++;
                if (isCurrentQueen) maxQueenCount++;
                score += isCurrentQueen ? queenValue * 3 : pieceValue;
            }
            else 
            {
                minPieceCount++;
                if (isCurrentQueen) minQueenCount++;
                score -= isCurrentQueen ? queenValue * 3 : pieceValue;
            }

            int32_t centerDistance = std::abs(static_cast<int32_t>(board.getSize() / 2) - static_cast<int32_t>(board.getX(x))) + std::abs(static_cast<int32_t>(board.getSize() / 2) - static_cast<int32_t>(board.getY(x)));
            float positionWeight = 1.0f / (1.0f + centerDistance); 
            int32_t forwardBonus = isMaxPiece ? board.getY(x) : (board.getSize() - board.getY(x));

            score += (isMaxPiece ? positionWeight : -positionWeight) * positionMultiplier;
        }

        for (const Move& move : maxMoves)
        {
            maxCapturedPieces += move.middlePositions.size();
            if (move.middlePositions.size() > 1) 
                score += multiCaptureBonus * move.middlePositions.size(); 
        }

        for (const Move& move : minMoves)
        {
            minCapturedPieces += move.middlePositions.size();
            if (move.middlePositions.size() > 1) 
                score -= multiCaptureBonus * move.middlePositions.size() * 2;
        }

        score += (maxCapturedPieces - minCapturedPieces) * captureMultiplier;

        int32_t remainingPieces = maxPieceCount + minPieceCount;
        if (remainingPieces <= endGameCount)
        {
            if (maxMoves.size() < 5)
            {
                score += max ? queenValue * 2 : -queenValue * 2;
            }
            else
            {
                score += max ? pieceValue * 2 : -pieceValue * 2;
            }
        }

        return max ? score : -score; 
    }

} // namespace NETWORK
