#include "checkersminmax.hpp"

#include <limits>
#include <algorithm>
#include <tarsmath/linear_algebra/vector_component.hpp>

#include <string>
#include <iostream>

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

    std::pair<float, Move> CheckersMinMax::findBestMove(std::vector<float>& board_state, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData, bool max, uint32_t currentDepth, float alpha, float beta)
    {
        if (currentDepth == depth)
            return {evaluatePosition(board_state, max), {}};

        std::vector<uint32_t> pieces = board.getPieces(max, board_state);

        float bestValue = max ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        Move chosenMove{0, 0};

        for (uint32_t pieceIndex : pieces)
        {
            std::vector<Move> pieceMoves = board.getMovesForPiece(pieceIndex, board_state);

            for (Move move : pieceMoves)
            {
                std::vector<float> tempBoard = board_state; 
                board.makeMove(move, tempBoard);

                float value = findBestMove(tempBoard, trainingData, !max, currentDepth + 1, alpha, beta).first;

                if ((max && value > bestValue) || (!max && value < bestValue))
                {
                    bestValue = value;
                    chosenMove = move;
                }

                if (max) 
                {
                    alpha = std::max(alpha, value);
                    if (alpha >= beta) break;
                } 
                else 
                {
                    beta = std::min(beta, value);
                    if (beta <= alpha) break;
                } 
            }
        }

        if (chosenMove.endPos != -1) {
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

    const float captureMultiplier = 3.f;
    const float moveCountMultiplier = 2.f;
    const float queenValue = 5.f;
    const uint32_t endGameCount = 6;

    float CheckersMinMax::evaluatePosition(std::vector<float>& board_state, bool max)
    {
        std::vector<Move> maxMoves = board.getMoves(true, board_state);
        std::vector<Move> minMoves = board.getMoves(false, board_state);
    
        float score = ((float)(maxMoves.size()) - (float)(minMoves.size())) * moveCountMultiplier;

        uint32_t maxPieceCount = 0;
        uint32_t minPieceCount = 0;
        uint32_t maxQueenCount = 0;
        uint32_t minQueenCount = 0;
    
        for (int32_t x = 0; x < board.getSize() * board.getSize(); ++x)
        {
            const float currentCell = board_state[x];
    
            if (currentCell == 0)
                continue;

            bool isMaxPiece = currentCell > 0;
            bool isCurrentQueen = abs(currentCell) == 1;

            isMaxPiece ? maxPieceCount++ : minPieceCount++;

            if (isCurrentQueen)
            {
                isMaxPiece ? maxQueenCount++ : minQueenCount++;
                isMaxPiece ? score = queenValue : score -= queenValue;
            }
        }
        
        /* END GAME */
        if (maxPieceCount + minPieceCount <= endGameCount)
        {
            if (maxMoves.size() < 5)
                score += max ? -queenValue : queenValue;
        }
    
        return score;
    }

} // namespace NETWORK
