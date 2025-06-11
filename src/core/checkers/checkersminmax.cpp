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
        for (int32_t iteration = 1; iteration < depth - 1; ++iteration)
            minimax(board_state, trainingData, max, 0, iteration);

        return minimax(board_state, trainingData, max, 0, depth).second;
    }

    std::pair<int32_t, Move> CheckersMinMax::minimax(
        std::vector<float>& board_state, 
        std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData, 
        bool max, uint32_t currentDepth, uint32_t maxDepth, int32_t alpha, int32_t beta)
    {
        if (currentDepth == maxDepth)
            return {evaluatePosition(board_state, max), Move()};

        if (currentDepth == 0)
            checkedMoves = 0;

        int32_t bestValue = max ? -std::numeric_limits<int32_t>::max() : std::numeric_limits<int32_t>::max();
        std::vector<Move> moves = board.getMoves(max, board_state);

        if (moves.size() == 0)
            return {bestValue, Move()};

        Move chosenMove{};
        sortMoves(board_state, moves);

        bool _inserted{true};

        for (Move move : moves)
        {
            std::vector<float> tempBoard = board_state; 
            board.makeMove(move, tempBoard, true);

            checkedMoves++;

            auto [iter, inserted] = permutations.try_emplace(tempBoard, 0);

            if (!inserted)
            {
                _inserted = false;
                int32_t value = iter->second;

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

            iter->second = minimax(tempBoard, trainingData, !max, currentDepth + 1, maxDepth, alpha, beta).first; 

            int32_t value = iter->second;

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
    const uint32_t multiCaptureBonus = 3;
    const uint32_t positionMultiplier = 2;
    const uint32_t queenValue = 7;
    const uint32_t pieceValue = 1;
    const uint32_t endGameCount = 10;

    int32_t CheckersMinMax::valueMove(std::vector<float>& board_state, const Move& move, const bool max)
    {
        int32_t score{0};

        int32_t moveCaptureCount = move.middlePositions.size();
        score += (max ? 1 : -1) * moveCaptureCount * captureMultiplier; 

        if (moveCaptureCount > 1) 
            score += (max ? 1 : -1) * multiCaptureBonus * moveCaptureCount * 2; 

        if (board_state[move.startPos] == (max ? 0.5f : -0.5f) && board.getY(move.endPos) == board.getSize() - 1) 
            score += (max ? 1 : -1) * queenValue * 2; 
        else if (board_state[move.startPos] == (max ? -0.5f : 0.5f) && board.getY(move.endPos) == 0)
            score -= (max ? 1 : -1) * queenValue * 4;

        std::vector<Move> opponentMoves = board.getMoves(!max, board_state);
        for (const Move& oppMove : opponentMoves)
        {
            if (oppMove.endPos == move.endPos) 
                score -= (max ? 1 : -1) * captureMultiplier * 1.5f; 
        }

        return score;    
    }

    void CheckersMinMax::sortMoves(std::vector<float>& board_state, std::vector<Move>& moves)
    {
        std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
            uint32_t scoreA = valueMove(board_state, a, board_state[a.startPos] > 0);
            uint32_t scoreB = valueMove(board_state, b, board_state[b.startPos] > 0);

            return scoreA > scoreB;
        });
    }

    int32_t CheckersMinMax::evaluatePosition(std::vector<float>& board_state, bool max)
    {
        if (isGameOver(board_state, !max))
            return max ? std::numeric_limits<int32_t>::min() : std::numeric_limits<int32_t>::max();

        std::vector<Move> maxMoves = board.getMoves(true, board_state);
        std::vector<Move> minMoves = board.getMoves(false, board_state);

        int32_t score = 0;

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
                score += isCurrentQueen ? queenValue * 4 : pieceValue;
            }
            else 
            {
                minPieceCount++;
                if (isCurrentQueen) minQueenCount++;
                score -= isCurrentQueen ? queenValue * 6 : pieceValue;
            }
        }

        for (const Move& move : maxMoves)
        {
            int32_t moveCaptureCount = move.middlePositions.size();
            maxCapturedPieces += moveCaptureCount;

            for (const auto& pos : move.middlePositions)
            {
                float capturedPiece = board_state[pos];
                bool isQueen = std::abs(capturedPiece) == 1;

                score += isQueen ? queenValue * 4 : pieceValue * 2; // Stronger capture bonus for queens
            }

            if (moveCaptureCount > 1) 
                score += multiCaptureBonus * moveCaptureCount * 2; 
        }

        for (const Move& move : minMoves)
        {
            int32_t moveCaptureCount = move.middlePositions.size();
            minCapturedPieces += moveCaptureCount;

            for (const auto& pos : move.middlePositions)
            {
                float capturedPiece = board_state[pos];
                bool isQueen = std::abs(capturedPiece) == 1;

                score -= isQueen ? queenValue * 6 : pieceValue * 3;
            }

            if (moveCaptureCount > 1) 
                score -= multiCaptureBonus * moveCaptureCount * 4;
        }

        score += (maxCapturedPieces - minCapturedPieces) * (captureMultiplier * 2);

        int32_t remainingPieces = maxPieceCount + minPieceCount;
        if (remainingPieces <= endGameCount)
        {
            score += maxMoves.size() < 5 ? (max ? queenValue * 3 : -queenValue * 3) : (max ? pieceValue * 3 : -pieceValue * 3);
        }

        return max ? score : -score; 
    }

} // namespace NETWORK
