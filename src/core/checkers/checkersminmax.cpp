#include "checkersminmax.hpp"

#include <limits>
#include <algorithm>
#include <tarsmath/linear_algebra/vector_component.hpp>

#include <string>
#include <iostream>

namespace NETWORK
{
    CheckersMinMax::CheckersMinMax(uint32_t depth, uint32_t board_size)
        : depth(depth), board_size(board_size)
    {
    }

    float distance(TMATH::Vector2 x0, TMATH::Vector2 x1)
    {
        return sqrtf(powf(x1.x - x0.x, 2) + powf(x1.y - x0.y, 2));
    }

    void CheckersMinMax::handleBoardCaptures(const uint32_t pieceIndex, const uint32_t moveIndex, std::vector<float> &board, std::vector<std::pair<uint32_t, float>>& undoList)
    {
        int currentX = pieceIndex / board_size;
        int currentY = pieceIndex % board_size;
        int moveX = moveIndex / board_size;
        int moveY = moveIndex % board_size;

        int32_t dist = distance(TMATH::Vector2(currentX, currentY), TMATH::Vector2(moveX, moveY));
        std::vector<uint32_t> capturesPossible = getCapturesByPiece(board, pieceIndex);

        uint32_t currentCaptureIndex = pieceIndex;

        if (dist > 2)
        {
            for (size_t i = 0; i < capturesPossible.size() && currentCaptureIndex != moveIndex; ++i)
            {
                uint32_t nextMove = capturesPossible[i];

                if (isQueen(pieceIndex, board)) 
                {
                    int dx = (nextMove / board_size > currentCaptureIndex / board_size) ? 1 : -1;
                    int dy = (nextMove % board_size > currentCaptureIndex % board_size) ? 1 : -1;

                    for (int x = currentX + dx, y = currentY + dy; x != moveX || y != moveY; x += dx, y += dy)
                    {
                        uint32_t currentIndex = x * board_size + y;
                        if (!isWithinBounds(x, y)) break;

                        if (board[currentIndex] != 0) 
                        {
                            undoList.emplace_back(currentIndex, board[currentIndex]);
                            board[currentIndex] = 0;
                            break;
                        }
                    }
                }
                else 
                {
                    uint32_t midIndex = getMiddle(currentCaptureIndex, nextMove);
                    undoList.emplace_back(midIndex, board[midIndex]);
                    board[midIndex] = 0;
                    currentCaptureIndex = nextMove;
                }

                capturesPossible = getCapturesByPiece(board, currentCaptureIndex); 
            }
        }
        else 
        {
            uint32_t midIndex = getMiddle(pieceIndex, moveIndex);
            undoList.emplace_back(midIndex, board[midIndex]);
            board[midIndex] = 0;
        }
    }

    std::vector<float> CheckersMinMax::getTrainingLabel(uint32_t moveIndex)
    {
        std::vector<float> emptyBoard(board_size * board_size, 0);
        emptyBoard[moveIndex] = 1.0;

        return emptyBoard;
    }

    std::tuple<float, int32_t, int32_t> CheckersMinMax::findBestMove(const std::vector<float> &board, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData, bool max, uint32_t currentDepth, float alpha, float beta)
    {
        if (currentDepth == depth)
            return {evaluatePosition(board, max), -1, -1};

        std::vector<float> boardclone = board; // so the moves won't be done on the real board
        std::vector<uint32_t> pieces = getPieces(board, max);

        float bestValue = max ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        int32_t chosenMove{-1};
        int32_t chosenPiece{-1};

        for (uint32_t pieceIndex : pieces)
        {
            std::pair<std::vector<uint32_t>, std::vector<uint32_t>> moves = getMovesByPieceWithCaptures(boardclone, pieceIndex);
            std::vector<uint32_t>& captures = moves.second;
            std::vector<uint32_t>& onlyMoves = moves.first;
            
            if (captures.empty() && onlyMoves.empty())
                continue;

            std::vector<std::pair<uint32_t, float>> undoStack;
            std::vector<uint32_t> allMoves = captures;
            allMoves.insert(allMoves.end(), onlyMoves.begin(), onlyMoves.end());

            for (int32_t i = 0; i < allMoves.size(); ++i)
            {
                uint32_t moveIndex = allMoves[i];
                doBoardMove(boardclone, pieceIndex, moveIndex);  

                if (i < captures.size() - 1)
                    handleBoardCaptures(pieceIndex, moveIndex, boardclone, undoStack);

                if (isGameOver(boardclone, !max))
                    return {max ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity(), pieceIndex, moveIndex};
            
                uint32_t pieceNewY = moveIndex % board_size;
                float& movePiece = boardclone[moveIndex];
        
                if (!isQueen(moveIndex, board) && ((pieceNewY == 0 && movePiece < 0) || // Negative pieces reach top
                    (pieceNewY == board_size - 1 && movePiece > 0))) // Positive pieces reach bottom
                {
                    movePiece = (movePiece > 0) ? 1.f : -1.f; 
                }

                float value = std::get<0>(findBestMove(boardclone, trainingData, !max, currentDepth + 1, alpha, beta));
                
                if (!undoStack.empty())
                    value += static_cast<float>(undoStack.size() * 5.f) * (max ? -1.f : 1.f);

                if ((max && value > bestValue) || (!max && value < bestValue))
                {
                    bestValue = value;
                    chosenMove = moveIndex;
                    chosenPiece = pieceIndex;
                }

                if (movePiece == 1.f || movePiece == -1.f)
                    movePiece = (movePiece > 0) ? 1.f : -1.f;

                undoBoardCaptures(boardclone, undoStack);
                undoBoardMove(boardclone, pieceIndex, moveIndex);

                if (max)
                {
                    alpha = std::max(alpha, bestValue);
                    if (beta <= alpha) break;
                }
                else
                {
                    beta = std::min(beta, bestValue);
                    if (beta <= alpha) break;
                }
            }
        }

        if (chosenMove != -1) {
            NTARS::DATA::TrainingData<std::vector<float>> moveData;
            moveData.data = std::move(boardclone);
            moveData.label = getTrainingLabel(chosenMove);

            auto exists = std::find_if(trainingData.begin(), trainingData.end(), 
            [&](const NTARS::DATA::TrainingData<std::vector<float>>& existingData)
            {
                return moveData.data == existingData.data;
            });

            if (exists == trainingData.end())
                trainingData.push_back(std::move(moveData));
        }
        
        return {bestValue, chosenMove, chosenPiece};
    }

    const float captureMultiplier = 3.f;
    const float moveCountMultiplier = 2.f;
    const float queenValue = 5.f;
    const uint32_t endGameCount = 6;

    float CheckersMinMax::evaluatePosition(const std::vector<float>& currentBoard, bool max)
    {
        std::pair<std::vector<uint32_t>, std::vector<uint32_t>> minMoves = getAllMovesWithCaptures(currentBoard, false);
        std::pair<std::vector<uint32_t>, std::vector<uint32_t>> maxMoves = getAllMovesWithCaptures(currentBoard, true);
    
        float score = ((float)(maxMoves.first.size() + maxMoves.second.size()) - (float)(minMoves.first.size() + minMoves.second.size())) * moveCountMultiplier;
        score += max ? (maxMoves.second.size() * captureMultiplier) : -(minMoves.second.size() * captureMultiplier);

        uint32_t maxPieceCount = 0;
        uint32_t minPieceCount = 0;
        uint32_t maxQueenCount = 0;
        uint32_t minQueenCount = 0;
    
        for (int32_t x = 0; x < board_size; ++x)
        {
            for (int32_t y = 0; y < board_size; ++y)
            {
                const float currentCell = currentBoard[x * board_size + y];
    
                if (currentCell == 0)
                    continue;
    
                bool isMaxPiece = currentCell > 0;
                bool isCurrentQueen = isQueen(x * board_size + y, currentBoard);

                isMaxPiece ? maxPieceCount++ : minPieceCount++;

                if (isCurrentQueen)
                {
                    isMaxPiece ? maxQueenCount++ : minQueenCount++;
                    isMaxPiece ? score = queenValue : score -= queenValue;
                }
            }
        }
        
        /* END GAME */
        if (maxPieceCount + minPieceCount <= endGameCount)
        {
            if (maxMoves.first.size() + maxMoves.second.size() < 5)
                score += max ? -queenValue : queenValue;
        }
    
        return score;
    }
    
    // Imported from Checkers.cpp

    std::vector<uint32_t> CheckersMinMax::getPieces(const std::vector<float> board, bool max)
    {
        std::vector<uint32_t> pieceIndexes;
        for (int32_t x = 0; x < board_size; ++x)
        {
            for (int32_t y = 0; y < board_size; ++y)
            {
                const uint32_t cellIndex = x * board_size + y;
                const float current_cell = board[cellIndex];
                if (current_cell == 0)
                    continue;

                const bool bot_piece = current_cell < 0;

                if (bot_piece && max)
                    continue;
                if (!bot_piece && !max)
                    continue;

                pieceIndexes.emplace_back(cellIndex);
            }
        }
        return pieceIndexes;
    }

    bool CheckersMinMax::canCapture(const std::vector<float> board, uint32_t pieceIndex, uint32_t moveIndex, bool pieceOwner)
    {
        int32_t moveX = moveIndex / board_size;
        int32_t moveY = moveIndex % board_size;

        if (!isMoveLegal(moveX, moveY, board))
            return false;

        int32_t currentX = pieceIndex / board_size;
        int32_t currentY = pieceIndex % board_size;
        int32_t midX = (currentX + moveX) / 2;
        int32_t midY = (currentY + moveY) / 2;
        uint32_t midIndex = midX * board_size + midY;

        float currentPiece = board[pieceIndex];
        float targetPiece = board[midIndex];

        if (targetPiece == 0 || (targetPiece > 0 == pieceOwner))
            return false;

        return abs(moveX - currentX) == 2 && abs(moveY - currentY) == 2;
    }

    void CheckersMinMax::checkCaptures(const std::vector<float> board, uint32_t pieceIndex, std::vector<uint32_t> &captures, int dir, int32_t pieceOwner)
    {
        int32_t currentX = pieceIndex / board_size;
        int32_t currentY = pieceIndex % board_size;
        float piece = board[pieceIndex];
        bool pieceOwner1 = pieceOwner > 0 ? pieceOwner : piece > 0;

        const int32_t direction = dir == -2 ? piece < 0 ? -1 : 1 : dir; // Plr moves up while Bot moves down;

        if (isQueen(pieceIndex, board))
        {
            const int32_t dx[] = {1, -1, 1, -1};
            const int32_t dy[] = {1, 1, -1, -1};
    
            for (int32_t d = 0; d < 4; ++d)
            {
                int32_t x = currentX, y = currentY;
    
                while (isWithinBounds(x, y))
                {
                    uint32_t newIndex = x * board_size + y;
                    uint32_t jumpIndex = (x + dx[d]) * board_size + (y + dy[d]);
        
                    if (canCapture(board, newIndex, jumpIndex, pieceOwner1))
                        captures.emplace_back(jumpIndex); 
    
                    x += dx[d];
                    y += dy[d];
                }
            }
        }
        else
        {
            const int32_t dx[] = {2, -2, 2, -2};
            const int32_t dy[] = {2, 2, -2, -2};

            for (int32_t d = 0; d < 4; ++d)
            {
                int32_t leftJumpX = currentX - dx[d], currentJumpY = currentY + direction * dy[d];
                int32_t rightJumpX = currentX + dx[d];

                int32_t leftJumpIndex = leftJumpX * board_size + currentJumpY;
                int32_t rightJumpIndex = rightJumpX * board_size + currentJumpY;

                if (canCapture(board, pieceIndex, leftJumpIndex, pieceOwner1) && std::find(captures.begin(), captures.end(), leftJumpIndex) == captures.end())
                {
                    captures.emplace_back(leftJumpIndex);
                    checkCaptures(board, leftJumpIndex, captures, direction, pieceOwner1);
                }

                if (canCapture(board, pieceIndex, rightJumpIndex, pieceOwner1) && std::find(captures.begin(), captures.end(), rightJumpIndex) == captures.end())
                {
                    captures.emplace_back(rightJumpIndex);
                    checkCaptures(board, rightJumpIndex, captures, direction, pieceOwner1);
                }
            }
        }
    }

    void CheckersMinMax::checkMoves(const std::vector<float> board, uint32_t pieceIndex, std::vector<uint32_t> &moves, int dir)
    {
        float piece = board[pieceIndex];

        if (piece == 0)
            return;

        const int32_t direction = dir == -2 ? piece < 0 ? -1 : 1 : dir;

        int32_t currentX = pieceIndex / board_size;
        int32_t currentY = pieceIndex % board_size;
        int32_t leftX = currentX + 1, leftY = currentY + direction;
        int32_t rightX = currentX - 1, rightY = currentY + direction;

        uint32_t leftIndex = leftX * board_size + leftY;
        uint32_t rightIndex = rightX * board_size + rightY;

        if (isQueen(pieceIndex, board))
        {
            const int32_t dx[] = {1, -1, 1, -1};
            const int32_t dy[] = {1, 1, -1, -1};

            for (int32_t d = 0; d < 4; ++d)
            {
                int32_t x = currentX, y = currentY;

                while (true)
                {
                    x += dx[d];
                    y += dy[d];

                    if (!isMoveLegal(x, y, board))
                        break;

                    uint32_t moveIndex = x * board_size + y;
                    moves.emplace_back(moveIndex);
                }
            }
        }
        else
        {
            if (isMoveLegal(leftX, leftY, board))
                moves.emplace_back(leftIndex);

            if (isMoveLegal(rightX, rightY, board))
                moves.emplace_back(rightIndex);
        }
    }

    std::vector<uint32_t> CheckersMinMax::getMovesByPiece(const std::vector<float> board, uint32_t pieceIndex)
    {
        std::vector<uint32_t> captures;
        std::vector<uint32_t> moves;

        checkMoves(board, pieceIndex, moves);
        checkCaptures(board, pieceIndex, captures);

        std::vector<uint32_t> allMoves = captures;
        allMoves.insert(allMoves.end(), moves.begin(), moves.end());

        return allMoves;
    }

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> CheckersMinMax::getMovesByPieceWithCaptures(const std::vector<float> board, uint32_t pieceIndex)
    {
        std::vector<uint32_t> captures;
        std::vector<uint32_t> moves;

        checkMoves(board, pieceIndex, moves);
        checkCaptures(board, pieceIndex, captures);

        return std::pair{moves, captures};
    }

    std::vector<uint32_t> CheckersMinMax::getCapturesByPiece(const std::vector<float> board, uint32_t pieceIndex)
    {
        std::vector<uint32_t> captures;
        checkCaptures(board, pieceIndex, captures);

        return captures;
    }

    std::vector<uint32_t> CheckersMinMax::getAllMoves(const std::vector<float> board, bool max)
    {
        std::vector<uint32_t> captures;
        std::vector<uint32_t> moves;

        for (int32_t x = 0; x < board_size; ++x)
        {
            for (int32_t y = 0; y < board_size; ++y)
            {
                const uint32_t cellIndex = x * board_size + y;
                const float current_cell = board[cellIndex];

                if (current_cell == 0)
                    continue;

                const bool bot_piece = current_cell < 0;

                if (bot_piece && max)
                    continue;
                if (!bot_piece && !max)
                    continue;

                checkMoves(board, cellIndex, moves);
                checkCaptures(board, cellIndex, captures);
            }
        }

        std::vector<uint32_t> allMoves = captures;
        allMoves.insert(allMoves.end(), moves.begin(), moves.end());

        return allMoves;
    }

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> CheckersMinMax::getAllMovesWithCaptures(const std::vector<float> board, bool max)
    {
        std::vector<uint32_t> captures;
        std::vector<uint32_t> moves;

        for (int32_t x = 0; x < board_size; ++x)
        {
            for (int32_t y = 0; y < board_size; ++y)
            {
                const uint32_t cellIndex = x * board_size + y;
                const float current_cell = board[cellIndex];

                if (current_cell == 0)
                    continue;

                const bool bot_piece = current_cell < 0;

                if (bot_piece && max)
                    continue;
                if (!bot_piece && !max)
                    continue;

                checkMoves(board, cellIndex, moves);
                checkCaptures(board, cellIndex, captures);
            }
        }

        return std::pair{moves, captures};
    }

} // namespace NETWORK
