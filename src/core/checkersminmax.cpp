#include "checkersminmax.hpp"

#include <limits>

namespace NETWORK
{
    CheckersMinMax::CheckersMinMax(uint32_t depth, uint32_t board_size)
        : depth(depth), board_size(board_size)
    {}

    std::tuple<float, int32_t, int32_t> CheckersMinMax::findBestMove(const std::vector<float>& board, bool max, uint32_t currentDepth)
    {
        if (currentDepth == depth)
            return {evaluatePosition(board, max), -1, -1};

        std::vector<float> boardclone = board; // so the moves won't be done on the real board
        std::vector<uint32_t> pieces = getPieces(board, max);

        float bestValue = max ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        int32_t chosenMove{-1};
        int32_t chosenPiece{-1};

        bool foundValidMove{false};
                
        for (uint32_t pieceIndex : pieces)
        {
            std::vector<uint32_t> moves = getMovesByPiece(boardclone, pieceIndex, max);
            if (moves.empty()) continue;

            for (uint32_t move : moves)
            {
                // Make the move
                boardclone[move] = boardclone[pieceIndex];
                boardclone[pieceIndex] = 0;

                float value = std::get<0>(findBestMove(boardclone, false, currentDepth + 1));

                if ((max && value > bestValue) || (!max && value < bestValue))
                {
                    bestValue = value;
                    chosenMove = move;
                    chosenPiece = pieceIndex;
                }

                // Undo the move
                boardclone[pieceIndex] = 0;
                boardclone[move] = boardclone[pieceIndex];

                foundValidMove = true; 
            }
        }

        if (!foundValidMove)
            return {evaluatePosition(board, max), -1, -1};

        return {bestValue, chosenMove, chosenPiece};
    }

    float CheckersMinMax::evaluatePosition(const std::vector<float> currentBoard, bool max)
    {
        std::vector<uint32_t> minMoves = getAllMoves(currentBoard, false);
        std::vector<uint32_t> maxMoves = getAllMoves(currentBoard, true);

        float score{0.f};
        score += (maxMoves.size() - minMoves.size()) / 2;

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
                    uint32_t newCurrentIndex = x * board_size + y;

                    int32_t jumpX = x + dx[d] * 2, jumpY = y + dy[d] * 2; 
                    uint32_t jumpIndex = jumpX * board_size + jumpY;
        
                    if (canCapture(board, newCurrentIndex, jumpIndex, pieceOwner1))
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

    std::vector<uint32_t> CheckersMinMax::getMovesByPiece(const std::vector<float> board, uint32_t pieceIndex, bool max)
    {
        std::vector<uint32_t> captures;
        std::vector<uint32_t> moves;

        checkMoves(board, pieceIndex, moves);
        checkCaptures(board, pieceIndex, captures);

        std::vector<uint32_t> allMoves = captures;
        allMoves.insert(allMoves.end(), moves.begin(), moves.end());

        return allMoves;
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
    
} // namespace NETWORK
