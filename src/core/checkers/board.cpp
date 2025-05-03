#include "board.hpp"

#include <algorithm>

Board::Board(uint32_t board_size)
    : board_size(board_size)
{
    board_state.resize(board_size * board_size);
    initiateBoard();
}

void Board::initiateBoard()
{
    moveHistory.clear();
    legalMoves.first.clear();
    legalMoves.second.clear();

    for (int x = 0; x < board_size; ++x)
    {
        for (int y = 0; y < board_size; ++y)
        {
            int index = x * board_size + y;
            board_state[index] = 0;

            if ((x + y) % 2 == 1)
            {
                if (y < board_size / 2 - 1)
                {
                    board_state[index] = 0.5;
                }
                else if (y > board_size / 2)
                {
                    board_state[index] = -0.5;
                }
            }
        }
    }
}

bool Board::handleBoardCaptures(const Move& move)
{
    uint32_t distance = std::abs(static_cast<int>(move.startPos - move.endPos));
    
    if (distance >= 2)
    {
        std::vector<float> captures{};

        uint32_t step = (move.endPos - move.startPos) / distance; 
        for (uint32_t pos = move.startPos + step; pos != move.endPos; pos += step)
        {
            captures.push_back(board_state[pos]);
            board_state[pos] = 0; 
        }

        moveHistory.emplace_back(HistoryEntry{move, captures});
        return true;
    }

    return false;
}

void Board::handleBoardUndoCaptures(const HistoryEntry& entry)
{
    uint32_t distance = std::abs(static_cast<int>(entry.move.startPos - entry.move.endPos));
    
    uint32_t step = (entry.move.endPos - entry.move.startPos) / distance; 
    auto& capturedPieces = entry.piecesCaptured;
    uint32_t captureIndex = 0;

    for (uint32_t pos = entry.move.startPos + step; pos != entry.move.endPos; pos += step)
    {
        board_state[pos] = capturedPieces[captureIndex];
        captureIndex++;
    }

    moveHistory.erase(std::find(moveHistory.begin(), moveHistory.end(), entry));
}

void Board::makeMove(const Move& move)
{
    board_state[move.endPos] = board_state[move.startPos];
    board_state[move.startPos] = 0;

    if (!handleBoardCaptures(move))
        moveHistory.emplace_back(HistoryEntry{move, {}});
}

void Board::unmakeMove(const Move& move)
{
    auto entryIndex = std::find(moveHistory.begin(), moveHistory.end(), HistoryEntry{move, {}});
    
    if (entryIndex != moveHistory.end())
    {
        board_state[move.startPos] = board_state[move.endPos];
        board_state[move.endPos] = 0;

        if (!entryIndex->piecesCaptured.empty())
            handleBoardUndoCaptures(*entryIndex);

        moveHistory.erase(entryIndex);
    }
}

bool Board::canCapture(const Move& move, bool max)
{
    if ((!isWithinBounds(move.startPos) || !isWithinBounds(move.endPos)))
        return false;
    
    if (board_state[move.endPos] != 0)
        return false;

    uint32_t midIndex = (move.startPos + move.endPos) / 2;
    float targetPiece = board_state[midIndex];

    if (targetPiece == 0 || (targetPiece > 0 == max))
        return false;
    
    return std::abs(static_cast<int>(move.startPos - move.endPos)) == 2;
}

std::vector<Move> Board::getMoves(bool max)
{
    std::vector<Move> moves{};

    for (int32_t x = 0; x < board_size; ++x)
    {
        for (int32_t y = 0; y < board_size; ++y)
        {
            uint32_t cellIndex = x * board_size + y;
            float current_piece = board_state[cellIndex];
            
            if (current_piece == 0)
                continue;
            
            if (!isQueen(cellIndex))
                checkMoves(cellIndex, max, moves);
            else
                checkMovesQueen(cellIndex, max, moves);
        }
    }

    return moves;
}

void Board::checkCaptures(const uint32_t pieceIndex, bool max, std::vector<Move>& moves)
{
    for (int32_t d = 0; d < 4; ++d)
    {
        int32_t direction = Move::getMoveOffsets()[d];
        Move moveToCheck{pieceIndex, pieceIndex + direction * 2};

        if (canCapture(moveToCheck, max))
        {
            moves.push_back(moveToCheck);
            checkCaptures(moveToCheck.endPos, max, moves);
        }
    }
}

void Board::checkMoves(const uint32_t pieceIndex, bool max, std::vector<Move>& moves)
{
    int32_t moveChecksStart = max ? 2 : 0;

    for (int32_t d = 0; d < 4; ++d)
    {
        int32_t direction = Move::getMoveOffsets()[d];
        
        Move moveToCheck{pieceIndex, pieceIndex + direction};

        if (isWithinBounds(moveToCheck.endPos) && (max ? d >= moveChecksStart : d <= 2) && board_state[moveToCheck.endPos] == 0)
            moves.push_back(moveToCheck);

        checkCaptures(pieceIndex, max, moves);
    }
}

void Board::checkMovesQueen(const uint32_t pieceIndex, bool max, std::vector<Move>& moves)
{
    for (int32_t d = 0; d < 4; ++d)
    {
        uint32_t currentPos = pieceIndex;
        bool foundPiece = false;

        while (true)
        {
            currentPos += Move::getMoveOffsets()[d];

            if (!isWithinBounds(currentPos))
                break;

            checkCaptures(currentPos, max, moves);

            if (board_state[currentPos] == 0 && !foundPiece)
            {
                moves.emplace_back(Move{pieceIndex, currentPos});
            } else {
                foundPiece = false;
            }
        }
    }
}