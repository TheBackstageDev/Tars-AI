#include "board.hpp"

#include <thread>
#include <algorithm>
#include <string>
#include <iostream>

Board::Board(uint32_t board_size)
    : board_size(board_size)
{
    board_state.resize(board_size * board_size);
    initiateBoard();
}

void Board::initiateBoard()
{
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

void Board::makeMove(const Move &move, std::vector<float> &board_state)
{
    if (!isWithinBounds(move.startPos) && !isWithinBounds(move.endPos))
        return;
        
    board_state[move.endPos] = board_state[move.startPos];
    board_state[move.startPos] = 0;

    for (uint32_t index : move.middlePositions)
        board_state[index] = 0;

    if ((board_state[move.endPos] == -0.5 && getY(move.endPos) == 0) ||           // PLR reaches top row
        (board_state[move.endPos] == 0.5 && getY(move.endPos) == board_size - 1)) // BOT reaches bottom row
    {
        board_state[move.endPos] *= 2;
    }
}

bool Board::canCapture(const Move &move, std::vector<float> &board_state, bool max)
{
    if (!isWithinBounds(move.startPos) || !isWithinBounds(move.endPos) || board_state[move.endPos] != 0)
        return false;

    uint32_t midIndex = GET_MIDDLE(move.startPos, move.endPos);
    float targetPiece = board_state[midIndex];

    if (targetPiece == 0 || (targetPiece > 0 == max))
        return false;

    return isCaptureDistance(move.startPos, move.endPos);
}

std::vector<uint32_t> Board::getPieces(bool max, std::vector<float> &board_state)
{
    std::vector<uint32_t> pieceIndices{};

    for (int32_t index = 0; index < board_state.size(); ++index)
    {
        float current_piece = board_state[index];

        if (current_piece == 0)
            continue;

        if ((max && current_piece > 0) || (!max && current_piece < 0))
            pieceIndices.push_back(index);
    }

    return pieceIndices;
}

std::vector<Move> Board::getMoves(bool max, std::vector<float> &board_state)
{
    std::vector<Move> moves{};

    for (int32_t index = 0; index < board_state.size(); ++index)
    {
        float current_piece = board_state[index];

        if (current_piece == 0)
            continue;

        if ((max && current_piece > 0) || (!max && current_piece < 0))
            if (!isQueen(index))
                checkMoves(index, max, moves, board_state);
            else
                checkMovesQueen(index, max, moves, board_state);
    }

    return moves;
}

void Board::checkCaptures(const uint32_t pieceIndex, bool max, std::vector<Move> &moves, std::vector<float> &board_state, int32_t origin, std::vector<uint32_t> middleIndices)
{
    for (int32_t d : Move::getMoveOffsets())
    {
        const int32_t originIndex = origin != -1 ? origin : pieceIndex;
        const uint32_t targetIndex = pieceIndex + d * 2;
        const int32_t midIndex = GET_MIDDLE(pieceIndex, targetIndex);

        if (targetIndex == originIndex || std::find(middleIndices.begin(), middleIndices.end(), midIndex) != middleIndices.end())
            continue;

        Move moveToCheck{pieceIndex, targetIndex, middleIndices};
        moveToCheck.middlePositions.push_back(midIndex);

        if (canCapture(moveToCheck, board_state, max))
        {
            moveToCheck.startPos = originIndex;

            if (std::find(moves.begin(), moves.end(), moveToCheck) != moves.end())
                continue;

            if (moveToCheck.middlePositions.size() > 1)
                moveToCheck.flag = MoveFlag::MULTICAPTURE;
            else
                moveToCheck.flag = MoveFlag::CAPTURE;

            moves.push_back(moveToCheck);

            checkCaptures(moveToCheck.endPos, max, moves, board_state, originIndex, moveToCheck.middlePositions);
        }
    }
}

void Board::checkMoves(const uint32_t pieceIndex, bool max, std::vector<Move> &moves, std::vector<float> &board_state)
{
    for (int32_t d = 0; d < 2; ++d)
    {
        int32_t direction = Move::getMoveOffsets()[d] * (max ? -1.f : 1.f);

        Move moveToCheck{pieceIndex, pieceIndex + direction};

        if (isWithinBounds(moveToCheck.endPos) && board_state[moveToCheck.endPos] == 0)
        {
            if ((board_state[moveToCheck.endPos] == -0.5 && getY(moveToCheck.endPos) == 0) ||           // PLR reaches top row
                (board_state[moveToCheck.endPos] == 0.5 && getY(moveToCheck.endPos) == board_size - 1)) // BOT reaches bottom row
            {
                moveToCheck.flag = MoveFlag::PROMOTION;
            }
            moves.push_back(moveToCheck);
        }

        checkCaptures(pieceIndex, max, moves, board_state);
    }
}

void Board::checkDirectionsEnabled(const uint32_t pieceIndex, std::vector<int32_t> &directions)
{
    std::array<int32_t, 4> offsets = Move::getMoveOffsets();

    uint32_t row = getY(pieceIndex); // Rows are Y-coordinates
    uint32_t col = getX(pieceIndex); // Columns are X-coordinates

    if (row > 0)
    {
        if (col > 0)
            directions.push_back(offsets[UP_LEFT]);
        if (col < board_size - 1)
            directions.push_back(offsets[UP_RIGHT]);
    }

    if (row < board_size - 1)
    {
        if (col > 0)
            directions.push_back(offsets[DOWN_RIGHT]);
        if (col < board_size - 1)
            directions.push_back(offsets[DOWN_LEFT]);
    }
}

void Board::checkMovesQueen(const uint32_t pieceIndex, bool max, std::vector<Move> &moves, std::vector<float> &board_state)
{
    std::vector<int32_t> directions;
    checkDirectionsEnabled(pieceIndex, directions);

    for (int32_t d : directions)
    {
        int32_t currentPos = pieceIndex;
        bool foundPiece = false;

        while (true)
        {
            currentPos += d;

            uint32_t newRow = getY(currentPos);
            uint32_t newCol = getX(currentPos);

            if (!isWithinBounds(currentPos))
                break;

            if (board_state[currentPos] == 0 && !foundPiece)
            {
                moves.emplace_back(Move{pieceIndex, static_cast<uint32_t>(currentPos)});
            }
            else
            {
                uint32_t captureCheckPos = currentPos - d;
                uint32_t capturePos = currentPos + d;

                if (canCapture(Move{captureCheckPos, capturePos}, board_state, max))
                {
                    Move captureOne{pieceIndex, capturePos, {GET_MIDDLE(captureCheckPos, capturePos)}};

                    captureOne.flag = MoveFlag::CAPTURE;
                    moves.push_back(captureOne);

                    checkCaptures(capturePos, max, moves, board_state);
                    checkCaptures(capturePos, max, moves, board_state, pieceIndex, captureOne.middlePositions);
                }

                foundPiece = true;
                break;
            }

            if (newRow <= 0 || newRow >= board_size - 1 || newCol <= 0 || newCol >= board_size - 1)
                break;
        }
    }
}