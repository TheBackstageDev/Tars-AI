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

bool Board::handleQueenCaptures(const Move &move, std::vector<float>& board_state)
{
    std::vector<float> captureSaving{};
    std::vector<Move> captures{};

    uint32_t currentCaptureIndex = move.startPos;
    checkCaptures(currentCaptureIndex, board_state[currentCaptureIndex] > 0, captures, board_state);

    while (!captures.empty() && currentCaptureIndex != move.endPos)
    {
        uint32_t nextMove = captures.front().endPos;
        captures.erase(captures.begin());

        int32_t dx = (nextMove / board_size > currentCaptureIndex / board_size) ? 0 : 1;
        int32_t dy = (nextMove % board_size > currentCaptureIndex % board_size) ? 0 : 1;

        int32_t x = currentCaptureIndex / board_size;
        int32_t y = currentCaptureIndex % board_size;

        while (isWithinBounds(x * board_size + y) && (x != move.endPos / board_size || y != move.endPos % board_size))
        {
            uint32_t currentIndex = x * board_size + y;

            if (board_state[currentIndex] != 0)
            {
                captureSaving.push_back(board_state[currentIndex]);
                board_state[currentIndex] = 0;
                break;
            }

            x += dx;
            y += dy;
        }

        checkCaptures(currentCaptureIndex, board_state[currentCaptureIndex] > 0, captures, board_state);
    }

    if (captureSaving.size() > 0)
    {
        moveHistory.emplace_back(HistoryEntry{move, captureSaving});
        return true;
    }

    return false;
}

bool Board::handleBoardCaptures(const Move &move, std::vector<float>& board_state)
{
    std::vector<float> captureSaving;
    uint32_t dist = distance(move.startPos, move.endPos);

    if (isCaptureDistance(move.startPos, move.endPos))
    {
        uint32_t midIndex = (move.startPos + move.endPos) / 2;
        captureSaving.push_back(board_state[midIndex]);

        board_state[midIndex] = 0;
        moveHistory.emplace_back(HistoryEntry{move, captureSaving});

        return true;
    }
    else if (dist > 2)
    {
        std::vector<Move> captures{};

        uint32_t currentCaptureIndex = move.startPos;
        checkCaptures(currentCaptureIndex, board_state[currentCaptureIndex] > 0, captures, board_state            );

        while (!captures.empty() && currentCaptureIndex != move.endPos)
        {
            uint32_t nextMove = captures.front().endPos;
            captures.erase(captures.begin());

            uint32_t midIndex = (currentCaptureIndex + nextMove) / 2;
            captureSaving.push_back(board_state[midIndex]);

            board_state[midIndex] = 0;

            currentCaptureIndex = nextMove;

            checkCaptures(currentCaptureIndex, board_state[currentCaptureIndex] > 0, captures, board_state);
        }

        moveHistory.emplace_back(HistoryEntry{move, captureSaving});
    }
    else if (dist > 2 && isQueen(move.startPos))
    {
        return handleQueenCaptures(move, board_state);
    }

    return false;
}

void Board::makeMove(const Move &move, std::vector<float>& board_state)
{
    if (!handleBoardCaptures(move, board_state))
        moveHistory.emplace_back(HistoryEntry{move, {}});

    board_state[move.endPos] = board_state[move.startPos];
    board_state[move.startPos] = 0;

    if ((board_state[move.endPos] == -0.5 && move.endPos % board_size == 0) ||           // PLR reaches top row
        (board_state[move.endPos] == 0.5 && move.endPos % board_size == board_size - 1)) // BOT reaches bottom row
    {
        board_state[move.endPos] *= 2;
    }
}

bool Board::canCapture(const Move &move, std::vector<float>& board_state, bool max)
{
    if (!isWithinBounds(move.startPos) || !isWithinBounds(move.endPos) || board_state[move.endPos] != 0)
        return false;

    uint32_t midIndex = (move.startPos + move.endPos) / 2;
    float targetPiece = board_state[midIndex];

    if (targetPiece == 0 || (targetPiece > 0 == max))
        return false;

    return isCaptureDistance(move.startPos, move.endPos);
}

std::vector<uint32_t> Board::getPieces(bool max, std::vector<float>& board_state)
{
    std::vector<uint32_t> pieceIndices{};

    for (int32_t x = 0; x < board_size; ++x)
    {
        for (int32_t y = 0; y < board_size; ++y)
        {
            uint32_t cellIndex = x * board_size + y;
            float current_piece = board_state[cellIndex];

            if (current_piece == 0)
                continue;

            pieceIndices.push_back(cellIndex);
        }
    }

    return pieceIndices;
}

std::vector<Move> Board::getMoves(bool max, std::vector<float>& board_state)
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
                checkMoves(cellIndex, max, moves, board_state);
            else
                checkMovesQueen(cellIndex, max, moves, board_state);
        }
    }

    return moves;
}

void Board::checkCaptures(const uint32_t pieceIndex, bool max, std::vector<Move> &moves, std::vector<float>& board_state, int32_t startIndex)
{
    for (int32_t d : Move::getMoveOffsets())
    {
        const uint32_t targetIndex = pieceIndex + d * 2;

        if (startIndex != -1 && targetIndex == startIndex)
            continue;

        int32_t originIndex = startIndex != -1 ? startIndex : pieceIndex;
        Move moveToCheck{pieceIndex, targetIndex};

        if (canCapture(moveToCheck, board_state ,max) && std::find(moves.begin(), moves.end(), moveToCheck) == moves.end())
        {
            moveToCheck.startPos = originIndex;
            moves.push_back(moveToCheck);

            checkCaptures(moveToCheck.endPos, max, moves, board_state, pieceIndex);
        }
    }
}

void Board::checkMoves(const uint32_t pieceIndex, bool max, std::vector<Move> &moves, std::vector<float>& board_state)
{
    for (int32_t d = 0; d < 2; ++d)
    {
        int32_t direction = Move::getMoveOffsets()[d] * (max ? -1.f : 1.f);

        Move moveToCheck{pieceIndex, pieceIndex + direction};

        if (isWithinBounds(moveToCheck.endPos) && board_state[moveToCheck.endPos] == 0)
            moves.push_back(moveToCheck);

        checkCaptures(pieceIndex, max, moves, board_state);
    }
}

void Board::checkMovesQueen(const uint32_t pieceIndex, bool max, std::vector<Move> &moves, std::vector<float>& board_state)
{
    for (int32_t d : Move::getMoveOffsets())
    {
        int32_t currentPos = pieceIndex;
        bool foundPiece = false;

        // std::cout << "DIR: " + std::to_string(d) << " Dir X: " + std::to_string(d / board_size) << " Dir Y: " + std::to_string(d % board_size) << std::endl;

        while (true)
        {   
            currentPos += d;
            
            std::cout << "X: " + std::to_string(currentPos / board_size) << " Y: " + std::to_string(currentPos % board_size) << std::endl;
            //std::this_thread::sleep_for(std::chrono::milliseconds(500));

            if (!isWithinBounds(currentPos))
                break;

            if (board_state[currentPos] == 0 && !foundPiece)
            {
                moves.emplace_back(Move{pieceIndex, static_cast<uint32_t>(currentPos)});
            }
            else
                foundPiece = true;
        }
    }
}