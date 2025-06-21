#include "board.hpp"

#include <thread>
#include <algorithm>
#include <string>
#include <iostream>

#include "core/audio.hpp"

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

void Board::makeMove(Move &move, std::vector<float> &board_state, bool isMinimax)
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
        move.flag = MoveFlag::PROMOTION;
    }

    if (!isMinimax)
        switch(move.flag)
        {
            // Just a normal Move
            case MoveFlag::NONE:
            {   
                core::SoundHandle::play("move");
                break;
            }
            case MoveFlag::CAPTURE:
            case MoveFlag::MULTICAPTURE:
            {
                core::SoundHandle::play("capture");
                break;
            }
            case MoveFlag::PROMOTION:
            {
                board_state[move.endPos] *= 2;
                core::SoundHandle::play("promote");
                break;
            }
            default:
                break;
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
            if ((board_state[pieceIndex] == -0.5 && getY(moveToCheck.endPos) == 0) ||           // PLR reaches top row
                (board_state[pieceIndex] == 0.5 && getY(moveToCheck.endPos) == board_size - 1)) // BOT reaches bottom row
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

// Bit board

#define BOARD_SIZE 64
#define BOARD_DIM 8 

BitBoard::BitBoard()
{
    initiateBoard();
}

void BitBoard::initiateBoard()
{
    currentPieceIndices.clear();
    board.board_state[MIN] = 0;
    board.board_state[MAX] = 0;
    board.queenBoard = 0; 
    board.occupiedBoard = 0;

    for (int row = 0; row < BOARD_DIM; ++row)  
    {
        for (int col = 0; col < BOARD_DIM; ++col)  
        {
            if ((row + col) % 2 == 1) 
            {
                int index = row * BOARD_DIM + col; 

                if (row < 3)  
                {
                    board.board_state[MAX] |= (1ULL << index);
                }
                else if (row >= BOARD_DIM - 3)  
                {
                    board.board_state[MIN] |= (1ULL << index);
                }
            }
        }
    }

    board.occupiedBoard = board.board_state[MIN] | board.board_state[MAX];
}

const uint64_t notAFile = 0xFEFEFEFEFEFEFEFEULL; 
const uint64_t notHFile = 0x7F7F7F7F7F7F7F7FULL; 

uint64_t soutOne (uint64_t b) {return  b >> 8;}
uint64_t nortOne (uint64_t b) {return  b << 8;}

void BitBoard::makeMove(const BitMove& move, BoardStruct& board, bool isMinimax)
{
    bool max = move.indexMask & board.board_state[MAX];
    const uint64_t promotionRankMask = !max
        ? 0x00000000000000FFULL  // MIN promotes on rank 1 (bits 0–7)
        : 0xFF00000000000000ULL; // MAX promotes on rank 8 (bits 56–63)

    uint64_t& opponentMask = board.board_state[!max];
    uint64_t& currentMask = board.board_state[max];

    opponentMask &= ~(move.captureMask);
    currentMask = (currentMask & ~move.indexMask) | move.moveMask;

    if (move.indexMask & board.queenBoard) 
        board.queenBoard = (board.queenBoard & ~move.indexMask) | move.moveMask;

    if (move.moveMask & promotionRankMask)
        board.queenBoard |= move.moveMask;

    board.occupiedBoard = board.board_state[MAX] | board.board_state[MIN];
        
    if (!isMinimax)
    {
        switch(move.flag)
        {
            case MoveFlag::NONE: core::SoundHandle::play("move"); break;
            case MoveFlag::CAPTURE:
            case MoveFlag::MULTICAPTURE: core::SoundHandle::play("capture"); break;
            case MoveFlag::PROMOTION: core::SoundHandle::play("promote"); break;
            default: break;
        }
    }
}

std::vector<float> BitBoard::vectorBoard(BoardStruct& board)
{
    std::vector<float> vec(64, 0.0f); 

    for (int i = 0; i < 64; ++i)
    {
        uint64_t bit = 1ULL << i;

        bool isMax = board.board_state[1] & bit;
        bool isMin = board.board_state[0] & bit;
        bool isQueen = board.queenBoard & bit;

        if (isMax)
            vec[i] = isQueen ? 1.0f : 0.5f;
        else if (isMin)
            vec[i] = isQueen ? -1.0f : -0.5f;
    }

    return vec;
}


std::vector<uint64_t> BitBoard::getPieceIndices(BoardStruct& board, bool max)
{
    std::vector<uint64_t> indices;

    uint64_t board_state = board.board_state[max];

    while (board_state)
    {
        unsigned long index;
        _BitScanForward64(&index, board_state); 

        indices.push_back(1ULL << index);
        board_state &= (board_state - 1);
    }

    return indices;
}

std::vector<BitMove> BitBoard::getMoves(BoardStruct& board, bool max)
{
    std::vector<BitMove> moves;

    currentPieceIndices = getPieceIndices(board, max);

    for (uint64_t index : currentPieceIndices)
    {
        if (!(board.queenBoard & index))
            checkMoves(index, max, moves, board);
        else
            checkMovesQueen(index, max, moves, board);
    }

    return moves;
}

std::vector<BitMove> BitBoard::getMovesForPiece(const uint64_t index, BoardStruct& board)
{
    std::vector<BitMove> moves;

    bool max = index & board.board_state[MAX];
    
    if (!(board.queenBoard & index))
        checkMoves(index, max, moves, board);
    else
        checkMovesQueen(index, max, moves, board);
    
    return moves;
}

void BitBoard::checkMoves(const uint64_t index, bool max, std::vector<BitMove>& moves, BoardStruct& board)
{
    std::array<BitDirection, 4> directions = BitMove::getMoveOffsets();

    for (int32_t d = 0; d < 2; ++d)
    {
        bool leftSide = (d % 2 == 0);
        bool validEdge = max
            ? (index & (leftSide ? notHFile : notAFile))
            : (index & (leftSide ? notAFile : notHFile));

        if (!validEdge) continue;

        uint64_t moveMask = max ? (index << directions[d].offset) : (index >> directions[d].offset);

        if (!(moveMask & board.occupiedBoard) && moveMask)
            moves.emplace_back(index, moveMask, 0, MoveFlag::NONE);
    }

    checkCaptures(index, max, moves, board);
}

inline int32_t getFileIndex(uint64_t squareMask)
{
    unsigned long rawIndex;
    _BitScanForward64(&rawIndex, squareMask);

    return rawIndex % 8;
}

void BitBoard::checkMovesQueen(const uint64_t index, bool max, std::vector<BitMove>& moves, BoardStruct& board)
{
    std::array<BitDirection, 4> directions = BitMove::getMoveOffsets();

    unsigned long rawIndex;
    _BitScanForward64(&rawIndex, index);

    for (auto& d : directions)
    {
        bool pieceFound = false; 
        uint64_t current = index;
        uint64_t jumpedPiece = 0;

        while (true)
        {
            uint64_t next = d.downwards ? current >> d.offset : current << d.offset;
            if (!next) break;

            int32_t nextFile = getFileIndex(current);
            int32_t currentFile = getFileIndex(next);

            if (std::abs(currentFile - nextFile) != 1)
                break; 

            if (!(board.occupiedBoard & next))
            {
                if (!pieceFound)
                {
                    moves.emplace_back(index, next, 0, MoveFlag::NONE);
                    current = next;
                }
                else
                {
                    if (!jumpedPiece)
                        break;

                    int32_t jumpedFile = getFileIndex(jumpedPiece);
                    int32_t landingFile = getFileIndex(next);

                    if (std::abs(jumpedFile - landingFile) != 1)
                        break; 

                    moves.emplace_back(index, next, jumpedPiece, MoveFlag::CAPTURE);
                    checkCaptures(next, max, moves, board, index, jumpedPiece);
                    break;
                }
            }
            else if (!pieceFound && (next & board.board_state[!max]))
            {
                int32_t currentFile = getFileIndex(current);
                int32_t jumpedFile  = getFileIndex(next);

                if (std::abs(currentFile - jumpedFile) != 1)
                    break;

                pieceFound = true;
                jumpedPiece = next;
                current = next;
            }
            else
            {
                break;
            }      
        }
    }
}

void BitBoard::checkCaptures(const uint64_t index, bool max, std::vector<BitMove>& moves, BoardStruct& board, int64_t origin, uint64_t captureMask)
{
    std::array<BitDirection, 4> directions = BitMove::getMoveOffsets();
    int64_t originIndex = (origin == -1) ? index : origin;

    for (const auto& d : directions)
    {  
        uint64_t jumpedPieceMask = !d.downwards ? index << d.offset : index >> d.offset;
        uint64_t landingMask     = !d.downwards ? index << (d.offset * 2) : index >> (d.offset * 2);

        if (!landingMask || !jumpedPieceMask) continue;

        int32_t jumpedFile = getFileIndex(jumpedPieceMask);
        int32_t landingFile = getFileIndex(landingMask);
        if (std::abs(jumpedFile - landingFile) != 1) continue;

        if (landingMask & board.occupiedBoard) continue; // is alreadly occupied
        if (!(jumpedPieceMask & board.board_state[!max]) || (captureMask & jumpedPieceMask)) continue;

        uint64_t newCaptureMask = captureMask | jumpedPieceMask;

        moves.emplace_back(originIndex, landingMask, newCaptureMask, MoveFlag::CAPTURE);

        checkCaptures(landingMask, max, moves, board, originIndex, newCaptureMask);
    }
}