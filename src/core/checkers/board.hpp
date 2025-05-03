#ifndef BOARD_HPP
#define BOARD_HPP

#include <stdint.h>
#include <vector>
#include <array>
#include <set>

#define MAX true
#define MIN false

// Moves
#define UP_LEFT 0
#define UP_RIGHT 1
#define DOWN_LEFT 2
#define DOWN_RIGHT 3

struct Move
{
    uint32_t startPos;
    uint32_t endPos;

    static constexpr std::array<int32_t, 4> getMoveOffsets()
    {
        return {7, 9, -9, -7}; // Diagonal move offsets, [0] = up_left, [1] = up_right, [2] = down_left, [3] = down_right;
    }

    constexpr bool operator==(const Move& other) const
    {
        return startPos == other.startPos && endPos == other.endPos;
    }
};

struct HistoryEntry
{
    Move move;
    std::vector<float> piecesCaptured{};

    constexpr bool operator==(const HistoryEntry& other) const
    {
        return move == other.move && piecesCaptured == other.piecesCaptured;
    }
};

class Board
{
public:
    Board(uint32_t board_size);
    ~Board() = default;

    void makeMove(const Move& move);
    void unmakeMove(const Move& move);

    inline void restart() { initiateBoard(); }

    inline uint32_t getSize() { return board_size; }
    inline bool getCurrentTurn() const { return currentTurn; }
    inline bool isGameOver(bool max)
    {
        return getMoves(max).size() == 0;
    }

    std::vector<Move> getMoves(bool max);
    std::vector<Move> getMovesForPiece(uint32_t pieceIndex)
    {
        if (!isWithinBounds(pieceIndex))
            return {};

        std::vector<Move> moves;

        bool max = board_state[pieceIndex] > 0;
        if (isQueen(pieceIndex))
            checkMovesQueen(pieceIndex, max, moves);
        else
            checkMoves(pieceIndex, max, moves);

        return moves;
    }
    inline std::vector<float>& board() { return board_state; }
private:
    void initiateBoard();

    bool isWithinBounds(uint32_t index) { return index < board_size * board_size; }
    bool isQueen(uint32_t pieceIndex) { return std::abs(board_state[pieceIndex]) == 1; }

    void checkMoves(const uint32_t pieceIndex, bool max, std::vector<Move>& moves);
    void checkMovesQueen(const uint32_t pieceIndex, bool max, std::vector<Move>& moves);

    void checkCaptures(const uint32_t pieceIndex, bool max, std::vector<Move>& moves);
    bool canCapture(const Move& move, bool max);

    bool handleBoardCaptures(const Move& move);
    void handleBoardUndoCaptures(const HistoryEntry& entry);

    bool currentTurn{MIN};
    uint32_t board_size{0};
    std::vector<float> board_state;

    std::pair<std::vector<Move>, std::vector<Move>> legalMoves;
    std::vector<HistoryEntry> moveHistory;
};

#endif //BOARD_HPP