#ifndef BOARD_HPP
#define BOARD_HPP

#include <stdint.h>
#include <vector>
#include <array>
#include <set>

#define MAX true
#define MIN false

#define GET_MIDDLE(x, y) ((x + y) / 2)

// Moves
#define UP_LEFT 0
#define UP_RIGHT 1
#define DOWN_LEFT 2
#define DOWN_RIGHT 3

enum class MoveFlag
{
    NONE,
    CAPTURE,
    MULTICAPTURE,
    PROMOTION
};

struct Move
{
    uint32_t startPos;
    uint32_t endPos;
    std::vector<uint32_t> middlePositions;
    MoveFlag flag{MoveFlag::NONE};

    static constexpr std::array<int32_t, 4> getMoveOffsets()
    {
        return {-9, 7, 9, -7}; // Diagonal move offsets, [0] = up_left, [1] = up_right, [2] = down_left, [3] = down_right;
    }

    constexpr bool operator==(const Move& other) const
    {
        return startPos == other.startPos && endPos == other.endPos && other.middlePositions == middlePositions;
    }
};

class Board
{
public:
    Board(uint32_t board_size);
    ~Board() = default;

    void makeMove(Move& move, std::vector<float>& board_state, bool isMinimax = false);

    inline void restart() { initiateBoard(); }

    inline uint32_t getSize() { return board_size; }
    inline bool getCurrentTurn() const { return currentTurn; }
    inline bool isGameOver(bool max, std::vector<float>& board_state)
    {
        return getMoves(max, board_state).size() == 0;
    }

    std::vector<uint32_t> getPieces(bool max, std::vector<float>& board_state);
    std::vector<Move> getMoves(bool max, std::vector<float>& board_state);
    std::vector<Move> getMovesForPiece(uint32_t pieceIndex, std::vector<float>& board_state)
    {
        if (!isWithinBounds(pieceIndex))
            return {};

        std::vector<Move> moves;

        bool max = board_state[pieceIndex] > 0;
        if (isQueen(pieceIndex))
            checkMovesQueen(pieceIndex, max, moves, board_state);
        else
            checkMoves(pieceIndex, max, moves, board_state);

        return moves;
    }
    inline std::vector<float>& board() { return board_state; }
    inline void changeTurn() { currentTurn = !currentTurn; }

    inline int32_t getY(int32_t index) { return (index % board_size); }
    inline int32_t getX(int32_t index) { return (index / board_size); }
private:
    void initiateBoard();

    bool isWithinBounds(uint32_t index) 
    { 
        return index < board_state.size(); 
    }
    bool isQueen(uint32_t pieceIndex) { return std::abs(board_state[pieceIndex]) == 1; }

    void checkMoves(const uint32_t pieceIndex, bool max, std::vector<Move>& moves, std::vector<float>& board_state);
    void checkMovesQueen(const uint32_t pieceIndex, bool max, std::vector<Move>& moves, std::vector<float>& board_state);

    void checkCaptures(const uint32_t pieceIndex, bool max, std::vector<Move>& moves, std::vector<float>& board_state, int32_t origin = -1, std::vector<uint32_t> middleIndices = {});
    bool canCapture(const Move& move, std::vector<float>& board_state, bool max);

    void checkDirectionsEnabled(const uint32_t pieceIndex, std::vector<int32_t>& directions);

    inline bool isCaptureDistance(uint32_t index1, uint32_t index2)
    {
        return abs(getX(index2) - getX(index1)) == 2 && abs(getY(index2) - getY(index1)) == 2;
    }

    inline float distance(uint32_t index1, uint32_t index2)
    {
        return sqrtf(powf(getX(index2) - getX(index1), 2) + powf(getY(index2) - getY(index1), 2));
    }

    bool currentTurn{MIN};
    uint32_t board_size{0};
    std::vector<float> board_state;
};

#endif //BOARD_HPP