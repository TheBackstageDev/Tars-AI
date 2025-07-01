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

enum MoveFlag
{
    NONE         = 0ULL,
    CAPTURE      = 1ULL << 0,
    MULTICAPTURE = 1ULL << 1,
    PROMOTION    = 1ULL << 2
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

struct BoardDistances
{
    static constexpr std::array<uint8_t, 64> toTopEdge = [] {
        std::array<uint8_t, 64> arr{};
        for (int i = 0; i < 64; ++i) arr[i] = i / 8; // Distance to top edge
        return arr;
    }();

    static constexpr std::array<uint8_t, 64> toBottomEdge = [] {
        std::array<uint8_t, 64> arr{};
        for (int i = 0; i < 64; ++i) arr[i] = 7 - (i / 8); // Distance to bottom edge
        return arr;
    }();

    static constexpr std::array<uint8_t, 64> toLeftEdge = [] {
        std::array<uint8_t, 64> arr{};
        for (int i = 0; i < 64; ++i) arr[i] = i % 8; // Distance to left edge
        return arr;
    }();

    static constexpr std::array<uint8_t, 64> toRightEdge = [] {
        std::array<uint8_t, 64> arr{};
        for (int i = 0; i < 64; ++i) arr[i] = 7 - (i % 8); // Distance to right edge
        return arr;
    }();
    
    static constexpr std::array<std::array<uint8_t, 64>, 4> toDiagonalEdges = [] {
        std::array<std::array<uint8_t, 64>, 4> arrs{};

        for (int i = 0; i < 64; ++i) {
            arrs[0][i] = std::min(BoardDistances::toTopEdge[i], BoardDistances::toLeftEdge[i]);  // Up-left (-9)
            arrs[1][i] = std::min(BoardDistances::toTopEdge[i], BoardDistances::toRightEdge[i]); // Up-right (7)
            arrs[2][i] = std::min(BoardDistances::toBottomEdge[i], BoardDistances::toLeftEdge[i]); // Down-left (-7)
            arrs[3][i] = std::min(BoardDistances::toBottomEdge[i], BoardDistances::toRightEdge[i]); // Down-right (9)
        }

        return arrs;
    }();
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

struct BitDirection
{
    uint64_t offset;
    bool downwards;

    constexpr BitDirection(uint64_t o, bool d) noexcept : offset(o), downwards(d) {}
};

struct BitMove
{
    uint64_t indexMask;
    uint64_t moveMask;
    uint64_t captureMask;

    MoveFlag flag{MoveFlag::NONE};

    static constexpr std::array<BitDirection, 4> getMoveOffsets()
    {
        return {BitDirection{9, false},   // UP_LEFT (<<)
                BitDirection{7, false},   // UP_RIGHT (<<)
                BitDirection{7, true},    // DOWN_LEFT (>>)
                BitDirection{9, true}};   // DOWN_RIGHT (>>)
    }

    constexpr bool operator==(const BitMove& other) const
    {
        return moveMask == other.moveMask && captureMask == other.captureMask;
    }
};

struct BoardStruct
{
    std::array<uint64_t, 2> board_state;
    uint64_t queenBoard;
    uint64_t occupiedBoard;

    bool operator==(const BoardStruct& other) const {
        return occupiedBoard == other.occupiedBoard &&
               board_state[0] == other.board_state[0] &&
               board_state[1] == other.board_state[1] &&
               queenBoard == other.queenBoard;
    }
};

class BitBoard
{
public:
    BitBoard();
    ~BitBoard() = default;

    inline void restart() { initiateBoard(); }  

    void makeMove(const BitMove& move, BoardStruct& board, bool isMinimax = false);

    std::vector<uint64_t> getPieceIndices(BoardStruct& board, bool max);
    std::vector<BitMove> getMoves(BoardStruct& board, bool max);
    std::vector<BitMove> getMovesForPiece(const uint64_t index, BoardStruct& board);

    inline BoardStruct& bitboard() { return board; }
    std::vector<float> vectorBoard(BoardStruct& board);
    inline void changeTurn() { currentTurn = !currentTurn; }

    inline uint32_t getSize() { return 8; }
    inline bool getCurrentTurn() const { return currentTurn; }
    inline bool isGameOver(bool max, BoardStruct& board)
    {
        return getMoves(board, max).size() == 0;
    }

private:
    void initiateBoard();

    void checkDirectionsEnabled(const uint64_t index, std::vector<BitDirection>& directions);
    void checkMoves(const uint64_t index, bool max, std::vector<BitMove>& moves, BoardStruct& board);
    void checkMovesQueen(const uint64_t index, bool max, std::vector<BitMove>& moves, BoardStruct& board);

    void checkCaptures(const uint64_t index, bool max, std::vector<BitMove>& moves, BoardStruct& board, int64_t origin = -1, uint64_t captureMask = 0);

    std::vector<uint64_t> currentPieceIndices;

    bool currentTurn{MIN};
    BoardStruct board;
};

#endif //BOARD_HPP