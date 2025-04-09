#include "checkers.hpp"
#include <string>
#include <iostream>

// Player Colors
#define PLR1_COLOR IM_COL32(225, 0, 0, 255)     // Red (Player 1)
#define PLR2_COLOR IM_COL32(225, 208, 150, 255) // Beige/Sepia (Player 2)

// Board Colors
#define BOARD_LIGHT IM_COL32(222, 184, 135, 255) // Light Wood (Beige/Brown)
#define BOARD_DARK IM_COL32(139, 69, 19, 255)    // Dark Brown (Classic Wood)

Checkers::Checkers(const uint32_t board_size, const float tile_size)
    : board_size(board_size), tile_size(tile_size)
{
    board_state.resize(board_size * board_size);
    initiateBoard();
}

void Checkers::initiateBoard()
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

std::string getHouse(int x, int y)
{
    return std::string(1, char('A' + x)) + std::to_string(8 - y);
}

float distance(const ImVec2 x0, const ImVec2 x1)
{
    return sqrtf(powf(x1.x - x0.x, 2) + powf(x1.y - x0.y, 2));
}

bool isClicked(const ImVec2 center, const float tile_size)
{
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    {
        if (distance(ImGui::GetMousePos(), center) < tile_size / 2)
            return true;
    }

    return false;
}

bool Checkers::canCapture(uint32_t moveIndex, uint32_t currentIndex)
{
    int currentX = currentIndex / board_size;
    int currentY = currentIndex % board_size;
    int moveX = moveIndex / board_size;
    int moveY = moveIndex % board_size;

    if (!isMoveLegal(moveX, moveY))
        return false;

    if (board_state[moveIndex] != 0)
        return false;

    int midX = (currentX + moveX) / 2;
    int midY = (currentY + moveY) / 2;
    uint32_t midIndex = midX * board_size + midY;

    float currentPiece = board_state[currentIndex];
    float targetPiece = board_state[midIndex];

    if (targetPiece == 0 || (currentPiece > 0 == targetPiece > 0))
        return false;

    if (abs(moveX - currentX) == 2 && abs(moveY - currentY) == 2)
        return true;

    return false;
}

void Checkers::checkCaptures(uint32_t pieceIndex, std::vector<uint32_t>& captures, int dir)
{
    int currentX = pieceIndex / board_size;
    int currentY = pieceIndex % board_size;
    float piece = board_state[pieceIndex];

    const int direction = dir == -2 ? piece < 0 ? -1 : 1 : dir; // Plr moves up while Bot moves down;

    int leftJumpX = currentX - 2, leftJumpY = currentY + direction * 2;
    int rightJumpX = currentX + 2, rightJumpY = currentY + direction * 2;

    int32_t leftJumpIndex = leftJumpX * board_size + leftJumpY;
    int32_t rightJumpIndex = rightJumpX * board_size + rightJumpY;

    if (canCapture(leftJumpIndex, pieceIndex) && std::find(captures.begin(), captures.end(), leftJumpIndex) == captures.end())
    {
        captures.emplace_back(leftJumpIndex);
        checkCaptures(leftJumpIndex, captures, direction);
        checkCaptures(leftJumpIndex, captures, -direction);
    }

    if (canCapture(rightJumpIndex, pieceIndex) && std::find(captures.begin(), captures.end(), rightJumpIndex) == captures.end())
    {
        captures.emplace_back(rightJumpIndex);
        checkCaptures(rightJumpIndex, captures, direction);
        checkCaptures(rightJumpIndex, captures, -direction);
    }
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> Checkers::getPossibleMoves(bool player)
{
    std::vector<uint32_t> captures;
    std::vector<uint32_t> moves;

    for (int32_t x = 0; x < board_size; ++x)
    {
        for (int32_t y = 0; y < board_size; ++y)
        {
            const float current_cell = board_state[x * board_size + y];
            if (current_cell == 0)
                continue;

            const bool bot_piece = current_cell < 0;
            if (bot_piece && player)
                continue;
            if (!bot_piece && !player)
                continue;

            const int direction = current_cell < 0 ? -1 : 1; // Plr moves up while Bot moves down;

            int leftX = x + 1, leftY = y + direction;
            int rightX = x - 1, rightY = y + direction;

            int32_t leftIndex = leftX * board_size + leftY;
            int32_t rightIndex = rightX * board_size + rightY;

            if (isMoveLegal(leftX, leftY) && std::find(moves.begin(), moves.end(), leftIndex) == moves.end())
                moves.emplace_back(leftIndex);

            if (isMoveLegal(rightX, rightY) && std::find(moves.begin(), moves.end(), rightIndex) == moves.end())
                moves.emplace_back(rightIndex);

            checkCaptures(x * board_size + y, captures);
        }
    }

    return std::pair(moves, captures);
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> Checkers::getPossiblePieceMoves(uint32_t pieceIndex)
{
    std::vector<uint32_t> captures;
    std::vector<uint32_t> moves;

    int currentX = pieceIndex / board_size;
    int currentY = pieceIndex % board_size;
    float piece = board_state[pieceIndex];

    if (piece == 0)
        return {moves, captures};

    const int direction = piece < 0 ? -1 : 1;

    int leftX = currentX + 1, leftY = currentY + direction;
    int rightX = currentX - 1, rightY = currentY + direction;

    if (isMoveLegal(leftX, leftY))
        moves.emplace_back(leftX * board_size + leftY);

    if (isMoveLegal(rightX, rightY))
        moves.emplace_back(rightX * board_size + rightY);

    int leftJumpX = currentX - 2, leftJumpY = currentY + direction * 2;
    int rightJumpX = currentX + 2, rightJumpY = currentY + direction * 2;

    int32_t leftIndex = leftX * board_size + leftY;
    int32_t rightIndex = rightX * board_size + rightY;

    checkCaptures(pieceIndex, captures);

    return std::pair(moves, captures);
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> movesPossibleCurrentPiece;
int32_t currentSelectedPiece{-1};

#define PLR false
#define BOT true

bool currentTurn = PLR;

void Checkers::handleAction(int32_t pieceIndex, int32_t moveIndex)
{
    bool actionHappened{false};

    if (pieceIndex == -1 || moveIndex == -1)
        return;

    float &currentPiece = board_state[pieceIndex];
    float &movePiece = board_state[moveIndex];

    if (movePiece != 0 || currentPiece == 0)
        return;

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && std::find(movesPossibleCurrentPiece.first.begin(), movesPossibleCurrentPiece.first.end(), moveIndex) != movesPossibleCurrentPiece.first.end())
    {
        movePiece = currentPiece;
        currentPiece = 0;
        actionHappened = true;
    }
    else if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && std::find(movesPossibleCurrentPiece.second.begin(), movesPossibleCurrentPiece.second.end(), moveIndex) != movesPossibleCurrentPiece.second.end())
    {
        movePiece = currentPiece;
        currentPiece = 0;

        uint32_t midIndex = ((pieceIndex / board_size + moveIndex / board_size) / 2) * board_size + ((pieceIndex % board_size + moveIndex % board_size) / 2);

        board_state[midIndex] = 0;

        actionHappened = true;

        std::vector<uint32_t> nextCaptures;
        checkCaptures(moveIndex, nextCaptures);
    
        while (!nextCaptures.empty()) 
        {
            uint32_t nextMove = nextCaptures.front();
            nextCaptures.erase(nextCaptures.begin());
    
            uint32_t nextMidIndex = ((moveIndex / board_size + nextMove / board_size) / 2) * board_size +
                                    ((moveIndex % board_size + nextMove % board_size) / 2);
            
            board_state[nextMidIndex] = 0;
            moveIndex = nextMove; 
            
            checkCaptures(moveIndex, nextCaptures);
        }
    }

    if (actionHappened)
    {
        currentTurn = currentTurn == PLR ? BOT : PLR;
        movesPossibleCurrentPiece = {};
        currentSelectedPiece = -1;
    }
}

void Checkers::drawBoard()
{
    const float margin = 20.f;
    ImGui::SetNextWindowSize(ImVec2(board_size * tile_size + (margin * 2), board_size * tile_size + (margin * 2)));
    ImGui::Begin("Board", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImDrawList *drawlist = ImGui::GetWindowDrawList();
    ImVec2 window_pos = ImGui::GetWindowPos();
    ImVec2 board_start = ImVec2(window_pos.x + margin, window_pos.y + margin);

    int32_t moveIndex = getCellMouseAt(board_start);

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && moveIndex != -1 && board_state[moveIndex])
    {
        currentSelectedPiece = moveIndex;
        movesPossibleCurrentPiece = getPossiblePieceMoves(moveIndex);
    }

    handleAction(currentSelectedPiece, moveIndex);

    for (int x = 0; x < board_size; ++x)
    {
        for (int y = 0; y < board_size; ++y)
        {
            ImVec2 top_left = ImVec2(board_start.x + x * tile_size, board_start.y + y * tile_size);
            ImVec2 bottom_right = ImVec2(top_left.x + tile_size, top_left.y + tile_size);

            const uint32_t pieceIndex = x * board_size + y;
            const float currentPiece = board_state[pieceIndex];

            ImU32 color = ((x + y) % 2 == 0) ? BOARD_LIGHT : BOARD_DARK;
            ImU32 pieceColor = currentPiece > 0 ? PLR1_COLOR : PLR2_COLOR;

            ImVec2 center(top_left.x + tile_size / 2, top_left.y + tile_size / 2);

            if (pieceIndex == currentSelectedPiece)
            {
                color = IM_COL32(200, 125, 75, 255);

                drawlist->AddRectFilled(top_left, bottom_right, color);
                drawlist->AddRect(top_left, bottom_right, IM_COL32_WHITE, 0.0f, ImDrawFlags_None, 1.0);
            }
            else
            {
                if (std::find(movesPossibleCurrentPiece.first.begin(), movesPossibleCurrentPiece.first.end(), pieceIndex) != movesPossibleCurrentPiece.first.end() || std::find(movesPossibleCurrentPiece.second.begin(), movesPossibleCurrentPiece.second.end(), pieceIndex) != movesPossibleCurrentPiece.second.end())
                    color *= 2;

                drawlist->AddRectFilled(top_left, bottom_right, color);
            }

            ImVec2 text_pos = ImVec2(board_start.x + x * tile_size + (tile_size / 20), board_start.y + (y + 1) * tile_size - (tile_size / 4));

            ImGui::GetWindowDrawList()->AddText(text_pos, ((x + y) % 2 == 0) ? BOARD_DARK : BOARD_LIGHT, getHouse(x, y).c_str());

            if (currentPiece != 0)
                drawPiece(drawlist, pieceColor, center, pieceIndex);

            if (std::abs(currentPiece) == 1)
                drawCrown(drawlist, center);
        }
    }

    ImGui::End();
}

void Checkers::drawCrown(ImDrawList *drawlist, const ImVec2 center)
{
}

void Checkers::drawPiece(ImDrawList *drawlist, const ImU32 color, const ImVec2 center, const uint32_t id)
{
    std::string windowId = "Piece##" + std::to_string(id);

    ImGui::SetNextWindowSize(ImVec2(tile_size, tile_size));
    ImGui::SetNextWindowPos(ImVec2(center.x - tile_size / 2, center.y - tile_size / 2));
    ImGui::Begin(windowId.c_str(), nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBackground);

    const float radius = (tile_size / 2) - 10;

    drawlist->AddCircleFilled(center, radius, color);
    drawlist->AddCircle(center, radius, color / 3, 0, tile_size / 10);
    drawlist->AddCircle(center, radius / 1.5, color / 3, 0, tile_size / 30);

    ImGui::End();
}