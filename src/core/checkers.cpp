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
                    board_state[index] = 1.0;
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

float distance(ImVec2 x0, ImVec2 x1)
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

bool Checkers::canCapture(uint32_t pieceIndex, uint32_t moveIndex, bool pieceOwner)
{
    int32_t moveX = moveIndex / board_size;
    int32_t moveY = moveIndex % board_size;

    if (!isMoveLegal(moveX, moveY))
        return false;

    int32_t currentX = pieceIndex / board_size;
    int32_t currentY = pieceIndex % board_size;
    int32_t midX = (currentX + moveX) / 2;
    int32_t midY = (currentY + moveY) / 2;
    uint32_t midIndex = midX * board_size + midY;

    float currentPiece = board_state[pieceIndex];
    float targetPiece = board_state[midIndex];

    if (targetPiece == 0 || (targetPiece > 0 == pieceOwner))
        return false;

    return abs(moveX - currentX) == 2 && abs(moveY - currentY) == 2;
}

void Checkers::checkCaptures(uint32_t pieceIndex, std::vector<uint32_t> &captures, int dir, int32_t pieceOwner)
{
    int32_t currentX = pieceIndex / board_size;
    int32_t currentY = pieceIndex % board_size;
    float piece = board_state[pieceIndex];
    bool pieceOwner1 = pieceOwner > 0 ? pieceOwner : piece > 0;

    const int32_t direction = dir == -2 ? piece < 0 ? -1 : 1 : dir; // Plr moves up while Bot moves down;

    if (isQueen(pieceIndex))
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
    
                if (canCapture(newCurrentIndex, jumpIndex, pieceOwner1))
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

            if (canCapture(pieceIndex, leftJumpIndex, pieceOwner1) && std::find(captures.begin(), captures.end(), leftJumpIndex) == captures.end())
            {
                captures.emplace_back(leftJumpIndex);
                checkCaptures(leftJumpIndex, captures, direction, pieceOwner1);
            }
    
            if (canCapture(pieceIndex, rightJumpIndex, pieceOwner1) && std::find(captures.begin(), captures.end(), rightJumpIndex) == captures.end())
            {
                captures.emplace_back(rightJumpIndex);
                checkCaptures(rightJumpIndex, captures, direction, pieceOwner1);
            }
        }
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
            const uint32_t cellIndex = x * board_size + y;
            const float current_cell = board_state[cellIndex];
            if (current_cell == 0)
                continue;

            const bool bot_piece = current_cell < 0;
            if (bot_piece && player)
                continue;
            if (!bot_piece && !player)
                continue;

            checkMoves(cellIndex, moves);
            checkCaptures(cellIndex, captures);
        }
    }

    return std::pair(moves, captures);
}

void Checkers::checkMoves(uint32_t pieceIndex, std::vector<uint32_t> &moves, int dir)
{
    float piece = board_state[pieceIndex];

    if (piece == 0)
        return;

    const int32_t direction = dir == -2 ? piece < 0 ? -1 : 1 : dir;

    int32_t currentX = pieceIndex / board_size;
    int32_t currentY = pieceIndex % board_size;
    int32_t leftX = currentX + 1, leftY = currentY + direction;
    int32_t rightX = currentX - 1, rightY = currentY + direction;

    uint32_t leftIndex = leftX * board_size + leftY;
    uint32_t rightIndex = rightX * board_size + rightY;

    if (isQueen(pieceIndex))
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

                if (!isMoveLegal(x, y))
                    break;

                uint32_t moveIndex = x * board_size + y;
                moves.emplace_back(moveIndex);
            }
        }
    }
    else
    {
        if (isMoveLegal(leftX, leftY))
            moves.emplace_back(leftIndex);

        if (isMoveLegal(rightX, rightY))
            moves.emplace_back(rightIndex);
    }
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> Checkers::getPossiblePieceMoves(uint32_t pieceIndex)
{
    std::vector<uint32_t> captures;
    std::vector<uint32_t> moves;

    checkMoves(pieceIndex, moves);
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

        int currentX = pieceIndex / board_size;
        int currentY = pieceIndex % board_size;
        int moveX = moveIndex / board_size;
        int moveY = moveIndex % board_size;

        int32_t dist = distance(ImVec2(currentX, currentY), ImVec2(moveX, moveY));

        if (dist > 2)
        {
            std::vector<uint32_t> captures = movesPossibleCurrentPiece.second;
            uint32_t currentCaptureIndex = pieceIndex;

            while (!captures.empty() && currentCaptureIndex != moveIndex)
            {
                uint32_t nextMove = captures.front();
                captures.erase(captures.begin());
                
                if (isQueen(moveIndex))
                {
                    int32_t dx = (nextMove / board_size > currentCaptureIndex / board_size) ? 1 : -1;
                    int32_t dy = (nextMove % board_size > currentCaptureIndex % board_size) ? 1 : -1;
                    
                    int32_t x = currentX, y = currentY;

                    while (x != moveX && y != moveY)
                    {
                        x += dx;
                        y += dy;

                        uint32_t currentIndex = x * board_size + y;
                        
                        if (isWithinBounds(x, y) && board_state[currentIndex] != 0) 
                        {
                            board_state[currentIndex] = 0;
                            break;
                        }
                    }
                }
                else
                {
                    uint32_t midIndex = getMiddle(currentCaptureIndex, nextMove);
                    board_state[midIndex] = 0;
    
                    currentCaptureIndex = nextMove;
                }

                checkCaptures(currentCaptureIndex, captures);
            }
        }
        else
        {
            board_state[getMiddle(pieceIndex, moveIndex)] = 0;
        }

        actionHappened = true;
    }

    if (actionHappened)
    {
        uint32_t pieceNewY = moveIndex % board_size;
        if ((pieceNewY == 0 && movePiece < 0) ||            // Negative pieces reach bottom
            (pieceNewY == board_size - 1 && movePiece > 0)) // Positive pieces reach top
        {
            movePiece = (movePiece > 0) ? 1 : -1;
        }

        currentTurn = currentTurn == PLR ? BOT : PLR;
        movesPossibleCurrentPiece = {};
        currentSelectedPiece = -1;
    }
}

void Checkers::drawBoard()
{
    ImGui::SetNextWindowSize(ImVec2(board_size * tile_size + (margin * 2), board_size * tile_size + (margin * 2)));
    ImGui::Begin("Board", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImDrawList *drawlist = ImGui::GetWindowDrawList();
    ImVec2 window_pos = ImGui::GetWindowPos();
    ImVec2 board_start = ImVec2(window_pos.x + margin, window_pos.y + margin);

    int32_t moveIndex = getCellMouseAt(board_start);

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
        moveIndex != -1 &&
        ((!currentTurn == false && board_state[moveIndex] > 0) ||
         (!currentTurn == true && board_state[moveIndex] < 0)))
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
        }
    }

    ImGui::End();
}

void Checkers::drawInfo()
{
}

void Checkers::drawCrown(ImDrawList *drawlist, ImVec2 center)
{
    const ImU32 crownColor = IM_COL32(225, 225, 0, 255);
    const float crownBaseHeight = tile_size * 0.01f;
    const float crownBaseWidth = tile_size * 0.2f;

    const float crownPeakHeight = tile_size * 0.2f;
    const float crownPeakWidth = tile_size * 0.3f;

    // Crown Base
    const ImVec2 bottomLeft((center.x - crownBaseWidth), (center.y - crownBaseHeight));
    const ImVec2 bottomRight((center.x + crownBaseWidth), (center.y - crownBaseHeight));

    // Crown Peaks
    const ImVec2 middlePeak(center.x, center.y - crownPeakHeight);
    const ImVec2 leftPeak(bottomLeft.x - crownPeakWidth * 0.1f, center.y - crownPeakHeight);
    const ImVec2 rightPeak(bottomRight.x + crownPeakWidth * 0.1f, center.y - crownPeakHeight);

    const ImVec2 midLeft(center.x - crownPeakWidth * 0.25f, center.y - crownPeakHeight * 0.5);
    const ImVec2 midRight(center.x + crownPeakWidth * 0.25f, center.y - crownPeakHeight * 0.5);

    const ImVec2 outlinePoints[] = {bottomLeft, leftPeak, midLeft, middlePeak, midRight, rightPeak, bottomRight, bottomLeft};

    drawlist->AddTriangleFilled(bottomLeft, middlePeak, bottomRight, crownColor);
    drawlist->AddTriangleFilled(bottomLeft, leftPeak, bottomRight, crownColor);
    drawlist->AddTriangleFilled(bottomLeft, rightPeak, bottomRight, crownColor);
    drawlist->AddPolyline(outlinePoints, IM_ARRAYSIZE(outlinePoints), IM_COL32(255, 255, 0, 255), ImDrawFlags_None, 2.0f);
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

    if (abs(board_state[id]) == 1)
        drawCrown(ImGui::GetWindowDrawList(), ImVec2(center.x, center.y + tile_size / 12.f));

    ImGui::End();
}