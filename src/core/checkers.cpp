#include "checkers.hpp"
#include <string>
#include <iostream>

// Player Colors
#define PLR1_COLOR IM_COL32(225, 0, 0, 255)      // Red (Player 1)
#define PLR2_COLOR IM_COL32(225, 208, 150, 255)  // Beige/Sepia (Player 2)

// Board Colors
#define BOARD_LIGHT IM_COL32(222, 184, 135, 255) // Light Wood (Beige/Brown)
#define BOARD_DARK  IM_COL32(139, 69, 19, 255)   // Dark Brown (Classic Wood)

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
                if (y < board_size / 2 - 1) {
                    board_state[index] = 0.5;  
                }
                else if (y > board_size / 2) {
                    board_state[index] = -0.5; 
                }
            }
        }
    }
}

int32_t currentSelectedPiece{-1};

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

void Checkers::drawBoard()
{
    const float margin = 20.f;
    ImGui::SetNextWindowSize(ImVec2(board_size * tile_size + (margin * 2), board_size * tile_size + (margin * 2)));
    ImGui::Begin("Board", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImDrawList *drawlist = ImGui::GetWindowDrawList();
    ImVec2 window_pos = ImGui::GetWindowPos();
    ImVec2 board_start = ImVec2(window_pos.x + margin, window_pos.y + margin);

    for (int x = 0; x < board_size; ++x)
    {
        for (int y = 0; y < board_size; ++y)
        {
            ImVec2 top_left = ImVec2(board_start.x + x * tile_size, board_start.y + y * tile_size);
            ImVec2 bottom_right = ImVec2(top_left.x + tile_size, top_left.y + tile_size);

            const uint32_t pieceIndex = x * board_size + y;
            const float currentPiece = board_state[pieceIndex];

            ImU32 color = ((x + y) % 2 == 0) ? BOARD_LIGHT : BOARD_DARK;

            ImVec2 center(top_left.x + tile_size / 2, top_left.y + tile_size / 2);
            
            if (isClicked(center, tile_size) && currentPiece != 0)
                currentSelectedPiece = pieceIndex;

            if (pieceIndex == currentSelectedPiece)
                color = IM_COL32(200, 125, 75, 255);

            drawlist->AddRectFilled(top_left, bottom_right, color);
            ImVec2 text_pos = ImVec2(board_start.x + x * tile_size + (tile_size / 20), board_start.y + (y + 1) * tile_size - (tile_size / 4)); 

            ImGui::GetWindowDrawList()->AddText(text_pos, ((x + y) % 2 == 0) ? BOARD_DARK : BOARD_LIGHT, getHouse(x, y).c_str());

            if (currentPiece != 0)
                drawPiece(drawlist, currentPiece > 0 ? PLR1_COLOR : PLR2_COLOR, center, pieceIndex);
        }
    }

    ImGui::End();
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