#include "checkers.hpp"
#include <string>
#include <iostream>

#include <unordered_map>
#include <unordered_set>

// Player Colors
#define PLR1_COLOR IM_COL32(225, 0, 0, 255)     // Red (Player 1)
#define PLR2_COLOR IM_COL32(225, 208, 150, 255) // Beige/Sepia (Player 2)

// Board Colors
#define BOARD_LIGHT IM_COL32(222, 184, 135, 255) // Light Wood (Beige/Brown)
#define BOARD_DARK IM_COL32(139, 69, 19, 255)    // Dark Brown (Classic Wood)

Checkers::Checkers(Board& board, const float tile_size)
    : board(board), tile_size(tile_size)
{
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

int32_t currentSelectedPiece{-1};
std::vector<Move> movesPossibleCurrentPiece{};

void Checkers::handleNetworkAction(std::vector<float>& activations)
{
    std::vector<float> board_state = board.board();
    Move selectedMove;
    bool foundValidMove = false;

    board.makeMove(selectedMove, board_state);
    board.changeTurn();
}

void Checkers::handleAction(int32_t pieceIndex, int32_t moveIndex)
{
    if (pieceIndex == -1 || moveIndex == -1 || pieceIndex == moveIndex)
        return;

    auto moveToMake = std::find_if(movesPossibleCurrentPiece.begin(), movesPossibleCurrentPiece.end(), 
        [&](const Move& a){ return a.startPos == pieceIndex && a.endPos == moveIndex; });

    if (moveToMake != movesPossibleCurrentPiece.end())
    {
        board.makeMove(*moveToMake, board.board());
        board.changeTurn();
        
        movesPossibleCurrentPiece = {};
        currentSelectedPiece = -1;
    }
}

void Checkers::drawGameOverScreen()
{
    if (board.isGameOver(board.getCurrentTurn(), board.board()) || board.isGameOver(!board.getCurrentTurn(), board.board())) 
    {
        ImGui::OpenPopup("Game Over");

        ImVec2 windowSize = ImGui::GetWindowSize();

        if (ImGui::BeginPopupModal("Game Over", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            std::string winnerText = (board.getCurrentTurn() == false) ? "Bot Wins!" : "Player Wins!";
            ImGui::TextColored(ImVec4(255, 215, 0, 255), "%s", winnerText.c_str()); 

            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(255, 50, 50, 255)); 
            if (ImGui::Button("Restart Game"))
            {
                board.restart(); 
                ImGui::CloseCurrentPopup();
            }
            ImGui::PopStyleColor();
            
            if (ImGui::Button("Exit"))
                exit(0);

            ImGui::EndPopup();
        }
    }
}

void Checkers::drawBoard()
{
    std::vector<float>& board_state = board.board();
    uint32_t board_size = board.getSize();
    bool currentTurn = board.getCurrentTurn();

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
        movesPossibleCurrentPiece = board.getMovesForPiece(currentSelectedPiece, board.board());
    }

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    {
        handleAction(currentSelectedPiece, moveIndex);
    }

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
                for (const auto& move : movesPossibleCurrentPiece)
                    if (move.startPos == pieceIndex || move.endPos == pieceIndex)
                        color = ((x + y) % 2 == 0) ? BOARD_LIGHT * 2 : BOARD_DARK * 2;

                drawlist->AddRectFilled(top_left, bottom_right, color);
            }

            ImVec2 text_pos = ImVec2(board_start.x + x * tile_size + (tile_size / 20), board_start.y + (y + 1) * tile_size - (tile_size / 4));

            ImGui::GetWindowDrawList()->AddText(text_pos, ((x + y) % 2 == 0) ? BOARD_DARK : BOARD_LIGHT, getHouse(x, y).c_str());

            if (currentPiece != 0)
                drawPiece(drawlist, pieceColor, center, pieceIndex);
        }
    }

    ImGui::End();

    drawGameOverScreen();
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

    if (abs(board.board()[id]) == 1)
        drawCrown(ImGui::GetWindowDrawList(), ImVec2(center.x, center.y + tile_size / 12.f));

    ImGui::End();
}