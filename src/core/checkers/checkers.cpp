#include "checkers.hpp"
#include <string>
#include <iostream>

#include <unordered_map>
#include <unordered_set>

#include <algorithm>

#include <cstdlib>
#include <ctime>

// Player Colors
#define PLR1_COLOR IM_COL32(225, 0, 0, 255)     // Red (Player 1)
#define PLR2_COLOR IM_COL32(225, 208, 150, 255) // Beige/Sepia (Player 2)

// Board Colors
#define BOARD_LIGHT IM_COL32(222, 184, 135, 255) // Light Wood (Beige/Brown)
#define BOARD_DARK IM_COL32(139, 69, 19, 255)    // Dark Brown (Classic Wood)

Checkers::Checkers(BitBoard& board, const float tile_size)
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

// global variables
int32_t currentSelectedPiece = -1;
std::vector<BitMove> movesPossibleCurrentPiece{};

void Checkers::handleNetworkAction(std::vector<float>& activations, NETWORK::CheckersMinMax& algorithm)
{
/*     BoardStruct& board_state = board.bitboard();
    std::vector<BitMove> currentMoves = board.getMoves(board_state, true);
    BitMove selectedMove{0, 0};

    std::vector<uint32_t> sortedIndices(board.getSize() * board.getSize());
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
    std::sort(sortedIndices.begin(), sortedIndices.end(), [&](uint32_t a, uint32_t b) {
        return activations[a] > activations[b]; 
    });

    for (uint32_t choice : sortedIndices)
    {
        std::vector<BitMove> candidateMoves;

        std::copy_if(currentMoves.begin(), currentMoves.end(), std::back_inserter(candidateMoves), [&](const BitMove& move){
            return move.moveMask == choice;
        });

        if (!candidateMoves.empty()) // Found at least one move
        {
            algorithm.sortMoves(board_state, candidateMoves);
            selectedMove = candidateMoves.front();
            break; 
        }

        activations[choice] = 0.0f; 
    }

    if (selectedMove.indexMask == 0 && selectedMove. == 0)
        return;

    board.makeMove(selectedMove, board_state);
    board.changeTurn(); */
}

void Checkers::handleAction(uint64_t pieceIndex, uint64_t moveIndex)
{
    if (pieceIndex == 0 || moveIndex == 0 || pieceIndex == moveIndex)
        return;

    auto moveToMake = std::find_if(movesPossibleCurrentPiece.begin(), movesPossibleCurrentPiece.end(), 
        [&](const BitMove& a){ return a.indexMask == pieceIndex && a.moveMask == moveIndex; });

    if (moveToMake != movesPossibleCurrentPiece.end())
    {
        board.makeMove(*moveToMake, board.bitboard());
        board.changeTurn();

        currentSelectedPiece = -1;
        movesPossibleCurrentPiece.clear();
    }
}

char winner[128] = "";

void Checkers::drawGameOverScreen()
{
    if (board.isGameOver(board.getCurrentTurn(), board.bitboard()) || board.isGameOver(!board.getCurrentTurn(), board.bitboard())) 
    {
        ImGui::OpenPopup("Game Over");

        ImVec2 windowSize = ImGui::GetWindowSize();

        if (ImGui::BeginPopupModal("Game Over", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            std::string winnerText = (board.getCurrentTurn() == false) ? "Bot Wins!" : "Player Wins!";
            ImGui::TextColored(ImVec4(255, 215, 0, 255), "%s", winnerText.c_str()); 

            ImGui::Text("Please insert the name of the Winner!");
            if (ImGui::InputText("Winner name...", winner, 128 * sizeof(char), ImGuiInputTextFlags_EnterReturnsTrue))
            {
                incrementLeaderboard(std::string(winner));
            }

            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(255, 50, 50, 255)); 
            if (ImGui::Button("Restart Game"))
            {
                board.restart(); 
                ImGui::CloseCurrentPopup();
            }
            ImGui::PopStyleColor();

            ImGui::EndPopup();
        }
    }
}

void Checkers::drawBoard()
{
    BoardStruct& board_state = board.bitboard();
    uint32_t board_size = board.getSize();
    bool currentTurn = board.getCurrentTurn();

    ImGui::SetNextWindowSize(ImVec2(board_size * tile_size + (margin * 2), board_size * tile_size + (margin * 2)));
    ImGui::Begin("Board", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImDrawList *drawlist = ImGui::GetWindowDrawList();
    ImVec2 window_pos = ImGui::GetWindowPos();
    ImVec2 board_start = ImVec2(window_pos.x + margin, window_pos.y + margin);

    int64_t moveIndex = getCellMouseAt(board_start);

    uint64_t moveMask = (1ULL << moveIndex);
    bool isMaxPiece = (board_state.board_state[MAX] & moveMask) != 0;

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && moveIndex != 0 && ((currentTurn && isMaxPiece) || (!currentTurn && !isMaxPiece)))
    {
        currentSelectedPiece = moveIndex;
        movesPossibleCurrentPiece = board.getMovesForPiece((1ULL << currentSelectedPiece), board_state);
    }

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsMouseReleased(ImGuiMouseButton_Left))
    {
        moveIndex = getCellMouseAt(board_start);
        moveMask = (1ULL << moveIndex);
        handleAction((1ULL << currentSelectedPiece), moveMask);

        if (!(board_state.occupiedBoard & moveMask))
        {
            currentSelectedPiece = -1;
            movesPossibleCurrentPiece.clear();
        }
    }

    for (int x = 0; x < board_size; ++x)
    {
        for (int y = 0; y < board_size; ++y)
        { 
            ImVec2 top_left = ImVec2(board_start.x + x * tile_size, board_start.y + y * tile_size);
            ImVec2 bottom_right = ImVec2(top_left.x + tile_size, top_left.y + tile_size);

            uint64_t pieceIndex = x * board_size + y;

            ImU32 color = ((x + y) % 2 == 0) ? BOARD_LIGHT : BOARD_DARK;
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
                    if (move.indexMask == (1ULL << pieceIndex) || move.moveMask == (1ULL << pieceIndex))
                        color = ((x + y) % 2 == 0) ? BOARD_LIGHT * 2 : BOARD_DARK * 2;

                drawlist->AddRectFilled(top_left, bottom_right, color);
            }

            ImVec2 text_pos = ImVec2(board_start.x + x * tile_size + (tile_size / 20), board_start.y + (y + 1) * tile_size - (tile_size / 4));
            ImGui::GetWindowDrawList()->AddText(text_pos, ((x + y) % 2 == 0) ? BOARD_DARK : BOARD_LIGHT, getHouse(x, y).c_str());
        }
    }

    ImVec2 selectedCenter;
    ImU32 selectedColor;

    for (int i = 0; i < board_size * board_size; ++i)
    {
        uint64_t currentIndex = (1ULL << i);

        const bool isMax = board_state.board_state[MAX] & currentIndex;
        ImU32 pieceColor = isMax ? PLR1_COLOR : PLR2_COLOR;

        ImVec2 top_left = ImVec2(board_start.x + (i / board_size) * tile_size, board_start.y + (i % board_size) * tile_size);
        ImVec2 bottom_right = ImVec2(top_left.x + tile_size, top_left.y + tile_size);
        
        ImVec2 center(top_left.x + tile_size / 2, top_left.y + tile_size / 2);

        if (i == currentSelectedPiece) {
            selectedCenter = center;
            selectedColor = pieceColor;
            continue;
        }

        if (board_state.occupiedBoard & currentIndex)
            drawPiece(drawlist, pieceColor, center, i);
    }

    if (currentSelectedPiece != -1)
    {
        drawPiece(drawlist, selectedColor, selectedCenter, currentSelectedPiece);
    }

    ImGui::End();

    drawGameOverScreen();
}

void Checkers::drawLeaderboard()
{
    const auto& leaders = leaderboard[selectedDifficulty];

    ImGui::Begin("Leaderboard");
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]); 
        const char* difficultyStr = selectedDifficulty == Difficulty::Easy ? "Easy" : 
                                selectedDifficulty == Difficulty::Medium ? "Medium" : "Hard";
        ImGui::TextColored(ImVec4(1.0f, 0.843f, 0.0f, 1.0f), "%s Difficulty Leaderboard", difficultyStr);
        ImGui::PopFont();
        ImGui::Separator();

        const float TABLE_WIDTH = 300.0f;
        ImGui::SetNextItemWidth(TABLE_WIDTH);
        
        if (ImGui::BeginTable("LeaderboardTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
        {
            ImGui::TableSetupColumn("Rank", ImGuiTableColumnFlags_WidthFixed, 50.0f);
            ImGui::TableSetupColumn("Player", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed, 70.0f);
            ImGui::TableHeadersRow();

            for (int32_t i = 0; i < leaders.size(); ++i)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::Text("#%d", i + 1);
                
                ImGui::TableSetColumnIndex(1);
                if (i < 3)
                {
                    ImVec4 medalColor = (i == 0) ? ImVec4(1.0f, 0.843f, 0.0f, 1.0f) :  // Gold
                                    (i == 1) ? ImVec4(0.753f, 0.753f, 0.753f, 1.0f) :  // Silver
                                                ImVec4(0.804f, 0.498f, 0.196f, 1.0f);  // Bronze
                    ImGui::TextColored(medalColor, "%s", leaders[i].first.c_str());
                }
                else
                {
                    ImGui::Text("%s", leaders[i].first.c_str());
                }
                
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%d", leaders[i].second);
            }
            
            ImGui::EndTable();
        }

        if (leaders.empty())
        {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No scores recorded yet!");
        }

    ImGui::End();
}

void Checkers::incrementLeaderboard(const std::string name)
{
    auto& leaders = leaderboard[selectedDifficulty];
    auto it = std::find_if(leaders.begin(), leaders.end(), [&](const auto& entry) {
        return entry.first == name;
    });

    if (it != leaders.end()) {
        it->second++;
    } else {
        leaders.emplace_back(name, 1);
    }

    std::sort(leaders.begin(), leaders.end(), [&](const auto& a, const auto& b ){
        return a.second > b.second;
    });
}

void Checkers::drawInfo(int32_t boardScore, Bot& bot)
{
    ImGui::Begin("Info");
        const char* difficulties[] = { "Easy", "Medium", "Hard" };

        ImGui::BeginChild("##Options");
            ImGui::Text("Select Difficulty:");
            int difficultyIndex = static_cast<int>(selectedDifficulty);
            if (ImGui::Combo("##DifficultyCombo", &difficultyIndex, difficulties, IM_ARRAYSIZE(difficulties))) {
                selectedDifficulty = static_cast<Difficulty>(difficultyIndex);
            }
        ImGui::EndChild();
        drawLeaderboard();
    ImGui::End();

    bot.drawBot(true);
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

void Checkers::drawPiece(ImDrawList *drawlist, const ImU32 color, ImVec2 center, const uint32_t id)
{
    std::string windowId = "Piece##" + std::to_string(id);

    ImGui::SetNextWindowSize(ImVec2(tile_size, tile_size));
    ImGui::SetNextWindowPos(ImVec2(center.x - tile_size / 2, center.y - tile_size / 2));
    ImGui::Begin(windowId.c_str(), nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBackground);
    
    if (id == currentSelectedPiece && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        ImVec2 mousePos = ImGui::GetMousePos();
        center = mousePos; 
    }

    const float radius = (tile_size / 2) - 10;

    drawlist->AddCircleFilled(center, radius, color);
    drawlist->AddCircle(center, radius, color / 3, 0, tile_size / 10);
    drawlist->AddCircle(center, radius / 1.5, color / 3, 0, tile_size / 30);

    if (board.bitboard().queenBoard & (1ULL << id))
        drawCrown(ImGui::GetWindowDrawList(), ImVec2(center.x, center.y + tile_size / 12.f));

    ImGui::End();
}