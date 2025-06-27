#include "application.hpp"

#include <imgui/imgui/backends/imgui_impl_glfw.h>
#include <imgui/imgui/backends/imgui_impl_opengl3.h>

#include <iostream>
#include <string>
#include "ntars/models/DenseNetwork.hpp"
#include "ntars/base/data.hpp"

#include "core/audio.hpp"
#include "core/gl/gltexture.hpp"

#include <random>
#include <chrono>
#include "../config.h"

    /* float learning_rate_threshold = 0.9;
    float result = 0.0;
    for (auto& minibatch : batches)
    {
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        result = network.train(minibatch, learningRate);
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

        if (result >= learning_rate_threshold)
        {
            learning_rate_threshold += 1 - (learning_rate_threshold / 2);
            learningRate /= 2;
        }

        std::cout << "Result (Rights / Total): " << std::to_string(result) << std::endl;
        std::cout << "it took " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << " seconds to complete this training session" << std::endl;
    } */

GLuint currentImage;
std::vector<uint8_t> image;
uint32_t actualLabel;

int32_t AIGuess = -1;

bool finishedTraining = false;

void trainCheckersNetwork()
{
    //NTARS::DenseNeuralNetwork network{{64, 1000, 500, 64}, "CheckinTime"};
    NTARS::DenseNeuralNetwork network{"CheckinTime.json"};

    const size_t batch_size = 300;
    float learningRate = 1.0;

    std::vector<NTARS::DATA::TrainingData<std::vector<float>>> rawData = NTARS::DATA::loadDataListJSON<std::vector<float>>("CheckersData");
    std::vector<std::vector<NTARS::DATA::TrainingData<std::vector<float>>>> batches{};

    for (size_t i = 0; i < rawData.size(); i += batch_size)
    {
        std::vector<NTARS::DATA::TrainingData<std::vector<float>>> batch;
        auto start = rawData.begin() + i;
        auto end = (i + batch_size < rawData.size()) ? (start + batch_size) : rawData.end();
        batch.insert(batch.end(), start, end);
        batches.push_back(std::move(batch));
    }

    float learning_rate_threshold = 0.9;
    float result = 0.0;

    for (int32_t epoch = 0; epoch < 10; ++epoch)
    {
        for (auto& minibatch : batches)
        {
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            result = network.trainCPU(minibatch, learningRate, false);
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    
            if (result >= learning_rate_threshold)
            {
                learning_rate_threshold += 1 - (learning_rate_threshold / 2);
                learningRate /= 2;
            }
    
            std::cout << "Result (Rights / Total): " << std::to_string(result) << std::endl;
            std::cout << "it took " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << " seconds to complete this training session" << std::endl;
        }
        
        network.save();
    }
}

    std::tuple<GLuint, std::vector<uint8_t>, uint32_t> getRandomImage(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& dataset)
    {
        static GLuint texture;

        if (texture)
            glDeleteTextures(1, &texture);

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dist(0, dataset.test_images.size()); 
        uint32_t guess = dist(gen);

        std::vector<uint8_t>& image = dataset.test_images.at(guess);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 28, 28, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, image.data());

        return {texture, image, guess};
    }

    void loadAudios()
    {
        std::string audioPath = "C:\\Users\\gabri\\OneDrive\\Documentos\\GitHub\\Tars-AI\\src\\resources\\audio";
        std::string movePath = audioPath + "/move.mp3";
        std::string capturePath = audioPath + "/capture.mp3";
        std::string promotePath = audioPath + "/promote.mp3";

        core::SoundHandle::add("move", movePath.c_str());
        core::SoundHandle::add("capture", capturePath.c_str());
        core::SoundHandle::add("promote", promotePath.c_str());

        // Bot Sounds

        //Terminator
        core::SoundHandle::add("terminator_neutral",       "audio/terminator/neutral.mp3");
        core::SoundHandle::add("terminator_capture",       "audio/terminator/capture.mp3");
        core::SoundHandle::add("terminator_multi_capture", "audio/terminator/multi_capture.mp3");
        core::SoundHandle::add("terminator_good_move",     "audio/terminator/good_move.mp3");
        core::SoundHandle::add("terminator_bad_move",      "audio/terminator/bad_move.mp3");
        core::SoundHandle::add("terminator_win",           "audio/terminator/win.mp3");
        core::SoundHandle::add("terminator_lose",          "audio/terminator/lose.mp3");
        core::SoundHandle::add("terminator_taunt",         "audio/terminator/taunt.mp3");
        core::SoundHandle::add("terminator_encouragement", "audio/terminator/encouragement.mp3");
        core::SoundHandle::add("terminator_surprise",      "audio/terminator/surprise.mp3");
    }

    BoardStruct generateRandomBoard()
    {
        const int32_t maxPiecesPerSide = 12;
        BoardStruct board;

        uint64_t occupied = 0ULL;
        board.board_state[MIN] = 0ULL;
        board.board_state[MAX] = 0ULL;
        board.queenBoard = 0ULL;

        std::vector<uint8_t> legalIndices;
        for (uint8_t i = 0; i < 64; ++i)
        {
            if ((i + (i / 8)) % 2 == 1) // dark squares only
                legalIndices.push_back(i);
        }

        std::shuffle(legalIndices.begin(), legalIndices.end(), std::mt19937{ std::random_device{}() });

        int32_t totalMax = std::rand() % (maxPiecesPerSide + 1);
        int32_t totalMin = std::rand() % (maxPiecesPerSide + 1);

        int32_t index = 0;
        for (int i = 0; i < totalMax; ++i)
        {
            uint64_t bit = 1ULL << legalIndices[index++];
            board.board_state[MAX] |= bit;
            if (std::rand() % 4 == 0) // 25% chance it's a queen
                board.queenBoard |= bit;
        }

        for (int i = 0; i < totalMin; ++i)
        {
            uint64_t bit = 1ULL << legalIndices[index++];
            board.board_state[MIN] |= bit;
            if (std::rand() % 4 == 0)
                board.queenBoard |= bit;
        }

        board.occupiedBoard = board.board_state[MAX] | board.board_state[MIN];

        return board;
    }

    void gatherCheckersData(NETWORK::CheckersMinMax& algorithm)
    {
        size_t checkersDataCurrent = 0;
        const size_t targetCheckersData = 100000;

        std::vector<NTARS::DATA::TrainingData<std::vector<float>>> trainingData;

        while (checkersDataCurrent < targetCheckersData)
        {
            BoardStruct trainingBoard = generateRandomBoard();

            algorithm.getBestMove(trainingBoard, trainingData, std::rand() % 2 == 0, 0.0f);

            checkersDataCurrent = trainingData.size();
        }

        NTARS::DATA::saveListDataJSON(trainingData, "CheckersData");
    }


namespace core
{
    application::application(const std::string& title, uint32_t width, uint32_t height)
    {
        //NeuralNetworkTrain();
        //trainCheckersNetwork();
        
        dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

        const auto& data = dataset.training_images;
        for (size_t i = 0; i < data.size() / batch_size; ++i)
        {
            std::vector<NTARS::DATA::TrainingData<std::vector<float>>> miniBatch{};
            for (size_t j = 0; j < batch_size && (i + j) < data.size(); ++j)
            {
                NTARS::DATA::TrainingData<std::vector<float>> newData{};
                newData.data = std::vector<float>(data.at(i + j).begin(), data.at(i + j).end());

                std::vector<float> expected(10, 0.0);
                const int32_t expectedLabel = static_cast<int32_t>(dataset.training_labels.at(i + j));
                expected.at(expectedLabel) = 1.0;
                newData.label = expected;
        
                miniBatch.emplace_back(std::move(newData));
            }
            batches.emplace_back(std::move(miniBatch));
        }

        window = std::make_unique<window_t>(title, width, height);

        auto imageTuple = getRandomImage(dataset);
        currentImage = std::get<0>(imageTuple);
        image = std::get<1>(imageTuple);
        actualLabel = std::get<2>(imageTuple);
        
        loadAudios();
        imguiSetup();
        initBots();
    }

    application::~application()
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();

        ImGui::DestroyContext();

        glDeleteTextures(1, &currentImage);

        if (networkThread != nullptr)
        {
            if (networkThread->joinable())
            {
                networkThread->join();
                delete networkThread;
                networkThread = nullptr;
            }
        }
    }

    void application::imguiSetup()
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io; 
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window->window(), true);
        ImGui_ImplOpenGL3_Init("#version 450");
    }

    void application::imguiNewFrame()
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void application::imguiEndFrame()
    {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    std::atomic<bool> checkersThreadRunning = false;

    void application::initBots()
    {
        // Terminator
        BotInfo terminator;
        terminator.name = "Terminator";
        terminator.blunderChance = 0.01f;
        terminator.speeches = {
            { "Processando... Continue a jogada.", "terminator_neutral", SpeechType::Neutral },
            { "Alvo identificado. Eliminado sem dó.", "terminator_capture", SpeechType::Capture },
            { "Múltiplas ameaças neutralizadas. Nem tenta fugir.", "terminator_multi_capture", SpeechType::MultiCapture },
            { "Aceitável. Mas, no fim das contas, inútil.", "terminator_good_move", SpeechType::GoodMove },
            { "Erro estratégico detectado. Ajuste ou será exterminado.", "terminator_bad_move", SpeechType::BadMove },
            { "Missão cumprida. Você foi finalizado.", "terminator_win", SpeechType::Win },
            { "Esse resultado é ilógico. Recalculando tática...", "terminator_lose", SpeechType::Lose },
            { "Pode lutar, mas seu destino já está determinado.", "terminator_taunt", SpeechType::Taunt },
            { "Continue. Suas chances de sobreviver são mínimas.", "terminator_encouragement", SpeechType::Encouragement },
            { "Inesperado. Você tá aprendendo... interessante.", "terminator_surprise", SpeechType::Surprise }
        };
        bots.emplace_back(Bot(terminator, "C:\\Users\\gabri\\OneDrive\\Documentos\\GitHub\\Tars-AI\\src\\resources\\images\\terminator.png"));

        // GlitchBot
        BotInfo glitch;
        glitch.name = "Et Bilu";
        glitch.blunderChance = 0.35f;
        glitch.speeches = {
            { "Uhh... Move? Maybe? Okay go!", "neutral", SpeechType::Neutral },
            { "Whoa! That actually worked?!", "capture", SpeechType::Capture },
            { "Double kill! Pure chaos.", "multi_capture", SpeechType::MultiCapture },
            { "Heh, total accident. I'll take it.", "good_move", SpeechType::GoodMove },
            { "Oops. Lag... right?", "bad_move", SpeechType::BadMove },
            { "I win! Wait... that was supposed to happen?", "win", SpeechType::Win },
            { "404: Skill not found.", "lose", SpeechType::Lose },
            { "Gotta admit, that was kinda slick.", "taunt", SpeechType::Taunt },
            { "You might actually pull this off. Weird.", "encouragement", SpeechType::Encouragement },
            { "Unexpected move. Processing glitch...", "surprise", SpeechType::Surprise }
        };
        bots.emplace_back(Bot(glitch, "C:\\Users\\gabri\\OneDrive\\Documentos\\GitHub\\Tars-AI\\src\\resources\\images\\terminator.png"));

        // Strategos
        BotInfo strategos;
        strategos.name = "Strategos";
        strategos.blunderChance = 0.05f;
        strategos.speeches = {
            { "The board is set. Your move.", "neutral", SpeechType::Neutral },
            { "Sacrifices are inevitable.", "capture", SpeechType::Capture },
            { "A calculated strike across the ranks.", "multi_capture", SpeechType::MultiCapture },
            { "Elegant play. But the war isn't over.", "good_move", SpeechType::GoodMove },
            { "Flawed tactics. You will regret that.", "bad_move", SpeechType::BadMove },
            { "Victory, as predicted.", "win", SpeechType::Win },
            { "Impressive. That deviation was not forecasted.", "lose", SpeechType::Lose },
            { "I anticipated your ambition. It will cost you.", "taunt", SpeechType::Taunt },
            { "Adaptation is a sign of strength. Continue.", "encouragement", SpeechType::Encouragement },
            { "Ah... you've shifted your tempo. Noted.", "surprise", SpeechType::Surprise }
        };
        bots.emplace_back(Bot(strategos, "C:\\Users\\gabri\\OneDrive\\Documentos\\GitHub\\Tars-AI\\src\\resources\\images\\terminator.png"));

        // Echo
        BotInfo echo;
        echo.name = "Echo";
        echo.blunderChance = 0.15f;
        echo.speeches = {
            { "Your turn. Lets play clean.", "neutral", SpeechType::Neutral },
            { "Bait taken. Nicely sprung.", "capture", SpeechType::Capture },
            { "That was surgical. Respect.", "multi_capture", SpeechType::MultiCapture },
            { "You surprise me sometimes.", "good_move", SpeechType::GoodMove },
            { "Let that one slip. My bad.", "bad_move", SpeechType::BadMove },
            { "See you on the next board.", "win", SpeechType::Win },
            { "One loss doesnt mean defeat.", "lose", SpeechType::Lose },
            { "Cocky much? Lets see you back it up.", "taunt", SpeechType::Taunt },
            { "Keep going. You've got something here.", "encouragement", SpeechType::Encouragement },
            { "Bold move. Didn't expect that line.", "surprise", SpeechType::Surprise }
        };
        bots.emplace_back(Bot(echo, "C:\\Users\\gabri\\OneDrive\\Documentos\\GitHub\\Tars-AI\\src\\resources\\images\\terminator.png"));

        BotInfo netrix;
        netrix.name = "Netrix"; // short for Neural Execution Terminal Replica Interface eXperiment
        netrix.blunderChance = 0.1f;

        netrix.speeches = {
            { "BEEP. BOOP. Ready to calculate.", "neutral", SpeechType::Neutral },
            { "Tactical pattern recognized. Executing response.", "capture", SpeechType::Capture },
            { "THREE MOVES. ONE DESTINY. BZZZ!", "multi_capture", SpeechType::MultiCapture },
            { "Hmm. That move has a 73%' admiration score.", "good_move", SpeechType::GoodMove },
            { "Error: Human move not found in database. Likely suboptimal.", "bad_move", SpeechType::BadMove },
            { "Evaluation complete. Victory assured. Shutting down ego.", "win", SpeechType::Win },
            { "LOSS detected. Adjusting neural weights... internally sobbing.", "lose", SpeechType::Lose },
            { "You think you've outsmarted me? Cute.", "taunt", SpeechType::Taunt },
            { "Maintain momentum. Performance graph trending upward.", "encouragement", SpeechType::Encouragement },
            { "Surprise spike in your move complexity. Impressed.", "surprise", SpeechType::Surprise }
        };

        bots.emplace_back(Bot(netrix, "C:\\Users\\gabri\\OneDrive\\Documentos\\GitHub\\Tars-AI\\src\\resources\\images\\neurabot.png"));
    }

    void renderBotSelection(Bot& bot, int selectedBotIndex, int botIndex, std::function<void(int)> onSelect)
    {
        const std::string label = bot.getName();
        const bool isSelected = (selectedBotIndex == botIndex);

        ImGui::BeginChild(label.c_str(), ImVec2(180, 250), ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);

        if (isSelected)
        {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetColorU32(ImGuiCol_FrameBgActive));
            ImVec2 winPos = ImGui::GetWindowPos();
            ImVec2 winSize = ImGui::GetWindowSize();
            ImVec2 winEnd = ImVec2(winPos.x + winSize.x, winPos.y + winSize.y);
            ImGui::GetWindowDrawList()->AddRect(
                winPos, winEnd,
                ImGui::GetColorU32(ImGuiCol_ButtonHovered), 5.0f, 0, 2.5f);
        }

        // Bot Image
        ImGui::Dummy(ImVec2(0, 5));
        ImGui::SetCursorPosX((180 - 130) * 0.5f);
        ImGui::Image(bot.getImage()->getId(), ImVec2(130, 130));

        // Bot Name
        ImGui::Dummy(ImVec2(0, 8));
        ImGui::SetCursorPosX((180 - ImGui::CalcTextSize(label.c_str()).x) * 0.5f);
        ImGui::Text("%s", label.c_str());

        // Select Button
        ImGui::Dummy(ImVec2(0, 8));
        ImGui::SetCursorPosX((180 - 100) * 0.5f);
        if (ImGui::Button(("Choose##" + label).c_str(), ImVec2(100, 28)))
        {
            onSelect(botIndex);
        }

        if (isSelected) ImGui::PopStyleColor();

        ImGui::EndChild();
    }

    void application::checkersBotSelectionMenu(BitBoard& board)
    {
        ImGui::Begin("Bot Selection Menu", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        for (size_t i = 0; i < bots.size(); ++i)
        {
            renderBotSelection(bots[i], currentBotIndex, static_cast<int>(i), [&](int newIndex) {
                currentBotIndex = newIndex;
            });
            ImGui::SameLine();
        }

        if (ImGui::Button("Enter Checkers Game", ImVec2(100, 20)))
        {
            part = CurrentPart::CHECKERS;
            board.restart();
        }

        ImGui::End();
    }

    float tileSize = 100.f;
    bool newMove = true;

    void application::runCheckers(Checkers& checkers, BitBoard& board, NETWORK::CheckersMinMax& algorithm, NTARS::DenseNeuralNetwork& network, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData)
    {
        auto& currentBot = bots.at(currentBotIndex);
        if (board.getCurrentTurn() && !checkersThreadRunning && currentBotIndex != 4 && !board.isGameOver(true, board.bitboard()))
        {
            std::thread aiThread([&]() 
            {  
                checkersThreadRunning = true;
                
                std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                auto move = algorithm.getBestMove(board.bitboard(), trainingData, true, currentBot.getBlunderChance());
                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

                board.makeMove(move, board.bitboard());
                algorithm.incrementMoveCount();

                board.changeTurn();

                std::cout << "Time to make a move: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
                std::cout << "It checked " << std::to_string(algorithm.getCheckedMoveCount()) << " moves \n";

                checkersThreadRunning = false;
                newMove = true;
            });

            if (aiThread.joinable())
                aiThread.join(); 

            currentBot.stopSpeech(currentBot.getCurrentSpeech());
            currentBot.handleSpeech(algorithm.getCurrentBoardScore());
        } 
        else if (board.getCurrentTurn() && currentBotIndex == 4) // Neural Network
        {
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            auto activations = network.run(board.vectorBoard(board.bitboard()));
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

            std::cout << "Time to make a move: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns" << std::endl;

            checkers.handleNetworkAction(activations, algorithm);

            currentBot.stopSpeech(currentBot.getCurrentSpeech());
            currentBot.handleSpeech(algorithm.getCurrentBoardScore());
        }

        if (board.isGameOver(true, board.bitboard()))
        {
            currentBot.chooseSpeech(SpeechType::Lose);
        }
        else if (board.isGameOver(false, board.bitboard()))
        {
            currentBot.chooseSpeech(SpeechType::Win);
        }

        if (newMove)
        {
            currentBot.playSpeech(currentBot.getCurrentSpeech());
            newMove = false;
        }

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetMainViewport()->Size);
        checkers.drawBoard(currentBot);
        checkers.drawInfo(algorithm.getCurrentBoardScore(), bots.at(currentBotIndex));

        ImGui::Begin("Extra Info##2", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

        ImGui::Text("Current Turn: %s", board.getCurrentTurn() == false ? "player" : "bot");
        ImGui::Text("Board Evaluation: %2.f", static_cast<float>(algorithm.getCurrentBoardScore()));
        
        if (ImGui::SliderFloat("Board Size", &tileSize, 50.f, 200.f))
        {
            checkers.setNewTileSize(tileSize);
        }

        if (ImGui::Button("Exit To Menu", ImVec2(300, 50)))
        {
            part = CurrentPart::MENU;
        }

        ImGui::End();
    }

    void application::runPresentation()
    {
        auto size = ImGui::GetWindowViewport()->Size;
        ImGui::SetNextWindowSize(size, ImGuiCond_Always);
        ImGui::Begin("Container", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);

        ImGui::End();
    }

    std::vector<float> activations;

    void application::runAITraining()
    {
        auto size = ImGui::GetWindowViewport()->Size;
        ImGui::SetNextWindowSize(size, ImGuiCond_Always);
        ImGui::Begin("Container", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_DockNodeHost);

            ImGui::SetNextWindowSize(ImVec2(500, 500));
            ImGui::Begin("Info", nullptr, ImGuiWindowFlags_NoBringToFrontOnFocus);
                ImGui::SetWindowFontScale(1.2f); 

                ImGui::Image(currentImage, ImVec2(450, 450));
                ImGui::Text("Current Number Displayed %i", dataset.test_labels.at(actualLabel));
                ImGui::Text("Current AI Guess: %i", AIGuess);

                std::vector<std::pair<size_t, float>> sortedActivations{};
                for (size_t i = 0; i < activations.size(); ++i)
                {
                    sortedActivations.emplace_back(i, activations[i]);
                }

                std::sort(sortedActivations.begin(), sortedActivations.end(), 
                        [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
                            return a.second > b.second; 
                        });

                ImGui::Separator();

                float totalActivations = std::accumulate(activations.begin(), activations.end(), 0.0f);

                ImGui::Separator();
                ImGui::Text("AI Confidence Scores:");

                for (const auto& [index, confidence] : sortedActivations)
                {
                    float normalizedConfidence = (totalActivations > 0.0f) ? (confidence / totalActivations) * 100.0f : 0.0f;
                    
                    ImGui::Text("Class %i: %.2f%%", index, normalizedConfidence);
                    ImGui::ProgressBar(normalizedConfidence / 100.0f, ImVec2(200, 30));
                }

            ImGui::End();

            drawNetwork();

            // Where it'll control what'll happen in the AI Network Demonstration
            ImGui::Begin("Controllers", nullptr, ImGuiWindowFlags_NoMove);
                if (ImGui::Button("Run Network", ImVec2(150, 50)))
                {
                    activations = numberNetwork.run(std::vector<float>(image.begin(), image.end()));
                    AIGuess = std::distance(activations.begin(), std::max_element(activations.begin(), activations.end()));
                }
                ImGui::SameLine();
                if (ImGui::Button("Choose Random Data", ImVec2(150, 50)))
                {
                    auto imageTuple = getRandomImage(dataset);
                    currentImage = std::get<0>(imageTuple);
                    image = std::get<1>(imageTuple);
                    actualLabel = std::get<2>(imageTuple);
                }
                ImGui::SameLine();
                if (ImGui::Button("Start Training", ImVec2(150, 50)))
                {
                    if (finishedTraining)
                    {
                        networkThread->join();
                        
                        delete networkThread;
                        networkThread = nullptr;
                        finishedTraining = false;
                    }

                    networkThread = new std::thread([&](){
                        float result = 0.0;
                        for (auto& minibatch : batches)
                        {
                            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                            result = numberNetwork.trainCPU(minibatch, learningRate);
                            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

                            std::cout << "Result (Rights / Total): " << std::to_string(result) << std::endl;
                            std::cout << "it took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds to complete this training session" << std::endl;

                            activations = numberNetwork.run(std::vector<float>(image.begin(), image.end()));
                            if (result > 0.97)
                                break;
                        }
                        finishedTraining = true;
                    });
                }
                if (ImGui::Button("Exit To Menu", ImVec2(150, 50)))
                {
                    part = CurrentPart::MENU;
                }
            ImGui::End();
        ImGui::End();
    }

    const size_t displayAmmount = 70;
    const float neuronSize = 3.f;

    void application::drawNetwork()
    {
        ImGui::Begin("Neural Network Display", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove);
     
        ImDrawList* drawlist = ImGui::GetWindowDrawList();
        std::vector<size_t> structure = numberNetwork.getStructure();

        const auto& weights = numberNetwork.getWeights();
        const auto& biases = numberNetwork.getBiases();

        ImVec2 windowSize = ImGui::GetWindowSize();
        ImVec2 windowPos = ImGui::GetWindowPos();
        ImVec2 center(windowPos.x + windowSize.x, windowPos.y + windowSize.y * 0.5);

        float layerSpacing = windowSize.x / (structure.size() + 1);
        size_t maxNeurons = *std::max_element(structure.begin(), structure.end());

        maxNeurons = maxNeurons > displayAmmount + 50 ? displayAmmount + 50 : maxNeurons;

        float neuronSpacing = (windowSize.y * 0.9) / (maxNeurons + 1);  

        for (int32_t i = 0; i < structure.size(); ++i)
        {
            size_t neurons = structure[i];
            const std::vector<float> layerActivations = i > 0 ? numberNetwork.getLayers()[i - 1].getActivations() : std::vector<float>{};

            float layerX = center.x - windowSize.x + (i + 1) * layerSpacing;
            float layerY = center.y + 20;

            for (size_t n = 0; n < neurons; ++n)
            {
                if (i == 0 && neurons > displayAmmount && (n >= displayAmmount / 2 && n < neurons - (displayAmmount / 2) + 1))
                {
                    if (n > displayAmmount && n < neurons - displayAmmount)
                        n = neurons - displayAmmount;

                    float dotsY = layerY + (n - (neurons > 30 ? maxNeurons : neurons) / 2.0f) * neuronSpacing;
                    float dotSpacing = 8.f; 
                    ImVec2 dotStart(layerX - 5.f, dotsY);

                    for (int i = 0; i < 3; ++i)
                    {
                        ImVec2 dotPosition(dotStart.x + i * dotSpacing, dotStart.y);
                        drawlist->AddCircleFilled(dotPosition, 1.f, IM_COL32(255, 255, 255, 255));
                    }

                    continue;
                }

                float Nindex = n > displayAmmount && i == 0 ? (((displayAmmount + displayAmmount / 2)) - (neurons - n)) : n;

                float neuronY = layerY + ((Nindex >= maxNeurons ? maxNeurons - (Nindex - maxNeurons) : Nindex) 
                    - ((neurons > displayAmmount) ? maxNeurons : neurons) / 2.0f) * neuronSpacing;

                ImVec2 neuronPosition(layerX, neuronY);
                uint8_t brightness = layerActivations.size() > 0 ? std::max<uint8_t>(20, 255 * layerActivations[std::min<size_t>(0, n - 1)]) : 255;

                if (i < structure.size() - 1)
                {
                    float nextLayerX = center.x - windowSize.x + (i + 2) * layerSpacing;
                    size_t nextNeurons = structure[i + 1];

                    const std::vector<float>& currentLineWeights = weights[i].getElementsRaw();
                    const std::vector<float>& currentLineBiases = biases[i].getElementsRaw();
                    for (int32_t j = 0; j < nextNeurons; ++j)
                    {
                        float nextNeuronY = layerY + (j - (nextNeurons > displayAmmount ? maxNeurons : nextNeurons) / 2.0f) * neuronSpacing;
                        ImVec2 nextNeuronPos(nextLayerX, nextNeuronY);

                        const float& currentBias = currentLineBiases[j];
                        const float& currentWeight = currentLineWeights[j];
                        ImU32 currentLineColor = currentLineWeights[j] > 0 ? IM_COL32(0, 255, 0, (currentWeight + currentBias) * brightness) : IM_COL32(200, 0, 0, (currentWeight + currentBias) * brightness);
                        
                        drawlist->AddLine(neuronPosition, nextNeuronPos, currentLineColor, 0.25f);
                    }
                }

                drawlist->AddCircleFilled(neuronPosition, neuronSize, IM_COL32_WHITE);
            }

            std::string layerLabel;

            if (i == 0)
            {
                layerLabel = "Input Layer (" + std::to_string(neurons) + ")";
            }
            else if (i < structure.size() - 1)
            {
                layerLabel = "Hidden Layer " + std::to_string(i) + " (" + std::to_string(neurons) + ")";
            }
            else
            {
                layerLabel = "Output Layer (" + std::to_string(neurons) + ")";
            }

            ImGui::SetWindowFontScale(1.3f); 
            drawlist->AddText(ImVec2(layerX - 100, layerY - 20 + (displayAmmount + 50) * neuronSize), IM_COL32_WHITE, layerLabel.c_str());
        }

        ImGui::End();
    }

    void application::runMenu()
    {
        ImGui::SetNextWindowSize(ImVec2(450, 350), ImGuiCond_Always);
        ImVec2 centerPos = ImVec2((ImGui::GetIO().DisplaySize.x - 450) * 0.5f, 
                                (ImGui::GetIO().DisplaySize.y - 350) * 0.5f);
                                
        ImGui::SetNextWindowPos(centerPos, ImGuiCond_Always);

        ImGui::Begin("Main Menu", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);
        ImGui::Separator();

        ImGui::Dummy(ImVec2(0.0f, 20.0f));

        ImVec2 buttonSize = ImVec2(320, 60);
        float buttonSpacing = 15.0f;
        float windowWidth = 450.0f;
        float buttonX = (windowWidth - buttonSize.x) * 0.5f; 

        ImGui::SetCursorPos(ImVec2(buttonX, ImGui::GetCursorPosY())); 
        if (ImGui::Button("PRESENTATION", buttonSize))
        {
            part = CurrentPart::AIPRESENTATION; // Change Later
        }

        ImGui::Dummy(ImVec2(0.0f, buttonSpacing));

        ImGui::SetCursorPos(ImVec2(buttonX, ImGui::GetCursorPosY())); 
        if (ImGui::Button("CHECKERS", buttonSize))
        {
            part = CurrentPart::CHECKERS_SELECTION_MENU;
        }

        ImGui::Dummy(ImVec2(0.0f, buttonSpacing));

        ImGui::SetCursorPos(ImVec2(buttonX, ImGui::GetCursorPosY())); 
        if (ImGui::Button("EXIT", buttonSize))
        {
            exit(0);
        }

        ImGui::End();
    }

    void application::run()
    {
        const uint32_t board_size = 8;

        BitBoard board;
        Checkers checkers(board, 100.f);

        NETWORK::CheckersMinMax algorithm(10, board);
        NTARS::DenseNeuralNetwork network{"CheckinTime.json"}; 

        //gatherCheckersData(algorithm);

        std::vector<NTARS::DATA::TrainingData<std::vector<float>>> trainingData;

        while (!window->should_close())
        {
            glClear(GL_COLOR_BUFFER_BIT);
            imguiNewFrame();

            switch(part)
            {
                case CurrentPart::AIPRESENTATION:
                {
                    runAITraining();
                    break;
                }
                case CurrentPart::PRESENTATION:
                {
                    runPresentation();
                    break;
                }
                case CurrentPart::CHECKERS_SELECTION_MENU:
                {
                    checkersBotSelectionMenu(board);
                    break;
                }
                case CurrentPart::CHECKERS:
                {
                    runCheckers(checkers, board, algorithm, network, trainingData);
                    break;
                }
                default: // Menu
                    runMenu();
            }

            imguiEndFrame();
            glfwSwapBuffers(window->window());
            glfwPollEvents();
        }
    }

    
} // namespace core


