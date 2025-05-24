#include "application.hpp"

#include <imgui/imgui/backends/imgui_impl_glfw.h>
#include <imgui/imgui/backends/imgui_impl_opengl3.h>

#include <iostream>
#include <string>
#include "ntars/models/DenseNetwork.hpp"
#include "ntars/base/data.hpp"
#include <thread>

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

void trainCheckersNetwork()
{
    //NTARS::DenseNeuralNetwork network{{64, 512, 256, 128, 64}, "CheckinTime"};
    NTARS::DenseNeuralNetwork network{"CheckinTime.json"};

    const size_t batch_size = 500;
    float learningRate = 0.5;

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

    for (int32_t epoch = 1; epoch <= 2; ++epoch)
    {
        for (auto& minibatch : batches)
        {
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            result = network.trainCPU(minibatch, learningRate);
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
        std::uniform_int_distribution<uint32_t> dist(0, 60000); 
        uint32_t guess = dist(gen);

        std::vector<uint8_t>& image = dataset.training_images.at(guess);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 28, 28, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, image.data());

        return {texture, image, guess};
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
                const int expectedLabel = static_cast<int>(dataset.training_labels.at(i + j));
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
        
        imguiSetup();
    }

    application::~application()
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();

        ImGui::DestroyContext();

        glDeleteTextures(1, &currentImage);
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

    void application::runCheckers(Checkers& checkers, Board& board, NETWORK::CheckersMinMax& algorithm, NTARS::DenseNeuralNetwork& network, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData)
    {
        if (board.getCurrentTurn() == false)
        {
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            auto move = algorithm.getBestMove(board.board(), trainingData, false);
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

            board.makeMove(move, board.board());
            algorithm.incrementMoveCount();
            board.changeTurn();

            std::cout << "Time to make a move: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
            std::cout << "It checked " << std::to_string(algorithm.getCheckedMoveCount()) << " moves \n";
        }  

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetMainViewport()->Size);
        checkers.drawBoard();

        ImGui::Begin("Extra Info", nullptr, ImGuiWindowFlags_NoCollapse);

        ImGui::Text("Current Turn: %s", board.getCurrentTurn() == false ? "player" : "bot");
        
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

    void application::runAITraining()
    {
        auto size = ImGui::GetWindowViewport()->Size;
        ImGui::SetNextWindowSize(size, ImGuiCond_Always);
        ImGui::Begin("Container", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_DockNodeHost);

            ImGui::SetNextWindowSize(ImVec2(500, 500));
            ImGui::Begin("Info", nullptr, ImGuiWindowFlags_NoBringToFrontOnFocus);
                ImGui::Image(currentImage, ImVec2(500, 500));
                ImGui::Text("Current Number Displayed %i", dataset.training_labels.at(actualLabel));
                ImGui::Text("Current AI Guess: %i", AIGuess);
            ImGui::End();

            drawNetwork();

            // Where it'll control what'll happen in the AI Network Demonstration
            ImGui::Begin("Controllers", nullptr, ImGuiWindowFlags_NoMove);
                if (ImGui::Button("Run Network", ImVec2(150, 50)))
                {
                    numberNetwork.run(std::vector<float>(image.begin(), image.end()));
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

                }
            ImGui::End();
        ImGui::End();
    }

    const size_t displayAmmount = 20;

    void application::drawNetwork()
    {
        ImGui::Begin("Neural Network Display", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove);
     
        ImDrawList* drawlist = ImGui::GetWindowDrawList();
        std::vector<size_t> structure = numberNetwork.getStructure();
        auto& weights = numberNetwork.getWeights();

        ImVec2 windowSize = ImGui::GetWindowSize();
        ImVec2 windowPos = ImGui::GetWindowPos();
        ImVec2 center(windowPos.x + windowSize.x, windowPos.y + windowSize.y * 0.5);

        float layerSpacing = windowSize.x / (structure.size() + 1);
        size_t maxNeurons = *std::max_element(structure.begin(), structure.end());

        maxNeurons = maxNeurons > displayAmmount + 10 ? displayAmmount + 10 : maxNeurons;

        float neuronSpacing = (windowSize.y * 0.9) / (maxNeurons + 1);  

        for (int32_t i = 0; i < structure.size(); ++i)
        {
            size_t neurons = structure[i];

            float layerX = center.x - windowSize.x + (i + 1) * layerSpacing;
            float layerY = center.y;

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
                        drawlist->AddCircleFilled(dotPosition, 2.f, IM_COL32(255, 255, 255, 255));
                    }

                    continue;
                }

                float Nindex = n > displayAmmount && i == 0 ? (((displayAmmount + displayAmmount / 2)) - (neurons - n)) : n;

                float neuronY = layerY + ((Nindex >= maxNeurons ? maxNeurons - (Nindex - maxNeurons) : Nindex) 
                    - ((neurons > displayAmmount) ? maxNeurons : neurons) / 2.0f) * neuronSpacing;

                ImVec2 neuronPosition(layerX, neuronY);
                drawlist->AddCircleFilled(neuronPosition, 5.f, IM_COL32(255, 255, 255, 255));

                if (i < structure.size() - 1)
                {
                    float nextLayerX = center.x - windowSize.x + (i + 2) * layerSpacing;
                    size_t nextNeurons = structure[i + 1];
                    for (int32_t j = 0; j < nextNeurons; ++j)
                    {
                        float nextNeuronY = layerY + (j - (nextNeurons > 30 ? maxNeurons : nextNeurons) / 2.0f) * neuronSpacing;
                        ImVec2 nextNeuronPos(nextLayerX, nextNeuronY);
                        drawlist->AddLine(neuronPosition, nextNeuronPos, IM_COL32(255, 255, 255, 120), 0.5f);
                    }
                }
            }
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
            part = CurrentPart::CHECKERS;
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

        Board board{board_size};
        Checkers checkers(board, 70.f);

        NETWORK::CheckersMinMax algorithm(8, board);
        NTARS::DenseNeuralNetwork network{"CheckinTime.json"}; 

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


