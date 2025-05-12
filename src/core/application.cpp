#include "application.hpp"

#include <imgui/imgui/backends/imgui_impl_glfw.h>
#include <imgui/imgui/backends/imgui_impl_opengl3.h>

#include <iostream>
#include <string>
#include <array>
#include "ntars/models/DenseNetwork.hpp"
#include "ntars/base/data.hpp"
#include "mnist/mnist_reader.hpp"

#include "checkers/checkers.hpp"
#include "checkers/checkersminmax.hpp"

#include <chrono>
#include "../config.h"

/* void NeuralNetworkTrain()
{
    //NTARS::DenseNeuralNetwork network{{784, 256, 128, 10}, "TARS"};
    NTARS::DenseNeuralNetwork network{"TARS.json"};
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::vector<std::vector<NTARS::DATA::TrainingData<std::vector<float>>>> batches{};

    const size_t batch_size = 500;
    float learningRate = 0.2;

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

    float learning_rate_threshold = 0.9;
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
    }

    network.save();
} */

void trainCheckersNetwork()
{
    //NTARS::DenseNeuralNetwork network{{64, 512, 256, 128, 64}, "CheckinTime.json"};
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
            result = network.train(minibatch, learningRate);
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

namespace core
{
    application::application(const std::string& title, uint32_t width, uint32_t height)
    {
        //NeuralNetworkTrain();
        //trainCheckersNetwork();
        window = std::make_unique<window_t>(title, width, height);
        imguiSetup();
    }

    application::~application()
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();

        ImGui::DestroyContext();
    }

    void application::imguiSetup()
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io; 
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

    void application::run()
    {
        const uint32_t board_size = 8;

        Board board{board_size};
        Checkers checkers(board, 70.f);

        NETWORK::CheckersMinMax algorithm(1, board);
        NTARS::DenseNeuralNetwork network{"CheckinTime.json"}; 

        std::vector<NTARS::DATA::TrainingData<std::vector<float>>> trainingData;

        while (!window->should_close())
        {
/*             if (board.getCurrentTurn() == false)
            {
                std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                auto move = algorithm.findBestMove(board.board(), trainingData, true).second;
                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

                board.makeMove(move, board.board());
                algorithm.incrementMoveCount();
                board.changeTurn();
            } */ 

            glClear(GL_COLOR_BUFFER_BIT);
            imguiNewFrame();

            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImGui::GetMainViewport()->Size);
            checkers.drawBoard();

            imguiEndFrame();
            glfwSwapBuffers(window->window());
            glfwPollEvents();
        }
    }
} // namespace core


