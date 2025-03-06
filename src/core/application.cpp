#include "application.hpp"

#include <imgui/imgui/backends/imgui_impl_glfw.h>
#include <imgui/imgui/backends/imgui_impl_opengl3.h>

#include <iostream>
#include <string>
#include <array>
#include "ntars/models/DenseNetwork.hpp"
#include "ntars/base/data.hpp"
#include "mnist/mnist_reader.hpp"

#include "../config.h"

void NeuralNetworkTrain()
{
    //NTARS::DenseNeuralNetwork network{{784, 256, 128, 10}, "TARS"};
    NTARS::DenseNeuralNetwork network{"TARS.json"};
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::vector<std::vector<NTARS::DATA::TrainingData<std::vector<double>>>> batches{};

    const size_t batch_size = 500;
    const double learningRate = 0.1;

    const auto& data = dataset.training_images;
    for (size_t i = 0; i < data.size() / batch_size; ++i)
    {
        std::vector<NTARS::DATA::TrainingData<std::vector<double>>> miniBatch{};
        for (size_t j = 0; j < batch_size && (i + j) < data.size(); ++j)
        {
            NTARS::DATA::TrainingData<std::vector<double>> newData{};
            newData.data = std::vector<double>(data.at(i + j).begin(), data.at(i + j).end());
            newData.label = dataset.training_labels.at(i + j);
    
            miniBatch.emplace_back(std::move(newData));
        }
        batches.emplace_back(std::move(miniBatch));
    }

    double result = 0.0;
    for (auto& minibatch : batches)
    {
        result = network.train(minibatch, learningRate);

        std::cout << "Result (Rights / Total): " << std::to_string(result) << std::endl;
    }

    network.save();
}

namespace core
{
    application::application(const std::string& title, uint32_t width, uint32_t height)
    {
        //NeuralNetworkTrain();
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

    std::tuple<GLuint, std::vector<uint8_t>, uint32_t> getRandomImage(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& dataset)
    {
        static GLuint texture;

        if (texture)
            glDeleteTextures(1, &texture);

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dist(0, 60000); 
        uint32_t guess = dist(gen);

        auto& image = dataset.training_images.at(guess);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 28, 28, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, image.data());

        return {texture, image, guess};
    }

    void application::run()
    {
        mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

        NTARS::DenseNeuralNetwork network{"TARS.json"};

        auto texture_image_tuple = getRandomImage(dataset);
        GLuint texture = std::get<0>(texture_image_tuple);
        uint32_t actualLabelIndex = std::get<2>(texture_image_tuple);

        int32_t aiGuess{-1};

        while (!window->should_close())
        {
            glClear(GL_COLOR_BUFFER_BIT);
    
            imguiNewFrame();
            
            ImGui::Begin("Current Image");
            ImGui::Text("Neural Network Guess: %d", aiGuess);
            if (aiGuess != -1) ImGui::Text("Actual Number: %d", dataset.training_labels.at(actualLabelIndex));

            ImGui::NewLine();
            if (ImGui::Button("Run"))
            {
                texture_image_tuple = getRandomImage(dataset);
                texture = std::get<0>(texture_image_tuple);
                actualLabelIndex = std::get<2>(texture_image_tuple);
                aiGuess = network.run(std::vector<double>(std::get<1>(texture_image_tuple).begin(), std::get<1>(texture_image_tuple).end()));
            }
            ImGui::Image((ImTextureID)texture, ImVec2(280, 280)); 
            ImGui::End();
    
            imguiEndFrame();
    
            glfwSwapBuffers(window->window());
            
            glfwPollEvents();
        }
    
        // Clean up
        glDeleteTextures(1, &texture);
    }
    
} // namespace core
