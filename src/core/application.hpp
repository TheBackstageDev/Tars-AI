#ifndef CORE_APPLICATION_HPP
#define CORE_APPLICATION_HPP

#include "window.hpp"
#include <memory>
#include <thread>

#include "mnist/mnist_reader.hpp"
#include "ntars/models/DenseNetwork.hpp"
#include "checkers/checkers.hpp"
#include "checkers/checkersminmax.hpp"

#include "audio.hpp"

namespace core
{
    enum class CurrentPart
    {
        MENU,
        CHECKERS,
        PRESENTATION,
        AIPRESENTATION
    };

    class application
    {
    public:
        application(const std::string& title, uint32_t width, uint32_t height);
        ~application();

        void run();
    private:
        void imguiSetup();
        void imguiNewFrame();
        void imguiEndFrame();

        void runCheckers(Checkers& checkers, Board& board, NETWORK::CheckersMinMax& algorithm, NTARS::DenseNeuralNetwork& network, std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& trainingData);
        void runPresentation();
        void runAITraining();
        void runMenu();

        void drawNetwork();

        CurrentPart part{CurrentPart::MENU};

        //AI PRESENTATION PART//

        const size_t batch_size = 200;
        float learningRate = 1.0;

        NTARS::DenseNeuralNetwork numberNetwork{{784, 100, 50, 10}, "ExampleNet_V1"};
        mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset{};
        std::vector<std::vector<NTARS::DATA::TrainingData<std::vector<float>>>> batches{};

        std::thread* networkThread = nullptr;

        SoundHandle soundHandle;
        std::unique_ptr<window_t> window;
    };
    
} // namespace core

#endif // CORE_APPLICATION_HPP