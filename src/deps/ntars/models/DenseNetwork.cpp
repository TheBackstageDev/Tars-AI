#include "DenseNetwork.hpp"

namespace NTARS
{
    DenseNeuralNetwork::DenseNeuralNetwork(const std::vector<size_t>& structure) : _structure(structure)
    {
        initializeWeightsAndBiases(structure);
        createLayers(structure);
    }

    void DenseNeuralNetwork::initializeWeightsAndBiases(const std::vector<size_t>& structure)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());

        for (size_t i = 0; i < structure.size() - 1; ++i)
        {
            size_t numInputs = structure[i];
            size_t numOutputs = structure[i + 1];

            double stddev = std::sqrt(2.0 / (numInputs + numOutputs));
            std::normal_distribution<> dis(0.0, stddev);

            TMATH::Matrix_t<double> weightMatrix(numOutputs, numInputs);
            TMATH::Matrix_t<double> biasMatrix(numOutputs, 1);

            for (size_t j = 0; j < numOutputs; ++j)
            {
                for (size_t k = 0; k < numInputs; ++k)
                {
                    weightMatrix.at(j, k) = dis(gen);
                }
                biasMatrix.at(j, 0) = dis(gen);
            }

            weights.emplace_back(weightMatrix);
            biases.emplace_back(biasMatrix);
        }
    }

    void DenseNeuralNetwork::createLayers(const std::vector<size_t>& structure)
    {
        for (size_t i = 1; i < structure.size(); ++i)
        {
            _layers.emplace_back(structure[i], structure[i - 1]);
        }
    };

    std::vector<double> DenseNeuralNetwork::run(const std::vector<double>& inputs)
    {
        std::vector<double> currentInputs = inputs;
        for (size_t l = 0; l < _layers.size(); ++l)
        {
            auto& layer = _layers[l];
            currentInputs = layer.forward(currentInputs, weights[l], biases[l]);
        }
        for (auto& input : currentInputs)
        {
            input = TMATH::sigmoid(input);
        }
        return currentInputs;
    }

    double DenseNeuralNetwork::train(std::vector<NTARS::DATA::TrainingData<std::vector<double>>>& miniBatch, double learningRate = 0.1)
    {
        int numCorrect{0};
        int numWrong{0};

        for (auto& image : miniBatch)
        {
            std::vector<double> outputs = run(image.data);
            std::vector<double> expected(outputs.size(), 0.0);
            expected.at(std::stoi(image.label)) = 1.0;

            double networkCost = cost(outputs, expected);

            for (size_t l = _layers.size(); l >= 0; --l)
            {
                auto& layer = _layers[l];
                
                
            }
        }

        return (double)numCorrect / (double)(numCorrect + numWrong);
    }

} // namespace NTARS
