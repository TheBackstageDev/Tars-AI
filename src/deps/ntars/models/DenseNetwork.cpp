#include "DenseNetwork.hpp"

//#define DEBUG_MODE

constexpr bool debug = 
#ifdef DEBUG_MODE
    true;
#else
    false;
#endif

#include <iostream>

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
        std::normal_distribution<double> dist(0.0, 1.0); 
    
        weights.clear(); 
        biases.clear();
    
        for (size_t i = 0; i < structure.size() - 1; ++i)
        {
            size_t numInputs = structure[i]; 
            size_t numOutputs = structure[i + 1]; 
    
            double scale = std::sqrt(2.0 / (numInputs + numOutputs));
    
            TMATH::Matrix_t<double> weightMatrix(numOutputs, numInputs);
            for (size_t j = 0; j < numOutputs; ++j)
            {
                for (size_t k = 0; k < numInputs; ++k)
                {
                    weightMatrix.at(j, k) = dist(gen) * scale; 
                }
            }
    
            TMATH::Matrix_t<double> biasMatrix(numOutputs, 1);
            for (size_t j = 0; j < numOutputs; ++j)
            {
                biasMatrix.at(j, 0) = 0.1; 
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
            currentInputs = _layers[l].forward(currentInputs, weights[l], biases[l]);
        }
        for (size_t i = 0; i < currentInputs.size(); ++i)
        {
            currentInputs[i] = TMATH::sigmoid(currentInputs[i]);
        }

        return currentInputs;
    }

    std::vector<double> DenseNeuralNetwork::runInternal(const std::vector<double>& inputs)
    {
        std::vector<double> currentInputs = inputs;
        for (size_t l = 0; l < _layers.size(); ++l)
        {
            currentInputs = _layers[l].forward(currentInputs, weights[l], biases[l]);
        }

        return currentInputs;
    }

    double DenseNeuralNetwork::train(std::vector<NTARS::DATA::TrainingData<std::vector<double>>>& miniBatch, double learningRate)
    {
        int numCorrect{0};
        int numWrong{0};

        std::vector<TMATH::Matrix_t<double>> weightGradients(_layers.size(), TMATH::Matrix_t<double>(0, 0));
        std::vector<TMATH::Matrix_t<double>> biasGradients(_layers.size(), TMATH::Matrix_t<double>(0, 1));

        for (size_t l = 0; l < _layers.size(); ++l)
        {
            weightGradients[l] = TMATH::Matrix_t<double>(weights[l].rows(), weights[l].cols());
            biasGradients[l] = TMATH::Matrix_t<double>(biases[l].rows(), 1);
        }
        
        for (auto& image : miniBatch)
        {
            std::vector<double> outputs = runInternal(image.data);
            std::vector<double> expected(outputs.size(), 0.0);

            const int expectedLabel = static_cast<int>(image.label[0]);

            expected.at(expectedLabel) = 1.0;

            std::vector<double> outputDelta(outputs.size());
            for (size_t i = 0; i < outputDelta.size(); ++i)
            {
                outputDelta[i] = expected[i] - outputs[i];
            }

            std::vector<TMATH::Matrix_t<double>> deltas;
            for (size_t i = 0; i < _layers.size(); ++i)
            {
                deltas.emplace_back(TMATH::Matrix_t<double>(_layers[i].getNumOutputs(), 1));
            }
            
            deltas.back() = outputDelta;

            for (size_t l = _layers.size() - 1; l > 0; --l)
            {
                TMATH::Matrix_t<double> errorTerm = deltas[l].transpose() * weights[l];
                TMATH::Matrix_t<double> derivatives = TMATH::sigmoid_derivative_matrix(_layers[l - 1].getActivations());
                deltas[l - 1] = derivatives.elementWiseMultiplication(errorTerm.transpose());
            }

            for (size_t l = _layers.size() - 1; l < _layers.size(); --l) 
            {
                TMATH::Matrix_t<double> prevActivations = (l == 0) 
                    ? TMATH::Matrix_t<double>(image.data) 
                    : TMATH::Matrix_t<double>(_layers[l - 1].getActivations());
                
                TMATH::Matrix_t<double> gradient = deltas[l] * prevActivations.transpose();
            
                weightGradients[l] += gradient;
                biasGradients[l] += deltas[l];
            }

            getMostActive(outputs) == expectedLabel
            ? ++numCorrect : ++numWrong;

            if (debug)
            {
                std::cout << "Current Cost: " + std::to_string(cost(outputs, expected)) << std::endl;
            }
        }

        double batchSize = static_cast<double>(miniBatch.size());
        for (size_t l = 0; l < _layers.size(); ++l)
        {
            weights[l] += weightGradients[l] * (learningRate / batchSize);
            biases[l] += biasGradients[l] * (learningRate / batchSize);
        }

        return (double)numCorrect / (double)(numCorrect + numWrong);
    }

} // namespace NTARS
