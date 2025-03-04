#include "DenseNetwork.hpp"

//#define DEBUG_MODE

constexpr bool debug = 
#ifdef DEBUG_MODE
    true;
#else
    false;
#endif

#include <iostream>
#include <filesystem>
#include <fstream>
#include "json/json.hpp"

namespace NTARS
{
    DenseNeuralNetwork::DenseNeuralNetwork(const std::vector<size_t>& structure, const std::string& name)
     : _structure(structure), name(name)
    {
        initializeWeightsAndBiases(structure);
        createLayers(structure);
    }

    DenseNeuralNetwork::DenseNeuralNetwork(const std::string& file)
    {
        nlohmann::json loaded;

        std::filesystem::path outputPath = std::filesystem::current_path() / "networks" / file;        
        std::ifstream inFile(outputPath);

        try 
        {
            if (inFile.is_open())
            {
                inFile >> loaded;
            
                if (!loaded.contains("weights") || 
                    !loaded.contains("biases") || 
                    !loaded.contains("name") || 
                    !loaded.contains("structure"))
                {
                    throw std::runtime_error("JSON file '" + file + "' doesn't contain required keys: 'weights', 'biases', 'name', or 'structure'");
                }

                name.clear();
                name = loaded["name"].get<std::string>();
            
                biases.clear();
                for (const auto& biasMatrix : loaded["biases"])
                {
                    biases.emplace_back(TMATH::Matrix_t(biasMatrix.get<std::vector<std::vector<double>>>()));
                }

                weights.clear();
                for (const auto& weightMatrix : loaded["weights"])
                {
                    weights.emplace_back(TMATH::Matrix_t(weightMatrix.get<std::vector<std::vector<double>>>()));
                }

                createLayers(loaded["structure"].get<std::vector<size_t>>());
            
                std::cout << "Network loaded successfully: " << outputPath.string() << std::endl;
            }
            else
            {
                std::cerr << "Could not open file for loading: " << outputPath.string() << std::endl;
            }            
        }
        catch(std::exception e)
        {
            std::cerr << e.what() << std::endl;
        }
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

    uint32_t DenseNeuralNetwork::run(const std::vector<double>& inputs)
    {
        std::vector<double> currentInputs = inputs;
        for (size_t l = 0; l < _layers.size(); ++l)
        {
            currentInputs = _layers[l].forward(currentInputs, weights[l], biases[l]);
        }

        return getMostActive(currentInputs);
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

    void DenseNeuralNetwork::save()
    {
        nlohmann::json saved;
        
        saved["structure"] = _structure;
        saved["name"] = name;
        
        for (const auto& weightMatrix : weights)
        {
            saved["weights"].push_back(weightMatrix.getElementsRaw());
        }

        for (const auto& biasMatrix : biases)
        {
            saved["biases"].push_back(biasMatrix.getElementsRaw());
        }
        
        std::filesystem::path outputPath = std::filesystem::current_path() / "networks";        
        
        if (!std::filesystem::exists(outputPath))
        {
            std::filesystem::create_directory(outputPath);
        }

        std::ofstream outFile(outputPath / std::string(name + ".json"));

        if (outFile.is_open())
        {
            outFile << saved.dump(); 
            outFile.close();
            std::cout << "Network saved successfully: " << outputPath << std::endl;
        }
        else
        {
            std::cerr << "Could not open file for writing: " << outputPath << std::endl;
        }
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

            std::vector<TMATH::Matrix_t<double>> deltas(_layers.size(), TMATH::Matrix_t<double>(0, 0));
            TMATH::Matrix_t<double> outputDelta(outputs.size(), 1);
            for (size_t i = 0; i < outputs.size(); ++i)
            {
                outputDelta.at(i, 0) = expected[i] - outputs[i];
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

                if (debug && l == 0)
                {
                    std::cout << "Gradient(0,0): " << gradient.at(0, 0) 
                              << " Delta(0): " << deltas[l].at(0, 0) 
                              << " PrevActiv(0): " << prevActivations.at(0, 0) << std::endl;
                }
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
