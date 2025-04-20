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
#include <random>

#include "json/json.hpp"
#include "deps/tarscuda/tensor_operations.hpp"

namespace NTARS
{
    DenseNeuralNetwork::DenseNeuralNetwork(const std::vector<size_t>& structure, const std::string& name)
     : _structure(structure), name(name)
    {
        initializeWeightsAndBiases(structure);
        createLayers(structure);
        initializeTrainingBuffers();
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
                    auto mat = biasMatrix.get<std::vector<float>>();
                    biases.emplace_back(TMATH::Matrix_t<float>(mat, mat.size(), 1));
                }

                weights.clear();
                for (const auto& weightJson : loaded["weights"])
                {
                    auto data = weightJson["data"].get<std::vector<float>>(); 
                    size_t rows = weightJson["rows"].get<size_t>();         
                    size_t cols = weightJson["cols"].get<size_t>();    
                    weights.emplace_back(TMATH::Matrix_t<float>(data, rows, cols));
                }

                _structure = loaded["structure"].get<std::vector<size_t>>();
                createLayers(_structure);
            
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

        initializeTrainingBuffers();
    }

    DenseNeuralNetwork::~DenseNeuralNetwork()
    {
    } 

    void DenseNeuralNetwork::initializeTrainingBuffers()
    {
        deltas.clear();
        weightGradients.clear();
        biasGradients.clear();
        for (size_t l = 0; l < _layers.size(); ++l) {
            deltas.emplace_back(TMATH::Matrix_t<float>(weights[l].rows(), weights[l].cols()));
            weightGradients.emplace_back(TMATH::Matrix_t<float>(weights[l].rows(), weights[l].cols()));
            biasGradients.emplace_back(TMATH::Matrix_t<float>(biases[l].rows(), 1));
        }
    }

    void DenseNeuralNetwork::initializeWeightsAndBiases(const std::vector<size_t>& structure)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0, 1.0); 
    
        weights.clear(); 
        biases.clear();
    
        for (size_t i = 0; i < structure.size() - 1; ++i)
        {
            size_t numInputs = structure[i]; 
            size_t numOutputs = structure[i + 1]; 
    
            float scale = std::sqrt(2.0 / (numInputs + numOutputs));
    
            TMATH::Matrix_t<float> weightMatrix(numOutputs, numInputs);
            for (size_t j = 0; j < numOutputs; ++j)
            {
                for (size_t k = 0; k < numInputs; ++k)
                {
                    weightMatrix.at(j, k) = dist(gen) * scale; 
                }
            }
    
            TMATH::Matrix_t<float> biasMatrix(numOutputs, 1);
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

    uint32_t DenseNeuralNetwork::run(const std::vector<float>& inputs)
    {
        std::vector<float> currentInputs = inputs;
        for (size_t l = 0; l < _layers.size(); ++l)
        {
            currentInputs = _layers[l].forward(currentInputs, weights[l], biases[l]);
        }

        return getMostActive(currentInputs);
    }

    std::vector<float> DenseNeuralNetwork::runInternal(const std::vector<float>& inputs)
    {
        std::vector<float> currentInputs = inputs;
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
            nlohmann::json weightJson;
            weightJson["data"] = weightMatrix.getElementsRaw();
            weightJson["rows"] = weightMatrix.rows();      
            weightJson["cols"] = weightMatrix.cols();          
            saved["weights"].push_back(weightJson);
        }

        for (auto& biasMatrix : biases)
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

    float DenseNeuralNetwork::train(std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& miniBatch, float learningRate)
    {
        int numCorrect{0};
        int numWrong{0};

        for (size_t l = 0; l < _layers.size(); ++l)
        {
            weightGradients[l] = TMATH::Matrix_t<float>(weights[l].rows(), weights[l].cols());
            biasGradients[l] = TMATH::Matrix_t<float>(biases[l].rows(), 1);
        }
        
        for (auto& train_data : miniBatch)
        {
            std::vector<float> outputs = runInternal(train_data.data);
            std::vector<float> expected = train_data.label;

            const int expectedLabel = std::distance(expected.begin(), std::find(expected.begin(), expected.end(), 1));

            TMATH::Matrix_t<float> outputDelta(outputs.size(), 1);
            for (size_t i = 0; i < outputs.size(); ++i)
            {
                outputDelta.at(i, 0) = expected[i] - outputs[i];
            }
            deltas.back() = outputDelta;

            for (size_t l = _layers.size() - 1; l < _layers.size(); --l) 
            {
                if (l != 0)
                {
                    TMATH::Matrix_t<float> errorTerm = deltas[l].transpose() * weights[l];
                    TMATH::Matrix_t<float> derivatives = TMATH::sigmoid_derivative_matrix(_layers[l - 1].getActivations());
                    deltas[l - 1] = derivatives.elementWiseMultiplication(errorTerm.transpose());
                }

                TMATH::Matrix_t<float> prevActivations = (l == 0) 
                    ? TMATH::Matrix_t<float>(train_data.data, train_data.data.size(), 1) 
                    : TMATH::Matrix_t<float>(_layers[l - 1].getActivations(), _layers[l - 1].getActivations().size(), 1);      
                
                TMATH::Matrix_t<float> gradient = deltas[l] * prevActivations.transpose();
            
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

        float batchSize = static_cast<float>(miniBatch.size());
        for (size_t l = 0; l < _layers.size(); ++l)
        {
            weights[l] += weightGradients[l] * (learningRate / batchSize);
            biases[l] += biasGradients[l] * (learningRate / batchSize);
        }

        return (float)numCorrect / (float)(numCorrect + numWrong);
    }

} // namespace NTARS
