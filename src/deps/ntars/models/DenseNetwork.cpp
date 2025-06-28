#include "DenseNetwork.hpp"

// #define DEBUG_MODE

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

#include <future>
#include <mutex>
#include <thread>

#include "json/json.hpp"

namespace NTARS
{
    DenseNeuralNetwork::DenseNeuralNetwork(const std::vector<size_t> &structure, const std::string &name, NeuralNetworkFlags_ flags)
        : _structure(structure), name(name), flags(flags)
    {
        initializeWeightsAndBiases(structure);
        createLayers(structure);
        initializeTrainingBuffers();
    }

    DenseNeuralNetwork::DenseNeuralNetwork(const std::string &file)
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
                flags = loaded["flags"].get<NeuralNetworkFlags_>();

                biases.clear();
                for (const auto &biasMatrix : loaded["biases"])
                {
                    auto mat = biasMatrix.get<std::vector<float>>();
                    biases.emplace_back(TMATH::Matrix_t<float>(mat, mat.size(), 1));
                }

                weights.clear();
                for (const auto &weightJson : loaded["weights"])
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
        catch (std::exception e)
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
        weightGradients.clear();
        biasGradients.clear();
        for (size_t l = 0; l < _layers.size(); ++l)
        {
            weightGradients.emplace_back(TMATH::Matrix_t<float>(weights[l].rows(), weights[l].cols()));
            biasGradients.emplace_back(TMATH::Matrix_t<float>(biases[l].rows(), 1));
        }
    }

    void DenseNeuralNetwork::initializeWeightsAndBiases(const std::vector<size_t> &structure)
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

    void DenseNeuralNetwork::createLayers(const std::vector<size_t> &structure)
    {
        for (size_t i = 1; i < structure.size(); ++i)
        {
            _layers.emplace_back(structure[i], structure[i - 1]);
        }
    };

    std::vector<float> DenseNeuralNetwork::run(const std::vector<float> &inputs, bool slowRun)
    {
        std::vector<float> currentInputs = inputs;
        for (size_t l = 0; l < _layers.size(); ++l)
        {
            currentInputs = _layers[l].forward(currentInputs, weights[l], biases[l]);
            if (slowRun)
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        return currentInputs;
    }

    void DenseNeuralNetwork::save()
    {
        nlohmann::json saved;

        saved["structure"] = _structure;
        saved["name"] = name;
        saved["flags"] = flags;

        for (const auto &weightMatrix : weights)
        {
            nlohmann::json weightJson;
            weightJson["data"] = weightMatrix.getElementsRaw();
            weightJson["rows"] = weightMatrix.rows();
            weightJson["cols"] = weightMatrix.cols();
            saved["weights"].push_back(weightJson);
        }

        for (auto &biasMatrix : biases)
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

    void DenseNeuralNetwork::train(std::vector<NTARS::DATA::TrainingData<std::vector<float>>> &miniBatch, float learningRate)
    {
    }

    void DenseNeuralNetwork::calcGradient(
        const NTARS::DATA::TrainingData<std::vector<float>> &data,
        int32_t &numCorrect,
        int32_t &numWrong)
    {
        std::vector<float> outputs = run(data.data);
        const std::vector<float> &expected = data.label;

        int expectedLabel = std::distance(expected.begin(), std::find(expected.begin(), expected.end(), 1));

        std::vector<TMATH::Matrix_t<float>> deltas;
        deltas.reserve(_layers.size());

        for (const auto &layer : _layers)
        {
            deltas.emplace_back(TMATH::Matrix_t<float>(layer.getNumOutputs(), 1));
        }

        TMATH::Matrix_t<float> outputDelta(outputs.size(), 1);
        for (size_t i = 0; i < outputs.size(); ++i)
            outputDelta.at(i, 0) = expected[i] - outputs[i];

        deltas.back() = outputDelta;

        for (int64_t l = _layers.size() - 1; l >= 0; --l)
        {
            if (l != 0)
            {
                auto errorTerm = deltas[l].transpose() * weights[l];
                auto deriv = TMATH::sigmoid_derivative_matrix(_layers[l - 1].getActivations());
                deltas[l - 1] = deriv.elementWiseMultiplication(errorTerm.transpose());
            }

            auto prevActivations = (l == 0)
                                       ? TMATH::Matrix_t<float>(data.data, data.data.size(), 1)
                                       : TMATH::Matrix_t<float>(_layers[l - 1].getActivations(), _layers[l - 1].getActivations().size(), 1);

            {
                std::lock_guard<std::mutex> lock(gradMutex);
                weightGradients[l] += deltas[l] * prevActivations.transpose();
                biasGradients[l] += deltas[l];
            }                             
        }

        (getMostActive(outputs) == expectedLabel) ? ++numCorrect : ++numWrong;
    }

    float DenseNeuralNetwork::trainCPU(std::vector<NTARS::DATA::TrainingData<std::vector<float>>> &miniBatch, float learningRate, bool slowTrain)
    {
        int32_t numCorrect = 0;
        int32_t numWrong = 0;

        for (size_t l = 0; l < _layers.size(); ++l)
        {
            weightGradients[l] = TMATH::Matrix_t<float>(weights[l].rows(), weights[l].cols());
            biasGradients[l] = TMATH::Matrix_t<float>(biases[l].rows(), 1);
        }

        const size_t numThreads = std::thread::hardware_concurrency();
        const size_t chunkSize = miniBatch.size() / numThreads;

        std::vector<std::future<std::pair<int32_t, int32_t>>> futures;

        for (size_t t = 0; t < numThreads; ++t)
        {
            futures.emplace_back(std::async(std::launch::async, [&, t]()
                                            {
                size_t start = t * chunkSize;
                size_t end = (t == numThreads - 1) ? miniBatch.size() : (t + 1) * chunkSize;

                std::vector<TMATH::Matrix_t<float>> localWGrads, localBGrads;
                for (size_t l = 0; l < _layers.size(); ++l) {
                    localWGrads.emplace_back(weights[l].rows(), weights[l].cols());
                    localBGrads.emplace_back(biases[l].rows(), 1);
                }

                int32_t localCorrect = 0, localWrong = 0;

                for (size_t i = start; i < end; ++i)
                    calcGradient(miniBatch[i], localCorrect, localWrong);

                return std::make_pair(localCorrect, localWrong); }));
        }

        for (auto& fut : futures) {
            auto [correct, wrong] = fut.get();
            numCorrect += correct;
            numWrong += wrong;
        }

        float batchSize = static_cast<float>(miniBatch.size());
        for (int64_t l = _layers.size() - 1; l >= 0; --l)
        {
            weights[l] += weightGradients[l] * (learningRate / batchSize);
            biases[l] += biasGradients[l] * (learningRate / batchSize);

            if (slowTrain)
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }

        return static_cast<float>(numCorrect) / (numCorrect + numWrong);
    }

} // namespace NTARS
