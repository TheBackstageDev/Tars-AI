#ifndef NTARS_DENSE_NETWORK_HPP
#define NTARS_DENSE_NETWORK_HPP

#include "tarsmath/calculus/sigmoid.hpp"
#include "tarsmath/linear_algebra/matrix_component.hpp"
#include "ntars/layers/dense_layer.hpp"
#include "ntars/base/data.hpp"
#include "ntars/base/utils.hpp"
#include <numeric>
#include <imgui/imgui/imgui.h>

#include <mutex>

namespace NTARS
{
    struct ForwardResult 
    {
        std::vector<float> output;
        std::vector<std::vector<float>> activations;
    };

    // Neural Network which uses dense layers
    class DenseNeuralNetwork 
    {
    public:
        DenseNeuralNetwork(const std::vector<size_t>& structure, const std::string& name, NeuralNetworkFlags_ flags = NeuralNetworkFlags_None);
        DenseNeuralNetwork(const std::string& file);

        ~DenseNeuralNetwork();

        ForwardResult run(const std::vector<float>& inputs);
        float trainCPU(std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& miniBatch, float learningRate = 1);
        void train(std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& miniBatch, float learningRate = 1);

        void save();

        inline std::vector<size_t> getStructure() const { return _structure; }
        inline std::vector<DenseLayer>& getLayers() { return _layers; }

        inline std::vector<TMATH::Matrix_t<float>>& getWeights() { return weights; }
        inline std::vector<TMATH::Matrix_t<float>>& getBiases() { return biases; }

        void drawNetwork(bool partial);
    private:
        void initializeWeightsAndBiases(const std::vector<size_t>& structure);
        void initializeTrainingBuffers();
        void createLayers(const std::vector<size_t>& structure);

        constexpr uint32_t getMostActive(const std::vector<float>& outputs) const
        {
            return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
        }

        float cost(const std::vector<float>& results, const std::vector<float>& expected) const
        {
            return meanSquaredError(results.data(), expected.data(), results.size());
        }

        void calcGradient(const NTARS::DATA::TrainingData<std::vector<float>>& data,        
            std::vector<TMATH::Matrix_t<float>>& localWGradient,
            std::vector<TMATH::Matrix_t<float>>& localBGradient, 
            int32_t& numCorrect, int32_t& numWrong);

        std::vector<TMATH::Matrix_t<float>> weights;
        std::vector<TMATH::Matrix_t<float>> biases;

        std::string name;

        std::vector<DenseLayer> _layers;
        std::vector<size_t> _structure;

        NeuralNetworkFlags_ flags;

        // Training Buffers
        std::vector<TMATH::Matrix_t<float>> weightGradients;
        std::vector<TMATH::Matrix_t<float>> biasGradients;
    };
    
} // namespace NTARS

#endif // NTARS_DENSE_NETWORK_HPP