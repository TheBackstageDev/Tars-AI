#ifndef NTARS_DENSE_NETWORK_HPP
#define NTARS_DENSE_NETWORK_HPP

#include "tarsmath/calculus/sigmoid.hpp"
#include "tarsmath/linear_algebra/matrix_component.hpp"
#include "ntars/layers/dense_layer.hpp"
#include "ntars/base/data.hpp"
#include "ntars/base/utils.hpp"
#include <numeric>

namespace NTARS
{
    // Neural Network which uses dense layers
    class DenseNeuralNetwork 
    {
    public:
        DenseNeuralNetwork(const std::vector<size_t>& structure, const std::string& name);
        DenseNeuralNetwork(const std::string& file);

        uint32_t run(const std::vector<float>& inputs);

        float train(std::vector<NTARS::DATA::TrainingData<std::vector<float>>>& miniBatch, float learningRate = 1);

        void save();

        inline std::vector<size_t> getStructure() const { return _structure; }
        inline std::vector<DenseLayer>& getLayers() { return _layers; }
    private:
        void initializeWeightsAndBiases(const std::vector<size_t>& structure);
        void createLayers(const std::vector<size_t>& structure);

        constexpr uint32_t getMostActive(const std::vector<float>& outputs) const
        {
            return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
        }

        float cost(const std::vector<float>& results, const std::vector<float>& expected) const
        {
            return meanSquaredError(results.data(), expected.data(), results.size());
        }

        std::vector<float> runInternal(const std::vector<float>& inputs);

        std::vector<TMATH::Matrix_t<float>> weights;
        std::vector<TMATH::Matrix_t<float>> biases;

        std::string name;

        std::vector<DenseLayer> _layers;
        std::vector<size_t> _structure;
    };
    
} // namespace NTARS

#endif // NTARS_DENSE_NETWORK_HPP