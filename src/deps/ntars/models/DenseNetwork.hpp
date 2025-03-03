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
        DenseNeuralNetwork(const std::vector<size_t>& structure);

        std::vector<double> run(const std::vector<double>& inputs);

        double train(std::vector<NTARS::DATA::TrainingData<std::vector<double>>>& miniBatch, double learningRate = 1);

        inline std::vector<size_t> getStructure() const { return _structure; }
        inline std::vector<DenseLayer>& getLayers() { return _layers; }
    private:
        void initializeWeightsAndBiases(const std::vector<size_t>& structure);
        void createLayers(const std::vector<size_t>& structure);

        constexpr uint32_t getMostActive(const std::vector<double>& outputs) const
        {
            return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
        }

        double cost(const std::vector<double>& results, const std::vector<double>& expected) const
        {
            return meanSquaredError(results.data(), expected.data(), results.size());
        }

        std::vector<double> runInternal(const std::vector<double>& inputs);

        std::vector<TMATH::Matrix_t<double>> weights;
        std::vector<TMATH::Matrix_t<double>> biases;

        std::vector<DenseLayer> _layers;
        std::vector<size_t> _structure;
    };
    
} // namespace NTARS

#endif // NTARS_DENSE_NETWORK_HPP