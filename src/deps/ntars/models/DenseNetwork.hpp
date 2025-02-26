#ifndef NTARS_DENSE_NETWORK_HPP
#define NTARS_DENSE_NETWORK_HPP

#include "ntars/layers/dense_layer.hpp"
#include <vector>
#include <array>

namespace NTARS
{
    // Neural Network which uses dense layers
    class DenseNeuralNetwork 
    {
    public:
        DenseNeuralNetwork(const std::vector<size_t>& structure) : _structure(structure)
        {
            for (size_t i = 1; i < structure.size(); ++i)
            {
                layers.emplace_back(structure[i], structure[i - 1]);
            }
        }

        std::vector<double> run(const std::vector<double>& inputs)
        {
            std::vector<double> currentInputs = inputs;
            for (auto& layer : layers)
            {
                currentInputs = layer.forward(currentInputs);
            }
            return currentInputs;
        }

        inline std::vector<size_t> getStructure() const { return _structure; }
    private:
        std::vector<DenseLayer> layers;
        std::vector<size_t> _structure;
    };
    
} // namespace NTARS

#endif // NTARS_DENSE_NETWORK_HPP