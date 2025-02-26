#ifndef NTARS_DENSE_LAYER_HPP
#define NTARS_DENSE_LAYER_HPP

#include <vector>
#include "ntars/base/neuron.hpp"

namespace NTARS
{
    class DenseLayer
    {
    public:
        DenseLayer(size_t numNeurons, size_t numInputs)
            : numNeurons(numNeurons), numInputs(numInputs)
        {
            for (size_t i = 0; i < numNeurons; ++i)
            {
                neurons.push_back(Neuron(numInputs));
            }
        }

        std::vector<double> forward(const std::vector<double>& inputs)
        {
            std::vector<double> results(numNeurons);
            for (size_t i = 0; i < numNeurons; ++i)
            {
                results[i] = neurons[i].activate(inputs);
            }
            return results;
        }

        void updateWeights(const std::vector<std::vector<double>>& newWeights)
        {
            for (size_t i = 0; i < numNeurons; ++i)
            {
                for (size_t j = 0; j < numInputs; ++j)
                {
                    neurons[i].updateWeight(j, newWeights[i][j]);
                }
            }
        }

        void updateBiases(const std::vector<double>& newBiases)
        {
            for (size_t i = 0; i < numNeurons; ++i)
            {
                neurons[i].updateBias(newBiases[i]);
            }
        }
        
        inline Neuron& getNeuron(size_t index) { return neurons.at(index); }
        inline std::vector<Neuron>& getNeurons() { return neurons; }

    private:
        size_t numNeurons;
        size_t numInputs;
        std::vector<Neuron> neurons;
    };
    
} // namespace NTARS

#endif // NTARS_DENSE_LAYER_HPP
