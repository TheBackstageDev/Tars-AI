#ifndef NTARS_DENSE_LAYER_HPP
#define NTARS_DENSE_LAYER_HPP

#include <vector>
#include "tarsmath/linear_algebra/matrix_component.hpp"
#include "ntars/base/neuron.hpp"
#include "json/json.hpp"

namespace NTARS
{
    class DenseLayer
    {
    public:
        DenseLayer(size_t numNeurons, size_t numInputs)
            : numNeurons(numNeurons), numInputs(numInputs)
        {
            _activations.resize(numNeurons);
            for (size_t i = 0; i < numNeurons; ++i)
            {
                _activations[i] = 1.0f;
                _neurons.emplace_back();
            }
        }
        
        std::vector<float>& forward(const std::vector<float>& inputs, const TMATH::Matrix_t<float>& weights, const TMATH::Matrix_t<float>& biases)
        {
            for (size_t i = 0; i < numNeurons; ++i)
            {
                _activations[i] = _neurons[i].activate(inputs, weights.rowAt(i), biases.at(i, 0));
            }
            return _activations;
        }
        
        inline size_t getNumInputs() const { return numInputs; }
        inline size_t getNumOutputs() const { return numNeurons; }
        inline Neuron& getNeuron(size_t index) { return _neurons.at(index); }
        inline std::vector<Neuron>& getNeurons() { return _neurons; }
        inline std::vector<float> getActivations() const { return _activations; }

    private:
        size_t numNeurons;
        size_t numInputs;
        std::vector<float> _activations;
        std::vector<Neuron> _neurons;
    };
    
} // namespace NTARS

#endif // NTARS_DENSE_LAYER_HPP
