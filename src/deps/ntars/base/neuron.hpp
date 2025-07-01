#ifndef NTARS_NEURON_HPP
#define NTARS_NEURON_HPP

#include <vector>
#include "tarsmath/calculus/sigmoid.hpp"
#include "tarsmath/calculus/relu.hpp"
#include <numeric>

namespace NTARS
{
    enum NeuronFlags_
    {
        NeuronFlags_None = 1ULL << 0, // Sigmoid function on default;
        NeuronFlags_ReLU = 1ULL << 1,
    };

    enum NeuralNetworkFlags_
    {
        NeuralNetworkFlags_None = 1ULL << 0,
        NeuralNetworkFlags_ReLU_Internal = 1ULL << 1, // ReLU is completely broken as of now;
        NeuralNetworkFlags_ReLU = 1ULL << 2, // on all layers, including output;
    };

    class Neuron
    {
    public:
        Neuron() = default;

        const float activate(const std::vector<float>& inputs, const std::vector<float>& weights, const float bias, NeuronFlags_ flag = NeuronFlags_None)
        {
            assert(inputs.size() == weights.size());

            float sum = std::inner_product(inputs.begin(), inputs.end(), weights.begin(), bias, std::plus<>(), std::multiplies<>());
            this->activation = flag == NeuronFlags_ReLU ? TMATH::relu(sum) : TMATH::sigmoid(sum);

            return activation;
        }

        inline float getActivation() const  { return activation; }

    private:
        float activation{0};
    };
} // namespace NTARS

#endif // NTARS_NEURON_HPP
