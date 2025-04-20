#ifndef NTARS_NEURON_HPP
#define NTARS_NEURON_HPP

#include <vector>
#include "tarsmath/calculus/sigmoid.hpp"
#include <numeric>

namespace NTARS
{
    class Neuron
    {
    public:
        Neuron() = default;

        const float activate(const std::vector<float>& inputs, const std::vector<float>& weights, const float bias)
        {
            assert(inputs.size() == weights.size());

            float sum = std::inner_product(inputs.begin(), inputs.end(), weights.begin(), bias, std::plus<>(), std::multiplies<>());
            this->activation = TMATH::sigmoid(sum);

            return activation;
        }

        inline float getActivation() const  { return activation; }

    private:
        float activation{0};
    };
} // namespace NTARS

#endif // NTARS_NEURON_HPP
