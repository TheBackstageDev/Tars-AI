#ifndef NTARS_NEURON_HPP
#define NTARS_NEURON_HPP

#include <vector>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <random>
#include "tarsmath/calculus/sigmoid.hpp"

namespace NTARS
{
    class Neuron
    {
    public:
        Neuron() = default;

        const float activate(const std::vector<float>& inputs, const std::vector<float>& weights, const float bias)
        {
            float sum{0.0};

            for (size_t i = 0; i < weights.size(); ++i)
            {
                sum += inputs[i] * weights[i];
            }
            sum += bias;

            this->activation = TMATH::sigmoid(sum);

            return activation;
        }

        inline float getActivation() const  { return activation; }

    private:
        float activation{0};
    };
} // namespace NTARS

#endif // NTARS_NEURON_HPP
