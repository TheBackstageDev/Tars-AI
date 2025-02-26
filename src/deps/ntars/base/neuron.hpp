#ifndef NTARS_NEURON_HPP
#define NTARS_NEURON_HPP

#include <vector>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include "tarsmath/calculus/sigmoid.hpp"

namespace NTARS
{
    class Neuron
    {
    public:
        Neuron(size_t numInputs) : weights(numInputs)
        {
            std::srand(static_cast<unsigned int>(std::time(nullptr)));

            for (auto& weight : weights)
            {
                weight = static_cast<double>(std::rand()) / RAND_MAX;
            }

            bias = static_cast<double>(std::rand()) / RAND_MAX;

            static uint32_t last_id;
            this->id = ++last_id;
        }

        inline uint32_t getId() const { return id; }

        double activate(const std::vector<double>& inputs) const
        {
            double sum{0.0};

            for (size_t i = 0; i < weights.size(); ++i)
            {
                sum += inputs[i] * weights[i];
            }
            sum += bias;

            return TMATH::sigmoid(sum);
        }

        inline void updateWeight(size_t index, double newWeight)
        {
            assert(index < weights.size() && "Out of Bounds");

            weights[index] = newWeight;
        }

        inline void updateBias(double newBias) { bias = newBias; }

        _NODISCARD inline std::vector<double> getWeights() const { return weights; }
        _NODISCARD inline double getBias() const { return bias; }

    private:
        std::vector<double> weights;
        uint32_t id;
        double bias;
    };
} // namespace NTARS

#endif // NTARS_NEURON_HPP
