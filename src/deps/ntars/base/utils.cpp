#include "utils.hpp"

namespace NTARS
{
    float meanSquaredError(const float y[], const float y_predicted[], uint32_t size)
    {
        float sum = std::inner_product(y, y + size, y_predicted, 0.0, std::plus<>(), [](float a, float b) {
            return std::pow(a - b, 2);
        });
        return sum / size;
    }

    float meanAbsoluteError(const float y[], const float y_predicted[], uint32_t size)
    {
        float sum = std::inner_product(y, y + size, y_predicted, 0.0, std::plus<>(), [](float a, float b) {
            return std::abs(a - b);
        });
        return sum / size;
    }

    float crossEntropyLoss(const float y[], float y_predicted[], uint32_t size)
    {
        float sum = std::inner_product(y, y + size, y_predicted, 0.0, std::plus<>(), [](float a, float b) {
            a += std::numeric_limits<float>().epsilon();
            return -(a * std::log(b) + (1 - a) * std::log(1 - b)); 
        });
        return sum / size;
    }
} // namespace NTARS
