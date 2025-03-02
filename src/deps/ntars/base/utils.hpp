#ifndef NTARS_UTILS_HPP
#define NTARS_UTILS_HPP

#include <limits>
#include <numeric>
#include <cmath>
#include <cstdint>

namespace NTARS
{
    #include <numeric>

    double meanSquaredError(const double y[], const double y_predicted[], uint32_t size)
    {
        double sum = std::inner_product(y, y + size, y_predicted, 0.0, std::plus<>(), [](double a, double b) {
            return std::pow(a - b, 2);
        });
        return sum / size;
    }

    double meanAbsoluteError(const double y[], const double y_predicted[], uint32_t size)
    {
        double sum = std::inner_product(y, y + size, y_predicted, 0.0, std::plus<>(), [](double a, double b) {
            return std::abs(a - b);
        });
        return sum / size;
    }

    double crossEntropyLoss(const double y[], double y_predicted[], uint32_t size)
    {
        double sum = std::inner_product(y, y + size, y_predicted, 0.0, std::plus<>(), [](double a, double b) {
            a += std::numeric_limits<double>().epsilon();
            return -(a * std::log(b) + (1 - a) * std::log(1 - b)); 
        });
        return sum / size;
    }
} // namespace NTARS


#endif // NTARS_UTILS_HPP