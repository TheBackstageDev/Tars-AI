#ifndef TARS_MATH_SIGMOID_HPP
#define TARS_MATH_SIGMOID_HPP

#include <cmath>

namespace TMATH
{
    inline double sigmoid(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline double sigmoid_derivative(double x)
    {
        return sigmoid(x) * (1.0 - sigmoid(x));
    }
} // namespace TMATH

#endif // TARS_MATH_SIGMOID_HPP