#ifndef TARS_MATH_RELU_HPP
#define TARS_MATH_RELU_HPP

#include "tarsmath/linear_algebra/matrix_component.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace TMATH
{
    inline float relu(float x)
    {
        return std::max(0.0f, x);
    }

    inline float relu_derivative(float x)
    {
        return x >= 0 ? 1 : 0;
    }
}

#endif //TARS_MATH_RELU_HPP