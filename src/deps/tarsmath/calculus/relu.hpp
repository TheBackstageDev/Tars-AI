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

    inline TMATH::Matrix_t<float> relu_derivative_matrix(const std::vector<float>& x)
    {
        TMATH::Matrix_t<float> derivatives(x.size(), 1);
        for (size_t i = 0; i < x.size(); ++i)
            derivatives[i] = x[i] > 0.0f ? 1.0f : 0.0f;

        return derivatives;
    }

    inline TMATH::Matrix_t<float> relu_derivative_matrix(const TMATH::Matrix_t<float>& x)
    {
        TMATH::Matrix_t<float> derivatives(x.rows(), x.cols());
        size_t N = x.rows() * x.cols();

        const std::vector<float>& in = x.getElementsRaw();
        std::vector<float>& out = derivatives.getElementsRaw();

        for (size_t i = 0; i < N; ++i)
            out[i] = in[i] > 0.0f ? 1.0f : 0.0f;

        return derivatives;
    }
}

#endif //TARS_MATH_RELU_HPP