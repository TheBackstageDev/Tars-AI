#ifndef TARS_MATH_SIGMOID_HPP
#define TARS_MATH_SIGMOID_HPP

#include "tarsmath/linear_algebra/matrix_component.hpp"
#include <cmath>
#include <vector>

namespace TMATH
{
    inline float sigmoid(float x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline float sigmoid_derivative(float x)
    {
        float result = sigmoid(x);
        return result * (1.0 - result);
    }

    inline std::vector<float> sigmoid_derivative(std::vector<float> x)
    {
        std::vector<float> derivatives(x.size(), 0.0);
        
        for (size_t i = 0; i < x.size(); ++i)
        {
            derivatives[i] = sigmoid_derivative(x[i]);
        }

        return derivatives;
    }

    inline TMATH::Matrix_t<float> sigmoid_derivative_matrix(std::vector<float> x)
    {
        TMATH::Matrix_t<float> derivatives(x.size(), 1);

        for (size_t i = 0; i < x.size(); ++i)
        {
            derivatives.at(i, 0) = sigmoid_derivative(x[i]);
        }

        return derivatives;
    }

    inline TMATH::Matrix_t<float> sigmoid_derivative_matrix(TMATH::Matrix_t<float> x)
    {
        TMATH::Matrix_t<float> derivatives(x.rows(), x.cols());

        for (size_t i = 0; i < x.rows(); ++i)
        {
            for (size_t j = 0; j < x.cols(); ++j)
            {
                derivatives.at(i, j) = sigmoid_derivative(x.at(i, j));
            }
        }

        return derivatives;
    }
} // namespace TMATH

#endif // TARS_MATH_SIGMOID_HPP