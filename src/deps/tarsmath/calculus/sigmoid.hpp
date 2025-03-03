#ifndef TARS_MATH_SIGMOID_HPP
#define TARS_MATH_SIGMOID_HPP

#include "tarsmath/linear_algebra/matrix_component.hpp"
#include <cmath>
#include <vector>

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

    inline std::vector<double> sigmoid_derivative(std::vector<double> x)
    {
        std::vector<double> derivatives(x.size(), 0.0);
        
        for (size_t i = 0; i < x.size(); ++i)
        {
            derivatives[i] = sigmoid_derivative(x[i]);
        }

        return derivatives;
    }

    inline TMATH::Matrix_t<double> sigmoid_derivative_matrix(std::vector<double> x)
    {
        TMATH::Matrix_t<double> derivatives(x.size(), 1);

        for (size_t i = 0; i < x.size(); ++i)
        {
            derivatives.at(i, 0) = sigmoid_derivative(x[i]);
        }

        return derivatives;
    }

    inline TMATH::Matrix_t<double> sigmoid_derivative_matrix(TMATH::Matrix_t<double> x)
    {
        TMATH::Matrix_t<double> derivatives(x.rows(), x.cols());

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