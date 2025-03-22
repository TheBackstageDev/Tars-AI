#ifndef TARSCUDA_TENSOR_OPERATIONS_HPP
#define TARSCUDA_TENSOR_OPERATIONS_HPP

#define TARS_SUCCESS true
#define TARS_FAILURE false

#include <vector>
#include "deps/tarsmath/linear_algebra/matrix_component.hpp"

namespace TCUDA
{
    bool matrixMultiply(float* x, float* y, float* result, int rowsX, int colsX, int colsY, bool transposeX = false, bool transposeY = false);
    bool matrixMultiply(std::vector<std::vector<float>>& matrixA, std::vector<std::vector<float>>& matrixB, std::vector<std::vector<float>>* resultMatrix, bool transposeX = false, bool transposeY = false);
    bool matrixElementWiseMultiply(float* x, float* y, float* result, size_t size);
} // namespace TCUDA

#endif // TARSCUDA_TENSOR_OPERATIONS_HPP