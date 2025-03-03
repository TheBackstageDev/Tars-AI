#ifndef NTARS_UTILS_HPP
#define NTARS_UTILS_HPP

#include <limits>
#include <numeric>
#include <cmath>
#include <cstdint>

namespace NTARS
{
    double meanSquaredError(const double y[], const double y_predicted[], uint32_t size);
    double meanAbsoluteError(const double y[], const double y_predicted[], uint32_t size);
    double crossEntropyLoss(const double y[], double y_predicted[], uint32_t size);
} // namespace NTARS


#endif // NTARS_UTILS_HPP