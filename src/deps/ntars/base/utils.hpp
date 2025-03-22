#ifndef NTARS_UTILS_HPP
#define NTARS_UTILS_HPP

#include <limits>
#include <numeric>
#include <cmath>
#include <cstdint>

namespace NTARS
{
    float meanSquaredError(const float y[], const float y_predicted[], uint32_t size);
    float meanAbsoluteError(const float y[], const float y_predicted[], uint32_t size);
    float crossEntropyLoss(const float y[], float y_predicted[], uint32_t size);
} // namespace NTARS


#endif // NTARS_UTILS_HPP