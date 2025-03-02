#ifndef NTARS_DATA_HPP
#define NTARS_DATA_HPP

#include <vector>
#include <string>
#include <variant>
#include "ntars/models/DenseNetwork.hpp"

namespace NTARS
{
    namespace DATA
    {
        template<typename T>
        struct TrainingData
        {
            T data;
            std::string label;
        };
    } // namespace DATA
} // namespace NTARS

#endif // NTARS_DATA_HPP