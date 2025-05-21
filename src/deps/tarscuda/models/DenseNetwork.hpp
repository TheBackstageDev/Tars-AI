#ifndef TARSCUDA_DENSENETWORK_HPP
#define TARSCUDA_DENSENETWORK_HPP

#include "ntars/base/data.hpp"
#include "ntars/models/DenseNetwork.hpp"

namespace NCUDA_NETWORK
{
    void denseTrain(std::vector<std::vector<NTARS::DATA::TrainingData<std::vector<float>>>> trainingData, float trainingRate, NTARS::DenseNeuralNetwork& network);
}

#endif // TARSCUDA_DENSENETWORK_HPP