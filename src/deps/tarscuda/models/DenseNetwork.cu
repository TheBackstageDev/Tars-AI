#include "DenseNetwork.hpp"
#include <cuda_fp16.h>
#include <mma.h>

namespace NCUDA_NETWORK
{
    __device__ float sigmoid(float x)
    {
        return 1 / (1 + exp(-x));
    }

    // M Is rows A, K is cols A, N is cols B
    __device__ void gemm(half* A, half* B, half* output, int M, int K, int N, bool transposeA, bool transposeB)
    {

    }

    // Outputs means the vector where all summed changes will go
    __global__ void trainKernel(half* weights, half* biases, half* outWeights, half* outBiases, float learningRate) 
    {

    }

    void denseTrain(std::vector<std::vector<NTARS::DATA::TrainingData<std::vector<float>>>> trainingData, float trainingRate, NTARS::DenseNeuralNetwork& network)
    {
        float* outputs; 
    }
}
