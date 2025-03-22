#include "tensor_operations.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

namespace TCUDA
{
    inline void CUDA_CHECK(cudaError_t cudaStatus) 
    {
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
            throw std::runtime_error("CUDA error");
        }
    }

    inline void CUBLAS_CHECK(cublasStatus_t cublasStatus) 
    {
        if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "CUBLAS error: %d\n", cublasStatus);
            exit(-1);
        }
    }

/*  uses Cuda Cores   bool matrixMultiply(double* x, double* y, double* result, int rowsX, int colsX, int colsY) 
    {
        double *d_x, *d_y, *d_result;
        size_t sizeX = rowsX * colsX * sizeof(double);
        size_t sizeY = colsX * colsY * sizeof(double);
        size_t sizeResult = rowsX * colsY * sizeof(double);
    
        CUDA_CHECK(cudaMalloc(&d_x, sizeX));
        CUDA_CHECK(cudaMalloc(&d_y, sizeY));
        CUDA_CHECK(cudaMalloc(&d_result, sizeResult));
    
        CUDA_CHECK(cudaMemcpy(d_x, x, sizeX, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, y, sizeY, cudaMemcpyHostToDevice));
    
        dim3 blockDim(16, 16);
        dim3 gridDim((colsY + 15) / 16, (rowsX + 15) / 16);
        matrixMultiplyKernel<<<gridDim, blockDim>>>(d_x, d_y, d_result, rowsX, colsX, colsY);

        CUDA_CHECK(cudaMemcpy(result, d_result, sizeResult, cudaMemcpyDeviceToHost));

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_result);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
    
        return true; 
    } */
    bool matrixMultiply(float* x, float* y, float* result, int rowsX, int colsX, int colsY, bool transposeX, bool transposeY) 
    {
        float *d_a, *d_b, *d_result;

        // scalar coefficients
        const float alpha = 1.0;
        const float beta = 0.0;

        size_t sizeX = rowsX * colsX * sizeof(float);
        size_t sizeY = colsX * colsY * sizeof(float);
        size_t sizeResult = rowsX * colsY * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_a, sizeX)); 
        CUDA_CHECK(cudaMalloc(&d_b, sizeY));
        CUDA_CHECK(cudaMalloc(&d_result, sizeResult));
        
        CUDA_CHECK(cudaMemcpy(d_a, x, sizeX, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, y, sizeY, cudaMemcpyHostToDevice));

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        cublasOperation_t opX = transposeX ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opY = transposeY ? CUBLAS_OP_T : CUBLAS_OP_N;

        CUBLAS_CHECK(cublasSgemm_v2(cublasHandle, opX, opY, colsY, rowsX, colsX, &alpha, d_b, colsY, d_a, colsX, &beta, d_result, colsY));
        CUDA_CHECK(cudaMemcpy(result, d_result, sizeResult, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_result));
        CUBLAS_CHECK(cublasDestroy(cublasHandle));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        return true;
    }

    bool matrixMultiply(std::vector<std::vector<float>>& matrixA, std::vector<std::vector<float>>& matrixB,  std::vector<std::vector<float>>* resultMatrix, bool transposeX, bool transposeY)
    {
        const float alpha = 1.0;
        const float beta = 0.0;

        size_t sizeX = matrixA.size() * matrixA[0].size() * sizeof(float);
        size_t sizeY =  matrixA[0].size() * matrixB[0].size() * sizeof(float);
        size_t sizeResult = matrixA.size() * matrixB[0].size() * sizeof(float);

        return false;
    }

    __global__ void matrixElementWiseMultiplyKernel(const float* x, const float* y, float* result, size_t size)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            result[idx] = x[idx] * y[idx];
        }
    }
    
    bool matrixElementWiseMultiply(float* x, float* y, float* result, size_t size)
    {
        float *d_x, *d_y, *d_result;
    
        size_t dataSize = size * sizeof(float);
    
        CUDA_CHECK(cudaMalloc(&d_x, dataSize));
        CUDA_CHECK(cudaMalloc(&d_y, dataSize));
        CUDA_CHECK(cudaMalloc(&d_result, dataSize));

        CUDA_CHECK(cudaMemcpy(d_x, x, dataSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, y, dataSize, cudaMemcpyHostToDevice));
    
        dim3 blockDim(256); 
        dim3 gridDim((size + blockDim.x - 1) / blockDim.x); 

        matrixElementWiseMultiplyKernel<<<gridDim, blockDim>>>(d_x, d_y, d_result, size);

        CUDA_CHECK(cudaMemcpy(result, d_result, dataSize, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_result));
    
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        return true;
    }
    
} // namespace TCUDA

