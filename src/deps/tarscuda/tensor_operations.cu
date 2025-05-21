#include "tensor_operations.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

namespace TCUDA
{
    inline void CUDA_CHECK(cudaError_t cudaStatus) 
    {
        if (cudaStatus != cudaSuccess) {
            throw fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
        }
    }

    inline void CUBLAS_CHECK(cublasStatus_t cublasStatus) 
    {
        if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
            throw fprintf(stderr, "CUBLAS error: %d\n", cublasStatus);
        }
    }

    void swapValue(int& val1, int& val2)
    {
        int temp = val1;
        val1 = val2;
        val2 = temp;
    }

    bool matrixMultiply(float* x, float* y, float* result, int rowsX, int colsX, int colsY, bool transposeX, bool transposeY) 
    {
        float *d_a, *d_b, *d_result;

        const float alpha = 1.0; // Scalar multiplier for matrix multiplication
        const float beta = 0.0;  // Scalar multiplier for initial values in result

        if (transposeX) {
            swapValue(rowsX, colsX);
        }
        if (transposeY) {
            swapValue(colsX, colsY);
        }

        size_t sizeX = rowsX * colsX * sizeof(float);
        size_t sizeY = colsX * colsY * sizeof(float);
        size_t sizeResult = rowsX * colsY * sizeof(float);

        CUDA_CHECK(cudaMallocManaged(&d_a, sizeX)); 
        CUDA_CHECK(cudaMallocManaged(&d_b, sizeY));
        CUDA_CHECK(cudaMallocManaged(&d_result, sizeResult));
        
        CUDA_CHECK(cudaMemcpy(d_a, x, sizeX, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, y, sizeY, cudaMemcpyHostToDevice));

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        cublasOperation_t opX = transposeX ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opY = transposeY ? CUBLAS_OP_T : CUBLAS_OP_N;

        CUBLAS_CHECK(cublasSgemm_v2(cublasHandle, opX, opY, colsY, rowsX, colsX, &alpha, d_b, colsY, d_a, colsX, &beta, d_result, colsY));
        cudaDeviceSynchronize();

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

    bool matrixElementWiseMultiply(float* x, float* y, float* result, size_t size)
    {
        float *d_a, *d_b, *d_result;
    
        size_t dataSize = size * sizeof(float);
    
        CUDA_CHECK(cudaMallocManaged(&d_a, dataSize));
        CUDA_CHECK(cudaMallocManaged(&d_b, dataSize));
        CUDA_CHECK(cudaMallocManaged(&d_result, dataSize));

        CUDA_CHECK(cudaMemcpy(d_a, x, dataSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, y, dataSize, cudaMemcpyHostToDevice));
    
        dim3 blockDim(256); 
        dim3 gridDim((size + blockDim.x - 1) / blockDim.x); 

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        CUBLAS_CHECK(cublasSdgmm(
            cublasHandle,                          // cuBLAS handle
            CUBLAS_SIDE_LEFT,                      // Side mode (LEFT means x is the diagonal matrix)
            size,                                  // Number of rows in the matrix
            1,                                     // Number of columns in the matrix
            d_b, size,                             // Input matrix y (column vector)
            d_a, 1,                                // Input vector x (diagonal-like input)
            d_result, size                         // Resulting output matrix
        ));
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaMemcpy(result, d_result, dataSize, cudaMemcpyDeviceToHost));

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
    
} // namespace TCUDA

