/*
一线程多元素 + 使用共享内存 + load并行 + float4
*/
const int BLOCK_DIM = 16;
const int BK = 8;
const int BM = 128, BN = 128;
const int TM = 8, TN = 8;

#include <stdio.h>
#include <random>
#include <chrono>
#include "utils.hpp"
#include <gtest/gtest.h>

template<int BLOCK_DIM, int BK, int BM, int BN, int TM, int TN>
__global__ void matmulV5(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int blockRowStart = BM * blockIdx.y; // 本block负责的元素的起点
    int blockColStart = BN * blockIdx.x;
    float tmp[TM*TN] = {0.f};
    __shared__ float SA[BM * BK];
    __shared__ float SB[BK * BN];
    int numOfWindow = (K+BK-1) / BK;
    int tidInBlock = threadIdx.x + threadIdx.y * blockDim.x; // 0~255

    // 在一次循环中, 为了计算128 * 128个C矩阵元素的值, 我们需要128 * 8的A的元素, 8 * 128个B的元素
    // 每个线程load 4个, 正好可以用128个线程load所有元素
    // aRow: 0 0 1 1 2 2 ... 127 127
    // aCol: 0 1 0 1 0 1 ...  0   1
    // bRow: 0 1 2 3 4 5 6 7 0 1 2 ... 6  7
    // bCol: 0 0 0 0 0 0 0 0 1 1 1 ... 31 31
    int aRow = tidInBlock / 2; // 0~127
    int aCol = tidInBlock % 2; // 0~1
    int bRow = tidInBlock % 8; // 0~7
    int bCol = tidInBlock / 8; // 0~31
    for (int w = 0; w < numOfWindow; w++)
    {
        int windowStart = w * BK;

        (float4 &)SA[aRow*BK + 4*aCol] = (float4 &)dA[(blockRowStart+aRow)*K + windowStart+4*aCol]; // Bank Conflict
        (float4 &)SB[bRow*BN + 4*bCol] = (float4 &)dB[(windowStart+bRow)*N + blockColStart+4*bCol];
        __syncthreads();

        int threadRowStart = TM * threadIdx.y;
        int threadColStart = TN * threadIdx.x;
        for (int row = 0; row < TM; row++)
        {
            for (int col = 0; col < TN; col++)
            {
                for (int k = 0; k < BK; k++)
                {
                    tmp[row*TN + col] += SA[(threadRowStart+row)*BK + k] * SB[k*BN + threadColStart+col]; // Bank Conflict
                }
            }
        }
        __syncthreads();
    }

    int rowStart = blockRowStart + TM*threadIdx.y;
    int colStart = blockColStart + TN*threadIdx.x;
    for (int row = 0; row < TM; row++)
    {
        for (int col = 0; col < TN; col++)
        {
            dC[(rowStart+row)*N + colStart+col] = tmp[row*TN + col];
        }
    }

}

TEST(test_matmul, test_v5){
    int M = 1024, K = 1024, N = 1024;
    float *hA= reinterpret_cast<float*>(malloc(M*K * sizeof(float)));
    float *hB = reinterpret_cast<float*>(malloc(K*N * sizeof(float)));
    float *hC = reinterpret_cast<float*>(malloc(M*N * sizeof(float)));
    initialize_data_random(hA, M*K);
    initialize_data_random(hB, K*N);

    auto start = std::chrono::high_resolution_clock::now();
    cpu_matmul(hA, hB, hC, M, K, N);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("CPU计算时间: %.2f 毫秒\n", float(duration.count()));

/*
    show_matrix(hA, M, N);
    printf("\n");
    show_matrix(hB, M, N);
    printf("\n");
    show_matrix(hC, M, N);
*/

    cudaEvent_t cudaStart, cudaStop, kernelStart, kernelStop;
    cudaEventCreate(&cudaStart);
    cudaEventCreate(&cudaStop);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    float gpuTime, kernelTime;

    cudaEventRecord(cudaStart, 0);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, M*K*sizeof(float));
    cudaMalloc(&dB, K*N*sizeof(float));
    cudaMalloc(&dC, M*N*sizeof(float));
    float *gpuC = reinterpret_cast<float*>(malloc(M*N*sizeof(float)));

    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim(M/BM, N/BN);
    cudaMemcpy(dA, hA, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(kernelStart);
    matmulV5<BLOCK_DIM, BK, BM, BN, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(kernelStop);

    cudaMemcpy(gpuC, dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(cudaStop, 0);
    cudaDeviceSynchronize();

    cudaEventElapsedTime(&gpuTime, cudaStart, cudaStop);
    cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop);
    printf("GPU时间: %f 毫秒\n", gpuTime);
    printf("核函数时间: %f 毫秒\n", kernelTime);
    

    printf("CPU结果:\n");
    show_matrix(hC, 1, 50);

    printf("GPU结果:\n");
    show_matrix(gpuC, 1, 50);

    error_check(hC, gpuC, M*N);


    cudaError_t err = cudaGetLastError(); // 检查是否有CUDA错误
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    free(hA);
    free(hB);
    free(hC);
    free(gpuC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(cudaStart);
    cudaEventDestroy(cudaStop);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);
}