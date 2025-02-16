/*
一线程一元素 + 使用共享内存 + load并行
*/
const int BLOCK_DIM = 16;

#include "utils.hpp"
#include <gtest/gtest.h>

/*
一线程一元素 + 使用共享内存 + load并行
考虑M, N, K//BLOCK_DIM
*/
template<int BLOCK_DIM>
__global__ void matmul(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float SA[BLOCK_DIM*BLOCK_DIM];
    __shared__ float SB[BLOCK_DIM*BLOCK_DIM];
    float tmp = 0.f;

    int numOfWindow = (K+BLOCK_DIM-1) / BLOCK_DIM;
    for (int w = 0; w < numOfWindow; w++)
    {
        int windowStart = w*BLOCK_DIM;
        SA[threadIdx.y*BLOCK_DIM + threadIdx.x] = dA[row*K + threadIdx.x+windowStart]; // 避免了bank conflict, 同时合并访存
        SB[threadIdx.y*BLOCK_DIM + threadIdx.x] = dB[(threadIdx.y+windowStart)*N + col];
        __syncthreads();
        
        for (int k = 0; k < BLOCK_DIM; k++)
        {
            tmp += SA[threadIdx.y*BLOCK_DIM + k] * SB[k*BLOCK_DIM + threadIdx.x]; // 固定k, 产生bank conflict
        }
        __syncthreads();
    }
    dC[row*N + col] += tmp;
}

TEST(test_matmul, test1){
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
    printf("CPU计算时间: %.2f seconds\n", float(duration.count())/1000.f);

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
    dim3 grid_dim(N/BLOCK_DIM, M/BLOCK_DIM);
    cudaMemcpy(dA, hA, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(kernelStart);
    matmul<BLOCK_DIM><<<grid_dim, block_dim>>>(dA, dB, dC, M, N, K);
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

