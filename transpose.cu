//
// Created by CSWH on 2024/12/1.
//

#include <bits/fs_path.h>
#include<gtest/gtest.h>
#include "utils.hpp"


/*     A1  A2               A1^T   A3^T
 *A =           Then A^T =
 *     A3  A4               A2^T   A4^T
 */
template<int BLOCK_DIM>
__global__ void transpose2(float *input, float *output, int M, int N) {
    __shared__ float sdata[BLOCK_DIM * BLOCK_DIM];
    int row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int col = blockIdx.x * BLOCK_DIM + threadIdx.x;
    // if (row >= M || col >= N) {
    //     return;
    // }提前终止线程是一个不好的做法

    // load分块数据, 原封不动load
    if (row < M && col < N) {
        sdata[threadIdx.y*BLOCK_DIM + threadIdx.x] = input[row*N + col];// 正常读, 合并访存, 也避免了bank conflict
    }
    __syncthreads();

    // 写回, 先转置整个分块矩阵(交换block维度), 再在每个分块矩阵内转置(分块sdata内访问对角元素). 主要是合并访存
    int target_row = blockIdx.x * BLOCK_DIM + threadIdx.y;
    int target_col = blockIdx.y * BLOCK_DIM + threadIdx.x;
    if (target_row < N && target_col < M) {
        output[target_row*M + target_col] = sdata[threadIdx.x*BLOCK_DIM + threadIdx.y];// 合并访存, 但是出现bank conflict
                                             // sdata访问对角元素
    }
}

const int BLOCK_DIM = 16;

void cpu_transpose(float *input, float* output, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            output[j*M + i] = input[i*N + j];
        }
    }
}

template<int BLOCK_DIM>
__global__ void transpose(float *input, float *output, int M, int N) {
    int row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int col = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (row < M && col < N) {
        output[col*M + row] = input[row*N + col]; // output无法合并访存
    }
}


TEST(test_transpose, test1){
    int M = 3, N = 4;
    float *hA= reinterpret_cast<float*>(malloc(M*N * sizeof(float)));
    initialize_data_random(hA, M*N);
    printf("hA:\n");
    show_matrix(hA, M, N);

    float *hA_cpu_result = reinterpret_cast<float*>(malloc(M*N * sizeof(float)));
    cpu_transpose(hA, hA_cpu_result, M, N);
    printf("cpu result:\n");
    show_matrix(hA_cpu_result, N, M);

    float *dA, *dA_result;
    cudaMalloc(&dA, M*N*sizeof(float));
    cudaMalloc(&dA_result, M*N*sizeof(float));
    float *hA_result = reinterpret_cast<float*>(malloc(M*N*sizeof(float)));

    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim((N + BLOCK_DIM - 1)/BLOCK_DIM, (M + BLOCK_DIM - 1)/BLOCK_DIM);
    cudaMemcpy(dA, hA, M*N*sizeof(float), cudaMemcpyHostToDevice);
    transpose2<BLOCK_DIM><<<grid_dim, block_dim>>>(dA, dA_result, M, N);

    cudaMemcpy(hA_result, dA_result, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("gpu result:\n");
    show_matrix(hA_result, N, M);

    cudaError_t err = cudaGetLastError(); // 检查是否有CUDA错误
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }


}
