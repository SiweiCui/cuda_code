#include <stdio.h>
#include <random>
#include <cub/cub.cuh>
#include "utils.hpp"
#include <gtest/gtest.h>

void rmsnorm_kernel_cpu(const float *in_ptr, const float *weight,
                        float *out_ptr, int size) {

    //   float* in_ptr = const_cast<float*>(input.ptr<float>());
    //   float* out_ptr = const_cast<float*>(output.ptr<float>());

    //   int size = static_cast<int32_t>(input.size());
    float sum = 0.f;
    for (int i = 0; i < size; ++i) {
        sum += in_ptr[i] * in_ptr[i];
    }

    const float eps = 1e-5f;
    float mean = sum / float(size) + eps;

    // printf("CPU平方和是%f\n", sum);

    const float rsqrt = 1.f / std::sqrt(mean);

    // printf("CPU的乘子是%f\n", rsqrt);
    for (int i = 0; i < size; ++i) {
        out_ptr[i] = weight[i] * rsqrt * in_ptr[i];
    }
}

static __global__ void row_rmsnorm_f32(const float* in, const float* wei, float* out,  const int size, const float eps) {
    const int tid = threadIdx.x; // 线程 id 多个线程是同时进来的。
    const int lane_id = tid % warpSize; // 这个线程id在warp内的编号

    float sum = 0.0f;
    // lane_id从0到31都有，是同时执行的
    for (int i = lane_id; i < size; i += warpSize) {
        sum += in[i] * in[i];
    }
    __syncthreads();

    // 根据局部和求出sum的全局和
    using WarpReduce = cub::WarpReduce<float, 32>;
    __shared__ WarpReduce::TempStorage temp;
    __shared__ float shared_val;
    sum = WarpReduce(temp).Reduce(sum, cub::Sum());

    if (lane_id == 0)
    {
        shared_val = sum;
    }
    __syncthreads();
    
    const float scale = rsqrtf(shared_val / static_cast<float>(size) + eps);
    // if(tid == 32)
    // {
    //     printf("tid: %d\n", tid);
    //     printf("lane_id: %d\n", lane_id);
    //     printf("GPU平方和是%f\n", sum);
    //     printf("GPU乘子是%f\n", scale);
    // }

    for (int i = lane_id; i < size; i += warpSize) {
        out[i] = scale * in[i] * wei[i];
    }
}

TEST(test_rmsnorm, test1){
    int D_MODEL = 64;
    float *weight = reinterpret_cast<float*>(malloc(D_MODEL * sizeof(float)));
    for (size_t i = 0; i < D_MODEL; i++)
    {
        weight[i] = 1.f;
    }
    
    float *hA= reinterpret_cast<float*>(malloc(D_MODEL * sizeof(float)));
    float *hC = reinterpret_cast<float*>(malloc(D_MODEL * sizeof(float)));
    initialize_data_random(hA, D_MODEL);

    auto start = std::chrono::high_resolution_clock::now();
    rmsnorm_kernel_cpu(hA, weight, hC, D_MODEL);
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
    float *dA, *dC, *dweight;
    cudaMalloc(&dA, D_MODEL*sizeof(float));
    cudaMalloc(&dC, D_MODEL*sizeof(float));
    cudaMalloc(&dweight, D_MODEL*sizeof(float));
    float *gpuC = reinterpret_cast<float*>(malloc(D_MODEL*sizeof(float)));

    dim3 block_dim(D_MODEL);
    dim3 grid_dim(1);
    cudaMemcpy(dA, hA, D_MODEL*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dweight, weight, D_MODEL*sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(kernelStart);
    row_rmsnorm_f32<<<grid_dim, block_dim>>>(dA, dweight, dC, D_MODEL, 1e-5f);
    cudaEventRecord(kernelStop);

    cudaMemcpy(gpuC, dC, D_MODEL*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(cudaStop, 0);
    cudaDeviceSynchronize();

    cudaEventElapsedTime(&gpuTime, cudaStart, cudaStop);
    cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop);
    printf("GPU时间: %f 毫秒\n", gpuTime);
    printf("核函数时间: %f 毫秒\n", kernelTime);
    
    // show_matrix(gpuC, M, N);
    error_check(hC, gpuC, D_MODEL);
    
    // printf("cpu:\n");
    // show_matrix(hC, 1, D_MODEL);
    // printf("gpu:\n");
    // show_matrix(gpuC, 1, D_MODEL);

    free(hA);
    free(hC);
    free(gpuC);
    cudaFree(dA);
    cudaFree(dC);
    cudaEventDestroy(cudaStart);
    cudaEventDestroy(cudaStop);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);
}
