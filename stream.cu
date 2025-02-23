//
// Created by CSWH on 2024/11/30.
//
#include <iostream>
#include <cuda_runtime.h>
#include <gtest/gtest.h>


// 按理说, 需要执行多个核函数的时候, cuda流才好用

#define N 1024  // 向量大小
#define STREAM_COUNT 2  // 使用两个流

// CUDA 核函数：计算向量加法
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		C[index] = A[index] + B[index];
	}
}

TEST(test_stream, test1) {
	float *A, *B, *C;
	float *d_A, *d_B, *d_C;
	size_t size = N * sizeof(float);

	// 分配主机内存
	A = (float*)malloc(size);
	B = (float*)malloc(size);
	C = (float*)malloc(size);

	// 初始化向量
	for (int i = 0; i < N; i++) {
		A[i] = i;
		B[i] = 2 * i;
	}

	// 分配设备内存
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	// 创建 CUDA 流
	cudaStream_t streams[STREAM_COUNT];
	for (int i = 0; i < STREAM_COUNT; i++) {
		cudaStreamCreate(&streams[i]);
	}

	// 将数据传输到设备内存
	cudaMemcpyAsync(d_A, A, size, cudaMemcpyHostToDevice, streams[0]);
	cudaMemcpyAsync(d_B, B, size, cudaMemcpyHostToDevice, streams[1]);

	// 启动向量加法核函数
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(d_A, d_B, d_C, N);

	// 从设备复制结果到主机
	cudaMemcpyAsync(C, d_C, size, cudaMemcpyDeviceToHost, streams[1]);

	// 等待所有流完成
	for (int i = 0; i < STREAM_COUNT; i++) {
		cudaStreamSynchronize(streams[i]);
	}

	// 打印结果中的前几个元素
	for (int i = 0; i < 10; i++) {
		std::cout << "C[" << i << "] = " << C[i] << std::endl;
	}

	// 清理资源
	free(A);
	free(B);
	free(C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	for (int i = 0; i < STREAM_COUNT; i++) {
		cudaStreamDestroy(streams[i]);
	}

}
