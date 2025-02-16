//
// Created by CSWH on 2024/12/25.
//
#include <cub/block/block_reduce.cuh>
#include <gtest/gtest.h>
#include "utils.hpp"

void gemv_cpu(float *x, float *W, float *y, int input_size, int output_size) {
	for (int i = 0; i < output_size; i++) {
		for (int j = 0; j < input_size; j++) {
			y[i] += x[j] * W[i*input_size+j];
		}
	}
}

// 线程模型: 每个block处理多个行, 一个block计算y的多个元素.
// 在block内做内积收集
template<int ROW_PER_BLOCK>
__global__ void gemv_gpu(float *x, float *W, float *y, int input_size, int output_size) {
	int tid = threadIdx.x;
	int row_start = blockIdx.x * ROW_PER_BLOCK;

	for (int row = row_start; row < min(row_start + ROW_PER_BLOCK, output_size); row++) {
		float row_inner_prod_local = 0.f;
		// 收集到一个block内
		for(int t = tid; t < input_size; t+=blockDim.x) {
			row_inner_prod_local += x[t] * W[row*input_size+t];
		}

		// block收集, 计算内积
		using BlockReduce = cub::BlockReduce<float, 128>;
		__shared__  BlockReduce::TempStorage temp;
		float row_inner_prod = BlockReduce(temp).Sum(row_inner_prod_local);
		__syncthreads();

		// 写回
		if(tid == 0) {
			y[row] = row_inner_prod;
		}

	}
}


// x与W的每一行做内积
TEST(test_gemv, test1) {
	int input_size = 1024;
	int output_size = 2048;

	float *x_cpu = reinterpret_cast<float *>(malloc(input_size * sizeof(float)));
	float *W_cpu = reinterpret_cast<float *>(malloc(output_size * input_size * sizeof(float)));
	initialize_data_random(x_cpu, input_size);
	initialize_data_random(W_cpu, output_size * input_size);

	float *y_cpu = reinterpret_cast<float *>(malloc(output_size * sizeof(float)));
	memset(y_cpu, 0, output_size * sizeof(float));
	gemv_cpu(x_cpu, W_cpu, y_cpu, input_size, output_size);


	float *x_gpu, *W_gpu, *y_gpu;
	cudaMalloc(&x_gpu, sizeof(float) * input_size);
	cudaMalloc(&W_gpu, sizeof(float) * output_size * input_size);
	cudaMalloc(&y_gpu, sizeof(float) * output_size);
	cudaMemcpy(x_gpu, x_cpu, sizeof(float) * input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(W_gpu, W_cpu, sizeof(float) * output_size * input_size, cudaMemcpyHostToDevice);

	const int row_per_block = 64;
	gemv_gpu<row_per_block><<<output_size/row_per_block, 128>>>(x_gpu, W_gpu, y_gpu, input_size, output_size);

	cudaError_t err = cudaGetLastError(); // 检查是否有CUDA错误
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
	}

	float* y_gpu_cpu = reinterpret_cast<float*>(malloc(sizeof(float) * output_size));
	cudaMemcpy(y_gpu_cpu, y_gpu, sizeof(float) * output_size, cudaMemcpyDeviceToHost);

	printf("y_gpu_cpu: \n");
	show_matrix(y_gpu_cpu, 1, 100);
	printf("y_cpu\n");
	show_matrix(y_cpu, 1, 100);

	error_check(y_gpu_cpu, y_cpu, output_size);
}