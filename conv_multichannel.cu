//
// Created by CSWH on 2024/11/30.
//
#include <cstdlib>
# include<stdio.h>
#include <thread>
#include <gtest/gtest.h>

/*
 *编程模型: 一个线程处理结果图片中的一个元素
 */
__global__ void conv1(float *input_gpu, float *kernel, float *output_gpu, int height, int width, int kernel_size) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.x + threadIdx.y;

	if (row >= height || col >= width) {
		return;
	}

	output_gpu[row * width + col] = 0.f;
	for (int i = 0; i < kernel_size; ++i) {
		for (int j = 0; j < kernel_size; ++j) {
			int processing_row = row - kernel_size/2 + i;
			int processing_col = col - kernel_size/2 + j;
			if (processing_row >= 0 && processing_col >= 0 && processing_row < height && processing_col < width) {
				output_gpu[row*width + col] += kernel[i*kernel_size + j] * input_gpu[processing_row * width + processing_col];
			}
		}
	}
}

__global__ void channel_add(float *channel1_gpu, float *channel2_gpu, float *channel3_gpu, float * output_gpu, int size) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= size) {
		return;
	}
	output_gpu[tid] = channel1_gpu[tid] + channel2_gpu[tid] + channel3_gpu[tid];
}

TEST(test_conv, test2) {
	// 定义图片
	int channel = 3;
	int height = 100;
	int width = 200;
	// C x H x W
	float *input = (float *)malloc(sizeof(float) * height * width * channel);
	for (int c = 0; c < channel; ++c) {
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				input[c * height * width + i * width + j] = i % 256 + j;
			}
		}
	}

	// 定义卷积核
	int kernel_size = 3;
	float *kernel = (float *)malloc(sizeof(float) * kernel_size * kernel_size);
	for (int i = 0; i < kernel_size * kernel_size; ++i) {
		switch (i%3) {
			case 0:
				kernel[i] = -1;
			break;
			case 1:
				kernel[i] = 0;
			break;
			case 2:
				kernel[i] = 1;
			break;
		}
	}


	// GPU
	float *channel1_gpu, *channel2_gpu, *channel3_gpu, *kernel_gpu, *output_gpu;
	float *output_channel1_gpu, *output_channel2_gpu, *output_channel3_gpu;
	cudaMalloc((void **)&channel1_gpu, sizeof(float) * height * width);
	cudaMalloc((void **)&channel2_gpu, sizeof(float) * height * width);
	cudaMalloc((void **)&channel3_gpu, sizeof(float) * height * width);
	cudaMalloc((void **)&kernel_gpu, sizeof(float) * kernel_size * kernel_size);
	cudaMalloc((void **)&output_channel1_gpu, sizeof(float) * height * width);
	cudaMalloc((void **)&output_channel2_gpu, sizeof(float) * height * width);
	cudaMalloc((void **)&output_channel3_gpu, sizeof(float) * height * width);
	cudaMalloc((void **)&output_gpu, sizeof(float) * height * width);

	float *channel1 = input;
	float *channel2 = input + height * width;
	float *channel3 = input + height * width * 2;

	// 创建3个流分别负责不同的channel
	cudaStream_t streams[channel];
	for (int i = 0; i < channel; i++) {
		cudaStreamCreate(&streams[i]);
	}
	// 异步传送数据
	cudaMemcpyAsync(channel1_gpu, channel1, height * width, cudaMemcpyHostToDevice, streams[0]);
	cudaMemcpyAsync(channel2_gpu, channel2, height * width, cudaMemcpyHostToDevice, streams[1]);
	cudaMemcpyAsync(channel3_gpu, channel3, height * width, cudaMemcpyHostToDevice, streams[2]);

	cudaMemcpy(kernel_gpu, kernel, sizeof(float) * kernel_size * kernel_size, cudaMemcpyHostToDevice);

	// 异步计算
	dim3 block_dim(32, 32);
	dim3 grid_dim((height + 31)/32, (width + 31)/32);
	conv1<<<grid_dim, block_dim, 0, streams[0]>>>(channel1_gpu, kernel_gpu, output_channel1_gpu, height, width, kernel_size);
	conv1<<<grid_dim, block_dim, 0, streams[1]>>>(channel2_gpu, kernel_gpu, output_channel2_gpu, height, width, kernel_size);
	conv1<<<grid_dim, block_dim, 0, streams[2]>>>(channel3_gpu, kernel_gpu, output_channel3_gpu, height, width, kernel_size);

	// 1, 2号流同步
	for (int i = 1; i < channel; i++) {
		cudaStreamSynchronize(streams[i]);
	}

	// 结果相加
	int size = height * width;
	int thread_num = 1024;
	int block_num = (size+thread_num-1)/thread_num;
	channel_add<<<block_num, thread_num, 0, streams[0]>>>(output_channel1_gpu, output_channel2_gpu, output_channel3_gpu, output_gpu, size);

	float *output = (float *)malloc(sizeof(float) * height * width);
	cudaMemcpyAsync(output, output_gpu, sizeof(float) * height * width, cudaMemcpyDeviceToHost, streams[0]);

	printf("input:\n");
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 10; ++j) {
			printf("%3.f ", input[i * width + j]);
		}
		printf("\n");
	}

	printf("kernel:\n");
	for (int i = 0; i < kernel_size; ++i) {
		for (int j = 0; j < kernel_size; ++j) {
			printf("%3.f ", kernel[i * kernel_size + j]);
		}
		printf("\n");
	}

	printf("output:\n");
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 10; ++j) {
			printf("%3.f ", output[i * width + j]);
		}
		printf("\n");
	}

	// 后处理
	for (int i = 0; i < channel; i++) {
		cudaStreamDestroy(streams[i]);
	}
	cudaFree(channel1_gpu);
	cudaFree(channel2_gpu);
	cudaFree(channel3_gpu);
	cudaFree(output_gpu);
	free(input);
	free(kernel);
	free(output);
}