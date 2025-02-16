//
// Created by CSWH on 2024/11/30.
//
#include <cstdlib>
# include<stdio.h>
#include <gtest/gtest.h>

/*
 *编程模型: 一个线程处理结果图片中的一个元素
 */
__global__ void conv(float *input_gpu, float *kernel, float *output_gpu, int height, int width, int kernel_size) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

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


TEST(test_conv, test1) {
	// 定义图片
	int height = 100;
	int width = 200;
	float *input = (float *)malloc(sizeof(float) * height * width);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			input[i * width + j] = i % 256 + j;
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
	float *input_gpu, *kernel_gpu, *output_gpu;
	cudaMalloc((void **)&input_gpu, sizeof(float) * height * width);
	cudaMalloc((void **)&kernel_gpu, sizeof(float) * kernel_size * kernel_size);
	cudaMalloc((void **)&output_gpu, sizeof(float) * height * width);

	cudaMemcpy(input_gpu, input, sizeof(float) * height * width, cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_gpu, kernel, sizeof(float) * kernel_size * kernel_size, cudaMemcpyHostToDevice);

	// 执行计算
	dim3 block_dim(32, 32);
	dim3 grid_dim( (width + 31)/32, (height + 31)/32) ;
	conv<<<grid_dim, block_dim>>>(input_gpu, kernel_gpu, output_gpu, height, width, kernel_size);

	// 接收结果
	float *output = (float *)malloc(sizeof(float) * height * width);
	cudaMemcpy(output, output_gpu, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

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

	cudaFree(input_gpu);
	cudaFree(kernel_gpu);
	cudaFree(output_gpu);
	free(input);
	free(kernel);
	free(output);
}
