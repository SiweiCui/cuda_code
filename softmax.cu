#include <gtest/gtest.h>
# include "utils.hpp"


void softmax_cpu(float *input, float *output, int length) {
	float sum = 0;
	float m = -FLT_MAX;
	for (int i = 0; i < length; i++) {
		m = max(m, input[i]);
	}
	for (int i = 0; i < length; i++) {
		sum += expf(input[i] - m);
	}
	for (int i = 0; i < length; i++) {
		// printf("input: %f\n", input[i]);
		// printf("exp: %f\n", expf(input[i] - m));
		output[i] = expf(input[i] - m)/sum;
	}
	// printf("sum: %f\n", sum);
	// printf("m: %f\n", m);
}

/*
 *online原理:
 *m_t+1 = max(m_t, max(当前分块))
 *l_t+1 = exp(m_t - m_t+1)lt + sum(exp(当前分块-m_t+1))
 *三次循环 -> 两次循环
 *假设length整除block大小
 *先写出来再考虑优化.
 */
__global__ void online_softmax_gpu(float *input, float *output, int length) {
	__shared__ float m;
	__shared__ float l;

	m = -FLT_MAX;
	l = 0;
	__syncthreads();

	int package_size = 4;
	int package_id = threadIdx.x + blockIdx.x * blockDim.x;
	int package_num = length / package_size;

	// online计算
	// 先在当前thread负责的package中online, 然后在共享内存中online
	float m_thread = -FLT_MAX;
	float l_thread = 0;
	for (int i = package_id; i < package_num; i += blockDim.x) {
		int package_start = i * package_size;
		float4& data = reinterpret_cast<float4 &>(input[package_start]);
		float temp = m_thread;

		m_thread = max(m_thread, data.x);
		m_thread = max(m_thread, data.y);
		m_thread = max(m_thread, data.z);
		m_thread = max(m_thread, data.w);

		l_thread *= expf(temp - m_thread);
		l_thread += expf(data.x - m_thread) + expf(data.y - m_thread) + expf(data.z - m_thread) + expf(data.w - m_thread);
	}
	// reduce m和l
	m = max(m, m_thread);
	__syncthreads();
	// printf("id: %d, m_thread: %f, l_thread: %f, exp: %f, add: %f\n", package_id, m_thread, l_thread, expf(m_thread - m), l_thread*expf(m_thread-m));
	float add = l_thread * expf(m_thread - m);
	// l += add; // 直接累加结果是错的. 只能覆盖, 不能累加
	atomicAdd(&l, add); // 需要用原子操作或者block reduce
	__syncthreads();
	// printf("gpu l:%f\n", l);
	// printf("gpu m:%f\n", m);

	// 更新
	for (int i = package_id; i < length; i += blockDim.x) {
		int package_start = i * package_size;
		auto& data = reinterpret_cast<float4 &>(input[package_start]);
		auto& output_data = reinterpret_cast<float4&>(output[package_start]);

		output_data.x = expf(data.x - m) / l;
		output_data.y = expf(data.y - m) / l;
		output_data.z = expf(data.z - m) / l;
		output_data.w = expf(data.w - m) / l;

		// output_data.x = expf(data.x - m_thread) / l_thread;
		// output_data.y = expf(data.y - m_thread) / l_thread;
		// output_data.z = expf(data.z - m_thread) / l_thread;
		// output_data.w = expf(data.w - m_thread) / l_thread;
	}
}


TEST(test_softmax, test1) {
	int size = 1024;
	float* vector_cpu = static_cast<float*>(malloc(size * sizeof(float)));
	initialize_data_random(vector_cpu, size);

	float* result_cpu = static_cast<float*>(malloc(size * sizeof(float)));
	softmax_cpu(vector_cpu, result_cpu, size);
	printf("vector_cpu:\n");
	show_matrix(vector_cpu, 1, 20);
	printf("result_cpu:\n");
	show_matrix(result_cpu, 1, 20);

	float* vector_gpu, *result_gpu;
	cudaMalloc(&vector_gpu, size * sizeof(float));
	cudaMalloc(&result_gpu, size * sizeof(float));
	cudaMemcpy(vector_gpu, vector_cpu, size * sizeof(float), cudaMemcpyHostToDevice);

	online_softmax_gpu<<<1, 128>>>(vector_gpu, result_gpu, size); // 如果是矩阵, 可以每一行开一个block

	float* result_gpu_cpu = reinterpret_cast<float*>(malloc(size * sizeof(float)));
	cudaMemcpy(result_gpu_cpu, result_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
	printf("result_gpu:\n");
	show_matrix(result_gpu_cpu, 1, 20);

	error_check(result_cpu, result_gpu_cpu, size);
}