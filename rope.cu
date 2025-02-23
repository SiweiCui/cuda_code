#include<gtest/gtest.h>
/*
 *优化:
 *1. shared memory. 将block所用的数据load进来
 *2. 一线程处理多seq/多head
 *3. 如果是GQA, 可以分成两个核函数来分别编码. 此时还可引入cuda stream.
 */

// 一个block处理一个seq的一个head
__global__ void rope(float* q, float* k, float* q_output, float* k_output, int seq, int head_num, int head_dim) {

	int seq_idx = blockIdx.x/head_num;
	int head_idx = blockIdx.x % head_num;
	int seq_offset = seq_idx * head_num * head_dim;
	int head_offset = head_idx * head_dim;

	int tid = threadIdx.x;
	// 线程0: (0, 1), (256, 257), ...
	// 线程1: (2, 3), (258, 259), ...
	// ...
	// 线程127: (254, 255), ...
	for(int i = tid; i < head_dim; i+= blockDim.x) {
		int idx = i*2;
		if(idx > head_dim){return;}

		float theta = 1.f / pow(10000.f, 2*(idx-1)/head_dim);
		q_output[seq_offset+head_offset+idx] = cos(seq_idx*theta)*q[seq_offset+head_offset+idx] - sin(seq_idx*theta)*q[seq_offset+head_offset+idx+1];
		q_output[seq_offset+head_offset+idx+1] = cos(seq_idx*theta)*q[seq_offset+head_offset+idx] + sin(seq_idx*theta)*q[seq_offset+head_offset+idx+1];

		k_output[seq_offset+head_offset+idx] = cos(seq_idx*theta)*k[seq_offset+head_offset+idx] - sin(seq_idx*theta)*k[seq_offset+head_offset+idx+1];
		k_output[seq_offset+head_offset+idx+1] = cos(seq_idx*theta)*k[seq_offset+head_offset+idx] + sin(seq_idx*theta)*k[seq_offset+head_offset+idx+1];
	}
}



TEST(test_rope, test1) {
	int seq = 128;
	int head_num = 4;
	int head_dim = 256;
	// 初始化
	float* q = (float*)malloc(sizeof(float)*seq*head_num*head_dim);
	float* k = (float*)malloc(sizeof(float)*seq*head_num*head_dim);

	for(int i = 0; i < seq*head_num*head_dim; i++) {
		q[i] = (rand()%10)/10.f;
		k[i] = (rand()%10)/10.f;
	}

	// 转移数据
	float* q_gpu, *k_gpu;
	cudaMalloc(&q_gpu, sizeof(float)*seq*head_num*head_dim);
	cudaMalloc(&k_gpu, sizeof(float)*seq*head_num*head_dim);
	cudaMemcpy(q_gpu, q, sizeof(float)*seq*head_num*head_dim, cudaMemcpyHostToDevice);
	cudaMemcpy(k_gpu, k, sizeof(float)*seq*head_num*head_dim, cudaMemcpyHostToDevice);

	// 定义输出
	float* q_output_gpu, *k_output_gpu;
	cudaMalloc(&q_output_gpu, sizeof(float)*seq*head_num*head_dim);
	cudaMalloc(&k_output_gpu, sizeof(float)*seq*head_num*head_dim);

	// rope
	dim3 block_dim(128);
	dim3 grid_dim(seq*head_num);
	rope<<<grid_dim, block_dim>>>(q_gpu, k_gpu, q_output_gpu, k_output_gpu, seq, head_num, head_dim);

	// 输出结果
	float *q_output_gpu2cpu = (float *) malloc(sizeof(float) * seq * head_num * head_dim);
	float *k_output_gpu2cpu = (float *) malloc(sizeof(float) * seq * head_num * head_dim);
	cudaMemcpy(q_output_gpu2cpu, q_output_gpu, sizeof(float)*seq*head_num*head_dim, cudaMemcpyDeviceToHost);
	cudaMemcpy(k_output_gpu2cpu, k_output_gpu, sizeof(float)*seq*head_num*head_dim, cudaMemcpyDeviceToHost);
}


