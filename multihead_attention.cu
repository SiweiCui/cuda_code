//
// Created by CSWH on 2024/12/25.
//
#include <thread>
#include <gtest/gtest.h>
#include <utils.hpp>
#include <cub/block/block_reduce.cuh>


// 一个KV Cache + GQA的实现
// 实现这个mha的过程, 尤其是构建模拟数据的过程, 极大加深了我的理解.
// 构造测试数据是一种好的实践考验.

const int thread_num = 128;

// 自回归的attention中, 一个head中有一个score, 一个head由一个block来处理. 通过在block内reduce来计算softmax
// 懒得处理out of package, 所以先不用float4
__device__ void softmax(float *input, int size) {

	float max_val = -FLT_MAX;
	// 局部最大值收集到一个block内
	for (int i = threadIdx.x; i < size; i+=blockDim.x) {
		max_val = max(max_val, input[i]);
	}
	// 最大值规约
	using BlockReduce = cub::BlockReduce<float, thread_num>;
	__shared__ BlockReduce::TempStorage temp;
	__shared__ float shared_val;
	max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
	if (threadIdx.x == 0) {
		shared_val = max_val;
	}
	__syncthreads();
	max_val = shared_val;
	// printf("head: %d, max: %f, local max: %f\n ", blockIdx.x, shared_val, max_val);

	float sum = 0.f;
	// 局部和
	for (int i = threadIdx.x; i < size; i+=blockDim.x) {
		// printf("head: %d, input[%d] before: %f, sub max: %f\n", blockIdx.x, i, input[i], input[i] - max_val);
		input[i] = expf(input[i] - max_val);
		// printf("head: %d, input[%d]: %f\n", blockIdx.x, i, input[i]);

		sum += input[i];
	}
	// 规约和
	sum = BlockReduce(temp).Sum(sum);
	if (threadIdx.x == 0) {
		shared_val = sum;
	}
	__syncthreads();
	// printf("head: %d, sum: %f, local sum: %f\n ", blockIdx.x, shared_val, sum);

	sum = shared_val;
	// 更新
	for (int i = threadIdx.x; i < size; i+=blockDim.x) {
		input[i] /= sum;
	}
}


// 全局内存尽情用就好了, 不要怕. 先有实现才能有优化.
// 不要着急不要着急, 一点一点来, 今天写不完明天写, 今天不想做安排以后做. 不要着急. 不要老是想着一下子做完.
__global__ void mha_gpu(float *q_pos_gpu, float *k_cache_gpu, float *v_cache_gpu, float *attn_output_gpu, float *score_temp_gpu,
			int pos, int head_num, int kv_head_num, int head_dim, int max_seq) {
	int head_id = blockIdx.x;
	int head_offset = head_id * head_dim;

	int tid = threadIdx.x;
	if (head_id >= head_num) {
		return;
	}

	// head_num / kv_head_num: 一个k/v head对应多少个q head
	int kv_head_id = head_id / (head_num / kv_head_num);

	float scale = 1.0f / sqrtf(head_dim);
	// q依次与之前各个时间点的key做内积, 计算score
	for(int t = tid; t <= pos; t+=blockDim.x) {
		float* k_t = k_cache_gpu + kv_head_id*max_seq*head_dim + t*head_dim;

		//float test_sum = 0.0f;
		//for(int k = 0; k < head_dim; k++) {
		//	test_sum += k_t[k];
		//}
		//if (test_sum == 0.0f) {
		//	printf("第%d个kv头, 第%d个时间点的key向量全为0\n", kv_head_id, t);
		//}

		// 内积, 假设维度能被4整除
		float prod = 0.f;
		for(int i = 0; i < head_dim; i+=4) {
			auto& input1 = reinterpret_cast<float4 &>(q_pos_gpu[head_offset + i]);
			auto& input2 = reinterpret_cast<float4 &>(k_t[i]);
			prod += input1.x * input2.x;
			prod += input1.y * input2.y;
			prod += input1.z * input2.z;
			prod += input1.w * input2.w;
		}
		// 记录到score上
		score_temp_gpu[head_id * max_seq + t] = prod * scale; // 内积很容易变得非常大, 需要加以数值稳定调整
	}
	// __syncthreads();
	// if(threadIdx.x <= pos) {
	// 	printf("head_id: %d, kv_head_id: %d, score[%d]: %f\n", head_id, kv_head_id, threadIdx.x, score_temp_gpu[head_id * max_seq + threadIdx.x]);
	// }

	// 对score使用softmax
	softmax(score_temp_gpu + head_id * max_seq, pos+1);
	__syncthreads();

	// if(threadIdx.x == 0) {
	// 	float test_sum = 0.f;
	// 	for(int t = 0; t <= pos; t++) {
	// 		test_sum += score_temp_gpu[head_id * max_seq + t];
	// 	}
	// 	if(abs(test_sum - 1.f) > 0.001) {
	// 		printf("head_id: %d, score_softmax之和不为1: %f\n", head_id, test_sum);
	// 	}
	// }


	// if(threadIdx.x <= pos) {
	// 	printf("head_id: %d, score_softmax[%d]: %f\n", head_id, threadIdx.x, score_temp_gpu[head_id * max_seq + threadIdx.x]);
	// }

	// 利用score = (score_1, ..., score_pos)对v_1, ..., v_pos进行加权.
	for(int t = tid; t <= pos; t+=blockDim.x) {
		float* v_t = v_cache_gpu + kv_head_id*max_seq*head_dim + t*head_dim;

		float score_t = score_temp_gpu[head_id * max_seq + t];

		for(int i = 0; i < head_dim; i++) {
			// attn_output_gpu[head_offset + i] += score_t * v_t[i]; // 累加会出现冲突问题, 累加需要使用atomicAdd. 赋值则不需要.
			atomicAdd(&attn_output_gpu[head_offset + i], score_t * v_t[i]);
		}
	}
}
/*
 *最后一步可以优化, 调整循环顺序后, 降低全局内存的访问频率
 * 其他优化方案:
 * 1. flash attention
 * 2.
 */


TEST(test_attention, test1) {
	int head_num = 8;
	int kv_head_num = 4; // MQA
	int head_dim = 256;
	int max_seq = 512;
	int pos = 10; // 假设当前处理第10个时间点(时间点从0开始计数)

	float* q_pos_cpu = reinterpret_cast<float*>(malloc(sizeof(float) * head_num * head_dim));
	// 这里kv cache忽略了layer维度. 假设已经定位到了cache所在的layer
	float* k_cache_cpu = reinterpret_cast<float*>(malloc(sizeof(float) * kv_head_num * max_seq * head_dim));
	float* v_cache_cpu = reinterpret_cast<float*>(malloc(sizeof(float) * kv_head_num * max_seq * head_dim));

	// 假设经过了线性变换, 即q_pos = x_pos W_q
	initialize_data_random(q_pos_cpu, head_num * head_dim, 0, 2, false, 100);
	// 填充过去时间点以及当前时间点的数据, 因为线性变换后, k_pos和v_pos会被添加到cache中
	for(int kv_head_id = 0; kv_head_id < kv_head_num; kv_head_id++) {
		initialize_data_random(k_cache_cpu + kv_head_id * max_seq * head_dim, (pos+1) * head_dim, 0, 2, false, 101);
		initialize_data_random(v_cache_cpu + kv_head_id * max_seq * head_dim, (pos+1) * head_dim, 0, 2, false, 102);
	}

	float* q_pos_gpu, * k_cache_gpu, * v_cache_gpu, * attn_output_gpu, * score_temp_gpu;
	cudaMalloc(&q_pos_gpu, head_num * head_dim * sizeof(float));
	cudaMalloc(&k_cache_gpu, kv_head_num * max_seq * head_dim * sizeof(float));
	cudaMalloc(&v_cache_gpu, kv_head_num * max_seq * head_dim * sizeof(float));
	cudaMalloc(&attn_output_gpu, head_num * head_dim * sizeof(float)); // 输出是跟q一样大的
	cudaMalloc(&score_temp_gpu, head_num * max_seq * sizeof(float)); // 给所有时间点都准备了权重.

	cudaMemcpy(q_pos_gpu, q_pos_cpu, head_num * head_dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(k_cache_gpu, k_cache_cpu, kv_head_num * max_seq * head_dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(v_cache_gpu, v_cache_cpu, kv_head_num * max_seq * head_dim * sizeof(float), cudaMemcpyHostToDevice);

	mha_gpu<<<head_num, thread_num>>>(q_pos_gpu, k_cache_gpu, v_cache_gpu, attn_output_gpu, score_temp_gpu,
			pos, head_num, kv_head_num, head_dim, max_seq);

	float* output_cpu = reinterpret_cast<float*>(malloc(sizeof(float) * head_num * head_dim));
	cudaMemcpy(output_cpu, attn_output_gpu, head_num * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

	printf("q_pos:\n");
	show_matrix(q_pos_cpu, 1, 100);// 第一个头的前100
	printf("result\n");
	show_matrix(output_cpu, 1, 100);

	cudaError_t err = cudaGetLastError(); // 检查是否有CUDA错误
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
	}

	free(q_pos_cpu);
	free(k_cache_cpu);
	free(v_cache_cpu);
	free(output_cpu);

	cudaFree(q_pos_gpu);
	cudaFree(k_cache_gpu);
	cudaFree(v_cache_gpu);
	cudaFree(attn_output_gpu);
	cudaFree(score_temp_gpu);
}
