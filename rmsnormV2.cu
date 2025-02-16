#include <cub/block/block_reduce.cuh>
//
// Created by CSWH on 2024/11/28.
//

// 不要蹑手蹑脚的, 有思路就写, 把思路实现. 不要想着忠于原文.

template <int32_t BLOCK_DIM>
__global__ void rmsnormV2(float *in_ptr, float *weight, float *out_ptr, int size, float eps) {
	int tid = threadIdx.x;
	int pack_num = size / 4;

	float sum = 0.f;
	float4 *in_float4 = (float4 *)in_ptr;
	for (int i = tid; i < pack_num; i+=blockDim.x) {
		float4 package = *(in_float4 + i); // 解引用
		sum += package.x * package.x;
		sum += package.y * package.y;
		sum += package.z * package.z;
		sum += package.w * package.w;
	}
	for (int i = threadIdx.x + pack_num; i < size; i += blockDim.x) {
		sum += in_ptr[i] * in_ptr[i];
	}

	using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
	__shared__ typename BlockReduce::Temp temp;
	__shared__ float shared_val;
	sum = BlockReduce(temp).Sum(sum);
	if (threadIdx.x == 0) {
		shared_val = sum;
	}
	__syncthreads();

	float scale = rsqrtf(shared_val / (float)size + eps);
	float4 *weight_float4 = (float4 *)weight;
	float4 *out_float4 = (float4 *)out_ptr;
	for(int i = tid; i < pack_num; i += blockDim.x) {
		float4 weight_package = *(weight_float4 + tid);
		float4 package = *(in_float4 + tid);
		*(out_float4 + i) = make_float4(scale * package.x * weight_package.x,
			scale * package.y * weight_package.y,
			scale * package.z * weight_package.z,
			scale * package.w);
	}
	for (int i = threadIdx.x + pack_num; i < size; i += blockDim.x) {
		out_ptr[i] = weight[i] * scale * in_ptr[i];
	}

}