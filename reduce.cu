//
// Created by CSWH on 2024/11/30.
//
#include <gtest/gtest.h>
#include <chrono>

void cpu_reduce(float *in, float &out, int size) {
	float sum = 0;
	for (int i = 0; i < size; i++) {
		sum += in[i];
	}
	out = sum;
}

// Version 1: 相邻规约
template<const int BLOCK_DIM> // 128
__global__ void reduce(float *in_gpu, float *out_gpu, int size) {
	int tid = threadIdx.x;
	__shared__ float block_data[BLOCK_DIM];

	// 把所有数值收集到一个block内 (本次测试中一共就一个block)
	for (int j = tid; j < size; j += blockDim.x) {
		block_data[tid] += in_gpu[j];
	}
	// block内的reduce
	for (int s = 1; s < blockDim.x; s*=2) {
		if (tid % (s*2) == 0) {
			block_data[tid] += block_data[tid + s];
		}
		// 第一次循环: 0<-1, 2<-3, 4<-5, ...
		// 第二次循环: 0<-2, 4<-6, ...
		// ...
		// 第七次循环: 0<-64
		__syncthreads();
	}
	if (tid == 0) {
		*out_gpu = block_data[tid];
	}
}

// Version 2: 解决线程束分化
/*
 *在第一次循环中, 有一半的线程在工作, 但是这一半的线程均匀分布在每个wrap中(0和1线程属于同一个wrap),
 *这导致了严重的线程束分化.
 *我们的思路是, 反正只有一半线程在工作, 我们就让一半wrap工作, 一半wrap不工作. 改变共享内存的访问和线程编号的关系，
 *一共128个线程, 4个wrap. 0-31, 32-63, 64-95, 96-127.
 *第一次循环中接收数据的是block_data的所有模2为0点, 由前两个wrap完成, 其他wrap不工作.
 *第二次循环中接收数据的是block_data的所有模4为0点, 由第一个wrap完成, 其他wrap不工作.
 *第三次循环中接收数据的是block_data的所有模8为0点, 由第一个wrap的一半完成, 其他wrap不工作. 此时才出现了线程束的分化.
 */
template<const int BLOCK_DIM>
__global__ void reduce2(float *in_gpu, float *out_gpu, int size) {
	int tid = threadIdx.x;
	__shared__ float block_data[BLOCK_DIM];

	// 把所有数值收集到一个block内
	for (int j = tid; j < size; j += blockDim.x) {
		block_data[tid] += in_gpu[j];
	}
	// block内的reduce
	for (int s = 1; s < blockDim.x; s*=2) {
		int index = 2*s*tid;
		if (index < blockDim.x) {
			block_data[index] += block_data[index + s];
		}
		// 第一次循环: 0<-1 (0), 2<-3 (1), 4<-5 (2), ... 注意, 共享内存编号与线程号不再对应, 括号内是完成工作的线程编号.
		// 第二次循环: 0<-2 (0), 4<-6 (1), ...
		// ...
		// 第七次循环: 0<-64 (0)
		__syncthreads();
	}
	if (tid == 0) {
		*out_gpu = block_data[tid];
	}
}

// Version 3: 解决bank conflict
/*
 *在Version2中, 线程0访问了0号和1号bank(索引分别为0和1), 线程16也访问了0号和1号bank(索引分别为32和33).
 *他们两个是在同一个wrap中的线程, 这会导致冲突.
 *解决方法是, 让线程访问同一个bank. 也就是先在每个bank中竖着做规约.
 *当竖着的任务完成了再在"一行"之内做规约, 这时候不会再产生bank conflict.
 */
template<const int BLOCK_DIM>
__global__ void reduce3(float *in_gpu, float *out_gpu, int size) {
	int tid = threadIdx.x;
	__shared__ float block_data[BLOCK_DIM];

	// 把所有数值收集到一个block内
	for (int j = tid; j < size; j += blockDim.x) {
		block_data[tid] += in_gpu[j];
	}
	// block内的reduce
	for (int s = blockDim.x / 2; s > 0; s/=2) {
		if (tid < s) {
			block_data[tid] += block_data[tid + s];
		}
		// 第一次循环: 0<-64 (0), 1<-65 (1), 2<-66 (2) , ...
		// 第二次循环: 0<-32 (0), 1<-33 (1), ...
		// 第三次循环: 0<-16 (0), 1<-17 (1), ...
		// ...
		// 第七次循环: 0<-1 (0)
		__syncthreads();
	}
	if (tid == 0) {
		*out_gpu = block_data[tid];
	}
}

// Version 4: shuffle
/*
 *上述方法还有瓶颈, 就在于每次线程的同步上. 由于wrap内的线程是完全并行的, 所以最后一步不需要同步. 可以考虑将其展开.
 *但是在这里我们直接使用洗牌指令实现wrap规约, 进而实现block内的规约.
 */
template<const int WRAP_SIZE>
__device__ float wrap_reduce(float val) {
	#pragma unroll
	for (int j = WRAP_SIZE/2; j > 0; j >>= 1) {
		val += __shfl_down_sync(0xffffffff, val, j);
	}
	return val;
}
template<const int BLOCK_DIM, const int WRAP_SIZE>
__global__ void reduce4(float *in_gpu, float *out_gpu, int size) {
	int tid = threadIdx.x;
	const int wrap_num = BLOCK_DIM / WRAP_SIZE;
	int tid_in_wrap = tid % WRAP_SIZE;
	int wrap_id = tid / WRAP_SIZE;

	__shared__ float block_wrap_data[wrap_num];

	// 把数值收集到block的线程中
	float val = 0.f;
	for (int j = tid; j < size; j += blockDim.x) {
		val += in_gpu[j];
	}

	// 在wrap内收集一次
	val = wrap_reduce<WRAP_SIZE>(val);
	// 写入每个wrap的和
	if (tid_in_wrap == 0) {
		block_wrap_data[wrap_id] = val;
	}
	__syncthreads();

	// 收集每个wrap的和
	val = (tid_in_wrap < wrap_num) ? block_wrap_data[tid_in_wrap] : 0.0f;
	val = wrap_reduce<WRAP_SIZE>(val);
	if (tid_in_wrap == 0) {
		*out_gpu = val;
	}
}

// Verison 5: 使用cub库进行BlockReduction: 略

// 其他: 使用float4, 一个线程负责一个float4 pack的计算: 略. rmsnorm包括了.


TEST(tset_reduce, test1) {
	int size = 1024;
	float* in = (float*)malloc(size * sizeof(float));
	for (int i = 0; i < size; i++) {
		in[i] = (float)(i%3);
	}
	float out;

	auto start = std::chrono::high_resolution_clock::now();
	cpu_reduce(in, out, size);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	printf("CPU结果: %f\n", out);
	printf("CPU计算时间: %.2f 毫秒\n", float(duration.count()));



	cudaEvent_t cudaStart, cudaStop, kernelStart, kernelStop;
	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);
	cudaEventCreate(&kernelStart);
	cudaEventCreate(&kernelStop);
	float gpuTime, kernelTime;

	float *in_gpu, *out_gpu;
	cudaEventRecord(cudaStart, 0);
	cudaMalloc(&in_gpu, size * sizeof(float));
	cudaMalloc(&out_gpu, sizeof(float));
	cudaMemcpy(in_gpu, in, sizeof(float) * size, cudaMemcpyHostToDevice);

	cudaEventRecord(kernelStart, 0);
	const int BLOCK_DIM = 128;
	const int WRAP_SIZE = 32;
	//reduce<BLOCK_DIM><<<1, 128>>>(in_gpu, out_gpu, size);
	reduce4<BLOCK_DIM, WRAP_SIZE><<<1, 128>>>(in_gpu, out_gpu, size);
	cudaEventRecord(kernelStop, 0);

	float gpu_result;
	cudaMemcpy(&gpu_result, out_gpu, sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(cudaStop, 0);
	cudaEventElapsedTime(&gpuTime, cudaStart, cudaStop);
	cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop);
	printf("GPU结果: %f\n", gpu_result);
	printf("GPU时间: %f 毫秒\n", gpuTime);
	printf("核函数时间: %f 毫秒\n", kernelTime);

	cudaError_t err = cudaGetLastError(); // 检查是否有CUDA错误
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
	}

	free(in);
	cudaFree(in_gpu);
	cudaFree(out_gpu);
	cudaEventDestroy(cudaStart);
	cudaEventDestroy(cudaStop);
	cudaEventDestroy(kernelStart);
	cudaEventDestroy(kernelStop);

}
