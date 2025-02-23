#include<gtest/gtest.h>

__global__ void sync(int pos, int size, float* s, float* o) {
	for(int t = threadIdx.x; t <= pos; t += blockDim.x) {
		o[t] = 1;
		__syncthreads();
		printf("threadIdx: %d\n", t);
	}
	__syncthreads();
}

TEST(test_sync, test1) {
	int pos = 9;
	int size = 128;

	float* s = (float*)malloc(sizeof(float) * 128);
	for (int i = 0; i < 128; i++) {
		s[i] = (rand()%10)/10.f;
	}

	float* s_gpu, *output_gpu;
	cudaMalloc((void**)&s_gpu, sizeof(float) * 128);
	cudaMalloc((void**)&output_gpu, sizeof(float) * 128);
	cudaMemcpy(s_gpu, s, sizeof(float) * 128, cudaMemcpyHostToDevice);

	sync<<<1, 128>>>(pos, size, s_gpu, output_gpu);

	float* output = (float*)malloc(sizeof(float) * 128);
	cudaMemcpy(output, output_gpu, sizeof(float) * 128, cudaMemcpyDeviceToHost);
}