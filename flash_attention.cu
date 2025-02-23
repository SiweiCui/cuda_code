#include<gtest/gtest.h>


//一个block处理一个head
__global__ void flash_attention_kernel(int32_t pos, int32_t seq_len, float* query,
												float* score_ptr, float* output, float* key_cache,
												float* value_cache, int32_t kv_dim, int32_t kv_mul,
												int32_t head_num, int32_t head_size,
												int32_t layer_offset) {
	int head_id = blockIdx.x;
	if (head_id >= head_num) {return;}
	int kv_head_offset = (head_id / kv_mul) * head_size;
	int q_head_offset = head_id * head_size;
	int tid = threadIdx.x;

	float scale = 1.f / sqrtf(head_size);

	extern __shared__ float shared_mem[];
	float* shared_q = shared_mem;
	float* shared_k_t = shared_q + head_size;
	float* shared_v_t = shared_k_t + head_size;
	float* shared_o_t = shared_v_t + head_size;
	float* m = shared_o_t + head_size;
	float* l = m+1;

	// 初始化共享变量
	*m = -FLT_MIN;
	*l = 0;
	for(int h = 0; h < head_size; h++) {
		shared_o_t[h] = 0.0f;
	}
	// 并行load query, output
	float* q_t = query + q_head_offset;
	for(int i = tid; i < head_size; i += blockDim.x) {
		shared_q[i] = q_t[i];
	}
	__syncthreads();

	for (int t = 0; t <= pos; t ++) { // 此处不做并行化
		// 并行load k_t, v_t
		float* k_t = key_cache + layer_offset + t * kv_dim + kv_head_offset;
		float* v_t = value_cache + layer_offset + t * kv_dim + kv_head_offset;
		for(int i = tid; i < head_size; i += blockDim.x) {
			shared_k_t[i] = k_t[i];
			shared_v_t[i] = v_t[i];
		}
		__syncthreads();

		// 内积<q, k_t>
		float s_t = 0.f;
		for (int j = 0; j < head_size; j++) {
			s_t += shared_q[j] * shared_k_t[j];
		}
		s_t *= scale; // block中每个线程都有一份, 而且一样大

		// 更新m和l
		float m_old = *m;
		if(tid == 0) { // 赋值, 让多个线程做也没关系, 这里让一个线程去做
			*m = max(m_old, s_t);
			*l = expf(m_old - *m) * *l + expf(s_t - *m);
		}
		__syncthreads();

		// 动员所有线程更新O
		float p_t = expf(s_t - *m);
		for(int i = tid; i < head_size; i += blockDim.x) {
			shared_o_t[i] = expf(m_old - *m) * shared_o_t[i] + p_t * shared_v_t[i]; // 线程之间不会有写冲突
		}
		__syncthreads();
	}

	// 用最终的l更新
	float* o = output + q_head_offset;
	for(int i = tid; i < head_size; i += blockDim.x) {
		o[i] = shared_o_t[i] / *l;
	}

}

TEST(test_flashattn, test1){
    int pos = 9; // 从0开始计数
    int layer_idx = 9; // 从0开始计数
    int layer_num = 22;
    int seq_len = 2048;
    int head_num = 32;
    const int head_size = 64;
    int kv_head_num = 4;
    int kv_mul = 8; // 一个query头对应多少kv头
    int kv_dim = 256; // KV Cache中seq之间的step

    // 所需数据
    float* q = (float*)malloc(sizeof(float) * head_num * head_size);
    float* k_cache = (float*)malloc(sizeof(float) * layer_num * seq_len * kv_dim);
    float* v_cache = (float*)malloc(sizeof(float) * layer_num * seq_len * kv_dim);
    for (int i = 0; i < head_num*head_size; i++) {
        q[i] = (rand()%10)/10.f;
    }
    // 补充0~9的k和v
    for (int i = 0; i < (pos+1)*kv_dim; i++) {
        k_cache[layer_idx*seq_len*kv_dim + i] = (rand()%10)/10.f;
        v_cache[layer_idx*seq_len*kv_dim + i] = (rand()%10)/10.f;
    }

    // 创建空间以及数据转移
    float* score_gpu, *output_gpu;
    float* q_gpu, *k_cache_gpu, *v_cache_gpu;
    cudaMalloc(&score_gpu, sizeof(float) * head_num * seq_len);
    cudaMalloc(&output_gpu, sizeof(float) * head_num * head_size);
    cudaMalloc(&q_gpu, sizeof(float) * head_num * head_size);
    cudaMalloc(&k_cache_gpu, sizeof(float) * layer_num * seq_len * kv_dim);
    cudaMalloc(&v_cache_gpu, sizeof(float) * layer_num * seq_len * kv_dim);

    cudaMemcpy(q_gpu, q, sizeof(float) * head_num * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(k_cache_gpu, k_cache, sizeof(float) * layer_num * seq_len * kv_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(v_cache_gpu, v_cache, sizeof(float) * layer_num * seq_len * kv_dim, cudaMemcpyHostToDevice);


    flash_attention_kernel<<<head_num, 128, (head_size * 4 + 2)*sizeof(float)>>>(pos, seq_len, q_gpu, score_gpu, output_gpu,
        k_cache_gpu, v_cache_gpu, kv_dim, kv_mul, head_num, head_size, layer_idx * seq_len * kv_dim);

    cudaError_t err = cudaGetLastError(); // 检查是否有CUDA错误
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    float* output_gpu2cpu = (float*)malloc(sizeof(float) * head_num * head_size);
    cudaMemcpy(output_gpu2cpu, output_gpu, sizeof(float) * head_num * head_size, cudaMemcpyDeviceToHost);
    printf("flash attn output:\n");
    for(int i = 0; i < 100; i++) {
        printf("%f\t", i, output_gpu2cpu[i]);
    }
}