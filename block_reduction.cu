#include<stdio.h>
#include<gtest/gtest.h>


// 当线程数不是warpSize的倍数时，函数失效。
template<const int kWarpSize = warpSize>
__device__ __forceinline__ float warp_reduce_sum(float val)
{
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}


template<const int NUM_THREADS=128, const int WARP_SIZE = 32>
// 传进来的参数val，是每个线程累计的局部和
__device__ __forceinline__ float block_reduce_sum(float val) {
    // always <= 32 warps per block (limited by 1024 threads per block)
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS]; // 共享内存数据
    
    //Wrap内部进行一次规约
    val = warp_reduce_sum<WARP_SIZE>(val);
    // 收集规约结果
    shared[warp] = val;
    __syncthreads();
    
    // 把每个wrap内的前NUM_WRAPS个线程的val改成共享内存的val, 其余的设为0
    val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    // 在WRAP内收集, 每个WRAP的第一个线程就存储了正确的结果.
    val = warp_reduce_sum<WARP_SIZE>(val);
    return val;
}

__global__ void testReduceBlock()
{
    int tid = threadIdx.x;
    float val = tid;
    float f_val = block_reduce_sum<64, 32>(val);
    printf("tid: %d, val: %f, fval: %f\n", tid, val, f_val);
}

TEST(test_block_reduction, test1){
    testReduceBlock<<<1, 64>>>();
    cudaDeviceSynchronize(); // 同步，确保内核执行完成
    cudaError_t err = cudaGetLastError(); // 检查是否有CUDA错误
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}