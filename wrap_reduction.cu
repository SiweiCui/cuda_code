#include <stdio.h>
#include <random>
#include <gtest/gtest.h>

// 当线程数不是warpSize的倍数时，函数失效。
template<const int kWarpSize = warpSize>
__device__ __forceinline__ float wrapReduceSum(float val)
{
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// wrap内的每一个线程存储了正确的结果.
__global__ void testReduce()
{
    int tid = threadIdx.x;
    float val = tid;
    float f_val = wrapReduceSum<32>(val);
    printf("tid: %d, val: %f, fval: %f\n", tid, val, f_val);
}

TEST(test_wrap_reduction, test1)
{
    testReduce<<<1, 64>>>();
    cudaDeviceSynchronize(); // 同步，确保内核执行完成
    cudaError_t err = cudaGetLastError(); // 检查是否有CUDA错误
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}