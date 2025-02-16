//
// Created by CSWH on 2024/11/23.
//
# include "utils.hpp"

void initialize_data_random(float *ini, int size, float lower, float upper, bool use_int, int seed)
{
	// 随机数生成器
	std::mt19937 gen(seed);

	if(! use_int) {
		std::uniform_real_distribution<> dis(lower, upper); // 生成1到3之间的随机数
		for(int i = 0; i < size; i++)
		{
			ini[i] = dis(gen);
		}
	}else {
		std::uniform_int_distribution<> dis((int)lower, (int)upper); // 例如，生成1到3之间的随机整数
		for(int i = 0; i < size; i++)
		{
			ini[i] = dis(gen);
		}
	}

}

void show_matrix(float *ini, int nrow, int ncol)
{
	for(int i = 0; i < nrow; i++)
	{
		for(int j = 0; j < ncol; j++)
		{
			printf("%.4f\t", ini[i*ncol + j]);
		}
		printf("\n");
	}
}

void cpu_matmul(float *hA, float *hB, float *hC, int M, int K, int N)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			float tmp = 0.f;
			for (int k = 0; k < K; k++)
			{
				tmp += hA[i*K + k] * hB[k*N + j];
			}
			hC[i*N + j] = tmp;
		}
	}
}

// void error_check(float *r1, float* r2, int numOfEle)
// {
// 	float error = 0.f;
// 	for (int i = 0; i < numOfEle; i++)
// 	{
// 		error += (r1[i]>r2[i])?(r1[i]-r2[i]):(r2[i]-r1[i]);
// 	}
// 	printf("The error is %.4f\n", error);
// }

void error_check(float *r1, float* r2, int numOfEle)
{
	float error = 0.f;
	for (int i = 0; i < numOfEle; i++) {
		error = std::max(error, std::abs(r1[i] - r2[i]));
	}
	if(error > 1e-3) {
		printf("biggest error is %f\n", error);
	}else {
		printf("No significant error\n");
	}
}