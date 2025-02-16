//
// Created by CSWH on 2024/11/23.
//

#ifndef UTILS_HPP
#define UTILS_HPP
#include <stdio.h>
#include <random>
#include <chrono>

void initialize_data_random(float *ini, int size, float lower = 1.f, float upper = 10.f, bool use_int = false, int seed = 213);

void show_matrix(float *ini, int nrow, int ncol);

void cpu_matmul(float *hA, float *hB, float *hC, int M, int K, int N);

void error_check(float *r1, float* r2, int numOfEle);

#endif //UTILS_HPP
