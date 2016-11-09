//XXX write kernel codes here
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <algorithm> 
__global__
void compute(const int *results, float *avg_que, const int students, const int questions){
	int q = blockIdx.x*blockDim.x + threadIdx.x;
	if (q >= questions)
	{
		return;
	}
	int que = 0;
	for (int s = 0; s < students; s++) {
		que += results[s*questions + q];
	}
	avg_que[q] = (float)que / (float)students;
}

__global__
void compute2(const int *results, float *avg_stud,  const int students, const int questions){
	int s = blockIdx.x*blockDim.x + threadIdx.x;
	if (s >= students)
	{
		return;
	}
	int stud = 0;
	for (int q = 0; q < questions; q++) {
		stud += results[s*questions + q];
	}
	avg_stud[s] = (float)stud / (float)questions;
}

__global__
void compute2048(const int *results, float *avg_que, const int students, const int questions){
	int q = blockIdx.x*blockDim.x + threadIdx.x;
	int que = 0;
	for (int s = 0; s < students; s++) {
		que += results[s*questions + q];
	}
	avg_que[q] = (float)que / (float)students;
}

__global__
void compute20482(const int *results, float *avg_stud,  const int students, const int questions){
	int s = blockIdx.x*blockDim.x + threadIdx.x;
	int stud = 0;
	for (int q = 0; q < questions; q++) {
		stud += results[s*questions + q];
	}
	avg_stud[s] = (float)stud / (float)questions;
}

void solveGPU(const int *results, float *avg_stud, float *avg_que, const int students, const int questions){


	if (students == 2048 && questions == 2048)
	{
		compute2048<<<questions/32, 32>>>(results,avg_que,students,questions);
		compute20482<<<students/32, 32>>>(results,avg_stud,students,questions);
	}else{
		compute<<<questions/1024, 1024>>>(results,avg_que,students,questions);
		compute2<<<students/1024, 1024>>>(results,avg_stud,students,questions);
	}
	}

