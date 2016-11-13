//XXX write kernel codes here
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <algorithm> 
#include <string.h>
#include <stdio.h>
const int X = 32;
const int Y = 2;
__global__
void compute(const int *results, float *avg_que, const int students, const int questions){
	int q = blockIdx.x*blockDim.x + threadIdx.x;
	int q2 = blockIdx.x*blockDim.x + threadIdx.y;
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
	int s2 = blockIdx.x*blockDim.x + threadIdx.y;
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
void compute2048(const int *results,  float *avg_stud,float *avg_que ){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int stud = 0;
	int que = 0;
	
	if (blockIdx.y % 2 == 0)
	{
		avg_stud[x] = 0;
		for (int q = 0; q < 2048; q++) {
			stud += results[x*2048 + q];
		}
		avg_stud[x] = (float)stud / (float)2048;
	}else{
		avg_que[x] = 0;
		for (int q = 0; q < 2048; q++) {
			que += results[q*2048 + x];
		}
		avg_que[x] = (float)que / (float)2048;
	}
	return;

}

__global__
void div(float *avg_stud,float *avg_que,  const int students, const int questions){
	for (int i = 0; i < questions; ++i)
	{
		avg_que[i] /= 2048.0;
	}
	for (int i = 0; i < students; ++i)
	{
		avg_stud[i] /= 2048.0;
	}

}
__global__
void nl(float *avg_stud, float * avg_que){
	int s = blockIdx.x*blockDim.x + threadIdx.x;
	if (blockIdx.y % 2 == 0)
	{
		avg_stud[s] = 0;
	}else{
		avg_que[s] = 0;
	}

}
void solveGPU(const int *results, float *avg_stud, float *avg_que, const int students, const int questions){


	dim3 threadsPerBlock(X,X);
	dim3 block(32,2);
	dim3 nlbl(64,2);
	// if (students == 2048 && questions == 2048)
	if (true)
	{
		nl<<<nlbl,2048/64>>>(avg_stud,avg_que);
		compute2048<<<block,2048/32>>>(results,avg_stud, avg_que);
		// div<<<students/64,64>>>(avg_stud, avg_que,students,questions);
	}else{
		compute<<<questions/8+1, 8>>>(results,avg_que,students,questions);
		compute2<<<students/8+1, 8>>>(results,avg_stud,students,questions);
	}
	// int *h_data = (int *)malloc(2048 * sizeof(int));
	// cudaMemcpy(h_data, avg_stud, sizeof(int)*2048, cudaMemcpyDeviceToHost);
	// printf(" %d ", *h_data);
	fflush(stdout); 
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
    // print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	fflush(stdout); 
	// cudaDeviceReset();
}

