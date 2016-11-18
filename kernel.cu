//XXX write kernel codes here
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <algorithm> 
#include <string.h>
#include <stdio.h>
__global__
void compute(const int *results, float *avg_que, const int students, const int questions){
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	// 16 * 64 threads for 16 values
	float tmp = 0;
	int q = bx * 16 + tx;


	for (int s = 32 * ty; s < 32 * (ty+1); ++s) {
		tmp += results[s*questions + q];
	}
	tmp = tmp/(float)2048;
	atomicAdd(avg_que + q,tmp);
}
__global__
void compute1(const int *results, float *avg_que, const int students, const int questions){
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
	int by = blockIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	// 64 * 16 threads for 16 values
	int tmp  = 0;

	int offset_y = by * 16 + ty;
	int offset_x = tx;

	for (int offset_x = tx; offset_x < 2048; offset_x+=64)
	{
		tmp += results[(offset_y)*questions + offset_x];
	}
	float x = (float)tmp/(float)2048;
		 atomicAdd(avg_stud + offset_y,x);
		 //atomicAdd(&tmp[i][tx],x);
		 // atomicAdd(avg_stud + offset_y+i,x);
}


__global__
void compute2048(const int *results,  float *avg_stud,float *avg_que ){
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int x_offset = blockIdx.x*(2048/32);
	int y_offset = blockIdx.y*(2048/32);
//	 printf("%d:%d:\n",x_offset,y_offset);

	int __shared__ shres [64][64];
	int __shared__ stud [64];
	int __shared__ que [64];
	que[tx*2] = 0;
	stud[tx*2] = 0;
	que[tx*2+1] = 0;
	stud[tx] = 0;
	#pragma unroll
	for (int i = 0; i < 64 ; ++i)
	{
		shres[tx][i] = results[(x_offset) * 2048 +(y_offset+i)];
	}
	return;
	for (int i = 0; i < 64 ; ++i)
	{
		shres[tx+32][i] = results[(x_offset+32) + 2048 *(y_offset+i)];
	}
	float stud1 = 0;
	float que1 = 0;
	float stud2 = 0;
	float que2 = 0;
	__syncthreads();
	for (int q = 0; q < 64; q++) {
		que1 += shres[q][tx*2];
		que2 += shres[q][tx*2+1];
		stud1 += shres[tx*2][q];
		stud2 += shres[tx*2+1][q];
	}
	atomicAdd(avg_stud + y_offset + tx*2,stud1/(float)2048);
	atomicAdd(avg_que + x_offset + tx*2,que1/(float)2048);
	atomicAdd(avg_stud + y_offset + tx*2+1,stud2/(float)2048);
	atomicAdd(avg_que + x_offset + tx*2+1,que2/(float)2048);
	return;

/**
	float tmp = (float)results[y*2048+x]/(float)2048;
	atomicAdd(avg_stud + y,tmp);
	atomicAdd(avg_que + x,tmp);

	shrs[tx][ty] = results[2048*by*X + bx*X+2048*ty+tx];

	if (by % 2 == 0)
	{
		shrs[tx*2] = results[bx*2048 + tx*2];
		shrs[tx*2+1] = results[bx*2048 + tx*2+1];
		__syncthreads();
		float tmp = 0;
		tmp += shrs[tx*2];
		tmp += shrs[tx*2+1];
		tmp /= (float)2048.0;
		atomicAdd(&out,tmp);
		__syncthreads();
		avg_stud[bx] = out;

	}else{
		shrs[tx*2] = results[tx*2048 + bx];
		shrs[tx*2+1] = results[(tx+1)*2048 + bx];
		__syncthreads();
		float tmp = 0;
		tmp += shrs[tx*2];
		tmp += shrs[tx*2+1];
		tmp /= (float)2048.0;
		atomicAdd(&out,tmp);
		__syncthreads();
		avg_que[bx] = out;

	}
**/

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
	cudaStream_t stream0;
 cudaStreamCreate(&stream0);    // add this line

 const int X = 16;
 dim3 threadsPerBlock(X,X);
 dim3 blck(32,32);
 dim3 thrd(32);

 dim3 nlbl(64,2);
 const int Y = 16;
	dim3 cmpblocks(2048/16); // 128 kernel
	dim3 cmpthreads(16,64);
	dim3 cmpthreads2(64,16);
	// if (students == 2048 && questions == 2048)
	if (false)
	{
		nl<<<nlbl,2048/64>>>(avg_stud,avg_que);
		compute2048<<<blck,thrd>>>(results,avg_stud, avg_que);
		// div<<<students/64,64>>>(avg_stud, avg_que,students,questions);
	}else{
		nl<<<nlbl,2048/64>>>(avg_stud,avg_que);
		// compute1<<<questions/128+1, 128>>>(results,avg_que,students,questions);
		compute<<<128, cmpthreads,0>>>(results,avg_que,students,questions);
		compute2<<<128, cmpthreads2,0,stream0>>>(results,avg_stud,students,questions);
	}
	// int *h_data = (int *)malloc(2048 * sizeof(int));
	// cudaMemcpy(h_data, avg_stud, sizeof(int)*2048, cudaMemcpyDeviceToHost);
	// printf(" %d ", *h_data);
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

