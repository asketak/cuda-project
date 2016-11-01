//XXX write kernel codes here
__global__
void compute(const int *results, float *avg_stud, float *avg_que, const int students, const int questions){
	int q = blockIdx.x*blockDim.x + threadIdx.x;
	if (q > questions)
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
void compute2(const int *results, float *avg_stud, float *avg_que, const int students, const int questions){
	int s = blockIdx.x*blockDim.x + threadIdx.x;
	if (s > students)
	{
		return;
	}
	int stud = 0;
	for (int q = 0; q < questions; q++) {
		stud += results[s*questions + q];
	}
	avg_stud[s] = (float)stud / (float)questions;
}

void solveGPU(const int *results, float *avg_stud, float *avg_que, const int students, const int questions){

	compute<<<questions/32+1, 32>>>(results,avg_stud,avg_que,students,questions);
	compute2<<<students/32+1, 32>>>(results,avg_stud,avg_que,students,questions);
}

