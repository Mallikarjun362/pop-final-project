// -------------------- STANDARD LIBRARIES ------------------------
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
#define SMALL_FLOAT_VAL 0.00000001f
#define THREAD_BLOCK_DIM_X 32
#define THREAD_BLOCK_DIM_Y 32
typedef float DATA_TYPE;

// ----------------------- CUSTOME LIBRARIES -----------------------
// ALGORITHMS
#include "utils.cu"
// #include "A1RowReuse.cu"
// #include "A2ColumnReuse.cu"
// #include "A3RowReuseModified.cu"
// #include "A4ColumnReuseModified.cu"
// STANDARD IMPLEMENTATIONS
#include "stdCudnn_winograd.cu"
#include "stdHost.cu"
// UTILITIES

// ----------------------- MAIN FUNCTION -------------------------
int main(int argc, char *argv[])
{   
	// PART 1 : VARIABLE DECLERATION
	// DYNAMIC PROBLEM SIZE
	int FILTER_SIZE = 3;
	int IMAGE_SIZE = atoi(argv[1]);
	int IMG_SIZE_IN_BYTES = sizeof(DATA_TYPE) * IMAGE_SIZE * IMAGE_SIZE;
	int FILTER_SIZE_IN_BYTES = sizeof(DATA_TYPE) * FILTER_SIZE * FILTER_SIZE;

	// CPU-HOST VARIABLES
	DATA_TYPE *IMG_IN, *IMG_OUT, *KERNEL_IN;
	
	IMG_IN = (DATA_TYPE *)malloc(IMG_SIZE_IN_BYTES);
	IMG_OUT = (DATA_TYPE *)malloc(IMG_SIZE_IN_BYTES);
	KERNEL_IN = (DATA_TYPE *)malloc(FILTER_SIZE_IN_BYTES);

	initializeImage(IMG_IN, IMAGE_SIZE, IMAGE_SIZE);
	initializeImage(KERNEL_IN, FILTER_SIZE, FILTER_SIZE);

	// PART 2 : COMPUTATION
    // COMPUTING RESULTS
	double t_h = myStdHost(IMG_IN, IMG_OUT, KERNEL_IN, IMAGE_SIZE, FILTER_SIZE);
	double t_cudnn_winograd = myStdCudnn_winograd(IMG_IN, IMG_OUT, KERNEL_IN, IMAGE_SIZE, FILTER_SIZE);

	// double t_a1 = A1(IMG_IN, IMG_OUT, KERNEL_IN, IMAGE_SIZE, FILTER_SIZE);
	// double t_a2 = A2(IMG_IN, IMG_OUT, KERNEL_IN, IMAGE_SIZE, FILTER_SIZE);
	// double t_a3 = A3(IMG_IN, IMG_OUT, KERNEL_IN, IMAGE_SIZE, FILTER_SIZE);
	// double t_a4 = A4(IMG_IN, IMG_OUT, KERNEL_IN, IMAGE_SIZE, FILTER_SIZE);

	printf("Host : %f\n",t_h);
	printf("Winograd : %f\n",t_cudnn_winograd);
	
	// PART 3 : DISPLAY EXECUTION TIME
	printf("END");
	free(IMG_IN);
	return 0;
}