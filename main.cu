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
#include "utils.cu"

// 1. ALGORITHMS
// #include "A1RowReuse.cu"
#include "A2ColumnReuse.cu"
// #include "A3RowReuseModified.cu"
// #include "A4ColumnReuseModified.cu"

// 2. STANDARD IMPLEMENTATIONS
// #include "stdCudnn_winograd.cu"
#include "stdCudnn.cu"
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
	// DATA_TYPE *IMG_IN, 
  DATA_TYPE *IMG_OUT;
  	// DATA_TYPE *FILTER_IN;
  DATA_TYPE	FILTER_IN[] = {1,1,1, 1,1,1, 1,1,1};
  DATA_TYPE	IMG_IN[] = {1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1};
	
	// IMG_IN = (DATA_TYPE *)malloc(IMG_SIZE_IN_BYTES);
	IMG_OUT = (DATA_TYPE *)malloc(IMG_SIZE_IN_BYTES);
	// FILTER_IN = (DATA_TYPE *)malloc(FILTER_SIZE_IN_BYTES);

	// initializeImage(IMG_IN, IMAGE_SIZE);
	// initializeImage(FILTER_IN, FILTER_SIZE);

	// ------------------------ PART 2 : COMPUTATION ------------------------
    // COMPUTING RESULTS
	double t_h = myStdHost(IMG_IN, IMG_OUT, FILTER_IN, IMAGE_SIZE, FILTER_SIZE);
	double t_cudnn = myStdCudnn(IMG_IN, IMG_OUT, FILTER_IN, IMAGE_SIZE, FILTER_SIZE);

	// double t_a1 = A1(IMG_IN, IMG_OUT, FILTER_IN, IMAGE_SIZE, FILTER_SIZE);
	double t_a2 = A2ColumnReuse(IMG_IN, IMG_OUT, FILTER_IN, IMAGE_SIZE, FILTER_SIZE);

	// ------------------------------------------------------------------------ 
	
	printf("Host : %f\n",t_h);
	printf("Cudnn : %f\n",t_cudnn);
	printf("Column Reuse : %f\n",t_a2);
	
	// PART 3 : DISPLAY EXECUTION TIME
	// free(IMG_IN);
	return 0;
}