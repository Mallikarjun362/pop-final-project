// -------------------- STANDARD LIBRARIES ------------------------
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>

// ----------------------- CUSTOME LIBRARIES -----------------------
// ALGORITHMS
#include "A1RowReuse.cu"
#include "A2ColumnReuse.cu"
#include "A3RowReuseModified.cu"
#include "A4ColumnReuseModified.cu"
// STANDARD IMPLEMENTATIONS
#include "stdCudnn.cu"
#include "stdHost.cu"
// UTILITIES
#include "utils.cu"

// ----------------------- MAIN FUNCTION -------------------------
int main(int argc, char *argv[])
{   
	// PART 1 : VARIABLE DECLERATION
	// DYNAMIC PROBLEM SIZE
	int KERNEL_SIZE = 3;
	int IMAGE_SIZE = atoi(argv[1]);
	int IMG_SIZE_IN_BYTES = sizeof(DATA_TYPE) * IMAGE_SIZE * IMAGE_SIZE;
	int KERNEL_SIZE_IN_BYTES = sizeof(DATA_TYPE) * KERNEL_SIZE * KERNEL_SIZE;

	// CPU-HOST VARIABLES
	DATA_TYPE *IMG_IN;	                                   // PROBLEM VARIABLES
	IMG_IN = (DATA_TYPE *)malloc(IMG_SIZE_IN_BYTES);       // MEMORY ALLOCATION
	IMG_OUT = (DATA_TYPE *)malloc(IMG_SIZE_OUT_BYTES);       // MEMORY ALLOCATION
	KERNEL_IN = (DATA_TYPE *)malloc(KERNEL_SIZE_IN_BYTES);
	initializeImage(IMG_IN, IMAGE_SIZE);                   // VARIABLES INITIALIZATION
	initializeImage(KERNEL_IN, KERNEL_SIZE);

	// PART 2 : COMPUTATION
    // COMPUTING RESULTS
	double t_h = myStdHost(IMG_IN, IMG_OUT, KERNEL_IN, IMAGE_SIZE, KERNEL_SIZE);
	// double t_cudnn = myStdCudnn(IMG_IN, IMG_OUT, KERNEL_IN, IMAGE_SIZE, KERNEL_SIZE);

	// double t_a1 = A1(IMG_IN, IMG_OUT, KERNEL_IN, IMAGE_SIZE, KERNEL_SIZE);
	// double t_a2 = A2(IMG_IN, IMG_OUT, KERNEL_IN, IMAGE_SIZE, KERNEL_SIZE);
	// double t_a3 = A3(IMG_IN, IMG_OUT, KERNEL_IN, IMAGE_SIZE, KERNEL_SIZE);
	// double t_a4 = A4(IMG_IN, IMG_OUT, KERNEL_IN, IMAGE_SIZE, KERNEL_SIZE);
	
	
	// PART 3 : DISPLAY EXECUTION TIME
	printf("END")
	free(IMG_IN);
	return 0;
}