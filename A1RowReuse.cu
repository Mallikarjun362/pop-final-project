// CUDA CONVOLUTION KERNEL
__global__ void conv2Dkernel_base(DATA_TYPE *X, DATA_TYPE *Y, const int IMAGE_SIZE)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	// CONV KERNEL 3x3
	c11 = +0.2; c12 = -0.3; c13 = +0.4;
	c21 = +0.5; c22 = +0.6; c23 = +0.7;
	c31 = -0.8; c32 = -0.9; c33 = +0.10;

	if ((i < IMAGE_SIZE - 1) && (j < IMAGE_SIZE - 1) && (i > 0) && (j > 0))
	{
		Y[i * IMAGE_SIZE + j] = c11 * X[(i - 1) * IMAGE_SIZE + (j - 1)] + c21 * X[(i - 1) * IMAGE_SIZE + (j + 0)] + c31 * X[(i - 1) * IMAGE_SIZE + (j + 1)] +
								  c12 * X[(i + 0) * IMAGE_SIZE + (j - 1)] + c22 * X[(i + 0) * IMAGE_SIZE + (j + 0)] + c32 * X[(i + 0) * IMAGE_SIZE + (j + 1)] +
								  c13 * X[(i + 1) * IMAGE_SIZE + (j - 1)] + c23 * X[(i + 1) * IMAGE_SIZE + (j + 0)] + c33 * X[(i + 1) * IMAGE_SIZE + (j + 1)];
	}
}

// CONVOLUTION CUDA
double conv2DCuda_base(DATA_TYPE *IMG_IN_H, DATA_TYPE *IMG_OUT_H, const int IMAGE_SIZE)
{
	// DEFINING NEW GPU MEMORY POINTERS
	DATA_TYPE *IMG_IN_D;
	DATA_TYPE *IMG_OUT_D;
	const int SIZE_IN_BYTES = sizeof(DATA_TYPE) * IMAGE_SIZE * IMAGE_SIZE;

	// ALLOCATING GPU MEMORY
	cudaMalloc((void **)&IMG_IN_D, SIZE_IN_BYTES);
	cudaMalloc((void **)&IMG_OUT_D, SIZE_IN_BYTES);

	// COPYING PROBLEM VARIABLE TO GPU
	cudaMemcpy(IMG_IN_D, IMG_IN_H, SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	// DEFINING DIMENSIONS OF THE KERNEL EXECUTION
    dim3 block(THREAD_BLOCK_DIM_X, THREAD_BLOCK_DIM_Y);
	dim3 grid(
		(size_t) ceil( ((float) IMAGE_SIZE) / ((float) block.x) ), 
		(size_t) ceil( ((float) IMAGE_SIZE) / ((float) block.y) ) 
	);

	// MEASURING THE EXECUTION TIME
	clock_t t;
	t = clock();

	// KERNEL INVOCATION
	conv2Dkernel_base<<<grid, block>>>(IMG_IN_D, IMG_OUT_D, IMAGE_SIZE);
	cudaDeviceSynchronize();

	// MEASURING AND DISPLAYING EXECUTION TIMR
	t = clock() - t;
	double time_taken_in_seconds = ((double)t) / CLOCKS_PER_SEC;

    // CUDA ERROR DETECTION
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess && false){
		printf("CUDA Error - %s : %s\n",cudaGetErrorName(err), cudaGetErrorString(err));
	}

    // DISPLAYING RESULTS
	// printf("CUDA Runtime: %f\n", time_taken_in_seconds);

	// COPYING RESULT VARIABL TO HOST
	cudaMemcpy(IMG_OUT_H, IMG_OUT_D, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

	// FREEING GPU MEMORY
	cudaFree(IMG_IN_D);
	cudaFree(IMG_OUT_D);

	return time_taken_in_seconds;
}