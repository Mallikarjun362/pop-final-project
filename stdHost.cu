// CONVOLUTION CUDA
// double conv2DHost(DATA_TYPE *X, DATA_TYPE *Y, const int NI)
double conv2DHost(IMG_IN, IMG_OUT, KERNEL_IN, IMAGE_SIZE, KERNEL_SIZE)
{
    // MEASURING THE EXECUTION TIME
	clock_t t;
	t = clock();

	int i, j;
	const int NJ = NI;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)]; c12 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)]; c13 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)];
	c21 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)]; c22 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)]; c23 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)];
	c31 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)]; c32 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)]; c33 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)];

	for (i = 1; i < KERNEL_SIZE - 1; ++i)
	{
		for (j = 1; j < KERNEL_SIZE - 1; ++j)
		{
			IMG_OUT[i * NJ + j] = c11 * X[(i - 1) * NJ + (j - 1)] + c12 * X[(i + 0) * NJ + (j - 1)] +  c13 * X[(i + 1) * NJ + (j - 1)] + 
							c21 * X[(i - 1) * NJ + (j + 0)] + c22 * X[(i + 0) * NJ + (j + 0)] +  c23 * X[(i + 1) * NJ + (j + 0)] + 
							c31 * X[(i - 1) * NJ + (j + 1)] + c32 * X[(i + 0) * NJ + (j + 1)] +  c33 * X[(i + 1) * NJ + (j + 1)];
		}
	}
    // MEASURING AND DISPLAYING EXECUTION TIMR
	t = clock() - t;
	double time_taken_in_seconds = ((double)t) / CLOCKS_PER_SEC;
	// printf("HOST Runtime: %f\n", time_taken_in_seconds);
	return time_taken_in_seconds;
}