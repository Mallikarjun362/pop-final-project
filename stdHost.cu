double myStdHost(float* X, float* IMG_OUT, float* KERNEL_IN, int IMAGE_SIZE, int KERNEL_SIZE)
{
    // printf("HOST START\n");
    // MEASURING THE EXECUTION TIME

	int i, j;
    const int NI = IMAGE_SIZE;
	const int NJ = NI;
	float c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)]; c12 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)]; c13 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)];
	c21 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)]; c22 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)]; c23 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)];
	c31 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)]; c32 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)]; c33 = KERNEL_IN[(1-1) * KERNEL_SIZE + (1-1)];

	clock_t t;
	t = clock();
    
	for (i = 1; i < NI - 1; ++i)
	{
		for (j = 1; j < NJ - 1; ++j)
		{
			IMG_OUT[i * NJ + j] = c11 * X[(i - 1) * NJ + (j - 1)] + c12 * X[(i + 0) * NJ + (j - 1)] +  c13 * X[(i + 1) * NJ + (j - 1)] + 
							c21 * X[(i - 1) * NJ + (j + 0)] + c22 * X[(i + 0) * NJ + (j + 0)] +  c23 * X[(i + 1) * NJ + (j + 0)] + 
							c31 * X[(i - 1) * NJ + (j + 1)] + c32 * X[(i + 0) * NJ + (j + 1)] +  c33 * X[(i + 1) * NJ + (j + 1)];
		}
	}
    // MEASURING AND DISPLAYING EXECUTION TIMR
	t = clock() - t;
	double time_in_milliseconds = (((double)t) / CLOCKS_PER_SEC) * 1000;
	// printf("HOST Runtime: %f\n", time_in_milliseconds);
	return time_in_milliseconds;
}