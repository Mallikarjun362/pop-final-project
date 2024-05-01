#ifndef MY_UTILS
#define MY_UTILS

// --------------------- CONFIGURATION ---------------------
// #define PERCENT_DIFF_ERROR_THRESHOLD 0.05
// #define SMALL_FLOAT_VAL 0.00000001f
// #define THREAD_BLOCK_DIM_X 32
// #define THREAD_BLOCK_DIM_Y 32
// typedef float DATA_TYPE;

// -------------------- UTIL CODE BLOCKS --------------------
void initializeImage(float *img, int size) {
	int size_squared = size * size;
    for (int i = 0; i < size_squared; ++i) {
        img[i] = (float)rand() / RAND_MAX;
    }
}

// ABSOLUTE VALUE OF THE INPUT
float absoluteValue(float a)
{
	if(a < 0)
	{
		return (a * -1);
	}
   	else
	{ 
		return a;
	}
}

// PERCENTAGE DIFFERENCE
float percentageDifference(double val1, double val2)
{
	if ((absoluteValue(val1) < 0.01) && (absoluteValue(val2) < 0.01))
	{
		return 0.0f;
	}
	else
	{
    	return 100.0f * (absoluteValue(absoluteValue(val1 - val2) / absoluteValue(val1 + SMALL_FLOAT_VAL)));
	}
} 

// COMPARING RESULTS
void resultAccuracy(DATA_TYPE *RES_H, DATA_TYPE *RES_CUDA, const int IMAGE_SIZE)
{
	int i, j, fail;
	fail = 0;

	for (i = 1; i < (IMAGE_SIZE - 1); i++)
	{
		for (j = 1; j < (IMAGE_SIZE - 1); j++)
		{
			if (percentageDifference(RES_H[i * IMAGE_SIZE + j], RES_CUDA[i * IMAGE_SIZE + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	if (fail != 0)
		printf("Failed instances: %d\n", fail);
}
#endif