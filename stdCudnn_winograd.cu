#include "utils.cu"

#define TILE_SIZE 2

__device__ float winograd_filter_transform(float* filter, int i, int j) {
    const float G[3][3] = {{2.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, -1.0f, 1.0f}};
    const float Bt[3][3] = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, -1.0f}, {0.0f, 0.5f, 0.5f}};

    float tmp = 0.0f;
    for (int k = 0; k < 3; ++k) {
        tmp += G[i][0] * G[j][k] * filter[k];
        tmp += G[i][1] * G[j][k] * filter[k + 3];
        tmp += G[i][2] * G[j][k] * filter[k + 6];
    }
    return tmp;
}

__device__ float winograd_input_transform(float* input, int i, int j, int width, int height) {
    const float B[3][3] = {{1.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}};

    float tmp = 0.0f;
    for (int k = 0; k < 3; ++k) {
        int x = i * 2 + k % 2;
        int y = j * 2 + k / 2;
        if (x >= 0 && x < width && y >= 0 && y < height) {
            tmp += B[i][k] * B[j][k] * input[y * width + x];
        }
    }
    return tmp;
}

__device__ float winograd_output_transform(float* output, int i, int j) {
    const float A[3][3] = {{1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, -1.0f, 1.0f}};

    float tmp = 0.0f;
    for (int k = 0; k < 3; ++k) {
        tmp += A[i][k] * A[j][k] * output[j * 2 + k / 2];
    }
    return tmp;
}

__global__ void winograd_convolution(float* input, float* filter, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float transformed_filter[9];
        float transformed_input[9];

        // Transform filter
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                transformed_filter[i * 3 + j] = winograd_filter_transform(filter, i, j);
            }
        }

        // Transform input
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                transformed_input[i * 3 + j] = winograd_input_transform(input, i, j, width, height);
            }
        }

        // Perform Winograd domain operations
        float tmp[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                tmp[i * 3 + j] = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    tmp[i * 3 + j] += transformed_filter[i * 3 + k] * transformed_input[k * 3 + j];
                }
            }
        }

        // Transform output
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                output[(y * width + x) * 4 + i * 2 + j] = winograd_output_transform(tmp, i, j);
            }
        }
    }
}

double myStdCudnn_winograd(float* IMG_IN, float* IMG_OUT, float* FILTER_IN, const int IMAGE_SIZE, const int FILTER_SIZE) {
    
    // Define input image and filter
    int MEM_SIZE_IMG = IMAGE_SIZE * IMAGE_SIZE * sizeof(float);
    int MEM_SIZE_FILTER = FILTER_SIZE * FILTER_SIZE * sizeof(float);
    
    float* h_output_current = (float *) malloc(MEM_SIZE_IMG);

    // Allocate memory on device
    float *d_input, *d_filter, *d_output;

    cudaMalloc((void **) &d_input, MEM_SIZE_IMG);
    cudaMalloc((void **) &d_output, MEM_SIZE_IMG);
    cudaMalloc((void **) &d_filter, MEM_SIZE_FILTER);

    // Copy data from host to device
    cudaMemcpy(d_input, IMG_IN, MEM_SIZE_IMG, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, FILTER_IN, MEM_SIZE_FILTER, cudaMemcpyHostToDevice);

    // Launch CUDA kernel for Winograd convolution
    dim3 blockDim(2, 2);
    dim3 gridDim((IMAGE_SIZE + blockDim.x - 1) / blockDim.x, (IMAGE_SIZE + blockDim.y - 1) / blockDim.y);

    clock_t t;
	  t = clock();
    
    winograd_convolution<<<gridDim, blockDim>>>(d_input, d_filter, d_output, IMAGE_SIZE, IMAGE_SIZE);
    cudaDeviceSynchronize();

    t = clock() - t;
	  double time_taken_in_seconds = ((double)t) / CLOCKS_PER_SEC;

    // Copy output data from device to host
    cudaMemcpy(h_output_current, d_output, MEM_SIZE_IMG, cudaMemcpyDeviceToHost);

    resultAccuracy(IMG_OUT, h_output_current, IMAGE_SIZE);

    // Free memory
    free(h_output_current);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    return 0;
}
