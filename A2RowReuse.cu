%%cuda
#include <cuda.h>   // For CUDA Runtime API
#include <stdio.h>  // For standard input/output
#include <stdlib.h> // For memory allocation (malloc, free)
#include <assert.h> // For error checking (assert)

#define IH 1000000
#define IW 1000000
#define IC 1

#define FH 3
#define FW 3

#define OC 3

// Assuming data types for input, filter, and output
typedef float DataType;

// Function to allocate and initialize host and device memory
void allocateAndInitialize(
    DataType** d_input, DataType** d_filter, DataType** d_output,
    DataType* h_input, DataType* h_filter, DataType* h_output
) {
    size_t input_size = IH * IW * IC * sizeof(DataType);
    size_t filter_size = FH * FW * IC * OC * sizeof(DataType);
    size_t output_size = IH * IW * OC * sizeof(DataType);

    // Allocate host memory
    h_input = (DataType*)malloc(input_size);
    h_filter = (DataType*)malloc(filter_size);
    h_output = (DataType*)malloc(output_size);

    // Initialize h_input and h_filter with example values (replace with your data)
    for (int i = 0; i < IH * IW * IC; ++i) {
        h_input[i] = i;
    }
    for (int i = 0; i < FH * FW * IC * OC; ++i) {
        h_filter[i] = 1.0f / (i + 1);
    }

    // Allocate device memory
    cudaMalloc(d_input, input_size);
    cudaMalloc(d_filter, filter_size);
    cudaMalloc(d_output, output_size);

    // Copy data to device
    cudaMemcpy(*d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_filter, h_filter, filter_size, cudaMemcpyHostToDevice);
}

// Function to verify results on host (replace with your verification logic)
void verifyResults(DataType* h_output) {
    /* printf("Output (first 5 elements):\n");
    for (int i = 0; i < 5; ++i) {
        printf("%f ", h_output[i]);
    }
    printf("\n"); */
}

__global__ void RowReuseConvolution(
    DataType* input, DataType* filter, DataType* output
) {
    // Calculate output element position
    int row_o = blockIdx.y * blockDim.y + threadIdx.y;
    int col_o = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds
    if (row_o >= IH || col_o >= IW) return;

    // Initialize output to 0
    output[row_o * IW * OC + col_o] = 0;

    // Loop over filter rows
    for (int row_f = 0; row_f < FH; ++row_f) {
        int row_i = row_o + row_f;

        // Check bounds
        if (row_i >= IH) break;

        // Load input row
        DataType row[IW * IC];
        int input_offset = row_i * IW * IC + col_o * IC;
        for (int i = 0; i < IW * IC; ++i) {
            row[i] = input[input_offset + i];
        }

        // Apply Algorithm 2 (RowReuse)
        if (row_f < FH - 1) {
            for (int i = 0; i <= row_f; ++i) {
                int output_offset = (row_o - row_f + i) * IW * OC + col_o;
                for (int c = 0; c < OC; ++c) {
                    output[output_offset + c] += row[i * IC + c] * filter[row_f * FW * IC * OC + i * IC * OC + c];
                }
            }
        } else if (row_f >= FH - 1 && row_f < IH - FH + 1) {
            for (int i = 0; i < FH; ++i) {
                int output_offset = (row_o - FH + 1 + i) * IW * OC + col_o;
                for (int c = 0; c < OC; ++c) {
                    output[output_offset + c] += row[i * IC + c] * filter[(FH - 1 - i) * FW * IC * OC + i * IC * OC + c];
                }
            }
        } else {
            for (int i = FH - 1; i >= 0; --i) {
                int output_offset = (IH - FH + 1) * IW * OC + col_o;
                for (int c = 0; c < OC; ++c) {
                    output[output_offset + c] += row[i * IC + c] * filter[(FH - i) * FW * IC * OC + i * IC * OC + c];
                }
            }
        }
    }
}

int main() {
    // Define input, filter, and output dimensions

    // Allocate and initialize host and device memory
    DataType *h_input, *h_filter, *h_output;
    h_input = (DataType*) malloc(IH * IW * IC * sizeof(float));
    h_filter = (DataType*) malloc(FH * FW * IC * sizeof(float));
    h_output = (DataType*) malloc(IH * IW * OC * sizeof(float));

    DataType *d_input, *d_filter, *d_output;
    d_input = (DataType*) malloc(IH * IW * IC * sizeof(float));
    d_filter = (DataType*) malloc(FH * FW * IC * sizeof(float));
    d_output = (DataType*) malloc(IH * IW * OC * sizeof(float));

    allocateAndInitialize(
        &d_input, &d_filter, &d_output,
        h_input, h_filter, h_output
    );

    // Define grid and block sizes
    dim3 blockDim(32, 32); // Define threads per block (adjust as needed)
    dim3 gridDim((IW + blockDim.x - 1) / blockDim.x,
                 (IH + blockDim.y - 1) / blockDim.y); // Define number of blocks
    clock_t t;
    t = clock();

    // Launch kernel
    RowReuseConvolution<<<gridDim, blockDim>>>(d_input, d_filter, d_output);

    t = clock() - t;
    double time_in_milliseconds = (((double)t) / CLOCKS_PER_SEC) * 1000;

    printf("%d,%f\n",IH,time_in_milliseconds);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, IH * IW * OC * sizeof(DataType), cudaMemcpyDeviceToHost);

    // Verify results
    verifyResults(h_output);

    // Free memory
    free(h_input);
    free(h_filter);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    return 0;
}