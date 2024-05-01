// CONVOLUTION CUDA
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(
    const float* input, const float* filter, float* output,
    int input_width, int input_height, int filter_width, int filter_height
) {
  // Thread ID and block dimensions
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int threads_per_block = blockDim.x;

  // Calculate output element coordinates
  int output_x = bid * threads_per_block + tid;
  int output_y = blockIdx.y;

  // Check if within output bounds
  if (output_x >= input_width || output_y >= input_height) return;

  // Calculate padding and stride (assuming padding = 1, stride = 1)
  int padding = (filter_width - 1) / 2;

  // Temporary buffer for input elements
  float iTemp[filter_width];

  // Load first and last elements for each thread
  iTemp[0] = input[(output_y * input_width) + output_x];
  iTemp[filter_width - 1] = input[(output_y * input_width) + output_x + filter_width - 1];

  // Apply column reuse optimization (Algorithm 1)
  if (tid < threads_per_block - 2) {
    unsigned long long exchange;
    asm volatile ("mov.b64 %0, {%1, %2};" : "=l"(exchange) : "r"(iTemp[0]), "r"(iTemp[filter_width - 1]));
    int shift = ((tid + 2) & 2) << 4;
    asm volatile ("shr.b64 %0, %1, %2;" : "=l"(exchange) : "r"(exchange), "r"(shift));
    asm volatile ("mov.b64 {%0, %1}, %2;" : "=r"(iTemp[1]), "=r"(iTemp[2]) : "l"(exchange));
    iTemp[2] = __shfl_xor(iTemp[1], 2);
  }
  __syncthreads();

  // Perform convolution for the current output element
  float sum = 0.0f;
  for (int fy = 0; fy < filter_height; fy++) {
    for (int fx = 0; fx < filter_width; fx++) {
      int input_x = output_x + fx - padding;
      int input_y = output_y + fy - padding;

      // Check if within input bounds (ignoring padding)
      if (input_x >= 0 && input_x < input_width && input_y >= 0 && input_y < input_height) {
        sum += iTemp[fx] * filter[(fy * filter_width) + fx];
      }
    }
  }
  output[(output_y * input_width) + output_x] = sum;
}

// Host code to launch kernel
double A2ColumnReuse(float* IMG_IN, float* IMG_OUT, float*  FILTER_IN, int IMAGE_SIZE, int FILTER_SIZE)
{
    int imgBytes = IMAGE_SIZE * IMAGE_SIZE * sizeof(float);
    int filterBytes = FILTER_SIZE * FILTER_SIZE * sizeof(float);

    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, imgBytes);
    cudaMalloc(&d_filter, filterBytes);
    cudaMalloc(&d_output, imgBytes);

    cudaMemcpy(d_input, IMG_IN, imgBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, FILTER_IN, filterBytes, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int threads_per_block = 32;
    int blocks_per_grid_x = (IMAGE_SIZE + threads_per_block - 1) / threads_per_block;
    int blocks_per_grid_y = IMAGE_SIZE;

    dim3 block_size(threads_per_block);
    dim3 grid_size(blocks_per_grid_x, blocks_per_grid_y);
    // Launch kernel
    clock_t t;
    t = clock();
    
    convolution_kernel<<<grid_size, block_size>>>(d_input, d_filter, d_output, IMAGE_SIZE, IMAGE_SIZE, FILTER_SIZE, FILTER_SIZE);
    cudaDeviceSynchronize();

    t = clock() - t;
    double time_taken_in_seconds = ((double)t) / CLOCKS_PER_SEC;

    return time_taken_in_seconds * 1000
}