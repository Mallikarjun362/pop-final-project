#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>

// Error handling macro
#define CHECK_CUDNN(expression)                                                 \
  do {                                                                          \
    cudnnStatus_t status = (expression);                                        \
    if (status != CUDNN_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "Error at line %d: %s\n", __LINE__, cudnnGetErrorString(status)); \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

// Function for convolution using CuDNN
void cudnnConvolution(
    float *inputImage, 
    float *kernel, 
    int imageSize, 
    int kernelSize, 
    float *outputImage
) {
  // CuDNN handle
  cudnnHandle_t cudnnHandle;
  CHECK_CUDNN(cudnnCreate(&cudnnHandle));

  // Data tensor descriptors
  cudnnTensorDescriptor_t inputDescriptor, kernelDescriptor, outputDescriptor;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDescriptor));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&kernelDescriptor));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDescriptor));

  // Set tensor dimensions (assuming NCHW format)
  const int n = 1, c = 1, h = imageSize, w = imageSize;
  const int k = 1, c_k = 1, h_k = kernelSize, w_k = kernelSize;
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(kernelDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, k, c_k, h_k, w_k));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

  // Convolution descriptor
  cudnnConvolutionDescriptor_t convolutionDescriptor;
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolutionDescriptor));
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convolutionDescriptor, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

  // Set up convolution algorithm
  cudnnConvolutionFwdAlgo_t convolutionAlgorithm;
  CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputDescriptor, kernelDescriptor, 
                                                 convolutionDescriptor, outputDescriptor, 
                                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convolutionAlgorithm));

  // Workspace size and allocation
  size_t workspaceSizeInBytes;
  CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputDescriptor, kernelDescriptor, 
                                                   convolutionDescriptor, outputDescriptor, 
                                                   convolutionAlgorithm, &workspaceSizeInBytes));
  void *workspace = nullptr;
  if (workspaceSizeInBytes > 0) {
    CHECK_CUDNN(cudaMalloc(&workspace, workspaceSizeInBytes));
  }

  // Perform convolution
  const float alpha = 1.0f, beta = 0.0f;
  CHECK_CUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, inputDescriptor, inputImage,
                                     kernelDescriptor, kernel, convolutionDescriptor, 
                                     convolutionAlgorithm, workspace, workspaceSizeInBytes,
                                     &beta, outputDescriptor, outputImage));

  // Cleanup
  if (workspace) {
    cudaFree(workspace);
  }
  cudnnDestroyTensorDescriptor(inputDescriptor);
  cudnnDestroyTensorDescriptor(kernelDescriptor);
  cudnnDestroyTensorDescriptor(outputDescriptor);
  cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
  cudnnDestroy(cudnnHandle);
}
// CONVOLUTION CUDA
double A1(DATA_TYPE *IMG_IN, DATA_TYPE *KERNEL_IN_H, const int IMAGE_SIZE, const int KERNEL_SIZE)
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