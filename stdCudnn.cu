#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CHECK_CUDNN(status) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        printf("Error: %s\n", cudnnGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    }

int myStdCudnn(float* IMG_IN, float* IMG_OUT, float* FILTER_IN, int IMAGE_SIZE, int FILTER_SIZE) {
    // Initialize cuDNN
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Define image and filter dimensions
    int IMAGE_SIZE = IMAGE_SIZE;  // Size of input image (assuming square image)
    int FILTER_SIZE = FILTER_SIZE;  // Size of filter (assuming square filter)
    int imgBytes = IMAGE_SIZE * IMAGE_SIZE * sizeof(float);
    int filterBytes = FILTER_SIZE * FILTER_SIZE * sizeof(float);

    // Allocate memory for input image, filter, and output
    // float *h_input = (float *) malloc(imgBytes);
    float *h_output = (float *) malloc(imgBytes);

    // float *FILTER_IN = (float *)malloc(filterBytes);
    // float FILTER_IN[] = {2,0,0, 0,0,0, 0,0,0};

    // Initialize input image and filter
    initializeImage(h_input, IMAGE_SIZE * IMAGE_SIZE);
    // initializeImage(FILTER_IN, FILTER_SIZE * FILTER_SIZE);

    // Allocate device memory for input image, filter, and output
    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, imgBytes);
    cudaMalloc(&d_filter, filterBytes);
    cudaMalloc(&d_output, imgBytes);

    // Copy input image and filter to device
    cudaMemcpy(d_input, h_input, imgBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, FILTER_IN, filterBytes, cudaMemcpyHostToDevice);

    // Define convolution parameters
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    int pad = 0;  // Padding
    int stride = 1;  // Stride
    int dilation = 1;  // Dilation
    cudnnSetConvolution2dDescriptor(convDesc, pad, pad, stride, stride, dilation, dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Define convolution operation parameters
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, IMAGE_SIZE, IMAGE_SIZE);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, IMAGE_SIZE - FILTER_SIZE + 1, IMAGE_SIZE - FILTER_SIZE + 1);
    cudnnFilterDescriptor_t filterDesc;
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, FILTER_SIZE, FILTER_SIZE);

    // Perform convolution
    cudnnConvolutionFwdAlgo_t algo;
    // cudnnGetConvolutionForwardAlgorithm_v7(cudnn, inputDesc, filterDesc, convDesc, outputDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, 0, &algo);
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    int numAlgos;

    clock_t t;
    t = clock();

    cudnnGetConvolutionForwardAlgorithm_v7(cudnn, inputDesc, filterDesc, convDesc, outputDesc, 1, &numAlgos, &perfResults);
    algo = perfResults.algo;
    printf("%d",algo);

    // t = clock() - t;
    // double time_taken_in_seconds = ((double)t) / CLOCKS_PER_SEC;


    void *workSpace = NULL;
    size_t workSpaceSize = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, algo, &workSpaceSize);
    if (workSpaceSize > 0) {
        cudaMalloc(&workSpace, workSpaceSize);
    }
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, filterDesc, d_filter, convDesc, algo, workSpace, workSpaceSize, &beta, outputDesc, d_output);

    t = clock() - t;
    double time_taken_in_seconds = ((double)t) / CLOCKS_PER_SEC;

    // Copy output from device to host
    cudaMemcpy(h_output, d_output, imgBytes, cudaMemcpyDeviceToHost);

    printf("Time %f \n",time_taken_in_seconds * 1000);
    // Print output
    printf("Output image:\n");
    for (int i = 0; i < IMAGE_SIZE - FILTER_SIZE + 1; ++i) {
        for (int j = 0; j < IMAGE_SIZE - FILTER_SIZE + 1; ++j) {
            printf("%f ", h_output[i * (IMAGE_SIZE - FILTER_SIZE + 1) + j]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < FILTER_SIZE; ++i) {
        for (int j = 0; j < FILTER_SIZE; ++j) {
            printf("%f ", FILTER_IN[i * FILTER_SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        for (int j = 0; j < IMAGE_SIZE; ++j) {
            printf("%f ", h_input[i * IMAGE_SIZE + j]);
        }
        printf("\n");
    }

    // Cleanup
    // free(h_input);
    // free(FILTER_IN);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);

    return 0;
}
