#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CHECK_CUDNN(status) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        printf("Error: %s\n", cudnnGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    }

void initializeImage(float *img, int size) {
    for (int i = 0; i < size; ++i) {
        img[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    // Initialize cuDNN
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Define image and filter dimensions
    int imgSize = 4;  // Size of input image (assuming square image)
    int filterSize = 3;  // Size of filter (assuming square filter)
    int imgBytes = imgSize * imgSize * sizeof(float);
    int filterBytes = filterSize * filterSize * sizeof(float);

    // Allocate memory for input image, filter, and output
    float *h_input = (float *)malloc(imgBytes);
    float *h_filter = (float *)malloc(filterBytes);
    float *h_output = (float *)malloc(imgBytes);

    // Initialize input image and filter
    initializeImage(h_input, imgSize * imgSize);
    initializeImage(h_filter, filterSize * filterSize);

    // Allocate device memory for input image, filter, and output
    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, imgBytes);
    cudaMalloc(&d_filter, filterBytes);
    cudaMalloc(&d_output, imgBytes);

    // Copy input image and filter to device
    cudaMemcpy(d_input, h_input, imgBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterBytes, cudaMemcpyHostToDevice);

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
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, imgSize, imgSize);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, imgSize - filterSize + 1, imgSize - filterSize + 1);
    cudnnFilterDescriptor_t filterDesc;
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, filterSize, filterSize);

    // Define convolution algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(cudnn, inputDesc, filterDesc, convDesc, outputDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);

    // Perform convolution
    void *workSpace = NULL;
    size_t workSpaceSize = 0;
    cudnnStatus_t status = cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, algo, &workSpaceSize);
    CHECK_CUDNN(status);
    if (workSpaceSize > 0) {
        cudaMalloc(&workSpace, workSpaceSize);
    }
    float alpha = 1.0f;
    float beta = 0.0f;
    status = cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, filterDesc, d_filter, convDesc, algo, workSpace, workSpaceSize, &beta, outputDesc, d_output);
    CHECK_CUDNN(status);

    // Copy output from device to host
    cudaMemcpy(h_output, d_output, imgBytes, cudaMemcpyDeviceToHost);

    // Print output
    printf("Output image:\n");
    for (int i = 0; i < imgSize - filterSize + 1; ++i) {
        for (int j = 0; j < imgSize - filterSize + 1; ++j) {
            printf("%f ", h_output[i * (imgSize - filterSize + 1) + j]);
        }
        printf("\n");
    }

    // Cleanup
    free(h_input);
    free(h_filter);
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
