#include <stdio.h>
#include <cuda_runtime.h>

#define MASK_SIZE 3  // Convolution kernel size
#define TILE_SIZE 16 // Tile size for shared memory optimization

// CUDA Kernel for 2D Convolution
__global__ void convolution2D(float *input, float *output, float *mask, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float sum = 0.0f;
    
    if (row < height && col < width) {
        for (int i = 0; i < MASK_SIZE; i++) {
            for (int j = 0; j < MASK_SIZE; j++) {
                int r = row + i - MASK_SIZE / 2;
                int c = col + j - MASK_SIZE / 2;
                
                if (r >= 0 && r < height && c >= 0 && c < width) {
                    sum += input[r * width + c] * mask[i * MASK_SIZE + j];
                }
            }
        }
        output[row * width + col] = sum;
    }
}

// Utility function for initializing input and kernel
void initializeData(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = rand() % 10;  // Random values between 0-9
    }
}

int main() {
    int width = 16, height = 16;
    int imageSize = width * height * sizeof(float);
    int maskSize = MASK_SIZE * MASK_SIZE * sizeof(float);

    // Allocate memory on host
    float *h_input = (float *)malloc(imageSize);
    float *h_output = (float *)malloc(imageSize);
    float *h_mask = (float *)malloc(maskSize);

    initializeData(h_input, width * height);
    initializeData(h_mask, MASK_SIZE * MASK_SIZE);

    // Allocate memory on device
    float *d_input, *d_output, *d_mask;
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_mask, maskSize);

    // Copy data to device
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, maskSize, cudaMemcpyHostToDevice);

    // Define grid and block size
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the CUDA kernel
    convolution2D<<<gridDim, blockDim>>>(d_input, d_output, d_mask, width, height);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Print a few values of the output
    printf("Sample Output:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
    free(h_input);
    free(h_output);
    free(h_mask);

    return 0;
}
