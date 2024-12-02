#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>  // Include the curand header

#define SIZE_OF_ARRAY 1048576  // Define the size of the array

__device__ void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Device function for randomized partitioning using curand for random number generation
__device__ int random_pivot_partition(int *array, int low, int high, curandState *state) {
    // Randomly choose a pivot index between low and high using curand
    int pivotIndex = low + (curand(state) % (high - low + 1));  // Random index between low and high
    swap(&array[pivotIndex], &array[high]);  // Swap the pivot with the last element

    int pivot = array[high];  // Now, the pivot is at the end of the sub-array
    int i = (low - 1);  // Index for the smaller element
    for (int j = low; j < high; j++) {
        if (array[j] < pivot) {
            i++;
            swap(&array[i], &array[j]);
        }
    }
    swap(&array[i + 1], &array[high]);
    return (i + 1);
}

// Kernel to initialize curand states for each thread
__global__ void init_curand_states(curandState *states, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);
}

__global__ void quicksortKernel(int *array, int N, curandState *states) {
    // Stack to simulate recursion
    int stack[1024];  // Stack size limit for simplicity
    int top = -1;

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread is responsible for a different chunk
    int chunkSize = (N + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x); // Divide array into chunks
    int low = threadId * chunkSize;
    int high = min(low + chunkSize - 1, N - 1);

    // Push the initial subarray (low, high) onto the stack
    if (low < high) {
        stack[++top] = low;
        stack[++top] = high;
    }

    // Iterative process using the stack
    while (top >= 0) {
        high = stack[top--];
        low = stack[top--];

        if (low < high) {
            // Use random pivot partitioning instead of the regular partition
            int pi = random_pivot_partition(array, low, high, states + threadId);

            // Push the left subarray (low, pi - 1) onto the stack
            if (low < pi - 1) {
                stack[++top] = low;
                stack[++top] = pi - 1;
            }

            // Push the right subarray (pi + 1, high) onto the stack
            if (pi + 1 < high) {
                stack[++top] = pi + 1;
                stack[++top] = high;
            }
	}
    }
}

void quicksort(int *array, int N) {
    int *d_array;
    curandState *d_states;

    cudaMalloc((void**)&d_array, N * sizeof(int));
    cudaMemcpy(d_array, array, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_states, N * sizeof(curandState));

    // Initialize random states on the device
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    init_curand_states<<<blocksPerGrid, threadsPerBlock>>>(d_states, time(NULL));  // Initialize states with a seed based on current time
    cudaDeviceSynchronize();

    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start, 0);

    // Launch the quicksort kernel
    quicksortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, N, d_states);
    cudaDeviceSynchronize();

    // Record stop time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // Convert milliseconds to seconds
    float seconds = milliseconds / 1000.0;
    printf("GPU Quicksort took %f seconds\n", seconds);

    // Copy back the sorted array
    cudaMemcpy(array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_array);
    cudaFree(d_states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int N = SIZE_OF_ARRAY; // Use the defined size of the array
    int *array = (int*)malloc(N * sizeof(int));

    // Initialize array with random values
    for (int i = 0; i < N; i++) {
        array[i] = rand() % 1000;
    }

    // Perform quicksort
    quicksort(array, N);

    // Optionally print the sorted array
    // for (int i = 0; i < N; i++) {
    //     printf("%d ", array[i]);
    // }
    // printf("\n");

    // Free host memory
    free(array);
    return 0;
}
