#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <sys/time.h>

using namespace std;
using namespace std::chrono;

#define WSIZE 10000000  // Set array size to 10,000,000
#define LOOPS 10         // Set to 1 for a single run
#define UPPER_BIT 10
#define LOWER_BIT 0

__device__ unsigned int ddata[WSIZE];

template <typename T, unsigned S>
inline unsigned arraysize(const T(&v)[S])
{
    return S;
}

template<typename T>
void fillArray(T &arr)
{
    srand(time(NULL));
    for (int i = 0; i < arraysize(arr); ++i)
    {
        arr[i] = rand() % 1024;  // Random values in the range of 0 to 1023
    }
}

__device__ int getMax(int arr[], int n)
{
    int mx = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > mx)
            mx = arr[i];
    return mx;
}

__device__ void countSort(int arr[], int n, int exp)
{
    int output[n]; // Output array
    int i, count[10] = { 0 };

    // Store count of occurrences in count[]
    for (i = 0; i < n; i++)
        count[(arr[i] / exp) % 10]++;

    // Change count[i] so that count[i] now contains actual
    // position of this digit in output[]
    for (i = 1; i < 10; i++)
    {
        count[i] += count[i - 1];
    }

    // Build the output array
    for (i = n - 1; i >= 0; i--)
    {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    // Copy the output array to arr[], so that arr[] now
    // contains sorted numbers according to current digit
    for (i = 0; i < n; i++)
        arr[i] = output[i];
}

__device__ void radixsort(int arr[], int n)
{
    // Find the maximum number to know number of digits
    int m = getMax(arr, n);

    // Do counting sort for every digit. Note that instead
    // of passing digit number, exp is passed. exp is 10^i
    // where i is current digit number
    for (int exp = 1; m / exp > 0; exp *= 10)
        countSort(arr, n, exp);
}

__global__ void parallelRadix()
{
    // Load from global into shared variable
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < WSIZE) {
        unsigned int mydata = ddata[tid];
        // Perform radix sort on the data
        radixsort(ddata, WSIZE);
    }
}

double get_clock() {
    struct timeval tv; int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok < 0) { 
        printf("gettimeofday error"); 
    }
    return (tv.tv_sec * 1. 0 + tv.tv_usec * 1.0E-6);
}

int main() {
    unsigned int hdata[WSIZE];  // Host array to store data
    float totalTime = 0;
    double start = get_clock();

    for (int lcount = 0; lcount < LOOPS; lcount++) {
        // Fill array with random elements
        for (int i = 0; i < WSIZE; i++) {
            hdata[i] = rand() % 1024;  // Random values in the range of 0 to 1023
        }

        // Copy data from host to device
        cudaMemcpyToSymbol(ddata, hdata, WSIZE * sizeof(unsigned int));

        // Execution time measurement, that point starts the clock
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        parallelRadix<<<(WSIZE + 255) / 256, 256>>>();
        // Make kernel function synchronous
        cudaDeviceSynchronize();
        // Execution time measurement, that point stops the clock
        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        // Execution time measurement, that is the result
        auto duration = duration_cast<milliseconds>(t2 - t1).count();
        totalTime += (float)duration / 1000.00;

        // Copy data from device to host
        cudaMemcpyFromSymbol(hdata, ddata, WSIZE * sizeof(unsigned int));
    }

    double end = get_clock();
    printf("Time per call: %f ns\n", (end - start));
    printf("Parallel Radix Sort:\n");
    printf("Array size = %d\n", WSIZE);
    printf("Time elapsed = %f seconds\n", totalTime);

    return 0;
}