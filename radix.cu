#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <chrono> // For high-resolution clock
#include <sys/time.h>

using namespace std;
using namespace std::chrono;

#define WSIZE 1000000  // Set array size to 10,000,000
#define LOOPS 10         // Set to 1 for a single run
#define UPPER_BIT 10
#define LOWER_BIT 0

// Use int type instead of unsigned int to match radixsort function
__device__ int ddata[WSIZE]; // Device data array changed to int

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
    int output[1024]; // Fixed-size output array, max size 1024 (adjustable if needed)
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

    // Do counting sort for every digit. 
    for (int exp = 1; m / exp > 0; exp *= 10)
        countSort(arr, n, exp);
}

__global__ void parallelRadix()
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < WSIZE) {
        radixsort(ddata, WSIZE); // Perform radix sort on the data
    }
}

double get_clock() {
    struct timeval tv; int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok < 0) { 
        printf("gettimeofday error"); 
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main() {
    unsigned int hdata[WSIZE];  // Host array to store data
    float totalTime = 0;

    auto start_time = high_resolution_clock::now(); 

    for (int lcount = 0; lcount < LOOPS; lcount++) {
        // Fill array with random elements
        for (int i = 0; i < WSIZE; i++) {
            hdata[i] = rand() % 1024;  // Random values in the range of 0 to 1023
        }

        cudaMemcpyToSymbol(ddata, hdata, WSIZE * sizeof(int)); // Ensure data is copied correctly

        auto kernel_start = high_resolution_clock::now();

        parallelRadix<<<(WSIZE + 255) / 256, 256>>>();
        // run kernel
        cudaDeviceSynchronize();

        auto kernel_end = high_resolution_clock::now();

        auto duration = duration_cast<nanoseconds>(kernel_end - kernel_start).count();
        totalTime += duration;

        cudaMemcpyFromSymbol(hdata, ddata, WSIZE * sizeof(int)); // Ensure data is copied correctly
    }

    // End timing using chrono high-resolution clock
    auto end_time = high_resolution_clock::now();
    auto total_duration = duration_cast<nanoseconds>(end_time - start_time).count();

    printf("Total time elapsed for %d iterations: %lld ns\n", LOOPS, total_duration);
    printf("Average time per iteration: %lld ns\n", totalTime / LOOPS);

    return 0;
}
