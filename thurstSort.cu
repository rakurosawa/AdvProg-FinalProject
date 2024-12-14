#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdlib>
#include <ctime>
#include <chrono> // Include chrono for high-precision timing

// Function to generate a random array of integers
void generateRandomArray(int arr[], int size, int max_value, unsigned int seed) {
    srand(seed); // Set the seed for reproducibility
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % (max_value + 1); // Generate a random number between 0 and max_value
    }
}

int main() {
    // Set up the array size and max value
    const int size = 10000000; // 10 million elements (adjust as needed)
    const int max_value = 10000000; // Max value for random numbers
    unsigned int seed = 42; // Seed for reproducibility

    // Allocate memory for the array
    int* h_A = new int[size];

    // Generate a random array of integers
    generateRandomArray(h_A, size, max_value, seed);

    // Print the first 10 elements of the original array for verification
    std::cout << "Original array (first 10 elements): ";
    for (int i = 0; i < 10 && i < size; ++i) {
        std::cout << h_A[i] << " ";
    }
    std::cout << "..." << std::endl;

    // Record the start time using high-precision timer (chrono)
    auto start_time = std::chrono::high_resolution_clock::now();

    // Copy the array to device memory
    thrust::device_vector<int> d_A(h_A, h_A + size);

    // Perform sorting using thrust::sort on the device (GPU)
    thrust::sort(thrust::device, d_A.begin(), d_A.end());

    // Copy the sorted array back to the host
    thrust::copy(d_A.begin(), d_A.end(), h_A);

    // Record the end time
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate and print the time taken for sorting in nanoseconds
    auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "Time taken for sorting with Thrust: " << elapsed_time.count() << " nanoseconds" << std::endl;

    // Print the first 10 elements of the sorted array for verification
    std::cout << "Sorted array (first 10 elements): ";
    //for (int i = 0; i < 10 && i < size; ++i) {
        //std::cout << h_A[i] << " ";
    //}
    //std::cout << "..." << std::endl;

    // Free dynamically allocated memory
    delete[] h_A;

    return 0;
}
