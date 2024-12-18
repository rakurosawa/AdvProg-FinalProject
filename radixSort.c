#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// A utility function to get the maximum value in arr[]
int getMax(int arr[], int n) {
    int mx = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > mx)
            mx = arr[i];
    return mx;
}

// A function to do counting sort of arr[] according to the digit represented by exp
void countSort(int arr[], int n, int exp) {
    int *output = (int*)malloc(n * sizeof(int)); // Output array
    int count[10] = {0}; // Initialize count array as 0

    // Store count of occurrences in count[]
    for (int i = 0; i < n; i++)
        count[(arr[i] / exp) % 10]++;

    // Change count[i] so that count[i] now contains actual position of this digit in output[]
    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];

    // Build the output array
    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    // Copy the output array to arr[], so that arr[] now contains sorted numbers according to current digit
    for (int i = 0; i < n; i++)
        arr[i] = output[i];

    free(output); // Free the dynamically allocated memory
}

// The main function to sort arr[] of size n using Radix Sort
void radixSort(int arr[], int n) {
    // Find the maximum number to know the number of digits
    int m = getMax(arr, n); 

    // Do counting sort for every digit. exp is 10^i where i is the current digit number
    for (int exp = 1; m / exp > 0; exp *= 10)
        countSort(arr, n, exp);
}

// A utility function to print an array (commented out for large arrays)
void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

// Function to generate a random array of integers
void generateRandomArray(int arr[], int size, int max_value, unsigned int seed) {
    srand(seed); // Set the seed for reproducibility
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % (max_value + 1); // Generate a random number between 0 and max_value
    }
}

// Function to get current time in nanoseconds
long long getNanoseconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts); // CLOCK_MONOTONIC for elapsed time
    return (long long)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

int main() {
    // Set up the array size and max value
    int size = 1000;  // 10 million elements
    int max_value = 1000000000; // Arbitrary large number for testing
    unsigned int seed = 42; // Seed for reproducibility

    // Allocate memory for the array
    int* arr = (int*)malloc(size * sizeof(int));
    if (arr == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Generate a random array of integers
    generateRandomArray(arr, size, max_value, seed);

    // Record the start time (in nanoseconds)
    long long start_time = getNanoseconds();

    // Perform Radix Sort
    radixSort(arr, size);

    // Record the end time (in nanoseconds)
    long long end_time = getNanoseconds();

    // Calculate and print the time taken for radix sort in nanoseconds
    long long elapsed_time = end_time - start_time;
    printf("Time taken for radix sort (10,000,000 elements): %lld nanoseconds\n", elapsed_time);

    // Optionally print the sorted array (commented out to avoid printing large arrays)
    // printArray(arr, size);

    // Free dynamically allocated memory
    free(arr);

    return 0;
}
