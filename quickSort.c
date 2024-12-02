#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define SIZE_OF_ARRAY 1048576// Array size of 10 million elements

// Function to swap two elements
void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

// Partition function
int partition(int arr[], int low, int high) {
    int pivot = arr[high]; // Choosing the last element as pivot
    int i = low - 1; // Index of smaller element

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]); // Place the pivot in the correct position
    return i + 1;
}

// Randomized Partition: Choosing a random pivot
int random_pivot_partition(int arr[], int low, int high) {
    int random_index = low + rand() % (high - low + 1); // Ensure pivot is between low and high
    swap(&arr[random_index], &arr[high]); // Swap random pivot with the last element
    return partition(arr, low, high); // Perform partition
}

// QuickSort function
void quickSort(int arr[], int low, int high) {
    if (low < high) {
        // Use randomized pivot selection
        int pi = random_pivot_partition(arr, low, high);

        // Recursively sort the elements before and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// Function to get the current time (used for performance measurement)
double get_clock() {
    struct timeval tv; 
    int ok;
    ok = gettimeofday(&tv, (void *)0);
    if (ok < 0) { 
        printf("gettimeofday error\n"); 
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main() {
    // Seed the random number generator once in main
    srand(time(NULL));

    // Initialize the array
    int *input = (int*)malloc(sizeof(int) * SIZE_OF_ARRAY);
    if (input == NULL) {
        printf("Memory allocation failed\n");
        return -1;
    }
    printf("Array memory allocated successfully\n");

    // Fill the array with values in descending order
    for (int i = 0; i < SIZE_OF_ARRAY; i++) {
        input[i] = SIZE_OF_ARRAY - i;
    }
    printf("Array initialized successfully\n");

    // Measure time for sorting
    double t0, t1;
    t0 = get_clock();
    quickSort(input, 0, SIZE_OF_ARRAY - 1); // Call quick sort
    t1 = get_clock();

    // Print the total time for sorting
    printf("Time for quickSort: %f seconds\n", t1 - t0);

    // Uncomment to check the array for errors
    // for (int j = 0; j < SIZE_OF_ARRAY; j++) {
    //     if (input[j] != j + 1) {
    //         printf("Error of unexpected value at index %d\n", j);
    //     }
    // }

    // Free allocated memory
    free(input);
    return 0;
}
