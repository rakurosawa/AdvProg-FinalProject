#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define SIZE_OF_ARRAY 2048

// func to swap two elements
void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

// Partition func
int partition(int arr[], int low, int high) {
    int pivot = arr[high]; // Choose last element as pivot
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

// Quick Sort func
void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1); // Recursively sort elements before partition
        quickSort(arr, pi + 1, high); // Recursively sort elements after partition
    }
}
// Time part
double get_clock() {
    struct timeval tv; 
    int ok;
    ok = gettimeofday(&tv, (void *)0);
    if (ok < 0) { 
        printf("gettimeofday error"); 
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main() {
    // Initialize the array
    int *input;
    input = (int*)malloc(sizeof(int) * SIZE_OF_ARRAY);

    // Initialize time points
    double t0, t1;

    // Fill the array with values in descending order
    for (int i = 0; i < SIZE_OF_ARRAY; i++) {
        input[i] = SIZE_OF_ARRAY - i;
    }

    // Get start time
    t0 = get_clock();
    quickSort(input, 0, SIZE_OF_ARRAY - 1); // Call quick sort
    // Get stop time
    t1 = get_clock();

    // Get and print total runtime in nanoseconds
    printf("Time for quickSort: %f ns\n", 1000000000.0 * (t1 - t0));

    // Uncomment to see array output after program run
    // printf("Output: ");
    // for (int i = 0; i < SIZE_OF_ARRAY; i++) {
    //     printf("%d  ", input[i]);
    // }
    // printf("\n");

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
