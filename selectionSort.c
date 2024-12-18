#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define SIZE_OF_ARRAY 1000000

// adopted from https://www.geeksforgeeks.org/selection-sort-algorithm-2/
void selectionSort(int* array){
    // go through the array "making" subarrays to perform the algorithm on
    for (int i = 0; i < SIZE_OF_ARRAY; i ++){
        int minIdx = i;

        // find the smallest item in the rest of the array 
        for (int j = i+1; j < SIZE_OF_ARRAY; j ++){
            if (array[j] < array[minIdx]){
                minIdx = j;
            }
        }

        // swap the smallest value to the "front of the sub array"
        int temp = array[i];
        array[i] = array[minIdx];
        // // uncomment to see array swap in progress
        // printf("swapping %d and %d\n", temp, array[i]);
        array[minIdx] = temp;

    }
}

void generateRandomArray(int arr[], int size, int max_value, unsigned int seed) {
    srand(seed); // Set the seed for reproducibility
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % (max_value + 1); // Generate a random number between 0 and max_value
    }
}

void generateInvertSortArray(int* arr, int size){
    for (int i = 0; i < size; i++){
        arr[i] = size - i;
    }
}

void generatePreSortedArray(int* arr, int size){
    for (int i = 0; i < size; i++){
        arr[i] = i;
    }
}

double get_clock() {
    struct timeval tv; int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok<0) { 
        printf("gettimeofday error"); 
        }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main(){

    // initialize the array
    int *array;
    array = (int*)malloc(sizeof(int)*SIZE_OF_ARRAY);

    // initialize time points
    double t0, t1;

    // Set up the array size and seed
    int max_value = 10000000;
    unsigned int seed = 42;

    // // Generate a random array of integers for testing
    // generateRandomArray(array, SIZE_OF_ARRAY, max_value, seed);

    // // Generate an inversely sorted array of integers for testing
    // generateInvertSortArray(array, SIZE_OF_ARRAY);

    // Generate an already sorteed array of integers for testing
    generatePreSortedArray(array, SIZE_OF_ARRAY);

    // // uncomment to see array input prior to program run (beware large arrays)
    // printf("input: ");
    // for (int i = 0; i < SIZE_OF_ARRAY; i ++){
    //     printf("%d  ", array[i]);
    // }
    // printf("\n");

    // get start time
    t0 = get_clock();
    selectionSort(array);
    // get stop time
    t1 = get_clock();

    // get and print total runtime (in seconds)
    printf("time: %f sec\n", (t1-t0));

    // // uncomment to see array output after program run (beware large arrays)
    // printf("output: ");
    // for (int i = 0; i < SIZE_OF_ARRAY; i ++){
    //     printf("%d  ", array[i]);
    // }
    // printf("\n");

    free(array);
    return 0;
}