#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define SIZE_OF_ARRAY 1024 // test values of 1024, 32768, 65536, 131072, 262144, 524288, 1048576 for time and scalability

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

    // Set up the array size and max value
    int max_value = 10000000;
    unsigned int seed = 42; // Seed for reproducibility

    // Generate a random array of integers for testing
    generateRandomArray(array, SIZE_OF_ARRAY, max_value, seed);

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