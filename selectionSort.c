#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define SIZE_OF_ARRAY 2048

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

        // // uncomment to see array swap after each pass
        // printf("array at i = %d : ", i);
        // for (int k = 0; k < SIZE_OF_ARRAY; k ++){
        //     printf("%d  ", array[k]);
        // }
        // printf("\n");
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
    int *input;
    input = (int*)malloc(sizeof(int)*SIZE_OF_ARRAY);

    // initialize time points
    double t0, t1;

    for (int i = 0; i < SIZE_OF_ARRAY; i ++){
        input[i] = SIZE_OF_ARRAY - i;
    }

    // // uncomment to see array input prior to program run
    // printf("input: ");
    // for (int i = 0; i < SIZE_OF_ARRAY; i ++){
    //     printf("%d  ", input[i]);
    // }
    // printf("\n");

    // get start time
    t0 = get_clock();
    selectionSort(input);
    // get stop time
    t1 = get_clock();

    // get and print total runtime
    printf("time: %f ns\n", 1000000000.0*(t1-t0));

    // // uncomment to see array output after program run
    // printf("output: ");
    // for (int i = 0; i < SIZE_OF_ARRAY; i ++){
    //     printf("%d  ", input[i]);
    // }
    // printf("\n");

    // check the array for errors
    for (int j = 0; j < SIZE_OF_ARRAY; j ++){
        if (input[j] != j + 1){
            printf("error of unexpected value at index %d\n", j);
        }
    }

    free(input);
    return 0;

}