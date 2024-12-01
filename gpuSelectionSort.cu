#include <stdlib.h>
#include <stdio.h>

#define SIZE_OF_ARRAY 16
#define NUM_THDS 8
#define BLOCK_SIZE 4


__global__ void gpuSelectionSort(int* array){
    __shared__ int temp[SIZE_OF_ARRAY];

    int globalIdx = threadIdx.x + (BLOCK_SIZE * blockIdx.x);
    //printf("%d ", globalIdx);

    for (int i = 0; i < SIZE_OF_ARRAY; i ++){
        __syncthreads();

        int minIdx = i;

        //printf("thread: %d\n", globalIdx);

        int arrIdx1 = 2 * globalIdx + i;
        int arrIdx2 = arrIdx1 + 1;
        //printf("arrIdx1: %d\n", arrIdx1);

        if (arrIdx1 >= SIZE_OF_ARRAY){
            temp[globalIdx * 2] = array[SIZE_OF_ARRAY - 1];
        }
        else {
            temp[globalIdx * 2] = array[arrIdx1];
        }
        
        if (arrIdx2 >= SIZE_OF_ARRAY){
            temp[globalIdx * 2 + 1] = array[SIZE_OF_ARRAY - 1];
        }
        else {
            temp[globalIdx * 2 + 1] = array[arrIdx2];
        }

        // for (int j = NUM_THDS; j > 0; j /= 2){
        //     __syncthreads();
        //     if (globalIdx < j){
        //         int idx1 = globalIdx;
        //         int val1 = temp[idx1];
        //         int idx2 = idx1 + 1;
        //         int val2 = temp[idx2];

        //         if (val1 < val2){
        //             temp[idx1] = val1;
        //         }
        //         else {
        //             temp[idx1] = val2;
        //         }
        //     }   
        // }

        // for (int k = 0; k < SIZE_OF_ARRAY; k ++){
        //     if (array[k] == temp[0]){
        //         minIdx = k;
        //     }
        // }

        // if (temp[0] <= array[i]){
        //     int tempVal = array[i];
        //     array[i] = array[minIdx];
        //     array[minIdx] = tempVal;
        // }

    }
    __syncthreads();
}
    


int main(){

    // initialize the array
    int *array;
    cudaMallocManaged(&array, (SIZE_OF_ARRAY)*sizeof(int));

    // set array values
    for (int i = 0; i < SIZE_OF_ARRAY; i ++){
        array[i] = SIZE_OF_ARRAY - i;
    }

    // uncomment to see array input prior to program run
    for (int i = 0; i < SIZE_OF_ARRAY; i ++){
        printf("%d  ", array[i]);
    }
    printf("\n");

    // <<< number of blocks, size of each block >>>
    gpuSelectionSort<<<NUM_THDS/BLOCK_SIZE, BLOCK_SIZE>>>(array);
    cudaDeviceSynchronize();

    // uncomment to see array output after program run
    for (int i = 0; i < SIZE_OF_ARRAY; i ++){
        printf("%d  ", array[i]);
    }
    printf("\n");

    // // check the array for errors
    // for (int j = 0; j < SIZE_OF_ARRAY; j ++){
    //     if (array[j] != j + 1){
    //         printf("error of unexpected value at index %d\n", j);
    //     }
    // }

    cudaFree(array);
}