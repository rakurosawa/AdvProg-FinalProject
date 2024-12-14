#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define SIZE_OF_ARRAY 1000

__global__ void gpuSelectionSort(int* array){
    __shared__ int temp[SIZE_OF_ARRAY];
    int globalIdx = threadIdx.x + ((SIZE_OF_ARRAY/2) * blockIdx.x);

    __syncthreads();

    for (int i = 0; i < SIZE_OF_ARRAY; i ++){ 
        int minIdx = i;

        int arrayIdx1 = (globalIdx * 2) + i;
        int arrayIdx2 = arrayIdx1 + 1;

        // load values to temp
        if (arrayIdx1 < SIZE_OF_ARRAY){
            temp[globalIdx * 2] = array[arrayIdx1];
        }
        else{ // buffer for if index is out of reach
            temp[globalIdx * 2] = array[SIZE_OF_ARRAY - 1];
        }
        if (arrayIdx2 < SIZE_OF_ARRAY){
            temp[(globalIdx * 2) + 1] = array[arrayIdx2];
        }
        else{ // buffer for if index is out of reach
            temp[(globalIdx * 2) + 1] = array[SIZE_OF_ARRAY - 1];
        }

        __syncthreads();
        
        // reduction with less thread division
        for (int j = (SIZE_OF_ARRAY/2); j > 0; j /= 2){
            __syncthreads();
            if (globalIdx < j){
                int idx1 = globalIdx;
                int val1 = temp[idx1];
                int idx2 = idx1 + j;
                int val2 = temp[idx2];

                if (val1 < val2){
                    temp[idx1] = val1;
                }
                else {
                    temp[idx1] = val2;
                }
            }   
        }

        // find the index of the minimum
        for (int k = 0; k < SIZE_OF_ARRAY; k ++){
            if (array[k] == temp[0]){
                minIdx = k;
            }
        }

        // swap values if necessary
        if (temp[0] <= array[i]){
            int tempVal = array[i];
            array[i] = array[minIdx];
            array[minIdx] = tempVal;
        }

    }

    __syncthreads();
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
    cudaMallocManaged(&array, (SIZE_OF_ARRAY)*sizeof(int));

    // initialize time points
    double t0, t1;

    // Set up the array size and seed
    int max_value = 10000000;
    unsigned int seed = 42;

    // Generate a random array of integers for testing
    generateRandomArray(array, SIZE_OF_ARRAY, max_value, seed);

    // // Generate an inversely sorted array of integers for testing
    // generateInvertSortArray(array, SIZE_OF_ARRAY);

    // // Generate an already sorteed array of integers for testing
    // generatePreSortedArray(array, SIZE_OF_ARRAY);

    // // uncomment to see array input prior to program run (beware large arrays)
    // for (int i = 0; i < SIZE_OF_ARRAY; i ++){
    //     printf("%d  ", array[i]);
    // }
    // printf("\n");

    // get start time
    t0 = get_clock();
    // <<< number of blocks, size of each block >>>
    gpuSelectionSort<<<1, SIZE_OF_ARRAY/2>>>(array);
    cudaDeviceSynchronize();
    // get stop time
    t1 = get_clock();

    // get and print total runtime (in seconds)
    printf("time: %f sec\n", (t1-t0));

    // // uncomment to see array output after program run (beware large arrays)
    // for (int i = 0; i < SIZE_OF_ARRAY; i ++){
    //     printf("%d  ", array[i]);
    // }
    // printf("\n");

    cudaFree(array);
    return 0;
}