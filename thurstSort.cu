#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <thrust/device_vector.h>

const int N = 6;
int A[N] = {1, 4, 2, 8, 5, 7};

int main() {
    // Print the original array
    std::cout << "Original array: ";
    for (int i = 0; i < N; ++i) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;

    // Copy array to device
    thrust::device_vector<int> d_A(A, A + N);

    // Perform sorting using thrust::sort on the device (GPU)
    thrust::sort(thrust::device, d_A.begin(), d_A.end());

    // Copy the sorted array back to the host
    thrust::copy(d_A.begin(), d_A.end(), A);

    // Print the sorted array
    std::cout << "Sorted array: ";
    for (int i = 0; i < N; ++i) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
