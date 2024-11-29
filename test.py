
import random
import time
from radix import radixSort  # Importing radixSort from radix_sort.py, import what u need

# Function to generate a random array of integers
def generate_random_array(size, max_value, seed=None):
    if seed is not None:
        random.seed(seed)
    
    # Generate the random array of integers
    arr = [random.randint(0, max_value) for _ in range(size)]
    return arr


# Function to test radixSort with a random array
def test_radix_sort():
    # Generate a random array with size 10 and values between 0 and 1000
    size = 1000000
    max_value = 10000000
    seed = 42  # Seed for reproducibility
    arr = generate_random_array(size, max_value, seed)
    
    #print("Original array:", arr)
    
    # Start timing the sorting process
    start_time = time.time()  # Capture start time
    
    # Perform radix sort
    radixSort(arr)
    
    # End timing the sorting process
    end_time = time.time()  # Capture end time
    
    # Calculate and print the time taken for the sort
    elapsed_time = end_time - start_time
    #print("Sorted array:", arr)
    print(f"Time taken for radixSort: {elapsed_time:.6f} seconds")

# Run the test
if __name__ == "__main__":
    test_radix_sort()
