"""
NumPy Data Explorer

This script demonstrates:
1. Array creation, indexing, and slicing.
2. Mathematical, axis-wise, and statistical operations.
3. Reshaping and broadcasting for efficient computation.
4. Save/load operations for NumPy arrays.
5. Performance comparison: NumPy arrays vs Python lists.
"""

import numpy as np
import time


def section_title(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def array_creation_indexing_slicing():
    section_title("1. Array Creation, Indexing, and Slicing")

    # 1.1 Array creation
    a = np.array([1, 2, 3, 4, 5])
    zeros = np.zeros((2, 3))
    ones = np.ones((2, 3))
    arange_arr = np.arange(0, 10, 2)       # 0, 2, 4, 6, 8
    linspace_arr = np.linspace(0, 1, 5)    # 5 numbers from 0 to 1

    print("Array a:", a)
    print("Zeros array:\n", zeros)
    print("Ones array:\n", ones)
    print("Arange array:", arange_arr)
    print("Linspace array:", linspace_arr)

    # 1.2 Indexing and slicing
    print("\nIndexing and slicing on array a:")
    print("a[0] (first element):", a[0])
    print("a[-1] (last element):", a[-1])
    print("a[1:4] (elements from index 1 to 3):", a[1:4])

    # 2D example
    b = np.array([[10, 20, 30],
                  [40, 50, 60],
                  [70, 80, 90]])

    print("\n2D array b:\n", b)
    print("b[0, 0] (first row, first col):", b[0, 0])
    print("b[1, :] (second row):", b[1, :])
    print("b[:, 2] (third column):", b[:, 2])
    print("b[0:2, 1:3] (rows 0-1, cols 1-2):\n", b[0:2, 1:3])


def mathematical_and_statistical_operations():
    section_title("2. Mathematical, Axis-wise, and Statistical Operations")

    # Create a sample dataset: 5 rows (samples) x 3 columns (features)
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9],
                     [10, 11, 12],
                     [13, 14, 15]])

    print("Data:\n", data)

    # Element-wise mathematical operations
    print("\nElement-wise operations:")
    print("data + 10:\n", data + 10)
    print("data * 2:\n", data * 2)

    # Axis-wise operations
    print("\nAxis-wise operations:")
    print("Sum of all elements:", np.sum(data))
    print("Sum along axis=0 (column-wise):", np.sum(data, axis=0))
    print("Sum along axis=1 (row-wise):", np.sum(data, axis=1))

    # Statistical operations
    print("\nStatistical operations:")
    print("Mean of all elements:", np.mean(data))
    print("Mean of each column:", np.mean(data, axis=0))
    print("Mean of each row:", np.mean(data, axis=1))
    print("Standard deviation (overall):", np.std(data))
    print("Min of each column:", np.min(data, axis=0))
    print("Max of each column:", np.max(data, axis=0))


def reshaping_and_broadcasting():
    section_title("3. Reshaping and Broadcasting")

    # Reshaping example
    arr = np.arange(1, 13)  # 1 to 12
    print("Original 1D arr:", arr)

    arr_2d = arr.reshape(3, 4)  # 3 rows, 4 columns
    print("Reshaped to 2D (3x4):\n", arr_2d)

    arr_3d = arr.reshape(2, 2, 3)  # 2 blocks of 2x3
    print("Reshaped to 3D (2x2x3):\n", arr_3d)

    # Broadcasting example
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

    print("\nData:\n", data)

    # Suppose this is a 1D array of column-wise means
    col_means = np.mean(data, axis=0)
    print("Column-wise means:", col_means)

    # Broadcasting: subtract column means from each row
    centered = data - col_means
    print("Data after broadcasting (centered by column means):\n", centered)

    # Another broadcasting example:
    # Add a 1D array of weights to each row
    weights = np.array([0.1, 0.2, 0.3])
    weighted = data * weights  # broadcasts weights over rows
    print("Data * weights (broadcasting):\n", weighted)


def save_and_load_arrays():
    section_title("4. Save and Load NumPy Arrays")

    # Create a sample array
    arr = np.random.rand(3, 4)
    print("Array to save:\n", arr)

    # Save as .npy (binary format)
    np.save("my_array.npy", arr)
    print("\nSaved array to 'my_array.npy'")

    # Load .npy file
    loaded_arr = np.load("my_array.npy")
    print("Loaded array from 'my_array.npy':\n", loaded_arr)

    # Save as text file
    np.savetxt("my_array.txt", arr, fmt="%.4f")
    print("\nSaved array to 'my_array.txt'")

    # Load from text file
    loaded_txt_arr = np.loadtxt("my_array.txt")
    print("Loaded array from 'my_array.txt':\n", loaded_txt_arr)


def compare_numpy_vs_python_lists():
    section_title("5. Performance: NumPy vs Python Lists")

    # Create a large list and a large NumPy array
    n = 1_000_000  # 1 million elements
    py_list = list(range(n))
    np_array = np.arange(n)

    # Python list sum using a loop
    start = time.perf_counter()
    total_list = 0
    for x in py_list:
        total_list += x
    end = time.perf_counter()
    time_list = end - start

    # NumPy array sum
    start = time.perf_counter()
    total_np = np.sum(np_array)
    end = time.perf_counter()
    time_np = end - start

    print(f"Python list sum: {total_list}, time: {time_list:.6f} seconds")
    print(f"NumPy array sum: {total_np}, time: {time_np:.6f} seconds")

    if time_np > 0:
        print(f"\nNumPy is approximately {time_list / time_np:.2f}x faster for this operation.")


def main():
    array_creation_indexing_slicing()
    mathematical_and_statistical_operations()
    reshaping_and_broadcasting()
    save_and_load_arrays()
    compare_numpy_vs_python_lists()


if __name__ == "__main__":
    main()

