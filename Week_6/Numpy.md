## What is NumPy?
NumPy, short for Numerical Python, is a fundamental library for numerical and scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of high-level mathematical functions to operate on these arrays. NumPy serves as the foundation for many data science and machine learning libraries, making it an essential tool for data analysis and scientific research in Python.

### Key aspects of NumPy in Python:
Efficient data structures: NumPy introduces efficient array structures, which are faster and more memory-efficient than Python lists. This is crucial for handling large data sets.

### Multi-dimensional arrays: 
NumPy allows you to work with multi-dimensional arrays, enabling the representation of matrices and tensors. This is particularly useful in scientific computing.

### Element-wise operations: 
NumPy simplifies element-wise mathematical operations on arrays, making it easy to perform calculations on entire data sets in one go.

### Random number generation: 
It provides a wide range of functions for generating random numbers and random data, which is useful for simulations and statistical analysis.

### Integration with other libraries: 
NumPy seamlessly integrates with other data science libraries like SciPy, Pandas, and Matplotlib, enhancing its utility in various domains.

### Performance optimization: 
NumPy functions are implemented in low-level languages like C and Fortran, which significantly boosts their performance. It's a go-to choice when speed is essential.

### Installation
If you haven't already installed NumPy, you can do so using pip

### Creating NumPy arrays
You can create NumPy arrays from Python lists. These arrays can be one-dimensional or multi-dimensional.


```python
import numpy as np
# Creating a 1D array
arr = np.array([1, 2, 3, 4, 5]) # **np.array()** is used to create NumPy arrays.
arr_1d
# mylst = [1, 2, 3, 4, 5]
# print(mylst)
arr_1d.shape

```




    (5,)




```python
arr[start:end:skip]
```


```python
arr[0:6:3]
```




    array([1, 4])




```python
arr[-3:]
```




    array([3, 4, 5])




```python
mylstt = [[2,3,1,5], [6,7,8,9], [10,11,12,13], [14,15,16,17]]
# mylstt
arr3 = np.array(mylstt)
# arr3.shape
arr3
```




    array([[ 2,  3,  1,  5],
           [ 6,  7,  8,  9],
           [10, 11, 12, 13],
           [14, 15, 16, 17]])




```python
arr3[2, 1]
```




    11




```python
arr3[1, 0:4:2]
```




    array([6, 8])




```python
arr3[1, ::2]
```




    array([6, 8])




```python
mylstt
```

arr_1d = np.array([1, 2, 3, 4, 5]): In this line, a one-dimensional 
NumPy array named arr_1d is created. It uses the np.array() function to convert a Python list [1, 2, 3, 4, 5] into a NumPy array. This array contains five elements, which are 1, 2, 3, 4, and 5. arr_1d is a 1D array because it has a single row of elements.


```python
# Creating a 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
```

arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]): 

In this line, a two-dimensional NumPy array named arr_2d is created. It uses the np.array() function to convert a list of lists into a 2D NumPy array.

The outer list contains three inner lists, each of which represents a row of elements. So, arr_2d is a 2D array with three rows and three columns. The elements in this array form a matrix with values from 1 to 9, organized in a 3x3 grid.


```python
# Array attributes
print(arr_2d.ndim)  # ndim : Represents the number of dimensions or "rank" of the array.
# output : 2
print(arr_2d.shape)  # shape : Returns a tuple indicating the number of rows and columns in the array.
# Output : (3, 3)
print(arr_2d.size) # size: Provides the total number of elements in the array.
# Output : 9
```

    2
    (3, 3)
    9
    


```python

```

### Indexing and slicing
You can access elements of a NumPy array using indexing and slicing:

In this line, the third element (index 2) of the 1D array arr_1d is accessed.


```python
# Indexing and slicing
print(arr_1d[-1])          # Accessing an element (3rd element)
```

    5
    

In this line, the element in the 2nd row (index 1) and 3rd column (index 2) of the 2D array arr_2d is accessed.


```python
print(arr_2d[1, 2])       # Accessing an element (2nd row, 3rd column)
```

In this line, the element in the 2nd row (index 1) and 3rd column (index 2) of the 2D array arr_2d is accessed.


```python
print(arr_2d[1, 2])       # Accessing an element (2nd row, 3rd column)
```

In this line, the 2nd row (index 1) of the 2D array arr_2d is accessed.


```python
print(arr_2d[1])          # Accessing a row (2nd row)
```

In this line, the 2nd column (index 1) of the 2D array arr_2d is accessed.


```python
print(arr_2d[:, 1])       # Accessing a column (2nd column)
```

### Basic operations
NumPy simplifies basic operations on arrays:

### Element-wise arithmetic operations:
Addition, subtraction, multiplication, and division of arrays with scalars or other arrays.


```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]
print(list1 + list2)
```

    [1, 2, 3, 4, 5, 6]
    


```python
# Array addition
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
result = array1 + array2
print(result)  # [5 7 9]
```

    [5 7 9]
    


```python
result + 12
```




    array([17, 19, 21])




```python
# Scalar multiplication
array = np.array([1, 2, 3])
result = array * 2 # each element of an array is multiplied by 2
print(result)  # [2 4 6]
```

    [2 4 6]
    


```python
# Element-wise multiplication (Hadamard product)
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
result = array1 * array2
print(result)  # [4 10 18]
```

    [ 4 10 18]
    


```python
# Matrix multiplication
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
result = np.dot(matrix1, matrix2)
print(result)
# print(matrix1)
# print(matrix2)
# [[19 22]

#  [43 50]]
```

    [[19 22]
     [43 50]]
    

![alt text](operations.JPG)