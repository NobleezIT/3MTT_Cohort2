# Pyton Functions
Python Functions is a block of statements that return the specific task. The idea is to put some commonly or repeatedly done tasks together and make a function so that instead of writing the same code again and again for different inputs, we can do the function calls to reuse code contained in it over and over again.

## Types of Functions in Python
Below are the different types of functions in Python:

- Built-in library function: These are Standard functions in Python that are available to use.
- User-defined function: We can create our own functions based on our requirements.

## Built-in Functions
Python provides a lot of built-in functions that ease the writing of code.


```python
# t# Examples of Builtin functions

# print()
# len()
# type()
# sum()
# # Built-in Functions
# list()
# dict()
# tuple()
# set()

```

## Creating a Function in Python
We can define a function in Python, using the def keyword. We can add any type of functionalities and properties to it as we require. By the following example, we can understand how to write a function in Python. In this way we can create Python function definition by using def keyword.

### function syntax
def function_name(parameters):

        statement

        return expression


```python
def greet():
    print("Hi! Welcome")

```

### Calling a Function in Python
After creating a function in Python we can call it by using the name of the functions Python followed by parenthesis containing parameters of that particular function. Below is the example for calling def function Python.


```python
# Calling the greet function
greet()
```

    Hi! Welcome
    

## Python Function with Parameters
If you have experience in C/C++ or Java then you must be thinking about the return type of the function and data type of arguments. That is possible in Python as well (specifically for Python 3.5 and above).

# Python function Syntax with Parameters
def function_name(parameter: data_type) -> 
    
    return_type:

        """Docstring"""
        # body of the function
        return expression


```python
def add(num1, num2):
    num3 = num1 + num2
    return num3
```


```python
add(2, 3)
```




    5




```python
def power(x, y):
    result = x ** y
    return result
```


```python
power(2, 7)
```




    128



## Python Function Arguments
Arguments are the values passed inside the parenthesis of the function. A function can have any number of arguments separated by a comma.

In this example, we will create a simple function in Python to check whether the number passed as an argument to the function is even or odd.


```python
# A simple Python function to check
# whether x is even or odd
def evenOdd(x):
    if (x % 2 == 0):
        print("even")
    else:
        print("odd")

```

Types of Python Function Arguments
Python supports various types of arguments that can be passed at the time of the function call. In Python, we have the following function argument types in Python:

- Default argument
- Keyword arguments (named arguments)
- Positional arguments
- Arbitrary arguments (variable-length arguments *args and **kwargs)

### Default Arguments
A default argument is a parameter that assumes a default value if a value is not provided in the function call for that argument. The following example illustrates Default arguments to write functions in Python.


```python
# Python program to demonstrate
# default arguments
def myFun(x, y=50):
    print("x: ", x)
    print("y: ", y)
```

### Keyword Arguments
The idea is to allow the caller to specify the argument name with values so that the caller does not need to remember the order of parameters.

# Positional Arguments
We used the Position argument during the function call so that the first argument (or value) is assigned to name and the second argument (or value) is assigned to age. By changing the position, or if you forget the order of the positions, the values can be used in the wrong places, as shown in the Case-2 example below, where 27 is assigned to the name and Suraj is assigned to the age.


```python
def nameAge(name, age):
    print("Hi, I am", name)
    print("My age is ", age)

```

### Arbitrary Keyword  Arguments
In Python Arbitrary Keyword Arguments, *args, and **kwargs can pass a variable number of arguments to a function using special symbols. There are two special symbols:

- args in Python (Non-Keyword Arguments)
- kwargs in Python (Keyword Arguments)


```python
# Python program to illustrate
# *args for variable number of arguments
def myFun(*argv):
    for arg in argv:
        print(arg)


myFun('Hello', 'Welcome', 'to', 'Python')
```

    Hello
    Welcome
    to
    Python
    


```python
# Python program to illustrate
# *kwargs for variable number of keyword arguments


def myFun(**kwargs):
    for key, value in kwargs.items():
        print("%s == %s" % (key, value))


# Driver code
myFun(first='New', mid='to', last='Python')
```

    first == New
    mid == to
    last == Python
    

### Docstring
The first string after the function is called the Document string or Docstring in short. This is used to describe the functionality of the function. The use of docstring in functions is optional but it is considered a good practice.

The below syntax can be used to print out the docstring of a function.

Syntax: print(function_name.__doc__)


```python
# A simple Python function to check
# whether x is even or odd


def evenOdd(x):
    """Function to check if the number is even or odd"""
    
    if (x % 2 == 0):
        print("even")
    else:
        print("odd")


# Driver code to call the function
print(evenOdd.__doc__)
```

    Function to check if the number is even or odd
    

## Use case for Python Functions
#### Grading student result:
Using python, you can easily define a function that can be used to grade students exams score as shown below:


```python
# Defining a function called determine grade.
def determine_grade(score):
    if score >= 70:
        return 'A'
    elif score >= 60:
        return 'B'
    elif score >= 50:
        return 'C'
    elif score >= 45:
        return 'D'
    elif score >= 39:
        return 'E'
    else:
        return 'F'
```


```python
# Calling determine grade on a particular score
determine_grade(65)
```




    'B'



This function can be used on a dataframe by creating a new column for the grade and appplying the function




```python
# import pandas for the dataframe
import pandas as pd
```


```python
df = pd.read_csv(r"C:\Users\abula\Documents\report.csv")
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Student Name</th>
      <th>Exam Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John Smith</td>
      <td>85</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mary Johnson</td>
      <td>72</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adam Brown</td>
      <td>67</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sara Davis</td>
      <td>93</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mike Wilson</td>
      <td>78</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Emily Moore</td>
      <td>61</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Mark Taylor</td>
      <td>87</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Laura Clark</td>
      <td>79</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Alex Parker</td>
      <td>95</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Olivia Adams</td>
      <td>82</td>
    </tr>
  </tbody>
</table>
</div>




```python
# cleaning the data
df['exam_score'] =  df["Exam Score"]
df["student_name"] = df["Student Name"]
```


```python
df = df[["student_name", "exam_score"]]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>student_name</th>
      <th>exam_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John Smith</td>
      <td>85</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mary Johnson</td>
      <td>72</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adam Brown</td>
      <td>67</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sara Davis</td>
      <td>93</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mike Wilson</td>
      <td>78</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Emily Moore</td>
      <td>61</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Mark Taylor</td>
      <td>87</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Laura Clark</td>
      <td>79</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Alex Parker</td>
      <td>95</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Olivia Adams</td>
      <td>82</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["grade"] = df["exam_score"].apply(determine_grade)
```

    C:\Users\abula\AppData\Local\Temp\ipykernel_16264\1005977092.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df["grade"] = df["exam_score"].apply(determine_grade)
    


```python
print(df)
```

       student_name  exam_score grade
    0    John Smith          85     A
    1  Mary Johnson          72     A
    2    Adam Brown          67     B
    3    Sara Davis          93     A
    4   Mike Wilson          78     A
    5   Emily Moore          61     B
    6   Mark Taylor          87     A
    7   Laura Clark          79     A
    8   Alex Parker          95     A
    9  Olivia Adams          82     A
    


```python
def evenodd(x):
    if (x % 2 == 0):
        print("even")
    else:
        print("odd")
```

### Return Statement in Python Function
The function return statement is used to exit from a function and go back to the function caller and return the specified value or data item to the caller. The syntax for the return statement is:

return [expression_list]    

The return statement can consist of a variable, an expression, or a constant which is returned at the end of the function execution. If none of the above is present with the return statement a None object is returned.

Example: Python Function Return Statement


```python
def square_value(num):
    """This function returns the square
    value of the entered number"""
    return num**2


print(square_value(2))
print(square_value(-4))
```

### Pass by Reference and Pass by Value
One important thing to note is, in Python every variable name is a reference. When we pass a variable to a function Python, a new reference to the object is created. 


```python
# Here x is a new reference to same list lst
def mytest(x):
    x[0] = 20


# Driver Code (Note that lst is modified
# after function call.
lst = [10, 11, 12, 13, 14, 15]
mytest(lst)
print(lst)

```

    [20, 11, 12, 13, 14, 15]
    

# Exercise
1. Create a function that takes any number as the argument and create a multiplication table from 1 to 12 for that particular number.


```python

```
