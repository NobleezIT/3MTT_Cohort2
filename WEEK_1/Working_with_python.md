**Python Shell**

Python is an interpreted scripting language, so it does not need to be compiled. It means it executes the code line by line. Python comes with a Python Shell (Python Interactive Shell). It is used to execute a single python command and get the result.

Python Shell waits for the Python code from the user. When you enter the code, it interprets the code and shows the result in the next line. Open your terminal or command prompt(cmd) and write:

```
python
```

**Python Syntax**

A Python script can be written in Python interactive shell or in the code editor. A Python file has an extension .py. While a notebook file created with jupyter notebook has an extension .ipynb.

**Python Indentation**

An indentation is a white space in a text. Indentation in many languages is used to increase code readability, however Python uses indentation to create block of codes. In other programming languages curly brackets are used to create blocks of codes instead of indentation. One of the common bugs when writing python code is wrong indentation.

**Python Syntax**

A Python script can be written in Python interactive shell or in the code editor. A Python file has an extension .py. While a notebook file created with jupyter notebook has an extension .ipynb.

**Python Indentation**

An indentation is a white space in a text. Indentation in many languages is used to increase code readability, however Python uses indentation to create block of codes. In other programming languages curly brackets are used to create blocks of codes instead of indentation. One of the common bugs when writing python code is wrong indentation.

+addition

-subtraction

*multiplication

/ division

// floor division

%  modulo

** exponential

Trying running this code in a python shell

2 + 3 = 5

3 - 2 = 1

3 * 2 = 6

3 / 2 = 1.5

3 // 2 = 1

3 % 2 = 1

Comments

Comments are very important to make the code more readable and to leave remarks in our code. Python does not run comment parts of our code. Any text starting with hash(#) in Python is a comment.

Example: Single Line Comment

    # This is the first comment
    # This is the second comment
    # Python is eating the world

Example: Multiline Comment

Triple quote can be used for multiline comment if it is not assigned to a variable

"""This is multiline comment
multiline comment takes multiple lines.
python is eating the world
"""

Data types

In Python there are several types of data types. Let us get started with the most common ones. Different data types will be covered in detail in other sections. For the time being, let us just go through the different data types and get familiar with them. You do not have to have a clear understanding now.

**Number**

There are three numeric types in Python:

Integer: Int, or integer, is a whole number, positive or negative, without decimals, of unlimited length.

Example: -2, -4, 0, 5, 6

**Float**

Float, or "floating point number" is a number, positive or negative, containing one or more decimals.

-3.5, -2.25, -1.0, 0.0, 1.1, 2.2, 3.5 ...

**Complex** 

Complex numbers are written with a "j" as the imaginary part:

Example 1 + j, 2 + 4j

**String**

A collection of one or more characters under a single or double quote. If a string is more than one sentence then we use a triple quote.

Example:

'Musa'
'Kano'
'Python'
'I love teaching'
'3MTT Week One Physical Training'

**Booleans**

A boolean data type is either a True or False value. T and F should be always uppercase.

Example:

    True  #  Is the light on? If it is on, then the value is True
    False # Is the light on? If it is off, then the value is False

**List**

Python list is an ordered collection which allows to store different data type items. A list is similar to an array in 

JavaScript.

Example:

[0, 1, 2, 3, 4, 5]  # all are the same data types - a list of numbers

['Banana', 'Orange', 'Mango', 'Avocado'] # all the same data types - a list of strings (fruits)

['Nigeria','Niger', 'Egypt','Namibia'] # all the same data types - a list of strings (countries)

['Banana', 10, False, 9.81] # different data types in the list - string, integer, boolean and float

**Dictionary**

A Python dictionary object is an unordered collection of data in a key value pair format.

Example:

{
'first_name':'Musa',
'last_name':'Alameen',
'country':'Kenya', 
'age':25, 
'is_married':True,
'skills':['JS', 'React', 'Node', 'Python']
}

**Tuple**

A tuple is an ordered collection of different data types like list but tuples can not be modified once they are created. They are immutable.

Example:

('Musa', 'Alameen', 'Jameel', 'Lawal', 'Sani') # Names

('Earth', 'Jupiter', 'Neptune', 'Mars', 'Venus', 'Saturn', 'Uranus', 'Mercury') # planets

**Set**

A set is a collection of data types similar to list and tuple. Unlike list and tuple, set is not an ordered collection of items. Like in Mathematics, set in Python stores only unique items.

In later sections, we will go in detail about each and every Python data type.

Example:

{2, 4, 3, 5}
{3.14, 9.81, 2.7} # order is not important in set

**Checking Data types**

To check the data type of certain data/variable we use the type function. In the following terminal you will see different python data types:


## **Week_1 Exercise**

1. Create a new Jupyter notebook and do the following operations using the numbers 5 and 3.

addition(+)

subtraction(-)

multiplication(*)

modulus(%)

division(/)

exponential(**)

floor division operator(//)

2. Write strings on the python interactive shell. The strings are the following:

Your name

Your family name

Your country

3. Write an example for different Python data types such as Number(Integer, Float, Complex), String, Boolean, List, Tuple, Set and Dictionary.

Proceed to >> [Week 2] :blush: