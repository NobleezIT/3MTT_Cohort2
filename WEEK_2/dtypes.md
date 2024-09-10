# Data Types

## Introduction

In programming, data type is an important concept. *Variables* can store data of different types, and different types can do different things.

But before getting into data types, what are variables?

**Variables** are containers for storing data values.

### Creating Variables
Python has no command for declaring a variable.

A variable is created the moment you first assign a value to it.

### Variable Names
A variable can have a short name (like x and y) or a more descriptive name (age, carname, total_volume). Rules for Python variables:
A variable name must start with a letter or the underscore character
A variable name cannot start with a number
A variable name can only contain alpha-numeric characters and underscores (A-z, 0-9, and _ )
Variable names are case-sensitive (age, Age and AGE are three different variables)
A variable name cannot be any of the Python keywords.


```python
# Examples of acceptable variable names
myvar = "Abdul"
my_var = "Abdul"
_my_var = "Abdul"
myVar = "Abdul"
MYVAR = "Abdul"
myvar2 = "Abdul"
```


```python
# Examples of unacceptable variable names
2myvar = "Abdul"
my-var = "Abdul"
my var = "Abdul"
```

### Multi Words Variable Names
Variable names with more than one word can be difficult to read.

There are several techniques you can use to make them more readable:

### Camel Case
Each word, except the first, starts with a capital letter:


```python
myVariableName = "Abdul"
```

### Snake Case
Each word is separated by an underscore character:


```python
my_variable_name = "Abdul"
```

### Many Values to Multiple Variables
Python allows you to assign values to multiple variables in one line:


```python
x, y, z = "Orange", "Banana", "Cherry"
print(x)
print(y)
print(z)
```

Python has the following data types built-in by default, in these categories:

Text Type:	str
Numeric Types:	int, float, complex
Sequence Types:	list, tuple, range
Mapping Type:	dict
Set Types:	set, frozenset
Boolean Type:	bool
Binary Types:	bytes, bytearray, memoryview
None Type:	NoneType


```python
# Examples
x = 5
y = "Aliyu"
print(x)
print(y)
```

    5
    Aliyu
    

Variables do not need to be declared with any particular type, and can even change type after they have been set.


```python
x = 4       # x is of type int
x = "Habeeb" # x is now of type str
print(x)
```

    Habeeb
    

### Case-Sensitive
Variable names are case-sensitive. Meaning the variable 'A' for instance is different from the variable 'a'.


```python
# Example
a = 25
A = 'Khalid'
print(a)
print(A)
```

    25
    Khalid
    

'A' will not overwrite 'a'.

Python has the following data types built-in by default, in these categories:

- Strings
- Integers
- Floats
- Booleans
- List
- Tuple
- Set
- Dictionary

### Getting the Data Type
You can get the data type of any object by using the type() function:


```python
y = "orange"
print(type(y))
```

    <class 'str'>
    


```python

```


```python
# Strings
print('Hello World')
```

    Hello World
    


```python
greet = 'Hello World'
```

string concatination

Python Numbers

There are three numeric types in Python:

- int
- float
- complex

Variables of numeric types are created when you assign a value to them:


```python
# Examples
x = 1    # int
y = 2.8  # float
z = 1j   # complex
```

To verify the type of any object in Python, use the type() function:


```python
print(type(x))
print(type(y))
print(type(z))
```

Int

Int, or integer, is a whole number, positive or negative, without decimals, of unlimited length.


```python
x = 1
y = 35656222554887711
z = -3255522

print(type(x))
print(type(y))
print(type(z))

```

Float

Float, or "floating point number" is a number, positive or negative, containing one or more decimals.


```python
x = 1.10
y = 3.4
z = -5.9

print(type(x))
print(type(y))
print(type(z))
```

    <class 'float'>
    <class 'float'>
    <class 'float'>
    

### Complex
Complex numbers are written with a "j" as the imaginary part:



```python
x = 3+5j
y = 5j
z = -5j

print(type(x))
print(type(y))
print(type(z))

```

### Type Conversion
You can convert from one type to another with the int(), float(), and complex() methods:


```python
# Example Conversion
x = 1    # int
y = 2.8  # float
z = 1j   # complex

#convert from int to float:
a = float(x)

#convert from float to int:
b = int(y)

#convert from int to complex:
c = complex(x)

print(a)
print(b)
print(c)

print(type(a))
print(type(b))
print(type(c))
```

### Strings
Strings in python are surrounded by either single quotation marks, or double quotation marks.

'hello' is the same as "hello".

You can display a string literal with the print() function:


```python
'single quote'
```




    'single quote'




```python
print('hello')
```

    hello
    


```python
"double quote"
```




    'double quote'




```python
# Strings
name = "My name is Abdulazeez "
address = "I live in Naibawa"
my_intro = name + address
my_intro
```


```python
"Double Quote"

```




    'Double Quote'




```python
multiline = """
The ice cream vanquished
my longing for sweets,
upon this diet I look away,
it no longer exists on this day.

"""
print(multiline)
```

    
    The ice cream vanquished
    my longing for sweets,
    upon this diet I look away,
    it no longer exists on this day.
    
    
    


```python
type(multiline)
```




    str



### String Length
To get the length of a string, use the len() function.


```python
a = "Hello, World!"
print(len(a))
```

Check String

To check if a certain phrase or character is present in a string, we can use the keyword in.

Check if "free" is present in the following text:


```python
txt = "The best things in life are free!"
print("free" in txt)
```

Index


```python
greet.find('world')
```




    -1




```python
greet.find(' ')
```




    5




```python
greet.find('o')
```




    4




```python
greet
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 1
    ----> 1 greet
    

    NameError: name 'greet' is not defined



```python
print(len(greet)) # length
```

    11
    


```python
greet.upper()
```




    'HELLO WORLD'




```python
greet.lower()
```




    'hello world'




```python
greet.title()
```




    'Hello World'




```python
price = "$12348"
```


```python
price.replace('$', '#')
```




    '#12348'




```python
type(price)
```




    str



#


```python
price.replace
```


```python
type(price)
```




    str




```python
print(type(greet))
```

    <class 'str'>
    

Accessing Characters in Strings by Index

In programming counting starts from zero. Therefore the first letter of a string is at zero index and the last letter of a string is the length of a string minus one.

String index


```python
language = 'Python'
first_letter = language[0]
print(first_letter) # P
second_letter = language[1]
print(second_letter) # y
last_index = len(language) - 1
last_letter = language[last_index]
print(last_letter) # n
```

If we want to start from right end we can use negative indexing. -1 is the last index.


```python
language = 'Python'
last_letter = language[-1]
print(last_letter) # n
second_last = language[-2]
print(second_last) # o
```

## Slicing Python Strings

In python we can slice strings into substrings.


```python
language = 'Python'
first_three = language[0:3] # starts at zero index and up to 3 but not include 3
print(first_three) #Pyt
last_three = language[3:6]
print(last_three) # hon
# Another way
last_three = language[-3:]
print(last_three)   # hon
last_three = language[3:]
print(last_three)   # hon
```

## Reversing a String

We can easily reverse strings in python.


```python
greeting = 'Hello, World!'
print(greeting[::-1]) # !dlroW ,olleH
```

## Skipping Characters While Slicing

It is possible to skip characters while slicing by passing step argument to slice method.



```python
language = 'Python'
pto = language[0:6:2] #
print(pto) # Pto
```

## String Methods

There are many string methods which allow us to format strings. See some of the string methods in the following example:

capitalize(): Converts the first character of the string to capital letter


```python
challenge = 'thirty days of python'
print(challenge.capitalize()) # 'Thirty days of python'
```

count(): returns occurrences of substring in string, count(substring, start=.., end=..). The start is a starting indexing for counting and end is the last index to count.


```python
challenge = 'thirty days of python'
print(challenge.count('y')) # 3
print(challenge.count('y', 7, 14)) # 1, 
print(challenge.count('th')) # 2`
```

You can access more string methods [here](https://www.w3schools.com/python/python_strings_methods.asp)


```python
greet[:3]
```




    'Hel'




```python
greet.replace('worl', 'porld')
```




    'Hello World'




```python
gf = "$ 2345"
```


```python
gf.strip('$ ')
```




    '2345'




```python
print(gf.replace('$', '#'))
```

    # 2345
    


```python
price = 345
```


```python
type(price)
```




    int




```python
ttt = f"the price for the journey is {price} Naira"
ttt
```




    'the price for the journey is 345 Naira'




```python
price = 40

txt = f"The price is {price:.2f} dollars"
txt
```




    'The price is 40.00 dollars'




```python
txtx = f"it is very {'Expensive' if price > 50 else 'cheap'}"
txtx
```




    'it is very cheap'



Boolean Values

In programming you often need to know if an expression is True or False.

You can evaluate any expression in Python, and get one of two answers, True or False.

When you compare two values, the expression is evaluated and Python returns the Boolean answer:


```python
print(3 == 5)
```

    False
    


```python
print(3 == 3)
```

    True
    


```python
a =  4
b = 7
print(a > b)
```

    False
    


```python
type(a > b)
```




    bool



## Python Collections (Arrays)
There are four collection data types in the Python programming language:

**List** is a collection which is ordered and changeable. Allows duplicate members.

**Tuple** is a collection which is ordered and unchangeable. Allows duplicate members.

**Set** is a collection which is unordered, unchangeable*, and unindexed. No duplicate members.

**Dictionary** is a collection which is ordered** and changeable. No duplicate members.

## [ ]- Square Bracket  { }- Curly Bracket  ( ) - Parenthesis

## List

Lists are used to store multiple items in a single variable.

List Items
List items are ordered, changeable, and allow duplicate values.

List items are indexed, the first item has index [0], the second item has index [1] etc.

Ordered
When we say that lists are ordered, it means that the items have a defined order, and that order will not change.

If you add new items to a list, the new items will be placed at the end of the list.

Note: There are some list methods that will change the order, but in general: the order of the items will not change.

### Changeable
The list is changeable, meaning that we can change, add, and remove items in a list after it has been created.

### Allow Duplicates
Since lists are indexed, lists can have items with the same value:

Example
Lists allow duplicate values:


```python
thislist = ["apple", "banana", "cherry", "apple", "cherry"]
print(thislist)
```

List Items - Data Types
List items can be of any data type:


```python
# Example
# String, int and boolean data types:

list1 = ["apple", "banana", "cherry"]
list2 = [1, 5, 7, 9, 3]
list3 = [True, False, False]
```

A list can contain different data types:

Example
A list with strings, integers and boolean values:


```python
list1 = ["abc", 34, True, 40, "male"]
```


```python
x = ['banana', 'mango', 'orange']
```


```python
type(x)
```




    list



list()


```python
y = list(('car', 'bike', 'bicycle'))
```


```python
y[2]
```




    'bicycle'




```python
thislist = ['Kano', 'Jigawa', 'Katsina', 'Kaduna']
```


```python
type(thislist)
```




    list




```python
mylist = ['Bashir', 'Zainab', 'Binta', 'Aminu']
```


```python
mylist[-3] 
```




    'Zainab'




```python
b
```




    'Zainab'




```python

```

Access list items using index

List items are indexed and you can access them by referring to the index number:

mylist[2]


```python
thislist = ["apple", "banana", "cherry"]
print(thislist[1])

```

Negative Indexing
Negative indexing means start from the end

-1 refers to the last item, -2 refers to the second last item etc.


```python
# Print the last item of the list:

thislist = ["apple", "banana", "cherry"]
print(thislist[-1])
```

Range of Indexes
You can specify a range of indexes by specifying where to start and where to end the range.

When specifying a range, the return value will be a new list with the specified items.


```python
Example
Return the third, fourth, and fifth item:

thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
print(thislist[2:5])
```

By leaving out the start value, the range will start at the first item:

Example
This example returns the items from the beginning to, but NOT including, "kiwi":



```python
thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
print(thislist[:4])
```

By leaving out the end value, the range will go on to the end of the list:


```python
# This example returns the items from "cherry" to the end:

thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
print(thislist[2:])
```

Range of Negative Indexes
Specify negative indexes if you want to start the search from the end of the list:



```python
# This example returns the items from "orange" (-4) to, 
# But NOT including "mango" (-1):
thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
print(thislist[-4:-1])
```

change list items using index


```python
x
```




    ['banana', 'mango', 'orange']




```python
x[-1] = 'grapes'
x
```




    ['banana', 'grapes', 'grapes']




```python
x
```




    ['banana', 'grapes', 'orange']



add list items using append(), extend(), insert()


```python
x[-1] = 'orange'
```


```python
x.append('pineapple')
x
```




    ['banana', 'grapes', 'orange', 'pineapple', 'pineapple']




```python
x.append('apple')
x
```




    ['banana', 'grapes', 'orange', 'pineapple', 'pineapple', 'apple', 'apple']




```python
thislist
```




    ['Kano', 'Jigawa', 'Katsina', 'Kaduna']




```python
thislist.extend('Kebbi')
thislist
```




    ['Kano',
     'Jigawa',
     'Katsina',
     'Kaduna',
     'K',
     'e',
     'b',
     'b',
     'i',
     'K',
     'e',
     'b',
     'b',
     'i']



remove list items using remove(), del list(), pop()


```python
x.remove('banana')
```


```python
x
```




    ['grapes', 'orange', 'pineapple', 'pineapple', 'apple', 'apple']




```python
del x[2]
x
```




    ['grapes', 'orange', 'apple', 'apple']




```python
thislist = ['apple', 'banana', 'mango']
for i in thislist:
    print(i)
```

    apple
    banana
    mango
    


```python
for i in range(len(thislist)):
    print(thislist[i])
```

    apple
    banana
    mango
    


```python
i = 0
while i < len(thislist):
    print(thislist[i])
    i = i + 1
```

    apple
    banana
    mango
    


```python
[print(i) for i in thislist]
```

    apple
    banana
    mango
    




    [None, None, None]



## Tuple

Tuples are used to store multiple items in a single variable.

Tuple is one of 4 built-in data types in Python used to store collections of data, the other 3 are List, Set, and Dictionary, all with different qualities and usage.

A tuple is a collection which is ordered and unchangeable.

Tuples are written with round brackets.

Tuple Items

Tuple items are ordered, unchangeable, and allow duplicate values.

Tuple items are indexed, the first item has index [0], the second item has index [1] etc.

Ordered

When we say that tuples are ordered, it means that the items have a defined order, and that order will not change.

Unchangeable

Tuples are unchangeable, meaning that we cannot change, add or remove items after the tuple has been created.

Allow Duplicates

Since tuples are indexed, they can have items with the same value:


```python
thistuple = ("apple", "banana", "cherry", "apple", "cherry")
print(thistuple)
```

    ('apple', 'banana', 'cherry', 'apple', 'cherry')
    

Access Tuple Items

You can access tuple items by referring to the index number, inside square brackets:

index
range of inde
range of negative index


```python
thistuple[1] = 'apple'
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[96], line 1
    ----> 1 thistuple[1] = 'apple'
    

    TypeError: 'tuple' object does not support item assignment



```python
newlist = list(thistuple)
```


```python
newlist
```




    ['apple', 'banana', 'cherry', 'apple', 'cherry']




```python
newlist[1] = 'strawberry'
```


```python
thistuple = newlist
thistuple
```




    ['apple', 'strawberry', 'cherry', 'apple', 'cherry']



### Update tuple

Tuples are unchangeable, meaning that you cannot change, add, or remove items once the tuple is created.

But there are some workarounds.

Change Tuple Values

Once a tuple is created, you cannot change its values. Tuples are unchangeable, or immutable as it also is called.

But there is a workaround. You can convert the tuple into a list, change the list, and convert the list back into a tuple.

### Add Items

Since tuples are immutable, they do not have a built-in append() method, but there are other ways to add items to a tuple.

1. Convert into a list: Just like the workaround for changing a tuple, you can convert it into a list, add your item(s), and convert it back into a tuple.

2. Add tuple to a tuple. You are allowed to add tuples to tuples, so if you want to add one item, (or many), create a new tuple with the item(s), and add it to the existing tuple:

Remove Items

Tuples are unchangeable, so you cannot remove items from it, but you can use the same workaround as we used for changing and adding tuple items:


Unpacking a Tuple
When we create a tuple, we normally assign values to it. This is called "packing" a tuple

But, in Python, we are also allowed to extract the values back into variables. This is called "unpacking":

## Set

Sets are used to store multiple items in a single variable.
A set is a collection which is unordered, unchangeable*, and unindexed.


Unordered

Unordered means that the items in a set do not have a defined order.

Set items can appear in a different order every time you use them, and cannot be referred to by index or key.

Unchangeable
Set items are unchangeable, meaning that we cannot change the items after the set has been created.

Once a set is created, you cannot change its items, but you can remove items and add new items.

Duplicates Not Allowed
Sets cannot have two items with the same value.

Access Items
You cannot access items in a set by referring to an index or a key.

But you can loop through the set items using a for loop, or ask if a specified value is present in a set, by using the in keyword.

Change Items
Once a set is created, you cannot change its items, but you can add new items.

Add Items
Once a set is created, you cannot change its items, but you can add new items.

To add one item to a set use the add() method.


```python
thisset = {"apple", "banana", "cherry"}

thisset.add("orange")

print(thisset)
```

    {'orange', 'banana', 'apple', 'cherry'}
    

Add Sets
To add items from another set into the current set, use the update() method.


```python
thisset = {"apple", "banana", "cherry"}
tropical = {"pineapple", "mango", "papaya"}

thisset.update(tropical)

print(thisset)
```

Add Any Iterable
The object in the update() method does not have to be a set, it can be any iterable object (tuples, lists, dictionaries etc.).

Remove Item

To remove an item in a set, use the remove(), or the discard() method.


```python
thisset = {"apple", "banana", "cherry"}

thisset.remove("banana")

print(thisset)
```

The del keyword will delete the set completely:




```python
thisset = {"apple", "banana", "cherry"}

del thisset

print(thisset)
```



## Dictionary

Dictionaries are used to store data values in key:value pairs.

A dictionary is a collection which is ordered*, changeable and do not allow duplicates.


```python
student_info = {'Name': 'Abdul', 'Age': 32}
student_info
```




    {'Name': 'Abdul', 'Age': 32}




```python
student_info['Name']
```




    'Abdul'




```python
student_info['Age']
```




    32




```python
student_info.values()
```




    dict_values(['Abdul', 32])




```python
students = {'Name': ['Zainab', 'Isa', 'Musa', 'Nura'], 'Age': [112, 234, 456, 543], 'Country': ['Nigeria', 'Ghana', 'Cameroon', 'Niger']}
students
```




    {'Name': ['Zainab', 'Isa', 'Musa', 'Nura'],
     'Age': [112, 234, 456, 543],
     'Country': ['Nigeria', 'Ghana', 'Cameroon', 'Niger']}




```python
import pandas as pd
```


```python
import matplotlib.pyplot as plt
```


```python
students_df = pd.DataFrame(students)

```


```python
students_df
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
      <th>Name</th>
      <th>Age</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Zainab</td>
      <td>112</td>
      <td>Nigeria</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Isa</td>
      <td>234</td>
      <td>Ghana</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Musa</td>
      <td>456</td>
      <td>Cameroon</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nura</td>
      <td>543</td>
      <td>Niger</td>
    </tr>
  </tbody>
</table>
</div>




```python
students_df['Name']
```




    0    Zainab
    1       Isa
    2      Musa
    3      Nura
    Name: Name, dtype: object




```python
new_students_df = students_df[['Name', 'Country']]
new_students_df
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
      <th>Name</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Zainab</td>
      <td>Nigeria</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Isa</td>
      <td>Ghana</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Musa</td>
      <td>Cameroon</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nura</td>
      <td>Niger</td>
    </tr>
  </tbody>
</table>
</div>



1. Dictionary containing name, age country
convert to a pandas dataframe
display


```python
df = pandas.read_csv('automobileEDA.csv')
```


```python
print(df)
```

         symboling  normalized-losses         make aspiration num-of-doors  \
    0            3                122  alfa-romero        std          two   
    1            3                122  alfa-romero        std          two   
    2            1                122  alfa-romero        std          two   
    3            2                164         audi        std         four   
    4            2                164         audi        std         four   
    ..         ...                ...          ...        ...          ...   
    196         -1                 95        volvo        std         four   
    197         -1                 95        volvo      turbo         four   
    198         -1                 95        volvo        std         four   
    199         -1                 95        volvo      turbo         four   
    200         -1                 95        volvo      turbo         four   
    
          body-style drive-wheels engine-location  wheel-base    length  ...  \
    0    convertible          rwd           front        88.6  0.811148  ...   
    1    convertible          rwd           front        88.6  0.811148  ...   
    2      hatchback          rwd           front        94.5  0.822681  ...   
    3          sedan          fwd           front        99.8  0.848630  ...   
    4          sedan          4wd           front        99.4  0.848630  ...   
    ..           ...          ...             ...         ...       ...  ...   
    196        sedan          rwd           front       109.1  0.907256  ...   
    197        sedan          rwd           front       109.1  0.907256  ...   
    198        sedan          rwd           front       109.1  0.907256  ...   
    199        sedan          rwd           front       109.1  0.907256  ...   
    200        sedan          rwd           front       109.1  0.907256  ...   
    
         compression-ratio  horsepower  peak-rpm city-mpg highway-mpg    price  \
    0                  9.0       111.0    5000.0       21          27  13495.0   
    1                  9.0       111.0    5000.0       21          27  16500.0   
    2                  9.0       154.0    5000.0       19          26  16500.0   
    3                 10.0       102.0    5500.0       24          30  13950.0   
    4                  8.0       115.0    5500.0       18          22  17450.0   
    ..                 ...         ...       ...      ...         ...      ...   
    196                9.5       114.0    5400.0       23          28  16845.0   
    197                8.7       160.0    5300.0       19          25  19045.0   
    198                8.8       134.0    5500.0       18          23  21485.0   
    199               23.0       106.0    4800.0       26          27  22470.0   
    200                9.5       114.0    5400.0       19          25  22625.0   
    
        city-L/100km  horsepower-binned  diesel  gas  
    0      11.190476             Medium       0    1  
    1      11.190476             Medium       0    1  
    2      12.368421             Medium       0    1  
    3       9.791667             Medium       0    1  
    4      13.055556             Medium       0    1  
    ..           ...                ...     ...  ...  
    196    10.217391             Medium       0    1  
    197    12.368421               High       0    1  
    198    13.055556             Medium       0    1  
    199     9.038462             Medium       1    0  
    200    12.368421             Medium       0    1  
    
    [201 rows x 29 columns]
    


```python
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(thisdict)
```

    {'brand': 'Ford', 'model': 'Mustang', 'year': 1964}
    

Dictionary Items

Dictionary items are ordered, changeable, and do not allow duplicates.

Dictionary items are presented in key:value pairs, and can be referred to by using the key name.

Ordered or Unordered?

As of Python version 3.7, dictionaries are ordered. In Python 3.6 and earlier, dictionaries are unordered.

When we say that dictionaries are ordered, it means that the items have a defined order, and that order will not change.

Unordered means that the items do not have a defined order, you cannot refer to an item by using an index.

Changeable
Dictionaries are changeable, meaning that we can change, add or remove items after the dictionary has been created.

Duplicates Not Allowed
Dictionaries cannot have two items with the same key:

Dictionary Items - Data Types
The values in dictionary items can be of any data type:

The dict() Constructor
It is also possible to use the dict() constructor to make a dictionary.


```python
thisdict = dict(name = "John", age = 36, country = "Norway")
print(thisdict)
```

Accessing Items
You can access the items of a dictionary by referring to its key name, inside square brackets:


```python
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
x = thisdict["model"]
```

There is also a method called get() that will give you the same result:


```python
x = thisdict.get("model")
```

Get Keys
The keys() method will return a list of all the keys in the dictionary.


```python
thisdict = {'Name': 'Isa', 'Age': 25
    }
```


```python
thisdict.keys()
```




    dict_keys(['Name', 'Age'])




```python

```


```python
thisdict.values()
```




    dict_values(['Isa', 25])



The list of the keys is a view of the dictionary, meaning that any changes done to the dictionary will be reflected in the keys list.

Get Values
The values() method will return a list of all the values in the dictionary.

The list of the values is a view of the dictionary, meaning that any changes done to the dictionary will be reflected in the values list.
