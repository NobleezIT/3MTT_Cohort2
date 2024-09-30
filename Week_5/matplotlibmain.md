# Matplotlib

## Introduction to Matplotlib
Matplotlib is a powerful and widely-used plotting library for Python. It allows for the creation of static, animated, and interactive visualizations in Python. Matplotlib is useful for a variety of visualization needs, from simple line graphs to more complex, customized plots. It provides a high level of flexibility in terms of plot customization, enabling users to create publication-quality figures.

The main module used for plotting is `pyplot`, which provides a MATLAB-like interface for generating plots.

### Installing Matplotlib


To use Matplotlib, you need to install it using pip:
```python
pip install matplotlib
```

You also need to import it in your code, typically as:


```python
# Importing matplotlib, pandas and Numpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

## Basic Plotting with Pyplot
The `pyplot` module is the core of Matplotlib. It provides functions for generating various types of plots, such as line plots, scatter plots, bar charts, histograms, etc.

### Line Plot
 **Use Case**: 
   - Ideal for visualizing trends over time or continuous data.
   - Best when data points are ordered and the relationship between them is important (e.g., time series, stock prices, etc.).

   **Data Type**: 
   - Continuous or ordered data (e.g., dates, time, temperature, etc.).
   
A simple line plot can be created using `plt.plot()`. Here's an example:




```python
# Example
import matplotlib.pyplot as plt
days = [1, 2, 3, 4, 5]
temperature = [22, 23, 21, 25, 24]
plt.plot(days, temperature)
plt.xlabel('Days')
plt.ylabel('Temperature')
plt.title('Temperature Over Days')
plt.show()
```


    
![png](output_4_0.png)
    


This code creates a line plot of `y = sin(x)`.


### Scatter Plot
**Use Case**: 
   - Best for visualizing relationships between two numerical variables.
   - Used for exploring potential correlations or patterns (e.g., height vs. weight, price vs. demand).

   **Data Type**: 
   - Two continuous numerical variables.



```python
# Example1:
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 7, 8]
plt.scatter(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot Example')
plt.show()
```


    
![png](output_7_0.png)
    



```python
# Example2:
x = np.random.rand(50)
y = np.random.rand(50)
plt.scatter(x, y, color='blue', marker='o')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Scatter Plot Example')
plt.show()
```


    
![png](output_8_0.png)
    


### Bar Plot

**Use Case**: 
   - Best for comparing different categories or groups.
   - Suitable for categorical data where the goal is to compare the size or frequency of different categories.

   **Data Type**: 
   - Categorical data (e.g., sales by product, population by country, etc.).
Bar plots are useful for visualizing data comparisons among categorie


```python
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 5, 9]

plt.bar(categories, values, color='green')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot Example')
plt.show()
```


    
![png](output_10_0.png)
    


### Histogram
A histogram represents the distribution of a set of data:


```python
data = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6]
plt.hist(data, bins=5)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram Example')
plt.show()
```


    
![png](output_12_0.png)
    


 **Pie Chart**
   **Use Case**: 
   - Used to show proportions or percentages of a whole.
   - Best for comparing parts of a whole (e.g., market share, budget distribution).

   **Data Type**: 
   - Categorical data with proportions.


```python
# Example 
labels = ['A', 'B', 'C', 'D']
sizes = [40, 30, 20, 10]
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Pie Chart Example')
plt.show()
```


    
![png](output_14_0.png)
    


**Box Plot (Box-and-Whisker Plot)**
   **Use Case**: 
   - Used for visualizing the distribution of data and detecting outliers.
   - Displays the median, quartiles, and extreme values in the dataset.

   **Data Type**: 
   - Continuous numerical data (e.g., exam scores, age distribution).


```python
data = [7, 8, 9, 10, 15, 20, 25, 30]
plt.boxplot(data)
plt.title('Box Plot Example')
plt.show()
```


    
![png](output_16_0.png)
    


**Area Plot**
   **Use Case**: 
   - Similar to line plots, but the area under the line is filled.
   - Best for showing cumulative data or trends over time.

   **Data Type**: 
   - Continuous data, often used for time series data.


```python
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
plt.fill_between(x, y, color="skyblue", alpha=0.4)
plt.plot(x, y, color="Slateblue", alpha=0.6)
plt.title('Area Plot Example')
plt.show()
```


    
![png](output_18_0.png)
    


## Subplots
You can create multiple plots within the same figure using `plt.subplot()`:



```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 5))

# First subplot
plt.subplot(1, 2, 1)
plt.plot(x, y1, 'r')
plt.title('Sine Plot')

# Second subplot
plt.subplot(1, 2, 2)
plt.plot(x, y2, 'b')
plt.title('Cosine Plot')
plt.tight_layout()
plt.show()
```


    
![png](output_20_0.png)
    


## Customizing Plots
Matplotlib offers a great deal of customization to make plots more informative.

### Titles and Labels
Adding titles, axis labels, and legends is essential for understanding a plot:

- `plt.title('Title')` to add a title.
- `plt.xlabel('X Axis')` and `plt.ylabel('Y Axis')` to label axes.
- `plt.legend()` to add a legend.

### Colors and Line Styles
Matplotlib allows for extensive customization of the colors and styles of lines:



```python
plt.plot(x, y, color='green', linestyle='--', linewidth=2)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[21], line 1
    ----> 1 plt.plot(x, y, color='green', linestyle='--', linewidth=2)
    

    File ~\miniconda3\Lib\site-packages\matplotlib\pyplot.py:3794, in plot(scalex, scaley, data, *args, **kwargs)
       3786 @_copy_docstring_and_deprecators(Axes.plot)
       3787 def plot(
       3788     *args: float | ArrayLike | str,
       (...)
       3792     **kwargs,
       3793 ) -> list[Line2D]:
    -> 3794     return gca().plot(
       3795         *args,
       3796         scalex=scalex,
       3797         scaley=scaley,
       3798         **({"data": data} if data is not None else {}),
       3799         **kwargs,
       3800     )
    

    File ~\miniconda3\Lib\site-packages\matplotlib\axes\_axes.py:1779, in Axes.plot(self, scalex, scaley, data, *args, **kwargs)
       1536 """
       1537 Plot y versus x as lines and/or markers.
       1538 
       (...)
       1776 (``'green'``) or hex strings (``'#008000'``).
       1777 """
       1778 kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
    -> 1779 lines = [*self._get_lines(self, *args, data=data, **kwargs)]
       1780 for line in lines:
       1781     self.add_line(line)
    

    File ~\miniconda3\Lib\site-packages\matplotlib\axes\_base.py:296, in _process_plot_var_args.__call__(self, axes, data, *args, **kwargs)
        294     this += args[0],
        295     args = args[1:]
    --> 296 yield from self._plot_args(
        297     axes, this, kwargs, ambiguous_fmt_datakey=ambiguous_fmt_datakey)
    

    File ~\miniconda3\Lib\site-packages\matplotlib\axes\_base.py:486, in _process_plot_var_args._plot_args(self, axes, tup, kwargs, return_kwargs, ambiguous_fmt_datakey)
        483     axes.yaxis.update_units(y)
        485 if x.shape[0] != y.shape[0]:
    --> 486     raise ValueError(f"x and y must have same first dimension, but "
        487                      f"have shapes {x.shape} and {y.shape}")
        488 if x.ndim > 2 or y.ndim > 2:
        489     raise ValueError(f"x and y can be no greater than 2D, but have "
        490                      f"shapes {x.shape} and {y.shape}")
    

    ValueError: x and y must have same first dimension, but have shapes (100,) and (5,)



    
![png](output_22_1.png)
    



You can use `linestyle` (`'--'`, `'-'`, `':'`, etc.), `color` (names, hex codes), and `linewidth` to control how your line appears.

### Markers
Markers are used to indicate data points:

```python
plt.plot(x, y, marker='o', color='red')
```

Markers can be customized using `marker` (`'o'`, `'*'`, `'s'`, etc.) and `markersize`.

## Figures and Axes
Matplotlib uses the object-oriented approach involving `Figure` and `Axes` classes for more control:

- **Figure**: The overall container that holds one or more `Axes`.
- **Axes**: A specific plot or graph.



```python
# Here’s an example:

fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(x, y, label='sin(x)')
ax.set_title('Figure and Axes Example')
ax.set_xlabel('x values')
ax.set_ylabel('y values')
ax.legend()
plt.show()

```


```python
### Adding Multiple Axes
# You can create multiple axes within the same figure:

fig = plt.figure()

ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # Main axes
ax2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])  # Inset axes

ax1.plot(x, y)
ax1.set_title('Main Plot')

ax2.plot(x, np.cos(x))
ax2.set_title('Inset Plot')

plt.show()

```

## Working with Legends
Adding a legend is useful for distinguishing different elements in the plot. You can use `plt.legend()` for simple cases or customize it:


```python
plt.plot(x, y1, label='Sine', color='blue')
plt.plot(x, y2, label='Cosine', color='red')
plt.legend(loc='upper right')  # Location of the legend
plt.show()
```

## Adding Annotations
Annotations help in marking specific points on a plot to highlight them:


```python
plt.plot(x, y)
plt.annotate('Max Point', xy=(7.85, 1), xytext=(10, 1.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.show()

```

## Advanced Plots

### Box Plot
Box plots provide a summary of a dataset's distribution:


```python
data = [np.random.normal(0, std, 100) for std in range(1, 4)]
plt.boxplot(data, vert=True, patch_artist=True)
plt.xlabel('Distribution')
plt.ylabel('Values')
plt.title('Box Plot Example')
plt.show()
```

### Heatmap
Heatmaps are used to represent data in matrix form:


```python
data = np.random.rand(10, 10)
plt.imshow(data, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Heatmap Example')
plt.show()
```


```python

```

### Pie Chart
Pie charts can visualize proportions:


```python
sizes = [15, 30, 45, 10]
labels = ['A', 'B', 'C', 'D']
explode = (0, 0.1, 0, 0)  # Explode the second slice

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Pie Chart Example')
plt.show()
```

## Working with Dates
Matplotlib can also plot data that includes dates:



```python
import matplotlib.dates as mdates
import datetime

dates = [datetime.datetime(2024, 9, i) for i in range(1, 11)]
values = [5, 6, 7, 8, 7, 6, 5, 4, 5, 6]

fig, ax = plt.subplots()
ax.plot(dates, values)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Plot with Dates')
plt.show()
```

## Saving Plots
Plots can be saved to files using `plt.savefig()`:


```python
plt.plot(x, y)
plt.title('Save Plot Example')
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')  # Save with specified resolution and margins
plt.show()
```

## Interactive Plots with `%matplotlib notebook`
For interactive plots within a Jupyter Notebook, you can use `%matplotlib notebook` to enable zooming and panning:


```python
%matplotlib notebook
plt.plot(x, y)
plt.show()
```

## Conclusion
Matplotlib is a versatile and powerful library for data visualization in Python. It provides a wide range of plotting capabilities, from basic plots to more advanced figures, all of which can be highly customized to fit the user’s needs. Understanding how to use Matplotlib effectively allows for the clear and insightful presentation of data, which is crucial for data analysis and communication.


```python

```

# Matplotlib with Pandas

## Introduction
Matplotlib can be effectively integrated with Pandas to create visualizations directly from dataframes, making it easier to analyze and understand data. By leveraging Pandas' data handling capabilities and Matplotlib's powerful plotting features, data analysis becomes more intuitive and visually appealing.

Pandas provides an interface to Matplotlib through its `.plot()` method, which allows for a streamlined and simple approach to generate a wide range of plots.

### Prerequisites
Ensure you have both `pandas` and `matplotlib` installed:

# Matplotlib with Pandas

## Introduction
Matplotlib can be effectively integrated with Pandas to create visualizations directly from dataframes, making it easier to analyze and understand data. By leveraging Pandas' data handling capabilities and Matplotlib's powerful plotting features, data analysis becomes more intuitive and visually appealing.

Pandas provides an interface to Matplotlib through its `.plot()` method, which allows for a streamlined and simple approach to generate a wide range of plots.

### Prerequisites
Ensure you have both `pandas` and `matplotlib` installed:

## Basic Plotting with Pandas
Pandas' `DataFrame` and `Series` have built-in `plot` methods that are based on Matplotlib. Here's an example of creating a simple plot directly from a DataFrame.

### Line Plot
Suppose you have a dataset representing sales data for different months:


```python
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Sales': [200, 220, 250, 270, 300, 310]
}
df = pd.DataFrame(data)

# Plotting the sales data
df.plot(x='Month', y='Sales', kind='line', title='Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid()
plt.show()
```


    
![png](output_48_0.png)
    


In this example, the `.plot()` method is used to generate a line plot, specifying the `x` and `y` values and the plot type.

## Different Plot Types

### Bar Plot
Bar plots are useful for comparing categorical data:


```python
df = pd.DataFrame({
    'Product': ['A', 'B', 'C', 'D'],
    'Revenue': [4000, 3000, 4500, 3500]
})

df.plot(x='Product', y='Revenue', kind='bar', color='skyblue', title='Product Revenue')
plt.xlabel('Product')
plt.ylabel('Revenue')
plt.show()
```


    
![png](output_50_0.png)
    



```python
### Horizontal Bar Plot
# For horizontal representation:

df.plot(x='Product', y='Revenue', kind='barh', color='orange', title='Product Revenue')
plt.xlabel('Revenue')
plt.ylabel('Product')
plt.show()

```


    
![png](output_51_0.png)
    



```python
### Scatter Plot
# Scatter plots help in showing the relationship between two numeric variables:

data = {
    'Temperature': [20, 21, 23, 24, 26, 27, 28, 30, 32, 35],
    'Ice_Cream_Sales': [200, 210, 220, 230, 250, 260, 270, 290, 310, 330]
}
df = pd.DataFrame(data)

df.plot(x='Temperature', y='Ice_Cream_Sales', kind='scatter', title='Ice Cream Sales vs Temperature')
plt.xlabel('Temperature (C)')
plt.ylabel('Ice Cream Sales')
plt.show()

```


    
![png](output_52_0.png)
    


### Histogram
Histograms can be used to visualize the distribution of data:



```python
data = {
    'Age': np.random.randint(18, 70, size=100)
}
df = pd.DataFrame(data)

df['Age'].plot(kind='hist', bins=10, color='purple', alpha=0.7, title='Age Distribution')
plt.xlabel('Age')
plt.show()

```


    
![png](output_54_0.png)
    


### Box Plot
Box plots are helpful for summarizing data distributions, highlighting median, quartiles, and outliers:



```python
data = {
    'Math': np.random.randint(50, 100, 50),
    'Science': np.random.randint(55, 95, 50),
    'English': np.random.randint(60, 90, 50)
}
df = pd.DataFrame(data)

df.plot(kind='box', title='Score Distribution by Subject')
plt.ylabel('Scores')
plt.show()

```


    
![png](output_56_0.png)
    


## Grouped and Stacked Bar Plots
When dealing with categorical data grouped by categories, you can create grouped and stacked bar plots.


```python
### Grouped Bar Plot
data = {
    'Year': ['2021', '2021', '2022', '2022'],
    'Product': ['A', 'B', 'A', 'B'],
    'Revenue': [5000, 3000, 5500, 3200]
}
df = pd.DataFrame(data)

# Using pivot to reshape data for plotting
pivot_df = df.pivot(index='Year', columns='Product', values='Revenue')

# Plotting grouped bar plot
pivot_df.plot(kind='bar', title='Revenue by Year and Product')
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.show()

```


    
![png](output_58_0.png)
    



```python
### Stacked Bar Plot
pivot_df.plot(kind='bar', stacked=True, title='Stacked Revenue by Year and Product')
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.show()

```


    
![png](output_59_0.png)
    



```python
## Plotting Time Series Data
# Pandas works well with time series data and allows easy visualization of trends.

# Creating a date range
dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
data = {
    'Sales': np.random.randint(200, 500, len(dates))
}
df = pd.DataFrame(data, index=dates)

# Plotting the time series
df.plot(title='Monthly Sales Over Time', ylabel='Sales', xlabel='Month')
plt.show()

```

    C:\Users\abula\AppData\Local\Temp\ipykernel_3496\1181150943.py:5: FutureWarning: 'M' is deprecated and will be removed in a future version, please use 'ME' instead.
      dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    


    
![png](output_60_1.png)
    


This example demonstrates how easily a time series can be plotted using Pandas. The `dates` become the index of the DataFrame, enabling automatic date formatting for the x-axis.

## Customizing Plots
To customize the look and feel of Pandas plots, we can access Matplotlib functions directly after calling `.plot()`. 

### Using Matplotlib for Further Customization
You can modify colors, labels, and add legends using Matplotlib functions:


```python
df = pd.DataFrame({
    'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
    'Visitors': [200, 230, 250, 270, 300]
})

ax = df.plot(x='Day', y='Visitors', kind='line', color='green', linestyle='--', marker='o', title='Daily Visitors')
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Number of Visitors')
plt.xticks(rotation=45)
plt.grid()
plt.show()
```


    
![png](output_62_0.png)
    


### Using Multiple Subplots
Pandas' plotting functionality can also be used to create multiple subplots for different columns:



```python
data = {
    'Temperature': np.random.randint(15, 35, size=10),
    'Humidity': np.random.randint(30, 70, size=10),
    'Wind_Speed': np.random.randint(5, 20, size=10)
}
df = pd.DataFrame(data)

df.plot(subplots=True, layout=(3, 1), figsize=(8, 10), title='Weather Data', grid=True)
plt.tight_layout()
plt.show()

```


    
![png](output_64_0.png)
    


## Adding Annotations
Annotations are useful for marking important events or trends in your plots:



```python
df = pd.DataFrame({
    'Year': [2018, 2019, 2020, 2021, 2022],
    'Profit': [100, 150, 200, 250, 300]
})

ax = df.plot(x='Year', y='Profit', kind='line', marker='o', title='Yearly Profit')
ax.annotate('Highest Profit', xy=(2022, 300), xytext=(2019, 270),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.xlabel('Year')
plt.ylabel('Profit')
plt.show()
```


    
![png](output_66_0.png)
    


## Saving Plots
Saving a plot generated from a Pandas DataFrame is simple with Matplotlib's `savefig()`:


```python
ax = df.plot(x='Year', y='Profit', kind='line', title='Yearly Profit')
plt.savefig('yearly_profit_plot.png', dpi=300, bbox_inches='tight')
```


    
![png](output_68_0.png)
    


## Conclusion
Matplotlib's integration with Pandas provides a simple and efficient way to create visualizations directly from dataframes. By leveraging Pandas' `.plot()` method, you can easily generate a wide range of visualizations, from basic line plots to advanced time series plots, all while utilizing Matplotlib’s customization capabilities.

Using Pandas with Matplotlib allows you to explore data quickly and effectively, leading to better insights and decisions.
