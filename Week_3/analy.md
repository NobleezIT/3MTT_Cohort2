```python
import pandas as pd
```


```python
# Loading the Data
df = pd.read_csv(r"C:\Users\abula\Desktop\dataanalytics\Week_3\customer.csv")
df
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 204 entries, 0 to 203
    Data columns (total 72 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   CUST_NAME            199 non-null    object 
     1   Gender_Code          199 non-null    object 
     2   ADDRESS1             199 non-null    object 
     3   CITY                 199 non-null    object 
     4   STATE                163 non-null    object 
     5   COUNTRY_CODE         199 non-null    object 
     6   POSTAL_CODE          199 non-null    object 
     7   POSTAL_CODE_PLUS4    199 non-null    float64
     8   ADDRESS2             0 non-null      float64
     9   EMAIL_ADDRESS        199 non-null    object 
     10  PHONE_NUMBER         199 non-null    object 
     11  CREDITCARD_TYPE      199 non-null    object 
     12  LOCALITY             0 non-null      float64
     13  SALESMAN_ID          199 non-null    object 
     14  NATIONALITY          199 non-null    object 
     15  NATIONAL_ID          199 non-null    object 
     16  CREDITCARD_NUMBER    199 non-null    float64
     17  DRIVER_LICENSE       0 non-null      float64
     18  CUST_ID              199 non-null    float64
     19  ORDER_ID             199 non-null    float64
     20  ORDER_DATE           199 non-null    object 
     21  ORDER_TIME           199 non-null    object 
     22  FREIGHT_CHARGES      199 non-null    float64
     23  ORDER_SALESMAN       199 non-null    object 
     24  ORDER_POSTED_DATE    199 non-null    object 
     25  ORDER_SHIP_DATE      199 non-null    object 
     26  AGE                  175 non-null    object 
     27  ORDER_VALUE          199 non-null    float64
     28  T_TYPE               199 non-null    object 
     29  PURCHASE_TOUCHPOINT  199 non-null    object 
     30  PURCHASE_STATUS      199 non-null    object 
     31  ORDER_TYPE           199 non-null    object 
     32  GENERATION           199 non-null    object 
     33  Baby Food            199 non-null    float64
     34  Diapers              199 non-null    float64
     35  Formula              199 non-null    float64
     36  Lotion               199 non-null    float64
     37  Baby wash            199 non-null    float64
     38  Wipes                199 non-null    float64
     39  Fresh Fruits         199 non-null    float64
     40  Fresh Vegetables     199 non-null    float64
     41  Beer                 199 non-null    float64
     42  Wine                 199 non-null    float64
     43  Club Soda            199 non-null    float64
     44  Sports Drink         199 non-null    float64
     45  Chips                199 non-null    float64
     46  Popcorn              199 non-null    float64
     47  Oatmeal              199 non-null    float64
     48  Medicines            199 non-null    float64
     49  Canned Foods         199 non-null    float64
     50  Cigarettes           199 non-null    float64
     51  Cheese               199 non-null    float64
     52  Cleaning Products    199 non-null    float64
     53  Condiments           199 non-null    float64
     54  Frozen Foods         199 non-null    float64
     55  Kitchen Items        199 non-null    float64
     56  Meat                 199 non-null    float64
     57  Office Supplies      199 non-null    float64
     58  Personal Care        199 non-null    float64
     59  Pet Supplies         199 non-null    float64
     60  Sea Food             199 non-null    float64
     61  Spices               199 non-null    float64
     62  Unnamed: 62          0 non-null      float64
     63  Unnamed: 63          0 non-null      float64
     64  Unnamed: 64          0 non-null      float64
     65  Unnamed: 65          0 non-null      float64
     66  Unnamed: 66          0 non-null      float64
     67  Unnamed: 67          0 non-null      float64
     68  Unnamed: 68          0 non-null      float64
     69  Unnamed: 69          0 non-null      float64
     70  Unnamed: 70          0 non-null      float64
     71  Unnamed: 71          0 non-null      float64
    dtypes: float64(48), object(24)
    memory usage: 114.9+ KB
    


```python
df.dtypes
```




    CUST_NAME       object
    Gender_Code     object
    ADDRESS1        object
    CITY            object
    STATE           object
                    ...   
    Unnamed: 67    float64
    Unnamed: 68    float64
    Unnamed: 69    float64
    Unnamed: 70    float64
    Unnamed: 71    float64
    Length: 72, dtype: object



### Data Cleaning and Data Preprocessing Techniques

### What is Data Cleaning
Data cleaning involves identifying and rectifying errors, inconsistencies, and missing values within a dataset. It's like preparing your ingredients before cooking; you want everything in order to get the perfect analysis or visualization.

Data cleaning is the initial phase of refining your dataset, making it readable and usable with techniques like removing duplicates, handling missing values and data type conversion while data preprocessing is similar to taking this refined data and scaling with more advanced techniques such as feature engineering, encoding categorical variables and and handling outliers to achieve better and more advanced results.

The goal is to turn your dataset into a refined masterpiece, ready for analysis or modeling.


## Exploratory Data Analysis (EDA)

EDA helps you understand the structure and characteristics of your dataset. Some Pandas functions help us gain insights into our dataset. We call these functions by calling the dataset variable plus the function.

For example:

df.head() will call the first 5 rows of the dataset. You can specify the number of rows to be displayed in the parentheses.
df.describe() gives some statistical data like percentile, mean and standard deviation of the numerical values of the Series or DataFrame.
df.info() gives the number of columns, column labels, column data types, memory usage, range index, and the number of cells in each column (non-null values).

### How to Handle Missing Values
As a newbie in this field, missing values pose a significant stress as they come in different formats and can adversely impact your analysis or model.

Machine learning models cannot be trained with data that has missing or "NAN" values as they can alter your end result during analysis. But do not fret, Pandas provides methods to handle this problem.


```python
#Check for missing values
# print(df.isnull().sum())

#Drop rows with missing valiues and place it in a new variable "df_cleaned"
# df_cleaned = df.dropna()

#Fill missing values with mean for numerical data and place it ina new variable called df_filled
# df_filled = df.fillna(df.mean())
# We would come back to this
```

But if the number of rows that have missing values is large, then this method will be inadequate.

For numerical data, you can simply compute the mean and input it into the rows that have missing values. Code snippet below:


```python
#Replace missing values with the mean of each column
df.fillna(df.mean(), inplace=True)

#If you want to replace missing values in a specific column, you can do it this way:
#Replace 'column_name' with the actual column name
df['column_name'].fillna(df['column_name'].mean(), inplace=True)

#Now, df contains no missing values, and NaNs have been replaced with column mean
```

### How to Remove Duplicate Records

Duplicate records can distort your analysis by influencing the results in ways that do not accurately show trends and underlying patterns (by producing outliers).

Pandas helps to identify and remove the duplicate values in an easy way by placing them in new variables.


```python
#Identify duplicates
print(df.duplicated().sum())

#Remove duplicates
df_no_duplicates = df.drop_duplicates()
```

### Data Types and Conversion

Data type conversion in Pandas is a crucial aspect of data preprocessing, allowing you to ensure that your data is in the appropriate format for analysis or modeling.

Data from various sources are usually messy and the data types of some values may be in the wrong format, for example some numerical values may come in 'float' or 'string' format instead of 'integer' format and a mix up of these formats leads to errors and wrong results.

You can convert a Column of type int to float with the following code:


```python
#Convert 'Column1' to float
df['Column1'] = df['Column1'].astype(float)

#Display updated data types
print(df.dtypes)
```

### How to Handle Outliers

Outliers are data points significantly different from the majority of the data, they can distort statistical measures and adversely affect the performance of machine learning models.

They may be caused by human error, missing NaN values, or could be accurate data that does not correlate with the rest of the data.

There are several methods to identify and remove outliers, they are:

Remove NaN values.
Visualize the data before and after removal.
Z-score method (for normally distributed data).
IQR (Interquartile range) method for more robust data.

### Inspecting the Dataset


```python
# Display the first 5 rows
df.head()
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
      <th>CUST_NAME</th>
      <th>Gender_Code</th>
      <th>ADDRESS1</th>
      <th>CITY</th>
      <th>STATE</th>
      <th>COUNTRY_CODE</th>
      <th>POSTAL_CODE</th>
      <th>POSTAL_CODE_PLUS4</th>
      <th>ADDRESS2</th>
      <th>EMAIL_ADDRESS</th>
      <th>...</th>
      <th>Unnamed: 62</th>
      <th>Unnamed: 63</th>
      <th>Unnamed: 64</th>
      <th>Unnamed: 65</th>
      <th>Unnamed: 66</th>
      <th>Unnamed: 67</th>
      <th>Unnamed: 68</th>
      <th>Unnamed: 69</th>
      <th>Unnamed: 70</th>
      <th>Unnamed: 71</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Allen Perl</td>
      <td>Mr.</td>
      <td>4707    Hillcrest Lane</td>
      <td>Abeto</td>
      <td>PG</td>
      <td>IT</td>
      <td>6040</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Allen.M.Perl@spambob.com</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Allen Perl</td>
      <td>Mr.</td>
      <td>4707    Hillcrest Lane</td>
      <td>Abeto</td>
      <td>PG</td>
      <td>IT</td>
      <td>6040</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Allen.M.Perl@spambob.com</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Allen Perl</td>
      <td>Mr.</td>
      <td>4707    Hillcrest Lane</td>
      <td>Abeto</td>
      <td>PG</td>
      <td>IT</td>
      <td>6040</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Allen.M.Perl@spambob.com</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Allen Perl</td>
      <td>Mr.</td>
      <td>4707    Hillcrest Lane</td>
      <td>Abeto</td>
      <td>PG</td>
      <td>IT</td>
      <td>6040</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Allen.M.Perl@spambob.com</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Allen Perl</td>
      <td>Mr.</td>
      <td>4707    Hillcrest Lane</td>
      <td>Abeto</td>
      <td>PG</td>
      <td>IT</td>
      <td>6040</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Allen.M.Perl@spambob.com</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 72 columns</p>
</div>




```python
# Summary of data types and non-null values
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 204 entries, 0 to 203
    Data columns (total 72 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   CUST_NAME            199 non-null    object 
     1   Gender_Code          199 non-null    object 
     2   ADDRESS1             199 non-null    object 
     3   CITY                 199 non-null    object 
     4   STATE                163 non-null    object 
     5   COUNTRY_CODE         199 non-null    object 
     6   POSTAL_CODE          199 non-null    object 
     7   POSTAL_CODE_PLUS4    199 non-null    float64
     8   ADDRESS2             0 non-null      float64
     9   EMAIL_ADDRESS        199 non-null    object 
     10  PHONE_NUMBER         199 non-null    object 
     11  CREDITCARD_TYPE      199 non-null    object 
     12  LOCALITY             0 non-null      float64
     13  SALESMAN_ID          199 non-null    object 
     14  NATIONALITY          199 non-null    object 
     15  NATIONAL_ID          199 non-null    object 
     16  CREDITCARD_NUMBER    199 non-null    float64
     17  DRIVER_LICENSE       0 non-null      float64
     18  CUST_ID              199 non-null    float64
     19  ORDER_ID             199 non-null    float64
     20  ORDER_DATE           199 non-null    object 
     21  ORDER_TIME           199 non-null    object 
     22  FREIGHT_CHARGES      199 non-null    float64
     23  ORDER_SALESMAN       199 non-null    object 
     24  ORDER_POSTED_DATE    199 non-null    object 
     25  ORDER_SHIP_DATE      199 non-null    object 
     26  AGE                  175 non-null    object 
     27  ORDER_VALUE          199 non-null    float64
     28  T_TYPE               199 non-null    object 
     29  PURCHASE_TOUCHPOINT  199 non-null    object 
     30  PURCHASE_STATUS      199 non-null    object 
     31  ORDER_TYPE           199 non-null    object 
     32  GENERATION           199 non-null    object 
     33  Baby Food            199 non-null    float64
     34  Diapers              199 non-null    float64
     35  Formula              199 non-null    float64
     36  Lotion               199 non-null    float64
     37  Baby wash            199 non-null    float64
     38  Wipes                199 non-null    float64
     39  Fresh Fruits         199 non-null    float64
     40  Fresh Vegetables     199 non-null    float64
     41  Beer                 199 non-null    float64
     42  Wine                 199 non-null    float64
     43  Club Soda            199 non-null    float64
     44  Sports Drink         199 non-null    float64
     45  Chips                199 non-null    float64
     46  Popcorn              199 non-null    float64
     47  Oatmeal              199 non-null    float64
     48  Medicines            199 non-null    float64
     49  Canned Foods         199 non-null    float64
     50  Cigarettes           199 non-null    float64
     51  Cheese               199 non-null    float64
     52  Cleaning Products    199 non-null    float64
     53  Condiments           199 non-null    float64
     54  Frozen Foods         199 non-null    float64
     55  Kitchen Items        199 non-null    float64
     56  Meat                 199 non-null    float64
     57  Office Supplies      199 non-null    float64
     58  Personal Care        199 non-null    float64
     59  Pet Supplies         199 non-null    float64
     60  Sea Food             199 non-null    float64
     61  Spices               199 non-null    float64
     62  Unnamed: 62          0 non-null      float64
     63  Unnamed: 63          0 non-null      float64
     64  Unnamed: 64          0 non-null      float64
     65  Unnamed: 65          0 non-null      float64
     66  Unnamed: 66          0 non-null      float64
     67  Unnamed: 67          0 non-null      float64
     68  Unnamed: 68          0 non-null      float64
     69  Unnamed: 69          0 non-null      float64
     70  Unnamed: 70          0 non-null      float64
     71  Unnamed: 71          0 non-null      float64
    dtypes: float64(48), object(24)
    memory usage: 114.9+ KB
    


```python
# Descriptive statistics for numerical columns
df.describe()
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
      <th>POSTAL_CODE_PLUS4</th>
      <th>ADDRESS2</th>
      <th>LOCALITY</th>
      <th>CREDITCARD_NUMBER</th>
      <th>DRIVER_LICENSE</th>
      <th>CUST_ID</th>
      <th>ORDER_ID</th>
      <th>FREIGHT_CHARGES</th>
      <th>ORDER_VALUE</th>
      <th>Baby Food</th>
      <th>...</th>
      <th>Unnamed: 62</th>
      <th>Unnamed: 63</th>
      <th>Unnamed: 64</th>
      <th>Unnamed: 65</th>
      <th>Unnamed: 66</th>
      <th>Unnamed: 67</th>
      <th>Unnamed: 68</th>
      <th>Unnamed: 69</th>
      <th>Unnamed: 70</th>
      <th>Unnamed: 71</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>199.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.990000e+02</td>
      <td>0.0</td>
      <td>199.000000</td>
      <td>199.000000</td>
      <td>199.000000</td>
      <td>199.000000</td>
      <td>199.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.555874e+15</td>
      <td>NaN</td>
      <td>10167.301508</td>
      <td>4388.417085</td>
      <td>16.853015</td>
      <td>101.523116</td>
      <td>0.045226</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.347786e+15</td>
      <td>NaN</td>
      <td>101.882828</td>
      <td>2728.801204</td>
      <td>10.637495</td>
      <td>114.116234</td>
      <td>0.208324</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.600000e+13</td>
      <td>NaN</td>
      <td>10003.000000</td>
      <td>28.000000</td>
      <td>1.490000</td>
      <td>5.100000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.405000e+14</td>
      <td>NaN</td>
      <td>10091.000000</td>
      <td>1825.000000</td>
      <td>8.995000</td>
      <td>27.340000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.530000e+15</td>
      <td>NaN</td>
      <td>10155.000000</td>
      <td>4184.000000</td>
      <td>13.880000</td>
      <td>44.290000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.710000e+15</td>
      <td>NaN</td>
      <td>10255.000000</td>
      <td>6948.000000</td>
      <td>21.970000</td>
      <td>180.205000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.010000e+15</td>
      <td>NaN</td>
      <td>10351.000000</td>
      <td>9180.000000</td>
      <td>68.570000</td>
      <td>800.900000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 48 columns</p>
</div>



### Data Cleaning

Correcting Misspelling


```python
# Checking errors in credit type
df['CREDITCARD_TYPE'].unique()
```




    array(['Master Card', 'VISA', 'Discover', 'American Express',
           'American Expres', 'Diners Club', 'jcb', nan], dtype=object)



The CREDITCARD_TYPE column contains the following entries:

Correct values: 'Master Card', 'VISA', 'Discover', 'American Express', 'Diners Club', 'JCB'.

Spelling errors: 'American Expres' (should be 'American Express'), 'jcb' (should be 'JCB').

Steps to fix the spelling errors:

Replace 'American Expres' with 'American Express'.

Capitalize 'jcb' to 'JCB'.


```python
# Replace incorrect entries with the correct ones
df['CREDITCARD_TYPE'].replace({'American Expres': 'American Express', 'jcb': 'JCB'}, inplace=True)

# Verify the changes
df['CREDITCARD_TYPE'].unique()

```

for any particular column the syntax is basically similar.


```python
# Replace misspellings in any column
# df['column_name'] = df['column_name'].replace({'old_value': 'new_value'})
```

Removing Empty Rows
There are missing data entries but you can also check if there are missing rows entirely.


```python
# Check if any row has all missing values
empty_rows = df.isna().all(axis=1)
print(df[empty_rows])
```

       CUST_NAME Gender_Code ADDRESS1 CITY STATE COUNTRY_CODE POSTAL_CODE  \
    69       NaN         NaN      NaN  NaN   NaN          NaN         NaN   
    70       NaN         NaN      NaN  NaN   NaN          NaN         NaN   
    71       NaN         NaN      NaN  NaN   NaN          NaN         NaN   
    72       NaN         NaN      NaN  NaN   NaN          NaN         NaN   
    73       NaN         NaN      NaN  NaN   NaN          NaN         NaN   
    
        POSTAL_CODE_PLUS4  ADDRESS2 EMAIL_ADDRESS  ... Cleaning Products  \
    69                NaN       NaN           NaN  ...               NaN   
    70                NaN       NaN           NaN  ...               NaN   
    71                NaN       NaN           NaN  ...               NaN   
    72                NaN       NaN           NaN  ...               NaN   
    73                NaN       NaN           NaN  ...               NaN   
    
       Condiments  Frozen Foods Kitchen Items Meat Office Supplies  Personal Care  \
    69        NaN           NaN           NaN  NaN             NaN            NaN   
    70        NaN           NaN           NaN  NaN             NaN            NaN   
    71        NaN           NaN           NaN  NaN             NaN            NaN   
    72        NaN           NaN           NaN  NaN             NaN            NaN   
    73        NaN           NaN           NaN  NaN             NaN            NaN   
    
        Pet Supplies  Sea Food  Spices  
    69           NaN       NaN     NaN  
    70           NaN       NaN     NaN  
    71           NaN       NaN     NaN  
    72           NaN       NaN     NaN  
    73           NaN       NaN     NaN  
    
    [5 rows x 62 columns]
    


```python
# Using isna()
missing_row = df.isna().sum(axis=1)
print(missing_row)
```

    0      3
    1      3
    2      3
    3      3
    4      3
          ..
    199    3
    200    3
    201    3
    202    3
    203    3
    Length: 204, dtype: int64
    

This will show the number of missing values per row. If a row has the total number of columns as NaN, that means it's an empty row.

To remove empty rows in your dataset, you can use the dropna() function, which drops rows where all the values are NaN (i.e., empty).


```python
# Drop rows where all values are missing
df.dropna(how='all', inplace=True)

# Verify the dataset after removing empty rows
df.shape  # This will show the new shape of the dataset after removal
```




    (199, 62)



## Checking and Removing Duplicate Values


```python
# Check for Duplicate values
duplicates = df.duplicated()

# Display duplicated rows
print(df[duplicates])
```

                    CUST_NAME Gender_Code           ADDRESS1      CITY STATE  \
    99   Quinn Perry              Master.       749 C Street  Amarillo    TX   
    100  Quinn Perry              Master.       749 C Street  Amarillo    TX   
    101  Kristin Mendoza             Mrs.  2909 Frank Avenue   Anaheim    CA   
    102  Kristin Mendoza             Mrs.  2909 Frank Avenue   Anaheim    CA   
    103  Kristin Mendoza             Mrs.  2909 Frank Avenue   Anaheim    CA   
    
        COUNTRY_CODE POSTAL_CODE  POSTAL_CODE_PLUS4  ADDRESS2  \
    99            US       79109                0.0       NaN   
    100           US       79109                0.0       NaN   
    101           US       92805                0.0       NaN   
    102           US       92805                0.0       NaN   
    103           US       92805                0.0       NaN   
    
                         EMAIL_ADDRESS  ... Cleaning Products Condiments  \
    99       Quinn.S.Perry@spambob.com  ...               1.0        0.0   
    100      Quinn.S.Perry@spambob.com  ...               0.0        0.0   
    101  Kristin.T.Mendoza@spambob.com  ...               0.0        1.0   
    102  Kristin.T.Mendoza@spambob.com  ...               0.0        0.0   
    103  Kristin.T.Mendoza@spambob.com  ...               0.0        0.0   
    
         Frozen Foods Kitchen Items Meat Office Supplies  Personal Care  \
    99            0.0           0.0  0.0             0.0            0.0   
    100           0.0           0.0  0.0             0.0            0.0   
    101           0.0           0.0  0.0             0.0            0.0   
    102           0.0           0.0  0.0             0.0            0.0   
    103           1.0           0.0  0.0             0.0            0.0   
    
         Pet Supplies  Sea Food  Spices  
    99            1.0       0.0     0.0  
    100           0.0       0.0     0.0  
    101           0.0       0.0     0.0  
    102           0.0       0.0     0.0  
    103           0.0       1.0     0.0  
    
    [5 rows x 62 columns]
    


```python
# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Verify that duplicates are removed
print(df.shape)  # Check the new shape of the dataset
```

    (194, 62)
    

### Check for Duplicates Based on Specific Columns

You can specify the columns on which you want to check for duplicates. This will identify rows that have identical values for those columns.


```python
# Check for duplicates based on specific columns (e.g., 'CUST_NAME' and 'EMAIL_ADDRESS')
duplicates_based_on_columns = df[df.duplicated(subset=['CUST_NAME', 'EMAIL_ADDRESS'], keep=False)]

# Display the duplicates
print(duplicates_based_on_columns)

```

                    CUST_NAME Gender_Code                  ADDRESS1  \
    0    Allen Perl                   Mr.    4707    Hillcrest Lane   
    1    Allen Perl                   Mr.    4707    Hillcrest Lane   
    2    Allen Perl                   Mr.    4707    Hillcrest Lane   
    3    Allen Perl                   Mr.    4707    Hillcrest Lane   
    4    Allen Perl                   Mr.    4707    Hillcrest Lane   
    ..                    ...         ...                       ...   
    195  James Sales                Miss.       3904 Capitol Avenue   
    197  Margaret Shelton            Mrs.      1034 Briarwood Drive   
    198  Margaret Shelton            Mrs.      1034 Briarwood Drive   
    201  Reynaldo Myers               Mr.  3923 Black Stallion Road   
    202  Reynaldo Myers               Mr.  3923 Black Stallion Road   
    
                  CITY STATE COUNTRY_CODE POSTAL_CODE  POSTAL_CODE_PLUS4  \
    0            Abeto    PG           IT        6040                0.0   
    1            Abeto    PG           IT        6040                0.0   
    2            Abeto    PG           IT        6040                0.0   
    3            Abeto    PG           IT        6040                0.0   
    4            Abeto    PG           IT        6040                0.0   
    ..             ...   ...          ...         ...                ...   
    195       Billings    MT           US       59102                0.0   
    197     Birmingham    AL           US       35222                0.0   
    198     Birmingham    AL           US       35222                0.0   
    201  Blaxland East   NSW           AU        2774                0.0   
    202  Blaxland East   NSW           AU        2774                0.0   
    
         ADDRESS2                    EMAIL_ADDRESS  ... Cleaning Products  \
    0         NaN         Allen.M.Perl@spambob.com  ...               0.0   
    1         NaN         Allen.M.Perl@spambob.com  ...               1.0   
    2         NaN         Allen.M.Perl@spambob.com  ...               0.0   
    3         NaN         Allen.M.Perl@spambob.com  ...               0.0   
    4         NaN         Allen.M.Perl@spambob.com  ...               0.0   
    ..        ...                              ...  ...               ...   
    195       NaN     James.K.Sales@trashymail.com  ...               0.0   
    197       NaN  Margaret.T.Shelton@pookmail.com  ...               0.0   
    198       NaN  Margaret.T.Shelton@pookmail.com  ...               0.0   
    201       NaN    Reynaldo.J.Myers@pookmail.com  ...               0.0   
    202       NaN    Reynaldo.J.Myers@pookmail.com  ...               0.0   
    
        Condiments  Frozen Foods Kitchen Items Meat Office Supplies  \
    0          1.0           0.0           0.0  0.0             0.0   
    1          0.0           0.0           0.0  0.0             0.0   
    2          0.0           0.0           0.0  0.0             0.0   
    3          0.0           0.0           0.0  0.0             0.0   
    4          0.0           0.0           0.0  0.0             0.0   
    ..         ...           ...           ...  ...             ...   
    195        0.0           1.0           0.0  0.0             0.0   
    197        0.0           0.0           0.0  0.0             0.0   
    198        0.0           0.0           0.0  0.0             0.0   
    201        1.0           0.0           0.0  0.0             0.0   
    202        1.0           0.0           0.0  0.0             0.0   
    
         Personal Care  Pet Supplies  Sea Food  Spices  
    0              0.0           0.0       0.0     0.0  
    1              0.0           0.0       0.0     0.0  
    2              0.0           0.0       0.0     0.0  
    3              0.0           0.0       0.0     0.0  
    4              0.0           0.0       0.0     0.0  
    ..             ...           ...       ...     ...  
    195            1.0           0.0       0.0     0.0  
    197            0.0           0.0       0.0     0.0  
    198            0.0           0.0       0.0     0.0  
    201            0.0           0.0       0.0     0.0  
    202            0.0           0.0       0.0     0.0  
    
    [173 rows x 62 columns]
    

### Drop duplicates based on specific columns

If you want to drop duplicates only based on specific columns (and not the entire row), you can specify the column names:


```python
# # Remove duplicates based on 'CUST_NAME' and 'EMAIL_ADDRESS', keeping the first occurrence
# cleaned_data = data.drop_duplicates(subset=['CUST_NAME', 'EMAIL_ADDRESS'], keep='first')

# # Display the cleaned data
# print(cleaned_data)

```

### Changing the Columns to lower case for Consistency

Consistency:

When you consistently use lowercase, it eliminates the confusion that may arise from mixing upper and lowercase letters, especially in larger datasets or when working in teams.
It's easier to remember that all column names are lowercase, avoiding typos like df['CreditCard_Type'] vs. df['creditcard_type'].

2. Simplicity:

Lowercase names are simple and reduce the chance of errors when typing column names.
They prevent issues with case sensitivity, which is important because column names in pandas are case-sensitive (df['Name'] is not the same as df['name']).

3. Improved Readability in Code:

Lowercase with underscores is a common convention in Python, matching the snake_case style for variable names. This makes the code more readable and consistent.
Example: df['first_name'] is easier to read and work with than df['FirstName'].

4. Cross-Platform Compatibility:

Lowercase column names are useful when moving data between different systems or databases, as some environments may be case-insensitive or treat column names differently (e.g., SQL databases).


```python
# # Convert all column names to lowercase
# df.columns = df.columns.str.lower()

# # Display the updated column names
# print(df.columns)

```

If it happens that you just want to change specific columns to Lowercase


```python
# List the specific columns you want to change to lowercase
columns_to_lowercase = ['CREDITCARD_TYPE', 'CUSTOMER_NAME']  # Example column names

# Apply lowercase transformation to the specific columns
df.rename(columns={col: col.lower() for col in columns_to_lowercase}, inplace=True)

# Display the updated column names
print(df.columns)

```

### Changing values to lower case



```python
# List of specific columns where you want to change the values to lowercase
columns_to_lower = ['CUSTOMER_NAME', 'EMAIL']  # Example column names

# Apply .str.lower() to each specified column
for col in columns_to_lower:
    df[col] = df[col].str.lower()

# Display the modified DataFrame
df.head()

```


```python
# Change the entries of a single column to lowercase
df['CUSTOMER_NAME'] = df['CUSTOMER_NAME'].str.lower()

# Display the updated DataFrame
df.head()

```


```python
# # Check for missing values in each column
# missing_values = df.isna().sum()
# print(missing_values)

```


```python
# # Check for duplicated rows
# duplicate_rows = df.duplicated().sum()
# print(f"Number of duplicated rows: {duplicate_rows}")

```

We can do some visualizations to better understand our data


## Customer Distribution by Gender
A bar chart can show the distribution of customers based on their gender.


```python
import seaborn as sns
import matplotlib.pyplot as plt

# Gender distribution bar plot
sns.countplot(x='Gender_Code', data=df)
plt.title('Customer Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

```


    
![png](output_52_0.png)
    


### Age Distribution of Customers
A histogram can show the distribution of customers' ages.


```python
# Age distribution histogram
sns.histplot(df['AGE'], kde=True, bins=20)
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_54_0.png)
    


It will not work as the age column contains inconsistent data.


```python
import pandas as pd

# Assume df is already your DataFrame and has the 'AGE' column

# Remove non-numeric characters from the 'AGE' column
df['AGE'] = df['AGE'].astype(str).str.extract('(\d+)')  # Extract numeric part of the string
df['AGE'] = df['AGE'].astype(float).astype('Int64')  # Convert to integer (handle NaNs if necessary)

# Check the result
print(df['AGE'].head())

```

    0    27
    1    27
    2    27
    3    27
    4    27
    Name: AGE, dtype: Int64
    

### Sales Distribution (Order Value)
A boxplot can show the spread of ORDER_VALUE and identify any outliers in the data.


```python
# Boxplot of order value
sns.boxplot(y='ORDER_VALUE', data=df)
plt.title('Distribution of Order Values')
plt.ylabel('Order Value')
plt.show()
```


    
![png](output_58_0.png)
    


### Sales Trend Over Time
We can look at sales trends, assuming you have ORDER_DATE, you can plot the trend over time.
