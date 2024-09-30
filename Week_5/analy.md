### Data Cleaning and Data Preprocessing Techniques
Data preprocessing is the process of cleaning, transforming, and organizing raw 

data to make it suitable for analysis. It involves various steps, 

such as dealing with missing values, normalizing data, encoding categorical variables, and scaling features.

In pandas, data preprocessing can be done through a variety of steps:

1. **Handling Missing Values:**
   - Replace with a value: `df.fillna(value)`
   - Drop rows/columns with missing values: `df.dropna()`

2. **Encoding Categorical Data:**
   - Convert categorical values to numerical: `pd.get_dummies(df['column'])` or using `LabelEncoder`

3. **Scaling Data:**
   - Normalizing columns: `(df['column'] - df['column'].mean()) / df['column'].std()`
   - Using scikit-learn's `StandardScaler` with pandas for standardization

4. **Removing Duplicates:**
   - `df.drop_duplicates()`

5. **Renaming Columns:**
   - `df.rename(columns={'old_name': 'new_name'})`

6. **Feature Engineering:**
   - Create new columns by combining or transforming existing ones: `df['new_column'] = df['col1'] + df['col2']`

These are some common steps, but the exact process depends on the dataset and problem requirements.

## Exploratory Data Analysis (EDA)

EDA helps you understand the structure and characteristics of your dataset. Some Pandas functions help us gain insights into our dataset. We call these functions by calling the dataset variable plus the function.

For example:

df.head() will call the first 5 rows of the dataset. You can specify the number of rows to be displayed in the parentheses.

df.describe() gives some statistical data like percentile, mean and standard deviation of the numerical values of the Series or DataFrame.

df.info() gives the number of columns, column labels, column data types, memory usage, range index, and the number of cells in each column (non-null values).


```python
import pandas as pd
```


```python
data = "https://github.com/NobleezIT/3MTT_Cohort2/blob/main/Week_3/customer.csv?raw=true"
df = pd.read_csv(data)
```


```python
# Loading the Data
df = pd.read_csv(r"C:\Users\abula\Desktop\dataanalytics\Week_3\customer.csv")
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
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>199</th>
      <td>Rebecca Uresti</td>
      <td>Mrs.</td>
      <td>4333 Morningview Lane</td>
      <td>Birmingham</td>
      <td>AL</td>
      <td>US</td>
      <td>35203</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Rebecca.C.Uresti@trashymail.com</td>
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
      <th>200</th>
      <td>Emma Perez</td>
      <td>Mrs.</td>
      <td>1199 Lightning Point Drive</td>
      <td>Black Hawk</td>
      <td>SD</td>
      <td>US</td>
      <td>57718</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Emma.V.Perez@spambob.com</td>
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
      <th>201</th>
      <td>Reynaldo Myers</td>
      <td>Mr.</td>
      <td>3923 Black Stallion Road</td>
      <td>Blaxland East</td>
      <td>NSW</td>
      <td>AU</td>
      <td>2774</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Reynaldo.J.Myers@pookmail.com</td>
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
      <th>202</th>
      <td>Reynaldo Myers</td>
      <td>Mr.</td>
      <td>3923 Black Stallion Road</td>
      <td>Blaxland East</td>
      <td>NSW</td>
      <td>AU</td>
      <td>2774</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Reynaldo.J.Myers@pookmail.com</td>
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
      <th>203</th>
      <td>Jennifer Becker</td>
      <td>Mrs.</td>
      <td>4374 Lindale Avenue</td>
      <td>Bloomington</td>
      <td>IL</td>
      <td>US</td>
      <td>61701</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Jennifer.D.Becker@spambob.com</td>
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
<p>204 rows × 72 columns</p>
</div>




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
df = df.dropna(axis=1, how='all')
```

### What is Data Cleaning
Data cleaning involves identifying and rectifying errors, inconsistencies, 

and missing values within a dataset. It's like preparing your ingredients 

before cooking; you want everything in order to get the perfect analysis or visualization.

Data cleaning is the initial phase of refining your dataset, 

making it readable and usable with techniques like removing duplicates, 

handling missing values and data type conversion while data preprocessing is 

similar to taking this refined data and scaling with more advanced 

techniques such as feature engineering, encoding categorical 

variables and and handling outliers to achieve better and more advanced results.

The goal is to turn your dataset into a refined masterpiece, ready for analysis or modeling.


### How to Handle Missing Values
As a newbie in this field, missing values pose a significant stress as they come in different formats and can adversely impact your analysis or model.

Machine learning models cannot be trained with data that has missing or "NAN" values as they can alter your end result during analysis. But do not fret, Pandas provides methods to handle this problem.


```python
#Check for missing values
print(df.isnull().sum())

#Drop rows with missing valiues and place it in a new variable "df_cleaned"
# df_cleaned = df.dropna()

#Fill missing values with mean for numerical data and place it ina new variable called df_filled
# df_filled = df.fillna(df.mean())
# We would come back to this
```

    CUST_NAME               5
    Gender_Code             5
    ADDRESS1                5
    CITY                    5
    STATE                  41
    COUNTRY_CODE            5
    POSTAL_CODE             5
    POSTAL_CODE_PLUS4       5
    EMAIL_ADDRESS           5
    PHONE_NUMBER            5
    CREDITCARD_TYPE         5
    SALESMAN_ID             5
    NATIONALITY             5
    NATIONAL_ID             5
    CREDITCARD_NUMBER       5
    CUST_ID                 5
    ORDER_ID                5
    ORDER_DATE              5
    ORDER_TIME              5
    FREIGHT_CHARGES         5
    ORDER_SALESMAN          5
    ORDER_POSTED_DATE       5
    ORDER_SHIP_DATE         5
    AGE                    29
    ORDER_VALUE             5
    T_TYPE                  5
    PURCHASE_TOUCHPOINT     5
    PURCHASE_STATUS         5
    ORDER_TYPE              5
    GENERATION              5
    Baby Food               5
    Diapers                 5
    Formula                 5
    Lotion                  5
    Baby wash               5
    Wipes                   5
    Fresh Fruits            5
    Fresh Vegetables        5
    Beer                    5
    Wine                    5
    Club Soda               5
    Sports Drink            5
    Chips                   5
    Popcorn                 5
    Oatmeal                 5
    Medicines               5
    Canned Foods            5
    Cigarettes              5
    Cheese                  5
    Cleaning Products       5
    Condiments              5
    Frozen Foods            5
    Kitchen Items           5
    Meat                    5
    Office Supplies         5
    Personal Care           5
    Pet Supplies            5
    Sea Food                5
    Spices                  5
    dtype: int64
    

But if the number of rows that have missing values is large, then this method will be inadequate.

For numerical data, you can simply compute the mean and input it into the rows that have missing values. Code snippet below:


```python
df['AGE'].isnull().sum()
```




    29




```python
df['AGE'].fillna(df['AGE'].mean(), inplace=True)
```


```python
#Replace missing values with the mean of each column
df.age.fillna(df.age.mean(), inplace=True)
df.fillna
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
df1 = df.drop_duplicates()
```

    9
    


```python
df1.duplicated().sum()
df1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 195 entries, 0 to 203
    Data columns (total 59 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   CUST_NAME            194 non-null    object 
     1   Gender_Code          194 non-null    object 
     2   ADDRESS1             194 non-null    object 
     3   CITY                 194 non-null    object 
     4   STATE                158 non-null    object 
     5   COUNTRY_CODE         194 non-null    object 
     6   POSTAL_CODE          194 non-null    object 
     7   POSTAL_CODE_PLUS4    194 non-null    float64
     8   EMAIL_ADDRESS        194 non-null    object 
     9   PHONE_NUMBER         194 non-null    object 
     10  CREDITCARD_TYPE      194 non-null    object 
     11  SALESMAN_ID          194 non-null    object 
     12  NATIONALITY          194 non-null    object 
     13  NATIONAL_ID          194 non-null    object 
     14  CREDITCARD_NUMBER    194 non-null    float64
     15  CUST_ID              194 non-null    float64
     16  ORDER_ID             194 non-null    float64
     17  ORDER_DATE           194 non-null    object 
     18  ORDER_TIME           194 non-null    object 
     19  FREIGHT_CHARGES      194 non-null    float64
     20  ORDER_SALESMAN       194 non-null    object 
     21  ORDER_POSTED_DATE    194 non-null    object 
     22  ORDER_SHIP_DATE      194 non-null    object 
     23  AGE                  170 non-null    object 
     24  ORDER_VALUE          194 non-null    float64
     25  T_TYPE               194 non-null    object 
     26  PURCHASE_TOUCHPOINT  194 non-null    object 
     27  PURCHASE_STATUS      194 non-null    object 
     28  ORDER_TYPE           194 non-null    object 
     29  GENERATION           194 non-null    object 
     30  Baby Food            194 non-null    float64
     31  Diapers              194 non-null    float64
     32  Formula              194 non-null    float64
     33  Lotion               194 non-null    float64
     34  Baby wash            194 non-null    float64
     35  Wipes                194 non-null    float64
     36  Fresh Fruits         194 non-null    float64
     37  Fresh Vegetables     194 non-null    float64
     38  Beer                 194 non-null    float64
     39  Wine                 194 non-null    float64
     40  Club Soda            194 non-null    float64
     41  Sports Drink         194 non-null    float64
     42  Chips                194 non-null    float64
     43  Popcorn              194 non-null    float64
     44  Oatmeal              194 non-null    float64
     45  Medicines            194 non-null    float64
     46  Canned Foods         194 non-null    float64
     47  Cigarettes           194 non-null    float64
     48  Cheese               194 non-null    float64
     49  Cleaning Products    194 non-null    float64
     50  Condiments           194 non-null    float64
     51  Frozen Foods         194 non-null    float64
     52  Kitchen Items        194 non-null    float64
     53  Meat                 194 non-null    float64
     54  Office Supplies      194 non-null    float64
     55  Personal Care        194 non-null    float64
     56  Pet Supplies         194 non-null    float64
     57  Sea Food             194 non-null    float64
     58  Spices               194 non-null    float64
    dtypes: float64(35), object(24)
    memory usage: 91.4+ KB
    

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


```python
# Summary of data types and non-null values
df.info()
```


```python
# Descriptive statistics for numerical columns
df.describe()
```

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


```python
# Using isna()
missing_row = df.isna().sum(axis=1)
print(missing_row)
```

This will show the number of missing values per row. If a row has the total number of columns as NaN, that means it's an empty row.

To remove empty rows in your dataset, you can use the dropna() function, which drops rows where all the values are NaN (i.e., empty).


```python
# Drop rows where all values are missing
# df.dropna(how='all', inplace=True)

# Verify the dataset after removing empty rows
df.shape  # This will show the new shape of the dataset after removal
```

## Checking and Removing Duplicate Values


```python
# Check for Duplicate values
duplicates = df.duplicated()

# Display duplicated rows
print(df['duplicates'])
```


```python
# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Verify that duplicates are removed
print(df.shape)  # Check the new shape of the dataset
```

### Check for Duplicates Based on Specific Columns

You can specify the columns on which you want to check for duplicates. This will identify rows that have identical values for those columns.


```python
# Check for duplicates based on specific columns (e.g., 'CUST_NAME' and 'EMAIL_ADDRESS')
duplicates_based_on_columns = df[df.duplicated(subset=['CUST_NAME', 'EMAIL_ADDRESS'], keep=False)]

# Display the duplicates
print(duplicates_based_on_columns)
```

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
df.columns
```




    Index(['CUST_NAME', 'Gender_Code', 'ADDRESS1', 'CITY', 'STATE', 'COUNTRY_CODE',
           'POSTAL_CODE', 'POSTAL_CODE_PLUS4', 'EMAIL_ADDRESS', 'PHONE_NUMBER',
           'CREDITCARD_TYPE', 'SALESMAN_ID', 'NATIONALITY', 'NATIONAL_ID',
           'CREDITCARD_NUMBER', 'CUST_ID', 'ORDER_ID', 'ORDER_DATE', 'ORDER_TIME',
           'FREIGHT_CHARGES', 'ORDER_SALESMAN', 'ORDER_POSTED_DATE',
           'ORDER_SHIP_DATE', 'AGE', 'ORDER_VALUE', 'T_TYPE',
           'PURCHASE_TOUCHPOINT', 'PURCHASE_STATUS', 'ORDER_TYPE', 'GENERATION',
           'Baby Food', 'Diapers', 'Formula', 'Lotion', 'Baby wash', 'Wipes',
           'Fresh Fruits', 'Fresh Vegetables', 'Beer', 'Wine', 'Club Soda',
           'Sports Drink', 'Chips', 'Popcorn', 'Oatmeal', 'Medicines',
           'Canned Foods', 'Cigarettes', 'Cheese', 'Cleaning Products',
           'Condiments', 'Frozen Foods', 'Kitchen Items', 'Meat',
           'Office Supplies', 'Personal Care', 'Pet Supplies', 'Sea Food',
           'Spices'],
          dtype='object')




```python
# # Convert all column names to lowercase
df.columns = df.columns.str.lower()
# x = "ABDUL"
# x.lower()
# # Display the updated column names
print(df.columns)

```

    Index(['cust_name', 'gender_code', 'address1', 'city', 'state', 'country_code',
           'postal_code', 'postal_code_plus4', 'email_address', 'phone_number',
           'creditcard_type', 'salesman_id', 'nationality', 'national_id',
           'creditcard_number', 'cust_id', 'order_id', 'order_date', 'order_time',
           'freight_charges', 'order_salesman', 'order_posted_date',
           'order_ship_date', 'age', 'order_value', 't_type',
           'purchase_touchpoint', 'purchase_status', 'order_type', 'generation',
           'baby food', 'diapers', 'formula', 'lotion', 'baby wash', 'wipes',
           'fresh fruits', 'fresh vegetables', 'beer', 'wine', 'club soda',
           'sports drink', 'chips', 'popcorn', 'oatmeal', 'medicines',
           'canned foods', 'cigarettes', 'cheese', 'cleaning products',
           'condiments', 'frozen foods', 'kitchen items', 'meat',
           'office supplies', 'personal care', 'pet supplies', 'sea food',
           'spices'],
          dtype='object')
    

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
sns.countplot(x='gender_code', data=df)
plt.title('Customer Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

```


    
![png](output_57_0.png)
    


### Age Distribution of Customers
A histogram can show the distribution of customers' ages.


```python
# Age distribution histogram
sns.histplot(df['age'], kde=True, bins=20)
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_59_0.png)
    


It will not work as the age column contains inconsistent data.


```python
import pandas as pd

# Assume df is already your DataFrame and has the 'AGE' column

# Remove non-numeric characters from the 'AGE' column
df['age'] = df['age'].astype(str).str.extract('(\d+)')  # Extract numeric part of the string
df['Aa'] = df['age'].astype(float)  # Convert to integer (handle NaNs if necessary)
# df.age.unique()
# Check the result
# print(df['AGE'].head())
df.age.unique()
```


```python
df['age'].fillna(48)
```




    48.17142857142857



### Sales Distribution (Order Value)
A boxplot can show the spread of ORDER_VALUE and identify any outliers in the data.


```python
# Boxplot of order value
sns.boxplot(y='order_value', data=df)
plt.title('Distribution of Order Values')
plt.ylabel('Order Value')
plt.show()
```


    
![png](output_64_0.png)
    


### Sales Trend Over Time
We can look at sales trends, assuming you have ORDER_DATE, you can plot the trend over time.

# Assignment
Download and load the dataset in this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv) in a notebook. Perform EDA and clean the data where needed.
