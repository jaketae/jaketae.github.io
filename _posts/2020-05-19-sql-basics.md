---
title: SQL Basics with Pandas
mathjax: false
toc: true
categories:
  - study
tags:
  - jupyter
  - sql
---

Recently, I was compelled by my own curiosity to study SQL, a language I have heard about quite a lot but never had a chance to study. At first, SQL sounded difficult and foreign largely because it was a language fundamentally different from other programming languages I had studied, such as Java, Python, or R. However, after watching this [fantastic video tutorial](https://www.youtube.com/watch?v=HXV3zeQKqGY&t=3677s) on YouTube, and completing a relevant course on DataCamp, I think I now finally have a somewhat concrete understanding of what SQL is and how to use it. Of course, I'm still so far away from being fluent in SQL, and the queries I can write are still pretty basic. Much like the [blog post on R](https://jaketae.github.io/development/R-tutorial/), this post will serve as a reference for myself.  

*Note*: This notebook was drafted in January of 2020, yet I never had a chance to finish it. Finally, while working on some R tutorial notebooks on the `dpylr` package, I was reminded of this draft and hence decided to publish it. Hopefully this editorial discontinuity does not affect the quality of writing and content of this article.

# SQL with Jupyter

There are many different ways of using and accessing SQL from Jupyter Notebooks. Here, I introduce two simple ways of practicing SQL without much complicated setup.

## Magic Command

The first on the list is `ipython-sql`, which allows us to use magic commands in Jupyter notebooks. To install, simply type the following line in the terminal, assuming that you have activated the conda virtual environment of choice. 

```bash
conda install ipython-sql
```

We can now use the `load_ext sql` magic command in Jupyter to connect to a local database. In my case I had a MySQL database initialized at localhost, and was able to connect to it as a root user. Note that you should replace `some_password` in the example command below according to your own configuration.


```python
%load_ext sql
%sql mysql+pymysql://root:some_password@localhost:3306/test
```


    'Connected: root@test'

Now that we have successfully connected to the data, we can use SQL commands in Jupyter!


```python
%sql SELECT * FROM employee ORDER BY sex, first_name, last_name LIMIT 5;
```

     * mysql+pymysql://root:***@localhost:3306/test
    5 rows affected.

<table>
    <tr>
        <th>emp_id</th>
        <th>first_name</th>
        <th>last_name</th>
        <th>birth_day</th>
        <th>sex</th>
        <th>salary</th>
        <th>super_id</th>
        <th>branch_id</th>
    </tr>
    <tr>
        <td>103</td>
        <td>Angela</td>
        <td>Martin</td>
        <td>1971-06-25</td>
        <td>F</td>
        <td>63000</td>
        <td>102</td>
        <td>2</td>
    </tr>
    <tr>
        <td>101</td>
        <td>Jan</td>
        <td>Levinson</td>
        <td>1961-05-11</td>
        <td>F</td>
        <td>110000</td>
        <td>100</td>
        <td>1</td>
    </tr>
    <tr>
        <td>104</td>
        <td>Kelly</td>
        <td>Kapoor</td>
        <td>1980-02-05</td>
        <td>F</td>
        <td>55000</td>
        <td>102</td>
        <td>2</td>
    </tr>
    <tr>
        <td>107</td>
        <td>Andy</td>
        <td>Bernard</td>
        <td>1973-07-22</td>
        <td>M</td>
        <td>65000</td>
        <td>106</td>
        <td>3</td>
    </tr>
    <tr>
        <td>100</td>
        <td>David</td>
        <td>Wallace</td>
        <td>1967-11-17</td>
        <td>M</td>
        <td>250000</td>
        <td>None</td>
        <td>1</td>
    </tr>
</table>



This method works, but it requires that you set up a MySQL server on your local workstation. While this is not particularly difficult, this method is somewhat made less compelling by the fact that it does not work right out of the box. The method I prefer, therefore, is the one that I would like to introduce next.

## PandaSQL

`pandas` is an incredibly widely used Python module for deailng with tabular data. It some similarities with SQL in that they both deal with tables at the highest level. Of course, the two serve very different purposes: SQl is intended as a backend exclusive language, powering huge database servers and allowing developers to quickly query through large amounts of data. `pandas`, on the other hand, is a must-have in the Python data scientist's toolbox, allowing them to extract new insight from organized tabular data. 

`pandasql` is a Python module that allows us to query `pandas.DataFrame`s using SQL syntax. In other words, it is a great way to learn SQL. The benefit of this approach is that no database setup is necessary: as long as there is some tabular data to work with, say some `.csv` file, we are ready to go. For the purposes of this post, we will thus be using this latter approach. 

With all that said, let's get started.


```python
import pandas as pd
from pandasql import sqldf
```

# Select Syntax

In this section, we will go over some basic core SQL statements to get our feet wet. It would be utterly impossible for me to cover SQL syntax in any level of detail in a single blog post, but this is a start nonetheless. At the minimum, I hope to continue this series as I start learning more SQL. The main references used to write this post were this [excellent Medium article](https://medium.com/jbennetcodes/how-to-rewrite-your-sql-queries-in-pandas-and-more-149d341fc53e) and the [official documentation](https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html) on the `pandas` website.

Let's begin by loading a library to import some sample toy datasets at our disposal.


```python
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
iris.head()
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



## Limit

Let's first see a simple example of `SELECT` in action, alongwith `LIMIT`.


```python
sqldf("SELECT sepal_length, petal_width FROM iris LIMIT 5;")
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
      <th>sepal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



One of the perks of SQL is that it somewhat reads like plain English instead of complicated computer code. Of course, SQL statements can get quite complex, in which case this rule starts to break down. However, it isn't too difficult to see what the statement above is doing: it is selecting the column `sepal_length` and `petal_width` from the `iris` dataset which we loaded, and showing the top five results only in accordance with the `LIMIT`.

We can also replicate the output of `iris.head()` by doing the following.


```python
sqldf("SELECT * FROM iris LIMIT 5;")
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



The `*` is essentially a wild card argument that tells SQL that we want information pulled from every column instead of a specified few. This can be handy when we want to take a glimpse of the contents of the database.

## Distinct

`LIMIT` is not the only addition we can make to a `SELECT` statement. For instance, consider the keyword `DISTINCT`, which does exactly what you think it does:


```python
sqldf("SELECT DISTINCT species FROM iris;")
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
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>versicolor</td>
    </tr>
    <tr>
      <th>2</th>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, `DISTINCT` allows us to select only unique values in the table. Note that `pandas` offers a simliar function, `unique`, with which we can somewhat recreate a similar result.


```python
iris.species.unique()
```




    array(['setosa', 'versicolor', 'virginica'], dtype=object)



## Where

Another useful fact to remember is that `SELECT` most often goes along with `WHERE`. We can imagine many instances where we would want to retrieve only those data entries that satisfy a certain condition. In the example below, we retrieve only those data entries whose species are labeled as `setosa`. 


```python
sqldf('''SELECT petal_length 
         FROM iris 
         WHERE species = 'setosa' 
         LIMIT 5;''')
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
      <th>petal_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.4</td>
    </tr>
  </tbody>
</table>
</div>



In `pandas` speak, we would have done the following:


```python
iris[iris.species == 'setosa'].petal_length.head(5)
```




    0    1.4
    1    1.4
    2    1.3
    3    1.5
    4    1.4
    Name: petal_length, dtype: float64



The `pandas` version is not too difficult just yet, butI prefer SQL's resemblance to plain human language. Just for the sake of it, let's take a look at a slightly more complicated conditioning we can perform with `WHERE`, namely by linking multiple conditions on top of each other. In this example, we select `petal_width` and `petal_length` for only those entries whose species is setosa and `sepal_width` is smaller than 3.2 (this number is entirely random).


```python
sqldf('''SELECT petal_width, petal_length
         FROM iris
         WHERE species = 'setosa'
         and sepal_width < 5
         LIMIT 5;''')
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
      <th>petal_width</th>
      <th>petal_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.2</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.2</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.2</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.2</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.2</td>
      <td>1.4</td>
    </tr>
  </tbody>
</table>
</div>



All we did there was join the two conditions via the `AND` keyword. In `pandas`, this is made slighty more confusing by the fact that we use slicing to make multi-column selections. 


```python
iris[(iris.species == 'setosa') & (iris.sepal_width < 5)][['petal_width', 'petal_length']][:5]
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
      <th>petal_width</th>
      <th>petal_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.2</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.2</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.2</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.2</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.2</td>
      <td>1.4</td>
    </tr>
  </tbody>
</table>
</div>



And by the same token, the `OR` SQL keyword translates into `|` in `pandas`.

Instead of sticking `[:5]` in the end, we could have used `.head()` as we have been doing so far. It isn't difficult to see that introducing more conditionals can easily result in somewhat more longer statements in Python, whereas that is not necessarily the case with SQL. This is not to say that `pandas` is inferior or poorly optimized; instead, it simply goes to show that the two platforms have their own comaprative advantages and that they mainly serve different purposes.

## Sort

Often time when sorting through some tabular data, we want to sort the entries in ascending or descending order according to some axis. For example, we might want to rearrange the entries so that one with the largest `petal_width` comes first. Let's see how we can achieve this with SQL.


```python
sqldf('''SELECT *
         FROM iris
         ORDER BY petal_width
         DESC
         LIMIT 10;''')
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.3</td>
      <td>3.3</td>
      <td>6.0</td>
      <td>2.5</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.2</td>
      <td>3.6</td>
      <td>6.1</td>
      <td>2.5</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.7</td>
      <td>3.3</td>
      <td>5.7</td>
      <td>2.5</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.8</td>
      <td>2.8</td>
      <td>5.1</td>
      <td>2.4</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.3</td>
      <td>3.4</td>
      <td>5.6</td>
      <td>2.4</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.7</td>
      <td>3.1</td>
      <td>5.6</td>
      <td>2.4</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6.4</td>
      <td>3.2</td>
      <td>5.3</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.7</td>
      <td>2.6</td>
      <td>6.9</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6.9</td>
      <td>3.2</td>
      <td>5.7</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7.7</td>
      <td>3.0</td>
      <td>6.1</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
</div>



By default, the `ORDER BY` keyword in SQL lists values in asending order. To reverse this, we can explicitly add the `DESC` keyword. We see that the entries with the largets `petal_width` is indeed at the top of the selected query result.

We can also achieve a similar result in `pandas`.


```python
iris.sort_values('petal_width', ascending=False).head(10)
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>6.3</td>
      <td>3.3</td>
      <td>6.0</td>
      <td>2.5</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>109</th>
      <td>7.2</td>
      <td>3.6</td>
      <td>6.1</td>
      <td>2.5</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>144</th>
      <td>6.7</td>
      <td>3.3</td>
      <td>5.7</td>
      <td>2.5</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>114</th>
      <td>5.8</td>
      <td>2.8</td>
      <td>5.1</td>
      <td>2.4</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>140</th>
      <td>6.7</td>
      <td>3.1</td>
      <td>5.6</td>
      <td>2.4</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>136</th>
      <td>6.3</td>
      <td>3.4</td>
      <td>5.6</td>
      <td>2.4</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>115</th>
      <td>6.4</td>
      <td>3.2</td>
      <td>5.3</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>143</th>
      <td>6.8</td>
      <td>3.2</td>
      <td>5.9</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
</div>



In this case, I think `pandas` also offers a simple, clean interface to access data. One point to note is that both SQL and `pandas` have the same default settings when it comes to ordering or sorting entries: by default, `ascending=True`. Also, it is interesting to see that SQL does not have references to row values or IDs because we did not set them up, whereas `pandas` automatically keeps track of the original location of each row and displays them in the queried result. 

## Is X

I decided to jam-pack this last section with a bunch of somewhat similar commands: namely, `isin()`, `isna()`, and `notna()`. These commands are loosely related to each other, which is why they are all grouped under this section. Speaking of groups, we will continue our discussion of SQL and `pandas` in another post, starting with things lilke `groupby()` and `GROUP BY`. Anyhow, let's begin by taking a look at `isin()`. 

### In

In SQL, we can make selections based on whether an entry falls into a certain category. For instance, we might want to select data points only for setosas and virginicas. In that case, we might use the following SQL statement.


```python
sqldf('''SELECT *
         FROM iris
         WHERE species IN ('virginica', 'setosa');''')
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>96</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>97</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>98</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>99</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 5 columns</p>
</div>



To demonstrate the fact that we have both setosas and virginicas, I decided to avoid the use of `LIMIT`. The resulting table is has 100 rows and five columns. Let's see if we can replicate this result in `pandas` using `isin()`.


```python
iris[iris.species.isin(['virginica', 'setosa'])]
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 5 columns</p>
</div>



As expected, we also get a 100-by-5 table containing only setosas and virginicas. 

It is worth noting that this was not the smartest way to go about the problem; we could have used negative boolean indexing: namely, we could have told `pandas` to pull every data point but those pertaining to versicolors. For example, 


```python
iris[~(iris.species == 'versicolor')]
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 5 columns</p>
</div>



In development settings, we would of course use the negative boolean indexing approach shown immediately above, but for demonstration purposes, it helps to see how `isin()` can be used to model `IN`.

### N/A

In SQL, empty values are encoded as `NULL`. We can perform selection based on whether or not there is a `NULL` entry in a row. This functionality is particularly important to preprocess data, which might be fed into some machine learning model. 

First, observe that the current `iris` data does not have any `NULL` values. 


```python
sum(iris.isna().any())
```




    0



Therefore, let's add two dummy rows for the purposes of this demonstration. There are many ways to go about adding a row. For example, we might want to assign a new row by saying `iris.loc[-1] = some_row_data`, or use `pd.concat([iris, dummy_df])`. Let's try the first approach for simplicity.


```python
iris.iloc[-1] = [5.9, 3.0, 5.1, 1.8, None]
```

Now that we have this dummy row, let's see what we can do with SQL. In fact, the syntax is not so much different from what we've been doing so far.


```python
sqldf('''SELECT *
         FROM iris
         WHERE species IS NULL;''')
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



The only new part is `IS NULL`, which specifies that a certain attribute or column is `NULL`. Again, this is one of those instances that show that SQL statements somewhat read like normal English statements. `pandas`, on the other hand, obviously doesn't flow as easily, but its syntax is not so much complicated either:


```python
iris[iris.species.isna()]
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



Again, this shows that boolean indexing is a huge component of sifting through `pandas` data frames. A lot of the inspiration behind this API obviously comes from R and  treatment of its own data frames. 

The `notna()` function does the exact opposite of `isna()`. Without running the function, we already know that substituting `isna()` with `notna()` will simply give us the rest of all the rows in the `iris` dataset.

# Conclusion

This post was intended as an introduction to SQL, but somehow it digressed into a comparison of SQL and `pandas` syntax. Nonetheless, for those who are already familiar with one framework, reading this cross comparison will help you glean a more intuitive sense of what the other side of the world looks like. As mentioned earlier, SQL and `pandas` each have their strenghts and weaknesses, and so it definitely helps to have both tools under the belt. As you might notice, my goal is to eventually gain some level of proficienchy in both Python, SQL, and R; hence the posts on R lately. It's interesting to see how different tools can be used to approach the same problem. Better illuminated in that process are the philosophies behind each frameworks: where they each borrowed inspiration from, what values or UX aspects they prioritize, and et cetera. 

I hope you enjoyed reading this post. Catch you up in the next one!
