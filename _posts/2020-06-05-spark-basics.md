---
title: Introduction to PySpark
mathjax: false
toc: true
categories:
  - study
tags:
  - spark
---

I've stumbled across the word "Apache Spark" on the internet so many times, yet I never had the chance to really get to know what it was. For one thing, it seemed rather intimidating, full of buzzwords like "cloud computing", "data streaming," or "scalability," just to name a few among many others. However, a few days ago, I decided to give it a shot and try to at least get a glimpse of what it was all about. So here I report my findings after binge watching online tutorials on Apache Spark. 

# Apache

If you're into data science or even just software development, you might have heard some other Apache variants like Apache Kafka, Apache Cassandra, and many more. When I first heard about these, I began wondering: is Apache some sort of umbrella software, with Spark, Kafka, and other variants being different spinoffs from this parent entity? I was slightly more confused because the Apache I had heard of, at least as far as I recalled, had to do with web servers and hosting. 

Turns out that there is an organization called the Apache Software Foundation, which is the world's largest open source foundation. This foundation, of course, has to do with the Apache HTTP server project, which was the web server side of things that I had ever so faintly heard about. Then what is Apache Spark? Spark was originally developed at UC Berkeley at the AMP Lab. Later, its code base was open sourced and eventually donated to the Apache Software Foundation; hence its current name, Apache Spark. 

# Setup

For this tutorial, we will be loading Apache Spark on Jupypter notebook. There are many tutorials on how to install Apache Spark, and they are easy to follow along. However, I'll also share a quick synopsis of my own just for reference.

## Installation

Installing Apache Spark is pretty straight forward if you are comfortable dealing with `.bash_profile` on macOS or `.bashrc` on Linux. The executive summary is that you need to add Apache Spark binaries to the `PATH` variable of your system. 

What is a `PATH`? Basically, the `PATH` variable is where all your little UNIX programs live. For example, when we run simple commands like `ls` or `mkdir`, we are essentially invoking built-in mini-programs in our POSIX system. The `PATH` variable tells the computer where these mini-programs reside in, namely `/usr/bin`, which is by default part of the `PATH` variable. 

Can the `PATH` variable be modified? The answer is a huge yes. Say we have our own little mini-program, and we want to be able to run it from the command line prompt. Then, we would simply modify `PATH` so that the computer knows where our custom mini-program is located and know what to do whenever we type some command in the terminal. 

This is why we enter the Python shell in interactive mode when we type `python` on the terminal. Here is the little setup I have on my own `.bash_profile`:

```bash
export PYTHON_HOME="/Library/Frameworks/Python.framework/Versions/3.7"
export SPARK_HOME="/Users/jaketae/opt/apache-spark/spark-2.4.5-bin-hadoop2.7"
export JAVA_HOME="/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home"
export PATH="${PYTHON_HOME}/bin:${PATH}:${SPARK_HOME}/bin"
```

Here, I prepended `PYTHON_HOME` to the default `PATH` then appended `SPARK_HOME` at the end. Appending and prepending result in different behaviors: by default, the computer searches for commands in the `PATH` variable in order. In other words, in the current setup, the computer will first search the `PYTHON_HOME` directory, then search the default `PATH` directory, and look at `SPARK_HOME` the very last, at least in my current setup. 

Note that Spark has specific Java requirements that may or may not align with the default Java installation on your workstation. In my case, I had to install a different version of Java and apply certain configurations. The `JAVA_HOME` path variable is a result of this minor modification.

The contents in the `SPARK_HOME` directory simply contains the result of unzipping the `.tar` file available for download on the Apache Spark website. 

Once the `PATH` variable has been configured, run `source ~/.bash_profile`, and you should be ready to run Apache Spark on your local workstation! To see if the installation and `PATH` configuration has been done correctly, type `pyspark` on the terminal:

```
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 2.4.5
      /_/

Using Python version 3.7.5 (default, Oct 25 2019 10:52:18)
SparkSession available as 'spark'.
>>> 
```

## Jupyter Notebook

To use Jupyter with Spark, we need to do a little more work. There are two ways to do this, but I will introduce the method that I found not only fairly simple, but also more applicable and generalizable. All we need is to install `findspark` package via `pip install findspark`. Then, on Jupyter, we can do:


```python
import findspark
findspark.init()
```

Then simply import Apache Spark via


```python
import pyspark
from pyspark import SparkContext, SparkConf
```

That is literally all we need! We can of course still use Apache Spark on the terminal simply by typing `pyspark` if we want, but it's always good to have more options on the table. 

# RDD Basics

The RDD API is the most basic way of dealing with data in Spark. RDD stands for "Resilient Distributed Dataset." Although more abstracted, higher-level APIs such as Spark SQL or Spark dataframes are becoming increasingly popular, thus challenging RDD's standing as a means of accessing and transforming data, it is a useful structure to learn nonetheless. One salient feature of RDDs is that computation in an RDD is parallelized across the cluster.

## Spark Context

To run Spark, we need to initialize a Spark context. A Spark context is the entry point to Spark that is needed to deal with RDDs. We can initialize one simply as follows: 


```python
sc = SparkContext(conf=SparkConf().setMaster("local[*]"))
```

Strictly speaking, the more proper way to do this would be to follow the syntax guideline on the [official website](https://spark.apache.org/docs/latest/rdd-programming-guide.html).

```python
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
```

However, we use a more simplified approach without initializing different apps for each example, simply for convenience purposes.

## Collect

Let's begin with a simple dummy example. Here, we turn a normal Python list into an RDD, then print out its contents after applying a squaring function. We use `.collect()` to turn the RDD into an iterable, specifically a list.


```python
nums = sc.parallelize([1,2,3,4])
squared = nums.map(lambda x: x**2)

for num in squared.collect():
    print(num)
```

    1
    4
    9
    16


## Count

Another useful function is `.count()` and `.countByValue()`. As you might have easily guessed, these functions are literally used to count the number of elements itself or their number of occurrences. This is perhaps best demonstrated by an example.


```python
input_words = ["spark", "hadoop", "spark", "hive", "pig", "cassandra", "hadoop"]

words = sc.parallelize(input_words)
words.count()
```




    7



`.countByValue()` works in a similar fashion, but it creates a dictionary of key value pairs, where the key is an element and the value is the count of that element in the input list.


```python
words.countByValue()
```




    defaultdict(int,
                {'spark': 2, 'hadoop': 2, 'hive': 1, 'pig': 1, 'cassandra': 1})



## Reduce

As the name implies, `.reduce()` is a way of reducing a RDD into something like a single value. In the example below, we reduce an RDD created with a list of numbers into a product of all the numbers in that original input list.


```python
print(f'RDD: {nums.collect()}')
prod = nums.reduce(lambda x, y: x * y)
prod
```

    RDD: [1, 2, 3, 4]





    24



## Filter

This was a pretty simple example. Let's take a look at a marginally more realistic of an example, albeit still extremely simple. I'm using the files from [this Spark tutorial](https://github.com/jleetutorial/python-spark-tutorial) on GitHub. After cloning the repo, we establish a base file path and retrieve the `airports.text` file we will be using in this example.


```python
BASE_PATH = '/Users/jaketae/documents/dev/python/spark-tutorial/in'

airports = sc.textFile(f'{BASE_PATH}/airports.text')
```

Let's see what this RDD looks like. We can do this via `.take()`, much like `.head()` in Pandas.


```python
airports.take(5)
```




    ['1,"Goroka","Goroka","Papua New Guinea","GKA","AYGA",-6.081689,145.391881,5282,10,"U","Pacific/Port_Moresby"',
     '2,"Madang","Madang","Papua New Guinea","MAG","AYMD",-5.207083,145.7887,20,10,"U","Pacific/Port_Moresby"',
     '3,"Mount Hagen","Mount Hagen","Papua New Guinea","HGU","AYMH",-5.826789,144.295861,5388,10,"U","Pacific/Port_Moresby"',
     '4,"Nadzab","Nadzab","Papua New Guinea","LAE","AYNZ",-6.569828,146.726242,239,10,"U","Pacific/Port_Moresby"',
     '5,"Port Moresby Jacksons Intl","Port Moresby","Papua New Guinea","POM","AYPY",-9.443383,147.22005,146,10,"U","Pacific/Port_Moresby"']



If you look carefully, you will realize that each element is a long string, not multiple elements separated by a comma as we would like. Let's define a helper function to split up each elements as we would like.


```python
def split_comma(line):
    words = line.split(',')
    for i, word in enumerate(words):
        if '"' in word:
            words[i] = word[1:-1]
    return words
```

Let's test out this function with the first element in the RDD.


```python
test = airports.take(1)[0]
split_comma(test)
```




    ['1',
     'Goroka',
     'Goroka',
     'Papua New Guinea',
     'GKA',
     'AYGA',
     '-6.081689',
     '145.391881',
     '5282',
     '10',
     'U',
     'Pacific/Port_Moresby']



Great! It worked as expected. Now let's say that we want to retrieve only those rows whose entries deal with airports in the United States. Specifically, we want the city and the name of the airport. How would we go about this task? Well, one simple idea would be to filter the data for airports in the United States, then only displaying the relevant information, namely the name of the airport and the city in which it is located.

Let's begin by defining the `airport_city` function.


```python
def airport_city(line):
    words = split_comma(line)
    return f'{words[1]}, {words[2]}'
```

And we test it on the first element to verify that it works as expected:


```python
airport_city(test)
```




    'Goroka, Goroka'



As stated earlier, we first filter the data set so that we only have entries that pertain to airports in the United States.


```python
us_airports = airports.filter(lambda line: split_comma(line)[3] == 'United States')
```

Then, we `map` the RDD using the `airport_city` function we defined above. This will transform all elements into the form we want: the name of the airport and the city. We actually used `.map()` above when we were dealing with square numbers. It's pretty similar to how map works in Python or other functional programming languages.


```python
us_airport_cities = us_airports.map(airport_city)
us_airport_cities.take(5)
```




    ['Putnam County Airport, Greencastle',
     'Dowagiac Municipal Airport, Dowagiac',
     'Cambridge Municipal Airport, Cambridge',
     'Door County Cherryland Airport, Sturgeon Bay',
     'Shoestring Aviation Airfield, Stewartstown']



## Flat Map

Now let's take a look at another commonly used operation: `flatMap()`. For this example, we load a text file containing prime numbers and create a RDD.


```python
prime_nums = sc.textFile(f'{BASE_PATH}/prime_nums.text')
prime_nums.take(5)
```




    ['  2\t  3\t  5\t  7\t 11\t 13\t 17\t 19\t 23\t 29',
     ' 31\t 37\t 41\t 43\t 47\t 53\t 59\t 61\t 67\t 71',
     ' 73\t 79\t 83\t 89\t 97\t101\t103\t107\t109\t113',
     '127\t131\t137\t139\t149\t151\t157\t163\t167\t173',
     '179\t181\t191\t193\t197\t199\t211\t223\t227\t229']



`flatMap()`, as the name implies, maps a certain operation over the elements of a RDD. The difference between `map()` and `flatMap()` is that the latter flattens the output. In this case, we split the numbers along `\t`. Normally, this would create a separate list for each line. However, since we also flatten that output, there is no distinction between one line and another.


```python
parsed_prime_nums = prime_nums.flatMap(lambda line: line.split('\t')).map(int)
parsed_prime_nums.take(20)
```




    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]



Just for a simple recap, let's try adding all the numbers in that RDD using `reduce()`.


```python
parsed_prime_nums.reduce(lambda x, y: x + y)
```




    24133



## Intersection

We can also consider RDDs to be likes sets, in the Pythonic or the mathematical sense, whichever you conceptually prefer. The idea is that we can use set operators, such as intersection or unions, to extract data we want from the RDD to create a new RDD. Below is an example using NASA records, each from July and August of 1995.


```python
july_logs = sc.textFile(f'{BASE_PATH}/nasa_19950701.tsv')
august_logs = sc.textFile(f'{BASE_PATH}/nasa_19950801.tsv')

july_logs.take(5)
```




    ['host\tlogname\ttime\tmethod\turl\tresponse\tbytes',
     '199.72.81.55\t-\t804571201\tGET\t/history/apollo/\t200\t6245\t\t',
     'unicomp6.unicomp.net\t-\t804571206\tGET\t/shuttle/countdown/\t200\t3985\t\t',
     '199.120.110.21\t-\t804571209\tGET\t/shuttle/missions/sts-73/mission-sts-73.html\t200\t4085\t\t',
     'burger.letters.com\t-\t804571211\tGET\t/shuttle/countdown/liftoff.html\t304\t0\t\t']



The task is to obtain hosts that are both in the July and August logs. We might want to break this task up into several discrete components. The first step would be to extract the host information from the original logs. We can do this simply by splitting and obtaining the first element of each resulting list. 


```python
july_hosts = july_logs.map(lambda line: line.split('\t')[0])
august_hosts = august_logs.map(lambda line: line.split('\t')[0])

august_hosts.take(5)
```




    ['host',
     'in24.inetnebr.com',
     'uplherc.upl.com',
     'uplherc.upl.com',
     'uplherc.upl.com']



Then, all we have to do is to apply the intersection operation, then filter out the first column header (i.e. `'host'`). 


```python
common_hosts = july_hosts.intersection(august_hosts)
common_hosts = common_hosts.filter(lambda line: line != 'host')
common_hosts.take(5)
```




    ['www-a1.proxy.aol.com',
     'www-d3.proxy.aol.com',
     'piweba1y.prodigy.com',
     'www-d4.proxy.aol.com',
     'piweba2y.prodigy.com']



Lastly, if we wanted to save the RDD to some output file, we would use the `saveAsTextFile()` function. Note that the output file would be split up into multiple files, since computation is distributed in Spark. Also, it is generally considered standard good practice to split up huge datasets into separate files, rather than coalescing all of them into a single one.


```python
common_hosts.saveAsTextFile("path_to_folder/file_title.csv")
```

# Pair RDD

So far, we have looked at RDDs. Now, let's turn our attention to another type of widely used RDDs: pair RDDs. Pair RDDs are widely used because they are, in a way, like dictionaries with key-value pairs. This key-value structure is very useful, since we can imagine there being a lot of operations where, for instance, a value is reduced according to their keys, or some elements are grouped by their keys, and et cetera. Let's take a look at what we can do with pair RDDs.

## Filter

Both `map` and `filter` operations work the same way as you would expect with normal RDDs. Let's first create a toy example using the `airports` RDD we looked at earlier. To remind ourselves of what this data looked like, we list the first five elements in the RDD.


```python
airports.take(5)
```




    ['1,"Goroka","Goroka","Papua New Guinea","GKA","AYGA",-6.081689,145.391881,5282,10,"U","Pacific/Port_Moresby"',
     '2,"Madang","Madang","Papua New Guinea","MAG","AYMD",-5.207083,145.7887,20,10,"U","Pacific/Port_Moresby"',
     '3,"Mount Hagen","Mount Hagen","Papua New Guinea","HGU","AYMH",-5.826789,144.295861,5388,10,"U","Pacific/Port_Moresby"',
     '4,"Nadzab","Nadzab","Papua New Guinea","LAE","AYNZ",-6.569828,146.726242,239,10,"U","Pacific/Port_Moresby"',
     '5,"Port Moresby Jacksons Intl","Port Moresby","Papua New Guinea","POM","AYPY",-9.443383,147.22005,146,10,"U","Pacific/Port_Moresby"']



Next, we define a function with which we will map the RDD. The biggest difference between this function and the ones we have defined previously is that this tuple returns a tuple instead of a single value. Therefore, this tuple functionally takes the structure of a `(key, value)` pair. In this case, we have the name of the airport and the country in which it is located in. 


```python
def tuple_airport_city(line):
    words = split_comma(line)
    return (words[1], words[3])
```

For demonstration and sanity check purposes, let's apply the function to an example:


```python
tuple_airport_city(test)
```




    ('Goroka', 'Papua New Guinea')



Works as expected! If we map the entire RDD with `tuples_airport_city()`, then we would end up with an RDD containing tuples as each of its elements. In a nutshell, this is what a pair RDD looks like.


```python
pair_airports = airports.map(tuple_airport_city)
pair_airports.take(5)
```




    [('Goroka', 'Papua New Guinea'),
     ('Madang', 'Papua New Guinea'),
     ('Mount Hagen', 'Papua New Guinea'),
     ('Nadzab', 'Papua New Guinea'),
     ('Port Moresby Jacksons Intl', 'Papua New Guinea')]



If we want to use filter, we can simply access keys or values as appropriate using list indexing with brackets. For example, if we want to obtain a list of airports in the United States, we might execute the following `filter()` statement.


```python
pair_us_airports = pair_airports.filter(lambda line: line[1] == 'United States')
pair_us_airports.take(5)
```




    [('Putnam County Airport', 'United States'),
     ('Dowagiac Municipal Airport', 'United States'),
     ('Cambridge Municipal Airport', 'United States'),
     ('Door County Cherryland Airport', 'United States'),
     ('Shoestring Aviation Airfield', 'United States')]



## Map Values

As stated earlier, one of the advantages of using pair RDDs is the ability to perform key or value-specific operations. For example, we might want to apply some map function on the values of the RDD while leaving the keys unchanged. Let;s say we want to change country names to uppercase. This might be achieved as follows:


```python
upper_case_boring = pair_us_airports.map(lambda line: (line[0], line[1].upper()))
upper_case_boring.take(5)
```




    [('Putnam County Airport', 'UNITED STATES'),
     ('Dowagiac Municipal Airport', 'UNITED STATES'),
     ('Cambridge Municipal Airport', 'UNITED STATES'),
     ('Door County Cherryland Airport', 'UNITED STATES'),
     ('Shoestring Aviation Airfield', 'UNITED STATES')]



However, this statement is rather verbose, since it requires us to specify that we want to leave the key unchanged by declaring the tuple as `(line[0], line[1].upper())`. Instead, we can use `mapValues()` to achieve the same result with much less boilerplate. 


```python
upper_case = pair_us_airports.mapValues(lambda value: value.upper())
upper_case.take(5)
```




    [('Goroka', 'PAPUA NEW GUINEA'),
     ('Madang', 'PAPUA NEW GUINEA'),
     ('Mount Hagen', 'PAPUA NEW GUINEA'),
     ('Nadzab', 'PAPUA NEW GUINEA'),
     ('Port Moresby Jacksons Intl', 'PAPUA NEW GUINEA')]



Note that we didn't have to tell Spark what to do with the keys: it already knew that the keys should be left unchanged, and that mapping should only be applied to the values of each pair element in the RDD.

## Reduce by Key

Earlier, we took a look at the `reduce()` operation, which was used to calculate things like sums or products. The equivalent for pair RDDs is `reduceByKey()`. Let's take a look at an example of a simple word frequency counting using a dummy text file.


```python
words = sc.textFile(f'{BASE_PATH}/word_count.text')
words.take(2)
```




    ["The history of New York begins around 10,000 BC, when the first Native Americans arrived. By 1100 AD, New York's main native cultures, the Iroquoian and Algonquian, had developed. European discovery of New York was led by the French in 1524 and the first land claim came in 1609 by the Dutch. As part of New Netherland, the colony was important in the fur trade and eventually became an agricultural resource thanks to the patroon system. In 1626 the Dutch bought the island of Manhattan from Native Americans.[1] In 1664, England renamed the colony New York, after the Duke of York (later James II & VII.) New York City gained prominence in the 18th century as a major trading port in the Thirteen Colonies.",
     '']



To count the occurrences of words, we first need to split the strings into words. Note that we want to use `flatMap()` since we don't want to establish a distinction between different sentences; instead, we want to flatten the output. 


```python
split_words = words.flatMap(lambda line: line.split(' '))
split_words.take(5)
```




    ['The', 'history', 'of', 'New', 'York']



A hacky way to go about this task is to first transform the RDD into a pair RDD where each key is a word and the value is 1. Then, we can add up the values according to each key. Let's accomplish this step-by-step.


```python
raw_pair_words = split_words.map(lambda word: (word, 1))
raw_pair_words.take(5)
```




    [('The', 1), ('history', 1), ('of', 1), ('New', 1), ('York', 1)]



As you might have guessed, this is where `reduceByKey()` comes into play. Applying the simple addition lambda function with `reduceByKey()` produces a pair RDD that contains the total count for each word. 


```python
pair_words = raw_pair_words.reduceByKey(lambda x, y: x + y)
pair_words.take(5)
```




    [('The', 10), ('of', 33), ('New', 20), ('begins', 1), ('around', 4)]



## Sort

One natural extension we might want to go from the word counting example is sorting. One can easily imagine situations where we might want to sort a pair RDD in some ascending or descending order according to value. This can be achieved via the `sortBy()` function. Here, we set `ascending=False` so that the most frequent words would come at the top.


```python
sorted_words = pair_words.sortBy(lambda x: x[1], ascending=False)
sorted_words.take(5)
```




    [('the', 71), ('of', 33), ('in', 21), ('and', 21), ('New', 20)]



## Group by Key

Another common operation with pair RDDs is `groupByKey()`. However, before we get into the details, it should be noted that this method is strongly discouraged for performance reasons, especially on large datasets. For more information, I highly recommend that you take a look at this [notebook](https://databricks.gitbooks.io/databricks-spark-knowledge-base/content/best_practices/prefer_reducebykey_over_groupbykey.html) by Databricks---although it is written in Scala, you will understand most of what is going on based on your knowledge of PySpark. 

With this caveat out of the way, let's take a look at what we can do with `groupByKey()`. We first use the `airport` RDD we've used before in other examples.


```python
airports.take(5)
```




    ['1,"Goroka","Goroka","Papua New Guinea","GKA","AYGA",-6.081689,145.391881,5282,10,"U","Pacific/Port_Moresby"',
     '2,"Madang","Madang","Papua New Guinea","MAG","AYMD",-5.207083,145.7887,20,10,"U","Pacific/Port_Moresby"',
     '3,"Mount Hagen","Mount Hagen","Papua New Guinea","HGU","AYMH",-5.826789,144.295861,5388,10,"U","Pacific/Port_Moresby"',
     '4,"Nadzab","Nadzab","Papua New Guinea","LAE","AYNZ",-6.569828,146.726242,239,10,"U","Pacific/Port_Moresby"',
     '5,"Port Moresby Jacksons Intl","Port Moresby","Papua New Guinea","POM","AYPY",-9.443383,147.22005,146,10,"U","Pacific/Port_Moresby"']



This time, we use a mapping function that returns a key-value pair in the form of `({country}, {airport})`. As you may have guessed, we want to group by country keys to build a new pair RDD.


```python
def tuple_airport_country(line):
    words = split_comma(line)
    return (words[3], words[1])
```

First, we check that the function works as expected, thus producing a pair RDD of country-airport pair elements.


```python
airports_country = airports.map(tuple_airport_country)
airports_country.take(5)
```




    [('Papua New Guinea', 'Goroka'),
     ('Papua New Guinea', 'Madang'),
     ('Papua New Guinea', 'Mount Hagen'),
     ('Papua New Guinea', 'Nadzab'),
     ('Papua New Guinea', 'Port Moresby Jacksons Intl')]



If we apply `groupByKey()` to this RDD, we get a pair RDD whose values are `ResultIterable` objects in PySpark speak. This is somewhat like a list but offered through the Spark interface and arguably less tractable than normal Python lists in that they can't simply be indexed with brackets. 


```python
airports_by_country = airports_country.groupByKey()
airports_by_country.take(5)
```




    [('Iceland', <pyspark.resultiterable.ResultIterable at 0x11af06510>),
     ('Algeria', <pyspark.resultiterable.ResultIterable at 0x11af06410>),
     ('Ghana', <pyspark.resultiterable.ResultIterable at 0x120e0b9d0>),
     ("Cote d'Ivoire", <pyspark.resultiterable.ResultIterable at 0x120e148d0>),
     ('Nigeria', <pyspark.resultiterable.ResultIterable at 0x120fcd6d0>)]



To get a sneak peak into what `ReslutIterable` objects look like, we can convert them into a list. Note that we normally wouldn't enforce list conversion on large datasets.


```python
country, airports = airports_by_country.take(1)[0]
print(country, list(airports))
```

    Iceland ['Akureyri', 'Egilsstadir', 'Hornafjordur', 'Husavik', 'Isafjordur', 'Keflavik International Airport', 'Patreksfjordur', 'Reykjavik', 'Siglufjordur', 'Vestmannaeyjar', 'Reykjahlid Airport', 'Bakki Airport', 'Vopnafjörður Airport', 'Thorshofn Airport', 'Grímsey Airport', 'Bildudalur Airport', 'Gjogur Airport', 'Saudarkrokur', 'Selfoss Airport', 'Reykjahlid', 'Seydisfjordur', 'Nordfjordur Airport']


## Join

Join, which comes from relational algebra, is a very common operation that comes from relational algebra. It is commonly used in SQL to bring two or more tables into the same picture. For a quick visual representation of what joins are, here is an image that might be of help.

<img src='https://upload.wikimedia.org/wikipedia/commons/9/9d/SQL_Joins.svg' />

We can perform joins on pair RDDs as well. We can consider pair RDDs to be somewhat like SQL tables with just a primary key and a single column to go with it. Let's quickly create a toy dataset to illustrate the join operators in PySpark. Here, we have a list of names, ages, and their countries of origin. To best demonstrate the join operation, we intentionally create a mismatch of keys in the `ages_rdd` and `countries_rdd`. 


```python
import random

names = ['Sarah', 'John', 'Tom', 'Clara', 'Ellie', 'Jake', 'Demir']
ages = [random.randint(15, 45) for _ in range(len(names))]
countries = ['USA', 'ROK', 'UK', 'FR', 'PRC', 'CAN', 'BEL']

ages_rdd = sc.parallelize(list(zip(names[:4], ages[:4])))
countries_rdd = sc.parallelize(list(zip(names[2:], countries[2:])))
```

Here is a little helper function to help us take a look at what the keys and values are in a pair RDD.


```python
def show_rdd(rdd):
    for key, value in rdd.collect():
        print(f'Key: {key}, Value: {value}')
```


```python
show_rdd(ages_rdd)
```

    Key: Sarah, Value: 16
    Key: John, Value: 20
    Key: Tom, Value: 16
    Key: Clara, Value: 15



```python
show_rdd(countries_rdd)
```

    Key: Tom, Value: UK
    Key: Clara, Value: FR
    Key: Ellie, Value: PRC
    Key: Jake, Value: CAN
    Key: Demir, Value: BEL


As we can see, the `ages_rdd` and `countries_rdd` each have some overlapping keys, but not all keys are in both RDDs. For instance, Tom and Clara are in both RDDs, but John is only in the `ages_rdd`. This intentional mismatch is going to be useful later when we discuss the difference between left and right joins.

First, let's take a look at `join()`, which in PySpark refers to an inner join. Since this an inner join, we only get results pertaining to keys that are present in both RDDs, namely Clara and Tom.


```python
show_rdd(ages_rdd.join(countries_rdd))
```

    Key: Clara, Value: (15, 'FR')
    Key: Tom, Value: (16, 'UK')


Note that if we flip the order of joins, the order of elements in the values of each key-value pairs also changes.


```python
show_rdd(countries_rdd.join(ages_rdd))
```

    Key: Clara, Value: ('FR', 15)
    Key: Tom, Value: ('UK', 16)


Things get slightly more interesting with other joins like `leftOuterJoin()`. I personally find it intuitive to image two van diagrams, with the left one being completely filled in the case of a left join. In other words, we keep all the keys of the left table while joining with the table on the right. In this case, since there exists a key mismatch, only those keys that are present in both tables will end up with a full joined tuple; others are `None`s in their joined values.


```python
show_rdd(ages_rdd.leftOuterJoin(countries_rdd))
```

    Key: John, Value: (20, None)
    Key: Clara, Value: (15, 'FR')
    Key: Tom, Value: (16, 'UK')
    Key: Sarah, Value: (16, None)


The same happens with `rightOuterJoin()`. This is exactly identical to what the left join is, except with the very obvious caveat that the right diagram is filled, not the left.


```python
show_rdd(ages_rdd.rightOuterJoin(countries_rdd))
```

    Key: Ellie, Value: (None, 'PRC')
    Key: Clara, Value: (15, 'FR')
    Key: Tom, Value: (16, 'UK')
    Key: Jake, Value: (None, 'CAN')
    Key: Demir, Value: (None, 'BEL')


Lastly, let's take a look at `fullOuterJoin()`. This is a combination of the left and right join---we fill up the entire van diagram, both left and right. Notice that this is exactly what happens when we add the results from `leftOuterJoin()` and `rightOuterJoin()`, with duplicates removed. 


```python
show_rdd(ages_rdd.fullOuterJoin(countries_rdd))
```

    Key: John, Value: (20, None)
    Key: Ellie, Value: (None, 'PRC')
    Key: Clara, Value: (15, 'FR')
    Key: Tom, Value: (16, 'UK')
    Key: Jake, Value: (None, 'CAN')
    Key: Sarah, Value: (16, None)
    Key: Demir, Value: (None, 'BEL')


# Conclusion

In this post, we explored the various aspects of Apache Spark: what it is, how to set it up, and what we can do with it via the RDD API. There is a lot more to Spark that we haven't discussed, such as Spark SQL or MLLib. I will most definitely be writing a post on these as I become more familiar with the various APIs and functionalities that Spark has to offer. 

I doubt I'll be using Spark for any personal project, since Spark is used for processing large datasets across different clusters, not on a single computer as we have done here. However, it was an interesting journey and one that was definitely worth the time and effort, since I feel like I've at least gained some glimpse of what all the hype behind the Spark keyword is. 

I hope you've enjoyed reading this post. See you in the next one!
