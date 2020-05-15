---
title: A Short R Tutorial
toc: true
categories:
  - study
tags:
  - jupyter
  - r
---

This is an experimental jupyter notebook written using `IRkernel`. The purpose of this notebook is threefolds: first, to document my progress with self-learning the R language; second, to test the functionality of the R kernel on jupyter; and third, to see if the`convert.sh` shell script is capable of converting notebooks written in R to `.md` markdown format. The example codes in this notebook were borrowed from [Hands on Programming with R](https://rstudio-education.github.io/hopr/) by Garrett Grolemund.

# Basic Operations and Assigment


```R
1 + 1
```


2



```R
die <- 1:6
```


```R
ls()
```


'die'



```R
dice <- sample(die, 2)
dice
```



<ol>
 <li>1</li>
 <li>6</li>
</ol>





# Creating Functions


```R
roll <- function(){
    die <- 1:6
    dice <- sample(die, size = 2, replace = TRUE)
    sum(dice)
}
```


```R
roll()
```


6



```R
roll()
```


5



```R
roll()
```


6


# Creating Plots


```R
library("ggplot2")
options(repr.plot.width=12, repr.plot.height=9)
```


```R
x <- c(-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1)
x
```


<ol>
	<li>-1</li>
	<li>-0.8</li>
	<li>-0.6</li>
	<li>-0.4</li>
	<li>-0.2</li>
	<li>0</li>
	<li>0.2</li>
	<li>0.4</li>
	<li>0.6</li>
	<li>0.8</li>
	<li>1</li>
</ol>




```R
y <- x^3
y
```


<ol>
	<li>-1</li>
	<li>-0.512</li>
	<li>-0.216</li>
	<li>-0.064</li>
	<li>-0.008</li>
	<li>0</li>
	<li>0.008</li>
	<li>0.064</li>
	<li>0.216</li>
	<li>0.512</li>
	<li>1</li>
</ol>




```R
qplot(x, y)
```

<img src="/assets/images/2020-01-09-R-tutorial_files/2020-01-09-R-tutorial_15_0.png">


```R
x2 <- c(1, 2, 2, 2, 3, 3)
qplot(x2, binwidth = 1)
```

<img src="/assets/images/2020-01-09-R-tutorial_files/2020-01-09-R-tutorial_16_0.png">



```R
replicate(10, roll())
```


<ol>
	<li>4</li>
	<li>10</li>
	<li>9</li>
	<li>7</li>
	<li>5</li>
	<li>7</li>
	<li>9</li>
	<li>9</li>
	<li>7</li>
	<li>8</li>
</ol>




```R
rolls <- replicate(10000, roll())
qplot(rolls, binwidth = 1)
```


<img src="/assets/images/2020-01-09-R-tutorial_files/2020-01-09-R-tutorial_18_0.png">


# Object Types


```R
typeof(die)
```


'integer'



```R
sum(die)
```


21



```R
is.vector(die)
```


TRUE



```R
sqrt(2)^2 - 2
```


4.44089209850063e-16



```R
3 > 4
```


FALSE



```R
typeof(F)
```


'logical'



```R
comp <- c(1 + 1i, 1 + 2i, 1 + 3i)
typeof(comp)
```


'complex'



```R
attributes(die)
```


    NULL


# Matrix and Data Frames


```R
names(die) <- c("one", "two", "three", "four", "five", "six")
attributes(die)
```


<strong>$names</strong>
<ol>
	<li>'one'</li>
	<li>'two'</li>
	<li>'three'</li>
	<li>'four'</li>
	<li>'five'</li>
	<li>'six'</li>
</ol>




```R
die
```


<dl>
	<dt>one</dt>
		<dd>1</dd>
	<dt>two</dt>
		<dd>2</dd>
	<dt>three</dt>
		<dd>3</dd>
	<dt>four</dt>
		<dd>4</dd>
	<dt>five</dt>
		<dd>5</dd>
	<dt>six</dt>
		<dd>6</dd>
</dl>




```R
die + 1
```


<dl>
	<dt>one</dt>
		<dd>2</dd>
	<dt>two</dt>
		<dd>3</dd>
	<dt>three</dt>
		<dd>4</dd>
	<dt>four</dt>
		<dd>5</dd>
	<dt>five</dt>
		<dd>6</dd>
	<dt>six</dt>
		<dd>7</dd>
</dl>




```R
dim(die) <- c(2, 3)
die
```


<table>
<tbody>
	<tr><td>1</td><td>3</td><td>5</td></tr>
	<tr><td>2</td><td>4</td><td>6</td></tr>
</tbody>
</table>




```R
m <- matrix(die, nrow = 2, byrow = TRUE)
m
```


<table>
<tbody>
	<tr><td>1</td><td>2</td><td>3</td></tr>
	<tr><td>4</td><td>5</td><td>6</td></tr>
</tbody>
</table>




```R
hand1 <- c("ace", "king", "queen", "jack", "ten", "spades", "spades", 
  "spades", "spades", "spades")
dim(hand1) <- c(5, 2)
hand1
```


<table>
<tbody>
	<tr><td>ace   </td><td>spades</td></tr>
	<tr><td>king  </td><td>spades</td></tr>
	<tr><td>queen </td><td>spades</td></tr>
	<tr><td>jack  </td><td>spades</td></tr>
	<tr><td>ten   </td><td>spades</td></tr>
</tbody>
</table>




```R
class(die)
```


'matrix'



```R
now <- Sys.time()
now
```


    [1] "2020-01-07 06:14:50 KST"



```R
gender <- factor(c("male", "female", "female", "male"))
gender
```


<ol>
	<li>male</li>
	<li>female</li>
	<li>female</li>
	<li>male</li>
</ol>

<details>
	<summary style=display:list-item;cursor:pointer>
		<strong>Levels</strong>:
	</summary>
	<ol>
		<li>'female'</li>
		<li>'male'</li>
	</ol>
</details>



```R
typeof(gender)
```


'integer'



```R
sum(c(TRUE, TRUE, FALSE, FALSE))
```


2



```R
card <- list("ace", "hearts", 1)
card
```


<ol>
	<li>'ace'</li>
	<li>'hearts'</li>
	<li>1</li>
</ol>




```R
df <- data.frame(face = c("ace", "two", "six"),  
  suit = c("clubs", "clubs", "clubs"), value = c(1, 2, 3))
df
```


<table>
<thead>
	<tr><th>face</th><th>suit</th><th>value</th></tr>
</thead>
<tbody>
	<tr><td>ace  </td><td>clubs</td><td>1    </td></tr>
	<tr><td>two  </td><td>clubs</td><td>2    </td></tr>
	<tr><td>six  </td><td>clubs</td><td>3    </td></tr>
</tbody>
</table>




```R
deck <- data.frame(
  face = c("king", "queen", "jack", "ten", "nine", "eight", "seven", "six",
    "five", "four", "three", "two", "ace", "king", "queen", "jack", "ten", 
    "nine", "eight", "seven", "six", "five", "four", "three", "two", "ace", 
    "king", "queen", "jack", "ten", "nine", "eight", "seven", "six", "five", 
    "four", "three", "two", "ace", "king", "queen", "jack", "ten", "nine", 
    "eight", "seven", "six", "five", "four", "three", "two", "ace"),  
  suit = c("spades", "spades", "spades", "spades", "spades", "spades", 
    "spades", "spades", "spades", "spades", "spades", "spades", "spades", 
    "clubs", "clubs", "clubs", "clubs", "clubs", "clubs", "clubs", "clubs", 
    "clubs", "clubs", "clubs", "clubs", "clubs", "diamonds", "diamonds", 
    "diamonds", "diamonds", "diamonds", "diamonds", "diamonds", "diamonds", 
    "diamonds", "diamonds", "diamonds", "diamonds", "diamonds", "hearts", 
    "hearts", "hearts", "hearts", "hearts", "hearts", "hearts", "hearts", 
    "hearts", "hearts", "hearts", "hearts", "hearts"), 
  value = c(13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 13, 12, 11, 10, 9, 8, 
    7, 6, 5, 4, 3, 2, 1, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 13, 12, 11, 
    10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
)
```


```R
head(deck, 7)
```


<table>
<thead><tr><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><td>king  </td><td>spades</td><td>13    </td></tr>
	<tr><td>queen </td><td>spades</td><td>12    </td></tr>
	<tr><td>jack  </td><td>spades</td><td>11    </td></tr>
	<tr><td>ten   </td><td>spades</td><td>10    </td></tr>
	<tr><td>nine  </td><td>spades</td><td> 9    </td></tr>
	<tr><td>eight </td><td>spades</td><td> 8    </td></tr>
	<tr><td>seven </td><td>spades</td><td> 7    </td></tr>
</tbody>
</table>




```R
deck[1, c(1, 2, 3)]
```


<table>
<thead><tr><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><td>king  </td><td>spades</td><td>13    </td></tr>
</tbody>
</table>




```R
deck[-(2:52), 1:3]
```


<table>
<thead><tr><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><td>king  </td><td>spades</td><td>13    </td></tr>
</tbody>
</table>




```R
deck[1, ]
```


<table>
<thead><tr><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><td>king  </td><td>spades</td><td>13    </td></tr>
</tbody>
</table>




```R
deck[1, c(T, T, F)]
```


<table>
<thead><tr><th>face</th><th>suit</th></tr></thead>
<tbody>
	<tr><td>king  </td><td>spades</td></tr>
</tbody>
</table>




```R
deck[1, c("face", "suit")]
```


<table>
<thead><tr><th>face</th><th>suit</th></tr></thead>
<tbody>
	<tr><td>king  </td><td>spades</td></tr>
</tbody>
</table>




```R
deal <- function(cards) {
    cards[1, ]
}
```


```R
deal(deck)
```


<table>
<thead><tr><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><td>king  </td><td>spades</td><td>13    </td></tr>
</tbody>
</table>




```R
shuffle <- function(cards){
    random <- sample(1:52, size = 52)
    cards[random, ]
}
```


```R
head(shuffle(deck))
```


<table>
<thead><tr><th></th><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><th>6</th><td>eight   </td><td>spades  </td><td> 8      </td></tr>
	<tr><th>5</th><td>nine    </td><td>spades  </td><td> 9      </td></tr>
	<tr><th>39</th><td>ace     </td><td>diamonds</td><td> 1      </td></tr>
	<tr><th>3</th><td>jack    </td><td>spades  </td><td>11      </td></tr>
	<tr><th>34</th><td>six     </td><td>diamonds</td><td> 6      </td></tr>
	<tr><th>38</th><td>two     </td><td>diamonds</td><td> 2      </td></tr>
</tbody>
</table>




```R
better_deal <- function(cards){
    card <- deal(shuffle(cards))
    card
}
```


```R
better_deal(deck)
```


<table>
<thead><tr><th></th><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><th>38</th><td>two     </td><td>diamonds</td><td>2       </td></tr>
</tbody>
</table>




```R
better_deal(deck)
```


<table>
<thead><tr><th></th><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><th>9</th><td>five  </td><td>spades</td><td>5     </td></tr>
</tbody>
</table>



# Dollar Sign Selection


```R
deck$value
```


<ol>
	<li>13</li>
	<li>12</li>
	<li>11</li>
	<li>10</li>
	<li>9</li>
	<li>8</li>
	<li>7</li>
	<li>6</li>
	<li>5</li>
	<li>4</li>
	<li>3</li>
	<li>2</li>
	<li>1</li>
	<li>13</li>
	<li>12</li>
	<li>11</li>
	<li>10</li>
	<li>9</li>
	<li>8</li>
	<li>7</li>
	<li>6</li>
	<li>5</li>
	<li>4</li>
	<li>3</li>
	<li>2</li>
	<li>1</li>
	<li>13</li>
	<li>12</li>
	<li>11</li>
	<li>10</li>
	<li>9</li>
	<li>8</li>
	<li>7</li>
	<li>6</li>
	<li>5</li>
	<li>4</li>
	<li>3</li>
	<li>2</li>
	<li>1</li>
	<li>13</li>
	<li>12</li>
	<li>11</li>
	<li>10</li>
	<li>9</li>
	<li>8</li>
	<li>7</li>
	<li>6</li>
	<li>5</li>
	<li>4</li>
	<li>3</li>
	<li>2</li>
	<li>1</li>
</ol>




```R
mean(deck$value)
```


7



```R
deck2 <- deck
```


```R
vec <- c(0, 0, 0, 0, 0, 0)
vec[1] <- 1000
vec
```


<ol>
	<li>1000</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
</ol>




```R
vec[c(1, 3, 5)] <- c(1, 1, 1)
vec
```


<ol>
	<li>1</li>
	<li>0</li>
	<li>1</li>
	<li>0</li>
	<li>1</li>
	<li>0</li>
</ol>




```R
vec[8] <- 0
vec
```


<ol>
	<li>1</li>
	<li>0</li>
	<li>1</li>
	<li>0</li>
	<li>1</li>
	<li>0</li>
	<li>&lt;NA&gt;</li>
	<li>0</li>
</ol>




```R
deck2$new <- 1:52
head(deck2)
```


<table>
<thead><tr><th>face</th><th>suit</th><th>value</th><th>new</th></tr></thead>
<tbody>
	<tr><td>king  </td><td>spades</td><td>13    </td><td>1     </td></tr>
	<tr><td>queen </td><td>spades</td><td>12    </td><td>2     </td></tr>
	<tr><td>jack  </td><td>spades</td><td>11    </td><td>3     </td></tr>
	<tr><td>ten   </td><td>spades</td><td>10    </td><td>4     </td></tr>
	<tr><td>nine  </td><td>spades</td><td> 9    </td><td>5     </td></tr>
	<tr><td>eight </td><td>spades</td><td> 8    </td><td>6     </td></tr>
</tbody>
</table>




```R
deck2$new <- NULL
head(deck2)
```


<table>
<thead><tr><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><td>king  </td><td>spades</td><td>13    </td></tr>
	<tr><td>queen </td><td>spades</td><td>12    </td></tr>
	<tr><td>jack  </td><td>spades</td><td>11    </td></tr>
	<tr><td>ten   </td><td>spades</td><td>10    </td></tr>
	<tr><td>nine  </td><td>spades</td><td> 9    </td></tr>
	<tr><td>eight </td><td>spades</td><td> 8    </td></tr>
</tbody>
</table>




```R
deck2$value[c(13, 26, 39, 52)]
```


<ol>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
</ol>




```R
deck2$value[c(13, 26, 39, 52)] <- 14
head(deck2, 13)
```


<table>
<thead><tr><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><td>king  </td><td>spades</td><td>13    </td></tr>
	<tr><td>queen </td><td>spades</td><td>12    </td></tr>
	<tr><td>jack  </td><td>spades</td><td>11    </td></tr>
	<tr><td>ten   </td><td>spades</td><td>10    </td></tr>
	<tr><td>nine  </td><td>spades</td><td> 9    </td></tr>
	<tr><td>eight </td><td>spades</td><td> 8    </td></tr>
	<tr><td>seven </td><td>spades</td><td> 7    </td></tr>
	<tr><td>six   </td><td>spades</td><td> 6    </td></tr>
	<tr><td>five  </td><td>spades</td><td> 5    </td></tr>
	<tr><td>four  </td><td>spades</td><td> 4    </td></tr>
	<tr><td>three </td><td>spades</td><td> 3    </td></tr>
	<tr><td>two   </td><td>spades</td><td> 2    </td></tr>
	<tr><td>ace   </td><td>spades</td><td>14    </td></tr>
</tbody>
</table>




```R
1 > c(0, 1, 2)
```


<ol>
	<li>TRUE</li>
	<li>FALSE</li>
	<li>FALSE</li>
</ol>




```R
typeof(1 > c(0, 1, 2))
```


'logical'



```R
c(1, 2, 3) == c(3, 2, 1)
```


<ol>
	<li>FALSE</li>
	<li>TRUE</li>
	<li>FALSE</li>
</ol>




```R
c(1, 2, 3) %in% c(3, 4, 5)
```


<ol>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>TRUE</li>
</ol>




```R
sum(deck2$face == "ace")
```


4



```R
deck3 <- shuffle(deck)
deck3$value[deck3$face == "ace"] <- 14
head(deck3)
```


<table>
<thead><tr><th></th><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><th>33</th><td>seven   </td><td>diamonds</td><td> 7      </td></tr>
	<tr><th>32</th><td>eight   </td><td>diamonds</td><td> 8      </td></tr>
	<tr><th>42</th><td>jack    </td><td>hearts  </td><td>11      </td></tr>
	<tr><th>26</th><td>ace     </td><td>clubs   </td><td>14      </td></tr>
	<tr><th>48</th><td>five    </td><td>hearts  </td><td> 5      </td></tr>
	<tr><th>36</th><td>four    </td><td>diamonds</td><td> 4      </td></tr>
</tbody>
</table>




```R
deck4 <- deck
deck4$value <- 0
deck4$value[deck4$suit == "hearts"] <- 1

queenOfSpades <- deck4$face == "queen" & deck4$suit == "spades"
deck4$value[queenOfSpades] <- 13
deck4[queenOfSpades, ]
```


<table>
<thead><tr><th></th><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><th>2</th><td>queen </td><td>spades</td><td>13    </td></tr>
</tbody>
</table>




```R
deck5 <- deck
facecard <- deck5$face %in% c("king", "queen", "jack")
deck5$value[facecard] <- 10
head(deck5, 13)
```


<table>
<thead><tr><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><td>king  </td><td>spades</td><td>10    </td></tr>
	<tr><td>queen </td><td>spades</td><td>10    </td></tr>
	<tr><td>jack  </td><td>spades</td><td>10    </td></tr>
	<tr><td>ten   </td><td>spades</td><td>10    </td></tr>
	<tr><td>nine  </td><td>spades</td><td> 9    </td></tr>
	<tr><td>eight </td><td>spades</td><td> 8    </td></tr>
	<tr><td>seven </td><td>spades</td><td> 7    </td></tr>
	<tr><td>six   </td><td>spades</td><td> 6    </td></tr>
	<tr><td>five  </td><td>spades</td><td> 5    </td></tr>
	<tr><td>four  </td><td>spades</td><td> 4    </td></tr>
	<tr><td>three </td><td>spades</td><td> 3    </td></tr>
	<tr><td>two   </td><td>spades</td><td> 2    </td></tr>
	<tr><td>ace   </td><td>spades</td><td> 1    </td></tr>
</tbody>
</table>



# N/A Representation


```R
1 + NA
```


&lt;NA&gt;



```R
mean(c(NA, 1:50))
```


&lt;NA&gt;



```R
mean(c(NA, 1:50), na.rm = TRUE)
```


25.5



```R
vec <- c(1, 2, 3, NA)
is.na(vec)
```


<ol>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>TRUE</li>
</ol>




```R
deck5$value[deck5$face == "ace"] <- NA
head(deck5, 13)
```


<table>
<thead><tr><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><td>king  </td><td>spades</td><td>10    </td></tr>
	<tr><td>queen </td><td>spades</td><td>10    </td></tr>
	<tr><td>jack  </td><td>spades</td><td>10    </td></tr>
	<tr><td>ten   </td><td>spades</td><td>10    </td></tr>
	<tr><td>nine  </td><td>spades</td><td> 9    </td></tr>
	<tr><td>eight </td><td>spades</td><td> 8    </td></tr>
	<tr><td>seven </td><td>spades</td><td> 7    </td></tr>
	<tr><td>six   </td><td>spades</td><td> 6    </td></tr>
	<tr><td>five  </td><td>spades</td><td> 5    </td></tr>
	<tr><td>four  </td><td>spades</td><td> 4    </td></tr>
	<tr><td>three </td><td>spades</td><td> 3    </td></tr>
	<tr><td>two   </td><td>spades</td><td> 2    </td></tr>
	<tr><td>ace   </td><td>spades</td><td>NA    </td></tr>
</tbody>
</table>



# Scope and Environments


```R
library(pryr)
parenvs(all = TRUE)
```

    Registered S3 method overwritten by 'pryr':
      method      from
      print.bytes Rcpp




       label                            name               
    1  <environment: R_GlobalEnv>       ""                 
    2  <environment: package:pryr>      "package:pryr"     
    3  <environment: package:ggplot2>   "package:ggplot2"  
    4  <environment: 0x7fcfdbe6cf88>    "jupyter:irkernel" 
    5  <environment: package:stats>     "package:stats"    
    6  <environment: package:graphics>  "package:graphics" 
    7  <environment: package:grDevices> "package:grDevices"
    8  <environment: package:utils>     "package:utils"    
    9  <environment: package:datasets>  "package:datasets" 
    10 <environment: package:methods>   "package:methods"  
    11 <environment: 0x7fcfd9a813d0>    "Autoloads"        
    12 <environment: base>              ""                 
    13 <environment: R_EmptyEnv>        ""                 



```R
as.environment("package:stats")
```


    <environment: package:stats>
    attr(,"name")
    [1] "package:stats"
    attr(,"path")
    [1] "/Users/jaketae/opt/anaconda3/envs/R/lib/R/library/stats"



```R
parent.env(globalenv())
```


    <environment: package:pryr>
    attr(,"name")
    [1] "package:pryr"
    attr(,"path")
    [1] "/Users/jaketae/opt/anaconda3/envs/R/lib/R/library/pryr"



```R
head(globalenv()$deck, 3)
```


<table>
<thead><tr><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><td>king  </td><td>spades</td><td>13    </td></tr>
	<tr><td>queen </td><td>spades</td><td>12    </td></tr>
	<tr><td>jack  </td><td>spades</td><td>11    </td></tr>
</tbody>
</table>




```R
assign("new", "Hello Global", envir = globalenv())
globalenv()$new
```


'Hello Global'



```R
environment()
```


    <environment: R_GlobalEnv>



```R
DECK <- deck
```


```R
deal <- function() {
  card <- deck[1, ]
  assign("deck", deck[-1, ], envir = globalenv())
  card
}
```


```R
shuffle <- function(){
  random <- sample(1:52, size = 52)
  assign("deck", DECK[random, ], envir = globalenv())
}
```


```R
shuffle()
deal()
```


<table>
<thead><tr><th></th><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><th>23</th><td>four </td><td>clubs</td><td>4    </td></tr>
</tbody>
</table>




```R
deal()
```


<table>
<thead><tr><th></th><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><th>17</th><td>ten  </td><td>clubs</td><td>10   </td></tr>
</tbody>
</table>



# Closure


```R
setup <- function(deck) {
  DECK <- deck

  DEAL <- function() {
    card <- deck[1, ]
    assign("deck", deck[-1, ], envir = parent.env(environment()))
    card
  }

  SHUFFLE <- function(){
    random <- sample(1:52, size = 52)
    assign("deck", DECK[random, ], envir = parent.env(environment()))
 }

 list(deal = DEAL, shuffle = SHUFFLE)
}
```


```R
cards <- setup(deck)
deal <- cards$deal
shuffle <- cards$shuffle
```


```R
shuffle()
deal()
```


<table>
<thead><tr><th></th><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><th>15</th><td>queen</td><td>clubs</td><td>12   </td></tr>
</tbody>
</table>




```R
deal()
```


<table>
<thead><tr><th></th><th>face</th><th>suit</th><th>value</th></tr></thead>
<tbody>
	<tr><th>37</th><td>three   </td><td>diamonds</td><td>3       </td></tr>
</tbody>
</table>




```R
get_symbols <- function() {
  wheel <- c("DD", "7", "BBB", "BB", "B", "C", "0")
  sample(wheel, size = 3, replace = TRUE, 
    prob = c(0.03, 0.03, 0.06, 0.1, 0.25, 0.01, 0.52))
}
get_symbols()
```


<ol>
	<li>'BBB'</li>
	<li>'B'</li>
	<li>'BBB'</li>
</ol>




```R
get_symbols()
```


<ol>
	<li>'0'</li>
	<li>'0'</li>
	<li>'0'</li>
</ol>



# If-Else Statements


```R
positive_negative <- function(num){
    if (num < 0){
        result <- "Negative"
    }
    else{
        result <- "Positive"
    }
    result
}
```


```R
positive_negative(10)
```


'Positive'



```R
positive_negative(-10)
```


'Negative'



```R
round_decimal <- function(num){
    decimal <- num - trunc(num)
    result <- 0
    if (decimal >= 0.5){
        result <- trunc(num) + 1
    }
    else{
        result <- trunc(num)
    }
    result
}
```


```R
round_decimal(3.14)
```


3



```R
round_decimal(1.9)
```


2



```R
score <- function (symbols) {
  # identify case
  same <- symbols[1] == symbols[2] && symbols[2] == symbols[3]
  bars <- symbols %in% c("B", "BB", "BBB")
  
  # get prize
  if (same) {
    payouts <- c("DD" = 100, "7" = 80, "BBB" = 40, "BB" = 25, 
      "B" = 10, "C" = 10, "0" = 0)
    prize <- unname(payouts[symbols[1]])
  } else if (all(bars)) {
    prize <- 5
  } else {
    cherries <- sum(symbols == "C")
    prize <- c(0, 2, 5)[cherries + 1]
  }
  
  # adjust for diamonds
  diamonds <- sum(symbols == "DD")
  prize * 2 ^ diamonds
}
```


```R
play <- function() {
    symbols <- get_symbols()
    print(symbols)
    score(symbols)
}
```


```R
play()
```

    [1] "0" "B" "B"



0



```R
play()
```

    [1] "B"   "BBB" "B"  



5


# Class and Attributes


```R
attributes(deck)
```


<dl>
	<dt>$names</dt>
		<dd><ol>
	<li>'face'</li>
	<li>'suit'</li>
	<li>'value'</li>
</ol>
</dd>
	<dt>$row.names</dt>
		<dd><ol>
	<li>47</li>
	<li>42</li>
	<li>9</li>
	<li>12</li>
	<li>5</li>
	<li>32</li>
	<li>26</li>
	<li>49</li>
	<li>27</li>
	<li>31</li>
	<li>18</li>
	<li>52</li>
	<li>19</li>
	<li>1</li>
	<li>46</li>
	<li>25</li>
	<li>13</li>
	<li>50</li>
	<li>45</li>
	<li>24</li>
	<li>3</li>
	<li>15</li>
	<li>44</li>
	<li>28</li>
	<li>8</li>
	<li>30</li>
	<li>36</li>
	<li>38</li>
	<li>21</li>
	<li>48</li>
	<li>11</li>
	<li>6</li>
	<li>14</li>
	<li>41</li>
	<li>34</li>
	<li>39</li>
	<li>35</li>
	<li>16</li>
	<li>22</li>
	<li>51</li>
	<li>20</li>
	<li>2</li>
	<li>33</li>
	<li>40</li>
	<li>10</li>
	<li>4</li>
	<li>37</li>
	<li>43</li>
	<li>7</li>
	<li>29</li>
</ol>
</dd>
	<dt>$class</dt>
		<dd>'data.frame'</dd>
</dl>




```R
levels(deck) <- c("level 1", "level 2", "level 3")
attributes(deck)
```


<dl>
	<dt>$names</dt>
		<dd><ol>
	<li>'face'</li>
	<li>'suit'</li>
	<li>'value'</li>
</ol>
</dd>
	<dt>$row.names</dt>
		<dd><ol>
	<li>47</li>
	<li>42</li>
	<li>9</li>
	<li>12</li>
	<li>5</li>
	<li>32</li>
	<li>26</li>
	<li>49</li>
	<li>27</li>
	<li>31</li>
	<li>18</li>
	<li>52</li>
	<li>19</li>
	<li>1</li>
	<li>46</li>
	<li>25</li>
	<li>13</li>
	<li>50</li>
	<li>45</li>
	<li>24</li>
	<li>3</li>
	<li>15</li>
	<li>44</li>
	<li>28</li>
	<li>8</li>
	<li>30</li>
	<li>36</li>
	<li>38</li>
	<li>21</li>
	<li>48</li>
	<li>11</li>
	<li>6</li>
	<li>14</li>
	<li>41</li>
	<li>34</li>
	<li>39</li>
	<li>35</li>
	<li>16</li>
	<li>22</li>
	<li>51</li>
	<li>20</li>
	<li>2</li>
	<li>33</li>
	<li>40</li>
	<li>10</li>
	<li>4</li>
	<li>37</li>
	<li>43</li>
	<li>7</li>
	<li>29</li>
</ol>
</dd>
	<dt>$class</dt>
		<dd>'data.frame'</dd>
	<dt>$levels</dt>
		<dd><ol>
	<li>'level 1'</li>
	<li>'level 2'</li>
	<li>'level 3'</li>
</ol>
</dd>
</dl>




```R
one_play <- play()
attr(one_play, "symbols") <- c("B", "0", "B")
one_play
```

    [1] "B" "0" "0"



0



```R
play <- function() {
    symbols <- get_symbols()
    prize <- score(symbols)
    attr(prize, "symbols") <- symbols
    prize
    #structure(score(symbols), symbols = symbols)
}
```


```R
two_play <- play()
two_play
```


0



```R
slot_display <- function(prize){

  # extract symbols
  symbols <- attr(prize, "symbols")

  # collapse symbols into single string
  symbols <- paste(symbols, collapse = " ")

  # combine symbol with prize as a character string
  # \n is special escape sequence for a new line (i.e. return or enter)
  string <- paste(symbols, prize, sep = "\n$")

  # display character string in console without quotes
  cat(string)
}
```


```R
slot_display(two_play)
```

    B BB 0
    $0


```R
print.slots <- function(x, ...) {
  slot_display(x)
}
```


```R
play <- function() {
  symbols <- get_symbols()
  structure(score(symbols), symbols = symbols, class = "slots")
}
```


```R
play()
```


    BB B 0
    $0


# Expand Grid


```R
rolls <- expand.grid(die, die)
head(rolls)
```


<table>
<thead><tr><th>Var1</th><th>Var2</th></tr></thead>
<tbody>
	<tr><td>1</td><td>1</td></tr>
	<tr><td>2</td><td>1</td></tr>
	<tr><td>3</td><td>1</td></tr>
	<tr><td>4</td><td>1</td></tr>
	<tr><td>5</td><td>1</td></tr>
	<tr><td>6</td><td>1</td></tr>
</tbody>
</table>




```R
rolls$value <- rolls$Var1 + rolls$Var2
head(rolls)
```


<table>
<thead><tr><th>Var1</th><th>Var2</th><th>value</th></tr></thead>
<tbody>
	<tr><td>1</td><td>1</td><td>2</td></tr>
	<tr><td>2</td><td>1</td><td>3</td></tr>
	<tr><td>3</td><td>1</td><td>4</td></tr>
	<tr><td>4</td><td>1</td><td>5</td></tr>
	<tr><td>5</td><td>1</td><td>6</td></tr>
	<tr><td>6</td><td>1</td><td>7</td></tr>
</tbody>
</table>




```R
prob <- c(1/8, 1/8, 1/8, 1/8, 1/8, 3/8)
rolls$prob = prob[rolls$Var1] * prob[rolls$Var2]
head(rolls)
```


<table>
<thead><tr><th>Var1</th><th>Var2</th><th>value</th><th>prob</th></tr></thead>
<tbody>
	<tr><td>1       </td><td>1       </td><td>2       </td><td>0.015625</td></tr>
	<tr><td>2       </td><td>1       </td><td>3       </td><td>0.015625</td></tr>
	<tr><td>3       </td><td>1       </td><td>4       </td><td>0.015625</td></tr>
	<tr><td>4       </td><td>1       </td><td>5       </td><td>0.015625</td></tr>
	<tr><td>5       </td><td>1       </td><td>6       </td><td>0.015625</td></tr>
	<tr><td>6       </td><td>1       </td><td>7       </td><td>0.046875</td></tr>
</tbody>
</table>




```R
expected_val <- sum(rolls$value * rolls$prob)
expected_val
```


8.25



```R
wheel <- c("DD", "7", "BBB", "BB", "B", "C", "0")
combos <- expand.grid(wheel, wheel, wheel, stringsAsFactors = FALSE)
head(combos)
```


<table>
<thead><tr><th>Var1</th><th>Var2</th><th>Var3</th></tr></thead>
<tbody>
	<tr><td>DD </td><td>DD </td><td>DD </td></tr>
	<tr><td>7  </td><td>DD </td><td>DD </td></tr>
	<tr><td>BBB</td><td>DD </td><td>DD </td></tr>
	<tr><td>BB </td><td>DD </td><td>DD </td></tr>
	<tr><td>B  </td><td>DD </td><td>DD </td></tr>
	<tr><td>C  </td><td>DD </td><td>DD </td></tr>
</tbody>
</table>




```R
prob <- c("DD" = 0.03, "7" = 0.03, "BBB" = 0.06, "BB" = 0.1, "B" = 0.25, "C" = 0.01, "0" = 0.52)
combos$prob <- prob[combos$Var1] * prob[combos$Var2] * prob[combos$Var3]
head(combos)
```


<table>
<thead><tr><th>Var1</th><th>Var2</th><th>Var3</th><th>prob</th></tr></thead>
<tbody>
	<tr><td>DD      </td><td>DD      </td><td>DD      </td><td>0.000027</td></tr>
	<tr><td>7       </td><td>DD      </td><td>DD      </td><td>0.000027</td></tr>
	<tr><td>BBB     </td><td>DD      </td><td>DD      </td><td>0.000054</td></tr>
	<tr><td>BB      </td><td>DD      </td><td>DD      </td><td>0.000090</td></tr>
	<tr><td>B       </td><td>DD      </td><td>DD      </td><td>0.000225</td></tr>
	<tr><td>C       </td><td>DD      </td><td>DD      </td><td>0.000009</td></tr>
</tbody>
</table>




```R
sum(combos$prob)
```


1


# For and While Loops


```R
for (value in c("My", "first", "for", "loop")) {
  print(value)
}
```

    [1] "My"
    [1] "first"
    [1] "for"
    [1] "loop"



```R
value
```


'loop'



```R
chars <- vector(length = 4)
words <- c("My", "fourth", "for", "loop")

for (i in 1:4) {
    chars[i] <- words[i]
}

chars
```


<ol>
	<li>'My'</li>
	<li>'fourth'</li>
	<li>'for'</li>
	<li>'loop'</li>
</ol>




```R
length(chars)
```


4



```R
combos$prize <- NA

for (i in 1:length(combos$prob)) {
    symbols <- c(combos[i, 1], combos[i, 2], combos[i, 3])
    combos$prize[i] <- combos$prob[i] * score(symbols)
}

head(combos)
```


<table>
<thead><tr><th>Var1</th><th>Var2</th><th>Var3</th><th>prob</th><th>prize</th></tr></thead>
<tbody>
	<tr><td>DD      </td><td>DD      </td><td>DD      </td><td>0.000027</td><td>0.021600</td></tr>
	<tr><td>7       </td><td>DD      </td><td>DD      </td><td>0.000027</td><td>0.000000</td></tr>
	<tr><td>BBB     </td><td>DD      </td><td>DD      </td><td>0.000054</td><td>0.000000</td></tr>
	<tr><td>BB      </td><td>DD      </td><td>DD      </td><td>0.000090</td><td>0.000000</td></tr>
	<tr><td>B       </td><td>DD      </td><td>DD      </td><td>0.000225</td><td>0.000000</td></tr>
	<tr><td>C       </td><td>DD      </td><td>DD      </td><td>0.000009</td><td>0.000072</td></tr>
</tbody>
</table>




```R
sum(combos$prize)
```


0.538014



```R
score <- function(symbols) {
  
  diamonds <- sum(symbols == "DD")
  cherries <- sum(symbols == "C")
  
  # identify case
  # since diamonds are wild, only nondiamonds 
  # matter for three of a kind and all bars
  slots <- symbols[symbols != "DD"]
  same <- length(unique(slots)) == 1
  bars <- slots %in% c("B", "BB", "BBB")

  # assign prize
  if (diamonds == 3) {
    prize <- 100
  } else if (same) {
    payouts <- c("7" = 80, "BBB" = 40, "BB" = 25,
      "B" = 10, "C" = 10, "0" = 0)
    prize <- unname(payouts[slots[1]])
  } else if (all(bars)) {
    prize <- 5
  } else if (cherries > 0) {
    # diamonds count as cherries
    # so long as there is one real cherry
    prize <- c(0, 2, 5)[cherries + diamonds + 1]
  } else {
    prize <- 0
  }
  
  # double for each diamond
  prize * 2^diamonds
}
```


```R
for (i in 1:length(combos$prob)) {
    symbols <- c(combos[i, 1], combos[i, 2], combos[i, 3])
    combos$prize[i] <- combos$prob[i] * score(symbols)
}

sum(combos$prize)
```


0.934356



```R
play_till_broke <- function(start_with) {
    n <- 0
    cash <- start_with
    while (cash > 0) {
        cash <- cash - 1 + play()
        n <- n + 1
    }
    n
}

play_till_broke(100)
```


373


# Function Vectorization


```R
# Unvectorized
abs_loop <- function(vec){
  for (i in 1:length(vec)) {
    if (vec[i] < 0) {
      vec[i] <- -vec[i]
    }
  }
  vec
}
```


```R
# Vectorized
abs_vec <- function(vec){
    index <- vec < 0
    vec[index] <- -1 * vec[index]
    vec
}
```


```R
long <- rep(c(-1, 1), 5000000)
```


```R
system.time(abs_loop(long))
```


       user  system elapsed 
      0.599   0.017   0.620 



```R
system.time(abs_vec(long))
```


       user  system elapsed 
      0.269   0.034   0.304 



```R
change_vec <- function (vec) {
  vec[vec == "DD"] <- "joker"
  vec[vec == "C"] <- "ace"
  vec[vec == "7"] <- "king"
  vec[vec == "B"] <- "queen"
  vec[vec == "BB"] <- "jack"
  vec[vec == "BBB"] <- "ten"
  vec[vec == "0"] <- "nine"
  vec
}
```


```R
# Lookup Tables
change_vec2 <- function(vec){
  tb <- c("DD" = "joker", "C" = "ace", "7" = "king", "B" = "queen", 
    "BB" = "jack", "BBB" = "ten", "0" = "nine")
  unname(tb[vec])
}
```


```R
vec <- c("DD", "C", "7", "B", "BB", "BBB", "0")
many <- rep(vec, 1000000)
```


```R
system.time(change_vec(many))
```


       user  system elapsed 
      0.518   0.035   0.555 



```R
system.time(change_vec2(many))
```


       user  system elapsed 
      0.181   0.032   0.213 



```R
get_many_symbols <- function(n) {
  wheel <- c("DD", "7", "BBB", "BB", "B", "C", "0")
  vec <- sample(wheel, size = 3 * n, replace = TRUE,
    prob = c(0.03, 0.03, 0.06, 0.1, 0.25, 0.01, 0.52))
  matrix(vec, ncol = 3)
}
```


```R
get_many_symbols(3)
```


<table>
<tbody>
	<tr><td>0 </td><td>0 </td><td>0 </td></tr>
	<tr><td>B </td><td>0 </td><td>7 </td></tr>
	<tr><td>B </td><td>0 </td><td>DD</td></tr>
</tbody>
</table>




```R
score_many <- function(symbols) {

  # Step 1: Assign base prize based on cherries and diamonds ---------
  ## Count the number of cherries and diamonds in each combination
  cherries <- rowSums(symbols == "C")
  diamonds <- rowSums(symbols == "DD") 
  
  ## Wild diamonds count as cherries
  prize <- c(0, 2, 5)[cherries + diamonds + 1]
  
  ## ...but not if there are zero real cherries 
  ### (cherries is coerced to FALSE where cherries == 0)
  prize[!cherries] <- 0
  
  # Step 2: Change prize for combinations that contain three of a kind 
  same <- symbols[, 1] == symbols[, 2] & 
    symbols[, 2] == symbols[, 3]
  payoffs <- c("DD" = 100, "7" = 80, "BBB" = 40, 
    "BB" = 25, "B" = 10, "C" = 10, "0" = 0)
  prize[same] <- payoffs[symbols[same, 1]]
  
  # Step 3: Change prize for combinations that contain all bars ------
  bars <- symbols == "B" | symbols ==  "BB" | symbols == "BBB"
  all_bars <- bars[, 1] & bars[, 2] & bars[, 3] & !same
  prize[all_bars] <- 5
  
  # Step 4: Handle wilds ---------------------------------------------
  
  ## combos with two diamonds
  two_wilds <- diamonds == 2

  ### Identify the nonwild symbol
  one <- two_wilds & symbols[, 1] != symbols[, 2] & 
    symbols[, 2] == symbols[, 3]
  two <- two_wilds & symbols[, 1] != symbols[, 2] & 
    symbols[, 1] == symbols[, 3]
  three <- two_wilds & symbols[, 1] == symbols[, 2] & 
    symbols[, 2] != symbols[, 3]
  
  ### Treat as three of a kind
  prize[one] <- payoffs[symbols[one, 1]]
  prize[two] <- payoffs[symbols[two, 2]]
  prize[three] <- payoffs[symbols[three, 3]]
  
  ## combos with one wild
  one_wild <- diamonds == 1
  
  ### Treat as all bars (if appropriate)
  wild_bars <- one_wild & (rowSums(bars) == 2)
  prize[wild_bars] <- 5
  
  ### Treat as three of a kind (if appropriate)
  one <- one_wild & symbols[, 1] == symbols[, 2]
  two <- one_wild & symbols[, 2] == symbols[, 3]
  three <- one_wild & symbols[, 3] == symbols[, 1]
  prize[one] <- payoffs[symbols[one, 1]]
  prize[two] <- payoffs[symbols[two, 2]]
  prize[three] <- payoffs[symbols[three, 3]]
 
  # Step 5: Double prize for every diamond in combo ------------------
  unname(prize * 2^diamonds)
  
}
```


```R
play_many <- function(n) {
  symb_mat <- get_many_symbols(n = n)
  data.frame(w1 = symb_mat[,1], w2 = symb_mat[,2],
             w3 = symb_mat[,3], prize = score_many(symb_mat))
}
```


```R
play_many(3)
```


<table>
<thead><tr><th>w1</th><th>w2</th><th>w3</th><th>prize</th></tr></thead>
<tbody>
	<tr><td>7  </td><td>B  </td><td>0  </td><td> 0 </td></tr>
	<tr><td>B  </td><td>B  </td><td>BBB</td><td> 5 </td></tr>
	<tr><td>DD </td><td>BB </td><td>B  </td><td>10 </td></tr>
</tbody>
</table>




```R
system.time(play_many(1000))
```


       user  system elapsed 
      0.005   0.000   0.005 
