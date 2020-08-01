---
title: Text Preprocessing with Blog Post Data
mathjax: false
toc: true
categories:
  - study
tags:
  - nlp
---

In today's post, we will finally start modeling the auto-tagger model that I wanted to build for more blog. As you may have noticed, every blog post is classified into a few different tags, which are essentially topic keywords that I have been manually assigning thus far. I have been getting increasingly lazier these past few weeks, which is ironically what compelled me into experimenting and studying more about the basics of NLP. As mentioned in previous posts, this is why I've been posting things like tf-idf vectorization and word embeddings. 

While there are so many SOTA models out there, for the purposes of this mini-project, I decided to go slow. In what may or may not become a little series of its own, I aspire to achieve the following:

* Design a basic parsing algorithm to clean blog posts in markdown
* Vectorize the cleaned string data
* Build a target vector corresponding to each blog post
* Construct and train a document classifier
* Develop a pipeline to generate and display the model's predictions

This is by no means a short, easy project for me. This is also the first time that I'm dealing with real data---data created by no one other than myself---so there is an interesting element of meta-ness to it that I enjoy. After all, this very post that I'm writing will also be a valuable addition to the training set for the model. 

With that said, let's jump right into it.

# Parsing Text Data

The first challenge is cleaning the textual data we have. By cleaning, I don't mean vectorizing; instead, we need to first retrieve the data, get rid of extraneous characters, code blocks, and MathJax expressions, and so on. After all, our simple model cannot be expected to understand code blocks or LaTeX expressions, as awesome as that sounds. 

There were two routes I could take with parsing. The first was web scraping; the second, using directly parsing raw markdown files. I'll detail each attempts I've made in the following sections, then explain why I chose one approach over the other.

## First Approach: Web Scraping

Because all my published posts are available on my blog website, I could crawl the blog and extract `<p>` tags to construct my training dataset. Here are some of the steps I took while experimenting with this approach. For demonstration purposes, I decided to use a recent post on Gaussian process regression, as it contains a nice blend of both code and MathJax expressions. 


```python
import bs4
import requests

url = "https://jaketae.github.io/study/gaussian-process/"
html = requests.get(url).text
soup = bs4.BeautifulSoup(html, "html.parser")
p_tags = soup.find_all("p", class_="")
```

Now we can take a look at the first `p` tag in the web scraped list. 


```python
p_tags[0]
```




    <p>In this post, we will explore the Gaussian Process in the context of regression. This is a topic I meant to study for a long time, yet was never able to due to the seemingly intimidating mathematics involved. However, after consulting some extremely well-curated resources on this topic, such as <a href="https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote15.html">Kilian’s lecture notes</a> and <a href="https://www.youtube.com/watch?v=MfHKW5z-OOA&amp;list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6&amp;index=9">UBC lecture videos by Nando de Freitas</a>, I think I’m finally starting to understand what GP is. I highly recommend that you check out these resources, as they are both very beginner friendly and build up each concept from the basics. With that out of the way, let’s get started.</p>



This is not a bad starting point, but obviously there is so much more work that has to be done. For one thing, we need to remove `<code>` tags that are often wrapped around `<p>` tags. We also have to remove inline LaTex expressions, which as written as `$ ... $`. Below is a function that that I wrote to clean the data with the following considerations in mind.


```python
def get_words(url):
    html = requests.get(url).text
    soup = bs4.BeautifulSoup(html, "html.parser")
    ps = soup.find_all("p", class_="")
    result = []
    for tag in ps:
        tmp = ""
        flag = True
        for code in tag.find_all("code"):
            code.extract()
        for char in tag.text:
            if char == "$":
                flag = not flag
                continue
            if flag:
                tmp += char
        result.append(tmp)
    return " ".join(result)
```

For demonstration purposes, let's try scraping the post on GP regression I mentioned earlier.


```python
text = get_words(url)
text[1000:2000]
```




    'ution of the predicted data at a given test point. Gaussian Processes (GPs) are similar to Bayesian linear regression in that the final result is a distribution which we can sample from. The biggest point of difference between GP and Bayesian regression, however, is that GP is a fundamentally non-parametric approach, whereas the latter is a parametric one. I think this is the most fascinating part about GPs—as we will see later on, GPs do not require us to specify any function or model to fit the data. Instead, all we need to do is to identify the mean and covariance of a multivariate Gaussian that defines the posterior of the GP. All of this sounds too good be true—how can a single multivariate Gaussian distribution be enough for what could potentially be a high-dimensional, complicated regression problem? Let’s discuss some mathematical ideas that enable GP to be so powerful. Gaussians are essentially a black hole of distributions: once a Gaussian, always a Gaussian. For example, we '



We see that the text has indeed been parsed, which is great! So we have the basic tools to parse a post given a URL. So naturally, the next step would be to figure out all the URLs for the blog posts I have on my website. Of course, I could do this manually, but that sort of defeats the point of building an auto-tagger. 

So after some trial and error, here is another function I wrote that scrapes all blog post URLs on my website.


```python
def get_urls():
    root = "https://jaketae.github.io/posts/"
    root_html = requests.get(root).text
    soup = bs4.BeautifulSoup(root_html, "html.parser")
    divs = soup.find_all("div", class_="list__item")
    return [f"https://jaketae.github.io{tag.find('a')['href']}" for tag in divs]
```

We start from the root URL, then basically extract hrefs from the `<div>` elements that each represent a single blog post. 


```python
urls = get_urls()
urls
```




    ['https://jaketae.github.io/development/tinkering-docker/',
     'https://jaketae.github.io/study/word2vec/',
     'https://jaketae.github.io/study/complex-fibonacci/',
     'https://jaketae.github.io/study/tf-idf/',
     'https://jaketae.github.io/study/gaussian-process/',
     'https://jaketae.github.io/study/genetic-algorithm/',
     'https://jaketae.github.io/study/revisiting-basel/',
     'https://jaketae.github.io/study/zeta-prime/',
     'https://jaketae.github.io/study/bfs-dfs/',
     'https://jaketae.github.io/study/numerical-methods/',
     'https://jaketae.github.io/study/gibbs-sampling/',
     'https://jaketae.github.io/development/sklearn-sprint/',
     'https://jaketae.github.io/study/spark-basics/',
     'https://jaketae.github.io/study/dissecting-lstm/',
     'https://jaketae.github.io/study/sklearn-pipeline/',
     'https://jaketae.github.io/study/natural-gradient/',
     'https://jaketae.github.io/blog/workflow-cleanup/',
     'https://jaketae.github.io/study/r-tutorial-4/',
     'https://jaketae.github.io/study/sql-basics/',
     'https://jaketae.github.io/study/r-tutorial-3/',
     'https://jaketae.github.io/development/c/',
     'https://jaketae.github.io/study/leibniz-rule/',
     'https://jaketae.github.io/study/r-tutorial-2/',
     'https://jaketae.github.io/study/r-tutorial-1/',
     'https://jaketae.github.io/study/fisher/',
     'https://jaketae.github.io/study/stieltjes/',
     'https://jaketae.github.io/study/stirling/',
     'https://jaketae.github.io/study/pca/',
     'https://jaketae.github.io/study/fourier/',
     'https://jaketae.github.io/study/gan-math/',
     'https://jaketae.github.io/study/kl-mle/',
     'https://jaketae.github.io/study/development/open-source/',
     'https://jaketae.github.io/study/development/flask/',
     'https://jaketae.github.io/study/gan/',
     'https://jaketae.github.io/study/vae/',
     'https://jaketae.github.io/study/autoencoder/',
     'https://jaketae.github.io/study/auto-complete/',
     'https://jaketae.github.io/study/rnn/',
     'https://jaketae.github.io/study/neural-net/',
     'https://jaketae.github.io/study/cnn/',
     'https://jaketae.github.io/blog/typora/',
     'https://jaketae.github.io/study/map-convex/',
     'https://jaketae.github.io/study/exponential-family/',
     'https://jaketae.github.io/study/bayesian-regression/',
     'https://jaketae.github.io/study/naive-bayes/',
     'https://jaketae.github.io/study/first-keras/',
     'https://jaketae.github.io/study/R-tutorial/',
     'https://jaketae.github.io/development/anaconda/',
     'https://jaketae.github.io/study/MCMC/',
     'https://jaketae.github.io/study/logistic-regression/',
     'https://jaketae.github.io/study/map-mle/',
     'https://jaketae.github.io/study/KNN/',
     'https://jaketae.github.io/study/information-entropy/',
     'https://jaketae.github.io/study/moment/',
     'https://jaketae.github.io/study/gaussian-distribution/',
     'https://jaketae.github.io/study/svd/',
     'https://jaketae.github.io/study/linear-regression/',
     'https://jaketae.github.io/study/monte-carlo/',
     'https://jaketae.github.io/study/likelihood/',
     'https://jaketae.github.io/blog/jupyter-automation/',
     'https://jaketae.github.io/study/bayes/',
     'https://jaketae.github.io/blog/test/',
     'https://jaketae.github.io/study/basel-zeta/',
     'https://jaketae.github.io/study/gamma/',
     'https://jaketae.github.io/study/poisson/',
     'https://jaketae.github.io/study/eulers-identity/',
     'https://jaketae.github.io/study/markov-chain/',
     'https://jaketae.github.io/tech/new-mbp/',
     'https://jaketae.github.io/study/pagerank-and-markov/',
     'https://jaketae.github.io/blog/studying-deep-learning/']



I knew that I had been writing somewhat consistently for the past few months, but looking at this full list made me realize how fast time has flown by. 

Continuing with our discussion on cleaning data, now we have all the basic tools we need to build our training data. In fact, we can simply build our raw strings training data simply by looping over all the URLs and extracting text from each:


```python
X_train = [get_words(url) for url in urls]
```

## Second Approach: Markdown Parsing

At this point, you might be wondering why I even attempted a second approach, given that these methods all work fine. 

The answer is that, although web scraping works okay---and we could certainly continue with this approach---but we would have to build a text parser anyway. Think about it: although we can build the training data through web scraping, to run the actual inference, we need to parse the draft, in markdown format, that has not been published yet. In other words, we have no choice but to deal with markdown files, since we have to parse and clean our draft to feed into the model. 

It is after this belated realization that I started building a parser. Now, the interesting part is that I tried two different approachdes going down this road as well. So really, the accurate description would be that I tried three different methods. 

### Brute Replace

The first sub-approach was the one I first thought about, and is thus naturally the more naive method of the two. This is simply an adaptation of the algorithm used in the `get_words()` function, involving a `flag` boolean variable that would switch on and off as we loop through the words, switching whenever we see a delimiter like `"```"` or `"$$"`. Shown below are part of the code I wrote while probing down this route.


```python
def remove_blocks(text_list, delimiter):
    res = []
    flag = True
    for text in text_list:
        if delimiter in text:
            flag = not flag
            continue
        flag and res.append(text)
    return res


def remove_inline(text_list, delimiter):
    res = []
    for i, text in enumerate(text_list):
        (i % 2 == 0) and res.append(text)
    return res
```

This approach works, but only to a certain extent. There are so many edge cases to consider. In particular, the method seemed to break the most when it saw raw HTML image tags. All in all, my impression was that this was an imperfect, fragile implementation, and that there would be a much better way to go about this for sure. 

If you take a look at part of the code I was working on, it isn't difficult to see why this approach turned out to be a nightmare.


```python
for delimiter in ("$$", "```"):
        text_list = remove_blocks(text_list, delimiter)
for delimiter in ("$", "`"):
    text_list = " ".join(text_list).split(delimiter)
    text_list = remove_inline(text_list, delimiter)
res = [
    text.replace('"', "")
    for text in "".join(text_list).split(" ")
    if not ("http" in text or "src" in text)
]
return " ".join(res)
```

And that's why I eventually ended up using regular expressions. 

### Regular Expressions

Although I'm a self-proclaimed Pythonista, I must confess that regular expressions, and the `re` module in particular, were not my friends. I always favored the simple `str.replace()` method and used it whenver I could. Well, turns out that when you're dealing with patterns like "remove square brackets from this string," or "remove everything in between two dollar signs," there is nothing better than regular expressions. 

So after some research and trial and error, I was able to arrive at the following function:


```python
def clean(text):
    regexps = (
        r"```.*?```",
        r"\$\$.*?\$\$",
        r"`.*?`",
        r"\$.*?\$",
        r"\(.*?\)",
        r"[\[\]]",
        r"<.*?\>",
    )
    for regexp in regexps:
        text = re.sub(regexp, "", text)
    return re.sub(r"[\W]", " ", text)
```

There are a total of eight regular expression replacement operations we have to make. The first, which is `"```.*?```"`, is probably the easiest out of all. All this is saying is that we want to get rid of code block expresions, which start with three ticks. The dot represents any single letter character; `*` means that we want to match any number of these single letter characters. Lastly, the `?` makes sure that the `*` matching is not greedy. This isn't particularly a difficult concept to grasp, but for a more in-depth explanation, I strongly recommend that you go check out [Python documentation](https://docs.python.org/3/howto/regex.html). 

The other seven expressions are basically variations of this pattern in one form or another, where we specify some preset pattern, such as dollar signs for equations, and so on. All of this is to show that regexp is a powerful tool, a lot more so than building a custom parser with some complex logic to navigate and process given text. 

Now all there is left is to read in the contents of the file. The only caveat to this is the fact that each line ends with a `\n`, and that `---\n` is a delilmiter used to specify the header of the file in YAML format. This function performs the basic preprocessing we need to successfully read in the contents of the post.


```python
def read(file_path):
    with open(file_path) as file:
        text_list = []
        count = 0
        for line in file:
            if line == "---\n" and count < 2:
                count += 1
                continue
            line = line.rstrip("\n")
            if line and line[:4].strip() and count > 1:
                text_list.append(line.lower())
    return " ".join(text_list)
```

Last but not least, we can combine the two functions to produce a basic parsing function that reads as input the file location and outputs a fully preprocessed text.


```python
def parse(file_path):
    text = read(file_path)
    return " ".join(clean(text).split())
```

Here is an example output I was able to obtain by running the script on the directory where all my blog posts are located, namely `_posts`. The example here was taken from my earlier post on the Fibonacci sequence and Binet's formula.

```
[...] binet s formula gives us what we might refer to as the interpolation of the fibonacci sequence in this case extended along the real number line plotting the fibonacci sequence a corollary of the real number interpolation of the fibonacci sequence via binet s formula is that now we can effectively plot the complex fibonacci numbers on the cartesian plane because can be continuous we would expect some graph to appear where the axis represents real numbers and the imaginary this requires a bit of a hack though note that the result of binet s formula is a complex number or a two dimensional data point the input to the function is just a one dimensional real number therefore we need a way of representing a map from a one dimensional real number line to a two dimensional complex plane [...]
```

As you can see, we have successfully removed all MathJax and punctuations. The end product looks perfect for a bag-of-words model, or a model that is syntax or sequence agonistic. For the purposes of this post, our model will not be paying attention to the order in which the words appear in the text. Although one might rightly point out that this is a simplistic approach, nonetheless it works well for simple tasks such as keyword extraction or document classification; after all, if you think about it, words that appear on a Football magazine will probably be very different from those that appear on culinary magazines, without considering the order in which the words show up into account. 

The bottom line of this long section of text preprocessing---and finding the optimal avenue of retrieving data---is that, after some trial and error, I decided to use markdown parsing instead of web scraping to prepare the training data, as it made the most sense and provided reliable results. 

But this is only the first of many steps to come. In particular, we have to address the question: given text data, how to we vectorize it into some numerical form that our model can digest? This is where various NLP techniques, such as tokenization and stemming come into play.

# Tokenizing Text Data

Tokenization refers to a process through which words are reduced and encoded into some representation of preference. In a braod sense, this is at the heart of data preparation and preprocessing. 

There are many tools out there that we can use for tokenization. I decided to use `nltk`, as it provides an easy way to deal with tokenization as well as stop word removal. Stop words simply refer to words such as "a" or "the"---those that appear quite often without carrying much meaning (yes, some linguists might say otherwise but from a general layman's perspective, they carry less substantive mass).

First, let's import the library to get started.


```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Unzipping corpora/wordnet.zip.





    True



## Stemming, Lemmatization

There are two popular ways of normalizing text: stemming and lemmatization. Broadly speaking, both stemming and lemmatization are ways of simplifying and cutting down words into their root form. We can go into a lot of detail into what these are, but for the purposes of this post, it suffices to see what they processes they are with a set of simple examples. 


```python
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer(language="english")
```

First, let's take a look at lemmatization. 


```python
lemmatizer.lemmatize("resulting")
```




    'resulting'



And here is the same result, but using a stemmer.


```python
stemmer.stem("resulting")
```




    'result'



It seems as if lemmatization did nothing to change the word, whereas stemming did apply some transformation to distill the word into its root form. Well, this is because we didn't pass into the lemmatizer any POS tag, or a tag indicating whether a word is a noun, verb, or anything else. If we specify the tag, the lemmatizer works as expected.


```python
lemmatizer.lemmatize("resulting", 'v')
```




    'result'



You might be thinking that lemmatization is obviously the more inferior of the two since it requires additional information. However, this is not quite true; in fact, lemmatization is a more advanced technique that can potentially generate better results. However, it also requires some more work due to the additional input field requirement. For the purposes of this post, therefore, we resort to the cruder of the two, stemming. In our case, stemming works quite well since the data returned by parsing `.md` files is a clean string, with all forms of punctuation removed. 

Another important piece to our preprocessing procedure is stop words removal. We don't want preprocessed vector outputs to be influenced by words that carry little indicative power---that is, words that have low correlation to the target values, or post tags.


```python
stop_words = set(stopwords.words('english')) 
```

Besides the obvious, there are number of words in this set. Here are some exmaples:


```python
for word in ("i", "the", "ve"):
    assert word in stop_words
```

I was personally surprised to find out that "ve" was a stop word, although it makes a lot of sense in hindsight. Put differently, the `stop_words` set is more comprehensive that I had assumed.


## Tokenization

So how do we use stemming and stop words to tokenize text? Well, the simple answer is the short code snippet below:


```python
def stem_tokenizer(text):
    return [
        stemmer.stem(word)
        for word in word_tokenize(text.lower())
        if word not in stop_words and word.isalpha()
    ]
```

The `stem_tokenizer` function receives as input a raw string and outputs a list of tokenized words. Here is a sample demonstration using a short text snippet on the Fibonacci sequence I used in another example above.


```python
text = "binet s formula gives us what we might refer to as the interpolation of the fibonacci sequence"

stem_tokenizer(text)
```




    ['binet',
     'formula',
     'give',
     'us',
     'might',
     'refer',
     'interpol',
     'fibonacci',
     'sequenc']



As you can see, we get a clean list of tokens. This is the form of data we want after this basic preprocessing step. Once we have cleaned data, we can then convert them into numerical form, which is where tf-idf vectors come into play.

# Next Steps

There is obviously a lot more to be done from here. Specifically, we will first need to convert tokenized data into tf-idf vectors. We explored tf-idf vectors in [this previous post](https://jaketae.github.io/study/tf-idf/), where we implemented the vectorization algorithm from scratch. tf-idf vectorization is a smart, efficient way of combining word count with inverse document frequency to encode text data into vector form, which can then be passed into some ML or DL models as input. 

Next, we will probably need to use some form of dimensionality reduction method to reduce sparsity. Alternatively, we could also simply filter out words that are too common or too rare; for instance, chop off the top and bottom 10 percent of words in the frequency spectrum. These steps are necessary because otherwise the tf-idf vectors will be extremely sparse and high-dimensional; I already tried a simple test drive of tf-idf vectorization with this data some time ago, albeit with coarser preprocessing and tokenization methods. Nonetheless, I ended up with 3000-dimensional vectors. For our purposes, there are probably advantages to dimensionality reduction. 

Then comes the exciting part: building our model. This is arguably the part which allows for the most degree of freedom---in the literal sense, not the statistical one, of course. As it stands, my plan is to build a simple fully connected model as a baseline and see its performance first, and experiment with other structures there. The reason why a dense neural network makes sense is that we are using a bag-of-words model---the order of words do not matter, as they all end up being jumbled up during tokenization and tf-idf vectorization. 

Ideally, the final step would be training the model and somehow integrating it with the current blog workflow such that, whenever I type and convert an `.ipynb` notebook to `.md` files, ready for upload, the model's predictions already appear in the converted `.md` file. This shouldn't be too difficult, but nonetheless it is an important detail that would be very cool to implement once we have a finalized model. It would be even better if we could implement some mechanism to train our model with new data per each blog post upload; after all, we don't want to use an old model trained with old data. Instead, we want to feed it with new data so that it is able to learn more. 

Hopefully we'll find a way to tackle these considerations as we move forward. Catch you up in the next post!
