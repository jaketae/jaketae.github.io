---
title: Keyword Extraction with BERT
mathjax: true
toc: true
categories:
  - study
tags:
  - deep_learning
  - nlp
---

I've been interested in blog post auto-tagging and classification for some time. Recently, I was able to fine-tune RoBERTa to develop a decent multi-label, multi-class classification model to assign labels to my draft blog posts. This automated a small yet nonetheless substantial part of my blog post writeup workflow. Now, running

```
format 2021-02-10-some-article.ipynb
```

automatically generates a markdown file that not only includes all the contents of the Jupyter notebook, but it also includes automatically generated tags that the fine-tuned BERT model inferred based on text processing. 

Nonetheless, I knew that more could be done. The BERT fine-tuning approach came with a number of different drawbacks. For instance, the model was only trained on a total of the eight most frequently occuring labels. This was in large part due to my naïve design of the model and the unavoidable limitations of multi-label classification: the more labels there are, the worse the model performs. The fact that the dataset had been manually labeled by me, who tagged articles back then without much thought, certainly did not help.

The supervised leanring approach I took with fine-tuning also meant that the model could not learn to classify new labels it had not seen before. After all, the classification head of the model was fixed, so unless a new classifier was trained from scratch using new data, the model would never learn to predict new labels. Retraining and fine-tuning the model again would be a costly, resource-intensive operation. 

While there might be many ways to go about this problem, I've come to two realistic, engineerable solutions: zero-shot classification and keyword extraction as a means of new label suggestion. In today's post, I hope to explore the latter in more detail by introducing an easy way of extracting keywords from a block of text using transformers and contextual embeddings. The method introduced in this post heavily borrows the methodology introduced in [this Medium article](https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea) by Maarten Grootendorst, author of the [KeyBERT](https://github.com/MaartenGr/KeyBERT). I highly recommend that you check out both his post as well as the library on GitHub.

Without futher ado, let's jump right in!

# Introduction

Before we get down into the engineering details, here's a bird's eye view of what we want to achieve. Given a block of text, we want to have a function or model that is able to extract important keywords. We might specify as a parameter how many keywords we want to extract from the given text. Finally, once we have those keywords, the idea is that each of these keywords could potentially be used as a tag for a blog post. This way, we can overcome the shortcomings of the supervised learning approach with BERT fine-tuning discussed earlier. 

I'm writing this tutorial on Google Colab, so let's go ahead and install the packages that Colab does not ship with my default: spaCy and HuggingFace transformers.


```
%%capture
!pip install spacy transformers
```

At its simplest form, I imagine the API of this keyword extractor to look something like this:

```python
extractor = Extractor()
extractor.generate(text, num_keywords=6)
```

Of course, we could have gone with a much simpler setup and aim for something like

```python
keywords = extract(text, num_keywords=5)
```

However, as we will see later, each extraction requires a transformer and spaCy model, so maybe it might be better to offer a reusable extractor object, so that the user can pass in another block of text for some other keyword extraction task without having to download different models all the time.

For this demo, we will be using the following block of text, taken from the Wikipedia page on supervised learning.


```python
text = """
         Supervised learning is the machine learning task of 
         learning a function that maps an input to an output based 
         on example input-output pairs.[1] It infers a function 
         from labeled training data consisting of a set of 
         training examples.[2] In supervised learning, each 
         example is a pair consisting of an input object 
         (typically a vector) and a desired output value (also 
         called the supervisory signal). A supervised learning 
         algorithm analyzes the training data and produces an 
         inferred function, which can be used for mapping new 
         examples. An optimal scenario will allow for the algorithm 
         to correctly determine the class labels for unseen 
         instances. This requires the learning algorithm to  
         generalize from the training data to unseen situations 
         in a 'reasonable' way (see inductive bias).
      """
```

Hopefully, we can build a simple keyword extraction pipeline that is able to identify and return salient keywords from the original text.

Note that this is not a generative method; in other words, the keyword extractor will never be able to return words that are not present in the provided text. Generating new words that somehow nicely summarize the provided passage requires a generative, potentially auto-regressive model, with tested and proven NLU and NLG capabilities. For the purposes of this demonstration, we take the simpler extractive approach.

# Candidate Selection

The first step to keyword extraction is producing a set of plausible keyword candidates. As stated earlier, those candidates come from the provided text itself. The important question, then, is how we can select keywords from the body of text. 

This is where n-grams come in. Recall that n-grams are simply consecutive words of text. For example, a 2-gram or bi-gram span all sets of two consecutive word pairs. 

Normally, keywords are either single words or two words. Rarely do we see long keywords: after all, long, complicated keywords are self-defeating since the very purpose of a keyword is to be impressionable, short, and concise. Using scikit-learn's count vectorizer, we can specify the n-gram range parameter, then obtain the entire list of n-grams that fall within the specified range. 


```python
from sklearn.feature_extraction.text import CountVectorizer

n_gram_range = (1, 2)
stop_words = "english"

# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text])
all_candidates = count.get_feature_names()
```

Let's look at the first then candidate n-grams that have been extracted. Notice that they are all either one or two words, which is what we want.


```python
all_candidates[:10]
```




    ['algorithm',
     'algorithm analyzes',
     'algorithm correctly',
     'algorithm generalize',
     'allow',
     'allow algorithm',
     'analyzes',
     'analyzes training',
     'based',
     'based example']



One glaring problem with the list of all candidates above is that there are some verbs or verb phrases that we do not want included in the list. Most often or not, keywords are nouns or noun phrases. To remove degenerate candidates such as "analyzes," we need to some basic part-of-speech or POS tagging. Then, we can safely extract only candidates that are nouns or noun phrases. 

To achieve this, we can using spaCy, a powerful NLP library with POS-tagging features. Below, we extract noun phrases from the chunk of text.


```python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
noun_phrases = set(chunk.text.strip().lower() for chunk in doc.noun_chunks)
```

Extracting only single noun tokens is also fairly easy. For an in-depth look at what can be achieved with spaCy, I recommend that you take a look at the [spaCy documentation](https://spacy.io/usage/linguistic-features).


```python
nouns = set()
for token in doc:
    if token.pos_ == "NOUN":
        nouns.add(token.text)
```

We simply check all the tokens in the document parsed by the spaCy model, then check the part-of-speech tag to only add nouns to hte `nouns` set. Then, we can combine this result with the set of noun phrases we obtained earlier to create a set of all nouns and noun phrases. 


```python
all_nouns = nouns.union(noun_phrases)
```

Great! The last step that is remaining in the candidate selection process, then, is filter the earlier list of all candidates and including only those that are in the all nouns set we obtained through spaCy. This can be achieved as a one-liner using the filter function (yes, I know, this is not the most Pythonic way to do it, but it is a useful trick nonetheless).


```python
candidates = list(filter(lambda candidate: candidate in all_nouns, all_candidates))
```

Let's take a look at the final list of candidates. Notice that they are all either nouns or noun phrases, which is what we wanted. Some of them already seem like they could be good keywords.


```python
candidates[:10]
```




    ['algorithm',
     'allow',
     'analyzes',
     'based',
     'bias',
     'called',
     'class',
     'consisting',
     'correctly',
     'data']



# Keyword Generation

We are now half way through. We have a sensible number of words to work with that could be keywords, so all is left is finding the best keyword out of the bunch. 

Let's think a little bit more about what a good keyword really is. Obviously, we aren't going to come up with some academically rigorous definition of what a keyword is. Nonetheless, I think many would agree that a good keyword is one that which accurately captures the semantics of the main text. This could also be seen as an extreme case of text summarization, in which only a single word or short n-grams can be used. 

The intuition behind embedding-based keyword extraction is the following: if we can embed both the text and keyword candidates into the same latent embeeding space, best keywords are most likely ones whose embeddings live in close proximity to the text embedding itself. In other words, keyword extraction simply amounts to calculating some distance metric between the text embedding and candidate keyword embeddings, and finding the top $k$ candidates that are closest to the full text. 

## Embedding

While there are many ways of creating embeddings, given the recent advances in NLP with transformer-based models and contextual embeddings, it makes the most amount of sense to use a transformer autoencoder, such as BERT. To achieve this, let's first import the HuggingFace transformers library.


```python
from transformers import AutoModel, AutoTokenizer
```

Here, we use a knowledge-distilled version of RoBERTa. But really, any BERT-based model, or even simply autoencoding, embedding-generating transformer model should do the job.


```python
model_name = "distilroberta-base"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```


Now that we have the model, it's time to create embeddings. Creating embeddings is extremely simple: all we need to do is to tokenize the candidate keywords, then pass them through the model itself. BERT-based models typically output a pooler output, which is a 768-dimensional vector for each input text. 


```python
candidate_tokens = tokenizer(candidates, padding=True, return_tensors="pt")
candidate_embeddings = model(**candidate_tokens)["pooler_output"]
```

We can see that the 53 keyword candidates have successfully been mapped to a 768-dimensional latent space.


```python
candidate_embeddings.shape
```




    torch.Size([53, 768])



Now, it's time to embed the block of text itself to the same dimension. From an implementation point of view, this looks no diffeent from the code snippet above.


```python
text_tokens = tokenizer([text], padding=True, return_tensors="pt")
text_embedding = model(**text_tokens)["pooler_output"]
```

Since the whole chunk of text was processed at once, we should only see one vector, and indeed that seems to be the case. Notice the fact that the entire text was mapped to the same latent space to which the candidate keywords were also projected.


```python
text_embedding.shape
```




    torch.Size([1, 768])



# Distance Measurement

What's left is calculating the distance between the main text embedding and the candidate keyword embeddings. For the distance metric, we will be using cosine similarity, as it is a simple yet robust way of measuring distances between vectors in high dimensional space. 

Let's first detach the embeddings from the computational graph and convert them into NumPy arrays.


```python
candidate_embeddings = candidate_embeddings.detach().numpy()
text_embedding = text_embedding.detach().numpy()
```

Next, we obtain the cosine similarity between the text embedding and candidate embeddings, perform an argsort operation to obtain the indices of the keywords that are closest to the text embedding, and slice the top $k$ keywords from the candidates list. 


```python
from sklearn.metrics.pairwise import cosine_similarity

top_k = 5
distances = cosine_similarity(text_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_k:]]
```

Voila! We have the 5 most relevant keywords from the original block of text. Given that the Wikipedia article from which the text was adapted was on the topic of supervised learning, these keywords do seem appropriate.


```python
keywords
```




    ['generalize', 'algorithm', 'examples', 'supervised learning', 'example']



# Conclusion

After writing this tutorial, I decided to make it into a little Python package that can be installed via PyPI. The source code for the project is available [here](https://github.com/jaketae/wordwise). Obviously, there is a lot of documentation work to be done, but it is a starting point nonetheless. 


```
%%capture
!pip install wordwise
```

The wordwise package interface is exactly identical to the original vision we had in the introduction: an extractor model that can generate a set number of keywords from the candidates list. 


```python
from wordwise import Extractor

extractor = Extractor()
keywords = extractor.generate(text, 3)
print(keywords)
```

    ['algorithm', 'learning', 'supervised learning']


This was an interesting post in which we explored one of the countless use cases of using BERT embeddings. This also brings me a step closer to the vision of a fully automated blog article tagging pipeline that not only uses a supervised model that can perform multi-label classification, but also a more creative, generative portion of the workflow that can suggest salient keywords. Then, the final piece to this puzzle is a zero-shot learner that can determine whether these keywords are indeed good quality keywords, and even perform an additional layer of filtering. In a future post, we will see how one can used MNLI-trained models as zero-shot classifiers in this context. 

I hope you've enjoyed reading this post. Catch you up in the next one!
