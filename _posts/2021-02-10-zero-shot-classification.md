---
title: NLI Models as Zero-Shot Classifiers
mathjax: true
toc: true
categories:
  - study
tags:
  - deep_learning
  - nlp
---

In the [previous post](https://jaketae.github.io/study/keyword-extraction/), we took a look at how to extract keywords from a block of text using transformer models like BERT. In that blog post, you might recall that we used cosine similarity as a ditance measure to compare the relevance of a keyword. Namely, the higher the cosine similarity between the embedding of a keyword and the main text, the better the keyword in encapsulating the contents of the text. 

While this approach is sound, it comes with one drawback: BERT models are not really trained to produce embeddings for short n-grams or a single word. Therefore, the quality of embeddings produced by the model for n-gram candidate keywords cannot be guaranteed. 

One interesting approach I recently learned about in [this article](https://joeddav.github.io/blog/2020/05/29/ZSL.html) by Joe Davison presents a way of using a zero-shot approach to text classification. This seems like a nice alternative way of performing text classification in situations where the dataset is very small or even unavailable. 

Without further ado, let's jump right into it!

# Zero-Shot Learning

Zero-shot learning refers to a problem setup in which a model has to perform classification on labels it has never seen before. One advantage we have in the domain of NLP is that, just like the input, the dataset labels are also in text format. In other words, language models can be applied to both the text and label data. This provides a nice starting point for language models to be used as zero-shot learners: since language models have some level of natural language understanding by learning features from input texts, they can also understand the meaning of labels. 

## NLI Task

NLI, or [natural language inference](http://nlpprogress.com/english/natural_language_inference.html), refers to one of the many types of downstream NLP tasks in which a model is asked to identify some relationship between two sentences: a hypothesis and a premise. If the premise and hypothesis are compatible, this pair represents an entailment; conversely, if they are incompatible, the pair would be labeled as a contradiction. If the two sentences are irrelevant to each other, this represents a neutral example. Let's put this into context with an example.

* Premise: A soccer game with multiple males playing.
* Hypothesis: Some men are playing a sport.
* Label: Entailment

To summarize, NLI task forces the model to find relationships between sentence pairs to find one of three possible relationships. 

## NLI to Classification

Joe Davison's article presents an approach outlined by [Yin et. al](https://arxiv.org/abs/1909.00161), which is also the crux of this blog post. The idea behind this is surprisingly simple and intuitive, yet also very compelling. Suppose that we have the following sentence pairs. 

* Premise: Joe Biden's inauguration... (text omitted)
* Hypthesis: This example is about politics.

This is obviously a classification task simply framed into an NLI problem. To us, it might seem like a simple hack or a flimsy workaround, but in practice, this means that any model pretrained on NLI tasks can be used as text classifiers, even without fine-tuning. In other words, we have a zero-shot text classifier. 

Now that we have a basic idea of how text classification can be used in conjunction with NLI models in a zero-shot setting, let's try this out in practice with HuggingFace transformers.

# Demo

This notebook was written on Colab, which does not ship with the transformers library by default. Let's install the package first.


```
%%capture
!pip install transformers
```

There are many NLI models out there, but a popular one is Facebook's [BART](https://arxiv.org/abs/1910.13461). I might write a separate post in the future about the BART model (and for that matter, other interesting SOTA models that came after BERT, such as XLNet and T5, but those are topics for another day), but the gist of it is that BART is a transformer encoder-decoder model that excels in summarization tasks. BART NLI is available on the HuggingFace model hub, which means they can be downloaded as follows. Also, note that this is model is the large model, weighing in at around 1.6 gigabytes. 


```
from transformers import BartForSequenceClassification, BartTokenizer

model_name = "facebook/bart-large-mnli"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForSequenceClassification.from_pretrained(model_name)
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=898823.0, style=ProgressStyle(descripti…


​    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=456318.0, style=ProgressStyle(descripti…


​    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=908.0, style=ProgressStyle(description_…


​    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=1629486723.0, style=ProgressStyle(descr…


​    


    Some weights of the model checkpoint at facebook/bart-large-mnli were not used when initializing BartForSequenceClassification: ['model.encoder.version', 'model.decoder.version']
    - This IS expected if you are initializing BartForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BartForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


Let's consider a dummy example. Consider the following pair of premise and hypothesis sentences.


```
premise = "Who are you voting for in 2020?"
hypothesis = "This text is about politics."
```

If the model performs well, we would expect the model to predict an entailment. Voting and politics are highly related topics, so the model should not predict neutral or contradiction labels. Let's see if this is indeed the case.


```
tokens = tokenizer(premise, hypothesis, return_tensors="pt")
outputs = model(**tokens)
outputs.keys()
```




    odict_keys(['logits', 'past_key_values', 'encoder_last_hidden_state'])



By default, HuggingFace models almost always output a dictionary, which is why I like to check their keys. Here, we see a number of different keys, but the one that is most relevant is the logits. Let's check its dimensions.


```
logits = outputs.logits
logits.shape
```




    torch.Size([1, 3])



The batch contained only one example, and the model should output logits for three labels, each for contradiction, neutral, and entailment. 

In his article, Joe Davison drops the neutral column to explicitly frame this as a binary classification problem. In other words, an entailment corresponds to a positive example that belongs to the target class, and a contradiction indicatess a negative sample. 


```
# (contradiction, neutral, entailment)
# drop neutral 
entail_contradiction_logits = logits[:,[0,2]]
entail_contradiction_logits
```




    tensor([[-2.5443,  1.3904]], grad_fn=<IndexBackward>)



Finally, we can pass the logits through a softmax to obtain a probability interpretation of the final result. 


```
probs = entail_contradiction_logits.softmax(dim=1)
probs
```




    tensor([[0.0192, 0.9808]], grad_fn=<SoftmaxBackward>)



In this case, the model correctly infers the relationship as an entailment, or a positive label in binary classification terms. 

Now, you can see how this trick can be understood as a zero-shot learner setting. The pretrained BART model was never fine-tuned on some specific downstream classification task in a supervised environment; however, it is able to correctly classify whether a given article belongs to a certain category. 

The obvious benefit of this model, therefore, is that it can be applied to any labels. While supervised, fine-tuned models can only accurately predict whether some data belongs to one or more of the labels it wass trained on, zero-shot learnings can be used to classify documents in contexts where the labels may dynamically grow or change over time.

# Transformer Pipeline

The transformers library comes with a number of abstracted, easy-to-use recipes, which are called pipelines. The application of NLI models as zero-shot text classifiers we have just seen above is also one of those recipes. Concretely, the process looks as follows.


```
from transformers import pipeline

tags = [ "robots", "space & cosmos", "scientific discovery", "microbiology", "archeology"]
text = """Europa is one of Jupiter’s four large Galilean moons. 
          And in a paper published Monday in the Astrophysical Journal, 
          Dr. Batygin and a co-author, Alessandro Morbidelli, 
          a planetary scientist at the Côte d’Azur Observatory in France, 
          present a theory explaining how some moons form around gas giants like 
          Jupiter and Saturn, suggesting that millimeter-sized grains of hail 
          produced during the solar system’s formation became trapped 
          around these massive worlds, taking shape one at a time into 
          the potentially habitable moons we know today."""

classifier = pipeline("zero-shot-classification")
outputs = classifier(text, tags, multi_class=True)
```

As you can see, we initialize a classifier through the pipeline, then pass to the pipline a chunk of text as well as a number of different labels in list form. We can also specify multi-class as true. 

Let's take a look at the outputs.


```
for label, score in zip(outputs["labels"], outputs["scores"]):
    print(f"{label}: {score:.3f}")
```

    scientific discovery: 0.937
    space & cosmos: 0.904
    archeology: 0.049
    microbiology: 0.016
    robots: 0.007


Skimming through the block of text, we can infer that it probably came from some article about astronomy. The model thus correctly identifies that the likely labels are scientific discovery, and space & cosmos. Other irrelevant labels, such as archeology and robots, have a very low score.

# Conclusion

In this post, we explored how language models pretrained on NLI tasks can be used as zero-shot learners in a text classification task. The intuition behind this is straightforward, and so is the implementation. 

In my journey towards building a complete blog auto-tagger, I now see a clearer roadmap. 

* A fine-tuned RoBERTa model for classification on the top $K$ most commonly occuring labels
* A pretrained sentence BERT model for embedding-based keyword extraction
* A pretrained BART model for zero-shot text classifier, with extracted keywords as possible labels

If I could add one more component to this pipeline, it would be a text summarizer to avoid chunking or dividing the original text into pieces. But at this point, the pipeline is really starting to make definitive shape. 

I hope you've enjoyed reading this post. Catch you up in the next one!
