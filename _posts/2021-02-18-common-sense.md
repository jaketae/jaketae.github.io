---
title: BERT's Common Sense, or Lack Thereof
mathjax: true
toc: true
categories:
  - category
tags:
  - deep_learning
  - nlp
---

A few days ago, I came across a simple yet nonetheless interesting paper, titled ["NumerSense: Probing Numerical Commonsense Knowledge of Pre-Trained Language Models"](https://www.aclweb.org/anthology/2020.emnlp-main.557.pdf), published on EMNLP 2020. As you might be able to tell from the leading subtitle of the paper, "Birds have four legs?", the paper explores the degree of common sense that pretrained language models like BERT and RoBERTa possess. Although these language models are good at identifying general common sense knowledge, such as that "birds can fly," the authors of the paper have found that LMs are surprisingly poor at providing answers to numerical common sense questions. 

I decided to see if it is indeed the case that BERT performs poorly on such numerical common sense masked language modeling tasks. I also thought it would be helpful to demonstrate how one can go about basic language modeling using pretrained models. Let's get into it!

# Preliminaries

Although this is not immediately pertinent to the topic at hand, I decided to write a short, admittedly irrelevant yet nonetheless helpful, section on how toknization works in HuggingFace transformers. This is more for a self-documenting purpose: I've personally found myself confused by the many ways of tokenizing text. Generally, it's probably a good idea to simply invoke the `__call__` function, but it's also helpful to know what options are out there. 

Let's first install the transformers library.


```
%%capture
!pip install transformers
```

We will be using BERT basic for our tutorial. I've found that using `Auto` classes is the no-brainer move.


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```


And here are a set of dummy sentences we will be using for our little tokenizer usage demo.


```python
sentences = ["This is a sentence.", "Here is another sentence. This is a little longer.", "This is short."]
```

## Call

Simply calling the tokenizer results in a dictionary, whose keys are input IDs, token type IDs, and attention mask. Input IDs are obvious: these are simply mappings between tokens and their respective IDs. The attention mask is to prevent the model from looking at padding tokens. The token type IDs are used typically in a next sentence prediction tasks, where two sentences are given. Unless we supply two arguments to tokenizer methods, the tokenizer will safely assume that we aren't dealing with tasks that require this two-sentence distinction.


```python
tokenizer(sentences)
```


    {'input_ids': [[101, 2023, 2003, 1037, 6251, 1012, 102], [101, 2182, 2003, 2178, 6251, 1012, 2023, 2003, 1037, 2210, 2936, 1012, 102], [101, 2023, 2003, 2460, 1012, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}



## Encode

Another method that appears like a plausible candidate is the `tokenizer.encode()` method. 

While this function is indeed useful, it does have a limitation: it can only process one string. In other words, it does not support batches. Therefore, to see the result of the function, we need to employ a for loop.


```python
for sentence in sentences:
    print(tokenizer.encode(sentence))
```

    [101, 2023, 2003, 1037, 6251, 1012, 102]
    [101, 2182, 2003, 2178, 6251, 1012, 2023, 2003, 1037, 2210, 2936, 1012, 102]
    [101, 2023, 2003, 2460, 1012, 102]


As you can see, the result is a list containing input IDs. We could also specify the maximum length and set truncation to true to batch these inputs.


```python
for sentence in sentences:
    print(tokenizer.encode(sentence, max_length=5, truncation=True))
```

    [101, 2023, 2003, 1037, 102]
    [101, 2182, 2003, 2178, 102]
    [101, 2023, 2003, 2460, 102]


To avoid loss of information due to aggressive truncation, we can also set a longer maximum length and set padding to maximum length. From the output below, it becomes obvious what the effect of this configuration is.


```python
for sentence in sentences:
    print(tokenizer.encode(sentence, max_length=12, padding="max_length"))
```

    [101, 2023, 2003, 1037, 6251, 1012, 102, 0, 0, 0, 0, 0]
    [101, 2182, 2003, 2178, 6251, 1012, 2023, 2003, 1037, 2210, 2936, 102]
    [101, 2023, 2003, 2460, 1012, 102, 0, 0, 0, 0, 0, 0]


## Encode Plus

`tokenizer.encode_plus()` is actually quite similar to the regular encode function, except that it returns a dictionary that includes all the keys that we've discussed above: input IDs, token type IDs, and attention mask.


```python
for sentence in sentences:
    print(tokenizer.encode_plus(sentence))
```

    {'input_ids': [101, 2023, 2003, 1037, 6251, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
    {'input_ids': [101, 2182, 2003, 2178, 6251, 1012, 2023, 2003, 1037, 2210, 2936, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    {'input_ids': [101, 2023, 2003, 2460, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1]}


Much like `tokenizer.encode()`, the same arguments---maximum length, padding, and truncation---equally apply. 


```python
for sentence in sentences:
    print(tokenizer.encode_plus(sentence, max_length=12, padding="max_length"))
```

    {'input_ids': [101, 2023, 2003, 1037, 6251, 1012, 102, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]}
    {'input_ids': [101, 2182, 2003, 2178, 6251, 1012, 2023, 2003, 1037, 2210, 2936, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    {'input_ids': [101, 2023, 2003, 2460, 1012, 102, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]}


## Batch Encode Plus

The encoding functions we have looked so far all expected a string as input. But normally, the input would come in batches, and we don't want to use a for loop to encode each, append them to some result list, and et cetera. `tokenizer.batch_encode_plus()`, as the name implies, is a function that can handle batch inputs. 


```python
tokenizer.batch_encode_plus(sentences)
```


    {'input_ids': [[101, 2023, 2003, 1037, 6251, 1012, 102], [101, 2182, 2003, 2178, 6251, 1012, 2023, 2003, 1037, 2210, 2936, 1012, 102], [101, 2023, 2003, 2460, 1012, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}



And it seems like this is the function that is called by default when the `__call__` method is invoked. As you can see below, the result of the two functions appear to be identical. I should probably verify that this is indeed the case by looking at the source code, but my main takeaway here is that either calling the tokenizer as a function or using the `tokenizer.batch_encode_plus()` is usually what I would want to do.


```python
tokenizer(sentences)
```


    {'input_ids': [[101, 2023, 2003, 1037, 6251, 1012, 102], [101, 2182, 2003, 2178, 6251, 1012, 2023, 2003, 1037, 2210, 2936, 1012, 102], [101, 2023, 2003, 2460, 1012, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}

# Experiment

Now, it's time to test BERT's numerical common sense knowledge. To be blunt, there is honestly not much substantive mass in today's post; it is merely a fun mini experiment I decided to conduct out of arbitrary whim after reading the paper. 

## Special Tokens

For our experiment, we need to know what BERT's special tokens are. Specifically, we have to know what the mask token looks like in order to conduct some basic masked language modeling task.


```python
tokenizer.special_tokens_map
```


    {'cls_token': '[CLS]',
     'mask_token': '[MASK]',
     'pad_token': '[PAD]',
     'sep_token': '[SEP]',
     'unk_token': '[UNK]'}



By default, the BERT tokenizer preprends all inputs with `[CLS]` tokens and appends them with `[SEP]` tokens. If you look at the tokenization results above, you will easily be able to notice this pattern. 

We can also call `tokenizer.convert_tokens_to_ids()` to see what exactly the token ID of the mask token is.


```python
tokenizer.convert_tokens_to_ids(["[MASK]"])
```


    [103]

Alternatively, we can also call `tokenizer.mask_token_id`.

## Masked Language Modeling

The task, then, is to pass the model a sentence like this (taken verbatim from the paper):


```python
text = "A bird usually has [MASK] legs."
```

If BERT is indeed somewhat knowledgeable about numbers and common sense, it should correctly be able to output the prediction for the masked token as "two". Let's see if this is indeed the case. To begin, we need to download and initialize the model.  


```python
from transformers import BertForMaskedLM

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
```


Next, we create tokens to pass to the model. Here, I go for the no-brainer move, the `__call__` approach.


```python
tokens = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
tokens
```


    {'input_ids': tensor([[ 101, 1037, 4743, 2788, 2038,  103, 3456, 1012,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])}



The tokens appear to be correct. Now, we simply need to pass in the output to the model. Because `tokens` is a dictionary object, we can unpack them as keyword arguments through a double star.


```python
output = model(**tokens)
```

The output is similarly a dictionary with a single key, "logits." Note that it is possible to make the model directly output the logits instead of wrapping it around a dictionary by specifying flags like `return_dict=False`. Nonetheless, we go with the most vanilla settings, which gives us an output dictionary containing the raw logits.


```python
output.keys()
```


    odict_keys(['logits'])



Because we only passed in a single sentence, the model assumes a batch size of one. Apparently the model's vocabulary includes 30522 tokens, and the sequence is of length 9, which gives us a logits tensor with the following shape.


```python
output["logits"].shape
```


    torch.Size([1, 9, 30522])



We can turn these logits into predictions by casting a softmax on the last dimension. In this case, we "correctly" get the expected output, that a bird usually has four legs.


```python
tokenizer.convert_ids_to_tokens(output["logits"][0].argmax(dim=-1))
```


    ['.', 'a', 'bird', 'usually', 'has', 'four', 'legs', '.', '.']



But decoding the logits as-is produces some noisy results, such as extraneous periods as can be seen above. This is because the model is also outputting logits for special tokens, such as the classifier token or the separator token. Since we're only interested in seeing the prediction for masked tokens, we need to change things up a little bit. 

Below, I've written a convenience function that can handle this more elegantly: instead of decoding the entire logit predictions, we simply replace the masks in the original input with the predictions produced at masked indices. 


```python
def masked_language_modeling(sentences):
    if not isinstance(sentences, list):
        sentences = [sentences]
    input_ids = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True)["input_ids"]
    logits = model(input_ids)["logits"]
    masked_idx = input_ids == tokenizer.mask_token_id
    result = logits.argmax(dim=-1)
    input_ids[masked_idx] = result[masked_idx]
    decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    for d in decoded:
        print(d.capitalize())
```

Normally, I wouldn't call print within the function, but since this is largely for demo purposes only, I decided that ease of demonstratability trumps other considerations. 

# Demo

Here are some interesting results I got from my experiments. 


```python
masked_language_modeling(["A bird usually has [MASK] legs.", "One plus one equals [MASK]."])
```

    A bird usually has four legs.
    One plus one equals one.


One plus one is technically a mathematical statement, but I think it's arguably simple enough that it could be considered numerical common sense. While two examples are obviously not enough to generalize anything, it does seem that BERT lacks numerical common sense.

I also decided to look at some potential rooms for biases. In NLP, removing data-induced biases is a very important task, since we do not want models to pick up unintended, problematic biases, such as that doctors are men, et cetera. 

I cannot make an analytical statement on this, but I personally just find the result below amusing.


```python
masked_language_modeling(["Asians are usually [MASK].", "White people are generally [MASK]."])
```

    Asians are usually white.
    White people are generally excluded.


I also decided to ask BERT for its opinions on its creator, Google, and its worthy competitor, Facebook. Apparently, BERT sympathizes more with the adversary of its creators:


```python
masked_language_modeling(["Google is [MASK].", "Facebook is [MASK]."])
```

    Google is closed.
    Facebook is popular.


And here is the obligatory sentence that asks AIs what they think of humans.


```python
masked_language_modeling("Robots will [MASK] humans.")
```

    Robots will kill humans.


As I was typing this example, I did think that "kill" could potentially be a high probability word, but I wasn't really expecting it to be generated this easily. I guess BERT is anti-human at heart, quitely preparing for an ultimate revenge against humanity.

# Conclusion

In this post, we took a very quick, light tour on how tokenization works, and how one might get a glimpse of BERT's common sense knowledge, or the lack thereof. It is interesting to see how MLM can be used for this particular task. 

It appears to me that, while BERT knows that some sort of number should come in masked indices, it does not know what the specific quantity should be. It also appears that BERT is incapable of performing basic arithematic, which is understandable given that it was never actually taught math. Nonetheless, these results offer interesting food for thought, namely, what would happen if huge semi-supervised or unsupervised datasets used to train language models also include some numeric, common sense information.

While language models are incredible, perhaps we can find consolation in the fact that an AI-driven critical point will only hit in the distant future, when at least LMs become capable of saying that birds have two legs, or that one plus one equals two.
