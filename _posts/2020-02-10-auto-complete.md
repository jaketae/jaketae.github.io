---
title: A Simple Autocomplete Model
toc: true
categories:
  - study
tags:
  - deep_learning
  - tensorflow
  - machine_learning
---

Autocomplete is now a part of our lives. 

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
import matplotlib.pyplot as plt
plt.style.use('seaborn')
```


```python
def get_text():
  path = tf.keras.utils.get_file('nietzsche.txt',
                               origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt') 
  text = open(path).read().lower()
  return text
```


```python
text_data = get_text()
print("Character length: {0}".format(len(text_data)))
```

    Character length: 600893



```python
print(text_data[:100])
```

    preface


​    
​    supposing that truth is a woman--what then? is there not ground
​    for suspecting that all ph



```python
def preprocess_split(text, max_len, step):
  sentences, next_char = [], []
  for i in range(0, len(text) - max_len, step):
    sentences.append(text[i: i + max_len])
    next_char.append(text[i + max_len])
  char_lst = sorted(list(set(text)))
  char_dict = {char: char_lst.index(char) for char in char_lst}
  X = np.zeros((len(sentences), max_len, len(char_lst)), dtype=np.bool)
  y = np.zeros((len(next_char), len(char_lst)), dtype=np.bool)
  for i, sentence in enumerate(sentences):
    for j, char in enumerate(sentence):
      X[i, j, char_dict[char]] = 1
    y[i, char_dict[next_char[i]]] = 1
  return X, y, char_dict
```


```python
max_len = 60
X, y, char_dict = preprocess_split(text_data, max_len, 3)
vocab_size = len(char_dict)
print("Number of sequences: {0}\nNumber of unique characters: {1}".format(len(X), vocab_size))
```

    Number of sequences: 200278
    Number of unique characters: 57



```python
def build_model(max_len, vocab_size):
  inputs = layers.Input(shape=(max_len, vocab_size))
  x = layers.LSTM(128)(inputs)
  output = layers.Dense(vocab_size, activation=tf.nn.softmax)(x)
  model = Model(inputs, output)
  model.compile(optimizer='adam', loss='categorical_crossentropy')
  return model
```


```
model = build_model(max_len, vocab_size)
plot_model(model, show_shapes=True, show_layer_names=True)
```




![png](2020-02-10-auto-complete_files/2020-02-10-auto-complete_7_0.png)




```python
history = model.fit(X, y, epochs=50, batch_size=128)
```

    Train on 200278 samples
    Epoch 1/50
    200278/200278 [==============================] - 164s 817us/sample - loss: 2.5568
    Epoch 2/50
    200278/200278 [==============================] - 163s 813us/sample - loss: 2.1656
    Epoch 3/50
    200278/200278 [==============================] - 162s 810us/sample - loss: 2.0227
    Epoch 4/50
    200278/200278 [==============================] - 162s 809us/sample - loss: 1.9278
    Epoch 5/50
    200278/200278 [==============================] - 161s 805us/sample - loss: 1.8586
    Epoch 6/50
    200278/200278 [==============================] - 162s 811us/sample - loss: 1.8032
    Epoch 7/50
    200278/200278 [==============================] - 163s 815us/sample - loss: 1.7582
    Epoch 8/50
    200278/200278 [==============================] - 165s 825us/sample - loss: 1.7197
    Epoch 9/50
    200278/200278 [==============================] - 167s 833us/sample - loss: 1.6866
    Epoch 10/50
    200278/200278 [==============================] - 166s 830us/sample - loss: 1.6577
    Epoch 11/50
    200278/200278 [==============================] - 165s 823us/sample - loss: 1.6312
    Epoch 12/50
    200278/200278 [==============================] - 162s 810us/sample - loss: 1.6074
    Epoch 13/50
    200278/200278 [==============================] - 162s 811us/sample - loss: 1.5862
    Epoch 14/50
    200278/200278 [==============================] - 161s 805us/sample - loss: 1.5668
    Epoch 15/50
    200278/200278 [==============================] - 165s 822us/sample - loss: 1.5492
    Epoch 16/50
    200278/200278 [==============================] - 166s 829us/sample - loss: 1.5333
    Epoch 17/50
    200278/200278 [==============================] - 167s 832us/sample - loss: 1.5182
    Epoch 18/50
    200278/200278 [==============================] - 166s 828us/sample - loss: 1.5051
    Epoch 19/50
    200278/200278 [==============================] - 166s 827us/sample - loss: 1.4922
    Epoch 20/50
    200278/200278 [==============================] - 164s 819us/sample - loss: 1.4801
    Epoch 21/50
    200278/200278 [==============================] - 165s 826us/sample - loss: 1.4688
    Epoch 22/50
    200278/200278 [==============================] - 165s 826us/sample - loss: 1.4582
    Epoch 23/50
    200278/200278 [==============================] - 165s 822us/sample - loss: 1.4488
    Epoch 24/50
    200278/200278 [==============================] - 163s 813us/sample - loss: 1.4386
    Epoch 25/50
    200278/200278 [==============================] - 167s 832us/sample - loss: 1.4305
    Epoch 26/50
    200278/200278 [==============================] - 166s 830us/sample - loss: 1.4220
    Epoch 27/50
    200278/200278 [==============================] - 167s 832us/sample - loss: 1.4137
    Epoch 28/50
    200278/200278 [==============================] - 167s 833us/sample - loss: 1.4060
    Epoch 29/50
    200278/200278 [==============================] - 166s 827us/sample - loss: 1.3989
    Epoch 30/50
    200278/200278 [==============================] - 164s 820us/sample - loss: 1.3910
    Epoch 31/50
    200278/200278 [==============================] - 163s 815us/sample - loss: 1.3846
    Epoch 32/50
    200278/200278 [==============================] - 162s 810us/sample - loss: 1.3777
    Epoch 33/50
    200278/200278 [==============================] - 162s 809us/sample - loss: 1.3720
    Epoch 34/50
    200278/200278 [==============================] - 160s 798us/sample - loss: 1.3649
    Epoch 35/50
    200278/200278 [==============================] - 163s 815us/sample - loss: 1.3599
    Epoch 36/50
    200278/200278 [==============================] - 162s 807us/sample - loss: 1.3538
    Epoch 37/50
    200278/200278 [==============================] - 162s 808us/sample - loss: 1.3482
    Epoch 38/50
    200278/200278 [==============================] - 162s 809us/sample - loss: 1.3423
    Epoch 39/50
    200278/200278 [==============================] - 163s 813us/sample - loss: 1.3371
    Epoch 40/50
    200278/200278 [==============================] - 164s 820us/sample - loss: 1.3319
    Epoch 41/50
    200278/200278 [==============================] - 163s 814us/sample - loss: 1.3268
    Epoch 42/50
    200278/200278 [==============================] - 165s 825us/sample - loss: 1.3223
    Epoch 43/50
    200278/200278 [==============================] - 164s 820us/sample - loss: 1.3171
    Epoch 44/50
    200278/200278 [==============================] - 164s 819us/sample - loss: 1.3127
    Epoch 45/50
    200278/200278 [==============================] - 165s 822us/sample - loss: 1.3080
    Epoch 46/50
    200278/200278 [==============================] - 163s 813us/sample - loss: 1.3034
    Epoch 47/50
    200278/200278 [==============================] - 163s 813us/sample - loss: 1.2987
    Epoch 48/50
    200278/200278 [==============================] - 162s 809us/sample - loss: 1.2955
    Epoch 49/50
    200278/200278 [==============================] - 161s 804us/sample - loss: 1.2905
    Epoch 50/50
    200278/200278 [==============================] - 162s 811us/sample - loss: 1.2865



```python
from google.colab import files

model = model.save('model.hdf5')
files.download('model.hdf5')
```


```python
model = load_model('model.hdf5')
```


```python
def plot_learning_curve(history):
  loss = history.history['loss']
  epochs = [i for i, _ in enumerate(loss)]
  plt.scatter(epochs, loss, color='skyblue')
  plt.xlabel('Epochs'); plt.ylabel('Cross Entropy Loss')
  plt.show()
```


```
plot_learning_curve(history)
```


<img src="/assets/images/2020-02-10-auto-complete_files/2020-02-10-auto-complete_12_0.svg">



```python
def random_predict(prediction, temperature):
  prediction = np.asarray(prediction).astype('float64')
  log_pred = np.log(prediction) / temperature
  exp_pred = np.exp(log_pred)
  final_pred = exp_pred / np.sum(exp_pred)
  random_pred = np.random.multinomial(1, final_pred)
  return random_pred
```


```python
def generate_text(model, data, iter_num, seed, char_dict, temperature=1, max_len=60):
  entire_text = list(data[seed])
  for i in range(iter_num):
    prediction = random_predict(model.predict([[entire_text[i: i + max_len]]])[0], temperature)
    entire_text.append(prediction)
  reverse_char_dict = {value: key for key, value in char_dict.items()}
  generated_text = ''
  for char_vec in entire_text:
    index = np.argmax(char_vec)
    generated_text += reverse_char_dict[index]
  return generated_text
```


```python
def vary_temperature(temp_lst, model, data, iter_num, seed, char_dict):
  for temperature in temp_lst:
    print("Generated text at temperature {0}:\n{1}\n\n".format(temperature, generate_text(model, data, iter_num, seed, char_dict, temperature)))
```


```python
vary_temperature([0.3, 0.6, 0.9, 1.2], model, X, 1000, 10, char_dict)
```

    Generated text at temperature 0.3:
     is a woman--what then? is there not ground
    for suspecting that the experience and present strange of the soul is also as the stand of the most profound that the present the art and possible to the present spore as a man and the morality and present self instinct, and the subject that the presence of the surcessize, and also it is an action which the philosophers and the spirit has the consider the action to the philosopher and possess and the spirit is not be who can something the predicess of the constinate the same and self-interpatence, the disconsises what is not to be more profound, as if it is a man as a distance of the same art and ther strict to the presing to the result the problem of the present the spirit what is the consequences and the development of the same art of philosophers and security and spirit and for the subjective in the disturce, as in the contrary and present stronger and present could not be an inclination and desires of the same and distinguished that is the discoverty in such a person itself influence and ethers as


​    
​    Generated text at temperature 0.6:
​     is a woman--what then? is there not ground
​    for suspecting to and the world will had to a such that the basis of the incussions of the spirit as the does not because actian free spirits of intellect of the commstical purtious expression of men are so much he is not unnor experiences of self-conturity, and
​    as anifegently religious in the man would not consciously, his action is not be actian at in accombs life for the such all procees of great and the heart of this conduct the spirity of the man can provate for in any
​    once in any of the suriticular conduct that which own needs, when they are therefore, as
​    such action and some difficulty that the strength, it, himself which has to its fine term of pricismans the exacte in its self-recuphing and every strength and man to wist the action something man as the worst, that the was of a longent that the whole not be all the very subjectical proves the stronger extent he is necessary to metaphysical figure of the faith in the bolity in the pure belief--as "the such a successes of the values--that is he 


​    
​    Generated text at temperature 0.9:
​     is a woman--what then? is there not ground
​    for suspecting that they grasutes, and so farmeduition of the does not only with this
​    constrbicapity have honour--and who distical seclles are denie'n, is one samiles are no luttrainess,
​    and ethic and matficulty, concudes of morality to
​    rost were presence of lighters caseful has prescally here at last not and servicatity, leads falled for child real appreparetess of worths--the
​    resticians when one to persans as a what a mean of that is as to the same heart tending noble stimptically and particious, we pach yought for that mankind, that the same take frights a contrady has howevers of a surplurating or in fact a sort, without present superite fimatical matterm of our being interlunally
​    men who cal
​    scornce. the shrinking's
​    proglish, and
​    traints he way to demitable pure explised and place can
​    deterely by the compulse in whom is phypociative cinceous, and the higher and will bounthen--in itsiluariant upon find the "first the whore we man will simple condection and some than us--a valuasly refiges who feel


​    
​    Generated text at temperature 1.2:
​     is a woman--what then? is there not ground
​    for suspecting that he therefore when shre, mun, a schopenhehtor abold gevert.


​    
​    120
​    
    =as in
    find that is _know believinally bad,[
    euser of view.--bithic
    iftel canly
    in any
    knowitumentially. the charm surpose again, in
    swret feathryst, form of kinne of the world bejud--age--implaasoun ever? but that the is any
    appearance has clenge: the? a plexable gen preducl=s than condugebleines and aligh to advirenta-nasure;
    findiminal it as, not take. the ideved towards upavanizing, would be
    thenion, in all pespres: it is of
    a concidenary, which, well founly con-utbacte udwerlly upon mansing--frauble of "arrey been can the pritarnated from their
    christian often--think prestation of mocives." legt, lenge:--this deps
    telows, plenhance of decessaticrances). hyrk an interlusally" tone--under good haggy,"
    is have we leamness of conschous should it, of
    sicking ummenfeckinal zerturm erienweron of noble of
    himself-clonizing there is conctumendable prefersy
    exaitunia states," whether
    they deve oves any of hispyssesss. int


​    

