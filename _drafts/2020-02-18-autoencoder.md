---
title: So What are Autoencoders?
toc: true
categories:
  - study
tags:
  - deep_learning
  - tensorflow
---

In today's post, we will take yet another look at an interesting application of a neural network: [autoencoders](https://en.wikipedia.org/wiki/Autoencoder). There are many types of autoencoders, but the one we will be looking at today is the simplest one, which might be considered the vanilla autoencoder.



Let's first 

```python
import os
import datetime
import numpy as np
from tensorflow.keras import datasets
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import callbacks
%load_ext tensorboard
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
import matplotlib.pyplot as plt
plt.style.use('seaborn')
```



```python
compressed_dim = 128
image_shape = (28, 28, 1)
```


```python
def build_model(image_shape, compressed_dim):
  encoder_input = Input(shape=image_shape)
  x = Conv2D(16, 3, activation='relu', padding='same')(encoder_input)
  x = BatchNormalization()(x)
  x = MaxPooling2D(2, padding='same')(x)
  x = Conv2D(32, 3, activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(2, padding='same')(x)
  x = Conv2D(32, 3, activation='relu', padding='same')(x)
  x = Flatten()(x)
  encoder_output = Dense(compressed_dim, activation='sigmoid')(x)
  x = Dense(7 * 7 * 32, activation='relu')(encoder_output)
  x = Reshape((7, 7, 32))(x)
  x = Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = UpSampling2D(2)(x)
  x = Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = UpSampling2D(2)(x)
  x = Conv2DTranspose(16, 3, activation='relu', padding='same')(x)
  decoder_output = Conv2D(1, 3, activation='sigmoid', padding='same')(x)
  encoder = Model(encoder_input, encoder_output)
  autoencoder = Model(encoder_input, decoder_output)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  print(autoencoder.summary())
  return encoder, autoencoder
```

```python
encoder, autoencoder = build_model(image_shape, compressed_dim)
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/nn_impl.py:183: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 28, 28, 1)]       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 16)        160       
    _________________________________________________________________
    batch_normalization (BatchNo (None, 28, 28, 16)        64        
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 16)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 32)        4640      
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 14, 14, 32)        128       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 32)          0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 7, 7, 32)          9248      
    _________________________________________________________________
    flatten (Flatten)            (None, 1568)              0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               200832    
    _________________________________________________________________
    dense_1 (Dense)              (None, 1568)              202272    
    _________________________________________________________________
    reshape (Reshape)            (None, 7, 7, 32)          0         
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 7, 7, 32)          9248      
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 7, 7, 32)          128       
    _________________________________________________________________
    up_sampling2d (UpSampling2D) (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_transpose_1 (Conv2DTr (None, 14, 14, 32)        9248      
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 14, 14, 32)        128       
    _________________________________________________________________
    up_sampling2d_1 (UpSampling2 (None, 28, 28, 32)        0         
    _________________________________________________________________
    conv2d_transpose_2 (Conv2DTr (None, 28, 28, 16)        4624      
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 28, 28, 1)         145       
    =================================================================
    Total params: 440,865
    Trainable params: 440,641
    Non-trainable params: 224
    _________________________________________________________________

```python
plot_model(autoencoder, show_shapes=True, show_layer_names=True)
```




<img src="/assets/images/2020-02-18-autoencoder_files/2020-02-18-autoencoder_3_0.png">




```python
def load_data():
  (X_train, _), (X_test, _) = datasets.mnist.load_data()
  X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
  X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))
  X_train, X_test = X_train.astype('float64') / 255., X_test.astype('float64') / 255.
  return X_train, X_test
```


```python
X_train, X_test = load_data()
```


```python
def add_noise(data, noise_factor):
  data_noise = data + noise_factor * np.random.normal(size=data.shape)
  data_noise = np.clip(data_noise, 0., 1.)
  return data_noise
```


```python
X_train_noise = add_noise(X_train, 0.5)
```


```python
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callback = callbacks.TensorBoard(log_dir, histogram_freq=0)

history = autoencoder.fit(X_train_noise, X_train, 
                          epochs=35,
                          batch_size=64,
                          shuffle=True,
                          validation_split=0.1,
                          callbacks=[callback])
```

    Train on 54000 samples, validate on 6000 samples
    Epoch 1/35
    54000/54000 [==============================] - 9s 170us/sample - loss: 0.1358 - val_loss: 0.1091
    Epoch 2/35
    54000/54000 [==============================] - 6s 117us/sample - loss: 0.1046 - val_loss: 0.1041
    Epoch 3/35
    54000/54000 [==============================] - 6s 118us/sample - loss: 0.1004 - val_loss: 0.1001
    Epoch 4/35
    54000/54000 [==============================] - 6s 118us/sample - loss: 0.0982 - val_loss: 0.1001
    Epoch 5/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0966 - val_loss: 0.0995
    Epoch 6/35
    54000/54000 [==============================] - 6s 117us/sample - loss: 0.0956 - val_loss: 0.0991
    Epoch 7/35
    54000/54000 [==============================] - 6s 117us/sample - loss: 0.0946 - val_loss: 0.0969
    Epoch 8/35
    54000/54000 [==============================] - 6s 117us/sample - loss: 0.0939 - val_loss: 0.0971
    Epoch 9/35
    54000/54000 [==============================] - 6s 117us/sample - loss: 0.0932 - val_loss: 0.0966
    Epoch 10/35
    54000/54000 [==============================] - 6s 117us/sample - loss: 0.0928 - val_loss: 0.0959
    Epoch 11/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0922 - val_loss: 0.0966
    Epoch 12/35
    54000/54000 [==============================] - 6s 117us/sample - loss: 0.0917 - val_loss: 0.0958
    Epoch 13/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0914 - val_loss: 0.0958
    Epoch 14/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0910 - val_loss: 0.0970
    Epoch 15/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0907 - val_loss: 0.0961
    Epoch 16/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0903 - val_loss: 0.0983
    Epoch 17/35
    54000/54000 [==============================] - 6s 118us/sample - loss: 0.0900 - val_loss: 0.0987
    Epoch 18/35
    54000/54000 [==============================] - 7s 121us/sample - loss: 0.0898 - val_loss: 0.0963
    Epoch 19/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0895 - val_loss: 0.0953
    Epoch 20/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0893 - val_loss: 0.0959
    Epoch 21/35
    54000/54000 [==============================] - 6s 117us/sample - loss: 0.0890 - val_loss: 0.0954
    Epoch 22/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0888 - val_loss: 0.0953
    Epoch 23/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0887 - val_loss: 0.0954
    Epoch 24/35
    54000/54000 [==============================] - 6s 117us/sample - loss: 0.0885 - val_loss: 0.0958
    Epoch 25/35
    54000/54000 [==============================] - 6s 117us/sample - loss: 0.0882 - val_loss: 0.0958
    Epoch 26/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0880 - val_loss: 0.0966
    Epoch 27/35
    54000/54000 [==============================] - 6s 117us/sample - loss: 0.0879 - val_loss: 0.0956
    Epoch 28/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0877 - val_loss: 0.0956
    Epoch 29/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0876 - val_loss: 0.0954
    Epoch 30/35
    54000/54000 [==============================] - 6s 117us/sample - loss: 0.0874 - val_loss: 0.0959
    Epoch 31/35
    54000/54000 [==============================] - 6s 118us/sample - loss: 0.0873 - val_loss: 0.0959
    Epoch 32/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0872 - val_loss: 0.0960
    Epoch 33/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0871 - val_loss: 0.0958
    Epoch 34/35
    54000/54000 [==============================] - 6s 117us/sample - loss: 0.0869 - val_loss: 0.0980
    Epoch 35/35
    54000/54000 [==============================] - 6s 116us/sample - loss: 0.0867 - val_loss: 0.0981



```python
%tensorboard --logdir logs
```




<div id="root"></div>
<script>
  (function() {
    window.TENSORBOARD_ENV = window.TENSORBOARD_ENV || {};
    window.TENSORBOARD_ENV["IN_COLAB"] = true;
    document.querySelector("base").href = "https://localhost:6006";
    function fixUpTensorboard(root) {
      const tftb = root.querySelector("tf-tensorboard");
      // Disable the fragment manipulation behavior in Colab. Not
      // only is the behavior not useful (as the iframe's location
      // is not visible to the user), it causes TensorBoard's usage
      // of `window.replace` to navigate away from the page and to
      // the `localhost:<port>` URL specified by the base URI, which
      // in turn causes the frame to (likely) crash.
      tftb.removeAttribute("use-hash");
    }
    function executeAllScripts(root) {
      // When `script` elements are inserted into the DOM by
      // assigning to an element's `innerHTML`, the scripts are not
      // executed. Thus, we manually re-insert these scripts so that
      // TensorBoard can initialize itself.
      for (const script of root.querySelectorAll("script")) {
        const newScript = document.createElement("script");
        newScript.type = script.type;
        newScript.textContent = script.textContent;
        root.appendChild(newScript);
        script.remove();
      }
    }
    function setHeight(root, height) {
      // We set the height dynamically after the TensorBoard UI has
      // been initialized. This avoids an intermediate state in
      // which the container plus the UI become taller than the
      // final width and cause the Colab output frame to be
      // permanently resized, eventually leading to an empty
      // vertical gap below the TensorBoard UI. It's not clear
      // exactly what causes this problematic intermediate state,
      // but setting the height late seems to fix it.
      root.style.height = `${height}px`;
    }
    const root = document.getElementById("root");
    fetch(".")
      .then((x) => x.text())
      .then((html) => void (root.innerHTML = html))
      .then(() => fixUpTensorboard(root))
      .then(() => executeAllScripts(root))
      .then(() => setHeight(root, 800));
  })();
</script>




```python
def show_image(data, num_row):
  num_image = num_row**2
  plt.figure(figsize=(10,10))
  for i in range(num_image):
    plt.subplot(num_row,num_row,i+1)
    plt.grid(False)
    plt.xticks([]); plt.yticks([])
    data_point = data[i].reshape(28, 28)
    plt.imshow(data_point, cmap=plt.cm.binary)
  plt.show()
```


```python
show_image(X_test, 5)
```


<img src="/assets/images/2020-02-18-autoencoder_files/2020-02-18-autoencoder_11_0.svg">



```python
X_test_noise = add_noise(X_test, 0.5)
show_image(X_test_noise, 5)
```


<img src="/assets/images/2020-02-18-autoencoder_files/2020-02-18-autoencoder_12_0.svg">



```python
decoded_images = autoencoder.predict(X_test_noise)
show_image(decoded_images, 5)
```


<img src="/assets/images/2020-02-18-autoencoder_files/2020-02-18-autoencoder_13_0.svg">

