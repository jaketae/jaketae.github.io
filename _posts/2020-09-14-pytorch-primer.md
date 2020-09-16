---
title: A PyTorch Primer
mathjax: true
toc: true
categories:
  - study
tags:
  - deep_learning
  - pytorch
---

I've always been a fan of TensorFlow, specifically `tf.keras`, for its simplicity and ease of use in implementing algorithms and building models. Today, I decided to give PyTorch a try. It is my understanding that TensorFlow is more often used in coporate production environments, whereas PyTorch is favored by academics, especially those in the field of NLP. I thought it would be an interesting idea to give it a try, so here is my first go at it. Note that the majority of the code shown here are either borrowed from or are adaptations of those available on the [PyTorch website](https://pytorch.org/tutorials/), which is full of rich content and tutorials for beginners. Of course, basic knowledge of DL and Python would be helpful, but otherwise, it is a great place to start.

Let's dive right in!

# Autograd

Like TensorFlow, PyTorch is a scientific computing library that makes use of GPU computing power to acceleration calculations. And of course, it can be used to create neural networks. In this section, we will take a look at how automatic differentiation works in PyTorch. Note that differentiation is at the core of backpropagation, which is why demonstrating what might seem like a relatively low-level portion of the API is valuable.

## Gradient Calculation

Let's begin our discussion by first importing the PyTorch module.


```python
import torch
```

It isn't difficult to see that `torch` is a scientific computing library, much like `numpy`. For instance, we can easily create a matrice of ones as follows:


```python
x = torch.ones(2, 2, requires_grad=True); print(x)
```

    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)


The `requires_grad` is a parameter we pass into the function to tell PyTorch that this is something we want to keep track of later for something like backpropagation using gradient computation. In other words, it "tags" the object for PyTorch.

Let's make up some dummy operations to see how this tagging and gradient calculation works. 


```python
y = x + 2; print(f"y = {y}")
z = y * y * 3; print(f"z = {z}")
out = z.mean(); print(f"out = {out}")
```

    y = tensor([[3., 3.],
            [3., 3.]], grad_fn=<AddBackward0>)
    z = tensor([[27., 27.],
            [27., 27.]], grad_fn=<MulBackward0>)
    out = 27.0


Note that `*` performs element-wise multiplication, otherwise known as the dot product for vectors and the hadamard product for matrics and tensors. 

Let's look at how autograd works. To initiate gradient computation, we need to first call `.backward()` on the final result, in which case `out`.


```python
out.backward()
```

Then, we can simply call `x.grad` to tell PyTorch to calculate the gradient. Note that this works only because we "tagged" `x` with the `require_grad` parameter. If we try to call `.grad` on any of the other intermediate variables, such as `y` or `z`, PyTorch will complain.


```python
x.grad
```




    tensor([[4.5000, 4.5000],
            [4.5000, 4.5000]])



Let's try to understand the result of this computation. Let $o$ denote the final `output` tensor. Since we called `z.mean()`, and since `z` has a total of four elements, we can write out our dummy calculations mathematically in the following fashion:

$$
\begin{align}
o 
&= \frac14 \sum_{i=1}^4 z_i \\
&= \frac34 \sum_{i=1}^4 y_i^2 \\
&= \frac34 \sum_{i=1}^4 (x_i + 2)^2 
\end{align}
$$

Using partial differentiation to obtain the gradients,

$$
\frac{d o_i}{d x_i} = \frac34 \cdot 2(x_i + 2) 
$$

Since $x_i = 1$,

$$
\frac{d o_i}{d x_i} = \frac92 = 4.5
$$

Since $i$ is just an arbitrary, non-specific index out of a total of four, we can easily see that the same applies for all other indices, and hence we will end up with a matrix whose all four entries take the value of 4.5, as PyTorch has rightly computed.

## Custom Autograd Functions

We can go even a step farther and declare custom operations. For example, here's a dummy implementation of the ReLU function. 


```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
```

Let's talk about the `MyReLU.forward()` method first. Note that it takes in two argument parameters: `ctx` and `input`. As you might have guessed, `input` is simply the value that the function will be provided with. The `ctx` can simply be thought of as a cache where we can store vectors or matrices to be used during backpropagation. In this case, we store the `input` by calling `ctx.save_for_backward()` method.

During the backward pass, we compute the gradient. Here, we need to retrieve the `input` variable which was stored in the context. This is because the ReLU function takes the following form:

$$
\text{ReLU}(x) = 
\begin{cases}
0 & x < 0 \\
x & x \geq 0
\end{cases}
$$

Thus, its derivative is

$$
\text{ReLU}^\prime(x) = 
\begin{cases}
0 & x < 0 \\
1 & x \geq 0
\end{cases}
$$

During backpropagation, this means that gradients will flow down to the next layer only for those indices whose input elements to te ReLU function were greater than 0. Thus, we need the input vector for reference purposes, and this is done via stashing it in the `ctx` variable.

We will see how we can incorporate `MyReLU` into the model in the next section.

# Simple Model

In this example, we'll take a look at an extremely simple model to gain a better understanding of how everything comes into play in a more practical example. 

## Manual Gradient Computation

This is the method that I've mostly been using when implementing simple dense fully-connected models in NumPy. The idea is that we would mathematically derive the formula for the gradients ourselves, then backpropagate these values during the optimization process. Of course, this can be done with PyTorch.

To build our simple model, let's first write out some variables to use, starting with the configuration of our model and its dimensions.


```python
dtype = torch.float
batch_size, input_dim, hidden_dim, output_dim = 64, 1000, 100, 10
```

We will also need some input and output tensors to be fed into the model for trainining and optimization.


```python
x = torch.randn(batch_size, input_dim, dtype=dtype)
y = torch.randn(batch_size, output_dim, dtype=dtype)
```

Next, here are the weight matrices we will use. For now, we assume a simple two layered dense feed forward network.


```python
w1 = torch.randn(input_dim, hidden_dim, dtype=dtype)
w2 = torch.randn(hidden_dim, output_dim, dtype=dtype)
```

Last but not least, let's define a simple squared error loss function to use during the training step.


```python
def loss_fn(y, y_pred):
    return (y_pred - y).pow(2).sum()
```

With this entire setup, we can now hash out what the entire training iteration is going to look like. Wrapped in a loop, we perform one forward pass, then perform backpropagation to adjust the weights. 


```python
learning_rate = 1e-6
for t in range(500):
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # .item() to access number from 1d tensor
    L = loss_fn(y, y_pred).item()
    if t % 100 == 99:
        print(f"Epoch {t}: {L}")

    # Backprop
    # dL/dy
    grad_y_pred = 2.0 * (y_pred - y)
    # dL/dw2 = dL/dy * dy/dw2
    grad_w2 = h_relu.t().mm(grad_y_pred)
    # dL/dh_relu = dL/dy * dy/dh_relu
    grad_h_relu = grad_y_pred.mm(w2.t())
    # dL/dh = dL/dh_relu if h > 0, else 0
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    # dL/dw1 = dL/dh * dh/dw1
    grad_w1 = x.t().mm(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

    Epoch 99: 714.8318481445312
    Epoch 199: 3.586176633834839
    Epoch 299: 0.03582914546132088
    Epoch 399: 0.0007462671492248774
    Epoch 499: 8.23544614831917e-05


Great! We see that the loss drops as more epochs elapse. 

While there is no problem with this approach, things can get a lot more unwieldy once we start building out more complicated models. In these cases, we will want to use the auto differentiation functionality we reviewed earlier. Let's see this in action. Also, let's make this more PyTorch-y by making use of classes. We will revisit why class-based implementations are important in the next section.


```python
class FeedForwardNet:
    def __init__(self):
        self.w1 = torch.randn(
            input_dim, hidden_dim, dtype=dtype, requires_grad=True
        )
        self.w2 = torch.randn(
            hidden_dim, output_dim, dtype=dtype, requires_grad=True
        )

    def forward(self, x):
        return x.mm(self.w1).clamp(min=0).mm(self.w2)

    def backward(self, x, y, verbose=False):
        y_pred = self.forward(x)
        L = loss_fn(y, y_pred)
        verbose and print(f"Loss: {L}")
        L.backward()

        with torch.no_grad():
            self.w1 -= learning_rate * self.w1.grad
            self.w2 -= learning_rate * self.w2.grad

            self.w1.grad.zero_()
            self.w2.grad.zero_()
```


```python
model = FeedForwardNet()

for t in range(500):
    verbose = t % 100 == 99
    model.backward(x, y, verbose=verbose)
```

    Loss: 321.8311767578125
    Loss: 0.6376868486404419
    Loss: 0.0022135386243462563
    Loss: 7.520567305618897e-05
    Loss: 1.823174170567654e-05


Notice we didn't have to explicitly specify the backpropagation formula with matrix derivatives: by simply calling `.grad` properties for each of the weights matrices, we were able to perform gradient descent. One detail to note is that, unlike in the case above where we had to explicitly call `L.item()` in order to obtain the loss value---which would be of type `float`---we leave the computed loss to remain as a tensor in order to call `L.backward()`. We also make sure to reset the gradients per epoch by calling `self.w.grad.zero_()`.

## Applying Custom Function

We can also improve our implementation by making use of the `MyReLU` class that we implemented earlier. This is simple as doing 


```python
def forward(self, x):
    return MyReLU.apply(x.mm(self.w1)).mm(self.w2)
```

This might be a better way to implement the function for reasons of simplicity and readability. Although `clamp`ing works, it's more arguably cleaner to write a ReLU this way. Also, this is a dummy example, and we can imagine a lot of situations where we might want to write custom functions to carry out specific tasks.

# Model Declarations

Much like TensorFlow, PyTorch offers to ways of declaring models: function-based and class-based methods. Although I have just started getting into PyTorch, my impression is that the later is more preferred by PyTorch developers, whereas this is not necessarily the case with Keras or `tf.keras`. Of course, this is a matter of preference and development setting, so perhaps such first impression generalizations do not carry much weight. Nonetheless, in this section, we will take a look at both ways of building models. Let's start with the function-based method.

## Sequential Model

The function-based method reminds me a lot of Keras's sequential method. Let's remind ourselves of Kera's basic sequential model API:

```python
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
```

Now let's compare this method with PyTorch's way of declaring sequential models:


```python
model = torch.nn.Sequential(
    torch.nn.Linear(input_dim, hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, output_dim),
)
```

This model declaration is in fact exactly identical to the simple model we have declared above.

You can easily see how similar this code snippet is to the Keras example. The only difference is that the activation function is declared independently of the layer itself in PyTorch, whereas Keras combines them into one via `activation="relu"` argument. Of course, you don't have to specify this argument, and we can import the ReLU function from TensorFlow to make it explicit like the PyTorch example. The point, however, is that the sequential model API for both libraries are pretty similar.

## Class-Based Declaration

Another way to build models is by subclassing `torch.nn`. The `nn` submodule in PyTorch is the one that deals with neural networks; hence the `nn`. This subclassing might look as follows:


```python
class TwoLayerNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
```

This model is no different from the `FeedForwardNet` we defined earlier. The only notable difference is that we didn't define a separate `self.backware()` type function. For the most part, the overall idea boils down to

* The weights are definited in the `__init__` function
* The `forward` function deals with forward pass

Now let's take a look at what the training code looks like.


```python
model = TwoLayerNet(input_dim, hidden_dim, output_dim)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(f"Epoch {t}: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

    Epoch 99: 2.7017245292663574
    Epoch 199: 0.050356119871139526
    Epoch 299: 0.001569570624269545
    Epoch 399: 5.8641602663556114e-05
    Epoch 499: 2.4734026737860404e-06


Although things might look a bit different, there's not much going on in this process, other than the fact that some of the functions and logic we wrote before are now abstracted away by PyTorch. For example, we see `criterion`, which is effectively the mean squared error loss, similar to how we defined `loss_func` above. Another difference we see is `optimizer`, which, as the variable name makes apparent, is the optimizer that we use for backpropagation. In this specific instance, we use SGD. Each backpropagation step is then performed simply via `optimizer.step()`. 

# Conclusion

In this tutorial, we took a very brief look at the PyTorch model. This is by no means a comprehensive guide, and I could not even tell anyone that I "know" how to use PyTorch. Nonetheless, I'm glad that I was able to gain some exposure to the famed PyTorch module. Also, working with Django has somewhat helped me grasp the idea of classes more easily, which certainly helped me take in class-based concepts in PyTorch more easily. I distinctively remember people saying that PyTorch is more object-oriented compared to TensorFlow, and I might express agreement to that statement after having gone through the extreme basics of PyTorch. 

In the upcoming articles, I hope to use PyTorch to build more realistic models, preferrably in the domain of NLP, as that seems to be where PyTorch's comparative advantage stands out the most compared to TensorFlow. Of course, this is not to say that I don't like TensorFlow anymore, or that PyTorch is not an appropriate module to use in non-NLP contexts: I think each of them are powerful libraries of their own that provide a unique set of functionalities for the user. And being bilinguial---or even a polyglot, if you can use things like Caffe perhaps---in the DL module landscape will certainly not hurt at all. 

I hope you've enjoyed this article. Catch you up in the next one!
