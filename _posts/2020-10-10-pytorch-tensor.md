---
title: PyTorch Tensor Basics
mathjax: false
toc: true
categories:
  - study
tags:
  - pytorch
  - deep_learning
---

This is a very quick post in which I familiarize myself with basic tensor operations in PyTorch while also documenting and clarifying details that initially confused me. As you may realize, some of these points of confusion are rather minute details, while others concern important core operations that are commonly used. This document may grow as I start to use PyTorch more extensively for training or model implementation. Let's get started.


```python
import torch
import numpy as np
```

# Size Declaration

There appear to be two ways of specifying the size of a tensor. Using `torch.ones` as an example, let's consider the difference between 


```python
torch.ones(2, 3)
```




    tensor([[1., 1., 1.],
            [1., 1., 1.]])



and 


```python
torch.ones((2, 3))
```




    tensor([[1., 1., 1.],
            [1., 1., 1.]])



It confused me how the two yielded identical results. Indeed, we can even verify that the two tensors are identical via


```python
torch.equal(torch.ones(2, 3), torch.ones((2, 3)))
```




    True



I thought different behaviors would be expected if I passed in more dimensions, plus some additional arguments like `dtype`, but this was not true.


```python
torch.ones((1, 2, 3), dtype=torch.long)
```




    tensor([[[1, 1, 1],
             [1, 1, 1]]])




```python
torch.ones(1, 2, 3, dtype=torch.long)
```




    tensor([[[1, 1, 1],
             [1, 1, 1]]])



The conclusion of this analysis is that the two ways of specifying the size of a tensor are exactly identical. However, one note of caution is that NumPy is more opinionated than PyTorch and exclusively favors the tuple approach over the unpacked one.


```python
np.ones((2, 3))
```




    array([[1., 1., 1.],
           [1., 1., 1.]])




```python
np.ones(2, 3)
```


    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    <ipython-input-8-307396d1a1d4> in <module>
    ----> 1 np.ones(2, 3)


    ~/opt/anaconda3/envs/pytorch/lib/python3.7/site-packages/numpy/core/numeric.py in ones(shape, dtype, order)
        190 
        191     """
    --> 192     a = empty(shape, dtype, order)
        193     multiarray.copyto(a, 1, casting='unsafe')
        194     return a


    TypeError: Cannot interpret '3' as a data type


The conclusion of this analysis is that either approach is fine; it is perhaps a good idea to stick to one convention and stay consistent with that coding style throughout.

# Resize, Reshape

Resizing or reshaping a tensor is an incredibly important tensor operation that is used all the time. The interesting thing is that there seems to be many ways of achieving the same behavior. As someone who prefers a more opinionated guideline, this was rather confusing at first. However, here is what I have gathered while sifting through Stack Overflow and PyTorch discussion forums. 

Let's first start with a dummy random tensor. (Note that I could have done `torch.rand((2, 3))`, as per the conclusion from the section above.)


```python
m = torch.rand(2, 3); m
```




    tensor([[2.9573e-01, 9.5378e-01, 5.3594e-01],
            [7.4571e-01, 5.8377e-04, 4.6509e-01]])



## Reshape

The `.reshape()` operation returns a new tensor whose dimensions match those that have been passed into the function as arguments. For example, the snippet below shows how we can reshape `m` into a `[1, 6]` tensor.


```python
m.reshape(1, 6)
```




    tensor([[2.9573e-01, 9.5378e-01, 5.3594e-01, 7.4571e-01, 5.8377e-04, 4.6509e-01]])



One very important detail, however, is that this operation is not in-place. In other words, if we check the size of `m` again, you will realize that it is still a `[2, 3]` tensor, as was originally initialized. 


```python
m.size()
```




    torch.Size([2, 3])



To change `m` itself, we could do 

```python
m = m.reshape(1, 6)
```

## Resize

Or even better, we can use `.resize_()`, which is an in-place operation by design.


```python
m.resize_(1, 6)
```




    tensor([[2.9573e-01, 9.5378e-01, 5.3594e-01, 7.4571e-01, 5.8377e-04, 4.6509e-01]])



Notice that, unlike when we called `.reshape()`, `.resize_()` changes the tensor itself, in-place.


```python
m.size()
```




    torch.Size([1, 6])



In older versions of PyTorch, `.resize()` existed as a non in-place operator. However, in newer versions of PyTorch, this is no longer the case, and PyTorch will complain with an informative deprecation error message. Note that `.resize()` is not an in-place operator, meaning its behavior will largely be identical to that of `.reshape()`.


```python
m.resize(2, 3)
```

    /Users/jaketae/opt/anaconda3/envs/pytorch/lib/python3.7/site-packages/torch/tensor.py:358: UserWarning: non-inplace resize is deprecated
      warnings.warn("non-inplace resize is deprecated")





    tensor([[2.9573e-01, 9.5378e-01, 5.3594e-01],
            [7.4571e-01, 5.8377e-04, 4.6509e-01]])



## In-Place Operations

PyTorch keeps an internal convention when it comes to differentiating between in-place and copy operations. Namely, functions that end with a `_` are in-place operators. For example, one can add a number to a tensor in-place via `add_()`, as opposed to the normal `+`, which does not happen in-place.


```python
m + 1
```




    tensor([[1.2957, 1.9538, 1.5359, 1.7457, 1.0006, 1.4651]])



Observe that the addition is not reflected in `m`, indicating that no operations happened in-place.


```python
m
```




    tensor([[2.9573e-01, 9.5378e-01, 5.3594e-01, 7.4571e-01, 5.8377e-04, 4.6509e-01]])



`.add_()`, however, achieves the result without copying and creating a new tensor into memory.


```python
m.add_(1)
```




    tensor([[1.2957, 1.9538, 1.5359, 1.7457, 1.0006, 1.4651]])



## View

`.view()` is another common function that is used to resize tensors. It has been part of the PyTorch API for quite a long time before `.reshape()` was introduced. Without getting into too much technical detail, we can roughly understand view as being similar to `.reshape()` in that it is not an in-place operation. 

However, there are some notable differences. For example, this [Stack Overflow post](https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch) introduces an interesting example:


```python
z = torch.zeros(3, 2)
y = z.t()
y.view(6)
```


    ---------------------------------------------------------------------------
    
    RuntimeError                              Traceback (most recent call last)
    
    <ipython-input-18-436455a54c93> in <module>
          1 z = torch.zeros(3, 2)
          2 y = z.t()
    ----> 3 y.view(6)


    RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.


On the other hand, `.reshape()` does not run into this error.


```python
z = torch.zeros(3, 2)
y = z.t()
y.reshape(6)
```




    tensor([0., 0., 0., 0., 0., 0.])



The difference between the two functions is that, whereas `.view()` can only be used on contiguous tensors. [This SO thread](https://stackoverflow.com/questions/48915810/pytorch-contiguous) gives a nice explanation of what it means for tensors to be contiguous; the bottom line is that, some operations, such `.t()`, do not create a completely new tensor, but returns a tensor that shares the data with the original tensor while having different index locations for each element. These tensors do not exist contiguously in memory. This is why calling `.view()` after a transpose operation raises an error. `.reshape()`, on the other hand, does not have this contiguity requirement. 

This felt somewhat overly technical, and I doubt I will personally ever use `.view()` over `.reshape()`, but I thought it is an interesting detail to take note of nonetheless. 

# tensor v. Tensor

Another point of confusion for me was the fact that there appeared to be two different ways of initializing tensors: `torch.Tensor()` and `torch.tensor()`. Not only do the two functions look similar, they also practically do the same thing.


```python
torch.Tensor([1, 2, 3])
```




    tensor([1., 2., 3.])




```python
torch.tensor([1, 2, 3])
```




    tensor([1, 2, 3])



Upon more observation, however, I realized that there were some differences, the most notable of which was the `dtype`. `torch.Tensor()` seemed to be unable to infer the data type from the input given. 


```python
torch.Tensor([1, 2, 3]).dtype
```




    torch.float32



On the other hand, `torch.tensor()` was sable to infer the data type from the given input, which was a list of integers. 


```python
torch.tensor([1, 2, 3]).dtype
```




    torch.int64



Sure enough, `torch.Tensor()` is generally non-configurable, especially when it comes to data types.


```python
torch.Tensor([1, 2, 3], dtype=torch.float16)
```


    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    <ipython-input-24-5c47175b637e> in <module>
    ----> 1 torch.Tensor([1, 2, 3], dtype=torch.float16)


    TypeError: new() received an invalid combination of arguments - got (list, dtype=torch.dtype), but expected one of:
     * (*, torch.device device)
          didn't match because some of the keywords were incorrect: dtype
     * (torch.Storage storage)
     * (Tensor other)
     * (tuple of ints size, *, torch.device device)
     * (object data, *, torch.device device)



`torch.tensor()` can accept `dtype` as a valid argument. 


```python
torch.tensor([1, 2, 3], dtype=torch.float16)
```




    tensor([1., 2., 3.], dtype=torch.float16)



The conclusion of this analysis is clear: use `torch.tensor()` instead of `torch.Tensor()`. Indeed, [this SO post](https://stackoverflow.com/questions/51911749/what-is-the-difference-between-torch-tensor-and-torch-tensor) also confirms the fact that `torch.tensor()` should generally be used, as `torch.Tensor()` is more of a super class from which other classes inherit. As it is an abstract super class, using it directly does not seem to make much sense. 

# Size v. Shape

In PyTorch, there are two ways of checking the dimension of a tensor: `.size()` and `.shape`. Note that the former is a function call, whereas the later is a property. Despite this difference, they essentially achieve the same functionality. 


```python
m = torch.ones((2, 3)); m
```




    tensor([[1., 1., 1.],
            [1., 1., 1.]])




```python
print(m.shape)
print(m.size())
```

    torch.Size([2, 3])
    torch.Size([2, 3])


To access one of the `torch.Size()` elements, we need appropriate indexing. In the case of `.shape`, it suffices to consider the size as a list, meaning that square bracket syntax can be used.


```python
m.shape[0]
```




    2



In the case of `torch.size()`, indices can directly be passed into as an argument to index individual elements in the size tensor.


```python
m.size(0)
```




    2



# Dimension

These past few days, I've spent a fair amount of time using PyTorch for basic modeling. One of the main takeaways from that experience is that an intuition on dimensionality and tensor operations in general is a huge plus. This gets especially important for things like batching. 

## (n,) v. (1, n)

One very basic thing I learned--admittedly perhaps too belatedly--is the difference between `(1, n)` and `(n,)` as dimensions. Here is a concrete example.


```python
a = torch.randn(3,); a
```




    tensor([-0.6378, -0.2795, -2.9941])



This creates a one-dimensional tensor, which is effectively a list. We can check the dimensions of this tensor by calling `.ndim`, which is very similar to how NumPy works.


```python
a.ndim
```




    1



On the other hand, specifying the size as `(1, 3)` results in a two-dimensional tensor. 


```python
b = torch.randn(1, 3); b
```




    tensor([[ 0.9485, -0.3227, -0.6983]])



The simple, barely passing answer to the question of why `b` is two-dimension would be that it has double layered brackets. More exactly speaking, having an additional layer means that it is capable of storing another tensor within it; hence, `b` is living in a dimension that is one above that of `a`. 


```python
b.ndim
```




    2



## (Un)Squeeze

As mentioned earlier, batch dimension is something that becomes very important later on. Some PyTorch layers, most notably RNNs, even have an argument `batch_first`, which accepts a boolean value. If `True`, PyTorch expects the first dimension of the input to be the batch dimension. If `False`, which is the case by default, PyTorch assumes that the first dimension would be the sequence length dimension. 

A common operation that is used when dealing with inputs is `.squeeze()`, or its inverse, `.unsqueeze()`. Before explaining what these operations perform, let's just take a look at an example. Let's start with `a`, the random tensor of size `(3,)` initialized above.


```python
a
```




    tensor([-0.6378, -0.2795, -2.9941])



If we apply `.unsqueeze(0)` to `a`, we essentially add a new dimension to the 0-th position of `a`'s shape. 


```python
a.unsqueeze(0)
```




    tensor([[-0.6378, -0.2795, -2.9941]])



As you can see, now there is an additional batch dimension, thus resulting in a tensor whose shape is `(1, 3)` as opposed to the original `(3,)`. 

However, of course this operation is not performed in-place, meaning that `a` will still remain unchanged. There are in-place versions of both `.squeeze()` and `.unsqueeze()` though, and that is simply adding a `_` to the end of the function. For example, 


```python
a.unsqueeze_(0); print(a)
```

    tensor([[-0.6378, -0.2795, -2.9941]])


Equivalently, calling `.squeeze(k)` will remove the `k`th dimension of the tensor. By default, `k` is 0. 


```python
a.squeeze()
```




    tensor([-0.6378, -0.2795, -2.9941])



Squeezing and unsqueezing can get handy when dealing with single images, or just single inputs in general. 

# Concat

Concatenation and stacking are very commonly used in deep learning. Yet they are also operations that I often had trouble imagining in my head, largely because concatenation can happen along many axes or dimensions. In this section, let's solidify our understanding of what concatenation really achieves with some dummy examples.


```python
m1 = (torch.rand(2, 3, 4) * 10).int()
m2 = (torch.rand(2, 3, 4) * 10).int()
print(m1)
print(m2)
```

    tensor([[[6, 4, 6, 0],
             [0, 6, 1, 9],
             [2, 6, 0, 3]],
    
            [[5, 0, 2, 7],
             [9, 5, 7, 0],
             [6, 2, 1, 0]]], dtype=torch.int32)
    tensor([[[4, 3, 1, 1],
             [9, 2, 3, 0],
             [7, 3, 2, 5]],
    
            [[0, 4, 0, 6],
             [8, 8, 7, 8],
             [1, 5, 7, 2]]], dtype=torch.int32)


With a basic example, we can quickly verify that each tensor is a three-dimensional tensor whose individual elements are two-dimensional tensors of shape `(3, 4)`. 


```python
m1[0]
```




    tensor([[6, 4, 6, 0],
            [0, 6, 1, 9],
            [2, 6, 0, 3]], dtype=torch.int32)



Now, let's perform the first concatenation along the 0-th dimension, or the batch dimension. 


```python
cat0 = torch.cat((m1, m2), 0); cat0
```




    tensor([[[6, 4, 6, 0],
             [0, 6, 1, 9],
             [2, 6, 0, 3]],
    
            [[5, 0, 2, 7],
             [9, 5, 7, 0],
             [6, 2, 1, 0]],
    
            [[4, 3, 1, 1],
             [9, 2, 3, 0],
             [7, 3, 2, 5]],
    
            [[0, 4, 0, 6],
             [8, 8, 7, 8],
             [1, 5, 7, 2]]], dtype=torch.int32)



We can verify that the concatenation occurred along the 0-th dimension by checking the shape of the resulting tensor.


```python
cat0.shape
```




    torch.Size([4, 3, 4])



Since we concatenated two tensors each of shape `(2, 3, 4)`, we would expect the resulting tensor to have the shape of `(2 + 2, 3, 4) == (4, 3, 4)`, which is indeed what we got. More generally speaking, we can think that concatenation effectively brought the two elements of each tensor together to form a larger tensor of four elements. 

I found concatenation along the first and second dimensions to be more difficult to imagine right away. The trick is to mentally draw a connection between the dimension of concatenation and the location of the opening and closing brackets that we should focus on. In the case of the example above, the opening and closing brackets were the outer most ones. In the example below in which we concatenate along the first dimension, the brackets are those that form the boundary of the inner two-dimensional 3-by-4 tensor. Let's take a look.


```python
cat1 = torch.cat((m1, m2), 1); cat1
```




    tensor([[[6, 4, 6, 0],
             [0, 6, 1, 9],
             [2, 6, 0, 3],
             [4, 3, 1, 1],
             [9, 2, 3, 0],
             [7, 3, 2, 5]],
    
            [[5, 0, 2, 7],
             [9, 5, 7, 0],
             [6, 2, 1, 0],
             [0, 4, 0, 6],
             [8, 8, 7, 8],
             [1, 5, 7, 2]]], dtype=torch.int32)



Notice that the rows of `m2` were essentially appended to those of `m1`, thus resulting in a tensor whose shape is `(2, 6, 4)`. 


```python
cat1.shape
```




    torch.Size([2, 6, 4])



For the sake of completeness, let's also take a look at the very last case, where we concatenate along the last dimension. Here, the brackets of focus are the innermost ones that form the individual one-dimensional rows of each tensor. Therefore, we end up with a "long" tensor whose one-dimensional rows have a total of 8 elements as opposed to the original 4.


```python
cat2 = torch.cat((m1, m2), 2); cat2
```




    tensor([[[6, 4, 6, 0, 4, 3, 1, 1],
             [0, 6, 1, 9, 9, 2, 3, 0],
             [2, 6, 0, 3, 7, 3, 2, 5]],
    
            [[5, 0, 2, 7, 0, 4, 0, 6],
             [9, 5, 7, 0, 8, 8, 7, 8],
             [6, 2, 1, 0, 1, 5, 7, 2]]], dtype=torch.int32)



# Conclusion

In this post, we took a look at some useful tensor manipulation operations and techniques. Although I do have some experience using Keras and TensorFlow, I never felt confident in my ability to deal with tensors, as that felt more low-level. PyTorch, on the other hand, provides a nice combination of high-level and low-level features. Tensor operation is definitely more on the low-level side, but I like this part of PyTorch because it forces me to think more about things like input and the model architecture.

I will be posting a series of PyTorch notebooks in the coming days. I hope you've enjoyed this post, and stay tuned for more!
