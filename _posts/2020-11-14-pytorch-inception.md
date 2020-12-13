---
title: InceptionNet in PyTorch
mathjax: true
toc: true
categories:
  - study
tags:
  - deep_learning
  - pytorch
---

In today's post, we'll take a look at the Inception model, otherwise known as GoogLeNet. I've actually written the code for this notebook in October 😱 but was only able to upload it today due to other PyTorch projects I've been working on these past few weeks (if you're curious, you can check out my projects [here](https://github.com/jaketae/pytorch-malware-detection) and [here](https://github.com/jaketae/bert-blog-tagger)). I decided to take a brief break and come back to this blog, so here goes another PyTorch model implementation blog post. Let's jump right into it!

First, we import PyTorch and other submodules we will need for this tutorial.


```python
import torch
from torch import nn
import torch.nn.functional as F
```

Because Inception is a rather big model, we need to create sub blocks that will allow us to take a more modular approach to writing code. This way, we can easily reduce duplicate code and take a bottom-up approach to model design.

The `ConvBlock` module is a simple convolutional layer followed by batch normalization. We also apply a ReLU activation after the batchnorm. 


```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, **kwargs)
        self.bn = nn.BatchNorm2d(out_chanels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
```

Next, we define the inception block. This is where all the fun stuff happens. The motivating idea behind InceptionNet is that we create multiple convolutional branches, each with different kernel (also referred to as filter) sizes. The standard, go-to kernel size is three-by-three, but we never know if a five-by-five might be better or worse. Instead of engaging in time-consuming hyperparameter tuning, we let the model decide what the optimal kernel size is. Specifically, we throw the model three options: one-by-one, three-by-three, and five-by-five kernels, and we let the model figure out how to weigh and process information from these kernels. 

In the `InceptionBlock` below, you will see that there are indeed various branches, and that the output from these branches are concatenated to produce a final output in the `forward()` function.


```python
class InceptionBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_1x1,
        red_3x3,
        out_3x3,
        red_5x5,
        out_5x5,
        out_pool,
    ):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1, padding=0),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )
    
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)
```

Researchers who conceived the InceptionNet architecture decided to add auxiliary classifiers to intermediary layers of the model to ensure that the model actually learns something useful. This was included in InceptionV1; as far as I'm aware, future versions of InceptionNet do not include auxiliary classifiers. Nonetheless, I've added it here, just for the fun of it. 


```python
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

Now we finally have all the ingredients needed to flesh out the entire model. This is going to a huge model, but the code isn't too long because we've abstracted out many of the building blocks of the model as `ConvBlock` or `InceptionBlock`. I won't get into the details here, as the number of parameters are simply from the original paper.


```python
class InceptionV1(nn.Module):
    def __init__(self, aux_logits=True, num_classes=1000):
        super(InceptionV1, self).__init__()
        self.aux_logits = aux_logits
        self.conv1 = ConvBlock(
            in_channels=3, 
            out_chanels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)
        
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        x = self.inception4a(x)
        
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        
        x = self.inception4e(x)
        x = self.maxpool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        if self.aux_logits and self.training:
            return aux1, aux2, x
        return x
```

As you can see, there are auxiliary classifiers here and there. If the model is training, we get three outputs in total: `aux1`, `aux2`, and `x`. When the model is in `eval()`, however, we only get `x`, as that's all we need as the final logits to be passed through a softmax function.

Let's see the gigantic beauty of this model.


```python
model = InceptionV1()
print(model)
```

    InceptionV1(
      (conv1): ConvBlock(
        (conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv2): ConvBlock(
        (conv): Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (inception3a): InceptionBlock(
        (branch1): ConvBlock(
          (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
          (1): ConvBlock(
            (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception3b): InceptionBlock(
        (branch1): ConvBlock(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(32, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
          (1): ConvBlock(
            (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4a): InceptionBlock(
        (branch1): ConvBlock(
          (conv): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(480, 96, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (bn): BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(16, 48, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
          (1): ConvBlock(
            (conv): Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4b): InceptionBlock(
        (branch1): ConvBlock(
          (conv): Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (bn): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
          (1): ConvBlock(
            (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4c): InceptionBlock(
        (branch1): ConvBlock(
          (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
          (1): ConvBlock(
            (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4d): InceptionBlock(
        (branch1): ConvBlock(
          (conv): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(512, 144, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (bn): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
          (1): ConvBlock(
            (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception4e): InceptionBlock(
        (branch1): ConvBlock(
          (conv): Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (bn): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(528, 32, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
          (1): ConvBlock(
            (conv): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception5a): InceptionBlock(
        (branch1): ConvBlock(
          (conv): Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (bn): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
          (1): ConvBlock(
            (conv): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (inception5b): InceptionBlock(
        (branch1): ConvBlock(
          (conv): Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (branch2): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(832, 192, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch3): Sequential(
          (0): ConvBlock(
            (conv): Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ConvBlock(
            (conv): Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (branch4): Sequential(
          (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
          (1): ConvBlock(
            (conv): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
      (dropout): Dropout(p=0.4, inplace=False)
      (fc): Linear(in_features=1024, out_features=1000, bias=True)
      (aux1): InceptionAux(
        (dropout): Dropout(p=0.7, inplace=False)
        (pool): AvgPool2d(kernel_size=5, stride=3, padding=0)
        (conv): ConvBlock(
          (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (fc1): Linear(in_features=2048, out_features=1024, bias=True)
        (fc2): Linear(in_features=1024, out_features=1000, bias=True)
      )
      (aux2): InceptionAux(
        (dropout): Dropout(p=0.7, inplace=False)
        (pool): AvgPool2d(kernel_size=5, stride=3, padding=0)
        (conv): ConvBlock(
          (conv): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (fc1): Linear(in_features=2048, out_features=1024, bias=True)
        (fc2): Linear(in_features=1024, out_features=1000, bias=True)
      )
    )


Great. To be honest, I don't think the output of the print statement is that helpful; all we know is that the model is huge, and that there is a lot of room for error. So let's conduct a quick sanity check with a dummy input to see if the model works properly.


```python
test_input = torch.randn(2, 3, 224, 224)
aux1, aux2, output = model(test_input)
print(output.shape)
```

    torch.Size([2, 1000])


Great! We've passed to the model a batch containing two RGB images of size 224-by-224, which is the standard input assumed by the InceptionNet model. We get in return a tensor of shape `(2, 1000)`, which means we got two predictions, as expected.

I'm not going to train this model on my GPU-less MacBook, and if you want to use InceptionNet, there are plenty of places to find pretrained models ready to be used right out of the box. However, I still think implementing this model helped me gain a finer grasp of PyTorch. I can say this with full confidence because a full month has passed since I coded out this Jupyter notebook, and I feel a lot more confident in PyTorch than I used to before. 

I hope you've enjoyed reading this blog post. Catch you up in the next one (where I'll probably post another old notebook that's been sitting on my computer for a month).
