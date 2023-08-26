
### Motivation

Fully connected(fc) network is very useful in some problem, and can applied in small image problem. But in term of big image like (W * H * 3), we flattened it and assume this as input, and the first layer is (W * H * 3 * num), a prohibitively large number of weights (in the millions)!

**Convolutional neural networks (CNNs, or ConvNets)** are a different neural network architecture that significantly reduces the number of weights, and in turn reduces variance. It introduce two new types of layers — **convolutional and pooling layers**. Let’s look at these two layers in detail.

### Convolutional Layers

A **convolutional layer** takes a W × H × D dimensional input I and convolves it with a w × h × D **dimensional filter (or kernel) G**. For example, we have W = H = 7,w = h = 3, stride = 1. the the out put is $n_0=5\times5=25$ . **Stride** is how much we shift our  filter at one line.**Padding** is to pad  the input with certain number of pixels on the both side.**Why we do that ?**  because after convolution the output's width will smaller than input like cut the slide, so we pad can make it like input.

For a $K\times K$ filter, we have $[H',W']=1+([H,W]+2\times pad -K)/stride$  
![[convolution network.png]]





**What exactly is convolution useful for, and why do we use it in the context of image classification?**

In simple terms, convolutions help us extract features called local. For example, consider a simple horizontal edge detector filter [1,-1]. This filter will produce large negative values for inputs in which the left pixel is bright and the right pixel is dark; conversely, it will produce large positive values for inputs in which the left pixel is dark and the right pixel is bright.

**Why can we use one filter through everywhere in one image?**

because there are repeated patterns in images — ie. a filter that can detect some kind of pattern in one area of the image can be used elsewhere in the image to detect the same pattern.

And while training, we will place limits on the feature we can learn that's called **inductive biases**: knowledge we build into the model.

In practice, we can apply several different filters to the image to detect different patterns in the input image. For example, we can use a filter that detects horizontal edges, one that detects vertical edges, and another that detects diagonal edges all at once. Suppose we have K filters, and the output will be $W'\times H'\times K$ .


### Pooling Layers

**Pooling layers** directly reduce the number of neurons in neural networks. The essence  is sliding a fixed window across a layer and choosing one value that effectively “represents” all of the units captured by the window. There are two common implementations of pooling. In max-pooling, we choose the max unit to represent all the units in the window, while in average-pooling, the representative value is the average of all the units in the window.

In practice, we stride pooling layers across the image with the stride equal to **the size of the pooling layer.** 


If we just stacking convolutions on top of each  other, there will be a huge issue. Because the convolution is linearity, so there just linear operation, we must introduce some **nonlinearities**. Just like before, we can use Relu or something like this.


### Math process

consider processing $a^{(l)}[H,W,C]$ into $z^{(l+1)}[H',W',C']$ using a filter $W[K,K,C',C]$ , $C$ is the input channel and the $C'$ is the output channel.

Let's first calculate the forward direction formula: 
$$z_c[i,j]=\sum_{a=0}^{K-1}\sum_{b=0}^{K-1}\sum_{c\in R,G,B}w_c[a,b]a_c[i+a,j+b]$$

 Now , let's do back forward:
 $$\begin{equation} \begin{aligned} 
 \frac{\partial l}{\partial a[i,j]} &= \sum_{a=0}^{K-1}\sum_{b=0}^{K-1}\frac{\partial l}{\partial z[i-a,j-b]}\times w[a,b]  &\text{(1)}
 \\ \frac{\partial l}{\partial w[a,b]} &= \sum_{i=0}^{H'-1}\sum_{j=0}^{W'-1}\frac{\partial l}{\partial z[i,j]}\times a[i+a,j+b] &\text{(2)}
  \end{aligned} \end{equation}$$

In this back forward process like former neural network backpropagation [[Neural networks#Backpropagation]] . We use chain rule to calculate gradient. Note that in backpropagation, **we assume we already know the partial derivative  of loss to Z.** we first calculate the gradient of input : 

![[convolution network1.png]]

The above image why first formula work, let's assume our $a[i,j]=7$ and filter is $[2,2]$ , the green and red region represent the entry 7 in, also means the entry influence  the output not only one value. Four region represent four output value, so is $z[i-a,j-b]$ . And of course, w don't need to change, it stay same. 

The gradient of $w[a,b]$ is out  main goal. We first think about what part of the output is influence by $w[a,b]$ , and come out that all entry in output is influenced by w. So the formula is the whole output tensor.

There is a salient point you may have not be recognize:**the two formula can also see as convolutional.** The first filter is $w[a,b]$ and the second filter is $a[i+a,j+b]$ . 
