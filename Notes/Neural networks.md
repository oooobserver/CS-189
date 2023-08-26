
### Motivation

Machine Learning is about predicting y from x, but what is x? x is my whole input d-dimensional vector. we should make it better to study by ourselves using: PCA, Kernelization... To make a better model, we are expect to pick good features. This is annoying and not "machine" learning. So, here comes **deep learning**, it can learn the featurization.


### Introduction

Deep learning is machine learning with multiple layers of learned representations. The function that represents the transformation  from input to output is a **deep neutral network**. 

The parameters for every layer are usually(not always) trained with respect to the overall objective. This is sometimes referred to **end-to-end learning**. 

Recap from Logistic Regression [[Classification#logistic regression]] , we can see this as a **single layer neural network**. But the actual layer in neural network non linear, because there may exit nonlinear features.

One of main reason make neural network great is it can represent complex non linear function. How do we do this, the most simple way is add **nonlinearities** after every linear layer, aka **activation functions**. 

These functions basically always operate every element in vector of the linear layer output.
There are some examples: $tanh(z),sigmoid(z),ReLU(z)=max(0,z)$ .

What is the whole function is?

$\theta$ represents all our parameters: $[w_1,b_1,....w_l,b_l,w_{final},b_{final}]$. If our neural network has $\theta$ and L(hidden layers) then it represents the function $$f_\theta(x)=softmax(A^{final}(\sigma( A^{L}...(\sigma A^1(x))...)))$$
$\sigma$ is activation function, $A^i =w^iv+b^i$ is ith linear layer.

![[Neural network 1.png]]

Computation flows left-to-right. The circles represent nodes, a.k.a. units or **neurons**.Observe that the nodes are organized into layers. The first layer is called the **input layer**, the last layer is called the **output layer**, and any other layers are referred to as **hidden layers**. The number and sizes of the hidden layers are hyperparameters to be chosen by the network designer.

Layers that have every node in that layer is connected to every node in the previous layer are described as **fully connected**. Each node computes a weighted sum of its inputs, with these connection strengths being the weights, and then applies a nonlinear function which is variously referred to as the **activation function or the nonlinearity**.











### How to train

Neural network can see as successive non linear transformations of input x that hopefully can result in features that can use in final linear model. How do we make the learned features good for learning. Usually, we utilize **end-to-end learning** : training the whole network on the overall objective **(the negative log likelihood loss)**

we want to update our $\theta$ use gradient descent, because we train the whole network, we need to calculate $[\bigtriangledown w_1,\bigtriangledown b_1....\bigtriangledown w_{final},\bigtriangledown b_{final}]$, how do we calculate these, there are two methods: **finite difference , backpropagation**.


### Finite difference

For any sufficiently smooth function f which operate on a vector x, the partial derivative $\frac{\partial f}{\partial x}$  is approximate by:
$$\frac{\partial f}{\partial x}\approx \frac{f(x+\epsilon e^i)-f(x-\epsilon e^i)}{2\epsilon}$$

This method is slow and need much calculate, we usually use this to check is we calculate right.


### Backpropagation

Backpropagation works backward the neural network, which allow for:
* reusing gradient value.
* computing matrix-vector product than matrix-matrix product, since loss is scalar.

let's work for two hidden layers as example, so $\theta=[w_1,b_1,w_2,b_2,w_{final},b_{final}]$  .

![[Neural networks.png]]

$$p(y|x)=[\frac{exp z}{\sum exp z}]_y$$

**note that numerator is vector and the denominator is scalar , the probability is the y index of the vector**.

$$\begin{equation} \begin{aligned} 
log\  p(y|x) &= [z]_y- log \sum exp z \\ 
l&=log \sum exp z-[z]_y
\end{aligned} \end{equation}$$
let's get gradient, first take the last layer's:

Because $z=w_{final}a_2+b_{final}$ , we can use chain rule to get w, b.

$$\begin{equation} \begin{aligned} 
\bigtriangledown_{z}&=\frac{exp z}{\sum expz} - e_{y_i} \\
\bigtriangledown_{w_{final}}&=\bigtriangledown_{z} \times a_2^T \\
\bigtriangledown_{b_{final}}&=\bigtriangledown_{z}  \\
\bigtriangledown_{a_2}&=w_{final}^T \times \bigtriangledown_{z}   \\
\end{aligned} \end{equation}$$
now let's look at $w_2,b_2$:
remember $a_2=\sigma(z_2), z_2=w_2a_1+b_2$

$$\begin{equation} \begin{aligned} 
\bigtriangledown_{z_2}&=\bigtriangledown_{a_2} \times \begin{vmatrix}  \sigma & & \\ & \sigma &\\& &\sigma \end{vmatrix}\\
\bigtriangledown_{w_{2}}&=\bigtriangledown_{z_2} \times a_1^T \\
\bigtriangledown_{b_{2}}&=\bigtriangledown_{z_2}  \\
\bigtriangledown_{a_1}&=w_{2}^T \times \bigtriangledown_{z_2}   \\
\end{aligned} \end{equation}$$

Now you can see this pattern, go backward can let us reuse gradient that already computed.



