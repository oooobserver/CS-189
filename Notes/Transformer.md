
### Attention

#### set up

Originally formulated for sequential data

E.g. translate or captioning a image. 
The features are not always sequential like image.

Let's take a example about captioning a image. To do this, the model will output one step a time. And the model need to know what it has generate otherwise there will be duplicate.
Historically, people used *recurrence*. 

#### Motivation

While generating different part, model focus on different part of image. But we don't know the order to focus or how large part should it focus, so we will use conv net first to generate a vector of image.                       

For example, our image can caption as: a bird  fly over lake, and we have first word 'a', how does it know the next word is 'bird'? There is a **"key-value-query" system**:

$$\begin{equation} \begin{aligned}
q_t &=q(\text{info from prev step}) \\
k_l &= k(c_l) \\
v_l &= v(c_l)
\end{aligned} \end{equation}$$

The query function for this step might be a subject. The key is the type of the $c_l$ , might be subject or background, and the value is the c represent like a lake or bird.

Let's dive into the detail:
$$\begin{vmatrix} k_1 \\. \\k_L\end{vmatrix}
\rightarrow
\begin{vmatrix} k_1^Tq2 \\. \\k_L^Tq2\end{vmatrix}
\rightarrow^{soft max}
\begin{vmatrix}a_1 \\. \\a_L\end{vmatrix}
\rightarrow a_2 = \sum a_iv_i
$$
All these functions can be learned.
 

#### self-attention

The goal of self-attention is to handle the sequential data as input, can be seen as a neural network layer.

Detail: use **scale dot product attention** which is former product divide by $\sqrt{d}$ . If query and key are more similar their dot product are more big. In practice, we don't add one by one, we usually use matrix:

$$\begin{vmatrix} q_{11} &  ... &  q_{1k}\\
... \\q_{n1} & ... &  q_{nk}\end{vmatrix} \times 
\begin{vmatrix} k_{11} &  ... &  k_{1k}\\
... \\k_{m1} & ... &  k_{mk}\end{vmatrix}^T\rightarrow R^{n\times m}
\times\begin{vmatrix} v_{11} &  ... &  v_{1v}\\
... \\v_{m1} & ... &  v_{mv}\end{vmatrix}\rightarrow Z $$

**Why divide $\sqrt{d}$ ?**

Because dot product could make a huge result, and cause a value extreme close to 1 , others close to 0, this will cause gradient small which harder to train.

Because the output of a layer is just linear combination of values, so like before we should add some nonlinearity, which is called **feedforward layer**, usually **GELU**.

### Transformer

Most competitive neural sequence transduction models have an **encoder-decoder** structure . Here, the encoder maps an input sequence of symbol representations  $(x_1​,...,x_n​)$ to a sequence of continuous representations $z=(z_1​,...,z_n​)$. Given z, the decoder then generates an output sequence $(y_1​,...,y_m​)$ of symbols one element at a time. At each step the model is auto-regressive: **consuming the previously generated symbols as additional input** when generating the next.
![[en-de coding.png]]
**Embedding** is a technique that transforms each word (or token) in a sentence into a numerical vector.

**N** represent the number of layer, and each layer has multiple sublayers

#### layer norm and batch norm

Suppose we have a matrix that each row is a data point and have B(batch) rows:

$$\begin{vmatrix} x_{11} &  x_{12} &  x_{13}\\. .. \\x_{B1} &  x_{B2} &  x_{B3}\end{vmatrix}$$

**Batch norm** is to make every feature(column) normalization: mean to 0, variation to 1.

**Layer norm** is to make every datapoint(row) normalization

Layer norm is more common because in practice each datapoint may not be same length. 


#### Encoding

At high level, encoder take original sequence data and  like above sue interleave self-attention layer to output a sequence, and we do this because we hope after this the data will be easier to work with.
![[transformer.png]]

The z is going to feed in simple model like regression or classification that will be more effective than original data.

**Note that in the above diagram the key, query, value are the same, so it called self-attention**


##### Details

* position encoding

![[position encoding.png]]


After first feed layer, a position encoding is add to each $h_t$. Why we do this? Because without this the model can's distinguish between different permutation of same sequence. The $d$ in the image is the dimension of the $h_t$.  


* multi-head attention

In practice, if we meet this situation that two keys are equivalent possible, we don't know how to handle. If merge the two values, this will loss two information. To fix this issue, we use  multi-head attention, add more layers on same step each layer has different key function and query function:

![[multi-head attention.png]]

This is projection each Q,K,V into smaller dimension(d/h), h is the number of heads and concatenate each output.   

#### Decoding

In this, model's job is to generate the future but we don't have future so to generate $y_t$ we use all of our previous y , and add the output $y_t$ to the input to next layer to output $y_{t+1}$.


![[Decoding.png]]


##### Details

* masked attention

But in training, we have the future so in order to train the model well, instead of let the output is the copy of the input, we have to let it **not look at the future**. One way do that is assign negative infinity to the input steps lager than current time step:

![[masked attention.png]]


#### Cross attention

So how do we connect encoder and decoder? Like the diagram above, in cross attention layer we use encoder's output as key and value, use decoder's output as query.
