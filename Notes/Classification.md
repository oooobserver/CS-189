### introduce

The task of classification differs from regression in that we are now  assign a d-dimensional data point one of a discrete number of classes, instead of assigning it a continuous value. In other words, in classification according to the input d-dimensional data point, we judge it belong to which classes.

There are two main types of classification models:

1. generative models
in generative method, x and y are both RV, we want to maximizes the joint probability:
$$\hat{y}=arg\underset{k}{} \ max P(x,y=k)$$
$P(y=k)=p(class = k)$ is A prior probability distribution over all classes

$$\hat{y}=arg\underset{k}{} \ max P(x,y=k)=arg\underset{k}{} \ maxP(x|y=k)p(y=k)=arg\underset{k}{} \ maxP(y=k|x)$$

Maximizing the posterior will get the class has the highest posterior probability and **decision boundaries** in between classes where the posterior probability of two classes are equal.

* discriminative models

in this method, we don't care about P(x), we directly learn $P(y=k|x)$ which is decision boundary.

### Bayes’ Decision Rule

in classification we want to minimize the risk of our model:
$$R(h) = E[l(h(x),y)]$$

in context of classification, the loss function can take many forms, but the simplest is the standard step function:

$$l(h(x),y) = \left\{ 
\begin{array}{c}
0 \ h(x) = y\\ 
1 \ h(x) \neq y\\ 
\end{array}
\right. $$

The **Bayes’ classifier** h  will minimize the risk:
$$h(x)=arg \ min_j \sum L(j, k)P(Y = k|x)$$
the Bayes’ classifier will pick the class that minimizes the expected loss for the given x. In the special case where the loss function is the standard step function.






### logistic regression

given $D = {(x_1,y_1),(x_2,y_2)....(x_n,y_n)}$ , each $y_i\in{0,1}$ 

in this classification , our output $f_\theta(x)=\theta^Tx$ is a real number, so we should turn the real number into probability, we often use logistic function also call as sigmoid function. **it can turn output into a number between 0 and 1, and this represent the probability of class 1.**

sigmoid function:
$$f(z)=\frac{1}{1+e^{-z}}$$
![[Figure_1.png]]

unlike linear regression, there is no analytical solution, instead we must rely on iterative optimization - specifically we refer to gradient based optimization.


#### the MLE for $\theta$

$$\begin{equation} \begin{aligned} arg \ max_\theta P(y|x) &=\sum(p(y=1|x))^{y_i}(p(y=0|x))^{1-y_i}  \\  &=\sum y_i\theta^Tx_i - log[exp(\theta^Tx_i)+1] \end{aligned} \end{equation}$$

$$\begin{equation} \begin{aligned}\bigtriangledown_\theta &=\sum y_ix_i-sigmoid(\theta^Tx_i)x_i \\ &=X^T(y-s_\theta)    \end{aligned} \end{equation} $$

there is no analytical solution:

in order to find the solution, we set the gradient equal to 0. so the $(y-s_\theta)$ is in the left null space of X. let's assume X is full rank, so the result is $s_\theta$ equal to y along with every entry. and we know the y is either 0 or 1. so any $s_\theta$ is 0 or 1. we can see from the image, when in this case the $||\theta||\rightarrow \infty$. so there is no analytical solution.

we avoid this issue by adding regularization just like [[MAP]]. and use gradient descent to get the  solution. you can see this on [[gradient descent]] 



### Linear discriminant analysis(LDA)

LDA is a generative method, here is a example:

suppose we have prior probability distribution $p(y=0)=p(y=1)=0.5$ and the class conditionals as MVG: $p(x|y=0)=N(x:\mu_0,\sigma),p(x|y=1)=N(x:\mu_1,\sigma)$   and we assume the two classes conditionals variance is  same.

$$\begin{equation} \begin{aligned} p(y=c|x) &= \frac{p(x|y=c)p(y=c)}{\sum p(x|y=c)p(y=c)}  \\  &= \frac{N(\mu_i,\sigma)}{\sum N(\mu_i,\sigma)} \end{aligned} \end{equation}$$

we predict 1 when $log(N(x:\mu_1,\sigma))-log(N(x:\mu_0,\sigma))>0$ 
$$\begin{equation} \begin{aligned} log(N(x:\mu_1,\sigma))-log(N(x:\mu_0,\sigma))&>0  \\  (x-\mu_1)^T\sigma(x-\mu_1)-(x-\mu_0)^T\sigma(x-\mu_0)&> 0 \\ w^Tx+b&>0 \end{aligned} \end{equation}$$

we eliminate the quadratic form and only left linear relation, so when you meet this , you can get a linear rule to determine that which classes is data point  belong.


### two method versus

* generative models

in generative models, because we have conditional:$p_\theta(x)$ we can know some data input are unlikely. this can detecting outliers or data anomalies.

sometimes, they train better when the training data is limited 

* discriminative models

this method is standard way

they train better - especially neutral network

in many cases, the input are complicated, it's hard to get conditional.

### classifiers

pedantically, a classifier is a combination of a model and **decision rule:**
* the model can be  probabilistic like: logistic regression
* the rule can be: predict the the class has the highest probability

for logistic regression, we choose class1 if $p(y=1|x)>0.5$ , and according to the image, we can just see the output $\theta^Tx$ if greater than 0, and this is the sign of the model. and the $\theta^Tx=0$ is referred as the decision  boundary.














### multiple classification

many problems have  K possible classes, like digit recognition, K = 10.

#### with logistic regression

we assume $\theta\in R^{d\times k}$ , so the output is K real numbers. as before we  are expect to transform the output into probability. we are going to make every number falls between 0 and 1, and sum to 1.

The most commonly used choice is $exp(z)$, which is bijective. one way is **softmax** function:
$$softmax(f_\theta(x))_c=\frac{exp[f_\theta(x)_c]}{\sum exp[f_\theta(x)_i]}=p(k=c)$$


#### one-hot vector encoding

**However, this approach gives an “ordering” to the classes, even if the classes themselves have no natural ordering. This is clearly a problem.**

For example, in fruit classification, suppose 1 is used to represent “peach,” 2 is used to represent “banana,” and 3 is used to represent “apple.” As a result, if we have an image that looks like some cross between an apple and a peach, we may simply end up classifying it as a banana.

if the i'th observation has class k, instead of using the representation yi = k, we can use the representation $y_i = e_k$ , the k’th canonical basis vector. Now there is no relative ordering in the representations of the classes.

When we have multiple classes, each yi is a K-dimensional one-hot vector, so for LS-SVM, we instead have a K × (d + 1) weight matrix to optimize over:

$$arg \ min_\theta=||Y_i-wx_i||_2^2+a||w||_2^2$$

To classify an arbitrary input x, we compute Wx and see which component k is the largest:

$$\hat{y}=max_k \ w_k^Tx$$
