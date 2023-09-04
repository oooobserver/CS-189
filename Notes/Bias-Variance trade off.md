
### set up

Recall from our previous discussion on supervised learning, that for a fixed input **x** the corresponding  Y is a noisy measurement of the true underlying response f(x):
$$Y=f(x) + Z$$
Where **Z** represent noise, is a zero-mean random variable, and is typically represented as a Gaussian distribution. And the **f(x)** is the true model behind the dataset. We previously mentioned **MLE** and **MAP** as two techniques that try to find of reasonable approximation to f(.) by solving a probabilistic objective. One question that naturally arises is: **how exactly can we measure the effectiveness of a hypothesis model?** So we would like to form a theoretical metric that can exactly measure the effectiveness of a hypothesis function. Keep in mind that this is only a theoretical metric that **cannot be measured in real life**, but it can be approximated via empirical experiments — more on this later.

We have dataset $D{(x_i,y_i)}$ , in this context, our $x_i$ is random sample from $X$ , so we can say that D is a RV, also Y is RV too. To  test our model, we have some test datapoint, and we treat these $x$ as fixed value, so  the expected squared error is:
$$\mathcal{E}(x;h)=E[(h(x;D)-Y)^2]$$


### Bias-Variance Decomposition

Let's take a decomposition about our expected squared error:

**Note:**
$$\begin{equation} \begin{aligned} E(Z) &= 0 \\ 
E(Y)&=E(f(x)+Z)=f(x)  \\ 
Var(Y)&=Var(f(x)+Z)=var(Z) \end{aligned} \end{equation}$$
These are prior equation we can conclude from the condition, and for any RV, we have:
$$E(X^2)=Var(x)+E^2(X)$$

Now, let's do this decomposition (for short I'll use h replace $h(x;D)$): 
$$\begin{equation} \begin{aligned} E[(h-y)^2] &= E[h^2-2hy+y^2]\\ 
&= E(h^2) - 2E(h)E(y) + E(y^2)  \\ 
&= var(h) + E^2(h) - 2E(h)E(y) + var(y) + E^2(y) \\
&= [E(h)-f(x)]^2 + var(h) + var(z)  \end{aligned} \end{equation}$$

The result shows that the expected squared error can split into three pieces: **$Bias^2$ of method**, **Variance of method** and **Irreducible error**.

### Experiment

Let’s confirm the theory behind the bias-variance decomposition with an empirical experiment that measures the bias and variance for polynomial regression with 0 degree, 1st degree, and 2nd degree polynomials.

![[B-V trade off.png]]


The bias-variance decomposition confirms our understanding that the true model is linear. While a quadratic model achieves the same theoretical bias as a linear model, it overfits to the data, as indicated by its high variance.

Let us conclude by stating some implications of the Bias-Variance Decomposition:
* Underfitting is equivalent to high bias; most overfitting correlates to high variance.
* Training error reflects bias but not variance. Test error reflects both. In practice, if the training error is much smaller than the test error, then there is overfitting.
* **Adding good features will decrease the bias, but adding a bad feature rarely increase the bias.**
* For real-world data, f is rarely known, and the noise model might be wrong, so we can’t calculate bias and variance. But we can test algorithms over **synthetic data**.

