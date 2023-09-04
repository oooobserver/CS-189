### Motivation

Previous method is all about linear regression which means our decision boundary is a line, but what if out boundary is not linear. We can **featurization** $\phi(x)$ instead x which means augment features to the data points. Suppose our original input $x\in R^l$ and the augment feature $\phi(x)\in R^d$ . And if we want to use an order p polynomial featurization there are $d^p$ term, which is are **prohibitive cost**, there is a better way called **kernel trick**.

### kernel trick

Suppose the original feature is $[x_1,x_2]$ and we want featurization it to quadratic, so we have:
$$[x_1^2,x_2^2,x_1x_2,x_1,x_2,1]$$
And the formula have a term like: $\phi(x)^T\phi(x')$ 

But we can use this to instead:
$$\begin{equation} \begin{aligned} (1+x^Tx')^2 &= (1+x_1x_1'+x_2x_2')\\ 
&= x_1^2x_1'^2+x_2^2x_2'^2+2x_1x_1'+2x_2x_2'+2x_1x_1'x_2x_2'+1    \end{aligned} \end{equation}$$

We can treat this as the inner product of previous but slight difference at coefficient.

**Note:** this trick is only useful for model require inner product between different $\phi(x)$


### Kernelization

The **kernel function** $k(x,x')=\phi(x)^T\phi(x')$, and the **kernel matrix** store the inner product between all pairs of  points.

There are two examples of **kernel function**:
1. The **polynomial kernel** $k(x;x')=(c+xTx')^p$ corresponds to a $\phi(x)$ containing all combinations of p (or fewer) dimension of x.
2. The **radial basis function (RBF)** kernel $k(x;x')=exp(-\frac{||x-x'||_2^2}{2\sigma^2})$ corresponds to infinite dimensional  $\phi(x)$. 

This allow us to use linear model to express nonlinear solutions!

### Perceptron

If you forget what it is about, see [[Support Vector Machine(SVM)#perceptron algorithm]] .

Let's kernelize this algorithm:

If we initialize $\theta = 0$ , then the formula become $\theta = \sum_{i=1}^N a_iy_ix_i$. And you may ask **what is a in this formula?**  Remember that in this algorithm we will go through the whole dataset many times until all datapoints are on correct side, so the $a_i$ is the times we should add this point.

After this step the algorithm become this:
```python
a = 0
for i in len(x):
	if sum(a*y[i]*x*x*[i]) != y[i]:
		a[i] += 1
```


The term $\sum_{i=1}^N a_iy_ix_i$ is also appear in **SVM** and kernel logistic regression, there is no accidence.


### Ridge regression

Recall that in this, the result is $w=(X^TX+\lambda I)^{-1}X^TY$ , and $x\in R^{N\times d}$ . Note that in this formula $X^TX \in R^{d\times d}$ is not the kernel function so we should do some transform to this:
$$\begin{equation} \begin{aligned} 
&= (X^TX+\lambda I)^{-1}X^T(XX^T + \lambda I_N)(XX^T + \lambda I_N)^{-1}Y \\ 
&= X^T(XX^T + \lambda I_N)^{-1}Y
\end{aligned} \end{equation}$$

For predicate a new point we have:
$$W^Tx=Y^T(XX^T + \lambda I_N)^{-1}Xx$$
You can easily see now all we have is inner product.


