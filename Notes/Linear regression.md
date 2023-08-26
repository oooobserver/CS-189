## 1 Ordinary Least Squares
***
Our goal in machine learning is to extract a relationship from data. In regression tasks, this relationship takes the form of a function y = f(x).

we assume the relation is $y = X^Tw$,X is called design matrix .we get answer by minimize the squared errors:
$$L(w) = \sum_{i=1}^{n}(x_iw-y_i)^2=min ||Xw − y||_2^2$$

**Note: $if L : R ^d → R$ is continuously differentiable, then any local optimum w ∗ satisfies ∇L(w ∗ ) = 0.

$L(w)=(Xw − y)^T(Xw − y)=w^TX^TXw − 2w^TX^Ty + y^T y$
$∇L(w ) = 2 X^TXw - 2X^Ty$
$w = (X^TX)^−1X^Ty$

here is question ,how do we know this will get the best answer .In this derivation we have used the condition ∇L(w ∗ ) = 0, which is a necessary but not sufficient condition for optimality. We found a critical point, but in general such a point could be a local minimum, a local maximum, or a saddle point.

**a function L(w) is considered convex if its Hessian matrix is positive semidefinite matrix.**
$∇_2L(w) = 2X^TX$
$2v^TX^Txv=(xv)^T(xv)=||Xv||_2^2>=0(for\ any\ v)$

**you can see this with probability perceptive as MLE in [[MAP]]**



## 2 Least absolute deviation
***
$laplace\ distribution:laplace(y,\mu,\sigma)=\frac{1}{2\sigma}exp\{{-\frac{|y-\mu|}{\sigma}}\}$





## 3.Ridge regression
***

we apply this idea($p(\theta)=\mathcal{N}(\theta;0,\sigma^2T)$), or equivalently $l_2-regularization$ to OLS
the $p(\theta)$ perspective is [[MAP]]

objective:
$$\begin{equation} \begin{aligned}arg_{\theta}min&=||Xw-Y||_2^2+a||w||_2^2\\ &= w^TX^TXw − 2w^TX^Ty + y^T y + \lambda w^tw\\\bigtriangledown_w&=2 X^TXw - 2X^Ty+2\lambda w\end{aligned} \end{equation}$$

$$\begin{equation} \begin{aligned} \bigtriangledown_w&=0\\2 X^TXw - 2X^Ty+2\lambda w&=0\\
w&=(X^TX+\lambda I)^{-1}X^TY\end{aligned} \end{equation}$$

the $X^TX$ not always have inverse but $(X^TX+\lambda I)$ always have, the proof can see in [[Math Review#4. positive semi-definite matrix(PSD)]] 


this method have some benefit:
* it make resulting matrix smaller because $(X^TX+\lambda I)$ is bigger
* it can fix ill condition, because it always has inverse

This value is guaranteed to achieve the (unique) global minimum, because the objective function is strongly convex. To show that f is strongly convex, it suffices to compute the Hessian of f :
$$\bigtriangledown_2L(w) = 2(X^TX+\lambda I)$$
**Since the Hessian is positive definite, we can equivalently say that the eigenvalues of the Hessian are strictly positive and that the objective function is strongly convex.**

A useful property of strongly convex functions is that they have a unique optimum point, so the solution to ridge regression is unique.

**In OLS, when the term $X^TX$ is not invertible, this does not imply that no solution exists! In OLS, there always exists a solution, there are infinitely many solutions. because the range space of $X^TX$ and X are equivalent.



#### select $\lambda$ 
we can't learn this as w, and this is called **hyperparameter** : a parameter is not learned but we have to set ourselves.




## 4.LASSO regression
***
we choose $p(\theta_i)=laplace(\theta_i;0,b)$ and $p(\theta_i)=\frac{1}{2b}exp(-\frac{|\theta_i|}{b})$ 

so $p(\theta)=\prod p(\theta_i)$ 

$$arg_{\theta}min\sum||Xw-Y||_2^2+\frac{1}{b}\sum|w|$$

in this method wo add $l_1$ norm as a regularization

* **this method tend to induce sparse solution which means some of the parameters might be 0**
* this method don't have an analytical solution

because of fist conclusion ,this method can do feature select
