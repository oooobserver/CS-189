# Maximum a Posteriori 
***
## Loss function

* $l(y,\hat{y})$ 
* $l(\theta;x,y)$ this loss function incurred by $\theta$ 

**use MLE to loss function**

max likelihood is equivalent minimize cross entropy
max likelihood is also equal to minimize the loss function

**MLE is a probabilistic justification for why using squared error (which is the basis of OLS) is a good metric for evaluating a regression model.**

$$l(\theta;x,y)=-logP_\theta(y|x)=arg_\theta \ min l(\theta;x,y)$$
we assume the $y \sim N(\hat{y},\sigma^2)$ 

$$arg_\theta maxP(D|\theta)=\prod_{i=1}^{n}P_\theta(y_i|x_i)$$
$$\begin{align*}l(\theta,x,y)=argmax-\sum\frac{(y_i-\hat{y})^2}{2\sigma^2}-nlog\sqrt{2\pi}\sigma\\argmin=\sum\frac{(y_i-\hat{y})^2}{2\sigma^2}\end{align*}$$
**the final formula is Ordinary Least Squares is our loss function**




## MAP

we have some prior brief the $\theta$ ,so the idea is treat  the  $\theta$ as a RV and put a probability on it.so $p(\theta)$ is a prior distribution.

we often want the function more smoother which means don't be so quickly change as the input change.so we want as the magnitude of $\theta$ increases,$p(\theta)$ should decrease.

by far the most commonly used prior is  $p(\theta)=\mathcal{N}(\theta;0,\sigma^2I)$ 

this distribution center at 0, and as the $\theta$ increase both negative and nonnegative,$p(\theta)$ will decrease

we take Bayesian approach to estimating dataset D
$$p(\theta|D)=\frac{p(D|\theta)p(\theta)}{p(D)}$$
$$p(D|\theta)=\sum p(x_i)p(y_i|x_i,\theta)$$

$p(D)$ is a constant and $p(D|\theta)$ is main formula of MLE
$$logP(D|\theta)=logP(D|\theta)+logP(\theta)-logP(D)$$


## MLE and MAP

in MLE ,the goal is to find the hypothesis model that maximizes the probability of the data.

in MAP, the goal is to find the model, for which the data maximizes the probability of the model.

this represent two inference: Bayesian inference, classical method.

in classical method, we treat $\theta$ as unknown constant, we maximizes the probability of the data.

in Bayesian inference, we treat $\theta$ as a RV, we maximizes the probability of the model use data.
it can say we are maximizing is known as the posterior. because we believe the  $\theta$ is distributed underly this.

we assume $\theta \sim N(\theta_j,\sigma_j^2)$
$$\begin{align*}\theta_{MAP}=argmin(\sum\frac{(y_i-\hat{y})^2}{2\sigma^2})+(\sum\frac{(\theta_i-\theta_j)^2}{2\sigma_j^2})\\=argmin\sum(y_i-x_i^T\theta)^2+\frac{\sigma^2}{\sigma_j^2}(\sum(\theta_i-\theta_j)^2)\\=argmin\sum(y_i-x_i^T\theta)^2+\frac{\sigma^2}{\sigma_j^2}\sum\theta_j^2\end{align*}$$

**We conclude that MAP is a probabilistic justification for adding the penalized ridge term in Ridge Regression.**

