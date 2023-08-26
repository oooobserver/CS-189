# multivariate Gaussians
***
### Definition

There are three equivalent definitions of a jointly Gaussian (JG) random vector:

1. A random vector $Z = (Z_1, Z_2, . . . , Z_k)^T$ is JG if there exists a base random vector $U = (U_1, U_2, . . . , U_l)^T$ whose components are independent standard normal random variables, a transition matrix $R \in R^{ k\times l}$  , and a mean vector $\mu \in R^k$  , such that $Z = RU + \mu.$



2. A random vector  $Z = (Z_1, Z_2, . . . , Z_k)^T$  is JG if $\sum_{i=1}^kaiZi$ is normally distributed for every $a = (a_1, a_2, . . . , a_k)^T \in R^k$ .

3. (Non-degenerate case only) A random vector $Z = (Z_1, Z_2, . . . , Z_k)^T$  is JG if
$$ f_z(Z) = \frac{1}{\sqrt{(2\pi)^ddet(\Sigma)}}exp(-\frac{1}{2}(z-\mu)^T\Sigma^{-1}(z -\mu))$$
$\mu\in R^d,\Sigma\in R^{d*d}$ and $\Sigma$ must be PSD   

**Note that all of these conditions are equivalent.**




### covariance and level set

**The covariance matrix $\Sigma$ contains covariances(variances along the diagonal)**
$$\Sigma_{ij}=cov(x_i,x_j)=E[(x_i-E[x_i])(x_j-E[x_j])]$$

when we draw two dimensional Gaussian ,we will often draw level sets:
$[(x_1,x_2):p({x_1},{x_2})=c]$

for MVG ,the level sets are ellipsoids:$(x-\mu)^T\Sigma^{-1}(x -\mu)=c$



### turn the standard MVG to any

since  $\Sigma$ is PSD  
$$\Sigma=Q\Lambda Q^T=Q\Lambda^{\frac{1}{2}}\Lambda^{\frac{1}{2}}Q^T$$
so ,$N(\mu,\Sigma)=Q\Lambda^{\frac{1}{2}}N(0,I)+\mu$
this process is shifting ,scaling ,rotating
* scale the dimensions by $\Lambda^{\frac{1}{2}}$
* rotate the axes by Q
* shift everything by $\mu$




### entropy and KL divergence

$$H(f)=\frac{k}{2}+\frac{k}{2}ln(2\pi)+\frac{1}{2}ln(|\Sigma|)$$
$$D_{KL}(N_0||N_1)=\frac{1}{2}\{tr(\Sigma_1^{-1}\Sigma_0)+(\mu_1-\mu_0)^T\Sigma_1^{-1}(\mu_1-\mu_0)-k+ln\frac{|\Sigma_1|}{|\Sigma_0|}\}$$





### MLE

$$\begin{equation} \begin{aligned} arg_{\mu,\Sigma} \ max&=-(\frac{1}{2}\sum[(x_i-\mu)^T\Sigma^{-1}(x_i -\mu)+log\sqrt{|\Sigma|}])\\&=-(\sum[(x_i-\mu)^T\Sigma^{-1}(x_i -\mu)-log|\Sigma^{-1}|])\end{aligned} \end{equation}$$

**Note that the objective above is not jointly convex, so we decompose the minimization over $\sigma$ and $\mu$ into a nested optimization problem:**

$$\begin{equation} \begin{aligned}min&=min_{\sigma} min_{\mu}\sum[(x_i-\mu)^T\Sigma^{-1}(x_i -\mu)-log|\Sigma^{-1}|]\end{aligned} \end{equation}$$

1. MLE for $\mu$
$$\begin{align*} \bigtriangledown_\mu=-\sum[2(x_i-\mu)^T\Sigma^{-1}] =0\\ N\mu^T\Sigma^{-1}= \sum x_i\Sigma^{-1} \\ \mu= \frac{\sum x_i}{N} \end{align*}$$

2. MLE for $\Sigma$
Having solved the inner optimization problem, we now have that $\hat{\mu}$

because the equation is filled with $\Sigma^{-1}$,so it's better to derivative $\Sigma^{-1}$
$$\begin{align*} \bigtriangledown_{\Sigma^{-1}}=\sum(x_i-\mu)(x_i-\mu)^T-\frac{1}{\Sigma^{-1}}=0\\ \Sigma= \frac{\sum(x_i-\mu)(x_i-\mu)^T}{N} \end{align*}$$

### Properties

Given any two general random vectors, we cannot necessarily say “if they are uncorrelated, then they are independent”. However in the case of random vectors from the same JG joint distribution, we can make this claim.


**for Gaussians RVs, if two uncorrelated, jointly Gaussian RVs are independent.**

this proof I have written on the paper 