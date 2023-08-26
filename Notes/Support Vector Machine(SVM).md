
### introduction

briefly,  SVM is an attempt to model decision boundaries directly use geometry perceptive.

Here’s the setup for the problem. We are given a training dataset $D = \{(xi , yi)\}$, where $x_i\in R^d$and $y_i\in \{−1, +1\}$ . Our goal is to find a d−1 dimensional hyperplane decision boundary H which separates the +1’s from the −1’s.
### perceptron algorithm

this algorithm is repeating following update for $\theta$ :

* randomly pick a $(x_i,y_i)$ from dataset, if $\theta^Tx_i=y_i$ move on, otherwise $\theta\leftarrow\theta+x_iy_i$ 
* we repeat this until every datapoint is correct

this algorithm is very simple but have two shortcomings:

1. if the data is not linearly separable, the perceptron fails to find a stable solution. but, soft-margin SVMs can fix this issue.
2. there are infinitely many hyperplanes, some hyperplanes are better than others, but the perceptron cannot distinguish between them. This leads to generalization issues.



### Hard-Margin SVM

Let’s express the first two constraints mathematically. First, note that **the vector w is perpendicular to the hyperplane $H = \{x : w^Tx − b = 0\}$.**

Proof: consider any two points on H. We will show that $(x_1 − x_0) \perp w.$
$$(x_1 − x_0)^T(w) = x_1^Tw − x_0^Tw = b − b = 0$$
the reason why transpose $(x_1 − x_0)$ is only doing this can make them multiple.

Since w is perpendicular to H, and d is perpendicular to H too. so **we have $d =\beta w$ , $\beta$ is a scalar.**
we assume $x_p$ is x projection on H. we have:
$$\begin{equation} \begin{aligned} w^Tx_p-b&=0  \\w^T(x-d)-b&=0    \\ w^T(x-\beta w)-b&=0 \\    \beta&=\frac{w^Tx-b}{||w||_2^2} \end{aligned} \end{equation}$$

bring back to $d =\beta w$:
$$||d||_2=\frac{|w^Tx-b|}{||w||_2}$$

Formulating the optimization - objective

margin:
$margin=arg\ min_i \frac{|w^Tx_i-b|}{||w||_2}$ and we want to maximize it, it hard to optimize min and max. For any $\alpha > 0$ , if I multiple $w,b$ by $\alpha$, the margin would not change. and the separate hyper space:$w^TX+b=0$ would not change too. so I can turn the numerator large or small, means I can let numerator be any value I want and we can set $min_i|w^Tx_i-b|=1$. and the margin is $\frac{1}{||w||_2}$ and now our goal is maximize the margin:
$$arg\ max_w\frac{1}{||w||_2}=arg\ min_w||w||_2=arg\ min_w\frac{1}{2}||w||_2^2$$
constraints: 
we want to max the margin but it has a huge precondition our point is at the right side of the line:
 $y_i(w^Tx_i-b)\geq0$ , because we set $min_i|w^Tx_i-b|=1$, so $y_i(w^Tx_i-b)\geq1$

our final formula is:

$$arg\ min_{w,b}\frac{1}{2}||w||_2^2 \ \ \ \text{s.t.}\ \ y_i(w^Tx_i-b)\geq1 \ \forall_i $$



### Soft-Margin SVM

In reality , we hardly can split the dataset perfectly using a line, more common there are some datapoint in other class's region. if we keep using hard SVM the hyper space will be wired. so in this situation, we use soft SVM.

A soft-margin SVM modifies the constraints from the hard-margin SVM by allowing some points to violate the margin. It introduces slack variables $\xi_i$  , one for each training point, into the constraints:
$$\begin{equation} \begin{aligned}y_i(w^Tx_i-b)&\geq1-\xi_i \\ \xi_i&\geq0
\end{aligned} \end{equation}$$
The constraints are now a less-strict,  because each point $x_i$ don't need to be strictly 1 “distance”  of the separating hyperplane, it can less than 1 or even in the other side of this line.

**But this doesn't mean we don't care about the datapoint accuracy, we of course want to minimize the number of point violate the line.**

so we add a regularization like before to penalize it, the formula now is:
$$arg\ min_{w,b,\xi_i}\frac{1}{2}||w||_2^2+C\sum\xi_i \ \ \ \text{s.t.}\ \ y_i(w^Tx_i-b)\geq1-\xi_i \ , \xi_i\geq0 \ \forall_i $$

C is hyperparameter, and C can impact out decision boundary. if C is very large means I really care about the magnitude of $\xi_i$ ,we more penalization the datapoint violate, means we don't want points to violate the boundary. if the C is small even 0. when it is 0, means I don't care about violation I only care about the margin, in that case, many points will be the wrong side.

this formula seems complicated, let's rewrite it concisely

there are two constraints but  we can write as one:
$$\xi_i\geq max\{1-y_i(w^Tx_i-b),0\}$$
and we can see the formula is minimize the $\xi_i$ , so either the $\xi_i$ equal to 0 or that term, now we rewrite as:
$$arg\ min_{w,b,\xi_i}\frac{1}{2}||w||_2^2+C\sum\max\{1-y_i(w^Tx_i-b),0\}  $$

now our formula is without constraints, this version define loss function use two component:
* a hinge loss which encourage accurate classification
* a $l_2$ norm regularization


### Solve

First, let wrote our soft margin SVM as Lagrangian function:
$$\begin{equation} \begin{aligned}L(w,b,\xi,\alpha,\beta)&=\frac{1}{2}||w||_2^2+C\sum\xi_i + \sum\alpha_i[1-\xi_i-y_i(w^Tx_i-b)] + \sum-\beta_i\xi_i   
\\&= \frac{1}{2}||w||_2^2 + \sum[\alpha_i + (c-\alpha_i-\beta_i)\xi_i-\alpha_iy_i(w^Tx_i-b)]
\end{aligned} \end{equation}$$

**The Lagrangian function defines a competing objective between $(w,b,\xi)$ and $(\alpha,\beta)$- the former want to minimize,  the latter wants to maximize.**

why say that, let's explain. the former is the original formula we want minimize so we say the former want to minimize. And the latter is our constraints, they are less or equal to 0, remember our Lagrangian multiplier is none negative, and in order to let the constraints obey(means less or equal to 0), we maximize it. **if constraints are violate, the multiplier can go $\infty$**, thus the $(w,b,\xi)$ must satisfy the constraints, because they want to minimize this is for their sake.

and the the former is called primal variable, the latter is called dual variable, you can see more in [[Math Review#Duality]].The primal problem is defined as:
$$min_{(w,b,\xi)}max_{(\alpha,\beta)}L(w,b,\xi,\alpha,\beta)$$

Because this problem is strong duality, we can also consider the dual problem:

$$max_{(\alpha,\beta)}min_{(w,b,\xi)}L(w,b,\xi,\alpha,\beta)$$

**The inner optimization $min_{(w,b,\xi)}L(w,b,\xi,\alpha,\beta)$ is a function of $(\alpha,\beta)$** , that's mean minimizer $(w*,b*,\xi*)$ depend on  $(\alpha,\beta)$ . this give us a intuition that **$(\alpha,\beta)$ have to "go first", $(w,b,\xi)$  have to "go second" after former set.** this also can be seen from the order, if we minimize $(w,b,\xi)$ first, we must treat $(\alpha,\beta)$ as variable otherwise we can not solve.

Let's solve this problem, as usual we set gradient equal to 0:

$$L(w,b,\xi,\alpha,\beta)=\frac{1}{2}||w||_2^2 + \sum[\alpha_i + (c-\alpha_i-\beta_i)\xi_i-\alpha_iy_i(w^Tx_i-b)]  $$

$$\begin{equation} \begin{aligned}
\bigtriangledown_w&=w*-\sum \alpha_iy_ix_i=0 \\
\frac{\partial b}{\partial L} &=\sum-\alpha_iy_i \\
\frac{\partial \xi}{\partial L} &= c - \alpha_i-\beta_i
\end{aligned} \end{equation}$$
the partial derivative of $b,\xi$ is constant, what it is this mean. this means if their partial derivative not equal to 0, in order to minimize L, they go $\infty$ , so their derivative  must be 0.

so
$$\begin{equation} \begin{aligned} 
w*&=\sum \alpha_iy_ix_i
\\\sum-\alpha_iy_i&=a*^Ty=0 \\
c - \alpha_i-\beta_i &= 0
\end{aligned} \end{equation}$$

we can use the  first formula to replace term:

$$\begin{equation} \begin{aligned} 
max \ L(w*,b*,\xi*,\alpha,\beta) \\
=max \ \frac{1}{2}w*^Tw + \sum(\alpha_i-\alpha_iy_iw*^Tx_i) \\
=max \  \frac{1}{2} \sum_j\sum_k \alpha_jy_j\alpha_ky_kx_jx_k^T-\sum_l\alpha_l\alpha_iy_ly_ix_l^Tx_i + a^T*1 \\
=max \  a^T*1 - \frac{1}{2}a^TQa \ \ \ \text{(for a appropriate matrix) }
\end{aligned} \end{equation}$$

The final form is:
$$arg \ max_\alpha \  a^T*1 - \frac{1}{2}a^TQa \ s.t.\  a*^Ty=0  , 0 < \alpha_i < c\ \forall_i$$
This is also a **QP**[[Math Review#Quadratic programming]], we previously had a optimization problem on the d-dimension w, and now we optimize N-dimension $\alpha$ .





