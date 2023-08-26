
## information theory
***
### Gibbs' inequality
P and Q are both distribution, with equality if and only if P = Q
$$ -\sum_{x}P(x)log P(x)\leq -\sum_{x}P(x)log Q(x)$$
proof(in x <= x - 1):
$$-\sum_{x}P(x)ln\frac{Q(x)}{P(X)}\geq-\sum_{x}P(x)(\frac{Q(x)}{P(X)}-1)=-\sum_{x}Q(x)+\sum_{x}P(x)=0$$
$$-\sum_{x}P(x)ln\frac{Q(x)}{P(X)} \geq 0$$
$$-\sum_{x}P(x)log P(x)\leq -\sum_{x}P(x)log Q(x)$$


### 1. entropy(expected surprise)
$$H(x)=- \sum_{x}P(x)log P(x)=E[-logP(X)]$$
the less P(X) is the high -logP(x) is , means when less possible event happens ,will be more surprise

### 2. cross-entropy
$$H(P,Q)=- \sum_{x}P(x)log Q(x)=E_P[-logQ(X)]$$
this measure how similar P and Q
$$min(H(P<Q))=H(P)=H(Q)$$
proof:
use Gibbs' inequality :$-\sum_{x}P(x)log P(x)\leq -\sum_{x}P(x)log Q(x)$

### 3. KL divergence
$$D_{KL}(P||Q)=\sum_{x}P(x)\frac{P(x)}{Q(X)}=H(P,Q)-H(P)\geq0$$
this is also measure the difference between P and Q

 
## linear algebra
***
### 1. $l_p~norm:$
$$||u||_p=(\sum|u_i|^p)^{\frac{1}{p}}$$

### 2. Frobenius norm:
$$||M||_F=\sqrt{\sum_i\sum_ja_{ij}^2}=\sqrt{tr(A^TA)}$$

### 3. **eigenvalues and eigenvector**:

when you do a linear transformation to a matrix M, like UV, compare the original space and the new space ,there is some vector that didn't change their position and just stretch or shrink .These vector called eigenvector and the times they stretch or shrink called eigenvalues.so the formula is 
$$Ax=\lambda x(\lambda \ is \ eigenvalue, x \ is \ eigenvector)$$
and if the number of eigenvector greater than the r(A),can compose eigen basis, suppose r(A) = 2
$$[x_1,x_2]^TA[x_1,x_2]=\begin{vmatrix}\lambda_1& 0\\ 0 & \lambda_2\end{vmatrix}$$
### 4. positive semi-definite matrix(PSD):
$$X^TMX\geq0 \ for \ all \ X  \in \ R^n$$
**A symmetric matrix is positive semi-definite if and only if all of its eigenvalues are nonnegative, and positive definite if and only if all of its eigenvalues are positive.**

proof:
Suppose A is positive semi-definite, and let x be an eigenvector of A with eigenvalue λ
$$X^TAX=X^T(\lambda X)=\lambda X^TX=\lambda||X||_2$$
because $||X||_2>0$ so $\lambda \geq0$

$Suppose A ∈ R^{m×n}. Then \ A^TA \ is \ positive \ semi-definite. \ If \ null(A) = {0}, then \ A^TA \ is \ positive \ definite.$
proof:
$$X^T(A^TA)X=(AX)^T(AX)=||AX||_2\geq0$$
and if null(A) = {0} means that if $x \neq 0,||AX||_2 > 0$ and x is eigenvector so it can't be 0

**If A is positive semi-definite and $\epsilon$ > 0, then A + $\epsilon$I is positive definite.**



### 5. singular value decomposition(SVD):
$$ A=U\Sigma V^T(A\in R^{m\times n})$$
$U\in R^{m\times m},V\in R^{n \times n}$ are orthogonal matrices and $\Sigma \in R^{m\times n}$ is a diagonal matrix with the singular values of A (denoted $\sigma_i$) on its diagonal.

and by convention:
$$\sigma_1>\sigma_2>...>\sigma_{min(m,n)}=0$$

Another way to write the SVD (cf. the sum-of-outer-products identity) is
$$A=\sum_{i=1}^r\sigma_iU_iV_i^T$$
$U_i,V_i$ are ith column vector

**Observe that the SVD factors provide eigen decompositions for $A^TA$ and$A^TA$**
$$A^TA=(U\Sigma V^T)^TU\Sigma V^T=V\Sigma^T\Sigma V^T$$
same as $AA^T$

it contain several step:
* rotate
* stretch
* dimension manipulate
* rotate





### 6. Fundamental Theorem of Linear Algebra
if $A \in R^{m\times n}$,then
* $null(A)=range(A^T)^{\perp}$ 

* $null(A)\oplus range(A^T)=R^n$

* $r(A) + r(null(A))=n$

* If $A=U\Sigma V^T$ is the singular value decomposition of A, then the columns of U and V form orthonormal bases for the four “fundamental subspaces” of A:
$$\begin{array}{c|lcr}
Subspace & \text{Columns}  \\
\hline
range(A) & The \ first \ r \ columns \ of \ U\\
range(A^T) & \text{The first r columns of V}\\
null(A^T) &  \text{The last m − r columns of U}\\
null(A) & \text{The last n − r columns of V}
\end{array}$$

proof:
(a):

first,$Ax = 0$ means two things:
1. x in null(A)
2. x is orthogonal to the row space of A, because $\forall_i A_ix=0$ 

the formula any vector in A's null space is  orthogonal to the column space of $A^T$ , and the column space of $A^T$ is equal to the row space of A

(b):

Recall our previous result on orthogonal complements: if S is a finite-dimensional subspace of V , then $V = S \oplus S^\perp.$  

(c):

Recall that if U and W are subspaces of a finite-dimensional vector space V , then $dim(U \oplus W) = dim U + dim W.$








### 7. Direct Sum
Suppose V and W are two vector spaces over the same field (usually the real numbers or complex numbers). The direct sum of V and W, denoted by V $\oplus$ W, is a new vector space that contains all possible combinations of vectors from V and W.

if $\forall v\in V$ can explain as $v = u+w$ ,we say that V is the direct sum of U and W, denoted as$V = U \oplus W.$ 



## vector calculate
***



## possibility theory
***
1. multivariate Gaussians(MVG):[[MVG]]





## Optimization
***

### iterative optimization:

[[Iterative optimization]]

### gradient descent:

[[gradient descent]]

###  Duality

**Duality** or the **duality principle** is the principle that optimization problem may be viewed from either of two perspectives, the **primal problem** or the **dual problem**.

If the primal is a minimization problem then the dual is a maximization problem (and vice versa). Any feasible solution to the primal (minimization) problem is at least as large as any feasible solution to the dual (maximization) problem. Therefore, the solution to the primal is an upper bound to the solution of the dual, and the solution of the dual is a lower bound to the solution of the primal. This fact is called **weak duality**.

In general, the optimal values of the primal and dual problems need not be equal. Their difference is called the duality gap For convex optimization problems, the duality gap is zero under a constraint qualification condition. This fact is called **strong duality**.

the dual problem has many kinds, and the common is  Lagrangian dual problem, other problems are the Wolfe dual problem and the Fenchel dual problem.

The Lagrangian dual problem is obtained by forming the Lagrangian of a minimization problem by using **nonnegative Lagrange multipliers** to add the constraints to the objective function, and then solving for the primal variable values that minimize the original objective function. **This solution gives the primal variables as functions of the Lagrange multipliers, which are called dual variables**, so that the new problem is to maximize the objective function with respect to the dual variables under the derived constraints on the dual variables.

The dual function represent the minimum of Lagrangian function.

### Quadratic programming

QP means to optimize (minimize or maximize) a multivariate quadratic function subject to linear constraints on the variables.

**The Lagrangian dual of a QP is also a QP.** Note Q is positive definite. We write the Lagrangian function as:

$$L = \frac{1}{2}x^TQx+\lambda^T(Ax-b)$$
we know that the dual problem is the infimum of L, assume dual function is $g(\lambda)$  we can find by set gradient to 0:
$$x^*=-Q^{-1}A^T\lambda$$

Hence the dual function is:
$$g(\lambda)=-\frac{1}{2}\lambda^TAQ^{-1}A^T\lambda-\lambda^Tb$$





