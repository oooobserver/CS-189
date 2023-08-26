
In mathematics, gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. The idea is to take repeated steps in the opposite direction of the gradient (or approximate gradient) of the function at the current point, because this is the direction of steepest descent. 
$$\theta\leftarrow\theta-\alpha\bigtriangledown_\theta\frac{1}{N}\sum l(\theta;x_i,y_i)$$
for strictly convex, you can find global minimum. $\alpha$ can be d-dimension, in this case I know move in certain direction is faster than other.




### Stochastic Optimization(SGD)

if the dataset is very big it's very difficult to compute the sum of loss and divide by N, and in classical gradient descent we iterate small so it will compute lots of times.

instead we pick a **batch size** $B\ll N$ , we randomly sample from the dataset and compute
$\bigtriangledown_\theta\frac{1}{B}\sum l(\theta;x_i,y_i)$. we can use Monte Carlo estimation to interpret this.

for the problem like logistic regression we may only need 1 datapoint 


