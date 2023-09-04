
### True risk and empirical risk

**Risk** is defined as expected loss : $R(\theta)=E[l(\theta;X,Y)]$ . And the **empirical risk** is the average loss on **training set** : $R(\theta)=\frac{1}{N}\sum l(\theta;x_i,y_i)$ , true risk is the loss on validation set. Supervised learning is oftentimes **empirical risk minimization (ERM)** .

When the true risk is much higher than empirical risk we called **overfitting**, when two thing are both high we called **underfitting**. 


### work flow

We know the two risks and now we can construct a work flow:
![[work flow.png]]

In this image after we avoid two problems we are done but how we know exactly about the accuracy about model? We will use a test set and this set **never use in train and validation**, only use to test.

Here arise a question, how do we avoid two problems?

We usually through selecting hyperparameter to avoid this.


### Cross validation

In practice, if we keeping use same validation set, it may have overfitting on this set. So we use **K-fold** method. 

In K-Fold Cross-Validation, the dataset is divided into K subsets (or "folds") only once. Then, in each iteration or "fold," one of the K subsets is used as the test set, while the remaining K-1 subsets are used as the training set. This process is repeated K times, with each of the K subsets used exactly once as the test set.