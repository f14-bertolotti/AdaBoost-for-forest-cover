# AdaBoost-for-forest-cover
This is an implementation of the boosting algorithm AdaBoost.
AdaBoost (Adaptive Boost) is a boosting algorithm which 
weights different independent predictors to generate a new
more capable predictor.

The algorithm works as follow:
'''
- Let A be a learning algorithm (like stumps).

- Let S be the training set.

- Let T the number of predictors to generate.

- Set the probability *P((x,y))* of getting each sample *(x,y)* from *S* as *1/|S|*.
  
- For each *t = 1,2,...,T*:

  - Generate a predictor *A(S,P)=h* where the samples are weighted per *P*.

  - Compute the probability of *h* mistaking on *S*, let this quantity be *e*.

  - Compute the weight of *h*, *w = ln((1-e)/e)/2*
  
  - Compute the new *P((x,y)) = P((x,y))*exp(-h(x)\*w\*y)*
  
  - Normalize *P((x,y)) = P((x,y))/sum(P(S))*. so that, the sum of all probabilities is 1.

- returns all the predictors and all the weights.
'''
Now the new prediction would be computed as the sign of the weighted sum of all prediction:

*sgn(h1(x)\*w1 + h2(x)\*w2 + ... + hT(x)\*wT).*

As base predictors tree stumps have been uses.
A stump is a tree predictor with only one decision node.
It can be represented by a couple *(sel, thr)* where:
sel is the index to select and the is the threshold to check.
Given the stump *(2, 7)* and a sample *x = [1,2,3,4,5,6]*
we would check if *x[2] <= 7*, which is true, in this case.

While the dataset comes from https://www.dropbox.com/s/sr856cj0s2re4qp/forest-cover-type.csv?dl=0,
in which the last column is the label to predict.

