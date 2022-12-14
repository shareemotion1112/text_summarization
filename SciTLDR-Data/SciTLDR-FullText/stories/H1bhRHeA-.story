Recent neural network and language models have begun to rely on softmax distributions with an extremely large number of categories.

In this context calculating the softmax normalizing constant is prohibitively expensive.

This has spurred a growing literature of efficiently computable but biased estimates of the softmax.

In this paper we present the first two unbiased algorithms for maximizing the softmax likelihood whose work per iteration is independent of the number of classes and datapoints (and does not require extra work at the end of each epoch).

We compare our unbiased methods' empirical performance to the state-of-the-art on seven real world datasets, where they comprehensively outperform all competitors.

Under the softmax model 1 the probability that a random variable y takes on the label ∈ {1, ..., K}, is given by p(y = |x; W ) = e where x ∈ R D is the covariate, w k ∈ R D is the vector of parameters for the k-th class, and W = [w 1 , w 2 , ..., w K ] ∈ R D×K is the parameter matrix.

Given a dataset of N label-covariate pairs D = {(y i , x i )} N i=1 , the ridge-regularized maximum log-likelihood problem is given by DISPLAYFORM0 where W 2 denotes the Frobenius norm.

This paper focusses on how to maximize (2) when N, K, D are all large.

Having large N, K, D is increasingly common in modern applications such as natural language processing and recommendation systems, where N, K, D can each be on the order of millions or billions BID15 BID6 BID4 .A natural approach to maximizing L(W ) with large N, K, D is to use Stochastic Gradient Descent (SGD), sampling a mini-batch of datapoints each iteration.

However if K, D are large then the O(KD) cost of calculating the normalizing sum K k=1 e x i w k in the stochastic gradients can still be prohibitively expensive.

Several approximations that avoid calculating the normalizing sum have been proposed to address this difficulty.

These include tree-structured methods BID2 BID7 BID9 , sampling methods BID1 BID14 BID10 and self-normalization BID0 .

Alternative models such as the spherical family of losses that do not require normalization have been proposed to sidestep the issue entirely BID13 .

BID11 avoid calculating the sum using a maximization-majorization approach based on lower-bounding the eigenvalues of the Hessian matrix.

All 2 of these approximations are computationally tractable for large N, K, D, but are unsatisfactory in that they are biased and do not converge to the optimal W * = argmax L(W ).Recently BID16 managed to recast (2) as a double-sum over N and K. This formulation is amenable to SGD that samples both a datapoint and class each iteration, reducing the per iteration cost to O(D).

The problem is that vanilla SGD when applied to this formulation is unstable, in that the gradients suffer from high variance and are susceptible to computational overflow.

BID16 deal with this instability by occasionally calculating the normalizing sum for all datapoints at a cost of O(N KD).

Although this achieves stability, its high cost nullifies the benefit of the cheap O(D) per iteration cost.

The goal of this paper is to develop robust SGD algorithms for optimizing double-sum formulations of the softmax likelihood.

We develop two such algorithms.

The first is a new SGD method called U-max, which is guaranteed to have bounded gradients and converge to the optimal solution of (2) for all sufficiently small learning rates.

The second is an implementation of Implicit SGD, a stochastic gradient method that is known to be more stable than vanilla SGD and yet has similar convergence properties BID18 .

We show that the Implicit SGD updates for the doublesum formulation can be efficiently computed and has a bounded step size, guaranteeing its stability.

We compare the performance of U-max and Implicit SGD to the (biased) state-of-the-art methods for maximizing the softmax likelihood which cost O(D) per iteration.

Both U-max and Implicit SGD outperform all other methods.

Implicit SGD has the best performance with an average log-loss 4.29 times lower than the previous state-of-the-art.

In summary, our contributions in this paper are that we:1.

Provide a simple derivation of the softmax double-sum formulation and identify why vanilla SGD is unstable when applied to this formulation (Section 2).

2.

Propose the U-max algorithm to stabilize the SGD updates and prove its convergence (Section 3.1).

3.

Derive an efficient Implicit SGD implementation, analyze its runtime and bound its step size (Section 3.2).

4.

Conduct experiments showing that both U-max and Implicit SGD outperform the previous state-of-the-art, with Implicit SGD having the best performance (Section 4).

In order to have an SGD method that samples both datapoints and classes each iteration, we need to represent (2) as a double-sum over datapoints and classes.

We begin by rewriting (2) in a more convenient form, DISPLAYFORM0 The key to converting (3) into its double-sum representation is to express the negative logarithm using its convex conjugate: DISPLAYFORM1 where u = − log(−v) and the optimal value of u is u * (a) = log(a).

Applying (4) to each of the logarithmic terms in (3) yields DISPLAYFORM2 is our double-sum representation that we seek to minimize and the optimal solution for u i is DISPLAYFORM3 Clearly f is a jointly convex function in u and W .

In Appendix A we prove that the optimal value of u and W is contained in a compact convex set and that f is strongly convex within this set.

Thus performing projected-SGD on f is guaranteed to converge to a unique optimum with a convergence rate of O(1/T ) where T is the number of iterations BID12 .

The challenge in optimizing f using SGD is that it can have problematically large magnitude gradients.

DISPLAYFORM0 where DISPLAYFORM1 is the inverse of the probability of class j being sampled either through i or k, and n j = |{i : y i = j, i = 1, ..., N }|.

The corresponding stochastic gradient is: DISPLAYFORM2 If u i equals its optimal value u * i (W ) = log(1 + k =yi e x i (w k −wy i ) ) then e x i (w k −wy i )−ui ≤ 1 and the magnitude of the N (K − 1) terms in the stochastic gradient are bounded by DISPLAYFORM3 1 and the magnitude of the gradients can become extremely large.

Extremely large gradients lead to two major problems: (a) the gradients may computationally overflow floating-point precision and cause the algorithm to crash, (b) they result in the stochastic gradient having high variance, which leads to slow convergence 3 .

In Section 4 we show that these problems occur in practice and make vanilla SGD both an unreliable and inefficient method 4 .The sampled softmax optimizers in the literature BID1 BID14 BID10 do not have the issue of large magnitude gradients.

Their gradients are bounded by N (K − 1) x i 2 due to their approximations to u * i (W ) always being greater than x i (w k − w yi ).

For example, in one-vs-each BID17 , u * i (W ) is approximated by log(1 + e x i (w k −wy i ) ) > x i (w k − w yi ).

However, as they only approximate u * i (W ) they cannot converge to the optimal W * .The goal of this paper is to design reliable and efficient SGD algorithms for optimizing the doublesum formulation f (u, W ) in (5).

We propose two such methods: U-max (Section 3.1) and an implementation of Implicit SGD (Section 3.2).

But before we introduce these methods we should establish that f is a good choice for the double-sum formulation.

The double-sum in (5) is different to that of BID16 .

Their formulation can be derived by applying the convex conjugate substitution to (2) instead of (3).

The resulting equations are DISPLAYFORM0 Although both double-sum formulations can be used as a basis for SGD, our formulation tends to have smaller magnitude stochastic gradients and hence faster convergence.

To see this, note that typically x i w yi = argmax k {

x i w k } and so theū i , x i w yi and e x i wy i −ūi terms in (8) are of the greatest magnitude.

Although at optimality these terms should roughly cancel, this will not be the case during the early stages of optimization, leading to stochastic gradients of large magnitude.

In contrast the function f ik in (6) only has x i w yi appearing as a negative exponent, and so if x i w yi is large then the magnitude of the stochastic gradients will be small.

In Section 4 we present numerical results confirming that our double-sum formulation leads to faster convergence.

As explained in Section 2.2, vanilla SGD has large gradients when u i x i (w k − w yi ).

This can only occur when u i is less than its optimum value for the current W , since u * DISPLAYFORM0 and so the gradients are bounded.

It also brings u i closer 5 to its optimal value for the current W and thereby decreases the the objective f (u, W ).

This is exactly the mechanism behind the U-max algorithm -see Algorithm 1 in Appendix C for its pseudocode.

U-max is the same as vanilla SGD except for two modifications: (a) u i is set equal to log(1 + e DISPLAYFORM1 DISPLAYFORM2 , then U-max with threshold δ converges to the optimum of (2), and the rate is at least as fast as SGD with same learning rate, in expectation.

Proof.

The proof is provided in Appendix D.U-max directly resolves the problem of extremely large gradients.

Modification (a) ensures that δ ≥ x i (w k − w yi ) − u i (otherwise u i would be increased to log(1 + e x i (w k −wy i ) )) and so the magnitude of the U-max gradients are bounded above by N (K − 1)e δ x i 2 .In U-max there is a trade-off between the gradient magnitude and learning rate that is controlled by δ.

For Theorem 1 to apply we require that the learning rate η t ≤ δ 2 /(4B 2 f ).

A small δ yields small magnitude gradients, which makes convergence fast, but necessitates a small η t , which makes convergence slow.

Another method that solves the large gradient problem is Implicit SGD 6 BID3 BID18 .

Implicit SGD uses the update equation DISPLAYFORM0 where θ (t) is the value of the t th iterate, f is the function we seek to minimize and ξ t is a random variable controlling the stochastic gradient such that ∇f (θ) = E ξt [∇f (θ, ξ t )].

The update (9) differs from vanilla SGD in that θ (t+1) appears on both the left and right side of the equation, DISPLAYFORM1 .

6 Also known to as an "incremental proximal algorithm" BID3 whereas in vanilla SGD it appears only on the left side.

In our case θ = (u, W ) and DISPLAYFORM2 Although Implicit SGD has similar convergence rates to vanilla SGD, it has other properties that can make it preferable over vanilla SGD.

It is known to be more robust to the learning rate BID18 , which important since a good value for the learning rate is never known a priori.

Another property, which is of particular interest to our problem, is that it has smaller step sizes.

Proposition 1.

Consider applying Implicit SGD to optimizing DISPLAYFORM3 and so the Implicit SGD step size is smaller than that of vanilla SGD.Proof.

The proof is provided in Appendix E.The bound in Proposition 1 can be tightened for our particular problem.

Unlike vanilla SGD whose step size magnitude is exponential in x i (w k − w yi ) − u i , as shown in FORMULA8 , for Implicit SGD the step size is asymptotically linear in x i (w k − w yi ) − u i .

This effectively guarantees that Implicit SGD cannot suffer from computational overflow.

Proposition 2.

Consider the Implicit SGD algorithm where in each iteration only one datapoint i and one class k = y i is sampled and there is no ridge regularization.

The magnitude of its step size in w is O( DISPLAYFORM4 Proof.

The proof is provided in Appendix F.2.The difficulty in applying Implicit SGD is that in each iteration one has to compute a solution to (9).

The tractability of this procedure is problem dependent.

We show that computing a solution to (9) is indeed tractable for the problem considered in this paper.

The details of these mechanisms are laid out in full in Appendix F.Proposition 3.

Consider the Implicit SGD algorithm where in each iteration n datapoints and m classes are sampled.

Then the Implicit SGD update θ (t+1) can be computed to within accuracy in runtime O(n(n + m)(D + n log( −1 ))).Proof.

The proof is provided in Appendix F.3.In Proposition 3 the log( −1 ) factor comes from applying a first order method to solve the strongly convex Implicit SGD update equation.

It may be the case that performing this optimization is more expensive than computing the x i w k inner products, and so each iteration of Implicit SGD may be significantly slower than that of vanilla SGD or U-max.

However, in the special case of n = m = 1 we can use the bisection method to give an explicit upper bound on the optimization cost.

Proposition 4.

Consider the Implicit SGD algorithm with learning rate η where in each iteration only one datapoint i and one class k = y i is sampled and there is no ridge regularization.

Then the Implicit SGD iterate θ (t+1) can be computed to within accuracy with only two D-dimensional vector inner products and at most log 2 ( DISPLAYFORM5 Proof.

The proof is provided in Appendix F.1For any reasonably large dimension D, the cost of the two D-dimensional vector inner products will outweigh the cost of the bisection, and Implicit SGD will have roughly the same speed per iteration as vanilla SGD or U-max.

In summary, Implicit SGD is robust to the learning rate, does not have overflow issues and its updates can be computed in roughly the same time as vanilla SGD.

Two sets of experiments were conducted to assess the performance of the proposed methods.

The first compares U-max and Implicit SGD to the state-of-the-art over seven real world datasets.

The second investigates the difference in performance between the two double-sum formulations discussed in Section 2.3.

We begin by specifying the experimental setup and then move onto the results.

Data.

We used the MNIST, Bibtex, Delicious, Eurlex, AmazonCat-13K, Wiki10, and Wiki-small datasets 7 , the properties of which are summarized in TAB1 .

Most of the datasets are multi-label and, as is standard practice (Titsias, 2016), we took the first label as being the true label and discarded the remaining labels.

To make the computation more manageable, we truncated the number of features to be at most 10,000 and the training and test size to be at most 100,000.

If, as a result of the dimension truncation, a datapoint had no non-zero features then it was discarded.

The features of each dataset were normalized to have unit L 2 norm.

All of the datasets were pre-separated into training and test sets.

We only focus on the performance on the algorithms on the training set, as the goal in this paper is to investigate how best to optimize the softmax likelihood, which is given over the training set.

Algorithms.

We compared our algorithms to the state-of-the-art methods for optimizing the softmax which have runtime O(D) per iteration 8 .

The competitors include Noise Contrastive Estimation (NCE) BID14 , Importance Sampling (IS) BID1 and One-Vs-Each (OVE) BID17 .

Note that these methods are all biased and will not converge to the optimal softmax MLE, but something close to it.

For these algorithms we set n = 100, m = 5, which are standard settings 9 .

For Implicit SGD we chose to implement the version in Proposition 4 which has n = 1, m = 1.

Likewise for U-max we set n = 1, m = 1 and the threshold parameter δ = 1.

The ridge regularization parameter µ was set to zero for all algorithms.

Epochs and losses.

Each algorithm is run for 50 epochs on each dataset.

The learning rate is decreased by a factor of 0.9 each epoch.

Both the prediction error and log-loss (2) are recorded at the end of 10 evenly spaced epochs over the 50 epochs.

Learning rate.

The magnitude of the gradient differs in each algorithm, due to either under-or overestimating the log-sum derivative from (2).

To set a reasonable learning rate for each algorithm on 7 All of the datasets were downloaded from http://manikvarma.org/downloads/XC/ XMLRepository.html, except Wiki-small which was obtained from http://lshtc.iit.

demokritos.gr/.8 BID16 have runtime O(N KD) per epoch, which is equivalent to O(KD) per iteration.

This is a factor of K slower than the methods we compare against.

9 We also experimented setting n = 1, m = 1 in these methods and there was virtually no difference except the runtime was slower.

For example, in Appendix G we plot the performance of NCE with n = 1, m = 1 and n = 100, m = 5 applied to the Eurlex dataset for different learning rates and there is very little difference between the two.

Table 2 : Tuned initial learning rates for each algorithm on each dataset.

The learning rate in 10 0,±1,±2,±3 with the lowest log-loss after 50 epochs using only 10% of the data is displayed.

Vanilla SGD applied to AmazonCat, Wiki10 and Wiki-small suffered from overflow with a learning rate of 10 −3 , but was stable with smaller learning rates (the largest learning rate for which it was stable is displayed).

Wiki-small Figure 1 : The x-axis is the number of epochs and the y-axis is the log-loss from (2) calculated at the current value of W .

each dataset, we ran them on 10% of the training data with initial learning rates η = 10 0,±1,±2,±3 .

The learning rate with the best performance after 50 epochs is then used when the algorithm is applied to the full dataset.

The tuned learning rates are presented in Table 2 .

Note that vanilla SGD requires a very small learning rate, otherwise it suffered from overflow.

Comparison to state-of-the-art.

Plots of the performance of the algorithms on each dataset are displayed in Figure 1 with the relative performance compared to Implicit SGD given in TAB2 .

The Implicit SGD method has the best performance on virtually all datasets.

Not only does it converge faster in the first few epochs, it also converges to the optimal MLE (unlike the biased methods that prematurely plateau).

On average after 50 epochs, Implicit SGD's log-loss is a factor of 4.29 lower than the previous state-of-the-art.

The U-max algorithm also outperforms the previous state-of-theart on most datasets.

U-max performs better than Implicit SGD on AmazonCat, although in general Implicit SGD has superior performance.

Vanilla SGD's performance is better than the previous state-of-the-art but worse than U-max and Implicit SGD.

The difference in performance between vanilla SGD and U-max can largely be explained by vanilla SGD requiring a smaller learning rate to avoid computational overflow.

The sensitivity of each method to the initial learning rate can be seen in Appendix G, where the results of running each method on the Eurlex dataset with learning rates η = 10 0,±1,±2,±3 is presented.

The results are consistent with those in Figure 1 , with Implicit SGD having the best performance for most learning rate settings.

For learning rates η = 10 3,4 the U-max log-loss is extremely large.

This can be explained by Theorem 1, which does not guarantee convergence for U-max if the learning rate is too high.

Comparison of double-sum formulations.

FIG2 illustrates the performance on the Eurlex dataset of U-max using the proposed double-sum in (6) compared to U-max using the double-sum of Raman et al. (2016) in (8) .

The proposed double-sum clearly outperforms for all 10 learning rates η = 10 0,±1,±2,−3,−4 , with its 50 th -epoch log-loss being 3.08 times lower on average.

This supports the argument from Section 2.3 that SGD methods applied to the proposed double-sum have smaller magnitude gradients and converge faster.

In this paper we have presented the U-max and Implicit SGD algorithms for optimizing the softmax likelihood.

These are the first algorithms that require only O(D) computation per iteration (without extra work at the end of each epoch) that converge to the optimal softmax MLE.

Implicit SGD can be efficiently implemented and clearly out-performs the previous state-of-the-art on seven real world datasets.

The result is a new method that enables optimizing the softmax for extremely large number of samples and classes.

So far Implicit SGD has only been applied to the simple softmax, but could also be applied to any neural network where the final layer is the softmax.

Applying Implicit SGD to word2vec type models, which can be viewed as softmaxes where both x and w are parameters to be fit, might be particularly fruitful.

10 The learning rates η = 10 3,4 are not displayed in the FIG2 for visualization purposes.

It had similar behavior as η = 10 2 .

We first establish that the optimal values of u and W are bounded.

Next, we show that within these bounds the objective is strongly convex and its gradients are bounded.

Lemma 1 ( BID16 ).

The optimal value of W is bounded as W * 2 DISPLAYFORM0 Proof.

DISPLAYFORM1 Rearranging gives the desired result.

Lemma 2.

The optimal value of u i is bounded as u * i ≤ B u where B u = log(1 + (K − 1)e 2BxBw ) and B x = max i { x i 2 } Proof.

DISPLAYFORM2 W and u i ≤ B u then f (u, W ) is strongly convex with convexity constant greater than or equal to min{exp(−B u ), µ}.Proof.

Let us rewrite f as DISPLAYFORM3 where θ = (u , w 1 , ..., w k ) ∈ R N +KD with a i and b ik being appropriately defined.

The Hessian of f is DISPLAYFORM4 where e i is the i th canonical basis vector, 0 N is an N -dimensional vector of zeros and 1 KD is a KD-dimensional vector of ones.

It follows that DISPLAYFORM5 W and u i ≤ B u then the 2-norm of both the gradient of f and each stochastic gradient f ik are bounded by DISPLAYFORM6 Proof.

By Jensen's inequality max DISPLAYFORM7 Using the results from Lemmas 1 and 2 and the definition of f ik from (6), DISPLAYFORM8 and for j indexing either the sampled class k = y i or the true label y i , DISPLAYFORM9 we have DISPLAYFORM10

We can write the equation for L(W ) from (3) as (where we have set µ = 0 for notational simplicity), DISPLAYFORM0 Here e i v = v i ∈ R is a variable that is explicitly kept track of with DISPLAYFORM1 k =yi e x i (w k −wy i ) (with exact equality in the limit as t → ∞).

Clearly v i in stochastic composition optimization has a similar role as u i has in our formulation for f in (5).If i, k are sampled with k = y i in stochastic composition optimization then the updates are of the form BID20 w yi = w yi + η t N K e DISPLAYFORM2 where z k is a smoothed value of w k .

These updates have the same numerical instability issues as vanilla SGD on f in (5): it is possible that

Algorithm 1: U-max DISPLAYFORM0 , number of classes K, number of datapoints N , learning rate η t , class sampling probability β k = N n k +(N −n k )(K−1) , threshold parameter δ > 0, bound B W on W such that W 2 ≤ B W and bound B u on u such that u i ≤ B u for i = 1, ..., N Output: DISPLAYFORM1

In this section we will prove the claim made in Theorem 1, that U-max converges to the softmax optimum.

Before proving the theorem, we will need a lemma.

Lemma 5.

For any δ > 0, if u i ≤ log(1+e x i (w k −wy i ) )−δ then setting u i = log(1+e DISPLAYFORM0 Proof.

As in Lemma 3, let θ = (u , w 1 , ..., w k ) ∈ R N +KD .

Then setting u i = log(1 + e x i (w k −wy i ) ) is equivalent to setting θ = θ + ∆e i where e i is the i th canonical basis vector and ∆ = log(1 + e x i (w k −wy i ) ) − u i ≥ δ.

By a second order Taylor series expansion DISPLAYFORM1 for some λ ∈ [0, 1].

Since the optimal value of u i for a given value of W is u * i (W ) = log(1 + k =yi e x i (w k −wy i ) ) ≥ log(1+e x i (w k −wy i ) ), we must have ∇f (θ+∆e i ) e i ≤ 0.

From Lemma 3 we also know that DISPLAYFORM2 Putting in bounds for the gradient and Hessian terms in (10), DISPLAYFORM3 Now we are in a position to prove Theorem 1.Proof of Theorem 1.

Let θ (t) = (u (t) , W (t) ) ∈ Θ denote the value of the t th iterate.

Here Θ = {θ : W 2 2 ≤ B 2 W , u i ≤ B u } is a convex set containing the optimal value of f (θ).

DISPLAYFORM4 If indices i, k are sampled for the stochastic gradient and u i ≤ log(1 + e x i (w k −wy i ) ) − δ, then the value of f at the t + 1 st iterate is bounded as DISPLAYFORM5 .

Taking expectations with respect to i, k, DISPLAYFORM6 Finally let P denote the projection of θ onto Θ. Since Θ is a convex set containing the optimum we have f (P (θ)) ≤ f (θ) for any θ, and so DISPLAYFORM7 which shows that the rate of convergence in expectation of U-max is at least as fast as that of standard SGD.

Proof of Theorem 2.

Let f (θ, ξ) be m-strongly convex for all ξ.

The vanilla SGD step size is η t ∇f (θ (t) , ξ t ) 2 where η t is the learning rate for the t th iteration.

The Implicit SGD step size is η t ∇f (θ (t+1) , ξ t ) 2 where DISPLAYFORM0 )/η t and so it must be the case that ∇f (θ DISPLAYFORM1 2 .

Our desired result follows: DISPLAYFORM2 where the first inequality is by Cauchy-Schwarz and the second inequality by strong convexity.

In this section we will derive the updates for Implicit SGD.

We will first consider the simplest case where only one datapoint (x i , y i ) and a single class is sampled in each iteration with no regularizer.

Then we will derive the more complicated update for when there are multiple datapoints and sampled classes with a regularizer.

F.1 SINGLE DATAPOINT, SINGLE CLASS, NO REGULARIZER Equation (6) for the stochastic gradient for a single datapoint and single class with µ = 0 is DISPLAYFORM0 The Implicit SGD update corresponds to finding the variables optimizing DISPLAYFORM1 where η is the learning rate and the tilde refers to the value of the old iterate (Toulis et al., 2016, Eq. 6) .

Since f ik is only a function of u i , w k , w yi the optimization reduces to DISPLAYFORM2 The optimal value of w k , w yi must deviate from the old valuew k ,w yi in the direction of x i .

Furthermore we can observe that the deviation of w k must be exactly opposite that of w yi , that is: DISPLAYFORM3 for some a ≥ 0.

The optimization problem reduces to min ui,a≥0 DISPLAYFORM4 We'll approach this optimization problem by first solving for a as a function of u i and then optimize over u i .

Once the optimal value of u i has been found, we can calculate the corresponding optimal value of a. Finally, substituting a into (11) will give us our updated value of W .

We solve for a by setting its derivative equal to zero in (12) DISPLAYFORM0 The solution for a can be written in terms of the principle branch of the Lambert W function P , DISPLAYFORM1 Substituting the solution to a(u i ) into FORMULA0 , we now only need minimize over u i : DISPLAYFORM2 where we used the fact that e −P (z) = P (z)/z.

The derivative with respect to u i in FORMULA0 is DISPLAYFORM3 where to calculate ∂ ui a(u i ) we used the fact that ∂ z P (z) = P (z) z(1+P (z)) and so DISPLAYFORM4 Bisection method for u i We can solve for u i using the bisection method.

Below we show how to calculate the initial lower and upper bounds of the bisection interval and prove that the size of the interval is bounded (which ensures fast convergence).Start by calculating the derivative in (16) at u i =ũ i .

If the derivative is negative then the optimal u i is lower bounded byũ i .

An upper bound is provided by DISPLAYFORM5 In the first inequality we set a(u i ) = 0, since by the envelop theorem the gradient of u i is monotonically increasing in a. In the second inequality we used the assumption that u i is lower bounded byũ i .

Thus if the derivative in (16) is negative at DISPLAYFORM6 then the size of the interval must be less than log(2), sinceũ i ≥ 0.

Otherwise the gap must be at most log(2(K −1)e x i (w k −wy i ) )

−ũ i = log(2(K −1))+x i (w k −w yi )−ũ i .

Either way, the gap is upper bounded by log(2(K − 1)) + |x i (w k −w yi ) −ũ i |.

Now let us consider if the derivative in (16) is positive at u i =ũ i .

Then u i is upper bounded byũ i .

Denoting a as the optimal value of a, we can lower bound u i using (12) DISPLAYFORM7 where the first inequality comes dropping the (u i −ũ i ) 2 term due to the assumption that u i <ũ i .

Recall FORMULA0 , DISPLAYFORM8 The solution for a is strictly monotonically increasing as a function of the right side of the equation.

Thus replacing the right side with an upper bound on its value results in an upper bound on a .

Substituting the bound for u i , DISPLAYFORM9 (18) Substituting this bound for a into (17) yields DISPLAYFORM10 Thus if the derivative in (16) is postive at u i =ũ i then log(K − 1) + x i (w k −w yi ) − 2ηN x i 2 2 ≤ u i ≤ũ i .

The gap between the upper and lower bound isũ i −x i (w k −w yi )+2ηN x i 2 2 −log(K−1).

In summary, for both cases of the sign of the derivative in (16) at u i =ũ i we are able to calculate a lower and upper bound on the optimal value of u i such that the gap between the bounds is at most |ũ i − x i (w k −w yi )| + 2ηN x i 2 2 + log(K − 1).

This allows us to perform the bisection method where for > 0 level accuracy we require only log 2 ( −1 )+log 2 (|ũ i −x i (w k −w yi )|+2ηN x i 2 2 + log(K − 1)) function evaluations.

Here we will prove that the step size magnitude of Implicit SGD with a single datapoint and sampled class with respect to w is bounded as O(x i (w k −w yi ) −ũ i ).

We will do so by considering the two cases u i ≥ũ i and u i <ũ i separately, where u i denotes the optimal value of u i in the Implicit SGD update andũ i is its value at the previous iterate.

Case: u i ≥ũ i Let a denote the optimal value of a in the Implicit SGD update.

From (14) a = a(u i ) = P (e DISPLAYFORM0 2 ) ).

Now using the fact that P (z) = O(log(z)), DISPLAYFORM1

Putting together the two cases, DISPLAYFORM0 The actual step size in w is ±a DISPLAYFORM1

The Implicit SGD update when there are multiple datapoints, multiple classes, with a regularizer is similar to the singe datapoint, singe class, no regularizer case described above.

However, there are a few significant differences.

Firstly, we will require some pre-computation to find a low-dimensional representation of the x values in each mini-batch.

Secondly, we will integrate out u i for each datapoint (not w k ).

And thirdly, since the dimensionality of the simplified optimization problem is large, we'll require first order or quasi-Newton methods to find the optimal solution.

The first step is to define our mini-batches of size n.

We will do this by partitioning the datapoint indices into sets S 1 , ..., S J with S j = {j : = 1, ..., n} for j = 1, ..., N/n , S J = {J : = 1, ..., N mod n}, S i ∩ S j = ∅ and ∪ J j=1 S j = {1, ..., N }.Next we define the set of classes C j which can be sampled for the j th mini-batch.

The set C j is defined to be all sets of m distinct classes that are not equal to any of the labels y for points in the mini-batch, that is, DISPLAYFORM0 Now we can write down our objective from (5) in terms of an expectation of functions corresponding to our mini-batches: DISPLAYFORM1 where j is sampled with probability p j = |S j |/N and C is sampled uniformly from C j and DISPLAYFORM2 The value of the regularizing constant β k is such that E[I[k ∈ C ∪ S j ]β k ] = 1, which requires that DISPLAYFORM3

The Implicit SGD update corresponds to solving DISPLAYFORM0 where η is the learning rate and the tilde refers to the value of the old iterate (Toulis et al., 2016, Eq. 6) .

Since f j,C is only a function of u Sj = {u i : i ∈ S j } and W j,C = {w k : k ∈ S j ∪ C} the optimization reduces to DISPLAYFORM1 The next step is to analytically minimize the u Sj terms.

The optimization problem in (21) decomposes into a sum of separate optimization problems in u i for i ∈ S j , DISPLAYFORM2 Setting the derivative of u i equal to zero yields the solution DISPLAYFORM3 where P is the principle branch of the Lambert W function.

Substituting this solution into our optimization problem and simplifying yields DISPLAYFORM4 where we have used the identity e −P (z) = P (z)/z.

We can decompose (19) into two parts by splitting W j,C = W j,C + W ⊥ j,C , its components parallel and perpendicular to the span of {x i : i ∈ S j } respectively.

Since the leading term in (19) only depends on W j,C , the two resulting sub-problems are DISPLAYFORM5 Let us focus on the perpendicular component first.

Simple calculus yields the optimal value w DISPLAYFORM6 Moving onto the parallel component, let the span of {x i : i ∈ S j } have an orthonormal basis 11 DISPLAYFORM7 D×n with x i = V j b i for some b i ∈ R n .

With this basis we can write DISPLAYFORM8 n which reduces the parallel component optimization problem to DISPLAYFORM9 where A j,C = {a k : k ∈ S j ∪ C} ∈ R (n+m)×n and DISPLAYFORM10 The e b i (a k −ay i ) factors come from DISPLAYFORM11 since V j is an orthonormal basis.

To optimize (21) we need to be able to take the derivative: DISPLAYFORM0 where we used that ∂ z P (z) = P (z) z(1+P (z)) and e −P (z) = P (z)/z.

To complete the calculation of the derivate we need, DISPLAYFORM1 .In order to calculate the full derivate with respect to A j,C we need to calculate b i a k for all i ∈ S j and k ∈ S j ∪ C. This is a total of n(n + m) inner products of n-dimensional vectors, costing O(n 2 (n + m)).

To find the optimum of (21) we can use any optimization procedure that only uses gradients.

Since (21) is strongly convex, standard first order methods can solve to accuracy in O(log( −1 )) iterations (Boyd & Vandenberghe, 2004, Sec. 9.3) .

Thus once we can calculate all of the terms in (21), we can solve it to accuracy in runtime O(n 2 (n + m) log( −1 )).Once we have solved for A j,C , we can reconstruct the optimal solution for the parallel component of w k as w k =w k + V j a k .

Recall that the solution to the perpendicular component is w DISPLAYFORM2 If the features x i are sparse, then we'd prefer to do a sparse update to w, saving computation time.

We can achieve this by letting DISPLAYFORM3 where γ k is a scalar and r k a vector.

Updating w k =w k + V j a k + 1 1+µβ k /2w ⊥ k is equivalent to γ k =γ k · 1 1 + µβ k /2 r k =r k + µβ k /2 ·r k +γ DISPLAYFORM4 Since we only update r k along the span of {x i : i ∈ S j }, its update is sparse.

F.3.4 RUNTIME There are two major tasks in calculating the terms in (21).

The first is to calculate x iw k for i ∈ S j and k ∈ S j ∪ C. There are a total of n(n + m) inner products of D-dimensional vectors, costing O(n(n + m)D).

The other task is to find the orthonormal basis V j of {x i : i ∈ S j }, which can be achieved using the Gram-Schmidt process in O(n 2 D).

We'll assume that {V j : j = 1, ..., J} is computed only once as a pre-processing step when defining the mini-batches.

It is exactly because calculating {V j : j = 1, ..., J} is expensive that we have fixed mini-batches that do not change during the optimization routine.

Adding the cost of calculating the x iw k inner products to the costing of optimizing (21) leads to the claim that solve the Implicit SGD update formula to accuracy in runtime O(n(n + m)D + n 2 (n + m) log( −1 )) = O(n(n + m)(D + n log( −1 ))).

As was the case in Section F.1, it is important to initialize the optimization procedure at a point where the gradient is relatively small and can be computed without numerical issues.

These numerical issues arise when an exponent x i (w k −w yi ) −ũ i + b i (a k − a yi ) 0.

To ensure that this does not occur for our initial point, we can solve the following linear problem, 13 R = min DISPLAYFORM0 Note that if k = y i then the constraint 0 ≥ x i (w k −w yi )−ũ i +b i (a k −a yi ) = −ũ i is automatically fulfilled sinceũ i ≥ 0.

Also observed that setting a k = −V jw k satisfies all of the constraints, and so Putting the bounds together we have that the optimal value of (21) is upper bounded by its value at the solution to (22), which in turn is upper bounded by n(1 + P (Kηp This bound is guarantees that our initial iterate will be numerically stable.

Here we present the results of using different learning rates for each algorithm applied to the Eurlex dataset.

In addition to the Implicit SGD, NCE, IS, OVE and U-max algorithms, we also provide results for NCE with n = 1, m = 1, denoted as NCE (1,1) .

NCE and NCE (1,1) have near identical performance.

@highlight

Propose first methods for exactly optimizing the softmax distribution using stochastic gradient with runtime independent on the number of classes or datapoints.