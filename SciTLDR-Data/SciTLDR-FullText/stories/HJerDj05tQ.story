Optimization on manifold has been widely used in machine learning, to handle optimization problems with constraint.

Most previous works focus on the case with a single manifold.

However, in practice it is quite common that the optimization problem involves more than one constraints, (each constraint corresponding to one manifold).

It is not clear in general how to optimize on multiple manifolds effectively and provably especially when the intersection of multiple manifolds is not a manifold or cannot be easily calculated.

We propose a unified algorithm framework to handle the optimization on multiple manifolds.

Specifically,  we integrate information from multiple manifolds and move along an ensemble direction by viewing the information from each manifold as a drift and adding them together.

We prove the convergence properties of the proposed algorithms.

We also apply the algorithms into  training neural network with batch normalization layers and achieve preferable empirical results.

Machine learning problem is often formulated as optimization problem.

It is common that the optimization problem comes with multiple constraints due to practical scenarios or human prior knowledge that adding some of them help model achieve a better result.

One way to handle these constraints is adding regularization terms to the objective, such as the 1 and 2 regularization.

However, it is hard to adjust the hyper-parameters of the regularization terms to guarantee that the original constraints get satisfied.

Another way to deal with the constraints is to optimize on manifolds determined by the constraints.

Then the optimization problem becomes unconstrained on the manifold, which could be easy to solve technically.

Furthermore, optimization on manifold indicates optimizing on a more compact space, and may bring performance gain when training neural networks, e.g., BID10 BID3 .Most previous works on manifold optimization focus on a single manifold BID13 .

However, in practice, we often face more than one constraints, each of them corresponding to one manifold.

If we still solve the optimization problem with multiple constraints by method on manifold, we need to handle it on the intersection of multiple manifolds, which may no longer be a manifold BID11 .

Due to this, traditional optimization methods on manifold does not work in this case.

In this paper, we consider the problem of optimization on multiple manifolds.

Specifically, the problem is written as arg min DISPLAYFORM0 where each M i is a manifold.

We propose a method solving this problem by choosing the moving direction as −∇f (x)(on manifold is −gradf (x)) with several drifts which are derived from the descent information on other manifolds.

By this method, we get sequence that has information from all manifolds.

There are several articles discussing the problem of optimization on manifold.

Most of them focus on a single manifold.

Readers can find a good summary about this topic and the advantages of op-timization on manifold in BID1 .

Recently, popular first order algorithms in Euclidean space are studied in the manifold setting, e.g., the convergence of gradient descent BID2 , sub-gradient method BID5 , stochastic variance reduction gradient (SVRG) and the gradient descent with momentum .Riemann approaches BID3 BID6 have also been applied to train deep neural network by noticing that the parameters of the neural network with batch normalization live on Grassmann manifold and Oblique manifold, respectively.

(1) This paper introduces an algorithm to deal with optimization with multiple manifolds.

The algorithm adds drifts obtained from other manifolds to the moving direction, in order to incorporate the information from multiple manifolds during the optimization process.(2) We prove the convergence of this algorithm under a very general framework.

The proof is also applicable to the convergence of many other algorithms including gradient descent with momentum and gradient descent with regularization.

Moreover, our proof does not depend on the choices of Retr x on the manifold.

The specific definition of manifold M can be found in any topology book.

For better understanding, we introduce several properties of manifold here.

A manifold is a subspace of R n .

For a given point x ∈ M, it has a tangent space T x M which is a linear space but M may not.

For gradient descent method, the iterates are generated via DISPLAYFORM0 η is step length.

However, the iterate generated by gradient descent may not on manifold anymore because manifold is not a linear space.

To fix this, we introduce a retraction function Retr x (η) : T x M → M to determine how point moves on manifold.

Specifically, if M is R n , the Retr x becomes x + η.

We can consider η in Retr x as the moving direction of the iterating point.

Then, the gradient descent on manifold BID2 ) is given by DISPLAYFORM1 where gradf (x) is Riemannian gradient.

Riemannian gradient is the orthogonal projection of gradient ∇f (x) to tangent space T x M as ∇f (x) may not in tangent space T x M and the moving direction on manifold is only decided by the vector in T x M. All of notations related to manifold can be referred to BID1 ).We next use a lemma to describe a property of the minimum point of the problem arg min x∈M f (x), which is a special case of Yang et al., 2014, Corollary 4.2 and BID2 , Proposition 1.Lemma 2.1 Let x be a local optimum for the optimization problem arg min x∈M f (x), which means there exists a neighborhood DISPLAYFORM2 We see that gradf (x) plays a role of ∇f (x) on manifold.

Similar as BID2 discussed, we assume function has the property of Lipschtiz gradient.

The definition of Lipschtiz gradient is Definition 2.1 (Lipschtiz gradient) For any two points x, y in the manifold M, f (x) satisfy: DISPLAYFORM3 Then we say that f satisfies the Lipschtiz gradient condition.

We next introduce a condition that guarantees the convergence of iterative algorithms.

Definition 2.2 (Descent condition) For a sequence {x k } and a k > 0, if DISPLAYFORM4 then we say the sequence satisfies the descent condition.

First, we introduce a theorem to describe the convergence when the object function f is lower finite, i.e., there exists a f * such that f (x) ≥ f * > −∞ for all x, and the iterates satisfy descent condition.

This theorem plays a key role in proof of the rest theorems.

Theorem 2.1 If f is lower finite, and the iteration sequence {x k } satisfies the descent condition for any given {a k }, where each a k > 0.

Then lim inf k→∞ a k gradf (x k ) = 0 Proof 1 The proof is available in Supplemental.

For better presentation, we first describe the algorithm under the circumstance of two manifolds.

Considering the objective function f constrained on two manifolds M 1 , M 2 , we aim to find the minimum point on M 1 M 2 .

Since M 1 M 2 may not be a manifold, previous methods on manifold optimization cannot apply directly.

We propose a method that integrates information from two manifolds over the optimization process.

Specifically, we construct two sequences {x k }, {y k }, each on one manifold respectively.

We add a drift which contains information from the other manifold to the original gradient descent on manifold (equation 2).

The updating rules are DISPLAYFORM0 DISPLAYFORM1 If b k = 0 in (equation 3) and (equation 4), the updating rules reduce to normal gradient descent on manifold equation 2.

The drift h k is in the tangent space T x M of each manifold, which represents information from the other manifold.

We call this algorithm gradient descent on manifold with drifting, whose procedure is described in Algorithm 1.Algorithm 1 Gradient descent with drift on manifold DISPLAYFORM2 We next present the convergence theorem of this algorithm, which illustrates how we set a k and b k in the algorithm.

Theorem 2.2 For function f (x) is lower finite, and Lipschtiz gradient.

If we construct the sequence {x k } like equation FORMULA6 , and for any 0 < δ < 2, we control δ ≤ a k ≤ 2.

Setting DISPLAYFORM3 then x k convergence to a local minimizer.

Proof 2 The proof is based on construction of the descent condition (equation 12) and is available in Supplemental.

From the construction of b k , we can see that the smaller the correlation between gradf (x k ) and h k is, the smaller effect the information from M 2 brings.

In fact, we set h DISPLAYFORM4 , where P(1)x k is the projection matrix to tangent space DISPLAYFORM5 k which exchanges x k and P(1)x k with y k and Py k (projection matrix of tangent space T y k M 2 ).

The drift intuitively gives x k a force moving towards the minimizer on the other manifold.

If the two manifolds are R n , then x k and y k are symmetry with each other.

We have DISPLAYFORM6 If the equation system is stable and x 0 , y 0 are mutually close, the distance between x k and y k will be small when k → ∞. By Schwarz inequality, we see b k ≤ 2(1 − a k ).

Since h k = gradf (x k ) , the scale of the drift is the same as the original Riemannian gradient.

Hence, information from another manifold will not affect much, when the points x k and y k are close to a minimizer.

We can control the contribution of the information from the other manifold by adjusting a k .

For instance, a k = 1 indicates we do not integrate information from the other manifold.

We can also prove the convergence rate of this algorithm.

DISPLAYFORM7 Proof 3 The proof is delegated to Supplemental.

Theorem 2.3 states the number of iterations we need to achieve a specific accuracy.

Here we can adjust a k as long as δ < a k < 2.

In this subsection, we describe our algorithm for the case with multiple (more than 2) manifolds.

Suppose we have n manifolds, M 1 , · · · , M n , and sequence on manifold M i is denoted as {x DISPLAYFORM0 In the following, we use sequence {x(1) k } on M 1 as an example, and other sequences on other manifolds can be derived accordingly.

Let g DISPLAYFORM1 Then let the updating rule be DISPLAYFORM2 Since f satisfies Lipschtiz gradient condition(2.1), we have DISPLAYFORM3 We choose a(1) DISPLAYFORM4 The way of choosing a DISPLAYFORM5 k ) ij , i, j from 2 to n, and α k = (a BID7 .

It transforms the input value to a neuron from z = w T x to DISPLAYFORM6 DISPLAYFORM7 We can calculate the derivative as follows DISPLAYFORM8 For any a = 0, a ∈ R, we see that BN (w) = BN (aw) and DISPLAYFORM9 .

These equations mean that after a batch normalization, the scale of parameter has no relationship with the output value, but scale of gradient is opposite with the scale of parameter.

BID3 have discussed that batch normalization could have an adverse effect in terms of optimization since there can be an infinite number of networks, with the same forward path but different scaling, which may converge to different local optima owing to different gradients.

To avoid this phenomenon, we can eliminate the effect of scale by considering the weight w on the Grassmann manifold or Oblique manifold.

On these two manifolds, we can ignore the scale of parameter.

BID3 ; BID6 respectively discuss that BN (w) has same image space on G(1, n) and St(n, 1) as well as R n , where G(1, n) is a Grassmann manifold and St(n, 1) is an Oblique manifold.

Due to these, we can consider applying optimization on manifold to batch normalization problem.

However, the property of these two manifold implies that we can actually optimize on the intersection of two manifolds.

Since optimization on a manifold rely on Riemannian gradient gradf (x) and Retr x , for a specific Retr x (9) of Grassmann manifold G(1, n), we get a unit point x when η = −gradf (x) = 0 in formula (9).

The condition gradf (x) = 0 means we obtain a unit critical point on Grassmann manifold which is also on Oblique manifold.

The specific discussion of Grassmann manifold and Oblique manifold can be found in BID1 .

G(1, n) is a quotient manifold defined on a vector space, it regards vector with same direction as same element.

For example (1, 1, 1) and (10, 10, 10) correspond to same element.

We represent elements on G(1, n) with same direction by choosing one of them as representation element.

Oblique manifold is given by St(n, p) = {X ∈ R n×p : ddiag(X T X) = I p }, where ddiag(·) is diagonal matrix of a matrix.

We have discussed above that iteration point on G(1, n) would be a unit point when it's a local minimizer.

Due to this, the local minimizer we find is actually live on the intersection of St(n, 1) and G(1, n).

Hence, training neural network with batch normalized weights can be converted to the problem arg min DISPLAYFORM10 Let Riemannian gradient be projection of ∇f (x) to tangent space of x. On G(1, n), we have DISPLAYFORM11 gradf (x) = P(1) DISPLAYFORM12

(1) DISPLAYFORM0 On St(n, 1), we have P (2) DISPLAYFORM1 x (η) = x + η x + η the P x is the projection matrix onto the tangent space at x. These results can be derived from the general formulas from BID0 and BID4 .In backward process of training neural network, weight parameter of each layer is a matrix.

Hence, we get gradient to a matrix in every layer.

To make calculation easier, we treat the gradient matrix and parameters matrix as vector.

For example a m × n gradient matrix can be viewed as a m × n dimensional vector.

Then we apply Algorithm 1 to update parameters, which means we optimize on a product manifold DISPLAYFORM2 k i is number of parameters for the i-th hidden layer, and n is number of hidden layers.

We need to operate algorithm for parameter vector on each hidden layer.

In other words, we update parameters layer by layer.

In this section, we use data set CIFAR-10 and CIFAR-100 BID8 ) to test our algorithm.

These two data sets are color images respectively have 10 and 100 classes, each of them has 50,000 training images and 10,000 test images.

The deep neural network we used is WideResNet BID15 , it output a vector which describe the probability of a data divided into each class.

In every hidden layer of neural network, we apply batch normalization to weight parameters and treat them as a vector.

We have already discussed that minimizers of a neural network with batch normalized weights live on the intersection of Grassmann manifolds and Oblique manifold.

Hence, we can train neural network with batch normalized weights by our algorithm(1).

The biases of every hidden layer is unrelated to batch normalization and are updated by SGD.

For every training step, we calculate mean loss 1 S xi∈S l(f (x i , θ), y i ) of a mini batch to substitute the real loss function E x [l(f (x, θ), y)], where S is batch size.

The process of algorithm on two manifolds follows Algorithm 1, where the two manifolds are G(1, n) and St(n, 1), respectively.

In Algorithm 1, we choose DISPLAYFORM0 In the updating rules of x k and y k , we add a norm-clip to vectors (a DISPLAYFORM1 k ).

Then we times η to the two vectors, where η is the learning rate.

In the experiments, we compare three methods: 1) stochastic gradient descent on manifold with drifting (Drift-SGDM), 2) stochastic gradient descent on manifold BID2 (SGDM), and 3) stochastic gradient descent (SGD).

In Algorithm 1, we can get two sequences each corresponding to a model on a manifold.

We predict output class by adding two output vectors of two models and choosing the biggest as prediction class.

For Drift-SGDM (Algorithm 1), we set δ = 0.9 and initial learning rate η m = 0.4 for weights parameters which is multiplied by 0.4 at 60, 120, and 160 epochs.

Initial learning rate η for biases is 0.01 which is multiplied by 0.4 at 60, 120, and 160 epochs.

Norm clip is 0.1.

Training batch size is 128.

The number of training epochs is 200.For SGDM, we choose a = 1 in Algorithm 1.

The other settings are the same as Drift-SGDM.

That a = 1 in Algorithm 1 means that SGDM optimizes on each manifold individually We set SGD as baseline.

The learning rate is 0.2 which is multiplied by 0.2 at epoch 60,120 and 160.

Weight decay is set as 0.0005, but we do not apply weight decay for algorithms on manifold.

All other settings are the same as the above two algorithms.

About Drift-SGDM and SGDM, the loss is achieved from the average of two model.

The parameter scale of the two model can be different, because they respectively live on Grassmann manifold and Oblique manifold.

Due to this, the comparing between Drift-SGDM and SGDM is more reasonable.

We also give the accuracy curve and a tubular of accuracy rate on test sets to validate our algorithms.

We see that our algorithm perform better on larger neural network.

Our algorithm does not have regularization term, and it does not perform well in the aspect of generalization.

We can actually add a regularization term like in BID3 ) to achieve better generalization.

We choose δ in Algorithm 1 as 0.9.

Since b DISPLAYFORM2 k ) where i = 1, 2 as we have discussed in section 2, we see drift term b DISPLAYFORM3 k in Algorithm 1 doesn't affect much to iteration point.

We can actually set a smaller δ to enhance the influence of drift term b DISPLAYFORM4

In this paper, we derive an intuitively method to approach optimization problem with multiple constraints which corresponds to optimizing on the intersection of multiple manifolds.

Specifically, the method is integrating information among all manifolds to determine minimum points on each manifold.

We don't add extra conditions to constraints of optimization problem, as long as each constraint can be converted to a manifold.

In the future, we may add some conditions to manifolds which derive a conclusion that minimum points on each manifold achieved by our algorithm are close with other.

If this conclusion is established, the problem of optimization on intersection of multiple manifolds is solved.

According to the updating rule (equation 3), we can derive many other algorithms, because the drift h k in (equation 3) is flexible.

On the other hand, Retr x on our algorithm does not limit to a specific one.

Since there are some results for Retr x = Exp x , for example Corollary 8 in , we may get more elegant results by using Exp x as retraction function in our algorithm.

The manifolds we encounter in optimization are mainly embedded sub-manifold and quotient manifold BID1 .

Embedded sub-manifold is F −1 (y) for a smooth function F : M 1 → M 2 , where M 1 , M 2 are two manifolds and y ∈ M 2 .

Quotient manifold is a quotient topology space generalized by a specific equivalence relationship ∼. In this paper, we use Oblique manifold and Grassmann manifold which are embedded sub-manifold and quotient manifold respectively.

The difficulty we faced in optimization on manifold is calculating tangent space T x M and Riemannian gradient gradf (x).

Giving a exact formula of a tangent space T x M is not a easy problem.

On the other hand, since Riemannian gradient is ∇f (x) projected to a tangent space T x M, finding projection matrix to a specific space T x M is nontrivial.

In this section, we study the frame work of gradient descent with drift.

In a special case, we regard R n as a manifold.

Then, Rienmann gradient gradf (x) = ∇f (x), tangent space T x M = R n and Retr x (η) = x + η.

In Algorithm (1), we set DISPLAYFORM0 Then we have DISPLAYFORM1 which is exactly a kind of gradient descent with momentum.

And this algorithm is convergence as we proved.

On the other hand, if choosing h k as gradient of a regularization term R(x) on x k .

For example, h k becomes 2x k when R(x) = x 2 .

The iteration point in Algorithm FORMULA0 is achieved by gradient descent with regularization term.

The drift in (equation 3) we have discussed is non-stochastic.

But actually, we can change the drift as a stochastic term to construct a non-descent algorithm.

Meanwhile, stochastic drift gives iteration sequence ability of jumping from local minimizer.

The update rule is DISPLAYFORM0 where ξ k is a random vector with mean vector µ, covariance matrix Σ. The process of this algorithm is Algorithm 2.Algorithm 2 Non-descent method with stochastic noise Input 0 < δ < 2, x 0 ∈ M, Retr x , ε > 0 k → 1 while gradf (x) > ε do Sample ξ k with mean vector µ and covariance matrix DISPLAYFORM1 We give convergence theorem of Algorithm 2.

The proof implies that this algorithm is non-descent, it also shows how we set a k and b k .Theorem A.1 For function f (x) ≥ f * > −∞, and Lipschtiz gradient.

If we construct the sequence DISPLAYFORM2 where 0 < δ < 2, we have lim inf k→∞ gradf (x k ) 2 = 0.In this theorem, b k control the speed of back fire.

The noise ξ k in (equation 10) has small effect to iteration process when k is large, because sequence is about to be stable after enough iterations.

But in beginning of iteration procedure, noise ξ k effects much which give iteration sequence ability of jumping from local minimizer.

In this section, we give proof of theorems in this paper.

The proof of Theorem 2.1 is Proof 4 (proof of Theorem 2.1) According to definition 2.2 of descent condition, we have DISPLAYFORM0 for any k. Since f is lower finite, we have DISPLAYFORM1

Proof 5 (proof of Theorem 2.2) Since f satisfy Lischtiz gradient(2.1), we have DISPLAYFORM0 By the definition of b k , we got DISPLAYFORM1 Before proof Theorem A.1, we need two lemmas.

Lemma B.1 A random vector with Ex = µ and Covx = Σ. Then for any symmetric matrix A, we have E(x T Ax) = µ T Aµ + tr(AΣ).This lemma can be derived from BID12 Here we use Rayleigh theorem of theorem 2.4.

BID12 Proof 8 (proof of Theorem A.1) Since P x k is a projection matrix, which is a symmetric idempotent matrix.

Because f satisfies Lipschtiz gradient(2.1), we have DISPLAYFORM2 where ε i = (P xi gradf (x i )) T ξ i = gradf (x i )) T ξ i , η i = ξ T i P T xi P xi ξ i = ξ T i P xi ξ i .

Due to the two random variables, algorithm (2) is not necessary descent.

Σ is a symmetric positive definite matrix.

By Schwarz equality and definition of a i , we have DISPLAYFORM3 By Fatou's lemma, we have DISPLAYFORM4

@highlight

This paper introduces an algorithm to handle optimization problem with multiple constraints under vision of manifold.