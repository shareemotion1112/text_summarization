Auto-encoders are commonly used for unsupervised representation learning and for pre-training deeper neural networks.

When its activation function is linear and the encoding dimension (width of hidden layer) is smaller than the input dimension, it is well known that auto-encoder is optimized to learn the principal components of the data distribution (Oja1982).

However, when the activation is nonlinear and when the width is larger than the input dimension (overcomplete), auto-encoder behaves differently from PCA, and in fact is known to perform well empirically for sparse coding problems.



We provide a theoretical explanation for this empirically observed phenomenon, when rectified-linear unit (ReLu) is adopted as the activation function and the hidden-layer width is set to be large.

In this case, we show that, with significant probability, initializing the weight matrix of an auto-encoder by sampling from a spherical Gaussian distribution followed by stochastic gradient descent (SGD) training converges towards the ground-truth representation for a class of sparse dictionary learning models.

In addition, we can show that, conditioning on convergence, the expected convergence rate is O(1/t), where t is the number of updates.

Our analysis quantifies how increasing hidden layer width helps the training performance when random initialization is used, and how the norm of network weights influence the speed of SGD convergence.

d .

An auto-encoder can be decomposed into two parts, encoder and decoder.

The encoder can be viewed as a composition function s e • a e : R d → R n ; function a e : R d → R n is defined as a e (x) := W e x + b e with W e ∈ R n×d , b e ∈ R n W e and b e are the network weights and bias associated with the encoder.

s e is a coordinate-wise activation function defined as s e (y) j := s(y j ) where s : R → R is typically a nonlinear functionThe decoder takes the output of encoder and maps it back to R d .

Let x e := s e (a e (x)).

The decoding function, which we denote asx, is defined aŝ DISPLAYFORM0 where (W d , b d ) and s d are the network parameters and the activation function associated with the decoder respectively.

Suppose the activation functions are fixed before training.

One can viewx as a reconstruction of the original signal/data using the hidden representation parameterized by (W e , b e ) and (W d , b d ).

The goal of training an auto-encoder is to learn the "right" network parameters, (W e , b e , W d , b d ), so that x has low reconstruction error.

Weight tying A folklore knowledge when training auto-encoders is that, it usually works better if one sets W d = W T e .

This trick is called "weight tying", which is viewed as a trick of regularization, since it reduces the total number of free parameters.

With tied weights, the classical auto-encoder is simplified asx(s e (a e (x))) = s d (W T s e (W x + b e ) + b d )In the rest of the manuscript, we focus on weight-tied auto-encoder with the following specific architecture:x W,b (x) = W T s ReLu (a(x)) = W T s ReLu (W x + b) with s ReLu (y) i := max{0, y i }Here we abuse notation to usex W,b to denote the encoder-decoder function parametrized by weights W and bias b. In the deep learning community, s ReLu is commonly referred to as the rectified-linear (ReLu) activation.

Reconstruction error A classic measure of reconstruction error used by auto-encoders is the expected squared loss.

Assuming that the data fed to the auto-encoder is i.i.d distributed according to an unknown distribution, i.e., x ∼ p(x), the population expected squared loss is defined as DISPLAYFORM1 Learning a "good representation" thus translates to adjusting the parameters (W, b) to minimize the squared loss function.

The implicit hope is that the squared loss will provide information about what is a good representation.

In other words, we have a certain level of belief that the squared loss characterizes what kind of network parameters are close to the parameters of the latent distribution p(x).

This unwarranted belief leads to two natural questions that motivated our theoretical investigation:• Does the global minimum (or any of global minima, if more than one) of L(W, b) correspond to the latent model parameters of distribution p(x)?• From an optimization perspective, since L(W, b) is non-convex in W and is shown to have exponentially many local minima Safran & Shamir (2016) , one would expect a local algorithm like stochastic gradient descent, which is the go-to algorithm in practice for optimizing L(W, b), to be stuck in local minima and only find sub-optimal solutions.

Then how should we explain the practical observation that auto-encoders trained with SGD often yield good representation?Stochastic-gradient based training Stochastic gradient descent (SGD) is a scalable variant of gradient descent commonly used in deep learning.

At every time step t, the algorithm evaluates a stochastic gradient g(·) of the population loss function with respect to the network parameters using back propagation by sampling one or a mini-batch of data points.

The weight and bias update has the following generic form DISPLAYFORM2 where η t w and η t b are the learning rates for updating W and b respectively, typically set to be a small number or a decaying function of time t.

The unbiased gradient estimate g(W t ) and g(b t ) can be obtained by differentiating the empirical loss function defined on a single or a mini-batch of size m, Then the stochastic or mini-batch gradient descent update can be written as DISPLAYFORM3 DISPLAYFORM4 n (width of hidden layer)Max-norm regularization A common trick called "max-norm regularization" Srivastava et al. (2014) or "weight clipping" is used in training deep neural networks.

1 In particular, after each step of stochastic gradient descent, the updated weights is forced to satisfy DISPLAYFORM5 for some constant c. This means the row norm of the weights can never exceed the prefixed constant c. In practice, whenever W i, 2 > c, the max-norm constraint is enforced by projecting the weights back to a ball of radius c.

In this section, we start by defining notations.

Then we introduce a norm-controlled variant of SGD algorithm that operates on the auto-encoder architecture formalized in (1).

Finally, we introduce assumptions on the data generating model.

We use the same notation for network parameters W, b, and for activation a(·), as in Section 1.

We use s(·) as a shorthand for the ReLu activation function s ReLu (·).

We use capital letters, such as W or F , either to denote a matrix or an event; we use lower case letters, such as x, for vectors.

W T denotes the transpose of W .

We use W s, to denote the s-th row of W .

When a matrix W is modified through time, we let W t denote the state of the matrix at time t, and W t s, for the state of the corresponding row.

We use · for l 2 -norm of vectors and | · | for absolute value of real numbers.

Matrix-vector multiplication between W and x (assuming their dimensions match) is denoted by W x. Inner product of vectors x and y is denoted by x, y .Organization of notations Throughout the manuscript, we introduce notations that can be divided into "model", "algorithm", and "analysis" categories according to their utility.

They are organized in TAB0 to help readers interpreting our results.

For example, If a reader is interested in knowing how to apply our result to parameter tuning in training auto-encoders, then she might ignore the auxiliary notations and only refer to algorithmic parameters and model parameters in TAB0 , and examine how does the setting of the former is influenced by the latter in Theorem 1.

We assume that the algorithm has access to i.i.d.

samples from an unknown distribution p(x).

This means the algorithm can access stochastic gradients of the population squared-loss objective in (2) via random samples from p(x).

The norm-controlled SGD variant we analyze is presented in Algorithm 1 (it can be easily extended to the mini-batch SGD version, where for each update we sample more than one data points).

It is almost the same as what is commonly used in practice: it random initializes the weight matrix by sampling unit spherical Gaussian, and at every step the algorithm moves towards the direction of the negative stochastic gradient with a linearly decaying learning rate.

However, there are two differences between Algorithm 1 and original SGD: first, we impose that the norm of the rows of W t be controlled; this is akin to the practical trick of "max-norm regularization" as explained in Section 1; second the update of bias is chosen differently than what is usually done in practice, which deserves additional explanation.

The stochastic gradient of bias b with respect to squared loss in (2) can be evaluated by sampling a single data point and differentiate against the empirical loss in (3), can be derived as DISPLAYFORM0 Since the gradient is noisy, the generic form of SGD suggests modifying b t j using the update DISPLAYFORM1 for a small learning rate η t b to mitigate noise.

This amounts to stepping towards the negative gradient direction and move a little.

On the other hand, we can directly find the next update b t+1 j as the point that sets the gradient to zero, that is, we find b * DISPLAYFORM2 The closed form solution to this is to choose DISPLAYFORM3 This strategy, which is essentially Newton's algorithm, should perform better than gradient descent if we have an accurate estimate of the true gradient, so it would likely benefit from evaluating the gradient using a mini-batch of data.

If, on the other hand, the gradient is very noisy, then this method will likely not work as well as the original SGD update.

Analyzing the evolvement of both W t and b t , which has dependent stochastic dynamic if we follow the original SGD update, would be a daunting task.

Thus, to simplify our analysis, we assume in our analysis that we have access to We assume that the data x we sample follows the dictionary learning model DISPLAYFORM4 DISPLAYFORM5 Here k is the size of the dictionary, which we assume to be at least two (otherwise, the model becomes degenerate), and the true value of k is unknown to the algorithm.

The rows of W * are the dictionary items; W * j satisfies DISPLAYFORM6 Let the incoherence between dictionary items be defined as λ : DISPLAYFORM7 .

In our simplified model, the coefficient vector s ∈ {0, 1} k is assumed to be 1-sparse, with P r(s j = 1) = 1 k DISPLAYFORM8 Finally, we assume that the noise has bounded norm 2 : max ≤ DISPLAYFORM9 Algorithm 1 Norm-controlled SGD training Input: width parameter n; norm parameter c; learning rate parameters c , t o , δ; total number of iterations, t max .

DISPLAYFORM10 While auto-encoders are often related to PCA, the latter cannot reveal any information about the true dictionary under this model even in the complete case, where d = k, due to the isotropic property of the underlying distribution.

The data generating model can be equivalently viewed as a mixture model: for example, when s j = 1, it means x is of the form W * j + .

When is Gaussian, the model coincides with mixture of Gaussians model, with the dictionary items being the latent locations of individual Gaussians.

Thus, we adopt the concept from mixture models, and use x ∼ C j to indicate that x is generated from the j-th component of the distribution.

To formally study the convergence property of Algorithm 1, we need a measure to gauge the distance between the learned representation at time t, W t , and the ground-truth representation, W * , which may have different number of rows.

There are potentially different ways to go about this.

The distance measure we use is DISPLAYFORM0 is the squared sine of the angle between the two vectors, which decreases monotonically as their angle decreases, and equals zero if and only if the vectors align.

Thus, DISPLAYFORM1 can be viewed as the angular distance from the best approximation in the learned hidden representations of the network, to the ground-truth dictionary item W * j .

And Θ(·, ·) measures this distance averaged over all dictionary items.

Our main result provides recovery and speed guarantee of Algorithm 1 under our data model.

Theorem 1.

Suppose we have access to i.i.d.

samples x ∼ p(x), where the distribution p(x) satisfies our model assumption in Section 2.2.

Fix any δ ∈ (0, n e ).

If we train auto-encoder with norm-controlled SGD as described in Algorithm 1, with the following parameter setting• The row norm of weights set to be DISPLAYFORM2 • If the bias update at t is chosen such that DISPLAYFORM3 • The learning rate of SGD is set to be η t := c t+to , with c > 2kc and DISPLAYFORM4 Then Algorithm 1 has the following guarantees• When random initialization with i.i.d.

samples from N (0, 1) is used, the algorithm will be initialized successfully (see definition of successful initialization in Definition 1) with probability at least 1 − k exp{−n( DISPLAYFORM5 • When random initialization with i.i.d.

samples x ∼ p(x) is used, the algorithm will be initialized successfully with probability at least DISPLAYFORM6 • Conditioning on successful initialization, let Ω denote the sample space of all realizations of the algorithm's stochastic output, (W 1 , W 2 , . . .

, ).

Then at any time t, there exists a large subset of the sample space, DISPLAYFORM7 Interpretation The first statement of the theorem suggests that the probability of successful initialization increases as the width of hidden layer increases.

In particular, when Gaussian initialization is used, in order to ensure a significantly large probability of successful initialization, the analysis suggests that the number of neurons required must scale as DISPLAYFORM8 , which is exponential in the ambient dimension.

When the neurons are initialized with samples from the unknown distribution, the analysis suggests that the number of neurons required scale as Ω( DISPLAYFORM9 , which is polynomial in the number of dictionary size.

Hence, our analysis suggests that, at least under our specific model, initializing with data is perhaps a better option than Gaussian initialization.

The second statement suggests that conditioning on a successful initialization, the algorithm will have expected convergence towards W * , measured by Θ(·, ·), of order O( 1 t ).

If we examine of form of bound on the convergence rate, we see that the rate will be dominated by the second term, whose constant is heavily influenced by the choice of learning rate parameter c .Explaining distributed sparse representation via gradient-based training The main advantage of gradient-based training of auto-encoders, as revealed by our analysis, is that it simultaneously updates all its neurons in parallel, in an independent fashion.

During training, a subset of neurons will specialize at learning a single dictionary item: some of them will be successful while others may fail to converge to a ground-truth representation.

However, since the update of each neuron is independent (in an algorithmic sense), when larger number of neurons are used (widening the hidden layer), it becomes more likely that each ground-truth dictionary will be learned by some neuron, even from random initialization.

Despite the simplicity of auto-encoders in comparison to other deep architectures, we still have a very limited theoretical understanding of them.

For linear auto-encoders whose width n is less than than its input dimension d, the seminal work of Oja (1982) revealed their connection to online stochastic PCA.

For non-linear auto-encoders, recent work Arpit et al. FORMULA1 analyzed sufficient conditions on the activation functions and the regularization term (which is added to the loss function) under which the auto-encoder learns a sparse representation.

Another work Rangamani et al. (2017) showed that under a class of sparse dictionary learning model (which is more general than ours) the ground-truth dictionary is a critical point (that is, either a saddle point or a local miminum) of the squared loss function, when ReLu activation is used.

We are not aware of previous work providing global convergence guarantee of SGD for non-linear auto-encoders, but our analysis techniques are closely related to recent works Balsubramani et al. (2013); Ge et al. (2015) ; Tang & Monteleoni (2017) that are at the intersection of stochastic (non-convex) optimization and unsupervised learning.

PCA, k-means, and sparse coding The work of Balsubramani et al. (2013) provided the first convergence rate analysis of Oja's and Krasulina's update rule for online learning the principal component (stochastic 1-PCA) of a data distribution.

The neural network corresponding to 1-PCA has a single node in the hidden layer without activation function.

We argue that a ReLu activated width n auto-encoder can be viewed as a generalized, multi-modal version of 1-PCA.

This is supported by our analysis: the expected improvement of each neuron, W t s , bears a striking similarity to that obtained in Balsubramani et al. (2013) .

The training of auto-encoders also has a similar flavor to online/stochastic k-means algorithm Tang & Monteleoni (2017): we may view each neuron as trying to learn a hidden dictionary item, or cluster center in k-means terminology.

However, there is a key difference between k-means and auto-encoders: the performance of k-means is highly sensitive to the number of clusters.

If we specify the number of clusters, which corresponds to the network width n in our notation, to be larger than the true k, then running n-means will over-partition data from each component, and each learned center will not converge to the true component center (because they converge to the mean of the sub-component).

For auto-encoders, however, even when n is much larger than k, the individual neurons can still converge to the true cluster center (dictionary item) thanks to the independent update of neurons.

SGD training of auto-encoders is perhaps closest to a family of sparse coding algorithms Schnass FORMULA1 ; Arora et al. (2015) .

For the latter, however, a critical hyper-parameter to tune is the threshold at which the algorithm decides to cut off insignificant signals.

Existing guarantees for sparse coding algorithms therefore depend on knowing this threshold.

For ReLu activated auto-encoders, the threshold is adaptively set for each neuron s at every iteration as −b t s via gradient descent.

Thus, they can be viewed as a sparse coding algorithm that self-tunes its threshold parameter.

In our analysis, we define an auxiliary variable DISPLAYFORM0 Note that φ(·, ·) is the squared cosine of the angle between W t s and W * j , which increases as their angle decreases.

Thus, φ can be thought as as measuring the angular "closeness" between two vectors; it is always bounded between zero and one and equals one if and only if the two vectors align.

Our analysis can be divided into three steps.

We first define what kind of initialization enables SGD to converge quickly to the correct solution, and show that when the number of nodes in the hidden layer is large, random initialization will satisfy this sufficient condition.

Then we derive expected the per-iteration improvement of SGD, conditioning on the algorithm's iterates staying in a local neighborhood (Definition 4).

Finally, we use martingale analysis to show that the local neighborhood condition will be satisfied with high probability.

Piecing these elements together will lead us to the proof of Theorem 1, which is in the Appendix.

Covering guarantee from random initialization Intuitively, for each ground-truth dictionary item, we only require that at least one neuron is initialized to be not too far from it.

Definition 1.

If the rows of W o have fixed norm c > 0.

Then we define the event of successful initialization as DISPLAYFORM0 Lemma 1 (Random initialization with Gaussian variables).

DISPLAYFORM1 Lemma 2 (Random initialization with data points).

Suppose W o ∈ R n×d is constructed by drawing X 1 , . . .

, X n from the data distribution p(x), and setting DISPLAYFORM2 Figure 1: The auto-encoder in this example has 5 neurons in the hidden layer and the dictionary has two items; in this case, g(1) = g(3) = 1, g(5) = 2, and the other two neurons do not learn any ground-truth (neurons mapped to 0 are considered useless).

Under unique firing condition, which holds when the dictionary is sufficiently incoherent, the red dashed connection will not take place (each neuron is learning at most one dictionary item).

DISPLAYFORM3 , according to the following firing map .

Note that some rows in W o may not be mapped to any dictionary item, in which case we let g(s) = 0.

This means such neurons are not close (in angular distance) to any ground-truth after random initialization.

Also note that for some rows W o s , there might exist multiple j ∈ [k] such that g(s) = j according to our criterion in the definition.

But when λ ≤ 1 2 , which is always the case by our model assumption on incoherence, Lemma 3 shows that the assignment must be unique, in which case the mapping is well defined.

Lemma 3 (Uniqueness of firing).

Suppose during training, the weight matrix has a fixed norm c. At time t, for any row of weight matrix W t s , we denote by τ s,1 := max j W t s c , W * j , and we denote by DISPLAYFORM4 DISPLAYFORM5 Thus, for any s ∈ [n] with g(s) > 0, the uniqueness of firing condition holds and the mapping g is defined unambiguously.

So we simplify notations on measure of distance and closeness as ∆ DISPLAYFORM6

This section lower bounds the expected increase of φ t s after each SGD update, conditioning on F t .

We first show that conditioning on F t , the firing of a neuron s with g(s) = j, will indicate that the data indeed comes from the j-th component, which is characterized by event E t .Definition 3.

At step t, we denote the event of correct firing of W t as DISPLAYFORM0 Definition 4.

At step t, we denote the event of satisfying local condition of W t as DISPLAYFORM1 DISPLAYFORM2 for some constant B > 0 where B is a constant depending on the model parameter and the norm of rows of weight matrix.

By Theorem 2, the sequence φ is conditional on E t , the correct firing condition.

So showing that the correct firing event indeed holds is crucial to our overall convergence analysis.

Since by Lemma 4, F t =⇒ E t , it suffices to show that F t holds.

To this end, note that F t 's form a nested sequence DISPLAYFORM0 We denote the limit of this sequence as DISPLAYFORM1 Theorem 3 shows that P r(F ∞ ) is in fact arbitrarily close to one, conditioning on FORMULA1 , where similar technical difficulty arise: to show local improvement of the algorithm on a non-convex functions, one usually needs to lower bound the probability of the algorithm entering a "bad" region, which can be saddle points Ge et al. FORMULA1 ; Balsubramani et al. (2013) DISPLAYFORM2 Then conditioning on F o , we have P r(F ∞ ) = 1 − δ

There are several interesting questions that are not addressed here.

First, as noted in our discussion in Section 2, the update of bias as analyzed in our algorithm is not exactly what is used in original SGD.

It would be interesting (and difficult) to explore whether the algorithm has fast convergence when b t is updated by SGD with a decaying learning rate.

Second, our model assumption is rather strong, and it would be interesting to see whether similar results hold on a relaxed model, for example, where one may relax to 1-sparse constraint to m-sparse, or one may relax the finite bound requirement on the noise structure.

Third, our performance guarantee of random initialization depends on a lower bound on the surface area of spherical caps.

Improving this bound can improve the tightness of our initialization guarantee.

Finally, it would be very interesting to examine whether similar result holds for activation functions other than ReLu, such as sigmoid function.

Derivation of stochastic gradients Upon receiving a data point x, the stochastic gradient with respect to W is a jacobian matrix whose (j * , i * )-th entry reads DISPLAYFORM0 For i = i * , the derivative of the second term can be written using the chain rule as DISPLAYFORM1 where we let a j * := w j * l x l + b j * , which is the activation of the j * -th neuron upon receiving x in the hidden layer before going through the ReLu unit.

For i = i * , the derivative of the second term can be written using product rule and chain rule as DISPLAYFORM2 Let r ∈ R d be the residual vector with r i : DISPLAYFORM3 In vector notation, the stochastic gradient of loss with respect to the j-th row of W can be written as DISPLAYFORM4 Similarly, we can obtain the stochastic gradient with respect to the j-th entry of the bias term as DISPLAYFORM5 Now let us examine the terms ∂s(aj ) ∂aj and r, W j .

By property of ReLu function, DISPLAYFORM6 Mathematically speaking, the derivative of ReLu at zero does not exist.

Here we follow the convention used in practice by setting the derivative of ReLu at zero to be 0.

In effect, the event {a j = 0} has zero probability, so what derivative to use at zero does not affect our analysis (as long as the derivative is finite).Proof of main theorem.

Consider any time t > 0.

By Lemma 1 and 2, the probability of successfully initializing the network can be lower bounded by DISPLAYFORM7 100k 2 }) if initialized with data Conditioning on F o * and applying Theorem 3, we get that for all t ≥ 0, P r( DISPLAYFORM8 Since F t =⇒ E t by Lemma 4, we can apply version 1 of Theorem 2 to get the expected increase in φ t s for any s such that g(s) > 0 as: DISPLAYFORM9 such that g(s) = j. Then, the inequality above translates to DISPLAYFORM10 by our choice of c and by our assumption on the initial value ∆ o s(j) .

Taking total expectation up to time t, conditioning on F t , and letting β denote a lower bound on β t , we get DISPLAYFORM11 where the last inequality is by the same argument as that in Lemma 8.

This has the exact same form as in Lemma D.1 of Balsubramani et al. (2013) .

Applying it with u t := E[∆ t s(j) |F t ], a = β, and b = (c ) 2 B (note our t + t o matches their notion of t), we get DISPLAYFORM12 By the upper bound on β t , we can choose β as small as 2c kc .

So we can get an upper expressed in algorithmic and model parameters as DISPLAYFORM13 The second inequality holds because by DISPLAYFORM14 Finally, DISPLAYFORM15 where the last inequality is by our requirement that c > 2kc.

Proof of Lemma 1.

Let u = z z , where z ∈ R d with z i ∼ N (0, 1).

We know that u is a random DISPLAYFORM0 where S cap (v, h) is the surface of the spherical cap centered at v with height h. By property of spherical Gaussian, we know that u is uniformly distributed on S d−1 .

So we can directly calculate the probability above as DISPLAYFORM1 where µ measures the area of surface.

The latter ratio can be lower bounded (see Lemma 5 in the Appendix) as a function of d and h: DISPLAYFORM2 By union bound, this implies that DISPLAYFORM3 Now by our choice of the form of lower bound on the inner product, we have DISPLAYFORM4 substituting this into the function f (d, h), we get a nice form DISPLAYFORM5 Substituting this into the previous inequality written in terms of h and letting ρ = DISPLAYFORM6 DISPLAYFORM7 where V ol(·) denotes measure of volume in R d .

So this lower bounds the ratio between volumes between spherical cap and the unit ball.

We show that we can use this to lower bound the ratio between surface areas between spherical cap and the unit ball.

Since by ?, we know that the ratio between their area can be expressed exactly as DISPLAYFORM8 and the ratio between their volume DISPLAYFORM9 where I x (a, b) is the regularized incomplete beta function.

By property of I x (a, b), DISPLAYFORM10 Proof of Lemma 2.

For any X i ∼ C j , for any j ∈ [k].

We first claim that DISPLAYFORM11 Proof of claim.

Let us consider the two-dimensional plane H determined by X i and W * j .

Clearly, i = W * j − X i also lies in H. Let θ := ∠(X i , W * j ) denote the angle between X i and W * j .

Note that cos θ = Xi Xi , W * j .

Fix the norm of noise i .

It is clear from elementary geometric insight that θ is maximized (and hence cos θ is minimized) when the line of X i is tangent to the ball centered at W * j with radius i .

We can directly calculate the value of cos θ at this point (see FIG4 as cos θ = 1 − i 2 , which finishes the proof of claim.

Now, we denote two events A := {min DISPLAYFORM12 The probability of event A can be lower bounded by concentration inequality for multinomial distribution Devroye (1983): Let n j := i∈[n] 1 {Xi∈Cj } .

We get DISPLAYFORM13 where the second inequality is by Lemma 3 of Devroye (1983) .

DISPLAYFORM14 where 2 is the empirical mean of i 2 for all X i , i ∈ [n] belonging to the same component C j for some j ∈ [k].

Conditioning on A, we know that the average is taken over at least n 2k samples for each component C j .

By one-sided Hoeffding's inequality, DISPLAYFORM15 (note, we abuse notation B in the exponent as an upper bound on 2 , as in other parts of the analysis).

Thus, DISPLAYFORM16 Proof of Lemma 3.

To simplify notation, we use τ 1 (τ 2 ) as a shorthand for τ s,1 (τ s,2 ).

Figure 3 ) that, DISPLAYFORM17 2 .

It can be verified that this implies DISPLAYFORM18 Observing that τ DISPLAYFORM19 Since F o holds, we know that the firing of neuron s is unique.

Let τ s,1 and τ s,2 as defined in Lemma 3.Let x ∼ C j .

Consider the case g(s) = j. In this case, Lemma 3 implies that DISPLAYFORM20 We will repeatedly use the following relation, as proven by Lemma 6, DISPLAYFORM21 Now, observe that 1 c 2 − 1 < 0 and DISPLAYFORM22 So we get, DISPLAYFORM23 4k .

Furthermore, since by our assumption, ≤ DISPLAYFORM24 Consider the case g(s) = j , j = j.

We first upper bound b o in this case.

Since 1 c 2 − 1 < 0, we would like to lower bound DISPLAYFORM25 On the other hand, DISPLAYFORM26 where the last inequality is by assumptions c > Case 0 < i ≤ t Suppose E i holds, we show that E i+1 holds for i ≤ t − 1.

Let x ∼ C j for any j ∈ [k].

Since E i holds, we know that is set such that DISPLAYFORM27 DISPLAYFORM28 Consider the case x ∼ C j , we have DISPLAYFORM29 where the last inequality is by assumption τ DISPLAYFORM30 holds by our assumptions that c ≤ √ 6k, and that ≤ DISPLAYFORM31 we can apply Lemma 3 to get DISPLAYFORM32 where the last inequality holds similarly as in the base case.

Lemma 6.

Let τ s,1 , τ s,2 be as defined in Lemma 3, and let λ be the incoherence parameter.

If τ DISPLAYFORM33 Proof.

Using the same argument as the proof of Lemma 3, we get DISPLAYFORM34 , and the second inequality is by Lemma 8.

For k ≥ 1, we define DISPLAYFORM35 We can similarly get, for k = 0, . . .

, i, DISPLAYFORM36 Since the bound is shrinking as β increases and β ≥ 2, DISPLAYFORM37 Recursively applying the relation until we get to the term DISPLAYFORM38 Combining all these recursive inequalities with the bound on λ (k) , we get Finally, we have DISPLAYFORM39 DISPLAYFORM40 Now recall that this holds for each s, that is, ∀s ∈ [n], 1 e , and t ≥ ( Lemma 10.

Suppose our model assumptions on parameters , c, α hold, and that our assumptions on the algorithmic parameter c in Theorem 1 holds, then DISPLAYFORM41 Proof.(1 − 1 c 2 )( where the last term is greater than zero because α < k − 1 4k 2 − 3k + 1 < 2k − 1/2 2k 2 − k So the term DISPLAYFORM42 DISPLAYFORM43

@highlight

theoretical analysis of nonlinear wide autoencoder