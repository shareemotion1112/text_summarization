Past works have shown that, somewhat surprisingly, over-parametrization can help generalization in neural networks.

Towards explaining this phenomenon, we adopt a margin-based perspective.

We establish: 1) for multi-layer feedforward relu networks, the global minimizer of a weakly-regularized cross-entropy loss has the maximum normalized margin among all networks, 2) as a result, increasing the over-parametrization improves the normalized margin and generalization error bounds for deep networks.

In the case of two-layer networks, an infinite-width neural network enjoys the best generalization guarantees.

The typical infinite feature methods are kernel methods; we compare the neural net margin with that of kernel methods and construct natural instances where kernel methods have much weaker generalization guarantees.

We validate this gap between the two approaches empirically.

Finally, this infinite-neuron viewpoint is also fruitful for analyzing optimization.

We show that a perturbed gradient flow on infinite-size networks finds a global optimizer in polynomial time.

In deep learning, over-parametrization refers to the widely-adopted technique of using more parameters than necessary (Krizhevsky et al., 2012; Livni et al., 2014) .

Both computationally and statistically, over-parametrization is crucial for learning neural nets.

Controlled experiments demonstrate that over-parametrization eases optimization by smoothing the non-convex loss surface (Livni et al., 2014; Sagun et al., 2017) .

Statistically, increasing model size without any regularization still improves generalization even after the model interpolates the data perfectly (Neyshabur et al., 2017b) .

This is surprising given the conventional wisdom on the trade-off between model capacity and generalization.

In the absence of an explicit regularizer, algorithmic regularization is likely the key contributor to good generalization.

Recent works have shown that gradient descent finds the minimum norm solution fitting the data for problems including logistic regression, linearized neural networks, and matrix factorization (Soudry et al., 2018; BID17 Li et al., 2018; BID16 Ji & Telgarsky, 2018) .

Many of these proofs require a delicate analysis of the algorithm's dynamics, and some are not fully rigorous due to assumptions on the iterates.

To the best of our knowledge, it is an open question to prove analogous results for even two-layer relu networks. (For example, the technique of Li et al. (2018) on two-layer neural nets with quadratic activations still falls within the realm of linear algebraic tools, which apparently do not suffice for other activations.)We propose a different route towards understanding generalization: making the regularization explicit.

The motivations are: 1) with an explicit regularizer, we can analyze generalization without fully understanding optimization; 2) it is unknown whether gradient descent provides additional implicit regularization beyond what 2 regularization already offers; 3) on the other hand, with a sufficiently weak 2 regularizer, we can prove stronger results that apply to multi-layer relu networks.

Additionally, explicit regularization is perhaps more relevant because 2 regularization is typically used in practice.

Concretely, we add a norm-based regularizer to the cross entropy loss of a multi-layer feedforward neural network with relu activations.

We show that the global minimizer of the regularized objective achieves the maximum normalized margin among all the models with the same architecture, if the regularizer is sufficiently weak (Theorem 2.1).

Informally, for models with norm 1 that perfectly classify the data, the margin is the smallest difference across all datapoints between the classifier score for the true label and the next best score.

We are interested in normalized margin because its inverse bounds the generalization error (see recent work BID5 Neyshabur et al., 2017a; BID14 or Proposition 3.1).

Our work explains why optimizing the training loss can lead to parameters with a large margin and thus, better generalization error (see Corollary 3.2).

We further note that the maximum possible margin is non-decreasing in the width of the architecture, and therefore the generalization bound of Corollary 3.2 can only improve as the size of the network grows (see Theorem 3.3).

Thus, even if the dataset is already separable, it could still be useful to increase the width to achieve larger margin and better generalization.

At a first glance, it might seem counterintuitive that decreasing the regularizer is the right approach.

At a high level, we show that the regularizer only serves as a tiebreaker to steer the model towards choosing the largest normalized margin.

Our proofs are simple, oblivious to the optimization procedure, and apply to any norm-based regularizer.

We also show that an exact global minimum is unnecessary: if we approximate the minimum loss within a constant factor, we obtain the max-margin within a constant factor (Theorem 2.2).To better understand the neural network max-margin, in Section 4 we compare the max-margin two-layer network obtained by optimizing both layers jointly to kernel methods corresponding to fixing random weights for the hidden layer and solving a 2-norm max-margin on the top layer.

We design a simple data distribution ( FIG3 ) where neural net margin is large but the kernel margin is small.

This translates to an Ω( √ d) factor gap between the generalization error bounds for the two approaches and demonstrates the power of neural nets compared to kernel methods.

We experimentally confirm that a gap does indeed exist.

In the setting of two-layer networks, we also study how over-parametrization helps optimization.

Prior works (Mei et al., 2018; BID10 Sirignano & Spiliopoulos, 2018; Rotskoff & Vanden-Eijnden, 2018) show that gradient descent on two-layer networks becomes Wasserstein gradient flow over parameter distributions in the limit of infinite neurons.

For this setting, we prove that perturbed Wasserstein gradient flow finds a global optimizer in polynomial time.

Finally, we empirically validate several claims made in this paper.

First, we confirm that neural networks do generalize better than kernel methods.

Second, we show that for two-layer networks, the test error decreases and margin increases as the hidden layer grows, as predicted by our theory.

Zhang et al. (2016) and Neyshabur et al. (2017b) show that neural network generalization defies conventional explanations and requires new ones.

Neyshabur et al. (2014) initiate the search for the "inductive bias" of neural networks towards solutions with good generalization.

Recent papers (Hardt et al., 2015; BID8 BID9 ) study inductive bias through training time and sharpness of local minima.

Neyshabur et al. (2015a) propose a new steepest descent algorithm in a geometry invariant to weight rescaling and show that this improves generalization.

Morcos et al. (2018) relate generalization in deep nets to the number of "directions" in the neurons.

Other papers BID15 Soudry et al., 2018; Nacson et al., 2018; BID17 Li et al., 2018; BID16 ) study implicit regularization towards a specific solution.

Ma et al. (2017) show that implicit regularization can help gradient descent avoid overshooting optima.

Rosset et al. (2004a; b) study logistic regression with a weak regularization and show convergence to the max margin solution.

We adopt their techniques and extend their results.

A line of work initiated by Neyshabur et al. (2015b) has focused on deriving tighter norm-based Rademacher complexity bounds for deep neural networks BID5 Neyshabur et al., 2017a; BID14 and new compression based generalization properties BID1 .

BID13 manage to compute non-vacuous generalization bounds from PAC-Bayes bounds.

Neyshabur et al. (2018) investigate the Rademacher complexity of two-layer networks and propose a bound that is decreasing with the distance to initialization.

Liang & Rakhlin (2018) and BID6 study the generalization of kernel methods.

On the optimization side, Soudry & Carmon (2016) explain why over-parametrization can remove bad local minima.

Safran & Shamir (2016) show that over-parametrization can improve the quality of the random initialization.

BID18 , Nguyen & Hein (2017), and Venturi et al. (2018) show that for sufficiently overparametrized networks, all local minima are global, but do not show how to find these minima via gradient descent.

BID11 show that for two-layer networks with quadratic activations, all second-order stationary points are global minimizers.

BID0 interpret over-parametrization as a means of implicit acceleration during optimization.

Mei et al. (2018) , BID10 , and Sirignano & Spiliopoulos (2018) take a distributional view of over-parametrized networks.

BID10 show that Wasserstein gradient flow converges to global optimizers under structural assumptions.

We extend this to a polynomial-time result.

Let R denote the set of real numbers.

We will use · to indicate a general norm, with · 1 , · 2 , · ∞ denoting the 1 , 2 , ∞ norms on finite dimensional vectors, respectively, and · F denoting the Frobenius norm on a matrix.

In general, we use¯on top of a symbol to denote a unit vector: when applicable,ū u/ u , where the norm · will be clear from context.

Let S DISPLAYFORM0 ) be the space of functions on S d−1 for which the p-th power of the absolute value is Lebesgue integrable.

DISPLAYFORM1 we overload notation and write α p DISPLAYFORM2 .

Throughout this paper, we reserve the symbol X = [x 1 , . . .

, x n ] to denote the collection of datapoints (as a matrix), and Y = [y 1 , . . .

, y n ] to denote labels.

We use d to denote the dimension of our data.

We often use Θ to denote the parameters of a prediction function f , and f (Θ; x) to denote the prediction of f on datapoint x. We will use the notation , to mean less than or greater than up to a universal constant, respectively.

Unless stated otherwise, O(·), Ω(·) denote some universal constant in upper and lower bounds, respectively.

The notation poly denotes a universal constant-degree polynomial in the arguments.

In this section, we will show that when we add a weak regularizer to cross-entropy loss with a positive-homogeneous prediction function, the normalized margin of the optimum converges to some max-margin solution.

As a concrete example, feedforward relu networks are positive-homogeneous.

Let l be the number of labels, so the i-th example has label y i ∈ [l].

We work with a family F of prediction functions f (Θ; ·) : R d → R l that are a-positive-homogeneous in their parameters for some a > 0: f (cΘ; x) = c a f (Θ; x), ∀c > 0.

We additionally require that f is continuous in Θ. For some general norm · , we study the λ-regularized cross-entropy loss L λ , defined as DISPLAYFORM0 1 We define the normalized margin of Θ λ as: DISPLAYFORM1 Define the · -max normalized margin as DISPLAYFORM2 and let Θ be a parameter achieving this maximum.

We show that with sufficiently small regularization level λ, the normalized margin γ λ approaches the maximum margin γ .

Our theorem and proof are inspired by the result of Rosset et al. (2004a; b) , who analyze the special case when f is a linear predictor.

In contrast, our result can be applied to non-linear f as long as f is homogeneous.

Theorem 2.1.

Assume the training data is separable by a network f (Θ ; ·) ∈ F with an optimal normalized margin γ > 0.

Then, the normalized margin of the global optimum of the weaklyregularized objective (equation 2.1) converges to γ as the strength of the regularizer goes to zero.

Mathematically, let γ λ be defined in equation 2.2.

Then DISPLAYFORM3 We formally show that L λ has a minimizer in Claim A.1 of Section A.An intuitive explanation for our result is as follows: because of the homogeneity, the loss L(Θ λ ) roughly satisfies the following (for small λ, and ignoring problem parameters such as n): DISPLAYFORM4 Thus, the loss selects parameters with larger margin, while the regularization favors parameters with a smaller norm.

The full proof of the theorem is deferred to Section A.1.Theorem 2.1 applies to feedforward relu networks and states that global minimizers of the weaklyregularized loss will obtain a maximum margin among all networks of the given architecture.

By considering global minimizers, Theorem 2.1 provides a framework for directly analyzing generalization properties of the solution without considering details of the optimization algorithm.

In Section 3 we leverage this framework and existing generalization bounds BID14 to provide a clean argument that over-parameterization can improve generalization.

We can also provide an analogue of Theorem 2.1 for the binary classification setting.

For this setting, our prediction is now a single real output and we train using logistic loss.

We provide formal definitions and results in Section A.2.

Our study of the generalization properties of the max-margin (see Section 3 and Section 4) is based in this setting.

Since L λ is typically hard to optimize exactly for neural nets, we study how accurately we need to optimize L λ to obtain a margin that approximates γ up to a constant.

The following theorem shows that it suffices to find Θ achieving a constant factor multiplicative approximation of L λ (Θ λ ), where λ is some sufficiently small polynomial in n, l, γ .

Though our theorem is stated for the general multi-class setting, it also applies for binary classification.

We provide the proof in Section A.3.Theorem 2.2.

In the setting of Theorem 2.1, suppose that we choose DISPLAYFORM0 for sufficiently large c (that only depends on r/a).

DISPLAYFORM1

In Section 2 we showed that optimizing a weakly-regularized logistic loss leads to the maximum normalized margin.

We now study the direct implications of this result on the generalization properties of the solution.

Specifically, we use existing Rademacher complexity bounds of BID14 to present a generalization bound that depends on the network architecture only through the inverse 2 -normalized margin and depth of the network (see Proposition 3.1).

Next, we combine this bound with Theorem 2.1 to conclude that parameters obtained by optimizing logistic loss with weak 2 -regularization will have a generalization bound that scales with the inverse of the maximum possible margin and depth.

Finally, we note that the maximum possible margin can only increase as the size of the network grows, which suggests that increasing the size of the network improves the generalization of the solution (see Theorem 3.3).We consider depth-K neural networks with 1-Lipschitz, 1-positive-homogeneous activation φ for K ≥ 2.

Suppose that the collection of parameters Θ is given by matrices W 1 , . . .

, W K .

The K-layer network will compute a real-valued score DISPLAYFORM0 where we overload notation to let φ(·) denote the element-wise application of the activation φ.

Let m i denote the size of the i-th hidden layer, so DISPLAYFORM1 ) denote the sequence of hidden layer sizes.

We will focus on 2 -regularized loss.

The weakly-regularized logistic loss of the depth-K architecture with hidden layer sizes M is therefore DISPLAYFORM2 We note that f is K-homogeneous in Θ, so the results of Section 2 apply to L λ,M .

2 Following our conventions from Section 2, we denote the optimizer of L λ,M by Θ λ,M , the normalized margin of Θ λ,M by γ λ,M , the max-margin solution by Θ ,M , and the max-margin by γ ,M .

Our notation emphasizes the architecture of the network.

Since the classifier f now predicts a single real value, we need to redefine DISPLAYFORM3 When the data is not separable by a neural network with architecture M, we define γ ,M to be zero.

Recall that X = [x 1 , . . .

, x n ] denotes the matrix with all the data points as columns, and Y = [y 1 , . . .

, y n ] denotes the labels.

We sample X and Y i.i.d.

from the data generating distribution p data , which is supported on X × {−1, +1}. We can define the population 0-1 loss and training 0-1 loss of the network parametrized by Θ by DISPLAYFORM4 Let C sup x∈X x 2 be an upper bound on the norm of a single datapoint.

Proposition 3.1 shows that the generalization error only depends on the parameters through the inverse of the margin on the training data.

We obtain Proposition 3.1 by applying Theorem 1 of BID14 with the standard technique of using margin loss to bound classification error.

There exist other generalization bounds which depend on the margin and some normalization (Neyshabur et al., 2015b; 2017a; BID5 Neyshabur et al., 2018) ; we choose the bounds of BID14 because they fit well with 2 normalization.

In the two-layer case K = 2, the bound below also follows from Neyshabur et al. (2015b) .

DISPLAYFORM5 where (γ) DISPLAYFORM6 .

Note that (γ) is typically small, and thus the above bound mainly scales with DISPLAYFORM7 For completeness, we state the proof in Section C.1.

By combining this bound with our Theorem 2.1 we can conclude that optimizing weakly-regularized logistic loss gives us generalization error bounds that depend on the maximum possible margin of a network with the given architecture.

Corollary 3.2.

In the setting of Proposition 3.1, with probability 1 − δ, DISPLAYFORM8 where (γ) is defined as in Proposition 3.1.

Above we implicitly assume γ ,M > 0, since otherwise the right hand side of the bound is vacuous.2 Although Theorem 2.1 is written in the language of multi-class prediction where the classifier outputs l ≥ 2 scores, the results translate to single-output binary classification.

See Section A.2.3 Although the 1 K (K−1)/2 factor of equation 3.3 decreases with depth K, the margin γ will also tend to decrease as the constraint Θ F ≤ 1 becomes more stringent.

By applying Theorem 2.2 with Proposition 3.1, we can also conclude that optimizing L λ,M within a constant factor gives a margin, and therefore generalization bound, approximating the best possible.

One consequence of Corollary 3.2 is that optimizing weakly-regularized logistic loss results in the best possible generalization bound out of all models with the given architecture.

This indicates that the widely used algorithm of optimizing deep networks with 2 -regularized logistic loss has an implicit bias towards solutions with good generalization.

Next, we observe that the maximum normalized margin is non-decreasing with the size of the architecture.

Formally This theorem is simple to prove and follows because we can directly implement any network of architecture M using one of architecture M , if M ≤ M .

This can explain why additional overparameterization has been empirically observed to improve generalization in two-layer networks (Neyshabur et al., 2017b) : the margin does not decrease with a larger network size, and therefore Corollary 3.2 gives a better generalization bound.

In Section 6, we provide empirical evidence that the test error decreases with larger network size while the margin is non-decreasing.

The phenomenon in Theorem 3.3 contrasts with standard 2 -normalized linear prediction.

In this setting, adding more features increases the norm of the data, and therefore the generalization error bounds could also increase.

On the other hand, Theorem 3.3 shows that adding more neurons (which can be viewed as learned features) can only improve the generalization of the max-margin solution.

We will continue our study of the max-margin neural network via comparison against kernel methods, a context in which margins have already been extensively studied.

We show that two-layer networks can obtain a larger margin, and therefore better generalization guarantees, than kernel methods.

Our comparison between the two methods is motivated by an equivalence between the 2 max-margin of an infinite-width two-layer network and the 1 -SVM (Zhu et al., 2004) over the lifted feature space defined by the activation function applied to all possible hidden units (Neyshabur et al., 2014; Rosset et al., 2007; BID7 .

The kernel method corresponds to the 2 -SVM in this same feature space, and is equivalent to fixing random hidden layer weights and solving an 2 -SVM over the top layer.

In Theorem 4.3, we construct a distribution for which the generalization upper bounds for the 1 -SVM on this feature space are smaller than those for the 2 -SVM by a Ω( √ d) factor.

Our work provides evidence that optimizing all layers of a network can be beneficial for generalization.

There have been works that compare 1 and 2 -regularized solutions in the context of feature selection and construct a feature space for which a generalization gap exists (e.g., see Ng (2004) ).

In contrast, we work in the fixed feature space of relu activations, which makes our construction particularly challenging.

We will use m to denote the width of the single hidden layer of the network.

Following the convention from Section 3, we will use γ ,m to denote the maximum possible normalized margin of a two-layer network with hidden layer size m (note the emphasis on the size of the single hidden layer).

The depth K = 2 case of Corollary 3.2 immediately implies that optimizing weakly-regularized 2 loss over width-m two-layer networks gives parameters whose generalization upper bounds depend on the hidden layer size only through 1/γ ,m .

Furthermore, from Theorem 3.3 it immediately follows that DISPLAYFORM0 The work of Neyshabur et al. (2014) links γ ,m to the 1 SVM over a lifted space.

Formally, we define a lifting function ϕ : DISPLAYFORM1 ) mapping data to an infinite feature vector: DISPLAYFORM2 where φ is the activation of Section 3.

We look at the margin of linear functionals corresponding to α ∈ L 1 (S d−1 ) .

The 1-norm SVM (Zhu et al., 2004) over the lifted feature ϕ(x) solves for the maximum margin: DISPLAYFORM3 where we rely on the inner product and 1-norm defined in Section 1.2.

This formulation is equivalent to a hard-margin optimization on "convex neural networks" BID7 .

Bach (2017) (2006) , our Theorem 2.1 implies that optimizing weaklyregularized logistic loss over two-layer networks is equivalent to solving equation 4.2 when the size of the hidden layer is at least n + 1, where n is the number of training examples.

Proposition 4.1 essentially restates this with the minor improvement that this equivalence 4 also holds when the size of the hidden layer is n. Proposition 4.1.

Let γ 1 be defined in equation 4.2.

Then DISPLAYFORM4 For completeness, we prove Proposition 4.1 in Section B, relying on the work of Tibshirani (2013) and Rosset et al. (2004a) .

Importantly, the 1 -max margin on the lifted feature space is obtainable by optimizing a finite neural network.

We compare this to the 2 margin attainable via kernel methods.

Following the setup of equation 4.2, we define the kernel problem over α ∈ L 2 (S d−1 ): DISPLAYFORM5 where κ Vol(S d−1 ).

(We scale α 2 by √ κ to make the lemma statement below cleaner.)

First, γ 2 can be used to obtain a standard upper bound on the generalization error of the kernel SVM.

Following the notation of Section 3, we will let L 2-svm denote the 0-1 population classification error for the optimizer of equation 4.3.

Lemma 4.2.

In the setting of Proposition 3.1, with probability at least 1−δ, the generalization error of the standard kernel SVM with relu feature (defined in equation 4.3) is bounded by DISPLAYFORM6 where 2 log max log 2 DISPLAYFORM7 is typically a lower-order term.

The bound above follows from standard techniques BID4 , and we provide a full proof in Section C.2.

We construct a data distribution for which this lemma does not give a good bound for kernel methods, but Corollary 3.2 does imply good generalization for two-layer networks.

Theorem 4.3.

There exists a data distribution p data such that the 1 SVM with relu features has a good margin: γ 1 1 and with probability 1 − δ over the choice of i.i.d.

samples from p data , obtains generalization error DISPLAYFORM8 is typically a lower order term.

Meanwhile, with high probability the 2 SVM has a small margin: γ 2 max log n n , 1/d and therefore the generalization upper bound from 4 The factor of 1 2is due the the relation that every unit-norm parameter Θ corresponds to an α in the lifted space with α = 2.

In particular, the 2 bound is larger than the 1 bound by a Ω( DISPLAYFORM0 Although Theorem 4.3 compares upper bounds, our construction highlights properties of distributions which result in better neural network generalization than kernel method generalization.

Furthermore, in Section 6 we empirically validate the gap in generalization between the two methods.

We briefly overview the construction of p data here.

The full proof is in Section D.1.Proof sketch for Theorem 4.3.

We base p data on the distribution D of examples (x, y) described below.

Here e i is the i-th standard basis vector and we use x e i to represent the i-coordinate of x (since the subscript is reserved to index training examples).

DISPLAYFORM1 . .

DISPLAYFORM2 , and DISPLAYFORM3 Figure 1 shows samples from D when there are 3 dimensions.

From the visualization, it is clear that there is no linear separator for D. As Lemma D.1 shows, a relu network with four neurons can fit this relatively complicated decision boundary.

On the other hand, for kernel methods, we prove that the symmetries in D induce cancellation in feature space.

As a result, the features are less predictive of the true label and the margin will therefore be small.

We formalize this argument in Section D.1.

We are able to prove an even larger Ω( n/d) gap between neural networks and kernel methods in the regression setting where we wish to interpolate continuous labels.

Analogously to the classification setting, optimizing a regularized squared error loss on neural networks is equivalent to solving a minimum 1-norm regression problem (see Theorem D.5).

Furthermore, kernel methods correspond to a minimum 2-norm problem.

We construct distributions p data where the 1-norm solution will have a generalization error bound of O( d/n), whereas the 2-norm solution will have a generalization error bound that is Ω(1) and thus vacuous.

In Section D.2, we define the 1-norm and 2-norm regression problems.

In Theorem D.10 we formalize our construction.

In the prior section, we studied the limiting behavior of the generalization of a two-layer network as its width goes to infinity.

In this section, we will now study the limiting behavior of the optimization algorithm, gradient descent.

Prior work (Mei et al., 2018; BID10 has shown that as the hidden layer size grows to infinity, gradient descent for a finite neural network approaches the Wasserstein gradient flow over distributions of hidden units (defined in equation 5.1).

BID10 assume the gradient flow converges, a non-trivial assumption since the space of distributions is infinite-dimensional, and given the assumption prove that Wasserstein gradient flow converges to a global optimizer in this setting, but do not specify a convergence rate.

Mei et al. (2018) show global convergence for the infinite-neuron limit of stochastic Langevin dynamics, but also do not provide a convergence rate.

We show that a perturbed version of Wasserstein gradient flow converges in polynomial time.

The informal take-away of this section is that a perturbed version of gradient descent converges in polynomial time on infinite-size neural networks (for the right notion of infinite-size.)Formally, we optimize the following functional over distributions ρ on R d+1 : DISPLAYFORM0 In this work, we consider 2-homogeneous Φ and V .

We will additionally require that R is convex and nonnegative and V is positive on the unit sphere.

Finally, we need standard regularity assumptions on R, Φ, and V : Assumption 5.1 (Regularity conditions on Φ, R, V ).

Φ and V are differentiable as well as upper bounded and Lipschitz on the unit sphere.

R is Lipschitz and its Hessian has bounded operator norm.

We provide more details on the specific parameters (for boundedness, Lipschitzness, etc.) in Section E.1.

We note that relu networks satisfy every condition but differentiability of Φ.5 We can fit a neural network under our framework as follows: Example 5.2 (Logistic loss for neural networks).

We interpret ρ as a distribution over the parameters of the network.

Let k n and Φ i (θ) wφ(u x i ) for θ = (w, u).

In this case, Φdρ is a distributional neural network that computes an output for each of the n training examples (like a standard neural network, it also computes a weighted sum over hidden units).

We can compute the distributional version of the regularized logistic loss in equation 3.2 by setting V (θ) λ θ 2 2 and R(a 1 , . . .

, a n ) DISPLAYFORM1 is the gradient of L with respect to ρ, and v is the induced velocity field.

For the standard Wasserstein gradient flow dynamics, ρ t evolves according to DISPLAYFORM2 where ∇· denotes the divergence of a vector field.

For neural networks, these dynamics formally define continuous-time gradient descent when the hidden layer has infinite size (see Theorem 2.6 of BID10 , for instance).We propose the following modification of the Wasserstein gradient flow dynamics: DISPLAYFORM3 where U d is the uniform distribution on S d .

In our perturbed dynamics, we add very small uniform noise over U d , which ensures that at all time-steps, there is sufficient mass in a descent direction for the algorithm to decrease the objective.

For infinite-size neural networks, one can informally interpret this as re-initializing a very small fraction of the neurons at every step of gradient descent.

We prove convergence to a global optimizer in time polynomial in 1/ , d, and the regularity parameters.

Theorem 5.3 (Theorem E.4 with regularity parameters omitted).

Suppose that Φ and V are 2-homogeneous and the regularity conditions of Assumption 5.1 are satisfied.

Also assume that from starting distribution ρ 0 , a solution to the dynamics in equation DISPLAYFORM4 , where the regularity parameters for Φ, V , and R are hidden in the poly(·).

Then, perturbed Wasserstein gradient flow converges to an -approximate global minimum in t time: DISPLAYFORM5 We provide a theorem statement that includes regularity parameters in Section E.1.

We prove the theorem in Section E.2.As a technical detail, Theorem 5.3 requires that a solution to the dynamics exists.

We can remove this assumption by analyzing a discrete-time version of equation 5.2: DISPLAYFORM6 and additionally assuming Φ and V have Lipschitz gradients.

In this setting, a polynomial time convergence result also holds.

We state the result in Section E.3.An implication of our Theorem 5.3 is that for infinite networks, we can optimize the weaklyregularized logistic loss in time polynomial in the problem parameters and λ −1 .

By Theorem 2.2, we only require λ −1 = poly(n) to approximate the maximum margin within a constant factor.

Thus, for infinite networks, we can approximate the max margin within a constant factor in polynomial time.

We first compare the generalization of neural networks and kernel methods for classification and regression.

In FIG4 we plot the generalization error and predicted generalization upper bounds 6 of a trained neural network against a 2 kernel method with relu features as we vary n. Our data comes from a synthetic distribution generated by a neural network with 6 hidden units; we provide a detailed setup in Section F.1.

For classification we plot 0-1 error, whereas for regression we plot squared error.

The variance in the neural network generalization bound for classification likely occured because we did not tune learning rate and training time, so the optimization failed to find the best margin.

The plots show that two-layer networks clearly outperform kernel methods in test error as n grows.

However, there seems to be looseness in the bounds: the kernel generalization bound appears to stay constant with n (as predicted by our theory for regression), but the test error decreases.

We also plot the dependence of the test error and margin on the hidden layer size in FIG5 for synthetic data generated from a ground truth network with 10 hidden units and also MNIST.

The plots indicate that test error is decreasing in hidden layer size while margin is increasing, as Theorem 3.3 predicts.

We provide more details on the experimental setup in Section F.2.In Section F.3, we verify the convergence of a simple neural network to the max-margin solution as regularization decreases.

In Section F.4, we train modified WideResNet architectures on CIFAR10 and CIFAR100.

Although ResNet is not homogeneous, we still report improvements in generalization from annealing the weight decay during training, versus staying at a fixed decay rate.

We have made the case that maximizing margin is one of the inductive biases of relu networks obtained from optimizing weakly-regularized cross-entropy loss.

Our framework allows us to directly analyze generalization properties of the network without considering the optimization algorithm used to obtain it.

Using this perspective, we provide a simple explanation for why over-parametrization can improve generalization.

It is a fascinating question for future work to characterize other generalization properties of the max-margin solution.

On the optimization side, we make progress towards understanding over-parametrized gradient descent by analyzing infinite-size neural networks.

A natural direction for future work is to apply our theory to optimize the margin of finite-sized neural networks.

Proof.

We will argue in the setting of Theorem 2.1 where L λ is the multi-class cross entropy loss, because the logistic loss case is analogous.

We first note that L λ is continuous in Θ because f is continuous in Θ and the term inside the logarithm is always positive.

DISPLAYFORM0 However, there must be a value Θ λ which attains inf Θ ≤M L λ (Θ), because {Θ : Θ ≤ M } is a compact set and L λ is continuous.

Thus, inf Θ L λ (Θ) is attained by some Θ λ .

Towards proving Theorem 2.1, we first show as we decrease λ, the norm of the solution Θ λ grows.

Lemma A.2.

In the setting of Theorem 2.1, as λ → 0, we have Θ λ → ∞.To prove Theorem 2.1, we rely on the exponential scaling of the cross entropy: L λ can be lower bounded roughly by exp(− Θ λ γ λ ), but also has an upper bound that scales with exp(− Θ λ γ ).

By Lemma A.2, we can take large Θ λ so the gap γ − γ λ vanishes.

This proof technique is inspired by that of Rosset et al. (2004a) .Proof of Theorem 2.1.

For any M > 0 and Θ with γ Θ min i f (Θ; DISPLAYFORM0 (by the homogeneity of f ) DISPLAYFORM1 We can also apply DISPLAYFORM2 Applying equation A.2 with M = Θ λ and Θ = Θ , noting that Θ ≤ 1, we have: DISPLAYFORM3 Next we lower bound L λ (Θ λ ) by applying equation A.3, DISPLAYFORM4 Combining equation A.4 and equation A.5 with the fact that DISPLAYFORM5 Recall that by Lemma A.2, as λ → 0, we have DISPLAYFORM6 Thus, we can apply Taylor expansion to the equation above with respect to exp(− Θ λ a γ ) and exp(− Θ λ a γ λ ).

DISPLAYFORM7 We claim this implies that γ ≤ lim inf λ→0 γ λ .

If not, we have lim inf λ→0 γ λ < γ , which implies that the equation above is violated with sufficiently large Θ λ ( Θ λ log(2( − 1)n) 1/a would suffice).

By Lemma A.2, Θ λ → ∞ as λ → 0 and therefore we get a contradiction.

Finally, we have γ λ ≤ γ by definition of γ .

Hence, lim λ→0 γ λ exists and equals γ .

Now we fill in the proof of Lemma A.2.Proof of Lemma A.2.

For the sake of contradiction, we assume that ∃C > 0 such that for any λ 0 > 0, there exists 0 < λ < λ 0 with Θ λ ≤ C. We will determine the choice of λ 0 later and pick λ such that Θ λ ≤ C. Then the logits (the prediction f j (Θ, x i ) before softmax) are bounded in absolute value by some constant (that depends on C), and therefore the loss function − log exp(fy i (Θ;xi)) l j=1 exp(fj (Θ;xi)) for every example is bounded from below by some constant D > 0 (depending on C but not λ.)

DISPLAYFORM8 (by the optimality of Θ λ ) DISPLAYFORM9 Taking a sufficiently small λ 0 , we obtain a contradiction and complete the proof.

For completeness, we state and prove our max-margin results for the setting where we fit binary labels y i ∈ {−1, +1} (as opposed to indices in [l]) and redefining f (Θ; ·) to assign a single real-valued score (as opposed to a score for each label).

This lets us work with the simpler λ-regularized logistic loss: DISPLAYFORM0 As before, let Θ λ ∈ arg min L λ (Θ), and define the normalized margin γ λ by γ λ min i y i f (Θ λ ; x i ).

Define the maximum possible normalized margin DISPLAYFORM1 Theorem A.3.

Assume γ > 0 in the binary classification setting with logistic loss.

Then as λ → 0, DISPLAYFORM2 The proof follows via simple reduction to the multi-class case.

Proof of Theorem A.3.

We prove this theorem via reduction to the multi-class case with l = 2.

Constructf : DISPLAYFORM3 Define new labelsỹ i = 1 if y i = −1 andỹ i = 2 if y i = 1.

Now note thatfỹ i (Θ; x i )−f j =ỹi (Θ; x i ) = y i f (Θ; x i ), so the multi-class margin for Θ underf is the same as binary margin for Θ under f .

Furthermore, definingL DISPLAYFORM4 , and in particular,L λ and L λ have the same set of minimizers.

Therefore we can apply Theorem 2.1 for the multi-class setting and conclude γ λ → γ in the binary classification setting.

Proof of Theorem 2.2.

Choose B 1 γ log DISPLAYFORM0 λ .

Now we note that DISPLAYFORM1 for sufficiently large c depending only on a/r.

Now using the fact that log(x) ≥ x 1+x ∀x ≥ −1, we additionally have the lower bound DISPLAYFORM2 The middle inequality followed because x 1−x is increasing in x for 0 ≤ x < 1, and the last because DISPLAYFORM3 We will first bound ♣.

First note that log( DISPLAYFORM4 where the last inequality follows from the fact that Finally, we note that if 1 + log DISPLAYFORM5 is a sufficiently large constant that depends only on a/r (which can be achieved by choosing c sufficiently large), it will follow that ♥ ≤ For the proof of this lemma, we find it convenient to work with a minimum norm formulation which we show is equivalent to equation 4.2: Proof.

Let opt denote the optimal objective for equation B.1.

We note that DISPLAYFORM6 DISPLAYFORM7 For any primal optimal solution α and dual optimal solution λ , it must hold that DISPLAYFORM8 Proof.

The dual form can be solved for by computation.

By strong duality, equation B.2 must follow from the KKT conditions.

Now define the mapping v : DISPLAYFORM9 We will show a general result about linearly dependent v(ū) forū ∈ supp(α ), after which we can reduce directly to the proof of Tibshirani (2013).

Claim B.4.

Let α be any optimal solution.

Suppose that there exists S ⊆ supp(α ) such that {v(ū) :ū ∈ S} forms a linearly dependent set, i.e. Proof.

Let λ be any dual optimal solution, then λ v(ū) = sign(α (ū)) ∀ū ∈ supp(α ) by Claim B.3.

Thus, we apply λ to both sides of equation B.3 to get the desired statement.

Proof of Lemma B.1.

The rest of the proof follows Lemma 14 in Tibshirani (2013).

The lemma argues that if the conclusion of Claim B.4 holds and an optimal solution α has S ⊆ supp(α ) with {v(ū) :ū ∈ S} linearly dependent, we can construct a new α with α 1 = α 1 and supp(α ) ⊂ supp(α ) (where the inclusion is strict).

Thus, if we consider an optimal α with minimal support, it must follow that {v(ū) :ū ∈ supp(α )} is a linearly independent set, and therefore |supp(α )| ≤ n.

We can now complete the proof of Proposition 4.1.Proof of Proposition 4.1.

For ease of notation, we will parametrize a two-layer network with m units by top layer weights w 1 , . . .

, w m ∈ R and bottom layer weights u 1 , . . . , u m ∈ R d .

As before, we use Θ to refer to the collection of parameters, so the network computes the real-valued function DISPLAYFORM10 Note that we simply renamed the variables from the parametrization of equation 3.1.We first apply Lemma B.1 to conclude that equation 4.2 admits a n-sparse optimal solution α .

Because of sparsity, we can now abuse notation and treat α as a real-valued function such that ū∈supp(α ) |α (ū)| ≤ 1.

We construct Θ corresponding to a two-layer network with m ≥ n hidden units and normalized margin at least γ 1 2 .

For clarity, we let W correspond to the top layer weights and U correspond to the bottom layer weights.

For everyū ∈ supp(α), we let Θ have a corresponding hidden unit j with (w j , u j ) = sign(α (ū)) DISPLAYFORM11 , and set the remaining hidden units to 0.

This is possible because m ≥ n. Now DISPLAYFORM12 Thus it follows that Θ has normalized margin at least γ 1 /2, so γ ,m ≥ γ 1 /2.To conclude, we show that γ ,m ≤ γ 1 /2.

Let Θ ,m denote the parameters obtaining optimal m-unit margin γ ,m with hidden units (w

We prove the generalization error bounds stated in Proposition 3.1 and Lemma 4.2 via Rademacher complexity and margin theory.

Assume that our data X, Y are drawn i.i.d.

from ground truth distribution p data supported on X × Y. For some hypothesis class F of real-valued functions, we define the empirical Rademacher complexitŷ R(F) as follows:R DISPLAYFORM0 where i are independent Rademacher random variables. (2009) DISPLAYFORM1 be drawn iid from p data .

We work in the binary classification setting, so Y = {−1, 1}. Assume that for all f ∈ F, we have sup x∈X f (x) ≤ C. Then with probability at least 1 − δ over the random draws of the data, for every γ > 0 and f ∈ F, DISPLAYFORM2 We will prove Proposition 3.1 by applying the Rademacher complexity bounds of BID14 with Theorem C.1.First, we show the following lemma bounding the generalization of neural networks whose weight matrices have bounded Frobenius norms.

Lemma C.2.

Define the hypothesis class F K over depth-K neural networks by DISPLAYFORM3 Recall that L(Θ) denotes the 0-1 population loss L(f (Θ; ·)).

Then for any f (Θ; ·) ∈ F K classifying the training data correctly with unnormalized margin γ Θ min i y i f (Θ; x i ) > 0, with probability at least 1 − δ, DISPLAYFORM4 Note the dependence on the unnormalized margin rather than the normalized margin.

Proof.

We first claim that sup f (Θ;·)∈F K sup x∈X f (Θ; x) ≤ C. To see this, for any f (Θ; ·) ∈ F K , DISPLAYFORM5 (since φ is 1-Lipschitz and φ(0) = 0, so φ performs a contraction) < x 2 ≤ C (repeatedly applying this argument and using W j F < 1)Furthermore, by Theorem 1 of BID14 ,R(F K ) has upper bound DISPLAYFORM6 Thus, we can apply Theorem C.1 to conclude that for all f (Θ; ·) ∈ F K and all γ > 0, with probability 1 − δ, DISPLAYFORM7 In particular, by definition choosing γ = γ Θ makes the first term on the LHS vanish and gives the statement of the lemma.

Proof of Proposition 3.1.

Given parameters Θ = (W 1 , . . .

, W K ), we first construct parameters Θ = (W 1 , . . .

,W K ) such that f (Θ; ·) and f (Θ; ·) compute the same function, and DISPLAYFORM8 Furthermore, we also have DISPLAYFORM9 (by the homogeneity of φ) DISPLAYFORM10 Now we note that by construction, L(Θ) = L(Θ).

Now f (Θ; ·) must also classify the training data perfectly, has unnormalized margin γ, and furthermore f (Θ; ·) ∈ F K .

As a result, Lemma C.2 allows us to conclude the desired statement.

To conclude Corollary 3.2, we apply the above on Θ λ,M and use Theorem A.3.

Let F 2,φ B denote the class of 2 -bounded linear functionals in lifted feature space: DISPLAYFORM0 We abuse notation and write α ∈ F 2,φ B to indicate a linear functional from F 2,φ B .

As before, we will use L(α) to indicate the 0-1 population loss of the classifier x → α, ϕ(x) and let C sup x∈X x 2 be an upper bound on the norm of the data.

We focus on analyzing the Rademacher complexityR(F 2,φ B ), mirroring derivations done in the past BID4 .

We include our derivations here for completeness.

DISPLAYFORM1 Proof.

We writeR DISPLAYFORM2 (terms where i = j cancel out)As an example, we can apply this bound to relu features: DISPLAYFORM3 Proof.

We first show that ϕ( DISPLAYFORM4 where the last line uses the computation provided in Lemma A.1 by BID12 .

Now we plug this into Lemma C.3 to get the desired bound.

We will now prove Lemma 4.2.Proof of Lemma 4.2.

From equation C.2, we first obtain sup x∈X ϕ(x) 2 C κ d .

Denote the optimizer for equation 4.3 by α 2 .

Note that √ κα 2 ∈ F 2,φ 1 , and furthermore L(α 2 ) = L( √ κα 2 ).

Since √ κα 2 has unnormalized margin √ κγ 2 , we apply Theorem C.1 on margin √ κγ 2 and hypothesis class F 2,φ 1 to get with probability 1 − δ, DISPLAYFORM5 In this section we will complete a proof of Theorem 4.3.

Recall the construction of the distribution D provided in Section 4.

We first provide a classifier of this data with small 1 norm.

Lemma D.1.

In the setting of Theorem 4.3, we have that DISPLAYFORM6 Now we will upper bound the margin attainable by the 2 SVM.

Lemma D.2 (Margin upper bound tool).

In the setting of Theorem 4.3, we have DISPLAYFORM7 Proof.

By the definition of γ 2 , we have that for any α with √ κ α 2 ≤ 1, we have DISPLAYFORM8 DISPLAYFORM9 Proof.

Let W i = ϕ(x i )

y i .

We will bound several quantities regarding W i '

s.

In the rest of the proof, we will condition on the event E that ∀i, x i 2 2 d log n. Note that E is a high probability event and conditioned on E, x i 's are still independent.

We omit the condition on E in the rest of the proof for simplicity.

We first show that assuming the following three inequalities that the conclusion of the Lemma follows.

DISPLAYFORM10 By bullets 1, 2, and Bernstein inequality, we have that with probability at least 1 − dn −10 over the randomness of the data (X, Y ), DISPLAYFORM11 κ log 1.5 n + nκ log 2 n nκ log 2 n By bullet 3 and equation above, we complete the proof with triangle inequality: DISPLAYFORM12 Therefore, it suffices to prove bullets 1, 2 and 3.

Note that 2 is a direct corollary of 1 so we will only prove 1 and 3.

We start with 3:By the definition of the 2 norm in L 2 (S d−1 ) and the independence of (x i , y i )'s, we can rewrite FIG3 .

Let z =ū −2 x −2 /τ , and therefore z has standard normal distribution.

With this change of the variables, by the definition of the distribution D, we have DISPLAYFORM13 DISPLAYFORM14 By claim D.4, and the 1-homogeneity of relu, we can simplify the above equation to DISPLAYFORM15 2) Combining equation D.1 and equation D.2 we complete the proof of bullet 3.

Next we prove bullet 1.

Note that ϕ(x)[ū]y is bounded by |ū 1 | + |ū 2 | + ū −2 x −2 2 .

Therefore, conditioned on DISPLAYFORM16 2 /d log n Hence we complete the proof.

Claim D.4.

Let Z ∼ N (0, 1) and a ∈ R. Then, there exists a universal constant c 1 and c 2 such that DISPLAYFORM17 Proof.

Without loss of generality we can assume a ≥ 0.

Then, DISPLAYFORM18 where the last equality uses the fact that c 1 DISPLAYFORM19 Now we will prove Theorem 4.3.Proof of Theorem 4.3.

To circumvent the technical issue of bounded support in Proposition 3.1 and Lemma 4.2, we construct p data to be a slightly modified version of D: perform rejection sampling of (x, y) ∼ D until we obtain a sample with x 2 2 d log n. Since this occurs with very high probability, the high probability result of Lemma D.3 still translates to p data .

Now apply Lemma D.2 to conclude that γ 2 log n √ n + 1 d .

Furthermore, Lemma D.1 allows us to conclude that γ 1 1.

We can therefore apply Proposition 3.1, and conclude that with probability 1 − δ, L 1-svm d log n n + log log(d log n) n + log(1/δ)

n Furthermore, plugging γ 2 into the bound of Lemma 4.2 gives us DISPLAYFORM20 We will first define the 1-norm and 2-norm regression problems.

The regression equivalent of equation 4.2 for α ∈ L 1 (S d−1 ) is as follows: DISPLAYFORM21 Next we define the regression version of equation 4.3: DISPLAYFORM22 We will briefly motivate our study of the regression setting by connecting the minimum 1-norm solution to neural networks.

To compare, in the classification setting, optimizing the weakly regularized loss over neural networks is equivalent to solving the 1 SVM.

In the regression setting, solving the weakly regularized squared error loss is equivalent is equivalent to finding the minimum 1-norm solution that fits the datapoints exactly.

Theorem D.5.

Let f (Θ; ·) be some two-layer neural network with m ≥ n hidden units parametrized by Θ, as in Section 4.

Define the λ-regularized squared error loss DISPLAYFORM23 Proof.

We can see that equation D.3 will have a n-sparse solution α using the same reasoning as the proof of Lemma B.1.

Furthermore, following the proof of Proposition 4.1, the function x → α , ϕ(x) is implementable by a neural network Θ ,m with Θ ,m 2 2 = 2 α 1 = 2 α 1 1 .

Following the same reasoning as before, we can also conclude that Θ ,m is an optimal solution for: DISPLAYFORM24 Under review as a conference paper at ICLR 2019 We proceed to provide similar generalization bounds as the classification setting.

This time, our bounds depend on the norms of the solution rather than the margin.

Let f φ (α; ·) x → α, ϕ(x) (for ϕ(x) defined in equation 4.1).

Following the convention in Section C.2, define hypothesis class DISPLAYFORM25 DISPLAYFORM26 B and letR(F) denote the empirical Rademacher complexity of hypothesis class F.We will first derive a Rademacher complexity bound for DISPLAYFORM27 Proof.

We writeR DISPLAYFORM28 We will now complete the bound onR(F 1,φ B ) for Lipschitz activations φ with φ(0) = 0.

Claim D.7.

Suppose that our activation φ is M -Lipschitz and φ(0) = 0.Then DISPLAYFORM29 Proof.

We will show that DISPLAYFORM30 from which the statement of the lemma follows via Claim D.6.

Fix anyū ∈ S d−1 .

Then we get the decomposition DISPLAYFORM31 We can bound the first term as DISPLAYFORM32 (since φ is Lipschitz and φ(0) = 0) DISPLAYFORM33 We note that the second term of equation D.6 can be bounded by DISPLAYFORM34 This follows from the general fact that the difference between the supremum and infimum of the absolute value of a quantity is bounded by the difference between the supremum and the infimum.

Furthermore, by symmetry of the Rademacher random variables, DISPLAYFORM35 This simply gives an empirical Rademacher complexity of the hypothesis class F {x → φ(ū x) : u ∈ S d−1 } scaled by n. By the Lipschitz contraction property of Rademacher complexity, using the fact that φ is M -Lipschitz, we can therefore bound equation D.8 by DISPLAYFORM36 DISPLAYFORM37 Proof.

Our starting point is Theorem 1 of Kakade et al. (2009) , which states that with probability 1 − δ, for any fixed hypothesis class F and f ∈ F, DISPLAYFORM38 n . and apply the above on F 1,φ Bj using δ j δ 2 j+1 .

Then using a union bound, with probability 1 − DISPLAYFORM39 Now for every α with α 1 < 1 X F , we use the inequality for F 1,φ B0 , and for every other α, we apply the inequality corresponding to F

Bj+1 , where 2 j ≤ α 1 X F ≤ 2 j+1 .

This gives the desired statement.

We can also provide the same generalization error bound for the 2-norm and relu features: Lemma D.9.

In the setting of Lemma D.8, choose φ to be the relu activation.

Then with probability DISPLAYFORM0 Proof.

We proceed the same way as in the proof of Lemma D.8.

We define B j as before, and this DISPLAYFORM1 from Corollary C.4.

Thus, again union bounding over all j, equation equation D.10 gives with probability 1 − δ, for all j ≥ 0 and f φ (α; ·) ∈ F 2,φ Bj DISPLAYFORM2 Now we assign the α to different j as before to obtain the statement in the lemma.

Note that if l is some bounded loss such that l(y; y) = 0 (for example, truncated squared error), for α 1 and α 2 the loss terms over the datapoints (in the bounds of Lemmas D.8 and D.9) vanish.

For loss l, define DISPLAYFORM3 Next, we will define the kernel matrix K with K ij = ϕ(x i ), ϕ(x j ) .

Now we are ready to state and prove the formal theorem describing the gap between the 1-norm solution and 2-norm solution.

Theorem D.10.

Recall the definitions of α 1 and α 2 in equation D.3 and equation D.4.

For any activation φ with the property that K is full rank for any X with no repeated datapoints, there exists a distribution p data such that with probability 1, DISPLAYFORM4 On the other hand, DISPLAYFORM5 2 2 ] = n κ For i.i.d samples from this choice of p data , if l is bounded (l(·; y) : R → [−1, 1]), 0 on correct predictions (l(y; y) = 0), and 1-Lipschitz, then with probability 1 − δ, DISPLAYFORM6 Meanwhile, in the case that α 2 2 2 ≥ n κ , the upper bound on L 2 -reg from Lemma D.9 is Ω(1) and in particular does not decrease with n.

We will first show that for any dataset X, there is a distribution over Y such that the expectation of α 2 2 is large.

When it is clear from context, y will denote the vector corresponding to Y .

DISPLAYFORM7 2 2 ] ≥ n κ and with probability 1, DISPLAYFORM8 We note the order of the quantifiers in Lemma D.11: the distribution A must not depend on the dataset X. We first provide a simple closed-form expression for α 2 2 2 .

Claim D.12.

If K is full rank, then α 2 2 2 = y K −1 y.

Proof.

This follows by taking the dual of equation D.4.Proof of Lemma D.11.

We sample β ∼ A as follows: first sampleū ∼ S d−1 uniformly.

Then set β to have a delta mass of 1 atū and be 0 everywhere else.

Define the vector vū [φ(ū x 1 ) · · · φ(ū x n )]; then it follows that we set our labels y to vū.

It is immediately clear that α 1 1 ≤ β 1 ≤ 1.

DISPLAYFORM9 Proof of Theorem D.10.

We note that since the distribution A of Lemma D.11 does not depend on the dataset X, it must follow that DISPLAYFORM10 2 2 ] = n κ Thus, there exists β such that if we sample X i.i.d.

from the standard normal and set y i = ϕ(x i ), β , the expectation of α 2 2 2 is at least n κ .

We choose p data corresponding to this β , with x sampled from the standard normal.

Now it is clear that p data will satisfy the norm conditions of Theorem D.10.For the generalization bounds, with high probability X F = Θ( √ nd) as x is sampled from the standard normal distribution.

Thus, Lemma D.8 immediately gives the desired generalization error bounds for L 1-reg .

On the other hand, if α 2 2 ≥ n κ , then the bound of Lemma D.9 is at least DISPLAYFORM11 We first write our regularity assumptions on Φ, R, and V in more detail:Assumption E.1 (Regularity conditions on Φ, R, V ).

R is convex, nonnegative, Lipschitz, and smooth: ∃M R , C R such that ∇ 2 R op ≤ C R , and ∇R 2 ≤ M R .Assumption E.2.

Φ is differentiable, bounded and Lipschitz on the sphere: DISPLAYFORM12 Assumption E.3.

V is Lipschitz and upper and lower bounded on the sphere: DISPLAYFORM13 We state the version of Theorem 5.3 that collects these parameters:Theorem E.4 (Theorem 5.3 with problem parameters).

Suppose that Φ and V are 2-homogeneous and Assumptions E.1, E.2, and E.3 hold.

Fix a desired error threshold > 0.

Suppose that from a starting distribution ρ 0 , a solution to the dynamics in equation 5.2 exists.

Choose DISPLAYFORM14 Throughout the proof, it will be useful to keep track of W t E θ∼ρt [ θ 2 2 ], the second moment of ρ t .

We first introduce a general lemma on integrals over vector field divergences.

DISPLAYFORM15 Proof.

The proof follows from integration by parts.

We note that ρ t will satisfy the boundedness condition of Lemma E.5 during the course of our algorithm -ρ 0 starts with this property, and Lemma E.9 proves that ρ t will continue to have this property.

We therefore freely apply Lemma E.5 in the remaining proofs.

We first bound the absolute value of L [ρ t ] over the sphere by DISPLAYFORM16 Proof.

We compute DISPLAYFORM17 Lemma E.7.

Under the perturbed Wasserstein gradient flow DISPLAYFORM18 Proof.

Applying the chain rule, we can compute DISPLAYFORM19 ], where we use Lemma E.5 with DISPLAYFORM20 Now we show that the decrease in objective value is approximately the average velocity of all parameters under ρ t plus some additional noise on the scale of σ.

At the end, we choose σ small enough so that the noise terms essentially do not matter.

DISPLAYFORM21 Proof.

By homogeneity, and Lemma DISPLAYFORM22 Combining these with Lemma E.7 gives the desired statement.

Now we show that if we run the dynamics for a short time, the second moment of ρ t will grow slowly, again at a rate that is roughly the scale of the noise σ.

DISPLAYFORM23 Integrating both sides of equation E.1, and rearranging, we get DISPLAYFORM24 We now plug this in and rearrange to get DISPLAYFORM25 The next statement allows us to argue that our dynamics will never increase the objective by too much.

Lemma E.10.

For any t 1 , t 2 with 0 DISPLAYFORM26 Integrating from t 1 to t 2 gives the desired result.

The following lemma bounds the change in expectation of a 2-homogeneous function over ρ t .

At a high level, we lower bound the decrease in our loss as a function of the change in this expectation.

DISPLAYFORM27 Proof.

Let Q(t) hdρ t .

We can compute: DISPLAYFORM28 Note that the first two terms are bounded by σB(W 2 + 1) by the assumptions for the lemma.

For the third term, we have from Lemma E.5: DISPLAYFORM29 (since h is Lipschitz on the sphere) DISPLAYFORM30 (by Corollary E.8)Plugging this into equation E.3, we get that DISPLAYFORM31 We apply this result to bound the change in L [ρ t ] over time in terms of the change of the objective value.

For clarity, we write the bound in terms of c 1 that is some polynomial in the problem constants.

Lemma E.12.

Define Q(t) DISPLAYFORM32 Integrating and applying the same reasoning to −L [ρ t ] gives us equation E.4.

Now we apply Lemma E.11 to get DISPLAYFORM33 We plug this into equation E.6 and then integrate both sides to obtain DISPLAYFORM34 Using c 1 max{kC R B 2 Φ , kC R B Φ M Φ , B L } gives the statement in the lemma.

Now we also show that L is Lipschitz on the unit ball.

For clarity, we let c 2 DISPLAYFORM35 Proof.

Using the definition of L and triangle inequality, DISPLAYFORM36 Now the remainder of the proof will proceed as follows: we show that if ρ t is far from optimality, either the expected velocity of θ under ρ t will be large in which case the loss decreases from Corollary E.8, or there will existθ such that L [ρ t ](θ) 0.

We will first show that in the latter case, the σU d noise term will grow mass exponentially fast in a descent direction until we make progress.

Define K Now the statement follows by Lemma 2.3 of BID3 ).Now we show that if a descent direction exists, the added noise will find it and our function value will decrease.

We start with a general lemma about the magnitude of the gradient of a 2-homogeneous function in the radial direction.

Lemma E.15.

Let h : R d+1 → R be a 2-homogeneous function.

Then for any θ ∈ R d+1 , θ ∇h(θ) = 2 θ 2 h(θ).

To circumvent the technical issue of existence of a solution to the continuous-time dynamics, we also note that polynomial time convergence holds for discrete-time updates.

Theorem E.21.

Along with Assumptions E.1, E.2, E.3 additionally assume that ∇Φ i and ∇V are C Φ and C V -Lipschitz, respectively.

Let ρ t evolve according to the following discrete-time update: DISPLAYFORM0 There exists a choice of DISPLAYFORM1 The proof follows from a standard conversion of the continuous-time proof of Theorem E.4 to discrete time, and we omit it here for simplicity.

and average over 100 trials for each plot point.

The left side of FIG5 shows the experimental results for synthetic data generated from a ground truth network with 10 hidden units, input dimension d = 20, and a ground truth unnormalized margin of at least 0.01.

We train for 80000 steps with learning rate 0.1 and λ = 10 −5 , using two-layer networks with 2 i hidden units for i ranging from 4 to 10.

We perform 20 trials per hidden layer size and plot the average over trials where the training error hit 0.

(At a hidden layer size of 2 7 or greater, all trials fit the training data perfectly.)

The right side of FIG5 demonstrates the same experiment, but performed on MNIST with hidden layer sizes of 2 i for i ranging from 6 to 15.

We train for 600 epochs using a learning rate of 0.01 and λ = 10 −6 and use a single trial per plot point.

For MNIST, all trials fit the training data perfectly.

The MNIST experiments are more noisy because we run one trial per plot point for MNIST, but the same trend of decreasing test error and increasing margin still holds.

We verify the normalized margin convergence on a two-layer networks with one-dimensional input.

A single hidden unit computes the following: x → a j relu(w j x + b j ).

We add · 2 2 -regularization to a, w, and b and compare the resulting normalized margin to that of an approximate solution of the 1 SVM problem with features relu(wx i + b) for w 2 + b 2 = 1.

Writing this feature vector is intractable, so we solve an approximate version by choosing 1000 evenly spaced values of (w, b).

Our theory predicts that with decreasing regularization, the margin of the neural network converges to the 1 SVM objective.

In FIG18 , we plot this margin convergence and visualize the final networks and ground truth labels.

The network margin approaches the ideal one as λ → 0, and the visualization shows that the network and 1 SVM functions are extremely similar.

Method CIFAR10 CIFAR100 Weight decay annealing 5.86 26.22 Fixed weight decay 6.01 27.00 Table 1 : Test error on CIFAR10 and CIFAR100 for initial λ = 0.0005.

We train a modified WideResNet architecture (Zagoruyko & Komodakis, 2016) on CIFAR10 and CIFAR100.

Our theory does not entirely apply because the identity mapping prevents ResNet architectures from being homogeneous, but our experiments show that reducing weight decay can still help generalization error in this setting.

Because batchnorm can cause the regularizer to have different effects (van Laarhoven, 2017), we remove batchnorm layers and train a 16 layer deep WideResNet.

We again compare a network trained with weight decayed annealing to one trained without annealing.

We used a fixed learning rate schedule that starts at 0.1 and decreases by a factor of 0.2 at epochs 60, 120, and 160.

For CIFAR10, we use an initial weight decay of 0.0002 and decrease the weight decay by 0.2 at epoch 60, and then by 0.5 at epochs 90, 120, 140, 160.

For CIFAR100, we initialize weight decay at 0.0005 and decrease it by 0.2 at epochs 60, 120, and 160.

We tried different parameters for the initial weight decay and chose the ones that worked best for the model without annealing.

We also tried using small weight decays at initialization, but these models failed to generalize well -we believe this is due to an optimization issue where the algorithm fails to find a true global minimum of the regularized loss.

We believe that annealing the weight decay directs the optimization algorithm closer towards the global minima for small λ.

Table 1 shows the test error achieved by models with and without annealing.

We see that the simple change of annealing weight decay can decrease the test error for this architecture.

<|TLDR|>

@highlight

We show that training feedforward relu networks with a weak regularizer results in a maximum margin and analyze the implications of this result.

@highlight

Studies margin theory for neural sets  and shows that max margin is monotonically increasing in size of the network

@highlight

This paper studies the implicit bias of minimizers of a regularized cross entropy loss of a two-layer network with ReLU activations, obtaining a generalization upper bound which does not increase with the network size.