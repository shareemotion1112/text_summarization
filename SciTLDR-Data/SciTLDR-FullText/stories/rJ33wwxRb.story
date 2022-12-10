Neural networks exhibit good generalization behavior in the over-parameterized regime, where the number of network parameters exceeds the number of observations.

Nonetheless, current generalization bounds for neural networks fail to explain this phenomenon.

In an attempt to bridge this gap, we study the problem of learning a two-layer over-parameterized neural network, when the data is generated by a linearly separable function.

In the case where the network has Leaky ReLU activations, we provide both optimization and generalization guarantees for over-parameterized networks.

Specifically, we prove convergence rates of SGD to a global minimum and provide generalization guarantees for this global minimum that are independent of the network size.

Therefore, our result clearly shows that the use of SGD for optimization both finds a global minimum, and avoids overfitting despite the high capacity of the model.

This is the first theoretical demonstration that SGD can avoid overfitting, when learning over-specified neural network classifiers.

Neural networks have achieved remarkable performance in many machine learning tasks.

Although recently there have been numerous theoretical contributions to understand their success, it is still largely unexplained and remains a mystery.

In particular, it is not known why in the overparameterized setting, in which there are far more parameters than training points, stochastic gradient descent (SGD) can learn networks that generalize well, as been observed in practice BID15 BID26 .In such over-parameterized settings, the loss function can contain multiple global minima that generalize poorly.

Therefore, learning can in principle lead to models with low training error, but high test error.

However, as often observed in practice, SGD is in fact able to find models with low training error and good generalization performance.

This suggests that the optimization procedure, which depends on the optimization method (SGD) and the training data, introduces some form of inductive bias which directs it towards a low complexity solution.

Thus, in order to explain the success of neural networks, it is crucial to characterize this inductive bias and understand what are the guarantees for generalization of over-parameterized neural networks.

In this work, we address these problems in a binary classification setting where SGD optimizes a two-layer over-parameterized network with the goal of learning a linearly separable function.

We study a relatively simple case of SGD where the weights of the second layer are fixed throughout the training process, and only the weights of the first layer are updated.

Clearly, an over-parameterized network is not necessary for classifying linearly separable data, since this is possible with linear classifiers (e.g., with the Perceptron algorithm) which also have good generalization guarantees BID20 .

But, the key question which we address here is whether a large network will overfit in such a case or not.

As we shall see, it turns out that although the networks we consider are rich enough to considerably overfit the data, this does not happen when SGD is used for optimization.

In other words, SGD introduces an inductive bias which allows it to learn over-parameterized networks that can generalize well.

Therefore, this setting serves as a good test bed for studying the effect of over-paramaterization.

Define X = {x ∈ R d : x ≤ 1}, Y = {±1}. We consider a distribution over linearly separable points.

Formally, let D be a distribution over X × Y such that there exists w * ∈ R d for which P (x,y)∼D (y w * , x ≥ 1) = 1.

1 Let S = {(x 1 , y 1 ), . . .

, (x n , y n )}

⊆

X × Y be a training set sampled i.i.d.

from D.

Consider the following two-layer neural network, with 2k > 0 hidden units.

3 The network parameters are W ∈ R 2k×d , v ∈ R 2k , which we denote jointly by W = (W, v).

The network output is given by the function N W : R d → R defined as: DISPLAYFORM0 where σ is a non-linear activation function applied element-wise.

We define the empirical loss over S to be the mean hinge-loss: DISPLAYFORM1 Note that for convenience of analysis, we will sometimes refer to L S as a function over a vector.

Namely, for a matrix W ∈ R 2k×d , we will consider instead its vectorized version W ∈ R 2kd (where the rows of W are concatenated) and define, with abuse of notation, that L S ( W ) = L S (W ).In our setting we fix the second layer to be v = ( DISPLAYFORM2 such that v > 0 and only learn the weight matrix W .

We will consider only positive homogeneous activations (Leaky ReLU and ReLU) and thus the network we consider with 2k hidden neurons is as expressive as networks with k hidden neurons and any vector v in the second layer.5 Hence, we can fix the second layer without limiting the expressive power of the two-layer network.

Although it is relatively simpler than the case where the second layer is not fixed, the effect of over-parameterization can be studied in this setting as well.

Hence, the objective of the optimization problem is to find: DISPLAYFORM3 where min DISPLAYFORM4 L S (W ) = 0 holds for the activations we will consider (Leaky ReLU and ReLU).1 This implies that w * ≥ 1.

2 Without loss of generality, we will ignore the event that yi w * , xi < 1 for some i, since this is an event of measure zero.

3 We have an even number of hidden neurons for ease of exposition.

See the definition of v below.

4 Our results hold in the case where the first layer contains bias terms.

This follows by the standard argument of adding another dimension to the input and setting the value 1 in the extra dimension for each data point.

5 For example, consider a network with k hidden neurons with positive homogeneous activations, where each hidden neuron i has incoming weight vector wi and outgoing weight vi.

Then, we can express this network with the network defined in Eq. 1 as follows.

For each i such that vi > 0, we define a neuron in the new network with incoming weight vector viwi and outgoing weight 1.

Similarly, if vi < 0, we define a neuron in the new network with incoming weight vector −viwi and outgoing weight −1.

For all other neurons in the new network we define an incoming zero weight vector.

Due to the positive homogeneity, it follows that this network is equivalent to the network with k hidden neurons.

We focus on the case where L S (W ) is minimized using an SGD algorithm with batch of size 1, and where only the weights of the first layer (namely W ) are updated.

At iteration t, SGD randomly chooses a point (x t , y t ) ∈ S and updates the weights with a constant learning rate η.

Formally, let W t = (W t , v) be the parameters at iteration t, then the update at iteration t is given by DISPLAYFORM5 We define a non-zero update at iteration t if it holds that ∂ ∂W L {(xt,yt)} (W t−1 ) = 0.

Finally, we will need the following notation.

For 1 ≤ i ≤ k, we denote by w (i) t ∈ R d the incoming weight vector of neuron i at iteration t. 6 Similarly, for 1 ≤ i ≤ k we define u (i) t ∈ R d to be the incoming weight vector of neuron k + i at iteration t.

We now present our main results, for the case where σ is the Leaky ReLU function.

Namely, σ(z) = max{αz, z} where 0 < α < 1.First, we show that SGD can find a global optimum of L S (W ).

Note that this is by no means obvious, since L S (W ) is a non-convex function (see Proposition 1).

Specifically, we show that SGD converges to such an optimum while making at most: DISPLAYFORM0 non-zero update steps (see Corollary 3).

In particular, the bound is independent of the number of neurons 2k.

To the best of our knowledge, this is the first convergence guarantee of SGD for neural networks with the hinge loss.

Furthermore, we prove a lower bound of Ω w * η + w * 2 for the number of non-zero updates (see Theorem 4).Next, we address the question of generalization.

As noted earlier, since the network is large, it can in principle overfit.

Indeed, there are parameter settings for which the network will have arbitrarily bad test error (see Section 6.2).

However, as we show here, this will not happen in our setting where SGD is used for optimization.

In Theorem 6 we use a compression bound to show that the model learned by SGD will have a generalization error of O M log n n .

7 This implies that for any network size, given a sufficiently large number of training samples that is independent of the network size, SGD converges to a global minimum with good generalization behaviour.

This is despite the fact that for sufficiently large k there are multiple global minima which overfit the training set (see Section 6.2).

This implies that SGD is biased towards solutions that can be expressed by a small set of training points and thus generalizes well.

To summarize, when the activation is the Leaky ReLU and the data is linearly separable, we provide provable guarantees of optimization, generalization and expressive power for over-parameterized networks.

This allows us to provide a rigorous explanation of the performance of over-parameterized networks in this setting.

This is a first step in unraveling the mystery of the success of overparameterized networks in practice.

We further study the same over-parameterized setting where the non-linear activation is the ReLU function (i.e., σ(z) = max{0, z}).

Surprisingly, this case has different properties.

Indeed, we show that the loss contains spurious local minima and thus the previous convergence result of SGD to a global minimum does not hold in this case.

Furthermore, we show an example where overparameterization is favorable from an optimization point of view.

Namely, for a sufficiently small number of hidden neurons, SGD will converge to a local minimum with high probability, whereas for a sufficiently large number of hidden neurons, SGD will converge to a global minimum with high probability.

The paper is organized as follows.

We discuss related work in Section 4 .

In Section 5 we prove the convergence bounds, in Section 6 we give the generalization guarantees and in Section 7 the results for the ReLU activation.

We conclude our work in Section 8.

The generalization performance of neural networks has been studied extensively.

Earlier results BID0 ) provided bounds that depend on the VC dimension of the network, and the VC dimension was shown to scale linearly with the number of parameters.

More recent works, study alternative notions of complexity, such as Rademacher compexity BID2 BID16 BID11 , Robustness BID24 ) and PAC-Bayes BID18 .

However, all of these notions do not provide provable guarantees for the generalization performance of over-parameterized networks trained with gradient based methods BID17 .

The main disadvantage of these approaches, is that they do not depend on the optimization method (e.g., SGD), and thus do not capture its role in the generalization performance.

In a recent paper, Dziugaite & Roy FORMULA0 numerically optimize a PAC-Bayes bound of a stochastic over-parameterized network in a binary classification task and obtain a nonvacuous generalization bound.

However, their bound is effective only when optimization succeeds, which their results do not guarantee.

In our work, we give generalization guarantees based on a compression bound that follows from convergence rate guarantees of SGD, and thus take into account the effect of the optimization method on the generalization performance.

This analysis results in generalization bounds that are independent of the network size and thus hold for over-parameterized networks.

Stability bounds for SGD in non-convex settings were given in Hardt et al. FORMULA0 ; BID12 .

However, their results hold for smooth loss functions, whereas the loss function we consider is not smooth due to the non-smooth activation functions (Leaky ReLU, ReLU).Other works have studied generalization of neural networks in a model recovery setting, where assumptions are made on the underlying model and the input distribution BID4 BID27 BID13 BID5 BID23 .

However, in their works the neural networks are not over-parameterized as in our setting.

BID21 analyze the optimization landscape of over-parameterized networks and give convergence guarantees for gradient descent to a global minimum when the data follows a Gaussian distribution and the activation functions are differentiable.

The main difference from our work is that they do not provide generalization guarantees for the resulting model.

Furthermore, we do not make any assumptions on the distribution of the feature vectors.

In a recent work, BID19 show that if training points are linearly separable then under assumptions on the rank of the weight matrices of a fully-connected neural network, every critical point of the loss function is a global minimum.

Their work extends previous results in BID9 ; BID7 ; BID25 .

Our work differs from these in several respects.

First, we show global convergence guarantees of SGD, whereas they only analyze the optimization landscape, without direct implications on performance of optimization methods.

Second, we provide generalization bounds and their focus is solely on optimization.

Third, we consider non-differentiable activation functions (Leaky ReLU, ReLU) while their results hold only for continuously differentiable activation functions.

In this section we consider the setting of Section 2 with a leaky ReLU activation function.

In Section 5.1 we show SGD will converge to a globally optimal solution, and analyze the rate of convergence.

In Section 5.1 we also provide lower bounds on the rate of convergence.

The results in this section are interesting for two reasons.

First, they show convergence of SGD for a non-convex objective.

Second, the rate of convergence results will be used to derive generalization bounds in Section 6.

Before proving convergence of SGD to a global minimum, we show that every critical point is a global minimum and the loss function is non-convex.

The proof is deferred to the appendix.

Proposition 1.

L S (W ) satisfies the following properties: 1) Every critical point is a global minimum.2) It is non-convex.

DISPLAYFORM0 be the vectorized version of W t and N t := N Wt where W t = (W t , v) (see Eq. 1).

Since we will show an upper bound on the number of non-zero updates, we will assume for simplicity that for all t we have a non-zero update at iteration t.

We assume that SGD is initialized such that the norms of all rows of W 0 are upper bounded by some constant R > 0.

Namely for all 1 ≤ i ≤ k it holds that: DISPLAYFORM1 We give an upper bound on the number of non-zero updates SGD makes until convergence to a critical point (which is a global minimum by Proposition 1).

The result is summarized in the following theorem.

Theorem 2.

SGD converges to a global minimum after performing at most M k non-zero updates.

We will briefly sketch the proof of Theorem 2.

The full proof is deferred to the Appendix (see Section 9.1.2).

The analysis is reminiscent of the Perceptron convergence proof (e.g. in ShalevShwartz & Ben-David (2014) ), but with key modifications due to the non-linear architecture.

Concretely, assume SGD performed t non-zero updates.

We consider the vector W t and the vec- DISPLAYFORM2 which is a global minimum of L S .

We define DISPLAYFORM3 DISPLAYFORM4 To obtain a simpler bound than the one obtained in Theorem 2, we use the fact that we can set R, v arbitrarily, and choose: DISPLAYFORM5 Then by Theorem 2 we get the following.

The derivation is given in the Appendix (Section 9.1.3).

DISPLAYFORM6 , then SGD converges to a global minimum after perfoming at most DISPLAYFORM7 Thus the bound consists of two terms, the first which only depends on the margin (via w * ) and the second which scales inversely with η.

More importantly, the bound is independent of the network size.

We use the same notations as in Section 5.1.

The lower bound is given in the following theorem, which is proved in the Appendix (Section 9.1.4).

Theorem 4.

Assume SGD is initialized according to Eq. 6, then for any d there exists a sequence of linearly separable points on which SGD will make at least Ω w * η + w * 2 mistakes.

Although this lower bound is not tight, it does show that the upper bound in Corollary 3 cannot be much improved.

Furthermore, the example presented in the proof of Theorem 4, demonstrates that η → ∞ can be optimal in terms of optimization and generalization, i.e., SGD makes the minimum number of updates ( w * 2 ) and the learned model is equivalent to the true classifier w * .

We will use this observation in the discussion on the dependence of the generalization bound in Theorem 6 on η (see Remark 1).

The bounds we provide in this section rely on the assumption that the weights of the second layer remain constant throughout the training process.

Although this does not limit the expressive power of the network, updating both layers effectively changes the dynamics of the problem, and it may not be clear why the above bounds apply to this case as well.

To answer this concern we show the following.

First, we run the same experiments as in FIG0 , but with both layers trained.

We show in Figure 2 that the training and generalization performance remain the same.

Second, in the complete proof of the upper bound given in Section 9.1.2, we relax the assumption that the weights of the second layer are fixed, and only assume that they do not change signs during the training process, and that their absolute values are bounded from below and from above.

This results in a similar bound, up to a constant factor.

We corroborate our theoretical result with experiments and show in Figure 3 that by choosing an appropriate constant learning rate, this in fact holds when updating both layers -the weights of the last layer do not change their sign, and are correctly bounded.

Furthermore, the performance of SGD is not affected by the choice of the learning rate.

A complete theoretical analysis of training both layers is left for future work.

In this section we give generalization guarantees for SGD learning of over-parameterized networks with Leaky ReLU activations.

These results are obtained by combining Theorem 2 with a compression generalization bound (see Section 6.1).

In Section 6.2 we show that over-parameterized networks are sufficiently expressive to contain global minima that overfit the training set.

Taken together, these results show that although there are models that overfit, SGD effectively avoids these, and finds the models that generalize well.

Given the bound in Theorem 2 we can invoke compression bounds for generalization guarantees with respect to the 0-1 loss BID14 .

Denote by N k a two-layer neural network with 2k hidden neurons defined in Section 1 where σ is the Leaky ReLU.

Let SGD k (S, W 0 ) be the output of running SGD for training this network on a set S and initialized with W 0 that satisfies Eq. 5.

Define H k to be the set of all possible hypotheses that SGD k (S, W 0 ) can output for any S and W 0 which satisfies Eq. 5.

Now, fix an initialization W 0 .

Then the key observation is that by Theorem 2 we have DISPLAYFORM0 Equivalently, SGD k (·, W 0 ) and B W0 define a compression scheme of size c k for hypothesis class H k (see Definition 30.4 in BID20 ).

Denote by V = {x j : j / ∈ {i 1 , ..., i c k }} the set of examples which were not selected to define DISPLAYFORM1 ) be the true risk of SGD k (S, W 0 ) and empirical risk of SGD k (S, W 0 ) on the set V , respectively.

Then by Theorem 30.2 and Corollary 30.3 in BID20 we can easily derive the following theorem.

The proof is deferred to the Appendix (Section 9.2.1).Theorem 5.

Let n ≥ 2c k , then with probability of at least 1 − δ over the choice of S and W 0 we have DISPLAYFORM2 We use a subscript W0 because the function is determined by W0.

Theorem 6.

If n ≥ 2c k and assuming the initialization defined in Eq. 6, then with probability at least 1 − δ over the choice of S and W 0 , SGD converges to a global minimum of L S with 0-1 test error at most DISPLAYFORM3 Thus for fixed w * and η we obtain a sample complexity guarantee that is independent of the network size (See Remark 1 for a discussion on the dependence of the bound on η).

This is despite the fact that for sufficiently large k, the network has global minima that have arbitrarily high test errors, as we show in the next section.

Thus, SGD and the linearly separable data introduce an inductive bias which directs SGD to the global minimum with low test error while avoiding global minima with high test error.

In FIG0 we demonstrate this empirically for a linearly separable data set (from a subset of MNIST) learned using over-parameterized networks.

The figure indeed shows that SGD converges to a global minimum which generalizes well.

Remark 1.

The generelization bound in Eq. 7 holds for η → ∞, which is unique for the setting that we consider, and may seem surprising, given that a choice of large η often fails in practice.

Furthermore, the bound is optimal for η → ∞. To support this theoretical result, we show in Theorem 4 an example where indeed η → ∞ is optimal in terms of the number of updates and generalization.

On the other hand, we note that in practice, it may not be optimal to use large η in our setting, since this bound results from a worst-case analysis of a sequence of examples encountered by SGD.

Finally, the important thing to note is that the bound holds for any η, and is thus applicable to realistic applications of SGD.

Let X ∈ R d×n be the matrix with the points x i in its columns, y ∈ {−1, 1} n the corresponding vector of labels and let N W (X) = v σ(W X) be the network defined in Eq. 1 applied on the matrix X. By Theorem 8 in BID22 we immediately get the following.

For completeness, the proof is given in the Appendix (Section 9.2.2).Theorem 7.

Assume that k ≥ 2 n 2d−2 .

Then for any y ∈ {−1, 1} n and for almost any X, DISPLAYFORM0 Theorem 7 implies that for sufficiently large networks, the optimization problem (2) can have arbitrarely bad global minima with respect to a given test set, i.e., ones which do not generalize well on a given test set.

In this section we consider the same setting as in section 5, but with the ReLU activation function σ(x) = max{0, x}. In Section 7.1 we show that the loss function contains arbitrarely bad local minima.

In Section 7.2 we give an example where for a sufficiently small network, with high probability SGD will converge to a local minimum.

On the other hand, for a sufficiently large network, with high probability SGD will converge to a global minimum.

The result is summarized in the following theorem and the proof is deferred to the Appendix (Section 9.3.1).

The main idea is to construct a network with weight paramater W such that for at least |S| 2 points (x, y) ∈ S it holds that w, x < 0 for each neuron with weight vector w. Furthermore, the remaining points satisfy yN W (x) > 1 and thus the gradient is zero and DISPLAYFORM0 .

Then, for every finite set of examples S ⊆ X × Y that is linearly separable, i.e., for which there exists w * ∈ R d such that for each (x, y) ∈ S we have y w * , x ≥ 1, there exists W ∈ R 2k×d such that W is a local minimum point with L S (W ) > 1 2 .

In this section we assume that S = {e 1 . . .

e d } × {1} ⊆ X × Y where {e 1 , . . .

, e d } is the standard basis of R d .

We assume all examples are labeled with the same label for simplicity, as the same result holds for the general case.

Let N Wt be the network obtained at iteration t, where W t = (W t , v).

Assume we initialize with DISPLAYFORM0 , and W 0 ∈ R 2k×d is randomly initialized from a continuous symmetric distribution with bounded norm, i.e DISPLAYFORM1 The main result of this section is given in the following theorem.

The proof is given in the Appendix (Section 9.3.2).

The main observation is that the convergence to non-global minimum depends solely on the initialization and occurs if and only if there exists a point x such that for all neurons, the corresponding initialized weight vector w satisfies w, x ≤ 0.

Theorem 9.

Fix δ > 0 and assume we run SGD with examples from S = {e 1 . . .

DISPLAYFORM2 , then with probability of at least 1 − δ, SGD will converge to a non global minimum point.

On the other hand, if k ≥ log 2 ( 2d δ ), then with probability of at least 1 − δ, SGD will converge to a global minimum point after max{ DISPLAYFORM3 10 That is, the set of entries of X which do not satisfy the statement is of Lebesgue measure 0.

Note that in the first part of the theorem, we can make the basin of attraction of the non-global minimum exponentially large by setting δ = e −αd for α ≤ 1 2 .

Understanding the performance of over-parameterized neural networks is essential for explaining the success of deep learning models in practice.

Despite a plethora of theoretical results for generalization of neural networks, none of them give guarantees for over-parameterized networks.

In this work, we give the first provable guarantees for the generalization performance of over-parameterized networks, in a setting where the data is linearly separable and the network has Leaky ReLU activations.

We show that SGD compresses its output when learning over-parameterized networks, and thus exhibits good generalization performance.

The analysis for networks with Leaky ReLU activations does not hold for networks with ReLU activations, since in this case the loss contains spurious local minima.

However, due to the success of over-parameterized networks with ReLU activations in practice, it is likely that similar results hold here as well.

It would be very interesting to provide convergence guarantees and generalization bounds for this case.

Another direction for future work is to show that similar results hold under different assumptions on the data.

9.1 MISSING PROOFS FOR SECTION 5 9.1.1 PROOF OF PROPOSITION 1 DISPLAYFORM0 the vector of all parameters where DISPLAYFORM1 Hence if we define DISPLAYFORM2 Otherwise, if yN W (x) ≥ 1, then the gradient vanishes and thus DISPLAYFORM3 It follows that if there exists (x, y) ∈ S, such that yN W (x) < 1, then we have DISPLAYFORM4 and thus DISPLAYFORM5 Therefore, for any critical point it holds that yN W (x) ≥ 1 for all (x, y) ∈ S, which implies that it is a global minimum.

2 ) which implies that the function is not convex.

We will start by analyzing a case with more relaxed assumptions -namely, we do not assume that the weights of the second layer are fixed, but rather that they do not change signs, and are bounded in absolute value.

Formally, let v (i) t be the weight of the second layer neuron corresponding to the weight vector w DISPLAYFORM0 t the weight corresponding to u (i) t .

Then we assume there exist c, C > 0 such that: DISPLAYFORM1 And note that we take v DISPLAYFORM2 Assume SGD performed t non-zero updates.

We will show that t ≤ M k .

We note that if there is no (x, y) ∈ S such that the corresponding update is non-zero, then SGD has reached a critical point of L S (which is a global minimum by Proposition 1).

Let DISPLAYFORM3 and note that L S ( W * ) = 0, i.e., W * is a global minimum.

Define the following two functions: DISPLAYFORM4 Then, from Cauchy-Schwartz inequality we have DISPLAYFORM5 Since the update at iteration t is non-zero, we have y t N t−1 (x t ) < 1 and the update rule is given by DISPLAYFORM6 where p DISPLAYFORM7 t−1 , x t ≥ 0 and q (i) t = α otherwise.

It follows that: DISPLAYFORM8 where the second inequality follows since y t DISPLAYFORM9 Using the above recursively, we obtain: DISPLAYFORM10 On the other hand, DISPLAYFORM11 where the inequality follows since y t x t , w * ≥ 1.

This implies that DISPLAYFORM12 By combining equations Eq. 9, Eq. 11 and Eq. 12 we get, DISPLAYFORM13 Since w DISPLAYFORM14 at ≤ b √ t + c where a = 2kηcvα, b = (4k 2 η 2 C 2 v 2 + 4ηk) w * and c = 4kR w * .

By inspecting the roots of the parabola P (x) = x 2 − b a x − c a we conclude that DISPLAYFORM15 Figure 2: Classifying MNIST images with over-parameterized networks and training both layers.

The setting of FIG0 is implemented, but now the second layer is trained as well.

The second layer is initialized as in FIG0 , i.e., all the weights are initialized to DISPLAYFORM16 .

The training and generalization performance are similar to the performance in the case where only the first layer is trained (see FIG0 .

DISPLAYFORM17 Figure 3: Classifying MNIST images with over-parameterized networks, training both layers and choosing an appropriate learning rate.

The setting of Figure 2 is implemented, but here a different learning rate is chosen for each network size, in order to satisfy the conditions of the proof in Section 9.1.2.

Figures (a) and (b) are train and test errors of MNIST classification for different network sizes and the chosen learning rates.

In this setting, SGD exhibits similar training and generalization performance as in Figure 2 .

Figure (c) shows the minimal and maximal value of the second layer weights divided by their initial value (denoted as c, C respectively in Section 9.1.2).

It can be seen that these values remain above zero, which implies that the weights do not flip signs during the training process (namely they satisfy the sign condition in Section 9.1.2) and that they behave similarly for different network sizes.

Notice that when assuming that the weights of the second layer are fixed, we get c = C = 1 and the above is simply equal to M k .

Otherwise, if c, C are independent constants, we get a similar bound, up to a constant factor.

Since R v = 1, we have by Theorem 2 and the inequality BID20 , for n ≥ 2c k we have that with probability of at least 1 − δ over the choice of S DISPLAYFORM0 DISPLAYFORM1 The above result holds for a fixed initialization W 0 .

We will show that the same result holds with high probability over S and W 0 , where W 0 is chosen independently of S and satisfies Eq. 5.

Define B to be the event that the inequality Eq. 15 does not hold.

Then we know that P S (B|W 0 ) ≤ δ for any fixed initialization W 0 .

11 Hence, by the law of total expectation, DISPLAYFORM2 We can easily extend Theorem 8 in BID22 to hold for labels in {−1, 1}. By the theorem we can construct networks N W1 and N W2 such that for all i: DISPLAYFORM3 9.3 MISSING PROOFS FOR SECTION 7 9.3.1 PROOF OF THEOREM 8We first need the following lemma.

Lemma 11.

There existsŵ ∈ R d that satisfies the following:1.

There exists α > 0 such that for each (x, y) ∈ S we have | x,ŵ | > α.

DISPLAYFORM4 Proof.

Consider the set V = {v ∈ R d : ∃ (x,y)∈S v, x = 0}. Clearly, V is a finite union of hyperplanes and therefore has measure zero, so there existsŵ ∈ R d \ V .

Let β = min (x,y)∈S {| ŵ, x |}, and since S is finite we clearly have α > 0.

Finally, if DISPLAYFORM5 we can chooseŵ and α = β 2 and we are done.

Otherwise, choosing −ŵ and α = β 2 satisfies all the assumptions of the lemma.

We are now ready to prove the theorem.

Chooseŵ ∈ R d that satisfies the assumptions in Lemma 11.

Now, let c > w * α , and let w = cŵ + w * and u = cŵ − w * .

Define DISPLAYFORM6 Let (x, y) ∈ S be an arbitrary example.

If ŵ, x > α, then DISPLAYFORM7 It follows that DISPLAYFORM8 Therefore yN W (x) > 1, so we get zero loss for this example, and therefore the gradient of the loss will also be zero.

If, on the other hand, ŵ, x < −α, then w, x = c ŵ, x + w * , x ≤ −cα + w * < 0 u, x = c ŵ, x − w * , x ≤ −cα + w * < 0 and therefore DISPLAYFORM9 In this case the loss on the example would be max{1 − yN W (x), 0} = 1, but the gradient will also be zero.

Along with assumption 2, we would conclude that: DISPLAYFORM10 Notice that since all the inequalities are strong, the following holds for all W ∈ R 2k×d that satisfies W − W < , for a small enough > 0.

Therefore, W ∈ R 2k×d is indeed a local minimum.

P (e j / ∈ K 0 ) DISPLAYFORM11 Therefore, with probability at least 1 − δ, there exists j ∈ [k] for which e j ∈ K 0 .

By Lemma 12, this implies that for all t ∈ N we will get e j ∈ K t , and therefore N Wt (e j ) ≤ 0.

Since e j is labeled 1, this implies that L S (W ) > 0.

By the separability of the data, and by the convergence of the SGD algorithm, this implies that the algorithm converges to a stationary point that is not a global minimum.

Note that convergence to a saddle point is possible only if we define σ (0) = 0, and for all i ∈ [k]

we have at the time of convergence w (i)t , e j = 0.

This can only happen if w (i) 0 , e j = ηN for some N ∈ N, which has probability zero over the initialization of w (i) t .

Therefore, the convergence is almost surely to a non-global minimum point.

On the other hand, assuming k ≥ log 2 ( d δ ), using the union bound we get: DISPLAYFORM12 So with probability at least 1 − δ, we get K 0 = ∅ and by Lemma 12 this means K t = ∅ for all t ∈ N. Now, if e j / ∈ K t for all t ∈ N, then there exists i ∈ [k] such that w (i)t , e j > 0 for all t ∈ N. If after performing T update iterations we have updated N > max{ Since we show that we never get stuck with zero gradient on an example with loss greater than zero, this means we converge to a global optimum after at most max{

@highlight

We show that SGD learns two-layer over-parameterized neural networks with Leaky ReLU activations that provably generalize on linearly separable data.

@highlight

The paper studies overparameterised models being able to learn well-generalising solutions by using a 1-hidden layer network with fixed output layer.

@highlight

This paper shows that on linearly seperabel data, SGD on an overparameterized network can still lean a classifier that provably generalizes.