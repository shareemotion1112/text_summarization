The information bottleneck (IB) problem tackles the issue of obtaining relevant compressed representations T of some random variable X for the task of predicting Y. It is defined as a constrained optimization problem which maximizes the information the representation has about the task, I(T;Y), while ensuring that a minimum level of compression r is achieved (i.e., I(X;T) <= r).

For practical reasons the problem is usually solved by maximizing the IB Lagrangian for many values of the Lagrange multiplier, therefore drawing the IB curve (i.e., the curve of maximal I(T;Y) for a given I(X;Y)) and selecting the representation of desired predictability and compression.

It is known when Y is a deterministic function of X, the IB curve cannot be explored and other Lagrangians have been proposed to tackle this problem (e.g., the squared IB Lagrangian).

In this paper we (i) present a general family of Lagrangians which allow for the exploration of the IB curve in all scenarios; (ii) prove that if these Lagrangians are used, there is a one-to-one mapping between the Lagrange multiplier and the desired compression rate r for known IB curve shapes, hence, freeing from the burden of solving the optimization problem for many values of the Lagrange multiplier.

Let X and Y be two statistically dependent random variables with joint distribution p(x, y).

The information bottleneck (IB) (Tishby et al., 2000) investigates the problem of extracting the relevant information from X for the task of predicting Y .

For this purpose, the IB defines a bottleneck variable T obeying the Markov chain Y ↔ X ↔ T so that T acts as a representation of X. Tishby et al. (2000) define the relevant information as the information the representation keeps from Y after the compression of X (i.e., I(T ; Y )), provided a minimum level of compression (i.e, I(X; T ) ≤ r).

Therefore, we select the representation which yields the value of the IB curve that best fits our requirements.

Definition 1 (IB functional).

Let X and Y be statistically dependent variables.

Let ∆ be the set of random variables T obeying the Markov condition Y ↔ X ↔ T .

Then the IB functional is F IB,max (r) = max T ∈∆ {I(T ; Y )} s.t.

I(X; T ) ≤ r, ∀r ∈ [0, ∞).

(1)

Definition 2 (IB curve).

The IB curve is the set of points defined by the solutions of F IB,max (r) for varying values of r ∈ [0, ∞).

Definition 3 (Information plane).

The plane is defined by the axes I(T ; Y ) and I(X; T ).

In practice, solving a constrained optimization problem such as the IB functional is difficult.

Thus, in order to avoid the non-linear constraints from the IB functional the IB Lagrangian is defined.

Definition 4 (IB Lagrangian).

Let X and Y be statistically dependent variables.

Let ∆ be the set of random variables T obeying the Markov condition Y ↔ X ↔ T .

Then we define the IB Lagrangian as L β IB (T ) = I(T ; Y ) − βI(X; T ).

Here β ∈ [0, 1] is the Lagrange multiplier which controls the trade-off between the information of Y retained and the compression of X. Note we consider β ∈ [0, 1] because (i) for β ≤ 0 many uncompressed solutions such as T = X maximizes L β IB , and (ii) for β ≥ 1 the IB Lagrangian is non-positive due to the data processing inequality (DPI) (Theorem 2.8.1 from Cover & Thomas (2012) ) and trivial solutions like T = const are maximizers with L β IB = 0 (Kolchinsky et al., 2019) .

We know the solutions of the IB Lagrangian optimization (if existent) are solutions of the IB functional by the Lagrange's sufficiency theorem (Theorem 5 in Appendix A of Courcoubetis (2003) ).

Moreover, since the IB functional is concave (Lemma 5 of Gilad-Bachrach et al. (2003) ) we know they exist (Theorem 6 in Appendix A of Courcoubetis (2003) ).

Therefore, the problem is usually solved by maximizing the IB Lagrangian with adaptations of the Blahut-Arimoto algorithm (Tishby et al., 2000) , deterministic annealing approaches (Tishby & Slonim, 2001 ) or a bottom-up greedy agglomerative clustering (Slonim & Tishby, 2000) or its improved sequential counterpart (Slonim et al., 2002) .

However, when provided with high-dimensional random variables X such as images, these algorithms do not scale well and deep learning based techniques, where the IB Lagrangian is used as the objective function, prevailed (Alemi et al., 2017; Chalk et al., 2016; .

Note the IB Lagrangian optimization yields a representation T with a given performance (I(X; T ), I(T ; Y )) for a given β.

However there is no one-to-one mapping between β and I(X; T ).

Hence, we cannot directly optimize for a desired compression level r but we need to perform several optimizations for different values of β and select the representation with the desired performance (e.g., Alemi et al. (2017) ).

The Lagrange multiplier selection is important since (i) sometimes even choices of β < 1 lead to trivial representations such that p T |X (t|x) = p T (t), and (ii) there exist some discontinuities on the performance level w.r.t.

the values of β (Wu et al., 2019).

Moreover, recently Kolchinsky et al. (2019) showed how in deterministic scenarios (such as many classification problems where an input x i belongs to a single particluar class y i ) the IB Lagrangian could not explore the IB curve.

Particularly, they showed that multiple β yielded the same performance level and that a single value of β could result in different performance levels.

To solve this issue, they introduced the squared IB Lagrangian, L βsq sq-IB = I(T ; Y ) − β sq I(X; T ) 2 , which is able to explore the IB curve in any scenario by optimizing for different values of β sq .

However, even though they realized a one-to-one mapping between β s q existed, they did not find such mapping.

Hence, multiple optimizations of the Lagrangian were still required to fing the best traded-off solution.

The main contributions of this article are:

1.

We introduce a general family of Lagrangians (the convex IB Lagrangians) which are able to explore the IB curve in any scenario for which the squared IB Lagrangian (Kolchinsky et al., 2019 ) is a particular case of.

More importantly, the analysis made for deriving this family of Lagrangians can serve as inspiration for obtaining new Lagrangian families which solve other objective functions with intrinsic trade-off such as the IB Lagrangian.

2.

We show that in deterministic scenarios (and other scenarios where the IB curve shape is known) one can use the convex IB Lagrangian to obtain a desired level of performance with a single optimization.

That is, there is a one-to-one mapping between the Lagrange multiplier used for the optmization and the level of compression and informativeness obtained, and we know such mapping.

Therefore, eliminating the need of multiple optimizations to select a suitable representation.

Furthermore, we provide some insight for explaining why there are discontinuities in the performance levels w.r.t.

the values of the Lagrange multipliers.

In a classification setting, we connect those discontinuities with the intrinsic clusterization of the representations when optimizing the IB bottleneck objective.

The structure of the article is the following: in Section 2 we motivate the usage of the IB in supervised learning settings.

Then, in Section 3 we outline the important results used about the IB curve in deterministic scenarios.

Later, in Section 4 we introduce the convex IB Lagrangian and explain some of its properties.

After that, we support our (proved) claims with some empirical evidence on the MNIST dataset (LeCun et al., 1998) in Section 5.

The reader can download the PyTorch (Paszke et al., 2017) implementation at https://gofile.io/?c=G9Dl1L.

In this section we will first give an overview of supervised learning in order to later motivate the usage of the information bottleneck in this setting.

In supervised learning we are given a dataset

of n pairs of input features and task outputs.

In this case, X and Y are the random variables of the input features and the task outputs.

We assume x i and y i are sampled i.i.d.

from the true distribution p XY (x, y) = p Y |X (y|x)p X (x).

The usual aim of supervised learning is to use the dataset D n to learn a particular conditional distribution qŶ |X,θ (ŷ|x) of the task outputs given the input features, parametrized by θ, which is a good approximation of p Y |X (y|x).

We useŶ andŷ to indicate the predicted task output random variable and its outcome.

We call a supervised learning task regression when Y is continuous-valued and classification when it is discrete.

Usually supervised learning methods employ intermediate representations of the inputs before making predictions about the outputs; e.g., hidden layers in neural networks (Chapter 5 from Bishop (2006)) or transformations in a feature space through the kernel trick in kernel machines like SVMs or RVMs (Sections 7.1 and 7.2 from Bishop (2006)).

Let T be a possibly stochastic function of the input features X with a parametrized conditional distribution q T |X,θ (t|x), then, T obeys the Markov condition Y ↔ X ↔ T .

The mapping from the representation to the predicted task outputs is defined by the parametrized conditional distribution qŶ |T,θ (ŷ|t).

Therefore, in representation-based machine learning methods the full Markov Chain is Y ↔ X ↔ T ↔Ŷ .

Hence, the overall estimation of the conditional probability p Y |X (y|x) is given by the marginalization of the representations,

In order to achieve the goal of having a good estimation of the conditional probability distribution p Y |X (y|x), we usually define an instantaneous cost function j θ (x, y) : X × Y → R. This serves as a heuristic to measure the loss our algorithm (parametrized by θ) obtains when trying to predict the realization of the task output y with the input realization x.

Clearly, we are interested in minimizing the expectation of the instantaneous cost function over all the possible input features and task outputs, which we call the cost function.

However, since we only have a finite dataset D n we have instead to minimize the empirical cost function.

Definition 5 (Cost function and empirical cost function).

Let X and Y be the input features and task output random variables and x ∈ X and y ∈ Y their realizations.

Let also j θ (x, y) be the instantaneous cost function, θ the parametrization of our learning algorithm, and

the given dataset.

Then we define:

1.

The cost function:

2.

The emprical cost function:

The discrepancy between the normal and empirical cost functions is called the generalization gap or generalization error (see Section 1 of Xu & Raginsky (2017) , for instance) and intuitevely, the smaller this gap is, the better our model generalizes (i.e., the better it will perform to new, unseen samples in terms of our cost function).

Definition 6 (Generalization gap).

Let J(θ) andĴ(θ, D n ) be the cost and the empirical cost functions as defined in Definition 5.

Then, the generalization gap is defined as

and it represents the error incurred when the selected distribution is the one parametrized by θ when the ruleĴ(θ, D n ) is used instead of J(θ) as the function to minimize.

Ideally, we would want to minimize the cost function.

Hence, we usually try to minimize the empirical cost function and the generalization gap simultaneously.

The modifications to our learning algorithm which intend to reduce the generalization gap but not hurt the performance on the empirical cost function are known as regularization.

Definition 7 (Representation cross-entropy cost function).

Let X and Y be two statistically dependent variables with joint distribution p XY (x, y) = p Y |X (y|x)p X (x).

Let also T be a random variable obeying the Markov condition Y ↔ X ↔ T and q T |X,θ (t|x) and qŶ |T,θ (ŷ|t) be the encoding and decoding distributions of our model, parametrized by θ.

Finally, let C(p(z)||q(z)) = −E p(Z) [log(q(z))] be the cross entropy between two probability distributions p and q. Then, the cross-entropy cost function is

where j CE,θ (x, y) = C(q T |X,θ (t|x)||qŶ |T,θ (ŷ|t)) is the instantaneous representation cross-entropy cost function and

The cross-entropy is a widely used cost function in classification tasks (e.g., Krizhevsky et al. (2012) ; Shore & Gray (1982) ; Teahan (2000)) which has many interesting properties (Shore & Johnson, 1981) .

Moreover, it is known that minimizing the J CE (θ) maximizes the mutual information I(T ; Y ) (see Section 2 of Kolchinsky et al. (2019) or Section II A. of Vera et al. (2018)).

Definition 8 (Nuisance).

A nuisance is any random variable which affects the observed data X but is not informative to the task we are trying to solve.

That is, Ξ is a nuisance for

Similarly, we know that minimizing I(X; T ) minimizes the generalization gap for restricted classes when using the cross-entropy cost function (Theorem 1 of Vera et al. (2018)), and when using I(T ; Y ) directly as an objective to maximize (Theorem 4 of Shamir et al. (2010) ).

Furthermore, Achille & Soatto (2018) in Proposition 3.1 upper bound the information of the input representations, T , with nuisances that affect the observed data, Ξ, with I(X; T ).

Therefore minimizing I(X; T ) helps generalization by not keeping useless information of Ξ in our representations.

Thus, jointly maximizing I(T ; Y ) and minimizing I(X; T ) is a good choice both in terms of performance in the available dataset and in new, unseen data, which motivates studies on the IB.

Kolchinsky et al. (2019) showed that when Y is a deterministic function of X (i.e., Y = f (X)), the IB curve is piecewise linear.

More precisely, it is shaped as stated in Proposition 1.

Proposition 1 (The IB curve is piecewise linear in deterministic scenarios).

Let X be a random variable and Y = f (X) be a deterministic function of X. Let also T be the bottleneck variable that solves the IB functional.

Then the IB curve in the information plane is defined by the following equation:

Furthermore, they showed that the IB curve could not be explored by optimizing the IB Lagrangian for multiple β because the curve was not strictly concave.

That is, there was not a one-to-one relationship between β and the performance level.

Theorem 1 (In deterministic scenarios, the IB curve cannot be explored using the IB Lagrangian).

Let X be a random variable and Y = f (X) be a deterministic function of X. Let also T be the bottleneck variable that solves arg max T ∈∆ {L β IB } with ∆ the set of r.v.

obeying the Markov condition Y ↔ X ↔ T .

Then:

is the only solution β ∈ (0, 1) yields.

Clearly, a situation like the one depicted in Theorem 1 is not desirable, since we cannot aim for different levels of compression or performance.

For this reason, we generalize the effort from Kolchinsky et al. (2019) and look for families of Lagrangians which are able to explore the IB curve.

Inspired by the squared IB Lagrangian,

, we look at the conditions a function of I(X; T ) requires in order to be able to explore the IB curve.

In this way, we realize that any monotonically increasing and strictly convex function will be able to do so, and we call the family of Lagrangians with these characteristics the convex IB Lagrangians, due to the nature of the introduced function.

Theorem 2 (Convex IB Lagrangians).

Let ∆ be the set of r.v.

T obeying the Markov condition Y ↔ X ↔ T .

Then, if h is a monotonically increasing and strictly convex function, the IB curve can always be recovered by the solutions of arg max T ∈∆ {L

That is, for each point (I(X; T ),

IB,h (T ) achieves this solution.

Furthermore, β h is strictly decreasing w.r.t.

The proof of this theorem can be found on Appendix A. Furthermore, by exploiting the IB curve duality (Lemma 10 of Gilad-Bachrach et al. (2003)) we were able to derive other families of Lagrangians which allow for the exploration of the IB curve (Appendix E).

Remark 1.

Clearly, we can see how if h is the identity function (i.e., h(I(X; T )) = I(X; T )) then we end up with the normal IB Lagrangian.

However, since the identity function is not strictly convex, it cannot ensure the exploration of the IB curve.

Let B h denote the domain of Lagrange multipliers β h for which we can find solutions in the IB curve with the convex IB Lagrangian.

Then the convex IB Lagrangians do not only allow us to explore the IB curve with different β h .

They also allow us to identify the specific β h that obtains a given point (I(X; T ), I(T ; Y )), provided we know the IB curve in the information plane.

Conversely, the convex IB Lagrangian allows to find the specific point (I(X; T ), I(T ; Y )) that is obtained by a given β h .

Proposition 2 (Bijective mapping between IB curve point and convex IB Lagrange multiplier).

Let the IB curve in the information plane be known; i.e., I(T ; Y ) = f IB (I(X; T )) is known.

Then there is a bijective mapping from Lagrange multipliers β h ∈ B h \{0} from the convex IB Lagrangian to points in the IB curve (I(X; T ), f IB (I(X; T )).

Furthermore, these mappings are:

and I(X; T ) = (h )

where h is the derivative of h and (h ) −1 is the inverse of h .

It is interesting since in deterministic scenarios we know the shape of the IB curve (Theorem 1) and since the convex IB Lagrangians allow for the exploration of the IB curve (Theorem 2).

A proof for Proposition 2 can be found in Appendix B. A direct result derived from this proposition is that we know the domain of Lagrange multipliers, B h , which allow for the exploration of the IB curve if the shape of the IB curve is known.

Furthermore, if the shape is not known we can at least bound that range.

Corollary 1 (Domain of convex IB Lagrange multiplier with known IB curve shape).

Let the IB curve in the information plane be I(T ; Y ) = f IB (I(X; T )) and let I max = I(X; Y ).

Let also I(X; T ) = r max be the minimum mutual information s.t.

f IB (r max ) = I max (i.e., r max = min r {f IB (r) = I max }).

Then, the range of Lagrange multipliers that allow the exploration of the IB curve with the convex IB Lagrangian is B h = [β h,min , β h,max ], with

where f IB (r) and h (r) are the derivatives of f IB (I(X; T )) and h(I(X; T )) w.r.t.

I(X; T ) evaluated at r respectively.

Corollary 2 (Domain of convex IB Lagrange multiplier bound).

h (r) is the derivative of h(I(X; T )) w.r.t.

I(X; T ) evaluated at r, X is the set of possible realizations of X and β 0 1 and Ω x are defined as in (Wu et al., 2019) .

That is,

Corollaries 1 and 2 allow us to reduce the range search for β when we want to explore the IB curve.

Practically, inf Ωx⊂X {β 0 (Ω x )} might be difficult to calculate so Wu et al. (2019) derived an algorithm to approximate it.

However, we still recommend 1 for simplicity.

The proofs for both corollaries are found in Appendices C and D.

In order to showcase our claims we use the MNIST dataset (LeCun et al., 1998) .

We simply modify the nonlinear-IB method , which is a neural network that minimizes the cross-entropy while also minimizing a differentiable kernel-based estimate of I(X; T ) (Kolchinsky & Tracey, 2017).

Then we use this technique to maximize a lower bound on the convex IB Lagrangians by applying the functions h to the I(X; T ) estimate.

For a fair comparison, we use the same network architecture as that in :

Here f θ,enc is a three fully-conected layer encoder with 800 ReLU units on the first two layers and 2 linear units on the last layer.

Second, a deterministic decoder qŶ |T,θ (ŷ|t) = f θ,dec (t).

Here, f θ,dec is a fully-conected 800 ReLU unit layers followed by an output layer with 10 softmax units.

For further details about the experiment setup and additional results for different values of α and η please refer to Appendix F.

In Figure 1 we show our results for two particularizations of the convex IB Lagrangians:

1 Note in (Wu et al., 2019) they consider the dual problem (see Appendix E) so when they refer to β −1 it translates to β in this article.

2 The encoder needs to be stochastic to (i) ensure a finite and well-defined mutual information (Kolchinsky et al., 2019; Amjad & Geiger, 2019) and (ii) make gradient-based optimization methods over the IB Lagrangian useful (Amjad & Geiger, 2019) .

3 The clusters were obtained using the DBSCAN algorithm (Ester et al., 1996; Schubert et al., 2017) .

We can clearly see how both Lagrangians are able to explore the IB curve (first column from Figure  1 ) and how the theoretical performance trend of the Lagrangians matches the experimental results (second and third columns from Figure 1 ).

There are small mismatches between the theoretical and experimental performance.

This is because using the nonlinear-IB, as stated by Kolchinsky et al. (2019) , does not guarantee that we find optimal representations due to factors like: (i) innacurate estimation of I(X; T ), (ii) restrictions on the structure of T , (iii) use of an estimation of the decoder instead of the real one and (iv) the typical non-convex optimization issues that arise with gradient-based methods.

The main difference comes from the discontinuities in performance for in-creasing β, which cause is still unknown (cf.

Wu et al. (2019)).

It has been observed, however, that the bottleneck variable performs an intrinsic clusterization in classification tasks (see, for instance Alemi et al., 2018) or Figure 2b ).

We realized how this clusterization matches with the quantized performance levels observed (e.g., compare Figure 2a with the top center graph in Figure 1) ; with maximum performance when the number of clusters is equal to the cardinality of Y and reducing performance with a reduction of the number of clusters.

We do not have a mathematical proof for the exact relationship between these two phenomena; however, we agree with Wu et al. (2019) that it is an interesting matter and hope this realization serves as motivation to derive new theory.

To sum up, in order to achieve a desired level of performance with the convex IB Lagrangian as an objective one should:

1.

In a deterministic or close to deterministic setting (see -deterministic definition in Kolchinsky et al. (2019)): Use the adequate β h for that performance using Proposition 2.

Then if the perfomance is lower than desired (i.e., we are placed in the wrong performance plateau), gradually reduce the value of β h until reaching the previous performance plateau.

2.

In a stochastic setting: Draw the IB curve with multiple values of β h on the range defined by Corollary 2 and select the representations that best fit their interests.

= In practice, there are different criterions for choosing the function h. For instance, the exponential IB Lagrangian could be more desirable than the power IB Lagrangian when we want to draw the IB curve since it has a finite range of β h .

This is

−1 , ∞) for the power IB Lagrangian.

Furthermore, there is a trade-off between (i) how much the selected h function ressembles the identity (e.g., with α or η close to zero), since it will suffer from similar problems as the original IB Lagrangian; and (ii) how fast it grows (e.g., higher values of α or η), since it will suffer from value convergence; i.e., optimizing for separate values of β h will achieve similar levels of performance (Figure 3) .

Please, refer to Appendix G for a more thorough explanation of this phenomenon.

The information bottleneck is a widely used and studied technique.

However, it is known that the IB Lagrangian cannot be used to achieve varying levels of performance in deterministic scenarios.

Moreover, in order to achieve a particular level of performance multiple optimizations with different Lagrange multipliers must be done to draw the IB curve and select the best traded-off representation.

In this article we introduced a general family of Lagrangians which allow to (i) achieve varying levels of performance in any scenario, and (ii) pinpoint a specific Lagrange multiplier β h to optimize for a specific performance level in known IB curve scenarios (e.g., deterministic).

Furthermore, we showed the β h domain when the IB curve is known and a β h domain bound for exploring the IB curve when it is unkown.

This way we can reduce and/or avoid multiple optimizations and, hence, reduce the computational effort for finding well traded-off representations.

Finally, (iii) we provided some insight to the discontinuities on the performance levels w.r.t.

the Lagange multipliers by connecting those with the intrinsic clusterization of the bottleneck variable.

Proof.

We start the proof by remembering the optimization problem at hand (Definition 1):

We can modify the optimization problem by

iff h is a monotonically non-decreasing function since otherwise h(I(X; T )) ≤ h(r) would not hold necessarily.

Now, let us assume ∃T * ∈ ∆ and β * h s.t.

T * maximizes L β * h IB,h (T ) over all T ∈ ∆, and I(X; T * ) ≤ r.

Then, we can operate as follows:

Here, the equality from equation (15) comes from the fact that since I(X; T ) ≤ r, then ∃ξ ≥ 0 s.t.

h(I(X; T )) − h(r) + ξ = 0.

Then, the inequality from equation (16) holds since we have expanded the optimization search space.

Finally, in equation (17) we use that T * maximizes L β * h IB,h (T ) and that I(X; T * ) ≤ r. Now, we can exploit that h(r) and ξ do not depend on T and drop them in the maximization in equation (16).

We can then realize we are maximizing over L

Therefore, since I(T * ; Y ) satisfies both the maximization with T * ∈ ∆ and the constraint

.

Now, we know if such β * h exists, then the solution of the Lagrangian will be a solution for F IB,max (r).

Then, if we consider Theorem 6 from the Appendix of Courcoubetis (2003) and consider the maximization problem instead of the minimization problem, we know if both I(T ; Y ) and −h(I(X; T )) are concave functions, then a set of Lagrange multipliers S * h exists with these conditions.

We can make this consideration because f is concave if −f is convex and max{f } = min{−f }.

We know I(T ; Y ) is a concave function of T for T ∈ ∆ (Lemma 5 of Gilad-Bachrach et al. (2003)) and I(X; T ) is convex w.r.t.

T given p X (x) is fixed (Theorem 2.7.4 of Cover & Thomas (2012)).

Thus, if we want −h(I(X; T )) to be concave we need h to be a convex function.

Finally, we will look at the conditions of h so that for every point (I(X; T ), I(T ; Y )) in the IB curve, there exists a unique β * h s.t.

L β * h IB,h (T ) is maximized.

That is, the conditions of h s.t.

|S * h | = 1.

For this purpose we will look at the solutions of the Lagrangian optimization:

Now, if we integrate both sides of equation (20) over all T ∈ ∆ we obtain

where β is the Lagrange multiplier from the IB Lagrangian (Tishby et al., 2000) and h (I(X; T )) is dh(I(X;T )) dI(X;T ) .

Also, if we want to avoid indeterminations of β h we need h (I(X; T )) not to be 0.

Since we already imposed h to be monotonically non-decreasing, we can solve this issue by strengthening this condition.

That is, we will require h to be monotonically increasing.

We would like β h to be continuous, this way there would be a unique β h for each value of I(X; T ).

We know β is a non-increasing function of I(X; T ) (Lemma 6 of Gilad-Bachrach et al. (2003)).

Hence, if we want β h to be a strictly decreasing function of I(X; T ), we will require h to be an strictly increasing function of I(X; T ).

Therefore, we will require h to be a strictly convex function.

Thus, if h is an strictly convex and monotonically increasing function, for each point (I(X; T ), I(T ; Y )) in the IB curve s.t.

dI(T ; Y )/dI(X; T ) > 0 there is a unique β h for which maximizing L β h IB,h (T ) achieves this solution.

Proof.

In Theorem 2 we showed how each point of the IB curve (I(X; T ), I(T ; Y )) can be found with a unique β h maximizing L β h IB,h .

Therefore since we also proved L β h IB,h is strictly concave w.r.t.

T we can find the values of β h that maximize the Lagrangian for fixed I(X; T ).

First, we look at the solutions of the Lagrangian maximization:

Then as before we can integrate at both sides for all T ∈ ∆ and solve for β h :

Moreover, since h is a strictly convex function its derivative h is strictly decreasing.

Hence, h is an invertible function (since a strictly decreasing function is bijective and a function is invertible iff it is bijective by definition).

Now, if we consider β h > 0 to be known and I(X; T ) to be the unknown we can solve for I(X; T ) and get:

Note we require β h not to be 0 so the mapping is defined.

Proof.

, we see that maximizing this Lagrangian is directly maximizing I(T ; Y ).

We know I(T ; Y ) is a concave function of T for T ∈ ∆ (Theorem 2.7.4 from Cover & Thomas (2012)); hence it has a maximum.

We also know I(T ; Y ) ≤ I(X; Y ).

Moreover, we know I(X; Y ) can be achieved if, for example, Y is a deterministic function of T (since then the Markov Chain X ↔ T ↔ Y is formed).

Thus, max T ∈∆ {L 0

For β h = 0 we know maximizing L IB,h (T ) can obtain the point in the IB curve (r max , I max ) (Lemma 1).

Moreover, we know that for every point (I(X; T ), f IB (I(X; T ))), ∃!β h s.t.

max{L β h IB,h (T )} achieves that point (Theorem 2).

Thus, ∃!β h,min s.t.

lim r→r − max (r, f IB (r)) is achieved.

From Proposition 2 we know this β h,min is given by

Since we know f IB (I(X; T )) is a concave non-decreasing function in (0, r max ) (Lemma 5 of GiladBachrach et al. (2003))

we know it is continuous in this interval.

In addition we know β h is strictly decreasing w.r.t.

I(X; T ) (Theorem 2).

Furthermore, by definition of r max and knowing I(T ; Y ) ≤ I(X; Y ) we know f IB (r) = 0, ∀r > r max .

Therefore, we cannot ensure the exploration of the IB curve for β h s.t.

0 < β h < β h,min .

Then, since h is a strictly increasing function in (0, r max ), h is positive in that interval.

Hence, taking into account β h is strictly decreasing we can find a maximum β h when I(X; T ) approaches to 0.

That is,

D PROOF OF COROLLARY 2

Proof.

If we use Corollary 1, it is straightforward to see that

for all IB curves f IB and functions h. Therefore, we look at a domain bound dependent on the function choice.

That is, if we can find β min ≤ f IB (r) and β max ≥ f IB (r) for all IB curves and all values of r, then

The region for all possible IB curves regardless of the relationship between X and Y is depicted in Figure 4 .

The hard limits are imposed by the DPI (Theorem 2.8.1 from Cover & Thomas (2012)) and the fact that the mutual information is non-negative (Corollary 2.90 for discrete and first Corollary of Theorem 8.6.1 for continuous random variables from Cover & Thomas (2012)).

Hence, a minimum and maximum values of f IB are given by the minimum and maximum values of the slope of the Pareto frontier.

Which means

Note 0/(lim r→r − max {h (r)}) = 0 since h is monotonically increasing and, thus, h will never be 0.

Figure 4: Graphical representation of the IB curve in the information plane.

Dashed lines in orange represent tight bounds confining the region (in light orange) of possible IB curves (delimited by the red line, also known as the Pareto frontier).

Black dotted lines are informative values.

In blue we show an example of a possible IB curve confining a region (in darker orange) of an IB curve which does not achieve the Pareto frontier.

Finally, the yellow star represents the point where the representation keeps the same information about the input and the output.

Finally, we can tighten the bound using the results from Wu et al. (2019) , where, in Theorem 2, they showed the slope of the Pareto frontier could be bounded in the origin by f IB ≤ (inf Ωx⊂X {β 0 (Ω x )}) −1 .

Finally, we know that in deterministic classification tasks inf Ωx⊂X {β 0 (Ω x )} = 1, which aligns with Kolchinsky et al. (2019) and what we can observe from Figure 4 .

Therefore,

We can use the same ideas we used for the convex IB Lagrangian to formulate new families of Lagrangians that allow the exploration of the IB curve.

For that we will use the duality of the IB curve (Lemma 10 of (Gilad-Bachrach et al., 2003) ).

That is: Definition 9 (IB dual functional).

Let X and Y be statistically dependent variables.

Let also ∆ be the set of random variables T obeying the Markov condition Y ↔ X ↔ T .

Then the IB dual functional is

Theorem 3 (IB curve duality).

Let the IB curve be defined by the solutions of F IB,max (r) for varying

and ∀i∃r s.t. (F IB,min (i), i) = (r, F IB,max (r)).

From this definition it follows that minimizing the dual IB Lagrangian, L βdual IB,dual (T ) = I(X; T ) − β dual I(T ; Y ), for β dual = β −1 is equivalent to maximizing the IB Lagrangian.

In fact, the original Lagrangian for solving the problem was defined this way (Tishby et al., 2000) .

We decided to use the maximization version because the domain of useful β is bounded while it is not for β dual .

Following the same reasoning as we did in the proof of Theorem 2, we can ensure the IB curve can be explored if:

Here, h is a monotonically increasing strictly convex function, g is a monotonically increasing strictly concave function, and β g , β g,dual , β h,dual are the Lagrange multipliers of the families of Lagrangians defined above.

In a similar manner, one could obtain relationships between the Lagrange multipliers of the IB Lagrangian and the convex IB Lagrangian with these Lagrangian families.

Also, one could find a range of values for these Lagrangians to allow for the IB curve exploration and define a bijective mapping between their Lagrange multipliers and the IB curve.

However, (i) as mentioned in Section 2.2, I(T ; Y ) is particularly interesting to maximize without transformations because of its meaning.

Moreover, (ii) like β dual , the domain of useful β g and β h,dual is not upper bounded.

These two reasons make these other Lagrangians less preferable.

We only include them here for completeness.

Nonetheless, we encourage the curiours reader to explore these families of Lagrangians too.

In order to generate the empirical support results from Section 5 we used the nonlinear IB on the MNIST dataset (LeCun et al., 1998) .

This dataset contains 60,000 training samples and 10,000 testing samples of hand-written digits.

The samples are 28x28 pixels and are labeled from 0 to 9; i.e., X = R 784 and Y = {0, 1, ..., 9}.

As in (Kolchinsky et al., 2019) we trained the neural network with the Adam optimization algorithm (Kingma & Ba, 2014 ) with a learning rate of 10 −4 but we introduced a 0.6 decay rate every 10 iterations.

After talking with the authors of the nonlinear IB , we decided to estimate the gradients of both I θ (X; T ) and the cross entropy with the same mini-batch of 128 samples.

Moreover, we did not learn the covariance of the mixture of Gaussians used for the kernel density estimation of I θ (X; T ) and we set it to (exp(−1)) 2 .

We trained for 100 epochs 6 .

All the weights were initialized according to the method described by Glorot & Bengio (2010) using a Gaussian distribution.

The reader can find the PyTorch (Paszke et al., 2017) implementation at https://gofile.io/?c=G9Dl1L.

Then, we used the DBSCAN algorithm (Ester et al., 1996; Schubert et al., 2017) for clustering.

Particularly, we used the scikit-learn (Pedregosa et al., 2011) implementation with = 0.3 and min samples = 50.

In Figure 5 we show how the IB curve can be explored with different values of α for the power IB Lagrangian and in Figure 6 for different values of η and the exponential IB Lagrangian.

Finally, in Figure 7 we show the clusterization for the same values of α and η as in Figures 5 and 6 .

In this way the connection between the performance discontinuities and the clusterization is more evident.

Furthermore, we can also observe how the exponential IB Lagrangian maintains better the theoretical performance than the power IB Lagrangian (see Appendix G for an explanation of why).

When chossing the right h function, it is important to find the right balance between avoiding value convergence and aiming for strong convexity.

Practically, this balance is found by looking at how much faster h grows w.r.t.

the identity function.

In order to explain this issue we are going to use the example of classification on MNIST (LeCun et al., 1998) , where I(X; Y ) = H(Y ) = log 2 (10), and again the power and exponential IB Lagrangians.

If we use Proposition 2 on both Lagrangians we obtain the bijective mapping between their Lagrange multipliers and a certain level of compression in the classification setting:

1.

Power IB Lagrangian:

2.

Exponential IB Lagrangian: β exp = (η exp(ηI(X; T ))) −1 and I(X; T ) = − log(ηβ exp )/η.

Hence, we can simply plot the curves of I(X; T ) vs. β h for different hyperparameters α and η (see Figure 8 ).

In this way we can observe how increasing the growth of the function (e.g., increasing α or η in this case) too much provokes that many different values of β h converge to very similar values of I(X; T ).

This is an issue both for drawing the curve (for obvious reasons) and for aiming for a specific performance level.

Due to the nature of the estimation of the IB Lagrangian, the theoretical and practical value of β h that yield a specific I(X; T ) may vary slightly (see Figure 1 ).

Then if we select a function with too high growth, a small change in β h can result in a big change in the performance obtained.

Definition 10 (µ-Strong convexity).

If a function f (r) is twice continuous differentiable and its domain is confined in the real line, then it is µ-strong convex if f (r) ≥ µ ≥ 0 ∀r.

Experimentally, we observed when the growth of our function h(r) is small in the domain of interest r > 0 the convex IB Lagrangian does not perform well.

Later we realized that this was closely related with the strength of the convexity of our function.

In Theorem 2 we imposed the function h to be strictly convex to enforce having a unique β h for each value of I(X; T ).

Hence, since in practice we are not exactly computing the Lagrangian but an estimation of it (e.g., with the nonlinear IB ) we require strong convexity in order to be able to explore the IB curve.

We now look at the second derivative of the power and exponential function: h (r) = (1+α)αr α−1 and h (r) = η 2 exp(ηr) respectivelly.

Here we see how both functions are inherently 0-strong convex for r > 0 and α, η > 0.

However, values of α < 1 and η < 1 could lead to low µ-strong convexity in certain domains of r. Particularly, the case of α < 1 is dangerous because the function approaches 0-strong convexity as r increases, so the power IB Lagrangian performs poorly when low α are used to find high performances.

<|TLDR|>

@highlight

We introduce a general family of Lagrangians that allow exploring the IB curve in all scenarios. When these are used, and the IB curve is known, one can optimize directly for a performance/compression level directly.