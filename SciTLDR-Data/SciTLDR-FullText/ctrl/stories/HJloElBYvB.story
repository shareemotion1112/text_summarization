In the Information Bottleneck (IB), when tuning the relative strength between compression and prediction terms, how do the two terms behave, and what's their relationship with the dataset and the learned representation?

In this paper, we set out to answer these questions by studying multiple phase transitions in the IB objective: IB_β[p(z|x)] = I(X; Z) − βI(Y; Z) defined on the encoding distribution p(z|x) for input X, target Y and representation Z, where sudden jumps of dI(Y; Z)/dβ and prediction accuracy are observed with increasing β.

We introduce a definition for IB phase transitions as a qualitative change of the IB loss landscape, and show that the transitions correspond to the onset of learning new classes.

Using second-order calculus of variations, we derive a formula that provides a practical condition for IB phase transitions, and draw its connection with the Fisher information matrix for parameterized models.

We provide two perspectives to understand the formula, revealing that each IB phase transition is finding a component of maximum (nonlinear) correlation between X and Y orthogonal to the learned representation, in close analogy with canonical-correlation analysis (CCA) in linear settings.

Based on the theory, we present an algorithm for discovering phase transition points.

Finally, we verify that our theory and algorithm accurately predict phase transitions in categorical datasets, predict the onset of learning new classes and class difficulty in MNIST, and predict prominent phase transitions in CIFAR10.

The Information Bottleneck (IB) objective (Tishby et al., 2000) :

explicitly trades off model compression (I(X; Z), I(·; ·) denoting mutual information) with predictive performance (I(Y ; Z)) using the Lagrange multiplier β, where X, Y are observed random variables, and Z is a learned representation of X. The IB method has proved effective in a variety of scenarios, including improving the robustness against adversarial attacks (Alemi et al., 2016; Fischer, 2018) , learning invariant and disentangled representations (Achille & Soatto, 2018a; b) , underlying information-based geometric clustering (Strouse & Schwab, 2017b) , improving the training and performance in adversarial learning (Peng et al., 2018) , and facilitating skill discovery (Sharma et al., 2019) and learning goal-conditioned policy (Goyal et al., 2019) in reinforcement learning.

From Eq.

(1) we see that when β → 0 it will encourage I(X; Z) = 0 which leads to a trivial representation Z that is independent of X, while when β → +∞, it reduces to a maximum likelihood objective 1 that does not constrain the information flow.

Between these two extremes, how will the IB objective behave?

Will prediction and compression performance change smoothly, or do there exist interesting transitions in between?

In Wu et al. (2019) , the authors observe and study the learnability transition, i.e. the β value such that the IB objective transitions from a trivial global minimum to learning a nontrivial representation.

They also show how this first phase transition relates to the structure of the dataset.

However, to answer the full question, we need to consider the full range of β.

Motivation.

To get a sense of how I(Y ; Z) and I(X; Z) vary with β, we train Variational Information Bottleneck (VIB) models (Alemi et al., 2016) on the CIFAR10 dataset (Krizhevsky & Hinton, 2009) , where each experiment is at a different β and random initialization of the model.

Fig. 1 shows the I(X; Z), I(Y ; Z) and accuracy vs. β, as well as I(Y ; Z) vs. I(X; Z) for CIFAR10 with 20% label noise (see Appendix I for details).

are discontinuous and the accuracy has discrete jumps.

The observation lets us refine our question: When do the phase transitions occur, and how do they depend on the structure of the dataset?

These questions are important, since answering them will help us gain a better understanding of the IB objective and its close interplay with the dataset and the learned representation.

Moreover, the IB objective belongs to a general form of two-term trade-offs in many machine learning objectives: L = Prediction-loss + β · Complexity, where the complexity term generally takes the form of regularization.

Usually, learning is set at a specific β.

Many more insights can be gained if we understand the behavior of the prediction loss and model complexity with varying β, and how they depend on the dataset.

The techniques developed to address the question in the IB setting may also help us understand the two-term tradeoff in other learning objectives.

Contributions.

In this work, we begin to address the above question in IB settings.

Specifically:

• We identify a qualitative change of the IB loss landscape w.r.t.

p(z|x) for varying β as IB phase transitions (Section 3).

• Based on the definition, we introduce a quantity G[p(z|x)] and use it to prove a theorem giving a practical condition for IB phase transitions.

We further reveal the connection between G[p(z|x)] and the Fisher information matrix when p(z|x) is parameterized by θ (Section 3).

• We reveal the close interplay between the IB objective, the dataset and the learned representation, by showing that in IB, each phase transition corresponds to learning a new nonlinear component of maximum correlation between X and Y , orthogonal to the previously-learned Z, and each with decreasing strength (Section 4).

To the best of our knowledge, our work provides the first theoretical formula to address IB phase transitions in the most general setting.

In addition, we present an algorithm for iteratively finding the IB phase transition points (Section 5).

We show that our theory and algorithm give tight matches with the observed phase transitions in categorical datasets, predict the onset of learning new classes and class difficulty in MNIST, and predict prominent transitions in CIFAR10 experiments (Section 6).

The Information Bottleneck Method (Tishby et al., 2000) provides a tabular method based on the Blahut-Arimoto (BA) Algorithm (Blahut, 1972) to numerically solve the IB functional for the optimal encoder distribution P (Z|X), given the trade-off parameter β and the cardinality of the representation variable Z. This work has been extended in a variety of directions, including to the case where all three variables X, Y, Z are multivariate Gaussians (Chechik et al., 2005) , cases of variational bounds on the IB and related functionals for amortized learning (Alemi et al., 2016; Achille & Soatto, 2018a; Fischer, 2018) , and a more generalized interpretation of the constraint on model complexity as a Kolmogorov Structure Function (Achille et al., 2018) .

Previous theoretical analyses of IB include Rey & Roth (2012) , which looks at IB through the lens of copula functions, and Shamir et al. (2010) , which starts to tackle the question of how to bound generalization with IB.

We will make practical use of the original IB algorithm, as well as the amortized bounds of the Variational Informormation Bottleneck (Alemi et al., 2016) and the Conditional Entropy Bottleneck (Fischer, 2018) .

Phase transitions, where key quantities change discontinuously with varying relative strength in the two-term trade-off, have been observed in many different learning domains, for multiple learning objectives.

In Rezende & Viola (2018) , the authors observe phase transitions in the latent representation of β-VAE for varying β.

Strouse & Schwab (2017b) utilize the kink angle of the phase transitions in the Deterministic Information Bottleneck (DIB) (Strouse & Schwab, 2017a) to determine the optimal number of clusters for geometric clustering.

Tegmark & Wu (2019) explicitly considers critical points in binary classification tasks using a discrete information bottleneck with a non-convex Pareto-optimal frontier.

In Achille & Soatto (2018a) x , and Σ x is the covariance matrix.

This work provides valuable insights for IB, but is limited to the special case that X, Y and Z are jointly Gaussian.

Phase transitions in the general IB setting have also been observed, which Tishby (2018) describes as "information bifurcation".

In Wu et al. (2019) , the authors study the first phase transition, i.e. the learnability phase transition, and provide insights on how the learnability depends on the dataset.

Our work is the first work that addresses all the IB phase transitions in the most general setting, and provides theoretical insights on the interplay between the IB objective, its phase transitions, the dataset, and the learned representation.

3 FORMULA FOR IB PHASE TRANSITIONS 3.1 DEFINITIONS Let X ∈ X , Y ∈ Y, Z ∈ Z be random variables denoting the input, target and representation, respectively, having a joint probability distribution p(X, Y, Z), with X × Y × Z its support.

X, Y and Z satisfy the Markov chain Z − X − Y , i.e. Y and Z are conditionally independent given X. We assume that the integral (or summing if X, Y or Z are discrete random variables) is on X × Y × Z. We use x, y and z to denote the instances of the respective random variables.

The above settings are used throughout the paper.

We can view the IB objective IB β [p(z|x)] (Eq. 1) as a functional of the encoding distribution p(z|x).

To prepare for the introduction of IB phase transitions, we first define relative perturbation function and second variation, as follows.

Definition 1.

Relative perturbation function: For p(z|x), its relative perturbation function r(z|x) is a bounded function that maps X × Z to R and satisfies E z∼p(z|x) [r(z|x)] = 0.

Formally, define

We have that r(z|x) ∈ Q Z|X iff r(z|x) is a relative perturbation function of p(z|x).

The perturbed probability (density) is p (z|x) = p(z|x) (1 + · r(z|x)) for some > 0.

Definition 2.

Second variation: Let functional F [f (x)] be defined on some normed linear space R. Let us add a perturbative function · h(x) to f (x), and now the functional F [f (x) + · h(x)] can be expanded as

is a linear functional of ·h(x), and is called the first variation, denoted as δF

is a quadratic functional of · h(x), and is called the second variation, denoted as δ

We can think of the perturbation function · h(x) as an infinite-dimensional "vector" (x being the indices), with being its amplitude and h(x) its direction.

Here β + and 0 − denote one-sided limits.

We can understand the δ 2 IB β [p(z|x)] as a local "curvature" of the IB objective IB β (Eq. 1) w.r.t.

p(z|x), along some relative perturbation r(z|x).

A phase transition occurs when the convexity of IB β [p(z|x)] w.r.t.

p(z|x) changes from a minimum to a saddle point in the neighborhood of its optimal solution p * β (z|x) as β increases from β c to β c + 0

+ .

This means that there exists a perturbation to go downhill and find a better minimum.

We validate this definition empirically below.

The definition for IB phase transition (Definition 3) indicates the important role δ 2 IB β [p(z|x)] plays on the optimal solution in providing the condition for phase transitions.

To concretize it and prepare for a more practical condition for IB phase transitions, we expand IB β [p(z|x)(1 + · r(z|x))] to the second order of , giving:

The proof is given in Appendix B, in which we also give Eq. (20) for empirical estimation.

Note that Lemma 0.1 is very general and can be applied to any p(z|x), not only at the optimal solution p * β (z|x).

The Fisher Information matrix.

In practice, the encoder p θ (z|x) is usually parameterized by some parameter vector θ = (θ 1 , θ 2 , ...θ k ) T ∈ Θ, e.g. weights and biases in a neural net, where Θ is the parameter field.

An infinitesimal change of θ ← θ + ∆θ induces a relative perturbation · r(z|x) ∆θ

, from which we can compute the threshold function

where

are the conditional Fisher information matrix (Zegers, 2015) of θ for Z conditioned on X and Y , respectively.

λ max is the largest eigenvalue of C −1 I Z|Y (θ) − I Z (θ) (C T ) −1 with v max the corresponding eigenvector, where CC T is the Cholesky decomposition of the matrix I Z|X (θ) − I Z (θ), and v max is the eigenvector for λ max .

The infimum is attained at ∆θ = (

The proof is in appendix C. We see that for parameterized encoders p θ (z|x), each term of G[p(z|x)] in Eq. (2) can be replaced by a bilinear form with the Fisher information matrix of the respective variables.

Although this lemma is not required to understand the more general setting of Lemma 0.1, where the model is described in a functional space, Lemma 0.2 helps understand G[p(z|x)] for parameterized models, which permits directly linking the phase transitions to the model's parameters.

Phase Transitions.

Now we introduce Theorem 1 that gives a concrete and practical condition for IB phase transitions, which is the core result of the paper: Theorem 1.

The IB phase transition points {β c i } as defined in Definition 3 are given by the roots of the following equation:

where

We can understand Eq. (4) as the condition when δ 2 IB β [p(z|x)] is about to be able to be negative at the optimal solution p * β (z|x) for a given β.

The proof for Theorem 1 is given in Appendix D. In Section 4, we will analyze Theorem 1 in detail.

In this section we set out to understand G[p(z|x)] as given by Eq. (2) and the phase transition condition as given by Theorem 1, from the perspectives of Jensen's inequality and representational maximum correlation.

The condition for IB phase transitions given by Theorem 1 involves

which is in itself an optimization problem.

We can understand

(2) using Jensen's inequality:

The equality between A and B holds when the perturbation r(z|x) is constant w.r.t.

x for any z; the equality between B and C holds when E x∼p(x|y,z) [r(z|x)] is constant w.r.t.

y for any z. Therefore, the minimization of A−C B−C encourages the relative perturbation function r(z|x) to be as constant w.r.t.

x as possible (minimizing intra-class difference), but as different w.r.t.

different y as possible (maximizing inter-class difference), resulting in a clustering of the values of r(z|x) for different examples x according to their class y. Because of this clustering property in classification problems, we conjecture that there are at most |Y| − 1 phase transitions, where |Y| is the number of classes, with each phase transition differentiating one or more classes.

Under certain conditions we can further simplify G[p(z|x)] and gain a deeper understanding of it.

Firstly, inspired by maximum correlation (Anantharam et al., 2013) , we introduce two new concepts, representational maximum correlation and conditional maximum correlation, as follows.

Definition 4.

Given a joint distribution p(X, Y ), and a representation Z satisfying the Markov chain Z − X − Y , the representational maximum correlation ρ r (X, Y ; Z) is defined as

where

The conditional maximum correlation ρ m (X, Y |Z) is defined as:

where

We prove the following Theorem 2, which expresses G[p(z|x)] in terms of representational maximum correlation and related quantities, with proof given in Appendix F.

Z|X and Q Z|X satisfy:

, then we have:

(i) The representation maximum correlation and G:

(ii) The representational maximum correlation and conditional maximum correlation:

where z * = arg max z∈Z ρ m (X, Y |Z = z), and h * (x) is the optimal solution for the learn- (iv) For discrete X, Y and Z, we have

where σ 2 (Z) is the second largest singular value of the matrix Q X,Y |Z :=

Theorem 2 furthers our understanding of G[p(z|x)] and the phase transition condition (Theorem 1), which we elaborate as follows.

Discovering maximum correlation in the orthogonal space of a learned representation: Intuitively, the representational maximum correlation measures the maximum linear correlation between f (X, Z) and g(Y, Z) among all real-valued functions f, g, under the constraint that f (X, Z) is "orthogonal" to p(X|Z) and

is the inverse square of this representational maximum correlation.

Theorem 2 (ii) further shows that G[p(z|x)] is finding a specific z * on which maximum (nonlinear) correlation between X and Y 2 For discrete X, Z such that the cardinality |Z| ≥ |X |, this is generally true since in this scenario, h(x, z) and s(z) have |X ||Z| + |Z| unknown variables, but the condition has only |X ||Z| + |X | linear equations.

The difference between Q Z|X and Q conditioned on Z can be found.

Combined with Theorem 1, we have that when we continuously increase β, for the optimal representation Z * β given by p * β (z|x) at β, ρ r (X, Y ; Z * β ) shall monotonically decrease due to that X and Y has to find their maximum correlation on the orthogonal space of an increasingly better representation Z * β that captures more information about X. A phase transition occurs when ρ r (X, Y ; Z * β ) reduces to

, after which as β continues to increase, ρ r (X, Y ; Z * β ) will try to find maximum correlation between X and Y orthogonal to the full previously learned representation.

This is reminiscent of canonical-correlation analysis (CCA) (Hotelling, 1992) in linear settings, where components with decreasing linear maximum correlation that are orthogonal to previous components are found one by one.

In comparison, we show that in IB, each phase transition corresponds to learning a new nonlinear component of maximum correlation between X and Y in Z, orthogonal to the previously-learned Z. In the case of classification where different classes may have different difficulty (e.g. due to label noise or support overlap), we should expect that classes that are less difficult as measured by a larger maximum correlation between X and Y are learned earlier.

Conspicuous subset conditioned on a single z: Furthermore, we show in (iii) that an optimal relative perturbation function r(z|x) can be decomposed into a product of two factors, a

factor that only focus on perturbing a specific point z * in the representation space, and an h * (x) factor that is finding the "conspicuous subset" (Wu et al., 2019) , i.e. the most confident, large, typical, and imbalanced subset in the X space for the distribution

Singular values In categorical settings, (iv) reveals a connection between G[p(z|x)] and the singular value of the Q X,Y |Z matrix.

Due to the property of SVD, we know that the square of the singular values of Q X,Y |Z equals the non-negative eigenvalue of the matrix Q T X,Y |Z Q X,Y |Z .

Then the phase transition condition in Theorem 1 is equivalent to a (nonlinear) eigenvalue problem.

This is resonant with previous analogy with CCA in linear settings, and is also reminiscent of the linear eigenvalue problem in Gaussian IB (Chechik et al., 2005) .

As a consequence of the theoretical analysis above, we are able to derive an algorithm to efficiently estimate the phase transitions for a given model architecture and dataset.

This algorithm also permits us to empirically confirm some of our theoretical results in Section 6.

Typically, classification involves high-dimensional inputs X. Without sweeping the full range of β where at each β it is a full learning problem, it is in general a difficult task to estimate the phase transitions.

In Algorithm 1, we present a two-stage approach.

In the first stage, we train a single maximum likelihood neural network f θ with the same encoder architecture as in the (variational) IB to estimate p(y|x), and obtain an N × C matrix p(y|x), where N is the number of examples in the dataset and C is the number of classes.

In the second stage, we perform an iterative algorithm w.r.t.

G and β, alternatively, to converge to a phase transition point.

Specifically, for a given β, we use a Blahut-Arimoto type IB algorithm (Tishby et al., 2000) to efficiently reach IB optimal p * β (z|x) at β, then use SVD (with the formula given in Theorem 2 (iv)) to efficiently estimate G[p * β (z|x)] at β (step 8).

We then use the G[p * β (z|x)] value as the new β and do it again (step 7 in the next iteration).

At convergence, we will reach the phase transition point given by G[p * β (z|x)] = β (Theorem 1).

After convergence as measured by patience parameter K, we slightly increase β by δ (step 13), so that the algorithm can discover the subsequent phase transitions.

We quantitatively and qualitatively test the ability of our theory and Algorithm 1 to provide good predictions for IB phase transitions.

We first verify them in fully categorical settings, where X, Y, Z are all discrete, and we show that the phase transitions can correspond to learning new classes as we increase β.

We then test our algorithm on versions of the MNIST and CIFAR10 datasets with added label noise.

8: 6.1 CATEGORICAL DATASET For categorical datasets, X and Y are discrete, and p(X) and p(Y |X) are given.

To test Theorem 1, we use the Blahut-Arimoto IB algorithm to compute the optimal p * β (z|x) for each β.

I(Y ; Z * ) vs. β is plotted in Fig. 2 (a) .

There are two phase transitions at β Moreover, starting at β = 1, Alg.

1 converges to each phase transition points within few iterations.

Our other experiments with random categorical datasets show similarly tight matches.

Furthermore, in Appendix G we show that the phase transitions correspond to the onset of separation of p(z|x) for subsets of X that correspond to different classes.

This supports our conjecture from Section 4.1 that there are at most |Y| − 1 phase transitions in classification problems.

For continuous X, how does our algorithm perform, and will it reveal aspects of the dataset?

We first test our algorithm in a 4-class MNIST with noisy labels 3 , whose confusion matrix and experimental settings are given in Appendix H. Fig. 3 (a) shows the path Alg.

1 takes.

We see again that in each Figure 3: (a) Path of Alg.

1 starting with β = 1, where the maximum likelihood model f θ is using the same encoder architecture as in the CEB model.

This stairstep path shows that Alg.

1 is able to ignore very large regions of β, while quickly and precisely finding the phase transition points.

Also plotted is an accumulation of G[p * β (z|x)] vs. β by running Alg.

1 with varying starting β (blue dots).

(b) Per-class accuracy vs. β, where the accuracy at each β is from training an independent CEB model on the dataset.

The per-class accuracy denotes the fraction of correctly predicted labels by the CEB model for the observed labelỹ.

phase Alg.

1 converges to the phase transition points within a few iterations, and it discovers in total 3 phase transition points.

Similar to the categorical case, we expect that each phase transition corresponds to the onset of learning a new class, and that the last class is much harder to learn due to a larger separation of β.

Therefore, this class should have a much larger label noise so that it is hard to capture this component of maximum correlation between X and Y , as analyzed in representational maximum correlation (Section 4.2).

Fig. 3 (b) plots the per-class accuracy with increasing β for running the Conditional Entropy Bottleneck (Fischer, 2018 ) (another variational bound on IB).

We see that the first two predicted phase transition points β c 0 , β c 1 closely match the observed onset of learning class 3 and class 0.

Class 1 is observed to learn earlier than expected, possibly due to the gap between the variational IB objective and the true IB objective in continuous settings.

By looking at the confusion matrix for the label noise (Fig. 7) , we see that the ordering of onset of learning: class 2, 3, 0, 1, corresponds exactly to the decreasing diagonal element p(ỹ = 1|y = 1) (increasing noise) of the classes, and as predicted, class 1 has a much smaller diagonal element p(ỹ = 1|y = 1) than the other three classes, which makes it much more difficult to learn.

This ordering of classes by difficulty is what our representational maximum correlation predicts.

The most interesting region is right before β = 2, where accuracy decreases with β.

Alg.

1 identifies both sides of that region, as well as points at or near all of the early obvious phase transitions.

It also seems to miss later transitions, possibly due to the gap between the variational IB objective and the true IB objective in continuous settings.

Finally, we investigate the CIFAR10 experiment from Section 1.

The details of the experimental setup are described in Appendix I. This experiment stretches the current limits of our discrete approximation to the underlying continuous representation being learned by the models.

Nevertheless, we can see in Fig. 4 that many of the visible empirical phase transitions are tightly identified by Alg.

1.

Particularly, the onset of learning is predicted quite accurately; the large interval between the predicted β 3 = 1.21 and β 4 = 1.61 corresponds well to the continuous increase of I(X; Z) and I(Y ; Z) at the same interval.

And Alg.

1 is able to identify many dense transitions not obviously seen by just looking at I(Y ; Z) vs. β curve alone.

Alg.

1 predicts 9 phase transitions, exactly equal to |Y| − 1 for CIFAR10.

In this work, we observe and study the phase transitions in IB as we vary β.

We introduce the definition for IB phase transitions, and based on it derive a formula that gives a practical condition for IB phase transitions.

We further understand the formula via Jensen's inequality and representational maximum correlation.

We reveal the close interplay between the IB objective, the dataset and the learned representation, as each phase transition is learning a nonlinear maximum correlation component in the orthogonal space of the learned representation.

We present an algorithm for finding the phase transitions, and show that it gives tight matches with observed phase transitions in categorical datasets, predicts onset of learning new classes and class difficulty in MNIST, and predicts prominent transitions in CIFAR10 experiments.

This work is a first theoretical step towards a deeper understanding of the phenomenon of phase transitions in the Information Bottleneck.

We believe our approach will be applicable to other "trade-off" objectives, like β-VAE (Higgins et al., 2017) and InfoDropout (Achille & Soatto, 2018a) , where the model's ability to predict is balanced against a measure of complexity.

Here we prove the Lemma 2.1, which will be crucial in the lemmas and theorems in this paper that follows.

Lemma 2.1.

For a relative perturbation function r(z|x) ∈ Q Z|X for a p(z|x), where r(z|x) satisfies E z∼p(z|x) [r(z|x)] = 0, we have that the IB objective can be expanded as Proof.

Suppose that we perform a relative perturbation r(z|x) on p(z|x) such that the perturbed conditional probability is p (z|x) = p(z|x) (1 + · r(z|x)), then we have

Therefore, we can denote the corresponding relative perturbation r(z) on p(z) as

Similarly, we have

And we can denote the corresponding relative perturbation r(z|y) on p(z|y) as

We have

The 0 th -order term is simply IB β [p(z|x)].

The first order term is

The n th -order term for n ≥ 2 is

In the last equality we have used

Combining the terms with all orders, we have

As a side note, the KL-divergence between p (z|x) = p(z|x)(1 + · r(z|x)) and p(z|x) is

Therefore, to the second order, we have

Similarly, we have

up to the second order.

Using similar procedure, we have up to the second-order,

B PROOF OF LEMMA 0.1

Proof.

From Lemma 2.1, we have

The condition of

is equivalent to

Using Jensen's inequality and the convexity of the square function, we have

The equality holds iff r(z|y) = E x∼p(x|y,z) [r(z|x)] is constant w.r.t.

y, for any z.

, where the equality holds iff r(z|x) is constant w.r.t.

x for any z.

where r(z|y) = E x∼p(x|y,z) [r(z|x)] and r(z) = E x∼p(x|z) [r(z|x)].

which is always true due to that E[r 2 (z|x)]

≥ E[r 2 (z)], and will be a looser condition than Eq. (17) above.

Above all, we have Eq. (17).

To empirically estimate G[p(z|x)] from a minibatch of {(x i , y i )}, i = 1, 2, ...N and the encoder p(z|x), we can make the following Monte Carlo importance sampling estimation, where we use the samples {x j } ∼ p(x) and also get samples of {z i } ∼ p(z) = p(x)p(z|x), and have:

Here Ω x (y i ) denotes the set of x examples that has label of y i , and 1[·] is an indicator function that takes value 1 if its argument is true, 0 otherwise.

for any x j .

Combining all terms, we have that the empiricalĜ[p(z|x)] is given bŷ

where {z i } ∼ p(z) and {x i } ∼ p(x).

It is also possible to use different distributions for importance sampling, which will results in different formulas for empirical estimation of

Proof.

For the parameterized 4 p θ (z|x) with θ ∈ Θ, after θ ← θ + ∆θ, where 5 ∆θ ∈ Θ is an infinitesimal perturbation on θ, we have that the distribution changes from p θ (z|x) to p θ+∆θ (z|x), 4 In this paper, θ = (θ1, θ2, ...θ k )

T and

, ...

is a k × k matrix with (i, j) element of

∂θ i ∂θ j .

5 Note that since Θ is a field, it is closed under subtraction, we have ∆θ ∈ Θ.

∂θ 2 1 = 0, and similarly E y,z∼p θ (y,z) [

In other words, the ∆θ 2 terms in the first-order variation δIB β [p θ (z|x)] vanish, and the remaining ∆θ 2 are all

Similarly, we have

Combining the continuity of T β (β ) at β = β, and Eq. (24) and (25) Proof.

When we r(z|x) is shifted by a global transformation r (z|x) ← r(z|x) + s(z), we have

, and similarly r (z|y) ← r(z|y) + s(z).

The numerator of G[r(z|x); p(z|x)] is then

Proof.

Using the condition of the theorem, we have that ∀r(z|x) ∈ Q 0 Z|X , there exists r 1 (z|x) ∈ Q Z|X and s(z) ∈ {s : Z → R|s bounded} s.t.

r(z|x) = r 1 (z|x) + s(z).

Note that the only difference between Q Z|X and Q (0) Z|X is that Q Z|X requires E p(z|x) [r 1 (z|x)] = 0.

Using Lemma 2.2, we have

where r(z|x) doesn't have the constraint of E p(z|x) [·] = 0.

After dropping the constraint of E z∼p(z|x) [r(z|x)] = 0, again using Lemma 2.2, we can let r(z) = E x∼p(x|z) [r(z|x)] = 0 (since we can perform the transformation r (z|x) ← r(z|x) − r(z), so that the new r (z) ≡ 0).

Now we get a simpler formula for G[p(z|x)], as follows:

where Q

(1)

From Eq. (26), we can further require that

We have

where F (y, z) := dxp(x|y, z)f (x, z).

We have used Cauchy-Schwarz inequality, where the equality holds when g(y, z) = αF (y, z) for some α.

Comparing Eq. (30) and the supremum:

we see that the only difference is that in the latter

Therefore,

where in the last equality we have let c(z) have "mass" only on the place where ρ 2 m (X, Y |Z = z) attains supremum w.r.t.

z.

Z|X , satisfying the requirement for ρ s (X, Y ; Z) (which equals ρ r (X, Y ; Z) by Eq. 28).

In this section we study the behavior of p(z|x) on the phase transitions.

We use the same categorical dataset (where |X| = |Y | = |Z| = 3 and p(x) is uniform, and p(y|x) is given in Fig. 5) .

In Fig. 6 we show the p(z|x) on the simplex before and after each phase transition.

We see that the first phase transition corresponds to the separation of x = 2 (belonging to y = 2) w.r.t.

x ∈ {0, 1} (belonging to classes y ∈ {0, 1}), on the p(z|x) simplex.

The second phase transition corresponds to the separation of x = 0 with x = 1.

Therefore, each phase transition corresponds to the ability to distinguish subset of examples, and learning of new classes.

We use the MNIST training examples with class 0, 1, 2, 3, with a hidden label-noise matrix as given in Fig. 7 , based on which at each minibatch we dynamically sample the observed label.

We use conditional entropy bottleneck (CEB) (Fischer, 2018) as the variational IB objective, and run multiple independent instances with different the target β.

We jump start learning by started training at β = 100 for 100 epochs, annealing β from 100 down to the target β over 600 epochs, and continue to train at the target epoch for another 800 epochs.

The encoder is a three-layer neural net, where each hidden layer has 512 neurons and leakyReLU activation, and the last layer has linear activation.

The classifier p(y|z) is a 2-layer neural net with a 128-neuron ReLU hidden layer.

The backward encoder p(z|y) is also a 2-layer neural net with a 128-neuron ReLU hidden layer.

We trained with Adam (Kingma & Welling, 2013) at learning rate of 10 −3 , and anneal down with factor 1/(1 + 0.01 · epoch).

For Alg.

1, for the f θ we use the same architecture as the encoder of CEB, and use |Z| = 50 in Alg.

1.

We use the same CIFAR10 class confusion matrix provided in Wu et al. (2019) to generate noisy labels with about 20% label noise on average (reproduced in Table 1 ).

We trained 28 × 1 Wide ResNet (He et al., 2016; Zagoruyko & Komodakis, 2016) models using the open source implementation from Cubuk et al. (2018) as encoders for the Variational Information Bottleneck (VIB) (Alemi et al., 2016) .

The 10 dimensional output of the encoder parameterized a mean-field Gaussian with unit covariance.

Samples from the encoder were passed to the classifier, a 2 layer MLP.

The marginal distributions were mixtures of 500 fully covariate 10-dimensional Gaussians, all parameters of which are trained.

With this standard model, we trained 251 different models at β from 1.0 to 6.0 with step size of 0.02.

As in Wu et al. (2019) , we jump-start learning by annealing β from 100 down to the target β.

We do this over the first 4000 steps of training.

The models continued to train for another 56,000 gradient steps after that, a total of 600 epochs.

We trained with Adam (Kingma & Ba, 2015) at a base learning rate of 10 −3 , and reduced the learning rate by a factor of 0.5 at 300, 400, and 500 epochs.

The models converged to essentially their final accuracy within 40,000 gradient steps, and then remained stable.

Figure 5: p(y|x) for the categorical dataset in Fig. 2 and Fig. 6 .

The value in i th row and j th column denotes p(y = j|x = i).

p(x) is uniform.

The accuracies reported in Figure 4 are averaged across five passes over the training set.

We use |Z| = 50 in Alg.

1.

<|TLDR|>

@highlight

We give a theoretical analysis of the Information Bottleneck objective to understand and predict observed phase transitions in the prediction vs. compression tradeoff.