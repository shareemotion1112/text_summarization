Background: Statistical mechanics results (Dauphin et al. (2014); Choromanska et al. (2015)) suggest that local minima with high error are exponentially rare in high dimensions.

However, to prove low error guarantees for Multilayer Neural Networks (MNNs), previous works so far required either a heavily modified MNN model or training method, strong assumptions on the labels (e.g., “near” linear separability), or an unrealistically wide hidden layer with \Omega\(N) units.



Results: We examine a MNN with one hidden layer of piecewise linear units, a single output, and a quadratic loss.

We prove that, with high probability in the limit of N\rightarrow\infty datapoints, the volume of differentiable regions of the empiric loss containing sub-optimal differentiable local minima is exponentially vanishing in comparison with the same volume of global minima, given standard normal input of dimension d_0=\tilde{\Omega}(\sqrt{N}), and a more realistic number of d_1=\tilde{\Omega}(N/d_0) hidden units.

We demonstrate our results numerically: for example, 0% binary classification training error on CIFAR with only N/d_0 = 16 hidden neurons.

Motivation.

Multilayer Neural Networks (MNNs), trained with simple variants of stochastic gradient descent (SGD), have achieved state-of-the-art performances in many areas of machine learning .

However, theoretical explanations seem to lag far behind this empirical success (though many hardness results exist, e.g., BID44 BID42 ).

For example, as a common rule-of-the-thumb, a MNN should have at least as many parameters as training samples.

However, it is unclear why such over-parameterized MNNs often exhibit remarkably small generalization error (i.e., difference between "training error" and "test error"), even without explicit regularization BID54 .Moreover, it has long been a mystery why MNNs often achieve low training error BID10 .

SGD is only guaranteed to converge to critical points in which the gradient of the expected loss is zero BID3 , and, specifically, to local minima BID35 ) (this is true also for regular gradient descent BID29 ).

Since loss functions parameterized by MNN weights are non-convex, it is unclear why does SGD often work well -rather than converging to sub-optimal local minima with high training error, which are known to exist BID16 BID50 .

Understanding this behavior is especially relevant in important cases where SGD does get stuck BID20 ) -where training error may be a bottleneck in further improving performance.

Ideally, we would like to quantify the probability to converge to a local minimum as a function of the error at this minimum, where the probability is taken with the respect to the randomness of the initialization of the weights, the data and SGD.

Specifically, we would like to know, under which conditions this probability is very small if the error is high, as was observed empirically (e.g., BID10 BID17 ).

However, this seems to be a daunting task for realistic MNNs, since it requires a characterization of the sizes and distributions of the basins of attraction for all local minima.

Previous works BID10 BID7 , based on statistical physics analogies, suggested a simpler property of MNNs: that with high probability, local minima with high error diminish exponentially with the number of parameters.

Though proving such a geometric property with realistic assumptions would not guarantee convergence to global minima, it appears to be a necessary first step in this direction (see discussion on section 6).

It was therefore pointed out as an open problem at the Conference of Learning Theory (COLT) 2015.

However, one has to be careful and use realistic MMN architectures, or this problem becomes "too easy".For example, one can easily achieve zero training error (Nilsson, 1965; BID2 -if the MNN's last hidden layer has more neurons than training samples.

Such extremely wide MNNs are easy to optimize (Yu, 1992; BID23 BID31 BID19 BID43 Nguyen & Hein, 2017) .

In this case, the hidden layer becomes linearly separable in classification tasks, with high probability over the random initialization of the weights.

Thus, by training the last layer we get to a global minimum (zero training error).

However, such extremely wide layers are not very useful, since they result in a huge number of weights, and serious overfitting issues.

Also, training only the last layer seems to take little advantage of the inherently non-linear nature of MNNs.

Therefore, in this paper we are interested to understand the properties of local and global minima, but at a more practical number of parameters -and when at least two weight layers are trained.

For example, Alexnet BID27 is trained using about 1.2 million ImageNet examples, and has about 60 million parameters -16 million of these in the two last weight layers.

Suppose we now train the last two weight layers in such an over-parameterized MNN.

When do the sub-optimal local minima become exponentially rare in comparison to the global minima?Main contributions.

We focus on MNNs with a single hidden layer and piecewise linear units, optimized using the Mean Square Error (MSE) in a supervised binary classification task (Section 2).

We define N as the number of training samples, d l as the width of the l-th activation layer, and g (x)<h (x) as an asymptotic inequality in the leading order (formally: lim x→∞ log g(x)log h(x) < 1).

We examine Differentiable Local Minima (DLMs) of the MSE: sub-optimal DLMs where at least a fraction of > 0 of the training samples are classified incorrectly, and global minima where all samples are classified correctly.

Our main result, Theorem 10, states that, with high probability, the total volume of the differentiable regions of the MSE containing sub-optimal DLMs is exponentially vanishing in comparison to the same volume of global minima, given that: Assumption 1.

The datapoints (MNN inputs) are sampled from a standard normal distribution.

4 N neurons.

This improves over previously known results (Yu, 1992; BID23 BID31 BID43 Nguyen & Hein, 2017 ) -which require an extremely wide hidden layer with d 1 ≥ N neurons (and thus N d 0 parameters) to remove sub-optimal local minima with high probability.

In section 5 we validate our results numerically.

We show that indeed the training error becomes low when the number of parameters is close to N .

For example, with binary classification on CIFAR and ImageNet, with only 16 and 105 hidden neurons (about N/d 0 ), respectively, we obtain less then 0.1% training error.

Additionally, we find that convergence to non-differentiable critical points does not appear to be very common.

Lastly, in section 6 we discuss our results might be extended, such as how to apply them to "mildly" non-differentiable critical points.

Plausibility of assumptions.

Assumption 1 is common in this type of analysis (Andoni et al., 2014; BID7 BID53 BID51 BID4 .

At first it may appear rather unrealistic, especially since the inputs are correlated in typical datasets.

However, this no-correlation part of the assumption may seem more justified if we recall that datasets are many times whitened before being used as inputs.

Alternatively, if, as in our motivating question, we consider the input to the our simple MNN to be the output of the previous layers of a deep MNN with fixed random weights, this also tends to de-correlate inputs (Poole et al., 2016, Figure 3) .

The remaining part of assumption 1, that the distribution is normal, is indeed strong, but might be relaxed in the future, e.g. using central limit theorem type arguments.

In assumption 2 we use this asymptotic limit to simplify our proofs and final results.

Multiplicative constants and finite (yet large) N results can be found by inspection of the proofs.

We assume a constant error since typically the limit → 0 is avoided to prevent overfitting.

In assumption 3, for simplicity we have d 0≤ N , since in the case d 0 ≥ N the input is generically linearly separable, and sub-optimal local minima are not a problem BID18 BID39 .

Additionally, we have √ N<d 0 , which seems very reasonable, since for example, d 0 /N ≈ 0.016, 0.061 and 0.055 MNIST, CIFAR and ImageNet, respectively.

In assumption 4, for simplicity we have d 1< N , since, as mentioned earlier, if d 1 ≥ N the hidden layer is linearly separable with high probability, which removes sub-optimal local minima.

The other bound N log 4 N<d 0 d 1 is our main innovation -a large over-parameterization which is nevertheless asymptotically mild and improves previous results.

Previous work.

So far, general low (training or test) error guarantees for MNNs could not be found -unless the underlying model (MNN) or learning method (SGD or its variants) have been significantly modified.

For example, BID10 made an analogy with high-dimensional random Gaussian functions, local minima with high error are exponentially rare in high dimensions; BID7 BID25 replaced the units (activation functions) with independent random variables; BID36 replaces the weights and error residuals with independent random variables; (Baldi, 1989; BID40 BID20 BID32 BID57 used linear units; BID55 used unconventional units (e.g., polynomials) and very large hidden layers (d 1 = poly (d 0 ), typically N ); BID4 BID11 BID41 used a modified convnet model with less then d 0 parameters (therefore, not a universal approximator BID9 BID22 ); BID51 BID47 BID30 assume the weights are initialized very close to those of the teacher generating the labels; and BID24 BID56 ) use a non-standard tensor method during training.

Such approaches fall short of explaining the widespread success of standard MNN models and training practices.

Other works placed strong assumptions on the target functions.

For example, to prove convergence of the training error near the global minimum, BID18 assumed linearly separable datasets, while BID39 assumed strong clustering of the targets ("near" linear-separability).

Also, (Andoni et al., 2014) showed a p-degree polynomial is learnable by a MNN, if the hidden layer is very large ( DISPLAYFORM0 , typically N ) so learning the last weight layer is sufficient.

However, these are not the typical regimes in which MNNs are required or used.

In contrast, we make no assumption on the target function.

Other closely related results BID48 BID53 ) also used unrealistic assumptions, are discussed in section 6, in regards to the details of our main results.

Therefore, in contrast to previous works, the assumptions in this paper are applicable in some situations (e.g., Gaussian input) where a MNN trained using SGD might be used and be useful (e.g., have a lower test error then a linear classier).

Model.

We examine a Multilayer Neural Network (MNN) with a single hidden layer and a scalar output.

The MNN is trained on a finite training set of N datapoints (features) X x (1) , . . .

, x (N ) ∈ R d0×N with their target labels y y (1) , . . .

, y DISPLAYFORM0 and z ∈ R d1 as the first and second weight layers (bias terms are ignored for simplicity), respectively, and f (·) as the common leaky rectifier linear unit (LReLU BID33 ) DISPLAYFORM1 for some ρ = 1 (so the MNN is non-linear) , where both functions f and a operate component-wise (e.g., for any matrix M: (f (M)) ij = f (M ij )).

Thus, the output of the MNN on the entire dataset can be written as DISPLAYFORM2 We use the mean square error (MSE) loss for optimization DISPLAYFORM3 where · is the standard euclidean norm.

Also, we measure the empiric performance as the fraction of samples that are classified correctly using a decision threshold at y = 0.5, and denote this as the mean classification error, or MCE 2 .

Note that the variables e, MSE, MCE and other related variables (e.g., their derivatives) all depend on W, z, X, y and ρ, but we keep this dependency implicit, to avoid cumbersome notation.

Additional Notation.

We define g (x)<h (x) if and only if lim x→∞ DISPLAYFORM4 log h(x) < 1 (and similarly≤ and=).

We denote "M ∼ N " when M is a matrix with entries drawn independently from a standard normal distribution (i.e., ∀i, j: M ij ∼ N (0, 1)).

The Khatari-rao product (cf.

BID0 DISPLAYFORM5 where a ⊗ x = a 1 x , . . .

, a d1 x is the Kronecker product.

MNNs are typically trained by minimizing the loss over the training set, using Stochastic Gradient Descent (SGD), or one of its variants (e.g., Adam (Kingma & Ba, 2014) ).

Under rather mild conditions BID35 BID3 , SGD asymptotically converges to local minima of the loss.

For simplicity, we focus on differentiable local minima (DLMs) of the MSE (eq. (2.3)).

In section 4 we will show that sub-optimal DLMs are exponentially rare in comparison to global minima.

Non-differentiable critical points, in which some neural input (pre-activation) is exactly zero, are shown to be numerically rare in section 5, and are left for future work, as discussed in section 6.Before we can provide our results, in this section we formalize a few necessary notions.

For example, one has to define how to measure the amount of DLMs in the over-parameterized regime: there is an infinite number of such points, but they typically occupy only a measure zero volume in the weight space.

Fortunately, using the differentiable regions of the MSE (definition 1), the DLMs can partitioned to a finite number of equivalence groups, so all DLMs in each region have the same error (Lemma 2).

Therefore, we use the volume of these regions (definition 3) as the relevant measure in our theorems.

DISPLAYFORM0 2 Formally (this expression is not needed later): DISPLAYFORM1 is an open set, since a (0) is undefined (from eq. 2.1).

Clearly, for all W ∈ D A (X) the MSE is differentiable, so any local minimum can be non-differentiable only if it is not in any differentiable region.

Also, all DLMs in a differentiable region are equivalent, as we prove on appendix section 7: Lemma 2.

At all DLMs in D A (X) the residual error e is identical, and furthermore DISPLAYFORM2 The proof is directly derived from the first order necessary condition of DLMs (∇MSE = 0) and their stability.

Note that Lemma 2 constrains the residual error e in the over-parameterized regime: DISPLAYFORM3 In this case eq. (3.2) implies e = 0, if rank (A • X) = N .

Therefore, we must have rank (A • X) < N for sub-optimal DLMs to exist.

Later, we use similar rank-based constraints to bound the volume of differentiable regions which contain DLMs with high error.

Next, we define this volume formally.

Angular Volume.

From its definition (eq. (3.1)) each region D A (X) has an infinite volume in R d1×d0 : if we multiply a row of W by a positive scalar, we remain in the same region.

Only by rotating the rows of W can we move between regions.

We measure this "angular volume" of a region in a probabilistic way: we randomly sample the rows of W from an isotropic distribution, e.g., standard Gaussian: W ∼ N , and measure the probability to fall in D A (X), arriving to the following Definition 3.

For any region R ⊂ R d1×d0 .

The angular volume of R is DISPLAYFORM4 4 MAIN RESULTS Some of the DLMs are global minima, in which e = 0 and so, MCE = MSE = 0, while other DLMs are sub-optimal local minima in which MCE > > 0.

We would like to compare the angular volume (definition 3) corresponding to both types of DLMs.

Thus, we make the following definitions.

Definition 4.

We define 3 L ⊂ R d1×d0 as the union of differentiable regions containing sub-optimal DLMs with MCE > , and G ⊂ R d1×d0 as the union of differentiable regions containing global minima with MCE = 0.

In this section, we use assumptions 1-4 (stated in section 1) to bound the angular volume of the region L encapsulating all sub-optimal DLMs, the region G, encapsulating all global minima, and the ratio between the two.

Angular volume of sub-optimal DLMs.

First, in appendix section 8 we prove the following upper bound in expectation Theorem 6.

Given assumptions 1-4, the expected angular volume of sub-optimal DLMs, with MCE > > 0, is exponentially vanishing in N as DISPLAYFORM5 and, using Markov inequality, its immediate probabilistic corollary Corollary 7.

Given assumptions 1-4, for any δ > 0 (possibly a vanishing function of N ), we have, with probability 1 − δ, that the angular volume of sub-optimal DLMs, with MCE > > 0, is exponentially vanishing in N as DISPLAYFORM6 Proof idea of Theorem 6: we first show that in differentiable regions with MCE > > 0, the condition in Lemma 2, (A • X) e = 0, implies that A = a (WX) must have a low rank.

Then, we show that, when X ∼ N and W ∼ N , the matrix A = a (WX) has a low rank with exponentially low probability.

Combining both facts, we obtain the bound.

Existence of global minima.

Next, to compare the volume of sub-optimal DLMs with that of global minima, in appendix section 9 we show first that, generically, global minima do exist (using a variant of the proof of BID2 , Theorem 1)): Theorem 8.

For any y ∈ {0, 1} N and X ∈ R d0×N almost everywhere 4 we find matrices DISPLAYFORM7 4 N/ (2d 0 − 2) and ∀i, n : DISPLAYFORM8 has a DLM which achieves zero error e = 0.Recently BID54 , Theorem 1) similarly proved that a 2-layer MNN with approximately 2N parameters can achieve zero error.

However, that proof required N neurons (similarly to (Nilsson, 1965; BID2 Yu, 1992; BID23 BID31 BID43 ), while Theorem 8 here requires much less: (Hardt & Ma, 2017, Theorem 3 .2) showed a deep residual network with N log N parameters can achieve zero error.

In contrast, here we require just one hidden layer with 2N parameters.

DISPLAYFORM9 Note the construction in Theorem 8 here achieves zero training error by overfitting to the data realization, so it is not expected to be a "good" solution in terms of generalization.

To get good generalization, one needs to add additional assumptions on the data (X and y).

Such a possible (common yet insufficient for MNNs) assumption is that the problem is "realizable", i.e., there exist a small "solution MNN", which achieves low error.

For example, in the zero error case:Assumption 5. (Optional) The labels are generated by some teacher y = f (W * X) z * with weight matrices DISPLAYFORM10

Angular volume of global minima.

We prove in appendix section 10:Theorem 9.

Given assumptions 1-3, we set δ= DISPLAYFORM0 or if assumption 5 holds, we set d * 1 as in this assumption.

Then, with probability 1 − δ, the angular volume of global minima is lower bounded as, DISPLAYFORM1 Proof idea: First, we lower bound V (G) with the angular volume of a single differentiable region of one global minimum (W * , z * ) -either from Theorem 8, or from assumption 5.

Then we show that this angular volume is lower bounded when W ∼ N , given a certain angular margin between the datapoints in X and the rows of W * .

We then calculate the probability of obtaining this margin when X ∼ N .

Combining both results, we obtain the final bound.

Main result: angular volume ratio.

Finally, combining Theorems 6 and 9 it is straightforward to prove our main result in this paper, as we do in appendix section 11:Theorem 10.

Given assumptions 1-3, we set δ .

= DISPLAYFORM2 Then, with probability 1 − δ, the angular volume of sub-optimal DLMs, with MCE > > 0, is exponentially vanishing in N, in comparison to the angular volume of global minima with DISPLAYFORM3 ≤ exp (−γ N log N ) .

Theorem 10 implies that, with "asymptotically mild" over-parameterization (i.e. in which #parameters =Ω (N )), differentiable regions in weight space containing sub-optimal DLMs (with high MCE) are FIG1 .1: Gaussian data: final training error (mean±std, 30 repetitions) in the overparameterized regime is low (right of the dashed black line).

We trained MNNs with one and two hiddens layer (with widths equal to d = d 0 ) on a synthetic random dataset in which ∀n = 1, . . .

, N , x (n) was drawn from a normal distribution N (0, 1), and y (n) = ±1 with probability 0.5.

Table 1 : Binary classification of MNIST, CIFAR and ImageNet: 1-hidden layer achieves very low training error (MCE) with a few hidden neurons, so that #parameters DISPLAYFORM0 DISPLAYFORM1 ImageNet we downsampled the images to allow input whitening.exponentially small in comparison with the same regions for global minima.

Since these results are asymptotic in N → ∞, in this section we examine it numerically for a finite number of samples and parameters.

We perform experiments on random data, MNIST, CIFAR10 and ImageNet-ILSVRC2012.

In each experiment, we used ReLU activations (ρ = 0), a binary classification target (we divided the original classes to two groups), MSE loss for optimization (eq. (2.3)), and MCE to determine classification error.

Additional implementation details are given in appendix part III.First, on the small synthetic Gaussian random data (matching our assumptions) we perform a scan on various networks and dataset sizes.

With either one or two hidden layers ( FIG1 .1) , the error goes to zero when the number of non-redundant parameters (approximately d 0 d 1 ) is greater than the number of samples, as suggested by our asymptotic results.

Second, on the non-syntehtic datasets, MNIST, CIFAR and ImageNet (In ImageNet we downsampled the images to size 64 × 64, to allow input whitening) we only perform a simulation with a single 1-hidden layer MNN for which #parameters ≈ N , and again find (Table 1) that the final error is zero (for MNIST and CIFAR) or very low (ImageNet).Lastly, in FIG1 .2 we find that, on the Gaussian dataset, the inputs to the hidden neurons converge to a distinctly non-zero value.

This indicates we converged to differentiable critical points -since nondifferentiable critical points must have zero neural inputs.

Note that occasionally, during optimization, we could find some neural inputs with very low values near numerical precision level, so convergence to non-differentiable minima may be possible.

However, as explained in the next section, as long as the number of neural inputs equal to zero are not too large, our bounds also hold for these minima.

In this paper we examine Differentiable Local Minima (DLMs) of the empiric loss of Multilayer Neural Networks (MNNs) with one hidden layer, scalar output, and LReLU nonlinearities (section 2).

We prove (Theorem 10) that with high probability the angular volume (definition 3) of sub-optimal DLMs is exponentially vanishing in comparison to the angular volume of global minima (definition 4), under assumptions 1-4.

This results from an upper bound on sub-optimal DLMs (Theorem 6) and a lower bound on global minima (Theorem 9).

2 /5 for 1000 epochs, then decreased the learning rate exponentially for another 1000 epochs.

This was repeated 30 times.

For all d and repeats, we see that (left) the final absolute value of the minimal neural input (i.e., min i,n w i x (n) ) in the range of 10 −3 − 10 0 , which is much larger then (right) the final MSE error for all d and all repeats -in the range 10 −31 − 10 −7 .Convergence of SGD to DLMs.

These results suggest a mechanism through which low training error is obtained in such MNNs.

However, they do not guarantee it.

One issue is that sub-optimal DLMs may have exponentially large basins of attraction.

We see two possible paths that might address this issue in future work, using additional assumptions on y. One approach is to show that, with high probability, no sub optimal DLM falls within the vanishingly small differentiable regions we bounded in Theorem 6.

Another approach would be to bound the size of these basins of attraction, by showing that sufficiently large of number of differentiable regions near the DLM are also vanishingly small (other methods might also help here BID15 ).

Another issue is that SGD might get stuck near differentiable saddle points, if their Hessian does not have strictly negative eigenvalues (i.e., the strict saddle property ).

It should be straightforward to show that such points also have exponentially vanishing angular volume, similar to sub-optimal DLMs.

Lastly, SGD might also converge to non-differentiable critical points, which we discuss next.

Non-differentiable critical points.

The proof of Theorem 6 stems from a first order necessary condition (Lemma 2): (A • X) e = 0, which is true for any DLM.

However, non-differentiable critical points, in which some neural inputs are exactly zero, may also exist (though, numerically, they don't seem very common -see FIG1 .2).

In this case, to derive a similar bound, we can replace the condition with P (A • X) e = 0, where P is a projection matrix to the subspace orthogonal to the non-differentiable directions.

As long as there are not too many zero neural inputs, we should be able to obtain similar results.

For example, if only a constant ratio r of the neural inputs are zero, we can simply choose P to remove all rows of (A • X) corresponding to those neurons, and proceed with exactly the same proof as before, with d 1 replaced with (1 − r) d 1 .

It remains a theoretical challenge to find reasonable assumptions under which the number of non-differentiable directions (i.e., zero neural inputs) does not become too large.

Related results.

Two works have also derived related results using the (A • X) e = 0 condition from Lemma 2.

In BID48 , it was noticed that an infinitesimal perturbation of A makes the matrix A • X full rank with probability 1 (Allman et al., 2009, Lemma 13 ) -which entails that e = 0 at all DLMs.

Though a simple and intuitive approach, such an infinitesimal perturbation is problematic: from continuity, it cannot change the original MSE at sub-optimal DLMs -unless the weights go to infinity, or the DLM becomes non-differentiable -which are both undesirable results.

An extension of this analysis was also done to constrain e using the singular values of A•X BID53 , deriving bounds that are easier to combine with generalization bounds.

Though a promising approach, the size of the sub-optimal regions (where the error is high) does not vanish exponentially in the derived bounds.

More importantly, these bounds require assumptions on the activation kernel spectrum γ m , which do not appear to hold in practice (e.g., BID53 , Theorems 1,3) require mγ m 1 to hold with high probability, while mγ m < 10 −2 in BID53 , FIG9 ).Modifications and extensions.

There are many relatively simple extensions of these results: the Gaussian assumption could be relaxed to other near-isotropic distributions (e.g., sparse-land model, (Elad, 2010, Section 9 .2)) and other convex loss functions are possible instead of the quadratic loss.

More challenging directions are extending our results to MNNs with multi-output and multiple hidden layers, or combining our training error results with novel generalization bounds which might be better suited for MNNs (e.g., BID14 BID46 BID12 ) than previous approaches BID54 .

The appendix is divided into three parts.

In part I we prove all the main theorems mentioned in the paper.

Some of these rely on other technical results, which we prove later in part II.

Lastly, in part III we give additional numerical details and results.

First, however, we define additional notation (some already defined in the main paper) and mention some known results, which we will use in our proofs.

• The indicator function I (A) 1 , if A 0 , else , for any event A.• Kronecker's delta δ ij I (i = j).• The Matrix I d as the identity matrix in R d×d , and I d×k is the relevant R d×k upper left sub-matrix of the identity matrix.• DISPLAYFORM0 • The vector m n as the n'th column of a matrix M, unless defined otherwise (then m n will be a row of M).• M > 0 implies that ∀i, j : M ij > 0.• M S is the matrix composed of the columns of M that are in the index set S.• A property holds "M-almost everywhere" (a.e.

for short), if the set of entries of M for which the property does not hold has zero measure (Lebesgue).

DISPLAYFORM1 • If x ∼ N (µ, Σ) the x is random Gaussian vector.• φ (x) DISPLAYFORM2 exp − 1 2 x 2 as the univariate Gaussian probability density function.• Φ (x)x −∞ φ (u) du as the Gaussian cumulative distribution function.• B (x, y) as the beta function.

Lastly, we recall the well known Markov Inequality:

Fact 11. (Markov Inequality) For any random variable X ≥ 0, we have ∀η > 0 DISPLAYFORM3 Part IProofs of the main results DISPLAYFORM4 , where diag (v) is the diagonal matrix with v in its diagonal, and vec (M) is vector obtained by stacking the columns of the matrix M on top of one another.

Then, we can re-write the MSE (eq. (2.3)) as DISPLAYFORM5 where G w is the output of the MNN.

Now, if (W, z) is a DLM of the MSE in eq. (2.3), then there is no infinitesimal perturbation of (W, z) which reduces this MSE.Next, for each row i, we will show that ∂MSE/∂w i = 0, since otherwise we can find an infinitesimal perturbation of (W, z) which decreases the MSE, contradicting the assumption that (W, z) is a local minimum.

For each row i, we divide into two cases:First, we consider the case z i = 0.

In this case, any infinitesimal perturbation q i inw i can be produced by an infinitesimal perturbation in w i :w i + q i = (w i + q i /z i )z i .

Therefore, unless the gradient ∂MSE/∂w i is equal to zero, we can choose an infinitesimal perturbation q i in the opposite direction to this gradient, which will decrease the MSE.Second, we consider the case z i = 0.

In this case, the MSE is not affected by changes made exclusively to w i .

Therefore, all w i derivatives of the MSE are equal to zero (∂ k MSE/∂ k w i , to any order k) .

Also, since we are at a differentiable local minimum, ∂MSE/∂z i = 0.

Thus, using a Taylor expansion, if we perturb (w i , z i ) by (ŵ i ,ẑ i ) then the MSE is perturbed bŷ DISPLAYFORM6 Therefore, unless ∂ 2 MSE/ (∂w i ∂z i ) = 0 we can chooseŵ i and a sufficiently smallẑ i such that the MSE is decreased.

Lastly, using the chain rule DISPLAYFORM7 Thus, ∂MSE/∂w i = 0.

This implies thatw is also a DLM 5 of eq. (7.2), which entails DISPLAYFORM8 Since G = A • X and e = y − G w this proves eq. (7.1).

Now, for any two solutionsw 1 andw 2 of eq. (7.3), we have DISPLAYFORM9 Multiplying by (w 2 −w 1 ) from the left we obtain DISPLAYFORM10 Therefore, the MNN output and the residual error e are equal for all DLMs in D A (X).8 SUB-OPTIMAL DIFFERENTIABLE LOCAL MINIMA: PROOF OF THEOREM 6 AND ITS COROLLARY Theorem 13. (Theorem 6 restated) Given assumptions 1-4, the expected angular volume of suboptimal DLMs, with MCE > > 0, is exponentially vanishing in N as DISPLAYFORM11 , and γ 0.23 3/4 if ρ = 0.To prove this theorem we upper bound the angular volume of L (definition 4), i.e., differentiable regions in which there exist DLMs with MCE > > 0.

Our proof uses the first order necessary condition for DLMs from Lemma 2, (A • X) e = 0, to find which configurations of A allow for a high residual error e with MCE > > 0.

In these configurations A • X cannot have full rank, and therefore, as we show (Lemma 14 below), A = a (WX) must have a low rank.

However, A = a (WX) has a low rank with exponentially low probability when X ∼ N and W ∼ N (Lemmas 15 and 16 below).

Thus, we derive an upper bound on E X∼N V (L (X, y)).Before we begin, let us recall some notation: [L] {1, 2, . . .

, L},M > 0 implies that ∀i, j : M ij > 0, M S is the matrix composed of the columns of M that are in the index set S, v 0 as the L 0 "norm" that counts the number of non-zero values in v. First we consider the case ρ = 0.

Also, we denote DISPLAYFORM12 First we consider the case ρ = 0.From definition 3 of the angular volume DISPLAYFORM13 r=1 S:|S|=Kr DISPLAYFORM14 where 1.

If we are at DLM a in D A (X), then Lemma 2 implies (A • X) e = 0.

Also, if e (n) = 0 on some sample, we necessarily classify it correctly, and therefore MCE ≤ e 0 /N .

Since MCE > in L this implies that N < e 0 .

Thus, this inequality holds for v = e.2.

We apply assumption 1, that X ∼ N .

Thus, we can apply the following Lemma, proven in appendix section 12.1: DISPLAYFORM0 Then, simultaneously for every possible A and S such that DISPLAYFORM1 we have that, X-a.e., v ∈ R N such that v n = 0 ∀n ∈ S and (A • X) v = 0 .

We use the union bound over all possible ranks r ≥ 1: we ignore the r = 0 case since for ρ = 0 (see eq. (2.1)) there is zero probability that rank (a (WX S )) = 0 for some non-empty S. For each rank r ≥ 1, it is required that |S| > K r = max [N , rd 0 ], so |S| = K r is a relaxation of the original condition, and thus its probability is not lower.5.

We again use the union bound over all possible subsets S of size K r .Thus, from eq. (8.1), we have DISPLAYFORM0 Kr+rd0(log d1+log Kr)+r DISPLAYFORM1 Kr+rd0(log d1+log Kr)+r DISPLAYFORM2 1.

Since we take the expectation over X, the location of S does not affect the probability.

Therefore, we can set without loss of generality DISPLAYFORM3 from assumptions 3 and 4.

Thus, with k = K r ≥ d 0 , we apply the following Lemma, proven in appendix section 12.2: Lemma 15.

Let X ∈ R d0×k be a random matrix with independent and identically distributed columns, and W ∈ R d1×d0 an independent standard random Gaussian matrix.

Then, in the limit FORMULA67 ) to simplify the combintaorial expressions.

5.

First, note that r = 1 is the maximal term in the sum, so we can neglect the other, exponentially smaller, terms.

Second, from assumption 3 we have d 0≤ N , so DISPLAYFORM4

Third, from assumption 4 we have N log 4 N<d 0 d 1 , so the 2 N log N term is negligible.

Thus, DISPLAYFORM0 which proves the Theorem for the case ρ = 0.Next, we consider the case ρ = 0.

In this case, we need to change transition (4) in eq. (8.1), so the sum starts from r = 0, since now we can have rank (a (WX S )) = 0.

Following exactly the same logic (except the modification to the sum), we only need to modify transition (5)in eq. (8.2) -since now the maximal term in the sum is at r = 0.

This entails γ = 0.23 3/4 .Corollary 17. (Corollary 7 restated) Given assumptions 1-4, for any δ > 0 (possibly a vanishing function of N ), we have, with probability 1 − δ, that the angular volume of sub-optimal DLMs, with MCE > > 0, is exponentially vanishing in N as DISPLAYFORM1 Proof.

Since V (L (X, y)) ≥ 0 we can use Markov's Theorem (Fact 11) ∀η > 0: DISPLAYFORM2 , and using Theorem (6) we prove the corollary.

DISPLAYFORM3 where we note that replacing a regular inequality< with inequality in the leading order≤ only removes constraints, and therefore increases the probability.

Recall the LReLU non-linearity 4 N/ (2d 0 − 2) and ∀i, n : w i x (n) = 0.

Therefore, every MNN with d 1 ≥ d * 1 has a DLM which achieves zero error e = 0.

DISPLAYFORM0 We prove the existence of a solution (W * ,z * ), by explicitly constructing it.

This construction is a variant of BID2 , Theorem 1), except we use LReLU without bias and MSE -instead of threshold units with bias and MCE.

First, we note that for any 1 > 2 > 0, the following trapezoid function can be written as a scaled sum of four LReLU: DISPLAYFORM1 Next, we examine the set of data points which are classified to 1: DISPLAYFORM2

, each with no more than d 0 − 1 samples.

For almost any dataset we can find K hyperplanes passing through the origin, with normals DISPLAYFORM0 DISPLAYFORM1 Then we have DISPLAYFORM2 which gives the correct classification on all the data points.

Thus, from eq. (9.1), we can construct a MNN with d * 1 = 4K hidden neurons which achieves zero error.

This is straightforward to do if we have a bias in each neuron.

To construct this MNN even without bias, we first find a vectorŵ i such that DISPLAYFORM3 Note that this is possible since X S + i,w i has full rank X-a.e. (the matrix X S + i ∈ R d0×d0−1 has, X-a.e., one zero left eigenvector, which isw i , according to eq. (9.2)).

Additionally, we can set DISPLAYFORM4 since changing the scale of w i would not affect the validity of eq. (9.2).

Then, we denote DISPLAYFORM5 Note, from eqs. (9.2) and (9.4) that this choice satisfies DISPLAYFORM6 (9.6) Also, to ensure that ∀n / ∈ S + i the sign of w (j) i x (n) does not change for different j, for some β, γ < 1 we define DISPLAYFORM7 where with probability 1, min n / DISPLAYFORM8 i , wi , w DISPLAYFORM9 and combining all the above facts, we have DISPLAYFORM10 Under review as a conference paper at ICLR 2018Thus, for DISPLAYFORM11 we obtain a MNN that implements DISPLAYFORM12 and thus achieves zero error.

Clearly, from this construction, if w i is a row of W * , then ∀n ∈ S + i ,∀i : w i x (n) ≥ 2 , and with probability 1 ∀n / ∈ S + i ,∀i : w i x (n) > 0, so this construction does not touch any non-differentiable region of the MSE.

Theorem 19. (Theorem 9 restated).Given assumptions 1-3, we set δ= DISPLAYFORM0 or if assumption 5 holds, we set d * 1 as in this assumption.

Then, with probability 1 − δ, the angular volume of global minima is lower bounded as, DISPLAYFORM1 In this section we lower bound the angular volume of G (definition 4), i.e., differentiable regions in which there exist DLMs with MCE = 0.

We lower bound V (G) using the angular volume corresponding to the differentiable region containing a single global minimum.

From assumption 4, we have d 0 d 1> N , so we can apply Theorem 8 and say that the labels are generated using a (X, y) -dependent MNN: y = f (W * X) z * with target weights DISPLAYFORM2 If, in addition, assumption 5 holds then we can assume W * and z * are independent from (X, y).

In both cases, the following differentiable regioñ DISPLAYFORM3 DISPLAYFORM4 Also, we will make use of the following definition.

Definition 20.

Let X have an angular margin α from W * if all datapoints (columns in X) are at an angle of at least α from all the weight hyperplanes (rows of W * ) , i.e., X is in the set DISPLAYFORM5 Using the definitions in eqs. (10.3) and (10.1), we prove the Theorem using the following three Lemmas.

First, In appendix section 13.2 we prove Lemma 21.

For any α, if W * is independent from W then, in the limit N → ∞, ∀X ∈ M α (W * ) with log sin α>d DISPLAYFORM6 Second, in appendix section 13.3 we prove DISPLAYFORM7 Lastly, in appendix section 13.4 we prove Lemma 23.

Let X ∈ R d0×N be a standard random Gaussian matrix of datapoints.

Then we can find, with probability 1, (X, y)-dependent matrices W * and z * as in Theorem 8 (where d * 14 N/ (2d 0 − 2) ).

Moreover, in the limit N → ∞, where N/d 0≤ d 0≤ N , for any y, we can bound the probability of not having an angular margin (eq. (10.3)) with sin α = 1/ (d * DISPLAYFORM8 Recall that ∀X, y and their corresponding W * , we have G (X, y) ⊂G (X, W * ) (eq. FIG0 ).

Thus, combining Lemmas 21 with sin α = 1/ (d * 1 d 0 N ) together with either Lemma 22 or 23, we prove the first (left) inequality of Theorem 9: FIG1 , we obtain the second (right) inequality DISPLAYFORM9 DISPLAYFORM10

Theorem 24. (Theorem 10 restated) Given assumptions 1-3, we set δ .

= DISPLAYFORM0 To prove this theorem we first calculate the expectation of the angular volume ratio given the X-event that the bound in Theorem 9 holds (given assumptions 1-3), i.e., V (G (X, y))≥ exp (−2N log N ).

Denoting this event 6 as M, we find: DISPLAYFORM1 (11.1) where 1.

We apply Theorem 9.

Fact 25.

For any variable X ≥ 0 and event A (whereĀ is its complement) DISPLAYFORM0 3.

We apply Theorem 6.4.

We apply Theorem 9.

For simplicity, in the reminder of the proof we denote , y) ) .

DISPLAYFORM0 From Markov inequality (Fact 11), since R (X) ≥ 0, we have ∀η (N ) > 0: DISPLAYFORM1 On the other hand, from fact 25, we have DISPLAYFORM2 Combining Eqs. (11.2)-(11.3) we obtain DISPLAYFORM3 , and so DISPLAYFORM4 We choose DISPLAYFORM5 so that DISPLAYFORM6 Then, from Theorem 9 we have DISPLAYFORM7 so we obtain the first (left) inequality in the Theorem (10) DISPLAYFORM8 .

where X ∈ R d0×N and A ∈ R d1×N .

The Khatari-Rao product between the two matrices is defined as DISPLAYFORM0 Then, simultaneously for every possible A and S such that DISPLAYFORM1 we have that, X-a.e., v ∈ R N such that v n = 0 ∀n ∈ S and (A • X) v = 0 .Proof.

We examine specific A ∈ {ρ, 1} d1×N and S ⊂ [N ], and such that |S| ≤ d S d 0 , where we defined d S rank (A S ).

We assume that d S ≥ 1, since otherwise the proof is trivial.

Also, we assume by contradiction that ∃v ∈ R N such that v i = 0 ∀i ∈ S and (A • X) v = 0 .

Without loss of generality, assume that S = {1, 2, ..., |S|} and that a 1 , a 2 , ..., a d S are linearly independent.

Then DISPLAYFORM2 From the definition of S we must have v n = 0 for every 1 ≤ n ≤ |S|.

Since a 1 , a 2 , ..., a d S are linearly independent, the rows of DISPLAYFORM3 Therefore, it is possible to find a matrix R such that DISPLAYFORM4 , where 0 i×j is the all zeros matrix with i columns and j rows.

Consider now A S • X S , i.e., the matrix composed of the columns of A • X in S. Applying R = R ⊗ I d0 to A S • X S , turns (12.2) into d 0 d S equations in the variables v 1 , ..., v |S| , of the form DISPLAYFORM5 for every 1 ≤ k ≤ d S .

We prove by induction that for every 1 ≤ d ≤ d S , the first d 0 d equations are linearly independent, except for a set of matrices X of measure 0.

This will immediately imply |S| > d S d 0 , or else eq. 12.2 cannot be true for v = 0.

which will contradict our assumption, as required.

The induction can be viewed as carrying out Gaussian elimination of the system of equations described by (12.3), where in each elimination step we characterize the set of matrices X that for which that step is impossible, and show it has measure 0.

We now extend the Gaussian elimination to the next d 0 equations, and eliminate all the variables in C from them.

The result of the elimination can be written down as, DISPLAYFORM6 where Y is a square matrix of size d 0 whose coefficients depend only on {ã k,n } n∈C,d>k≥1 and on {x n } n∈C , and in particular do not depend on x d and {x n } n∈S \C .

DISPLAYFORM7 but a set of measure zero (linear subspace of with dimension less than d 0 ), we must have dim Span{x n } n∈S \C = d 0 .

From the independence of {x n } n∈S \C on x d it follows that dim Span{x n } n∈S \C = d 0 holds a.e.

with respect to the Lebesgue measure over x.

Whenever dim Span{x n } n∈S \C = d 0 we must have |S \ C| ≥ d 0 and therefore Thus, we have proven, that for some A ∈ {ρ, 1} d1×N and DISPLAYFORM8 DISPLAYFORM9 has zero measure.

The event discussed in the theorem is a union of these events: DISPLAYFORM10 and it also has zero measure, since it is a finite union of zero measure events.

For completeness we note the following corollary, which is not necessary for a our main results.

DISPLAYFORM11 e., if and only if, DISPLAYFORM12 Proof.

We define d S rank (A S ) and A • X. The necessity of the condition |S| ≤ d 0 d S holds for every X, as can be seen from the following counting argument.

Since the matrix A S has rank d S , there exists an invertible row transformation matrix R, such that RA S has only d S non-zero rows.

Consider now G S = A S • X S , i.e., the matrix composed of the columns of G in S. We have (12.6) where R = R ⊗ I d0 is also an invertible row transformation matrix, which applies R separately on the d 0 sub-matrices of G S that are constructed by taking one every d 0 rows.

Since G S has at most d 0 d S non-zero rows, the rank of DISPLAYFORM13 DISPLAYFORM14 have full column rank, and hence neither will G. To demonstrate sufficiency a.e., suppose G does not have full column rank.

Let S be the minimum set of columns of G which are linearly dependent.

Since the columns of G S are assumed linearly dependent there exists v ∈ R |S| such v 0 = |S| and G S v = 0.

Using Lemma 28 we complete the proof.

In this section we will prove Lemma 15 in subsection 12.3.3.

This proof relies on two rather basic results, which we first prove in subsections 12.2.1 and 12.2.2.

Fact 29.

A hyperplane w ∈ d 0 can separate a given set of points DISPLAYFORM0 into several different dichotomies, i.e., different results for sign w X .

The number of dichotomies is upper bounded as follows: DISPLAYFORM1 Proof.

See BID8 , Theorem 1) for a proof of the left inequality as equality (the Schläfli Theorem) in the case that the columns of X are in "general position" (which holds X-a.e, see definition in BID8 ) .

If X is not in general position then this result becomes an upper bound, since some dichotomies might not be possible.

Next, we prove the right inequality.

For N = 1 and N = 2 the inequality trivially holds.

For N ≥ 3, we have DISPLAYFORM2 where in (1) we used the bound DISPLAYFORM3 we used the sum of a geometric series.

Lemma 30.

Let H = h 1 , . . .

, h d1 ∈ {−1, 1} d1×k be a deterministic binary matrix, W = w 1 , . . .

, w d1 ∈ R d1×d0 be an independent standard random Gaussian matrix, and X ∈ R d0×k be a random matrix with independent and identically distributed columns.

DISPLAYFORM0 Proof.

By direct calculation DISPLAYFORM1 where 1.

We used the independence of the w i .

: ±h S > 0 as the sets in which h is always positive/negative, andŜ (h) as the maximal set between these two.

Note that w i has a standard normal distribution which is symmetric to sign flips, so ∀S : DISPLAYFORM0 3.

Note that Ŝ (h) ≥ k/2 .

Therefore, we define S * = argmax DISPLAYFORM1 4.

We used the independence of the w i .5.

The maximum is a single term in the following sum of non-negative terms.6.

Taking the expectation over X, since the columns of X are independent and identically distributed, the location of S does not affect the probability.

Therefore, we can set without loss of generality S = [ k/2 ].

Recall the function a (·) from eq. (2.1): DISPLAYFORM0 where ρ = 1.

Lemma 31. (Lemma 15 restated).

Let X ∈ R d0×k be a random matrix with independent and identically distributed columns, and W ∈ R d1×d0 an independent standard random Gaussian matrix.

Then, in the limit DISPLAYFORM1 Proof.

We denote A = a (WX) ∈ {ρ, 1} d1×k .

For any such A for which rank (A) = r, we have a collection of r rows that span the remaining rows.

There are d 1 r possible locations for these r spanning rows.

In these rows there exist a collection of r columns that span the remaining columns.

There are k r possible locations for these r spanning columns.

At the intersection of the spanning rows and columns, there exist a full rank sub-matrix D. We denoteÃ as the matrix A which rows and columns are permuted so that D is the lower right block (12.8) where D is an invertible r × r matrix, and we divided X and W to the corresponding block matrices DISPLAYFORM2 DISPLAYFORM3 with W 2 ∈ R r×d0 rows and X 2 ∈ R d0×r .Since rank Ã = r, the first d 1 − r rows are contained in the span of the last r rows.

Therefore, there exists a matrix Q such that QC = Z and QD = B. Since D is invertible, this implies that Q = BD −1 and therefore DISPLAYFORM4 9) i.e., B, C and D uniquely determine Z.Using the union bound over all possible permutations from A toÃ, and eq. (12.9), we have P (rank (A) = r) (12.10) DISPLAYFORM5 Using Lemma 30, we have (12.11) an upper bound which does not depend on H.

So all that remains is to compute the sum: DISPLAYFORM6 DISPLAYFORM7

(k−r) I ∃w : sign w X 1 = h counts the number of dichotomies that can be induced by the linear classifier w on X 1 .

Using eq. (12.7) we can bound this number by 2 (k − r) d0 .

Similarly, the other sum can be bounded by 2 (d 1 − r) r .Combining eqs. (12.10), (12.11) and (12.13) we obtain DISPLAYFORM0 Next, we take the log.

To upper bound DISPLAYFORM1 Thus, we obtain log P (rank (A) = r) ≤ rd 0 (log (d 1 − r) + log (k − r)) + r 2 + 2r log 2 (12.14) DISPLAYFORM2 Recalling that W 1 ∈ R (d1−r)×d0 while W ∈ R d1×d0 , we obtain from Jensen's inequality (12.15) Taking the limit min [k, d 0 , d 1 ]>r on eqs. (12.14) and (12.15) we obtain DISPLAYFORM3 DISPLAYFORM4

In this section we will prove Lemma 16 in subsection 12.3.3.

This proof relies on more elementary results, which we first prove in subsections 12.3.1 and 12.3.2.

Recall that φ (x) and Φ (x) are, respectively, the probability density function and cumulative distribution function for a scalar standard normal random variable.

Definition 32.

We define the following functions ∀x ≥ 0 (12.17) where the inverse function g −1 (x) : [0, ∞) → [0, ∞) is well defined since g (x) monotonically increase from 0 to ∞, for x ≥ 0.

DISPLAYFORM0 Lemma 33.

Let z ∼ N (0, Σ) be a random Gaussian vector in R K , with a covariance matrix Σ ij = 1 − θK −1 δ mn + θK −1 where K θ > 0.

Then, recalling ψ (θ) in eq. (12.17), we have log P (∀i : DISPLAYFORM1 Proof.

Note that we can write z = u+η, where u ∼ N 0, 1 − θK −1 I K , and η ∼ N 0, θK −1 .

Using this notation, we have (12.18) where in (1) we changed the variable of integration to ξ = θ/ (K − θ)η.

We denote, for a fixed θ, (12.20) and ξ 0 as its global maximum.

Since q is twice differentiable, we can use Laplace's method (e.g., BID5 ) to simplify eq. (12.18) DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 To find ξ 0 , we differentiate q (ξ) and equate to zero to obtain (12.22) which implies (recall eq. (12.16)) DISPLAYFORM5 DISPLAYFORM6 This is a monotonically increasing function from 0 to ∞ in the range ξ ≥ 0.

Its inverse function can also be defined in that range DISPLAYFORM7 .

This implies that this equation has only one solution, ξ 0 = g −1 (θ).

Since lim ξ→∞ q (ξ) = −∞, this ξ 0 is indeed the global maximum of q (ξ).

Substituting this solution into q (ξ), we get (recall eq. (12.17)) DISPLAYFORM8 Using eq. (12.18), (12.21) and (12.24) we obtain: DISPLAYFORM9 Since the columns are independent, we have DISPLAYFORM10 where in (1) we used the bound from Lemma 36.

Proof.

For some θ > 0, and subset S such that |S| = K < L, we have DISPLAYFORM11 where in the last equality we used the fact that the rows of C are independent and identically distributed.

We choose a specific subset DISPLAYFORM12 to minimize the second term and then upper bound it using Lemma 37 with θ = K ; additionally, we apply Corollary 34 on the first term with the components of the vector u being DISPLAYFORM13 which is a Gaussian random vector with mean zero and covariance Σ for which ∀i : Σ ii = 1 and ∀i = j : Σ ij ≤ = θK −1 .

Thus, we obtain DISPLAYFORM14 where we recall ψ (θ) is defined in eq. (12.17).Next, we wish to select good values for θ and K, which minimize this bound for large (M, N, L, K).

Thus, keeping only the first order terms in each exponent (assuming L K 1), we aim to minimize the function as much as possible (12.26) Note that the first term is decreasing in K, while the second term increases.

Therefore, for any θ the minimum of this function in K would be approximately achieved when both terms are equal, i.e., DISPLAYFORM15 DISPLAYFORM16 so we choose DISPLAYFORM17 To minimize this function in θ, we need to maximize the function ψ 3 (θ) θ 2 (which has a single maximum).

Doing this numerically gives us DISPLAYFORM18 DISPLAYFORM19 where in the last line we used N ≥ L,N ≥ M and min [N, M, L]>α>1.

Taking the log, and denoting α M L/N , we thus obtain DISPLAYFORM20 Therefore, in the limit that N → ∞ and α (N ) → ∞, with α (N )<N , we have To prove the results in the next appendix sections, we will rely on the following basic Lemma.

Lemma 39.

For any vector y and x ∼ N (0, I d0 ), we have DISPLAYFORM21 DISPLAYFORM22 where we recall that B (x, y) is the beta function.

Proof.

Since N (0, I d0 ) is spherically symmetric, we can set y = [1, 0 . . . , 0] , without loss of generality.

Therefore, DISPLAYFORM23 Suppose Z ∼ B (α, β), α ∈ (0, 1), and β > 1 .

DISPLAYFORM24 Therefore, for > 0, DISPLAYFORM25 , which proves eq. (13.1).Similarly, for α ∈ (0, 1) and β > 1 DISPLAYFORM26 Therefore, for > 0, DISPLAYFORM27 , which proves eq. (13.2).

Given three matrices: datapoints, DISPLAYFORM0 , and target weights DISPLAYFORM1 we recall the following definitions: DISPLAYFORM2 (13.4) Using these definitions, in this section we prove the following Lemma.

Lemma 40. (Lemma 21 restated).

For any α, if W * is independent from W then, in the limit N → ∞, ∀X ∈ M α (W * ) with log sin α>d DISPLAYFORM3 Proof.

To lower bound DISPLAYFORM4 , we define the event that all weight hyperplanes (with normals w i ) have an angle of at least α from the corresponding target hyperplanes (with normals w * i ).

DISPLAYFORM5 In order that sign w i x (n) = sign w * 1 x (n) , w i must be rotated in respect to w * i by an angle greater then the angular margin α, which is the minimal the angle between x (n) and the solution hyperplanes (with normals w * i ).

Therefore, we have that, given X ∈ M α (W * ), DISPLAYFORM6 = γ 1 / 1 + γ 2 2 1 ŵ 1 max n<d0 x (n) , (13.8)where in (1) we used ∀n < d 0 : x (n) ŵ 1 = 1 and x (n) w 1 = 0 , from the construction ofw 1 and w 1 (eqs. (9.2), (9.5), and (9.4)), and in (2) we used the fact thatŵ 1w1 = 0 from eq. (9.4) together with w 1 = ŵ 1 from eq. (9.5), and 2 = γ 1 from eq. (9.7).

ŵ 1 x (n) , (13.9) where we used the fact that ∀n ≥ d 0 : 2 ŵ 1 x (n) ≤ γβ w 1 x (n) , from eq. (9.7), and also that w 1w1 = 0 from eq. (9.4).We substitute eqs. (13.8) and (13.9) into P (X ∈ M α 1 (W * )): DISPLAYFORM7 ≥ P γ 1 / 1 + γ 2 2 1 ŵ 1 max n<d0 x (n) > sin α, ŵ 1 x (n) > sin α≥ P γκ ŵ 1 max n<d0 x (n) > sin α, , and the fact thatŵ 1 andw 1 are functions of x (n) for n < d 0 (from eqs. (9.4) and (9.2)), and as such, they are independent from x (n) for n ≥ d 0 , in (2) we use that fact that ŵ 1 and max n<d0 x (n) are functions of x (n) for n < d 0 , and as such, they are independent from x (n) for n ≥ d 0 .

Thus, DISPLAYFORM8 − P γκ η sin α ≤ ŵ 1 (13.11) DISPLAYFORM9 where in (1) we use the union bound on both probability terms.

All that remains is to calculate each remaining probability term in eq. (13.11).

First, we have 12) where in (1) we used eq. (9.7), in (2) we recall that in eq. (13.10) we rotated the axes so that w 1 ∝ [1, 0, 0 . . .

, 0] axesw 1 ∝ [0, 1, 0, 0 . . .

, 0], in (3) we used the independence of different x (n) , and in (4) we used the fact that the ratio of two independent Gaussian variables is distributed according to the symmetric Cauchy distribution, which has the cumulative distribution function P (X > x) = (13.16) where the last inequality stems from the fact that u 1ŵ1 =w 1ŵ1 = 0 (from eq. (9.4)), so the minimal possible value is attained when u 2ŵ1 = ŵ 1 .

The minimal nonzero singular value, σ 2 , can be bounded using the following result from (Rudelson & Vershynin, 2010, eq. (3. 2)) P min Lastly, combining eqs. (13.12), (13.13), (13.14) and (13.17) into eqs. (13.7) and (13.11), we get, for η 2 > d 0 , DISPLAYFORM10 DISPLAYFORM11 DISPLAYFORM12

@highlight

"Bad" local minima are vanishing in a multilayer neural net: a proof with more reasonable assumptions than before

@highlight

In networks with a single hidden layer, the volume of suboptimal local minima exponentially decreases in comparison to global minima.

@highlight

This paper aims to answer why standard SGD based algorithms on neural network converge to 'good' solutions.