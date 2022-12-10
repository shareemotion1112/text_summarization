We study the evolution of internal representations during deep neural network (DNN) training, aiming to demystify the compression aspect of the information bottleneck theory.

The theory suggests that DNN training comprises a rapid fitting phase followed by a slower compression phase, in which the mutual information I(X;T) between the input X and internal representations T decreases.

Several papers observe compression of estimated mutual information on different DNN models, but the true I(X;T) over these networks is provably either constant (discrete X) or infinite (continuous X).

This work explains the discrepancy between theory and experiments, and clarifies what was actually measured by these past works.

To this end, we introduce an auxiliary (noisy) DNN framework for which I(X;T) is a meaningful quantity that depends on the network's parameters.

This noisy framework is shown to be a good proxy for the original (deterministic) DNN both in terms of performance and the learned representations.

We then develop a rigorous estimator for I(X;T) in noisy DNNs and observe compression in various models.

By relating I(X;T) in the noisy DNN to an information-theoretic communication problem, we show that compression is driven by the progressive clustering of hidden representations of inputs from the same class.

Several methods to directly monitor clustering of hidden representations, both in noisy and deterministic DNNs, are used to show that meaningful clusters form in the T space.

Finally, we return to the estimator of I(X;T) employed in past works, and demonstrate that while it fails to capture the true (vacuous) mutual information, it does serve as a measure for clustering.

This clarifies the past observations of compression and isolates the geometric clustering of hidden representations as the true phenomenon of interest.

Recent work by BID10 uses the Information Bottleneck framework BID13 BID12 to study the dynamics of DNN learning.

The framework considers the mutual information pair I(X; T ), I(Y ; T ) between the input X or the label Y and the network's hidden layers T .

Plotting the evolution of these quantities during training, BID10 made two interesting observations: (1) while I(Y ; T ) remains mostly constant as the layer index increases, I(X; T ) decreases, suggesting that layers gradually shed irrelevant information about X; and (2) after an initial fitting phase, there is a long compression phase during which I(X; T ) slowly decreases.

It was suggested that this compression is responsible for the generalization performance of DNNs.

A follow-up paper contends that compression is not inherent to DNN training, claiming double-sided saturating nonlinearities yield compression while single-sided/non-saturating ones do not necessarily compress.

BID10 and present many plots of I(X; T ), I(Y ; T ) evolution across training epochs.

These plots, however, are inadvertently misleading: they show a dynamically changing I(X; T ) when the true mutual information is provably either infinite or a constant independent of the DNN's parameters (see BID1 for a discussion of further degeneracies related to to the Information Bottleneck framework).

Recall that the mutual information I(X; T ) is a functional of the joint distribution of (X, T ) ∼ P X,T = P X P T |X , and that, in standard DNNs, T is a deterministic function of X. Hence, if P X is continuous, then so is T , and thus I(X; T ) = ∞ (cf. (Polyanskiy & Wu, 2012 , Theorem 2.4)).

If P X is discrete (e.g., when the features are discrete or if X adheres to an empirical distribution over the dataset), then the mutual information is a finite constant that does not depend on the parameters of the DNN.

Specifically, for deterministic DNNs, the mapping from a discrete X to T is injective for strictly monotone nonlinearities such as tanh or sigmoid, except for a measure-zero set of weights.

In other words, deterministic DNNs can encode all information about a discrete X in arbitrarily fine variations of T , causing no loss of information and implying I(X; T ) = H(X), even if deeper layers have fewer neurons.

The compression observed in BID10 and therefore cannot be due to changes in mutual information.

This discrepancy between theory and experiments originates from a theoretically unjustified discretization of neuron values in their approximation of I(X; T ).

To clarify, the quantity computed and plotted in these works is I(X; Bin(T )), where Bin is a per-neuron discretization of each hidden activity of T into a user-selected number of bins.

This I X; Bin(T ) is highly sensitive to the selection of bin size (as illustrated in FIG0 ) and does not track I(X; T ) for any choice of bin size.1 Nonetheless, compression results based on I X; Bin(T ) are observed by BID10 and in many interesting cases.

To understand this curious phenomenon we first develop a rigorous framework for tracking the flow of information in DNNs.

In particular, to ensure I(X; T ) is meaningful for studying the learned representations, we need to make the map X → T a stochastic parameterized channel whose parameters are the DNN's weights and biases.

We identify several desirable criteria that such a stochastic DNN framework should fulfill for it to provide meaningful insights into commonly used practical systems.(1) The stochasticity should be intrinsic to the operation of the DNN, so that the characteristics of mutual information measures are related to the learned internal representations, and not to an arbitrary user-defined parameter.

(2) The stochasticity should relate the mutual information to the deterministic binned version I X; Bin(T ) , since this is the object whose compression was observed; this requires the injected noise to be isotropic over the domain of T analogously to the per-neuron binning operation.

And most importantly, (3) the network trained under this stochastic model should be closely related to those trained in practice.

We propose a stochastic DNN framework in which independent and identically distributed (i.i.d.)

Gaussian noise is added to the output of each of the DNN's neurons.

This makes the map from X to T stochastic, ensures the data processing inequality (DPI) is satisfied, and makes I(X; T ) reflect the true operating conditions of the DNN, following Point (1).

Since the noise is centered and isotropic, Point (2) holds.

As for Point (3), Section 2 experimentally shows the DNN's learned representations and performance are not meaningfully affected by the addition of noise, for variances β 2 not too large.

Furthermore, randomness during training has long been used to improve neural network performance, e.g., to escape poor local optima BID4 , improve generalization performance BID11 , encourage learning of disentangled representations BID0 , and ensure gradient flow with hard-saturating nonlinearities BID3 .

Under the stochastic model, I(X; T ) has no exact analytic expression and is impossible to approximate numerically.

In Section 3 we therefore propose a sampling technique that decomposes the estimation of I(X; T ) into several instances of a simpler differential entropy estimation problem: estimating h(S + Z) given n samples of the d-dimensional random vector S and knowing the distribution of Z ∼ N (0, DISPLAYFORM0 We analyze this problem theoretically and show that any differential entropy estimator over the noisy DNN requires at least exponentially many samples in the dimension d. Leveraging the explicit modeling of S + Z, we then propose a new estimator that converges 1 Another approach taken in considers I(X; T + Z) (instead of I X; Bin(T ) ), where Z is an independent Gaussian with a user-defined variance.

This approach has two issues: (i) the values as a function of may violate the data processing inequality, and (ii) they do not reflect the operation of the actual DNN, which was trained without noise.

We focus on I X; Bin(T ) because it was commonly used in BID10 and , and since both methods have a similar effect of blurring T .as O (log n) d/4 / √ n , which significantly outperforms the convergence rate of general-purpose differential entropy estimators when applied to the noisy DNN framework.

We find that I(X; T ) exhibits compression in many cases during training of small DNN classifiers.

To explain compression in an insightful yet rigorous manner, Section 4 relates I(X; T ) to the well-understood notion of data transmission over additive white Gaussian noise (AWGN) channels.

Namely, I(X; T ) is the aggregate information transmitted over the channel P T |X with input X drawn from a constellation defined by the data samples and the noisy DNN parameters.

As training progresses, the representations of inputs from the same class tend to cluster together and become increasingly indistinguishable at the channel's output, thereby decreasing I(X; T ).

Furthermore, these clusters tighten as one moves into deeper layers, providing evidence that the DNN's layered structure progressively improves the representation of X to increase its relevance for Y .Finally, we examine clustering in deterministic DNNs.

We identify methods for measuring clustering that are valid for both noisy and deterministic DNNs, and show that clusters of inputs in learned representations typically form in both cases.

We complete the circle back to I X; Bin(T ) by clarifying why this binned mutual information measures clustering.

This explains what previous works were actually observing: not compression of mutual information, but increased clustering by hidden representations.

The geometric clustering of hidden representations is thus the fundamental phenomenon of interest, and we aim to test its connection to generalization performance, theoretically and experimentally, in future work.

Figure 2: kth noisy neuron in layer with nonlinearity σ; W (k) and b (k) are the kth row/entry of the weight matrix and the bias, respectively.

DISPLAYFORM0 is a deterministic function of the previous layer and Z ∼ N 0, β 2 I d ; no noise is injected to the output, i.e., T L = f L (T L−1 ).

We set S f (T −1 ) and use ϕ for the probability density function (PDF) of Z .

The functions {f } ∈[L] can represent any type of layer (fully connected, convolutional, max-pooling, etc.).

FIG14 shows a neuron in the th layer of a noisy DNN.

Deterministic 50 ± 4.6 Noisy (β = 0.05) 50 ± 5.0 Noisy (β = 0.1) 51 ± 6.9 Noisy (β = 0.2) 86 ± 9.8 Noisy (β = 0.5) 2200 ± 520 Dropout (p = 0.2) 39 ± 3.9 Table 1 : Total MNIST validation errors for different models, showing mean ± standard deviation over eight initial random seeds.

To explore the relation between noisy and deterministic DNNs under conditions representative of current machine learning practices, we trained four-layer convolutional neural networks (CNNs) on MNIST BID6 .

The CNNs used different levels of internal noise, including no noise, and one used dropout in place of additive noise.

We measured their performance on the validation set and characterized the cosine similarities between their internal representations.

Full details of the CNN architecture and training procedure are in Supplement 9.3.

The results in Table 1 show small amounts of internal additive noise (β ≤ 0.1) have a minimal impact on classification performance, while dropout strongly improves it.

The histograms in FIG15 show that the noisy (for small β) and dropout models learn internal representations similar to the representations learned by the deterministic model.

In this high-dimensional space, unrelated representations would create cosine similarity histograms with zero mean and standard deviation between 0.02-0.3, so the observed values are quite large.

As expected, dissimilarity increases as the noise increases, and similarity is lower for the internal layers (2 and 3).Mutual Information: Noisy DNNs induce a stochastic map from X to the rest of the network, described by the conditional distribution P T1,...,T L |X .

The corresponding PDF Cosine similarity to noiseless model FIG15 : Histograms of cosine similarities between internal representations of deterministic, noisy, and dropout MNIST CNN models.

To encourage comparable internal representations, all models were initialized with the same random weights and accessed the training data in the same order.the input dataset, andP X be its empirical distribution, described by the probability mass function (PMF)p X (x) = 1 m i∈[m] 1 {xi=x} , for x ∈ X .

Since data sets typically contain no repetitions, we assumep X (x) = 1 m , ∀x ∈ X .

The input and the hidden layers are jointly distributed according DISPLAYFORM0 , we study the mutual information (Supplement 7 explains this factorization) DISPLAYFORM1 where log(·) is with respect to the natural base.

Although P T and P T |X are readily sampled from using the DNN's forward pass, these distributions are too complicated (due to the composition of Gaussian noises and nonlinearities) to analytically compute I(X; T ) or even to evaluate their densities at the sampled points.

Therefore, we must estimate I(X; T ) directly from the available samples.

Expanding I(X; T ) as in (1), our goal is to estimate h(p T ) and h(p T |X=x ), ∀x ∈ X : a problem that we show is hard in high dimensions.

Each differential entropy term is estimated and computed via a two-step process.

First, we develop the sample propagation (SP) estimator, which exploits the ability to propagate samples up the DNN layers and the known noise distribution.

This estimator approximates each true entropy by the differential entropy of a known Gaussian mixture (defined only through the available resources: the samples we obtain from the DNN and the noise parameter).

This estimate is shown to converge to the true entropy when the number of samples grows.

However, since the entropy of a Gaussian mixture has no closed-form expression, in the second (computational) step we use Monte Carlo (MC) integration to numerically evaluate it.

In what follows, we denote the empirical PMF associated with a set A = {a i } i∈[n] ⊂ R d byp A .

Unconditional Entropy: Since T = S + Z , where S and Z are independent, we have DISPLAYFORM0 be n i.i.d.

samples from P X .

Feed eachx j into the DNN and collect the outputs it produces at the ( − 1)-th layer.

The function f is then applied on each collected output to obtain S s ,1 , s ,2 , . . .

, s ,n , which is a set of n i.i.d.

samples from p S .

We estimate h(p T ) by h(p S * ϕ), which is the differential entropy of a Gaussian mixture with centers s ,j , j ∈ [n].

The term h(p S * ϕ) is referred to as the SP estimator of h(p T ) = h(p S * ϕ).

Conditional Entropies: Fix i ∈ [m] and consider the estimation of h(p T |X=xi ).

Note that p T |X=xi = p S |X=xi * ϕ since Z is independent of (X, T −1 ).

To sample from p S |X=xi , we feed x i into the DNN n i times, collect outputs from T −1 corresponding to different noise realizations, and apply f on each.

The obtained samples S DISPLAYFORM1 Mutual Information Estimator: Combining the above described pieces, we estimate I(X; T ) by DISPLAYFORM2 3 We set X ∼ Unif(X ) to conform with past works BID10 .

DISPLAYFORM3 log(2πeβ 2 ) because its previous layer is X (fixed).

DISPLAYFORM4 Before analyzing the performance ofĥ SP , we note that this estimation problem is statistically difficult in the sense that any good estimator of h(p S * ϕ) based on S n and ϕ requires exponentially many samples in d (Theorem 2 from Supplement 10).

Nonetheless, the following theorem shows that the SP estimator absolute-error risk converges at a satisfactory rate (Theorem 4 from Supplement 10 states this with all constants explicit, and Theorem 5 gives the results for ReLU).

DISPLAYFORM5 Evaluating the SP estimatorĥ SP (S n , ϕ) of the true entropy h(p S * ϕ) requires computing the differential entropy of the (known) Gaussian mixturep s n * ϕ sincê DISPLAYFORM6 Noting that the differential entropy h(p) = −E X∼p [log p(X)], we rewrite the SP estimator aŝ DISPLAYFORM7 where G ∼p s n * ϕ is distributed according to the Gaussian mixture.

We numerically approximate the right-hand side of (4) via efficient Monte Carlo (MC) integration .

Specifically, we generate n MC i.i.d.

samples fromp s n * ϕ and approximate the expectation by an empirical average.

This unbiased approximation achieves a mean squared error DISPLAYFORM8 (Supplement 10).

This approximation thus only adds a negligible amount to the error of the SP estimator h(p S * ϕ) −ĥ SP (S n , ϕ) itself.

There are other ways to numerically evaluate this expectation, such as the Gaussian mixture bounds from BID5 ; however, our proposed method is the fastest approach of which we are aware.

Remark 1 (Choosing Noise Parameter and Number of Samples) We describe practical guidelines for selecting the noise standard deviation β and the number of samples n for estimating I(X; T ) in an actual classifier.

Ideally, β should be treated as a hyperparameter tuned to optimize the performance of the classifier on held-out data, since internal noise serves as a regularizer similar to dropout.

In practice, we find it is sometimes necessary to back off from the β value that optimizes performance to a higher value to ensure accurate estimation of mutual information (the smaller β is, the more samples our estimator requires), depending on factors such as the dimensionality of the layer being analyzed and the number of data samples available for a task.

The number of samples n can be selected using the bound in Theorem 1, but because this theorem is a worst-case result, in practice it is quite pessimistic.

Specifically, generating the estimated mutual information curves shown in Section 5 requires running the SP estimator multiple times

, which makes the number of samples dictated by Theorem 1 infeasible.

To overcome this computational burden while adhering to the theoretical result, we tested the value of n given by the theorem on a few points of each curve and reduced it until the overall computation cost became reasonable.

To ensure estimation accuracy was not compromised we empirically tested that the estimate remained stable.

As a concrete example, to achieve an error bound of 5% of FIG3 plot's vertical scale (which amounts to an 0.4 absolute error bound), the number of samples required by Theorem 1 is n = 4 · 10

.

This number is too large for our computational budget.

Performing the above procedure for reducing n, we find good accuracy is achieved for n = 4 · 10 6 samples (Theorem 1 has the pessimistic error bound of 3.74 for this value).

Adding more samples beyond this value does not change the results.

Before presenting our empirical results, we connect compression to clustering using an informationtheoretic perspective.

Consider a single noisy neuron with a one-dimensional input X. Let T (k) = S(k) + Z be the neuron's output at epoch k, where S(k) σ(w k X + b k ), for a strictly monotone nonlinearity σ, and Z ∼ N (0, β 2 ).

Invariance of mutual information to invertible operations implies DISPLAYFORM0 From an information-theoretic perspective, I S(k); S(k) + Z is the aggregate information transmitted over an AWGN channel with input constellation DISPLAYFORM1 In other words, I S(k); S(k) + Z is a measure of how distinguishable the symbols of S k are when composed with Gaussian noise (roughly equals log of the number of resolvable clusters under noise level β).

Since the distribution of T (k) = S(k) + Z is a Gaussian mixture with means s ∈ S k , the closer two constellation points s and s are, the more overlapping the Gaussians around them will be.

Hence reducing point spacing in S k (by changing w k and b k ) directly reduces I X; T (k) .

Let σ = tanh and β = 0.01, and set X = X −1 ∪ X 1 , with X −1 = {−3, −1, 1} and X 1 = {3}, labeled −1 and 1, respectively.

We train the neuron using mean squared loss and gradient descent (GD) with a fixed learning rate of 0.01 to best illustrate the behavior of I X; T (k) .

The Gaussian mixture p T (k) is plotted across epochs k in FIG2 .

The learned bias is approximately −2.3w, ensuring that the tanh transition region correctly divides the two classes.

Initially w = 0, so all four Gaussians in p T (0) are superimposed.

As k increases, the Gaussians initially diverge, with the three from X −1 eventually re-converging as they each meet the tanh boundary.

This is reflected in the mutual information trend in FIG2 , with the dips in I X; T (k) around k = 10 3 and k = 10 4 corresponding to the second and third Gaussians respectively merging into the first.

Thus, there is a direct connection between clustering and compression.

FIG2 shows the mutual information for different noise levels β as a function of epoch.

For small β (as above) the X −1 Gaussians are distinct and merge in two stages as w grows.

For larger β, however, the X −1 Gaussians are indistinguishable for any w, making I(X; T ) only increase as the two classes gradually separate.

A similar example for a two-neuron network with leaky-ReLU nonlinearities is provided in the Supplement 8.

We now show the observations from our minimal examples also hold for two larger networks.

Namely, the presented experiments demonstrate the compression of mutual information in noisy networks is driven by clustering of internal representation, and that deterministic networks cluster samples as well (despite I(X; T ) being constant over these systems).

The DNNs we consider are: (1) the small, fully connected network (FCN) studied in BID10 , which we call the SZT model; and (2) a convolutional network for MNIST classification, called MNIST CNN.

We present selected results; additional details and experiments are found in the supplement.

Consider the data and model of BID10 for binary classification of 12-dimensional inputs using a fully connected 12-10-7-5-4-3-2 architecture.

The FCN was tested with tanh and ReLU nonlinearities as well as a linear model.

example, I(X; T 5 ) grows until epoch 28, when the Gaussians move away from each other along a curve (see scatter plots on the right).

Around epoch 80 they start clustering and I(X; T 5 ) drops.

At the end of training, the saturating tanh nonlinearities push the Gaussians to two furthest corners of the cube, reducing I(X; T 5 ) even more.

To confirm that clustering (via saturation) was central to the compression observed in FIG3 , we also trained the model using the regularization from ) (test classification accuracy 96%), which encourages orthonormal weight matrices.

The results are shown in FIG3 .

Apart from minor initial fluctuations, the bulk of compression is gone.

The scatter plots show that the vast majority of neurons do not saturate and no clustering is observed at the later stages of training.

Saturation is not the only mechanism that can cause clustering and consequently reduce I(X; T ).

classification accuracy 89%).

As seen from the scatter plots, due to the formation of several clusters and projection to a lower dimensional space, I(X; T ) drops even without the nonlinearities.

The results in FIG3 (a) and (b) also show that the relationship between compression and generalization performance is not a simple one.

In FIG3 , the test loss begins to increase at roughly epoch 3200 and continues to increase until training ends, while at the same time compression occurs in layers 4 and 5.

In contrast, in FIG3 (b) the test loss does not increase, and compression does not occur in layers 4 and 5.

We believe that this is a subject that deserves further examination in future work.

To provide another perspective on clustering that is sensitive to class membership, we compute histograms of pairwise distances between representations of samples, distinguishing within-class distances from between-class distances.

FIG18 shows histograms for the SZT models from Figs. 5(a) and (b).

As training progresses, the formation of clusters is clearly seen (layer 3 and beyond) for the unnormalized SZT model in FIG3 .

In the normalized model FIG3 ), however, no tight clustering is apparent, supporting the connection between clustering and compression.

Once clustering is identified as the source of compression, we focus on it as the point of interest.

To measure clustering, the discrete entropy of Bin(T ) is considered, where the number of equal-sized bins, B, is a tuning parameter.

Note that Bin(T ) partitions the dynamic range (e.g., [−1, 1] d for a tanh layer) into B d cells or bins.

When hidden representations are spread out, many bins will be non-empty, each assigned with a positive probability mass.

On the other hand, for clustered representations, the distribution is concentrated on a small number of bins, each with relatively high probability.

Recalling that discrete entropy is maximized by the uniform distribution, we see why reduction in H Bin(T ) measures clustering.

To illustrate this measure, we compute H Bin(T ) for each of the SZT models using bin size B = 10β (bottom plots in FIG3 ).

We can see a clear correspondence between H Bin(T ) and I(X; T ), indicating that although H Bin(T ) does not capture the exact value of I(X; T ), it follows this mutual information in measuring clustering.

This is particularly important when moving back to deterministic DNNs, where I(X; T ) is no longer an informative measure, being either a constant or infinity, for discrete or continuous X, respectively.

FIG0 shows H Bin(T ) for the deterministic SZT model (β = 0).

The bin size is a free parameter, and depending on its value, H Bin(T ) reveals different clustering granularities.

Moreover, since in deterministic networks T = f (X), for a deterministic map f , we have H Bin(T ) X = 0, and therefore I X; Bin(T ) = H Bin(T ) .

Thus, the plots from (Shwartz-Ziv & Tishby, 2017), ) and our Figs. 1 and 5(a), (b) and (c) all show the entropy of the binned T .

We now examine a model that is more representative of current machine learning practice: the MNIST CNN trained with dropout from Section 2.

FIG5 portrays the near-injective behavior of this model.

Even when only two bins are used to compute H Bin(T ) , it takes values that are approximately ln(10000) = 9.210, for all layers and training epochs, even though the two convolutional layers use max-pooling.

In FIG6 we show histograms of pairwise distances between MNIST validation set samples in the input (pixel) space and in the four layers of the CNN.

The histograms were computed for epochs 0, 1, 32, and 128, where epoch 0 is the initial random weights and epoch 128 is the final weights.

The histogram for the input shows that the mode of within-class pairwise distances is lower than the mode of between-class pairwise distances, but that there is substantial overlap.

Layers 1 and 2, which are convolutional and therefore do not contain any units that receive the full input, do little to reduce this overlap, suggesting that the features learned in these layers are somewhat generic.

In contrast, even after one epoch of training, layers 3 and 4, which are fully connected, separate the distribution of within-class distances from the distribution of between-class distances.

Regularization that limits the ability of a network to drive hidden units into saturation may limit or eliminate compression (and clustering) as seen in FIG3 .

FIG3 also demonstrated that I(X; T ) and H Bin(T ) are highly correlated, establishing the latter as an additional measure for clustering (applicable both in noisy and deterministic DNNs).(iv) Clustering of internal representations can also be observed in a somewhat larger, convolutional network trained on MNIST.

While FIG5 shows that due to the dimensionality, H Bin(T ) fails to track compression in the larger CNN, strong evidence for clustering is found via estimates done at the level of individual units (described in the text on the MNIST CNN) and the analysis of pairwise distances between samples shown in FIG6 .

In this work we reexamined the compression aspect of the Information Bottleneck theory (ShwartzZiv & Tishby, 2017) , noting that fluctuations of I(X; T ) in deterministic networks with strictly monotone nonlinearities are theoretically impossible.

Setting out to discover the source of compression observed in past works, we: (i) created a rigorous framework for studying and accurately estimating information-theoretic quantities in DNNs whose weights are fixed; (ii) identified clustering of the learned representations as the phenomenon underlying compression; and (iii) demonstrated that the compression-related experiments from past works were in fact measuring this clustering through the lens of the binned mutual information.

In the end, although binning-based measures do not accurately estimate mutual information, they are simple to compute and prove useful for tracking changes in clustering, which is the true effect of interest in deterministic DNNs.

We believe that further study of geometric phenomena driven by DNN training is warranted to better understand the learned representations and to potentially establish connections with generalization.

Paper under double-blind review

Let (A, B) be a pair of random variables with values in the product set A × B and a joint distribution P A,B (whose marginals are denoted by P A and P B ).

The mutual information between A and B is: DISPLAYFORM0 where DISPLAYFORM1 is the Radon-Nikodym derivative of P A,B with respect to the product measure P A × P B .

We are mostly interested in the scenario where A is discrete with a probability mass function (PMF) p A , and given A = a ∈ A, B is continuous with probability density function (PDF) p B|A=a p B|A (·|a).

In this case, (5) simplifies to DISPLAYFORM2 Defining the differential entropy of a continuous random variable C with PDF p C supported in C as DISPLAYFORM3 the mutual information from (6) can also be expressed as DISPLAYFORM4 The subtracted term above is the conditional differential entropy of B given A, denoted by h(B|A).

To expand upon Section 4, we provide here a second example to illustrate the relation between clustering and compression of mutual information.

In particular, this example also shows that as opposed to the claim from , non-saturating nonlinearities can achieve compression.

Consider the non-saturating Leaky-ReLU nonlinearity R(x) max(x, x/10).

Let X = X 0 ∪ X1 /4 , with X 0 = {1, 2, 3, 4} and X1 /4 = {5, 6, 7, 8}, and labels 0 and 1/4, respectively.

We train the network via GD with learning rate 0.001 and mean squared loss.

Initialization (shown in FIG7 ) was chosen to best illustrate the connection between the Gaussians' motion and mutual information.

The network converges to a solution where w 1 < 0 and b 1 is such that the elements in X1 /4 cluster.

The output of the first layer is then negated using w 2 < 0 and the bias ensures that the elements in X 0 are clustered without spreading out the elements in X1 /4 .

Figs. 9(b) show the Gaussian motion at the output of the first layer and the resulting clustering.

For the second layer ( FIG7 ), the clustered bundle X1 /4 is gradually raised by growing b 2 , such that its elements successively split as they cross the origin; further tightening of the bundle is due to shrinking |w 2 |.

FIG7 shows the mutual information of the first (blue) and second (red) layers.

The merging of the elements in X1 /4 after their initial divergence is clearly reflected in the mutual information.

Likewise, the spreading of the bundle, and successive splitting and coalescing of the elements in X1 /4 are visible in the spikes in the red mutual information curve.

The figure also shows how the bounds on I X; T (k) precisely track its evolution.

9 EXPERIMENTAL DETAILS 9.1 SZT MODELIn this section we provide additional experimental details and results for the SZT model discussed in Section 5 of the main paper.

To regularize the network weights, we followed ) and adopted their approach for enforcing an orthonormality constraint.

Specifically, we first update the weights {W } ∈[L] using the standard gradient descent step, and then perform a secondary update to set DISPLAYFORM0 where the regularization parameter α controls the strength of the orthonormality constraint.

The value of α was was selected from the set {1.0 × 10 for both the tanh and ReLU.In FIG0 we present additional experimental results that provide further insight into the clustering and compression phenomena for both tanh and ReLU nonlinearities.

FIG0 shows what happens when the additive noise has a high variance.

In this case, although saturation still occurs (see the histograms on top of FIG0 ) and the Gaussians still cluster together (see the scatter plots on the right for the epoch 54 and epoch 8990), compression overall is very mild.

The effect of increasing the noise parameter was explained in Section 4 of the main text (see, in particular, FIG2 .

Comparing FIG0 to FIG3 of the main text, for which β = 0.005 was used and compression was observed, further highlights the effect of large β.

Recall that smaller β values correspond to narrow Gaussians, while larger β values correspond to wider Gaussians.

When β is small, even Gaussians that belong to the same cluster are distinguishable so long as they are not too close.

When clusters tighten, the in-class movement brings these Gaussians closer together, effectively merging them, and causing a reduction in mutual information (compression).

One the other hand, for large β, the in-class movement is blurred at the outset (before clusters tighten).

Thus, the only effect on mutual information is the separation between the clusters: as these blobs move away from each other, mutual information rises.

Based on the above observation, we can conclude that while the two notions of "clustering Gaussians" and "compression/decrease in mutual information" are strongly related in the low-beta regime, once the noise becomes large, these phenomena decouple, i.e., the network may cluster inputs and neurons may saturate, but this will not be reflected in a decrease of mutual information.

Finally, we present results for ReLU activation without weight normalization FIG0 ) and with orthonormal weight regularization FIG0 ).

We see that both these networks exhibit almost no compression.

FIG0 , the lack of compression is attributed to regularization of the weight matrices, as explained in Section 5 of the main text.

FIG0 , the reduction in compression can be explained by the fact that although ReLU forces saturation of the neurons at the origin (which promotes clustering), since the positive axes remain unconstrained, the Gaussians can move off towards infinity without bound.

This is visible from the histograms in the top row of FIG0 , where, for example, in layer 5 the neurons can take arbitrarily large positive values (note that the bin corresponding to the value 5 accumulates all the values from 5 to infinity).

Therefore, the clustering at the origin and the potential drop in mutual information is counterbalanced by the spread of Gaussians along the positive axes and the potential increase of mutual information it causes.

Eventually, this leads to the approximately constant profile of the mutual information plot in FIG0 .

The behavior of the weight-normalized ReLU in FIG0 is similar to FIG0 , although now the growth of the network weights is bounded and the saturation around origin is reduced.

For example, for layers 4 and 5 we can see an upward trend in the mutual information, which is then flattened at the end of training.

This occurs since more Gaussians are moving away from the origin, although their motion remains bounded (see the histograms on the top and the scatter plots on the right), thus decreasing the clustering density, leading to the rise in the mutual information profile.

Once the Gaussians are prevented from moving any further along the positive axes, a slight compression occurs and the mutual information flattens.

In this section we present results for another synthetic example.

We generated data in the form of spiral as in FIG0 .

The network architecture was similar to SZT model, except that the size of each layer was set to 3.

FIG0 shows MI estimates I(X; T ) computed using SP estimator and the discrete entropy estimates H Bin(T ) for weight un-normalized FIG0 and normalized models FIG0 and using additive noise β = 0.005.

Similar as in the main paper, the results in the figure illustrate a connection between clustering and compression.

Figure 11: Generated spiral data for binary classification problem.

Finally, in FIG0 we also show an estimate of H Bin(T ) for the case of deterministic DNN trained on spiral data.

For the particular choice of the bin size, the result of the estimated entropy reveal a certain level of clustering granularity.

In this section, we describe in detail the architecture of the MNIST CNN models used in Sections 2 and 5 in the main paper.

The MNIST CNNs were trained using PyTorch BID28 version 0.3.0.post4.

The CNNs use the following fairly standard architecture with two convolutional layers, two fully connected layers, and batch normalization.1.

2-d convolutional layer with 1 input channel, 16 output channels, 5x5 kernels, and input padding of 2 pixels 2.

Batch normalization 3.

Tanh() activation function 4.

Zero-mean additive Gaussian noise with variance β 2 or dropout with a dropout probability of 0.2 5.

2x2 max-pooling 6.

2-d convolutional layer with 16 input channels, 32 output channels, 5x5 kernels, and input padding of 2 pixels 7.

Batch normalization 8.

Tanh() activation function 9.

Zero-mean additive Gaussian noise with variance β 2 or dropout with a dropout probability of 0.2 10.

2x2 max-pooling 11.

Fully connected layer with 1586 (32x7x7) inputs and 128 outputs 12.

Batch normalization 13.

Tanh() activation function 14.

Zero-mean additive Gaussian noise with variance β 2 or dropout with a dropout probability of 0.2 15.

Fully connected layer with 128 inputs and 10 outputs All convolutional and fully connected layers have weights and biases, and the weights are initialized using the default initialization, which draws weights from Unif[−1/ √ m, 1/ √ m], with m the fanin to a neuron in the layer.

Training uses cross-entropy loss, and is performed using stochastic gradient descent with no momentum, 128 training epochs, and 32-sample minibatches.

The initial learning rate is 5 × 10 DISPLAYFORM0 , and it is reduced following a geometric schedule such that the learning rate in the final epoch is 5 × 10 −4.

To improve the test set performance of our models, we applied data augmentation to the training set by translating, rotating, and shear-transforming each training example each time it was selected.

Translations in the x-and y-directions were drawn uniformly from {−2, −1, 0, 1, 2}, rotations were drawn from Unif(−10• , 10 • ), and shear transforms were drawn from Unif(−10• , 10 • ).To obtain more reliable performance results, we train eight different models and report the mean number of errors and standard deviation of the number of errors on the MNIST validation set.

To ensure that the internal representations of different models are comparable, which is necessary for the use of the cosine similarity measure between internal representations, for each noise condition (deterministic, noisy with β = 0.05, noisy with β = 0.1, noisy with β = 0.2, noisy with β = 0.5, and dropout with p = 0.2), we use a common random seed (different for the eight replications, of course) so the models have the same initial weights and access the training data in the same order (use the same minibatches).At test time, all models are fully deterministic: the additive noise blocks and dropout layers are replaced by identities.

Thus, in the figures and text in the main paper, "Layer 1" is the output of step 5 (2x2 max-pooling), "Layer 2" is the output of step 10 (2x2 max-pooling), "Layer 3" is the output of step 13 (Tanh() activation function), and "Layer 4" is the output of step 15 (fully connected layer with 10 outputs).

Both conditional and unconditional entropy estimators reduce to the problem of estimating h(p S * ϕ) using i.i.d.

samples S n (S i ) i∈[n] from S ∼ p S while knowing ϕ. In this section we state performance guarantees for the SP estimator.

These results are excerpted from our work BID15 , where this estimation problem is thoroughly studied.

The interested reader is referred to BID15 for proofs of the subsequently stated results.

Let F d be the set of distributions P with supp(P ) DISPLAYFORM0 The minimax absolute-error risk over DISPLAYFORM1 whereĥ is an estimator of h(P * ϕ) based on the empirical data S n = (S 1 , . . .

, S n ) of i.i.d.

samples from P and the noise parameter β Definition 1 (Subgaussian Random Variable) A random variable X is subgaussian if it satisfies either of the following equivalent properties 1.

Tail condition: DISPLAYFORM2 3.

Super-exponential moment: DISPLAYFORM3 where K i , for i = 1, 2, 3, differ by at most an absolute constant.

Furthermore, the subgaussian norm X ψ2 of a subgaussian random variable X is defined as the smallest K 2 in property 2, i.e., DISPLAYFORM4 DISPLAYFORM5 for all n ∈ N and δ > 0, whenever K ≥ 1.

As explained in the following remark, the considered subgaussianity requirement is naturally satisfied by our noisy DNN framework.

2.

Discrete distributions over a finite set, which is a special case of bounded support.3.

Distributions of the random variable S = f (T −1 ) in a noisy ReLU DNN, so long as the input X to the network is itself subgaussian.

To see this recall that linear combinations of independent subgaussian random variables is also subgaussian.

Furthermore, for any (scalar) random variable A, we have that ReLU(A) = max{0, A} ≤ |A|, almost surely.

Now, since each layer in a noisy ReLU DNN is nothing but a coordinate-wise ReLU applied to a linear transformation of the previous layer plus a Gaussian noise, one may upper bound FORMULA7 of Definition 1, provided that the input X is coordinate-wise subgaussian.

The constant K 2 will depend on the network's weights and biases, the depth of the hidden layer, the subgaussian norm of the input X ψ2 and the noise variance.

This input subgaussianity assumption is, in particular, satisfied by the distribution of X considered herein, i.e., by X ∼ Unif(X ).

DISPLAYFORM6

We start with two converse claims establishing that the sample complexity is exponential in d. The first claim states that there exists a class of distributions P , for which the estimation of h(P * ϕ) cannot be done with fewer than exponentially many samples in d, when d is sufficiently large.

DISPLAYFORM0 The fact that the exponent γ(β) is monotonically decreasing in β suggests that larger values of β are favorable for estimation.

Theorem 2 shows that an exponential sample complexity is inevitable when d is large.

As a complementary result, the next theorem gives a sample complexity lower bound valid in any dimension but only for small enough noise variances.

Nonetheless, the result is valid for orders of β considered in this work. , where DISPLAYFORM1 Remark 2 We state Theorem 3 asymptotically in β for the sake of simplicity, but for any d it is possible to follow the constants through the proof to determine a value c such that Theorem 3 holds for all β < c. For example for d = 1, a careful analysis gives that Theorem 3 holds for all β < 0.08, which is satisfied by most of the experiments run in this paper.

This threshold on β changes very slowly with increasing d due to the rapid decay of the PDF of the normal distribution.

We next focus on analyzing the performance of the SP estimator.

For any fixed S n = s n , denote the empirical PMF associated with s DISPLAYFORM0 The estimatorĥ SP (s n ) also depends on β, but we omit this from our notation.

The following theorem shows that the expected absolute error ofĥ SP decays like O

√ n for all dimensions d. We provide explicit constants (in terms of β and d), which present an exponential dependence on the dimension, in accordance to the results of Theorems 2 and 3.Theorem 4 (SP Estimator Absolute-Error Risk for Bounded Support) Fix β > 0, d ≥ 1 and any > 0.

The absolute-error risk of the SP estimator (11) over the class F d , for all n sufficiently large, is bounded as DISPLAYFORM0 where DISPLAYFORM1 and the right-hand sides (RHSs) of (12) and (13) are, respectively, explicit and implicit upper bounds on the minimax absolute-error risk R d (n, β).Remark 3 (Comparison to General-Purpose Estimators) Note that one could always sample ϕ and add up these noise samples to S n to obtain a sample set from P * ϕ. These samples can be used to get a proxy of h(P * ϕ) via a kNN-or a KDE-based differential entropy estimator.

However, P * ϕ violated the boundedness away from zero assumption that most of the convergence rate results in the literature rely on BID27 BID20 BID23 BID21 BID32 BID18 BID24 BID31 BID25 .

The only result we are aware of that analyses a differential entropy estimator (namely, the kNN-based estimator from (A. BID14 ) without assuming the density is bounded from below BID22 ) relies on the density being supported inside [0, 1] d , satisfying periodic boundary conditions and having a Hölder smoothness parameter s ∈ (0, 2].

The convolved density P * ϕ satisfies neither of these three conditions.

Furthermore, because the SP estimator is constructed to exploit the particular structure of our estimation setup it achieves a fast convergence rate of .

This highlights the advantage of ad-hoc estimation as opposed to general-purpose estimation.

Theorem 1 provides convergence rates when estimating differential entropy (or mutual information) over DNNs with bounded activation functions, such as tanh or sigmoid.

To account for networks with unbounded nonlinearities, such as ReLU networks, the following theorem gives a more general result of estimation over the nonparametric class F d,K , for all n sufficiently large, is bounded as DISPLAYFORM2 where c β,d d 2 log(2πβ 2 ).

In particular, DISPLAYFORM3 and the RHSs of (14) and (15) are, respectively, explicit and implicit upper bounds on the minimax absolute-error risk R d,K (n, β).As mentioned in Remark 1, the class F (SG) d,K is rather general, and, in particular, includes F d whenever K ≥ 1.

This means that Theorem 5 also provides an upper bound on the minimax risk under the setup of Theorem 1.

Nonetheless, we chose to separately state Theorem 1 since the derivation under the bounded support assumption enables extracting slightly better constants (which is important for our applications -see Section 5).

We do highlight, however, that the expressions from FORMULA4 and FORMULA4 with K = 1 not only have the same convergence rates, but their constants are also very close.

Remark 4 (Near Minimax Rate-Optimality) A convergence rate faster than 1 √ n cannot be attained for parameter estimation under the absolute-error loss.

This follows from, e.g., Proposition 1 of BID16 , which establishes this convergence rate as a lower bound for the parametric estimation problem given n i.i.d.

samples.

Consequently, the convergence rate of O σ,d

√ n established in Theorems 1 and 5 for the SP estimator is near minimax rate-optimal (i.e., up to logarithmic factors).Remark 5 (Mutual Information Estimation) Denoting the upper bound on the estimation error from Theorem 1 or 5 by ∆ n (β, d), we see that the error of the mutual information estimator from (2) is bounded as by 2∆ n (β, d), which vanishes as n → ∞.

The results of the previous subsection are of minimax flavor.

That is, they state worst-case convergence rates of the SP estimation over a certain nonparametric class of distributions.

In practice, the true distribution may very well not be one that attains these worst-case rates, and convergence may be faster.

However, while variance ofĥ SP (S n ) can be empirically evaluated using bootstrapping, there is no empirical test for the bias.

Even if multiple estimations of h(P * ϕ) viaĥ SP (S n ) consistently produce similar values, this does not necessarily suggest that these values are close to the true h(P * ϕ).

To have a guideline to the least number of samples needed to avoid biased estimation, we present the following lower bound on sup P ∈F d E S n h(P * ϕ) −ĥ SP (S n ) .

is the inverse of the Q-function.

By the choice of , clearly k ≥ 2, and the bias of the SP estimator over the class F d is bounded as DISPLAYFORM0 Consequently, the bias cannot be less than a given δ > 0 so long as n ≤ k DISPLAYFORM1 .Since H b ( ) shrinks with , for sufficiently small values the lower bound from FORMULA4 .

Thus, with these parameters, the number of estimation samples n should be at least 2

, for any conceivably relevant dimension, in order to have negligible bias.

Evaluating the mutual information estimator from (2) requires computing the differential entropy of a Gaussian mixture.

Although it cannot be computed in closed form, this section presents a method for approximate computation via MC integration .

To simplify the presentation, we present the method for an arbitrary Gaussian mixture without referring to the notation of the estimation setup.

1 n i∈[n] ϕ(t − µ i ) be a d-dimensional, n-mode Gaussian mixture, with {µ i } i∈[n] ⊂ R d and ϕ as the PDF of N (0, β 2 I d ).

Let C ∼ Unif{µ i } i∈[n] be independent of Z ∼ ϕ and note that DISPLAYFORM0 We use Monte Carlo (MC) integration to compute the h(g).

First note that h(g) = −E log g(V ) = − 1 n i∈ [n]

E log g(µ i + Z) C = µ i = − 1 n i∈ [n]

E log g(µ i + Z),where the last step follows by the independence of Z and C. Let Z samples from ϕ.

For each i ∈ [n], we estimate the i-th summand on the RHS of (17) bŷ DISPLAYFORM1 3 The Q-function is defined as Q(x) We have the following bounds on the MSE for tanh and ReLU networks.

Theorem 7 (MSE Bounds for MC Estimator) DISPLAYFORM2 almost surely (i.e., tanh network), then DISPLAYFORM3 (ii) Assume M C E C 2 2 < ∞ (e.g., ReLU network with bounded 2nd moments), then DISPLAYFORM4 The bounds on the MSE scale only linearly with the dimension d, making σ 2 in the denominator often the dominating factor experimentally.

We briefly present empirical results illustrating the convergence of the SP estimator and comparing it to two current state-of-the-art methods: the KDE-based estimator of BID25 and the kNN-based estimator often known as the Kozachenko-Leonenko (KL) nearest neighbor estimator BID26 BID22 .

In this example, the distribution P of S is set to be a mixture of Gaussians truncated to have support in [−1, 1] The kernel width for the KDE estimate was chosen via cross-validation, varying with both d and n; the kNN estimator andĥ SP (S n ) require no tuning parameters.

We found that the KDE estimate is highly sensitive to the choice of kernel width, the curves shown correspond to optimized values and are highly unstable to any change in kernel width.

Note that both the kNN and the KDE estimators converge slowly, at a rate that degrades with increased d. This rate is worse than that ofĥ SP , which also lower bounds the true entropy (as according to our theory -see BID15 , Equation FORMULA18 ).

@highlight

Deterministic deep neural networks do not discard information, but they do cluster their inputs.

@highlight

This paper provides a principled way to examine the compression phrase in deep neural networks by providing an theoretical sounding entropy estimator to estimate mutual information. 