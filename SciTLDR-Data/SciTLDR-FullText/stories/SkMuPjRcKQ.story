Probabilistic Neural Networks deal with various sources of stochasticity: input noise, dropout, stochastic neurons, parameter uncertainties modeled as random variables, etc.

In this paper we revisit a feed-forward propagation approach that allows one to estimate for each neuron its mean and variance w.r.t.

all mentioned sources of stochasticity.

In contrast, standard NNs propagate only point estimates, discarding the uncertainty.

Methods propagating also the variance have been proposed by several authors in different context.

The view presented here attempts to clarify the assumptions and derivation behind such methods, relate them to classical NNs and broaden their scope of applicability.

The main technical contributions are new approximations for the distributions of argmax and max-related transforms, which allow for fully analytic uncertainty propagation in networks with softmax and max-pooling layers as well as leaky ReLU activations.

We evaluate the accuracy of the approximation and suggest a simple calibration.

Applying the method to networks with dropout allows for faster training and gives improved test likelihoods without the need of sampling.

Despite the massive success of Neural Networks (NNs) considered as deterministic predictors, there are many scenarios where a probabilistic treatment is highly desirable.

One of the best known techniques to improve the network generalization is dropout BID29 , which introduces multiplicative Bernoulli noise in the network.

At test time, however, it is commonly approximated by substituting the mean value of the noise variables.

Computing the expectation more accurately by Monte Carlo (MC) sampling has been shown to improve test likelihood and accuracy BID29 BID5 but is computationally expensive.

Another challenging problem in NNs is the sensitivity of the output to perturbations of the input, in particular random and adversarial perturbations BID19 BID3 BID23 .

In FIG6 we illustrate the point that the average of the network output under noisy input differs from propagating the clean input.

It is therefore desirable to estimate the output uncertainty resulting from the uncertainty of the input.

In classification networks, propagating the uncertainty of the input can impact the confidence of the classifier and its robustness BID1 .

We would like that a classifier is not overconfident when making errors.

However such high confidences of wrong predictions are typically observed in NNs.

Similarly, when predicting real values (e.g. in optical flow estimation), it is desirable to estimate also their confidences.

Taking into account uncertainties from input or dropout allows to predict output uncertainties better correlated with the test error BID14 BID6 BID26 .

Another important problem is overfitting, which may be addressed in a sound way with Bayesian learning.

The parameters are considered as random variables and are determined up to an uncertainty implied by the training data.

This uncertainty needs then to be propagated to predictions at the test-time.

The above scenarios motivate considering NNs with different sources of stochasticity not as deterministic feed-forward networks but as directed probabilistic graphical models.

We focus on the

In probabilistic NNs, all units are considered random.

In a typical network, units are organized by layers.

There are l layers of hidden random vectors X k , k = 1, . . .

l and X 0 is the input layer.

Each vector X k has n k components (layer units) denoted X k i .

The network is modeled as a conditional Bayesian network (a.k.a.

belief network, BID21 ) defined by the pdf DISPLAYFORM0 We further assume that the conditional distribution p(X k | X k−1 ) factorizes as p( DISPLAYFORM1 are activations.

In this work we do not consider Bayesian learning and the weights w are assumed to be non-random, for clarity.

We will denote values of r.v.

X k by x k , so that the event X k = x k can be unambiguously denoted just by x k .

Notice also that we consider biases of the units implicitly via an additional input fixed to value one.

The posterior distribution of each layer k > 0, given the observations x 0 , recurrently expresses as DISPLAYFORM2 The posterior distribution of the last layer, p(X l | x 0 ) is the model's predictive distribution.

Standard NNs with injected noises give rise to Bayesian networks of the form (1) as follows.

Consider a deterministic nonlinear mapping applied component-wise to noised activations: DISPLAYFORM3 where f : R → R and Z k i are independent real-valued random variables with a known distribution (such as the standard normal distribution).

From representation (3) we can recover the conditional cdf DISPLAYFORM4 and the respective conditional density of the belief network.

Example 1.

Stochastic binary unit BID32 .

Let Y be a binary valued r.v.

given by Y = Θ(A − Z), where Θ is the Heaviside step function and Z is noise with cdf F Z .

Then P(Y =1 | A) = F Z (A).

This is easily seen from P(Y =1 | A) = P(Θ(A − Z) = 1 A) = P(Z ≤ A|A = F Z (A).If, for instance, Z has standard logistic distribution, then P(Y =1 | A) = S(A), where S is the logistic sigmoid function S(a) = (1 + e −a ) −1 .In general, the expectation (2) is intractable to compute and the resulting posterior can have a combinatorial number of modes.

However, in many cases of interest it is suitable to approximate the posterior p(X k | x 0 ) for a given x 0 with a factorized distribution q(X k ) = i q(X k i ).

We expect that in many recognition problems, given the input image, all hidden states and the final prediction are concentrated around some specific values (unlike in generative problems, where the posterior distributions are typically multi-modal).

A similar factorized approximation is made for the activations.

The exact shape of distributions q(X k i ) and q(A k i ) can be chosen appropriately depending on the unit type: e.g., a Bernoulli distribution for binary X k i a Gaussian or Logistic distribution for realvalued activations A k i .

We will rely on the fact that the mean and variance are sufficient statistics for such approximating distributions.

Then, as long as we can calculate these sufficient statistics for the layer of interest, the exact shape of distributions for the intermediate outputs need not be assumed.

The information-theoretic optimal factorized approximation to the posterior p( DISPLAYFORM5 and is given by the marginals i p(X k i | x 0 ).

Furthermore, in the case when q(X k i ) is from an exponential family, the optimal approximation is given by matching the moments of q( DISPLAYFORM6 .

The factorized approximation then can be computed layer-by-layer, assuming that the preceding layer was already approximated.

Substituting DISPLAYFORM7 Thus we need to propagate the factorized approximation layer-by-layer with the marginalization update (5) until we get the approximate posterior output q(X l ).

This method is closely related to the assumed density filtering (see BID18 , in which, in the context of learning, one chooses a family of distributions that is easy to work with and "projects" the true posterior onto the family after each measurement update.

Here, the projection takes place after propagating each layer, for the purpose of the inference.

We now detail how (5) is computed (approximately) for a single layer consisting of a linear mapping A = w T X (scalar output, for clarity) and a non-linear noisy activation Y = f (A − Z).Linear Mapping An activation A in a typical deep network is a linear combination of many inputs X from the previous layer.

This justifies the assumption that A − Z (where Z is a smoothly distributed injected noise) can be approximated by a uni-modal distribution fully specified by its mean and variance such as normal or logistic distribution 1 .

Knowing the statistics of Z, we can estimate the mean and the variance of the activation A as DISPLAYFORM0 DISPLAYFORM1 where µ is the mean and Cov[X] is the covariance matrix of X. The approximation of the covariance matrix by its diagonal is implied by the factorization assumption for the activations A.Nonlinear Coordinate-wise Mappings Let A be a scalar r.v.

with statistics (µ, σ 2 ) and let Y = f (A−Z) with independent noise Z. Assuming that A = A−Z is distributed normally or logistically with statisticsμ,σ 2 , we can approximate the expectation and the variance of Y = f ( A), DISPLAYFORM2 by analytic expressions for most of the commonly used non-linearities.

For binary variables, occurring in networks with Heaviside nonlinearities, the distribution q(Y ) is fully described by one DISPLAYFORM3 , and the propagation rule (5) becomes DISPLAYFORM4 where the variance is dependent but will be needed in propagation through further layers.

Example 2.

Heaviside Nonlinearity with Noise.

Consider the model Y = Θ(A − Z), where Z is logistic noise.

The statistics of A = A−Z are given byμ = µ andσ 2 = σ 2 +σ 2 S , where σ DISPLAYFORM5 is the variance of Z. Assuming noisy activations A to have logistic distribution, we obtain the mean of Y as: DISPLAYFORM6 where the dotted equality holds because −(Ã −μ) σ S σ has standard logistic distribution whose cdf is the sigmoid function S. The variance of Y is expressed as in (8).

Summarizing, we can represent the approximate inference in networks with binary and continuous variables as a feed-forward moment propagation: given the approximate moments of X k−1 | x 0 , the moments of X

The standard NN can be viewed as a further simplification of the proposed method: it makes the same factorization assumption but does not compute variances of the activations (6b) and propagates only the means.

Consequently, a zero variance is assumed in propagation through non-linearities.

In this case the expected values of mappings such as Θ(A) and ReLU(A) are just these functions evaluated at the input mean.

For injected noise models we obtain smoothed versions: e.g., substituting σ = 0 in the noisy Heaviside function (9) recovers the standard sigmoid function.

We thus can view standard NNs as making a simpler form of factorized inference in the same Bayesian NN model.

We designate this simplification (in figures and experiments) by AP1 and the method using variances by AP2 ("AP" stands for approximation).

In this section we present our main technical contribution: propagation rules for argmax, softmax and max mappings, which are non-linear and multivariate.

Similar to how a sigmoid function is obtained as the expectation of the Heaviside function with injected noise in Example 2, we observe that softmax is the expectation of argmax with injected noise.

It follows that the standard NN with softmax layer can be viewed as AP1 approximation of argmax layer with injected noise.

We propose a new approximation for the argmax posterior probability that takes into account uncertainty (variances) of the activations and enables propagation through argmax and softmax layers.

Next, we observe that the maximum of several variables (used in max-pooling) can be expressed through argmax.

This gives a new one-shot approximation of the expected maximum using argmax probabilities.

The logarithm of softmax, important in variational Bayesian methods can be also handled as shown in § A.2.

Finally, we consider the case of leaky ReLU, which is a maximum of two correlated variables.

The proposed approximations are relatively easy to compute and are continuously differentiable, which facilitates their usage in NNs.

The softmax function, most commonly used to model a categorical distribution, thus ubiquitous in classification, is defined as p(Y =y|x) = e xy / k e x k , where y is the class index.

We explore the following latent variable representation known in the theory of discrete choice: DISPLAYFORM0 n is the indicator of the noisy argmax: DISPLAYFORM1 and Γ k follow the standard Gumbel distribution.

Standard NNs implement the AP1 approximation of this latent model: conditioned on X = x, the expectation over latent noises Γ is the softmax(x).For the AP2 approximation we need to compute the expectation w.r.t.

both: X and Γ, or, what is the same, to compute the expectation of softmax(X) over X. This task is difficult, particularly because variances of X i may differ across components.

First, we derive an approximation for the expectation of argmax indicator without injected noise: DISPLAYFORM2 The injected noise case can be treated by simply increasing the variance of each X i by the variance of standard Gumbel distribution.

Let X k , k = 1, . . .

, n be independent, with mean µ k and variance σ 2 k .

We need to estimate DISPLAYFORM3 The vector U with components U k = X y − X k for k = y is from R n−1 with component means DISPLAYFORM4 Notice that the components of U are not independent.

More precisely, the covariance matrix hasσ 2 k on diagonal and all off-diagonal elements equal σ 2 k .

We approximate the distribution of U by the (n−1)-variate logistic distribution defined by BID17 .

This choice is motivated by the following facts: its cdf S n−1 (u) = 1 1+ k e −u k is tractable and is seen to be equivalent to the softmax function; its covariance matrix is (I + 1)σ 2 S /2, where I is the identity matrix, i.e. it has similar structure to that of U .

The approximation is made by shifting and rescaling the distribution of U in order to match the means and marginal variances, i.e. (U k −μ k )σ S /σ k is approximated with standard (n−1)-variate logistic distribution.

This approximation allows to evaluate the necessary probability as DISPLAYFORM5 Expandingμ,σ 2 and noting that µ k − µ y = 0 for y = k, we obtain the approximation DISPLAYFORM6 Computing this approximation has linear memory complexity but requires quadratic time in the number of inputs, which may be prohibitive for some applications.

We now derive a simpler linear-time approximation used in all our experiments.

The variable X y is decomposed as DISPLAYFORM0 , where σ a is chosen as σ a = min k σ k so that the decomposition is valid for all k. The variables U k are introduced as DISPLAYFORM1 .

The probability P(U ≥ −Z|Z) is approximated, the same way as above, by fitting U with (n − 1)-variate logistic distribution, DISPLAYFORM2 To achieve linear complexity, this is simplified now with approximatingσ DISPLAYFORM3 where we denoted Z = Z/s + log S and S = k =y exp(−μ k /s).

The latter expectation in FORMULA0 is that of a regular sigmoid function, which we approximate similar to (9) as DISPLAYFORM4 where DISPLAYFORM5 Expanding S in (16), as it depends on the label y, and rearranging we obtain the approximation: DISPLAYFORM6 This approximation is similar to softmax but reweighs the summands differently if σ y differs from σ a .

Clearly, it can be computed in linear time.

In case when all input variances are equal, the approximation is equivalent to (13).

In case when input variances are that of standard Gumbel distribution, the approximation recovers back the standard softmax of µ k .

Let X k , k = 1, . . .

, n be independent, with mean µ k and variance σ 2 k .

The moments of the maximum Y = max k X k , assuming the distributions of X k are known, can be computed from the cdf of Y given by F Y (y) = P(X k ≤ y ∀k) = k F X k (y), by numerical integration of this cdf (Ross, 2010, sec. 3

We seek a simpler approximation.

One option is to compose the maximum of n > 2 variables hierarchically using maximum of two variables (discussed below) assuming normality and independence of intermediate results.

We propose a new non-trivial one-shot approximations for the mean and variance provided that the argmax probabilities q k = P(X k ≥ X j ∀j) are already estimated.

The derivation of these approximations and proofs of their accuracy are given in § A.1.

DISPLAYFORM0 where H(q k ) is the entropy of the Bernoulli distribution with probability q k .

Notice that the entropy is non-negative, and thus µ increases when the argmax is ambiguous, as expected in the extreme value theory.

The variance of Y can be approximated as DISPLAYFORM1 where a = −1.33751 and b = 0.886763 are coefficients originating from a Taylor expansion.

The function max(X 1 , X 2 ) allows to model popular leaky ReLU and maxOut layers.

Although the expressions for the moments are known and have been used in the literature, e.g., (Hernández-Lobato & BID10 BID6 , we propose approximations that are more practical for end-to-end learning: cheap to compute and having asymptotically correct output to input variance ratio for small noises.

The exact expressions for the moments for the maximum of two Gaussian random variables X 1 , X 2 are as follows BID20 .

Denoting s = (σ DISPLAYFORM0 1 2 and a = (µ 1 −µ 2 )/s, the mean and variance of max(X 1 , X 2 ) can be expressed as: DISPLAYFORM1 DISPLAYFORM2 where φ and Φ are the pdf and the cdf of the standard normal distribution, resp.

As Φ has to be numerically approximated with other functions, this has high computational cost and poor relative accuracy for large |a|.

The difference of such functions occurring in (20b) may result in a negative output variance, the approximation becomes inaccurate for small noises.

For the mean, we can substitute Φ(a) with an approximation such as logistic cdf S(a/σ S ).

To approximate the variance, we express it as DISPLAYFORM3 We observe that the function of one variable a 2 Φ(a) + aφ(a) − (aΦ(a) + φ(a)) 2 is always negative, quickly vanishes with increasing |a| and is above −0.16.

By neglecting it, we obtain a rather tight upper bound σ 2 ≤ σ 2 1 Φ(a) + σ 2 2 (1 − Φ(a)), i.e., in the form of two non-negative summands.

In case of LReLU defined as Y = max(αX, X), the variance can be approximated more accurately.

Assume that α < 1, let X 2 = αX 1 and denote µ = µ 1 and σ 2 = σ 2 1 .

Substituting, we obtain DISPLAYFORM4 The variance σ 2 expresses as DISPLAYFORM5 where DISPLAYFORM6 2 is a sigmoid-shaped function of one variable.

In practice we approximate σ 2 with the simpler function DISPLAYFORM7 where t = 0.3758 is set by fitting the approximation.

The approximation is shown in FIG0 with more detailed evaluation given in § B.1.

In the experiments we evaluate the accuracy of the proposed approximation and compare it to the standard propagation.

We also test the method in the end-to-end learning and show that with a simple calibration it achieves better test likelihoods than the state-of-the-art.

Full details of the implementation, training protocols, used datasets and networks are given in § C. The running time of AP2 is 2× more for a forward pass and 2-3× more for a forward-backward pass than that of AP1.

The accuracy of approximations of the individual layers is evaluated in § B and is deemed sufficient for approximately propagating uncertainty and computing derivatives.

We now consider multiple layers.

We conduct two experiments: how well the proposed method approximates the real posterior of neurons, w.r.t.

noise in the network input and w.r.t.

dropout.

The first case (illustrated in FIG6 ) is studied on the LeNet5 model of LeCun et al. FORMULA0 , a 5-layer net with max pooling detailed in § C.4, trained on MNIST dataset using standard methods.

We set LReLU activations with α = 0.01 to test the proposed approximations.

We estimate the ground truth statistics µ * , σ * of all neurons by the Monte Carlo (MC) method: drawing 1000 samples of noise per input image and collecting samplebased statistics for each neuron.

Then we apply AP1 to compute µ 1 and AP2 to compute µ 2 and σ 2 for each unit from the clean input and known noise variance σ 2 0 .

The error measure of the means ε µ is the average |µ − µ * | relative to the average σ * .

The averages are taken over all units in the layer and over input images.

The error of the standard deviation ε σ is the geometric mean of σ/σ * , representing the error as a factor from the true value (e.g., 1.0 is exact, 0.9 is under-estimating and 1.1 is over-estimating).

Table 1 shows average errors per layer.

Our main observation is that AP2 is more accurate than AP1 but both methods suffer from the factorization assumption.

The variance computed by AP2 provides a good estimate and the estimated categorical distribution obtained by propagating the variance through softmax is much closer to the MC estimate.

Next, we study a widely used ALL-CNN network by BID28 trained with standard dropout on CIFAR-10.

Bernoulli dropout noise with dropout rate 0.2 is applied after each activation.

The accuracies of estimated statistics w.r.t.

dropout noises are shown in Table 2 .

Here, each layer receives uncertainty propagated from preceding layers, but also new noises are mixed-in in each layer, which works in favor of the factorization assumption.

The results are shown in Table 2 .

Observe that GT noise std σ * changes significantly across layers, up to 1-2 orders and AP2 gives a useful estimate.

Furthermore, having estimated the average factors suggests a simple calibration.

Calibration We divide the std in the last layer by the average factor σ/σ * estimated on the training set.

With this method, denoted AP2 calibrated, we get significantly better test likelihoods in the endto-end learning experiment.

The AP2 method can be used to approximate neuron statistics w.r.t.

the input chosen at random from the training dataset as was proposed by BID27 .

Instead of propagating sample instances, the method takes the dataset statistics (µ 0 , (σ 0 ) 2 ) and propagates them once through all network layers, averaging over spatial dimensions.

The obtained neuron mean and variance are then used to normalize the output the same way as in batch normalization BID13 .

This normalization leads to a better conditioned initialization and training and is batch-independent.

We verify the efficiency of this method for a network that includes the proposed approximations for LReLU and max pooling layers in § C.5 and use it in the end-to-end learning experiment below.

DISPLAYFORM0 Noisy input with noise std σ 0 = 10 Table 1 : Accuracy of approximation of mean and variance statistics for each layer in a fully trained LeNet5 (MNIST) tested with noisy input.

Observe the following: MC std σ * is growing significantly from the input to the output; both AP1 and AP2 have a significant drop of accuracy at linear (FC and Conv) layers, due to factorized approximation assumption; AP2 approximation of the standard deviation is within a factor close to one, and makes a meaningful estimate, although degrading with depth; AP2 approximation of the mean is more accurate than AP1; the KL divergence from the MC class posterior is improved with AP2.

Table 2 : Accuracy of approximation of mean and variance statistics for each layer in All-CNN (CIFAR-10) trained and tested with dropout.

The table shows accuracies after all layers (Cconvolution, A-activation, P-average pooling) and the final KL divergence.

A similar effect to propagating input noise is observed: the MC std σ * grows with depth; a significant drop of accuracy is observed in convolutional and pooling layers, which rely on the independence assumption.

DISPLAYFORM1

In this experiment we approximate the dropout analytically at training time similar to BID31 but including the new approximations for LReLU and softmax layers.

We compare training All-CNN network on CIFAR-10 without dropout, with standard dropout BID29 and analytic (AP2) dropout.

All three cases use exactly the same initialization, the AP2 normalization as discussed above and the same learning setup.

Only the learning rate is optimized individually per method § C.3.

Dropout layers with dropout rate 0.2 are applied after every activation.

FIG8 shows the progress of the three methods.

The analytic dropout is efficient as a regularizer (reduces overfitting in the validation likelihood), is non-stochastic and allows for faster learning than standard dropout.

While the latter slows the training down due to increased stochasticity of the gradient, the analytic dropout smoothes the loss function and speeds the training up.

This is especially visible on the training loss plot in FIG8 .

Furthermore, analytic dropout can be applied as the testtime inference method in a network trained with any variant of dropout.

Table 3 shows that AP2, calibrated as proposed above, achieves the best test likelihood, significantly improving SOA results for this network.

Differently from BID31 , we find that when trained with standard dropout, all test methods achieve approximately the same accuracy and only differ in likelihoods.

We also attempted comparison with other approaches.

Gaussian dropout BID29 performed similarly or slightly worse than Bernoulli dropout.

Variational dropout BID15 in our implementation for convolutional networks has diverged or has not improved over the nodropout baseline (we tried correlated and uncorrelated versions with or without local reparametrization trick and with different KL divergence factors 1, 0.1, 0.01, 0.001).

We have described uncertainty propagation method for approximate inference in probabilistic neural networks that takes into account all noises analytically.

Latent variable models allow a transparent interpretation of standard propagation in NNs as the simplest approximation and facilitate the devel- Table 3 : Results for All-CNN on CIFAR-10 test set: negative log likelihood (NLL) and accuracy.

Left: state of the art results for this network BID6 , table 3).

Middle: All-CNN trained with standard dropout (our learning schedule and analytic normalization) evaluated using different test-time methods.

Observe that "AP2 calibrated" well approximates dropout: the test likelihood is better than MC-100.

Right: All-CNN trained with analytic dropout (same schedule and normalization).

Observe that "AP2 calibrated" achieves the best likelihood and accuracy.opment of variance propagating approximations.

We proposed new such approximations allowing to handle max, argmax, softmax and log-softmax layers using latent variable models ( § 4 and § A.2).We measured the quality of the approximation of posterior in isolated layers and complete networks.

The accuracy is improved compared to standard propagation and is sufficient for several use cases such as estimating statistics over the dataset (normalization) and dropout training, where we report improved test likelihoods.

We identified the factorization assumption as the weakest point of the approximation.

While modeling of correlations is possible (e.g. BID22 , it is also more expensive.

We showed that a calibration of a cheap method can give a significant improvement and thus is a promising direction for further research.

Argmax and softmax may occur not only as the final layer but also inside the network, in models such as capsules BID25 or multiple hypothesis BID11 , etc.

Further applications of the developed technique may include generative and semi-supervised learning and Bayesian model estimation.

Approximation of the Mean For each k let A k ⊂ Ω denote the event that X k > X j ∀j, i.e. that X k is the maximum of all variables.

Let q k = P(A k ) be given.

Note that events {A k } k partition the probability space.

The expected value of the maximum Y = max k X k can be written as the following total expectation: DISPLAYFORM0 In order to compute each conditional expectation, we approximate the conditional density p(X k = x k | A k ), which is the marginal of the joint conditional density p(X = x | A k ), i.e. the distribution of X restricted to the part of the probability space A k as illustrated in FIG6 .

The approximation is a simpler conditional density p( DISPLAYFORM1 and the threshold m k is chosen to satisfy the proportionality: DISPLAYFORM2 DISPLAYFORM3 This can be also seen as the approximation of the conditional probability P(A k | X k = r) = j =k F Xj (r), as a function of r, with the indicator [[m k ≤ r]], i.e. the smooth step function given by the product of sigmoid-like functions F X k (r) with a sharp step function.

Assuming X k is logistic, we find DISPLAYFORM4 ).

Then the conditional expectation DISPLAYFORM5 where p S is the density of the standard Logistic distribution, a = DISPLAYFORM6 is the changed variable under the integral and H(q k ) = −q k log(q k ) − (1 − q k ) log(1 − q k ) is the entropy of a Bernoulli variable with probability q k .

This results in the following interesting formula for the mean: DISPLAYFORM7 Assuming X k is normal, we obtain the approximation DISPLAYFORM8 Figure A.1: The joint conditional density p(X 1 = x 1 , X 2 = x 2 | X 2 > X 1 ), its marginal density p(X 2 = x 2 | X 2 > X 1 ) and the approximation p(X 2 = x 2 | X 2 > m 2 ), all up to the same normalization factor P(X 2 > X 1 ).

Lemma A.1.

The approximationμ k is an upper bound on DISPLAYFORM9 Proof.

We need to show that DISPLAYFORM10 Let us subtract the integral over the common part A k ∩Â k .

It remains to show DISPLAYFORM11 In the RHS integral we have DISPLAYFORM12 The inequality (32) follows.

Corollary A.1.

The approximations of the expected maximum FORMULA2 , FORMULA3 are upper bounds in the respective cases when X k are logistic, resp., normal.

Consider the case that X k are i.i.d., all logistic or normal with µ k = 0 and σ k = 1.

We then have Approximation of the Variance For the variance we write DISPLAYFORM13 DISPLAYFORM14 where the approximation is due toÂ k , and further rewrite the expression as DISPLAYFORM15 2 p(x)dx expresses as 2 : DISPLAYFORM16 where Li 2 is dilogarithm.

The function f can be well approximated on [0, 1] with DISPLAYFORM17 where a = −1.33751 and b = 0.886763 are obtained from the first order Tailor expansion of S −1 (f (S(t))) at t = 0.

This approximation is shown in FIG0 and is in fact an upper bound on f .

We thus obtained a rather simple approximation for the variance DISPLAYFORM18 A.2 SMOOTH MAXIMUM -LOGSUMEXPIn variational Bayesian learning it is necessary to compute the expectation of log p(y|x, θ) w.r.t.

to random parameters θ.

The expectation of the logarithm rather that the logarithm of expectation originates from the variational lower bound on the marginal likelihood obtained with Jensen's inequality.

In this section we extend our approximations to also handle log p(y|x, θ) for classification problems.

The same propagation rules apply up to the difference that the last layer is log of softmax rather than softmax, i.e. DISPLAYFORM19 It remains therefore to handle the log-sum-exp operation, also known as the smooth maximum.

Proposition A.1.

The LogSumExp operation has the following latent variable representation: DISPLAYFORM20 where Γ k are independent Gumbel random variables such that E[Γ k ] = 0, i.e., Γ k ∼ Gumbel(−γ, 1), where γ is the Euler-Mascheroni constant. (x+γ) .

We can write the cdf of Y as DISPLAYFORM21 DISPLAYFORM22 where S = log k e x k .

It follows that the mean value of Y is S − γ + γ = S.We therefore propose to approximate the expectation of log softmax(X) by increasing the variances of all inputs X k by σ 2 S /2 and applying the approximation for the maximum (A.1).

Summarizing, we obtain the following.

Proposition A.2.

Let X i have statistics (µ i , σ 2 i ).

Then using expressions (39) and FORMULA2 , DISPLAYFORM23 where q k are the expected softmax values § 4.1.Two classes For p(y = 1|x) = 1/(1 + e −x ), we have log p(y = 1|x) = − log(1 + e −x ).

The analogue of Proposition A.1 is the latent variable expression DISPLAYFORM24 where Z is a standard Logistic r.v.

Therefore, to approximate E[log(1 + e X )], we can increase the variance of X by the variance of standard logistic distribution σ 2 S and apply the existing approximation for ReLU § 4.3.

We evaluate the simplified approximation of Leaky ReLU (25), which does not use the normal cdf function.

Since LReLU is 1-homogenous, it is clear that scaling the input will scale the output proportionally.

We therefore fix the input variance to 1 and plot the results as the function of the input mean µ. FIG6 shows that the approximation of the mean and variance as well as the approximation of the output distribution defined by these values are all reasonable.

FIG0 shows the implied approximation of derivatives.

The baseline for the derivatives is the MC estimate with pathwise derivative method (PD) BID8 , also known as the reparametrization trick.

This is also the method for the ground truth (with 10 5 samples).

Despite the approximation of the variance and its gradients are somewhat deviating from the GT model that assumes the perfect normal distribution on the input, the overall behavior of the function is similar to the desired one and it makes a cheap computational element for NNs.

To evaluate the proposed approximation for softmax we perform the following experiment.

We consider n = 10 inputs X 1 , . . .

, X n to be independent with X k ∼ N (µ k , σ 2 k ).

The means µ i are generated uniformly in the interval [0, U ].

Then we sample σ k such that log σ k is uniform in the interval [−5, 0] .

We then estimate the ground truth output class distribution q(y) = (softmax(X))

y by MC sampling using 10 5 samples of X and evaluate the KL divergence from this GT estimate to the approximations.

We test both: quadratic time approximation (13) as well as linear time approximation (17).

The evaluation is repeated with scaled variances, σ such that max k (σ k ) ranges from 10 −3 U to U , i.e., covers the practically relevant interval.

The experiment is repeated 1000 trials in which different samples of µ and σ are evaluated.

As a baseline we take the AP1 approximation and MC sampling using fewer samples (10 and 100).

This evaluation is shown in FIG8 .We further check how well the Jacobian is approximated.

The Jacobian has two parts: J µ y,k = ∂q(y)/∂µ k and J σ y,k = ∂q(y)/∂σ k .

For each part we compute the average cosine similarity of the gradients: DISPLAYFORM0 where J y denotes the gradient of output y in the part of the inputs (µ or σ).

The baseline AP1 obviously cannot estimate the gradient in σ.

This evaluation is shown in Fig. B .3 (middle, right).

The intervals around analytic estimates are due to the variance of the ground truth.

Towards smaller input variance the ground truth estimate of the gradient degrades to completely random and the scalar product with it approaches zero on average.

It does not imply that our analytic estimates are poor for small input variance.

This experiment is similar to Softmax but with several differences.

Unlike in softmax, the range of µ is not important (as there is no latent logistic noise with fixed variance added to the inputs).

We therefore can fix U = 1 because scaling both µ and σ is guaranteed to give the same output distribution.

Approximating the value of expected argmax indicator is shown in FIG9 .

In the baseline methods we include AP1, which computes y * = argmax k µ k and assigns the output probability q(y * ) = 1 and q(y) = 10 −20 for y = y * .

It is seen that the proposed approximations accurately model the expected value.

Computing the gradient with MC methods is more difficult in this case, since the pathwise derivative cannot be applied (because argmax indicator is not differentiable).

We therefore used the score function (SF) estimator BID4 , also known as REINFORCE method.

This method requires much more samples.

In fact we had problems to get a reliable ground truth even with as many as 10 7 samples.

In Fig. B .4(b,c) we illustrate the gradient estimation for a single random instance of µ, σ.

These plots show that baseline MC estimates have very high variance with 10 3 and 10 5 samples used.

The accuracy of the gradients with analytic method for small variances remains largely unmeasured because the GT estimate also degrades quickly and becomes close to random for small input noises.

A more accurate GT could be possibly computed by variance reduction techniques, in particular using our analytic estimates as (biased) baselines.

In this section we give all details necessary to ensure reproducibility of results.

We implemented our inference and learning in the pytorch 4 framework.

The source code will be publicly available.

The implementation is modular: with each of the standard layers we can do 3 kinds of propagation: AP1: standard propagation in deterministic layers and taking the mean in stochastic layers (e.g., in dropout we need to multiply by the Bernoulli probability), AP2: proposed propagation rules with variances and sample: by drawing samples of any encountered stochasticity (such as sampling from Bernoulli distribution in dropout).

The last method is also essential for computing Monte Carlo (MC) estimates of the statistics we want to approximate.

When the training method is sample, the test method is assumed to be AP1, which matches the standard practice of dropout training.

In the implementation of AP2 propagation the input and the output of each layer is a pair of mean and variance.

At present we use only higher-level pytorch functions to implement AP2 propagation.

The feed-forward propagation with AP2 is about 3 times slower than AP1 or sample.

The relative times of a forward-backward computation in our higher-level implementation are as follows: s t a n d a r d t r a i n i n g 1 BN 1 .

5 i n f e r e n c e =AP2 3 i n f e r e n c e =AP2−norm=AP2 6Please note that these times hold for unoptimized implementations.

In particular, the computational cost of the AP2 normalization, which propagates single pixel statistics, should be more efficient in comparison to propagating a batch of input images.

We used MNIST 5 and CIFAR10 6 datasets.

Both datasets provide a split into training and test sets.

From the training set we split 10 percent (at random) to create a validation set.

The validation set is meant for model selection and monitoring the validation loss and accuracy during learning.

The test sets were currently used only in the stability tests.

For the optimization we used batch size 32, SGD optimizer with Nesterov Momentum 0.9 (pytorch default) and the learning rate lr · γ k , where k is the epoch number, lr is the initial learning rate, γ is the decrease factor.

In all reported results for CIFAR we used γ such that γ 600 = 0.1 and 1200 epochs.

This is done in order to make sure we are not so much constrained by the performance of the optimization and all methods are given sufficient iterations to converge.

The initial learning rate was selected by an automatic numerical search optimizing the training loss in 5 epochs.

This is performed individually per training case to take care for the differences introduced by different initializations and training methods.

When not said otherwise, parameters of linear and convolutional layers were initialized using pytorch defaults, i.e., uniformly distributed in DISPLAYFORM0 where c is the number of inputs per one output.

Standard minor data augmentation was applied to the training and validation sets in CIFAR-10, consisting in random translations ±2 pixels (with zero padding) and horizontal flipping.

When we train with normalization, it is introduced after each convolutional and fully connected layer.

The LeNet5 architecture BID16 is:Conv2d ( 1 , 6 , k s =5 , s t = 2 ) , A c t i v a t i o n MaxPooling 5 http://yann.lecun.com/exdb/mnist/ 6 https://www.cs.toronto.edu/˜kriz/cifar.html Convolutional layer parameters list input channels, output channels, kernel size and stride.

DISPLAYFORM0 The All-CNN network BID28 has the following structure of convolutional layers:k s i z e = [ 3 , 3 , 3 , 3 , 3 , 3 , 3 , 1 , 1 ] s t r i d e = [ 1 , 1 , 2 , 1 , 1 , 2 , 1 , 1 , 1 ] d e p t h = [ 9 6 , 9 6 , 9 6 , 1 9 2 , 1 9 2 , 1 9 2 , 1 9 2 , 1 9 2 , 1 0 ] each but the last one ending with activation (we used LReLU).

The final layers of the network are ConvPool-CNN-C model replaces stride-2 convolutions by stride-1 convolutions of the same shape followed by 2x2 max pooling with stride 2.

We test the analytic normalization method BID27 in a network with max pooling and Leaky ReLU layers.

We consider the "ConvPool-CNN-C" model of BID28 on CIFAR-10 dataset.

It's structure is shown on the x-axis of FIG6 .

We first apply different initialization methods and compute variances in each layer over the training dataset.

FIG6 shows that standard initialization with weights distributed uniformly in [−1/ √ n in , 1/ √ n in ], where n in is the number of inputs per single output of a linear mapping results in the whole dataset concentrated around one output point with standard deviation 10 −5 .

Initialization of BID9 , using statistical arguments, improves this behavior.

For the analytic approximation, we take statistics of the dataset itself (µ 0 , σ 0 ) and propagate them through the network, ignoring spatial dimensions of the layers.

When normalized by this estimates, the real dataset statistics have variances close to one and means close to zero, i.e. the normalization is efficient.

For comparison, we also show normalization by the batch statistics with a batch of size 32.

FIG0 further demonstrates that the initialization is crucial for efficient learning, and that keeping track of the normalization during training and back propagating through it (denoted norm=AP2 in the figure) performs even better and may be preferable to batch normalization in many scenarios such as recurrent NNs.

The effect of initialization/normalization on the progress of training.

Observe that the initialization alone significantly influences the automatically chosen initial learning rate (lr) and the "trainability" of the network.

Using the normalization during the training further improves performance for both batch and analytic normalization.

BN has an additional regularization effect BID12 , the square markers in the left plot show BN training loss using averaged statistics.

Table C .1: Accuracy of approximation of mean and variance statistics for each layer in a fully trained ConvPool-CNN-C network with dropout.

A significant drop of accuracy is observed as well after max pooling, we believe due to the violation of the independence assumption.

C.6 ACCURACY WITH MAX POOLING Table C .1 shows accuracy of posterior approximation results for ConvPool-CNN-C, discussed above which includes max pooling layers.

The network is trained and evaluated on CIFAR-10 with dropout the same way as in § 5.1.

@highlight

Approximating mean and variance of the NN output over noisy input / dropout / uncertain parameters. Analytic approximations for argmax, softmax and max layers.

@highlight

The authors focus on the problem of uncertainty propagation DNN

@highlight

This paper revisits the feed-forward propagation of mean and variance in neurons, by addressing the problem of propagating uncertainty through max-pooling layers and softmax.