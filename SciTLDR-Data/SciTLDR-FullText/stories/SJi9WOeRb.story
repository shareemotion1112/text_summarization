Implicit models, which allow for the generation of samples but not for point-wise evaluation of probabilities, are omnipresent in real-world problems tackled by machine learning and a hot topic of current research.

Some examples include data simulators that are widely used in engineering and scientific research, generative adversarial networks (GANs) for image synthesis, and hot-off-the-press approximate inference techniques relying on implicit distributions.

The majority of existing approaches to learning implicit models rely on approximating the intractable distribution or optimisation objective for gradient-based optimisation, which is liable to produce inaccurate updates and thus poor models.

This paper alleviates the need for such approximations by proposing the \emph{Stein gradient estimator}, which directly estimates the score function of the implicitly defined distribution.

The efficacy of the proposed estimator is empirically demonstrated by examples that include meta-learning for approximate inference and entropy regularised GANs that provide improved sample diversity.

Modelling is fundamental to the success of technological innovations for artificial intelligence.

A powerful model learns a useful representation of the observations for a specified prediction task, and generalises to unknown instances that follow similar generative mechanics.

A well established area of machine learning research focuses on developing prescribed probabilistic models BID8 , where learning is based on evaluating the probability of observations under the model.

Implicit probabilistic models, on the other hand, are defined by a stochastic procedure that allows for direct generation of samples, but not for the evaluation of model probabilities.

These are omnipresent in scientific and engineering research involving data analysis, for instance ecology, climate science and geography, where simulators are used to fit real-world observations to produce forecasting results.

Within the machine learning community there is a recent interest in a specific type of implicit models, generative adversarial networks (GANs) BID10 , which has been shown to be one of the most successful approaches to image and text generation BID56 BID2 BID5 .

Very recently, implicit distributions have also been considered as approximate posterior distributions for Bayesian inference, e.g. see BID25 ; BID53 ; BID22 ; BID19 ; BID29 ; BID15 ; BID23 ; BID48 .

These examples demonstrate the superior flexibility of implicit models, which provide highly expressive means of modelling complex data structures.

Whilst prescribed probabilistic models can be learned by standard (approximate) maximum likelihood or Bayesian inference, implicit probabilistic models require substantially more severe approximations due to the intractability of the model distribution.

Many existing approaches first approximate the model distribution or optimisation objective function and then use those approximations to learn the associated parameters.

However, for any finite number of data points there exists an infinite number of functions, with arbitrarily diverse gradients, that can approximate perfectly the objective function at the training datapoints, and optimising such approximations can lead to unstable training and poor results.

Recent research on GANs, where the issue is highly prevalent, suggest that restricting the representational power of the discriminator is effective in stabilising training (e.g. see BID2 BID21 .

However, such restrictions often intro- A comparison between the two approximation schemes.

Since in practice the optimiser only visits finite number of locations in the parameter space, it can lead to over-fitting if the neural network based functional approximator is not carefully regularised, and therefore the curvature information of the approximated loss can be very different from that of the original loss (shown in (a)).

On the other hand, the gradient approximation scheme (b) can be more accurate since it only involves estimating the sensitivity of the loss function to the parameters in a local region.duce undesirable biases, responsible for problems such as mode collapse in the context of GANs, and the underestimation of uncertainty in variational inference methods BID49 .In this paper we explore approximating the derivative of the log density, known as the score function, as an alternative method for training implicit models.

An accurate approximation of the score function then allows the application of many well-studied algorithms, such as maximum likelihood, maximum entropy estimation, variational inference and gradient-based MCMC, to implicit models.

Concretely, our contributions include:• the Stein gradient estimator, a novel generalisation of the score matching gradient estimator BID16 , that includes both parametric and non-parametric forms; • a comparison of the proposed estimator with the score matching and the KDE plug-in estimators on performing gradient-free MCMC, meta-learning of approximate posterior samplers for Bayesian neural networks, and entropy based regularisation of GANs.

Given a dataset D containing i.i.d.

samples we would like to learn a probabilistic model p(x) for the underlying data distribution p D (x).

In the case of implicit models, p(x) is defined by a generative process.

For example, to generate images, one might define a generative model p(x) that consists of sampling randomly a latent variable z ∼ p 0 (z) and then defining x = f θ (z).

Here f is a function parametrised by θ, usually a deep neural network or a simulator.

We assume f to be differentiable w.r.t.

θ.

An extension to this scenario is presented by conditional implicit models, where the addition of a supervision signal y, such as an image label, allows us to define a conditional distribution p(x|y) implicitly by the transformation x = f θ (z, y).

A related methodology, wild variational inference BID25 BID22 ) assumes a tractable joint density p(x, z), but uses implicit proposal distributions to approximate an intractable exact posterior p(z|x).

Here the approximate posterior q(z|x) can likewise be represented by a deep neural network, but also by a truncated Markov chain, such as that given by Langevin dynamics with learnable step-size.

Whilst providing extreme flexibility and expressive power, the intractability of density evaluation also brings serious optimisation issues for implicit models.

This is because many learning algorithms, e.g. maximum likelihood estimation (MLE), rely on minimising a distance/divergence/discrepancy measure D[p||p D ], which often requires evaluating the model density (c.f.

BID33 BID25 .

Thus good approximations to the optimisation procedure are the key to learning implicit models that can describe complex data structure.

In the context of GANs, the Jensen-Shannon divergence is approximated by a variational lower-bound represented by a discriminator BID3 BID10 .

Related work for wild variational inference BID22 BID29 BID15 BID48 ) uses a GAN-based technique to construct a density ratio estimator for q/p 0 BID46 BID47 BID50 BID30 and then approximates the KL-divergence term in the variational lower-bound: DISPLAYFORM0 In addition, BID22 and BID29 exploit the additive structure of the KLdivergence and suggest discriminating between q and an auxiliary distribution that is close to q, making the density ratio estimation more accurate.

Nevertheless all these algorithms involve a minimax optimisation, and the current practice of gradient-based optimisation is notoriously unstable.

The stabilisation of GAN training is itself a recent trend of related research (e.g. see BID36 BID2 .

However, as the gradient-based optimisation only interacts with gradients, there is no need to use a discriminator if an accurate approximation to the intractable gradients could be obtained.

As an example, consider a variational inference task with the approximate pos- DISPLAYFORM1 Notice that the variational lower-bound can be rewritten as DISPLAYFORM2 the gradient of the variational parameters φ can be computed by a sum of the path gradient of the first term (i.e. DISPLAYFORM3 ) and the gradient of the entropy term DISPLAYFORM4 .

Expanding the latter, we have DISPLAYFORM5 in which the first term in the last line is zero BID35 .

As we typically assume the tractability of ∇ φ f , an accurate approximation to ∇ z log q(z|x) would remove the requirement of discriminators, speed-up the learning and obtain potentially a better model.

Many gradient approximation techniques exist BID44 BID9 BID57 BID7 , and in particular, in the next section we will review kernel-based methods such as kernel density estimation BID39 and score matching BID16 in more detail, and motivate the main contribution of the paper.

We propose the Stein gradient estimator as a novel generalisation of the score matching gradient estimator.

Before presenting it we first set-up the notation.

Column vectors and matrices are boldfaced.

The random variable under consideration is x ∈ X with X = R d×1 if not specifically mentioned.

To avoid misleading notation we use the distribution q(x) to derive the gradient approximations for general cases.

As Monte Carlo methods are heavily used for implicit models, in the rest of the paper we mainly consider approximating the gradient g( DISPLAYFORM0 We use x i j to denote the jth element of the ith sample x i .

We also denote the matrix form of the col- DISPLAYFORM1 T ∈ R K×d , and its approximation DISPLAYFORM2

We start from introducing Stein's identity that was first developed for Gaussian random variables BID42 BID43 then extended to general cases BID11 .

Let h : R d×1 → R d ×1 be a differentiable multivariate test function which maps x to a column vector DISPLAYFORM0 T .

We further assume the boundary condition for h: DISPLAYFORM1 This condition holds for almost any test function if q has sufficiently fast-decaying tails (e.g. Gaussian tails).

Now we introduce Stein's identity BID43 BID11 ) DISPLAYFORM2 in which the gradient matrix term DISPLAYFORM3 This identity can be proved using integration by parts: for the ith row of the matrix h(x)∇ x log q(x) T , we have DISPLAYFORM4 Observing that the gradient term ∇ x log q(x) of interest appears in Stein's identity (5), we propose the Stein gradient estimator by inverting Stein's identity.

As the expectation in (5) is intractable, we further approximate the above with Monte Carlo (MC): DISPLAYFORM5 with err ∈ R d ×d the random error due to MC approximation, which has mean 0 and vanishes as K → +∞. Now by temporarily denoting FORMULA14 can be rewritten as DISPLAYFORM6 DISPLAYFORM7 Thus we consider a ridge regression method (i.e. adding an 2 regulariser) to estimate G: DISPLAYFORM8 with || · || F the Frobenius norm of a matrix and η ≥ 0.

Simple calculation shows that DISPLAYFORM9 where DISPLAYFORM10 One can show that the RBF kernel satisfies Stein's identity .In this case h(x) = K(x, ·), d = +∞ and by the reproducing kernel property (Berlinet & ThomasAgnan, 2011) DISPLAYFORM11

In this section we derive the Stein gradient estimator again, but from a divergence/discrepancy minimisation perspective.

Stein's method also provides a tool for checking if two distributions q(x) andq(x) are identical.

If the test function set H is sufficiently rich, then one can define a Stein discrepancy measure by DISPLAYFORM0 see BID11 for an example derivation.

When H is defined as a unit ball in an RKHS induced by a kernel K(x, ·), and BID6 showed that the supremum in (10) can be analytically obtained as (with K xx shorthand for K(x, x )): DISPLAYFORM1 which is also named the kernelised Stein discrepancy (KSD).

BID6 showed that for C 0 -universal kernels satisfying the boundary condition, KSD is indeed a discrepancy measure: S 2 (q,q) = 0 ⇔ q =q. BID12 further characterised the power of KSD on detecting non-convergence cases.

Furthermore, if the kernel is twice differentiable, then using the same technique as to derive (16) one can compute KSD by FORMULA17 is equivalent to the V-statistic of KSD if h(x) = K(x, ·), and we have the following: DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 is the solution of the following KSD V-statistic minimisation problem DISPLAYFORM5 One can also minimise the U-statistic of KSD to obtain gradient approximations, and a full derivation of which, including the optimal solution, can be found in the appendix.

In experiments we use Vstatistic solutions and leave comparisons between these methods to future work.

There exist other gradient estimators that do not require explicit evaluations of ∇ x log q(x), e.g. the denoising auto-encoder (DAE) BID52 BID51 BID0 which, with infinitesimal noise, also provides an estimate of ∇ x log q(x) at convergence.

However, applying such gradient estimators result in a double-loop optimisation procedure since the gradient approximation is repeatedly required for fitting implicit distributions, which can be significantly slower than the proposed approach.

Therefore we focus on "quick and dirty" approximations and only include comparisons to kernel-based gradient estimators in the following.

A naive approach for gradient approximation would first estimate the intractable densityq(x) ≈ q(x) (up to a constant), then approximate the exact gradient by DISPLAYFORM0 , then differentiated through the KDE estimate to obtain the gradient estimator: DISPLAYFORM1 Interestingly for translation invariant kernels K(x, x ) = K(x − x ) the KDE gradient estimator (14) can be rewritten asĜ KDE = −diag (K1) −1 ∇, K .

Inspecting and comparing it with the Stein gradient estimator (9), one might notice that the Stein method uses the full kernel matrix as the pre-conditioner, while the KDE method computes an averaged "kernel similarity" for the denominator.

We conjecture that this difference is key to the superior performance of the Stein gradient estimator when compared to the KDE gradient estimator (see later experiments).

The KDE method only collects the similarity information between x k and other samples x j to form an estimate of ∇ x k log q(x k ), whereas for the Stein gradient estimator, the kernel similarity between x i and x j for all i, j = k are also incorporated.

Thus it is reasonable to conjecture that the Stein method can be more sample efficient, which also implies higher accuracy when the same number of samples are collected.

The KDE gradient estimator performs indirect approximation of the gradient via density estimation, which can be inaccurate.

An alternative approach directly approximates the gradient ∇ x log q(x) by minimising the expected 2 error w.r.t.

the approximationĝ( DISPLAYFORM0 It has been shown in BID16 that this objective can be reformulated as DISPLAYFORM1 The key insight here is again the usage of integration by parts: after expanding the 2 loss objective, the cross term can be rewritten as DISPLAYFORM2 , if assuming the boundary condition FORMULA10 forĝ (see FORMULA13 ).

The optimum of FORMULA0 is referred as the score matching gradient estimator.

The 2 objective (15) is also called Fisher divergence BID18 which is a special case of KSD (11) by selecting K(x, x ) = δ x=x .

Thus the Stein gradient estimator can be viewed as a generalisation of the score matching estimator.

The comparison between the two estimators is more complicated.

Certainly by the Cauchy-Schwarz inequality the Fisher divergence is stronger than KSD in terms of detecting convergence .

However it is difficult to perform direct gradient estimation by minimising the Fisher divergence, since (i) the Dirac kernel is non-differentiable so that it is impossible to rewrite the divergence in a similar form to (12), and (ii) the transformation to (16) involves computing ∇ xĝ (x).

So one needs to propose a parametric approximation to G and then optimise the associated parameters accordingly, and indeed BID38 and BID45 derived a parametric solution by first approximating the log density up to a constant as logq(x) := K k=1 a k K(x, x k ) + C, then minimising (16) to obtain the coefficientsâ score k and constructing the gradient estimator aŝ DISPLAYFORM3 Therefore the usage of parametric estimation can potentially remove the advantage of using a stronger divergence.

Conversely, the proposed Stein gradient estimator FORMULA18 is non-parametric in that it directly optimises over functions evaluated at locations {x k } K k=1 .

This brings in two key advantages over the score matching gradient estimator: (i) it removes the approximation error due to the use of restricted family of parametric approximations and thus can be potentially more accurate; (ii) it has a much simpler and ubiquitous form that applies to any kernel satisfying the boundary condition, whereas the score matching estimator requires tedious derivations for different kernels repeatedly (see appendix).In terms of computation speed, since in most of the cases the computation of the score matching gradient estimator also involves kernel matrix inversions, both estimators are of the same order of complexity, which is O(K 3 + K 2 d) (kernel matrix computation plus inversion).

Low-rank approximations such as the Nyström method BID40 BID55 ) can enable speed-up, but this is not investigated in the paper.

Again we note here that kernel-based gradient estimators can still be faster than e.g. the DAE estimator since no double-loop optimisation is required.

Certainly it is possible to apply early-stopping for the inner-loop DAE fitting.

However the resulting gradient approximation might be very poor, which leads to unstable training and poorly fitted implicit distributions.

Though providing potentially more accurate approximations, the non-parametric estimator (9) has no predictive power as described so far.

Crucially, many tasks in machine learning require predicting gradient functions at samples drawn from distributions other than q, for example, in MLE q(x) corresponds to the model distribution which is learned using samples from the data distribution instead.

To address this issue, we derive two predictive estimators, one generalised from the nonparametric estimator and the other minimises KSD using parametric approximations.

Predictions using the non-parametric estimator.

Let us consider an unseen datum y. If y is sampled from q, then one can also apply the non-parametric estimator (9) for gradient approximation, given the observed data X = {x 1 , ..., DISPLAYFORM0 then the non-parametric Stein gradient estimator computed on X ∪ {y} is ĝ(y) DISPLAYFORM1 with ∇ y K(·, y) denoting a K × d matrix with rows ∇ y K(x k , y), and ∇ y K(y, y) only differentiates through the second argument.

Then we demonstrate in the appendix that, by simple matrix calculations and assuming a translation invariant kernel, we have (with column vector 1 ∈ R K×1 ): DISPLAYFORM2 In practice one would store the computed gradientĜ Stein V , the kernel matrix inverse (K + ηI) −1 and η as the "parameters" of the predictive estimator.

For a new observation y ∼ p in general, one can "pretend" y is a sample from q and apply the above estimator as well.

The approximation quality depends on the similarity between q and p, and we conjecture here that this similarity measure, if can be described, is closely related to the KSD.Fitting a parametric estimator using KSD.

The non-parametric predictive estimator could be computationally demanding.

Setting aside the cost of fitting the "parameters", in prediction the time complexity for the non-parametric estimator is O(K 2 + Kd).

Also storing the "parameters" needs O(Kd) memory forĜ Stein V .

These costs make the non-parametric estimator undesirable for high-dimensional data, since in order to obtain accurate predictions it often requires K scaling with d as well.

To address this, one can also minimise the KSD using parametric approximations, in a similar way as to derive the score matching estimator in Section 3.3.2.

More precisely, we define a parametric approximation in a similar fashion as FORMULA0 , and in the appendix we show that if the RBF kernel is used for both the KSD and the parametric approximation, then the linear coefficients FIG0 T can be calculated analytically:â DISPLAYFORM3 with X the "gram matrix" that has elements X ij = (x i ) T x j .

Then for an unseen observation y ∼ p the gradient approximation returns ∇

y log q(y) ≈ (â DISPLAYFORM4 T ∇ y K(·, y).

In this case one only maintains the linear coefficientsâ Stein V and computes a linear combination in prediction, which takes O(K) memory and O(Kd) time and therefore is computationally cheaper than the non-parametric prediction model (27).

We present some case studies that apply the gradient estimators to implicit models.

Detailed settings (architecture, learning rate, etc.) are presented in the appendix.

Implementation is released at https://github.com/YingzhenLi/SteinGrad.

We first consider a simple synthetic example to demonstrate the accuracy of the proposed gradient estimator.

More precisely we consider the kernel induced Hamiltonian flow (not an exact sampler) BID45 on a 2-dimensional banana-shaped object: x ∼ B(x; b = 0.03, v = 100) ⇔ x 1 ∼ N (x 1 ; 0, v), x 2 = + b(x 2 1 − v), ∼ N ( ; 0, 1).

The approximate Hamiltonian flow is constructed using the same operator as in Hamiltonian Monte Carlo (HMC) BID31 , except that the exact score function ∇ x log B(x) is replaced by the approximate gradients.

We still use the exact target density to compute the rejection step as we mainly focus on testing the accuracy of the gradient estimators.

We test both versions of the predictive Stein gradient estimator (see section 3.4) since we require the particles of parallel chains to be independent with each other.

We fit the gradient estimators on K = 200 training datapoints from the target density.

The bandwidth of the RBF kernel is computed by the median heuristic and scaled up by a scalar between [1, 5].

All three methods are simulated for T = 2, 000 iterations, share the same initial locations that are constructed by target distribution samples plus Gaussian noises of standard deviation 2.0, and the results are averaged over 200 parallel chains.

We visualise the samples and some MCMC statistics in FIG1 .

In general all the resulting Hamiltonian flows are HMC-like, which give us the confidence that the gradient estimators extrapolate reasonably well at unseen locations.

However all of these methods have trouble exploring the extremes, because at those locations there are very few or even no training data-points.

Indeed we found it necessary to use large (but not too large) bandwidths, in order to both allow exploration of those extremes, and ensure that the corresponding test function is not too smooth.

In terms of quantitative metrics, the acceptance rates are reasonably high for all the gradient estimators, and the KSD estimates (across chains) as a measure of sample quality are also close to that computed on HMC samples.

The returned estimates of E[x 1 ] are close to zero which is the ground true value.

We found that the non-parametric Stein gradient estimator is more sensitive to hyper-parameters of the dynamics, e.g. the stepsize of each HMC step.

We believe a careful selection of the kernel (e.g. those with long tails) and a better search for the hyper-parameters (for both the kernel and the dynamics) can further improve the sample quality and the chain mixing time, but this is not investigated here.

One of the recent focuses on meta-learning has been on learning optimisers for training deep neural networks, e.g. see BID1 .

Could analogous goals be achieved for approximate inference?

In this section we attempt to learn an approximate posterior sampler for Bayesian neural networks (Bayesian NNs, BNNs) that generalises to unseen datasets and architectures.

A more detailed introduction of Bayesian neural networks is included in the appendix, and in a nutshell, we consider a binary classification task: p(y = 1|x, θ) = sigmoid(NN θ (x)), p 0 (θ) = N (θ; 0, I).

After observing the training data D = {(x n , y n )} N n=1 , we first obtain the approximate posterior DISPLAYFORM0 p(y n |x n , θ), then approximate the predictive distribution for a new observation as p(y DISPLAYFORM1 .

In this task we define an implicit approximate posterior distribution q φ (θ) as the following stochastic normalising flow (Rezende & Mohamed, 2015) θ t+1 = f (θ t , ∇ t , t ): given the current location θ t and the mini-batch data {(x m , y m )} M m=1 , the update for the next step is DISPLAYFORM2 The coordinates of the noise standard deviation σ φ (θ t , ∇ t ) and the moving direction ∆ φ (θ t , ∇ t ) are parametrised by a coordinate-wise neural network.

If properly trained, this neural network will learn the best combination of the current location and gradient information, and produce approximate posterior samples efficiently on different probabilistic modelling tasks.

Here we propose using the variational inference objective (2) computed on the samples {θ k t } to learn the variational parameters φ.

Since in this case the gradient of the log joint distribution can be computed analytically, we only approximate the gradient of the entropy term H[q] as in (3), with the exact score function replaced by the presented gradient estimators.

We report the results using the non-parametric Stein gradient estimator as we found it works better than the parametric version.

The RBF kernel is applied for gradient estimation, with the hyper-parameters determined by a grid search on the bandwidth σ 2 ∈ {0.25, 1.0, 4.0, 10.0, median trick} and η ∈ {0.1, 0.5, 1.0, 2.0}.We briefly describe the test protocol.

We take from the UCI repository BID24 six binary classification datasets (australian, breast, crabs, ionosphere, pima, sonar), train an approximate sampler on crabs with a small neural network that has one 20-unit hidden layer with ReLU activation, and generalise to the remaining datasets with a bigger network that has 50 hidden units and uses sigmoid activation.

We use ionosphere as the validation set to tune ζ.

The remaining 4 datasets are further split into 40% training subset for simulating samples from the approximate sampler, and 60% test subsets for evaluating the sampler's performance. ( BID54 ) evaluated on the test datasets directly.

In summary, SGLD returns best results in KSD metric.

The Stein approach performs equally well or a little better than SGLD in terms of test-LL and test error.

The KDE method is slightly worse and is close to MAP, indicating that the KDE estimator does not provide a very informative gradient for the entropy term.

Surprisingly the score matching estimator method produces considerably worse results (except for breast dataset), even after carefully tuning the bandwidth and the regularisation parameter η.

Future work should investigate the usage of advanced recurrent neural networks such as an LSTM BID14 , which is expected to return better performance.

GANs are notoriously difficult to train in practice.

Besides the instability of gradient-based minimax optimisation which has been partially addressed by many recent proposals BID36 BID2 BID5 , they also suffer from mode collapse.

We propose adding an entropy regulariser to the GAN generator loss.

Concretely, assume the generative model p θ (x) is implicitly defined by x = f θ (z), z ∼ p 0 (z), then the generator's loss is defined bỹ DISPLAYFORM0 where J gen (θ) is the original loss function for the generator from any GAN algorithm and α is a hyper-parameter.

In practice (the gradient of) FORMULA0 is estimated using Monte Carlo.

We empirically investigate the entropy regularisation idea on the very recently proposed boundary equilibrium GAN (BEGAN) BID5 ) method using (continuous) MNIST, and we refer to the appendix for the detailed mathematical set-up.

In this case the non-parametric V-statistic Stein gradient estimator is used.

We use a convolutional generative network and a convolutional auto-encoder and select the hyper-parameters of BEGAN γ ∈ {0.3, 0.5, 0.7}, α ∈ [0, 1] and λ = 0.001.

The Epanechnikov kernel K(x, x ) : DISPLAYFORM1 2 ) is used as the pixel values lie in a unit interval (see appendix for the expression of the score matching estimator), and to ensure the boundary condition we clip the pixel values into range [10 −8 , 1 − 10 −8 ].

The generated images are visualised in FIG3 .

BEGAN without the entropy regularisation fails to generate diverse samples even when trained with learning rate decay.

The other three images clearly demonstrate the benefit of the entropy regularisation technique, with the Stein approach obtaining the highest diversity without compromising visual quality.

We further consider four metrics to assess the trained models quantitatively.

First 500 samples are generated for each trained model, then we compute their nearest neighbours in the training set using l 1 distance, and obtain a probability vector p by averaging over these neighbour images' label vectors.

In FIG4 we depict the entropy of p (top left), averaged l 1 distances to the nearest neighbour (top right), and the difference between the largest and smallest elements in p (bottom right).

The error bars are obtained by 5 independent runs.

These results demonstrate that the Stein approach performs significantly better than the other two, in that it learns a better generative model not only faster but also in a more stable way.

Interestingly the KDE approach achieves the lowest average l 1 distance to nearest neighbours, possibly because it tends to memorise training examples.

We next train a fully connected network π(y|x) on MNIST that achieves 98.16% text accuracy, and compute on the generated images an empirical estimate of the inception score BID36 DISPLAYFORM2 .

High inception score indicates that the generate images tend to be both realistic looking and diverse, and again the Stein approach out-performs the others on this metric by a large margin.

Concerning computation speed, all the three methods are of the same order: 10.20s/epoch for KDE, 10.85s/epoch for Score, and 10.30s/epoch for Stein.

1 This is because K < d (in the experiments K = 100 and d = 784) so that the complexity terms are dominated by kernel computations (O(K 2 d)) required by all the three methods.

Also for a comparison, the original BEGAN method without entropy regularisation runs for 9.05s/epoch.

Therefore the main computation cost is dominated by the optimisation of the discriminator/generator, and the proposed entropy regularisation can be applied to many GAN frameworks with little computational burden.

We have presented the Stein gradient estimator as a novel generalisation to the score matching gradient estimator.

With a focus on learning implicit models, we have empirically demonstrated the efficacy of the proposed estimator by showing how it opens the door to a range of novel learning tasks: approximating gradient-free MCMC, meta-learning for approximate inference, and unsupervised learning for image generation.

Future work will expand the understanding of gradient estimators in both theoretical and practical aspects.

Theoretical development will compare both the V-statistic and U-statistic Stein gradient estimators and formalise consistency proofs.

Practical work will improve the sample efficiency of kernel estimators in high dimensions and develop fast yet accurate approximations to matrix inversion.

It is also interesting to investigate applications of gradient approximation methods to training implicit generative models without the help of discriminators.

Finally it remains an open question that how to generalise the Stein gradient estimator to non-kernel settings and discrete distributions.

In this section we provide more discussions and analytical solutions for the score matching estimator.

More specifically, we will derive the linear coefficient a = (a 1 , ..., a K ) for the case of the Epanechnikov kernel.

A.1 SOME REMARKS ON SCORE MATCHING Remark.

It has been shown in BID37 ; BID0 that de-noising autoencoders (DAEs) BID52 , once trained, can be used to compute the score function approximately.

Briefly speaking, a DAE learns to reconstruct a datum x from a corrupted input x = x+σ , ∼ N (0, I) by minimising the mean square error.

Then the optimal DAE can be used to approximate the score function as ∇ x log p(x) ≈ 1 σ 2 (DAE * (x)−x).

Sonderby et al. (2017) applied this idea to train an implicit model for image super-resolution, providing some promising results in some metrics.

However applying similar ideas to variational inference can be computationally expensive, because the estimation of ∇ z log q(z|x) is a sub-routine for VI which is repeatedly required.

Therefore in the paper we deploy kernel machines that allow analytical solutions to the score matching estimator in order to avoid double loop optimisation.

Remark.

As a side note, score matching can also be used to learn the parameters of an unnormalised density.

In this case the target distribution q would be the data distribution andq is often a Boltzmann distribution with intractable partition function.

As a parameter estimation technique, score matching is also related to contrastive divergence BID13 , pseudo likelihood estimation BID17 , and DAEs BID51 BID0 .

Generalisations of score matching methods are also presented in e.g. BID27 BID28 .

The derivations for the RBF kernel case is referred to BID45 , and for completeness we include the final solutions here.

Assume the parametric approximation is defined as logq(x) = K k=1 a k K(x, x k ) + C, where the RBF kernel uses bandwidth parameter σ.

then the optimal solution of the coefficientsâ score = (Σ + ηI) DISPLAYFORM0

The Epanechnikov kernel is defined as DISPLAYFORM0 , where the first and second order gradients w.r.t.

DISPLAYFORM1 Thus the score matching objective with logq( DISPLAYFORM2 with the matrix elements DISPLAYFORM3 Define the "gram matrix" X ij = (x i ) T x j , we write the matrix form of Σ as DISPLAYFORM4 Thus with an l 2 regulariser, the fitted coefficients arê DISPLAYFORM5 1.

The V-statistic of KSD is the following: given samples x k ∼ q, k = 1, ..., K and recall DISPLAYFORM0 (23) The last term ∇ x j ,x l K jl will be ignored as it does not depend on the approximationĝ.

Using matrix notations defined in the main text, readers can verify that the V-statistic can be computed as DISPLAYFORM1 Using the cyclic invariance of matrix trace leads to the desired result in the main text.

The U-statistic of KSD removes terms indexed by j = l in (23), in which the matrix form is DISPLAYFORM2 with the jth row of ∇diag(K) defined as ∇ x j K(x j , x j ).

For most translation invariant kernels this extra term ∇diag(K) = 0, thus the optimal solution ofĜ by minimising KSD U-statistic iŝ DISPLAYFORM3

Let us consider an unseen datum y. If y is sampled from the q distribution, then one can also apply the non-parametric estimator (9) for gradient approximations, given the observed data X = {x 1 , ..., x K } ∼ q. Concretely, if writingĝ(y) ≈ ∇

y log q(y) ∈ R d×1 then the non-parametric Stein gradient estimator (using V-statistic) is DISPLAYFORM0 with ∇ y K(·, y) denoting a K ×d matrix with rows ∇ y K(x k , y), and ∇ y K(y, y) only differentiates through the second argument.

Thus by simple matrix calculations, we have: DISPLAYFORM1 For translation invariant kernels, typically ∇ y K(y, y) = 0, and more conveniently, DISPLAYFORM2 Thus equation FORMULA2 can be further simplified to (with column vector 1 ∈ R K×1 ) ∇ y log q(y) DISPLAYFORM3 The solution for the U-statistic case can be derived accordingly which we omit here.

We define a parametric approximation in a similar way as for the score matching estimator: DISPLAYFORM0 Now we show the optimal solution of a = (a 1 , ..., a K ) T by minimising (23).

To simplify derivations we assume the approximation and KSD use the same kernel.

First note that the gradient of the RBF kernel is DISPLAYFORM1 Substituting FORMULA5 into FORMULA2 : DISPLAYFORM2 DISPLAYFORM3 We first consider summing the j, l indices in ♣.

Recall the "gram matrix" X ij = (x i ) T x j , the inner product term in ♣ can be expressed as X kk + X jl − X kl − X jk .

Thus the summation over j, l can be re-written as DISPLAYFORM4 And thus ♣ = 1 σ 4 a T Λa.

Similarly the summation over j, l in ♠ can be simplified into Similarly we can derive the solution for KSD U-statistic minimisation.

The U statistic can also be represented in quadratic form S 2 U (q,q) = C +♣ + 2♠, with♠ = ♠ and DISPLAYFORM5 DISPLAYFORM6 Summing over the j indices for the second term, we have DISPLAYFORM7 Working through the analogous derivations reveals thatâ DISPLAYFORM8

We describe the detailed experimental set-up in this section.

All experiments use Adam optimiser BID20 with standard parameter settings.

We start by reviewing Bayesian neural networks with binary classification as a running example.

In this task, a normal deep neural network is constructed to predict y = f θ (x), and the neural network is parameterised by a set of weights (and bias vectors which we omit here for simplicity DISPLAYFORM0 .

In the Bayesian framework these network weights are treated as random variables, and a prior distribution, e.g. Gaussian, is also attached to them: p 0 (θ) = N (θ; 0, I).

The likelihood function of θ is then defined as DISPLAYFORM1 and p(y = 0|x, θ) = 1 − p(y = 1|x, θ) accordingly.

One can show that the usage of Bernoulli distribution here corresponds to applying cross entropy loss for training.

After framing the deep neural network as a probabilistic model, a Bayesian approach would find the posterior of the network weights p(θ|D) and use the uncertainty information encoded in it for future predictions.

By Bayes' rule, the exact posterior is DISPLAYFORM2 p(y n |x n , θ), and the predictive distribution for a new input x * is DISPLAYFORM3 Again the exact posterior is intractable, and approximate inference would fit an approximate posterior distribution q φ (θ) parameterised by the variational parameters φ to the exact posterior, and then use it to compute the (approximate) predictive distribution.p(y * = 1|x * , D) ≈ p(y * = 1|x * , θ)q φ (θ)dθ.

Since in practice analytical integration for neural network weights is also intractable, the predictive distribution is further approximated by Monte Carlo: DISPLAYFORM4 Now it remains to fit the approximate posterior q φ (θ), and in the experiment the approximate posterior is implicitly constructed by a stochastic flow.

For the training task, we use a one hidden layer neural network with 20 hidden units to compute the noise variance and the moving direction of the next update.

In a nutshell it takes the ith coordinate of the current position and the gradient θ t (i), ∇ t (i) as the inputs, and output the corresponding coordinate of the moving direction ∆ φ (θ t , ∇ t )(i) and the noise variance σ φ (θ t , ∇ t )(i).

Softplus non-linearity is used for the hidden layer and to compute the noise variance we apply ReLU activation to ensure non-negativity.

The step-size ζ is selected as 1e-5 which is tuned on the KDE approach.

For SGLD step-size 1e-5 also returns overall good results.

The training process is the following.

We simulate the approximate sampler for 10 transitions and sum over the variational lower-bounds computed on the samples of every step.

Concretely, the maximisation objective is DISPLAYFORM5 where T = 100 and q t (θ) is implicitly defined by the marginal distribution of θ t that is dependent on φ.

In practice the variational lower-bound L VI (q t ) is further approximated by Monte Carlo and data sub-sampling: DISPLAYFORM6 strategies are considered to approximate the contribution of the entropy term.

Given K samples x 1 , ..., x k ∼ p θ (x), The first proposal considers a plug-in estimate of the entropy term with a KDE estimate of p θ (x), which is consistent with the KDE estimator but not necessary with the other two (as they use kernels when representing log p θ (x) or ∇ x log p θ (x)).

The second one uses a proxy of the entropy loss −H[p]

≈ 1 K K k=1 ∇ x k log p θ (x k ) T x k with generated samples {x k } and ∇ x k log p θ (x k ) approximated by the gradient estimator in use.

In the experiment, we construct a deconvolutional net for the generator and a convolutional autoencoder for the discriminator.

The convolutional encoder consists of 3 convolutional layers with filter width 3, stride 2, and number of feature maps [32, 64, 64] .

These convolutional layers are followed by two fully connected layers with [512, 64] units.

The decoder and the generative net have a symmetric architecture but with stride convolutions replaced by deconvolutions.

ReLU activation function is used for all layers except the last layer of the generator, which uses sigmoid non-linearity.

The reconstruction loss in use is the squared 2 norm || · || 2 2 .

The randomness p 0 (z) is selected as uniform distribution in [-1, 1] as suggested in the original paper BID5 .

The minibatch size is set to K = 100.

Learning rate is initialised at 0.0002 and decayed by 0.9 every 10 epochs, which is tuned on the KDE model.

The selected γ and α values are: for KDE estimator approach γ = 0.3, αγ = 0.05, for score matching estimator approach γ = 0.3, αγ = 0.1, and for Stein approach γ = 0.5 and αγ = 0.3.

The presented results use the KDE plug-in estimator for the entropy estimates (used to tune β) for the KDE and score matching approaches.

Initial experiments found that for the Stein approach, using the KDE entropy estimator works slightly worse than the proxy loss, thus we report results using the proxy loss.

An advantage of using the proxy loss is that it directly relates to the approximate gradient.

Furthermore we empirically observe that the performance of the Stein approach is much more robust to the selection of γ and α when compared to the other two methods.

@highlight

We introduced a novel gradient estimator using Stein's method, and compared with other methods on learning implicit models for approximate inference and image generation.