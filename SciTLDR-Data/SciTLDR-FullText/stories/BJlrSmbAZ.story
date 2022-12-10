Deep neural networks have led to a series of breakthroughs, dramatically improving the state-of-the-art in many domains.

The techniques driving these advances, however, lack a formal method to account for model uncertainty.

While the Bayesian approach to learning provides a solid theoretical framework to handle uncertainty, inference in Bayesian-inspired deep neural networks is difficult.

In this paper, we provide a practical approach to Bayesian learning that relies on a regularization technique found in nearly every modern network, batch normalization.

We show that training a deep network using batch normalization is equivalent to approximate inference in Bayesian models, and we demonstrate how this finding allows us to make useful estimates of the model uncertainty.

Using our approach, it is possible to make meaningful uncertainty estimates using conventional architectures without modifying the network or the training procedure.

Our approach is thoroughly validated in a series of empirical experiments on different tasks and using various measures, showing it to outperform baselines on a majority of datasets with strong statistical significance.

Deep learning has dramatically advanced the state of the art in a number of domains, and now surpasses human-level performance for certain tasks such as recognizing the contents of an image BID10 and playing Go (Silver et al., 2017) .

But, despite their unprecedented discriminative power, deep networks are prone to make mistakes.

Sometimes, the consequences of mistakes are minor -misidentifying a food dish or a species of flower (Liu et al., 2016) may not be life threatening.

But deep networks can already be found in settings where errors carry serious repercussions such as autonomous vehicles BID2 and high frequency trading.

In medicine, we can soon expect automated systems to screen for skin cancer BID4 , breast cancer (Shen, 2017) , and to diagnose biopsies BID3 .

As autonomous systems based on deep learning are increasingly deployed in settings with the potential to cause physical or economic harm, we need to develop a better understanding of when we can be confident in the estimates produced by deep networks, and when we should be less certain.

Standard deep learning techniques used for supervised learning lack methods to account for uncertainty in the model, although sometimes the classification network's output vector is mistakenly understood to represent the model's uncertainty.

The lack of a confidence measure can be especially problematic when the network encounters conditions it was not exposed to during training.

For example, if a network trained to recognize dog breeds is given an image of a cat, it may predict it to belong to a breed of small dog with high probability.

When exposed to data outside of the distribution it was trained on, the network is forced to extrapolate, which can lead to unpredictable behavior.

In such cases, if the network can provide information about its uncertainty in addition to its point estimate, disaster may be avoided.

This work focuses on estimating such predictive uncertainties in deep networks (Figure 1 ).The Bayesian approach provides a solid theoretical framework for modeling uncertainty BID7 , which has prompted several attempts to extend neural networks (NN) into a Bayesian setting.

Most notably, Bayesian neural networks (BNNs) have been studied since the 1990's (Neal, 2012) .

Although they are simple to formulate, BNNs require substantially more computational resources than their non-Bayesian counterparts, and inference is difficult.

Importantly, BNNs do 2 RELATED WORK Bayesian models provide a natural framework for modeling uncertainty, and several approaches have been developed to adapt NNs to Bayesian reasoning.

A common approach is to place a prior distribution (often a Gaussian) over each weight.

For infinite weights, the resulting model corresponds to a Gaussian process (Neal, 1995) , and for a finite number of weights it corresponds to a Bayesian neural network (MacKay, 1992) .

Although simple to formulate, inference in BNNs is difficult BID5 .

Therefore, focus has shifted to techniques to approximate the posterior distribution, leading to approximate BNNs.

Methods based on variational inference (VI) typically rely on a fully factorized approximate distribution (Kingma & Welling, 2014; Hinton & Van Camp, 1993) but these methods do not scale easily.

To alleviate these difficulties, BID9 proposed a model using sampling methods to estimate a factorized posterior.

Another approach, probabilistic backpropagation (PBP), also estimates a factorized posterior based on expectation propagation (Hernández-Lobato & Adams, 2015) .Deep Gaussian Processes (DGPs) formulate GPs as Bayesian models capable of working on large datasets with the aid of a number of strategies to address scaling and complexity requirements BID1 .

The authors compare DGP with a number of state-of-the-art approximate BNNs, showing superior performance in terms of RMSE and uncertainty quality 2 .

Another recent approach to Bayesian learning, Bayesian hypernetworks, use a neural network to learn a distribution of paramaters over another neural network (Krueger et al., 2017) .

Although these recent techniques address some of the difficulties with approximate BNNs, they all require modifications to the architecture or the way networks are trained, as well as specialized knowledge from practitioners.

Recently, BID5 showed that a network trained with dropout implicitly performs the VI objective.

Therefore any network trained with dropout can be treated as an approx.

Bayesian model by making multiple predictions as forward passes through the network while sampling different dropout masks for each prediction.

An estimate of the posterior can be obtained by computing the mean and variance of the predictions.

This technique, referred to here as MCDO, has been empirically demonstrated to be competitive with other approx.

BNN methods and DGPs in terms of RMSE and uncertainty quality (Li & Gal, 2017) .

However, as the name implies, MCDO depends on dropout.

While once ubiquitous in training deep learning models, dropout has largely been replaced by batch normalization in modern networks, limiting its usefulness.

The methodology of this work is to pose a deep network trained with batch normalization as a Bayesian model in order to obtain uncertainty estimates associated with its predictions.

In the following, we briefly introduce Bayesian models and a variational approximation to it using KullbackLeibler (KL) divergence following BID6 .

We continue by showing a batch normalized deep network can be seen as an approximate Bayesian model.

Then, by employing theoretical insights as well as empirical analysis, we study the induced prior on the parameters when using batch normalization.

Finally, we describe the procedure we use for estimating uncertainty of batch normalized deep networks' output.

We assume a finite training set D = {(x i , y i )} i=1:N where each (x i , y i ) is a sample-label pair.

Using D, we are interested in learning an inference function f ω (x, y) with parameters ω.

In deterministic models, the estimated labelŷ is obtained as follows: DISPLAYFORM0 We assume f ω (x, y) = p(y|x, ω) (e.g. in soft-max classifiers), and is normalized to a proper probability distribution.

In Bayesian modeling, in contrast to finding a point estimate of the model parameters, the idea is to estimate an (approximate) posterior distribution of the model parameters p(ω|D) to be used for probabilistic prediction: DISPLAYFORM1 The predicted label,ŷ, can then be accordingly obtained by sampling p(y|x, D) or takings its maxima.

Variational Approximation In approximate Bayesian modeling, it is a common approach to learn a parametrized approximating distribution q θ (ω) that minimizes KL(q θ (ω)||p(ω|D)); the Kullback-Leibler (KL) divergence of posterior w.r.t.

its approximation, instead of the true posterior.

Minimizing this KL divergence is equivalent to the following minimization while being free of the data term p(D) 3 : DISPLAYFORM2 Using Monte Carlo integration to approximate the integral with one realizedω i for each sample i 4 , and optimizing over mini-batches of size M , the approximated objective becomes: DISPLAYFORM3 The first term is the data likelihood and the second term is divergence of the model prior w.r.t.

the approximated distribution.

We now describe the optimization procedure of a deep network with batch normalization and draw the resemblance to the approximate Bayesian modeling in Eq (1).

The inference function of a feed-forward deep network with L layers can be described as: DISPLAYFORM0 where a(.) is an element-wise nonlinearity function and W l is the weight vector at layer l. Furthermore, we denote the input to layer l as x l with x 1 = x and we then set h l = W l x l .

Parenthesized super-index for matrices (e.g. W (j) ) and vectors (e.g. x (j) ) indicates jth row and element respectively.

Super-index u refers to a specific unit at layer l, (e.g. DISPLAYFORM1 Batch Normalization Each layer of a deep network is constructed by several linear units whose parameters are the rows of the weight matrix W. Batch normalization is a unit-wise operation proposed in Ioffe & Szegedy (2015) to standardize the distribution of each unit's input.

It essentially converts a unit's output h u in the following way: DISPLAYFORM2 where the expectations are computed over the training set 6 .

However, often in deep networks, the weight matrices are optimized using back-propagated errors calculated on mini-batches of data.

Therefore, during training, the estimated mean and variance on the mini-batch B is used, which we denote by µ B and σ B respectively.

This makes the inference at training time for a sample x a stochastic process, varying based on other samples in the mini-batch.3 achieved by constructing the Evidence Lower Bound, called ELBO, and assuming i.i.d.

observation noise; details can be found in the appendix sec 6.1.4 while a MC integration using a single sample is a weak approximation, in an iterative optimization for θ several samples will be taken over time.

5 For a (softmax) classification network, fω(x) is a vector with fω(x, y) = fω(x) (y) , for regression networks with i.i.d.

Gaussian noise we have fω(x, y) = N (fω(x), τ −1 I).

6 It further learns an affine transformation for each unit using parameters γ and β, which we omit in favor of brevity:x DISPLAYFORM3 Loss Function and Optimization Training deep networks with mini-batch optimization involves a (regularized) risk minimization with the following form: DISPLAYFORM4 Where the first term is the empirical loss on the training data and the second term is a regularization penalty acting as a prior on model parameters ω.

If the loss l is cross-entropy for classification or sum-of-squares for regression problems (assuming i.i.d.

Gaussian noise on labels), the first term is equivalent to minimizing the negative log-likelihood: DISPLAYFORM5 with τ = 1 for classification.

In a batch normalized network the model parameters are DISPLAYFORM6 B }, we get the following objective at each step of the mini-batch optimization of a batch normalized network: DISPLAYFORM7 whereω i is the mean and variances for sample i's mini-batch at a certain training step.

Note that whileω i formally needs to be i.i.d.

for each training example, a batch normalized network samples the stochastic parameters once per training step (mini-batch).

For a large number of epochs, however, the distribution of sampled batch members for a given training example converges to the i.i.d.

case.

Comparing Eq.(1) and Eq. (2) reveals that the optimization objectives are identical, if there exists a prior p(ω) corresponding to Ω(θ) such that DISPLAYFORM8 In a batch normalized network, q θ (ω) corresponds to the joint distribution of the normalization parameters µ 1:L B , σ 1:L B , as implied by the repeated sampling from D during training.

This is an approximation of the true posterior, where we have restricted the posterior to lie within the domain of our parametric network and source of randomness.

With that we can use a pre-trained batch normalized network to estimate the uncertainty of its prediction using the inherent stochasticity of BN.

Before that, we briefly discuss what Bayesian prior is induced in a typical batch normalized network.

The purpose of Ω(θ) is to reduce variance in deep networks.

L2-regularization, also referred to as weight decay ( DISPLAYFORM0 , is a popular technique in deep learning.

The induced prior from L2-regularization is studied in Appendix 6.5.

Under some approximations as outlined in the Appendix, we find that BN for a deep network with FC layers and ReLU activations induce Gaussian distributions over BN unit's means and standard deviations, centered around the population values given by D (Eq. (6), details in Appendix 6.3).

Factorizing this distribution across all stochastic parameters and assuming Gaussian priors, we find the approximate corresponding priors: DISPLAYFORM1 where J l−1 is the dimensionality of the layer's inputs and x is the average input over D for all input units.

In the absence of scale and shift transformations from the previous BN layer, it converges towards an exact prior for large training datasets and deep networks (under the assumptions of the factorized distribution).

The mean and variance for the BN unit's standard deviation, µ p and σ 2 p , have no relevance for the reconciliation of the optimization objectives of Eq. (1) and (2).

In the absence of the true posterior we rely on the approximate posterior to express an approximate predictive distribution: DISPLAYFORM0 Following BID6 we estimate the first and second moment of the predictive distribution empirically (see Appendix 6.4 for details).

For regression, the first two moments are: DISPLAYFORM1 where eachω i corresponds to sampling the net's stochastic parameters ω = {µ DISPLAYFORM2 B } the same way as during training.

Samplingω i therefore involves sampling a batch B from the training set and updating the parameters in the BN units, just as if we were taking a training step with B. Recall that from a VA perspective, training the network amounted to minimizing KL(q θ (ω)||p(ω|D)) wrt θ.

Samplingω i from the training set, and keeping the size of B consistent with the mini-batch size used during training, ensures that q θ (ω) during inference remains identical to the approximate posterior optimized during training.

After each update of the net's stochastic parameters, we take a forward pass with input x, producing output fω i (x).

After T such stochastic forward passes, we compute the mean and sample variance of outputs to find the mean E p * [y] and variance Cov p * [y] of the approximate predictive distribution.

Note that Cov p * [y] also requires addition of constant variance from observation noise, τ −1 I.The network is trained just as a regular BN network.

The difference is in using the trained network for prediction.

Instead of replacing ω = {µ DISPLAYFORM3 with population values from D, we update these parameters stochastically, once for each forward pass .

The form of p * can be approximated by a Gassuian for each output dimension (for regression).

We assume bounded domains for each input dimension, wide layers throughout the network, and a unimodal distribution of weights centered at 0.

By the Liapounov CLT condition, the first layer then receives approximately Gaussian inputs (a proof can be found in Lehmann (1999) ).

Having sampled µ u B and σ u B from a mini-batch, each BN unit's output is bounded.

CLT thereby continues to hold for deeper layers, including DISPLAYFORM0 A similar motivation for a Gaussian approximation of Dropout has been presented by Wang & Manning (2013) .The actual form of p * is likely to be highly multimodal, as can be seen immediately from DISPLAYFORM1 with elements in x L normalized, scaled and shifted differently.

BID6 note the multimodality as well, since MCDO implies a bimodal variational distribution over each weight matrix column.

We assess the uncertainty quality of MCBN quantitatively and qualitatively.

Our quantitative analysis relies on eight standard regression datasets, listed in Table 1 .

Publicly available from the UCI Machine Learning Repository (University of California, 2017) and Delve (Ghahramani, 1996) , these datasets have been used to benchmark comparative models in recent related literature (see Hernández-Lobato & Adams (2015) , BID6 , BID1 and Li & Gal (2017) ).

We report results using standard metrics, and also propose useful upper and lower bounds to normalize these metrics for a more meaningful interpretation in Section 4.2.Under review as a conference paper at ICLR 2018 Table 1 : Properties of the eight regression datasets used to evaluate MCBN.

N is the dataset size and Q is the n.o.

input features.

Only one target feature was used.

In cases where the raw datasets contain more than one target feature, the feature used is specified by target feature.

Our qualitative results consist of three parts.

First, in Figure 1 we demonstrate that MCBN produces reasonable uncertainty bounds on a toy dataset in the style of (Karpathy, 2015) .

Second, we develop a new visualization of uncertainty quality by plotting test errors sorted by predicted variance in FIG0 .

Finally, we apply MCBN to SegNet BID0 , demonstrating the benefits of MCBN in an existing batch normalized network.

We evaluate uncertainty quality based on two metrics, described below: Predictive Log Likelihood (PLL) and Continuous Ranked Probability Score (CRPS).

We also propose upper and lower bounds for these metrics which can be used to normalize them and provide a more meaningful interpretation.

Predictive Log Likelihood (PLL) Predictive Log Likelihood is a widely accepted metric for uncertainty quality, used as the main uncertainty quality metric for regression (e.g. (Hernández-Lobato & Adams, 2015) , BID6 , BID1 and (Li & Gal, 2017) ).

A key property is that PLL makes no assumtions about the form of the distribution.

The measure is defined for a probabilistic model f ω (x) and a single observation (y i , x i ) as: DISPLAYFORM0 where p(y i |f ω (x i )) is the model's predicted PDF evaluated at y i , given the input x i .

A more detailed description is given in Appendix 6.4.

The metric is unbounded and maximized by a perfect prediction (mode at y i ) with no variance.

As the predictive mode moves away from y i , increasing the variance tends to increase PLL (by maximizing probability mass at y i ).

While PLL is an elegant measure, it has been criticized for allowing outliers to have an overly negative effect on the score (Selten, 1998).Continuous Ranked Probability Score (CRPS) Continuous Ranked Probability Score is a less sensitive measure that takes the full predicted PDF into account.

A prediction with low variance that is slightly offset from the true observation will receive a higher score form CRPS than PLL.

In order for CRPS to be analytically tractable, we need to assume a Gaussian unimodal predictive distribution.

CRPS is defined as DISPLAYFORM1 where F (y) is the predictive CDF, and 1(y ≥ y i ) = 1 if y ≥ y i and 0 otherwise (for univariate distributions) BID8 .

CRPS is interpreted as the sum of the squared area between the CDF and 0 where y < y i and between the CDF and 1 where y ≥ y i .

A perfect prediction with no variance yields a CRPS of 0; for all other cases the value is larger.

CRPS has no upper bound.

In order to establish a lower bound on useful performance for uncertainty estimates, we define a baseline that predicts constant variance regardless of input.

This benchmark model produces identical point estimates as MCBN, which yield the same predictive means.

The variance is set to a fixed value that optimizes CRPS on validation data.

This model reflects our best guess of constant variance on test data -any improvement in uncertainty quality from MCBN would indicate a sensible estimate of uncertainty.

We call this model Constant Uncertainty BN (CUBN).

Implementing MCDO as a comparative model, we similarly define a baseline for dropout, Constant Uncertainty Dropout (CUDO).

The difference in variance modeling between MCBN, CUBN, MCDO and CUDO are visualized in plots of uncertainty bounds on toy data in Figure 1 .For a probabilistic model f , an upper bound on uncertainty performance can also be defined for CRPS and PLL.

For each observation (y i , x i ), a value for the predictive variance T i can be chosen that maximizes PLL or minimizes CRPS 8 .

Using CUBN as a lower bound and the optimized CRPS score as the upper bound, uncertainty estimates can be normalized between these bounds (1 indicating optimal performance, and 0 indicating performance on par with fixed uncertainty).

We call this normalized measure CRPS = DISPLAYFORM0 This normalized measure gives an intuitive understanding of how close a Bayesian model is to estimating the perfect uncertainty for each prediction.

We also evaluate CRPS and PLL for an adaptation of the authors' implementation of Multiplicative Normalizing Flows (MNF) for variational Bayesian networks (Louizos & Welling, 2017) .

This is a recent model specialized to allow a more flexible posterior what is achievable by e.g. MCDO's bimodal variational over weight columns.

MNF uses auxillary variables on which the posterior is a latent.

By applying normalizing flows to the auxillary variable such that it can take on complex distributions, the approximate posterior becomes highly flexible.

Our evaluation of MCBN and MCDO is largely comparable to that of Hernández-Lobato & Adams (2015) , in that we use similar datasets and metrics.

This setup was later also followed by BID6 , where we in comparison implement a different hyperparameter selection, allow for a larger range of dropout rates, and use larger networks with two hidden layers.

With the exception of Protein Tertiary Structure 9 , all our models share a similar architecture: two hidden layers with 50 units each, using ReLU activations.

Input and output data were normalized during training.

Results were averaged over five random splits of 20% test and 80% training and cross-validation (CV) data.

For each split, 5-fold CV by grid search with a RMSE minimization objective was used to find training hyperparameters and optimal n.o.

epochs.

For BN-based models, the hyperparameter grid consisted of a weight decay factor ranging from 0.1 to 1 −15 by a log 10 scale, and a batch size range from 32 to 1024 by a log 2 scale.

For DO-based models, the hyperparameter grid consisted of the same weight decay range, and dropout probabilities in {0.2, 0.1, 0.05, 0.01, 0.005, 0.001}. DO-based models used a batch size of 32 in all evaluations.

The model with optimal training hyperparameters was used to optimize τ numerically.

This optimization was made in terms of average CV CRPS for MCBN, CUBN, MCDO, and CUDO respectively, before evaluation on the test data.

All estimates for the predictive distribution were obtained by taking 500 stochastic forward passes through the network, throughout training and testing.

The implementation was done with TensorFlow.

The Adam optimizer was used to train all networks, with a learning rate of 0.001.

The extensive part of the experiments (i.e. training and cross validation) was done on Amazon web services using 3000 machine-hours.

All code necessary for reproducing both the quantitative and qualitative results is released in an anonymous github repository (https://github.com/iclr-mcbn/mcbn).

A summary of the results measuring uncertainty quality of MCBN, MCDO and MNF are provided in TAB2 .

Tests are run over eight datasets using 5 random 80-20 splits of the data with 5 different random seeds each split.

We report CRPS and PLL, expressed as a percentage, which reflects how close the model is to the upper bound.

The upper bounds and lower bounds for each metric are de- scribed in Section 4.2.

We check to see if the reported values of CRPS and PLL significantly exceed the lower bound models (CUBN and CUDO) using a one sample t-test, where the significance level is indicated by *'s.

Further details from the experiment are available in Appendix 6.6.In FIG0 , we provide a novel visualization of uncertainty quality visualization in regression datasets.

Errors in the model predictions are sorted by estimated uncertainty.

The shaded areas show the model uncertainty and gray dots show absolute prediction errors on the test set.

A gray line depicts a running mean of the errors.

The dashed line indicates the optimized constant uncertainty.

In these plots, we can see a correlation between estimated uncertainty (shaded area) and mean error (gray).

This trend indicates that the model uncertainty estimates can recognize samples with larger (or smaller) potential for predictive errors.

Qualitative results for Bayesian SegNet using MCBN was produced by using the main CamVid model in BID0 .

The pre-trained model was obtained from the online model zoo and was used without modification.

10 instances of mini-batches with size 6 were used to estimate the mean and variance of MCBN.

Qualitative results can be found in FIG1 depicting intuitive , 2015) .

In the upper left, a scene from the CamVid driving scenes dataset.

In the upper right, the Bayesian estimated segmentation.

In the lower left, estimated uncertainty using MCBN for the car class.

In the lower right, the estimated uncertainty of MCBN for all 11 classes.uncertainty at object boundaries.

Quantitative measures on various segmentation datasets can be obtained and is beyond the scope of this work.

We provide additional experimental results in Appendix 6.6.

In TAB5 , we show the mean CRPS and PLL values for MCBN and MCDO.

These results indicate that MCBN performs on par with MCDO across several datasets.

In Table 6 we provide RMSE results of the MCBN and MCDO networks in comparison with non-stochastic BN and DO networks.

These results indicate that the procedure of multiple forward passes in MCBN and MCDO show slight improvements in the predictive accuracy of the network.

The results presented in TAB2 and Appendix 6.6 indicate that MCBN generates meaningful uncertainty estimates which correlate with actual errors in the model's prediction.

We show statistically significant improvements over CUBN in the majority of the datasets, both in terms of CRPS and PLL.

The visualizations in FIG0 and in Appendix 6.6 show clear correlations between the estimated model uncertainty and actual errors produced by the network.

We perform the same experiments using MCDO, and find that MCBN generally performs on par with MCDO.

Looking closer, in terms of CRPS, MCBN performs better than MCDO in more cases than not.

However, care must be used when comparing different models.

The learned network parameters are different, leading to different predictive means which can confound direct comparison.

The results on the Yacht Hydrodynamics dataset seem contradictory.

The CRPS score for MCBN is extremely negative, while the PLL score is extremely positive.

The opposite trend is observed for MCDO.

To add to the puzzle, the visualization in FIG0 depicts an extremely promising uncertainty estimation that models the predictive errors with high fidelity.

We hypothesize that this strange behavior is due to the small size of the data set, which only contains 60 test samples, or due to the Gaussian assumption of CRPS.

There is also a large variability in the model's accuracy on this dataset, which further confounds the measurements for such limited data.

One might criticize the overall quality of the uncertainty estimates of MCBN and MCDO based on the magnitude of the CRPS and PLL scores in TAB2 .

The scores rarely exceed 10% improvement over the lower bound.

However, we caution that these measures should be taken in context.

The upper bound is very difficult to achieve in practice (it is optimized for each test sample individually), and the lower bound is a quite reasonable estimate for uncertainty.

We have further compared against the recent work of Louizos & Welling (2017) , and find comparable results to their MNF-based variational technique specifically targeted to increase the flexibility of the approximate posterior.

Our approximation of the implied prior in Appendix 6.5 also provides a new interpretation of the empirical evidence that significantly lower λ should be used in batch normalized networks (Ioffe & Szegedy, 2015) .

From a VA perspective, too strong a regularization for a given dataset size could be seen as constraining the prior distribution of BN units' means, effectively narrowing the approximate posterior.

In this work, we have shown that training a deep network using batch normalization is equivalent to approximate inference in Bayesian models.

Using our approach, it is possible to make meaningful uncertainty estimates using conventional architectures without modifying the network or the training procedure.

We show evidence that the uncertainty estimates from MCBN correlate with actual errors in the model's prediction, and are useful for practical tasks such as regression or semantic image segmentation.

Our experiments show that MCBN yields an improvement over the baseline of optimized constant uncertainty on par with MCDO and MNF.

Finally, we make contributions to the evaluation of uncertainty quality by suggesting new evaluation metrics based on useful baselines and upper bounds, and proposing a new visualization tool which gives an intuitive visual explanation of uncertainty quality.

Finally, it should be noted that, over the past few years, batch normalization has become an integral part of most-if-not-all cutting edge deep networks which signifies the relevance of our work for estimating model uncertainty.

Assume we were to come up with a faimly of distributions parametrised by θ in order to approximate the posterior, q θ (ω).

Our goal is to set θ such that q θ (ω) is as similar to p(ω|D) as possible.

One strategy is to minimizing KL(q θ (ω)||p(ω|D)), the KL divergence of p(ω|D) wrt q θ (ω).

Minimizing KL(q θ (ω)||p(ω|D)) is equivalent to maximizing the ELBO: DISPLAYFORM0 Assuming i.i.d.

observation noise, this is equivalent to minimizing: DISPLAYFORM1 Instead of making the optimization on the full training set, we can use a subsampling (yielding an unbiased estimate of L VA (θ)) for iterative optimization (as in mini-batch optimization): DISPLAYFORM2 We now make a reparametrisation: set ω = g(θ, ) where is a RV.

The function g and the distribution of must be such that p(g(θ, )) = q θ (ω).

Assume q θ (ω) can be written q θ (ω| )p( )d where q θ (ω| ) = δ(ω − g(θ, )).

Using this reparametrisation we get: DISPLAYFORM3

If q θ (ω) and p(ω) factorize over all stochastic parameters: DISPLAYFORM0 such that KL(q θ (ω)||p(ω)) is the sum of the KL divergence terms for the individual stochastic parameters ω i .

If the factorized distributions are Gaussians, where DISPLAYFORM1 for each KL divergence term.

Here DISPLAYFORM2 Here we approximate the distribution of mean and standard deviation of a mini-batch, separately to two Gaussians.

For the mean we get: DISPLAYFORM3 where x m are the examples in the sampled batch.

We will assume these are sampled i.i.d.

10 .

Samples of the random variable W (j) x m are then i.i.d..

Then by central limit theorem (CLT) the following holds for sufficiently large M (often ≥ 30): DISPLAYFORM4 For standard deviation: DISPLAYFORM5 We want to rewrite DISPLAYFORM6 .

We take a Taylor expansion of f (x) = √ x around a = σ 2 .

DISPLAYFORM7 which by CLT is approximately Gaussian for large M .

We can then make use of the Cramer-Slutzky Theorem, which states that if (X n ) n≥1 and (Y n ) n≥1 are two sequences such that X n d − → X and DISPLAYFORM8 Thus, Term B is approximately 0 for large M.

We have DISPLAYFORM0 we can make the same use of Cramer-Slutzky as for Term B, such that Term C is approximately 0 for large M.

We have approximately DISPLAYFORM0

This section provides derivations of properties of the predictive distribution p * (y|x, D) in section 3.4, following BID5 .

We first find the approximate predictive mean and variance for the approximate predictive distribution, then show how to estimate the predictive log likelihood, a measure of uncertainty quality used in the evaluation 4.Predictive mean Assuming Gaussian iid noise defined by model precision τ , i.e. f ω (x, y) = p(y|f ω (x)) = N (y; f ω (x), τ −1 I): DISPLAYFORM0 where we take the MC Integral with T samples of ω for the approximation in the final step.

Predictive variance Our goal is to estimate: DISPLAYFORM1 We find that: DISPLAYFORM2 where we use MC integration with T samples for the final step.

The predictive covariance matrix is given by: DISPLAYFORM3 which is the sum of the variance from observation noise and the sample covariance from T stochastic forward passes though the network.

Predictive Log Likelihood We use the Predictive Log Likelihood (PLL) as a measure to estimate the model's uncertainty quality.

For a certain test point (y i , x i ), the PLL definition and approximation can be expressed as: DISPLAYFORM4 whereω j represents a sampled set of stochastic parameters from the approximate posterior distrubtion q θ (ω) and we take a MC integration with T samples.

For regression, due to the iid Gaussian noise, we can further develop the derivation into the form we use when sampling: DISPLAYFORM5 Note that PLL makes no assumption on the form of the approximate predictive distribution.

The measure is based on repeated samplingω j from q θ (ω), which may be highly multimodal (see section 3.4).

We assume training by SGD with mini-batch size M , L2-regularization on weights and Fully Connected layers.

With θ k ∈ θ, equivalence between the objectives of Eq. FORMULA3 and FORMULA11 then requires: DISPLAYFORM0 To proceed with the LHS of Eq. FORMULA45 we first need to find the approximate posterior q θ (ω) that batch normalization induces.

As shown in Appendix 6.3, with some weak assumptions and approximations the Central Limit Theorem (CLT) yields Gaussian distributions of the stochastic variables µ u B , σ u B , for large enough M : DISPLAYFORM1 where µ u and σ u are the population-level moments (i.e. moments over D), and h u is the BN unit's input.

We use i as an index of the set of stochastic variables, i.e. ω i ∈ {µ DISPLAYFORM2 Using the result that σ q = 0, one can easily find for ω i = σ u B that µ q = 0 and σ q = 0, nullifying Eq. (7).

We need only consider the partial derivatives of the KL divergence terms where DISPLAYFORM3 If we let µ p = 0, Eq. (7) of an actual training of Yacht dataset for one unit in the first hidden layer and the second hidden layer.

Data is provided for different epochs and for different batch sizes.

The distribution of standard deviation of mini-batches during training of one of our datasets.

The distribution closely follows our analytically approximated Gaussian distribution.

The data is collected for one unit of each layer and is provided for different epochs and for different batch sizes.

@highlight

We show that training a deep network using batch normalization is equivalent to approximate inference in Bayesian models, and we demonstrate how this finding allows us to make useful estimates of the model uncertainty in conventional networks.

@highlight

This paper proposes using batch normalisation at test time to get the predictive uncertainty, and shows Monte Carlo prediction at test time using batch norm is better than dropout.

@highlight

Proposes that the regularization procedure called batch normalization can be understood as performing approximate Bayesian inference, which performs similarly to MC dropout in terms of the estimates of uncertainty that it produces.