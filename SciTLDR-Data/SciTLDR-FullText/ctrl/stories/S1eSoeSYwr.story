Deterministic neural networks (NNs) are increasingly being deployed in safety critical domains, where calibrated, robust and efficient measures of uncertainty are crucial.

While it is possible to train regression networks to output the parameters of a probability distribution by maximizing a Gaussian likelihood function, the resulting model remains oblivious to the underlying confidence of its predictions.

In this paper, we propose a novel method for training deterministic NNs to not only estimate the desired target but also the associated evidence in support of that target.

We accomplish this by  placing evidential priors over our original Gaussian likelihood function and training our NN to infer the hyperparameters of our evidential distribution.

We impose priors during training such that the model is penalized when its predicted evidence is not aligned with the correct output.

Thus the model estimates not only the probabilistic mean and variance of our target but also the underlying uncertainty associated with each of those parameters.

We observe that our evidential regression method learns well-calibrated measures of uncertainty on various benchmarks, scales to complex computer vision tasks, and is robust to adversarial input perturbations.

Figure 1: Evidential distributions.

Maximum likelihood optimization learns a likelihood distribution given data, while evidential distributions model higher-order probability distribution over the likelihood parameters.

Recent advances in deep supervised learning have yielded super human level performance and precision.

While these models empirically generalize well when placed into new test enviornments, they are often easily fooled by adversarial perturbations (Goodfellow et al., 2014) , and have difficulty understanding when their predictions should not be trusted.

Today, regression based neural networks (NNs) are being deployed in safety critical domains of computer vision (Godard et al., 2017) as well as in robotics and control (Bojarski et al., 2016) where the ability to infer model uncertainty is crucial for eventual wide-scale adoption.

Furthermore, precise uncertainty estimates are useful both for human interpretation of confidence and anomaly detection, and also for propagating these estimates to other autonomous components of a larger, connected system.

Existing approaches to uncertainty estimation are roughly split into two categories: (1) learning aleatoric uncertainty (uncertainty in the data) and (2) epistemic uncertainty (uncertainty in the prediction).

While representations for aleatoric uncertainty can be learned directly from data, approaches for estimating epistemic uncertainty focus on placing probabilistic priors over the weights and sampling to obtain a measure of variance.

In practice, many challenges arise with this approach, such as the computational expense of sampling during inference, how to pick an appropriate weight prior, or even how to learn such a representation given your prior.

Instead, we formulate learning as an evidence acquisition process, where the model can acquire evidence during training in support of its prediction (Sensoy et al., 2018; Malinin & Gales, 2018) .

Every training example adds support to a learned higher-order, evidential distribution.

Sampling from this distribution yields instances of lower-order, likelihood functions from which the data was drawn (cf.

Fig. 1 ).

We demonstrate that, by placing priors over our likelihood function, we can learn a grounded representation of epistemic and aleatoric uncertainty without sampling during inference.

1.

A novel and scalable method for learning representations of epistemic and aleatoric uncertainty, specifically on regression problems, by placing evidential priors over the likelihood; 2.

Formulation of a novel evidential regularizer for continuous regression problems, which we show is necessary for expressing lack of a evidence on out-of-distribution examples; 3.

Evaluation of learned epistemic uncertainty on benchmark regression tasks and comparison against other state-of-the-art uncertainty estimation techniques for neural networks; and 4.

Robustness evaluation against out of distribution and adversarially perturbed test data.

Consider the following supervised optimization problem: given a dataset, D, of N paired training examples, (x 1 , y 1 ), . . .

, (x N , y N ), we aim to learn a function f , parameterized by a set of weights, w, which approximately solves the following optimization problem:

where L i (·) describes a loss function.

In this work, we consider deterministic regression problems, which commonly optimize the sum of squared errors,

In doing so, the model is encouraged to learn the average correct answer for a given input, but does not explicitly model any underlying noise or uncertainty in the data when making its estimation.

We can also approach our optimization problem from a maximum likelihood perspective, where we learn model parameters that maximize the likelihood of observing a particular set of training data.

In the context of deterministic regression, we assume our targets, y i , were drawn i.i.d.

from a Gaussian distribution with mean and variance parameters θ = (µ, σ 2 ).

In maximum likelihood estimation, we aim to learn a model to infer θ = (µ, σ 2 ) that maximize the likelihood of observing our targets, y, given by p(y i |θ).

In practice, we minimize the negative log likelihood by setting:

In learning the parameters θ, this likelihood function allows us to successfully model the uncertainty of our data, also known as the aleatoric uncertainty.

However, our model remains oblivious to the predictive model or epistemic uncertainty (Kendall & Gal, 2017) .

In this paper, we present a novel approach for estimating the evidence in support of network predictions by directly learning both the inferred aleatoric uncertainty as well as the underlying epistemic uncertainty over its predictions.

We achieve this by placing higher-order prior distributions over the learned parameters governing the distribution from which our observations are drawn.

We consider the problem where our observed targets, y i , are drawn i.i.d.

from a Gaussian distribution now with unknown mean and variance (µ, σ 2 ), which we seek to probabilistically estimate.

We model this by placing a conjugate prior distribution on (µ, σ 2 ).

If we assume our observations are drawn from a Gaussian, this leads to placing a Gaussian prior on our unknown mean and an Inverse-Gamma prior on our unknown variance: 2 ).

Sampling from a single realization of a higher-order evidential distribution (B), yields lower-order likelihoods (C) over the data (e.g. p(y|µ, σ 2 )).

Darker shading indicates higher probability mass.

We aim to learn a model (D) that predicts the target, y, from an input, x, with an evidential prior imposed on our likelihood to enable uncertainty estimation.

where Γ(·) is the gamma function, m = (γ, λ, α, β), and γ ∈ R, λ > 0, α > 0, β > 0.

Our aim is to estimate a posterior distribution q(µ, σ 2 ) = p(µ, σ 2 |y 1 , . . .

, y N ).

To obtain an approximation for the true posterior, we assume that the estimated distribution can be factorized (Parisi, 1988) such that q(µ, σ 2 ) = q(µ) q(σ 2 ).

Thus, our approximation takes the form of the Gaussian conjugate prior, the Normal Inverse-Gamma (N.I.G.) distribution:

A popular interpretation of the parameters of the conjugate prior distribution is in terms of "virtualobservations" in support of a given property (Jordan, 2009) .

For example, the mean of a N.I.G. distribution can be interpreted as being estimated from λ virtual-observations with sample mean γ while its variance was estimated from 2α virtual-observations with sample mean γ and sum of squared deviations 2β.

Following from this interpretation, we define the total evidence, Φ, of our evidential distributions as the sum of all inferred virtual-observations counts: (Φ = λ + 2α).

Drawing a sample θ j from the N.I.G. distribution yields a single instance of our likelihood function, namely N (µ j , σ 2 j ).

Thus, the N.I.G. hyperparameters, (γ, λ, α, β), determine not only the location but also the dispersion concentrations, or uncertainty, associated with our inferred likelihood function.

Therefore, we can interpret the N.I.G. distribution as higher-order, evidential, distribution on top of the unknown lower-order likelihood distribution from which observations are drawn.

For example, in Fig. 2A we visualize different evidential N.I.G. distributions with varying model parameters.

We illustrate that by increasing the evidential parameters (i.e. λ, α) of this distribution, the p.d.f. becomes tightly concentrated about its inferred likelihood function.

Considering a single parameter realization of this higher-order distribution, cf.

Fig. 2B , we can subsequently sample many lower-order realizations of our likelihood function, as shown in Fig. 2C .

In this work, we use neural networks to infer the hyperparameters of this higher-order, evidential distribution, given an input.

This approach presents several distinct advantages compared to prior work.

First, our method enables simultaneous learning of the desired regression task, along with aleatoric and epistemic uncertainty estimation, built in, by enforcing evidential priors.

Second, since the evidential prior is a higher-order N.I.G. distribution, the maximum likelihood Gaussian can be computed analytically from the expected values of the (µ, σ 2 ) parameters, without the need for sampling.

Third, we can effectively estimate the epistemic or model uncertainty associated with the network's prediction by simply evaluating the variance of our inferred evidential distribution.

Having formalized the use of an evidential distribution to capture both aleatoric and epistemic uncertainty, we next describe our approach for learning a model (c.f.

Fig. 2D ) to output the hyperparameters of this distribution.

For clarity, we will structure the learning objective into two distinct parts: (1) acquiring or maximizing model evidence in support of our observations and (2) minimizing evidence or inflating uncertainty when the prediction is wrong.

At a high level, we can think of (1) as a way of fitting our data to the evidential model while (2) enforces a prior to inflate our uncertainty estimates.

(1) Maximizing the model fit.

From Bayesian probability theory, the "model evidence", or marginal likelihood, is defined as the likelihood of an observation, y i , given the evidential distribution parameters m and is computed by marginalizing over the likelihood parameters θ:

The model evidence is not, in general, straightforward to evaluate since computing it involves integrating out the dependence on latent model parameters:

However, by placing a N.I.G. evidential prior on our Gaussian likelihood function an analytical solution for the model evidence does exist.

For computational reasons, we minimize the negative logarithm of the model evidence (L NLL i (w)).

For a complete derivation please refer to Sec. 7.1,

Instead of modeling this loss using empirical Bayes, where the objective is to maximize model evidence, we alternatively can minimize the sum-of-squared (SOS) errors, between the evidential prior and the data that would be sampled from the associated likelihood.

Thus, we define L

A step-by-step derivation is given in Sec. 7.1.

In our experiments, using L SOS i (w) resulted in greater training stability and increased performance, compared to the

(2) Minimizing evidence on errors.

In the first term of our objective above, we outlined a loss function for training a NN to output parameters of a N.I.G. distribution to fit our observations, either by maximizing the model evidence or minimizing the sum-of-squared errors.

Now, we describe how to regularize training by applying a lack of evidence prior (i.e., maximum uncertainty).

Therefore, during training we aim to minimize our evidence (or maximize our uncertainty) everywhere except where we have training data.

This can be done by minimizing the KL-divergence between the inferred posterior, q(θ), and a prior, p(θ).

This has been demonstrated with success in the categorical setting where the uncertainty prior can be set to a uniform Dirichlet (Malinin & Gales, 2018; Sensoy et al., 2018) .

In the regression setting, the KL-divergence between our posterior and a N.I.G. zero evidence prior (i.e., {α, λ} = 0) is not well defined (Soch & Allefeld, 2016) , please refer to Sec. 7.2 for a derivation.

Furthermore, this prior needs to be enforced specifically where there is no support from the data.

Past works in classification accomplish this by using the ground truth likelihoood classification (i.e., the one-hot encoded labels) to remove the non-misleading evidence.

However, in regression, labels are provided as point targets (not ground truth Gaussian likelihoods).

Unlike classification, it is not possible to penalize evidence everywhere except our single point estimate, as this space is infinite and unbounded.

Thus, these previously explored approaches for evidential optimization are not directly applicable.

To address both of these shortcomings of past works, now in the regression setting, we formulate a novel evidence regularizer, L R i , based on the error of the i-th prediction,

where x p represents the L-p norm of x. The value of p impacts the penalty imposed on the evidence when a wrong prediction is made.

For example, p = 2, heavily over-penalizes the evidence on larger errors, whereas p = 1 and p = 0.5 saturate the evidence penalty for larger errors.

We found that p = 1 provided the optimal stability during training and use this value in all presented results.

This regularization loss imposes a penalty whenever there is an error in the prediction that scales with the total evidence of our inferred posterior.

Conversely, large amounts of predicted evidence will not be penalized as long as the prediction is close to the target observation.

We provide an ablation analysis to quantitatively demonstrate the added value of this evidential regularizer in Sec 7.3.2.

The combined loss function employed during training consists of the two loss terms for maximizing model evidence and regularizing evidence,

The aleatoric uncertainty, also referred to as statistical or data uncertainty, is representative of unknowns that differ each time we run the same experiment.

We evaluate the aleatoric uncertainty from E[σ 2 ] = β α−1 .

The epistemic, also known as the model uncertainty, describes the estimated uncertainty in the learned model and is defined as

which is expected as λ is one of our two evidential virtual-observation counts.

We first qualitatively compare the performance of our approach against a set of benchmarks on a one-dimensional toy regression dataset (Fig. 3) .

For training and dataset details please refer to Sec. 7.3.1.

We compare deterministic regression, as well as techniques using empirical variance of the networks' predictions such as MC-dropout, model-ensembles, and Bayes-byBackprop which underestimate the uncertainty outside the training distribution.

In contrast, evidential regression estimates uncertainty appropriately and grows the uncertainty estimate with increasing distance from the training data.

Additionally, we compare our approach to stateof-the-art methods for predictive uncertainty estimation using NNs on common real world datasets used in (Hernández-Lobato & Adams, 2015; Lakshminarayanan et al., 2017; Gal & Ghahramani, 2016) .

We evaluate our proposed evidential regression method against model-ensembles and BBB based on root mean squared error (RMSE), and negative log-likelihood (NLL).

We do not provide results for MC-dropout since it consistently performed inferior to the other baselines.

The results in Table 1 indicate that although the loss function for evidential regression is more complex than competing approaches, it is the top performer in RMSE and NLL in 8 out of 9 datasets.

Furthermore, we demonstrate that, on a synthetic dataset with a priori known noise, evidential models can additionally estimate and recover the underlying aleatoric uncertainty.

For more information please refer to Sec. 7.3.3 for results and experiment details.

Ensembles BBB Evidential Ensembles BBB Evidential Boston 0.09 ± 4.3e-4 0.09 ± 3.7e-4 0.09 ± 1.0e-6 -0.89 ± 6.5e-2 -0.67 ± 1.5e-2 -0.87 ± 2.2e-2 Concrete 0.07 ± 4.4e-3 0.06 ± 3.3e-6 0.06 ± 7.0e-7 -1.29 ± 4.1e-2 -1.32 ± 4.3e-3 -1.31 ± 1.9e-2 Energy 0.10 ± 2.3e-4 0.10 ± 1.6e-5 0.10 ± 9.0e-7 -0.61 ± 8.9e-2 -0.60 ± 2.0e-2 -0.75 ± 1.4e-2 Kin8nm 0.07 ± 3.5e-4 0.17 ± 3.5e-4 0.08 ± 3.8e-3 -0.78 ± 1.4e-2 -0.32 ± 6.3e-3 -1.17 ± 2.6e-2 Naval 0.01 ± 1.0e-7 0.04 ± 1.2e-2 0.01 ± 3.4e-4 -2.55 ± 3.3e-2 -1.83 ± 2.4e-1 -3.17 ± 2.1e-3 Power 0.06 ± 4.0e-7 0.06 ± 2.3e-6 0.06 ± 5.3e-6 -1.29 ± 6.9e-2 -1.33 ± 2.5e-3 -1.40 ± 6.2e-3 Protein 0.17 ± 1.0e-6 0.17 ± 8.0e-4 0.17 ± 1.6e-6 -0.27 ± 6.7e-2 0.32 ± 5.9e-2 -0.29 ± 1.1e-2 Wine 0.10 ±

3.0e-4 0.10 ± 2.9e-4 0.10 ± 3.8e-5 -0.46 ± 2.5e-1 -0.89 ± 2.4e-3 -0.85 ± 6.9e-3 Yacht 0.07 ±

1.3e-3 0.07 ± 3.4e-3 0.06 ± 6.2e-5 -1.16 ± 6.3e-2 -0.74 ± 5.8e-2 -1.28 ± 9.4e-3 Table 1 : Benchmark regression tests.

We evaluate RMSE and negative log-likelihood (NLL) for model ensembling (Lakshminarayanan et al., 2017) , Bayes-By-Backprop (BBB) (Blundell et al., 2015) and evidential regression.

Evidential achieves top scores (bolded, within statistical significance) on 8 of the 9 datasets.

After establishing benchmark comparison results, in this subsection we demonstrate the scalability of our evidential learning by extending to the complex, high-dimensional task of depth estimation.

Monocular end-to-end depth estimation is a central problem in computer vision which aims to learn a representation of depth directly from an RGB image of the scene.

This is a challenging learning task since the output target y is very high-dimensional.

For every pixel in the image, we regress over the desired depth and simultaneously estimate the uncertainty associated to that individual pixel.

Our training data consists of over 27k RGB-to-depth pairs of indoor scenes (e.g. kitchen, bedroom, etc.) from the NYU Depth v2 dataset (Nathan Silberman & Fergus, 2012) .

We train a U-Net style NN (Ronneberger et al., 2015) for inference.

The final layer of our model outputs a single H × W activation map in the case of deterministic regression, dropout, ensembling and BBB.

Evidential models output four final activation maps, corresponding to (γ, λ, α, β).

Table 2 summarizes the size and speed of all models.

Evidential models contain significantly fewer trainable parameters than ensembles (where the number of parameters scales linearly with the size of the ensemble).

BBB maintains a trainable mean and variance for every weight in the network, so its size is roughly 2× larger as well.

Since evidential regression models do not require sampling in order to estimate their uncertainty, their forward-pass inference times are also significantly more efficient.

Finally, we demonstrate comparable predictive accuracy (through RMSE and NLL) to the other models.

For a more detailed breakdown of how the number of samples effects the baselines please refer to Tab.

3. Note that the output size of the depth estimation problem presented significant learning challenges for the BBB baseline, and it was unable to converge during training.

As a result, for the remainder of this analysis we compare against only spatial dropout and ensembles.

We evaluate these models in terms of their accuracy and their predictive uncertainty on unseen test data.

Fig. 4A -C visualizes the predicted depth, absolute error from ground truth, and predictive uncertainty across three randomly picked test images.

Ideally, a strong predictive uncertainty would capture any errors in the prediction (i.e., roughly correspond to where the model is making errors).

Compared to dropout and ensembling, evidential uncertainty modeling captures the depth errors while providing clear and localized predictions of confidence.

In general, dropout drastically underestimates the amount of uncertainty present, while ensembling occasionally overestimates the uncertainty.

To evaluate uncertainty calibration to the ground-truth errors, we fit receiver operating characteristic (ROC) curves to normalized estimates of error and uncertainty.

Thus, we test the network's ability to detect how likely it is to make an error at a given pixel using its predictive uncertainty.

-C) .

Ideally, the model should predict high uncertainty whenever it does not know the answer (i.e., large error).

We evaluate the sensitivity and specificity of the predictive uncertainty in identifying likely errors with ROC curves (D).

ROC curves take into account sensitivity and specificity of the uncertainties towards error predictions and are stronger if they contain greater area under their curve (AUC).

Fig. 4D demonstrates that our evidential model provides uncertainty estimates concentrate to where the model is making the errors.

In addition to epistemic uncertainty, we also evaluate the aleatoric uncertainty estimates that are learned from our evidential models as well.

Fig. 5 compares the evidential aleatoric uncertainty to those obtained by Gaussian likelihood optimization in several domains with high data uncertainty (mirror reflections and poor illumination).

The results between both methods are in strong agreement, identifying mirror reflections and dark regions without visible geometry as sources of high uncertainty.

A key use of uncertainty estimation is to understand when a model is faced with test samples that fall out-of-distribution (OOD) or when the model's output cannot be trusted.

In the previous subsection, we showed that our evidential uncertainties were well calibrated with the model's errors.

In this subsection, we investigate the performance on out-ofdistribution samples.

Fig. 6 illustrates predicted depth on various test input images (left) and outside (right) of the original distribution.

All images have not been seen by the model during training.

We qualitatively and quantitatively demonstrate that the epistemic uncertainty predicted by our evidential model consistently increases on the OOD samples.

Next, we consider the extreme case of OOD detection where the inputs are adversarially perturbed to inflict maximum error on the model.

We compute adversarial perturbations to our test set using the fast gradient sign method (Goodfellow et al., 2014) , with increasing scales, , of noise.

Fig. 7A Figure 6 : Out-of-distribution (OOD) data samples.

Evidential models estimate and inflate epistemic uncertainty on OOD data, where the prediction should not be trusted.

All samples were not seen during training.

confirms that the absolute error of all methods increasing as adversarial noise is added.

We also observe a positive effect noise on our predictive uncertainty estimates in Fig. 7B .

An additional desirable property of evidential uncertainty modeling is that it presents a higher overall uncertainty when presented with adversarial inputs compared to dropout and ensembling methods.

Furthermore, we observe this strong overall uncertainty estimation despite the model losing calibration accuracy from the adversarial examples (Fig. 7C ).

The robustness of evidential uncertainty against adversarial perturbations is visualized in greater detail in Fig. 7D , which illustrates the predicted depth, error, and estimated pixel-wise uncertainty as we perturb the input image with greater amounts of noise (left-to-right).

Note that the predictive uncertainty not only steadily increases as we increase the noise, but the spatial concentrations of uncertainty throughout the image maintain tight correspondence with the error.

Uncertainty estimation has a long history in neural networks, from modeling probability distribution parameters over outputs (Bishop, 1994) to Bayesian deep learning (Kendall & Gal, 2017) .

Our work builds on this foundation and presents a scalable representation for inferring the parameters of an evidential uncertainty distribution while simultaneously learning regression tasks via MLE.

In Bayesian deep learning, priors are placed over network weights and estimated using variational inference (Kingma et al., 2015) .

Dropout (Gal & Ghahramani, 2016; Molchanov et al., 2017) and BBB (Blundell et al., 2015) rely on multiple samples to estimate predictive variance.

Ensembles (Lakshminarayanan et al., 2017) provide a tangential approach where sampling occurs over multiple trained instances.

In contrast, we place uncertainty priors over the likelihood function and thus only need a single forward pass to evaluate both prediction and uncertainty.

Additionally, our approach of uncertainty estimation proved to be better calibrated and capable of predicting where the model fails.

A large topic of research in Bayesian inference focuses on placing prior distributions over hierarchical models to estimate uncertainty (Gelman et al., 2006; 2008) .

Our methodology falls under the class of evidential deep learning which models higher-order distribution priors over neural network predictions to interpret uncertainty.

Prior works in this field (Sensoy et al., 2018; Malinin & Gales, 2018) have focused exclusively on modeling uncertainty in the classification domain with Dirichlet prior distributions.

Our work extends this field into the broad range of regression learning tasks (e.g. depth estimation, forecasting, robotic control learning, etc.) and demonstrates generalizability to out-of-distribution test samples and complex learning problems.

In this paper, we develop a novel method for training deterministic NNs that both estimates a desired target and evaluates the evidence in support of the target to generate robust metrics of model uncertainty.

We formalize this in terms of learning evidential distributions, and achieve stable training by penalizing our model for prediction errors that scale with the available evidence.

Our approach for evidential regression is validated on a benchmark regression task.

We further demonstrate that this method robustly scales to a key task in computer vision, depth estimation, and that the predictive uncertainty increases with increasing out-of-distribution adversarial perturbation.

This framework for evidential representation learning provides a means to achieve the precise uncertainty metrics required for robust neural network deployment in safety-critical domains.

For convenience, define τ = 1/σ 2 be the precision of a Gaussian distribution.

The change of variables transforms the Normal Inverse-Gamma distribution p(µ, σ 2 |γ, λ, α, β) to the equivalent Normal Gamma distribution p(µ, τ |γ, λ, α, β), parameterized by precision τ ∈ (0, ∞) instead of variance σ 2 ,

Marginalizing out µ and τ gives the result of equation 5,

For computational reasons it is common to instead minimize the negative logarithm of the model evidence.

Similarly, we can marignalize out µ and σ 2 to receive the result of equation 8,

The KL-divergence between two Normal Inverse-Gamma functions is given by (Soch & Allefeld, 2016) :

Γ(·) is the Gamma function and Ψ(·) is the Digamma function.

The evidence is defined by (2α + λ).

For zero evidence, both α = 0 and λ = 0.

To compute the KL divergence between one N.I.G distribution and another with zero evidence we can set either {α 2 , λ 2 } = 0 (i.e., forward-KL) in which case, Γ(0) is not well defined, or {α 1 , λ 1 } = 0 (i.e. reverse-KL) which causes a divide-by-zero error of λ 1 .

In either approach, the KL-divergence between an arbitrary N.I.G and one with zero evidence can not be evaluated.

The training set consists of training examples drawn from y = sin(3x)/(3x) + , where ∼ N (0, 0.02) in the region −3 ≤ x ≤ 3, whereas the test data is unbounded.

All models consisted of 100 neurons with 3 hidden layers and were trained to convergence.

The data presented in Fig. 3 illustrates the estimated epistemic uncertainty and predicted mean accross the entire test set, −3 ≤ x ≤ 3.

In the following experiment, we demonstrate the importance of augmenting the training objective with our evidential regularizer L R as introduced in Sec. 3.2.

Fig. 8 provides quantitative results on training the same regression problem presented in 7.3.1 with and without this evidential regularization term.

This term introduces an "uncertain" prior into our learning process so out-of-distribution (OOD) samples exhibit high epistemic uncertainty.

Without the use of this novel loss term, the learned epistemic uncertainty is unreliable on OOD data.

No Data Data Ground Truth Prediction Uncertainty Figure 8 : Evidential regularizer.

The use of our novel L R loss during training helps minimize evidence (maximize uncertainty) on out-of-distribution data, thus enabling OOD uncertainty robustness for regression prediction problems.

The training set consists of training examples drawn from y = sin(3x)/(3x) + (x), where (x) ∼ N (0, s(x)), and s(x) = 1 20 cos(3.3x) + 0.1.

We evaluate against (Kendall & Gal, 2017) which presents an algorithm for heteroscedastic aleatoric uncertainty estimation by inferring the mean and variance of a Gaussian likelihood function.

As presented in the paper, training is done by minimizing the negative log-likelihood of the data given the inferred likelihood parameters.

Both our network and the baseline Gaussian NLL network consisted of 100 neurons with 3 hidden layers and were trained to convergence.

Figure 9 : Aleatoric uncertainty estimation.

Comparing the ability to learn the heteroscedastic aleatoric uncertainty in a synthetic dataset.

Evidential modelling is able to match the performance of Gaussian likelihood optimization (Kendall & Gal, 2017 Table 3 : Depth estimation performance.

Comparison of different epistemic uncertainty estimation algorithms and predictive performance on an unseen test set.

Dropout, ensembles, and Bayes-by-Backprop were sampled N times on parallel threads.

The evidential method outperforms all other algorithms in terms of space (#Parameters) and inference speed while maintaining competetive RMSE and NLL.

<|TLDR|>

@highlight

Fast, calibrated uncertainty estimation for neural networks without sampling

@highlight

This paper proposes a novel approach to estimate the confidence of predictions in a regression setting, opening the door to online applications with fully integrated uncertainty estimates.

@highlight

This paper proposed deep evidential regression, a method for training neural networks to not only estimate the output but also the associated evidence in support of that output.