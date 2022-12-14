We propose Bayesian hypernetworks: a framework for approximate Bayesian inference in neural networks.

A Bayesian hypernetwork, h, is a neural network which learns to transform a simple noise distribution, p(e) = N(0,I), to a distribution q(t) := q(h(e)) over the parameters t of another neural network (the ``primary network).

We train q with variational inference, using an invertible h to enable efficient estimation of the variational lower bound on the posterior p(t | D) via sampling.

In contrast to most methods for Bayesian deep learning, Bayesian hypernets can represent a complex multimodal approximate posterior with correlations between parameters, while enabling cheap iid sampling of q(t).

In practice, Bayesian hypernets provide a better defense against adversarial examples than dropout, and also exhibit competitive performance on a suite of tasks which evaluate model uncertainty, including regularization, active learning, and anomaly detection.

Simple and powerful techniques for Bayesian inference of deep neural networks' (DNNs) parameters have the potential to dramatically increase the scope of applications for deep learning techniques.

In real-world applications, unanticipated mistakes may be costly and dangerous, whereas anticipating mistakes allows an agent to seek human guidance (as in active learning), engage safe default behavior (such as shutting down), or use a "reject option" in a classification context.

DNNs are typically trained to find the single most likely value of the parameters (the "MAP estimate"), but this approach neglects uncertainty about which parameters are the best ("parameter uncertainty"), which may translate into higher predictive uncertainty when likely parameter values yield highly confident but contradictory predictions.

Conversely, Bayesian DNNs model the full posterior distribution of a model's parameters given the data, and thus provides better calibrated confidence estimates, with corresponding safety benefits BID9 BID0 .

1 Maintaining a distribution over parameters is also one of the most effective defenses against adversarial attacks BID4 .Techniques for Bayesian DNNs are an active research topic.

The most popular approach is variational inference BID2 BID8 , which typically restricts the variational posterior to a simple family of distributions, for instance a factorial Gaussian BID2 BID16 .

Unfortunately, from a safety perspective, variational approximations tend to underestimate uncertainty, by heavily penalizing approximate distributions which place mass in regions where the true posterior has low density.

This problem can be exacerbated by using a restricted family of posterior distribution; for instance a unimodal approximate posterior will generally only capture a single mode of the true posterior.

With this in mind, we propose learning an extremely flexible and powerful posterior, parametrized by a DNN h, which we refer to as a Bayesian hypernetwork in reference to BID17 .A Bayesian hypernetwork (BHN) takes random noise ??? N (0, I) as input and outputs a sample from the approximate posterior q(??) for another DNN of interest (the "primary network").

The key insight for building such a model is the use of an invertible hypernet, which enables Monte Carlo estimation of the entropy term ??? logq(??) in the variational inference training objective.

We begin the paper by reviewing previous work on Bayesian DNNs, and explaining the necessary components of our approach (Section 2).

Then we explain how to compose these techniques to yield Bayesian hypernets, as well as design choices which make training BHNs efficient, stable and robust (Section 3).

Finally, we present experiments which validate the expressivity of BHNs, and demonstrate their competitive performance across several tasks (Section 4).

We begin with an overview of prior work on Bayesian neural networks in Section 2.1 before discussing the specific components of our technique in Sections 2.2 and 2.3.

Bayesian DNNs have been studied since the 1990s BID30 BID29 .

For a thorough review, see BID8 .

Broadly speaking, existing methods either 1) use Markov chain Monte Carlo BID39 BID30 , or 2) directly learn an approximate posterior distribution using (stochastic) variational inference BID16 BID9 BID2 , expectation propagation BID19 BID36 , or ??-divergences BID27 .

We focus here on the most popular approach: variational inference.

Notable recent work in this area includes BID9 , who interprets the popular dropout BID37 algorithm as a variational inference method ("MC dropout").

This has the advantages of being simple to implement and allowing cheap samples from q(??).

emulates Gaussian dropout, but yields a unimodal approximate posterior, and does not allow arbitrary dependencies between the parameters.

The other important points of reference for our work are Bayes by Backprop (BbB) BID2 , and multiplicative normalizing flows BID28 .

Bayes by Backprop can be viewed as a special instance of a Bayesian hypernet, where the hypernetwork only performs an element-wise scale and shift of the input noise (yielding a factorial Gaussian distribution).More similar is the work of BID28 , who propose and dismiss BHNs due to the issues of scaling BHNs to large primary networks, which we address in Section 3.3.2 Instead, in their work, they use a hypernet to generate scaling factors, z on the means ?? of a factorial Gaussian distribution.

Because z follows a complicated distribution, this forms a highly flexible approximate posterior: q(??) = q(??|z)q(z)dz.

However, this approach also requires them to introduce an auxiliary inference network to approximate q(z|??) in order to estimate the entropy term of the variational lower bound, resulting in lower bound on the variational lower bound.

Finally, the variational autoencoder (VAE) BID21 BID23 family of generative models is likely the best known application of variational inference in DNNs, but note that the VAE is not a Bayesian DNN in our sense.

VAEs approximate the posterior over latent variables, given a datapoint; Bayesian DNNs approximate the posterior over model parameters, given a dataset.

A hypernetwork BID17 BID3 BID1 ) is a neural net that outputs parameters of another neural net (the "primary network").

3 The hypernet and primary net together form a single model which is trained by backpropagation.

The number of parameters of a DNN scales quadratically in the number of units per layer, meaning naively parametrizing a large primary net requires an impractically large hypernet.

One method of addressing this challenge is Conditional Batch Norm (CBN) BID7 , and the closely related Conditional Instance Normalization (CIN) BID20 BID38 , and Feature-wise Linear Modulation (FiLM) BID31 BID26 , which can be viewed as specific forms of a hypernet.

In these works, the weights of the primary net are parametrized directly, and the hypernet only outputs scale (??) and shift (??) parameters for every neuron; this can be viewed as selecting which features are significant (scaling) or present (shifting).

In our work, we employ the related technique of weight normalization , which normalizes the input weights for every neuron and introduces a separate parameter g for their scale.

Our proposed Bayesian hypernetworks employ a differentiable directed generator network (DDGN) BID15 ) as a generative model of the primary net parameters.

DDGNs use a neural net to transform simple noise (most commonly isotropic Gaussian) into samples from a complex distribution, and are a common component of modern deep generative models such as variational autoencoders (VAEs) BID23 BID21 and generative adversarial networks (GANs) BID13 BID12 .We take advantage of techniques for invertible DDGNs developed in several recent works on generative modeling BID5 and variational inference of latent variables BID32 .

Training these models uses the change of variables formula, which involves computing the log-determinant of the inverse Jacobian of the generator network.

This computation involves a potentially costly matrix determinant, and these works propose innovative architectures which reduce the cost of this operation but can still express complicated deformations, which are referred to as "normalizing flows".

We now describe how variational inference is applied to Bayesian deep nets (Section 3.1), and how we compose the methods described in Sections 2.2 and 2.3 to produce Bayesian hypernets (Section 3.2).

In variational inference, the goal is to maximize a lower bound on the marginal log-likelihood of the data, log p(D) under some statistical model.

This involves both estimating parameters of the model, and approximating the posterior distribution over unobserved random variables (which may themselves also be parameters, e.g., as in the case of Bayesian DNNs).

Let ?? ??? R D be parameters given the Bayesian treatment as random variables, D a training set of observed data, and q(??) a learned approximation to the true posterior p(??|D).

Since the KL divergence is always non-negative, we have, for any q(??): DISPLAYFORM0 The right hand side of FORMULA0 is the evidence lower bound, or "ELBO".The above derivation applies to any statistical model and any dataset.

In our experiments, we focus on modeling conditional likelihoods p(D) = p(Y|X ).

Using the conditional independence assumption, we further decompose log p(D|??) := log p(Y|X , ??) as n i=1 log p(y i |x i , ??), and apply stochastic gradient methods for optimization.

Computing the expectation in (2) is generally intractable for deep nets, but can be estimated by Monte Carlo sampling.

For a given value of ??, log p(D|??) and log p(??) can be computed and differentiated exactly as in a non-Bayesian DNN, allowing training by backpropagation.

The entropy term E q [??? logq(??)] is also straightforward to evaluate for simple families of approximate posteriors such as Gaussians.

Similarly, the likelihood of a test data-point under the predictive posterior using S samples can be estimated using Monte Carlo: DISPLAYFORM0

Bayesian hypernets (BHNs) express a flexible q(??) by using a DDGN (section 2.3), h ??? R D ??? R D , to transform random noise ??? N (0, I D ) into independent samples from q(??).

This makes it cheap to compute Monte Carlo estimations of expectations with respect to q; these include the ELBO, and its derivatives, which can be backpropagated to train the hypernet h.

Since BHNs are both trained and evaluated via samples of q(??), expressing q(??) as a generative model is a natural strategy.

However, while DDGNs are convenient to sample from, computing the entropy term (E q [??? logq(??)]) of the ELBO additionally requires evaluating the likelihood of generated samples, and most popular DDGNs (such as VAEs and GANs) do not provide a convenient way of doing so.

5 In general, these models can be many-to-one mappings, and computing the likelihood of a given parameter value requires integrating over the latent noise variables ??? R D : DISPLAYFORM0 To avoid this issue, we use an invertible h, allowing us to compute q(??) simply by using the change of variables formula: DISPLAYFORM1 where q is the distribution of and ?? = h( ).As discussed in Section 2.3, a number of techniques have been developed for efficiently training such invertible DDGNs.

In this work, we employ both RealNVP (RNVP) BID6 and Inverse Autoregressive Flows (IAF) .

Note that the latter can be efficiently applied, since we only require the ability to evaluate likelihood of generated samples (not arbitrary points in the range of h, as in generative modeling applications, e.g., BID6 ); and this also means that we can use a lower-dimensional to generate samples along a submanifold of the entire parameter space, as detailed below.

In order to scale BHNs to large primary networks, we use the weight normalization reparametrization 6 : DISPLAYFORM0 where ?? j are the input weights associated with a single unit j in the primary network.

We only output the scaling factors g from the hypernet, and learn a maximum likelihood estimate of v. 7 This allows us to overcome the computational limitations of naively-parametrized BHNs noted by BID28 , since computation now scales linearly, instead of quadratically, in the number of BID2 .

In the second subplot, we place a prior on the scaling factor g and infer the posterior distribution using a BHN, while in the third subplot the hypernet is used to generate the whole weight matrices of the primary net.

Each shaded region represents half a standard deviation in the posterior on the predictive mean.

The red crosses are 50 examples from the training dataset.primary net units.

Using this parametrization restricts the family of approximate posteriors, but still allows for a high degree of multimodality and dependence between the parameters.

We also employ weight normalization within the hypernet, and found this stabilizes training dramatically.

Initialization plays an important role as well; we recommend initializing the hypernet weights to small values to limit the impact of noise at the beginning of training.

We also find clipping the outputs of the softmax to be within (0.001, 0.999) critical for numerical stability.

We perform experiments on MNIST, CIFAR10, and a 1D regression task.

There is no single metric for how well a model captures uncertainty; to evaluate our model, we perform experiments on regularization (Section 4.2), active learning (Section 4.3), anomaly detection (Section 4.4), and detection of adversarial examples (Section 4.5).

Active learning and anomaly detection problems make natural use of uncertainty estimates: In anomaly detection, higher uncertainty indicates a likely anomaly.

In active learning, higher uncertainty indicates a greater opportunity for learning.

Parameter uncertainty also has regularization benefits: integrating over the posterior creates an implicit ensemble.

Intuitively, when the most likely hypothesis predicts "A", but the posterior places more total mass on hypotheses predicting "B", we prefer predicting "B".

By improving our estimate of the posterior, we more accurately weigh the evidence for different hypotheses.

Adversarial examples are an especially difficult kind of anomaly designed to fool a classifier, and finding effective defenses against adversarial attacks remains an open challenge in deep learning.

For the hypernet architecture, we try both RealNVP BID6 and IAF(Kingma et al., 2016) with MADE BID11 , with 1-layer ReLU-MLP coupling functions with 200 hidden units (each).

In general, we find that IAF performs better.

We use an isotropic standard normal prior on the scaling factors (g) of the weights of the network.

We also use Adam with default hyper-parameter settings BID22 and gradient clipping in all of our experiments.

Our mini-batch size is 128, and to reduce computation, we use the same noise-sample (and thus the same primary net parameters) for all examples in a mini-batch.

We experimented with independent noise, but did not notice any benefit.

Our baselines for comparison are Bayes by Backprop (BbB) BID2 , MC dropout (MCdropout) BID9 , and non-Bayesian DNN baselines (with and without dropout).

We first demonstrate the behavior of the network on the toy 1D-regression problem from BID2 in FIG0 .

As expected, the uncertainty of the network increases away from the observed data.

We also use this experiment to evaluate the effects of our proposal for scaling BHNs via the weight norm parametrization (Section 3.3) by comparing with a model which generates the full set of parameters, and find that the two models produce very similar results, suggesting that our proposed method strikes a good balance between scalability and expressiveness.

Next, we demonstrate the distinctive ability of Bayesian hypernets to learn multi-modal, dependent distributions.

FIG4 (appendix) shows that BHNs do learn approximate posteriors with dependence between different parameters, as measured by the Pearson correlation coefficient.

Meanwhile, FIG1 shows that BHNs are capable of learning multimodal posteriors.

For this experiment, we trained an over-parametrized linear (primary) network:?? = a ?? b ?? x on a dataset generated as y = x + , and the BHN learns capture both the modes of a = b = 1 and a = b = ???1.

We now show that BHNs act as a regularizer, outperforming dropout and traditional mean field (BbB).

Results are presented in TAB0 .

In our experiments, we find that BHNs perform on par with dropout on full datasets of MNIST and CIFAR10; furthermore, increasing the flexibility of the posterior by adding more coupling layers improves performance, especially compared with models with 0 coupling layers, which cannot model dependencies between the parameters.

We also evaluate on a subset of MNIST (the first 5,000 examples); results are presented in the last two columns of TAB0 .

Replicating these experiments (with 8 coupling layers) for 10 trials yields Figure 3.

In these MNIST experiments, we use MLPs with 2 hidden layers of 800 or 1200 hidden units each.

For CIFAR10, we train a convolutional neural net (CNN) with 4 hidden layers of [64, 64, 128, 128 ] channels, 2 ?? 2 max pooling after the second and the fourth layers, filter size of 3, and a single fully connected layer of 512 units.

Figure 4: Active learning: Bayesian hypernets outperform other approaches after sufficient acquisitions when warm-starting (left), for both random acquisition function (top) and BALD acquisition function (bottom).

Warm-starting improves stability for all methods, but hurts performance for other approaches, compared with randomly re-initializing parameters as in Gal et al. (2017) (right) .

We also note that the baseline model (no dropout) is competitive with MCdropout, and outperforms the Dropout baseline used by .

9 These curves are the average of three experiments.

We now turn to active learning, where we compare to the MNIST experiments of , replicating their architecture and training procedure.

Briefly, they use an initial dataset of 20 examples (2 from each class), and acquire 10 new examples at a time, training for 50 epochs between each acquisition.

While re-initialize the network after every acquisition, we found that "warm-starting" from the current learned parameters was essential for good performance with BHNs, although it is likely that longer training or better initialization schemes could perform the same role.

Overall, warm-started BHNs suffered at the beginning of training, but outperformed all other methods for moderate to large numbers of acquisitions.

For anomaly detection, we take BID18 as a starting point, and perform the same suite of MNIST experiments, evaluating the ability of networks to determine whether an input came from their training distribution ("Out of distribution detection").

BID18 found that the confidence expressed in the softmax probabilities of a (non-Bayesian) DNN trained on a single dataset provides a good signal for both of these detection problems.

We demonstrate that Bayesian DNNs outperform their non-Bayesian counterparts.

Just as in active learning, in anomaly detection, we use MC to estimate the predictive posterior, and use this to score datapoints.

For active learning, we would generally like to acquire points where there is higher uncertainty.

In a well-calibrated model, these points are also likely to be challenging or anomalous examples, and thus acquisition functions from the active learning literature are good candidates for scoring anomalies.

We consider all of the acquisition functions listed in as possible scores for the Area Under the Curve (AUC) of Precision-Recall (PR) and Receiver Operating Characteristic (ROC) metrics, but found that the maximum confidence of the softmax probabilities (i.e., "variation ratio") acquisition function used by BID18 gave the best performance.

Both BHN and MCdropout achieve significant performance gains over the non-Bayesian baseline, and MCdropout performs significantly better than BHN in this task.

Results are presented in TAB1 .Second, we follow the same experimental setup, using all the acquisition functions, and exclude one class in the training set of MNIST at a time.

We take the excluded class of the training data as out-of-distribution samples.

The result is presented in TAB2 (Appendix).

This experiment shows the benefit of using scores that reflect dispersion in the posterior samples (such as mean standard deviation and BALD value) in Bayesian DNNs.

Finally, we consider this same anomaly detection procedure as a novel tool for detecting adversarial examples.

Our setup is similar to BID27 and BID28 , where it is shown that when more perturbation is added to the data, model uncertainty increases and then drops.

We use the Fast Gradient Sign method (FGS) BID14 for adversarial attack, and use one sample of our model to estimate the gradient.

10 We find that, compared with dropout, BHNs are less confident on data points which are far from the data manifold.

In particular, BHNs constructed with IAF consistently outperform RealNVP-BHNs and dropout in detecting adversarial examples and errors.

Results are shown in FIG3 .

We introduce Bayesian hypernets (BHNs), a new method for variational Bayesian deep learning which uses an invertible hypernetwork as a generative model of parameters.

BHNs feature efficient we found the BALD values our implementation computes provide a better-than-random acquisition function (compare the blue line in the top and bottom plots).

10 Li & Gal (2017) and BID28 used 10 and 1 model samples, respectively, to estimate gradient.

We report the result with 1 sample; results with more samples are given in the appendix.

when more perturbation is added to the data (left), uncertainty measures also increase (first row).

In particular, the BALD and Mean STD scores, which measure epistemic uncertainty, are strongly increasing for BHNs, but not for dropout.

The second row and third row plots show results for adversary detection and error detection (respectively) in terms of the AUC of ROC (y-axis) with increasing perturbation along the x-axis.

Gradient direction is estimated with one Monte Carlo sample of the weights/dropout mask.training and sampling, and can express complicated multimodal distributions, thereby addressing issues of overconfidence present in simpler variational approximations.

We present a method of parametrizing BHNs which allows them to scale successfully to real world tasks, and show that BHNs can offer significant benefits over simpler methods for Bayesian deep learning.

Future work could explore other methods of parametrizing BHNs, for instance using the same hypernet to output different subsets of the primary net parameters.

A

We replicate the experiments of anomaly detection with unseen classes of MNIST.

Here we use 32 samples to estimate the gradient direction with respect to the input.

A better estimate of gradient amounts to a stronger attack, so accuracy drops lower for a given step size while an adversarial example can be more easily detected with a more informative uncertainty measure.

In this paper, we employ weight normalization in the primary network (7), treating (only) the scaling factors g as random variables.

We choose an isotropic Gaussian prior for g: p(g) = N (g; 0, ??I), which results in an L 2 weight-decay penalty on g, or, equivalently, w = g ??? E ???q ( ),g=h ?? ( ) [log p(D|g; v, b) + log p(g) ??? log q( ) + log det DISPLAYFORM0 where v and b are the direction and bias parameters of the primary net, and ?? is the parameters of the hypernetwork.

We optimize this bound with respect to {v, b, ??} during training.

<|TLDR|>

@highlight

We propose Bayesian hypernetworks: a framework for approximate Bayesian inference in neural networks.