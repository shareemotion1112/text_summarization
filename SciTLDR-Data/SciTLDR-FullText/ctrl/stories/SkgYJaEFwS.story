We introduce a new and rigorously-formulated PAC-Bayes few-shot meta-learning algorithm that implicitly learns a model prior distribution of interest.

Our proposed method extends the PAC-Bayes framework from a single task setting to the few-shot meta-learning setting to upper-bound generalisation errors on unseen tasks.

We  also propose a generative-based approach to model the shared prior and task-specific posterior more expressively compared to the usual diagonal Gaussian assumption.

We show that the models trained with our proposed meta-learning algorithm are well calibrated and accurate, with state-of-the-art calibration and classification results on mini-ImageNet benchmark, and competitive results in a multi-modal task-distribution regression.

One unique ability of humans is to be able to quickly learn new tasks with only a few training examples.

This is due to the fact that humans tend to exploit prior experience to facilitate the learning of new tasks.

Such exploitation is markedly different from conventional machine learning approaches, where no prior knowledge (e.g. training from scratch with random initialisation) (Glorot & Bengio, 2010) , or weak prior knowledge (e.g., fine tuning from pre-trained models) (Rosenstein et al., 2005) are used when encountering an unseen task for training.

This motivates the development of novel learning algorithms that can effectively encode the knowledge learnt from training tasks, and exploit that knowledge to quickly adapt to future tasks (Lake et al., 2015) .

Prior knowledge can be helpful for future learning only if all tasks are assumed to be distributed according to a latent task distribution.

Learning this latent distribution is, therefore, useful for solving an unseen task, even if the task contains a limited number of training samples.

Many approaches have been proposed and developed to achieve this goal, namely: multi-task learning (Caruana, 1997) , domain adaptation (Bridle & Cox, 1991; Ben-David et al., 2010) and meta-learning (Schmidhuber, 1987; Thrun & Pratt, 1998) .

Among these, meta-learning has flourished as one of the most effective methods due to its ability to leverage the knowledge learnt from many training tasks to quickly adapt to unseen tasks.

Recent advances in meta-learning have produced state-of-the-art results in many benchmarks of few-shot learning data sets (Santoro et al., 2016; Ravi & Larochelle, 2017; Munkhdalai & Yu, 2017; Snell et al., 2017; Finn et al., 2017; Rusu et al., 2019) .

Learning from a few examples is often difficult and easily leads to over-fitting, especially when no model uncertainty is taken into account.

This issue has been addressed by several recent Bayesian meta-learning approaches that incorporate model uncertainty into prediction, notably LLAMA that is based on Laplace method (Grant et al., 2018) , or PLATIPUS (Finn et al., 2017) , Amortised Meta-learner (Ravi & Beatson, 2019) and VERSA (Gordon et al., 2019 ) that use variational inference (VI).

However, these works have not thoroughly investigated the generalisation errors for unseen samples, resulting in limited theoretical generalisation guarantees.

Moreover, most of these papers are based on variational functions that may not represent well the richness of the underlying distributions.

For instance, a common choice for the variational function relies on the diagonal Gaussian distribution, which can potentially worsen the prediction accuracy given its limited representability.

In this paper, we address the two problems listed above with the following technical novelties: (i) derivation of a rigorous upper-bound for the generalisation errors of few-shot meta-learning using PAC-Bayes framework, and (ii) proposal of a novel variational Bayesian learning based on implicit

The few-shot meta-learning problem is modelled using a hierarchical model that learns a prior p(w i ; θ) using a few data points s

ij )}.

Shaded nodes denote observed variables, while white nodes denote hidden variables.

generative models to facilitate the learning of unseen tasks.

Our evaluation shows that the models trained with our proposed meta-learning algorithm is at the same time well calibrated and accurate, with competitive results in terms of Expected Calibration Error (ECE) and Maximum Calibration Error (MCE), while outperforming state-of-the-art methods in a few-shot classification benchmark (mini-ImageNet).

Our paper is related to Bayesian few-shot meta-learning techniques that have been developed to incorporate uncertainty into model estimation.

LLAMA (Grant et al., 2018) employs the Laplace method to extend the deterministic estimation assumed in MAML to a Gaussian distribution.

However, the need to estimate and invert the Hessian matrix makes this approach computationally challenging for large-scale models, such as deep neural networks.

Variational inference (VI) addresses such scalability issue -remarkable examples of VI-based methods are PLATIPUS , BMAML (Yoon et al., 2018) , Amortised meta-learner (Ravi & Beatson, 2019) and VERSA (Gordon et al., 2019) .

Although these VI-based approaches have demonstrated impressive results in regression, classification as well as reinforcement learning, they do not provide any theoretical guarantee on generalisation errors for unseen samples within a task.

Moreover, the overly-simplified family of diagonal Gaussian distributions used in most of these works limits the expressiveness of the variational approximation, resulting in a less accurate prediction.

Our work is also related to the PAC-Bayes framework used in multi-task learning (Pentina & Lampert, 2014; Amit & Meir, 2018 ) that provides generalisation error bounds with certain confidence levels.

These previously published papers jointly learn a single shared prior and many task-specific posteriors without relating the shared prior to any task-specific posterior.

Hence, these approaches need to store all task-specific posteriors, resulting in un-scalable solutions, especially when the number of tasks is large.

In contrast, our proposed method learns only the shared prior of model parameters and uses that prior to estimate the task-specific posterior through the likelihood function by performing a fixed number of gradient updates.

This proposed variant of amortised inference allows a memory efficient solution, and therefore, more favourable for applications with large number of tasks, such as few-shot meta-learning.

In this section, we first define and formulate the few-shot meta-learning problem.

Subsequently, we derive the generalisation upper-bound based on PAC-Bayes framework.

We then present our proposed approach that employs implicit variational distributions for few-shot meta-learning.

We use the notation of task environment (Baxter, 2000) to describe the unknown distribution p(T ) over a family of tasks, from where tasks are sampled.

Each task T i in this family is indexed by i ∈ {1, ..., T } and associated with a dataset {X i , Y i } consisting of a training/support set {X ij given the small support set for task T i .

We rely on a Bayesian hierarchical model (Grant et al., 2018) as shown in Figure 1 , where w i represents the model parameters for task T i , and θ denotes the meta-parameters shared across all tasks.

For example, in MAML (Finn et al., 2017) , w i are the neural network weights for task T i that is initialised from θ and obtained by performing truncated gradient descent using {X

While the conventional graphical model methods in meta-learning learn the joint probability p(θ, w 1:T |Y 1:T , X 1:T ) (Amit & Meir, 2018, Section A. 3), our objective function for the few-shot meta-learning is to minimise the negative log predictive probability w.r.t.

the meta-parameters θ as follows:

where we simplify the notation by dropping the explicit dependence on X

from the set of conditioning variables (this simplification is adopted throughout the paper).

The predictive probability term inside the expectation in (1) can be expanded by applying the sum rule of probability and lower-bounded by Jensen's inequality:

In practice, the task-specific posterior p(w i |Y (t)

i ; θ) is often intractable, and therefore, approximated by a distribution q(

i , θ).

Given this assumption and the result in (2), the upper bound of the objective function in (1) can be presented as:

where:

Hence, instead of minimising the objective function in (1), we minimise the upper bound in (3).

There are two issues related to the optimisation of this upper bound: (i) the generalisation error for (x

i }, and (ii) how to estimate q(w i ; λ i ) that can approximate the true posterior p(w i |Y (t) i ; θ) accurately, so that we can evaluate and minimise the upper-bound in (3).

We address the generalisation error in Section 3.2 and present a variational method to obtain an expressive variational posterior q(w i ; λ i ) in Section 3.3.

We first introduce the PAC-Bayes bound for the single-task problem in Theorem 1.

Theorem 1 (PAC-Bayes bound for single-task setting (McAllester, 1999) ).

Let D be an arbitrary distribution over an example domain Z. Let H be a hypothesis class, : H × Z → [0, 1] be a loss function, π be a prior distribution over H, and δ ∈ (0, 1).

is an i.i.d.

training set sampled according to D, then for any "posterior" Q over H, the following holds:

where

Theorem 1 indicates that with a high probability, the expected error of an arbitrary posterior Q on data distribution p(z) is upper-bounded by the empirical error plus a complexity regularisation term.

These two terms express the trade-off between fitting data (bias) and regularising model complexity (variance).

Remark 1.

Despite the assumption based on bounded loss function, the PAC-Bayes bound can also be extended to unbounded loss function (McAllester, 1999, Section 5) .

Before presenting the novel bound for few-shot meta-learning, we define some notations.

Recall that m

is the number of samples in the query set {X

The novel bound on the generalisation error for the few-shot meta-learning problem is shown in Theorem 2.

Please refer to Appendix A for the proof.

Theorem 2 (PAC-Bayes bound for few-shot meta-learning in (3)).

For the general error of few-shot meta-learning in (3), the following holds:

Remark 2.

The result derived in Theorem 2 is different from the one in (Amit & Meir, 2018, Theorem 2) .

As mentioned in Section 2, the prior work (Amit & Meir, 2018) does not relate the posterior of model parameters q(w i ; λ i ) to the shared prior p(w i ; θ).

The "hypothesis" in that case is a tuple including the model parameters sampled from the prior and task-specific posterior.

In contrast, our approach is a variant of amortised inference that relates the posterior from the prior and likelihood function by gradient updates (see Section 3.3).

Hence, the "hypothesis" in our case includes the parameters sampled from the task-specific posterior only.

The discrepancy of the "hypothesis" used between the two approaches results in different upper-bounds, particularly at the regularisation term.

Given the result in Theorem 2, the objective function of interest is to minimise the generalisation upper-bound:

As denoted in Section 3.1, q(w i ; λ i ) is a variational posterior that approximates the true posterior p(w i |Y (t) i ; θ) for task T i , and therefore, can be obtained by minimising the following KL divergence:

The resulting cost function (excluding the constant term) in (7) is often known as the variational free energy (VFE).

For simplicity, we denote the cost function as

The first term of VFE can be considered as a regularisation that penalises the difference between the shared prior p(w i ; θ) and the variational task-specific posterior q(w i ; λ i ), while the second term is referred as data-dependent or likelihood cost.

Exactly minimising the cost function in (8) is computationally challenging, so gradient descent is used with θ as the initialisation:

where α t is the learning rate and the truncated gradient descent consists of a single step (the extension to a larger number of steps is trivial).

Given the approximated posterior q(w i ; λ i ) with parameter λ i obtained from (9), we can calculate and optimise the generalisation upper bound in (6) w.r.t.

θ.

In Bayesian statistics, the shared prior p(w i ; θ) represents a modelling assumption, and the variational task-specific posterior q(w i ; λ i ) is a flexible function that can be adjusted to achieve a good trade-off between performance and complexity.

In general, p(w i ; θ) and q(w i ; λ i ) can be modelled using two general types of probabilistic models: prescribed and implicit (Diggle & Gratton, 1984) .

For example, Amortised Meta-learner (Ravi & Beatson, 2019 ) is a prescribed approach where both distributions are assumed to be diagonal Gaussians.

In this paper, we present a more expressive way of implicitly modelling the shared prior and task-specific posterior.

Both distributions p(w i ; θ) and q(w i ; λ i ) are now defined at a more fundamental level whereby data is generated through a stochastic mechanism without specifying parametric distributions.

We use a parameterised model (i.e., a generator G represented by a deep neural network) to model the sample generation from the prior and posterior:

where p(z) is usually denoted by a Gaussian model N (0, I) or a uniform model U(0, 1).

Due to the nature of implicit models, the KL divergence term in (8), in particular the density ratio q(w i ; λ i )

/p(w i ; θ), cannot be evaluated either analytically or symbolically.

We, therefore, propose to employ the probabilistic classification approach (Sugiyama et al., 2012, Chapter 4) to estimate the KL divergence term.

We use a parameterised model -a discriminator D represented by a deep neural network -as a classifier to distinguish different w i sampled from the prior p(w i ; θ) (label 1) or the posterior q(w i ; λ i ) (label 0).

The objective function to train the discriminator D can be written as:

where ω i is the parameters of D for task T i .

Given the discriminator D, the KL divergence term in (8) can be estimated as:

where z (l) ∼ p(z), L t is the number of Monte Carlo samples, and V (., ω i ) is the output of the discriminator D without sigmoid activation.

The variational-free energy in (8) can, therefore, be rewritten as:

One problem that arises when estimating the loss in (13) is how to obtain the local optimal parameters ω * i for the discriminator D. One simple approach is to generate several model parameters w i from the prior p(w i ; θ) and posterior q(w i ; λ i ) following (10) to train D(.; ω i ) by optimising the cost in (11).

The downside is the significant increase in training time and memory usage to store the computational graph to later be used for minimising the upper-bound in (6) w.r.t.

θ.

To overcome this limitation, we propose to meta-learn ω i using MAML (Finn et al., 2017) .

In this scenario, we define ω 0 as the meta-parameters (or initialisation) of ω i .

Within each task, we initialise ω i at ω 0 and use the generated w i from (10) as training data.

This approach leads to our proposed algorithm, named Statistical Implicit Bayesian Meta-Learning (SImBa), shown in Algorithm 1.

Our assumption here is that the discriminator can provide an optimal estimate of the KL divergence term as shown in (12).

This strong theoretical property only holds when the discriminator model is correctly-specified (Sugiyama et al., 2012, Remark 4.7) .

To this end, we employ the universal approximation theorem (Cybenko, 1989; Hornik, 1991) to model the discriminator as a feed-forward fully connected neural network.

We expect that under this modelling approach, the discriminator model is approximately correctly-specified.

Output: meta-parameters θ of the shared prior p(w i ; θ), and discriminator meta-parameters ω 0 1: initialise θ and ω 0 2: while θ not converged do

repeat step 7 to calculate discriminator loss L D (ω i )

end for 15:

Another approach to estimate the KL divergence term in (8) is to use a lower bound of fdivergence (Nguyen et al., 2010; Nowozin et al., 2016) .

There is a difference between the lower bound approach and the probabilistic classification presented in this subsection.

In the former approach, the lower bound of the KL divergence is maximised to tighten the bound.

In the latter approach, a discriminator is trained to minimise the logistic regression loss to estimate the ratio q(w i ; λ i )

/p(w i ; θ), and use Monte Carlo sampling to approximate the KL divergence of interest.

One potential drawback of the implicit modelling used in this paper is the curse of dimensionality, resulting in an expensive computation during training.

This is an active research question when dealing with generative models in general.

This issue can be addressed by encoding the highdimensional data, such as images, to a feature embedding space by supervised-learning on the same training data set (Rusu et al., 2019) .

This strategy reduces the dimension of the input space, leading to smaller generator and discriminator models.

The trade-off lies in the possibility of losing relevant information that can affect the performance on held-out tasks.

It is also worthy noting that our proposed method is easier to train than prior Bayesian few-shot meta-learning Ravi & Beatson, 2019) because we no longer need to estimate the weighting factor of the KL divergence term in (8).

The trade-off of our approach lies in the need to set the significance level δ, but tuning δ is arguably more intuitive than estimating the correct weighting factor for the KL divergence term.

We evaluate SImBa in both few-shot regression and classification problems.

We also compare to prior state-of-art meta-learning methods to show the strengths and weaknesses of SImBa.

The experiment in this subsection is a multi-modal task distribution where half of the data is generated from sinusoidal functions, while the other half is from linear functions .

The details of the experimental setup and additional visualisation results are presented in Appendix B. The results in Figure 2 (leftmost and middle graphs) show that SImBa is able to vary the prediction variance, especially when there is more uncertainty in the training data, while MAML can only output a single value at each data point.

To further evaluate the predictive uncertainty, we employ the reliability diagram based on the quantile calibration for regression (Song et al., 2019 ).

The reliability diagram shows a correlation between predicted and actual probability.

A perfect calibrated model will have its predicted probability equal to the actual probability, and hence, align well with the diagonal y = x. Figure 2 (Right) shows the results for SImBa and some published meta-learning methods.

As expected, Bayesian metalearning approaches, and in particular, Amortised Meta-learner, which relies on diagonal Gaussian distributions, are better calibrated than MAML -a deterministic approach.

However, the averaged slope of the Amortised Meta-learner correlation curve is quite small, implying that its predicted probability is peaked at the mean of the ground-truth distribution with small covariances.

In contrast, SImBa employs a much richer variational distribution, and therefore, resulting in a model with better calibration.

We evaluate SImBa on the N -way k-shot setting, where a meta learner is trained on many related tasks containing N classes with k examples per class.

We use the train-test split that consists of 64 classes for training, 16 for validation, and 20 for testing (Ravi & Larochelle, 2017) .

Please refer to Appendix C for the details of the model used.

Although we target the estimation of model uncertainty, we also present the accuracy of SImBa against the state of the art on mini-ImageNet (Vinyals et al., 2016; Ravi & Larochelle, 2017) .

The results in Table 1 shows that SImBa achieves state-of-the-art in 1-shot setting when the base model is the 4-layer convolutional neural network (CNN) (Vinyals et al., 2016) , and in 5-shot setting when different network architecture is used.

We also show in Appendix D that generators with larger networks tend to classify better.

Similar to the experiment for regression, we use reliability diagrams (Guo et al., 2017) to evaluate the predictive uncertainty.

The reliability diagrams show how well calibrated a model is when testing across many unseen tasks.

A perfectly calibrated model will have its values overlapped with the identity function y = x, indicating that the probability associated with the label prediction is the same as the true probability.

Figures 3a and 3b show the results of SImBa and other Bayesian meta-learning methods.

Visually, the model trained with SImBa shows better calibration than the ones trained with MAML and PLATIPUS, while being competitive to Amortised Meta-learner.

To further evaluate, we compute the expected calibration error (ECE) and maximum calibration error (MCE) (Guo et al., 2017) of the models trained with these methods.

The results plotted in Figure 3c show that the model trained with SImBa has smaller ECE and MCE compared to MAML and PLATIPUS.

SImBa also has lower ECE and competitive MCE compared to Amortised Metalearner, but notice that Amortised Meta-learner has a worse classification result than SImBa, as shown in Table 1 .

Table 1 : The few-shot 5-way classification accuracy results (in percentage, with 95% confidence interval) of SImBa averaged over 600 mini-ImageNet tasks are competitive to the state-of-the-art methods.

SImBa outperforms other prior methods in 1-shot setting when using the standard 4-layer CNN, and 5-shot setting when using non-standard network architectures.

Matching nets (Vinyals et al., 2016) 43.56 ± 0.84 55.31 ± 0.73 Meta-learner LSTM (Ravi & Larochelle, 2017) 43.44 ± 0.77 60.60 ± 0.71 MAML (Finn et al., 2017) 48.70 ± 1.84 63.15 ± 0.91 Prototypical nets (Snell et al., 2017) 1 49.42 ± 0.78 68.20 ± 0.66 LLAMA (Grant et al., 2018) 49.40 ± 1.83 PLATIPUS 50.13 ± 1.86 Amortised ML (Ravi & Beatson, 2019) 45.00 ± 0.60 SImBa 51.01 ± 0.31 63.94 ± 0.43

Relation nets (Sung et al., 2018) 50.44 ± 0.82 65.32 ± 0.70 VERSA (Gordon et al., 2019) 53.40 ± 1.82 67.37 ± 0.86 SNAIL (Mishra et al., 2018) 55.71 ± 0.99 68.88 ± 0.92 adaResNet (Munkhdalai et al., 2018) 56.88 ± 0.62 71.94 ± 0.57 TADAM (Oreshkin et al., 2018) 58.50 ± 0.30 76.70 ± 0.30 LEO (Rusu et al., 2019) 61.76 ± 0.08 77.59 ± 0.12 LGM-Net (Li et al., 2019) 69 .

We introduce and formulate a new Bayesian algorithm for few-shot meta-learning.

The proposed algorithm, SImBa, is based on PAC-Bayes framework which theoretically guarantees prediction generalisation on unseen tasks.

In addition, the proposed method employs a generative approach that implicitly models the shared prior p(w i ; θ) and task-specific posterior q(w i ; λ i ), resulting in more expressive variational approximation compared to the usual diagonal Gaussian methods, such as PLATIPUS or Amortised Meta-learner (Ravi & Beatson, 2019) .

The uncertainty, in the form of the learnt implicit distributions, can introduce more variability into the decision made by the model, resulting in well-calibrated and highly-accurate prediction.

The algorithm can be combined with different base models that are trainable with gradient-based optimisation, and is applicable in regression and classification.

We demonstrate that the algorithm can make reasonable predictions about unseen data in a multi-modal 5-shot learning regression problem, and achieve state-of-the-art calibration and classification results with on few-shot 5-way tasks on mini-ImageNet data set.

First, we present the two auxiliary lemmas that helps to prove Theorem 2.

Lemma 1.

For i = 1 : n, if X i and Y i are random variables, then:

Proof.

The proof is quite direct:

Hence, the proof.

Lemma 2.

For n events A i with i = 1 : n, the following holds:

Proof.

Proof can be done by induction.

For n = 2:

Suppose that it is true for case n:

We prove that this is also true for case (n + 1):

It is, therefore, true for (n + 1), and hence, the proof.

Secondly, we apply the PAC-Bayes bound in Theorem 1 on the task i to obtain an upper-bound for a single task i shown in Corollary 1.

Corollary 1.

For a single task T i in Eq. (3) and δ i ∈ (0, 1), the following holds: Finally, we can employ Lemmas 1 and 2 combined with Corollary 1 to derive the novel upper-bound for few-shot meta-learning setting.

Theorem 2 (PAC-Bayes bound for few-shot meta-learning in (3)).

For the general error of few-shot meta-learning in (3), the following holds:

Proof.

Applying the inequality in Lemma 1 by replacing

where

are defined at Eqs. (4), (3) and (5), respectively.

Applying Lemma 2 the right hand side term of Ineq. (15) gives:

Applying the transitive property for Ineqs. (15), (16) and Corollary 1, and setting δ i = δ/T prove the theorem.

The experiment is carried out with half of the data being generated from sinusoidal functions, while the other half from linear functions.

The amplitude and phase of the sinusoidal functions are uniformly sampled from [0.1, 5] and [0, π], respectively, while the slope and intercept of the lines are sampled from [-3, 3] .

Data is uniformly generated from [-5, 5] , and the corresponding label is added a zero-mean Gaussian noise with a standard deviation of 0.3.

Each task consists of 5 data points used for training (|Y i | = 15).

The base model used in the regression experiment is a three-hidden fully connected layer neural network.

Each hidden layer has 100 hidden units (1 → 40 → 40 → 40 → 1), followed by tanh activation.

No batch normalisation is used.

The generator is a fully connected network with two hidden layers consisting of 256 and 1024 units, respectively (dim(z) → 256 → 1024 → dim(w i )).

The discriminator is also fully connected (dim(w i ) → 512 → dim(z) → 1).

These networks are activated by ReLU, except the last layer of the discriminator is activated by sigmoid function.

No batch normalisation is used across these two networks.

The variational parameters λ i and ω i are estimated by performing five gradient updates with learning rate α t = 0.001 and γ t = 0.001.

The meta-parameters θ and the meta-parameter of the discriminator ω 0 are obtained with Adam (Kingma & Ba, 2015) with fixed step size α v = 10 −4 and γ v = 10 −5 .

At the beginning of training, we clip the gradient when updating λ i with a value of 10, and then gradually increase the clipping value.

After 50,000 tasks, we remove the gradient clipping and continue to train until convergence.

i | = 15N ).

The latent noise z is a 100-dimensional vector sampled from a uniform distribution U(0, 1).

Adam optimiser is employed to optimise both θ and ω 0 .

Please refer to Table 2 for other hyper-parameters used.

This model corresponds to the top part of Table 1.

All input images are down-sampled to 84-by-84 pixels before performing experiments to be consistent with prior few-shot meta-learning works.

The base model is a 4-block CNN, where each block consists of 32 filters with a size of 3-by-3, followed by a batch normalisation and a ReLU activation function.

The generator is a 2-hiddenlayer fully connected network (dim(z) → 256 → 1024 → dim(w i )), where each layer is activated by ReLU without batch normalisation.

The discriminator is also a fully connected network (dim(w i ) → 1024 → 256 → dim(z) → 1) with ReLU activation and without batch normalisation (the last activation function is a sigmoid).

This corresponds to the bottom part of Table 1 .

Here, we employ the features extracted from (Rusu et al., 2019, Section 4.2.2) as the encoding of the input images.

The training for the feature embedding consists of 3 steps.

First, raw input images are down-sampled to 80-by-80 pixels.

Second, a wide residual neural network WRN-28-10 is trained with data and labels from the 64 classes of the training set.

Finally, the intermediate features of 640 dimensions at layer 21 are chosen as the embedding features used for our classification experiments.

The base model used in this experiment is a fully connected network with 1 hidden layer that consists of 128 hidden units (640 → 128 → N ) followed by ReLU activation and batch normalisation.

The generator model is constructed as a 1-hidden layer fully connected network with 512 hidden units, followed by ReLU without batch normalisation.

The discriminator is also a fully connected network (dim(w i ) → 512 → dim(z) → 1) with ReLU without batch normalisation.

To study the effect of network architecture on the classification performance presented in Table 1 , we repeat the classification experiment with the same setup, but different base networks.

We vary the number of hidden units in the base network from 16 to 128, and also increase the size the the hidden layer of the generator from 256 to 512.

The results in Table 3 show that the larger the base network and the generator are, the better the classification accuracy.

<|TLDR|>

@highlight

Bayesian meta-learning using PAC-Bayes framework and implicit prior distributions