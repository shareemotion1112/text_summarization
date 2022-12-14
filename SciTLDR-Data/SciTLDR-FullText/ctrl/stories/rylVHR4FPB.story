Bayesian learning of model parameters in neural networks is important in scenarios where estimates with well-calibrated uncertainty are important.

In this paper, we propose Bayesian quantized networks (BQNs), quantized neural networks (QNNs) for which we learn a posterior distribution over their discrete parameters.

We provide a set of efficient algorithms for learning and prediction in BQNs without the need to sample from their parameters or activations, which not only allows for differentiable learning in quantized models but also reduces the variance in gradients estimation.

We evaluate BQNs on MNIST, Fashion-MNIST and KMNIST classification datasets compared against bootstrap ensemble of QNNs (E-QNN).

We demonstrate BQNs achieve both lower predictive errors and better-calibrated uncertainties than E-QNN (with less than 20% of the negative log-likelihood).

A Bayesian approach to deep learning considers the network's parameters to be random variables and seeks to infer their posterior distribution given the training data.

Models trained this way, called Bayesian neural networks (BNNs) (Wang & Yeung, 2016) , in principle have well-calibrated uncertainties when they make predictions, which is important in scenarios such as active learning and reinforcement learning (Gal, 2016) .

Furthermore, the posterior distribution over the model parameters provides valuable information for evaluation and compression of neural networks.

There are three main challenges in using BNNs: (1) Intractable posterior: Computing and storing the exact posterior distribution over the network weights is intractable due to the complexity and high-dimensionality of deep networks.

(2) Prediction:

Performing a forward pass (a.k.a.

as probabilistic propagation) in a BNN to compute a prediction for an input cannot be performed exactly, since the distribution of hidden activations at each layer is intractable to compute.

(3) Learning:

The classic evidence lower bound (ELBO) learning objective for training BNNs is not amenable to backpropagation as the ELBO is not an explicit function of the output of probabilistic propagation.

These challenges are typically addressed either by making simplifying assumptions about the distributions of the parameters and activations, or by using sampling-based approaches, which are expensive and unreliable (likely to overestimate the uncertainties in predictions).

Our goal is to propose a sampling-free method which uses probabilistic propagation to deterministically learn BNNs.

A seemingly unrelated area of deep learning research is that of quantized neural networks (QNNs), which offer advantages of computational and memory efficiency compared to continuous-valued models.

QNNs, like BNNs, face challenges in training, though for different reasons: (4.1) The non-differentiable activation function is not amenable to backpropagation. (4.2) Gradient updates cease to be meaningful, since the model parameters in QNNs are coarsely quantized.

In this work, we combine the ideas of BNNs and QNNs in a novel way that addresses the aforementioned challenges (1)(2)(3)(4) in training both models.

We propose Bayesian quantized networks (BQNs), models that (like QNNs) have quantized parameters and activations over which they learn (like BNNs) categorical posterior distributions.

BQNs have several appealing properties:

??? BQNs solve challenge (1) due to their use of categorical distributions for their model parameters.

??? BQNs can be trained via sampling-free backpropagation and stochastic gradient ascent of a differentiable lower bound to ELBO, which addresses challenges (2), (3) and (4) above.

??? BQNs leverage efficient tensor operations for probabilistic propagation, further addressing challenge (2).

We show the equivalence between probabilistic propagation in BQNs and tensor contractions (Kolda & Bader, 2009) , and introduce a rank-1 CP tensor decomposition (mean-field approximation) that speeds up the forward pass in BQNs.

??? BQNs provide a tunable trade-off between computational resource and model complexity: using a refined quantization allows for more complex distribution at the cost of more computation.

??? Sampling from a learned BQN provides an alternative way to obtain deterministic QNNs .

In our experiments, we demonstrate the expressive power of BQNs.

We show that BQNs trained using our sampling-free method have much better-calibrated uncertainty compared with the stateof-the-art Bootstrap ensemble of quantized neural networks (E-QNN) trained by Courbariaux et al. (2016) .

More impressively, our trained BQNs achieve comparable log-likelihood against Gaussian Bayesian neural network (BNN) trained with stochastic gradient variational Bayes (SGVB) (Shridhar et al., 2019) (the performance of Gaussian BNNs are expected to be better than BQNs since they allows for continuous random variables).

We further verify that BQNs can be easily used to compress (Bayesian) neural networks and obtain determinstic QNNs.

Finally, we evaluate the effect of mean-field approximation in BQN, by comparing with its Monte-Carlo realizations, where no approximation is used.

We show that our sampling-free probabilistic propagation achieves similar accuracy and log-likelihood -justifying the use of mean-field approximation in BQNs.

In Appendix A, we survey different approaches for training Bayesian neural networks including sampling-free assumed density filtering (Minka, 2001; Soudry et al., 2014; Hern??ndez-Lobato & Adams, 2015; Ghosh et al., 2016) , sampling-based variational inference (Graves, 2011; Blundell et al., 2015; Shridhar et al., 2019) , as well as sampling-free variational inference (Wu et al., 2018) , probabilistic neural networks (Wang et al., 2016; Shekhovtsov & Flach, 2018; Gast & Roth, 2018) , quantized neural network (Han et al., 2015; Courbariaux et al., 2015; Zhu et al., 2016; Kim & Smaragdis, 2016; Zhou et al., 2016; Rastegari et al., 2016; Hubara et al., 2017; Esser et al., 2015; Peters & Welling, 2018; Shayer et al., 2017) , and tensor networks and tensorial neural networks (Grasedyck et al., 2013; Or??s, 2014; Cichocki et al., 2016; 2017; Su et al., 2018; Newman et al., 2018; Robeva & Seigal, 2017) .

??? We propose an alternative evidence lower bound (ELBO) for Bayesian neural networks such that optimization of the variational objective is compatible with the backpropagation algorithm.

??? We introduce Bayesian quantized networks (BQNs), establish a duality between BQNs and hierarchical tensor networks, and show prediction a BQN is equivalent to a series of tensor contractions.

??? We derive a sampling-free approach for both learning and inference in BQNs using probabilistic propagation (analytical inference), achieving better-calibrated uncertainty for the learned models.

??? We develop a set of fast algorithms to enable efficient learning and prediction for BQNs.

Notation.

We use bold letters such as ?? to denote random variables, and non-bold letters such as ?? to denote their realizations.

We abbreviate Pr [?? = ??]

of N data points, we aim to learn a neural network with model parameters ?? that predict the output y ??? Y based on the input x ??? X .

(1) We first solve the learning problem to find an approximate posterior distribution Q(??; ??) over ?? with parameters ?? such that Q(??; ??) ??? Pr[??|D].

(2) We then solve the prediction problem to compute the predictive distribution Pr[y|x, D] for arbitrary input x = x given Q(??; ??).

For notational simplicity, we will omit the conditioning on D and write Pr [y|x, D] as Pr [y|x] in what follows.

In order to address the prediction and learning problems in BNNs, we analyze these models in their general form of probabilistic graphical models (shown in Figure 3b in Appendix B).

Let h (l) , ??

and h (l+1) denote the inputs, model parameters, and (hidden) outputs of the l-th layer respectively.

We assume that ?? (l) 's are layer-wise independent, i.e. Q(??; ??) =

Computing the predictive distribution Pr[y|x, D] with a BNN requires marginalizing over the random variable ??.

The hierarchical structure of BNNs allows this marginalization to be performed in multiple steps sequentially.

In Appendix B, we show that the predictive distribution of h (l+1) given input x = x can be obtained from its preceding layer h (l) by

This iterative process to compute the predictive distributions layer-by-layer sequentially is known as probabilistic propagation (Soudry et al., 2014; Hern??ndez-Lobato & Adams, 2015; Ghosh et al., 2016) .

With this approach, we need to explicitly compute and store each intermediate result

is a function of x).

Therefore, probabilistic propagation is a deterministic process that computes ?? (l+1) as a function of ?? (l) and ?? (l) , which we denote as

Challenge in Sampling-Free Probabilistic Propagation.

If the hidden variables h (l) 's are continuous, Equation (1) generally can not be evaluated in closed form as it is difficult to find a family of parameterized distributions P for h (l) such that h (l+1) remains in P under the operations of a neural network layer.

Therefore most existing methods consider approximations at each layer of probabilistic propagation.

In Section 4, we will show that this issue can be (partly) addressed if we consider the h (l) 's to be discrete random variables, as in a BQN.

Objective Function.

A standard approach to finding a good approximation Q(??; ??) is variational inference, which finds ?? such that the KL-divergence KL(Q(??; ??)||Pr [??|D] ) from Q(??; ??) to Pr[??|D] is minimized.

In Appendix B, we prove that to minimizing the KL-divergence is equivalent to maximizing an objective function known as the evidence lower bound (ELBO), denoted as L(??).

where

Probabilistic Backpropagation.

Optimization in neural networks heavily relies on the gradientbased methods, where the partial derivatives ???L(??)/????? of the objective L(??) w.r.t.

the parameters ?? are obtained by backpropagation.

Formally, if the output produced by a neural network is given by a (sub-)differentiable function g(??), and the objective L(g(??)) is an explicit function of g(??) (and not just an explicit function of ??), then the partial derivatives can be computed by chain rule:

Published as a conference paper at ICLR 2020

The learning problem can then be (approximately) solved by first-order methods, typically stochastic gradient descent/ascent.

Notice that (1) For classification, the function g(??) returns the probabilities after the softmax function, not the categorical label; (2) An additional regularizer R(??) on the parameters will not cause difficulty in backpropagation, given ???R(??)/????? is easily computed.

Challenge in Sampling-Free Probabilistic Backpropagation.

Learning BNNs is not amenable to standard backpropagation because the ELBO objective function L(??) in (4b) is not an explicit (i.e. implicit) function of the predictive distribution g(??) in (4a):

Although L n (??) is a function of ??, it is not an explicit function of g n (??).

Consequently, the chain rule in Equation (3) on which backpropagation is based is not directly applicable.

Alternative Evidence Lower Bound.

We make learning in BNNs amenable to backpropagation by developing a lower bound

is an explicit function of the results from the forward pass.)

With L n (??) in hand, we can (approximately) find ?? by maximizing the alternative objective via gradient-based method:

In Appendix C.1, we proved one feasible L n (??) which only depends on second last output h (L???1) .

) is deterministic given input x and all parameters before the last layer ??

Analytic Forms of L n (??).

While the lower bound in Theorem 3.1 applies to BNNs with arbitrary distributions P on hidden variables h, Q on model parameters ??, and any problem setting (e.g. classification or regression), in practice sampling-free probabilistic backpropagation requires that L n (??) can be analytically evaluated (or further lower bounded) in terms of ?? (L???1) and ?? (L???1) .

This task is nontrivial since it requires redesign of the output layer, i.e. the function of Pr[y|h (L???1) , ?? (L???1) ].

In this paper, we develop two layers for classification and regression tasks, and present the classification case in this section due to space limit.

Since L n (??) involves the last layer only, we omit the superscripts/subsripts of

, x n , y n , and denote them as h, ??, ??, x, y .

with K the number of classes) be the pre-activations of a softmax layer (a.k.a.

logits), and ?? = s ??? R + be a scaling factor that adjusts its scale such that

are pairwise independent (which holds under mean-field approximation) and

The regression case and proofs for both layers are deferred to Appendix C.

While Section 3 provides a general solution to learning in BNNs, the solution relies on the ability to perform probabilistic propagation efficiently.

To address this, we introduce Bayesian quantized networks (BQNs) -BNNs where both hidden units h (l) 's and model parameters ?? (l) 's take discrete values -along with a set of novel algorithms for efficient sampling-free probabilistic propagation in BQNs.

For simplicity of exposition, we assume activations and model parameters take values from the same set Q, and denote the degree of quantization as D = |Q|, (e.g. Q = {???1, 1}, D = 2).

Lemma 4.1 (Probabilistic Propagation in BQNs).

After quantization, the iterative step of probabilistic propagation in Equation (1) is computed with a finite sum instead of an integral:

and a categorically distributed h (l) results in h (l+1) being categorical as well.

The equation holds without any assumption on the operation

Notice all distributions in Equation (7) are represented in high-order tensors:

Suppose there are I input units, J output units, and K model parameters at the l-th layer, then

and h (l+1) ??? Q J , and their distributions are characterized by

respectively.

Therefore, each step in probabilistic propagation is a tensor contraction of three tensors, which establishes the duality between BQNs and hierarchical tensor networks (Robeva & Seigal, 2017) .

Since tensor contractions are differentiable w.r.t.

all inputs, BQNs thus circumvent the difficulties in training QNNs (Courbariaux et al., 2015; Rastegari et al., 2016) , whose outputs are not differentiable w.r.t.

the discrete parameters.

This result is not surprising: if we consider learning in QNNs as an integer programming (IP) problem, solving its Bayesian counterpart is equivalent to the approach to relaxing the problem into a continuous optimization problem (Williamson & Shmoys, 2011) .

Complexity of Exact Propagation.

The computational complexity to evaluate Equation (7) is exponential in the number of random variables O(D IJK ), which is intractable for quantized neural network of any reasonable size.

We thus turn to approximations.

We propose a principled approximation to reduce the computational complexity in probabilistic propagation in BQNs using tensor CP decomposition, which factors an intractable high-order probability tensor into tractable lower-order factors (Grasedyck et al., 2013) .

In this paper, we consider the simplest rank-1 tensor CP decomposition, where the joint distributions of P and Q are fully factorized into products of their marginal distributions, thus equivalent to the mean-field approximation (Wainwright et al., 2008) .

With rank-1 CP decomposition on

, the tensor contraction in (7) reduces to a standard Tucker contraction (Kolda & Bader, 2009 )

where each term of ??

k parameterizes a single categorical variable.

In our implementation, we store the parameters in their log-space, i.e. Q(??

Fan-in Number E. In a practical model, for the l-th layer, an output unit h k } according to the connectivity pattern in the layer.

We denote the set of dependent input units and parameters for h , and define the fan-in number E for the layer as max j I

Complexity of Approximate Propagation.

The approximate propagation reduces the computational complexity from O(D IJK ) to O(JD E ), which is linear in the number of output units J if we assume the fan-in number E to be a constant (i.e. E is not proportional to I).

Different types of network layers have different fan-in numbers E, and for those layers with E greater than a small constant, Equation (8) is inefficient since the complexity grows exponential in E. Therefore in this part, we devise fast(er) algorithms to further lower the complexity.

Small Fan-in Layers: Direct Tensor Contraction.

If E is small, we implement the approximate propagation through tensor contraction in Equation (8).

The computational complexity is O(JD E ) as discussed previously.

See Appendix D.1 for a detailed discussion.

Medium Fan-in Layers: Discrete Fourier Transform.

If E is medium, we implement approximate propagation through fast Fourier transform since summation of discrete random variables is equivalent to convolution between their probability mass function.

See Appendix D.2 for details.

With the fast Fourier transform, the computational complexity is reduced to O(JE 2 D log(ED)).

Large Fan-in Layers: Lyapunov Central Limit Theorem.

In a typical linear layer, the fan-in E is large, and a super-quadratic algorithm using fast Fourier transform is still computational expensive.

Therefore, we derive a faster algorithm based on the Lyapunov central limit theorem (See App D.3) With CLT, the computational complexity is further reduced to O(JED).

Remarks: Depending on the fan-in numbers E, we adopt CLT for linear layers with sufficiently large E such as fully connected layers and convolutional layers; DFT for those with medium E such as average pooling layers and depth-wise layers; and direct tensor contraction for those with small E such as shortcut layers and nonlinear layers.

In this section, we demonstrate the effectiveness of BQNs on the MNIST, Fashion-MNIST, KM-NIST and CIFAR10 classification datasets.

We evaluate our BQNs with both multi-layer perceptron (MLP) and convolutional neural network (CNN) models.

In training, each image is augmented by a random shift within 2 pixels (with an additional random flipping for CIFAR10), and no augmentation is used in test.

In the experiments, we consider a class of quantized neural networks, with both binary weights and activations (i.e. Q = {???1, 1}) with sign activations ??(??) = sign(??).

For BQNs, the distribution parameters ?? are initialized by Xavier's uniform initializer, and all models are trained by ADAM optimizer (Kingma & Ba, 2014) for 100 epochs (and 300 epochs for CIFAR10) with batch size 100 and initial learning rate 10 ???2 , which decays by 0.98 per epoch.

Table 1 : Comparison of performance of BQNs against the baseline E-QNN.

Each E-QNN is an ensemble of 10 networks, which are trained individually and but make predictions jointly.

We report both NLL (which accounts for prediction uncertainty) and 0-1 test error (which doesn't account for prediction uncertainty).

All the numbers are averages over 10 runs with different seeds, the standard deviation are exhibited following the ?? sign.

Training Objective of BQNs.

To allow for customized level of uncertainty in the learned Bayesian models, we introduce a regularization coefficient ?? in the alternative ELBO proposed in Equation (5) (i.e. a lower bound of the likelihood), and train the BQNs by maximizing the following objective:

where ?? controls the uncertainty level, i.e. the importance weight of the prior over the training set.

Baselines.

(1) We compare our BQN against the baseline -Bootstrap ensemble of quantized neural networks (E-QNN).

Each member in the ensemble is trained in a non-Bayesian way (Courbariaux et al., 2016) , and jointly make the prediction by averaging over the logits from all members.

Note Evaluation of BQNs.

While 0-1 test error is a popular metric to measure the predictive performance, it is too coarse a metric to assess the uncertainty in decision making (for example it does not account for how badly the wrong predictions are).

Therefore, we will mainly use the negative log-likelihood (NLL) to measure the predictive performance in the experiments.

Once a BQN is trained (i.e. an approximate posterior Q(??) is learned), we consider three modes to evaluate the behavior of the model: (1) analytic inference (AI), (2) Monte Carlo (MC) sampling and (3) Maximum a Posterior (MAP) estimation:

1.

In analytic inference (AI, i.e. our proposed method), we analytically integrate over Q(??) to obtain the predictive distribution as in the training phase.

Notice that the exact NLL is not accessible with probabilistic propagation (which is why we propose an alternative ELBO in Equation (5)), we will report an upper bound of the NLL in this mode.

2.

In MC sampling, S sets of model parameters are drawn independently from the posterior posterior ?? s ??? Q(??), ???s ??? [S], and the forward propagation is performed as in (non-Bayesian) quantized neural network for each set ?? s , followed by an average over the model outputs.

The difference between analytic inference and MC sampling will be used to evaluate (a) the effect of mean-field approximation and (b) the tightness of the our proposed alternative ELBO.

3.

MAP estimation is similar to MC sampling, except that only one set of model parameters ?? is obtained ?? = arg max ?? Q(??).

We will exhibit our model's ability to compress a Bayesian neural network by comparing MAP estimation of our BQN with non-Bayesian QNN.

Expressive Power and Uncertainty Calibration in BQNs.

We report the performance via all evaluations of our BQN models against the Ensemble-QNN in Table 1 and Figure 1.

(1) Compared to E-QNNs, our BQNs have significantly lower NLL and smaller predictive error (except for Fashion-MNIST with architecture CNN).

(2) As we can observe in Figure 1 , BQNs impressively achieve comparable NLL to continuous-valued BNN, with slightly higher test error.

As our model parameters only take values {???1, 1}, small degradation in predictive accuracy is expected.

Evaluations of Mean-field Approximation and Tightness of the Alternative ELBO.

If analytic inference (by probabilistic propagation) were computed exactly, the evaluation metrics would have been equal to the ones with MC sampling (with infinite samples).

Therefore we can evaluate the approximations in probabilistic propagation, namely mean-field approximation in Equation (8) and relaxation of the original ELBO in Equation (5), by measuring the gap between analytic inference and MC sampling.

As shown in Figure 2 , such gaps are small for all scenarios, which justifies the approximations we use in BQNs.

To further decouple these two factors of mean-field approximation and relaxation of the original ELBO, we vary the regularization coefficient ?? in the learning objective.

(1) For ?? = 0 (where the prior term is removed), the models are forced to become deterministic during training.

Since the deterministic models do not have mean-field approximation in the forward pass, the gap between analytic inference and MC-sampling reflects the tightness of our alternative ELBO.

(2) As ?? increases, the gaps increases slightly as well, which shows that the mean-field approximation becomes slightly less accurate with higher learned uncertainty in the model.

Table 3 : Bayesian Model compression through direct training of Ensemble-QNN vs a Monte-Carlo sampling on our proposed BQN.

Each ensemble consists of 5 quantized neural networks, and for fair comparison we use 5 samples for Monte-Carlo evaluation.

All the numbers are averages over 10 runs with different seeds, the standard deviation are exhibited following the ?? sign.

interpreted as another approach to compress a BQN, which reduces the original size to its S/64 (with the same number of bits as an ensemble of S QNNs).

In Tables 2 and 3 , we compare the models by both approaches to their counterparts (a single QNN for MAP, and E-QNN for MC sampling) trained from scratch as in Courbariaux et al. (2016) .

For both approaches, our compressed models outperform their counterparts (in NLL) .

We attribute this to two factors: (a) QNNs are not trained in a Bayesian way, therefore the uncertainty is not well calibrated; and (b) Non-differentiable QNNs are unstable to train.

Our compression approaches via BQNs simultaneously solve both problems.

We present a sampling-free, backpropagation-compatible, variational-inference-based approach for learning Bayesian quantized neural networks (BQNs).

We develop a suite of algorithms for efficient inference in BQNs such that our approach scales to large problems.

We evaluate our BQNs by Monte-Carlo sampling, which proves that our approach is able to learn a proper posterior distribution on QNNs.

Furthermore, we show that our approach can also be used to learn (ensemble) QNNs by taking maximum a posterior (or sampling from) the posterior distribution.

assuming g n (??) can be (approximately) computed by sampling-free probabilistic propagation as in Section 2.

However, this approach has two major limitations: (a) the Bayes' rule needed to be derived case by case, and analytic rule for most common cases are not known yet.

(b) it is not compatible to modern optimization methods (such as SGD or ADAM) as the optimization is solved analytically for each data point, therefore difficult to cope with large-scale models.

(2) Sampling-based Variational inference (SVI), formulates an optimization problem and solves it approximately via stochastic gradient descent (SGD).

The most popular method among all is, Stochastic Gradient Variational Bayes (SGVB), which approximates L n (??) by the average of multiple samples (Graves, 2011; Blundell et al., 2015; Shridhar et al., 2019) .

Before each step of learning or prediction, a number of independent samples of the model parameters {?? s } S s=1 are drawn according to the current estimate of Q, i.e. ?? s ??? Q, by which the predictive function g n (??) and the loss L n (??) can be approximated by

where f n (??) = Pr[y n |x n , ??] denotes the predictive function given a specific realization ?? of the model parameters.

The gradients of L n (??) can now be approximated as

This approach has multiple drawbacks: (a) Repeated sampling suffers from high variance, besides being computationally expensive in both learning and prediction phases; (b) While g n (??) is differentiable w.r.t.

??, f n (??) may not be differentiable w.r.t.

??.

One such example is quantized neural networks, whose backpropagation is approximated by straight through estimator (Bengio et al., 2013

Our approach considers a wider scope of problem settings, where the model could be stochastic, i.e.

] is an arbitrary function.

Furthermore, Wu et al. (2018) considers the case that all parameters ?? are Gaussian distributed, whose sampling-free probabilistic propagation requires complicated approximation (Shekhovtsov & Flach, 2018) .

Quantized Neural Networks These models can be categorized into two classes: (1) Partially quantized networks, where only weights are discretized (Han et al., 2015; Zhu et al., 2016) ; (2) Fully quantized networks, where both weights and hidden units are quantized (Courbariaux et al., 2015; Kim & Smaragdis, 2016; Zhou et al., 2016; Rastegari et al., 2016; Hubara et al., 2017) .

While both classes provide compact size, low-precision neural network models, fully quantized networks further enjoy fast computation provided by specialized bit-wise operations.

In general, quantized neural networks are difficult to train due to their non-differentiability.

Gradient descent by backpropagation is approximated by either straight-through estimators (Bengio et al., 2013) or probabilistic methods (Esser et al., 2015; Shayer et al., 2017; Peters & Welling, 2018) .

Unlike these papers, we focus on Bayesian learning of fully quantized networks in this paper.

Optimization of quantized neural networks typically requires dedicated loss function, learning scheduling and initialization.

For example, Peters & Welling (2018) considers pre-training of a continuous-valued neural network as the initialization.

Since our approach considers learning from scratch (with an uniform initialization), the performance could be inferior to prior works in terms of absolute accuracy.

Tensor Networks and Tensorial Neural Networks Tensor networks (TNs) are widely used in numerical analysis (Grasedyck et al., 2013) , quantum physiscs (Or??s, 2014), and recently machine learning (Cichocki et al., 2016; 2017) to model interactions among multi-dimensional random objects.

Various tensorial neural networks (TNNs) (Su et al., 2018; Newman et al., 2018) have been proposed that reduce the size of neural networks by replacing the linear layers with TNs.

Recently, (Robeva & Seigal, 2017) points out the duality between probabilistic graphical models (PGMs) and TNs.

I.e. there exists a bijection between PGMs and TNs.

Our paper advances this line of thinking by connecting hierarchical Bayesian models (e.g. Bayesian neural networks) and hierarchical TNs.

The problem settings of general Bayesian model and Bayesian neural networks for supervised learning are illustrated in Figures 3a and 3b using graphical models.

...

General Bayesian model Formally, the graphical model in Figure 3a implies the joint distribution of the model parameters ??, the observed dataset D = {(x n , y n )} N n=1 and any unseen data point (x, y) is factorized as follows: [y|x, ??] .

In other words, we assume that (1) the samples (x n , y n )'s (and unseen data point (x, y)) are are identical and independent distributed according to the same data distribution; and (2) x n (or x) and ?? together predict the output y n (or y) according to the same conditional distribution.

Notice that the factorization above also implies the following equations:

With these implications, the posterior predictive distribution Pr[y|x, D] can now expanded as:

d?? (17) where we approximate the posterior distribution Pr[??|D] by a parameterized distribution Q(??; ??).

const.

( 22) where (1) L n (??) is the expected log-likelihood, which reflects the predictive performance of the Bayesian model on the data point (x n , y n ); and (2) R(??) is the KL-divergence between Q(??; ??) and its prior Pr[??], which reduces to entropy H(Q) if the prior of ?? follows a uniform distribution.

Hierarchical Bayesian Model A Bayesian neural network can be considered as a hierarchical Bayesian model depicted in Figure 3b , which further satisfies the following two assumptions:

Assumption B.1 (Independence of Model Parameters ?? (l) ).

The approximate posterior Q(??; ??) over the model parameters ?? are partitioned into L disjoint and statistically independent layers {?? (l) } L???1 l=0 (where each ?? (l) parameterizes ?? (l) in the l-th layer) such that:

satisfy the Markov property that h (l+1) depends on the input x only through its previous layer h (l) :

where we use short-hand notations h ( : l) and ?? ( : l) to represent the sets of previous layers {h (k) } l k=0

and {?? (k) } l k=0 .

For consistency, we denote h (0) = x and h (L) = y.

Proof of probabilistic prorogation Based on the two assumptions above, we provide a proof for probabilistic propagation in Equation (1) as follows:

C ALTERNATIVE EVIDENCE LOWER BOUND AND ITS ANALYTIC FORMS C.1 ALTERNATIVE EVIDENCE LOWER BOUND (PROOF FOR THEOREM 3.1)

The steps to prove the inequality (6) almost follow the ones for probabilistic propagation above:

where the key is the Jensen's inequality

is not random variable (typical for an output layer), L n (??) can be simplified as:

where we write Pr[

(L???1) can be obtained by differentiating over Equation (37), while other gradients ???L n (??)/?? ( : L???2) further obtained by chain rule:

which requires us to compute ???L n (??)/????? (L???1) and ?????

can be derived from Equation (37), ????? (L???1) /????? ( : L???2) can be obtained by backpropagating outputs of the (L ??? 2) th layer obtained from probabilistic propagation in Equation (1).

In other words:

is a function of all parameters from previous layers ?? ( : L???2) , and if each step

can be obtained by iterative chain rule.

In this part, we first prove the alternative evidence lower bound (ELBO) for Bayesian neural networks with softmax function as their last layers.

Subsequently, we derive the corresponding backpropagation rule for the softmax layer.

Finally, we show a method based on Taylor's expansion to approximately evaluate a softmax layer without Monte Carlo sampling.

Theorem C.1 (Analytic Form of L n (??) for Classification).

Let h ??? R K (with K the number of classes) be the pre-activations of a softmax layer (a.k.a.

logits), and ?? = s ??? R + be a scaling factor that adjusts its scale such that

are pairwise independent (which holds under mean-field approximation) and

) and s is a deterministic parameter.

Then L n (??) can be further upper bound by the following analytic form:

Proof.

The lower bound follows by plugging Pr [y|h, s] and Pr[h k |x] into Equation (6).

where the last equation follows

where the under-braced term is unity since it takes the form of Gaussian distribution.

From Equation (43) to (44), we use the Jensen's inequality to achieve a lower bound for integral of log-sum-exp.

The bound can be tighten with advanced techniques in Khan (2012).

Derivatives of L n (??) in (39) To use probabilistic backpropagation to obtain the gradients w.r.t.

the parameters from previous layers, we first need to obtain the derivatives w.r.t.

Furthermore, the scale s can be (optionally) updated along with other parameters using the gradient

Prediction with Softmax Layer Once we learn the parameters for the Bayesian neural network, in principle we can compute the predictive distribution of y by evaluating the following equation:

where we denote the softmax function as

Unfortunately, the equation above can not be computed in closed form.

The most straight-forward work-around is to approximate the integral by Monte Carlo sampling: for each h k we draw S samples {h

independently and compute the prediction:

Despite its conceptual simplicity, Monte Carlo method suffers from expensive computation and high variance in estimation.

Instead, we propose an economical estimate based on Taylor's expansion.

First, we expand the function c (h) by Taylor's series at the point ?? (up to the second order):

Before we derive the forms of these derivatives, we first show the terms of odd orders do not contribute to the expectation.

For example, if c (h) is approximated by its first two terms (i.e. a linear function), Equation (54) can be written as

where the second term is zero by the symmetry of Pr[h k |x] around ?? k (or simply the definition of ?? k 's).

Therefore, the first-order approximation results exactly in a (deterministic) softmax function of the mean vector ??. In order to incorporate the variance into the approximation, we will need to derive the exact forms of the derivatives of c (h).

Specifically, the first-order derivatives are obtained from the definition of c (h).

and subsequently the second-order derivatives from the first ones:

with these derivatives we can compute the second-order approximation as

The equation above can be further written in vector form as:

In this part, we develop an alternative evidence lower bound (ELBO) for Bayesian neural networks with Gaussian output layers, and derive the corresponding gradients for backpropagation.

Despite the difficulty to obtain an analytical predictive distribution for the output, we show that its central moments can be easily computed given the learned parameters.

Theorem C.2 (Analytic Form of L n (??) for Regression).

Let h ??? R I be the output of last hidden layer (with I the number of hidden units), and ?? = (w, s) ??? R I ?? R + be the parameters that define the predictive distribution over output y as

Suppose the hidden units {h k } K k=1 are pairwise independent (which holds under mean-field approximation), and each h i has mean ?? i and variance ?? i , then L n (??) takes an analytic form:

where ( Proof.

The Equation (68) is obtained by plugging Pr[y|h; w, s] into Equation (6).

where the long summation in the first term can be further simplified with notations of ?? and ??:

where w ???2 denotes element-wise square, i.e.

Derivatives of L n (??) in Equation (68) It is not difficult to show that the gradient of L n (??) can be backpropagated through the last layer.

by computing derivatives of L n (??) w.r.t.

?? and ??:

Furthermore, the parameters {w, s} can be updated along with other parameters with their gradients:

Prediction with Gaussian Layer Once we determine the parameters for the last layer, in principle we can compute the predictive distribution Pr[y|x] for the output y given the input x according to

Unfortunately, exact computation of the equation above for arbitrary output value y is intractable in general.

However, the central moments of the predictive distribution Pr[y|x] are easily evaluated.

Consider we interpret the prediction as y = w h + , where ??? N (0, s), its mean and variance can be easily computed as

Furthermore, if we denote the (normalized) skewness and kurtosis of h i as ?? i and ?? i :

Then the (normalized) skewness and kurtosis of the prediction y are also easily computed with the

In this section, we present fast(er) algorithms for sampling-free probabilistic propagation (i.e. evaluating Equation (8)).

According to Section 4, we divide this section into three parts, each part for a specific range of fan-in numbers E.

If E is small, tensor contraction in Equation (8) is immediately applicable.

Representative layers of small E are shortcut layer (a.k.a.

skip-connection) and what we name as depth-wise layers.

Shortcut Layer With a skip connection, the output h (l+1) is an addition of two previous layers h (l) and h (m) .

Therefore and the distribution of h (l+1) can be directly computed as

Depth-wise Layers In a depth-wise layer, each output unit h

Depth-wise layers include dropout layers (where ?? (l) are dropout rates), nonlinear layers (where ?? (l) are threshold values) or element-wise product layers (where ?? (l) are the weights).

For both shortcut and depth-wise layers, the time complexity is O(JD 2 ) since E <= 2.

In neural networks, representative layers with medium fan-in number E are pooling layers, where each output unit depends on a medium number of input units.

Typically, the special structure of pooling layers allows for faster algorithm than computing Equation (8) directly.

Max and Probabilistic Pooling For each output, (1) a max pooling layer picks the maximum value from corresponding inputs, i.e. h

i , while (2) a probabilistic pooling layer selects the value the inputs following a categorical distribution, i.e. Pr[h

For both cases, the predictive distribution of h (l+1) j can be computed as

Prob: P (h

where

is the culminative mass function of P .

Complexities for both layers are O(ID).

Average Pooling and Depth-wise Convolutional Layer Both layers require additions of a medium number of inputs.

We prove a convolution theorem for discrete random variables and show that discrete Fourier transform (DFT) (with fast Fourier transform (FFT)) can accelerate the additive computation.

We also derive its backpropagation rule for compatibility of gradient-based learning.

Then C v (f ) is the element-wise product of all Fourier transforms C ui (f ), i.e.

Proof.

We only prove the theorem for two discrete random variable, and the extension to multiple variables can be proved using induction.

Now consider

, where b = b 1 + b 2 and B = B 1 + B 2 .

Denote the probability vectors of u 1 , u 2 and v as P 1 ??? B1???b1 , P 2 ??? B2???b2 and P ??? B???b respectively, then the entries in P are computed with P 1 and P 2 by standard convolution as follows:

The relation above is usually denoted as P = P 1 * P 2 , where * is the symbol for convolution.

Now define the characteristic functions C, C 1 , and C 2 as the discrete Fourier transform (DFT) of the probability vectors P , P 1 and P 2 respectively:

where R controls the resolution of the Fourier transform (typically chosen as R = B ??? b + 1, i.e. the range of possible values).

In this case, the characteristic functions are complex vectors of same length R, i.e. C, C 1 , C 2 ??? C R , and we denote the (functional) mappings as C = F(P ) and C i = F i (P i ).

Given a characteristic function, its original probability vector can be recovered by inverse discrete Fourier transform (IDFT):

which we denote the inverse mapping as P = F ???1 (C) and P i = F ???1 i (C i ).

Now we plug the convolution in Equation (91) into the characteristic function C(f ) in (92a) and rearrange accordingly:

The equation above can therefore be written as C = C 1 ???C 2 , where we use ??? to denote element-wise product.

Thus, we have shown summation of discrete random variables corresponds to element-wise product of their characteristic functions.

With the theorem, addition of E discrete random variables can be computed efficiently as follows

where F denotes the Fourier transforms in Equations (93a) and (93b).

If FFT is used in computing all DFT, the computational complexity of Equation (99) is O(ER log R) = O(E 2 D log(ED)) (since R = O(ED)), compared to O(D E ) with direct tensor contraction.

Backpropagation When fast Fourier transform is used to accelerate additions in Bayesian quantized network, we need to derive the corresponding backpropagation rule, i.e. equations that relate ???L/???P to {???L/???P i } I i=1 .

For this purpose, we break the computation in Equation (99) into three steps, and compute the derivative for each of these steps.

where in (100b) we use C/C i to denote element-wise division.

Since P i lies into real domain, we need to project the gradients back to real number in (100c).

Putting all steps together:

In this part, we show that Lyapunov central limit approximation (Lyapunov CLT) accelerates probabilistic propagation in linear layers.

For simplicity, we consider fully-connected layer in the derivations, but the results can be easily extended to types of convolutional layers.

We conclude this part by deriving the corresponding backpropagation rules for the algorithm.

Linear Layers Linear layers (followed by a nonlinear transformations ??(??)) are the most important building blocks in neural networks, which include fully-connected and convolutional layers.

A linear layer is parameterized by a set of vectors ?? (l) 's, and maps

where u .

Let v j = ??(??? j ) = ??( i???I(j) ?? ji u i ) be an activation of a linear layer followed by nonlinearity ??.

Suppose both inputs {u i } i???I and parameters {?? ji } i???I(j) have bounded variance, then for sufficiently large |I(j)|, the distribution of??? j converges to a Gaussian distribution N (?? j ,?? j ) with mean and variance as Published as a conference paper at ICLR 2020 for CNN, we use a 4-layers network with two 5 ?? 5 convolutional layers with 64 channels followed by 2 ?? 2 average pooling, and two fully-connected layers with 1024 hidden units.

(2) For CIFAR10, we evaluate our models on a smaller version of VGG (Peters & Welling, 2018) , which consists of 6 convolutional layers and 2 fully-connected layers: 2 x 128C3 -MP2 -2 x 256C3 -MP2 -2 x 512C3 -MP2 -1024FC -SM10.

Table 4 : Performance of different networks in terms of RMSE.

The numbers for BQN are averages over 10 runs with different seeds, the standard deviation are exhibited following the ?? sign.

The results for PBP, EBP are from Ghosh et al. (2016) , and the one for NPN is from (Wang et al., 2016) .

10

<|TLDR|>

@highlight

We propose Bayesian quantized networks, for which we learn a posterior distribution over their quantized parameters.