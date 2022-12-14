Meta-learning allows an intelligent agent to leverage prior learning episodes as a basis for quickly improving performance on a novel task.

Bayesian hierarchical modeling provides a theoretical framework for formalizing meta-learning as inference for a set of parameters that are shared across tasks.

Here, we reformulate the model-agnostic meta-learning algorithm (MAML) of Finn et al. (2017) as a method for probabilistic inference in a hierarchical Bayesian model.

In contrast to prior methods for meta-learning via hierarchical Bayes, MAML is naturally applicable to complex function approximators through its use of a scalable gradient descent procedure for posterior inference.

Furthermore, the identification of MAML as hierarchical Bayes provides a way to understand the algorithm’s operation as a meta-learning procedure, as well as an opportunity to make use of computational strategies for efficient inference.

We use this opportunity to propose an improvement to the MAML algorithm that makes use of techniques from approximate inference and curvature estimation.

A remarkable aspect of human intelligence is the ability to quickly solve a novel problem and to be able to do so even in the face of limited experience in a novel domain.

Such fast adaptation is made possible by leveraging prior learning experience in order to improve the efficiency of later learning.

This capacity for meta-learning also has the potential to enable an artificially intelligent agent to learn more efficiently in situations with little available data or limited computational resources BID45 BID4 BID37 .In machine learning, meta-learning is formulated as the extraction of domain-general information that can act as an inductive bias to improve learning efficiency in novel tasks (Caruana, 1998; BID52 .

This inductive bias has been implemented in various ways: as learned hyperparameters in a hierarchical Bayesian model that regularize task-specific parameters BID18 , as a learned metric space in which to group neighbors BID7 , as a trained recurrent neural network that allows encoding and retrieval of episodic information BID43 , or as an optimization algorithm with learned parameters BID45 BID3 .The model-agnostic meta-learning (MAML) of BID12 is an instance of a learned optimization procedure that directly optimizes the standard gradient descent rule.

The algorithm estimates an initial parameter set to be shared among the task-specific models; the intuition is that gradient descent from the learned initialization provides a favorable inductive bias for fast adaptation.

However, this inductive bias has been evaluated only empirically in prior work BID12 .In this work, we present a novel derivation of and a novel extension to MAML, illustrating that this algorithm can be understood as inference for the parameters of a prior distribution in a hierarchical Bayesian model.

The learned prior allows for quick adaptation to unseen tasks on the basis of an implicit predictive density over task-specific parameters.

The reinterpretation as hierarchical Bayes gives a principled statistical motivation for MAML as a meta-learning algorithm, and sheds light on the reasons for its favorable performance even among methods with significantly more parameters.

More importantly, by casting gradient-based meta-learning within a Bayesian framework, we are able to improve MAML by taking insights from Bayesian posterior estimation as novel augmentations to the gradient-based meta-learning procedure.

We experimentally demonstrate that this enables better performance on a few-shot learning benchmark.

The goal of a meta-learner is to extract task-general knowledge through the experience of solving a number of related tasks.

By using this learned prior knowledge, the learner has the potential to quickly adapt to novel tasks even in the face of limited data or limited computation time.

Formally, we consider a dataset D that defines a distribution over a family of tasks T .

These tasks share some common structure such that learning to solve a single task has the potential to aid in solving another.

Each task T defines a distribution over data points x, which we assume in this work to consist of inputs and either regression targets or classification labels y in a supervised learning problem (although this assumption can be relaxed to include reinforcement learning problems; e.g., see BID12 .

The objective of the meta-learner is to be able to minimize a task-specific performance metric associated with any given unseen task from the dataset given even only a small amount of data from the task; i.e., to be capable of fast adaptation to a novel task.

In the following subsections, we discuss two ways of formulating a solution to the meta-learning problem: gradient-based hyperparameter optimization and probabilistic inference in a hierarchical Bayesian model.

These approaches were developed orthogonally, but, in Section 3.1, we draw a novel connection between the two.

A parametric meta-learner aims to find some shared parameters θ that make it easier to find the right task-specific parameters φ when faced with a novel task.

A variety of meta-learners that employ gradient methods for task-specific fast adaptation have been proposed (e.g., BID2 BID26 BID57 .

MAML BID12 is distinct in that it provides a gradient-based meta-learning procedure that employs a single additional parameter (the meta-learning rate) and operates on the same parameter space for both meta-learning and fast adaptation.

These are necessary features for the equivalence we show in Section 3.1.To address the meta-learning problem, MAML estimates the parameters θ of a set of models so that when one or a few batch gradient descent steps are taken from the initialization at θ given a small sample of task data x j 1 , . . .

, x j N ∼ p T j (x) each model has good generalization performance on another sample x j N +1 , . . .

, x j N +M ∼ p T j (x) from the same task.

The MAML objective in a maximum likelihood setting is DISPLAYFORM0 where we use φ j to denote the updated parameters after taking a single batch gradient descent step from the initialization at θ with step size α on the negative log-likelihood associated with the task T j .

Note that since φ j is an iterate of a gradient descent procedure that starts from θ, each φ j is of the same dimensionality as θ.

We refer to the inner gradient descent procedure that computes φ j as fast adaptation.

The computational graph of MAML is given in Figure 1 (left).

An alternative way to formulate meta-learning is as a problem of probabilistic inference in the hierarchical model depicted in Figure 1 (right).

In particular, in the case of meta-learning, each task-specific parameter φ j is distinct from but should influence the estimation of the parameters {φ j | j = j} from other tasks.

We can capture this intuition by introducing a meta-level parameter θ on which each task-specific parameter is statistically dependent.

With this formulation, the mutual dependence of the task-specific parameters φ j is realized only through their individual dependence DISPLAYFORM0 The computational graph of the MAML BID12 algorithm covered in Section 2.1.Straight arrows denote deterministic computations and crooked arrows denote sampling operations. (Right) The probabilistic graphical model for which MAML provides an inference procedure as described in Section 3.1.

In each figure, plates denote repeated computations (left) or factorization (right) across independent and identically distributed samples.on the meta-level parameters θ.

As such, estimating θ provides a way to constrain the estimation of each of the φ j .Given some data in a multi-task setting, we may estimate θ by integrating out the task-specific parameters to form the marginal likelihood of the data.

Formally, grouping all of the data from each of the tasks as X and again denoting by x j 1 , . . .

, x j N a sample from task T j , the marginal likelihood of the observed data is given by DISPLAYFORM1 Maximizing FORMULA2 as a function of θ gives a point estimate for θ, an instance of a method known as empirical Bayes BID5 BID15 due to its use of the data to estimate the parameters of the prior distribution.

Hierarchical Bayesian models have a long history of use in both transfer learning and domain adaptation (e.g., BID25 BID58 BID14 BID8 BID56 .

However, the formulation of meta-learning as hierarchical Bayes does not automatically provide an inference procedure, and furthermore, there is no guarantee that inference is tractable for expressive models with many parameters such as deep neural networks.

In this section, we connect the two independent approaches of Section 2.1 and Section 2.2 by showing that MAML can be understood as empirical Bayes in a hierarchical probabilistic model.

Furthermore, we build on this understanding by showing that a choice of update rule for the taskspecific parameters φ j (i.e., a choice of inner-loop optimizer) corresponds to a choice of prior over task-specific parameters, p( φ j | θ ).

In general, when performing empirical Bayes, the marginalization over task-specific parameters φ j in FORMULA2 is not tractable to compute exactly.

To avoid this issue, we can consider an approximation that makes use of a point estimateφ j instead of performing the integration over φ in (2).

Usingφ j as an estimator for each φ j , we may write the negative logarithm of the marginal likelihood as FORMULA3 recovers the unscaled form of the one-step MAML objective in (1).

This tells us that the MAML objective is equivalent to a maximization with respect to the meta-level parameters θ of the marginal likelihood p( X | θ ), where a point estimate for each task-specific parameter φ j is computed via one or a few steps of gradient descent.

By taking only a few steps from the initialization at θ, the point estimateφ j trades off DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 Algorithm 2: Model-agnostic meta-learning as hierarchical Bayesian inference.

The choices of the subroutine ML-· · · that we consider are defined in Subroutine 3 and Subroutine 4.

DISPLAYFORM3 Subroutine 3: Subroutine for computing a point estimateφ using truncated gradient descent to approximate the marginal negative log likelihood (NLL).minimizing the fast adaptation objective −

log p( x j 1 , . . .

, x j N | θ ) with staying close in value to the parameter initialization θ.

We can formalize this trade-off by considering the linear regression case.

Recall that the maximum a posteriori (MAP) estimate of φ j corresponds to the global mode of the posterior DISPLAYFORM4 In the case of a linear model, early stopping of an iterative gradient descent procedure to estimate φ j is exactly equivalent to MAP estimation of φ j under the assumption of a prior that depends on the number of descent steps as well as the direction in which each step is taken.

In particular, write the input examples as X and the vector of regression targets as y, omit the task index from φ, and consider the gradient descent update DISPLAYFORM5 for iteration index k and learning rate α ∈ R + .

Santos (1996) shows that, starting from φ (0) = θ, φ (k) in (4) solves the regularized linear least squares problem DISPLAYFORM6 with Q-norm defined by z Q = z T Q −1 z for a symmetric positive definite matrix Q that depends on the step size α and iteration index k as well as on the covariance structure of X. We describe the exact form of the dependence in Section 3.2.

The minimization in (5) can be expressed as a posterior maximization problem given a conditional Gaussian likelihood over y and a Gaussian prior over φ.

The posterior takes the form DISPLAYFORM7 Since φ (k) in (4) maximizes (6), we may conclude that k iterations of gradient descent in a linear regression model with squared error exactly computes the MAP estimate of φ, given a Gaussian-noised observation model and a Gaussian prior over φ with parameters µ 0 = θ and Σ 0 = Q. Therefore, in the case of linear regression with squared error, MAML is exactly empirical Bayes using the MAP estimate as the point estimate of φ.

In the nonlinear case, MAML is again equivalent to an empirical Bayes procedure to maximize the marginal likelihood that uses a point estimate for φ computed by one or a few steps of gradient descent.

However, this point estimate is not necessarily the global mode of a posterior.

We can instead understand the point estimate given by truncated gradient descent as the value of the mode of an implicit posterior over φ resulting from an empirical loss interpreted as a negative log-likelihood, and regularization penalties and the early stopping procedure jointly acting as priors (for similar interpretations, see BID47 BID6 BID9 .The exact equivalence between early stopping and a Gaussian prior on the weights in the linear case, as well as the implicit regularization to the parameter initialization the nonlinear case, tells us that every iterate of truncated gradient descent is a mode of an implicit posterior.

In particular, we are not required to take the gradient descent procedure of fast adaptation that computesφ to convergence in order to establish a connection between MAML and hierarchical Bayes.

MAML can therefore be understood to approximate an expectation of the marginal negative log likelihood (NLL) for each task T j as DISPLAYFORM8 The algorithm for MAML as probabilistic inference is given in Algorithm 2; Subroutine 3 computes each marginal NLL using the point estimate ofφ as just described.

Formulating MAML in this way, as probabilistic inference in a hierarchical Bayesian model, motivates the interpretation in Section 3.2 of using various meta-optimization algorithms to induce a prior over task-specific parameters.

From Section 3.1, we may conclude that early stopping during fast adaptation is equivalent to a specific choice of a prior over task-specific parameters, p( φ j | θ ).

We can better understand the role of early stopping in defining the task-specific parameter prior in the case of a quadratic objective.

Omit the task index from φ and x, and consider a second-order approximation of the fast adaptation objective (φ) = − log p( x 1 . . . , x N | φ ) about a minimum φ * : DISPLAYFORM0 where the Hessian H = ∇ 2 φ (φ * ) is assumed to be positive definite so that˜ is bounded below.

Furthermore, consider using a curvature matrix B to precondition the gradient in gradient descent, giving the update DISPLAYFORM1 If B is diagonal, we can identify (8) as a Newton method with a diagonal approximation to the inverse Hessian; using the inverse Hessian evaluated at the point φ (k−1) recovers Newton's method itself.

On the other hand, meta-learning the matrix B matrix via gradient descent provides a method to incorporate task-general information into the covariance of the fast adaptation prior, p( φ | θ ).

For instance, the meta-learned matrix B may encode correlations between parameters that dictates how such parameters are updated relative to each other.

Formally, taking k steps of gradient descent from φ (0) = θ using the update rule in (8) gives a φ (k) that solves DISPLAYFORM2 The minimization in (9) corresponds to taking a Gaussian prior p( φ | θ ) with mean θ and covariance BID44 where B is a diagonal matrix that results from a simultaneous diagonalization of H and B as O T HO = diag(λ 1 , . . .

, λ n ) = Λ and DISPLAYFORM3 DISPLAYFORM4 . .

, n (Theorem 8.7.1 in BID16 .

If the true objective is indeed quadratic, then, assuming the data is centered, H is the unscaled covariance matrix of features, X T X.

Identifying MAML as a method for probabilistic inference in a hierarchical model allows us to develop novel improvements to the algorithm.

In Section 4.1, we consider an approach from Bayesian parameter estimation to improve the MAML algorithm, and in Section 4.2, we discuss how to make this procedure computationally tractable for high-dimensional models.

We have shown that the MAML algorithm is an empirical Bayes procedure that employs a point estimate for the mid-level, task-specific parameters in a hierarchical Bayesian model.

However, the use of this point estimate may lead to an inaccurate point approximation of the integral in (2) if the posterior over the task-specific parameters, p( φ j | x j N +1 , . . .

, x j N +M , θ ), is not sharply peaked at the value of the point estimate.

The Laplace approximation BID24 BID29 a) is applicable in this case as it replaces a point estimate of an integral with the volume of a Gaussian centered at a mode of the integrand, thereby forming a local quadratic approximation.

We can make use of this approximation to incorporate uncertainty about the task-specific parameters into the MAML algorithm at fast adaptation time.

In particular, suppose that each integrand in (2) has a mode φ * j at which it is locally well-approximated by a quadratic function.

The Laplace approximation uses a second-order Taylor expansion of the negative log posterior in order to approximate each integral in the product in (2) as DISPLAYFORM0 where H j is the Hessian matrix of second derivatives of the negative log posterior.

Classically, the Laplace approximation uses the MAP estimate for φ * j , although any mode can be used as an expansion site provided the integrand is well enough approximated there by a quadratic.

We use the point estimateφ j uncovered by fast adaptation, in which case the MAML objective in (1) becomes an appropriately scaled version of the approximate marginal likelihood DISPLAYFORM1 The term log p(φ j | θ ) results from the implicit regularization imposed by early stopping during fast adaptation, as discussed in Section 3.1.

The term 1 /2 log det(H j ), on the other hand, results from the Laplace approximation and can be interpreted as a form of regularization that penalizes model complexity.

Using (11) as a training criterion for a neural network model is difficult due to the required computation of the determinant of the Hessian of the log posterior H j , which itself decomposes into a sum of the Hessian of the log likelihood and the Hessian of the log prior as DISPLAYFORM0 In our case of early stopping as regularization, the prior over task-specific parameters p( φ j | θ ) is implicit and thus no closed form is available for a general model.

Although we may use the quadratic approximation derived in Section 3.2 to obtain an approximate Gaussian prior, this prior is not diagonal and does not, to our knowledge, have a convenient factorization.

Therefore, in our experiments, we instead use a simple approximation in which the prior is approximated as a diagonal Gaussian with precision τ .

We keep τ fixed, although this parameter may be cross-validated for improved performance.

DISPLAYFORM1 Subroutine 4: Subroutine for computing a Laplace approximation of the marginal likelihood.

Similarly, the Hessian of the log likelihood is intractable to form exactly for all but the smallest models, and furthermore, is not guaranteed to be positive definite at all points, possibly rendering the Laplace approximation undefined.

To combat this, we instead seek a curvature matrixĤ that approximates the quadratic curvature of a neural network objective function.

Since it is well-known that the curvature associated with neural network objective functions is highly non-diagonal (e.g., BID32 , a further requirement is that the matrix have off-diagonal terms.

Due to the difficulties listed above, we turn to second order gradient descent methods, which precondition the gradient with an inverse curvature matrix at each iteration of descent.

The Fisher information matrix BID13 has been extensively used as an approximation of curvature, giving rise to a method known as natural gradient descent BID1 .

A neural network with an appropriate choice of loss function is a probabilistic model and therefore defines a Fisher information matrix.

Furthermore, the Fisher information matrix can be seen to define a convex quadratic approximation to the objective function of a probabilistic neural model BID38 BID31 .

Importantly for our use case, the Fisher information matrix is positive definite by definition as well as non-diagonal.

However, the Fisher information matrix is still expensive to work with.

BID33 developed Kronecker-factored approximate curvature (K-FAC), a scheme for approximating the curvature of the objective function of a neural network with a block-diagonal approximation to the Fisher information matrix.

Each block corresponds to a unique layer in the network, and each block is further approximated as a Kronecker product (see BID54 of two much smaller matrices by assuming that the second-order statistics of the input activation and the back-propagated derivatives within a layer are independent.

These two approximations ensure that the inverse of the Fisher information matrix can be computed efficiently for the natural gradient.

For the Laplace approximation, we are interested in the determinant of a curvature matrix instead of its inverse.

However, we may also make use of the approximations to the Fisher information matrix from K-FAC as well as properties of the Kronecker product.

In particular, we use the fact that the determinant of a Kronecker product is the product of the exponentiated determinants of each of the factors, and that the determinant of a block diagonal matrix is the product of the determinants of the blocks BID54 .

The determinants for each factor can be computed as efficiently as the inverses required by DISPLAYFORM2 We make use of the Laplace approximation and K-FAC to replace Subroutine 3, which computes the task-specific marginal NLLs using a point estimate forφ.

We call this method the Lightweight Laplace Approximation for Meta-Adaptation (LLAMA), and give a replacement subroutine in Subroutine 4.

The goal of our experiments is to evaluate if we can use our probabilistic interpretation of MAML to generate samples from the distribution over adapted parameters, and futhermore, if our method can be applied to large-scale meta-learning problems such as miniImageNet.

and amplitudes, and the interpretation of the method as hierarchical Bayes makes it practical to directly sample models from the posterior.

In this figure, we illustrate various samples from the posterior of a model that is meta-trained on different sinusoids, when presented with a few datapoints (in red) from a new, previously unseen sinusoid.

Note that the random samples from the posterior predictive describe a distribution of functions that are all sinusoidal and that there is increased uncertainty when the datapoints are less informative (i.e., when the datapoints are sampled only from the lower part of the range input, shown in the bottom-right example).

The connection between MAML and hierarchical Bayes suggests that we should expect MAML to behave like an algorithm that learns the mean of a Gaussian prior on model parameters, and uses the mean of this prior as an initialization during fast adaptation.

Using the Laplace approximation to the integration over task-specific parameters as in (10) assumes a task-specific parameter posterior with mean at the adapted parametersφ and covariance equal to the inverse Hessian of the log posterior evaluated at the adapted parameter value.

Instead of simply using this density in the Laplace approximation as an additional regularization term as in FORMULA0 , we may sample parameters φ j from this density and use each set of sampled parameters to form a set of predictions for a given task.

To illustrate this relationship between MAML and hierarchical Bayes, we present a meta-dataset of sinusoid tasks in which each task involves regressing to the output of a sinusoid wave in Figure We observe in FIG0 that our method allows us to directly sample models from the task-specific parameter distribution after being presented with 10 datapoints from a new, previously unseen sinusoid curve.

In particular, the column on the right of FIG0 demonstrates that the sampled models display an appropriate level of uncertainty when the datapoints are ambiguous (as in the bottom right).

We evaluate LLAMA on the miniImageNet BID40 1-shot, 5-way classification task, a standard benchmark in few-shot classification.

miniImageNet comprises 64 training classes, 12 validation classes, and 24 test classes.

Following the setup of BID55 , we structure the N -shot, J-way classification task as follows: The model observes N instances of J unseen classes, and is evaluated on its ability to classify M new instances within the J classes.

We use a neural network architecture standard to few-shot classification (e.g., BID55 BID40 , consisting of 4 layers with 3 × 3 convolutions and 64 filters, followed by batch normalization (BN) BID19 , a ReLU nonlinearity, and 2 × 2 max-pooling.

For the scaling variable β and centering variable γ of BN (see BID19 , we ignore the fast adaptation update as well as the Fisher factors for K-FAC.

We use Adam BID20 as the meta-optimizer, and standard batch gradient descent with a fixed learning rate to update the model BID53 49.82 ± 0.78 BID12 48.70 ± 1.84 LLAMA (Ours) DISPLAYFORM0 49.40 ± 1.83 during fast adaptation.

LLAMA requires the prior precision term τ as well as an additional parameter η ∈ R + that weights the regularization term log detĤ contributed by the Laplace approximation.

We fix τ = 0.001 and selected η = 10 −6 via cross-validation; all other parameters are set to the values reported in BID12 .We find that LLAMA is practical enough to be applied to this larger-scale problem.

In particular, our TensorFlow implementation of LLAMA trains for 60,000 iterations on one TITAN Xp GPU in 9 hours, compared to 5 hours to train MAML.

As shown in TAB1 , LLAMA achieves comparable performance to the state-of-the-art meta-learning method by BID53 .

While the gap between MAML and LLAMA is small, the improvement from the Laplace approximation suggests that a more accurate approximation to the marginalization over task-specific parameters will lead to further improvements.

Meta-learning and few-shot learning have a long history in hierarchical Bayesian modeling (e.g., BID51 BID11 BID25 BID58 BID14 BID8 BID56 .

A related subfield is that of transfer learning, which has used hierarchical Bayes extensively (e.g., BID39 .

A variety of inference methods have been used in Bayesian models, including exact inference BID22 , sampling methods BID41 , and variational methods BID10 .

While some prior works on hierarchical Bayesian models have proposed to handle basic image recognition tasks, the complexity of these tasks does not yet approach the kinds of complex image recognition problems that can be solved by discriminatively trained deep networks, such as the miniImageNet experiment in our evaluation BID30 .Recently, the Omniglot benchmark Lake et al. FORMULA0 has rekindled interest in the problem of learning from few examples.

Modern methods accomplish few-shot learning either through the design of network architectures that ingest the few-shot training samples directly (e.g., BID21 BID55 BID48 BID17 BID53 , or formulating the problem as one of learning to learn, or meta-learning (e.g., BID45 BID4 BID46 BID3 .

A variety of inference methods have been used in Bayesian models, including exact inference (Lake et al., 2011), sampling methods BID42 , and variational methods BID10 .Our work bridges the gap between gradient-based meta-learning methods and hierarchical Bayesian modeling.

Our contribution is not to formulate the meta-learning problem as a hierarchical Bayesian model, but instead to formulate a gradient-based meta-learner as hierarchical Bayesian inference, thus providing a way to efficiently perform posterior inference in a model-agnostic manner.

We have shown that model-agnostic meta-learning (MAML) estimates the parameters of a prior in a hierarchical Bayesian model.

By casting gradient-based meta-learning within a Bayesian framework, our analysis opens the door to novel improvements inspired by probabilistic machinery.

As a step in this direction, we propose an extension to MAML that employs a Laplace approximation to the posterior distribution over task-specific parameters.

This technique provides a more accurate estimate of the integral that, in the original MAML algorithm, is approximated via a point estimate.

We show how to estimate the quantity required by the Laplace approximation using Kroneckerfactored approximate curvature (K-FAC), a method recently proposed to approximate the quadratic curvature of a neural network objective for the purpose of a second-order gradient descent technique.

Our contribution illuminates the road to exploring further connections between gradient-based metalearning methods and hierarchical Bayesian modeling.

For instance, in this work we assume that the predictive distribution over new data-points is narrow and well-approximated by a point estimate.

We may instead employ methods that make use of the variance of the distribution over task-specific parameters in order to model the predictive density over examples from a novel task.

Furthermore, it is known that the Laplace approximation is inaccurate in cases where the integral is highly skewed, or is not unimodal and thus is not amenable to approximation by a single Gaussian mode.

This could be solved by using a finite mixture of Gaussians, which can approximate many density functions arbitrarily well BID49 BID0 .

The exploration of additional improvements such as this is an exciting line of future work.

<|TLDR|>

@highlight

A specific gradient-based meta-learning algorithm, MAML, is equivalent to an inference procedure in a hierarchical Bayesian model. We use this connection to improve MAML via methods from approximate inference and curvature estimation.