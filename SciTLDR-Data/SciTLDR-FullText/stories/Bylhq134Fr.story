Bayesian inference offers a theoretically grounded and general way to train neural networks and can potentially give calibrated uncertainty.

However, it is challenging to specify a meaningful and tractable prior over the network parameters, and deal with the weight correlations in the posterior.

To this end, this paper introduces two innovations: (i) a Gaussian process-based hierarchical model for the network parameters based on recently introduced unit embeddings that can flexibly encode weight structures, and (ii) input-dependent contextual variables for the weight prior that can provide convenient ways to regularize the function space being modeled by the network through the use of kernels.

We show these models provide desirable test-time uncertainty estimates, demonstrate cases of modeling inductive biases for neural networks with kernels and demonstrate competitive predictive performance on an active learning benchmark.

The question of which priors one should use for Bayesian neural networks is largely unanswered, as two considerations need to be balanced: First, we want to keep inference in the high dimensional weight posterior tractable; Second, we desire to express our beliefs about the properties of the modeled functions compactly by modeling the collection of weights.

Especially the latter is typically hard, as functional regularization for weight-based models is non-trivial.

In order to cope with richer posterior inference than mean-field typically achieves, a variety of structured posterior models have been proposed recently, for instance utilizing radial posteriors (Oh et al., 2019) , or rich weight posteriors based on Gaussian processes (Louizos and Welling, 2016) .

When it comes to modeling priors on weights with correlations, recent work has attempted to capture feature-level correlations using for instance a horseshoe prior (Ghosh et al., 2018) .

One interesting direction of inquiry has focused on utilizing hyper-networks in order to model distributions over weights for an entire network (Ha et al., 2016; Pradier et al., 2018) , or alternatively to utilize unit-level level variables combined with compact hyper-networks to regress to single weights and capture weight correlations through the auxiliary variables (Karaletsos et al., 2018) .

We propose to tackle some of the challenges in modeling weight priors by extending the latter work and combining it with ideas from the Gaussian process literature to replace the hyper-network with a Gaussian process prior over weights.

We explore the use of compositional kernels to add input-dependence to the prior for our model and obtain rich models with beneficial properties in tasks such as active learning, and generalization, while maintaining tractable inference properties.

In (Karaletsos et al., 2018 ) each unit (visible or hidden) of the l-th layer of the network has a corresponding latent hierarchical variable z l,i , of dimensions D z , where i denotes the index of the unit in a layer.

These latent variables are used to construct the weights in the network such that a weight in the l-th weight layer, w l,i,j is linked to the latent variables z's of the i-th input unit and the j-th output unit of the weight layer.

We can summarize this relationship by introducing a set of weight encodings, C w (z), one for each individual weight, c w l,i,j = z l+1,i , z l,j .

The probabilistic description of the relationship between the weight codes and the weights w is:

, where l denotes a visible or hidden layer and H l is the number of units in that layer, and w denotes all the weights in this network.

In (Karaletsos et al., 2018) , a small parametric neural network regression model maps the latent variables to the weights,

.

We will call this network a meta mapping.

We assume p(z) = N (z; 0, I).

We can thus write down the joint density of the resulting hierarchical model as follows,

Variational inference was employed in prior work to infer z (and w implicitly), and to obtain a point estimate of ??, as a by-product of optimising the variational lower bound.

Notice that in Sec.2, the meta mapping from the hierarchical latent variables to the weights is a parametric non-linear function, specified by a neural network.

We replace the parametric neural network by a probabilistic functional mapping and place a nonparametric Gaussian process (GP) prior over this function.

That is,

where we have assumed a zero-mean GP, k ?? (??, ??) is a covariance function and ?? is a small set of hyper-parameters.

The effect is that the latent function introduces correlations for the individual weight predictions,

Notably, while the number of latent variables and weights can be large, the input dimension to the GP mapping is only 2D z , where D z is the dimensionality of each latent variable z. The GP mapping effectively performs one-dimensional regression from latent variables to individual weights while capturing their correlations.

We will refer to this mapping as a GP-MetaPrior (metaGP).

We define the following factorized kernel at the example of two weights in the network,

In this section and what follows, we will use the popular exponentiated quadratic (EQ) kernel with ARD lengthscales,

are the lengthscales and ?? 2 k is the kernel variance.

We cover inference and learning in App.

A.

We first note that whilst the hierarchical latent variables and meta mappings introduce nontrivial coupling between the weights a priori, the weights and latent variables are inherently global.

That is, a function drawn from the model, represented by a set of weights, does not take into account the inputs at which the function will be evaluated.

To this end, we introduce the input variable into the weight codes c w l,i,j = z l+1,i , z l,j , x n .

In turn, this yields input-conditional weight models p(w n,l,i,j |f, z l+1,i , z l,j , x n ).

We again turn to compositional kernels and introduce a new input kernel K x which we use as follows,

As a result of having private contextual inputs to the meta mapping, the weight priors are now also local to each data point.

We can utilize multiple useful kernels from the GP literature that allow modelers to describe relationships between data, but were previously inaccessible to neural network modelers.

We consider this a novel form of functional regularization, as the entire network can be given structure that will constrain its function space.

To scale this to large inputs, we learn transformations of inputs for the conditional weight model n = g(Vx n ), for a learned mapping V and a nonlinearity g:

We write down the joint density of all variables in the model when using our weight prior in a neural network:

We discuss inference and learning in the Appendix Sec. A.

We study our suggested priors empirically in two distinct settings in the following: first, we study the effect of kernel choice in the local model for a regression problem where we may have available intuitions as inductive biases.

Second, we explore how the input-dependence behaves in out of distribution generalization tasks.

We explore the utility of the contextual variable towards modeling inductive biases for neural networks and evaluate on predictive performance on a regression example.

In particular, we generate 100 training points from a synthetic sinusoidal function and create two test sets that contains in-sample inputs and out-of-sample inputs, respectively.

We test an array of models and inference methods, including BNN with MFVI, metaGP and metaGP with contextual variables.

We can choose the covariance function to be used for the auxiliary variables to encode our belief about how the weights should be modulated by the input.

We pick EQ and periodic kernels (MacKay, 1998) in this example.

Fig. 2 summarizes the results and illustrate the qualitative difference between models.

Note that the periodic kernel allows the model to discover and encode periodicity, allowing for more long-range confident predictions compared to that of the EQ kernel.

We test the ability of this model class to produce calibrated predictive uncertainty to outof-distribution samples.

We first train a neural network classifier with one hidden layer of 100 rectified linear units on the MNIST dataset, and apply the metaGP prior only to the last layer of the network.

After training, we compute the entropy of the predictions on various test sets, including notMNIST, fashionMNIST, Kuzushiji-MNIST, and uniform and Gaussian noise inputs.

Following (Lakshminarayanan et al., 2017; Louizos and Welling, 2017) , the CDFs of the predictive entropies for various methods are shown in Fig. 3 .

In most out-of-distribution sets considered, metaGP and metaGP with local auxiliary variables demonstrate competitive performance to Gaussian MFVI.

Notably, MAP estimation tends to give wildly poor uncertainty estimates on out-of-distribution samples.

We illustrated the utility of a GP-based hierarchical prior over neural network weights and a variational inference scheme that captures weight correlations and allows input-dependent contextual variables.

We plan to evaluate the performance of the model on more challenging decision making tasks and to extend the inference scheme to handle continual learning.

Appendix A. Appendix: Inference and learning using stochastic structured variational inference

Performing inference is challenging due to the non-linearity of the neural network and the need to infer an entire latent function f .

To address these problems, we derive a structured variational inference scheme that makes use of innovations from inducing point GP approximation literature (Titsias, 2009; Hensman et al., 2013; Qui??onero-Candela and Rasmussen, 2005; Matthews et al., 2016; Bui et al., 2017) and previous work on inferring meta-representations (Karaletsos et al., 2018) .

As a reminder, we write down the joint density of all variables in the model:

We first partition the space Z of inputs to the function f into a finite set of M variables called inducing inputs z u and the remaining inputs, Z = {x u , Z =xu }.

The function f is partitioned identically, f = {u, f =u }, where u = f (x u ).

We can then rewrite the GP prior as follows,

The inducing inputs and outputs, {x u , u}, will be used to parameterize the approximation.

In particular, a variational approximation is judiciously chosen to mirror the form of the joint density:

where the variational distribution over w is made to explicitly depend on remaining variables through the conditional prior, and q(z) is chosen to be a diagonal (mean-field) Gaussian densitie, q(z) = N (z; ?? ?? ?? z , diag(?? ?? ?? 2 z )), and q(u) is chosen to be a correlated multivariate Gaussian, q(u) = N (u; ?? ?? ?? u , ?? u ).

This approximation allows convenient cancellations yielding a tractable variational lower bound as follows,

where the last expectation has been partly approximated using simple Monte Carlo with the reparameterization trick, i.e. z k ??? q(z).

We will next discuss how to approximate the expectation F k = w,f q(w, f |z k ) log p(y|w, x).

Note that we split f into f =u and u, and that we can integrate f =u out exactly to give, q(w|z k , u) = N (w; A (k) u, B (k) ),

At this point, we can either (i) sample u from q(u), or (ii) integrate u out analytically.

We opt for the second approach, which gives

In contrast to GP regression and classification in which the likelihood term is factorized point-wise w.r.t.

the parameters and thus their expectations only involve a low dimensional integral, we have to integrate out w in this case, which is of much higher dimensions.

When necessary or practical, we resort to Kronecker factored models or make an additional diagonal approximation as follows,

Whilst the diagonal approximation above might look poor from the first glance, it is conditioned on a sample of the latent variables z k and thus the weights' correlations are retained after integrating out z. Such correlation is illustrated in 4 where we show the marginal and conditional covariance structures for the weights of a small neural network, separated into diagonal and full covariance models.

The diagonal approximation above has been observed to give pathological behaviours in the GP regression case (Bauer et al., 2016 ), but we did not observe these in practice.

F k is approximated by F k ??? wq (w|z k ) log p(y|w, x) which can be subsequently efficiently estimated using the local reparameterization trick (Kingma et al., 2015) .

The final lower bound is then optimized to obtain the variational parameterers of q(u), q(z), and estimates for the noise in the meta-GP model, the kernel hyper-parameters and the inducing inputs.

selection and more crucially using the proposed model and inference scheme seems to yield comparable or better predictive errors with a similar number of queries.

This simple setting quantitatively reveals the inferior performance of MFVI, compared to MAP and metaGP.

@highlight

We introduce a Gaussian Process Prior over weights in a neural network and explore its ability to model input-dependent weights with benefits to various tasks, including uncertainty estimation and generalization in the low-sample setting.