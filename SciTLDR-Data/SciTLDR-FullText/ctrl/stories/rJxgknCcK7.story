A promising class of generative models maps points from a simple distribution to a complex distribution through an invertible neural network.

Likelihood-based training  of  these  models  requires  restricting  their  architectures  to  allow  cheap computation of Jacobian determinants.

Alternatively, the Jacobian trace can be used if the transformation is specified by an ordinary differential equation.

In this paper, we use Hutchinson’s trace estimator to give a scalable unbiased estimate of the log-density.

The result is a continuous-time invertible generative model with unbiased density estimation and one-pass sampling, while allowing unrestricted neural network architectures.

We demonstrate our approach on high-dimensional density  estimation,  image  generation,  and  variational  inference,  achieving  the state-of-the-art among exact likelihood methods with efficient sampling.

Reversible generative models use cheaply invertible neural networks to transform samples from a fixed base distribution.

Examples include NICE BID3 , Real NVP BID4 , and Glow BID12 .

These models are easy to sample from, and can be trained by maximum likelihood using the change of variables formula.

However, this requires placing awkward restrictions on their architectures, such as partitioning dimensions or using rank one weight matrices, in order to avoid an O(D 3 ) cost determinant computation.

In contrast to directly parameterizing a normalized distribution (e.g. BID17 ; BID5 ), the change of variables formula allows one to specify a complex normalized distribution p x (x) implicitly by warping a normalized base distribution p z (z) through an invertible function f : R D → R D .

Given a random variable z ∼ p z (z) the log density of x = f (z) follows log p x (x) = log p z (z) − log det ∂f (z) ∂zwhere ∂f (z) /∂z is the Jacobian of f .

In general, computing the log determinant has a time cost of O(D 3 ).

Much work has gone into developing restricted neural network architectures which make computing the Jacobian's determinant more tractable.

These approaches broadly fall into three categories:Normalizing flows.

By restricting the functional form of f , various determinant identities can be exploited (Rezende & Mohamed, 2015; Berg et al., 2018) .

These models cannot be trained as generative models from data because they do not have a tractable inverse f −1 .

However, they are useful for specifying approximate posteriors for variational inference BID13 .

Autoregressive transformations.

By using an autoregressive model and specifying an ordering of the dimensions, the Jacobian of f is enforced to be lower triangular BID14 BID16 .

These models excel at density estimation for tabular datasets (Papamakarios et al., 2017) , but require D sequential evaluations of f to invert, which is prohibitive when D is large.

Partitioned transformations.

Partitioning the dimensions and using affine transformations makes the determinant of the Jacobian cheap to compute, and the inverse f −1 computable with the same cost as f BID3 BID4 .

This method allows the use of convolutional architectures, excelling at density estimation for image data BID4 BID12 .Throughout this work, we refer to reversible generative models as those which use the change of variables to transform a base distribution to the model distribution while maintaining both efficient density estimation and efficient sampling capabilities using a single pass of the model.

There exist several approaches to generative modeling approaches which do not use the change of variables equation for training.

Generative adversarial networks (GANs) BID6 use large, unrestricted neural networks to transform samples from a fixed base distribution.

Lacking a closed-form likelihood, an auxiliary discriminator model must be trained to estimate divergences or density ratios in order to provide a training signal.

Autoregressive models BID5 BID17 directly specify the joint distribution p(x) as a sequence of explicit conditional distributions using the product rule.

These models require at least O(D) evaluations to sample from.

Variational autoencoders (VAEs) BID13 use an unrestricted architecture to explicitly specify the conditional likelihood p(x|z), but can only efficiently provide a stochastic lower bound on the marginal likelihood p(x).

Chen et al. (2018) define a generative model for data x ∈ R D similar to those based on (1), but replace the warping function with an integral of continuous-time dynamics.

The generative process first samples from a base distribution z 0 ∼ p z0 (z 0 ).

Then, given an ODE whose dynamics are defined by the parametric function ∂z(t) /∂t = f (z(t), t; θ), we solve the initial value problem with z(t 0 ) = z 0 to obtain a data sample x = z(t 1 ).

These models are called Continous Normalizing Flows (CNF).

The change in log-density under this model follows a second differential equation, called the instantaneous change of variables formula BID2 : DISPLAYFORM0 We can compute total change in log-density by integrating across time:

Given a datapoint x, we can compute both the point z 0 which generates x, as well as log p(x) under the model by solving the combined initial value problem: DISPLAYFORM1 DISPLAYFORM2 which integrates the combined dynamics of z(t) and the log-density of the sample backwards in time from t 1 to t 0 .

We can then compute log p(x) using the solution of (4) and adding log p z0 (z 0 ).

The existence and uniqueness of (4) require that f and its first derivatives be Lipschitz continuous BID10 , which can be satisfied in practice using neural networks with smooth Lipschitz activations, such as softplus or tanh.

CNFs are trained to maximize (3).

This objective involves the solution to an initial value problem with dynamics parameterized by θ.

For any scalar loss function which operates on the solution to an initial value problem DISPLAYFORM0 then Pontryagin (1962) shows that its derivative takes the form of another initial value problem DISPLAYFORM1 The quantity − ∂L /∂z(t) is known as the adjoint state of the ODE.

BID2 use a black-box ODE solver to compute z(t 1 ), and then a separate call to a solver to compute (6) with the initial value ∂L /∂z(t1).

This approach is a continuous-time analog to the backpropgation algorithm (Rumelhart et al., 1986; BID1 and can be combined with gradient-based optimization to fit the parameters θ by maximum likelihood.

Switching from discrete-time dynamics to continuous-time dynamics reduces the primary computational bottleneck of normalizing flows from DISPLAYFORM0 , at the cost of introducing a numerical ODE solver.

This allows the use of more expressive architectures.

For example, each layer of the original normalizing flows model of Rezende & Mohamed (2015) is a one-layer neural network with only a single hidden unit.

In contrast, the instantaneous transformation used in planar continuous normalizing flows BID2 ) is a one-layer neural network with many hidden units.

In this section, we construct an unbiased estimate of the log-density with O(D) cost, allowing completely unrestricted neural network architectures to be used.

In general, computing Tr ( ∂f /∂z(t)) exactly costs O(D 2 ), or approximately the same cost as D evaluations of f , since each entry of the diagonal of the Jacobian requires computing a separate derivative of f BID7 .

However, there are two tricks that can help.

First, vector-Jacobian products v T ∂f ∂z can be computed for approximately the same cost as evaluating f using reverse-mode automatic differentiation.

Second, we can get an unbiased estimate of the trace of a matrix by taking a double product of that matrix with a noise vector: DISPLAYFORM0 The above equation holds for any D-by-D matrix A and distribution p( ) over D-dimensional vectors such that E[ ] = 0 and Cov( ) = I. The Monte Carlo estimator derived from FORMULA7 is known as Hutchinson's trace estimator BID9 BID0 .To keep the dynamics deterministic within each call to the ODE solver, we can use a fixed noise vector for the duration of each solve without introducing bias: DISPLAYFORM1 Typical choices of p( ) are a standard Gaussian or Rademacher distribution BID9 .

Often, there exist bottlenecks in the architecture of the dynamics network, i.e. hidden layers whose width H is smaller than the dimensions of the input D. In such cases, we can reduce the variance of Hutchinson's estimator by using the cyclic property of trace.

Since the variance of the estimator for Tr(A) grows asymptotic to ||A|| DISPLAYFORM0 When f has multiple hidden layers, we choose H to be the smallest dimension.

This bottleneck trick can reduce the norm of the matrix which may also help reduce the variance of the trace estimator.

As introducing a bottleneck limits our model capacity, we do not use this trick in our experiments.

However this trick can reduce variance when a bottleneck is used, as shown in our ablation studies.

Our complete method uses the dynamics defined in (2) and the efficient log-likelihood estimator of (8) to produce the first scalable and reversible generative model with an unconstrained Jacobian.

We call this method Free-Form Jacobian of Reversible Dyanamics (FFJORD).

Pseudo-code of our method is given in Algorithm 1, and TAB1 summarizes the capabilities of our model compared to other recent generative modeling approaches.

Assuming the cost of evaluating f is on the order of O(DH) where D is the dimensionality of the data and H is the size of the largest hidden layer in f , then the cost of computing the likelihood in models with repeated use of invertible transformations FORMULA0 is DISPLAYFORM0 where L is the number of transformations used.

For CNF, this reduces to O((DH + D 2 )L) for CNFs, whereL is the number of evaluations of f used by the ODE solver.

With FFJORD, this reduces further to DISPLAYFORM1 Algorithm 1 Unbiased stochastic log-density estimation using the FFJORD model Require: dynamics f θ , start time t 0 , stop time t 1 , data samples x, data dimension D.← sample unit variance(x.shape) Sample outside of the integral DISPLAYFORM2 Augment f with log-density dynamics.

DISPLAYFORM3 Compute vector-Jacobian product with automatic differentiation DISPLAYFORM4 Figure 2: Comparison of trained Glow, planar CNF, and FFJORD models on 2-dimensional distributions, including multi-modal and discontinuous densities.

We demonstrate FFJORD on a variety of density estimation tasks, and for approximate inference in variational autoencoders BID13 .

Experiments were conducted using a suite of GPU-based ODE-solvers and an implementation of the adjoint method for backpropagation 1 .

In all experiments the RungeKutta 4(5) algorithm with the tableau from Shampine (1986) was used to solve the ODEs.

We ensure tolerance is set low enough so numerical error is negligible; see Appendix C.We used Hutchinson's trace estimator (7) during training and the exact trace when reporting test results.

This was done in all experiments except for our density estimation models trained on MNIST and CIFAR10 where computing the exact Jacobian trace was too expensive.

The dynamics of FFJORD are defined by a neural network f which takes as input the current state z(t) ∈ R D and the current time t ∈ R. We experimented with several ways to incorporate t as an input to f , such as hyper-networks, but found that simply concatenating t on to z(t) at the input to every layer worked well and was used in all of our experiments.

We first train on 2 dimensional data to visualize the model and the learned dynamics.2 In FIG1 , we show that by warping a simple isotropic Gaussian, FFJORD can fit both multi-modal and even discontinuous distributions.

The number of evaluations of the ODE solver is roughly 70-100 on all datasets, so we compare against a Glow model with 100 discrete layers.

The learned distributions of both FFJORD and Glow can be seen in FIG1 .

Interestingly, we find that Glow learns to stretch the unimodal base distribution into multiple modes but has trouble modeling the areas of low probability between disconnected regions.

In contrast, FFJORD is capable of modeling disconnected modes and can also learn convincing approximations of discontinuous density functions (middle row in FIG1 ).

Since the main benefit of FFJORD is the ability to train with deeper dynamics networks, we also compare against planar CNF BID2 BID8 , cannot be sampled from without resorting to correlated or expensive sampling algorithms such as MCMC.On MNIST we find that FFJORD can model the data as effectively as Glow and Real NVP using only a single flow defined by a single neural network.

This is in contrast to Glow and Real NVP which must compose many flows to achieve similar performance.

When we use multiple flows in a multiscale architecture (like those used by Glow and Real NVP) we obtain better performance on MNIST and comparable performance to Glow on CIFAR10.

Notably, FFJORD is able to achieve this performance while using less than 2% as many parameters as Glow.

We also note that Glow uses a learned base distribution whereas FFJORD and Real NVP use a fixed Gaussian.

A summary of our results on density estimation can be found in TAB4 and samples can be seen in Figure 3 .

Full details on architectures used, our experimental procedure, and additional samples can be found in Appendix B.1.In general, our approach is slower than competing methods, but we find the memory-efficiency of the adjoint method allows us to use much larger batch sizes than those methods.

On the tabular datasets we used a batch sizes up to 10,000 and on the image datasets we used a batch size of 900.

We compare FFJORD to other normalizing flows for use in variational inference.

In VAEs it is common for the encoder network to also output the parameters of the flow as a function of the input x. With FFJORD, we found this led to differential equations which were too difficult to integrate numerically.

Instead, the encoder network outputs a low-rank update to a global weight matrix and an input-dependent bias vector.

When used in recognition nets, neural network layers defining the dynamics inside FFJORD take the form DISPLAYFORM0 where h is the input to the layer, σ is an element-wise activation function, D in and D out are the input and output dimension of this layer, andÛ (x),V (x),b(x) are input-dependent parameters returned from an encoder network.

A full description of the model architectures used and our experimental setup can be found in Appendix B.2.On every dataset tested, FFJORD outperforms all other competing normalizing flows.

A summary of our variational inference results can be found in TAB6 .

We performed a series of ablation experiments to gain a better understanding of the proposed model.

We plotted the training losses on MNIST using an encoder-decoder architecture (see Appendix B.1 for details).

Loss during training is plotted in FIG2 , where we use the trace estimator directly on the D×D Jacobian, or we use the bottleneck trick to reduce the dimension to H × H. Interestingly, we find that while the bottleneck trick (9) can lead to faster convergence when the trace is estimated using a Gaussian-distributed , we did not observe faster convergence when using a Rademacherdistributed .

The full computational cost of integrating the instantaneous change of variables FORMULA1 is O(DH L) where D is dimensionality of the data, H is the size of the hidden state, and L is the number of function evaluations (NFE) that the adaptive solver uses to integrate the ODE.

In general, each evaluation of the model is O(DH) and in practice, H is typically chosen to be close to D. Since the general form of the discrete change of variables equation FORMULA0 Figure 5: NFE used by the adaptive ODE solver is approximately independent of data-dimension.

Lines are smoothed using a Gaussian filter.

We train VAEs using FFJORD flows with increasing latent dimension D. The NFE throughout training is shown in Figure 5 .

In all models, we find that the NFE increases throughout training, but converges to the same value, independent of D. We conjecture that the number of evaluations is not dependent on the dimensionality of the data but the complexity of its distribution, or more specifically, how difficult it is to transform its density into the base distribution.

200 400 600 800 NFE 0.9 DISPLAYFORM0 Bits/dimSingle FFJORD Multiscale FFJORD Figure 6 : For image data, a single FFJORD flow can achieve near performance to multi-scale architecture while using half the number of evaluations.

Crucial to the scalability of Real NVP and Glow is the multiscale architecture originally proposed in BID4 .

We compare a single-scale encoder-decoder style FFJORD with a multiscale FFJORD on the MNIST dataset where both models have a comparable number of parameters and plot the total NFE-in both forward and backward passes-against the loss achieved in Figure 6 .

We find that while the single-scale model uses approximately one half as many function evaluations as the multiscale model, it is not able to achieve the same performance as the multiscale model.

Number of function evaluations can be prohibitive.

The number of function evaluations required to integrate the dynamics is not fixed ahead of time, and is a function of the data, model architecture, and model parameters.

This number tends to grow as the models trains and can become prohibitively large, even when memory stays constant due to the adjoint method.

Various forms of regularization such as weight decay and spectral normalization BID15 can be used to reduce the this quantity, but their use tends to hurt performance slightly.

Limitations of general-purpose ODE solvers.

In theory, our model can approximate any differential equation (given mild assumptions based on existence and uniqueness of the solution), but in practice our reliance on general-purpose ODE solvers restricts us to non-stiff differential equations that can be efficiently solved.

ODE solvers for stiff dynamics exist, but they evaluate f many more times to achieve the same error.

We find that a small amount of weight decay regularizes the ODE to be sufficiently non-stiff.

We have presented FFJORD, a reversible generative model for high-dimensional data which can compute exact log-likelihoods and can be sampled from efficiently.

Our model uses continuoustime dynamics to produce a generative model which is parameterized by an unrestricted neural network.

All required quantities for training and sampling can be computed using automatic differentiation, Hutchinson's trace estimator, and black-box ODE solvers.

Our model stands in contrast to other methods with similar properties which rely on restricted, hand-engineered neural network architectures.

We demonstrated that this additional flexibility allows our approach to achieve on-par or improved performance on density estimation and variational inference.

We believe there is much room for further work exploring and improving this method.

FFJORD is empirically slower to evaluate than other reversible models like Real NVP or Glow, so we are interested specifically in ways to reduce the number of function evaluations used by the ODE-solver without hurting predictive performance.

Advancements like these will be crucial in scaling this method to even higher-dimensional datasets.

We thank Yulia Rubanova and Roger Grosse for helpful discussions.

Samples from our FFJORD models trained on MNIST and CIFAR10 can be found in Figure 7 .

Figure 7: Samples and data from our image models.

MNIST on left, CIFAR10 on right.

On the tabular datasets we performed a grid-search over network architectures.

We searched over models with 1, 2, 5, or 10 flows with 1, 2, 3, or 4 hidden layers per flow.

Since each dataset has a different number of dimensions, we searched over hidden dimensions equal to 5, 10, or 20 times the data dimension (hidden dimension multiplier in TAB10 ).

We tried both the tanh and softplus nonlinearities.

The best performing models can be found in the TAB10 .On the image datasets we experimented with two different model architectures; a single flow with an encoder-decoder style architecture and a multiscale architecture composed of multiple flows.

While they were able to fit MNIST and obtain competitive performance, the encoder-decoder architectures were unable to fit more complicated image datasets such as CIFAR10 and Street View House Numbers.

The architecture for MNIST which obtained the results in TAB4 was composed of four convolutional layers with 64 → 64 → 128 → 128 filters and down-sampling with strided convolutions by two every other layer.

There are then four transpose-convolutional layers who's filters mirror the first four layers and up-sample by two every other layer.

The softplus activation function is used in every layer.

The multiscale architectures were inspired by those presented in BID4 .

We compose multiple flows together interspersed with "squeeze" operations which down-sample the spatial resolution of the images and increase the number of channels.

These operations are stacked into a "scale block" which contains N flows, a squeeze, then N flows.

For MNIST we use 3 scale blocks and for CIFAR10 we use 4 scale blocks and let N = 2 for both datasets.

Each flow is defined by 3 convolutional layers with 64 filters and a kernel size of 3.

The softplus nonlinearity is used in all layers.

Both models were trained with the Adam optimizer BID11 .

We trained for 500 epochs with a learning rate of .001 which was decayed to .0001 after 250 epochs.

Training took place on six GPUs and completed after approximately five days.

Our experimental procedure exactly mirrors that of Berg et al. (2018) .

We use the same 7-layer encoder and decoder, learning rate (.001), optimizer (Adam Kingma & Ba (2015) ), batch size (100), and early stopping procedure (stop after 100 epochs of no validaiton improvment).

The only difference was in the nomralizing flow used in the approximate posterior.

We performed a grid-search over neural network architectures for the dynamics of FFJORD.

We searched over networks with 1 and 2 hidden layers and hidden dimension 512, 1024, and 2048.

We used flows with 1, 2, or 5 steps and wight matrix updates of rank 1, 20, and 64.

We use the softplus activation function for all datasets except for Caltech Silhouettes where we used tanh.

The best performing models can be found in the TAB11 .

Models were trained on a single GPU and training took between four hours and three days depending on the dataset.

Table 6 : Negative log-likehood on test data for density estimation models.

Means/stdev over 3 runs.

Real NVP, MADE, MAF, TAN, and MAF-DDSF results on are taken from BID8 .

In reproducing Glow, we were able to get comparable results to the reported Real NVP by removing the invertible fully connected layers.

ODE solvers are numerical integration methods so there is error inherent in their outputs.

Adaptive solvers (like those used in all of our experiments) attempt to predict the errors that they accrue and modify their step-size to reduce their error below a user set tolerance.

It is important to be aware of this error when we use these solvers for density estimation as the solver outputs the density that we report and compare with other methods.

When tolerance is too low, we run into machine precision errors.

Similarly when tolerance is too high, errors are large, our training objective becomes biased and we can run into divergent training dynamics.

Since a valid probability density function integrates to one, we take a model trained on FIG0 and numerically find the area under the curve using Riemann sum and a very fine grid.

We do this for a range of tolerance values and show the resulting error in FIG3 .

We set both atol and rtol to the same tolerance.

The numerical error follows the same order as the tolerance, as expected.

During training, we find that the error becomes non-negligible when using tolerance values higher than 10 −5 .

For most of our experiments, we set tolerance to 10 −5 as that gives reasonable performance while requiring few number of evaluations.

For the tabular experiments, we use atol=10 −8 and rtol=10 −6 .

<|TLDR|>

@highlight

We use continuous time dynamics to define a generative model with exact likelihoods and efficient sampling that is parameterized by unrestricted neural networks.