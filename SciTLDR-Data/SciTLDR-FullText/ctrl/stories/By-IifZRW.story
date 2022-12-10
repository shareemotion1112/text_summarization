We propose a method to learn stochastic activation functions for use in probabilistic neural networks.

First, we develop a framework to embed stochastic activation functions based on Gaussian processes in probabilistic neural networks.

Second, we analytically derive expressions for the propagation of means and covariances in such a network, thus allowing for an efficient implementation and training without the need for sampling.

Third, we show how to apply variational Bayesian inference to regularize and efficiently train this model.

The resulting model can deal with uncertain inputs and implicitly provides an estimate of the confidence of its predictions.

Like a conventional neural network it can scale to datasets of arbitrary size and be extended with convolutional and recurrent connections, if desired.

The popularity of deep learning and the implied race for better accuracy and performance has lead to new research of the fundamentals of neural networks.

Finding an optimal architecture often focusses on a hyperparameter search over the network architecture, regularization parameters, and one of a few standard activation functions: tanh, ReLU BID5 ), maxout ) . . .

Focussing on the latter, looking into activation functions has only taken off since BID12 introduced the rectified linear unit (ReLU), which were shown to produce significantly better results on image recognition tasks BID10 .

BID11 then introduced the leaky ReLU, which has a very small, but non-zero, slope for negative values.

BID8 proposed the parameterized ReLU, by making the slope of the negative part of the leaky ReLU adaptable.

It was trained as an additional parameter for each neuron alongside the weights of the neural network using stochastic gradient descent.

Thus, the activation function was not treated as a fixed hyper-parameter anymore but as adaptable to training data.

While the parameterized ReLU only has one parameter, this was generalized in BID0 to piecewise linear activation functions that can have an arbitrary (but fixed) number of points where the function changes it slope.

This can be interpreted as a different parameterization of a Maxout network ), in which each neuron takes the maximum over a set of different linear combinations of its inputs.

Instead of having a fixed parameter for the negative slope of the ReLU, BID18 introduced stochasticity into the activation function by sampling the value for the slope with each training iteration from a fixed uniform distribution.

BID3 and BID9 replaced the negative part of ReLUs with a scaled exponential function and showed that, under certain conditions, this leads to automatic renormalization of the inputs to the following layer and thereby simplifies the training of the neural networks, leading to an improvement in accuracy in various tasks.

Nearly fully adaptable activation functions have been proposed by BID4 .

The authors use a Fourier basis expansion to represent the activation function; thus with enough coefficients any (periodic) activation function can be represented.

The coefficients of this expansion are trained as network parameters using stochastic gradient descent or extensions thereof.

Promoting a more general approach, BID1 proposed to learn the activation functions alongside the layer weights.

Their adaptive piecewise linear units consist of a sum of hinge-shaped functions with parameters to control the hinges and the slopes of the linear segments.

However, by construction the derivative of these activation functions is not continuous at the joints between two linear segments, which often leads to non-optimal optimizer performance.

To our knowledge, previous research on learning activation functions took place in a fully deterministic setting, i.e. deterministic activation functions were parameterized and included in the optimization of a conventional neural network.

Here instead, we explore the setting of probabilistic activation functions embedded in a graphical model of random variables resembling the structure of a neural network.

We develop the theory of Gaussian-process neurons and subsequently derive a lower-bound approximation using variational inference, in order to develop a computationally efficient version of the Gaussian Process neuron.

To define the model we will need to slice matrices along rows and columns.

Given a matrix X, we will write X i to select all elements of the i-th row and X j to select all elements of the j-th column.

Gaussian Processes (GPs) are nonparametric models that provide flexible probabilistic approaches for function estimation.

A Gaussian Process BID14 defines a distribution over a function f (x) ∼ GP(m(x), k(x, x )) where m is called the mean function and k is the covariance function.

For S inputs x ∈ R S×N of N dimensions the corresponding function values f with f i f (X i ) follow a multivariate normal distribution DISPLAYFORM0 with mean vector m i m(X i ) and K(X, X) is the covariance matrix defined by (K(X, X)) ij k(X i , X j ).

In this work we use the zero mean function m(x) = 0 and the squared exponential (SE) covariance function with scalar inputs, DISPLAYFORM1 where ν is called the length-scale and determines how similar function values of nearby inputs are according to the GP distribution.

Since this covariance function is infinitely differentiable, all function samples from a GP using it are smooth functions.

We will first describe the fundamental, non-parametric model, which will be approximated in the following sections for efficient training and inference.

Let the input to the l-th layer of Gaussian Process neurons (GPNs) be denoted by X l−1 ∈ R S×N l−1 where S is the number of data points (samples) and N l−1 is the number of input dimensions.

A layer l ∈ {1, . . .

, L} of N l GPNs indexed by n ∈ {1, . . .

N l } is defined by the joint probability DISPLAYFORM0 with the GP prior F l conditioned on the layer inputs X l−1 multiplied with the weights W l , DISPLAYFORM1 and an additive Gaussian noise distribution, DISPLAYFORM2 This corresponds to a probabilistic activation function DISPLAYFORM3 This GP has scalar inputs and uses the standard squared exponential covariance function.

Analogous to standard neural networks, GPN layers can be stacked to form a multi-layer feed-forward network.

The joint probability of such a stack is DISPLAYFORM4 inputs activation response output Figure 1 : The auxiliary parametric representation of a GPN using virtual observation inducing points V and targets U .All input samples X 0 s , s ∈ {1, . . .

, S}, are assumed to be normally distributed with known mean and covariance, DISPLAYFORM5 To obtain predictions P(X L | X 0 ), all latent variables in eq. (6) would need to be marginalized out; unfortunately due to the occurrence of X l in the covariance matrix in eq. FORMULA3 analytic integration is intractable.

The path to obtain a tractable training objective is to temporarily parameterize the activation function f l n (z) of each GPN using virtual observations (originally proposed by BID13 for sparse approximations of GPs) of inputs and outputs of the function.

These virtual observations are only introduced as an auxiliary device and will be marginalized out later.

Each virtual observation r consists of a scalar inducing point V l rn and corresponding target U l rn .

Under these assumptions on f DISPLAYFORM0 where the mean and variance are those obtained by using the virtual observations as "training" points for a GP regression evaluated at the "test" points DISPLAYFORM1 where DISPLAYFORM2 .

Given enough inducing points that lie densely between the layer's activations A l n = X l−1 W l n , the shape of the activation function becomes predominantly determined by the corresponding targets of these inducing points.

Consequently, the inter-sample correlation in eq. (8) becomes negligible, allowing us to further approximate this conditional by factorizing it over the samples; thus we have DISPLAYFORM3 We now marginalize eq. (4) over F l n and get DISPLAYFORM4 Figure 2: A GPN feed-forward network prior distribution (a) for three layers and the approximation of its posterior obtained using variational inference and the central limit theorem on the activations (b).

Each node corresponds to all samples and GPN units within a layer.

Dotted circles represent variational parameters.

Although we now have a distribution for X l that is conditionally normal given the values of the previous layer, the marginals P(X l n ), l ∈ {1, . . .

, L}, will, in general, not be normally distributed, because the input from the previous layer X l−1 appears non-linearly through the kernel function in the mean eq. (9).By putting a GP prior on the distribution of the virtual observation targets U l as shown in fig. 1 ., DISPLAYFORM5 it can easily be verified that the marginal distribution of the response, DISPLAYFORM6 recovers the original, non-parametric GPN response distribution given by (3).

The first use of this prior-restoring technique was presented in Titsias (2009) for finding inducing points of sparse GP regression using variational methods.

The joint distribution of a GPN feed-forward network is given by DISPLAYFORM0 DISPLAYFORM1 A graphical model corresponding to that distribution for three layers is shown in fig. 2a .

Since exact marginalization over the latent variables is infeasible we apply the technique of variational inference BID16 to approximate the posterior of the model given some training data by a variational distribution Q.The information about the activation functions learned from the training data is mediated via the virtual observation targets U l , thus their variational posterior must be adaptable in order to store that information.

Hence, we choose a normal distribution factorized over the GPN units within a layer with free mean and covariance for the approximative posterior of U l , DISPLAYFORM2 This allows the inducing targets of a GPN to be correlated, but the covariance matrix can be constrained to be diagonal, if it is desired to reduce the number of variational parameters.

We keep the rest of the model distribution unchanged from the prior; thus the overall approximating posterior is DISPLAYFORM3 Estimating the variational parameters µ DISPLAYFORM4 Substituting the distributions into this equation results in Ł = −Ł reg + Ł pred with DISPLAYFORM5 DISPLAYFORM6 The term Ł reg can be identified as the sum of the KL-divergences between the GP prior on the virtual observation targets and their approximative posterior Q(U l ).

Since this term enters Ł with a negative sign, its purpose is to keep the approximative posterior close to the prior; thus it can be understood as a regularization term.

Evaluating using the formula for the KL-divergence between two normal distributions gives DISPLAYFORM7 The term Ł pred cannot be evaluated yet because the exact marginal Q(F L ) is still intractable.

Due to the central limit theorem the activation A l (weighted sum of inputs) of each GPN will converge to a normal distribution, if the number of incoming connections is sufficiently large (≥ 50) and the weights W l have a sufficiently random distribution.

For standard feed forward neural networks Wang & Manning (2013) experimentally showed that even after training the weights are sufficiently random for this assumption to hold.

Hence we postulate that the same is true for GPNs and assume that the marginal distributions Q(A l ), l ∈ {1, . . .

, L}, can be written as DISPLAYFORM0 A graphical model corresponding to this approximate posterior is shown in fig. 2b .

This allows the moments of Q(A l ) to be calculated exactly and propagated analytically from layer to layer.

For this purpose we need to evaluate the conditional distributions DISPLAYFORM1 DISPLAYFORM2 Since Q(F l | A l ) is the conditional of a GP with normally distributed observations, the joint distri- DISPLAYFORM3 and we can find the values for the unknown parameters µ DISPLAYFORM4 Thus by solving the resulting equations we obtain for eq. (24), DISPLAYFORM5 where DISPLAYFORM6 with DISPLAYFORM7 For deterministic observations, that is Σ DISPLAYFORM8 and thus recover the standard GP regression distribution as expected.

If U l follows its prior, that is µ DISPLAYFORM9 , we obtain K U l n = 0 and thus recover the GP prior on F l .

In that case the virtual observations behave as if they were not present.

DISPLAYFORM10 immediately allows us to evaluate eq. FORMULA2 since P(X l | F l ) just provides additive Gaussian noise; thus we obtain DISPLAYFORM11 Returning to Ł pred from (20) and writing it as DISPLAYFORM12 shows that we first need to obtain the distribution Q(A L ).

This is done by iteratively calculating the marginals Q(A l ) for l ∈ {1, . . .

, L}.

For l ≥ 1 the marginal distribution of the activations is DISPLAYFORM0 where Q(F l | A l ) is given by (26).

We first evaluate the mean and covariance of the marginal DISPLAYFORM1 For the marginal mean of the response µ DISPLAYFORM2 with DISPLAYFORM3 , which was calculated by expressing the squared exponential kernel as a normal PDF and applying the product formula for Gaussian PDFs BID2 .

For the marginal covariances of the response Σ we obtain by applying the law of total expectation DISPLAYFORM4 For the elements representing the variance, i.e. the diagonal n = n , this becomes DISPLAYFORM5 n and, using the same method as above, Ω l strn DISPLAYFORM6 For off-diagonal elements, n = n , we observe that F l sn and F l sn are conditionally independent given A l because the activation functions of GPNs n and n are represented by two different GPs.

Hence we have Σ DISPLAYFORM7 where DISPLAYFORM8 This concludes the calculation of the moments of F l .

We can now state how the activation distribution propagates from a layer to the next.

The marginal distribution of A l+1 is given by DISPLAYFORM9 Thus Q(A L ) in (31) can be calculated by iterating the application of eqs. (33), (35), (37), (39) and (40) over the layers l. To save computational power only the variances can be propagated by assuming that Σ F l s is diagonal and therefore ignoring (37).

Now Ł pred can be identified as the expected log-probability of the observations under the marginal distribution Q(F L ) and thus we can expand it as follows, DISPLAYFORM10 where S is the number of training samples and X L are the training targets.

The distribution Q(F L ) is of arbitrary form, but only its first and second moments are required to evaluate Ł pred .

For the first moment we obtain DISPLAYFORM11 DISPLAYFORM12 DISPLAYFORM13 This concludes the calculation of all terms of the variational lower bound (18).

The resulting objective is a fully deterministic function of the parameters.

Training of the model is performed by maximizing Ł w.r.t.

to the variational parameters µ DISPLAYFORM14 and the model parameters σ l , W l and V l .

This can be performed using any gradient-descent based algorithm.

The necessary derivatives are not derived here and it is assumed that this can be performed automatically using symbolic or automatic differentiation in an appropriate framework.

The activation function are represented using 2R variational parameters per GPN, where R is the number of inducing points and targets.

It can be shown that R = 10 linearly spaced inducing points are enough to represent the most commonly used activation functions (sigmoid, tanh, soft ReLU) with very high accuracy.

The number of required parameters can be reduced by sharing the same activation function within groups of neurons or even across whole layers of neurons.

If the inducing points V l are fixed (for example by equally distributing them in the interval [−1, 1]), the kernel matrices K(V l , V l ) and their inverses can be precomputed since they are constant.

The number of parameters and the computational complexity of propagating the means and covariances only depend on R and are therefore independent of the number of training samples.

Thus, like a conventional neural network, a GPN network can inherently be trained on datasets of unlimited size.

We have presented a non-parametric model based on GPs for learning of activation functions in a multi-layer neural network.

We then successively applied variational to make fully Bayesian inference feasible and efficient while keeping its probabilistic nature and providing not only best guess predictions but also confidence estimations in our predictions.

Although we employ GPs, our parametric approximation allows our model to scale to datasets of unlimited size like conventional neural networks do.

We have validated networks of Gaussian Process Neurons in a set of experiments, the details of which we submit in a subsequent publication.

In those experiments, our model shows to be significantly less prone to overfitting than a traditional feed-forward network of same size, despite having more parameters.

<|TLDR|>

@highlight

We model the activation function of each neuron as a Gaussian Process and learn it alongside the weight with Variational Inference.

@highlight

Propose placing Gaussian process priors on the functional form of each activation function in the neural net to learn the form of activation functions.