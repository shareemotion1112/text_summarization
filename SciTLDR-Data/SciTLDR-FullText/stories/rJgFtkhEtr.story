We introduce a neural architecture to perform amortized approximate Bayesian inference over latent random permutations of two sets of objects.

The method involves approximating permanents of matrices of pairwise probabilities using recent ideas on functions defined over sets.

Each sampled permutation comes with a probability estimate, a quantity unavailable in MCMC approaches.

We illustrate the method in sets of 2D points and MNIST images.

Posterior inference in generative models with discrete latent variables presents well-known challenges when the variables live in combinatorially large spaces.

In this work we focus on the popular and non-trivial case where the latent variables represent random permutations.

While inference in these models has been studied in the past using MCMC techniques (Diaconis, 2009 ) and variational methods , here we propose an amortized approach, whereby we invest computational resources to train a model, which later is used for very fast posterior inference (Gershman and Goodman, 2014) .

Unlike the variational autoencoder approach (Kingma and Welling, 2013) , in our case we do not learn a generative model.

Instead, the latter is postulated (through its samples) and posterior inference is the main focus of the learning phase.

This approach has been recently explored in sundry contexts, such as Bayesian networks (Stuhlmüller et al., 2013) , sequential Monte Carlo (Paige and Wood, 2016) , probabilistic programming (Ritchie et al., 2016; Le et al., 2016) , neural decoding (Parthasarathy et al., 2017) and particle tracking (Sun and Paninski, 2018) .

Our method is inspired by the technique introduced in (Pakman and Paninski, 2018 ) to perform amortized inference over discrete labels in mixture models.

The basic idea is to use neural networks to express posteriors in the form of multinomial distributions (with varying support) in terms of fixed-dimensional, distributed representations that respect the permutation symmetries imposed by the discrete variables.

After training the neural architecture using labeled samples from a particular generative model, we can obtain independent approximate posterior samples of the permutation posterior for any new set of observations of arbitrary size.

These samples can be used to compute approximate expectations, as high quality importance samples, or as independent Metropolis-Hastings proposals.

Let us consider the generative model

Here p(c 1:N ) = 1/N ! is a uniform distribution over permutations, with the random variable

denoting that x c i is paired with y i .

As a concrete example, think of y i as a noise-corrupted version of a permuted sample x c i .

Given two sets of N data points x = {x i }, y = {

y i }, we are interested in iid sampling the posterior of the c i 's, using a decomposition

note now that p(c N |c 1:N −1 , x, y) = 1, since the last point y N is always matched with the last unmatched point among the x i '

s. A generic factor in (2) is

where c n takes values in {1, . . .

, N } not taken by c 1:n−1 .

Consider first

where we defined

and s n = {1 . . .

N }/{c 1 . . .

c n } is the set of available indices after choosing c 1:n .

Note that R in (5) is the permanent of a (N − n)×(N − n) matrix, an object whose computation is known to be a #P problem (Valiant, 1979) .

Inserting (4) into (3) gives

Note that (6) does not depend on {x c i }

, except for restricting the allowed values for c n .

Now, the function R in (5) depends on the unmatched points {x c i } N i=n+1 , and {y i }

N i=n+1 , in such a way that it is invariant under separate permutations of the elements of each set.

Following (Zaheer et al., 2017; Gui et al., 2019) , these permutation symmetries can be captured by introducing functions h :

and approximating R(c n+1:N , x, y|c 1 . . .

c n ) e f (Hx,c n ,Hy) .

The subindex c n in H x,cn indicates a value that cannot be taken by the c i 's in the sum in (7).

Inserting (8) into (6) gives q θ (c n |c 1:n−1 , x, y) = p(x cn , y n )e f (Hx,c n ,Hy) c n p(x c n , y n )e f (H x,c n ,Hy)

which is our proposed approximation for (6), with θ representing the parameters of the neural networks for h, f .

The neural architecture is schematized in Figure 2 .

The pairwise density p(y n , x cn ) can be either known in advance, or represented by a parametrized function to be learned (in the latter case we assume we have samples from it).

We call this approach the Neural Permutation Process (NPP).

In order to learn the parameters θ of the neural networks h, f (and possibly p(x cn , y n )), we use stochastic gradient descent to minimize the expected KL divergence,

log q θ (c n |c 1:n−1 , x, y) + const.

Samples from p(N )p(c 1:N , x, y) are obtained from the generative model (1).

If we can take an unlimited number of samples, we can potentially train a neural network to approximate p(c n |c 1:n−1 , x) arbitrarily accurately.

In Figure 1 we show results for the following two examples.

Both cases illustrate how a probabilistic approach captures the ambiguities in the observations.

Noisy pairs in 2D: the generative model is

MNIST digits: the generative model is

An additional symmetry.

Note finally that if p(y n , x cn ) = p(x cn , y n ) (as is the case in these examples), the additional symmetry f (H x,cn , H y ) = f (H y , H x,cn ) can be captured by introducing a new function g and defining f (H x,cn , H y ) = f (g(H x,cn )+g(H y )).

Interestingly, as shown in Figure 2 , we find that a higher likelihood is obtained instead by f (H x,cn , H y ) = f (H x,cn H y ), where indicates componentwise multiplication.

To our knowledge, this type of encoding has not been studied in the literature, and we plan to explore it further in the future.

Our results on simple datasets validate this approach to posterior inference over latent permutations.

More complex generative models with latent permutations can be approached using similar tools, a research direction we are presently exploring.

The curves show mean training negative log-likelihood/iteration in the MNIST example.

f = 0 is a baseline model, were we ignore the unassigned points in (9).

The other two curves correspond to encoding the symmetry p(y n , x cn ) = p(x cn , y n ) as f (g(H x,cn ) + g(H y )) or as f (H x,cn H y ).

@highlight

A novel neural architecture for efficient amortized inference over latent permutations 