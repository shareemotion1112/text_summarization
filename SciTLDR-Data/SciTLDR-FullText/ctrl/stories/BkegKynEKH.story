We present a probabilistic framework for session based recommendation.

A latent variable for the user state is updated as the user views more items and we learn more about their interests.

We provide computational solutions using both the re-parameterization trick and  using the Bouchard bound for the softmax function, we further explore employing a variational auto-encoder and a variational Expectation-Maximization algorithm for tightening the variational bound.

Finally we show that the Bouchard bound causes the denominator of the softmax to decompose into a sum enabling fast noisy gradients of the bound giving a fully probabilistic algorithm reminiscent of word2vec and a fast online EM algorithm.

Our model describes a generative process for the types of products that user's co-view in sessions.

We use u to denote a user or a session, we use t time to denote sequential time and v to denote which product they viewed from 1 to P where P is the number of products, the user's interest is described by a K dimensional latent variable ω u which can be interpreted as the user's interest in K topics.

The session length of user u is given by T u .

We then assume the following generative process for the views in each session: ω u ∼ N (0 K , I K ), v u,1 , .., v u,Tu ∼ categorical(softmax(Ψω u + ρ)).

This model is a linear version of the model presented in Liang et al. (2018) .

Consider the case where we have estimated that Ψ and ρ.

In production we have observed a user's online viewing history v u,1 , .., v u,Tu and we would like to produce a representation of the user's interests.

Our proposal is to use Bayesian inference in order to infer p(ω|v u,1 , .., v u,Tu , Ψ, ρ) as a representation of their interests.

This representation of interests can then be used as a feature for training a recommender system.

If we have a recommender system that has just seven products and the products have embeddings: We now consider how different user histories affect p(ω|v u,1 , .., v u,Tu , Ψ, ρ).

Approximation of this quantity can be made accurately and easily using the Stan probabilistic programming language Stan Development Team (2018) or using variational approximations.

In Figure 1 -2 the intuitive behavior of this simple model is demonstrated.

The results of the three approximate methods are presented and shown to be in good agreement.

A single product view indicates interest in that class of products, but considerable uncertainty remains.

Many product views in the same class represent high certainty that the user is interested in that class.

For next item prediction we consider both taking the plug-in predictive based on the posterior mean VB (approx) and using Monte Carlo samples to approximate the true predictive distribution MCMC and VB (MC).

The model we introduce has the form:

If we use a normal distribution ω ∼ N (µ q , Σ q ), then variational bound has the form:

but we still need to be able to integrate under the denominator of the softmax.

The Bouchard bound Bouchard (2007) introduces a further approximation and additional variational parameters a, ξ but produces an analytical bound:

Where λ JJ (·) is the Jaakola and Jordan function Jaakkola and Jordan (1997) :

.

Alternatively the re-parameterization trick Kingma and Welling (2014) proceeds by simulating: (s) ∼ N (0 K , I K ), and then computing:

, and then optimizing the noisy lower bound:

In order to prevent the variational parameters growing with the amount of data we employ a variational auto-encoder.

This involves using a flexible function i.e. µ q , Σ q = f Ξ (v 1 , ...v T ), or in the case of the Bouchard bound:

Where any function (e.g. a deep net) can be used for f Ξ (·) and f Bouch Ξ (·).

We demonstrate that our method using the RecoGym simulation environment Rohde et al. (2018) .

We fit the model to the training set, we then evaluate by providing the model v 1 , ..v Tu−1 events and testing the model's ability to predict v Tu .

A further approximate algorithm which is useful when P is large is to note that the bound can be written as a sum that decomposes not only in data but also over the denominator of the softmax, The noisy lower bound becomes:

where v 1 , ..., v T are the items associated with the session and n 1 , ...n S are S < P negative items randomly sampled.

Similar to the word2vec algorithm Mikolov et al. (2013) but without any non-probabilistic heuristics.

We consider two alternative methods for training the model: Bouch/AE -A linear variational auto-encoder using the Bouchard bound; RT/Deep AE -A deep auto-encoder again using the re-parameterization trick.

The deep auto-encoder consists of mapping an input of size P to three linear rectifier layers of K units each.

Results showing recall@5 and discounted cumulative gain at 5 are shown in Table 1 .

The EM algorithm allows an approximation to be made of q(ω) assuming (Ψ, ρ) and a user history v 1 , .., v T are known and can be used in place of a variational auto-encoder.

The algorithm here is the dual of the one presented in Bouchard (2007) as we assume the embedding Ψ is fixed and ω is updated where the algorithm they present does the opposite.

The EM algorithm consists of cycling the following update equations:

We further note that the EM algorithm is (with the exception of the a variational parameter) a fixed point update (of the natural parameters) that decomposes into a sum.

The terms in the sum come from the softmax in the denominator.

After substituting a co-ordinate descent update of a with a gradient descent step update, then the entire fixed point update becomes a sum:

That is the EM algorithm can be written:

As noted in Cappé and Moulines (2009) when an EM algorithm can be written as a fixed point update over a sum, then the Robbins Monro algorithm can be applied.

Allowing updates of the form (p is chosen randomly):

where ∆ is a slowly decaying Robbins Monro sequence (Robbins and Monro (1951) ) with ∆ 1 = 1 (meaning no initial value of (Σ −1 ) ) is needed.

For large P this algorithm is many times faster than the generic EM algorithm.

What is distinct about both this online EM algorithm and the negative sampling SGD approach is that it is the denominator of the softmax that may be sub-sampled rather than individual records.

The Bouchard bound is also used for decomposing the softmax into a sum in Titsias (2016) but they do per-batch optimization of the variational parameters, instead we use an auto-encoder allowing direct SGD.

Our method also differs from Ruiz et al. (2018) again in using an auto-encoder allowing the more flexible SGD algorithm in place of stochastic variational inference (Hoffman et al. (2013) ) which requires complete data exponential family assumptions.

Finally unlike those methods we are considering variational inference of a latent variable model as well as using variational bounds to approximate the softmax.

<|TLDR|>

@highlight

Fast variational approximations for approximating a user state and learning product embeddings