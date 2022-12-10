The variational autoencoder (VAE) is a popular combination of deep latent variable model and accompanying variational learning technique.

By using a neural inference network to approximate the model's posterior on latent variables, VAEs efficiently parameterize a lower bound on marginal data likelihood that can be optimized directly via gradient methods.

In practice, however, VAE training often results in a degenerate local optimum known as "posterior collapse" where the model learns to ignore the latent variable and the approximate posterior mimics the prior.

In this paper, we investigate posterior collapse from the perspective of training dynamics.

We find that during the initial stages of training the inference network fails to approximate the model's true posterior, which is a moving target.

As a result, the model is encouraged to ignore the latent encoding and posterior collapse occurs.

Based on this observation, we propose an extremely simple modification to VAE training to reduce inference lag: depending on the model's current mutual information between latent variable and observation, we aggressively optimize the inference network before performing each model update.

Despite introducing neither new model components nor significant complexity over basic VAE, our approach is able to avoid the problem of collapse that has plagued a large amount of previous work.

Empirically, our approach outperforms strong autoregressive baselines on text and image benchmarks in terms of held-out likelihood, and is competitive with more complex techniques for avoiding collapse while being substantially faster.

@highlight

To address posterior collapse in VAEs, we propose a novel yet simple training procedure that aggressively optimizes inference network with more updates. This new training procedure mitigates posterior collapse and leads to a better VAE model. 

@highlight

Looks into the phenomenon of posterior collapse, showing that increased training of the inference network can reduce the problem and lead to better optima.

@highlight

Authors propose changing the training procedure of VAEs only as a solution to posterior collapse, leaving the model and objective untouched.