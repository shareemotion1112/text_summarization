We formulate stochastic gradient descent (SGD) as a novel factorised Bayesian filtering problem, in which each parameter is inferred separately, conditioned on the corresopnding backpropagated gradient.

Inference in this setting naturally gives rise to BRMSprop and BAdam: Bayesian variants of RMSprop and Adam.

Remarkably, the Bayesian approach recovers many features of state-of-the-art adaptive SGD methods, including amongst others root-mean-square normalization, Nesterov acceleration and AdamW.  As such, the Bayesian approach provides one explanation for the empirical effectiveness of state-of-the-art adaptive SGD algorithms.

Empirically comparing BRMSprop and BAdam with naive RMSprop and Adam on MNIST, we find that Bayesian methods have the potential to considerably reduce test loss and classification error.

Deep neural networks have recently shown huge success at a range of tasks including machine translation BID37 , dialogue systems BID29 , handwriting generation BID9 and image generation BID26 .

These successes have been facilitated by the development of a broad range of adaptive SGD methods, including ADAGrad BID6 , RMSprop BID13 , Adam BID16 , and variants thereof, including Nesterov acceleratation (Nesterov, 1983; BID2 BID5 and AdamW BID18 .

However, such a broad range of approaches raises the question of whether it is possible to obtain a unified theoretical understanding of adaptive SGD methods.

Here we provide such a theory by reconciling state-of-the-art adaptive SGD algorithms with very early work that used Bayesian (Kalman) filtering to optimize the parameters of neural networks BID23 BID30 BID24 BID25 BID7 BID22 .There have recently been attempts to connect adaptive SGD algorithms to natural gradient variational inference (VI) BID39 BID14 .

These approaches give a momentum-free algorithm with a mean-square normalizer, in contrast to perhaps the most popular adaptive method, Adam BID16 , which combines momentum with a root-meansquare normalizer.

To achieve a closer match to Adam, they modified their natural gradient VI updates, without a principled justification based on approximate inference, to incorporate momentum BID39 BID15 , and the root-mean-square normalizer BID14 .

As such, there appears to be only a loose connection between successful adaptive SGD algorithms such as Adam, and natural gradient VI.There is a formal correspondence between natural gradient VI BID39 BID14 and Bayesian filtering BID22 .

While BID22 did not examine the relationship between their filtering updates and RMSprop/Adam, the equivalence of this particular filtering approach and natural gradient VI indicates that they would encounter the issues described above, and thus be unable to obtain momentum or the root-mean-square normalizer BID39 BID14 .

More problematically, BID22 introduces dynamics into the Kalman filter, but these dynamics correspond to the "addition of an artificial process noise Q t proportional to [the posterior covariance] P t−1 ".

Thus, their generative model depends on inferences made under that model: a highly unnatural assumption that most likely does not correspond to any "real" generative process.

DISPLAYFORM0 Figure 1: The heirarchy of generative models underlying our updates.

A Full model for the gradients for a single parameter.

The current estimate for all the other parameters, µ −i (t) vary slowly over time, and give rise to the current optimal value for the ith parameter, w * i .

The gradient then arises from the current estimate of the ith parameter, µ i (t) (which is treated as an input here), and the optimal value, ith parameter, w * i .

B The graphical model obtained by integrating over trajectories for the other parameter estimates, µ −i (t).

In practice, we use a simplified model as reasoning about all possible trajectories of µ −i (t) is intractable.

C To convert the model in B into a tractable hidden Markov model (HMM), we define a new variable, z i (t), which incorporates w * i along with other information about the dynamics.

How might we obtain a principled Bayesian filtering approach that recovers the two key features of state-of-the-art adaptive SGD algorithms: momentum and the root-mean-square normalizer?

Here, we note that past approaches including natural gradient VI take a complex generative model over all N parameters jointly, and use a very strong approximation: factorisation.

Given that we know that the true posterior is a highly complicated, correlated distribution, it is legitimate to worry that these strong approximations might meaningfully disrupt the ability of Bayesian filtering to give closeto-optimal updates.

Here we take an alternative approach, baking factorisation into our generative model, so that we can use Bayesian inference to reason about (Bayes) optimal updates under the constraints imposed by factorisation.

In particular, we split up the single large inference problem over all N parameters, w, into N small inference problems over a single parameter.

Remarkably, by incorporating factorisation into the problem setting, we convert intractable, high-dimensional correlations in the original posterior into tractable low-dimensional dynamics in the factorised model.

This dynamical prior has a "natural" form, at least compared with BID22 , in that it does not depend on the posterior.

Next, we give a generic derivation showing that Bayesian SGD is an adaptive SGD method, where the uncertainty is used to precondition the gradient.

We then adapt the generic derivation to the two cases of interest: RMSprop BID13 and Adam BID16 .

Finally, we discuss the general features of Bayesian adaptive SGD methods, including AdamW BID18 and Nesterov acceleration (Nesterov, 1983; BID5 , amongst others.

The most obvious factorised approach is to compute the true posterior, P (w i |D), over a single weight, w i , conditioned on all the data, D. However, this approach immediately fails, because symmetries involving permuting the hidden units BID34 imply that marginal posteriors such as P (w i |D) are always so broad as to be useless.

To obtain an informative, narrow posterior, one approach is to eliminate these symmetries by conditioning on our current estimate of the other parameters.

In particular, we define a random variable, w * i , representing the optimal value for the ith weight, conditioned on the other weights, w −i , being equal to our current estimate of those parameters, µ −i , DISPLAYFORM0 where d is a random minibatch drawn from the true, unknown underlying data distribution (and not just the data we have), so this is analogous to maximum-likelihood with infinite data.

As such, even if µ −i is fixed, w * i will be unknown if only have finite data, and therefore we do not know the true underlying data distribution.

The optimal weight, w * i , is a random variable because it depends on our current estimate of the other weights, µ −i , which is a random variable because it depends on a random initialization, and potentially on a randomised optimization algorithm.

However, we cannot now use the usual Bayesian approach of inferring w * i based on the data, because, in the Bayesian setting, the data are assumed to be generated from a model with parameters, w, not w * i .

Instead, note that the updates for almost all neural network optimization algorithms depend only on the current value of that parameter, and the (history of) backpropagated gradients.

This suggests a potential approach: writing down a generative model for the backpropagated gradients that depends on w * i , then inverting that model to infer w * i from those gradients.

Following standard Laplace/Extended Kalman Filter-like approximations (e.g. BID39 BID14 BID22 BID15 as closely as possible, we approximate the likelihood as a second-order Taylor series expansion.

In particular, we consider the log-likelihood for a single minibatch, d, where the minibatch is treated as a random variable drawn from the underlying true data distribution, and we expand in the ith direction, keeping the other parameters, w −i , fixed to their current estimates, µ −i , DISPLAYFORM1 where, DISPLAYFORM2 DISPLAYFORM3 Here, both the mode, µ like,i , and the negative Hessian, Λ like,i , are random variables as they are deterministic functions of the minibatch, d, and the current estimate of all the other parameters, µ −i , which are both treated as random variables.

As such, the gradient of the log-likelihood, which depends on Λ like,i and µ −i , is also a random variable, DISPLAYFORM4 where the approximation comes from the second-order Taylor expansion in Eq. (2).

Our goal is to understand the distribution of the gradients, conditioned on the random variable Λ like,i being equal to some specific value, λ like,i .

In practice, we will set Λ like,i , using the standard approximations in the natural gradient literature BID39 BID14 .

The expected gradient is zero at µ i = w * i , DISPLAYFORM5 where the approximation again comes only from the second-order Taylor expansion.

Thus, DISPLAYFORM6 with equality if the expected value of µ like,i is independent of Λ i , Substituting this into Eq. FORMULA5 , we obtain, DISPLAYFORM7 However, obtaining an analytic form for the full distribution of the gradient is more difficult.

Given that all work in this domain makes fairly strong assumptions (e.g. BID39 BID14 , an alternative approach opens up: choosing a model for the gradients to match as closely as possible any given approximation of the original likelihood in Eq. FORMULA2 .

In particular, we take, DISPLAYFORM8 Note that the variance is the negative Hessian, λ like,i , which is reminiscent of Fisher-Information like results (Sec. 9.7).

However, we chose this form such that, for any given approximation to the original log-likelihood (Eq. 2), the log-likelihood induced by conditioning on the gradient (Eq. 9) has the same form, DISPLAYFORM9 The only material difference between the right-hand-side of this expression and the original likelihood (Eq. 2) is that here the parameter of interest is the optimal weight, w * i , as opposed to the weight in the underlying generative model, w i .

Also note that here we have specified the values for all the random variables: g i = g i and Λ like,i = λ like,i which fix the value of µ like,i to µ like = gi λlike,i + µ i .

The key difference between the methods is that in the original problem, where we infer a single distribution over all weights jointly, the underlying weights, w, were fixed, and as such, Bayesian filtering reduces to recursively applying Bayes theorem, which does not give rise to interesting dynamics.

In contrast, when we consider N separate inference problems, in which w * i is inferred from the gradient, then we are forced to introduce dynamics, because w * i varies over time, as it depends on all the other parameters in the network, µ −i , which are also being optimized (Fig. 1A) .

However, the full generative process in Fig. 1A is very difficult to reason about, as it requires us to integrate over all possible trajectories for the other parameters, µ −i .

Instead, we write down simplified approximate dynamics over w * i directly (Fig. 1B) , and evaluate the quality of these simplified dynamics empirically.

We choose the model in Fig. 1 such that it has finite-length temporal dependencies, and hence can be written in terms of a Markov model with an expanded state-space.

In particular, we define a new random variable, z i , incorporating w * i , whose generative process is Markovian (Fig. 1C ).

For Adam, z i will have two elements representing a parameter, w * i and the associated momentum, p i , whereas for RMSprop it will have only one element representing just the parameter, DISPLAYFORM10 As z i represents the optimal setting for a parameter it will change over time, and we assume a simple Gaussian form for these changes, DISPLAYFORM11 where Q is the covariance of Gaussian perturbations and A is the dynamics matrix incorporating weight decay and momentum.

We can write the second-order Taylor expansion of the likelihood as a function of the expanded latent, z i (Eq. 11), DISPLAYFORM12 where we have omitted conditioning on Λ like,i , as in practice Λ like,i is estimated from the gradients, and we have omitted time indices for clarity, DISPLAYFORM13 .

This likelihood is equivalent to our original likelihood (Eq. 10) because the gradients depend only on w * i (t).

As such, the negative Hessians must take the form, DISPLAYFORM14 3 BAYESIAN (KALMAN) FILTERING AS ADAPTIVE SGDThe Gaussian prior and approximate likelihood allows us to use standard two-step Kalman filter updates.

First, we propagate the previous time-step's posterior, P (z i (t − 1)|D(t − 1)), with mean µ post,i (t − 1) and covariance Σ post,i (t − 1), forward in time to obtain a prior at time t, DISPLAYFORM15 where, DISPLAYFORM16 DISPLAYFORM17 and where DISPLAYFORM18 } is all gradients up to time t − 1.

Note that we have also defined abbreviations for µ prior,i (t) and Σ prior,i (t), omitting the temporal index, t. Second we use Bayes theorem to incorporate new data, DISPLAYFORM19 where, DISPLAYFORM20 DISPLAYFORM21 Thus far, we have simply repeated standard Kalman filtering results.

To relate Kalman filtering to gradient ascent, we compute the gradient of Eq. (13) at z i = µ prior,i , DISPLAYFORM22 Note that as the log-likelihood depends only on w * i , we have, DISPLAYFORM23 Now, we identify this gradient (Eq. 17) in the mean updates (Eq. 16b), DISPLAYFORM24 This form is extremely intuitive: it states that the uncertainty should be used to precondition gradient updates, such that the updates are larger when there is more uncertainty, and smaller when past data gives high confidence in the current estimates.

As the precision is always rank-1 (Eq. 14), we can always write it as, DISPLAYFORM25 As such, the updates for the posterior covariance (Eq. 16a) can be rewritten using the Sherman Morrison formula BID11 , DISPLAYFORM26 To estimate e i , we use the Fisher information (see SI Sec. 9.7).

In particular, we could use the gradient itself, but could also (under weaker approximations) use the centred gradient BID9 , or the gradient for data sampled under the prior.

Here, we develop a Bayesian variant of RMSprop, which we call BRMSprop.

We consider each parameter to be inferred by considering a separate Bayesian inference problem, so the latent variable, z = w * , is a single scalar, representing a single parameter (we omit the index i on z for brevity).

We use A = η 2 /(2σ 2 ) and Q = η 2 giving a dynamical prior, DISPLAYFORM0 such that the stationary distribution over z has standard-deviation σ.

To obtain a complete description of our updates, we substitute these choices into the updates for the prior (Eq. 15) and the posterior (Eq. 19 and Eq. 21), DISPLAYFORM1 DISPLAYFORM2 For an efficient implementation of the full algorithm, see SI Alg.

1.

Now we show that with in steady state, BRMSprop closely approximates RMSprop.

Making this comparison is non-trivial because the "additional" variables in RMSprop and BRMSprop (i.e. the average squared gradient and the uncertainty respectively) are not directly comparable.

However, the implied learning rate is directly comparable.

We therefore look at the learning rate when the average squared gradient and the uncertainty have reached steady-state.

As t → ∞, we expect σ Substituting this form into Eq. (23) recovers the root-mean-square normalization used in RMSprop.

We now develop a Bayesian variant of Adam BID9 BID16 , which we call BAdam.

To introduce momentum into our Bayesian updates, we introduce an auxiliary momentum variable, p(t), corresponding to each parameter, w * (t), DISPLAYFORM0 , then we infer both the parameter and momentum jointly.

Under the prior, the momentum, p(t), evolves through time independently of w * (t), obeying an AR(1) (or equivalently a discretised Ornstein-Uhlenbeck) process, with decay η p and injected noise variance η DISPLAYFORM1 where ξ p (t) is standard Gaussian noise.

This particular coupling of the injected noise variance and the decay rate ensures that the influence of the momentum on the weight is analogous to unitvariance Gaussian noise (see SI Sec. 9.2).

The weight obeys similar dynamics, with the addition of a momentum-dependent term which in practice causes changes in w * i (t) to be correlated across time (i.e. multiple consecutive increases or decreases in w * i (t)), DISPLAYFORM2 where ξ w (t) is again standard Gaussian noise, η is the strength of the momentum coupling, DISPLAYFORM3 is the strength of the weight decay, and η 2 w is the variance of the noise injected to the weight.

It is possible to write these dynamics in the generic form given above (Eq. 12), by using, DISPLAYFORM4 and these settings fully determine the updates, according to Eq. FORMULA1 and Eq. (16).

For an efficient implementation of the full algorithm, see SI Alg.

2.

Now we show that with suitable choices for the parameters, BAdam closely approximates Adam.

In particular, we compare the updates for the (posterior) parameter estimate, µ w , and the momentum, µ p , when we eliminate weight decay by setting σ 2 = ∞, and eliminate noise injected directly into the weight by setting η w = 0, DISPLAYFORM0 DISPLAYFORM1 These updates depend on two quantities, Σ ww and Σ wp , which are related to e 2 , but have no direct analogue in standard Adam.

As such, to make a direct comparison, we use the same approach as we used previously for RMSprop: we compare the updates for the parameter and momentum, when Σ ww , Σ wp in BAdam and e 2 in Adam have reached their steady-state values.

To find the steadystates for Σ ww and Σ wp , we again use the simplified covariance updates derived in SI Sec. 9.1, DISPLAYFORM2 Substituting Eq. FORMULA2 and Eq. FORMULA2 into Eq. FORMULA2 , and again using σ 2 = ∞ and η 2 w = 0, we obtain, DISPLAYFORM3 We now assume that the data is informative, in the sense that it is strong enough to give a narrow posterior relative to the prior (without which any neural network training algorithm is unlikely to be able to obtain good performance).

This implies Σ pp η p /2 (see SI Sec. 9.2), allowing us to solve for Σ wp , using the lower-right element of the above expression (i.e. η 2 p ≈ 2η p Σ pp + e 2 Σ 2 wp ), DISPLAYFORM4 To recover a very close approximation to Adam, we need a specific relationship between learning rates and evidence strength, such that, DISPLAYFORM5 While this may seem restrictive, it is only necessary to achieve the closest possible match between Bayesian filtering and plain Adam.

However, our goal is not to match Adam exactly, given that Adam does not even converge BID27 .

Instead, our goal is to capture the essential dynamical insights of Adam in a Bayesian method.

Indeed, we hope that the resulting Bayesian method constitutes an improvement over Adam, which implies that it must exhibit some differences.

Nonetheless, focusing on the regime where filtering and Adam are most similar, the filtering updates become, DISPLAYFORM6 e 2 ≈ µ w (t) + ηµ p (t + 1) (32a) DISPLAYFORM7 where we have substituted for µ p (t + 1) into Eq. 32a, which assumes that η p 1.

This is very close to the Adam updates, except that the root-mean-square normalization is in the momentum updates, rather than the parameter updates.

To move the location of the root-mean-square normalization, we rewrite the updates in terms of a rescaled momentum, DISPLAYFORM8 giving, DISPLAYFORM9 DISPLAYFORM10 which recovers Adam.

See SI Sec. 9.6 for a similar discussion for NAG/NAdam

We compared Bayesian and standard methods on MNIST.

In particular, we trained a CNN with relu nonlinearities and maxpooling for 50 epochs.

The network had two convolutional layers with 10 channels in the first layer and 20 in the second, with 5 × 5 convolutional kernels, followed by a single fully connected layer with 50 units, and was initialized with draws from a Gaussian with variance 2/N inputs .

The network did not use dropout (or any other form of stochastic regularisation), which would increase the variance of the gradients, artificially inflating e 2 .The key Bayesian-specific parameters are the variance of the stationary distribution and the initial uncertainty; both of which were set to 1/(2N inputs ).

In principle, they should be similar to the initialization variance, but in practice we found that using a somewhat lower value gave better performance, though this needs further investigation.

For the RMSprop and Adam specific parameters, we used the PyTorch defaults.

Comparing the Bayesian and non-Bayesian methods, we do not expect to see very large discrepancies, because they are approximately equivalent in steady-state.

Nonetheless, the Bayesian methods show considerably lower test loss, somewhat lower classification error, and similar training loss and training error FIG3 .

The local maximum in test-loss may arise because we condition on each datapoint multiple times, which is not theoretically justified (correcting this is non-trivial in the dynamical setting, so we leave it for future work).

Here we summarise the features of Bayesian stochastic gradient descent schemes, pointing out where we recover previously known best practice and where our approach suggests new ideas.7.1 WEIGHT DECAY, L2 REGULARIZATION AND BAYESIAN PRIORS BID18 recently examined different forms of weight decay in adaptive SGD methods such as Adam.

In particular, they asked whether or not the root-mean-square normalizer should be applied to the weight decay term.

In common practice, we take weight decay as arising from the gradient of an L2 regularizer, in which case it is natural to normalize the gradient of the objective and the gradient of the regularizer in the same way.

However, there is an alternative: to normalize only the gradient of the objective, but to keep the weight decay constant (in which case, weight decay cannot be interpreted as arising from an L2 normalizer).

BID18 show that this second method, which they call AdamW, gives better test accuracy than the standard approach.

Remarkably, AdamW arises naturally in BRMSprop and BAdam (e.g. Eq. 23), providing a potential explanation for its improved performance.

In Nesterov accelerated gradients (NAG), we compute the gradient at a "predicted" location formed by applying a momentum update to the current parameters (Nesterov, 1983) .

These updates arise naturally in our Bayesian scheme, as the required gradient term (Eq. 17) is evaluated at µ prior (Eq. 15a), which is precisely a prediction formed by combining the current setting of the parameters, µ post (t), with momentum and decay, embodied in A (Eq. 27) to form a prediction, µ prior (t + 1).

It should be noted that the original Nesterov acceleration (Nesterov, 1983) had no gradient preconditioning.

Instead, our approach corresponds to NAdam BID5 .

Interestingly, as we also implement weight decay through the dynamics matrix, A, we should also apply the updates from weight decay before computing the gradient, giving, to our knowledge, a novel method that we denote "Nesterov accelerated weight decay" (NAWD).

A series of recent papers have discussed the convergence properties of RMSprop and Adam, noting that they may fail to converge BID36 BID27 if the exponentially decaying average over the squared gradients is computed with a too-small timescale BID27 .Our method circumvents these issues by coupling the learning rate to the timescale over which the implicit exponential moving average for the mean-square gradient is performed.

As such, in the limit as the learning rates go to zero (i.e. η → 0 for BRMSprop and η w → 0 and η p → 0 for BAdam), our method becomes SGD with adaptive learning rates that scale as 1/t and is therefore likely to be convergent for convex functions BID28 BID3 , though we leave a rigorous proof to future work.

ADAM BID16 first updates the root-mean-square gradient normalizer before computing the parameter update.

This is important, because it ensures that updates are bounded in the pathological case that gradients are initially all very small, such that the root-mean-square normalizer is also small, and then there is a single large gradient.

Bayesian filtering naturally recovers this choice as the gradient preconditioner in Eq. FORMULA1 is the posterior, rather than the prior, covariance (i.e. updated with the current gradient).

We believe that it should be possible to combine the temporal changes induced by the factorised approximations with other types of inference, including probabilistic backpropagation BID12 or assumed density filtering BID8 .

At present, our method bears closest relation to natural gradient variational inference methods BID39 BID15 , as they also use the Fisher Information to approximate the likelihood.

Indeed, our method becomes equivalent to these approaches in the limit as we send the learning rate, η to zero.

The key difference is that because they do not formulate a factorised generative model, they are unable to provide a strong justification for the introduction of rich dynamics, and they are unable to reason about optimal inference under these dynamics.

Bayesian filtering presents a novel approach to neural network optimization, and as such, there are variety of directions for future work.

First, Bayesian filtering converts the problem of neural network optimization into the statistical problem of understanding the dynamics of changes in the optimal weight induced by optimization in the other parameters.

In particular, we can perform an empirical investigation in large scale systems, or attempt to find closed-form expressions for the dynamics in simplified domains such as linear regression.

Second, here we wrote down a statistical model for the gradient.

However, there are many circumstances where the gradient is not available.

Perhaps a low precision or noisy gradient is available due to noise in the parameters (e.g. due to dropout BID33 , or perhaps we wish to consider a biological setting, where the gradient is not present at all BID0 .

The Bayesian approach presented here gives a straightforward recipe for developing (Bayes) optimal algorithms for such problems.

Third, stochastic regularization has been shown to be extremely effective at reducing generalization error in neural networks.

This Bayesian interpretation of adaptive SGD methods presents opportunities for new stochastic regularization schemes.

Fourth, it should be possible to develop filtering methods that represent the covariance of a full weight matrix by exploiting Kronecker factorisation BID19 BID10 BID39 µ ← µ + Σg µ = µ post (t) 8:ĝ ← (1 − η g )ĝ + η g g Update average gradient 9: µ ← 1 − η 2 /(2σ 2 ) µ µ = µ prior (t + 1) 11: end while 12: return µ DISPLAYFORM0

We require that the initialization for Σ varies across parameters, according to the number of inputs (as in typical neural network initialization schemes).

While this is possible in automatic differentiation libraries, including PyTorch, it is extremely awkward.

As such, we reparameterise the network, such that all parameters are initialized with a draw from a standard Gaussian, and to ensure that the outputs have the same scale, we explicitly scale the output.

To begin, we note that accelerated stochastic gradient descent remains an open research problem, in which the optimality of plain NAG is unclear BID4 .

As such, our goal is again, not to match NAG exactly, but to capture its essential insights within a Bayesian framework, so that we can suggest improved methods.

To highlight the link between our approach and Nesterov accelerated gradient, we rewrite Eq. (32) in terms of v(t + 1) = ηµ p (t + 1), µ w (t) ≈ µ w (t − 1) + v(t) (53a) DISPLAYFORM0 which results in updates with a very similar form to standard momentum and Nesterov accelerated gradient (e.g. BID35 , with the addition of root-mean-square normalization for the gradient.

The key difference between momentum and Nesterov accelerated gradient is where we evaluate the gradient: for momentum, we evaluate the gradient at µ w (t − 1) (i.e. at µ post (t − 1)), and for Nesterov accelerated gradient, we evaluate the gradient at a "predicted" location (i.e. at µ prior (t)), g Mom (t) = g (µ w (t − 1)) = g (µ post, w (t − 1)) (54a) g NAG (t) = g (µ w (t − 1) + (1 − η p )v(t − 1)) ≈ g (µ w (t − 1) + v(t − 1)) = g (µ prior, w (t)) (54b)As noted in Eq. (17), Bayesian filtering requires us to evaluate the gradient at µ prior , implying that we use updates based on NAG (Eq. 54b), rather than updates based on standard momentum (Eq. 54a).

One approach that works well in practice in natural-gradient methods BID1 BID19 BID10 BID32 BID31 is to use a Fisher-

@highlight

We formulated SGD as a Bayesian filtering problem, and show that this gives rise to RMSprop, Adam, AdamW, NAG and other features of state-of-the-art adaptive methods

@highlight

The paper analyzes stochastic gradient descent through Bayesian filtering as a framework for analyzing adaptive methods.

@highlight

The authors attempt to unify existing adaptive gradient methods under the Bayesian filtering framework with the dynamical prior