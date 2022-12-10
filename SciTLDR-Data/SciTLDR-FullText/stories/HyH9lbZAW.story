Recent efforts on combining deep models with probabilistic graphical models are promising in providing flexible models that are also easy to interpret.

We propose a variational message-passing algorithm for variational inference in such models.

We make three contributions.

First, we propose structured inference networks that incorporate the structure of the graphical model in the inference network of variational auto-encoders (VAE).

Second, we establish conditions under which such inference networks enable fast amortized inference similar to VAE.

Finally, we derive a variational message passing algorithm to perform efficient natural-gradient inference while retaining the efficiency of the amortized inference.

By simultaneously enabling structured, amortized, and natural-gradient inference for deep structured models, our method simplifies and generalizes existing methods.

To analyze real-world data, machine learning relies on models that can extract useful patterns.

Deep Neural Networks (DNNs) are a popular choice for this purpose because they can learn flexible representations.

Another popular choice are probabilistic graphical models (PGMs) which can find interpretable structures in the data.

Recent work on combining these two types of models hopes to exploit their complimentary strengths and provide powerful models that are also easy to interpret BID10 BID14 BID0 BID3 .To apply such hybrid models to real-world problems, we need efficient algorithms that can extract useful structure from the data.

However, the two fields of deep learning and PGMs traditionally use different types of algorithms.

For deep learning, stochastic-gradient methods are the most popular choice, e.g., those based on back-propagation.

These algorithms are not only widely applicable, but can also employ amortized inference to enable fast inference at test time BID17 BID12 .

On the other hand, most popular algorithms for PGMs exploit the model's graphical conjugacy structure to gain computational efficiency, e.g., variational message passing (VMP) BID18 , expectation propagation BID16 , Kalman filtering BID4 BID5 , and more recently natural-gradient variational inference BID9 and stochastic variational inference BID8 .

In short, the two fields of deep learning and probabilistic modelling employ fundamentally different inferential strategies and a natural question is, whether we can design algorithms that combine their respective strengths.

There have been several attempts to design such methods in the recent years, e.g., BID14 ; BID3 ; BID0 ; BID10 ; BID2 .

Our work in this paper is inspired by the previous work of BID10 that aims to combine message-passing, natural-gradient, and amortized inference.

Our proposed method in this paper simplifies and generalizes the method of BID10 .To do so, we propose Structured Inference Networks (SIN) that incorporate the PGM structure in the standard inference networks used in variational auto-encoders (VAE) BID12 BID17 .

We derive conditions under which such inference networks can enable fast amortized inference similar to VAE.

By using a recent VMP method of BID11 , we The generative models are just like the decoder in VAE but they employ a structured prior, e.g., Fig. (a) has a mixture-model prior while Fig. (b) has a dynamical system prior.

SINs, just like the encoder in VAE, mimic the structure of the generative model by using parameters φ.

One main difference is that in SIN the arrows between y n and x n are reversed compared to the model, while rest of the arrows have the same direction.derive a variational message-passing algorithm whose messages automatically reduce to stochasticgradients for the deep components of the model, while perform natural-gradient updates for the PGM part.

Overall, our algorithm enables Structured, Amortized, and Natural-gradient (SAN) updates and therefore we call our algorithm the SAN algorithm.

We show that our algorithm give comparable performance to the method of BID10 while simplifying and generalizing it.

The code to reproduce our results is available at https://github.com/emtiyaz/vmp-for-svae/.

We consider the modelling of data vectors y n by using local latent vectors x n .

Following previous works BID10 BID0 BID14 , we model the output y n given x n using a neural network with parameters θ NN , and capture the correlations among data vectors y := {y 1 , y 2 , . . .

, y N } using a probabilistic graphical model (PGM) over the latent vectors x := {x 1 , x 2 , . . .

, x N }.

Specifically, we use the following joint distribution: DISPLAYFORM0 where θ NN and θ PGM are parameters of a DNN and PGM respectively, and θ := {θ NN , θ PGM }.This combination of probabilistic graphical model and neural network is referred to as structured variational auto-encoder (SVAE) by BID10 .

SVAE employs a structured prior p(x|θ PGM ) to extract useful structure from the data.

SVAE therefore differs from VAE BID12 where the prior distribution over x is simply a multivariate Gaussian distribution p(x) = N (x|0, I) with no special structure.

To illustrate this difference, we now give an example.

Example (Mixture-Model Prior) : Suppose we wish to group the outputs y n into K distinct clusters.

For such a task, the standard Gaussian prior used in VAE is not a useful prior.

We could instead use a mixture-model prior over x n , as suggested by BID10 , DISPLAYFORM1 where z n ∈ {1, 2, . . .

, K} is the mixture indicator for the n'th data example, and π k are mixing proportions that sum to 1 over k. Each mixture component can further be modelled, e.g., by using a Gaussian distribution p(x n |z n = k) := N (x n |µ k , Σ k ) giving us the Gaussian Mixture Model (GMM) prior with PGM hyperparameters DISPLAYFORM2 .

The graphical model of an SVAE with such priors is shown in FIG0 .

This type of structured-prior is useful for discovering clusters in the data, making them easier to interpret than VAE.Our main goal in this paper is to approximate the posterior distribution p(x, θ|y).

Specifically, similar to VAE, we would like to approximate the posterior of x by using an inference network.

In VAE, this is done by using a function parameterized by DNN, as shown below: DISPLAYFORM3 where the left hand side is the posterior distribution of x, and the first equality is obtained by using the distribution of the decoder in the Bayes' rule.

The right hand side is the distribution of the encoder where q is typically an exponential-family distribution whose natural-parameters are modelled by using a DNN f φ with parameters φ.

The same function f φ (·) is used for all n which reduces the number of variational parameters and enables sharing of statistical strengths across n. This leads to both faster training and faster testing BID17 .Unfortunately, for SVAE, such inference networks may give inaccurate predictions since they ignore the structure of the PGM prior p(x|θ PGM ).

For example, suppose y n is a time-series and we model x n using a dynamical system as depicted in FIG0 .

In this case, the inference network of FORMULA3 is not an accurate approximation since it ignores the time-series structure in x.

This might result in inaccurate predictions of distant future observations, e.g., prediction for an observation y 10 given the past data {y 1 , y 2 , y 3 } would be inaccurate because the inference network has no path connecting x 10 to x 1 , x 2 , or x 3 .

In general, whenever the prior structure is important in obtaining accurate predictions, we might want to incorporate it in the inference network.

A solution to this problem is to use an inference network with the same structure as the model but to replace all its edges by neural networks BID14 BID3 ).

This solution is reasonable when the PGM itself is complex, but might be too aggressive when the PGM is a simple model, e.g., when the prior in FIG0 is a linear dynamical system.

Using DNNs in such cases would dramatically increase the number of parameters which will lead to a possible deterioration in both speed and performance.

BID10 propose a method to incorporate the structure of the PGM part in the inference network.

For SVAE with conditionally-conjugate PGM priors, they aim to obtain a mean-field variational inference by optimizing the following standard variational lower bound 1 : DISPLAYFORM4 where q(x|λ x ) is a minimal exponential-family distribution with natural parameters λ x .

To incorporate an inference network, they need to restrict the parameter of q(x|λ x ) similar to the VAE encoder shown in (3), i.e., λ x must be defined using a DNN with parameter φ.

For this purpose, they use a two-stage iterative procedure.

In the first stage, they obtain λ * x by optimizing a surrogate lower bound where the decoder in (4) is replaced by the VAE encoder of (3) (highlighted in blue), DISPLAYFORM5 The optimal λ * x is a function of θ and φ and they denote it by λ * x (θ, φ).

In the second stage, they substitute λ * x into (4) and take a gradient step to optimize L(λ * x (θ, φ), θ) with respect to θ and φ.

This is iterated until convergence.

The first stage ensures that q(x|λ x ) is defined in terms of φ similar to VAE, while the second stage improves the lower bound while maintaining this restriction.

The advantage of this formulation is that when the factors q(x n |f φ (y n )) are chosen to be conjugate to p(x|θ PGM ), the first stage can be performed efficiently using VMP.

However, the overall method might be difficult to implement and tune.

This is because the procedure is equivalent to an implicitly-constrained optimization 2 that optimizes (4) with the constraint λ * x (θ, φ) = arg max λxL (λ x , θ, φ).

Such constrained problems are typically more difficult to solve than their unconstrained counterparts, especially when the constraints are nonconvex BID6 .

Theoretically, the convergence of such methods is difficult to guarantee when the constraints are violated.

In practice, this makes the implementation difficult because in every iteration the VMP updates need to run long enough to reach close to a local optimum of the surrogate lower bound.

Another disadvantage of the method of BID10 is that its efficiency could be ensured only under restrictive assumptions on the PGM prior.

For example, the method does not work for PGMs that contain non-conjugate factors because in that case VMP cannot be used to optimize the surrogate lower bound.

In addition, the method is not directly applicable when λ x is constrained and when p(x|θ PGM ) has additional latent variables (e.g., indicator variables z n in the mixture-model example).

In summary, the method of BID10 might be difficult to implement and tune, and also difficult to generalize to cases when PGM is complex.

In this paper, we propose an algorithm to simplify and generalize the algorithm of BID10 .

We propose structured inference networks (SIN) that incorporate the structure of the PGM part in the VAE inference network.

Even when the graphical model contains a non-conjugate factor, SIN can preserve some structure of the model.

We derive conditions under which SIN can enable efficient amortized inference by using stochastic gradients.

We discuss many examples to illustrate the design of SIN for many types of PGM structures.

Finally, we derive a VMP algorithm to perform natural-gradient variational inference on the PGM part while retaining the efficiency of the amortized inference on the DNN part.

We start with the design of inference networks that incorporate the PGM structure into the inference network of VAE.

We propose the following structured inference network (SIN) which consists of two types of factors, DISPLAYFORM0 The DNN factor here is similar to (3) while the PGM factor is an exponential-family distribution which has a similar graph structure as the PGM prior p(x|θ PGM ).

The role of the DNN term is to enable flexibility while the role of the PGM term is to incorporate the model's PGM structure into the inference network.

Both factors have their own parameters.

φ NN is the parameter of DNN and φ PGM is the natural parameter of the PGM factor.

The parameter set is denoted by φ := {φ NN , φ PGM }.How should we choose the two factors?

As we will show soon that, for fast amortized inference, these factors need to satisfy the following two conditions.

The first condition is that the normalizing constant 3 log Z(φ) is easy to evaluate and differentiate.

The second condition is that we can draw samples from SIN, i.e., x * (φ) ∼ q(x|y, φ) where we have denoted the sample by x * (φ) to show its dependence on φ.

An additional desirable, although not necessary, feature is to be able to compute the gradient of x * (φ) by using the reparameterization trick.

Now, we will show that given these two conditions we can easily perform amortized inference.

We show that when the above two conditions are met, a stochastic gradient of the lower bound can be computed in a similar way as in VAE.

For now, we assume that θ is a deterministic variable (we will relax this in the next section).

The variational lower bound in this case can be written as follows: DISPLAYFORM1 The first term above is identical to the lower bound of the standard VAE, while the rest of the terms are different (shown in blue).

The second term differs due to the PGM prior in the generative model.

In VAE, p(x|θ PGM ) is a standard normal, but here it is a structured PGM prior.

The last two terms arise due to the PGM term in SIN.

If we can compute the gradients of the last three terms and generate samples x * (φ) from SIN, we can perform amortized inference similar to VAE.

Fortunately, the second and third terms are usually easy for PGMs, therefore we only require the gradient of Z(φ) to be easy to compute.

This confirms the two conditions required for a fast amortized inference.

The resulting expressions for the stochastic gradients are shown below where we highlight in blue the additional gradient computations required on top of a VAE implementation (we also drop the explicit dependence of x * (φ) over φ for notational simplicity).

DISPLAYFORM2 DISPLAYFORM3 The gradients of Z(φ) and x * (φ) might be cheap or costly depending on the type of PGM.

For example, for LDS, these require a full inference through the model which costs O(N ) computation and is infeasible for large N .

However, for GMM, each x n can be independently sampled and therefore computations are independent of N .

In general, if the latent variables in PGM are highly correlated (e.g., Gaussian process prior), then Bayesian inference is not computationally efficient and gradients are difficult to compute.

In this paper, we do not consider such difficult cases and assume that Z(φ) and x * (φ) can be evaluated and differentiated cheaply.

We now give many examples of SIN that meet the two conditions required for a fast amortized inference.

When p(x|θ PGM ) is a conjugate exponential-family distribution, choosing the two factors is a very easy task.

In this case, we can let q(x|φ PGM ) = p(x|φ PGM ), i.e., the second factor is the same distribution as the PGM prior but with a different set of parameters φ PGM .

To illustrate this, we give an example below when the PGM prior is a linear dynamical system.

Example (SIN for Linear Dynamical System (LDS)) : When y n is a time series, we can model the latent x n using an LDS defined as p(x|θ) := N (x 0 |µ 0 , Σ 0 ) N n=1 N (x n |Ax n−1 , Q), where A is the transition matrix, Q is the process-noise covariance, and µ 0 and Σ 0 are the mean and covariance of the initial distribution.

Therefore, θ PGM := {A, Q, µ 0 , Σ 0 }.

In our inference network, we choose q(x|φ PGM ) = p(x|φ PGM ) as show below, where φ PGM := {Ā,Q,μ 0 ,Σ 0 } and, since our PGM is a Gaussian, we choose the DNN factor to be a Gaussian as well: DISPLAYFORM4 where m n := m φNN (y n ) and V n := V φNN (y n ) are mean and covariance parameterized by a DNN with parameter φ NN .

The generative model and SIN are shown in FIG0 , respectively.

The above SIN is a conjugate model where the marginal likelihood and distributions can be computed in O(N ) using the forward-backward algorithm, a.k.a.

Kalman smoother BID1 .

We can also compute the gradient of Z(φ) as shown in BID13 .When the PGM prior has additional latent variables, e.g., the GMM prior has cluster indicators z n , we might want to incorporate their structure in SIN.

This is illustrate in the example below.

Example (SIN for GMM prior): The prior shown in (2) has an additional set of latent variables z n .

To mimic this structure in SIN, we choose the PGM factor as shown below with parameters DISPLAYFORM5 , while keeping the DNN part to be a Gaussian distribution similar to the LDS case: DISPLAYFORM6 The model and SIN are shown in FIG0 and 1b, respectively.

Fortunately, due to conjugacy of the Gaussian and multinomial distributions, we can marginalize x n to get a closed-form expression for log Z(φ) := n log k N (m n |μ k , V n +Σ k )π k .

We can sample from SIN by first sampling from the marginal q(z n = k|y, φ) ∝ N m n |μ k , V n +Σ k π k .

Given z n , we can sample x n from the following conditional: DISPLAYFORM7 In all of the above examples, we are able to satisfy the two conditions even when we use the same structure as the model.

However, this may not always be possible for all conditionally-conjugate exponential family distributions.

However, we can still obtain samples from a tractable structured mean-field approximation using VMP.

We illustrate this for the switching state-space model in Appendix A. In such cases, a drawback of our method is that we need to run VMP long enough to get a sample, very similar to the method of BID10 .

However, our gradients are simpler to compute than theirs.

Their method requires gradients of λ * (θ, φ) which depends both on θ and φ (see Proposition 4.2 in BID10 ).

In our case, we require gradient of Z(φ) which is independent of θ and therefore is simpler to implement.

An advantage of our method over the method of BID10 is that our method can handle non-conjugate factors in the generative model.

When the PGM prior contains some non-conjugate factors, we might replace them by their closest conjugate approximations while making sure that the inference network captures the useful structure present in the posterior distribution.

We illustrate this on a Student's t mixture model.

To handle outliers in the data, we might want to use the Student's t-mixture component in the mixture model shown in (2), i.e., we set p(x n |z n = k) = T (x n |µ k , Σ k , γ k ) with mean µ k , scale matrix Σ k and degree of freedom γ k .

The Student's t-distribution is not conjugate to the multinomial distribution, therefore, if we use it as the PGM factor in SIN, we will not be able to satisfy both conditions easily.

Even though our model contains a t-distribution components, we can still use the SIN shown in (12) that uses a GMM factor.

We can therefore simplify inference by choosing an inference network which has a simpler form than the original model.

In theory, one can do this even when all factors are non-conjugate, however, the approximation error might be quite large in some cases for this approximation to be useful.

In our experiments, we tried this for non-linear dynamical system and found that capturing non-linearity was essential for dynamical systems that are extremely non-linear.

Previously, we assumed θ PGM to be deterministic.

In this section, we relax this condition and assume θ PGM to follow an exponential-family prior p(θ PGM |η PGM ) with natural parameter η PGM .

We derive a VMP algorithm to perform natural-gradient variational inference for θ PGM .

Our algorithm works even when the PGM part contains non-conjugate factors, and it does not affect the efficiency of the amortized inference on the DNN part.

We assume the following mean-field approximation: q(x, θ|y) := q(x|y, φ)q(θ PGM |λ PGM ) where the first term is equal to SIN introduced in the previous section, and the second term is an exponential-family distribution with natural parameter λ PGM .

For θ NN and φ, we will compute point estimates.

We build upon the method of BID11 which is a generalization of VMP and stochastic variational inference (SVI).

This method enables natural-gradient updates even when PGM contains non-conjugate factors.

This method performs natural-gradient variational inference by using a mirror-descent update with the Kullback-Leibler (KL) divergence.

To obtain natural-gradients with respect to the natural parameters of q, the mirror-descent needs to be performed in the mean parameter space.

We will now derive a VMP algorithm using this method.

We start by deriving the variational lower bound.

The variational lower bound corresponding to the mean-field approximation can be expressed in terms of L SIN derived in the previous section.

Compute q(x|y, φ) for SIN shown in (6) either by using an exact expression or using VMP.

DISPLAYFORM0

Sample x * ∼ q(x|y, φ), and compute ∇ φ Z and ∇ φ x * .

Update λ PGM using the natural-gradient step given in (16).

Update θ NN and φ using the gradients given in FORMULA8 - FORMULA0 with θ PGM ∼ q(θ PGM |λ PGM ).

7: until ConvergenceWe will use a mirror-descent update with the KL divergence for q(θ PGM |λ PGM ) because we want natural-gradient updates for it.

For the rest of the parameters, we will use the usual Euclidean distance.

We denote the mean parameter corresponding to λ PGM by µ PGM .

Since q is a minimal exponential family, there is a one-to-one map between the mean and natural parameters, therefore we can reparameterize q such that q(θ PGM |λ PGM ) = q(θ PGM |µ PGM ).

Denoting the values at iteration t with a superscript t and using Eq. 19 in BID11 with these divergences, we get: DISPLAYFORM0 DISPLAYFORM1 where β 1 to β 3 are scalars, , is an inner product, and ∇L t is the gradient at the value in iteration t.

As shown by BID11 , the maximization in FORMULA0 can be obtained in closed-form: DISPLAYFORM2 When the prior p(θ PGM |η PGM ) is conjugate to p(x|θ PGM ), the above step is equal to the SVI update of the global variables.

The gradient itself is equal to the message received by θ PGM in a VMP algorithm, which is also the natural gradient with respect to λ PGM .

When the prior is not conjugate, the gradient can be approximated either by using stochastic gradients or by using the reparameterization trick BID11 .

Therefore, this update enables natural-gradient update for PGMs that may contain both conjugate and non-conjugate factors.

The update of the rest of the parameters can be done by using a stochastic-gradient method.

This is because the solution of the update FORMULA0 is equal to a stochastic-gradient descent update (one can verify this by simplify taking the gradient and setting it to zero).

We can compute the stochasticgradients by using a Monte Carlo estimate with a sample θ * DISPLAYFORM3 where θ * := {θ * PGM , θ NN }.

As discussed in the previous section, these gradients can be computed similar to VAE-like by using the gradients given in (9)-(10).

Therefore, for the DNN part we can perform amortized inference, and use a natural-gradient update for the PGM part using VMP.The final algorithm is outlined in Algorithm 1.

Since our algorithm enables Structured, Amortized, and Natural-gradient (SAN) updates, we call it the SAN algorithm.

Our updates conveniently separate the PGM and DNN computations.

Step 3-6 operate on the PGM part, for which we can use existing implementation for the PGM.Step 7 operates on the DNN part, for which we can reuse VAE implementation.

Our algorithm not only generalizes previous works, but also simplifies the implementation by enabling the reuse of the existing software.

The main goal of our experiments is to show that our SAN algorithm gives similar results to the method of BID10 .

For this reason, we apply our algorithm to the two examples considered in BID10 , namely the latent GMM and latent LDS (see FIG0 ).

In this section we discuss results for latent GMM.

An additional result for LDS is included in Appendix C. Our results show that, similar to the method of BID10 our algorithm can learn complex Even with 70% outliers, SAN-TMM performs better than SAN-GMM with 10% outliers.

DISPLAYFORM0 Figure 3: Top row is for the Pinwheel dataset, while the bottom row is for the Auto dataset.

Point clouds in the background of each plot show the samples generated from the learned generative model, where each mixture component is shown with a different color and the color intensities are proportional to the probability of the mixture component.

The points in the foreground show data samples which are colored according to the true labels.

We use K = 10 mixture components to train all models.

For the Auto dataset, we show only the first two principle components.representations with interpretable structures.

The advantage of our method is that it is simpler and more general than the method of BID10 .We compare to three baseline methods.

The first method is the variational expectation-maximization (EM) algorithm applied to the standard Gaussian mixture model.

We refer to this method as 'GMM'.

This method is a clustering method but does not use a DNN to do so.

The second method is the VAE approach of BID12 , which we refer to as 'VAE'.

This method uses a DNN but does not cluster the outputs or latent variables.

The third method is the SVAE approach of BID10 applied to latent GMM shown in FIG0 .

This method uses both a DNN and a mixture model to cluster the latent variables.

We refer to this as 'SVAE'.

We compare these methods to our SAN algorithm applied to latent GMM model.

We refer to our method as 'SAN'.

All methods employ a Normal-Wishart prior over the GMM hyperparameters (see BID1 for details).We use two datasets.

The first dataset is the synthetic two-dimensional Pinwheel dataset (N = 5000 and D = 2) used in BID10 .

The second dataset is the Auto dataset (N = 392 and D = 6, available in the UCI repository) which contains information about cars.

The dataset also contains a five-class label which indicates the number of cylinders in a car.

We use these labels to validate our results.

For both datasets we use 70% data for training and the rest for testing.

For all methods, we tune the step-sizes, the number of mixture components, and the latent dimensionality on a validation set.

We train the GMM baseline using a batch method, and, for VAE and SVAE, we use minibatches of size 64.

DNNs in all models consist of two layers with 50 hidden units and an output layer of dimensionality 6 and 2 for the Auto and Pinwheel datasets, respectively.

FIG1 and 2b compare the performances during training.

In FIG1 , we compare to SVAE and GMM, where we see that SAN converges faster than SVAE.

As expected, both SVAE and SAN achieve similar performance upon convergence and perform better than GMM.

In FIG1 , we compare to VAE and GMM, and observe similar trends.

The performance of GMM is represented as a constant because it converges after a few iterations already.

We found that the implementation provided by BID10 does not perform well on the Auto dataset which is why we have not included it in the comparison.

We also compared the test log-likelihoods and imputation error which show very similar trends.

We omit these results due to space constraints.

In the background of each plot in Figure 3 , we show samples generated from the generative model.

In the foreground, we show the data with the true labels.

These labels were not used during training.

The plots (a)- FORMULA29 show results for the Pinwheel dataset, while plots FORMULA29 - FORMULA29 shows results for the Auto dataset.

For the Auto dataset, each label corresponds to the number of cylinders present in a car.

We observe that SAN can learn meaningful clusters of the outputs.

On the other hand, VAE does not have any mechanisms to cluster and, even though the generated samples match the data distribution, the results are difficult to interpret.

Finally, as expected, both SAN and VAE learn flexible patterns while GMM fails to do so.

Therefore, SAN enables flexible models that are also easy to interpret.

An advantage of our method over the method of BID10 is that our method applies even when PGM contains non-conjugate factors.

Now, we discuss a result for such a case.

We consider the SIN for latent Student's t-mixture model (TMM) discussed in Section 3.

The generative model contains the student's t-distribution as a non-conjugate factor, but our SIN replaces it with a Gaussian factor.

When the data contains outliers, we expect the SIN for latent TMM to perform better than the SIN for latent GMM.

To show this, we add artificial outliers to the Pinwheel dataset using a Gaussian distribution with a large variance.

We fix the degree of freedom for the Student's t-distribution to 5.

We test on four different levels of noise and report the test MSE averaged over three runs for each level.

FIG1 shows a comparison of GMM, SAN on latent GMM, and SAN on latent TMM where we see that, as the noise level is increased, latent TMM's performance degrades slower than the other methods (note that the y-axis is in log-scale).

Even with 70% of outliers, the latent TMM still performs better than the latent GMM with only 10% of outliers.

This experiment illustrates that a conjugate SIN can be used for inference on a model with a non-conjugate factor.

We propose an algorithm to simplify and generalize the algorithm of BID10 for models that contain both deep networks and graphical models.

Our proposed VMP algorithm enables structured, amortized, and natural-gradient updates given that the structured inference networks satisfy two conditions.

The two conditions derived in this paper generally hold for PGMs that do not force dense correlations in the latent variables x. However, it is not clear how to extend our method to models where this is the case, e.g., Gaussian process models.

It is possible to use ideas from sparse Gaussian process models and we will investigate this in the future.

An additional issue is that our results are limited to small scale data.

We found that it is non-trivial to implement a message-passing framework that goes well with the deep learning framework.

We are going to pursue this direction in the future and investigate good platforms to integrate the capabilities of these two different flavors of algorithms.

In SLDS, we introduce discrete variable z n ∈ {1, 2, . . .

, K} that are sampled using a Markov chain: p(z n = i|z n−1 = j) = π ij such that π ij sum to 1 over all i given j. The transition for LDS is defined conditioned on z n : p(x n |x n−1 , z n = i, θ PGM ) := N (x n |A i x n−1 , Q i ) where A i and Q i are parameters for the i'th indicator.

These two dynamics put together define the SLDS prior p(x, z|θ PGM ).

We can use the following SIN which uses the SLDS prior as the PGM factor but with parameters φ PGM instead of θ PGM .

The expression for q(x, z|y, φ) is shown below: DISPLAYFORM0 Even though the above model is a conditionally-conjugate model, the partition function is not tractable and sampling is also not possible.

However, we can use a structured mean-field approximation.

First, we can combine the DNN factor with the Gaussian observation of SLDS factor and then use a mean-field approximation q(x, z|y, φ) ≈ q(x|λ x )q(z|λ z ), e.g., using the method of BID5 .

This will give us a structured approximation where the edges between y n and x n and z n and z n−1 are maintained but x n and z n independent of each other.

In this section we give detailed derivations for the SIN shown in (12).

We derive the normalizing constant Z(φ) and show how to generate samples from SIN.We start by a simple rearrangement of SIN defined in (12): DISPLAYFORM0 DISPLAYFORM1 where the first step follows from the definition (12), the second step follows by taking the sum over k outside, and the third step is obtained by defining each component as a joint distribution over x n and the indicator variable z n .We will express this joint distribution as a multiplication of the marginal of z n and conditional of x n given z n .

We will see that this will give us the expression for the normalizing constant, as well as a way to sample from SIN.We can simplify the joint distribution further as shown below.

The first step follows from the definition.

The second step is obtained by swapping m n and x n in the first term.

The third step is obtained by completing the squares and expressing the first term as a distribution over x n (the second and third terms are independent of x n ).q(x n , z n = k|y n , φ) ∝ N (x n |m n , V n )N (x n |μ k ,Σ k )π k (20) DISPLAYFORM2 where Σ Using the above we get the marginal of z n and conditional of x n given z n : DISPLAYFORM3 q(x n |z n = k, y n , φ) := N (x n | µ n , Σ n )The normalizing constant of the marginal of z n is obtained by simply summing over all k: DISPLAYFORM4 and since q(x n |z n = k, y n , φ) is already a normalized distribution, we can write the final expression for the SIN as follows: DISPLAYFORM5 q(x n |z n = k, y n , φ)q(z n = k|y n , φ)where components are defined in FORMULA1 , FORMULA1 , and (25).

The normalizing constant is available in closed-form and we can sample z n first and then generate x n .

This completes the derivation.

In this experiment, we apply our SAN algorithm to the latent LDS discussed in Section 3.

For comparison, we compare our method, Structured Variational Auto-Encoder (SVAE) BID10 , and LDS on the Dot dataset used in BID10 .

Our results show that our method achieves comparable performance to SVAE.

For LDS, we perform batch learning for all model parameters using the EM algorithm.

For SVAE and SAN, we perform mini-batch updates for all model parameters.

We use the same neutral network architecture as in BID10 , which contains two hidden layers with tanh activation function.

We repeat our experiments 10 times and measure model performance in terms of the following mean absolute error for τ -steps ahead prediction.

The error measures the absolute difference between the ground truth and the generative outputs by averaging across generated results.

27) where N is the number of testing time series with T time steps, d is the dimensionality of observation y, and observation y * t+τ,n denotes the ground-truth at time step t + τ .

From FIG4 , we can observe that our method performs as good as SVAE and outperforms LDS.

Our method is slightly robust than SVAE.

In FIG5 , there are generated images obtained from all methods.

From FIG5 , we also see that our method performs as good as SAVE and is able to recover the ground-truth observation.

@highlight

We propose a variational message-passing algorithm for models that contain both the deep model and probabilistic graphical model.