We introduce two approaches for conducting efficient Bayesian inference in stochastic simulators containing nested stochastic sub-procedures, i.e., internal procedures for which the density cannot be calculated directly such as rejection sampling loops.

The resulting class of simulators are used extensively throughout the sciences and can be interpreted as probabilistic generative models.

However, drawing inferences from them poses a substantial challenge due to the inability to evaluate even their unnormalised density, preventing the use of many standard inference procedures like Markov Chain Monte Carlo (MCMC).

To address this, we introduce inference algorithms based on a two-step approach that first approximates the conditional densities of the individual sub-procedures, before using these approximations to run MCMC methods on the full program.

Because the sub-procedures can be dealt with separately and are lower-dimensional than that of the overall problem, this two-step process allows them to be isolated and thus be tractably dealt with, without placing restrictions on the overall dimensionality of the problem.

We demonstrate the utility of our approach on a simple, artificially constructed simulator.

Stochastic simulators are used in a myriad of scientific and industrial settings, such as epidemiology (Patlolla et al., 2004) , physics (Heermann, 1990) , engineering (Hangos and Cameron, 2001 ) and climate modelling (Held, 2005) .

They can be complex and highdimensional, often incorporating domain-specific expertise accumulated over many years of research and development.

As shown by the probabilistic programming (Gordon et al., 2014; van de Meent et al., 2018; and approximate Bayesian computation (ABC) (Csilléry et al., 2010; Marin et al., 2012) literatures, these simulators can be interpreted as probabilistic generative models, implicitly defining a probability distribution over their internal variables and outputs.

As such, they form valid targets for drawing Bayesian inferences.

In particular, by constraining selected internal variables or outputs to take on specific values, we implicitly define a conditional distribution, or posterior, over the remaining variables.

This effectively allows us, amongst other things, to run the simulator in "reverse", fixing the outputs to some observed values and figuring out what parameter values might have led to them.

For example, given a simulator for visual scenes, we can run inference on the simulator with an observed image to predict what objects are present in the scene (Kulkarni et al., 2015) .

Though recent advances in probabilistic programming systems (PPSs, Tran et al. (2017) ; Bingham et al. (2019) ; ; Casado et al. (2017) ) have provided convenient mechanisms for encoding, reasoning about, and constructing inference algorithms for such simulators, performing the necessary inference is still often extremely challenging, particularly for complex or high-dimensional problems.

In this paper, we consider a scenario where this inference is particularly challenging to perform: when the simulator makes calls to nested stochastic sub-procedures (NSSPs).

These NSSPs can take several different forms, such as internal rejection sampling loops, separate inference procedures, external sub-simulators we have no control over, or even realworld experiments.

Their unifying common feature is that the density of their outputs cannot be evaluated up to an input-independent normalising constant in closed form.

This, in turn, means the normalised density of the overall simulator cannot be evaluated, preventing one from using most common inference methods, including almost all Markov chain Monte Carlo (MCMC) and variational methods.

Though some inference methods can still be applied in these scenarios, such as nested importance sampling (Rainforth, 2018) , these tend to scale very poorly in the dimensionality and often even have fundamentally slower convergence rates than standard Monte Carlo approaches (Rainforth et al., 2018) .

To address this issue, we introduce two new approaches for performing inference in such models.

Both are based around approximating the individual NSSPs.

The first approach directly approximates the conditional density of the NSSP outputs using an amortized inference artefact.

This then forms a surrogate density for the NSSP, which, once trained, is used to replace it.

While this first approach is generally applicable, our second approach focuses on the specific case where the unnormalized density of the NSSP can be evaluated in isolation (such as a nested probabilistic program or rejection sampling loop), but its normalizing constant depends on the NSSP inputs.

Here, we train a regressor to approximate the normalising constant of the NSSP as a function of its inputs.

Once learnt, this allows the NSSP to be collapsed into the outer program: the ratio of the known unnormalised density and the approximated normalizing constant can be directly used as a factor in the overall density.

Both approaches lead to an approximate version of the overall unnormalised density, which can then be used as a target for conventional inference methods like MCMC and variational inference.

Because these approximations can be calculated separately for each NSSP, this allows them to scale to higher dimensional overall simulators far more gracefully than existing approaches, opening the door to tractably running inference for more complex problems.

Furthermore, once trained, the approximations can be reused for different datasets and configurations of the outer simulator, thereby helping amortise the cost of running multiple different inferences for no extra cost.

The approaches themselves are also amenable to automation, making them suitable candidates for PPS inference engines.

We now introduce our two approaches for approximating NSSPs and show how these, in turn, produce efficient inference algorithms for the overall simulator.

Both our approaches involve the gradient-based learning of a neural-network-based amortised approximation for each NSSP that takes in the NSSP inputs and either returns and approximation of the density of the outputs (method 1) or the normalizing constant (method 2).

For any simulator or program, we can define the program density over valid program traces x 1:nx as (Rainforth, 2017, Section 4.3.2):

where n x is the length of the trace; each f a j (x j |φ j ) represents the density of the j th random draw, which is made at location a j and takes in parameters φ j ; and n y is a number of "observations", each of which factor the trace density by g b k (y k |ψ k ), where b k is the location of this observation statement, y k is the observed value, and ψ k are parameters of the factorization.

Here all terms-i.e., x j , n x , a j , φ j , n y , b k , y k , and ψ k -may be random variables, but each is deterministically calculable from the trace x 1:nx (see Rainforth (2017, Section 4.3.2)) A NSSP can now be formally defined as a f a j (x j |φ j ) term which cannot be directly evaluated exactly, but where for a given φ j either [Case A] we can draw samples from f a j (x j |φ j ) directly and/or [Case B] f a j (x j |φ j ) corresponds to the normalized density of a nested probabilistic program that we can draw approximate samples from by running an separate inference procedure.

Many simulators contain such sampling procedures (Di Pasquale et al., 2015; Gleisberg et al., 2009; Smith et al., 2006; Heermann, 1990; Rainforth, 2018; , and it is these simulators that we target with our inference schemes.

We can denote the unnormalized density for a program containing NSSPs as

where

is a representation of the "forward" or "prior" program which ignores all conditioning statements; S r = {a 1 , . . . , a n } represents the set of addresses that produce intractable densities; and we use P in a j (x j |φ j ) to distinguish the NSSPs from tractable sampling terms.

Both our methods are now based on replacing each of the P in a j (x j |φ j ) with an approximation, for which we only need to consider the prior program.

Once learned, these can then be used to construct a directly evaluable approximate target densityγ(x 1:nx ) by replacing each P in a j (x j |φ j ) in (3), then running an MCMC sampler onγ(x 1:nx ).

Our first method replaces each P in a j (x j |φ j ) by an approximate surrogate q in a j (x j |φ j ; η a j ):

where κ = {η a j ; a j ∈ S r } are the surrogate parameters.

As per existing amortized variational approaches (Kingma and Welling, 2014; Rezende and Mohamed, 2015; Le et al., 2016; Ritchie et al., 2016; Paige and Wood, 2016) , each q in a j (x j |φ j ; η a j ) is taken as a variational distribution parametrized by deep neural network with weights η a j and which takes φ j as its input.

Training of these networks is done by minimising the Kullback-Leibler (KL) divergence from P pr (x 1:nx ) to q(x 1:nx ; κ) (Paige and Wood, 2016)

This minimization can be done using stochastic gradient descent where the updates for NSSP r ∈ S r use the following gradient estimate (see Appendix A)

and the x n 1:nx can be shared such that the variational approximations for each r ∈ S r are made simultaneously.

Carrying out these updates requires us to draw samples from P pr .

If all of our NSSPs satisfy [Case A], this is not a problem as by assumption we can then draw samples from each P in a j (x j |φ j ) and, in turn, samples from P pr .

However, if our program contains NSSPs which only satisfy [Case B], this will require us to run a separate nested inference (Rainforth, 2018) to generate the required x n j from the corresponding φ j .

Though this may be potentially non-trivial, it is, crucially, far easier that running inference on the overall program: because P pr itself does not include any conditioning statements, generating these samples does not require inference to be run for the outer program.

As such, each nested inference problem constitutes its own isolated problem which is far simpler than the overall inference problem.

In other words, the role of sampling from P pr is only to generate example input-output pairs for each NSSP, with each surrogate than separately trained based on its local pairs.

If all of our NSSPs satisfy [Case B], this implies that each has a known unnormalised density on its internal variables and unknown input-dependent normalizing constant that causes a double-intractability.

If the functional form for all these normalizing constant where known, this would be sufficient to collapse all the NSSPs into the outer program and produce a directly evaluable density for the overall program.

Our second method thus looks to learn regressors to predict the normalizing constants and thereby facilitate this.

To formalize this, let us for now assume that the x j returned by each NSSP corresponds to its full set of internal random draws z , such that we can write

where γ in a j (z j 1:n j x |φ j ) can be evaluated directly (because it is itself an unnormalized probabilistic program density of the form (1)), but I in a j (φ j ) is an intractable normalization constant.

If we now introduce a set of regressors R r (φ j ; τ r ), ∀r ∈ S r (with parameters τ r ) to approximate each I in r (φ j ), we can approximate P pr as

We can extend this approach to the case where

by instead defining our reference measure in the space of X a := {x j } j∈1:nx|a j / ∈Sr ∪ {z j 1:n j x } j∈1:nx|a j / ∈Sr and using the pre-image of the prior program density: P pr (X a ).

We can then run inference in this pre-image space and rely on the law of the unconscious statistician to ensure the samples produces are from the desired posterior (see e.g. Rainforth (2017, Section 4.3.2)).

Learning the regressors R r (φ j ; τ r ) is done in an analogous manner to method one.

Namely we run the program forward to gather pairs {φ j ,Î r (φ j )} for each NSSP, whereÎ r (φ j ) is an unbiased approximation of I r (φ j ), and then use this as a training dataset for learning the regressor.

Specifically, for each NSSP we train a neural network regressor to minimize the expected squared error between R r (φ j ; τ r ) andÎ r (φ j ).

As shown in Appendix B, with a sufficiently expressive neural network, this scheme ensures R r (φ j ; τ r ) →

I r (φ j ) ∀φ j as the number of training pairs tends to infinity.

In this section, we use a 60-d nested Guassian example, details of which are given in Appendix C. The model has been contrived so that we can analytically calculate the posterior means and therefore validate against ground truth values.

Figure 2 demonstrates this for Method 1 (results for Method 2 are still being developed).

We see that accurate inference was achieved for all but two of the marginal distributions (these were caused by issues in the stability of the neural network training, which is currently being investigated).

Though still preliminary, these results are very promising in that they demonstrate that we are able to perform effective inference in far higher dimensions that can be realistically achieved by importance sampling based approaches, which are the current standard in the field.

For a given simulator, or program, we denote the proposal for the program as:

where κ = {η a j ; a j ∈ S r } are the variational parameters.

Using the information projection we construct the variational objective as follows:

Thus, we define the gradients to use for the stochastic gradient ascent for each subproblem r ∈ S r as:

where x n 1:nx iid ∼ P pr (x).

During training we extract samples from each forward run and train each NSSP separately.

To train our regressors, we use the L 2 -norm E R r (φ j ; τ r ) −Î in r (φ j ) 2 2 between our regressor R r (φ j ; τ r ) and our approximations of the marginalÎ in r (φ j ).

We then learn parameters τ r so that it minimises this objective, resulting in R r (φ j ; τ r ) = I in r (φ j ) in the limit of a large number of training samples if our neural network has sufficient capacity to exactly capture I in r (φ j ).

To see this note that

where the second term does not depend on τ r and the first is minimized when when R r (φ j ; τ r ) = I a j (φ j ).

Our objective is defined as:

where the expectation over the inputs φ j is defined by running P pr forward and, if necessary, randomly selecting between the inputs that are passed to NSSP r if it is called more than once (this can further be Rao-Blackwellized by averaging over all the inputs passed to the NSSP instead of choosing between them).

Thus, by running the simulator forward, collecting samples from the NSSPs generated from sampling the priors of each NSSP, we can make updates based on ∇ τ (R r (φ j ; τ ) −Î r (φ j )) 2 to minimise L r .

With this approach, we must be careful to avoid over and under-fitting.

Once trained, we can run inference on the approximate, unnormalised, target:

The approach outlined in Method 2 can be improved upon in the case where our nested sub-procedures are rejection samplers.

For rejection samplers, we always have I(φ) = E[I(A(z, φ) = 1)] where A(z, , φ) = 1 indicates an accepted sample and the expectation is with respect to running a single iteration of the rejection sampling loop.

The naive Monte Carlo estimate for I(φ), 1 N N n=1 I(A(z n , φ) = 1), is only unbiased, if N is independent of the z n .

Typically, one would like to instead run the rejection sampler in the standard manner, by which we generate samples by running the sampler until a sample is accepted, at which point we have generated N a samples, where N a is not independent of the z n , such that the naive estimate is now biased.

However, not doing this could, for example, return an estimatê (I)(φ) = 0 which could cause significant issues if not dealt with properly, while it may not be possible to generate both strictly positive and unbiased estimates for I(φ).

This conundrum can be circumvented by instead trying to directly estimate 1/I(φ) and use this as the basis for the regressor.

This is possible because rejection samplers have the property E[N a |φ] = 1/I(φ) as follows:

(1 − I(φ)) n = 1 I(φ) Therefore, we learn our regressor R to go from φ j to E[N a |φ], exploiting the fact that N a is an unbiased estimate of the latter, and subsequently use P in a j (x j |φ j ) ≈ γ in a j (x j |φ j )R a j (φ j ; τ a j )

to construct the approximate objective.

It is interesting to further note that

such that it should also be useful to use this result to develop pseudo-marginal samplers for such problems.

We take the model of a high-dimensional multivariate Gaussian with unkown mean and sample certain dimensions such that they rely on Gaussian NSSPs.

The purpose of such an example is to demonstrate the validity of our methodoly, as it is one o the few examples in which we analytically calculate the correct ground truth.

The model takes the following form: and we can sample x 1:d sequentially from a Markov process.

As the covariance matrix takes this structure we can use standard identities, as in Petersen et al., to analytically calculate the value of µ x , which is plotted in Figure 1 .

Histograms both the predicted and ground truth values are provided in Figure 2 for all 60 dimensions.

@highlight

We introduce two approaches for efficient and scalable inference in stochastic simulators for which the density cannot be evaluated directly due to, for example, rejection sampling loops.