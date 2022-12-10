Universal probabilistic programming systems (PPSs) provide a powerful framework for specifying rich and complex probabilistic models.

However, this expressiveness comes at the cost of substantially complicating the process of drawing inferences from the model.

In particular, inference can become challenging when the support of the model varies between executions.

Though general-purpose inference engines have been designed to operate in such settings, they are typically inefficient, often relying on proposing from the prior to make transitions.

To address this, we introduce a new inference framework: Divide, Conquer, and Combine (DCC).

DCC divides the program into separate straight-line sub-programs, each of which has a fixed support allowing more powerful inference algorithms to be run locally, before recombining their outputs in a principled fashion.

We show how DCC can be implemented as an automated and general-purpose PPS inference engine, and empirically confirm that it can provide substantial performance improvements over previous approaches.

Universal PPSs, such as Church (Goodman et al., 2008) , Venture (Mansinghka et al., 2014) , Anglican (Wood et al., 2014) and Pyro (Bingham et al., 2018) , are set up to try and support the widest possible range of models a user might wish to write.

Though this means that such systems can be used to write models which would be otherwise difficult to encode, this expressiveness comes at the cost of significantly complicating the automation of inference.

In particular, models may contain variables with mixed types or have varying, or even unbounded, dimensionalities; characteristics which cause significant challenges at the inference stage.

In this paper, we aim to address one of the most challenging of these complicating factors: variables whose very existence is stochastic, often, though not always, leading to the overall dimensionality of the model varying between realizations.

Some very basic inference algorithms, such as importance sampling from the prior, are able to deal with this problem naturally, but they are catastrophically inefficient for all but the most simple models.

Sequential Monte Carlo (Wood et al., 2014) and variational (Paige, 2016) approaches can sometimes also be applied, but only offer improvements for models with particular exploitable structures.

MCMC approaches, on the other hand, are difficult to apply due to the need to construct proposals able to switch between the different variable configurations, something which is difficult to achieve even in a problem specific manner, let alone automate for generic problems.

Moreover, ensuring these proposals remain efficient can be almost impossible, as different configurations might not have natural similarities or "neighboring regions"; the problem is analogous to running MCMC on a highly multi-modal distribution without any knowledge of where the different modes are.

In short, there are a wide range of models for which no effective PPS-suitable inference methods currently exist.

More discussion can be seen in Appendix B.

To this end, we introduce a new framework-Divide, Conquer, and Combine (DCC)-for performing inference in such models.

DCC works by dividing the program into separate straight-line sub-programs with fixed support, conquering these separate sub-problems using an inference strategy that exploits the fixed support to remain efficient, and then combining the resulting sub-estimators to an overall approximation of the posterior.

By splitting the original program up into its separate configurations, we effectively transfer this transitioning problem to one of estimating the marginal likelihood for the different models, something which is typically much easier to achieve.

Furthermore, this approach also allows us to introduce meta-strategies for allocating resources between sub-problems, thereby explicitly controlling the exploration-exploitation trade-off in a manner akin to Rainforth et al. (2018) ; Lu et al. (2018) .

To demonstrate its potential utility, we implement a specific realization of our DCC framework as an automated and general-purpose inference engine in the PPS Anglican (Wood et al., 2014) , finding that it is able to achieve substantial performance improvements and tackle more challenging models than existing approaches.

To aid exposition and formalize these programs, we will focus on the particular universal PPS Anglican (Wood et al., 2014) , but note that our ideas are applicable to any universal PPS for which the program's support is not necessarily fixed.

The density of an Anglican program is derived by executing it in a forward manner, drawing from sample statements when encountered, and keeping track of density components originating from both the sample and observe terms.

Specifically, let {x i } nx i=1 = x 1 , . . .

, x nx represent the random variables generated from the encountered sample statements, where the i-th encountered sample statement has a lexical program address a i .

More formal definition of the density is provided in Appendix A.1.

For clarity, we refer to the sequence a 1:nx as the path of a trace and x 1:nx as the draws.

A program with stochastic support means that the path a 1:nx of the program varies between different realizations: a different value for the path corresponds to a different configuration of variables being sampled.

Unlike most existing inference approaches which directly target the full program density, DCC breaks the problem into individual sub-problems with fixed support and tackles them separately.

Specifically, it divides the overall program into separate straight-line sub-programs according to their execution paths, conquers each sub-program by running inference locally, and combines the results together in a principled manner.

Divide The first step of DCC is to divide the given probabilistic program into its constituent straight-line programs (SLPs), where each SLP is a partition of the overall program

(let [z2 (sample (normal 5 2)) 8 z3 (sample (normal z2 2))] 9 (observe (normal z3 2) y1)) 10

[z0 z2 z3])))

Figure 1: A branching model written in Anglican (left) and its execution trace (right).

corresponding to a particular sequence of sample addresses encountered during execution, i.e. a particular path a k

.

Each SLP has a fixed support as the set of variables x k 1:n x,k it draws are fixed by the path, i.e. the program always draws from the same set of sample statements in the same fixed order.

Using the shorthand

, the set of of all possible execution paths is now given by {A k } K k=1 , where K must be countable (but may not be finite) and k indexes the individual SLPs (the ordering of which is inconsequential).

For the example in Figure 1 , this set consists of two paths

where we use #l j to denote the lexical address of the sample statement at the j th line.

Dividing a program into its constituent SLPs implicitly partitions the overall target density into disjoint regions, with each part defining a sub-model on the corresponding sub-space.

The density π k (x) of the SLP A k is defined with respect to the variables {x i } n x,k i=1 that are paired with the addresses {a i } n x,k i=1 of A k , which we only have access to the unnormalized version γ k (x).

We use X k to denote the corresponding sub-space of x 1:n x,k and note that the union of X k for all k is the entire latent space defined by the overall program, X = K k=1 X k .

Unlike previously, n x,k and A k are now, critically, deterministic variables so that the support of the sub-model is fixed.

The formal definition of the density for a SLP is given in Appendix A.2.

Following our example in Figure 1 , SLP A 1 corresponds to the sub-space X 1 = {[x 1 , x 2 ] ∈ R 2 | x 1 < 0} and has the density γ 1 (x) = N (x 1 ; 0, 2)N (x 2 ; −5, 2)N (y 1 ; x 2 , 2)I[x 1 < 0], while A 2 corresponds to the sub-space X 2 = {[x 1 , x 2 , x 3 ] ∈ R 3 | x 1 ≥ 0} and has density γ 2 (x) = N (x 1 ; 0, 2)N (x 2 ; 5, 2)N (x 3 ; x 2 , 2)N (y 1 ; x 3 , 2)I[x 1 ≥ 0].

More details are in Appendix C.2.

Conquer Given the set of SLPs produced by the divide step, we now carry out the required local inference for each.

This forms our conquer step and its aim is to produce a set of estimates for the individual SLP densities π k (x) and corresponding marginal likelihoods Z k .

As each SLP has a fixed support, this can be achieved with conventional inference approaches, with a large variety of methods potentially suitable.

Note that π k (x) and Z k need not be estimated using the same approach, e.g. we may use an MCMC scheme to estimate π k (x) and then introduce a separate estimator for Z k .

In short, we will propose the use of a combination of Metropolis-with-Gibbs (MwG) and the parallel interacting Markov adaptive importance sampling (PI-MAIS) algorithm of Martino et al. (2017) for performing the local inference with further details in Appendix C.1.

An important component in carrying out this conquer step effectively, is to note that it is not usually necessary to obtain equally high fidelity estimates for each SLP.

Specifically, SLPs with small marginal likelihoods Z k only make a small contribution to the overall density and thus do not require as accurate estimation as SLPs with large Z k s. As such, it will typically be beneficial to carry out resource allocation as part of the conquer step, that is, to generate our estimates in an online manner where at each iteration we use information from previous samples to decide the best SLP(s) to update our estimates for.

Further details on this, along with our suggested approach for the local inference itself, are given in and C.3.

Combine The last stage of DCC is to combine the local estimates from the individual SLPs to an overall estimate of the conditional distribution for the original program.

For this, we can simply note that, because the supports of the individual SLPs are disjoint and their union is the complete program, we have γ(x) = K k=1 γ k (x) and Z = K k=1 Z k , such that the unnormalized density and marginal likelihoods are both additive.

We then have

whereπ k (x) andẐ k are the SLP estimates generated during the conquer step.

When using an MCMC sampler for π k (x),π k (x) will take the form of an empirical measure comprising of a set of samples, i.e.π k (x) =

If we instead use an importance sampling or particle filtering based approach, our empirical measure will instead compose of weighted samples.

We note that in this case, theẐ k term in the numerator of (1) will cancel with any potential self-normalization term used inπ k (x), such that we can instead think of using the estimate π(x) ≈ (

Specific strategies for implementing each component are introduced in Appendix C.

The first model is a GMM where the number of the mixtures as well as the mean of each mixture are unknown.

We first examine the convergence of the overall log marginal likelihood Z, and present the median (solid line) and 25% − 75% quantiles (shaded area) of the squared error of the estimates in Figure 2 (top) among three methods.

As RMH cannot be used directly here, we instead draw importance samples centered around the RMH chain in a manner akin to PI-MAIS (Martino et al., 2017) .

It shows that DCC outperforms both baselines by many orders of magnitude.

To further investigate, we look into the posterior distribution of K and compare the estimates ofp(K = 5 | y 1:Ny ) in Figure 2 the most accurately and consistently whereas IS occasionally gives reasonable estimates and RMH has a tendency to get stuck in one sub-model.

More details are given in Appendix D.1.

The second model is about function induction generated by a Probabilistic Context Free Grammar (PCFG) (Manning et al., 1999) .

We specify the structure of a candidate function using a PCFG and distribution over the function parameters, and estimate the posterior of both for given data.

Our PCFG model consists of four production rules: Conditioned on some training data, we want to infer the posterior distribution of the function structure as well the underlying parameters, which can be used to do prediction given the test data.

We report the mean and one standard derivation of the test log marginal likelihood (LML) estimates (the higher the better) in Table 1 and DCC outperforms the baselines both in terms of predictive accuracy and stability.

A more qualitatively comparison of the posterior distribution are provided in Figure 3 .

DCC samples capture the periodicity of the training data and in general interpolates them well, while remaining uncertain in the regions of no data.

Though RMH does find some good functions, it becomes stuck in a particular mode and doe not fully capture the uncertainty in the model, leading to poor predictive performance.

See more results in Appendix D.2.

In this paper, we have proposed Divide, Conquer and Combine (DCC), a new inference strategy for probabilistic programs with stochastic support.

We have shown that by breaking down the overall inference problem into a number of separate inferences of subprograms of fixed support, the DCC framework can provide substantial performance improvements over existing approaches which directly target the full program.

To realize this potential, we have shown how to implement a particular instance of DCC as an automated engine in the PPS Anglican, and demonstrated its effectiveness through two example problems.

Anglican inherits its general syntax from Clojure, extending this with two special forms: sample and observe, between which the distribution of the program is defined.

sample statements are used to draw random variables from provided probability distributions, while observe statements are used to condition on data.

Informally, they can be respectively thought of as prior and likelihood terms.

The density of an Anglican program is derived by executing it in a forward manner, drawing from sample statements when encountered, and keeping track of density components originating from both the sample and observe terms.

Specifically, let {x i } nx i=1 = x 1 , . . .

, x nx represent the random variables generated from the encountered sample statements, where the i-th encountered sample statement has a lexical program address a i , an input η i , and a density f a i (x i |η i ).

Analogously, let {y j } ny j=1 = y 1 , . . .

, y ny represent the observed values of the n y encountered observe statements, which have lexical addresses b j and corresponding densities g b j (y j |φ j ), where φ j is analogous to η i .

The program density is now given by π(x) = γ(x)/Z where

and the associated reference measure is implicitly defined through the encountered sample statements.

Note here that everything (i.e. n x , n y , x 1:nx , y 1:ny , a 1:nx , b 1:ny , η 1:nx , and φ 1:ny ) is a random variable, but each is deterministically calculable given x 1:nx .

See Rainforth (2017, §4.3.2) for a more detailed introduction.

We denote an execution trace (i.e. realization) of an Anglican program by the sequence of the addresses of sample statements and the corresponding variables, namely [a i ,

For clarity, we refer to the sequence a 1:nx as the path of a trace and x 1:nx as the draws.

A program with stochastic support can now be more formally defined as one for which the path a 1:nx varies between different realizations: a different value for the path corresponds to a different configuration of variables being sampled.

Dividing a program into its constituent SLPs implicitly partitions the overall target density into disjoint regions, with each part defining a sub-model on the corresponding sub-space.

The unnormalized density γ k (x) of the straight-line program A k is defined with respect to the variables {x i } n x,k i=1 that are paired with the addresses {a i } n x,k i=1 of A k .

We use X k to denote the corresponding sub-space of x 1:n x,k .

Note that the union of X k for all k is the entire latent space defined by the overall program, X = K k=1 X k .

Analogously to (2), we now have that the density of SLP k is π k (x) = γ k (x)/Z k where

Unlike for (2), n x,k and A k are now, critically, deterministic variables so that the support of the problem is fixed.

Though b j and n y,k may still be stochastic, these do not effect the reference measure of the program (see Rainforth (2017, §4.4.3)) and so this does not cause a problem when trying to perform MCMC sampling.

Designing inference algorithms for models with stochastic support is typically very challenging.

Some basic inference schemes, such as importance sampling from the prior, can be directly applied, but their performance deteriorates rapidly as the dimension of the model increases.

Particle based inference methods such as Sequential Monte Carlo (SMC) (Wood et al., 2014; Doucet et al., 2001 ) can offer improvements for models with natural sequential structure, but similarly rapidly succumb to the curse of dimensionality in the majority of cases.

Variational approaches, on the other hand, are typically ill-suited to this setting: though some strategies have been proposed in Paige (2016), they require substantial approximations to be made and are again only applicable to very simple problems due to difficulties with gradient estimation.

Ulam, 1949) have the potential to tackle more difficult problems.

In particular, reversible jump Markov chain Monte Carlo (RJMCMC) (Green, 1995 (Green, , 2003 methods allow one to perform MCMC on problems with stochastic support by introducing proposals capable of transitioning between configurations.

However, their application is fundamentally challenging due to the difficulty in designing proposals which can transition efficiently.

Namely, proposing changes in the variable configuration introduces new variables that are not present in the current sample.

Further, the posterior on the other variables may shift substantially.

Consequently, one loses a notion of locality when switching configurations; having a sample in a high density region of one configuration typically provides little information about which regions have a high density for another configuration.

In turn, this means that it is extremely difficult to design proposals which both efficiently move between configurations and maintain a high acceptance rate; once in a high density region of one configuration, it becomes extremely difficult to switch to another configuration.

This is then compounded by the fact that RJMCMC only estimates the relative mass of each configuration through the relative frequency of transitions, giving a very slow mixing rate for the overall sampler.

The difficulty in applying RJMCMC is exacerbated in universal PPSs due to the desire to construct proposals in an automated fashion.

Thus, though RJMCMC has recently been applied in the PPS context by Roberts et al. (2019) , they rely on manual specification of the proposal by the user, thereby losing most of the automation that forms a core part of the motivation for PPSs in the first place.

Moreover, for many programs, it will be impractically difficult to even hand-design such a proposal.

One MCMC method that can be fully automated for PPSs is the Lightweight Metropolis Hastings algorithm (LMH) of Wingate et al. (2011) and its extensions (Ritchie et al., 2016; Tolpin et al., 2015) , for which implementations are provides in a number of systems such as Venture (Mansinghka et al., 2014) , WebPPL (Goodman and Stuhlmüller, 2014) , and Anglican (Wood et al., 2014) .

LMH is based around a Metropolis-within-Gibbs (MwG) approach (Brooks et al., 2011) whereby one first samples a variable in the execution trace, k ∈ 1 : n x , uniformly at random and then proposes a MwG transition to this sample, x k → x k .

Unlike in a standard MwG scheme, one must further now check if this transition influences the downstream control flow of the program: we must check that the transition does not cause the downstream path to change, i.e. that we have a k+1:n x = a k+1:nx .

When the path remains the same, we can reuse the downstream draws x k+1:nx and, in turn, a standard MwG accept-reject step.

However, when the path changes, the downstream draws no longer produce a valid execution trace.

To account for this, the remainder of the trace is instead redrawn afresh by simulating from the prior, such that the proposed trace is instead [{a 1:k , a k+1:n x }, {x 1:k−1 , x k , x k+1:n x }], where [a k+1:n x , x k+1:n x ] is the new partial execution trace generating by this redrawing.

This new sample is now accepted or rejected in the standard manner, except for an additional n x /n x term in the acceptance ratio.

Though widely applicable, LMH relies on proposing from the prior whenever the configuration changes for the downstream variables.

This inevitably forms a highly inefficient proposal (akin to importance sampling from the prior), such that LMH typically performs very poorly for programs with stochastic support, particularly in high dimensions.

We now outline a particular realization of our DCC framework that we have implemented for Anglican, which can be used to perform inference in an automated fashion for any input program of Anglican.

For this, we suggest particular strategies for the individual components left unspecified in the last section, emphasizing that these are not the only possible choices.

Specifically, we will propose the use of a combination of Metropolis-withGibbs (MwG) and the parallel interacting Markov adaptive importance sampling (PI-MAIS) algorithm of Martino et al. (2017) for performing the local inference, a dynamic model discovery approach for establishing the SLPs, and a resource allocation approach based on the exploration-exploration strategy introduced in Rainforth et al. (2018) .

Recall that the goal for the local inference is to estimate the local target density π k (x) (where we only have access to γ k (x)), and the local marginal likelihood Z k .

Straightforward choices for this include (self-normalized) importance sampling and SMC as both return a marginal likelihood estimateẐ k .

However, knowing good proposals for these a priori is challenging and, as we discussed in §B, naïve choices like sampling from the prior are unlikely to perform well.

Thankfully, each SLP has a fixed support, which means many of complications that make inference challenging for universal PPSs no longer apply.

In particular, we can use conventional MCMC samplers-such as MH, HMC, or MwG-to approximate π k (x).

Due to a combination of restrictions from our underlying PPS and the fact that individual variable types may be unknown or not even fixed, we have elected to use MwG in our implementation, but note that more powerful inference approaches like HMC may be preferable when they can be safely applied.

To encourage sample diversity and assist in estimating Z k (see below), we further run N independent MwG samplers for each SLP.

As MCMC samplers do not directly provide an estimate for the marginal likelihood, we must introduce a further estimator for Z k .

For this, we use the PI-MAIS approach of Martino et al. (2017) .

Though ostensibly an adaptive importance sampling algorithm, PI-MAIS is based around using a set of N proposals each centered on the outputs of an MCMC chain.

We can thus also think of it as a method for generating marginal likelihood estimates from a set of MCMC chains, which is what we require.

To be more precise, given a series of samples,x k,1:T,1:N , from N MwG chains run for T iterations each on the SLP A k , for each iteration of the chain PI-MAIS introduces a mixture proposal distribution by using the combination of separate proposals (e.g. a Gaussian) centered on each of these chains:

This can then be used to produce an importance sampling estimate for the target, with Rao-Blackwellization typically applied across the mixture components, such that one draws M samples separately from each q k,t,n , rather than N M samples from q k,t .

By proxy, this also produces a marginal likelihood estimateẐ k , which is equal to the empirical average of the importance weights, where this average is taken of N , T , and M .

An interesting point of note is that one can use either the originally MCMC samples, or the importance samples generated by the PI-MAIS for the estimateπ k (x).

The relative meric of these approaches depends on the exact problem (we will use the latter in our experiments).

For problems where the PI-MAIS forms an efficient adaptive importance sampler, the estimate it produces will typically be preferable.

However, in some cases, particularly high-dimensional problems, this sampler may struggle, so that it is more effective to take the original MCMC samples.

Though it might seem that we are doomed to fail anyway in such situations, as the struggling of the PI-MAIS estimator is likely to indicate our Z k estimates are poor, this is certainly not always the case.

In particular, for many problems, one SLP will dominate, i.e. Z k * Z k =k * for some k * .

Here we do not necessarily need an accurate estimates of the Z k to achieve an overall good approximation of the posterior, we need only identify the dominant Z k .

To divide a given model into its constituent sub-models expressed by SLPs, we need a mechanism for discovering these sub-models automatically.

One possible approach (Chaganty et al., 2013; Nori et al., 2014) would be to analyze the source code of the program defining the model using a static analysis, thereby extracting the set of possible execution paths of the program at compilation time.

However, this is a difficult, and potentially impossible, feat to achieve for all possible programs in a universal PPS.

In particular, it fails to deal with cases where the number of the sub-models is countably infinite.

Because of these issues, we take an alternative approach based on discovering models dynamically at run time.

Not only does this circumvent the need for a complex static program analysis, in settings where the number of potential models is too large to tractably enumerate, it further provides a natural approach to ensuring we only investigate models with a high potential to make a significant contribution to the overall density.

Our approach starts by executing the program forward for T 0 iterations to generate sample execution traces.

This corresponds to drawing samples from the prior of the model.

The paths traversed by these sampled traces are recorded, and our set of SLPs is initialized as that of these recorded paths.

At subsequent iterations, after each local inference stage, we then perform one global LMH proposal based on the sub-model A k * that was chosen to run local inference on, generating a new possible path A k .

If A k corresponds to an existing SLP, this sample is simply discarded.

However, if it corresponds to an unseen path, it is added to our stack of models as a new SLP.

To avoid the rate of models being generated outstripping our ability to perform inference on current models, this new SLP is not considered a candidate for the resource allocation (as per the next section) until some threshold for the number of times it has been proposed is reached.

This also provides a mechanism for providing distinct starting points for the N MCMC changes that will be run.

We note that in cases where there is a small number of discrete draws, it can sometimes be beneficial to partition our SLPs further into separate models for distinct values of these discrete variables to aid the mixing of the local MCMC sampler.

Given this dynamic set of candidate SLPs, we must now, at each iteration, choose a SLP to perform local inference on.

Though valid, it is not wise to evenly split our computational resources evenly among all SLPs; it is more important to ensure we have accurate estimates for SLPs with large Z k .

To address this, we introduce a resource allocation scheme, based on Rainforth et al. (2018) .

The resource allocation scheme is based on an Upper Confidence Bound (UCB) scheme (Carpentier et al., 2015) .

Specifically, at each iteration we will update the estimate for the SLP which has the largest utility, defined as

where L k is the number of times DCC has performed local inference on A k , τ k is the "exploitation target" of A k (explained below) andτ k = τ k / max{τ 1:K },p j is a target exploration term (explained below), and δ and β are hyper-parameters, adapted from Rainforth et al.

(2018, Eq. 6 in §5).

As proved in Rainforth et al. (2018, §5.1) , the optimal asymptotic allocation strategy is to choose each A k in proportion to τ k = Z 2 k + (1 + κ)σ 2 k where κ is a smoothness hyper-parameter, Z k is the local marginal likelihood, and σ 2 k is the variance of the weights of the individual samples used to generate Z k .

Intuitively, this allocates resources not only We compare DCC (ours) against IS and RMH on the convergence of the log marginal likelihood (4(a)) and the posterior distribution of the number of the clusters (4(b)) over 15 independent runs.

The ground truth of the log marginal and the posterior probability p(K = 5 | y 1:Ny ) were estimated using a large number of samples with a manually adapted proposal.

In Figure 4 (a), we show the squared error of the log marginal likelihood estimates with the solid line being the median and the shading region 25% − 75% quantiles.

In Figure 4 (b), we report the estimated posterior probability of K = 5 of each run, where the true estimate is around 0.9998.

We see that DCC substantially outperforms the baselines for both.

to the SLPs with high marginal probability mass, but also to the ones having high variance on our estimate of it.

We normalize τ k by the maximum of τ 1:K as the reward function in UCB is usually in [0, 1].

The target exploration termp j is a subjective tail-probability estimate on how much the local inference could improve in estimating the local marginal likelihood if given more computations.

This is motivated by the fact that estimating Z k accurately is difficult, especially at the early stage of inference.

One might miss substantial modes if only relying on optimism boost to undertake exploration.

As per Rainforth et al. (2018) , we realize this insight by extracting additional information from the log weights.

Namely, we definê p k := P (ŵ k (T a ) > w th ) ≈ 1 − Ψ k (log w th ) Ta , which means the probability of obtaining at least one sample with weight w that exceeds some threshold weight w th if provided with T a "look-ahead" samples.

Here Ψ k (·) is a cumulative density estimator of the log local weights, T a is a hyperparameter, and w th can be set to the maximum weight so far among all SLPs.

Ifp k is high, it implies that there is a high chance that one can produce higher estimates of Z k given more budget.

To generate a function in this model, we first sample its structure from the PCFG R. Next, we decide parameters in the sampled structure, by treating the parameters as all different variables and sampling them from the prior distribution.

Let Θ be the collection of all the latent variables used in this generative process.

That is, Θ consists of the discrete variables recording the choices of the grammar rules and the coefficients in the sampled structure.

Conditioned on the training data D, we want to infer the posterior distribution p(Θ|D), and calculate the predictive distribution for the test data D = {(x * j , y * j )} N j=1 .

In our experiment, we control the number of sub-models by requiring that the model use the PCFG in a restricted way: a sampled function structure should have depth at most 3 and cannot use the plus rule consecutively.

We generate a synthetic dataset of 30 training data points ( Figure 5 , blue points) and compare the performance of DCC against IS and RMH on estimating the posterior distribution and the posterior predictive under the same computational budget (one million samples in total) over 15 independent runs.

Figure 5 shows the posterior samples generated by DCC, IS and RMH over one run, with the training data D marked blue and the test data D in orange.

The DCC samples capture the periodicity of the training data and in general interpolates them well, while remaining uncertain in the regions of no data.

This indicates good inference results on both the structure of a function (determined by the PCFG) and the coefficients of the structure.

Though RMH does find some good functions, it becomes stuck in a particular mode and doe not fully capture the uncertainty in the model, leading to poor predictive performance.

Table 2 shows the test log marginal likelihood (LML) estimates of the three algorithms, i.e. LML := log N j=1 Θ p(y * j |x * j , Θ)p(Θ|D)dΘ. The LML measures how likely the predicted y * j on each test x * j is to be the true y * j in log scale.

We compared the LML for the three algorithms over 15 independent runs.

DCC clearly outperforms the baselines both in terms of predictive accuracy and stability.

The samples from IS approximate the posterior badly so unsurprisingly its LML is low.

RMH has a LML "close" to DCC, though the probability is 200 smaller in non-log space.

A more substantial problem in RMH is its high variance of the LML.

This is caused by it struggling to move and the results from runs to runs vary significantly.

To test the effectiveness of the resource allocation strategy of DCC, we also compare computational resource spent for each sub-model A k with the convergence of the local marginal likelihood estimate logẐ k of A k .

Our comparison is shown in Figure 6 , which implies that DCC indeed spends more computational resource on sub-models with high probability mass, while also exploring the other sub-models occasionally.

For our training data D, four sub-models (out of 26) contain most of the probability mass.

Two of them (models 15, 18) are functions of the form f (x) = a 1 x + a 2 sin(a 3 x 2 ) modulo symmetry, which is used to generate D. The other two submodels (models 23,24) are functions having the form f (x) = a 1 sin(a 2 x) + a 3 sin(a 4 x 2 ), which can also match the data well in the region of the training data (−1.5, 1.5) (under appropriately chosen a i 's).

@highlight

Divide, Conquer, and Combine is a new inference scheme that can be performed on the probabilistic programs with stochastic support, i.e. the very existence of variables is stochastic.