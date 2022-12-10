Bayesian phylogenetic inference is currently done via Markov chain Monte Carlo with simple mechanisms for proposing new states, which hinders exploration efficiency and often requires long runs to deliver accurate posterior estimates.

In this paper we present an alternative approach: a variational framework for Bayesian phylogenetic analysis.

We approximate the true posterior using an expressive graphical model for tree distributions, called a subsplit Bayesian network, together with appropriate branch length distributions.

We train the variational approximation via stochastic gradient ascent and adopt multi-sample based gradient estimators for different latent variables separately to handle the composite latent space of phylogenetic models.

We show that our structured variational approximations are flexible enough to provide comparable posterior estimation to MCMC, while requiring less computation due to a more efficient tree exploration mechanism enabled by variational inference.

Moreover, the variational approximations can be readily used for further statistical analysis such as marginal likelihood estimation for model comparison via importance sampling.

Experiments on both synthetic data and real data Bayesian phylogenetic inference problems demonstrate the effectiveness and efficiency of our methods.

Bayesian phylogenetic inference is an essential tool in modern evolutionary biology.

Given an alignment of nucleotide or amino acid sequences and appropriate prior distributions, Bayesian methods provide principled ways to assess the phylogenetic uncertainty by positing and approximating a posterior distribution on phylogenetic trees .

In addition to uncertainty quantification, Bayesian methods enable integrating out tree uncertainty in order to get more confident estimates of parameters of interest, such as factors in the transmission of Ebolavirus BID4 .

Bayesian methods also allow complex substitution models BID24 , which are important in elucidating deep phylogenetic relationships (Feuda et al., 2017) .Ever since its introduction to the phylogenetic community in the 1990s, Bayesian phylogenetic inference has been dominated by random-walk Markov chain Monte Carlo (MCMC) approaches BID43 BID26 ).

However, this approach is fundamentally limited by the complexities of tree space.

A typical MCMC method for phylogenetic inference involves two steps in each iteration: first, a new tree is proposed by randomly perturbing the current tree, and second, the tree is accepted or rejected according to the Metropolis-Hastings acceptance probability.

Any such random walk algorithm faces obstacles in the phylogenetic case, in which the high-posterior trees are a tiny fraction of the combinatorially exploding number of trees.

Thus, major modifications of trees are likely to be rejected, restricting MCMC tree movement to local modifications that may have difficulty moving between multiple peaks in the posterior distribution BID41 .

Although recent MCMC methods for distributions on Euclidean space use intelligent proposal mechanisms such as Hamiltonian Monte Carlo BID30 , it is not straightforward to extend such algorithms to the composite structure of tree space, which includes both tree topology (discrete object) and branch lengths (continuous positive vector) BID3 .Variational inference (VI) is an alternative approximate inference method for Bayesian analysis which is gaining in popularity BID17 BID40 BID0 .

Unlike MCMC methods that sample from the posterior, VI selects the best candidate from a family of tractable distributions to minimize a statistical distance measure to the target posterior, usually the Kullback-Leibler (KL) divergence.

By reformulating the inference problem into an optimization problem, VI tends to be faster and easier to scale to large data (via stochastic gradient descent) BID0 .

However, VI can also introduce a large bias if the variational distribution is insufficiently flexible.

The success of variational methods, therefore, relies on having appropriate tractable variational distributions and efficient training procedures.

To our knowledge, there have been no previous variational formulations of Bayesian phylogenetic inference.

This has been due to the lack of an appropriate family of approximating distributions on phylogenetic trees.

However the prospects for variational inference have changed recently with the introduction of subsplit Bayesian networks (SBNs) BID46 , which provide a family of flexible distributions on tree topologies (i.e. trees without branch lengths).

SBNs build on previous work BID13 BID23 , but in contrast to these previous efforts, SBNs are sufficiently flexible for real Bayesian phylogenetic posteriors BID46 .In this paper, we develop a general variational inference framework for Bayesian phylogenetics.

We show that SBNs, when combined with appropriate approximations for the branch length distribution, can provide flexible variational approximations over the joint latent space of phylogenetic trees with branch lengths.

We use recently-proposed unbiased gradient estimators for the discrete and continuous components separately to enable efficient stochastic gradient ascent.

We also leverage the similarity of local structures among trees to reduce the complexity of the variational parameterization for the branch length distributions and provide an extension to better capture the between-tree variation.

Finally, we demonstrate the effectiveness and efficiency of our methods on both synthetic data and a benchmark of challenging real data Bayesian phylogenetic inference problems.

Phylogenetic Posterior A phylogenetic tree is described by a tree topology τ and associated nonnegative branch lengths q. The tree topology τ represents the evolutionary diversification of the species.

It is a bifurcating tree with N leaves, each of which has a label corresponding to one of the observed species.

The internal nodes of τ represent the unobserved characters (e.g. DNA bases) of the ancestral species.

A continuous-time Markov model is often used to describe the transition probabilities of the characters along the branches of the tree.

Let DISPLAYFORM0 be the observed sequences (with characters in Ω) of length M over N species.

The probability of each site observation Y i is defined as the marginal distribution over the leaves DISPLAYFORM1 where ρ is the root node (or any internal node if the tree is unrooted and the Markov model is time reversible), a i ranges over all extensions of Y i to the internal nodes with a i u being the assigned character of node u, E(τ ) denotes the set of edges of τ , P ij (t) denotes the transition probability from character i to character j across an edge of length t and η is the stationary distribution of the Markov model.

Assuming different sites are identically distributed and evolve independently, the likelihood of observing the entire sequence set DISPLAYFORM2 The phylogenetic likelihood for each site in equation 1 can be evaluated efficiently through the pruning algorithm (Felsenstein, 2003) , also known as the sum-product algorithm in probabilistic graphical models BID38 BID21 BID12 .

Given a proper prior distribution with density p(τ, q) imposed on the tree topologies and the branch lengths, the phylogenetic posterior p(τ, q|Y ) is proportional to the joint density DISPLAYFORM3 where p(Y ) is the intractable normalizing constant.

We now review subsplit Bayesian networks BID46 ) and the flexible distributions on tree topologies they provide.

Let X be the set of leaf labels.

The dashed gray subgraphs represent fake splitting processes where splits are deterministically assigned, and are used purely to complement the networks such that the overall network has a fixed structure.

Right: The SBN for these examples.

We call a nonempty subset of X a clade.

Let be a total order on clades (e.g., lexicographical order).

A subsplit (W, Z) of a clade X is an ordered pair of disjoint subclades of X such that W ∪ Z = X and W Z. A subsplit Bayesian network B X on a leaf set X of size N is a Bayesian network whose nodes take on subsplit or singleton clade values that represent the local topological structure of trees FIG0 ).

Following the splitting processes (see the solid dark subgraphs in FIG0 , middle right), rooted trees have unique subsplit decompositions and hence can be uniquely represented as compatible SBN assignments.

Given the subsplit decomposition of a rooted tree τ = {s 1 , s 2 , . . .}, where s 1 is the root subsplit and {s i } i>1 are other subsplits, the SBN tree probability is DISPLAYFORM0 where S i denotes the subsplit-or singleton-clade-valued random variables at node i and π i is the index set of the parents of S i .

The Bayesian network formulation of SBNs enjoys many benefits: i) flexibility.

The expressiveness of SBNs is freely adjustable by changing the dependency structures between nodes, allowing for a wide range of flexible distributions; ii) normality.

SBN-induced distributions are all naturally normalized if the associated conditional probability tables (CPTs) are consistent, which is a common property of Bayesian networks.

The SBN framework also generalizes to unrooted trees, which are the most common type of trees in phylogenetics.

Concretely, unrooted trees can be viewed as rooted trees with unobserved roots.

Marginalizing out the unobserved root node S 1 , we have the SBN probability estimates for unrooted trees DISPLAYFORM1 where ∼ means all root subsplits that are compatible with τ .To reduce model complexity and encourage generalization, the same set of CPTs for parent-child subsplit pairs is shared across the SBN network, regardless of their locations.

Similar to weight sharing used in convolutional networks BID25 for detecting translationally-invariant structure of images (e.g., edges, corners), this heuristic parameter sharing used in SBNs is for identifying conditional splitting patterns of phylogenetic trees.

See BID46 for more detailed discussion on SBNs.

The flexible and tractable tree topology distributions provided by SBNs serve as an essential building block to perform variational inference BID17 for phylogenetics.

Suppose that we have a family of approximate distributions Q φ (τ ) (e.g., SBNs) over phylogenetic tree topologies, where φ denotes the corresponding variational parameters (e.g., CPTs for SBNs).

For each tree τ , we posit another family of densities Q ψ (q|τ ) over the branch lengths, where ψ is the branch length variational parameters.

We then combine these distributions and use the product DISPLAYFORM0 as our variational approximation.

Inference now amounts to finding the member of this family that minimizes the Kullback-Leibler (KL) divergence to the exact posterior, DISPLAYFORM1 which is equivalent to maximizing the evidence lower bound (ELBO), DISPLAYFORM2 As the ELBO is based on a single-sample estimate of the evidence, it heavily penalizes samples that fail to explain the observed sequences.

As a result, the variational approximation tends to cover only the high-probability areas of the true posterior.

This effect can be minimized by averaging over K > 1 samples when estimating the evidence BID2 BID29 , which leads to tighter lower bounds DISPLAYFORM3 where DISPLAYFORM4 ; the tightness of the lower bounds improves as the number of samples K increases BID2 .

We will use multi-sample lower bounds in the sequel and refer to them as lower bounds for short.

The CPTs in SBNs are, in general, associated with all possible parent-child subsplit pairs.

Therefore, in principle a full parameterization requires an exponentially increasing number of parameters.

In practice, however, we can find a sufficiently large subsplit support of CPTs (i.e. where the associated conditional probabilities are allowed to be nonzero) that covers favorable subsplit pairs from trees in the high-probability areas of the true posterior.

In this paper, we will mostly focus on the variational approach and assume the support of CPTs is available, although in our experiments we find that a simple bootstrap-based approach does provide a reasonable CPT support estimate for real data.

We leave the development of more sophisticated methods for finding the support of CPTs to future work.

Now denote the set of root subsplits in the support as S r and the set of parent-child subsplit pairs in the support as S ch|pa .

The CPTs are defined according to the following equations DISPLAYFORM0 where S ·|t denotes the set of child subsplits for parent subsplit t.

We use the Log-normal distribution Lognormal(µ, σ 2 ) as our variational approximation for branch lengths to accommodate their non-negative nature in phylogenetic models.

Instead of a naive parameterization for each edge on each tree (which would require a large number of parameters when the high-probability areas of the posterior are diffuse), we use an amortized set of parameters over the shared local structures among trees.

A simple choice of such local structures is the split, a bipartition (X 1 , X 2 ) of the leaf labels X (i.e. X 1 ∪ X 2 = X , X 1 ∩ X 2 = ∅), and each edge of a phylogenetic tree naturally corresponds to a split, the bipartition that consists of the leaf labels from both sides of the edge.

Note that a split can be viewed as a root subsplit.

We then assign µ(·, ·), σ(·, ·) for each split (·, ·) in S r .

We denote the corresponding split of edge e of tree τ as e/τ .A Simple Independent Approximation Given a phylogenetic tree τ , we start with a simple model that assumes the branch lengths for the edges of the tree are independently distributed.

The approximate density Q ψ (q|τ ), therefore, has the form DISPLAYFORM1 Figure 2: Branch length parameterization using primary subsplit pairs, which is the sum of parameters for a split and its neighboring subsplit pairs.

Edge e represents a split (W, Z).

Parameterization for the variance is the same as for the mean.

The above approximation equation 4 implicitly assumes that the branch lengths in different trees have the same distribution if they correspond to the same split, which fails to account for between-tree variation.

To capture this variation, one can use a more sophisticated parameterization that allows other tree-dependent terms for the variational parameters µ and σ.

Specifically, we use additional local structure associated with each edge as follows:Definition 1 (primary subsplit pair) Let e be an edge of a phylogenetic tree τ which represents a split e/τ = (W, Z).

Assume that at least one of W or Z, say W , contains more than one leaf label and denote its subsplit as (W 1 , W 2 ).

We call the parent-child subsplit pair (W 1 , W 2 )|(W, Z) a primary subsplit pair.

We assign additional parameters for each primary subsplit pair.

Denoting the primary subsplit pair(s) of edge e in tree τ as e/ /τ , we then simply sum all variational parameters associated with e to form the mean and variance parameters for the corresponding branch length (Figure 2 ): DISPLAYFORM0 This modifies the density in equation 4 by adding contributions from primary subsplit pairs and hence allows for more flexible between-tree approximations.

Note that the above structured parameterizations of branch length distributions also enable joint learning across tree topologies.

In practice, the lower bound is usually maximized via stochastic gradient ascent (SGA).

However, the naive stochastic gradient estimator obtained by differentiating the lower bound has very large variance and is impractical for our purpose.

Fortunately, various variance reduction techniques have been introduced in recent years including the control variate BID31 BID32 BID28 BID29 for general latent variables and the reparameterization trick BID20 for continuous latent variables.

In the following, we apply these techniques to different components of our latent variables and derive efficient gradient estimators with much lower variance, respectively.

In addition, we also consider a stable gradient estimator based on an alternative variational objective.

See Appendix A for derivations.

BID29 propose a localized learning signal strategy that significantly reduces the variance of the naive gradient estimator by utilizing the independence between the multiple samples and the regularity of the learning signal, which estimates the gradient as follows DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 is the per-sample local learning signal, withf φ,ψ (τ −j , q −j ) being some estimate of f φ,ψ (τ j , q j )for sample j using the rest of samples (e.g., the geometric mean), DISPLAYFORM3 is the self-normalized importance weight.

This gives the following VIMCO estimator DISPLAYFORM4 The Reparameterization Trick The VIMCO estimator also works for the branch length gradient.

However, as branch lengths are continuous latent variables, we can use the reparameterization trick to estimate the gradient.

Because the Log-normal distribution has a simple reparameterization, q ∼ Lognormal(µ, σ 2 ) ⇔ q = exp(µ + σ ), ∼ N (0, 1), we can rewrite the lower bound: DISPLAYFORM5 where g ψ ( |τ ) = exp(µ ψ,τ + σ ψ,τ ).

Then the gradient of the lower bound w.r.t.

ψ is DISPLAYFORM6 DISPLAYFORM7 is the same normalized importance weight as in equation equation 5.

Therefore, we can form the Monte Carlo estimator of the gradient DISPLAYFORM8 Self-normalized Importance Sampling Estimator In addition to the standard variational formulation equation 2, one can reformulate the optimization problem by minimizing the reversed KL divergence, which is equivalent to maximizing the likelihood of the variational approximation DISPLAYFORM9 We can use an importance sampling estimator to compute the gradient of the objective DISPLAYFORM10 with the same importance weightsw j as in equation 5.

This can be viewed as a multi-sample generalization of the wake-sleep algorithm BID11 and was first used in the reweighted wake-sleep algorithm BID1 for training deep generative models.

We therefore call the gradient estimator in equation 10 the RWS estimator.

Like the VIMCO estimator, the RWS estimator also provides gradients for branch lengths.

However, we find in practice that equation 8 that uses the reparameterization trick is more useful and often leads to faster convergence, although it uses a different optimization objective.

A better understanding of this phenomenon would be an interesting subject of future research.

All stochastic gradient estimators introduced above can be used in conjunction with stochastic optimization methods such as SGA or some of its adaptive variants (e.g. Adam BID19 to maximize the lower bounds.

See algorithm 1 in Appendix B for a basic variational Bayesian phylogenetic inference (VBPI) approach.

Throughout this section we evaluate the effectiveness and efficiency of our variational framework for inference over phylogenetic trees.

The simplest SBN (the one with a full and complete binary tree structure) is used for the phylogenetic tree topology variational distribution; we have found it to provide sufficiently accurate approximation.

For real datasets, we estimate the CPT supports from ultrafast maximum likelihood phylogenetic bootstrap trees using UFBoot BID27 , which is a fast approximate bootstrap method based on efficient heuristics.

We compare the performance of the VIMCO estimator and the RWS estimator with different variational parameterizations for the branch length distributions, while varying the number of samples in the training objective to see how these affect the quality of the variational approximations.

For VIMCO, we use Adam for stochastic gradient ascent with learning rate 0.001 BID19 .

For RWS, we also use AMSGrad BID36 , a recent variant of Adam, when Adam is unstable.

Results were collected after 200,000 parameter updates.

The KL divergences reported are over the discrete collection of phylogenetic tree structures, from trained SBN distribution to the ground truth, and a low KL divergence means a high quality approximation of the distribution of trees.

To empirically investigate the representative power of SBNs to approximate distributions on phylogenetic trees under the variational framework, we first conduct experiments on a simulated setup.

We use the space of unrooted phylogenetic trees with 8 leaves, which contains 10395 unique trees in total.

Given an arbitrary order of trees, we generate a target distribution p 0 (τ ) by drawing a sample from the symmetric Dirichlet distributions Dir(β1) of order 10395, where β is the concentration parameter.

The target distribution becomes more diffuse as β increases; we used β = 0.008 to provide enough information for inference while allowing for adequate diffusion in the target.

Note that there are no branch lengths in this simulated model and the lower bound is DISPLAYFORM0 with the exact evidence being log(1) = 0.

We then use both the VIMCO and RWS estimators to optimize the above lower bound based on 20 and 50 samples (K).

We use a slightly larger learning rate (0.002) in AMSGrad for RWS.

FIG1 shows the empirical performance of different methods.

From the left plot, we see that the lower bounds converge rapidly and the gaps between lower bounds and the exact model evidence are close to zero, demonstrating the expressive power of SBNs on phylogenetic tree probability estimations.

The evolution of KL divergences (middle plot) is consistent with the lower bounds.

All methods benefit from using more samples, with VIMCO performing better in the end and RWS learning slightly faster at the beginning.

1 The slower start of VIMCO is partly due to the regularization term in the lower bounds, which turns out to be beneficial for the overall performance since the regularization encourages the diversity of the variational approximation and leads to more sufficient exploration in the starting phase, similar to the exploring starts (ES) strategy in reinforcement learning BID39 .

The right plot compares the variational approximations obtained by VIMCO and RWS, both with 50 samples, to the ground truth p 0 (τ ).

We see that VIMCO slightly underestimates trees in high-probability areas as a result of the regularization effect.

While RWS provides better approximations for trees in high-probability areas, it tends to underestimate trees in low-probability areas which deteriorates the overall performance.

We expect the biases in both approaches to be alleviated with more samples.

In the second set of experiments we evaluate the proposed variational Bayesian phylogenetic inference (VBPI) algorithms at estimating unrooted phylogenetic tree posteriors on 8 real datasets commonly used to benchmark phylogenetic MCMC methods BID22 BID13 BID23 BID41 TAB0 .

We concentrate on the most challenging part of the phylogenetic model: joint learning of the tree topologies and the branch lengths.

We assume a uniform prior on the tree topology, an i.i.d.

exponential prior (Exp(10)) for the branch lengths and the simple BID18 substitution model.

We consider two different variational parameterizations for the branch length distributions as introduced in section 3.1.

In the first case, we use the simple split-based parameterization that assigns parameters to the splits associated with the edges of the trees.

In the second case, we assign additional parameters for the primary subsplit pairs (PSP) to better capture the between-tree variation.

We form our ground truth posterior from an extremely long MCMC run of 10 billion iterations (sampled each 1000 iterations with the first 25% discarded as burn-in) using MrBayes BID34 , and gather the support of CPTs from 10 replicates of 10000 ultrafast maximum likelihood bootstrap trees BID27 .

Following BID33 , we use a simple annealed version of the lower bound which was found to provide better results.

The modified bound is: DISPLAYFORM0 where β t ∈ [0, 1] is an inverse temperature that follows a schedule β t = min(0.001, t/100000), going from 0.001 to 1 after 100000 iterations.

We use Adam with learning rate 0.001 to train the variational approximations using VIMCO and RWS estimators with 10 and 20 samples.

FIG2 (left and middle plots) shows the resulting KL divergence to the ground truth on DS1 as a function of the number of parameter updates.

The results for methods that adopt the simple splitbased parameterization of variational branch length distributions are shown in the left plot.

We see that the performance of all methods improves significantly as the number of samples is increased.

The middle plot, containing the results using PSP for variational parameterization, clearly indicates that a better modeling of between-tree variation of the branch length distributions is beneficial for all method / number of samples combinations.

Specifically, PSP enables more flexible branch length distributions across trees which makes the learning task much easier, as shown by the considerably smaller gaps between the methods.

To benchmark the learning efficiency of VBPI, we also compare to MrBayes 3.2.5 BID34 , a standard MCMC implementation.

We run MrBayes with 4 chains and 10 runs for two million iterations, sampling every 100 iterations.

For each run, we compute the KL divergence to the ground truth every 50000 iterations with the first 25% discarded 2 Although MCMC converges faster at the start, we see that VBPI methods (especially those with PSP) quickly surpass MCMC and arrive at good approximations with much less computation.

This is because VBPI iteratively updates the approximate distribution of trees (e.g., SBNs) which in turn allows guided exploration in the tree topology space.

VBPI also provides the same majority-rule consensus tree as the ground truth MCMC run ( FIG4 in Appendix D).The variational approximations provided by VBPI can be readily used to perform importance sampling for phylogenetic inference (more details in Appendix C).

The right plot of FIG2 compares VBPI using VIMCO with 20 samples and PSP to the state-of-the-art generalized stepping-stone (GSS) algorithm for estimating the marginal likelihood of trees in the 95% credible set of DS1.

For GSS, we use 50 power posteriors and for each power posterior we run 1,000,000 MCMC iterations, sampling every 1000 iterations with the first 10% discarded as burn-in.

The reference distribution for GSS was obtained from an independent Gamma approximation using the maximum a posterior estimate.

TAB0 shows the estimates of the marginal likelihood of the data (i.e., model evidence) using different VIMCO approximations and one of the state-of-the-art methods, the stepping-stone (SS) algorithm BID42 .

For each data set, all methods provide estimates for the same marginal likelihood, with better approximation leading to lower variance.

We see that VBPI using 1000 samples is already competitive with SS using 100000 samples and provides estimates with much less variance (hence more reproducible and reliable).

Again, the extra flexibility enabled by PSP alleviates the demand for larger number of samples used in the training objective, making it possible to achieve high quality variational approximations with less samples.

In this work we introduced VBPI, a general variational framework for Bayesian phylogenetic inference.

By combining subsplit Bayesian networks, a recent framework that provides flexible distributions of trees, and efficient structured parameterizations for branch length distributions, VBPI exhibits guided exploration (enabled by SBNs) in tree space and provides competitive performance to MCMC methods with less computation.

Moreover, variational approximations provided by VBPI can be readily used for further statistical analysis such as marginal likelihood estimation for model comparison via importance sampling, which, compared to MCMC based methods, dramatically reduces the cost at test time.

We report promising numerical results demonstrating the effectiveness and efficiency of VBPI on a benchmark of real data Bayesian phylogenetic inference problems.

When the data are weak and posteriors are diffuse, support estimation of CPTs becomes challenging.

However, compared to classical MCMC approaches in phylogenetics that need to traverse the enormous support of posteriors on complete trees to accurately evaluate the posterior probabilities, the SBN parameterization in VBPI has a natural advantage in that it alleviates this issue by factorizing the uncertainty of complete tree topologies into local structures.

Many topics remain for future work: constructing more flexible approximations for the branch length distributions (e.g., using normalizing flow BID33 for within-tree approximation and deep networks for the modeling of between-tree variation), deeper investigation of support estimation approaches in different data regimes, and efficient training algorithms for general variational inference on discrete / structured latent variables.

In this section we will derive the gradient for the multi-sample objectives introduced in section 3.

We start with the lower bound DISPLAYFORM0 Again, p(τ, q|Y ) is independent of φ, ψ, and we have DISPLAYFORM1 The second to last step uses self-normalized importance sampling with K samples.

∇ ψL (φ, ψ) can be computed in a similar way.

In this section, we provide a detailed importance sampling procedure for marginal likelihood estimation for phylogenetic inference based on the variational approximations provided by VBPI.

For each tree τ that is covered by the subsplit support, DISPLAYFORM0 p Lognormal (q e | µ(e, τ ), σ(e, τ ))can provide accurate approximation to the posterior of branch lengths on τ , where the mean and variance parameters µ(e, τ ), σ(e, τ ) are gathered from the structured variational parameters ψ as introduced in section 3.1.

Therefore, we can estimate the marginal likelihood of τ using importance sampling with Q ψ (q|τ ) being the importance distribution as follows DISPLAYFORM1 Q ψ (q j |τ ) with q j iid ∼ Q ψ (q|τ )

Similarly, we can estimate the marginal likelihood of the data as follows DISPLAYFORM0 Q φ (τ j )Q ψ (q j |τ j ) with τ j , q j iid ∼ Q φ,ψ (τ, q).In our experiments, we use K = 1000.

When taking a log transformation, the above Monte Carlo estimate is no longer unbiased (for the evidence log p(Y )).

Instead, it can be viewed as one sample Monte Carlo estimate of the lower bound DISPLAYFORM1 whose tightness improves as the number of samples K increases.

Therefore, with a sufficiently large K, we can use the lower bound estimate as a proxy for Bayesian model selection.

Alligator mississippiensis

<|TLDR|>

@highlight

The first variational Bayes formulation of phylogenetic inference, a challenging inference problem over structures with intertwined discrete and continuous components

@highlight

Explores an approximate inference solution to the problem of Bayesian inference of phylogenetic trees by leveraging recently proposed subsplit Bayesian networks and modern gradient estimators for VI.

@highlight

Proposes a variational approach to Bayesian posterior inference in phylogenetic trees.