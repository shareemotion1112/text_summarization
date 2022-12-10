Progress in probabilistic generative models has accelerated, developing richer models with neural architectures, implicit densities, and with scalable algorithms for their Bayesian inference.

However, there has been limited progress in models that capture causal relationships, for example, how individual genetic factors cause major human diseases.

In this work, we focus on two challenges in particular:

How do we build richer causal models, which can capture highly nonlinear relationships and interactions between multiple causes?

How do we adjust for latent confounders, which are variables influencing both cause and effect and which prevent learning of causal relationships?

To address these challenges, we synthesize ideas from causality and modern probabilistic modeling.

For the first, we describe implicit causal models, a class of causal models that leverages neural architectures with an implicit density.

For the second, we describe an implicit causal model that adjusts for confounders by sharing strength across examples.

In experiments, we scale Bayesian inference on up to a billion genetic measurements.

We achieve state of the art accuracy for identifying causal factors: we significantly outperform the second best result by an absolute difference of 15-45.3%.

Probabilistic models provide a language for specifying rich and flexible generative processes BID5 Murphy, 2012) .

Recent advances expand this language with neural architectures, implicit densities, and with scalable algorithms for their Bayesian inference BID9 BID15 .

However, there has been limited progress in models that capture high-dimensional causal relationships (Pearl, 2000; BID13 Imbens & Rubin, 2015) .

Unlike models which learn statistical relationships, causal models let us manipulate the generative process and make counterfactual statements, that is, what would have happened if the distributions changed.

As the running example in this work, consider genome-wide association studies (GWAS) BID19 BID7 Kang et al., 2010) .

The goal of GWAS is to understand how genetic factors, i.e., single nucleotide polymorphisms (SNPs), cause traits to appear in individuals.

Understanding this causation both lets us predict whether an individual has a genetic predisposition to a disease and also understand how to cure the disease by targeting the individual SNPs that cause it.

With this example in mind, we focus on two challenges to combining modern probabilistic models and causality.

The first is to develop richer, more expressive causal models.

Probabilistic causal models represent variables as deterministic functions of noise and other variables, and existing work usually focuses on additive noise models (Hoyer et al., 2009 ) such as linear mixed models (Kang et al., 2010) .

These models apply simple nonlinearities such as polynomials, hand-engineered low order interactions between inputs, and assume additive interaction with, e.g., Gaussian noise.

In GWAS, strong evidence suggests that susceptibility to common diseases is influenced by epistasis (the interaction between multiple genes) (Culverhouse et al., 2002; McKinney et al., 2006) .

We would like to capture and discover such interactions.

This requires models with nonlinear, learnable interactions among the inputs and the noise.

The second challenge is how to address latent population-based confounders.

In GWAS, both latent population structure, i.e., subgroups in the population with ancestry differences, and relatedness among sample individuals produce spurious correlations among SNPs to the trait of interest.

Existing methods correct for this correlation in two stages BID19 BID7 Kang et al., 2010) : first, estimate the confounder given data; then, run standard causal inferences given the estimated confounder.

These methods are effective in some settings, but they are difficult to understand as principled causal models, and they cannot easily accommodate complex latent structure.

To address these challenges, we synthesize ideas from causality and modern probabilistic modeling.

For the first challenge, we develop implicit causal models, a class of causal models that leverages neural architectures with an implicit density.

With GWAS, implicit causal models generalize previous methods to capture important nonlinearities, such as gene-gene and gene-population interaction.

Building on this, for the second challenge, we describe an implicit causal model that adjusts for population-confounders by sharing strength across examples (genes).

We derive conditions that prove the model consistently estimates the causal relationship.

This theoretically justifies existing methods and generalizes them to more complex latent variable models of the confounder.

In experiments, we scale Bayesian inference on implicit causal models on up to a billion genetic measurements.

Validating these results are not possible for observational data (Pearl, 2000) , so we first perform an extensive simulation study of 11 configurations of 100,000 SNPs and 940 to 5,000 individuals.

We achieve state of the art accuracy for identifying causal factors: we significantly outperform existing genetics methods by an absolute difference of 15-45.3%.

In a real-world GWAS, we also show our model discovers real causal relationships-identifying similar SNPs as previous state of the art-while being more principled as a causal model.

There has been growing work on richer causal models.

Louizos et al. (2017) develop variational auto-encoders for causality and address local confounders via proxy variables.

Our work is complementary: we develop implicit models for causality and address global confounders by sharing strength across examples.

In other work, Mooij et al. (2010) propose a Gaussian process over causal mechanisms, and BID21 study post-nonlinear models, which apply a nonlinearity after adding noise.

These models typically focus on the task of causal discovery, and they assume fixed nonlinearities (post-nonlinear models) or impose strong smoothness assumptions (Gaussian processes) which we relax using neural networks.

In the potential outcomes literature, much recent work has considered decision trees and neural networks (e.g., Hill (2011); BID16 ; Johansson et al. (2016) ).

These methods tackle a related but different problem of balancing covariates across treatments.

Causality with population-confounders has primarily been studied for genome-wide association studies (GWAS).

A popular approach is to first, estimate the confounder using the top principal components of the genotype matrix of individuals-by-SNPs; then, linearly regress the trait of interest onto the genetic factors and these components BID7 BID1 ).

Another approach is to first, estimate the confounder via a "kinship matrix" on the genotypes; then, fit a linear mixed model of the trait given genetic factors, and where the covariance of random effects is the kinship matrix BID19 Kang et al., 2010) .

Other work adjusts for the confounder via admixture models and factor analysis BID12 Hao et al., 2016) .

This paper builds on all these methods, providing a theoretical understanding about when causal inferences can succeed while adjusting for latent confounders.

We also develop a new causal model with nonlinear, learnable gene-gene and gene-population interactions; and we describe a Bayesian inference algorithm that justifies the two-stage estimation.

The problem of epistasis, that is, nonadditive interactions between multiple genes, dates back to classical work on epigenetics by Bateson and R.A. Fisher (Fisher, 1918) .

Primary methods for capturing epistasis include adding interactions within a linear model, permutation tests, exhaustive search, and multifactor dimensionality reduction BID4 .

These methods require hand-engineering over all possible interactions, which grows exponentially in the number of genetic factors.

Neural networks have been applied to address epistasis for epigenomic data, such as to predict sequence specificities of protein bindings given DNA sequences BID0 .

These methods use discriminative neural networks (unlike neural networks within a generative model), and they focus on prediction rather than causality.

We describe the framework of probabilistic causal models.

We then describe implicit causal models, an extension of implicit models for encoding complex, nonlinear causal relationships.

Probabilistic causal models (Pearl, 2000) , or structural equation models, represent variables as deterministic functions of noise and other variables.

As illustration, consider the causal diagram in FIG0 .

It represents a causal model where there is a global variable DISPLAYFORM0 and for each data point n = 1, . . .

, N , DISPLAYFORM1 The noise are background variables, representing unknown external quantities which are jointly independent.

Each variable β, x, y is a function of other variables and its background variable.

We are interested in estimating the causal mechanism f y .

It lets us calculate quantities such as the causal effect p(y | do(X = x), β), the probability of an outcome y given that we force X to a specific value x and under fixed global structure β.

This quantity differs from the conditional p(y | x, β).

The conditional takes the model and filters to the subpopulation where X = x; in general, the processes which set X to that value may also have influenced Y .

Thus the conditional is not the same as if we had manipulated X directly (Pearl, 2000) .Under the causal graph of FIG0 , the adjustment formula says that p(y | do(x), β) = p(y | x, β).

This means we can estimate f y from observational data {(x n , y n )}, assuming we observe the global structure β.

For example, an additive noise model (Hoyer et al., 2009 ) posits DISPLAYFORM2 where f (·) might be a linear function of the concatenated inputs, f (·) = [x n , β] θ, or it might use spline functions for nonlinearities.

If s(·) is standard normal, the induced density for y is normal with unit variance.

Placing a prior over parameters p(θ), Bayesian inference yields DISPLAYFORM3 The right hand side is a joint density whose individual components can be calculated.

We can use standard algorithms, such as variational inference or MCMC (Murphy, 2012) .A limitation in additive noise models that they typically apply simple nonlinearities such as polynomials, hand-engineered low-order interactions between inputs, and assume additive interaction with, e.g., Gaussian noise.

Next we describe how to build richer causal models which relax these restrictions.

(An additional problem is that we typically don't observe β; we address this in § 3.)

Implicit models capture an unknown distribution by hypothesizing about its generative process (Diggle & Gratton, 1984; BID15 .

For a distribution p(x) of observations x, recent advances define a function g that takes in noise ∼ s(·) and outputs x given parameters θ, Unlike models which assume additive noise, setting g to be a neural network enables multilayer, nonlinear interactions.

Implicit models also separate randomness from the transformation; this imitates the structural invariance of causal models (Equation 1).

DISPLAYFORM0 To enforce causality, we define an implicit causal model as a probabilistic causal model where the functions g form structural equations, that is, causal relations among variables.

Implicit causal models extend implicit models in the same way that causal networks extend Bayesian networks BID6 and path analysis extends regression analysis BID18 .

They are nonparametric structural equation models where the functional forms are themselves learned.

A natural question is the representational capacity of implicit causal models.

In general, they are universal approximators: we can use a fully connected network with a sufficiently large number of hidden units to approximate each causal mechanism.

We describe this formally.

Theorem (Universal Approximation Theorem).

Let the tuple (E, V, F, s(E)) denote a probabilistic causal model, where E represent the set of background variables with probability s(E), V the set of endogenous variables, and F the causal mechanisms.

Assume each causal mechanism is a continuous function on the m-dimensional unit cube f ∈ C([0, 1] m ).

Let σ be a nonconstant, bounded, and monotonically-increasing continuous function.

For each causal mechanism f and any error δ > 0, there exist parameters θ = {v, w, b}, for real constants v i , b i ∈ R and real vectors w i ∈ R m for i = 1, . . .

, H and fixed H, such that the following function approximates f : DISPLAYFORM1 The implicit model defined by the collection of functions g and same noise distributions universally approximates the true causal model.(This directly follows from the approximator theorem of, e.g., Cybenko (1989) .)

A key aspect is that implicit causal models are not only universal approximators, but that we can use fast algorithms for their Bayesian inference (to calculate Equation 2).

In particular, variational methods both scale to massive data and provide accurate posterior approximations ( § 4).

This lets us obtain good performance in practice with finite-sized neural networks; § 5 describes such experiments.

We described implicit causal models, a rich class of models that can capture arbitrary causal relations.

For simplicity, we assumed that the global structure is observed; this enables standard inference methods.

We now consider the typical setting when it is unobserved.

Consider the running example of genome-wide association studies (GWAS) FIG1 ).

There are N data points (individuals).

Each data point consists of an input vector of length M (measured SNPs), x n = [x n1 , . . .

, x nM ] and a scalar outcome y n (the trait of interest).

Typically, the number of measured SNPs M ranges from 100,000 to 1 million and the number of individuals N ranges from 500 to 10,000.We are interested in how changes to each SNP X m cause changes to the trait Y .

Formally, this is the causal effect p(y | do(x m ), x −m ), which is the probability of an outcome y given that we force SNP X m = x m and consider fixed remaining SNPs x −m .

Standard inference methods are confounded by the unobserved population structure of SNPs for each individual, as well as the individual's cryptic relatedness to other samples in the data set.

This confounder is represented as a latent variable z n , which influences x nm and y n for each data index n; see FIG1 .

Because we do not observe the z n 's, the causal effect p(y | do(x m ), x −m ) is unidentifiable BID13 .Building on previous GWAS methods ( BID7 BID19 BID1 , we build a model that jointly captures z n 's and the mechanisms for X m → Y .

Consider the implicit causal model where for each data point n = 1, . . .

, N and for each SNP m = 1, . . .

, M , DISPLAYFORM0 The function g z (·) for the confounder is fixed.

Each function g xm (· | w m ) per SNP depends on the confounder and has parameters w m .

The function g y (· | θ) for the trait depends on the confounder and all SNPs, and it has parameters θ.

We place priors over the parameters p(w m ) and p(θ).

To estimate the mechanism f y , we calculate the posterior of the outcome parameters θ, DISPLAYFORM1 Note how this accounts for the unobserved confounders: it assumes that p(z | x, y) accurately reflects the latent structure.

In doing so, we perform inferences for p(θ | x, y, z), averaged over posterior samples from p(z | x, y).In general, causal inference with latent confounders can be dangerous: it uses the data twice (once to estimate the confounder; another to estimate the mechanism), and thus it may bias our estimates of each arrow X m → Y .

Why is this justified?

We answer this below.

(See Appendix A for the proof.)

Proposition 1 rigorizes previous methods in the framework of probabilistic causal models.

The intuition is that as more SNPs arrive ("M → ∞, N fixed"), the posterior concentrates at the true confounders z n , and thus we can estimate the causal mechanism given each data point's confounder z n .

As more data points arrive ("N → ∞, M fixed"), we can estimate the causal mechanism given any confounder z n as there are infinity of them.

Connecting to Two-Stage Estimation.

Existing GWAS methods adjust for latent population structure using two stages BID1 ): first, estimate the confounders z 1:N ; second, infer the outcome parameters θ given the data set and the estimate of the confounders.

To incorporate uncertainty, a Bayesian version would not use a point estimate of z 1:N but the full posterior p(z 1:N | x, y); then it would infer θ given posterior samples of z 1:N .

Following Equation 5, this is the same as joint posterior inference.

Thus the two stage approach is justified as a Bayesian approximation that uses a point estimate of the posterior.

Above, we outlined how to specify an implicit causal model for GWAS.

We now specify in detail the functions and priors for the confounders z n , the SNPs x nm , and the traits y n (Equation 4).

FIG1 (right) visualizes the model we describe below.

Appendix B provides an example implementation in the Edward probabilistic programming language BID14 .Generative Process of Confounders z n .

We use standard normal noise and set the confounder function g z (·) to the identity.

This implies the distribution of confounders p(z n ) is standard normal.

Their dimension z n ∈ R K is a hyperparameter.

The dimension K should be set to the highest value such that the latent space most closely approximates the true population structure but smaller than the total number of SNPs to avoid overfitting.

Generative Process of SNPs x nm .

Designing nonlinear processes that return matrices is an ongoing research direction (e.g., Lawrence FORMULA3 ; Lloyd et al. FORMULA1 ).

To design one for GWAS (the SNP matrix), we build on an implicit modeling formulation of factor analysis; it has been successful in GWAS applications BID7 BID12 .

Let each SNP be encoded as a 0, 1, or 2 to denote the three possible genotypes.

This is unphased data, where 0 indicates two major alleles; 1 indicates one major and one minor allele; and 2 indicates two minor alleles.

Set DISPLAYFORM0 ).

This defines a Binomial(2, π nm ) distribution on x nm .

Analogous to generalized linear models, the Binomial's logit probability is linear with respect to z n .

We then sum up two Bernoulli trials: they are represented as indicator functions of whether a uniform sample is greater than the probability. (The uniform noises are newly drawn for each index n and m.)Assuming a standard normal prior on the variables w m , this generative process is equivalent to logistic factor analysis.

The variables w m act as "principal components," embedding the M -many SNPs within a subspace of lower dimension K.Logistic factor analysis makes strong assumptions: linear dependence on the confounder and that one parameter per dimension has sufficient representational capacity.

We relax these assumptions using a neural network over concatenated inputs, DISPLAYFORM1 Similar to the above, the variables w m serve as principal components.

The neural network takes an input of dimension 2K and outputs a scalar real value; its weights and biases φ are shared across SNPs m and individuals n. This enables learning of nonlinear interactions between z n and w m , preserves the model's conditional independence assumptions, and avoids the complexity of a neural net that outputs the full N × M matrix.

We place a standard normal prior over φ.

Generative Process of Traits y n .

To specify the traits, we build on an implicit modeling formulation of linear regression.

It is the mainstay tool in GWAS applications BID7 BID12 .

Formally, for real-valued y ∈ R, we model each observed trait as DISPLAYFORM2 n ∼ Normal(0, 1), This process assumes linear dependence on SNPs, no gene-gene and gene-population interaction, and additive noise.

We generalize this model using a neural network over the same inputs, DISPLAYFORM3 n ∼ Normal(0, 1).

The neural net takes an input of dimension M +K +1 and outputs a scalar real value; for categorical outcomes, the output is discretized over equally spaced cutpoints.

We also place a group Lasso prior on weights connecting a SNP to a hidden layer.

This encourages sparse inputs: we suspect few SNPs affect the trait BID20 .

We use standard normal for other weights and biases.

We described a rich causal model for how SNPs cause traits and that can adjust for latent populationconfounders.

Given GWAS data, we aim to infer the posterior of outcome parameters θ (Equation 5).

Calculating this posterior reduces to calculating the joint posterior of confounders z n , SNP parameters w m and φ, and trait parameters θ, DISPLAYFORM0 This means we can use typical inference algorithms on the joint posterior.

We then collapse variables to obtain the marginal posterior of θ. (For Monte Carlo methods, we drop the auxiliary samples; for variational methods, it is given if the variational family follows the posterior's factorization.)One difficulty is that with implicit models, evaluating the density is intractable: it requires integrating over a nonlinear function with respect to a high-dimensional noise (Equation 3).

Thus we require likelihood-free methods, which assume that one can only sample from the model's likelihood (Marin et al., 2012; BID15 .

Here we apply likelihood-free variational inference (LFVI), which we scale to billions of genetic measurements BID15 .As with all variational methods, LFVI posits a family of distributions over the latent variables and then optimizes to find the member closest to the posterior.

For the variational family, we specify normal distributions with diagonal covariance for the SNP components w m and confounder z n , q(w m ) = Normal(w m ; µ wm , σ wm I), q(z n ) = Normal(z n ; µ zn , σ zn I).We specify a point mass for the variational family on both neural network parameters φ and θ. (This is equivalent to point estimation in a variational EM setting.)For LFVI to scale to massive GWAS data, we use stochastic optimization by subsampling SNPs (Gopalan et al., 2016) .

At a high level, the algorithm proceeds in two stages.

In the first stage, LFVI cycles through the following steps:1.

Sample SNP location m and collect the observations at that location from all individuals.

This first stage infers the posterior distribution of confounders z n and SNPs parameters w m and φ.

Each step's computation is independent of the number of SNPs, allowing us to scale to millions of genetic factors.

In experiments, the algorithm converges while scanning over the full set of SNPs only once or twice.

In the second stage, we infer the posterior of outcome parameters θ given the inferred confounders from the first stage.

Appendix C describes the algorithm in more detail; it expands on the LFVI implementation in Edward BID14 .

We described implicit causal models, how to adjust for latent population-based confounders, and how to perform scalable variational inference.

In general, validating causal inferences on observational data is not possible (Pearl, 2000) .

Therefore to validate our work, we perform an extensive simulation study on 100,000 SNPs, 940 to 5,000 individuals, and across 100 replications of 11 settings.

The study indicates that our model is significantly more robust to spurious associations, with a state-of-the-art gain of 15-45.3% in accuracy.

We also apply our model to a real-world GWAS of Northern Finland Birth Cohorts; our model indeed captures real causal relationships-identifying similar SNPs as previous state of the art.

We compare against three methods that are currently state of the art: PCA with linear regression (Price et al., 2006) ("PCA"); a linear mixed model (with the EMMAX software) (Kang et al., 2010) ("LMM"); and logistic factor analysis with inverse regression BID12 ("GCAT").

In all experiments, we use Adam with a initial step-size of 0.005, initialize neural network parameters uniformly with He variance scaling (He et al., 2015) , and specify the neural networks for traits and SNPs as fully connected with two hidden layers, ReLU activation, and batch normalization (hidden layer sizes described below).

For the trait model's neural network, we found that including latent variables as input to the final output layer improves information flow in the network.

We analyze 11 simulation configurations, where each configuration uses 100,000 SNPs and 940 to 5,000 individuals.

We simulate 100 GWAS data sets per configuration for a grand total of 4,400 fitted models (4 methods of comparison Table 1 : Precision accuracy over an extensive set of configurations and methods; we average over 100 simulations for a grand total of 4,400 fitted models.

The setting a in PSD and Spatial determines the amount of sparsity in the latent population structure: lower a means higher sparsity.

ICM is significantly more robust to spurious associations, outperforming other methods by up to 45.3%. ; and four variations of a configuration where population structure is determined by a latent spatial position of individuals.

Only 10 of the 100,000 SNPs are set to be causal.

Appendix D provides more detail.

Table 1 displays the precision for predicting causal factors across methods.

When failing to account for population structure, "spurious associations" occur between genetic markers and the trait of interest, despite the fact that there is no biological connection.

Precision is the fraction of the number of true positives over the number of true and false positives.

This measures a method's robustness to spurious associations: higher precision means fewer false positives and thus more robustness.1 Table 1 shows that our method achieves state of the art across all configurations.

Our method especially dominates in difficult tasks with sparse (small a), spatial (Spatial), and/or mixed membership structure (PSD): there is over a 15% margin in difference to the second best in general, and up to a 45.3% margin on the Spatial (a = 0.01) configuration.

For simpler configurations, such as a mixture model (HapMap), our method has comparable performance.

We analyze a real-world GWAS of Northern Finland Birth Cohorts BID11 , which measure several metabolic traits and height and which contain 324,160 SNPs and 5,027 individuals.

We separately fitted 10 implicit causal models, each of which models the effect of SNPs on one of ten traits.

To specify the implicit causal models, we set the latent dimension of confounders to be 6 (following BID12 ).

We use 512 units in both hidden layers of the SNP neural network and use 32 and 256 units for the trait neural network's first and second hidden layers respectively.

Appendix E provides more detail.

TAB2 compares the number of identified causal SNPs across methods, with an additional "uncorrected" baseline, which does not adjust for any latent population structure.

Each method is performed with a subsequent correction as measured by the genomic control inflation factor BID11 ).

Our models identify similar causal SNPs as previous methods.

Interestingly, our model tends to agree with BID12 , identifying a total of 15 significant loci BID12 identified 16; others identified 11-14 loci).

This makes sense intuitively, as BID12 uses logistic factor analysis which, compared to all methods, most resembles our model.

TAB2 ).

The implicit causal model (ICM) captures causal relationships comparable to previously work.

We described implicit causal models, a rich class of models that can capture high-dimensional, nonlinear causal relationships.

With genome-wide association studies, implicit causal models generalize previous successful methods to capture important nonlinearities, such as gene-gene and gene-population interaction.

In addition, we described an implicit causal model that adjusts for confounders by sharing strength across examples.

Our model achieves state-of-the-art accuracy, significantly outperforming existing genetics methods by 15-45.3%.There are several limitations to learning true causal associations.

For example, alleles at different loci typically exhibit linkage disequilibrium, which is a local non-random association influenced by factors such as the rate of recombination, mutation, and genetic drift.

The implicit causal model might be extended with variables shared across subsets of SNPs to model the recombination process.

Another limitation involves the data, where granularity of sequenced loci may lose signal or attribute causation to a region involving multiple SNPs.

Better technology, and accounting for mishaps in the sequencing process in the model, can help.

While we focused on GWAS applications in this paper, we also believe implicit causal models have significant potential in other sciences: for example, to design new dynamical theories in high energy physics; and to accurately model structural equations of discrete choices in economics.

We're excited about applications to these new domains, leveraging modern probabilistic modeling and causality to drive new scientific understanding.

Consider the simplest setting in § 2, where the causal graph is as shown with a global confounder with finite dimension and where we observe the data set {(x n , y n )}.

Assume the specified model family over the causal graph includes the true data generating process.

First consider an atomic intevention do(X = x) and let β * be the true structural value that generated our observations.

The probability of a new outcome given the intervention and global structure is DISPLAYFORM0 This follows from the backdoor criterion on the empty set.

By Bernstein von-Mises, the posterior for our model p(β | x, y) concentrates at β * .

Thus, similarly, our posterior for θ given β * concentrates to the true functional mechanism f y .

This implies we have a consistent estimate of DISPLAYFORM1 This simple proof rigorizes the ideas behind learning and fixing population structure, a common heuristic in GWAS methods.

Moreover, it lets us understand how to extend them to more complex latent variable models of the confounder and also provide uncertainty estimates of the latent structure become important as we apply these methods to finite data in practice.

We provide an example of an implicit causal model written in the Edward language below.

It writes neural net weights and biases as model parameters.

It does not include priors on the weights or biases; we add these as penalties to the objective function during training.

The log-likelihood per-individual and per-SNP is log p(x nm , y n | w m , z n , θ, φ) = log p(y n | x n,1:M , z n , θ) + log p(x nm | w m , z n , φ).There are local priors p(w m ), p(z n ) and global priors p(θ), p(φ).

The posterior factorizes as DISPLAYFORM0 Let the variational family follow the posterior's factorization above.

For notational convenience, we drop the data dependence in q, DISPLAYFORM1 We write the evidence lower bound and decompose the model's log joint density and the variational density, DISPLAYFORM2 Assume that the variational family for z n and w m are independent of other variables, q(z n ) and q(w m ).

Also assume delta point masses for q(θ) = I[θ−θ ] parameterized by θ and q(φ) = I[φ−φ ] parameterized by φ .

This simplifies the objective, reducing to DISPLAYFORM3 Each expectation can be unbiasedly estimated with Monte Carlo.

For gradient-based optimization, we use reparameterization gradients BID9 .

We describe them next.

We provide details for the gradients.

Let λ wm parameterize q(w m ; λ wm ) and λ zn parameterize q(z n ; λ zn ).

We are interested in training the parameters λ wm , λ zn , θ , φ .

Subsample a SNP location m ∈ {1, . . .

, M }.

Draw a sample z n ∼ q(z n ; λ zn ) for n = 1, . . .

, N and w m ∼ q(w m ; λ wm ) for m = 1, . . .

, M , where the samples are reparameterizable (see BID9 for details).The gradient with respect to parameters λ zn is unbiasedly estimated by DISPLAYFORM0 This gradient scales linearly with the number of SNPs M .

This is undesirable as the number of SNPs ranges from the hundreds of thousands to millions.

We prevent this scaling by observing that for large M , the information in x n,1:M will influence the posterior far more than the single scalar y n .

In math, p(z n | x n,1:M , y n , θ, φ, w 1:M ) ≈ p(z n | x n,1:M , θ, φ, w 1:M ).

This is a tacit assumption in all GWAS methods that adjust for the confounder BID19 BID7 BID1 Kang et al., 2010) .The gradient with respect to parameters λ zn simplifies to DISPLAYFORM1 which scales to massive numbers of SNPs.

The gradients with respect to parameters λ wm and φ are unbiasedly estimated by DISPLAYFORM2 Note how none of this depends on the trait y or trait parameters θ.

We can thus perform inference to first approximate the posterior p(z 1:N , w 1:M , φ | x, y).

In a second stage, we can then perform inference to approximate the posterior p(θ | z 1:N , w 1:M , x, y).

The computational savings is significant not only within task but across tasks: when modelling many traits of interest (for example, § 5.2), inference over the SNP confounders only needs to be done once and can be re-used.

We perform stochastic gradient ascent using these gradients to maximize the variational objective.

C.3 SECOND STAGE: LEARNING THE TRAIT Above we described the first stage of an algorithm which performs stochastic gradient ascent to optimize parameters so that q(z n )q(w 1:M )q(θ ) ≈ p(z 1:N , w 1:M , φ | x, y).

Given these parameters, we are interested in training θ .

Dropping constants with respect to θ in the objective, we have DISPLAYFORM3 We maximize this objective using stochastic gradients with a single sample z n ∼ q(z n ), DISPLAYFORM4 This corresponds to Monte Carlo EM.

Its primary computation per-iteration is the backward pass of the trait's neural network.

Unlike in the first stage, we do not subsample SNPs as the likelihood depends on all SNPs.

In general, the density of y n is intractable: we exploit its tractable density if y n is discrete (it induces a categorical likelihood); otherwise for real-valued traits, we perform likelihood-free inference with respect to y n .

Following LFVI BID15 , define q(y) to be the empirical distribution over observed data {y n }.

Then subtract it as a constant to the objective, so DISPLAYFORM0 We approximate this log-ratio with a ratio estimator, r(y n , x n,1:M , z n , θ ; λ r ).

It is a function of all inputs in the log-ratio and is parameterized by λ r .We train the ratio estimator by minimizing a loss function with respect to its parameters, DISPLAYFORM1 The global minima of this objective with respect to the ratio estimator is the desired log-ratio, r * (y n , x n,1:M , z n , θ ) = log p(y n | x n,1:M , z n , θ ) − log q(y n ).Unfortunately, the ratio estimator has inputs of many dimensions.

In particular, it has the problematic property of scaling with the number of SNPs, which can be on the order of hundreds of thousands.

We can efficiently parameterize the ratio estimator by studying two extreme cases with respect to computational efficiency and statistical efficiency.

In one extreme, suppose y n has a tractable Gaussian density with mean given by the outcome model's neural network and unit variance (that is, the neural net is parameterized to apply a location-shift on the noise input, y n = NN(·) + n ).

Up to additive and multiplicative constants, the optimal log-ratio is DISPLAYFORM2 This implies the ratio estimator must relearn the neural network's forward pass in order to estimate the optimal log-ratio.

This is computationally redundant and can lead to unstable training.

On another extreme, suppose we parameterize r as DISPLAYFORM3 This dramatically reduces r's input dimensions, from hundreds of thousands to just two.

However, while computationally efficient, this is a poor statistical approximation: there is only a single dimension to preserve information about x n,1:M , z n , n , and θ relevant to y n ; this dimension is lossy for even Gaussian densities.

As a middleground, we use the neural net's first hidden layer as input into the ratio estimator, r(y n , x n,1:M , z n , θ ; λ r ) = r(y n , h n ; λ r ).This reduces the ratio estimator's inputs to be the trait y n and first hidden layer's units h n .

This hidden layer has much fewer dimensions than the raw inputs, such as 32 units (making it computationally efficient).

Moreover, under the data processing inequality (Cover & Thomas, 1991) , h n preserves more information relevant to y n than subsequent layers of the neural network (making it statistically efficient).

For all experiments, we parameterized r with two fully connected hidden layers with equal number of hidden units.

The gradient with respect to parameters θ is estimated by DISPLAYFORM4 This substitutes in the ratio estimator as a proxy to the intractable likelihood.

We minimize the auxiliary loss function in order to train the ratio estimator.

Sample y n ∼ p(y n | x n,1:M , z n , θ ) and subsample a data point y n ∼ q(y n ).

The gradient is estimated by ∇ λr · ≈ ∇ λr − log σ(r(y n , h n ; λ r )) − log(1 − σ(r(y n , h n ; λ r ))) .This corresponds to maximum likelihood, balanced with an adversarial objective to estimate the likelihood, and is relatively fast.

We perform stochastic gradient ascent, alternating between these two sets of gradients.

We provide more detail to § 5.1.

Implicit causal models can not only represent many causal structures but, more importantly, learn them from data.

To demonstrate this, we simulate data from a comprehensive collection of popular models in GWAS and analyze how well the fitted model can capture them.

These configurations exactly follow Hao et al. (2016) with same hyperparameters, which we describe below.

For each of the 11 simulation configurations, we generate 100 independent data sets.

Each data set consists of a M × N matrix of genotypes X and vector of N traits y.

Each individual n has M SNPs and one trait.

To simulate the M × N matrix of genotypes X, we draw x mn ∼ Binomial(2, π mn ) for m = 1, . . .

, M SNPs and n = 1, . . . , N individuals.

The probabilities π mn can be encoded under a real-valued M × N matrix of allele frequencies F where π mn = [logit(F )] mn .Many models in GWAS can be described under the factorization F = ΓS, where Γ is a M ×K matrix and S is a K ×N matrix for a fixed rank K ≤ N .

This includes principal components analysis BID7 , the Balding-Nichols model BID2 , and the Pritchard-StephensDonnelly model BID8 .

The M × K matrix Γ describes how structure manifests in the allele frequencies across SNPs.

The K × N matrix S encapsulates the genetic population structure across individuals.

We describe how we form Γ and S for each of the 11 simulation configurations.

Balding-Nichols Model (BN) + HapMap.

The BN model describes individuals according to a discrete mixture of ancestral subpopulations BID2 .

The HapMap data set was collected from three discrete populations (Gibbs et al., 2003) , which allows us to populate each row m of Γ with three i.i.d.

draws from the Balding-Nichols model: DISPLAYFORM0 , where k ∈ {1, 2, 3}. Each Γ mk is interpreted to be the allele frequency for subpopulation k at SNP m. The pairs (p m , F m ) are computed by randomly selecting a SNP in the HapMap data set, calculating its observed allele frequency, and estimating its F ST value using the estimator of BID17 .

The columns of S are populated with indicator vectors such that each individual is assigned to one of the three subpopulations.

The subpopulation assignments are drawn independently with probabilities 60/210, 60/210, and 90/210, which reflect the subpopulation proportions in the HapMap data set.

The simulated data has M = 100, 000 SNPs and N = 5000 individuals.

TGP is a project that comprehensively catalogs human genetic variation by producing complete genome sequences of well over 1000 individuals of diverse ancestries BID3 .

To form Γ, we sample Γ mk ∼ 0.9 Uniform(0, 1/2) for k = 1, 2 and set Γ m3 = 0.05.

To form S, we compute the first two principal components of the TGP genotype matrix after mean centering each SNP.

We then transform each principal component to be between (0, 1) and set the first two rows of S to be the transformed principal components.

The third row of S is set to 1 as an intercept.

The simulated data has M = 100, 000 SNPs and N = 1500 individuals (the total number of individuals in the TGP data set).Human Genome Diversity Project (HGDP).

HGDP is an international project that has genotyped a large collection of DNA samples from individuals distributed around the world, aiming to assess worldwide genetic diversity at the genomic level BID10 .

We followed the same scheme as for TGP above.

The simulated data has M = 100, 000 SNPs and N = 940 individuals (the total number of individuals in the HGDP data set).Pritchard-Stephens-Donnelly (PSD) + HGDP.

The PSD model describes individuals according to an admixture of ancestral subpopulations BID8 .

The rows of Γ are drawn from three i.i.d.

draws from the Balding-Nichols model: Γ mk ∼ BN(p m , F m ), where k ∈ {1, 2, 3}. The pairs (p m , F m ) are computed by randomly selecting a SNP in the HGDP data set, calculating its observed allele frequency, and estimating its F ST value using the estimator of BID17 .

The estimator requires each individual to be assigned to a subpopulation, which are made according to the K = 5 subpopulations from the analysis in BID10 .

The columns of S are sampled (s 1n , s 2n , s 3n ) ∼ Dirichlet(α = (a, a, a)) for n = 1, . . .

, N .

We apply four PSD configurations with hyperparameter settings of a = 0.01, 0.1, 0.5, 1.

Varying a from 1 to 0 varies the level of sparsity as individuals are placed from uniformly to corners of the simplex.

The simulated data has M = 100, 000 SNPs and N = 5000 individuals.

Spatial.

In this setting, we simulate genotypes such that the population structure relates to the spatial position of individuals.

The matrix Γ is populated by sampling Γ mk ∼ 0.9 Uniform(0, 1/2) for k = 1, 2 and setting Γ m3 = 0.05.

The first two rows of S correspond to coordinates for each individual on the unit square and are set to be independent and identically distributed samples from Beta(a, a), while the third row of S is set to 1 as an intercept.

We apply four spatial configurations with hyperparameter settings of a = 0.01, 0.1, 0.5, 1.

As with the Dirichlet distribution in the PSD model, varying a from 1 to 0 varies the level of sparsity as individuals are placed from uniformly to corners of the unit square.

The simulated data has M = 100, 000 SNPs and N = 5000 individuals.

To simulate traits y, we simulate from a linear model: for each individual n's SNPs {x mn }, DISPLAYFORM0 β m x mn + λ n + n , n ∼ Normal(0, σThe p-value threshold is fixed to t = 0.0025 across all methods BID12 .

To calculate the number of observed positives, we count the number of p-values for that are less than or equal to t. The true positives are the subset of p-values associated with causal SNPs; false positives are those associated with null SNPs.

Spurious associations occur when p-values corresponding to null SNPs are artificially small.

Namely, false positives are spurious associations.

In general, we expect there to be m 0 × t false positives among the m 0 p-values corresponding to null SNPs; in our setting, this corresponds to (100, 000 − 10) · 0.0025 ≈ 250 SNPs.

A method properly accounts for structure when the average excess is no more than this number.

Our precision count involved only the number of false positives higher than this calculation (which depends on the number of SNPs in that setting).To specify the implicit causal model in our experiments, we set the latent dimension of confounders equal to 3 or 5.

We use 512 units in both hidden layers of the SNP neural network and use 32 and 256 units for the trait neural network's first and second hidden layers respectively.

We provide more detail to § 5.2.

The data was obtained from the database of Genotypes and Phenotypes (dbGaP) (phs000276.v1.p1).

We follow the same preprocessing as BID12 , Individuals were filtered for completeness (maximum 1% missing genotypes) and pregnancy.(Pregnant women were excluded because we did not receive IRB approval for these individuals.)

SNPs were first filtered for completeness (maximum 5% missing genotypes) and minor allele frequency (minimum 1% minor allele frequency), then tested for Hardy-Weinberg equilibrium (p-value < 1/328348).

The final dimensions of the genotype matrix are m = 324, 160 SNPs and n = 5027 individuals.

A Box-Cox transform was applied to each trait, where the parameter was chosen such that the values in the median 95% value of the trait was as close to the normal distribution as possible.

Indicators for sex, oral contraception, and fasting status were added as adjustment variables.

For glucose, the individual with the minimum value was removed from the analysis as an extreme outlier.

No additional changes were made to the data.

After fitting each model, we follow the same procedure as in the simulation study for predicting causal factors.

We set the p-value threshold to be the genome-wide threshold of 7.2×10 −8 following Kang et al. (2010) .

<|TLDR|>

@highlight

Implicit models applied to causality and genetics

@highlight

The authors propose to use the implicit model to tackle Genome-Wide Association problem.

@highlight

This paper proposes solutions for the problems in genome-wide association studies of confounding due to population structure and the potential presence of non-linear interactions between different parts of the genome, and bridges statistical genetics and ML.

@highlight

Presents a non-linear generative model for GWAS that models population structure where non-linearities are modeled using neural networks as non-linear function approximators and inference is performed using likelihood-free variational inference