We establish a theoretical link between evolutionary algorithms and variational parameter optimization of probabilistic generative models with binary hidden variables.

While the novel approach is independent of the actual generative model, here we use two such models to investigate its applicability and scalability: a noisy-OR Bayes Net (as a standard example of binary data) and Binary Sparse Coding (as a model for continuous data).



Learning of probabilistic generative models is first formulated as approximate maximum likelihood optimization using variational expectation maximization (EM).

We choose truncated posteriors as variational distributions in which discrete latent states serve as variational parameters.

In the variational E-step, the latent states are then   optimized according to a tractable free-energy objective.

Given a data point, we can show that evolutionary algorithms can be used for the variational optimization loop by (A)~considering the bit-vectors of the latent states as genomes of individuals, and by (B)~defining the fitness of the individuals as the (log) joint probabilities given by the used generative model.



As a proof of concept, we apply the novel evolutionary EM approach to the optimization of the parameters of noisy-OR Bayes nets and binary sparse coding on artificial and real data (natural image patches).

Using point mutations and single-point cross-over for the evolutionary algorithm, we find that scalable variational EM algorithms are obtained which efficiently improve the data likelihood.

In general we believe that, with the link established here, standard as well as recent results in the field of evolutionary optimization can be leveraged to address the difficult problem of parameter optimization in generative models.

Evolutionary algorithms (EA) have been introduced (e.g. BID3 BID23 as a technique for function optimization using methods inspired by biological evolutionary processes such as mutation, recombination, and selection.

As such EAs are of interest as tools to solve Machine Learning problems, and they have been frequently applied to a number of tasks such as clustering BID21 BID9 , reinforcement learning BID25 , and hierarchical unsupervised BID16 or deep supervised learning (e.g., BID32 BID32 BID33 BID22 for recent examples).

In some of these tasks EAs have been investigated as alternatives to standard procedures BID9 ), but most frequently EAs are used to solve specific sub-problems.

For example, for classification with Deep Neural Networks (DNNs LeCun et al., 2015; BID27 , EAs are frequently applied to solve the sub-problem of selecting the best DNN architectures for a given task (e.g. BID32 BID33 or more generally to find the best hyper-parameters of a DNN (e.g. BID12 BID22 .Inspired by these previous contributions, we here ask if EAs and learning algorithms can be linked more tightly.

To address this question we make use of the theoretical framework of probabilistic generative models and expectation maximization (EM Dempster et al., 1977) approaches for parameter optimization.

The probabilistic approach in combination with EM is appealing as it establishes a very general unifying framework able to encompass diverse algorithms from clustering and dimensionality reduction BID24 BID34 over feature learning and sparse coding BID18 to deep learning approaches BID20 .

However, for most generative data models, EM is computationally intractable and requires approximations.

Variational EM is a very prominent such approximation and is continuously further developed to become more efficient, more accurate and more autonomously applicable.

Variational EM seeks to approximately solve optimization problems of functions with potentially many local optima in potentially very high dimensional spaces.

The key observation exploited in this study is that a variational EM algorithm can be formulated such that latent states serve as variational parameters.

If the latent states are then considered as genomes of individuals, EAs emerge as a very natural choice for optimization in the variational loop of EM.

A probabilistic generative model stochastically generates data points y using a set of hidden (or latent) variables s. The generative process can be formally expressed in the form of joint probability p( s, y | Θ), where Θ are the model parameters.

Given a set of N data points, y(1) , . . .

, y (N ) = y(1:N ) , learning seeks to change the parameters Θ so that the data generated by the generative model becomes as similar as possible to the N real data points.

One of the most popular approaches to achieve this goal is to seek maximum likelihood (ML) parameters Θ * , i.e., parameters that maximize the data log-likelihood for a given generative model: DISPLAYFORM0 To efficiently find (approximate) ML parameters we follow BID26 ; BID17 BID10 who reformulated the problem in terms of a maximization of a lower bound of the log-likelihood, the free energy F ( q, Θ).

Free energies are given by DISPLAYFORM1 where q (n) ( s) are variational distributions, and where H(q) denotes the entropy of a distribution q. For the purposes of this study, we consider elementary generative models which are difficult to train because of exponentially large state spaces.

These models serve well for illustrating the approach but we stress that any generative model which gives rise to a joint distribution p( s, y | Θ) can be trained with the approach discussed here as long as the latents s are binary.

In order to find approximate maximum likelihood solutions, distributions q (n) ( s) are sought that approximate the intractable posterior distributions p( s | y (n) , Θ) as well as possible, which results in the free-energy being as similar (or tight) as possible to the exact log-likelihood.

At the same time variational distributions have to result in tractable parameter updates.

Standard approaches include Gaussian variational distributions (e.g. BID19 or mean-field variational distributions BID10 .

If we denote the parameters of the variational distributions by Λ, then a variational EM algorithm consists of iteratively maximizing F(Λ, Θ) w.r.t.

Λ in the variational E-step and w.r.t.

Θ in the M-step.

The M-step can hereby maintain the same functional form as for exact EM but the expectation values now have to be computed w.r.t.

the variational distributions.

Instead of using parametric functions such as Gaussians or factored (mean-field) distributions, for our purposes we choose truncated variational distributions defined as a function of a finite set of states BID28 BID29 .

These states will later serve as populations of evolutionary algorithms.

If we denote K n a population of hidden states for a given data point y (n) , then variational distributions and their corresponding expectation values are given by (e.g. BID28 : DISPLAYFORM2 where δ( s ∈ K n ) is 1 if K n contains the hidden state s, zero otherwise.

If the set K n contains all states with significant posterior mass, then (3) approximates expectations w.r.t.

full posteriors very well.

By inserting truncated distributions as variational distribution of the free-energy (2), it can be shown BID13 that the free-energy takes a very compact simplified form given by: DISPLAYFORM3 As the variational parameters of the variational distribution (3) are now given by populations of hidden states, a variational E-step now consists of finding for each data point n the population K n that maximizes s∈K n p ( y n , s | Θ).

For the generative models considered here, each latent state s takes the form of a bit vector.

Hence, each population K n is a collection of bit vectors.

Because of the specific form (4), the free-energy is increased in the variational E-step if and only if we replace and individual s in population K DISPLAYFORM0 such that: DISPLAYFORM1 More generally, this means that the free energy is maximized in the variational E-step if we find for each n those S individuals with the largest joints p( s, y n | Θ), where p( s, y n | Θ) is given by the respective generative model (compare BID13 BID5 , for formal derivations).Full maximization of the free-energy is often a computationally much harder problem than increasing the free-energy; and in practice an increase is usually sufficient to finally approximately maximize the likelihood.

As we increase the free-energy by applying (5) we can choose any fitness function F ( s; y n , Θ) for an evolutionary optimization which fulfils the property: DISPLAYFORM2 Any mutations selected such that the fitness F ( s; y n , Θ) increases will result in provably increased free-energies.

Together with M-step optimizations of model parameters, the resulting variational EM algorithm will monotonously increase the free-energy.

The freedom in choosing a fitness function satisfying (6) leaves us free to pick a form that enables an efficient parent selection procedure.

More concretely (while acknowledging that other choices are possible) we define the fitness F ( s new ; y n , Θ) to be: DISPLAYFORM3 where logP is defined as the logarithm of the joint probability where summands that do not depend on the state s have been elided.

logP is usually more efficiently computable than the joint probabilities and has better numerical stability, while being a monotonously increasing function of the joints when the data-point y n is considered fixed.

As we will want to sample states proportionally to their fitness, an offset is applied to logP to make sure F always takes positive values.

As previously mentioned, other choices of F are possible as long as FORMULA6 holds.

From now on we will drop the argument y n or index n (while keeping in mind that an optimization is performed for each data point y n ).Our applied EAs then seek to optimize F ( s) for a population of individual K (we also drop the index n here).

More concretely, given the current population K of unique individuals s, the EA iteratively seeks a new set K with higher overall fitness.

For our models, s are bit-vectors of length H, and we usually require that populations K and K to have the same size as is customary for truncated approximations (e.g. BID29 .

Our example algorithm includes three common genetic operators, discussed in more detail below: parent selection, generation of children by single-point crossover and stochastic mutation of the children.

We repeat this process over N g generations in which subsequent iterations use the output of previous iterations as input population.

Parent Selection.

This step selects N p parents from the population K. Ideally, the selection procedure should be balanced between exploitation of parents with high fitness (which will more likely produce children with high fitness) and exploration of mutations of poor performing parents (which might eventually produce children with high fitness while increasing population diversity).

Diversity is crucial, as K is a set of unique individuals and therefore the improvement of the overall DISPLAYFORM4 Figure 1: Components of the genetic algorithm.fitness of the population depends on generating different children with high fitness.

In our numerical experiments we explored both fitness-proportional selection of parents (a classic strategy in which the probability of an individual being selected as a parent is proportional to its fitness) and random uniform selection of parents.

Crossover.

During the crossover step, random pairs of parents are selected; then each pair is assigned a number c from 1 to H − 1 with uniform probability (this is the single crossover point); finally the parents swap the last H − c bits to produce the offspring.

We denote N c the number of children generated in this way.

The crossover step can be skipped, making the EA more lightweight but decreasing variety in the offspring.

Mutation.

Finally, each of the N c children undergoes one or more random bitflips to further increase offspring diversity.

In our experiments we compare results of random uniform selection of the bits to flip with a more refined sparsity-driven bitflip algorithm.

This latter bitflip schemes assignes to 0's and 1's different probabilities of being flipped in order to produce children with a sparsity compatible with the one learned by the model.

In case the crossover step is skipped, a different bitflip mutation is performed on N c identical copies of each parent.

Algorithm 1: Evolutionary Expectation Maximization choose initial model parameters Θ and initial sets Krepeat for each data-point n do candidates = {} for g = 0 to N g do parents = select parents children = mutation(crossover(parents)) candidates = candidates ∪ children DISPLAYFORM5 update Θ using M-steps with (3) and K (n) until F has increased sufficiently A full run of the evolutionary algorithm therefore produces N g N c N p children (or new states s * ).

Finally we compute the union set of the original population K with all children and select the S fittest individuals of the union as the new population K .The EEM Algorithm.

We now have all elements required to formulate a learning algorithm with EAs as its integral part.

Alg.

1 summarizes the essential computational steps.

Note that this E-step can be trivially parallelized over data-points.

Finally, it is worth pointing out that algorithm 1, by construction, never decreases the free-energy.

We will use the EA formulated above as integral part of an unsupervised learning algorithm.

The objective of the learning algorithm is the optimization of the log-likelihood 1.

D denotes the number of observed variables, H the number of hidden units, and N the number of data points.

Noisy-OR.

The noisy-OR model is a highly non-linear bipartite data model with all-to-all connectivity among hidden and observable variables.

All variables take binary values.

The model assumes a Bernoulli prior for the latents, and active latents are then combined via the actual noisy-OR rule.

Section A of the appendix contains the explicit forms of the free energies and the M-step update rules for noisy-OR.

DISPLAYFORM0 Binary Sparse Coding.

As a second model and one for continuous data, we consider Binary Sparse Coding (BSC; BID6 .

BSC differs from standard Sparse Coding in its use of binary latent variables.

The latents are assumed to follow a univariate Bernoulli distribution which uses the same activation probability for each hidden unit.

The combination of the latents is described by a linear superposition rule.

Given the latents, the observables are independently and identically drawn from a Gaussian distribution: DISPLAYFORM1 The parameters of the model are Θ = (π, W, σ 2 ), where W is a D × H matrix whose columns contain the weights associated with each hidden unit s h and where σ 2 determines the variance of the Gaussian.

M-step update rules for BSC can be derived in close-form by optimizing the free energy (2) wrt.

all model parameters (compare, e.g., BID6 .

We report the final expressions in appendix B.

We describe numerical experiments performed to test the applicability and scalability of EEM.

Throughout the section, the different evolutionary algorithms are named by indicating which parent selection procedure was used ("fitparents" for fitness-proportional selection, "randparents" for random uniform selection) and which bitflip algorithm ("sparseflips" or "randflips").

We add "cross" to the name of the EA when crossover was employed.

First we investigate EMM using artificial data where the ground-truth components are known.

We use the bars test as a standard setup for such purposes BID4 BID7 BID15 .

In the standard setup, H gen /2 non-overlapping vertical and H gen /2 non-overlapping horizontal bars act as components on D = H gen × H gen pixel images.

N images are then generated by first selecting each bar with probability π gen .

The bars are then superimposed according to the noisy-OR model (non-linear superposition) or according to the BSC model.

In the case of BSC Gaussian noise is then added.

Noisy-OR.

Let us start with the standard bars test which uses a non-linear superposition BID4 of 16 different bars BID30 BID15 , and a standard average crowdedness of two bars per images (π gen = 2 H gen ).

We apply EEM for noisy-OR using different configurations of the EA.

We use H = 16 generative fields.

As a performance metric we here employ reliability (compare, e.g., BID30 BID15 , i.e., the fraction of runs whose learned free energies are above a certain minimum threshold and which learn the full dictionary of bars as well as the correct values for the prior probabilities π.fi tp a re n ts -c ro s s -r a n d fl ip s fi tp a re n ts -c ro s s -s p a rs e fl ip s ra n d p a re n ts -c ro s s -s p a rs e fl ip s ra n d p a re n ts -r a n d fl ip s fi tp a re n ts - FIG0 shows reliabilities over 10 different runs for each of the EAs.

On 8x8 images the more exploitative nature of "fitparents-sparseflips" is advantageous over the simpler and more explorative "randparents-randflips".

Note that this is not necessarily true for lower dimensionalities or otherwise easier-to-explore state spaces, in which also a naive random search might quickly find high-fitness individuals.

In this test the addition of crossover reduces the probability of finding all bars and leads to an overestimation of the crowdedness πH.After the initial verification on a standard bars test, we now make the component extraction problem more difficult by increasing overlap among the bars.

A highly non-linear generative model such as noisy-OR is a good candidate to model occlusion effects in images.

FIG1 shows the results of training noisy-OR with EEM on a bars data-set in which the latent causes have sensible overlaps.

The test parameters were chosen to be equal to those in BID15 FIG6 ).

After applying EEM with noisy-OR (H = 32) to N = 400 images with 16 strongly overlapping bars, we observed that all H gen = 16 bars were recovered in 13 of 25 runs, which is competitive especially when keeping in mind that no additional assumptions (e.g., compared to other models applied to this test) are used by EEM for noisy-OR.

BSC.

Like for the non-linear generative model, we first evaluate EEM for the linear BSC model on a bars test.

For BSC, the bars are superimposed linearly BID6 , which makes the problem easier.

As a consequence, standard bars test were solved with very high reliability using EEM for BSC even if merely random bitflips were used for the EA.

In order to make the task more challenging, we therefore (A) increased the dimensionality of the data to D = 10 × 10 bars images, (B) increased the number of components to H gen = 20, and (C) increased the average number of bars per data point from two (the standard setting) to five.

We employed N = 5, 000 training data points and tested the same five different configurations of the EA as were evaluated for noisy-OR.

We set the number of hidden units to H = H gen = 20 and used S = 120 variational states.

Per data point and per iteration, in total 112 new states (N p = 8, N c = 7, N g = 2) were sampled to vary K n .

Per configuration of the EA, we performed 20 independent runs, each with 300 iterations.

The results of the experiment are depicted in Fig. 5 .

We observe that a basic approach such as random uniform selection of parents and random uniform bitflips for the EA works well.

However, more sophisticated EAs improve performance.

For instance, combining bitflips with crossover and selecting parents proportionally to their fitness shows to be very benefical.

The results also show that sparseness-driven bitflips lead generally to very poor performance, even if crossover or fitness-fi tp a re n ts -s p a rs e fl ip s fi tp a re n ts -c ro ss -s p a rs e fl ip s ra n d p a re n ts -c ro ss -s p a rs e fl ip s ra n d p a re n ts -r a n d proportional selection of the parents is included.

This effect may be explained with the initialization of K n .

The initial states are drawn from a Bernoulli distribution with parameter 1 H which makes it more difficult for sparseness-driven EAs to explore and find solutions with higher crowdedness.

FIG5 in appendix C depicts the averaged free energy values for this experiment.

Next, we verify the approach on natural data.

We use patches of natural images, which are known to have a multi-component structure, which are well investigated, and for which typically models with high-dimensional latent spaces are applied.

The image patches used are extracted from the van Hateren image database BID35 .Noisy-OR.

First we consider raw images patches, i.e., images without substantial pre-processing which directly reflect light intensities.

Such image patches were generated by extracting random square subsections of a single 255x255 image of overlapping grass wires (part of image 2338 of the database).

We removed the brightest 1% pixels from the data-set, scaled each data-point to have gray-scale values in the range [0, 1] and then created data points with binary entries by repeatedly choosing a random gray-scale image and sampling binary pixels from a Bernoulli distribution with parameter equal to the gray-scale value of the original pixel (cfr.

FIG3 ).

Note that components in such light-intensity images can be expected to superimpose non-linearly because of occlusion, which motivates the application of a non-linear generative model such as noisy-OR.

We employ the "fitparents-sparseflips" evolutionary algorithm that was shown to perform best on artificial data (3).

Parameters were H = 100, S = 120, N g = 2, N p = 8, N c = 7.

FIG3 shows the generative fields learned over 200 iterations.

EEM allows learning of generative fields resembling curved edges, in line with expectations and with the results obtained in BID15 .

BSC.

Finally, we consider pre-processed image patches using common whitening approaches as they are customary for sparse coding approaches BID18 .

We use N = 100, 000 patches of size D = 16 × 16, randomly picked from the whole data set.

The highest 2 % of the amplitudes were clamped to compensate for light reflections and patches without significant structure were excluded for learning.

ZCA whitening BID0 was applied retaining 95 % of the variance (we used the procedure of a recent paper BID2 .

We trained the BSC model for 4,000 iterations using the "fitparents-cross-sparseflips" EA and employing H = 300 hidden units and S = 200 variational states.

Per data point and per iteration, in total 360 new states (N p = 10, N c = 9, N g = 4) were sampled to vary K n .

The results of the experiment are depicted in FIG4 .

The obtained generative fields primarily take the form of Gabor functions with different locations, orientations, phase, and spatial frequencies.

This is a typical outcome of sparse coding being applied to images.

On average more than five units were activated per data point showing that the learned code makes use of the generative model's multiple causes structure.

The generative fields converged faster than prior and noise parameters (similar effects are known from probabilistic PCA for the variance parameter).

The finit slope of the free-energy after 4000 iterations is presumably due to these parameters still changing slowly.

The training of generative models is a very intensively studied branch of Machine Learning.

If EM is applied for training, most non-elementary models require approximations.

For this reason, sophisticated and mathematically grounded approaches such as sampling or variational EM have been developed in order to derive sufficiently precise and efficient learning algorithms.

Evolutionary algorithms (EAs) have also been applied in conjunction with EM.

BID21 , for instance, have used EAs for clustering with Gaussian mixture models (GMMs).

However, the GMM parameters are updated by their approach relatively conventionally using EM, while EAs are used to select the best GMM models for the clustering problem (using a min.

description length criterion).

Such a use of EAs is similar to DNN optimization where EAs optimize DNN hyperparameters in an outer optimization loop BID32 BID12 BID22 Suganuma et al., 2017, etc) , while the DNNs themselves are optimized using standard error-minimization algorithms.

Still other approaches have used EAs to directly optimize, e.g., a clustering objective.

But in these cases EAs replace EM approaches for optimization (compare Hruschka et al., 2009) .

In contrast to all such previous applications, we have here shown that EAs and EM can be combined directly and intimately: Alg.

1 defines EAs as an integral part of EM, and as such EAs address the key optimization problem arising in the training of generative models.

We see the main contribution of our study in the establishment of this close theoretical link between EAs and EM.

This novel link will make it possible to leverage an extensive body of knowledge and experience from the community of evolutionary approaches for learning algorithms.

Our numerical experiments are a proof of concept which shows that EAs are indeed able to train generative models with large hidden spaces and local optima.

For this purpose we used very basic EAs with elementary selection, mutation, cross-over operators.

EAs more specialized to the specific optimization problems arising in the training of generative models have great potentials in future improvements of accuracy and scalability, we believe.

In our experiments, we have only just started to exploit the abilities of EAs for learning algorithms.

Still, our results represent, to the knowledge of the authors, the first examples of noisy-OR or sparse coding models trained with EAs (although both models have been studied very extensively before).

Most importantly, we have pointed out a novel mathematically grounded way how EAs can be used for generative models with binary latents in general.

The approach here established is, moreover, not only very generically formulated using the models' joint probabilities but it is also very straightforward to apply.

The truncated free energy takes on the following form for Noisy-OR: DISPLAYFORM0 The M-step equations for noisy-OR are obtained by taking derivatives of the free energy, equating them to zero and solving the resulting set of equations.

We report the results here for completeness: DISPLAYFORM1 where DISPLAYFORM2 The update rule for π is quite straightforward.

The update equations for the weights W dh , on the other hand, do not allow a closed form solution (i.e. no exact M-step equation can be derived).

The rule presented here, instead, expresses each W new dh as a function of all current W ; this is a fixedpoint equation whose fixed point would be the exact solution of the maximization step.

Rather than solving the equation numerically at each step of the learning algorithm, we exploit the fact that in practice one single evaluation of 13 is enough to (noisily, not optimally) move towards convergence.

Since TV-EM is guaranteed to never decrease F, drops of the free-energy during training can only be ascribed to this fixed-point equation; this provides a simple mechanism to check and possibly correct for misbehaviors of 13 if needed.

The free energy for BSC follows from inserting (10) into (2).

Update rules can be obtained by optimizing the resulting expression separately for the model parameters π, σ 2 and W (compare, e.g., BID6 .

For the sake of completeness, we show the result here: DISPLAYFORM0 DISPLAYFORM1 Exact EM can be obtained by setting q n to the exact posterior p( s | y (n) , Θ).

As this quickly becomes computational intractable with higher latent dimensionality, we approximate exact posteriors by truncated variational distributions (3).

For BSC, the truncated free energy (4) takes the form DISPLAYFORM2 where DISPLAYFORM3

When performing sparsity-driven bitflips, we flip each bit of a particular child s * with probability p 0 if it is 0, with probability p 1 otherwise.

We call p bf the average probability of flipping any bit in s * .

We impose the following constraints on p 0 and p 1 :• p 1 = αp 0 for some constant α • the average number of on bits after mutation is set at s which yield the following expressions for p 0 and p 1 : DISPLAYFORM0 Trivially, random uniform bitflips correspond to the case p 0 = p 1 = p bf .

With respect to the tests shown in FIG1 and discussed in section 5.1, it is worth to spend a few more words on comparisons with the other algorithms shown (Lücke & Sahani, 2008, Fig. 9 ).

Quantitative comparison to NMF approaches, neural nets (DI BID31 , and MCA BID15 shows that EMM for noisy-OR performs well but there are also approaches with higher reliability.

Of all the approaches which recover more than 15 bars on average, most require additional assumptions.

E.g., all NMF approaches, non-negative sparse coding BID8 and R-MCA 2 require constraints on weights and/or latent activations.

Only MCA 3 does not require constraints and presumably neither DI.

DI is a neural network approach, which makes the used assumptions difficult to infer.

MCA 3 is a generative model with a max-non-linearity as superposition model.

For learning it explores all sparse combinations with up to 3 components.

Applied with H = 32 latents, it hence evaluates more than 60000 states per data point per iteration for learning.

For comparison, EEM for noisy-OR evaluates on the order of S = 100 states per data point per iteration.

Figure 10 : Generative fields learned running EEM for noisy-OR ("fitparents-sparseflips") for 175 iterations with H = 200 latent variables.

Learned crowdedness πH was 1.6.

@highlight

We present Evolutionary EM as a novel algorithm for unsupervised training of generative models with binary latent variables that intimately connects variational EM with evolutionary optimization

@highlight

The paper presents a combination of evolutionary computation and variational EM for models with binary latent variables represented via a particle-based approximation

@highlight

The paper makes an attempt to tightly integrate expectation-maximization training algorithms with evolutionary algorithms.