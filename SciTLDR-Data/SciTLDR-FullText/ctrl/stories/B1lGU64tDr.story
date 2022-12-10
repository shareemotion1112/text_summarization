Real-world dynamical systems often consist of multiple stochastic subsystems that interact with each other.

Modeling and forecasting the behavior of such dynamics are generally not easy, due to the inherent hardness in understanding the complicated interactions and evolutions of their constituents.

This paper introduces the relational state-space model (R-SSM), a sequential hierarchical latent variable model that makes use of graph neural networks (GNNs) to simulate the joint state transitions of multiple correlated objects.

By letting GNNs cooperate with SSM, R-SSM provides a flexible way to incorporate relational information into the modeling of multi-object dynamics.

We further suggest augmenting the model with normalizing flows instantiated for vertex-indexed random variables and propose two auxiliary contrastive objectives to facilitate the learning.

The utility of R-SSM is empirically evaluated on synthetic and real time series datasets.

Many real-world dynamical systems can be decomposed into smaller interacting subsystems if we take a fine-grained view.

For example, the trajectories of coupled particles are co-determined by perparticle physical properties (e.g., mass and velocity) and their physical interactions (e.g., gravity); traffic flow can be viewed as the coevolution of a large number of vehicle dynamics.

Models that are able to better capture the complex behavior of such multi-object systems are of wide interest to various communities, e.g., physics, ecology, biology, geoscience, and finance.

State-space models (SSMs) are a wide class of sequential latent variable models (LVMs) that serve as workhorses for the analysis of dynamical systems and sequence data.

Although SSMs are traditionally designed under the guidance of domain-specific knowledge or tractability consideration, recently introduced deep SSMs (Fraccaro, 2018) use neural networks (NNs) to parameterize flexible state transitions and emissions, achieving much higher expressivity.

To develop deep SSMs for multi-object systems, graph neural networks (GNNs) emerge to be a promising choice, as they have been shown to be fundamental NN building blocks that can impose relational inductive bias explicitly and model complex interactions effectively .

Recent works that advocate GNNs for modeling multi-object dynamics mostly make use of GNNs in an autoregressive (AR) fashion.

AR models based on recurrent (G)NNs can be viewed as special instantiations of SSMs in which the state transitions are restricted to being deterministic (Fraccaro, 2018, Section 4.2) .

Despite their simplicity, it has been pointed out that their modeling capability is bottlenecked by the deterministic state transitions (Chung et al., 2015; Fraccaro et al., 2016) and the oversimplified observation distributions (Yang et al., 2018) .

In this study, we make the following contributions: (i) We propose the relational state-space model (R-SSM), a novel hierarchical deep SSM that simulates the stochastic state transitions of interacting objects with GNNs, extending GNN-based dynamics modeling to challenging stochastic multi-object systems. (ii) We suggest using the graph normalizing flow (GNF) to construct expressive joint state distributions for R-SSM, further enhancing its ability to capture the joint evolutions of correlated stochastic subsystems. (iii) We develop structured posterior approximation to learn R-SSM using variational inference and introduce two auxiliary training objectives to facilitate the learning.

Our experiments on synthetic and real-world time series datasets show that R-SSM achieves competitive test likelihood and good prediction performance in comparison to GNN-based AR models and other sequential LVMs.

The remainder of this paper is organized as follows: Section 2 briefly reviews neccesary preliminaries.

Section 3 introduces R-SSM formally and presents the methods to learn R-SSM from observations.

Related work is summarized in Section 4 and experimental evaluation is presented in Section 5.

We conclude the paper in Section 6.

In this work, an attributed directed graph is given by a 4-tuple: G = (V, E, V, E), where V = [N ] := {1, . . .

, N } is the set of vertices, E ⊆ [N ] × [N ] is the set of edges, V ∈ R N ×dv is a matrix of static vertex attributes, and E ∈ R N ×N ×de is a sparse tensor storing the static edge attributes.

The set of direct predecessors of vertex i is notated as

We use the notation x i to refer to the i-th row of matrix X and write x ij to indicate the (i, j)-th entry of tensor X (if the corresponding matrix or tensor appears in the context).

For sequences, we write x ≤t = x 1:t := (x 1 , . . .

, x t ) and switch to x (i) t for referring to the i-th row of matrix X t .

GNNs are a class of neural networks developed to process graph-structured data and support relational reasoning.

Here we focus on vertex-centric GNNs that iteratively update the vertex representations of a graph G while being equivariant (Maron et al., 2019) When updating the representation of vertex i from h i to h ′ i , a GNN takes the representations of other nearby vertices into consideration.

Popular GNN variants achieve this through a multi-round message passing paradigm, in which the vertices repeatedly send messages to their neighbors, aggregate the messeages they received, and update their own representations accordingly.

Formally, the operations performed by a basic block of a message-passing GNN are defined as follows:

Throughout this work, we implement Equations (1) and (2) by adopting a multi-head attention mechanism similar to Vaswani et al. (2017) and Velikovi et al. (2018) .

For Equation (3), we use either a RNN cell or a residual block (He et al., 2016) , depending on whether the inputs to GNN are RNN states or not.

We write such a block as H ′ = MHA(G, g, H) and give its detailed implementation in the Appendix.

A GNN simply stacks L separately-parameterized MHA blocks and iteratively computes H =:

. .

, L. We write this construction as H ′ = GNN(G, g, H) and treat it as a black box to avoid notational clutter.

State-space models are widely applied to analyze dynamical systems whose true states are not directly observable.

Formally, an SSM assumes the dynamical system follows a latent state process {z t } t≥1 , which possibly depends on exogenous inputs {u t } t≥1 .

Parameterized by some (unknown) static parameter θ, the latent state process is characterized by an initial density z 1 ∼ π θ (·|u 1 ) and a transition density z t+1 ∼ f θ (·|z ≤t , u ≤t+1 ).

Moreover, at each time step, some noisy measurements of the latent state are observed through an observation density:

(c) Inference model.

x t ∼ g θ (·|z ≤t , u ≤t ) .

The joint density of x 1:T and z 1:T factors as:

The superior expressiveness of SSMs can be seen from the fact that the marginal predictive distribution p(x t |x <t , u ≤t ) = p(x t , z ≤t |x <t , u ≤t ) dz ≤t can be far more complex than unimodal distributions and their finite mixtures that are common in AR models.

Recently developed deep SSMs use RNNs to compress z ≤t (and u ≤t ) into fixed-size vectors to achieve tractability.

As shown in next section, R-SSM can be viewed as enabling multiple individual deep SSMs to communicate.

Normalizing flows (Rezende & Mohamed, 2015) are invertible transformations that have the capability to transform a simple probability density into a complex one (or vice versa).

Given two domains X ⊆ R D and Y ⊆ R D , let f : X → Y be an invertible mapping with inverse f −1 .

Applying f to a random variable z ∈ X with density p(z), by the change of variables rule, the resulting random variable z ′ = f (z) ∈ Y will have a density:

A series of invertible mappings with cheap-to-evaluate determinants can be chained together to achieve complex transformations while retaining efficient density calculation.

This provides a powerful way to construct expressive distributions.

Suppose there is a dynamical system that consists of multiple interacting objects, and observing this system at a specific time is accomplished by acquiring measurements from every individual object simultaneously.

We further assume these objects are homogeneous, i.e., they share the same measurement model, and leave systems whose constituents are nonhomogeneous for future work.

To generatively model a time-ordered series of observations collected from this system, the straightforward approach that builds an individual SSM for each object is usually unsatisfactory, as it simply assumes the state of each object evolves independently and ignores the interactions between objects.

To break such an independence assumption, our main idea is to let multiple individual SSMs interact through GNNs, which are expected to capture the joint state transitions of correlated objects well.

Given the observations for a multi-object dynamical system, our model further assumes its interaction structure is known as prior knowledge.

The interaction structure is provided as a directed graph, in which each object corresponds to a vertex, and a directed edge indicates that the state of its head is likely to be affected by its tail.

In situations where such graph structure is not available, a complete graph can be specified.

However, to model dynamical systems comprising a large number of objects, it is often beneficial to explicitly specify sparse graph structures, because they impose stronger relational inductive bias and help save the computational cost.

A relational state-space model assumes a set of correlated dynamical subsystems evolve jointly under the coordination of graph neural networks.

Formally, given a graph G = (V, E, V, E), in which an edge (i, j) ∈ E indicates that the state of vertex j may be affected by vertex i. Let u

t ∈ R dx be the input and observation for vertex i at time step t, respectively.

For T steps,

we introduce a set of unobserved random variables {z

t ∈ R dz represents the latent state of vertex i at time step t. Furthermore, we introduce a global latent variable z g t ∈ R dg for each time step to represent the global state shared by all vertices.

Conditioning on the graph and exogenous inputs, an R-SSM factorizes the joint density of observations and latent states as follows:

For notational simplicity, we switch to the matrix notation

on.

The joint transition density f θ is further factorized as a product of global transition density f g θ and local transition density f

. .) .

To instantiate these conditional distributions, a GNN accompanied by RNN cells is adopted to recurrently compress the past dependencies at each time step into fixed-size context vectors.

Specifically, the observations are assumed to be generated from following process: to be a diagonal Gaussian distribution whose mean and variance are parameterized by the output of a multilayer perceptron (MLP), and the local transition density f ⋆ θ will be discussed later.

The local observation distribution g θ can be freely selected in line with the data, and in our experiments it is either a Gaussian distribution or a mixture of logistic distributions parameterized by MLPs.

The graphical structure of two consecutive steps of the generating process is illustrated in Figure 1a .

An intuitive way to think about this generative model is to note that the N + 1 latent state processes interact through the GNN, which enables the new state of a vertex to depend on not only its own state trajectory but also the state trajectories of other vertices and the entire graph.

As illustrated in Figure 1b , writing Z t = (z g t , Z t ) and suppressing the dependencies on the graph G and exogenous inputs U 1:T , an R-SSM can be interpreted as an ordinary SSM in which the entire graph evolves as a whole, i.e., the joint density of latent states and observations factors as:

Given observations X 1:T , we are interested in learning unknown parameters θ and inferring unobserved states Z 1:T .

For the learning task we wish to maximize the marginal likelihood p θ (X 1:T ) = p θ (X 1:T , Z 1:T ) dZ 1:T , but in our case the integral is intractable.

We adopt a recently developed variational inference (VI) approach called variational sequential Monte Carlo (VSMC) (Maddison et al., 2017; Naesseth et al., 2018; Le et al., 2018) , which maximizes a variational lower bound on the log marginal likelihood instead and learns the proposal distributions for the inference task simultaneously.

Given a sequence of proposal distributions {q ϕ (Z t |Z <t , X ≤t )} T t=1 parameterized by ϕ, running the sequential Monte Carlo (SMC) algorithm with K particles yields an unbiased marginal likelihood

where w k t is the unnormalized importance weight of particle k at time t. The variational lower bound is obtained by applying the Jensen's inequality:

Assuming the proposal distributions are reparameterizable (Kingma & Welling, 2014) , we use the biased gradient estimator ∇L

Proposal design.

We make the proposal for Z t depend on the information up to time t and share some parameters with the generative model.

We also choose to factorize

. .) .

The proposal distributions for all time steps are structured as follows:

for i = 1, . . .

, N and t = 1, . . .

, T , where h g t and H 1:T are computed using the relevant parts of the generative model.

r g ϕ is specified to be a diagonal Gaussian parameterized by an MLP, and r ⋆ ϕ will be discussed soon.

Here B t can be interpreted as a belief state (Gregor et al., 2019) , which summarizes past observations X ≤t (and inputs U ≤t ) deterministically.

The graphical structure of this proposal design is shown in Figure 1c , and the detailed VSMC implementation using this proposal is given in Appendix A.4.

The local transition density f

, which assumes that the object states are conditionally independent, i.e., the joint state distribution is completely factorized over objects.

We believe that such an independence assumption is an oversimplification for situations where the joint state evolution is multimodal and highly correlated.

One possible way to introduce inter-object dependencies is modeling joint state distributions as Markov random fields (MRFs) (Naesseth et al., 2019) , but this will significantly complicate the learning.

Here we introduce the Graph Normalizing Flow (GNF) 1 , which adapts Glow (Kingma & Dhariwal, 2018) to graph settings and enables us to build expressive joint distributions for correlated random variables indexed by graph nodes.

As described earlier, the key ingredient for a flow is a series invertible mappings that are iteratively applied to the samples of a base distribution.

Now we are interested in the case where the samples are vertex states Z t , and thus the invertible mappings should be further constrained to be equivariant under vertex relabeling.

This rules out popular autoregressive flows, e.g., IAF (Kingma et al., 2016) and MAF (Papamakarios et al., 2017) .

Our GNF is built upon the coupling layer introduced in Dinh et al. (2017) , which provides a flexible framework to construct efficient invertible mappings.

A GNF coupling layer splits the input Z ∈ R N ×D into two parts, Z a ∈ R N ×d and

1 GNF has been independently developed by Liu et al. (2019) for different purpose.

where ⊙ denotes the element-wise product, and the functions s(·) and t(·) are specified to be GNNs to enforce the equivariance property.

A GNF combines a coupling layer with a trainable elementwise affine layer and an invertible 1×1 convolution layer (Hoogeboom et al., 2019) , organizing them as: Input → Affine → Coupling → Conv 1×1 → Output.

A visual illustration of this architecture is provided in Appendix A.5.

In order to obtain more expressive prior and variational posterior approximation, the local transition density and local proposal density can be constructed by stacking multiple GNFs on top of diagonal Gaussian distributions parameterized by MLPs.

With the message passing inside the coupling layers, GNFs can transform independent noise into correlated noise and thus increase model expressivity.

The 1 × 1 convolution layers free us from manually permuting the dimensions, and the element-wise affine layers enable us to tune their initial weights to stablize training.

In our initial experiments, we found that learning R-SSM suffered from the posterior collpase phenomenon, which is a well known problem in the training of variational autoencoders (VAEs).

It means that the variational posterior approximation q ϕ (Z t |Z <t , X ≤t ) degenerate into the prior f θ (Z t |Z <t ) in the early stage of optimization, making the training dynamics get stuck in undesirable local optima.

Besides, we also encountered a more subtle problem inherent in likelihood-based training of deep sequential models.

That is, for relatively smooth observations, the learned model tended to only capture short-term local correlations but not the interaction effects and long-term transition dynamics.

Motivated by recent advances in unsupervised representation learning based on mutual information maximization, in particular the Contrastive Predictive Coding (CPC) approach (Oord et al., 2018), we alleviate these problems by forcing the latent states to perform two auxiliary contrastive prediction tasks.

At each time step t, the future observations of each vertex i are summarized into a vector using a backward RNN: c

>t ).

Then we define two auxiliary CPC objectives:

t ), and Ω t,i is a set that contains c (i) t and some negative samples.

The expectation is over negative samples and the latent states sampled from the filtering distributions.

The positive score functions λ ψ,1 and λ ψ,2 are specified to be simple log-bilinear models.

Intuitively, L aux 1 encourages the latent states to encode useful information that helps distinguish the future summaries from negative samples.

L aux 2 encourages the deterministic states to reflect the interaction effects, as it contrastingly predicts the future summary of vertex i based on the states of i's neighbors only.

The negative samples are selected from the future summaries of other vertices within the minibatch.

The final objective to maximize is

2 , in which β 1 ≥ 0 and β 2 ≥ 0 are tunable hyperparameters.

The procedure to estimate this objective is described in Appendix A.4.

GNN-based dynamics modeling.

GNNs (Scarselli et al., 2009; Duvenaud et al., 2015; Defferrard et al., 2016; Gilmer et al., 2017; Hamilton et al., 2017; Velikovi et al., 2018; Xu et al., 2019; Maron et al., 2019) provide a promising framework to learn on graph-structured data and impose relational inductive bias in learning models.

We refer the reader to for a recent review.

GNNs (or neural message passing modules) are the core components of recently developed neural physics simulators (Battaglia et al., 2016; Watters et al., 2017; Chang et al., 2017; Janner et al., 2019; Mrowca et al., 2018; Li et al., 2019) and spatiotemporal or multi-agent dynamics models (Alahi et al., 2016; Hoshen, 2017; Zhang et al., 2018; Tacchetti et al., 2019; Chen et al., 2020) .

In these works, GNNs usually act autoregressively or be integrated into the sequence-to-sequence (seq2seq) framework (Sutskever et al., 2014) .

Besides, recently they have been combined with generative adversarial networks (Goodfellow et al., 2014) and normalizing flows for multi-agent forecasting (Gupta et al., 2018; Kosaraju et al., 2019; Rhinehart et al., 2019) .

R-SSM differs from all these works by introducing structured latent variables to represent the uncertainty on state transition and estimation.

GNNs in sequential LVMs.

A few recent works have combined GNNs with a sequential latent variable model, including R-NEM (van Steenkiste et al., 2018), NRI (Kipf et al., 2018) , SQAIR (Kosiorek et al., 2018) , VGRNN (Hajiramezanali et al., 2019) , MFP (Tang & Salakhutdinov, 2019) , and Graph VRNN (Sun et al., 2019; Yeh et al., 2019) .

The latent variables in R-NEM and NRI are discrete and represent membership relations and types of edges, respectively.

In contrast, the latent variables in our model are continuous and represent the states of objects.

SQAIR is also a deep SSM for multi-object dynamics, but the GNN is only used in its inference network.

VGRNN is focused on modeling the topological evolution of dynamical graphs.

MFP employs a conditional VAE architecture, in which the per-agent discrete latent variables are shared by all time steps.

The work most relevant to ours is Graph VRNN, in which the hidden states of per-agent VRNNs interact through GNNs.

Our work mainly differs from it by introducing a global latent state process to make the model hierarchical and exploring the use of normalizing flows as well as the auxiliary contrastive objectives.

More subtle differences are discussed in Section 5.2.

Deep LVMs for sequential data.

There has been growing interest in developing latent variable models for sequential data with neural networks as their building blocks, among which the works most relevant to ours are stochastic RNNs and deep SSMs.

Many works have proposed incorporating stochastic latent variables into vanilla RNNs to equip them with the ability to express more complex data distributions (Bayer & Osendorfer, 2014; Chung et al., 2015; Fraccaro et al., 2016; Goyal et al., 2017; Ke et al., 2019) or, from another perspective, developing deep SSMs by parameterizing flexible transition and emission distributions using neural networks (Krishnan et al., 2017; Fraccaro et al., 2017; Buesing et al., 2018; Zheng et al., 2017; Hafner et al., 2019) .

Approximate inference and parameter estimation methods for nonlinear SSMs have been extensively studied in the literature (Doucet & Johansen, 2009; Andrieu et al., 2010; Kantas et al., 2015; Gu et al., 2015; Karl et al., 2016; Marino et al., 2018; Gregor et al., 2019; Hirt & Dellaportas, 2019) .

We choose VSMC (Maddison et al., 2017; Naesseth et al., 2018; Le et al., 2018) as it combines the powers of VI and SMC.

The posterior collapse problem is commonly addressed by KL annealing, which does not work with VSMC.

The idea of using auxiliary costs to train deep SSMs has been explored in Z-forcing (Goyal et al., 2017; Ke et al., 2019) , which predicts the future summaries directly rather than contrastingly.

As a result, the backward RNN in Z-forcing may degenerate easily.

We implement R-SSM using the TensorFlow Probability library (Dillon et al., 2017) .

The experiments are organized as follows: In Section 5.1, we sample a toy dataset from a simple stochastic multi-object model and validate that R-SSM can fit it well while AR models and non-relational models may struggle.

In Section 5.2, R-SSM is compared with state-of-the-art sequential LVMs for multi-agent modeling on a basketball gameplay dataset, and the effectiveness of GNF is tested through ablation studies.

Finally, in Section 5.3, the prediction performance of R-SSM is compared with strong GNN-based seq2seq baselines on a road traffic dataset.

Due to the space constraint, the detailed model architecture and hyperparameter settings for each dataset are given in the Appendix.

Below, all values reported with error bars are averaged over 3 or 5 runs.

First we construct a simple toy dataset to illustrate the capability of R-SSM.

Each example in this dataset is generated by the following procedure: for i = 1, . . .

, N and t = 1, . . .

, T .

Here SBM is short for the symmetric stochastic block model, in which each vertex i belongs to exact one of the K communities, and two vertices i and j are connected with probability p 0 if they are in the same community, p 1 otherwise.

A vertex-specific covariate vector v i ∈ R dv is attached to each vertex i, and by Equation (6) ⊤ , σ x = σ z = 0.05, and ε = 2.5, we generate 10K examples for training, validation, and test, respectively.

A typical example is visualized in the Appendix.

Despite the simple generating process, the resulting dataset is highly challenging for common models to fit.

To show this, we compare R-SSM with several baselines, including (a) VAR: Fitting a first-order vector autoregression model for each example; (b) VRNN: A variational RNN (Chung et al., 2015) shared by all examples; (c) GNN-AR: A variant of the recurrent decoder of NRI (Kipf et al., 2018) , which is exactly a GNN-based AR model when given the ground-truth graph.

VAR and VRNN are given access to the observations {x

only, while GNN-AR and R-SSM are additionally given access to the graph structure (V, E) (but not the vertex covariates).

GNF is not used in R-SSM because the true joint transition distribution is factorized over vertices.

For each model, we calculate three metrics: (1) LL:

Average log-likelihood (or its lower bound) of test examples; (2) MSE: Average mean squared one-step prediction error given the first 75 time steps of each test example; (3) CP: Average coverage probability of a 90% one-step prediction interval.

For nonanalytic models, point predictions and prediction intervals are computed using 1000 Monte Carlo samples.

The results are reported in Table 1 .

The generating process involves latent factors and nonlinearities, so VAR performs poorly as expected.

VRNN largely underfits the data and struggles to generalize, which may be caused by the different topologies under the examples.

In contrast, GNN-AR and R-SSM generalize well as expected, while R-SSM achieves much higher test log-likelihood and produces good one-step probabilistic predictions.

This toy case illustrates the generalization ability of GNNs and suggests the importance of latent variables for capturing the uncertainty in stochastic multi-object systems.

We also observed that without L aux 1 the training dynamics easily get stuck in posterior collapse at the very early stage, and adding L aux 2 help improve the test likelihood.

In basketball gameplay, the trajectories of players and the ball are highly correlated and demonstrate rich, dynamic interations.

Here we compare R-SSM with a state-of-the-art hierarchical sequential LVM for multi-agent trajectories (Zhan et al., 2019) , in which the per-agent VRNNs are coordinated by a global "macro intent" model.

We note it as MI-VRNN.

The dataset 2 includes 107,146 training examples and 13,845 test examples, each of which contains the 2D trajectories of ten players and the ball recorded at 6Hz for 50 time steps.

Following their settings, we use the trajectories of offensive team only and preprocess the data in exactly the same way to make the results directly comparable.

The complete graph of players is used as the input to R-SSM.

Several ablation studies are performed to verify the utility of the proposed ideas.

In Table 3 , we report test likelihood bounds and the rollout quality evaluated with three heuristic statistics: average speed (feet/step), average distance traveled (feet), and the percentage of out-of-bound (OOB) time steps.

The VRNN baseline developed by Zhan et al. (2019) is also included for comparison.

Note that the VSMC bound L SMC 1000 is a tighter log-likelihood approximation than the ELBO (which is equivalent to L SMC 1 ).

The rollout statistics of R-SSMs are calculated from 150K 50-step rollouts with 10 burn-in steps.

Several selected rollouts are visualized in the Appendix.

Table 2 , all R-SSMs outperform the baselines in terms of average test log-likelihood.

Again, we observed that adding L aux 1 is necessary for training R-SSM successfully on this dataset.

Training with the proposed auxiliary loss L aux 2 and adding GNFs do improve the results.

R-SSM with 8 GNFs (4 in prior, 4 in proposal) achieves higher likelihood than R-SSM with 4 GNFs, indicating that increasing the expressivity of joint state distributions helps fit the data better.

As for the rollout quality, the OOB rate of the rollouts sampled from our model matches the ground-truth significantly better, while the other two statistics are comparable to the MI-VRNN baseline.

In Table 3 , we also provide preliminary results for the setting that additionally includes the trajectory of the ball.

This enables us to compare with the results reported by Yeh et al. (2019) for Graph VRNN (GVRNN).

The complete graph of ball and players served as input to R-SSM is annotated with two node types (player or ball) and three edge types (player-to-ball, ball-to-player or player-to-player) .

R-SSM achieves competitive test likelihood, and adding GNFs helps improve the performance.

We point out that several noticeable design choices of GVRNN may help it outperform R-SSM: (i) GVRNN uses a GNN-based observation model, while R-SSM uses a simple factorized observation model. (ii) GVRNN encodes X 1:t−1 into H t and thus enables the prior of Z t to depend on past observations, which is not the case in R-SSM. (iii) GVRNN uses several implementation tricks, e.g., predicting the changes in observations only (X t = X t−1 + ∆X t ) and passing raw observations as additional input to GNNs.

We would like to investigate the effect of these interesting differences in future work.

Traffic speed forecasting on road networks is an important but challenging task, as the traffic dynamics exhibit complex spatiotemporal interactions.

In this subsection, we demonstrate that R-SSM is comparable to the state-of-the-art GNN-based seq2seq baselines on a real-world traffic dataset.

The METR-LA dataset contains 4 months of 1D traffic speed measurements that were recorded via 207 sensors and aggregated into 5 minutes windows.

For this dataset, all conditional inputs G = (V, E, V, E) and U 1:T are provided to R-SSM, in which E is constructed by connecting two sensors if their road network distance is below a threshold, V stores the geographic positions and learnable embeddings of sensors, E stores the road network distances of edges, and U 1:T provides the time information (hour-of-day and day-of-week).

We impute the missing values for training and exclude them from evaluation.

GNF is not used because of GPU memory limitation.

Following the settings in , we train our model on small time windows spanning 2 hours and use a 7:1:2 split for training, validation, and test.

The comparison of mean absolute forecast errors (MAE) is reported in Table 4 .

The three forecast horizons correspond to 15, 30, and 60 minutes.

We give point predictions by taking the elementwise median of 2K Monte Carlo forecasts.

Compared with DCRNN and GaAN (Zhang et al., 2018) , R-SSM delivers comparable short-term forecasts but slightly worse long-term forecasts.

We argue that the results are admissible because: (i) By using MAE loss and scheduled sampling, the DCRNN and GaAN baselines are trained on the multi-step objective that they are later evaluated on, making them hard to beat. (ii) Some stochastic systems are inherently unpredictable beyond a few steps due to the process noise, e.g., the toy model in Section 5.1.

In such case, multi-step MAE may not be a reasonable metric, and probabistic forecasts may be prefered.

The average coverage probabilities (CP) of 90% prediction intervals reported in Table 4 indicate that R-SSM provides good uncertainty estimates. (iii) Improving the multi-step prediction ability of deep SSMs is still an open problem with a few recent attempts (Ke et al., 2019; Hafner et al., 2019) .

We would like to explore it in future work.

In this work, we present a deep hierarchical state-space model in which the state transitions of correlated objects are coordinated by graph neural networks.

To effectively learn the model from observation data, we develop a structured posterior approximation and propose two auxiliary contrastive prediction tasks to help the learning.

We further introduce the graph normalizing flow to enhance the expressiveness of the joint transition density and the posterior approximation.

The experiments show that our model can outperform or match the state-of-the-arts on several time series modeling tasks.

Directions for future work include testing the model on high-dimensional observations, extending the model to directly learn from visual data, and including discrete latent variables in the model.

c∈Ωt,i λ ψ,1 (ẑ

t,k ), and Ω t,i is a set that contains c The element-wise affine layer is proposed by Kingma & Dhariwal (2018) for normalizing the activations.

Its parameters γ ∈ R D and β ∈ R D are initialized such that the per-channel activations have roughly zero mean and unit variance at the beginning of training.

The invertible linear transformation W ∈ R D×D is parameterized using a QR decomposition (Hoogeboom et al., 2019) .

<|TLDR|>

@highlight

A deep hierarchical state-space model in which the state transitions of correlated objects are coordinated by graph neural networks.

@highlight

A hierarchical latent variable model of sequential dynamic processes of multiple objects when each object exhibits significant stochasticity.

@highlight

The paper presents a relational state-space model that simulates the joint state transitions of correlated objects which are hierarchically coordinated in a graph structure.