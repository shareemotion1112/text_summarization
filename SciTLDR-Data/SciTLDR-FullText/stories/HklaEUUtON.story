Sequential data often originates from diverse environments.

Across them exist both shared regularities and environment specifics.

To learn robust cross-environment descriptions of sequences we introduce disentangled state space models (DSSM).

In the latent space of DSSM environment-invariant state dynamics is explicitly disentangled from environment-specific information governing that dynamics.

We empirically show that such separation enables robust prediction, sequence manipulation and environment characterization.

We also propose an unsupervised VAE-based training procedure to learn DSSM as Bayesian filters.

In our experiments, we demonstrate state-of-the-art performance in controlled generation and prediction of bouncing ball video sequences across varying gravitational influences.

Learning dynamics and models from sequential data is a central task in various domains of science BID5 .

This includes managing input of diverse complexity e.g. natural language BID11 , videos BID28 or financial time-series (Øksendal, 2003) .

It is also crucial for building interactive agents which use reinforcement and control algorithms on top BID6 .

Traditional choice in engineering are state space models (SSM) BID21 , typically found in form of Kalman filters BID9 where well-crafted, relatively simple state representations and (normally linear) functional forms are used.

To improve flexibility, new solutions rather learn model-free SSM "from scratch".

Due to their non-autoregressive architecture they make an attractive alternative to recurrent neural networks.

Several recent works have already recognized the benefits of introducing additional structure into SSM: the requirement of separating confounders from actions, observations and rewards BID24 or content from dynamics BID32 BID8 , especially for transfer learning and extrapolation BID18 .

Complementary to these approaches, we focus on learning structured SSM to decouple system dynamics into its generic (enviromentinvariant) and environment-specific components.

Some examples of sequential data which naturally admit this structure are given in figure 1.

Dynamics of these are defined by some constant external factors which we jointly refer to as environment.

More concretely, we explore a panel data setting in which we are given multiple sequences describing the same time-evolving phenomena, one or more per environment e.

We would like to learn a robust non-parametric SSM to represent the dynamics of that phenomena across these environments, and robustly extrapolate to the unseen ones.

To do so, we explicitly model e as a learnable static element of the latent space.

Our idea is based on the assumption that one can decouple sequence dynamics to: (i) the generic part which is invariant across environments; and (ii) the environmentspecific part.

In other words, true e integrates all unobserved environment-specific influences which bias generic system dynamics.

Our hypothesis is that considering disentangled, implicitly causal structure of SSM enhances predictive robustness, domain adaptation, and allows for environment characterization and reasoning under interventions e.g. counterfactual inference.

Figure 1: Sequential systems across environments.

Examples include, from left to right: (i) Michaelis-Menten model for enzyme kinetics, governed by reaction rate constants k; (ii) bouncing ball kinematics, determined by ball weight and playground characteristics; (iii) ODE dynamics, governed by model parameters; (iv) bat swinging motion, influenced by the person performing it.

In each example, environments are defined differently, depending on what governs sequence dynamics.

DSSM.

We introduce a class of non-parametric SSM tailored to exploit invariance from sequential data originating from heterogeneous environments.

Disentangled state space models (DSSM) (see FIG2 ) form a joint environment model while explicitly decoupling what is generic in sequence dynamics from what is environment-specific.

This enhances robustness and the ability to extrapolate knowledge to unseen environments.

Bayesian filtering.

We extend on recent advances in amortized variational inference to design an unsupervised training procedure and implement DSSM in form of Bayesian filters.

In the spirit of BID19 , well-established reparameterization trick is applied such that the gradient propagates through time.

While VAE heuristic provides no convergence guarantees, it is fast, robust and allows end-to-end training.

Video prediction and manipulation.

We analyze video sequences of a bouncing ball, influenced by varying gravity (environment).

We outperform state-of-the-art K-VAE BID8 in predictions, and also do interventions by "swapping environments" i.e. we enforce a specific dynamic behaviour by using an environment from another sequence which exhibits the desired behaviour.

Example videos are available at: https://sites.google.com/view/dssm.

Closely related to our proposal are approaches which consider structured and disentangled representation of videos, separating the pose from the content BID4 BID29 ...

BID19 .

(b) Disentangled sequential autoencoder (DSA) BID32 decouples time-invariant content from the time-varying features.

(c) Kalman-VAE BID8 separates object (content) representation from its dynamics (we did not depict control input here).

(d) DSSM introduce environments E to model environment-specific effects on sequence dynamics.

BID30 BID32 or the object appearance from its dynamics BID8 .

Proposed models were shown to improve the prediction BID30 BID4 and enable controlled generation with "feature swapping" BID29 BID32 .

This so called content-based disentanglement was also performed in speech analysis where the structure imposed the explicit separation of sequence-and segment-level attributes BID16 .VAE frameworks have already been extended to sequence modeling BID25 , and applied to speech BID0 BID3 BID7 BID10 , videos BID32 ) and text BID1 .

However, these (mainly) recurrent neural network-based approaches are autoregressive and hence not always suitable e.g. for planning and control from raw pixel space BID31 BID14 .To "image the world" BID13 ) from the latent space directly and circumvent the autoregressive feedback, alternative methods learn SSM instead BID19 BID8 BID23 .

In DVBF BID19 SSM is trained using VAE-based learning procedure which allows gradient to propagate through time during training.

K-VAE by Fraccaro et al. FORMULA0 is a two-layered model which decomposes object's representation from its dynamics.

DKF BID22 , and very closely related DMM BID23 , admit SSM structure but the state inference is conditioned on both past and future observations, so the structure of a filter is not preserved.

This is problematic as noted by BID19 .

Similar issues can be found in BID32 .As opposed to content-based methods which focus on the observation model, our work is focused on dynamics-based disentanglement.

This makes our approach complementary to existing (see also FIG2 ).

For example, while DSA can represent and manipulate the shape or color of a bouncing ball, our method can manipulate its trajectory.

To implement our Bayesian filter, we blended some recent ideas in amortized variational inference BID19 BID25 and adapted them to fit our novel DSSM architecture.

In this work, we assume that the underlying system is deterministic i.e. the latent process noise β and observation noise ω are both uncorrelated in time.

We consider the following DSSM description: DISPLAYFORM0 where f and g represent arbitrary flexible functions.

X i ∈ R D represents the observation in time step i and S i ∈ R N is the corresponding latent state.

Σ β and Σ ω are noise covariances which we for simplicity assume are isotropic Gaussian.

Our goal is to jointly learn the generative model which consists of the transition function f and observation function g, together with the corresponding recognition networks φ enc β , φ enc E and φ enc S which infer the process noise residual β i , the environment E, and the initial state S 0 respectively.

The framework overview is given in figure 3.Generative model.

Given an observed sequence X of length T , the joint distribution is: DISPLAYFORM1 This follows from figure 2d and the assumption that the process noise is serially uncorrelated.

We set the prior probabilities of the initial state p 0 (S 0 ), environment p 0 (E) and process noise p 0 (β i ) = p 0 (β) to be zero-mean unit-variance Gaussian.

Conditioned on β i and E, state transition is deterministic and the probability p(S i |S i−1 , E, β i ) is a Dirac function with the peak defined by equation (2).

The emission probability p(X i |S i ) is defined by equation (1).

Inference.

Joint variational distribution over the unobserved random variables E, S and β, for a sequence of observation X of length T factorizes as: DISPLAYFORM2 Here, the conditionals S − i |S i−1 , E and S i |β i , S − i are deterministic and defined by equation (2).

The remaining factors are given as follows: DISPLAYFORM3 Learning.

To match the posterior distributions of E, S 0 and β to the assigned prior probabilities p 0 (E), p 0 (S 0 ) and p 0 (β), we utilize reparametrization trick BID20 BID27 .

This enables end-to-end training.

To define the objective function we derive the variational lower bound L, which we consequently attempt to maximize during the training.

We start from the well-known equality BID20 : DISPLAYFORM4 Due to the conditional independence of the observations given the latent states, we can decompose the first term as: DISPLAYFORM5 The KL term can be shown to simplify into a sum of the following KL terms: DISPLAYFORM6 + KL(q(s 0 | X)||p 0 (S 0 )) DISPLAYFORM7 where we dropped the conditional dependency β i |S i−1 , X i , E in q to ease the notation.

Full L derivation is given in Appendix.

Algorithm 1 shows the details of the training procedure for one iteration, for a batch of size 1.

The extension to the batch training is trivial.

and φ enc E are given as bi-directional LSTM followed by a multilayer perceptron to convert LSTM-based sequence embedding into S 0 and E, and f as an LSTM cell as elaborated in FIG3 .

LSTM equations are taken from BID11 .Optimization Challenges.

Some performance improvements were observed with an additional heuristic regularization term, which ensures the consistency during the inference of environment E. Namely, we penalize the step-wise change in time embeddings produced by bi-directional LSTM used to model φ enc E , in order to enforce E to remain time-invariant.

To that end, we add an additional term to our objective function, the moment matching regularization term defined as: DISPLAYFORM8 where h i is the hidden state of the φ enc E LSTM cell in step i.

This idea is related to the approaches based on the maximum mean discrepancy BID12 .

Namely, enforcing equality of consecutive cell states corresponds to matching of their first moments.

Furthermore, similarly to BID1 BID19 we used a KL annealing scheme.

This was helpful for circumventing local minimum and preventing the KL term to converge to zero too early during the training.

The exact details are given in our experiments.

Bouncing ball in varying gravity settings.

We test our framework on a 2D bouncing ball problem where the ball kinematics is affected by a varying gravity vector.

The idea is to evaluate model robustness across environments -gravitational settings.

See also visualizations at: https:// sites.google.com/view/dssm.

Using the physics engine code from BID8 , we simulate video sequences of a bouncing ball.

During generation, we randomly change the gravity such that it remains constant within a sequence, but may vary across sequences.

The gravity vector takes 4 values, depending on whether the gravity points up, down, left or right.

The magnitude is kept fixed.

Each video frame is a 32x32 binary image.

We generate 16'000 trajectories across 40 time steps for training, and another 2000 trajectories across 70 time steps for testing.

We first perform long-term forecasting analysis comparing our method against one of the state-of-the-art approaches, the K-VAE from BID8 .

Next, we demonstrate controlled generation, by manipulating video sequences.

Effectively we perform interventions by "swapping environments" between video sequences, and similarly we swap initial states.

Furthermore, we show the ability to perform uncontrolled generation where both initial state and gravity vector are sampled from a prior.

Finally we visualize the environment embeddings to provide further intuiton.

Forecasting ball trajectory.

We use OpenCV inbuilt functions to detect ground truth ball position p t in each time frame.

The exact algorithm for the position extraction is provided in the Appendix Figure 4 : Bouncing ball trajectory forecasting.

DSSM-based Bayesian filter against K-VAE on the task of long-term forecasting: (left) velocity magnitude; (right) cosine similarity.

MM denotes moment-matching regularization.

Shown error curves are the test set averages.

B.2.

Following BID15 BID2 we define the ball velocity as v t = p t+1 − p t and compute the relative error in the predicted velocities of the balls for forecasting.

Evaluated models observe the first 25 frames of a test sequence and then forecast the next 45.

The results for relative error in magnitude and cosine similarity of the velocities are then averaged across all test sequences.

This is shown in figure 4 .

We observe an increase in prediction quality with respect to both metrics in comparison to the benchmarking model K-VAE.Video manipulation for controlled generation.

Firstly, the initial state which consists of the velocity vector and the ball position, is extracted from the baseline video sequence and then "injected" into a series of other test sequences.

Similarly, we performed the gravity environment replacement.

We then enrolled the sequence effectively performing controlled generation (see FIG4 ).

Environment identification.

We trained an auxiliary multi layer perceptron classifier to map E to true gravity value.

The cross-validation results performed on the training set rendered accuracy of 99.15%.

Visualized embeddings (for E ∈ R 3 ) are given in figure 6a.

Well-defined clusters can be observed, indicating that E indeed represents the true gravity.

Uncontrolled Generation.

We demonstrate the uncontrolled generation of the sequences where s 0 and E are sampled from the priors p 0 (E) and p 0 (S 0 ) respectively.

In FIG5 we observe how the generated sequences preserve natural bouncing ball dynamics.

This work proposes a novel view on data-driven learning of dynamics from diverse environments.

We proposed a new class of state space models particularly crafted to exploit this kind of a setting.

In disentangled state space models one separates generic system dynamics which is assumed to be invariant across environments and environment-specific information which governs this dynamics.

We showed that such separation is beneficial and allows us to learn robust cross-environment models which hold promise to generalize on unseen environments.

Our particular application was learning of the video dynamics of a bouncing ball affected by varying gravitational influences where we achieved state-of-the-art results.

Our future work will include other types of data.

A LOWER BOUND DERIVATION (SECTION 3) DISPLAYFORM0 (where the conditional independence follows from the state space model formulation) DISPLAYFORM1 (where we used the factorization of the variational and the prior distribution.

S0 is vector S without S0) DISPLAYFORM2 q(S0| x)q(E| x)q( β| X, S0, E)q( S0|S0, E, β) log p0( S0|S0, E, β) q( S0|S0, E, β) (where we dropped the integral sums for which the corresponding term does not depend on) DISPLAYFORM3 + KL(q( β| X, E, S0)||p0( β|E, S0)) (where the last term vanishes since s0|s0, E, β is deterministic) DISPLAYFORM4 (where we have p0(βi|E, si) = p0(β) by design) B EXPERIMENTS (SECTION 4) B.1 DETAILS To get compressed representation of each frame, the images are first passed through a shallow convolutional network.

Kernel size was set to 3x3, while the network depth was 64.

The step size was 1 in both directions.

We used ReLU activation units.

All of the hidden latent states were equal to 64.

To parameterize g we used a deconvolutional network with transposed convolutions.

The kernel size was set to 5.Following the insights from BID1 , we tried different settings for KL annealing in the model.

Since we have three KL terms in our model which have different roles, we do not penalize KL terms of time-invariant components i.e. KL(q(S 0 | X)||p 0 (S)) and KL(q(E| X)||p 0 (E)) as forcefully as KL(q(β i |S i , X i )||p 0 (β)) during training.

This makes it relatively easier for the model to learn the time-invariant components.

Similarly to BID8 , we also found that down-weighing the reconstruction term helps in faster convergence.

In particular we applied scaling coefficients of [0.1,0.2,0.3,1.0] for terms E q( S| X) [log p( X| S)], KL(q(E| X)||p 0 (E)), KL(q(S 0 | X)||p 0 (S)) and KL(q(β i |S i , X i )||p 0 (β))] respectively.

We use ADAM as the optimizer with 0.0008 as the initial learning rate, and weight decay of 0.6 applied every 20 epochs.

We use OpenCV's BID17 inbuilt functions to detect the pixel level positions of the ball in the images.

Predict observation:X i = g(S i ) end for ll_loss = LL( X, X ) (see Eq (9)) kl_loss = KL(E, S 0 , β); (see Eq (10)) mm_loss = MM(φ enc E ( X)); (see Eq (11)) Backpropagate(ll_loss, kl_loss, mm_loss)

@highlight

DISENTANGLED STATE SPACE MODELS

@highlight

The paper presents a generative state space model using a global latent variable E to capture environment-specific information.