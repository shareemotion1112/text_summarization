Vanilla RNN with ReLU activation have a simple structure that is amenable to systematic dynamical systems analysis and interpretation, but they suffer from the exploding vs. vanishing gradients problem.

Recent attempts to retain this simplicity while alleviating the gradient problem are based on proper initialization schemes or orthogonality/unitary constraints on the RNN’s recurrency matrix, which, however, comes with limitations to its expressive power with regards to dynamical systems phenomena like chaos or multi-stability.

Here, we instead suggest a regularization scheme that pushes part of the RNN’s latent subspace toward a line attractor configuration that enables long short-term memory and arbitrarily slow time scales.

We show that our approach excels on a number of benchmarks like the sequential MNIST or multiplication problems, and enables reconstruction of dynamical systems which harbor widely different time scales.

Theories of complex systems in biology and physics are often formulated in terms of sets of stochastic differential or difference equations, i.e. as stochastic dynamical systems (DS).

A long-standing desire is to retrieve these generating dynamical equations directly from observed time series data (Kantz & Schreiber, 2004) .

A variety of machine and deep learning methodologies toward this goal have been introduced in recent years (Chen et al., 2017; Champion et al., 2019; Jordan et al., 2019; Duncker et al., 2019; Ayed et al., 2019; Durstewitz, 2017; Koppe et al., 2019) , many of them based on recurrent neural networks (RNN) which can universally approximate any DS (i.e., its flow field) under some mild conditions (Funahashi & Nakamura, 1993; Kimura & Nakano, 1998) .

However, vanilla RNN as often used in this context are well known for their problems in capturing long-term dependencies and slow time scales in the data (Hochreiter & Schmidhuber, 1997; Bengio et al., 1994) .

In DS terms, this is generally due to the fact that flexible information maintenance over long periods requires precise fine-tuning of model parameters toward 'line attractor' configurations ( Fig. 1) , a concept first propagated in computational neuroscience for addressing animal performance in parametric working memory tasks (Seung, 1996; Seung et al., 2000; Durstewitz, 2003) .

Line attractors introduce directions of zero-flow into the model's state space that enable long-term maintenance of arbitrary values (Fig. 1) .

Specially designed RNN architectures equipped with gating mechanisms and (linear) memory cells have been suggested for solving this issue (Hochreiter & Schmidhuber, 1997; Cho et al., 2014) .

However, from a DS perspective, simpler models that can more easily be analyzed and interpreted in DS terms, and for which more efficient inference algorithms exist that emphasize approximation of the true underlying DS would be preferable.

Recent solutions to the vanishing vs. exploding gradient problem attempt to retain the simplicity of vanilla RNN by initializing or constraining the recurrent weight matrix to be the identity (Le et al., 2015) , orthogonal (Henaff et al., 2016; Helfrich et al., 2018) or unitary (Arjovsky et al., 2016) .

In this way, in a system including piecewise linear (PL) components like rectified-linear units (ReLU), line attractor dimensions are established from the start by construction or ensured throughout training by a specifically parameterized matrix decomposition.

However, for many DS problems, line attractors instantiated by mere initialization procedures may be unstable and quickly dissolve during training.

On the other hand, orthogonal or unitary constraints are too restrictive for reconstructing DS, and more generally from a computational perspective as well (Kerg et al., 2019) : For instance, neither 2) with flow field (grey) and nullclines (set of points at which the flow of one of the variables vanishes, in blue and red).

Insets: Time graphs of z 1 for T = 30 000.

A) Perfect line attractor.

The flow converges to the line attractor from all directions and is exactly zero on the line, thus retaining states indefinitely in the absence of perturbations, as illustrated for 3 example trajectories (green) started from different initial conditions.

B) Slightly detuned line attractor (cf.

Durstewitz (2003) ).

The system's state still converges toward the 'line attractor ghost ' (Strogatz, 2015) , but then very slowly crawls up within the 'attractor tunnel' (green trajectory) until it hits the stable fixed point at the intersection of nullclines.

Within the tunnel, flow velocity is smoothly regulated by the gap between nullclines, thus enabling arbitrary time constants.

Note that along other, not illustrated dimensions of the system's state space the flow may still evolve freely in all directions.

C) Simple 2-unit solution to the addition problem exploiting the line attractor properties of ReLUs in the positive quadrant.

The output unit serves as a perfect integrator, while the input unit will only convey those input values to the output unit that are accompanied by a '1' in the second input stream (see 7.1.1 for complete parameters).

chaotic behavior (that requires diverging directions) nor settings with multiple isolated fixed point or limit cycle attractors are possible.

Here we therefore suggest a different solution to the problem, by pushing (but not strictly enforcing) ReLU-based, piecewise-linear RNN (PLRNN) toward line attractor configurations along some (but not all) directions in state space.

We achieve this by adding special regularization terms for a subset of RNN units to the loss function that promote such a configuration.

We demonstrate that our approach outperforms, or is en par with, LSTM and other, initialization-based, methods on a number of 'classical' machine learning benchmarks (Hochreiter & Schmidhuber, 1997) .

More importantly, we demonstrate that while with previous methods it was difficult to capture slow behavior in a DS that exhibits widely different time scales, our new regularization-supported inference efficiently captures all relevant time scales.

Long-range dependency problems in RNN.

Error gradients in vanilla RNN trained by some form of gradient descent, like back-propagation through time (BPTT, Rumelhart et al. (1986) ), tend to either explode or vanish due to the large product of derivative terms that results from recursive application of the chain rule over time steps (Hochreiter, 1991; Bengio et al., 1994; Hochreiter & Schmidhuber, 1997) .

Formally, RNN z t = F θ (z t−1 , s t ) are discrete time dynamical systems that tend to either converge, e.g. to fixed point or limit cycle attractors, or diverge (to infinity or as in chaotic systems) over time, unless parameters of the system are precisely tuned to create directions of zero-flow in the system's state space (Fig. 1) , called line attractors (Seung, 1996; Seung et al., 2000; Durstewitz, 2003) .

Convergence of the RNN in general implies vanishing and global divergence exploding gradients.

To address this issue, RNN with gated memory cells have been specifically designed (Hochreiter & Schmidhuber, 1997; Cho et al., 2014) , but these are complicated and tedious to analyze from a DS perspective.

Recently, Le et al. (2015) observed that initialization of the recurrent weight matrix W to the identity in ReLU-based RNN may yield performance en par with LSTMs on standard machine learning benchmarks.

For a ReLU with activity z t ≥ 0, zero bias and unit slope, this results in the identity mapping, hence a line attractor configuration.

Talathi & Vartak (2016) expanded on this idea by initializing the recurrence matrix such that its largest absolute eigenvalue is 1, arguing that this would leave other directions in the system's state space free for computations other than memory maintenance.

Later work enforced orthogonal (Henaff et al., 2016; Helfrich et al., 2018; Jing et al., 2019) or unitary (Arjovsky et al., 2016) constraints on the recurrent weight matrix during training.

While this appears to yield long-term memory performance superior to that of LSTMs, these networks are limited in their computational power (Kerg et al., 2019) .

This may be a consequence of the fact that RNN with orthogonal recurrence matrix are quite restricted in the range of dynamical phenomena they can produce, e.g. chaotic attractors are not possible since diverging eigen-directions are disabled.

Our approach therefore is to establish line attractors only along some but not all directions in state space, and to only push the RNN toward these configurations but not strictly enforce them, such that convergence or divergence of RNN dynamics is still possible.

We furthermore implement these concepts through regularization terms in the loss functions, such that they are encouraged throughout training unlike when only established through initialization.

Dynamical systems reconstruction.

From a natural science perspective, the goal of reconstructing the underlying DS fundamentally differs from building a system that 'merely' yields good ahead predictions, as in DS reconstruction we require that the inferred model can freely reproduce (when no longer guided by the data) the underlying attractor geometries and state space properties (see section 3.5, Fig. S2 ; Kantz & Schreiber (2004) ).

Earlier work using RNN for DS identification (Roweis & Ghahramani, 2002; Yu et al., 2006) mainly focused on inferring the posterior over latent trajectories Z = {z 1 , . . .

, z T } given time series data X = {x 1 , . . .

, x T }, p(Z|X), and on ahead predictions (Lu et al., 2017) , hence did not show that inferred models can generate the underlying attractor geometries on their own.

Others (Trischler & D'Eleuterio, 2016; Brunton et al., 2016) attempt to approximate the flow field, obtained e.g. by numerical differentiation, directly through basis expansions or neural networks, but numerical derivatives are problematic for their high variance and other numerical issues (Raissi, 2018; Baydin et al., 2018; Chen et al., 2017) .

Some approaches assume the form of the DS equations basically to be given (Raissi, 2018; Gorbach et al., 2017) and focus on estimating the system's latent states and parameters, rather than approximating an unknown DS based on the observed time series information alone.

In many biological systems like the brain the intrinsic dynamics are highly stochastic with many noise sources, like probabilistic synaptic release (Stevens, 2003) , such that models that do not explicitly account for dynamical process noise (Champion et al., 2019; Rudy et al., 2019) may be less suitable.

Finally, some fully probabilistic models for DS reconstruction based on GRU (Fraccaro et al. (2016) , cf.

Jordan et al. (2019) ), LSTM (Zheng et al., 2017) , or radial basis function (Zhao & Park, 2017) networks are not easily interpretable and amenable to DS analysis.

Most importantly, none of these previous approaches considers the long-range dependency problem within more easily tractable RNN for DS reconstruction.

Assume we are given two multivariate time series S = {s t } and X = {x t }, one we will denote as 'inputs' (S) and the other as 'outputs' (X).

We will first consider the 'classical' (supervised) machine learning setting where we wish to map S on X through a RNN with latent state equation z t = F θ (z t−1 , s t ), as for instance in the 'addition problem' (Hochreiter & Schmidhuber, 1997) .

In DS reconstruction, in contrast, we usually have a dense time series X from which we wish to infer (unsupervised) the underlying DS, where S may provide an additional forcing function or sparse experimental inputs or perturbations.

The latent RNN we consider here takes the specific form

where

the input mapping, h ∈ R M ×1 a bias, and ε t a Gaussian noise term with diagonal covariance ma-

This specific formulation is originally motivated by firing rate (population) models in computational neuroscience (Song et al., 2016; Durstewitz, 2017) , where latent states z t may represent membrane voltages or currents, A the neurons' passive time constants, W the synaptic coupling among neurons, and φ(·) the voltage-to-rate transfer function.

However, for a RNN in the form z t = W φ (z t−1 ) + h, note that the simple change of variables y t → W −1 (z t − h) will yield the more familiar form y t = φ (W y t−1 + h) (Beer, 2006) .

Besides its neuroscience motivation, note that by letting A = I, W = 0, h = 0, we get a strict line attractor system across the variables' whole support which we conjecture will be of advantage for establishing long short-term memory properties.

Also we can solve for all of the system's fixed points analytically by solving the equations z * = (I − A − W D Ω ) −1 h, with D Ω as defined in Suppl.

7.1.2, and can determine their stability from the eigenvalues of matrix A + W D Ω .

We could do the same for limit cycles, in principle, which are fixed points of the r-times iterated map F r θ , although practically the number of configurations to consider increases exponentially as 2 M ·r .

Finally, we remark that a discrete piecewise-linear system can, under certain conditions, be transformed into an equivalent continuous-time (ODE) piecewise-linear systemζ = G Ω (ζ(t), s(t)) (Suppl.

7.1.2, Ozaki (2012)), in the sense that if ζ(t) = z t , then ζ(t + ∆t) = z t+1 after a defined time step ∆t.

These are among the properties that make PLRNNs more amenable to rigorous DS analysis than other RNN formulations.

We will assume that the latent RNN states z t are coupled to the actual observations x t through a simple observation model of the form

in the case of real-valued observations x t ∈ R N ×1 , where B ∈ R N ×M is a factor loading matrix and diag(Γ) ∈ R N + the diagonal covariance matrix of the Gaussian observation noise, or

in the case of multi-categorical observations x i,t ∈ {0, 1}, i x i,t = 1.

We start from a similar idea as Le et al. (2015) , who initialized RNN parameters such that it performs an identity mapping for z i,t ≥ 0.

However, 1) we use a neuroscientifically motivated network architecture (eq. 1) that enables the identity mapping across the variables whole support, z i,t ∈ [−∞, +∞], 2) we encourage this mapping only for a subset M reg ≤ M of units (Fig. S1 ), leaving others free to perform arbitrary computations, and 3) we stabilize this configuration throughout training by introducing a specific L 2 regularization for parameters A, W , and h in eq. 1.

That way, we divide the units into two types, where the regularized units serve as a memory that tends to decay very slowly (depending on the size of the regularization term), while the remaining units maintain the flexibility to approximate any underlying DS, yet retaining the simplicity of the original RNN model (eq. 1).

Specifically, the following penalty is added to the loss function ( Fig. S1 ):

While this formulation allows us to trade off, for instance, the tendency toward a line attractor (A → I, h → 0) vs. the sensitivity to other units' inputs (W → 0), for all experiments performed here a common value, τ A = τ W = τ h = τ , was assumed for the three regularization factors.

For comparability with other approaches like LSTMs (Hochreiter & Schmidhuber, 1997) or iRNN (Le et al., 2015) , we will assume that the latent state dynamics eq. 1 are deterministic (i.e., Σ = 0), will take g(z t ) = z t and Γ = I N in eq. 2 (leading to an implicit Gaussian assumption with identity covariance matrix), and will use stochastic gradient descent (SGD) for training to minimize the squared-error loss across R samples,

, between estimated and actual outputs for the addition and multiplication problems, and the cross entropy loss

i,T ) for sequential MNIST, to which penalty eq. 4 was added for the regularized PLRNN (rPLRNN).

We used the Adam algorithm (Kingma & Ba, 2014) from the PyTorch package (Paszke et al., 2017 ) with a learning rate of 0.001, a gradient clip parameter of 10, and batch size of 16.

In all cases, SGD is stopped after 100 epochs and the fit with the lowest loss across all epochs is chosen.

For DS reconstruction we request that the latent RNN approximates the true generating system of equations, which is a taller order than learning the mapping S → X or predicting future values in a time series (cf. sect.

3.5).

This point has important implications for the design of models, inference algorithms and performance metrics if the primary goal is DS reconstruction rather than 'mere' time series forecasting.

In this context we consider the fully probabilistic, generative RNN eq. 1.

Together with eq. 2 (where we take g(z t ) = φ(z t )) this gives the typical form of a nonlinear state space model (Durbin & Koopman, 2012) with observation and process noise.

We solve for the parameters θ = {A, W , C, h, Σ, B, Γ} by maximum likelihood, for which an efficient ExpectationMaximization (EM) algorithm has recently been suggested (Durstewitz, 2017; Koppe et al., 2019) , which we will briefly summarize here.

Since the involved integrals are not tractable, we start off from the evidence-lower bound (ELBO) to the log-likelihood which can be rewritten in various useful ways:

In the E-step, given a current estimate θ * for the parameters, we seek to determine the posterior p θ (Z|X) which we approximate by a global Gaussian q(Z|X) instantiated by the maximizer (mode) Z * of p θ (Z|X) as an estimator of the mean, and the negative inverse Hessian around this maximizer as an estimator of the state covariance, i.e.

since Z integrates out in p θ (X) (equivalently, this result can be derived from a Laplace approximation to the log-likelihood, log p(X|θ)

where L * is the Hessian evaluated at the maximizer).

We solve this optimization problem by a fixed-point iteration scheme that efficiently exploits the model's piecewise linear structure (see Suppl.

7.1.3, Durstewitz (2017) ; Koppe et al. (2019) ).

Using this approximate posterior for p θ (Z|X), based on the model's piecewise-linear structure most of the expectation values

, and E z∼q [φ(z)φ(z) ], could be solved for (semi-)analytically (where z is the concatenated vector form of Z, as in Suppl.

7.1.3).

In the Mstep, we seek θ * := arg max θ L(θ, q * ), assuming proposal density q * to be given from the E-step, which for a Gaussian observation model amounts to a simple linear regression problem (see Suppl.

eq. 23).

To force the PLRNN to really capture the underlying DS in its governing equations, we use a previously suggested (Koppe et al. 2019 ) stepwise annealing protocol that gradually shifts the burden of fitting the observations X from the observation model eq. 2 to the latent RNN model eq. 1 during training, the idea of which is to establish a mapping from latent states Z to observations X first, fixing this, and then enforcing the temporal consistency constraints implied by eq. 1 while accounting for the actual observations.

Measures of prediction error.

For the machine learning benchmarks we employed the same criteria as used for optimization (MSE or cross-entropy, sect.

3.3) as performance metrics, evaluated across left-out test sets.

In addition, we report the relative frequency P correct of correctly predicted trials Agreement in attractor geometries.

From a DS perspective, it is not sufficient or even sensible to judge a method's ability to infer the underlying DS purely based on some form of (ahead-)prediction error like the MSE defined on the time series itself (Ch.12 in Kantz & Schreiber (2004) ).

Rather, we require that the inferred model can freely reproduce (when no longer guided by the data) the underlying attractor geometries and state space properties.

This is not automatically guaranteed for a model that yields agreeable ahead predictions on a time series.

Vice versa, if the underlying attractor is chaotic, with a tiny bit of noise even trajectories starting from the same initial condition will quickly diverge and ahead-prediction errors are not even meaningful as a performance metric (Fig. S2A ).

To quantify how well an inferred PLRNN captured the underlying dynamics we therefore followed Koppe et al. (2019) and used the Kullback-Leibler divergence between the true and reproduced probability distributions across states in state space, thus assessing the agreement in attractor geometries (cf.

Takens (1981) ; Sauer et al. (1991) ) rather than in precise matching of time series,

where p true (x) is the true distribution of observations across state space (not time!), p gen (x|z) is the distribution of observations generated by running the inferred PLRNN, and the sum indicates a spatial discretization (binning) of the observed state space (see Suppl.

7.1.4 for more details).

We emphasize thatp (k) gen (x|z) is obtained from freely simulated trajectories, i.e. drawn from the prior p(z), not from the inferred posteriorsp(z|x train ). (The form ofp(z) is given by the dynamical model eq. 1 and has a 'mixture of piecewise-Gaussians' structure, see Koppe et al. (2019) .)

In addition, to assess reproduction of time scales by the inferred PLRNN, we computed the average correlation between the power spectra of the true and generated time series.

Fig. S1 , with additional regularization term (eq. 4) during training LSTM Long Short-Term Memory (Hochreiter & Schmidhuber, 1997) 4 NUMERICAL EXPERIMENTS

We compared the performance of our rPLRNN to other models on the following three benchmarks requiring long short-term maintenance of information (as in Talathi & Vartak (2016) and Hochreiter & Schmidhuber (1997) ): 1) The addition problem of time length T consists of 100 000 training and 10 000 test samples of 2 × T input series S = {s 1 , . . .

, s T }, where entries s 1,: ∈ [0, 1] are drawn from a uniform random distribution and s 2,: ∈ {0, 1} contains zeros except for two indicator bits placed randomly at times t 1 < 10 and t 2 < T /2.

Constraints on t 1 and t 2 are chosen such that every trial requires a long memory of at least T /2 time steps.

At the last time step T , the target output of the network is the sum of the two inputs in s 1,: indicated by the 1-entries in s 2,: , x target T = s 1,t1 + s 1,t2 .

2) The multiplication problem is the same as the addition problem, only that the product instead of the sum has to be produced by the RNN as an output at time T , x target T = s 1,t1 ·s 1,t2 .

3) The MNIST dataset (LeCun & Cortes, 2010) consists of 60 000 training and 10 000 28 × 28 test images of hand written digits.

To make this a time series problem, in sequential MNIST the images are presented sequentially, pixel-by-pixel, scanning lines from upper left to bottom-right, resulting in time series of fixed length T = 784.

On all three benchmarks we compare the performance of the rPLRNN (eq. 1) to several other models summarized in Table 1 .

To achieve a meaningful comparison, all models have the same number of hidden states M , except for the LSTM, which requires three additional parameters for each hidden state and hence has only M/4 hidden states, yielding the overall same number of trainable parameters as for the other models.

In all cases, M = 40, which initial numerical exploration suggested to be a good compromise between model complexity (bias) and data fit (variance) (Fig. S3) .

Fig. 2 summarizes the results for the machine learning benchmarks.

As can be seen, on the addition and multiplication tasks, and in terms of either the MSE or percentage correct, our rPLRNN outperforms all other tested methods, including LSTMs.

Indeed, the LSTM performs even significantly worse than the iRNN and the iPLRNN.

The large error bars in Fig. 2 result from the fact that the networks mostly learn these tasks in an all-or-none fashion, i.e. either learn the task and succeed in almost 100 percent of the cases or fail completely.

The results for the sequential MNIST problem are summarized in Fig. 2C .

While in this case the LSTM outperforms all other methods, the rPLRNN is almost en par with it.

In addition, the iPLRNN outperforms the iRNN.

Similar results were obtained for M = 100 units (M = 25, respectively, for LSTM; Fig. S6 ).

While the rPLRNN in general outperformed the pure initialization-based models (iRNN, npRNN, iPLRNN) , confirming that a line attractor subspace present at initialization may be lost throughout training, we conjecture that this difference in performance will become even more pronounced as noise levels or task complexity increase.

Fig. 3 : Reconstruction of a 2-time scale DS (biophysical bursting neuron model) in limit cycle regime.

A) KL divergence (D KL ) between true and generated state space distributions as a function of τ .

Unstable (globally diverging) system estimates were removed.

B) Average MSE between power spectra of true and reconstructed DS.

C) Average normalized MSE between power spectra of true and reconstructed DS split according to low (≤ 50 Hz) and high (> 50 Hz) frequency components.

Error bars = SEM in all graphs.

D) Example of (best) generated time series (red=reconstruction with τ = 2 3 ).

Here our goal was to examine whether our regularization approach would also help with the identification of DS that harbor widely different time scales.

By tuning systems in the vicinity of line attractors, multiple arbitrary time scales can be realized in theory (Durstewitz, 2003) .

To test this, we used a biophysically motivated (highly nonlinear) bursting cortical neuron model with one voltage and two conductance recovery variables (see Durstewitz (2009) ), one slow and one fast (Suppl.

7.1.5).

Reproduction of this DS is challenging since it produces very fast spikes on top of a slow nonlinear oscillation (Fig. 3D) .

Time series of standardized variables of length T = 1500 were generated from this model and provided as observations to the rPLRNN inference algorithm.

rPLRNNs with M = {8 . . .

18} states were estimated, with the regularization factor varied within τ ∈ {0, 10 1 , 10 2 , 10 3 , 10 4 , 10 5 }/1500.

Fig. 3A confirms our intuition that stronger regularization leads to better DS reconstruction as assessed by the KL divergence between true and generated state distributions.

This decrease in D KL is accompanied by a likewise decrease in the MSE between the power spectra of true (Suppl. eq. 27) and generated (rPLRNN) voltage traces as τ increased (Fig. 3B) .

Fig. 3D gives an example of voltage traces and gating variables freely simulated (i.e., sampled) from the generative rPLRNN trained with τ = 2 3 , illustrating that our model is in principle capable of capturing both the stiff spike dynamics and the slower oscillations in the second gating variable at the same time.

Fig. 3C provides more insight into how the regularization worked: While the high frequency components (> 50 Hz) related to the repetitive spiking activity hardly benefitted from increasing τ , there was a strong reduction in the MSE computed on the power spectrum for the lower frequency range (≤ 50 Hz), suggesting that increased regularization helps to map slowly evolving components of the dynamics.

In this work we have introduced a simple solution to the long short-term memory problem in RNN that on the one hand retains the simplicity and tractability of vanilla RNN, yet on the other hand does not curtail the universal computational capabilities of RNN (Koiran et al., 1994; Siegelmann & Sontag, 1995) and their ability to approximate arbitrary DS (Funahashi & Nakamura, 1993; Kimura & Nakano, 1998; Trischler & D'Eleuterio, 2016) .

We achieved this by adding regularization terms to the loss function that encourage the system to form a 'memory subspace', that is, line attractor dimensions (Seung, 1996; Durstewitz, 2003) which would store arbitrary values for, if unperturbed, arbitrarily long periods.

At the same time we did not rigorously enforce this constraint which has important implications for capturing slow time scales in the data: It allows the RNN to slightly depart from a perfect line attractor, which has been shown to constitute a general dynamical mechanism for regulating the speed of flow and thus the learning of arbitrary time constants that are not naturally included qua RNN design (Durstewitz, 2003; 2004) .

This is because as we come infinitesimally close to a line attractor and thus a bifurcation in the system's parameter space, the flow along this direction becomes arbitrarily slow until it vanishes completely in the line attractor configuration (Fig. 1) .

Moreover, part of the RNN's latent space was not regularized at all, leaving the system enough degrees of freedom for realizing arbitrary computations or dynamics.

We showed that the rPLRNN is en par with or outperforms initialization-based approaches and LSTMs on a number of classical benchmarks, and, more importantly, that the regularization strongly facilitates the identification of challenging DS with widely different time scales in PLRNN-based algorithms for DS reconstruction.

Future work will explore a wider range of DS models and empirical data with diverse temporal and dynamical phenomena.

Another future direction may be to replace the EM algorithm by black-box variational inference, using the re-parameterization trick for gradient descent (Kingma & Welling, 2013; Rezende et al., 2014; Chung et al., 2015) .

While this would come with better scaling in M , the number of latent states (the scaling in T is linear for EM as well, see Paninski et al. (2010) ), the EM used here efficiently exploits the model's piecewise linear structure in finding the posterior over latent states and computing the parameters (see Suppl.

7.1.3).

It may thus be more accurate and suitable for smaller-scale problems where high precision is required, as often encountered in neuroscience or physics.

7 SUPPLEMENTARY MATERIAL 7.1 SUPPLEMENTARY TEXT 7.1.1 Simple exact PLRNN solution for addition problem

The exact PLRNN parameter settings (cf. eq. 1) for solving the addition problem with 2 units (cf.

Fig. 1C ) are as follows:

Under some conditions we can translate the discrete into an equivalent continuous time PLRNN.

Using D Ω(t) as defined below (7.1.3) for a single time step t, we can rewrite (ignoring the noise term and inputs) PLRNN eq. 1 in the form

where Ω(t) := {m|z m,t > 0} is the set of all unit indices with activation larger 0 at time t. To convert this into an equivalent (in the sense defined in eq. 11) system of (piecewise) ordinary differential equations (ODE), we need to find parameters W Ω and h,

such that

where ∆t is the time step with which the empirically observed time series X was sampled.

From these conditions it follows that for each of the s ∈ {1, . . .

, 2 M } subregions (orthants) defined by fixed index sets Ω s ⊆ {1, . . .

, M } we must have

where we assume that D Ω s is constant for one time step, i.e. between 0 and ∆t.

We approach this by first solving the homogeneous system using the general ansatz for systems of linear ODEs,

where we have used z 0 = k c k v k on lines 15 and 16.

Hence we can infer matrix W Ω s from the eigendecomposition of matrix W Ω s , by lettingλ k = 1 ∆t log λ k , where λ k are the eigenvalues of W Ω s , and reassembling

We obtain the general solution for the inhomogeneous case by requiring that for all fixed points z * = F (z * ) of the map eq. 9 we have G(z * ) = 0.

Using this we obtaiñ

Assuming inputs s t to be constant across time step ∆t, we can apply the same transformation to input matrix C. Fig. S5 illustrates the discrete to continuous PLRNN conversion for a nonlinear oscillator.

Note that in the above derivations we have assumed that matrix W Ω s can be diagonalized, and that all its eigenvalues are nonzero (in fact, W Ω s should not have any negative real eigenvalues).

In general, not every discrete-time PLRNN can be converted into a continuous-time ODE system in the sense defined above.

For instance, we can have chaos in a 1d nonlinear map, while we need at least a 3d ODE system to create chaos (Strogatz, 2015) .

Here we briefly outline the fixed-point-iteration algorithm for solving the maximization problem in eq. 6 (for more details see Durstewitz (2017) ; Koppe et al. (2019) ).

Given a Gaussian latent PLRNN and a Gaussian observation model, the joint density p(X, Z) will be piecewise Gaussian, hence eq. 6 piecewise quadratic in Z. Let us concatenate all state variables across m and t into one long column vector z = (z 1,1 , . . .

, z M,1 , . . .

, z 1,T , . . .

, z M,T ) , arrange matrices A, W into large M T × M T block tri-diagonal matrices, define d Ω := 1 z1,1>0 , 1 z2,1>0 , . . .

, 1 z M,T >0 as an indicator vector with a 1 for all states z m,t > 0 and zeros otherwise, and D Ω := diag(d Ω ) as the diagonal matrix formed from this vector.

Collecting all terms quadratic, linear, or constant in z, we can then write down the optimization criterion in the form

In essence, the algorithm now iterates between the two steps:

2.

Given fixed z * , recompute D Ω until either convergence or one of several stopping criteria (partly likelihood-based, partly to avoid loops) is reached.

The solution may afterwards be refined by one quadratic programming step.

Numerical experiments showed this algorithm to be very fast and efficient (Durstewitz, 2017; Koppe et al., 2019) .

At z * , an estimate of the state covariance is then obtained as the inverse negative Hessian,

In the M-step, using the proposal density q * from the E-step, the solution to the maximization problem θ * := arg max θ L(θ, q * ), can generally be expressed in the form

where, for the latent model, eq. 1, α t = z t and β t := z t−1 , φ(z t−1 ) , s t , 1 ∈ R 2M +K+1 , and for the observation model, eq. 2, α t = x t and β t = g (z t ).

The measure D KL introduced in the main text for assessing the agreement in attractor geometries only works for situations where the ground truth p true (X) is known.

Following Koppe et al. (2019) , here we would like to briefly indicate how a proxy for D KL may be obtained in empirical situations where no ground truth is available.

Reasoning that for a well reconstructed DS the inferred posterior p inf (z|x) given the observations should be a good representative of the prior generative dynamics p gen (z), one may use the Kullback-Leibler divergence between the distribution over latent states, obtained by sampling from the prior density p gen (z), and the (data-constrained) posterior distribution p inf (z|x) (where z ∈ R M ×1 and x ∈ R N ×1 ), taken across the system's state space:

As evaluating this integral is difficult, one could further approximate p inf (z|x) and p gen (z) by Gaussian mixtures across trajectories, i.e.

, where the mean and covariance of p(z t |x 1:T ) and p(z l |z l−1 ) are obtained by marginalizing over the multivariate distributions p(Z|X) and p gen (Z), respectively, yielding E[z t |x 1:T ], E[z l |z l−1 ], and covariance matrices Var(z t |x 1:T ) and Var(z l |z l−1 ).

Supplementary eq. 24 may then be numerically approximated through Monte Carlo sampling (Hershey & Olsen, 2007) by

For high-dimensional state spaces, for which MC sampling becomes challenging, there is luckily a variational approximation of eq. 24 available (Hershey & Olsen, 2007) :

where the KL divergences in the exponentials are among Gaussians for which we have an analytical expression.

The neuron model used in section 4.2 is described by

σ(V ) = 1 + .33e

where C m refers to the neuron's membrane capacitance, the g • to different membrane conductances, E • to the respective reversal potentials, and m, h, and n are gating variables with limiting values given by

Different parameter settings in this model lead to different dynamical phenomena, including regular spiking, slow bursting or chaos (see Durstewitz (2009) with parameters in the chaotic regime (blue curves) and with simple fixed point dynamics in the limit (red line).

Although the system has vastly different limiting behaviors (attractor geometries) in these two cases, as visualized in the state space, the agreement in time series initially seems to indicate a perfect fit.

B) Same as in A) for two trajectories drawn from exactly the same DS (i.e., same parameters) with slightly different initial conditions.

Despite identical dynamics, the trajectories immediately diverge, resulting in a high MSE.

Dash-dotted grey lines in top graphs indicate the point from which onward the state space trajectories were depicted.

@highlight

We develop a new optimization approach for vanilla ReLU-based RNN that enables long short-term memory and identification of arbitrary nonlinear dynamical systems with widely differing time scales.