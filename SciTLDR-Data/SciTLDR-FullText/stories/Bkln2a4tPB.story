Dynamical system models (including RNNs) often lack the ability to adapt the sequence generation or prediction to a given context, limiting their real-world application.

In this paper we show that hierarchical multi-task dynamical systems (MTDSs) provide direct user control over sequence generation, via use of a latent  code z that specifies the customization to the individual data sequence.

This enables style transfer, interpolation and morphing within generated sequences.

We show the MTDS can improve predictions via latent code interpolation, and avoid the long-term performance degradation of standard RNN approaches.

Time series data often arise as a related 'family' of sequences, where certain characteristic differences exist between the sequences in a dataset.

Examples include the style of handwritten text (Graves, 2013) , the response of a patient to an anaesthetic (Bird et al., 2019) , or the style of locomotion in motion capture (mocap) data (Ghosh et al., 2017) .

In this paper, we will consider how such variation may be modelled, and effectively controlled by an end user.

Such related data is often pooled to train a single dynamical system, despite the internal variation.

For a simple model, such as a linear dynamical system (LDS), this will result in learning only an average effect.

In contrast, a recurrent neural network (RNN) may model this variation, but in an implicit and opaque manner.

Such a 'black-box' approach prohibits end-user control, and may suffer from mode drift, such as in Ghosh et al. (2017) , where a generated mocap sequence performs an unprompted transition from walking to drinking.

Some of these problems may be alleviated by appending 'context labels' to the inputs (see e.g. Goodfellow et al., 2016, §10.2.4) which describe the required customization.

However, such labels are often unavailable, and the approach may fail to model the variation adequately even when they are.

To move beyond these approaches, we consider latent variable models, where a latent variable z characterizes each sequence.

This may be seen as a form of multi-task learning (MTL, see Zhang & Yang, 2017) , from which we derive the name multi-task dynamical system (MTDS), with each sequence treated as a task.

A straightforward approach is to append the latent z to the inputs of the model, similarly to the 'context label' approach, thereby providing customization of the various bias (or offset) parameters of the model.

A number of examples of this have been proposed recently, e.g. in Yingzhen & Mandt (2018) and Miladinović et al. (2019) .

Nevertheless, this 'bias customization' has limited expressiveness and is often unsuitable for customizing simple models.

In this paper we investigate a more powerful form of customization which modulates all the system and emission parameters.

In this approach, the parameters of each task are constrained to lie on a learned low dimensional manifold, indexed by the latent z. Our experiments show that this approach results in improved performance and/or greater data efficiency than existing approaches, as well as greater robustness to unfamiliar test inputs.

Further, varying z can generate a continuum of models, allowing interpolation between sequence predictions (see Figure 1b for an example), and potentially morphing of sequence characteristics over time.

Contributions In this paper we propose the MTDS, which goes beyond existing work by allowing full adaptation of all parameters of general dynamical systems via use of a learned nonlinear manifold.

We show how the approach may be applied to various popular models, and provide general purpose . . . . . .

learning and inference algorithms.

Our experimental studies use synthetic data (sum of two damped harmonic oscillators) and real-world human locomotion mocap data.

We illuminate various properties of the MTDS formulation in our experiments, such as data efficiency, user control, and robustness to dataset shift, and show how these go beyond existing approaches to time series modelling.

We finally utilize the increased user control in the context of mocap data to demonstrate style morphing.

To this end, we introduce the model in Section 2, giving examples and discussing the particular challenges in learning and inference.

We discuss the relation to existing work in Section 3.

Experimental setup and results are given in Section 4 with a conclusion in Section 5.

Consider a collection of input-output sequences D = {Y

Ti }, i = 1, . . .

, N , where T i denotes the length of sequence i. Each sequence i is described by a different dynamical system, whose parameter θ (i) depends on the hierarchical latent variable z (i) ∈ Z:

for t = 1, . . .

, T i .

The state variables X (i) = {x

Ti }, x t ∈ X follow the latent dynamics (2) starting from x 0 := 0 (other choices of initial state are possible).

See Figure 1a for a graphical model.

In this paper we assume Z = R k which the vector-valued function h φ (·) transforms to conformable model parameters θ ∈ R d , d k. Note that h φ may keep some dimensions of θ constant with respect to z. We call this a Multi-Task Dynamical System, going beyond the usage in Bird et al. (2019) .

In order to make the framework more concrete we will describe two general choices of the base model.

In what follows we will write each parameter with a subscript z to denote dependence on z (e.g. A z := A(z)) to reduce notational clutter.

The choice of p(z) and h φ will depend on the application, but a fairly general choice is a deep latent Gaussian model (Kingma & Welling, 2014; Rezende et al., 2014) .

See section A.1.1 in the supplementary material for further discussion.

For a given z, a multi-task linear dynamical system (MTLDS) can be described by:

w t ∼ N (0, R z ) , t ∼ N (0, S z ), with θ z = {A z , B z , b z , C z , D z , d z , R z , S z } = h φ (z).

The parameterization of θ z must satisfy the constraints of positive definite R z and S z and stable A z (i.e. A z 2 ≤ 1) for all z, hence projection methods such as in Siddiqi et al. (2008) are not applicable.

We choose an alternative formulation of the LDS, replacing the latent dynamics in eq. (4) by:

where Σ z is a diagonal matrix and Q z orthogonal with no loss of generality (proof in supp.

mat.).

Since Σ z Q z ≤ Σ z Q z = Σ z , stability can be enforced e.g. by Σ = diag{tanh (υ)} for some vector υ.

For more details see section A.1.2 in the supplementary material.

Due to the nonlinearity of an RNN, enforcing stability of A z is not strictly required (see e.g. Miller & Hardt, 2019, §4.4) , although bounding the spectral radius may be useful for learning (e.g. Pascanu et al., 2013) .

The dynamics of a multi-task RNN (MT-RNN) are described by:

Combined with the emission model of eq. (5), we have

If long-term dependencies are important we may consider an orthogonal transition matrix (parameterized as for the MTLDS) to create a multi-task version of the Orthogonal RNN (ORNN, Helfrich et al., 2018) .

The parameters φ of an MTDS can be learned from a dataset

The first term in the integrand, Naesseth et al. (2018) .

For clarity of exposition we only consider models with deterministic state, which extends to the MTRNN and MTLDS above (in the case w t = 0 for all t).

This also avoids the interaction effect with the choice of approximate marginalization over X.

Equation (8) also cannot be computed in closed form in general, and so we resort to approximate learning.

A natural choice is via the ELBO (see an alternative MCO approach for unsupervised tasks in Section A.1.3, supp.

mat.).

We write the ELBO of eq. (8) as:

where D KL is the Kullback-Leibler divergence and q λ (z|Y, U ) an approximate posterior for z. We can now optimize a lower bound of the marginal likelihood via arg max φ,λ

where low variance unbiased gradients of eq. (9) are available via reparameterization (Kingma & Welling, 2014; Rezende et al., 2014)

, where µ λ , s λ are inference networks (e.g. Fabius & van Amersfoort, 2015) .

It can be difficult to learn a sensible latent representation if the base model is a powerful RNN.

When each output can be identified unambiguously via the inputs preceding it, a larger ELBO can be obtained by the RNN learning the relationship without using the latent variable (see e.g. Chen et al., 2017) .

A useful heuristic for avoiding such optima is KL annealing (e.g. Bowman et al., 2016) .

In our experiments we perform an initial optimization without the KL penalty (second term in eq. 9), initializing s λ (Y, U ) to a small constant value.

For an unseen test sequence {Y , U }, the posterior predictive distribution is p(y t+1:T | y 1:t , u 1:T ) = Z p(y t+1:T | u 1:T , z) p(z | y 1:t , u 1:t ) dz, usually estimated via Monte Carlo.

The key quantity is the posterior over z, which may be approximated by the inference networks µ λ , s λ .

However, for novel test sequences, the inference networks may perform poorly and standard approximate inference techniques may be preferred.

For further discussion and a description of our inference approach, see sections A.1.4, A.1.5 in the supp.

mat.

We note that z may not require inference, for instance by using the posterior of a sequence in the the training set.

This may be useful for artistic control, style transfer, embedding domain knowledge or overriding misleading observations.

We also note that the latent code can be varied during the state rollout to simulate a task which varies over time.

A number of dynamical models following the 'bias customization' approach have been proposed recently.

Miladinović et al. (2019) and Hsu et al. (2017) propose models where the biases of an LSTM cell depend on a (hierarchical) latent variable.

Yingzhen & Mandt (2018) propose a dynamical system where the latent dynamics are concatenated with a time-constant latent variable.

In contrast, our MTDS model performs full parameter customization, and this on both the dynamics and emission distributions.

A number of other proposals may be considered specialized applications of the MTDS.

Bird et al. (2019) use a small deterministic nonlinear dynamical system for the base model whose parameters depend on z using a nonlinear factor analysis structure.

Spieckermann et al. (2015) use a small RNN as the base model, where the transition matrix depends on z via multilinear decomposition.

Lin et al. (2019) use a small stochastic nonlinear dynamical system with h φ a set of parameter vectors chosen discretely (or in convex combination) via z.

Controlling and customizing sequence prediction has received much attention in the case of video data.

As in the MTDS, these approaches learn features that are constant (or slowly varying) within a subsequence.

Denton & Birodkar (2017) and Villegas et al. (2017) propose methods for disentangling time-varying and static features, but do not provide a useful density over the latter, nor an obvious way to customize the underlying dynamics.

Tulyakov et al. (2018) use a GAN architecture where z factorizes into content and motion components.

Hsieh et al. (2018) force a parts-based decomposition of the scene before inferring the latent content z. However, as before, the dynamic evolution cannot be easily customized with these methods.

Hierarchical approaches for dynamical systems with time-varying parameters are proposed in Luttinen et al. (2014) (corresponding to non-stationary assumptions) and in Karl et al. (2017) (for the purposes of local LDS approximation).

These models, like the MTDS can adapt all the parameters, but are linear and correspond to single task problems.

Rangapuram et al. (2018) predict the parameters of simple time-varying LDS models directly via an RNN.

While this is a multi-task problem, it is assumed that all necessary variation can be inferred from the inputs U .

Multi-task GPs are commonly used for sequence prediction.

Examples include those in Osborne et al. (2008); Titsias & Lázaro-Gredilla (2011); Álvarez et al. (2012) ; Roberts et al. (2013) .

MTGPs however can only be linear combinations of (a small number of) latent functions, further, predictions depend critically upon often unknown mean functions, and inputs are not easily integrated.

Note that an MTDS with no inputs, an LDS base model, a linear-Gaussian prior over the emission parameters

We investigate the performance of the MTDS on two datasets.

The first experiment investigates the performance of the MTDS on synthetic data generated by linear superposition of damped harmonic oscillation (DHO).

The second experiment considers real-world mocap data for human locomotion.

Data The generative model for J oscillators with constant amplitudes γ and variable frequency and decay factors,

. .

, 80 for tasks i = 1, 2, . . .

, N .

The emission noise is distributed iid as Model We model the DHO data using an MTLDS with deterministic state X = R 4 and a k = 4 latent variable z. All LDS parameters were adapted via the latent z except D := 0 and the emission variance s 2 , which was learned.

For optimization, we use the MCO algorithm of section A.1.3.

This can obtain a tighter bound than the ELBO, and is useful to investigate convergence to the true model over increasing N .

We contrast this with a bias customization approach (e.g. Miladinović et al., 2019) , implemented similarly, but such that only the parameter b (eq. 6) depends on z. We also train a Pooled LDS, which is the standard approach, using the same parameters for all tasks, and a single-task (STL) LDS which is learned from scratch using Bayesian inference over all parameters for each task.

The Pooled-LDS was initialized using spectral methods (see Van Evaluation We assess how quickly and effectively the models can adapt to novel test sequences with a training set size of N = 2 1 , 2 2 , . . .

, 2 7 .

(The STL approach effectively uses N = 0.)

The test set comprises 20 additional sequences drawn from the generating distribution.

For an initial subsequence y 1:t , we estimate the predictive posterior p(y t+1:T |y 1:t ) for various t and assess the predictions via root mean squared error (RMSE) and negative log likelihood (NLL).

For MTL we use the Monte Carlo inference method described in supp.

mat.

A.1.5 and for STL we use Hamiltonian Monte Carlo (NUTS, Hoffman & Gelman, 2014) .

Each experiment is repeated 10 times to estimate sampling variance.

The results, shown in Table 1 and supp.

mat.

section A.2.2, show substantial advantage of using the MTLDS ('MT Full') over single-task or pooled approaches.

The MTLDS consistently outperforms the Pooled-LDS for all training sizes N ≥ 4.

Merely performing bias customization ('MT Bias') is insufficient to perform much better than a pooled approach.

An example of MTLDS test time prediction is shown in Figure 2 , with Figures 2c and 2d demonstrating effective generalization from the N = 4 training examples (Figure 2a ).

Even after 40 observations, the STL approach (which is capable of fitting each sequence exactly) does not significantly outperform the N = 4 MTLDS.

Furthermore, the runtime was approx.

1000 times longer since STL inference is higher dimensional and poorly conditioned, and requires a more expensive algorithm.

Note that with a larger training set size of N = 128, the MLTDS approaches the likelihood of the true model (Figure 7 , supp.

mat.).

Data The dataset consists of 31 sequences from Mason et al. (2018) (ca.

2000 frames average at 30fps) in 8 styles: angry, childlike, depressed, neutral, old, proud, sexy, strutting.

In this case the family of possible sequences corresponds to differing walking styles.

Each observation represents a 21-joint skeleton in a Lagrangian frame, y t ∈ R 64 where the root movement is represented by a smoothed component and its remainder.

Following Mason et al. (2018) we represent joints by their spatial position rather than their rotation.

We also provide inputs that an animator may wish to control: the root trajectory over the next second, the gait cycle and a boolean value determining whether the skeleton turns around the inside or outside of a corner.

See section A.3.1 in the supplementary materials for more details.

Model We use a recurrent 2-layer base model where the first hidden layer is a 1024 unit GRU (Cho et al., 2014) and the second hidden layer is a 128 unit standard RNN, follwed by a linear decoding layer.

The first-layer GRU does not vary with z, i.e. it learns a shared representation of the input sequence across all i. Explicitly, omitting index i, the model for a given z is: for t = 1, . . . , T .

The parameters are θ = {ψ 1 , ψ 2 , H, C, d} where ψ 1 and H are constant wrt.

z.

The matrix H ∈ R ×1024 ( < 1024) induces a bottleneck between layers, forcing z to explain more of the variance.

For our experiments, a small can be used (we use = 24).

The first layer GRU uses 1024 units since it was observed experimentally to produce smoother animations than smaller networks.

The second layer does not use a gated architecture, as gates appear to learn style inference more easily, and result in less use of z.

For learning, each sequence was broken into overlapping segments of length 64 (approx.

two second intervals), which allows z to vary across a sequence.

We learn the model using an open-loop objective, i.e. the y t are not appended to the inputs.

This forces the model to recover from its mistakes as in Martinez et al. (2017) , although unlike these approaches, we do not append predictions to the inputs either.

Our rationale is that the state captures the same information as the predictions, and while previous approaches required observations y 1:τ as inputs to seed the state, we can use the latent z. The model was optimized using the variational procedure in section 2.2, where a slower learning rate (by a factor of 10-50) for the first layer parameters (i.e. ψ 1 , H) usually resulted in a more descriptive z. We also found that standard variational inference for each z (i) worked better in general than using amortized inference.

For comparison, we implement a bias customization model ('MTBias') via a deterministic state version of Miladinović et al., 2019 , which follows eqs. (10)- (13) but only the RNN bias in eq. (12) is a function of z. We also implement a 1-layer and 2-layer GRU without the multi-task apparatus, which serves both as an ablation test and a competitor model (Martinez et al., 2017) on the new dataset.

Style inference is performed with the same network given an initial seed sequence y 1:τ .

We train these in closed-loop (i.e. traditional next step 'teacher forcing' criterion) and open-loop (Martinez et al., 2017) settings.

For baselines, we use constant predictions of (i) the training set mean and (ii) the last observed frame of the seed sequence ('zero-velocity' prediction).

We test the data efficiency of the MTDS by training the models on subsets of the original dataset.

Besides the models described above, 8 'single-task' versions of the GRU models are trained which only see data for a single style.

We use six training sets of approximate size 2 8 , 2 9 , 2 10 , 2 11 , 2 12 , 2 13 frames per style, where sampling is stratified carefully across all styles, and major variations thereof.

For all experiments, the model fit (MSE) is calculated from the same 32 held out sequences (each of length 64).

The results are shown in Figure 3a .

As expected, the MTDS, MTBias and Pooled models obtain 'multi-task' gains over STL approaches for small datasets.

However, the MTDS demonstrates much greater data efficiency, achieving close to the minimum error with only 7% of the dataset.

The MTBias model requires more than twice this amount to obtain the same performance, and the Pooled model requires more than four times this amount.

More details, as with all mocap experiments, can be found in supp.

mat.

section A.2.3.

We investigate how well the MTDS can generalize to novel sequence styles via use of a leave-one-out (LOO) setup, similar to transfer learning.

For each test style, a model is trained on the other 7 styles in the training set, and hence encounters novel sequence characteristics at test time.

We average the test error over the LOO folds as well as 32 different starting locations on each test sequence.

The results are given in Figure 3b .

We see that while the competitor (pooled) models perform well initially, they usually degrade quickly (worse for closed-loop models).

In contrast, the multi-task models finds a better customization which evidences no obvious worsening over the predictive interval.

Unlike pooled-RNNs, the MTDS and MTBias models can firstly perform correct inference of their customization, and secondly can 'remember' it over long intervals.

We note that all models struggle to customize the arms effectively, since their test motions are often entirely novel.

Customization to the legs and trunk is easier since less extrapolation is required (see animation videos linked in section A.4.1).

We investigate the control available in the latent z by performing style transfer.

For various inputs U (s1) from each source style s 1 , we generate predictions from the model using target style s 2 , encoded by z (s2) .

We use a classifier with multinomial outputs, trained on the 8 styles of the training set, to test whether the target style s 2 can be recognized from the data generated by the MTDS.

Figure 3c gives the classifier 'probability' for each target style s 2 , averaged over all the inputs {U (s1) : s 1 = s 2 }.

Successful style transfer should result in a the classifier assigning a high probability to the target style.

These results suggest that the prediction style can be well controlled by z (s2) in the case of the full MTDS, but the MTBias demonstrates reduced control for some (source, target) pairs.

See the videos linked in section A.4.1 for examples, and sec. A.2.3 for more details.

Qualitative investigation Qualitatively, the MTDS appears to learn a sensible manifold of walking styles, which we assess through visualization of the latent space.

A k = 2 latent embedding can be seen in Figure 4 where the z (i) for each training segment i is coloured by the true style label.

Some example motions are plotted in the figure.

The MTDS embedding broadly respects the style label, but learns a more nuanced representation, splitting some labels into multiple clusters and coalescing others.

These appear broadly valid, e.g. the 'proud' style contains both marching and arm-waving, with the latter similar to an arm-waving motion in the 'childlike' style.

This highlights the limitation of relying on task labels.

Visualizations such as Fig. 1b indicate that smooth style interpolation is available via interpolation in latent space.

We take advantage of this in the animations (linked from sec. A.4.1) by morphing styles dynamically.

In this work we have shown how to extend dynamical systems with a general-purpose hierarchical structure for multi-task learning.

Our MTDS framework performs customization at the level of all parameters, not just the biases, and adapts all parameters for general classes of dynamical systems.

We have seen that the latent code can learn a fine-grained embedding of sequence variation and can be used to modulate predictions.

Clearly good predictive performance for sequences requires task inference, whether implicit or explicit.

There are three advantages of making this inference explicit.

Firstly, it enhances control over predictions.

This might be used by animators to control the style of predictions for mocap models, or to express domain knowledge, such as ensuring certain sequences evolve similarly.

Secondly, it can improve generalization from small datasets since task interpolation is available out-of-the-box.

Thirdly, it can be more robust against changes in distribution at test time than a pooled model: (2014) is a unit Gaussian p(z) = N (0, I).

This choice allows simple sampling schemes, and straight-forward posterior approximations.

It is also a useful choice for interpolation, since it allows continuous deformation of its outputs.

An alternative choice might be a uniform distribution over a compact set, however posterior approximation is more challenging, see Svénsen (1998) for one approach.

Sensible default choices for h φ include affine operators and multilayer perceptrons (MLPs).

However, when the parameter space R d is large, it may be infeasible to predict d outputs from an MLP.

Consider an RNN with 100k parameters.

If an MLP has m L−1 = 300 units in the final hidden layer, the expansion to the RNN parameters in the final layer will require 30×10 6 parameters alone.

A practical approach is to use a low rank matrix for this transformation, equivalent to adding an extra linear layer of size m L where we must have m L m L−1 to reduce the parameterization sufficiently.

Since we will typically need m L to be O(10), we are restricting the parameter manifold of θ to lie in a low dimensional subspace.

Since MLP approaches with a large base model will then usually have a restricted final layer, are there any advantages over a simple linear-Gaussian model for the prior p(z) and h φ ?

There may indeed be many situations where this simpler model is reasonable.

However, we note some advantages of the MLP approach:

1.

The MLP parameterization can shift the density in parameter space to more appropriate regions via nonlinear transformation.

2.

A linear space of recurrent model parameters can yield highly non-linear changes even to simple dynamical systems (see e.g. the bifurcations in §8 of Strogatz, 2018).

We speculate it might be advantageous to curve the manifold to avoid such phenomena.

3.

More expressive choices may help utilization of the latent space (e.g. Chen et al., 2017) .

This may in fact motivate moving beyond a simple MLP for the h φ .

The matrices A, B, R, S of the MTLDS can benefit from specific parameterizations, which we will discuss in turn.

Degeneracy of LDS.

It will be useful to begin with the well-known over-parameterization of linear dynamical systems.

The hidden dynamics of a LDS can be transformed by any invertible matrix G while retaining the same distribution over the emissions Y .

This follows essentially because the basis used to represent X is arbitrary.

The distribution over Y is unchanged under the following parameter transformations:

Parameterization of A. The stability constraint,

is equivalent to ensuring that the singular values of A lie within the unit hypercube (since singular values are non-negative).

Let A = U ΣV T be the singular value decomposition (SVD) of A. Now we have from the previous result that if an LDS has latent dynamics with transition parameter A, we may replace the dynamics under the similarity transform G −1 AG.

Choose G = U , i.e. the left singular values of A, and hence A = ΣV T U =: ΣQ for some orthogonal matrix Q. This follows from the closure of the orthogonal group under multiplication, which is easily verified.

Note that in choosing this transformation, no additional constraints are placed on the other parameters in the LDS.

Orthogonal matrices can be parameterized in a number of ways (see e.g. Khuri et al., 1989) .

A straight-forward choice is the Cayley transform.

From Khuri et al. (1989) : "if Q is an orthogonal matrix that does not have the eigenvalue -1, then it may be written in Cayley's form:

where S is skew-symmetric".

In order to permit negative eigenvalues, we can pre-multiply by a diagonal matrix E with elements in {+1, −1}. Since we then have A = ΣEQ, E can be absorbed into Σ, and so the stability constraint (15) can be satisfied with the parameterization A = ΣQ where Σ is a diagonal matrix with elements in [−1, +1] and Q is a Cayley-transform of a skew-symmetric matrix.

This follows from the overparameterization of the LDS, and we emphasise that the system equations (4) and (6) are not equivalent, but any LDS distribution over Y can be written with latent dynamics of the form (6).

Choose G = κ −1 I in eq. (14).

It may be observed that the scale κ of the latent system can be chosen arbitrarily without affecting A. We wish to avoid such degeneracies in a hierachical model, since we may otherwise waste statistical strength and computation on learning equivalent representations.

We can remove this by fixing the scale of B. An indirect but straightforward approach is to upper bound the magnitude of each element of B. ForB predicted by h φ (z) we might choose the transformation B = tanh(B) where tanh acts element-wise.

If a sparse B is desired, one can use an over-parameterization of two matricesB 1 ,B 2 , and choose B = σ(B 1 ) • tanh(B 2 ), where • is element-wise multiplication, and σ a logistic sigmoid.

The former parameterization is unlikely to find a sparse representation since the gradient of tanh is greatest at 0.

Parameterization of R, S. The covariance matrices R, S must be in the positive definite cone.

Where a diagonal covariance will suffice, any parameterization for enforcing positivity can be used, such as exponentiation, squaring or softplus.

A number of parameterizations are available for full covariance matrices (see Pinheiro & Bates, 1996) .

A simple choice is to decompose the matrix, say R = LL T , where L is a lower triangular Cholseky factor.

As before, it is useful to enforce uniqueness, which can be done by ensuring the diagonal is positive.

We provide an alternative learning algorithm to the VB approach in section 2.2 which obtains a tighter lower bound.

This was important for the DHO experiments in order to monitor convergence to the true model.

The below is perhaps a novel approach for learning in unsupervised cases (i.e. where U = ∅), but cannot be performed efficiently for supervised problems without modification.

Monte Carlo Objectives (MCOs, Mnih & Rezende, 2016 ) construct a lower bound for marginal likelihoods via a transformation of an appropriate Monte Carlo estimator.

Specifically we consider the logarithmic transformation of:

m = 1, . . . , M ; an importance sampling estimator for p(Y ).

Using Jensen's inequality, we show that the following is a lower bound on the log marginal likelihood:

where p(z 1:M ) := p(z 1 )...p(z M ).

The tightness of the bound can be increased by increasing the number of samples M (Burda et al., 2016) .

Assuming p(z) has been re-parameterized (Kingma & Welling, 2014) to be parameter-free, we can easily calculate the gradient (if not, see Mnih & Rezende, 2016) .

By exchanging integration and differentiation, we can calculate the gradient as:

Note that eq. (22) is an importance sampled version of the Fisher identity.

We might expect this estimator to suffer from high variance, since the prior is a poor proposal for the posterior.

However, the prior should not be a poor proposal for the aggregate posterior, i.e. Seth et al., 2017) .

In fact, importance sampling from the prior may serve as a useful bias in this case, attracting the posterior distributions which have a large D KL (p(z | Y (i) )||p(z)) towards the prior.

Our observation is that sampling from the prior can be amortized over each sequence Y (i) , i = 1, . . .

, N .

Specifically, for each particle z m , the dynamics (2), (3) can be run forward once to calculatê Y m , from which the likelihood Y (i) , for all tasks i = 1, . . .

, N can be calculated inexpensively.

The amortized cost of taking M samples (e.g. M ∈ O(10 3 )) now becomes M/N , which may be relatively small.

We can also take advantage of low-discrepancy random variates such as Sobol sequences (Lemieux, 2009) to reduce variance.

We propose that each sequence i resamples a small number M rsmp ≤ 5 of particles from the importance weights for each i to reduce the cost of backpropagation (a similar resampling scheme is suggested in Burda et al., 2016) .

See Algorithm 1.

In the supervised case (i.e. where each observation Y (i) has a different input U (i) ), running the dynamics forward from a particle z m can no longer be amortized over all {Y (i) } since the prediction

We can therefore only amortize the parameter generation θ = h φ (z), which is often less expensive than running the dynamics forward.

For this reason Algorithm 1 is primarily restricted to unsupervised problems.

A hybrid approach would essentially result in the importance weighted autoencoder (IWAE) of Burda et al. (2016) .

Inference at test time can be performed by any number of variational or Monte Carlo approaches.

As in the main text, our focus here is on deterministic state dynamical systems.

For stochastic state models, additional reasoning similar to Miladinović et al. (2019) will be required.

A gold standard of inference over z may be the No U-Turn Sampler (NUTS) of Hoffman & Gelman (2014) (a form of Hamiltonian Monte Carlo), provided k is not too large and efficiency is not a concern.

However, given the sequential nature of the model, it is natural to consider exploiting the posterior at time t for calculating the posterior at time t + 1.

Bayes' rule suggests an update of the

end end Optimize(optimizer, φ, g); end end following form:

following the conditional independence assumptions of the MTDS.

This update (in principle) incorporates the information learned at time t in an optimal way, and further suggests a constant time update wrt t. However, evaluation of p(y t+1 | u 1:t+1 , h φ (z)) usually scales linearly with t, since the state x t+1 must be calculated recursively from x 0 given z and u 1:t+1 .

Nevertheless, sequential incorporation of previous information will perform a kind of annealing (Chopin, 2002) which reduces the difficulty, and hopefully the runtime of inference at each stage.

We first provide some background of the difficulties of such an approach, looking first at Monte Carlo (MC) methods.

Naïve application of Sequential Monte Carlo (SMC) will result in severe particle depletion over time.

To see this, let the posterior after time t be p(z | y 1:t , u 1:t ) = 1 M M m=1 w m δ(z− z m ).

Then the updated posterior at time t + 1 will be:

, simply a re-weighting of existing particles.

Over time, the number of particles with significant weights w m will substantially reduce.

But since the model is static with respect to z (see Chopin, 2002) , there is no dynamic process to 'jitter' the {z m } as in a typical particle filter, and hence a resampling step cannot improve diversity.

Chopin (2002) discusses two related solutions: firstly using 'rejuvenation steps' (cf.

Gilks & Berzuini, 2001 ) which applies a Markov transition kernel to each particle.

The downside to this approach is the requirement to run until convergence; and the diagnosis thereof, which can result in substantial extra computation.

One might instead sample from a fixed proposal distribution (accepting a move with the usual Metropolis-Hastings probability) for which convergence is more easily monitored.

A Sequential Monte Carlo sampler approach (Del Moral et al., 2006) may be preferred, which permits local moves, and can reduce sample impoverishment via resampling (similar to SMC).

However, the approach requires careful choices of both forward and backward Markov kernels which substantially reduces its ease of use.

A well-known variational approach to problems with the structure of eq. (23) is assumed density filtering (ADF, see e.g. Opper & Winther, 1998) .

For each t, ADF performs the Bayesian update and the projects the posterior into a parametric family Q. The projection is done with respect to the reverse KL Divergence, i.e. q t+1 = arg min q∈Q D KL p(z | y 1:t+1 , u 1:t+1 ) || q .

Intuitively, the projection finds an 'outer approximation' of the true posterior, avoiding the 'mode seeking' behaviour of the forward KL, which is particularly problematic if it attaches to the wrong mode.

Clearly the performance of ADF depends crucially on the choice of Q. Unfortunately, where Q is expressive enough to capture a good approximation, the optimization problem will usually be challenging, and must resort to stochastic gradient approaches, resulting in an expensive inner loop.

Furthermore, when the changes from q t to q t+1 are relatively small, the gradient signal will be weak, resulting perhaps in misdiagnosed convergence and hence accumulation of error over increasing t. A recent suggestion of Tomasetti et al. (2019) is to improve efficiency via re-use of previous (stale) gradient evaluations.

Standard variance reduction techniques may also be considered to improve convergence in the inner loop.

In our experiments, we found sampling approaches faster and more reliable for each update, as well as providing diagnostic information, and so we eschew variational approaches.

(Our experiments used a fairly small k (≤ 10); variational approaches may be preferred in higher dimensional problems.)

Specifically we use iterated importance sampling (IS) to update the posterior at each t. The key quantity for IS is the proposal distribution q prop : we need a proposal that is well-matched to the target distribution.

Our observation is that the natural annealing properties of the filtering distributions (eq. 23) allow a slow and reliable adaptation of q prop .

In order to capture complex multimodal posteriors, we parameterize q prop by a mixture of Gaussians (MoG).

For each t, the proposal distribution is improved over N AIS iterations using adaptive importance sampling (AdaIS), described for mixture models in Cappé et al. (2008) .

We briefly review the methodology for a target distribution p * .

Let the AdaIS procedure at the nth iteration use the proposal:

α j ∈ R + s.t.

For our experiments, this approach worked robustly and efficiently, and appears superior to the alternatives discussed.

Unlike SMC, we obtain a q prop which is a good parameteric approximation of the true posterior.

We therefore avoid the sample impoverishment problem discussed above (eq. 25).

Due to the small number of iterations of AdaIS required (usually ≤ 5 for our problems), it is substantially faster than MCMC moves, and since stochastic gradients are avoided, convergence is much faster than variational approaches.

The scheme benefits from the observed fast initial convergence rates of the EM algorithm (see e.g. Xu & Jordan, 1996) , particularly since early stopping can be used for the initial iterates.

In practice, one may not wish to calculate a posterior at every t, but instead intervals of length τ .

In our DHO experiments (k = 4) we use τ = 5, and usually have ESS > 0.6M after n = 4 inner iterations, with total computation per q t requiring 250-300ms on a laptop.

We observe in our experiments that posteriors are often multimodal for t ≤ 20 and sometimes beyond, motivating the MoG parameterization.

In these experiments, the MoG appears to capture the salient characteristics of the target distribution well.

Note as in section A.1.3, Sobol or other low-discrepancy sequences may be used to reduce sampling variance from q prop .

Under review as a conference paper at ICLR 2020 for t = 1, . . .

, T , z ∈ R 4 , x t ∈ R 4 , suppressing task index i for clarity.

Define x 0 := 0, and for all tasks u = [1, 0, 0, 0, . . .].

A is parameterized as discussed in section A.1.2 using a product of a diagonal and orthogonal matrix ΣQ.

The diagonal of Σ is constrained to lie in [−1, +1] using the tanh function, and Q is parameterized by the Cayley transform of a skew symmetric matrix S. Using an upper triangular matrix Γ, we have S = Γ − Γ T , and Q = (I − S)(I + S) −1 .

We parameterize B via the product of logistic sigmoid and tanh functions as in section A.1.2 in order to learn a sparse parameterization.

C is unconstrained, and the parameter s is optimized as a constant wrt.

z. The STLDS is parameterized in the same way.

The prior p(z) is a unit Gaussian distribution, and h φ is a 2 hidden-layer neural network.

We use a fixed feature extractor

T in the first layer in order to help encode a rectangular support within a spherically symmetric distribution.

The second layer is a fully-connected 300 unit layer with sigmoid activations.

Learning.

The output of an MTLDS is very sensitive to the parameter A, and care must be taken to avoid divergence during optimization.

The diagonal-orthogonal parameterization greatly helped to stabilize the optimization over a more naïve orthogonal-diagonal-orthogonal SVD parameterization.

We also reduced the learning rate by a factor of 10 for A. It proved useful to artificially elevate the estimate of s during training using a prior log s ∼ N (−1.5, 0.05) (derived from preliminary experiments) since the MTDS can otherwise overfit small datasets (see also discussion in §3, Svénsen, 1998), with associated instability in optimization.

The learning rate schedule is given in Table 2 , for which the prior over log s ∼ N (m, 0.05) was annealed from m = −1.0 to m = −1.5.

The "momentum" parameter β 1 (c.f.

Kingma & Ba, 2014) is also reduced at the end of optimization.

The latter was motivated by oscillation and moderate deviations observed near optima, apparently caused (upon investigation) by strong curvature of the loss surface.

Inference.

The latent z are inferred online using the adaptive IS scheme of section A.1.5.

We also perform inference over log s since it is held artificially high for optimization, and its true optimal value is not known.

An informative prior close to the learned value log s ∼ N −2.0, 0.1 2 , was Epoch η β 1 log s mean M 1 8e-4 0.9 -1.0 1 000 200 8e-4 0.9 -1.3 1 000 600 4e-4 0.9 -1.5 2 000 1000 2e-4 0.8 -1.5 4 000 nevertheless used since the posterior was sometimes approximately singular, causing high condition numbers in the estimated covariance matrix of the proposal.

The hyperparameters are given in Table  3 .

These parameters did not require tuning as for optimization, but were sensible defaults.

These also seem to work well without tuning for other experiments such as the Mocap data.

Each posterior for a given time t took on average approx.

0.3 seconds.

We used the No U-Turn Sampler (Hoffman & Gelman, 2014) for the STL experiments due to poor conditioning and the higher complexity and dimensionality of the posterior (19 dimensions).

Tuning is performed using ideas from Hoffman & Gelman (2014, Algorithm 4, 5) , and the mass matrix is estimated from the warmup phase.

2 The warmup stage lasted 1000 samples and the subsequent 600 samples were used for inference.

Each sampler was initialized from a MAP value, obtained via optimization with 10 random restarts.

For both MAP optimization and sampling, we found it essential to enforce a low standard deviation (we used log s = −2 and log s ∼ N −2, 0.2 2 respectively) similarly to the MTL experiments.

The autocorrelation-based effective sample size (Gelman et al., 2013, ch.

11.5 ) typically exceeds 100 for each parameter.

Each posterior for a given time t took on average approx.

300 seconds.

Note that as discussed in section A.1.4, unlike our AdaIS procedure, we cannot make much re-use of previous computation here.

The average results (over the 10 repetitions) are given in Table 4 , which extends Table 1 in the main text with the NLL results.

The distribution of these results can be seen in the violin plots of Figure 8 .

The RMSE results of the MTLDS are all significantly better than both the pooled and single-task models according to a Welch's t-test and Mann-Whitney U-test, except for MTLDS-4 at t = 40.

The latter is significantly better than the pooled model, but is indistinguishable from the STLDS at the level α = 0.05.

We also consider the convergence of the MTLDS to the true model with increasing N .

For each experiment, we average the log marginal likelihood of the test sequences estimated via 10 000 (Sobol) samples from the prior.

As before, the prior should be a good proposal for the aggregate posterior, and we amortize the same samples over all test sequences.

In order to interpret the difference to the true distribution log p * (Y test ) − log p(Y test | φ), we use the Bayes Factor interpretations given by Kass & Raftery (1995) .

For instance a difference of 1.0 is 'barely worth mentioning', but a difference of 4.0 is 'strong evidence' that the distributions are different.

We average over 10 000 test examples to avoid sampling variation of the test set.

Figure 7 show boxplots of the log marginal likelihood for each model over increasing N , where the boxes show the interquartile range (IQR) over the 10 repetitions.

We see convergence towards the true value with increasing N , with the difference of the MTLDS-128 'barely worth mentioning'.

Figure 9a , which is a subset of the CMU skeleton.

Representation in observation space.

We choose a Lagrangian representation (Figure 9c ) where the coordinate frame is centered at the root joint of the skeleton (joint 1 in Fig. 9a , the pelvis), projected onto the ground.

The frame is rotated such that the z-axis points in the "forward" direction, roughly normal to the body.

This is in contrast to the Eulerian frame (Figure 9b ) which has an absolute fixed position for all t. In the Lagrangian frame, the joint positions are always relative to the root joint, which avoids confusing the overall trajectory of the skeleton (typified by the root joint), and the overall rotation of the skeleton, with the local motions of the joints.

The relative joint positions can be represented by spatial position or by joint angle.

For the latter, the spatial positions of all joints can be recovered from the angle made with their parent joint via use of forward kinematics (FK).

This construction ensures the constant bone length of the skeleton over time, which is a desirable property.

However, it also substantially increases the sensitivity of internal joints.

For instance, the rotation of the trunk will disproportionately affect the error of the joints in both arms.

For this reason, we have chosen to model the spatial position of joints, which may result in violations of bone length, but avoids these sensitivity issues.

See also §2.1 Pavllo et al. (2018) .

One can further encode the joint positions via velocity (i.e. differencing) which may result in smoother predictions.

We avoid this encoding for the local joint motion (joints 2 to 21) since it can suffer from accumulated errors, but we do use it to predict the co-ordinate frame as is standard in mocap models.

Hence our per-frame representation consists of the velocityẋ,ż,ω of the co-ordinate frame, the relative vertical position of the root joint, and 3-d position of the remaining 20 joints, which gives y t ∈ R 64 .

Choice of inputs.

Our choice of inputs will reflect controls that an animator may wish to manipulate.

The first input will be the trajectory that the skeleton is to follow.

As in Holden et al. (2017b), we provide the trajectory over the next second (30 frames), sampled uniformly every 5 frames.

Unlike previous work, there is no trajectory history in the inputs since this can be kept in the recurrent state.

The (2-d) trajectory co-ordinates are given wrt.

the current co-ordinate frame, and hence can rotate rapidly during a tight corner.

In order to provide some continuity in the inputs, we also provide a first difference of the trajectory in Eulerian co-ordinates.

Table 5 : Hyper-parameters of mocap models.

η denotes the learning rate.

The velocity implied by the differenced trajectory does not disambiguate the gait frequency vs. stride length.

The same motion might be achieved with fast short steps, or slower long strides.

We therefore provide the gait frequency via a phasor (as in Holden et al., 2017b) , whose frequency may be externally controlled.

This is provided by sine and cosine components to avoid the discontinuity at 2π.

A final ambiguity exists from the trajectory at tight corners: the skeleton can rotate either towards the focus of the corner, or towards the outside.

Figure 9d demonstrates the latter, which appears not infrequently in the data.

We provide a boolean indicator alongside the trajectory which identifies corners for which this happens.

Altogether we have u t ∈ R 32 : 12 inputs for the Lagrangian trajectory, 12 inputs for the differenced Eulerian trajectory, 2 inputs for the gait phase and 6 inputs for the turning indicators.

Extracting the root trajectory.

The root trajectory is computed by projecting the root joint onto the ground.

However, this projection may still contain information about the style of locomotion, for instance via swaying.

We wish to remove all such information, since a model can otherwise learn the style without reference to a latent z. Our goal is to find an appropriately smoothed version of the extracted trajectory T .

We use a cubic B-spline fit to control points fitted to the 'corners' of the trajectory.

These control points are selected using a polygonal approximation to T using the Ramer-Douglas-Peucker algorithm (RDP, e.g. Ramer, 1972) .

Briefly, the RDP algorithm uses a divide-and-conquer approach which greedily chooses points that minimize the Hausdorff distance of T to the polygonal approximation.

Some per-style tuning of the RDP parameter, and a small number of manually added control points rendered this a semi-automatic process.

Extracting the gait phase.

Foot contacts are calculated via the code used by Holden et al. (2017b) , which is based on thresholding of vertical position and velocity of each foot.

As in Mason et al. (2018) we check visually for outliers and correct misclassified foot contacts manually.

The leading edge of each foot contact is taken to represent 0 (left) and π (right), and the gait phase is calculated by interpolation.

In this section we discuss elements of the experimental setup, learning and inference common to all experiments.

Details particular to each experiment can be found in the following section.

Further Model Details The MTDS architecture is described in section A.3, aside from the choice of prior.

We tested both linear and nonlinear h φ in preliminary experiments and the performance was often similar.

The nonlinear version used a one hidden layer MLP with 300 hidden units with tanh activations.

For the final affine layer, we used a rank 30 matrix which, chosen pragmatically as a trade-off between flexibility and parameter count (see discussion in section A.1.1).

Both choices often performed similarly, however the linear approach was chosen, since optimization of the latent z on new data was faster, and apparently more robust to choice of initialization.

A nonlinear h φ may be more important when the base model is simpler.

The benchmark models use an encoding length of τ = 64 frames.

The encoder shares parameters with the decoder, i.e. the RNN is simply 'warm started' for 64 frames before prediction.

The benchmark models, unlike the MTDS, predict the difference from the previous frame (or 'velocity') via a residual architecture, as this performs better in Martinez et al. (2017) .

Further Learning Details Our primary goal was qualitative: to obtain good style-content separation, high quality animations and smooth interpolation between sequences.

Therefore hyperparameter selection for the MTDS proceeded via quantitative means (via the ELBO) and visual inspection of the qualitative criteria.

The qualitative desiderata motivated split learning rates between shared and multi-task networks (cf. section 4.2), and the amount of L2 regularization.

See Table 5 for the chosen values.

The main learning rate η applies to the fixed parameters wrt.

z (i.e. ψ 1 , H), and the multi-task learning rate applies to the parameter generation parameters φ and inference parameters λ.

Standard variational inference proved more reliable than amortized inference: we used a Gaussian with diagonal covariance (parameterized using softplus) for the variational posterior over each z. L2 regularization was applied to φ, ψ 1 , H.

Unless otherwise specified, we optimized each model using a batch size N batch = 16 for 20 000 iterations.

The ELBO had often reached a plateau by this time, and training even longer resulted in a worse latent representation at times (as evidenced through poor style transfer).

As noted in the main text, we remove the KL penalty of eq. (9) for the initial 2 000 iterations, and enforce a small posterior standard deviation (s λ = 10 −3 ) for the same duration.

This is similar to finding a MAP estimate for the {z}. For the remaining iterations, the original ELBO criterion is used, and the constraint on s λ is removed.

The model is implemented in PyTorch (Paszke et al., 2017) and trained on GPUs.

Since we use a fairly small max.

sequence length L = 64, truncated backpropagation through time was not necessary.

The hyper-parameters for the benchmark models were found (Table 5 ) using a grid search over learning rate and regularization, as well as the optimizers {Adam, (vanilla) SGD}. We performed the search over the pooled data for all 8 styles, with a stratified sample of 12.5% held out for a validation set.

Once the hyperparameters were chosen, benchmark models were also trained for 20 000 iterations, recording the validation error every 1 000 iterations on a stratified 12.5% held out sample.

The model with the lowest validation error during optimization is chosen.

We standardize the data so that when pooled, each dimension has zero mean and unit variance.

Finally, note that as discussed in section A.3.1, the data are represented in Lagrangian form, therefore drifts in the predicted trajectory from the true one are not necessarily heavily penalized.

This can be altered by changing the weights on the root velocities, but we did not do this.

Inference.

At test time, especially for experiment 2, we cannot expect amortized inference to perform optimally, and we consider standard inference techniques.

We want to understand the nature of the posterior distributions, and so we again used the AdaIS approach of section A.1.5.

In practice, each posterior was unimodal and approximately Gaussian.

Furthermore, the variation in sequence space for different z in the posterior was usually fairly small, and the posterior predictive mean performed similarly to using a point estimate.

Each observation from which z is inferred is of size 64 × 64 and hence the posterior is fairly concentrated.

Unlike the DHO model, this is a more expensive procedure.

Our k = 3 experiments took approx.

24 seconds per observation for inference.

An optimization approach using standard techniques may be expected to perform similarly at a reduced computational cost.

Hence unless otherwise specified, inference was done via optimization.

Experiment 1 -MTL The training data for each style uses 4 subsequences chosen carefully to represent the inter-style variation.

Obviously it is important that frames are consecutive rather than randomly sampled.

Over the increasing size training sets, each of these subsequences is a superset of the previous one.

The 6 training set sizes (2 8 , 2 9 , 2 10 , 2 11 , 2 12 , 2 13 frames per style) are not exact since short subsequences are discarded (e.g. at file boundaries), and the largest set contains all the training data except the test set 3 , where data are not evenly distributed over styles.

The test set comprises 4 sequences from each style, each of length 64, and is the same for all experiments.

A length-64 seed sequence immediately preceding each test sequence was used for inference for all models.

The models are trained as described above, except for the single task (STL) models.

The STL models use an identical architecture to the pooled 1-layer GRU models, except they are trained only on the data for their style.

Since there is less data for these models, we train them for a Table 6 : Mocap Experiment 1 (MTL): predictive MSE for length-64 predictions where training sets are a given fraction of the original dataset.

maximum of 5 000 iterations.

We do not train 2-layer GRUs, since the amount of data is small for most experiments.

The full results are given in Table 6 .

We use fractions of the dataset instead of absolute training set sizes to aid understanding.

The performance of the MTDS appears to increase with larger k, and suggests that we need k > 3 to achieve optimal performance on unseen training data.

The results demonstrate substantial benefit of the MTDS over a pooled RNN model in terms of sample efficiency, but not in asymptotic performance, as might be expected.

According to a paired t-test, the improvements of the k = 7 MTDS over the (1-layer, open loop) pooled GRU are significant for training set sizes 3%, 7%, 13% and 27%.

4 At a style level, the k = 7 MTDS performs at least as well as the pooled GRUs for the first four training set sizes.

See Figure 10 .

Note that the 'angry' and 'childlike' styles appear to be harder than the others, most likely due to their relatively high speed.

For example animations of the MTL experiments, see the linked video in section A.4.1.

Table 7 provides the aggregate results of experiment 2 for each of the mocap models.

A visualization is given in Figure 11 .

The 2-layer competitors are shown here for completeness, but they achieve similar performance to the 1-layer models on aggregate.

Figure  12 provides a breakdown of these results on a per-style basis.

Styles 5-8 appear to be easier from the point of view of the benchmarks, but the MTDS shows equal or better performance on all styles except style 5.

The competitor results achieve better short-term performance than the MTDS.

However, note that the zero-velocity baseline performs similarly to the open-loop GRUs for the first 5 predictions.

This suggests that the MTDS may be improved for these early predictions simply by interpolating from the zero-velocity baseline for small values of t. We are unable to conclude from these experiments that the benchmark models can represent the style better initially, but simply that they can smooth the transition from the seed sequence better.

The classifier is learned on the original observations to distinguish between the 8 styles.

We use a 512-unit GRU to encode an observation sequence (usually of 64 frames), and transform the final state via a 300-unit hidden layer MLP with sigmoid activations into multinomial emissions.

The model is trained via cross-entropy with 20% of the training data held out as a validation set; training was stopped as the validation error approached 0.

We perform a standardization of the gait frequency across all styles, since some styles can be identified purely by calculating the frequency.

The mean frequency across all styles (1 cycle per 33 frames) is applied to all sequences via linear interpolation.

In this we make use of the instantaneous phase given in the inputs.

We use a k = 8 latent code for the MTDS as the model is trained on all styles.

for each source style s 1 , with examples j = 1, . . . , 4.

We next seek the 'archetypal' latent code z associated with each target style s 2 .

For each s 2 , we optimize z over 20 candidate values, obtained from the posterior mean of the style s 2 in the training set.

Data are generated from all {U (s1) j } and the z which provides the greatest success in style transfer is chosen.

The 32 highly varied input sequences guard against overfitting -the 'archetypal' codes for each style must perform well across much of the variety of the original dataset.

We provide a scalar measurement of the 'success' of style transfer for each pair (s 1 , s 2 ) by using the resulting 'probability' that the classifier assigns the target style s 2 , averaged across the four input sequences for the source s 1 .

The results of these experiments are shown in Figure 13a for the model with multi-task bias, and Figure 13b shows the results for the full MTDS.

Table 3c in the main text gives the marginal of these results wrt.

the target style.

The cells in Figure 13 give the classifier probability for the target style for each (source, target) combination, averaged over the four source inputs.

Successful style transfer should result in a the classifier assigning a high score in every cell of the table.

For most (source, target) pairs, the full MTDS model substantially outperforms the MTBias model: it appears that MTDS can control the prediction well in the majority of cases, and the MTBias model offers reduced control in general.

However, we observe for both models that it is more difficult when styles are associated with extremes of the input distribution.

Specifically, both the 'childlike' and 'angry' styles have unusually high speed inputs, and the 'old' style has unusually low speeds.

Note that in order to provide style transfer, the models are mostly ignoring these correlations, even though they are very useful for prediction.

Further improvements may be available, perhaps by using an adversarial loss, or applying domain knowledge to the model.

This is orthogonal to our contribution, and we leave this to future work.

Providing style transfer from all varieties of source style is a challenging task.

For instance, some styles include sources with widely varying speeds and actions, which may be mismatched to the target style.

To understand what may be more typical use of the model, we provide an easier variant of this experiment where only one example of each source style is provided, rather than four.

Note nevertheless that the same z (s2) is still used across all sources s 1 .

The results of this secondary experiment are provided in Figure 14 .

In this case, style transfer is successful for almost all (source, target) pairs in the case of the MTDS, except for the angry style.

The MTBias model still has many notable failures.

1.

In-sample predictions https://vimeo.com/362069486.

The goal is to showcase the best possible performance of the models by predicting from inputs in the training set.

2.

MTL examples https://vimeo.com/362122944.

Examples from Experiment 1.

We compare the quality of animations and fit to the ground truth for two limited training set sizes (6.7% and 13.3% of the full data).

For both models, MSE to the ground truth is given, averaged over the entire predictive window (length 256).

This is different to the experimental setup which uses only the first 64 frames.

3.

Novel test examples https://vimeo.com/362068342.

Examples from Experiment 2.

We show the adaptions obtained by each model to novel sequences, in particular showcasing examples of the pooled GRU models inferring suboptimal styles wrt.

MSE.

Again, MSE to the ground truth is given averaged over the predictive window (length 256).

4.

Style morphing https://vimeo.com/361910646.

This animation demonstrates the effect of changing the latent code over time.

This also demonstrates style transfer and style interpolation from experiment 3.

For style morphing, we found it useful to fix the dynamical bias of the second layer (parameter b in eq. 7) wrt.

z since it otherwise resulted in 'jumps' while interpolating between sequences.

We speculate that shifting the bias induces bifurcations in the state space, whereas adapting the transition matrix allows for smooth interpolation.

@highlight

Tailoring predictions from sequence models (such as LDSs and RNNs) via an explicit latent code.