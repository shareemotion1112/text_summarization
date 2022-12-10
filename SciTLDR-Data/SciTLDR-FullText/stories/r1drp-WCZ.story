Long Short-Term Memory (LSTM) is one of the most powerful sequence models.

Despite the strong performance, however, it lacks the nice interpretability as in state space models.

In this paper, we present a way to combine the best of both worlds by introducing State Space LSTM (SSL), which generalizes the earlier work \cite{zaheer2017latent} of combining topic models with LSTM.

However, unlike \cite{zaheer2017latent}, we do not make any factorization assumptions in our inference algorithm.

We present an efficient sampler based on sequential Monte Carlo (SMC) method that draws from the joint posterior directly.

Experimental results confirms the superiority and stability of this SMC inference algorithm on a variety of domains.

State space models (SSMs), such as hidden Markov models (HMM) and linear dynamical systems (LDS), have been the workhorse of sequence modeling in the past decades From a graphical model perspective, efficient message passing algorithms BID34 BID17 are available in compact closed form thanks to their simple linear Markov structure.

However, simplicity comes at a cost: real world sequences can have long-range dependencies that cannot be captured by Markov models; and the linearity of transition and emission restricts the flexibility of the model for complex sequences.

A popular alternative is the recurrent neural networks (RNN), for instance the Long Short-Term Memory (LSTM) BID14 ) which has become a standard for sequence modeling nowadays.

Instead of associating the observations with stochastic latent variables, RNN directly defines the distribution of each observation conditioned on the past, parameterized by a neural network.

The recurrent parameterization not only allows RNN to provide a rich function class, but also permits scalable stochastic optimization such as the backpropagation through time (BPTT) algorithm.

However, flexibility does not come for free as well: due to the complex form of the transition function, the hidden states of RNN are often hard to interpret.

Moreover, it can require large amount of parameters for seemingly simple sequence models BID35 .In this paper, we propose a new class of models State Space LSTM (SSL) that combines the best of both worlds.

We show that SSLs can handle nonlinear, non-Markovian dynamics like RNNs, while retaining the probabilistic interpretations of SSMs.

The intuition, in short, is to separate the state space from the sample space.

In particular, instead of directly estimating the dynamics from the observed sequence, we focus on modeling the sequence of latent states, which may represent the true underlying dynamics that generated the noisy observations.

Unlike SSMs, where the same goal is pursued under linearity and Markov assumption, we alleviate the restriction by directly modeling the transition function between states parameterized by a neural network.

On the other hand, we bridge the state space and the sample space using classical probabilistic relation, which not only brings additional interpretability, but also enables the LSTM to work with more structured representation rather than the noisy observations.

Indeed, parameter estimation of such models can be nontrivial.

Since the LSTM is defined over a sequence of latent variables rather than observations, it is not straightforward to apply the usual BPTT algorithm without making variational approximations.

In BID35 , which is an instance of SSL, an EM-type approach was employed: the algorithm alternates between imputing the latent states and optimizing the LSTM over the imputed sequences.

However, as we show below, the inference implicitly assumes the posterior is factorizable through time.

This is a restrictive assumption since the benefit of rich state transition brought by the LSTM may be neutralized by breaking down the posterior over time.

We present a general parameter estimation scheme for the proposed class of models based on sequential Monte Carlo (SMC) BID8 , in particular the Particle Gibbs BID1 .

Instead of sampling each time point individually, we directly sample from the joint posterior without making limiting factorization assumptions.

Through extensive experiments we verify that sampling from the full posterior leads to significant improvement in the performance.

Related works Enhancing state space models using neural networks is not a new idea.

Traditional approaches can be traced back to nonlinear extensions of linear dynamical systems, such as extended or unscented Kalman filters (Julier & Uhlmann, 1997), where both state transition and emission are generalized to nonlinear functions.

The idea of parameterizing them with neural networks can be found in BID12 , as well as many recent works BID22 BID2 BID15 BID23 BID18 thanks to the development of recognition networks BID20 BID32 .

Enriching the output distribution of RNN has also regain popularity recently.

Unlike conventionally used multinomial output or mixture density networks BID4 , recent approaches seek for more flexible family of distributions such as restricted Boltzmann machines (RBM) BID6 or variational auto-encoders (VAE) BID13 BID7 .On the flip side, there have been studies in introducing stochasticity to recurrent neural networks.

For instance, BID30 and BID3 incorporated independent latent variables at each time step; while in BID9 the RNN is attached to both latent states and observations.

We note that in our approach the transition and emission are decoupled, not only for interpretability but also for efficient inference without variational assumptions.

On a related note, sequential Monte Carlo methods have recently received attention in approximating the variational objective BID27 BID24 BID29 .

Despite the similarity, we emphasize that the context is different: we take a stochastic EM approach, where the full expectation in E-step is replaced by the samples from SMC.

In contrast, SMC in above works is aimed at providing a tighter lower bound for the variational objective.

In this section, we provide a brief review of some key ingredients of this paper.

We first describe the SSMs and the RNNs for sequence modeling, and then outline the SMC methods for sampling from a series of distributions.

Consider a sequence of observations x 1:T = (x 1 , . . .

, x T ) and a corresponding sequence of latent states z 1:T = (z 1 , . . .

, z T ).

The SSMs are a class of graphical models that defines probabilistic dependencies between latent states and the observations.

A classical example of SSM is the (Gaussian) LDS, where real-valued states evolve linearly over time under the first-order Markov assumption.

Let x t ∈ R d and z t ∈ R k , the LDS can be expressed by two equations: DISPLAYFORM0 where A ∈ R k×k , C ∈ R d×k , and Q and R are covariance matrices of corresponding sizes.

They are widely applied in modeling the dynamics of moving objects, with z t representing the true state of the system, such as location and velocity of the object, and x t being the noisy observation under zero-mean Gaussian noise.

We mention two important inference tasks BID21 ) associated with SSMs.

The first tasks is filtering: at any time t, compute p(z t |x 1:t ), i.e. the most up-to-date belief of the state z t conditioned on all past and current observations x 1:t .

The other task is smoothing, which computes p(z t |x 1:T ), i.e. the update to the belief of a latent state by incorporating future observations.

One of the beauties of SSMs is that these inference tasks are available in closed form, thanks for p = 1, . . .

, P .to the simple Markovian dynamics of the latent states.

For instance, the forward-backward algorithm BID34 , the Kalman filter BID17 , and RTS smoother BID31 are widely appreciated in the literature of HMM and LDS.Having obtained the closed form filtering and smoothing equations, one can make use of the EM algorithm to find the maximum likelihood estimate (MLE) of the parameters given observations.

In the case of LDS, the E-step can be computed by RTS smoother and the M-step is simple subproblems such as least-squares regression.

We refer to BID11 for a full exposition on learning the parameters of LDS using EM iterations.

RNNs have received remarkable attention in recent years due to their strong benchmark performance as well as successful applications in real-world problems.

Unlike SSMs, RNNs aim to directly learn the complex generative distribution of p(x t |x 1:t−1 ) using a neural network, with the help of a deterministic internal state s t : DISPLAYFORM0 where RNN(·, ·) is the transition function defined by a neural network, and g(·) is an arbitrary differentiable function that maps the RNN state s t to the parameter of the distribution of x t .

The flexibility of the transformation function allows the RNN to learn from complex nonlinear nonGaussian sequences.

Moreover, since the state s t is a deterministic function of the past observations x 1:t−1 , RNNs can capture long-range dependencies, for instance matching brackets in programming languages BID19 .The BPTT algorithm can be used to find the MLE of the parameters of RNN(·, ·) and g(·).

However, although RNNs can, in principle, model long-range dependencies, directly applying BPTT can be difficult in practice since the repeated application of a squashing nonlinear activation function, such as tanh or logistic sigmoid, results in an exponential decay in the error signal through time.

LSTMs BID14 are designed to cope with the such vanishing gradient problems, by introducing an extra memory cell that is constructed as a linear combination of the previous state and signal from the input.

In this work, we also use LSTMs as building blocks, as in BID35 .

Sequential Monte Carlo (SMC) BID8 ) is an algorithm that samples from a series of potentially unnormalized densities π 1 (z 1 ), . . .

, π T (z 1:T ).

At each step t, SMC approximates the target density π t with P weighted particles using importance distribution f (z t |z 1:t−1 ): DISPLAYFORM0 where α p t is the importance weight of the p-th particle and δ x is the Dirac point mass at x. Repeating this approximation for every t leads to the SMC method, outlined in Algorithm 1.The key to this method lies in the resampling, which is implemented by repeatedly drawing the ancestors of particles at each step.

Intuitively, it encourages the particles with a higher likelihood to survive longer, since the weight reflects the likelihood of the particle path.

The final Monte Carlo estimate (4) consists of only survived particle paths, and sampling from this point masses is equivalent to choosing a particle path according to the last weights α T .

We refer to BID8 ; BID1 for detailed proof of the method.

In this section, we present the class of State Space LSTM (SSL) models that combines interpretability of SSMs and flexibility of LSTMs.

The key intuition, motivated by SSMs, is to learn dynamics in the state space, rather than in the sample space.

However, we do not assume transition in the state space is linear, Gaussian, or Markovian.

Existing approaches such as the extended Kalman filter (EKF) attempted to work with a general nonlinear transition function.

Unfortunately, additional flexibility also introduced extra difficulty in the parameter estimation: EKF relies heavily on linearizing the nonlinear functions.

We propose to use LSTM to model the dynamics in the latent state space, as they can learn from complex sequences without making limiting assumptions.

The BPTT algorithm is also well established so that no additional approximation is needed in training the latent dynamics.

Generative process Let h(·) be the emission function that maps a latent state to a parameter of the sample distribution.

As illustrated in FIG2 (a), the generative process of SSL for a single sequence is: DISPLAYFORM0 The generative process specifies the following joint likelihood, with a similar factorization as SSMs except for the Markov transition: DISPLAYFORM1 where p ω (z t |z 1:t−1 ) = p(z t ; g(s t )), ω is the set of parameters of LSTM(·, ·) and g(·), and φ is the parameters of h(·).

The structure of the likelihood function is better illustrated in FIG2 , where each latent state z t is dependent to all previous states z 1:t−1 after substituting s t recursively.

This allows the SSL to have non-Markovian state transition, with parsimonious parameterization thanks to the recurrent structure of LSTMs.

Parameter estimation We continue with a single sequence for the ease of notation.

A variational lower bound to the marginal data likelihood is given by DISPLAYFORM2 where q(z 1:T ) is the variational distribution.

Following the (stochastic) EM approach, iteratively maximizing the lower bound w.r.t.

q and the model parameters (ω, φ) leads to the following updates:• E-step: The optimal variational distribution is given by the posterior: DISPLAYFORM3 In the case of LDS or HMM, efficient smoothing algorithms such as the RTS smoother or the forward-backward algorithm are available for computing the posterior expectations of sufficient statistics.

However, without Markovian state transition, although the forward messages can still be computed, the backward recursion can no longer evaluated or efficiently approximated.• S-step: Due to the difficulties in taking expectations, we take an alternative approach to collect posterior samples instead: DISPLAYFORM4 given only the filtering equations.

We discuss the posterior sampling algorithm in detail in the next section.• M-step: Given the posterior samples z 1:T , which can be seen as Monte Carlo estimate of the expectations, the subproblem for ω and φ are DISPLAYFORM5 which is exactly the MLE of an LSTM, with z 1:T serving as the input sequence, and the MLE of the given emission model.

Having seen the generative model and the estimation algorithm, we can now discuss some instances of the proposed class of models.

In particular, we provide two examples of SSL, for continuous and discrete latent states respectively.

Example 1 (Gaussian SSL) Suppose z t and x t are real-valued vectors.

A typical choice of the transition and emission is the Gaussian distribution: DISPLAYFORM6 where g µ (·) and g σ (·) map to the mean and the covariance of the Gaussian respectively, and similarly h µ (·) and h σ (·).

For closed form estimates for the emission parameters, one can further assume DISPLAYFORM7 where C is a matrix that maps from state space to sample space, and R is the covariance matrix with appropriate size.

The MLE of φ = (C, b, R) is then given by the least squares fit.

Example 2 (Topical SSL, BID35 ) Consider x 1:T as the sequence of websites a user has visited.

One might be tempted to model the user behavior using an LSTM, however due to the enormous size of the Internet, it is almost impossible to even compute a softmax output to get a discrete distribution over the websites.

There are approximation methods for large vocabulary problems in RNN, such as the hierarchical softmax BID28 .

However, another interesting approach is to operate on a sequence with a "compressed" vocabulary, while learning how to perform such compression at the same time.

Let z t be the indicator of a "topic", which is a distribution over the vocabulary as in BID5 .

Accordingly, define DISPLAYFORM8 where W is a matrix that maps LSTM states to latent states, b is a bias term, and φ zt is a point in the probability simplex.

If z t lies in a lower dimension than x t , the LSTM is effectively trained over a sequence z 1:T with a reduced vocabulary.

On the other hand, the probabilistic mapping between z t and x t is interpretable, as it learns to group similar x t 's together.

The estimation of φ is typically performed under a Dirichlet prior, which then corresponds to the MAP estimate of the Dirichlet distribution BID35 .

In this section, we discuss how to draw samples from the posterior (7), corresponding to the S-step of the stochastic EM algorithm: DISPLAYFORM0 Assuming the integration and normalization can be performed efficiently, the following quantities can be evaluated in the forward pass without Markov state transition: DISPLAYFORM1 The task is to draw from the joint posterior of z 1:T only given access to these forward messages.

One way to circumvent the tight dependencies in z 1:T is to make a factorization assumption, as in BID35 .

More concretely, the joint distribution is decomposed as DISPLAYFORM2 where z prev 1:t−1 is the assignments from the previous inference step.

However, as we confirm in the experiments, this assumption can be restrictive since the flexibility of LSTM state transitions is offset by considering each time step independently.

In this work, we propose to use a method based on SMC, which is a principled way of sampling from a sequence of distributions.

More importantly, it does not require the model to be Markovian BID10 BID26 .

As described earlier, the idea is to approximate the posterior (15) with point masses, i.e., weighted particles.

Let f (z t |z 1:t−1 , x t ) be the importance density, and P be the number of particles.

We then can run Algorithm 1 with π t (z 1:t ) = p(x 1:t , z 1:t ) being the unnormalized target distribution at time t, where the weight becomes DISPLAYFORM3 As for the choice of the proposal distribution f (·), one could use the transition density p ω (z t |z 1:t−1 ), in which case the algorithm is also referred to as the bootstrap particle filter.

An alternative is the predictive distribution, a locally optimal proposal in terms of variance BID1 : DISPLAYFORM4 which is precisely one of the available forward messages: DISPLAYFORM5 Notice the similarity between terms in FORMULA8 and FORMULA0 .

Indeed, with the choice of predictive distribution as the proposal density, the importance weight simplifies to DISPLAYFORM6 which is not a coincidence that the name collides with the message α t .

Interestingly, this quantity no longer depends on the current particle z p t .

Instead, it marginalizes over all possible particle assignments of the current time step.

This is beneficial computationally since the intermediate terms from (20) can be reused in (22) .

Also note that the optimal proposal relies on the fact that the normalization in (20) can be performed efficiently, otherwise the bootstrap proposal should be used.

After a full pass over the sequence, the algorithm produces Monte Carlo approximation of the posterior and the marginal likelihood: The inference is completed by a final draw from the approximate posterior, DISPLAYFORM7 DISPLAYFORM8 which is essentially sampling a particle path indexed by the last particle.

Specifically, the last particle z p T is chosen according to the final weights α T , and then earlier particles can be obtained by tracing backwards to the beginning of the sequence according to the ancestry indicators a p t at each position.

Since SMC produces a Monte Carlo estimate, as the number of particles P → ∞ the approximate posterior (23) is guaranteed to converge to the true posterior for a fixed sequence.

However, as the length of the sequence T increases, the number of particles needed to provide a good approximation grows exponentially.

This is the well-known depletion problem of SMC BID1 .One elegant way to avoid simulating enormous number of particles is to marry the idea of MCMC with SMC BID1 .

The idea of such Particle MCMC (PMCMC) methods is to treat the particle estimatep(·) as a proposal, and design a Markov kernel that leaves the target distribution invariant.

Since the invariance is ensured by the MCMC, it does not demand SMC to provide an accurate approximation to the true distribution, but only to give samples that are approximately distributed according to the target.

As a result, for any fixed P > 0 the PMCMC methods ensure the target distribution is invariant.

We choose the Gibbs kernel that requires minimal modification from the basic SMC.

The resulting algorithm is Particle Gibbs (PG), which is a conditional SMC update in a sense that a reference path z ref 1:T with its ancestral lineage is fixed throughout the particle propagation of SMC.

It can be shown that this simple modification to SMC produces a transition kernel that is not only invariant, but also ergodic under mild assumptions.

In practice, we use the assignments from previous step as the reference path.

The final algorithm is summarized in Algorithm 2.

Combined with the stochastic EM outer iteration, the final algorithm is an instance of the particle SAEM BID25 BID33 , under non-Markovian state transition.

We conclude this section by deriving forward messages for the previous examples.

Example 1 (Gaussian SSL, continued) The integration and normalization preserves normality thanks to the Gaussian identify.

The messages are given by DISPLAYFORM9 where DISPLAYFORM10 Example 2 (Topical SSL, continued) Let θ t = softmax(W s t + b).

Since the distributions are discrete, we have DISPLAYFORM11 where • denotes element-wise product.

Note that the integration for α t corresponds to a summation in the state space.

It is then normalized across P particles to form a weight distribution.

For γ t the normalization is performed in the state space as well, hence the computation of the messages are manageable.

We now present empirical studies for our proposed model and inference (denoted as SMC) in order to establish that (1) SSL is flexible in capturing underlying nonlinear dynamics, (2) our inference is accurate yet easily applicable to complicated models, and (3) it opens new avenues for interpretable yet nonlinear and non-Markovian sequence models, previously unthinkable.

To illustrate these claims, we evaluate on (1) synthetic sequence tracking of varying difficulties, (2) language modeling, and (3) user modeling utilizing complicated models for capturing the intricate dynamics.

For SMC inference, we gradually increase the number of particles P from 1 to K during training.

Software & hardware All the algorithms are implemented on TensorFlow BID0 .

We run our experiments on a commodity machine with Intel R Xeon R CPU E5-2630 v4 CPU, 256GB RAM, and 4 NVidia R Titan X (Pascal) GPU.

To test the flexibility of SSL, we begin with inference using synthetic data.

We consider four different dynamics in 2D space: (i) a straight line, (ii) a sine wave, (iii) a circle, and (iv) a swiss role.

Note that we do not add additional states such as velocity, keeping the dynamics nonlinear except for the first case.

Data points are generated by adding zero mean Gaussian noise to the true underlying dynamics.

The true dynamics and the noisy observations are plotted in the top row of FIG4 .

The first 60% of the sequence is used for training and the rest is left for testing.

The middle and bottom row of FIG4 show the result of SSL and vanilla LSTM trained for same number of iterations until both are sufficiently converged.

The red points refer to the prediction of z t after observing x 1:t , and the green points are blind predictions without observing any data.

We can observe that while both methods are capturing the dynamics well in general, the predictions of LSTM tend to be more sensitive to initial predictions.

In contrast, even when the initial predictions are not incorrect, SSL can recover in the end by remaining on the latent dynamic.

For Topical SSL, we compare our SMC inference method with the factored old algorithm BID35 documents and test on the rest, using the same settings in BID35 .

FIG5 shows the test perplexity (lower is better) and number of nonzeros in the learned word topic count matrix (lower is better).

In all cases, the SMC inference method consistently outperforms the old factored method.

For comparison, we also run LSTM with the same number of parameters, which gives the lowest test perplexity of 1942.26.

However, we note that LSTM needs to perform expensive linear transformation for both embedding and softmax at every step, which depends linearly on the vocabulary size V .

In contrast, SSL only depends linearly on number of topics K V .

Ablation study We also want to explore the benefit of the newer inference as dataset size increases.

We observe that in case of natural languages which are highly structured the gap between factored approximation and accurate SMC keeps reducing as dataset size increases.

But as we will see in case of user modeling when the dataset is less structured, the factored assumption leads to poorer performance.

Also when the data size is fixed and the number of topics are varying, the SMC algorithm gives better perplexity compared to the old algorithm.

Therefore we the SMC inference is consistently better in various settings.

Visualizing particle paths In FIG7 , we show the particle paths on a snippet of an article about a music album 1 .

As we can see from the top row, which plots the particle paths at the initial iteration, the model proposed a number of candidate topic sequences since it is uncertain about the latent semantics yet.

However, after 100 epochs, as we can see from the bottom row, the model is much more confident about the underlying topical transition.

Moreover, by inspecting the learned parameters φ of the probabilistic emission, we can see that the topics are highly concentrated on topics related to music and time.

This confirms our claim about flexible sequence modeling while retaining interpretability.

We use an anonymized sample of user search click history to measure the accuracy of different models on predicting users future clicks.

An accurate model would enable better user experience by presenting the user with relevant content.

The dataset is anonymized by removing all items appearing less than a given threshold, this results in a dataset with 100K vocabulary and we vary the number of users from 500K to 1M.

We fix the number of topics at 500 for all user experiments.

We used the same setup to the one used in the experiments over the Wikipedia dataset for parameters.

The dataset is less structured than the language modeling task since users click patterns are less predictable than the sequence of words which follow definite syntactic rules.

As shown in table 1, the benefit of new inference method is highlighted as it yields much lower perplexity than the factored model.

In this paper we revisited the problem of posterior inference in Latent LSTM models as introduced in BID35 .

We generalized their model to accommodate a wide variety of state space models and most importantly we provided a more principled Sequential Monte-Carlo (SMC) algorithm for posterior inference.

Although the newly proposed inference method can be slower, we showed over a variety of dataset that the new SMC based algorithm is far superior and more stable.

While computation of the new SMC algorithm scales linearly with the number of particles, this can be naively parallelized.

In the future we plan to extend our work to incorporate a wider class of dynamically changing structured objects such as time-evolving graphs.

@highlight

We present State Space LSTM models, a combination of state space models and LSTMs, and propose an inference algorithm based on sequential Monte Carlo. 