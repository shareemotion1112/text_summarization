Gradient-based optimization is the foundation of deep learning and reinforcement learning.

Even when the mechanism being optimized is unknown or not differentiable, optimization using high-variance or biased gradient estimates is still often the best strategy.

We introduce a general framework for learning low-variance, unbiased gradient estimators for black-box functions of random variables, based on gradients of a learned function.

These estimators can be jointly trained with model parameters or policies, and are applicable in both discrete and continuous settings.

We give unbiased, adaptive analogs of state-of-the-art reinforcement learning methods such as advantage actor-critic.

We also demonstrate this framework for training discrete latent-variable models.

Gradient-based optimization has been key to most recent advances in machine learning and reinforcement learning.

The back-propagation algorithm BID21 , also known as reverse-mode automatic differentiation BID25 BID16 computes exact gradients of deterministic, differentiable objective functions.

The reparameterization trick BID33 BID7 BID17 allows backpropagation to give unbiased, lowvariance estimates of gradients of expectations of continuous random variables.

This has allowed effective stochastic optimization of large probabilistic latent-variable models.

Unfortunately, there are many objective functions relevant to the machine learning community for which backpropagation cannot be applied.

In reinforcement learning, for example, the function being optimized is unknown to the agent and is treated as a black box BID23 .

Similarly, when fitting probabilistic models with discrete latent variables, discrete sampling operations create discontinuities giving the objective function zero gradient with respect to its parameters.

Much recent work has been devoted to constructing gradient estimators for these situations.

In reinforcement learning, advantage actor-critic methods BID27 give unbiased gradient estimates with reduced variance obtained by jointly optimizing the policy parameters with an estimate of the value function.

In discrete latent-variable models, low-variance but biased gradient estimates can be given by continuous relaxations of discrete variables BID10 BID4 .A recent advance by BID30 used a continuous relaxation of discrete random variables to build an unbiased and lower-variance gradient estimator, and showed how to tune the free parameters of these relaxations to minimize the estimator's variance during training.

We generalize the method of BID30 to learn a free-form control variate parameterized by a neural network.

This gives a lower-variance, unbiased gradient estimator which can be applied to a wider variety of problems.

Most notably, our method is applicable even when no continuous relaxation is available, as in reinforcement learning or black-box function optimization.

How can we choose the parameters of a distribution to maximize an expectation?

This problem comes up in reinforcement learning, where we must choose the parameters θ of a policy distribution π(a|s, θ) to maximize the expected reward E τ ∼π [R] over state-action trajectories τ .

It also comes up in fitting latent-variable models, when we wish to maximize the marginal probability p(x|θ) = z p(x|z)p(z|θ) = E p(z|θ) [p(x|z)] .

In this paper, we'll consider the general problem of optimizing DISPLAYFORM0 When the parameters θ are high-dimensional, gradient-based optimization is appealing because it provides information about how to adjust each parameter individually.

Stochastic optimization is essential for scalablility, but is only guaranteed to converge to a fixed point of the objective when the stochastic gradientsĝ are unbiased, i.e. BID18 .

How can we build unbiased, stochastic gradient estimators?

There are several standard methods: DISPLAYFORM1 The score-function gradient estimator One of the most generally-applicable gradient estimators is known as the score-function estimator, or REINFORCE (Williams, 1992): DISPLAYFORM2 This estimator is unbiased, but in general has high variance.

Intuitively, this estimator is limited by the fact that it doesn't use any information about how f depends on b, only on the final outcome f (b).The reparameterization trick When f is continuous and differentiable, and the latent variables b can be written as a deterministic, differentiable function of a random draw from a fixed distribution, the reparameterization trick BID33 BID7 BID17 creates a low-variance, unbiased gradient estimator by making the dependence of b on θ explicit through a reparameterization function b = T (θ, ): DISPLAYFORM3 This gradient estimator is often used when training high-dimensional, continuous latent-variable models, such as variational autoencoders.

One intuition for why this gradient estimator is preferable to REINFORCE is that it depends on ∂f /∂b, which exposes the dependence of f on b.

Control variates Control variates are a general method for reducing the variance of a stochastic estimator.

A control variate is a function c(b) with a known mean DISPLAYFORM4 Given an estimator g(b), subtracting the control variate from this estimator and adding its mean gives us a new estimator: DISPLAYFORM5 This new estimator has the same expectation as the old one, but has lower variance if c(b) is positively correlated withĝ(b).

In this section, we introduce a gradient estimator for the expectation of a function DISPLAYFORM0 ] that can be applied even when f is unknown, or not differentiable, or when b is discrete.

Our estimator combines the score function estimator, the reparameterization trick, and control variates.

First, we consider the case where b is continuous, but that f cannot be differentiated.

Instead of differentiating through f , we build a surrogate of f using a neural network c φ , and differentiate c φ instead.

Since the score-function estimator and reparameterization estimator have the same expectation, we can simply subtract the score-function estimator for c φ and add back its reparameterization estimator.

This gives a gradient estimator which we call LAX: DISPLAYFORM1 This estimator is unbiased for any choice of c φ .

When c φ = f , then LAX becomes the reparameterization estimator for f .

Thus LAX can have variance at least as low as the reparameterization estimator.

An example of the relative bias and variance of each term in this estimator can be seen below.

Figure 2 : Histograms of samples from the gradient estimators that create LAX.

Samples generated from our one-layer VAE experiments (Section 6.2).

Sinceĝ LAX is unbiased for any choice of the surrogate c φ , the only remaining problem is to choose a c φ that gives low variance toĝ LAX .

How can we find a φ which gives our estimator low variance?

We simply optimize c φ using stochastic gradient descent, at the same time as we optimize the parameters θ of our model or policy.

To optimize c φ , we require the gradient of the variance of our estimator.

To estimate these gradients, we could simply differentiate through the empirical variance over each mini-batch.

Or, following BID19 and BID30 , we can construct an unbiased, single-sample estimator using the fact that our gradient estimator is unbiased.

For any unbiased gradient estimatorĝ with parameters φ: DISPLAYFORM0 Thus, an unbiased single-sample estimate of the gradient of the variance ofĝ is given by ∂ĝ 2 /∂φ.

This method of directly minimizing the variance of the gradient estimator stands in contrast to other methods such as Q-Prop and advantage actor-critic BID27 , which train the control variate to minimize the squared error (f (b) − c φ (b)) 2 .

Our algorithm, which jointly optimizes the parameters θ and the surrogate c φ is given in Algorithm 1.

What is the form of the variance-minimizing c φ ?

Inspecting the square of (5), we can see that this loss encourages c φ (b) to approximate f (b), but with a weighting based on ∂ ∂θ log p(b|θ).

Moreover, as c φ → f thenĝ LAX → ∂ ∂θ c φ .

Thus, this objective encourages a balance between the variance of the reparameterization estimator and the variance of the REINFORCE estimator.

FIG2 shows the learned surrogate on a toy problem.

Algorithm 1 LAX: Optimizing parameters and a gradient control variate simultaneously.

DISPLAYFORM0 Estimate gradient of variance of gradient DISPLAYFORM1 Update parameters DISPLAYFORM2 Update control variate end while return θ

We can adapt the LAX estimator to the case where b is a discrete random variable by introducing a "relaxed" continuous variable z. We require a continuous, reparameterizable distribution p(z|θ) and a deterministic mapping H(z) such that H(z) = b ∼ p(b|θ) when z ∼ p(z|θ).

In our implementation, we use the Gumbel-softmax trick, the details of which can be found in appendix B.The discrete version of the LAX estimator is given by: DISPLAYFORM0 This estimator is simple to implement and general.

However, if we were able to replace the ∂ ∂θ log p(z|θ) in the control variate with ∂ ∂θ log p(b|θ) we should be able to achieve a more correlated control variate, and therefore a lower variance estimator.

This is the motivation behind our next estimator, which we call RELAX.To construct a more powerful gradient estimator, we incorporate a further refinement due to BID30 .

Specifically, we evaluate our control variate both at a relaxed input z ∼ p(z|θ), and also at a relaxed input conditioned on the discrete variable b, denotedz ∼ p(z|b, θ).

Doing so gives us: DISPLAYFORM1 This estimator is unbiased for any c φ .

A proof and a detailed algorithm can be found in appendix A. We note that the distribution p(z|b, θ) must also be reparameterizable.

We demonstrate how to perform this conditional reparameterization for Bernoulli and categorical random variables in appendix B.

The variance-reduction objective introduced above allows us to use any differentiable, parametric function as our control variate c φ .

How should we choose the architecture of c φ ?

Ideally, we will take advantage of any known structure in f .In the discrete setting, if f is known and happens to be differentiable, we can use the concrete relaxation BID4 BID10 and let c φ (z) = f (σ λ (z)).

In this special case, our estimator is exactly the REBAR estimator.

We are also free to add a learned component to the concrete relaxation and let c φ (z) = f (σ λ (z)) + r ρ (z) where r ρ is a neural network with parameters ρ making φ = {ρ, λ}. We took this approach in our experiments training discrete variational autoencoders.

If f is unknown, we can simply let c φ be a generic function approximator such as a neural network.

We took this simpler approach in our reinforcement learning experiments.

We now describe how we apply the LAX estimator in the reinforcement learning (RL) setting.

By reinforcement learning, we refer to the problem of optimizing the parameters θ of a policy distribution π(a|s, θ) to maximize the sum of rewards.

In this setting, the random variable being integrated over is τ , which denotes a series of T actions and states DISPLAYFORM0 The function whose expectation is being optimized, R, maps τ to the sum of rewards R(τ ) = T t=1 r t (s t , a t ).

Again, we want to estimate the gradient of an expectation of a black-box function: DISPLAYFORM1 The de facto standard approach is the advantage actor-critic estimator (A2C) BID27 : DISPLAYFORM2 Where c φ (s t ) is an estimate of the state-value function, DISPLAYFORM3 This estimator is unbiased when c φ does not depend on a t .

The main limitations of A2C are that c φ does not depend on a t , and that it's not obvious how to optimize c φ .

Using the LAX estimator addresses both of these problems.

First, we assume π(a t |s t , θ) is reparameterizable, meaning that we can write a t = a( t , s t , θ), where t does not depend on θ.

We again introduce a differentiable surrogate c φ (a, s).

Crucially, this surrogate is a function of the action as well as the state.

The extension of LAX to Markov decision processes is: DISPLAYFORM4 ∂ log π(a t |s t , θ) ∂θ DISPLAYFORM5 DISPLAYFORM6 This estimator is unbiased if the true dynamics of the system are Markovian w.r.t.

the state s t .

When T = 1, we recover the special caseĝ RL LAX =ĝ LAX .

Comparingĝ RL LAX to the standard advantage actor-critic estimator in (9), the main difference is that our baseline c φ (a t , s t ) is action-dependent while still remaining unbiased.

To optimize the parameters φ of our control variate c φ (a t , s t ), we can again use the single-sample estimator of the gradient of our estimator's variance given in (6).

This approach avoids unstable training dynamics, and doesn't require storage and replay of previous rollouts.

Details of this derivation, as well as the discrete and conditionally reparameterized version of this estimator can be found in appendix C.

The work most related to ours is the recently-developed REBAR method BID30 , which greatly inspired our work.

The REBAR estimator is a special case of the RELAX estimator, when the surrogate is set to c φ (z) = η · f (softmax λ (z)).

The only free parameters of the REBAR estimator are the scaling factor η, and the temperature λ, which gives limited scope to optimize the surrogate.

REBAR can only be applied when f is known and differentiable.

Furthermore, it depends on essentially undefined behavior of the function being optimized, since it evaluates the discrete loss function at continuous inputs.

Because LAX and RELAX can construct a surrogate from scratch, they can be used for optimizing black-box functions, as in reinforcement learning settings where the reward is an unknown function of the environment.

LAX and RELAX only require that we can query the function being optimized, and can sample from and differentiate p(b|θ).Direct dependence on parameters Above, we assumed that the function f being optimized does not depend directly on θ, which is usually the case in black-box optimization settings.

However, a dependence on θ can occur when training probabilistic models, or when we add a regularizer.

In both these settings, if the dependence on θ is known and differentiable, we can use the fact that DISPLAYFORM0 and simply add ∂ ∂θ f (b, θ) to any of the gradient estimators above to recover an unbiased estimator.

BID11 reduce the variance of reparameterization gradients in an orthogonal way to ours by approximating the gradient-generating procedure with a simple model and using that model as a control variate.

NVIL BID12 and VIMCO BID13 provide reduced variance gradient estimation in the special case of discrete latent variable models and discrete latent variable models with Monte Carlo objectives.

BID22 estimate gradients using a form of finite differences, evaluating hundreds of different parameter values in parallel to construct a gradient estimate.

In contrast, our method is a single-sample estimator.

Staines & Barber (2012) address the general problem of developing gradient estimators for deterministic black-box functions or discrete optimization.

They introduce a sampling distribution, and optimize an objective similar to ours.

also introduce a sampling distribution to build a gradient estimator, and consider optimizing the sampling distribution.

In the context of general Monte Carlo integration, Oates et al. FORMULA0 introduce a non-parametric control variate that also leverages gradient information to reduce the variance of an estimator.

In parallel to our work, there has been a string of recent developments on action-dependent baselines for policy-gradient methods in reinforcement learning.

Such works include and BID3 which train an action-dependent baseline which incorporates off-policy data.

The optimal relaxation for a toy loss function, using different gradient estimators.

Because REBAR uses the concrete relaxation of f , which happens to be implemented as a quadratic function, the optimal relaxation is constrained to be a warped quadratic.

In contrast, RELAX can choose a free-form relaxation.

We demonstrate the effectiveness of our estimator on a number of challenging optimization problems.

Following BID30 we begin with a simple toy example to illuminate the potential of our method and then continue to the more relevant problems of optimizing binary VAE's and reinforcement learning.

As a simple example, we follow BID30 in minimizing DISPLAYFORM0 as a function of the parameter θ where p(b|θ) = Bernoulli(b|θ).

BID30 set the target t = .45.

We focus on the more challenging case where t = .499.

FIG0 show the relative performance and gradient log-variance of REINFORCE, REBAR, and RE-LAX.

FIG2 plots the learned surrogate c φ for a fixed value of θ.

We can see that c φ is near f for all z, keeping the variance of the REINFORCE part of the estimator small.

Moreover the derivative of c φ is positive for all z meaning that the reparameterization part of the estimator will produce gradients pointing in the correct direction to optimize the expectation.

Conversely, the concrete relaxation of REBAR is close to f only near 0 and 1 and its gradient points in the correct direction only for values of z > log( DISPLAYFORM1 .

These factors together result in the RELAX estimator achieving the best performance.

Next, we evaluate the RELAX estimator on the task of training a variational autoencoder BID7 BID17 with Bernoulli latent variables.

We reproduced the variational autoencoder experiments from BID30 , training models with one or two layers of 200 Bernoulli random variables with linear or nonlinear mappings between them, on both the MNIST and Omniglot BID8 datasets.

Details of these models and our experimental procedure can be found in Appendix E.1.To take advantage of the available structure in the loss function, we choose the form of our control variate to be c φ (z) = f (σ λ (z)) +r ρ (z) wherer ρ is a neural network with parameters ρ and f (σ λ (z)) is the discrete loss function, the evidence lower-bound (ELBO), evaluated at continuously relaxed inputs as in REBAR.

In all experiments, the learned control variate improved the training performance, over the state-of-the-art baseline of REBAR.

In both linear models, we achieved improved validation performance as well increased convergence speed.

We believe the decrease in validation performance for the nonlinear models was due to overfitting caused by improved optimization of an under-regularized model.

We leave exploring this phenomenon to further work.

To obtain training curves we created our own implementation of REBAR, which gave identical or slightly improved performance compared to the implementation of BID30 .While we obtained a modest improvement in training and validation scores (tables 1 and 3), the most notable improvement provided by RELAX is in its rate of convergence.

Training curves for all models can be seen in FIG3 and in Appendix D. In Table 4 we compare the number of training epochs that are required to match the best validation score of REBAR.

In both linear models, RELAX provides an increase in rate of convergence.

Since the gradient estimator is defined at the end of each episode, we display log-variance per episode.

After every 10th training episode 100 episodes were run and the sample log-variance is reported averaged over all policy parameters.

Cart-pole Lunar lander Inverted pendulum A2C 1152 ± 90 162374 ± 17241 6243 ± 164 LAX/RELAX 472 ± 114 68712 ± 20668 2067 ± 412 Table 2 : Mean episodes to solve tasks.

Definitions of solving each task can be found in Appendix E.

We apply our gradient estimator to a few simple reinforcement learning environments with discrete and continuous actions.

We use the RELAX and LAX estimators for discrete and continuous actions, respectively.

We compare with the advantage actor-critic algorithm (A2C) (Sutton et al., 2000) as a baseline.

As our control variate does not have the same interpretation as the value function of A2C, it was not directly clear how to add reward bootstrapping and other variance reduction techniques common in RL into our model.

For instance, to do reward bootstrapping, we would need to use the statevalue function.

In the discrete experiments, due to the simplicity of the tasks, we chose not to use reward bootstrapping, and therefore omitted the use of state-value function.

However, with the more complicated continuous tasks, we chose to use the value function to enable bootstrapping.

In this case, the control variate takes the form: c φ (a, s) = V (s) +ĉ(a, s), where V (s) is trained as it would be in A2C.

Full details of our experiments can be found in Appendix E.In the discrete action setting, we test our approach on the Cart Pole and Lunar Lander environments as provided by the OpenAI gym BID0 .

In the continuous action setting, we test on the MuJoCo-simulated BID29 environment Inverted Pendulum also found in the OpenAI gym.

In all tested environments we observe improved performance and sample efficiency using our method.

The results of our experiments can be seen in FIG4 , and Table 2 .We found that our estimator produced policy gradients with drastically reduced variance (see FIG4 ) allowing for larger learning rates to be used while maintaining stable training.

In both discrete environments our estimator achieved greater than a 2-times speedup in convergence over the baseline.

In this work we synthesized and generalized several standard approaches for constructing gradient estimators.

We proposed a generic gradient estimator that can be applied to expectations of known or black-box functions of discrete or continuous random variables, and adds little computational overhead.

We also derived a simple extension to reinforcement learning in both discrete and continuous-action domains.

Future applications of this method could include training models with hard attention or memory indexing BID36 .

One could also apply our estimators to continuous latentvariable models whose likelihood is non-differentiable, such as a 3D rendering engine.

Extensions to the reparameterization gradient estimator BID20 BID14 could also be applied to increase the scope of distributions that can be modeled.

In the reinforcement learning setting, our method could be combined with other variance-reduction techniques such as generalized advantage estimation BID5 BID24 , or other optimization methods, such as KFAC BID35 .

One could also train our control variate off-policy, as in Q-prop .

Proof.

We show thatĝ RELAX is an unbiased estimator of DISPLAYFORM0 Expanding the expectation for clarity of exposition, we account for each term in the estimator separately: DISPLAYFORM1 DISPLAYFORM2 Term FORMULA0 is an unbiased score-function estimator of DISPLAYFORM3 .

It remains to show that the other three terms are zero in expectation.

Following BID30 (see the appendices of that paper for a derivation), we rewrite term (14) as follows: DISPLAYFORM4 Note that the first term on the right-hand side of equation FORMULA0 is equal to term (13) with opposite sign.

The second term on the right-hand side of equation FORMULA0 is the score-function estimator of term (15), opposite in sign.

The sum of these terms is zero in expectation.

Algorithm 2 RELAX: Low-variance control variate optimization for black-box gradient estimation.

DISPLAYFORM5 Estimate gradient of variance of gradient DISPLAYFORM6 Update parameters DISPLAYFORM7 Update control variate end while return θ

When applying the RELAX estimator to a function of discrete random variables b ∼ p(b|θ), we require that there exists a distribution p(z|θ) and a deterministic mapping H(z) such that if z ∼ p(z|θ) then H(z) = b ∼ p(b|θ).

Treating both b and z as random, this procedure defines a probabilistic model p(b, z|θ) = p(b|z)p(z|θ).

The RELAX estimator requires reparameterized samples from p(z|θ) and p(z|b, θ).

We describe how to sample from these distributions in the common cases of p(b|θ) = Bernoulli(θ) and p(b|θ) = Categorical(θ).Bernoulli When p(b|θ) is Bernoulli distribution we let H(z) = I(z > 0) and we sample from p(z|θ) with DISPLAYFORM0 We can sample from p(z|b, θ) with DISPLAYFORM1 Categorical When p(b|θ) is a Categorical distribution where θ i = p(b = i|θ), we let H(z) = argmax(z) and we sample from p(z|θ) with DISPLAYFORM2 where k is the number of possible outcomes.

To sample from p(z|b, θ), we note that the distribution of the largestẑ b is independent of θ, and can be sampled asẑ b = − log(− log v b ) where v b ∼ uniform[0, 1].

Then, the remaining v i =b can be sampled as before but with their underlying noise truncated soẑ i =b <ẑ b .

As shown in the appendix of BID30 , we can then sample from p(z|b, θ) with: DISPLAYFORM3 where DISPLAYFORM4

We give the derivation of the LAX estimator used for continuous RL tasks.

Theorem C.1.

The LAX estimator, DISPLAYFORM0 ∂ log π(a t |s t , θ) ∂θ DISPLAYFORM1 DISPLAYFORM2 is unbiased.

Proof.

Note that by using the score-function estimator, for all t, we have DISPLAYFORM3 Then, by adding and subtracting the same term, we have DISPLAYFORM4 ∂ log π(a t |s t , θ) ∂θ DISPLAYFORM5 ∂ log π(a t |s t , θ) ∂θ DISPLAYFORM6 In the discrete control setting, our policy parameterizes a soft-max distribution which we use to sample actions.

We define z t ∼ p(z t |s t ), which is equal to σ(log π − log(− log(u))) where u ∼ uniform[0, 1], a t = argmax(z t ), σ is the soft-max function.

We also definez t ∼ p(z t |a t , s t ) and uses the same reparametrization trick for samplingz t as explicated in Appendix B. Theorem C.2.

The RELAX estimator, DISPLAYFORM7 ∂ log π(a t |s t , θ) ∂θ DISPLAYFORM8 DISPLAYFORM9 Proof.

Note that by using the score-function estimator, for all t, we have E p(a1:t,s1:t)∂ log π(a t |s t , θ) ∂θ DISPLAYFORM10 Then, by adding and subtracting the same term, we have DISPLAYFORM11 ∂ log π(a t |s t , θ) ∂θ DISPLAYFORM12 ∂ log π(a t |s t , θ) ∂θ DISPLAYFORM13 Since p(z t |s t ) is reparametrizable, we obtain the estimator in Eq.(19).

Table 4 : Epochs needed to achieve REBAR's best validation score.

"-" indicates that the nonlinear RELAX models achieved lower validation scores than REBAR.

We run all models for 2, 000, 000 iterations with a batch size of 24.

For the REBAR models, we tested learning rates in {.005, .001, .0005, .0001, .00005}.RELAX adds more hyperparameters.

These are the depth of the neural network component of our control variate r ρ , the weight decay placed on the network, and the scaling on the learning rate for the control variate.

We tested neural network models with l layers of 200 units using the ReLU nonlinearity with l ∈ {2, 4}. We trained the control variate with weight decay in {.001, .0001}. We trained the control variate with learning rate scaling in {1, 10}.To limit the size of hyperparameter search for the RELAX models, we only test the best performing learning rate for the REBAR baseline and the next largest learning rate in our search set.

In many cases, we found that RELAX allowed our model to converge at learning rates which made the REBAR estimators diverge.

We believe further improvement could be achieved by tuning this parameter.

It should be noted that in our experiments, we found the RELAX method to be fairly insensitive to all hyperparameters other than learning rate.

In general, we found the larger (4 layer) control variate architecture with weight decay of .001 and learning rate scaling of 1 to work best, but only slightly outperformed other configurations.

All presented results are from the models which achieve the highest ELBO on the validation data.

In the one-layer linear models we optimize the evidence lower bound (ELBO): DISPLAYFORM0 where q(b 1 |x) = σ(x · W q + β q ) and p(x|b 1 ) = σ(b 1 · W p + β p ) with weight matrices W q , W p and bias vectors β q , β p .

The parameters of the prior p(b) are also learned.

In the two layer linear models we optimize the ELBO DISPLAYFORM0 , and p(b 1 |b 2 ) = σ(b 2 ·W p2 +β p2 ) with weight matrices W q1 , W q2 , W p1 , W p2 and biases β q1 , β q2 , β p1 , β p2 .

As in the one-layer model, the prior p(b 2 ) is also learned.

In the one-layer nonlinear model, the mappings between random variables consist of 2 deterministic layers with 200 units using the hyperbolic-tangent nonlinearity followed by a linear layer with 200 units.

We run an identical hyperpameter search in all models.

In both the baseline A2C and RELAX models, the policy and control variate (value function in the baseline model) were two-layer neural networks with 10 units per layer.

The ReLU non linearity was used on all layers except for the output layer which was linear.

For these tasks we estimate the policy gradient with a single Monte Carlo sample.

We run one episode of the environment to completion, compute the discounted rewards, and run one iteration of gradient descent.

We believe using larger batches will improve performance but would less clearly demonstrate the potential of our method.

Both models were trained with the RMSProp BID28 optimizer and a reward discount factor of .99 was used.

Entropy regularization with a weight of .01 was used to encourage exploration.

Both models have 2 hyperparameters to tune; the global learning rate and the scaling factor on the learning rate for the control variate (or value function).

We complete a grid search for both parameters in {0.01, 0.003, 0.001} and present the model which "solves" the task in the fewest number of episodes averaged over 5 random seeds.

"Solving" the tasks was defined by the creators of the OpenAI gym BID0 ).

The Cart Pole task is considered solved if the agent receives an average reward greater than 195 over 100 consecutive episodes.

The Lunar Lander task is considered solved if the agent receives an average reward greater than 200 over 100 consecutive episodes.

The Cart Pole experiments were run for 250,000 frames.

The Lunar Lander experiments were run for 5,000,000 frames.

The results presented for the CartPole and LunarLander environments were obtained using a slightly biased sampler for p(z|b, θ).

The three models-policy, value, and control variate, are two-layer neural networks with 64 hidden units per layer.

The value and control variate networks are identical, with the ELU (Djork-Arné Clevert & Hochreiter, 2016) nonlinearity in each hidden layer.

The policy network has tanh nonlinearity.

The policy network, which parameterizes the Gaussian policy comprises of a network (with the architecture mentioned above) that outputs the mean, and a separate, trainable log standard deviation value that is not input dependent.

All three networks have a linear output layer.

We selected the batch size to be 2500, meaning for a fixed timestep (2500) we collect multiple rollouts of a task and update the networks' parameters with the batch of episodes.

Per one policy update, we optimize both the value and control variate network multiple times.

The number of times we train the value network is fixed to 25, while for the control variate, it was chosen to be a hyperparameter.

All models were trained using ADAM BID6 , with β 1 = 0.9, β 2 = 0.999, and = 1e − 08.The baseline A2C case has 2 hyperparameters to tune: the learning rate for the optimizer for the policy and value network.

A grid search was done over the set: {0.03, 0.003, 0.0003}. RELAX has 4 hyperparameters to tune: 3 learning rates for the optimizer per network, and the number of training iterations of the control variate per policy gradient update.

Due to the large number of hyperparameters, we restricted the size of the grid search set to {0.003, 0.0003} for the learning rates, and {1, 5, 25} for the control variate training iteration number.

We chose the hyperparameter setting that yielded the shortest episode-to-completion time averaged over 5 random seeds.

As with the discrete case, we used the definition of completion provided by the OpenAI gym BID0 for each task.

The Inverted Pendulum experiments were run for 1,000,000 frames.

BID31 pointed out a bug in our initially released code for the continuous RL experiments.

This issue has been fixed in the publicly available code and the results presented in this paper were generated with the corrected code.

For continuous RL tasks, it is convention to employ a batch of a fixed number of timesteps (here, 2500) in which the number of episodes vary.

We follow this convention for the sake of providing a fair comparison to the baseline.

However, this causes a complication when calculating the variance loss for the control variate because we must compute the variance averaged over completed episodes, which is difficult to obtain when the number of episodes is not fixed.

For this reason, in our implementation we compute the gradients for the control variate outside of the Tensorflow computation graph.

However, for practical reasons we recommend using a batch of fixed number of episodes when using our method.

<|TLDR|>

@highlight

We present a general method for unbiased estimation of gradients of black-box functions of random variables. We apply this method to discrete variational inference and reinforcement learning. 

@highlight

Suggests a new approach to performing gradient descent for blackbox optimization or training discrete latent variable models.