Current model-based reinforcement learning approaches use the model simply as a learned black-box simulator to augment the data for policy optimization or value function learning.

In this paper, we show how to make more effective use of the model by exploiting its differentiability.

We construct a policy optimization algorithm that uses the pathwise derivative of the learned model and policy across future timesteps.

Instabilities of learning across many timesteps are prevented by using a terminal value function, learning the policy in an actor-critic fashion.

Furthermore, we present a derivation on the monotonic improvement of our objective in terms of the gradient error in the model and value function.

We show that our approach (i) is consistently more sample efficient than existing state-of-the-art model-based algorithms, (ii) matches the asymptotic performance of model-free algorithms, and (iii) scales to long horizons, a regime where typically past model-based approaches have struggled.

Model-based reinforcement learning (RL) offers the potential to be a general-purpose tool for learning complex policies while being sample efficient.

When learning in real-world physical systems, data collection can be an arduous process.

Contrary to model-free methods, model-based approaches are appealing due to their comparatively fast learning.

By first learning the dynamics of the system in a supervised learning way, it can exploit off-policy data.

Then, model-based methods use the model to derive controllers from it either parametric controllers (Luo et al., 2019; Buckman et al., 2018; Janner et al., 2019) or non-parametric controllers (Nagabandi et al., 2017; Chua et al., 2018) .

Current model-based methods learn with an order of magnitude less data than their model-free counterparts while achieving the same asymptotic convergence.

Tools like ensembles, probabilistic models, planning over shorter horizons, and meta-learning have been used to achieved such performance (Kurutach et al., 2018; Chua et al., 2018; .

However, the model usage in all of these methods is the same: simple data augmentation.

They use the learned model as a black-box simulator generating samples from it.

In high-dimensional environments or environments that require longer planning, substantial sampling is needed to provide meaningful signal for the policy.

Can we further exploit our learned models?

In this work, we propose to estimate the policy gradient by backpropagating its gradient through the model using the pathwise derivative estimator.

Since the learned model is differentiable, one can link together the model, reward function, and policy to obtain an analytic expression for the gradient of the returns with respect to the policy.

By computing the gradient in this manner, we obtain an expressive signal that allows rapid policy learning.

We avoid the instabilities that often result from back-propagating through long horizons by using a terminal Q-function.

This scheme fully exploits the learned model without harming the learning stability seen in previous approaches (Kurutach et al., 2018; .

The horizon at which we apply the terminal Q-function acts as a hyperparameter between model-free (when fully relying on the Q-function) and model-based (when using a longer horizon) of our algorithm.

The main contribution of this work is a model-based method that significantly reduces the sample complexity compared to state-of-the-art model-based algorithms (Janner et al., 2019; Buckman et al., 2018) .

For instance, we achieve a 10k return in the half-cheetah environment in just 50 trajectories.

We theoretically justify our optimization objective and derive the monotonic improvement of our learned policy in terms of the Q-function and the model error.

Furtermore, we experimentally analyze the theoretical derivations.

Finally, we pinpoint the importance of our objective by ablating all the components of our algorithm.

The results are reported in four model-based benchmarking environments Todorov et al., 2012) .

The low sample complexity and high performance of our method carry high promise towards learning directly on real robots.

Model-Based Reinforcement Learning.

Learned dynamics models offer the possibility to reduce sample complexity while maintaining the asymptotic performance.

For instance, the models can act as a learned simulator on which a model-free policy is trained on (Kurutach et al., 2018; Luo et al., 2019; Janner et al., 2019) .

The model can also be used to improve the target value estimates (Feinberg et al., 2018) or to provide additional context to a policy (Du & Narasimhan, 2019) .

Contrary to these methods, our approach uses the model in a different way: we exploit the fact that the learned simulator is differentiable and optimize the policy with the analytical gradient.

Long term predictions suffer from a compounding error effect in the model, resulting in unrealistic predictions.

In such cases, the policy tends to overfit to the deficiencies of the model, which translates to poor performance in the real environment; this problem is known as model-bias (Deisenroth & Rasmussen, 2011) .

The model-bias problem has motivated work that uses meta-learning , interpolation between different horizon predictions (Buckman et al., 2018; Janner et al., 2019) , and interpolating between model and real data (Kalweit & Boedecker, 2017 ).

To prevent model-bias, we exploit the model for a short horizon and use a terminal value function to model the rest of the trajectory.

Finally, since our approach returns a stochastic policy, dynamics model, and value function could use model-predictive control (MPC) for better performance at test time, similar to (Lowrey et al., 2018; Hong et al., 2019) .

MPC methods (Nagabandi et al., 2017) have shown to be very effective when the uncertainty of the dynamics is modelled (Chua et al., 2018; .

Differentable Planning.

Previous work has used backpropagate through learned models to obtain the optimal sequences of actions.

For instance, Levine & Abbeel (2014) learn linear local models and obtain the optimal sequences of actions, which is then distilled into a neural network policy.

The planning can be incorporated into the neural network architecture (Okada et al., 2017; Tamar et al., 2016; Srinivas et al., 2018; Karkus et al., 2019) or formulated as a differentiable function (Pereira et al., 2018; Amos et al., 2018) .

Planning sequences of actions, even when doing model-predictive control (MPC), does not scale well to high-dimensional, complex domains Janner et al. (2019) .

Our method, instead learns a neural network policy in an actor-critic fashion aided with a learned model.

In our study, we evaluate the benefit of carrying out MPC on top of our learned policy at test time, Section 5.4.

The results suggest that the policy captures the optimal sequence of action, and re-planning does not result in significant benefits.

Policy Gradient Estimation.

The reinforcement learning objective involves computing the gradient of an expectation (Schulman et al., 2015a) .

By using Gaussian processes (Deisenroth & Rasmussen, 2011) , it is possible to compute the expectation analytically.

However, when learning expressive parametric non-linear dynamical models and policies, such closed form solutions do not exist.

The gradient is then estimated using Monte-Carlo methods (Mohamed et al., 2019) .

In the context of model-based RL, previous approaches mostly made use of the score-function, or REINFORCE estimator (Peters & Schaal, 2006; Kurutach et al., 2018) .

However, this estimator has high variance and extensive sampling is needed, which hampers its applicability in high-dimensional environments.

In this work, we make use of the pathwise derivative estimator (Mohamed et al., 2019) .

Similar to our approach, uses this estimator in the context of model-based RL.

However, they just make use of real-world trajectories that introduces the need of a likelihood ratio term for the model predictions, which in turn increases the variance of the gradient estimate.

Instead, we entirely rely on the predictions of the model, removing the need of likelihood ratio terms.

Actor-Critic Methods.

Actor-critic methods alternate between policy evaluation, computing the value function for the policy; and policy improvement using such value function (Sutton & Barto, 1998; Barto et al., 1983) .

Actor-critic methods can be classified between on-policy and off-policy.

On-policy methods tend to be more stable, but at the cost of sample efficiency (Sutton, 1991; Mnih et al., 2016) .

On the other hand, off-policy methods offer better sample complexity .

Recent work has significantly stabilized and improved the performance of off-policy methods using maximum-entropy objectives (Haarnoja et al., 2018a) and multiple value functions (Fujimoto et al., 2018) .

Our method combines the benefit of both.

By using the learned model we can have a learning that resembles an on-policy method while still being off-policy.

In this section, we present the reinforcement learning problem, two different lines of algorithms that tackle it, and a summary on Monte-Carlo gradient estimators.

A discrete-time finite Markov decision process (MDP) M is defined by the tuple (S, A, f, r, γ, p 0 , T ).

Here, S is the set of states, A the action space, s t+1 ∼ f (s t , a t ) the transition distribution, r : S × A → R is a reward function, p 0 : S → R + represents the initial state distribution, γ the discount factor, and T is the horizon of the process.

We define the return as the sum of rewards r(s t , a t ) along a trajectory τ := (s 0 , a 0 , ..., s T −1 , a T −1 , s T ).

The goal of reinforcement learning is to find a policy π θ : S × A → R + that maximizes the expected return, i.e., max

Actor-Critic.

In actor-critic methods, we learn a functionQ (critic) that approximates the expected return conditioned on a state s and action a,

Then, the learned Q-function is used to optimize a policy π (actor).

Usually, the Q-function is learned by iteratively minimizing the Bellman residual:

The above method is referred as one-step Q-learning, and while a naive implementation often results in unstable behaviour, recent methods have succeeded in stabilizing the Q-function training (Fujimoto et al., 2018) .

The actor then can be trained to maximize the learnedQ function

The benefit of this form of actor-critic method is that it can be applied in an off-policy fashion, sampling random mini-batches of transitions from an experience replay buffer (Lin, 1992) .

Model-Based RL.

Model-based methods, contrary to model-free RL, learn the transition distribution from experience.

Typically, this is carried out by learning a parametric function approximatorf φ , known as a dynamics model.

We define the state predicted by the dynamics model asŝ t+1 , i.e.,ŝ t+1 ∼f φ (s t , a t ).

The models are trained via maximum likelihood:

In order to optimize the reinforcement learning objective, it is needed to take the gradient of an expectation.

In general, it is not possible to compute the exact expectation so Monte-Carlo gradient estimators are used instead.

These are mainly categorized into three classes: the pathwise, score function, and measure-valued gradient estimator (Mohamed et al., 2019) .

In this work, we use the pathwise gradient estimator, which is also known as the re-parameterization trick (Kingma & Welling, 2013) .

This estimator is derived from the law of the unconscious statistician (LOTUS) (Grimmett & Stirzaker, 2001 )

Here, we have stated that we can compute the expectation of a random variable x without knowing its distribution, if we know its corresponding sampling path and base distribution.

A common case, and the one used in this manuscript, θ parameterizes a Gaussian distribution:

, which is equivalent to x = µ θ + σ θ for ∼ N (0, 1).

Exploiting the full capability of learned models has the potential to enable complex and highdimensional real robotics tasks while maintaining low sample complexity.

Our approach, modelaugmented actor-critic (MAAC), exploits the learned model by computing the analytic gradient of the returns with respect to the policy.

In contrast to sample-based methods, which one can think of as providing directional derivatives in trajectory space, MAAC computes the full gradient, providing a strong learning signal for policy learning, which further decreases the sample complexity.

In the following, we present our policy optimization scheme and describe the full algorithm.

Among model-free methods, actor-critic methods have shown superior performance in terms of sample efficiency and asymptotic performance (Haarnoja et al., 2018a) .

However, their sample efficiency remains worse than modelbased approaches, and fully off-policy methods still show instabilities comparing to on-policy algorithms (Mnih et al., 2016) .

Here, we propose a modification of the Q-function parametrization by using the model predictions on the first time-steps after the action is taken.

Specifically, we do policy optimization by maximizing the following objective:

whereby, s t+1 ∼f (s t , a t ) and a t ∼ π θ (s t ).

Note that under the true dynamics and Q-function, this objective is the same as the RL objective.

Contrary to previous reinforcement learning methods, we optimize this objective by back-propagation through time.

Since the learned dynamics model and policy are parameterized as Gaussian distributions, we can make use of the pathwise derivative estimator to compute the gradient, resulting in an objective that captures uncertainty while presenting low variance.

The computational graph of the proposed objective is shown in Figure 1 .

While the proposed objective resembles n-step bootstrap (Sutton & Barto, 1998) , our model usage fundamentally differs from previous approaches.

First, we do not compromise between being offpolicy and stability.

Typically, n-step bootstrap is either on-policy, which harms the sample complexity, or its gradient estimation uses likelihood ratios, which presents large variance and results in unstable learning .

Second, we obtain a strong learning signal by backpropagating the gradient of the policy across multiple steps using the pathwise derivative estimator, instead of the REINFORCE estimator (Mohamed et al., 2019; Peters & Schaal, 2006) .

And finally, we prevent the exploding and vanishing gradients effect inherent to back-propagation through time by the means of the terminal Q-function (Kurutach et al., 2018) .

The horizon H in our proposed objective allows us to trade off between the accuracy of our learned model and the accuracy of our learned Q-function.

Hence, it controls the degree to which our algorithm is model-based or well model-free.

If we were not to trust our model at all (H = 0), we would end up with a model-free update; for H = ∞, the objective results in a shooting objective.

Note that we will perform policy optimization by taking derivatives of the objective, hence we require accuracy on the derivatives of the objective and not on its value.

The following lemma provides a bound on the gradient error in terms of the error on the derivatives of the model, the Q-function, and the horizon H. Lemma 4.1 (Gradient Error).

Letf andQ be the learned approximation of the dynamics f and Q-function Q, respectively.

Assume that Q andQ have L q /2-Lipschitz continuous gradient and f and f have L f /2-Lipschitz continuous gradient.

Let f = max t ∇f (ŝ t ,â t ) − ∇f (s t , a t ) 2 be the error on the model derivatives and Q = ∇Q(ŝ H ,â H ) − ∇Q(s H , a H ) 2 the error on the Q-function derivative.

Then the error on the gradient between the learned objective and the true objective can be bounded by:

The result in Lemma 4.1 stipulates the error of the policy gradient in terms of the maximum error in the model derivatives and the error in the Q derivatives.

The functions c 1 and c 2 are functions of the horizon and depend on the Lipschitz constants of the model and the Q-function.

Note that we are just interested in the relation between both sources of error, since the gradient magnitude will be scaled by the learning rate, or by the optimizer, when applying it to the weights.

In the previous section, we presented our objective and the error it incurs in the policy gradient with respect to approximation error in the model and the Q function.

However, the error on the gradient is not indicative of the effect of the desired metric: the average return.

Here, we quantify the effect of the modeling error on the return.

First, we will bound the KL-divergence between the policies resulting from taking the gradient with the true objective and the approximated one.

Then we will bound the performance in terms of the KL.

Lemma 4.2 (Total Variation Bound).

Under the assumptions of the Lemma 4.1, let θ = θ o + α∇ θ J π be the parameters resulting from taking a gradient step on the exact objective, andθ = θ o + α∇ θĴπ the parameters resulting from taking a gradient step on approximated objective, where α ∈ R + .

Then the following bound on the total variation distance holds

Proof.

See Appendix.

The previous lemma results in a bound on the distance between the policies originated from taking a gradient step using the true dynamics and Q-function, and using its learned counterparts.

Now, we can derive a similar result from Kakade & Langford (2002) to bound the difference in average returns.

Theorem 4.1 (Monotonic Improvement).

Under the assumptions of the Lemma 4.1, be θ andθ as defined in Lemma 4.2, and assuming that the reward is bounded by r max .

Then the average return of the πθ satisfies

Proof.

See Appendix.

Hence, we can provide explicit lower bounds of improvement in terms of model error and function error.

Theorem 4.1 extends previous work of monotonic improvement for model-free policies (Schulman et al., 2015b; Kakade & Langford, 2002) , to the model-based and actor critic set up by taking the error on the learned functions into account.

From this bound one could, in principle, derive the optimal horizon H that minimizes the gradient error.

However, in practice, approximation errors are hard to determine and we treat H as an extra hyper-parameter.

In section 5.2, we experimentally analyze the error on the gradient for different estimators and values of H.

Based on the previous sections, we develop a new algorithm that explicitly optimizes the modelaugmented actor-critic (MAAC) objective.

The overall algorithm is divided into three main steps: model learning, policy optimization, and Q-function learning.

Model learning.

In order to prevent overfitting and overcome model-bias (Deisenroth & Rasmussen, 2011) , we use a bootstrap ensemble of dynamics models {f φ1 , ...,f φ M }.

Each of the dynamics models parameterizes the mean and the covariance of a Gaussian distribution with diagonal covariance.

The bootstrap ensemble captures the epistemic uncertainty, uncertainty due to the limited capacity or data, while the probabilistic models are able to capture the aleatoric uncertainty (Chua et al., 2018) , inherent uncertainty of the environment.

We denote by f φ the transitions dynamics resulting from φ U , where U ∼ U[M ] is uniform random variable on {1, ..., M }.

The dynamics models are trained via maximum likelihood with early stopping on a validation set.

Sample trajectories from the real environment with policy π θ .

Add them to D env .

4: for i = 1 . . .

G 2 do 10:

end for 13: until the policy performs well in the real environment 14: return Optimal parameters θ * Policy Optimization.

We extend the MAAC objective with an entropy bonus (Haarnoja et al., 2018b) , and perform policy learning by maximizing

We learn the policy by using the pathwise derivative of the model through H steps and the Q-function by sampling multiple trajectories from the sameŝ 0 .

Hence, we learn a maximum entropy policy using pathwise derivative of the model through H steps and the Q-function.

We compute the expectation by sampling multiple actions and states from the policy and learned dynamics, respectively.

Q-function Learning.

In practice, we train two Q-functions (Fujimoto et al., 2018) since it has been experimentally proven to yield better results.

We train both Q functions by minimizing the Bellman error (Section 3.1):

Similar to (Janner et al., 2019) , we minimize the Bellman residual on states previously visited and imagined states obtained from unrolling the learned model.

Finally, the value targets are obtained in the same fashion the Stochastic Ensemble Value Expansion (Buckman et al., 2018) , using H as a horizon for the expansion.

In doing so, we maximally make use of the model by not only using it for the policy gradient step, but also for training the Q-function.

Our method, MAAC, iterates between collecting samples from the environment, model training, policy optimization, and Q-function learning.

A practical implementation of our method is described in Algorithm 1.

First, we obtain trajectories from the real environment using the latest policy available.

Those samples are appended to a replay buffer D env , on which the dynamics models are trained until convergence.

The third step is to collect imaginary data from the models: we collect k-step transitions by unrolling the latest policy from a randomly sampled state on the replay buffer.

The imaginary data constitutes the D model , which together with the replay buffer, is used to learn the Q-function and train the policy.

Our algorithm consolidates the insights built through the course of this paper, while at the same time making maximal use of recently developed actor-critic and model-based methods.

All in all, it consistently outperforms previous model-based and actor-critic methods.

Our experimental evaluation aims to examine the following questions: 1) How does MAAC compares against state-of-the-art model-based and model-free methods?

2) Does the gradient error correlate with the derived bound?, 3) Which are the key components of its performance?, and 4) Does it benefit from planning at test time?

In order to answer the posed questions, we evaluate our approach on model-based continuous control benchmark tasks in the MuJoCo simulator (Todorov et al., 2012; .

We compare our method on sample complexity and asymptotic performance against state-of-the-art model-free (MF) and model-based (MB) baselines.

Specifically, we compare against the model-free soft actor-critic (SAC) (Haarnoja et al., 2018a) , which is an off-policy algorithm that has been proven to be sample efficient and performant; as well as two state-of-the-art model-based baselines: modelbased policy-optimization (MBPO) (Janner et al., 2019) and stochastic ensemble value expansion (STEVE) (Buckman et al., 2018) .

The original STEVE algorithm builds on top of the model-free algorithm DDPG , however this algorithm is outperformed by SAC.

In order to remove confounding effects of the underlying model-free algorithm, we have implemented the STEVE algorithm on top of SAC.

We also add SVG(1) to comparison, which similar to our method uses the derivative of dynamic models to learn the policy.

The results, shown in Fig. 2 , highlight the strength of MAAC in terms of performance and sample complexity.

MAAC scales to higher dimensional tasks while maintaining its sample efficiency and asymptotic performance.

In all the four environments, our method learns faster than previous MB and MF methods.

We are able to learn near-optimal policies in the half-cheetah environment in just over 50 rollouts, while previous model-based methods need at least the double amount of data.

Furthermore, in complex environments, such as ant, MAAC achieves near-optimal performance within 150 rollouts while other take orders of magnitudes more data.

Here, we investigate how the bounds obtained relate to the empirical performance.

In particular, we study the effect of the horizon of the model predictions on the gradient error.

In order to do so, we construct a double integrator environment; since the transitions are linear and the cost is quadratic for a linear policy, we can obtain the analytic gradient of the expect return.

Figure 3: L1 error on the policy gradient when using the proposed objective for different values of the horizon H as well as the error obtained when using the true dynamics.

The results correlate with the assumption that the error in the learned dynamics is lower than the error in the Q-function, as well as they correlate with the derived bounds.

Figure 3 depicts the L1 error of the MAAC objective for different values of the horizon H as well as what would be the error using the true dynamics.

As expected, using the true dynamics yields to lower gradient error since the only source comes from the learned Q-function that is weighted down by γ H .

The results using learned dynamics correlate with our assumptions and the derived bounds: the error from the learned dynamics is lower than the one in the Q-funtion, but it scales poorly with the horizon.

For short horizons the error decreases as we increase the horizon.

However, large horizons is detrimental since it magnifies the error on the models.

In order to investigate the importance of each of the components of our overall algorithm, we carry out an ablation test.

Specifically, we test three different components: 1) not using the model to train the policy, i.e., set H = 0, 2) not using the STEVE targets for training the critic, and 3) using a single sample estimate of the path-wise derivative.

The ablation test is shown in Figure 4 .

The test underpins the importance of backpropagating through the model: setting H to be 0 inflicts a severe drop in the algorithm performance.

On the other hand, using the STEVE targets results in slightly more stable training, but it does not have a significant effect.

Finally, while single sample estimates can be used in simple environments, they are not accurate enough in higher dimensional environments such as ant.

Figure 4: Ablation test of our method.

We test the importance of several components of our method: not using the model to train the policy (H = 0), not using the STEVE targets for training the Q-function (-STEVE), and using a single sample estimate of the pathwise derivative.

Using the model is the component that affects the most the performance, highlighting the importance of our derived estimator.

One of the key benefits of methods that combine model-based reinforcement learning and actor-critic methods is that the optimization procedure results in a stochastic policy, a dynamics model and a Q-function.

Hence, we have all the components for, at test time, refine the action selection by the means of model predictive control (MPC).

Here, we investigate the improvement in performance of planning at test time.

Specifically, we use the cross-entropy method with our stochastic policy as our initial distributions.

The results, shown in Table 2 , show benefits in online planning in complex domains; however, its improvement gains are more timid in easier domains, showing that the learned policy has already interiorized the optimal behaviour.

HalfCheetahEnv HopperEnv Walker2dEnv

MAAC+MPC 3.97e3 ± 1.48e3 1.09e4 ± 94.5 2.8e3 ± 11 1.76e3 ± 78 MAAC 3.06e3 ± 1.45e3 1.07e4 ± 253 2.77e3 ± 3.31 1.61e3 ± 404 Table 1 :

Performance at test time with (maac+mpc) and without (maac) planning of the converged policy using the MAAC objective.

In this work, we present model-augmented actor-critic, MAAC, a reinforcement learning algorithm that makes use of a learned model by using the pathwise derivative across future timesteps.

We prevent instabilities arisen from backpropagation through time by the means of a terminal value function.

The objective is theoretically analyzed in terms of the model and value error, and we derive a policy improvement expression with respect to those terms.

Our algorithm that builds on top of MAAC is able to achieve superior performance and sample efficiency than state-of-the-art model-based and model-free reinforcement learning algorithms.

For future work, it would be enticing to deploy the presented algorithm on a real-robotic agent.

Then, the error in the gradient in the previous term is bounded by

In order to bound the model term we need first to bound the rewards since

Similar to the previous bounds, we can bound now each reward term by

With this result we can bound the total error in models

Then, the gradient error has the form

A.2 PROOF OF LEMMA 4.2

The total variation distance can be bounded by the KL-divergence using the Pinsker's inequality

Then if we assume third order smoothness on our policy, by the Fisher information metric theorem then

Given that θ −θ 2 = α ∇ θ J π − ∇ θĴπ 2 , for a small enough step the following inequality holds Given the bound on the total variation distance, we can now make use of the monotonic improvement theorem to establish an improvement bound in terms of the gradient error.

Let J π (θ) and J π (θ) be the expected return of the policy π θ and πθ under the true dynamics.

Let ρ andρ be the discounted state marginal for the policy π θ and πθ, respectively Then, combining the results from Lemma 4.2 we obtain the desired bound.

In order to show the significance of each component of MAAC, we conducted more ablation studies.

The results are shown in Figure 5 .

Here, we analyze the effect of training the Q-function with data coming from just the real environment, not learning a maximum entropy policy, and increasing the batch size instead of increasing the amount of samples to estimate the value function.

Figure 5: We further test the significance of some components of our method: not use the dynamics to generate data, and only use real data sampled from environments to train policy and Q-functions (real_data), remove entropy from optimization objects (no_entropy), and using a single sample estimate of the pathwise derivative but increase the batch size accordingly (5x batch size).

Considering entropy and using dynamic models to augment data set are both very important.

A.5 EXECUTION TIME COMPARISON

<|TLDR|>

@highlight

Policy gradient through backpropagation through time using learned models and Q-functions. SOTA results in reinforcement learning benchmark environments.