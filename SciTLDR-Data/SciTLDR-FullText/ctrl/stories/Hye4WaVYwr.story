We compare the model-free reinforcement learning with the model-based approaches through the lens of the expressive power of neural networks for policies, $Q$-functions, and dynamics.

We show, theoretically and empirically, that even for one-dimensional continuous state space, there are many MDPs whose optimal $Q$-functions and policies are much more complex than the dynamics.

We hypothesize many real-world MDPs also have a similar property.

For these MDPs, model-based planning is a favorable algorithm, because the resulting policies can approximate the optimal policy significantly better than a neural network parameterization can, and model-free or model-based policy optimization rely on policy parameterization.

Motivated by the theory, we apply a simple multi-step model-based bootstrapping planner (BOOTS) to bootstrap a weak $Q$-function into a stronger policy.

Empirical results show that applying BOOTS on top of model-based or model-free policy optimization algorithms at the test time improves the performance on MuJoCo benchmark tasks.

Model-based deep reinforcement learning (RL) algorithms offer a lot of potentials in achieving significantly better sample efficiency than the model-free algorithms for continuous control tasks.

We can largely categorize the model-based deep RL algorithms into two types: 1.

model-based policy optimization algorithms which learn policies or Q-functions, parameterized by neural networks, on the estimated dynamics, using off-the-shelf model-free algorithms or their variants (Luo et al., 2019; Janner et al., 2019; Kaiser et al., 2019; Kurutach et al., 2018; Feinberg et al., 2018; Buckman et al., 2018) , and 2.

model-based planning algorithms, which plan with the estimated dynamics Nagabandi et al. (2018) ; Chua et al. (2018) ; Wang & Ba (2019) .

A deeper theoretical understanding of the pros and cons of model-based and the model-free algorithms in the continuous state space case will provide guiding principles for designing and applying new sample-efficient methods.

The prior work on the comparisons of model-based and model-free algorithms mostly focuses on their sample efficiency gap, in the case of tabular MDPs (Zanette & Brunskill, 2019; Jin et al., 2018) , linear quadratic regulator (Tu & Recht, 2018) , and contextual decision process with sparse reward (Sun et al., 2019) .

In this paper, we theoretically compare model-based RL and model-free RL in the continuous state space through the lens of approximability by neural networks, and then use the insight to design practical algorithms.

What is the representation power of neural networks for expressing the Qfunction, the policy, and the dynamics?

How do the model-based and model-free algorithms utilize the expressivity of neural networks?

Our main finding is that even for the case of one-dimensional continuous state space, there can be a massive gap between the approximability of Q-function and the policy and that of the dynamics:

The optimal Q-function and policy can be significantly more complex than the dynamics.

We construct environments where the dynamics are simply piecewise linear functions with constant pieces, but the optimal Q-functions and the optimal policy require an exponential (in the horizon) number of linear pieces, or exponentially wide neural networks, to approximate.

1 The approximability gap can also be observed empirically on (semi-) randomly generated piecewise linear dynamics with a decent chance.

(See Figure 1 for two examples.)

When the approximability gap occurs, any deep RL algorithms with policies parameterized by neural networks will suffer from a sub-optimal performance.

These algorithms include both model-free algorithms such as DQN (Mnih et al., 2015) and SAC (Haarnoja et al., 2018) , and model-based policy optimization algorithms such as SLBO (Luo et al., 2019) and MBPO (Janner et al., 2019) .

To validate the intuition, we empirically apply these algorithms to the constructed or the randomly generated MDPs.

Indeed, they fail to converge to the optimal rewards even with sufficient samples, which suggests that they suffer from the lack of expressivity.

However, in such cases, model-based planning algorithms should not suffer from the lack of expressivity, because they only use the learned, parameterized dynamics, which are easy to express.

The policy obtained from the planning is the maximizer of the total future reward on the learned dynamics, and can have an exponential (in the horizon) number of pieces even if the dynamics has only a constant number of pieces.

In fact, even a partial planner can help improve the expressivity of the policy.

If we plan for k steps and then resort to some Q-function for estimating the total reward of the remaining steps, we can obtain a policy with 2 k more pieces than what Q-function has.

We hypothesize that the real-world continuous control tasks also have a more complex optimal Qfunction and a policy than the dynamics.

The theoretical analysis of the synthetic dynamics suggests that a model-based few-steps planner on top of a parameterized Q-function will outperform the original Q-function because of the addtional expressivity introduced by the planning.

We empirically verify the intuition on MuJoCo benchmark tasks.

We show that applying a model-based planner on top of Q-functions learned from model-based or model-free policy optimization algorithms in the test time leads to significant gains over the original Q-function or policy.

In summary, our contributions are:

1.

We construct continuous state space MDPs whose Q-functions and policies are proved to be more complex than the dynamics (Sections 4.1 and 4.2.) 2.

We empirically show that with a decent chance, (semi-)randomly generated piecewise linear MDPs also have complex Q-functions (Section 4.3.) 3.

We show theoretically and empirically that the model-free RL or model-based policy optimization algorithms suffer from the lack of expressivity for the constructed MDPs (Sections 4.3), whereas model-based planning solve the problem efficiently (Section 5.2.) 4.

Inspired by the theory, we propose a simple model-based bootstrapping planner (BOOTS), which can be applied on top of any model-free or model-based Q-learning algorithms at the test time.

Empirical results show that BOOTS improves the performance on MuJoCo benchmark tasks, and outperforms previous state-of-the-art on MuJoCo humanoid environment.

Comparisons with Prior Theoretical Work.

Model-based RL has been extensively studied in the tabular case (see (Zanette & Brunskill, 2019; Azar et al., 2017) and the references therein), but much less so in the context of deep neural networks approximators and continuous state space. (Luo et al., 2019) give sample complexity and convergence guarantees suing principle of optimism in the face of uncertainty for non-linear dynamics.

Below we review several prior results regarding model-based versus model-free dichotomy in various settings.

We note that our work focuses on the angle of expressivity, whereas the work below focuses on the sample efficiency.

Tabular MDPs.

The extensive study in tabular MDP setting leaves little gap in their sample complexity of model-based and model-free algorithms, whereas the space complexity seems to be the main difference. (Strehl et al., 2006) .

The best sample complexity bounds for model-based tabular RL (Azar et al., 2017; Zanette & Brunskill, 2019) and model-free tabular RL (Jin et al., 2018) only differ by a poly(H) multiplicative factor (where H is the horizon.)

Linear Quadratic Regulator.

Dean et al. (2018) and Dean et al. (2017) provided sample complexity bound for model-based LQR.

Recently, Tu & Recht (2018) analyzed sample efficiency of the modelbased and model-free problem in the setting of Linear Quadratic Regulator, and proved an O(d) gap in sample complexity, where d is the dimension of state space.

Unlike tabular MDP case, the space complexity of model-based and model-free algorithms has little difference.

The sample-efficiency gap mostly comes from that dynamics learning has d-dimensional supervisions, whereas Q-learning has only one-dimensional supervision.

Contextual Decision Process (with function approximator).

Sun et al. (2019) prove an exponential information-theoretical gap between mode-based and model-free algorithms in the factored MDP setting.

Their definition of model-free algorithms requires an exact parameterization: the value-function hypothesis class should be exactly the family of optimal value-functions induced by the MDP family.

This limits the application to deep reinforcement learning where overparameterized neural networks are frequently used.

Moreover, a crucial reason for the failure of the model-free algorithms is that the reward is designed to be sparse.

A large family of model-based RL algorithms uses existing model-free algorithms of its variant on the learned dynamics.

MBPO (Janner et al., 2019) , STEVE (Buckman et al., 2018), and MVE (Feinberg et al., 2018) are model-based Q-learning-based policy optimization algorithms, which can be viewed as modern extensions and improvements of the early model-based Q-learning framework, Dyna (Sutton, 1990) .

SLBO (Luo et al., 2019 ) is a model-based policy optimization algorithm using TRPO as the algorithm in the learned environment.

Another way to exploit the dynamics is to use it to perform model-based planning.

Racanière et al. (2017) and Du & Narasimhan (2019) use the model to generated additional extra data to do planning implicitly.

Chua et al. (2018) study how to combine an ensemble of probabilistic models and planning, which is followed by Wang & Ba (2019) , which introduces a policy network to distill knowledge from a planner and provides a prior for the planner.

Piché et al. (2018) uses methods in Sequential Monte Carlo in the context of control as inference.

Oh et al. (2017) trains a Q-function and then perform lookahead planning.

Nagabandi et al. (2018) uses random shooting as the planning algorithm.

Lowrey et al. (2018) uses the dynamics to improve the performance of model-free algorithms.

Heess et al. (2015) backprops through a stochastic computation graph with a stochastic gradient to optimize the policy under the learned dynamics.

Levine & Koltun (2013) distills a policy from trajectory optimization.

Rajeswaran et al. (2016) trains a policy adversarially robust to the worst dynamics in the ensemble.

Clavera et al. (2018) reformulates the problem as a meta-learning problem and using meta-learning algorithms.

Predictron (Silver et al., 2017 ) learns a dynamics and value function and then use them to predict the future reward sequences.

Another line of work focus on how to improve the learned dynamics model.

Many of them use an ensemble of models (Kurutach et al., 2018; Rajeswaran et al., 2016; Clavera et al., 2018) , which are further extended to an ensemble of probabilistic models (Chua et al., 2018; Wang & Ba, 2019

Markov Decision Process.

A Markov Decision Process (MDP) is a tuple S, A, f, r, γ , where S is the state space, A the action space, f : S × A → ∆(S) the transition dynamics that maps a state action pair to a probability distribution of the next state, γ the discount factor, and r ∈ R S×A the reward function.

Throughout this paper, we will consider deterministic dynamics, which, with slight abuse of notation, will be denoted by f : S × A → S. A deterministic policy π : S → A maps a state to an action.

The value function for the policy is defined as is defined

An RL agent aims to find a policy π that maximizes the expected total reward defined as

where µ is the distribution of the initial state.

Let π be the optimal policy, and V the optimal value function (that is, the value function for policy π ).

The value function V π for policy π and optimal value function V satisfy the Bellman equation and Bellman optimality equation, respectively.

Let Q π and Q defines the state-action value function for policy π and optimal state-action value function.

Then, for a deterministic dynamics f , we have

Denote the Bellman operator for dynamics f by Problem Setting and Notations.

In this paper, we focus on continuous state space, discrete action space MDPs with S ⊂ R. We assume the dynamics is deterministic (that is, s t+1 = f (s t , a t )), and the reward is known to the agent.

Let x denote the floor function of x, that is, the greatest integer less than or equal to x. We use I[·] to denote the indicator function.

We show that there exist MDPs in one-dimensional continuous state space that have simple dynamics but complex Q-functions and policies.

Moreover, any polynomial-size neural network function approximator of the Q-function or policy will result in a sub-optimal expected total reward, and learning Q-functions parameterized by neural networks requires fundamentally an exponential number of samples (Section 4.2).

Section 4.3 illustrates the phenomena that Q-function is more complex than the dynamics occurring frequently and naturally even with random MDP, beyond the theoretical construction.

(a) Visualization of dynamics for action a = 0, 1.

(b) The reward function r(s, 0) and r(s, 1).

(c) Approximation of optimal Qfunction Q (s, a)

Figure 2: A visualization of the dynamics, the reward function in the MDP defined in Definition 4.1, and the approximation of its optimal Q-function for the effective horizon H = 4.

We can also construct slightly more involved construction with Lipschitz dynamics and very similar properties.

Please see Appendix C.

Recall that we consider the infinite horizon case and 0 < γ < 1 is the discount factor.

Let H = (1 − γ) −1 be the "effective horizon" -the rewards after H steps become negligible due to the discount factor.

For simplicity, we assume that H > 3 and it is an integer.

(Otherwise we take just take H = (1 − γ) −1 .)

Throughout this section, we assume that the state space S = [0, 1) and the action space A = {0, 1}. Definition 4.1.

Given the effective horizon H = (1 − γ) −1 , we define an MDP M H as follows.

Let κ = 2 −H .

The dynamics f by the following piecewise linear functions with at most three pieces.

The reward function is defined as

The initial state distribution µ is uniform distribution over the state space [0, 1).

The dynamics and the reward function for H = 4 are visualized in Figures 2a, 2b .

Note that by the definition, the transition function for a fixed action a is a piecewise linear function with at most 3 pieces.

Our construction can be modified so that the dynamics is Lipschitz and the same conclusion holds (see Appendix C).

Attentive readers may also realize that the dynamics can be also be written succinctly as f (s, 0) = 2s mod 1 and f (s, 1) = 2s + κ mod 1 2 , which are key properties that we use in the proof of Theorem 4.2 below.

Optimal Q-function Q and the optimal policy π .

Even though the dynamics of the MDP constructed in Definition 4.1 has only a constant number of pieces, the Q-function and policy are very complex: (1) the policy is a piecewise linear function with exponentially number of pieces, (2) the optimal Q-function Q and the optimal value function V are actually fractals that are not continuous anywhere.

These are formalized in the theorem below.

Theorem 4.2.

For s ∈ [0, 1), let s (k) denotes the k-th bit of s in the binary representation.

3 The optimal policy π for the MDP defined in Definition 4.1 has 2 H+1 number of pieces.

In particular,

2 The mod function is defined as: x mod 1 x − x .

More generally, for positive real k, we define x mod k x − k x/k .

3 Or more precisely, we define s

And the optimal value function is a fractal with the expression:

The close-form expression of Q can be computed by Q (s, a) = r(s, a) + V (f (s, a)), which is also a fractal.

We approximate the optimal Q-function by truncating the infinite sum to 2H terms, and visualize it in Figure 2c .

We discuss the main intuitions behind the construction in the following proof sketch of the Theorem.

A rigorous proof of Theorem 4.2) is deferred to Appendix B.1.

Proof Sketch.

The key observation is that the dynamics f essentially shift the binary representation of the states with some addition.

We can verify that the dynamics satisfies f (s, 0) = 2s mod 1 and f (s, 1) = 2s + κ mod 1 where κ = 2 −H .

In other words, suppose s = 0.s (1) s (2) · · · is the binary representation of s, and let left-shift(s) = 0.s

Moreover, the reward function is approximately equal to the first bit of the binary representation

(Here the small negative drift of reward for action a = 1, −2(γ H−1 − γ H ), is only mostly designed for the convenience of the proof, and casual readers can ignore it for simplicity.)

Ignoring carries, the policy pretty much can only affect the H-th bit of the next state s = f (s, a): the H-th bit of s is either equal to (H + 1)-th bit of s when action is 0, or equal its flip when action is 1.

Because the bits will eventually be shifted left and the reward is higher if the first bit of a future state is 1, towards getting higher future reward, the policy should aim to create more 1's.

Therefore, the optimal policy should choose action 0 if the (H + 1)-th bit of s is already 1, and otherwise choose to flip the (H + 1)-th bit by taking action 1.

A more delicate calculation that addresses the carries properly would lead us to the form of the optimal policy (Equation (2).)

Computing the total reward by executing the optimal policy will lead us to the form of the optimal value function (equation (3).) (This step does require some elementary but sophisticated algebraic manipulation.)

With the form of the V , a shortcut to a formal, rigorous proof would be to verify that it satisfies the Bellman equation, and verify π is consistent with it.

We follow this route in the formal proof of Theorem 4.2) in Appendix B.1.

A priori, the complexity of Q or π does not rule out the possibility that there exists an approximation of them that do an equally good job in terms of maximizing the rewards.

However, we show that in this section, indeed, there is no neural network approximation of Q or π with a polynomial width.

We prove this by showing any piecewise linear function with a sub-exponential number of pieces cannot approximate either Q or π with a near-optimal total reward.

Theorem 4.3.

Let M H be the MDP constructed in Definition 4.1.

Suppose a piecewise linear policy π has a near optimal reward in the sense that η(π) ≥ 0.92 · η(π ), then it has to have at least Ω (exp(cH)/H) pieces for some universal constant c > 0.

As a corollary, no constant depth neural networks with polynomial width (in H) can approximate the optimal policy with near optimal rewards.

Consider a policy π induced by a value function Q, that is, π(s) = arg max a∈A Q(s, a).

Then,when there are two actions, the number of pieces of the policy is bounded by twice the number of pieces of Q. This observation and the theorem above implies the following inapproximability result of Q .

Corollary 4.4.

In the setting of Theorem 4.3, let π be the policy induced by some Q. If π is nearoptimal in a sense that η(π) ≥ 0.92 · η(π ), then Q has at least Ω (exp(cH)/H) pieces for some universal constant c > 0.

The intuition behind the proof of Theorem 4.3 is as follows.

Recall that the optimal policy has the form π (s) = I[s (H+1) = 0].

One can expect that any polynomial-pieces policy π behaves suboptimally in most of the states, which leads to the suboptimality of π.

Detailed proof of Theorem 4.3 is deferred to Appendix B.2.

Beyond the expressivity lower bound, we also provide an exponential sample complexity lower bound for Q-learning algorithms parameterized with neural networks (see Appendix B.4).

In this section, we show the phenomena that the Q-function not only occurs in the crafted cases as in the previous subsection, but also occurs more robustly with a decent chance for (semi-) randomly generated MDPs.

(Mathematically, this says that the family of MDPs with such a property is not a degenerate measure-zero set.)

It is challenging and perhaps requires deep math to characterize the fractal structure of Q-functions for random dynamics, which is beyond the scope of this paper.

Instead, we take an empirical approach here.

We generate random piecewise linear and Lipschitz dynamics, and compute their Q-functions for the finite horizon, and then visualize the Q-functions or count the number of pieces in the Q-functions.

We also use DQN algorithm (Mnih et al., 2015) with a finite-size neural network to learn the Q-function.

We set horizon H = 10 for simplicity and computational feasibility.

The state and action space are [0, 1) and {0, 1} respectively.

We design two methods to generate random or semi-random piecewise dynamics with at most four pieces.

First, we have a uniformly random method, called RAND, where we independently generate two piecewise linear functions for f (s, 0) and f (s, 1), by generating random positions for the kinks, generating random outputs for the kinks, and connecting the kinks by linear lines (See Appendix D.1 for a detailed description.)

In the second method, called SEMI-RAND, we introduce a bit more structure in the generation process, towards increasing the chance to see the phenomenon.

The functions f (s, 0) and f (s, 1) have 3 pieces with shared kinks.

We also design the generating process of the outputs at the kinks so that the functions have more fluctuations.

The reward for both of the two methods is r(s, a) = s, ∀a ∈ A. (See Appendix D.1 for a detailed description.)

Figure 1 illustrates the dynamics of the generated MDPs from SEMI-RAND.

More details of empirical settings can be found in Appendix D.1.

The optimal policy and Q can have a large number of pieces.

Because the state space has one dimension, and the horizon is 10, we can compute the exact Q-functions by recursively applying Bellman operators, and count the number of pieces.

We found that, 8.6% fraction of the 1000 MDPs independently generated from the RAND method has policies with more than 100 pieces, much larger than the number of pieces in the dynamics (which is 4).

Using the SEMI-RAND method, a 68.7% fraction of the MDPs has polices with more than 10 3 pieces.

In Section D.1, we plot the histogram of the number of pieces of the Q-functions.

Figure 1 visualize the Q-functions and dynamics of two MDPs generated from RAND and SEMI-RAND method.

These results suggest that the phenomenon that Q-function is more complex than dynamics is not a degenerate phenomenon and can occur with non-zero measure.

For more empirical results, see Appendix D.2.

Model-based policy optimization methods also suffer from a lack of expressivity.

As an implication of our theory in the previous section, when the Q-function or the policy are too complex to be approximated by a reasonable size neural network, both model-free algorithms or model-based policy optimization algorithms will suffer from the lack of expressivity, and as a consequence, the sub-optimal rewards.

We verify this claim on the randomly generated MDPs discussed in Section 4.3, by running DQN (Mnih et al., 2015) , SLBO (Luo et al., 2019) , and MBPO (Janner et al., 2019) with various architecture size.

For the ease of exposition, we use the MDP visualized in the bottom half of Figure 1 .

The optimal policy for this specific MDP has 765 pieces, and the optimal Q-function has about 4 × 10 4 number of pieces, and we can compute the optimal total rewards.

First, we apply DQN to this environment by using a two-layer neural network with various widths to parameterize the Q-function.

The training curve is shown in Figure 3 (Left).

Model-free algorithms Figure 1 .

The number after the acronym is the width of the neural network used in the parameterization of Q.

We see that even with sufficiently large neural networks and sufficiently many steps, these algorithms still suffers from bad approximability and cannot achieve optimal reward. (Right):

Performance of BOOTS + DQN with various planning steps.

A near-optimal reward is achieved with even k = 3, indicating that the bootstrapping with the learned dynamics improves the expressivity of the policy significantly.

Algorithm 1 Model-based Bootstrapping Planner (BOOTS) + RL Algorithm X 1: training: run Algorithm X, store the all samples in the set R, store the learned Q-function Q, and the learned dynamicsf if it is available in Algorithm X. 2: testing:

iff is not available, learnf from the data in R Given: query oracle for function Q andf 3:

using a zero-th order optimization algorithm (which only requires oracle query of the function value) such as cross-entropy method or random shooting.

can not find near-optimal policy even with 2 14 hidden neurons and 1M trajectories, which suggests that there is a fundamental approximation issue.

This result is consistent with , in a sense that enlarging Q-network improves the performance of DQN algorithm at convergence.

Second, we apply SLBO and MBPO in the same environment.

Because the policy network and Q-function in SLOBO and MBPO cannot approximate the optimal policy and value function, we see that they fail to achieve near-optimal rewards, as shown in Figure 3 (Left).

Our theory and experiments in Section 4.2 and 4.3 demonstrate that when the Q-function or the policy is complex, model-free or model-based policy optimization algorithms will suffer from the lack of expressivity.

The intuition suggests that model-based planning algorithms will not suffer from the lack of expressivity because the final policy is not represented by a neural network.

For the construction in Section 4.1, we can actually prove that even a few-steps planner can bootstrap the expressivity of the Q-function (formalized in Theorem 5.1 below).

Inspired the theoretical result, we apply a simple k-step model-based bootstrapping planner on top of existing Q-functions (trained from either model-based or model-free approach) in the test time, on either the one-dimensional MDPs considered in Section 4 or the continuous control benchmark tasks in MuJoCo.

The bootstrapping planner is reminiscent of MCTS using in AlphaGo (Silver et al., 2016; .

However, here, we use the learned dynamics and deal with the continuous state space.

The test policies for MBSAC and SAC are the deterministic policy that takes the mean of the output of the policy network, because the deterministic policy performs better than the stochastic policy in the test time.

Given a function Q that is potentially not expressive enough to approximate the optimal Q-function, we can apply the Bellman operator with a learned dynamicsf for k times to get a bootstrapped version of Q:

where s 0 = s, a 0 = a and s h+1 =f (s h , a h ).

Given the bootstrapped version, we can derive a greedy policy w.r.t it:

Algorithm 1, called BOOTS summarizes how to apply the planner on top of any RL algorithm with a Q-function (straightforwardly).

For the MDPs constructed in Section 4.1, we can prove that representing the optimal Q-function by B k f

[Q] requires fewer pieces in Q than representing the optimal Q-function by Q directly.

Theorem 5.1.

Consider the MDP M H defined in Definition 4.1.

There exists a constant-piece piecewise linear dynamicsf and a 2 H−k+1 -piece piecewise linear function Q, such that the bootstrapped policy π boots k,Q,f (s) achieves the optimal total rewards.

By contrast, recall that in Theorem 4.3, we show that approximating the optimal Q-function directly with a piecewise linear function requires ≈ 2 H piecewise.

Thus we have a multiplicative factor of 2 k gain in the expressivity by using the bootstrapped policy.

Here the exponential gain is only magnificent enough when k is close to H because the gap of approximability is huge.

However, in more realistic settings -the randomly-generated MDPs and the MuJoCo environment -the bootstrapping planner improvs the performance significantly as shown in the next subsection.

BOOTS on random piecewise linear MDPs.

We implement BOOTS (Algorithm 1) with various steps of planning and with the learned dynamics.

4 .

The planner is an exponential-time planner which enumerates all the possible future sequence of actions.

We also implement bootstrapping with partial planner with varying planning horizon.

As shown in Figure 3 , BOOTS + DQN not only has the best sample-efficiency, but also achieves the optimal reward.

In the meantime, even a partial planner helps to improve both the sample-efficiency and performance.

More details of this experiment are deferred to Appendix D.3.

BOOTS on MuJoCo environments.

We work with the OpenAI Gym environments (Brockman et al., 2016) based on the Mujoco simulator (Todorov et al., 2012) with maximum horizon 1000 and discount factor 1.

We apply BOOTS on top of three algorithms: (a) SAC (Haarnoja et al., 2018) We use k = 4 steps of planning unless explicitly mentioned otherwise in the ablation study (Section A.2).

In Figure 4 , we compare BOOTS+SAC with SAC, and BOOTS + MBSAC with MBSAC on Gym Ant and Humanoid environments, and demonstrate that BOOTS can be used on top of existing strong baselines.

We found that BOOTS has little help for other simpler environments, and we suspect that those environments have much less complex Q-functions so that our theory and intuitions do not necessarily apply.

(See Section A.2 for more ablation study.)

In Figure 5 , we compare BOOTS+MBSAC and BOOTS+MBPO with other MBPO, SAC, and STEVE (Buckman et al., 2018) 5 on the humanoid environment.

We see a strong performance surpassing the previous state-of-the-art MBPO.

Our study suggests that there exists a significant representation power gap of neural networks between for expressing Q-function, the policy, and the dynamics in both constructed examples and empirical benchmarking environments.

We show that our model-based bootstrapping planner BOOTS helps to overcome the approximation issue and improves the performance in synthetic settings and in the difficult MuJoCo environments.

We raise some interesting open questions.

• Can we theoretically generalize our results to high-dimensional state space, or continuous actions space?

Can we theoretically analyze the number of pieces of the optimal Q-function of a stochastic dynamics?

• In this paper, we measure the complexity by the size of the neural networks.

It's conceivable that for real-life problems, the complexity of a neural network can be better measured by its weights norm.

Could we build a more realistic theory with another measure of complexity?

• The BOOTS planner comes with a cost of longer test time.

How do we efficiently plan in high-dimensional dynamics with a long planning horizon?

• The dynamics can also be more complex (perhaps in another sense) than the Q-function in certain cases.

How do we efficiently identify the complexity of the optimal Q-function, policy, and the dynamics, and how do we deploy the best algorithms for problems with different characteristics? (Luo et al., 2019) , the stochasticity in the dynamics can play a similar role as the model ensemble.

Our algorithm is a few times faster than MBPO in wall-clock time.

It performs similarlty to MBPO on Humanoid, but a bit worse than MBPO in other environments.

In MBSAC, we use SAC to optimize the policy π β and the Q-function Q ϕ .

We choose SAC due to its sample-efficiency, simplicity and off-policy nature.

We mix the real data from the environment and the virtual data which are always fresh and are generated by our learned dynamics modelf θ .

Algorithm 2 MBSAC 1: Parameterize the policy π β , dynamicsf θ , and the Q-function Q ϕ by neural networks.

Initialize replay buffer B with n init steps of interactions with the environments by a random policy, and pretrain the dynamics on the data in the replay buffer.

2: t ← 0, and sample s 0 from the initial state distribution.

3: for n iter iterations do Perform action a t ∼ π β (·|s t ) in the environment, obtain s as the next state from the environment.

s t+1 ← s , and add the transition (s t , a t , s t+1 , r t ) to B.

t ← t + 1.

If t = T or the trajectory is done, reset to t = 0 and sample s 0 from the initial state distribution.

for n policy iterations do

for n model iterations do

Optimizef θ with a mini-batch of data from B by one step of Adam.

Sample n real data B real and n start data B start from B.

Perform q steps of virtual rollouts usingf θ and policy π β starting from states in B start ; obtain B virtual .

Update π β and Q ϕ using the mini-batch of data in B real ∪ B virtual by SAC.

For Ant, we modify the environment by adding the x and y axis to the observation space to make it possible to compute the reward from observations and actions.

For Humanoid, we add the position of center of mass.

We don't have any other modifications.

All environments have maximum horizon 1000.

For the policy network, we use an MLP with ReLU activation function and two hidden layers, each of which contains 256 hidden units.

For the dynamics model, we use a network with 2 Fixup blocks (Zhang et al., 2019), with convolution layers replaced by a fully connected layer.

We found out that with similar number of parameters, fixup blocks leads to a more accurate model in terms of validation loss.

Each fixup block has 500 hidden units.

We follow the model training algorithm in Luo et al. (2019) in which non-squared 2 loss is used instead of the standard MSE loss.

Planning with oracle dynamics and more environments.

We found that BOOTS has smaller improvements on top of MBSAC and SAC for the environment Cheetah and Walker.

To diagnose the issue, we also plan with an oracle dynamics (the true dynamics).

This tells us whether the lack of improvement comes from inaccurate learned dynamics.

The results are presented in two ways in Figure 6 and Figure 7 .

In Figure 6 , we plot the mean rewards and the standard deviation of various methods across the randomness of multiple seeds.

However, the randomness from the seeds somewhat obscures the gains of BOOTS on each individual run.

Therefore, for completeness, we also plot the relative gain of BOOTS on top of MBSAC and SAC, and the standard deviation of the gains in Figure 7 .

From Figure 7 we can see planning with the oracle dynamics improves the performance in most of the cases (but with various amount of improvements).

However, the learned dynamics sometimes not always can give an improvement similar to the oracle dynamics.

This suggests the learned dynamics is not perfect, but oftentimes can lead to good planning.

This suggests the expressivity of the Q-functions varies depending on the particular environment.

How and when to learn and use a learned dynamics for planning is a very interesting future open question.

The effect of planning horizon.

We experimented with different planning horizons in Figure 8 .

By planning with a longer horizon, we can earn slightly higher total rewards for both MBSAC and SAC.

Planning horizon k = 16, however, does not work well.

We suspect that it's caused by the compounding effect of the errors in the dynamics.

In this section we provide the proofs omitted in Section 4.

Proof of Theorem 4.2.

Since the solution to Bellman optimal equations is unique, we only need to verify that V and π defined in equation (1) satisfy the following,

Recall that s (i) is the i-th bit in the binary representation of s, that is,

, which ensures the H-bit of the next state is 1, we haveŝ

For simplicity, define ε = 2(γ

Now, we verify Eq. (10) by plugging in the proposed solution (namely, Eq. (13)).

As a result,

which verifies Eq. (10).

In the following we verify Eq. (11).

Consider any a = π (s).

Lets = f (s, a) for shorthand.

Note thats (i) = s (i+1) for i > H. As a result,

For the case where s (H+1) = 0, we have π (s) = 1.

For a = 0,s

where the last inequality holds when γ H − ε > 0, or equivalently, γ > 2/3.

For the case where s (H+1) = 1, we have π (s) = 0.

For a = 1, we have s

, where we define the max of an empty set is 0.

The dynamics f (s, 1) implies thats

Therefore,

In both cases, we have V − γV (s) > r(s, a) for a = π (s), which proves Eq. (11).

For a fixed parameter H, let z(π) be the number of pieces in π.

For a policy π, define the state distribution when acting policy π at step h as µ π h .

In order to prove Theorem 4.3, we show that if 1/2 − 2Hz(π)/2 H < 0.3, then η(π) < 0.92η(π ).

The proof is based on the advantage decomposition lemma.

Lemma B.1 (Advantage Decomposition Lemma (Schulman et al., 2015; Kakade & Langford, 2002)

Corollary B.2.

For any policy π, we have

Intuitively speaking, since π = I[s (H+1) = 0], the a policy π with polynomial pieces behaves suboptimally in most of the states.

Lemma B.3 shows that the single-step suboptimality gap V (s)− Q (s, π(s)) is large for a constant portion of the states.

On the other hand, Lemma B.4 proves that the state distribution µ π h is near uniform, which means that suboptimal states can not be avoided.

Combining with Corollary B.2, the suboptimal gap of policy π is large.

Let ν h (k) = inf s∈ k µ π h (s), then by advantage decomposition lemma (namely, Corollary B.2), we have

By Lemma B.4 and union bound, we get

For the sake of contradiction, we assume z(π) = o (exp(cH)/H), then for large enough H we have,

49 for all h ≤ 10H.

Consequently, for H > 500, we have

Now, since η(π ) ≤ 1/(1 − γ), we have η(π) < 0.92η(π ).

Therefore for near-optimal policy π, z(π) = Ω (exp(cH)/H) .

In this section, we present the proofs of two lemmas used in Section B.1

Proof of Lemma B.3.

Note that for any k ∈ K, s (H) = 1, ∀s ∈ k .

Now fix a parameter k ∈ K. Suppose π(s) = a i for s ∈ k .

Then for any s such that s (H+1) + i = 1, we have

For H > 500, we have γ H − ε > 0.366.

Therefore,

Proof of Lemma B.4.

Now let us fix a parameter H and policy π.

For every h, we prove by induction that there exists a function ξ h (s), such that

For the base case h = 1, we define

as the left and right endpoints of

be the set of 2 solutions of equation

where 0 ≤ x < 1, and we define y

k ) can reach states in interval k by a single transition.

We define a set I k = {i :

That is, the intervals where policy π acts unanimously.

Consequently, for i ∈ I k , the set {s :

an interval of length 2 −H−1 , and has the form

for some integer w

Now, the density ξ h+1 (s) for s ∈ k is defined as,

The intuition of the construction is that, we discard those density that cause non-uniform behavior (that is, the density in intervals [x

When the number of pieces of π is small, we can keep most of the density.

Now, statement (b) is naturally satisfied by definition of ξ h+1 .

We verify statement (a) and (c) below.

For any set B ⊆ k , let (T π ) −1 (B) = {s ∈ S : f (s, π(s)) ∈ B} be the inverse of Markov transition T π .

Then we have,

where | · | is the shorthand for standard Lebesgue measure.

By definition, we have

which verifies statement (a).

For statement (c), recall that S = [0, 1) is the state space.

Note that T π preserve the overall density.

That is (T π ξ h ) (S) = ξ h (S).

We only need to prove that

and statement (c) follows by induction.

By definition of ξ h+1 (s) and the induction hypothesis that ξ h (s) ≤ 1, we have

On the other hand, for any s ∈ S, the set {k

k )} has cardinality 2, which means that one intermittent point of π can correspond to at most 2 intervals that are not in I k for some k. Thus, we have

which proves statement (c).

Recall that corollary 4.4 says that in order to find a near-optimal policy by a Q-learning algorithm, an exponentially large Q-network is required.

In this subsection, we show that even if an exponentially large Q-network is applied for Q learning, still we need to collect an exponentially large number of samples, ruling out the possibility of efficiently solving the constructed MDPs with Q-learning algorithms.

Towards proving the sample complexity lower bound, we consider a stronger family of Q-learning algorithm, Q-learning with Oracle (Algorithm 3).

We assume that the algorithm has access to a Q-ORACLE, which returns the optimal Q-function upon querying any pair (s, a) during the training process.

Q-learning with Oracle is conceptually a stronger computation model than the vanilla Q-learning algorithm, because it can directly fit the Q functions with supervised learning, without relying on the rollouts or the previous Q function to estimate the target Q value.

Theorem B.5 proves a sample complexity lower bound for Q-learning algorithm on the constructed example.

Require: A hypothesis space Q of Q-function parameterization.

1: Sample s 0 ∼ µ from the initial state distribution µ 2: for i = 1, 2, · · · , n do 3:

Decide whether to restart the trajectory by setting s i ∼ µ based on historical information 4:

Query Q-ORACLE to get the function Q (s i , ·).

Apply any action a i (according to any rule) and sample s i+1 ∼ f (s i , a i ).

6: Learn the Q-function that fit all the data the best:

Return the greedy policy according to Q.

Theorem B.5 (Informal Version of Theorem B.7).

Suppose Q is an infinitely-wide two-layer neural networks, and R(Q) is 1 norm of the parameters and serves as a tiebreaker.

Then, any instantiation of the Q-LEARNING WITH ORACLE algorithm requires exponentially many samples to find a policy π such that η(π) > 0.99η(π ).

Formal proof of Theorem B.5 is given in Appendix B.5.

The proof of Theorem B.5 is to exploit the sparsity of the solution found by minimal-norm tie-breaker.

It can be proven that there are at most O(n) non-zero neurons in the minimal-norm solution, where n is the number of data points.

The proof is completed by combining with Theorem 4.3.

A two-layer ReLU neural net Q(s, ·) with input s is of the following form,

where d is the number of hidden neurons.

w i,a , c a , k i , b i are parameters of this neural net, where c i,a , b i are bias terms.

[x] + is a shorthand for ReLU activation I[x > 0]x.

Now we define the norm of a neural net.

Definition B.6 (Norm of a Neural Net).

The norm of a two-layer ReLU neural net is defined as,

Recall that the Q-learning with oracle algorithm finds the solution by the following supervised learning problem,

Then, we present the formal version of theorem B.5.

Theorem B.7.

Let Q be the minimal 1 norm solution to Eq. (24), and π the greedy policy according to Q. When n = o(exp(cH)/H), we have η(π) < 0.99η(π ).

The proof of Theorem B.5 is by characterizing the minimal-norm solution, namely the sparsity of the minimal-norm solution as stated in the next lemma.

Lemma B.8.

The minimal-norm solution to Eq. (24) has at most 32n + 1 non-zero neurons.

That is, |{i : k i = 0}| ≤ 32n + 1.

We first present the proof of Theorem B.7, followed by the proof of Theorem B.8.

Proof of Theorem B.7.

Recall that the policy is given by π(s) = arg max a∈A Q(s, a).

For a Qfunction with 32n + 2 pieces, the greedy policy according to Q(s, a) has at most 64n + 4 pieces.

Combining with Theorem 4.3, in order to find a policy π such that η(π) > 0.99η(π ), n needs to be exponentially large (in effective horizon H).

Proof of Lemma B.8 is based on merging neurons.

Let

, and c = (c 1 , c 2 ).

In vector form, neural net defined in Eq. (22) can be written as,

First we show that neurons with the same x i can be merged together.

Lemma B.9.

Consider the following two neurons,

with k 1 > 0, k 2 > 0.

If x 1 = x 2 , then we can replace them with one single neuron of the form k [x − x 1 ] + w without changing the output of the network.

Furthermore, if w 1 = 0, w 2 = 0, the norm strictly decreases after replacement.

Proof.

We set k = |k 1 w 1 + k 2 w 2 | 1 , and w = (k 1 w 1 + k 2 w 2 )/k , where |w| 1 represents the 1-norm of vector w.

Then, for all s ∈ R,

The norm of the new neuron is |k | + |w | 1 .

By calculation we have,

Note that the inequality (a) is strictly less when |k 1 w 1 | 1 = 0 and |k 2 w 2 | 1 = 0.

Next we consider merging two neurons with different intercepts between two data points.

Without loss of generality, assume the data points are listed in ascending order.

That is, s i ≤ s i+1 .

Lemma B.10.

Consider two neurons

with k 1 > 0, k 2 > 0.

If s i ≤ x 0 < x 0 + δ ≤ s i+1 for some 1 ≤ i ≤ n, then the two neurons can replaced by a set of three neurons,

such that for s ≤ s i or s ≥ s i+1 , the output of the network is unchanged.

Furthermore, if δ ≤ (s i+1 − s i )/16 and |w 1 | 1 = 0, |w 2 | 1 = 0, the norm decreases strictly.

Proof.

For simplicity, define ∆ = s i+1 − s i .

We set

Note that for s ≤ s i , all of the neurons are inactive.

For s ≥ s i+1 , all of the neurons are active, and

which means that the output of the network is unchanged.

Now consider the norm of the two networks.

Without loss of generality, assume |k 1 w 1 | 1 > |k 2 w 2 | 1 .

The original network has norm |k 1 | + |w 1 | 1 + |k 2 | + |w 2 | 1 .

And the new network has norm

where the inequality (a) is a result of Lemma E.1, and is strictly less when |w 1 | 1 = 0, |w 2 | 1 = 0.

Similarly, two neurons with k 1 < 0 and k 2 < 0 can be merged together.

Now we are ready to prove Lemma B.8.

As hinted by previous lemmas, we show that between two data points, there are at most 34 non-zero neurons in the minimal norm solution.

Proof of Lemma B.8.

Consider the solution to Eq. (24).

Without loss of generality, assume that s i ≤ s i+1 .

In the minimal norm solution, it is obvious that |w i | 1 = 0 if and only if k i = 0.

Therefore we only consider those neurons with k i = 0, denoted by index 1 ≤ i ≤ d .

Next we prove that in the minimal norm solution, |B t | ≤ 15.

For the sake of contradiction, suppse |B t | > 15.

Then there exists i, j such that,

, and k i > 0, k j > 0.

By Lemma B.10, we can obtain a neural net with smaller norm by merging neurons i, j together without violating Eq. (24), which leads to contradiction.

By Lemma B.9, |B t | ≤ 15 implies that there are at most 15 non-zero neurons with s t < −b i /k i < s t+1 and k i > 0.

For the same reason, there are at most 15 non-zero neurons with

On the other hand, there are at most 2 non-zero neurons with s t = −b i /k i for all t ≤ n, and there are at most 1 non-zero neurons with −b i /k i < s 1 .

Therefore, we have d ≤ 32n + 1.

B.6 PROOF OF THEOREM 5.1

In this section we present the full proof of Theorem 5.1.

Proof.

First we define the true trajectory estimator

the true optimal action sequence

and the true optimal trajectory

It follows from the definition of optimal policy that, a j = π (s j ).

Consequently we have

Define the set G = {s :

We claim that the following function satisfies the statement of Theorem 5.1

Since s k ∈ G, and s k ∈ G for s k generated by non-optimal action sequence, we have

where the second inequality comes from the optimality of action sequence a h .

As a consequence, for any

In this section, we present an extension to our construction such that the dynamics is Lipschitz.

The action space is A = {0, 1, 2, 3, 4}. We define CLIP(x) = max{min{x, 1}, 0}. Definition C.1.

Given effective horizon H = (1 − γ) −1 , we define an MDP M H as follows.

Let κ = 2 −H .

The dynamics is defined as

Reward function is given by

The intuition behind the extension is that, we perform the mod operation manually.

The following theorem is an analog to Theorem 4.2.

Theorem C.2.

The optimal policy π for M H is defined by,

And the corresponding optimal value function is,

We can obtain a similar upper bound on the performance of policies with polynomial pieces.

Theorem C.3.

Let M H be the MDP constructed in Definition C.1.

Suppose a piecewise linear policy π has a near optimal reward in the sense that η(π) ≥ 0.99 · η(π ), then it has to have at least Ω (exp(cH)/H) pieces for some universal constant c > 0.

The proof is very similar to that for Theorem 4.3.

One of the difference here is to consider the case where f (s, a) = 0 or f (s, a) = 1 separately.

Attentive readers may notice that the dynamics where f (s, a) = 0 or f (s, a) = 1 may destroy the "near uniform" behavior of state distribution µ π h (see Lemma B.4).

Here we show that such destroy comes with high cost.

Formally speaking, if the clip is triggered in an interval, then the averaged single-step suboptimality gap is 0.1/(1 − γ).

for large enough H.

Proof.

Without loss of generality, we consider the case where f (s, π(s)) = 0.

The proof for f (s, π(s)) = 1 is essentially the same.

By elementary manipulation, we have

RAND method.

As stated in Section 4.3, the RAND method generates kinks {x i } and the corresponding values {x i } randomly.

In this method, the generated MDPs are with less structure.

The details are shown as follows.

• State space S = [0, 1).

• Action space A = {0, 1}.

• Number of pieces is fixed to 3.

The positions of the kinks are generated by, x i ∼ U (0, 1) for i = 1, 2 and x 0 = 0, x 1 = 1.

The values are generated by x i ∼ U (0, 1).

• The reward function is given by r(s, a) = s, ∀s ∈ S, a ∈ A.

• The horizon is fixed as H = 10.

• Initial state distribution is U (0, 1).

Figure 1 visualizes one of the RAND-generated MDPs with complex Q-functions.

SEMI-RAND method.

In this method, we add some structures to the dynamics, resulting in a more significant probability that the optimal policy is complex.

We generate dynamics with fix and shared kinks, generate the output at the kinks to make the functions fluctuating.

The details are shown as follows.

• State space S = [0, 1).

• Action space A = {0, 1}.

• Number of pieces is fixed to 3.

The positions of the kinks are generated by, x i = i/3, ∀0 ≤ i ≤ 3.

And the values are generated by x i ∼ 0.65 × I[i mod 2 = 0] + 0.35 × U (0, 1).

• The reward function is r(s, a) = s for all a ∈ A.

• The horizon is fixed as H = 10.

• Initial state distribution is U (0, 1).

We randomly generate 10 3 1-dimensional MDPs whose dynamics has constant number of pieces.

The histogram of number of pieces in optimal policy π is plotted.

As shown in Figure 9 , even for horizon H = 10, the optimal policy tends to have much more pieces than the dynamics.

• The Q-network is a fully connected neural net with one hidden-layer.

The width of the hidden-layer is varying.

• The optimizer is SGD with learning rate 0.001 and momentum 0.9.

• The size of replay buffer is 10 4 .

• Target-net update frequency is 50.

• Batch size in policy optimization is 128.

• The behavior policy is greedy policy according to the current Q-network with -greedy.

exponentially decays from 0.9 to 0.01.

Specifically, = 0.01 + 0.89 exp(−t/200) at the t-th episode.

Implementation details of MBPO algorithm For the model-learning step, we use 2 loss to train our model, and we use Soft Actor-Critic (SAC) (Haarnoja et al., 2018) in the policy optimization step.

The parameters are set as,

• number of hidden neurons in model-net: 32,

• number of hidden neurons in value-net: 512,

• optimizer for model-learning: Adam with learning rate 0.001.

• temperature: τ = 0.01,

• the model rollout steps: M = 5,

• the length of the rollout: k = 5,

• number of policy optimization step: G = 5.

Other hyper-parameters are kept the same as DQN algorithm.

Implementation details of TRPO algorithm For the model-learning step, we use 2 loss to train our model.

Instead of TRPO (Schulman et al., 2015) , we use PPO (Schulman et al., 2017) as policy optimizer.

The parameters are set as,

• number of hidden neurons in model-net: 32,

• number of hidden neurons in policy-net: 512,

• number of hidden neurons in value-net: 512,

• optimizer: Adam with learning rate 0.001,

• number of policy optimization step: 5.

• The behavior policy is -greedy policy according to the current policy network.

exponential decays from 0.9 to 0.01.

Specifically, = 0.01 + 0.89 exp(−t/20000) at the t-th episode.

Implementation details of Model-based Planning algorithm The perfect model-based planning algorithm iterates between learning the dynamics from sampled trajectories, and planning with the learned dynamics (with an exponential time algorithm which enumerates all the possible future sequence of actions).

The parameters are set as,

• number of hidden neurons in model-net: 32,

• optimizer for model-learning: Adam with learning rate 0.001.

Implementation details of bootstrapping The training time behavior of the algorithm is exactly like DQN algorithm, except that the number of hidden neurons in the Q-net is set to 64.

Other parameters are set as,

• number of hidden neurons in model-net: 32,

• optimizer for model-learning: Adam with learning rate 0.001.

• planning horizon varies.

In this section, we present the technical lemmas used in this paper.

Lemma E.1.

For A, B, C, D ≥ 0 and AC ≥ BD, we have

Furthermore, when BD > 0, the inequality is strict.

Proof.

Note that A + B + And when BD > 0, the inequality is strict.

<|TLDR|>

@highlight

We compare deep model-based and model-free RL algorithms by studying the approximability of $Q$-functions, policies, and dynamics by neural networks. 