Policy gradient methods have enjoyed great success in deep reinforcement learning but suffer from high variance of gradient estimates.

The high variance problem is particularly exasperated in problems with long horizons or high-dimensional action spaces.

To mitigate this issue, we derive a bias-free action-dependent baseline for variance reduction which fully exploits the structural form of the stochastic policy itself and does not make any additional assumptions about the MDP.

We demonstrate and quantify the benefit of the action-dependent baseline through both theoretical analysis as well as numerical results, including an analysis of the suboptimality of the optimal state-dependent baseline.

The result is a computationally efficient policy gradient algorithm, which scales to high-dimensional control problems, as demonstrated by a synthetic 2000-dimensional target matching task.

Our experimental results indicate that action-dependent baselines allow for faster learning on standard reinforcement learning benchmarks and high-dimensional hand manipulation and synthetic tasks.

Finally, we show that the general idea of including additional information in baselines for improved variance reduction can be extended to partially observed and multi-agent tasks.

Deep reinforcement learning has achieved impressive results in recent years in domains such as video games from raw visual inputs BID10 , board games , simulated control tasks BID16 , and robotics ).

An important class of methods behind many of these success stories are policy gradient methods BID28 BID22 BID5 BID18 BID11 , which directly optimize parameters of a stochastic policy through local gradient information obtained by interacting with the environment using the current policy.

Policy gradient methods operate by increasing the log probability of actions proportional to the future rewards influenced by these actions.

On average, actions which perform better will acquire higher probability, and the policy's expected performance improves.

A critical challenge of policy gradient methods is the high variance of the gradient estimator.

This high variance is caused in part due to difficulty in credit assignment to the actions which affected the future rewards.

Such issues are further exacerbated in long horizon problems, where assigning credits properly becomes even more challenging.

To reduce variance, a "baseline" is often employed, which allows us to increase or decrease the log probability of actions based on whether they perform better or worse than the average performance when starting from the same state.

This is particularly useful in long horizon problems, since the baseline helps with temporal credit assignment by removing the influence of future actions from the total reward.

A better baseline, which predicts the average performance more accurately, will lead to lower variance of the gradient estimator.

The key insight of this paper is that when the individual actions produced by the policy can be decomposed into multiple factors, we can incorporate this additional information into the baseline to further reduce variance.

In particular, when these factors are conditionally independent given the current state, we can compute a separate baseline for each factor, whose value can depend on all quantities of interest except that factor.

This serves to further help credit assignment by removing the influence of other factors on the rewards, thereby reducing variance.

In other words, information about the other factors can provide a better evaluation of how well a specific factor performs.

Such factorized policies are very common, with some examples listed below.• In continuous control and robotics tasks, multivariate Gaussian policies with a diagonal covariance matrix are often used.

In such cases, each action coordinate can be considered a factor.

Similarly, factorized categorical policies are used in game domains like board games and Atari.• In multi-agent and distributed systems, each agent deploys its own policy, and thus the actions of each agent can be considered a factor of the union of all actions (by all agents).

This is particularly useful in the recent emerging paradigm of centralized learning and decentralized execution BID2 BID9 .

In contrast to the previous example, where factorized policies are a common design choice, in these problems they are dictated by the problem setting.

We demonstrate that action-dependent baselines consistently improve the performance compared to baselines that use only state information.

The relative performance gain is task-specific, but in certain tasks, we observe significant speed-up in the learning process.

We evaluate our proposed method on standard benchmark continuous control tasks, as well as on a high-dimensional door opening task with a five-fingered hand, a synthetic high-dimensional target matching task, on a blind peg insertion POMDP task, and a multi-agent communication task.

We believe that our method will facilitate further applications of reinforcement learning methods in domains with extremely highdimensional actions, including multi-agent systems.

Videos and additional results of the paper are available at https://sites.google.com/view/ad-baselines.

Three main classes of methods for reinforcement learning include value-based methods BID26 , policy-based methods BID28 BID5 BID18 , and actor-critic methods BID6 BID14 BID11 .

Valuebased and actor-critic methods usually compute a gradient of the objective through the use of critics, which are often biased, unless strict compatibility conditions are met BID22 BID6 .

Such conditions are rarely satisfied in practice due to the use of stochastic gradient methods and powerful function approximators.

In comparison, policy gradient methods are able to compute an unbiased gradient, but suffer from high variance.

Policy gradient methods are therefore usually less sample efficient, but can be more stable than critic-based methods BID1 .A large body of work has investigated variance reduction techniques for policy gradient methods.

One effective method to reduce variance without introducing bias is through using a baseline, which has been widely studied BID21 BID27 BID3 .

However, fully exploiting the factorizability of the policy probability distribution to further reduce variance has not been studied.

Recently, methods like Q-Prop BID4 ) make use of an action-dependent control variate, a technique commonly used in Monte Carlo methods and recently adopted for RL.

Since Q-Prop utilizes off-policy data, it has the potential to be more sample efficient than pure on-policy methods.

However, Q-prop is significantly more computationally expensive, since it needs to perform a large number of gradient updates on the critic using the off-policy data, thus not suitable with fast simulators.

In contrast, our formulation of action-dependent baselines has little computational overhead, and improves the sample efficiency compared to on-policy methods with state-only baseline.

The idea of using additional information in the baseline or critic has also been studied in other contexts.

Methods such as Guided Policy Search BID13 and variants train policies that act on high-dimensional observations like images, but use a low dimensional encoding of the problem like joint positions during the training process.

Recent efforts in multi-agent systems BID2 BID9 ) also use additional information in the centralized training phase to speed-up learning.

However, using the structure in the policy parameterization itself to enhance the learning speed, as we do in this work, has not been explored.

In this section, we establish the notations used throughout this paper, as well as basic results for policy gradient methods, and variance reduction via baselines.

This paper assumes a discrete-time Markov decision process (MDP), defined by (S, A, P, r, ρ 0 , γ), in which S ⊆ R n is an n-dimensional state space, A ⊆ R m an m-dimensional action space, P : S × A × S → R + a transition probability function, r : S × A → R a bounded reward function, ρ 0 : S → R + an initial state distribution, and γ ∈ (0, 1] a discount factor.

The presented models are based on the optimization of a stochastic policy π θ : S × A → R + parameterized by θ.

Let η(π θ ) denote its expected return: DISPLAYFORM0 , where τ = (s 0 , a 0 , . . .) denotes the whole trajectory, s 0 ∼ ρ 0 (s 0 ), a t ∼ π θ (a t |s t ), and s t+1 ∼ P(s t+1 |s t , a t ) for all t. Our goal is to find the optimal policy arg max θ η(π θ ).

We will useQ(s t , a t ) to describe samples of cumulative discounted return, and Q(a t , s t ) to describe a function approximation ofQ(s t , a t ).

We will use "Q-function" when describing an abstract action-value function.

For a partially observable Markov decision process (POMDP), two more components are required, namely Ω, a set of observations, and O : S × Ω → R ≥0 , the observation probability distribution.

In the fully observable case, Ω ≡ S. Though the analysis in this article is written for policies over states, the same analysis can be done for policies over observations.

An important technique used in the derivation of the policy gradient is known as the score function (SF) estimator BID28 , which also comes up in the justification of baselines.

Suppose that we want to estimate ∇ θ E x [f (x)] where x ∼ p θ (x), and the family of distributions {p θ (x) : θ ∈ Θ} has common support.

Further suppose that log p θ (x) is continuous in θ.

In this case we have DISPLAYFORM0

The Policy Gradient Theorem BID22 states that DISPLAYFORM0 For convenience, define ρ π (s) = ∞ t=0 γ t p(s t = s) as the state visitation frequency, and DISPLAYFORM1 We can rewrite the above equation (with abuse of notation) as DISPLAYFORM2 It is further shown that we can reduce the variance of this gradient estimator without introducing bias by subtracting off a quantity dependent on s t fromQ(s t , a t ) BID28 BID3 .

See Appendix A for a derivation of the optimal state-dependent baseline.

DISPLAYFORM3 This is valid because, applying the SF estimator in the opposite direction, we have DISPLAYFORM4

In practice there can be rich internal structure in the policy parameterization.

For example, for continuous control tasks, a very common parameterization is to make π θ (a t |s t ) a multivariate Gaussian with diagonal variance, in which case each dimension a i t of the action a t is conditionally independent of other dimensions, given the current state s t .

Another example is when the policy outputs a tuple of discrete actions with factorized categorical distributions.

In the following subsections, we show that such structure can be exploited to further reduce the variance of the gradient estimator without introducing bias by changing the form of the baseline.

Then, we derive the optimal action-dependent baseline for a class of problems and analyze the suboptimality of non-optimal baselines in terms of variance reduction.

We then propose several practical baselines for implementation purposes.

We conclude the section with the overall policy gradient algorithm with action-dependent baselines for factorized policies.

We provide an exposition for situations when the conditional independence assumption does not hold, such as for stochastic policies with general covariance structures, in Appendix E, and for compatibility with other variance reduction techniques in Appendix F.

In the following, we analyze action-dependent baselines for policies with conditionally independent factors.

For example, multivariate Gaussian policies with a diagonal covariance structure are commonly used in continuous control tasks.

Assuming an m-dimensional action space, we have DISPLAYFORM0 In this case, we can set b i , the baseline for the ith factor, to depend on all other actions in addition to the state.

Let a

t denote all dimensions other than i in a t and denote the ith baseline by b i (s t , a −i t ).

Due to conditional independence and the score function estimator, we have DISPLAYFORM0 Hence we can use the following gradient estimator DISPLAYFORM1 This is compatible with advantage function form of the policy gradient : DISPLAYFORM2 DISPLAYFORM3 Note that the policy gradient now comprises of m component policy gradient terms, each with a different advantage term.

In Appendix E, we show that the methodology also applies to general policy structures (for example, a Gaussian policy with a general covariance structure), where the conditional independence assumption does not hold.

The result is bias-free albeit different baselines.

In this section, we derive the optimal action-dependent baseline and show that it is better than the state-only baseline.

We seek the optimal baseline to minimize the variance of the policy gradient estimate.

First, we write out the variance of the policy gradient under any action-dependent baseline.

Let us define z i := ∇ θ log π θ (a which translates to meaning that different subsets of parameters strongly influence different action dimensions or factors.

We note that this assumption is primarily for the theoretical analysis to be clean, and is not required to run the algorithm in practice.

In particular, even without this assumption, the proposed baseline is bias-free.

When the assumption holds, the optimal actiondependent baseline can be analyzed thoroughly.

Some examples where these assumptions do hold include multi-agent settings where the policies are conditionally independent by construction, cases where the policy acts based on independent components BID0 of the observation space, and cases where different function approximators are used to control different actions or synergies BID23 BID24 without weight sharing.

The optimal action-dependent baseline is then derived to be: DISPLAYFORM0 See Appendix B for the full derivation.

Since the optimal action-dependent baseline is different for different action coordinates, it is outside the family of state-dependent baselines barring pathological cases.

How much do we reduce variance over a traditional baseline that only depends on state?

We use the following notation: DISPLAYFORM0 DISPLAYFORM1 Then, using Equation (51) (Appendix C), we show the following improvement with the optimal action-dependent baseline: DISPLAYFORM2 See Appendices C and D for the full derivation.

We conclude that the optimal action-dependent baseline does not degenerate into the optimal state-dependent baseline.

Equation FORMULA1 states that the variance difference is a weighted sum of the deviation of the per-component score-weighted marginalized Q (denoted Y i ) from the component weight (based on score only, not Q) of the overall aggregated marginalized Q values (denoted j Y j ).

This suggests that the difference is particularly large when the Q function is highly sensitive to the actions, especially along those directions that influence the gradient the most.

Our empirical results in Section 5 additionally demonstrate the benefit of action-dependent over state-only baselines.

Using the previous theory, we now consider various baselines that could be used in practice and their associated computational cost.

Marginalized Q baseline Even though the optimal state-only baseline is known, it is rarely used in practice BID1 .

Rather, for both computational and conceptual benefit, the choice of b( DISPLAYFORM0 which is the action-dependent analogue.

In particular, when log probability of each policy factor is loosely correlated with the action-value function, then the proposed baseline is close to the optimal baseline.

DISPLAYFORM1 This has the added benefit of requiring learning only one function approximator, for estimating Q(s t , a t ), and implicitly using it to obtain the baselines for each action coordinate.

That is, Q(s t , a t ) is a function approximating samplesQ(s t , a t )

.Monte Carlo marginalized Q baseline After fitting Q π θ (s t , a t )

we can obtain the baselines through Monte Carlo estimates: DISPLAYFORM2 where α j ∼ π θ (a i t |s t ) are samples of the action coordinate i. In general, any function may be used to aggregate the samples, so long as it does not depend on the sample value a i t .

For instance, for discrete action dimensions, the sample max can be computed instead of the mean.

Mean marginalized Q baseline Though we reduced the computational burden from learning m functions to one function, the use of Monte Carlo samples can still be computationally expensive.

In particular, when using deep neural networks to approximate the Q-function, forward propagation through the network can be even more computationally expensive than stepping through a fast simulator (e.g. MuJoCo).

In such settings, we further propose the following more computationally practical baseline: DISPLAYFORM3 DISPLAYFORM4 t is the average action for coordinate i.

The final practical algorithm for fully factorized policies is as follows.

Require: number of iterations N , batch size B, initial policy parameters θ Initialize action-value function estimate Q π θ (s t , a t ) ≡ 0 and policy DISPLAYFORM0 , ∀t Perform a policy update step on θ usingÂ i (s t , a t ) [Equation FORMULA10 ]

Update action-value function approximation with current batch: Q π θ (s t , a t ) end forComputing the baseline can be done with either proposed technique in Section 4.4.

A similar algorithm can be written for general policies (Appendix E), which makes no assumptions on the conditional independence across action dimensions.

Continuous control benchmarks Firstly, we present the results of the proposed action-dependent baselines on popular benchmark tasks.

These tasks have been widely studied in the deep reinforcement learning community BID1 BID4 BID17 .

The studied tasks include the hopper, half-cheetah, and ant locomotion tasks simulated in MuJoCo BID25 .1 In addition to these tasks, we also consider a door opening task with a high-dimensional multi-fingered hand, introduced in BID16 to study the effectiveness of the proposed approach in high-dimensional tasks.

FIG0 presents the learning curves on these tasks.

We compare the action-dependent baseline with a baseline that uses only information about the states, which is the most common approach in the literature.

We observe that the action-dependent baselines perform consistently better.

A popular baseline parameterization choice is a linear function on a small number of non-linear features of the state BID1 , especially for policy gradient methods.

In this work, to enable a fair comparison, we use a Random Fourier Feature representation for the baseline BID15 BID17 .

The features are constructed as: y(x) = sin( 1 ν P x + φ) where P is a matrix with each element independently drawn from the standard normal distribution, φ is a random phase shift in [−π, π) and, and ν is a bandwidth parameter.

These features approximate the RKHS features under an RBF kernel.

Using these features, the baseline is parameterized as b = w T y(x) where x are the appropriate inputs to the baseline, and w are trainable parameters.

P and φ are not trained in this parameterization.

Such a representation was chosen for two reasons: (a) we wish to have the same number of trainable parameters for all the baseline architectures, and not have more parameters in the action-dependent case (which has a larger number of inputs to the baseline); (b) since the final representation is linear, it is possible to accurately estimate the optimal parameters with a Newton step, thereby alleviating the results from confounding optimization issues.

For policy optimization, we use a variant of the natural policy gradient method as described in BID17 .

See Appendix G for further experimental details.

Choice of action-dependent baseline form Next, we study the influence of computing the baseline by using empirical averages sampled from the Q-function versus using the mean-action of the action-coordinate for computing the baseline (both described in 4.4).

In our experiments, as shown in Figure 2 we find that the two variants perform comparably, with the latter performing slightly better towards the end of the learning process.

This suggests that though sampling from the Q-function might provide a better estimate of the conditional expectation in theory, function approximation from finite samples injects errors that may degrade the quality of estimates.

In particular, sub-sampling from the Q-function is likely to produce better results if the learned Q-function is accurate for a large fraction of the action space, but getting such high quality approximations might be hard in practice.

High-dimensional action spaces Intuitively, the benefit of the action-dependent baseline can be greater for higher dimensional problems.

We show this effect on a simple synthetic example called m-DimTargetMatching.

The example is a one-step MDP comprising of a single state, S = {0}, an m-dimensional action space, A = R m , and a fixed vector c ∈ R m .

The reward is given as the negative squared 2 loss of the action vector, r(s, a) = − a − c 2 2 .

The optimal action is thus to match Figure 2 : Variants of the action-dependent baseline that use: (i) sampling from the Q-function to estimate the conditional expectation; (ii) Using the mean action to form a linear approximation to the conditional expectation.

We find that both variants perform comparably, with the latter being more computationally efficient.

the given vector by selecting a = c. The results for the demonstrative example are shown in Table 1 , which shows that the action-dependent baseline successfully improves convergence more for higher dimensional problems than lower dimensional problems.

Due to the lack of state information, the linear baseline reduces to whitening the returns.

The action-dependent baseline, on the other hand, allows the learning algorithm to assess the advantage of each individual action dimension by utilizing information from all other action dimensions.

Additionally, this experiment demonstrates that our algorithm scales well computationally to high-dimensional problems.

Table 1 : Shown are the results for the synthetic high-dimensional target matching task (5 seeds), for 12 to 2000 dimensional action spaces.

At high dimensions, the linear feature action-dependent baseline provides notable and consistent variance reduction, as compared to a linear feature baseline, resulting in around 10% faster convergence.

For the corresponding learning curves, see Appendix G.Partially observable and multi-agent tasks Finally, we also consider the extension of the core idea of using global information, by studying a POMDP task and a multi-agent task.

We use the blind peg-insertion task which is widely studied in the robot learning literature BID12 .

The task requires the robot to insert the peg into the hole (slot), but the robot is blind to the location of the hole.

Thus, we expect a searching behavior to emerge from the robot, where it learns that the hole is present on the table and performs appropriate sweeping motions till it is able to find the hole.

In this case, we consider a baseline that is given access to the location of the hole.

We observe that a baseline with this additional information enables faster learning.

For the multi-agent setting, we analyze a two-agent particle environment task in which the goal is for each agent to reach their goal, where their goal is known by the other agent and they have a continuous communication channel.

Similar training procedures have been employed in recent related works BID9 ; .

FIG2 shows that including the inclusion of information from other agents into the action-dependent baseline improves the training performance, indicating that variance reduction may be key for multi-agent reinforcement learning.

An action-dependent baseline enables using additional signals beyond the state to achieve bias-free variance reduction.

In this work, we consider both conditionally independent policies and general policies, and derive an optimal action-dependent baseline.

We provide analysis of the variance DISPLAYFORM0 (a) Success percentage on the blind peg insertion task.

The policy still acts on the observations and does not know the hole location.

However, the baseline has access to this goal information, in addition to the observations and action, and helps to speed up the learning.

By comparison, in blue, the baseline has access only to the observations and actions.

reduction improvement over non-optimal baselines, including the traditional optimal baseline that only depends on state.

We additionally propose several practical action-dependent baselines which perform well on a variety of continuous control tasks and synthetic high-dimensional action problems.

The use of additional signals beyond the local state generalizes to other problem settings, for instance in POMDP and multi-agent tasks.

In future work, we propose to investigate related methods in such settings on large-scale problems.

We provide a derivation of the optimal state-dependent baseline, which minimizes the variance of the policy gradient estimate, and is based in (Greensmith et al., 2004, Theorem 8) .

More precisely, we minimize the trace of the covariance of the policy gradient; that is, the sum of the variance of the components of the vectors.

Recall the policy gradient expression with a state-dependent baseline: DISPLAYFORM0 Denote g to be the associated random variable, that is, ∇ θ η(π θ ) = E ρπ,π [g]: DISPLAYFORM1 The variance of the policy gradient is: DISPLAYFORM2 Note that E [η(π θ )]) contains a bias-free term, by the score function argument, which then does not affect the minimizer.

Terms which do not depend on b(s t ) also do not affect the minimizer.

DISPLAYFORM3

We derive the optimal action-dependent baseline, which minimizes the variance of the policy gradient estimate.

First, we write out the variance of the policy gradient under any action-dependent baseline.

Recall the following notations: we define z i := ∇ θ log π θ (a i t |s t ) and the component policy gradient: DISPLAYFORM0 Denote g i to be the associated random variables: DISPLAYFORM1 such that DISPLAYFORM2 Recall the following assumption: DISPLAYFORM3 which translates to meaning that different subsets of parameters strongly influence different action dimensions or factors.

This is true in case of distributed systems by construction, and also true in a single agent system if different action coordinates are strongly influenced by different policy network channels.

Under this assumption, we have: DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 where we denote the mean correction term DISPLAYFORM7 , and thus does not affect the optimal value.

The overall variance is minimized when each component variance is minimized.

We now derive the optimal baselines b * i (s t , a −i t ) which minimize each respective component.

DISPLAYFORM8 + E ρπ,a DISPLAYFORM9 Having written down the expression for variance under any action-dependent baseline, we seek the optimal baseline that would minimize this variance.

DISPLAYFORM10 The optimal action-dependent baseline is: DISPLAYFORM11

We now turn to quantifying the reduction in variance of the policy gradient estimate under the optimal baseline derived above.

Let Var * ( i g i ) denote the variance resulting from the optimal action-dependent baseline, and let Var( i g i ) denote the variance resulting from another baseline DISPLAYFORM0 , which may be suboptimal or action-independent.

Recall the notation: DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 Finally, define the variance improvement DISPLAYFORM4 Using these definitions, the variance can be re-written as: DISPLAYFORM5 Furthermore, the variance of the gradient with the optimal baseline can be written as DISPLAYFORM6 The difference in variance can be calculated as: DISPLAYFORM7 DISPLAYFORM8

Using the notation from Appendix C and working off of Equation FORMULA1 , we have: DISPLAYFORM0 DISPLAYFORM1 E BASELINES FOR GENERAL ACTIONSIn the preceding derivations, we have assumed policy actions are conditionally independent across dimensions.

In the more general case, we only assume that there are m factors a which altogether forms the action a t .

Conditioned on s t , the different factors form a certain directed acyclic graphical model (including the fully dependent case).

Without loss of generality, we assume that the following factorization holds: DISPLAYFORM2 where f (i) denotes the indices of the parents of the ith factor.

Let D(i) denote the indices of descendants of i in the graphical model (including i itself).

In this case, we can set the ith baseline to be b i (s t , a DISPLAYFORM3 ), where [m] = {1, 2, . . .

, m}. In other words, the ith baseline can depend on all other factors which the ith factor does not influence.

The overall gradient estimator is given by DISPLAYFORM4 In the most general case without any conditional independence assumptions, we have f (i) = {1, 2, . . .

, i − 1}, and D(i) = {i, i + 1, . . .

, m}. The above equation reduces to DISPLAYFORM5 The above analysis for optimal baselines and variance suboptimality transfers also to the case of general actions.

The applicability of our techniques to general action spaces may be of crucial importance for many application domains where the conditional independence assumption does not hold up, such as language tasks and other compositional domains.

Even in continuous control tasks, such as hand manipulation, and many other tasks where it is common practice to use conditionally independent factorized policies, it is reasonable to expect training improvement from policies without a full conditionally independence structure.

Computing action-dependent baselines for general actions The marginalization presented in Section 4.4 does not apply for the general action setting.

Instead, m individual baselines can be trained according to the factorization, and each of them can be fitted from data collected from the previous iteration.

In the general case, this means fitting m functions b i (s t , a 1 t , . . .

, a i−1 t ), for i ∈ {1, . . . , m}. The resulting method is described in Algorithm 2.There may also exist special cases like conditional independent actions, for which more efficient baseline constructions exist.

A closely related example to the conditionally independent case is the case of block diagonal covariance structures (e.g. in multi-agent settings), where we may wish to instead learn an overall Q function and marginalize over block factors.

Another interesting example to explore is sparse covariance structures.

Algorithm 2 Policy gradient for general factorization policies using action-dependent baselines Require: number of iterations N , batch size B, initial policy parameters θ Initialize baselines allow us to smoothly interpolate between high-bias, low-variance estimates and low-bias, high-variance estimates of the policy gradient.

These methods are based on the idea of being able to predict future returns, thereby bootstrapping the learning procedure.

In particular, when using the value function as a baseline, we have DISPLAYFORM6 DISPLAYFORM7 is an unbiased estimator for V (s).

GAE uses an exponential averaging of such temporal difference terms over a trajectory to significantly reduce the variance of the advantage at the cost of a small bias (it allows us to pick where we want to be on the bias-variance curve DISPLAYFORM8 Thus, the temporal difference error with the action dependent baselines is an unbiased estimator for the advantage function as well.

This allows us to use the GAE procedure to further reduce variance at the cost of some bias.

The following study shows that action-dependent baselines are consistent with TD procedures with their temporal differences being estimates of the advantage function.

Our results summarized in FIG3 suggests that slightly biasing the gradient to reduce variance produces the best results, while high-bias estimates perform poorly.

Prior work with baselines that utilize global information BID2 employ the high-bias variant.

The results here suggest that there is potential to further improve upon those results by carefully studying the bias-variance trade-off.

G HIGH-DIMENSIONAL ACTION SPACES: TRAINING CURVES Figure 5 shows the resulting training curves for a synthetic high-dimensional target matching task, as described in Section 5.

For higher dimensional action spaces (100 dimensions or greater), the action-dependent baseline consistently converges to the optimal solution 10% faster than the stateonly baseline.

For reference, Figure 6 shows the result of the original high-dimensional action space experiment.

Due to a discovered issue in the TensorFlow version of rllab, which results in training instability, both methods (action-dependent and state-dependent baselines) under-performed relative to the revised experiment ( Figure 5 ), which uses a clean implementation based on the implementation referenced inRajeswaran et al. (2017b) .

The regression in training is most evident by the number of iterations required to solve the task; for instance, the old experiment could take as long as five times more iterations to solve the same task, even for a 12-dimensional task.

Parameters: Unless otherwise stated, the following parameters are used in the experiments in this work: γ = 0.995, λ GAE = 0.97, kl desired = 0.025.

Policies:

The policies used are 2-layer fully connected networks with hidden sizes=(32, 32).Initialization: the policy is initialized with Xavier initialization except final layer weights are scaled down (by a factor of 100x).

Note that since the baseline is linear (with RBF features) and estimated with a Newton step, the initialization is inconsequential.

Per-experiment configuration: The following parameters in TAB5 are for both state-only and action-dependent versions of the experiments.

The m-DimTargetMatching experiments use a linear feature baseline.

@highlight

Action-dependent baselines can be bias-free and yield greater variance reduction than state-only dependent baselines for policy gradient methods.