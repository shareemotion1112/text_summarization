Real-world tasks are often highly structured.

Hierarchical reinforcement learning (HRL) has attracted research interest as an approach for leveraging the hierarchical structure of a given task in reinforcement learning (RL).

However, identifying the hierarchical policy structure that enhances the performance of RL is not a trivial task.

In this paper, we propose an HRL method that learns a latent variable of a hierarchical policy using mutual information maximization.

Our approach can be interpreted as a way to learn a discrete and latent representation of the state-action space.

To learn option policies that correspond to modes of the advantage function, we introduce advantage-weighted importance sampling.

In our HRL method, the gating policy learns to select option policies based on an option-value function, and these option policies are optimized based on the deterministic policy gradient method.

This framework is derived by leveraging the analogy between a monolithic policy in standard RL and a hierarchical policy in HRL by using a deterministic option policy.

Experimental results indicate that our HRL approach can learn a diversity of options and that it can enhance the performance of RL in continuous control tasks.

Reinforcement learning (RL) has been successfully applied to a variety of tasks, including board games BID13 , robotic manipulation tasks (Levine et al., 2016) , and video games (Mnih et al., 2015) .

Hierarchical reinforcement learning (HRL) is a type of RL that leverages the hierarchical structure of a given task by learning a hierarchical policy BID15 Dietterich, 2000) .

Past studies in this field have shown that HRL can solve challenging tasks in the video game domain BID18 BID0 and robotic manipulation BID3 BID9 .

In HRL, lower-level policies, which are often referred to as option policies, learn different behavior/control patterns, and the upper-level policy, which is often referred to as the gating policy, learns to select option policies.

Recent studies have developed HRL methods using deep learning (Goodfellow et al., 2016) and have shown that HRL can yield impressive performance for complex tasks BID0 Frans et al., 2018; BID18 Haarnoja et al., 2018a) .

However, identifying the hierarchical policy structure that yields efficient learning is not a trivial task, since the problem involves learning a sufficient variety of types of behavior to solve a given task.

In this study, we present an HRL method via the mutual information (MI) maximization with advantage-weighted importance, which we refer to as adInfoHRL.

We formulate the problem of learning a latent variable in a hierarchical policy as one of learning discrete and interpretable repre-sentations of states and actions.

Ideally, each option policy should be located at separate modes of the advantage function.

To estimate the latent variable that corresponds to modes of the advantage function, we introduce advantage-weighted importance weights.

Our approach can be considered to divide the state-action space based on an information maximization criterion, and it learns option policies corresponding to each region of the state-action space.

We derive adInfoHRL as an HRL method based on deterministic option policies that are trained based on an extension of the deterministic policy gradient BID12 Fujimoto et al., 2018) .

The contributions of this paper are twofold:1.

We propose the learning of a latent variable of a hierarchical policy as a discrete and hidden representation of the state-action space.

To learn option policies that correspond to the modes of the advantage function, we introduce advantage-weighted importance.

2.

We propose an HRL method, where the option policies are optimized based on the deterministic policy gradient and the gating policy selects the option that maximizes the expected return.

The experimental results show that our proposed method adInfoHRL can learn a diversity of options on continuous control tasks.

Moreover, our approach can improve the performance of TD3 on such tasks as the Walker2d and Ant tasks in OpenAI Gym with MuJoco simulator.

In this section, we formulate the problem of HRL in this paper and describe methods related to our proposal.

We consider tasks that can be modeled as a Markov decision process (MDP), consisting of a state space S, an action space A, a reward function r : S × A → R, an initial state distribution ρ(s 0 ), and a transition probability p(s t+1 |s t , a t ) that defines the probability of transitioning from state s t and action a t at time t to next state s t+1 .

The return is defined as R t = T i=t γ i−t r(s i , a i ), where γ is a discount factor, and policy π(a|s) is defined as the density of action a given state s. Let d π (s) = T t=0 γ t p(s t = s) denote the discounted visitation frequency induced by the policy π.

The goal of reinforcement learning is to learn a policy that maximizes the expected return J(π) = E s0,a0,... [R 0 ] where s 0 ∼ ρ(s 0 ), a ∼ π and s t+1 ∼ p(s t+1 |s t , a t ).

By defining the Q-function as Q π (s, a) = E s0,a0,... [R t |s t = s, a t = a], the objective function of reinforcement learning can be rewritten as follows: DISPLAYFORM0 Herein, we consider hierarchical policy π(a|s) = o∈O π(o|s)π(a|s, o), where o is the latent variable and O is the set of possible values of o. Many existing HRL methods employ a policy structure of this form (Frans et al., 2018; BID18 BID0 Florensa et al., 2017; BID3 .

In general, latent variable o can be discrete (Frans et al., 2018; BID0 Florensa et al., 2017; BID3 BID7 or continuous BID18 .

π(o|s) is often referred to as a gating policy BID3 BID7 , policy over options BID0 , or manager BID18 .

Likewise, π(a|s, o) is often referred to as an option policy BID7 , sub-policy BID3 , or worker BID18 .

In HRL, the objective function is given by DISPLAYFORM1 As discussed in the literature on inverse RL BID19 , multiple policies can yield equivalent expected returns.

This indicates that there exist multiple solutions to latent variable o that maximizes the expected return.

To obtain the preferable solution for o, we need to impose additional constraints in HRL.

Although prior work has employed regularizers BID0 and constraints BID3 to obtain various option policies, the method of learning a good latent variable o that improves sample-efficiency of the learning process remains unclear.

In this study we propose the learning of the latent variable by maximizing MI between latent variables and state-action pairs.

The deterministic policy gradient (DPG) algorithm was developed for learning a monolithic deterministic policy µ θ (s) : S → A by BID12 .

In off-policy RL, the objective is to maximize the expectation of the return, averaged over the state distribution induced by a behavior policy β(a|s): DISPLAYFORM0 When a policy is deterministic, the objective becomes BID12 have shown that the gradient of a deterministic policy is given by DISPLAYFORM1 DISPLAYFORM2 The DPG algorithm has been extended to the deep deterministic policy gradient (DDPG) for continuous control problems that require neural network policies BID13 .

Twin Delayed Deep Deterministic policy gradient algorithm (TD3) proposed by Fujimoto et al. (2018) is a variant of DDPG that outperforms the state-of-the-art on-policy methods such as TRPO BID10 and PPO BID11 in certain domains.

We extend this deterministic policy gradient to learn a hierarchical policy.

Recent studies such as those by BID2 Hu et al. (2017); Li et al. (2017) have shown that an interpretable representation can be learned by maximizing MI.

Given a dataset X = (x 1 , ..., x n ), regularized information maximization (RIM) proposed by Gomes et al. (2010) involves learning a conditional modelp(y|x; η) with parameter vector η that predicts a label y. The objective of RIM is to minimize DISPLAYFORM0 where (η) is the regularization term, I η (x, y) is MI, and λ is a coefficient.

MI can be decomposed as I η (x, y) = H(y) − H(y|x) where H(y) is entropy and H(y|x) the conditional entropy.

Increasing H(y) conduces the label to be uniformly distributed, and decreasing H(y|x) conduces to clear cluster assignments.

Although RIM was originally developed for unsupervised clustering problems, the concept is applicable to various problems that require learning a hidden discrete representation.

In this study, we formulate the problem of learning the latent variable o of a hierarchical policy as one of learning a latent representation of the state-action space.

In this section, we propose a novel HRL method based on advantage-weighted information maximization.

We first introduce the latent representation learning via advantage-weighted information maximization, and we then describe the HRL framework based on deterministic option policies.

Although prior work has often considered H(o|s) or I(s, o), which results in a division of the state space, we are interested in using I (s, a), o for dividing the state-action space instead.

A schematic sketch of our approach is shown in FIG0 .

As shown in the left side of FIG0 , the advantage function often has multiple modes.

Ideally, each option policies should correspond to separate modes of the advantage function.

However, it is non-trivial to find the modes of the advantage function in practice.

For this purpose, we reduce the problem of finding modes of the advantage function to that of finding the modes of the probability density of state action pairs.

We consider a policy based on the advantage function of the form where DISPLAYFORM0 DISPLAYFORM1 is the state value function, and Z is the partition function.

f (·) is a functional, which is a function of a function.

f (·) is a monotonically increasing function with respect to the input variable and always satisfies f (·) > 0.

In our implementation we used the exponential function f (·) = exp(·).

When following such a policy, an action with the larger advantage is drawn with a higher probability.

Under this assumption, finding the modes of the advantage function is equivalent to finding modes of the density induced by π Ad .

Thus, finding the modes of the advantage function can be reduced to the problem of clustering samples induced by π Ad .Following the formulation of RIM introduced in Section 2.3, we formulate the problem of clustering samples induced by π Ad as the learning of discrete representations via MI maximization.

For this purpose, we consider a neural network that estimates p(o|s, a; η) parameterized with vector η, which we refer to as the option network.

We formulate the learning of the latent variable o as minimizing DISPLAYFORM2 where I(o, (s, a)) =Ĥ(o|s, a; η) −Ĥ(o; η), and (η) is the regularization term.

In practice, we need to approximate the advantage function, and we learn the discrete variable o that corresponds to the modes of the current estimate of the advantage function.

For regularization, we used a simplified version of virtual adversarial training (VAT) proposed by Miyato et al. (2016) .

Namely, we set (η) = D KL p(o|s noise , a noise ; η)||p(o|s, a; η) where s noise = s + s , a noise = a + a , s and a denote white noise.

This regularization term penalizes dissimilarity between an original state-action pair and a perturbed one, and Hu et al. (2017) empirically show that this regularization improves the performance of learning latent discrete representations.

When computing MI, we need to compute p(o) and H(o|s, a) given by DISPLAYFORM3 DISPLAYFORM4 Thus, the probability density of (s, a) induced by π Ad is necessary for computing MI for our purpose.

To estimate the probability density of (s, a) induced by π Ad , we introduce the advantageweighted importance in the next section.

Although we show that the problem of finding the modes of the advantage function can be reduced to MI maximization with respect to the samples induced by π Ad , samples induced by π Ad are not available in practice.

While those induced during the learning process are available, a discrete representation obtained from such samples does not correspond to the modes of the advantage function.

To estimate the density induced by π Ad , we employ an importance sampling approach.

We assume that the change of the state distribution induced by the policy update is sufficiently small, DISPLAYFORM0 Then, the importance weight can be approximated as DISPLAYFORM1 and the normalized importance weight is given gỹ DISPLAYFORM2 As the partition function Z is canceled, we do not need to compute Z when computing the importance weight in practice.

We call this importance weight W the advantage-weighted importance and employ it to compute the objective function used to estimate the latent variable.

This advantage-weighted importance is used to compute the entropy terms for computing MI in Equation FORMULA8 .

The empirical estimate of the entropy H(o) is given bŷ DISPLAYFORM3 where the samples (s i , a i ) are drawn from p β (s, a) induced by a behavior policy β(a|s).

Likewise, the empirical estimate of the conditional entropy H(o|s, a) is given bŷ DISPLAYFORM4 The derivations of Equations FORMULA0 and FORMULA0 are provided in Appendix A. To train the option network, we store the samples collected by the M most recent behavior policies, to which we refer as onpolicy buffer D on .

Although the algorithm works with entire samples stored in the replay buffer, we observe that the use of the on-policy buffer for latent representation learning exhibits better performance.

For this reason, we decided to use the on-policy buffer in our implementation.

Therefore, while the algorithm is off-policy in the sense that the option is learned from samples collected by behavior policies, our implementation is "semi"on-policy in the sense that we use samples collected by the most recent behavior policies.

Instead of stochastic option policies, we consider deterministic option policies and model them using separate neural networks.

We denote by π(a|s, o) = µ o θ (s) deterministic option policies parameterized by vector θ.

The objective function of off-policy HRL with deterministic option policies can then be obtained by replacing π(a|s) with o∈O π(o|s)π(a|s, o) in Equation FORMULA2 : DISPLAYFORM0 where Q π (s, a; w) is an approximated Q-function parameterized using vector w. This form of the objective function is analogous to Equation (3).

Thus, we can extend standard RL techniques to the learning of the gating policy π(o|s) in HRL with deterministic option policies.

In HRL, the goal of the gating policy is to generate a value of o that maximizes the conditional expectation of the return: DISPLAYFORM1 which is often referred to as the option-value function BID15 .

When option policies are stochastic, it is often necessary to approximate the option-value function Q π Ω (s, o) in addition to the action-value function Q π (s, a).

However, in our case, the option-value function for deterministic option policies is given by DISPLAYFORM2 DISPLAYFORM3 end if end for until the convergence return θ which we can estimate using the deterministic option policy µ o θ (s) and the approximated actionvalue function Q π (s, a; w).

In this work we employ the softmax gating policy of the form DISPLAYFORM4 which encodes the exploration in its form BID3 .

The state value function is given as DISPLAYFORM5 which can be computed using Equation (17).

We use this state-value function when computing the advantage-weighted importance as A(s, a) = Q(s, a) − V (s).

In this study, the Q-function is trained in a manner proposed by Fujimoto et al. (2018) .

Two neural networks (Q π w1 , Q π w2 ) are trained to estimate the Q-function, and the target value of the Q-function is computed as y i = r i + γ min 1,2 Q(s i , a i ) for sample (s i , a i , a i , r i ) in a batch sampled from a replay buffer, where r i = r(s i , a i ).

In this study, the gating policy determines the option once every N time steps, i.e., t = 0, N, 2N, . . .

Neural networks that model µ o θ (a|s) for o = 1, ..., O, which we refer to as option-policy networks, are trained separately for each option.

In the learning phase, p(o|s, a) is estimated by the option network.

Then, samples are assigned to option o * = arg max o p(o|s, a; η) and are used to update the option-policy network that corresponds to o * .

When performing a rollout, o is drawn by following the gating policy in Equation FORMULA0 , and an action is generated by the selected option-policy network.

Differentiating the objective function in Equation FORMULA0 , we obtain the deterministic policy gradient of our option-policy µ o θ (s) given by DISPLAYFORM6 The procedure of adInfoHRL is summarized by Algorithm 1.

As in TD3 (Fujimoto et al., 2018) , we employed the soft update using a target value network and a target policy network.

We evaluated the proposed algorithm adInfoHRL on the OpenAI Gym platform BID1 with the MuJoCo Physics simulator BID16 .

We compared its performance with that of PPO implemented in OpenAI baselines (Dhariwal et al., 2017) and TD3.

Henderson et al. (2018) have recently claimed that algorithm performance varies across environment, there is thus no clearly best method for all benchmark environments, and off-policy and on-policy methods have advantages in different problem domains.

To analyze the performance of adInfoHRL, we compared it with state-of-the-art algorithms for both on-policy and off-policy methods, although we focused on the comparison with TD3, as our implementation of adInfoHRL is based on it.

To determine the effect of learning the latent variable via information maximization, we used the same network architectures for the actor and critic in adInfoHRL and TD3.

In addition, to evaluate the benefit of the advantage-weighted importance, we evaluated a variant of adInfoHRL, which does not use the advantage-weighted importance for computing mutual information.

We refer to this variant of adInfoHRL as infoHRL.

The gating policy updated variable o once every three time steps.

We tested the performance of adInfoHRL with two and four options.

The activation of options over time and snapshots of the learned option policies on the Walker2d task are shown in FIG1 , which visualizes the result from adInfoHRL with four options.

One can see that the option policies are activated in different phases of locomotion.

While the option indicated by yellow in FIG1 corresponds to the phase for kicking the floor, the option indicated by blue corresponds to the phase when the agent was on the fly.

Visualization of the options learned on the HalfCheetah and Ant tasks are shown in Appendix D.The averaged return of five trials is reported in FIG3 (a)-(d).

AdIfoHRL yields the best performance on Ant 1 and Walker2d, whereas the performance of TD3 and adInfoHRL was comparable on HalfCheetah and Hopper, and PPO outperformed the other methods on Hopper.

Henderson et al. (2018) claimed that on-policy methods show their superiority on tasks with unstable dynamics, and our experimental results are in line with such previous studies.

AdinfoHRL outperformed infoHRL, which isthe variant of adInfoHRL without the advantage-weighted importance on all the tasks.

This result shows that the adavatage-weighted importance enhanced the performance of learning options.

AdInfoHRL exhibited the sample efficiency on Ant and Walker2d in the sense that it required fewer samples than TD3 to achieve comparable performance on those tasks.

The concept underlying adInfoHRL is to divide the state-action space to deal with the multi-modal advantage function and learn option policies corresponding to separate modes of the advantage function.

Therefore, adInfoHRL shows its superiority on tasks with the multi-modal advantage function and not on tasks with a simple advantage function.

Thus, it is natural that the benefit of adInfoHRL is dependent on the characteristics of the task.

The outputs of the option network and the activation of options on Walker2d are shown in FIG3 , which visualize the result from adInfoHRL with four options.

For visualization, the dimensionality was reduced using t-SNE (van der BID17 FORMULA0 also considered the MI in their formulation.

In these methods, MI between the state and the latent variable is considered so as to obtain diverse behaviors.

Our approach is different from the previous studies in the sense that we employ MI between the latent variable and the state-action pairs, which leads to the division of the state-action space instead of considering only the state space.

We think that dividing the state-action space is an efficient approach when the advantage function is multi-modal, as depicted in FIG0 .

InfoGAIL proposed by Li et al. (2017) learns the interpretable representation of the state-action space via MI maximization.

InfoGAIL can be interpreted as a method that divides the state-action space based on the density induced by an expert's policy by maximizing the regularized MI objective.

In this sense, it is closely related to our method, although their problem setting is imitation learning BID8 , which is different from our HRL problem setting.

The use of the importance weight based on the value function has appeared in previous studies BID4 Kober & Peters, 2011; BID6 BID7 .

For example, the method proposed by BID6 employs the importance weight based on the advantage function for learning a monolithic policy, while our method uses a similar importance weight for learning a latent variable of a hierarchical policy.

Although BID7 proposed to learn a latent variable in HRL with importance sampling, their method is limited to episodic settings where only a single option is used in an episode.

Our method can be interpreted as an approach that divides the state-action space based on the MI criterion.

This concept is related to that of Divide and Conquer (DnC) proposed by Ghosh et al. (2018) , although DnC clusters the initial states and does not consider switching between option policies during the execution of a single trajectory.

In this study we developed adInfoHRL based on deterministic option policies.

However, the concept of dividing the state-action space via advantage-weighted importance can be applied to stochastic policy gradients as well.

Further investigation in this direction is necessary in future work.

We proposed a novel HRL method, hierarchical reinforcement learning via advantage-weighted information maximization.

In our framework, the latent variable of a hierarchical policy is learned as a discrete latent representation of the state-action space.

Our HRL framework is derived by considering deterministic option policies and by leveraging the analogy between the gating policy for HRL and a monolithic policy for the standard RL.

The results of the experiments indicate that adInfoHRL can learn diverse options on continuous control tasks.

Our results also suggested that our approach can improve the performance of TD3 in certain problem domains.

The mutual information (MI) between the latent variable o and the state action pair (s, a) is defined as DISPLAYFORM0 where H(o) = p(o) log p(o)do and H(o|s, a) = p(o|s, a) log p(o|s, a)do.

We make the empirical estimate of MI employed by Gomes et al. (2010); Hu et al. (2017) and modify it to employ the importance weight.

The empirical estimate of MI with respect to the density induced by a policy π is given byÎ DISPLAYFORM1 We consider the case where we have samples collected by a behavior policy β(s|a) and need to estimate MI with respect to the density induced by policy π.

Given a model p(o|s, a; η) parameterized by vector η, p(o) can be rewritten as DISPLAYFORM2 where DISPLAYFORM3 is the importance weight.

Therefore, the empirical estimate of p(o) with respect to the density induced by a policy π is given bŷ DISPLAYFORM4 DISPLAYFORM5 is the normalized importance weight.

Likewise, the conditional entropy with respect to the density induced by a policy π is given by H(o|s, a) = p π (s, a)p(o|s, a; η)

log p(o|s, a; η)dsda DISPLAYFORM6 Therefore, the empirical estimate of the conditional entropy with respect to the density induced by a policy π is given bŷ DISPLAYFORM7 Thus, the empirical estimates of MI can be computed by Equations FORMULA0 , FORMULA1 and (27).

In HRL, the value function is given by DISPLAYFORM0 Since option policies are deterministic given by µ o θ (s), the state-value function is given by DISPLAYFORM1

We performed evaluations using benchmark tasks in the OpenAI Gym platform BID1 with Mujoco physics simulator BID16 .

Hyperparameters of reinforcement learning methods used in the experiment are shown in TAB3 .

For exploration, both adInfoHRL and TD3 used the clipped noise drawn from the normal distribution as ∼ clip N (0, σ), −c, c , where σ = 0.2 and c = 0.5.

For hyperparameters of PPO, we used the default values in OpenAI baselines (Dhariwal et al., 2017) .

For the Walker2d, HalfCheetah, and Hopper tasks, we used the Walker2d-v1, HalfCHeetah-v1, and Hopper-v1 in the OpenAI Gym, respectively.

For the Ant task, we used the AntEnv implemented in the rllab BID2 .

When training a policy with AdInfoHRL, infoHRL, and TD3, critics are trained once per time step, and actors are trained once every after two updates of the critics.

The source code is available at https://github.com/ TakaOsa/adInfoHRL.We performed the experiments five times with different seeds, and reported the averaged test return where the test return was computed once every 5000 time steps by executing 10 episodes without exploration.

When performing the learned policy without exploration, the option was drawn as DISPLAYFORM0 instead of following the stochastic gating policy in Equations (17).

On the HalfCheetah task, adInfoHRL delivered the best performance with two options.

The distribution of options on HalfCheetah0v1 after one million steps is shown in FIG6 .

Although the state-action space is evenly divided, the options are not evenly activated.

This behavior can occur because the state-action space is divided based on the density induced by the behavior policy while the activation of options is determined based on the quality of the option policies in a given state.

Moreover, an even division in the action-state space is not necessarily the even division in the state space.

The activation of the options over time is shown in FIG7 .

It is clear that one of the option corresponds to the stable running phase and the other corresponds to the phase for recovering from unstable states.

Figure 6: Distribution of options on Ant-rllab task using adInfoHRL with four options.

The dimensionality is reduced by t-SNE for visualization.

The distribution of four options on the Ant-rllab task after one million steps is shown in Figure 6 .

Four options are activated in the different domains of the state space.

The activation of the options over time on the Ant-rllab task is shown in FIG8 .

While four options are actively used in the beginning of the episode, two (blue and yellow) options are mainly activated during the stable locomotion.

Since the Ant task implemented in rllab is known to be harder than the Ant-v1 implemented in the OpenAI gym, we reported the result of the Ant task in rllab in the main manuscript.

Here, we report the result of the Ant-v1 task implemented in the OpenAI gym.

On the Ant-v1 task, adInfoHRL yielded the best performance with two options.

The performance of adInfoHRL with two options is comparable to that of TD3 on Ant-v1.

This result indicates that the Ant-v1 task does not require a hierarchical policy structure, while a hierarchical policy improves the performance of learning on Ant-rllab.

The distribution of options on Ant-v1 task after one million steps is shown in FIG10 .

The activation of the options over time is shown in FIG11 .

It is evident that two option policies on the Ant-v1 task corresponded to different postures of the agent.

A recent study on HRL by BID14 reported the performance of IOPG on Walker2d-v1, Hopper-v1, and HalfCheetah-v1.

The study by Haarnoja et al. (2018a) reported the performance of SAC-LSP on Walker2d-v1, Hopper-v1, HalfCheetah-v1, and Ant-rllab.

A comparison of performance between our method, IOPG, and SAC-LSP is summarized in TAB6 .

We report the performance after 1 million steps.

It is worth noting that adInfoHRL outperformed IOPG on these tasks in terms of the achieved return, although we are aware that the qualitative performance is also important in HRL.

AdInfoHRL outperformed SAC-LSP on Walker2d-v1 and Ant-rllab, and SAC-LSP shows its superiority on HalfCheetah-v1 and Hopper-v1.

However, the results of SAC-LSP were obtained by using reward scaling, which was not used in the evaluation of adInfoHRL.

Therefore, further experiments are necessary for fair comparison under the same condition.

@highlight

This paper presents a hierarchical reinforcement learning framework based on deterministic option policies and mutual information maximization. 

@highlight

Proposes an HRL algorithm that attempts to learn options that maximize their mutual information with the state-action density under the optimal policy.

@highlight

This paper proposes an HRL system in which the mutal information of the latent variable and the state-action pairs is approximately maximized.

@highlight

Proposes a criterion that aims to maximize the mutual information between options and state-action pairs and show empirically that the learned options decompose the state-action space but not the state space. 