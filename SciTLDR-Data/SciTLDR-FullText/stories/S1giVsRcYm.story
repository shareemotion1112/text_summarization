The problem of exploration in reinforcement learning is well-understood in the tabular case and many sample-efficient algorithms are known.

Nevertheless, it is often unclear how the algorithms in the tabular setting can be extended to tasks with large state-spaces where generalization is required.

Recent promising developments generally depend on problem-specific density models or handcrafted features.

In this paper we introduce a simple approach for exploration that allows us to develop theoretically justified algorithms in the tabular case but that also give us intuitions for new algorithms applicable to settings where function approximation is required.

Our approach and its underlying theory is based on the substochastic successor representation, a concept we develop here.

While the traditional successor representation is a representation that defines state generalization by the similarity of successor states, the substochastic successor representation is also able to implicitly count the number of times each state (or feature) has been observed.

This extension connects two until now disjoint areas of research.

We show in traditional tabular domains (RiverSwim and SixArms) that our algorithm empirically performs as well as other sample-efficient algorithms.

We then describe a deep reinforcement learning algorithm inspired by these ideas and show that it matches the performance of recent pseudo-count-based methods in hard exploration Atari 2600 games.

Reinforcement learning (RL) tackles sequential decision making problems by formulating them as tasks where an agent must learn how to act optimally through trial and error interactions with the environment.

The goal in these problems is to maximize the sum of the numerical reward signal observed at each time step.

Because the actions taken by the agent influence not just the immediate reward but also the states and associated rewards in the future, sequential decision making problems require agents to deal with the trade-off between immediate and delayed rewards.

Here we focus on the problem of exploration in RL, which aims to reduce the number of samples (i.e., interactions) an agent needs in order to learn to perform well in these tasks when the environment is initially unknown.

The sample efficiency of RL algorithms is largely dependent on how agents select exploratory actions.

In order to learn the proper balance between immediate and delayed rewards agents need to navigate through the state space to learn about the outcome of different transitions.

The number of samples an agent requires is related to how quickly it is able to explore the state-space.

Surprisingly, the most common approach is to select exploratory actions uniformly at random, even in high-profile success stories of RL (e.g., BID26 BID17 .

Nevertheless, random exploration often fails in environments with sparse rewards, that is, environments where the agent observes a reward signal of value zero for the majority of states.

In model-based approaches agents explicitly learn a model of the dynamics of the environment which they use to plan future actions.

In this setting the problem of exploration is well understood.

When all states can be enumerated and uniquely identified (tabular case), we have algorithms with proven sample complexity bounds on the maximum number of suboptimal actions an agent selects before 1 When we refer to environments with sparse rewards we do so for brevity and ease of presentation.

Actually, any sequential decision making problem has dense rewards.

In the RL formulation a reward signal is observed at every time step.

By environments with sparse rewards we mean environments where the vast majority of transitions lead to reward signals with the same value.converging to an -optimal policy (e.g., BID4 BID10 BID23 .

However, these approaches are not easily extended to large environments where it is intractable to enumerate all of the states.

When using function approximation, the concept of state visitation is not helpful and learning useful models is by itself quite challenging.

Due to the difficulties in learning good models in large domains, model-free methods are much more popular.

Instead of building an explicit model of the environment, they estimate state values directly from transition samples (state, action, reward, next state).

Unfortunately, this approach makes systematic exploration much more challenging.

Nevertheless, because model-free methods make up the majority of approaches scalable to large domains, practitioners often ignore the exploration challenges these methods pose and accept the high sample complexity of random exploration.

Reward bonuses that promote exploration are one alternative to random walks (e.g., BID2 BID15 , but none such proposed solutions are widely adopted in the field.

In this paper we introduce an algorithm for exploration based on the successor representation (SR).

The SR, originally introduced by BID5 , is a representation that generalizes between states using the similarity between their successors, i.e., the similarity between the states that follow the current state given the environment's dynamics and the agent's policy.

The SR is defined for any problem, it can be learned through temporal-difference learning BID25 and, as we discuss below, it can also be seen as implicitly estimating the transition dynamics of the environment.

Our approach is inspired by the substochastic successor representation (SSR), a concept we introduce here.

The SSR is defined so that it implicitly counts state visitation, allowing us to use it to encourage exploration.

This idea connects representation learning and exploration, two otherwise disjoint areas of research.

The SSR allows us to derive an exploration bonus that when applied to model-based RL generates algorithms that perform as well as theoretically sample-efficient algorithms.

Importantly, the intuition developed with the SSR assists us in the design of a model-free deep RL algorithm that achieves performance similar to pseudo-count-based methods in hard exploration Atari 2600 games BID2 BID20 .

We consider an agent interacting with its environment in a sequential manner.

Starting from a state S 0 ??? S, at each step the agent takes an action A t ??? A, to which the environment responds with a state S t+1 ??? S according to a transition probability function p(s |s, a) = Pr(S t+1 = s |S t = s, A t = a), and with a reward signal R t+1 ??? R, where r(s, a) indicates the expected reward for a transition from state s under action a, that is, r(s, a).

DISPLAYFORM0 The value of a state s when following a policy ??, v ?? (s), is defined to be the expected sum of discounted rewards from that state: DISPLAYFORM1 , with ?? being the discount factor.

When the transition probability function p and the reward function r are known, we can compute v ?? (s) recursively by solving the system of equations below BID3 : DISPLAYFORM2 This equation can also be written in matrix form with v ?? , r ??? R |S| and P ?? ??? R |S|??|S| : DISPLAYFORM3 where P ?? is the state to state transition probability function induced by ??, that is, P ?? (s, s ) = a ??(a|s)p(s |s, a).

Traditional model-based algorithms for RL work by learning estimates of the matrix P ?? and of the vector r and using them to estimate v ?? , for example by solving Equation 1.

We useP ?? andr to denote empirical estimates of P ?? and r. Formally, DISPLAYFORM4 wherer(i) denotes the i-th entry in the vectorr, n(s, s ) is the number of times the transition s ??? s was observed, n(s) = s ???S n(s, s ), and C(s, s ) is the sum of the rewards associated with the n(s, s ) transitions (we drop the action in the discussion to simplify notation).Alternatively, in model-free RL, instead of estimating P ?? and r we estimate v ?? (s) directly from samples.

We often use temporal-difference (TD) learning BID25 to update our estimates of DISPLAYFORM5 where ?? is the step-size parameter.

Generalization is required in problems with large state spaces, where it is unfeasible to learn an individual value for each state.

We do so by parametrizingv(s) with a set of weights ??.

We write, given the weights ??,v(s; ??) ??? v ?? (s) andq(s, a; ??) ??? q ?? (s, a), where q ?? (s, a) = r(s, a) + ?? s p(s |s, a)v ?? (s ).

Model-free methods have performed well in problems with large state spaces, mainly due to the use of neural networks as function approximators (e.g., BID17 .Our algorithm is based on the successor representation (SR; BID5 .

The successor representation, with respect to a policy ??, ?? ?? , is defined as DISPLAYFORM6 where we assume the sum is convergent with I denoting the indicator function.

BID5 has shown that this expectation can be estimated from samples through TD learning.

It also corresponds to the Neumann series of ??P : DISPLAYFORM7 Notice that the SR is part of the solution when computing a value function: DISPLAYFORM8 We use?? ?? to denote the SR computed throughP ?? , the approximation of P ?? .The definition of the SR can also be extended to features.

Successor features generalize the SR to the function approximation setting BID0 .

We use the definition for the uncontrolled case in this paper.

Importantly, the successor features can also be learned with TD learning.

Definition 2.1 (Successor Features).

For a given 0 ??? ?? < 1, policy ??, and for a feature representation ??(s) ??? R d , the successor features for a state s are: DISPLAYFORM9 Notice that this definition reduces to the SR in the tabular case, where ?? = I.

In this section we introduce the concept of the substochastic successor representation (SSR).

The SSR is derived from an empirical transition matrix similar to Equation 2, but where each state incorporates a small (1/(n(s) + 1)) probability of terminating at that state, rather than transiting to a next state.

As we will show, we can recover the visit counts n(s) through algebraic manipulation on the SSR.While computing the SSR is usually impractical, we use it as inspiration in the design of a new deep reinforcement learning algorithm for exploration (Section 4).

In a nutshell, we view the SSR as approximating the process of learning the SR from an uninformative initialization (i.e., the zero vector), and using a stochastic update rule.

While this approximation is relatively coarse, we believe it gives qualitative justification to our use of the learned SR to guide exploration.

To further this claim, we demonstrate that using the SSR in synthetic, tabular settings yields comparable performance to that of theoretically-derived exploration algorithms.

Definition 3.1 (Substochastic Successor Representation).

LetP ?? denote the substochastic matrix induced by the environment's dynamics and by the policy ?? such thatP ?? (s |s) = n(s,s ) n(s)+1 .

For a given 0 ??? ?? < 1, the substochastic successor representation,?? ?? , is defined as: DISPLAYFORM0 The theorem below formalizes the idea that the 1 norm of the SSR implicitly counts state visitation.

Theorem 1.

Let n(s) denote the number of times state s has been visited and let ??(s) = (1 + ??) ??? ||?? ?? (s)|| 1 , where?? ?? is the substochastic SR as in Definition 3.1.

For a given 0 ??? ?? < 1, DISPLAYFORM1 Proof of Theorem 1.

LetP ?? be the empirical transition matrix.

We first rewriteP ?? in terms ofP ?? : DISPLAYFORM2 The expression above can also be written in matrix form:P ?? = (I ??? N )P ?? , where N ??? R |S|??|S| denotes the diagonal matrix of augmented inverse counts.

Expanding?? ?? we have: DISPLAYFORM3 The top eigenvector of a stochastic matrix is the all-ones vector, e (Meyn & Tweedie, 2012), and it corresponds to the eigenvalue 1.

Using this fact and the definition ofP ?? with respect toP ?? we have: DISPLAYFORM4 We can now bound the term ??

?? e using the fact that e is also the top eigenvector of the successor representation and has eigenvalue DISPLAYFORM0 Plugging FORMULA14 into the definition of ?? we have (notice that ??(s)e = ||??(s)|| 1 ): DISPLAYFORM1 When we also use the other bound on the quadratic term we conclude that, for any state s, DISPLAYFORM2 In other words, the SSR, obtained after a slight change to the SR, can be used to recover state visitation counts.

The intuition behind this result is that the phantom transition, represented by the +1 in the denominator of the SSR, serves as a proxy for the uncertainty about that state by underestimating the SR.

This is due to the fact that s P ?? (s, s ) gets closer to 1 each time state s is visited.

This result can now be used to convert the SSR into a reward function in the tabular case.

We do so by using the SSR to define an exploration bonus, r int , such that the reward being maximized by the agent becomes r(s, a) + ??r int (s), where ?? is a scaling parameter.

Since we want to incentivize agents to visit the least visited states as quickly as possible, we can trivially define r int = ???||?? ?? (s)|| 1 , where we penalize the agent by visiting the states that lead to commonly visited states.

Notice that the shift (1 + ??) in ??(s) has no effect as an exploration bonus because it is the same across all states.

BID23 .

The performance of our algorithm is the average over 100 runs.

A 95% confidence interval is reported between parentheses.

E 3 R-MAX MBIE ESSR RIVERSWIM 3,000,000 3,000,000 3,250,000 3,088,924 (?? 57,584) SIXARMS 1,800,000 2,800,000 9,250,000 7,327,222 (?? 1,189,460) DISPLAYFORM3 We evaluated the effectiveness of the proposed exploration bonus in a standard model-based algorithm.

In our implementation the agent updates its transition probability model and reward model through Equation 2 and its SSR estimate as in Definition 3.1 (the pseudo-code of this algorithm is available in the Appendix), which is then used for the exploration bonus r int .

We used the domains RiverSwim and SixArms BID23 to assess the performance of this algorithm.

2 These are traditional domains in the PAC-MDP literature BID9 and are often used to evaluate provably sampleefficient algorithms.

Details about these environments are also available in the Appendix.

We used the same protocol used by BID23 .

Our results are available in TAB0 .

It is interesting to see that our algorithm performs as well as R-MAX BID4 ) and E 3 (Kearns & Singh, 2002) on RiverSwim and it clearly outperforms these algorithms on SixArms.

In large environments, where enumerating all states is not an option, directly using the SSR as described in the previous section is not viable.

Learning the SSR becomes even more challenging when the representation, ??(??), is also being learned and so is non-stationary.

In this section we design an algorithm for the function approximation setting inspired by the results from the previous section.

Since explicitly estimating the transition probability function is not an option, we learn the SR directly using TD learning.

In order to capture the SSR we rely on TD's tendency to underestimate values when the estimates are pessimistically initialized, just as the SSR underestimates the true successor representation; with larger underestimates for states (and similarly features) that are rarely observed.

This is mainly due to the fact that when the SR is being learned with TD learning, because a reward of 1 is observed at each time step, there is no variance in the target and the predictions slowly approach the true value of the SR.

When pessimistically initialized, the predictions approach the target from below.

In this sense, what defines how far a prediction is from its final target is indeed how many times it has been updated in a given state.

Finally, recent work BID11 BID14 have shown successor features can be learned jointly with the feature representation itself.

These ideas are combined together to create our algorithm.

The neural network we used to learn the agent's value function while also learning the feature representation and the successor representation is depicted in Figure 1 .

The layers used to compute the state-action value function,q(S t , ??), are structured as in DQN BID17 , but with different numbers of parameters (i..e, filter sizes, stride, and number of nodes).

This was done to match Oh et al.'s (2015) architecture, which is known to succeed in the auxiliary task we define below.

From here on, we will call the part of our architecture that predictsq(S t , ??) DQN e .

It is trained to minimize DISPLAYFORM0 DISPLAYFORM1 This loss is known as the mixed Monte-Carlo return (MMC) and it has been used in the past by the algorithms that achieved succesful exploration in deep reinforcement learning BID2 BID20 .

The distinction between ?? and ?? ??? is standard in the field, with ?? ??? denoting the parameters of the target network, which is updated less often for stability purposes BID17 .

As before, we use r int to denote the exploration bonus obtained from the successor features of the internal representation, ??(??), which will be defined below.

Moreover, to ensure all features are in the same range, we normalize the feature vector so that ||??(??)|| 2 = 1.

In Figure 1 we highlight the layer in which we normalize its output with the symbol ?? .

Notice that the features are always non-negative due to the use of ReLU gates.

The successor features are computed by the two bottom layers of the network, which minimize the loss DISPLAYFORM2 Zero is a fixed point for the SR.

This is particularly concerning in settings with sparse rewards.

The agent might learn to set ??(??) = 0 to achieve zero loss.

We address this problem by not propagating ???L SR to ??(??) (this is depicted in Figure 1 as an open circle stopping the gradient), and by creating an auxiliary task BID7 to encourage a representation to be learned before a non-zero reward is observed.

As Machado et al. FORMULA3 , we use the auxiliary task of predicting the next observation, learned through the architecture proposed by BID18 , which is depicted as the top layers in Figure 1 .

The loss we minimize for this last part of the network is L Recons = ?? t+1 ??? S t+1 2 .

The last step in describing our algorithm is to define r int (S t ; ?? ??? ), the intrinsic reward we use to encourage exploration.

We choose the exploration bonus to be the inverse of the 2 -norm of the vector of successor features of the current state, that is, DISPLAYFORM0 where ??(S t ; ?? ??? ) denotes the successor features of state S t parametrized by ?? ??? .

The exploration bonus comes from the same intuition presented in the previous section, but instead of penalizing the agent with the norm of the SR we make r int (S t ; ?? ??? ) into a bonus (we observed in preliminary experiments not discussed here that DQN performs better when dealing with positive rewards).

Moreover, instead of using the 1 -norm we use the 2 -norm of the SR since our features have unit length in 2 (whereas the successor probabilities in the tabular-case have unit length in 1 ).Finally, we initialize our network the same way BID18 does.

We use Xavier initialization BID6 in all layers except the fully connected layers around the element-wise multiplication denoted by ???, which are initialized uniformly with values between ???0.1 and 0.1.

We followed the evaluation protocol proposed by BID13 .

We used MONTEZUMA'S REVENGE to tune our parameters (training set).

The reported results are the average over 10 seeds after 100 million frames.

We evaluated our agents in the stochastic setting (sticky actions, ?? = 0.25) using a frame skip of 5 with the full action set (|A| = 18).

The agent learns from raw pixels, that is, it uses the game screen as input.

Our results were obtained with the algorithm described in Section 4.

We set ?? = 0.025 after a rough sweep over values in the game MONTEZUMA'S REVENGE.

We annealed in DQN's -greedy exploration over the first million steps, starting at 1.0 and stopping at 0.1 as done by BID2 .

We trained the network with RMSprop with a step-size of 0.00025, an value of 0.01, and a decay of 0.95, which are the standard parameters for training DQN BID17 .

The discount factor, ??, is set to 0.99 and w TD = 1, w SR = 1000, w Recons = 0.001.

The weights w TD , w SR , and w Recons were set so that the loss functions would be roughly the same scale.

All other parameters are the same as those used by BID17 .

TAB2 summarizes the results after 100 million frames.

The performance of other algorithms is also provided for reference.

Notice we are reporting learning performance for all algorithms instead of the maximum scores achieved by the algorithm.

We use the superscript MMC to distinguish between the algorithms that use MMC from those that do not.

When comparing our algorithm, DQN MMC e +SR, to DQN we can see how much our approach improves over the most traditional baseline.

By comparing our algorithm's performance to DQN MMC +CTS BID2 and DQN MMC +PixelCNN BID20 we compare our algorithm to established baselines for exploration.

As highlighted in Section 4, the parameters of the network we used are different from those used in the traditional DQN network, so we also compared the performance of our algorithm to the performance of the same network our algorithm uses but without the additional modules (next state prediction and successor representation) by setting w SR = w Recons = 0 and without the intrinsic reward bonus by setting ?? = 0.0.

The column labeled DQN MMC e contains the results for this baseline.

This comparison allows us to explicitly quantify the improvement provided by the proposed exploration bonus.

The learning curves of these algorithms, their performance after different amounts of experience, and additional results analyzing, for example, the impact of the introduced auxiliary task, are available in the Appendix.

We can clearly see that our algorithm achieves scores much higher than those achieved by DQN, which struggles in games that pose hard exploration problems.

Moreover, by comparing DQN MMC e +SR to DQN MMC e we can see that the provided exploration bonus has a big impact in the game MONTEZUMA'S REVENGE, which is probably known as the hardest game among those we used in our evaluation.

Interestingly, the change in architecture and the use of MMC leads to a big improvement in games such as GRAVITAR and VENTURE, which we cannot fully explain.

However, notice that the change in architecture does not have any effect in MONTEZUMA'S REVENGE.

The proposed exploration bonus seems to be essential in this game.

Finally, we also compared our algorithm to DQN MMC +CTS and DQN MMC +PixelCNN.

We can observe that, on average, it performs as well as these algorithms, but instead of requiring a density model it requires the SR, which is already defined for every problem since it is a component of the value function estimates, as discussed in Section 2.

There are multiple algorithms in the tabular, model-based case with guarantees about their performance (e.g., BID4 BID10 BID23 BID19 .

RiverSwim and SixArms are domains traditionally used when evaluating these algorithms.

In this paper we have given evidence that our algorithm performs as well as some of these algorithms with theoretical guarantees.

Among these algorithms, R-MAX seems the closest approach to ours.

As with R-MAX, the algorithm we presented in Section 3 augments the state-space with an imaginary state and encourages the agent to visit that state, implicitly reducing the algorithm's uncertainty in the state-space.

However, R-MAX deletes the transition to this imaginary state once a state has been visited a given number of times.

Ours lets the probability of visiting this imaginary state vanish with additional visitations.

Moreover, notice that it is not clear how to apply these traditional algorithms such as R-MAX and E 3 to large domains where function approximation is required.

Conversely, there are not many model-free approaches with proven sample-complexity bounds (e.g., BID24 , but there are multiple model-free algorithms for exploration that actually work in large domains (e.g., BID22 BID2 BID20 BID21 .

Among these algorithms, the use of pseudo-counts through density models is the closest to ours BID2 BID20 .

Inspired by those papers we used the mixed Monte-Carlo return as a target in the update rule.

In Section 5 we have shown that our algorithm performs generally as well as these approaches without requiring a density model.

Importantly, BID15 had already shown that counting activations of fixed, handcrafted features in Atari 2600 games leads to good exploration behavior.

Nevertheless, by using the SSR we are not only counting learned features but we are also implicitly capturing the induced transition dynamics.

Finally, the SR has already been used in the context of exploration.

However, it was used to help the agent learn how to act in a higher level of abstraction in order to navigate through the state space faster BID12 BID14 .

Such an approach has led to promising results in the tabular case but only anecdotal evidence about its scalability has been provided when the idea was applied to large domains such as Atari 2600 games.

Importantly, the work developed by BID14 , BID11 and BID18 are the main motivation for the neural network architecture presented here.

BID18 have shown how one can predict the next screen given the current observation and action (our auxiliary task), while BID14 and BID11 have proposed different architectures for learning the successor representation from raw pixels.

RL algorithms tend to have high sample complexity, which often prevents them from being used in the real-world.

Poor exploration strategies is one of the main reasons for this high sample-complexity.

Despite all of its shortcomings, uniform random exploration is, to date, the most commonly used approach for exploration.

This is mainly due to the fact that most approaches for tackling the exploration problem still rely on domain-specific knowledge (e.g., density models, handcrafted features), or on having an agent learn a perfect model of the environment.

In this paper we introduced a general method for exploration in RL that implicitly counts state (or feature) visitation in order to guide the exploration process.

It is compatible to representation learning and the idea can also be adapted to be applied to large domains.

This result opens up multiple possibilities for future work.

Based on the results presented in Section 3, for example, we conjecture that the substochastic successor representation can be actually used to generate algorithms with PAC-MDP bounds.

Investigating to what extent different auxiliary tasks impact the algorithm's performance, and whether simpler tasks such as predicting feature activations or parts of the input BID7 are effective is also worth studying.

Finally, it might be interesting to further investigate the connection between representation learning and exploration, since it is also known that better representations can lead to faster exploration BID8 .

This supplementary material contains details omitted from the main text due to space constraints.

The list of contents is below:??? Pseudo-code of the model-based algorithm discussed in Section 3;??? Description of RiverSwim and SixArms, the tabular domains we used in our evaluation;??? Learning curves of DQN e and DQN MMC e +SR and their performance after different amounts of experience in the Atari 2600 games used for evaluation;??? Results of additional experiments designed to evaluate the role of the auxiliary task in the results reported in the paper for ESSR.

In the main paper we described our algorithm as a standard model-based algorithm where the agent updates its transition probability model and reward model through Equation 2 and its SSR estimate as in Definition 3.1.

The pseudo-code with details about the implementation is presented in Algorithm 1.

n(s, s ) ??? 0 ???s, s ??? S t(s, a, s ) ??? 1 ???s, s ??? S, ???a ??? ?? r(s, a) ??? 0 ???s ??? S, ???a ??? ?? P (s, a) ??? 1/|S| ???s ??? S, ???a ??? ?? P (s, s ) ??? 0 ???s, s ??? S ?? ??? random over A while episode is not over do Observe s ??? S, take action a ??? A selected according to ??(s), and observe a reward R and a next state s ??? S n(s, s ) ??? n(s, s ) + 1 DISPLAYFORM0 r int ??? ?????e ?? ??? POLICYITERATION(P ,r + ??r int ) end while

The two domains we used as testbed to evaluate the proposed model-based algorithm with the exploration bonus generated by the substochastic successor representation are shown in FIG2 .

These domains are the same used by BID23 .

For SixArms, the agent starts in state 0.

For RiverSwim, the agent starts in either state 1 or 2 with equal probability.

The algorithm we introduced in the paper, ESSR, relies on a network that estimates the state-action value function, the successor representation, and the next observation to be seen given the agent's current observation and action.

While the results depicted in TAB2 allow us to clearly see the benefit of using an exploration bonus derived from the successor representation, they do not inform us about the impact of the auxiliary task in the results.

The experiments in this section aim at addressing this issue.

We focus on Montezumas Revenge because it is the game where the problem of exploration is maximized, with most algorithms not being able to do anything without an exploration bonus.

The first question we asked was whether the auxiliary task was necessary in our algorithm.

We evaluated this by dropping the reconstruction module from the network to test whether the initial random noise generated by the successor representation is enough to drive representation learning.

It is not.

When dropping the auxiliary task, the average performance of this baseline over 4 seeds in MON-TEZUMA'S REVENGE after 100 million frames was 100.0 points (?? 2 = 200.0; min: 0.0, max: 400.0).

As comparison, our algorithm obtains 1778.6 points (?? 2 = 903.6, min: 400.0, max: 2500.0).

These results suggest that auxiliary tasks seem to be necessary for our method to perform well.

We also evaluated whether the auxiliary task was sufficient to generate the results we observed.

To do so we dropped the SR module and set ?? = 0.0 to evaluate whether our exploration bonus was actually improving the agent's performance or whether the auxiliary task was doing it.

The exploration bonus seems to be essential in our algorithm.

When dropping the exploration bonus and the successor representation module, the average performance of this baseline over 4 seeds in MONTEZUMA'S REVENGE after 100 million frames was 398.5 points (?? 2 = 230.1; min: 0.0, max: 400.0).

Again, clearly, the auxiliary task is not a sufficient condition for the performance we report.

The reported results use the same parameters as those reported in the main paper.

Learning curves for each individual run are depicted in Figure 3 .

after different amounts of experience (10, 50, and 100 million frames) in TAB5 Finally, Figure 4 depicts the learning curves obtained with the evaluated algorithms in each game.

Lighter lines represent individual runs while the solid lines encode the average over the multiple runs.

learning curves in the Atari 2600 games used as testbed.

The curves are smoothed with a running average computed using a window of size 100.

DISPLAYFORM0

@highlight

We propose the idea of using the norm of the successor representation an exploration bonus in reinforcement learning. In hard exploration Atari games, our the deep RL algorithm matches the performance of recent pseudo-count-based methods.