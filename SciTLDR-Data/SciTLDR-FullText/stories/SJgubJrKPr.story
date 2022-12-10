Conventional deep reinforcement learning typically determines an appropriate primitive action at each timestep, which requires enormous amount of time and effort for learning an effective policy, especially in large and complex environments.

To deal with the issue fundamentally, we incorporate macro actions, defined as sequences of primitive actions, into the primitive action space to form an augmented action space.

The problem lies in how to find an appropriate macro action to augment the primitive action space.

The agent using a proper augmented action space is able to jump to a farther state and thus speed up the exploration process as well as facilitate the learning procedure.

In previous researches, macro actions are developed by mining the most frequently used action sequences or repeating previous actions.

However, the most frequently used action sequences are extracted from a past policy, which may only reinforce the original behavior of that policy.

On the other hand, repeating actions may limit the diversity of behaviors of the agent.

Instead, we propose to construct macro actions by a genetic algorithm, which eliminates the dependency of the macro action derivation procedure from the past policies of the agent.

Our approach appends a macro action to the primitive action space once at a time and evaluates whether the augmented action space leads to promising performance or not.

We perform extensive experiments and show that the constructed macro actions are able to speed up the learning process for a variety of deep reinforcement learning methods.

Our experimental results also demonstrate that the macro actions suggested by our approach are transferable among deep reinforcement learning methods and similar environments.

We further provide a comprehensive set of ablation analysis to validate our methodology.

Conventional deep reinforcement learning (DRL) has been shown to demonstrate superhuman performance on a variety of environments and tasks (Mnih et al., 2013 (Mnih et al., , 2015 Salimans et al., 2017; Moriarty et al., 1999; .

However, in conventional methods, agents are restricted to make decisions at each timestep, which differs much from the temporally-extended framework of decision-making in human beings.

As a consequence, traditional methods (Mnih et al., 2013 Houthooft et al., 2016) require enormous amounts of sampling data in environments where goals are hard to reach or rewards are sparse.

In complex environments where goals can only be achieved by executing a long sequence of primitive actions, it is difficult to perform exploration efficiently.

As most real-world environments are large, complex, and usually offer sparse rewards, finding an optimal policy is still hard and challenging.

It becomes crucial to explore new mechanisms to deal with these environments more efficiently and effectively.

Researchers in the past few years have attempted various techniques to expand the realm of DRL to temporally-extended frameworks (Sutton et al., 1999; Vezhnevets et al., 2016; Kulkarni et al., 2016; Bacon et al., 2017; Frans et al., 2017; Daniel et al., 2016; Florensa et al., 2017; Machado et al., 2017) .

In such frameworks, a high-level controller interacts with the environment by selecting temporal-extended policies usually named as "options".

Once an option is selected, it interacts with the environment for a certain timesteps and perform primitive actions until a termination condition for that option is met.

However, developing effective options either requires a significant amount of domain knowledge (Girgin et al., 2010) , or often restricted to low-dimensional and/or relatively simple environments only (Bacon et al., 2017; Heess et al., 2016; Kulkarni et al., 2016) .

Instead of developing options, another branch of research directions focus on constructing macro actions (Fikes and Nilsson, 1971; Siklossy and Dowson, 1977; Minton, 1985; Pickett and Barto, 2002; Botea et al., 2005; Newton et al., 2005 Newton et al., , 2007 .

A macro action (or simply "a macro") is an open-loop (DiStefano III et al., 1967 ) policy composed of a finite sequence of primitive actions.

Once a macro is chosen, the actions will be taken by the agent without any further decision making process.

Some researches in DRL attempt to construct macros from the experience of an agent (Durugkar et al., 2016; Randlov, 1999; Yoshikawa and Kurihara, 2006; Onda and Ozawa, 2009; Garcia et al., 2019) .

A key benefit of these approaches is the ease to construct a desired macro without supervision (Durugkar et al., 2016) .

However, these approaches may lead to biased macros.

For example, the most frequently used sequence of actions may not correspond to a macro that can lead the agent to outperform its past policies.

Furthermore, as agents generally perform exploration extensively in the early stages of training, the inconsistency in the early experience may perturb the construction of macros.

A few researchers proposed to employ a reduced form of macro called action repeat Sharma et al., 2017) .

In this formulation, primitive actions are repeated several times in a macro before the agent makes another decision.

However, this formulation may limit the diversity of macros.

By relaxing the agent to perform macros consisting of diversified actions, the agent is granted more chances to achieve higher performance.

In addition, there are a handful of researches that requires human supervision to derive macros for improving training efficiency.

The authors in McGovern et al. (1997) show that handcrafted macros can speed up training in certain tasks but hinder performance in others.

The authors in Heecheol et al. (2019) generate macros from expert demonstrations via a variational auto-encoder.

However, the process of obtaining such demonstrations is expensive.

It would thus be favorable if there exists a method to find a macro without human intervention.

Nevertheless, little attention has been paid to the construction of such macros.

Our goal is to develop a methodology for constructing a macro action from possible candidates.

As possible macros are allowed to have different lengths and arbitrary compositions of primitive actions, such diversified macro actions essentially form an enormous space.

We define this space as the macro action space (or simply "macro space").

Repeated action sequences are simply a small subset of the macro space.

For a specific task in an environment, we hypothesize that there are good macros and bad macros in the macro space.

Different macro actions have different performance impacts to an agent.

Good macro actions enable the agent to jump over multiple states and reach a target state quicker and easier.

On the other hand, bad macro actions may lead the agent to undesirable states.

We argue that whether a macro is good or bad can only be determined by direct evaluation.

In this study, we propose an evaluation method to test whether a macro is satisfactory for an agent to perform a specific task in an environment.

Our method first relaxes the conventional action space (Sutton and Barto, 2018 ) with a macro to form an augmented action space.

We then equip the agent with the augmented action space, and utilize the performance results as the basis for our evaluation.

In order to find a good macro in the vast macro space, a systematic method is critically important and necessary.

The method entails two prerequisites: a macro construction mechanism and a macro evaluation method.

Although the second one is addressed above, there is still a lack of an appropriate approach to construct macros.

To satisfy the above requirement, we embrace an genetic algorithm (or simply "GA") for macro construction.

GA offers two promising properties.

First, it eliminates the dependency of the macro action derivation procedure from the past policies of an agent and/or human supervision.

Second, it produces diversified macros by mutation.

In order to combine GA with our evaluation method, our approach comprises of three phases: (1) macro construction by GA; (2) action space augmentation; and (3) evaluation of the augmented action space.

Our augmented action space contains not only the original action space defined by DRL, but also the macro(s) constructed by GA.

To validate the proposed approach, we perform our experiments on Atari 2600 (Brockman et al., 2016) and ViZDoom (Kempka et al., 2016) , and compare them to two representative DRL baseline methods.

We demonstrate that our proposed method is complementary to existing DRL methods, and perform favorably against the baselines.

Moreover, we show that the choice of the macro have a crucial impact on the performance of an agent.

Furthermore, our results reveal the existence of transferability of a few macros over similar environments or DRL methods.

We additionally provide a comprehensive set of ablation analysis to justify various aspects of our approach.

The contributions of this paper are summarized as follows:

• We define the proposed approach as a framework.

• We provide a definition of macro action space.

• We introduce an augmentation method for action spaces.

• We propose an evaluation method to determine whether a macro is good or not.

• We establish a macro action construction method using GA for DRL.

• We investigate and reveal the transferability of macro actions.

The rest of this paper is organized as follows.

Section 2 explains our framework.

Section 3 describes our implementation details.

Section 4 presents our results.

Section 5 concludes.

In this section, we first provide the definition of macro actions.

Then, we provide a model of the environment with macros, which is a special case of Semi-Markov Decision Processes (SMDP) (Sutton et al., 1999) .

Next, we provide definitions of functions in DRL.

Finally, we formulate a framework for constructing good macros.

The essential notations used in this paper can be referenced in Table A1 in our appendices.

Macro action.

A macro action is defined as a finite sequence of primitive actions m = (a 1 , a 2 , . . .

, a k ), for all a i in action space A, and some natural number k. Macros can be selected atomically as one of the actions by an agent.

The set of macros form a macro action space M, which can be represented as M = A + , where '+' stands for Kleene plus (Hopcroft and Ullman, 1979) .

Deep reinforcement learning.

An agent interacts with an environment under a policy ν, where ν is a mapping, ν : S × M → [0, 1].

The expected cumulative rewards it receives from each state s under ν can be denoted as V ν (s).

The objective of DRL is to train an agent to learn an optimal policy such that it is able to maximize its expected return.

The maximal expected return from each state s under the optimal policy can be denoted as V * M (s).

The expressions of V ν and V * M can be represented as Eqs. 4 and 5, respectively, where γ ∈ [0, 1].

We only use γ between macros but not between the primitive actions within a macro.

This encourages the agent to prefer the provided macros over a series of primitive actions.

Framework.

We define our framework as a 4-tuple (R, C , A , E ), where R is the collection of all DRL methods, C the collection of all macro action construction methods, A the collection of all action space augmentation methods, and E the collection of all evaluation methods.

Following this framework, in this study, we select Proximal Policy Optimization (PPO) and Advantage Actor Critic (A2C) as our DRL methods.

Our implementations of the latter three components of the 4-tuple are formulated as pseudocodes and are presented in Algorithms 1, A1, and A2, respectively.

Algorithm 1 Macro construction algorithm based on GA 1: input: Environment E; DRL algorithm R; Total number of evaluated macros k 2: input: The sizes of sets Q, Q+, Q * = q, q+, q * respectively 3: output: List of the top q highest-performing macros evaluated Q 4: function Construction(E, R, k, q, q+, q * ) 5: initialize: A = the primitive action space of E 6:

initialize: M = [m1, . . . , mq], a list of q randomly generated macros s.t.

∀ m ∈ M, |m| = 2 7:

initialize: F = [f1, . . . , fq], ∀ f ∈ F, f = 0, the fitness scores for all the macros in M 8:

initialize: Q = ∅, Q+ = ∅, Q * = ∅, i = 0 Q contains the top q macros at any generation 9:

while i < k do Each iteration is one generation 10:

for j in 1 to q do 11: M = Augment(A, mj) Please refer to Algorithm A1 12: fj = Evaluate(E, R, M) Please refer to Algorithm A2 13:

if i >= k then break 15:

end for 16:

for m in List of q+ randomly selected macros in Q do 18:

end for Q+ will hold q+ mutated macros using append operator 20:

for m in List of q * randomly selected macros in Q do 21:

Q * = Q * ∪ {Alter(A, m)} Please refer to Algorithm A4 22:

end for Q * will hold q * mutated macros using alternation operator 23:

end while 25:

return Q 26: end function

In this section, we present our implementation details of GA for constructing macro actions.

We formulate our macro construction algorithm based on GA as Algorithm 1, which is established atop (1) the action space augmentation method (presented as Algorithm A1 and accessed at line 11 of Algorithm 1), (2) the macro evaluation method (presented as Algorithm A2 and accessed at line 12 of Algorithm 1), as well as (3) the append operator (accessed at line 19) and (4) the alteration operator (accessed at line 22) presented in Fig. 1 (a) and the appendices for mutating the macros.

We walk through Algorithm 1 and highlight the four phases of GA as follows.

Lines 1-3 declare the input and output of the algorithm.

Lines 5-8 correspond to the "initialization phase", which initializes the essential parameters.

The population of the macros M are initialized at line 6.

Lines 10-15 correspond to the "fitness phase", which augments the action space and performs fitness evaluation.

The fitness phase can be executed in parallel by multiple threads.

Lines 16-17 corresponds to the "selection phase", which selects the top performers from the population.

Lastly, lines 18-23 correspond to the "mutation phase", which alters the selected macros and updates the population.

In this section, we present our experimental results of Algorithm 1 and discuss their implications.

We arrange the organization of our presentation as follows in order to validate the methodology we proposed in the previous sections.

First, we walk through the experimental setups.

Second, we compare the constructed macro actions of different generations to examine the effectiveness of Algorithm 1.

Third, we investigate the compatibility of the constructed macros with two off-the-shelf DRL methods.

Then, we explore the transferability of the constructed macros between the two DRL methods.

We next demonstrate that the constructed macros can be transferred to harder environments.

Finally, we present a set of ablation analysis to inspect the proposed methodology from different perspectives.

Please note that we only present the most representative results in this section.

We strongly suggest the interested readers to refer to our appendices for more details.

We first present the environments used in our experiments, followed by a brief description of the baselines adopted for comparison purposes.

All of the macros presented throughout this section are constructed by Algorithm 1, if not specifically specified.

We tabularize the environmental configurations, the hyper-parameters of Algorithm 1, and the hyperparameters used during the training phase in our appendices.

Except for Section 4.2, all of the curves presented in this section are generated based on five random seeds and drawn with 95% confidence interval (displayed as the shaded areas).

Environments.

We employ two environments with discrete primitive action spaces in our experiments: Atari 2600 (Bellemare et al., 2013) and ViZDoom (Wydmuch et al., 2018) .

For Atari 2600 , we select seven representative games to evaluate our methodology: BeamRider , Breakout, Pong, Q*bert, Seaquest, SpaceInvaders, and Enduro.

Due to the limited space, we use only the former six games for presenting our comparisons and analyses in Section 4, while leaving the remainder of our results in the appendices.

For ViZDoom, we evaluate our methodology on the default task my way home (denoted as "Dense") for comparing the macros constructed among generations.

We further use the "Sparse", "Very Sparse" and, "Super Sparse" (developed by us) tasks for analyzing the properties of compatibility and transferability of the constructed macros mentioned above.

The Super Sparse task comes with extended rooms and corridors in which the distance between the spawn point of the agent and the goal is farther than the other three tasks.

The map layouts for these four tasks are depicted in Figs. 1 (b) and 1 (c) .

Baselines.

In our experiments, we select PPO and A2C as our DRL baselines for training the agents.

For the ViZDoom environment, we further incorporate the intrinsic curiosity module (ICM) (Pathak et al., 2017) in the Sparse, Very Sparse, and Super Sparse tasks to generate intrinsic rewards for motivating exploration.

ViZDoom setups for transferability evaluation.

We use ViZDoom to validate our hypothesis that macro actions constructed from an easy environment can be transferred to complex environments.

In order to perform this validation study, we use Algorithm 1 to construct macro actions from the Dense task and evaluate their performance after 5M training timesteps.

The highest performing macro action is then selected and evaluated in the Sparse, Very Sparse, and Super Sparse tasks respectively after 10M training timesteps.

with the corresponding confidence interval stands for the average performance of the best macros in the generation (i.e., Q in Algorithm 1).

It is observed from the trends that the mean episode rewards obtained by the agents improve with generations, revealing that later generations of the constructed macros may outperform earlier ones.

The improving trends suggest that the evaluation method of Algorithm A2 is effective and reliable for Algorithm 1.

In order to examine whether the best macros constructed by our methodology is complementary to the baseline DRL methods, we first execute Algorithm 1 with A2C and PPO, and determine the best macrosm A2C andm PPO for the two DRL baselines respectively.

We then form the augmented action spaces M A2C = A ∪ {m A2C } and M PPO = A ∪ {m PPO }.

We next train A2C using M A2C and PPO using M PPO for 10M timesteps respectively.

The results evaluated on BeamRider are shown in Fig. 3 (a) .

The learning curves reveal that the baseline DRL methods are both significantly benefited from the augmented action spaces.

More supporting evidences are presented in Figs. 3 and 4.

Table 1 presents our evaluation results for the seven Atari games.

It is observed that all of the constructed macros are able to considerably enhance the performance of the baseline DRL methods, especially for Enduro.

In order to inspect whether the macro constructed for one DRL baseline can be transferred to and utilized by another DRL baseline, we train PPO with M A2C for 10M timesteps and plot the results on Q*bert, Pong, and Breakout in Figs. 3 (b) , 3 (c), and 3 (d), respectively.

The results show that PPO is able to usem A2C to enhance both the performance and the learning speed of the agents.

Additional experimental results are provided in our appendices.

In order to investigate the transferability of the constructed macros among similar environments, we first execute Algorithm 1 with A2C and ICM (together denoted as "Curiosity") on Dense to construct a macrom D .

We choose to train on Dense because the agent can learn from both the dense and sparse reward scenarios at different spawn points, enabling it to adapt to different reward sparsity settings easier.

We then form the augmented action space thus validate the transferability of the constructed macros to harder environments.

Fig. 4 also reveals that the gap between 'Curiosity+m D ' and 'Curiosity' grows as the sparsity of the reward signal increases.

For the Super Sparse task, 'Curiosity+m D ' converges at around 13M timesteps, while 'Curiosity' just begins to learn at about 25M timestep.

We compare mean time required to reach the goal w/ and w/om D over 100 episodes in our appendices.

The macro actionm D constructed by Algorithm 1 is (2, 2, 1), which corresponds to two forward moves followed by one right turn.

We speculate that this macro action is transferable because the map layouts favor forward moves and right turns.

Moving forward is more essential than turning right for the four ViZDoom tasks, as the optimal trajectories comprise of a number of straight moves.

On the contrary, left turn is rarely performed by the agent, as this choice of action is only adopted when the agent is deviated from its optimal routes.

To justify our decision of selecting GA as our macro construction method, we additionally compare GA with three different macro construction methods to validate that GA is superior to them in most cases and possess several promising properties required by our methodology.

Comparison with random macros.

We first compare the performance of the macros constructed by Algorithm 1 against randomly constructed macros.

We present the learning curves of 'A2C withm A2C ' and 'A2C withm D ' against 'A2C with the best random macrô m Random ' in Figs. 5 (a) and 5 (b), respectively.

For a fair comparison, we randomly select 50 out of all possible macros with lengths from 2 to 5, and evaluate each random macro for 5M timesteps for 2 times similar to how GA evaluates.

We limit the lengths of the macros from 2 to 5 because it is the range GA has attempted.

Following this procedure, the macro with highest evaluation score is selected to bem Random .

The agents are then trained for 10M timesteps, and the results are plotted in Figs. 5 (a) and 5 (b) (denoted as "Random").

It is observed that the two curves for 'A2C withm Random ' are not obviously superior to those of 'A2C withm A2C ' and 'A2C withm D '.

We further compare the 50 randomly constructed macros with the 50 macros constructed in the course of Algorithm 1, and summarize their impacts on performance as distributions over mean scores in Fig. 5 (c) .

It is observed that a larger portion of the macros constructed by our approach tend to result in higher mean scores than those constructed randomly.

A breakdown and analysis are offered in our appendices.

Comparison with action repeat.

We next compare the performance of 'A2C withm A2C ' and 'A2C withm D ' against 'A2C with the best action repeat macrom Repeat ', and plot the learning curves in Figs we evaluate all possible action repeat macros with the same length tom A2C andm D for five random seeds and for 10M timesteps.

The macro with highest evaluation score is then selected asm Repeat .

It is observed that the curves of 'A2C withm Repeat ' are significantly worse than those of 'A2C withm A2C ', 'A2C withm D ', and 'A2C withm Random ' in the figures.

The low performance of them is probably due to the lack of diversity inm Repeat .

Comparison with handcrafted macros.

We further investigate whether human intervention affects the outcome of Algorithm 1.

As the initial population of the macros used by Algorithm 1 are randomly constructed, in this ablation study we replace the initial population by handcrafted macros (1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1, 1), which correspond to consecutive 'fire' action in BeamRider .

These handcrafted macros are advantageous to the agent from the perspective of human players, as waves of enemies continue to emerge throughout the game.

The macro constructed in this manner is denoted asm Human .

Fig. 5 (d) shows a comparison of the learning curves of 'A2C withm A2C ' and 'A2C withm Human '.

It is observed that the two curves are almost aligned.

From the analysis, we therefore conclude that our approach does not need human intervention, and is unaffected by the initial population of the macros.

We have formally presented a methodology to construct macro actions that may potentially improve both the performance and learning efficiency of the existing DRL methods.

The methodology falls within the scope of a broader framework that permits other possible combinations of the DRL method, the action space augmentation method, the evaluation method, as well as the macro action construction method.

We formulated the proposed methodology as a set of algorithms, and used them as the basis for investigating the interesting properties of macro actions.

Our results revealed that the macro actions constructed by our methodology are complementary to two representative DRL methods, and may demonstrate transferability among different DRL methods and similar environments.

We additionally compared our methodology against three other macro construction methods to justify our design decisions.

Our work paves a way for future research on macros and their applications.

The appendices are organized as follows.

Section A1 lists our essential notations.

Section A2 describes the implementations of the genetic operators.

Section A3 summarizes the hyperparameters used.

Section A4 shows the additional experimental results.

Finally, Section A5 describes our computing infrastructure.

initialize: E = [e1, . . .

, eN ] ∀ e ∈ E, e = 0 5:

Learn a policy over M in E using R 7: ei = the average of all "last 100 episode rewards" 8: end for 9:

return Average of E 10: end function Let α = random element in A 6:

return m = (α, a2, . . .

, a |m| ) 7: end function

We used A2C and PPO implementations from Hill et al. (2018) and established our models on top of them.

We implemented ICM (Pathak et al., 2017) along with A2C (together denoted as "Curiosity") instead of Asynchronous Advantage Actor-Critic (A3C) , and found that our implementations tend to be stabler in terms of the variance of learning curves from several runs.

The detailed hyperparameters for each method are summarized in Table A2 .

The GA-related parameters used in Algorithm 1 are presented in Table A3 .

The primitive actions we used are summarized in Table A4 .

In addition to the results presented in Section 4.6, we further illustrate a more detailed comparison of the macros constructed by our methodology and the randomly constructed macros in Fig. A4 .

In this study, we use both our customized machines and the virtual machines named n1-statnd-64 on Google Cloud Platform (GCP).

We list the specification per instance in Table A6 .

In total, we utilized up to 320 virtual CPU cores, 8 graphics cards, and around 1 TB memory in our experiments.

Figure A4 : Performance comparison of the macros constructed via Algorithm 1 (the red bars) and the random construction method described in Section 4.6 (the gray bars) in terms of the mean scores of the last 100 episodes achieved by the A2C agents using the macros.

Please note that we present only the results of the top 50 performers out of the constructed macros for the Seaquest and Dense tasks.

@highlight

We propose to construct macro actions by a genetic algorithm, which eliminates the dependency of the macro action derivation procedure from the past policies of the agent.

@highlight

This paper proposes a generic algorithm for constructing macro actions for deep reinforcement learning by appending a macro action to the primitive action space.