Deep reinforcement learning has achieved many recent successes, but our understanding of its strengths and limitations is hampered by the lack of rich environments in which we can fully characterize optimal behavior, and correspondingly diagnose individual actions against such a characterization.



Here we consider a family of combinatorial games, arising from work of Erdos, Selfridge, and Spencer, and we propose their use as environments for evaluating and comparing different approaches to reinforcement learning.

These games have a number of appealing features: they are challenging for current learning approaches, but they form (i) a low-dimensional, simply parametrized environment where (ii) there is a linear closed form solution for optimal behavior from any state, and (iii) the difficulty of the game can be tuned by changing environment parameters in an interpretable way.

We use these Erdos-Selfridge-Spencer games not only to compare different algorithms, but also to compare approaches based on supervised and reinforcement learning, to analyze the power of multi-agent approaches in improving performance, and to evaluate generalization to environments outside the training set.

Deep reinforcement learning has seen many remarkable successes over the past few years BID5 BID9 .

But developing learning algorithms that are robust across tasks and policy representations remains a challenge.

Standard benchmarks like MuJoCo and Atari provide rich settings for experimentation, but the specifics of the underlying environments differ from each other in many different ways, and hence determining the principles underlying any particular form of sub-optimal behavior is difficult.

Optimal behavior in these environments is generally complex and not fully characterized, so algorithmic success is generally associated with high scores, making it hard to analyze where errors are occurring in any sort of fine-grained sense.

An ideal setting for studying the strengths and limitations of reinforcement learning algorithms would be (i) a simply parametrized family of environments where (ii) optimal behavior can be completely characterized, (iii) the inherent difficulty of computing optimal behavior is tightly controlled by the underlying parameters, and (iv) at least some portions of the parameter space produce environments that are hard for current algorithms.

To produce such a family of environments, we look in a novel direction -to a set of two-player combinatorial games with their roots in work of Erdos and Selfridge BID3 , and placed on a general footing by BID10 .

Roughly speaking, these Erdos-Selfridge-Spencer (ESS) games are games in which two players take turns selecting objects from some combinatorial structure, with the feature that optimal strategies can be defined by potential functions derived from conditional expectations over random future play.

These ESS games thus provide an opportunity to capture the general desiderata noted above, with a clean characterization of optimal behavior and a set of instances that range from easy to very hard as we sweep over a simple set of tunable parameters.

We focus in particular on one of the best-known games in this genre, Spencer's attacker-defender game (also known as the "tenure game"; BID10 , in which -roughly speaking -an attacker advances a set of pieces up the levels of a board, while a defender destroys subsets of these pieces to try prevent any of them from reaching the final level ( FIG0 ).

An instance of the game can be parametrized by two key quantities.

The first is the number of levels K, which determines both the size of the state space and the approximate length of the game; the latter is directly related to the sparsity of win/loss signals as rewards.

The second quantity is a potential function φ, whose magnitude characterizes whether the instance favors the defender or attacker, and how much "margin of error" there is in optimal play.

The environment therefore allows us to study learning by the defender, or by the attacker, or in a multi-agent formulation where the defender and attacker are learning concurrently.

Because we have a move-by-move characterization of optimal play, we can go beyond simple measures of reward based purely on win/loss outcomes and use supervised learning techniques to pinpoint the exact location of the errors in a trajectory of play.

In the process, we are able to develop insights about the robustness of solutions to changes in the environment.

These types of analyses have been long-standing goals, but they have generally been approached much more abstractly, given the difficulty in characterizing step-by-step optimally in non-trivial environments such as this one.

The main contributions of this work are thus the following:1.

The development of these combinatorial games as environments for studying the behavior of reinforcement learning algorithms, with sensitive control over the difficulty of individual instances using a small set of natural parameters.2.

A comparison of the performance of an agent trained using deep RL to the performance of an agent trained using supervised learning on move-by-move decisions.

Exploiting the fact that we can characterize optimal play at the level of individual moves, we find an intriguing phenomenon: while the supervised learning agent is, not surprisingly, more accurate on individual move decisions than the deep RL agent, the deep RL agent is better at playing the game!

We further interpret this result by studying fatal mistakes.3.

An investigation of the way in which the success of one of the two players (defender or attacker) in training turns out to depend crucially on the algorithm being used to implement the other player.

We explore properties of this other player's algorithm, and also properties of mulitagent learning, that lead to more robust policies with better generalization.

This is a largely empirical paper, building on a theoretically grounded environment derived from a combinatorial game.

We present learning and generalization experiments for a variety of commonly used model architectures and learning algorithms.

We aim to show that despite the simple structure of the game, it provides both significant challenges for standard reinforcement learning approaches and a number of tools for precisely understanding those challenges.

We first introduce the family of Attacker-Defender Games BID10 , a set of games with two properties that yield a particularly attractive testbed for deep reinforcement learning: the ability to continuously vary the difficulty of the environment through two parameters, and the existence of a closed form solution that is expressible as a linear model.

An Attacker-Defender game involves two players: an attacker who moves pieces, and a defender who destroys pieces.

An instance of the game has a set of levels numbered from 0 to K, and N pieces that are initialized across these levels.

The attacker's goal is to get at least one of their pieces to level K, and the defender's is to destroy all N pieces before this can happen.

In each turn, the attacker proposes a partition A, B of the pieces still in play.

The defender then chooses one of the sets to destroy and remove from play.

All pieces in the other set are moved up a level.

The game ends when either one or more pieces reach level K, or when all pieces are destroyed.

FIG0 shows one turn of play.

With this setup, varying the number of levels K or the number of pieces N changes the difficulty for the attacker or the defender.

One of the most striking aspects of the Attacker-Defender game is that it is possible to make this tradeoff precise, and en route to doing so, also identify a linear optimal policy.

We start with a simple special case -rather than initializing the board with pieces placed arbitrarily, we require the pieces to all start at level 0.

In this special case, we can directly think of the game's difficulty in terms of the number of levels K and the number of pieces N .

Theorem 1.

Consider an instance of the Attacker-Defender game with K levels and N pieces, with all N pieces starting at level 0.

Then if N < 2 K , the defender can always win.

There is a simple proof of this fact: the defender simply always destroys the larger one of the sets A or B. In this way, the number of pieces is reduced by at least a factor of two in each step; since a piece must travel K steps in order to reach level K, and N < 2 K , no piece will reach level K.When we move to the more general case in which the board is initialized at the start of the game with pieces placed at arbitrary levels, it will be less immediately clear how to define the "larger" one of the sets A or B. We therefore give a second proof of Theorem 1 that will be useful in these more general settings.

This second proof BID10 ) uses Erdos's probabilistic method and proceeds as follows: for any attacker strategy, assume the defender plays randomly.

Let T be a random variable for the number of pieces that reach level K. Then T = T i where T i is the indicator that piece i reaches level K. DISPLAYFORM0 as the defender is playing randomly, any piece has probability 1/2 of advancing a level and 1/2 of being destroyed.

As all the pieces start at level 0, they must advance K levels to reach the top, which happens with probability 2 −K .

But now, by choice of N , we have that i 2 −K = N 2 −K < 1.

Since T is an integer random variable, E [T ] < 1 implies that the distribution of T has nonzero mass at 0 -in other words there is some set of choices for the defender that guarantees destroying all pieces.

This means that the attacker does not have a strategy that wins with probability 1 against random play by the defender; since the game has the property that one player or the other must be able to force a win, it follows that the defender can force a win.

Now consider the general form of the game, in which the initial configuration can have pieces at arbitrary levels.

Thus, at any point in time, the state of the game can be described by a K-dimensional vector S = (n 0 , n 1 , ..., n K ), with n i the number of pieces at level i.

Extending the argument used in the second proof above, we note that a piece at level l has a 2 DISPLAYFORM1 chance of survival under random play.

This motivates the following potential function on states: Definition 1.

Potential Function:

Given a game state S = (n 0 , n 1 , ..., n K ), we define the potential of the state as DISPLAYFORM2 Note that this is a linear function on the input state, expressible as φ(S) = w T · S for w a vector with w l = 2 −(K−l) .

We can now state the following generalization of Theorem 1.Theorem 2.

Consider an instance of the Attacker-Defender game that has K levels and N pieces, with pieces placed anywhere on the board, and let the initial state be S 0 .

Then (a) If φ(S 0 ) < 1, the defender can always win (b) If φ(S 0 ) ≥ 1, the attacker can always win.

One way to prove part (a) of this theorem is by directly extending the proof of Theorem 1, with This definition of the potential function gives a natural, concrete strategy for the defender: the defender simply destroys whichever of A or B has higher potential.

We claim that if φ(S 0 ) < 1, then this strategy guarantees that any subsequent state S will also have φ(S) < 1.

Indeed, suppose (renaming the sets if necessary) that A has a potential at least as high as B's, and that A is the set destroyed by the defender.

Since φ(B) ≤ φ(A) and φ(A) + φ(B) = φ(S) < 1, the next state has potential 2φ(B) (double the potential of B as all pieces move up a level) which is also less than 1.

In order to win, the attacker would need to place a piece on level K, which would produce a set of potential at least 1.

Since all sets under the defender's strategy have potential strictly less than 1, it follows that no piece ever reaches level K. DISPLAYFORM3 If φ(S 0 ) ≥ 1, we can devise a similar optimal strategy for the attacker.

The attacker picks two sets A, B such that each has potential ≥ 1/2.

The fact that this can be done is shown in Theorem 3, and in BID10 .

Then regardless of which of A, B is destroyed, the other, whose pieces all move up a level, doubles its potential, and thus all subsequent states S maintain φ(S) ≥ 1, resulting in an eventual win for the attacker.

The Atari benchmark BID5 is a well known set of tasks, ranging from easy to solve (Breakout, Pong) to very difficult (Montezuma's Revenge).

BID2 proposed a set of continuous environments, implemented in the MuJoCo simulator BID13 .

An advantage of physics based environments is that they can be varied continuously by changing physics parameters BID7 , or by randomizing rendering BID12 .

Deepmind Lab BID0 ) is a set of 3D navigation based environments.

OpenAI Gym BID1 contains both the Atari and MuJoCo benchmarks, as well as classic control environments like Cartpole BID11 and algorithmic tasks like copying an input sequence.

The difficulty of algorithmic tasks can be easily increased by increasing the length of the input.

Our proposed benchmark merges properties of both the algorithmic tasks and physics-based tasks, letting us increase difficulty by discrete changes in length or continuous changes in potential.

From Section 2, we see that the Attacker-Defender games are a family of environments with a difficulty knob that can be continuously adjusted through the start state potential φ(S 0 ) and the number of levels K. In this section, we describe a set of baseline results on Attacker-Defender games that motivate the exploration in the remainder of this paper.

We set up the Attacker-Defender environment as follows: the game state is represented by a K + 1 dimensional vector for levels 0 to K, with coordinate l representing the number of pieces at level l.

For the defender agent, the input is the concatenation of the partition A, B, giving a 2(K + 1) dimensional vector.

The game start state S 0 is initialized randomly from a distribution over start states of a certain potential.

We first look at training a defender agent against an attacker that randomly chooses between (mostly) playing optimally, and (occasionally) playing suboptimally, with the Disjoint Support Strategy.

This strategy unevenly partitions the occupied levels between A, B so that one set has higher potential than the other, with the proportional difference between the two sets being sampled randomly.

Note that this strategy gives rise to very different states A, B (uneven potential, disjoint occupied levels) than the optimal strategy, and we find that the model learns a much more generalizable policy when mixing between the two (Section 6).When testing out reinforcement learning, we have two choices of difficulty parameters.

The potential of the start state, φ(S 0 ), changes how optimally the defender has to play, with values close to 1 giving much less leeway for mistakes in valuing the two sets.

Changing K, the number of levels, directly affects the sparsity of the reward, with higher K resulting in longer games and less feedback.

Additionally, K also greatly increases the number of possible states and game trajectories (see Theorem 4).

theoretically expressive enough to learn the optimal policy for the defender agent.

In practice, we see that for many difficulty settings and algorithms, RL struggles to learn the optimal policy and performs more poorly than when using deeper models (compare to Figure 3 ).

An exception to this is DQN which performs relatively well on all difficulty settings.

Recall that the optimal policy can be expressed as a linear network, with the weights given by the potential function, Definition 1.

We therefore first try training a linear model for the defender agent.

We evaluate Proximal Policy Optimization (PPO) , Advantage Actor Critic (A2C) BID6 , and Deep Q-Networks (DQN) BID5 , using the OpenAI Baselines implementations BID4 .

Both PPO and A2C find it challenging to learn the harder difficulty settings of the game, and perform better with deeper networks FIG2 ).

DQN performs surprisingly well, but we see some improvement in performance variance with a deeper model.

In summary, while the policy can theoretically be expressed with a linear model, empirically we see gains in performance and a reduction in variance when using deeper networks (c.f.

Figures 3, 4.)

Having evaluated the performance of linear models, we try a deeper model for our policy net: a fully connected neural network with two hidden layers of width 300.

(Hyperparameters were chosen without extensive tuning and by trying a few different possible settings.

We found that two hidden layers generally performed best and the width of the network did not have much effect on the resutls.)

Identically to above, we evaluate PPO, A2C and DQN on varying start state potentials and K. Each algorithm is run with 3 random seeds, and in all plots we show minimum, mean, and maximum performance.

of potential (top and bottom row) with a deep network.

All three algorithms show a noticeable variation in performance over different difficulty settings, though we note that PPO seems to be more robust to larger K (which corresponds to longer episodes).

A2C tends to fare worse than both PPO and DQN. .

Unsurprisingly, we see that supervised learning is better on average at getting the ground truth correct move.

However, RL is better at playing the game: a policy trained through RL significantly outperforms a policy trained through supervised learning (right pane), with the difference growing for larger K.

One remarkable aspect of the Attacker-Defender game is that not only do we have an easily expressible optimal policy, but we know the ground truth on a per move basis.

We can thus compare RL to a Supervised Learning setup, where we classify the correct action on a large set of sampled states.

To carry out this test in practice, we first train a defender policy with reinforcement learning, saving all observations seen to a dataset.

We then train a supervised network (with the same architecture as the defender policy) to classify the optimal action.

This ensures both methods see exactly the same data points.

We then test the supervised network on how well it can play.

The results, shown in FIG4 are counter intuitive.

Supervised learning (unsurprisingly) has a higher proportion of correct moves: keeping count of the ground truth correct move for each turn in the game, the trained supervised policy network has a higher proportion of ground truth correct moves in play.

However, despite this, reinforcement learning is better at playing the game, winning a larger proportion of games.

These results are shown in FIG4 for varying choices of K.We conjecture that reinforcement learning is learning to focus most on moves that matter for winning.

To investigate this conjecture, we perform two further experiments.

Define a fatal mistake to be when the defender moves from a winning state (potential < 1) to a losing state (potential > 1) due to an incorrect move.

We count the number of fatal mistakes made by the trained supervised policy, and trained RL policy.

The results are shown in the left pane of FIG5 .

We see that supervised learning is much more prone to make fatal mistakes, with a sharp increase in fatal mistakes for larger K, supporting its sharp decrease in performance.

We also look at where mistakes are made by RL and Supervised Learning based on distance of the move from the end of the game.

We find that RL is better at the final couple of moves, and then consistently better in most of the earlier parts of the game.

This contrast forms an interesting counterpart to recent findings of BID9 , who in the context of Go also compared reinforcement learning to supervised approaches.

A key distinction is that their supervised work was relative to a heuristic objective, whereas in our domain we are able to compare to provably optimal play.

Returning to our RL Defender Agent, we would like to know how robust its learned policy is.

In particular, as we have so far been training our agent with a randomized but hard coded attacker, we would like to test how sensitive a defender agent is to the particular attacker strategy.

We investigate this in FIG7 where we first train a defender agent on the optimal attacker and test on the disjoint support attacker.

We notice a large drop in performance when switching from the optimal attacker to (less than 1.0 potential) to losing state (greater than 1.0 potential)) made by supervised learning compared to RL.

We find that Supervised Learning makes many more fatal mistakes, explaining its collapse in performance.

(2) (right pane) plot showing when (measured as distance to end game) RL and supervised learning make mistakes.

RL is more accurate than supervised learning at predicting the right action for the final couple of moves, and then drops quickly to a constant, whereas supervised learning is less accurate right at the very end and drops more slowly but much further, having lower accuracy than RL for many of the earlier moves.

and then tested on (a) another optimal attacker environment (b) the disjoint support attacker environment.

The left pane shows the resulting performance drop when switching to testing on the same opponent strategy as in training to a different opponent strategy.

The right pane shows the result of testing on an optimal attacker vs a disjoint support attacker during training.

We see that performance on the disjoint support attacker converges to a significantly lower level than the optimal attacker.

Figure 8: Performance of PPO and A2C on training the attacker agent for different difficulty settings.

DQN performance was very poor (reward < −0.8 at K = 5 with best hyperparams).

We see much greater variation of performance with changing K, which now affects the sparseness of the reward as well as the size of the action space.

There is less variation with potential, but we see a very high performance variance (top right pane) with lower (harder) potentials.

the disjoint support attacker.

As we know there exists an optimal policy which generalizes perfectly across all attacker strategies, this result suggests that the defender is overfitting to the particular attacker strategy.

One way to mitigate this overfitting issue is to set up a method of also training the attacker, with the goal of training the defender against a learned attacker, or even better, in the multiagent setting.

However, determining the correct setup to train the attacker agent first requires devising a tractable parametrization of the action space.

A naive implementation of the attacker would be to have the policy output how many pieces should be allocated to A for each of the K + 1 levels (as described in BID10 ).

This can grow exponentially in K, which is clearly impractical.

To address this, we first prove a theorem that enables us to show that we can parametrize an optimal attacker with a much smaller action space.

Theorem 3.

For any Attacker-Defender game with K levels, start state S 0 and φ(S 0 ) ≥ 1, there exists a partition A, B such that φ(A) ≥ 0.5, φ(B) ≥ 0.5, and for some l, A contains pieces of level i > l, and B contains all pieces of level i < l.

Proof.

For each l ∈ {0, 1, . . . , K}, let A l be the set of all pieces from levels K down to and excluding level l, with A K = ∅. We have φ(A i+1 ) ≤ φ(A i ), φ(A K ) = 0 and φ(A 0 ) = φ(S 0 ) ≥ 1.

Thus, there exists an l such that φ(A l ) < 0.5 and φ(A l−1 ) ≥ 0.5.

If φ(A l−1 ) = 0.5, we set A l−1 = A and B the complement, and are done.

So assume φ(A l ) < 0.5 and φ(A l−1 ) > 0.5Since A l−1 only contains pieces from levels K to l, potentials φ(A l ) and φ(A l−1 ) are both integer multiples of 2 −(K−l) , the value of a piece in level l. Letting φ(A l ) = n · 2 −(K−l) and φ(A l−1 ) = m · 2 −(K−l) , we are guaranteed that level l has m − n pieces, and that we can move k < m − n pieces from A l−1 to A l such that the potential of the new set equals 0.5.

Figure 9: Performance of attacker and defender agents when learning in a multiagent setting.

In the top panes, solid lines denote attacker performance.

In the bottom panes, solid lines are defender performance.

The sharp changes in performance correspond to the times we switch which agent is training.

We note that the defender performs much better in the multiagent setting: comparing the top and bottom left panes, we see far more variance and lower performance of the attacker compared to the defender performance below.

Furthermore, the attacker loses to the defender for potential 1.1 at K = 15, despite winning against the optimal defender in Figure 8 .

We also see (right panes) that the attacker has higher variance and sharper changes in its performance even under conditions when it is guaranteed to win.

This theorem gives a different attacker parametrization.

The attacker outputs a level l.

The environment assigns all pieces before level l to A, all pieces after level l to B, and splits level l among A and B to keep the potentials of A and B as close as possible.

Theorem 3 guarantees the optimal policy is representable, and the action space linear in K instead of exponential in K.With this setup, we train an attacker agent against the optimal defender with PPO, A2C, and DQN.

The DQN results were very poor, and so we show results for just PPO and A2C.

In both algorithms we found there was a large variation in performance when changing K, which now affects both reward sparsity and action space size.

We observe less outright performance variability with changes in potential for small K but see an increase in the variance (Figure 8 ).

With this attacker training, we can now look at learning in a multiagent setting.

We first explore the effects of varying the potential and K as shown in Figure 9 .

Overall, we find that the attacker fares worse in multiagent play than in the single agent setting.

In particular, note that in the top left pane of Figure 9 , we see that the attacker loses to the defender even with φ(S 0 ) = 1.1 for K = 15.

We can compare this to Figure 8 where with PPO, we see that with K = 15, and potential 1.1, the single agent attacker succeeds in winning against the optimal defender.

defender.

The figure single agent defender trained on the optimal attacker and then tested on the disjoint support attacker and a multiagent defender also tested on the disjoint support attacker for different values of K. We see that multiagent defender generalizes better to this unseen strategy than the single agent defender.

Finally, we return again to our defender agent, and test generalization between the single and multiagent settings.

We train a defender agent in the single agent setting against the optimal attacker, and test on a an attacker that only uses the Disjoint Support strategy.

We also test a defender trained in the multiagent setting (which has never seen any hardcoded strategy of this form) on the Disjoint Support attacker.

The results are shown in FIG0 .

We find that the defender trained as part of a multiagent setting generalizes noticeably better than the single agent defender.

We show the results over 8 random seeds and plot the mean (solid line) and shade in the standard deviation.

In this paper, we have proposed Erdos-Selfridge-Spencer games as rich environments for investigating reinforcement learning, exhibiting continuously tunable difficulty and an exact combinatorial characterization of optimal behavior.

We have demonstrated that algorithms can exhibit wide variation in performance as we tune the game's difficulty, and we use the characterization of optimal behavior to expose intriguing contrasts between performance in supervised learning and reinforcement learning approaches.

Having reformulated the results to enable a trainable attacker, we have also been able to explore insights on overfitting, generalization, and multiagent learning.

We also develop further results in the Appendix, including an analysis of catastrophic forgetting, generalization across different values of the game's parameters, and a method for investigating measures of the model's confidence.

We believe that this family of combinatorial games can be used as a rich environment for gaining further insights into deep reinforcement learning.

On the left we train on different potentials and test on potential 0.99.

We find that training on harder games leads to better performance, with the agent trained on the easiest potential generalizing worst and the agent trained on a harder potential generalizing best.

This result is consistent across different choices of test potentials.

The right pane shows the effect of training on a larger K and testing on smaller K. We see that performance appears to be inversely proportional to the difference between the train K and test K.

In the main text we examined how our RL defender agent performance varies as we change the difficulty settings of the game, either the potential or K. Returning again to the fact that the AttackerDefender game has an expressible optimal that generalizes across all difficulty settings, we might wonder how training on one difficulty setting and testing on a different setting perform.

Testing on different potentials in this way is straightforwards, but testing on different K requires a slight reformulation.

our input size to the defender neural network policy is 2(K + 1), and so naively changing to a different number of levels will not work.

Furthermore, training on a smaller K and testing on larger K is not a fair test -the model cannot be expected to learn how to weight the lower levels.

However, testing the converse (training on larger K and testing on smaller K) is both easily implementable and offers a legitimate test of generalization.

We find (a subset of plots shown in FIG0 ) that when varying potential, training on harder games results in better generalization.

When testing on a smaller K than the one used in training, performance is inverse to the difference between train K and test K.

Recently, several papers have identified the issue of catastrophic forgetting in Deep Reinforcement Learning, where switching between different tasks results in destructive interference and lower performance instead of positive transfer.

We witness effects of this form in the Attacker-Defender games.

As in Section 7, our two environments differ in the K that we use -we first try training on a small K, and then train on larger K. For lower difficulty (potential) settings, we see that this curriculum learning improves play, but for higher potential settings, the learning interferes catastrophically, FIG0

The significant performance drop we see in FIG7 motivates investigating whether there are simple rules of thumb that the model has successfully learned.

Perhaps the simplest rule of thumb is learning the value of the null set: if one of A, B (say A) consists of only zeros and the other (B) has some pieces, the defender agent should reliably choose to destroy B. Surprisingly, even this simple rule of thumb is violated, and even more frequently for larger K, FIG0 .

We can also test to see if the model outputs are well calibrated to the potential values: is the model more confident in cases where there is a large discrepancy between potential values, and fifty-fifty where the potential is evenly split?

The results are shown in FIG0 .

In the main paper, we mixed between different start state distributions to ensure a wide variety of states seen.

This begets the natural question of how well we can generalize across start state distribution if we train on purely one distribution.

The results in FIG0 show that training naively Confidence as a function of potential difference between states.

The top figure shows true potential differences and model confidences; green dots are moves where the model prefers to make the right prediction, while red moves are moves where it prefers to make the wrong prediction.

The right shows the same data, plotting the absolute potential difference and absolute model confidence in its preferred move.

Remarkably, an increase in the potential difference associated with an increase in model confidence over a wide range, even when the model is wrong.

In fact, the amount of possible starting states for a given K and potential φ(S 0 ) = 1 grows super exponentially in the number of levels K. We can state the following theorem:Theorem 4.

The number of states with potential 1 for a game with K levels grows like 2 DISPLAYFORM0 We give a sketch proof.

Proof.

Let such a state be denoted S.

Then a trivial upper bound can be computed by noting that each s i can take a value up to 2 (K−i) , and producting all of these together gives roughly 2 K/2 .For the lower bound, we assume for convenience that K is a power of 2 (this assumption can be avoided).

Then look at the set of non-negative integer solutions of the system of simultaneous equations a j−1 2 1−j + a j 2 −j = 1/K where j ranges over all even numbers between log(K) + 1 and K. The equations don't share any variables, so the solution set is just a product set, and the number of solutions is just the product

As the optimal defender policy is expressible as a linear model, we empirically investigate whether depth is helpful.

We find that even with a temperature included, linear models perform worse than models with one or two hidden layers.

<|TLDR|>

@highlight

We adapt a family of combinatorial games with tunable difficulty and an optimal policy expressible as linear network, developing it as a rich environment for reinforcement learning, showing contrasts in performance with supervised learning, and analyzing multiagent learning and generalization. 