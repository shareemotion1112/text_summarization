In compressed sensing, a primary problem to solve is to reconstruct a high dimensional sparse signal from a small number of observations.

In this work, we develop a new sparse signal recovery algorithm using reinforcement learning (RL) and Monte CarloTree Search (MCTS).

Similarly to orthogonal matching pursuit (OMP), our RL+MCTS algorithm chooses the support of the signal sequentially.

The key novelty is that the proposed algorithm learns how to choose the next support as opposed to following a pre-designed rule as in OMP.

Empirical results are provided to demonstrate the superior performance of the proposed RL+MCTS algorithm over existing sparse signal recovery algorithms.

We consider the compressed sensing (CS) problem [1; 2; 3] , where for a given matrix A ∈ R m×n , m n, and a (noiseless) observation vector y = Ax 0 , we want to recover a k-sparse vector/signal x 0 (k < m).

Formally, it can be formulated as:

subject to Ax = Ax 0 (2)

Related work There is a large collection of algorithms for solving the CS problem.

Some foundational and classic algorithms include convex relaxation, matching and subspace pursuit [4; 5; 6] and iterative thresholding [7; 8] .

In particular, two well-established methods are (i) Orthogonal Matching Pursuit (OMP) and (ii) Basis Pursuit (BP).

OMP recovers x 0 by choosing the columns of A iteratively until we choose k columns [9] .

BP recovers x 0 by solving min Ax=y ||x|| 1 [2] .

Because OMP and BP are extremely well studied theoretically [1; 2] and empirically [10] , we use these two algorithms as the main baseline methods to compare against when evaluating the proposed RL+MCTS algorithm.

Recent advancements in machine learning have opened a new frontier for signal recovery algorithms.

Specifically, these algorithms take a deep learning approach to CS and the related error correction problem.

The works in [11] , [12] , [13] and [14] apply ANNs and RNNs for encoding and/or decoding of signals x 0 .

Modern generative models such as Autoencoder, Variational Autoencoder, and Generative Adversarial Networks have also been used to tackle the CS problem with promising theoretical and empirical results [15; 16; 17] .

These works involve using generative models for encoding structured signals, as well as for designing the measurement matrix A. Notably, the empirical results in these works typically use structured signals in x 0 .

For example, in [16] and [17] , MNIST digits and celebrity images are used for training and testing.

Our contribution Differently from the above learning-based works, our innovation with machine learning is on signal recovery algorithms (as opposed to signal encoding or measurement matrix design).

We do not assume the signals to be structured (such as images), but cope with general sparse signals.

This underlying model for x 0 is motivated by the same assumptions in the seminal work on universal phase transitions by Donoho and Tanner in [10] .

Moreover, we assume the measurement matrix A is given.

Extending to varying matrices A is left for future investigation.

In this work, we approach the signal recovery problem using reinforcement learning (RL).

Specifically, we leverage the Monte Carlo Tree Search (MCTS) technique with RL, which was shown to achieve outstanding performance in the game of Go [18; 19] .

We further introduce special techniques to reduce the computational complexity for dealing with higher signal sparsity in CS.

Experimental results show that the proposed RL+MCTS algorithm significantly outperforms OMP and BP for matrix A of various sizes.

In this section, we formulate the sparse signal recovery problem as a special sequential decision making problem, which we will solve using RL and MCTS.

In the context of compressed sensing, a key challenge is to correctly choose the columns of A, or equivalently, the support of x 0 , such that the problem (1) is solved.

To address this problem, we formulate it as a sequential decision making problem: an agent sequentially chooses one column of A at a time until it selects up to k columns such that the constraint in (2) holds and the 0 -loss in (1) is minimized.

The MDP for compressed sensing can then be defined as follows.

A state s ∈ S is a pair (y, S), where y is the observed signal generated according to x 0 , and S ⊆

[n] is the set of the already selected columns of A, where [n] {1, . . .

, n}. In our current setup, we assume the matrix A is fixed, so a state is not dependent on the sensing matrix.

Terminal states are states s = (y, S) which satisfy one or more of the following conditions: (i) |S| = k (the maximum possible signal sparsity), or (ii) ||A S x s − y|| 2 2 < for some pre-determined .

Here, A S stands for the submatrix of A that is constructed by the columns of A indexed by the set S, and x s is the optimal solution given that the signal support is S,

For the action space, the set of all feasible actions at state

Note that in compressed sensing, when an action a is taken (i.e., a new column of A is selected) for a particular state s = (y, S), the next state s is determined; that is, the MDP transition is deterministic.

Finally, we define our reward function R:

where α, γ > 0 are fixed hyperparameters, and x s is determined by (3).

Different from existing compressed sensing algorithms, we propose to learn, via RL and MCTS, a policy to sequentially select the columns of A and reconstruct the sparse signal x 0 , based on data generated for training.

We generate the training data by generating k-sparse signals x 0 and computing the corresponding vectors y = Ax 0 (each k is randomly generated from 1 to m).

For each signal y, we then use a "policy network" (to be explained in details later) along with MCTS to choose columns sequentially until k columns have been chosen.

The traversed states will be used as our new training data for updating the policy network.

Such a strategy allows us to move as much of the computational complexity as possible in testing (i.e., performing the sparse signal recovery task) into training, which shares a similar spirit to the work in [20] .

3 The RL+MCTS Algorithm

To learn a policy in the above sequential decision making formulation of CS, we employ a single neural network f θ to jointly model the policy π θ (a|s) and the state-value function V θ (s), where θ is the model parameter (i.e., the weights in a neural network).

The policy π θ (a|s) defines a probability over all actions for a given state s, where the action set includes the possible next columns of A to pick and a stopping action.

The value V θ (s) defines the long-term reward that an agent receives when we start from the state s and follow the given policy.

We design two sets of input features for the policy/value network.

The first set of input features is x s extended to a vector in R n with zeros in components whose indices are not in s. The second set of features is motivated by OMP, which is given by λ s := A T (y − A S x s ) ∈ R n , where y − A S x s is the residual vector associated with the solution x s .

For the root state r in which no columns are chosen, x r is set to be the n-dimensional zero vector, and λ r := A T y. Note that the OMP rule is exactly choosing the next column index whose corresponding component in |λ s | is the largest, where | · | is the absolute value taken component wise.

The goal of the RL+MCTS algorithm is to iteratively train the policy network f θ .

The high-level training structure is given in Algorithm 1.

Algorithm 1 High-Level Training Procedure 1: initialize: j = 0, θ = θ0, θ0 random, fixed matrix A ∈ R m×n 2: while j < i (where i is a hyperparameter) do 3:

1) generate training samples from each (y, x0) pair by building a tree using Monte Carlo Tree Search (MCTS) and current f θ

2) train/update neural network parameters to getθ using the training samples from step 1.

θ ←θ 6:

Most of the details arise in step 1) of Algorithm 1.

Similar to the AlphaGo Zero algorithm [18] , the proposed RL+MCTS algorithm uses Monte Carlo Tree Search (MCTS) as a policy improvement operator to iteratively improve the policy network in the training phase.

For a randomly generated pair (y, x 0 ), we use MCTS and the current f θ to generate new training samples to feed back into the neural network.

We note that in the testing phase, MCTS can also be combined with the policy/value network to further boost the performance.

Specifically, for each given observation vector y and the desired sparsity k, we run MCTS simulations multiple times to construct a search tree [21; 22; 23].

When training the proposed RL+MCTS algorithm, we employ the following technique for reducing the training complexity.

First, we remark that using MCTS as a policy improvement operator can potentially be computationally expensive for relatively large matrix A (depending on the available computation resources).

To address this challenge, we fix the maximum depth d of the MCTS tree; that is, we build the MCTS tree until we reach a depth of d. From then on, we roll-out the remaining levels of the tree by simply using the OMP rule to select all remaining columns until a total of k columns are chosen.

This technique will be evaluated in the experiments in the next section.

In this section, we present experimental results for evaluating our proposed RL+MCTS algorithm and comparing it against two baseline methods: (i) OMP and (ii) BP (i.e., 1 minimization).

We first present results on the proposed RL+MCTS algorithm without limiting the tree depth.

In this setting, we will be training and testing on matrices of size 7 × 15 and 15 × 50.

The training parameters we use in our experiment is given in Table 2 We next show the results using the RL+MCTS algorithm with reduced complexity as described in Section 3.3.

Specifically, in a single MCTS search, we expand the tree to depth d, and then proceed to follow the OMP rule until a terminal state is reached.

We now show the experiment results for this version of the RL+MCTS algorithm.

Specifically, we consider the 10 × 100 matrix in our evaluation.

The training details of this experiment can be found in Table 2 in Appendix A.

We train two models.

A) We train a policy value network using the vanilla RL+MCTS algorithm without tree depth constraint.

We train a policy value network by limiting the tree depth d = 6, which leads to a 40% reduction in training time per sample.

Next, we first test the policy/value network trained from A) above.

This policy/value network will select each column without MCTS.

We then test the policy/value network trained from B) above: First, we test the policy/value network to pick the first column; For all subsequent columns up to k, we invoke the OMP rule.

This is equivalent to setting the tree depth during testing to d = 2 and with no MCTS (M = 0).

Using the same policy/value network, we also conduct an experiment where d = 6 and MCTS simulations is set to 1500 during testing.

From Figure 1 (c), note that the vanilla RL+MCTS policy π θ (a|s) still performs slightly better than both OMP and BP.

We see that training the RL+MCTS algorithm with a fixed tree depth gives us favorable results versus OMP, vanilla RL+MCTS policy π θ (a|s), and BP.

Average Prediction Times In Table 1 , we give the average prediction times per signal in seconds.

For OMP and BP, we use python libraries sklearn and cvx respectively.

To illustrate the speed during testing, we measure the prediction times on a much less powerful machine than what was used during training.

While training was accomplished on a i7 4790 (3.6 GHz) with a single GTX 780, the testing speeds in Table 2 were conducted on a Macbook Air with an Intel i5 clocked at 1.4 GHz and an integrated Intel HD 5000.

We predict that the testing speeds can be greatly improved with a more powerful machine and further optimization in the source code.

In general, we see that using just the policy/value network for prediction is in general slower than OMP, but on par with or better than BP.

We have shown that the proposed RL+MCTS algorithm is a highly effective sparse signal decoder for the compressed sensing problem assuming no signal structure other than sparsity.

Even without using MCTS in testing, the RL+MCTS algorithm's performance exceeds that of existing sparse signal recovery algorithms such as OMP and BP.

The flexibility in the RL+MCTS algorithm's design further offers many interesting avenues for future research.

For one, it is possible that the features chosen in our model can be further improved.

Secondly, since the true signal x 0 is known in training, one may be able to leverage the information about x 0 to increase training sample efficiency.

The training hyper-parameters may also be further tuned to improve performance.

Broader settings of problems such as noisy observations and varying observation matrices A are under active investigation.

In this appendix, we include the hyper-parameters of our experiments -see Table 2 .

@highlight

Formulating sparse signal recovery as a sequential decision making problem, we develop a method based on RL and MCTS that learns a policy to discover the support of the sparse signal. 