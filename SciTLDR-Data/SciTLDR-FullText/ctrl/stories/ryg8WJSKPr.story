Delusional bias is a fundamental source of error in approximate Q-learning.

To date, the only techniques that explicitly address delusion require comprehensive search using tabular value estimates.

In this paper, we develop efficient methods to mitigate delusional bias by training Q-approximators with labels that are "consistent" with the underlying greedy policy class.

We introduce a simple penalization scheme that encourages Q-labels used across training batches to remain (jointly) consistent with the expressible policy class.

We also propose a search framework that allows multiple Q-approximators to be generated and tracked, thus mitigating the effect of premature (implicit) policy commitments.

Experimental results demonstrate that these methods can improve the performance of Q-learning in a variety of Atari games, sometimes dramatically.

Q-learning (Watkins & Dayan, 1992; Sutton & Barto, 2018) lies at the heart of many of the recent successes of deep reinforcement learning (RL) (Mnih et al., 2015; , with recent advancements (e.g., van Hasselt (2010); Bellemare et al. (2017) ; Wang et al. (2016) ; Hessel et al. (2017) ) helping to make it among the most widely used methods in applied RL.

Despite these successes, many properties of Q-learning are poorly understood, and it is challenging to successfully apply deep Q-learning in practice.

When combined with function approximation, Q-learning can become unstable (Baird, 1995; Boyan & Moore, 1995; Tsitsiklis & Roy, 1996; Sutton & Barto, 2018) .

Various modifications have been proposed to improve convergence or approximation error (Gordon, 1995; 1999; Szepesvári & Smart, 2004; Melo & Ribeiro, 2007; Maei et al., 2010; Munos et al., 2016) ; but it remains difficult to reliably attain both robustness and scalability.

Recently, Lu et al. (2018) identified a source of error in Q-learning with function approximation known as delusional bias.

It arises because Q-learning updates the value of state-action pairs using estimates of (sampled) successor-state values that can be mutually inconsistent given the policy class induced by the approximator.

This can result in unbounded approximation error, divergence, policy cycling, and other undesirable behavior.

To handle delusion, the authors propose a policy-consistent backup operator that maintains multiple Q-value estimates organized into information sets.

Each information set has its own backed-up Q-values and corresponding "policy commitments" responsible for inducing these values.

Systematic management of these sets ensures that only consistent choices of maximizing actions are used to update Q-values.

All potential solutions are tracked to prevent premature convergence on any specific policy commitments.

Unfortunately, the proposed algorithms use tabular representations of Q-functions, so while this establishes foundations for delusional bias, the function approximator is used neither for generalization nor to manage the size of the state/action space.

Consequently, this approach is not scalable to RL problems of practical size.

In this work, we propose CONQUR (CONsistent Q-Update Regression), a general framework for integrating policy-consistent backups with regression-based function approximation for Q-learning and for managing the search through the space of possible regressors (i.e., information sets).

With suitable search heuristics, our framework provides a computationally effective means for minimizing the effects of delusional bias in Q-learning, while admitting scaling to practical problems.

Our main contributions are as follows.

First we define novel augmentations of standard Q-regression to increase the degree of policy consistency across training batches.

While testing exact consistency is expensive, we introduce an efficient soft-consistency penalty that promotes consistency of new labels with earlier policy commitments.

Second, drawing on the information-set structure of Lu et al. (2018) , we define a search space over Q-regressors to allow consideration of multiple sets of policy commitments.

Third, we introduce heuristics for guiding the search over regressors, which is critical given the combinatorial nature of information sets.

Finally, we provide experimental results on the Atari suite (Bellemare et al., 2013) demonstrating that CONQUR can offer (sometimes dramatic) improvements over Q-learning.

We also show that (easy-to-implement) consistency penalization on its own (i.e., without search) can improve over both standard and double Q-learning.

We assume a discounted, infinite horizon Markov decision process (MDP), M = (S, A, P, p 0 , R, γ).

The state space S can reflect both discrete and continuous features, but we take the action space A to be finite (and practically enumerable).

We consider Q-learning with a function approximator Q θ to learn an (approximately) optimal Q-function (Watkins, 1989; Sutton & Barto, 2018) , drawn from some approximation class parameterized by Θ (e.g., the weights of a neural network).

When the approximator is a deep network, we generically refer to the algorithm as DQN, the method at the heart of many recent RL successes (Mnih et al., 2015; .

For online Q-learning, at a transition s, a, r, s , the Q-update is given by:

Batch versions of Q-learning, including DQN, are similar, but fit a regressor repeatedly to batches of training examples (Ernst et al., 2005; Riedmiller, 2005) .

Batch methods are usually more data efficient and stable than online Q-learning.

Abstractly, batch Q-learning works through a sequence of (possibly randomized) data batches D 1 , · · · D T to produce a sequence of regressors Q θ1 , . . .

, Q θ T = Q θ , estimating the Q-function.

1 For each (s, a, r, s ) ∈ D k , we use a prior estimator Q θ k−1 to bootstrap the Q-label q = r + γ max a Q θ k−1 (s , a ).

We then fit Q θ k to this training data using a suitable regression procedure with an appropriate loss function.

Once trained, the (implicit) induced policy π θ is the greedy policy w.r.t.

Q θ , i.e., π θ (s) = arg max a∈A Q θ (s, a).

Let F(Θ), resp.

G(Θ), be the corresponding class of expressible Q-functions, resp.

greedy policies.

Intuitively, delusional bias occurs whenever a backed-up value estimate is derived from action choices that are not (jointly) realizable in G(Θ) (Lu et al., 2018) .

Standard Q-updates back up values for each (s, a) pair by independently choosing maximizing actions at the corresponding next states s .

However, such updates may be "inconsistent" under approximation: if no policy in G(Θ) can jointly express all past action choices, backed up values may not be realizable by any expressible policy.

Lu et al. (2018) show that delusion can manifest itself with several undesirable consequences.

Most critically, it can prevent Q-learning from learning the optimal representable policy in G(Θ); it can also cause divergence.

To address this, they propose a non-delusional policy consistent Q-learning (PCQL) algorithm that provably eliminates delusion.

We refer to the original paper for details, but review the main concepts we need to consider below.

The first key concept is that of policy consistency.

For any S ⊆ S, an action assignment σ S : S → A associates an action σ(s) with each s ∈ S.

We say σ is policy consistent if there is a greedy policy π ∈ G(Θ) s.t.

π(s) = σ(s) for all s ∈ S. We sometimes equate a set SA of state-action pairs with an implied assignment π(s) = a for all (s, a) ∈ SA.

If SA contains multiple pairs with the same state s, but different actions a, it is a multi-assignment (though we loosely use the term "assignment" in both cases when there is no risk of confusion).

In (batch) Q-learning, each successive regressor uses training labels generated by assuming maximizing actions (under the prior regressor) are taken at its successor states.

Let σ k reflect the collection of states and corresponding maximizing actions taken to generate labels for regressor Q θ k (assume it is policy consistent).

Suppose we train Q θ k by bootstrapping on Q θ k−1 and consider a training sample (s, a, r, s ).

Q-learning generates label r + γ max a Q θ k−1 (s , a ) for input (s, a).

Notice, however, that taking action a * = argmax a Q θ k (s , a ) at s may not be policy consistent with σ k .

Thus Q-learning will estimate a value for (s, a) assuming the execution of a policy that cannot be realized given the limitations of the approximator.

The PCQL algorithm (Lu et al., 2018) prevents this by insisting that any action assignment σ used to generate bootstrapped labels is consistent with earlier assignments.

Notice that this means Q-labels will often not be generated using maximizing actions relative to the prior regressor.

The second key concept is that of information sets.

One will generally not be able to use maximizing actions to generate labels, so tradeoffs can be made when deciding which actions to assign to different states.

Indeed, even if it is feasible to assign a maximizing action a to state s early in training, say at batch k, since it may prevent assigning a maximizing a to s later, say batch k + , we may want to consider a different assignment to s to give more flexibility to maximize at other states later.

PCQL doesn't try to anticipate the tradeoffs-rather it maintains multiple information sets, each corresponding to a different assignment to the states seen in the training data so far.

Each gives rise to a different Q-function estimate, resulting in multiple hypotheses.

At the end of training, the best hypothesis is the one maximizing expected value w.r.t.

an initial state distribution.

PCQL provides strong convergence guarantees, but it is a tabular algorithm: the function approximator retricts the policy class, but is not used to generalize Q-values.

Furthermore, its theoretical guarantees come at a cost: it uses exact policy consistency tests-tractable for linear approximators, but not practical for large problems; and it maintains all consistent assignments.

As a result, PCQL cannot be used for large RL problems of the type tackled by DQN.

We develop the CONQUR framework to provide a practical approach to reducing delusion in Qlearning, specifically addressing the limitations of PCQL identified above.

CONQUR consists of three main components: a practical soft-constraint penalty that promotes policy consistency; a search space to structure the search over multiple regressors (information sets, action assignments); and heuristic search schemes (expansion, scoring) to find good Q-regressors.

We assume a set of training data consisting of quadruples (s, a, r, s ), divided into (possibly nondisjoint) batches D 1 , . . .

D T for training.

This perspective is quite general: online RL corresponds to |D i | = 1; off-line batch training (with sufficiently exploratory data) corresponds to a single batch (i.e., T = 1); and online or batch methods with replay are realized when the D i are generated by sampling some data source with replacement.

For any data batch D, let χ(D) = {s : (s, a, r, s ) ∈ D} denote the collection of successor states of D. An action assignment σ D for D is an assignment (or multi-assignment) from χ(D) to A: this dictates which action σ D (s ) is considered "maximum" for the purpose of generating a Q-label for pair (s, a); i.e., (s, a) will be assigned training label r + γQ(s , σ(s )) rather than r + γ max a ∈A Q(s , a ).

The set of all such assignments is Σ(D) = A χ(D) ; note that it grows exponentially with |D|.

2 This is simple policy consistency, but with notation that emphasizes the policy class.

Let Σ Θ (D) denote the set of all Θ-consistent assignments over D. The union σ 1 ∪ σ 2 of two assignments (over D 1 , D 2 , resp.) is defined in the usual way.

Enforcing strict Θ-consistency as regressors θ 1 , θ 2 , . . .

, θ T are generated is computationally challenging.

Suppose assignments σ 1 , . . .

, σ k−1 , used to generate labels for D 1 , . . .

D k−1 , are jointly Θ-consistent (let σ ≤k−1 denote their multi-set union).

Maintaining Θ-consistency when generating θ k imposes two requirements.

First, one must generate an assignment σ k over D k s.t.

σ ≤k−1 ∪ σ k is consistent.

Even testing assignment consistency can be problematic: for linear approximators this is a linear feasibility program (Lu et al., 2018) whose constraint set grows linearly with |D 1 ∪ . . .

∪ D k |.

For DNNs, this is a complex, and much more expensive, polynomial program.

Second, the regressor θ k should itself be consistent with σ ≤k−1 ∪ σ k .

Again, this imposes a significant constraint on the regression optimization: in the linear case, this becomes a constrained least-squares problem (solvable, e.g., as a quadratic program); while with DNNs, it could be solved, say, using a much more complicated projected SGD.

However, the sheer number of constraints makes this impractical.

Rather than enforcing consistency, we propose a simple, computationally tractable scheme that "encourages" it: a penalty term that can be incorporated into the regression itself.

Specifically, we add a penalty function to the usual squared loss to encourage updates of the Q-regressors to be consistent with the underlying information set, i.e., the prior action assignments used to generate its labels.

When constructing θ k , let D ≤k = ∪{D j : j ≤ k}, and σ ∈ Σ Θ (D ≤k ) be the collective (possibly multi-) assignment used to generate labels for all prior regressors (including θ k itself).

The multiset of pairs B = {(s , σ(s ))|s ∈ χ(D ≤k )}, is called a consistency buffer.

The collective assignment need not be consistent (as we elaborate below), nor does the regressor θ k need to be consistent with σ.

Instead, we incorporate the following soft consistency penalty when constructing θ k :

where

.

This penalizes Q-values of actions at state s that are larger than that of action σ(s).

We note that σ is Θ-consistent if and only if min θ∈Θ C θ (B) = 0.

We incorporate this penalty into our regression loss for batch D k :

Here Q θ k is prior estimator on which labels are bootstrapped (other prior regressors may be used).

The penalty effectively acts as a "regularizer" on the squared Bellman error, where λ controls the degree of penalization, allowing a tradeoff between Bellman error and consistency with the action assignment used to generate labels.

It thus promotes consistency without incurring the expense of testing strict consistency.

It is a simple matter to replace the classical Q-learning update (1) with one using a consistency penalty:

This scheme is quite general.

First, it is agnostic as to how the prior action assignments are made, which can be the standard maximizing action at each stage w.r.t.

the prior regressor like in DQN, Double DQN (DDQN) , or other variants.

It can also be used in conjunction with a search through alternate assignments (see below).

Second, the consistency buffer B may be populated in a variety of ways.

Including all max-action choices from all past training batches promotes full consistency in an attempt to minimize delusion.

However, this may be too constraining since action choices early in training are generally informed by very inaccurate value estimates.

Hence, B may be implemented in other ways to focus only on more recent data (e.g., with a sliding recency window, weight decay, or subsampling); and the degree of recency bias may adapt during training (e.g., becoming more inclusive as training proceeds and the Q-function approaches convergence).

Reducing the size of B also has various computational benefits.

We discuss other practical means of promoting consistency in Sec. 5.

The proposed consistency penalty resembles the temporal-consistency loss of Pohlen et al. (2018) , but our aims are very different.

Their temporal consistency notion penalizes changes in a next state's Q-estimate over all actions, whereas we discourage inconsistencies in the greedy policy induced by the Q-estimator, regardless of the actual estimated values.

Ensuring optimality requires that PCQL track all Θ-consistent assignments.

While the set of such assignments is shown to be of polynomial size (Lu et al., 2018) , it is still impractical to track this set in realistic problems.

As such, in CONQUR we recast information set tracking as a search problem and propose several strategies for managing the search process.

We begin by defining the search space and discussing its properties.

We discuss search procedures in Sec. 3.4.

As above, assume training data is divided into batches D 1 , . . .

D T and we have some initial Qfunction estimate θ 0 (for bootstrapping D 1 's labels).

The regressor θ k for D k can, in principle, be trained with labels generated by any assignment σ ∈ Σ Θ (D k ) of actions to its successor states χ(D k ), not necessarily maximizing actions w.r.t.

θ k−1 .

Each σ gives rise to a different updated Q-estimator θ k .

There are several restrictions we could place on "reasonable" σ-candidates:

(ii) σ is jointly Θ-consistent with all σ j , for j < k, used to construct the prior regressors on which

, and this inequality is strict for at least one s .

Conditions (i) and (ii) are the strict consistency requirements of PCQL.

We will, however, relax these below for reasons discussed in Sec. 3.2.

Condition (iii) is inappropriate in general, since we may add additional assignments (e.g., to new data) that render all non-dominated assignments inconsistent, requiring that we revert to some dominated assignment.

This gives us a generic search space for finding policy-consistent, delusion-free Q-function, as illustrated in Fig k can also be viewed as an information set).

We assume the root n 0 is based on an initial regression θ 0 , and has an empty action assignment σ 0 .

Nodes at level k of the tree are defined as follows.

For each node n

, and its regressor θ i k is trained using the following data set:

The entire search space constructed in this fashion to a maximum depth of T .

See Appendix B, Algorithm 1 for pseudocode of a simple depth-first recursive specification.

The exponential branching factor in this search tree would appear to make complete search intractable; however, since we only allow Θ-consistent "collective" assignments we can bound the size of the tree-it is polynomial in the VC-dimension of the approximator.

Theorem 1.

The number of nodes in the search tree is no more than

VCDim(G) ) where VCDim(·) is the VC-dimension (Vapnik, 1998 ) of a set of boolean-valued functions, and G is the set of boolean functions defining all feasible greedy policies under Θ:

A linear approximator with a fixed set of d features induces a policy-indicator function class G with VC-dimension d, making the search tree polynomial in the size of the MDP.

Similarly, a fixed ReLU DNN architecture with W weights and L layers has VC-dimension of size O(W L log W ) again rendering the tree polynomially sized.

Even with this bound, navigating the search space exhaustively is generally impractical.

Instead, various search methods can be used to explore the space, with the aim of reaching a "high quality" regressor at some leaf of the tree (i.e., trained using all T data sets/batches).

We discuss several key considerations in the next subsection.

Even with the bound in Theorem 1, traversing the search space exhaustively is generally impractical.

Moreover, as discussed above, enforcing consistency when generating the children of a node, and their regressors, may be intractable.

Instead, various search methods can be used to explore the space, with the aim of reaching a "high quality" regressor at some (depth T ) leaf of the tree.

We outline three primary considerations in the search process: child generation, node evaluation or scoring, and the search procedure.

Generating children.

Given node n i k−1 , there are, in principle, exponentially many action assignments, or children, Σ Θ (D k ) (though Theorem 1 significantly limits the number of children if we enforce consistency).

For this reason, we consider heuristics for generating a small set of children.

Three primary factors drive these heuristics.

The first factor is a preference for generating high-value assignments.

To accurately reflect the intent of (sampled) Bellman backups, we prefer to assign actions to state s ∈ χ(D k ) with larger predicted Q-values over actions with lower values, i.e., a preference for a over a if Q θ

(s , a ).

However, since the maximizing assignment may be Θ-inconsistent (in isolation, or jointly with the parent's information set, or with future assignments), candidate children should merely have higher probability of a high-value assignment.

The second factor is the need to ensure diversity in the assignments among the set of children.

Policy commitments at stage k constrain the possible assignments at subsequent stages.

In many search procedures (e.g., beam search), we avoid backtracking, so we want the policy commitments we make at stage k to offer as much flexibility as possible in later stages.

The third is the degree to which we enforce consistency.

There are several ways to generate such high-value assignments.

We focus on just one natural technique: sampling action assignments using a Boltzmann distribution.

Specifically, let σ denote the assignment (information set) of some node (parent) at level k − 1 in the tree.

We can generate an assignment σ k for D k as follows.

Assume some permutation s 1 , . . .

, s |D k | of χ(D k ).

For each s i in turn, we sample a i with probability proportional to e τ Q θ k−1 (s i ,ai) .

This can be done without regard to consistency, in which case we would generally use the consistency penalty when constructing the regressor θ k for this child to "encourage" consistency rather than enforce it.

If we want strict consistency, we can use rejection sampling without replacement to ensure a i is consistent with σ j k−1 ∪ σ ≤i−1 (we can also use a subset of σ j k−1 as a less restrictive consistency buffer).

3 The temperature parameter τ controls the degree to which we focus on purely maximizing assignments versus more diverse, random assignments.

While stochastic sampling ensures some diversity, this procedure will bias selection of high-value actions to states s ∈ χ(D k ) that occur early in the permutation.

To ensure sufficient diversity, we use a new random permutation for each child.

Scoring children.

Once the children of some expanded node are generated (and, optionally, their regressors constructed), we need some way of evaluating the quality of each child as a means of deciding which new nodes are most promising for expansion.

Several techniques can be used.

We could use the average Q-label (overall, or weighted using some initial state distribution), Bellman error, or loss incurred by the regressor (including the consistency penalty or other regularizer).

However, care must be taken when comparing nodes at different depths of the search tree, since deeper nodes have a greater chance to accrue rewards or costs-simple calibration methods can be used.

Alternatively, when a simulator is available, rollouts of the induced greedy policy can be used evaluate the quality of a node/regressor.

Notice that using rollouts in this fashion incurs considerable computational expense during training relative to more direct scoring based on properties on the node, regressor, or information set.

Search Procedure.

Given any particular way of generating/sampling and scoring children, a variety of different search procedures can be applied: best-first search, beam search, local search, etc.

all fit very naturally within the CONQUR framework.

Moreover, hybrid strategies are possible-one we develop below is a variant of beam search in which we generate multiple children only at certain levels of the tree, then do "deep dives" using consistency-penalized Q-regression at the intervening levels.

This reduces the size of the search tree considerably and, when managed properly, adds only a constant-factor (proportional to beam size) slowdown to standard Q-learning methods like DQN.

We now outline a specific instantiation of the CONQUR framework that can effectively navigate the large search space that arises in practical RL settings.

We describe a heuristic, modified beamsearch strategy with backtracking and priority scoring.

Pseudocode is provided in Algorithm 2 (see Appendix B); here we simply outline some of the key refinements.

Our search process grows the tree in a breadth-first manner, and alternates between two phases.

In an expansion phase, parent nodes are expanded, generating one or more child nodes with action assignments sampled from the Boltzmann distribution.

For each child, we create target Q-labels, then optimize the child's regressor using consistency-penalized Bellman error (Eq. 2) as our loss.

We thus forego strict policy consistency, and instead "encourage" consistency in regression.

In a dive phase, each parent generates one child, whose action assignment is given by the usual max-actions selected by the parent node's regressor as in standard Q-learning.

No additional diversity is considered in the dive phase, but consistency is promoted using consistency-penalized regression.

From the root, the search begins with an expansion phase to create c children-c is the splitting factor.

Each child inherits its parent's consistency buffer from which we add the new action assignments that were used to generate that child's Q-labels.

To limit the size of the tree, we only track a subset of the children, the frontier nodes, selected using one of several possible scoring functions.

We select the top -nodes for expansion, proceed to a dive phase and iterate.

It is possible to move beyond this "beam-like" approach and consider backtracking strategies that will return to unexpanded nodes at shallower depths of the tree.

We consider this below as well.

Other work has considered multiple hypothesis tracking in RL.

One particularly direct approach has been to use ensembling, where multiple Q-approximators are updated in parallel (Faußer & Schwenker, 2015; Osband et al., 2016; Anschel et al., 2017) then combined straightforwardly to reduce instability and variance.

An alternative approach has been to consider population-based methods inspired by evolutionary search.

For example, Conti et al. (2018) combine a novelty-search and quality diversity technique to improve hypothesis diversity and quality in RL.

Khadka & Tumer (2018) consider augmenting an off-policy RL method with diversified population information from an evolutionary algorithm.

Although these techniques do offer some benefit, they do not systematically target an identified weakness of Q-learning, such as delusion.

We experiment using the Atari test suite (Bellemare et al., 2013) to assess the performance of CONQUR.

We first assess the impact of using the consistency penalty in isolation (without search) as a "regularizer" that promotes consistency with both DQN and DDQN.

We then test the modified beam search described in Appendix B to assess the full power of CONQUR.

We first study the effects of introducing the soft-policy consistency in isolation, augmenting both DQN and DDQN with the consistency penalty term.

We train our models using an open-source implementation (Guadarrama et al., 2018) of both DQN and DDQN (with the same hyperparameters).

We call these modified algorithms DQN(λ) and DDQN(λ), respectively, where λ is the penalty coefficient defined in Eq. 2.

Note that λ = 0 recovers the original methods.

This is a lightweight modification that can be applied readily to any regression-based Q-learning method, and serves to demonstrate the effectiveness of soft-policy consistency penalty.

Since we don't consider search (i.e., don't track multiple hypotheses), we maintain a small consistency buffer using only the current data batch by sampling from the replay buffer-this prevents getting "trapped" by premature policy constraints.

As the action assignment is the maximizing action of some network, σ(s ) can be computed easily for each batch.

This results in a simple algorithmic extension that adds only an additional penalty term to the original TD-loss.

We train and evaluate DQN(λ) and DDQN(λ) for λ = {0.25, 0.5, 1, 1.5, 2} on 19 Atari games.

In training, λ is initialized to 0 and slowly annealed to the desired value to avoid premature commitment to poor action assignments.

Without annealing, the model tends fit to poorly informed action assignments during early phases of training, and thus fails to learn a good model.

The best λ is generally different across games, depending on the nature of the game and the extent of delusional bias.

Though a single λ = 0.5 works well across all games tested, Fig. 2 illustrates the effect of increasing λ on two games.

In Gravitar, increasing λ generally results in better performance for both DQN and DDQN, whereas in SpaceInvaders, λ = 0.5 gives improvement over both baselines, but performance starts to degrade for λ = 2.

We compare the performance of the algorithms using each λ value separately, as well as using the best λ for each game.

Under the best λ, DQN(λ) and DDQN(λ) outperform their "potentially delusional" counterparts on all except 3 and 2 games, respectively.

In 9 of these games, each of DQN(λ) and DDQN(λ) beats both baselines.

With a constant λ = 0.5, each algorithm still beats their respective baseline in 11 games.

These results suggest that consistency penalization (independent of the general CONQUR model) can improve the performance of DQN and DDQN by addressing the delusional bias that is critical to learning a good policy.

Moreover, we see that consistency penalization seems to have a different effect on learned Q-values than double Q-learning, which addresses maximization bias.

Indeed, consistency penalization, when applied to DQN, can achieve gains that are greater than DDQN (in 15 games).

Third, in 9 games DDQN(λ) provides additional performance gains over DQN(λ).

A detailed description of the experiments and further results can be found in Appendix C. Table 1 : Results of CONQUR with 8 (split 2) nodes on 9 games using the proposed scoring function compared to evaluation using rollouts.

We test the full CONQUR framework using the modified beam search discussed above.

Rather than training a full Q-network, for effective testing of its core principles, we leverage pre-trained networks from the Dopamine package .

4 .

These networks have the same architecture as in Mnih et al. (2015) and are trained on 200M frames with sticky actions using DQN.

We use CONQUR to retrain only the last (fully connected) layer (implicitly freezing the other layers), which can be viewed as a linear Q-approximator over the features learned by the CNN.

We run CONQUR using only 4M addtional frames to train our Q-regressors.

5 We consider splitting factors c of 2 and 4; impose a limit on the frontier size of 8 or 16; and an expansion factor of 2 or 4.

The dive phase is always of length 9 (i.e., 9 batches of data), giving an expansion phase every 10 iterations.

Regressors are trained using the loss in Eq. 2 and the consistency buffer comprises all prior action assignments.

(See Appendix D for details, hyperparameter choices and more results.)

We run CONQUR with λ = {1, 10, 100, 1000} and select the best performing policy.

We initially test two scoring approaches, policy evaluation using rollouts and scoring using the loss function (Bellman error with consistency penalty).

Results comparing the two on a small selection of games are shown in Table 1 .

While rollouts, not surprisingly, tend to give rise to better-performing policies, consistent-Bellman scoring is competitive.

Since the latter much less computationally intense, and does not require sampling the environment, we use it throughout our remaining experiments.

We compare CONQUR with the value of the pre-trained DQN.

We also evaluate a "multi-DQN" baseline that applies multiple versions of DQN independently, warm-starting from the same pretrained DQN.

It uses the same number of frontier nodes as CONQUR, and is otherwise trained identically as CONQUR but with direct Bellman error (no consistency penalty).

This gives DQN the same advantage of multiple-hypothesis tracking as CONQUR but without policy consistency.

We test on 59 games, comparing CONQUR with frontier size 16 and expansion factor 4 and splitting factor 4 (16-4-4) with backtracking (as described in the Appendix D) resulted in significant improvements to the pre-trained DQN, with an average score improvement of 125% (excluding games with non-positive pre-trained score).

The only games without improvement are Montezuma's Revenge, Tennis, PrivateEye and BankHeist.

This demonstrates that, even when simply retraining the last layer of a highly tuned DQN network, removing delusional bias has the potential to offer strong improvements in policy performance.

It is able exploit the reduced parameterization to obtain these gains with only 4M frames of training data.

Roughly, a half-dozen games have outsized score improvements, including Solaris (11 times greater value), Tutankham (6.5 times) and WizardOfWor (5 times).

Compared to the stronger multi-DQN baseline (with 16 nodes), CONQUR wins by at least a 10% margin in 20 games, while 22 games see improvements of 1-10% and 8 games show little effect (plus/minus 1%) and 7 games show a decline of greater than 1% (most are 1-6% with the exception of Centipede at -12% and IceHockey at -86%).

Results are similar when comparing CONQUR and multi-DQN each with 8 nodes (8-2-2): 9 games exhibit 10%+ improvement, 21 games show 1-8% improvement, 12 games perform comparably and 7 games do worse under CONQUR.

A table of complete results appears in Appendix D.3, Table 4 , and training curves (all games, all λ) in Fig. 11 .

Increasing the number of nodes from 8 to 16 generally leads to better performance for CONQUR, with 38 games achieving strictly higher scores with 16 nodes (16-4-4): 16 games with 10%+ improvement, 5 games tied and the remaining 16 games performing worse (only a few with a 5%+ decline).

Fig. 3 shows the (smoothed) effect of increasing the number of nodes for a fixed λ = 10.

The y-axis represents the rollout value of the best frontier node (i.e., the greedy policy of its Q-regressor) as a function of the training iteration.

For both Alien and Solaris, the multi-DQN (baseline) training curve is similar with both 8 and 16 nodes, but CONQUR improves Alien from 3k to 4.3k while Solaris improves from 2.2k to 3.5k.

Fig. 4 and Fig. 5 (smoothed, best frontier node) shows node policy values and training curves, respectively, for Solaris.

When considering nodes ranked by their policy value, comparing nodes of equal rank generated by CONQUR and by multi-DQN (baseline), we see that CONQUR nodes dominate their multi-DQN counterparts: the three highest-ranked nodes achieve a score improvement of 18%, 13% and 15%, respectively, while the remaining nodes achieve improvements of roughly 11-12%.

Fig. 6 (smoothed, best frontier node) shows the effects of varying λ.

In Alien, increasing λ from 1 to 10 improves performance, but it starts to decline for higher values of 100 and 1000.

This is similar to patterns observed in 4.1 and represents a trade-off between emphasizing consistency and not over-committing to action assignments.

In Atlantis, stronger penalization tends to degrade performance.

In fact, the stronger the penalization, the worse the performance.

We have introduced CONQUR, a framework for mitigating delusional bias in value-based RL that relaxes some of the strict assumptions of exact delusion-free algorithms to ensure scalability.

Its two main components are (a) a tree-search procedure used to create and maintain diverse, promising Q-regressors (and corresponding information sets); and (b) a consistency penalty that encourages "maximizing" actions to be consistent with the FA class.

CONQUR embodies elements of both value-based and policy-based RL: it can be viewed as using partial policy constraints to bias the value estimator or as a means of using candidate value functions to bias the search through policy space.

Empirically, we find that CONQUR can improve the quality of existing approximators by removing delusional bias.

Moreover, the consistency penalty applied on its own, directly in DQN or DDQN, itself can improve the quality of the induced policies.

Given the generality of the CONQUR framework, there remain numerous interesting directions for future research.

Other methods for nudging regressors to be policy-consistent include exact consistency (constrained regression), other regularization schemes that bias the regressor to fall within the information set, etc.

Given its flexibility, more extensive exploration of search strategies (e.g., best first), child-generation strategies, and node scoring schemes should be examined within CONQUR.

Our (full) experiments should also be extended beyond those that warm-start from a DQN model, as should testing CONQUR in other domains.

Other connections and generalizations are of interest as well.

We believe our methods can be extended to both continuous actions and soft max-action policies.

We suspect that there is a connection between maintaining multiple "hypotheses" (i.e., Q-regressors) and notions in distributional RL, which maintains distributions over action values Bellemare et al. (2017) .

We describe an example, taken directly from (Lu et al., 2018) , to show concretely how delusional bias causes problems for Q-learning with function approximation.

The MDP in Fig. 7 illustrates the phenomenon: Lu et al. (2018) use a linear approximator over a specific set of features in this MDP to show that:

(a) No π ∈ G(Θ) can express the optimal (unconstrained) policy (which requires taking a 2 at each state); (b) The optimal feasible policy in G(Θ) takes a 1 at s 1 and a 2 at s 4 (achieving a value of 0.5).

(c) Online Q-learning (Eq. 1) with data generated using an ε-greedy behavior policy must converge to a fixed point (under a range of rewards and discounts) corresponding to a "compromise" admissible policy which takes a 1 at both s 1 and s 4 (value of 0.3).

Algorithm 1 CONQUR SEARCH (Generic, depth-first)

Training set S ← {} 4:

end for end if 14: end for Q-learning fails to find a reasonable fixed-point because of delusion.

Consider the backups at (s 2 , a 2 ) and (s 3 , a 2 ).

Supposeθ assigns a "high" value to (s 3 , a 2 ), so that Qθ(s 3 , a 2 ) > Qθ(s 3 , a 1 ) as required by π θ * .

They show that any suchθ also accords a "high" value to (s 2 , a 2 ).

But Qθ(s 2 , a 2 ) > Qθ(s 2 , a 1 ) is inconsistent the first requirement.

As such, any update that makes the Q-value of (s 2 , a 2 ) higher undercuts the justification for it to be higher (i.e., makes the "max" value of its successor state (s 3 , a 2 ) lower).

This occurs not due to approximation error, but the inability of Q-learning to find the value of the optimal representable policy.

The pseudocode of (depth-first) version of the CONQUR search framework is listed in Algorithm 1.

As discussed in Sec. 3.5, a more specific instantiation of the CONQUR algorithm is listed in Algorithm.

2.

Both DQN and DDQN uses a delayed version of the Q-network Q θ − (s , a ) for label generation, but in a different way.

In DQN, Q θ − (s , a ) is used for both value estimate and action assignment σ DQN (s ) = argmax a Q θ k (s , a ), whereas in DDQN, Q θ − (s , a ) is used only for value estimate and the action assignment is computed from the current network σ DDQN (s ) = argmax a Q θ k (s , a ).

With respect to delusional bias, action assignment of DQN is consistent for all batches after the latest network weight transfer, as σ DQN (s ) is computed from the same Q θ − (s , a ) network.

DDQN, on the other hand, could have very inconsistent assignments, since the action is computed from the current network that is being updated at every step.

Select top scoring nodes n 1 , ..., n ∈ P 10:

for each selected node n i do

Generate c children n i,1 , ..., n i,c using Boltzmann sampling on D k with Q θ i 12:

for each child n i,j do 13:

Let assignment history σ i,j be σ i ∪ {new assignment} 14:

Determine regressor θ i,j by applying update (3) if k is a refinement ("dive") level then

for each frontier node n i,j ∈ F do 25:

Update regressor θ i,j by applying update (3) to θ (Guadarrama et al., 2018) .

In particular, we modify existing DqnAgent and DdqnAgent by adding a consistency penalty term to the original TD loss.

We use TF-Agents implementation of DQN training on Atari with the default hyperparameters, which are mostly the same as that used in the original DQN paper (Mnih et al., 2015) .

For conveniece to the reader, some important hyperparameters are listed in Table 2 .

The reward is clipped between [−1, 1] following the original DQN.

We empirically evaluate our modified DQN and DDQN agents trained with consistency penalty on 15 Atari games.

Evaluation is run using the training and evaluation framework for Atari provided in TF-Agents without any modifications.

C.4 DETAILED RESULTS Fig. 8 shows the effects of varying λ on both DQN and DDQN.

Table 3 summarizes the best penalties for each game and their corresponding scores.

Fig. 9 shows the training curves of the best penalization constants.

Finally, Fig. 10 shows the training curves for a fixed penalization of λ = 1/2.

steps, and within each window, we take the largest policy value (and over ≈2-5 multiple runs).

This is done to reduce visual clutter.

Our results use a frontier queue of size (F ) 8 or 16 (these are the top scoring leaf nodes which receive gradient updates and rollout evaluations during training).

To generate training batches, we select the best node's regressor according to our scoring function, from which we generate training samples (transitions) using ε-greedy.

Results are reported in Table 4 and 5, and related figures where max number of nodes are 8 or 16.

We used Bellman error plus consistency penalty as our scoring function.

During the training process, we also calibrated the scoring to account for the depth difference between the leaf nodes at the frontier versus the leaf nodes in the candidate pool.

We calibrated by taking the mean of the difference between scores of the current nodes in the frontier with their parents.

We scaled this difference by multiplying with a constant of 2.5.

In our implementation, we initialized our Q-network with a pre-trained DQN.

We start with the expansion phase.

During this phase, each parent node splits into l children nodes and the Q-labels are generated using action assignments from the Boltzmann sampling procedure, in order to create high quality and diversified children.

We start the dive phase until the number of children generated is at least F .

In particular, with F = 16 configuration, we performed the expansion phase at the zero-th and first iterations, and then at every tenth iteration starting at iteration 10, then at 20, and so on until ending at iteration 90.

In the F = 8 configuration, the expansion phase occurred at the zero-th and first iterations, then at every tenth iterations starting at iterations 10 and 11, then at iterations 20 and 21, and so on until ending at iterations 90 and 91.

All other iterations execute the "dive" phase.

For every fifth iteration, Q-labels are generated from action assignments sampled according to the Boltzmann distribution.

For all other iterations, Q-labels are generated in the same fashion as the standard Q-learning (taking the max Q-value).

The generated Q-labels along with the consistency penalty are then converted into gradient updates that applies to one or more generated children nodes.

Each iteration consists of 10k transitions sampled from the environment.

Our entire training process has 100 iterations which consumes 1M transitions or 4M frames.

We used RMSProp as the optimizer with a learning rate of 2.5 × 10 −6 .

One training iteration has 2.5k gradient updates and we used a batch size of 32.

We replace the target network with the online network every fifth iteration and reward is clipped between [−1, 1].

We use a discount value of γ = 0.99 and ε-greedy with ε = 0.01 for exploration.

Details of hyper-parameter settings can be found in Table 6 (for 16 nodes) and Table 7 (for 8 nodes).

We empirically evaluate our algorithms on 59 Atari games (Bellemare et al., 2013) , and followed the evaluation procedure as in .

We evaluate our agents every 10th iterations (and also the initial and first iteration) by suspending our training process.

We evaluate on 500k frames, and we cap the length of the episodes for 108k frames.

We used ε-greedy as the evaluation policy with ε = 0.001.

Fig. 11 shows training curves of CONQUR with 16 nodes under different penalization strengths λ.

Each plotted step of each training curve (including the baseline) shows the best performing node's policy value as evaluated with full rollouts.

Table 4 shows the summary of the highest policy values achieved for all 59 games for CONQUR and the baseline under 8 and 16 nodes.

Table 5 shows a similar summary, but without no-op starts (i.e. policy actions are applied immediately).

Both the baseline and CONQUR improve overall, but CONQUR's advantage over the baseline is amplified.

This may suggest that for more deterministic MDP environments, CONQUR may have even better improvements.

The results on 16 and 8 nodes use a splitting factor of 4 and 2, respectively.

Table 4 : Summary of scores with ε-greedy (ε = 0.001) evaluation with up to 30 no-op starts.

Mini-batch size Size of the mini batch data used to train the Qnetwork.

ε train ε-greedy policy for exploration during training.

0.01 ε eval ε-greedy policy for evaluating Q-regressors.

0.001 Training calibration parameter Calibration to adjust the difference between the nodes from the candidate pool m which didn't selected during both the expansion nor the dive phases.

The calibration is performed based on the average difference between the frontier nodes and their parents.

We denote this difference as .

Discount factor γ Discount factor during the training process.

0.99

<|TLDR|>

@highlight

We developed a search framework and consistency penalty to mitigate delusional bias.