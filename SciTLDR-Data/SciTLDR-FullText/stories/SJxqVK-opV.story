We extend the learning from demonstration paradigm by providing a method for learning unknown constraints shared across tasks, using demonstrations of the tasks, their cost functions, and knowledge of the system dynamics and control constraints.

Given safe demonstrations, our method uses hit-and-run sampling to obtain lower cost, and thus unsafe, trajectories.

Both safe and unsafe trajectories are used to obtain a consistent representation of the unsafe set via solving a mixed integer program.

Additionally, by leveraging a known parameterization of the constraint, we modify our method to learn parametric constraints in high dimensions.

We show that our method can learn a six-dimensional pose constraint for a 7-DOF robot arm.

Inverse optimal control and inverse reinforcement learning (IOC/IRL) BID5 can enable robots to perform complex goaldirected tasks by learning a cost function which replicates the behavior of an expert demonstrator when optimized.

However, planning for many robotics and automation tasks also requires knowing constraints, which define what states or trajectories are safe.

Existing methods learn local trajectory-based constraints BID3 BID4 or a cost penalty to approximate a constraint BID1 , neither of which extracts states that are guaranteed unsafe for all trajectories.

In contrast, recent work BID2 recovers a binary representation of globally-valid constraints from expert demonstrations by sampling lower cost (and hence constraintviolating) trajectories and then recovering a constraint consistent with the data by solving an integer program over a gridded constraint space.

The learned constraint can be then used to inform a planner to generate safe trajectories connecting novel start and goal states.

However, the gridding restricts the scalability of this method to higher dimensional constraints.

The contributions of this workshop paper are twofold:• By assuming a known parameterization of the constraint, we extend BID2 to higher dimensions by writing a mixed integer program over parameters which recovers a constraint consistent with the data.• We evaluate the method by learning a 6-dimensional pose constraint on a 7 degree-of-freedom (DOF) robot arm.

II.

PRELIMINARIES AND PROBLEM STATEMENT We consider a state-control demonstration (ξ * x .

= {x 0 , . . . , x T }, ξ * u .

= {u 0 , . . . , u T −1 }) which steers a controlconstrained system x t+1 = f (x t , u t , t), u t ∈ U for all t, from a start state x 0 to a goal state x T , while minimizing cost c(ξ x , ξ u ) and obeying safety constraints φ(ξ) .

DISPLAYFORM0 Formally, a demonstration solves the following problem 1 : DISPLAYFORM1 are known functions mapping (ξ x , ξ u ) to some constraint spaces C andC, where subsets S ⊆ C and S ⊆C are considered safe.

In particular,S is known and represents the set of all constraints known to the learner.

In this paper, we consider the problem of learning the unsafe set A .

DISPLAYFORM2 , each with different start and goal states.

We assume that the dynamics, control constraints, and start and goal constraints are known and are embedded inφ(ξ x , ξ u ) ∈S. We also assume the cost function c(·, ·) is known.

BID0 Details for continuous-time and suboptimal demonstrations are in BID2 .

DISPLAYFORM3

We review BID2 , which reduces the ill-posedness of the constraint learning problem by using the insight that each safe, optimal demonstration induces a set of lower-cost trajectories that must be unsafe.

These unsafe trajectories are sampled (Section III-A) and used with the demonstrations to reduce the number of consistent unsafe sets.

Then, an integer program is used to find a gridded representation of A consistent with both safe and unsafe trajectories (Section III-B).

We are interested in sampling from the set of lower-cost trajectories which are dynamically feasible, satisfy the control constraints, and have fixed start and goal state x 0 , x T :

DISPLAYFORM0 Each trajectory ξ ¬s ∈ A ξ is unsafe, since the optimal demonstrator would have provided any safe lower-cost trajectory, and thus at least one state in ξ ¬s belongs to A. We sample from A ξ using hit-and-run BID0 BID2 (see FIG0 , providing a uniform distribution of samples in the limit.

Furthermore, if the demonstrator is boundedly suboptimal and satisfies c(ξ DISPLAYFORM1

As the constraint is not assumed to have any parametric structure, the constraint space C is gridded into G cells z 1 , . . .

, z G , and we recover a safety value for each grid cell O(z i ) ∈ {0, 1} which is consistent with the N s safe and N ¬s sampled unsafe trajectories by solving the integer problem: Problem 2 (Grid-based constraint recovery problem).

DISPLAYFORM0 Here, O(z i ) = 1 if cell z i is considered unsafe, and 0 otherwise.

The first constraint restricts all cells that a demonstration passes through to be marked safe, while the second constraint restricts that for each unsafe trajectory, at least one grid cell it passes through is unsafe.

Furthermore, denote as G z ¬s the set of guaranteed learned unsafe cells.

One can check if cell z i ∈ G z ¬s by checking the feasibility of Problem 2 with an additional constraint that O(z i ) = 0 (forcing z i to be safe).

Suppose that the unsafe set can be described by some parameterization A(θ) .

= {k ∈ C | g(k, θ) ≤ 0}, where constraint state k is some element of C, g(·, ·) is known, and θ are parameters to be learned.

Then, another feasibility problem analogous to Problem 2 can be written to find a feasible θ consistent with the data: Problem 3 (Parametric constraint recovery problem).

DISPLAYFORM0 Denote G s and G ¬s as the set of guaranteed learned safe and unsafe constraint states.

One can check if a constraint state k ∈ G ¬s or k ∈ G s by enforcing g(k, θ) > 0 or g(k, θ) ≤ 0, respectively, and checking feasibility of Problem 3.

Crucially, G ¬s and G s are guaranteed underapproximations of A and A c (for space, we omit the proof; c.f.

BID2 ).A particularly common parameterization of an unsafe set is as a polytope A(θ) = {k | H(θ)k ≤ h(θ)}, where H(θ) and h(θ) are affine in θ.

In this case, θ can be found by solving a mixed integer feasibility problem: Problem 4 (Polytopic constraint recovery problem).

DISPLAYFORM1 where M is a large positive number and 1 N h is a column vector of ones of length N h .

Constraints (2a) and (2b) use big-M formulations to enforce that each safe constraint state lies outside A(θ) and that at least one constraint state on each unsafe trajectory lies inside A(θ).A few remarks are in order:• If the safe set is a polytope or if the safe set or unsafe set is a union of polytopes, a mixed integer feasibility program similar to Problem 4 can be solved to find θ.

A more general case where g(k, θ) is described by a Boolean conjunction of convex inequalities can be solved using satisfiability modulo convex optimization BID6 .•

In addition to recovering sets of guaranteed learned unsafe and safe constraint states, a probability distribution over possibly unsafe constraint states can be estimated by sampling unsafe sets from the feasible set of Problem 3.

V. EVALUATION ON 6D CONSTRAINT In this example, we learn a 6D hyper-rectangular pose constraint for the end effector of a 7-DOF Kuka iiwa arm.

In this scenario, the robot's task is to pick up a cup and bring it to a human, all while ensuring the cup's contents do not spill and proxemics constraints are satisfied (i.e. the end effector never gets too close to the human).

To this end, the end effector orientation (parametrized in Euler angles) is constrained to satisfy DISPLAYFORM2 DISPLAYFORM3 are generated by solving trajectory optimization problems for the kinematic, discrete-time model in 7D joint space, where for each demonstration T = 6 and control constraints u t ∈ [−2, 2] 7 , for all t (see Figures 2, 3 ).

The constraint is recovered with Problem 4, where H(θ) = [I, −I] and h(θ) = θ = [x,ȳ,z,ᾱ,β,γ, x, y, z, α, β, γ] .

From this data, Problem 4 is solved in 1.19 seconds on a 2017 Macbook Pro and returns the true θ and G s = S. G s is efficiently recovered using the insight that the axis-aligned bounding box of any two constraint states in G s must be contained in G s , since G s is the union of axis-aligned boxes and therefore must also be an axis-aligned box.

VI.

CONCLUSION In this paper, we extend BID2 to learn higher dimensional constraints by leveraging a known parameterization.

We show that the constraint recovery problem for the parameterized case can be solved with mixed integer programming, and evaluate the method on learning a 6D pose constraint for a 7-DOF robot arm.

Future work involves using learned constraints for probabilistically safe planning and developing safe exploration strategies and active demonstration-querying strategies to reduce the uncertainty in the learned constraint.

@highlight

We can learn high-dimensional constraints from demonstrations by sampling unsafe trajectories and leveraging a known constraint parameterization.