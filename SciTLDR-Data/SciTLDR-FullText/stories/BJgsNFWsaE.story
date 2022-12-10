We present a data driven approach to construct a library of feedback motion primitives for non-holonomic vehicles that guarantees bounded error in following arbitrarily long trajectories.

This ensures that motion re-planning can be avoided as long as disturbances to the vehicle remain within a certain bound and also potentially when the obstacles are displaced within a certain bound.

The library is constructed along local abstractions of the dynamics that enables addition of new motion primitives through abstraction refinement.

We provide sufficient conditions for construction of such robust motion primitives for a large class of nonlinear dynamics, including commonly used models, such as the standard Reeds-Shepp model.

The algorithm is applied for motion planning and control of a rover with slipping without its prior modelling.

Various state-the-art motion planning approaches for carlike vehicles use the bicycle model to generate feasible trajectories for high level planning BID3 .

The model is either discretized in lattice based methods or used as a heuristic for measuring distance between two states in sampling based methods such as rapidly exploring random trees (RRT) BID1 .

It is then up to the low level feedback controllers of the vehicle to follow the prescribed trajectory; an overview of this group of approaches can be found in Paden et al. BID2 .

This might prove a challenge in cases where the bicycle model does not resemble the actual vehicle dynamics closely enough; this may result in growing error between the prescribed trajectory and vehicles position which in turn may require trajectory re-planning BID3 .

Recently, approaches such as Howard et al. BID0 and Schwarting et al. BID4 have been proposed that can incorporate the vehicle dynamics in planning to ensure collision avoidance by using model predictive control.

While model predictive control can provide feasible trajectories for a large class of nonlinear models, it becomes prohibitively complex for long prediction horizons and may fall into local optima for short prediction horizons in non-convex problem settings BID5 .In this work we follow the input discretization approach similar to lattice based methods for motion planning.

Instead of relying on a model, we sample from the input space similar to Howard et al. BID0 .

The main contribution in this work is that we construct locally linear abstractions of the system around samples in the input space and design local feedback rules to ensure fixed upper bound on state error after applying any motion primitive considering both linearization error and initial state error.

Therefore, we can guarantee bounded state error through application of the motion primitives at all times.

The idea of feedback based motion primitives has also been presented in Vukosavljev et al. BID6 for multi-agent drones with omni-directional controllability; the main contrast here is that we provide a tool for construction of such motion primitives for non-holonomic vehicles.

We pose an assumption we refer to as robustifiability in order to be able to synthesize such motion primitives.

Consider a vehicle whose dynamics is governed by the following discrete-time nonlinear system: DISPLAYFORM0 with x(t) ∈ X ⊆ R n as system state and u(t) ∈ U ⊂ R m as system input and W as a bounded disturbance.

Let X and U be compact sets.

The operator ⊕ is used to denote Minkowski sum of two sets: DISPLAYFORM1 Let us be given a starting position of the agent x 0 , a free sub-space F ⊆ X , and a goal region X g ⊂ F .

We define the problem as follows:Problem 1: Given an agent with translation invariant dynamics (1), Find a sequence of state subsets (X(0), X(1), ..., X(T )) and a corresponding sequence of motion primitives, i.e. control strategy (U 1 (.),U 2 (.), ...,U T (.)) such that: DISPLAYFORM2 III.

APPROACH Our approach builds on defining each motion primitive as a composition of a constant input and a feedback control term.

We start by defining coarse motion primitives by splitting the input space using a coarse grid.

We take each grid cell center as the constant input for the respective motion primitive.

We design a feedback control law around each center such that there exists a bound ε and a number of time steps k with the property that if the state uncertainty at time step t < T − k is less than ε, the state uncertainty at t + k will also be less than ε.

By state uncertainty being less than ε we mean that the set of states, where the system can be at time t + k fits inside an ε-ball.

We have proved that under certain assumptions, such feedback control and bound ε can be found.

The assumptions are three-fold: (i) function f is twice differentiable, (ii) its Hessian is element-wise bounded, and (iii) it is so-called robustifiable, which is defined as follows:Definition 1: f is said to be robustifiable on X × U in k steps if and only if for all x ∈ int(X ) and u 1 , ..., u n ∈ int(U ) the robustifiability matrix [ DISPLAYFORM3 ] is full rank, where f k is a multi-step extension of dynamics f .

Note that being full rank is equivalent to not having any singular value equal to zero, as a result we can associate a well conditioned robustifiability matrix with good robustifiability, i.e. possibility to steer the state in any arbitrary direction.

An example of a robustifiable system is the Reeds-Shepp model as can be seen in FIG1 .

It is robustifiable even when controlled only through the steering angle, but it is much better conditioned when controlled through both steering angle and velocity which is also intuitive.

For a linear system f (x, u) = Ax + Bu robustifiability is equivallent to controllability: We have DISPLAYFORM4 Having ensured bounded state uncertainty, we will now attempt to find a sequence of motion primitives satisfying Problem 1.

We translate the problem into a planning problem on a discrete graph, where vertices represent centers of ε-balls that are entirely in the free space, and edges are defined by motion primitives driving the system from one center of an ε-ball to another.

The feedback control term ensures that regardless of where within the former ε-ball the system is, it will end up within the latter ε-ball.

The discrete planning problem can then be addressed e.g., via A*. If a satisfying plan cannot be found, we compute an over-approximation of the reachable set to determine, whether the plan does not exist for the original system, or whether the grid was not fine enough to prove or disprove existence of such plan.

In the latter case, we refine the grid on the input space, and repeat the procedure.

The algorithm is asymptotically complete for deterministic systems.

The size of the graph treated by A* grows exponentially with the number of refinements.

The construction of motion primitives does not require a model of the dynamics.

Only through input sampling, and under the above stated bounded Hessian and robustifiability assumptions, it is possible to construct motion primitives that guarantee bounded state uncertainty at any point in time.

Furthermore, for an environment with a single convex moving obstacle or in cases where obstacles can be considered one at a time based on their proximity to the vehicle, it is straightforward to extend the feedback strategy of motion primitives using the vehicle's relative position to the obstacle rather than its absolute position, having the obstacle's motion rate in place of the bounded disturbance.

In this case however, the end state of the vehicle may not converge to the goal set as the obstacle moves and if the obstacle is close to the goal set.

In general the reach-avoid problem is non-convex and as a result cannot be addressed only through continuous feedback, but in many practical cases it may help avoid re-planning.

We tested our approach on an Erle-Rover Unmanned Ground Vehicle (UGV) in a room with motion capture for positioning and slippery floor.

The algorithm is run on MAT-LAB communicating with the rover through ROS.

The system model is derived through input sampling and is shown for one of the motion primitives in FIG0 .

@highlight

We show that under some assumptions on vehicle dynamics and environment uncertainty it is possible to automatically synthesize motion primitives that do not accumulate error over time.