Hierarchical reinforcement learning methods offer a powerful means of planning flexible behavior in complicated domains.

However, learning an appropriate hierarchical decomposition of a domain into subtasks remains a substantial challenge.

We present a novel algorithm for subtask discovery, based on the recently introduced multitask linearly-solvable Markov decision process (MLMDP) framework.

The MLMDP can perform never-before-seen tasks by representing them as a linear combination of a previously learned basis set of tasks.

In this setting, the subtask discovery problem can naturally be posed as finding an optimal low-rank approximation of the set of tasks the agent will face in a domain.

We use non-negative matrix factorization to discover this minimal basis set of tasks, and show that the technique learns intuitive decompositions in a variety of domains.

Our method has several qualitatively desirable features: it is not limited to learning subtasks with single goal states, instead learning distributed patterns of preferred states; it learns qualitatively different hierarchical decompositions in the same domain depending on the ensemble of tasks the agent will face; and it may be straightforwardly iterated to obtain deeper hierarchical decompositions.

Hierarchical reinforcement learning methods hold the promise of faster learning in complex state spaces and better transfer across tasks, by exploiting planning at multiple levels of detail BID0 .

A taxi driver, for instance, ultimately must execute a policy in the space of torques and forces applied to the steering wheel and pedals, but planning directly at this low level is beset by the curse of dimensionality.

Algorithms like HAMS, MAXQ, and the options framework permit powerful forms of hierarchical abstraction, such that the taxi driver can plan at a higher level, perhaps choosing which passengers to pick up or a sequence of locations to navigate to BID19 BID3 BID13 .

While these algorithms can overcome the curse of dimensionality, they require the designer to specify the set of higher level actions or subtasks available to the agent.

Choosing the right subtask structure can speed up learning and improve transfer across tasks, but choosing the wrong structure can slow learning BID17 BID1 .

The choice of hierarchical subtasks is thus critical, and a variety of work has sought algorithms that can automatically discover appropriate subtasks.

One line of work has derived subtasks from properties of the agent's state space, attempting to identify states that the agent passes through frequently BID18 .

Subtasks are then created to reach these bottleneck states (van Dijk & Polani, 2011; BID17 BID4 .

In a domain of rooms, this style of analysis would typically identify doorways as the critical access points that individual skills should aim to reach (??im??ek & Barto, 2009 ).

This technique can rely only on passive exploration of the agent, yielding subtasks that do not depend on the set of tasks to be performed, or it can be applied to an agent as it learns about a particular ensemble of tasks, thereby suiting the learned options to a particular task set.

Another line of work converts the target MDP into a state transition graph.

Graph clustering techniques can then identify connected regions, and subtasks can be placed at the borders between connected regions BID11 .

In a rooms domain, these connected regions might correspond to rooms, with their borders again picking out doorways.

Alternately, subtask states can be identified by their betweenness, counting the number of shortest paths that pass through each specific node (??im??ek & Barto, 2009; BID17 .

Other recent work utilizes the eigenvectors of the graph laplacian to specify dense rewards for option policies that are defined over the full state space BID10 .

Finally, other methods have grounded subtask discovery in the information each state reveals about the eventual goal (van Dijk & Polani, 2011) .

Most of these approaches aim to learn options with a single or low number of termination states, can require high computational expense BID17 , and have not been widely used to generate multiple levels of hierarchy (but see BID24 ; BID12 ).Here we describe a novel subtask discovery algorithm based on the recently introduced Multitask linearly-solvable Markov decision process (MLMDP) framework BID14 , which learns a basis set of tasks that may be linearly combined to solve tasks that lie in the span of the basis BID21 .

We show that an appropriate basis can naturally be found through non-negative matrix factorization BID8 BID3 , yielding intuitive decompositions in a variety of domains.

Moreover, we show how the technique may be iterated to learn deeper hierarchies of subtasks.

In line with a number of prior methods, BID17 BID12 our method operates in the batch off-line setting; with immediate application to probabilistic planning.

The subtask discovery method introduced in BID10 , which also utilizes matrix factorization techniques to discover subtasks albeit from a very different theoretical foundation, is notable for its ability to operate in the online RL setting, although it is not immediately clear how the approach taken therein might achieve a deeper hierarchical architecture, or enable immediate generalization to novel tasks.

In the multitask framework of BID14 , the agent faces a set of tasks where each task has an identical transition structure, but different terminal rewards, modeling the setting where an agent pursues different goals in the same fixed environment.

Each task is modeled as a finite-exit LMDP BID21 .

The LMDP is an alternative formulation of the standard MDP that carefully structures the problem formulation such that the Bellman optimality equation becomes linear in the exponentiated cost-to-go.

As a result of this linearity, optimal policies compose naturally: solutions for rewards corresponding to linear combinations of two optimal policies are simply the linear combination of their respective exponentiated cost-to-go functions BID22 .

This special property of LMDPs is exploited by BID14 to develop a multitask reinforcement learning method that uses a library of basis tasks, defined by their boundary rewards, to perform a potentially infinite variety of other tasks-any tasks that lie in the subspace spanned by the basis can be performed optimally.

Briefly, the LMDP BID21 is defined by a three-tuple L = S, P, R , where S is a set of states, P is a passive transition probability distribution P : S ?? S ??? [0, 1], and R is an expected instantaneous reward function R : S ??? R. The 'action' chosen by the agent is a full transition probability distribution over next states, a(??|s).

A control cost is associated with this choice such that a preference for energy-efficient actions is inherently specified: actions corresponding to distributions over next states that are very different from the passive transition probability distribution are expensive, while those that are similar are cheap.

In this way the problem is regularized by the passive transition structure.

Finally, the LMDP has rewards r i (s) for each interior state, and r b (s) for each boundary state in the finite exit formulation.

The LMDP can be solved by finding the desirability function z(s) = e V (s)/?? which is the exponentiated cost-to-go function for a specific state s.

Here ?? is a temperature-like parameter related to the stochasticity of the solution.

Given z(s), the optimal control can be computed in closed form (see BID20 for details).

Despite the restrictions inherent in the formulation, the LMDP is generally applicable; see the supplementary material in BID14 for examples of how the LMDP can be applied to non-navigational, and conceptual tasks.

A primary difficulty in translating standard MDPs into LMDPs is the construction of the action-free passive dynamics P (although a general way of approximating MDPs using LMDPs is given in BID20 ); however, in many cases, this can simply be taken as the resulting Markov chain under a uniformly random policy.

In this instance the problem is said to be 'entropy regularized'.

A similar problem set-up appears in a number of recent works BID15 BID6 .The Multitask LMDP (MLDMP) BID14 operates by learning a set of N t tasks, defined by LMDPs L t = S, P, q i , q t b , t = 1, ?? ?? ?? , N t with identical state space, passive dynamics, and internal rewards, but different instantaneous exponentiated boundary reward structures q for the multitask module.

With this machinery in place, if a new task with boundary reward q can be expressed as a linear combination of previously learned tasks, q = Qw.

Then the same weighting can be applied to derive the corresponding optimal desirability function, z = Zw, due to the compositionality of the LMDP.

More generally, if the new task cannot be exactly expressed as a linear combination of previously learned tasks, a significant jump-start in learning may nevertheless be gained by finding an approximate representation.

The multitask module can be stacked to form deep hierarchies BID14 by iteratively constructing higher order MLMDPs in which higher levels select the instantaneous reward structure that defines the current task for lower levels in a feudal-like architecture.

This recursive procedure is carried out by firstly augmenting the layer l state spaceS l = S l ???S l t with a set of N t terminal boundary states S l t called subtask states.

Transitioning into a subtask state corresponds to a decision by the layer l MLMDP to access the next level of the hierarchy, and is equivalent to entering a state of the higher layer.

These subtask transitions are governed by a new N ] are then suitably defined BID14 .

Solving the higher layer MLMDP will yield an optimal action a(??|s) making some transitions more likely than they would be under the passive dynamic, indicating that they are more desirable for the current task.

Similarly, some transitions will be less likely than they would be under the passive dynamic, indicating that they should be avoided for the current task.

The instantaneous rewards for the lower layer are therefore set to be proportional to the difference between the controlled and passive dynamic, r DISPLAYFORM0 Return control to lower layer to execute g) Figure 1 : Execution model for the hierarchical MLMDP BID14 .

a) Beginning at some start state, the agent will make a transition underP 1 .

This transition may be to an interior, boundary, or subtask state.

b) Transitioning into a subtask state is equivalent to entering a state of the higher layer MLMDP.

No 'real' time passes during this transition.

c) The higher layer MLMDP is then solved and a next higher layer state is drawn.

d) Knowing the next state at the higher layer allows us to specify the reward structure defining the current task at the lower layer.

Control is then passed back to the lower layer to achieve this new task.

Notice that the details of how this task should be solved are left to the lower layer (one possible trajectory being shown).

e) At some point in the future the agent may again elect to transition into a subtask state -in this instance the transition is into a different subtask corresponding to a different state in the higher layer.

f) The higher layer MLMDP is solved, and a next state drawn.

This specifies the reward structure for a new task at the lower layer.

g) Control is again passed back to the lower layer, which attempts to solve the new task.

This process continues until the agent transitions into a boundary state.

Prior work has assumed that the task basis Q is given a priori by the designer.

Here we address the question of how a suitable basis may be learned.

A natural starting point is to find a basis that retains as much information as possible about the ensemble of tasks to be performed, analogously to how principal component analysis yields a basis that maximally preserves information about an ensemble of vectors.

In particular, to perform new tasks well, the desirability function for a new task must be representable as a (positive) linear combination of the desirability basis matrix Z. This naturally suggests decomposing Z using PCA (i.e., the SVD) to obtain a low-rank approximation that retains as much variance as possible in Z. However, there is one important caveat: the desirability function is the exponentiated cost-to-go, such that Z = exp(V /??).

Therefore Z must be non-negative, otherwise it does not correspond to a well-defined cost-to-go function.

Our approach to subtask discovery is thus to uncover a low-rank representation through non-negative matrix factorization, to realize this positivity constraint BID8 BID3 .

We seek a decomposition of Z into a data matrix D ??? R (m??k) and a weight matrix W ??? R (k??n) as: DISPLAYFORM0 where d ij , w ij ??? 0.

The value of k in the decomposition must be chosen by a designer to yield the desired degree of abstraction, and is referred to as the decomposition factor.

A small value of k corresponds to a high degree of abstraction since the variance in the desirability space Z must be captured in a k dimensional subspace spanned by the vectors in the data matrix D. Conversely, a large value of k corresponds to a low degree of abstraction.

Since Z is strictly positive, the non-negative decomposition is not unique for any value of k BID5 .

Formally then, we seek a decomposition which minimizes the cost function DISPLAYFORM1 where d denotes the ??-divergence, a subclass of the more familiar Bregman Divergences BID7 , between the true basis Z and the approximate basis D. The ??-divergence collapses to the better known statistical distances for ?? ??? {1, 2}, corresponding to the Kullback-Leibler and Euclidean distances respectively BID2 .Crucially, since Z depends on the set of tasks that the agent will perform in the environment, the representation is defined by the tasks taken against it, and is not simply a factorization of the domain structure.

To keep the focus on the decomposition strategy, we assume, here and throughout, that Z ??? R n??n is given.

The basis set of tasks can be a tiny fraction of the set of possible tasks in the space.

As an example, suppose we consider tasks with boundary rewards at any of two separate locations in an n-dimensional world such that there are n-choose-2 possible tasks (corresponding to tasks like 'navigate to point A or B').

We require only an n-dimensional Z matrix containing tasks to navigate to each point individually.

The resulting subtasks we uncover will aid in solving all of these n-choose-2 tasks.

More generally we might consider tasks in which boundary rewards are placed at three or more locations, etc.

To know Z therefore means to know an optimal policy to achieve n of ??? 2 n tasks in a space.

An online version of this method would estimate Z from data, either directly or by learning a transition model (see BID10 for some possibilities).

Nested roomHairpin maze DISPLAYFORM0 Desirability functions for subtasksFigure 2: Intuitive decompositions in structured domains.

All colour-plots correspond to the desirability functions for subtasks overlaid onto the base domains shown in panels a) and e).

b,c,d) Subtasks correspond to 'regions', distributed patterns over preferred states, rather than single states.

Where the decomposition factor is chosen to match the structure of the domain (here k = 16 for example), subtasks correspond to an intuitive semantic -"go to room X".

f,g,h) Again, subtasks correspond to regions rather than single states.

Collectively the subtasks form an approximate cover for the space.

To demonstrate that the proposed scheme recovers an intuitive decomposition, we consider the resulting low-rank approximation to the desirability basis in two domains, for a few hand-picked decomposition factors.

All results presented in this section correspond to solutions to Eqn.(2) for ?? = 1 so that the cost function is taken to be the KL-divergence (although the method does not appear to be overly sensitive to ?? ??? [1, 2]).

Note that in the same way that the columns of Z represent the exponentiated cost-to-go for the single-states tasks in the basis, so the columns in D represent the exponentiated cost-to-go for the discovered subtasks.

In Fig. 2 , we compute the data matrix D ??? R m??k for k = {4, 9, 16} for both the nested rooms domain, and the hairpin domain.

The desirability functions for each of the subtasks is then plotted over the base domain.

All of the decompositions share a number of properties intrinsic to the proposed scheme.

Most notably, the subtasks themselves do not correspond to single states (like bottle-neck states), but rather to complex distributions over preferred states.

By way of example, semantically, a single subtask in Fig. 2-d corresponds to the task 'Go to Room', where any state in the room is suitable as a terminal state for the subtask.

Also, since Z is taken to be the full basis matrix in this example, the distributed patterns of the subtasks collectively form an approximate cover for the full space.

This is true regardless of the decomposition factor chosen.

It is worthwhile noting that the decompositions discovered are refactored for larger values of k. That is to say that the decomposition for k = 5 is not the same as the decomposition for k = 4 just with the addition of an extra subtask.

Instead all five of the subtasks in the decomposition are adjusted allowing for maximum expressiveness in the representation.

It follows that there is no intrinsic ordering of the subtasks.

It only matters that they collectively form a good representation of the task space Z. While we have shown only spatial decomposition thus far, our scheme is applicable to tasks more general than simple navigation-like tasks.

To make this point clear, we consider the scheme's application to the standard TAXI domain BID3 with one passenger and four pickup/drop-off locations.

The 5 ?? 5 TAXI domain considered is depicted in FIG2 .

Here the agent operates in the product space of the base domain (5 ?? 5 = 25), and the possible passenger locations (5-choose-1 = 5) for a complete state-space of 125 states.

We consider a decomposition with factor k = 5.

FIG2 is a depiction of the subtask structure we uncover.

Each column of FIG2 is one of the subtasks we discover.

Each of these is a policy over the full state space.

For visual clarity, these are then divided into the five copies of the base domain, each being defined by the passenger's location.

The color-map corresponds to the desirability function for each subtask.

To help interpret the semantic nature of the subtasks discovered, consider the first column of FIG2 .

This subtask has almost all of its desirability function mass focused at states in which the passenger is in the Taxi.

This task is thus a general pick-up action.

By a similar analysis, column two of FIG2 depicts a subtask whose desirability function is essentially uniform over all states where the passenger is at location A. Semantically this subtask seeks to enter states with the passenger at location A regardless of taxi position.

This subtask thus corresponds to the drop-off action at location A. Also note the slight probability leakage into the 'in taxi' state for the drop off point -the precondition for the passenger to be dropped off.

Considered as a whole, the subtask basis represents policies for getting the passenger to each of the pick-up/drop-off locations, and for having the passenger in the taxi.

The proposed scheme discovers a set of subtasks by finding a low-rank approximation to the desirability basis matrix Z. By leveraging the stacking mechanism defined in BID14 , this approximation procedure can simply be reapplied to find an approximate desirability basis for each subsequent layer of the hierarchy, by factoring the desirability matrix Z l+1 at each layer.

However, As a demonstration of the recursive and multiscale nature of the scheme, we consider a spacial domain inspired by the multiscale nature of cities, see FIG3 .

At the highest level we consider a city which is comprised of three major communities, each of which is comprised of five houses.

Each house is further comprised of four rooms, each of which is comprised of sixteen base states in a 4 ?? 4 grid.

We consider a decomposition in line with the natural scales of the domain and take k l = {3 ?? 5 ?? 4 = 60, 3 ?? 5 = 15, 3} respectively for l = 2, 3, 4.

As expected, the scheme discovers subtasks corresponding to the multiscale nature of the domain with the highest layer subtasks intuitively corresponding to whole communities, etc.

Of course the semantic clarity of the subtasks is due to the specific decomposition factors chosen, but any decomposition factors would work to solve tasks in the domain.

At this point the scheme has automated the discovery of the subtasks themselves, and the transitions into these subtasks.

What remains is for a designer to specify the decomposition factors k l at each layer.

In an analogy to neural network architectures, the scheme has defined the network connectivity but not the number of neurons at each layer.

While this is a typical hyperparameter, by leveraging the unique construction in Eqn.(2), a good value for this parameter may be estimated from data.

By increasing the decomposition factor k l the approximation error, given by Eqn.

FORMULA2 , is monotonically decreased.

For some domains there is an obvious inflection point at which increasing the decomposition factor only slightly improves the approximation.

Let us denote the dependence of d ?? (??) on the decomposition factor simply as f (k).

Then we may somewhat naively take the smallest value that demonstrates diminishing incremental returns as a good value for k. In this instance the (LEFT) By projecting the state at each layer back into the base domain, it becomes apparent that subtasks correspond to distributed patterns of preferred states, rather than single goal states.

In this way hierarchical subtasks are ever more abstracted in space and time, as higher layers are accessed.

Tangibly, where the states at the lowest layer correspond to individual locations, higher layer states correspond to entire rooms, houses, and communities correspondingly. (RIGHT) An abstract representation of 'subtasks' as states of a higher layer MLMDP.

A key contribution of this paper is to define an autonomous way of uncovering the contents of higher layer states, and the transition structures into these states.

approximation error, Eqn.

FORMULA2 , is said to exhibit elbow-joint behaviour: DISPLAYFORM0 In practice, when the task ensemble is drawn uniformly from the domain, the observed elbow-joint behaviour is an encoding of the high-level domain structure.

Choosing the right set of subtasks is known to speed up learning and improve transfer between tasks.

However, choosing the wrong subtasks can actually slow learning.

While in general it is not possible to assess a priori whether a set of subtasks is 'good' or 'bad', the new approach taken here provides a natural measure of the quality of a set of subtasks, by evaluating the quality of the approximation in Eqn.(1).

It follows immediately that different sets of subtasks can be compared simply by evaluating Eqn.(1) for each set individually.

This leads naturally to the notion of subtask equivalence.

Suppose some standard metric is defined on the space of matrices as m(A, B).

Then a formal pseudoequivalence relation may be defined on the set of subtasks, encoded as the columns of the data matrix D, by assigning subtasks that provide similar approximations to the desirability basis to the same classes.

Explicitly, for DISPLAYFORM0 The pseudo-equivalence class follows as: DISPLAYFORM1 A full equivalence relation here fails since transitivity does not hold.

As noted above, our scheme typically uncovers subtasks as complex distributions over preferred states, rather than individual states themselves.

As in Fig.(2) , we uncover regions such as 'rooms', whereas other methods typically uncover single states such as 'doorways'.

There is a natural duality between these abstractions, which we consider below.

A weight vector can be assigned to each state by solving Eqn.(1) for a specific z: DISPLAYFORM0 This weight vector can be thought of as the representation of s in D. To each state we then assign a real-valued measure of stability, by considering how much this representation changes under state-transition.

Explicitly, we consider the stability function g : S ??? R: DISPLAYFORM1 which is a measure of how the representations of neighbour states differ from the current state, weighted by the probability of transitioning to those neighbours.

States for which g(s) takes a high value are considered to be unstable, whereas states for which g(s) takes a small value are considered to be stable.

Unstable states are those which fall on the boundary between subtask 'regions'.

A cursory analysis of Fig.(6) immediately identifies doorways as being those unstable states.

Figure 6: A natural duality exists between the subtasks uncovered by our scheme, and those typically uncovered by other methods.

a) A filter-stack of four subtasks corresponding to the layer one decomposition.

Here k 1 = 4 and we present the full set of subtasks.

Each of the four subtasks corresponds to one of the four rooms in the domain.

b) A hand picked example path through the domain, chosen to illustrate the changing representation for different domain states in terms of the higher layer states.

This path and does not correspond to a real agent trajectory.

c) For each state along the example path we compute the desirability function z s and approximate it using a linear blend of our subtasks according to Eqn.(6).

The task weights are plotted as a function of steps, revealing the change in representation for different states along the example path.

d) Agnostic to any particular path, we compute the stability function g(s) for each state in the domain.

It is immediately clear that unstable states, those for which the representation in D l changes starkly, correspond to 'doorways'.

We present a novel subtask discovery mechanism based on the low rank approximation of the desirability basis afforded by the LMDP framework.

The new scheme reliably uncovers intuitive decompositions in a variety of sample domains.

Unlike methods based on pure state abstraction, the proposed scheme is fundamentally dependent on the task ensemble, recovering different subtask representations for different task ensembles.

Moreover, by leveraging the stacking procedure for hierarchical MLMDPs, the subtask discovery mechanism may be straightforwardly iterated to yield powerful hierarchical abstractions.

Finally, the unusual construction allows us to analytically probe a number of natural questions inaccessible to other methods; we consider specifically a measure of the quality of a set of subtasks, and the equivalence of different sets of subtasks.

A current drawback of the approach is its reliance on a discrete, tabular, state space.

Scaling to high dimensional problems will require applying state function approximation schemes, as well as online estimation of Z directly from experience.

These are avenues of current work.

More abstractly, the method might be extended by allowing for some concept of nonlinear regularized composition allowing more complex behaviours to be expressed by the hierarchy.

AMS thanks the Swartz Program in Theoretical Neuroscience at Harvard University for support.

<|TLDR|>

@highlight

We present a novel algorithm for hierarchical subtask discovery which leverages the multitask linear Markov decision process framework.