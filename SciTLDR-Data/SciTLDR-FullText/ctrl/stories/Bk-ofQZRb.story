Temporal Difference Learning with function approximation is known to be unstable.

Previous work like \citet{sutton2009fast} and \citet{sutton2009convergent} has presented alternative objectives that are stable to minimize.

However, in practice, TD-learning with neural networks requires various tricks like using a target network that updates slowly \citep{mnih2015human}.

In this work we propose a constraint on the TD update that minimizes change to the target values.

This constraint can be applied to the gradients of any TD objective, and can be easily applied to nonlinear function approximation.

We validate this update by applying our technique to deep Q-learning, and training without a target network.

We also show that adding this constraint on Baird's counterexample keeps Q-learning from diverging.

Temporal Difference learning is one of the most important paradigms in Reinforcement Learning (Sutton & Barto) .

Techniques based on nonlinear function approximators and stochastic gradient descent such as deep networks have led to significant breakthroughs in the class of problems that these methods can be applied to BID9 BID13 BID12 .

However, the most popular methods, such as TD(λ), Q-learning and Sarsa, are not true gradient descent techniques BID2 and do not converge on some simple examples BID0 .

BID0 and BID1 propose residual gradients as a way to overcome this issue.

Residual methods, also called backwards bootstrapping, work by splitting the TD error over both the current state and the next state.

These methods are substantially slower to converge, however, and BID16 show that the fixed point that they converge to is not the desired fixed point of TD-learning methods.

BID16 propose an alternative objective function formulated by projecting the TD target onto the basis of the linear function approximator, and prove convergence to the fixed point of this projected Bellman error is the ideal fixed point for TD methods.

BID5 extend this technique to nonlinear function approximators by projecting instead on the tangent space of the function at that point.

Subsequently, BID11 has combined these techniques of residual gradient and projected Bellman error by proposing an oblique projection, and BID8 has shown that the projected Bellman objective is a saddle point formulation which allows a finite sample analysis.

However, when using deep networks for approximating the value function, simpler techniques like Q-learning and Sarsa are still used in practice with stabilizing techniques like a target network that is updated more slowly than the actual parameters BID10 .In this work, we propose a constraint on the update to the parameters that minimizes the change to target values, freezing the target that we are moving our current predictions towards.

Subject to this constraint, the update minimizes the TD-error as much as possible.

We show that this constraint can be easily added to existing techniques, and works with all the techniques mentioned above.

We validate our method by showing convergence on Baird's counterexample and a gridworld domain.

On the gridworld domain we parametrize the value function using a multi-layer perceptron, and show that we do not need a target network.

Reinforcement Learning problems are generally defined as a Markov Decision Process (MDP), (S, A, P , R, R, d 0 , γ).

We use the definition and notation as defined in Sutton & Barto, second edition, unless otherwise specified.

In case of a function approximation, we define the value and action value functions with parameters by θ.

DISPLAYFORM0 We focus on TD(0) methods, such as Sarsa, Expected Sarsa and Q-learning.

The TD error that all these methods minimize is as follows: DISPLAYFORM1 The choice of π determines if the update is on-policy or off-policy.

For Q-learning the target is max a q(s t+1 , a).If we consider TD-learning using function approximation, the loss that is minimized is the squared TD error.

For example, in Q-learning DISPLAYFORM2 The gradient of this loss is the direction in which you update the parameters.

We shall define the gradient of the TD loss with respect to state s t and parameters θ t as g T D (s t ).

The gradient of some other function f (s t |θ t ) can similarly be defined as g f (s t ).

The parameters are then updated according to gradient descent with step size α as follows: DISPLAYFORM3

A key characteristic of TD-methods is bootstrapping, i.e. the update to the prediction at each step uses the prediction at the next step as part of it's target.

This method is intuitive and works exceptionally well in a tabular setting (Sutton & Barto) .

In this setting, updates to the value of one state, or state-action pair do not affect the values of any other state or state-action.

TD-learning using a function approximator is not so straightforward, however.

When using a function approximator, states nearby will tend to share features, or have features that are very similar.

If we update the parameters associated with these features, we will update the value of not only the current state, but also states nearby that use those features.

In general, this is what we want to happen.

With prohibitively large state spaces, we want to generalize across states instead of learning values separately for each one.

However, if the value of the state visited on the next step, which often does share features, is also updated, then the results of the update might not have the desired effect on the TD-error.

Generally, methods for TD-learning using function approximation do not take into account that updating θ t in the direction that minimizes TD-error the most, might also change v(s t+1 |θ t+1 ).

Though they do not point out this insight as we have, previous works that aims to address convergence of TD methods using function approximation do deal with this issue indirectly, like residual gradients BID0 and methods minimizing MSPBE BID16 .

Residual gradients does this by essentially updating the parameters of the next state in the opposite direction of the update to the parameters of the current state.

This splits the error between the current state and the next state, and the fixed point we reach does not act as a predictive representation of the reward.

MSPBE methods act by removing the component of the error that is not in the span of the features of the current state, by projecting the TD targets onto these features.

The update for these methods involves the product of three expectations, which is handled by keeping a separate set of weights that approximate two of these expectations, and is updated at a faster scale.

The idea also does not immediately scale to nonlinear function approximation.

BID5 propose a solution by projecting the error on the tangent plane to the function at the point at which it is evaluated.

We propose to instead constrain the update to the parameters such that the change to the values of the next state is minimized, while also minimizing the TD-error.

To do this, instead of modifying the objective, we look at the gradients of the update.

DISPLAYFORM0 is the gradient at s t that minimizes the TD error.

g v (s t+1 ) is the gradient at s t+1 that will change the value the most.

We update the parameters θ t with g update (s t ) such that the update is orthogonal to g v (s t+1 ).

That is, we update the parameters θ t such that there is no change in the direction that will affect v(s t+1 ).

Graphically, the update can be seen in figure 1.

The actual updates to the parameters are as given below.

DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 As can be seen, the proposed update is orthogonal to the direction of the gradient at the next state.

Which means that it will minimize the impact on the next state.

On the other hand, DISPLAYFORM4 • .

This implies that applying g update (s t ) to the parameters θ minimizes the TD error, unless it would change the values of the next state.

Furthermore, our technique can be applied on top of any of these techniques to improve their behavior.

We show this for residual gradients and Q-learning in the following experiments.

To show that our method learns just as fast as TD while guaranteeing convergence similar to residual methods, we show the behavior of our algorithm on the following 3 examples.

Figure 2: Baird's Counterexample is specified by 6 states and 7 parameters.

The value of each state is calculated as given inside the state.

At each step, the agent is initialized at one of the 6 states uniformly at random, and transitions to the state at the bottom, shown by the arrows.

Baird's counterexample is a problem introduced in BID0 to show that gradient descent with function approximation using TD updates does not converge.

The comparison of our technique with Q-learning and Residual Gradients is shown in figure 2 .

We compare the average performance for all tehcniques over 10 independent runs.

If we apply gradient projection while using the TD error, we show that both Q-learning (TD update) and updates using residual gradients BID0 converge, but not to the ideal values of 0.

In the figure, these values are almost overlapping.

Our method constrains the gradient to not modify the weights of the next state, which in this case means that w 0 and w 6 never get updated.

This means that the values do not converge to the true values (0), but they do not blow up as they do if using regular TD updates.

Residual gradients converge to the ideal values of 0 eventually.

GTD2 BID16 ) also converges to 0, as was shown in the paper, but we have not included that in this graph to avoid cluttering.

The Gridworld domain we use is a (10×10) room with d 0 = S, and R((0, 4)) = 1 and 0 everywhere else.

We have set the goal as (0, 4) arbitrarily and our results are similar for any goal on this grid.

The input to the function approximation is only the (x, y) coordinates of the agent.

We use a deep network with 2 hidden layers, each with 32 units, for approximating the Q-values.

We execute a softmax policy, and the target values are also calculated as v(s t+1 ) = a π(a|s t+1 )q(s t+1 , a), Figure 4 : Comparison of DQN and Constrained on the Cartpole Problem, taken over 10 runs.

The shaded area specifies std deviation in the scores of the agent across independent runs.

The agent is cut off after it's average performance exceeds 199 over a running window of 100 episodes where the policy π is a softmax over the Q-values.

The room can be seen in FIG2 with the goal in red, along with a comparison of the value functions learnt for the 2 methods we compare.-Q-learning Constrained Q-learning MSE 0.0335 ± 0.0017 0.0076 ± 0.0028 et al., 1983) .

We use implementations from OpenAI baselines BID6 for Deep Qlearning to ensure that the code is reproducible and to ensure fairness.

The network we use is with 2 hidden layers of [5, 32] .

The only other difference compared to the implemented baseline is that we use RMSProp BID17 as the particular machinary for optimization instead of Adam BID7 .

This is just to stay close to the method used in BID10 and in practice, Adam works just as well and the comparison is similar.

The two methods are trained using exactly the same code except for the updates, and the fact that Constrained DQN does not use a target network.

We can also train COnstrained DQN with a larger step size (10 −3 ), while DQN requires a smaller step size (10 −4 ) to learn.

The comparison with DQN is shown in 4.

We see that constrained DQN learns much faster, with much less variance than regular DQN.

In this paper we introduce a constraint on the updates to the parameters for TD learning with function approximation.

This constraint forces the targets in the Bellman equation to not move when the update is applied to the parameters.

We enforce this constraint by projecting the gradient of the TD error with respect to the parameters for state s t onto the orthogonal space to the gradient with respect to the parameters for state s t+1 .We show in our experiments that this added constraint stops parameters in Baird's counterexample from exploding when we use TD-learning.

But since we do not allow changes to target parameters, this also keeps Residual Gradients from converging to the true values of the Markov Process.

On a Gridworld domain we demonstrate that we can perform TD-learning using a 2-layer neural network, without the need for a target network that updates more slowly.

We compare the solution obtained with DQN and show that it is closer to the solution obtained by tabular policy evaluation.

Finally, we also show that constrained DQN can learn faster and with less variance on the classical Cartpole domain.

For future work, we hope to scale this approach to larger problems such as the Atari domain BID4 .

We would also like to prove convergence of TD-learning with this added constraint.

<|TLDR|>

@highlight

We show that adding a constraint to TD updates stabilizes learning and allows Deep Q-learning without a target network