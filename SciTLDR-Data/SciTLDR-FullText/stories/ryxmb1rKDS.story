In this paper, we introduce Symplectic ODE-Net (SymODEN), a deep learning framework which can infer the dynamics of a physical system from observed state trajectories.

To achieve better generalization with fewer training samples, SymODEN incorporates appropriate inductive bias by designing the associated computation graph in a physics-informed manner.

In particular, we enforce Hamiltonian dynamics with control to learn the underlying dynamics in a transparent way which can then be leveraged to draw insight about relevant physical aspects of the system, such as mass and potential energy.

In addition, we propose a parametrization which can enforce this Hamiltonian formalism even when the generalized coordinate data is embedded in a high-dimensional space or we can only access velocity data instead of generalized momentum.

This framework, by offering interpretable, physically-consistent models for physical systems, opens up new possibilities for synthesizing model-based control strategies.

In recent years, deep neural networks (Goodfellow et al., 2016) have become very accurate and widely used in many application domains, such as image recognition (He et al., 2016) , language comprehension (Devlin et al., 2019) , and sequential decision making (Silver et al., 2017) .

To learn underlying patterns from data and enable generalization beyond the training set, the learning approach incorporates appropriate inductive bias (Haussler, 1988; Baxter, 2000) by promoting representations which are simple in some sense.

It typically manifests itself via a set of assumptions, which in turn can guide a learning algorithm to pick one hypothesis over another.

The success in predicting an outcome for previously unseen data then depends on how well the inductive bias captures the ground reality.

Inductive bias can be introduced as the prior in a Bayesian model, or via the choice of computation graphs in a neural network.

In a variety of settings, especially in physical systems, wherein laws of physics are primarily responsible for shaping the outcome, generalization in neural networks can be improved by leveraging underlying physics for designing the computation graphs.

Here, by leveraging a generalization of the Hamiltonian dynamics, we develop a learning framework which exploits the underlying physics in the associated computation graph.

Our results show that incorporation of such physics-based inductive bias offers insight about relevant physical properties of the system, such as inertia, potential energy, total conserved energy.

These insights, in turn, enable a more accurate prediction of future behavior and improvement in out-of-sample behavior.

Furthermore, learning a physically-consistent model of the underlying dynamics can subsequently enable usage of model-based controllers which can provide performance guarantees for complex, nonlinear systems.

In particular, insight about kinetic and potential energy of a physical system can be leveraged to synthesize appropriate control strategies, such as the method of controlled Lagrangian (Bloch et al., 2001 ) and interconnection & damping assignment (Ortega et al., 2002) , which can reshape the closed-loop energy landscape to achieve a broad range of control objectives (regulation, tracking, etc.) .

Inferring underlying dynamics from time-series data plays a critical role in controlling closed-loop response of dynamical systems, such as robotic manipulators (Lillicrap et al., 2015) and building HVAC systems (Wei et al., 2017) .

Although the use of neural networks towards identification and control of dynamical systems dates back to more than three decades (Narendra & Parthasarathy, 1990) , recent advances in deep neural networks have led to renewed interest in this domain.

Watter et al. (2015) learn dynamics with control from highdimensional observations (raw image sequences) using a variational approach and synthesize an iterative LQR controller to control physical systems by imposing a locally linear constraint.

Karl et al. (2016) and Krishnan et al. (2017) adopt a variational approach and use recurrent architectures to learn state-space models from noisy observation.

SE3-Nets (Byravan & Fox, 2017) learn SE(3) transformation of rigid bodies from point cloud data.

Ayed et al. (2019) use partial information about the system state to learn a nonlinear state-space model.

However, this body of work, while attempting to learn state-space models, does not take physics-based priors into consideration.

The main contribution of this work is two-fold.

First, we introduce a learning framework called Symplectic ODE-Net (SymODEN) which encodes a generalization of the Hamiltonian dynamics.

This generalization, by adding an external control term to the standard Hamiltonian dynamics, allows us to learn the system dynamics which conforms to Hamiltonian dynamics with control.

With the learned structured dynamics, we are able to synthesize controllers to control the system to track a reference configuration.

Moreover, by encoding the structure, we can achieve better predictions with smaller network sizes.

Second, we take one step forward in combining the physics-based prior and the data-driven approach.

Previous approaches (Lutter et al., 2019; Greydanus et al., 2019) require data in the form of generalized coordinates and their derivatives up to the second order.

However, a large number of physical systems accommodate generalized coordinates which are non-Euclidean (e.g., angles), and such angle data is often obtained in the embedded form, i.e., (cos q, sin q) instead of the coordinate (q) itself.

The underlying reason is that an angular coordinate lies on S 1 instead of R 1 .

In contrast to previous approaches which do not address this aspect, SymODEN has been designed to work with angle data in the embedded form.

Additionally, we leverage differentiable ODE solvers to avoid the need for estimating second-order derivatives of generalized coordinates.

Lagrangian dynamics and Hamiltonian dynamics are both reformulations of Newtonian dynamics.

They provide novel insights into the laws of mechanics.

In these formulations, the configuration of a system is described by its generalized coordinates.

Over time, the configuration point of the system moves in the configuration space, tracing out a trajectory.

Lagrangian dynamics describes the evolution of this trajectory, i.e., the equations of motion, in the configuration space.

Hamiltonian dynamics, however, tracks the change of system states in the phase space, i.e. the product space of generalized coordinates q = (q 1 , q 2 , ..., q n ) and generalized momenta p = (p 1 , p 2 , ..., p n ).

In other words, Hamiltonian dynamics treats q and p on an equal footing.

This not only provides symmetric equations of motion but also leads to a whole new approach to classical mechanics (Goldstein et al., 2002) .

Hamiltonian dynamics is also widely used in statistical and quantum mechanics.

In Hamiltonian dynamics, the time-evolution of a system is described by the Hamiltonian H(q, p), a scalar function of generalized coordinates and momenta.

Moreover, in almost all physical systems, the Hamiltonian is the same as the total energy and hence can be expressed as

where the mass matrix M(q) is symmetric positive definite and V (q) represents the potential energy of the system.

Correspondingly, the time-evolution of the system is governed bẏ

where we have dropped explicit dependence on q and p for brevity of notation.

Moreover, sincė

the total energy is conserved along a trajectory of the system.

The RHS of Equation (2) is called the symplectic gradient (Rowe et al., 1980) of H, and Equation (3) shows that moving along the symplectic gradient keeps the Hamiltonian constant.

In this work, we consider a generalization of the Hamiltonian dynamics which provides a means to incorporate external control (u), such as force and torque.

As external control is usually affine and only influences changes in the generalized momenta, we can express this generalization as

where the input matrix g(q) is typically assumed to have full column rank.

For u = 0, the generalized dynamics reduces to the classical Hamiltonian dynamics (2) and the total energy is conserved; however, when u = 0, the system has a dissipation-free energy exchange with the environment.

Once we have learned the dynamics of a system, the learned model can be used to synthesize a controller for driving the system to a reference configuration q .

As the proposed approach offers insight about the energy associated with a system, it is a natural choice to exploit this information for synthesizing controllers via energy shaping (Ortega et al., 2001 ).

As energy is a fundamental aspect of physical systems, reshaping the associated energy landscape enables us to specify a broad range of control objectives and synthesize nonlinear controllers with provable performance guarantees.

If rank(g(q)) = rank(q), the system is fully-actuated and we have control over any dimension of "acceleration" inṗ.

For such fully-actuated systems, a controller u(q, p) = β β β(q) + v(p) can be synthesized via potential energy shaping β β β(q) and damping injection v(p).

For completeness, we restate this procedure (Ortega et al., 2001 ) using our notation.

As the name suggests, the goal of potential energy shaping is to synthesize β β β(q) such that the closed-loop system behaves as if its time-evolution is governed by a desired Hamiltonian H d .

With this, we have

where the difference between the desired Hamiltonian and the original one lies in their potential energy term, i.e.

In other words, β β β(q) shape the potential energy such that the desired Hamiltonian H d (q, p) has a minimum at (q , 0).

Then, by substituting Equation (1) and Equation (6) into Equation (5), we get

Thus, with potential energy shaping, we ensure that the system has the lowest energy at the desired reference configuration.

Furthermore, to ensure that trajectories actually converge to this configuration, we add an additional damping term 2 given by

However, for underactuated systems, potential energy shaping alone cannot 3 drive the system to a desired configuration.

We also need kinetic energy shaping for this purpose (Chang et al., 2002) .

Remark If the desired potential energy is chosen to be a quadratic of the form

the external forcing term can be expressed as

This can be interpreted as a PD controller with an additional energy compensation term.

In this section, we introduce the network architecture of Symplectic ODE-Net.

In Subsection 3.1, we show how to learn an ordinary differential equation with a constant control term.

In Subsection 3.2, we assume we have access to generalized coordinate and momentum data and derive the network architecture.

In Subsection 3.3, we take one step further to propose a data-driven approach to deal with data of embedded angle coordinates.

In Subsection 3.4, we put together the line of reasoning introduced in the previous two subsections to propose SymODEN for learning dynamics on the hybrid space R n × T m .

Now we focus on the problem of learning the ordinary differential equation (ODE) from time series data.

Consider an ODE:ẋ = f (x).

Assume we don't know the analytical expression of the right hand side (RHS) and we approximate it with a neural network.

If we have time series data X = (x t0 , x t1 , ..., x tn ), how could we learn f (x) from the data?

Chen et al. (2018) introduced Neural ODE, differentiable ODE solvers with O(1)-memory backpropagation.

With Neural ODE, we make predictions by approximating the RHS function using a neural network f θ and feed it into an ODE solver x t1 ,x t2 , ...,x tn = ODESolve(x t0 , f θ , t 1 , t 2 , ..., t n )

We can then construct the loss function L = X−X 2 2 and update the weights θ by backpropagating through the ODE solver.

In theory, we can learn f θ in this way.

In practice, however, the neural net is hard to train if n is large.

If we have a bad initial estimate of the f θ , the prediction error would in general be large.

Although |x t1 −x t1 | might be small,x t N would be far from x t N as error accumulates, which makes the neural network hard to train.

In fact, the prediction error ofx t N is not as important asx t1 .

In other words, we should weight data points in a short time horizon more than the rest of the data points.

In order to address this and better utilize the data, we introduce the time horizon τ as a hyperparameter and predict x ti+1 , x ti+2 , ..., x ti+τ from initial condition x ti , where i = 0, ..., n − τ .

One challenge toward leveraging Neural ODE to learn state-space models is the incorporation of the control term into the dynamics.

Equation (4) has the formẋ = f (x, u) with x = (q, p).

A function of this form cannot be directly fed into Neural ODE directly since the domain and range of f have different dimensions.

In general, if our data consist of trajectories of (x, u) t0,...,tn where u remains the same in a trajectory, we can leverage the augmented dynamics

2 if we have access toq instead of p, we useq instead in Equation (8) 3 As gg T is not invertible, we cannot solve the matching condition given by Equation 7.

4

Under review as a conference paper at ICLR 2020

With this improvisation, we can match the input and output dimension off θ , which enables us to feed it into Neural ODE.

The idea here is to use different constant external forcing to get the system responses and use those responses to train the model.

With a trained model, we can apply a timevarying u to the dynamicsẋ = f θ (x, u) and generate estimated trajectories.

When we synthesize the controller, u remains constant in each integration step.

As long as our model interpolates well among different values of constant u, we could get good estimated trajectories with a time-varying u.

The problem is then how to design the network architecture off θ , or equivalently f θ such that we can learn the dynamics in an efficient way.

Suppose we have trajectory data consisting of (q, p, u) t0,...,tn , where u remains constant in a trajectory.

If we have the prior knowledge that the unforced dynamics of q and p is governed by Hamiltonian dynamics, we can use three neural nets -M −1 θ1 (q), V θ2 (q) and g θ3 (q) -as function approximators to represent the inverse of mass matrix, potential energy and the control coefficient.

Thus,

where

The partial derivative in the expression can be taken care of by automatic differentiation.

by putting the designed f θ (q, p, u) into Neural ODE, we obtain a systematic way of adding the prior knowledge of Hamiltonian dynamics into end-to-end learning.

In the previous subsection, we assume (q, p, u) t0,...,tn .

In a lot of physical system models, the state variables involve angles which reside in the interval [−π, π).

In other words, each angle resides on the manifold S 1 .

From a data-driven perspective, the data that respects the geometry is a 2 dimensional embedding (cos q, sin q).

Furthermore, the generalized momentum data is usually not available.

Instead, the velocity is often available.

For example, in OpenAI Gym (Brockman et al., 2016) Pendulum-v0 task, the observation is (cos q, sin q,q).

From a theoretical perspective, however, the angle itself is often used, instead of the 2D embedding.

The reason being both the Lagrangian and the Hamiltonian formulations are derived using generalized coordinates.

Using an independent generalized coordinate system makes it easier to solve for the equations of motion.

In this subsection, we take the data-driven standpoint.

We assume all the generalized coordinates are angles and the data comes in the form of (x 1 (q), x 2 (q), x 3 (q), u) t0,...,tn = (cos q, sin q,q, u) t0,...,tn .

We aim to incorporate our theoretical prior -Hamiltonian dynamics -into the data-driven approach.

The goal is to learn the dynamics of x 1 , x 2 and x 3 .

Noticing p = M(x 1 , x 2 )q, we can write down the derivative of x 1 , x 2 and x 3 ,

where "•" represents the elementwise product (Hadamard product).

We assume q and p evolve with the generalized Hamiltonian dynamics Equation (4).

Here the Hamiltonian H(x 1 , x 2 , p) is a function of x 1 , x 2 and p instead of q and p.

Then the right hand side of Equation (14) can be expressed as a function of state variables and control (x 1 , x 2 , x 3 , u).

Thus, it can be fed into the Neural ODE.

We use three neural nets -M

2 ) and g θ3 (x 1 , x 2 ) -as function approximators.

Substitute Equation (15) and Equation (16) into Equation (14), then the RHS serves as f θ (x 1 , x 2 , x 3 , u).

where

In Subsection 3.2, we treated the generalized coordinates as translational coordinates.

In Subsection 3.3, we developed a method to better deal with embedded angle data.

In most of physical systems, these two types of coordinates coexist.

For example, robotics systems are usually modelled as interconnected rigid bodies.

The positions of joints or center of mass are translational coordinates and the orientations of each rigid body are angular coordinates.

In other words, the generalized coordinates lie on R n × T m , where T m denotes the m-torus, with T 1 = S 1 and

In this subsection, we put together the architecture of the previous two subsections.

We assume the generalized coordinates are q = (r, φ φ φ) ∈ R n × T m and the data comes in the form of (x 1 , x 2 , x 3 , x 4 , x 5 , u) t0,...,tn = (r, cos φ φ φ, sin φ φ φ,ṙ,φ φ φ, u) t0,...,tn .

With similar line of reasoning, we use three neural nets -M

with Hamiltonian dynamics, we havė

where theṙ andφ φ φ come from Equation (22).

Now we obtain a f θ which can be fed into Neural ODE.

Figure 1 shows the flow of the computation graph based on Equation (20)- (24).

In real physical systems, the mass matrix M is positive definite, which ensures a positive kinetic energy with a non-zero velocity.

The positive definiteness of M implies the positive definiteness of 4 In Equation (17), the derivative of M −1 θ 1 (x1, x2) can be expanded using chain rule and expressed as a function of the states.

, where L θ1 is a lower-triangular matrix.

The positive definiteness is ensured if the diagonal elements of M −1 θ1 are positive.

In practice, this can be done by adding a small constant to the diagonal elements of M θ1 .

It not only makes M θ1 invertible, but also stabilize the training.

We use the following four tasks to evaluate the performance of Symplectic ODE-Net model -(i) Task 1: a pendulum with generalized coordinate and momentum data (learning on R 1 ); (ii) Task 2: a pendulum with embedded angle data (learning on S 1 ); (iii) Task 3: a cart-pole system (learning on R 1 × S 1 ); and (iv) Task 4: an acrobot (learning on T 2 ).

Model Variants.

Besides the Symplectic ODE-Net model derived above, we consider a variant by approximating the Hamiltonian using a fully connected neural net H θ1,θ2 .

We call it Unstructured Symplectic ODE-Net (Unstructured SymODEN) since this model does not exploit the structure of the Hamiltonian (1).

Baseline Models.

In order to show that we can learn the dynamics better with less parameters by leveraging prior knowledge, we set up baseline models for all four experiments.

For the pendulum with generalized coordinate and momentum data, the naive baseline model approximates Equation (12) -f θ (x, u) -by a fully connected neural net.

For all the other experiments, which involves embedded angle data, we set up two different baseline models: naive baseline approximates f θ (x, u) by a fully connected neural net.

It doesn't respect the fact that the coordinate pair, cos φ φ φ and sin φ φ φ, lie on T m .

Thus, we set up the geometric baseline model which approximatesq andṗ with a fully connected neural net.

This ensures that the angle data evolves on T m .

Data Generation.

For all tasks, we randomly generated initial conditions of states and subsequently combined them with 5 different constant control inputs, i.e., u = −2.0, −1.0, 0.0, 1.0, 2.0 to produce the initial conditions and input required for simulation.

The simulators integrate the corresponding dynamics for 20 time steps to generate trajectory data which is then used to construct the training set.

The simulators for different tasks are different.

For Task 1, we integrate the true generalized Hamiltonian dynamics with a time interval of 0.05 seconds to generate trajectories.

All the other tasks deal with embedded angle data and velocity directly, so we use OpenAI Gym (Brockman et al., 2016) simulators to generate trajectory data.

One drawback of using OpenAI Gym is that not all environments use the Runge-Kutta method (RK4) to carry out the integration.

OpenAI Gym favors other numerical schemes over RK4 because of speed, but it is harder to learn the dynamics with inaccurate data.

For example, if we plot the total energy as a function of time from data generated by Pendulum-v0 environment with zero action, we see that the total energy oscillates around a constant by a significant amount, even though the total energy should be conserved.

Thus, for Task 2 and Task 3, we use Pendulum-v0 and CartPole-v1, respectively, and replace the numerical integrator of the environments to RK4.

For Task 4, we use the Acrobot-v1 environment which is already using RK4.

We also change the action space of Pendulum-v0, CartPole-v1 and Acrobot-v1 to a continuous space with a large enough bound.

Model training.

In all the tasks, we train our model using Adam optimizer (Kingma & Ba, 2014) with 1000 epochs.

We set a time horizon τ = 3, and choose "RK4" as the numerical integration scheme in Neural ODE.

We vary the size of the training set by doubling from 16 initial state conditions to 1024 initial state conditions.

Each initial state condition is combined with five constant control u = −2.0, −1.0, 0.0, 1.0, 2.0 to produce initial condition and input for simulation.

Each trajectory is generated by integrating the dynamics 20 time steps forward.

We set the size of minibatches to be the number of initial state conditions.

We logged the train error per trajectory and the prediction error per trajectory in each case for all the tasks.

The train error per trajectory is the mean squared error (MSE) between the estimated trajectory and the ground truth over 20 time steps.

To evaluate the performance of each model in terms of long time prediction, we construct the metric of prediction error per trajectory by using the same initial state condition in the training set with a constant control of u = 0.0, integrating 40 time steps forward, and calculating the MSE over 40 time steps.

The reason for using only the unforced trajectories is that a constant nonzero control might cause the velocity to keep increasing or decreasing over time, and large absolute values of velocity are of little interest for synthesizing controllers.

In this task, we use the model described in Section 3.2 and present the predicted trajectories of the learned models as well as the learned functions of SymODEN.

We also point out the drawback of treating the angle data as a Cartesian coordinate.

The dynamics of this task has the following forṁ

with Hamiltonian H(q, p) = 1.5p 2 + 5(1 − cos q).

In other words M (q) = 3, V (q) = 5(1 − cos q) and g(q) = 1.

In Figure 2 , The ground truth is an unforced trajectory which is energyconserved.

The prediction trajectory of the baseline model does not conserve energy, while both the SymODEN and its unstructured variant predict energy-conserved trajectories.

For SymODEN, the learned g θ3 (q) and M −1 θ1 (q) matches the ground truth well.

V θ2 (q) differs from the ground truth with a constant.

This is acceptable since the potential energy is a relative notion.

Only the derivative of V θ2 (q) plays a role in the dynamics.

Here we treat q as a variable in R 1 and our training set contains initial conditions of q ∈ [−π, 3π].

The learned functions do not extrapolate well outside this range, as we can see from the left part in the figures of M −1 θ1 (q) and V θ2 (q).

We address this issue by working directly with embedded angle data, which leads us to the next subsection.

In this task, the dynamics is the same as Equation (25) but the training data are generated by the OpenAI Gym simulator, i.e. we use embedded angle data and assume we only have access toq instead of p. We use the

Under review as a conference paper at ICLR 2020 model described in Section 3.3 and synthesize an energy-based controller (Section 2.2).

Without true p data, the learned function matches the ground truth with a scaling β, as shown in Figure 3 .

To explain the scaling, let us look at the following dynamicṡ

with Hamiltonian H = p 2 /(2α) + 15α(1 − cos q).

If we only look at the dynamics of q, we havë q = −15 sin q+3u, which is independent of α.

If we don't have access to the generalized momentum p, our trained neural network may converge to a Hamiltonian with a α e which is different from the true value, α t = 1/3, in this task.

By a scaling β = α t /α e = 0.357, the learned functions match the ground truth.

Even we are not learning the true α t , we can still perform prediction and control since we are learning the dynamics of q correctly.

We let V d = −V θ2 (q), then the desired Hamiltonian has minimum energy when the pendulum rests at the upward position.

For the damping injection, we let K d = 3.

Then from Equation (7) and (8), the controller we synthesize is Only SymODEN out of all models we consider provides the learned potential energy which is required to synthesize the controller.

Figure 4 shows how the states evolve when the controller is fed into the OpenAI Gym simulator.

We can successfully control the pendulum into the inverted position using the controller based on the learned model even though the absolute maximum control u, 7.5, is more than three times larger than the absolute maximum u in the training set, which is 2.0.

This shows SymODEN extrapolates well.

The CartPole system is an underactuated system and to synthesize a controller to balance the pole from arbitrary initial condition requires trajectory optimization or kinetic energy shaping.

We show that we can learn its dynamics and perform prediction in Section 4.6.

We also train SymODEN in a fully-actuated version of the CartPole system (see Appendix E).

The corresponding energy-based controller can bring the pole to the inverted position while driving the cart to the origin.

The Acrobot is an underactuated double pendulum.

As this system exhibits chaotic motion, it is not possible to predict its long-term behavior.

However, Figure 6 shows that SymODEN can provide reasonably good short-term prediction.

We also train SymODEN in a fully-actuated version of the Acrobot and show that we can control this system to reach the inverted position (see Appendix E).

In this subsection, we show the train error, prediction error, as well as the MSE and total energy of a sample test trajectory for all the tasks.

Figure 5 shows the variation in train error and prediction error with changes in the number of initial state conditions in the training set.

We can see that SymODEN yields better generalization in every task.

In Task 3, although the Geometric Baseline Model yields lower train error in comparison to the other models, SymODEN generates more accurate predictions, indicating overfitting in the Geometric Baseline Model.

By incorporating the physics-based prior of Hamiltonian dynamics, SymODEN learns dynamics that obeys physical laws and thus provides better predictions.

In most cases, SymODEN trained with a smaller training dataset performs better than other models in terms of the train and prediction error, indicating that better generalization can be achieved even with fewer training samples.

R2-C4 R2-C5 (16, 32, 64, 128, 256, 512, 1024) in the training set.

Both the horizontal axis and vertical axis are in log scale.

Figure 6 shows the evolution of MSE and total energy along a trajectory with a previously unseen initial condition.

For all the tasks, MSE of the baseline models diverges faster than SymODEN.

Unstructured SymODEN performs well in all tasks except Task 3.

As for the total energy, in Task 1 and Task 2, SymODEN and Unstructured SymODEN conserve total energy by oscillating around a constant value.

In these models, the Hamiltonian itself is learned and the prediction of the future states stay around a level set of the Hamiltonian.

Baseline models, however, fail to find the conservation and the estimation of future states drift away from the initial Hamiltonian level set.

Here we have introduced Symplectic ODE-Net which provides a systematic way to incorporate prior knowledge of Hamiltonian dynamics with control into a deep learning framework.

We show that SymODEN achieves better prediction with fewer training samples by learning an interpretable, physically-consistent state-space model.

Future works will incorporate a broader class of physicsbased prior, such as the port-Hamiltonian system formulation, to learn dynamics of a larger class of physical systems.

SymODEN can work with embedded angle data or when we only have access to velocity instead of generalized momentum.

Future works would explore other types of embedding, such as embedded 3D orientations.

Another interesting direction could be to combine energy shaping control (potential as well as kinetic energy shaping) with interpretable end-to-end learning frameworks.

Tianshu Wei, Yanzhi Wang, and Qi Zhu.

Deep Reinforcement Learning for Building HVAC Control.

In Proceedings of the 54th Annual Design Automation Conference (DAC), pp.

22:1-22:6, 2017.

The architectures used for our experiments are shown below.

For all the tasks, SymODEN has the lowest number of total parameters.

To ensure that the learned function is smooth, we use Tanh activation function instead of ReLu.

As we have differentiation in the computation graph, nonsmooth activation functions would lead to discontinuities in the derivatives.

This, in turn, would result in an ODE with a discontinuous RHS which is not desirable.

All the architectures shown below are fully-connected neural networks.

The first number indicates the dimension of the input layer.

The last number indicates the dimension of output layer.

The dimension of hidden layers is shown in the middle along with the activation functions.

Task 1: Pendulum

• Input: 2 state dimensions, 1 action dimension

• Baseline Model (0.36M parameters): 2 -600Tanh -600Tanh -2Linear

• Unstructured SymODEN (0.20M parameters):

• SymODEN (0.13M parameters):

Task 2: Pendulum with embedded data

• Input: 3 state dimensions, 1 action dimension • Naive Baseline Model (0.65M parameters): 4 -800Tanh -800Tanh -3Linear

• Geometric Baseline Model (0.46M parameters):

, where L θ1 : 1 -300Tanh -300Tanh -300Tanh -1Linear -approximate (q,ṗ): 4 -600Tanh -600Tanh -2Linear

, where

Task 3: CartPole

• Input: 5 state dimensions, 1 action dimension • Naive Baseline Model (1.01M parameters): 6 -1000Tanh -1000Tanh -5Linear

, where L θ1 : 3 -400Tanh -400Tanh -400Tanh -3Linear -H θ2 : 5 -500Tanh -500Tanh -1Linear -g θ3 : 3 -300Tanh -300Tanh -2Linear

, where L θ1 : 3 -400Tanh -400Tanh -400Tanh -3Linear -V θ2 : 3 -300Tanh -300Tanh -1Linear -g θ3 : 3 -300Tanh -300Tanh -2Linear

Task 4:Acrobot • Input: 6 state dimensions, 1 action dimension • Naive Baseline Model (1.46M parameters): 7 -1200Tanh -1200Tanh -6Linear

, where L θ1 : 4 -400Tanh -400Tanh -400Tanh -3Linear -approximate (q,ṗ): 7 -800Tanh -800Tanh -4Linear

, where L θ1 : 4 -400Tanh -400Tanh -400Tanh -3Linear -H θ2 : 6 -600Tanh -600Tanh -1Linear -g θ3 : 4 -300Tanh -300Tanh -2Linear

, where L θ1 : 4 -400Tanh -400Tanh -400Tanh -3Linear -V θ2 : 4 -300Tanh -300Tanh -1Linear -g θ3 : 4 -300Tanh -300Tanh -2Linear

The energy-based controller has the form u(q, p) = β β β(q) + v(p), where the potential energy shaping term β β β(q) and the damping injection term v(p) are given by Equation (7) and Equation (8), respectively.

If the desired potential energy V q (q) is given by a quadratic, as in Equation (9), then

and the controller can be expressed as

The corresponding external forcing term is then given by

which is same as Equation (10) in the main body of the paper.

The first term in this external forcing provides an energy compensation, whereas the second term and the last term are proportional and derivative control terms, respectively.

Thus, this control can be perceived as a PD controller with an additional energy compensation.

In Hamiltonian Neural Networks (HNN), Greydanus et al. (2019) incorporate the Hamiltonian structure into learning by minimizing the difference between the symplectic gradients and the true gradients.

When the true gradient is not available, which is often the case, the authors suggested using finite difference approximations.

In SymODEN, true gradients or gradient approximations are not necessary since we integrate the estimated gradient using differentiable ODE solvers and set up the loss function with the integrated values.

Here we perform an ablation study of the differentiable ODE Solver.

Both HNN and the Unstructured SymODEN approximate the Hamiltonian by a neural network and the main difference is the differentiable ODE solver, so we compare the performance of HNN and the Unstructured SymODEN.

We set the time horizon τ = 1 since it naturally corresponds to the finite difference estimate of the gradient.

A larger τ would correspond to higher-order estimates of gradients.

Since there is no angle-aware design in HNN, we use Task 1 to compare the performance of these two models.

We generate 25 training trajectories, each of which contains 45 time steps.

This is consistent with the HNN paper.

In the HNN paper Greydanus et al. (2019) , the initial conditions of the trajectories are generated randomly in an annulus, whereas in this paper, we generate the initial state conditions uniformly in a reasonable range in each state dimension.

We guess the reason the authors of HNN choose the annulus data generation is that they do not have an angle-aware design.

Take the pendulum for example; all the training and test trajectories they generate do not pass the inverted position.

If they make prediction on a trajectory with a large enough initial speed, the angle would go over ±2π, ±4π, etc.

in the long run.

Since these are away from the region where the model gets trained, we can expect the prediction would be poor.

In fact, this motivates us to design the angle-aware SymODEN in Section 3.3.

In this ablation study, we generate the training data in both ways.

Table 1 shows the train error and the prediction error per trajectory of the two models.

We can see Unstructured SymODEN performs better than HNN.

This is an expected result.

To see why this is the case, let us assume the training loss per time step of HNN is similar to that of Unstructured SymODEN.

Since the training loss is on the symplectic gradient, the error would accumulate while integrating the symplectic gradient to get the estimated state values, and MSE of the state values would likely be one order of magnitude greater than that of Unstructured SymODEN.

Figure 7 shows the MSE and total energy of a particular trajectory.

It is clear that the MSE of the Unstructured SymODEN is lower than that of HNN.

The MSE of HNN periodically touches zero does not mean it has a good prediction at that time step.

Since the trajectories in the phase space are closed circles, those zeros mean the predicted trajectory of HNN lags behind (or runs ahead of) the true trajectory by one or more circles.

Also, the energy of the HNN trajectory drifts instead of staying constant, probably because the finite difference approximation is not accurate enough.

Incorporating the differential ODE solver also introduces two hyperparameters: solver types and time horizon τ .

For the solver types, the Euler solver is not accurate enough for our tasks.

The adaptive solver "dopri5" lead to similar train error, test error and prediction error as the RK4 solver, but requires more time during training.

Thus, in our experiments, we choose RK4.

Time horizon τ is the number of points we use to construct our loss function.

Table 2 shows the train error, test error and prediction error per trajectory in Task 2 when τ is varied from 1 to 5.

We can see that longer time horizons lead to better models.

This is expected since long time horizons penalize worse long term predictions.

We also observe in our experiments that longer time horizons require more time to train the models.

CartPole and Acrobot are underactuated systems.

Incorporating the control of underactuated systems into the end-to-end learning framework is our future work.

Here we trained SymODEN on

Under review as a conference paper at ICLR 2020 fully actuated versions of Cartpole and Acrobot and synthesized controllers based on the learned model.

For the fully-actuated CartPole, Figure 8 shows the snapshots of the system of a controlled trajectory with an initial condition where the pole is below the horizon.

Figure 9 shows the time series of state variables and control inputs.

We can successfully learn the dynamics and control the pole to the inverted position and the cart to the origin.

For the fully-actuated Acrobot, Figure 10 shows the snapshots of a controlled trajectory.

Figure 11 shows the time series of state variables and control inputs.

We can successfully control the Acrobot from the downward position to the upward position, though the final value of q 2 is a little away from zero.

Taking into account that the dynamics has been learned with only 64 different initial state conditions, it is most likely that the upward position did not show up in the training data.

Here we show statistics of train, test, and prediction per trajectory in all four tasks.

The train errors are based on 64 initial state conditions and 5 constant inputs.

The test errors are based on 64 previously unseen initial state conditions and the same 5 constant inputs.

Each trajectory in the train and test set contains 20 steps.

The prediction error is based on the same 64 initial state conditions (during training) and zero inputs.

@highlight

This work enforces Hamiltonian dynamics with control to learn system models from embedded position and velocity data, and exploits this physically-consistent dynamics to synthesize model-based control via energy shaping.