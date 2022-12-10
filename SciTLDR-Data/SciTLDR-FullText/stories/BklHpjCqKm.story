Deep learning has achieved astonishing results on many tasks with large amounts of data and generalization within the proximity of training data.

For many important real-world applications, these requirements are unfeasible and additional prior knowledge on the task domain is required to overcome the resulting problems.

In particular, learning physics models for model-based control requires robust extrapolation from fewer samples – often collected online in real-time – and model errors may lead to drastic damages of the system.

Directly incorporating physical insight has enabled us to obtain a novel deep model learning approach that extrapolates well while requiring fewer samples.

As a first example, we propose Deep Lagrangian Networks (DeLaN) as a deep network structure upon which Lagrangian Mechanics have been imposed.

DeLaN can learn the equations of motion of a mechanical system (i.e., system dynamics) with a deep network efficiently while ensuring physical plausibility.

The resulting DeLaN network performs very well at robot tracking control.

The proposed method did not only outperform previous model learning approaches at learning speed but exhibits substantially improved and more robust extrapolation to novel trajectories and learns online in real-time.

A Deep learning has achieved astonishing results on many tasks with large amounts of data and generalization within the proximity of training data.

For many important real-world applications, these requirements are unfeasible and additional prior knowledge on the task domain is required to overcome the resulting problems.

In particular, learning physics models for model-based control requires robust extrapolation from fewer samples -often collected online in real-time -and model errors may lead to drastic damages of the system.

Directly incorporating physical insight has enabled us to obtain a novel deep model learning approach that extrapolates well while requiring fewer samples.

As a first example, we propose Deep Lagrangian Networks (DeLaN) as a deep network structure upon which Lagrangian Mechanics have been imposed.

DeLaN can learn the equations of motion of a mechanical system (i.e., system dynamics) with a deep network efficiently while ensuring physical plausibility.

The resulting DeLaN network performs very well at robot tracking control.

The proposed method did not only outperform previous model learning approaches at learning speed but exhibits substantially improved and more robust extrapolation to novel trajectories and learns online in real-time.

In the last five years, deep learning has propelled most areas of learning forward at an impressive pace BID22 BID31 BID41 -with the exception of physically embodied systems.

This lag in comparison to other application areas is somewhat surprising as learning physical models is critical for applications that control embodied systems, reason about prior actions or plan future actions (e.g., service robotics, industrial automation).

Instead, most engineers prefer classical off-the-shelf modeling as it ensures physical plausibility -at a high cost of precise measurements1 and engineering effort.

These plausible representations are preferred as these models guarantee to extrapolate to new samples, while learned models only achieve good performance in the vicinity of the training data.

To learn a model that obtains physically plausible representations, we propose to use the insights from physics as a model prior for deep learning.

In particular, the combination of deep learning and physics seems natural as the compositional structure of deep networks enables the efficient computation of the derivatives at machine precision BID35 and, thus, can encode a differential equation describing physical processes.

Therefore, we suggest to encode the physics prior in the form of a differential in the network topology.

This adapted topology amplifies the information content of the training samples, regularizes the end-to-end training, and emphasizes robust models capable of extrapolating to new samples while simultaneously ensuring physical plausibility.

Hereby, we concentrate on learning models of mechanical systems using the Euler-Lagrange-Equation, a second order ordinary differential equation (ODE) originating from Lagrangian Mechanics, as physics prior.

We focus on learning models of mechanical systems as this problem is one of the fundamental challenges of robotics BID7 BID39 .

The contribution of this work is twofold.

First, we derive a network topology called Deep Lagrangian Networks (DeLaN) encoding the Euler-Lagrange equation originating from Lagrangian Mechanics.

This topology can be trained using standard end-to-end optimization techniques while maintaining physical plausibility.

Therefore, the obtained model must comply with physics.

Unlike previous approaches to learning physics BID1 BID25 , which engineered fixed features from physical assumptions requiring knowledge of the specific physical embodiment, we are 'only' enforcing physics upon a generic deep network.

For DeLaN only the system state and the control signal are specific to the physical system but neither the proposed network structure nor the training procedure.

Second, we extensively evaluate the proposed approach by using the model to control a simulated 2 degrees of freedom (dof) robot and the physical 7-dof robot Barrett WAM in real time.

We demonstrate DeLaN's control performance where DeLaN learns the dynamics model online starting from random initialization.

In comparison to analytic-and other learned models, DeLaN yields a better control performance while at the same time extrapolates to new desired trajectories.

In the following we provide an overview about related work (Section 2) and briefly summarize Lagrangian Mechanics (Section 3).

Subsequently, we derive our proposed approach DeLaN and the necessary characteristics for end-to-end training are shown (Section 4).

Finally, the experiments in Section 5 evaluate the model learning performance for both simulated and physical robots.

Here, DeLaN outperforms existing approaches.

Models describing system dynamics, i.e. the coupling of control input τ τ τ and system state q, are essential for model-based control approaches BID17 .

Depending on the control approach, the control law relies either on the forward model f , mapping from control input to the change of system state, or on the inverse model f −1 , mapping from system change to control input, i.e., DISPLAYFORM0 Examples for application of these models are inverse dynamics control BID7 , which uses the inverse model to compensate system dynamics, while model-predictive control BID4 and optimal control BID46 use the forward model to plan the control input.

These models can be either derived from physics or learned from data.

The physics models must be derived for the individual system embodiment and requires precise knowledge of the physical properties BID0 .

When learning the model2, mostly standard machine learning techniques are applied to fit either the forward-or inverse-model to the training data.

E.g., authors used Linear Regression BID39 BID14 , Gaussian Mixture Regression BID3 BID20 , Gaussian Process Regression BID21 BID34 BID32 , Support Vector Regression BID5 BID10 , feedforward- BID18 BID26 BID25 BID38 or recurrent neural networks BID37 to fit the model to the observed measurements.

Only few approaches incorporate prior knowledge into the learning problem.

BID38 use the graph representation of the kinematic structure as input.

While the work of BID1 , commonly referenced as the standard system identification technique for robot manipulators BID40 , uses the Newton-Euler formalism to derive physics features using the kinematic structure and the joint measurements such that the learning of the dynamics model simplifies to linear regression.

Similarly, BID25 hard-code these physics features within a neural network and learn the dynamics parameters using gradient descent rather than linear regression.

Even though these physics features are derived from physics, the 2Further information can be found in the model learning survey by BID33 learned parameters for mass, center of gravity and inertia must not necessarily comply with physics as the learned parameters may violate the positive definiteness of the inertia matrix or the parallel axis theorem BID44 .

Furthermore, the linear regression is commonly underdetermined and only allows to infer linear combinations of the dynamics parameters and cannot be applied to close-loop kinematics BID40 .DeLaN follows the line of structured learning problems but in contrast to previous approaches guarantees physical plausibility and provides a more general formulation.

This general formulation enables DeLaN to learn the dynamics for any kinematic structure, including kinematic trees and closed-loop kinematics, and in addition does not require any knowledge about the kinematic structure.

Therefore, DeLaN is identical for all mechanical systems, which is in strong contrast to the NewtonEuler approaches, where the features are specific to the kinematic structure.

Only the system state and input is specific to the system but neither the network topology nor the optimization procedure.

The combination of differential equations and Neural Networks has previously been investigated in literature.

Early on BID23 BID24 proposed to learn the solution of partial differential equations (PDE) using neural networks and currently this topic is being rediscovered by BID35 ; BID42 ; BID28 .

Most research focuses on using machine learning to overcome the limitations of PDE solvers.

E.g., BID42 proposed the Deep Galerkin method to solve a high-dimensional PDE from scattered data.

Only the work of BID36 took the opposite standpoint of using the knowledge of the specific differential equation to structure the learning problem and achieve lower sample complexity.

In this paper, we follow the same motivation as BID36 but take a different approach.

Rather than explicitly solving the differential equation, DeLaN only uses the structure of the differential equation to guide the learning problem of inferring the equations of motion.

Thereby the differential equation is only implicitly solved.

In addition, the proposed approach uses different encoding of the partial derivatives, which achieves the efficient computation within a single feed-forward pass, enabling the application within control loops.

Describing the equations of motion for mechanical systems has been extensively studied and various formalisms to derive these equations exist.

The most prominent are Newtonian-, Hamiltonianand Lagrangian-Mechanics.

Within this work Lagrangian Mechanics is used, more specifically the Euler-Lagrange formulation with non-conservative forces and generalized coordinates.3 Generalized coordinates are coordinates that uniquely define the system configuration.

This formalism defines the Lagrangian L as a function of generalized coordinates q describing the complete dynamics of a given system.

The Lagrangian is not unique and every L which yields the correct equations of motion is valid.

The Lagrangian is generally chosen to be DISPLAYFORM0 where T is the kinetic energy and V is the potential energy.

The kinetic energy T can be computed for all choices of generalized coordinates using T = 1 2 q T H(q) q, whereas H(q) is the symmetric and positive definite inertia matrix BID7 .

The positive definiteness ensures that all non-zero velocities lead to positive kinetic energy.

Applying the calculus of variations yields the Euler-Lagrange equation with non-conservative forces described by DISPLAYFORM1 where τ τ τ are generalized forces.

Substituting L and dV/dq = g(q) into Equation 3 yields the second order ordinary differential equation (ODE) described by DISPLAYFORM2 3More information can be found in the textbooks BID13 BID7 BID9 where c describes the forces generated by the Centripetal and Coriolis forces BID9 .

Using this ODE any multi-particle mechanical system with holonomic constraints can be described.

For example various authors used this ODE to manually derived the equations of motion for coupled pendulums BID13 , robotic manipulators with flexible joints BID2 BID43 , parallel robots BID30 BID11 BID27 or legged robots BID15 BID12 .

DISPLAYFORM3 Starting from the Euler-Lagrange equation FORMULA4 ), traditional engineering approaches would estimate H(q) and g(q) from the approximated or measured masses, lengths and moments of inertia.

On the contrary most traditional model learning approaches would ignore the structure and learn the inverse dynamics model directly from data.

DeLaN bridges this gap by incorporating the structure introduced by the ODE into the learning problem and learns the parameters in an end-to-end fashion.

More concretely, DeLaN approximates the inverse model by representing the unknown functions g(q) and H(q) as a feed-forward networks.

Rather than representing H(q) directly, the lower-triangular matrix L(q) is represented as deep network.

Therefore, g(q) and H(q) are described bŷ DISPLAYFORM4 where.

refers to an approximation and θ and ψ are the respective network parameters.

The parameters θ and ψ can be obtained by minimizing the violation of the physical law described by Lagrangian Mechanics.

Therefore, the optimization problem is described by DISPLAYFORM5 withf DISPLAYFORM6 wheref −1 is the inverse model and can be any differentiable loss function.

The computational graph off −1 is shown in FIG0 .Using this formulation one can conclude further properties of the learned model.

NeitherL nor g are functions of q or q and, hence, the obtained parameters should, within limits, generalize to arbitrary velocities and accelerations.

In addition, the obtained model can be reformulated and used as a forward model.

Solving Equation 6 for q yields the forward model described bŷ DISPLAYFORM7 whereLL T is guaranteed to be invertible due to the positive definite constraint FORMULA8 ).

However, solving the optimization problem of Equation 5 directly is not possible due to the ill-posedness of the Lagrangian L not being unique.

The Euler-Lagrange equation is invariant to linear transformation and, hence, the Lagrangian L = αL + β solves the Euler-Lagrange equation if α is non-zero and L is a valid Lagrangian.

This problem can be mitigated by adding an additional penalty term to Equation 5 described by DISPLAYFORM8 where Ω is the L 2 -norm of the network weights.

Solving the optimization problem of Equation 9 with a gradient based end-to-end learning approach is non-trivial due to the positive definite constraint FORMULA8 ) and the derivatives contained inf −1 .

In particular, d(LL T )/dt and ∂ q T LL T q /∂q i cannot be computed using automatic differentiation as t is not an input of the network and most implementations of automatic differentiation do not allow the backpropagation of the gradient through the computed derivatives.

Therefore, the derivatives contained inf −1 must be computed analytically to exploit the full gradient information for training of the parameters.

In the following we introduce a network structure that fulfills the positive-definite constraint for all parameters (Section 4.1), prove that the derivatives d(LL T )/dt and ∂ q T LL T q /∂q i can be computed analytically (Section 4.2) and show an efficient implementation for computing the derivatives using a single feed-forward pass (Section 4.3).

Using these three properties the resulting network architecture can be used within a real-time control loop and trained using standard end-to-end optimization techniques.

The derivatives d LL T /dt and ∂ q T LL T q /∂q i are required for computing the control signal τ τ τ using the inverse model and, hence, must be available within the forward pass.

In addition, the second order derivatives, used within the backpropagation of the gradients, must exist to train the network using end-to-end training.

To enable the computation of the second order derivatives using automatic differentiation the forward computation must be performed analytically.

Both derivatives, d LL T /dt and ∂ q T LL T q /∂q i , have closed form solutions and can be derived by first computing the respective derivative of L and second substituting the reshaped derivative of the vectorized form l. For the temporal derivative d LL T /dt this yields DISPLAYFORM0 whereas dL/dt can be substituted with the reshaped form of DISPLAYFORM1 where i refers to the i-th network layer consisting of an affine transformation and the non-linearity g, i.e., DISPLAYFORM2 can be simplified as the network weights W i and biases b i are time-invariant, i.e., dW i /dt = 0 and db i /dt = 0.

Therefore, dl/dt is described by DISPLAYFORM3 Due to the compositional structure of the network and the differentiability of the non-linearity, the derivative with respect to the network input dl/dq can be computed by recursively applying the chain rule, i.e., DISPLAYFORM4 where g is the derivative of the non-linearity.

Similarly to the previous derivation, the partial derivative of the quadratic term can be computed using the chain rule, which yields DISPLAYFORM5 whereas ∂L/∂q i can be constructed using the columns of previously derived ∂l/∂q.

Therefore, all derivatives included withinf can be computed in closed form.

The derivatives of Section 4.2 must be computed within a real-time control loop and only add minimal computational complexity in order to not break the real-time constraint.

l and ∂l/∂q, required within Equation 10 and Equation 14, can be simultaneously computed using an extended standard layer.

Extending the affine transformation and non-linearity of the standard layer with an additional sub-graph for computing ∂h i /∂h i−1 yields the Lagrangian layer described by DISPLAYFORM0 The computational graph of the Lagrangian layer is shown in FIG1 .

Chaining the Lagrangian layer yields the compositional structure of ∂l/∂q (Equation 13) and enables the efficient computation of ∂l/∂q.

Additional reshaping operations compute dL/dt and ∂L/∂q i .

To demonstrate the applicability and extrapolation of DeLaN, the proposed network topology is applied to model-based control for a simulated 2-dof robot FIG2 ) and the physical 7-dof robot Barrett WAM FIG2 ).

The performance of DeLaN is evaluated using the tracking error on train and test trajectories and compared to a learned and analytic model.

This evaluation scheme follows existing work BID34 BID38 as the tracking error is the relevant performance indicator while the mean squared error (MSE)4 obtained using sample based optimization exaggerates model performance BID16 .

In addition to most previous work, we strictly limit all model predictions to real-time and perform the learning online, i.e., the models are randomly initialized and must learn the model during the experiment.

Within the experiment the robot executes multiple desired trajectories with specified joint positions, velocities and accelerations.

The control signal, consisting of motor torques, is generated using a non-linear feedforward controller, i.e., a low gain PD-Controller augmented with a feed-forward torque τ τ τ f f to compensate system dynamics.

The control law is described by DISPLAYFORM0 where K p , K d are the controller gains and q d , q d , q d the desired joint positions, velocities and accelerations.

The control-loop is shown in FIG2 .

For all experiments the control frequency is set to 500Hz while the desired joint state and respectively τ τ τ f f is updated with a frequency of f d = 200Hz.

All feed-forward torques are computed online and, hence, the computation time is strictly limited to T ≤ 1/200s.

The tracking performance is defined as the sum of the MSE evaluated at the sampling points of the reference trajectory.

For the desired trajectories two different data sets are used.

The first data set contains all single stroke characters5 while the second data set uses cosine curves in joint space FIG2 ).

The 20 characters are spatially and temporally re-scaled to comply with the robot kinematics.

The joint references are computed using the inverse kinematics.

Due to the different characters, the desired trajectories contain smooth and sharp turns and cover a wide variety of different shapes but are limited to a small task space region.

In contrast, the cosine trajectories are smooth but cover a large task space region.

The performance of DeLaN is compared to an analytic inverse dynamics model, a standard feedforward neural network (FF-NN) and a PD-Controller.

For the analytic models the torque is computed using the Recursive Newton-Euler algorithm (RNE) BID29 , which computes the feedforward torque using estimated physical properties of the system, i.e. the link dimensions, masses and moments of inertia.

For implementations the open-source library PyBullet (Coumans & Bai, 2016 ) is used.

Both deep networks use the same dimensionality, ReLu nonlinearities and must learn the system dynamics online starting from random initialization.

The training samples containing joint states and applied torques (q, q, q, τ τ τ) 0,...T are directly read from the control loop as shown in FIG2 .4An offline comparisons evaluating the MSE on datasets can be found in the Appendix A. 5The data set was created by BID45 and is available at Dheeru & Karra Taniskidou (2017)) The training runs in a separate process on the same machine and solves the optimization problem online.

Once the training process computed a new model, the inverse modelf −1 of the control loop is updated.

The 2-dof robot shown in FIG2 is simulated using PyBullet and executes the character and cosine trajectories.

FIG3 shows the ground truth torques of the characters 'a', 'd', 'e', the torque ground truth components and the learned decomposition using DeLaN FIG3 .

Even though DeLaN is trained on the super-imposed torques, DeLaN learns to disambiguate the inertial force H q , the Coriolis and Centrifugal force c(q, q) and the gravitational force g(q) as the respective curves overlap closely.

Hence, DeLaN is capable of learning the underlying physical model using the proposed network topology trained with standard end-to-end optimization.

FIG3 shows the offline MSE on the test set averaged over multiple seeds for the FF-NN and DeLaN w.r.t.

to different training set sizes.

The different training set sizes correspond to the combination of n random characters, i.e., a training set size of 1 corresponds to training the model on a single character and evaluating the performance on the remaining 19 characters.

DeLaN clearly obtains a lower test MSE compared to the FF-NN.

Especially the difference in performance increases when the training set is reduced.

This increasing difference on the test MSE highlights the reduced sample complexity and the good extrapolation to unseen samples.

This difference in performance is amplified on the real-time control-task where the models are learned online starting from random initialization.

FIG4 and b shows the accumulated tracking error per testing character and the testing error averaged over all test characters while FIG4 shows the qualitative comparison of the control performance6.

It is important to point out that all shown results are averaged over multiple seeds and only incorporate characters not used for training and, hence, focus the evaluation on the extrapolation to new trajectories.

The qualitative comparison shows that DeLaN is able to execute all 20 characters when trained on 8 random characters.

The obtained tracking error is comparable to the analytic model, which in this case contains the simulation parameters and is optimal.

In contrast, the FF-NN shows significant deviation from the desired trajectories when trained on 8 random characters.

The quantitative comparison of the accumulated tracking error over seeds FIG4 shows that DeLaN obtains lower tracking error on all training set sizes compared to the FF-NN.

This good performance using only few training characters shows that DeLaN has a lower sample complexity and better extrapolation to unseen trajectories compared to the FF-NN.

comparable.

When the velocities are increased the performance of FF-NN deteriorates because the new trajectories do not lie within the vicinity of the training distribution as the domain of the FF-NN is defined as (q, q, q).

Therefore, FF-NN cannot extrapolate to the testing data.

In contrast, the domain of the networksL andĝ composing DeLaN only consist of q, rather than (q, q, q).

This reduced domain enables DeLaN, within limit, to extrapolate to the test trajectories.

The increase in tracking error is caused by the structure off −1 , where model errors to scale quadratic with velocities.

However, the obtained tracking error on the testing trajectories is significantly lower compared to FF-NN.

For physical experiments the desired trajectories are executed on the Barrett WAM, a robot with direct cable drives.

The direct cable drives produce high torques generating fast and dexterous movements but yield complex dynamics, which cannot be modelled using rigid-body dynamics due to the variable stiffness and lengths of the cables7.

Therefore, the Barrett WAM is ideal for testing the applicability of model learning and analytic models8 on complex dynamics.

For the physical experiments we focus on the cosine trajectories as these trajectories produce dynamic movements while character trajectories are mainly dominated by the gravitational forces.

In addition, only the dynamics of the four lower joints are learned because these joints dominate the dynamics and the upper joints cannot be sufficiently excited to retrieve the dynamics parameters.

FIG5 and d show the tracking error on the cosine trajectories using the the simulated Barrett WAM while FIG5 and f show the tracking error of the physical Barrett WAM.

It is important to note, that the simulation only simulates the rigid-body dynamics not including the direct cables drives and the simulation parameters are inconsistent with the parameters of the analytic model.

Therefore, the analytic model is not optimal.

On the training trajectories executed on the physical system the FF-NN performs better compared to DeLaN and the analytic model.

DeLaN achieves slightly better tracking error than the analytic model, which uses the same rigid-body assumptions as DeLaN. That shows DeLaN can learn a dynamics model of the WAM but is limited by the model assumptions of Lagrangian Mechanics.

These assumptions cannot represent the dynamics of the 7The cable drives and cables could be modelled simplistically using two joints connected by massless spring.

8The analytic model of the Barrett WAM is obtained using a publicly available URDF (JHU LCSR, 2018)

We introduced the concept of incorporating a physics prior within the deep learning framework to achieve lower sample complexity and better extrapolation.

In particular, we proposed Deep Lagrangian Networks (DeLaN), a deep network on which Lagrangian Mechanics is imposed.

This specific network topology enabled us to learn the system dynamics using end-to-end training while maintaining physical plausibility.

We showed that DeLaN is able to learn the underlying physics from a super-imposed signal, as DeLaN can recover the contribution of the inertial-, gravitational and centripetal forces from sensor data.

The quantitative evaluation within a real-time control loop assessing the tracking error showed that DeLaN can learn the system dynamics online, obtains lower sample complexity and better generalization compared to a feed-forward neural network.

DeLaN can extrapolate to new trajectories as well as to increased velocities, where the performance of the feedforward network deteriorates due to the overfitting to the training data.

When applied to a physical systems with complex dynamics the bounded representational power of the physics prior can be limiting.

However, this limited representational power enforces the physical plausibility and obtains the lower sample complexity and substantially better generalization.

In future work the physics prior should be extended to represent a wider system class by introducing additional non-conservative forces within the Lagrangian.

The mean squared error averaged of 20 seeds on the training-(a) and test-set (b) of the character trajectories for the two joint robot.

The models are trained offline using n characters and tested using the remaining 20 − n characters.

The training samples are corrupted with white noise, while the performance is tested on noise-free trajectories.

To evaluate the performance of DeLaN without the control task, DeLaN was trained offline on previously collected data and evaluated using the mean squared error (MSE) on the test and training set.

For comparison, DeLaN is compared to the system identification approach (SI) described by BID1 , a feed-forward neural network (FF-NN) and the Recursive Newton Euler algorithm (RNE) using an analytic model.

For this comparison, one must point out that the system identification approach relies on the availability of the kinematics, as the Jacobians and transformations w.r.t.

to every link must be known to compute the necessary features.

In contrast, neither DeLaN nor the FF-NN require this knowledge and must implicitly also learn the kinematics.

FIG6 shows the MSE averaged over 20 seeds on the character data set executed on the two-joint robot.

For this data set, the models are trained using noisy samples and evaluated on the noise-free and previously unseen characters.

The FF-NN performs the best on the training set, but overfits to the training data.

Therefore, the FF-NN does not generalize to unseen characters.

In contrast, the SI approach does not overfit to the noise and extrapolates to previously unseen characters.

In comparison, the structure of DeLaN regularizes the training and prevents the overfitting to the corrupted training data.

Therefore, DeLaN extrapolates better than the FF-NN but not as good as the SI approach.

Similar results can be observed on the cosine data set using the Barrett WAM simulated in SL FIG7 .

The FF-NN performs best on the training trajectory but the performance deteriorates when this network extrapolates to higher velocities.

SI performs worse on the training trajectory but extrapolates to higher velocities.

In comparison, DeLaN performs comparable to the SI approach on the training trajectory, extrapolates significantly better than the FF-NN but does not extrapolate as good as the SI approach.

For the physical system FIG7 , the results differ from the results in simulation.

On the physical system the SI approach only achieves the same performance as RNE, which is significantly worse compared to the performance of DeLaN and the FF-NN.

When evaluating the extrapolation to higher velocities, the analytic model and the SI approach extrapolate to higher velocities, while the MSE for the FF-NN significantly increases.

In comparison, DeLaN extrapolates better compared to the FF-NN but not as good as the analytic model or the SI approach.

This performance difference between the simulation and physical system can be explained by the underlying model assumptions and the robustness to noise.

While DeLaN only assumes rigidbody dynamics, the SI approach also assumes the exact knowledge of the kinematic structure.

For simulation both assumptions are valid.

However, for the physical system, the exact kinematics are unknown due to production imperfections and the direct cable drives applying torques to flexible joints violate the rigid-body assumption.

Therefore, the SI approach performs significantly worse on the physical system.

Furthermore, the noise robustness becomes more important for the physical system due to the inherent sensor noise.

While the linear regression of the SI approach is easily corrupted by noise or outliers, the gradient based optimization of the networks is more robust to noise.

This robustness can be observed in Figure 9 , which shows the correlation between the variance of Gaussian noise corrupting the training data and the MSE of the simulated and noise-free cosine trajectories.

With increasing noise levels, the MSE of the SI approach increases significantly faster compared to the models learned using gradient descent.

Concluding, the extrapolation of DeLaN to unseen trajectories and higher velocities is not as good as the SI approach but significantly better than the generic FF-NN.

This increased extrapolation compared to the generic network is achieved by the Lagrangian Mechanics prior of DeLaN. Even though this prior promotes extrapolation, the prior also hinders the performance on the physical robot, because the prior cannot represent the dynamics of the direct cable drives.

Therefore, DeLaN performs worse than the FF-NN, which does not assume any model structure.

However, DeLaN outperforms the SI approach on the physical system, which also assumes rigid-body dynamics and requires the exact knowledge of the kinematics.

@highlight

This paper introduces a physics prior for Deep Learning and applies the resulting network topology for model-based control.