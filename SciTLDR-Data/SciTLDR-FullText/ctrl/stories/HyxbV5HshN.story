This paper explores many immediate connections between adaptive control and machine learning, both through common update laws as well as common concepts.

Adaptive control as a field has focused on mathematical rigor and guaranteed convergence.

The rapid advances in machine learning on the other hand have brought about a plethora of new techniques and problems for learning.

This paper elucidates many of the numerous common connections between both fields such that results from both may be leveraged together to solve new problems.

In particular, a specific problem related to higher order learning is solved through insights obtained from these intersections.

The fields of adaptive control and machine learning have evolved in parallel over the past few decades, with a significant overlap in goals, problem statements, and tools.

Machine learning as a field has focused on computer based systems that improve through experience BID16 BID6 BID24 BID17 BID21 BID35 .

Often times the process of learning is encapsulated in the form of a parameterized model, whose parameters are learned in order to approximate a function.

Optimization methods are commonly employed to reduce the function approximation error using any and all available data.

The field of adaptive control, on the other hand, has focused on the process of controlling engineering systems in order to accomplish regulation and tracking of critical variables of interest (e.g. position and force in robotics, Mach number and altitude in aerospace systems, frequency and voltage in power systems) in the presence of uncertainties in the underlying system models, changes in the environment, and unforeseen variations in the overall infrastructure BID48 BID59 Åström & Wittenmark, 1995; BID33 BID49 .

The approach used for accomplishing such regulation and tracking in adaptive control is the learning of underlying parameters through an online estimation algorithm.

Stability theory is employed for enabling guarantees for the safe evolution of the critical variables, and convergence of the regulation and tracking errors to zero.

Learning parameters of a model in both machine learning and adaptive control occurs through the use of input-output data.

In both cases, the main algorithm used for updating the parameters is often based on a gradient descent-like algorithm.

Related tools of analysis, convergence, and robustness in both fields have a tremendous amount of similarity.

As the scope of problems in both fields increases, the associated complexity and challenges increase as well.

Therefore it is highly attractive to understand these similarities and connections so that the two communities can develop new methods for addressing new challenges.

Two types of error models are common in machine learning and adaptive control, where output errors e y may be related to regressors (features) φ and parameter errorsθ as DISPLAYFORM0 DISPLAYFORM1 where W (s) denotes a dynamic operator andθ = θ − θ * (θ * unknown).

Our goal with both perspectives will be to adjust a parameter θ with knowledge of the regressor φ and output error e y , such that a loss function L(θ; e y ) is minimized.

For the adaptive control perspective we present solutions in terms of gradient flow in continuous time t while the machine learning updates are presented as gradient descent in discrete time steps indexed by k, i.e., DISPLAYFORM2 where γ > 0 is the learning rate in gradient flow and γ k is the step size in gradient descent.

For a more detailed discussion of the problem statements see Appendix A. In this section we consider the question: What common modifications to the update laws in (3) and (4) have been developed?

While the update laws in (3) and (4) are designed primarily to reduce the output error e y , there are several secondary reasons to modify these update laws from robustness considerations due to perturbations stemming from disturbances, noise, and other unmodeled causes.

Historically the adaptive update law in (3) has been modified to ensure robustness to bounded disturbances aṡ DISPLAYFORM0 where σ > 0 is a tuneable parameter that scales the extra term G. Common choices for G include the σ-modification G = θ BID32 , and the e-modification G = e y θ BID47 .Regularization is often included in a machine learning optimization problem in order to help cope with overfitting by including constraints on the parameter, thus resulting in an augmented loss function BID24 BID12 : DISPLAYFORM1 where σ > 0 is a tuneable parameter, often referred to as a Lagrange multiplier.

The gradient descent update (4) for this augmented loss function is often referred to as the "regularized follow the leader" algorithm in online learning BID25 and may be expressed as DISPLAYFORM2 The common choice of 2 regularization in machine learning of R = (1/2) θ 2 2 , can be seen to coincide with the σ-modification BID32 , as ∇ θ R = G.

This subsection details common modifications of the update laws in both fields adopted to cease updating the parameter estimate after sufficient tuning.

Another method in adaptive control employed to increase robustness in the presence of bounded disturbances is to employ a "dead-zone" BID54 , for the update in (3): DISPLAYFORM0 where d 0 > 0 is the dead-zone width that may correspond to an upper bound on the disturbance, and > 0 is a small constant.

The function D is a non-negative metric on the output error to stop adaptation in desired regions of the output space.

A common choice is D = e y such that adaptation stops after a small output error is achieved above a noise level with upper bound d 0 .The training processes is often stopped in machine learning applications as a method to deal with overfitting BID24 ).

This may be done by using multiple data sets and stopping the parameter update process (4) when the loss computed for a validation data set starts to increase BID56 .

Early stopping is often seen to be needed for training neural networks due to their large number of parameters BID21 and can act as regularization BID61 .

It is often desirable to define a compact region a priori for the parameters θ, such that during the learning process the parameters are not allowed to leave that region.

In physical systems there are natural constraints which may aid in the design of that region, and for non-physical systems, the constraints are often engineered by the algorithm designer.

The continuous projection algorithm, commonly employed in adaptive control for increased robustness to unmodeled dynamics BID39 BID41 BID31 , is defined as DISPLAYFORM0 where Ω, θ i,max , θ i,max define a user-specified boundary layer region inside of a compact convex set Θ. The update law in (3) may then be modified aṡ DISPLAYFORM1 The following projection operation commonly used in online learning BID72 BID26 BID27 BID25 finds the closest point in a convex set DISPLAYFORM2 which may be employed in the update law (4) modified as DISPLAYFORM3

The following parameter update law is one example which alters the gain of the standard update law (3) as a function of the time varying regressors φ (Narendra BID48 BID33 : DISPLAYFORM0 where Υ ≥ 0 is a forgetting factor and N (t) is a normalizing signal, with common choice N (t) = (1 + µφ T (t)φ(t)).

It can be seen that the update for Γ may be used in the update for θ to result in a gain adaptive to the regressor φ.

Adaptive step size methods BID15 BID71 BID38 BID57 have seen widespread use in machine learning problems due to their ability to handle sparse and small gradients by adjusting the step size as a function of features as they are processed online.

A common update law for adaptive step size methods BID57 can then be seen to be similar to (11) as DISPLAYFORM1 where m k and V k are functions of previous gradients, which can be compared to normalization by the regressor in (12).

In this section we consider concepts and tools common to machine learning and adaptive control.

Stability and convergence tools in adaptive control and online machine learning are analyzed in this section.

Suppose we consider the error model in (2) where W (s) = c(sI − A) −1 b, and a corresponding state space representation of the form BID48 DISPLAYFORM0 The termφ is due to exponentially decaying terms in the regressor φ.

That is,φ =φ − φ andφ = Λφ for a Hurwitz matrix Λ ∈ R N ×N .

1 Stability is often proven in adaptive control by the use of a Lyapunov function V , such as DISPLAYFORM1 Note that the last two terms in V are not needed for the algebraic error model in (1).

The time derivative of the Lyapunov function may then be stated using the update law in (3) and the KYP lemma asV = −e T Qe − αφ TQφ + 2e DISPLAYFORM2 1 This formulation is common in the design of non-minimal adaptive observers BID48 .

It can be noted thatφ → φ as t → ∞ as Λ is Hurwitz.

Also forφ = φ, (14) is the same as (2).

A Hurwitz matrix Λ implies the existence of a positive definite matrixP DISPLAYFORM3 In online learning, efficiency of an algorithm is often analyzed using the notion of "regret" as DISPLAYFORM4 where regret can be seen to correspond to the sum of the time varying convex costs C k associated with the choice of the time varying parameter estimate θ k , minus the cost associated with the best static parameter estimate choice, over a time horizon of T steps BID72 BID26 BID27 BID25 .

Suppose we consider a quadratic cost DISPLAYFORM5 A continuous time limit of (17) leads to an integral of the form FORMULA0 whereδ(t) is an exponentially decaying signal which is due to nonzero initial conditions in (2) or similarly in (14).

A strong similarity can thus be seen between FORMULA0 and (18).

DISPLAYFORM6 It is desired to have regret grow sub-linearly with time, such that average regret, (1/T )regret T , goes to zero in the limit T → ∞, to provide for an efficient algorithm BID25 .

For adaptive control, convergence of state/output errors is shown from a similar integral which is akin to constant regret upper bounded by V (t 0 ) in (16).

Models used to design adaptive controllers, including the examples of (1) and (2), are approximations with a certain amount of modeling errors.

As such, they may only hold about an operating point and need to contend with unmodeled dynamics.

This implies that any stabilizing controller must be designed to not only adapt to parametric uncertainties, but also be robust to unmodeled dynamics.

In addition, constraints on the state and input may also be present in adaptive control problems BID36 BID1 .

Analysis becomes more complicated when considering unmodeled dynamics and constraints, resulting in non-global guarantees.

Many of the update law modifications in adaptive control from Section 2 were initially derived to ensure robustness in such cases.

This same notion of robustness to modeling errors exists in machine learning in which an estimator is constructed from a finite training data set.

It is then desired that this estimator produces a low prediction error based on a test data set consisting of unseen data.

Generalization thus refers to the concept of a designed estimator having low loss when applied to new problems.

In particular it can be seen that in specific cases, generalization pertains to stability, where algorithms that are stable and train in a small amount of time result in a small generalization error BID8 BID23 .

Persistent excitation (PE) of the system regressor in adaptive control is a condition that has been shown to be necessary and sufficient for parameter convergence BID34 .

It can be shown that if the regressor φ is persistently exciting, then the parameter estimation errorθ(t) converges to zero uniformly in time BID48 .

The PE condition essentially corresponds to certain spectral conditions being satisfied by the regressor BID9 .

A detailed exposition of system identification and parameter convergence in both deterministic and stochastic cases can be found in BID22 BID0 BID45 BID46 BID42 .

Another way to think of the PE condition is that it leads to a perfect test error, since it provides for convergence of the parameter error to zero, and therefore zero output error once transients decay to zero.

Many machine learning problems consider the case when stochastic perturbations are present.

In this context, significant improvements may be possible by leveraging well known concepts in system identification BID42 .

For example BID13 purposely includes a Gaussian random input into a dynamical system in order to provide for PE by construction.

Such stochastic perturbations can guarantee a PE condition only in the limit, when infinite samples can be obtained.

In order to address the realistic case of finite samples, approaches in machine learning algorithms for system identification and control have attempted to obtain performance bounds with probability 1 − p f for p f ∈ (0, 1), where the bound usually scales inversely with p f .

The probability of failure given by the choice of p f allows for error due to the presence of finite samples.

Gradient based methods to solve for estimates of unknown parameters via back propagation, in what would develop into the foundations of neural networks have been used for decades in control, with early examples consisting of finding optimal trajectories BID55 in flight control BID37 , and resource allocation problems BID11 ) (see BID14 ) for a brief history).

Since then, the use of neural networks in control systems has expanded to include stabilizing nonlinear dynamical systems BID43 .

Design and analysis of stable controllers based on neural networks was taken up by the adaptive control community due to the the similarities of gradient-like update laws used in neural networks and adaptive control.

The adaptive control community developed a well established literature for the use of neural networks in nonlinear dynamical systems in the 1990s BID43 BID50 BID51 BID69 BID56 .The use of neural networks in the machine learning community greatly expanded as of recent due to the increase in computing power available and an increase in applications BID40 BID63 BID21 .

Recurrent neural networks BID30 BID28 BID29 , while often similar in structure to nonlinear dynamical systems, have historically been trained in a manner similar to feed-forward neural networks BID58 ) using back propagation through time BID65 .

While a theoretical understanding of why deep neural networks work as well as they do for given problems has been lacking, the machine learning community has worked to rigorously analyze sub-classes of deep neural network architectures such as deep linear networks BID2 BID3 .

The update laws employed in training deep neural networks often include selections of modifications of the update laws as discussed in Section 2 BID60 .

Higher Order LearningGiven the many similarities in problem statements, tools, concepts, and algorithms, we now demonstrate how methods from the field of adaptive control can be used to solve a new problem related to higher-order learning.

Many of the update laws addressed thus far were first-order in nature, and made use of gradient-like quantities for learning.

A question of increasing interest is when accelerated learning can occur for higher-order learning methods.

In particular, Nesterov's accelerated method BID52 was able to certify a convergence rate of O(1/k 2 ) as compared to the standard gradient descent (4) rate of O(1/k) for a class of convex functions.

A parameterization of Nesterov's higher order method may be stated as DISPLAYFORM0 where β > 0 is a design parameter that weighs the effect of past parameters.

Continuous time problem formulations have been explored in BID62 BID66 , with rate-matching discretizations established in BID67 BID5 .

Many of these methods however become inadequate for time varying inputs and features.

Adaptive update laws which include additional levels of integration appeared in the "higher order tuners" in BID44 BID18 , and take the forṁ 20) where N t (1 + µφ T (t)φ(t)) for a µ > 0.

In contrast to (19), the update law in (20) can be shown to be stable in the presence of time varying regressors as in (1) and as well as in adaptive control applications with error model as in (2) BID19 ).

This solution was only possible by leveraging techniques from the field of adaptive control.

DISPLAYFORM1

In this section, we state typical problems that are addressed in the areas of adaptive control and machine learning.

In both cases, we illustrate the role of learning, the input-output data used, and the overall problem that is desired to be solved.

The main goal in adaptive control is to carry out problems such as estimation or tracking in the presence of parametric uncertainties.

The underlying model that relates inputs, outputs, and the unknown parameters is assumed to stem from either the underlying physics or from data-driven approaches.

Often these models take the form DISPLAYFORM0 DISPLAYFORM1 where u ∈ R m is an exogenous input, x ∈ R n denotes the state, y ∈ R p corresponds to output measurements, φ ∈ R N corresponds to measured and computed variables, and θ * ∈ R N denotes the uncertain parameter.

In an estimation problem, the goal is to estimate the state x in (22) and output y in both (21), (22), alongside the unknown parameter θ * simultaneously, using all available variables.

In a control problem, the goal is to determine a control input u so that the output y in (22) follows a desired outputŷ.

A typical approach taken in order to solve the estimation problem in (21) is to choose an estimator structure of the formŷ DISPLAYFORM2 where θ ∈ R N denotes the estimate of θ * and adjust θ so that the estimation error e y =ŷ − y is minimized, i.e., choose a function g 1 (e y , φ) witḣ DISPLAYFORM3 so that the estimator has bounded signals, e y (t) converges to zero and θ(t) converges to θ * .

Similarly, the control problem consists of constructing an output tracking error e y =ŷ − y, whereŷ denotes the desired output that y is required to track.

The goal is to then choose functions g 2 (e y , φ, θ) and g 3 (e y , φ, θ) so that the control input u and parameter estimate θ can be chosen as leading to closed-loop signals remaining bounded, e y (t) converging to zero and θ(t) converging to its true value θ * .

Denote the corresponding parameter errors asθ = θ − θ * .

DISPLAYFORM4 In order to derive the function g 1 for the estimation problem in (21) and the functions g 2 and g 3 for the control problem in (22) so as to realize the underlying goals, a stability framework together with an error model approach is often employed in adaptive control.

The error model approach consists of identifying the basic relationship between the two errors that are commonly present in these adaptive systems, which are the estimation (or tracking) error e y and the parameter errorθ.

While the estimation error is measurable and correlated with the parameter error, the parameter error is unknown but adjustable through the parameter estimate.

In order to determine the update laws g i , the relationship (error model) that relates these two errors is used as a cue.

Two types of error models frequently occur in adaptive systems, and are presented below (see FIG1 .

The first corresponds to the case when the relation in FORMULA0 is linear, and the underlying error model is simply of the form (cf.

BID49 ) DISPLAYFORM5 and as a result, the function g 1 in (24) can be determined simply using the gradient rule that minimizes e y 2 .

The second is of the form (cf.

BID49 ) DISPLAYFORM6 where W (s)[ζ] denotes a dynamic operator operating on ζ(t).

It has been shown in the adaptive control literature BID48 BID59 BID4 BID33 BID49 ) that for specific classes of dynamic operators W (s), a stable, gradient-like rule can be determined for adjustingθ.

Most of these results apply uniformly to the case when u and y are scalars or vectors, with the latter introducing additional technicalities.

In this paper we consider the case where inputs and outputs are scalars for notational simplicity, and to focus on the core of the learning problem with multi-dimensional regressors φ and parameter estimates θ.

Often the unknown parameter θ * is assumed to reside in a compact convex set, which we will denote as Θ.

Machine learning is a broad field encompassing a wide variety of learning techniques and problems such as classification and regression.

A large portion of machine learning considers supervised learning problems, where regressors φ and outputs y are related to one another in an unknown algebraic manner BID16 BID6 BID24 BID17 BID21 BID35 .

A typical approach taken in order to perform classification or regression is to choose an output estimatorŷ k parameterized with adjustable weights θ k aŝ DISPLAYFORM0 A common form of the estimator as in FORMULA1 is that of neural networks, where the parameters θ k represent the adjustable weights in the network BID16 BID6 BID24 BID17 BID21 .Similar to adaptive control, θ k is often adjusted using the output error e y,k =ŷ k − y k .

A loss function L : Θ → R of e y,k is minimized through the adjustable weights.

An example loss function for regression is p loss (with p ∈ N, p > 0 and even) L(θ k ) = (1/p) e y,k p p .

For binary classification (y k ∈ {−1, 1}) common loss functions include hinge loss L(θ k ) = max(0, 1 − y kŷk ), and logistic loss L(θ k ) = ln(1 + exp(−y kŷk )).

Additionally, as in empirical risk minimization (ERM) BID64 , the total loss function considered for the purpose of a parameter update may be an average of loss functions over m samples as: DISPLAYFORM1 The above descriptions make it clear that the structure of the estimation problem in both adaptive control and machine learning are strikingly similar.

In the next section, we examine the nature of the adjustment of θ k .

As previously stated, the goal in adaptive control is to design a rule to adjust θ in an online continuous manner using knowledge of φ and e y such that e y tends toward zero.

Given that the output errors may be corrupted by noise, an iterative, gradient-like update is usually employed.

To do so for the algebraic error model (26), consider the squared loss cost function: L(θ(t)) = (1/2)e 2 y (t).

The gradient of this function with respect to the parameters can be expressed as: ∇ θ L(θ(t)) = φ(t)e y (t).

The standard gradient flow update law BID48 ) may be expressed as follows with user-designed gain parameter γ > 0 aṡ θ(t) = −γ∇ θ L(θ(t)) = −γφ(t)e y (t).For dynamical error models such as (27), a stability approach rather than a gradient based one is taken using Lyapunov methods, which leads to an adaptive law identical to (29) for a class of dynamic systems W (s) that are strictly positive real BID48 BID53 .The common update law for supervised machine learning problems, gradient descent 2 , is akin to the time varying regression law (29) in discrete time, and of the form DISPLAYFORM0 where the "stepsize" γ k is usually chosen as a decreasing function of time BID26 BID27 BID25 BID12 BID72 , a standard feature of stochastic gradient algorithms.

<|TLDR|>

@highlight

History of parallel developments in update laws and concepts between adaptive control and optimization in machine learning.