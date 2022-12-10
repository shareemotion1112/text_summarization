Actor-critic methods solve reinforcement learning problems by updating a parameterized policy known as an actor in a direction that increases an estimate of the expected return known as a critic.

However, existing actor-critic methods only use values or gradients of the critic to update the policy parameter.

In this paper, we propose a novel actor-critic method called the guide actor-critic (GAC).

GAC firstly learns a guide actor that locally maximizes the critic and then it updates the policy parameter based on the guide actor by supervised learning.

Our main theoretical contributions are two folds.

First, we show that GAC updates the guide actor by performing second-order optimization in the action space where the curvature matrix is based on the Hessians of the critic.

Second, we show that the deterministic policy gradient method is a special case of GAC when the Hessians are ignored.

Through experiments, we show that our method is a promising reinforcement learning method for continuous controls.

The goal of reinforcement learning (RL) is to learn an optimal policy that lets an agent achieve the maximum cumulative rewards known as the return BID31 .

Reinforcement learning has been shown to be effective in solving challenging artificial intelligence tasks such as playing games BID20 and controlling robots BID6 .Reinforcement learning methods can be classified into three categories: value-based, policy-based, and actor-critic methods.

Value-based methods learn an optimal policy by firstly learning a value function that estimates the expected return.

Then, they infer an optimal policy by choosing an action that maximizes the learned value function.

Choosing an action in this way requires solving a maximization problem which is not trivial for continuous controls.

While extensions to continuous controls were considered recently, they are restrictive since specific structures of the value function are assumed BID10 BID3 .On the other hand, policy-based methods, also called policy search methods BID6 , learn a parameterized policy maximizing a sample approximation of the expected return without learning the value function.

For instance, policy gradient methods such as REIN-FORCE BID34 use gradient ascent to update the policy parameter so that the probability of observing high sample returns increases.

Compared with value-based methods, policy search methods are simpler and naturally applicable to continuous problems.

Moreover, the sample return is an unbiased estimator of the expected return and methods such as policy gradients are guaranteed to converge to a locally optimal policy under standard regularity conditions BID32 .

However, sample returns usually have high variance and this makes such policy search methods converge too slowly.

Actor-critic methods combine the advantages of value-based and policy search methods.

In these methods, the parameterized policy is called an actor and the learned value-function is called a critic.

The goal of these methods is to learn an actor that maximizes the critic.

Since the critic is a low variance estimator of the expected return, these methods often converge much faster than policy search methods.

Prominent examples of these methods are actor-critic BID32 BID15 , natural actor-critic BID25 , trust-region policy optimization BID27 , and asynchronous advantage actor-critic BID21 .

While their approaches to learn the actor are different, they share a common property that they only use the value of the critic, i.e., the zero-th order information, and ignore higher-order ones such as gradients and Hessians w.r.t.

actions of the critic 1 .

To the best of our knowledge, the only actor-critic methods that use gradients of the critic to update the actor are deterministic policy gradients (DPG) BID29 and stochastic value gradients .

However, these two methods do not utilize the second-order information of the critic.

In this paper, we argue that the second-order information of the critic is useful and should not be ignored.

A motivating example can be seen by comparing gradient ascent to the Newton method: the Newton method which also uses the Hessian converges to a local optimum in a fewer iterations when compared to gradient ascent which only uses the gradient BID24 .

This suggests that the Hessian of the critic can accelerate actor learning which leads to higher data efficiency.

However, the computational complexity of second-order methods is at least quadratic in terms of the number of optimization variables.

For this reason, applying second-order methods to optimize the parameterized actor directly is prohibitively expensive and impractical for deep reinforcement learning which represents the actor by deep neural networks.

Our contribution in this paper is a novel actor-critic method for continuous controls which we call guide actor-critic (GAC).

Unlike existing methods, the actor update of GAC utilizes the secondorder information of the critic in a computationally efficient manner.

This is achieved by separating actor learning into two steps.

In the first step, we learn a non-parameterized Gaussian actor that locally maximizes the critic under a Kullback-Leibler (KL) divergence constraint.

Then, the Gaussian actor is used as a guide for learning a parameterized actor by supervised learning.

Our analysis shows that learning the mean of the Gaussian actor is equivalent to performing a second-order update in the action space where the curvature matrix is given by Hessians of the critic and the step-size is controlled by the KL constraint.

Furthermore, we establish a connection between GAC and DPG where we show that DPG is a special case of GAC when the Hessians and KL constraint are ignored.

In this section, we firstly give a background of reinforcement learning.

Then, we discuss existing second-order methods for policy learning and their issue in deep reinforcement learning.

We consider discrete-time Markov decision processes (MDPs) with continuous state space S Ď R ds and continuous action space A Ď R da .

We denote the state and action at time step t P N by s t and a t , respectively.

The initial state s 1 is determined by the initial state density s 1 " ppsq.

At time step t, the agent in state s t takes an action a t according to a policy a t " πpa|s t q and obtains a reward r t " rps t , a t q. Then, the next state s t`1 is determined by the transition function s t`1 " pps 1 |s t , a t q. A trajectory τ " ps 1 , a 1 , r 1 , s 2 , . . .

q gives us the cumulative rewards or return defined as ř 8 t"1 γ t´1 rps t , a t q, where the discount factor 0 ă γ ă 1 assigns different weights to rewards given at different time steps.

The expected return of π after executing an action a in a state s can be expressed through the action-value function which is defined as DISPLAYFORM0 where E p r¨s denotes the expectation over the density p and the subscript t ě 1 indicates that the expectation is taken over the densities at time steps t ě 1.

We can define the expected return as DISPLAYFORM1 where G P R d θˆdθ is a curvature matrix.

The behavior of second-order methods depend on the definition of a curvature matrix.

The most well-known second-order method is the Newton method where its curvature matrix is the Hessian of the objective function w.r.t.

the optimization variables: DISPLAYFORM2 The natural gradient method is another well-known second-order method which uses the Fisher information matrix (FIM) as the curvature matrix (Amari, 1998): DISPLAYFORM3 Unlike the Hessian matrix, FIM provides information about changes of the policy measured by an approximated KL divergence: E ppsq rKLpπ θ pa|sq||π θ 1 pa|sqqs « pθ´θ 1 q J G FIM pθ´θ 1 q BID13 .

We can see that G Hessian and G FIM are very similar but the former also contains the critic and the Hessian of the actor while the latter does not.

This suggests that the Hessian provides more information than that in FIM.

However, FIM is always positive semi-definite while the Hessian may be indefinite.

Please see BID7 for detailed comparisons between the two curvature matrices in policy search 4 .

Nonetheless, actor-critic methods based on natural gradient were shown to be very efficient BID25 BID27 ).We are not aware of existing work that considers second-order updates for DPG or SVG.

However, their second-order updates can be trivially derived.

For example, a Newton update for DPG is DISPLAYFORM4 where the pi, jq-th entry of the Hessian matrix DISPLAYFORM5 Note that Bπ θ psq{Bθ and B 2 π θ psq{BθBθ 1 are vectors since π θ psq is a vector-valued function.

Interestingly, the Hessian of DPG contains the Hessians of the actor and the critic.

In contrast, the Hessian of the actor-critic method contains the Hessian of the actor and the value of the critic.

Second-order methods are appealing in reinforcement learning because they have high data efficiency.

However, inverting the curvature matrix (or solving a linear system) requires cubic computational complexity in terms of the number of optimization variables.

For this reason, the secondorder updates in Eq.(8) and Eq.(11) are impractical in deep reinforcement learning due to a large number of weight parameters in deep neural networks.

In such a scenario, an approximation of the curvature matrix is required to reduce the computational burden.

For instance, BID7 proposed to use only diagonal entries of an approximated Hessian matrix.

However, this approximation clearly leads to a loss of useful curvature information since the gradient is scaled but not rotated.

More recently, BID35 proposed a natural actor-critic method that approximates block-diagonal entries of FIM.

However, this approximation corresponds to ignoring useful correlations between weight parameters in different layers of neural networks.

In this section, we propose the guide actor-critic (GAC) method that performs second-order updates without the previously discussed computational issue.

Unlike existing methods that directly learn the parameterized actor from the critic, GAC separates the problem of learning the parameterized actor into problems of 1) learning a guide actor that locally maximizes the critic, and 2) learning a parameterized actor based on the guide actor.

This separation allows us to perform a second-order update for the guide actor where the dimensionality of the curvature matrix is independent of the parameterization of the actor.

We formulate an optimization problem for learning the guide actor in Section 3.1 and present its solution in Section 3.2.

Then in Section 3.3 and Section 3.4, we show that the solution corresponds to performing second-order updates.

Finally, Section 3.5 presents the learning step for the parameterized actor using supervised learning.

The pseudo-code of our method is provided in Appendix B and the source code is available at https://github.com/voot-t/guide-actor-critic.

Our first goal is to learn a guide actor that maximizes the critic.

However, greedy maximization should be avoided since the critic is a noisy estimate of the expected return and a greedy actor may change too abruptly across learning iterations.

Such a behavior is undesirable in real-world problems, especially in robotics BID6 .

Instead, we maximize the critic with additional constraints: DISPLAYFORM0 subject to E p β psq rKLpπpa|sq||π θ pa|sqqs ď , DISPLAYFORM1 whereπpa|sq is the guide actor to be learned, π θ pa|sq is the current parameterized actor that we want to improve upon, and p β psq is the state distribution induced by past trajectories.

The objective function differs from the one in Eq.(5) in two important aspects.

First, we maximize for a policy functionπ and not for the policy parameter.

This is more advantageous than optimizing for a policy parameter since the policy function can be obtained in a closed form, as will be shown in the next subsection.

Second, the expectation is defined over a state distribution from past trajectories and this gives us off-policy methods with higher data efficiency.

The first constraint is the Kullback-Leibler (KL) divergence constraint where KLpppxq||qpxqq " E ppxq rlog ppxq´log qpxqs.

The second constraint is the Shannon entropy constraint where Hpppxqq "´E ppxq rlog ppxqs.

The KL constraint is commonly used in reinforcement learning to prevent unstable behavior due to excessively greedy update BID25 BID26 BID16 BID27 .

The entropy constraint is crucial for maintaining stochastic behavior and preventing premature convergence BID36 BID0 BID21 BID11 .

The final constraint ensures that the guide actor is a proper probability density.

The KL bound ą 0 and the entropy bound´8 ă κ ă 8 are hyper-parameters which control the explorationexploitation trade-off of the method.

In practice, we fix the value of and adaptively reduce the value of κ based on the current actor's entropy, as suggested by BID0 .

More details of these tuning parameters are given in Appendix C.This optimization problem can be solved by the method of Lagrange multipliers.

The solution is DISPLAYFORM2 where η ‹ ą 0 and ω ‹ ą 0 are dual variables corresponding to the KL and entropy constraints, respectively.

The dual variable corresponding to the probability density constraint is contained in the normalization term and is determined by η ‹ and ω ‹ .

These dual variables are obtained by minimizing the dual function: DISPLAYFORM3 All derivations and proofs are given in Appendix A. The solution in Eq. FORMULA0 tells us that the guide actor is obtained by weighting the current actor with p Qps, aq.

If we set Ñ 0 then we haveπ « π θ and the actor is not updated.

On the other hand, if we set Ñ 8 then we havẽ πpa|sq 9 expp p Qps, aq{ω ‹ q, which is a softmax policy where ω ‹ is the temperature parameter.

Computingπpa|sq and evaluating gpη, ωq are intractable for an arbitrary π θ pa|sq.

We overcome this issue by imposing two assumptions.

First, we assume that the actor is the Gaussian distribution: DISPLAYFORM0 where the mean φ θ psq and covariance Σ θ psq are functions parameterized by a policy parameter θ.

Second, we assume that Taylor's approximation of p Qps, aq is locally accurate up to the secondorder.

More concretely, the second-order Taylor's approximation using an arbitrary action a 0 is given by DISPLAYFORM1 where g 0 psq " ∇ a p Qps, aq| a"a0 and H 0 psq " ∇ 2 a p Qps, aq| a"a0 are the gradient and Hessian of the critic w.r.t.

a evaluated at a 0 , respectively.

By assuming that the higher order term Op}a} 3 q is sufficiently small, we can rewrite Taylor's approximation at a 0 as DISPLAYFORM2 where ψ 0 psq " g 0 psq´H 0 psqa 0 and ξ 0 psq " DISPLAYFORM3 Note that H 0 psq, ψ 0 psq, and ξ 0 psq depend on the value of a 0 and do not depend on the value of a. This dependency is explicitly denoted by the subscript.

The choice of a 0 will be discussed in Section 3.3.Substituting the Gaussian distribution and Taylor's approximation into Eq. FORMULA0 yields another Gaussian distributionπpa|sq " N pa|φ`psq, Σ`psqq, where the mean and covariance are given by DISPLAYFORM4 The matrix F psq P R daˆda and vector Lpsq P R da are defined as DISPLAYFORM5 The dual variables η ‹ and ω ‹ are obtained by minimizing the following dual function: DISPLAYFORM6 where F η psq and L η psq are defined similarly to F psq and Lpsq but with η instead of η ‹ .The practical advantage of using the Gaussian distribution and Taylor's approximation is that the guide actor can be obtained in a closed form and the dual function can be evaluated through matrixvector products.

The expectation over p β psq can be approximated by e.g., samples drawn from a replay buffer BID20 .

We require inverting F η psq to evaluate the dual function.

However, these matrices are computationally cheap to invert when the dimension of actions is not large.

As shown in Eq.(19), the mean and covariance of the guide actor is computed using both the gradient and Hessian of the critic.

Yet, these computations do not resemble second-order updates discussed previously in Section 2.2.

Below, we show that for a particular choice of a 0 , the mean computation corresponds to a second-order update that rotates gradients by a curvature matrix.

For now we assume that the critic is an accurate estimator of the true action-value function.

In this case, the quality of the guide actor depends on the accuracy of sample approximation in p gpη, ωq and the accuracy of Taylor's approximation.

To obtain an accurate Taylor's approximation of p Qps, aq using an action a 0 , the action a 0 should be in the vicinity of a. However, we did not directly use any individual a to compute the guide actor, but we weight π θ pa|sq by expp p Qps, aqq (see Eq. FORMULA0 ).

Thus, to obtain an accurate Taylor's approximation of the critic, the action a 0 needs to be similar to actions sampled from π θ pa|sq.

Based on this observation, we propose two approaches to perform Taylor's approximation.

Taylor's approximation around the mean.

In this approach, we perform Taylor's approximation using the mean of π θ pa|sq.

More specifically, we use a 0 " E π θ pa|sq ras " φ θ psq for Eq.(18).

In this case, we can show that the mean update in Eq.(19) corresponds to performing a second-order update in the action space to maximize p Qps, aq: DISPLAYFORM0 where DISPLAYFORM1 Qps, aq| a"φ θ psq .

This equivalence can be shown by substitution and the proof is given in Appendix A.2.

This update equation reveals that the guide actor maximizes the critic by taking a step in the action space similarly to the Newton method.

However, the main difference lies in the curvature matrix where the Newton method uses Hessians H φ θ psq but we use a damped Hessian F φ θ psq.

The damping term η ‹ Σ´1 θ psq corresponds to the effect of the KL constraint and can be viewed as a trust-region that controls the step-size.

This damping term is particularly important since Taylor's approximation is accurate only locally and we should not take a large step in each update BID24 .Expectation of Taylor's approximations.

Instead of using Taylor's approximation around the mean, we may use an expectation of Taylor's approximation over the distribution.

More concretely, we define r Qps, aq to be an expectation of p Q 0 ps, aq over π θ pa 0 |sq: DISPLAYFORM2 Note that E π θ pa0|sq rH 0 psqs " E π θ pa0|sq r∇ 2 a p Qps, aq| a"a0 s and the expectation is computed w.r.t.

the distribution π θ of a 0 .

We use this notation to avoid confusion even though π θ pa 0 |sq and π θ pa|sq are the same distribution.

When Eq. FORMULA1 is used, the mean update does not directly correspond to any second-order optimization step.

However, under an (unrealistic) independence assumption E π θ pa0|sq rH 0 psqa 0 s " E π θ pa0|sq rH 0 psqsE π θ pa0|sq ra 0 s, we can show that the mean update corresponds to the following second-order optimization step: DISPLAYFORM3 where E π θ pa0|sq rF 0 psqs " η ‹ Σ´1 θ psq´E π θ pa0|sq rH 0 psqs.

Interestingly, the mean is updated by rotating an expected gradient using an expected Hessians.

In practice, the expectations can be approximated using sampled actions ta 0,i u S i"1 " π θ pa|sq.

We believe that this sampling can be advantageous for avoiding local optima.

Note that when the expectation is approximated by a single sample a 0 " π θ pa|sq, we obtain the update in Eq.(24) regardless of the independence assumption.

In the remainder, we use F psq to denote both of F φ θ psq and E π θ pa0|sq rF 0 psqs, and use Hpsq to denote both of H φ θ psq and E π θ pa0|sq rH 0 psqs.

In the experiments, we use GAC-0 to refer to GAC with Taylor's approximation around the mean, and we use GAC-1 to refer to GAC with Taylor's approximation by a single sample a 0 " π θ pa|sq.

The covariance update in Eq.(19) indicates that F psq " η ‹ Σ´1 θ psq´Hpsq needs to be positive definite.

The matrix F psq is guaranteed to be positive definite if the Hessian matrix Hpsq is negative semi-definite.

However, this is not guaranteed in practice unless p Qps, aq is a concave function in terms of a. To overcome this issue, we firstly consider the following identity: DISPLAYFORM0 The proof is given in Appendix A.3.

The first term is always negative semi-definite while the second term is indefinite.

Therefore, a negative semi-definite approximation of the Hessian can be obtained as DISPLAYFORM1 The second term in Eq. FORMULA1 is proportional to expp´p Qps, aqq and it will be small for high values of p Qps, aq.

This implies that the approximation should gets more accurate as the policy approach a local maxima of p Qps, aq.

We call this approximation Gauss-Newton approximation since it is similar to the Gauss-Newton approximation for the Newton method BID24 .

The second step of GAC is to learn a parameterized actor that well represents the guide actor.

Below, we discuss two supervised learning approaches for learning a parameterized actor.

Since the guide actor is a Gaussian distribution with a state-dependent mean and covariance, a natural choice for the parameterized actor is again a parameterized Gaussian distribution with a state-dependent mean and covariance:

π θ pa|sq " N pa|φ θ psq, Σ θ psqq.

The parameter θ can be learned by minimizing the expected KL divergence to the guide actor: DISPLAYFORM0 where DISPLAYFORM1 ı is the weighted-mean-squared-error (WMSE) which only depends on θ of the mean function.

The const term does not depend on θ.

Minimizing the KL divergence reveals connections between GAC and deterministic policy gradients (DPG) BID29 .

By computing the gradient of the WMSE, it can be shown that DISPLAYFORM2 The proof is given in Appendix A.4.

The negative of the first term is precisely equivalent to DPG.

Thus, updating the mean parameter by minimizing the KL loss with gradient descent can be regarded as updating the mean parameter with biased DPG where the bias terms depend on η ‹ .

We can verify that ∇ a p Qps, aq| a"φ`psq " 0 when η ‹ " 0 and this is the case of Ñ 8.

Thus, all bias terms vanish when the KL constraint is ignored and the mean update of GAC coincides with DPG.

However, unlike DPG which learns a deterministic policy, we can learn both the mean and covariance in GAC.

While a state-dependent parameterized covariance function is flexible, we observe that learning performance is sensitive to the initial parameter of the covariance function.

For practical purposes, we propose using a parametrized Gaussian distribution with state-independent covariance: π θ pa|sq " N pa|φ θ psq, Σq.

This class of policies subsumes deterministic policies with additive independent Gaussian noise for exploration.

To learn θ, we minimize the mean-squared-error (MSE): DISPLAYFORM0 For the covariance, we use the average of the guide covariances: Σ " pη ‹`ω‹ qE p β psq " F´1psq ‰ .

For computational efficiency, we execute a single gradient update in each learning iteration instead of optimizing this loss function until convergence.

Similarly to the above analysis, the gradient of the MSE w.r.t.

θ can be expanded and rewritten into DISPLAYFORM1 Again, the mean update of GAC coincides with DPG when we minimize the MSE and set η ‹ " 0 and Hpsq "´I where I is the identity matrix.

We can also substitute these values back into Eq.(22).

By doing so, we can interpret DPG as a method that performs first-order optimization in the action space: DISPLAYFORM2 and then uses the gradient in Eq.(30) to update the policy parameter.

This interpretation shows that DPG is a first-order method that only uses the first-order information of the critic for actor learning.

Therefore in principle, GAC, which uses the second-order information of the critic, should learn faster than DPG.

Beside actor learning, the performance of actor-critic methods also depends on the accuracy of the critic.

We assume that the critic p Q ν ps, aq is represented by neural networks with a parameter ν.

We adopt the approach proposed by with some adjustment to learn ν.

More concretely, we use gradient descent to minimize the squared Bellman error with a slowly moving target critic: DISPLAYFORM0 where α ą 0 is the step-size.

The target value y " rps, aq`γE πpa 1 |s 1 q r p Qν ps 1 , a 1 qs is computed by the target critic p Qν ps 1 , a 1 q whose parameterν is updated byν Ð τ ν`p1´τ qν for 0 ă τ ă 1.

As suggested by , the target critic improves the learning stability and we set τ " 0.001 in experiments.

The expectation for the squared error is approximated using mini-batch samples tps n , a n , r n , s 1 n qu N n"1 drawn from a replay buffer.

The expectation over the current actor πpa 1 |s 1 q is approximated using samples ta DISPLAYFORM1 We do not use a target actor to compute y since the KL upper-bound already constrains the actor update and a target actor will further slow it down.

Note that we are not restricted to this evaluation method and more efficient methods such as Retrace BID22 can also be used.

Our method requires computing ∇ a p Q ν ps, aq and its outer product for the Gauss-Newton approximation.

The computational complexity of the outer product operation is Opd 2 a q and is inexpensive when compared to the dimension of ν.

For a linear-in-parameter model p Q ν ps, aq " ν J µps, aq, the gradient can be efficiently computed for common choices of the basis function µ such as the Gaussian function.

For deep neural network models, the gradient can be computed by the automaticdifferentiation BID9 where its cost depends on the network architecture.

Besides the connections to DPG, our method is also related to existing methods as follows.

A similar optimization problem to Eq.(13) was considered by the model-free trajectory optimization (MOTO) method BID1 .

Our method can be viewed as a non-trivial extension of MOTO with two significant novelties.

First, MOTO learns a sequence of time-dependent log-linear Gaussian policies π t pa|sq " N pa|B t s`b t , Σ t q, while our method learns a log-nonlinear Gaussian policy.

Second, MOTO learns a time-dependent critic given by p Q t ps, aq " 1 2 a J C t a`a J D t sà J c t`ξt psq and performs policy update with these functions.

In contrast, our method learns a more complex critic and performs Taylor's approximation in each training step.

Besides MOTO, the optimization problem also resembles that of trust region policy optimization (TRPO) BID27 .

TRPO solves the following optimization problem: DISPLAYFORM0 where p Qps, aq may be replaced by an estimate of the advantage function BID28 ).

There are two major differences between the two problems.

First, TRPO optimizes the policy parameter while we optimize the guide actor.

Second, TRPO solves the optimization problem by conjugate gradient where the KL divergence is approximated by the Fisher information matrix, while we solve the optimization problem in a closed form with a quadratic approximation of the critic.

Our method is also related to maximum-entropy RL BID36 BID4 BID11 BID23 , which maximizes the expected cumulative reward with an additional entropy bonus: ř 8 t"1 E p π psq rrps t , a t q`αHpπpa t |s t qqs, where α ą 0 is a trade-off parameter.

The optimal policy in maximum-entropy RL is the softmax policy given by DISPLAYFORM1 where Q ‹ soft ps, aq and V ‹ soft psq are the optimal soft action-value and state-value functions, respectively BID11 BID23 .

For a policy π, these are defined as Q π soft ps, aq " rps, aq`γE pps 1 |s,aq DISPLAYFORM2 DISPLAYFORM3 The softmax policy and the soft state-value function in maximum-entropy RL closely resemble the guide actor in Eq.(14) when η ‹ " 0 and the log-integral term in Eq.(15) when η " 0, respectively, except for the definition of action-value functions.

To learn the optimal policy of maximum-entropy RL, BID11 proposed soft Q-learning which uses importance sampling to compute the soft value functions and approximates the intractable policy using a separate policy function.

Our method largely differs from soft Q-learning since we use Taylor's approximation to convert the intractable integral into more convenient matrix-vector products.

The idea of firstly learning a non-parameterized policy and then later learning a parameterized policy by supervised learning was considered previously in guided policy search (GPS) BID16 .

However, GPS learns the guide policy by trajectory optimization methods such as an iterative linear-quadratic Gaussian regulator BID18 , which requires a model of the transition function.

In contrast, we learn the guide policy via the critic without learning the transition function.

We evaluate GAC on the OpenAI gym platform BID5 with the Mujoco Physics simulator BID33 .

The actor and critic are neural networks with two hidden layers of 400 and 300 units, as described in Appendix C. We compare GAC-0 and GAC-1 against deep DPG (DDPG) , Q-learning with a normalized advantage function (Q-NAF) BID10 , and TRPO BID27 BID35 .

FIG0 shows the learning performance on 9 continuous control tasks.

Overall, both GAC-0 and GAC-1 perform comparably with existing methods and they clearly outperform the other methods in Half-Cheetah.

The performance of GAC-0 and GAC-1 is comparable on these tasks, except on Humanoid where GAC-1 learns faster.

We expect GAC-0 to be more stable and reliable but easier to get stuck at local optima.

On the other hand, the randomness introduced by GAC-1 leads to high variance approximation but this could help escape poor local optima.

We conjecture GAC-S that uses S ą 1 samples for the averaged Taylor's approximation should outperform both GAC-0 and GAC-1.

While this is computationally expensive, we can use parallel computation to reduce the computation time.

The expected returns of both GAC-0 and GAC-1 have high fluctuations on the Hopper and Walker2D tasks when compared to TRPO as can be seen in FIG0 and FIG0 .

We observe that they can learn good policies for these tasks in the middle of learning.

However, the policies quickly diverge to poor ones and then they are quickly improved to be good policies again.

We believe that this happens because the step-size F´1psq "`η ‹ Σ´1´Hpsq˘´1 of the guide actor in Eq. FORMULA1 can be very large near local optima for Gauss-Newton approximation.

That is, the gradients near local optima have small magnitude and this makes the approximation Hpsq " ∇ a p Qps, aq∇ a p Qps, aq J small as well.

If η ‹ Σ´1 is also relatively small then the matrix F´1psq can be very large.

Thus, under these conditions, GAC may use too large step sizes to compute the guide actor and this results in high fluctuations in performance.

We expect that this scenario can be avoided by reducing the KL bound or adding a regularization constant to the Gauss-Newton approximation.

TAB1 in Appendix C shows the wall-clock computation time.

DDPG is computationally the most efficient method on all tasks.

GAC has low computation costs on tasks with low dimensional actions and its cost increases as the dimensionality of action increases.

This high computation cost is due to the dual optimization for finding the step-size parameters η and ω.

We believe that the computation cost of GAC can be significantly reduced by letting η and ω be external tuning parameters.

Actor-critic methods are appealing for real-world problems due to their good data efficiency and learning speed.

However, existing actor-critic methods do not use second-order information of the critic.

In this paper, we established a novel framework that distinguishes itself from existing work by utilizing Hessians of the critic for actor learning.

Within this framework, we proposed a practical method that uses Gauss-Newton approximation instead of the Hessians.

We showed through experiments that our method is promising and thus the framework should be further investigated.

Our analysis showed that the proposed method is closely related to deterministic policy gradients (DPG).

However, DPG was also shown to be a limiting case of the stochastic policy gradients when the policy variance approaches zero BID29 .

It is currently unknown whether our framework has a connection to the stochastic policy gradients as well, and finding such a connection is our future work.

Our main goal in this paper was to provide a new actor-critic framework and we do not claim that our method achieves the state-of-the-art performance.

However, its performance can still be improved in many directions.

For instance, we may impose a KL constraint for a parameterized actor to improve its stability, similarly to TRPO BID27 .

We can also apply more efficient policy evaluation methods such as Retrace BID22 ) to achieve better critic learning.

The solution of the optimization problem: DISPLAYFORM0 subject to E p β psq rKLpπpa|sq||π θ pa|sqqs ď , E p β psq rHpπpa|sqqs ě κ, DISPLAYFORM1 can be obtained by the method of Lagrange multipliers.

The derivation here follows the derivation of similar optimization problems by BID26 and BID0 .

The Lagrangian of this optimization problem is DISPLAYFORM2 where η, ω, and ν are the dual variables.

Then, by taking derivative of L w.r.t.

r π we obtain DISPLAYFORM3 We set this derivation to zero in order to obtain 0 " E p β psq "ż´p Qps, aq´pη`ωq log r πpa|sq`η log π θ pa|sq¯da ´p η`ω´νq " p Qps, aq´pη`ωq log r πpa|sq`η log π θ pa|sq´pη`ω´νq.

Then the solution is given by DISPLAYFORM4 To obtain the dual function gpη, ωq, we substitute the solution to the constraint terms of the Lagrangian and this gives us DISPLAYFORM5 After some calculation, we obtain Lpη, ω, νq " η ´ωκ`E p β psq rη`ω´νs DISPLAYFORM6 where in the second line we use the fact that expp´η`ω´ν η`ω q is the normalization term of r πpa|sq.

Firstly, we show that GAC performs second-order optimization in the action space when Taylor's approximation is performed with a 0 " E πpa|sq ras " φ θ psq.

Recall that Taylor's approximation with φ θ is given by DISPLAYFORM0 where ψ φ θ psq " ∇ a p Qps, aq| a"φ θ psq´H φ θ psqφ θ psq.

By substituting ψ φ θ psq into Lpsq " η ‹ Σ´1 θ psqφ θ psq´ψ θ psq, we obtain DISPLAYFORM1 Therefore, the mean update is equivalent to φ`psq " F´1psqLpsq DISPLAYFORM2 which is a second-order optimization step with a curvature matrix F psq " η ‹ Σ´1 θ psq´H φ θ psq.

Similarly, for the case where a set of samples ta 0 u " π θ pa 0 |sq " N pa 0 |φ θ psq, Σpsqq is used to compute the averaged Taylor's approximation, we obtain DISPLAYFORM3 Then, by assuming that E π θ pa0|sq rH 0 psqa 0 psqs " E π θ pa0|sq rH 0 s E π θ pa0|sq ra 0 s, we obtain DISPLAYFORM4 Therefore, we have a second-order optimization step DISPLAYFORM5 where F´1psq " η ‹ Σ´1 θ psq´E π θ pa0|sq rH 0 psqs is a curvature matrix.

As described in the main paper, this interpretation is only valid when the equality E π θ pa0|sq rH 0 psqa 0 psqs " E π θ pa0|sq rH 0 s E π θ pa0|sq ra 0 s holds.

While this equality does not hold in general, it holds when only one sample a 0 " π θ pa 0 |sq is used.

Nonetheless, we can still use the expectation of Taylor's approximation to perform policy update regardless of this assumption.

Let f ps, aq " expp p Qps, aqq, then the Hessian Hpsq " ∇ 2 a p Qps, aq can be expressed as Hpsq " ∇ a r∇ a log f ps, aqs DISPLAYFORM0 a f ps, aqf ps, aq´1 " ∇ a f ps, aq∇ f f ps, aq´1 p∇ a f ps, aqq J`∇2 a f ps, aqf ps, aq´1 "´∇ a f ps, aqf ps, aq´2 p∇ a f ps, aqq J`∇2 a f ps, aqf ps, aq´1 "´`∇ a f ps, aqf ps, aq´1˘`∇ a f ps, aqf ps, aq´1˘J`∇ 2 a f ps, aqf ps, aq´1 "´∇ a log f ps, aq∇ a log f ps, aq J`∇2 a f ps, aqf ps, aq´1 "´∇ a p Qps, aq∇ a p Qps, aq DISPLAYFORM1 which concludes the proof.

Beside Gauss-Newton approximation, an alternative approach is to impose a special structure on p Qps, aq so that Hessians are always negative semi-definite.

In literature, there exists two special structures that satisfies this requirement.

Normalized advantage function (NAF) BID10 : NAF represents the critic by a quadratic function with a negative curvature: DISPLAYFORM2 where a negative-definite matrix-valued function W psq, a vector-valued function bpsq and a scalarvalued function V psq are parameterized functions whose their parameters are learned by policy evaluation methods such as Q-learning BID31 .

With NAF, negative definite Hessians can be simply obtained as Hpsq " W psq.

However, a significant disadvantage of NAF is that it assumes the action-value function is quadratic regardless of states and this is generally not true for most reward functions.

Moreover, the Hessians become action-independent even though the critic is a function of actions.

Input convex neural networks (ICNNs) BID3 : ICNNs are neural networks with special structures which make them convex w.r.t.

their inputs.

Since Hessians of concave functions are always negative semi-definite, we may use ICNNs to represent a negative critic and directly use its Hessians.

However, similarly to NAF, ICNNs implicitly assume that the action-value function is concave w.r.t.

actions regardless of states and this is generally not true for most reward functions.

We first consider the weight mean-squared-error loss function where the guide actor is N pa|φ`psq, Σ`psqq and the current actor is N pa|φ θ psq, Σ θ psqq.

Taylor's approximation of DISPLAYFORM0 By assuming that H φ θ psq is strictly negative definite 5 , we can take a derivative of this approximation w.r.t.

a and set it to zero to obtain a " H´1 φ θ psq∇ a p Qps, aq´H´1 φ θ psqψ φ θ psq.

Replacing a by φ θ psq and φ`psq yields DISPLAYFORM1 φ`psq " H´1 φ θ psq∇ a p Qps, aq| a"φ`psq´H´1 φ θ psqψ φ θ psq.

Recall that the weight mean-squared-error is defined as DISPLAYFORM2 Published as a conference paper at ICLR 2018Firstly, we expand the quadratic term of NAF as follows: DISPLAYFORM3 where ψpsq "´W psqbpsq and ξpsq " 1 2 bpsq J W psqbpsq`V psq.

By substituting the quadratic model obtained by NAF into the GAC framework, the guide actor is now given byπpa|sq " N pa|φ`psq, Σ`psqqq with DISPLAYFORM4 To obtain Q-learning with NAF, we set η ‹ " 0, i.e., we perform a greedy maximization where the KL upper-bound approaches infinity, and this yields φ`psq "´W psq´1p´W psqbpsqq DISPLAYFORM5 which is the policy obtained by performing Q-learning with NAF.

Thus, NAF with Q-learning is a special case of GAC if Q-learning is also used in GAC to learn the critic.

The pseudo-code of GAC is given in Algorithm 1.

The source code is available at https:// github.com/voot-t/guide-actor-critic.

We try to follow the network architecture proposed by the authors of each baseline method as close as possible.

For GAC and DDPG, we use neural networks with two hidden layers for the actor network and the critic network.

For both networks the first layer has 400 hidden units and the second layer has 300 units.

For NAF, we use neural networks with two hidden layers to represent each of the functions bpsq, W psq and V psq where each layer has 200 hidden units.

All hidden units use the relu activation function except for the output of the actor network where we use the tanh activation function to bound actions.

We use the Adam optimizer BID14 with learning rate 0.001 and 0.0001 for the critic network and the actor network, respectively.

The moving average step for target networks is set to τ " 0.001.

The maximum size of the replay buffer is set to 1000000.

The mini-batches size is set to N " 256.

The weights of the actor and critic networks are initialized as described by BID8 , except for the output layers where the initial weights are drawn uniformly from Up´0.003, 0.003q, as described by .

The initial covariance Σ in GAC is set to be an identity matrix.

DDPG and QNAF use the OU-process with noise parameters θ " 0.15 and σ " 0.2 for exploration .For TRPO, we use the implementation publicly available at https://github.com/openai/ baselines.

We also use the provided network architecture and hyper-parameters except the batch size where we use 1000 instead of 1024 since this is more suitable in our test setup.

For GAC, the KL upper-bound is fixed to " 0.0001.

The entropy lower-bound κ is adjusted heuristically by κ " maxp0.99pE´E 0 q`E 0 , E 0 q,where E « E p β psq rHpπ θ pa|sqqs denotes the expected entropy of the current policy and E 0 denotes the entropy of a base policy N pa|0, 0.01Iq.

This heuristic ensures that the lower-bound gradually decreases but the lower-bound cannot be too small.

We apply this heuristic update once every 5000 training steps.

The dual function is minimize by the sequential least-squares quadratic programming (SLSQP) method with an initial values η " 0.05 and ω " 0.05.

The number of samples for computing the target critic value is M " 10.

Sample N mini-batch samples tps n , a n , r n , s Compute y n , update ν by, e.g., Adam, and updateν by moving average: DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 13:end procedure

procedure LEARN GUIDE ACTOR 15:Compute a n,0 for each s n by a n,0 " φ θ ps n q or a n,0 " N pa|φ θ ps n q, Σq.

Compute g 0 psq " ∇ a p Qps n , aq| a"an,0 and H 0 ps s q "´g 0 ps n qg 0 ps n q J .

Solve for pη ‹ , ω ‹ q " argmin ηą0,ωą0 p gpη, ωq by a non-linear optimization method.

Compute the guide actor r πpa|s n q " N pa|φ`ps n q, Σ`ps n qq for each s n .

end procedure

procedure UPDATE PARAMETERIZED ACTOR

Update policy parameter by, e.g., Adam, to minimize the MSE: DISPLAYFORM0 ∇ θ }φ θ ps n q´φ`ps n q} 2 2 .22:Update policy covariance by averaging the guide covariances: DISPLAYFORM1 23:end procedure

end procedure 25: end for 26: Output: Learned actor π θ pa|sq.

We perform experiments on the OpenAI gym platform BID5 with Mujoco Physics simulator BID33 where all environments are v1.

We use the state space, action space and the reward function as provided and did not perform any normalization or gradient clipping.

The maximum time horizon in each episode is set to 1000.

The discount factor γ " 0.99 is only used for learning and the test returns are computed without it.

Experiments are repeated for 10 times with different random seeds.

The total computation time are reported in TAB1 .

The figures below show the results averaged over 10 trials.

The y-axis indicates the averaged test returns where the test returns in each trial are computed once every 5000 training time steps by executing 10 test episodes without exploration.

The error bar indicates standard error.

The total computation time for training the policy for 1 million steps (0.1 million steps for the Invert-Pendulum task).

The mean and standard error are computed over 10 trials with the unit in hours.

TRPO is not included since it performs a lesser amount of update using batch data samples.

<|TLDR|>

@highlight

This paper proposes a novel actor-critic method that uses Hessians of a critic to update an actor.