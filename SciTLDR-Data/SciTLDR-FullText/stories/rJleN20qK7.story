A key component for many reinforcement learning agents is to learn a value function, either for policy evaluation or control.

Many of the algorithms for learning values, however, are designed for linear function approximation---with a fixed basis or fixed representation.

Though there have been a few sound extensions to nonlinear function approximation, such as nonlinear gradient temporal difference learning, these methods have largely not been adopted, eschewed in favour of simpler but not sound methods like temporal difference learning and Q-learning.

In this work, we provide a two-timescale network (TTN) architecture that enables linear methods to be used to learn values, with a nonlinear representation learned at a slower timescale.

The approach facilitates the use of algorithms developed for the linear setting, such as data-efficient least-squares methods, eligibility traces and the myriad of recently developed linear policy evaluation algorithms, to provide nonlinear value estimates.

We prove convergence for TTNs, with particular care given to ensure convergence of the fast linear component under potentially dependent features provided by the learned representation.

We empirically demonstrate the benefits of TTNs, compared to other nonlinear value function approximation algorithms, both for policy evaluation and control.

Value function approximation-estimating the expected returns from states for a policy-is heavily reliant on the quality of the representation of state.

One strategy has been to design a basis-such as radial basis functions (Sutton and Barto, 1998) or a Fourier basis BID17 )-for use with a linear function approximator and temporal difference (TD) learning (Sutton, 1988) .

For low-dimensional observation vectors, this approach has been effective, but can be onerous to extend to high-dimensional observations, potentially requiring significant domain expertise.

Another strategy has been to learn the representation, such as with basis adaptation or neural networks.

Though there is still the need to specify the parametric form, learning these representations alleviates the burden of expert specification.

Further, it is more feasible to scale to high-dimensional observations, such as images, with neural networks (Mnih et al., 2015; BID16 .

Learning representations necessitates algorithms for nonlinear function approximation.

Despite the deficiencies in specification for fixed bases, linear function approximation for estimating value functions has several benefits over nonlinear estimators.

They enable least-squares methods, which can be much more data-efficient for policy evaluation BID5 Szepesvari, 2010; van Seijen and Sutton, 2015) , as well as robust to meta-parameters (Pan et al., 2017) .

Linear algorithms can also make use of eligibility traces, which can significantly speed learning (Sutton, 1988; BID9 White and White, 2016 ), but have not been able to be extended to nonlinear value function approximation.

Additionally, there have been a variety of algorithms derived for the linear setting, both for on-policy and off-policy learning (Sutton et al., 2009; BID23 van Seijen and Sutton, 2014; van Hasselt et al., 2014; Mahadevan et al., 2014; Sutton et al., 2016; Mahmood et al., 2017) .

These linear methods have also been well-explored theoretically (Tsitsiklis and Van Roy, 1997; BID23 Mahmood and Sutton, 2015; Yu, 2015) and empirically BID9 White and White, 2016) , with some insights into improvements from gradient methods (Sutton et al., 2009 ), true-online traces (van Seijen and Sutton, 2014) and emphatic weightings (Sutton et al., 2016) .

These algorithms are easy to implement, with relatively simple objectives.

Objectives for nonlinear value function approximation, on the other hand, can be quite complex (Maei et al., 2009) , resulting in more complex algorithms (Menache et al., 2005; BID10 BID2 or requiring a primal-dual formulation as has been done for control BID8 .In this work, we pursue a simple strategy to take advantage of the benefits of linear methods, while still learning the representation.

The main idea is to run two learning processes in parallel: the first learns nonlinear features using a surrogate loss and the second estimates the value function as a linear function of those features.

We show that these Two-timescale Networks (TTNs) converge, because the features change on a sufficiently slow scale, so that they are effectively fixed for the fast linear value function estimator.

Similar ideas have previously been explored for basis adaptation, but without this key aspect of TTNs-namely the separation of the loss for the representation and value function.

This separation is critical because it enables simpler objectives-for which the gradient can be easily sampled-to drive the representation, but still enables use of the mean squared projected Bellman error (MSPBE)-on which all the above linear algorithms are based.

This separation avoids the complexity of the nonlinear MSPBE, but maintains the useful properties of the (linear) MSPBE.

A variety of basis adaptation approaches have used a two-timescale approach, but with the same objective for the representation and the values (Menache et al., 2005; BID10 BID2 .

Yu and Bertsekas (2009) provided algorithms for basis adaptation using other losses, such as Bellman error using Monte carlo samples, taking derivatives through fixed point solutions for the value function.

BID21 periodically compute a closed form least-squares solution for the last layer of neural network, with a Bayesian update to prevent too much change.

Because these methods did not separate the value learn and basis adaptation, the resulting algorithms are more complex.

The strategy of using two different heads-one to drive the representation and one to learn the values-has yet to be systematically explored.

We show that TTNs are a promising direction for nonlinear function approximation, allowing us to leverage linear algorithms while retaining the flexibility of nonlinear function approximators.

We first discuss a variety of possible surrogate losses, and their potential for learning a useful representation.

We then show that TTNs converge, despite the fact that a linear algorithm is used with a changing representation.

This proof is similar to previous convergence proofs for policy evaluation, but with a relaxation on the requirement that features be independent, which is unlikely for learned features.

We then show empirically that TTNs are effective compared to other nonlinear value function approximations and that they can exploit several benefits of linear value approximations algorithms.

In particular, for both low-dimensional and high-dimensional (image-based) observations, we show (a) the utility of least-squares (or batch) methods, (b) advantages from eligibility traces and (c) gains from being able to select amongst different linear policy evaluation algorithms.

We demonstrate that TTNs can be effective for control with neural networks, enabling use of fitted Q-iteration within TTNs as an alternative to target networks.

We assume the agents act in a finite Markov Decision Process (MDP), with notation from (White, 2017) .

The dynamics of the MDP are defined by the 3-tuple (S, A, P ), where S is the set of states, A the set of actions and P : S × A × S → [0, 1] the transition probability function.

The task in this environment is defined by a reward function R : S × A × S → R and a discount function γ : S × A × S → [0, 1].

At each time step, the agent takes an action A t according to a policy π : S × A → [0, 1] and the environment returns reward R t+1 , next state S t+1 and discount γ t+1 .The goal in policy evaluation is to compute the value function: the expected sum of discounted rewards from every state under a fixed policy π.

The value function V π : S → R is defined recursively from each state s ∈ S as DISPLAYFORM0 s ∈S P (s, a, s )(r + γV π (s )).(When using linear function approximation, this goal translates into finding parameters w ∈ R d to approximate the value function DISPLAYFORM1 More generally, a nonlinear functionV (s) could be learned to estimate V π .To formulate this learning problem, we need to consider the objective for learning the functionV .

Let V π ,V ∈ R |S| be the vectors of values for V π ,V .

The recursive formula (1) defines a Bellman operator B π where the fixed point satisfies B π V π = V π .

Consider a restricted value function class, such as the set of linear value functionsV ∈ F = {Xw | w ∈ R d } where X ∈ R |S|×d is a matrix with the i-th row set to x(s) for ith state s ∈ S.

Then, it may no longer be possible to satisfy the recursion.

Instead, an alternative is to find a projected fixed point Π F B πV =V where the projection operator Π F projects B πV to the space spanned by this linear basis: DISPLAYFORM2 |S| is a vector which weights each state in the weighted norm DISPLAYFORM3 Many linear policy evaluation algorithms estimate this projected fixed point, including TD (Sutton, 1988) , least-squares TD BID5 and gradient TD (Sutton et al., 2009 ).The objective formulated for this projected fixed-point, however, is more complex for nonlinear function approximation.

For linear function approximation, the projection operator simplifies into a closed form solution involving only the features X. Letting δ t = R t+1 + γV (S t+1 ) −V (S t ), the resulting mean-squared projected Bellman error (MSPBE) can be written as DISPLAYFORM4 where DISPLAYFORM5 For nonlinear function classes, the projection does not have a closed form solution and may be expensive to compute.

Further, the projection involves the value function parameters, so the projection changes as parameters change.

The nonlinear MSPBE and resulting algorithm are more complex (Maei et al., 2009) , and have not seen widespread use.

Another option is simply to consider different objectives.

However, as we discuss below, other objectives for learning the value function either are similarly difficult to optimize or provide poor value estimates.

In the next section, we discuss some of these alternatives and introduce Two-timescale Networks as a different strategy to enable nonlinear value function approximation.

We first introduce Two-timescale Networks (TTNs), and then describe different surrogate objectives that can be used in TTNs.

We discuss why these surrogate objectives within TTNs are useful to drive the representation, but are not good replacements for the MSPBE for learning the value function.

TTNs use two concurrent optimization processes: one for the parameters of the network θ and one for the parameters of the value function w. The value function is approximated asV (s) def = x θ (s) w where the features x θ : S → R d are a parametrized function and θ ∈ R m is adjusted to provide better features.

For a neural network, θ consists of all the parameters in the hidden layers, to produce the final hidden layer x θ (s).

The two optimization processes maintain different time scales, with the parameters θ for the representation changed as a slow process, and the parameters w for the value estimate changed as a fast process relative to θ.

The separation between these two processes could be problematic, since the target problemestimating the value function-is not influencing the representation!

The slow process is driven by a completely separate objective than the fast process.

However, the key is to select this surrogate loss for the slow process so that it is related to the value estimation process, but still straightforward to compute the gradient of the loss.

We useV (s) as the output of the fast part, which corresponds to the value estimate used by the agent.

To distinguish,Ŷ (s) denotes the output for the slow-part (depicted in FIG0 ), which may or may not be an estimate of the value, as we discuss below.

Consider first the mean-squared TD error (MSTDE), which corresponds to DISPLAYFORM0 Notice that this does not correspond to the mean-squared Bellman error (MSBE), for which it is more difficult to compute gradients DISPLAYFORM1 2 .

Using the MSTDE as a surrogate loss, withŶ (s) = x θ (s) w, the slow part of the network minimizes DISPLAYFORM2 This slow part has its own weightsw associated with estimating the value function, but learned instead according to the MSTDE.

The advantage here is that stochastic gradient descent on the MSTDE is straightforward, with gradient δ t ∇ {θ,w} [γ t+1Ŷ (S t+1 ) −Ŷ (S t )] where ∇ {θ,w}Ŷ (S t ) is the gradient of the neural network, including the head of the slow part which uses weightsw.

Using the MSTDE has been found to provide worse value estimates than the MSPBE-which we re-affirm in our experiments.

It could, nonetheless, play a useful role as a surrogate loss, where it can inform the representation towards estimating values.

There are a variety of other surrogate losses that could be considered, related to the value function.

However, many of these losses are problematic to sample incrementally, without storing large amounts of data.

For example, the mean-squared return error (MSRE) could be used, which takes samples of return and minimizes mean-squared error to those sampled returns.

Obtaining such returns requires waiting many steps, and so delays updating the representation for the current state.

Another alternative is the MSBE.

The gradient of the nonlinear MSBE is not as complex as the gradient of the nonlinear MSPBE, because it does not involve the gradient of a projection.

However, it suffers from the double sampling problem: sampling the gradient requires two independent samples.

For these reasons, we explore the MSTDE as the simplest surrogate loss involving the value function.

Finally, surrogate losses could also be defined that are not directly related to the value function.

Two natural choices are losses based on predicting the next state and reward.

The output of the slow part could correspond to a vector of values, such as DISPLAYFORM3 Rt+1 .

The ability to predict the next state and reward is intuitively useful for enabling prediction of value, that also has some theoretical grounding.

Szepesvari (2010, Section 3.2.1) shows that the Bellman error is small, if the features can capture a horizon of immediate rewards and expected next states.

For linear encoders, Song et al. (2016) show that an optimal set of features enables predictions of next state and reward.

More generally, learning representations using auxiliary tasks or self-supervised tasks have had some successes in RL, such as using pixel control BID16 or classifying the temporal distance between frames BID0 .

In computer vision, BID12 showed that using rotated images as self-supervised tasks produced a useful representation for the main loss, without training the representation with the main loss.

Any of these self-supervised tasks could also be used for the surrogate objective, and motivate that separating out representation learning does not degrade performance.

For now, we restrict focus on simpler surrogate objectives, as the main purpose of this work is to demonstrate that the separation in TTNs is a sound approach for learning values.

Training TTNs is fully online, using a single transition from the environment at a time.

Projected stochastic gradient descent is used to reduce the surrogate loss, L slow (θ) and a linear policy evaluation algorithm, such as GTD2 or TD(λ), is coupled to the network where the prediction vector w is callibrated proportional to −∇ w MSPBE θ (w).

The full procedure is summarized in Algorithm 1, in Appendix A. Regarding the convergence of TTNs, a few remarks are in order:1.

The network needs to evolve sufficiently slowly relative to the linear prediction weights.

In our theoretical analysis, this is achieved by ensuring that the step sizes ξ t and α t of the network and the linear policy evaluation algorithm respectively decay to zero at different rates.

In particular, ξ t /α t → 0 as t → ∞. With this relative disparity in magnitudes, one can assume that the network is essentially quasi-static, while the faster linear component is equilibrated relative to the static features.2.

The linear prediction algorithms need to converge for any set of features provided by the neural network, particularly linearly dependent features.

This induces a technical bottleneck since linear independence of the features are a necessary condition for the convergence of the prediction methods GTD and GTD2.

We overcome this by following a differential inclusion based analysis for GTD2.3.

Finally, we need to guarantee the stability of the iterates (both feature vector θ t and the prediction vector w t ) and this is ensured by projecting the iterates to respective compact, convex sets.

The analysis for the convergence of the neural network is general, enabling any network architectures that are twice continuously differentiable.

We prove that the TTNs converge asymptotically to the stable equilibria of a projected ODE which completely captures the mean dynamics of the algorithm.

We now state our main result (for notations and technical details, please refer Appendix B).

The results are provided for cases when TD(λ) or GTD2 is used as the linear prediction method.

However, note that similar results can be obtained for other linear prediction methods.

Theorem 1.

Letθ = (θ,w) and Θ ⊂ R m+d be a compact, convex subset with smooth boundary.

Let the projection operator Γ Θ be Frechet differentiable and Γ Θ θ (− 1 2 ∇L slow )(θ) be Lipschitz continuous.

Also, let Assumptions 1-3 hold.

Let K be the set of asymptotically stable equilibria of the following ODE contained inside Θ: DISPLAYFORM0 Then the stochastic sequence {θ t } t∈N generated by the TTN converges almost surely to K (sample path dependent).

Further, TD(λ) Convergence:

Under the additional Assumption 4-TD(λ), we obtain the following result: For any λ ∈ [0, 1], the stochastic sequence {w t } t∈N generated by the TD(λ) algorithm (Algorithm 2) within the TTN setting converges almost surely to the limit w * , where w * satisfies DISPLAYFORM1 withθ * ∈ K (sample path dependent).

We investigate the performance of TTNs versus a variety of other nonlinear policy evaluation algorithms, as well as the impact of choices within TTNs.

We particularly aim to answer (a) is it beneficial to optimize the MSPBE to obtain value estimates, rather than using value estimates from surrogate losses like the MSTDE; (b) do TTNs provide gains over other nonlinear policy evaluation algorithms; and (c) can TTNs benefit from the variety of options in linear algorithms, including leastsquares approaches, eligibility traces and different policy evaluation algorithms.

More speculatively, we also investigate if TTNs can provide a competitive alternative to deep Q-learning in control.

Experiments were performed on-policy in five environments.

We use three classic continuous-state domains: Puddle World, a continuous-state grid world with high-magnitude negative rewards for walking through a puddle; Acrobot, where a robot has to swing itself up; and Cartpole, which involves balancing a pole.

We also use two game domains: Catcher, which involves catching falling apples; and Puck World, in which the agent has to chase a puck (Tasfi, 2016) .

Catcher includes both a variant with 4-dimensional observations-position and velocity of the paddle, and (x,y) of the appleand one with image-based observations-with two consecutive 64-by-64 grayscale images as input.

This domain enables us to analyze the benefit of the algorithms, on the same domain, both with low-dimensional and high-dimensional observations.

We describe the policies evaluated for these domains in Appendix D. We include a subset of results in the main body, with additional results in the appendix.

Results in Cartpole are similar to Acrobot; Cartpole results are only in the appendix.

The value estimates are evaluated using root-mean-squared value error (RMSVE), where value error DISPLAYFORM0 The optimal values for a set of 500 states are obtained using extensive rollouts from each state and the RMSVE is computed across these 500 states.

For the algorithms, we use the following settings, unless specified otherwise.

For the slow part (features), we minimize the mean-squared TD error (MSTDE) using the AMSGrad optimizer (Reddi et al., 2018) with β 1 = 0 and β 2 = 0.99.

The network weights use Xavier initialization BID13 ; the weights for the fast part were initialized to 0.

In Puddle World, the neural network consists of a single hidden layer of 128 units with ReLU activations.

In the other environments, we use 256 units instead.

To choose hyperparameters, we first did a preliminary sweep on a broad range and then chose a smaller range where the algorithms usually made progress, summarized in Appendix D. Results are reported for hyperparameters in the refined range, chosen based on RMSVE over the latter half of a run with shaded regions corresponding to one standard error.

TTN vs. competitors.

We compare to the following algorithms: nonlinear TD, nonlinear GTD (Maei et al., 2009) , Adaptive Bases (ABBE and ABTD) BID10 , nonlinear TD + LSTD regularization (inspired by BID21 ).

We describe these algorithms in more depth in Appendix D. All of the algorithms involve more complex updates compared to TTNs, except for nonlinear TD, which corresponds to a semi-gradient TD update with nonlinear function approximation.

For TTNs, we use LSTD for the linear, fast part.

In FIG1 , TTN is able to perform as well or better than the competitor algorithms.

Especially in Puddle World, its error is significantly lower than the second best algorithm.

Interestingly, Nonlinear GTD also performs well across domains, suggesting an advantage for theoretically-sound algorithms.

The utility of optimizing the MSPBE.

First, we show that the TTN benefits from having a second head learning at a faster timescale.

To do so, we compare the prediction errors of using TTN, with the fast process optimizing the MSPBE (using LSTD) and the slow one optimizing the MSTDE, and one trained end-to-end using the MSTDE with AMSGrad.

As a baseline, we include TTN with a fixed representation (a randomly initialized neural network) to highlight that the slow process is indeed improving the representation.

We also include results for optimizing the MSTDE with the fixed representation.

In FIG2 , we see that optimizing the MSPBE indeed gives better results than optimizing the MSTDE.

Additionally, we can conclude that using the MSTDE, despite being a poor objective to learn the value function, can still be effective for driving feature-learning since it outperforms the fixed representation.

Linear algorithms and eligibility traces.

TTNs give us the flexibility to choose any linear policy evaluation algorithm for the fast part.

We compare several choices: TD, least-squares TD (LSTD) BID5 , forgetful LSTD (FLSTD) (van Seijen and Sutton, 2015) , emphatic TD (Sutton et al., 2016) , gradient TD (the TDC variant) (Sutton et al., 2009 ) and their true-online versions (van Seijen and Sutton, 2014; van Hasselt et al., 2014) to learn the value function.

GTD and ETD are newer temporal difference methods which have better convergence properties and can offer increased stability.

The true-online variants modify the update rules to improve the behavior of the algorithms when learning online and seem to outperform their counterparts empirically (van Seijen and Sutton, 2014) .

Least-squares methods summarize past interaction, but are often avoided due to quadratic computation in the number of features.

For TTNs, however, there is no computational disadvantage to using LSTD methods, for two reasons.

It is common to choose deep but skinny architectures (Mnih et al., 2015; BID14 .

Furthermore, if the last layer is fully connected, then we already need to store O(d 2 ) weights and use O(d 2 ) time to compute a forward pass-the same as LSTD.

We include FLSTD, which progressively forgets older interaction, as this could be advantageous when the feature representation changes over time.

For TTN, incremental versions of the least-squares algorithms are used to maintain estimates of the required quantities online (see appendix D).All of these linear algorithms can use eligibility traces to increase their sample efficiency by propagating TD errors back in time.

The trace parameter λ can also provide a bias-variance tradeoff for the value estimates (Sutton, 1988; BID9 .

For nonlinear function approximation, eligibility traces can no longer be derived for TD.

Though invalid, we can naively extend them to this case by keeping one trace per weight, giving us nonlinear TD(λ).

The results overall indicate that TTNs can benefit from the ability to use different linear policy evaluation algorithms and traces, in particular from the use of least-squares methods as shown in FIG3 for Puddle World and Catcher.

The dominance of LSTD versus the other linear algorithms is consistent, including in terms of parameter sensitivity, persists for the other three domains.

We additionally investigated sensitivity to λ, and found that most of the TTN variants benefit from a nonzero λ value and, in many cases, the best setting is high, near 1.

One exception is the least-squares methods, where LSTD performs similarly for most values of λ.

Nonlinear TD(λ), on the other hand, performs markedly worse as λ increases.

This is unsurprising considering the naive addition of eligibility traces is unsound.

We include these sensitivity plots in the appendix, in Figure ? ?

.Surrogate loss functions.

For all the previous experiments, we optimized the MSTDE for the slow part of the network, but as discussed in Section 3, other objectives can be used.

We compare a variety of objectives, by choosing different Figure 5 a), we can see that every auxiliary loss performed well.

This does not appear to be universally true, as in Acrobot we found that the MSTDE was a less effective surrogate loss, leading to slower learning (see Figure 5 b ).

Alternate losses such as the semi-gradient MSTDE and next state predictions were more successful in that domain.

These results suggest that there is no universally superior surrogate loss and that choosing the appropriate one can yield benefits in certain domains.

Control Although the focus of this work is policy evaluation, we also provide some preliminary results for the control setting.

For control, we include some standard additions to competitor learning algorithms to enable learning with neural networks.

The DQN algorithm (Mnih et al., 2015) utilizes two main tricks to stabilize training: experience replay-storing past transitions and replaying them multiple times-and a target network-which keeps the value function in the Q-learning targets fixed, updating the target network infrequently (e.g., every k = 10, 000 steps).

DISPLAYFORM1 We use an alternative strategy to target networks for TTN.

The use of a target network is motivated by fitted Q-iteration (FQI) BID11 , which updates towards fixed Q-values with one sweep through a batch of data.

TTNs provide a straightforward mechanism to instead directly use FQI, where we can solve for the weights on the entire replay buffer, taking advantage of the closed form solution for linear regression towards the Q-values from the last update.

Batch FQI requires storing all data, whereas we instead have a sliding window of experience.

We therefore additionally incorporate a regularization term, which prevents the weights from changing too significantly between updates, similarly to BID21 .

Each FQI iteration requires solving a least squares problem on the entire buffer, an operation costing O(nd 2 ) computation where d is the number of features in the last layer of the network and n is the size of the buffer; we update the network every k steps, which reduces the per-step computation to O(nd 2 /k).

The slow part drives feature-learning by minimizing the semi-gradient MSTDE for state-action values.

As another competitor, we include LS-DQN BID21 , a DQN variant which also utilizes adjustments to the final layer's weights towards the FQI solution, similar to TTN-FQI.The experimental details differ for control.

On nonimage Catcher, we do a sweep over α slow and λ reg , the regularization parameter, for TTN and sweep over the learning rate and the number of steps over which is annealed for DQN.

On image Catcher, runs require significantly more computation so we only tune hyperparameters by hand.

The FQI update in TTNs was done every 1000 (10000) steps for non-image (image) Catcher.

We run each algorithm 10 times (5 times) for 200 thousand steps (10 million steps) on the non-image (image) Catcher.

We see that TTN is able to perform well on both versions of Catcher in Figure 6 , particularly learning more quickly than the DQN variants.

This difference is especially pronounced in the image version of catcher, where TTN is also able to achieve much higher average returns than DQN.

Both algorithms seem to suffer from catastrophic forgetting later during training as the performance dips down after an initial rise, although TTN still stabilizes on a better policy.

Overall, these results suggest that TTNs are a promising direction for improving sample efficiency in control, whilst still maintaining stability when training neural networks.

In this work, we proposed Two-timescale Networks as a new strategy for policy evaluation with nonlinear function approximation.

As opposed to many other algorithms derived for nonlinear value function approximation, TTNs are intentionally designed to be simple to promote ease-of-use.

The algorithm combines a slow learning process for adapting features and a fast process for learning a linear value function, both of which are straightforward to train.

By leveraging these two timescales, we are able to prove convergence guarantees for a broad class of choices for both the fast and slow learning components.

We highlighted several cases where the decoupled architecture in TTNs can improve learning, particularly enabling the use of linear methods-which facilitates use of least-squares methods and eligibility traces.

This work has only begun the investigation into which combinations for surrogate losses and linear value function approximation algorithms are most effective.

We provided some evidence that, when using stochastic approximation algorithms rather than least-squares algorithms, the addition of traces can have a significant effect within TTNs.

This contrasts nonlinear TD, where traces were not effective.

The ability to use traces is potentially one of the most exciting outcomes for TTNs, since traces have been so effective for linear methods.

More generally, TTNs provide the opportunity to investigate the utility of the many linear value function algorithms, in more complex domains with learned representations.

For example, emphatic algorithms have improved asymptotic properties (Sutton et al., 2016) , but to the best of our knowledge, have not been used with neural networks.

Another promising direction for TTNs is for off-policy learning, where many value functions are learned in parallel.

Off-policy learning can suffer from variance due to large magnitude corrections (importance sampling ratios).

With a large collection of value functions, it is more likely that some of them will cause large updates, potentially destabilizing learning in the network if trained in an end-to-end fashion.

TTNs would not suffer from this problem, because a different objective can be used to drive learning in the network.

We provide some preliminary experiments in the appendix supporting this hypothesis (Appendix C.7).

θ,w ← GradientDescent on L slow using sample (s, r, s ) DISPLAYFORM0 w ← Update on L value using sample (s, r, s )8: DISPLAYFORM1 end while 10:return learned parameters w, θ,w 11: end procedure B CONVERGENCE PROOF OF TWO-TIMESCALE NETWORKS B.1 DEFINITIONS & NOTATIONS -Let R + denote the set of non-negative real numbers, N = {0, 1, 2, . . . } and · denote the Euclidean norm or any equivalent norm.

DISPLAYFORM2 3.

h is upper-semicontinuous, i.e., if {x n } n∈N → x and {y n } n∈N → y, where x n ∈ R d , y n ∈ h(x n ), ∀n ∈ N, then y ∈ h(x).-For x 1 , x 2 ∈ R d and D ∈ R k×k a diagonal matrix, we define the inner-product < x 1 , x 2 > D x 1 Dx 2 .

We also define the semi-norm DISPLAYFORM3 D .

If all the diagonal elements of D are strictly positive, then · D is a norm.-For any set X, letX denote the interior of X and ∂X denote the boundary of X. -For brevity, letθ = (θ,w) and Φθ be the feature matrix corresponding to the feature parameterθ, i.e. DISPLAYFORM4 where x θ (s) is the row-vector corresponding to state s. Further, define the |S| × |S|-matrix P π as follows: P π s,s a∈A π(s, a)P (s, a, s ), s, s ∈ S.-Also, recall that DISPLAYFORM5 is Frechet differentiable at x ∈ U if there exists a bounded linear operator Γ x : R d1 → R d2 such that the limit DISPLAYFORM6 exists and is equal to Γ x (y).

We say Γ is Frechet differentiable if Frechet derivative of Γ exists at every point in its domain.

Assumption 1: The pre-determined, deterministic, step-size sequence {ξ t } t∈N satisfies DISPLAYFORM0 Assumption 2: The Markov chain induced by the given policy π is ergodic, i.e., aperiodic and irreducible.

Assumption 2 implies that the underlying Markov chain is asymptotically stationary and henceforth it guarantees the existence of a unique steady-state distribution d π over the state space S (Levin and Peres, 2017), i.e., lim t→∞ P(S t = s) = d π (s), ∀s ∈ S.Assumption 3:

Given a realization of the transition dynamics of the MDP in the form of a sample trajectory O π = {S 0 , A 0 , R 1 , S 1 , A 1 , R 2 , S 2 , . . . }, where the initial state S 0 ∈ S is chosen arbitrarily, while the action A A t ∼ π(S t , ·), the transitioned state S S t+1 ∼ P (S t , A t , ·) and the reward R R t+1 = R(S t , A t , S t+1 ).

Faster timescale TD GTD2 TDC ETD LSTD LSPE

To analyze the long-run behaviour of our algorithm, we employ the ODE based analysis BID4 BID18 BID22 of the stochastic recursive algorithms.

Here, we consider a deterministic ordinary differential equation (ODE) whose asymptotic flow is equivalent to the long-run behaviour of the stochastic recursion.

Then we analyze the qualitative behaviour of the solutions of the ODE to determine the asymptotically stable sets.

The ODE-based analysis is elegant and conclusive and it further guarantees that the limit points of the stochastic recursion will almost surely belong to the compact connected internally chain transitive invariant set of the equivalent ODE.

Since the algorithm follows a multi-timescale stochastic approximation framework, we will also resort to the more generalized multi-timescale differential inclusion based analysis proposed in BID3 Ramaswamy and Bhatnagar, 2016) .Note that there exists only a unilateral coupling between the neural network (where the feature vectors θ t are calibrated by following a stochastic gradient descent w.r.t.

L slow ) and the various policy evaluation algorithms (see Figure 7 ).

This literally implies that the policy evaluation algorithms depend on the feature vectorsθ t but not vice-versa.

Therefore, one can independently analyze the asymptotic behaviour of the feature vectors {θ t } t∈N .

Also, as a technical requirement, note that since one cannot guarantee the stability (almost sure boundedness) of the iterates {θ t } t∈N (which is a necessary condition required for the ODE based analysis.

Please refer Chapter 2 of Borkar FORMULA2 ), we consider the following projected stochastic recursion: DISPLAYFORM0 where Γ Θ (·) is the projection onto a pre-determined compact and convex subset Θ ⊂ R m+d , i.e., Γ Θ (x) = x, for x ∈Θ, while for x / ∈Θ, it is the nearest point in Θ w.r.t.

the Euclidean distance (or equivalent metric).Define the filtration {F t } t∈N , a family of increasing natural σ-fields, where F t σ {θ i , S i , R i ; 0 ≤ i ≤ t} .The following lemma characterizes the limiting behaviour of the iterates {θ t } t∈N :Lemma 1.

Let Assumptions 1-3 hold.

Let Θ ⊂ R m+d be a compact, convex subset with smooth boundary.

Let Γ Θ be Frechet differentiable.

Further, let Γ Θ θ (− 1 2 ∇L slow )(θ) be Lipschitz continuous.

Let K be the set of asymptotically stable equilibria of the following ODE contained inside Θ: DISPLAYFORM1 Then the stochastic sequence {θ t } t∈N generated by the TTN converges almost surely to K.Proof.

We employ here the ODE based analysis as proposed in BID4 BID19 .

Firstly, we recall here the stochastic recursion which updatesθ t : DISPLAYFORM2 where Γ Θ is the projection onto a pre-determined compact and convex subset Θ ⊂ R m+d .

Here, DISPLAYFORM3 (m+d)×|S| is the Jacobian ofŶθ atθ =θ t and ∇θ tŶθ (s) is the column corresponding to state s.

where DISPLAYFORM0 Further,θ DISPLAYFORM1 where Γ Θ is the Frechet derivative (defined in Eq. (8).

Note that Γ Θ is single-valued since Θ is convex and also the above limit exists since the boundary ∂Θ is assumed smooth.

Further, for x ∈Θ, we have DISPLAYFORM2 i.e., Γ Θ x (·) is an identity map for x ∈Θ.A few observations are in order: DISPLAYFORM3 is a Lipschitz continuous function inθ.

This follows from the hypothesis of the Lemma.

DISPLAYFORM4 t+1 is a truncated martingale difference noise.

Indeed, it is easy to verify that the noise sequence {M 1 t+1 } t∈N is a martingale-difference noise sequence w.r.t to the filtration {F t+1 } t∈N , i.e., M 1 t+1 is F t+1 -measurable and integrable, ∀t ∈ N and E M 1 t+1 |F t = 0 a.s.

, ∀t ∈ N. Also, since Γ Θ (·) is a continuous linear operator, we have Γ Θ (M 1 t+1 ) to be F t+1 -measurable and integrable, ∀t ∈ N likewise.

DISPLAYFORM5 By taking t → ∞, C3 follows directly from the ergodicity (Levin and Peres, 2017) (Assumption 2) and finiteness of the underlying Markov chain.

C4: o(ξ t ) → 0 as t → ∞ (follows from Assumption 1).C5: Iterates {θ t } t∈N are stable (forcefully), i.e. bounded almost surely, since θ t ∈ Θ, ∀t ∈ N (ensured by the projection operator Γ Θ ) and Θ is compact (i.e., closed and bounded).

DISPLAYFORM6 This follows directly from the finiteness of the Markov chain and from the assumption that the boundary ∂Θ is smooth.

Now, by appealing to Theorem 2, Chapter 2 of BID4 ), we conclude that the stochastic recursion (10) asymptotically tracks the following ODE DISPLAYFORM7 In other words, the stochastic recursion (10) converges to the asymptotically stable equilibria of the ODE (15) contained inside Θ.Remark 1.

It is indeed non-trivial to determine the constraint set Θ without prior adequate knowledge about the limit set of the ODE (15).

A pragmatic approach to overcome this concern is to initiate the stochastic recursion with an arbitrary convex, compact set Θ with a smooth boundary and gradually spread to the whole of R m+d BID7 .Remark 2.

It is also important to characterize the hypothesis of the above lemma (i.e., Γ DISPLAYFORM8 is Lipschitz continuous) with respect to the featuresŶθ.

To achieve that one has to consider the non-projected form of the ODE (15).

Apparently, when one considers the spreading approach proposed in the above remark, then it is essentially encouraged to consider the non-projected form since the limiting flow of the ODE arising from the projected stochastic recursion is more likely to lie inside the compact, convex set as Θ becomes larger.

Thereupon, it is easy to observe that the conditionŶθ is twice continuously-differentiable is sufficient to ensure the Lipschitz continuity of Γ Θ θ (− 1 2 ∇L slow )(θ).

Additionally, in that case K = {θ|∇θL slow (θ) = 0} which is the set of local extrema of J.

One can directly apply the TD(λ) with linear function approximation algorithm to estimate the value function with respect to the features provided by the neural network.

The TD(λ) algorithm is provided in Algorithm 2.Here e t , w t ∈ R d .

Further, δ t R t+1 + γ t+1 w t x θt (S t+1 ) − w t x θt (S t ) is the temporal difference.

Parameters: α t > 0, λ ∈ [0, 1]; Initialization: w 0 = 0, e 0 = 0; DISPLAYFORM0 DISPLAYFORM1 Assumption 4-TD(λ): The pre-determined, deterministic, step-size sequence {α t } t∈N satisfies: DISPLAYFORM2 Note that the step-size schedules {α t } t∈N and {ξ t } t∈N satisfy ξt αt → 0, which implies that {ξ t } converges to 0 relatively faster than {α t }.

This disparity in terms of the learning rates induces an asynchronous convergence behaviour asymptotically BID3 , with feature parameter sequence {θ t } converging slower relative to the TD(λ) sequence {w t }.

The rationale being that the increment term of the underlying stochastic gradient descent of the neural network is smaller compared to that of the TD(λ) recursion FORMULA20 , since the neural network SGD is weighted by the step-size schedule {ξ t } t∈N which is smaller than {α t } t∈N for all but finitely many t. This unique pseudo heterogeneity induces multiple perspectives, i.e., when viewed from the faster timescale recursion (recursion controlled by α t ), the slower timescale recursion (recursion controlled by ξ t ) seems quasi-static ('almost a constant'), while viewed from the slower timescale, the faster timescale recursion seems equilibrated.

Further, it is analytically admissible BID3 to consider the slow timescale stochastic recursion (i.e., the neural network SGD) to be quasi-stationary (i.e.,θ t ≡θ, ∀t ∈ N), while analyzing the asymptotic behaviour of the relatively faster timescale stochastic recursion (17).

Thereupon, we obtain the following directly from Theorem 1 of (Tsitsiklis and Van Roy, 1997).

Lemma 2.

Assumeθ t ≡θ, ∀t ∈ N. Let Assumptions 1-3 and 4-TD(λ) hold.

Then for any λ ∈ [0, 1], the stochastic sequence {w t } t∈N generated by the TD(λ) algorithm (Algorithm 2) within the TTN setting converges almost surely to the limit w * , where w * satisfies DISPLAYFORM3 with DISPLAYFORM4 For other single-timescale prediction methods like ETD and LSPE, similar results follow.

Regarding the least squares method LSTD, which offers the significant advantage of non-dependency on stepsizes (albeit computationally expensive) couples smoothly within the TTN setting without any additional consideration.

However, one cannot directly apply the original GTD2 and TDC algorithms to the TTN setting, since a necessary condition required for the convergence of these algorithms is the non-singularity of the feature specific matrices E x θt (S t )x θt (S t ) and E (x θt (S t ) − γ t+1 x θt (S t+1 )) x θt (S t ) .

Please refer Theorem 1 and Theorem 2 of (Sutton et al., 2009) .

Without the non-singularity assumption, it is indeed hard to guarantee the almost sure boundedness of the GTD2/TDC iterates.

In the TTN setting that we consider in this paper, one cannot explicitly assure this condition, since the features are apparently administered by a neural network and it is not directly intuitive on how to control the neural network to generate a collection of features with the desired non-singularity characteristic.

Henceforth, one has to consider the projected versions of these algorithms.

We consider here the projected GTD2 algorithm provided in Algorithm 3.

Parameters: α t , β t ; Initialization: u 0 ∈ U, w 0 ∈ W ;For each transition (S t , R t+1 , S t+1 ) in O π do: DISPLAYFORM0 DISPLAYFORM1 Here DISPLAYFORM2 Here, Γ W (·) is the projection operator onto a pre-determined convex, compact subset W ⊂ R d with a smooth boundary ∂W .

Therefore, Γ W maps vectors in R d to the nearest vectors in W w.r.t.

the Euclidean distance (or equivalent metric).

Convexity and compactness ensure that the projection is unique and belongs to W .

Similarly, U is a pre-determined convex, compact subset of R d with a smooth boundary ∂U .Projection is required since the stability of the iterates {w t } t∈N and {u t } t∈N are hard to guarantee otherwise.

The pre-determined, deterministic, step-size sequences {α t } t∈N and {β t } t∈N satisfy DISPLAYFORM0 Define the filtration {F t } t∈N , a family of increasing natural σ-fields, where DISPLAYFORM1 Similar to the TD(λ) case, here also we follow the quasi-stationary argument.

Henceforth, we analyze the asymptotic behaviour of GTD2 algorithm under the assumption that feature vectorθ t is quasi-static, i.e.θ t ≡θ = (θ,w) .

Lemma 3.

Assumeθ t ≡θ = (θ,w) , ∀t ∈ N. Let Assumptions 1-3 and 4-GTD2 hold.

Then DISPLAYFORM2 where A * is the set of asymptotically stable equilibria of the following ODE: DISPLAYFORM3 and A u is the asymptotically stable equilibria of the following ODE: DISPLAYFORM4 , w(0) ∈W and t ∈ R + , with δ u defined in Eq. (29).Proof.

The two equations in the modified GTD2 algorithm constitute a multi-timescale stochastic approximation recursion, where there exists a bilateral coupling between the stochastic recursions (19) and (20) .

Since the step-size sequences {α t } t∈N and {β t } t∈N satisfy βt αt → 0, we have β t → 0 faster than α t → 0.

This disparity in terms of the learning rates induces a pseudo heterogeneous rate of convergence (or timescales) between the individual stochastic recursions which results in a pseudo asynchronous convergence behaviour when considered over a finite time window.

Also note that the coherent long-run behaviour of the multi-timescale stochastic recursion will asymptotically follow this short-term behaviour with the window size extending to infinity BID3 BID4 .

This pseudo behaviour induces multiple viewpoints, i.e., when observed from the faster timescale recursion (recursion controlled by α t ), the slower timescale recursion (recursion controlled by β t ) appears quasi-static ('almost a constant'), while observed from the slower timescale, the faster timescale recursion seems equilibrated.

Further, it is analytically admissible BID3 to consider the slow timescale stochastic recursion (20) to be quasi-stationary (i.e., u t ≡ u, ∀t ∈ N), while analyzing the limiting behaviour of the relatively faster timescale stochastic recursion 19.Analysis of the faster time-scale recursion: The faster time-scale stochastic recursion of the GTD2 algorithm is the following: DISPLAYFORM5 Under the previously mentioned quasi-stationary premise that u t ≡ u andθ t ≡θ = (θ,w) , ∀t ∈ N, thereupon, we analyze the long-term behaviour of the following recursion: DISPLAYFORM6 where x t = x θ (S t ) and δ u t+1 DISPLAYFORM7 The above equation can be rearranged as the following: DISPLAYFORM8 x t − w t x t x t and the bias DISPLAYFORM9 Similar to Equation (12), we can rewrite the above recursion as follows: DISPLAYFORM10 where Γ W wt (·) is the Frechet derivative (defined in Equation FORMULA22 ) of the projection operator Γ W .A few observations are in order:D1: The iterates {w t } t∈N are stable, i.e., sup t∈N w t < ∞ a.s.

This immediately follows since W is bounded.

DISPLAYFORM11 } t∈N is a martingale-difference noise sequence with respect to the filtration {F t+1 } t∈N .

This follows directly since {M 2 t+1 } t∈N is a martingale-difference noise sequence with respect to the same filtration.

DISPLAYFORM12 This follows directly from the finiteness of the underlying Markov chain and from the assumption that the boundary ∂W is smooth.

DISPLAYFORM13 is Lipschitz continuous with respect to w. Proof similar to C1.

DISPLAYFORM14 Now by appealing to Theorem 2, Chapter 2 of BID4 ) along with the above observations, we conclude that the stochastic recursion 23 asymptotically tracks the following ODE almost surely: DISPLAYFORM15 Therefore, w t converges asymptotically to the stable equilibria of the above ODE contained inside W almost surely.

Qualitative analysis of the solutions of ODE FORMULA2 : A trivial qualitative analysis of the long-run behaviour of the flow induced by the above ODE attests that the stable limit set is indeed the solutions of the following linear system inside W (This follows since Γ W w (y) = y for w ∈W and also because Γ W w (·) does not contribute any additional limit points on the boundary other than the roots of h 2 since ∂W is smooth).

DISPLAYFORM16 Note that E x t x t = Φ θ D dπ Φθ.

Claim 1: The above linear system of equations is consistent, i.e., E δ u t+1 x t ∈ R(Φ θ D dπ Φθ), i.e., the range-space of Φ θ D dπ Φθ: To see that, note that the above system can indeed be viewed as the least squares solution to the Φθw = δ u with respect to the weighted-norm · D dπ , where DISPLAYFORM17 whereR is the expected reward. (Note that E δ DISPLAYFORM18 The least-squares solution w 0 ∈ R d (which certainly exists but may not be unique) satisfies DISPLAYFORM19 Since Φ θ D dπ Φθ may be singular (i.e., Φ θ D dπ Φθ is not invertible), the above least squares solution may not be unique and hence the collection of asymptotically stable equilibria of the flow induced by the ODE (27) may not be singleton for every u. Let's denote the asymptotically stable equilibria of the flow induced by the said ODE to be the set A u , where ∅ = A u ⊆ W .Analysis of the slower time-scale recursion: The slower time-scale stochastic recursion of the GTD2 algorithm is the following: DISPLAYFORM20 Note that since ξt βt → 0, the stochastic recursion FORMULA2 is managed on a faster timescale relative to the the neural network stochastic recursion (10) and henceforth, we continue to maintain here the quasi-stationary conditionθ t ≡θ = (θ,w) .

Now the above equation can be rearranged as follows: DISPLAYFORM21 where ∆ wt t+1 DISPLAYFORM22 |F t and the bias DISPLAYFORM23 .

Similar to Equation (12), we can rewrite the above recursion as follows: DISPLAYFORM24 where Γ U ut (·) is the Frechet derivative (defined in Equation FORMULA22 ) of the projection operator Γ U .Now the above equation can be interpreted in terms of stochastic recursive inclusion as follows: DISPLAYFORM25 , where w ∈ A u .Indeed h 3 (u) = Γ U u (Bw) , where B = E (x t − γ t+1 x t+1 ) x t and w ∈ A u .

It is easy to DISPLAYFORM26 Here, one cannot directly apply the multi-timescale stochastic approximation results from BID3 since the said paper assumes that the limit point from the slower timescale recursion is unique (Please see Chapter 6 of BID4 ).

But in our setting, the slower timescale recursion (23) has several limit points (note that the stable limit set A u is not singleton).

This is where our analysis differs from that of the seminal paper on GTD2 algorithm, where it is assumed that both the matrices E x t x t and E (x t − γ t+1 x t+1 )x t are certainly non-singular.

However, in our TTN setting, one cannot guarantee this condition, since the features are apparently provided by a neural network and it is hard to fabricate the neural network to generate a collection of features with the desired non-singularity properties.

In order to analyze the limiting behaviour of the GTD2 algorithm under the relaxed singularity setting, henceforth one has to view the stochastic recursion (30) as a stochastic recursion inclusion BID1 and apply the recent results from (Ramaswamy and BID15 which analyzes the asymptotic behaviour of general multi-timescale stochastic recursive inclusions.

A few observations are in order:E1:

For each u ∈ U , h 3 (u) is a singleton: This follows from the definition of h 3 and Claim 1 above, where we established that each w ∈ A u is the least squares solution to the linear system of equations Φθw = δ u .

Therefore, it further implies that that h 3 is a Marchaud map as well.

E2: sup t∈N ( w t + u t ) < ∞ a.s.

This follows since W and U are bounded sets.

DISPLAYFORM27 } t∈N is a martingale-difference noise sequence with respect to the filtration {F t+1 } t∈N .

This follows directly since {M 3 t+1 } t∈N is a martingale-difference noise sequence with respect to the same filtration.

DISPLAYFORM28 This follows directly from the finiteness of the underlying Markov chain and from the assumption that the boundary ∂U is smooth.

E5: Γ U ut 3 t → 0 as t → ∞ a.s.

Proof similar to C3.

This implies that the bias is asymptotically irrelevant.

E6: For each u ∈ U , the set A u is a globally attracting set of the ODE (27) and is also Lyapunov stable.

Further, there exists K 4 ∈ (0, ∞) such that sup w∈Au w ≤ K 4 (1 + u ).

This follows since A u ⊆ W and W is bounded.

E7: The set-valued map q : U → {subsets of R d } given by q(u) = A u is upper-semicontinuous: Consider the convergent sequences {u n } n∈N → u and {w n } n∈N → w with u n ∈ U and w n ∈ q(u n ) = A u .

Note that w ∈ W , u ∈ U since W and U are compact.

Also Φ θ D dπ Φθw n = Φ θ D dπ δ un (from Claim 1).

Now taking limits on both sides we get DISPLAYFORM29 This implies that w ∈ A u = q(u).

The claim thus follows.

Thus we have established all the necessary conditions demanded by Theorem 3.10 of (Ramaswamy and Bhatnagar, 2016) to characterize the limiting behaviour of the stochastic recursive inclusion (33).Now by appealing to the said theorem, we obtain the following result on the asymptotic behaviour of the GTD2 algorithm: DISPLAYFORM30 where A * is the set of asymptotically stable equilibria of the following ODE: DISPLAYFORM31 One can obtain similar results for projected TDC.We now state our main result:

Theorem 2.

Let Θ ⊂ R m+d be a compact, convex subset with smooth boundary.

Let Γ Θ be Frechet differentiable.

Further, let Γ Θ θ (− 1 2 ∇L slow )(θ) be Lipschitz continuous.

Also, let Assumptions 1-3 hold.

Let K be the set of asymptotically stable equilibria of the following ODE contained inside Θ: DISPLAYFORM32 Then the stochastic sequence {θ t } t∈N generated by the TTN converges almost surely to K (sample path dependent).

Further, TD(λ) Convergence:

Under the additional Assumption 4-TD(λ), we obtain the following result: For any λ ∈ [0, 1], the stochastic sequence {w t } t∈N generated by the TD(λ) algorithm (Algorithm 2) within the TTN setting converges almost surely to the limit w * , where w * satisfies DISPLAYFORM33 with T (λ) defined in Lemma 2 andθ * ∈ K (sample path dependent).GTD2 Convergence: Let W, U ⊂ R d be compact, convex subsets with smooth boundaries.

Let Assumption 4-GTD2 hold.

Let Γ W and Γ U be Frechet differentiable.

Then the stochastic sequences {w t } t∈N and {u t } t∈N generated by the GTD2 algorithm (Algorithm 3) within the TTN setting satisfy DISPLAYFORM34 where A * is the set of asymptotically stable equilibria of the following ODE: DISPLAYFORM35 and A u is the asymptotically stable equilibria of the following ODE: DISPLAYFORM36

We also ran policy evaluation experiments on image-based catcher with 2 stacked 64x64 frames as input.

The policy evaluated was the same as was used in the non-image setting.

Similar to the non-imaged based catcher experiments, we have similar plots.

Published as a conference paper at ICLR 2019

In the classic Cartpole environment, the agent has to balance a pole on a cart.

The state is given by vector of 4 numbers (cart position, cart velocity, pole angle, pole velocity).

The two available actions are applying a force towards the left or the right.

Rewards are +1 at every timestep and an episode terminates once the pole dips below a certain angle or the cart moves too far from the center.

We use the OpenAI gym implementation BID6 .The policy to be evaluated consists of applying force in the direction the pole is moving with probability 0.9 (stabilizing the pole) or applying force in the direction of the cart's velocity with probability 0.1.

We inject some stochasticity so that the resulting policy does not perform overly well, which would lead to an uninteresting value function.

In Puck World (Tasfi, 2016) , the agent has to move in a two-dimensional box towards a good puck while staying away from a bad puck.

The 8-dimensional state consists of (player x location, player y location, player x velocity, player y velocity, good puck x location, good puck y location, bad puck x location, bad puck y location).

Each action increases the agent's velocity in one of the four cardinal directions apart from a "None" action which does nothing.

The reward is the negative distance to the good puck plus a penalty of −10 + x if the agent is within a certain radius of the bad puck, where x ∈ [−2, 0] depends on the distance to the bad puck (the reward is slightly modified from the original game to make the value function more interesting).The policy moves the agent towards the good puck, while having a soft cap on the agent's velocity.

In more detail, to choose one action, it is defined by the following procedure: First, we choose some eligible actions.

The None action is always eligible.

The actions which move the agent towards the good puck are also eligible.

For example, if the good puck is Northeast of the agent, the North and East actions are eligible.

If the agent's velocity in a certain direction is above 30, then the action for that direction is no longer eligible.

Finally, the agent picks uniformly at random from all eligible actions.

We run a preliminary experiment to check if TTN can have an advantage in the off-policy setting.

The target policy is the same as the one used for other Catcher experiments (described in Appendix D).The behaviour policy is slightly different.

If the apple is within 20 units (the target policy is 25 units), then the agent takes the action in the direction of the apple with probability 0.7 and one of the other two actions with probability 0.15 each.

If the apple is not within range, then the agent takes the None action 10% of the time and one of the other two with equal probability.

This combination of behaviour and target policies results in importance sampling ratios in the range of 0 to 8.7, moderately large values.

We try TTN with three off-policy algorithms (TD, TDC and LSTD) and compare to off-policy Nonlinear TD.

For TTN, the features are learnt optimizing the MSTDE on the behaviour policy while the values are learned off-policy.

The main difference between TTN and Nonlinear TD is the fact that Nonlinear TD does off-policy updates to the entire network while TTN only changes the linear part with off-policy updates.

From figure C.7, we see that TTN can outperform nonlinear TD in terms of average error and also has reduced variance (smaller error bars).

This seems to suggest that the TTN architecture can grant additional stability in the off-policy setting.

LSTD This version of LSTD was used in the policy evaluation experiments.

Inputs: η inv , λ, w 1 (initial weight vector) Initialization: DISPLAYFORM0 is the number of features For each transition (S t , R t+1 , S t+1 ) do: DISPLAYFORM1 Description of policies The policy evaluated in Puddle World randomly took the north and east actions with equal probability, with episodes initialized in the southwest corner.

The policy evaluated in Catcher increased the velocity in the direction of the apple if the agent was within 25 units or else chose the "None" action with a 20% chance of a random choice.

Competitor details Nonlinear TD uses the semi-gradient TD update with nonlinear function approximation.

This is known not to be theoretically-sound and there exist counterexamples where the weights and predictions diverge (Tsitsiklis and Van Roy, 1997).

Nevertheless, since this is the simplest algorithm, we use this as a baseline.

Nonlinear GTD, conversely, has proven convergence results as it is an extension of the gradient TD methods to nonlinear function approximation.

Castro et al. proposed three variants of adaptive bases algorithms, each optimizing a different objective.

We include two of them: ABTD, which is based on the plain TD update and ABBE, which is derived from the MSTDE.

We omit ABPBE, which optimizes the MSPBE, since the algorithm is computationally inefficient, requiring O(d 2 m) memory, where d is the number of features in the last layer and m is the number of parameters for the bases (ie. weights in the neural network).

Also, the derivation is similar in spirit to that of nonlinear GTD, so we would expect both to perform similarly.

Levine et al.'s algorithm was proposed for the control setting, combining DQN with periodic LSTD resolving for the last layer's weights.

We adapt their idea for policy evaluation by adding regularization to the last layer's weights to bias them towards the LSTD solution and using semi-gradient TD on this objective.

Then, we can then train in a fully online manner, which makes the algorithm comparable to the other competitors.

This algorithm is labeled as Nonlinear TD-Reg.

Hyperparameters Here we present the refined hyperparameter ranges that were swept over for the experimental runs.

First, we outline the hyperparameters of all the algorithms and, afterwards, we give the values tested for each hyperparameter.

There is one range for Puddle World and another for Catcher (both versions) since different ranges worked well for each domain.

For the two-timescale networks, for each environment, the learning rate for the slow part was set to a fixed value which was sufficiently low for the fast part to do well but was not tuned extensively.

This was done for all the experiments except for those on surrogate losses, where we swept over a range of values (since different losses could have different optimization properties).

For all algorithms, the learning rate for the slow part α slow .

TD, ETD, TOTD, TOETD : learning rate α, trace parameter λ TDC : primary weights learning rate α, secondary weights learning rate β, trace parameter λ LSTD: initializer for the A −1 matrix η inv , trace parameter λ FLSTD: initializer for the A matrix η, learning rate α, forgetting rate ψ, trace parameter λ Castro MSTDE, Castro TD: final layer learning rate α, other layers learning rate β Nonlinear GTD : primary weights learning rate α, secondary weights learning rate β Nonlinear TD : learning rate α, trace parameter λ Nonlinear TD -Reg : learning rate α, learning rate towards LSTD solution (regularization) β, initializer for the A matrix η For all algorithms that used eligibility traces, the range swept over was λ ∈ {0, 0.3, 0.5, 0.9, 0.99, 1} The other hyperparameters are shown individually below.

Control experiments For both control experiments, we modified the Catcher environment slightly from its default settings.

The agent is given only 1 life and an episode terminates after it fails to catch a single apple.

The reward for catching an apple is +1 and is -1 at the end of an episode.

The discount factor is set to 0.99.For image-based Catcher, we stack two consecutive frames which we treat as the state.

This is done so that the agent can perceive movements in the paddle, thus making the state Markov.

Both DQN and TTN use an -greedy policy.

For DQN, is annealed from 1.0 to 0.01 (0.1) for nonimage (image) Catcher over a certain number of steps.

For TTN, is fixed to a constant value, 0.01 (0.1) for nonimage (image) Catcher.

For both algorithms, we use a replay buffer of size 50000 (200000) for nonimage (image) Catcher.

The buffer is initialized with 5000 (50000) transitions from a random policy.

The minibatch size for DQN and feature learning was 32.

TTN uses the AMSGrad optimizer with β 1 = 0 and β 2 = 0.99 and DQN uses the ADAM optimizer with default settings, β 1 = 0.9, β 2 = 0.999.For image catcher, due to the long training times, the hyperparameters were manually tuned.

For nonimage catcher, we concentrated our hyperparameter tuning efforts on the most important ones and use the following strategy.

We first used a preliminary search to find a promising range of values, followed by a grid search.

For TTN with FQI, we focused tuning on the step sizes for the features and the regularization factor.

For DQN, we focused on adjusting the step size and the number of steps over which , the probability of picking a random action, was annealed.

The other hyperparameters were left to reasonable values.

The final hyperparameters for nonimage catcher:TTN -α slow = 10 −3 , λ reg = 10 −2 (regularization towards previous weights) DQN -α = 10 −3.75 , decaying over 20 thousand steps, update the target network every 1000 steps.

For LS-DQN, do a FQI upate every 50,000 steps with a regularization weight of 1.The final hyperparameters for image catcher:TTN -α slow = 10 −5 , λ reg = 10 −3 (regularization towards previous weights), solve for new weights using FQI every 10,000 steps.

−3.75 , decaying over 1 million steps, update the target network every 10,000 steps.

For LS-DQN, do a FQI update every 500,000 steps with a regularization weight of 1.

@highlight

We propose an architecture for learning value functions which allows the use of any linear policy evaluation algorithm in tandem with nonlinear feature learning.

@highlight

The paper proposes a two-timescale framework for learning the value function and a state representation altogether with nonlinear approximators.

@highlight

This paper proposes Two-Timescale Networks (TTNs) and prove the convergence of this method using methods from two time-scale stochastic approximation. 

@highlight

This paper presents a Two-Timescale Network (TTN) that enables linear methods to be used to learn values. 