Many real-world sequential decision-making problems can be formulated as optimal control with high-dimensional observations and unknown dynamics.

A promising approach is to embed the high-dimensional observations into a lower-dimensional latent representation space, estimate the latent dynamics model, then utilize this model for control in the latent space.

An important open question is how to learn a representation that is amenable to existing control algorithms?

In this paper, we focus on learning representations for locally-linear control algorithms, such as iterative LQR (iLQR).

By formulating and analyzing the representation learning problem from an optimal control perspective, we establish three underlying principles that the learned representation should comprise: 1) accurate prediction in the observation space, 2) consistency between latent and observation space dynamics, and 3) low curvature in the latent space transitions.

These principles naturally correspond to a loss function that consists of three terms: prediction, consistency, and curvature (PCC).

Crucially, to make PCC tractable, we derive an amortized variational bound for the PCC loss function.

Extensive experiments on benchmark domains demonstrate that the new variational-PCC learning algorithm benefits from significantly more stable and reproducible training, and leads to superior control performance.

Further ablation studies give support to the importance of all three PCC components for learning a good latent space for control.

Decomposing the problem of decision-making in an unknown environment into estimating dynamics followed by planning provides a powerful framework for building intelligent agents.

This decomposition confers several notable benefits.

First, it enables the handling of sparse-reward environments by leveraging the dense signal of dynamics prediction.

Second, once a dynamics model is learned, it can be shared across multiple tasks within the same environment.

While the merits of this decomposition have been demonstrated in low-dimensional environments (Deisenroth & Rasmussen, 2011; Gal et al., 2016) , scaling these methods to high-dimensional environments remains an open challenge.

The recent advancements in generative models have enabled the successful dynamics estimation of high-dimensional decision processes (Watter et al., 2015; Ha & Schmidhuber, 2018; Kurutach et al., 2018) .

This procedure of learning dynamics can then be used in conjunction with a plethora of decision-making techniques, ranging from optimal control to reinforcement learning (RL) (Watter et al., 2015; Banijamali et al., 2018; Finn et al., 2016; Chua et al., 2018; Ha & Schmidhuber, 2018; Kaiser et al., 2019; Hafner et al., 2018; Zhang et al., 2019) .

One particularly promising line of work in this area focuses on learning the dynamics and conducting control in a low-dimensional latent embedding of the observation space, where the embedding itself is learned through this process (Watter et al., 2015; Banijamali et al., 2018; Hafner et al., 2018; Zhang et al., 2019) .

We refer to this approach as learning controllable embedding (LCE).

There have been two main approaches to this problem: 1) to start by defining a cost function in the high-dimensional observation space and learn the embedding space, its dynamics, and reward function, by interacting with the environment in a RL fashion (Hafner et al., 2018; Zhang et al., 2019) , and 2) to first learn the embedding space and its dynamics, and then define a cost function in this low-dimensional space and conduct the control (Watter et al., 2015; Banijamali et al., 2018) .

This can be later combined with RL for extra fine-tuning of the model and control.

In this paper, we take the second approach and particularly focus on the important question of what desirable traits should the latent embedding exhibit for it to be amenable to a specific class of control/learning algorithms, namely the widely used class of locally-linear control (LLC) algorithms?

We argue from an optimal control standpoint that our latent space should exhibit three properties.

The first is prediction: given the ability to encode to and decode from the latent space, we expect the process of encoding, transitioning via the latent dynamics, and then decoding, to adhere to the true observation dynamics.

The second is consistency: given the ability to encode a observation trajectory sampled from the true environment, we expect the latent dynamics to be consistent with the encoded trajectory.

Finally, curvature: in order to learn a latent space that is specifically amenable to LLC algorithms, we expect the (learned) latent dynamics to exhibit low curvature in order to minimize the approximation error of its first-order Taylor expansion employed by LLC algorithms.

Our contributions are thus as follows: (1) We propose the Prediction, Consistency, and Curvature (PCC) framework for learning a latent space that is amenable to LLC algorithms and show that the elements of PCC arise systematically from bounding the suboptimality of the solution of the LLC algorithm in the latent space.

(2) We design a latent variable model that adheres to the PCC framework and derive a tractable variational bound for training the model.

(3) To the best of our knowledge, our proposed curvature loss for the transition dynamics (in the latent space) is novel.

We also propose a direct amortization of the Jacobian calculation in the curvature loss to help training with curvature loss more efficiently.

(4) Through extensive experimental comparison, we show that the PCC model consistently outperforms E2C (Watter et al., 2015) and RCE (Banijamali et al., 2018) on a number of control-from-images tasks, and verify via ablation, the importance of regularizing the model to have consistency and low-curvature.

We are interested in controlling the non-linear dynamical systems of the form s t+1 = f S (s t , u t ) + w, over the horizon T .

In this definition, s t ??? S ??? R ns and u t ??? U ??? R nu are the state and action of the system at time step t ??? {0, . . .

, T ??? 1}, w is the Gaussian system noise, and f S is a smooth non-linear system dynamics.

We are particularly interested in the scenario in which we only have access to the high-dimensional observation x t ??? X ??? R nx of each state s t (n x n s ).

This scenario has application in many real-world problems, such as visual-servoing (Espiau et al., 1992) , in which we only observe high-dimensional images of the environment and not its underlying state.

We further assume that the high-dimensional observations x have been selected such that for any arbitrary control sequence U = {u t } T ???1 t=0 , the observation sequence {x t } T t=0 is generated by a stationary Markov process, i.e., x t+1 ??? P (??|x t , u t ), ???t ??? {0, . . .

, T ??? 1}.

A common approach to control the above dynamical system is to solve the following stochastic optimal control (SOC) problem (Shapiro et al., 2009 ) that minimizes expected cumulative cost: where c t : X ?? U ??? R ???0 is the immediate cost function at time t, c T ??? R ???0 is the terminal cost, and x 0 is the observation at the initial state s 0 .

Note that all immediate costs are defined in the observation space X , and are bounded by c max > 0 and Lipschitz with constant c lip > 0.

For example, in visualservoing, (SOC1) can be formulated as a goal tracking problem (Ebert et al., 2018) , where we control the robot to reach the goal observation x goal , and the objective is to compute a sequence of optimal open-loop actions U that minimizes the cumulative tracking error

Since the observations x are high dimensional and the dynamics in the observation space P (??|x t , u t ) is unknown, solving (SOC1) is often intractable.

To address this issue, a class of algorithms has been recently developed that is based on learning a low-dimensional latent (embedding) space Z ??? R nz (n z n x ) and latent state dynamics, and performing optimal control there.

This class that we refer to as learning controllable embedding (LCE) throughout the paper, include recently developed algorithms, such as E2C (Watter et al., 2015) , RCE (Banijamali et al., 2018), and SOLAR (Zhang et al., 2019) .

The main idea behind the LCE approach is to learn a triplet, (i) an encoder E : X ??? P(Z); (ii) a dynamics in the latent space F : Z ?? U ??? P(Z); and (iii) a decoder D : Z ??? P(X ).

These in turn can be thought of as defining a (stochastic) mapping P : X ?? U ??? P(X ) of the form P =

D ??? F ??? E. We then wish to solve the SOC in latent space Z:

such that the solution of (SOC2), U 3 PCC MODEL: A CONTROL PERSPECTIVE As described in Section 2, we are primarily interested in solving (SOC1), whose states evolve under dynamics P , as shown at the bottom row of Figure 1 (a) in (blue).

However, because of the difficulties in solving (SOC1), mainly due to the high dimension of observations x, LCE proposes to learn a mapping P by solving (SOC2) that consists of a loss function, whose states evolve under dynamics F (after an initial transition by encoder E), as depicted in Figure 1 (b), and a regularization term.

The role of the regularizer R 2 is to account for the performance gap between (SOC1) and the loss function of (SOC2), due to the discrepancy between their evolution paths, shown in Figures 1(a)(blue) and 1(b)(green).

The goal of LCE is to learn P of the particular form P = D ??? F ??? E, described in Section 2, such that the solution of (SOC2) has similar performance to that of (SOC1).

In this section, we propose a principled way to select the regularizer R 2 to achieve this goal.

Since the exact form of (SOC2) has a direct effect on learning P , designing this regularization term, in turn, provides us with a recipe (loss function) to learn the latent (embedded) space Z. In the following subsections, we show that this loss function consists of three terms that correspond to prediction, consistency, and curvature, the three ingredients of our PCC model.

Note that these two SOCs evolve in two different spaces, one in the observation space X under dynamics P , and the other one in the latent space Z (after an initial transition from X to Z) under dynamics F .

Unlike P and F that only operate in a single space, X and Z, respectively, P can govern the evolution of the system in both X and Z (see Figure 1 (c)).

Therefore, any recipe to learn P , and as a result the latent space Z, should have at least two terms, to guarantee that the evolution paths resulted from P in X and Z are consistent with those generated by P and F .

We derive these two terms, that are the prediction and consistency terms in the loss function used by our PCC model, in Sections 3.1 and 3.2, respectively.

While these two terms are the result of learning P in general SOC problems, in Section 3.3, we concentrate on the particular class of LLC algorithms (e.g., iLQR (Li & Todorov, 2004) ) to solve SOC, and add the third term, curvature, to our recipe for learning P .

Figures 1(a)(blue) and 1(c)(red) show the transition in the observation space under P and P , where x t is the current observation, and x t+1 andx t+1 are the next observations under these two dynamics, respectively.

Instead of learning a P with minimum mismatch with P in terms of some distribution norm, we propose to learn P by solving the following SOC:

whose loss function is the same as the one in (SOC1), with the true dynamics replaced by P .

In Lemma 1 (see Appendix A.1, for proof), we show how to set the regularization term R 3 in (SOC3), such that the control sequence resulted from solving (SOC3), U

In Section 3.1, we provided a recipe for learning P (in form of D ??? F ??? E) by introducing an intermediate (SOC3) that evolves in the observation space X according to dynamics P .

In this section we first connect (SOC2) that operates in Z with (SOC3) that operates in X .

For simplicity and without loss generality, assume the initial cost c 0 (x, u) is zero.

4 Lemma 2 (see Appendix A.2, for proof)

suggests how we shall set the regularizer in (SOC2), such that its solution performs similarly to that of (SOC3), under their corresponding dynamics models.

Lemma 2.

Let (U * 3 , P * 3 ) be a solution to (SOC3) and (U * 2 , P * 2 ) be a solution to (SOC2) with

and

Similar to Lemma 1, in Eq. 2, the expectation is over the state-action stationary distribution of the policy used to generate the training samples.

Moreover,

are the probability over the next latent state z , given the current observation x and action u, in (SOC2) and (SOC3) (see the paths x t ??? z t ???z t+1 and x t ??? z t ???z t+1 ???x t+1 ?????? t+1 in Figures 1(b)(green) and 1(c)(red)).

Therefore R 2 ( P ) can be interpreted as the measure of discrepancy between these models, which we term as consistency loss.

Although Lemma 2 provides a recipe to learn P by solving (SOC2) with the regularizer (2), unfortunately this regularizer cannot be computed from the data -that is of the form (x t , u t , x t+1 ) -because the first term in the D KL requires marginalizing over current and next latent states (z t andz t+1 in Figure 1 (c)).

To address this issue, we propose to use the (computable) regularizer

in which the expectation is over (x, u, x ) sampled from the training data.

Corollary 1 (see Appendix A.3, for proof) bounds the performance loss resulted from using R 2 ( P ) instead of R 2 ( P ), and shows that it could be still a reasonable choice.

Corollary 1.

Let (U * 3 , P * 3 ) be a solution to (SOC3) and (U * 2 , P * 2 ) be a solution to (SOC2) with R 2 ( P ) and and ?? 2 defined by (3) and (2).

Then, we have L(U *

where R 3 ( P ) and R 2 ( P ) are defined by (1) and (3).

Then, we have

3.3 LOCALLY-LINEAR CONTROL IN THE LATENT SPACE AND CURVATURE REGULARIZATION In Sections 3.1 and 3.2, we derived a loss function to learn the latent space Z. This loss function, that was motivated by the general SOC perspective, consists of two terms to enforce the latent space to not only predict the next observations accurately, but to be suitable for control.

In this section, we focus on the class of locally-linear control (LLC) algorithms (e.g., iLQR), for solving (SOC2), and show how this choice adds a third term, that corresponds to curvature, to the regularizer of (SOC2), and as a result, to the loss function of our PCC model.

The main idea in LLC algorithms is to iteratively compute an action sequence to improve the current trajectory, by linearizing the dynamics around this trajectory, and use this action sequence to generate the next trajectory (see Appendix B for more details about LLC and iLQR).

This procedure implicitly assumes that the dynamics is approximately locally linear.

To ensure this in (SOC2), we further restrict the dynamics P and assume that it is not only of the form P = D ??? F ??? E, but F , the latent space dynamics, has low curvature.

One way to ensure this in (SOC2) is to directly impose a penalty over the curvature of the latent space transition function

where w is a Gaussian noise.

Consider the following SOC problem:

where R 2 is defined by (4); U is optimized by a LLC algorithm, such as iLQR; R LLC ( P ) is given by,

where = ( z , u ) ??? N (0, ?? 2 I), ?? > 0 is a tunable parameter that characterizes the "diameter" of latent state-action space in which the latent dynamics model has low curvature.

, where 1/X is the minimum non-zero measure of the sample distribution w.r.t.

X , and 1 ??? ?? ??? [0, 1) is a probability threshold.

Lemma 4 (see Appendix A.5, for proof and discussions on how ?? affects LLC performance) shows that a solution of (SOC-LLC) has similar performance to a solution of (SOC1, and thus, (SOC-LLC) is a reasonable optimization problem to learn P , and also the latent space Z.

Lemma 4.

Let (U * LLC , P * LLC ) be a LLC solution to (SOC-LLC) and U * 1 be a solution to (SOC1).

Suppose the nominal latent state-action trajectory {(z z z t , u u u t )} T ???1 t=0 satisfies the condition:

t=0 is the optimal trajectory of (SOC2).

Then with proba-

In practice, instead of solving (SOC-LLC) jointly for U and P , we treat (SOC-LLC) as a bi-level optimization problem, first, solve the inner optimization problem for P , i.e.,

where R 3 ( P ) = ???E x,u,x [log P (x |x, u)] is the negative log-likelihood, 5 and then, solve the outer optimization problem, min U L(U, F * ,c, z 0 ), where P * = D * ??? F * ??? E * , to obtain the optimal control sequence U * .

Solving (SOC-LLC) this way is an approximation, in general, but is justified, when the regularization parameter ?? LLC is large.

Note that we leave the regularization parameters (?? p , ?? c , ?? cur ) as hyper-parameters of our algorithm, and do not use those derived in the lemmas of this section.

Since the loss for learning P * in (PCC-LOSS) enforces (i) prediction accuracy, (ii) consistency in latent state prediction, and (iii) low curvature over f Z , through the regularizers R 3 , R 2 , and R LLC , respectively, we refer to it as the prediction-consistency-curvature (PCC) loss.

The PCC-Model objective in (PCC-LOSS) introduces the optimization problem min P ?? p R 3 ( P ) + ?? c R 2 ( P ) + ?? cur R LLC ( P ).

To instantiate this model in practice, we de-

In this section, we propose a variational approximation to the intractable negative log-likelihood R 3 and batch-consistency R 2 losses, and an efficient approximation of the curvature loss R LLC .

The negative log-likelihood 6 R 3 admits a variational bound via Jensen's Inequality,

which holds for any choice of recognition model Q. For simplicity, we assume the recognition model employs bottom-up inference and thus factorizes as Q(z t ,??? t+1 |x t , x t+1 , u t ) = Q(??? t+1 |x t+1 )Q(z t |??? t+1 , x t , u t ).

The main idea behind choosing a backward-facing model is to allow the model to learn to account for noise in the underlying dynamics.

We estimate the expectations in (6) via Monte Carlo simulation.

To reduce the variance of the estimator, we decompose R 3,NLE-Bound further into

log P (???t+1 | zt, ut) , and note that the Entropy H(??) and Kullback-Leibler D KL (?? ??) terms are analytically tractable when Q is restricted to a suitably chosen variational family (i.e. in our experiments, Q(??? t+1 | x t+1 ) and Q(z t |??? t+1 , x t , u t ) are factorized Gaussians).

The derivation is provided in Appendix C.1.

Interestingly, the consistency loss R 2 admits a similar treatment.

We note that the consistency loss seeks to match the distribution of??? t+1 | x t , u t with z t+1 | x t+1 , which we represent below as

Here, P (??? t+1 | x t , u t ) is intractable due to the marginalization of z t .

We employ the same procedure as in (6) to construct a tractable variational bound

We now make the further simplifying assumption that

).

This allows us to rewrite the expression as

which is a subset of the terms in (6).

See Appendix C.2 for a detailed derivation.

In practice we use a variant of the curvature loss where Taylor expansions and gradients are evaluated

(8) When n z is large, evaluation and differentiating through the Jacobians can be slow.

To circumvent this issue, the Jacobians evaluation can be amortized by treating the Jacobians as the coefficients of the best linear approximation at the evaluation point.

This leads to a new amortized curvature loss

where A and B are function approximators to be optimized.

Intuitively, the amortized curvature loss seeks-for any given (z, u)-to find the best choice of linear approximation induced by A(z, u) and B(z, u) such that the behavior of F ?? in the neighborhood of (z, u) is approximately linear.

In this section, we highlight the key differences between PCC and the closest previous works, namely E2C and RCE.

A key distinguishing factor is PCC's use of a nonlinear latent dynamics model paired with an explicit curvature loss.

In comparison, E2C and RCE both employed "locally-linear dynamics" of the form z = A(z,??)z + B(z,??)u + c(z,??) wherez and?? are auxiliary random variables meant to be perturbations of z and u. When contrasted with (9), it is clear that neither A and B in the E2C/RCE formulation can be treated as the Jacobians of the dynamics, and hence the curvature of the dynamics is not being controlled explicitly.

Furthermore, since the locally-linear dynamics are wrapped inside the maximum-likelihood estimation, both E2C and RCE conflate the two key elements prediction and curvature together.

This makes controlling the stability of training much more difficult.

Not only does PCC explicitly separate these two components, we are also the first to explicitly demonstrate theoretically and empirically that the curvature loss is important for iLQR.

Furthermore, RCE does not incorporate PCC's consistency loss.

Note that PCC, RCE, and E2C are all Markovian encoder-transition-decoder frameworks.

Under such a framework, the sole reliance on minimizing the prediction loss will result in a discrepancy between how the model is trained (maximizing the likelihood induced by encoding-transitioning-decoding) versus how it is used at test-time for control (continual transitioning in the latent space without ever decoding).

By explicitly minimizing the consistency loss, PCC reduces the discrapancy between how the model is trained versus how it is used at test-time for planning.

Interestingly, E2C does include a regularization term that is akin to PCC's consistency loss.

However, as noted by the authors of RCE, E2C's maximization of pair-marginal log-likelihoods of (x t , x t+1 ) as opposed to the conditional likelihood of x t+1 given x t means that E2C does not properly minimize the prediction loss prescribed by the PCC framework.

In this section, we compare the performance of To generate our training and test sets, each consists of triples (x t , u t , x t+1 ), we: (1) sample an underlying state s t and generate its corresponding observation x t , (2) sample an action u t , and (3) obtain the next state s t+1 according to the state transition dynamics, add it a zero-mean Gaussian noise with variance ?? 2 I ns , and generate corresponding observation x t+1 .To ensure that the observation-action data is uniformly distributed (see Section 3), we sample the state-action pair (s t , u t ) uniformly from the state-action space.

To understand the robustness of each model, we consider both deterministic (?? = 0) and stochastic scenarios.

In the stochastic case, we add noise to the system with different values of ?? and evaluate the models' performance under various degree of noise.

Each task has underlying start and goal states that are unobservable to the algorithms, instead, the algorithms have access to the corresponding start and goal observations.

We apply control using the iLQR algorithm (see Appendix B), with the same cost function that was used by RCE and E2C, namely,c(z

where z goal is obtained by encoding the goal observation, and Q = ?? ?? I nz , R = I nu .

Details of our implementations are specified in Appendix D.3.

We report performance in the underlying system, specifically the percentage of time spent in the goal region 10 .

A Reproducible Experimental Pipeline In order to measure performance reproducibility, we perform the following 2-step pipeline.

For each control task and algorithm, we (1) train 10 models independently, and (2) solve 10 control tasks per model (we do not cherry-pick, but instead perform a total of 10 ?? 10 = 100 control tasks).

We report statistics averaged over all the tasks (in addition, we report the best performing model averaged over its 10 tasks).

By adopting a principled and statistically reliable evaluation pipeline, we also address a pitfall of the compared baselines where the best model needs to be cherry picked, and training variance was not reported.

7 Code will become available with the camera-ready version.

8 For the RCE implementation, we directly optimize the ELBO loss in Equation (16) of the paper.

We also tried the approach reported in the paper on increasing the weights of the two middle terms and then annealing them to 1.

However, in practice this method is sensitive to annealing schedule and has convergence issues.

9 See a control demo on the TORCS simulator at https://youtu.be/GBrgALRZ2fw 10 Another possible metric is the average distance to goal, which has a similar behavior.

18.8 ?? 2.1 9.1 ?? 1.5 13.1 ?? 1.9 11.5 ?? 1.8

Results Table 1 shows how PCC outperforms the baseline algorithms in the noiseless dynamics case by comparing means and standard deviations of the means on the different control tasks (for the case of added noise to the dynamics, which exhibits similar behavior, refer to Appendix E.1).

It is important to note that for each algorithm, the performance metric averaged over all models is drastically different than that of the best model, which justifies our rationale behind using the reproducible evaluation pipeline and avoid cherry-picking when reporting.

Figure 2 depicts 2 instances (randomly chosen from the 10 trained models) of the learned latent space representations on the noiseless dynamics of Planar and Inverted Pendulum tasks for PCC, RCE, and E2C models (additional representations can be found in Appendix E.2).

Representations were generated by encoding observations corresponding to a uniform grid over the state space.

Generally, PCC has a more interpretable representation of both Planar and Inverted Pendulum Systems than other baselines for both the noiseless dynamics case and the noisy case.

Finally, in terms of computation, PCC demonstrates faster training with 64% improvement over RCE, and 2% improvement over E2C.

Ablation Analysis On top of comparing the performance of PCC to the baselines, in order to understand the importance of each component in (PCC-LOSS), we also perform an ablation analysis on the consistency loss (with/without consistency loss) and the curvature loss (with/without curvature loss, and with/without amortization of the Jacobian terms).

Table 2 shows the ablation analysis of PCC on the aforementioned tasks.

From the numerical results, one can clearly see that when consistency loss is omitted, the control performance degrades.

This corroborates with the theoretical results in Section 3.2, which indicates the relationship of the consistency loss and the estimation error between the next-latent dynamics prediction and the next-latent encoding.

This further implies that as the consistency term vanishes, the gap between control objective function and the model training loss is widened, due to the accumulation of state estimation error.

The control performance also decreases when one removes the curvature loss.

This is mainly attributed to the error between the iLQR control algorithm and (SOC2).

Although the latent state dynamics model is parameterized with neural networks, which are smooth, without enforcing the curvature loss term the norm of the Hessian (curvature) might still be high.

This also confirms with the analysis in Section 3.3 about sub-optimality performance and curvature of latent dynamics.

Finally, we observe that the performance of models trained without amortized curvature loss are slightly better than with their amortized counterpart, however, since the amortized curvature loss does not require computing gradient of the latent dynamics (which means that in stochastic optimization one does not need to estimate its Hessian), we observe relative speed-ups in model training with the amortized version (speed-up of 6%, 9%, and 15% for Planar System, Inverted Pendulum, and Cartpole, respectively).

In this paper, we argue from first principles that learning a latent representation for control should be guided by good prediction in the observation space and consistency between latent transition and the embedded observations.

Furthermore, if variants of iterative LQR are used as the controller, the low-curvature dynamics is desirable.

All three elements of our PCC models are critical to the stability of model training and the performance of the in-latent-space controller.

We hypothesize that each particular choice of controller will exert different requirement for the learned dynamics.

A future direction is to identify and investigate the additional bias for learning an effective embedding and latent dynamics for other type of model-based control and planning methods.

where D TV is the total variation distance of two distributions.

The first inequality is based on the result of the above lemma, the second inequality is based on Pinsker's inequality (Ordentlich & Weinberger, 2005) , and the third inequality is based on Jensen's inequality (Boyd & Vandenberghe, 2004) of (??) function.

Now consider the expected cumulative KL cost:

t=0 KL(P (??|x t , u t )|| P (??|x t , u t )) | P, x 0 with respect to some arbitrary control action sequence {u t } T ???1 t=0 .

Notice that this arbitrary action sequence can always be expressed in form of deterministic policy u t = ?? (x t , t) with some nonstationary state-action mapping ?? .

Therefore, this KL cost can be written as:

where the expectation is taken over the state-action occupation measure

t=0 P(x t = x, u t = u|x 0 , U ) of the finite-horizon problem that is induced by data-sampling policy U .

The last inequality is due to change of measures in policy, and the last inequality is due to the facts that (i) ?? is a deterministic policy, (ii) dU (u t ) is a sampling policy with lebesgue measure 1/U over all control actions, (iii) the following bounds for importance sampling factor holds:

To conclude the first part of the proof, combining all the above arguments we have the following inequality for any model P and control sequence U :

For the second part of the proof, consider the solution of (SOC3), namely (U * 3 , P * 3 ).

Using the optimality condition of this problem one obtains the following inequality:

Using the results in (11) and (12), one can then show the following chain of inequalities:

where U * 1 is the optimizer of (SOC1) and (U * 3 , P *

3 ) is the optimizer of (SOC3).

Therefore by letting ?? 3 = ??? 2T 2 ?? c max U and R 3 ( P ) = E x,u KL(P (??|x, u)|| P (??|x, u)) and by combining all of the above arguments, the proof of the above lemma is completed.

A.2 PROOF OF LEMMA 2

For the first part of the proof, at any time-step t ??? 1, for any arbitrary control action sequence {u t } T ???1 t=0 , and any model P , consider the following decomposition of the expected cost :

.

Now consider the following cost function: E[c(x t???1 , u t???1 ) + c(x t , u t ) | P , x 0 ] for t > 2.

Using the above arguments, one can express this cost as

By continuing the above expansion, one can show that

where the last inequality is based on Jensen's inequality of (??) function.

For the second part of the proof, following similar arguments as in the second part of the proof of Lemma 1, one can show the following chain of inequalities for solution of (SOC3) and (SOC2):

where the first and third inequalities are based on the first part of this Lemma, and the second inequality is based on the optimality condition of problem (SOC2).

This completes the proof.

To start with, the total-variation distance D TV x ???X d P (x |x, u)E(??|x )||(F ??? E)(??|x, u) can be bounded by the following inequality using triangle inequality:

where the second inequality follows from the convexity property of the D TV -norm (w.r.t.

convex weights E(??|x ), ???x ).

Then by Pinsker's inequality, one obtains the following inequality:

We now analyze the batch consistency regularizer:

and connect it with the inequality in (15).

Using Jensen's inequality of convex function x log x, for any observation-action pair (x, u) sampled from U ?? , one can show that

Therefore, for any observation-control pair (x, u) the following inequality holds:

By taking expectation over (x, u) one can show that

is the lower bound of the batch consistency regularizer.

Therefore, the above arguments imply that

The inequality is based on the property that

Equipped with the above additional results, the rest of the proof on the performance bound follows directly from the results from Lemma 2, in which here we further upper-bound

A.4 PROOF OF LEMMA 3

For the first part of the proof, at any time-step t ??? 1, for any arbitrary control action sequence {u t } T ???1 t=0 and for any model P , consider the following decomposition of the expected cost :

Under review as a conference paper at ICLR 2020

Now consider the following cost function: E[c(x t???1 , u t???1 ) + c(x t , u t ) | P , x 0 ] for t > 2.

Using the above arguments, one can express this cost as

Continuing the above expansion, one can show that

where the last inequality is based on the fact that

and is based on Jensen's inequality of (??) function.

For the second part of the proof, following similar arguments from Lemma 2, one can show the following inequality for the solution of (SOC3) and (SOC2):

where the first and third inequalities are based on the first part of this Lemma, and the second inequality is based on the optimality condition of problem (SOC2).

This completes the proof.

A Recap of the Result: Let (U * LLC , P * LLC ) be a LLC solution to (SOC-LLC) and U * 1 be a solution to (SOC1).

Suppose the nominal latent state-action pair {(z z z t , u u u t )} T ???1 t=0 satisfies the condition: (z z z t , u u u t ) ??? N ((z * 2,t , u * 2,t ), ?? 2 I), where {(z * 2,t , u * 2,t } T ???1 t=0 is the optimal trajectory of problem (SOC2).

Then with probability 1 ??? ??, we have L(U *

Discussions of the effect of ?? on LLC Performance: The result of this lemma shows that when the nominal state and actions are ??-close to the optimal trajectory of (SOC2), i.e., at each time step (z z z t , u u u t ) is a sample from the Gaussian distribution centered at (z * 2,t , u * 2,t ) with standard deviation ??, then one can obtain a performance bound of LLC algorithm that is in terms of the regularization loss R LLC .

To quantify the above condition, one can use Mahalanobis distance (De Maesschalck et al., 2000) to measure the distance of (z z z t , u u u t ) to distribution N ((z * 2,t , u * 2,t ), ?? 2 I), i.e., we want to check for the condition:

for any arbitrary error tolerance > 0.

While we cannot verify the condition without knowing the optimal trajectory {(z * 2,t , u * 2,t )} T ???1 t=0 , the above condition still offers some insights in choosing the parameter ?? based on the trade-off of designing nominal trajectory {(z z z t , u u u t )} T ???1 t=0 and optimizing R LLC .

When ?? is large, the low-curvature regularization imposed by the R LLC regularizer will cover a large portion of the state-action space.

In the extreme case when ?? ??? ???, R LLC can be viewed as a regularizer that enforces global linearity.

Here the trade-off is that the loss R LLC is generally higher, which in turn degrades the performance bound of the LLC control algorithm in Lemma 4.

On the other hand, when ?? is small the low-curvature regularization in R LLC only covers a smaller region of the latent state-action space, and thus the loss associated with this term is generally lower (which provides a tighter performance bound in Lemma 4).

However the performance result will only hold when (z z z t , u u u t ) happens to be close to (z * 2,t , u * 2,t ) at each time-step t ??? {0, . . .

, T ??? 1}.

Proof: For simplicity, we will focus on analyzing the noiseless case when the dynamics is deterministic (i.e., ?? w = 0).

Extending the following analysis for the case of non-deterministic dynamics should be straight-forward.

First, consider any arbitrary latent state-action pair (z, u), such that the corresponding nominal state-action pair (z z z, u u u) is constructed by z z z = z ??? ??z, u u u = u ??? ??u, where (??z, ??u) is sampled from the Gaussian distribution N (0, ?? 2 I). (The random vectors are denoted as (??z , ??u )) By the two-tailed Bernstein's inequality (Murphy, 2012) , for any arbitrarily given ?? ??? (0, 1] one has the following inequality with probability 1 ??? ??:

The second inequality is due to the basic fact that variance is less than second-order moment of a random variable.

On the other hand, at each time step t ??? {0, . . . , T ??? 1} by the Lipschitz property of the immediate cost, the value function V t (z) = min U t:

is also Lipchitz with constant (T ??? t + 1)c lip .

Using the Lipschitz property of V t+1 , for any (z, u) and (??z, ??u), such that (z z z, u u u) = (z ??? ??z, u ??? ??u), one has the following property:

Therefore, at any arbitrary state-action pair (z,??), for z z z = z ??? ??z, and u u u =?? ??? ??u with Gaussian sample (??z, ??u) ??? N (0, ?? 2 I), the following inequality on the value function holds w.p.

1 ??? ??:

which further implies

Now let?? * be the optimal control w.r.t.

Bellman operator T t [V t+1 ](z) at any latent statez.

Based on the assumption of this lemma, at each statez the nominal latent state-action pair (z z z, u u u) is generated by perturbing (z,?? * ) with Gaussian sample (??z, ??u) ??? N (0, ?? 2 I) that is in form of z z z =z ??? ??z, u u u =?? ??? ??u.

Then by the above arguments the following chain of inequalities holds w.p.

1 ??? ??:

Recall the LLC loss function is given by

Also consider the Bellman operator w.r.t.

latent SOC:

, and the Bellman operator w.r.t.

LLC:

.

Utilizing these definitions, the inequality in (21) can be further expressed as

This inequality is due to the fact that all latent states are generated by the encoding observations, i.e., z ??? E(??|x), and thus by following analogous arguments as in the proof of Lemma 1, one has

Therefore, based on the dynamic programming result that bounds the difference of value function w.r.t.

different Bellman operators in finite-horizon problems (for example see Theorem 1.3 in Bertsekas et al. (1995) ), the above inequality implies the following bound in the value function, w.p.

1 ??? ??:

Notice that here we replace ?? in the result in (22) with ??/T .

In order to prove (23), we utilize (22) for each t ??? {0, . . .

, T ??? 1}, and this replacement is the result of applying the Union Probability bound (Murphy, 2012) (to ensure (23) holds with probability 1 ??? ??).

Therefore the proof is completed by combining the above result with that in Lemma 3.

We follow the same control scheme as in Banijamali et al. (2018) .

Namely, we use the iLQR (Li & Todorov, 2004) solver to plan in the latent space.

Given a start observation x start and a goal observation x goal , corresponding to underlying states {s start , s goal }, we encode the observations to retrieve z start and z goal .

Then, the procedure goes as follows: we initialize a random trajectory (sequence of actions), feed it to the iLQR solver and apply the first action from the trajectory the solver outputs.

We observe the next observation returned from the system (closed-loop control), and feed the updated trajectory to the iLQR solver.

This procedure continues until the it reaches the end of the problem horizon.

We use a receding window approach, where at every planning step the solver only optimizes for a fixed length of actions sequence, independent of the problem horizon.

Consider the latent state SOC problem

At each time instance t ??? {0, . . .

, T } the value function of this problem is given by

Recall that the nonlinear latent space dynamics model is given by:

where F ?? (z t , u t ) is the deterministic dynamics model and F ?? F ?? is the covariance of the latent dynamics system noise.

Notice that the deterministic dynamics model F ?? (z t , u t ) is smooth, and therefore the following Jacobian terms are well-posed:

By the Bellman's principle of optimality, at each time instance t ??? {0, . . . , T ??? 1} the value function is a solution of the recursive fixed point equation

where the state-action value function at time-instance t w.r.t.

state-action pair (z t , u t ) = (z, u) is given by

In the setting of the iLQR algorithm, assume we have access to a trajectory of latent states and actions that is in form of {(z z z t , u u u t , z z z t+1 )} T ???1 t=0 .

At each iteration, the iLQR algorithm has the following steps:

1.

Given a nominal trajectory, find an optimal policy w.r.t.

the perturbed latent states 2.

Generate a sequence of optimal perturbed actions that locally improves the cumulative cost of the given trajectory 3.

Apply the above sequence of actions to the environment and update the nominal trajectory 4.

Repeat the above steps with new nominal trajectory Denote by ??z t = z t ??? z z z t and ??u t = u t ??? u u u t the deviations of state and control action at time step t respectively.

Assuming that the nominal next state z z z t+1 is generated by the deterministic transition F ?? (z z z t , u u u t ) at the nominal state and action pair (z z z t , u u u t ), the first-order Taylor series approximation of the latent space transition is given by

To find a locally optimal control action sequence u * t = ?? * ??z,t (??z t ) + u u u t , ???t, that improves the cumulative cost of the trajectory, we compute the locally optimal perturbed policy (policy w.r.t.

perturbed latent state) {?? * ??z,t (??z t )} T ???1 t=0 that minimizes the following second-order Taylor series approximation of Q t around nominal state-action pair (z z z t , u u u t ), ???t ??? {0, . . . , T ??? 1}:

where the first and second order derivatives of the Q???function are given by

and the first and second order derivatives of the value functions are given by

Notice that the Q-function approximation Q t in (28) is quadratic and the matrix

is positive semi-definite.

Therefore the optimal perturbed policy ?? * ??z,t has the following closed-form solution:

where the controller weights are given by

Furthermore, by putting the optimal solution into the Taylor expansion of the Q-function Q t , we get

where the closed-loop first and second order approximations of the Q-function are given by

.

Notice that at time step t the optimal value function also has the following form:

Therefore, the first and second order differential value functions can be V t,z (z z z t , u u u t ) = Q * t,21 (z z z t , u u u t ), V t,zz (z z z t , u u u t ) = Q * t,22 (z z z t , u u u t ), and the value improvement at the nominal state z z z t at time step t is given by

While iLQR provides an effective way of computing a sequence of (locally) optimal actions, it has two limitations.

First, unlike RL in which an optimal Markov policy is computed, this algorithm only finds a sequence of open-loop optimal control actions under a given initial observation.

Second, the iLQR algorithm requires the knowledge of a nominal (latent state and action) trajectory at every iteration, which restricts its application to cases only when real-time interactions with environment are possible.

In order to extend the iLQR paradigm into the closed-loop RL setting, we utilize the concept of model predictive control (MPC) (Rawlings & Mayne, 2009; Borrelli et al., 2017) and propose the following iLQR-MPC procedure.

Initially, given an initial latent state z 0 we generate a single nominal trajectory: {(z z z t , u u u t , z z z t+1 )} We derive the bound for the conditional log-likelihood log P (x t+1 |x t , u t ).

log P (x t+1 |x t , u t ) = log zt,???t+1

Where (a) holds from the log function concavity, (b) holds by the factorization Q(z t ,??? t+1 |x t , x t+1 , u t ) = Q(??? t+1 |x t+1 )Q(z t |??? t+1 , x t , u t ), and (c) holds by a simple decomposition to the different components.

We derive the bound for the consistency loss Consistency ( P ).

Where (a) holds by the assumption that Q(??? t+1 | x t+1 ) = P (z t+1 | x t+1 ), (b) holds from the log function concavity, and (c) holds by a simple decomposition to the different components.

In the following sections we will provide the description of the data collection process, domains, and implementation details used in the experiments.

To generate our training and test sets, each consists of triples (x t , u t , x t+1 ), we: (1) sample an underlying state s t and generate its corresponding observation x t , (2) sample an action u t , and (3) obtain the next state s t+1 according to the state transition dynamics, add it a zero-mean Gaussian noise with variance ?? 2 I ns , and generate it's corresponding observation x t+1 .To ensure that the observation-action data is uniformly distributed (see Section 3), we sample the state-action pair (s t , u t ) uniformly from the state-action space.

To understand the robustness of each model, we consider both deterministic (?? = 0) and stochastic scenarios.

In the stochastic case, we add noise to the system with different values of ?? and evaluate the models' performance under various degree of noise.

Planar System In this task the main goal is to navigate an agent in a surrounded area on a 2D plane (Breivik & Fossen, 2005) , whose goal is to navigate from a corner to the opposite one, while avoiding the six obstacles in this area.

The system is observed through a set of 40 ?? 40 pixel images taken from the top view, which specifies the agent's location in the area.

Actions are two-dimensional and specify the x ??? y direction of the agent's movement, and given these actions the next position of the agent is generated by a deterministic underlying (unobservable) state evolution function.

Start State: one of three corners (excluding bottom-right).

Goal State: bottom-right corner.

Agent's Objective: agent is within Euclidean distance of 2 from the goal state.

Inverted Pendulum -SwingUp & Balance This is the classic problem of controlling an inverted pendulum (Furuta et al., 1991) from 48 ?? 48 pixel images.

The goal of this task is to swing up an under-actuated pendulum from the downward resting position (pendulum hanging down) to the top position and to balance it.

The underlying state s t of the system has two dimensions: angle and angular velocity, which is unobservable.

The control (action) is 1-dimensional, which is the torque applied to the joint of the pendulum.

To keep the Markovian property in the observation (image) space, similar to the setting in E2C and RCE, each observation x t contains two images generated from consecutive time-frames (from current time and previous time).

This is because each image only shows the position of the pendulum and does not contain any information about the velocity.

Start State:

Pole is resting down (SwingUp), or randomly sampled in ????/6 (Balance).

Agent's Objective: pole's angle is within ????/6 from an upright position.

CartPole This is the visual version of the classic task of controlling a cart-pole system (Geva & Sitte, 1993) .

The goal in this task is to balance a pole on a moving cart, while the cart avoids hitting the left and right boundaries.

The control (action) is 1-dimensional, which is the force applied to the cart.

The underlying state of the system s t is 4-dimensional, which indicates the angle and angular velocity of the pole, as well as the position and velocity of the cart.

Similar to the inverted pendulum, in order to maintain the Markovian property the observation x t is a stack of two 80 ?? 80 pixel images generated from consecutive time-frames.

Start State:

Pole is randomly sampled in ????/6.

Agent's Objective: pole's angle is within ????/10 from an upright position.

3-link Manipulator -SwingUp & Balance The goal in this task is to move a 3-link manipulator from the initial position (which is the downward resting position) to a final position (which is the top position) and balance it.

In the 1-link case, this experiment is reduced to inverted pendulum.

In the 2-link case the setup is similar to that of arcobot (Spong, 1995) , except that we have torques applied to all intermediate joints, and in the 3-link case the setup is similar to that of the 3-link planar robot arm domain that was used in the E2C paper, except that the robotic arms are modeled by simple rectangular rods (instead of real images of robot arms), and our task success criterion requires both swing-up (manipulate to final position) and balance.

12 The underlying (unobservable) state s t of the system is 2N -dimensional, which indicates the relative angle and angular velocity at each link, and the actions are N -dimensional, representing the force applied to each joint of the arm.

The state evolution is modeled by the standard Euler-Lagrange equations (Spong, 1995; Lai et al., 2015) .

Similar to the inverted pendulum and cartpole, in order to maintain the Markovian property, the observation state x t is a stack of two 80 ?? 80 pixel images of the N -link manipulator generated from consecutive time-frames.

In the experiments we will evaluate the models based on the case of N = 2 (2-link manipulator) and N = 3 (3-link manipulator).

Start State: 1 st pole with angle ??, 2 nd pole with angle 2??/3, and 3 rd pole with angle ??/3, where angle ?? is a resting position.

Agent's Objective: the sum of all poles' angles is within ????/6 from an upright position.

TORCS Simulaotr This task takes place in the TORCS simulator (Wymann et al., 2000) (specifically in michegan f1 race track, only straight lane).

The goal of this task is to control a car so it would remain in the middle of the lane.

We restricted the task to only consider steering actions (left / right in the range of [???1, 1]), and applied a simple procedure to ensure the velocity of the car is always around 10.

We pre-processed the observations given by the simulator (240 ?? 320 RGB images) to receive 80 ?? 80 binary images (white pixels represent the road).

In order to maintain the Markovian property, the observation state x t is a stack of two 80 ?? 80 images (where the two images are 7 frames apart -chosen so that consecutive observation would be somewhat different).

The task goes as follows: the car is forced to steer strongly left (action=1), or strongly right (action=-1) for the initial 20 steps of the simulation (direction chosen randomly), which causes it to drift away from the center of the lane.

Then, for the remaining horizon of the task, the car needs to recover from the drift, return to the middle of the lane, and stay there.

Start State: 20 steps of drifting from the middle of the lane by steering strongly left, or right (chosen randomly).

Agent's Objective: agent (car) is within Euclidean distance of 1 from the middle of the lane (full width of the lane is about 18).

In the following we describe architectures and hyper-parameters that were used for training the different algorithms.

All the algorithms were trained using:

??? Batch size of 128.

??? ADAM (Goodfellow et al., 2016) with ?? = 5 ?? 10 ???4 , ?? 1 = 0.9, ?? 2 = 0.999, and = 10 ???8 .

??? L 2 regularization with a coefficient of 10 ???3 .

??? Additional VAE (Kingma & Welling, 2013) loss term given by VAE t = ???E q(z|x) [log p(x|z)] + D KL (q(z|x) p(z)), where p(z) ??? N (0, 1).

The term was added with a very small coefficient of 0.01.

We found this term to be important to stabilize the training process, as there is no explicit term that governs the scale of the latent space.

??? ?? from the loss term of E2C was tuned using a parameter sweep in {0.25, 0.5, 1}, and was chosen to be 0.25 across all domains, as it performed the best independently for each domain.

PCC training specifics:

??? ?? p was set to 1 across all domains.

??? ?? c was set to be 7 across all domains, after it was tuned using a parameter sweep in {1, 3, 7, 10} on the Planar system.

??? ?? cur was set to be 1 across all domains without performing any tuning.

??? {z,??}, for the curvature loss, were generated from {z, u} by adding Gaussian noise N (0, 0.1 2 ), where ?? = 0.1 was set across all domains without performing any tuning.

??? Motivated by Hafner et al. (2018) , we added a deterministic loss term in the form of cross entropy between the output of the generative path given the current observation and action (while taking the means of the encoder output and the dynamics model output) and the observation of the next state.

This loss term was added with a coefficient of 0.3 across all domains after it was tuned using a parameter sweep over {0.1, 0.3, 0.5} on the Planar system.

??? E ADDITIONAL RESULTS E.1 PERFORMANCE ON NOISY DYNAMICS Table 3 shows results for the noisy cases.

1.2 ?? 0.6 0.6 ?? 0.3 17.9 ?? 3.1 5.5 ?? 1.2 6.1 ?? 0.9 44.7 ?? 3.6 Planar2 0.4 ?? 0.2 1.5 ?? 0.9 14.5 ?? 2.3 1.7 ?? 0.5 15.5 ?? 2.6 29.7 ?? 2.9 Pendulum1 6.4 ?? 0.3 23.8 ?? 1.2 16.4 ?? 0.8 8.1 ?? 0.4 36.1 ?? 0.3 29.5 ?? 0.2 Cartpole1 8.1 ?? 0.6 6.6 ?? 0.4 9.8 ?? 0.7 20.3 ?? 11 16.5 ?? 0.4 17.9 ?? 0.8 3-link1 0.3 ?? 0.1 0 ?? 0 0.5 ?? 0.1 1.3 ?? 0.2 0 ?? 0 1.8 ?? 0.3

Under review as a conference paper at ICLR 2020

The following figures depicts 5 instances (randomly chosen from the 10 trained models) of the learned latent space representations for both the noiseless and the noisy planar system from PCC, RCE, and E2C models.

<|TLDR|>

@highlight

Learning embedding for control with high-dimensional observations