We consider the problem of generating configurations that satisfy physical constraints for optimal material nano-pattern design, where multiple (and often conflicting) properties need to be simultaneously satisfied.

Consider, for example, the trade-off between thermal resistance, electrical conductivity, and mechanical stability needed to design a nano-porous template with optimal thermoelectric efficiency.

To that end, we leverage the posterior regularization framework andshow that this constraint satisfaction problem can be formulated as sampling froma Gibbs distribution.

The main challenges come from the black-box nature ofthose physical constraints, since they are obtained via solving highly non-linearPDEs.

To overcome those difficulties, we introduce Surrogate-based Constrained Langevin dynamics for black-box sampling.

We explore two surrogate approaches.

The first approach exploits zero-order approximation of gradients in the Langevin Sampling and we refer to it as Zero-Order Langevin.

In practice, this approach can be prohibitive since we still need to often query the expensive PDE solvers.

The second approach approximates the gradients in the Langevin dynamics with deep neural networks, allowing us an efficient sampling strategy using the surrogate model.

We prove the convergence of those two approaches when the target distribution is log-concave and smooth.

We show the effectiveness of both approaches in designing optimal nano-porous material configurations, where the goal is to produce nano-pattern templates with low thermal conductivity and reasonable mechanical stability.

In many real-world design problems, the optimal design needs to simultaneously satisfy multiple constraints, which can be expensive to estimate.

For example, in computational material design, the goal is to come up with material configurations, or samples, satisfying a list of physical constraints that are given by black-box numerical Partial Differential Equations (PDE) solvers.

Such solvers (for example, the Boltzmann Transport Equation solver) are often complex, expensive to evaluate, and offer no access to their inner variables or their gradients.

We pose this design-under-constraints problem as sampling from a Gibbs distribution defined on some compact support.

The problem of sampling from a distribution with unknown likelihood that can only be point-wise evaluated is called black-box sampling (Chen & Schmeiser, 1998; Neal, 2003) .

We show in this paper that constrained black-box sampling can be cast as a constrained Langevin dynamics with gradient-free methods.

Zero-order optimization via Gaussian smoothing was introduced in Nesterov & Spokoiny (2017) and extended to black-box sampling with Langevin dynamics in Shen et al. (2019) .

We extend this approach to the constrained setting from a black-box density with compact support.

However, one shortcoming of this approach is that it is computationally very expensive since it requires repeatedly querying PDE solvers in order to get an estimate of the gradient.

To alleviate computational issues, we propose Surrogate Model Based Langevin dynamics, that consists of two steps: (i) Learning (using training data) an approximation of the gradient of the potential of the Gibbs distribution.

We show that learning the gradient, rather than the potential itself, is important for the mixing of the Langevin dynamics towards the target Gibbs distribution.

We devise several objective functions, as well as deep neural-network architectures for parameterizing the approximating function class, for learning the gradient of the potential function. (ii) We then use the surrogate gradient model in the constrained Langevin dynamics in lieu of the black-box potential.

Using the surrogate enables more efficient sampling, since it avoids querying the expensive PDE solvers, and obtaining gradients is as efficient as evaluating the functions themselves using automatic differentiation frameworks such as PyTorch or TensorFlow.

To summarize, our main contributions are as follows:

1.

We cast the problem of generating samples under constraints in the black-box setting as sampling from a Gibbs distribution.

2.

We introduce Constrained Zero-Order Langevin Monte Carlo, using projection or proximal methods, and provide the proof of its convergence to the target Gibbs distribution.

3.

We introduce Surrogate Model Based Projected Langevin Monte Carlo via learning the gradient of the potential of the Gibbs distribution using deep neural networks or reproducing kernel spaces, and prove its convergence to the target distribution when used in conjunction with projection or proximal based methods.

We shed the light on the importance of the approximation of the gradient of the potential, and we show how to achieve this using Hermite and Taylor learning.

4.

We showcase the usability and effectiveness of the proposed methods for the design of nanoporous configurations with improved thermoelectric efficiency.

The design consists of finding new configurations with optimized pore locations, such that the resulting configurations have favorable thermal conductivity (i.e., minimal κ) and desired mechanical stability (von Mises Stress σ ≤ τ , where τ is some preset threshold).

In black-box optimization problems (such as the material design under consideration), the goal is to find a posterior distribution q of samples satisfying a list of equality and inequality constraints: ψ j (x) = y k , j = 1 . . .

C e , and φ k (x) ≤ b k , k = 1 . . .

C i where x ∈ Ω and Ω ⊂ R d is a bounded domain.

We assume a prior distribution p 0 (whose analytical form is known).

The main challenge in black-box optimization is that the functions ψ j and φ k can be only evaluated point-wise, and neither do we have functional forms nor access to their gradients.

For example, ψ and φ might be obtained via aggregating some statistics on the solution of a nonlinear PDE given by a complex solver.

To make the problem of learning under constraints tractable, we choose Lagrangian parameters λ j > 0 and obtain the following relaxed objective:

The formulation in Eq. 1 is similar in spirit to the posterior regularization framework of Ganchev et al. (2010) ; Hu et al. (2018) .

However, we highlight two differences: (i) our focus is on constrained settings (where Ω is bounded), and (ii) we assume a black-box setting.

We first obtain: Lemma 1 (Constraint Satisfaction as Sampling from a Gibbs Distribution).

The solution to the distribution learning problem given in Eq. 1 is given by:

where

Lemma 1 shows that the constraint satisfaction problem formulated in Eq. 1 amounts to sampling from a Gibbs distribution defined on a compact support given in Eq. 2.

Sampling from a Gibbs 1 Note that both properties κ and σ for a given configuration are obtained by numerically solving highly non-linear PDEs.

The material configuration is defined by the pore locations, the material used, and the response of the material to heat (thermal) or stress (mechanical) flows.

distribution (also known as Boltzmann distribution) has a long history using Langevin dynamics.

In the white-box setting when the functions defining the constraints have explicit analytical forms as well as their gradients, Langevin dynamics for Gibbs distribution sampling defined on a compact domain Ω and their mixing properties were actively studied in Bubeck et al. (2015) ; Brosse et al. (2017) .

In the next Section, we provide a more detailed review.

Remark 1 (Relation to Bayesian Optimization).

While in Bayesian optimization we are interested in finding a point that satisfies the constraints, in our setting we are interested in finding a distribution of candidate samples that satisfy (black-box) constraints.

See (Suzuki et al., 2019) for more details.

Remark 2.

For the rest of the paper, we will assume p 0 to be the uniform distribution on Ω, which means that its gradients are zero on the support of the domain Ω. Otherwise, if p 0 is known and belongs to, for instance, an exponential family or a generative model prior (such as normalizing flows), we can sample from π using a mixture of black-box sampling on the constraints (ψ j , φ k ) and white-box sampling on log(p 0 ).

We review in this section Langevin dynamics in the unconstrained case (Ω = R d ) and the constrained setting (Ω ⊂ R d ).

Below, · denotes the Euclidean norm unless otherwise specified.

We are interested in sampling from

Preliminaries.

We give here assumptions, definitions and few preliminary known facts that will be useful later.

Those assumptions are commonly used in Langevin sampling analysis (Dalalyan, 2017; Bubeck et al., 2015; Brosse et al., 2017; Durmus et al., 2019) .

We assume Ω is a convex such that 0 ∈ Ω, Ω contains a Euclidean ball of radius r, and Ω is contained in a Euclidean ball of radius R. (For example, Ω might encode box constraints.)

The projection onto Ω, P Ω (x) is defined as follows: for all x ∈ Ω, P Ω (x) = arg min z∈Ω x − z 2 .

Let R = sup x,x ∈Ω ||x − x || < ∞.

We assume that U is convex, β-smooth, and with bounded gradients:

The Total Variation (TV) distance between two measures µ, ν is defined as follows:

Unconstrained Langevin Dynamics.

In the unconstrained case, the goal is to sample from a Gibbs distribution π(x) = exp(−U (x))/Z that has unbounded support.

This sampling can be done via the Langevin Monte Carlo (LMC) algorithm, which is given by the following iteration:

where ξ k ∼ N (0, I d ), η is the learning rate, and λ > 0 is a variance term.

Constrained Langevin Dynamics.

In the constrained case, the goal is to sample from π(x) = exp(−U (x))/Z1 x∈Ω ,.

We discuss two variants:

Projected Langevin Dynamics.

Similar to projected gradient descent, Bubeck et al. (2015) introduced Projected Langevin Monte Carlo (PLMC) and proved its mixing propreties towards the stationary distribution π.

PLMC is given by the following iteration :

In essence, PLMC consists of a single iteration of LMC, followed by a projection on the set Ω using the operator P Ω .

Proximal Langevin Dynamics.

Similar to proximal methods in constrained optimization, Brosse et al. (2017) introduced Proximal LMC (ProxLMC) that uses the iteration:

where η is the step size and γ is a regularization parameter.

In essence, ProxLMC (Brosse et al., 2017) performs an ordinary LMC on

where i Ω (x) = 0 for x ∈ Ω and i Ω (x) = ∞ for x / ∈ Ω. Therefore, the update in Eq. 6 is a regular Langevin update (as in Eq. 4) with potential gradient

We denote by µ PLMC K and µ

ProxLMC K the distributions of X K obtained by iterating Eq. 5 and Eq. 6 respectively.

Under Assumptions A and B, both these distributions converge to the target Gibbs distribution π in the total variation distance.

In particular, Bubeck et al. (2015) showed that for η =Θ(R 2 /K), we obtain:

Likewise, Brosse et al. (2017) showed that for 0 < η ≤ γ(1 + β 2 γ 2 ) −1 , we obtain:

where the notation α n =Ω(β n ) means that there exists c ∈ R, C > 0 such that α n ≥ Cβ n log c (β n ).

We now introduce our variants of constrained LMC for the black-box setting where explicit potential gradients are unavailable.

We explore in this paper two strategies for approximating the gradient of U in the black-box setting.

In the first strategy, we borrow ideas from derivative-free optimization (in particular, evolutionary search).

In the second strategy we learn a surrogate deep model that approximates the gradient of the potential.

Below, let G : Ω → R d be a vector valued function that approximates the gradient of the potential, ∇ x U .

We make:

Surrogate Projected Langevin Dynamics.

Given Y 0 , the Surrogate Projected LMC (S-PLMC) replaces the potential gradient ∇ x U in Eq. 5 with the surrogate gradient G:

Surrogate Proximal Langevin Dynamics.

Similarly, the Surrogate Proximal LMC (S-ProxLMC) replaces the unknown potential gradient ∇ x U in Eq. 6 with the gradient surrogate G:

We now present our main theorems on the approximation properties of surrogate LMC (S-PLMC, and S-ProxLMC).

We do so by bounding the total variation distance between the trajectories of the surrogate Langevin dynamics (S-PLMC, and S-ProxLMC) and the true LMC dynamics (PLMC and ProxLMC).

Theorem 1 is an application of techniques in Stochastic Differential Equations (SDE) introduced in Dalalyan & Tsybakov (2012) and is mainly based on a variant of Grisanov's Theorem for change of measures (Lipster & Shiryaev, 2001 ) and Pinsker's Inequality that bounds total variation in terms of Kullback-Leibler divergence.

Theorem 1 (S-PLMC and S-ProxLMC Mixing Properties).

Under Assumption C, we have:

be the distribution of the random variable X K obtained by iterating PLMC Eq. 5, and µ S-PLMC K be the distribution of the random variable Y K obtained by iteration S-PLMC given in Eq. 9.

We have:

2. S-ProxLMC Convergence.

Let µ ProxLMC K be the distribution of the random variable X K obtained by iterating ProxLMC Eq. 6, and µ S-ProxLMC K be the distribution of the random variable Y K obtained by iterating S-ProxLMC given in Eq. 10.

We have:

From Theorem 1, we see that it suffices to approximate the potential gradient ∇ x U (X) (and not the potential U (X)) in order to guarantee convergence of surrogate-based Langevin sampling.

Using the triangle inequality, and combining Theorem 1 and bounds in Eqs 7 and 8 we obtain: Theorem 2.

(Convergence of Surrogate Constrained LMC to the Gibbs distribution.)

Under assumptions A,B and C we have:

we have:

In zero-order optimization (Nesterov & Spokoiny, 2017; Duchi et al., 2015; Ghadimi & Lan, 2013; Shen et al., 2019) , one considers the Gaussian smoothed potential

, and its gradient is given by

where g 1 , . . .

g n are i.i.d.

standard normal vectors.

Zero-Order sampling from log-concave densities was recently studied in Shen et al. (2019) .

We extend it here to the constrained sampling case of log-concave densities with compact support.

We define Constrained Zero-Order Projected LMC (Z-PLMC) and Zero-Order Proximal LMC (Z-ProxLMC) by setting G(x) =Ĝ n U(x) in Eq. 9 and Eq. 10 respectively.

Lemma 2 (Zero-Order Gradient Approximation (Nesterov & Spokoiny, 2017; Shen et al., 2019) ).

Under Assumption B, we have for all x ∈ Ω:

Thanks to Lemma 2 that ensures uniform approximation of gradients in expectation, we can apply Theorem 2 and get the following corollary for Z-PLMC and Z-ProxLMC: Corollary 1 (Zero-order Constrained Langevin approximates the Gibbs distribution).

Under As-

we have the following bounds in expectation:

2.

Set λ = 1, and

we have:

Remark 3.

For simplicity, we state the above bound in terms of expectations over the randomness in estimating the gradients.

It is possible to get finite-sample bounds using the Vector Bernstein concentration inequality, coupled with covering number estimates of Ω but omit them due to space.

Despite its theoretical guarantees, zero-order constrained Langevin (Z-PLMC and Z-ProxLMC) has a prohibitive computation cost as it needs O(nK) black-box queries (in our case, invocations of a nonlinear PDE solver).

To alleviate this issue, we introduce in this Section a neural surrogate model as an alternative to the gradient of the true potential.

From Theorem 2, we saw that in order to guarantee the convergence of constrained Langevin dynamics, we need a good estimate of the gradient of the potential of the Gibbs distribution.

Recall that the potential given in Lemma 1 depends on ψ j and φ k , which are scalar outputs of computationally heavy PDE solvers in our material design problem.

To avoid this, we propose to train surrogate neural network models approximating each PDE output and their gradients.

Concretely, suppose we are given a training set S for a PDE solver for the property ψ (dropping the index j for simplicity):

where ρ Ω is the training distribution andĜ n ψ(.) is the zero-order estimate of the gradient of ψ given in Eq. 13.

We propose to learn a surrogate model belonging to a function class H θ ,f θ ∈ H θ , that regresses the value of ψ and matches the zero-order gradient estimates as follows:

The problem in Eq. 17 was introduced and analyzed in Shi et al. (2010) where H θ is a ball in a Reproducing Kernel Hilbert Space (RKHS).

Following Shi et al. (2010) , we refer to this type of learning as Hermite Learning.

In the deep learning community, this type of learning is called Jacobian matching and was introduced in Srinivas & Fleuret (2018) ; Czarnecki et al. (2017) where H θ is a deep neural network parameterized with weights θ.

When f θ is a deep network, we can optimize this objective efficiently using common deep learning frameworks (PyTorch, TensorFlow). (Shi et al., 2010) have shown that when H θ is an RKHS ball and whenỹ i = ∇ x ψ(x i ) are exact gradients, for a sufficiently large training set with N = O(1/ 1/(2rζ) ) (where r, ζ are exponents in [0, 1] that depend on the regularity of the function ψ).

Under the assumption that ψ ∈ H θ we have:

Since we are using inexact zero-order gradients, we will incur an additional numerical error that is also bounded as shown in Lemma 2.

While Jacobian matching of zero-order gradients is a sound approach, it remains expensive to construct the dataset, as we need for each point to have 2n + 1 queries of the PDE solver.

We exploit in this section the Taylor learning framework of gradients that was introduced in Mukherjee & Zhou (2006) ; Wu (2006), and Wu et al. (2010) .

In a nutshell, Mukherjee & Zhou (2006) suggests to learn a surrogate potential f θ and gradient G Λ that are consistent with the first-order taylor expansion.

Given a training set Wu et al. (2010) suggest the following objective:

where

, H θ is an RKHS ball of scalar valued functions, and H d Λ is an RKHS ball of vector valued functions.

Under mild assumptions, Mukherjee & Zhou (2006) shows that we have for

We simplify the problem in Eq. 18 and propose the following two objective functions and leverage the deep learning toolkit to parameterize the surrogate f θ :

The objective in Eq. 19 uses a single surrogate to parameterize the potential and its gradient.

The objective in Eq. 20 is similar in spirit to the Jacobian matching formulation in the sense that it adds a regularizer on the gradient of the surrogate to be consistent with the first-order Taylor expansion in local neighborhoods.

The advantage of the Taylor learning approach is that we do not need to perform zero-order estimation of gradients to construct the training set and we rely instead on first-order approximation in local neighborhood.

Consider the surrogate model f θ obtained via Hermite Learning (Eq. 17) or via Taylor learning (Eqs 18, 19, 20) .

We are now ready to define the surrogate model LMC by replacing

in the constrained Langevin dynamics in Eqs 9 and 10.

Both Hermite and Taylor learning come with theoretical guarantees when the approximation function space is an RKHS under some mild assumptions on the training distribution and the regularity of the target function ψ.

In Hermite learning (Theorem 2 in Shi et al. (2010)) we have:

) (where exponents ζ, r ∈ [0, 1] depend on regularity of ψ).

In Taylor Learning with the objective function given in Eq. 18 (Proposition 7 in Wu et al. (2010) we have:

.

In order to apply Theorem 2 we need this gradient approximation error to hold in expectation on all intermediate distributions in the Langevin sampling.

Hence, we need the following extra-assumption on the training distribution p Ω :

Assumption D:

Assume we have a learned surrogate G on training distribution ρ Ω such that

2 ≤ .

Assume ρ Ω (x) > 0, ∀x ∈ Ω and that it is a dominating measure of Langevin (PLMC, S-PLMC, Prox-LMC, S-ProxLMC ) intermediate distributions µ k , i.e. there exists C > 0 such that:

and hence we can apply Theorem 2 for δ = C , and we obtain ε-approximation of the target Gibbs distribution in terms of total variation distance.

Remark 4.

Assumption D on the -approximation of the gradient can be achieved for a large enough training set N , when we use Hermite learning in RKHS under mild assumptions and in Taylor learning.

The assumption on the dominance of the training distribution is natural and means that we need a large training set that accounts to what we may encounter in Surrogate LMC iterations.

In what follows we refer to surrogate constrained LMC, as x-PLMC or x-ProxLMC where x is one of four suffixes ({Z-Hermite, Taylor-2, Taylor-1, Taylor-Reg}).

Zero-Order Methods.

Zero-order optimization with Gaussian smoothing was studied in Nesterov & Spokoiny (2017) and Duchi et al. (2015) in the convex setting.

Non-convex zero order optimization was also addressed in Ghadimi & Lan (2013) .

The closest to our work is the zero-order Langevin Shen et al. (2019) introduced recently for black-box sampling from log concave density.

The main difference in our setting, is that the density has a compact support and hence the need to appeal to projected LMC (Bubeck et al., 2015) and Proximal LMC (Brosse et al., 2017) .

It is worth nothing that Hsieh et al. (2018) introduced recently mirror Langevin sampling that can also be leveraged in our framework.

Gradients and Score functions Estimators.

We used the approach of gradient distillation (Srinivas & Fleuret, 2018) and learning gradients of (Wu et al., 2010) , since they are convenient for training on different constraints and they come with theoretical guarantees.

However, other approaches can be also leveraged such as the score matching approach for learning the gradient of the log likelihood (Hyvärinen, 2005) and other variants appealing to dual embeddings (Dai et al., 2018) .

Estimating gradients can be also performed using Stein's method as in (Li & Turner, 2017) , or via maintaining a surrogate of the gradient as in Stein descent without gradient (Han & Liu, 2018) .

Optimization approaches.

Due to space limitation, we restrict the discussion to the optimization methods that are most commonly and recently used for optimal material (or molecule) design.

A popular approach to deal with optimization of expensive black-box functions is Bayesian Optimization (BO) (Mockus, 1994; Jones et al., 1998; Frazier, 2018) .

The standard BO protocol is comprised of estimating the black-box function from data through a probabilistic surrogate model, usually a Gaussian process, and maximizing an acquisition function to decide where to sample next.

BO is often performed over a latent space, as in (Gómez-Bombarelli et al., 2018) .

Hernández-Lobato et al. (2016) proposed an information-theoretic framework for extending BO to address optimization under black-box constraints, which is close to current problem scenario.

Genetic Algorithms (GA), a class of meta-heuristic based evolutionary optimization techniques, is another widely used approach for generating (material) samples with desired property (Jennings et al., 2019) and has been also used for handling optimization under constraints (Chehouri et al., 2016) .

However, GA typically requires a large number of function evaluations, can get stuck in local optima, and does not scale well with complexity.

Finally, Zhou et al. (2019) has used deep reinforcement learning technique of Deep Q-networks to optimize molecules under a specific constraint using desired properties as rewards.

The advantage of our framework is that we obtain a distribution of optimal configurations (as opposed to a single optimized sample) that does not rely on training on a specific pre-existing dataset and can be further screened and tested for their optimality for the task at hand.

In this section, we demonstrate the usability of our black-blox Langevin sampling approach for the design of nano-porous configurations.

We first show the performance of the surrogate models in learning the potential function, showcasing the results using four different variants: standard regression, Taylor regularization, Taylor-1 and Taylor-2.

We then show how well the surrogate-based Langevin MC generates new samples under the thermal and mechanical constraints.

We compare the sample quality on multiple criteria between the surrogate and zero-order approaches with either projection or proximal update step.

Data.

We want to learn surrogate models to approximate the gradient of the potential from data.

To this end, we generate a dataset of 50K nano-porous structures, each of size 100nm × 100nm.

One such example is displayed in Fig. 1 .

Number of pores is fixed to 10 in this study and each pore is a square with a side length of 17.32nm.

We sample the pore centers uniformly over the unit square and construct the corresponding structure after re-scaling them appropriately.

Then, using the solvers OpenBTE (Romano & Grossman, 2015) and Summit ( MIT Development Group, 2018), we obtain for each structure x a pair of values: thermal conductivity κ and von Mises stress σ.

Finally, we collect two datasets:

with the same inputs x i 's and N = 50K samples.

More details are given in Appendices B and C on the PDEs and their corresponding solvers.

Features.

The pore locations are the natural input features to the surrogate models.

Apart from the coordinates, we also derive some other features based on physical intuitions.

For example, the distances between pores and the alignment along axes are informative of thermal conductivity (Romano & Grossman, 2016) .

As such, we compute pore-pore distances along each coordinate axis and add them as additional features.

Surrogate gradient methods.

We use feed-forward neural networks to model the surrogates since obtaining gradients for such networks is efficient thanks to automatic differentiation frameworks.

We use networks comprised of 4 hidden layers with sizes 128, 72, 64, 32 and apply the same architecture to approximate the gradients for κ and σ separately.

The hidden layers use ReLU activations whereas sigmoid was used at the output layer (after the target output is properly normalized).

For the Taylor-2 variant (in Eq. 18), we have an additional output vector of the same size as the input for the gradient prediction.

The networks are trained on the corresponding objective functions set up earlier by an Adam optimizer with learning rate 10 −4 and decay 1.0.

We fine-tune the networks with simple grid-search and select the best models for comparison.

Due to the space constraint, we present the results in Appendix A and emphasize that Z-Hermite is not included in the entire comparison but in a small experiment performed with a more lightweight OpenBTE version.

Incorporating constraints and comparison metrics.

We demonstrate the usability of our proposed black-box Langevin sampling for the design of nano-configurations under thermal conductivity and mechanical stability constraints that are provided by the corresponding PDE solvers.

To compare sampling outcomes, we use the following metrics.

We report the minimum value of κ and Monte Carlo estimates for both κ and σ to compare the samples generated by different sampling methods and surrogate models.

The Monte Carlo estimates are computed on 20 samples.

Single constraint.

Our first task is to design nano-configurations under the thermal conductivity constraint where we want κ as low as possible in order to achieve high thermo-electric efficiency.

From the posterior regularization formulation Section 2, we pose the constraint satisfaction as sampling from the following Gibbs distribution:

where p 0 (x) is the uniform distribution over the unit square, which is equivalent to the Poisson process of 10 pores on the square, and κ(x) is the thermal conductivity we want to minimize.

Starting from 20 samples initialized from p 0 (x), we run our proposed black-box Langevin MCs and obtain 20 new realizations from the target distribution π(x).

We use four different surrogates (including simple regression, Taylor-Reg, Taylor-1 and zero-order) and each surrogate with either projection or proximal update.

We show the summary statistics of these samples in Table 1 .

The regression-PMLC in the first row and regression-ProxLMC in the fifth represent the sampling where the surrogate model are fitted on solely the mean square error objective.

In all methods, we set λ = 100, the step size η = 1e−3 and the exponential decay rate 0.8.

Since keeping track of the true κ value is expensive, we stop after K = 10 iterations.

We first observe that the regression-based method (PLMC, ProxLMC) is less effective than the others simply because they do not have an implicit objective for approximating the gradients.

Taylor-Reg and Taylor-1 demonstrate its effectiveness in approximating the gradient and are able to achieve lower thermal conductivity.

In particular, Taylor-1-ProxLMC and Zero-order-PLMC perform in the similar range in terms of the minimum achieved, but the learned surrogate offers 17x speed up (per sample) over zero order methods.

Due to the space limit, we do not report Taylor-2 results in Table 1 , and note that Taylor-2 works in the similar vein as Taylor-1.

Multiple constraints.

Achieving the minimal thermal conductivity can be fulfilled without much difficulty (e.g. structures with all pores aligned along the vertical axis), but such structures are often mechanically unstable.

In the next step, we study whether adding more (conflicting) constraints helps us design better nano-configurations.

Hence, we consider both thermal conductivity κ and mechanical stability provided via von Mises stress σ.

We want a sample x that minimizes κ(x) to achieve high thermo-electric efficiency while maintaining σ(x) less than some threshold (which we explain below).

Like the single constraint case, we pose this as sampling from the following Gibbs distribution:

where p 0 (x) is the same as above, σ(x) is the von Mises stress and τ is a threshold on the maximum value of σ.

With this framework, we relax the inequality constraint to the Hinge loss term on von Mises stress.

The results are summarized in Table 2 .

Note that all the surrogate Langevin MCs are initialized from the same set of 20 samples as above.

In this experiment, we set τ = 0.5, λ 1 = 100, λ 2 = 10 the step size η = 1e−3 and the exponential decay rate 0.8.

Comparing with Table 1 , one can see that not only better κ be achieved but also the σ can be reduced simultaneously.

These results suggest that our approach can effectively sample new configurations under multiple competing constraints.

Examples of new nano-configurations are show in Fig. 1 Table 2 : Summary statistics of 20 new samples obtained by our sampling method on π(x) with κ and σ constraints Eq. 22.

The starting samples are reused from the single constraint case (min κ = 0.0759, mean κ = 0.1268, and mean σ = 0.8181; note that σ can be as high as 16.)

In this paper we introduced Surrogate-Based Constrained Langevin Sampling for black-box sampling from a Gibbs distribution defined on a compact support.

We studied two approaches for defining the surrogate: the first through zero-order methods and the second via learning gradient approximations using deep neural networks.

We showed the proofs of convergence of the two approaches in the log-concave and smooth case.

While zero-order Langevin had prohibitive computational cost, learned surrogate model Langevin enjoy a good tradeoff of lightweight computation and approximation power.

We applied our black-box sampling scheme to the problem of nano-material configuration design, where the black box constraints are given by expensive PDE solvers, and showed the efficiency and the promise of our method in finding optimal configurations.

Among different approaches for approximating the gradient, the zero-order ones (PLMC, ProxLMC) show overall superior performance, at a prohibitive computational cost.

We established that the deep the surrogate (Taylor-1 ProxLMC) is a viable alternative to zero-order methods, achieving reasonable performance, and offering 15x speedup over zero-order methods.

Surrogate gradient methods We use feed-forward neural networks to model the surrogates since obtaining gradients for such networks is efficient thanks to automatic differentiation frameworks.

We use networks comprised of 4 hidden layers with sizes 128, 72, 64, 32 and apply the same architecture to approximate the gradients for κ and σ separately.

The hidden layers compute ReLU activation whereas sigmoid was used at the output layer (after the target output is properly normalized).

For the Taylor-2 variant (in Eq. 18), we have an output vector for the gradient prediction.

The networks are trained on the corresponding objective functions set up earlier by Adam optimizer with learning rate 10 −4 and decay 1.0.

We fine-tune the networks with simple grid-search and select the best models for comparison.

As emphasized throughout, our focus is more on approximating the gradient rather than learning the true function.

However, we need to somehow evaluate the surrogate models on how well they generalize on a hold-out test set.

Like canonical regression problems, we compare the surrogate variants against each other using root mean square error (RMSE) on the test set.

Figures 2 and 3 shows the results.

The left figure shows RMSE for predicting κ and the right one shows RMSE for the von Mises stress σ.

We can see that the Taylor-Reg generalizes better and also converges faster than Taylor-1 and Taylor-2 to target RMSE for κ, while all methods result similarly for σ prediction.

This is reasonable because the objectives of Taylor-1 and Taylor-2 are not to optimize the mean square error, which we evaluate on here.

Figure 3 shows the learning in terms of sample complexity.

Again, Taylor-Reg outperforms Taylor-1 and Taylor-2 for κ prediction.

In contrast, most models work similarly for σ regression, particularly when the training size is reduced to 50% (25K).

Effectiveness of Z-Hermite learning Notice that Z-Hermite learning is not included in this comparison and as a surrogate model in the black-blox Langevin sammpling in Section 8.

The reason is that apart from the usual sample pair (x i , y i ), we need the gradientỹ i (See Eq. 17).

Since we can query the solvers, this gradient can only be estimated using finite difference.

For both κ and σ in our experiment, obtaining such data is extremely expensive.

As a consequence, we do not have the full results of the Z-Hermite model.

Instead, we ran a separate study to show the effectiveness of Z-Hermite surrogate LMC on a smaller data with a lightweight OpenBTE version (0.9.55).

The results in Table 3 shows the working of Z-Hermite learning in learning the gradient of κ(x).

Here, the entropy is based nearest neighbor estimate to demonstrate the diversity of the pore centers in the unit square.

With the (x p , y p )-coordinates of each pore p, the entropy estimate is given by:

Mean A hybrid algorithm between zero-order and Taylor-1 surrogate We can see in Tables 1, 2 and 3 the trade-off between computation and accuracy of our approach.

While zero-order PLMC and ProxLMC can achieve the lowest thermal conductivity, their computational costs are prohibitive.

In contrast, deep surrogate models (including Taylor-Reg, Taylor-1) are far more time-efficient but slightly worse in terms of achieving the optimal κ.

To mitigate the trade-off, we propose a simple hybrid method that combines the best of the zero-order and Taylor-1 surrogate models.

The algorithm is shown in Figure A that alternates between using the gradient from the zero-order estimate and the gradient of the deep surrogate depending on whether taking this step would decrease the potential function (i.e. κ).

We show and compare the achieved κ and running time in Table 3 .

Examples of the samples generated by Zero-order PLMC, Taylor-1 PLMC and the hybrid method are also depicted in Figure 4 .

The hybrid achieves the thermal conductivity that is lower than Taylor-1 PMLC while running almost 2x faster than zero-order PLMC.

This suggests that the hybrid strategy offers a better trade-off in accuracy and computation.

One way to further improve the hybrid is to collect the zero-order gradients while mixing and re-update the surrogate with Z-Hermite learning.

Algorithm 1 A hybrid PLMC algorithm alternating between zero-order and Taylor-1 surrogate gradients.

Train a network f θ (x) with Taylor-1 Randomly sample x 0 from the uniform p(x) Perform a Langevin dynamic step Additional generated samples We show additional configurations generated by our sampling approach (Taylor-Reg ProxLMC, Taylor

At the nanoscale, heat transport may exhibit strong ballistic behaviour and a non-diffusive model must be used (Chen, 2005) .

In this work we use the Boltzmann transport equation under the relaxation time approximation and in the mean-free-path (MFP) formulation (Romano & Grossman, 2015) Λŝ

where T (Λ) is the effective temperature associated to phonons with MFP Λ and directionŝ; the notation .

stands for an angular average.

The coefficients α(Λ ) are given by

where K(Λ ) is the bulk MFP distribution.

In general, such a quantity can span several orders of magnitude; however, for simplicity we assume the gray model, i.e. all phonons travel with the same MFP, Λ 0 .

Within this approximation, we have K(Λ) = κ bulk δ(Λ − Λ 0 ).

In this work we choose Λ 0 = 10 nm, namely as large as the unit cell, so that significant phonons size effects occur.

With no loss of generality, we set κ bulk = 1 Wm −1 K −1 .

Eq. 23 is an integro-differential PDE, which is solved iteratively for each phonon direction over an unstructured mesh (Romano & Di Carlo, 2011) .

We apply periodic boundary conditions along the unit cell while imposing a difference of temperature of ∆T = 1 K along the x-axis.

At the pores' walls we apply diffusive boundary conditions.

Upon convergence, the effective thermal conductivity is computed using Fourier's law, i.e.

where J = (κ bulk /Λ 0 ) T (Λ 0 )ŝ n is the heat flux, L is the size of the unit cell, A is the area of the cold contact (with normaln).

Throughout the text we use the quantity κ = κ eff /κ bulk as a measure of phonon size effects.

We model mechanical stress by using the continuum linear elasticity equations

where f i is the body force (which is zero in this case), and σ ij is the stress tensor.

Note that we used the Einstein notation, i.e. repeated indexes are summed over.

The strain kl is related to the stress via the fourth-rank tensor elastic constant C ijkl σ ij = C ijkl kl .

The strain is then related to the displacement u via kl = 1 2

We apply periodic boundary conditions along the unit-cell and applied solicitation is a small in-plane expansion.

Once the stress tensor is calculated, we compute the von Mises stress as

where σ i are the principal stress axis.

As a mechanical stability estimator we use σ = max x∈D (σ V M ) where D is the simulation domain.

To avoid material's plasticity, σ needs to be smaller than the yield stress of a given material.

For mechanical simulation we used the SUMIT code ( MIT Development Group, 2018 Setting first order optimality conditions on q, we have for x ∈ Ω:

Hence we have:

, x ∈ Ω and q(x) = 0, x / ∈ Ω, First order optimality on η give us: Ω q(x) = 1, we conclude by setting e exp(−η) = Z.

Proof of Theorem 1 1) Projected Langevin.

Let us define the following continuous processes by interpolation of X k and Y K (Piecewise constant):

whereŨ t (X) = − ∞ k=0 ∇ x U (X kη )1 t∈[kη,(k+1)η] (t).

Similarly let us define :

where G t (Ỹ ) = − Note that :

Hence we have :

Assume that X 0 = Y 0 there exists Q such that , X T = Q({W t } t∈ [0,T ] ) and Y T = Q((W t ) t∈[0,T ] ).

Let µX T be the law ofX t∈ [0,T ] .

Same for µỸ T .

The proof here is similar to the proof of Lemma 8 in (Bubeck et al., 2015) .

By the data processing inequality we have:

@highlight

We propose surrogate based Constrained Langevin sampling with application in nano-porous material configuration design.