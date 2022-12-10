In complex transfer learning scenarios new tasks might not be tightly linked to previous tasks.

Approaches that transfer information contained only in the final parameters of a source model will therefore struggle.

Instead, transfer learning at at higher level of abstraction is needed.

We propose Leap, a framework that achieves this by transferring knowledge across learning processes.

We associate each task with a manifold on which the training process travels from initialization to final parameters and construct a meta-learning objective that minimizes the expected length of this path.

Our framework leverages only information obtained during training and can be computed on the fly at negligible cost.

We demonstrate that our framework outperforms competing methods, both in meta-learning and transfer learning, on a set of computer vision tasks.

Finally, we demonstrate that Leap can transfer knowledge across learning processes in demanding reinforcement learning environments (Atari) that involve millions of gradient steps.

Transfer learning is the process of transferring knowledge encoded in one model trained on one set of tasks to another model that is applied to a new task.

Since a trained model encodes information in its learned parameters, transfer learning typically transfers knowledge by encouraging the target model's parameters to resemble those of a previous (set of) model(s) (Pan & Yang, 2009 ).

This approach limits transfer learning to settings where good parameters for a new task can be found in the neighborhood of parameters that were learned from a previous task.

For this to be a viable assumption, the two tasks must have a high degree of structural affinity, such as when a new task can be learned by extracting features from a pretrained model BID12 BID14 Mahajan et al., 2018) .

If not, this approach has been observed to limit knowledge transfer since the training process on one task will discard information that was irrelevant for the task at hand, but that would be relevant for another task BID15 BID1 .We argue that such information can be harnessed, even when the downstream task is unknown, by transferring knowledge of the learning process itself.

In particular, we propose a meta-learning framework for aggregating information across task geometries as they are observed during training.

These geometries, formalized as the loss surface, encode all information seen during training and thus avoid catastrophic information loss.

Moreover, by transferring knowledge across learning processes, information from previous tasks is distilled to explicitly facilitate the learning of new tasks.

Meta learning frames the learning of a new task as a learning problem itself, typically in the few-shot learning paradigm BID20 Santoro et al., 2016; Vinyals et al., 2016) .

In this environment, learning is a problem of rapid adaptation and can be solved by training a meta-learner by backpropagating through the entire training process (Ravi & Larochelle, 2016; BID6 BID11 .

For more demanding tasks, meta-learning in this manner is challenging; backpropagating through thousands of gradient steps is both impractical and susceptible to instability.

On the other hand, truncating backpropagation to a few initial steps induces a short-horizon bias (Wu et al., 2018) .

We argue that as the training process grows longer in terms of the distance traversed on the loss landscape, the geometry of this landscape grows increasingly important.

When adapting to a new task through a single or a handful of gradient steps, the geometry can largely be ignored.

In contrast, with more gradient steps, it is the dominant feature of the training process.

To scale meta-learning beyond few-shot learning, we propose Leap, a light-weight framework for meta-learning over task manifolds that does not need any forward-or backward-passes beyond those already performed by the underlying training process.

We demonstrate empirically that Leap is a superior method to similar meta and transfer learning methods when learning a task requires more than a handful of training steps.

Finally, we evaluate Leap in a reinforcement Learning environment (Atari 2600; BID8 , demonstrating that it can transfer knowledge across learning processes that require millions of gradient steps to converge.

We start in section 2.1 by introducing the gradient descent algorithm from a geometric perspective.

Section 2.2 builds a framework for transfer learning and explains how we can leverage geometrical quantities to transfer knowledge across learning processes by guiding gradient descent.

We focus on the point of initialization for simplicity, but our framework can readily be extended.

Section 2.3 presents Leap, our lightweight algorithm for transfer learning across learning processes.

Central to our framework is the notion of a learning process; the harder a task is to learn, the harder it is for the learning process to navigate on the loss surface ( fig. 1 ).

Our framework is based on the idea that transfer learning can be achieved by leveraging information contained in similar learning processes.

Exploiting that this information is encoded in the geometry of the loss surface, we leverage geometrical quantities to facilitate the learning process with respect to new tasks.

We focus on the supervised learning setting for simplicity, though our framework applies more generally.

Given a learning objective f that consumes an input x ∈ R m and a target y ∈ R c and maps a parameterization θ ∈ R n to a scalar loss value, we have the gradient descent update as DISPLAYFORM0 where ∇f (θ i ) = E x,y∼p(x,y) ∇f (θ i , x, y) .

We take the learning rate schedule {α i } i and preconditioning matrices {S i } i as given, but our framework can be extended to learn these jointly with the initialization.

Different schemes represent different optimizers; for instance α i = α, S i = I n yields gradient descent, while defining S i as the inverse Fisher matrix results in natural gradient descent BID4 .

We assume this process converges to a stationary point after K gradient steps.

To distinguish different learning processes originating from the same initialization, we need a notion of their length.

The longer the process, the worse the initialization is (conditional on reaching equivalent performance, discussed further below).

Measuring the Euclidean distance between initialization and final parameters is misleading as it ignores the actual path taken.

This becomes crucial when we compare paths from different tasks, as gradient paths from different tasks can originate from the same initialization and converge to similar final parameters, but take very different paths.

Therefore, to capture the length of a learning process we must associate it with the loss surface it traversed.

The process of learning a task can be seen as a curve on a specific task manifold M .

While this manifold can be constructed in a variety of ways, here we exploit that, by definition, any learning process traverses the loss surface of f .

As such, to accurately describe the length of a gradient-based learning process, it is sufficient to define the task manifold as the loss surface.

In particular, because the learning process in eq. 1 follows the gradient trajectory, it constantly provides information about the DISPLAYFORM1 Figure 1: Example of gradient paths on a manifold described by the loss surface.

Leap learns an initialization with shorter expected gradient path that improves performance.geometry of the loss surface.

Gradients that largely point in the same direction indicate a well-behaved loss surface, whereas gradients with frequently opposing directions indicate an ill-conditioned loss surface-something we would like to avoid.

Leveraging this insight, we propose a framework for transfer learning that exploits the accumulation of geometric information by constructing a meta objective that minimizes the expected length of the gradient descent path across tasks.

In doing so, the meta objective intrinsically balances local geometries across tasks and encourages an initialization that makes the learning process as short as possible.

To formalize the notion of the distance of a learning process, we define a task manifold M as a submanifold of R n+1 given by the graph of f .

Every point p = (θ, f (θ)) ∈ M is locally homeomorphic to a Euclidean subspace, described by the tangent space T p M .

Taking R n+1 to be Euclidean, it is a Riemann manifold.

By virtue of being a submanifold of R n+1 , M is also a Riemann manifold.

As such, M comes equipped with an smoothly varying inner product g p : T p M × T p M → R on tangent spaces, allowing us to measure the length of a path on M .

In particular, the length (or energy) of any curve γ : [0, 1] → M is defined by accumulating infinitesimal changes along the trajectory, DISPLAYFORM2 DISPLAYFORM3 We use parentheses (i.e. γ(t)) to differentiate discrete and continuous domains.

With M being a submanifold of R n+1 , the induced metric on M is defined by g γ(t) (γ(t),γ(t)) = γ(t),γ(t) .

Different constructions of M yield different Riemann metrics.

In particular, if the model underlying f admits a predictive probability distribution P (y | x), the task manifold can be given an information geometric interpretation by choosing the Fisher matrix as Riemann metric, in which case the task manifold is defined over the space of probability distributions BID5 .

If eq. 1 is defined as natural gradient descent, the learning process corresponds to gradient descent on this manifold BID4 Martens, 2010; Pascanu & Bengio, 2014; Luk & Grosse, 2018) .Having a complete description of a task manifold, we can measure the length of a learning process by noting that gradient descent can be seen as a discrete approximation to the scaled gradient floẇ θ(t) = −S(t)∇f (θ(t)).

This flow describes a curve that originates in γ(0) = (θ 0 , f (θ 0 )) and follows the gradient at each point.

Going forward, we define γ to be this unique curve and refer to it as the gradient path from θ 0 on M .

The metrics in eq. 2 can be computed exactly, but in practice we observe a discrete learning process.

Analogously to how the gradient update rule approximates the gradient flow, the gradient path length or energy can be approximated by the cumulative chordal distance BID2 , DISPLAYFORM4 Figure 2: Left: illustration of Leap (algorithm 1) for two tasks, τ and τ .

From an initialization θ 0 , the learning process of each task generates gradient paths, Ψ τ and Ψ τ , which Leap uses to minimize the expected path length.

Iterating the process, Leap converges to a locally Pareto optimal initialization.

Right: the pull-forward objective (eq. 6) used to minimize the expected gradient path length.

Any gradient path Ψ τ = {ψ We write d when the distinction between the length or energy metric is immaterial.

Using the energy yields a slightly simpler objective, but the length normalizes each length segment and as such protects against differences in scale between task objectives.

In appendix C, we conduct an ablation study and find that they perform similarly, though using the length leads to faster convergence.

Importantly, d involves only terms seen during task training.

We exploit this later when we construct the meta gradient, enabling us to perform gradient descent on the meta objective at negligible cost (eq. 8).We now turn to the transfer learning setting where we face a set of tasks, each with a distinct task manifold.

Our framework is built on the idea that we can transfer knowledge across learning processes via the local geometry by aggregating information obtained along observed gradient paths.

As such, Leap finds an initialization from which learning converges as rapidly as possible in expectation.

Formally, we define a task τ = (f τ , p τ , u τ ) as the process of learning to approximate the relationship x → y through samples from the data distribution p τ (x, y).

This process is defined by the gradient update rule u τ (as defined in eq. 1), applied K τ times to minimize the task objective f τ .

Thus, a learning process starts at θ To understand how d transfers knowledge across learning processes, consider two distinct tasks.

We can transfer knowledge across these tasks' learning processes by measuring how good a shared initialization is.

Assuming two candidate initializations converge to limit points with equivalent performance on each task, the initialization with shortest expected gradient path distance encodes more knowledge sharing.

In particular, if both tasks have convex loss surfaces a unique optimal initialization exists that achieves Pareto optimality in terms of total path distance.

This can be crucial in data sparse regimes: rapid convergence may be the difference between learning a task and failing due to overfitting BID11 .Given a distribution of tasks p(τ ), each candidate initialization θ 0 is associated with a measure of its expected gradient path distance, DISPLAYFORM0 , that summarizes the suitability of the initialization to the task distribution.

The initialization (or a set thereof) with shortest expected gradient path distance maximally transfers knowledge across learning processes and is Pareto optimal in this regard.

Above, we have assumed that all candidate initializations converge to limit points of equal performance.

If the task objective f τ is non-convex this is not a trivial assumption and the gradient path distance itself does not differentiate between different levels of final performance.

As such, it is necessary to introduce a feasibility constraint to ensure only initializations with some minimum level of performance are considered.

We leverage that transfer learning never happens in a vacuum; we always have a second-best option, such as starting from a random initialization or a pretrained model.

This "second-best" initialization, ψ 0 , provides us with the performance we for all τ ∈ B do 6: DISPLAYFORM1 for all i ∈ {0, . . .

, K τ −1} do 8: DISPLAYFORM2 increment ∇F using the pull-forward gradient (eq. 8) would obtain on a given task in the absence of knowledge transfer.

As such, performance obtained by initializing from ψ 0 provides us with an upper bound for each task: a candidate solution θ 0 must achieve at least as good performance to be a viable solution.

Formally, this implies the task-specific requirement that a candidate θ 0 must satisfy DISPLAYFORM3 As this must hold for every task, we obtain the canonical meta objective DISPLAYFORM4 This meta objective is robust to variations in the geometry of loss surfaces, as it balances complementary and competing learning processes ( fig. 2 ).

For instance, there may be an initialization that can solve a small subset of tasks in a handful of gradient steps, but would be catastrophic for other related tasks.

When transferring knowledge via the initialization, we must trade off commonalities and differences between gradient paths.

In eq. 4 these trade-offs arise naturally.

For instance, as the number of tasks whose gradient paths move in the same direction increases, so does their pull on the initialization.

Conversely, as the updates to the initialization renders some gradient paths longer, these act as springs that exert increasingly strong pressure on the initialization.

The solution to eq. 4 thus achieves an equilibrium between these competing forces.

Solving eq. 4 naively requires training to convergence on each task to determine whether an initialization satisfies the feasibility constraint, which can be very costly.

Fortunately, because we have access to a second-best initialization, we can solve eq. 4 more efficiently by obtaining gradient paths from ψ 0 and use these as baselines that we incrementally improve upon.

This improved initialization converges to the same limit points, but with shorter expected gradient paths (theorem 1).

As such, it becomes the new second-best option; Leap (algorithm 1) repeats this process of improving upon increasingly demanding baselines, ultimately finding a solution to the canonical meta objective.

Leap starts from a given second-best initialization ψ 0 , shared across all tasks, and constructs baseline DISPLAYFORM0 for each task τ in a batch B. These provide a set of baselines Ψ = {Ψ τ } τ ∈B .

Recall that all tasks share the same initialization, ψ 0 τ = ψ 0 ∈ Θ. We use these baselines, corresponding to task-specific learning processes, to modify the gradient path distance metric in eq. 3 by freezing the forward point γ i+1 τ in all norms, DISPLAYFORM1 DISPLAYFORM2 represents the frozen forward point from the baseline and γ DISPLAYFORM3 ) the point on the gradient path originating from θ 0 .

This surrogate distance metric encodes the feasibility constraint; optimizing θ 0 with respect to Ψ pulls the initialization forward along each task-specific gradient path in an unconstrained variant of eq. 4 that replaces Θ with Ψ, DISPLAYFORM4 We refer to eq. 6 as the pull-forward objective.

Incrementally improving θ 0 over ψ 0 leads to a new second-best option that Leap uses to generate a new set of more demanding baselines, to further improve the initialization.

Iterating this process, Leap produces a sequence of candidate solutions to eq. 4, all in Θ, with incrementally shorter gradient paths.

While the pull-forward objective can be solved with any optimization algorithm, we consider gradient-based methods.

In theorem 1, we show that gradient descent onF yields solutions that always lie in Θ. In principle,F can be evaluated at any θ 0 , but a more efficient strategy is to evaluate DISPLAYFORM5 Theorem 1 (Pull-forward).

Define a sequence of initializations {θ DISPLAYFORM6 DISPLAYFORM7 For β s > 0 sufficiently small, there exist learning rates schedules {α DISPLAYFORM8 for all tasks such that θ 0 k→∞ is a limit point in Θ.Proof: see appendix A. Because the meta gradient requires differentiating the learning process, we must adopt an approximation.

In doing so, we obtain a meta-gradient that can be computed analytically on the fly during task training.

DifferentiatingF , we have DISPLAYFORM9 where J i τ denotes the Jacobian of θ i τ with respect to the initialization, ∆f DISPLAYFORM10 To render the meta gradient tractable, we need to approximate the Jacobians, as these are costly to compute.

Empirical evidence suggest that they are largely redundant BID11 Nichol et al., 2018) .

Nichol et al. (2018) further shows that an identity approximation yields a meta-gradient that remains faithful to the original meta objective.

We provide some further support for this approximation (see appendix B).

First, we note that the learning rate directly controls the quality of the approximation; for any K τ , the identity approximation can be made arbitrarily accurate by choosing a sufficiently small learning rates.

We conduct an ablation study to ascertain how severe this limitation is and find that it is relatively loose.

For the best-performing learning rate, the identity approximation is accurate to four decimal places and shows no signs of significant deterioration as the number of training steps increases.

As such, we assume J i ≈

I n throughout.

Finally, by evaluating ∇F at θ 0 = ψ 0 , the meta gradient contains only terms seen during standard training and can be computed asynchronously on the fly at negligible cost.

In practice, we use stochastic gradient descent during task training.

This injects noise in f as well as in its gradient, resulting in a noisy gradient path.

Noise in the gradient path does not prevent Leap from converging.

However, noise reduces the rate of convergence, in particular when a noisy gradient step results in f τ (ψ DISPLAYFORM11 If the gradient estimator is reasonably accurate, this causes the term ∆f i τ ∇f τ (θ i τ ) in eq. 8 to point in the steepest ascent direction.

We found that adding a stabilizer to ensure we always follow the descent direction significantly speeds up convergence and allows us to use larger learning rates.

In this paper, we augmentF with a stabilizer of the form DISPLAYFORM12 Adding ∇µ (re-scaled if necessary) to the meta-gradient is equivalent to replacing ∆f i τ with −|∆f i τ | in eq. 8.

This ensures that we never follow ∇f τ (θ i τ ) in the ascent direction, instead reinforcing the descent direction at that point.

This stabilizer is a heuristic, there are many others that could prove helpful.

In appendix C we perform an ablation study and find that the stabilizer is not necessary for Leap to converge, but it does speed up convergence significantly.

Transfer learning has been explored in a variety of settings, the most typical approach attempting to infuse knowledge in a target model's parameters by encouraging them to lie close to those of a pretrained source model (Pan & Yang, 2009 ).

Because such approaches can limit knowledge transfer BID15 BID1 , applying standard transfer learning techniques leads to catastrophic forgetting, by which the model is rendered unable to perform a previously mastered task (McCloskey & Cohen, 1989; BID13 .

These problems are further accentuated when there is a larger degree of diversity among tasks that push optimal parameterizations further apart.

In these cases, transfer learning can in fact be worse than training from scratch.

Recent approaches extend standard finetuning by adding regularizing terms to the training objective that encourage the model to learn parameters that both solve a new task and retain high performance on previous tasks.

These regularizers operate by protecting the parameters that affect the loss function the most (Miconi et al., 2018; Zenke et al., 2017; BID18 BID22 Serrà et al., 2018) .

Because these approaches use a single model to encode both global task-general information and local task-specific information, they can over-regularize, preventing the model from learning further tasks.

More importantly, Schwarz et al. (2018) found that while these approaches mitigate catastrophic forgetting, they are unable to facilitate knowledge transfer on the benchmark they considered.

Ultimately, if a single model must encode both task-generic and task-specific information, it must either saturate or grow in size (Rusu et al., 2016) .In contrast, meta-learning aims to learn the learning process itself (Schmidhuber, 1987; BID9 Santoro et al., 2016; Ravi & Larochelle, 2016; BID6 Vinyals et al., 2016; BID11 .

The literature focuses primarily on few-shot learning, where a task is some variation on a common theme, such as subsets of classes drawn from a shared pool of data BID21 Vinyals et al., 2016) .

The meta-learning algorithm adapts a model to a new task given a handful of samples.

Recent attention has been devoted to three main approaches.

One trains the meta-learner to adapt to a new task by comparing an input to samples from previous tasks (Vinyals et al., 2016; Mishra et al., 2018; Snell et al., 2017) .

More relevant to our framework are approaches that parameterize the training process through a recurrent neural network that takes the gradient as input and produces a new set of parameters (Ravi & Larochelle, 2016; Santoro et al., 2016; BID6 BID16 ).

The approach most closely related to us learns an initialization such that the model can adapt to a new task through one or a few gradient updates BID11 Nichol et al., 2018; BID3 BID23 .

In contrast to our work, these methods focus exclusively on few-shot learning, where the gradient path is trivial as only a single or a handful of training steps are allowed, limiting them to settings where the current task is closely related to previous ones.

It is worth noting that the Model Agnostic Meta Learner (MAML: BID11 can be written as DISPLAYFORM0 1 As such, it arises as a special case of Leap where only the final parameterization is evaluated in terms of its final performance.

Similarly, the Reptile algorithm (Nichol et al., 2018) , which proposes to update rule DISPLAYFORM1 , can be seen as a naive version of Leap that assumes all task geometries are Euclidean.

In particular, Leap reduces to Reptile if f τ is removed from the task manifold and the energy metric without stabilizer is used.

We find this configuration to perform significantly worse than any other (see section 4.1 and appendix C).

Related work studying models from a geometric perspective have explored how to interpolate in a generative model's learned latent space (Tosi et al., 2014; Shao et al., 2017; BID7 BID10 BID19 .

Riemann manifolds have also garnered attention in the context of optimization, as a preconditioning matrix can be understood as the instantiation of some Riemann metric BID5 BID0 Luk & Grosse, 2018) .

We consider three experiments with increasingly complex knowledge transfer.

We measure transfer learning in terms of final performance and speed of convergence, where the latter is defined as the area under the training error curve.

We compare Leap to competing meta-learning methods on the Omniglot dataset by transferring knowledge across alphabets (section 4.1).

We study Leap's ability to transfer knowledge over more complex and diverse tasks in a Multi-CV experiment (section 4.2) and finally evaluate Leap on in a demanding reinforcement environment (section 4.3).

The Omniglot BID21 dataset consists of 50 alphabets, which we define to be distinct tasks.

We hold 10 alphabets out for final evaluation and use subsets of the remaining alphabets for metalearning or pretraining.

We vary the number of alphabets used for meta-learning / pretraining from 1 to 25 and compare final performance and rate of convergence on held-out tasks.

We compare against no pretraining, multi-headed finetuning, MAML, the first-order approximation of MAML (FOMAML; BID11 , and Reptile.

We train on a given task for 100 steps, with the exception of MAML where we backpropagate through 5 training steps during meta-training.

For Leap, we report performance under the length metric (d 1 ); see appendix C for an ablation study on Leap hyper-parameters.

For further details, see appendix D.Any type of knowledge transfer significantly improves upon a random initialization.

MAML exhibits a considerable short-horizon bias (Wu et al., 2018) .

While FOMAML is trained full trajectories, but because it only leverages gradient information at final iteration, which may be arbitrarily uninformative, it does worse.

Multi-headed finetuning is a tough benchmark to beat as tasks are very similar.

Nevertheless, for sufficiently rich task distributions, both Reptile and Leap outperform finetuning, with Leap outperforming Reptile as the complexity grows.

Notably, the AUC gap between Reptile and Leap grows in the number of training steps ( FIG2 ), amounting to a 4 percentage point difference in final validation error TAB3 .

Overall, the relative performance of meta-learners underscores the importance of leveraging geometric information in meta-learning.

FORMULA0 , we consider a set of computer vision datasets as distinct tasks.

We pretrain on all but one task, which is held out for final evaluation.

For details, see appendix E. To reduce the computational burden during meta training, we pretrain on each task in the meta batch for one epoch using the energy metric (d 2 ).

We found this to reach equivalent performance to training on longer gradient paths or using the length metric.

This indicates that it is sufficient for Leap to see a partial trajectory to correctly infer shared structures across task geometries.

We compare Leap against a random initialization, multi-headed finetuning, a non-sequential version of HAT (Serrà et al., 2018) (i.e. allowing revisits) and a non-sequential version of Progressive Nets (Rusu et al., 2016), where we allow lateral connection between every task.

Note that this makes Progressive Nets over 8 times larger in terms of learnable parameters.

The Multi-CV experiment is more challenging both due to greater task diversity and greater complexity among tasks.

We report results on held-out tasks in table 1.

Leap outperforms all baselines on all but one transfer learning tasks (Facescrub), where Progressive Nets does marginally better than a random initialization owing to its increased parameter count.

Notably, while Leap does marginally worse than a random initialization, finetuning and HAT leads to a substantial drop in performance.

On all other tasks, Leap converges faster to optimal performance and achieves superior final performance.

To demonstrate that Leap can scale to large problems, both in computational terms and in task complexity, we apply it in a reinforcement learning environment, specifically Atari 2600 games BID8 .

We use an actor-critic architecture (Sutton et al., 1998) with the policy and the value function sharing a convolutional encoder.

We apply Leap with respect to the encoder using the energy metric (d 2 ).

During meta training, we sample mini-batches from 27 games that have an action space dimensionality of at most 10, holding out two games with similar action space dimensionality for evaluation, as well as games with larger action spaces (table 6).

During meta-training, we train on each task for five million training steps.

See appendix F for details.

We train for 100 meta training steps, which is sufficient to see a distinct improvement; we expect a longer meta-training phase to yield further gains.

We find that Leap generally outperforms a random initialization.

This performance gain is primarily driven by less volatile exploration, as seen by the confidence intervals in FIG4 ).

Leap finds a useful exploration space faster and more consistently, demonstrating that Leap can find shared structures across a diverse set of complex learning processes.

We note that these gains may not cater equally to all tasks.

In the case of WizardOfWor (part of the meta-training set), Leap exhibits two modes: in one it performs on par with the baseline, in the other exploration is protracted ( fig. 8 ).

This phenomena stems from randomness in the learning process, which renders an observed gradient path relatively less representative.

Such randomness can be marginalized by training for longer.

That Leap can outperform a random initialization on the pretraining set (AirRaid, UpNDown) is perhaps not surprising.

More striking is that it exhibits the same behavior on out-of-distribution tasks.

In particular, Alien, Gravitar and RoadRunner all have at least 50% larger state space than anything encountered during pretraining (appendix F, table 6), yet Leap outperforms a random initialization.

This suggests that transferring knowledge at a higher level of abstraction, such as in the space of gradient paths, generalizes to unseen task variations as long as underlying learning dynamics agree.

Transfer learning typically ignores the learning process itself, restricting knowledge transfer to scenarios where target tasks are very similar to source tasks.

In this paper, we present Leap, a framework for knowledge transfer at a higher level of abstraction.

By formalizing knowledge transfer as minimizing the expected length of gradient paths, we propose a method for meta-learning that scales to highly demanding problems.

We find empirically that Leap has superior generalizing properties to finetuning and competing meta-learners.

Proof.

We first establish that, for all s, DISPLAYFORM0 with strict inequality for at least some s. Because {β s } ∞ s=1 satisfies the gradient descent criteria, it follows that the sequence {θ To establish DISPLAYFORM1 , with strict inequality for some s, let DISPLAYFORM2 with ψ DISPLAYFORM3 .

Denote by E τ,i the expectation over gradient paths, DISPLAYFORM4 .

Note that DISPLAYFORM5 with p = 2 defining the meta objective in terms of the gradient path energy and p = 1 in terms of the gradient path length.

As we are exclusively concerned with the Euclidean norm, we omit the subscript.

By assumption, every β s is sufficiently small to satisfy the gradient descent criteriā DISPLAYFORM6 ; Ψ s ).

Adding and subtractingF (θ 0 s+1 , Ψ s+1 ) to the RHS, we have DISPLAYFORM7 As our main concern is existence, we will show something stronger, namely that there exists α i τ such that DISPLAYFORM8 with at least one such inequality strict for some i, τ, s, in which case DISPLAYFORM9 for any p ∈ {1, 2}. We proceed by establishing the inequality for p = 2 and obtain p = 1 as an immediate consequence of monotonicity of the square root.

Expanding h DISPLAYFORM10 Every term except z DISPLAYFORM11 Consider ĥi τ −ẑ DISPLAYFORM12 .

Using the above identities and first-order Taylor series expansion, we have DISPLAYFORM13 and similarly for (f τ (ŷ DISPLAYFORM14 Finally, consider the inner product DISPLAYFORM15 , where R i τ denotes an upper bound on the residual.

We extend g to operate on z DISPLAYFORM16 The first term is non-negative, and importantly, always non-zero whenever β s = 0.

Furthermore, α DISPLAYFORM17 Thus, for α DISPLAYFORM18 for all τ, s, with strict inequality for at least some τ, s. To also establish it for the gradient path length (p = 1), taking square roots on both sides of h DISPLAYFORM19 with strict inequality for at least some τ, s, in particular whenever β s = 0 and α i τ sufficiently small.

Then, to see that the limit point of Ψ s+1 is the same as that of Ψ s for β s sufficiently small, note that x i τ = y i−1 τ .

As before, by the gradient descent criteria, β s is such that To understand the role of the Jacobians, note that (we drop task subscripts for simplicity) DISPLAYFORM20 DISPLAYFORM21 where H f (θ j ) denotes the Hessian of f at θ j .

Thus, changes to θ i+1 are translated into θ 0 via all intermediary Hessians.

This makes the Jacobians memoryless up to second-order curvature.

Importantly, the effect of curvature can directly be controlled via α i , and by choosing α i small we can ensure J i (θ 0 ) ≈ I n to be a arbitrary precision.

In practice, this approximation works well (c.f.

BID11 Nichol et al., 2018) .

Moreover, as a practical matter, if the alternative is some other approximation to the Hessians, the amount of noise injected grows exponentially with every iteration.

The problem of devising an accurate low-variance estimator for the J i (θ 0 ) is highly challenging and beyond the scope of this paper.

To understand how this approximation limits our choice of learning rates α i , we conduct an ablation study in the Omniglot experiment setting.

We are interested in the relative precision of the identity approximation under different learning rates and across time steps, which we define as DISPLAYFORM22 where the norm is the Schatten 1-norm.

We use the same four-layer convolutional neural network as in the Omniglot experiment (appendix D).

For each choice of learning rate, we train a model from a random initialization for 20 steps and compute ρ every 5 steps.

Due to exponential growth of memory Average training loss p=2, µ=0, f τ =1 p=2, µ=1, f τ =1 p=2, µ=0, f τ =0 p=1, µ=0, f τ =1 p=1, µ=1, f τ =1 p=1, µ=0, f τ =0 Figure 6 : Average task training loss over meta-training steps.

p denotes thed p used in the meta objective, µ = 1 the use of the stabilizer, and f τ = 1 the inclusion of the loss in the task manifold.consumption, we were unable to compute ρ for more than 20 gradient steps.

We report the relative precision of the first convolutional layer.

We do not report the Jacobian with respect to other layers, all being considerably larger, as computing their Jacobians was too costly.

We computed ρ for all layers on the first five gradient steps and found no significant variation in precision across layers.

Consequently, we prioritize reporting how precision varies with the number of gradient steps.

As in the main experiments, we use stochastic gradient descent.

We evaluate α i = α ∈ {0.01, 0.1, 0.5} across 5 different tasks.

FIG14 summarizes our results.

Reassuringly, we find the identity approximation to be accurate to at least the fourth decimal for learning rates we use in practice, and to the third decimal for the largest learning rate (0.5) we were able to converge with.

Importantly, except for the smallest learning rate, the quality of the approximation is constant in the number of gradient steps.

The smallest learning rate that exhibits some deterioration on the fifth decimal, however larger learning rates provide an upper bound that is constant on the fourth decimal, indicating that this is of minor concern.

Finally, we note that while these results suggest the identity approximation to be a reasonable approach on the class of problems we consider, other settings may put stricter limits on the effective size of learning rates.

As Leap is a general framework, we have several degrees of freedom in specifying a meta learner.

In particular, we are free to choose the task manifold structure, the gradient path distance metric, d p , and whether to incorporate stabilizers.

These are non-trivial choices and to ascertain the importance of each, we conduct an ablation study.

We vary (a) the task manifold between using the full loss surface and only parameter space, (b) the gradient path distance metric between using the energy or length, and (c) inclusion of the stabilizer µ in the meta objective.

We stay as close as possible to the set-up used in the Omniglot experiment (appendix D), fixing the number of pretraining tasks to 20 and perform 500 meta gradient updates.

All other hyper-parameters are the same.

Our ablation study indicates that the richer the task manifold and the more accurate the gradient path length is approximated, the better Leap performs ( fig. 6) .

Further, adding a stabilizer has the intended effect and leads to significantly faster convergence.

The simplest configuration, defined in terms of the gradient path energy and with the task manifold identifies as parameter space, yields a meta gradient equivalent to the update rule used in Reptile.

We find this configuration to be less efficient in terms of convergence and we observe a significant deterioration in performance.

Extending the task manifold to the loss surface does not improve meta-training convergence speed, but does cut prediction error in half.

Adding the stabilizer significantly speeds up convergence.

These conclusions also hold under the gradient path length as distance measure, and in general using the gradient path length does better than using the gradient path energy as the distance measure.

Omniglot contains 50 alphabets, each with a set of characters that in turn have 20 unique samples.

We treat each alphabet as a distinct task and pretrain on up to 25 alphabets, holding out 10 out for final evaluation.

We use data augmentation on all tasks to render the problem challenging.

In particular, we augment any image with a random affine transformation by (a) random sampling a scaling factor between [0.8, 1.2], (b) random rotation between [0, 360), and (c) randomly cropping the height and width by a factor between [−0.2, 0.2] in each dimension.

This setup differs significantly from previous protocols (Vinyals et al., 2016; BID11 , where tasks are defined by selecting different permutations of characters and restricting the number of samples available for each character.

We use the same convolutional neural network architecture as in previous works (Vinyals et al., 2016; Schwarz et al., 2018) .

This model stacks a module, comprised of a 3 × 3 convolution with 64 filters, followed by batch-normalization, ReLU activation and 2 × 2 max-pooling, four times.

All images are downsampled to 28 × 28, resulting in a 1 × 1 × 64 feature map that is passed on to a final linear layer.

We define a task as a 20-class classification problem with classes drawn from a distinct alphabet.

For alphabets with more than 20 characters, we pick 20 characters at random, alphabets with fewer characters (4) are dropped from the task set.

On each task, we train a model using stochastic gradient descent.

For each model, we evaluated learning rates in the range [0.001, 0.01, 0.1, 0.5]; we found 0.1 to be the best choice in all cases.

See table 3 for further hyper-parameters.

We meta-train for 1000 steps unless otherwise noted; on each task we train for 100 steps.

Increasing the number of steps used for task training yields similar results, albeit at greater computational expense.

For each character in an alphabet, we hold out 5 samples in order to create a task validation set.

We allow different architectures between tasks by using different final linear layers for each task.

We use the same convolutional encoder as in the Omniglot experiment (appendix D).

Leap learns an initialization for the convolutional encoder; on each task, the final linear layer is always randomly initialized.

We compare Leap against (a) a baseline with no pretraining, (b) multitask finetuning, (c) HAT (Serrà et al., 2018) , and (d) Progressive Nets (Rusu et al., 2016) .

For HAT, we use the original formulation, but allow multiple task revisits (until convergence).

For Progressive Nets, we allow lateral connections between all tasks and multiple task revisits (until convergence).

Note that this makes Progressive Nets over 8 times larger in terms of learnable parameters than the other models.

inproceedings We train using stochastic gradient descent with cosine annealing (Loshchilov & Hutter, 2017) .

During meta training, we sample a batch of 10 tasks at random from the pretraining set and train until the early stopping criterion is triggered or the maximum amount of epochs is reached (see TAB6 ).

We used the same interval for selecting learning rates as in the Omniglot experiment (appendix D).

Only Leap benefited from using more than 1 epoch as the upper limit on task training steps during pretraining.

In the case of Leap, the initialization is updated after all tasks in the meta batch has been trained to convergence; for other models, there is no distinction between initialization and task parameters.

On a given task, training is stopped if the maximum number of epochs is reached TAB6 or if the validation error fails to improve over 10 consecutive gradient steps.

Similarly, meta training is stopped once the mean validation error fails to improve over 10 consecutive meta training batches.

We use Adam BID17 for the meta gradient update with a constant learning rate of 0.01.

We use no dataset augmentation.

MNIST images are zero padded to have 32 × 32 images; we use the same normalizations as Serrà et al. (2018) .

We use the same network as in Mnih et al. (2013) , adopting it to actor-critic algorithms by estimating both value function and policy through linear layers connected to the final output of a shared convolutional network.

Following standard practice, we use downsampled 84 × 84 × 3 RGB images as input.

Leap is applied with respect to the convolutional encoder (as final linear layers vary in size across environments).

We use all environments with an action space of at most 10 as our pretraining pool, holding out Breakout and SpaceInvaders.

During meta training, we sample a batch of 16 games at random from a pretraining pool of 27 games.

On each game in the batch, a network is initialized using the shared initialization and trained independently for 5 million steps, accumulating the meta gradient across games on the fly.

Consequently, in any given episode, the baseline and Leap differs only with respect to the initialization of the convolutional encoder.

We trained Leap for 100 steps, equivalent to training 1600 agents for 5 million steps.

The meta learned initialization was evaluated on the held-out games, a random selection of games seen during pretraining, and a random selection of games with action spaces larger than 10 TAB7 .

On each task, we use a batch size of 32, an unroll length of 5 and update the model parameters with RMSProp (using = 10 −4 , α = 0.99) with a learning rate of 10 −4 .

We set the entropy cost to 0.01 and clip the absolute value of the rewards to maximum 5.0.

We use a discounting factor of 0.99.

Figure 7 : Mean normalized episode scores on Atari games across training steps.

Scores are reported as moving average over 500 episodes.

Shaded regions depict two standard deviations across ten seeds.

KungFuMaster, RoadRunner and Krull have action state spaces that are twice as large as the largest action state encountered during pretraining.

Leap (orange) generally outperforms a random initialization, except for WizardOfWor, where a random initialization does better on average due to outlying runs under Leap's initialization.

@highlight

We propose Leap, a framework that transfers knowledge across learning processes by  minimizing the expected distance the training process travels on a task's loss surface.

@highlight

The article proposes a novel meta-learning objective aimed at outperforming state-of-the-art approaches when dealing with collections of tasks that exhibit substantial between-task diversity