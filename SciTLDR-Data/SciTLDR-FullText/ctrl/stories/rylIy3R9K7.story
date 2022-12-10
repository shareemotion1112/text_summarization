Generative adversarial network (GAN) is one of the best known unsupervised learning techniques these days due to its superior ability to learn data distributions.

In spite of its great success in applications, GAN is known to be notoriously hard to train.

The tremendous amount of time it takes to run the training algorithm and its sensitivity to hyper-parameter tuning have been haunting researchers in this area.

To resolve these issues, we need to first understand how GANs work.

Herein, we take a step toward this direction by examining the dynamics of GANs.

We relate a large class of GANs including the Wasserstein GANs to max-min optimization problems with the coupling term being linear over the discriminator.

By developing new primal-dual optimization tools, we show that, with a proper stepsize choice, the widely used first-order iterative algorithm in training GANs would in fact converge to a stationary solution with a sublinear rate.

The same framework also applies to multi-task learning and distributional robust learning problems.

We verify our analysis on numerical examples with both synthetic and real data sets.

We hope our analysis shed light on future studies on the theoretical properties of relevant machine learning problems.

Since it was first invented by Ian Goodfellow in his seminal work BID8 , generative adversarial networks (GANs) have been considered as one of the greatest discoveries in machine learning community.

It is an extremely powerful tool to estimate data distributions and generate realistic samples.

To train its implicit generative model, GAN uses a discriminator since traditional Bayesian methods that require analytic density functions are no longer applicable.

This novel approach inspired by zero sum game theory leads to a significant performance boost; GANs are able to generate samples in a fidelity level that is way beyond traditional Bayesian methods.

During the last few years, there have been numerous research articles in this area aiming at improving its performance (Radford et al., 2015; Zhao et al., 2016; Nowozin et al., 2016; Mao et al., 2017) .

GANs have now become one of most recognized unsupervised learning techniques and have been widely used in a variety of domains such as image generation (Nguyen et al., 2017) , image super resolution BID16 , imitation learning BID12 .Despite the great progress of GANs, many essential problems remain unsolved.

Why is GAN so hard to train?

How to tune the hyper-parameters to reduce instability in GAN training?

How to eliminate mode collapse and fake images that show up frequently in training ?

Comparing with many other machine learning techniques, the properties of GANs are far from being well understood.

It is quite likely that the theoretical foundation of GANs will become a longstanding problem.

The theoretical difficulty of GANs mainly lies in the following several aspects.

First, it is a non-convex optimization problem with a complicated landscape.

It is unclear how to solve such optimization problems efficiently.

The first-order method widely used in the literature via updating the generator and discriminator along descent/ascent direction does not seem to converge all the time.

Although some techniques were proposed to stabilize the training performance of the network, e.g., spectral normalization Miyato et al. (2018) , in fact, there is no evidence that these algorithms guarantee even local optimality.

Second, even if there were an efficient algorithm to solve this optimization problem, we do not know how well they generalize.

After all, the optimization formulation is based only on the samples generated by the underlying distribution but our goal is to recover this underlying distribution.

Of course, this is a problem faced by all machine learning techniques.

Last, there are no reliable ways to evaluate the quality of trained models.

There are a number of works in this topic (Salimans et al., 2016; BID11 , but human eyes inspection remains the primary approach to judge a GAN model.

In the present work, we focus on the first problem and analyze the dynamics of GANs from an optimization point of view.

More precisely, we study the convergence properties of the first-order method in GAN training.

Our contributions can be summarized as follows.

1) We formulate a large class of GAN problems as a primal-dual optimization problem with a coupling term that is linear over discriminator (see Section 2 for the exact formulation); 2) We prove that the simple primal-dual first-order algorithm converges to a stationary solution with a sublinear convergent rate O(1/t).There have been a number of papers that study the dynamics of GANs from an optimization viewpoint.

These works can be roughly divided into three categories.

In the first category, the authors focus on high level idea using nonparametric models.

This includes the original GAN paper BID8 , the Wasserstein GAN papers ; and many other works proposing new GAN structures.

In the second category, the authors consider the unrolled dynamics (Metz et al., 2016) , that is, the discriminator remains optimal or almost optimal during the optimization processes.

This is considerably different to the first-order iterative algorithm widely used in GAN training.

Recent works BID11 ; BID17 Sanjabi et al. (2018) provide global convergence analysis for this algorithm.

The last category is on the first-order primal-dual algorithm, in which both the discriminator and the generator update via (stochastic) gradient descent.

However, most of the convergence analysis are local BID4 Mescheder et al., 2017; Nagarajan & Kolter, 2017; BID18 .

Other related work including the following: In Qian et al. (2018) the authors consider a gradient descent/ascent algorithm for a special min-max problem arising from robust learning (min problem is unconstrained, max problem has simplex constraints); In Yadav et al. (2018) the GANs are treated as convex-concave primal-dual optimization problems.

This formulation is considerably different to our setup where GANs, as they should be, are formulated as nonconvex saddle point problems.

In BID5 , the authors investigated the properties of the optimal solutions, which is also different from our work focusing on convergence analysis of the first-order primal-dual algorithm.

In Zhao et al. (2018) , some unified framework covering several generative models, e.g., VAE, infoGAN, were proposed in the Lagrangian framework.

However, the dual variable in their problem is a Lagrangian multiplier, while in our problem, it is the discriminator of GAN.

Besides, the focus of their paper is not the optimization algorithm.

In BID2 , the authors related a class of GANs to constrained convex optimization problems.

More specifically, such GANs can be viewed as Lagrangian forms of these convex optimization problems.

The optimization variables in their formulation are the probability density of the generator and the function values of the discriminator.

Many issues like nonconvexity do not show up.

This is essentially a nonparametric model, which doesn't apply to cases when the discriminator and the generator are represented by parametric models.

On the other hand, our analysis is carried out on the parametric models directly and we have to deal with the nonconvexity of neural networks.

In BID9 a primal-dual algorithm has been studied for a non-convex linearly constrained problem (which can be reformulated into a min-max problem, with the max problem being linear and unconstrained, and with linear coupling between variables); In BID10 , BID3 and the references therein, first-order methods have been developed for convex-concave saddle point problems.

Compared to these works, our considered problem is more general, allowing non-convexity and non-smoothness in the objective, non-convex coupling between variables, and can further include constraints.

Moreover, we provide global convergence rate analysis, which is much stronger than the local analysis mentioned above.

It turns out that the primal-dual framework we study in this paper can also be applied to the distributional robust machine learning problems (Namkoong & Duchi, 2016 ) and the multi-task learning problems (Qian et al., 2018) .

In multi-task learning, the goal is to train a single neural network that would work for several different machine learning tasks.

Similarly, in distributional robust learning, the purpose is to have a single model that would work for a set of data distributions.

In both problems, an adversarial layer is utilized to improve the worst case performance, which leads to a primal-dual optimization structure that falls into the scope of problems we consider.

The rest of the paper is structured as follows.

In Section 2 we introduce GAN and its primal-dual formulation.

We provide details of the algorithms with proof sketches in Section 3.

The full proofs are relegated to the appendix.

We highlight our theoretical results in Section 4 via several numerical examples, with both synthetic and real datasets.

GAN is a type of deep generative model with implicit density functions.

It consists of two critical components: a generator and a discriminator.

The generator takes random variables with known distribution as input and outputs fake samples.

The discriminator is trained to distinguish real samples and fake samples.

In the original form of GAN, the generator is a neural network and the discriminator is a standard classifier with cross entropy cost.

Denote the underlying probability distributions of the data sample {y i } ny i=1 ⊂ R d and the random seed x (a random variable in R k ) of the generative model by P y and P x respectively.

The original or vanilla GAN is of the form BID8 DISPLAYFORM0 Here G stands for the set of possible generators and F the set of possible discriminators.

There have been numerous modifications of the original GAN, among which the Wasserstein GAN ) attracts a lot of attention.

It has the form DISPLAYFORM1 where G is the set of parameterized generators and F is the set of Lipschitz functions with Lipschitz constant 1.

This is a special case of a more general class of GANs with the same structure but different constraint set F. The inner maximization loop defines different notions of integral probability metrics (Müller, 1997) depending on the choices of F. Other than Wasserstein GAN, one interesting example with this structure is the generative moment matching networks (Li et al., 2015; .

Wasserstein GAN can be extended to general optimal transport cost c(x, y), which results in saddle point formulation DISPLAYFORM2 Note that the discriminator now becomes two functions ψ, φ instead of one.

When c(x, y) = x − y , the above reduces to the standard Wasserstein GAN in equation 2.

We observe that, compared to vanilla GAN in equation 1, the Wasserstein GAN in equation 2 and equation 3 has a special structure.

In the nonparametric form, the coupling between the generator G and the discriminator F is linear on F in Wasserstein GAN while nonlinear in vanilla GAN.

Indeed, replacing F by αF 1 + βF 2 in the coupling term E(F (G(x))) yields DISPLAYFORM0 This structure motivates us to study the following min-max primal-dual optimization problem DISPLAYFORM1 with l being strictly convex.

the other two functions h and g can be non-convex.

Here Y represents the discriminator F and X represents the generator G. Note that the coupling term g(X), Y is linear over the discriminator Y , but nonlinear in X.In real applications, we need to parameterize the discriminator and generator, which may lead to the loss of the property that the coupling is linear over discriminator.

Next, we present several cases where this linear structure stays.

Wasserstein GAN in linear quadratic Gaussian (LQG) setting was proposed in BID6 to understand GAN.

In this simplified case, the data distribution is Gaussian, the cost is quadratic and the generator is linear, namely, DISPLAYFORM0 It can be shown that it suffice to consider discriminator of the form DISPLAYFORM1 with A, B being positive definite.

The constraint φ(x) + ψ(y) ≤ 1 2 x − y 2 implies that B A −1 .

Consequently, the discriminator can be parametrized by a single variable A 0.

Additionally, G(x) is a zero mean Gaussian random variable with covariance θθ T .

Therefore, the Wasserstein GAN in equation 3 then reduces to DISPLAYFORM2 which is apparently in the form of equation 4.

When only samples DISPLAYFORM3 of x, y are available, this becomes DISPLAYFORM4

In general quadratic discriminators are not sufficient to distinguish complicate high dimension distributions.

In order to deal with more general data sets, we consider the setting where the discriminator is a linear combination of predefined basis functions.

More specifically, let {F i } n i=1 be the basis functions, and F = α i F i with some constraint on α, then the above formulation becomes DISPLAYFORM0 Here the term λ α 2 with λ > 0 is used to regularize α, or equivalently, the discriminator.

Similarly, for GAN structure in equation 3, we can restrict our discriminators φ and ψ to be linear combinations of basis functions DISPLAYFORM1 in equation 3 is difficult to impose precisely.

Instead, we use a regularization term l and obtain DISPLAYFORM2 Clearly, both equation 7 and equation 8 are of the form equation 4.

We remark that no constraint has been imposed on the generator G; it can be any general neural network.

The requirement that the discriminator is a linear combination of predefined basis functions could be strong, but in principle, any function can be approximated to an arbitrary precision with large enough bases.

In this work, we consider the following general min-max problem, DISPLAYFORM0 where X is a convex and compact set and the size of X is upper bounded by σ X ; h i (X) : DISPLAYFORM1 ∀i is a non-convex function and has Lipschitz continuous gradient with constant L X ; DISPLAYFORM2 ∀i are strongly convex with modulus γ > 0 and Lipschitz gradient constant L Y ; the matrix function g i (X) : X → dim(Y ) can also be non-convex, and it is assumed to be Lipschitz and Lipschitz gradient continuous with constants L g,1 and L g,2 .

We note that regardless of whether Y is a bounded set or not, one can show that for all x ∈ X , the maximizer Y * for the maximization problem lies in a bounded set; see Lemma 3 in the appendix for proof.

We note that by allowing constraints in the form of x ∈ X , and y ∈ Y, one can also include nonsmooth regularizers in the formulation.

As an example, if we add λ X 1 into the objective function, one can introduce a new variable z, and consider an equivalent problem with the constraints X 1 ≤ z, with the objective function changed to DISPLAYFORM3 The above formulation is quite general.

Compared with the existing convex-concave saddle point literature BID10 , BID3 , our formulation allows non-convexity in the minimization, which is essential to modelling the neural network structure of generators in GANs and general non-convex supervised tasks in multi-task learning; Compared with the non-convex linearly constrained problems considered in BID9 , it further allows non-linear and non-convex function g(X) to couple with Y .

Compared with the robust learning formulation given in equation 18, equation 9 can further include constraints and nonsmooth objective functions (thus can include nonsmooth regularizers such as 1 norm).It is important to note that due to the generality of problem equation 9, developing performance guaranteed first-order method, which only utilizing the gradient information about functions h i , g i , l i is very challenging.

To the best of our knowledge, there has been no such algorithm that can provably compute even first-order stationary solutions for problem equation 9.

Our proposed gradient primal-dual algorithm of solving equation 9 is listed below, in which we alternatingly perform first-order optimization to update X and Y : DISPLAYFORM0 DISPLAYFORM1 To be consistent with the optimization literature, we will refer to the X-step the "primal step" and the Y -step as the "dual step".

We note that 1/β and ρ are two positive parameters, that represent stepsizes of the two updates, and both of them should be small.

A few remarks are ready.

Remark 1 (projected gradient).

It can be easily verified that the updates of X r and Y r can be written down in closed form using the following alternating projected gradient descent/ascent steps: DISPLAYFORM2 Remark 2 (stochastic vs deterministic algorithm).

This work will be focused on the deterministic algorithm given in equation 10, because such an algorithm is representative of the primal-dual first-order dynamics used in training GANs and optimizing robust ML problems, and it is already challenging to analyze.

However, we do want to remark that, it is relatively straightforward to build upon our proof, by incorporating the standard technique in stochastic constrained optimization BID7 ) (such as using decreasing stepsizes, and certain randomization rule in picking the final solutions), to analyze the stochastic version of the algorithm, in which mini-batches of the component functions are randomly selected to update at each iteration.

However, in order to keep the discussion of the paper simple, we choose not to present such results.

In this section, we present our main convergence results for the primal-dual first-order algorithm given in equation 10.

We first present a few necessary lemmas.

Lemma 1. (Descent Lemma) Let (X r , Y r ) be a sequence generated by equation 10.

The descent of the objective function can be quantified by DISPLAYFORM0 From Lemma 1, it is not clear whether the objective function is decreased or not, since the primal step will consistently decrease the objective value while the dual step will increase the objective value.

The key in our analysis is to identify a proper "potential function", which can capture the essential dynamics of the algorithm, and will be able to reduce in all iterations.

Lemma 2.

When the following conditions are satisfied, DISPLAYFORM1 (12) then there exist c 1 , c 2 , c 3 , d > 0 such that potential function will monotonically decrease, i.e., DISPLAYFORM2 where DISPLAYFORM3 To state our main result, let us define the proximal gradient of the objective function as DISPLAYFORM4 where proj denotes the convex projection operator.

Clearly, when ∇L(X, Y ) = 0, then a first-order stationary solution of the problem equation 4 is obtained.

Theorem 1.

Suppose that the sequence (X r , Y r ) is generated by equation 10 and ρ, β satisfy the conditions equation 12.

For a given small constant , let T ( ) denote the iteration index satisfying the following inequality DISPLAYFORM5 Then there exists some constant C > 0 such that DISPLAYFORM6 where P denotes the lower bound of P r .The above result shows that our proposed algorithm converges to the first-order stationary point of the original problem in a sublinear rate.

We conduct several experiments to illustrate our results.

The first two examples are on GANs with both synthetic data and MNIST dataset, and the last one (supplemental material) is on multi-task learning with real data.

Our intention is by no means to show our algorithm generates superior samples than other methods.

Our main goal is to show that our first-order primal-dual algorithm would converge at least to a local solution.

All experiments are implemented on a NVIDIA TITAN Xp.

In the LQG setting, as discussed in Section 2.2.1, the generator is modeled by a linear map G(x) = θx with parameter θ and the discriminator is parametrized by a positive definite matrix A. The seed x is a zero-mean random variable with unit covariance.

We randomly generate a positive definite matrix Σ y as our covariance of the data samples.

The solution to this GAN problem satisfies A = I, θθ T = Σ y .We implement our algorithm with different step-sizes.

The results are shown in FIG0 for 20 dimensional data, from which we see that the algorithm converges when the step-size is sufficiently small, while diverges for a large step-size.

We also compare our algorithm with the one proposed in Sanjabi et al. FORMULA0 , which requires solving the inner maximization problem in each iteration.

As can be seen from the last plot of FIG0 , while these two algorithms take a similar number of iterations to converge, the total time consumption is more for Sanjabi et al. FORMULA0 as it takes more time to solve the maximization problem than to update the parameter one step along the gradient direction.

A detailed comparison is given in TAB0 .

To visualize the effectiveness of LQG GAN, we consider the problem in 2 and 3 dimensional spaces.

We plot the samples corresponding to both the learned and the real covariance matrices in FIG1 .

Clearly, the learned models match the underlying truth.

We test the GANs framework with discriminators linear on features (discussed in Section 2.2.2) on the MNIST (LeCun et al., 1998) data of size 28 × 28.

The network architecture of the generator is same as DCGAN (Radford et al., 2015) .

To get the reasonable basis functions for the discriminator, we first train a Wasserstein GAN model with a subset of the MNIST data for a small number (5k) of iterations.

We then use the last hidden layer of the discriminator as our bases.

We implement our algorithm with a different number of basis and the results are shown in Figures 3 and 4 .

We have two major observations here: i) the GANs with discriminators linear on features generate reasonable samples and the performance improves as we increase the number of bases; ii) the algorithm converges with a small step-size while diverges for a large step-size.

In this work, we presented a convergence result for a first-order algorithm on a class of non-convex max-min optimization problems that arise in many machine learning applications such as generative adversarial networks and multi-task learning.

To the best of our knowledge, this is the first convergence result for this type of primal-dual algorithms.

Our results allow us to analyze GANs with neural network generator as well as general multi-task non-convex supervised learning problems.

A critical assumption we made is that the inner maximization loop is a strictly convex problem.

For applications in GANs, our assumptions require the discriminator to be a linear combination of predefined basis functions.

Extending this to the most general cases where the discriminator is a neural network requires further investigations and will be a future research topic.

Multi-task machine learning (Qian et al., 2018) aims at learning a single model that would work for several different machine learning tasks.

Let DISPLAYFORM0 be n supervised learning problems, then a multi-task formulation is DISPLAYFORM1 where p is a probability vector to weight the tasks.

A common choice is the uniform distribution p = [1/n, . . .

, 1/n].

To improve the worst case performance, one can change p adaptively and attain the max-min formulation DISPLAYFORM2 where D is a distance function to regularize p.

Here we have also added an regularization term on W .A closely related topic is the distributional robustness (Namkoong & Duchi, 2016) problem DISPLAYFORM3 where P is a subset of the space of probability vectors.

Relaxing the hard constraint on p points to a regularized version DISPLAYFORM4 While p represents weights on different tasks in multi-task learning, it describes data distribution in distributional robustness.

It beautifully incorporates data uncertainties in the learning problems.

This formulation has also applications in adversarial learning BID18 , where p i f (x i , W ) denotes the loss given by data x i after adversarial reweighing through the inner maximization problem.

The goal of the outer minimization is to learn a model with parameters W such that the worst case loss is minimized.

We consider two supervised learning tasks with MNIST (LeCun et al., 1998) dataset and CIFAR10 BID13 ) data set.

We seek a single neural network that works for these two completely unrelated problems (see Section A).

First, we convert the MNIST data from 28 × 28 gray images to 32 × 32 color images so that it is in the same format as CIFAR10.

We use a standard AlexNet BID14 as our model.

We train the model with the robust multi-task learning framework and compare it to the results from three other methods: train with MNIST only, train with CIFAR10 only and train with both data sets but with even weight [0.5, 0.5].

The batch size we use is 128.The results are presented in TAB2 .

The last row is the results for the robust multi-task learning, and the "even mixture" in the second last row standards for uniform weight [0.5, 0.5] between the two tasks.

We see that the result of using robust multi-task learning framework is better than the one with even weight.

Moreover, its performance on each task is comparable to that trained from a single data set alone.

The optimal p is [0.205, 0.795] , where the first value is for the MNIST dataset.

We also implement our algorithm with different levels λ of regularizations.

The results are shown in FIG3 , where we display the loss functions in the first two plots and the weight for MNIST in the last plot.

Even though changing λ doesn't affect the convergence rate that much, it changes the optimal p significantly.

In our convergence analysis of the primal-dual algorithm, we use the optimality conditions of X rand Y r -subproblems repeatedly so that the quantities of measuring the size of the difference of the iterates, e.g., X r+1 − X r and Y r+1 − Y r , can be obtained.

Also, for the simplicity of the notation, we have the following definitions: DISPLAYFORM0 Then, we give the optimality condition of X r -subproblem and Y r -subproblem as follows, DISPLAYFORM1 DISPLAYFORM2 where − r+1 denotes the subgradient of the convex indicator function 1(y r+1 ∈ Y), Y r j,k denotes the entry at the j row and kth column of Y r , and g j,k (X r ) denotes the matrix value function mapping from X r to the value at the jth row and kth column of g(X r ).

Note that equation 23 is also equivalent to DISPLAYFORM3 Before going to the details, we will first introduce the following lemma that characterizes the upper bound of Y r .Lemma 3.

Let (X r , Y r ) be a sequence generated by equation 10.

The size of Y r is upper bounded by some constant number denoted by σ Y .

DISPLAYFORM4 From the optimality conditions, we know that DISPLAYFORM5 Adding these two inequalities, we have DISPLAYFORM6 which implies that DISPLAYFORM7 where in the first inequality we used the Lipschitz continuity; in the last inequality we used the strong convexity of function l(Y ).

Therefore, we have DISPLAYFORM8 Since X ∈ X , with X bounded by σ X , we have the claim that the distance between any two Y * 1 and Y * 2 are bounded by 2L g,1 σ X /γ.

In other words, Y * X r , ∀X r ∈ X are within a compact sets, where Y * X r arg min Y ∈Y ζ(X r , Y ) and the radius of the set is upper bounded by 2L g,1 σ X /γ.

Let Y r+1 denote the r + 1th iterate of the projected gradient descent method of solving the dual problem which is parameterized by X r+1 .

Because the dual problem is strongly convex, it is standard to show that DISPLAYFORM9 This implies that the distance between the iterate and the set is not increasing.

The proof is complete.

Based on this definition and the previous lemma, it is easy to check that the following holds DISPLAYFORM0 where L X and L g,2 are two constants defined after equation 4.

First, suppose that we choose β large enough such that β ≥ L X + σ Y L g,2 .

Then we have the following estimate of the descent of the objective value: DISPLAYFORM1 when in (a) we used the equation 30; In (b) wej used the optimality condition equation 22, and we choose β ≥ L X + σ Y L g,2 .Finally, combing the above results, we have DISPLAYFORM2 DISPLAYFORM3 Second, we can get the lower bound of A r , X r+1 − X r as follows: DISPLAYFORM4 Step 1).

First, we have DISPLAYFORM5 DISPLAYFORM6 Define a potential function P r+1 (X r+1 , Y r+1 ) + dQ r+1 .

We can obtain DISPLAYFORM7

<|TLDR|>

@highlight

We show that, with a proper stepsize choice, the widely used first-order iterative algorithm in training GANs would in fact converge to a stationary solution with a sublinear rate.

@highlight

This paper uses GANs and multi-task learning to provide a convergence guarantee for primal-dual algorithms on certain min-max problems.

@highlight

Analyses the learning dynamics of GANs by formulating the problem as a primal-dual optimisation problem by assuming a limited class of models