Hyperparameter optimization can be formulated as a bilevel optimization problem, where the optimal parameters on the training set depend on the hyperparameters.

We aim to adapt regularization hyperparameters for neural networks by fitting compact approximations to the best-response function, which maps hyperparameters to optimal weights and biases.

We show how to construct scalable best-response approximations for neural networks by modeling the best-response as a single network whose hidden units are gated conditionally on the regularizer.

We justify this approximation by showing the exact best-response for a shallow linear network with L2-regularized Jacobian can be represented by a similar gating mechanism.

We fit this model using a gradient-based hyperparameter optimization algorithm which alternates between approximating the best-response around the current hyperparameters and optimizing the hyperparameters using the approximate best-response function.

Unlike other gradient-based approaches, we do not require differentiating the training loss with respect to the hyperparameters, allowing us to tune discrete hyperparameters, data augmentation hyperparameters, and dropout probabilities.

Because the hyperparameters are adapted online, our approach discovers hyperparameter schedules that can outperform fixed hyperparameter values.

Empirically, our approach outperforms competing hyperparameter optimization methods on large-scale deep learning problems.

We call our networks, which update their own hyperparameters online during training, Self-Tuning Networks (STNs).

Regularization hyperparameters such as weight decay, data augmentation, and dropout (Srivastava et al., 2014) are crucial to the generalization of neural networks, but are difficult to tune.

Popular approaches to hyperparameter optimization include grid search, random search BID3 , and Bayesian optimization (Snoek et al., 2012) .

These approaches work well with low-dimensional hyperparameter spaces and ample computational resources; however, they pose hyperparameter optimization as a black-box optimization problem, ignoring structure which can be exploited for faster convergence, and require many training runs.

We can formulate hyperparameter optimization as a bilevel optimization problem.

Let w denote parameters (e.g. weights and biases) and λ denote hyperparameters (e.g. dropout probability).

Let L T and L V be functions mapping parameters and hyperparameters to training and validation losses, respectively.

We aim to solve 1 : DISPLAYFORM0 Substituting the best-response function w * (λ) = arg min w L T (λ, w) gives a single-level problem: DISPLAYFORM1 If the best-response w * is known, the validation loss can be minimized directly by gradient descent using Equation 2, offering dramatic speed-ups over black-box methods.

However, as the solution to a high-dimensional optimization problem, it is difficult to compute w * even approximately.

Following Lorraine & Duvenaud (2018) , we propose to approximate the best-response w * directly with a parametric functionŵ φ .

We jointly optimize φ and λ, first updating φ so thatŵ φ ≈ w * in a neighborhood around the current hyperparameters, then updating λ by usingŵ φ as a proxy for w * in Eq. 2: DISPLAYFORM2 Finding a scalable approximationŵ φ when w represents the weights of a neural network is a significant challenge, as even simple implementations entail significant memory overhead.

We show how to construct a compact approximation by modelling the best-response of each row in a layer's weight matrix/bias as a rank-one affine transformation of the hyperparameters.

We show that this can be interpreted as computing the activations of a base network in the usual fashion, plus a correction term dependent on the hyperparameters.

We justify this approximation by showing the exact best-response for a shallow linear network with L 2 -regularized Jacobian follows a similar structure.

We call our proposed networks Self-Tuning Networks (STNs) since they update their own hyperparameters online during training.

STNs enjoy many advantages over other hyperparameter optimization methods.

First, they are easy to implement by replacing existing modules in deep learning libraries with "hyper" counterparts which accept an additional vector of hyperparameters as input 2 .

Second, because the hyperparameters are adapted online, we ensure that computational effort expended to fit φ around previous hyperparameters is not wasted.

In addition, this online adaption yields hyperparameter schedules which we find empirically to outperform fixed hyperparameter settings.

Finally, the STN training algorithm does not require differentiating the training loss with respect to the hyperparameters, unlike other gradient-based approaches (Maclaurin et al., 2015; Larsen et al., 1996) , allowing us to tune discrete hyperparameters, such as the number of holes to cut out of an image BID12 , data-augmentation hyperparameters, and discrete-noise dropout parameters.

Empirically, we evaluate the performance of STNs on large-scale deep-learning problems with the Penn Treebank (Marcus et al., 1993) and CIFAR-10 datasets (Krizhevsky & Hinton, 2009) , and find that they substantially outperform baseline methods.

A bilevel optimization problem consists of two sub-problems called the upper-level and lower-level problems, where the upper-level problem must be solved subject to optimality of the lower-level problem.

Minimax problems are an example of bilevel programs where the upper-level objective equals the negative lower-level objective.

Bilevel programs were first studied in economics to model leader/follower firm dynamics (Von Stackelberg, 2010) and have since found uses in various fields (see BID10 for an overview).

In machine learning, many problems can be formulated as bilevel programs, including hyperparameter optimization, GAN training (Goodfellow et al., 2014) , meta-learning, and neural architecture search BID18 .Even if all objectives and constraints are linear, bilevel problems are strongly NP-hard (Hansen et al., 1992; Vicente et al., 1994) .

Due to the difficulty of obtaining exact solutions, most work has focused on restricted settings, considering linear, quadratic, and convex functions.

In contrast, we focus on obtaining local solutions in the nonconvex, differentiable, and unconstrained setting.

Let F, f : R n × R m → R denote the upper-and lower-level objectives (e.g., L V and L T ) and λ ∈ R n , w ∈ R m denote the upper-and lower-level parameters.

We aim to solve: DISPLAYFORM0 subject to w ∈ arg min DISPLAYFORM1 It is desirable to design a gradient-based algorithm for solving Problem 4, since using gradient information provides drastic speed-ups over black-box optimization methods (Nesterov, 2013) .

The simplest method is simultaneous gradient descent, which updates λ using ∂F /∂λ and w using ∂f /∂w.

However, simultaneous gradient descent often gives incorrect solutions as it fails to account for the dependence of w on λ.

Consider the relatively common situation where F doesn't depend directly on λ , so that ∂F /∂λ ≡ 0 and hence λ is never updated.

A more principled approach to solving Problem 4 is to use the best-response function (Gibbons, 1992 ).

Assume the lower-level Problem 4b has a unique optimum w * (λ) for each λ.

Substituting the best-response function w * converts Problem 4 into a single-level problem: DISPLAYFORM0 If w * is differentiable, we can minimize Eq. 5 using gradient descent on F * with respect to λ.

This method requires a unique optimum w * (λ) for Problem 4b for each λ and differentiability of w * .

In general, these conditions are difficult to verify.

We give sufficient conditions for them to hold in a neighborhood of a point (λ 0 , w 0 ) where w 0 solves Problem 4b given λ 0 .

Lemma 1. (Fiacco & Ishizuka, 1990 ) Let w 0 solve Problem 4b for λ 0 .

Suppose f is C 2 in a neighborhood of (λ 0 , w 0 ) and the Hessian ∂ 2 f /∂w 2 (λ 0 , w 0 ) is positive definite.

Then for some neighborhood U of λ 0 , there exists a continuously differentiable function w * : U → R m such that w * (λ) is the unique solution to Problem 4b for each λ ∈ U and w * (λ 0 ) = w 0 .Proof.

See Appendix B.1.The gradient of F * decomposes into two terms, which we term the direct gradient and the response gradient.

The direct gradient captures the direct reliance of the upper-level objective on λ, while the response gradient captures how the lower-level parameter responds to changes in the upper-level parameter: DISPLAYFORM1 Even if ∂F /∂λ ≡ 0 and simultaneous gradient descent is possible, including the response gradient can stabilize optimization by converting the bilevel problem into a single-level one, as noted by Metz et al. (2016) for GAN optimization.

Conversion to a single-level problem ensures that the gradient vector field is conservative, avoiding pathological issues described by Mescheder et al. (2017) .

In general, the solution to Problem 4b is a set, but assuming uniqueness of a solution and differentiability of w * can yield fruitful algorithms in practice.

In fact, gradient-based hyperparameter optimization methods can often be interpreted as approximating either the best-response w * or its Jacobian ∂w * /∂λ, as detailed in Section 5.

However, these approaches can be computationally expensive and often struggle with discrete hyperparameters and stochastic hyperparameters like dropout probabilities, since they require differentiating the training loss with respect to the hyperparameters.

Promising approaches to approximate w * directly were proposed by Lorraine & Duvenaud (2018) , and are detailed below.1.

Global Approximation.

The first algorithm proposed by Lorraine & Duvenaud (2018) approximates w * as a differentiable functionŵ φ with parameters φ.

If w represents neural net weights, then the mappingŵ φ is a hypernetwork (Schmidhuber, 1992; Ha et al., 2016) .

If the distribution p(λ) is fixed, then gradient descent with respect to φ minimizes: DISPLAYFORM0 If support(p) is broad andŵ φ is sufficiently flexible, thenŵ φ can be used as a proxy for w * in Problem 5, resulting in the following objective: min DISPLAYFORM1 2.

Local Approximation.

In practice,ŵ φ is usually insufficiently flexible to model w * on support(p).

The second algorithm of Lorraine & Duvenaud (2018) locally approximates w * in a neighborhood around the current upper-level parameter λ.

They set p( |σ) to a factorized Gaussian noise distribution with a fixed scale parameter σ ∈ R n + , and found φ by minimizing the objective: DISPLAYFORM2 Intuitively, the upper-level parameter λ is perturbed by a small amount, so the lower-level parameter learns how to respond.

An alternating gradient descent scheme is used, where φ is updated to minimize equation 9 and λ is updated to minimize equation 8.

This approach worked for problems using L 2 regularization on MNIST (LeCun et al., 1998) .

However, it is unclear if the approach works with different regularizers or scales to larger problems.

It requiresŵ φ , which is a priori unwieldy for high dimensional w.

It is also unclear how to set σ, which defines the size of the neighborhood on which φ is trained, or if the approach can be adapted to discrete and stochastic hyperparameters.

In this section, we first construct a best-response approximationŵ φ that is memory efficient and scales to large neural networks.

We justify this approximation through analysis of simpler situations.

Then, we describe a method to automatically adjust the scale of the neighborhood φ is trained on.

Finally, we formally describe our algorithm and discuss how it easily handles discrete and stochastic hyperparameters.

We call the resulting networks, which update their own hyperparameters online during training, Self-Tuning Networks (STNs).

We propose to approximate the best-response for a given layer's weight matrix W ∈ R Dout×Din and bias b ∈ R Dout as an affine transformation of the hyperparameters λ 3 : DISPLAYFORM0 Here, indicates elementwise multiplication and row indicates row-wise rescaling.

This architecture computes the usual elementary weight/bias, plus an additional weight/bias which has been scaled by a linear transformation of the hyperparameters.

Alternatively, it can be interpreted as directly operating on the pre-activations of the layer, adding a correction to the usual pre-activation to account for the hyperparameters: DISPLAYFORM1 This best-response architecture is tractable to compute and memory-efficient: it requires D out (2D in + n) parameters to representŴ φ and D out (2 + n) parameters to representb φ , where n is the number of hyperparameters.

Furthermore, it enables parallelism: since the predictions can be computed by transforming the pre-activations (Equation 11), the hyperparameters for different examples in a batch can be perturbed independently, improving sample efficiency.

In practice, the approximation can be implemented by simply replacing existing modules in deep learning libraries with "hyper" counterparts which accept an additional vector of hyperparameters as input 4 .

Given that the best-response function is a mapping from R n to the high-dimensional weight space R m , why should we expect to be able to represent it compactly?

And why in particular would equation 10 be a reasonable approximation?

In this section, we exhibit a model whose best-response function can be represented exactly using a minor variant of equation 10: a linear network with Jacobian norm regularization.

In particular, the best-response takes the form of a network whose hidden units are modulated conditionally on the hyperparameters.

Consider using a 2-layer linear network with weights w = (Q, s) ∈ R D×D × R D to predict targets t ∈ R from inputs x ∈ R D : a(x; w) = Qx, y(x; w) = s a(x; w)Suppose we use a squared-error loss regularized with an L 2 penalty on the Jacobian ∂y /∂x, where the penalty weight λ lies in R and is mapped using exp to lie R + : DISPLAYFORM0 Theorem 2.

Let w 0 = (Q 0 , s 0 ), where Q 0 is the change-of-basis matrix to the principal components of the data matrix and s 0 solves the unregularized version of Problem 13 given DISPLAYFORM1 where σ is the sigmoid function.

Proof.

See Appendix B.2.Observe that y(x; w * (λ)) can be implemented as a regular network with weights w 0 = (Q 0 , s 0 ) with an additional sigmoidal gating of its hidden units a(x; w * (λ)): DISPLAYFORM2 This architecture is shown in FIG0 .

Inspired by this example, we use a similar gating of the hidden units to approximate the best-response for deep, nonlinear networks.

The sigmoidal gating architecture of the preceding section can be further simplified if one only needs to approximate the best-response function for a small range of hyperparameter values.

In particular, for a narrow enough hyperparameter distribution, a smooth best-response function can be approximated by an affine function (i.e. its first-order Taylor approximation).

Hence, we replace the sigmoidal gating with linear gating, in order that the weights be affine in the hyperparameters.

The following theorem shows that, for quadratic lower-level objectives, using an affine approximation to the best-response function and minimizing E ∼p( |σ) [f (λ + ,ŵ φ (λ + ))] yields the correct best-response Jacobian, thus ensuring gradient descent on the approximate objective F (λ,ŵ φ (λ)) converges to a local optimum: DISPLAYFORM0 is Gaussian with mean 0 and variance DISPLAYFORM1 Proof.

See Appendix B.3.

The effect of the sampled neighborhood.

Left:

If the sampled neighborhood is too small (e.g., a point mass) the approximation learned will only match the exact best-response at the current hyperparameter, with no guarantee that its gradient matches that of the best-response.

Middle: If the sampled neighborhood is not too small or too wide, the gradient of the approximation will match that of the best-response.

Right: If the sampled neighborhood is too wide, the approximation will be insufficiently flexible to model the best-response, and again the gradients will not match.

The entries of σ control the scale of the hyperparameter distribution on which φ is trained.

If the entries are too large, thenŵ φ will not be flexible enough to capture the best-response over the samples.

However, the entries must remain large enough to forceŵ φ to capture the shape locally around the current hyperparameter values.

We illustrate this in FIG1 .

As the smoothness of the loss landscape changes during training, it may be beneficial to vary σ.

To address these issues, we propose adjusting σ during training based on the sensitivity of the upperlevel objective to the sampled hyperparameters.

We include an entropy term weighted by τ ∈ R + which acts to enlarge the entries of σ.

The resulting objective is: DISPLAYFORM0 This is similar to a variational inference objective, where the first term is analogous to the negative log-likelihood, but τ = 1.

As τ ranges from 0 to 1, our objective interpolates between variational optimization (Staines & Barber, 2012) and variational inference, as noted by Khan et al. (2018) .

Similar objectives have been used in the variational inference literature for better training BID5 and representation learning (Higgins et al., 2017) .Minimizing the first term on its own eventually moves all probability mass towards an optimum λ * , resulting in σ = 0 if λ * is an isolated local minimum.

This compels σ to balance between shrinking to decrease the first term while remaining sufficiently large to avoid a heavy entropy penalty.

When benchmarking our algorithm's performance, we evaluate F (λ,ŵ φ (λ)) at the deterministic current hyperparameter λ 0 .

(This is a common practice when using stochastic operations during training, such as batch normalization or dropout.)

We now describe the complete STN training algorithm and discuss how it can tune hyperparameters that other gradient-based algorithms cannot, such as discrete or stochastic hyperparameters.

We use an unconstrained parametrization λ ∈ R n of the hyperparameters.

Let r denote the element-wise function which maps λ to the appropriate constrained space, which will involve a non-differentiable discretization for discrete hyperparameters.

Let L T and L V denote training and validation losses which are (possibly stochastic, e.g., if using dropout) functions of the hyperparameters and parameters.

Define functions f, F by f (λ, w) = L T (r(λ), w) and F (λ, w) = L V (r(λ), w).

STNs are trained by a gradient descent scheme which alternates between updating φ for T train steps to minimize E ∼p( |σ) [f (λ + ,ŵ φ (λ + ))] (Eq. 9) and updating λ and σ for T valid steps to minimize DISPLAYFORM0 (Eq. 15).

We give our complete algorithm as Algorithm 1 and show how it can be implemented in code in Appendix G. The possible non-differentiability of r due to discrete hyperparameters poses no problem.

To estimate the derivative of E ∼p( |σ) [f (λ + ,ŵ φ (λ + ))] with respect to φ, we can use the reparametrization trick and compute ∂f /∂w and ∂ŵ φ/∂φ, neither of whose computation paths involve the discretization r. DISPLAYFORM1 with respect to a discrete hyperparameter λ i , there are two cases we must consider:

Initialize: Best-response approximation parameters φ, hyperparameters λ, learning rates DISPLAYFORM0 while not converged do DISPLAYFORM1 Case 1:

For most regularization schemes, L V and hence F does not depend on λ i directly and thus the only gradient is throughŵ φ .

Thus, the reparametrization gradient can be used.

Case 2: If L V relies explicitly on λ i , then we can use the REINFORCE gradient estimator BID15 to estimate the derivative of the expectation with respect to λ i .

The number of hidden units in a layer is an example of a hyperparameter that requires this approach since it directly affects the validation loss.

We do not show this in Algorithm 1, since we do not tune any hyperparameters which fall into this case.

We applied our method to convolutional networks and LSTMs (Hochreiter & Schmidhuber, 1997), yielding self-tuning CNNs (ST-CNNs) and self-tuning LSTMs (ST-LSTMs).

We first investigated the behavior of STNs in a simple setting where we tuned a single hyperparameter, and found that STNs discovered hyperparameter schedules that outperformed fixed hyperparameter values.

Next, we compared the performance of STNs to commonly-used hyperparameter optimization methods on the CIFAR-10 (Krizhevsky & Hinton, 2009) and PTB (Marcus et al., 1993) datasets.

Due to the joint optimization of the hypernetwork weights and hyperparameters, STNs do not use a single, fixed hyperparameter during training.

Instead, STNs discover schedules for adapting the hyperparameters online, which can outperform any fixed hyperparameter.

We examined this behavior in detail on the PTB corpus (Marcus et al., 1993) using an ST-LSTM to tune the output dropout rate applied to the hidden units.

The schedule discovered by an ST-LSTM for output dropout, shown in Figure 3 , outperforms the best, fixed output dropout rate (0.68) found by a fine-grained grid search, achieving 82.58 vs 85.83 validation perplexity.

We claim that this is a consequence of the schedule, and not of regularizing effects from sampling hyperparameters or the limited capacity ofŵ φ .To rule out the possibility that the improved performance is due to stochasticity introduced by sampling hyperparameters during STN training, we trained a standard LSTM while perturbing its dropout rate around the best value found by grid search.

We used (1) random Gaussian perturbations, and (2) sinusoid perturbations for a cyclic regularization schedule.

STNs outperformed both perturbation methods ( Table 2 : Final validation and test performance of each method on the PTB word-level language modeling task, and the CIFAR-10 image-classification task.

To determine whether the limited capacity ofŵ φ acts as a regularizer, we trained a standard LSTM from scratch using the schedule for output dropout discovered by the ST-LSTM.

Using this schedule, the standard LSTM performed nearly as well as the STN, providing evidence that the schedule itself (rather than some other aspect of the STN) was responsible for the improvement over a fixed dropout rate.

To further demonstrate the importance of the hyperparameter schedule, we also trained a standard LSTM from scratch using the final dropout value found by the STN (0.78), and found that it did not perform as well as when following the schedule.

The final validation and test perplexities of each variant are shown in TAB0 .Next, we show in Figure 3 that the STN discovers the same schedule regardless of the initial hyperparameter values.

Because hyperparameters adapt over a shorter timescale than the weights, we find that at any given point in training, the hyperparameter adaptation has already equilibrated.

As shown empirically in Appendix F, low regularization is best early in training, while higher regularization is better later on.

We found that the STN schedule implements a curriculum by using a low dropout rate early in training, aiding optimization, and then gradually increasing the dropout rate, leading to better generalization.

We evaluated an ST-LSTM on the PTB corpus (Marcus et al., 1993) , which is widely used as a benchmark for RNN regularization due to its small size (Gal & Ghahramani, 2016; Merity et al., 2018; BID14 .

We used a 2-layer LSTM with 650 hidden units per layer and 650-dimensional word embeddings.

We tuned 7 hyperparameters: variational dropout rates for the input, hidden state, and output; embedding dropout (that sets rows of the embedding matrix to 0); DropConnect (Wan et al., 2013) on the hidden-to-hidden weight matrix; and coefficients α and β that control the strength of activation regularization and temporal activation regularization, respectively.

For LSTM tuning, we obtained the best results when using a fixed perturbation scale of 1 for the hyperparameters.

Additional details about the experimental setup and the role of these hyperparameters can be found in Appendix D.

We compared STNs to grid search, random search, and Bayesian optimization.

6 FIG2 shows the best validation perplexity achieved by each method over time.

STNs outperform other meth- ods, achieving lower validation perplexity more quickly.

The final validation and test perplexities achieved by each method are shown in Table 2 .

We show the schedules the STN finds for each hyperparameter in FIG2 ; we observe that they are nontrivial, with some forms of dropout used to a greater extent at the start of training (including input and hidden dropout), some used throughout training (output dropout), and some that are increased over the course of training (embedding and weight dropout).

We evaluated ST-CNNs on the CIFAR-10 (Krizhevsky & Hinton, 2009) dataset, where it is easy to overfit with high-capacity networks.

We used the AlexNet architecture (Krizhevsky et al., 2012), and tuned: (1) continuous hyperparameters controlling per-layer activation dropout, input dropout, and scaling noise applied to the input, (2) discrete data augmentation hyperparameters controlling the length and number of cut-out holes (DeVries & Taylor, 2017), and (3) continuous data augmentation hyperparameters controlling the amount of noise to apply to the hue, saturation, brightness, and contrast of an image.

In total, we considered 15 hyperparameters.

We compared STNs to grid search, random search, and Bayesian optimization.

FIG4 shows the lowest validation loss achieved by each method over time, and Table 2 shows the final validation and test losses for each method.

Details of the experimental setup are provided in Appendix E. Again, STNs find better hyperparameter configurations in less time than other methods.

The hyperparameter schedules found by the STN are shown in FIG3 .

Bilevel Optimization.

BID10 provide an overview of bilevel problems, and a comprehensive textbook was written by BID1 .

When the objectives/constraints are restricted to be linear, quadratic, or convex, a common approach replaces the lower-level problem with its KKT conditions added as constraints for the upper-level problem (Hansen et al., 1992; Vicente et al., 1994) .

In the unrestricted setting, our work loosely resembles trust-region methods BID9 , which repeatedly approximate the problem locally using a simpler bilevel program.

In closely related work, Sinha et al. (2013) used evolutionary techniques to estimate the best-response function iteratively.

Hypernetworks.

First considered by Schmidhuber (1993; BID15 , hypernetworks are functions mapping to the weights of a neural net.

Predicting weights in CNNs has been developed in various forms BID11 BID16 .

Ha et al. (2016) used hypernetworks to generate weights for modern CNNs and RNNs.

BID6 used hypernetworks to globally approximate a bestresponse for architecture search.

Because the architecture is not optimized during training, they require a large hypernetwork, unlike ours which locally approximates the best-response.

Gradient-Based Hyperparameter Optimization.

There are two main approaches.

The first approach approximates w * (λ 0 ) using w T (λ 0 , w 0 ), the value of w after T steps of gradient descent on f with respect to w starting at (λ 0 , w 0 ).

The descent steps are differentiated through to approximate ∂w * /∂λ(λ 0 ) ≈ ∂w T /∂λ(λ 0 , w 0 ).

This approach was proposed by BID13 and used by Maclaurin et al. (2015) , Luketina et al. (2016) and Franceschi et al. (2018) .

The second approach uses the Implicit Function Theorem to derive ∂w * /∂λ(λ 0 ) under certain conditions.

This was first developed for hyperparameter optimization in neural networks (Larsen et al., 1996) and developed further by Pedregosa (2016).

Similar approaches have been used for hyperparameter optimization in log-linear models (Foo et al., 2008) , kernel selection BID8 Seeger, 2007) , and image reconstruction (Kunisch & Pock, 2013; BID7 .

Both approaches struggle with certain hyperparameters, since they differentiate gradient descent or the training loss with respect to the hyperparameters.

In addition, differentiating gradient descent becomes prohibitively expensive as the number of descent steps increases, while implicitly deriving ∂w * /∂λ requires using Hessian-vector products with conjugate gradient solvers to avoid directly computing the Hessian.

Model-Based Hyperparameter Optimization.

A common model-based approach is Bayesian optimization, which models p(r|λ, D), the conditional probability of the performance on some metric r given hyperparameters λ and a dataset D = {(λ i , r i )}.

We can model p(r|λ, D) with various methods (Hutter et al., 2011; Bergstra et al., 2011; Snoek et al., 2012; .

D is constructed iteratively, where the next λ to train on is chosen by maximizing an acquisition function C(λ; p(r|λ, D)) which balances exploration and exploitation.

Training each model to completion can be avoided if assumptions are made on learning curve behavior (Swersky et al., 2014; Klein et al., 2017) .

These approaches require building inductive biases into p(r|λ, D) which may not hold in practice, do not take advantage of the network structure when used for hyperparameter optimization, and do not scale well with the number of hyperparameters.

However, these approaches have consistency guarantees in the limit, unlike ours.

Model-Free Hyperparameter Optimization.

Model-free approaches include grid search and random search.

BID3 advocated using random search over grid search.

Successive Halving (Jamieson & Talwalkar, 2016) and Hyperband (Li et al., 2017) extend random search by adaptively allocating resources to promising configurations using multi-armed bandit techniques.

These methods ignore structure in the problem, unlike ours which uses rich gradient information.

However, it is trivial to parallelize model-free methods over computing resources and they tend to perform well in practice.

Hyperparameter Scheduling.

Population Based Training (PBT) (Jaderberg et al., 2017) considers schedules for hyperparameters.

In PBT, a population of networks is trained in parallel.

The performance of each network is evaluated periodically, and the weights of under-performing networks are replaced by the weights of better-performing ones; the hyperparameters of the better network are also copied and randomly perturbed for training the new network clone.

In this way, a single model can experience different hyperparameter settings over the course of training, implementing a schedule.

STNs replace the population of networks by a single best-response approximation and use gradients to tune hyperparameters during a single training run.

We introduced Self-Tuning Networks (STNs), which efficiently approximate the best-response of parameters to hyperparameters by scaling and shifting their hidden units.

This allowed us to use gradient-based optimization to tune various regularization hyperparameters, including discrete hyperparameters.

We showed that STNs discover hyperparameter schedules that can outperform fixed hyperparameters.

We validated the approach on large-scale problems and showed that STNs achieve better generalization performance than competing approaches, in less time.

We believe STNs offer a compelling path towards large-scale, automated hyperparameter tuning for neural networks.

We thank Matt Johnson for helpful discussions and advice.

MM is supported by an NSERC CGS-M award, and PV is supported by an NSERC PGS-D award.

RG acknowledges support from the CIFAR Canadian AI Chairs program.

Best-response of the parameters to the hyperparameters The (validation loss) direct (hyperparameter) gradient DISPLAYFORM0 The (elementary parameter) response gradient DISPLAYFORM1 The (validation loss) response gradient DISPLAYFORM2 The hyperparameter gradient: a sum of the validation losses direct and response gradients B PROOFS B.1 LEMMA 1Because w 0 solves Problem 4b given λ 0 , by the first-order optimality condition we must have: DISPLAYFORM3 The Jacobian of ∂f /∂w decomposes as a block matrix with sub-blocks given by: DISPLAYFORM4 We know that f is C 2 in some neighborhood of (λ 0 , w 0 ), so ∂f /∂w is continuously differentiable in this neighborhood.

By assumption, the Hessian ∂ 2 f /∂w 2 is positive definite and hence invertible at (λ 0 , w 0 ).

By the Implicit Function Theorem, there exists a neighborhood V of λ 0 and a unique continuously differentiable function w * : V → R m such that ∂f /∂w(λ, w * (λ)) = 0 for λ ∈ V and w * (λ 0 ) = w 0 .Furthermore, by continuity we know that there is a neighborhood DISPLAYFORM5 Combining this with ∂f /∂w(λ, w * (λ)) = 0 and using second-order sufficient optimality conditions, we conclude that w * (λ) is the unique solution to Problem 4b for all λ ∈ U .

This discussion mostly follows from Hastie et al. (2001) .

We let X ∈ R N ×D denote the data matrix where N is the number of training examples and D is the dimensionality of the data.

We let t ∈ R N denote the associated targets.

We can write the SVD decomposition of X as: DISPLAYFORM0 where U and V are N × D and D × D orthogonal matrices and D is a diagonal matrix with entries DISPLAYFORM1 We next simplify the function y(x; w) by setting u = s Q, so that y(x; w) = s Qx = u x.

We see that the Jacobian ∂y /∂x ≡ u is constant, and Problem 13 simplifies to standard L 2 -regularized least-squares linear regression with the following loss function: DISPLAYFORM2 It is well-known (see Hastie et al. (2001) , Chapter 3) that the optimal solution u * (λ) minimizing Equation 19 is given by: DISPLAYFORM3 Furthermore, the optimal solution u * to the unregularized version of Problem 19 is given by: DISPLAYFORM4 Recall that we defined Q 0 = V , i.e., the change-of-basis matrix from the standard basis to the principal components of the data matrix, and we defined s 0 to solve the unregularized regression problem given Q 0 .

Thus, we require that Q 0 s 0 = u * which implies s 0 = D −1 U t.

There are not unique solutions to Problem 13, so we take any functions Q(λ), s(λ) which satisfy Q(λ) s(λ) = v * (λ) as "best-response functions".

We will show that our chosen functions Q * (λ) = σ(λv + c) row Q 0 and s * (λ) = s 0 , where v = −1 and c i = 2 log(d i ) for i = 1, . . .

, D, meet this criteria.

We start by noticing that for any d ∈ R + , we have: DISPLAYFORM5 It follows that: DISPLAYFORM6 . . .

DISPLAYFORM7 . . .

DISPLAYFORM8 . . .

DISPLAYFORM9 B.3 THEOREM 3By assumption f is quadratic, so there exist A ∈ R n×n , B ∈ R n×m , C ∈ R m×m and d ∈ R n , e ∈ R m such that: DISPLAYFORM10 One can easily compute that: DISPLAYFORM11 Since we assume ∂ 2 f /∂w 2 0, we must have C 0.

Setting the derivative equal to 0 and using second-order sufficient conditions, we have: DISPLAYFORM12 Hence, we find: DISPLAYFORM13 We letŵ φ (λ) = U λ + b, and definef to be the function given by: DISPLAYFORM14 Substituting and simplifying: DISPLAYFORM15 Expanding, we find that equation 36 is equal to: DISPLAYFORM16 where we have: DISPLAYFORM17 We can simplify these expressions considerably by using linearity of expectation and that ∼ p( |σ) has mean 0: DISPLAYFORM18 We can use the cyclic property of the Trace operator, E ∼p( |σ) [ ] = σ 2 I, and commutability of expectation and a linear operator to simplify the expectations of 2 and 3 : DISPLAYFORM19 We can then differentiatef by making use of various matrix-derivative equalities (Duchi, 2007) to find: DISPLAYFORM20 Setting the derivative ∂f /∂b(λ 0 , U , b, σ) equal to 0, we have: DISPLAYFORM21 Setting the derivative for ∂f /∂U(λ 0 , U , b, σ) equal to 0, we have: DISPLAYFORM22 DISPLAYFORM23 DISPLAYFORM24 This is exactly the best-response Jacobian ∂w * /∂λ(λ) as given by Equation 34.

Substituting U = C −1 B into the equation 50 gives: DISPLAYFORM25 This is w * (λ 0 ) − ∂w * /∂λ(λ 0 ), thus the approximate best-response is exactly the first-order Taylor series of w * about λ 0 .

updated the model parameters, but did not update hyperparameters.

We terminated training when the learning rate dropped below 0.0003.We tuned variational dropout (re-using the same dropout mask for each step in a sequence) on the input to the LSTM, the hidden state between the LSTM layers, and the output of the LSTM.

We also tuned embedding dropout, which sets entire rows of the word embedding matrix to 0, effectively removing certain words from all sequences.

We regularized the hidden-to-hidden weight matrix using DropConnect (zeroing out weights rather than activations) (Wan et al., 2013) .

Because DropConnect operates directly on the weights and not individually on the mini-batch elements, we cannot use independent perturbations per example; instead, we sample a single DropConnect rate per mini-batch.

Finally, we used activation regularization (AR) and temporal activation regularization (TAR).

AR penalizes large activations, and is defined as: DISPLAYFORM26 where m is a dropout mask and h t is the output of the LSTM at time t. TAR is a slowness regularizer, defined as: DISPLAYFORM27 For AR and TAR, we tuned the scaling coefficients α and β.

For the baselines, the hyperparameter ranges were: [0, 0.95] for the dropout rates, and [0, 4] for α and β.

For the ST-LSTM, all the dropout rates and the coefficients α and β were initialized to 0.05 (except in Figure 3 , where we varied the output dropout rate).

Here, we present additional details on the CNN experiments.

For all results, we held out 20% of the training data for validation.

We trained the baseline CNN using SGD with initial learning rate 0.01 and momentum 0.9, on mini-batches of size 128.

We decay the learning rate by 10 each time the validation loss fails to decrease for 60 epochs, and end training if the learning rate falls below 10 −5 or validation loss has not decreased for 75 epochs.

For the baselines-grid search, random search, and Bayesian optimization-the search spaces for the hyperparameters were as follows: dropout rates were in the range We trained the ST-CNN's elementary parameters using SGD with initial learning rate 0.01 and momentum of 0.9, on mini-batches of size 128 (identical to the baselines).

We use the same decay schedule as the baseline model.

The hyperparameters are optimized using Adam with learning rate 0.003.

We alternate between training the best-response approximation and hyperparameters with the same schedule as the ST-LSTM, i.e. T train = 2 steps on the training step and T valid = 1 steps on the validation set.

Similarly to the LSTM experiments, we used five epochs of warm-up for the model parameters, during which the hyperparameters are fixed.

We used an entropy weight of τ = 0.001 in the entropy regularized objective (Eq. 15).

The cutout length was restricted to lie in {0, . . .

, 24} while the number of cutout holes was restricted to lie in {0, . . .

, 4}. All dropout rates, as well as the continuous data augmentation noise parameters, are initialized to 0.05.

The cutout length is initialized to 4, and the number of cutout holes is initialized to 1.

Overall, we found the ST-CNN to be relatively robust to the initialization of hyperparameters, but starting with low regularization aided optimization in the first few epochs.

Here, we draw connections between hyperparameter schedules and curriculum learning.

Curriculum learning BID2 ) is an instance of a family of continuation methods BID0 , which optimize non-convex functions by solving a sequence of functions that are ordered by increasing difficulty.

In a continuation method, one considers a family of training criteria C λ (w) with a parameter λ, where C 1 (w) is the final objective we wish to minimize, and C 0 (w) represents the training criterion for a simpler version of the problem.

One starts by optimizing C 0 (w) and then gradually increases λ from 0 to 1, while keeping w at a local minimum of C λ (w) BID2 ).

This has been hypothesized to both aid optimization and improve generalization.

In this section, we explore how hyperparameter schedules implement a form of curriculum learning; for example, a schedule that increases dropout over time increases stochasticity, making the learning problem more difficult.

We use the results of grid searches to understand the effects of different hyperparameter settings throughout training, and show that greedy hyperparameter schedules can outperform fixed hyperparameter values.

First, we performed a grid search over 20 values each of input and output dropout, and measured the validation perplexity in each epoch.

FIG8 shows the validation perplexity achieved by different combinations of input and output dropout, at various epochs during training.

We see that at the start of training, the best validation loss is achieved with small values of both input and output dropout.

As we train for more epochs, the best validation performance is achieved with larger dropout rates.

Next, we present a simple example to show the potential benefits of greedy hyperparameter schedules.

For a single hyperparameter-output dropout-we performed a fine-grained grid search and constructed a dropout schedule by using the hyperparameter values that achieve the best validation perplexity at each epoch in training.

As shown in FIG9 , the schedule formed by taking the best output dropout value in each epoch yields better generalization than any of the fixed hyperparameter values from the initial grid search.

In particular, by using small dropout values at the start of training, the schedule achieves a fast decrease in validation perplexity, and by using larger dropout later in training, it achieves better overall validation perplexity.

FIG10 shows the perturbed values for output dropout we used to investigate whether the improved performance yielded by STNs is due to the regularization effect, and not the schedule, in Section 4.1.

In this section, we provide PyTorch code listings for the approximate best-response layers used to construct ST-LSTMs and ST-CNNs: the HyperLinear and HyperConv2D classes.

We also provide a simplified version of the optimization steps used on the training set and validation set.

@highlight

We use a hypernetwork to predict optimal weights given hyperparameters, and jointly train everything together.