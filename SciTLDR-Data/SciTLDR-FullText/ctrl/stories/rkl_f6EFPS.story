The loss of a few neurons in a brain rarely results in any visible loss of function.

However, the insight into what “few” means in this context is unclear.

How many random neuron failures will it take to lead to a visible loss of function?

In this paper, we address the fundamental question of the impact of the crash of a random subset of neurons on the overall computation of a neural network and the error in the output it produces.

We study fault tolerance of neural networks subject to small random neuron/weight crash failures in a probabilistic setting.

We give provable guarantees on the robustness of the network to these crashes.

Our main contribution is a bound on the error in the output of a network under small random Bernoulli crashes proved by using a Taylor expansion in the continuous limit, where close-by neurons at a layer are similar.

The failure mode we adopt in our model is characteristic of neuromorphic hardware, a promising technology to speed up artificial neural networks, as well as of biological networks.

We show that our theoretical bounds can be used to compare the fault tolerance of different architectures and to design a regularizer improving the fault tolerance of a given architecture.

We design an algorithm achieving fault tolerance using a reasonable number of neurons.

In addition to the theoretical proof, we also provide experimental validation of our results and suggest a connection to the generalization capacity problem.

Understanding the inner working of artificial neural networks (NNs) is currently one of the most pressing questions (20) in learning theory.

As of now, neural networks are the backbone of the most successful machine learning solutions (37; 18) .

They are deployed in safety-critical tasks in which there is little room for mistakes (10; 40) .

Nevertheless, such issues are regularly reported since attention was brought to the NNs vulnerabilities over the past few years (37; 5; 24; 8) .

Fault tolerance as a part of theoretical NNs research.

Understanding complex systems requires understanding how they can tolerate failures of their components.

This has been a particularly fruitful method in systems biology, where the mapping of the full network of metabolite molecules is a computationally quixotic venture.

Instead of fully mapping the network, biologists improved their understanding of biological networks by studying the effect of deleting some of their components, one or a few perturbations at a time (7; 12) .

Biological systems in general are found to be fault tolerant (28) , which is thus an important criterion for biological plausibility of mathematical models.

Neuromorphic hardware (NH).

Current Machine Learning systems are bottlenecked by the underlying computational power (1) .

One significant improvement over the now prevailing CPU/GPUs is neuromorphic hardware.

In this paradigm of computation, each neuron is a physical entity (9) , and the forward pass is done (theoretically) at the speed of light.

However, components of such hardware are small and unreliable, leading to small random perturbations of the weights of the model (41) .

Thus, robustness to weight faults is an overlooked concrete Artificial Intelligence (AI) safety problem (2) .

Since we ground the assumptions of our model in the properties of NH and of biological networks, our fundamental theoretical results can be directly applied in these computing paradigms.

Research on NN fault tolerance.

In the 2000s, the fault tolerance of NNs was a major motivation for studying them (14; 16; 4) .

In the 1990s, the exploration of microscopic failures was fueled by the hopes of developing neuromorphic hardware (NH) (22; 6; 34) .

Taylor expansion was one of the tools used for the study of fault tolerance (13; 26) .

Another line of research proposes sufficient conditions for robustness (33) .

However, most of these studies are either empirical or are limited to simple architectures (41) .

In addition, those studies address the worst case (5) , which is known to be more severe than a random perturbation.

Recently, fault tolerance was studied experimentally as well.

DeepMind proposes to focus on neuron removal (25) to understand NNs.

NVIDIA (21) studies error propagation caused by micro-failures in hardware (3) .

In addition, mathematically similar problems are raised in the study of generalization (29; 30) and robustness (42) .

The quest for guarantees.

Existing NN approaches do not guarantee fault tolerance: they only provide heuristics and evaluate them experimentally.

Theoretical papers, in turn, focus on the worst case and not on errors in a probabilistic sense.

It is known that there exists a set of small worstcase perturbations, adversarial examples (5), leading to pessimistic bounds not suitable for the average case of random failures, which is the most realistic case for hardware faults.

Other branch of theoretical research studies robustness and arrives at error bounds which, unfortunately, scale exponentially with the depth of the network (29) .

We define the goal of this paper to guarantee that the probability of loss exceeding a threshold is lower than a pre-determined small value.

This condition is sensible.

For example, self-driving cars are deemed to be safe once their probability of a crash is several orders of magnitude less than of human drivers (40; 15; 36) .

In addition, current fault tolerant architectures use mean as the aggregation of copies of networks to achieve redundancy.

This is known to require exponentially more redundancy compared to the median approach and, thus, hardware cost.

In order to apply this powerful technique and reduce costs, certain conditions need to be satisfied which we will evaluate for neural networks.

Contributions.

Our main contribution is a theoretical bound on the error in the output of an NN in the case of random neuron crashes obtained in the continuous limit, where close-by neurons compute similar functions.

We show that, while the general problem of fault tolerance is NP-hard, realistic assumptions with regard to neuromorphic hardware, and a probabilistic approach to the problem, allow us to apply a Taylor expansion for the vast majority of the cases, as the weight perturbation is small with high probability.

In order for the Taylor expansion to work, we assume that a network is smooth enough, introducing the continuous limit (39) to prove the properties of NNs: it requires neighboring neurons at each layer to be similar.

This makes the moments of the error linear-time computable.

To our knowledge, the tightness of the bounds we obtain is a novel result.

In turn, the bound allows us to build an algorithm that enhances fault tolerance of neural networks.

Our algorithm uses median aggregation which results in only a logarithmic extra cost -a drastic improvement on the initial NP-hardness of the problem.

Finally, we show how to apply the bounds to specific architectures and evaluate them experimentally on real-world networks, notably the widely used VGG (38) .

Outline.

In Sections 2-4, we set the formalism, then state our bounds.

In Section 5, we present applications of our bounds on characterizing the fault tolerance of different architectures.

In Section 6 we present our algorithm for certifying fault tolerance.

In Section 7, we present our experimental evaluation.

Finally, in Section 8, we discuss the consequences of our findings.

Full proofs are available in the supplementary material.

Code is provided at the anonymized repo github.com/iclr-2020-fault-tolerance/code.

We abbreviate Assumption 1 → A1, Proposition 1 → P1, Theorem 1 → T1, Definition 1 → D1.

In this section, we define a fully-connected network and fault tolerance formally.

Notations.

For any two vectors x, y ∈ R n we use the notation (x, y) = n i=1 x i y i for the standard scalar product.

Matrix γ-norm for γ = (0, +∞] is defined as A γ = sup x =0 Ax γ / x γ .

We use the infinity norm x ∞ = max |x i | and the corresponding operator matrix norm.

We call a vector 0 = x ∈ R n q-balanced if min |x i | ≥ q max |x i |.

We denote [n] = {1, 2, ..., n}. We define the Hessian H ij = ∂ 2 y(x)/∂x i ∂x j as a matrix of second derivatives.

We write layer indices down and element indices up: W ij l .

For the input, we write x i ≡ x i .

If the layer is fixed, we omit its index.

We use the element-wise Hadamard product (x y) i = x i y i .

A neural network with L layers is a function y L : R n0 → R n L defined by a tuple (L, W, B, ϕ) with a tuple of weight matrices W = (W 1 , ..., W L ) (or their distributions) of size W l : n l × n l−1 , biases B = (b 1 , ..., b L ) (or their distributions) of size b l ∈ R n l by the expression y l = ϕ(z l ) with pre-activations z l = W l y l−1 + b l , l ∈ [L], y 0 = x and y L = z L .

Note that the last layer is linear.

We additionally require ϕ to be 1-Lipschitz 1 .

We assume that the network was trained 1 1-Lipschitz ϕ s.t.

|ϕ(x) − ϕ(y)| |x − y|.

If ϕ is K-Lipschitz, we rescale the weights to make K = 1: W ij l → W ij l /K.

This is the general case.

Indeed, if we rescale ϕ(x) → Kϕ(x), then, y l−1 → Ky l−1 , and in the sum z l = W ij /K · Ky l−1 ≡ z l using input-output pairs x, y * ∼ X × Y using ERM 2 for a loss ω.

Loss layer for input x and the true label y * (x) is defined as y L+1 (x) = E y * ∼Y |x ω(y L (x), y * )) with ω ∈ [−1, 1] This definition means that neurons crash independently, and they start to output 0 when they do.

We use this model because it mimics essential properties of NH (41) .

Components fail relatively independently, as we model faults as random (41) .

In terms of (41), we consider stuck-at-0 crashes, and passive fault tolerance in terms of reliability.

We extend the definition of ε-fault tolerance from (23) to the probabilistic case: Definition 5. (Probabilistic fault tolerance) A network (L, W, B, ϕ) is said to be (ε, δ)-fault tolerant over an input distribution (x, y * ) ∼ X × Y and a crash distribution U ∼ D|(x, W ) if P (x,y * )∼X×Y, U ∼D|(x,W ) {∆ L+1 (x) ≥ ε} ≤ δ.

For such network, we write (W, B) ∈ FT(L, ϕ, p, ε, δ).

Interpretation.

To evaluate the fault tolerance of a network, we compute the first moments of ∆ L+1 .

Next, we use tail bounds to guarantee (ε, δ)-FT.

This definition means that with high probability 1 − δ additional loss due to faults does not exceed ε.

Expectation over the crashes U ∼ D|x can be interpreted in two ways.

First, for a large number of neural networks, each having permanent crashes, E∆ is the expectation over all instances of a network implemented in the hardware multiple times.

For a single network with intermittent crashes, E∆ is the output of this one network over repetitions.

The recent review study (41) identifies three types of faults: permanent, transient, and intermittent.

Our definition 2 thus covers all these cases.

Now that we have a definition of fault tolerance, we show in the next section that the task of certifying or even computing it is hard.

In this section, we show why fault tolerance is a hard problem.

Not only it is NP-hard in the most general setting but, also, even for small perturbations, the error of the output of can be unacceptable.

3.1 NP-HARDNESS A precise assessment of an NN's fault tolerance should ideally diagnose a network by looking at the outcome of every possible failure, i.e. at the Forward Propagated Error (23) resulting from removing every possible subset of neurons.

This would lead to an exact assessment, but would be impractical in the face of an exponential explosion of possibilities as by Proposition 1 (proof in the supplementary material).

Proposition 1.

The task of evaluating E∆ k for any k = 1, 2, ... with constant additive or multiplicative error for a neural network with ϕ ∈ C ∞ , Bernoulli neuron crashes and a constant number of layers is NP-hard.

We provide a theoretical alternative for the practical case of neuromorphic hardware.

We overcome NP-hardness in Section 4 by providing an approximation dependent on the network, and not a constant factor one: for weights W we give ∆ and ∆ dependent on W such that ∆(W ) ≤ E∆ ≤ ∆(W ).

In addition, we only consider some subclass of all networks.

3.2 PESSIMISTIC SPECTRAL BOUNDS By Definition 4, the fault tolerance assessment requires to consider a weight perturbation W + U given current weights W and the loss change y L+1 (W +U )−y L+1 (W ) caused by it.

Mathematically, 2 Empirical Risk Minimization -the standard task 1/k

The loss is bounded for the proof of Algorithm 1's running time to work Quantity Discrete Continuous Input

In the literature, there are known spectral bounds on the Lipschitz coefficient for the case of input perturbations.

These bounds use the spectral norm of the matrix · 2 and give a global result, valid for any input.

This estimate is loose due to its exponential growth in the number of layers, as W 2 is rarely < 1.

See Proposition 2 for the statement:

The proof can be found in (29) or in the supplementary material.

It is also known that high perturbations under small input changes are attainable.

Adversarial examples (5) are small changes to the input resulting in a high change in the output.

This bound is equal to the one of (23), which is tight in case if the network has the fewest neurons.

In contrast, in Section 4, we derive our bound in the limit n → ∞. We have now shown that even evaluating fault tolerance of a given network can be a hard problem.

In order to make the analysis practical, we use additional assumptions based on the properties of neuromorphic hardware.

In this section, we introduce realistic simplifying assumptions grounded in neuromorphic hardware characteristics.

We first show that if faults are not too frequent, the weight perturbation would be small.

Inspired by this, we then apply a Taylor expansion to the study of the most probable case.

This assumption is based on the properties of neuromorphic hardware (35) .

Next, we then use the internal structure of neural networks.

Assumption 2.

The number of neurons at each layer n l is sufficiently big, n l 10 2 This assumption comes from the properties of state-of-the-art networks (1) .

The best and the worst fault tolerance.

Consider a 1-layer NN with n = n 0 and n L = n 1 = 1 at input x i = 1: y(x) = x i /n.

We must divide 1/n to preserve y(x) as n grows.

This is the most robust network, as all neurons are interchangeable.

Here E∆ = −p and Var∆ = p/n, variance decays with n. In contrast, the worst case y(x) = x 1 has all but one neuron unused.

Therefore E∆ = p and Var∆ = p, variance does not decay with n. The next proposition shows that under a mild additional regularity assumption on the network, Assumptions 1 and 2 are sufficient to show that the perturbation of the norm of the weights is small.

i=1 are q-balanced, for α > p, the norm of the weight perturbation U i l at layer l is probabilistically bounded as:

Inspired by this result, next, we compute the error ∆ given a small weight perturbation U using a Taylor expansion.

4 The inspiration for splitting the loss calculation into favorable and unfavorable cases comes from (27) 5 In order to certify fault tolerance, we need a precise bounds on the remainder of the Taylor approximation.

For example, for ReLU functions, Taylor approximation fails.

The supplementary material contains another counter-example to the Taylor expansion of an NN.

Instead, we give sufficient conditions for which the Taylor approximation indeed holds.

Figure 1: Discrete (standard) neural network approximating a continuous network Assumption 3.

As the width n increases, networks N N n have a continuous limit (39) N N n → N N c , where N N c is a continuous neural network (19) , and n = min{n l }.

That network N N c has globally bounded operator derivatives D k for orders k = 1, 2.

We define D 12 = max{D 1 , D 2 }.

See Figure 1 for a visualization of A3 and Table 1 for the description of A3.

The assumption means that with the increase of n, the network uses the same internal structure which just becomes more fine-grained.

The continuous limit holds in the case of explicit duplication, convolutional networks and corresponding explicit regularization.

The supplementary material contains a more complete explanation.

The derivative bound for order 2 is in contrast to the worse-case spectral bound which would be exponential in depth as in Proposition 2.

This is consistent with experimental studies (11) and can be connected to generalization properties via minima sharpness (17) .

Proposition 4.

Under A3, derivatives are equal the operator derivatives of the continuous limit:

Factors 1/n 0 and 1/n 1 appear because the network must represent the same y * as n 0 , n 1 → ∞. Then, ∂y/∂x i = ϕ (1)/n 1 and ∂ 2 y/∂x i ∂x j = ϕ (1)/n 2 1 .

Theorem 1.

For crashes at layer l and output error ∆ L at layer L under A1-3 with q = 1/n l and r = p + q, the mean and variance of the error can be approximated as

By Θ ± (1) we denote any function taking values in [−1, 1].

The full proof of the theorem is in the supplementary material.

The remainder terms are small as both p and q = 1/n l are small quantities under A1-2.

In addition, P4 implies ∂y L /∂ξ i ∼ 1/n l and thus, when n l → ∞, E∆ = O(1) remains constant, and Var∆ L = O(1/n l ).

This is the standard rate in case if we estimate the mean of a random variable by averaging over n l independent samples, and our previous example in the beginning of the Section shows that it is the best possible rate.

Our result shows sufficient conditions under which neural networks allow for such a simplification.

9 In the next sections we use the obtained theoretical evaluation to develop a regularizer increasing fault tolerance, and say which architectures are more fault-tolerant.

In this section, we apply the results from the previous sections to obtain a probabilistic guarantee on fault tolerance.

We identify which kinds of architectures are more fault-tolerant.

6 A necessary condition for D k to be bounded is to have a reasonable bound on the derivatives of the ground truth function y * (x).

We assume that this function is sufficiently smooth.

7 The proposition is illustrated in proof-of-concept experiments with explicit regularization in the supplementary material.

There are networks for which the conclusion of P4 would not hold, for example, a network with w ij = 1.

However, such a network does not approximate the same function as n increases since y(x) → ∞, violating A3 8 The derivative ∂yL/∂ξ

l is interpreted as if ξ i was a real variable.

9 However, the dependency Var∆ ∼ 1/n l is only valid if n < p −2 ∼ 10 8 to guarantee the first-order term to dominate, p/n > r 3 .

In case if this is not true, we can still render the network more robust by aggregating multiple copies with a mean, instead of adding more neurons.

Our current guarantees thus work in case if

In the supplementary material, we show that a more tight remainder, depending only on p/n, hence decreasing with n, is possible.

However, it complicates the equation as it requires D3.

Under the assumptions of previous sections, the variance of the error decays as Var∆ ∼ C l p l /n l as the error superposition is linear (see supplementary material for a proof), with C l not dependent on n l .

Given a fixed budget of neurons, the most fault-tolerant NN has its layers balanced: one layer with too few neurons becomes a single point of failure.

Specifically, an optimal architecture with a fixed sum N = n l has n l ∼ √ p l C l Given the previous results, certifying (ε, δ)-fault tolerance is trivial via a Chebyshev tail bound (proof in the supplementary material):

Var∆ L for E∆ and Var∆ calculated by Theorem 1.

Evaluation of E∆ or Var∆ using Theorem 1 would take the same amount of time as one forward pass.

However, the exact assessment would need O(2 n ) forward passes by Proposition 1.

In order to make the networks more fault tolerant, we now want to solve the problem of loss minimization under fault tolerance rather than ERM (as previously formulated in (41) (from T1) is connected to the target probability (P5).

Moreover, the network is required to be continuous by A3, which is achieved by making nearby neurons' weights close using a smoothing regularizing function smooth(W ) ≈ |W t (t, t )|dtdt .

The µ term for q-balancedness comes from P3 as it is a necessary condition for A3.

See the supplementary material for complete details.

HereL is the regularized loss, L the original one, and λ, µ, ν, ψ are the parameters:

We define the terms corresponding to λ, µ, ψ as

If we have achieved δ < 1/3 by P5, we can apply the well-known median trick technique (31), drastically increasing fault tolerance.

We only use R repetitions of the network with component-wise median aggregation to obtain (ε, δ · exp(−R))-fault tolerance guarantee.

See supplementary material for the calculations.

In addition, we show that after training, when E x ∇ W y L+1 (x) = 0, then E x E ξ ∆ L+1 = 0 + O(r 2 ) (proof in the supplementary material).

This result sheds some light on why neural networks are inherently fault-tolerant in a sense that the mean ∆ L+1 is 0.

Convolutional networks of architecture Conv-Activation-Pool can be seen as a sub-type of fully connected ones, as they just have locally-connected matrices W l , and therefore our techniques still apply.

Using large kernel sizes (see supplementary material for discussion), smooth pooling and activations lead to a better approximation.

We developed techniques to assess fault tolerance and to improve it.

Now we combine all the results into a single algorithm to certify fault tolerance.

We are now in the position to provide an algorithm (Algorithm 1) allowing to reach the desired (ε, δ)-fault tolerance via training with our regularizer and then physically duplicating the network a logarithmic amount of times in hardware, assuming independent faults.

We note that our algorithm works for a single input x but is easily extensible if the expressions in Propositions are replaced with expectations over inputs (see supplementary material).

In order to estimate the required number of neurons, we use bounds from T1 and P5 which require n ∼ p/ε 2 .

However, using the median approach allows for a fast exponential decrease in failure probability.

Once the threshold of failing with probability 1/3 is reached by P5, it becomes easy to reach any required guarantee.

The time complexity (compared to the one of training) of the algorithm is O(D 12 + C l p l /ε 2 ) and space complexity is equal to that of one training call.

See supplementary material for the proofs of resource requirements and correctness.

∞ , target ε and δ , the error tolerance parameters from the Definition 5, maximal complexity guess C ≈ |y l (t)|dt ≈ R

In this section, we test the theory developed in previous sections in proof-of-concept experiments.

We first show that we can correctly estimate the first moments of the fault tolerance using T1 for small (10-50 neurons) and larger networks (VGG).

We test the predictions of our theory such as decay of Var∆, the effect of our regularizer and the guarantee from Figure 2 : the variance decays as 1/n l .

We regularize with ψ = (10 −4 , 10 −2 ) for derivatives and smoothing respectively (see supplementary material for explanation of coefficients) and λ = 0.001.

Testing the algorithm.

We test the Algorithm 1 on the MNIST dataset for ε = 9 · 10 −3 , δ = 10

and obtain R = 20, n 1 = 500, λ = 10 −6 , µ = 10 −10 , ψ = (10 −4 , 10 −2 ).

We evaluate the tail bound experimentally.

Our experiment demonstrates the guarantee given by Proposition 5 and can be seen as an experimental confirmation of the algorithm's correctness.

See TheAlgorithm.ipynb.

We hence conclude that our proof-of-concept experiments show an overall validity of our assumptions and of our approach.

Fault tolerance is an important overlooked concrete AI safety issue (2) .

This paper describes a probabilistic fault tolerance framework for NNs that allows to get around the NP-hardness of the problem.

Since the crash probability in neuromorphic hardware is low, we can simplify the problem to allow for a polynomial computation time.

We use the tail bounds to motivate the assumption that the weight perturbation is small.

This allows us to use a Taylor expansion to compute the error.

To bound the remainder, we require sufficient smoothness of the network, for which we use the continuous limit: nearby neurons compute similar things.

After we transform the expansion into a tail bound to give a bound on the loss of the network.

This gives a probabilistic guarantee of fault tolerance.

Using the framework, we are able to guarantee sufficient fault tolerance of a neural network given parameters of the crash distribution.

We then analyze the obtained expressions to compare fault tolerance between architectures and optimize for fault tolerance of one architecture.

We test our findings experimentally on small networks (MNIST) as well as on larger ones (VGG-16, MobileNet).

Using our framework, one is able to deploy safer networks into neuromorphic hardware.

Mathematically, the problem that we consider is connected to the problem of generalization (29; 27) since the latter also considers the expected loss change under a small random perturbation

, except that these papers consider Gaussian noise and we consider Bernoulli noise.

Evidence (32) , however, shows that sometimes networks that generalize well are not necessarily fault-tolerant.

Since the tools we develop for the study of fault tolerance could as well be applied in the context of generalization, they could be used to clarify this matter.

12 Variance for P2 is derived in the supplementary material are formal statements of the results referred to in the main paper, they are in the same section as the reference.

4 We abbreviate Assumption 1 → A1, Proposition 1 → P1, Theorem 1 → T1, Definition 1 → D1.

Less precise statements with possible future research directions on fundamental questions required to make the guarantee even more strong are flushed right.

Notations.

For any two vectors x, y ∈ R n we use the notation (x, y) = n i=1 x i y i for the standard scalar product.

Matrix γ-norm for γ = (0, +∞] is defined as A γ = sup x =0 Ax γ / x γ .

We use the infinity norm x ∞ = max |x i |

and the corresponding operator matrix norm.

We call a vector 0 = x ∈ R n q-balanced if min |x i | ≥ q max |x i |.

We

Note that the last layer is linear.

We additionally require ϕ to 20 be 1-Lipschitz 1 .

We assume that the network was trained using input-output pairs x, y

for a loss ω.

Loss layer for input x and the true label y

We denote a (random) output of this network as

with activationsŷ l and pre-activationsẑ l , as in D1.

If ϕ is K-Lipschitz, we rescale the weights to make K = 1:

.

This is the general case.

Indeed, if we rescale ϕ(x) → Kϕ(x), then, y l−1 → Ky l−1 , and in the sum

The loss is bounded for the proof of Algorithm 1's running time to work

We extend the definition of ε-fault tolerance from [18] to the probabilistic case:

For such network, we write (W, B) ∈ FT(L, ϕ, p, ε, δ).

Proof.

To prove that a problem is NP-hard, it suffices to take another NP-hard problem, and reduce any instance of that problem to our problem, meaning that solving our problem would solve the original one.

We take the NP-hard Subset Sum problem.

It states: given a finite set of integers

there exist a non-empty subset S ⊆

[M ] such that x∈S x i = 0.

We take a subset sum instance x i ∈ Z and feed it as an input to a neural network with two first layer neurons 51 with ϕ being a piecewise-linear function from 0 to 1 at points −ε and ε for some fixed ε ∈ (0, 1).

Note that in 52 this proof we only compute ϕ at integer points, and therefore it is possible re-define ϕ ∈ C ∞ such that it has same 53 values in natural points.

Note that inputs to it are integers, therefore the outputs are 1 if and only if (iff) the sum is greater than zero.

First neuron has coefficients all 1s and second all −1s with no bias.

The next neuron has coefficients 1 and 1 for 56 both inputs, a bias −1.5 and threshold activation function, outputting 1 only if both inputs are 1.

Again, since 57 inputs to this neuron are integers, we can re-define ϕ to be C ∞ .

The final neuron is a (linear) identity function to 58 satisfy Definition 1.

It takes the previous neuron as its only input.

Now, we see that y(x) = 1 if and only if the 59 sum of inputs to a network is 0.

We now feed the entire set S to the network as its input x. In case if y(x) = 1 (which is easy to check), we have 61 arrived at a solution, as the whole set has zero sum.

In the following we consider the case when y(x) = 0.

Suppose that there exist an algorithm calculating answer to the question E∆ k > z for any finite-precision z.

Now, the expectation has terms inside

.

Suppose that one of the terms is non-zero.

Then y(x s) = 0, which means that the threshold neuron outputs a value > 0 which means that sum of its inputs is greater than 1.5.

This can only happen if they are both 1, which means that sum for a particular subset is both 66 ≥ 0 and ≤ 0.

By setting z = 0 we solve the subset sum problem using one call to an algorithm determining if 67 E∆ k > 0.

Indeed, in case if the original problem has no solution, the algorithm will output E∆ k ≤ 0 as there are 68 no non-zero terms inside the sum.

In case if there is a solution, there will be a non-zero term in the sum and thus 69 E∆ k > 0 (all terms are non-negative).

Now, suppose there exist an algorithm which always outputs an additive approximation to E∆ k giving numbers 71 µ and ε such that E∆ k ∈ (µ − ε, µ + ε) for some constant ε.

Take a network from previous example with additional analogous proof where we scale the outputs even more.

We note that computing the distribution of ∆ for one neuron, binary input and weights and a threshold activation 76 function is known as noise sensitivity in theoretical Computer Science.

There exists an exact assessment for

, however for the case w i = 1 the exact distribution is unknown [6] .

Additional Proposition 1 (Norm bound or bound b1).

For any norm · the error ∆ L at the last layer on input x and failures in input can be upper-bounded as (for ξ x being the crashed input)

Proof.

We assume a faulty input (crashes at layer 0).

By Definition 4 the error at the last layer is

, we use the 1-Lipschitzness of ϕ. In 85 particular, we use that for two vectors x and y and ϕ(x), ϕ(y) applied element-wise, we have ϕ(x)−ϕ(y) ≤ x−y 86 because the absolute difference in each component is less on the left-hand side.

We thus get

Now, since the inner layer act in a similar manner, we inductively apply the same argument for underlying layers and get

Moreover we have failing input, thusx = ξ x which completes the proof.

Proposition 2 (K using spectral properties [21] ).

Proof.

Application of AP1 for · = · 2 and x 2 = ξ x

We see, there are bounds considering different norms of matrices A γ (AP1) or using the triangle inequality 95 and absolute values (AP2).

They still lead to pessimistic estimates.

All these bounds are known to be loose [19] 96 due to W = sup x =0 W x / x being larger than the input-specific W x / x .

We will circumvent this issue by 97 considering the average case instead of the worst case.

Additional Corollary 1 (Infinity norm, connecting [9] to norm bounds).

For an input x with C = x ∞ for failures at the input with O(1) dependent only on layer size, but not on the weights,

Proof.

First we examine the expression (4) from the other paper [9] and show that it is equivalent to the result we are proving now:

here we have C l the maximal value at layer l, K the Lipschitz constant, w m is the maximal over output neurons

(rows of W ) and mean absolute value over input neurons (columns of W ) weight, f l is the number of crashed 103 neurons.

Now we set f 1 = pN 1 and f i = 0 for i > 1 and moreover we assume K = 1 as in the main paper.

4 As we only have one neuron, the index is one-dimensional Therefore the bound is rewritten as:

Now we notice that the quantity N l w l m = W l ∞ and therefore

Now we assume that the network has one more layer so that the bound from [9] works for a faulty input in the original network:

We write the definition of the expectation with

being the probability that a binary string of length n has a particular configuration with k ones, if its entries are i.i.d.

Bernoulli Be(p).

Here S l is the set of all possible network crash configurations at layer l.

Each configuration s l ∈ S l describes which neurons are crashed and which are working.

We have |S l | = 2 n l .

Now since for |s| = 0 the max{s x − x} = max{x − x} = 0, we consider cases |s| = 1 and |s| > 1.

For |s| > 1 the quantity f (p, n, |s|) = O(p 2 ) and therefore, in the first order:

Next we plug that back into the expression for E ∆ ∞ :

Now we note that this expression and the expression from [9] differ only in a numerical constant in front of the Additional Proposition 2 (Absolute value bound or bound b2).

The error on input x can be upper-bounded as:

For |W | being the matrix of absolute values (|W |) ij = |W ij |.

|x| means component-wise absolute values of the 110 vector.

Proof.

This expression involves absolute value of the matrices multiplied together as matrices and then multiplied 113 by an absolute value of the column vector.

The absolute value of a column vector is a vector of element-wise 114 absolute values.

We assume a faulty input.

By Definition 4, the error at the last layer is

Thus for the i'th component of the error,

Next we go one level deeper according to Definition 1:

And then apply the 1-Lipschitzness property of ϕ:

This brings us to the previous case and thus we analogously have

Inductively repeating these steps, we obtain:

Now we take the expectation and move it inside the matrix product by linearity of expectation:

The last expression involves E|x − x|.

This is component-wise expectation of a vector and we examine a single component.

Since ξ i is a Bernoulli random variable,

Plugging it into the last expression for E|∆ L | proves the proposition.

In this section we give sufficient conditions for which case the probability of large weight perturbation under a crash 120 distribution is small.

First, we define the properties of neuromorphic hardware.

5 A similar assumption on an "even distribution" of weights was made in [20] .

Toy examples.

Naturally, we expect the error to decay with an increase in number of neurons n, because of redundancy.

We show that this might not be always the case.

First, consider a 1-layer NN with n = n 0 and 130 n L = n 1 = 1 at input x i = 1: y(x) = x i /n.

This is most robust network, as all neurons are interchangeable as 131 they are computing the same function each.

Essentially, this is an estimate of the mean of ξ i given n samples.

Here

variance does not decay with n.

This proposition gives sufficient conditions under which the weight perturbation is small and it is less and less

as n increases and p decreases:

137 Proposition 3.

Under A1,2 and AA1, for α > p, the norm of the weight perturbation U i l at layer l is probabilistically 138 bounded as:

Proof.

See [5] for the proof details as this is a standard technique based on Chernoff bounds for Binomial distribution 141 with p → 0.

These are a quantified version of the Law of Large Numbers (which states that an average of 142 identical and independent random variables tends to its mean).

Specifically, Chernoff bounds show that the tail of 143 a distribution is exponentially small: P{X ≥ EX + εEX} ≤ exp(−cε 2 ).

Specifically, if we consider a case X = Bin(n, p) with p → 0, for which q = 1, we have [5, 1] for α = k/n > p:

In case if we rewrite k = αn, this gives us the result.

Specifically, if we consider

.

This shows that the probability that a fraction of Bernoulli successes is more and more concentrated 147 around its mean, np.

Therefore, it is less and less probable that the this fraction is ≥ α >

p.

Factor q appears because in the analysis of the sum

In this section, we develop a more precise expression for E∆ and Var∆. Previously, we have seen that the perfect 151 fault-tolerant network has Var∆ = O(p/n).

In this section, we give sufficient conditions when complex neural 152 networks behave as the toy example from the previous section as well.

We would like to obtain a Taylor expansion of the expectation of random variable E∆ = T 1 + T 2 in terms of p 154 and q = 1/n with r = p + q where

.

For the variance, we want to have Var∆ = T 3 + T 4

.

Our goal here is to make this expression decay as n → ∞ as in the toy example.

We will show in Theorem 1, the first-order terms indeed behave as we expect.

However, the expansion also contains 157 a remainder (terms T 2 and T 4 ).

In order for the remainder to be small, we need additional assumptions.

It is easy 158 to come up with an example for which the first term T 1 is zero, but the error is still non-zero, illustrating that the weights is not at all like its neighbors.

We thus show that discontinuity can lead to a lack of fault tolerance.

Next,

we generalize this sketch and show that some form of continuity is sufficient for the network to be fault-tolerant.

First, we reiterate on the toy motivating example from the previous section.

Consider a 1-layer neural network

We assume that all neurons and inputs are used.

Specifically, for the q-factor q(x) = max |x i |/ min |x i |, we have q(x) ≈ q(w) ∼ 1.

We are interested in how the network behaves as n → ∞ (the infinite width limit).

For the input,

we want the magnitude of individual entries to stay constant in this limit.

Thus, |x i | = O(1).

Now we look at 168 the function y(x) that the network computes.

Since the number of terms grows, each of them must decay as 1/n:

The simplest example has x = (1, ..., 1) and w = (1/n, ..., 1/n), which results in y(x) ≡ 1 for all n. Now

we consider fault tolerance of such a network.

We take ∆ =

that the i'th input neuron has failed.

Therefore, E∆ = −p/n x i w i = −p, and Var∆ = x

We see that the expectation does not change when n grows, but variance decays as 1/n.

These are the values 173 that we will try to obtain from real networks.

Intuitively, we expect that the fault tolerance increases when width 174 increases, because there are more neurons.

This is the case for the simple example above.

However, it is not the 175 case if all but one neuron are unused.

Then, the probability of failure is always p, no matter how many neurons 176 there are.

Thus, the variance does not decrease with n. We thus are interested in utilizing the neurons we have to 177 their maximal capacity to increase the fault tolerance.

conditions for which the remainder terms are small.

We use the first term explicitly when regularizing the network.

In order for the expansion to work, we formalize the difference between "all neurons doing the same task" and "all 181 but one are unused".

Specifically, we define a class of "good" networks for which fault tolerance is sufficient, via 182 the continuous limit [23] .

Functions, functionals and operators.

We call maps from numbers to numbers as functions, maps from 184 functions to numbers as functionals and maps from functions to functions as operators.

We consider a subset of the space of real-valued functions with domain T :

This is a space of bounded piecewise-continuous functions f ∈ F, f : T → R such that there is a finite set of

We note that a regular neural networks has T l = [n l ].

Continuous networks were re-introduced in [15] .

to exactly one from P d .

We define the approximation error A as

We define n = min{n 0 , ..., n L } the minimal width of the discrete network.

If a series of discrete networks N N n has A n → 0, n → ∞ for some N N c , we say that N N n → N N c .

The summary of correspondences between discrete and continuous quantities is shown in Table 1 208

Number of changes Proof.

Based on the proof of [15] .

By the property of discrete networks, they are universal approximators [15] .

Take a discrete network y with a sufficiently low error, and define a continuous network N N c by using a piecewise- n l discontinuities at each layer.

In addition, all functions are bounded since the weights are finite.

In the following we will always use H = F, as it is expressive enough (AP3), and it is useful for us.

We give a 214 sufficient condition for which A n → 0 as n → ∞.

In the following we writeî l = i−1 n l −1 for an index i = 1..n l , as the range for each of the indices is known.

Proof.

For layer 0, the error is 0 by definition of x i .

Also, x and its derivatives are bounded.

Suppose we 217 have shown that the error for layers 1..l − 1 is ≤ ε, and that y l is bounded with globally bounded deriva- |f (t)| = |W t (î, t)y l−1 (t) + W (î, t)y l−1 (t)| ≤ sup |W | sup |y| + sup |W | sup |y | < ∞ and does not depend on n or the 222 input, as the bound is global.

Therefore, the error at layer l is ≤ (ε + 1/n)C ≤ D/n for the initial choice ε = 1/n.

Thus, the full error is decaying as 1/n: A n = O(1/n).

Using that result, we conclude that for any sufficiently smooth operator y, there exist a sequence N N n ap-225 proximating y. First, by [15] , continuous networks are universal approximators.

Taking one and creating discrete derivatives of the function we want to approximate

Here for i s ∈ [n] and q j = |{i s = j}| we define (i 0 , ..., i k )! = n!

q1!...qn! a multinomial coefficient.

In the paper, we 231 only need k ∈ {1, 2}.

The limit means that close-by neurons inside a discrete network compute similar functions.

This helps fault 233 tolerance because neurons become redundant.

There can be many of such sequences N N n , but for the sake of clarity,

one might consider that gradient descent always converges to the "same way" to represent a function, regardless 235 of how many neurons the network has.

This can be made more realistic if we instead consider the distribution of 236 continuous networks to which the gradient descent converges to.

In this limit, the distribution of activations at 237 each layer (including inputs and outputs) stays the same as n grows, and only the number of nodes changes.

Note 238 that this limit makes neurons ordered: their order is not random anymore.

The limit works when n l are sufficiently 239 large.

After that threshold, the network stops learning new features and only refines the ones that it has created 240 already.

The derivative bound part of A3 can be enforced if there is a very good fit with the ground truth function 242ỹ (x).

Indeed, ifỹ(x) ≡ y L (x), then the derivative of the network depends on the function being approximated.

For 243 discrete-continuous quantities we write X d ≈ X c meaning that |X d − X c | ≤ ε for a sufficiently large n.

We note that our new assumptions extend the previous ones:

245 Additional Proposition 5.

A3 results in AA1 with

Proof.

A simple calculation:

Dividing these two gives the result.

Can we make the input norm bounded?

Sometimes, input data vectors are normalized: x = 1 [14] .

In 247 our analysis, it is better to do the opposite: x 1 = O(n 0 ).

First, we want to preserve the magnitude of outputs 248 to fit the same function as n increases.

In addition, the magnitude of pre-activations must stay in the same range 249 to prevent vanishing or exploding gradients.

We extend that condition to the input layer as well for convenience.

Another approach would be to, for example, keep the norm x constant as n grows.

In that case, to guarantee 251 the same output magnitude, input weights must be larger.

This is just a choice of scaling for the convenience of 252 notation.

We note that inputs can be still normalized component-wise.

Less weight decay leads to no fault tolerance.

Consider the simplest case with Gaussian weights: y = w T x, [24] .

In the Gaussian case, the error ∆ = − ξ i x i w i .

E∆ = 0 and Var∆ = x

This does not decay with n. A more formal statement can be found in AP6.

The NTK limit.

In the NTK limit [14] , there is a decay of the variance Var∆ since both x 2 = 1 and σ 2 ∼ 1/n, 261 but that happens because variance decreases in any case, not related to fault tolerance, since σ 2 x 2 2 ∼ 1/n.

The

"lazy" regime [7] of the NTK limit implies that every hidden neuron is close to its initialization, which is random.

Thus, there is no continuity in the network: close-by neurons do not compute similar functions.

Thus, the NTK 264 limit is incompatible with our continuous limit.

Below we formalize the claim that a network with constant pre-activations variance is not fault-tolerant.

This 266 means that an untrained network with standard initialization is not fault tolerant.

Next, we want to harness the benefits of having a continuous limit by A3.

We want to bound the Taylor 272 remainder, and for that, we bound higher-order derivatives.

We bound them via the operator derivative of the 273 continuous limit.

Operator derivatives.

The derivatives for functionals and operators are more cumbersome to handle because of the many arguments that they have (the function the operator is taken at, the function the resulting derivative operator acts on, and the arguments for these functions), so we explicitly define all of them below.

We consider operators of the following form, where x ∈ F and Y [x] ∈ F are functions:

This is one layer from AD1.

We define the operator derivative of Y [x] point-wise with the RHS being a functional derivative:

we use the standard functional derivative:

Next, we consider the functional derivative at a point.

This quantity is similar to just one component of the gradient of an ordinary function, with the complete gradient being similar to the functional derivative defined above.

We define the derivative at a point δF [x]/δx(s) via the Euler-Lagrange equation, since we only consider functionals of the form F t [x] = Y [x](t) for some fixed t which are then given by an integral expression

).

In this case, since the integral only depends on the function x explicitly, but not on its derivatives, the functional derivative at point s is

The definition of a functional derivative at a point δF [x]/δx(s) ("component of the gradient") can be reconciled with the definition of the functional derivative δF [x]/δx ("full gradient") if we consider the Dirac delta-function:

We define the operator derivative at a point in a point-wise manner via the functional derivative at a point:

Now we see that the rules for differentiating operators in our case are the same as the well-known rules for 276 the derivatives of standard vector-functions.

Indeed, if we consider y i (x) = σ( j W ij x j ) with the inner part imply that these quantities are equal.

In fact, we will show that they differ by a factor of 1/n k l where k is the order 280 of the derivative.

We characterize the derivatives of a discrete NN ∂ k y L /∂y

Intuitively, this means that the more neurons we have, the less is each of them important.

First, consider a 284 simple example y = σ( w i x i ).

Here the weight function is w i = 1/n, w(î) = 1, x i = 1, and A = 0.

Then 285 ∂y/∂x i = σ (·)w i ∼ 1/n and ∂ 2 y/∂x i ∂x j = σ (·)w i w j ∼ 1/n 2 .

We note that the expression inside the sigmoid has 286 a limit and it's close to the integral by continuity of σ and σ .

Proof.

Now we prove P4.

Consider the first and second derivatives.

Note that the operator derivatives only depend 288 on the number of layers and the dataset.

It does not depend on any n anymore.

By definition of a neural net

Crucially, the factor 1/n 0 appears because we do not sum over i 0 , as it is fixed, but we have a weight vector 290 W 1 ∼ 1/n 0 nevertheless.

For all other indices, we have a weight matrix W l ∼ 1/n l−1 as well as a summation over

From previous

Here, the factor 1/n 2 0 appears because we never sum over i 0 , but the weight matrix W 1 ∼ 1/n 0 appears twice.

When does the limit hold?

Now, we have assumed that a network has a continuous limit in A3.

However,

this might not be the case: the NTK limit [14] is an example of that, as weights there stay close to their random 295 initialization [7] , thus, they are extremely dissimilar.

1.

Explicit duplication.

We copy each neuron multiple times and reduce the outgoing weights.

If we set 297 W (t, t ) to be piecewise-constant, then the approximation error is zero A = 0, and it does not depend on 298 the degree of duplication.

This is the obvious case where the network is becoming more fault-tolerant using 299 duplication, and our framework confirms that.

The problem with explicit duplication of non-regularized 300 networks is that their fault-tolerance is suboptimal.

Not all neurons are equally important to duplicate.

Thus, it's more efficient to utilize all the neurons in the best way for fault tolerance by duplicating only the 302 important ones.

2.

Explicit regularization.

We make adjacent neurons compute similar functions, thus, allowing for redun-304 dancy.

We first consider the local "number of changes" metric (Table 1) .

Specifically, for some function is to use that for the weights to quantify their discontinuity:

The term above, if small, guarantees that, for each input neuron, neighboring output neurons will use it in similar ways.

The same is applied for W T as well:

guarantees that, for each output neuron, neighboring input neurons are used similarly by it.

In addition to making adjacent neurons computing similar functions, we add another term C 3 by Gaussian 311 smoothing: a Gaussian kernel is convolved with the weights, and the weights are subtracted from the result.

The difference shows how much the current value differs from an aggregated local "mean" value.

We explicitly enforce the continuous limit by adding a regularization term of smooth(W ) := C 1 +C 2 +C 3 .

Here can be found in continuity.py.

We check the derivative decay prediction (P4) which must follow from the continuity assumption A3, exper-

We repeat each experiment 10 times and report mean and standard deviation, see Figure 2 .

In contrast, that this does not demonstrate that our approach necessarily leads to a decrease in accuracy, as continuous Note: the condition above of C 1 + C 2 + C 3 being small is, strictly speaking, only a necessary condition for A3, but not a sufficient one.

Even if networks are smooth enough, they might not have a limit N N n → N N c , as they could implement the function in drastically different ways: for example, the networks N N n can be all smooth, but approximate different continuous functions.

However, this condition is sufficient for the derivatives D k to stay constant (second part of A3), which is the only requirement for T1 to work.

a Thus, our approach can give a formal guarantee of fault tolerance.

An attempt to give a truly sufficient condition for A3 would be to train the bigger network given duplicated weights of a smaller network, and penalizing a bigger network from having weights different from their initialization.

Different smoothing techniques.

Currently, C 2 is unused in our implementation, as it was sufficient to use C 1 + C 3 to achieve the fault tolerance required in our experiments.

Another method to make close-by neurons similar could be to use some clustering loss in the space of weights accounting for locality, like the Kohonen self-organizing map.

One more idea is to regularize the weights with a likelihood of a Gaussian process, enforcing smoothness and a required range of dependencies.

Another idea is to use the second derivative, which is connected to the curvature of a curve (x, y(x)) in the 2D space: κ = y /(1 + (y )

2 ) 3/2 .

The interpretation here is that κ = 1/R for R being the radius of curvature, a geometrical property of a curve showing how much it deviates from being a line.

a A discussion of why D k stay constant is given in the analysis of the correctness for Algorithm 1.

3.

Convolutional networks.

For images, this limit naturally corresponds to scaling the images [12] with 343 intermediary pixels being just added between the original ones.

Images are piecewise-continuous because 344 adjacent pixels are likely to belong to the same object having the same color, and there are only finitely many objects on the image.

Convolutional networks, due to their locality, fit our assumption on one condition.

We 346 need to have a large enough kernel size, as otherwise the non-smoothness is high.

Specifically, for CNNs, C 1 347 is small, as neighboring neurons have similar receptive fields.

In contrast, C 2 can be large in case if the kernel 348 size is small: for example, the kernel (−1, 1) in the 1D case will always result in a high discontinuity: the 349 coefficients are vastly different and require more values in between them to allow for a continuous limit and 350 redundancy.

The notebook ConvNetTest-VGG16-ManyImages.ipynb investigates into this.

We note that we do not present 352 this result in the main paper and make it a future research direction instead.

In this paper, we give qualitative 353 description of applicability of our techniques to convolutional networks (big kernel size for a small C 2 with 354 respect to kernel size, smooth activation and pooling to allow for T1 to work), as it would be out of the scope 355 of a single paper to go into details.

How does the network behave when the number of neurons increases, and it is trained with gradient descent from scratch?

First, there are permutations of neurons, which we ignore.

Secondly, there could be many ways to represent the same function.

One constraint is that the magnitude of outputs in the output layer is preserved.

Intermediate layers need to have a non-vanishing pre-activation values to overcome vanishing/exploding gradients.

In addition, input limit might be enforced such that x i ≈ x(î).

Now, gradient descent results in a discrete network N N d which can be seen as a discretization of some continuous network N N c .

Since NN's derivatives are globally bounded, GD converges to a critical point.

Each critical point determines the range of initializations which lead to it, partitioning the whole space into regions.

Each fixed point with a sufficiently low loss thus corresponds to a set of continuous networks "passing through" a resulting discrete network.

Each of the continuous networks can have different implementations in discrete networks of larger size.

We choose a path in networks of different sizes n and denote the probability s n over initializations to choose that particular continuous network.

Therefore, on that path, derivatives decay as we want since N N n → N N c .

The problem might arise if a particular continuous limit N N c has an extremely small probability (over initializations) of gradient descent giving N N n : if s n → 0, this particular network N N c is unlikely to appear.

We leave the study of this as a future research direction.

Now we use the derivative decay from P4 to show fault tolerance using a Taylor expansion.

We write q = 1/n l 360 and r = p + q. In the following we will use Assumption 3 only by its consequence -Proposition 4.

We note that 361 the conclusion of it can hold in other cases as well.

We just give sufficient conditions for which it holds.

is interpreted as if ξ i l was a real variable.

Proof.

We consider crashes at layer l as crashes in the input x to the rest of the layers of the network.

Thus,

without loss of generality, we set l = 0.

) − y(x).

Then we explicitly compute

)

ξ by the Taylor theorem with a Lagrange remainder.

We assume x ∞ ≤ 1 (otherwise we rescale W 1 ).

We group the terms into distinct cases i = j and i = j:

The second term is

The third term is

Therefore, we have an expansion

.

The expectation just decays with p, but not with n 0 .

Now, consider the variance Var∆ = Var(∆ (0)ξ

.

This is the leading term, the rest are smaller.

And the second term VarV 2 ≤ EV

Consider the final term Cov( consider the expression for the variance and explicitly compute a correlation between ξ i ∆ i (t(ξ)) and ξ j ∆ j (t(ξ)).

However, we were unsuccessful in doing so.

Another approach is to take one more term in the expansion: O(r 5 ) -it 388 will make the previous term with p 3 go away, leaving only terms p 2 /n and p/n 2 , as the difference expressions in the

, we get a remainder p/n right away, but need to compute f (µ) which

is the network at a modified input (x − px).

we sum the individual layer terms, hence l .

Proposition 4 motivates the inverse dependency on n l .

Median trick.

Suppose that we have a random variable X satisfying P{X ≥ ε} < 1/3.

Then we create R 398 independent copies of X and calculate

.

Then P{X ≥ ε} < (1/3) R/2 because in order for the 399 median to be larger than value ε, at least half of its arguments must be larger than ε, and all X i are independent.

Thus R = O(log 1/δ) in order to guarantee (1/3) R/2 < δ.

This is a standard technique.

Proof.

We apply the Taylor expansion from Theorem 1.

We directly apply the Chebyshev's inequality for X = ∆:

Proposition 6.

Suppose that a C 2 network is at a stationary point after training:

Proof.

Consider the quantity in question, use Additional Proposition 9 for it and apply E x on both sides:

Now since we know that E x ∂L ∂W = 0, the linear term is 0 408 Additional Proposition 7. (Linearity of error superposition for small p limit) Consider a network (L, W, B, ϕ) with crashes at each layer with probability p l , l ∈ 0, L. Then in the first order on p, the total mean or variance of the error at the last layer is equal to a vector sum of errors in case crashes were at a single layer

Consider each layer i having a probability of failure p i ∈ [0, 1] for i ∈ 0, L.

In this proof we utilize Assumption 1.

We write the definition of the expectation with

being the probability that a binary string of length n has a particular configuration with k ones, if its entries are i.i.d.

Bernoulli Be(p).

Here S l is the set of all possible network crash configurations at layer l.

Each configuration s l ∈ S l describes which neurons are crashed and which are working.

We have

We utilize the fact that the quantity

Only those sets of crashing neurons 411 (s 0 , ...., s L ) matter in the first order, which have one crash in total (in all layers).

We denote it as s

This is equivalent to a sum of individual layer crashes up to the first order on p:

The proof for the variance is analogous.

In case if we consider the RHS quantities in Theorem 1 averaged over all data examples (x, y * ), then the Algorithm 418 1 from the main paper would give a guarantee for every example: it will guarantee that P x,ξ [∆ ≥ E∆ + t] is small.

Indeed, if we know that E x Var∆ is small, we know that the total variance Var x,ξ ∆ = E x Var ξ ∆ + Var x E ξ ∆ (by the 420 law of total variance) is small as well.

Indeed, the second term Var x E ξ ∆ bounded by sup x E ξ ∆ which we assume to 421 be small.

In case if it is not small, it is unsafe to use the network, as even the expectation of the error is too high.

Given a small total variance Var x,ξ ∆, we apply, as in Proposition 5 a Chebyshev's inequality to the random variable ∆ over the joint probability distribution x, D|x.

This will give P x,ξ [∆ ≥ E∆ + t] ≤ t −2 Var x,ξ ∆. This probability

indicates how likely it is that a network with crashes and random inputs will encounter a too high error.

Median aggregation of R copies works for the input distribution as well.

Denote "y i (x) is bad" as the event

that the loss exceeds ε for the i'th copy.

We denote [T rue] = 1 and [F alse] = 0 (Iverson brackets).

Now,

Since the inner probability can be bounded as

Var∆(x) exp(−R), taking an expectation over E x results in the quantity discussed before, E x Var∆.

We note that it would not be possible to consider y L+1 to be the total loss, as it makes quantities such as

∂L/∂W ij (x, W ) ill-defined, as they depend on x as well.

The algorithm consists of the main loop which is executed until we can guarantee the desired (ε, δ)-fault tolerance.

It trains networks, and upon obtaining a good enough network with δ < 1/3, it repeats the network a logarithmic 434 number of times.

We note that the part on q and δ 0 is not strictly required to guarantee fault tolerance.

Rather,

satisfying these conditions is a natural necessary conditions to satisfy the more strict ones (on R 3 , E∆ and δ).

These conditions are necessary because, by AP5, continuous limit implies that the q is reasonable.

Space requirements.

After each iteration of the main loop of the algorithm, the previous network can be deleted

as it is no longer used.

Therefore, each new iteration does not require any additional space.

We only need to store changes that the weights make, which is > 1, and the loss is bounded by 1.

> 1/3 (see Section 5) for C l dependent on the function approximated by the NN and 461 the continuous limit.

Therefore, n ∼ O(C l p l /ε 2 ).

The total number of iterations is therefore O(D 12 + C l p l /ε 2 ) for C l = n l Var∆/p l for some n l (this is now a property 463 of the function being approximated and the continuous limit).

We note that the constants 1/3 and 10 −2 are chosen 464 for the simplicity of the proof.

The asymptotic behavior of the algorithm does not depend on them, as long as they 465 are constant.

6 For e : log e = 1 7 Technically, here we silently implied that e 2 p 1/ √ D 12 in order to make α ≥ e 2 p, which means that we cannot implement a function with too high second derivatives in neuromorphic hardware with a constant p, not matter how many neurons we take.

Intuitively, this happens because for such a function, even a failure of e 2 p fraction of neurons, which is a reasonable expectation, is too large to begin with.

We assume that we are fitting a function which is not like that.

If we encounter this, we will see that by having a too high E∆ and the algorithm will output infeasible Correctness: guarantee of robustness.

Now we analyze correctness.

We note that the algorithm is not guaranteed to find a good trade-off between accuracy and fault-tolerance.

Up to this point, there is no complete 468 theory explaining the generalization behavior of neural networks or their capacity.

Therefore, we cannot give a 469 proof for a sufficient trade-off without discovering first the complete properties of NNs capacity.

We only show that 470 the algorithm can achieve fault tolerance.

First, the condition on E∆ and Var∆ implies that the first-order terms in the expansion from T1 are small 472 enough.

Now we argue that the remainder is small as well.

The condition R 3 < C implies that discrete function is smooth enough to apply |W t (t, t )dtdt | ≈ C 1 < R 3 < C 474 as well as |W t (t, t )|dtdt ≈ C 2 < R 3 < C. This means that the integral is small, which allows to bound the 475 Riemann remainder R 3 /n l from the proof of AP4.

This implies that there exist a continuous network N N c such 476 that the approximation error A from AD2 is small.

Here, the right metric is R 3 /n l ≈ C/n l 1.

The number of 477 changes C must be less than a number of neurons at a layer n l . ) than the ground truth function.

Specifically, in the task of image recognition, we (humans) assume the problem to be quite smooth: a picture of a cat with a missing ear or whiskers is still a picture of a cat.

However, it was shown [13] that modern CNNs use non-robust features.

This implies that CNNs are much more sensitive to small changes in input x, in contrast to the smooth ground truth function y * that we want it to learn, making the bound for the derivative D 1 large.

For the hidden layers, it is possible that the continuous limit has large derivatives.

For example, we can first apply an injective transformation F with high derivatives, and then apply the inverse transform implemented as another neural network F −1 existing by AP3 a .

Then, we have an overall smooth (identity) operator y L which, however, consists of two very non-smooth parts.

We list the following approaches that potentially could resolve this issue theoretically rather than experimentally.

First, we can consider the infinite depth limit [23] .

This would allow to have regularities throughout the network' layers.

Another approach is to study mathematically the ways that an operator can be decomposed into a hierarchical composition of operators.

For example, for images, a natural decomposition of the image classification operator could first detect edges, then simple shapes, then groups them into elements pertinent to specific classes and then detects the most probable class [26] .

At each stage, only robust features are used.

Thus, for such a decomposition, output-hidden derivatives would be reasonable as well as the output-input ones: indeed, the decision to recognize a cat would not change significantly if some internal hidden layer features (ears or whiskers) are not present.

Interestingly, just enforcing the continuous limit seems to make features more robust in our experiments, see hidden layer weights on Figures 2 and 3 .

Without regularization, the weights seem noisy, there are a lot of unused neurons, and even the used ones contain noisy patterns.

In contrast, continuity-regularized networks seem to have first-layer weights similar to the input images.

This could be an interesting research direction to continue.

We note, however, that such a study is not connected to the fault tolerance anymore, as it is a fundamental investigation into the properties of the hierarchical functions that we want to approximate, and into the properties of the neural networks which we can find by gradient descent.

a The idea to construct y = F • F −1 with F −1 implementable by a neural network is taken from [8]

Experiments were performed on a single machine with 12 cores, 60GB of RAM and 2 NVIDIA GTX 1080 cards run-

For the Dropout experiment p < 0.03 is used as a threshold after which no visible change is happening.

We use 496 the unscaled version of dropout 8 .

In the experiments, we use computationally tractable evaluations of the result given by T1.

The "b1 bound" is 498 the Spectral bound from P2.

The "b2 bound" corresponds to AP2.

The "b3 bound" corresponds to first-order terms 499 from T1, or the Additional Corollary 2, and the "b4 bound" corresponds to an exact evaluation of single-neuron 500 crashes (taking O(n l ) forward passes).

We compare sigmoid networks with N ∼ 50 trained on Boston Housing dataset (see ErrorComparisonBoston.ipynb).

We use different inputs on a single network and single input on different networks.

We compare the bounds using 506 rank loss which is the average number of incorrectly ordered pairs.

The motivation is that even if the bound does 507 not predict the error exactly, it could be still useful if it is able to tell which network is more resilient to crashes.

The second quantity is the relative error of ∆ prediction, which is harder to obtain.

Experimental error computed Table 2 presents the results of ConvNetTest-ft.ipynb.

There are two stages.

In the first stage (red, iterations ≤ 4) the algorithm increases continuity of the network via increasing ψ.

In the second stage (green, iterations ≥ 5) the algorithm increases the number of neurons n 1 and the regularization parameter λ.

First, it can be seen that at first stages the network is not continuous enough, as the algorithm makes it more continuous.

This leads to an increase in the empirical probability δ of the network outputting a loss > ε and in the increase in the gap between Theoretical Var∆ and Experimental Var∆. This happens because there are not enough neurons in the network.

Later, as the number of neurons increases, the gap becomes smaller and the empirical probability decreases.

Note that the first network (at iteration 0) empirically satisfies our fault tolerance guarantee.

Nevertheless, we do not have a proof for such a network because it is not continuous enough.

Therefore, in order to guarantee robustness, we need to proceed with the iterations.

The result predicts the expected error E x E ξ ∂L ∂y ∆ to decay with the decay of the gradient of the loss.

We tested 536 that experimentally on the Boston dataset using sigmoid networks (see ErrorOnTraining.ipynb) and note that 537 a similar result holds for ReLU 9 .

The results are shown in Figure 12 .

The chart shows first that the experiment 538 corresponds to b4, and b3 is close to b4.

b3 is also equal to the result of AP9, both of which decay, as predicted.

The equality also holds for one particular y Proof.

Fix some layer l.

The output of the network y depends on the weight matrix W l and on the input to the 543 l'th layer y l−1 .

However we note that it only depends on their product and not on these quantities separately.

Differentiating the above expression gives a connection between second derivatives w.r.t weights and activations.

Additional Proposition 9.

For a neural network with C 1 activation function we have for a particular input x in the first order as in T1: Skip-connections.

If we consider a model with skip-connections (which do not fit Definition 1) with faults at every node, we expect that an assumption similar to A3 would lead to a result similar to T1.

However, we did not 560 test our theory on models with skip-connections.

Another idea for fault tolerance.

In case if we know p exactly, we can compensate for E∆ by multiplying 562 each neuron's output by (1 − p) −1 .

In this way, the mean input would be preserved, and the network output will 563 be unbiased.

However, this only works in case if we know exactly p of the hardware.

Additional Proposition 10 (Variance for bound b1, not useful since the bound is not tight.

Done in a similar manner as the mean bound in [9] ).

The variance of the error Var∆ is upper-bounded as

Proof.

In the notation of [9] , consider

is the corrupted output from layer L and K is the activation function

7.

P L = E(y −ŷξ) 2 = Ey

9.

Consider a recurrence x 1 = a 1 , x n = a n + b n x n−1 .

Then

10.

Define α L = max{(w (L+1) , w (L+1) )} and β L = max{ w

The goal of this proposition was to give an expression for the variance in a similar manner as it is done for the 588 mean in [9] .

However this proposition did not make it to the article because bound b1 was not showing any good 589 experimental results.

forming weights.

Thus, the computation is done (theoretically) at the speed of light, compared to many CPU/GPU 596 cycles.

The surge in performance could arguably even exceed the one that followed the switch from training on CPUs 597 to training on GPUs and TPUs [3] .

Recent results on neuromorphic computing report on concrete successes such as 598 milliwatt image recognition [10] or basic vowel recognition using only four coupled nano-oscillators [22] .

Since the 599 components of a neuromorphic network are small [25] and unreliable [17] , there are crashes in individual neurons 600 or weights inside the network [25, 17] .

They lead to a performance degradation.

A failure within a neuromorphic 601 architecture involved in a mission-critical application could be disastrous.

Hence, fault tolerance in neural networks 602 is an important concrete Artificial Intelligence (AI) safety problem [4] .

In terms of fault tolerance, the unit of failure 603 in these architectures is fine-grained, i.e., an individual neuron or a single synapse, with failure mode frequently 604 being a complete crash.

This is in contrast with the now classical case of a neural network as a software deployed 605 on a single machine where the unit of failure is coarse-grained, i.e., the whole machine holding the entire neural 606 network.

For instance, in the popular distributed setting of ML, the so-called parameter-server scheme [16] , the 607 unit of failure is a worker or a server, but never a neuron or a synapse.

Whilst very important, fine-grained fault 608 tolerance in neural networks has been overlooked as a concrete AI safety problem.

distribution.

Accessed: 2019-09-19.

[2] agronskiy.

Taking the expectation of taylor series (especially the remainder).

<|TLDR|>

@highlight

We give a bound for NNs on the output error in case of random weight failures using a Taylor expansion in the continuous limit where nearby neurons are similar

@highlight

This paper considers the problem of dropping neurons from a neural network, showing that if the goal is to become robust to randomly dropped neurons during evaluation, then it is sufficient to just train with dropout.

@highlight

This contribution studies the impact of deletions of random neurons on prediction accuracy of trained architecture, with the application to failure analysis and the specific context of neuromorphic hardware.