Despite their popularity and successes, deep neural networks are poorly understood theoretically and treated as 'black box' systems.

Using a functional view of these networks gives us a useful new lens with which to understand them.

This allows us us to theoretically or experimentally probe properties of these networks, including the effect of standard initializations, the value of depth, the underlying loss surface, and the origins of generalization.

One key result is that generalization results from smoothness of the functional approximation, combined with a flat initial approximation.

This smoothness increases with number of units, explaining why massively overparamaterized networks continue to generalize well.

Deep neural networks, trained via gradient descent, have revolutionized the field of machine learning.

Despite their widespread adoption, theoretical understanding of fundamental properties of deep learning -the true value of depth, the root cause of implicit regularization, and the seemingly 'unreasonable' generalization achieved by overparameterized networks -remains mysterious.

Empirically, it is known that depth is critical to the success of deep learning.

Theoretically, it has been proven that maximum expressivity grows exponentially with depth, with a smaller number of trainable parameters (Raghu et al., 2017; Poole et al., 2016) .

This theoretical capacity may not be used, as recently shown explicitly by (Hanin & Rolnick, 2019) .

Instead, the number of regions within a trained network is proportional to the total number of hidden units, regardless of depth.

Clearly deep networks perform better, but what is the value of depth if not in increasing expressivity?

Another major factor leading to the success and widespread adoption of deep learning has been its surprisingly high generalization performance (Zhang et al., 2016) .

In contrast to other machine learning techniques, continuing to add parameters to a deep network (beyond zero training loss) tends to improve generalization performance.

This is even for networks that are massively overparameterized, wherein according to traditional ML theory they should (over)fit all the training data (Neyshabur et al., 2015) .

How does training deep networks with excess capacity lead to generalization?

And how can it be that this generalization error decreases with overparameterization?

We believe that taking a functional view allows us a new, useful lens with which to explore and understand these issues.

In particular, we focus on shallow and deep fully connected univariate ReLU networks, whose parameters will always result in a Continuous Piecewise Linear (CPWL) approximation to the target function.

We provide theoretical results for shallow networks, with experiments showing that these qualitative results hold in deeper nets.

Our approach is related to previous work from (Savarese et al., 2019; Arora et al., 2019; Frankle & Carbin, 2018) in that we wish to characterize parameterization and generalization.

We differ from these other works by using small widths, rather than massively overparamaterized or infinite, and by using a functional parameterization to measure properties such as smoothness.

Other prior works such as (Serra et al., 2017; Arora et al., 2016; Montufar et al., 2014) attempt to provide theoretical upper or lower bounds to the number of induced pieces in ReLU networks, whereas we are more interested in the empirical number of pieces in example tasks.

Interestingly, (Serra et al., 2017) also takes a functional view, but is not interested in training and generalization as we are.

Previous work (Advani & Saxe, 2017) has hinted at the importance of small norm initialization, but the functional perspective allows us to prove generalization properties in shallow networks.

The main contribution of this work are as follows:

-Functional Perspective of Initialization: Increasingly Flat with Depth.

In the functional perspective, neural network parameters determine the locations of breakpoints and their delta-slopes (defined in Section 2.1) in the CPWL reparameterization.

We prove that, for common initializations, these distributions are mean 0 with low standard deviation.

The delta-slope distribution becomes increasingly concentrated as the depth of the network increases, leading to flatter approximations.

In contrast, the breakpoint distribution grows wider, allowing deeper network to better approximate over a broader range of inputs.

-Value of Depth: Optimization, not Expressivity.

Theoretically, depth adds an exponential amount of expressivity.

Empirically, this is not true in trained deep networks.

We find that expressivity scales with the number of total units, and weakly if at all with depth.

However, we find that depth makes it easier for GD to optimize the resulting network, allowing for a greater flexibility in the movement of breakpoints, as well as the number of breakpoints induced during training.

-Generalization is due to Flat Initialization in the Overparameterized Regime.

We find that generalization in overparametrized FC ReLu nets is due to three factors: (i) the very flat initialization, (ii) the curvature-based parametrization of the approximating function (breakpoints and deltaslopes) and (iii) the role of gradient descent (GD) in preserving (i) and regularizing via (ii).

In particular, the global, rather than local, impact of breakpoints and delta-slopes helps regularize the approximating function in the large gaps between training data, resulting in their smoothness.

Due to these nonlocal effects, more overparameterization leads to smoother approximations (all else equal), and thus typically better generalization (Neyshabur et al., 2018; .

Consider a fully connected ReLU neural netf θ (x) with a single hidden layer of width H, scalar input x ∈ R and scalar output y ∈ R.f (·; θ) is continuous piecewise linear function (CPWL) since the ReLU nonlinearity is CPWL.

We want to understand the function implemented by this neural net, and so we ask: How do the CPWL parameters relate to the NN parameters?

We answer this by transforming from the NN parametrization (weights and biases) to two CPWL parametrizations:

where the Iversen bracket b is 1 when the condition b is true, and 0 otherwise.

Here the NN parameters

denote the input weight, bias, and output weight of neuron i, and (·) + max{0, ·} denotes the ReLU function.

The first CPWL parametrization is

, where β i − bi wi is (the x-coordinate of) the breakpoint (or knot) induced by neuron i, µ i w i v i is the delta-slope contribution of neuron i, and s i sgn w i ∈ {±1} is the orientation of β i (left for s i = −1, right for s i = +1).

Intuitively, in a good fit the breakpoints β i will congregate in areas of high curvature in the ground truth function |f (x)| ≥ 0, while deltaslopes µ i will actually implement the needed curvature by changing the slope by µ i from one piece p(i) to the next p(i) + 1.

As the number of pieces grows, the approximation will improve, and the delta-slopes (scaled by the piece lengths) approach the true curvature of f :

We note that the BDSO parametrization of a ReLU NN is closely related to but different than a traditional roughness-minimizing m-th order spline parametrizationf spline (x)

BDSO (i) lacks the base polynomial, and (ii) it has two possible breakpoint orientations s i ∈ {±1} whereas the spline only has one.

We note in passing that adding in the base polynomial (for linear case m = 1) into the BDSO ReLU parametrization yields a ReLU ResNet parametrization.

We believe this is a novel viewpoint that may shed more light on the origin of the effectiveness of ResNets, but we leave it for future work.

The second parametrization is the canonical one for PWL functions:

< . . .

< β P is the sorted list of (the x-coordinates of) the P H + 1 breakpoints (or knots), m p , γ p are the slope and y-intercept of piece p.

Computing the analogous reparametrization to function space for deep networks is more involved, so we present a basic overview here, and a more detailed treatment in Appendix B. For L ≥ 2 layers with widths H ( ) , the neural network's activations are defined as: z

for all hidden layers ∈ {1, 2, . . .

, L} and for all neurons

i is a breakpoint induced by neuron i in layer if it is a zero-crossing of the net input i.e. z ( )

Considering these parameterizations (especially the BDSO parameterization) provides a new, useful lens with which to analyze neural nets, enabling us to reason more easily and transparently about the initialization, loss surface, and training dynamics.

The benefits of this approach derive from two main properties: (1) that we have 'modded out' the degeneracies in the NN parameterization and (2) the loss depends on the NN parameters θ N N only through the BDSO parameters (the approximating function) θ BDSO i.e. (θ N N ) = (θ BDSO (θ N N )), analogous to the concept of a minimum sufficient statistic in exponential family models.

Much recent related work has also veered in this direction, analyzing function space (Hanin & Rolnick, 2019; Balestriero et al., 2018) .

We now study the random initializations commonly used in deep learning in function space.

These include the independent Gaussian initialization, with

, and independent Uniform initialization, with

We find that common initializations result in flat functions, becoming flatter with increasing depth.

Theorem 1.

Consider a fully connected ReLU neural net with scalar input and output, and a single hidden layer of width H. Let the weights and biases be initialized randomly according to a zero-mean Gaussian or Uniform distribution.

Then the induced distributions of the function space parameters (breakpoints β, delta-slopes µ) are as follows:

(a) Under an independent Gaussian initialization,

Using this result, we can immediately derive marginal and conditional distributions for the breakpoints and delta-slopes.

Corollary 1.

Consider the same setting as Theorem 1.

(a) In the case of an independent Gaussian initialization,

where G nm pq (·|·) is the Meijer G-function and K ν (·) is the modified Bessel function of the second kind.

(b) In the case of an independent Uniform initialization,

where Tri(·; a) is the symmetric triangular distribution with base [−a, a] and mode 0.

Implications.

Corollary 1 implies that the breakpoint density drops quickly away from the origin for common initializations.

If f has significant curvature far from the origin, then it may be far more difficult to fit.

We show that this is indeed the case by training a shallow ReLU NN with an initialization that does not match the underlying curvature, with training becoming easier if the initial breakpoint distribution better matches the function curvature.

We also show that during training, breakpoint distributions move to better match the underlying function curvature, and that this effect increases with depth (see Section 3, Table 1 , and Appendix A.6).

This implies that a data-dependent initialization, with a breakpoint distribution near areas of high curvature, could potentially be faster and easier to train.

Next, we consider the typical Gaussian He (He et al., 2015) or Glorot (Glorot & Bengio) initializations.

In the He initialization, we have σ w = √ 2, σ v = 2/H. In the Glorot initalization, we have σ w = σ v = 2/(H + 1).

We wish to consider their effect on the smoothness of the initial function approximation.

From here on, we measure the smoothness using a roughness metric, defined as ρ i µ 2 i , where lower roughness indicates a smoother approximation.

Theorem 2.

Consider the initial roughness ρ 0 under a Gaussian initialization.

In the He initialization, we have that the tail probability is given by

, where E[ρ 0 ] = 4.

In the Glorot initialization, we have that the tail probability is given by

Thus, as the width H increases, the distribution of the roughness of the initial functionf 0 gets tighter around its mean.

In the case of the He initialization, this mean is constant; in the Glorot initialization, it decreases with H. In either case, for reasonable widths, the initial roughness is small with high probability.

This smoothness has implications for the implicit regularization/generalization phenomenon observed in recent work (Neyshabur et al., 2018 ) (see Section 3 for generalization/smoothness analysis during training).

Work.

Several recent works analyze the random initialization in deep networks.

However, there are two main differences, First, they focus on the infinite width case (Savarese et al., 2019; Jacot et al., 2018; Lee et al., 2017) and can thus use the Central Limit Theorem (CLT), whereas we focus on finite width case and cannot use the CLT, thus requiring nontrivial mathematical machinery (see Supplement for detailed proofs).

Second, they focus on the activations as a function of input whereas we also compute the joint densities of the BDSO parameters i.e. breakpoints and deltaslopes.

The latter is particularly important for understanding the non-uniform density of breakpoints away from the origin as noted above.

We now consider the mean squared error (MSE) loss as a function of either the NN parameters

such that the restriction off BDSO to any piece of this partition, denotedf (·; θ BDSO )| πp , is a linear function.

An open question is how many such critical points exist.

A starting point is to consider that there are C(N +H, H) (N +H)!/N !

H! possible partitions of the data.

Not every such partition will admit a piecewise-OLS solution which is also continuous, and it is difficult to analytically characterize such solutions, so we resort to simulation and find a lower bound that suggests the number of critical points grows at least polynomially in N and H (Figure 7 ).

Using Theorem 3, we can characterize growth of global minima in the overparameterized case.

Call a partition Π lonely if each piece π p contains at most one datapoint.

Then, we can prove the following results: Theorem 4.

For any lonely partition Π, there are infinitely many parameter settings θ BDSO that induce Π and are global minima with˜ (θ BDSO ) = 0.

Proof.

Note that each linear piece p has two degrees of freedom (slope and intercept).

By way of induction, start at (say) the left-most piece.

If there is a datapoint in this piece, choose an arbitrary slope and intercept that goes through it; otherwise, choose an arbitrary slope and intercept.

At each subsequent piece, we can use one degree of freedom to ensure continuity with the previous piece, and use one degree of freedom to match the data (if there is any).

Remark 1.

Suppose that the H breakpoints are uniformly spaced and that the N data points are uniformly distributed within the region of breakpoints.

Then in the overparametrized regime H ≥ αN 2 for some constant α > 1, the induced partition Π is lonely with high probabilility 1 − e −N 2 /(H+1) = 1 − e −1/α .

Furthermore, the total number of lonely partitions, and thus a lower bound on the total number of global minima of˜ is

Thus, with only order N 2 units, we can almost guarantee lonely partitions, where the piecewise OLS solution on these lonely paratitions will be the global optimum.

Note how simple and transparent the function space explanation is for why overparametrization makes optimization easy, as compared to the weight space explanation (Arora et al., 2019) , requiring order N 7 units.

The above sections argue that overparameterization leads to a flatter initial function approximation, and an easier time reaching a global minima over the training data.

However, neural networks also exhibit unreasonably high generalization performance, which must be due to implicit regularization, since the effect is independent of loss function.

Here we provide an argument that overparameterization directly leads to this implicit regularization, due to the increasing flatness of the initialization and the non-locality of the delta-slope parameters.

Consider a dataset like that shown in Figure 8 with a data gap between regions of two continuous functions f L , f R and consider a breakpoint i with orientation s i in the gap.

Starting with a flat initialization, the dynamics of the i-th delta-slope areμ

where r 2,s (t), r 3,s (t) are the (negative) net correlation and residual on the active side of i, in this case including data from the function f si but not f −si .

Note that the both terms of the gradientμ i have a weak dependence on i through the orientation s i , and the second term additionally depends on i through β i (t).

Thus the vector of delta-slopes with orientation s evolves according toμ s = r 2,s (t)1 + r 3,s (t)β s .

Now consider the regime of overparametrization H N .

It will turn out to be identical to taking a continuum limit

f (x, t), the curvature of the approximation (the discrete index i has become a continuous index x) andβ i (t) → 0 (following from Theorem 5, multiplyingβ i (t) by v i (t)/w i (t) and factoring out µ i (t) → 0).

Integrating the dynamicsμ s (x, t) = r 2,s (t) + r 3,s (t)x over all time yields µ s (x, t = ∞) = µ s (x, t = 0) + R * 2,s + R * 3,s x, where the curvature µ s (x, t = 0) ≈ 0 (Section 3) and R * j,s ∞ 0 dt r j,s (t ) < ∞ (convergence of residuals n (t) and immobility of breakpointsβ i (t) = 0 implies convergence of r j,s (t)).

Integrating over space twice(from x = ξ s to x = x) yields a cubic splinef (x, t) = c 0,s + c 1,

where c 0,s , c 1,s are integration constants determined by the per-piece boundary conditions (PBCs)

, thus matching the 0-th and 1st derivatives at the gap endpoints.

The other two coefficients c k,s R * k,s , k ∈ {2, 3} and serve to match the 2nd and 3rd derivatives at the gap endpoints.

Clearly, matching the training data only requires the two parameters c 0,s , c 1,s ; and yet, surprisingly, two unexpected parameters c 2,s , c 3,s emerge that endowf with smoothness in the data gap, despite the loss function not possessing any explicit regularization term.

Tracing back to find the origin of these smoothness-inducing terms, we see that they emerge as a consequence of (i) the smoothness of the initial function and (ii) the active half space structure, which in turn arises due to the discrete curvature-based (delta-slope) parameterization.

Stepping back, the ReLU net parameterization is a discretization of this underlying continuous 2nd-order ordinary differential equation.

In Section 3 we conduct experiments to test this theory.

Breaking Bad: Breakpoint densities that are mismatched to function curvature makes optimization difficult We first test our initialization theory against real networks.

We initialize fullyconnected ReLU networks of varying depths, according to the popular He initializations (He et al., 2015) .

Figure 1 shows experimentally measured densities of breakpoints and delta-slopes.

Our theory matches the experiments well.

The main points to note are that: (i) breakpoints are indeed more highly concentrated around the origin, and that (ii) as depth increases, delta-slopes have lower variance and thus lead to even flatter initial functions.

We next ask whether the standard initializations will experience difficulty fitting functions that have significant curvature away from the origin (e.g. learning the energy function of a protein molecule).

We train ReLU networks to fit a periodic function (sin(x)), which has high curvature both at and far from the origin.

We find that the standard initializations do quite poorly away from the origin, consistent with our theory that breakpoints are essential for modeling curvature.

Probing further, we observe empirically that breakpoints cannot migrate very far from their initial location, even if there are plenty of breakpoints overall, leading to highly suboptimal fits.

We additionally show (see Appendix A.6 for details) that breakpoint distributions change throughout training to more accurately match the ground truth curvature.

In order to prove that it is indeed the breakpoint density that is causally responsible, we attempt to rescue the poor fitting by using a simple data-dependent initialization that samples breakpoints uniformly over the training data range [x min , x max ], achieved by exploiting Eq. (2).

Sine Quadratic Standard 4.096 ± 2.25 .1032 ± 0404 Uniform 2.280 ± .457 .1118 ±

.0248 We train shallow ReLU networks on training data sampled from a sine and a quadratic function, two extremes on the spectrum of curvature.

The data shows that uniform breakpoint density rescues bad fits in cases with significant curvature far from the origin, with less effect on other cases, confirming the theory.

We note that this could be a potentially useful data-dependent initialization strategy, one that can scale to high dimensions, but we leave this for future work.

Explaining and Quantifying the Suboptimality of Gradient Descent.

The suboptimality seen above begs a larger question: under what conditions will GD be successful?

Empirically, it has been observed that neural nets must be massively overparameterized (relative to the number of parameters needed to express the underlying function), in order to ensure good training performance.

Our theory provides a possible explanation for this phenomenon: if GD cannot move breakpoints too far from their starting point, then one natural strategy is to sample as many breakpoints as possible everywhere, allowing us to fit an arbitrary f .

The downside of this strategy is that many breakpoints will add little value.

In order to test this explanation and, more generally, understand the root causes 55.5 ± 2.9 52 ± 1.414 50 ± .7 49.25 ± 3.3 51.25 ± 6.1 49.25 ± 4.5 4 68 ± 3.1 57.25 ± 6.8 48.5 ± 2.5 42.5 ± 4.8 40.25 ± 3.9 40.25 ± 3.3 5 62.25 ± 15.1 49 ± 3.5 44.5 ± 5.1 38 ± 5.1 33.75 ± 1.1 31.5 ± 1.7 of the GD's difficulty, we focus on the case of a fully connected shallow ReLU network.

A univariate input (i) enables us to use our theory, (ii) allows for visualization of the entire learning trajectory, and (iii) enables direct comparison with existing globally (near-)optimal algorithms for fitting PWL functions.

The latter include the Dynamic Programming algorithm (DP, (Bai & Perron, 1998) ), and a very fast greedy approximation known as Greedy Merge (GM, (Acharya et al., 2016) ).

How do these algorithms compare to GD, across different target function classes, in terms of training loss, and the number of pieces/hidden units?

We use this metric for the neural network as well, rather than the total number of trainable parameters.

Taking the functional approximation view allows us to directly compare neural network performance to these PWL approximation algorithms.

For a quadratic function (e.g. with high curvature, requiring many pieces), we find that the globally optimal DP algorithm can quickly reduce training error to near 0 with order 100 pieces.

The GM algorithm, a relaxation of the DP algorithm, requires slightly higher pieces, but requires significantly less computational power.

On the other hand all variants of GD (vanilla, Adam, SGD w/ BatchNorm) all require far more pieces to reduce error below a target threshold, and may not even monotonically decrease error with number of pieces.

Interestingly, we observe a strict ordering of optimization quality with Adam outperforming BatchNorm SGD outperforming Vanilla GD.

These results (Figure 1) show how inefficient GD is with respect to (functional) parameters, requiring orders of magnitude more for similar performance to exact or approximate PWL fitting algorithms.

Learned Expressivity is not Exponential in Depth.

In the previous experiment, we counted the number of linear pieces in the CPWL approximation as the number of parameters, rather than the number of weights.

Empirically, we know that the greatest successes have come from deep learning.

This raises the question: how does the depth of a network affect its expressivity (as measured in the number of pieces)?

Theoretically, it is well known that maximum expressivity increases exponentially with depth, which, in a deep ReLU neural network, means an exponential increase in the number of linear pieces in the CPWL approximation.

Thus, theoretically the main power of depth is that it allows for more powerful function approximation relative to a fixed budget of parameters compared to a shallow network.

However, recent work (Hanin & Rolnick, 2019) has called this into question, finding that in realistic networks expressivity does not scale exponentially with depth.

We perform a similar experiment here, asking how the number of pieces in the CPWL function approximation of a deep ReLU network varies with depth.

The results in Table 2 clearly show that the number of pieces does not exponentially scale with depth.

In fact, we find that depth only has a weak effect overall, although more study is needed to determine exactly what effect depth has on the number and variability of pieces.

These results lend more support to the recent findings of (Hanin & Rolnick, 2019) From the functional approximation, we know that a unit induces breakpoints only when the ReLU function applied to the unit's input has zero crossings.

In layer one, this happens exactly once per unit as the input to each ReLU is just a line over the input space.

In deeper layers, the function approximation is learned, allowing for a varying number of new breakpoints.

Given our previous results on the flatness of the standard initializations, this will generally only happen once per unit, implying that the number of pieces will strongly correlate with the number of units at initialization.

Depth helps with Optimization by enabling the Creation, Annihilation and Mobility of Breakpoints.

If depth does not strongly increase expressivity, then it is natural to ask whether its value lies with optimization.

In order to test this, we examine how the CPWL function approximation develops in each layer during learning, and how it depends on the target function.

A good fit requires that breakpoints accumulate at areas of higher curvature in the training data, as these regions require more pieces.

We argue that the deeper layers of a network help with this optimization procedure, allowing the breakpoints more mobility as well as the power to create and annihilate breakpoints.

One key difference between the deeper layers of a network and the first layer is the ability for a single unit to induce multiple breakpoints.

As these units' inputs change during learning, the number of breakpoints induced by deeper units in a network can vary, allowing for another degree of freedom for the network to optimize.

Through the functional parameterization of the hidden layers, these "births and deaths" of breakpoints can be tracked as changes in the number of breakpoints induced per layer.

Another possible explanation for the value added of depth is breakpoint mobility, or that breakpoints in deeper layers can move more than those in shallow layers.

We run experiments comparing how the velocity and number of induced breakpoints varies between layers of a deeper network.

Figure 2: Total changes in number of breakpoints induced and average velocity of breakpoints relative to the first layer in each layer of a five layer ReLU network Figure 2 shows the results.

The number of breakpoints in deeper layers changes more often than in shallow layers.

The breakpoint velocity in deeper layers is also higher than the first layer, although not monotonically increasing.

Both of these results provide support for the idea that later layers help significantly with optimization and breakpoint placement, even if they do not help as strongly with expressivity.

Note that breakpoints induced by a layer of the network are present in the basis functions of all deeper layers.

Their functional approximations thus become more complex with depth.

However the roughness of the basis functions at initialization in the deeper layers is lower than that of the shallow layers.

But, as the network learns, for complex functions most of the roughness is in the later layers as seen in Figure 3 (right). , 2018; 2015) has argued that it comes from an implicit regularization inherent in the optimization algorithm itself (i.e. SGD).

In contrast, for the case of shallow and deep univariate fully connected ReLU nets, we provide causal evidence that it is due to the specific, very flat CPWL initialization induced by common initialization methods.

In order to test this in both shallow and deep ReLU networks, we compare training with the standard flat initialization to a 'spiky' initialization.

For a shallow ReLU network, we can test a 'spiky' initialization by exactly solving for network parameters to generate a given arbitrary CPWL function.

This network initialization is then compared against a standard initialization, and trained against a smooth function with a small number of training data points.

Note that in a 1D input space we need a small number of training data points to create a situation similar to that of the sparsity caused by high dimensional input, and to allow for testing generalization between data points.

We find that both networks fit the training data near perfectly, reaching a global minima of the training loss, but that the 'spiky' initialization has much worse generalization error (Table 3) .

Visually, we find that the initial 'spiky' features of the starting point CPWL representation are preserved in the final approximation of the smooth target function (Figures 4 and 6 ).

For a deep ReLU network, it is more difficult to exactly solve for a 'spiky' initialization.

Instead, we train a network to approximate an arbitrary CPWL function, and call those trained network parameters the 'spiky' initialization.

Once again, the 'spiky' initialization has near identical training performance, hitting all data points, but has noticeably worse generalization performance.

Figure 4: 'Spiky' (orange) and standard initialization (blue), compared before (left) and after (right) training.

Note both cases had similar, very low training set error.

It appears that generalization performance it not automatically guaranteed by GD, but instead due to the flat initializations which are then preserved by GD. '

Spiky' initializations also have their (higher) curvature preserved by GD.

This idea makes sense, as generalization depends on our target function smoothly varying, and a smooth approximation is promoted by a smooth initialization.

Variance.

Our last experiment examines how smoothness (roughness) depends on the number of units, particularly in the case where there are large gaps in the training data.

We use a continuous and discontinuous target function (shown in Figure 8 ).

We trained shallow ReLU networks with varying width H and initial weight variance σ w on these training data until convergence, and measured the total roughness of resulting CPWL approximation in the data gaps.

Figure 5: Roughness vs. Width (left) and the variance of the initialization (right) for both data gap cases shown in Figure 8 .

Each data point is the result of averaging over 4 trials trained to convergence.

Figure 5 shows that roughness in the data gaps decreases with width and increases with initial weight variance, confirming our theory.

A spiky (and thus rougher) initialization leads to increased roughness at convergence as well, lending support to the idea that roughness in data gaps can be 'remembered' from initialization.

On the other hand, higher number of pieces spreads out the curvature work over more units, leading to smaller overall roughness.

Taken together, our experiments indicate that smooth, flat initialization is partly (if not wholly) responsible for the phenomenon of implicit regularization in univariate fully connected ReLU nets, and that increasing overparameterization leads to even better generalization.

Conclusions.

We show in this paper that examining deep networks through the lens of function space can enabled new theoretical and practical insights.

We have several interesting findings: the value of depth in deep nets seems to be less about expressivity and more about learnability, enabling GD to finding better quality solutions.

The functional view also highlights the importance initialization: a smooth initial approximation seems to encourage a smoother final solution, improving generalization.

Fortunately, existing initializations used in practice start with smooth initial approximations, with smoothness increasing with depth.

Analyzing the loss surface for a ReLU net in function space gives us a surprisingly simple and transparent view of the phenomenon of overparameterization: it makes clear that increasing width relative to training data size leads w.h.p.

to lonely partitions of the data which are global minima.

Function space shows us that the mysterious phenomenon of implicit regularization may arise due to a hidden 2nd order differential equation that underlies the ReLU parameterization.

In addition, this functional lens suggests new tools, architectures and algorithms.

Can we develop tools to help understand how these CPWL functions change across layers or during training?

Finally, our analysis shows that bad local minima are often due to breakpoints getting trapped in bad local minima: Can we design new learning algorithms that make global moves in the BDSO parameterization in order to avoid these local minima?

A EXPERIMENTAL DETAILS Figure 6 : 'Spiky' (orange) and standard initialization (blue), compared before training (left) and post-training (right) using a deep network

Trained on a deep, 5 layer network, with 4 hidden layers of width 8.

Trained on function over the interval [-2,2] .

Learning rate = 1e-4, trained via GD over 10000 epochs, with roughness measured every 50 epochs.

Roughness per layer was summed over all units within that layer.

Shallow version trained on a 21 unit FC ReLU Network.

Deep version trained on a deep, 5-layer network with 4 hidden layers of width 8.

In both cases, the 'spiky' initialization was a 20 -breakpoint CPWL function, with y n ∼ Uniform([−2, 2]).

In the deep case, the spiky model was initialized with the same weights as the non-spiky model, and then pre-trained for 10,000 epochs to fit the CPWL.

After that, gradient descent training proceeded on both models for 20,000 epochs, with all training having learning rate 1e-4.

Training data was 20 random points in the range [-2,2] , while the testing data (used to measure generalization) was spaced uniformly at every ∆x = .01 of the target interval of the target function.

In the shallow case, there was no pre-training, as the 'spiky' model was directly set to be equal to the CPWL.

In the shallow model, training occurred for 20,000 epochs.

All experiment were run over 5 trials, and values in table are reported as mean ± standard deviation.

Base shallow learning rate was 1e-4 using gradient descent method, with learning rate divided by 5 for the spiky case due to the initialization method generating larger weights.

Despite differing learning rates, both models had similar training loss curves and similar final training loss values, e.g. for sine, final training loss was .94 for spiky and 1.02 for standard.

Functions used were sin(x), arctan(x), a sawtooth function from [-2,2] with minimum value of -1 at the endpoints, and 4 peaks of maximum value 1, cubic

2 , and exp(.5x) Note GD was chosen due to the strong theoretical focus of this paper -similar results were obtained using ADAM optimizer, in which case no differing learning rates were necessary.

We used networks with a total of H = 40 hidden units, spread over L ∈ {1, 2, 3, 4, 5} hidden layers.

Training data consiste of uniform samples of function over the interval x ∈ [−3, 3].

Learning rate = 5 · 10 −5 , trained via GD over 25, 000 epochs.

The target functions tested were sin(πx), a 5-piece polynomial with maximum value of 2 in the domain [−3, 3] , a sawtooth with period 3 and amplitude 1, arctan(x), exp(x), and 1 9 x 2 .

Each value in the table was the average of 5 trials.

We use a deep, 6-layer network, with 5 hidden layers of width 8.

Training data consists of the 'smooth' and 'sharp' functions over the interval x ∈ [−3, 3].

Learning rate = 5e-5, trained via GD until convergence, where convergence was defined as when the loss between two epochs changed by less than 10 −8 .

Breakpoints were calculated every 50 epochs.

The velocity of breakpoints was then calculated, and the values seen in the figure are normalized to the velocity of the first layer.

Various function classes were trained until convergence on a depth 1 or 4 ReLU network, with 500 total units distributed evenly across layers.

Initial and final breakpoint distributions were measured using a kernel density estimate, and compared with the underlying curvature (absolute value of 2nd derivative) of the ground truth function.

The cubic spline was a cubic spline fit to a small number of arbitrary data points.

Table 4 shows that the breakpoint densities moved over training to become more correlated with the underlying curvature of the ground truth function.

This effect was more pronounced with depth.

In certain very simple functions (e.g. x 2 or exp(x), not shown), a failure case emerged where there was no real change in correlation over training.

Diagnostics appeared to show this was due to the function being so simple as to train almost instantaneously in our overparameterized network, meaning breakpoints had no time to move.

Figure 9 shows what happens to the breakpoint densities Table 4 : Top: Correlation of the BP distribution before and after training for depth 1 and 4 networks across function classes.

Bottom : Change in correlation over training over training -in the shallow case, they are more constrained by the initial condition, and continue to have a higher density near the origin even when not necessary or appropriate.

Each neuron of the second hidden layer receives as input the result of a CPWL function z

i (x) as defined above.

The output of this function is then fed through a ReLU, which has two implications: first, every zero crossing of z (2) i is a breakpoint of x (2) i ; second, any breakpoints β

j ) < 0 will not be breakpoints of x (2) i .

Importantly, the number of breakpoints in g θ (x) is now a function of the parameters θ, rather than equal to fixed H as in the L = 1 case; in other words, breakpoints can be dynamically created and annihilated throughout training.

This fact will have dramatic implications when we explore how gradient descent optimizes breakpoints in order to model curvature in the training data (see Section 3).

But first, due to complexities of depth, we must carefully formalize the notion of a breakpoint for a deep network.

Let a π (x) = i∈π a i .

Then, β ( ) i is active iff there exists some path π such that a π is discontinuous at (2) and (3): 2 ), and a cubic spline of a few arbitrary data points This gives us Equation (2), as desired.

Let the subscripts p, q denote the parameters sorted by β p value.

In this setting, let β 0 −∞, and β H+1 ∞. Then,

This gives us Equation (3), as desired.

Lemma 1.

Suppose (b i , w i , v i ) are initialized independently with densities f B (b i ), f W (w i ), and f V (v i ).

Then, the density of (β i , µ i ) is given by

Then, we can derive the density of (β i , µ i ) by considering the invertable continuous transformation given by

where J is the Jacobian determinant of g −1 .

Then, we have J = − sgn w i and |J| = 1.

The density of (β i , µ i ) is then derived by integrating out the dummy variable u:

are independent, this expands to

Theorem 1(a).

Consider a fully connected ReLU neural net with scalar input and output, and a single hidden layer of width H. Let the weights and biases be initialized randomly according to a zero-mean Gaussian or Uniform distribution.

Then, under an independent Gaussian initialization,

Proof.

Starting with Lemma 1,

unknown otherwise but the integrand is even in µ, giving

Corollary 1(a).

Consider the same setting as Theorem 1.

In the case of an independent Gaussian initialization,

where G nm pq (·|·) is the Meijer G-function and K ν (·) is the modified Bessel function of the second kind.

Proof.

Marginalizing out µ from the joint density in Sympy returns the desired f β (β) from above.

Sympy cannot compute the other marginal, so we verify it by hand: Gradshteyn & Ryzhik (2015) , Eq. 3.462.20, we have

applying this with a = |µ| σ b σvσw and b = σ b ,

We can then use these densities to derive the conditional:

Theorem 1(b).

Consider a fully connected ReLU neural net with scalar input and output, and a single hidden layer of width H. Let the weights and biases be initialized randomly according to a zero-mean Gaussian or Uniform distribution.

Then, under an independent Uniform initialization,

Proof.

Starting with Lemma 1,

Corollary 1(b).

Consider the same setting as Theorem 1.

In the case of an independent Uniform initialization,

where Tri(·; a) is the symmetric triangular distribution with base [−a, a] and mode 0.

Proof.

Beginning with the marginal of β i , Remarks.

Note that the marginal distribution on µ i is the distribution of a product of two independent random variables, and the marginal distribution on β i is the distribution of the ratio of two random variables.

For the Gaussian case, the marginal distribution on µ i is a symmetric distribution with variance σ Proof.

Computing the time derivatives of the BDSO parameters and using the loss gradients of the loss with respect to the NN parameters gives us:

This completes the proof.

<|TLDR|>

@highlight

A functional approach reveals that flat initialization, preserved by gradient descent, leads to generalization ability.