A recent line of work has studied the statistical properties of neural networks to great success from a {\it mean field theory} perspective, making and verifying very precise predictions of neural network behavior and test time performance.

In this paper, we build upon these works to explore two methods for taming the behaviors of random residual networks (with only fully connected layers and no batchnorm).

The first method is {\it width variation (WV)}, i.e. varying the widths of layers as a function of depth.

We show that width decay reduces gradient explosion without affecting the mean forward dynamics of the random network.

The second method is {\it variance variation (VV)}, i.e. changing the initialization variances of weights and biases over depth.

We show VV, used appropriately, can reduce gradient explosion of tanh and ReLU resnets from $\exp(\Theta(\sqrt L))$ and $\exp(\Theta(L))$ respectively to constant $\Theta(1)$.

A complete phase-diagram is derived for how variance decay affects different dynamics, such as those of gradient and activation norms.

In particular, we show the existence of many phase transitions where these dynamics switch between exponential, polynomial, logarithmic, and even constant behaviors.

Using the obtained mean field theory, we are able to track surprisingly well how VV at initialization time affects training and test time performance on MNIST after a set number of epochs: the level sets of test/train set accuracies coincide with the level sets of the expectations of certain gradient norms or of metric expressivity (as defined in \cite{yang_meanfield_2017}), a measure of expansion in a random neural network.

Based on insights from past works in deep mean field theory and information geometry, we also provide a new perspective on the gradient explosion/vanishing problems: they lead to ill-conditioning of the Fisher information matrix, causing optimization troubles.

Deep mean field theory studies how random neural networks behave with increasing depth, as the width goes to infinity.

In this limit, several pieces of seminal work used statistical physics BID7 Sompolinsky et al., 1988) and Gaussian Processes (Neal, 2012) to show that neural networks exhibit remarkable regularity.

Mean field theory also has a substantial history studying Boltzmann machines BID0 and sigmoid belief networks (Saul et al., 1996) .Recently, a number of results have revitalized the use of mean field theory in deep learning, with a focus on addressing practical design questions.

In Poole et al. (2016) , mean field theory is combined with Riemannian geometry to quantify the expressivity of random neural networks.

In Schoenholz et al. (2017) and Yang and Schoenholz (2017) , a study of the critical phenomena of mean field neural networks and residual networks 1 is leveraged to theoretically predict test time relative performance of differential initialization schemes.

Additionally, BID5 and Pennington and Bahri (2017) have used related techniques to investigate properties of the loss landscape of deep networks.

Together these results have helped a large number of experimental observations onto more rigorous footing (Montfar et al., 2014; BID9 BID3 .

Finally, deep mean field theory has proven to be a necessary underpinning for studies using random matrix theory to 1 without batchnorm and with only fully connected layers understand dynamical isometry in random neural networks (Pennington et al., 2017; Pennington and Worah, 2017) .

Overall, a program is emerging toward building a mean field theory for state-of-the-art neural architectures as used in the wild, so as to provide optimal initialization parameters quickly for any deep learning practitioner.

In this paper, we contribute to this program by studying how width variation (WV), as practiced commonly, can change the behavior of quantities mentioned above, with gradient norm being of central concern.

We find that WV can dramatically reduce gradient explosion without affecting the mean dynamics of forward computation, such as the activation norms, although possibly increasing deviation from the mean in the process (Section 6).We also study a second method, variance variation (VV), for manipulating the mean field dynamics of a random neural network (Section 7 and Appendix B).

In this paper, we focus on its application to tanh and ReLU residual networks, where we show that VV can dramatically ameliorate gradient explosion, and in the case of ReLU resnet, activation explosion 2 .

Affirming the results of Yang and Schoenholz (2017) and predicted by our theory, VV improves performances of tanh and ReLU resnets through these means.

Previous works (Poole et al., 2016; Schoenholz et al., 2017; Yang and Schoenholz, 2017) have focused on how network architecture and activation functions affect the dynamics of mean field quantities, subject to the constraint that initialization variances and widths are constant across layers.

In each combination of (architecture, activation), the mean field dynamics have the same kinds of asymptotics regardless of the variances.

For example, tanh feedforward networks have exp(Θ(l)) forward and backward dynamics, while tanh residual networks have poly(l) forward and exp(Θ( √ l)) backward dynamics.

Such asymptotics were considered characteristics of the (architecture, activation) combination (Yang and Schoenholz, 2017) .

We show by counterexample that this perception is erroneous.

In fact, as discussed above, WV can control the gradient dynamics arbitrarily and VV can control forward and backward dynamics jointly, all without changing the network architecture or activation.

To the best of our knowledge, this is the first time methods for reducing gradient explosion or vanishing have been proposed that vary initialization variance and/or width across layers.

With regard to ReLU resnets, we find that gradient norms and "metric expressivity" (as introduced in Yang and Schoenholz (2017) , also defined in Defn 4.2), make surprisingly good predictors, respectively in two separate phases, of how VV at initialization affects performance after a fixed amount of training time (Section 7.1).

However, in one of these phases, larger gradient explosion seems to cause better performance, with no alternative course of explanation.

In this paper we have no answer for why this occurs but hope to elucidate it for future work.

With regard to tanh resnets, we find that, just as in Yang and Schoenholz (2017) , the optimal initialization balances trainability and expressivity: Decaying the variance too little means we suffer from gradient explosion, but decaying the variance too much means we suffer from not enough metric expressivity.

We want to stress that in this work, by "performance" we do not mean absolute performance but rather relative performance between different initialization schemes.

For example, we do not claim to know what initialization scheme is needed to make a particular neural network architecture solve ImageNet, but rather, conditioned on the architecture, whether one initialization is better than another in terms of test set accuracy after the same amount of training iterations.

Before we begin the mean field analysis, we present a perspective on gradient explosion/vanishing problem from a combination of mean field theory and information geometry, which posits that such problem manifests in the ill-conditioning of the Fisher information matrix.

Given a parametric family of probability distributions on R m , P := {P θ } θ with θ = (θ 1 , . . .

, θ n ) in R n , its Fisher information matrix is defined as F (θ) := [E z∼P θ (∂ i log P θ (z))(∂ j log P θ (z))] DISPLAYFORM0 (here ∂ i is partial derivative against θ i ).

It is known from information geometry that, under regularity conditions, P forms a Riemannian manifold with θ → P θ as its coordinate map and F (θ) as its Riemannian metric tensor (a fortiori it is positive definite) BID2 .

This fact is most famously used in the natural gradient method, which, akin to second order methods, computes from a gradient vector ∂E/∂θ a "natural direction of greatest descent" F (θ) −1 ∂E/∂θ that is invariant to reparametrization θ → θ BID1 .

This method and related ideas have been applied to great success in supervised, unsupervised, and reinforcement learning (for example, Pascanu and Bengio (2013); BID8 ; Martens and Grosse (2015) ; BID10 ; Wu et al. (2017) ).

An F (θ) with eigenvalues all approximately equal means that the neighborhood around P θ is isotropically curved and the gradient is approximately just the natural gradient up to a multiplicative constant.

Conversely, an F (θ) with a large condition number κ(F (θ)) (the ratio of the largest over the smallest eigenvalue) means that the gradient is a poor proxy for the natural gradient and thus is much less efficient.

From another angle, F (θ) is also the Hessian of the KL divergence τ → KL(P θ P τ ) at τ = θ.

If we were simply to minimize this KL divergence through gradient descent, then the number of iterations to convergence is proportional to κ(F (θ)) (in general, there is a lower bound of Ω( κ(F (θ))) for first order methods satisfying a mild condition) (Nesterov, 2004) .For a random deep network (residual or not) suffering from gradient explosion in the mean, we show heuristically in this section that the condition number of its Fisher information matrix is exponentially large in depth with high probability 3 .

First partition θ into groups of parameters according to layer, θ = (θ 11 , θ 12 , . . .

, θ 1k1 , θ 21 , . . .

, θ 2k2 , . . .

, θ L1 , . . .

, θ Lk L ), with θ lj denoting parameter j of layer l.

We can then partition the Fisher information matrix F (θ) into blocks, with the diagonal blocks having sizes DISPLAYFORM1 According to the Hermitian min-max theorem, the largest eigenvalue of F (θ) is given by max x =1 x T F (θ)x and the smallest eigenvalue is given by min x =1 x T F (θ)x (both are positive as F (θ) is positive definite under our regularity assumptions).

Thus κ(F (θ)) equals to their ratio and is lower bounded by a ratio of extremal diagonal terms max lj F (θ) lj,lj / min lj F (θ) lj,lj .

Let Y (θ) be the expectation of Y (θ) with respect to random initialization of θ in some fixed method.

Suppose there is gradient explosion such that E z (∂ lj log P θ (z)) 2 ∈ [exp(c l), exp(C l)] for universal constants c , C > 0 independent of j (this is true, for example, for feedforward tanh networks initialized in the chaotic region Schoenholz et al. (2017) ).

By concentration of measure phenomenon (as seen in BID6 Poole et al. (2016); Schoenholz et al. (2017); Yang and Schoenholz (2017) and this work), over randomization of parameters, E z (∂ lj log P θ (z)) 2 will in fact concentrate around its mean as width goes to infinity.

Thus we have, with high probability, that diagonal entries F (θ) lj,lj = E z (∂ lj log P θ (z)) 2 ∈ [exp(cl), exp(Cl)] for some new constants 0 < c < c < C < C.

Then the ratio max lj F (θ) lj,lj / min lj F (θ) lj,lj is at least exp(cL)/ exp(C1) = exp(cL − C), so that the κ(F (θ)) is exponential in L. The argument can be easily modified to accommodate gradient vanishing and other rates of gradient explosion/vanishing like exp(Θ(l α )).Thus such gradient dynamical problems cause the gradient to deviate from the natural gradient exponentially in an appropriate sense and violate the information geometry of the information manifold P θ .

For the case of minimizing KL divergence from a specific distribution, they directly cause the number of gradient descent iterations to diverge exponentially.

These issues cannot be solved by just adjusting the learning rate (though it can somewhat ameliorate the problem by taking conservative steps in such ill-conditioned regions).

The desire to understand how initialization can affect final performances of a deep neural network has led to a resurgence of mean field techniques, this time applied to deep learning.

A series of papers (Poole et al., 2016; Schoenholz et al., 2017; Yang and Schoenholz, 2017; Pennington et al., 2017) have established the depth-wise dynamics of random neural networks (i.e. networks at initialization time).

For example, Poole et al. (2016) showed that, in a random tanh classical feedforward neural network, activation norm converges exponentially fast in depth to a constant value, and so does the angle between images of two different input vectors at successive depths, which the authors proposed as a measure of expressivity that Yang and Schoenholz (2017) called "angular expressivity." Schoenholz et al. (2017) then showed that the gradient norm of such a random network suffers from exponential explosion or vanishing during the course of backpropagation.

But when the initialization variances lie on a "critical curve," the gradient is neither vanishing nor exploding, and, more importantly, the networks initialized on this "critical line" has the best test time performance after training for a fixed number of iterations.

The mean field framework was extended to residual networks (with only fully connected layers and no batchnorm) in Yang and Schoenholz (2017) .

There the authors showed that just by adding a skip connection to the feedforward network, the dynamics of a tanh network becomes subexponential.

More crucially, they investigated both tanh and ReLU residual networks, and found that whereas gradient dynamics controls the test time performances of tanh resnets, "expressivity" controls those of ReLU resnets.

This expressivity is, roughly speaking, how much distance a random network on average puts between two different input vectors; it was aptly named "metric expressivity." On the other hand, the "angular expressivity" proposed in Poole et al. (2016) (how much angle the network puts between two input vectors, as explained above) was not found to be predictive of relative test time performance of either tanh or ReLU resnets.

More precisely, the optimal initialization scheme for tanh resnet seems to strike a delicate balance between trainability and expressivity, in that weight variance too large causes too much gradient explosion and causes training to fail, whereas weight variance too small causes the typical network to collapse to a constant function Yang and Schoenholz (2017) .

The optimal variance σ 2 w satisfies σ 2 w L = const where L is depth.

On the other hand, ReLU resnets have completely different behavior with respect to initialization variance; here the best initialization scheme is obtained by maximizing the weight variance (and as a consequence also maximizing the metric expressivity) without overflowing activation values of deeper layers into numerical infs.

Indeed, trainability seems to not be a problem at all, as the gradient norm of weight parameters at each layer stays constant within O(1) over the course of backpropagation.

In this paper, we extend the results of Yang and Schoenholz (2017) to include depthwise variation of widths and of variances.

We show that they can be used to great effect to reduce gradient explosion as well as manipulating the expressivity (metric or angular) of the random network.

Corroborating Yang and Schoenholz (2017), we find that they improve tanh resnet performance by taming gradient dynamics and improve ReLU resnet performance by preventing activations from numerically overflowing while maximizing metric expressivity.

However, in certain regimes, worsening gradient explosion can mysteriously make ReLU resnet perform better, and we currently do not know how to explain this phenomenon.

Notations and Settings.

We adopt the notations of Yang and Schoenholz (2017) and review them briefly.

Consider a vanilla feedforward neural network of L layers, with each layer l having N (l) neurons; here layer 0 is the input layer.

Let DISPLAYFORM0 N (0) ) be the input vector to the network, and let x (l) for l > 0 be the activation of layer l.

Then a neural network is given by the equations x DISPLAYFORM1 is the bias vector, and (iv) φ is a nonlinearity, for example tanh or ReLU, which is applied coordinatewise to its input.

To lighten up notation, we suppress the explicit layer numbers l and write BID11 BID7 adds an identity connection or skip shortcut that "jumps ahead" every couple layers.

We adopt one of the simplified residual architectures defined in Yang and Schoenholz (2017) for ease of analysis 4 , in which every residual block is given by DISPLAYFORM2 DISPLAYFORM3 where M (l) is the width of the "hidden layer" of the residual block, (v DISPLAYFORM4 is a new set of weights and (a DISPLAYFORM5 i=1 is a new set of biases for every layer l. If we were to change the width of a 4 It is called full residual network by Yang and Schoenholz (2017), but in this paper, we will simply assume this architecture whenever we say residual network.residual network, as is done in practice, we need to insert "projection" residual blocks BID11 BID7 every couple layers.

We assume the following simplified projection residual block in this paper, for the ease of presentation 5 : DISPLAYFORM6 6 , and (π ij ) N,N i,j=1 is the "projection" matrix.

Note that we only consider fully-connected affine layers instead of convolutional layers.

Deep mean field theory is interested in the "average behavior" of these network when the weights and biases, w DISPLAYFORM7 ij , and a DISPLAYFORM8 a ; here we normalize the variance of weight parameters so that, for example, the variance of each h i is σ 2 w , assuming each x j is fixed.

While previous works have all focused on fixing σ (l) • to be constant across depth, in this paper we are interested in studying varying σ (l)• .

In particular, other than σ (l) π which we fix at 1 across depth (so that "projection" doesn't act like an "expansion" or "contraction"), we let σ DISPLAYFORM9 Hereafter the bar notation •, •, • do not apply to σs, so that, by σ a , for example, we always mean the constant σ a .We make the same statistical assumptions as in Yang and Schoenholz (2017) .

In the interest of space, we relegate their discussion to Appendix A.Mean Field Quantities.

Now we define the central quantities studied in this paper.

Inevitably, as deep mean field theory analyzes neural networks closer and closer to those used in practice, the variables and notations become more and more complex; our paper is no different.

We have however included a glossary of symbols (Table A.1) that will hopefully reduce notation confusions to the first time reader.

Definition 4.1.

Fix an input x (0) .

Define the length quantities q DISPLAYFORM10 Here (and in the following definitions) the expectations • are taken over all random initialization of weights and biases for all layers l, as N (l) , M (l) → ∞ (large width limit).

Note that in our definition, the index 1 can be replaced by any other index by Axiom 1.

Thus p (l) is the typical magnitude (squared) of a neuronal activation at layer l. Definition 4.2.

Fix two inputs x (0) and x (0) .

We write • to denote a quantity • with respect to the input x (0) .

Then define the correlation quantities γ DISPLAYFORM11 .

Again, here the index 1 does not matter by Axiom 1.

By metric expressivity, we mean s DISPLAYFORM12 , and we will also call e (l) angular expressivity.

Following Yang and Schoenholz (2017), we assume p (0) = p (0) for the ease of presentation, but this is a nonessential assumption.

Then, as we will see, DISPLAYFORM13 for all l, and as a result, DISPLAYFORM14 Definition 4.3.

Fix an input x (0) and a gradient vector (∂E/∂x (L) i ) i of some loss function E with respect to the last layer x (L) .

Then define the gradient quantities χ (l) := (∂E/∂x DISPLAYFORM15 2 for • = a, b, and χ (l) DISPLAYFORM16 Here the expectations are taken with Axiom 2 in mind, over both random initialization of forward and backward weights and biases, as N → ∞ (large width limit).

Again, the index 1 or 11 does not matter by Axiom 1.

Just as in previous works in deep mean field theory (Poole et al., 2016; Schoenholz et al., 2017; Yang and Schoenholz, 2017) , the primary tool for investigating behaviors of large width networks is the central limit theorem.

Every time the activations of the previous layer pass through an affine layer whose weights are sampled i.i.d.

, the output is a sum of a large number of random variables, and thus follows an approximately Gaussian law.

The output of the next nonlinearity is then a nonlinear transform of a Gaussian variable, with computable mean and variance.

Repeating this logic gives us a depthwise dynamical system of the activation random variables.

The gradient dynamics can be similarly derived, assuming Axiom 2.

Theoretically and empirically, WV does not change the mean dynamics of forward quantities like activation norm p but can be used to control gradient dynamics χ.

Intuitively, this is because each neuron at a width-changing layer "receives messages" from different numbers of neurons in the forward and backward computations.

If for example N (l) = N (l−1) /2, then on the backward pass, the neuron receives half as many messages on the backward pass than in the forward, so we expect that its gradient should be half of what it would be when DISPLAYFORM0 On the other hand, VV will usually change both the forward and backward dynamics of mean field quantities.

The phase transitions are many and complicated, but the overall trend is that, as we dampen the variance with depth, both forward and backward dynamics will dampen as well; the only major exception is weight gradients in ReLU resnets (see Appendix B.1).

In contrast to WV which works the same for any nonlinearity, the phase diagram for VV is controlled by different quantities depending on the nonlinearity.

We show through experiments that all of the complexities involved in VV theory are reflected in the practice of training neural networks: we can predict the contour lines of test time accuracy using only our mean field theory (Section 7 and Appendix B)Expressivity vs Trainability Tradeoff.

Yang and Schoenholz (2017) made the observation that the optimal initialization scheme for tanh resnets makes an optimal tradeoff between expressivity and trainability: if the initialization variances are too big, then the random network will suffer from gradient explosion with high probability; if they are too small, then the random network will be approximately constant (i.e. has low metric expressivity) with high probability.

They posited that such tradeoff between expressivity and trainability in ReLU resnets is not observed because the gradient against weight parameters w and v are bounded w.r.t.

depth (so that there is no gradient explosion) 7 , while (metric) expressivity is exponential, thus strongly dominating the effect on final performance.

We confirm this behavior in tanh resnets when decaying their initialization variances with depth: When there is no decay, gradient explosion bottlenecks the test set accuracy after training; when we impose strong decay, gradient dynamics is mollified but now (metric) expressivity, being strongly constrained, bottlenecks performance (Section 7.2).

Indeed, we can predict test set accuracy by level curves of gradient norm ratio χ DISPLAYFORM1 w in the region of small variance decay, while we can do the same with level curves of metric expressivity s when in the region of large decay FIG4 .

The performance peaks at the intersection of these two regions.

The left two plots show that mean forward dynamics are more or less preserved, albeit variance explodes toward the deeper layers, where WV is applied.

The last plot show that the gradient dynamics is essentially suppressed to be a constant compared to the exp( √ L) dynamics of tanh resnet without width decay.

Dashed lines indicate theoretical estimates in all three plot; solid, simulated data, which is generated from random residual networks with 100 layers and N (0) = 2048, and we half the widths at layers l = m 2 for m = 4, 5, . . . , 9.Also corroborating Yang and Schoenholz FORMULA2 , we did not observe a tradeoff in ReLU resnets VV.In the regime with small to moderate variance decay, VV exerts its effect through metric expressivity, not gradient dynamics (Section 7.1) 8 .

However, when we impose strong decay, gradient explosion, but not metric expressivity, predicts performance, in the unexpected way that worse gradient explosion correlates with better performance; that is, expressivity and trainability (as measured by gradient explosion) are both worsening, yet the performance increases!

We currently have no explanation for this phenomenon but hope to find one in future work.

Width variation is first passingly mentioned in Schoenholz et al. (2017) as a potential way to guide gradient dynamics for feedforward networks.

We develop a complete mean field theory of WV for residual networks.

Via TAB0 .2 and Thm C.3, we see that width variation (WV) has two kinds of effects on the mean gradient norm: when compared to no width variation, it can multiply the squared gradient norm of biases b i or weights w ij by N/M (which doesn't "stack", i.e. does not affect the squared gradient norm of lower layers), or it can multiply the squared gradient norm of x i by N/N (which "stacks", in the sense above, through the dynamics of χ).

We will focus on the latter "stacking" effect and assume DISPLAYFORM0 Suppose from layer l to layer m, χ (m) rises to r times χ (l) .

If we vary the width so that N (m−1) is rN (m) , then this gradient expansion is canceled, and χ DISPLAYFORM1 so that it is as if we restarted backpropagation at layer m.

Remarkably, changing the width does not change the mean field forward dynamics (for example the recurrences for p, q, γ, λ remain the same) (Thm C.2).

But, as always, if we reduce the width as part of WV (say, keeping N (0) the same but reducing the widths of later layers), the variance of the sampled dynamics will also increase; if we increase the width as part of WV (say, keeping N (L) the same but increasing the widths of earlier layers), the variance of the sampled dynamics will decrease.

We can apply this theory of WV to tanh residual networks (φ = tanh in TAB0 .2) without VV.

By Yang and Schoenholz (2017), tanh residual networks with all β • = 0 have gradient dynamics DISPLAYFORM2 If we place projection blocks projecting DISPLAYFORM3 . .

, then the gradient norms would be bounded (above and below) across layers, as reasoned above.

Indeed, this is what we see in FIG1 .

The rightmost subfigure compares, with log scale y-axis, the gradient The zig: fix βv = βa = 0; fix βw = β b and increase both from 0 to 2 (making Vr = βw + βv go from 0 to 2 as well); The zag: fix βv = 0, βw = β b = 2; increase βa from 0 to 2 (increasing Ur from 0 to 2 as well).

For each setting of the hyperparameters, we train a network on MNIST with those hyperparameters for 30 epochs.

We then report the accuracy that the network achieved on the training and test sets.

The plots above are, in order from left to right, (a) zig/test, (b) zag/test, (c) zig/train, (d) zag/train.

In the zig, we have overlaid a contour plot of s (computed from Thm C.2), which is almost identical to the contour plots of p and χ (0) /χ (l) ; numbers indicate log 1 + log s.

The dashed line is given DISPLAYFORM4 In the zag, we have overlaid a contour plot of χ DISPLAYFORM5 dynamics with no WV to that with WV as described above.

We see that our theory tracks the mean gradient dynamics remarkably precisely for both the WV and the no WV cases, and indeed, WV effectively caps the gradient norm for l ≥ 16 (where WV is applied).

The left two figures show the forward dynamics of p and e, and we see that the WV does not affect the mean dynamics as predicted by theory.

However, we also see dramatic increase in deviation from the mean dynamics at every projection layer in the forward case.

The backward dynamics (rightmost figure) similarly sees large deviations (1 standard deviation below mean is negative for χ a and χ w ), although the deviations for χ is more tamed but still much larger than without WV.Therefore, width variation is unique in a few ways among all the techniques discussed in the mean field networks literature so far, including variance decay as studied below, adding skip connections, or changing activation functions: It can ameliorate or suppress altogether gradient explosion (or vanishing) problems without affecting the mean forward dynamics of p, q, λ, γ, c, e. To do so, it has to choose a trade-off from the following spectrum: At one end, we truncate neurons from the original network (say, keeping N (0) the same), so that we have fewer parameters, less compute, but larger deviations from the mean dynamics.

At the other, we add neuron to the original network (say, keeping N (L) the same), so that we have more parameters, more compute, and smaller deviations from the mean dynamics.7 VARIANCE VARIATION 7.1 RELU A zigzag of parameters controls various asymptotics of ReLU resnet: "zigging" V r := min(β v + β b , β a ) from 0 to > 1, and then "zagging" U r := β v + β w from 0 to > 1.

During the zig, the asymptotics of p is subdued from exp(poly(l)) to poly(l).

During the zag, it is further reduced to Θ(log l) (at U r = 1) and Θ(1) (when U r > 1).

On the contrary, the gradient dynamics of weight parameters become more explosive along the zigzag.

During the zig, χ DISPLAYFORM6 v increases from Θ(l βv ) to a bigger poly(l).

During the zig, both quantities increase the exponents of their polynomial dependence in l. In the interest of space, we stop here our sketch of VV dynamics for ReLU resnet, but refer the reader to Appendix B.1 for more detailed descriptions and Appendix C for proofs.

To test our theory, we sweep through these two macro-phases of the parameter space in our experiments and train an array of randomly initialized ReLU resnets; results are demonstrated in FIG2 .

The figure caption gives the experimental details involved.

In addition, we provide heatmap and contour plots of various quantities of interest such as p, e, and χ In all experiments here, we pin σ•s all to 1.

From left to right: (a) and (b).

We sweep Ut : 0 1 in to ways, testing to what extent Ut determines the final performance.

In the first way (a), we set all β•s equal to a common β and increase β : 0 1.

In the second way (b), we clamp β b = βa = 1 and set βw = βv, increasing their common values from 0 to 1.

The heatmaps are produced from final test set performance as in FIG2 .

As we can easily see, these two sweeps produce almost identical heatmaps.

In both plots, there is a visible peak in the heatmaps in the upper left.

On each of (a) and (b), we overlay the contours of χ DISPLAYFORM7 w (in blue) to the left of the peak and those of p (in green) to the right of the peak, the latter being very similar to those of s. The blue numbers indicate log χ DISPLAYFORM8 w while the green numbers indicate log p. (c).

We fix Ut = 1 by fixing βv = βa = 1 and sweep βw = β b from 0 to 1, thereby sweeping Vt from 1 to 2.

The heatmap is obtained again with the same procedure as in FIG2 from the test set after training.

Overlaid on top is a contour plot of s, and the numbers indicate log s.

Discussion.

First notice that training fails in the upper left corner of the zig.

This happens because of numerical instability caused by exploding p and χ (0) /χ (l) , which grow like exp( DISPLAYFORM9 ) (Thms C.9 and C.11).

Indeed, one of the contour lines of p traces out almost exactly where training fails.

The dashed line is a level set of the dominant term of asymptotic expansion of log p, and we see it agrees with the contour of p very well.

By increasing β w = β b , we effectively solved the activation explosion problem observed by Yang and Schoenholz (2017) without changing the activation function.

Second, observe that performance actually dips in the direction where χ (0) /χ (l) decreases, quite counterintuitively 9 .

This can be explained (as in Yang and Schoenholz (2017) ) by noting that gradient against weights, χ w and χ v , in fact respectively remain bounded and polynomial in l (and changes rather mildly with V r ; see FIG1 ; gradient against biases do experience the same behavior as χ, but in general they are much less important than the weights, as parameters go.

In addition, the performance is also dipping in the direction where s decreases (exponentially) in (V r , L)-space.

This is the quantity that essentially underlies the exponential expressivity result (as told from an extrinsic curvature perspective) of Poole et al. FORMULA2 ; as s decreases dramatically, it gets harder and harder for a linear functional in the final layer to tell apart two input vectors.

This exponential loss in expressivity dominates the effect on performance more than a polynomial reduction in gradient explosion.

Third, it is remarkable that in the zag regime, the level curves of χ DISPLAYFORM10 v (but not those of p, s, e, χ, or χ w !) accurately predict the contour of the test set performance, in such a counterintuitive way that greater gradient explosion χ DISPLAYFORM11 v correlates with better performance FIG2 ).

Especially when β a (and thus also U r ) is large, the weight gradient dynamics are much more explosive than that of metric expressivity, so according to prevailing theory, gradient explosion should bottleneck performance, but instead the reality is the exact opposite.

It is currently unclear if in certain situations like this, larger gradient expansion is actually beneficial, or if there is a quantity yet undiscovered which has the same level curves and can explain away this seeming paradox (like how s explains away χ (0) /χ (l) above, in the zig regime).

Of the quantities that appear in Fig. A .2, none foots this bill.

As in the previous section, we briefly sketch the VV dynamics of tanh resnets, but defer a more detailed discussion to Appendix B.2 and the proofs to Appendix C. We are concerned with the scenario that q → ∞ with l, as otherwise higher layers become essentially unused by the network.

The major phases of tanh resnet VV dynamics are then determined by when U t := min(β v , β a ) < 1 and when U t = 1 (Thms C.5 and C.6).

Within the former, the gradient dynamics is controlled by W t := 1 − β w + U t − 2β v ; as W t starts positive, decrease to 0, and then becomes negative, the gradient ratio χ (0) /χ (l) starts out as exp(poly(l)), turns into polynomial, and finally becomes bounded.

When U t = 1, χ (0) /χ (l) is always subpolynomial, with V t := β v + β w making it bounded as V t increases past 1.

On the other hand, the dynamics of p is quite simple, with p = Θ(l 1−Ut ) when U t < 1 and p = Θ(log l) when U t = 1.This theory enables us to predict and optimize an VV initialization scheme for tanh resnets.

We sweep through the the two phases described above, train the corresponding random networks, and exhibit the results in FIG4 .

The figure caption details the experimental setup.

Discussions.

FIG4 and FIG4 sweep through U t from 0 to 1 in two different ways while obtaining almost identical test set performance heatmap, showing indeed that the hyperparameters β • exert their effect through U t = min(β v , β a ).

In each of these two plots, peak accuracies happen in the upper left.

To the left of such a peak, gradient norm χ DISPLAYFORM0 w predicts the accuracy, while to the right of such a peak, metric expressivity s (and in this case p as well because they induce similar contours) does.

But χ DISPLAYFORM1 w would not do well in the large β region because the slopes of its contours are too steep; conversely s would not predict well in the small β region because its contour slopes are not steep enough.

Indeed, one sees that the slopes of the heatmap level set boundaries decrease as the accuracy levels increase (and as β decreases from 1), but when the level peaks, the slope suddenly becomes much steeper (compare the left boundary of the peak to its right boundary).

Our observation here reaffirms the trainability vs expressivity tradeoff studied in Yang and Schoenholz (2017).In FIG4 , we study the U t = 1 phase.

Here s alone predicts performance (though in the large β w = β b region, final accuracy becomes more random and the prediction is not as great).

This is expected, as χ (0) DISPLAYFORM2 • is now subpolynomial for all • = v, w, a, b (Thm C.6), so that trainability is not an issue.

In this paper, we derived the mean field theory of width and variance variation and showed that they are powerful methods to control forward (VV) and backward (VV + WV) dynamics.

We proved that even with a fixed architecture and activation function, the mean field dynamics of a residual neural network can still be manipulated at will by these two methods.

Extraordinarily, the mean field theory we developed allowed us to accurately predict the performances of trained MNIST models relative to different initializations, but one puzzling aspect remains where test set accuracy seems to increase as gradient explosion worsens in one regime of random ReLU resnets.

Open Problems.

We solved a small part, width variation, of the program to construct mean field theories of state-of-the-art neural networks used in practice.

Many open problems still remain, and the most important of them include but is not limited to 1.

batchnorm, 2.

convolution layers, and 3.

recurrent layers.

In addition, more work is needed to mathematically justify our "physical" assumptions Axiom 1 and Axiom 2 to a "math" problem.

We hope readers will take note and contribute toward deep mean field theory.

Jeffrey Pennington, Samuel Schoenholz, and Surya Ganguli.

Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice.

In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30, pages 4788-4798.

Curran Associates, Inc., 2017.URL http://papers.nips.cc/paper/ 7064-resurrecting-the-sigmoid-in-deep-learning-through-dynamical-isometry-theory-a pdf.

Ben Poole, Subhaneil Lahiri, Maithreyi Raghu, Jascha Sohl-Dickstein, and Surya Ganguli.

Exponential expressivity in deep neural networks through transient chaos.

In Advances In Neural Information Processing Systems, pages 3360-3368, 2016.

DISPLAYFORM0 DISPLAYFORM1 The two cases for χ/χ are resp.

for a projection and a normal residual block, assuming σπ = 1.

The V and W operators are defined in Defn C.1.

We make several key "mean field" assumptions, which were formulated in their entirety first in Yang and Schoenholz (2017) (though Axiom 2(a) has been stated first by Schoenholz et al. (2017) ).

While these assumptions may be mathematically unsatisfying, identifying them and discovering that they lead to highly precise prediction is in fact one of the most important contributions of deep mean field theory.

We also assume that the gradient ∂E/∂x (l) i with respect to the loss function E satisfies (∂E/∂x DISPLAYFORM0 Axiom 2 (Gradient independence).

(a) We assume the we use a different set of weights for backpropagation than those used to compute the network outputs, but sampled i.i.d.

from the same distributions.(b) For any loss function E, we assume that the gradient at layer l, ∂E/∂x DISPLAYFORM1 i , is independent from all activations h (l) j and x (l−1) j from the previous layer.

One can see that Axiom 1(a) is satisfied if the input x (0) ∈ {±1} N and Axiom 1(b) is satisfied if Axiom 2 below is true and the gradient at the last layer ∂E/∂xL ∈ {±1} N .

Axiom 2(a) was first made in Schoenholz et al. (2017) for computing the mean field theory of gradients for feedforward tanh networks.

This is similar to the practice of feedback alignment (Lillicrap et al., 2016) .

As discuss in Section 6, WV is essentially a post-hoc technique to tweak an existing gradient dynamic without changing the forward dynamics.

Thus in this section we assume all widths are constant, DISPLAYFORM0 for any m, l, n, so that WV can be applied as a "touch-up" if necessary.

We will overview the phases and transitions due to VV, but defer all proofs to later sections.

It was shown in Yang and Schoenholz (2017) that, with no variance decay, in an ReLU resnet, both the mean squared activation norm (p) and the mean squared gradient norm (χ) explode exponentially with depth, and this causes training to fail for even 100 layer networks.

We show that this problem is in fact extremely easy to fix, requiring no architectural changes at all, only that β • be increased from 0 so that the randomization variances decay across depth (Thms C.9 and C.11).Gradient quantities.

The main driver of this gradient mollification is V r := β v + β w .

When V r ∈ [0, 1), the gradient norm varies like χ (0) /χ (l) = exp(Θ(l 1−Vr )); when V r = 0 this recapitulate the exponential behavior derived in Yang and Schoenholz (2017) .

When V r = 1, it experiences a sharp phase transition, where now χ (0) /χ (l) = poly(l).

As V r becomes larger than 1, χ (0) /χ (l) = Θ(1), all bounded!

Fig. B .3 verifies this result empirically, and in fact show that our computed asymptotic expansions in Thm C.11 are highly accurate predictors.

It is both interesting and important to note that the gradient norms for actual trainable parameters, such as χ w and χ v , are affected differently by V r .

In fact, χ DISPLAYFORM0 w is bounded(!) with l when V r < 1 (the V r = 0 case is already observed in Yang and Schoenholz (2017)) but phase transitions to poly(l) for V r ≥ 1, while χ DISPLAYFORM1 v is already poly(l) when V r < 1, and remains so as V r is increased to > 1.

Curiously, greater gradient explosion in χ v predicts better performance in the V r > 1 regime, and we currently do not know if this is intrinsic or there are confounding variables; see Section 7.1.Length quantities.

Similarly, V r is the primary conduit for mollifying the behavior of squared activation norms p and q (Thm C.9).

Like the gradient dynamics, p = exp(Θ(l 1−Vr )) when V r < 1; when V r = 0 this recapitulates the results of Yang and Schoenholz (2017).

As V r rise to 1 and above, p experiences a phase transition into polynomial dynamics, but unlike the case of χ, it is not constant when V r > 1.

Instead, a different parameter, U r := min(β v + β b , β a ) drives the asymptotics of p in the V r > 1 regime.

When U r ∈ [0, 1) is small, p grows like Θ(l 1−Ur ).

The instant U r hits 1, p is just logarithmic, p = Θ(log l).

As U r shoots past 1, p becomes constant.

Thus the dynamics of p is governed by a zigzag through (β • ) •=v,w,a space.

Fig. B .4 goes through each of the five cases discussed above and verifies that our asymptotics are correct.

Cosine distance.

e = γ/p measures how well the input space geometry (angles, in this case) is preserved as the input space is propagated through each layer.

Its dynamics are much simpler than those of p and χ above.

If e (0) = 1, then e (l) = 1 for all l trivially.

If e (0) < 1, then we have one of the following two cases, • If V r ≤ 1 or U r ≤ 1, then e → 1 irrespective of initial data p (0) and γ (0) .

DISPLAYFORM2 is bounded above when Vr = 1.6.

TAB0 .2, while dashed lines indicate asymptotics proved in Thm C.9.

In all but the leftmost plot, we show both p and ∆p = p − p (possibly adjusted for log factors) and their asymptotics.

To facilitate visualization, all dashed lines except the red one in the leftmost plot are shifted vertically to match the end point of the corresponding solid lines.

In the leftmost plot, the red lines are respectively log p (solid) and the leading term • If V r > 1 and U r > 1, then e converges to a fixed point e * < 1 which depends on the initial data p (0) and γ (0) , at a rate of Θ(l 1−Ur ).

Thus ReLU very much likes to collapse the input space into a single point (e = 1 means every two input vectors get mapped to the same output vector), and the only way to prevent this from happening is to make the β • s so high, that higher layers barely modifies the computation done by lower layers at all.

Indeed, the second condition V r > 1 and U r > 1 ensures that p = Θ(1) as discussed above (Thm C.9), so that as l → ∞, layer l's residual adds to x (l−1) only a vector of vanishing size compared to the size of x (l−1) .

While ReLU resnet depends heavily on V r = β v +β w and U r = min(β v +β b , β a ), U t := min(β v , β a ) and W t := 1 − β w + U t − 2β v are the key quantities determining the dynamics in the case of tanh resnet, with V t = β v + β w = V r playing a minor role.

We will study tanh resnets in the setting where q → ∞ as l → ∞; otherwise, p is bounded, meaning that higher layers become essentially unused by the neural network (similar to the discussion made in Appendix B.1(Cosine distance) above).

In this setting, it can be shown that U t ≤ 1 (Thm C.5).Gradient quantities.

By Thm C.6, as long as U t stays below 1, the asymptotics of χ is entirely governed by W t , which is 1 when all β • s are 0 and most of the time decreases as β • s are increased.

DISPLAYFORM0 ; the results of Yang and Schoenholz (2017) are recovered by setting all β • s to 0, thus W t = 1 and DISPLAYFORM1 becomes polynomial, and as W t dips below 0, gradient expansion becomes bounded.

DISPLAYFORM2 is automatically suppressed to be subpolynomial.

The only minor phase transition here is going from V r = 1 to V r > 1 (and V r cannot be less than 1 by our assumption that q → ∞).

In the former case, the gradient expansion is exp(Θ( √ log l)), while in the latter case it is bounded.

Length quantities have simpler asymptotics determined by U t : either U t < 1 and p = Θ(l 1−Ut ), or U t = 1 and p = Θ(log l) (Thm C.5).

Cosine distance, unlike the case of ReLU resnets, can be controlled effectively by β a and β v (Thm C.7).

When β a > β v , the magnitude of a (l) i drops much more quickly with depth than that of v (l) ij , so that higher layers experience the chaotic phase (Schoenholz et al., 2017; Yang and Schoenholz, 2017) , driving e (l) toward the limit point e * = 0.

On the other end, when DISPLAYFORM3 i with large l, so that the higher layers experience the stability phase (Schoenholz et al., 2017; Yang and Schoenholz, 2017) , collapsing all inputs to the same output vector, sending e (l) → 1.

Only when β a = β v could the fixed point e * be controlled explicitly by σ v and σ a , with e * given by the equation DISPLAYFORM4 DISPLAYFORM5 for some k ∈ Z (this is slightly different from the standard usage ofÕ), and DISPLAYFORM6 All asymptotic notations are sign-less, i.e. can indicate either positive or negative quantities, unless stated otherwise.

We recall integral transforms from Yang and Schoenholz (2017):Definition C.1.

Define the transforms V and W by Vφ(ρ) := E[φ(z) 2 : z ∼ N (0, ρ)] and DISPLAYFORM7

Yang and Schoenholz (2017) gave recurrences for mean field quantities p, q, γ, λ, χ under the assumption of constant initialization variances across depth.

The proofs there carry over straightforwardly when variance varies from layer to layer.

Schoenholz et al. (2017) also derived backward dynamics for when the width of the network is not constant.

Generalizing to the residual network case requires some careful justifications of independences, so we provide proof for gradient dynamics; but we omit the proof for the forward dynamics as it is not affected by nonconstant width and is almost identical to the constant variance case.

Theorem C.2.

For any nonlinearity φ in an FRN, regardless of whether widths vary across layers, DISPLAYFORM0 Theorem C.3.

Suppose a random residual network receives a fixed gradient vector ∂E/∂x DISPLAYFORM1 with respect to some cost function E, at its last layer.

For any nonlinearity φ in an FRN, under Axiom 1 and Axiom 2, wheneverφ(ζ) 2 has finite variance for any Gaussian variable ζ, DISPLAYFORM2 Proof.

We will show the proof for the projection connection case; the identity connection case is similar but easier.

DISPLAYFORM3 .

We have the following derivative computations: DISPLAYFORM4 where in the second equality, we expanded algebraically, and in the third equality, we use the symmetry assumption Axiom 1 and the independence assumption Axiom 2.

Now, DISPLAYFORM5 by the independence of {v ik } i,k ∪ {h k } k ∪ {w kj } k,j (by our independence assumptions Axiom 2).

Similarly, because {π ij } j ∪ {π i j } j ∪ {v ik } k ∪ {v i k } k for i = i is mutually independent by our assumptions, one can easily see that DISPLAYFORM6 For the other gradients, we have (where we apply Axiom 2 implicitly) DISPLAYFORM7

In this section we derive the asymptotics of various mean field quantities for tanh resnet.

The main proof technique is to bound the dynamics in question with known dynamics of difference equations (as in Yang and Schoenholz (2017) ).

DISPLAYFORM0 Theorem C.5.

Suppose φ = tanh, and q (l) → ∞ as l → ∞.1.

If U t < 1, then 1 > β w + U t and DISPLAYFORM1 More specifically, DISPLAYFORM2 2.

If U t = 1, then β w = 0, and DISPLAYFORM3 3.

U t cannot be greater than 1.Proof.

Claim 1.

We have Vφ(q) = 1 − 2 π q −1/2 + Θ(q −3/2 ) by Lemma D.5.

Thus DISPLAYFORM4 where we used the assumption 1 − U t > 0.

Thus p (l) = Θ(l 1−Ut ) and goes to infinity with l. DISPLAYFORM5 Because we assumed q → ∞, the first term necessarily dominates the second, and 1 − U t − β w > 0.

The possible asymptotics of q are then DISPLAYFORM6 Then for q to go to infinity, β w has to be 0, so that q = Θ(log l) as well, and DISPLAYFORM7 Theorem C.6.

Suppose φ = tanh, and q (l) → ∞ with l. Recall W t = 1 − β w + U t − 2β v and DISPLAYFORM8 Proof.

By Thm C.3 and Lemma D.4, we have DISPLAYFORM9 where C = 2 3 2 π .

Since q (l) has different growth rates depending on the hyperparameters, we need to consider different cases:• If U t < 1, then Thm C.5 implies β w + U t < 1 and q = Θ(l DISPLAYFORM10 • DISPLAYFORM11 • If U t = 1, then by Thm C.5, β w = 0 and q = Θ(log l).

So l −βv−βw q −1/2 = Θ(l −βv−βw / √ log l).• If β v + β w < 1, then β v < 1 =⇒ U t < 1, contradiction.• DISPLAYFORM12 • DISPLAYFORM13 Theorem C.7.

Suppose φ = tanh and q → ∞. If e (0) < 1, then e (l) converges to a fixed point e * , given by the equations DISPLAYFORM14 Note that in the case β a = β v , we recover the fixed point of tanh residual network without variance decay.

Proof.

We have DISPLAYFORM15 Using Lemma D.1 and Lemma D.5, we can see that the LHS is monotonic (increasing or decreasing) for large enough l.

Therefore e (l) is a bounded monotonic sequence for large enough l, a fortiori it has a fixed point.

If we express e = e * + , DISPLAYFORM16 RHS It's easy to verify via Thm C.5 that either p/(p−p) = Θ(l log l) (when U t = 1) or p/(p−p) = Θ(l) (all other cases).

If − = Ω((l log l) −1 ), then = Ω(log log l) (by Euler-MacLaurin formula), which would be a contradiction to = o(1).

DISPLAYFORM17 10 Hence the RHS goes to 0 with l.

LHS If β v > β a , then the LHS converges to 1.

So e * = 1.

DISPLAYFORM18 Vφ(q) .

As l → ∞, c ∼ e → e * , and Wφ(q, e * q) → 2 π arcsin(e * ),and Vφ(q) → 1.

Therefore, e * = 2 π arcsin(e * ), for which there are 2 solutions, 0, the stable fixed point, and 1, the unstable fixed point.

In particular, for all e (0) < 1, e (l) →

0.If β v = β a , then taking limits l → ∞, we get DISPLAYFORM19

The asymptotics of ReLU resnet depends on the following values:Definition C.8.

DISPLAYFORM0 Theorem C.9.

Let φ = ReLU.

Then we have the following asymptotics of p and q: DISPLAYFORM1 • If U r = 1 DISPLAYFORM2 • If V r = 1• If W r = 1 − U r p = Θ(l max(Wr,1−Ur) ), and q = Θ(l max(max(Wr,1−Ur)−βw,−β b ) ).•

Otherwise DISPLAYFORM3 10 In fact, since must be o(log log · · · log l) for any chain of logs, DISPLAYFORM4 for any k, where log (k) is k-wise composition of log; so ( − ) DISPLAYFORM5 • p, q = exp( DISPLAYFORM6 , R is the same R as in Lemma D.7, depending on only l, W r , and V r , with DISPLAYFORM7 and q = exp( DISPLAYFORM8 In particular, p, q = poly(l) if V r ≥ 1.

DISPLAYFORM9 We apply Lemma D.8 with β = β v + β w , δ = • If V r ≤ 1 or U r ≤ 1, then lim l→∞ e (l) = 1.• If V r > 1 and U r > 1, then e (l) converges to a fixed point e * < 1, dependent on the initial data p (0) and γ (0) , at a rate of Θ(l −Ur+1 ).Proof.

By BID4 ; Yang and Schoenholz (2017), we have Wφ(q, cq) = Vφ(q)J 1 (c) = DISPLAYFORM10 As in the proof of Thm C.7, we have DISPLAYFORM11 where the last inequality holds for all large l, by Lemma D.1.

So the LHS is nonnegative for large l, which implies e (l) is a bounded monotonically increasing sequence for large enough ls and thus has a limit e * .Writing e = e * + , we have DISPLAYFORM12 LHS.

By Thm C.9, p = ω(1) (assuming σ v , σ w > 0), so that p/(p − p) =Õ(l).

As in the proof of Thm C.7, − cannot beΩ(l −1 ), or else → ∞. Thus ( − )p/(p − p) = o(1), and the LHS in the limit l → ∞ becomes e * .

In all cases, we will find e * = 1.

• If p = ω(l βw−β b ), then c ∼ e → e * , so that in the limit l → ∞, e * = J 1 (e * ).

This equation has solution e * = 1.

DISPLAYFORM0 • If p = o(l βw−β b ), then c → 1, so that e * = J 1 (1) = 1.• DISPLAYFORM1 , and we have an equation e * = J 1 ( DISPLAYFORM2 Note that since e * ∈ [0, 1], ) iff the equality condition above is satisifed, i.e. e * = 1.

DISPLAYFORM3 DISPLAYFORM4 by the same logic as in the proof of Lemma D.1.

As in the above case, Lemma D.1 yields the following results: DISPLAYFORM5 .

Since the RHS of this equation is a convex combination of J 1 (e * ) and 1, and J 1 (e * ) ≥ e * by the monotonicity of J 1 , the equality can hold only if J 1 (e * ) = e * .

The only such e * is 1.• DISPLAYFORM6 , and we have an equation DISPLAYFORM7 .

By the monotonicity of J 1 , the RHS is at least DISPLAYFORM8 , which is a convex combination of e * and 1.

Since e * ≤ 1, the equality can hold only if e * = 1.

DISPLAYFORM9 By Thm C.9, p = Θ(1) and therefore γ = Θ(1).

Both converge to fixed points p * and γ * (possibly dependent on initial data p (0) and γ (0) ) because they are monotonically increasing sequences.

Thus e * = γ * /p * .Unwinding the proof of Thm C.9, we see that p − p = l −Ur , so that p/(p − p) = Θ(l Ur ).

Since the RHS of Eq. ( ) cannot blow up, it must be the case that DISPLAYFORM10 1) for some constant F, then the LHS becomes e * + F in the limit.

Yet, unless γ (0) = p (0) , e * < 1.

Therefore, F > 0 (or else, like in the case of V r ≤ 1 or U r ≤ 1, e * = 1) whenever γ (0) < p (0) , and = Θ(l −Ur+1 ).Theorem C.11.

Suppose φ = ReLU.• DISPLAYFORM11 • If U r ∈ [0, 1) DISPLAYFORM12 Under review as a conference paper at ICLR 2018 DISPLAYFORM13 • DISPLAYFORM14 Proof.

Using Thm C.3 and the fact that Vφ(q) = 1 2 q, Vφ(q) = 1 2 , we get DISPLAYFORM15 χ (l) = Θ(1) by Lemma D.7.

By Thm C.9:• If V r > 1, then DISPLAYFORM16 χ (l) = Θ(1) by Lemma D.7.• If U r ∈ [0, 1) p = Θ(l 1−Ur ) and q = Θ(l max(1−Ur−βw,−β b ) ).

So • If W r = 1 − U r p = Θ(l max(Wr,1−Ur) ), and q = Θ(l max(max(Wr,1−Ur)−βw,−β b ) ).

So • We have p = exp(

In this section, we present many lemmas used in the proofs of the main theorems.

In all cases, the lemmas here have already appeared in some form in Yang and Schoenholz (2017) , and for completeness, we either include them and their proofs here or improve upon and extend them, with the blessing of the authors.

Lemma D.1.

Asymptotically, c = σ DISPLAYFORM0 DISPLAYFORM1 Proof.

Euler-MacLaurin formula.

Lemma D.7.

Suppose (l) satisfies the recurrence (l) = (l−1) (1 + δ l β ).

for some nonzero constant δ ∈ R independent of l.• If β > 1, then (l) = Θ(1).• If β = 1, then (l) = Θ(l δ ).•

If 0 < β < 1, then (l) = exp( Exponentiating gives the desired result.

Lemma D.8.

Suppose (l) = Cl −α + (l−1) (1 + δ/l β ) for α ∈ R, C = 0, and δ = 0.

Then• If β > 1, then DISPLAYFORM2 • (l) = Θ(log l) if α = 1;• (l) = Θ(1) if α > 1.• If β = 1, then DISPLAYFORM3 • (l) = Θ(l δ log l) if 1 − δ = α.• If β < 1, then Furthermore, for β = −δ = 1: (l) ∼ l −1 if α > 2, (l) ∼ l 1−α if α < 2, and (l) ∼ l δ log l if α = 2.

DISPLAYFORM4 Proof.

We can unwind the recurrence to get A fortiori, (l) = e δ 1−β l 1−β +Θ(l max(0,1−2β) ) .For our "furthermore" claim: the case of δ = −1 telescopes, so that the upper and lower constants hidden in Θ can both be taken to be 1.

<|TLDR|>

@highlight

By setting the width or the initialization variance of each layer differently, we can actually subdue gradient explosion problems in residual networks (with fully connected layers and no batchnorm). A mathematical theory is developed that not only tells you how to do it, but also surprisingly is able to predict, after you apply such tricks, how fast your network trains to achieve a certain test set performance. This is some black magic stuff, and it's called "Deep Mean Field Theory."