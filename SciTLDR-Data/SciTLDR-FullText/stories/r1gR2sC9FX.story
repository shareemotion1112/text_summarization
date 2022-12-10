Neural networks are known to be a class of highly expressive functions able to fit even random input-output mappings with 100% accuracy.

In this work we present properties of neural networks that complement this aspect of expressivity.

By using tools from Fourier analysis, we show that deep ReLU networks are biased towards low frequency functions, meaning that they cannot have local fluctuations without affecting their global behavior.

Intuitively, this property is in line with the observation that over-parameterized networks find simple patterns that generalize across data samples.

We also investigate how the shape of the data manifold affects expressivity by showing evidence that learning high frequencies gets easier with increasing manifold complexity, and present a theoretical understanding of this behavior.

Finally, we study the robustness of the frequency components with respect to parameter perturbation, to develop the intuition that the parameters must be finely tuned to express high frequency functions.

While universal approximation properties of neural networks have been known since the early 90s (Hornik et al., 1989; BID6 Leshno et al., 1993; BID2 , recent research has shed light on the mechanisms underlying such expressivity (Montufar et al., 2014; Raghu et al., 2016; Poole et al., 2016) .

At the same time, deep neural networks, despite being massively overparameterized, have been remarkably successful at generalizing to natural data.

This fact is at odds with the traditional notions of model complexity and their empirically demonstrated ability to fit arbitrary random data to perfect accuracy (Zhang et al., 2017a; BID1 .

It has prompted the recent investigations of possible implicit regularization mechanisms inherent in the learning process, inducing biases towards low complexity solutions (Soudry et al., 2017; Poggio et al., 2018; Neyshabur et al., 2017) .In this work, our main goal is to expose one such bias by taking a closer look at neural networks through the lens of Fourier analysis 1 .

We focus the discussion on ReLU networks, whose piecewise linear structure enables an analytic treatment.

While they can approximate arbitrary functions, we find that these networks favour low frequency ones; in other words, they exhibit a bias towards smooth functions, a phenomenon we call the spectral bias 2 .

We find that this bias manifests itself not just in the process of learning, but also in the parameterization of the model itself: in fact we show that the lower frequencies of trained networks are more robust with respect to random parameter perturbations.

Finally, we also exhibit and analyze a rather intricate interplay between the spectral bias and the geometry of the data manifold: we show that high frequencies get easier to learn when the data lies on a lower dimensional manifold of complex shape embedded in the input space.

CONTRIBUTIONS 1.

We exploit the piecewise-linear structure of ReLU networks to evaluate and bound its Fourier spectrum.2.

We demonstrate the peculiar behaviour of neural networks with illustrative and minimal experiments and find evidence of a spectral bias: i.e. lower frequencies are learned first.

1 The Fourier transform affords a natural way of measuring how fast a function can change within a small neighborhood in its input of a model.

See Appendix B for a brief recap of Fourier analysis.2 A similar result has been independently found and reported in Xu et al. (2018).

3.

We illustrate how the manifold hypothesis adds a layer of subtlety by showing how the geometry of the data manifold attenuates the spectral bias in a non-trivial way.

We present a theoretical analysis of this phenomenon and derive conditions on the manifolds that facilitate learning higher frequencies.4.

Given a trained network, we investigate the relative robustness of the lower frequencies with respect to random perturbations of the network parameters.

The paper is organized as follows.

In Section 2, we derive the Fourier spectrum of deep ReLU networks.

Section 3 presents minimal experiments that demonstrate the spectral bias of ReLU networks.

In Section 4, we study and discuss the role of the geometry of the data manifold.

In Section 5, we empirically illustrate and theoretically explain our robustness result.

Consider the class of scalar functions f : R d → R defined by a ReLU network with L hidden layers of widths d 1 , · · · d L and a single output neuron: DISPLAYFORM0 where each T (k) : ReLU networks are known to be continuous piece-wise linear (CPWL) functions, where the linear regions are convex polytopes (Raghu et al., 2016; Montufar et al., 2014; Zhang et al., 2018; BID0 .

Remarkably, the converse is also true: every CPWL function can be represented by a ReLU network BID0 , Theorem 2.1), which in turn endows ReLU networks with universal approximation properties.

Given the ReLU network f from Eqn.

1, we can make the piecewise linearity explicit by writing, DISPLAYFORM1 DISPLAYFORM2 where is an index for the linear regions P and 1 P is the indicator function on P .

As shown in Appendix C in more detail, each region corresponds to an activation pattern 3 of all hidden neurons of the network, which is a binary vector with components conditioned on the sign of the input of the respective neuron.

The 1 × d matrix W is given by DISPLAYFORM3 where W (k) is obtained from the original weight W (k) by setting its j th column to zero whenever the neuron j of the k th layer is inactive.

We will henceforth assume that the input data lies in a bounded domain of DISPLAYFORM4 for some A > 0 and thus restrict ourselves to ReLU networks with bounded support 4 .

In the following, we study the structure of ReLU networks in the Fourier domain, which is defined as: DISPLAYFORM0 where dx, dk are the uniform Lebesgue measure on R d andf denotes the Fourier transform of f (see Appendix B for a short recap of the Fourier transform).

Lemmas 1 and 2 (proved in appendix D) yield the explicit form of the Fourier components.

Lemma 1.

The Fourier transform of ReLU networks decomposes as, DISPLAYFORM1 where k = k and1 P (k) = P e −ik·x dx is the Fourier transform of the indicator function of P .The Fourier transform of a polytope appearing in Eqn.

5 is a fairly intricate mathematical object; BID8 develop an elegant procedure for evaluating it in arbitrary dimensions via a recursive application of Stokes theorem.

We describe this procedure in detail in Appendix D.2, and present here its main corollary.

Lemma 2.

Let P be a full dimensional polytope in R d .

The Fourier spectrum of its indicator function1 P satisfies the following: DISPLAYFORM2 where DISPLAYFORM3 Note that since a polytope has a finite number of facets (of any dimension), the k's for which ∆ (P ) k = j for some j < d lie on a finite union of j-dimensional subspaces of R d .

The Lebesgue measure of all such lower dimensional subspaces for all such j equals 0, leading us to the conclusion that the spectrum decays as DISPLAYFORM4 Lemmas 1, 2 together yield the main result of this section.

Given a ReLU network f , its linear regions form a cell decomposition of R d as union of polytopes; we denote by F the set of faces (of any dimension) of all such polytopes.

For k ∈ R d , let ∆ k be the minimum integer 1 ≤ j ≤ d such that k lies orthogonal to some (d − j)-dimensional face in F. Theorem 1.

The Fourier components of the ReLU network f satisfy the following: DISPLAYFORM5 where N f is the number of linear regions and L f = max W 2 is the Lipschitz constant of f .

(a) The spectral decay of ReLU networks is highly anisotropic in large dimensions.

In almost all directionsk of DISPLAYFORM0 However, the decay can be as slow as O(k −2 ) in specific directions orthogonal to the facets bounding linear regions 5 .

As we prove in Appendix D.3, the Lipschitz constant L f can be bounded as, DISPLAYFORM1 where · is the spectral norm, W is the ravelled parameter vector of the network and · ∞ is the max-norm.

This makes the bound on L f scale exponentially in depth and polynomial in width.

As for the number N f of linear regions, Montufar et al. (2014) and Raghu et al. (2016) obtain tight bounds that exhibit the same scaling behaviour (Raghu et al., 2016, Theorem 1) .

This makes the overall bound in Eqn.

7 -and with it, the ability to express larger frequencies -scale exponentially in depth and polynomially in width 6 .

This result complements the well-known universal approximation property of neural networks by explicitly incorporating a control on the capacity 7 of the network, namely the width, depth and the norm of parameters.

Architecture dependent controls show the measured amplitude of the network spectrum at the corresponding frequency, normalized by the target amplitude at the same frequency (i.e. |f k i |/Ai) and the colorbar is clipped between 0 and 1.

Right (a, b): Evolution of the spectral norm (y-axis) of each layer during training (x-axis).

Figure-set (a) shows the setting where all frequency components in the target function have the same amplitude, and (b) where higher frequencies have larger amplitudes.

Gist: We find that even when higher frequencies have larger amplitudes, the model prioritizes learning lower frequencies first.

We also find that the spectral norm of weights increases as the model fits higher frequency, which is what we expect from Theorem 1.

on approximation have been formalized in the literature through approximation bounds and depth separation results, see e.g Barron (1993); Telgarsky (2016); BID10 .

(b) For a given architecture (i.e. fixed width and depth), the high frequency contributions of the network can be increased by increasing the norm of the parameters.

Assuming the weight norm increases with training iterations, this suggests that the training of ReLU networks might be biased towards lower frequencies.

We investigate this fact empirically in the next section.

In this section and the ones that follow, we present experiments that illustrate the peculiar behaviour of deep ReLU networks in the Fourier domain.

We begin with an experiment to demonstrate that networks tend to fit lower frequencies first during training.

We refer to this phenomenon as the spectral bias, and discuss it in light of the results of Section 2.

Experiment 1.

The setup is as follows 8 : Given frequencies κ = (k 1 , k 2 , ...) with corresponding amplitudes α = (A 1 , A 2 , ...), and phases φ = (ϕ 1 , ϕ 2 , ...), we consider the mapping λ : DISPLAYFORM0 8 More experimental details and additional plots are provided in Appendix A.1.A 6-layer deep 256-unit wide ReLU network f θ is trained to regress λ with κ = (5, 10, ..., 45, 50) and N = 200 input samples spaced equally over [0, 1] ; its spectrumf θ (k) in expectation over ϕ i ∼ U (0, 2π) is monitored as training progresses.

In the first setting, we set equal amplitude A i = 1 for all frequencies and in the second setting, the amplitude increases from A 1 = 0.1 to A 10 = 1.

FIG2 the normalized magnitudes |f θ (k i )|/A i at various frequencies, as training progresses.

The result is that lower frequencies (i.e. smaller k i 's) are regressed first, regardless of their amplitudes.

Discussion.

Multiple theoretical aspects may underlie these observations.

First, for a fixed architecture, the bound in Theorem 1 allows for larger Fourier coefficients at higher frequencies if the parameter norm is large.

However, the parameter norm can increase only gradually during training by gradient descent, which leads to the higher frequencies being learned late in the optimization process.

To confirm that the bound indeed increases as the model fits higher frequencies, we plot in FIG2 DISPLAYFORM1 where the second equality follows from Plancherel theorem.

We make two observations -first, the square error in input space translates into square error in Fourier domain, with a priori no structural bias towards any particular frequency component 9 , i.e. all frequencies are weighted the same.

Since the same cannot be said about e.g. cross-entropy loss, we use the MSE loss in most of our experiments to avoid a potential confounding factor.

Second, the parameterization of the network can be exploited by considering the gradient of the MSE loss w.r.t.

parameters, DISPLAYFORM2 where Re(z) denotes the real part of z. We find that a bias naturally emerges as a consequence of the spectral decay rate found in Theorem 1, in the sense that the magnitude of the residual |f θ (k)−λ(k)| contributes less to the net gradient for large k. This generalizes the argument made in Xu (2018) for two layer sigmoid networks by observing that the gradient w.r.t parameters of the network function inherits the spectral decay rate of the function itself 10 .

In Section 5, we use that the integral off θ w.r.t.

the standard measure in parameter space dθ also inherits the spectral decay rate of f to make a statement about the robustness off θ (k) against random parameter perturbations.

In this section, we investigate the subtleties that arise when the data lies on a lower dimensional manifold embedded in the higher dimensional input space of the model BID11 .

We find that the shape of the data-manifold impacts the learnability of high frequencies in a nontrivial way.

As we shall see, this is because low frequencies functions in the input space may have high frequency components when restricted to lower dimensional manifolds of complex shapes.

To systematically investigate the impact of manifold shape on the spectral bias, we demonstrate results in an illustrative minimal setting 11 free from unwanted confounding factors.

We also present a mathematical exposition of the relationship between the Fourier spectrum of the network, the spectrum of the target function defined on the manifold, and the geometry of the manifold itself.

9 Note that the finite range in frequencies is due to sampling 10 Observe that the partial derivative w.r.t.

θ can be swapped with the integrals in Eqn 4.

11 We include experiments on MNIST and CIFAR-10 in appendices A.4 and A.5.

Manifold hypothesis.

We consider the case where the data lies on a lower dimensional data manifold M ⊂ R d embedded in input space, which we assume to be the image DISPLAYFORM0 Under this hypothesis and in the context of the standard regression problem, a target function τ : M → R defined on data manifold can identified with a function λ = τ • γ defined on the latent space.

Regressing τ is therefore equivalent to finding f : DISPLAYFORM1 Further, assuming that the data probability distribution µ supported on M is induced by γ from the uniform distribution U in the latent space [0, 1] m , the mean square error can be expressed as, DISPLAYFORM2 Observe that there is a vast space of degenerate solutions f that minimize the mean squared errornamely all functions on R d that yield the same function when restricted to the data manifold M.Our findings from the previous section suggest that neural networks are biased towards expressing a particular subset of such solutions, namely those that are low frequency.

It is also worth noting that there exist methods that restrict the space of solutions: notably adversarial training BID12 and Mixup (Zhang et al., 2017b) .Experimental set up.

The experimental setting is designed to afford control over both the shape of the data manifold and the target function defined on it.

We will consider the family of curves in R 2 generated by mappings DISPLAYFORM3 Here, γ L ([0, 1]) defines the data-manifold and corresponds to a flower-shaped curve with L petals, or a unit circle when L = 0 (see e.g. FIG3 .

Given a signal λ : [0, 1]

→ R defined on the latent space [0, 1], the task entails learning a network f : DISPLAYFORM4 Experiment 2.

The set-up is similar to that of Experiment 1, and λ is as defined in Eqn.

9 with frequencies κ = (20, 40, ..., 180, 200) , and amplitudes DISPLAYFORM5 with N = 1000 uniformly spaced samples z i between 0 and 1.

Experiment 3.

Here, we adapt the setting of Experiment 2 to binary classification by simply thresholding the function λ at 0.5 to obtain a binary target signal.

To simplify visualization, we only use DISPLAYFORM6 DISPLAYFORM7 (e) Loss curves Figure 3 : (a,b,c,d ): Evolution of the network spectrum (x-axis for frequency, colorbar for magnitude) during training (y-axis) for the same target functions defined on manifolds γL for various L. Since the target function has amplitudes Ai = 1 for all frequencies ki plotted, the colorbar is clipped between 0 and 1.

(e): Corresponding learning curves.

Gist: Some manifolds (here with larger L) make it easier for the network to learn higher frequencies than others.

signals with a single frequency mode k, such that λ(z) = sin(2πkz + ϕ).

We train the same network on the resulting classification task with cross-entropy loss 12 for k ∈ {50, 100, ..., 350, 400} and L ∈ {0, 2, ..., 18, 20}. The heatmap in Fig Observe that increasing L (i.e. going up a column in FIG5 results in better (classification) performance for the same target signal.

This is the same behaviour as we observed in Experiment 2 (Fig 3a-d) , but now with binary cross-entropy loss instead of the MSE.Discussion.

These experiments hint towards a rich interaction between the shape of the manifold and the effective difficulty of the learning task.

The key technical reason underlying this phenomenon (as we formalize below) is that the relationship between frequency spectrum of the network f and that of the fit f • γ L is mediated by the embedding map γ L .

In particular, we will argue that a given signal defined on the manifold is easier to fit when the coordinate functions of the manifold embedding itself has high frequency components.

Thus, in our experimental setting, the same signal embedded in a flower with more petals can be captured with lower frequencies of the network.

To understand this mathematically, we address the following questions: given a target function λ, how small can the frequencies of a solution f be such that f • γ = λ?

And further, how does this relate to the geometry of the data-manifold M induced by γ?

To find out, we write the Fourier transform of the composite function, DISPLAYFORM8 The kernel P γ depends on only γ and elegantly encodes the correspondence between frequencies k ∈ R d in input space and frequencies l ∈ R m in the latent space [0, 1] m .

Following a procedure from Bergner et al., we can further investigate the behaviour of the kernel in the regime where the stationary phase approximation is applicable, i.e. when l 2 + k 2 → ∞ (cf.

section 3.2.

of Bergner et al.) .

In this regime, the integral P γ is dominated by critical pointsz of its phase, which satisfy DISPLAYFORM9 Non-zero values of the kernel correspond to pairs (l, k) such that Eqn 15 has a solution.

Further, given that the components of γ (i.e. its coordinate functions) are defined on an interval [0, 1] m , one can use their Fourier series representation together with Eqn 15 to obtain a condition on their frequencies (shown in appendix D.4).

More precisely, we find that the i-th component of the RHS in Eqn 15 is proportional to pγ i [p]k i where p ∈ Z m is the frequency of the coordinate function γ i .

This yields that we can get arbitrarily large frequencies l i ifγ i [p] is large 13 enough for large p, even when k i is fixed.

12 We use Pytorch's BCEWithLogitsLoss.

Internally, it takes a sigmoid of the network's output (the logits) before evaluating the cross-entropy.

13 Consider that the data-domain is bounded, implying thatγ cannot be arbitrarily scaled.

This is precisely what Experiments 2 and 3 demonstrate in a minimal setting.

From Eqn 13, observe that the coordinate functions have a frequency mode at L. For increasing L, it is apparent that the frequency magnitudes l (in the latent space) that can be expressed with the same frequency k (in the input space) increases with increasing L. This allows the remarkable interpretation that the neural network function can express large frequencies on a manifold (l) with smaller frequencies w.r.t its input domain (k), provided that the coordinate functions of the data manifold embedding itself has high-frequency components 14 .

The goal of this section is to show that lower frequency components of trained networks are more robust than their higher frequency counterparts with respect to random perturbations in parameter space.

More precisely, we observe that in the neighbourhood of a solution in parameter space, the high frequency components decay faster than the low frequency ones.

This property does not directly depend on the training process, but rather on the parametrization of the trained model.

We present empirical evidence and a theoretical explanation of this phenomenon.

Experiment 4.

The set up is the same as in Experiment 1, where λ is given by Eqn.

9.

Training is performed for the frequencies κ = (10, 15, 20, ..., 45, 50) and amplitudes A i = 1 ∀ i.

After convergence to θ * , we consider random (isotropic) perturbations θ = θ * + δθ of given magnitude δ, whereθ ∼ U (S dim(θ * ) ) is a unit vector.

We evaluate the network function f θ at the perturbed parameters, and compute the magnitude of its discrete Fourier transform at frequencies k i , |f θ (k i )|.

We also average over 100 samples ofθ to obtain |f Eθ (k i )|, which we normalize by |f θ * (k i )|.

The result, shown in FIG7 , demonstrate that higher frequencies are significantly less robust than the lower ones.

Discussion.

The interpretation is as follows: parameters that contribute towards expressing highfrequency components occupy a small volume in the parameter space.

To formalize this intuition, given a bounded domain Θ of parameter space, let us define, DISPLAYFORM0 to be the set of parameters such that f θ has Fourier components larger than for some k with larger norm than k.

Then the following Proposition holds (proved in appendix E).

Proposition 1.

The volume ratio, DISPLAYFORM1 inherits the spectral decay rate of |f θ (k)|, given by Theorem 1.Intuitively, expressing larger frequencies requires the parameters to be finely-tuned to work together.

While we focus on showing the spectral bias of deep ReLU networks towards learning functions with dominant lower frequency components, most of existing work has focused on showing that in theory, these networks are capable of learning arbitrarily complex functions.

Hornik et al. (1989) ; BID6 ; Leshno et al. (1993) have shown that neural networks can be universal approximators when given sufficient width; more recently, Lu et al. (2017) proved that this property holds also for width-bounded networks.

Montufar et al. (2014) showed that the number of linear regions of deep ReLU networks grows polynomially with width and exponentially with depth; Raghu et al. (2016) generalized this result and provided asymptotically tight bounds.

There have been various results of the benefits of depth for efficient approximation (Poole et al., 2016; Telgarsky, 2016; BID10 .

These analysis on the expressive power of deep neural networks can in part explain why over-parameterized networks can perfectly learn random input-output mappings (Zhang et al., 2017a) .

Our Fourier analysis of deep ReLU networks also reflects the width and depth dependence of their expressivity, but more interestingly reveals their spectral bias towards learning simple functions.

Thus our work may be seen as a formalization of the findings of BID1 , where it is empirically shown that deep networks prioritize learning simple functions during training.

A few other works studied neural networks through the lens of harmonic analysis.

In light of our findings, it is worth comparing the case of neural networks and other popular algorithms such that kernel machines (KM) and K-nearest neighbor classifiers.

We refer to the Appendix F for a detailed discussion and references.

In summary, our discussion there suggests that 1.

DNNs strike a good balance between function smoothness and expressivity/parameter-efficiency compared with KM; 2.

DNNs learn a smoother function compared with KNNs since the spectrum of the DNN decays faster compared with KNNs in the experiments shown there.

We studied deep ReLU networks through the lens of Fourier analysis.

Several conclusions can be drawn from our analysis.

While neural networks can approximate arbitrary functions, we find that they favour low frequency ones -hence they exhibit a bias towards smooth functions -a phenomenon that we called spectral bias.

We also illustrated how the geometry of the data manifold impacts expressivity in a non-trivial way, as high frequency functions defined on complex manifolds can be expressed by lower frequency network functions defined in input space.

Finally, we found that the parameters contributing towards expressing lower frequencies are more robust to random perturbations than their higher frequency counterparts.

We view future work that explore the properties of neural networks in Fourier domain as promising.

For example, the Fourier transform affords a natural way of measuring how fast a function can change within a small neighborhood in its input domain ; as such, it is a strong candidate for quantifying and analyzing the sensitivity of a model -which in turn provides a natural measure of complexity (Novak et al., 2018) .

We hope to encourage more research in this direction.

We fit a 6 layer ReLU network with 256 units per layer f θ to the target function λ, which is a superposition of sine waves with increasing frequencies: DISPLAYFORM0 where k i = (5, 10, 15, ..., 50), and ϕ i is sampled from the uniform distribution U (0, 2π).

In the first setting, we set equal amplitude for all frequencies, i.e. A i = 1 ∀ i, while in the second setting we assign larger amplitudes to the higher frequencies, i.e. A i = (0.1, 0.2, ..., 1).

We sample λ on 200 uniformly spaced points in [0, 1] and train the network for 80000 steps of full-batch gradient descent with Adam (Kingma & Ba, 2014) .

Note that we do not use stochastic gradient descent to avoid the stochasticity in parameter updates as a confounding factor.

We evaluate the network on the same 200 point grid every 100 training steps and compute the magnitude of its (single-sided) discrete fourier transform at frequencies k i which we denote with |f ki |.

Finally, we plot in figure 1 the normalized magnitudes |f k i | Ai averaged over 10 runs (with different sets of sampled phases ϕ i ).

We also record the spectral norms of the weights at each layer as the training progresses, which we plot in figure 1 for both settings (the spectral norm is evaluated with 10 power iterations).

In FIG10 , we show an example target function and the predictions of the network trained on it (over the iterations), and in figure 7 we plot the loss curves.

We use the same 6-layer deep 256-unit wide network and define the target function where k i = (20, 40, ..., 180, 200), A i = 1 ∀ i and ϕ ∼ U (0, 2π).

We sample φ on a grid with 1000 uniformly spaced points between 0 and 1 and map it to the input domain via γ L to obtain a dataset DISPLAYFORM0 DISPLAYFORM1 , on which we train the network with 50000 full-batch gradient descent steps of Adam.

On the same 1000-point grid, we evaluate the magnitude of the (single-sided) discrete Fourier transform of f θ • γ L every 100 training steps at frequencies k i and average over 10 runs (each with a different set of sampled z i 's).

Fig 3 shows

Theorem 1 exposes the relationship between the fourier spectrum of a network and its depth, width and max-norm of parameters.

The following experiment is a qualitative ablation study over these variables.

Experiment 5.

In this experiment, we fit various networks to the δ-function at x = 0.5 (see FIG13 .

Its spectrum is constant for all frequencies FIG13 , which makes it particularly useful for testing how well a given network can fit large frequencies.

FIG2 We make the following observations.(a) FIG14 shows that increasing the depth (for fixed width) significantly improves the network's ability to fit higher frequencies (note that the depth increases linearly).(b) FIG2 shows that increasing the width (for fixed depth) also helps, but the effect is considerably weaker (note that the width increases exponentially).(c) FIG2 shows that increasing the weight clip (or the max parameter max-norm) also helps the network fit higher frequencies.

Figure 12: Evolution with training iterations (y-axis) of the network prediction (x-axis for input, and colormap for predicted value) for a network with varying weight clip, depth = 6 and width = 64.

The target function is a δ peak at x = 0.5.The above observations are all consistent with Theorem 1, and further show that lower frequencies are learned first (i.e. the spectral bias, cf.

Experiment 1).

In the following experiment, we show that given two manifolds of the same dimension -one flat and the other not -the task of learning random labels is harder to solve if the input samples lie on the same manifold.

We demonstrate on MNIST under the assumption that the manifold hypothesis is true, and use the fact that the spectrum of the target function we use (white noise) is constant in expectation, and therefore independent of the underlying coordinate system when defined on the manifold.

Experiment 6.

In this experiment, we investigate if it is easier to learn a signal on a more realistic data-manifold like that of MNIST (assuming the manifold hypothesis is true), and compare with a flat manifold of the same dimension.

To that end, we use the 64-dimensional feature-space E of a denoising 15 autoencoder as a proxy for the real data-manifold of unknown number of dimensions.

The decoder functions as an embedding of E in the input space X = R 784 , which effectively amounts to training a network on the reconstructions of the autoencoder.

For comparision, we use an injective embedding 16 of a 64-dimensional hyperplane in X. The latter is equivalent to sampling 784-dimensional vectors from U ([0, 1]) and setting all but the first 64 components to zero.

The target function is white-noise, sampled as scalars from the uniform distribution U ([0, 1]).

Two identical networks are trained under identical conditions, and FIG2 shows the resulting loss curves, each averaged over 10 runs.

This result complements the findings of BID1 and Zhang et al. (2017a) , which show that it's easier to fit random labels to random inputs if the latter is defined on the full dimensional input space (i.e. the dimension of the flat manifold is the same as that of the input space, and not that of the underlying data-manifold being used for comparison).

A.5 CIFAR-10: IT'S ALL CONNECTED We have seen that deep neural networks are biased towards learning low frequency functions.

This should have as a consequence that isolated bubbles of constant prediction are rare.

This in turn implies that given any two points in the input space and a network function that predicts the same class for the said points, there should be a path connecting them such that the network prediction does not change along the path.

In the following, we present an experiment where we use a path finding method to find such a path between all Cifar-10 input samples indeed exist.

Experiment 7.

Using AutoNEB Kolsbjerg et al. (2016) , we construct paths between (adversarial) Cifar-10 images that are classified by a ResNet20 to be all of the same target class.

AutoNEB bends 15 This experiment yields the same result if variational autoencoders are used instead.

16 The xy-plane is R 3 an injective embedding of a subset of FIG2 : Path between CIFAR-10 adversarial examples (e.g. "frog" and "automobile", such that all images are classified as "airplane").

DISPLAYFORM0 a linear path between points in some space R m so that some maximum energy along the path is minimal.

Here, the space is the input space of the neural network, i.e. the space of 32 × 32 × 3 images and the logit output of the ResNet20 for a given class is minimized.

We construct paths between the following points in image space:• From one training image to another,• from a training image to an adversarial,• from one adversarial to another.

We only consider pairs of images that belong to the same class c (or, for adversarials, that originate from another class = c, but that the model classifies to be of the specified class c).

For each class, we randomly select 50 training images and select a total of 50 random images from all other classes and generate adversarial samples from the latter.

Then, paths between all pairs from the whole set of images are computed.

The AutoNEB parameters are chosen as follows: We run four NEB iterations with 10 steps of SGD with learning rate 0.001 and momentum 0.9.

This computational budget is similar to that required to compute the adversarial samples.

The gradient for each NEB step is computed to maximize the logit output of the ResNet-20 for the specified target class c. We use the formulation of NEB without springs BID9 .The result is very clear: We can find paths between all pairs of images for all CIFAR10 labels that do not cross a single decision boundary.

This means that all paths belong to the same connected component regarding the output of the DNN.

This holds for all possible combinations of images in the above list.

FIG2 shows connecting training to adversarial images and FIG2 paths between pairs of adversarial images.

Paths between training images are not shown, they provide no further insight.

Note that the paths are strikingly simple: Visually, they are hard to distinguish from the linear interpolation.

Quantitatively, they are essentially (but not exactly) linear, with an average length (3.0 ± 0.3)% longer than the linear connection.

The Fourier transform is a powerful mathematical tool used to represent functions as a weighted sum of oscillating functions, given that the function satisfies certain conditions.

In the realm of signal processing and beyond, it is used to represent a time (space) domain signal f as a sum of sinusoids of various (spatial) frequencies k, where the weights are referred to as the Fourier coefficientsf (k).

FIG2 : Each row is a path through the image space from an adversarial sample (right) to a true training image (left).

All images are classified by a ResNet-20 to be of the class of the training sample on the right with at least 95% softmax certainty.

This experiment shows we can find a path from adversarial examples (right, Eg. "(cat)") that are classified as a particular class ("airplane") are connected to actual training samples from that class (left, "airplane") such that all samples along that path are also predicted by the network to be of the same class.

Let f : R n → R be a squared-integrable function 17 , i.e. such that DISPLAYFORM0 With the Fourier inversion theorem, it holds: DISPLAYFORM1 Informally, equation 17 expresses the function f (x) as a weighted sum (the integral) of plane waves e ±ik·x of the angular wavenumber k, where the unit vectork gives the direction of the corresponding wave in n-D space and the magnitude k is inversely proportional to the wavelength.

Equation FORMULA0 gives the expression forf (k), which is called the Fourier transform or the Fourier spectrum or simply the spectrum of f .

The 1 2π coefficient and sign in the exponential functions are matters of convention.

The asymptotic behaviour off for k → ∞ is a measure of smoothness of f .

In Bachmann-Landau or asymptotic notation 18 , we sayf = O(k −1 ) if for k → ∞, the functionf decays at least as fast as 1 k .

A function whose spectrum is O(k −2 ) is in a sense smoother than one whose spectrum is O(k −1 ), while the spectrum of an infinitely differentiable (or smooth) function must decay faster than any rational function of k, assuming the function is integrable, i.e. the integral of its absolute value over its domain is finite (or the function is L 1 ).

Intuitively, the higher-frequency oscillations in a smoother function must vanish faster.

Formally, this is a straightforward consequence of the Riemann-Lebesgue lemma, stating that the spectrum of any L 1 function must vanish at infinity (potentially arbitrarily slowly), taken together with the well known property of the Fourier transform that it diagonalizes the differential operator i.e. DISPLAYFORM2 17 On a formal note, the squared integrability is only required for the inverse Fourier transform to exist; for the forward transform, integrability is enough.

Moreover, the Fourier transform can be generalized to tempered distributions, which allow for evaluating the Fourier coefficients of non-integrable e.g. non-zero constant functions.18 Formally, DISPLAYFORM3 < ∞.1.

If Proj F (k) = 0, then φ k (x) = Φ k is constant on F , and we have: DISPLAYFORM4

The above theorem provides a recursive relation for computing the Fourier transform of an arbitrary polytope.

More precisely, the Fourier transform of a m-dimensional polytope is expressed as a sum of fourier transforms over the m − 1 dimensional boundaries of the said polytope (which are themselves polytopes) times a O(k −1 ) weight term (with k = k ).

The recursion terminates if Proj F (k) = 0, which then yields a constant.

To structure this computation, BID8 introduce a book-keeping device called the face poset of the polytope.

It can be understood as a weighted tree diagram with polytopes of various dimensions as its nodes.

We start at the root node which is the full dimensional polytope P (i.e. we initially set m = n).

For all of the codimension-one boundary faces F of P , we then draw an edge from the root P to node F and weight it with a term given by: DISPLAYFORM0 and repeat the process iteratively for each F .

Note that the weight term is O(k −1 ) where Proj F (k) = 0.

This process yields tree paths T : P → F 1 → ...

→ F q where each F i+1 ∈ ∂F i has one dimension less than F i .

For a given path and k, the terminal node for this path, F q , is the first polytope for which Proj Fq (k) = 0.

The final Fourier transform is obtained by multiplying the weights along each path and summing over all tree paths: DISPLAYFORM1 where we wrote F 0 = P .

Together with Lemma 1, this gives the closed form expression of the Fourier transform of ReLU networks.

For a generic vector k, all paths terminate at the zero-dimensional vertices of the original polytope, i.e. dim(F q ) = 0, implying the length of the path q equals the number of dimensions d, yielding a O(k −d ) spectrum.

The exceptions occur if a path terminates prematurely, because k happens to lie orthogonal to some d − r-dimensional face F r in the path, in which case we are left with a O(k −r ) term (with r < d) which dominates asymptotically.

Note that all vectors orthogonal to the d − r dimensional face F r lie on a r-dimensional subspace of R d .

Since a polytope has a finite number of faces (of any dimension), the k's for which the Fourier transform is O(k −r ) (instead of O(k −d )) lies on a finite union of closed subspaces of dimension r (with r < d).

The Lebesgue measure of all such lower dimensional subspaces for all such r is 0, leading us to the conclusion that the spectrum decays as O(k −d ) for almost all k's.

We formalize this in the following corollary.

Corollary 1.

Let P be a full dimensional polytope in R n .

The Fourier spectrum of its indicator function1 P satisfies the following: DISPLAYFORM2 where 1 ≤ ∆ k ≤ n, and ∆ k = j for k on a finite union of j-dimensional subspaces of R n .

Proposition 2.

The Lipschitz constant L f of the ReLU network f is bound as follows (for all ): DISPLAYFORM0 Proof.

The first equality is simply the fact that L f = max W , and the second inequality follows trivially from the parameterization of a ReLU network as a chain of function compositions 19 , together with the fact that the Lipschitz constant of the ReLU function is 1 (cf.

Miyato et al. (2018) , equation 7).

To see the third inequality, consider the definition of the spectral norm of a I × J matrix W : DISPLAYFORM1 Now, W h = i |w i · h|, where w i is the i-th row of the weight matrix W and i = 1, ..., I. Further, if h = 1, we have |w i · h| ≤ w i h = w i .

Since w i = j |w ij | (with j = 1, ..., J) and |w ij | ≤ θ ∞ , we find that DISPLAYFORM2 IJ θ ∞ and we obtain: .

The general idea is to investigate the behaviour of P γ (l, k) for large frequencies l on manifold but smaller frequencies k in the input domain.

In particular, we are interested in the regime where the stationary phase approximation is applicable to P γ , i.e. when l 2 +k 2 → ∞ (cf.

section 3.2.

of Bergner et al.) .

In this regime, the integrand in P γ (k, l) oscillates fast enough such that the only constructive contribution originates from where the phase term u(z) = k · γ(z) − l · z does not change with changing z. This yields the condition that ∇ z u(z) = 0, which translates to the condition (with Einstein summation convention implied and ∂ ν = ∂ /∂xν): DISPLAYFORM3 DISPLAYFORM4 Now, we impose periodic boundary conditions 20 on the components of γ, and without loss of generality we let the period be 2π.

Further, we require that the manifold be contained in a box 21 of some size in R d .

The µ-th component γ µ can now be expressed as a Fourier series: DISPLAYFORM5 Equation FORMULA3 can be substituted in equation 32 to obtain: DISPLAYFORM6 where we have split k µ and l ν in to their magnitudes k and l and directionsk ν andl µ (respectively).We are now interested in the conditions on γ under which the RHS can be large in magnitude, even when k is fixed.

Recall that γ is constrained to a box -consequently, we can not arbitrarily scale upγ µ .

However, ifγ µ [p] decays slowly enough with increasing p, the RHS can be made arbitrarily large (for certain conditions on z,l µ andk ν ).E VOLUME IN PARAMETER SPACE AND PROOF OF PROPOSITION 1For a given neural network, we now show that the volume of the parameter space containing parameters that contribute -non-negligibly to frequency components of magnitude k above a certain cut-off k decays with increasing k. For notational simplicity and without loss of generality, we absorb the directionk of k in the respective mappings and only deal with the magnitude k.

Definition 1.

Given a ReLU network f θ of fixed depth, width and weight clip K with parameter vector θ, an > 0 and Θ = BDeep networks on the other hand are also capable of approximating any target function (as shown by the universal approximation theorems Hornik et al. (1989); BID6 ), but they are also parameter efficient in contrast to KM.

For instance, we have seen that deep ReLU networks separate the input space into number of linear regions that grow polynomially in width of layers and exponentially in the depth of the network (Montufar et al., 2014; Raghu et al., 2016) .

A similar result on the exponentially growing expressive power of networks in terms of their depth is also shown in (Poole et al., 2016) .

In this paper we have further shown that DNNs are inherently biased towards lower frequency (smooth) functions over a finite parameter space.

This suggests that DNNs strike a good balance between function smoothness and expressibility/parameter-efficiency compared with KM.

K-nearest neighbor (KNN) also has a historical importance as a classification algorithm due to its simplicity.

It has been shown to be a consistent approximator BID7 , i.e., asymptotically its empirical risk goes to zero as K → ∞ and K/N → 0, where N is the number of training samples.

However, because it is a memory based algorithm, it is prohibitively slow for large datasets.

Since the smoothness of a KNN prediction function is not well studied, we compare the smoothness between KNN and DNN.

For various values of K, we train a KNN classifier on a k = 150 frequency signal (which is binarized) defined on the L = 20 manifold (see section 4), and extract probability predictions on a box interval in R 2 .

On this interval, we evaluate the 2D FFT and integrate out the angular components to obtain ζ(k): DISPLAYFORM0 Finally, we plot ζ(k) for various K in FIG2 .

Furthermore, we train a DNN on the very same dataset and overlay the radial spectrum of the resulting probability map on the same plot.

We find that while DNN's are as expressive as a K = 1 KNN classifier at lower (radial) frequencies, the frequency spectrum of DNNs decay faster than KNN classifier for all values of K considered, indicating that the DNN is smoother than the KNNs considered.

We also repeat the experiment corresponding to FIG5 with KNNs (see FIG2 ) for various K's, to find that unlike DNNs, KNNs do not necessarily perform better for larger L's, suggesting that KNNs do not exploit the geometry of the manifold like DNNs do.

@highlight

We investigate ReLU networks in the Fourier domain and demonstrate peculiar behaviour.

@highlight

Fourier analysis of ReLU network, finding that they are biased towards learning low frequency 

@highlight

This paper has theoretical and empirical contributions on topic of Fourier coefficients of neural networks