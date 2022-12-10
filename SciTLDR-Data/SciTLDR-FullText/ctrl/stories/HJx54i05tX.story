We study the behavior of weight-tied multilayer vanilla autoencoders under the assumption of random weights.

Via an exact characterization in the limit of large dimensions, our analysis reveals interesting phase transition phenomena when the depth becomes large.

This, in particular, provides quantitative answers and insights to three questions that were yet fully understood in the literature.

Firstly, we provide a precise answer on how the random deep weight-tied autoencoder model performs “approximate inference” as posed by Scellier et al. (2018), and its connection to reversibility considered by several theoretical studies.

Secondly, we show that deep autoencoders display a higher degree of sensitivity to perturbations in the parameters, distinct from the shallow counterparts.

Thirdly, we obtain insights on pitfalls in training initialization practice, and demonstrate experimentally that it is possible to train a deep autoencoder, even with the tanh activation and a depth as large as 200 layers, without resorting to techniques such as layer-wise pre-training or batch normalization.

Our analysis is not specific to any depths or any Lipschitz activations, and our analytical techniques may have broader applicability.

The autoencoder is a cornerstone in machine learning, first as a response to the unsupervised learning problem (Rumelhart & Zipser (1985) ), then with applications to dimensionality reduction (Hinton & Salakhutdinov (2006) ), unsupervised pre-training (Erhan et al. (2010) ), and also as a precursor to many modern generative models (Goodfellow et al. (2016) ).

Its reconstruction power is well utilized in applications such as anomaly detection (Chandola et al. (2009) ) and image recovery (Mousavi et al. (2015) ).

With the surge of deep learning, thousands of papers have studied multilayer variants of this architecture, but theoretical understanding has been limited, since analyzing the learning dynamics of a highly nonlinear structure is typically a difficult problem even for the shallow autoencoder.

To get around this, we tackle the task with a critical assumption: the weights are random and the autoencoder is weight-tied.

One enjoys much analytical tractability from the randomness assumption, whereas weight tying enforces the random autoencoder to perform "autoencoding".

We also study this in the high-dimensional setting, where all dimensions are comparably large and ideally jointly approaching infinity.

We consider the simplest setting: vanilla autoencoders (i.e., ones with fully connected layers only) and their reconstruction capability.

This is done for the sake of understanding the effect of depth, while we note our techniques may have broader applicability.

The aforementioned assumptions are not without justifications.

There is a growing literature on deep neural networks with random weights, (Li & Saad (2018) ; Giryes et al. (2016) ; Poole et al. (2016) ; Schoenholz et al. (2016) ; Gabrié et al. (2018) ; Amari et al. (2018) ) to name a few, revealing certain properties of deep feedforward networks 1 .

Several recent works have also studied random multilayer feedforward networks through the lens of statistical inference (Manoel et al. (2017) ; Reeves (2017); Fletcher et al. (2018) ).

The idea of weight tying is considered in the important paper Vincent et al. (2010) with an empirical finding that autoencoders with and without weight tying perform comparably, and has become standard in autoencoders.

Similar features of random connection and symmetry also appear in other neural models (Lillicrap et al. (2016) ; Scellier et al. (2018) ).

Finally the high-dimensional setting is common in recent statistical learning advances (Bühlmann & Van De Geer (2011) ), and not too far from the actual practice where many large datasets have dimensions of at least a few hundreds and are harnessed by large-scaled models.

We seek quantitative answers to three specific questions that are motivated by previous works:• In exactly what way does the (vanilla) random weight-tied autoencoder perform "approximate inference"?

This term is coined in Scellier et al. (2018) in connection with the theoretical results in Arora et al. (2015) , which implicitly studies the said model.

In particular, Arora et al. (2015) proves an upper bound on x − x 2 , where x andx are the input and the output of the network, but is limited in the number of layers and specific to the ReLU activation.

This direction has been recently extended by Gilbert et al. (2017) .

In our work, we establish precisely what this approximate inference is by obtaining a general and asymptotically exact characterization 2 ofx, for any number of layers and any Lipschitz continuous activations (Theorem 1 and Section 3.3).

Theorem 1 is the key theoretical result of our work and lays the foundation for all analyses that follow.• In what way is the deep autoencoder different from the shallow counterpart?

Li & Saad (2018) ; Poole et al. (2016) reveal this in terms of the candidate function space and expressivity for feedforward networks.

It is unclear how these notions are applicable to weighttied autoencoders, which seek replication of the input rather than a generic mapping.

In this work, we show that the deep autoencoder exhibits a higher order of sensitivity to perturbations of the parameters (Section 3.4).

Burkholz & Dubatovka (2018) demonstrate a connection between the study of random networks, or ones at initialization, and their trainability.

Note that these works either do not study weight-tied structures, or assume the analysis of the untying case for weight-tied structures.

In our work, we derive and experimentally verify insights on how (not) to initialize deep weight-tied autoencoders, demonstrating that it is possible to train them without resorting to techniques such as greedy layer-wise pretraining, drop-out and batch normalization (Section 3.5).

Specifically we experiment with 200-layer autoencoders.

No prior works have attempted all three tasks.

The quantitative difference between weight-tied and weight-untied networks is in fact not negligible, yet the analysis is non-trivial due to the weight tying constraint (Arora et al. (2015) ; Chen et al. (2018) ).

To address this issue and obtain Theorem 1, we apply the Gaussian conditioning technique, which first appears in the studies of TAP equations in spin glass theory (Bolthausen (2014) ) and is extensively used in the approximate message passing algorithm literature (Bayati & Montanari (2011); Javanmard & Montanari (2013) ; Berthier et al. (2017) ).

This should be contrasted with untied random networks, whose analysis is typically more straightforward.

More importantly, the difference is not only analytical: the overall picture of deep random weight-tied autoencoders is rich and drastically different from that of feedforward networks.

An analysis in the limit of infinite depth reveals three fundamental equations governing the picture (Section 3.1), which displays multiple phase transition phenomena (Section 3.2) .

Consider the following 2L-layers autoencoder with weight tying: DISPLAYFORM0 Here x ∈ R n0 is the input, W ∈ R n ×n −1 is the weight, b ∈ R n is the encoder bias, and v ∈ R n −1 is the decoder bias, for = 1, ..., L. Also ϕ : R → R and σ : R → R are the activations (where for a vector u ∈ R n and a function ϕ : R → R, we write ϕ (u) to denote the vector (ϕ (u 1 ) , ..., ϕ (u n )) ).

It is usually the case in practice that σ 0 (u) = u the identity function.

We introduce some convenient quantities inductively: FIG6 of Appendix A.1 for a schematic diagram.

We assume weights are random.

Specifically we generate the weights and biases according to DISPLAYFORM1 DISPLAYFORM2 independently of each other.

The scaling of the variances accords with the literature and actual practice (Glorot & Bengio (2010); Vincent et al. (2010) ).

We also consider the asymptotic highdimensional regime, indexed by n: DISPLAYFORM3 Here σ W, , σ b, , σ v, and α are finite constants independent of n. We enforce σ W, > 0, but allow σ b, and σ v, to be zero.

We assume that all activations are Lipschitz continuous, and the encoder activations σ 's are non-trivial in the sense that for any τ > 0, E z σ (τ z)

> 0 where z ∼ N (0, 1).

We also assume that 1 n0 σ 0 (x) 2 tends to a finite and strictly positive constant as n → ∞. We refer to Appendix A.1 for more clarifications of notations.

We motivate our main result via a simplified shallow autoencoder: DISPLAYFORM0 Noticex is a sum of independent terms, and by Stein's lemma (cf.

Appendix E.2), E {x} ∝ E m i=1 ϕ w i x /m x. One thus expectŝ x ≈ c 1 x + c 2 z for scalars c 1 , c 2 and z ∼ N (0, I n ), for large n and m.

It is then important to specify exactly c 1 and c 2 .

Theorem 1 formalizes this intuition with precise formulas for the scalars.

We now define some scalar sequences, which will then be related to the (vector) quantities of the autoencoders in Theorem 1.

First we define {τ } =1,...,L and {τ } =0,...,L inductively: DISPLAYFORM1 for z ∼ N (0, 1).

Next, we define {γ , ρ } =2,...,L+1 inductively: DISPLAYFORM2 for z 1 , z 2 ∼ N (0, 1) independently.

With these sequences defined, we can state the main theorem.

Its statement uses the relational operator ∼ =, which is defined formally in Appendix A.1.

Roughly speaking, a n ∼ = b n means a n and b n are asymptotically equal in distribution as n → ∞. Theorem 1.

Consider the settings and assumptions as in Section 2.1, and the sequences {τ ,τ } and {γ , ρ } defined as above.

Then in the limit n → ∞:(a) {τ } describes the behavior of the encoder output x : DISPLAYFORM3 (b) {τ , γ , ρ } describes the behavior of the decoder outputx : DISPLAYFORM4 for z 1 , z 2 ∼ N 0, I n −1 independently.

One can replaceτ −1 z 1 with x −1 in the above, with z 2 independent of x −1 , in which case the statement also holds for = 1 with x 0 = x.(c) For the autoencoder's outputx, DISPLAYFORM5 The proof of the theorem, as well as an outline of the key ideas, are in Appendix A.

The theorem says that x ,x andx admit simple descriptions which are tracked by scalar sequences {τ , γ , ρ }.

Hence we can learn about the autoencoder by analyzing {τ , γ , ρ }, which is generally a simpler task than studying x ,x andx directly.

Numerical simulations in Appendix B suggest that, although the theorem's statement is in the infinite dimension limit, the agreement is already good for dimensions of a few hundreds.

We note that while the theorem assumes Gaussian biases, the same proof technique allows to obtain a similar result with a more relaxed condition on the biases.

Remark 2.

While the theorem is specific to W following the Gaussian distribution, simulations in Appendix B suggest that the conclusion holds for a much broader class of distributions.

We conjecture that it should hold so long as each W has i.i.d.

entries and is independent of each other, its distribution has bounded k-th moment for some sufficiently large k, and the activations as well as the input x satisfy certain mild regularity conditions.

Remark 3.

We comment on the range of ρ and γ .

We have ρ ≥ 0, which is obvious, and if DISPLAYFORM6 .

If the activations are non-decreasing, then γ ≥ 0.

Furthermore, if the activations are Lipschitz, then |γ | ≤ Cc for some constants C and c.

In the following, we adopt a semi-rigorous approach, with an emphasis on the overall picture.

We make several analytical simplifications.

First consider α = α > 0, σ DISPLAYFORM0 b ≥ 0, ϕ = ϕ and σ = σ all independent of , except for ϕ L which is chosen separately (but we shall see that the specific choice of ϕ L is largely immaterial).

We also assume that σ 2 v, = 0, and σ 0 and ϕ 0 are the identity 4 .

We introduce a parameterτ ≥ 0, whose role will be clear shortly, and which satisfies: DISPLAYFORM1 4 The assumption σ Figure 2: The mapping γ → G (γ, ρ) forτ 2 = 1 and β = 5 (blue), β = 2.7 (red), β = 0.8 (green).

The color intensity varies with ρ ∈ [0.1, 1] with equal spacings, where the darkest curve corresponds to ρ = 0.1, and the lightest is ρ = 1.

From left to right: ϕ, σ are ReLU; ϕ is ReLU, σ is tanh; ϕ, σ are tanh; ϕ is tanh, σ is ReLU.

A fixed point is an intersection between this mapping and the identity line (black dashed).which case Eq. (3) also has a solution, for instance, when ϕ (0) = 0 such as ReLU or tanh (which admits ρ = 0).

However γ = 0 is trivial, since it implies S sig is zero.

We will be interested in the existence of non-trivial and stable fixed points.

To ease visualization, for the moment, let us consider Eq. (2) only.

Fig. 2 shows γ → G (γ, ρ) for different ρ, β, ϕ and σ forτ 2 = 1.

For a given ϕ and σ, depending on β and ρ, one may observe one or more fixed points, one of which is at γ = 0 and can be stable or unstable.

When γ = 0 is the only fixed point but is unstable, we have γ = ∞ as the "stable solution" to Eq. (2).

The solution landscape changes drastically with β; for instance, when σ = ϕ = tanh, γ = 0 is the only and stable fixed point when β is small, but it becomes unstable and a new fixed point at γ > 0 emerges when β is sufficiently large.

This hints at certain phase transition behaviors as β varies.

In Appendix C.2, we perform a detailed analysis of Eq. (2) and (3), supported by several rigorously proven properties.

In the following, by an initialization for γ and ρ, we mean γ L+1 and ρ L+1 as in Section 2.2, and by convergence to γ and ρ, we mean the convergences as in Section 3.1.

We highlight some results from the analysis for specific pairs of ϕ and σ:ReLU ϕ and σ.

We have two phase transitions at β = 2 and at β = 4.

When β < 2, with any initialization, we have convergence to γ = 0 and ρ = 0.

When 2 < β < 4, we have, with certain initializations, convergence to γ = 0 and divergence to ρ = +∞, and with certain other initializations, divergence to γ = +∞ and ρ = +∞. These include almost all possible initializations.

When β ≥ 4, with any non-zero initialization, we have divergence to γ = +∞ and ρ = +∞.ReLU ϕ and tanh σ.

We have two phase transitions at β = 2 and β = β 0 (τ ) ∈ (2, ∞).

When β < 2 , with any initialization, we have convergence to γ = 0 and ρ = 0.

When 2 < β < β 0 , with any non-zero initialization, we have convergence to γ = 0 and divergence to ρ = +∞. When β > β 0 , with any non-zero initialization, we have divergence to γ = +∞ and hence ρ = +∞. tanh ϕ and σ.

We have two phase transitions at β = 1 and β = β 0 (τ ) > 1.

When β ≤ 1, we have convergence to γ = ρ = 0.

When 1 < β < β 0 , with any non-zero initialization, we have convergence to γ = 0 and ρ ∈ (0, 1).

When β > β 0 , with any non-zero initialization, we have convergence to γ > 0 and ρ ∈ (0, 1).

Forτ > 0, γ cannot grow to +∞ as β varies.

We note that β 0 → 1 ifτ 2 → 0, and in the case α = 1, this implies σ 2 W → 1.

With respect to Eq. (1), we then have σ 2 b → 0.

An illustration is given in FIG3 .

tanh ϕ and ReLU σ.

We have a picture similar to the case ϕ = σ = tanh, with a crucial difference that one cannot have β 0 be close to 1.

An illustration is given in FIG3 .

γ and ρ thus exhibit phase transitions, depend crucially on the choice of activations (especially the decoder activation ϕ), and can be trivialized (i.e., being zero or infinity) as in the case of ReLU ϕ and σ.

It is remarkable that the above pictures are general for many other activations, as suggested by our analysis.

In the next sections, we explore the implications of these behaviors.

Theorem 1 gives a quantitatively exact sense of how the random weight-tied autoencoder performs "approximate inference".

Here we will be interested in stronger notions.

A first question is: does it explain reversibility?

Reversibility, as mathematically formalized in previous works Arora et al. (2015) ; Gilbert et al. (2017) , quantitatively concerns with how small the quantity E = x −x 2 /n 0 is.

The smaller it is, the better the decoder "reverses" the encoder.

This formalized notion is an attempt to give a theoretical understanding of empirical findings that the input could be reproduced from the values of hidden layers of a trained feedforward network.

Let us now consider the infinite depth simplification under Interpretation 2 of Section 3.1.

We have E (S sig − 1) 2τ 2 0 + S 2 var .

As such, for E ≈ 0 with high probability, one must have S sig ≈ 1 and S var ≈ 0, hence ρ ≈ 0.

DISPLAYFORM0 = 0 is impossible for any non-trivial activations (unless the activation outputs zero almost everywhere).

Strikingly, in light of Section 3.2, when ϕ and σ are both ReLU, we have that S sig and S var are either 0 or +∞, in which case E ≥τ 2 0 and can become unbounded.

While this does not contradict the results in Arora et al. FORMULA12 (which also concerns with ReLU activations, but with specific choices of the biases and limited depth, and hence is in a different setting), our discussion suggests that random weight-tied models may be insufficient to explain reversibility.

A second question is: does the model perform signal recovery?

In this case, we are interested in whetherx ∼ = cx for some constant c not necessarily 1.

Similar to the above, this requires S var = 0, hence ρ = 0, and E ϕ (βγσ (τ z 1 )) 2 = 0.

For non-trivial ϕ and σ, this requires ϕ (0) = 0 and γ = 0.

Many activations do not conform with the former, and the latter impliesx ∼ = 0 undesirably.

This provides a negative answer to the question.

A critic may argue that in expectation, E {x} ≈ S sig x, and as per Section 3.2, there are cases where γ > 0 and hence S sig > 0.

Yet in fact, this in-expectation property can already be observed in the simple setting of linear shallow autoencoders (Arora et al. FORMULA12 ).

What is ignored in such argument is that in many cases, S var > 0 whenever S sig > 0, in light Section 3.2.

Our analysis hence mitigates the shortcoming of the in-expectation approach, and gives a more precise understanding of what the random weight-tied autoencoder can and cannot achieve when the depth becomes large.

Our result also allows for the case of L = 1 a shallow autoencoder.

In particular, taking a parallel setting with Section 3.1 (in particular, Interpretation 2), by Theorem 1, x ∼ = S sig x + S var z, S sig = βγ, S var = βρ, in which, with ϕ hid being the activation in the hidden layer, DISPLAYFORM0 Some observations follow.

In the shallow case, γ > 0 and ρ > 0 and both are bounded regardless of the parameters, except for trivial edge cases such asτ 2 = 0 or ϕ hid (·) = 0.

Furthermore, γ and ρ are independent of β, for a fixedτ .

As such, there is no phase transition in γ and ρ as β changes.

We also have S sig (β) ∝ β and S var (β) ∝ √ β.

Hence, the signal component dominates with S sig /S var ∝ √ β.

Again this happens regardless of parameter choices.

In comparison with the infinite depth case, for ϕ = σ = tanh or ϕ = tanh and σ being the ReLU, as observed from FIG3 , in certain regimes, γ, ρ and γ/ √ ρ can grow (sublinearly) with β, and hence S sig (β) = Ω (β) 6 , S var (β) = Ω √ β , and the signal component dominates with DISPLAYFORM1 In particular, near the phase transition of γ, S sig /S var = Ω β 1.5 .

Recalling that β = ασ 2 W , this implies for the infinite depth case, as compared to the shallow one, firstly a slight perturbation in σ 2 W may result in a larger perturbation in the signal's strength, and secondly an architecture using larger α may gain more in terms of amplification of the signals.

In short, the deep autoencoder is more sensitive to slight changes in the parameters.

As evident in Section 3.2, the case ϕ being the ReLU also exhibits extreme sensitivity, in that it is possible for a slight perturbation in β to drastically change γ and ρ.

As suggested by Fig. 1 , it should be the case already for L about a few tens.

It is, however, at the expense of much care in the selection of parameters, since there are continuous regimes in which the infinite depth diminishes S sig and S var to zero or boost them to infinity, a situation that never occurs in the shallow case.

Remark 4.

Sensitivity to perturbations is implied by expressivity, a notion put forth in Poole et al. (2016) in the study of random feedforward networks.

Hence we expect that sensitivity is a common feature of various types of deep neural networks.

We examine the implications of Interpretation 1 in Section 3.1 to trainability of the weight-tied autoencoder.

We first state our hypothesis, then test it with experiments.

The hypothesis.

Since the majority of intermediate layers can be described approximately by γ and ρ (as well asτ ) and the random weight-tied autoencoder is in fact one at initialization, appropriate values of γ, ρ andτ (by a suitable choice of σ 2 W and σ 2 b ) should lead to better trainability.

In particular, if one of them is ∞, we expect numerical errors or too large values resulting in quick saturation, both of which render the autoencoder untrainable.

If γ = ρ = 0 in a neighborhood of the chosen σ 2 W and σ 2 b , we expect that the progress is slowed down in the beginning.

If such pitfalls are avoided, the autoencoder is expected to show a faster progress.

Our analysis in Section 3.2 shows that those pitfalls can occur for a wide range of parameters.

If the hypothesis is true, σ 2 W and σ 2 b must then be chosen carefully when L is large.

As a special remark on the case ϕ is the ReLU, when α = 1, the hypothesis suggests taking σ FORMULA12 ), which however considers feedforward networks only.

Interestingly this does require σ to be the ReLU; for instance, σ can be tanh, in which case the argument in (He et al. FORMULA12 ) is not applicable.

We shall also examine edge of chaos (EOC) initializations (Schoenholz et al. (2016) ; Pennington et al. FORMULA12 ) (see Appendix E.1).

The EOC initialization enables better signal propagation in deep feedforward networks, and in our context, is relevant to the encoder part with the activation σ.

Table 1 : List of initialization schemes for each pair of ϕ and σ, for α = 1.

Here "−" indicates a positive finite value that depends on the choice of ϕ L (for which we choose the ReLU), but its exact value is irrelevant for our purpose.

"EOC" indicates whether the scheme is an EOC initialization with respect to σ, and "xx" indicates an EOC scheme that is found to be the better one among all EOC initializations with Gaussian weights (Pennington et al. FORMULA12 ).

"Trainable" indicates better trainability in the beginning as predicted by our theory.

"Slowed" indicates γ = ρ = 0 in a neighborhood.

"Inf" indicates either γ → ∞ or ρ → ∞. The schemes with ϕ = tanh should be reflected against FIG3 .Experiments.

Table 1 lists several initialization schemes with α = 1, which are chosen such that the hypothesis can be tested separately for each pair ϕ and σ.

We perform simple experiments on a weight-tied vanilla autoencoder as described in Section 3.1: L = 100, all hidden dimensions of 400, identity input activation σ 0 , and decoder biases initialized to zero.

This sets α = α = 1 for ≥ 2; here α 1 = 1 is irrelevant in light of Interpretation 1.

We train the autoencoder on the MNIST dataset with mini-batch gradient descent with a batch size of 250 and without regularizations, for 5 × 10 5 iterations (equivalent to 2500 epochs).

We perform the experiments in two settings:• Setting 1: The output activation ϕ 0 is tanh, MNIST images are normalized to [−1, +1], and the learning rate is fixed at 5 × 10 −3 .

This is standard for MNIST.• Setting 2: ϕ 0 is the identity, MNIST images are unnormalized (i.e., normalized to [0, +1]), and the learning rate is fixed at 3 × 10 −3 .

This is common for regression.

These learning rates are chosen so that the learning dynamics is typically smooth, in light of recent works Mei et al. FORMULA12 ; Smith & Le (2018) .

We use the normalized 2 2 loss x − x 2 / x 2 , and are primarily interested in this loss as a quality measure, since we only focus on trainability 7 .

We also do not apply techniques such as greedy layer-wise pre-training, drop-out or batch normalization.

The results are plotted in FIG5 .

See also Appendix D.1 for visualization of the reconstructions, and Appendix D.2 for the evolution over a broader range of parameters.

Note that we plot the evolution in the logarithmic scale of time, since it is typically smooth and revealing on this scale, as found in prior works Baity-Jesi et al. (2018); Mei et al. (2018) and also evident from the plots.

Discussion.

The results are in good agreement with our hypothesis.

(Recall we test the hypothesis separately for each pair ϕ and σ, for which the involved schemes share the same architecture and only differ in the initialization.)

Note that as predicted, in Setting 2, Scheme 3 and 6 are trapped with numerical errors, and in Setting 1, they saturate quickly at a high loss.

As such, we do not include the results of Scheme 3 and 6 in FIG5 .

7 The chosen loss is slightly different from the traditional 2 2 loss x − x 2 .

On one hand, we found from our experiments that these two losses perform comparably, with the normalized loss typically yielding slight improvements, provided that the learning rates are scaled appropriately.

On the other hand, the normalized loss allows ease for interpretation.

2 / x 2 of the schemes from Table 1 .

Left: the setting with ϕ 0 = tanh (Setting 1).

Right: the setting where ϕ 0 is the identity (Setting 2).We see from the figure that Scheme 2, 5, 8, 10 and 14 show much slower progresses, by a factor of 3 to 10 times in terms of training iterations to reach the same loss.

Hence a good amount of training time can be saved by an appropriate initialization.

Interestingly Scheme 5 is in fact a special EOC initialization that Pennington et al. (2017) found to be the better one among all EOC schemes with Gaussian weights for tanh activation.

This last observation shows that having good signal propagation through the encoder is far from being a sufficient condition for trainability.

Among the schemes, only Scheme 1 and 4 in Setting 1 and only Scheme 1, 4 and 7 in Setting 2 have their eventual trained networks produce meaningful reconstructions, whereas the rest always output some "average" of the training set regardless of the input, at the end of 5 × 10 5 iterations (see Appendix D.1).

It is unclear whether this is a bad local minimum, or whether these schemes take much longer to show further progresses.

An explanation is beyond our current theory, and it is an open question how to create a scheme with meaningful trainability.

Remarkably all the schemes that show slower initial progresses (Scheme 2, 5, 8, 10 and 14) are among those that could not yield meaningful reconstructions.

We observe that in Setting 2, the tanh network under Scheme 7 is best performing in terms of the reconstruction loss, and its progress does not seem to reach a plateau after 5 × 10 5 iterations.

In both settings, Scheme 4, which is a hybrid of ReLU and tanh activations, shows slight improvements over Scheme 1, which is a purely ReLU network.

This extends the conclusion in Pennington et al. (2017) to the context of weight-tied autoencoders: reasonable training at a large depth is possible even for the notoriously difficult tanh activation, and this necessarily requires careful initializations.

Overall we see that our experiments confirm the hypothesis, showing an intimate connection between the phase transition behaviors found by our theory and trainability of the autoencoders.

This paper has shown quantitative answers to the three questions posed in Section 1.

This feat is enabled by an exact analysis via Theorem 1.

The theorem is stated in a general setting, allowing varying activations, weight variances, etc, but our analyses in Section 3 have made several simplifications.

This leaves a question of whether these simplifications can be relaxed, and how the picture changes accordingly, for instance, when the parameters vary across layers, similar to Yang & Schoenholz (2018) .

Many other questions also remain.

For example, what would be the covariance structure between the outputs of two distinct inputs?

How does the network's Jacobian matrix look like?

These questions have been answered in the feedforward case (Poole et al. (2016) ; Pennington et al. FORMULA12 ), but we believe answering them is more technically involved in our case.

We have also seen that an autoencoder that shows initial progress may not necessarily produce meaningful reconstruction eventually after training, and hence much more work is needed to understand the training dynamics far beyond initialization.

Recent works Mei et al. FORMULA12 In the following, we give an outline of the proof of Theorem 1, and the complete proof.

First, we start with a few notations and definitions.

DISPLAYFORM0

We recall the setting in Section 2.1 (see also FIG6 for a schematic diagram of the autoencoder).

We define {g : DISPLAYFORM0 ..,L inductively as follows: DISPLAYFORM1 It is easy to see thatx = g (x −1 ) for = 1, ..., L. Essentially g 's represent the autoencoding mappings computed by the inner layers.

We use bold-face letters (e.g. x, W ) to denote vectors or matrices.

We use C throughout to denote an arbitrary (and immaterial) constant that is independent of the dimensions.

For two vectors x and y, x, y denotes their inner product.

We use I for the identity matrix, and I n to emphasize its dimensions n × n.

We use · to denote the usual Euclidean norm, · ∞ the infinity norm, and M 2 the maximum singular value of a matrix M .

For a sigma-algebra F and two random variables X and Y , X|F d = Y means that for any integrable function φ and any F-measurable DISPLAYFORM2 A sequence of functions φ n : R n → R is uniformly pseudo-Lipschitz if there exists a constant C, independent of n, such that for any x, y ∈ R n , DISPLAYFORM3 A sequence of functions φ n : R n → R n is uniformly Lipschitz if there exists a constant C, independent of n, such that for any x, y ∈ R n , φ n (x) − φ n (y) ≤ C x − y .

These definitions are adopted from Berthier et al. (2017) .

For two sequences of random variables X n ∈ R and Y n ∈ R indexed by n, we write X n Y n to mean that X n − Y n → 0 in probability.

The same meaning holds when Y n is deterministic.

For two sequences of random vectors X n ∈ R n and Y n ∈ R n indexed by n, we write X n ∼ = Y n if for any sequences of uniformly pseudo-Lipschitz test functions φ n : R n → R, φ n (X n ) E {φ n (Y n )} (and hence in this context, we do not need X n and Y n to be defined on a joint probability space).

We state several results that are key to prove Theorem 1.

Proposition 5.

Consider the asymptotic setting n → ∞, with some sequence m = m (n) such that m/n → α > 0 as n → ∞. Let ϕ : R → R and σ : R → R be Lipschitz continuous scalar functions.

Consider a sequence of uniformly Lipschitz functions g : R m → R m , and sequences of DISPLAYFORM0 for all sufficiently large n. Let us define DISPLAYFORM1 forz ∼ N (0, I m ).

Then: DISPLAYFORM2 DISPLAYFORM3 for z, z ∼ N (0, I n ) independently.

In particular, for φ : R → R Lipschitz continuous, DISPLAYFORM4 DISPLAYFORM5 in which DISPLAYFORM6 wherez ∼ N (0, I m ), and z 1 , z 2 are independently distributed as N (0, 1).In a nutshell, the proposition and the corollary consider a random weight-tied "autoencoder" with a single hidden layer, with a mapping g in the middle.

Since g is not a separable function (i.e., g does not apply entry-wise), this is different from the usual shallow autoencoder, a case that has been investigated in Pennington & Worah (2017); Louart et al. FORMULA12 with techniques and objectives different from ours.

By understanding this structure, one can understand the random weight-tied multi-layer autoencoder.

Indeed, for each , the proposition applies with u being x −1 , f (u) beinĝ x and g being g +1 .

In other words, this studies the mapping g .

One can start with g L , then progressively move to outer layers DISPLAYFORM7 , etc, and hence analyze the autoencoder completely by repeating the same procedure L times.

We note that in doing so, at each step, one requires certain information about the inner layers to perform calculations for the outer layers, and this is worked out in Corollary 6 via a simple recursive relation.

In particular, the left-hand sides of Eq. (4) and Eq. (5) play the role of ρ and γ of the outer layer, whereas their right-hand sides involve ρ and γ from the inner layers.

It is also important to note that the assumption that the weights at different layers are independent is crucial in making u, g and (W , b, v) to be independent, allowing the proposition to be applicable at all steps.

This is the idea behind the proof of Theorem 1.We quickly mention the key proof technique for Proposition 5.

The main technical challenge in working with the weight-tied structure W h (W u), for some h : R m → R m and W ∈ R m×n Gaussian, is that whereas y = W u is Gaussian with zero mean thanks to independence between u and W , W h (y) is not since y is correlated with W .

It is observed in Bolthausen FORMULA12 that conditioning on a linear constraint y = W u (or the sigma-algebra F generated by y and u), one has that W is distributed as a conditional projection component plus an independent Gaussian component: DISPLAYFORM8 is an appropriate projection.

Here E {W |F} is what propagates the information about u. In addition,W is Gaussian and independent of F, and hence exact calculations can be then worked out.

We note that the assumption W is Gaussian is crucial for this identity to hold.

We state the Gaussian Poincaré inequality, which will be used multiple times throughout the proof.

We remark that the use of the Gaussian Poincaré inequality is unlikely to lead to a tight nonasymptotic result, but is sufficient for our asymptotic analysis.

Theorem 7 (Gaussian Poincaré inequality).

For z ∼ N 0, σ 2 I n and φ : R n → R continuous and weakly differentiable, there exists a universal constant C such that Var {φ (z)} ≤ Cσ 2 E ∇φ (z) 2 .Now we are ready for the proof.

Step 1.

We perform Gaussian conditioning.

Let y = W σ (u).

Let F be the sigma-algebra generated by y and u. Conditioning on F is equivalent to conditioning on the linear constraint y = W σ (u).

Following Bayati & Montanari (2011), we have DISPLAYFORM0 2 the projection onto σ (u), and P ⊥ σ(u) = I −P σ(u) the corresponding orthogonal projection.

As such, since ϕ (g (y + b)) is F-measurable, DISPLAYFORM1 For a sequence of uniformly pseudo-Lipschitz functions φ n : R n → R: DISPLAYFORM2 Up to this point, there is no need for the asymptotics n → ∞. The rest of the proof focuses on Φ n .Step 2.

We show that τ , γ and ρ are uniformly bounded as n → ∞. This is trivial for τ , and we note τ > 0.

Consider ρ.

We have for any r ∈ R m , DISPLAYFORM3 for sufficiently large m.

It is then easy to see that ρ is uniformly bounded.

Regarding γ, DISPLAYFORM4 by Cauchy-Schwarz inequality.

Therefore γ is also uniformly bounded.

Step 3.

We analyze the first term in Φ n .

Notice that y d = τz for somez ∼ N (0, I m ).

In addition, the mapping y → y, ϕ (g (y + b)) /m is uniformly pseudo-Lipschitz, by noticing that DISPLAYFORM5 along with the fact that y → ϕ (g (y + b)) is uniformly Lipschitz, and Eq. (7).

As such, by Theorem 7, DISPLAYFORM6 which tends to 0 as n → ∞. By Chebyshev's inequality, we thus have DISPLAYFORM7 Step 4.

We analyze the second term in Φ n .

We have: DISPLAYFORM8 Note that the mapping y → DISPLAYFORM9 Recall thatW is independent of F, and as such, there exists a random variable z ∼ N (0, 1) independent of y such that DISPLAYFORM10 which converges to 0 in probability, where we have used the fact that ρ is uniformly bounded fromStep 2.

Furthermore, there also exists a random vector z ∼ N (0, I n ) independent of F such that DISPLAYFORM11 Step 5.

We finish the proof.

From the definition of uniform pseudo-Lipschitz functions and Eq. FORMULA38 and FORMULA41 , we obtain: DISPLAYFORM12 Here notice that P ⊥ σ(u) 2 = 1, and as a standard fact from the random matrix theory, W 2 ≤ c α, σ 2 W a constant with high probability (see e.g. Vershynin FORMULA12 ).

Furthermore, 1 √ n z ≤ C with high probability due to the law of large numbers.

Combining with the facts from Step 2, 3 and 4, it is then easy to see that DISPLAYFORM13 Finally, since γ and ρ are uniformly bounded from Step 2, it is easy to show that the mapping DISPLAYFORM14 is uniformly pseudo-Lipschitz.

Hence by Theorem 7, DISPLAYFORM15 which yields DISPLAYFORM16 Together with Eq. FORMULA32 and FORMULA12 , this completes the proof.

By Proposition 5, for any sequence of uniformly pseudo-Lipschitz φ n : R n → R: DISPLAYFORM0 is uniformly pseudo-Lipschitz, and hence by Theorem 7, DISPLAYFORM1 Performing a similar argument for ρ (τ , b), we thus have: DISPLAYFORM2 We note by Stein's lemma, DISPLAYFORM3

and hence γ γ.

Therefore, DISPLAYFORM0 The mapping (u, v) → h (u, v) is uniformly pseudo-Lipschitz and hence by Theorem 7, DISPLAYFORM1 which follows from the fact that u, v and z are independent and normally distributed.

The first claim is hence proven.

Eq.

(4) follows immediately from the above.

The proof of Eq. FORMULA26 is similar and hence omitted.

The following lemma states that, roughly speaking, g is uniformly Lipschitz (indexed by n) with high probability.

First, recall that W = W (n) is indexed by n, and one can easily construct a joint probability space on which the sequence {W (n)} n≥1 is defined.

The exact construction is immaterial, so long as for each n, the marginal distribution satisfies the setting in Section 2.1.

We work in this joint space in the following.

Lemma 8.

For each = 1, ..., L, there exists a finite constant c > 0 such that P E N → 0 as N → ∞, where the event E N is defined as FORMULA12 ).

As such, using this fact for = L, by the union bound, DISPLAYFORM2 DISPLAYFORM3 which proves the claim for = L. To see the claim for general , we have from a similar calculation: DISPLAYFORM4 Therefore, DISPLAYFORM5 by the union bound.

This proves the claim.

Lemma 9.

For each ≥ 0, τ +1 andτ +1 are finite and strictly positive.

Furthermore, DISPLAYFORM6 , and x +1 ∼ =τ +1 z for z ∼ N 0, I n +1 .Proof.

We prove the lemma by induction.

The claim is trivially true for = 0 by assumption.

Assume the claim for some ≥ 0.

Due to independence, conditioning on x , DISPLAYFORM7 , we then have DISPLAYFORM8 for z ∼ N (0, 1).

By the induction hypothesis,τ +1 > 0.

Hence by assumption, the right-hand side is finite and strictly positive.

One also recognizes that it is equal to τ 2 +2 /σ 2 W, +2 .

Since σ W, +2 is strictly positive and finite, so is τ +2 , and hence so isτ +2 .

This completes the proof.

We are now ready for the proof of Theorem 1.

Claim (a) of the theorem, which follows directly from Lemma 9, is just a forward pass through the encoder and hence is the same as in the case of random feedforward networks (without weight tying) Poole et al. (2016) .

For = 2, ..., L, let H denote the following three claims: DISPLAYFORM9 DISPLAYFORM10 DISPLAYFORM11 for z 1 , z 2 , ∼ N 0, I n −1 independently, andz −1 ∼ N 0, I n −1 independent of g , with the understanding that g L+1 is the identity.

We prove them by induction, and Claim (b) then follows immediately.

We first note that with high probability, for any , DISPLAYFORM12 bounded by the law of large numbers.

Consider H L .

Note that x L−1 is independent of g L , and that DISPLAYFORM13 2 converges to a finite non-zero constant in probability by Lemma 9.

Hence by Corollary 6, with g being the identity, f being DISPLAYFORM14 and φ being ϕ L−1 , and the fact x L−1 ∼ =τL−1z −1 from Claim (a), H L is proven.

Assuming H +1 for some , we prove H .

Recall that g +1 is independent of Θ = {W , b , v , x −1 ,z −1 }, and g is independent of x −1 andz −1 .

Also from Claim (a), x −1 ∼ = τ −1z −1 .

Consider some N ∈ N and let E N +1 denote the event as defined in Lemma 8.

On the event ¬E N +1 (the complement of E N +1 ), by Corollary 6, with respect to the randomness of Θ , DISPLAYFORM15 Note that here γ and ρ are functions of g +1 and hence random, and z ∼ N (0, I n ) independent of g +1 .

On the event ¬E N +1 , with the fact thatτ is finite and non-zero by Lemma 9, we have the mapping DISPLAYFORM16 is uniformly pseudo-Lipschitz, and hence applying Theorem 7, one obtains that DISPLAYFORM17 with the right-hand side independent of g +1 for n > N .

Consequently, due to Lemma 8 and Chebyshev's inequality, for any > 0, DISPLAYFORM18 where o n (1) → 0 as n → ∞. On the other hand, since z is independent of g +1 , invoking H +1 , DISPLAYFORM19 As such, DISPLAYFORM20 Letting n → ∞ then N → ∞, we then have γ γ +1 .

Similarly, we also have ρ ρ +1 .

It is then easy to deduce H , recalling the definitions of γ and ρ .The claim thatτ −1 z 1 can be replaced with x −1 in the expression forx in Claim (b) can be recognized easily by doing the same replacement in the proof of Corollary 6 and the above proof.

The proof of Claim (c) is similar.

We omit these repetitive steps.

We perform simple simulations to verify Theorem 1 at finite dimensions.

In particular, we simulate a random weight-tied autoencoder, as described in Section 2.1, with L = 50, σ 2 W, = 2.312, σ 2 b, = 0.211, σ 2 v, = 0, ϕ = σ = tanh, identity ϕ 0 and σ 0 , α = 1, and consequently n = n 0 , for an input x ∈ R n0 whose first half of the entries are +1 and the rest are −1.

Then we compute the following:γ DISPLAYFORM0 We also compute {γ , ρ } =2,...,L+1 as in Section 2.2, and γ and ρ as in Section 3.1.

Theorem 1 predicts thatγ γ andρ ρ .

Section 3.1 also asserts that γ ≈ γ and ρ ≈ ρ for L. Fig. 6 shows the results for n 0 = 500 and n 0 = 2000.

We observe a quantitative agreement already for n 0 = 500, and this improves with larger n 0 .Next we verify the normality of the variation component inx.

We computê DISPLAYFORM1 Its empirical distribution should be close to N (0, 1), in light of Theorem 1.

We make this comparison in FIG11 , and again observe good agreement already for n 0 = 500.Finally we re-simulate the autoencoder with different distributions of the weights W .

In particular, we try with the Bernoulli distribution, the uniform distribution, and the Laplace distribution, with their means and variances adjusted to zero and σ 2 W, /n −1 respectively.

The results for n 0 = 2000 are plotted in FIG12 .

We observe good quantitative agreements between the simulations for these non-Gaussian distributions and the prediction as in Theorem 1, although the theorem is proven only for Gaussian weights.

Figure 6 : The agreement amongγ , γ and γ, and amongρ , ρ and ρ, for = 2, ..., 51 and Gaussian weights.

The setting is described in Appendix B.

We take a single run for the simulation of the autoencoder.

Here n 0 = 500 (left) and n 0 = 2000 (right).

Figure 8 : The agreement amongγ , γ and γ, and amongρ , ρ and ρ, for = 2, ..., 51, for different distributions of the weights.

The setting is described in Appendix B.

We take a single run for the simulation of the autoencoder.

Here n 0 = 2000.

Computing γ and ρ as from Eq. (2) and (3) amounts to evaluating double integrals.

For simple ϕ, such as the ReLU, the integrals f ϕ (a, b) = E {ϕ (a + bz)} and g ϕ (a, b) = E ϕ (a + bz) 2 , for z ∼ N (0, 1), can be calculated in closed forms.

In such cases, one can make reduction to one-dimensional integrals: DISPLAYFORM2 DISPLAYFORM3 We proceed with computing γ and ρ as follows.

From a random initialization γ (0) > 0 and DISPLAYFORM4 , and stops when the incremental update is negligible or the number of iterations exceeds a threshold.

Upon convergence, this procedure finds a stable fixed point.

We prove several properties of γ and ρ.

We recall G (γ, ρ) and R (γ, ρ) from Eq. FORMULA62 and FORMULA63 for ease of reading: DISPLAYFORM0 for z 1 , z 2 ∼ N (0, 1) independently, where (i) is due to Stein's lemma.

Recall that the fixed points equations are γ = G (γ, ρ) and ρ = R (γ, ρ).

We also recall that β = ασ 2 W > 0.

In light of Remark 3, we will consider Lipschitz continuous, non-decreasing ϕ and σ, so that γ ≥ 0 (and of course, ρ ≥ 0).

We will study these equations, first by stating some propositions, then discussing their implications, although we caution that the link between the propositions and the suggested implications is not entirely rigorous.

All the proofs are deferred to Section C.3.

We note that while the discussions concern with ReLU or tanh activations, the propositions apply to broader classes of functions.

In the following, when we say an initialization for γ and ρ, we mean either an initialization in the context of an iterative process to find the fixed points as in Section C.1, or γ L+1 and ρ L+1 as in Section 2.2 in the context of autoencoders with L → ∞. We also say γ is a fixed point, without referencing to ρ, to mean that it is only a fixed point of γ = G (γ, ρ) for a given ρ, and similarly for ρ.

When we mention both γ and ρ as a fixed point, we mean a fixed point to both γ = G (γ, ρ) and ρ = R (γ, ρ).

We will use ∂ k u f to denote the kth-order partial derivative of f with respect to u.

The following result is exclusive to ReLU ϕ. Proposition 10.

Consider that ϕ is the ReLU and σ is Lipschitz continuous and non-decreasing:(a) γ = 0 and ρ = 0 is a fixed point.

Furthermore, at γ = 0 and β = 2, the mapping ρ → R (0, ρ) admits ρ = 0 as the only fixed point, which is stable if β < 2 and unstable if β > 2.

Also at γ = 0 and β = 2, any ρ is a stable fixed point.(b) Assume σ is positive on a set of positive Lebesgue measure.

If γ → +∞, it must be that ρ → +∞.(c) Considerτ ∈ (0, ∞).

Assume σ is non-zero on a set of positive Lebesgue measure, and that σ (u) = 0 for all u ≤ 0.

Then no γ > 0 is a stable fixed point.

Furthermore,• if βE {σ (τ z 1 )} ≤ 1, there is only one fixed point at γ = 0, which is stable;• if βE {σ (τ z 1 )} ∈ (1, 2), there are two: one at γ = 0, which is stable, and the other at γ > 0, which is unstable;• if βE {σ (τ z 1 )} ≥ 2, there is only one fixed point at γ = 0, which is unstable.(d) Assume σ is non-zero on a set of positive Lebesgue measure, and that σ is an odd function.

Then for any ρ, the mapping γ → G (γ, ρ) is a straight line through the point (0, 0).

Furthermore, if βE {σ (τ z 1 )} = 2, there is only one fixed point at γ = 0, which is stable for βE {σ (τ z 1 )} < 2 and unstable for βE {σ (τ z 1 )} > 2; if βE {σ (τ z 1 )} = 2, any γ is a fixed point.(e) Assume σ is odd, and consider β > 2.

Given γ ≥ 0, we have R (γ, ρ) > ρ for all ρ ≥ 0.The proposition suggests the following picture.

First consider σ is ReLU.

Since E {σ (τ z 1 )} = P (τ z 1 ≥ 0) = 0.5, we have two phase transitions at β = 2 and at β = 4.

In particular, based on Proposition 10:• When β < 2, with any initialization, we have convergence to γ = 0 and ρ = 0.

This is based on Claim (a) and (c).• When β ∈ (2, 4), with certain initializations, we have convergence to γ = 0 and divergence to ρ = +∞; with certain other initializations, we have divergence to γ = +∞ and ρ = +∞. This excludes a special initialization at the unstable fixed point, which is a singleton and essentially a rare case.

This is based on Claim (a), (b) and (c).• When β ≥ 4, with any non-zero initialization, we have divergence to γ = +∞ and hence ρ = +∞. This is based on Claim (b) and (c).Now we consider σ is tanh.

Let β 0 = β 0 (τ ) = 2/E {σ (τ z 1 )}, and it is easy to see that β 0 > 2 since σ = tanh.

The following picture is then expected:• When β < 2, with any initialization, we have convergence to γ = ρ = 0.

This is based on Claim (a) and (d).• When β ∈ (2, β 0 ), with any non-zero initialization, we have convergence to γ = 0 and divergence to ρ = +∞. This is based on Claim (a) and (d).• When β > β 0 , with any non-zero initialization, we have divergence to γ = +∞ and ρ = +∞. This is based on Claim (b) and (d).• When β = β 0 , we have that γ is unchanged from the initialization.

Since β 0 > 2, we then have divergence to ρ = +∞. This is based on Claim (b) and (e).One crucial property of the ReLU is that it is unbounded at infinity and its derivative at infinity is bounded away from zero.

This allows γ and ρ to grow to infinity.

This is a stark contrast to the case ϕ is bounded, for instance, ϕ = tanh as we shall see.

We state a result that is relevant to ϕ = tanh.

Proposition 11.

Assume that ϕ thrice-differentiable with ϕ (0) = 0, ϕ (0) = κ, and ϕ DISPLAYFORM0 is the k-th derivative of ϕ. Assume σ is Lipschitz continuous, nondecreasing.

Then: (a) γ = 0 and ρ = 0 is a fixed point.

Furthermore, assuming that σ is non-zero on a set of positive Lebesgue measure, ϕ is non-zero almost everywhere andτ ∈ (0, ∞), we have if ρ = 0, it must be that γ = 0; in other words, if γ > 0, then ρ > 0.(b) If ϕ is bounded, then 0 ≤ ρ ≤ C, and |γ| ≤ C/τ .(c) Given γ = 0, if β < 1/κ 2 , ρ = 0 is a stable fixed point, and if β > 1/κ 2 , ρ = 0 is unstable.

DISPLAYFORM1 • σ is positive on a set of positive Lebesgue measure, and E {σ (τ z)} > 0 for z ∼ N (0, 1), • either -case 1: σ (u) = 0 for u ≤ 0, and ∆ ρ (u, t) < 0 for u, t, ρ > 0, or -case 2: σ is an odd function, and DISPLAYFORM2 • ϕ satisfies E {zI (z)} > 0 for z is any Gaussian with zero mean and non-zero variance, and I (u) = ϕ (u) + uϕ (u).Then for any given ρ > 0, there exists β * = β * (ρ,τ ) > 0 finite such that if β ≤ β * , then γ = 0 is the only fixed point of the equation γ = G (γ, ρ) and is stable; if β > β * , γ = 0 is unstable, and there is one more fixed point at γ > 0 finite, which is stable.(e) Considerτ ∈ (0, ∞).

The same conclusion as in Claim (c) holds for ρ = 0 with β * = 1/ (κE {σ (τ z 1 )}), assuming• σ is positive on a set of positive Lebesgue measure, and E {σ (τ z)} > 0 for z ∼ N (0, 1), • ϕ (u) < 0 for u > 0, and either -case 1: σ (u) = 0 for u ≤ 0 , or -case 2: ϕ and σ are odd functions.

The assumption ϕ (k) ∞ ≤ C for k = 0, ..., 3 is not critical, only serves to ensure integrability of various terms in the proof and is likely relaxable, but is made for simplicity.

The following lemma establishes certain properties of the tanh function.

Lemma 12.

Consider ϕ = tanh.

Then:(a) For ∆ (u, t) = ϕ (u + t) − ϕ (u − t), we have ∆ (u, t) < 0 for u, t > 0, and ∆ (u, t) < ∆ (−u, t) for u, t > 0.(b) lim s→∞ E {zϕ (z)} = +∞ for z ∼ N 0, s 2 .(c) E {zI (z)} > 0 for z ∼ N 0, s 2 , s = 0, and I (u) = ϕ (u) + uϕ (u).Now let us consider ϕ = σ = tanh.

Note that tanh (u) ∈ (0, 1) for any u = 0, tanh (0) = 1 and E tanh (τ z 1 ) < 1 unlessτ = 0.

By Lemma 12, Proposition 11 applies.

The following picture is suggested based on Proposition 11:• When β < 1, we have convergence to γ = ρ = 0.

This is based on Claim (c) and (e).• The phase transition for ρ locates at β = 1, above which we have convergence to ρ ∈ (0, 1) given a non-zero initialization, and below which ρ = 0.

Here ρ < 1 since tanh is bounded by 1.

This is based on Claim (a), (b), and (c).• The phase transition of γ locates at some β > 1, above which we have convergence to γ > 0 given a non-zero initialization, and below which γ = 0.

Forτ > 0, γ cannot grow to +∞ as β varies.

This is based on Claim (a), (b) and (d).The proposition also suggests that the two phase transitions are close to each other if E {σ (τ z 1 )} ≈ 1.

This requires thatτ 2 ≈ 0, and σ We also expect from the proposition a similar picture for σ being the ReLU, with a crucial difference.

In this case, E {σ (τ z 1 )} = 0.5, and therefore one cannot have that the two phase transitions being close to each other.

Interestingly Claim (a) implies that the phase transition of ρ never occurs before that of γ, regardless of the specific ϕ and σ.

One way for the phase transitions to be close to each other is, as above, taking ϕ = σ = tanh andτ 2 ≈ 0.

Claim (a), (c) and (e) of the proposition also suggests that if E {σ (τ z 1 )} > ϕ (0), then γ and ρ will share the exact same location of the phase transitions, below which they are zero and above which they are positive.

Proof of Proposition 10.

Let θ = βγσ (τ z 1 ) + √ βρz 2 for brevity.

Claim (a).

We have G (0, ρ) = 0 and R (0, 0) = 0.

Simple calculations yield R (0, ρ) = βρ/2.

Claim (a) is then immediate.

To study ρ in the case γ → ∞, we calculate: Table 1 , as described in Appendix D.1, in Setting 1 (i.e., ϕ 0 = tanh).

From the top row: original images, reconstructions from Scheme 1, 4 and 2.

We omit the reconstructions from other schemes, since they are almost identical to those of Scheme 2.

For each digit/letter category, the image is selected from the test set by ranking the reconstruction loss, averaged across Scheme 1 and 4, and picking one at the 75% percentile.

In FIG14 , 11, 12 and 13, we show the reconstructions of several images by the trained networks after 5 × 10 5 training iterations, under the schemes from Table 1 , in the experiments of Section 3.5.

We draw 10 digit images from the MNIST test set, as well as 3 letter images from the EMNIST Letters test set (Cohen et al. (2017) ).

Note that the networks are not trained with any letter images from the EMNIST data set.

The reconstruction quality is visually imperfect even after intensive training, which is entirely expected for vanilla autoencoders and regression problems.

DISPLAYFORM0 Observe that for the schemes that yield meaningful reconstructions, they output recognizable digits for digit images, while for letter images, most of their reconstructions are hardly recognizable as letters.

As such, the trained networks of these schemes do not simply approximate the identity function, but rather capture some low-dimensional structures of the data.

An exception is Scheme 7 under Setting 2, which is not surprising since it is a purely tanh network and tanh is almost identity near zero.

We show the evolution of the test reconstruction loss on the plane σ 2 W , σ 2 b , in conjunction with the experiments of Section 3.5.

To make the computation more manageable, we opt for L = 50 with less iterations, while maintaining other parameters the same as in Section 3.5.

The results are shown in FIG5 .

Several patterns emerge in good agreement with our hypotheses.

Firstly, for ϕ = ReLU, the evolution starts earliest near σ 2 W = 2, shows almost no progress or is numerically unstable when σ 2 W 2, and is much slower when σ 2 W 2.

Secondly, for ϕ = tanh, the evolution is much slower when σ 2 W 1.

Intriguingly the evolution is almost insensitive to σ

<|TLDR|>

@highlight

We study the behavior of weight-tied multilayer vanilla autoencoders under the assumption of random weights. Via an exact characterization in the limit of large dimensions, our analysis reveals interesting phase transition phenomena.

@highlight

A theoretical analysis of autoencoders with weights tied between encoder and decoder (weight-tied) via mean field analysis

@highlight

Analyses the performances of weighted tied auto-encoders by building on recent progress in analysis of high-dimensional statistics problems and specifically, the message passing algorithm

@highlight

This paper studies auto-encoders under several assumptions, and points out that this model of random autoencoder can be elegantly and rigorously analysed with one-dimensional equations.