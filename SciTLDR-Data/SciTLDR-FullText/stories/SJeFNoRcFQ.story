Random Matrix Theory (RMT) is applied to analyze the weight matrices of Deep Neural Networks (DNNs), including both production quality, pre-trained models such as AlexNet and Inception, and smaller models trained from scratch, such as LeNet5 and a miniature-AlexNet.

Empirical and theoretical results clearly indicate that the empirical spectral density (ESD) of DNN layer matrices displays signatures of traditionally-regularized statistical models, even in the absence of exogenously specifying traditional forms of regularization, such as Dropout or Weight Norm constraints.

Building on recent results in RMT, most notably its extension to Universality classes of Heavy-Tailed matrices, we develop a theory to identify 5+1 Phases of Training, corresponding to increasing amounts of Implicit Self-Regularization.

For smaller and/or older DNNs, this Implicit Self-Regularization is like traditional Tikhonov regularization, in that there is a "size scale" separating signal from noise.

For state-of-the-art DNNs, however, we identify a novel form of Heavy-Tailed Self-Regularization, similar to the self-organization seen in the statistical physics of disordered systems.

This implicit Self-Regularization can depend strongly on the many knobs of the training process.

By exploiting the generalization gap phenomena, we demonstrate that we can cause a small model to exhibit all 5+1 phases of training simply by changing the batch size.

The inability of optimization and learning theory to explain and predict the properties of NNs is not a new phenomenon.

From the earliest days of DNNs, it was suspected that VC theory did not apply to these systems (1) .

It was originally assumed that local minima in the energy/loss surface were responsible for the inability of VC theory to describe NNs (1) , and that the mechanism for this was that getting trapped in local minima during training limited the number of possible functions realizable by the network.

However, it was very soon realized that the presence of local minima in the energy function was not a problem in practice (2; 3).

Thus, another reason for the inapplicability of VC theory was needed.

At the time, there did exist other theories of generalization based on statistical mechanics (4; 5; 6; 7), but for various technical and nontechnical reasons these fell out of favor in the ML/NN communities.

Instead, VC theory and related techniques continued to remain popular, in spite of their obvious problems.

More recently, theoretical results of Choromanska et al. (8) (which are related to (4; 5; 6; 7)) suggested that the Energy/optimization Landscape of modern DNNs resembles the Energy Landscape of a zero-temperature Gaussian Spin Glass; and empirical results of Zhang et al. (9) have again pointed out that VC theory does not describe the properties of DNNs.

Martin and Mahoney then suggested that the Spin Glass analogy may be useful to understand severe overtraining versus the inability to overtrain in modern DNNs (10) .We should note that it is not even clear how to define DNN regularization.

The challenge in applying these well-known ideas to DNNs is that DNNs have many adjustable "knobs and switches," independent of the Energy Landscape itself, most of which can affect training accuracy, in addition to many model parameters.

Indeed, nearly anything that improves generalization is called regularization (11) .

Evaluating and comparing these methods is challenging, in part since there are so many, and in part since they are often constrained by systems or other not-traditionally-ML considerations.

Motivated by this situation, we are interested here in two related questions.• Theoretical Question.

Why is regularization in deep learning seemingly quite different than regularization in other areas on ML; and what is the right theoretical framework with which to investigate regularization for DNNs?

• Practical Question.

How can one control and adjust, in a theoretically-principled way, the many knobs and switches that exist in modern DNN systems, e.g., to train these models efficiently and effectively, to monitor their effects on the global Energy Landscape, etc.?

That is, we seek a Practical Theory of Deep Learning, one that is prescriptive and not just descriptive.

This theory would provide useful tools for practitioners wanting to know How to characterize and control the Energy Landscape to engineer larger and betters DNNs; and it would also provide theoretical answers to broad open questions as Why Deep Learning even works.

Main Empirical Results.

Our main empirical results consist in evaluating empirically the ESDs (and related RMT-based statistics) for weight matrices for a suite of DNN models, thereby probing the Energy Landscapes of these DNNs.

For older and/or smaller models, these results are consistent with implicit Self-Regularization that is Tikhonov-like; and for modern state-of-the-art models, these results suggest novel forms of Heavy-Tailed Self-Regularization.• Self-Regularization in old/small models.

The ESDs of older/smaller DNN models (like LeNet5 and a toy MLP3 model) exhibit weak Self-Regularization, well-modeled by a perturbative variant of MP theory, the Spiked-Covariance model.

Here, a small number of eigenvalues pull out from the random bulk, and thus the MP Soft Rank and Stable Rank both decrease.

This weak form of Self-Regularization is like Tikhonov regularization, in that there is a "size scale" that cleanly separates "signal" from "noise," but it is different than explicit Tikhonov regularization in that it arises implicitly due to the DNN training process itself.• Heavy-Tailed Self-Regularization.

The ESDs of larger, modern DNN models (including AlexNet and Inception and nearly every other large-scale model we have examined) deviate strongly from the common Gaussian-based MP model.

Instead, they appear to lie in one of the very different Universality classes of Heavy-Tailed random matrix models.

We call this HeavyTailed Self-Regularization.

The ESD appears Heavy-Tailed, but with finite support.

In this case, there is not a "size scale" (even in the theory) that cleanly separates "signal" from "noise."

Main Theoretical Results.

Our main theoretical results consist in an operational theory for DNN Self-Regularization.

Our theory uses ideas from RMT-both vanilla MP-based RMT as well as extensions to other Universality classes based on Heavy-Tailed distributions-to provide a visual taxonomy for 5 + 1 Phases of Training, corresponding to increasing amounts of Self-Regularization.• Modeling Noise and Signal.

We assume that a weight matrix W can be modeled as W W rand + ∆ sig , where W rand is "noise" and where ∆ sig is "signal."

For small to medium sized signal, W is well-approximated by an MP distribution-with elements drawn from the Gaussian Universality class-perhaps after removing a few eigenvectors.

For large and strongly-correlated signal, W rand gets progressively smaller, but we can model the non-random strongly-correlated signal ∆ sig by a Heavy-Tailed random matrix, i.e., a random matrix with elements drawn from a Heavy-Tailed (rather than Gaussian) Universality class.• 5+1 Phases of Regularization.

Based on this, we construct a practical, visual taxonomy for 5+1 Phases of Training.

Each phase is characterized by stronger, visually distinct signatures in the ESD of DNN weight matrices, and successive phases correspond to decreasing MP Soft Rank and increasing amounts of Self-Regularization.

The 5+1 phases are: RANDOM-LIKE, BLEEDING-OUT, BULK+SPIKES, BULK-DECAY, HEAVY-TAILED, and RANK-COLLAPSE.

Based on these results, we speculate that all well optimized, large DNNs will display Heavy-Tailed Self-Regularization in their weight matrices.

Evaluating the Theory.

We provide a detailed evaluation of our theory using a smaller MiniAlexNew model that we can train and retrain.• Effect of Explicit Regularization.

We analyze ESDs of MiniAlexNet by removing all explicit regularization (Dropout, Weight Norm constraints, Batch Normalization, etc.) and characterizing how the ESD of weight matrices behave during and at the end of Backprop training, as we systematically add back in different forms of explicit regularization.• Exhibiting the 5+1 Phases.

We demonstrate that we can exhibit all 5+1 phases by appropriate modification of the various knobs of the training process.

In particular, by decreasing the batch size from 500 to 2, we can make the ESDs of the fully-connected layers of MiniAlexNet vary continuously from RANDOM-LIKE to HEAVY-TAILED, while increasing generalization accuracy along the way.

These results illustrate the Generalization Gap pheneomena (12; 13; 14) , and they explain that pheneomena as being caused by the implicit Self-Regularization associated with models trained with smaller and smaller batch sizes.

In this section, we summarize results from RMT that we use.

Several overviews of RMT are available (15; 16; 17; 18; 19; 20; 21; 22) .

Here, we will describe a more general form of RMT.

MP theory considers the density of singular values ρ(ν i ) of random rectangular matrices W. This is equivalent to considering the density of eigenvalues ρ(λ i ), i.e., the ESD, of matrices of the form X = W T W. MP theory then makes strong statements about such quantities as the shape of the distribution in the infinite limit, it's bounds, expected finite-size effects, such as fluctuations near the edge, and rates of convergence.

To apply RMT, we need only specify the number of rows and columns of W and assume that the elements W i,j are drawn from a distribution that is a member of a certain Universality class (there are different results for different Universality classes).

RMT then describes properties of the ESD, even at finite size; and one can compare perdictions of RMT with empirical results.

Most well-known is the Universality class of Gaussian distributions.

This leads to the basic or vanilla MP theory, which we describe in this section.

More esoteric-but ultimately more useful for us-are Universality classes of Heavy-Tailed distributions.

In Section 2.2, we describe this important variant.

Gaussian Universality class.

We start by modeling W as an N × M random matrix, with elements from a Gaussian distribution, such that: W ij ∼ N (0, σ 2 mp ).

Then, MP theory states that the ESD of the correlation matrix, X = W T W, has the limiting density given by the MP distribution ρ(λ): DISPLAYFORM0 Here, σ 2 mp is the element-wise variance of the original matrix, Q = N/M ≥ 1 is the aspect ratio of the matrix, and the minimum and maximum eigenvalues, λ ± , are given by DISPLAYFORM1 Finite-size Fluctuations at the MP Edge.

In the infinite limit, all fluctuations in ρ N (λ) concentrate very sharply at the MP edge, λ ± , and the distribution of the maximum eigenvalues ρ ∞ (λ max ) is governed by the TW Law.

Even for a single finite-sized matrix, however, MP theory states the upper edge of ρ(λ) is very sharp; and even when the MP Law is violated, the TW Law, with finitesize corrections, works very well at describing the edge statistics.

When these laws are violated, this is very strong evidence for the onset of more regular non-random structure in the DNN weight matrices, which we will interpret as evidence of Self-Regularization.

MP-based RMT is applicable to a wide range of matrices; but it is not in general applicable when matrix elements are strongly-correlated.

Strong correlations appear to be the case for many welltrained, production-quality DNNs.

In statistical physics, it is common to model strongly-correlated systems by Heavy-Tailed distributions (32) .

The reason is that these models exhibit, more or less, the same large-scale statistical behavior as natural phenomena in which strong correlations exist (32; 19) .

Moreover, recent results from MP/RMT have shown that new Universality classes exist for matrices with elements drawn from certain Heavy-Tailed distributions (19) .We use these Heavy-Tailed extensions of basic MP/RMT to build an operational and phenomenological theory of Regularization in Deep Learning; and we use these extensions to justify our analysis of both Self-Regularization and Heavy-Tailed Self-Regularization.

Briefly, our theory for simple Self-Regularization is insipred by the Spiked-Covariance model of Johnstone (33) and it's interpretation as a form of Self-Organization by Sornette (34) ; and our theory for more sophisticated Heavy-Tailed Self-Regularization is inspired by the application of MP/RMT tools in quantitative finance by Bouchuad, Potters, and coworkers (35; 36; 37; 23; 25; 19; 22) , as well as the relation of Heavy-Tailed phenomena more generally to Self-Organized Criticality in Nature (32) .

Here, we No edge.

Frechet Table 1 : Basic MP theory, and the spiked and Heavy-Tailed extensions we use, including known, empirically-observed, and conjectured relations between them.

Boxes marked " * " are best described as following "TW with large finite size corrections" that are likely Heavy-Tailed (23), leading to bulk edge statistics and far tail statistics that are indistinguishable.

Boxes marked " * * " are phenomenological fits, describing large (2 < µ < 4) or small (0 < µ < 2) finite-size corrections on N → ∞ behavior.

See (24; 23; 25; 26; 27; 28; 29; 30; 19; 31) for additional details.highlight basic results for this generalized MP theory; see (24; 23; 25; 26; 27; 28; 29; 30; 19; 31) in the physics and mathematics literature for additional details.

Universality classes for modeling strongly correlated matrices.

Consider modeling W as an N × M random matrix, with elements drawn from a Heavy-Tailed-e.g., a Pareto or Power Law (PL)-distribution: DISPLAYFORM0 In these cases, if W is element-wise Heavy-Tailed, then the ESD ρ N (λ) likewise exhibits HeavyTailed properties, either globally for the entire ESD and/or locally at the bulk edge.

Table 1 summarizes these recent results, comparing basic MP theory, the Spiked-Covariance model, and Heavy-Tailed extensions of MP theory, including associated Universality classes.

To apply the MP theory, at finite sizes, to matrices with elements drawn from a Heavy-Tailed distribution of the form given in Eqn.

FORMULA2 , we have one of the following three Universality classes.• (Weakly) Heavy-Tailed, 4 < µ: Here, the ESD ρ N (λ) exhibits "vanilla" MP behavior in the infinite limit, and the expected mean value of the bulk edge is λ + ∼ M −2/3 .

Unlike standard MP theory, which exhibits TW statistics at the bulk edge, here the edge exhibits PL / Heavy-Tailed fluctuations at finite N .

These finite-size effects appear in the edge / tail of the ESD, and they make it hard or impossible to distinguish the edge versus the tail at finite N .• (Moderately) Heavy-Tailed, 2 < µ < 4: Here, the ESD ρ N (λ) is Heavy-Tailed / PL in the infinite limit, approaching ρ(λ) ∼ λ −1−µ/2 .

In this regime, there is no bulk edge.

At finite size, the global ESD can be modeled by ρ N (λ) ∼ λ −(aµ+b) , for all λ > λ min , but the slope a and intercept b must be fit, as they display large finite-size effects.

The maximum eigenvalues follow Frechet (not TW) statistics, with λ max ∼ M 4/µ−1 (1/Q) 1−2/µ , and they have large finite-size effects.

Thus, at any finite N , ρ N (λ) is Heavy-Tailed, but the tail decays moderately quickly.• (Very) Heavy-Tailed, 0 < µ < 2: Here, the ESD ρ N (λ) is Heavy-Tailed / PL for all finite N , and as N → ∞ it converges more quickly to a PL distribution with tails ρ(λ) ∼ λ −1−µ/2 .

In this regime, there is no bulk edge, and the maximum eigenvalues follow Frechet (not TW) statistics.

Finite-size effects exist, but they are are much smaller here than in the 2 < µ < 4 regime of µ. Fitting PL distributions to ESD plots.

Once we have identified PL distributions visually, we can fit the ESD to a PL in order to obtain the exponent α.

We use the Clauset-Shalizi-Newman (CSN) approach (38) , as implemented in the python PowerLaw package (39), 1 .

Fitting a PL has many subtleties, most beyond the scope of this paper (38; 40; 41; 42; 43; 44; 39; 45; 46) .

Identifying the Universality class.

Given α, we identify the corresponding µ and thus which of the three Heavy-Tailed Universality classes (0 < µ < 2 or 2 < µ < 4 or 4 < µ, as described in Table 1 ) is appropriate to describe the system.

The following are particularly important points.

First, observing a Heavy-Tailed ESD may indicate the presence of a scale-free DNN.

This suggests that the underlying DNN is strongly-correlated, and that we need more than just a few separated spikes, plus some random-like bulk structure, to model the DNN and to understand DNN regularization.

Second, this does not necessarily imply that the matrix elements of W l form a Heavy-Tailed distribution.

Rather, the Heavy-Tailed distribution arises since we posit it as a model of the strongly correlated, highly non-random matrix W l .

Third, we conjecture that this is more general, and that very welltrained DNNs will exhibit Heavy-Tailed behavior in their ESD for many the weight matrices.

In this section, we describe our main empirical results for existing, pretrained DNNs.

Early on, we observed that small DNNs and large DNNs have very different ESDs.

For smaller models, ESDs tend to fit the MP theory well, with well-understood deviations, e.g., low-rank perturbations.

For larger models, the ESDs ρ N (λ) almost never fit the theoretical ρ mp (λ), and they frequently have a completely different form.

We use RMT to compare and contrast the ESDs of a smaller, older NN and many larger, modern DNNs.

For the small model, we retrain a modern variant of one of the very early and well-known Convolutional Nets-LeNet5.

For the larger, modern models, we examine selected layers from AlexNet, InceptionV3, and many other models (as distributed with pyTorch).Example: LeNet5 (1998).

LeNet5 is the prototype early model for DNNs (2) .

Since LeNet5 is older, we actually recoded and retrained it.

We used Keras 2.0, using 20 epochs of the AdaDelta optimizer, on the MNIST data set.

This model has 100.00% training accuracy, and 99.25% test accuracy on the default MNIST split.

We analyze the ESD of the FC1 Layer.

The FC1 matrix W F C1 is a 2450 × 500 matrix, with Q = 4.9, and thus it yields 500 eigenvalues.

FIG1 (b) zoomed-in along the X-axis.

We show (red curve) our fit to the MP distribution ρ emp (λ).

Several things are striking.

First, the bulk of the density ρ emp (λ) has a large, MP-like shape for eigenvalues λ < λ + ≈ 3.5, and the MP distribution fits this part of the ESD very well, including the fact that the ESD just below the best fit λ + is concave.

Second, some eigenvalue mass is bleeding out from the MP bulk for λ ∈ [3.5, 5], although it is quite small.

Third, beyond the MP bulk and this bleeding out region, are several clear outliers, or spikes, ranging from ≈ 5 to λ max 25.

Overall, the shape of ρ emp (λ), the quality of the global bulk fit, and the statistics and crisp shape of the local bulk edge all agree well with MP theory augmented with a low-rank perturbation.

Example: AlexNet (2012).

AlexNet was the first modern DNN (47) .

AlexNet resembles a scaledup version of the LeNet5 architecture; it consists of 5 layers, 2 convolutional, followed by 3 FC layers (the last being a softmax classifier).

We refer to the last 2 layers before the final softmax as layers FC1 and FC2, respectively.

FC2 has a 4096 × 1000 matrix, with Q = 4.096.Consider AlexNet FC2 (full in FIG1 , and zoomed-in in 1(d)).

This ESD differs even more profoundly from standard MP theory.

Here, we could find no good MP fit.

The best MP fit (in red) does not fit the Bulk part of ρ emp (λ) well.

The fit suggests there should be significantly more bulk eigenvalue mass (i.e., larger empirical variance) than actually observed.

In addition, the bulk edge is indeterminate by inspection.

It is only defined by the crude fit we present, and any edge statistics obviously do not exhibit TW behavior.

In contrast with MP curves, which are convex near the bulk edge, the entire ESD is concave (nearly) everywhere.

Here, a PL fit gives good fit α ≈ 2.25, indicating a µ 3.

For this layer (and others), the shape of ρ emp (λ), the quality of the global bulk fit, and the statistics and shape of the local bulk edge are poorly-described by standard MP theory.

Empirical results for other pre-trained DNNs.

We have also examined the properties of a wide range of other pre-trained models, and we have observed similar Heavy-Tailed properties to AlexNet in all of the larger, state-of-the-art DNNs, including VGG16, VGG19, ResNet50, InceptionV3, etc.

Space constraints prevent a full presentation of these results, but several observations can be made.

First, all of our fits, except for certain layers in InceptionV3, appear to be in the range 1.5 < α 3.5 (where the CSN method is known to perform well).

Second, we also check to see whether PL is the best fit by comparing the distribution to a Truncated Power Law (TPL), as well as an exponential, stretch-exponential, and log normal distributions.

In all cases, we find either a PL or TPL fits best (with a p-value ≤ 0.05), with TPL being more common for smaller values of α.

Third, even when taking into account the large finite-size effects in the range 2 < α < 4, nearly all of the ESDs appear to fall into the 2 < µ < 4 Universality class.

Towards a Theory of Self-Regularization.

For older and/or smaller models, like LeNet5, the bulk of their ESDs (ρ N (λ); λ λ + ) can be well-fit to theoretical MP density ρ mp (λ), potentially with distinct, outlying spikes (λ > λ + ).

This is consistent with the Spiked-Covariance model of Johnstone (33) , a simple perturbative extension of the standard MP theory.

This is also reminiscent of traditional Tikhonov regularization, in that there is a "size scale" (λ + ) separating signal (spikes) from noise (bulk).

This demonstrates that the DNN training process itself engineers a form of implicit Self-Regularization into the trained model.

For large, deep, state-of-the-art DNNs, our observations suggest that there are profound deviations from traditional RMT.

These networks are reminiscent of strongly-correlated disordered-systems that exhibit Heavy-Tailed behavior.

What is this regularization, and how is it related to our observations of implicit Tikhonov-like regularization on LeNet5?To answer this, recall that similar behavior arises in strongly-correlated physical systems, where it is known that strongly-correlated systems can be modeled by random matrices-with entries drawn from non-Gaussian Universality classes (32), e.g., PL or other Heavy-Tailed distributions.

Thus, when we observe that ρ N (λ) has Heavy-Tailed properties, we can hypothesize that W is stronglycorrelated, 2 and we can model it with a Heavy-Tailed distribution.

Then, upon closer inspection, we find that the ESDs of large, modern DNNs behave as expected-when using the lens of HeavyTailed variants of RMT.

Importantly, unlike the Spiked-Covariance case, which has a scale cut-off (λ + ), in these very strongly Heavy-Tailed cases, correlations appear on every size scale, and we can not find a clean separation between the MP bulk and the spikes.

These observations demonstrate that modern, state-of-the-art DNNs exhibit a new form of Heavy-Tailed Self-Regularization.

In this section, we develop an operational/phenomenological theory for DNN Self-Regularization.

MP Soft Rank.

We first define the MP Soft Rank (R mp ), that is designed to capture the "size scale" of the noise part of W l , relative to the largest eigenvalue of W T l W l .

Assume that MP theory fits at least a bulk of ρ N (λ).

Then, we can identify a bulk edge λ + and a bulk variance σ 2 bulk , and define the MP Soft Rank as the ratio of λ + and λ max : R mp (W) := λ + /λ max .

Clearly, R mp ∈ [0, 1]; R mp = 1 for a purely random matrix; and for a matrix with an ESD with outlying spikes, λ max > λ + , and R mp < 1.

If there is no good MP fit because the entire ESD is wellapproximated by a Heavy-Tailed distribution, then we can define λ + = 0, in which case R mp = 0.Visual Taxonomy.

We characterize implicit Self-Regularization, both for DNNs during SGD training as well as for pre-trained DNNs, as a visual taxonomy of 5+1 Phases of Training (RANDOM-LIKE, BLEEDING-OUT, BULK+SPIKES, BULK-DECAY, HEAVY-TAILED, and RANK-COLLAPSE).

See TAB1 Each phase is visually distinct, and each has a natural interpretation in terms of RMT.

One consideration is the global properties of the ESD: how well all or part of the ESD is fit by an MP distriution, for some value of λ + , or how well all or part of the ESD is fit by a Heavy-Tailed or PL distribution, for some value of a PL parameter.

A second consideration is local properties of the ESD: the form of fluctuations, in particular around the edge λ + or around the largest eigenvalue λ max .

For example, the shape of the ESD near to and immediately above λ + is very different in FIG5 Theory of Each Phase.

RMT provides more than simple visual insights, and we can use RMT to differentiate between the 5+1 Phases of Training using simple models that qualitatively describe the shape of each ESD.

We model the weight matrices W as "noise plus signal," where the "noise" is modeled by a random matrix W rand , with entries drawn from the Gaussian Universality class (well-described by traditional MP theory) and the "signal" is a (small or large) correction ∆ sig : TAB1 summarizes the theoretical model for each phase.

Each model uses RMT to describe the global shape of ρ N (λ), the local shape of the fluctuations at the bulk edge, and the statistics and information in the outlying spikes, including possible Heavy-Tailed behaviors.

DISPLAYFORM0 In the first phase (RANDOM-LIKE), the ESD is well-described by traditional MP theory, in which a random matrix has entries drawn from the Gaussian Universality class.

In the next phases (BLEEDING-OUT, BULK+SPIKES), and/or for small networks such as LetNet5, ∆ is a relativelysmall perturbative correction to W rand , and vanilla MP theory (as reviewed in Section 2.1) can be applied, as least to the bulk of the ESD.

In these phases, we will model the W rand matrix by a vanilla W mp matrix (for appropriate parameters), and the MP Soft Rank is relatively large (R mp (W) 0).

In the BULK+SPIKES phase, the model resembles a Spiked-Covariance model, and the Self-Regularization resembles Tikhonov regularization.

In later phases (BULK-DECAY, HEAVY-TAILED), and/or for modern DNNs such as AlexNet and InceptionV3, ∆ becomes more complex and increasingly dominates over W rand .

For these more strongly-correlated phases, W rand is relatively much weaker, and the MP Soft Rank decreases.

Vanilla MP theory is not appropriate, and instead the Self-Regularization becomes Heavy-Tailed.

We will treat the noise term W rand as small, and we will model the properties of ∆ with HeavyTailed extensions of vanilla MP theory (as reviewed in Section 2.2) to Heavy-Tailed non-Gaussian universality classes that are more appropriate to model strongly-correlated systems.

In these phases, the strongly-correlated model is still regularized, but in a very non-traditional way.

The final phase, the RANK-COLLAPSE phase, is a degenerate case that is a prediction of the theory.

To validate and illustrate our theory, we analyzed MiniAlexNet, 3 a simpler version of AlexNet, similar to the smaller models used in (9) , scaled down to prevent overtraining, and trained on CIFAR10.

Space constraints prevent a full presentation of these results, but we mention a few key results here.

The basic architecture consists of two 2D Convolutional layers, each with Max Pooling and Batch Normalization, giving 6 initial layers; it then has two Fully Connected (FC), or Dense, layers with ReLU activations; and it then has a final FC layer added, with 10 nodes and softmax activation.

W F C1 is a 4096 × 384 matrix (Q ≈ 10.67); W F C2 is a 384 × 192 matrix (Q = 2); and W F C3 is a 192 × 10 matrix.

All models are trained using Keras 2.x, with TensorFlow as a backend.

We use SGD with momentum, with a learning rate of 0.01, a momentum parameter of 0.9, and a baseline batch size of 32; and we train up to 100 epochs.

We save the weight matrices at the end of every epoch, and we analyze the empirical properties of the W F C1 and W F C2 matrices.

For each layer, the matrix Entropy (S(W)) gradually lowers; and the Stable Rank (R s (W)) shrinks.

These decreases parallel the increase in training/test accuracies, and both metrics level off as the training/test accuracies do.

These changes are seen in the ESD, e.g., see FIG6 .

For layer FC1, the initial weight matrix W 0 looks very much like an MP distribution (with Q ≈ 10.67), consistent with a RANDOM-LIKE phase.

Within a very few epochs, however, eigenvalue mass shifts to larger values, and the ESD looks like the BULK+SPIKES phase.

Once the Spike(s) appear(s), substantial changes are hard to see visually, but minor changes do continue in the ESD.

Most notably, λ max increases from roughly 3.0 to roughly 4.0 during training, indicating further Self-Regularization, even within the BULK+SPIKES phase.

Here, spike eigenvectors tend to be more localized than bulk eigenvectors.

If explicit regularization (e.g., L 2 norm weight regularization or Dropout) is added, then we observe a greater decrease in the complexity metrics (Entropies and Stable Ranks), consistent with expectations, and this is casued by the eigenvalues in the spike being pulled to much larger values in the ESD.

We also observe that eigenvector localization tends to be more prominent, presumably since explicit regularization can make spikes more well-separated from the bulk.

In this section, we demonstrate that we can exhibit all five of the main phases of learning by changing a single knob of the learning process.

We consider the batch size since it is not traditionally considered a regularization parameter and due to its its implications for the generalization gap.

The Generalization Gap refers to the peculiar phenomena that DNNs generalize significantly less well when trained with larger mini-batches (on the order of 10 3 − 10 4 ) (48; 12; 13; 14) .

Practically, this is of interest since smaller batch sizes makes training large DNNs on modern GPUs much less efficient.

Theoretically, this is of interest since it contradicts simplistic stochastic optimization theory for convex problems.

Thus, there is interest in the question: what is the mechanism responsible for the drop in generalization in models trained with SGD methods in the large-batch regime?To address this question, we consider here using different batch sizes in the DNN training algorithm.

We trained the MiniAlexNet model, just as in Section 5, except with batch sizes ranging from moderately large to very small (b ∈ {500, 250, 100, 50, 32, 16, 8, 4, 2}) .

as a function of Batch Size.

The MP Soft Rank (R mp ) and the Stable Rank (R s ) both track each other, and both systematically decrease with decreasing batch size, as the test accuracy increases.

In addition, both the training and test accuracy decrease for larger values of b: training accuracy is roughly flat until batch size b ≈ 100, and then it begins to decrease; and test accuracy actually increases for extremely small b, and then it gradually decreases as b increases.

ESDs: Comparisons with RMT.

FIG9 shows the final ensemble ESD for each value of b for Layer FC1.

We see systematic changes in the ESD as batch size b decreases.

At batch size b = 250 (and larger), the ESD resembles a pure MP distribution with no outliers/spikes; it is RANDOM-LIKE.

As b decreases, there starts to appear an outlier region.

For b = 100, the outlier region resembles BLEEDING-OUT.

For b = 32, these eigenvectors become well-separated from the bulk, and the ESD resembles BULK+SPIKES.

As batch size continues to decrease, the spikes grow larger and spread out more (observe the scale of the X-axis), and the ESD exhibits BULK-DECAY.

Finally, at b = 2, extra mass from the main part of the ESD plot almost touches the spike, and the curvature of the ESD changes, consistent with HEAVY-TAILED.

In addition, as b decreases, some of the extreme eigenvectors associated with eigenvalues that are not in the bulk tend to be more localized.

Implications for the generalization gap.

Our results here (both that training/test accuracies decrease for larger batch sizes and that smaller batch sizes lead to more well-regularized models) demonstrate that the generalization gap phenomenon arises since, for smaller values of the batch size b, the DNN training process itself implicitly leads to stronger Self-Regularization.

(This SelfRegularization can be either the more traditional Tikhonov-like regularization or the Heavy-Tailed Self-Regularization corresponding to strongly-correlated models.)

That is, training with smaller batch sizes implicitly leads to more well-regularized models, and it is this regularization that leads to improved results.

The obvious mechanism is that, by training with smaller batches, the DNN training process is able to "squeeze out" more and more finer-scale correlations from the data, leading to more strongly-correlated models.

Large batches, involving averages over many more data points, simply fail to see this very fine-scale structure, and thus they are less able to construct strongly-correlated models characteristic of the HEAVY-TAILED phase.

Clearly, our theory opens the door to address numerous very practical questions.

One of the most obvious is whether our RMT-based theory is applicable to other types of layers such as convolutional layers.

Initial results suggest yes, but the situation is more complex than the relatively simple picture we have described here.

These and related directions are promising avenues to explore.

This results from correlations arising at all size scales, which for DNNs arises implicitly due to the training process itself.

This implicit Self-Regularization can depend strongly on the many knobs of the training process.

In particular, by exploiting the generalization gap phenomena, we demonstrate that we can cause a small model to exhibit all 5+1 phases of training simply by changing the batch size.

This demonstrates that-all else being equal-DNN optimization with larger batch sizes leads to less-well implicitly-regularized models, and it provides an explanation for the generalization gap phenomena.

Our results suggest that large, welltrained DNN architectures should exhibit Heavy-Tailed Self-Regularization, and we discuss the theoretical and practical implications of this.

Very large very deep neural networks (DNNs) have received attention as a general purpose tool for solving problems in machine learning (ML) and artificial intelligence (AI), and they perform remarkably well on a wide range of traditionally hard if not impossible problems, such as speech recognition, computer vision, and natural language processing.

The conventional wisdom seems to be "the bigger the better," "the deeper the better," and "the more hyper-parameters the better." Unfortunately, this usual modus operandi leads to large, complicated models that are extremely hard to train, that are extremely sensitive to the parameters settings, and that are extremely difficult to understand, reason about, and interpret.

Relatedly, these models seem to violate what one would expect from the large body of theoretical work that is currently popular in ML, optimization, statistics, and related areas.

This leads to theoretical results that fail to provide guidance to practice as well as to confusing and conflicting interpretations of empirical results.

For example, current optimization theory fails to explain phenomena like the so-called Generalization Gap-the curious observation that DNNs generalize better when trained with smaller batches sizes-and it often does not provide even qualitative guidance as to how stochastic algorithms perform on non-convex landscapes of interest; and current statistical learning theory, e.g., VC-based methods, fails to provide even qualitative guidance as to the behavior of this class of learning methods that seems to have next to unlimited capacity and yet generalize without overtraining.

The inability of optimization and learning theory to explain and predict the properties of NNs is not a new phenomenon.

From the earliest days of DNNs, it was suspected that VC theory did not apply to these systems.

For example, in 1994, Vapnik, Levin, and LeCun BID191 said:[T]he [VC] theory is derived for methods that minimize the empirical risk.

However, existing learning algorithms for multilayer nets cannot be viewed as minimizing the empirical risk over [the] entire set of functions implementable by the network.

It was originally assumed that local minima in the energy/loss surface were responsible for the inability of VC theory to describe NNs BID191 , and that the mechanism for this was that getting trapped in local minima during training limited the number of possible functions realizable by the network.

However, it was very soon realized that the presence of local minima in the energy function was not a problem in practice BID126 39] .

(More recently, this fact seems to have been rediscovered BID155 37, BID103 BID182 .)

Thus, another reason for the inapplicability of VC theory was needed.

At the time, there did exist other theories of generalization based on statistical mechanics BID174 BID194 BID107 43] , but for various technical and nontechnical reasons these fell out of favor in the ML/NN communities.

Instead, VC theory and related techniques continued to remain popular, in spite of their obvious problems.

More recently, theoretical results of Choromanska et al. [30] (which are related to BID174 BID194 BID107 43] ) suggested that the Energy/optimization Landscape of modern DNNs resembles the Energy Landscape of a zero-temperature Gaussian Spin Glass; and empirical results of Zhang et al. BID203 have again pointed out that VC theory does not describe the properties of DNNs.

Motivated by these results, Martin and Mahoney then suggested that the Spin Glass analogy may be useful to understand severe overtraining versus the inability to overtrain in modern DNNs BID140 .Many puzzling questions about regularization and optimization in DNNs abound.

In fact, it is not even clear how to define DNN regularization.

In traditional ML, regularization can be either explicit or implicit.

Let's say that we are optimizing some loss function L(·), specified by some parameter vector or weight matrix W .

When regularization is explicit, it involves making the loss function L "nicer" or "smoother" or "more well-defined" by adding an explicit capacity control term directly to the loss, i.e., by considering a modified objective of the form L(W ) + α W .

In this case, we tune the regularization parameter α by cross validation.

When regularization is implicit, we instead have some adjustable operational procedure like early stopping of an iterative algorithm or truncating small entries of a solution vector.

In many cases, we can still relate this back to the more familiar form of optimizing an effective function of the form L(W ) + α W .

For a precise statement in simple settings, see BID136 BID162 BID100 ; and for a discussion of implicit regularization in a broader context, see BID135 and references therein.

With DNNs, the situation is far less clear.

The challenge in applying these well-known ideas to DNNs is that DNNs have many adjustable "knobs and switches," independent of the Energy Landscape itself, most of which can affect training accuracy, in addition to many model parameters.

Indeed, nearly anything that improves generalization is called regularization, and a recent review presents a taxonomy over 50 different regularization techniques for Deep Learning BID123 .

The most common include ML-like Weight Norm regularization, so-called "tricks of the trade" like early stopping and decreasing the batch size, and DNN-specific methods like Batch Normalization and Dropout.

Evaluating and comparing these methods is challenging, in part since there are so many, and in part since they are often constrained by systems or other not-traditionally-ML considerations.

Moreover, Deep Learning avoids cross validation (since there are simply too many parameters), and instead it simply drives training error to zero (followed by subsequent fiddling of knobs and switches).

Of course, it is still the case that test information can leak into the training process (indeed, perhaps even more severely for DNNs than traditional ML methods).

Among other things, this argues for unsupervised metrics to evaluate model quality.

Motivated by this situation, we are interested here in two related questions.• Theoretical Question.

Why is regularization in deep learning seemingly quite different than regularization in other areas on ML; and what is the right theoretical framework with which to investigate regularization for DNNs?• Practical Question.

How can one control and adjust, in a theoretically-principled way, the many knobs and switches that exist in modern DNN systems, e.g., to train these models efficiently and effectively, to monitor their effects on the global Energy Landscape, etc.?That is, we seek a Practical Theory of Deep Learning, one that is prescriptive and not just descriptive.

This theory would provide useful tools for practitioners wanting to know How to characterize and control the Energy Landscape to engineer larger and betters DNNs; and it would also provide theoretical answers to broad open questions as Why Deep Learning even works.

For example, it would provide metrics to characterize qualitatively-different classes of learning behaviors, as predicted in recent work BID140 .

Importantly, VC theory and related methods do not provide a theory of this form.

Let us write the Energy Landscape (or optimization function) for a typical DNN with L layers, with activation functions h l (·), and with weight matrices and biases W l and b l , as follows: DISPLAYFORM0 For simplicity, we do not indicate the structural details of the layers (e.g., Dense or not, Convolutions or not, Residual/Skip Connections, etc.).

We imagine training this model on some labeled data {d i , y i } ∈ D, using Backprop, by minimizing the loss L (i.e., the cross-entropy), between E DN N and the labels y i , as follows: DISPLAYFORM1 We can initialize the DNN using random initial weight matrices W 0 l , or we can use other methods such as transfer learning (which we will not consider here).

There are various knobs and switches to tune such as the choice of solver, batch size, learning rate, etc.

Most importantly, to avoid overtraining, we must usually regularize our DNN.

Perhaps the most familiar approach from ML for implementing this regularization explicitly constrains the norm of the weight matrices, e.g., modifying Objective (2) to give: DISPLAYFORM2 where · is some matrix norm, and where α is an explicit regularization control parameter.

The point of Objective FORMULA3 is that explicit regularization shrinks the norm(s) of the W l matrices.

We may expect similar results to hold for implicit regularization.

We will use advanced methods from Random Matrix Theory (RMT), developed in the theory of self organizing systems, to characterize DNN layer weight matrices, W l , 1 during and after the training process.

Here is an important (but often under-appreciated) point.

We call E DN N the Energy Landscape.

By this, we mean that part of the optimization problem parameterized by the heretofore unknown elements of the weight matrices and bias vectors, for a fixed α (in FORMULA3 ), and as defined by the data {d i , y i } ∈ D. Because we run Backprop training, we pass the data through the Energy function E DN N multiple times.

Each time, we adjust the values of the weight matrices and bias vectors.

In this sense, we may think of the total Energy Landscape (i.e., the optimization function that is nominally being optimized) as changing at each epoch.

We analyze the distribution of eigenvalues, i.e., the Empirical Spectral Density (ESD), ρ N (λ), of the correlation matrix X = W T W associated with the layer weight matrix W.

We do this for a wide range of large, pre-trained, readily-available state-of-the-art models, including the original LetNet5 convolutional net (which, due to its age, we retrain) and pre-trained models available in Keras and PyTorch such as AlexNet and Inception.

In some cases, the ESDs are very well-described by Marchenko-Pastur (MP) RMT.

In other cases, the ESDs are well-described by MP RMT, with the exception of one or more large eigenvalues that can be modeled by a Spiked-Covariance model BID139 BID115 .

In still other cases-including nearly every current state-ofthe-art model we have examined-the EDSs are poorly-described by traditional RMT, and instead they are more consistent with Heavy-Tailed behavior seen in the statistical physics of disordered systems BID181 24] .

Based on our observations, we develop a develop a practical theory of Implicit Self-Regularization in DNNs.

This theory takes the form of an operational theory characterizing 5+1 phases of DNN training.

To test and validate our theory, we consider two smaller models, a 3-layer MLP (MLP3) and a miniature version of AlexNet (MiniAlexNet), trained on CIFAR10, that we can train ourselves repeatedly, adjusting various knobs and switches along the way.

Main Empirical Results.

Our main empirical results consist in evaluating empirically the ESDs (and related RMT-based statistics) for weight matrices for a suite of DNN models, thereby probing the Energy Landscapes of these DNNs.

For older and/or smaller models, these results are consistent with implicit Self-Regularization that is Tikhonov-like; and for modern state-of-the-art models, these results suggest novel forms of Heavy-Tailed Self-Regularization.• Capacity Control Metrics.

We study simple capacity control metrics, the Matrix Entropy, the linear algebraic or Hard Rank, and the Stable Rank.

We also use MP RMT to define a new metric, the MP Soft Rank.

These metrics track the amount of Self-Regularization that arises in a weight matrix W, either during training or in a pre-trained DNN.• Self-Regularization in old/small models.

The ESDs of older/smaller DNN models (like LeNet5 and a toy MLP3 model) exhibit weak Self-Regularization, well-modeled by a perturbative variant of MP theory, the Spiked-Covariance model.

Here, a small number of eigenvalues pull out from the random bulk, and thus the MP Soft Rank and Stable Rank both decrease.

This weak form of Self-Regularization is like Tikhonov regularization, in that there is a "size scale" that cleanly separates "signal" from "noise," but it is different than explicit Tikhonov regularization in that it arises implicitly due to the DNN training process itself.• Heavy-Tailed Self-Regularization.

The ESDs of larger, modern DNN models (including AlexNet and Inception and nearly every other large-scale model we have examined) deviate strongly from the common Gaussian-based MP model.

Instead, they appear to lie in one of the very different Universality classes of Heavy-Tailed random matrix models.

We call this Heavy-Tailed Self-Regularization.

Here, the MP Soft Rank vanishes, and the Stable Rank decreases, but the full Hard Rank is still retained.

The ESD appears fully (or partially) Heavy-Tailed, but with finite support.

In this case, there is not a "size scale" (even in the theory) that cleanly separates "signal" from "noise."Main Theoretical Results.

Our main theoretical results consist in an operational theory for DNN Self-Regularization.

Our theory uses ideas from RMT-both vanilla MP-based RMT as well as extensions to other Universality classes based on Heavy-Tailed distributions-to provide a visual taxonomy for 5 + 1 Phases of Training, corresponding to increasing amounts of SelfRegularization.• Modeling Noise and Signal.

We assume that a weight matrix W can be modeled as W W rand + ∆ sig , where W rand is "noise" and where ∆ sig is "signal.

"

For small to medium sized signal, W is well-approximated by an MP distribution-with elements drawn from the Gaussian Universality class-perhaps after removing a few eigenvectors.

For large and strongly-correlated signal, W rand gets progressively smaller, but we can model the nonrandom strongly-correlated signal ∆ sig by a Heavy-Tailed random matrix, i.e., a random matrix with elements drawn from a Heavy-Tailed (rather than Gaussian) Universality class.• 5+1 Phases of Regularization.

Based on this approach to modeling noise and signal, we construct a practical, visual taxonomy for 5+1 Phases of Training.

Each phase is characterized by stronger, visually distinct signatures in the ESD of DNN weight matrices, and successive phases correspond to decreasing MP Soft Rank and increasing amounts of Self-Regularization.

The 5+1 phases are: Random-like, Bleeding-out, Bulk+Spikes, Bulk-decay, Heavy-Tailed, and Rank-collapse.• Rank-collapse.

One of the predictions of our RMT-based theory is the existence of a pathological phase of training, the Rank-collapse or "+1" Phase, corresponding to a state of over-regularization.

Here, one or a few very large eigenvalues dominate the ESD, and the rest of the weight matrix loses nearly all Hard Rank.

Based on these results, we speculate that all well optimized, large DNNs will display Heavy-Tailed Self-Regularization in their weight matrices.

Evaluating the Theory.

We provide a detailed evaluation of our theory using a smaller MiniAlexNew model that we can train and retrain.• Effect of Explicit Regularization.

We analyze ESDs of MiniAlexNet by removing all explicit regularization (Dropout, Weight Norm constraints, Batch Normalization, etc.) and characterizing how the ESD of weight matrices behave during and at the end of Backprop training, as we systematically add back in different forms of explicit regularization.• Implementation Details.

Since the details of the methods that underlies our theory (e.g., fitting Heavy-Tailed distributions, finite-size effects, etc.) are likely not familiar to ML and NN researchers, and since the details matter, we describe in detail these issues.• Exhibiting the 5+1 Phases.

We demonstrate that we can exhibit all 5+1 phases by appropriate modification of the various knobs of the training process.

In particular, by decreasing the batch size from 500 to 2, we can make the ESDs of the fully-connected layers of MiniAlexNet vary continuously from Random-like to Heavy-Tailed, while increasing generalization accuracy along the way.

These results illustrate the Generalization Gap phenomena BID111 BID119 BID104 , and they explain that phenomena as being caused by the implicit Self-Regularization associated with models trained with smaller and smaller batch sizes.

By adding extreme Weight Norm regularization, we can also induce the Rank-collapse phase.

Main Methodological Contribution.

Our main methodological contribution consists in using empirical observations as well as recent developments in RMT to motivate a practical predictive DNN theory, rather than developing a descriptive DNN theory based on general theoretical considerations.

Essentially, we treat the training of different DNNs as if we are running novel laboratory experiments, and we follow the traditional scientific method:Make Observations → Form Hypotheses → Build a Theory →

Test the theory, literally.

In particular, this means that we can observe and analyze many large, production-quality, pretrained models directly, without needing to retrain them, and we can also observe and analyze smaller models during the training process.

In adopting this approach, we are interested in both "scientific questions" (e.g., "Why is regularization in deep learning seemingly quite different . . . ?

") as well as "engineering questions" (e.g., "How can one control and adjust . . .

?)

.To accomplish this, recall that, given an architecture, the Energy Landscape is completely defined by the DNN weight matrices.

Since its domain is exponentially large, the Energy Landscape is challenging to study directly.

We can, however, analyze the weight matrices, as well as their correlations.

(This is analogous to analyzing the expected moments of a complicated distribution.)

In principle, this permits us to analyze both local and global properties of the Energy Landscape, as well as something about the class of functions (e.g., VC class, Universality class, etc.) being learned by the DNN.

Since the weight matrices of many DNNs exhibit strong correlations and can be modeled by random matrices with elements drawn from the Universality class of Heavy-Tailed distributions, this severely restricts the class of functions learned.

It also connects back to the Energy Landscape since it is known that the Energy Landscape of Heavy-Tailed random matrices is very different than that of Gaussian-like random matrices.

In Section 2, we provide a warm-up, including simple capacity metrics and their transitions during Backprop.

Then, in Sections 3 and 4, we review background on RMT necessary to understand our experimental methods, and we present our initial experimental results.

Based on this, in Section 5, we present our main theory of 5+1 Phases of Training.

Then, in Sections 6 and 7, we evaluate our main theory, illustrating the effect of explicit regularization, and demonstrating implications for the generalization gap phenomenon.

Finally, in Section 8, we provide a discussion of our results in a broader context.

The accompanying code is available at ((link anonymized for ICLR Supplementary Material)).

For reference, we provide in TAB1 DISPLAYFORM0 , between α and µ (for 2 < µ < 4) ∆λ = λ − λ + empirical uncertainty, due to finite-size effects, in theoretical MP bulk edge ∆ model of perturbations and/or strong correlations in W TAB1 : Definitions of notation used in the text.

In this section, we describe simple spectral metrics to characterize DNN weight these matrices as well as initial empirical observations on the capacity properties of training DNNs.

A DNN is defined by its detailed architecture and the values of the weights and biases at each layer.

We seek a simple capacity control metric for a learned DNN model that: is easy to compute both during training and for already-trained models; can describe changes in the gross behavior of weight matrices during the Backprop training process; and can identify the onset of subtle structural changes in the weight matrices.

One possibility is to use the Euclidean distance between the initial weight matrix, W 0 l , and the weight matrix at epoch e of training, W e l , i.e., ∆(W e l ) = W 0 l − W e l 2 .

This distance, however, is not scale invariant.

In particular, during training, and with regularization turned off, the weight matrices may shift in scale, gaining or losing Frobenius mass or variance, 2 and this distance metric is sensitive to that change.

Indeed, the whole point of a BatchNorm layer is to try to prevent this.

To start, then, we will consider two scale-invariant measures of capacity control: the Matrix Entropy (S), and the Stable Rank (R s ).

For an arbitrary matrix W, both of these metrics are defined in terms of its spectrum.

Consider N × M (real valued) layer weight matrices W l , where DISPLAYFORM0 where ν i = Σ ii is the i th singular value 3 of W, and let p i = ν 2 i / i ν 2 i .

We also define the associated M × M (uncentered) correlation matrix DISPLAYFORM1 where we sometimes drop the (l) subscript for X, and where X is normalized by 1/N .

We compute the eigenvalues of X, DISPLAYFORM2 where {λ i , i = 1, . . .

, M } are the squares of the singular values: λ i = ν 2 i .

Given the singular values of W and/or eigenvalues of X, there are several well-known matrix complexity metrics.• The Hard Rank (or linear algebraic rank), DISPLAYFORM3 is the number of singular values greater than zero, ν i > 0, to within a numerical cutoff.• The Matrix Entropy, Matrix Entropy : DISPLAYFORM4 is also known as the Generalized von-Neumann Matrix Entropy.

4 • The Stable Rank, DISPLAYFORM5 the ratio of the Frobenius norm to Spectral norm, is a robust variant of the Hard Rank.

We also refer to the Matrix Entropy S(X) and Stable Rank R s (X) of X. By this, we mean the metrics computed with the associated eigenvalues.

Note S(X) = S(W) and R s (X) = R s (W).

It is known that a random matrix has maximum Entropy, and that lower values for the Entropy correspond to more structure/regularity.

If W is a random matrix, then S(W) = 1.

For example, we initialize our weight matrices with a truncated random matrix W 0 , then S(W 0 ) 1.

When W has significant and observable non-random structure, we expect S(W) < 1.

We will see, however, that in practice these differences are quite small, and we would prefer a more discriminative metric.

In nearly every case, for well-trained DNNs, all the weight matrices retain full Hard Rank R; but the weight matrices do "shrink," in a sense captured by the Stable Rank.

Both S and R s measure matrix capacity, and, up to a scale factor, we will see that they exhibit qualitatively similar behavior.

We start by illustrating the behavior of two simple complexity metrics during Backprop training on MLP3, a simple 3-layer Multi-Layer Perceptron (MLP), described in Table 4 .

MLP3 consists of 3 fully connected (FC) / dense layers with 512 nodes and ReLU activation, with a final FC layer with 10 nodes and softmax activation.

This gives 4 layer weight matrices of shape (N × M ) and with Q = N/M : DISPLAYFORM0 For the training, each W l matrix is initialized with a Glorot normalization BID101 .

The model is trained on CIFAR10, up to 100 epochs, with SGD (learning rate=0.01, momentum=0.9) and with a stopping criteria of 0.0001 on the MSE loss.

5 FIG1 presents the layer entropy (in FIG1 ) and the stable rank (in FIG1 ), plotted as a function of training epoch, for FC1 and FC2.

Both metrics decrease during training (note the scales of the Y axes): the stable rank decreases by approximately a factor of two, and the matrix entropy decreases by a small amount, from roughly 0.92 to just below 0.91 (this is for FC2, and there is an even more modest change for FC1).

They both track nearly the same changes; and the stable rank is more informative for our purposes; but we will see that the changes to the matrix entropy, while subtle, are significant.

Figure 2 presents scree plots for the initial W 0 l and final W l weight matrices for the FC1 and FC2 layers of our MLP3.

A scree plot plots the decreasing variability in the matrix as a function of the increasing index of the corresponding eigenvector BID106 .

Thus, such scree plots present similar information to the stable rank-e.g., observe the Y-axis of FIG5 (b), which shows that there is a slight increase in the largest eigenvalue for FC1 (again, note the scales of the Y axes) and a larger increase in the largest eigenvalue for FC2, which is consistent with the changes in the stable rank in FIG1 )

-but they too give a coarse picture of the matrix.

In particular, they lack the detailed insight into subtle changes in the entropy and rank associated with the Self-Regularization process, e.g., changes that reside in just a few singular values and vectors, that we will need in our analysis.

Limitations of these metrics.

We can gain more detailed insight into changes in W l during training by creating histograms of the singular values and/or eigenvalues (λ i = ν 2 i ).

FIG6 (a) displays the density of singular values of W 0 F C2 and W F C2 for the FC2 layer of the MLP3 model.

FIG6 (b) displays the associated eigenvalue densities, ρ N (λ), which we call the Empirical Spectral Density (ESD) (defined in detail below) plots.

Observe that the initial density of singular values (shown in red/purple), resembles a quarter circle, 6 and the final density of singular values (blue) consists of a bulk quarter circle, of about the same width, with several spikes of singular value density beyond the bulk's edge.

Observe that the similar heights and widths and shapes of 6 The initial weight matrix W 0 F C2 is just a random (Glorot Normal) matrix.the bulks imply the variance, or Frobenius norm, does not change much: W F C2 F ≈ W 0 F C2 F .

Observe also that the initial ESD, ρ N (λ) (red/purple), is crisply bounded between λ − = 0 and λ + ∼ 3.2 (and similarly for the density of singular values at the square root of this value), whereas the final ESD (blue) has less density at λ − = 0 and several spikes λ λ + .

The largest eigenvalue is λ max ∼ 7.2, i.e., W F C2 DISPLAYFORM1 .

We see now why the stable rank for FC2 decreases by ∼ 2X; the Frobenius norm does not change much, but the squared Spectral norm is ∼ 2X larger.

The fine-scale structure that is largely hidden from FIG1 but that is easily-revealed by singular/eigen value density plots of FIG6 suggests that a RMT analysis might be fruitful.

In this section, we summarize results from RMT that we use.

RMT provides a kind-of Central Limit Theorem for matrices, with unique results for both square and rectangular matrices.

Perhaps the most well-known results from RMT are the Wigner Semicircle Law, which describes the eigenvalues of random square symmetric matrices, and the Tracy Widom (TW) Law, which states how the maximum eigenvalue of a (more general) random matrix is distributed.

Two issues arise with applying these well-known versions of RMT to DNNs.

First, very rarely do we encounter symmetric weight matrices.

Second, in training DNNs, we only have one instantiation of each weight matrix, and so it is not generally possible to apply the TW Law.

7 Several overviews of RMT are available BID190 41, BID118 BID189 22, 42, BID157 24] .

Here, we will describe a more general form of RMT, the Marchenko-Pastur (MP) theory, applicable to rectangular matrices, including (but not limited to) DNN weight matrices W.

MP theory considers the density of singular values ρ(ν i ) of random rectangular matrices W. This is equivalent to considering the density of eigenvalues ρ(λ i ), i.e., the ESD, of matrices of the form X = W T W. MP theory then makes strong statements about such quantities as the shape of the distribution in the infinite limit, it's bounds, expected finite-size effects, such as fluctuations near the edge, and rates of convergence.

When applied to DNN weight matrices, MP theory assumes that W, while trained on very specific datasets, exhibits statistical properties that do not depend on the specific details of the elements W i,j , and holds even at finite size.

This Universality concept is "borrowed" from Statistical Physics, where it is used to model, among other things, strongly-correlated systems and so-called critical phenomena in nature BID181 .To apply RMT, we need only specify the number of rows and columns of W and assume that the elements W i,j are drawn from a specific distribution that is a member of a certain Universality class (there are different results for different Universality classes).

RMT then describes properties of the ESD, even at finite size; and one can compare perdictions of RMT with empirical results.

Most well-known and well-studied is the Universality class of Gaussian distributions.

This leads to the basic or vanilla MP theory, which we describe in this section.

More esoteric-but ultimately more useful for us-are Universality classes of Heavy-Tailed distributions.

In Section 3.2, we describe this important variant.

Gaussian Universality class.

We start by modeling W as an N × M random matrix, with elements drawn from a Gaussian distribution, such that: DISPLAYFORM0 Then, MP theory states that the ESD of the correlation matrix, X = W T W, has the limiting density given by the MP distribution ρ(λ): DISPLAYFORM1 Here, σ 2 mp is the element-wise variance of the original matrix, Q = N/M ≥ 1 is the aspect ratio of the matrix, and the minimum and maximum eigenvalues, λ ± , are given by FORMULA17 and FORMULA18 , as the aspect ratio Q and variance parameter σ are modified.

DISPLAYFORM2 The MP distribution for different aspect ratios Q and variance parameters σ mp .

The shape of the MP distribution only depends on two parameters, the variance σ 2 mp and the aspect ratio Q. See FIG8 for an illustration.

In particular, see FIG8 (a) for a plot of the MP distribution of Eqns.

FORMULA17 and FORMULA18 , for several values of Q; and see FIG8 (b) for a plot of the MP distribution for several values of σ mp .As a point of reference, when Q = 4 and σ mp = 1 (blue in both subfigures), the mass of ρ N skews slightly to the left, and is bounded in [0.3 − 2.3].

For fixed σ mp , as Q increases, the support (i.e., [λ − , λ + ]) narrows, and ρ N becomes less skewed.

As Q → 1, the support widens and ρ N skews more leftward.

Also, ρ N is concave for larger Q, and it is partially convex for smaller Q = 1.Although MP distribution depends on Q and σ 2 mp , in practice Q is fixed, and thus we are interested how σ 2 mp varies-distributionally for random matrices, and empirically for weight matrices.

Due to Eqn.

(9), if σ 2 mp is fixed, then λ + (i.e., the largest eigenvalue of the bulk, as well as λ − ) is determined, and vice versa.

8 8 In practice, relating λ + and σ 2 mp raises some subtle technical issues, and we discuss these in Section 6.3.The Quarter Circle Law for Q = 1.

A special case of Eqn.

FORMULA17 arises when Q = 1, i.e., when W is a square non-symmetric matrix.

In this case, the eigenvalue density ρ(λ) is very peaked with a bounded tail, and it is sometimes more convenient to consider the density of singular values of W l , ρ(ν), which takes the form of a Quarter-Circle: DISPLAYFORM3 We will not pursue this further, but we saw this earlier, in FIG6 (b), with our toy MLP3 model.

Finite-size Fluctuations at the MP Edge.

In the infinite limit, all fluctuations in ρ N (λ) concentrate very sharply at the MP edge, λ ± , and the distribution of the maximum eigenvalues ρ ∞ (λ max ) is governed by the TW Law.

Even for a single finite-sized matrix, however, MP theory states the upper edge of ρ(λ) is very sharp; and even when the MP Law is violated, the TW Law, with finite-size corrections, works very well at describing the edge statistics.

When these laws are violated, this is very strong evidence for the onset of more regular non-random structure in the DNN weight matrices, which we will interpret as evidence of Self-Regularization.

In more detail, in many cases, one or more of the empirical eigenvalues will extend beyond the sharp edge predicted by the MP fit, i.e., such that λ max > λ + (where λ max is the largest eigenvalue of X).

It will be important to distinguish the case that λ max > λ + simply due the finite size of W from the case that λ max is "truly" outside the MP bulk.

According to MP theory [22] , for finite (N, M ), and with DISPLAYFORM4 where λ + is given by Eqn (9).

Since Q = N/M , we can also express this in terms of N −2/3 , but with different prefactors BID115 .

Most importantly, within MP theory (and even more generally), the λ max fluctuations, centered and rescaled, will follow TW statistics.

In the DNNs we consider, M 400, and so the maximum deviation is only ∆λ M 0.02.

In many cases, it will be obvious whether a given λ max is an outlier.

When it is not, one could generate an ensemble of N R runs and study the information content of the eigenvalues (shown below) and/or apply TW theory (not discussed here).Fitting MP Distributions.

Several technical challenges with fitting MP distributions, i.e., selecting the bulk edge λ + , are discussed in Section 6.3.

MP-based RMT is applicable to a wide range of matrices (even those with large low-rank perturbations ∆ large to i.i.d.

normal behavior); but it is not in general applicable when matrix elements are strongly-correlated.

Strong correlations appear to be the case for many well-trained, production-quality DNNs.

In statistical physics, it is common to model strongly-correlated systems by Heavy-Tailed distributions BID181 .

The reason is that these models exhibit, more or less, the same large-scale statistical behavior as natural phenomena in which strong correlations exist BID181 22] .

Moreover, recent results from MP/RMT have shown that new Universality classes exist for matrices with elements drawn from certain Heavy-Tailed distributions [22] .We use these Heavy-Tailed extensions of basic MP/RMT to build an operational and phenomenological theory of Regularization in Deep Learning; and we use these extensions to justify DISPLAYFORM0 No edge.

Frechet DISPLAYFORM1 No edge.

Frechet Table 3 : Basic MP theory, and the spiked and Heavy-Tailed extensions we use, including known, empirically-observed, and conjectured relations between them.

Boxes marked " * " are best described as following "TW with large finite size corrections" that are likely Heavy-Tailed [20] , leading to bulk edge statistics and far tail statistics that are indistinguishable.

Boxes marked " * * " are phenomenological fits, describing large (2 < µ < 4) or small (0 < µ < 2) finite-size corrections on N → ∞ behavior.

See [38, 20, 19, BID158 7, 40, 8, 26, 22, 21] for additional details.our analysis of both Self-Regularization and Heavy-Tailed Self-Regularization.

9 Briefly, our theory for simple Self-Regularization is insipred by the Spiked-Covariance model of Johnstone BID115 and it's interpretation as a form of Self-Organization by Sornette BID139 ; and our theory for more sophisticated Heavy-Tailed Self-Regularization is inspired by the application of MP/RMT tools in quantitative finance by Bouchuad, Potters, and coworkers BID96 BID124 BID125 20, 19, 22, 24] , as well as the relation of Heavy-Tailed phenomena more generally to Self-Organized Criticality in Nature BID181 .

Here, we highlight basic results for this generalized MP theory; see [38, 20, 19, BID158 7, 40, 8, 26, 22, 21] in the physics and mathematics literature for additional details.

Universality classes for modeling strongly correlated matrices.

Consider modeling W as an N × M random matrix, with elements drawn from a Heavy-Tailed-e.g., a Pareto or Power Law (PL)-distribution: DISPLAYFORM2 In these cases, if W is element-wise Heavy-Tailed, 10 then the ESD ρ N (λ) likewise exhibits HeavyTailed properties, either globally for the entire ESD and/or locally at the bulk edge.

Table 3 summarizes these (relatively) recent results, comparing basic MP theory, the SpikedCovariance model, 11 and Heavy-Tailed extensions of MP theory, including associated Universality classes.

To apply the MP theory, at finite sizes, to matrices with elements drawn from a HeavyTailed distribution of the form given in Eqn. (10), then, depending on the value of µ, we have 9 The Universality of RMT is a concept broad enough to apply to classes of problems that appear well beyond its apparent range of validity.

It is in this sense that we apply RMT to understand DNN Regularization.10 Heavy-Tailed phenomena have many subtle properties BID168 ; we consider here only the most simple cases.

11 We discuss Heavy-Tailed extensions to MP theory in this section.

Extensions to large low-rank perturbations are more straightforward and are described in Section 5.3.one of the following three 12 Universality classes:• (Weakly) Heavy-Tailed, 4 < µ: Here, the ESD ρ N (λ) exhibits "vanilla" MP behavior in the infinite limit, and the expected mean value of the bulk edge is λ + ∼ M −2/3 .

Unlike standard MP theory, which exhibits TW statistics at the bulk edge, here the edge exhibits PL / Heavy-Tailed fluctuations at finite N .

These finite-size effects appear in the edge / tail of the ESD, and they make it hard or impossible to distinguish the edge versus the tail at finite N .• (Moderately) Heavy-Tailed, 2 < µ < 4: Here, the ESD ρ N (λ) is Heavy-Tailed / PL in the infinite limit, approaching the form ρ(λ) ∼ λ −1−µ/2 .

In this regime of µ, there is no bulk edge.

At finite size, the global ESD can be modeled by the form ρ N (λ) ∼ λ −(aµ+b) , for all λ > λ min , but the slope a and intercept b must be fit, as they display very large finitesize effects.

The maximum eigenvalues follow Frechet (not TW) statistics, with λ max ∼ M 4/µ−1 (1/Q) 1−2/µ , and they have large finite-size effects.

Even if the ESD tends to zero, the raw number of eigenvalues can still grow-just not as quickly as N (i.e., we may expect some λ max > λ + , in the infinite limit, but the eigenvalue density ρ(λ) → 0).

Thus, at any finite N , ρ N (λ) is Heavy-Tailed, but the tail decays moderately quickly.• (Very) Heavy-Tailed, 0 < µ < 2: Here, the ESD ρ N (λ) is Heavy-Tailed / PL for all finite N , and as N → ∞ it converges more quickly to a PL distribution with tails ρ(λ) ∼ λ −1−µ/2 .

In this regime, there is no bulk edge, and the maximum eigenvalues follow Frechet (not TW) statistics.

Finite-size effects exist here, but they are are much smaller here than in the 2 < µ < 4 regime of µ. The log-log histogram plots of the ESD for three Heavy-Tailed random matrices M with same aspect ratio Q = 3, with µ = 1.0, 3.0, 5.0, corresponding to the three Heavy-Tailed Universality classes (0 < µ < 2 vs 2 < µ < 4 and 4 < µ) described in Table 3 .Visualizing Heavy-Tailed distributions.

It is often fruitful to perform visual exploration and classification of ESDs by plotting them on linear-linear coordinates, log-linear coordinates (linear horizontal/X axis and logarithmic vertical/Y axis), and/or log-log coordinates (logarithmic horizontal/X axis and logarithmic vertical/Y axis).

It is known that data from a PL distribution will appear as a convex curve in a linear-linear plot and a log-linear plot and as a straight line in a log-log plot; and that data from a Gaussian distribution will appear as a bell-shaped curve in a linear-linear plot, as an inverted parabola in a log-linear plot, and as a strongly concave curve in a log-log plot.

Examining data from an unknown ESD on different axes suggests a classification for them.

(See FIG1 .)

More quantitative analysis may lead to more definite conclusions, but that too comes with technical challenges.

To illustrate this, we provide a visual and operational approach to understand the limiting forms for different µ. See FIG9 .

FIG9 displays the log-log histograms for the ESD ρ N (λ) for three Heavy-Tailed random matrices M N (µ), with µ = 1.0, 3.0, 5.0 For µ = 1.0 (blue), the log-log histogram is linear over 5 log scales, from 10 3 − 10 8 .

If N increases (not shown), λ max will grow, but this plot will remain linear, and the tail will not decay.

In the infinite limit, the ESD will still be Heavy-Tailed.

Contrast this with the ESD drawn from the same distribution, except with µ = 3.0 (green).

Here, due to larger finite-size effects, most of the mass is confined to one or two log scales, and it starts to vanish when λ > 10 3 .

This effect is amplified for µ = 5.0 (red), which shows almost no mass for eigenvalues beyond the MP bulk (i.e. λ > λ + ).

Zooming in, in FIG9 (b), we see that the log-log plot is linear-in the central region only-and the tail vanishes very quickly.

If N increases (not shown), the ESD will remain Heavy-Tailed, but the mass will grow much slower than when µ < 2.

This illustrates that, while ESDs can be HeavyTailed at finite size, the tails decay at different rates for different Heavy-Tailed Universality classes (0 < µ < 2 or 2 < µ < 4 or 4 < µ).Fitting PL distributions to ESD plots.

Once we have identified PL distributions visually (using a log-log histogram of the ESD, and looking for visual characteristics of FIG9 ), we can fit the ESD to a PL in order to obtain the exponent α.

For this, we use the Clauset-Shalizi-Newman (CSN) approach [33] , as implemented in the python PowerLaw package [2] , 13 which computes an α such that DISPLAYFORM3 Generally speaking, fitting a PL has many subtleties, most beyond the scope of this paper [33, BID102 BID138 BID147 16, BID120 35, 2, BID192 BID105 .

For example, care must be taken to ensure the distribution is actually linear (in some regime) on a log-log scale before applying the PL estimator, lest it give spurious results; and the PL estimator only works reasonably well for exponents in the range 1.5 < α 3.5.To illustrate this, consider FIG20 .

In particular, FIG20 (a) shows that the CSN estimator performs well for the regime 0 < µ < 2, while for 2 < µ < 4 there are substantial deviations due to finite-size effects, and for 4 < µ no reliable results are obtained; and FIG20 show that the finite-size effects can be quite complex (for fixed M , increasing Q leads to larger finite-size effects, while for fixed N , decreasing Q leads to larger finite-size effects).Identifying the Universality class.

Given α, we identify the corresponding µ (as illustrated in FIG20 ) and thus which of the three Heavy-Tailed Universality classes (0 < µ < 2 or 2 < µ < 4 or 4 < µ, as described in Table 5 ) is appropriate to describe the system.

For our theory, the following are particularly important points.

First, observing a Heavy-Tailed ESD may indicate the presence of a scale-free DNN.

This suggests that the underlying DNN is strongly-correlated, and that we need more than just a few separated spikes, plus some random-like bulk structure, to model the DNN and to understand DNN regularization.

Second, this does not necessarily imply .

In (6(a)), the PL exponent α is fit, using the CSN estimator, for the ESD ρ emp (λ) for a random, rectangular Heavy-Tailed matrix W(µ) (Q = 2, M = 1000), with elements drawn from a Pareto distribution p(x) ∼ x −1−µ .

For 0 < µ < 2, finite-size effects are modest, and the ESD follows the theoretical prediction ρ emp (λ) ∼ λ −1−µ/2 .

For 2 < µ < 4, the ESD still shows roughly linear behavior, but with significant finite-size effects, giving the more general phenomenological relation ρ emp (λ) ∼ λ −aµ+b .

For 4 < µ, the CSN method is known to fail to perform well.

In (6(b)) and (6(c)), plots are shown for varying Q, with M and N fixed, respectively.that the matrix elements of W l form a Heavy-Tailed distribution.

Rather, the Heavy-Tailed distribution arises since we posit it as a model of the strongly correlated, highly non-random matrix W l .

Third, we conjecture that this is more general, and that very well-trained DNNs will exhibit Heavy-Tailed behavior in their ESD for many the weight matrices (as we have observed so far with many pre-trained models).

When entries of a random matrix are drawn from distributions in the Gaussian Universality class, and under typical assumptions, eigenvectors tend to be delocalized, i.e., the mass of the eigenvector tends to be spread out on most or all the components of that vector.

For other models, eigenvectors can be localized.

For example, spike eigenvectors in Spiked-Covariance models as well as extremal eigenvectors in Heavy-Tailed random matrix models tend to be more localized BID157 24] .

Eigenvector delocalization, in traditional RMT, is modeled using the Thomas Porter Distribution BID165 .

Since a typical bulk eigenvector v should have maximum entropy, therefore it's components v i should be Gaussian distributed, according to: DISPLAYFORM0 Here, we normalize v such that the empirical variance of the elements is unity, σ 2 v i = 1.

Based on this, we can define several related eigenvector localization metrics.• The Generalized Vector Entropy, S(v) := i P (v i ) ln P (v i ), is computed using a histogram estimator.• The Localization Ratio, L(v) := v 1 v ∞ , measures the sum of the absolute values of the elements of v, relative to the largest absolute value of an element of v.• The Participation Ratio, DISPLAYFORM1 , is a robust variant of the Localization Ratio.

For all three metrics, the lower the value, the more localized the eigenvector v tends to be.

We use deviations from delocalization as a diagnostic that the corresponding eigenvector is more structured/regularized.

In this section, we describe our main empirical results for existing, pretrained DNNs.

14 Early on, we observed that small DNNs and large DNNs have very different ESDs.

For smaller models, ESDs tend to fit the MP theory well, with well-understood deviations, e.g., low-rank perturbations.

For larger models, the ESDs ρ N (λ) almost never fit the theoretical ρ mp (λ), and they frequently have a completely different functional form.

We use RMT to compare and contrast the ESDs of a smaller, older NN and many larger, modern DNNs.

For the small model, we retrain a modern variant of one of the very early and well-known Convolutional Nets-LeNet5.

We use Keras (2), and we train LeNet5 on MNIST.

For the larger, modern models, we examine selected layers from AlexNet, InceptionV3, and many other models (as distributed with pyTorch).

Table 4 provides a summary of models we analyzed in detail.

LeNet5 predates both the current Deep Learning revolution and the so-called AI Winter, dating back to the late 1990s BID126 .

It is the prototype early model for DNNs; it is the most widely-known example of a Convolutional Neural Network (CNN); and it was used in production systems for recognizing hand written digits BID126 .

The basic design consists of 2 Convolutional (Conv2D) and MaxPooling layers, followed by 2 Dense, or Fully Connected (FC), layers, FC1 and FC2.

This design inspired modern DNNs for image classification, e.g., AlexNet, VGG16 and VGG19.

All of these latter models consist of a few Conv2D and MaxPooling layers, followed by a few FC layers.

Since LeNet5 is older, we actually recoded and retrained it.

We used Keras 2.0, using 20 epochs of the AdaDelta optimizer, on the MNIST data set.

This model has 100.00% training accuracy, and 99.25% test accuracy on the default MNIST split.

We analyze the ESD of the FC1 Layer (but not the FC2 Layer since it has only 10 eigenvalues).

The FC1 matrix W F C1 is a 2450 × 500 matrix, with Q = 4.9, and thus it yields 500 eigenvalues.

FC1: MP Bulk+Spikes, with edge Bleeding-out.

FIG21 presents the ESD for FC1 of LeNet5, with FIG21 (a) showing the full ESD and FIG21 showing the same ESD, zoomed-in along the X-axis to highlight smaller peaks outside the main bulk of our MP fit.

In both cases, we show (red curve) our fit to the MP distribution ρ emp (λ).

Several things are striking.

First, the bulk of the density ρ emp (λ) has a large, MP-like shape for eigenvalues λ < λ + ≈ 3.5, and the MP distribution fits this part of the ESD very well, including the fact that the ESD just below the best fit λ + is concave.

Second, some eigenvalue mass is bleeding out from the MP bulk for λ ∈ [3.5, 5], although it is quite small.

Third, beyond the MP bulk and this bleeding out region, are several clear outliers, or spikes, ranging from ≈ 5 to λ max 25.Summary.

The shape of ρ emp (λ), the quality of the global bulk fit, and the statistics and crisp shape of the local bulk edge all agree well with standard MP theory, or at least the variant of Exhibits all 5+1 Phases of Training by changing batch size Table 4 : Description of main DNNs used in our analysis and the key observations about the ESDs of the specific layer weight matrices using RMT.

Names in the "Key observation" column are defined in Section 5 and described in Table 7 .MP theory augmented with a low-rank perturbation.

In this sense, this model can be viewed as a real-world example of the Spiked-Covariance model BID115 .

AlexNet was the first modern DNN, and its spectacular performance opened the door for today's revolution in Deep Learning.

Specifically, it was top-5 on the ImageNet ILSVRC2012 classification task BID121 , achieving an error of 16.4%, over 11% ahead of the first runner up.

AlexNet resembles a scaled-up version of the LeNet5 architecture; it consists of 5 layers, 2 convolutional, followed by 3 FC layers (the last being a softmax classifier).

15 We will analyze the version of AlexNet currently distributed with pyTorch (version 0.4.1).

In this version, FC1 has a 9216 × 4096 matrix, with Q = 2.25; FC2 has a 4096 × 4096 matrix, with Q = 1.0; and FC3 has a 4096 × 1000 matrix, with Q = 4.096 ≈ 4.1.

Notice that FC3 is the final layer and connects AlexNet to the labels.

FIG22 , and 10 present the ESDs for weight matrices of AlexNet for Layers FC1, FC2, and FC3, with FIG1 showing the full ESD, and FIG1 showing the results "zoomed-in" along the X-axis.

In each cases, we present best MP fits, as determined by holding Q fixed, adjusting the σ parameter, and selecting the best bulk fit by visual inspection.

Fitting σ fixes λ + , and the λ + estimates differ for different layers because the matrices have different aspect ratios Q. In each case, the ESDs exhibit moderate to strong deviations from the best standard MP fit.

FC1: Bulk-decay into Heavy-Tailed.

Consider first AlexNet FC1 (in FIG22 ).

The eigenvalues range from near 0 up to ca.

30, just as with LeNet5.

The full ESD, however, is shaped very differently than any theoretical ρ mp (λ), for any value of λ.

The best MP fit (in red in FIG22 ) does capture a good part of the eigenvalue mass, but there are important differences: the peak is not filled in, there is substantial eigenvalue mass bleeding out from the bulk, and the shape of the ESD is convex in the region near to and just above the best fit for λ + of the bulk edge.

Contrast this with the excellent MP fit for the ESD for FC1 of LeNet5 FIG21 ), where the red curve captures all of the bulk mass, and only a few outlying spikes appear.

Moreover, and very importantly, in AlexNet FC1, the bulk edge is not crisp.

In fact, it is not visible at all; and λ + is solely defined operationally by selecting the σ parameter.

As such, the edge fluctuations, ∆λ, do not resemble a TW distribution, and the bulk itself appears to just decay into the heavy tail.

Finally, a PL fit gives good fit α ≈ 2.29, suggesting (due to finite size effects) µ 2.5.

FIG23 ).

This ESD differs even more profoundly from standard MP theory.

Here, we could find no good MP fit, even by adjusting σ and Q simultaneously.

The best MP fit (in red) does not fit the Bulk part of ρ emp (λ) at all.

The fit suggests there should be significantly more bulk eigenvalue mass (i.e., larger empirical variance) than actually observed.

In addition, as with FC1, the bulk edge is indeterminate by inspection.

It is only defined by the crude fit we present, and any edge statistics obviously do not exhibit TW behavior.

In contrast with MP curves, which are convex near the bulk edge, the entire ESD is concave (nearly) everywhere.

Here, a PL fit gives good fit α ≈ 2.25, smaller than FC1 and FC3, indicating a µ 3.

FIG1 ).

Here, too, the ESDs deviate strongly from predictions of MP theory, both for the global bulk properties and for the local edge properties.

A PL fit gives good fit α ≈ 3.02, which is larger than FC1 and FC2.

This suggests a µ 2.5 (which is also shown with a log-log histogram plot in FIG1 in Section 5 below).

Summary.

For all three layers, the shape of ρ emp (λ), the quality of the global bulk fit, and the statistics and shape of the local bulk edge are poorly-described by standard MP theory.

Even when we may think we have moderately a good MP fit because the bulk shape is qualitatively captured with MP theory (at least visual inspection), we may see a complete breakdown RMT at the bulk edge, where we expect crisp TW statistics (or at least a concave envelope of support).

In other cases, the MP theory may even be a poor estimator for even the bulk.

In the few years after AlexNet, several new, deeper DNNs started to win the ILSVRC ImageNet completions, including ZFNet(2013) BID202 , VGG(2014) BID177 , GoogLeNet/Inception (2014) BID186 , and ResNet (2015) BID108 .

We have observed that nearly all of these DNNs have properties that are similar to AlexNet.

Rather than describe them all in detail, in Section 4.4, we perform power law fits on the Linear/FC layers in many of these models.

Here, we want to look more deeply at the Inception model, since it displays some unique properties.

16 In 2014, the VGG BID177 and GoogLeNet BID186 models were close competitors in the ILSVRC2014 challenges.

For example, GoogLeNet won the classification challenge, but VGG performed better on the localization challenge.

These models were quite deep, with GoogLeNet having 22 layers, and VGG having 19 layers.

The VGG model is ∼2X as deep as AlexNet, but it replaces each larger AlexNet filter with more, smaller filters.

Presumably this deeper architecture, with more non-linearities, can capture the correlations in the network better.

The VGG features of the second to last FC layer generalize well to other tasks.

A downside of the VGG models is that they have a lot of parameters and that they use a lot of memory.

The GoogleLeNet/Inception design resembles the VGG architecture, but it is even more computationally efficient, which (practically) means smaller matrices, fewer parameters (12X fewer than AlexNet), and a very different architecture, including no internal FC layers, except those connected to the labels.

In particular, it was noted that most of the activations in these DNNs are redundant because they are so strongly correlated.

So, a sparse architecture should perform just as well, but with much less computational cost-if implemented properly to take advantage of low level BLAS calculations on the GPU.

So, an Inception module was designed.

This module approximates a sparse Convolutional Net, but using many smaller, dense matrices, leading to many small filters of different sizes, concatenated together.

The Inception modules are then stacked on top of each other to give the full DNN.

GoogLeNet also replaces the later FC layers (i.e., in AlexNet-like architectures) with global average pooling, leaving only a single FC / Dense layer, which connects the DNN to the labels.

Being so deep, it is necessary to include an Auxiliary block that also connects to the labels, similar to the final FC layer.

From this, we can extract a single rectangular 768 × 1000 tensor.

This gives 2 FC layers to analyze.

For our analysis of InceptionV3 BID186 , we select a layer (L226) from in the Auxiliary block, as well as the final (L302) FC layer.

FIG1 presents the ESDs for InceptionV3 for Layer L226 and Layer L302, two large, fully-connected weight matrices with aspect ratios Q ≈ 1.3 and Q = 2.048, respectively.

We also show typical MP fits for matrices with the same aspect ratios 16 Indeed, these results suggest that Inception models do not truly account for all the correlations in the data.

Q. As with AlexNet, the ESDs for both the L226 and L302 layers display distinct and strong deviations from the MP theory.

L226: Bimodal ESDs.

Consider first L226 of InceptionV3.

FIG1 (a) displays the L226 ESD.

(Recall this is not a true Dense layer, but it is part of the Inception Auxiliary module, and it looks very different from the other FC layers, both in AlexNet and below.)

At first glance, we might hope to select the bulk edge at λ + ≈ 5 and treat the remaining eigenvalue mass as an extended spike; but this visually gives a terrible MP fit (not shown).

Selecting λ + ≈ 10 produces an MP fit with a reasonable shape to the envelope of support of the bulk; but this fit strongly over-estimates the bulk variance / Frobenius mass (in particular near λ ≈ 5), and it strongly under-estimates the spike near 0.

We expect this fit would fail any reasonable statistical confidence test for an MP distribution.

As in all cases, numerous Spikes extend all the way out to λ max ≈ 30, showing a longer, heavier tail than any MP fit.

It is unclear whether or not the edge statistics are TW.

There is no good MP fit for the ESD of L226, but it is unclear whether this distribution is "truly" Heavy-Tailed or simply appears Heavy-Tailed as a result of the bimodality.

Visually, at least the envelope of the L226 ESD to resembles a Heavy-Tailed MP distribution.

It is also possible that the DNN itself is also not fully optimized, and we hypothesize that further refinements could lead to a true Heavy-Tailed ESD.L302: Bimodal fat or Heavy-Tailed ESDs.

Consider next L302 of InceptionV3 (in FIG1 ).

The ESD for L302 is slightly bimodal (on a log-log plot), but nowhere near as strongly as L226, and we can not visually select any bulk edge λ + .

The bulk barely fits any MP density; our best attempt is shown.

Also, the global ESD the wrong shape; and the MP fit is concave near the edge, where the ESD is convex, illustrating that the edge decays into the tail.

For any MP fit, significant eigenvalue mass extends out continuously, forming a long tail extending al the way to λ max ≈ 23.

The ESD of L302 resembles that of the Heavy-Tailed FC2 layer of AlexNet, except for the small bimodal structure.

These initial observations illustrate that we need a more rigorous approach to make strong statements about the specific kind of distribution (i.e., Pareto vs other Heavy-Tailed) and what Universality class it may lay in.

We present an approach to resolve these technical details this in Section 5.5.

In addition to the models from Table 4 that we analyzed in detail, we have also examined the properties of a wide range of other pre-trained models, including models from both Computer Vision as well as Natural Language Processing (NLP).

This includes models trained on ImageNet, distributed with the pyTorch package, including VGG16, VGG19, ResNet50, InceptionV3, etc.

See Table 5 .

This also includes different NLP models, distributed in AllenNLP BID98 , including models for Machine Comprehension, Constituency Parsing, Semantic Role Labeling, Coreference Resolution, and Named Entity Recognition, giving a total of 84 linear layers.

See TAB8 .

Rather remarkably, we have observed similar Heavy-Tailed properties, visually and in terms of Power Law fits, in all of these larger, state-of-the-art DNNs, leading to results that are nearly universal across these widely different architectures and domains.

We have also seen Hard Rank deficiency in layers in several of these models.

We provide a brief summary of those results here.

Power Law Fits.

We have performed Power Law (PL) fits for the ESD of selected (linear) layers from all of these pre-trained ImageNet and NLP models.

17 Table 5 summarizes the detailed results for the ImageNet models.

Several observations can be made.

First, all of our fits, except for certain layers in InceptionV3, appear to be in the range 1.5 < α 3.5 (where the CSN method is known to perform well).

Second, we also check to see whether PL is the best fit by comparing the distribution to a Truncated Power Law (TPL), as well as an exponential, stretchexponential, and log normal distributions.

Column "Best Fit" reports the best distributional fit.

In all cases, we find either a PL or TPL fits best (with a p-value ≤ 0.05), with TPL being more common for smaller values of α.

Third, even when taking into account the large finite-size effects in the range 2 < α < 4, as illustrated in FIG20 , nearly all of the ESDs appear to fall into the 2 < µ < 4 Universality class.

FIG1 displays the distribution of PL exponents α for each set of models.

FIG1 shows the fit power law exponents α for all of the linear layers in pre-trained ImageNet models available in PyTorch (in Table 5 ), with Q 1; and FIG1 (b) shows the same for the pre-trained models available in AllenNLP (in TAB8 ).

Overall, there are 24 ImageNet layers with Q 1, and 82 AllenNet FC layers.

More than 80% of all the layers have α ∈ [2, 4] , and nearly all of the rest have α < 6.

One of these, InceptionV3, was discussed above, precisely since it was unusual, leading to an anomalously large value of α due to the dip in its ESD.Rank Collapse.

RMT also predicts that for matrices with Q > 1, the minimum singular value will be greater than zero, i.e., ν min > 0.

We test this by again looking at all of the FC layers in the pre-trained ImageNet and AllenNLP models.

See FIG1 for a summary of the results.

While the ImageNet models mostly follow this rule, 6 of the 24 of FC layers have ν min ∼ 0.

In fact, for 4 layers, ν min < 0.00001, i.e., it is close to the numerical threshold for 0.

In these few cases, the ESD still exhibits Heavy-Tailed properties, but the rank loss ranges from one eigenvalue equal to 0 up to 15% of the eigenvalue mass.

For the NLP models, we see no rank collapse, i.e., all of the 82 AllenNLP layers have ν min > 0.

In a few cases (e.g., LetNet5 in Section 4.1), MP theory appears to apply to the bulk of the ESD, with only a few outlying eigenvalues larger than the bulk edge.

In other more realistic cases (e.g., AlexNet Table 5 : Fit of PL exponents for the ESD of selected (2D Linear) layer weight matrices W l in pre-trained models distributed with pyTorch.

Layer is identified by the enumerated id of the pyTorch model; Q = N/M ≥ 1 is the aspect ratio; (M × N ) is the shape of W T l ; α is the PL exponent, fit using the numerical method described in the text; D is the Komologrov-Smirnov distance, measuring the goodness-of-fit of the numerical fitting; and "Best Fit" indicates whether the fit is better described as a PL (Power Law) or TPL (Truncated Power Law) (no fits were found to be better described by Exponential or LogNormal).

we have examined, as summarized in Section 4.4), the ESDs do not resemble anything predicted by standard RMT/MP theory.

This should not be unexpected-a well-trained DNN should have highly non-random, strongly-correlated weight matrices W, in which case MP theory would not seem to apply.

Moreover, except for InceptionV3, which was chosen to illustrate several unusual properties, nearly every DNN displays Heavy-Tailed properties such as those seen in AlexNet.

These empirical results suggest the following: first, that we can construct an operational and phenomenological theory (both to obtain fundamental insights into DNN regularization and to help guide the training of very large DNNs); and second, that we can build this theory by applying the full machinery of modern RMT to characterize the state of the DNN weight matrices.

For older and/or smaller models, like LeNet5, the bulk of their ESDs (ρ N (λ); λ λ + ) can be well-fit to theoretical MP density ρ mp (λ), potentially with several distinct, outlying spikes (λ > λ + ).

This is consistent with the Spiked-Covariance model of Johnstone BID115 , a simple perturbative extension of the standard MP theory.

18 This is also reminiscent of traditional Tikhonov regularization, in that there is a "size scale" (λ + ) separating signal (spikes) from noise (bulk).

In this sense, the small NNs of yesteryear-and smallish models used in many research studies-may in fact behave more like traditional ML models.

In the context of disordered systems theory, as developed by Sornette BID139 , this model is a form of Self-Organizaton.

Putting this all together demonstrates that the DNN training process itself engineers a form of implicit Self-Regularization into the trained model.

For large, deep, state-of-the-art DNNs, our observations suggest that there are profound deviations from traditional RMT.

These networks are reminiscent of strongly-correlated disorderedsystems that exhibit Heavy-Tailed behavior.

What is this regularization, and how is it related to our observations of implicit Tikhonov-like regularization on LeNet5?To answer this, recall that similar behavior arises in strongly-correlated physical systems, where it is known that strongly-correlated systems can be modeled by random matrices-with entries drawn from non-Gaussian Universality classes BID181 , e.g., PL or other Heavy-Tailed distributions.

Thus, when we observe that ρ N (λ) has Heavy-Tailed properties, we can hypothesize that W is strongly-correlated, 19 and we can model it with a Heavy-Tailed distribution.

Then, upon closer inspection, we find that the ESDs of large, modern DNNs behave as expected-when using the lens of Heavy-Tailed variants of RMT.

Importantly, unlike the Spiked-Covariance case, which has a scale cut-off (λ + ), in these very strongly Heavy-Tailed cases, correlations appear on every size scale, and we can not find a clean separation between the MP bulk and the spikes.

These observations demonstrate that modern, state-of-the-art DNNs exhibit a new form of Heavy-Tailed Self-Regularization.

In the next few sections, we construct and test (on miniature AlexNet) our new theory.

In this section, we develop an operational and phenomenological theory for DNN Self-Regularization that is designed to address questions such as the following.

How does DNN Self-Regularization differ between older models like LetNet5 and newer models like AlexNet or Inception?

What happens to the Self-Regularization when we adjust the numerous knobs and switches of the solver itself during SGD/Backprop training?

How are knobs, e.g., early stopping, batch size, and learning rate, related to more familiar regularizers like Weight Norm constraints and Tikhonov regularization?

Our theory builds on empirical results from Section 4; and our theory has consequences and makes predictions that we test in Section 6.MP Soft Rank.

We first define a metric, the MP Soft Rank (R mp ), that is designed to capture the "size scale" of the noise part of the layer weight matrix W l , relative to the largest eigenvalue of W T l W l .

Going beyond spectral methods, this metric exploits MP theory in an essential way.

Let's first assume that MP theory fits at least a bulk of ρ N (λ).

Then, we can identify a bulk edge λ + and a bulk variance σ 2 bulk , and define the MP Soft Rank as the ratio of λ + and λ max :MP Soft Rank : DISPLAYFORM0 Clearly, R mp ∈ [0, 1]; R mp = 1 for a purely random matrix (as in Section 5.1); and for a matrix with an ESD with outlying spikes (as in Section 5.3), λ max > λ + , and R mp < 1.

If there is no good MP fit because the entire ESD is well-approximated by a Heavy-Tailed distribution (as described in Section 5.5, e.g., for a strongly correlated weight matrix), then we can define λ + = 0 and still use Eqn.

FORMULA1 , in which case R mp = 0.

The MP Soft Rank is interpreted differently than the Stable Rank (R s ), which is proportional to the bulk MP variance σ 2 mp divided by λ max : DISPLAYFORM1 As opposed to the Stable Rank, the MP Soft Rank is defined in terms of the MP distribution, and it depends on how the bulk of the ESD is fit.

While the Stable Rank R s (M) indicates how many eigencomponents are necessary for a relatively-good low-rank approximation of an arbitrary matrix, the MP Soft Rank R mp (W) describes how well MP theory fits part of the matrix ESD ρ N (λ).

Empirically, R s and R mp often correlate and track similar changes.

Importantly, though, there may be no good low-rank approximation of the layer weight matrices W l of a DNNespecially a well trained one.

Visual Taxonomy.

We characterize implicit Self-Regularization, both for DNNs during SGD training as well as for pre-trained DNNs, as a visual taxonomy of 5+1 Phases of Training (Random-like, Bleeding-out, Bulk+Spikes, Bulk-decay, Heavy-Tailed, and Rankcollapse).

See Table 7 for a summary.

The 5+1 phases can be ordered, with each successive phase corresponding to a smaller Stable Rank / MP Soft Rank and to progressively more SelfRegularization than previous phases.

FIG1 depicts typical ESDs for each phase, with the MP fits (in red).

Earlier phases of training correspond to the final state of older and/or smaller models like LeNet5 and MLP3.

Later phases correspond to the final state of more modern models like AlexNet, Inception, etc.

Thus, while we can describe this in terms of SGD training, this taxonomy does not just apply to the temporal ordering given by the training process.

It also allows us to compare different architectures and/or amounts of regularization in a trained-or even pre-trained-DNN.

Each phase is visually distinct, and each has a natural interpretation in terms of RMT.

One consideration is the global properties of the ESD: how well all or part of the ESD is fit by an MP distribution, for some value of λ + , or how well all or part of the ESD is fit by a Heavy-Tailed or PL distribution, for some value of a PL parameter.

A second consideration is local properties of the ESD: the form of fluctuations, in particular around the edge λ + or around the largest eigenvalue λ max .

For example, the shape of the ESD near to and immediately above λ + is very different in FIG1 FIG1 and Sxn.

5.4Heavy-Tailed Table 7 : The 5+1 phases of learning we identified in DNN training.

We observed Bulk+Spikes and Heavy-Tailed in existing trained models (LeNet5 and AlexNet/InceptionV3, respectively; see Section 4); and we exhibited all 5+1 phases in a simple model (MiniAlexNet; see Section 7).As an illustration, FIG1 depicts the 5+1 phases for a typical (hypothetical) run of Backprop training for a modern DNN.

FIG1 (a) illustrates that we can track the decrease in MP Soft Rank, as W e l changes from an initial random (Gaussian-like) matrix to its final W l = W f l form; and FIG1 (b) illustrates that (at least for the early phases) we can fit its ESD (or the bulk of its ESD) using MP theory, with ∆ corresponding to non-random signal eigendirections.

Observe that there are eigendirections (below λ + ) that fit very well the MP bulk, there are eigendirections (well above λ + ) that correspond to a spike, and there are eigendirections (just slightly above λ + ) with (convex) curvature more like FIG1 Theory of Each Phase.

RMT provides more than simple visual insights, and we can use RMT to differentiate between the 5+1 Phases of Training using simple models that qualitatively describe the shape of each ESD.

In each phase, we model the weight matrices W as "noise plus signal," where the "noise" is modeled by a random matrix W rand , with entries drawn from the Gaussian Universality class (well-described by traditional MP theory) and the "signal" is a (small or very large) correction ∆ sig : Table 7 summarizes the theoretical model for each phase.

Each model uses RMT to describe the global shape of ρ N (λ), the local shape of the fluctuations at the bulk edge, and the statistics and information in the outlying spikes, including possible Heavy-Tailed behaviors.

DISPLAYFORM2 In the first phase (Random-like), the ESD is well-described by traditional MP theory, in which a random matrix has entries drawn from the Gaussian Universality class.

This does not mean that the weight matrix W is random, but it does mean that the signal in W is too weak to be seen when viewed via the lens of the ESD.

In the next phases (Bleeding-out, Bulk+Spikes), and/or for small networks such as LetNet5, ∆ is a relatively-small perturbative correction to W rand , and vanilla MP theory (as reviewed in Section 3.1) can be applied, as least to the bulk of the ESD.

In these phases, we will model the W rand matrix by a vanilla W mp matrix (for appropriate parameters), and the MP Soft Rank is relatively large (R mp (W) 0).

In the Bulk+Spikes phase, the model resembles a Spiked-Covariance model, and the SelfRegularization resembles Tikhonov regularization.

In later phases (Bulk-decay, Heavy-Tailed), and/or for modern DNNs such as AlexNet and InceptionV3, ∆ becomes more complex and increasingly dominates over W rand .

For these more strongly-correlated phases, W rand is relatively much weaker, and the MP Soft Rank collapses (R mp (W) → 0).

Consequently, vanilla MP theory is not appropriate, and instead the SelfRegularization becomes Heavy-Tailed.

In these phases, we will treat the noise term W rand as small, and we will model the properties of ∆ with Heavy-Tailed extensions of vanilla MP theory (as reviewed in Section 3.2) to Heavy-Tailed non-Gaussian universality classes that are more appropriate to model strongly-correlated systems.

In these phases, the strongly-correlated model is still regularized, but in a very non-traditional way.

The final phase, the Rank-collapse phase, is a degenerate case that is a prediction of the theory.

We now describe in more detail each phase in turn.

In the first phase, the Random-like phase, shown in FIG1 (a), the DNN weight matrices W resemble a Gaussian random matrix.

The ESDs are easily-fit to an MP distribution, with the same aspect ratio Q, by fitting the empirical variance σ 2 emp .

Here, σ 2 emp is the element-wise variance (which depends on the normalization of W).Of course, an initial random weight matrix W 0 l will show a near perfect MP fit.

Even in well trained DNNs, however, the empirical ESDs may be Random-like, even when the model has a non-zero, and even somewhat large, generalization accuracy.

20 That is, being fit well by an MP distribution does not imply that the weight matrix W is random.

It simply implies that W, while having structure, can be modeled as the sum of a random "noise" matrix W rand , with the same Q and σ 2 emp , and some small-sized matrix ∆ small , as: DISPLAYFORM0 where ∆ small represents "signal" learned during the training process.

In this case, λ max is sharply bounded, to within M DISPLAYFORM1 , to the edge of the MP distribution.

In the second phase, the Bleeding-out phase, shown in FIG1 , the bulk of the ESD still looks reasonably random, except for one or a small number K min{N, M } of eigenvalues that extend at or just beyond the MP edge λ + .

That is, for the given value of Q, we can choose a σ emp (or λ + ) parameter so that: (1) most of the ESD is well-fit; and (2) the part of the ESD that is not well-fit consists of a "shelf" of mass, much more than expected by chance, just above λ + : DISPLAYFORM0 This corresponds to modeling W as the sum of a random "noise" matrix W rand and some medium-sized matrix ∆ medium , as: DISPLAYFORM1 where ∆ medium represents "signal" learned during the training process.

As the spikes just begin to pull out from the bulk, i.e., when λ max − λ + is small, it may be difficult to determine unambiguously whether any particular eigenvalue is spike or bulk.

The reason is that, since the matrix is of finite size, we expect the spike locations to be Gaussiandistributed, with fluctuations of order N − 1 2 .

One option is to try to estimate σ bulk precisely from a single run.

Another option is to perform an ensemble of runs and plot ρ N R (λ) for the ensemble.

Then, if the model is in the Bleeding-out phase, there will be a small bump of eigenvalue mass, shaped like a Gaussian, 21 which is very close to but bleeding-out from the bulk edge.

When modeling DNN training in terms of RMT and MP theory, the transition from Randomlike to Bleeding-out corresponds to the so-called BPP phase transition [9, 23, 45, 20] .

This transition represents a "condensation" of the eigenvector corresponding to the largest eigenvalue λ max onto the eigenvalue of the rank-one (or, more generally, rank-k, if the perturbation is higher rank) perturbation ∆ [23] .

In the third phase, the Bulk+Spikes phase, shown in FIG1 , the bulk of the ESD still looks reasonably random, except for one or a small number K min{N, M } of eigenvalues that extend well beyond the MP edge λ + .

That is, for the given value of Q, we can choose a σ emp (or λ + ) parameter so that: (1) most of the ESD is well-fit; and (2) the part of the ESD that is not well-fit consists of several (K) eigenvalues, or Spikes, that are much larger than λ + : DISPLAYFORM0 This corresponds to modeling W as the sum of a random "noise" matrix W rand and some moderately large-sized matrix ∆ large , as: DISPLAYFORM1 where ∆ large represents "signal" learned during the training process.

For a single run, it may be challenging to identify the spike locations unambiguously.

If we perform an ensemble of runs, however, then the Spike density is clearly visible, distinct, and separated from bulk, although it is much smaller in total mass.

We can try to estimate σ bulk precisely, but in many cases we can select the edge of the bulk, λ + , by visual inspection.

As in the Bleeding-out phase, the empirical bulk variance σ 2 bulk is smaller than both the full elementwise variance, σ 2 bulk < σ 2 f ull , and the shuffled variance (fit to the MP bulk), σ 2 bulk < σ 2 shuf , because we remove several large eigendirections from the bulk. (See Section 6.3 for more on this.)When modeling DNN training in terms of RMT and MP theory, the Bulk+Spikes phase corresponds to vanilla MP theory plus a large low-rank perturbation, and it is what we observe in the LeNet5 model.

In statistics, this corresponds to the Spiked Covariance model BID115 BID156 BID116 .

Relatedly, in the Bulk+Spikes phase, we see clear evidence of Tikhonov-like Self-Regularization.

MP theory with large low-rank perturbations.

To understand, from the perspective of MP theory, the properties of the ESD as eigenvalues bleed out and start to form spikes, consider modeling W as W W rand + ∆ large .

If ∆ is a rank-1 perturbation 22 , but now larger, then one can show that the maximum eigenvalue λ max that bleeds out will extend beyond theoretical MP bulk edge λ + and is given by DISPLAYFORM2 Here, by σ 2 , we mean the theoretical variance of the un-perturbed W rand .

23 Moreover, in an ensemble of runs, each of these Spikes will have Gaussian fluctuations on the order N −1/2 .Eigenvector localization.

Eigenvector localization on extreme eigenvalues can be a diagnostic for Spike eigenvectors (as well as for extreme eigenvectors in the Heavy-Tailed phase).

The interpretation is that when the perturbation ∆ is large, "information" in W will concentrate on a small number of components of the eigenvectors associated with the outlier eigenvalues.

The fourth phase, the Bulk-decay phase, is illustrated in FIG1 (d), and is characterized by the onset of Heavy-Tailed behavior, both in the very long tail, and at the Bulk edge.

24 The Bulkdecay phase is intermediate between having a large, low-rank perturbation ∆ to an MP Bulk (as in the Bulk+Spikes phase) and having strong correlations at all scales (as in the Heavy-Tailed phase).

Viewed naïvely, the ESDs in Bulk-decay resemble a combination of the Bleeding-out and Bulk+Spikes phases: there is a large amount of mass above λ + (from any reasonable MP fit); and there are a large number of eigenvectors much larger than this value of λ + .

However, quantitatively, the ESDs are quite different than either Bleeding-out or Bulk+Spikes: there is much more mass bleeding-out; there is much greater deterioration of the Bulk; and the Spikes lie much farther out.

In Bulk-decay, the Bulk region is both hard to identify and difficult to fit with MP theory.

Indeed, the properties of the Bulk start to look less and less consistent the an MP distribution (with elements drawn from the Universality class of Gaussian matrices), for any parameter values.

This implies that λ max can be quite large, in which case the MP Soft Rank is much smaller.

The best MP fit neglects a large part of the eigenvalue mass, and so we usually have to select λ + numerically.

Most importantly, the mass at the bulk edge now starts to exhibit Heavy-Tailed, not Gaussian, properties; and the overall shape of the ESD is itself taking on a Heavy-Tailed form.

Indeed, the ESDs are may be consistent with (weakly) Heavy-Tailed (4 < µ) Universality class, in which the local edge statistics exhibit Heavy-Tailed behavior due to finite-size effects.

25 22 More generally, ∆ may be a rank-k perturbation, for k M , and similar results should hold.

23 In typical theory, this is scaled to unity (i.e., σ 2 = 1).

In typical practice, we do not a priori know σ 2 , and it may be non-trivial to estimate because the scale W may shift during Backprop training.

As a rule-of-thumb, one can select the bulk edge λ + well to provide a good fit for the bulk variance σ 2 bulk .

24 We observe Bulk-decay in InceptionV3 FIG1 ).

This may indicate that this model, while extremely good, might actually lend itself to more fine tuning and might not be fully optimized.

25 Making this connection more precise-e.g., measuring α in this regime, relating α to µ in this regime, having precise theory for finite-size effects in this regime, etc.-is nontrivial and left for future work.

The final of the 5 main phases, the Heavy-Tailed phase, is illustrated in FIG1 (e).

This phase is formally, and operationally, characterized by an ESD that resembles the ESD of a random matrix in which the entries are drawn i.i.d.

from a Heavy-Tailed distribution.

This phase corresponds to modeling W as the sum of a small "noise" matrix W rand and a large "stronglycorrelated" matrix ∆ str.corr. , as: DISPLAYFORM0 where ∆ str.corr.

represents strongly-correlated "signal" learned during the training process.

26 As usual, W rand can be modeled as a random matrix, with entries drawn i.i.d.

from a distribution in the Gaussian Universality class.

Importantly, the strongly-correlated signal matrix ∆ str.corr. can also be modeled as a random matrix, but (as described in Section 3.2) one with entries drawn i.i.d.

from a distribution in a different, Heavy-Tailed, Universality class.

In this phase, the ESD visually appears Heavy-Tailed, and it is very difficult if not impossible to get a reasonable MP fit of the layer weight matrices W (using standard Gaussian-based MP/RMT).

Thus, the matrix W has zero (R mp (W) = 0) or near-zero (R mp (W) 0) MP Soft Rank; and it has intermediate Stable Rank (1 R s (W) min{N, M }).

27 When modeling DNN training in terms of RMT and MP theory, the Heavy-Tailed phase corresponds to the variant of MP theory in which elements are chosen from a non-Gaussian Universality class [38, 20, 19, BID158 7, 40, 8, 26, 22, 21] .

In physics, this corresponds to modeling strongly-correlated systems with Heavy-Tailed random matrices BID181 23] .

Relatedly, in the Heavy-Tailed phase, the implicit Self-Regularization is strongest.

It is, however, very different than the Tikhonov-like regularization seen in the Bulk+Spikes phases.

Although there is a decrease in the Stable Rank (for similar reasons to why it decreases in the Bulk+Spikes phases, i.e., Frobenius mass moves out of the bulk and into the spikes), Heavy-Tailed Self-Regularization does not exhibit a "size scale" in the eigenvalues that separates the signal from the noise.

28 Heavy-Tailed ESDs.

Although FIG1 (e) is presented on the same linear-linear plot as the other subfigures in FIG1 , the easiest way to compare Heavy-Tailed ESDs is with a log-log histogram and/or with PL fits.

Consider FIG1 (a), which displays the ESD for FC3 of pretrained AlexNet, as a log-log histogram; and consider also FIG1 (b), which displays an overlay (in red) of a log-log histogram of the ESD of a random matrix M. This matrix M has the same aspect ratio as W F C3 , but the elements M i,j are drawn from a Heavy-Tailed Pareto distribution, Eqn. (10), with µ = 2.5.

We call ESDs such as W F C3 of AlexNet Heavy-Tailed because they resemble the ESD of a random matrix with entries drawn from a Heavy-Tailed distribution, as observed with a log-log histogram.

29 We can also do a PL fit to estimate α and then try to estimate the Universality class we are in.

Our PL estimator works well for µ ∈ [1.5, 3.5]; but, due to large finite-size effects, it is difficult to determine µ from α precisely.

This is discussed in more detail in Section 3.2.

As a rule of thumb, if α < 2, then we can say α ≈ 1 + µ/2, and we are in the (very) Heavy-Tailed Universality class; and if 2 < α < 4, but not too large, then α is well-modeled by α ≈ b + aµ, and we are mostly likely in the (moderately, or "fat") Heavy-Tailed Universality class.

In addition to the 5 main phases, based on MP theory we also expect the existence of an additional "+1" phase, which we call the Rank-collapse Phase, and which is illustrated in FIG1 (f).

For many parameter settings, the minimum singular value (i.e., λ − in Eqn.

FORMULA18 for vanilla MP theory) is strictly positive.

For certain parameter settings, the MP distribution has a spike at the origin, meaning that there is a non-negligible mass of eigenvalues equal to 0, i.e., the matrix is rank-deficient, i.e., Hard Rank is lost.

30 For vanilla Gaussian-based MP theory, this happens when Q > 1, and this phenomenon exists more generally for Heavy-Tailed MP theory.

In this section, we validate and illustrate how to use our theory from Section 5.

This involved extensive training and re-training, and thus we used the smaller MiniAlexNet model.

Section 6.1 describes the basic setup; Section 6.2 presents several baseline results; Section 6.3 provides some important technical details; and Section 6.4 describes the effect of adding explicit regularization.

We postpone discussing the effect of changing batch size until Section 7.

Here, we describe the basic setup for our empirical evaluation.

Model Deep Neural Network.

We analyzed MiniAlexNet, 31 a simpler version of AlexNet, similar to the smaller models used in BID203 , scaled down to prevent overtraining, and trained on CIFAR10.

The basic architecture follows the same general design as older NNs such as LeNet5, VGG16, and VGG19.

It is illustrated in FIG1 .

It consists of two 2D Convolutional layers, each with Max Pooling and Batch Normalization, giving 6 initial layers; it then has two Fully Connected (FC), or Dense, layers with ReLU activations; and it then has a final FC layer added, with 10 nodes and softmax activation.

For the FC layers: DISPLAYFORM0 The W F C1 and W F C2 matrices are initialized with a Glorot normalization BID101 .

32 We apply Batch Normalization to the Conv2D layers, but we leave it off the FC layer; results do not change if remove all Batch Normalization.

All models are trained using Keras 2.x, with TensorFlow as a backend.

We use SGD with momentum, with a learning rate of 0.01, a momentum parameter of 0.9, and a baseline batch size of 32; and we train up to 100 epochs.

To compare different batch sizes and other tunable knobs, we employed early stopping criteria on the total loss which causes termination at fewer than 100 epochs.

We save the weight matrices at the end of every epoch, and we study the complexity of the trained model by analyzing the empirical properties of the W F C1 and W F C2 matrices.

Experimental Runs.

It is important to distinguish between several different types of analysis.

First, analysis of ESDs (and related quantities) during Backprop training during 1 training run.

In this case, we consider a single training run, and we monitor empirical properties of weight matrices as they change during the training process.

Second, analysis of the final ESDs from 1 training run.

In this case, we consider a single training run, and we analyze the empirical properties of the single weight matrix that is obtained after the training process terminates.

This is similar to analyzing pre-trained models.

Third, analysis of an ensemble of final ESDs from N R training runs.

In this case, we rerun the model N R ∼ 10-100 times, using different initial random weight matrices W 0 l , and we form an ensemble of N R of final weight matrices [W DISPLAYFORM1 ].

We do this in order to compensate for finite-size effects, to provide a better visual interpretation of our claims, and to help clarify our scientific claims about the learning process.

Of course, as 32 Here and elsewhere, most DNNs are initialized with random weight matrices, e.g., as with the Glorot Normal initialization BID101 (which involves a truncation step).

If we naïvely fit the MP distribution to W 0 trunc , then the empirical variance will be larger than one, i.e., σ 2 emp > 1.

That is because the Glorot normalization is 2/N + M , whereas the MP theory is presented with normalization √ N −1 .

To apply MP theory, we must rescale our empirically-fit σ an engineering matter, one wants exploit our results on a single "production" run of the training process.

In that case, we expect to observe (and do observe) a noisy version of what we present.

33 Empirically Measured Quantities.

We compute several RMT-based quantities of interest for each layer weight matrices W l , for layers l = F C1, F C2, including the following: Matrix complexity metrics, such as the Matrix Entropy S(W e l ), Hard Rank R(W e l ), Stable Rank R e s (W l ), and MP Soft Rank R e mp (W l ); ESDs, ρ(λ) for a single run, both during Backprop training and for the final weight matrices, and/or ρ N R (λ) for the final states an ensemble of N R runs; and Eigenvector localization metrics, including the Generalized Vector Entropy S(x), Localization Ratio L(x), and Participation Ratio P(x), of the eigenvectors of X, for an ensemble of runs.

Knobs and Switches of the Learning Process.

We vary knobs and switches of the training process, including the following: number of epochs (typically ≈ 100, well past when entropies and measured training/test accuracies saturate); Weight Norm regularization (on the fully connected layers-in Keras, this is done with an L 2 -Weight Norm kernel regularizer, with value 0.0001); various values of Dropout; and batch size 34 (varied between 2 to 1024).

Here, we present several baseline results for our RMT-based analysis of MiniAlexNet.

For our baseline, the batch size is 16; and Weight Norm regularization, Dropout, and other explicit forms of regularization are not employed.

FIG1 shows the Matrix Entropy (S(W)) and Stable Rank (R s (W)) for layers FC1 and FC2, as well as of the training and test accuracies, for MiniAlexNet, as a function of the number of epochs.

This is for an ensemble of N R = 10 runs.

Both layers start off with an Entropy close to but slightly less than 1.0; and both retrain full rank during training.

For each layer, the matrix Entropy gradually lowers; and the Stable Rank shrinks, but more prominently.

These decreases parallel the increase in training and test accuracies, and both complexity metrics level off as the training/test accuracies do.

The Matrix Entropy decreases relatively more for FC2, and the Stable Rank decreases relatively more for FC1; but they track the same gross changes.

The large difference between training and test accuracy should not be surprising since-for these baseline results-we have turned off regularization like removing Batch Norm, Dropout layers, and any Weight Norm constraints.

Eigenvalue Spectrum: Comparisons with RMT.

FIG1 show, for FC1 and FC2, respectively, the layer matrix ESD, ρ(λ), every few epochs during the training process.

For layer FC1 (with Q ≈ 10.67), the initial weight matrix W 0 looks very much like an MP distribution (with Q ≈ 10.67), consistent with a Random-like phase.

Within a very few epochs, however, eigenvalue mass shifts to larger values, and the ESD looks like the Bulk+Spikes phase.

Once the Spike(s) appear(s), substantial changes are hard to see in FIG1 , but minor changes do continue in the ESD.

Most notably, λ max increases from roughly 3.0 to roughly 4.0 during training, indicating further Self-Regularization, even within the Bulk+Spikes phase.

For layer FC2 (with Q = 2), the initial weight matrix also resembles an MP distribution, also consistent with a Random-like phase, but with a much smaller value of Q than FC1 (Q = 2 here).

Here too, the ESD changes during the first few epochs, after which there are not substantial changes.

The most prominent change is that eigenvalue mass pulls out slightly from the bulk and λ max increases from roughly 3.0 to slightly less than 4.0.Eigenvector localization.

FIG1 plots three eigenvector localization metrics, for an ensemble N R = 10 runs, for eigenvectors in the bulk and spike of layer FC1 of MiniAlexNet, after training.

35 Spike eigenvectors tend to be more localized than bulk eigenvectors.

This effect is less pronounced for FC2 (not shown) since the spike is less well-separated from the bulk.

35 More precisely, bulk here refers to eigenvectors associated with eigenvalues less than λ + , defined below and illustrated in FIG5 , and spike here refers to those in the main part of the spike.

There are several technical issues with applying RMT that we discuss here.

36 • Single run versus an ensemble of runs.

FIG5 shows ESDs for Layer FC1 before and after training, for a single run, as well as after training for an ensemble of runs.

FIG5 does the same for FC2.

There are two distinct effects of doing an ensemble of runs: first, the histograms get smoother (which is expected); and second, there are fluctuations in λ max .

These fluctuations are not due to finite-size effects; and they can exhibit Gaussian or TW or other Heavy-Tailed properties, depending on the phase of learning.

Thus, they can be used as a diagnostic, e.g., to differentiate between Bulk+Spikes versus Bleeding-out.

36 While we discuss these in the context of our baseline, the same issues arise in all applications of our theory.

• Finite-size effects.

FIG5 (a) shows that we can estimate finite-size effects in RMT by shuffling the elements of a single weight matrices W l → W shuf l and recomputing the eigenvalue spectrum ρ shuf (λ) of X shuf .

We expect ρ shuf (λ) to fit an MP distribution well, even for small sample sizes, and we see that it does.

We also expect and see a very crisp edge in λ + .

More generally, we can visually observe the quality of the fit at this sample size to gauge whether deviations are likely spurious.

This is relatively-easy to do for Random-like, and also for Bulk+Spikes (since we can simply remove the spikes before shuffling).

For Bleeding-out and Bulk-decay, it is somewhat more difficult due to the need to decide which eigenvalues to keep in the bulk.

For Heavy-Tailed, it is much more complicated since finite-size effects are larger and more pronounced.• Fitting the bulk edge λ + , i.e., the bulk variance σ 2 bulk .

Estimating λ + (or, equivalently, σ 2 bulk ) can be tricky, even when the spike is well-separated from the bulk.

We illustrate this in FIG5 .

In particular, compare FIG5 , λ + is chosen to reproduce very well the bulk edge of the ESD, at the expense of having some "missing mass" in the ESD just below λ + (leading to a "Bulk + 9 Spikes" model).

In FIG5 (b), λ + is chosen to reproduce very well the ESD just below λ + , at the expense of having a slight bleeding-out region just above λ + (leading to a "Bulk + 18 Spikes" or a "Bulk + 9 Bleeding-out + 9 Spikes" model).

If we hypothesize that a MP distribution fits the bulk very well, then the fit in FIG5 (b) is more appropriate, but FIG5 (b) shows this can be challenging to identify in a single run.

We recommend choosing the bulk maximum λ + and (from Eqn.

(9)) selecting σ 2 bulk as σ 2 bulk = λ + 1 + 1/ √ Q −2 .

In fitting σ 2 bulk , we expect to lose some variance due to the eigenvalue mass that "bleeds out" from the bulk (e.g., due to Bleeding-out or Bulkdecay), relative to a situation where the MP distribution provides a good fit for the entire ESD (as in Random-like).

Rather than fitting σ 2 mp directly on the ESD of W l , without removing the outliers (which may thus lead to poor estimates since λ max is particularly large), we can always define a baseline variance for any weight matrix W by shuffling it elementwise W → W shuf , and then finding the MP σ 2 shuf from the ESD of W shuf .

In doing so, the Frobenius norm is preserved W shuf l F = W l F , thus providing a way to (slightly over-) estimate the unperturbed variance of W l for comparison.

37 Since at least one eigenvalue bleeds out, σ 2 bulk < σ 2 shuf , i.e., the empirical bulk variance σ 2 bulk will always be (slightly) less that than shuffled bulk variance σ 2 shuf .The best way to automate these choices, e.g., with a kernel density estimator, remains open.

We consider here how explicit regularization affects properties of learned DNN models, in light of baseline results of Section 6.2.

We focus on L 2 Weight Norm and Dropout regularization.

37 We suggest shuffling W l at least 100 times then fitting the ESD to obtain an σ 2 .

We can then estimate σ 2 bulk as σ 2 minus a contribution for each of the K bleeding-out eigenvalues, giving, as a rule of thumb, σ Transition in Layer Entropy and Stable Rank.

See FIG5 for plots for FC1 and FC2 when L 2 norm weight regularization is included; and see FIG5 for plots when Dropout regularization is included.

In both cases, baseline results are provided, and compare with FIG1 .

In each case, we observe a greater decrease in the complexity metrics with explicit regularization than without, consistent with expectations; and we see that explicit regularization affects these metrics dramatically.

Here too, the Layer Entropy decreases relatively more for FC2, and the Stable Rank decreases relatively more for FC1.

DISPLAYFORM0 Eigenvalue Spectrum: Comparisons with RMT.

See FIG5 for the ESD for layers FC1 and FC2 of MiniAlexNet, with explicit Dropout, including MP fits to a bulk when 9 or 10 spikes are removed.

Compare with FIG5 (for FC1) and FIG5 (for FC2).

Note, in particular, the differences in the scale of the X axis.

FIG5 shows that when explicit Dropout regularization is added, the eigenvalues in the spike are pulled to much larger values (consistent with a much more implicitly-regularized model).

A subtle but important consequence of this regularization 38 is the following: this leads to a smaller bulk MP variance parameter σ 2 mp , and thus smaller values for λ + , when there is a more prominent spike.

See FIG5 for similar results for the ESD for layers FC1 and FC2 of MiniAlexNet, with explicit L 2 norm weight regularization.

Eigenvalue localization.

We observe that eigenvector localization tends to be more prominent when the explicit regularization is stronger, presumably since explicit (L 2 Weight Norm or Dropout) regularization can make spikes more well-separated from the bulk.

In this section, we demonstrate that we can exhibit all five of the main phases of learning by changing a single knob of the learning process.

39 We consider the batch size (used in the construction of mini-batches during SGD training) since it is not traditionally considered a regularization parameter and due to its its implications for the generalization gap phenomenon.

The Generalization Gap refers to the peculiar phenomena that DNNs generalize significantly less well when trained with larger mini-batches (on the order of 10 3 − 10 4 ) BID127 BID111 BID119 BID104 .

Practically, this is of interest since smaller batch sizes makes training large DNNs on modern GPUs much less efficient.

Theoretically, this is of interest since it contradicts simplistic stochastic optimization theory for convex problems.

The latter suggests that larger batches should allow better gradient estimates with smaller variance and should therefore improve the SGD optimization process, thereby increasing, not decreasing, the generalization performance.

For these reasons, there is interest in the question: what is the mechanism responsible for the drop in generalization in models trained with SGD methods in the large-batch regime?To address this question, we consider here using different batch sizes in the DNN training algorithm.

We trained the MiniAlexNet model, just as in Section 6 for the Baseline model, except with batch sizes ranging from moderately large to very small (b ∈ {500, 250, 100, 50, 32, 16, 8, 4, 2}).

39 We can also exhibit the "+1" phase, but in this section we are interested in changing only the batch size.and then it begins to decrease; and test accuracy actually increases for extremely small b, and then it gradually decreases as b increases.

• At batch size b = 250 (and larger), the ESD resembles a pure MP distribution with no outliers/spikes; it is Random-like.• As b decreases, there starts to appear an outlier region.

For b = 100, the outlier region resembles Bleeding-out.• Then, for b = 32, these eigenvectors become well-separated from the bulk, and the ESD resembles Bulk+Spikes.• As batch size continues to decrease, the spikes grow larger and spread out more (observe the increasing scale of the X-axis), and the ESD exhibits Bulk-decay.• Finally, at the smallest size, b = 2, extra mass from the main part of the ESD plot almost touches the spike, and the curvature of the ESD changes, consistent with Heavy-Tailed.

While the shape of the ESD is different for FC2 (since the aspect ratio of the matrix is less), very similar properties are observed.

In addition, as b decreases, some of the extreme eigenvectors associated with eigenvalues that are not in the bulk tend to be more localized.

Implications for the generalization gap.

Our results here (both that training/test accuracies decrease for larger batch sizes and that smaller batch sizes lead to more well-regularized models) demonstrate that the generalization gap phenomenon arises since, for smaller values of the batch size b, the DNN training process itself implicitly leads to stronger Self-Regularization.(Depending on the layer and the batch size, this Self-Regularization is either the more traditional Tikhonov-like regularization or the Heavy-Tailed Self-Regularization corresponding to strongly-correlated models.)

That is, training with smaller batch sizes implicitly leads to more well-regularized models, and it is this regularization that leads to improved results.

The obvious mechanism is that, by training with smaller batches, the DNN training process is able to "squeeze out" more and more finer-scale correlations from the data, leading to more strongly-correlated models.

Large batches, involving averages over many more data points, simply fail to see this very fine-scale structure, and thus they are less able to construct strongly-correlated models characteristic of the Heavy-Tailed phase.

Our results also suggest that, if one hopes to compensate for this by decreasing the learning rate, then one would have to decrease the learning rate by an extraordinary amount.

There is a large body of related work, much of which either informed our approach or should be informed by our results.

This includes: work on large-batch learning and the generalization gap BID195 BID119 BID111 BID104 BID114 BID178 BID113 BID193 BID141 BID198 BID199 ; work on Energy Landscape approaches to NN training BID117 BID196 44, BID170 30, 29, 27, BID112 47, 12, BID151 BID197 BID97 BID130 BID129 BID142 ; work on using weight matrices or properties of weight matrices [15, BID149 BID150 4, 14, BID200 BID148 3, BID133 BID145 ; work on different Heavy-Tailed Universality classes [46, 32, 18, 25, 20, 5, BID158 7, 38, 17, BID137 8, BID146 ; other work on RMT approaches BID180 BID167 BID161 BID159 BID134 BID134 BID184 BID132 BID173 ; other work on statistical physics approaches BID179 BID99 BID164 BID172 BID184 BID166 BID160 ; work on fitting to noisy versus reliable signal BID183 BID203 BID122 BID169 6] ; and several other related lines of work BID110 BID143 BID163 34, 1, BID153 BID154 BID144 BID131 .

We conclude by discussing several aspects of our results in this broader context.

Failures of VC theory.

In light of our results, we have a much better understanding of why VC theory does not apply to NNs.

VC theory assumes, at its core, that a learning algorithm could sample a very large, potentially infinite, space of hypothesis functions; and it then seeks a uniform bound on this process to get a handle on the generalization error.

It thus provides a very lose, data-independent bound.

Our results suggest a very different reason why VC theory would fail than is sometimes assumed: naïvely, the VC hypothesis space of a DNN would include all functions described by all possible values of the layer weight matrices (and biases).

Our results suggest, in contrast, that the actual space is in some sense "smaller" or more restricted than this, in that the FC layers (at least) cover only one Universality class-the class of Heavy (or Fat) Tailed matrices, with PL exponent µ ∈ [2, 4] .

During the course of training, the space becomes smaller-through Self-Regularization-since even if the initial matrices are random, the class of possible final matrices is very strongly correlated.

The process of Self-Regularization and Heavy-Tailed Self-Regularization collapses the space of available functions that can be learned.

Indeed, this also suggests why transfer learning is so effective-the initial weigh matrices are much closer to their final versions, and the space of functions need not shrink so much.

The obvious conjecture is that what we have observed is characteristic of general NN/DNN learning systems.

Since there is nothing like this in VC theory, our results suggest revisiting more generally the recent suggestions of BID140 .Information bottleneck.

Recent empirical work on modern DNNs has shown two phases of training: an initial "fast" phase and a second "slower" phase.

To explain this, Tishby et al. BID188 BID176 have suggested using the Information Bottleneck Theory for DNNs.

See also BID187 BID175 BID171 BID201 .

While this theory may be controversial, the central concept embodies the old thinking that DNNs implicitly lose some capacity (or information/entropy) during training.

This is also what we observe.

Two important differences with our approach are the following: we provide a posteriori guarantees; and we provide an unsupervised theory.

An a posteriori unsupervised theory provides a mechanism to minimize the risk of "label leakage," clearly a practical problem.

The obvious hypothesis is that the initial fast phase corresponds to the initial drop in entropy that we observe (which often corresponds to a Spike pulling out of the Bulk), and that the second slower phase corresponds to "squeezing out" more and more correlations from the data (which, in particular, would be easier with smaller batches than larger batches, and which would gradually lead to a very strongly-correlated model that can then be modeled by Heavy-Tailed RMT).Energy landscapes and rugged convexity.

Our observations about the onset of HeavyTailed or scale-free behavior in realistic models suggest that (relatively) simple (i.e., Gaussian) Spin-Glass models, used by many researchers, may lead to very misleading results for realistic systems.

Results derived from such models are very specific to the Gaussian Universality class; and other Spin-Glass models can show very different behaviors.

In particular, if we select the elements of the Spin-Glass Hamiltonian from a Heavy-Tailed Levy distribution, then the local minima do not concentrate near the global minimum [31, 32] .

See also BID196 BID96 48, 28, BID185 .

Based on this, as well as the results we have presented, we expect that well-trained DNNs will exhibit a ruggedly convex global energy landscape, as opposed to a landscape with a large number of very different degenerate local minima.

This would clearly provide a way to understand phenomena exhibited by DNN learning that are counterintuitive from the perspective of traditional ML BID140 .Connections with glass theory.

It has been suggested that the slow training phase arises because the DNN optimization landscape has properties that resemble a glassy system (in the statistical physics sense), meaning that the dynamics of the SGD is characterized by slow HeavyTailed or PL behavior.

See [13, BID119 BID111 10] -and recall that, while this connection is sometimes not explicitly noted, glasses are defined in terms of their slow dynamics.

Using the glass analogy, however, it can also shown that very large batch sizes can, in fact, be used-if one adjusts the learning rate (potentially by an extraordinary amount).

For example, it is argued that, when training with larger batch sizes, one needs to change the learning rate adaptively in order to take effectively more times steps to reach a obtain good generalization performance.

Our results are consistent with the suggestion that DNNs operate near something like a finite size form of a spin-glass phase transition, again consistent with previous work BID140 .

This is likewise similar in spirit to how certain spin glass models are Bayes optimal in that their optimal state lies on the Nishimori Line BID152 .

Indeed, these ideas have been a great motivation in looking for our empirical results and formulating our theory.

Self-Organization in Natural (and Engineered) Phenomena.

Typical implementations of Tikhonov regularization require setting a specific regularization parameter or regularization size scale, whereas Self-Regularization just arises as part of the DNN training process.

A different mechanism for this has been described by Sornette, who suggests it can arise more generally in natural Self-Organizing systems, without needing to tune specific exogenous control parameters BID181 .

Such Self-Organization can manifest itself as Bulk+Spikes BID139 , as true (infinite order) Power Laws, or as a finite-sized Heavy-Tailed (or Fat-Tailed) phenomena BID181 .

This corresponds to the three Heavy-Tailed Universality classes we described.

To the best of our knowledge, ours is the first observation and suggestion that a Heavy-Tailed ESD could be a signature/diagnostic for such Self-Organization.

That we are able to induce both Bulk+Spikes and Heavy-Tailed Self-Organization by adjusting a single internal control parameter (the batch size) suggests similarities between Self-Organized Criticality (SOC) [11] (a very general phenomena also thought to be "a fundamental property of neural systems" more generally BID109 36] ) and modern DNN training.

There are subtle issues that make RMT particularly appropriate for analyzing weight matrices.

Taking the right limit.

The matrix X is an empirical correlation matrix of the weight layer matrix W l , akin to an estimator of the true covariance of the weights.

It is known, however, that this estimator is not good, unless the aspect ratio is very large (i.e., unless Q = N/M 1, in which case X l is very tall and thin).

The limit Q → ∞ (e.g., N → ∞ for fixed M ) is the case usually considered in mathematical statistics and traditional VC theory.

For DNNs, however, M ∼ N , and so Q = O(1); and so a more appropriate limit to consider is (M → ∞, N → ∞) such that Q is a fixed constant BID140 .

This is the regime of MP theory, and this is why deviations from the limiting MP distribution provides the most significant insights here.

Relation to the SMTOG.

In recent work BID140 , Martin and Mahoney examined DNNs using the Statistical Mechanics Theory of Generalization (SMTOG) BID174 BID194 BID107 43] .

As with RMT, the STMOG also applies in the limit (M → ∞, N → ∞) such that Q = 1 or Q = O(1), i.e., in the so-called Thermodynamic Limit.

Of course, RMT has a long history in theoretical physics, and, in particular, the statistical mechanics of the energy landscape of strongly-correlated disordered systems such as polymers.

For this reason, we believe RMT will be very useful to study broader questions about the energy landscape of DNNs.

Martin and Mahoney also suggested that overtrained DNNs-such as those trained on random labelings-may effectively be in a finite size analogue of the (mean field) spin glass phase of a neural network, as suggested by the SMTOG BID140 .

We should note that, in this phase, self-averaging may (or may not) break down.

The importance of Self-Averaging.

Early RMT made use of replica-style analysis from statistical physics BID174 BID194 BID107 43] , and this assumes that the statistical ensemble of interest is Self-Averaging.

This property implies that the theoretical ESD ρ(λ) is independent of the specific realization of the matrix W, provided W is a typical sample from the true ensemble.

In this case, RMT makes statements about the empirical ESD ρ N (λ) of a large random matrix like X, which itself is drawn from this ensemble.

To apply RMT, we would like to be able inspect a single realization of W, from one training run or even one epoch of our DNN.

If our DNNs are indeed self-averaging, then we may confidently interpret the ESDs of the layer weight matrices of a single training run.

As discussed by Martin and Mahoney, this may not be the case in certain situations, such as severe overtraining BID140 .

From the SMTOG perspective, NN overfitting, 40 which results in NN overtraining, 41 is an example of non-self-averaging.

When a NN generalizes well, it can presumably be trained, using the same architecture and parameters, on any large random subset of the training data, and it will still perform well on any test/holdout example.

In this sense, the trained NN is a typical random draw from the implicit model class.

In contrast, an overtrained model is when this random draw from this implicit model class is atypical, in the sense that it describes well the training data, but it describes poorly test data.

A model can enter the spin glass phase when there is not enough training data and/or the model is too complicated BID174 BID194 BID107 43] .

The spin glass phase is (frequently) non-self-averaging, and this is why overtraining was traditionally explained using spin glass models from statistical mechanics.

42 For this reason, it is not obvious that RMT can be applied to DNNs that are overtrained; we leave this important subtly for future work.

Our practical theory opens the door to address very practical questions, including the following.• What are design principles for good models?

Our approach might help to incorporate domain knowledge into DNN structure as well as provide finer metrics (beyond simply depth, width, etc.) to evaluate network quality.• What are ways in which adversarial training/learning or training/learning in new environments affects the weight matrices and thus the loss surface?

Our approach might help characterize robust versus non-robust and interpretable versus non-interpretable models.• When should training be discontinued?

Our approach might help to identify empirical properties of the trained models, e.g., of the weight matrices-without explicitly looking at labels-that will help to determine when to stop training.

Finally, one might wonder whether our RMT-based theory is applicable to other types of layers such as convolutional layers and/or other types of data such as natural language data.

Initial results suggest yes, but the situation is more complex than the relatively simple picture we have described here.

These and related directions are promising avenues to explore.

@highlight

See the abstract.  (For the revision, the paper is identical, except for a 59 page Supplementary Material, which can serve as a stand-along technical report version of the paper.)