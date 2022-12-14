Over the past four years, neural networks have been proven vulnerable to adversarial images: targeted but imperceptible image perturbations lead to drastically different predictions.

We show that adversarial vulnerability increases with the gradients of the training objective when viewed as a function of the inputs.

For most current network architectures, we prove that the L1-norm of these gradients grows as the square root of the input size.

These nets therefore become increasingly vulnerable with growing image size.

Our proofs rely on the network’s weight distribution at initialization, but extensive experiments confirm that our conclusions still hold after usual training.

Following the work of BID7 , Convolutional Neural Networks (CNNs) have been found vulnerable to adversarial examples: an adversary can drive the performance of state-of-the art CNNs down to chance level with imperceptible changes of the inputs.

A number of studies have tried to address this issue, but only few have stressed that, because adversarial examples are essentially small input changes that create large output variations, they are inherently caused by large gradients of the neural network with respect to its inputs.

Of course, this view, which we will focus on here, assumes that the network and loss are differentiable.

It has the advantage to yield a large body of specific mathematical tools, but might not be easily extendable to masked gradients, non-smooth models or the 0-1-loss.

Nevertheless, our conclusions might even hold for non-smooth models, given that the latter can often be viewed as smooth at a coarser level.

Contributions.

More specifically, we provide theoretical and empirical arguments supporting the existence of a monotonic relationship between the gradient norm of the training objective (of a differentiable classifier) and its adversarial vulnerability.

Evaluating this norm based on the weight statistics at initialization, we show that CNNs and most feed-forward networks, by design, exhibit increasingly large gradients with input dimension d, almost independently of their architecture.

That leaves them increasingly vulnerable to adversarial noise.

We corroborate our theoretical results by extensive experiments.

Although some of those experiments involve adversarial regularization schemes, our goal is not to advocate a new adversarial defense (these schemes are already known), but to show how their effect can be explained by our first order analysis.

We do not claim to explain all aspects of adversarial vulnerability, but we claim that our first order argument suffices to explain a significant part of the empirical findings on adversarial vulnerability.

This calls for researching the design of neural network architectures with inherently smaller gradients and provides useful guidelines to practitioners and network designers.

Suppose that a given classifier ϕ classifies an image x as being in category ϕ(x).

An adversarial image is a small modification of x, barely noticeable to the human eye, that suffices to fool the classifier into predicting a class different from ϕ(x).

It is a small perturbation of the inputs, that creates a large variation of outputs.

Adversarial examples thus seem inherently related to large gradients of the network.

A connection, that we will now clarify.

Note that visible adversarial examples sometimes appear in the literature, but we deliberately focus on imperceptible ones.

Adversarial vulnerability and adversarial damage.

In practice, an adversarial image is constructed by adding a perturbation δ to the original image x such that δ ≤ for some (small) number and a given norm · over the input space.

We call the perturbed input x + δ an -sized · -attack and say that the attack was successful when ϕ(x + δ) = ϕ(x).

This motivates Definition 1.

Given a distribution P over the input-space, we call adversarial vulnerability of a classifier ϕ to an -sized · -attack the probability that there exists a perturbation δ of x such that δ ≤ and ϕ(x) = ϕ(x + δ) .We call the average increase-after-attack E x∼P [∆L] of a loss L the (L-) adversarial damage (of the classifier ϕ to an -sized · -attack).When L is the 0-1-loss L 0/1 , adversarial damage is the accuracy-drop after attack.

The 0-1-loss damage is always smaller than adversarial vulnerability, because vulnerability counts all class-changes of ϕ(x), whereas some of them may be neutral to adversarial damage (e.g. a change between two wrong classes).

The L 0/1 -adversarial damage thus lower bounds adversarial vulnerability.

Both are even equal when the classifier is perfect (before attack), because then every change of label introduces an error.

It is hence tempting to evaluate adversarial vulnerability with L 0/1 -adversarial damage.

From ∆L 0/1 to ∆L and to ∂ x L. In practice however, we do not train our classifiers with the non-differentiable 0-1-loss but use a smoother loss L, such as the cross-entropy loss.

For similar reasons, we will now investigate the adversarial damage E x [∆L(x, c)] with loss L rather than L 0/1 .

Like for BID7 ; BID13 ; Sinha et al. (2018) and many others, a classifier ϕ will hence be robust if, on average over x, a small adversarial perturbation δ of x creates only a small variation δL of the loss.

Now, if δ ≤ , then a first order Taylor expansion in shows that DISPLAYFORM0 where ∂ x L denotes the gradient of L with respect to x, and where the last equality stems from the definition of the dual norm |||·||| of · .

Now two remarks.

First: the dual norm only kicks in because we let the input noise δ optimally adjust to the coordinates of ∂ x L within its -constraint.

This is the brand mark of adversarial noise: the different coordinates add up, instead of statistically canceling each other out as they would with random noise.

For example, if we impose that δ 2 ≤ , then δ will strictly align with ∂ x L.

If instead δ ∞ ≤ , then δ will align with the sign of the coordinates of ∂ x L. Second remark: while the Taylor expansion in (2) becomes exact for infinitesimal perturbations, for finite ones it may actually be dominated by higher-order terms.

Our experiments FIG1 however strongly suggest that in practice the first order term dominates the others.

Now, remembering that the dual norm of an p -norm is the corresponding q -norm, and summarizing, we have proven Lemma 2.

At first order approximation in , an -sized adversarial attack generated with norm · increases the loss L at point x by |||∂ x L|||, where |||·||| is the dual norm of · .

In particular, an -sized p -attack increases the loss by ∂ x L q where 1 ≤ p ≤ ∞ and 1 p + 1 q = 1.

Consequently, the adversarial damage of a classifier with loss L to -sized attacks generated with norm · is E x |||∂ x L|||.

This is valid only at first order, but it proves that at least this kind of first-order vulnerability is present.

We will see that the first-order predictions closely match the experiments, and that this insight helps protecting even against iterative (non-first-order) attack methods (Figure 1 ).Calibrating the threshold to the attack-norm · .

Lemma 2 shows that adversarial vulnerability depends on three main factors: (i) · , the norm chosen for the attack (ii) , the size of the attack, and (iii) E x |||∂ x L||| , the expected dual norm of ∂ x L. We could see Point (i) as a measure of our sensibility to image perturbations, (ii) as our sensibility threshold, and (iii) as the classifier's expected marginal sensibility to a unit perturbation.

E x |||∂ x L||| hence intuitively captures the discrepancy between our perception (as modeled by · ) and the classifier's perception for an input-perturbation of small size .

Of course, this viewpoint supposes that we actually found a norm · (or more generally a metric) that faithfully reflects human perception -a project in its own right, far beyond the scope of this paper.

However, it is clear that the threshold that we choose should depend on the norm · and hence on the input-dimension d. In particular, for a given pixel-wise order of magnitude of the perturbations δ, the p -norm of the perturbation will scale like d

.

This suggests to write the threshold p used with p -attacks as: DISPLAYFORM0 where ∞ denotes a dimension-independent constant.

In Appendix D we show that this scaling also preserves the average signal-to-noise ratio x 2 / δ 2 , both across norms and dimensions, so that p could correspond to a constant human perception-threshold.

With this in mind, the impatient reader may already jump to Section 3, which contains our main contributions: the estimation of E x ∂ x L q for standard feed-forward nets.

Meanwhile, the rest of this section shortly discusses two straightforward defenses that we will use later and that further illustrate the role of gradients.

A new old regularizer.

Lemma 2 shows that the loss of the network after an 2 -sized · -attack is DISPLAYFORM1 It is thus natural to take this loss-after-attack as a new training objective.

Here we introduced a factor 2 for reasons that will become clear in a moment.

Incidentally, for · = · 2 , this new loss reduces to an old regularization-scheme proposed by BID4 called double-backpropagation.

At the time, the authors argued that slightly decreasing a function's or a classifier's sensitivity to input perturbations should improve generalization.

In a sense, this is exactly our motivation when defending against adversarial examples.

It is thus not surprising to end up with the same regularization term.

Note that our reasoning only shows that training with one specific norm |||·||| in (4) helps to protect against adversarial examples generated from · .

A priori, we do not know what will happen for attacks generated with other norms; but our experiments suggest that training with one norm also protects against other attacks (see FIG1 and Section 4.1).Link to adversarially-augmented training.

In (1), designates an attack-size threshold, while in (4), it is a regularization-strength.

Rather than a notation conflict, this reflects an intrinsic duality between two complementary interpretations of , which we now investigate further.

Suppose that, instead of using the loss-after-attack, we augment our training set with -sized · -attacks x + δ, where for each training point x, the perturbation δ is generated on the fly to locally maximize the loss-increase.

Then we are effectively training with DISPLAYFORM2 where by construction δ satisfies (2).

We will refer to this technique as adversarially augmented training.

It was first introduced by BID7 with · = · ∞ under the name of FGSM 1 -augmented training.

Using the first order Taylor expansion in of (2), this 'old-plus-postattack' loss of (5) simply reduces to our loss-after-attack, which proves Proposition 3.

Up to first-order approximations in ,L , · = L ,|||·||| .

Said differently, for small enough , adversarially-augmented training with -sized · -attacks amounts to penalizing the dual norm |||·||| of ∂ x L with weight /2.

In particular, double-backpropagation corresponds to training with 2 -attacks, while FGSM-augmented training corresponds to an 1 -penalty on ∂ x L.This correspondence between training with perturbations and using a regularizer can be compared to Tikhonov regularization: Tikhonov regularization amounts to training with random noise BID2 , while training with adversarial noise amounts to penalizing ∂ x L. Section 4.1 verifies the correspondence between adversarial augmentation and gradient regularization empirically, which also strongly suggests the empirical validity of the first-order Taylor expansion in (2).3 ESTIMATING ∂ x L q TO EVALUATE ADVERSARIAL VULNERABILITY In this section, we evaluate the size of ∂ x L q for standard neural network architectures.

We start with fully-connected networks, and finish with a much more general theorem that, not only encompasses CNNs (with or without strided convolutions), but also shows that the gradient-norms are essentially independent of the network topology.

We start our analysis by showing how changing q affects the size of ∂ x L q .

Suppose for a moment that the coordinates of ∂ x L have typical magnitude DISPLAYFORM3 This equation carries two important messages.

First, we see how ∂ x L q depends on d and q. The dependence seems highest for q = 1.

But once we account for the varying perceptibility threshold DISPLAYFORM4 , we see that adversarial vulnerability scales like d · |∂ x L|, whatever p -norm we use.

Second, (6) shows that to be robust against any type of p -attack at any input-dimension d, the average absolute value of the coefficients of ∂ x L must grow slower than 1/d.

Now, here is the catch, which brings us to our core insight.

In order to preserve the activation variance of the neurons from layer to layer, the neural weights are usually initialized with a variance that is inversely proportional to the number of inputs per neuron.

Imagine for a moment that the network consisted only of one output neuron o linearly connected to all input pixels.

For the purpose of this example, we assimilate o and L. Because we initialize the weights with a variance of 1/d, their average absolute value |∂ x o| ≡ |∂ x L| grows like 1/ √ d, rather than the required 1/d.

By (6), the adversarial vulnerability DISPLAYFORM0 This toy example shows that the standard initialization scheme, which preserves the variance from layer to layer, causes the average coordinate-size |∂ x L| to grow like 1/ √ d instead of 1/d.

When an ∞ -attack tweaks its -sized input-perturbations to align with the coordinate-signs of ∂ x L, all coordinates of ∂ x L add up in absolute value, resulting in an output-perturbation that scales like √ d and leaves the network increasingly vulnerable with growing input-dimension.

Our next theorems generalize the previous toy example to a very wide class of feedforward nets with ReLU activation functions.

For illustration purposes, we start with fully connected nets and only then proceed to the broader class, which includes any succession of (possibly strided) convolutional layers.

In essence, the proofs iterate our insight on one layer over a sequence of layers.

They all rely on the following set (H) of hypotheses: H1 Non-input neurons are followed by a ReLU killing half of its inputs, independently of the weights.

H2 Neurons are partitioned into layers, meaning groups that each path traverses at most once.

H3 All weights have 0 expectation and variance 2/(in-degree) ('He-initialization').

H4 The weights from different layers are independent.

H5 Two distinct weights w, w from a same node satisfy E [w w ] = 0.If we follow common practice and initialize our nets as proposed by BID8 , then H3-H5 are satisfied at initialization by design, while H1 is usually a very good approximation BID1 .

Note that such i.i.d.

weight assumptions have been widely used to analyze neural nets and are at the heart of very influential and successful prior work (e.g., equivalence between neural nets and Gaussian processes as pioneered by Neal 1996).

Nevertheless, they do not hold after training.

That is why all our statements in this section are to be understood as orders of magnitudes that are very well satisfied at initialization in theory and in practice, and that we will confirm experimentally after training in Section 4.

Said differently, while our theorems rely on the statistics of neural nets at initialization, our experiments confirm their conclusions after training.

Theorem 4 (Vulnerability of Fully Connected Nets).

Consider a succession of fully connected layers with ReLU activations which takes inputs x of dimension d, satisfies assumptions (H), and outputs logits f k (x) that get fed to a final cross-entropy-loss layer L.

Then the coordinates of DISPLAYFORM0 These networks are thus increasingly vulnerable to p -attacks with growing input-dimension.

Theorem 4 is a special case of the next theorem, which will show that the previous conclusions are essentially independent of the network-topology.

We will use the following symmetry assumption on the neural connections.

For a given path p, let the path-degree d p be the multiset of encountered in-degrees along path p.

For a fully connected network, this is the unordered sequence of layer-sizes preceding the last path-node, including the input-layer.

Now consider the multiset {d p } p∈P(x,o) of all path-degrees when p varies among all paths from input x to output o.

The symmetry assumption (relatively to o) is (S) All input nodes x have the same multiset {d p } p∈P(x,o) of path-degrees from x to o.

Intuitively, this means that the statistics of degrees encountered along paths to the output are the same for all input nodes.

This symmetry assumption is exactly satisfied by fully connected nets, almost satisfied by CNNs (up to boundary effects, which can be alleviated via periodic or mirror padding) and exactly satisfied by strided layers, if the layer-size is a multiple of the stride.

Theorem 5 (Vulnerability of Feedforward Nets).

Consider any feed-forward network with linear connections and ReLU activation functions.

Assume the net satisfies assumptions (H) and outputs logits f k (x) that get fed to the cross-entropy-loss L. DISPLAYFORM1 Moreover, if the net satisfies the symmetry assumption (S), then DISPLAYFORM2 Theorems 4 and 5 are proven in Appendix B. The main proof idea is that in the gradient norm computation, the He-initialization exactly compensates the combinatorics of the number of paths in the network, so that this norm becomes independent of the network topology.

In particular, we getCorollary 6 (Vulnerability of CNNs).

In any succession of convolution and dense layers, strided or not, with ReLU activations, that satisfies assumptions (H) and outputs logits that get fed to the cross-entropy-loss L, the gradient of the logit-coordinates scale like 1/ √ d and FORMULA8 is satisfied.

It is hence increasingly vulnerable with growing input-resolution to attacks generated with any p -norm.

Appendix A shows that the network gradient are dampened when replacing strided layers by average poolings, essentially because average-pooling weights do not follow the He-init assumption H3.

In Section 4.1, we empirically verify the validity of the first-order Taylor approximation made in (2) (Fig.1) , for example by checking the correspondence between loss-gradient regularization and adversarially-augmented training FIG1 ).

Section 4.2 then empirically verifies that both the average 1 -norm of ∂ x L and the adversarial vulnerability grow like √ d as predicted by Corollary 6.

For all experiments, we approximate adversarial vulnerability using various attacks of the Foolboxpackage (Rauber et al., 2017) .

We use an ∞ attack-threshold of size ∞ = 0.005 (and later 0.002) which, for pixel-values ranging from 0 to 1, is completely imperceptible but suffices to fool the classifiers on a significant proportion of examples.

This ∞ -threshold should not be confused with the regularization-strengths appearing in (4) and (5), which will be varied in some experiments.

Figure 1: Adversarial vulnerability approximated by different attack-types for 10 trained networks as a function of (a) the 1 gradient regularization-strength used to train the nets and (b) the average gradient-norm.

These curves confirm that the first-order expansion term in (2) is a crucial component of adversarial vulnerability.

DISPLAYFORM0 suggests that protecting against a given attack-norm also protects against others. (f ): Merging 2band 2c shows that all adversarial augmentation and gradient-regularization methods achieve similar accuracy-vulnerability trade-offs.

We train several CNNs with same architecture to classify CIFAR-10 images BID12 .For each net, we use a specific training method with a specific regularization value .

The training methods used were 1 -and 2 -penalization of ∂ x L (Eq. 4), adversarial augmentation with ∞ -andValidity of first order expansion.

The following observations support the validity of the first order Taylor expansion in (2) and suggest that it is a crucial component of adversarial vulnerability: (i) the efficiency of the first-order defense against iterative (non-first-order) attacks (Fig.1a) ; (ii) the striking similarity between the PGD curves (adversarial augmentation with iterative attacks) and the other adversarial training training curves (one-step attacks/defenses); (iii) the functional-like dependence between any approximation of adversarial vulnerability and E x ∂ x L 1 (Fig.1b) , and its independence on the training method FIG1 . (iv) the excellent correspondence between the gradient-regularization and adversarial training curves (see next paragraph).

Said differently, adversarial examples seem indeed to be primarily caused by large gradients of the classifier as captured via the induced loss. .

The excellent match between the adversarial augmentation curve with p = ∞ (p = 2) and its gradient-regularization dual counterpart with q = 1 (resp.

q = 2) illustrates the duality between as a threshold for adversarially-augmented training and as a regularization constant in the regularized loss (Proposition 3).

It also supports the validity of the first-order Taylor expansion in (2).Confirmation of (3).

Still on the upper row, the curves for p = ∞, q = 1 have no reason to match those for p = q = 2 when plotted against , because -threshold is relative to a specific attack-norm.

However, (3) suggested that the rescaled thresholds d 1/p may approximately correspond to a same 'threshold-unit' across p -norms and across dimension.

This is well confirmed by the upper row plots: by rescaling the x-axis, the p = q = 2 and q = 1, p = ∞ curves get almost super-imposed.

Accuracy-vs-Vulnerability Trade-Off.

FIG1 by taking out , FIG1 shows that all gradient regularization and adversarial training methods yield equivalent accuracyvulnerability trade-offs.

Incidentally, for higher penalization values, these trade-offs appear to be much better than those given by cross Lipschitz regularization.

The penalty-norm does not matter.

We were surprised to see that on Figures 2d and 2f, the L ,q curves are almost identical for q = 1 and 2.

This indicates that both norms can be used interchangeably in (4) (modulo proper rescaling of via FORMULA2 ), and suggests that protecting against a specific attacknorm also protects against others.

(6) may provide an explanation: if the coordinates of ∂ x L behave like centered, uncorrelated variables with equal variance -which follows from assumptions (H) -, then the 1 -and 2 -norms of FIG1 confirms this explanation.

The slope is independent of the training method.

Therefore, penalizing ∂ x L(x) 1 during training will not only decrease E x ∂ x L 1 (as shown in FIG1 ), but also drive down E x ∂ x L 2 and vice-versa.

DISPLAYFORM1

Theorems 4-5 and Corollary 6 predict a linear growth of the average 1 -norm of ∂ x L with the square root of the input dimension d, and therefore also of adversarial vulnerability (Lemma 2).

To test these predictions, we upsampled the CIFAR-10 images (of size 3 x 32 x 32) by copying pixels so as to get 4 datasets with, respectively, 32, 64, 128 and 256 pixels per edge.

We then trained a CNN on each dataset All networks had exactly the same amount of parameters and very similar structure across the various input-resolutions.

The CNNs were a succession of 8 'convolution → batchnorm → ReLU' layers with 64 output channels, followed by a final full-connection to the 12 logit-outputs.

We used 2 × 2-max-poolings after the convolutions of layers 2,4, 6 and 8, and a final max-pooling after layer 8 that fed only 1 neuron per channel to the fully-connected layer.

To ensure that the convolution-kernels cover similar ranges of the images across each of the 32, 64, 128 and 256 input-resolutions, we respectively dilated all convolutions ('à trous') by a factor 1, 2, 4 and 8.

There is a clear discrepancy: on the training set, the gradient norms decrease (after an initialization phase) and are dimension-independent; on the test set, they increase and scale like √ d. This suggests that, outside the training points, the nets tend to recover their prior gradient-properties (i.e. naturally large gradients).

Our theoretical results show that the priors of classical neural networks yield vulnerable functions because of naturally high gradients.

And our experiments FIG3 suggest that usual training does not escape these prior properties.

But how may these insights help understanding the vulnerability of robustly trained networks?

Clearly, to be successful, robust training algorithms must escape ill-behaved priors, which explains why most methods (e.g. FGSM, PGD) are essentially gradient penalization techniques.

But, MNIST aside, even state-of-the-art methods largely fail at protecting current network architectures BID14 , and understanding why is motivation to this and many other papers.

Interestingly, BID14 recently noticed that those methods actually do protect the nets on training examples, but fail to generalize to the test set.

They hence conclude that state-of-the-art robustification algorithms work, but need more data.

Alternatively however, when generalization fails, one can also reduce the model's complexity.

Large fully connected nets for example typically fail to generalize to out-of-sample examples: getting similar accuracies than CNNs would need prohibitively many training points.

Similarly, Schmidt et al.'s observations may suggest that, outside the training points, networks tend to recover their prior properties, i.e. naturally large gradients.

FIG4 corroborates this hypothesis.

It plots the evolution over training epochs of the 1 -gradient-norms of the CNNs from Section 4.2 FIG3 on the training and test sets respectively.

The discrepancy is unmistakable: after a brief initialization phase, the norms decrease on the training set, but increase on the test set.

They are moreover almost input-dimension independent on the training set, but scale as √ d on the test set (as seen in FIG3 up to respectively 2, 4, 8 and 16 times the training set values.

These observations suggest that, with the current amount of data, tackling adversarial vulnerability may require new architectures with inherently smaller gradients.

Searching these architectures among those with well-behaved prior-gradients seems a reasonable start, where our theoretical results may prove very useful.

On network vulnerability.

BID7 already stressed that adversarial vulnerability increases with growing dimension d. But their argument only relied on a linear 'one-output-to-manyinputs'-model with dimension-independent weights.

They therefore concluded on a linear growth of adversarial vulnerability with d. In contrast, our theory applies to almost any standard feed-forward architecture (not just linear), and shows that, once we adjust for the weight's dimension-dependence, adversarial vulnerability increases like √ d (not d), almost independently of the architecture.

Nevertheless, our experiments confirm Goodfellow et al.'s idea that our networks are "too linear-like", in the sense that a first-order Taylor expansion is indeed sufficient to explain the adversarial vulnerability of neural networks.

As suggested by the one-output-to-many-inputs model, the culprit is that growing 3 Appendix A investigates such a preliminary direction by introducing average poolings, which have a weight-size 1 /in−channels rather than the typical 1 / √ in−channels of the other He-initialized weights.

dimensionality gives the adversary more and more room to 'wriggle around' with the noise and adjust to the gradient of the output neuron.

This wriggling, we show, is still possible when the output is connected to all inputs only indirectly, even when no neuron is directly connected to all inputs, like in CNNs.

This explanation of adversarial vulnerability is independent of the intrinsic dimensionality or geometry of the data (compare to BID0 BID6 .

Finally, let us mention that show a close link between the vulnerability to small worst-case perturbation (as studied here) and larger average perturbations.

Our findings on the adversarial vulnerability NNs to small perturbation could thus be translated accordingly.

On robustification algorithms.

Incidentally, BID7 also already relate adversarial vulnerability to large gradients of the loss L, an insight at the very heart of their FGSM-algorithm.

They however do not propose any explicit penalizer on the gradient of L other than indirectly through adversarially-augmented training.

Conversely, Ross & Doshi-Velez (2018) propose the old double-backpropagation to robustify networks but make no connection to FGSM and adversarial augmentation.

BID13 discuss and use the connection between gradient-penalties and adversarial augmentation, but never actually compare both in experiments.

This comparison however is essential to test the validity of the first-order Taylor expansion in (2), as confirmed by the similarity between the gradient-regularization and adversarial-augmentation curves in FIG1 .

BID9 derived yet another gradient-based penalty -the cross-Lipschitz-penalty-by considering (and proving) formal guarantees on adversarial vulnerability itself, rather than adversarial damage.

While both penalties are similar in spirit, focusing on the adversarial damage rather than vulnerability has two main advantages.

First, it achieves better accuracy-to-vulnerability ratios, both in theory and practice, because it ignores class-switches between misclassified examples and penalizes only those that reduce the accuracy.

Second, it allows to deal with one number only, ∆L, whereas Hein & Andriushchenko's cross-Lipschitz regularizer and theoretical guarantees explicitly involve all K logit-functions (and their gradients).

See Appendix C. Penalizing network-gradients is also at the heart of contractive auto-encoders as proposed by Rifai et al. (2011) , where it is used to regularize the encoder-features.

Seeing adversarial training as a generalization method, let us also mention BID10 , who propose to enhance generalization by searching for parameters in a "flat minimum region" of the loss.

This leads to a penalty involving the gradient of the loss, but taken with respect to the weights, rather than the inputs.

In the same vein, a gradientregularization of the loss of generative models also appears in Proposition 6 of Ollivier (2014), where it stems from a code-length bound on the data (minimum description length).

More generally, the gradient regularized objective (4) is essentially the first-order approximation of the robust training objective max δ ≤ L(x + δ, c) which has a long history in math (Wald, 1945) , machine learning (Xu et al., 2009 ) and now adversarial vulnerability (Sinha et al., 2018) .

Finally, BID3 propose new network-architectures that have small gradients by design, rather than by special training: an approach that makes all the more sense, considering the conclusion of Theorems 4 and 5.

For further details and references on adversarial attacks and defenses, we refer to Yuan et al. (2017) .

For differentiable classifiers and losses, we showed that adversarial vulnerability increases with the gradients ∂ x L of the loss, which is confirmed by the near-perfect functional relationship between gradient norms and vulnerability FIG1 We then evaluated the size of ∂ x L q and showed that, at initialization, usual feed-forward nets (convolutional or fully connected) are increasingly vulnerable to p -attacks with growing input dimension d (the image-size), almost independently of their architecture.

Our experiments show that, on the tested architectures, usual training escapes those prior gradient (and vulnerability) properties on the training, but not on the test set.

BID14 suggest that alleviating this generalization gap requires more data.

But a natural (complementary) alternative would be to search for architectures with naturally smaller gradients, and in particular, with well-behaved priors.

Despite all their limitations (being only first-order, assuming a prior weight-distribution and a differentiable loss and architecture), our theoretical insights may thereby still prove to be precious future allies.

It is common practice in CNNs to use average-pooling layers or strided convolutions to progressively decrease the number of pixels per channel.

Corollary 6 shows that using strided convolutions does not protect against adversarial examples.

However, what if we replace strided convolutions by convolutions with stride 1 plus an average-pooling layer?

Theorem 5 considers only randomly initialized weights with typical size 1/ √ in-degree.

Average-poolings however introduce deterministic weights of size 1/(in-degree).

These are smaller and may therefore dampen the input-to-output gradients and protect against adversarial examples.

We confirm this in our next theorem, which uses a slightly modified version (H ) of (H) to allow average pooling layers.

(H ) is (H), but where the He-init H3 applies to all weights except the (deterministic) average pooling weights, and where H1 places a ReLU on every non-input and non-average-pooling neuron.

Theorem 7 (Effect of Average-Poolings).

Consider a succession of convolution layers, dense layers and n average-pooling layers, in any order, that satisfies (H ) and outputs logits f k (x).

Assume the n average pooling layers have a stride equal to their mask size and perform averages over a 1 , ..., a n nodes respectively.

Then ∂ x f k 2 and |∂ x f k | scale like 1/ √ a 1 · · · a n and 1/ √ d a 1 · · · a n respectively.

Proof in Appendix B.4.

Theorem 7 suggest to try and replace any strided convolution by its non-strided counterpart, followed by an average-pooling layer.

It also shows that if we systematically reduce the number of pixels per channel down to 1 by using only non-strided convolutions and average-pooling layers (i.e. d = n i=1 a i ), then all input-to-output gradients should become independent of d, thereby making the network completely robust to adversarial examples.

Our following experiments ( FIG6 show that after training, the networks get indeed robustified to adversarial examples, but remain more vulnerable than suggested by Theorem 7.Experimental setup.

Theorem 7 shows that, contrary to strided layers, average-poolings should decrease adversarial vulnerability.

We tested this hypothesis on CNNs trained on CIFAR-10, with 6 blocks of 'convolution → BatchNorm →ReLU' with 64 output-channels, followed by a final average pooling feeding one neuron per channel to the last fully-connected linear layer.

Additionally, after every second convolution, we placed a pooling layer with stride and mask-size (2, 2) (thus acting on 2 × 2 neurons at a time, without overlap).

We tested average-pooling, strided and max-pooling layers and trained 20 networks per architecture.

Results are shown in FIG6 .

All accuracies are very close, but, as predicted, the networks with average pooling layers are more robust to adversarial images than the others.

However, they remain more vulnerable than what would follow from Theorem 7.

We also noticed that, contrary to the strided architectures, their gradients after training are an order of magnitude higher than at initialization and than predicted.

This suggests that assumptions (H) get more violated when using average-poolings instead of strided layers.

Understanding why will need further investigations.

Proof.

Let δ be an adversarial perturbation with δ = 1 that locally maximizes the loss increase at point x, meaning that δ = arg max δ ≤1 ∂ x L · δ .

Then, by definition of the dual norm of ∂ x L we have: DISPLAYFORM0

Proof.

Let x designate a generic coordinate of x. To evaluate the size of ∂ x L q , we will evaluate the size of the coordinates ∂ x L of ∂ x L by decomposing them into DISPLAYFORM0 where f k (x) denotes the logit-probability of x belonging to class k. We now investigate the statistical properties of the logit gradients ∂ x f k , and then see how they shape ∂ x L.Step 1: Statistical properties of ∂ x f k .

Let P(x, k) be the set of paths p from input neuron x to output-logit k. Let p − 1 and p be two successive neurons on path p, andp be the same path p but without its input neuron.

Let w p designate the weight from p − 1 to p and ω p be the path-product ω p := p∈p w p .

Finally, let σ p (resp.

σ p ) be equal to 1 if the ReLU of node p (resp.

if path p) is active for input x, and 0 otherwise.

As previously noticed by BID1 using the chain rule, we see that ∂ x f k is the sum of all ω p whose path is active, i.e. ∂ x f k (x) = p∈P(x,k) ω p σ p .

Consequently: DISPLAYFORM1 The first equality uses H1 to decouple the expectations over weights and ReLUs, and then applies Lemma 10 of Appendix B.3, which uses H3-H5 to kill all cross-terms and take the expectation over weights inside the product.

The second equality uses H3 and the fact that the resulting product is the same for all active paths.

The third equality counts the number of paths from x to k and we conclude by noting that all terms cancel out, except d p−1 from the input layer which is d. Equation 8 shows DISPLAYFORM2 Step 2: Statistical properties of DISPLAYFORM3 f h (x) (the probability of image x belonging to class k according to the network), we have, by definition of the cross-entropy loss, L(x, c) := − log q c (x), where c is the label of the target class.

Thus: DISPLAYFORM4 otherwise, and DISPLAYFORM5 Using again Lemma 10, we see that the ∂ x f k (x) are K centered and uncorrelated variables.

So DISPLAYFORM6 is approximately the sum of K uncorrelated variables with zero-mean, and its total variance is given by DISPLAYFORM7 .

(6) concludes.

Remark 1.

Equation 9 can be rewritten as DISPLAYFORM8 As the term k = c disappears, the norm of the gradients ∂ x L(x) appears to be controlled by the total error probability.

This suggests that, even without regularization, trying to decrease the ordinary classification error is still a valid strategy against adversarial examples.

It reflects the fact that when increasing the classification margin, larger gradients of the classifier's logits are needed to push images from one side of the classification boundary to the other.

This is confirmed by Theorem 2.1 of BID9 .

See also (16) in Appendix C.

The proof of Theorem 5 is very similar to the one of Theorem 4, but we will need to first generalize the equalities appearing in (8).

To do so, we identify the computational graph of a neural network to an abstract Directed Acyclic Graph (DAG) which we use to prove the needed algebraic equalities.

We then concentrate on the statistical weight-interactions implied by assumption (H), and finally throw these results together to prove the theorem.

In all the proof, o will designate one of the output-logits f k (x).Lemma 8.

Let x be the vector of inputs to a given DAG, o be any leaf-node of the DAG, x a generic coordinate of x. Let p be a path from the set of paths P(x, o) from x to o,p the same path without node x, p a generic node inp, and d p be its input-degree.

Then: DISPLAYFORM0 Proof.

We will reason on a random walk starting at o and going up the DAG by choosing any incoming node with equal probability.

The DAG being finite, this walk will end up at an input-node x with probability 1.

Each path p is taken with probability p∈p 1 dp .

And the probability to end up at an input-node is the sum of all these probabilities, i.e. DISPLAYFORM1 The sum over all inputs x in (11) being 1, on average it is 1/d for each x, where d is the total number of inputs (i.e. the length of x).

It becomes an equality under assumption (S):Lemma 9.

Under the symmetry assumption (S), and with the previous notations, for any input x ∈ x: p .

By using (11) and the fact that, by (S), the multiset D(x, o) is independent of x, we hence conclude DISPLAYFORM2 DISPLAYFORM3 Now, let us relate these considerations on graphs to gradients and use assumptions (H).

We remind that path-product ω p is the product p∈p w p .Lemma 10.

Under assumptions (H), the path-products ω p , ω p of two distinct paths p and p starting from a same input node x, satisfy: DISPLAYFORM4 Furthermore, if there is at least one non-average-pooling weight on path p, then E W [ω p ] = 0.Proof.

Hypothesis H4 yields DISPLAYFORM5 Now, take two different paths p and p that start at a same node x. Starting from x, consider the first node after which p and p part and call p and p the next nodes on p and p respectively.

Then the weights w p and w p are two weights of a same node.

Applying H4 and H5 hence gives DISPLAYFORM6 Finally, if p has at least one non-average-pooling node p, then successively applying H4 and H3 yields: DISPLAYFORM7 We now have all elements to prove Theorem 5.Proof. (of Theorem 5) For a given neuron p inp, let p − 1 designate the previous node in p of p. Let σ p (resp.

σ p ) be a variable equal to 0 if neuron p gets killed by its ReLU (resp.

path p is inactive), and 1 otherwise.

Then: DISPLAYFORM8 Consequently: DISPLAYFORM9 where the firs line uses the independence between the ReLU killings and the weights (H1), the second uses Lemma 10 and the last uses Lemma 9.

The gradient ∂ x o thus has coordinates whose squared expectations scale like 1/d.

Thus each coordinate scales like 1/ √ d and ∂ x o q like d DISPLAYFORM10 Step 2 of the proof of Theorem 4.Finally, note that, even without the symmetry assumption (S), using Lemma 8 shows that DISPLAYFORM11 Thus, with or without (S), ∂ x o 2 is independent of the input-dimension d.

To prove Theorem 7, we will actually prove the following more general theorem, which generalizes Theorem 5.

Theorem 7 is a straightforward corollary of it.

Theorem 11.

Consider any feed-forward network with linear connections and ReLU activation functions that outputs logits f k (x) and satisfies assumptions (H).

Suppose that there is a fixed multiset of integers {a 1 , . . .

, a n } such that each path from input to output traverses exactly n average pooling nodes with degrees {a 1 , . . .

, a n }.

Then: DISPLAYFORM0 Furthermore, if the net satisfies the symmetry assumption (S), then: DISPLAYFORM1 Two remarks.

First, in all this proof, "weight" encompasses both the standard random weights, and the constant (deterministic) weights equal to 1/(in-degree) of the average-poolings.

Second, assumption H5 implies that the average-pooling nodes have disjoint input nodes: otherwise, there would be two non-zero deterministic weights w, w from a same neuron that would hence satisfy: DISPLAYFORM2 Proof.

As previously, let o designate any fixed output-logit f k (x).

For any path p, let a be the set of average-pooling nodes of p and let q be the set of remaining nodes.

Each path-product ω p satisfies: ω p = ω q ω a , where ω a is a same fixed constant.

For two distinct paths p, p , Lemma 10 therefore yields: DISPLAYFORM3 Combining this with Lemma 9 and under assumption (S), we get similarly to (13): DISPLAYFORM4 Again, note that, even without assumption (S), using (15) and Lemma 8 shows that DISPLAYFORM5 which proves (14).

In their Theorem 2.1, BID9 show that the minimal = δ p perturbation to fool the classifier must be bigger than: DISPLAYFORM0 They argue that the training procedure typically already tries to maximize f c (x) − f k (x), thus one only needs to additionally ensure that ∂ x f c (x) − ∂ x f k (x) q is small.

They then introduce what they call a Cross-Lipschitz Regularization, which corresponds to the case p = 2 and involves the gradient differences between all classes: DISPLAYFORM1 In contrast, using (10), (the square of) our proposed regularizer ∂ x L q from (4) can be rewritten, for p = q = 2 as: DISPLAYFORM2 Although both FORMULA0 and FORMULA0 cross-interaction between the K classes, the big difference is that while in (17) all classes play exactly the same role, in (18) the summands all refer to the target class c in at least two different ways.

First, all gradient differences are always taken with respect to ∂ x f c .

Second, each summand is weighted by the probabilities q k (x) and q h (x) of the two involved classes, meaning that only the classes with a non-negligible probability get their gradient regularized.

This reflects the idea that only points near the margin need a gradient regularization, which incidentally will make the margin sharper.

To keep the average pixel-wise variation constant across dimensions d, we saw in (3) that the threshold p of an p -attack should scale like d 1/p .

We will now see another justification for this scaling.

Contrary to the rest of this work, where we use a fixed p for all images x, here we will let p depend on the 2 -norm of x. If, as usual, the dataset is normalized such that the pixels have on average variance 1, both approaches are almost equivalent.

Suppose that given an p -attack norm, we want to choose p such that the signal-to-noise ratio (SNR) x 2 / δ 2 of a perturbation δ with p -norm ≤ p is never greater than a given SNR threshold 1/ .

For p = 2 this imposes 2 = x 2 .

More generally, studying the inclusion of p -balls in 2 -balls yields DISPLAYFORM0 Note that this gives again DISPLAYFORM1 .

This explains how to adjust the threshold with varying p -attack norm.

Now, let us see how to adjust the threshold of a given p -norm when the dimension d varies.

Suppose that x is a natural image and that decreasing its dimension means either decreasing its resolution or cropping it.

Because the statistics of natural images are approximately resolution and scale invariant BID11 , in either case the average squared value of the image pixels remains unchanged, which implies that x 2 scales like √ d. Pasting this back into (19), we again get: DISPLAYFORM2 In particular, ∞ ∝ is a dimension-free number, exactly like in (3) of the main part.

Now, why did we choose the SNR as our invariant reference quantity and not anything else?

One reason is that it corresponds to a physical power ratio between the image and the perturbation, which we think the human eye is sensible to.

Of course, the eye's sensitivity also depends on the spectral frequency of the signals involved, but we are only interested in orders of magnitude here.

Another point: any image x yields an adversarial perturbation δ x , where by constraint x 2 / δ x ≤ 1/ .

For 2 -attacks, this inequality is actually an equality.

But what about other p -attacks: (on average over x,) how far is the signal-to-noise ratio from its imposed upper bound 1/ ?

For p ∈ {1, 2, ∞}, the answer unfortunately depends on the pixel-statistics of the images.

But when p is 1 or ∞, then the situation is locally the same as for p = 2.

Specifically:Lemma 12.

Let x be a given input and > 0.

Let p be the greatest threshold such that for any δ with δ p ≤ p , the SNR DISPLAYFORM3 Moreover, for p ∈ {1, 2, ∞}, if δ x is the p -sized p -attack that locally maximizes the loss-increase i.e. δ x = arg max δ p ≤ p |∂ x L · δ|, then: DISPLAYFORM4 and E x [SNR(x)] = 1 .Proof.

The first paragraph follows from the fact that the greatest p -ball included in an 2 -ball of radius x 2 has radius x 2 d DISPLAYFORM5 Under review as a conference paper at ICLR 2019The second paragraph is clear for p = 2.

For p = ∞, it follows from the fact that δ x = ∞ sign ∂ x L which satisfies: DISPLAYFORM6 Intuitively, this means that for p ∈ {1, 2, ∞}, the SNR of p -sized p -attacks on any input x will be exactly equal to its fixed upper limit 1/ .

And in particular, the mean SNR over samples x is the same (1/ ) in all three cases.

We also ran a similar experiment as in Section 4.2, but instead of using upsampled CIFAR-10 images, we created a 12-class dataset of approximately 80, 000 3 × 256 × 256-sized RGBimages by merging similar ImageNet-classes, resizing the smallest image-edge to 256 pixels and quantiles.

The conclusions are identical to Section 4.2: after usual training, the vulnerability and gradient-norms still increase like √ d. Note that, as the gradients get much larger at higher dimensions, the first order approximation in (2) becomes less and less valid, which explains the little inflection of the adversarial vulnerability curve.

For smaller -thresholds, we verified that the inflection disappears.

Here we plot the same curves as in the main part, but using an 2 -attack threshold of size 2 = 0.005 √ d instead of the ∞ -threshold and deep-fool attacks (Moosavi-Dezfooli et al., 2016) instead of iterative ∞ -ones in Figs. 8 and 9.

Note that contrary to ∞ -thresholds, 2 -thresholds must be rescaled by √ d to stay consistent across dimensions (see Eq.3 and Appendix D).

All curves look essentially the same as their counterparts in the main text.

In usual adversarially-augmented training, the adversarial image x + δ is generated on the fly, but is nevertheless treated as a fixed input of the neural net, which means that the gradient does not get backpropagated through δ.

This need not be.

As δ is itself a function of x, the gradients could actually also be backpropagated through δ.

As it was only a one-line change of our code, we used this opportunity to test this variant of adversarial training (FGSM-variant in FIG1 ) and thank Martín Arjovsky for suggesting it.

But except for an increased computation time, we found no significant difference compared to usual augmented training.

<|TLDR|>

@highlight

Neural nets have large gradients by design; that makes them adversarially vulnerable.