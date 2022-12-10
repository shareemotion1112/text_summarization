MixUp is a data augmentation scheme in which pairs of training samples and their corresponding labels are mixed using linear coefficients.

Without label mixing, MixUp becomes a more conventional scheme: input samples are moved but their original labels are retained.

Because samples are preferentially moved in the direction of other classes \iffalse -- which are typically clustered in input space -- \fi we refer to this method as directional adversarial training, or DAT.

We show that under two mild conditions, MixUp asymptotically convergences to a subset of DAT.

We define untied MixUp (UMixUp), a superset of MixUp wherein training labels are mixed with different linear coefficients to those of their corresponding samples.

We show that under the same mild conditions, untied MixUp converges to the entire class of DAT schemes.

Motivated by the understanding that UMixUp is both a generalization of MixUp and a form of adversarial training, we experiment with different datasets and loss functions to show that UMixUp provides improved performance over MixUp.

In short, we present a novel interpretation of MixUp as belonging to a class highly analogous to adversarial training, and on this basis we introduce a simple generalization which outperforms MixUp.

Deep learning applications often require complex networks with a large number of parameters (He et al., 2016; Zagoruyko & Komodakis, 2016; Devlin et al., 2018) .

Although neural networks perform so well that their ability to generalize is an area of study in itself (Zhang et al., 2017a; Arpit et al., 2017) , their high complexity nevertheless causes them to overfit their training data (Kukacka et al., 2017) .

For this reason, effective regularization techniques are in high demand.

There are two flavors of regularization: complexity curtailing and data augmentation 1 .

Complexity curtailing methods constrain models to learning in a subset of parameter space which has a higher probability of generalizing well.

Notable examples are weight decay (Krogh & Hertz, 1991) and dropout (Srivastava et al., 2014) .

Data augmentation methods add transformed versions of training samples to the original training set.

Conventionally, transformed samples retain their original label, so that models effectively see a larger set of data-label training pairs.

Commonly applied transformations in image applications include flips, crops and rotations.

A recently devised family of augmentation schemes called adversarial training has attracted active research interest (Szegedy et al., 2013; Goodfellow et al., 2014; Miyato et al., 2016; Athalye et al., 2018; Shaham et al., 2018; He et al., 2018) .

Adversarial training seeks to reduce a model's propensity to misclassify minimally perturbed training samples, or adversarials.

While attack algorithms used for testing model robustness may search for adversarials in unbounded regions of input space, adversarial training schemes generally focus on perturbing training samples within a bounded region, while retaining the sample's original label (Goodfellow et al., 2015; Shaham et al., 2018) .

Another recently proposed data augmentation scheme is MixUp (Zhang et al., 2017b) , in which new samples are generated by mixing pairs of training samples using linear coefficients.

Despite its well established generalization performance (Zhang et al., 2017b; Guo et al., 2018; Verma et al., 2018) , the working mechanism of MixUp is not well understood.

Guo et al. (2018) suggest viewing MixUp as imposing local linearity on the model using points outside of the data manifold.

While this perspective is insightful, we do not believe it paints a full picture of how MixUp operates.

A recent study (Lamb et al., 2019) provides empirical evidence that MixUp improves adversarial robustness, but does not present MixUp as a form of adversarial training.

We build a framework to understand MixUp in a broader context: we argue that adversarial training is a central working principle of MixUp.

To support this contention, we connect MixUp to a MixUplike scheme which does not perform label mixing, and we relate this scheme to adversarial training.

Without label mixing, MixUp becomes a conventional augmentation scheme: input samples are moved, but their original labels are retained.

Because samples are moved in the direction of other samples -which are typically clustered in input space -we describe this method as 'directional'.

Because this method primarily moves training samples in the direction of adversarial classes, this method is analogous to adversarial training.

We thus refer to MixUp without label mixing as directional adversarial training (DAT).

We show that MixUp converges to a subset of DAT under mild conditions, and we thereby argue that adversarial training is a working principle of MixUp.

Inspired by this new understanding of MixUp as a form of adversarial training, and upon realizing that MixUp is (asymptotically) a subset of DAT, we introduce Untied MixUp (UMixUp), a simple enhancement of MixUp which converges to the entire family of DAT schemes, as depicted in Figure  1 .

Untied Mixup mixes data-label training pairs in a similar way to MixUp, with the distinction that the label mixing ratio is an arbitrary function of the sample mixing ratio.

We perform experiments to show that UMixUp's classification performance improves upon MixUp.

In short, this research is motivated by a curiosity to better understand the working of MixUp.

In-sodoing we aim to:

1.

Establish DAT as analogous to adversarial training.

This is discussed in section 4.

2.

Establish UMixUp as a superset of MixUp, and as converging to the entire family of DAT schemes.

In-so-doing, a) establish MixUp's convergence to a subset of DAT, and thereby that it operates analogously to adversarial training; and b) establish UMixUp as a broader class of MixUp-like schemes that operate analogously to adversarial training.

This is discussed in 5.

3.

Establish empirically that UMixUp's classification performance improves upon MixUp.

This is discussed in section 6.

Finally we note that this paper has another contribution.

Conventionally, MixUp is only applicable to baseline models that use cross entropy loss.

All analytical results we develop in this paper are applicable to a wider family of models using any loss function which we term target-linear.

We define target-linearity and experiment with a new loss function called negative cosine-loss to show its potential.

Regular (non-calligraphic) capitalized letters such as X will denote random variables, and their lowercase counterparts, e.g., x, will denote realizations of a random variable.

Any sequence, (a 1 , a 2 , . . . , a n ) will be denoted by a n 1 .

Likewise (A 1 , A 2 , . . .

, A n ) will be denoted by A n 1 , and a sequence of sample pairs ((x 1 , x 1 ), (x 2 , x 2 ), . . .

, (x n , x n )) denoted by (x, x ) n 1 .

For any value a ∈ [0, 1], we will use a as a short notation for 1 − a.

Classification Setting Consider a standard classification problem, in which one wishes to learn a classifier that predicts the class label for a sample.

Formally, let X be a vector space in which the samples of interest live and let Y be the set of all possible labels associated with these samples.

The set of training samples will be denoted by D, a subset of X .

We will use t(x) to denote the true label of x. Let F be a neural network function, parameterized by θ, which maps X to another vector space Z. Let ϕ : Y → Z be a function that maps a label in Y to an element in Z such that for any y, y ∈ Y, if y = y , then ϕ(y) = ϕ(y ).

In the space Z, we refer to F (x) as the model's prediction.

With slight abuse of language, we will occasionally refer to both t(x) and ϕ(t(x)) as the "label" of x. Let : Z ×Z → R be a loss function, using which one defines an overall loss function as

Here we have taken the notational convention that the first argument of represents the model's prediction and the second represents the target label.

In this setting, the learning problem is formulated as minimizing L with respect to its characterizing parameters θ.

We say that a loss function (z, z ) is target-linear if for any scalars α and β,

Target-linear loss functions arise naturally in many settings, for which we now provide two examples.

For convenience, we define the vectors v = F (x) and y = ϕ(t(x)).

Cross-Entropy Loss The conventional cross-entropy loss function, written in our notation, is defined as:

where v and y are constrained to being probability vectors.

We note that in conventional applications, dim(Z) = |Y|, and the target label v is a one-hot vector where

otherwise.

Constraining v to being a probability vector is achieved using a softmax output layer.

Negative-Cosine Loss The "negative-cosine loss", usually used in its negated version, i.e., as the cosine similarity, can be defined as follows.

where v and y are constrained to being unit-length vectors.

For v this can be achieved by simple division at the output layer, and for y by limiting the range of ϕ to an orthonormal basis (making it a conventional label embedding function).

It is clear that the cross-entropy loss CE and the negative-cosine loss NC are both target-linear, directly following from the definition of target-linearity.

Assumptions The theoretical development of this paper relies on two fundamental assumptions, which we call "axioms".

Axiom 1 (Target linearity) The loss function used for the classification setting is target-linear.

That is, the study of MixUp in this paper is in fact goes beyond the standard MixUp, which uses the cross-entropy loss.

Much of the development in this paper concerns drawing sample pairs

n 1 is said to be symmetric if for every (a, b) ∈ D × D, the number of occurrences of (a, b) in the sequence is equal to that of (b, a).

Axiom 2 (Symmetric pair-sampling distribution) Whenever a sample pair (x, x ) is drawn from a distribution Q, Q is assumed to be symmetric.

In the standard MixUp, two samples are drawn independently from D to form a pair, making this condition satisfied.

3 MIXUP, DAT, UNTIED MIXUP

We first provide a summary of each scheme for the reader's convenience.

We then describe each scheme more systematically.

For concision of equations to follow, we define

and

MixUp is a data augmentation scheme in which samples are linearly combined using some mixing ratio λ ∈ [0, 1]:

where λ ∼ P Mix .

A target label is generated using the same mixing ratio λ:

DAT and UMixUp use the same method (2) for generating samples, but use different λ distributions (P DAT and P uMix respectively).

DAT and UMixUp also differ from MixUp in their target labels.

DAT retains the sample's original label:

whereas UMixUp's label mixing ratio is a function of λ:

In Untied MixUp, the label mixing ratio is "untied" from the sample mixing ratio, and can be any γ(λ).

We will refer to γ as the weighting function.

An Untied MixUp scheme is specified both by the its mixing policy P uMix and a weighting function γ.

To draw comparisons between MixUp, DAT, and Untied MixUp schemes, we establish a framework for characterizing their optimization problems. .

We denote the expected value of each scheme's overall loss, L m E , with respect to its mixing ratio Λ. Let n be a positive integer.

In every scheme, a sequence (x, x ) n 1 := ((x 1 , x 1 ), (x 2 , x 2 ), . . . , (x n , x n )) of sample pairs are drawn i.i.d.

from Q, and a sequence λ

In MixUp, we refer to P Mix as the mixing policy.

Directional Adversarial Training (DAT) For any x, x ∈ D and any λ ∈ [0, 1], we denote

In DAT, we refer to P DAT as the adversarial policy.

Let γ be a function mapping [0, 1] to [0, 1].

For any x, x ∈ D and any λ ∈ [0, 1], we denote

Let P m be P uMix , and denote the overall and expected overall loss functions

At this end, it is apparent that MixUp is a special case of Untied MixUp, where the function γ(λ) takes the simple form γ(λ) = λ.

The main theoretical result of this paper is the relationship established between DAT and UMixUp, and by extension MixUp.

Both MixUp and UMixUp will be shown to converge to DAT as the number of mixed sample pairs, n, tends to infinity.

Prior to developing these results, we provide insight into DAT, in terms of its similarity to adversarial training and its regularization mechanisms.

Conventional adversarial training schemes augment the original training dataset by searching for approximations of true adversarials within bounded regions around each training sample.

For a training sample x, a bounded region U known as an L p ball is defined as U = {x + η η η : ||η η η|| p < }.

Over this region, the loss function with respect to the true label of x is maximized.

A typical loss function for an adversarial scheme is

where b is the baseline loss function.

Simply put, baseline training serves to learn correct classification over the training data, whereas adversarial training moves the classification boundary to improve generalization.

DAT, on the other hand, combines intra-class mixing (mixing two samples of the same class) and inter-class mixing (mixing samples of different classes).

Intra-class mixing serves to smooth classification boundaries of inner-class regions, while inter-class mixing perturbs training samples in the direction of adversarial classes, which improves generalization.

Inter-class mixing dwarves intraclass mixing by volume of generated samples seen by the learning model in most many-class learning problems (by a 9-1 ratio in balanced 10-class problems for instance).

DAT, which primarily consists of inter-class mixing, can therefore be seen as analogous to adversarial training.

The key distinction between conventional adversarial training and inter-class mixing is that MixUp movement is determined probabilistically within a bounded region, while adversarial movement is deterministic.

Figure 2 illustrates the connection between standard adversarial training and DAT.

Consider the problem of classifying the blue points and the black points in Figure 2a) , where the dashed curve is a ground-truth classifier and the black curve indicates the classification boundary of F (x), which overfits the training data.

In adversarial training, a training sample x is moved to a location within an L p -ball around x while keeping its label to further train the model; the location, denoted by x 1 in Figure 2b ), is chosen to maximize training loss.

In DAT, a second sample x governs the direction in which x is perturbed.

If x is chosen from a different class as shown in Figure 2c ), then the generated sample x 2 is used to further train the model.

If x is chosen from the same class as shown in Figure 2d ), then the sample x 3 is used in further training.

Note that the inter-class mixed sample x 2 pushes the model's classification boundary closer to the ground-truth classifier, thus connecting adversarial training and DAT.

The intra-class sample x 3 , on the other hand, mainly helps to smooth inner parts of the class region.

The latter behaviour is an additional feature of DAT and MixUp, which distinguishes these schemes from adversarial training.

We now show that Untied MixUp and DAT are equivalent when n tends to infinity.

A consequence of this equivalence is that it infuses both MixUp and UMixUp with the intuition of adversarial training.

To that end, we relate the Untied MixUp loss function, uMix , with the DAT loss function, DAT .

Lemma 1 For any (x, x ) ∈ D × D and any λ ∈ [0, 1],

This result follows directly from the target-linearity of the loss function.

The next two lemmas show that as n tends to infinity, the overall loss of both DAT and UMixUp converge in probability to their respective overall expected losses.

n 1 in probability.

These two lemmas have similar proofs, thus only the proof of Lemma 2 is given in section A.1.

Next we show that as n tends to infinity, UMixUp converges in probability to a subset of DAT, and DAT converges in probability to a subset of UMixUp.

In other words, we show that as n increases, UMixUp converges to being equivalent to the entire class of DAT schemes.

For that purpose, let F denote the space of all functions mapping [0, 1] to [0, 1].

Each configuration in P × F defines an Untied MixUp scheme.

We now define U, which maps a DAT scheme to an Untied MixUp scheme.

Specifically U is a map from P to P × F such that for any p ∈ P, U(p) is a configuration (p , g) ∈ P × F, where

Lemma 4 Let (x, x ) n 1 be a sequence of sample pairs on which an Untied MixUp scheme specified by (P uMix , γ) and a DAT scheme with policy P DAT will apply independently.

If (x, x ) n 1 is symmetric and

We now define another map D u that maps an Untied MixUp scheme to a DAT scheme.

Specifically D u is a map from P × F to P such that for any (p, g)

It is easy to verify that 1 0 p (λ)dλ = 1.

Thus p is indeed a distribution in P and D u is well defined.

Lemma 5 Let (x, x ) n 1 be a sequence of sample pairs on which an Untied MixUp scheme specified by (P uMix , γ) and a DAT scheme with policy P DAT will apply independently.

If (x, x ) n 1 is symmetric and

Lemmas 2, 3, 4 and 5 provide the building blocks for theorem 1, which we state hereafter.

As n increases, both DAT and UMixUp converge in probability toward their respective expected loss (lemmas 2 and 3).

Since as n increases, the sequence (x, x ) n 1 becomes arbitrarily close to the symmetric sampling distribution Q, then by lemma 4 the family of DAT schemes converges in probability to a subset of UMixUp schemes.

Lemma 5 proves the converse, i.e. that as n increases the family of UMixUp schemes converges in probability to a subset of DAT schemes.

As n n n increases, the family of UMixUp schemes therefore converges in probability to the entire family of DAT schemes.

On this sample-pair data, an Untied MixUp scheme specified by (P Mix , γ) and a DAT scheme specified by P DAT will apply.

In the Untied MixUp scheme, let Λ ∞ 1 be drawn i.i.d.

from P Mix ; in the DAT scheme, let Υ

The equivalence between the two families of schemes also indicates that there are DAT schemes that do not correspond to a MixUp scheme.

These DAT schemes correspond to Untied MixUp scheme beyond the standard MixUp.

The relationship between MixUp, DAT and Untied MixUp is shown in Figure 1 .

We consider an image classification task on the Cifar10, Cifar100, MNIST and Fashion-MNIST datasets.

The baseline classifier chosen is PreActResNet18 (see Liu (2017) Two target-linear loss functions are essayed: cross-entropy (CE) loss and the negative-cosine (CE) loss as defined earlier.

We implement CE loss similarly to previous works, which use CE loss to implement the baseline model.

In our implementation of the NC loss model, for each label y, ϕ(y) is mapped to a randomly selected unit-length vector of dimension d and fixed during training; the feature map of the original PreActResNet18 is linearly transformed to a d-dimensional vector.

The dimension d is chosen as 300 for Cifar10, MNIST and Fashion-Mnist (which have one black-andwhite channel) and 700 for Cifar100 (which has 3 colored channels).

Our implementation of MixUp and Untied MixUp improves upon the published implementation from the original authors of MixUp Zhang et al. (2017b) .

For example, the original authors' implementation samples only one λ per mini-batch, giving rise to unnecessarily higher stochasticity of the gradient signal.

Our implementation samples λ independently for each sample.

Additionally, the original code combines inputs by mixing a mini-batch of samples with a shuffled version of itself.

This approach introduces a dependency between sampled pairs and again increases the stochasticity of training.

Our implementation creates two shuffled copies of the entire training dataset prior to each epoch, pairs them up, and then splits them into mini-batches.

This gives a closer approximation to i.i.d.

sampling and makes training smoother.

While these implementation improvements have merit on their own, they do not provide a theoretical leap in understanding, and so we do not quantify their impact in our results analysis.

All models examined are trained using mini-batched backpropagation, for 200 epochs.

We sweep over the policy space of MixUp and Untied MixUp.

For MixUp, it is sufficient to consider distribution P Mix to be symmetric about 0.5.

Thus we consider only consider P Mix in the form of B(α, α), and scan through a single parameter α systematically.

Since the policy of Untied MixUp is in the form of U(B(α, β)), searching through (α, β) becomes more difficult.

Thus our policy search for Untied MixUp is restricted to an ad hoc heuristic search.

For this reason, the found best policy for Untied MixUp might be quite far from the true optimal.

The main results of our experiments are given in tables 1 to 4.

As shown in the tables, each setting is run 100 times.

For each run, we compute the error rate in a run as the average test error rate over the final 10 epochs.

The estimated mean ("MEAN") performance of a setting is computed as the average of the error rates over all runs for the same setting.

The 95%-confidence interval ("ConfInt") for the estimated mean performance is also computed and shown in the table.

From these results, we see that the Untied MixUp schemes each outperform their MixUp counterparts.

Specifically, in 6 of the 8 cases (those printed in bold font), the confidence interval of Untied MixUp is completely disjoint from that of the corresponding MixUp scheme; and in some cases, the separation of confidence intervals is by a large margin.

Note that the baseline model (PreActResNet18) has been designed with highly focused inductive bias for image classification tasks.

Under such an inductive bias, one expects that the room for regularization (or the "amount of overfitting") isn't abundant.

As such, we consider the improvement of Untied MixUp over MixUp rather significant.

The results show empirically that MixUp and Untied MixUp both work on the NC loss models.

This validates our generalization of MixUp (and Untied MixUp) to models built with target linear losses.

model policy runs MEAN ConfInt baseline-CE − 100 5.476% 0.027% mixUp-CE B(0.9, 0.9) 100 4.199% 0.023% uMixUp-CE U(B(2.2, 0.9)) 100 4.177% 0.025% baseline-NC − 100 5.605% 0.030% mixUp-NC B(1.0, 1.0) 100 4.508% 0.022% uMixUp-NC U(B(1.8, 1.0)) 100 4.455% 0.025% 1.3, 0.9) ) 100 23.819% 0.054% 1.7, 1.0) ) 100 0.609% 0.005% baseline-NC − 100 0.720% 0.007% mixUp-NC B(1.0, 1.0) 100 0.607% 0.004% uMixUp-NC U(B(1.3, 0.9)) 100 0.592% 0.005%

) be defined according to (4), with the first n elements of (x, x ) ∞ 1 and the first n elements of Λ ∞ 1 as input.

Define

For any given λ E λ∼P Mix γ(λ)

where (a) is due to a change of variable in the integration, (b) is due to the symmetry of (x, x ) n 1 .

Note that in equation 5 g(λ) is undefined at values of λ for which the denominator is zero.

But the lemma holds true because the denominator is only zero when p(λ) = 0, so those λ for which g(λ) is undefined never get drawn in the DAT scheme.

A.3 PROOF OF LEMMA 5:

DAT (x k , x k , λ) γ(λ)P Mix (λ) + γ(λ)P Mix (1 − λ)

) .

where (a) is due to the symmetry of (x, x ) n 1 , and (b) is by a change of variable in the second term (renaming 1 − λ as λ).

<|TLDR|>

@highlight

We present a novel interpretation of MixUp as belonging to a class highly analogous to adversarial training, and on this basis we introduce a simple generalization which outperforms MixUp