Regularization is one of the crucial ingredients of deep learning, yet the term regularization has various definitions, and regularization methods are often studied separately from each other.

In our work we present a novel, systematic, unifying taxonomy to categorize existing methods.

We distinguish methods that affect data, network architectures, error terms, regularization terms, and optimization procedures.

We identify the atomic building blocks of existing methods, and decouple the assumptions they enforce from the mathematical tools they rely on.

We do not provide all details about the listed methods; instead, we present an overview of how the methods can be sorted into meaningful categories and sub-categories.

This helps revealing links and fundamental similarities between them.

Finally, we include practical recommendations both for users and for developers of new regularization methods.

Regularization is one of the key elements of machine learning, particularly of deep learning BID37 , allowing to generalize well to unseen data even when training on a finite training set or with an imperfect optimization procedure.

In the traditional sense of optimization and also in older neural networks literature, the term "regularization" is reserved solely for a penalty term in the loss function BID12 .

Recently, the term has adopted a broader meaning: Goodfellow et al. (2016, Chap.

5 ) loosely define it as "any modification we make to a learning algorithm that is intended to reduce its test error but not its training error".

We find this definition slightly restrictive and present our working definition of regularization, since many techniques considered as regularization do reduce the training error (e.g. weight decay in AlexNet ).

Definition 1.

Regularization is any supplementary technique that aims at making the model generalize better, i.e. produce better results on the test set.

This can include various properties of the loss function, the loss optimization algorithm, or other techniques.

Note that this definition is more in line with machine learning literature than with inverse problems literature, the latter using a more restrictive definition.

In this work, we create a novel, systematic, unifying taxonomy of regularization methods for deep learning.

We analyze existing methods and identify their atomic building blocks.

This leads to decoupling of two important concepts: Which assumptions the methods rely on (and try to enforce), and which mathematical and algorithmic tools they use.

In turn, this enables better understanding of existing methods and speeds up development of new ones: The researchers can focus either on finding new, better ways of enforcing existing assumptions, or focus on discovery of new assumptions that can be enforced in some existing way.

Before we proceed to the presentation of our taxonomy, we revisit some basic machine learning theory in Section 2.

This will provide a justification of the top level of the taxonomy.

In Sections 3-7, we continue with a finer division of the individual classes of the regularization techniques, aiming at separating as many clearly separable concepts as possible and isolating atomic building blocks of individual methods.

Finally, in Section 8 we present our practical recommendations for using existing methods and designing new methods.

We are aware that the many research works discussed in this taxonomy cannot be summarized in a single sentence.

For the sake of structuring the multitude of papers, we decided to merely describe a certain subset of their properties according to the focus of our taxonomy.

The central task of our interest is model fitting: finding a function that can well approximate a desired mapping from inputs to desired outputs ( ).

A given input can have an associated target which dictates the desired output ( ) directly (or in some applications indirectly BID104 BID57 ).

A typical example of having available targets is supervised learning.

Data samples ( , ) then follow a ground truth probability distribution .In many applications, neural networks have proven to be a good family of functions to choose from.

A neural network is a function : ↦ → with trainable weights ∈ .

Training the network means finding a weight configuration * , which is a result of performing a minimization procedure of a loss function ℒ :→ R as follows: DISPLAYFORM0 Usually the loss function takes the form of expected risk : DISPLAYFORM1 where we identify two parts, an error function and a regularization term .

The error function depends on the targets and assigns a penalty to model predictions according to their consistency with the targets.

The regularization term assigns a penalty to the model based on other criteria.

It may depend on anything except the targets, for example on the weights (see Section 6).The expected risk cannot be minimized directly since the data distribution is unknown.

Instead, a training set sampled from the distribution is given.

The minimization of the expected risk can be then approximated by (approximately) minimizing the empirical riskL: DISPLAYFORM2 where ( , ) are samples from .

Now we have the minimal background to formalize the division of regularization methods into a systematic taxonomy.

In the minimization of the empirical risk, Eq. (3), we can identify the following elements that are responsible for the value of the learned weights, and thus can contribute to regularization:• : The training set, discussed in Section 3• : The selected model family, discussed in Section 4• : The error function, briefly discussed in Section 5• : The regularization term, discussed in Section 6• The optimization procedure itself, discussed in Section 7Ambiguity regarding the splitting of methods into these categories and their subcategories is discussed in Appendix A using notation from Section 3.

The quality of a trained model depends largely on the training data.

Apart from acquisition/selection of appropriate training data, it is possible to employ regularization via data.

This is done by applying some transformation to the training set , resulting in a new set .

Some transformations perform feature extraction or pre-processing, modifying the feature space or the distribution of the data to some representation simplifying the learning task.

Other methods allow generating new samples to create a larger, possibly infinite, augmented dataset.

These two principles are somewhat independent and may be combined.

The goal of regularization via data is either one of them, or the other, or both.

They both rely on transformations with (stochastic) parameters: Definition 2.

Transformation with stochastic parameters is a function with parameters which follow some probability distribution.

In this context we consider which can operate on network inputs, activations in hidden layers, or targets.

An example of a transformation with stochastic parameters is the corruption of inputs by Gaussian noise BID13 BID1 : DISPLAYFORM0 The stochasticity of the transformation parameters is responsible for generating new samples, i.e. data augmentation.

Note that the term data augmentation often refers specifically to transformations of inputs or hidden activations, but here we also list transformations of targets for completeness.

The exception to the stochasticity is when follows a delta distribution, in which case the transformation parameters become deterministic and the dataset size is not augmented.

We can categorize the data-based methods according to the properties of the used transformation and of the distribution of its parameters.

We identify the following criteria for categorization (some of them later serve as columns in Tables 1-2) :Stochasticity of the transformation parameters• Deterministic parameters: Parameters follow a delta distribution, size of the dataset remains unchanged • Stochastic parameters: Allow generation of a larger, possibly infinite, dataset.

Various strategies for sampling of exist: -Random: Draw a random from the specified distribution -Adaptive: Value of is the result of an optimization procedure, usually with the objective of maximizing the network error on the transformed sample (such "challenging" sample is considered to be the most informative one at current training stage), or minimizing the difference between the network prediction and a predefined fake target ′ * Constrained optimization: found by maximizing error under hard constraints (support of the distribution of controls the strongest allowed transformation) * Unconstrained optimization: found by maximizing modified error function, using the distribution of as weighting (proposed herein for completeness, not yet tested) * Stochastic: found by taking a fixed number of samples of and using the one yielding the highest error Effect on the data representation• Representation-preserving transformations: Preserve the feature space and attempt to preserve the data distribution • Representation-modifying transformations:

Map the data to a different representation (different distribution or even new feature space) that may disentangle the underlying factors of the original representation and make the learning problem easier

• Input: Transformation is applied to• Hidden-feature space: Transformation is applied to some deep-layer representation of samples (this also uses parts of and to map the input into the hidden-feature space; such transformations act inside the network and thus can be considered part of the architecture, additionally fitting Section 4)• Target: Transformation is applied to (can only be used during the training phase since labels are not shown to the model at test time)

• Generic: Applicable to all data domains• Domain-specific: Specific (handcrafted) for the problem at hand, for example image rotationsDependence of the distribution of• ( ): distribution of is the same for all samples DISPLAYFORM0 distribution of can be different for each input vector (with implicit dependence on and if the transformation is in hidden-feature space)• ( | ): distribution of depends on the whole training dataset• ( |x): distribution of depends on a batch of training inputs (for example (parts of) the current mini-batch, or also previous mini-batches)• ( |time): distribution of depends on time (current training iteration)• ( | ): distribution of depends on some trainable parameters subject to loss minimization (i.e. the parameters evolve during training along with the network weights )• Combinations of the above, e.g. DISPLAYFORM1

• Training: Transformation of training samples• Test: Transformation of test samples, for example multiple augmented variants of a sample are classified and the result is aggregated over them A review of existing methods that use generic transformations can be found in Table 1 .

Dropout in its original form BID101 ) is one of the most popular methods from the generic group, but also several variants of Dropout have been proposed that provide additional theoretical motivation and improved empirical results (Standout BID5 , Random dropout probability BID16 , Bayesian dropout BID73 , Test-time dropout BID31 ).

Table 2 contains a list of some domain-specific methods focused especially on the image domain.

Here the most used method is rigid and elastic image deformation.

Target-preserving data augmentation In the following, we discuss an important group of methods: target-preserving data augmentation.

These methods use stochastic transformations in input and hidden-feature spaces, while preserving the original target .

As can be seen in the respective two columns in Tables 1-2, most of the listed methods have exactly these properties.

These methods transform the training set to a distribution , which is used for training instead.

In other words, the training samples ( , ) ∈ are replaced in the empirical risk loss function (Eq. Gaussian noise on input BID12 BID1 Label smoothing (Szegedy et al., 2016, Sec. 7; BID37 , Chap.

7) DISPLAYFORM0 Model compression (mimic models, distilled models) BID17 BID4 BID48 ( | , ) Target Deterministic Training Table 1 : Existing generic data-based methods classified according to our taxonomy.

BID6 BID113 BID97 BID21 Table 2 : Existing domain-specific data-based methods classified according to our taxonomy.

Table columns are described in Section 3.

Note that these methods are never applied on the hidden features, because domain knowledge cannot be applied on them.bridge the limited-data gap between the expected and the empirical risk, Eqs. FORMULA1 - FORMULA2 .

While unlimited sampling from provides more data than the original dataset , both of them usually are merely approximations of the ground truth data distribution or of an ideal training dataset; both and have their own distinct biases, advantages and disadvantages.

For example, elastic image deformations result in images that are not perfectly realistic; this is not necessarily a disadvantage, but it is a bias compared to the ground truth data distribution; in any case, the advantages (having more training data) often prevail.

In some cases, it may be even desired for to be deliberately different from the ground truth data distribution.

For example, in case of class imbalance (unbalanced abundance or importance of classes), a common regularization strategy is to undersample or oversample the data, sometimes leading to a less realistic but better models.

This is how an ideal training dataset may be different from the ground truth data distribution.

If the transformation is additionally representation-preserving, then the distribution created by the transformation attempts to mimic the ground truth data distribution .

Otherwise, the notion of a "ground truth data distribution" in the modified representation may be vague.

We provide more details about the transition from to in Appendix B.Summary of data-based methods Data-based regularization is a popular and very useful way to improve the results of deep learning.

In this section we formalized this group of methods and showed that seemingly unrelated techniques such as Target-preserving data augmentation, Dropout, or Batch normalization are methodologically surprisingly close to each other.

In Section 8 we discuss future directions that we find promising.

A network architecture can be selected to have certain properties or match certain assumptions in order to have a regularizing effect.

Method class Assumptions about an appropriate learnable input-output mapping Any chosen (not overly complex) architecture * Mapping can be well approximated by functions from the chosen family which are easily accessible by optimization.

Small network * Mapping is simple (complexity of the mapping depends on the number of network units and layers).Deep network * The mapping is complex, but can be decomposed into a composition (or generally into a directed acyclic graph) of simple nonlinear transformations, e.g. affine transformation followed by simple nonlinearity (fully-connected layer), "multi-channel convolution" followed by simple nonlinearity (convolutional layer), etc.

Hard bottleneck (layer with few neurons); soft bottleneck (e.g. Jacobian penalty BID88 ), see Section 6)Layer operation Data concentrates around a lower-dimensional manifold; has few factors of variation.

Convolutional networks BID30 Rumelhart et al., 1986, pp.

348-352; BID64 BID97 Layer operation Spatially local and shift-equivariant feature extraction is all we need.

Dilated convolutions BID115 Layer operation Like convolutional networks.

Additionally: Sparse sampling of wide local neighborhoods provides relevant information, and better preserves relevant high-resolution information than architectures with downscaling and upsampling.

Strided convolutions (see BID26 Layer operation The mapping is reliable at reacting to features that do not vary too abruptly in space, i.e. which are present in several neighboring pixels and can be detected even if the filter center skips some of the pixels.

The output is robust towards slight changes of the location of features, and changes of strength/presence of spatially strongly varying features.

Pooling Layer operation The output is invariant to slight spatial distortions of the input (slight changes of the location of (deep) features).

Features that are sensitive to such distortions can be discarded.

Stochastic pooling Layer operation The output is robust towards slight changes of the location (like pooling) but also of the strength/presence of (deep) features.

Training with different kinds of noise (including Dropout; see Section 3)

The mapping is robust to noise: the given class of perturbations of the input or deep features should not affect the output too much.

Dropout BID101 , DropConnect BID106 , and related methods Noise Extracting complementary (non-coadapted) features is helpful.

Noncoadapted features are more informative, better disentangle factors of variation.

(We want to disentangle factors of variation because they are entangled in different ways in inputs vs. in outputs.)

When interpreted as ensemble learning: usual assumptions of ensemble learning (predictions of weak learners have complementary info and can be combined to strong prediction).Maxout units BID36 Layer operation Assumptions similar to Dropout, with more accurate approximation of model averaging (when interpreted as ensemble learning) Skip-connections BID67 BID53 Connections between layersCertain lower-level features can directly be reused in a meaningful way at (several) higher levels of abstraction Linearly augmented feed-forward network BID105 Connections between layers Skip-connections that share weights with the non-skip-connections.

Helps against vanishing gradients.

Rather changes the learning algorithm than the network mapping.

Residual learning BID45 Connections between layersLearning additive difference of a mapping (or its compositional parts) from the identity mapping is easier than learning itself.

Meaningful deep features can be composed as a sum of lower-level and intermediate-level features.

Stochastic depth BID54 , DropIn BID99 Connections between layers; noise Similar to Dropout: extracting complementary (non-coadapted) features across different levels of abstraction is helpful; implicit model ensemble.

Similar to Residual learning: meaningful deep features can be composed as a sum of lower-level and intermediate-level features, with the intermediate-level ones being optional, and leaving them out being meaningful data augmentation.

Similar to Mollifying networks: simplifying random parts of the mapping improves training.

Mollifying networks BID40 Connections between layers; noise The mapping can be easier approximated by estimating its decreasingly linear simplified version Network information criterion BID78 , Network growing and network pruning (see Bishop, 1995a, Sec. 9.5) Model selection Optimal generalization is reached by a network that has the right number of units (not too few, not too many)Multi-task learning (see BID18 BID90 * Several tasks can help each other to learn mutually useful feature extractors, as long as the tasks do not compete for resources (network capacity) Assumptions about the mapping An input-output mapping must have certain properties in order to fit the data well.

Although it may be intractable to enforce the precise properties of an ideal mapping, it may be possible to approximate them by simplified assumptions about the mapping.

These properties and assumptions can then be imposed upon model fitting in a hard or soft manner.

This limits the search space of models and allows finding better solutions.

An example is the decision about the number of layers and units, which allows the mapping to be neither too simple nor too complex (thus avoiding underfitting and overfitting).

Another example are certain invariances of the mapping, such as locality and shift-equivariance of feature extraction hardwired in convolutional layers.

Overall, the approach of imposing assumptions about the input-output mapping discussed in this section is the selection of the network architecture .

The choice of architecture on the one hand hardwires certain properties of the mapping; additionally, in an interplay between and the optimization algorithm (Section 7), certain weight configurations are more likely accessible by optimization than others, further limiting the likely search space in a soft way.

A complementary way of imposing certain assumptions about the mapping are regularization terms (Section 6), as well as invariances present in the (augmented) data set (Section 3).Assumptions can be hardwired into the definition of the operation performed by certain layers, and/or into the connections between layers.

This distinction is made in TAB3 , where these and other methods are listed.

In Section 3 about data, we mentioned regularization methods that transform data in the hidden-feature space.

They can be considered part of the architecture.

In other words, they fit both Sections 3 (data) and 4 (architecture).

These methods are listed in Table 1 with hidden features as their transformation space.

Weight sharing Reusing a certain trainable parameter in several parts of the network is referred to as weight sharing.

This usually makes the model less complex than using separately trainable parameters.

An example are convolutional networks BID64 .

Here the weight sharing does not merely reduce the number of weights that need to be learned; it also encodes the prior knowledge about the shift-equivariance and locality of feature extraction.

Another example is weight sharing in autoencoders.

Activation functions Choosing the right activation function is quite important; for example, using Rectified linear units (ReLUs) improved the performance of many deep architectures both in the sense of training times and accuracy as well as overcoming the need for greedy layer-wise pre-training BID41 BID56 BID79 BID35 .

The success of ReLUs can be partially attributed to the fact that they provide more expressive families of mappings compared to sigmoid activations (in the sense that the classical sigmoid nonlinearity can be approximated very well 2 with only two ReLUs, but it takes an infinite number of sigmoid units to approximate a ReLU) and their affine extrapolation to unknown regions of data space seems to provide better generalization in practice than the "stagnating" extrapolation of sigmoid units.

However, their hard negative cut-off and unbounded positive part are not always desired properties.

Some activation functions were designed explicitly for regularization.

For Dropout, Maxout units BID36 allow a more precise approximation of the geometric mean of the model ensemble predictions at test time.

Stochastic pooling , on the other hand, is a noisy version of max-pooling.

The authors claim that this allows modelling distributions of activations instead of taking just the maximum.

Noisy models Stochastic pooling was one example of a stochastic generalization of a deterministic model.

Some models are stochastic by injecting random noise into various parts of the model.

The most frequently used noisy model is Dropout BID101 .Multi-task learning A special type of regularization is multi-task learning (see BID18 BID90 , where the network is modified to predict targets for several tasks at once.

It can be combined with semi-supervised learning to utilize unlabeled data on an auxiliary task BID85 .

A similar concept of sharing knowledge between tasks is also utilized in meta-learning, where multiple tasks from the same domain are learned sequentially, using previously gained knowledge as bias for new tasks BID7 ; and transfer learning, where knowledge from one domain is transferred into another domain BID82 .

These approaches differ from other methods in the sense that they require some additional target data, which are not always available.

Model selection The best among several trained models (e.g. with different architectures) can be selected by evaluating the predictions on a validation set.

It should be noted that this holds for selecting the best combination of all techniques (Sections 3-7), not just architecture; and that the validation set used for model selection in the "outer loop" should be different from the validation set used e.g. for Early stopping (Section 7), and different from the test set BID19 .

However, there are also model selection methods that specifically target the selection of the number of units in a specific network architecture, e.g. using network growing and network pruning (see BID12 , Sec. 9.5), or additionally do not require a validation set, e.g. the Network information criterion to compare models based on the training error and second derivatives of the loss function BID78 .

Ideally, the error function reflects an appropriate notion of quality, and in some cases some assumptions about the data distribution.

Typical examples are mean squared error or cross-entropy.

The error function can also have a regularizing effect.

An example is Dice coefficient optimization BID75 which is robust to class imbalance.

Moreover, the overall form of the loss function can be different than Eq. (3).

For example, in certain loss functions that are robust to class imbalance, the sum is taken over pairwise combinations × of training samples BID114 , rather than over training samples.

But such alternatives to Eq. (3) are rather rare, and similar principles apply.

If additional tasks are added for a regularizing effect (multi-task learning (see BID18 BID90 ), then targets are modified to consist of several tasks, the mapping is modified to produce an according output , and is modified to account for the modified and .

Besides, there are regularization terms that depend on / .

They depend on and thus in our definition are considered part of rather than of , but they are listed in Section 6 among (rather than here) for a better overview.

Regularization can be achieved by adding a regularizer into the loss function.

Unlike the error function (which expresses consistency of outputs with targets), the regularization term is independent of the targets.

Instead, it is used to encode other properties of the desired model, to provide inductive bias (i.e. assumptions about the mapping other than consistency of outputs with targets).

The value of can thus be computed for an unlabeled test sample, whereas the value of cannot.

The independence of from has an important implication: it allows additionally using unlabeled samples (semi-supervised learning) to improve the learned model based on its compliance with some desired properties BID92 .

For example, semi-supervised learning with ladder networks BID85 combines a supervised task with an unsupervised auxiliary denoising task in a "multi-task" learning fashion. (For alternative interpretations, see Appendix A.) Unlabeled samples are extremely useful when labeled samples are scarce.

A Bayesian perspective on the combination of labeled and unlabeled data in a semi-supervised manner is offered by BID62 .

A classical regularizer is weight decay (see BID83 BID61 ; Goodfellow et al., 2016, Chap.

7): DISPLAYFORM0 where is a weighting term controlling the importance of the regularization over the consistency.

From the Bayesian perspective, weight decay corresponds to using a symmetric multivariate normal distribution as prior for the weights: ( ) = ( |0, −1 I) BID81 .

Indeed, − log ( |0, DISPLAYFORM1 Weight decay has gained big popularity, and it is being successfully used; BID59 even observe reduction of the error on the training set.

Another common prior assumption that can be expressed via the regularization term is "smoothness" of the learned mapping (see BID8 , Section 3.2): if 1 ≈ 2 , then( 1 ) ≈ ( 2 ).

It can be expressed by the following loss term: DISPLAYFORM2 where ‖·‖ denotes the Frobenius norm, and ( ) is the Jacobian of the neural network input-to-output mapping for some fixed network weights .

This term penalizes mappings with large derivatives, and is used in contractive autoencoders BID88 .The domain of loss regularizers is very heterogeneous.

We propose a natural way to categorize them by their dependence.

We saw in Eq. (5) that weight decay depends on only, whereas the Jacobian penalty in Eq. (6) depends on , , and .

More precisely, the Jacobian penalty uses the derivative / of output = ( ) w.r.t.

input .

(We use vector-by-vector derivative notation from matrix calculus, i.e. / = ( )/ = is the Jacobian of with fixed weights .)

We identify the following dependencies of :• Dependence on the weights • Dependence on the network output = ( ) • Dependence on the derivative / of the output = ( ) w.r.t.

the weights • Dependence on the derivative / of the output = ( ) w.r.t.

the input • Dependence on the derivative / of the error term w.r.t.

the input ( depends on , and according to our definition such methods belong to Section 5, but they are listed here for overview) A review of existing methods can be found in Table 4 .

Weight decay seems to be still the most popular of the regularization terms.

Some of the methods are equivalent or nearly equivalent to other methods from different taxonomy branches.

For example, Tangent prop simulates minimal data augmentation BID96 ; Injection of small-variance Gaussian noise BID13 BID1 is an approximation of Jacobian penalty BID88 ; and Fast dropout BID108 ) is (in shallow networks) a deterministic approximation of Dropout.

This is indicated in the Equivalence column in Table 4 .

The last class of the regularization methods according to our taxonomy is the regularization through optimization.

While this may sound unusual, optimization and regularization cannot be clearly separated in the context of deep learning where it is not so crucial what the optimum of the empirical risk is (because it cannot be found exactly, and the ultimate goal is minimizing the expected risk anyway).

Instead, the shape of the loss function and the optimization procedure play together to dictate how the training proceeds in the weight space and where it ends up.

To demonstrate the overlap of regularization and optimization, we show in FIG1 how one of the most prominent regularization methods, Dropout, can be seen as a modification of the optimization procedure.

Stochastic gradient descent (SGD) (see BID15 ) (along with its derivations) is the most frequently used optimization algorithm in the context of deep neural networks and is the center of our attention.

We also list some alternative methods below.

Weight decay (see BID83 BID61 Goodfellow et al., 2016, Chap.

7) 2 norm on network weights (not biases).Favors smaller weights, thus for usual architectures tends to make the mapping less "extreme", more robust to noise in the input.

Early stopping (see BID22 Goodfellow et al., 2016, Chap.

7) Weight smoothing BID61 Penalizes 2 norm of gradients of learned filters, making them smooth.

Not beneficial in practice.

Weight elimination BID109 Similar to weight decay but favors few stronger connections over many weak ones.

Goal similar to Narrow and broad Gaussians Soft weight-sharing BID81 Mixture-of-Gaussians prior on weights.

Generalization of weight decay.

Weights are pushed to form a predefined number of groups with similar values.

Narrow and broad Gaussians BID81 BID14 Weights come from two Gaussians, a narrow and a broad one.

Special case of Soft weight-sharing.

Fast dropout approximation BID108 Approximates the loss that dropout minimizes.

Weighted 2 weight penalty.

Only for shallow networks.

Mutual exclusivity BID92 Unlabeled samples push decision boundaries to low-density regions in input space, promoting sharp (confident) predictions.

Segmentation with binary potentials BID11 Penalty on anatomically implausible image segmentations.

Flat minima search BID50 Penalty for sharp minima, i.e. for weight configurations where small weight perturbation leads to high error increase.

Flat minima have low Minimum description length (i.e. exhibit ideal balance between training error and model complexity) and thus should generalize better BID89 .Tangent prop BID96 2 penalty on directional derivative of mapping in the predefined tangent directions that correspond to known input-space transformations.

Simple data augmentation Jacobian penalty BID88 2 penalty on the Jacobian of (parts of) the network mappingsmoothness prior.

Noise on inputs injection (not exact (see BID1 Manifold tangent classifier BID86 Like tangent prop, but the input "tangent" directions are extracted from manifold learned by a stack of contractive autoencoders and then performing SVD of the Jacobian at each input sample.

Hessian penalty BID87 Fast way to approximate 2 penalty of the Hessian of by penalizing Jacobian with noisy input.

Tikhonov regularizers BID13 2 penalty on (up to) -th derivative of the learned mapping w.r.t.

input.

For penalty on first derivative: noise on inputs injection (not exact (see BID1 ) Loss-invariant backpropagation (Demyanov et al., 2015, Sec. 3.1; BID72 (2 ) norm of gradient of loss w.r.t.

input.

Changes the mapping such that the loss becomes rather invariant to changes of the input.

Prediction-invariant backpropagation (Demyanov et al., 2015, Sec. 3.2) (2 ) norm of directional derivative of mapping w.r.t.

input in the direction of causing the largest increase in loss.

Table 4 : Regularization terms, with dependencies marked by .

Methods that depend on / implicitly depend on targets and thus can be considered part of the error function (Section 5) rather than regularization term (Section 6).

Stochastic gradient descent is an iterative optimization algorithm using the following update rule: DISPLAYFORM0 where ∇ℒ( , ) is the gradient of the loss ℒ evaluated on a mini-batch from the training set .

It is frequently used in combination with momentum and other tweaks improving the convergence speed (see BID110 .

Moreover, the noise induced by the varying mini-batches helps the algorithm escape saddle points BID32 ; this can be further reinforced by adding supplementary gradient noise BID80 BID20 .If the algorithm reaches a low training error in a reasonable time (linear in the size of the training set, allowing multiple passes through ), the solution generalizes well under certain mild assumptions; in that sense SGD works as an implicit regularizer : a short training time prevents overfitting even without any additional regularizer used BID42 .

This is in line with BID117 ) who find in a series of experiments that regularization (such as Dropout, data augmentation, and weight decay) is by itself neither necessary nor sufficient for good generalization.

We divide the methods into three groups: initialization/warm-start methods, update methods, and termination methods, discussed in the following.

Initialization and warm-start methods These methods affect the initial selection of the model weights.

Currently the most frequently used method is sampling the initial weights from a carefully tuned distribution.

There are multiple strategies based on the architecture choice, aiming at keeping the variance of activations in all layers around 1, thus preventing vanishing or exploding activations (and gradients) in deeper layers (Glorot and Bengio, 2010, Sec. 4.2; BID44 .Another (complementary) option is pre-training on different data, or with a different objective, or with partially different architecture.

This can prime the learning algorithm towards a good solution before the fine-tuning on the actual objective starts.

Pre-training the model on a different task in the same domain may lead to learning useful features, making the primary task easier.

However, pre-trained models are also often misused as a lazy approach to problems where training from scratch or using thorough domain adaptation, transfer learning, or multi-task learning methods would be worth trying.

On the other hand, pre-training or similar techniques may be a useful part of such methods.

Finally, with some methods such as Curriculum learning BID10 , the transition between pre-training and fine-tuning is smooth.

We refer to them as warm-start methods.• Initialization without pre-training -Random weight initialization (Rumelhart et al., 1986, p. 330; BID34 BID44 BID46 ) -Orthogonal weight matrices BID94 -Data-dependent weight initialization BID58 • Initialization with pre-training -Greedy layer-wise pre-training BID49 BID9 BID27 (has become less important due to advances (e.g. ReLUs) in effective end-to-end training that optimizes all parameters simultaneously) -Curriculum learning BID10 ) -Spatial contrasting BID51 -Subtask splitting BID38 Update methods This class of methods affects individual weight updates.

There are two complementary subgroups: Update rules modify the form of the update formula; Weight and gradient filters are methods that affect the value of the gradient or weights, which are used in the update formula, e.g. by injecting noise into the gradient BID80 .

Figure 1: Effect of Dropout on weight optimization.

Starting from the current weight configuration (red dot), all weights of certain neurons are set to zero (black arrow), descent step is performed in that subspace (teal arrow), and then the discarded weight-space coordinates are restored (blue arrow).Again, it is not entirely clear which of the methods only speed up the optimization and which actually help the generalization.

BID110 show that some of the methods such as AdaGrad or Adam even lose the regularization abilities of SGD.• Update rules -Momentum, Nesterov's accelerated gradient method, AdaGrad, AdaDelta, RMSProp, Adam-overview in BID110 ) -Learning rate schedules BID33 BID52 -Online batch selection BID69 -SGD alternatives: L-BFGS BID65 BID63 , Hessianfree methods BID74 , Sum-of-functions optimizer BID100 , ProxProp BID29 • Gradient and weight filters -Annealed Langevin noise BID80 -AnnealSGD BID20 -Dropout BID101 corresponds to optimization steps in subspaces of weight space, see FIG1 -Annealed noise on targets BID107 ) (works as noise on gradient, but belongs rather to data-based methods, Section 3)Termination methods There are numerous possible stopping criteria and selecting the right moment to stop the optimization procedure may improve the generalization by reducing the error caused by the discrepancy between the minimizers of expected and empirical risk:The network first learns general concepts that work for all samples from the ground truth distribution before fitting the specific sample and its noise BID60 .The most successful and popular termination methods put a portion of the labeled data aside as a validation set and use it to evaluate performance (validation error ).

The most prominent example is Early stopping (see BID84 .

BID22 show that Early stopping has the same effect as Weight decay regularization penalty term in multi-layered perceptrons with linear output units; however, its hyperparameters are easier to tune.

In scenarios where the training data are scarce it is possible to resort to termination methods that do not use a validation set.

The simplest case is fixing the number of passes through the training set.• Termination using a validation set -Early stopping (see BID77 BID84 -Choice of validation set size based on test set size BID0 • Termination without using a validation set -Fixed number of iterations -Optimized approximation algorithm BID66 8 Recommendations, discussion, conclusionsWe see the main benefits of our taxonomy to be two-fold: Firstly, it provides an overview of the existing techniques to the users of regularization methods and gives them a better idea of how to choose the ideal combination of regularization techniques for their problem.

Secondly, it is useful for development of new methods, as it gives a comprehensive overview of the main principles that can be exploited to regularize the models.

We summarize our recommendations 3 in the following paragraphs:Recommendations for users of existing regularization methods Overall, using the information contained in data as well as prior knowledge as much as possible, and primarily starting with popular methods, the following procedure can be helpful:• Common recommendations for the first steps: -Deep learning is about disentangling the factors of variation.

An appropriate data representation should be chosen; known meaningful data transformations should not be outsourced to the learning.

Redundantly providing the same information in several representations is okay.

-Output nonlinearity and error function should reflect the learning goals.-A good starting point are techniques that usually work well (e.g. ReLU, successful architectures).

Hyperparameters (and architecture) can be tuned jointly, but "lazily" (interpolating/extrapolating from experience instead of trying too many combinations).

-Often it is helpful to start with a simplified dataset (e.g. fewer and/or easier samples) and a simple network, and after obtaining promising results gradually increasing the complexity of both data and network while tuning hyperparameters and trying regularization methods.• Regularization via data:-When not working with nearly infinite/abundant data: * Gathering more real data (and using methods that take its properties into account) is advisable if possible: · Labeled samples are best, but unlabeled ones can also be helpful (compatible with semi-supervised learning).

·

Samples from the same domain are best, but samples from similar domains can also be helpful (compatible with domain adaptation and transfer learning).

· Reliable high-quality samples are best, but lower-quality ones can also be helpful (their confidence/importance can be adjusted accordingly).

·

Labels for an additional task can be helpful (compatible with multi-task learning). ·

Additional input features (from additional information sources) and/or data preprocessing (i.e. domain-specific data transformations) can be helpful (the network architecture needs to be adjusted accordingly).

* Data augmentation (e.g. target-preserving handcrafted domain-specific transformations) can well compensate for limited data.

If natural ways to augment data (to mimic natural transformations sufficiently well) are known, they can be tried (and combined).

* If natural ways to augment data are unknown or turn out to be insufficient, it may be possible to infer the transformation from data (e.g. learning imagedeformation fields) if a sufficient amount of data is available for that.

-Popular generic methods (e.g. advanced variants of Dropout) often also help.• Architecture and regularization terms: -Knowledge about possible meaningful properties of the mapping can be used to e.g. hardwire invariances (to certain transformations) into the architecture, or be formulated as regularization terms.

-Popular methods may help as well (see TAB3 , but should be chosen to match the assumptions about the mapping (e.g. convolutional layers are fully appropriate only if local and shift-equivariant feature extraction on regular-grid data is desired).• Optimization:-Initialization: Even though pre-trained ready-made models greatly speed up prototyping, training from a good random initialization should also be considered.

-Optimizers: Trying a few different ones, including advanced ones (e.g. Nesterov momentum, Adam, ProxProp), may lead to improved results.

Correctly chosen parameters, such as learning rate, usually make a big difference.

Recommendations for developers of novel regularization methods Getting an overview and understanding the reasons for the success of the best methods is a great foundation.

Promising empty niches (certain combinations of taxonomy properties) exist that can be addressed.

The assumptions to be imposed upon the model can have a strong impact on most elements of the taxonomy.

Data augmentation is more expressive than loss terms (loss terms enforce properties only in infinitesimally small neighborhood of the training samples; data augmentation can use rich transformation parameter distributions).

Data and loss terms impose assumptions and invariances in a rather soft manner, and their influence can be tuned, whereas hardwiring the network architecture is a harsher way to impose assumptions.

Different assumptions and options to impose them have different advantages and disadvantages.

Future directions for data-based methods There are several promising directions that in our opinion require more investigation: Adaptive sampling of might lead to lower errors and shorter training times BID28 ) (in turn, shorter training times may additionally work as implicit regularization BID42 , see also Section 7).

Secondly, learning class-dependent transformations (i.e. ( | )) in our opinion might lead to more plausible samples.

Furthermore, the field of adversarial examples (and network robustness to them) is gaining increased attention after the recently sparked discussion on real-world adversarial examples and their robustness/invariance to transformations such as the change of camera position BID71 BID2 .

Countering strong adversarial examples may require better regularization techniques.

Summary In this work we proposed a broad definition of regularization for deep learning, identified five main elements of neural network training (data, architecture, error term, regularization term, optimization procedure), described regularization via each of them, including a further, finer taxonomy for each, and presented example methods from these subcategories.

Instead of attempting to explain referenced works in detail, we merely pinpointed their properties relevant to our categorization.

Our work demonstrates some links between existing methods.

Moreover, our systematic approach enables the discovery of new, improved regularization methods by combining the best properties of the existing ones.

Although our proposed taxonomy seems intuitive, there are some ambiguities: Certain methods have multiple interpretations matching various categories.

Viewed from the exterior, a neural network maps inputs to outputs .

We formulate this as = ( ( )) for transformations in input space (and similarly for hidden-feature space, where is applied in between layers of the network ).

However, how to split this -to-mapping into "the part" and "the part", and thus into Section 3 vs. Section 4, is ambiguous and up to one's taste and goals.

In our choices (marked with " " below), we attempt to use common notions and Occam's razor.• Ambiguity of attributing noise to , or to , or to data transformations :-Stochastic methods such as Stochastic depth BID54 can have several interpretations if stochastic transformations are allowed for or : Stochastic transformation of the architecture (randomly dropping some connections), TAB3 Stochastic transformation of the weights (setting some weights to 0 in a certain random pattern) Stochastic transformation of data in hidden-feature space; dependence is ( ), described in Table 1 for completeness• Ambiguity of splitting into and :-Dropout: Parameters are the dropout mask; dependence is ( ); transformation applies the dropout mask to the hidden features Parameters are the seed state of a pseudorandom number generator; dependence is ( ); transformation internally generates the random dropout mask from the random seed and applies it to the hidden features -Projecting dropout noise into input space (Bouthillier et al., 2015, Sec. 3) can fit our taxonomy in different ways by defining and accordingly.

It can have similar interpretations as Dropout above (if is generalized to allow for dependence on , , ), but we prefer the third interpretation without such generalizations:Parameters are the dropout mask (to be applied in a hidden layer); dependence is ( ); transformation transforms the input to mimic the effect of the mask Parameters are the seed state of a pseudorandom number generator; dependence is ( ); transformation internally generates the random dropout mask from the random seed and transforms the input to mimic the effect of the mask Parameters describe the transformation of the input in any formulation; dependence is ( | , , ); transformation merely applies the transformation in input space• Ambiguity of splitting the network operation into layers: There are several possibilities to represent a function (neural network) as a composition (or directed acyclic graph) of functions (layers).• Many of the input and hidden-feature transformations (Section 3) can be considered layers of the network (Section 4).

In fact, the term "layer" is not uncommon for Dropout or Batch normalization.• The usage of a trainable parameter in several parts of the network is called weight sharing.

However, some mappings can be expressed with two equivalent formulas such that a parameter appears only once in one formulation, and several times in the other.• Ambiguity of vs. : Auxiliary denoising task in ladder networks BID85 and similar autoencoder-style loss terms can be interpreted in different ways:Regularization term without given auxiliary targetsThe ideal reconstructions can be considered as targets (if the definition of "targets" is slightly modified) and thus the denoising task becomes part of the error term

To understand the success of target-preserving data augmentation methods, we consider the data-augmented loss function, which we obtain by replacing the training samples ( , ) ∈ in the empirical risk loss function (Eq. (3)) by augmented training samples ( ( ), ): DISPLAYFORM0

@highlight

Systematic categorization of regularization methods for deep learning, revealing their similarities.

@highlight

Attempts to build a taxonomy for regularization techniques employed in deep learning.