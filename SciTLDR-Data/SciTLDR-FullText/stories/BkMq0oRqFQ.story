Batch Normalization (BN) and its variants have seen widespread adoption in the deep learning community because they improve the training of deep neural networks.

Discussions of why this normalization works so well remain unsettled.

We make explicit the relationship between ordinary least squares and partial derivatives computed when back-propagating through BN.

We recast the back-propagation of BN as a least squares fit, which zero-centers and decorrelates partial derivatives from normalized activations.

This view, which we term {\em gradient-least-squares}, is an extensible and arithmetically accurate description of BN.

To further explore this perspective, we motivate, interpret, and evaluate two adjustments to BN.

Training deep neural networks has become central to many machine learning tasks in computer vision, speech, and many other application areas.

BID10 showed empirically that Batch Normalization (BN) enables deep networks to attain faster convergence and lower loss.

Reasons for the effectiveness of BN remain an open question BID12 .

Existing work towards explaining this have focused on covariate shift; Santurkar et al. (2018) described how BN makes the loss function smoother.

This work examines the details of the back-propagation of BN, and recasts it as a least squares fit.

This gradient regression zero-centers and decorrelates partial derivatives from the normalized activations; it passes on a scaled residual during back-propagation.

Our view provides novel insight into the effectiveness of BN and several existing alternative normalization approaches in the literature.

Foremost, we draw an unexpected connection between least squares and the gradient computation of BN.

This motivates a novel view that complements earlier investigations into why BN is so effective.

Our view is consistent with recent empirical surprises regarding ordering of layers within ResNet residual maps BID5 and within shake-shake regularization branches BID6 .

Finally, to demonstrate the extensibility of our view, we motivate and evaluate two variants of BN from the perspective of gradient-least-squares.

In the first variant, a least squares explanation motivates the serial chaining of BN and Layer Normalization (LN) BID0 .

In the second variant, regularization of the least-squares leads to a version of BN that performs better on batch size two.

In both variants, we provide empirical support on CIFAR-10.In summary, our work presents a view, which we term gradient-least-squares, through which the back-propagation of BN and related work in a neural network can be recast as least squares regression.

This regression decomposes gradients into an explained portion and a residual portion; BN back-propagation will be shown to remove the explained portion.

Hopefully, gradient-least-squares will be broadly useful in the future design and understanding of neural network components.

Figure 1 reviews normalization with batch statistics, and illustrates our main theorem.

Gradient-least-squares relates quantities shown in hexagons Figure 1 : The left figure reviews, for a single channel at a particular layer within a single batch, notable quantities computed during the forward pass and during back-propagation of BN.

Let DISPLAYFORM0 Let L be a function dependent on the normalized activations z i defined for each j by z j = (x j ??? ??) ?? This, along with partial derivatives, are shown in the left figure.

Our work establishes a novel identity on the quantities shown in hexagons.

The right figure illustrates our main result in a scatter plot, in which each pair z i , ???L ???z i is shown as a data point in the regression.

Consider any particular channel within which {x i } are activations to be normalized in BN moment calculations.

BID10 defined BN as DISPLAYFORM0 where ??, ?? are batch moments, but b and c are learned per-channel parameters persistent across batches.

In BN, the batch dimension and spatial dimensions are marginalized out in the computation of batch moments.

For clarity, we consider a simplified version of BN.

We ignore the variables b and c in equation 1 responsible for a downstream channel-wise affine transformation.

Ignoring b and c is done without loss of generality, since the main observation in this work will focus on the Gaussian normalization and remains agnostic to downstream computations.

We also ignore a numerical stability hyperparameter .We examine back-propagation of partial derivatives through this normalization, where ?? and ?? are viewed as functions of x. Notably, ?? and ?? are functions of each x i , and thus the division by ?? is not affine.

We write the normalized output as DISPLAYFORM1 We review ordinary least squares of a single variable with intercept BID2 .Let g j = ?? + ??z j + j where ?? and ?? are parameters, z and g are observations.

z j and g j are entries in z and g respectively.

j are i.i.d.

Gaussian residuals.

We wish to fit ?? and ?? ??,?? = arg min DISPLAYFORM2 The least-squares problem in equation 3 is satisfied by?? = Cov(z, g) DISPLAYFORM3 When z are normalized activations and g are partial derivatives, then Ez = 0 and Var(z) = 1.

In this special case, the solution simplifies int?? DISPLAYFORM4 Theorem 1 (Normalization gradients are least-squares residuals).

Let i ??? {1 . . .

N } be indices over some set of activations {x i }.

Then the moment statistics are defined by ?? = N i=1x i N and DISPLAYFORM5 Let L be a function dependent on the normalized activations z i defined for each j by z j = (x j ??? ??) ?? .

Then, the gradients of L satisfy, for all j ??? {1, . . .

, N }, the following: DISPLAYFORM6 where DISPLAYFORM7 Proof: Normalization gradients are least-squares residuals.

The proof involves a derivation of partial derivatives by repeated applications of the chain rule and rules of total derivative.

Because {z i } normalized over i has mean 0 and variance 1, the partial derivatives can be rearranged to satisfy the single variable ordinary least squares framework.

Fix j.

We expand ???L ???x j as a linear combination of DISPLAYFORM8 We state ???z i ???x j directly.

Steps are in Appendix A under Lemma 1.

DISPLAYFORM9 Through substitution of equations 10 into 9, we get DISPLAYFORM10 Noting that {z i } normalized over i has mean 0 and variance 1, we recover?? and??, in the sense of equations 4 and 5, from equation 13.

DISPLAYFORM11 Finally, we rearrange equations 15 and 14 into 13 to conclude, as desired, DISPLAYFORM12 During back-propagation of a single batch, the normalization function takes in partial derivatives ???L ???z (??) , and removes that which can be explained by least squares of ???L ???z (??) against z (??) .

As illustrated in Figure 1 , during back-propagation, the residual then divides away ?? to become ???L ???x (??) , the gradient for the unnormalized activations.

CALCULATIONS BN aims to control its output to have mean near 0 and variance near 1, normalized over the dataset; this is related to the original explanation termed internal covariate shift BID10 .

Most existing work that improve or re-purpose BN have focused on describing the distribution of activations.

Definition 1.

In the context of normalization layers inside a neural network, activations are split into partitions, within which means and variances are computed.

We refer to these partitions as normalization partitions.

Definition 2.

Within the context of a normalization partition, we refer to the moments calculated on the partitions as partition statistics.

Theorem 1 shows that BN has least squares fitting built into the gradient computation.

Gradients of the activations being normalized in each batch moment calculation are fit with a single-variable with-intercept least squares model, and only a rescaled residual is kept during back-propagation.

We emphasize that the data on which the regression is trained and applied is a subset of empirical activations within a batch, corresponding to the normalization partitions of BN.To show extensibility, we recast several popular normalization techniques into the gradient-leastsquares view.

We refer to activations arising from a single member of a particular batch as an item.

BHW C refers to dimensions corresponding to items, height, width, and channels respectively.

In non-image applications or fully connected layers, H and W are 1.

BN marginalizes out the items and spatial dimensions, but statistics for each channel are kept separate.

In the subsequent sections, we revisit several normalization methods from the perspective of the gradient.

FIG1 reviews the normalization partitions of these methods, and places our main theorem about gradient-least-squares into context.

BID0 introduced Layer Normalization (LN) in the context of large LSTM models and recurrent networks.

Only the (H, W, C) dimensions are marginalized in LN, whereas BN marginalizes out the (B, H, W ) dimensions.

In our regression framework, the distinction can be understood as changing the data point partitions in which least squares are fit during back-propagation.

LN marginalizes out the channels, but computes separate statistics for each batch item.

To summarize, the regression setup in the back-propagation of LN is performed against other channels, rather than against other batch items.

Huang & Belongie (2017) introduced Instance Normalization (IN) in the context of transferring styles across images.

IN is is closely related to contrast normalization, an older technique used in image processing.

IN emphasizes end-to-end training with derivatives passing through the moments.

Only the (H, W ) dimensions are marginalized in IN, whereas BN marginalizes (B, H, W ) dimensions.

In our framework, this can be understood as using fewer data points and a finer binning to fit the least squares during back-propagation, as each batch item now falls into its own normalization partition.

Wu & He (2018) introduced Group Normalization (GN) to improve performance on image-related tasks when memory constrains the batch size.

Similar to LN, GN also marginalizes out the (H, W, C) dimensions in the moment computations.

The partitions of GN are finer: the channels are grouped into disjoint sub-partitions, and the moments are computed for each sub-partition.

When the number of groups is one, GN reduces to LN.In future normalization methods that involve normalizing with respect to different normalization partitions; such methods can pattern match with BN, LN, IN, or GN; the back-propagation can be formulated as a least-squares fit, in which the partial derivatives at normalized activations ???L ???z (??) are fitted against the normalized z (??) , and then the residual of the fit is rescaled to become ???L ???x (??) .Figure 2 summarize the normalization partitions for BN, LN, IN, and GN; the figure visualizes, as an example, a one-to-one correspondence between an activation in BN, and a data point in the gradient regression.

Theorem 1 is agnostic to the precise nature of how activations are partitioned before being normalized; thus, equation 9 applies directly to any method that partitions activations and performs Gaussian normalization on each partition.

DISPLAYFORM0 DISPLAYFORM1 The L2 normalization of weights in WN appears distinct from the Gaussian normalization of activations in BN; nevertheless, WN can also be recast as a least squares regression.

BID4 and improved BID5 residual mappings in ResNets.

Arrows point in the direction of the forward pass.

Dotted lines indicate that gradients are zero-centered and decorrelated with respect to downstream activations in the residual mapping.

The improved ordering has BN coming first, and thus constrains that gradients of the residual map must be decorrelated with respect to some normalized activations inside the residual mapping.

An update to the popular ResNet architecture showed that the network's residual mappings can be dramatically improved with a new ordering BID5 .

The improvement moved BN operations into early positions and surprised the authors; we support the change from the perspective of gradient-least-squares.

FIG2 reviews the precise ordering in the two versions.

BID6 provides independent empirical support for the BN-early order, in shake-shake regularization BID3 architectures.

We believe that the surprise arises from a perspective that views BN only as a way to control the distribution of activations; one would place BN after a sequence of convolution layers.

In the gradient-least-squares perspective, the first layer of each residual mapping is also the final calculation for these gradients before they are added back into BID0 0.9102 0.3548 the main trunk.

The improved residual branch constrains the gradients returning from the residual mappings to be zero-centered and decorrelated with respect to some activations inside the branch.

We illustrate this idea in FIG2 .

Gradient-least-squares views back-propagation in deep neural networks as a solution to a regression problem.

Thus, formulations and ideas from a regression perspective would motivate improvements and alternatives to BN.

We pursue and evaluate two of these ideas.

BN and LN are similar to each other, but they normalize over different partitioning of the activations; in back-propagation, the regressions occur respectively with respect to different partitions of the activations.

Suppose that a BN and a LN layer are chained serially in either order.

This results in a two-step regression during back-propagation; in reversed order, the residual from the first regression is further explained by a second regression on a different partitioning.

In principle, whether this helps would depend on the empirical characteristics of the gradients encountered during training.

The second regression could further decorrelate partial gradients from activations.

Empirically, we show improvement in a reference ResNet-34-v2 implementation on CIFAR-10 relative to BN with batch size 128.

In all cases, only a single per-channel downstream affine transformation is applied, after both normalization layers, for consistency in the number of parameters.

See table 1 for CIFAR-10 validation performances.

We kept all default hyperparameters from the reference implementation: learning schedules, batch sizes, and optimizer settings.

BN performs less well on small batches BID9 .

Gradient-least-squares interprets this as gradient regressions failing on correlated data, an issue typically addressed by regularization.

We pursue this idea to recover some performance on small batches by use of regularization.

Our regularization uses streaming estimates of past gradients to create virtual data in the regression.

This performed better than standard BN on the same batch size, but we did not recover the performance of large batches; this is consistent with the idea that regularization could not in general compensate for having much less data.

See Appendix C for CIFAR-10 validation performances.

DISPLAYFORM0 to rescale the contributions to the batch mean for each normalization scheme.

It uses an analogous set of parameters ?? k and activations w k for variances.

We sketch the back-propagation of a simplified version of SN in the perspective of gradient-least-squares.

We ignore both the division and downstream affine z ??? c ?? z + b. The normalization calculation inside SwN can be written as: DISPLAYFORM1 where ??? = {BN, LN, IN }.

There is potentially a unique mean and variance used for each activation.

Equation 19 bears similarities to the setup in Theorem 1, but we leave unresolved whether there is a gradient-least-squares regression interpretation for SN.

Decorrelated Batch Normalization (DBN) ) is a generalization of BN that performs Mahalanobis ZCA whitening to decorrelate the channels, using differentiable operations.

On some level, the matrix gradient equation resemble the least squares formulation in Theorem 1.Spectral Normalization (SpN) BID15 is an approximate spectral generalization of WN.

For DBN and SpN, the regression interpretations remain unresolved.

BN has been instrumental in the training of deeper networks BID10 .

Subsequent work resulted in Batch Renormalization BID9 , and further emphasized the importance of passing gradients through the minibatch moments, instead of a gradient-free exponential running average.

In gradient-least-squares, use of running accumulators in the training forward pass would stop the gradients from flowing through them during training, and there would be no least-squares.

BID5 demonstrate empirically the unexpected advantages of placing BN early in residual mappings of ResNet.

Santurkar et al. FORMULA1 showed that BN makes the loss landscape smoother, and gradients more predictable across stochastic gradient descent steps.

BID1 found evidence that spatial correlation of gradients explains why ResNet outperforms earlier designs of deep neural networks.

BID11 proved that BN accelerates convergence on least squares loss, but did not consider back-propagation of BN as a least squares residual.

BID14 has recast BN as a stochastic process, resulting in a novel treatment of regularization.

This work makes explicit how BN back-propagation regresses partial derivatives against the normalized activations and keeps the residual.

This view, in conjunction with the empirical success of BN, suggests an interpretation of BN as a gradient regression calculation.

BN and its variants decorrelate and zero-center the gradients with respect to the normalized activations.

Subjectively, this can be viewed as removing systematic errors from the gradients.

Our view also support empirical results in literature preferring early BN placement within neural network branches.

Leveraging gradient-least-squares considerations, we ran two sets of normalization experiments, applicable to large batch and small batch settings.

Placing a LN layer either before or after BN can be viewed as two-step regression that better explains the residual.

We show empirically on CIFAR-10 that BN and LN together are better than either individually.

In a second set of experiments, we address BN's performance degradation with small batch size.

We regularize the gradient regression with streaming gradient statistics, which empirically recovers some performance on CIFAR-10 relative to basic BN, on batch size two.

Why do empirical improvements in neural networks with BN keep the gradient-least-squares residuals and drop the explained portion?

We propose two open approaches for investigating this in future work.

A first approach focuses on how changes to the gradient regression result in different formulations; the two empirical experiments in our work contribute to this.

A second approach examines the empirical relationships between gradients of activations evaluated on the same parameter values; we can search for a shared noisy component arising from gradients in the same normalization partition.

Suppose that the gradient noise correlates with the activations -this is plausible because the population of internal activations arise from using shared weights -then normalizations could be viewed as a layer that removes systematic noise during back-propagation.

In DISPLAYFORM0 Then, the partial derivatives satisfy DISPLAYFORM1 Proof.

In deriving ???z j ???x i , we will treat the cases of when j = i and when j = i separately.

We start by examining intermediate quantities of interest as a matter of convenience for later use.

We define helper quantities u i = x i ??? ??. Note that each u j depends on all of x i via ??. Next, we write out useful identities DISPLAYFORM2 We prepare to differentiate with rule of total derivative: DISPLAYFORM3 Making use of equations 21, 22, 23 and 25, We simplify ????? ???x i for any i as follows.

DISPLAYFORM4 We apply the quotient rule on ???z j ???x i when j = i, then substitute equation 33 DISPLAYFORM5 Similarly, when i = j, inputs in batch b. In our work, we keep track of am exponential running estimates across batches, DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 that marginalize the (B, H, W ) dimensions into accumulators of shape C. The b subscript of the outer expectation is slightly abusive notation indicating that?? * and?? * are running averages across recent batches with momentum as a hyperparameter that determines the weighting.

We regularize the gradient regression with virtual activations and virtual gradients, defined as follows.

We append two virtual batch items, broadcast to an appropriate shape, x + = ?? b + ?? b and x ??? = ?? b ??? ?? b .

Here, ?? b and ?? b are batch statistics of the real activations.

The concatenated tensor undergoes standard BN, which outputs the usual {z i } for the real activations, but z + = 1 and z ??? = ???1 for the virtual items.

The z + and z ??? do not affect the feed forward calculations, but they receive virtual gradients during back-propagation: DISPLAYFORM9 Virtual data z + , ???L ???z + and z ??? , ???L ???z ??? regularizes the gradient-least-squares regression.

???L ???z + and ???L ???z ??? eventually modify the gradients received by the real x i activations.

The virtual data can be weighted with hyperparameters.

In our experiments, we see improvements, robust to a hyperparameter cross-product search over the weightings and the momentum for?? * and?? * .

The momentum for?? * and?? * were in {.997, .5} and the virtual item weights were in {2 i???1 } i???{0,1,2,3} .

The performance of larger batches are not recovered; regularized regression could not be reasonably expected to recover the performance of regressing with more data.

See table 2 for final validation performances with a reference Tensorflow ResNet-34-v2 implementation on batch size of two.

The baseline evaluation with identity (no normalization) experienced noticeable overfitting in terms of cross entropy but not accuracy.

The base learning rate was multiplied by 1 64 relative to the baseline rate used in runs with batch size 128.

@highlight

Gaussian normalization performs a least-squares fit during back-propagation, which zero-centers and decorrelates partial derivatives from normalized activations.