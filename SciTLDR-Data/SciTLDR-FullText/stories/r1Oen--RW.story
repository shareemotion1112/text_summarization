Saliency methods aim to explain the predictions of deep neural networks.

These methods lack reliability when the explanation is sensitive to factors that do not contribute to the model prediction.

We use a simple and common pre-processing step ---adding a mean shift to the input data--- to show that a transformation with no effect on the model can cause numerous methods to incorrectly attribute.

We define input invariance as the requirement that a saliency method mirror the sensitivity of the model with respect to transformations of the input.

We show, through several examples, that saliency methods that do not satisfy a input invariance property are unreliable and can lead to misleading and inaccurate attribution.

While considerable research has focused on discerning the decision process of neural networks BID1 BID9 BID2 BID15 BID12 BID0 BID14 BID6 BID5 BID16 BID13 BID11 BID4 , there remains a trade-off between model complexity and interpretability.

Research to address this tension is urgently needed; reliable explanations build trust with users, helps identify points of model failure, and removes barriers to entry for the deployment of deep neural networks in domains high stakes like health care and others.

In deep models, data representation is delegated to the model.

We cannot generally say in an informative way what led to a model prediction.

Instead, saliency methods aim to infer insights about the f (x) learnt by the model by ranking the explanatory power of constituent inputs.

While unified in purpose, these methods are surprisingly divergent and non-overlapping in outcome.

Evaluating the reliability of these methods is complicated by a lack of ground truth, as ground truth would depend upon full transparency into how a model arrives at a decision -the very problem we are trying to solve for in the first place BID13 BID4 .Given the need for a quantitative method of comparison, several properties such as completeness BID0 BID13 , implementation invariance and sensitivity BID13 have been articulated as desirable to ensure that saliency methods are reliable.

Implementation invariance, proposed as an axiom for attribution methods by BID13 , is the requirement that functionally equivalent networks (models with different architectures but equal outputs all inputs), always attribute in an identical way.

This work posits that a second invariance axiom, which we term input invariance, needs to be satisfied to ensure reliable interpretation of input contribution to the model prediction.

Input invariance requires that the saliency method mirror the sensitivity of the model with respect to transformations of the input.

We demonstrate that numerous methods do not satisfy input invariance using a simple transformation -mean shifts of the input-that does not affect model prediction or weights.

We limit our treatment of input invariance to showing that there exist cases where this property is not satisfied and welcome future research on a broader treatment of this topic.

In this work we:• introduce the axiom input invariance and demonstrate that certain saliency methods do not satisfy this property when considering a simple mean shift in the input. (See FIG3 ).• show that when input invariance is missing, the saliency method becomes unreliable and misleading.

Using two example reference points for each method we demonstrate that changing the reference causes the attribution to diverge.

The attributions are visualized by multiplying them with the input image as is done in the IG paper 1 BID13 .

Visualisations were made on ImageNet BID7 and the VGG16 architecture BID9 .• demonstrate that "reference point" methods-Integrated gradients and the Deep Taylor Decomposition-have diverging attribution and input invariance breaking points that depends upon the choice of reference FIG0 .In Section 2, we detail our experiment framework.

In Section 3, we determine that while the model is invariant to the input transformation considered, several saliency methods attribute to the mean shift.

In Section 4 we discuss "reference point" methods and illustrate the importance of choosing an appropriate reference before discussing some directions of future research in Section 5.

We show that, by construction, the bias of a neural network compensates for the mean shift resulting in two networks with identical weights and predictions.

We first demonstrate this point and then describe the details of our experiment setup to evaluate the input invariance of a set of saliency methods.

We compare the attribution across two networks, f 1 (x) and f 2 (x).

f 1 (x) is a network trained on input x i 1 that denotes sample i from training set X 1 .

The classification task of network 1 is: DISPLAYFORM0 is a network that predicts the classification of a transformed input x ∀i, DISPLAYFORM1 Network 1 and 2 differ only by construction.

Consider the first layer neuron before non-linearity in DISPLAYFORM2 We alter the biases in the first layer neuron by adding the mean shift m 2 .

This now becomes Network 2: DISPLAYFORM3 As a result the first layer activations are the same for f 1 (x) and f 2 (x): DISPLAYFORM4 Note that the gradient with respect to the input remains unchanged as well: DISPLAYFORM5 We have shown that Network 2 cancels out the mean shift transformation.

This means that f 1 (x) and f 2 (x) have identical weights and produce the same output for the corresponding samples, DISPLAYFORM6

In the implementation of this experimental framework, Network 1 is a 3 layer multi-layer perceptron with 1024 ReLu-activated neurons each.

Network 1 classifies MNIST image inputs in a [0,1] encoding.

Network 2 classifies MNIST image inputs in a [-1,0] MNIST encoding.

The first network is trained for 10 epochs using mini-batch stochastic gradient descent (SGD).

The second network is created using the approach above.

The final accuracy is 98.3% for both 3 .

In 3.1 we introduce key approaches to the classification of inputs as salient and the saliency methods we evaluate.

In 3.2 we find that gradient and signal methods are input invariant.

In 3.3 we find that most attribution methods considered have points where they start to break down.

Most saliency research to date has centered on convolutional neural networks.

These saliency methods broadly fall into three different categories:1.

Gradients (Sensitivity) BID1 BID10 ) shows how a small change to the input affects the classification score for the output of interest.2.

Signal methods such as DeConvNet BID15 , Guided BackProp BID12 and PatternNet BID4 aim to isolate input patterns that stimulate neuron activation in higher layers.3.

Attribution methods such as Deep-Taylor Decomposition BID5 and Integrated Gradients BID13 ) assign importance to input dimensions by decomposing the value y j at an output neuron j into contributions from the individual input dimensions: DISPLAYFORM0 s j is the decomposition into input contributions and has the same number of dimensions as x, A(x) j signifies the attribution method applied to output j for sample x. Attribution methods are distinguished from gradients by the insistence on completeness: the sum of all attributions should be approximately equal to the original output y i .

We consider the input invariance of each category separately (by evaluating raw gradients, GuidedBackprop, Integrated Gradients and Deep Taylor Decomposition) and also benchmark the input invariance of SmoothGrad ( BID11 ), a method that wraps around an underlying saliency approach and uses the addition of noise to produce a sharper visualization of the saliency heatmap.

The experiment setup and methodology is as described in Section 2.

Each method is evaluated by comparing the saliency heatmaps for the predictions of network 1 and 2, where x i 2 is simply the mean shifted input (x i 1 + m 2 ).

A saliency method that is not input invariant will not produce identical saliency heatmap for Network 1 and 2 despite the mean shift of the input.

Sensitivity and signal methods are not sensitive to the mean shift in inputs.

In FIG2 raw gradients, PatternNet (PN, BID4 ) and Guided Backprop (GB, Springenberg et al. (2015) ) produce identical saliency heatmaps for both networks.

Intuitively, gradient, PN and GB are input invariant given that we are comparing two networks with an identical f (x).

Both methods determine attribution entirely as a function of the network/pattern weights and thus will be input invariant as long as we are comparing networks with identical weights.

In the same manner, we can say that these methods will not be input invariant when comparing networks with different weights (even if we consider models with different architectures but identical predictions for every input).

We evaluate the following attribution methods: gradient times input (GI), integrated gradients (IG, BID13 ) and the deep-taylor decomposition (DTD, BID5 ).In 3.3.1 we find GI to be sensitive to meaningless input shifts.

In 3.3.2 we group discussion of IG and DTD under "reference point" methods because both require that attribution is done in reference to a defined point.

We find that the choice of reference point can cause input invariance to become arbitrary.

We find that the multiplication of raw gradients by the image breaks attribution reliability.

In FIG3 GI produces different saliency heatmaps for both networks. .

Gradient x Input, IG and DTD with a zero reference point, which is equivalent to LRP BID0 BID5 , are not reliable and produce different attribution for each network.

IG with a black image reference point and DTD with a PA reference point are not sensitive to the transformation of the input.

In 3.2 we determined that a heatmap of gradients alone is not sensitive to the input transformation.

GI multiplies the gradient w.r.t.

the input with the input image.

DISPLAYFORM0 Multiplying by the input means attribution is no longer reliable because the input shift is carried through to final attribution.

Naive multiplication by the input, as noted by BID11 , also constrains attribution without justification to inputs that are not 0.

Both Integrated Gradients (IG, BID13 ) and Deep Taylor Decomposition (DTD, BID5 ) determine the importance of inputs relative to a reference point.

DTD refers to this as the root point and IG terms the reference point a baseline.

The choice of reference point is not determined a priori by the method and instead left to end user.

The choice of reference point determines all subsequent attribution.

In FIG0 IG and DTD show different attribution depending on the choice of reference point.

We show that the certain reference point also cause IG and DTD to are not input invariant.

Integrated gradients (IG) Integrated Gradients (IG, BID13 ) attribute the predicted score to each input with respect to a baseline x 0 .

This is achieved by constructing a set of inputs interpolating between the baseline and the input.

DISPLAYFORM0 Since this integral cannot be computed analytically, it is approximated by a finite sum ranging over α ∈ [0, 1].

DISPLAYFORM1 We evaluate whether two possible IG reference points satisfy input invariance.

Firstly, we consider an image populated uniformly with the minimum pixel from the dataset (x 0 = min(x)) (black image) and a zero vector image.

We find that IG attribution under certain reference points is not input invariant.

In FIG3 , IG with black reference point produces identical attribution heatmaps whereas IG with a zero vector reference point is not input invariant.

IG using a black reference point is not sensitive to the mean input shift because x 0 = min(x) is determined after the mean shift of the input so the difference between x and x 0 remains the same for both networks.

In network 1 this is (x 1 ) − min(x 1 ) and in network 2 this is (x 2 + m 2 ) − min(x 2 + m 2 ).IG with a zero vector reference point is not input invariant because while the difference in network 1 is (x 1 − x 0 ), the difference in network 2 becomes (x 2 + m 2 ) − x 0 .Deep Taylor Decomposition (DTD) determines attribution relative to a reference point neuron.

DTD can satisfy input invariant if the right reference point is chosen.

In the general formulation, the attribution of an input neuron j is initialized to be equal to the output of that neuron.

The attribution of other output neurons is set to zero.

This attribution is backpropagated to its input neurons using the following distribution rule where s l j is the attribution assigned to neuron j in layer l: DISPLAYFORM2 We evaluate the input invariance of DTD using a reference point determined by Layer-wise Relevance Propagation (LRP) and PatternAttribution (PA).

In FIG3 , DTD satisfies input invariance when using a reference point defined by PA however it loses reliability when using a reference point defined by LRP.Layer-wise Relevance Propagation (LRP, BID0 ) is sensitive to the input shift because it is a case of DTD where a zero vector is chosen as the root point.

2 .

The back-propagation rule becomes: DISPLAYFORM3 s l−1,j depends only upon the input and so attribution will change between network 1 and 2 because x 1 andx 2 differ by a constant vector.

PatternAttribution (PA) satisfies input invariance because the reference point x 0 is defined as the natural direction of variation in the data BID4 .

The natural direction of the data is determined based upon covariances and thus compensates explicitly for the mean in the data.

Therefore it is by construction input invariant.

The PA root point is: DISPLAYFORM4 where DISPLAYFORM5 In a linear model: DISPLAYFORM6 For neurons followed by a ReLu non-linearity the vector a accounts for the non-linearity and is computed as: DISPLAYFORM7 .Here E + denotes the expectation taken over values where y is positive.

Figure 4 : Smoothgrad inherits the invariance properties of the underlying attribution method.

SG is not sensitive to the input transformation for gradient and signal methods (SG-PA and and SG-GB).

SG lacks input invariance for integrated gradients and deep taylor decomposition when a zero vector refernce point is used, but is not sensitive when PatternAttribution (SG-PA) or a black image (SG-Black) are used.

SG is not input invariant for gradient x input.

PA reduces to the following step: DISPLAYFORM8 The vector a depends upon covariances and thus removes the mean shift of the input.

The attribution for both networks is identical.

SmoothGrad (SG, Smilkov et al. (2017) ) replaces the input with N identical versions of the input with added random noise.

These noisy inputs are injected into the underlying attribution method and final attribution is the average attribution across N. For example, if the underlying methods are gradients w.r.t.

the input.

g(x) j = ∂f (x)j ∂x SG becomes: DISPLAYFORM0 SG often results in aesthetically sharper visualizations when applied to multi-layer neural networks with non-linearities.

SG does not alter the attribution method itself so will always inherit the input Figure 5 : Evaluation of attribution method sensitivity using MNIST.

Gradient x Input, IG with both a black and zero reference point and DTD with a LRP reference point, do not satisfy input invariance and produce different attribution for each network.

DTD with a PA reference point are not sensitive to the transformation of the input.invariance of the underlying method.

In Fig. 4 applying SG on top of gradients and signal methods (PA and GB) produces identical saliency maps.

SG is not input invariant when applied to gradient x input, LRP and zero vector reference points which compares SG heatmaps generated for all methods discussed so far.

SG is not sensitive to the input transformation when applied to PA and a black image.

IG and DTD do not satisfy input invariance under certain reference points.

The reference point determines subsequent attribution.

In FIG0 attribution visually diverges for the same method if multiple reference points are considered.

A reasonable reference point for IG and DTD will naturally depend upon domain and task.

Unintentional misrepresentation of the model is very possible when the choice of vantage point can lead to very different results.

Thus far, we have discussed attribution for image recognition tasks with the assumption that preprocessing steps are known and visual inspection of the points determined to be salient is possible.

For Audio and Language based models where input interaction is more intricate, attribution becomes even more challenging.

If we cannot determine the implications of reference point choice, we are limited in our ability to say anything about the reliability of the method.

To demonstrate this point, we construct a constant shift of the input that takes advantage of attribution points of failure discussed thus far.

Almost all methods are sensitive to this input transformation which results in a misleading explanation of the model prediction.

Network 1 is the same as introduced in Section 2.

We consider a transformation x ∀i, DISPLAYFORM0 Network 2 is identical to network 1 by construction (see Section 2).

Note that x In Fig. 5 all attribution methods except for PA are sensitive to this constant shift.

The result is that we are able to manipulate the attribution heatmap of an MNIST prediction so that the chosen samplê x appears.

Using a black image as a reference point for IG no longer satisfies input invariance (as it did in the experiments in Section 3).The samplex can be any abitrary vector.

We conduct the same experiment with a hand drawn kitten image.

We construct m 2 by choosing a desired attributionŝ that should be assigned to a specific samplex when the gradient is multiplied with the input.

We compute a m 2 that will ensure the specific x i 2 receives the desired attribution as follows: DISPLAYFORM1 To make sure that the original image is still recognizable as belonging to its class, we clip the shift to be within [-.3,.3] .

Of course, the attributions of the other samples in the dataset is impacted too.

In FIG6 we see that we are again able to purposefully misrepresent the explanation of the model prediction.

It is important to note that that some of these methods would have satisfied input invariance if the data had been normalized prior to attribution.

For example, IG with a black baseline will satisfy input invariance if the data is always normalized.

However, this is far from a systematic treatment of the reference point selection and there are cases outside of our experiment scope where this would not be sufficient.

We believe an open research question is furthering the understanding of reference point choice that guarantee reliability without relying on case-by-case solutions.

Saliency methods are powerful tools to gain intuition about our model.

We consider some examples that can cause a break in the reliability of these methods.

We show that we are able to purposefully create a deceptive explanation of the network using a hand drawn kitten image.

We introduce input invariance as a prerequisite for reliable attribution.

Our treatment of input invariance is restricted to demonstrating there is at least one input transformation that causes attribution to fail.

We hope this work drives further discussion on this subject.

We also acknowledge that saliency methods may still provide intuition for image recognition tasks even if they are not input invariant.

Our work is motivated in part because while we can visually inspect for catasthropic attribution failure in images, other modalities (like audio or word vectors) are more opaque and prone to unintentional misrepresentation.

@highlight

Attribution can sometimes be misleading