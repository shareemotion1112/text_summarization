Understanding the flow of information in Deep Neural Networks (DNNs) is a challenging problem that has gain increasing attention over the last few years.

While several methods have been proposed to explain network predictions, there have been only a few attempts to compare them from a theoretical perspective.

What is more, no exhaustive empirical comparison has been performed in the past.

In this work we analyze four gradient-based attribution methods and formally prove conditions of equivalence and approximation between them.

By reformulating two of these methods, we construct a unified framework which enables a direct comparison, as well as an easier implementation.

Finally, we propose a novel evaluation metric, called Sensitivity-n and test the gradient-based attribution methods alongside with a simple perturbation-based attribution method on several datasets in the domains of image and text classification, using various network architectures.

While DNNs have had a large impact on a variety of different tasks BID10 BID8 BID12 BID21 BID28 , explaining their predictions is still challenging.

The lack of tools to inspect the behavior of these black-box models makes DNNs less trustable for those domains where interpretability and reliability are crucial, like autonomous driving, medical applications and finance.

In this work, we study the problem of assigning an attribution value, sometimes also called "relevance" or "contribution", to each input feature of a network.

More formally, consider a DNN that takes an input x = [x 1 , ..., x N ] ∈ R N and produces an output S(x) = [S 1 (x), ..., S C (x)], where C is the total number of output neurons.

Given a specific target neuron c, the goal of an attribution method is to determine the contribution R c = [R c 1 , ..., R c N ] ∈ R N of each input feature x i to the output S c .

For a classification task, the target neuron of interest is usually the output neuron associated with the correct class for a given sample.

When the attributions of all input features are arranged together to have the same shape of the input sample we talk about attribution maps FIG0 , which are usually displayed as heatmaps where red color indicates features that contribute positively to the activation of the target output, and blue color indicates features that have a suppressing effect on it.

The problem of finding attributions for deep networks has been tackled in several previous works BID22 BID30 BID24 BID2 BID20 BID25 BID31 .

Unfortunately, due to slightly different problem formulations, lack of compatibility with the variety of existing DNN architectures and no common benchmark, a comprehensive comparison is not available.

Various new attribution methods have been published in the last few years but we believe a better theoretical understanding of their properties is fundamental.

The contribution of this work is twofold:1.

We prove that -LRP BID2 and DeepLIFT (Rescale) BID20 can be reformulated as computing backpropagation for a modified gradient function (Section 3).

This allows the construction of a unified framework that comprises several gradient-based attribution methods, which reveals how these methods are strongly related, if not equivalent under certain conditions.

We also show how this formulation enables a more convenient implementation with modern graph computational libraries.2.

We introduce the definition of Sensitivity-n, which generalizes the properties of Completeness BID25 and Summation to Delta BID20 and we compare several methods against this metric on widely adopted datasets and architectures.

We show how empirical results support our theoretical findings and propose directions for the usage of the attribution methods analyzed (Section 4).

Perturbation-based methods directly compute the attribution of an input feature (or set of features) by removing, masking or altering them, and running a forward pass on the new input, measuring the difference with the original output.

This technique has been applied to Convolutional Neural Networks (CNNs) in the domain of image classification BID30 , visualizing the probability of the correct class as a function of the position of a grey patch occluding part of the image.

While perturbation-based methods allow a direct estimation of the marginal effect of a feature, they tend to be very slow as the number of features to test grows (ie. up to hours for a single image BID31 ).

What is more, given the nonlinear nature of DNNs, the result is strongly influenced by the number of features that are removed altogether at each iteration ( Figure 1 ).In the remainder of the paper, we will consider the occluding method by BID30 as a comparison benchmark for perturbation-based methods.

We will use this method, referred to as Occlusion-1, replacing one feature x i at the time with a zero baseline and measuring the effect of this perturbation on the target output, ie.

DISPLAYFORM0 ) where we use x [xi=v] to indicate a sample x ∈ R N whose i-th component has been replaced with v. The choice of zero as a baseline is consistent with the related literature and further discussed in Appendix B.Original (label: "garter snake") Occlusion-1 Occlusion-5x5 Occlusion-10x10 Occlusion-15x15Figure 1: Attributions generated by occluding portions of the input image with squared grey patches of different sizes.

Notice how the size of the patches influence the result, with focus on the main subject only when using bigger patches.

Backpropagation-based methods compute the attributions for all input features in a single forward and backward pass through the network 1 .

While these methods are generally faster then perturbationbased methods, their outcome can hardly be directly related to a variation of the output.

Gradient * Input BID19 was at first proposed as a technique to improve the sharpness of the attribution maps.

The attribution is computed taking the (signed) partial derivatives of the output with respect to the input and multiplying them with the input itself.

Refer to TAB0 for the mathematical definition.

Integrated Gradients BID25 , similarly to Gradient * Input, computes the partial derivatives of the output with respect to each input feature.

However, while Gradient * Input computes a single derivative, evaluated at the provided input x, Integrated Gradients computes the average gradient while the input varies along a linear path from a baselinex to x. The baseline is defined by the user and often chosen to be zero.

We report the mathematical definition in TAB0 .Integrated Gradients satisfies a notable property: the attributions sum up to the target output minus the target output evaluated at the baseline.

Mathematically, DISPLAYFORM0 In related literature, this property has been variously called Completeness BID25 , Summation to Delta BID20 or Efficiency in the context of cooperative game theory BID15 , and often recognized as desirable for attribution methods.

Layer-wise Relevance Propagation (LRP) BID2 is computed with a backward pass on the network.

Let us consider a quantity r (l) i , called "relevance" of unit i of layer l.

The algorithm starts at the output layer L and assigns the relevance of the target neuron c equal to the output of the neuron itself and the relevance of all other neurons to zero (Eq. 1).The algorithm proceeds layer by layer, redistributing the prediction score S i until the input layer is reached.

One recursive rule for the redistribution of a layer's relevance to the following layer is the -rule described in Eq. 2, where we defined z ji = w DISPLAYFORM1 i to be the weighted activation of a neuron i onto neuron j in the next layer and b j the additive bias of unit j. A small quantity is added to the denominator of Equation 2 to avoid numerical instabilities.

Once reached the input layer, the final attributions are defined as DISPLAYFORM2 LRP together with the propagation rule described in Eq. 2 is called -LRP, analyzed in the remainder of this paper.

There exist alternative stabilizing methods described in BID2 and which we do not consider here.

DeepLIFT BID20 proceeds in a backward fashion, similarly to LRP.

Each unit i is assigned an attribution that represents the relative effect of the unit activated at the original network input x compared to the activation at some reference inputx (Eq. 3).

Reference valuesz ji for all hidden units are determined running a forward pass through the network, using the baselinex as input, and recording the activation of each unit.

As in LRP, the baseline is often chosen to be zero.

The relevance propagation is described in Eq. 4.

The attributions at the input layer are defined as DISPLAYFORM3 In Equation 4,z ji = w DISPLAYFORM4 i is the weighted activation of a neuron i onto neuron j when the baselinex is fed into the network.

As for Integrated Gradients, DeepLIFT was designed to satisfy Completeness.

The rule described in Eq. 4 ("Rescale rule") is used in the original formulation of the method and it is the one we will analyze in the remainder of the paper.

The "Reveal-Cancel" rule BID20 is not considered here. :

Attribution generated by applying several attribution methods to an Inception V3 network for natural image classification BID27 .

Notice how all gradient-based methods produce attributions affected by higher local variance compared to perturbation-based methods (Figure 1 ).Other back-propagation methods exist.

Saliency maps BID22 constructs attributions by taking the absolute value of the partial derivative of the target output S c with respect to the input features x i .

Intuitively, the absolute value of the gradient indicates those input features (pixels, for image classification) that can be perturbed the least in order for the target output to change the most.

However, the absolute value prevents the detection of positive and negative evidence that might be present in the input, reason for which this method will not be used for comparison in the remainder of the paper.

Similarly, Deep Taylor Decomposition , although showed to produce sparser explanations, assumes no negative evidence in the input and produces only positive attribution maps.

We show in Section 4 that this assumption does not hold for our tasks.

Other methods that are designed only for specific architectures (ie.

3 A UNIFIED FRAMEWORK Gradient * Input and Integrated Gradients are, by definition, computed as a function of the partial derivatives of the target output with respect to each input feature.

In this section, we will show that -LRP and DeepLIFT can also be computed by applying the chain rule for gradients, if the instant gradient at each nonlinearity is replaced with a function that depends on the method.

In a DNN where each layer performs a linear transformation z j = i w ji x i + b j followed by a nonlinear mapping x j = f (z j ), a path connecting any two units consists of a sequence of such operations.

The chain rule along a single path is therefore the product of the partial derivatives of all linear and nonlinear transformations along the path.

For two units i and j in subsequent layers we have ∂x j /∂x i = w ji · f (z j ), whereas for any two generic units i and c connected by a set of paths P ic the partial derivative is sum of the product of all weights w p and all derivatives of the nonlinearities f (z) p along each path p ∈ P ic .

We introduce a notation to indicate a modified chain-rule, where the derivative of the nonlinearities f () is replaced by a generic function g(): DISPLAYFORM5 When g() = f () this is the definition of partial derivative of the output of unit c with respect to unit i, computed as the sum of contributions over all paths connecting the two units.

Given that a zero weight can be used for non-existing or blocked paths, this is valid for any architecture that involves fully-connected, convolutional or recurrent layers without multiplicative units, as well as for pooling operations.

Proposition 1.

-LRP is equivalent the feature-wise product of the input and the modified partial derivative DISPLAYFORM6 e. the ratio between the output and the input at each nonlinearity.

Proposition 2.

DeepLIFT (Rescale) is equivalent to the feature-wise product of the x −x and the modified partial derivative DISPLAYFORM7 the ratio between the difference in output and the difference in input at each nonlinearity, for a network provided with some input x and some baseline inputx defined by the user.

The proof for Proposition 1 and 2 are provided in Appendix A.1 and Appendix A.2 respectively.

Given these results, we can write all methods with a consistent notation.

TAB0 summaries the four methods considered and shows examples of attribution maps generated by these methods on MNIST.

As pointed out by BID25 a desirable property for attribution methods is their immediate applicability to existing models.

Our formulation makes this possible for -LRP and DeepLIFT.

Since all modern frameworks for graph computation, like the popular TensorFlow BID0 , implement backpropagation for efficient computation of the chain rule, it is possible to implement all methods above by the gradient of the graph nonlinearities, with no need to implement custom layers or operations.

Listing 1 shows an example of how to achieve this on Tensorflow.

Listing 1: Example of gradient override for a Tensorflow operation.

After registering this function as the gradient for nonlinear activation functions, a call to tf.gradients() and the multiplication with the input will produce the -LRP attributions.

DISPLAYFORM8

The formulation of TAB0 facilitates the comparison between these methods.

Motivated by the fact that attribution maps for different gradient-based methods look surprisingly similar on several tasks, we investigate some conditions of equivalence or approximation.

Proposition 3.

-LRP is equivalent to i) Gradient * Input if only Rectified Linear Units (ReLUs) are used as nonlinearities; ii) DeepLIFT (computed with a zero baseline) if applied to a network with no additive biases and with nonlinearities f such that f (0) = 0 (eg.

ReLU or Tanh).The first part of Proposition 3 comes directly as a corollary of Proposition 1 by noticing that for ReLUs the gradient at the nonlinearity f is equal to g LRP for all inputs.

This relation has been previously proven by BID19 and BID5 .

Similarly, we notice that, in a network with no additive biases and nonlinearities that cross the origin, the propagation of the baseline produces a zero reference value for all hidden units (ie.

∀i : DISPLAYFORM0 DL , which proves the second part of the proposition.

Notice that g LRP (z) = (f (z) − 0)/(z − 0) which, in the case of ReLU and Tanh, is the average gradient of the nonlinearity in [0, z] .

It also easy to see that lim z→0 g LRP (z) = f (0), which explain why g can not assume arbitrarily large values as z → 0, even without stabilizers.

On the contrary, if the discussed condition on the nonlinearity is not satisfied, for example with Sigmoid or Softplus, we found empirically that -LRP fails to produce meaningful attributions as shown in the empirical comparison of Section 4.

We speculate this is due to the fact g LRP (z) can become extremely large for small values of z, being its upper-bound only limited by the stabilizer.

This causes attribution values to concentrate on a few features as shown in TAB0 .

Notice also that the interpretation of g LRP as average gradient of the nonlinearity does not hold in this case, which explains why -LRP diverges from other methods 2 .DeepLIFT and Integrated Gradients are related as well.

While Integrated Gradients computes the average partial derivative of each feature as the input varies from a baseline to its final value, DeepLIFT approximates this quantity in a single step by replacing the gradient at each nonlinearity with its average gradient.

Although the chain rule does not hold in general for average gradients, we show empirically in Section 4 that DeepLIFT is most often a good approximation of Integrated Gradients.

This holds for various tasks, especially when employing simple models (see FIG5 .

However, we found that DeepLIFT diverges from Integrated Gradients and fails to produce meaningful results when applied to Recurrent Neural Networks (RNNs) with multiplicative interactions (eg.

gates in LSTM units BID4 ).

With multiplicative interactions, DeepLIFT does not satisfy Completeness, which can be illustrated with a simple example.

Take two variables x 1 and x 2 and a the function h(x 1 , x 2 ) = ReLU (x 1 − 1) · ReLU (x 2 ).

It can be easily shown that, by applying the methods as described by TAB0 , DeepLIFT does not satisfy Completeness, one of its fundamental design properties, while Integrated gradients does.

The formulation in TAB0 highlights how all the gradient-based methods considered are computed from a quantity that depends on the weights and the architecture of the model, multiplied by the input itself.

Similarly, Occlusion-1 can also be interpreted as the input multiplied by the average value of the partial derivatives, computed varying one feature at the time between zero and their final value: DISPLAYFORM0 The reason justifying the multiplication with the input has been only partially discussed in previous literature BID23 BID25 BID19 .

In many cases, it contributes to making attribution maps sharper although it remains unclear how much of this can be attributed to the sharpness of the original image itself.

We argue the multiplication with the input has a more fundamental justification, which allows to distinguish attribution methods in two broad categories: global attribution methods, that describe the marginal effect of a feature on the output with respect to a baseline and; local attribution methods, that describe how the output of the network changes for infinitesimally small perturbations around the original input.

For a concrete example, we will consider the linear case.

Imagine a linear model to predict the total capital in ten years C, based on two investments x 1 and x 2 : C = 1.05 · x 1 + 10 · x 2 .

Given this simple model, R 1 = ∂C/∂x 1 = 1.05, R 2 = ∂C/∂x 2 = 10 represents a possible local attribution.

With no information about the actual value of x 1 and x 2 we can still answer the question "

Where should one invest in order to generate more capital?.

The local attributions reveal, in fact, that by investing x 2 we will get about ten times more return than investing in x 1 .

Notice, however, that this does not tell anything about the contribution to the total capital for a specific scenario.

Assume x 1 = 100 000$ and x 2 = 1 000$. In this scenario C = 115000$.

We might ask ourselves "How the initial investments contributed to the final capital?

".

In this case, we are looking for a global attribution.

The most natural solution would be R 1 = 1.05x 1 = 105 000$, R 2 = 10x 2 = 1 000$, assuming a zero baseline.

In this case the attribution for x 1 is larger than that for x 2 , an opposite rank with respect to the results of the local model.

Notice that we used nothing but Gradient * Input as global attribution method which, in the linear case, is equivalent to all other methods analyzed above.

The methods listed in TAB0 are examples of global attribution methods.

Although local attribution methods are not further discussed here, we can mention Saliency maps BID22 as an example.

In fact, showed that Saliency maps can be seen as the first-order term of a Taylor decomposition of the function implemented by the network, computed at a point infinitesimally close to the actual input.

Finally, we notice that global and local attributions accomplish two different tasks, that only converge when the model is linear.

Local attributions aim to explain how the input should be changed in order to obtain a desired variation on the output.

One practical application is the generation of adversarial perturbations, where genuine input samples are minimally perturbed to cause a disruptive change in the output BID26 .

On the contrary, global attributions should be used to identify the marginal effect that the presence of a feature has on the output, which is usually desirable from an explanation method.

Attributions methods are hard to evaluate empirically because it is difficult to distinguish errors of the model from errors of the attribution method explaining the model BID25 .

For this reason the final evaluation is often qualitative, based on the inspection of the produced attribution maps.

We argue, however, that this introduces a strong bias in the evaluation: as humans, one would judge more favorably methods that produce explanations closer to his own expectations, at the cost of penalizing those methods that might more closely reflect the network behavior.

In order to develop better quantitative tools for the evaluation of attribution methods, we first need to define the goal that an ideal attribution method should achieve, as different methods might be suitable for different tasks (Subsection 3.2).Consider the attribution maps on MNIST produced by a CNN that uses Sigmoid nonlinearities ( FIG3 .

Integrated Gradients assigns high attributions to the background space in the middle of the image, while Occlusion-1 does not.

One might be tempted to declare Integrated Gradients a better attribution method, given that the heatmap is less scattered and that the absence of strokes in the middle of the image might be considered a good clue in favor of a zero digit.

In order to evaluate the hypothesis, we apply a variation of the region perturbation method BID17 removing pixels according to the ranking provided by the attribution maps (higher first (+) or lower first (-)).

We perform this operation replacing one pixel at the time with a zero value and measuring the variation in the target activation.

The results in FIG3 show that pixels highlighted by Occlusion-1 initially have a higher impact on the target output, causing a faster variation from the initial value.

After removing about 20 pixels or more, Integrated Gradients seems to detect more relevant features, given that the variation in the target output is stronger than for Occlusion-1.This is an example of attribution methods solving two different goals: we argue that while Occlusion-1 is better explaining the role of each feature considered in isolation, Integrated Gradients is better in capturing the effect of multiple features together.

It is possible, in fact, that given the presence of several white pixels in the central area, the role of each one alone is not prominent, while the deletion of several of them together causes a drop in the output score.

In order to test this assumption systematically, we propose a property called Sensitivity-n.

Sensitivity-n.

An attribution method satisfies Sensitivity-n when the sum of the attributions for any subset of features of cardinality n is equal to the variation of the output S c caused removing the features in the subset.

Mathematically when, for all subsets of features DISPLAYFORM0 When n = N , with N being the total number of input features, we have DISPLAYFORM1 , wherex is an input baseline representing an input from which all features have been removed.

This is nothing but the definition of Completeness or Summation to Delta, for which Sensitivity-n is a generalization.

Notice that Occlusion-1 satisfy Sensitivity-1 by construction, like Integrated Gradients and DeepLIFT satisfy Sensitivity-N (the latter only without multiplicative units for the reasons discussed in Section 3.1).

-LRP satisfies Sensitivity-N if the conditions of Proposition 3-(ii) are met.

However no methods in TAB0 can satisfy Sensitivity-n for all n:Proposition 4.

All attribution methods defined in TAB0 satisfy Sensitivity-n for all values of n if and only if applied to a linear model or a model that behaves linearly for a selected task.

In this case, all methods of TAB0 are equivalent.

The proof of Proposition 4 is provided in Appendix A.3.

Intuitively, if we can only assign a scalar attribution to each feature, there are not enough degrees of freedom to capture nonlinear interactions.

Besides degenerate cases when DNNs behave as linear systems on a particular dataset, the attribution methods we consider can only provide a partial explanation, sometimes focusing on different aspects, as discussed above for Occlusion-1 and Integrated Gradients.

Although no attribution method satisfies Sensitivity-n for all values of n, we can measure how well the sum of the attributions While it is intractable to test all possible subsets of features of cardinality n, we estimate the correlation by randomly sampling one hundred subsets of features from a given input x for different values of n. FIG5 reports the Pearson correlation coefficient (PCC) computed between the sum of the attributions and the variation in the target output varying n from one to about 80% of the total number of features.

The PCC is averaged across a thousand of samples from each dataset.

The sampling is performed using a uniform probability distribution over the features, given that we assume no prior knowledge on the correlation between them.

This allows to apply this evaluation not only to images but to any kind of input.

We test all methods in TAB0 on several tasks and different architectures.

We use the well-known MNIST dataset BID9 to test how the methods behave with two different architectures (a Multilayer Perceptron (MLP) and a CNN) and four different activation functions.

We also test a simple CNN for image classification on CIFAR10 BID7 ) and the more complex Inception V3 architecture BID27 on ImageNet BID16 samples.

Finally, we test a model for sentiment classification from text data.

For this we use the IMDB dataset BID11 , applying both a MLP and an LSTM model.

Details about the architectures can be found in Appendix C. Notice that it was not our goal, nor a requirement, to reach the state-of-the-art in these tasks since attribution methods should be applicable to any model.

On the contrary, the simple model architecture used for sentiment analysis enables us to show a case where a DNN degenerates into a nearly-linear behavior, showing in practice the effects of Proposition 4.

From these results we can formulate some considerations:1.

Input might contain negative evidence.

Since all methods considered produce signed attributions and the correlation is close to one for at least some value of n, we conclude that the input samples can contain negative evidence and that it can be correctly reported.

This conclusion is further supported by the results in FIG3 where the occlusion of negative evidence produces an increase in the target output.

On the other hand, on complex models like Inception V3, all gradient-based methods show low accuracy in predicting the attribution sign, leading to heatmaps affected by high-frequency noise FIG0 ).2.

Occlusion-1 better identifies the few most important features.

This is supported by the fact that Occlusion-1 satisfies Sensitivity-1, as expected, while the correlation decreases monotonically as n increases in all our experiments.

For simple models, the correlation remains rather high even for medium-size sets of pixels but Integrated Gradients, DeepLIFT and LRP should be preferred when interested in capturing global nonlinear effects and cross-interactions between different features.

Notice also that Occlusion-1 is much slower than gradient-based methods.3.

In some cases, like in MNIST-MLP w/ Tanh, Gradient * Input approximates the behavior of Occlusion-1 better than other gradient-based methods.

This suggests that the instant gradient computed by Gradient * Input is feature-wise very close to the average gradient for these models.4.

Integrated Gradients and DeepLIFT have very high correlation, suggesting that the latter is a good (and faster) approximation of the former in practice.

This does not hold in presence of multiplicative interactions between features (eg.

IMDB-LSTM).

In these cases the analyzed formulation of DeepLIFT should be avoided for the reasons discussed in Section 3.1.5.

-LRP is equivalent to Gradient * Input when all nonlinearities are ReLUs, while it fails when these are Sigmoid or Softplus.

When the nonlinearities are such that f (0) = 0, -LRP diverges from other methods, cannot be seen as a discrete gradient approximator and may lead to numerical instabilities for small values of the stabilizer (Section 3.1).

It has been shown, however, that adjusting the propagation rule for multiplicative interactions and avoiding critical nonlinearities, -LRP can be applied to LSTM networks, obtaining interesting results BID1 .Unfortunately, these changes obstacle the formulation as modified chain-rule and make ad-hoc implementation necessary.6.

All methods are equivalent when the model behaves linearly.

On IMDB (MLP), where we used a very shallow network, all methods are equivalent and the correlation is maximum for almost all values of n. From Proposition 4 we can say that the model approximates a linear behavior (each word contributes to the output independently from the context).

In this work, we have analyzed Gradient * Input, -LRP, Integrated Gradients and DeepLIFT (Rescale) from theoretical and practical perspectives.

We have shown that these four methods, despite their apparently different formulation, are strongly related, proving conditions of equivalence or approximation between them.

Secondly, by reformulating -LRP and DeepLIFT (Rescale), we have shown how these can be implemented as easy as other gradient-based methods.

Finally, we have proposed a metric called Sensitivity-n which helps to uncover properties of existing attribution methods but also traces research directions for more general ones.

Nonlinear operations.

For a nonlinear operation with a single input of the form x i = f (z i ) (i.e. any nonlinear activation function), the DeepLIFT multiplier (Sec. 3.5.2 in Shrikumar et al. BID20 ) is: DISPLAYFORM0 Nonlinear operations with multiple inputs (eg.

2D pooling) are not addressed in BID20 .

For these, we keep the original operations' gradient unmodified as in the DeepLIFT public implementation.

By linear model we refer to a model whose target output can be written as S c (x) = i h i (x i ), where all h i are compositions of linear functions.

As such, we can write DISPLAYFORM1 for some some a i and b i .

If the model is linear only in the restricted domain of a task inputs, the following considerations hold in the domain.

We start the proof by showing that, on a linear model, all methods of TAB0 are equivalent.

Proof.

In the case of Gradient * Input, on a linear model it holds DISPLAYFORM2 , being all other derivatives in the summation zero.

Since we are considering a linear model, all nonlinearities f are replaced with the identity function and therefore ∀z : g DL (z) = g LRP (z) = f (z) = 1 and the modified chain-rules for LRP and DeepLIFT reduce to the gradient chain-rule.

This proves that -LRP and DeepLIFT with a zero baseline are equivalent to Gradient * Input in the linear case.

For Integrated Gradients the gradient term is constant and can be taken out of the integral: DISPLAYFORM3 , which completes the proof the proof of equivalence for the methods in TAB0 in the linear case.

If we now consider any subset of n features x S ⊆ x, we have for Occlusion-1: DISPLAYFORM4 where the last equality holds because of the definition of linear model (Equation 9 ).

This shows that Occlusion-1, and therefore all other equivalent methods, satisfy Sensitivity-n for all n if the model is linear.

If, on the contrary, the model is not linear, there must exists two features x i and x j such that DISPLAYFORM5 .

In this case, either Sensitivity-1 or Sensitivity-2 must be violated since all methods assign a single attribution value to x i and x j .

In general, a non-zero attribution for a feature implies the feature is expected to play a role in the output of the model.

As pointed out by BID25 , humans also assign blame to a cause by comparing the outcomes of a process including or not such cause.

However, this requires the ability to test a process with and without a specific feature, which is problematic with current neural network architectures that do not allow to explicitly remove a feature without retraining.

The usual approach to simulate the absence of a feature consists of defining a baseline x , for example the black image or the zero input, that will represent absence of information.

Notice, however, that the baseline must necessarily be chosen in the domain of the input space and this creates inherently an ambiguity between a valid input that incidentally assumes the baseline value and the placeholder for a missing feature.

On some domains, it is also possible to marginalize over the features to be removed in order to simulate their absence.

BID31 showed how local coherence of images can be exploited to marginalize over image patches.

Unfortunately, this approach is extremely slow and only provide marginal improvements over a pre-defined baseline.

What is more, it can only be applied to images, where contiguous features have a strong correlation, hence our decision to use the method by BID30 as our benchmark instead.

When a baseline value has to be defined, zero is the canonical choice BID25 BID30 BID20 .

Notice that Gradient * Input and LRP can also be interpreted as using a zero baseline implicitly.

One possible justification relies on the observation that in network that implements a chain of operations of the form z j = f ( i (w ji · z i ) + b j ), the all-zero input is somehow neutral to the output (ie. ∀c ∈ C : S c (0) ≈ 0).

In fact, if all additive biases b j in the network are zero and we only allow nonlinearities that cross the origin, the output for a zero input is exactly zero for all classes.

Empirically, the output is often near zero even when biases have different values, which makes the choice of zero for the baseline reasonable, although arbitrary.

C EXPERIMENTS SETUP C.1 MNIST The MNIST dataset (LeCun et al., 1998) was pre-processed to normalize the input images between -1 (background) and 1 (digit stroke).

We trained both a DNN and a CNN, using four activation functions in order to test how attribution methods generalize to different architectures.

The lists of layers for the two architectures are listed below.

The activations functions are defined as ReLU (x) = max(0, x), T anh(x) = sinh(x)/cosh(x), Sigmoid(x) = 1/(1 + e −x ) and Sof tplus(x) = ln(1 + e x ) and have been applied to the output of the layers marked with † in the tables below.

The networks were trained using Adadelta BID29 and early stopping.

We also report the final test accuracy.

The CIFAR-10 dataset BID7 ) was pre-processed to normalized the input images in range [-1; 1] .

As for MNIST, we trained a CNN architecture using Adadelta and early stopping.

For this dataset we only used the ReLU nonlinearity, reaching a final test accuracy of 80.5%.

For gradient-based methods, the attribution of each pixel was computed summing up the attribution of the 3 color channels.

Similarly, Occlusion-1 was performed setting all color channels at zero at the same time for each pixel being tested.

We used a pre-trained Inception V3 network.

The details of this architecture can be found in BID27 .

We used a test dataset of 1000 ImageNet-compatible images, normalized in [-1; 1] that was classified with 95.9% accuracy.

When computing attributions, the color channels were handled as for CIFAR-10.

We trained both a shallow MLP and an LSTM network on the IMDB dataset (Maas et al., 2011) for sentiment analysis.

For both architectures, we trained a small embedding layer considering only the 5000 most frequent words in the dataset.

We also limited the maximum length of each review to 500 words, padding shorter ones when necessary.

We used ReLU nonlinearities for the hidden layers and trained using Adam BID6 and early stopping.

The final test accuracy is 87.3% on both architectures.

For gradient-based methods, the attribution of each word was computed summing up the attributions over the embedding vector components corresponding to the word.

Similarly, Occlusion-1 was performed setting all components of the embedding vector at zero for each word to be tested.

Dense FORMULA3 Dense (1) IMDB LSTM Embedding (5000x32)LSTM FORMULA4 Dense (1)

@highlight

Four existing backpropagation-based attribution methods are fundamentally similar. How to assess it?