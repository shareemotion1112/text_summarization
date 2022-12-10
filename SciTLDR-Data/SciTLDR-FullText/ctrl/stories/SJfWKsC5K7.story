This paper presents a method to explain the knowledge encoded in a convolutional neural network (CNN) quantitatively and semantically.

How to analyze the specific rationale of each prediction made by the CNN presents one of key issues of understanding neural networks, but it is also of significant practical values in certain applications.

In this study, we propose to distill knowledge from the CNN into an explainable additive model, so that we can use the explainable model to provide a quantitative explanation for the CNN prediction.

We analyze the typical bias-interpreting problem of the explainable model and develop prior losses to guide the learning of the explainable additive model.

Experimental results have demonstrated the effectiveness of our method.

Convolutional neural networks (CNNs) BID17 BID15 BID10 have achieved superior performance in various tasks, such as object classification and detection.

Besides the discrimination power of neural networks, the interpretability of neural networks has received an increasing attention in recent years.

In this paper, we focus on a new problem, i.e. explaining the specific rationale of each network prediction semantically and quantitatively.

"

Semantic explanations" and "quantitative explanations" are two core issues of understanding neural networks.

We hope to explain the logic of each network prediction using clear visual concepts, instead of using middle-layer features without clear meanings or simply extracting pixel-level correlations between network inputs and outputs.

We believe that semantic explanations may satisfy specific demands in real applications.

In contrast to traditional qualitative explanations for neural networks, quantitative explanations enable people to diagnose feature representations inside neural networks and help neural networks earn trust from people.

We expect the neural network to provide the quantitative rationale of the prediction, i.e. clarifying which visual concepts activate the neural network and how much they contribute to the prediction score.

Above two requirements present significant challenges to state-of-the-art algorithms.

To the best of our knowledge, no previous studies simultaneously explained network predictions using clear visual concepts and quantitatively decomposed the prediction score into value components of these visual concepts.

Task: Therefore, in this study, we propose to learn another neural network, namely an explainer network, to explain CNN predictions.

Accordingly, we can call the target CNN a performer network.

Besides the performer, we also require a set of models that are pre-trained to detect different visual concepts.

These visual concepts will be used to explain the logic of the performer's prediction.

We are also given input images of the performer, but we do not need any additional annotations on the images.

Then, the explainer is learned to mimic the logic inside the performer, i.e. the explainer receives the same features as the performer and is expected to generate similar prediction scores.

As shown in Fig. 1 , the explainer uses pre-trained visual concepts to explain each prediction.

The explainer is designed as an additive model, which decomposes the prediction score into the sum of Figure 1 : Explainer.

We distill knowledge of a performer into an explainer as a paraphrase of the performer's representations.

The explainer decomposes the prediction score into value components of semantic concepts, thereby obtaining quantitative semantic explanations for the performer.multiple value components.

Each value component is computed based on a specific visual concept.

In this way, we can roughly consider these value components as quantitative contributions of the visual concepts to the final prediction score.

More specifically, we learn the explainer via knowledge distillation.

Note that we do not use any ground-truth annotations on input images to supervise the explainer.

It is because the task of the explainer is not to achieve a high prediction accuracy, but to mimic the performer's logic in prediction, no matter whether the performer's prediction is correct or not.

Thus, the explainer can be regarded as a semantic paraphrase of feature representations inside the performer, and we can use the explainer to understand the logic of the performer's prediction.

Theoretically, the explainer usually cannot recover the exact prediction score of the performer, owing to the limit of the representation capacity of visual concepts.

The difference of the prediction score between the performer and the explainer corresponds to the information that cannot be explained by the visual concepts.

Challenges: Distilling knowledge from a pre-trained neural network into an additive model usually suffers from the problem of bias-interpreting.

When we use a large number of visual concepts to explain the logic inside the performer, the explainer may biasedly select very few visual concepts, instead of all visual concepts, as the rationale of the prediction (Fig. 4 in the appendix visualizes the bias-interpreting problem).

Just like the typical over-fitting problem, theoretically, the bias interpreting is an ill-defined problem.

To overcome this problem, we propose two types of losses for prior weights of visual concepts to guide the learning process.

The prior weights push the explainer to compute a similar Jacobian of the prediction score w.r.t.

visual concepts as the performer in early epochs, in order to avoid bias-interpreting.

Originality: Our "semantic-level" explanation for CNN predictions has essential differences from traditional studies of "pixel-level" interpreting neural networks, such as the visualization of features in neural networks BID36 BID21 BID27 BID6 BID7 BID24 , the extraction of pixellevel correlations between network inputs and outputs BID14 BID22 BID20 , and the learning of neural networks with interpretable middle-layer features BID38 BID23 .In particular, the explainer explains the performer without affecting the original discrimination power of the performer.

As discussed in BID1 , the interpretability of features is not equivalent to, and usually even conflicts with the discrimination power of features.

Compared to forcing the performer to learn interpretable features, our strategy of explaining the performer solves the dilemma between the interpretability and the discriminability.

In addition, our quantitative explanation has special values beyond the qualitative analysis of CNN predictions BID39 .Potential values of the explainer: Quantitatively and semantically explaining a performer is of considerable practical values when the performer needs to earn trust from people in critical applications.

As mentioned in BID37 , owing to the potential bias in datasets and feature representations, a high testing accuracy still cannot fully ensure correct feature representations in neural networks.

Thus, semantically and quantitatively clarifying the logic of each network prediction is a direct way to diagnose feature representations of neural networks.

Fig. 3 shows example explanations for the performer's predictions.

Predictions whose explanations conflict people's common sense may reflect problematic feature representations inside the performer.

Contributions of this study are summarized as follows.

(i) In this study, we focus on a new task, i.e. semantically and quantitatively explaining CNN predictions. (ii) We propose a new method to explain neural networks, i.e. distilling knowledge from a pre-trained performer into an interpretable additive explainer.

Our strategy of using the explainer to explain the performer avoids hurting the discrimination power of the performer. (iii) We develop novel losses to overcome the typical biasinterpreting problem.

Preliminary experimental results have demonstrated the effectiveness of the proposed method. (iv) Theoretically, the proposed method is a generic solution to the problem of interpreting neural networks.

We have applied our method to different benchmark CNNs for different applications, which has proved the broad applicability of our method.

In this paper, we limit our discussion within the scope of understanding feature representations of neural networks.

Network visualization: The visualization of feature representations inside a neural network is the most direct way of opening the black-box of the neural network.

Related techniques include gradient-based visualization BID36 BID21 BID27 BID35 and up-convolutional nets BID6 BID20 BID13 BID7 BID24 extracted rough pixel-level correlations between network inputs and outputs, i.e. estimating image regions that directly contribute the network output.

Network-attack methods BID14 BID29 computed adversarial samples to diagnose a CNN.

BID16 ) discovered knowledge blind spots of a CNN in a weakly-supervised manner.

BID37 ) examined representations of conv-layers and automatically discover biased representations of a CNN due to the dataset bias.

However, above methods usually analyzed a neural network at the pixel level and did not summarize the network knowledge into clear visual concepts.

BID1 defined six types of semantics for CNN filters, i.e. objects, parts, scenes, textures, materials, and colors.

Then, BID41 proposed a method to compute the image-resolution receptive field of neural activations in a feature map.

Other studies retrieved middle-layer features from CNNs representing clear concepts.

BID25 retrieved features to describe objects from feature maps, respectively.

BID41 selected neural units to describe scenes.

Note that strictly speaking, each CNN filter usually represents a mixture of multiple semantic concepts.

Unlike previous studies, we are more interested in analyzing the quantitative contribution of each semantic concept to each prediction, which was not discussed in previous studies.

A new trend in the scope of network interpretability is to learn interpretable feature representations in neural networks BID12 BID28 BID18 in an un-/weakly-supervised manner.

Capsule nets BID23 and interpretable RCNN BID33 learned interpretable features in intermediate layers.

InfoGAN BID4 and β-VAE BID11 ) learned well-disentangled codes for generative networks.

Interpretable CNNs BID38 learned filters in intermediate layers to represent object parts without given part annotations.

However, as mentioned in BID1 BID39 , interpretable features usually do not have a high discrimination power.

Therefore, we use the explainer to interpret the pre-trained performer without hurting the discriminability of the performer.

Explaining neural networks via knowledge distillation: Distilling knowledge from a black-box model into an explainable model is an emerging direction in recent years.

BID40 used a tree structure to summarize the inaccurate 1 rationale of each CNN prediction into generic decision-making models for a number of samples.

In contrast, we pursue the explicitly quantitative explanation for each CNN prediction.

BID5 learned an explainable additive model, and BID31 ) distilled knowledge of a network into an additive model.

BID30 BID2 BID32 ) distilled representations of neural networks into tree structures.

These methods did not explain the network knowledge using humaninterpretable semantic concepts.

More crucially, compared to previous additive models BID31 , our research successfully overcomes the bias-interpreting problem, which is the core challenge when there are lots of visual concepts for explanation.

In this section, we distill knowledge from a pre-trained performer f to an explainable additive model.

We are given a performer f and n neural networks {f i |i = 1, 2, . . .

, n} that are pre-trained to detect n different visual concepts.

We learn the n neural networks along with the performer, and the n neural networks are expected to share low-layer features with the performer.

Our method also requires a set of training samples for the performer f .

The goal of the explainer is to use inference values of the n visual concepts to explain prediction scores of the performer.

Note that we do not need any annotations on training samples w.r.t.

the task, because additional supervision will push the explainer towards a good performance of the task, instead of objectively reflecting the knowledge in the performer.

Given an input image I, letŷ = f (I) denote the output of the performer.

Without loss of generality, we assume thatŷ is a scalar.

If the performer has multiple outputs (e.g. a neural network for multicategory classification), we can learn an explainer to interpret each scalar output of the performer.

In particular, when the performer takes a softmax layer as the last layer, we use the feature score before the softmax layer asŷ, so thatŷ's neighboring scores will not affect the value ofŷ.

We design the following additive explainer model, which uses a mixture of visual concepts to approximate the function of the performer.

The explainer decomposes the prediction scoreŷ into value components of pre-defined visual concepts.

DISPLAYFORM0 Quantitative contribution from the first visual concept DISPLAYFORM1 where y i and α i (I) denote the scalar value and the weight for the i-th visual concept, respectively.

b is a bias term.

y i is given as the strength or confidence of the detection of the i-th visual concept.

We can regard the value of α i (I) ·

y i as the quantitative contribution of the i-th visual concept to the final prediction.

In most cases, the explainer cannot recover all information of the performer.

The prediction difference between the explainer and the performer reflects the limit of the representation capacity of visual concepts.

According to the above equation, the core task of the explainer is to estimate a set of weights α = [α 1 , α 2 , . . .

, α n ], which minimizes the difference of the prediction score between the performer and the explainer.

Different input images may obtain different weights α, which correspond to different decision-making modes of the performer.

For example, a performer may mainly use head patterns to classify a standing bird, while it may increase the weight for the wing concept to classify a flying bird.

Therefore, we design another neural network g with parameters θ g (i.e. the explainer), which uses the input image I to estimate the n weights.

We learn the explainer with the following knowledge-distillation loss.

DISPLAYFORM2 However, without any prior knowledge about the distribution of the weight α i , the learning of g usually suffers from the problem of bias-interpreting.

The neural network g may biasedly select very few visual concepts to approximate the performer as a shortcut solution, instead of sophisticatedly learning relationships between the performer output and all visual concepts.

Thus, to overcome the bias-interpreting problem, we use a loss L for priors of α to guide the learning process in early epochs.

DISPLAYFORM3 Loss, where w denotes prior weights, which represent a rough relationship between the performer's prediction value and n visual concepts.

Just like α, different input images also have different prior weights w. The loss L(α, w) penalizes the dissimilarity between α and w. DISPLAYFORM4 Note that the prior weights w are approximated with strong assumptions (we will introduce two different ways of computing w later).

We use inaccurate w to avoid significant bias-interpreting, rather than pursue a high accuracy.

Thus, we set a decreasing weight for L, i.e. λ(t) = DISPLAYFORM5 , where β is a scalar constant, and t denotes the epoch number.

In this way, we mainly apply the prior loss L in early epochs.

Then, in late epochs, the influence of L gradually decreases, and our method gradually shifts its attention to the distillation loss for a high distillation accuracy.

We design two types of losses for prior weights, as follows.

DISPLAYFORM6 Some applications require a positive relationship between the prediction of the performer and each visual concept, i.e. each weight α i must be a positive scalar.

In this case, we use the cross-entropy between α and w as the prior loss.

In other cases, the MSE loss between α and w is used as the loss.

· 1 and · 2 denote the L-1 norm and L-2 norm, respectively.

In particular, in order to ensure α i ≥ 0 in certain applications, we add a non-linear activation layer as the last layer of g, i.e. α = log[1 + exp(x)], where x is the output of the last conv-layer.

In this subsection, we will introduce two techniques to efficiently compute rough prior weights w, which are oriented to the following two cases in application.

Case 1, filters in intermediate conv-layers of the performer are interpretable: As shown in FIG0 , learning a neural network with interpretable filters is an emerging research direction in recent years.

For example, BID38 proposed a method to learn CNNs for object classification, where each filter in a high conv-layer is exclusively triggered by the appearance of a specific object part (see FIG4 in the appendix for the visualization of filters).

Thus, we can interpret the classification score of an object as a linear combination of elementary scores for the detection of object parts.

Because such interpretable filters are automatically learned without part annotations, the quantitative explanation for the CNN (i.e. the performer) can be divided into the following two tasks: (i) annotating the name of the object part that is represented by each filter, and (ii) learning an explainer to disentangle the exact additive contribution of each filter (or each object part) to the performer output.

In this way, each f i , i = 1, 2, . . .

, n, is given as an interpretable filter of the performer.

According to BID37 , we can roughly represent the network prediction aŝ DISPLAYFORM0 where x ∈ R H×W ×n denotes a feature map of the interpretable conv-layer, and x hwi is referred to as the activation unit in the location (h, w) of the i-th channel.

y i measures the confidence of detecting the object part corresponding to the i-th filter.

Here, we can roughly use the Jacobian of the network output w.r.t.

the filter to approximate the weight w i of the filter.

Z is for normalization.

Considering that the normalization operation in Equation (4) eliminates Z, we can directly use h,w ∂ŷ ∂x hwi as prior weights w in Equation (4) without a need to compute the exact value of Z.Case 2, neural networks for visual concepts share features in intermediate layers with the performer: As shown in FIG0 , given a neural network for the detection of multiple visual concepts, using certain visual concepts to explain a new visual concept is a generic way to interpret network predictions with broad applicability.

Let us take the detection of a certain visual concept as the targetŷ and use other visual concepts as {y i } to explainŷ.

All visual concepts share features in intermediate layers.

Then, we estimate a rough numerical relationship betweenŷ and the score of each visual concept y i .

Let x be a middle-layer feature shared by both the target and the i-th visual concept.

When we modify the feature x, we can represent the value change of y i using a Taylor series, ∆y i =

We designed two experiments to use our explainers to interpret different benchmark CNNs oriented to two different applications, in order to demonstrate the broad applicability of our method.

In the first experiment, we used the detection of object parts to explain the detection of the entire object.

In the second experiment, we used various face attributes to explain the prediction of another face attribute.

We evaluated explanations obtained by our method qualitatively and quantitatively.

In this experiment, we used the method proposed in BID38 to learn a CNN, where each filter in the top conv-layer represents a specific object part.

We followed exact experimental settings in BID38 , which used the Pascal-Part dataset BID3 to learn six CNNs for the six animal 2 categories in the dataset.

Each CNN was learned to classify the target animal from random images.

We considered each CNN as a performer and regarded its interpretable filters in the top conv-layer as visual concepts to interpret the classification score.

Following experimental settings in BID38 , we applied our method to four types of CNNs, including the AlexNet BID15 , the VGG-M, VGG-S, and VGG-16 networks BID26 , i.e. we learned CNNs for six categories based on each network structure.

Note that as discussed in BID38 , skip connections in residual networks BID10 increased the difficulty of learning part features, so they did not learn interpretable filters in residual networks.

The AlexNet/VGG-M/VGG-S/VGG-16 performer had 256/512/512/512 filters in its top conv-layer, so we set n = 256, 512, 512, 512 for these networks.

We used the masked output of the top conv-layer as x and plugged x to Equation (5) to compute {y i } 1 .

We used the 152-layer ResNet BID10 3 as g to estimate weights of visual concepts 4 .

We set β = 10 for the learning of all explainers.

Note that all interpretable filters in the performer represented object parts of the target category on positive images, instead of describing random (negative) images.

Table 2 : Classification accuracy and relative deviations of the explainer and the performer.

We used relative deviations and the decrease of the classification accuracy to measure the information that could not be explained by pre-defined visual concepts.

Please see the appendix for more results.

Intuitively, we needed to ensure a positive relationship betweenŷ and y i .

Thus, we filtered out negative prior weights w i ← max{w i , 0} and applied the cross-entropy loss in Equation FORMULA6 to learn the explainer.

Evaluation metric: The evaluation has two aspects.

Firstly, we evaluated the correctness of the estimated explanation for the performer prediction.

In fact, there is no ground truth about exact reasons for each prediction.

We showed example explanations of for a qualitative evaluation of explanations.

We also used grad-CAM visualization BID24 of feature maps to prove the correctness of our explanations (see the appendix).

In addition, we normalized the absolute contribution from each visual concept as a distribution of contributions c i = |α i y i |/ j |α j y j |.

We used the entropy of contribution distribution H(c) as an indirect evaluation metric for biasinterpreting.

A biased explainer usually used very few visual concepts, instead of using most visual concepts, to approximate the performer, which led to a low entropy H(c).Secondly, we also measured the performer information that could not be represented by the visual concepts, which was unavoidable.

We proposed two metrics for evaluation.

The first metric is the prediction accuracy.

We compared the prediction accuracy of the performer with the prediction accuracy of using the explainer's output i α i y i + b. Another metric is the relative deviation, which measures a normalized output difference between the performer and the explainer.

The relative deviation of the image I is normalized as |ŷ I − i α I,i y I,i − b|/(max I ∈IŷI − min I ∈IŷI ), whereŷ I denotes the performer's output for the image I .Considering the limited representation power of visual concepts, the relative deviation on an image reflected inference patterns, which were not modeled by the explainer.

The average relative deviation over all images was reported to evaluate the overall representation power of visual concepts.

Note that our objective was not to pursue an extremely low relative deviation, because the limit of the representation power is an objective existence.

In this experiment, we learned a CNN based on the VGG-16 structure to estimate face attributes.

We used the Large-scale CelebFaces Attributes (CelebA) dataset BID19 to train a CNN to estimate 40 face attributes.

We selected a certain attribute as the target and used its prediction score asŷ.

Other 39 attributes were taken as visual concepts to explain the score ofŷ (n = 39).

The target attribute was selected from those representing global features of the face, i.e. attractive, heavy makeup, male, and young.

It is because global features can usually be described by local visual concepts, but the inverse is not.

We learned an explainer for each target attribute.

We used the same 152-layer ResNet structure as in Experiment 1 (expect for n = 39) as g to estimate weights.

We followed the Case-2 implementation in Section 3.1 to compute prior weights w, in which we used the 4096-dimensional output of the first fully-connected layer as the shared feature x. We set β = 0.2 and used the L-2 norm loss in Equation FORMULA6 to learn all explainers.

We used the same evaluation metric as in Experiment 1.

The quantitative explanation for the prediction of the attractive attribute.

Figure 3: Quantitative explanations for the object classification (top) and the face-attribution prediction (bottom) made by performers.

For performers oriented to object classification, we annotated the part that was represented by each interpretable filter in the performer, and we assigned contributions of filters α i y i to object parts (see the appendix).

Thus, this figure illustrates contributions of different object parts.

All object parts made positive contributions to the classification score.

Note that in the bottom, bars indicate elementary contributions α i y i from features of different face attributes, rather than prediction values y i of these attributes.

For example, the network predicts a negative goatee attribute y goatee < 0, and this information makes a positive contribution to the target attractive attribute, α i y i > 0.

Please see the appendix for more results.

We compared our method with the traditional baseline of only using the distillation loss to learn the explainer.

TAB2 evaluates bias-interpreting of explainers that were learned using our method and the baseline.

In addition, Table 2 uses the classification accuracy and relative deviations of the explainer to measure the representation capacity of visual concepts.

Our method suffered much less from the bias-interpreting problem than the baseline.

Fig. 3 shows examples of quantitative explanations for the prediction made by the performer.

We also used the grad-CAM visualization BID24 of feature maps of the performer to demonstrate the correctness of our explanations in Fig. 9 in the appendix.

In particular, Fig. 4 in the appendix illustrates the distribution of contributions of visual concepts {c i } when we learned the explainer using different methods.

Compared to our method, the distillation baseline usually used very few visual concepts for explanation and ignored most strongly activated interpretable filters, which could be considered as bias-interpreting.

In this paper, we focus on a new task, i.e. explaining the logic of each CNN prediction semantically and quantitatively, which presents considerable challenges in the scope of understanding neural networks.

We propose to distill knowledge from a pre-trained performer into an interpretable additive explainer.

We can consider that the performer and the explainer encode similar knowledge.

The additive explainer decomposes the prediction score of the performer into value components from semantic visual concepts, in order to compute quantitative contributions of different concepts.

The strategy of using an explainer for explanation avoids decreasing the discrimination power of the performer.

In preliminary experiments, we have applied our method to different benchmark CNN performers to prove the broad applicability.

Note that our objective is not to use pre-trained visual concepts to achieve super accuracy in classification/prediction.

Instead, the explainer uses these visual concepts to mimic the logic of the performer and produces similar prediction scores as the performer.

In particular, over-interpreting is the biggest challenge of using an additive explainer to interpret another neural network.

In this study, we design two losses to overcome the bias-interpreting problems.

Besides, in experiments, we also measure the amount of the performer knowledge that could not be represented by visual concepts in the explainer.

Table 4 : Classification accuracy of the explainer and the performer.

We use the the classification accuracy to measure the information loss when using an explainer to interpret the performer.

Note that the additional loss for bias-interpreting successfully overcame the bias-interpreting problem, but did not decrease the classification accuracy of the explainer.

Another interesting finding of this research is that sometimes, the explainer even outperformed the performer in classification.

A similar phenomenon has been reported in BID9 .

A possible explanation for this phenomenon is given as follows.

When the student network in knowledge distillation had sufficient representation power, the student network might learn better representations than the teacher network, because the distillation process removed abnormal middle-layer features corresponding to irregular samples and maintained common features, so as to boost the robustness of the student network.

Table 5 : Relative deviations of the explainer.

The additional loss for bias-interpreting successfully overcame the bias-interpreting problem and just increased a bit (ignorable) relative deviation of the explainer.

BID40 ) used a tree structure to summarize the inaccurate rationale of each CNN prediction into generic decision-making models for a number of samples.

This method assumed the significance of a feature to be proportional to the Jacobian w.r.t.

the feature, which is quite problematic.

This assumption is acceptable for BID40 , because the objective of BID40 ) is to learn a generic explanation for a group of samples, and the inaccuracy in the explanation for each specific sample does not significantly affect the accuracy of the generic explanation.

In comparisons, our method focuses on the quantitative explanation for each specific sample, so we design an additive model to obtain more convincing explanations.

Baseline Our method Figure 4 : We compared the contribution distribution of different visual concepts (filters) that was estimated by our method and the distribution that was estimated by the baseline.

The baseline usually used very few visual concepts to make predictions, which was a typical case of bias-interpreting.

In comparisons, our method provided a much more reasonable contribution distribution of visual concepts.

Legs & feet Tail Figure 9 : Quantitative explanations for object classification.

We assigned contributions of filters to their corresponding object parts, so that we obtained contributions of different object parts.

According to top figures, we found that different images had similar explanations, i.e. the CNN used similar object parts to classify objects.

Therefore, we showed the grad-CAM visualization of feature maps BID24 on the bottom, which proved this finding.

We visualized interpretable filters in the top conv-layer of a CNN, which were learned based on BID38 .

We projected activation regions on the feature map of the filter onto the image plane for visualization.

Each filter represented a specific object part through different images.

BID38 ) learned a CNN, where each filter in the top conv-layer represented a specific object part.

Thus, we annotated the name of the object part that corresponded to each filter based on visualization results (see FIG4 for examples).

We simply annotate each filter of the top conv-layer in a performer once, so the total annotation cost was O(N ), where N is the filter number.

Then, we assigned the contribution of a filter to its corresponding part, i.e. Contri part = i:i-th filter represents the part α i y i .

We changed the order of the ReLU layer and the mask layer after the top conv-layer, i.e. placing the mask layer between the ReLU layer and the top conv-layer.

According to BID38 , this operation did not affect the performance of the pre-trained performer.

We used the output of the mask layer as x and plugged x to Equation (5) to compute {y i }.Because the distillation process did not use any ground-truth class labels, the explainer's output i α i y i + b was not sophisticatedly learned for classification.

Thus, we used a threshold i α i y i + b > τ (τ ≈ 0), instead of 0, as the decision boundary for classification.

τ was selected as the one that maximized the accuracy.

Such experimental settings made a fairer comparison between the performer and the explainer.

<|TLDR|>

@highlight

This paper presents a method to explain the knowledge encoded in a convolutional neural network (CNN) quantitatively and semantically.