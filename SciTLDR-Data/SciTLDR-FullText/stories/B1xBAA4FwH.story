A plethora of methods attempting to explain predictions of black-box models have been proposed by the Explainable Artificial Intelligence (XAI) community.

Yet, measuring the quality of the generated explanations is largely unexplored, making quantitative comparisons non-trivial.

In this work, we propose a suite of multifaceted metrics that enables us to objectively compare explainers based on the correctness, consistency, as well as the confidence of the generated explanations.

These metrics are computationally inexpensive, do not require model-retraining and can be used across different data modalities.

We evaluate them on common explainers such as Grad-CAM, SmoothGrad, LIME and Integrated Gradients.

Our experiments show that the proposed metrics reflect qualitative observations reported in earlier works.

Over the past few years, deep learning has made significant progress, outperforming the state-ofthe-art in many tasks like image classification (Mahajan et al., 2018) , semantic segmentation (Zhu et al., 2018) , machine translation (Kalchbrenner et al., 2016) and even surpassing humans in the games of Chess and Go (Silver et al., 2016) .

As these models are deployed in more mission-critical systems, we notice that despite their incredible performance on standard metrics, they are fragile (Szegedy et al., 2013; Goodfellow et al., 2014) and can be easily fooled by small perturbations to the inputs (Engstrom et al., 2017) .

Further research has also exposed that these models are biased in undesirable ways exacerbating gender and racial biases (Howard et al., 2017; Escudé Font & Costa-Jussà, 2019) .

These issues have amplified the need for making these black-box models interpretable.

Consequently, the XAI community has proposed a variety of algorithms that aim to explain predictions of these models (Ribeiro et al., 2016; Lundberg & Lee, 2017; Shrikumar et al., 2017; Smilkov et al., 2017; Selvaraju et al., 2016; Sundararajan et al., 2017) .

With such an explosion of interpretability methods (hereon referred to as explainers), evaluating them has become non-trivial.

This is due to the lack of a widely accepted metric to quantitatively compare them.

There have been several attempts to propose such metrics.

Unfortunately, they tend to suffer from major drawbacks like computational cost (Hooker et al., 2018) , inability to be extended to non-image domains (Kindermans et al., 2017a) , or simply focusing only one desirable attribute of a good explainer. (Yeh et al., 2019) .

In this paper, we propose a suite of metrics that attempt to alleviate these drawbacks and can be applied across multiple data modalities.

Unlike the vast majority of prior work, we not only consider the correctness of an explainer, but also the consistency and confidence of the generated explanations.

We use these metrics to evaluate and compare widely used explainers such as LIME (Ribeiro et al., 2016) , Grad-CAM (Selvaraju et al., 2016) , SmoothGrad (Smilkov et al., 2017) and Integrated Gradients (Sundararajan et al., 2017) on an Inception-V3 (Szegedy et al., 2015) model pretrained on the ImageNet dataset (ILSVRC2012) (Deng et al., 2009) , in an objective manner (i.e., without the need of a human-in-the-loop).

Moreover, our proposed metrics are general and computationally inexpensive.

Our main contributions are:

1.

Identifying and formulating the properties of a good explainer.

2.

Proposing a generic, computationally inexpensive suite of metrics to evaluate explainers.

3.

Comparing common explainers and discussing pros and cons of each.

We find that while Grad-CAM seems to perform best overall, it does suffer from drawbacks not reported in prior works.

On the other hand, LIME consistently underperforms in comparison to the other models.

The field of XAI has become an active area of research (Doshi-Velez & Kim, 2017; Lipton, 2016) with significant efforts being made to explain AI models, either by generating local (Ribeiro et al., 2016; Shrikumar et al., 2017; Sundararajan et al., 2017; Selvaraju et al., 2016; Smilkov et al., 2017) or global (Lundberg & Lee, 2017; Ribeiro et al., 2018) explanations.

Simultaneously, there are growing research efforts into methods to formally evaluate and compare explainers (Mohseni et al., 2018; Gunning, 2019; Wolf, 2019; Gilpin et al., 2019) .

Notably, Murdoch et al. (2019) introduced a framework with three desiderata for evaluation, viz.

predictive accuracy, descriptive accuracy and relevancy, with relevancy judged relative to a human.

In contrast, Hall et al. (2019) compiled a set of desired characteristics around effectiveness, versatility, constraints (i.e., privacy, computation cost, information collection effort) and the type of generated explanations, which do not need human evaluation, and therefore are objective.

However, they focus very little on aspects such as correctness.

Recently, DeConvNet (Noh et al., 2015) , Guided BackProp (Springenberg et al., 2015) and LRP (Bach et al., 2015) have been shown to not produce theoretically correct explanations of linear models (Kindermans et al., 2017b) .

As a result, two explanation techniques, PatternNet and PatternAttribution, that are theoretically sound for linear models were proposed.

Other efforts focus on evaluating saliency methods (Kindermans et al., 2017a; Adebayo et al., 2018) and show that they are unreliable for tasks that are sensitive to either data or model.

Samek et al. (2017) and its variations (Hooker et al., 2018; Fong & Vedaldi, 2017; Ancona et al., 2018) infer whether a feature attribution is correct by measuring performance degradation when highly attributed features are removed.

For instance, Hooker et al. (2018) shows that commonly used interpretability methods are less accurate or are on-par with a random designation of feature importance, whereas ensemble approaches such as SmoothGrad (Smilkov et al., 2017) are superior.

Yang & Kim (2019) proposed three complementary metrics to evaluate explainers: model contrast score -comparing two models trained to consider opposite concepts as important, input dependence score -comparing one model with two inputs of different concepts, and input dependence ratecomparing one model with two functionally identical inputs.

These metrics aim to specifically cover aspects of false-positives.

Alvarez-Melis & Jaakkola (2018) define an alternative set of metrics, around explicitness -intelligibility of explanations, faithfulness -feature relevance, and stabilityconsistency of explanations for similar or neighboring samples.

Finally, Yeh et al. (2019) define and evaluate fidelity of explanations, namely quantifying the degree to which an explanation captures how the underlying model itself changes in response to significant perturbations.

Similar to previous work, we focus on objective metrics to evaluate and compare explainers.

However, we not only consider correctness, but also consistency and confidence (as defined next).

3.1 PRELIMINARIES In the following discussions, let x ∈ R n be an arbitrary data point and y be the corresponding ground truth label from the dataset D = {(x i , y i ), 1 ≤ i ≤ M }.

Let f be the classifier realized as a neural network parameterized by weights θ.

Let T be the set of transformations under which semantic information in the input remains unchanged.

If t is an arbitrary transform from this set, let t −1 be its inverse transform.

For example, if t = Rot−90

Let E f be any explainer that generates explanations for predictions made by classifier f 1 .

Finally, let d(, ) be a function that computes the difference between the generated explanations 2 .

For example, if the explainer E generates saliency maps (e.g. GradCAM and SmoothGrad), d could be a simple p norm.

Additionally, in order to ensure that we minimize the impact that pathologies of the underlying classifier have on the properties we are interested in, we assume that the classifier has acceptable test-set performance.

Furthermore, we also assume that the classifier performance does not degrade significantly under the transformations we perform (described in Sec. 3.2.2).

If the classifier does not satisfy these conditions, it is prudent to improve its performance to acceptable levels prior to attempting to explain its outputs.

One cannot extract reliable explanations from underperforming underlying models (Ghorbani et al., 2019; Samek et al., 2017) .

Inspired by earlier works on important aspects of an explainer's quality (Yang & Kim, 2019; Alvarez-Melis & Jaakkola, 2018; Yeh et al., 2019) , our proposed evaluation framework consists of the following components:

We elaborate on these components as well as methods to compute them in the image classification scenario.

Even though these are evaluated independently, they can be combined together to give a single scalar value to compare explainers in a straightforward way.

However, the weight for each component depends heavily on the use case and end-user preference.

This is beyond the scope of the current work and thus is not discussed further.

Further, since we elaborate on the image classification scenario, we use inputs and images interchangeably with the understanding that the described methods or equivalents can be trivially adapted in other modalities.

Correctness (sensitivity or fidelity in literature) refers to the ability of an explainer to correctly identify components of the input that contribute most to the prediction of the classifier.

Most metrics proposed so far focus solely on correctness and attempt to compute it in different ways, often requiring retraining of the underlying classifier.

Moreover, they do not capture all aspects of correctness nor do they generalize to other data modalities.

We propose a novel computationally-inexpensive method that addresses these drawbacks.

It takes into consideration both that the explainer identifies most of the relevant components and does not incorrectly select non-important components as important.

If the explainers are performing as expected, a simple masking of the input image with the associated explanation should provide better accuracy as the network is unlikely to be confused by the nonimportant pixels.

However, we do not observe this in practice, as we show empirically that vanilla masking results in severe performance deterioration (see Table 9 and 10 for results).

We hypothesize that this is because of the following reasons:

• The masked image has a large proportion of empty pixels 3 and thus does not belong to the data distribution (p data ) • Extracted pixels are important in the context of the background pixels, and as such removing context makes the masking meaningless.

Additionally, Convolutions have the inductive bias that the neighbouring pixels are highly correlated that helps perform well on visual tasks (LeCun et al., 1998) .

Simple masking breaks this correlation.

Based on the above observations, we conclude that it is crucial to have a realistic background for the extracted patches to properly evaluate them.

We propose the following procedure to provide a background such that the resulting image is closer the data distribution 2 We do not require d(, ) to be a distance metric in the strictest sense.

3 using the first convolution layer bias as a values for the blank pixels does not help either

For each class in the dataset, we select the top k and bottom k images, sorted in decreasing order based on the probability assigned by the classifier to the ground-truth class.

We then randomly pair each of the top images with one of the bottom images.

For each pair, we extract important regions identified by the explainer from the top image (i.e high confidence images) and overlap them over the corresponding bottom image (i.e low confidence images).

We use the bottom k images for this task as we know that they are uninformative for the classifier as evidenced by the assigned probability.

We thus obtain a dataset of masked images with important regions from the most important images along with relevant yet non-informative backgrounds for each class (see Fig. 1 for an example).

Formally, the masking operation can be represented as:

Where M is the new masked image, a threshold function, H the high confidence image, L the low confidence image and ⊗theelement − wisemultiplicationoperator.

We then measure the accuracy on this masked dataset and compare it with the accuracy on the bottom k images subset.

Note that the above mentioned process only evaluates if the explainer is capturing important pixels.

In order to verify that the explainer does not select non-important pixels, we repeat the same process but instead use the inverted saliency map 4 and recompute accuracy on this dataset.

In this scenario, we expect the accuracy to deteriorate.

Formally, the inverse masking process can be defined as follow:

Figure 1: Examples of the proposed algorithm for correctness Interestingly, these masked accuracies are similar to the precision and recall metrics used in information retrieval (Manning et al., 2009 ).

This provides motivation to combine these differences into a Pseudo-F1 score by computing the harmonic mean of accuracy on normal masked images and 1 -accuracy on inverse masked images.

Formally this can be computed as:

We define consistency as the ability of the explainer to capture the same relevant components under various transformations to the input.

More specifically, if the classifier predicts the same class for both the original and transformed inputs.

Then, consistency measures whether the generated explanation for the transformed input (after applying an inverse transform) is similar to the one generated for the original input.

For example, if we apply vertical flip as a semantically invariant transformation, we flip the generated heatmap from the transformed image before comparing with the heatmap generated for the original image.

Formally, this can be represented as

Semantically Invariant Transforms We focus on a subset of all potential transformations which does not change the semantic information contained in the input.

We call this subset Semantically Invariant Transforms.

Most work so far has considered only noising as a method of transforming the input.

By constraining the magnitude of the added noise, we can control the size of the neighbourhood in which we perturb the images.

In this work, we consider not only simple transformations that perturb the image in a small neighbourhood but also those that move the data point to vastly different regions in the input space while still retaining the semantic information contained within.

This allows us to verify whether the explainer works as expected across larger regions of the input space.

For example, in the case of images, the family of transformations T include affine transformations (translations and rotations), horizontal and vertical flips, noising (white, blue etc.), scaling etc.

In the image domain, d could be realized as the 2 (Euclidean) distance between explanations of the ground truth and inverted the transformed images (according to Eq. 4).

However, Wang et al. (2005) and Zhao & Itti (2016) have shown that 2 is not robust for images and may result in larger distances between the pairs of mostly similar explanations.

This is attributed to the fact that 2 is only a summation of the pixel-wise intensity differences and, as a result, small deformations may results in large distances.

Even when the images are normalized before hand, 2 is still not a suitable distance for images.

Therefore, we instead use Dynamic Time Warping (DTW) (Sakoe & Chiba, 1978) which allows for computing distances between two time series, even when misaligned or out of phase.

Ibrahim & Valli (2008) has shown that DTW is effective for images as well, not only for temporal data as originally intended.

Due to DTW's high computational cost (quadratic time and space complexity), we use FastDTW (Salvador & Chan, 2007) , an approximation of DTW that has linear complexity in order to compute the distance between pairs of explanations.

Finally, confidence is concerned with whether the generated explanation and the masked input result in high confidence predictions.

This is a desirable property to enable explanations to be useful for downstream processes including human inspection (Hall et al., 2019) .

So far, our method for computing correctness sheds light only on the average case and is not particularly useful for individual explanations.

Generating high-confidence predictions is related to the well researched field of max-margin classifiers (Gong & Xu, 2007) .

A large margin in classifiers is widely accepted as a desirable property.

Here, we extend this notion to explainers and propose that explainers generating explanations that result in high confidence predictions are desirable to those that do not.

In addition to the desirable statistical properties that this enforces, high confidence predictions are also vital for building trust with human users of the explainer as they are more interested in the per-instance performance than the average (Narayanan et al., 2018; Ross et al., 2017) .

Concretely, we use the same procedure as in Sec. 3.2.1.

Instead of computing the increase in accuracy, we compute instead the difference in probability assigned to the ground-truth class, as well as the difference in entropy of the softmax distributions of the original and masked images.

We report this for both normal and inverted saliency maps.

We expect to observe a positive probability difference and negative entropy difference under normal masking and an inverted behavior under inverse masking owing to similar reasons discussed in Sec. 3.2.1.

However, explainers that generate coarse explanations can easily fool this metric.

An extreme case is when the explainer considers the entire input as useful.

Such an explainer is useless but will have the theoretically highest change in confidence and entropy.

To combat this and to establish how sparse the generated explanations are, we also report the average number of pixels in the explanations, normalized by the total number of pixels in the image.

We do not combine these numbers into one as different situations have different preferences.

For example, in computational biology domains, sparsity is not as important as increase in confidence.

The right weighting again depends on the use case and user preference.

We use an Inception v3 (Szegedy et al., 2015) architecture pretrained on ImageNet (ISLVC-2012) 5 .

We compare LIME, Grad-CAM, SmoothGrad and Integrated Gradients and measure how they perform on the metrics described previously.

All experiments and explainers (except LIME) were implemented in PyTorch (Paszke et al., 2017) .

Wherever possible, we reused the official implementation or kept our re-implementation as close to the official codebase as possible.

The correctness and confidence metrics for every explainer are computed over 5 runs and mean values are reported.

We use a fixed constant threshold to binarize explanation masks.

We conducted further experiments by thresholding on the percentiles 6 instead(as done in (Smilkov et al., 2017) ).

These results have been reported in tables 6 and 7.

We found that the choice did not affect the relative trends observed.

We consider the following semantically invariant transforms: translations (x = ±0.2, y = ±0.2), rotations ( −15

• , −10

• ), flips (horizontal and vertical).

To establish that these do not produce too many out-of-distribution samples (causing a decrease in classifier performance), we compute the accuracy of the underlying classifier under these transformations.

Table 1 shows that, indeed, drops in accuracy are not significant.

Even though noising is semantically invariant in the image domain, we do not consider it in our experiments as some explainers like Smoothgrad would be unfairly favoured.

Table 2 .

The baseline acc@1 and acc@5 were 11.42% and 53.8% respectively.

We hypothesized that a good explainer's accuracy increases with the normal masking and decreases with inverse masking.

We see the expected increases in accuracies across all the explainers with Grad-CAM obtaining the highest increase at 97.44%.

However, for the inverse masking, we see that both LIME and Grad-CAM show results contrary to our hypothesis.

This can be explained by observing examples of maps generated in Figs. 1 and 4 .

We see that, on average, Grad-CAM generates much larger explanations than all other explainers (can be seen in Table 3 as well).

This implies that Grad-CAM misidentifies several non-important pixels as important and thus when we compute the inverse masks, we remove non-important pixels that could confuse the classifier.

In the case of LIME, we again see from Table 3 and Figs. 1 and 4 that LIME generates the smallest explanations.

We further see from Table 2 that LIME has the smallest accuracy gain (in both acc@1 and acc@5).

These indicate that LIME fails to select important pixels that were selected by all other explainers.

Thus, we can conclude that the inverse masks in case of LIME would contain important pixels as well and thus would cause increase in accuracy as observed.

As detailed previously, our methodology for computing correctness involves choosing a number k of top and bottom images to be used for masking.

We evaluate how sensitive the measured correctness of explainers are to the value of k. We report the changes in accuracy with respect to the unmasked bottom images for k={5,10,15,20,25} in Fig. 2 .

The actual accuracy numbers are also reported in Tables 4 and 5 .

We see that for both acc@1 and acc@5, the change in accuracy for normal masking decreases as we increase k. This is as expected since the average confidence gap between the top-k and the bottom-k images decreases as k increases.

This means that the important pixels in the background images are masked with non-important pixels from the foreground images.

On the contrary, LIME shows a smaller decrease in accuracy (both acc@1 and acc@5).

This can be explained by the fact that LIME does not capture all important pixels, and therefore all important pixels from the background are not replaced by less-informative pixels.

Similarly, for acc@1 and acc@5 for inverse masking, we see that LIME, Smoothgrad and Integrated Gradients behave as expected, i.e., the drop in accuracy is diminished with k is increased as we are retaining the informative parts from the new background images.

Interestingly, the drop in accuracy for Grad-CAM is stable and close to zero.

To understand this, we refer again to Table 3 and note that Grad-CAM produces the smallest inverse maps on average.

This implies that when we perform the inverse masking, we retain much of the informative pixels of the background image and thus do not see significant drops in accuracy relative to the unmasked bottom-k image dataset.

Next, we evaluate consistency by computing the distance with FastDTW between the saliency maps generated on the original images and those generated when transformations are applied to the input image (Following Eq. 4).

Fig. 3 and Table 8 report the normalized distances relative to each transformation (i.e., heatmaps sum to 1).

First, as transformations become more drastic relative to the original saliency maps, the distances also increase.

This is the desired behavior one would expect, thus motivating our choice for using FastDTW.

Second, Grad-CAM outperforms all other explainers, as reflected by the fact that its corresponding distances are always smallest.

It is followed by Smoothgrad, Integrated Gradients and LIME.

This is expected given the grainy saliency maps obtained with Integrated Gradients and Smoothgrad, as well as the patchy heatmaps generated with LIME.

Measuring confidence quantifies the performance of the explainers on a per-instance case and not only in the average.

As described in Sec. 3.2.3, we compute the change in probability assigned to the ground-truth class (∆ conf) as well as the change in entropy of the softmax distribution (∆ entropy) as proxies for estimating the confidence of explanations.

Additionally, we report the proportions of pixels in the heatmaps to the total number of pixels 7 , averaged across the top-k dataset.

We see that for confidence, the trends mimic the ones observed in Table 2 .

This implies that masking with extracted heatmaps not only increases accuracy but also results in high-confidence predictions across explainers.

More specifically, we see that Grad-CAM again outperforms the other explainers (both ∆ conf and ∆ entropy) in the normal heatmaps by large margins.

In the case of inverse masking, confidence and entropy for LIME show behaviours contrary to our expectations.

This can be attributed to the "patchiness" of the explanations generated by LIME which was discussed in the previous sections.

7 89401 in the standard ImageNet preprocessing pipeline for Inception-v3 In this paper, we formulated desired properties of a good explainer and proposed a generic, computationally inexpensive suite of metrics -correctness, consistency and confidence -to objectively evaluate and compare explainers.

We compared well-known explainers, such as LIME, Grad-CAM, Integrated Gradients and SmoothGrad, on a pretrained Inception-V3 model on the ImageNet dataset.

Our experiments show that the metrics proposed capture various pros and cons of each explainer allowing users to make an informed choice about which explainer to use for their use case.

Specifically, we observe that Grad-CAM often performs better than the other explainers but suffers from drawbacks when inverse masking situations are considered.

On the other hand, LIME performs poorly in all situations we consider.

Moreover, we also point out the pitfalls of trying to combine results from multiple metrics as they tend to hide anomalous behaviours of the underlying metrics (as seen from Pseudo-F1 from Table 2 ).

We recommend that users sanity-check explainers by looking at individual metrics before making a decision based on the combined metric.

Furthermore, we urge the XAI community to resist the temptation to propose an one-size-fits-all metrics as we have shown that such metrics tend to hide nuanced trade-offs that practitioners need to be aware of.

Going forward, we invite the research community to test our metrics on other explainers, datasets, underlying classifiers and data modalities.

Additionally, since the metrics proposed are differentiable, we believe exciting new liens of research would be to develop explainers that directly optimize for these metrics, as well as self-explaining models that incorporate such metrics into their learning regiment.

A ADDITIONAL EXPERIMENTAL RESULTS

@highlight

We propose a suite of metrics that capture desired properties of explainability algorithms and use it to objectively compare and evaluate such methods