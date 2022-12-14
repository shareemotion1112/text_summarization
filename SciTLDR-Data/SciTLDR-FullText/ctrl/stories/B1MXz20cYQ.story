When an image classifier makes a prediction, which parts of the image are relevant and why?

We can rephrase this question to ask: which parts of the image, if they were not seen by the classifier, would most change its decision?

Producing an answer requires marginalizing over images that could have been seen but weren't.

We can sample plausible image in-fills by conditioning a generative model on the rest of the image.

We then optimize to find the image regions that most change the classifier's decision after in-fill.

Our approach contrasts with ad-hoc in-filling approaches, such as blurring or injecting noise, which generate inputs far from the data distribution, and ignore informative relationships between different parts of the image.

Our method produces more compact and relevant saliency maps, with fewer artifacts compared to previous methods.

The decisions of powerful image classifiers are difficult to interpret.

Saliency maps are a tool for interpreting differentiable classifiers that, given a particular input example and output class, computes the sensitivity of the classification with respect to each input dimension.

BID3 and BID2 cast saliency computation an optimization problem informally described by the following question: which inputs, when replaced by an uninformative reference value, maximally change the classifier output?

Because these methods use heuristic reference values, e.g. blurred input BID3 or random colors BID2 , they ignore the context of the surrounding pixels, often producing unnatural in-filled images (Figure 2 ).

If we think of a saliency map as interrogating the neural network classifier, these approaches have to deal with a somewhat unusual question of how the classifier responds to images outside of its training distribution.

To encourage explanations that are consistent with the data distribution, we modify the question at hand: which region, when replaced by plausible alternative values, would maximally change classifier output?

In this paper we provide a new model-agnostic framework for computing and visualizing feature importance of any differentiable classifier, based on variational Bernoulli dropout BID4 .

We marginalize out the masked region, conditioning the generative model on the non-masked parts of the image to sample counterfactual inputs that either change or preserve classifier behavior.

By leveraging a powerful in-filling conditional generative model we produce saliency maps on ImageNet that identify relevant and concentrated pixels better than existing methods.

Gradient-based approaches BID12 BID13 BID17 BID10 ) derive a saliency map for a given input example and class target by computing the gradient of the classifier output with respect to each component (e.g., pixel) of the input.

The reliance on the local gradient information induces a bias due to gradient saturation or discontinuity in the DNN activations BID11 .

BID0 observed that some gradientbased saliency computation reflect an inductive bias due to the convolutional architecture, which is independent of the network parameter values.

To explain its response to a particular input x we partition the input x into masked (unobserved) region x r and their complement x = x r ??? x \r .

Then we replace the x r with uninformative reference valu?? x r to test which region x r is important for classifier's output p M (c|x r , x \r ).

Heuristic in-filling BID3 computesx r ad-hoc such as image blur.

This biases the explanation when samples [x r , x \r ] deviate from the data distribution p(x r , x \r ). (1c) We instead sample x r efficiently from a conditional generative model x r ??? p G (x r |x \r ) that respects the data distribution.

Reference-based approaches analyze the sensitivity of classifier outputs to the substitution of certain inputs/pixels with an uninformative reference value.

BID11 linearly approximates this change in classifier output using an algorithm resembling backpropagation.

This method is efficient and addresses gradient discontinuity, but ignores nonlinear interactions between inputs.

BID1 optimizes a variational bound on the mutual information between a subset of inputs and the target, using a variational family that sets input features outside the chosen subset to zero.

In both cases, the choice of background value as reference limits applicability to simple image domains with static background like MNIST.

BID18 computes the saliency of a pixel (or image patch) by treating it as unobserved and marginalizing it out, then measuring the change in classification outcome.

This approach is similar in spirit to ours.

The key difference is that where BID18 iteratively execute this computation for each region, we leverage a variational Bernoulli distribution to efficiently search for optimal solution while encouraging sparsity.

This reduces computational complexity and allows us to model the interaction between disjoint regions of input space.

BID3 computes saliency by optimizing the change in classifier outputs with respect to a perturbed input, expressed as the pixel-wise convex combination of the original input with a reference image.

They offer three heuristics for choosing the reference: mean input pixel value (typically gray), Gaussian noise, and blurred input.

BID2 amortize the cost of estimating these perturbations by training an auxiliary neural network.3 PROPOSED METHOD Dabkowski & Gal (2017) propose two objectives for computing the saliency map:??? Smallest Deletion Region (SDR) considers a saliency map as an answer to the question: What is the smallest input region that could be removed and swapped with alternative reference values in order to minimize the classification score?

??? Smallest Supporting Region (SSR) instead poses the question: What is the smallest input region that could substituted into a fixed reference input in order to maximize the classification score?Solving these optimization problems (which we formalize below) involves a search over input masks, and necessitates reference values to be substituted inside (SDR) or outside (SSR) the masked region.

These values were previously chosen heuristically, e.g., mean pixel value per channel.

We instead consider inputs inside (SDR) or outside (SSR) the masked region as unobserved variables to be marginalized efficiently by sampling from a strong conditional generative model 1 .

We describe our approach for an image application where the input comprises pixels, but our method is more broadly applicable to any domain where the classifier is differentiable.

Generative Methods Consider an input image x comprising U pixels, a class c, and a classifier with output distribution p M (c|x).

Denote by r a subset of the input pixels that implies a partition of the input x = x r ??? x \r .

We refer to r as a region, although it may be disjoint.

We are interested in the classifier output when x r are unobserved, which can be expressed by marginalization as

We then approximate p(x r |x \r ) by some generative model with distribution p G (x r |x \r ) (specific implementations are discussed in section 4.1).

Then given a binary mask 2 z ??? {0, 1} U and the original image x, we define an infilling function 3 ?? as a convex mixture of the input and reference with binary weights, DISPLAYFORM0

The classification score function s M (c) represents a score of classifier confidence on class c; in our experiments we use log-odds: DISPLAYFORM0 SDR seeks a mask z yielding low classification score when a small number of reference pixels are mixed into the mask regions.

Without loss of generality 4 , we can specify a parameterized distribution Given an input, FIDO-CA finds a minimal pixel region that preserves the classifier score following in-fill by CA-GAN BID16 .

BID2 (Realtime) assigns saliency coarsely around the central object, and the heuristic infill reduces the classifier score.

We mask further regions (head and body) of the FIDO-CA saliency map by hand, and observe a drop in the infilled classifier score.

The label for this image is "goose".over masks q ?? (z) and optimize its parameters.

The SDR problem is a minimization w.r.t ?? of DISPLAYFORM1 On the other hand, SSR aims to find a masked region that maximizes classification score while penalizing the size of the mask.

For sign consistency with the previous problem, we express this as a minimization w.r.t ?? of DISPLAYFORM2 Naively searching over all possible z is exponentially costly in the number of pixels U .

Therefore we specify q ?? (z) as a factorized Bernoulli: DISPLAYFORM3 Bern(z u |?? u ).This corresponds to applying Bernoulli dropout BID14 to the input pixels and optimizing the per-pixel dropout rate.

?? is our saliency map since it has the same dimensionality as the input and provides a probability of each pixel being marginalized (SDR) or retained (SSR) prior to classification.

We call our method FIDO because it uses a strong generative model (see section 4.1) to Fill-In the DropOut region.

To optimize the ?? through the discrete random mask z, we follow in computing biased gradients via the Concrete distribution BID9 BID7 ; we use temperature 0.1.

We initialize all our dropout rates ?? to 0.5 since we find it increases the convergence speed and avoids trivial solutions.

We optimize using Adam BID8 with learning rate 0.05 and linearly decay the learning rate for 300 batches in all our experiments.

Our PyTorch implementation takes about one minute on a single GPU to finish one image.

Fong & Vedaldi (2017) compute saliency by directly optimizing the continuous mask z ??? [0, 1] U under the SDR objective, withx chosen heuristically; we call this approach Black Box Meaningful Perturbations (BBMP).

We instead optimize the parameters of a Bernoulli dropout distribution q ?? (z), which enables us to sample reference valuesx from a learned generative model.

Our method uses mini-batches of samples z ??? q ?? (z) to efficiently explore the huge space of binary masks and obtain uncertainty estimates, whereas BBMP is limited to a local search around the current point estimate of the mask z. See Figure 5 for a pseudo code comparison.

In Appendix A.1 we investigate how the choice of algorithm affects the resulting saliency maps.

To avoid unnatural artifacts in ??(x, z), BID3 and BID2 additionally included two forms of regularization: upsampling and total variation penalization.

DISPLAYFORM0 With ??, compute L by Equation 4 or 5 Update z with ??? z L end while Return z as per-feature saliency map DISPLAYFORM1 With ??, compute L by Equation FORMULA3 Upsampling is used to optimize a coarser ?? (e.g. 56 ?? 56 pixels), which is upsampled to the full dimensionality (e.g. 224 ?? 224) using bilinear interpolation.

Total variation penalty smoothes ?? by a 2 regularization penalty between spatially adjacent ?? u .

To avoid losing too much signal from regularization, we use upsampling size 56 and total variation as 0.01 unless otherwise mentioned.

We examine the individual effects of these regularization terms in Appendices A.2 and A.4, respectively.

We first evaluate the various infilling strategies and objective functions for FIDO.

We then compare explanations under several classifier architectures.

In section 4.5 we show that FIDO saliency maps outperform BBMP BID3 in a successive pixel removal task where pixels are in-filled by a generative model (instead of set to the heuristic value).

FIDO also outperforms the method from BID2 on the so-called Saliency Metric on ImageNet.

Appendices A.1-A.6 provide further analysis, including consistency and the effects of additional regularization.

We describe several methods for producing the reference valuex.

The heuristics do not depend on z and are from the literature.

The proposed generative approaches, which producex by conditioning on the non-masked inputs x z=0 , are novel to saliency computation.

Heuristics: Mean sets each pixel ofx according to its per-channel mean across the training data.

Blur generatesx by blurring x with Gaussian kernel (?? = 10) BID3 .

Random samplesx from independent per-pixel per-channel uniform color with Gaussians (?? = 0.2).

Generative Models: Local computesx as the average value of the surrounding non-dropped-out pixels x z=0 (we use a 15 ?? 15 window).

VAE is an image completion Variational Autoencoder BID6 .

Using the predictive mean of the decoder network worked better than sampling.

CA is the Contextual Attention GAN BID16 ; we use the authors' pre-trained model.

Here we examine the choice of objective function between L SDR and L SSR ; see Figure 6 .

We observed more artifacts in the L SDR saliency maps, especially when a weak in-filling method (Mean) is used.

We suspect this unsatisfactory behavior is due to the relative ease of optimizing L SDR .

There are many degrees of freedom in input space that can increase the probability of any of the 999 classes besides c; this property is exploited when creating adversarial examples BID15 ).

Since Figure 6 : Choice of objective between L SDR and L SSR .

The classifier (ResNet) gives correct predictions for all the images.

We show the L SDR and L SSR saliency maps under 2 infilling methods: Mean and CA.

Here the red means important and blue means non-important.

We find that L SDR is more susceptible to artifacts in the resulting saliency maps than L SSR .it is more difficult to infill unobserved pixels that increase the probability of a particular class c, we believe L SSR encourages FIDO to find explanations more consistent with the classifier's training distribution.

It is also possible that background texture is easier for a conditional generative model to fit.

To mitigate the effect of artifacts, we use L SSR for the remaining experiments.

Figure 7: Comparison of saliency map under different infilling methods by FIDO SSR using ResNet.

Heuristics baselines (Mean, Blur and Random) tend to produce more artifacts, while generative approaches (Local, VAE, CA) produce more focused explanations on the targets.

Here we demonstrate the merits of using strong generative model that produces substantially fewer artifacts and a more concentrated saliency map.

In Figure 7 we generate saliency maps of different infilling techniques by interpreting ResNet using L SSR with sparsity penalty ?? = 10 ???3 .

We observed a susceptibility of the heuristic in-filling methods (Mean, Blur, Random) to artifacts in the resulting saliency maps, which may fool edge filters in the low level of the network.

The use of generative in-filling (Local, VAE, CA) tends to mitigate this effect; we believe they encourage in-filled images to lie closer to the natural image manifold.

To quantify the artifacts in the saliency maps by a proxy: the proportion of the MAP configuration (?? > 0.5) that lies outside of the ground truth bounding box.

FIDO-CA produces the fewest artifacts by this metric FIG5 ).

We use FIDO-CA to compute saliency of the same image under three classifier architectures: AlexNet, VGG and ResNet; see FIG6 .

Each architecture correctly classified all the examples.

We observed a qualitative difference in the how the classifiers prioritize different input regions (according to the saliency maps).

For example in the last image, we can see AlexNet focuses more on the body region of the bird, while Vgg and ResNet focus more on the head features.

We follow BID3 and BID11 in measuring the classifier's sensitivity to successively altering pixels in order of their saliency scores.

Intuitively, the "best" saliency map should compactly identify relevant pixels, so that the predictions are changed with a minimum number of altered pixels.

Whereas previous works flipped salient pixel values or set them to zero, we note that this moves the classifier inputs out of distribution.

We instead dropout pixels in saliency order and infill their values with our strongest generative model, CA-GAN.

To make the log-odds score suppression comparable between images, we normalize per-image by the final log-odds suppression score (all pixels infilled).

In FIG0 we evaluate on ResNet and carry out our scoring procedure on 1, 533 randomly-selected correctly-predicted ImageNet validation images, and report the number of pixels required to reduce the normalized log-odds score by a given percent.

We evaluate FIDO under various in-filling strategies as well as BBMP with Blur and Random in-filling strategies.

We put both algorithms on equal footing by using ?? = 1e???3 for FIDO and BBMP (see Section A.1 for further comparisons).

We find that strong generative infilling (VAE and CA) yields more parsimonious saliency maps, which is consistent with our qualitative comparisons.

FIDO-CA can achieve a given normalized log-odds score suppression using fewer pixels than competing methods.

While FIDO-CA may be better adapted to evaluation using CA-GAN, we note that other generative in-filling approaches (FIDO-Local and FIDO-VAE) still out-perform heuristic in-filling when evaluated with CA-CAN.We compare our algorithm to several strong baselines on two established metrics.

We first evaluate whether the FIDO saliency map can solve weakly supervised localization (WSL) BID2 .

After thresholding the saliency map ?? above 0.5, we compute the smallest bounding box containing all salient pixels.

This prediction is "correct" if it has intersection-over-union (IoU) ratio over 0.5 with any of the ground truth bounding boxes.

Using FIDO with various infilling methods, we report the average error rate across all 50, 000 validation images in Table 1 .

We evaluate the authors' pre-trained model of BID2 5 , denoted as "Realtime" in the results.

We also include five baselines: Max (entire input as the bounding box), Center (centered bounding box occupying half the image), Grad BID12 , Deconvnet (Springenberg et al., 2014) , and GradCAM BID10 .

We follow the procedure of mean thresholding in BID3 : we normalize the heatmap between 0 and 1 and binarize by threshold ?? = ???? i where ?? i is the average heatmap for image i.

Then we take the smallest bounding box that encompasses all the binarized heatmap.

We search ?? between 0 to 5 with 0.2 step size on a holdout set to get minimun WSL error.

The best ?? are 1.2, 2 and 1 respectively.

FIDO-CA frugally assigns saliency to contextually important pixels while preserving classifier confidence (Figure 4 ), so we do not necessarily expect our saliency maps to correlate with the typically large human-labeled bounding boxes.

The reliance on human-labeled bounding boxes makes WSL suboptimal for evaluating saliency maps, so we evaluate the so-called Saliency Metric proposed by BID2 , which eschews the human labeled bounding boxes.

The smallest bounding box A is computed as before.

The image is then cropped using this bounding box and upscaling to its original size.

The Saliency Metric is log max(Area(A), 0.05) ??? log p(c|CropAndUpscale(x, A)), the log ratio between the bounding box area and the in-class classifier probability after upscaling.

This metric represents the information concentration about the label within the bounded region.

From the superior performance of FIDO-CA we conclude that a strong generative model regularizes explanations towards the natural image manifold and finds concentrated region of features relevant to the classifier's prediction.

FIG0 : Examples from the ablation study.

We show how each of our two innovations, FIDO and generative infilling, improve from previous methods that adopts BBMP with hueristics infilling (e.g. Blur and Random).

Specifically, we compare with a new variant BBMP-CA that uses strong generative in-filling CA-GAN via thresholding the continous masks: we test a variety of decreasing thresholds.

We find both FIDO (searching over Bernoulli masks) and generative in-filling (CAGAN) are needed to produce compact saliency maps (the right-most column) that retain class information.

See Appendix B for more qualitative examples and in section A.7 for quantitative results.

Can existing algorithms be improved by adding an in-filling generative model without modeling a discrete distribution over per-feature masks?

And does filling in the dropped-out region suffice without an expressive generative model?

We carried out a ablation study that suggests no on both counts.

We compare FIDO-CA to a BBMP variant that uses CA-GAN infilling (called BBMP-CA); we also evaluate FIDO with heuristic infilling (FIDO-Blur, FIDO-Random).

Because the continuous mask of BBMP does not naturally partition the features into observed/unobserved, BBMP-CA first thresholds the masked region r = I(z > ?? ) before generating the reference ??(x r , x \r ) with a sample from CA-GAN.

We sweep the value of ?? as 1, 0.7, 0.5, 0.3, 0.1 and 0.

We find BBMP-CA is brittle with respect to its threshold value, producing either too spread-out or stringent saliency maps ( FIG0 ).

We observed that FIDO-Blur and FIDO-Random produce more concentrated saliency map than their BBMP counterpart with less artifacts, while FIDO-CA produces the most concentrated region on the target with fewest artifacts.

Each of these baselines were evaluated on the two quantitative metrics (Appendix A.7); BBMP-CA considerably underperformed relative to FIDO-CA.

Because the classifier behavior is ill-defined for out-of-distribution inputs, any explanation that relies on out-of-distribution feature values is unsatisfactory.

By modeling the input distribution via an expressive generative model, we can encourage explanations that rely on counterfactual inputs close to the natural manifold.

However, our performance is then upper-bounded by the ability of the generative model to capture the conditional input density.

Fortunately, this bound will improve alongside future improvements in generative modeling.

We proposed FIDO, a new framework for explaining differentiable classifiers that uses adaptive Bernoulli dropout with strong generative in-filling to combine the best properties of recently proposed methods BID3 BID2 BID18 .

We compute saliency by marginalizing over plausible alternative inputs, revealing concentrated pixel areas that preserve label information.

By quantitative comparisons we find the FIDO saliency map provides more parsimonious explanations than existing methods.

FIDO provides novel but relevant explanations for the classifier in question by highlighting contextual information relevant to the prediction and consistent with the training distribution.

We released the code in PyTorch at https://github.

com/zzzace2000/FIDO-saliency.

Here we compare FIDO with two previously proposed methods, BBMP with Blur in-filling strategy BID3 and BBMP with Random in-filling strategy BID2 .

One potential concern in qualitatively comparing these methods is that each method might have a different sensitivity to the sparsity parameter ??.

Subjectively, we observe that BBMP requires roughly 5 times higher sparsity penalty ?? to get visually comparable saliency maps.

In our comparisons we sweep ?? over a reasonable range for each method and show the resulting sequence of increasingly sparse saliency maps FIG0 ).

We use ?? = 5e???4, 1e???3, 2e???3, 5e???3.We observe that all methods are prone to artifacts in the low ?? regime, so the appropriate selection of this value is clearly important.

Interestingly, BBMP Blur and Random respectively find artifacts with different quality: small patches and pixels for Blur and structure off-object lines for Random.

FIDO with CA is arguably the best saliency map, producing fewer artifacts and concentrating saliency on small regions for the images.

Here we examine the effect of learning a reduced dimensionality ?? that upsampled to the full image size during optimization.

We consider a variety of upsampling rates, and in a slight abuse of terminology we refer to the upsampling "size" as the square root of the dimensionality of ?? before upsampling, so smaller size implies more upsampling.

In FIG0 , we demonstrate two examples with different upsampling size under Mean and CA infilling methods with SSR objectives.

The weaker infilling strategy Mean apparently requires stronger regularization to avoid artifacts compared to CA.

Note that although CA produces much less artifacts compared to Mean, it still produces some small artifacts outside of the objects which is unfavored.

We then choose 56 for the rest of our experiments to balance between details and the removal of the artifacts.

FIG0 : Comparisons of upsampling effect in Mean and CA infilling methods with no total variation penalty.

We show the upsampling regularization removes the artifacts especially in the weaker infilling method Mean.

To show the stability of our method, we test our method with different random seeds and observe if they are similar.

In FIG0 , our method produces similar saliency map for 4 different random seeds.

Here we test the effect of total variation prior regularization in FIG0 .

We find the total variation can reduce the adversarial artifacts further, while risking losing signals when the total variation penalty is too strong.

Here we quantitatively compare the in-filling strategies.

The generative approaches (VAE and CA) perform visually sharper images than four other baselines.

Since we expect this random removal should not remove the target information, we use the classification probability of the ResNet as our metric to measure how good the infilling method recover the target prediction.

We quantitatively evaluate the probability for 1, 000 validation images in FIG0 .

We find that VAE and CA consistently outperform other methods, having higher target probability.

We also note that all the heuristic baselines (Mean, Blur, Random) perform much worse since the heuristic nature of these approaches, the images they generate are not likely under the distribution of natural images leading to the poor performance by the classifier.

FIG0 : Box plot of the classifier probability under different infilling with respect to random masked pixels using ResNet under 1, 000 images.

We show that generative models (VAE and CA) performs much better in terms of classifier probability.

A.6 BATCH SIZE EFFECTS FIG0 shows the effect of batch size on the saliency map.

We found unsatisfactory results for batch size less than 4, which we attribute this to the high variance in the resulting gradient estimates.

We show the performance of BBMP-CA with various thresholds ?? on both WSL and SM on subset of 1, 000 images in TAB3 .

We also show more qaulitative examples in FIG14 .

We find BBMP-CA is relatively brittle across different thresholds of ?? .

Though with ?? = 0.3, the BBMP-CA perform slightly better than BBMP and FIDO with heuristics infilling, it still performs substantially inferior to FIDO-CA.

We also perform the flipping experiment in FIG0 and show our FIDO-CA substantially outperforms BBMP-CA with varying different thresholds.

B MORE EXAMPLES FIG0 shows several more infilled counterfactual images, along with the counterfactuals produced by the method from BID2 .

More examples comparing the various FIDO infilling approaches can be found in FIG12 and 21.

BID2 ; FIDO-CA is our method with CA-GAN infilling BID16 .

Classifier confidence p(c|x) is reported below the input and each infilled image.

We hypothesize by that FIDO-CA is able to isolate compact pixel areas of contextual information.

For example, in the upper right image pixels in the net region around the fish are highlighted; this context information is missing from the Realtime saliency map but are apparently relevant to the classifier's prediction.

These 4 examples are bulbul, tench, junco, and ptarmigan respectively.

<|TLDR|>

@highlight

We compute saliency by using a strong generative model to efficiently marginalize over plausible alternative inputs, revealing concentrated pixel areas that preserve label information.