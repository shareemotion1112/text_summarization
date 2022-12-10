Deep Convolutional Networks (DCNs) have been shown to be sensitive to Universal Adversarial Perturbations (UAPs): input-agnostic perturbations that fool a model on large portions of a dataset.

These UAPs exhibit interesting visual patterns, but this phenomena is, as yet, poorly understood.

Our work shows that visually similar procedural noise patterns also act as UAPs.

In particular, we demonstrate that different DCN architectures are sensitive to Gabor noise patterns.

This behaviour, its causes, and implications deserve further in-depth study.

Deep Convolutional Networks (DCNs) have enabled deep learning to become one the primary tools for computer vision tasks.

However, adversarial examples-slightly altered inputs that change the model's output-have raised concerns on their reliability and security.

Adversarial perturbations can be defined as the noise patterns added to natural inputs to generate adversarial examples.

Some of these perturbations are universal, i.e. the same pattern can be used to fool the classifier on a large fraction of the tested dataset (MoosaviDezfooli et al., 2017; BID4 .

As shown in FIG1 , it is interesting to observe that such Universal Adversarial Perturbations (UAPs) for DCNs contain structure in their noise patterns.

Results from BID1 together with our results here suggest that DCNs are sensitive to procedural noise perturbations, and more specifically here to Gabor noise.

Existing UAPs have some visual similarities with Gabor noise as in FIG2 .

Convolutional layers induce a prior on DCNs to learn local spatial information BID2 , and DCNs trained on natural image datasets, such as ImageNet, learn convolution filters that are similar UAPs generated for VGG-19 targeting specific layers using singular vector method BID4 .

BID10 and decreasing frequency from left to right.

in appearance to Gabor kernels and colour blobs BID15 BID11 .

Gabor noise is a convolution between a Gabor kernel 2 and a sparse white noise.

Thus, we hypothesize that DCNs are sensitive to Gabor noise, as it exploits specific features learned by the convolutional filters.

In this paper we demonstrate the sensitivity of 3 different DCN architectures (Inception v3, , to Gabor noise on the ImageNet image classification task.

We empirically observed that even random Gabor noise patterns can be effective to generate UAPs.

Understanding this behaviour is important, as the generation and injection of Gabor noise is computationally inexpensive and, therefore, can become a threat to the security and reliability of DCNs.

Compared to standard adversarial examples, UAPs reveal more general features that the DCN is sensitive to.

In contrast, adversarial perturbations generated for specific inputs, though less detectable in many cases, can "overfit" and evade only on inputs they were generated for BID16 .

Previous approaches to generate UAPs use knowledge of the model's learned parameters.

BID8 use the DeepFool algorithm BID7 iteratively over a set of images to construct a UAP.

A different approach is proposed in BID9 , where UAPs are computed using Generative Adversarial Nets (GANs).

BID4 proposed the singular vector method to generate UAPs targeting specific layers of DCNs, learning a perturbation s that maximises the L p -norm of the differences in the activations for that specific layer, f i : DISPLAYFORM0 where the L q -norm of s is constrained to ε.

This can approximated using the Jacobian for that layer: DISPLAYFORM1 The solution s that maximizes this is the (p, q)-singular vector can be computed with the power method BID0 ).

Then, s is effective to generate UAPs targeting a specific layer in the DCN.

The solutions obtained with this method for the first layers of DCNs (see FIG1 ) resemble the Gabor noise patterns shown in FIG2 .However none of these works highlight the interesting visual patterns that manifest from these UAPs.

In contrast, we show that procedural noise can generate UAPs targeting DCNs in a systematic and efficient way.

Gabor noise is the convolution of a sparse white noise and a Gabor kernel, making it a type of Sparse Convolution Noise BID5 BID6 .

The Gabor kernel g with parameters {κ, σ, λ, ω} is the product of a circular Gaussian and a harmonic function DISPLAYFORM0 where κ and σ are the magnitude and width of the Gaussian, and λ and ω are the frequency and orientation of the Harmonic BID6 .

The value of the Gabor noise at point (x, y) is given by DISPLAYFORM1 where (x i , y i ) are the coordinates of sparse random points and w i are random weights.

Gabor noise is an expressive noise function and has exponentially many parameterizations to explore.

To simplify the analysis, we choose anisotropic Gabor noise, where the Gabor kernel parameters and weights are the same for each i.

This results in noise patterns that have uniform orientation and thickness.

We also normalize the variance spectrum of the Gabor noise using the algorithm in BID10 ) to achieve min-max oscillations within the pattern.

For our experiments we use the validation set from the ILSVRC2012 ImageNet image classification task BID12 with 1,000 distinct categories.

We use 3 pre-trained ImageNet DCN architectures from keras.applications: Inception v3 BID14 , ResNet-50 BID3 , and VGG-19 BID13 .Inception v3 take input images with dimensions 299×299× 3 while the other two networks take images with dimensions 224 × 224 × 3.

The kernel size κ = 23 is fixed so that the Gabor kernels will fill the entire image regardless of the distribution of points.

The number of points i distributed will be proportional to the image dimensions, which is independent of the Gabor kernel parameters.

The resulting Gabor noise parameters we control are Θ = {σ, ω, λ}. We test the sensitivity of the models with 1,000 random Gabor noise perturbations generated from uniformly drawn parameters Θ with σ, λ ∈ [1.5, 9] and ω ∈ [0, π].We evaluate our Gabor noise on 5,000 random images from the validation set with an ∞ norm constraint of ε = 12 on the noise.

The choice of 12 256 ≈ 0.047 is consistent with other attacks on ImageNet-scale models with less than 5% perturbation magnitude.

To provide a baseline, we also measure the sensitivity of the models to 1,000 uniform random noise perturbations from {−ε, ε} D×D×3 where D is the image's side length.

This is useful for showing that the sensitivity to Gabor noise is not trivial.

Given model output f , input x ∈ X, perturbation s, and small ε > 0, we define the universal sensitivity of a model on perturbation s over X as DISPLAYFORM0 The norm constraint on s ensures that the perturbation is small.

For this paper, we choose ∞-norm as it is straightforward to impose for Gabor noise perturbations and is often used in the adversarial machine learning literature.

For classification tasks, it is also useful to consider the universal evasion rate of a perturbation s over X |{x ∈ X : arg max f (x) = arg max f (x + s)}| |X| .This corresponds to the definition that an adversarial perturbation is a small change that alters the predicted output label.

Note that we are not interested in the ground truth labels for x or x +

s. We focus instead on how small changes to the input result in large changes to the model's original predictions.

It is worth using both the universal sensitivity and the universal evasion metrics, as the former gives a continuous measure of the sensitivity, while the latter tells us on how much of the dataset that perturbation changes the decision of the model.

Our results show that the order from least to most sensitive models are Inception v3, ResNet-50, and then VGG-19.

This is not surprising as the validation accuracies of these models also appear in the same order.

Overall, our experiments show that the three models are significantly more sensitive to the Gabor noise than random noise.

The universal sensitivity and evasion rates of random noise have very small variance and their values are clustered around their medians.

TAB0 shows how close the quartiles of random noise's are for VGG-19.Inception v3 is also insensitive to random noise, but has a moderate sensitivity to Gabor noise.

ResNet-50 appears to be more sensitive to the random noise than VGG-19, but VGG-19 is more sensitive to Gabor noise than ResNet-50.

This implies that when comparing models higher sensitivity to one type of perturbation does not imply the same relationship for another type of perturbation.

The results in FIG3 suggest that across the three models a random Gabor noise is likely to affect the model outputs on a third or more of the input dataset.

From the histograms, the Gabor noise perturbations appear to centre around relatively high modes for both metrics.

As an example, the first quartile of Gabor noise, as seen in "Best" Parameters.

Taking the top 10 perturbations that VGG-19 is most sensitive to, we see that the other two models are also very sensitive to these noise patterns.

The ranges of the universal evasion rate for these are 69.7% to 71.4% for VGG-19, 50.7% to 53.4% for ResNet-50, and 37.9% to 39.4% for Inception v3.

These values are all above the 3rd quartile for each of these models, showing its generalizability to the other models.

In FIG5 we see a strong correlation (≥ 0.74) between the universal sensitivity and evasion rates across models.

This further suggests that strong perturbations transfer across these models.

We also see a weak correlation between λ and the sensitivity and evasion rates for Inception v3, though there appears to be none between λ and the sensitivity values for ResNet50.The universal evasion rate of the perturbations appears to be insensitive to its Gaussian width σ and orientation ω.

However, the sensitivity for small λ < 0.3 appears to fall below the average, suggesting that below a certain value the Gabor noise does not affect the model's decision.

Interestingly, λ corresponds to the width or thickness of the bands in the image.

Examples of Gabor noise perturbations can be seen in the appendix.

Sensitivity of Inputs.

The model's sensitivity could vary across the input dataset, meaning that the model's predictions is stable on some inputs while more susceptible to small perturbations on others.

To measure this, we look at the sensitivity of single inputs over all perturbations.

Given a set of perturbations s ∈ S, we define the average sensitivity of a model on input x over S as DISPLAYFORM0 and the average evasion rate on x over S as |{s ∈ S : arg max f (x) = arg max f (x + s)}| |S| .The bimodal distribution of the average evasion rate in FIG4 shows that for each model there are two large subsets of the data: One that is very sensitive and another that is very insensitive.

The remaining data points are somewhat uniformly spread in the middle.

Note that for Inception v3, there is a much larger fraction of data points whose prediction is not affected by Gabor perturbations.

The distribution for the average sensitivity appears to have similar shape, but with more inputs in the 0-20% range for Inception v3.

The dataset is far less sensitive against random noise with upwards of 60% of the dataset being insensitive to that noise across all models.

The results show that the tested DCN models are sensitive to Gabor noise for a large fraction of the inputs, even when the parameters of the Gabor noise are chosen at random.

This hints that it may be representative of patterns learned at the earlier layers as Gabor noise appears visually similar to some UAPs targeting earlier layers in DCNs BID4 .This phenomenon has important implications on the security and reliability of DCNs, as it can allow attackers to craft inexpensive black-box attacks.

On the defender's side, Gabor noise patterns can also be used to efficiently generate data for adversarial training to improve DCNs robustness.

However, both the sensitivity exploited and the potential to mitigate it require a more in-depth understanding of the phenomena at play.

In future work, it may be worth analyzing the sensitivity of hidden layer activations across different families of procedural noise patterns and to investigate techniques to reduce the sensitivity of DCNs to perturbations.

As seen in Figure 6 , sensitivity metric values for random noise fall in a narrow range and are significantly smaller than the metric values of the Gabor noise.

This is further shown when comparing the quartiles of the universal evasion and sensitivity in TAB1 Figures 9, 10, 11, 12, and 13 show some adversarial examples with the top perturbations.

Large part of the input dataset is insensitive to random noise as shown in TAB3 , 6 and Figure 7 .

With about 60% of the dataset on having near 0% average evasion over the random noise perturbations for all three models.

@highlight

Existing Deep Convolutional Networks in image classification tasks are sensitive to Gabor noise patterns, i.e. small structured changes to the input cause large changes to the output.