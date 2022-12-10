Over the last few years, the phenomenon of adversarial examples --- maliciously constructed inputs that fool trained machine learning models --- has captured the attention of the research community, especially when the adversary is restricted to making small modifications of a correctly handled input.

At the same time, less surprisingly, image classifiers lack human-level performance on randomly corrupted images, such as images with additive Gaussian noise.

In this work, we show that these are two manifestations of the same underlying phenomenon.

We establish this connection in several ways.

First, we find that adversarial examples exist at the same distance scales we would expect from a linear model with the same performance on corrupted images.

Next, we show that Gaussian data augmentation during training improves robustness to small adversarial perturbations and that adversarial training improves robustness to several types of image corruptions.

Finally, we present a model-independent upper bound on the distance from a corrupted image to its nearest error given test performance and show that in practice we already come close to achieving the bound, so that improving robustness further for the corrupted image distribution requires significantly reducing test error.

All of this suggests that improving adversarial robustness should go hand in hand with improving performance in the presence of more general and realistic image corruptions.

This yields a computationally tractable evaluation metric for defenses to consider: test error in noisy image distributions.

State-of-the-art computer vision models can achieve superhuman performance on many image classification tasks.

Despite this, these same models still lack the robustness of the human visual system to various forms of image corruptions.

For example, they are distinctly subhuman when classifying images distorted with additive Gaussian noise BID12 , they lack robustness to different types of blur, pixelation, and changes in brightness BID17 , lack robustness to random translations of the input BID2 , and even make errors when foreign objects are inserted into the field of view BID25 .

At the same time, they also are sensitive to small, worst-case perturbations of the input, so-called "adversarial examples" BID28 .

This latter phenomenon has struck many in the machine learning community as surprising and has attracted a great deal of research interest, while the former seems to inspire less surprise and has received considerably less attention.

Our classification models make errors on two different sorts of inputs: those found by randomly sampling from some predetermined distribution, and those found by an adversary deliberately searching for the closest error to a given point.

In this work, we ask what, if anything, is the difference between these two types of error.

Given that our classifiers make errors in these corrupted image distributions, there must be a closest such error; do we find that this closest error appears at the distance we would expect from the model's performance in noise, or is it in fact "surprisingly" close?The answer to this question has strong implications for the way we approach the task of eliminating these two types of errors.

An assumption underlying most of the work on adversarial examples is that solving it requires a different set of methods than the ones being developed to improve model generalization.

The adversarial defense literature focuses primarily on improving robustness to small perturbations of the input and rarely reports improved generalization in any distribution.

We claim that, on the contrary, adversarial examples are found at the same distance scales that one should expect given the performance on noise that we see in practice.

We explore the connection between small perturbation adversarial examples and test error in noise in two different ways.

First, in Sections 4 and 5, we provide empirical evidence of a close relationship between test performance in Gaussian noise and adversarial perturbations.

We show that the errors we find close to the clean image and the errors we sample under Gaussian noise are part of the same large set and show some visualizations that illustrate this relationship.

(This analysis builds upon prior work which makes smoothness assumptions on the decision boundary to relate these two quantities.)

This suggests that training procedures designed to improve adversarial robustness might reduce test error in noise and vice versa.

We provide results from experiments which show that this is indeed the case: for every model we examined, either both quantities improved or neither did.

In particular, a model trained on Gaussian noise shows significant improvements in adversarial robustness, comparable to (but not quite as strong as) a model trained on adversarial examples.

We also found that an adversarially trained model on CIFAR-10 shows improved robustness to random image corruptions.

Finally, in Section 6, we establish a relationship between the error rate of an image classification model in the presence of Gaussian noise and the existence of adversarial examples for noisy versions of test set images.

In this setting we can actually prove a rigorous, model-independent bound relating these two quantities that is achieved when the error set is a half space, and we see that the models we tested are already quite close to this optimum.

Therefore, for these noisy image distributions, our models are already almost as adversarially robust as they can be given the error rates we see, so the only way to defend against adversarial examples is to reduce test error.

In this work we will investigate several different models trained on the MNIST, CIFAR-10 and ImageNet datasets.

For MNIST and CIFAR-10 we look at the naturally trained and adversarially trained models which have been open-sourced by BID22 .

We also trained the same model on CIFAR-10 with Gaussian data augmentation.

For ImageNet, we investigate Wide ResNet-50 trai]ned with Gaussian data augmentation.

We were unable to study the effects of adversarial training on ImageNet because no robust open sourced model exists (we considered the models released in BID29 but found that they only minimally improve robustness to the white box PGD adversaries we consider here).

Additional training details can be found in Appendix A.

The broader field of adversarial machine learning studies general ways in which an adversary may interact with an ML system, and dates back to 2004 BID8 BID3 .

Since the work of BID28 , a subfield has focused specifically on the phenomenon of small adversarial perturbations of the input, or "adversarial examples."

In BID28 it was proposed these adversarial examples occupy a dense, measure-zero subset of image space.

However, more recent work has provided evidence that this is not true.

For example, BID14 ; BID16 shows that under linearity assumptions of the decision boundary small adversarial perturbations exist when test error in noise is non-zero.

Gilmer et al. (2018b) showed for a specific data distribution that there is a fundamental upper bound on adversarial robustness in terms of test error.

BID23 has generalized these results to a much broader class of distributions.

Recent work has proven for a synthetic data distribution that adversarially robust generalization requires more data BID26 .

The distribution they consider when proving this result is a mixture of high dimensional Gaussians.

As we will soon discuss, every set E of small measure in the high dimensional Gaussian distribution has large boundary measure.

Therefore, at least for the data distribution considered, the main conclusion of this work, "adversarially robust generalization requires more data", is a direct corollary of the statement "generalization requires more data."

Understanding the relationship between nearby errors and model generalization requires understanding the geometry of the error set of a statistical classifier, that is, the set of points in the input space on which the classifier makes an incorrect prediction.

In particular, the assertion that these adversarial examples are a distinct phenomenon from test error is equivalent to stating that the error set is in some sense poorly behaved.

We study two functions of a model's error set E.The first quantity, test error under a given distribution of inputs q(x), is the probability that a random sample from the distribution q is in E. We will denote this P x∼q [x ∈ E]; reducing this quantity when q is the natural data distribution is the goal of supervised learning.

While one usually takes q to be the distribution from which the training set was sampled, we will also consider other distributions over the course of this paper.

When q includes points from outside the natural data distribution, a decision needs to be made about the labels in order to define E. The only such cases we will consider in this paper are noisy perturbations of training or test points, and we will always assume that the noise is at a scale which is small enough not to change the label.

This assumption is commonly made in works which study model robustness to random corruptions of the input BID17 BID12 .

Some examples noisy images can be found in FIG2 in the appendix.

The second quantity is called adversarial robustness.

For an input x and a metric on the input space d, let d(x, E) denote the distance from x to the nearest point of E. For any , let E denote the set {x : d(x, E) < }, the set of points within of an error.

The adversarial robustness of the model is then P x∼q [x ∈ E ], the probability that a random sample from q is within distance of some point in the error set.

Reducing this quantity is the goal of much of the adversarial defense literature.

When we refer to "adversarial examples" in this paper, we will always mean these nearby errors.

In geometric terms we can think of P x∼q [x ∈ E] as a sort of volume of the error set while P x∼q [x ∈ E ] is related to its surface area.

More directly, P x∼q [x ∈ E ] is what we will call the -boundary measure, the volume under q of the region within of the surface or the interior.

The adversarial example phenomenon is then simply that, for small , P x∼q [x ∈ E ] can be large even when P x∼q [x ∈ E] is small.

In other words, most correctly classified inputs are very close to a misclassified point, even though the model is very accurate.

In high-dimensional spaces this phenomenon is not isolated to the error sets of statistical classifiers.

In fact almost every nonempty set of small volume has large -boundary measure, even sets that seem very well-behaved.

As a simple example, consider the measure of the set E = {x ∈ R n : ||x|| 2 < 1} under the Gaussian distribution q = N (0, σ 2 I).

For n = 1000, σ = 1.05/ √ n, and = 0.1, we have P x∼q [x ∈ E] ≈ 0.02 and P x∼q [x ∈ E ] ≈ 0.98, so most samples from q will be close to E despite the fact that E has relatively little measure under the Gaussian distribution.

If we relied only on our low-dimensional spatial intuition, we might be surprised to find how consistently small adversarial perturbations could be found -98% of our test points would have an error at distance 0.1 or less even though only 2% are misclassified.

In high dimensions, it is much easier for most points to be close to some set even if that set itself has a small volume.

Contrary to what one might expect from our low-dimensional intuition, this does not require the set in question to be somehow pathological; in our example, it was just a ball.

Therefore, when we see that some image classifier has errors in some noise distribution q (so that P x∼q [x ∈ E] is appreciably bigger than zero) it is possible that E is much larger even if E is quite simple, so the existence of small worst-case perturbations should be expected given imperfect robustness to large average-case corruptions.

In the sections that follow we will make this precise.

The Linear Case.

For linear models, the relationship between errors in Gaussian noise and small perturbations of a clean image is exact.

For an image x, let d(x) be the distance from x to decision boundary and let σ(x, µ) be the σ for which P x∼q [x ∈ E] is some fixed error rate µ. (As we mentioned in the introduction, we assume that σ is small enough that adding this noise does not change the "correct" label.)

Then we have d(x) = −σ(x, µ)Φ −1 (µ), where DISPLAYFORM0 is the cdf of the univariate standard normal distribution.

Note that this equality depends only on the error rate µ and the standard deviation σ of a single component, and not directly on the dimension.

This might seem at odds with the emphasis on high-dimensional geometry in Section 3.

The dimension does appear if we consider the norm of a typical sample from N (0, σ 2 I), which is σ √ n. As the dimension increases, so does the ratio between the distance to a noisy image and the distance to the decision boundary.

The decision boundary of a neural network is, of course, not linear.

However, by computing the ratio between d(x) and σ(x, µ) for neural networks and comparing it to what it would be for a linear model, we can investigate the question posed in the introduction: do we see adversarial examples at the distances we do because of pathologies in the shape of the error set, or do we find them at about the distances we would expect given the error rates we see in noise?

We ran experiments on the error sets of several neural image classifiers and found evidence that is much more consistent with the second of these two possibilities.

This relationship was also explored in BID14 BID1 ; here we additionally measure how data augmentation affects this relationship.

We examined this relationship for neural networks when µ = 0.01.

For each test point, we compared σ(x, µ) to an estimate of d(x).

It is not actually possible to compute d(x) precisely for the error set of a neural network.

In fact, finding the distance to the nearest error is NP-hard BID19 .

Instead, the best we can do is to search for an error using a method like PGD BID22 and report the nearest error we can find.

FIG0 shows the results for several CIFAR-10 and ImageNet models, including ordinary trained models, models trained on noise with σ = 0.4, and an adversarially trained CIFAR-10 model.

We also included a line representing how these quantities would be related for a linear model.

We can see that none of the models we examined have nearby errors at a scale much smaller than we would expect from a linear model.

Indeed, while the adversarially trained model does deviate from the linear case to a greater extent than the others, it does so in the direction of greater distances to the decision boundary.

Moreover, we can see from the histograms that both of the interventions that increase d(x) also increase σ(x, µ).

So, to explain the distances to the errors we can find using PGD, it is not necessary to rely on any great complexity in the shape of the error set; a linear model with the same error rates in noise would have errors just as close.

An image from the test set (black), a random misclassified Gaussian perturbation at standard deviation 0.08 (blue), and an error found using PGD (red).

The estimated measure of the cyan region ("miniature poodle") in the Gaussian distribution is about 0.1%.

The small diamond-shaped region in the center of the image is the l ∞ ball of radius 8/255.

Right: A slice at a larger scale with the same black point, together with an error from the clean set (blue) and an adversarially constructed error (red) which are both assigned to the same class ("elephant").Visualizing the Decision Boundary.

In FIG1 we drew some pictures of two-dimensional slices of image space through several different triples of points. (Similar visualizations have previously appeared in , and are called "church window plots.") We see some common themes.

In the figure on the left, we see that an error found in Gaussian noise lies in the same connected component of the error set as an error found using PGD, and that at this scale that component visually resembles a half space.

This figure also illustrates the relationship between test error and adversarial robustness.

To measure adversarial robustness is to ask whether or not there are any errors in the l ∞ ball -the small diamond-shaped region in the center of the image -and to measure test error in noise is to measure the volume of the error set in the defined noise distribution.

At least in this slice, nothing distinguishes the PGD error from any other point in the error set apart from its proximity to the center point.

The figure on the right shows a different slice through the same test point but at a larger scale.

This slice includes an ordinary test error along with an adversarial perturbation of the center image constructed with the goal of maintaining visual similarity while having a large l 2 distance.

The two errors are both classified (incorrectly) by the model as "elephant."

This adversarial error is actually farther from the center than the test error, but they still clearly belong to the same connected component.

This suggests that defending against worst-case content-preserving perturbations (Gilmer et al., 2018a) requires removing all errors at a scale comparable to the distance between unrelated pairs of images.

Many more church window plots can be found in Appendix G.

For a linear model, improving generalization in the presence of noise is equivalent to increasing the distance to the decision boundary.

The results from the previous section suggest that a similar relationship should hold for other statistical classifiers, including neural networks.

That is, augmenting the training data distribution with noisy images ought to increase the distance to the decision boundary, and augmenting the training distribution with small-perturbation adversarial examples should improve performance in noise.

Here we present evidence that this is the case.

We analyzed the performance of the models described in Section 1 on four different noise distributions: two types of Gaussian noise, pepper noise BID17 , and a randomized variant of the stAdv adversarial attack introduced in BID32 .

We used both ordinary, spherical Gaussian noise and what we call "PCA noise," which is Gaussian noise supported only on the Table 1 : The performance of the models we considered under various noise distributions, together with our measurements of those models' robustness to small l p perturbations.

For all the robustness tests we used PGD with 100 steps and a step size of /25.

The adversarially trained CIFAR-10 model is the open sourced model from BID22 .subspace spanned by the first 100 principal components of the training set.

Pepper noise randomly assigns channels of the image to 1 with some fixed probability.

Details of the stAdv attack can be found in Appendix B, but it visually similar to Gaussian blurring where σ controls the severity of the blurring.

Example images that have undergone each of the noise transformations we used can be found in Appendix I. Each model was also tested for l p robustness with a variety of norms and 's using the same PGD attack as in Section 4.For CIFAR-10, standard Gaussian data augmentation yields comparable (but slightly worse) results to adversarial training on all considered metrics.

For ImageNet we found that Gaussian data augmentation improves robustness to small l 2 perturbations as well as robustness to other noise corruptions.

The results are shown in Table 1 .

This holds both for generalization in all noises considered and for robustness to small perturbations.

We found that performing data augmentation with heavy Gaussian noise (σ = 0.4 for CIFAR-10 and σ = 0.8 for ImageNet) worked best.

The adversarially trained CIFAR-10 models were trained in the l ∞ metric and they performed especially well on worst-case perturbations in this metric.

Prior work has observed that Gaussian data augmentation helps small perturbation robustness on MNIST BID18 , but to our knowledge we are the first to measure this on CIFAR-10 and ImageNet.

Neither augmentation method shows much improved generalization in PCA noise.

We hypothesize that adversarially trained models learn to project away the high-frequency information in the input, which would do little to improve performance in PCA noise, which is supported in the low-frequency subspace of the data distribution.

Further work would be required to establish this.

We also considered the MNIST adversarially trained model from BID22 , and found it to be a special case where although robustness to small perturbations was increased generalization in noise was not improved.

This is because this model violates the linearity assumption discussed in Section 4.

This overfitting to the l ∞ metric has been observed in prior work BID27 .

More details can be found in Appendix D.Although no l p -robust open sourced ImageNet model exists, recent work has found that the adversarially trained models on Tiny ImageNet from BID18 generalize very well on a large suite of common image corruptions BID17 .Failed Adversarial Defenses Do Not Improve Generalization in Noise.

We performed a similar analysis on seven previously published adversarial defense strategies.

These methods have already been shown to result in masking gradients, which cause standard optimization procedures to fail to find errors, rather than actually improving small perturbation robustness BID0 .

We find Figure 3 : The performance in Gaussian noise of several previously published defenses for ImageNet, along with a model trained on Gaussian noise at σ = 0.4 for comparison.

For each point we ran ten trials; the error bars show one standard deviation.

All of these defenses are now known not to improve adversarial robustness BID0 .

The defense strategies include bitdepth reduction (Guo et al., 2017 ), JPEG compression (Guo et al., 2017 BID13 BID1 BID10 , Pixel Deflection BID24 , total variance minimization (Guo et al., 2017), respresentation-guided denoising BID20 , and random resizing and random padding of the input image BID33 .that these methods also show no improved generalization in Gaussian noise.

The results are shown in Figure 3 .

Given how easy it is for a method to show improved robustness to standard optimization procedures without changing the decision boundary in any meaningful way, we strongly recommend that future defense efforts evaluate on out-of-distribution inputs such as the noise distributions we consider here.

The current standard practice of evaluating solely on gradient-based attack algorithms is making progress more difficult to measure.

Obtaining Zero Test Error in Noise is Nontrivial.

It is important to note that applying Gaussian data augmentation does not reduce error rates in Gaussian noise to zero.

For example, we performed Gaussian data augmentation on CIFAR-10 at σ = .15 and obtained 99.9% training accuracy but 77.5% test accuracy in the same noise distribution.

(For comparison, the naturally trained obtains 95% clean test accuracy.)

Previous work BID12 has also observed that obtaining perfect generalization in large Gaussian noise is nontrivial.

This mirrors BID26 , which found that small perturbation robustness did not generalize to the test set.

This is perhaps not surprising given that error rates on the clean test set are also non-zero.

Although the model is in some sense "superhuman" with respect to clean test accuracy, it still makes many mistakes on the clean test set that a human would never make.

We collected some examples in Appendix I. More detailed results on training and testing in noise can be found in Appendices C and H.

The Gaussian Isoperimetric Inequality.

Let x be a correctly classified image and consider the distribution q of Gaussian perturbations of x with some fixed variance σ 2 I. For this distribution, there is a precise sense in which small adversarial perturbations exist only because test error is nonzero.

That is, given the error rates we actually observe on noisy images, most noisy images must be close to the error set.

This result holds completely independently of any assumptions about the model and follows from a fundamental geometric property of the high-dimensional Gaussian distribution, which we will now make precise.

For an image x and the corresponding noisy image distribution q, let * q (E) be the median distance from one of these noisy images to the nearest error.

(In other words, it is the for which P x∼q [x ∈ E ] = 1 2 .)

As before, let P x∼q [x ∈ E] be the probability that a random Gaussian perturbation Figure 4 : The adversarial example phenomenon occurs for noisy images as well as clean ones.

Starting with a noisy image that that is correctly classified, one can apply carefully crafted imperceptible noise to it which causes the model to output an incorrect answer.

This occurs even though the error rate among random Gaussian perturbations of this image is small (less than .1% for the ImageNet panda shown above).

In fact, we prove that the presence of errors in Gaussian noise logically implies that small adversarial perturbations exists around noisy images.

The only way to "defend" against such adversarial perturbations is to reduce the error rate in Gaussian noise.

of x lies in E. It is possible to deduce a bound relating these two quantities from the Gaussian isoperimetric inequality BID4 .

The form we will use is:Theorem (Gaussian Isoperimetric Inequality).

Let q = N (0, σ 2 I) be the Gaussian distribution on R n with variance σ 2 I, and let DISPLAYFORM0 2)dx, the cdf of the univariate standard normal distribution.

If DISPLAYFORM1 , with equality when E is a half space.

In particular, for any machine learning model for which the error rate in the distribution q is at least µ, the median distance to the nearest error is at most −σΦ −1 (µ).

(Note that Φ −1 (µ) is negative when µ < 1 2 .) Because each coordinate of a multivariate normal is a univariate normal, −Φ −1 (µ) is the distance to a half space for which the error rate is µ when σ = 1. (We have the same indirect dependence on dimension here as we saw in Section 4: the distance to a typical sample from the Gaussian is σ √ n.)In Appendix E we will give the more common statement of the Gaussian isoperimetric inequality along with a proof of the version presented here.

In geometric terms, we can say that a half space is the set E of a fixed volume that minimizes the surface area under the Gaussian measure, similar to how a circle is the set of fixed area that minimizes the perimeter.

So among models with some fixed test error P x∼q [x ∈ E], the most robust on this distribution are the ones whose error set is a half space.

Comparing Neural Networks to the Isoperimetric Bound.

We evaluated these quantities for several models and many images from the CIFAR-10 and ImageNet test sets.

Just like for clean images, we found that most noisy images are both correctly classified and very close to a visually similar image which is not. (See Figure 4 .)As we mentioned in Section 4, it is not actually possible to compute * q precisely for the error set of a neural network, so we again report an estimate.

For each test image, we took 1,000 samples from the corresponding Gaussian and estimated * q using PGD with 200 steps on each sample and reported the median.

We find that for the five models we considered on CIFAR-10 and ImageNet, the relationship between our estimate of * q (E) and P x∼q [x ∈ E] is already close to optimal.

This is visualized in Figure 5 .

Note that in both cases, adversarial training does improve robustness to small perturbations, but the gains are primarily because error rates in Gaussian noise were dramatically improved, and less because the surface area of the error set was decreased.

In particular, many test points do not appear on these graphs because error rates in noise were so low that we did not find any errors among the 100,000 samples we used.

For example, for the naturally trained CIFAR model, about 1% of the points lie off the left edge of the plot, compared to about 59% for the adversarially trained model and 70% for the model trained on noise.

This shows that adversarial training on small perturbations improved generalization to large random perturbations, as the isoperimetric inequality says it must.

Figure 5: These plots give two ways to visualize the relationship between the error rate in noise and the distance from noisy points to the decision boundary (found using PGD).

Each point on each plot represents one image from the test set.

On the left, we compare the error rate of the model on Gaussian perturbations at σ = 0.1 to the distance from the median noisy point to its nearest error.

On the right, we compare the σ at which the error rate is 0.01 to this same median distance.

(The plots on the right are therefore similar to the plots in FIG0 .)

The thick black line at the top of each plot is the upper bound provided by the Gaussian isoperimetric inequality.

We include data from a model trained on clean images, an adversarially trained model, and a model trained on Gaussian noise (σ = 0.4.)

As mentioned in Section 1, we were unable to run this experiment on an adversarially robust ImageNet model.

Not all models or functions will be this close to optimal.

As a simple example, if we took one of the CIFAR models shown in Figure 5 and modified it so that the model outputs an error whenever each coordinate of the input is an integer multiple of 10 −6 , the resulting model would have an error within 1 2 · 10 −6 · dim(CIFAR) ≈ 0.039 of every point.

In this case, adversarial examples would be a distinct phenomenon from test performance, since * q (E) would be far from optimal.

The contrast between these two settings is important for adversarial defense design.

If adversarial examples arose from a badly behaved decision boundary (as in the latter case), then it would make sense to design defenses which attempt to smooth out the decision boundary in some way.

However, because we observe that image models are already close to the optimal bound on robustness for a fixed error rate in noise, future defense design should attempt to improve generalization in noise.

Currently there is a considerable subset of the adversarial defense literature which develops methods that would remove any small "pockets" of errors but which don't improve model generalization.

One example is BID33 which proposes randomly resizing the input to the network as a defense strategy.

Unfortunately, this defense, like many others, has been shown to be ineffective against stronger adversaries BID5 b; BID0 .

We proved a fundamental relationship between generalization in noisy image distributions and the existence of small adversarial perturbations.

By appealing to the Gaussian isoperimetric inequality, we formalized the notion of what it means for a decision boundary to be badly behaved.

We showed that, for noisy images, there is very little room to improve robustness without also decreasing the volume of the error set, and we provided evidence that small perturbations of clean images can also be explained in a similar way.

These results show that small-perturbation adversarial robustness is closely related to generalization in the presence of noise and that future defense efforts can measure progress by measuring test error in different noise distributions.

Indeed, several such noise distributions have already been proposed, and other researchers have developed methods which improve generalization in these distributions BID17 BID12 a; BID30 BID35 .

Our work suggests that adversarial defense and improving generalization in noise involve attacking the same set of errors in two different ways -the first community tries to remove the errors on the boundary of the error set while the second community tries to reduce the volume of the error set.

The isoperimetric inequality connects these two perspectives, and suggests that improvements in adversarial robustness should result in improved generalization in noise and vice versa.

Adversarial training on small perturbations on CIFAR-10 also improved generalization in noise, and training on noise improved robustness to small perturbations.

In the introduction we referred to a question from BID28 about why we find errors so close to our test points while the test error itself is so low.

We can now suggest an answer: despite what our low-dimensional visual intuition may lead us to believe, these errors are not in fact unnaturally close given the error rates we observe in noise.

There is a sense, then, in which we simply haven't reduced the test error enough to expect to have removed most nearby errors.

While we focused on the Gaussian distribution, similar conclusions can be made about other distributions.

In general, in high dimensions, the -boundary measure of a typical set is large even when its volume is small, and this observation does not depend on anything specific about the Gaussian distribution.

The Gaussian distribution is a special case in that we can easily prove that all sets will have large -boundary measure.

BID23 proved a similar theorem for a larger class of distributions.

For other data distributions not every set has large -boundary measure, but under some additional assumptions it still holds that most sets do.

An investigation of this relationship on the MNIST distribution can be found in Gilmer et al. (2018b, Appendix G) .We believe it would be beneficial for the adversarial defense literature to start reporting generalization in noisy image distributions, such as the common corruption benchmark introduced in BID17 , rather than the current practice of only reporting empirical estimates of adversarial robustness.

There are several reasons for this recommendation.1.

Measuring test error in noise is significantly easier than measuring adversarial robustnesscomputing adversarial robustness perfectly requires solving an NP-hard problem for every point in the test set BID19 .

Since BID28 , hundreds of adversarial defense papers have been published.

To our knowledge, only one BID22 has reported robustness numbers which were confirmed by a third party.

We believe the difficulty of measuring robustness under the usual definition has contributed to this unproductive situation.

2.

Measuring test error in noise would also allow us to determine whether or not these methods improve robustness in a trivial way, such as how the robust MNIST model learned to threshold the input, or whether they have actually succeeded in improving generalization outside the natural data distribution.

3.

All of the failed defense strategies we examined failed to improve generalization in noise.

For this reason, we should be highly skeptical of defense strategies that only claim improved l p -robustness but do not demonstrate robustness in more general settings.

4. Finally, if the goal is improving the security of our models in adversarial settings, errors in the presence of noise are already indicative that our models are not secure.

Until our models are perfectly robust in the presence of average-case corruptions, they will not be robust in worst-case settings.

The usefulness of l p -robustness in realistic threat models is limited when attackers are not constrained to making small modifications.

The interest in measuring l p robustness arose from a sense of surprise that errors could be found so close to correctly classified points.

But from the perspective described in this paper, the phenomenon is less surprising.

Statistical classifiers make a large number of errors outside the data on which they were trained, and small adversarial perturbations are simply the nearest ones.

Table 3 : The models from Section 1 trained and tested on ImageNet with Gaussian noise with standard deviation σ; the column labeled 0 refers to a model trained only on clean images.

Models trained on CIFAR-10.

We trained the Wide-ResNet-28-10 model BID34 using standard data augmentation of flips, horizontal shifts and crops in addition to Gaussian noise independently sampled for each image in every minibatch.

The models were trained with the open-source code by Cubuk et al. (2018) for 200 epochs, using the same hyperparameters which we summarize here: a weight decay of 5e-4, learning rate of 0.1, batch size of 128.

The learning rate was decayed by a factor of 0.2 at epochs 60, 120, 160.Models trained on ImageNet.

The ResNet-50 model (He et al., 2016) was trained with a learning rate of 1.6, batch size of 4096, and weight decay of 1e-4.

During training, random crops and horizontal flips were used, in addition to the Gaussian noise independently sampled for each image in every minibatch.

The models were trained for 90 epochs, where the learning rate was decayed by a factor of 0.1 at epochs 30, 60, and 80.

Learning rate was linearly increased from 0 to the value of 1.6 over the first 5 epochs.

Here we provide more detail for the noise distributions considered in Section 5.

The stAdv attack defines a flow field over the pixels of the image and shifts the pixels according to this flow.

The field is parameterized by a latent Z. When we measure accuracy against our randomized variant of this attack, we randomly sample Z from a multivariate Gaussian distribution with standard deviation σ.

To implement this attack we used the open sourced code from BID32 .

PCA-100 noise first samples noise from a Gaussian distribution N (0, σ), and then projects this noise onto the first 100 PCA components of the data.

For ImageNet, the input dimension is too large to perform a PCA decomposition on the entire dataset.

So we first perform a PCA decomposition on 30x30x1 patches taken from different color channels of the data.

To general the noise we first sample from a 900 dimensional Gaussian, then project this into the basis spanned by the top 100 PCA components, then finally tile this projects to the full 299x299 dimension of the input.

Each color channel is constructed independently in this fashion.

In Section 5, we mentioned that it is not trivial to learn the distribution of noisy images simply by augmenting the training data distribution.

In TAB2 we present more information about the performance of the models we trained and tested on various scales of Gaussian noise.

MNIST is a special case when it comes to the relationship between small adversarial perturbations and generalization in noise.

Indeed prior has already observed that an MNIST model can trivially become robust to small l ∞ perturbations by learning to threshold the input BID26 , and observed that the model from BID22 indeed seems to do this.

When we investigated this model in different noise distributions we found it generalizes worse than a naturally trained model, results are shown in TAB4 .

Given that it is possible for a defense to overfit to a particular l p metric, future work would be strengthened by demonstrating improved generalization outside the natural data distribution.

Here we will discuss the Gaussian isoperimetric inequality more thoroughly than we did in the text.

We will present some of the geometric intuition behind the theorem, and in the end we will show how the version quoted in the text follows from the form in which the inequality is usually stated.

The historically earliest version of the isoperimetric inequality, and probably the easiest to understand, is about areas of subsets of the plane and has nothing to do with Gaussians at all.

It is concerned with the following problem: among all measurable subsets of the plane with area A, which ones have the smallest possible perimeter?

1 One picture to keep in mind is to imagine that you are required to fence off some region of the plane with area A and you would like to use as little fence as possible.

The isoperimetric inequality says that the sets which are most "efficient" in this sense are balls.

Some care needs to be taken with the definition of the word "perimeter" here -what do we mean by the perimeter of some arbitrary subset of R 2 ?

The definition that we will use involves the concept of the -boundary measure we discussed in the text.

For any set E and any > 0, recall that we defined the -extension of E, written E , to be the set of all points which are within of a point in E; writing A(E) for the area of E, we then define the perimeter of E to be surf(E) := lim inf →0

A good way to convince yourself that this is reasonable is to notice that, for small , E − E looks like a small band around the perimeter of E with width .

The isoperimetric inequality can then be formally expressed as giving a bound on the quantity inside the limit in terms of what it would be for a ball.

(This is slightly stronger than just bounding the perimeter, that is, bounding the limit itself, but this stronger version is still true.)

That is, for any measurable set E ⊆ R 2 , DISPLAYFORM0 It is a good exercise to check that we have equality here when E is a ball.

There are many generalizations of the isoperimetric inequality.

For example, balls are also the subsets in R n which have minimal surface area for a given fixed volume, and the corresponding set on the surface of a sphere is a "spherical cap," the set of points inside a circle drawn on the surface of the sphere.

The version we are most concerned with in this paper is the generalization to a Gaussian distribution.

Rather than trying to relate the volume of E to the volume of E , the Gaussian Figure 6 : The Gaussian isoperimetric inequality relates the amount of probability mass contained in a set E to the amount contained in its -extension E .

A sample from the Gaussian is equally likely to land in the pink set on the left or the pink set on the right, but the set on the right has a larger -extension.

The Gaussian isoperimetric inequality says that the sets with the smallest possible -extensions are half spaces.isoperimetric inequality is about the relationship between the probability that a random sample from the Gaussian distribution lands in E or E .

Other than this, though, the question we are trying to answer is the same: for a given probability p, among all sets E for which the probability of landing in E is p, when is the probability of landing in E as small as possible?The Gaussian isoperimetric inequality says that the sets that do this are half spaces.

(See Figure 6 .)

Just as we did in the plane, it is convenient to express this as a bound on the probability of landing in E for an arbitrary measurable set E. This can be stated as follows: Theorem.

Consider the standard normal distribution q on R n , and let E be a measurable subset of R n .

Write DISPLAYFORM1 the cdf of the one-variable standard normal distribution.

For a measurable subset E ⊆ R n , write α(E) = Φ −1 (P x∼q [x ∈ E]).

Then for any ≥ 0, DISPLAYFORM2 The version we stated in the text involved * q (E), the median distance from a random sample from q to the closest point in E. This is the same as the smallest for which P x∼q [x ∈ E ] = 1 2 .

So, when = * q (E), the left-hand side of the Gaussian isoperimetric inequality is 1 2 , giving us that DISPLAYFORM3 is a strictly increasing function, applying it to both sides preserves the direction of this inequality.

But Φ −1 ( 1 2 ) = 0, so we in fact have that * q (E) ≤ −α, which is the statement we wanted.

The optimal bound according to the isoperimetric inequality gives surprisingly strong bounds in terms of the existence of worst-case l 2 perturbations and error rates in Gaussian noise.

In FIG2 we plot the optimal curves for various values of σ, visualize images sampled from x + N (0, σ), and visualize images at various l 2 distance from the unperturbed clean image.

Even for very large noise (σ = .6), test error needs to be less than 10 −15 in order to have worst-case perturbations be larger than 5.0.

In order to visualize worst-case perturbations at varying l 2 distances, we visualize an image that minimizes similarity according to the SSIM metric BID31 ).

These images are found by performing gradient descent to minimize the SSIM metric subject to the containt that ||x − x adv || 2 < .

The optimal curves on Imagenet for different values of σ.

Middle: Visualizing different coordinates of the optimal curves.

First, random samples from x + N (0, σI) for different values of σ.

Bottom: Images at different l 2 distances from the unperturbed clean image.

Each image visualized is the image at the given l 2 distance which minimizes visual similarity according to the SSIM metric.

Note that images at l 2 < 5 have almost no perceptible change from the clean image despite the fact that SSIM visual similarity is minimized.

In this section we include many more visualizations of the sorts of church window plots we discussed briefly in Section 4.

We will show an ordinarily trained model's predictions on several different slices through the same CIFAR test point which illustrate different aspects of the story told in this paper.

These images are best viewed in color.

Figure 8 : A slice through a clean test point (black, center image), the closest error found using PGD (blue, top image), and a random error found using Gaussian noise (red, bottom image).

For this visualization, and all others in this section involving Gaussian noise, we used noise with σ = 0.05, at which the error rate was about 1.7%.

In all of these images, the black circle indicates the distance at which the typical such Gaussian sample will lie.

The plot on the right shows the probability that the model assigned to its chosen class.

Green indicates a correct prediction, gray or white is an incorrect prediction, and brighter means more confident.

Figure 9 : A slice through a clean test point (black, center image), the closest error found using PGD (blue, top image), and the average of a large number of errors randomly found using Gaussian noise (red, bottom image).

The distance from the clean image to the PGD error was 0.12, and the distance from the clean image to the averaged error was 0.33.

The clean image is assigned the correct class with probability 99.9995% and the average and PGD errors are assigned the incorrect class with probabilities 55.3% and 61.4% respectively.

However, it is clear from this image that moving even a small amount into the orange region will increase these latter numbers significantly.

For example, the probability assigned to the PGD error can be increased to 99% by moving it further from the clean image in the same direction by a distance of 0.07.

FIG0 : A slice through a clean test point (black, center image), a random error found using Gaussian noise (blue, top image), and the average of a large number of errors randomly found using Gaussian noise (red, bottom image).

FIG0 : A slice through a clean test point (black, center image) and two random errors found using Gaussian noise (blue and red, top and bottom images).

Note that both random errors lie very close to the decision boundary, and in this slice the decision boundary does not appear to come close to the clean image.

The cdf of the error rates in noise for images in the test set.

The blue curve corresponds to a model trained and tested on noise with σ = 0.1, and the green curve is for a model trained and tested at σ = 0.3.

For example, the left most point on the blue curve indicates that about 40% of test images had an error rate of at least 10 −3 .

FIG0 : Some visualizations of the same phenomenon, but using the "pepper noise" discussed in Section 5 rather than Gaussian noise.

In all of these visualizations, we see the slice through the clean image (black, center image), the same PGD error as above (red, bottom image), and a random error found using pepper noise (blue, top image).

In the visualization on the left, we used an amount of noise that places the noisy image further from the clean image than in the Gaussian cases we considered above.

In the visualization in the center, we selected a noisy image which was assigned to neither the correct class nor the class of the PGD error.

In the visualization on the right, we selected a noisy image which was assigned to the same class as the PGD error.

Using some of the models that were trained on noise, we computed, for each image in the CIFAR test set, the probably that a random Gaussian perturbation will be misclassified.

A histogram is shown in FIG0 .

Note that, even though these models were trained on noise, there are still many errors around most images in the test set.

While it would have been possible for the reduced performance in noise to be due to only a few test points, we see clearly that this is not the case.

In this section we first show a collection of iid test errors for the ResNet-50 model on the ImageNet validation set.

We also visualize the severity of the different noise distributions considered in this work, along with model errors found by random sampling in these distributions.

@highlight

Small adversarial perturbations should be expected given observed error rates of models outside the natural data distribution.

@highlight

This paper proposes an alternative view for adversarial examples in high dimension spaces by considering the "error rate" in a Gaussian distribution centered at each test point.