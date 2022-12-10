Generative modeling of high dimensional data like images is a notoriously difficult and ill-defined problem.

In particular, how to evaluate a learned generative model is unclear.

In this paper, we argue that *adversarial learning*, pioneered with generative adversarial networks (GANs), provides an interesting framework to implicitly define more meaningful task losses for unsupervised tasks, such as for generating "visually realistic" images.

By relating GANs and structured prediction under the framework of statistical decision theory, we put into light links between recent advances in structured prediction theory and the choice of the divergence in GANs.

We argue that the insights about the notions of "hard" and "easy" to learn losses can be analogously extended to adversarial divergences.

We also discuss the attractive properties of parametric adversarial divergences for generative modeling, and perform experiments to show the importance of choosing a divergence that reflects the final task.

For structured prediction and data generation the notion of final task is at the same time crucial and not well defined.

Consider machine translation; the goal is to predict a good translation, but even humans might disagree on the correct translation of a sentence.

Moreover, even if we settle on a ground truth, it is hard to define what it means for a candidate translation to be close to the ground truth.

In the same way, for data generation, the task of generating pretty pictures or more generally realistic samples is not well defined.

Nevertheless, both for structured prediction and data generation, we can try to define criteria which characterize good solutions such as grammatical correctness for translation or non-blurry pictures for image generation.

By incorporating enough criteria into a task loss, one can hope to approximate the final task, which is otherwise hard to formalize.

Supervised learning and structured prediction are well-defined problems once they are formulated as the minimization of such a task loss.

The usual task loss in object classification is the generalization error associated with the classification error, or 0-1 loss.

In machine translation, where the goal is to predict a sentence, a structured loss, such as the BLEU score BID37 , formally specifies how close the predicted sentence is from the ground truth.

The generalization error is defined through this structured loss.

In both cases, models can be objectively compared and evaluated with respect to the task loss (i.e., generalization error).

On the other hand, we will show that it is not as obvious in generative modeling to define a task loss that correlates well with the final task of generating realistic samples.

Traditionally in statistics, distribution learning is formulated as density estimation where the task loss is the expected negative-log-likelihood.

Although log-likelihood works fine in low-dimension, it was shown to have many problems in high-dimension .

Among others, because the Kullback-Leibler is too strong of a divergence, it can easily saturate whenever the distributions are too far apart, which makes it hard to optimize.

Additionally, it was shown in BID47 that the KL-divergence is a bad proxy for the visual quality of samples.

In this work we give insights on how adversarial divergences BID26 can be considered as task losses and how they address some problems of the KL by indirectly incorporating hard-to-define criteria.

We define parametric adversarial divergences as the following : DISPLAYFORM0 where {f φ : X → R d ; φ ∈ Φ} is a class of parametrized functions, such as neural networks, called the discriminators in the Generative Adversarial Network (GAN) framework BID15 .

The constraints Φ and the function ∆ : R d × R d → R determine properties of the resulting divergence.

Using these notations, we adopt the view 1 that training a GAN can be seen as training a generator network q θ (parametrized by θ) to minimize the parametric adversarial divergence Div NN (p||q θ ), where the generator network defines the probability distribution q θ over x.

Our contributions are the following:• We show that compared to traditional divergences, parametric adversarial divergences offer a good compromise in terms of sample complexity, computation, ability to integrate prior knowledge, flexibility and ease of optimization.• We relate structured prediction and generative adversarial networks using statistical decision theory, and argue that they both can be viewed as formalizing a final task into the minimization of a statistical task loss.• We explain why it is necessary to choose a divergence that adequately reflects our final task in generative modeling.

We make a parallel with results in structured learning (also dealing with high-dimensional data), which quantify the importance of choosing a good objective in a specific setting.• We explore with some simple experiments how the properties of the discriminator transfer to the adversarial divergence.

Our experiments suggest that parametric adversarial divergences are especially adapted to problems such as image generation, where it is hard to formally define a perceptual loss that correlates well with human judgment.• We illustrate the importance of having a parametric discriminator by running experiments with the true (nonparametric) Wasserstein, and showing its shortcomings on complex datasets, on which GANs are known to perform well.• We perform qualitative and quantitative experiments to compare maximum-likelihood and parametric adversarial divergences under two settings: very high-dimensional images, and learning data with specific constraints.

Here we briefly introduce the structured prediction framework because it can be related to generative modeling in some ways.

We will later link them formally, and present insights from recent theoretical results to choose a better divergence.

We also unify parametric adversarial divergences with traditional divergences in order to compare them in the next section.

The goal of structured prediction is to learn a classifier h θ : X → Y which predicts a structured output y from an input x. The key difficulty is that Y usually has size exponential in the input 2 (e.g. it could be all possible sequence of symbols with a given length).

Being able to handle this exponentially large set of outputs is one of the key challenges in structured prediction because it makes traditional multi-class classification methods unusable in general.3 Standard practice in structured prediction BID46 BID9 BID38 is to consider predictors based on score functions h θ (x) = arg max y ∈Y s θ (x, y ), where s θ : X × Y → R, called the score/energy function BID23 , assigns a score to each possible label y for an input x. Typically, 1 We focus in this paper on the divergence minimization perspective of GANs.

There are other views, such as those based on game theory BID2 , ratio matching and moment matching BID29 .2 Additionally, Y might depend on the input x, but we ignore this effect for clarity of exposition.

3 Such as ones based on maximum likelihood.as in structured SVMs BID46 , the score function is linear: s θ (x, y) = θ, g(x, y) , where g(·) is a predefined feature map.

Alternatively, the score function could also be a learned neural network BID5 .In order to evaluate the predictions objectively, we need to define a task-dependent structured loss (y , y ; x) which expresses the cost of predicting y for x when the ground truth is y. We discuss the relation between the loss function and the actual final task in Section 4.2 .

The goal is then to find a parameter θ which minimizes the generalization error: DISPLAYFORM0 Directly minimizing (2) is often an intractable problem; this is the case when the structured loss is the 0-1 loss BID1 .

Instead, the usual practice is to minimize a surrogate loss et al., 2006) which has nicer properties, such as subdifferentiability or convexity, to get a tractable optimization problem.

The surrogate loss is said to be consistent when its minimizer is also a minimizer of the task loss.

DISPLAYFORM1 A simple example of structured prediction task is machine translation.

Suppose we want to translate French sentences to English; the input x is then a sequence of French words, and the output y is a sequence of English words belonging to a dictionary D with typically |D| ≈ 10000 words.

If we restrict the output sequence to be shorter than T words, then |Y| = |D| T , which is exponential.

An example of desirable criterion is to have a translation with many words in common with the ground truth, which is typically enforced using BLEU scores to define the task loss.

Because we will compare properties of adversarial and traditional divergences throughout this paper, we choose to first unify them with a formalism similar to BID45 ; BID26 : DISPLAYFORM0 Under this framework we give some examples of traditional nonparametric divergences:• ψ-divergences with generator function ψ (which we call f-divergences) can be written in dual form BID33 4 Div ψ (p||q θ ) = sup DISPLAYFORM1 where ψ * is the convex conjugate.

Depending on ψ, one can obtain any ψ-divergence such as the (reverse) Kullback-Leibler, the Jensen-Shannon, the Total Variation, the ChiSquared 5 .•

Wasserstein-1 distance induced by an arbitrary norm · and its corresponding dual norm · * BID45 : DISPLAYFORM2 which can be interpreted as the cost to transport all probability mass of p into q, where x − x is the unit cost of transporting x to x .•

Maximum Mean Discrepancy : DISPLAYFORM3 where (H, K) is a Reproducing Kernel Hilbert Space induced by a Kernel K(x, x ) on X with the associated norm · H .

The MMD has many interpretations in terms of momentmatching BID24 .

4 The standard form is Ex∼q θ [ψ( DISPLAYFORM4 .

Some ψ require additional constraints, such as ||f ||∞ ≤ 1 for the Total Variation.

Table 1 : Properties of Divergences.

Explicit and Implicit models refer to whether the density q θ (x) can be computed.

p is the number of parameters of the parametric discriminator.

Sample complexity and computational cost are defined and discussed in Section 3.1, while the ability to integrate desirable properties of the final loss is discussed in Section 3.2.

Although f-divergences can be estimated with Monte-Carlo for explicit models, they cannot be easily computed for implicit models without additional assumptions (see text).

Additionally, by design, they cannot integrate a final loss directly.

The nonparametric Wasserstein can be computed iteratively with the Sinkhorn algorithm, and can integrate the final loss in its base distance, but requires exponentially many samples to estimate.

Maximum Mean Discrepancy has good sample complexity, can be estimated analytically, and can integrate the final loss in its base distance, but it is known to lack discriminative power for generic kernels, as discussed below.

Parametric adversarial divergences have reasonable sample complexities, can be computed iteratively with SGD, and can integrate the final loss in the choice of class of discriminators.

DISPLAYFORM5 In particular, the parametric Wasserstein has the additional possibility of integrating the final loss into the base distance.

In the optimization problems FORMULA4 and FORMULA5 , whenever f is additionally constrained to be in a given parametric family, the associated divergence will be termed a parametric adversarial divergence.

In practice, that family will typically be specified as a neural network architecture, so in this work we will use the term neural adversarial divergences interchangeably with the slightly more generic parametric adversarial divergence.

For instance, the parametric adversarial Jensen-Shannon optimized in GANs corresponds to (4) with specific ψ BID33 , while the parametric adversarial Wasserstein optimized in WGANs corresponds to (5) where f is a neural network.

See BID26 for interpretations and a review and interpretation of other divergences like the Wasserstein with entropic smoothing BID3 , energy-based distances BID24 which can be seen as adversarial MMD, and the WGAN-GP BID18 objective.

We argue that parametric adversarial divergences have many good properties which make them attractive for generative modeling.

In this section, we compare them to traditional divergences in terms of sample complexity and computational cost (Section 3.1), and ability to integrate criteria related to the final task (Section 3.2).

We also discuss the shortcomings of combining the KL-divergence with generators that have a special structure in Section 3.3.

We refer the reader to the Appendix for additional interesting properties of parametric adversarial divergences: the optimization and stability issues are discussed in Appendix A.1, the fact that parametric adversarial divergences only make the assumption that one can sample from the generative model, and provide useful learning signal even when their nonparametric counterparts are not well-defined, is discussed in Appendix A.2.

Since we want to learn from finite data, we would like to know how well empirical estimates of a divergence approximate the population divergence.

In other words, we want to control the sample complexity, that is, how many samples n do we need to have with high probability that |Div(p||q) − Div( p n || q n )| ≤ , where > 0, and p n , q n are empirical distributions associated with p, q. Sample complexities for adversarial and traditional divergences are summarized in Table 1 .For explicit models which allow evaluating the density q θ (x), one could use Monte-Carlo to evaluate the f-divergence with sample complexity n = O(1/ 2 ), according to the Central-Limit theorem.

For implicit models, there is no one good way of estimating f-divergences from samples.

There are some techniques for it BID32 BID30 BID43 , but they all make additional assumptions about the underlying densities (such as smoothness), or they solve the dual in a restricted family, such as a RKHS, which makes the divergences no longer f-divergences.

Parametric adversarial divergences can be formulated as a classification/regression problem with a loss depending on the specific adversarial divergence.

Therefore, they have a reasonable sample complexity of O(p/ 2 ), where p is the VC-dimension/number of parameters of the discriminator BID2 , and can be solved using classic stochastic gradient methods.

A straightforward nonparametric estimator of the Wasserstein is simply the Wasserstein distance between the empirical distributions p n and q n , for which smoothed versions can be computed in O(n 2 ) using specialized algorithms such as Sinkhorn's algorithm BID11 or iterative Bregman projections BID7 .

However, this empirical Wasserstein estimator has sample complexity n = O(1/ d+1 ) which is exponential in the number of dimensions (see Sriperumbudur et al., 2012, Corollary 3.5) .

Thus the empirical Wasserstein is not a viable estimator in high-dimensions.

Maximum Mean Discrepancy admits an estimator with sample complexity n = O(1/ 2 ), which can be computed analytically in O(n 2 ).

More details are given in the original MMD paper BID16 .

One should note that MMD depends fundamentally on the choice of kernel.

As the sample complexity is independent of the dimension of the data, one might believe that the MMD estimator behaves well in high dimensions.

However, it was experimentally illustrated in Dziugaite et al. FORMULA0 that with generic kernels like RBF, MMD performs poorly for MNIST and Toronto face datasets, as the generated images have many artifacts and are clearly distinguishable from the training dataset.

See Section 3.2 for more details on the choice of kernel.

It was also shown theoretically in BID40 that the power of the MMD statistical test can drop polynomially with increasing dimension, which means that with generic kernels, MMD might be unable to discriminate well between high-dimensional generated and training distributions.

Note that comparing divergences in terms of sample complexity can give good insights on what is a good divergence, but should be taken with a grain of salt as well.

On the one hand, the sample complexities we give are upper-bounds, which means the estimators could potentially converge faster.

On the other hand, one might not need a very good estimator of the divergence in order to learn in some cases.

This is illustrated in our experiments with the empirical Wasserstein (Section 6) which has bad sample complexity but yields reasonable results.

In Section 4, we will argue that in structured prediction, optimizing for the right task losses is more meaningful and can make learning considerably easier.

Similarly in generative modeling, we would like divergences to integrate criteria that characterize the final task.

We discuss that although not all divergences can easily integrate final task-related criteria, adversarial divergences provide a way to do so.

Pure f-divergences cannot directly integrate any notion of final task, 6 at least not without tweaking the generator.

The Wasserstein distance and MMD are respectively induced by a base metric d(x, x ) and a kernel K(x, x ).

The metric and kernel give us the opportunity to specify a task by letting us express a (subjective) notion of similarity.

However, the metric and kernel generally have to be defined by hand, as there is no obvious way to learn them end-to-end.

For instance, BID14 learn to generate MNIST by minimizing a smooth Wasserstein based on the L2-distance, while Dziugaite et al. FORMULA0 ; BID25 also learn to generate MNIST by minimizing the MMD induced by kernels obtained externally: either generic kernels based on the L2-distance or on autoencoder features.

However, the results seems to be limited to simple datasets.

Recently there has been a surge of interest in combining MMD with kernel learning, with convincing results on LSUN, CelebA and ImageNet images.

BID31 learn a feature map and try to match its mean and covariance, BID24 learn kernels end-to-end, while BID6 do end-to-end learning of energy distances, which are closely related to MMD.Parametric adversarial divergences are defined with respect to a parametrized class of discriminators, thus changing properties of the discriminator is a primary way to affect the associated divergence.

The form of the discriminator may determine what aspects the divergence will be sensitive or blind to.

For instance using a convolutional network as the discriminator may render the divergence insen-sitive to small image translations.

Additionally, the parametric adversarial Wasserstein distance can also incorporate a custom metric.

In Section 6 we give interpretations and experiments to assess the relation between the discriminator and the divergence.

In some cases, imposing a certain structure on the generator (e.g. a Gaussian or Laplacian observation model) yields a Kullback-Leibler divergence which involves some form of component-wise distance between samples, reminiscent of the Hamming loss (see Section 4.3) used in structured prediction.

However, doing maximum likelihood on generators having an imposed special structure can have drawbacks which we detail here.

For instance, the generative model of a typical variational autoencoder can be seen as an infinite mixture of Gaussians BID19 .

The loglikelihood thus involves a "reconstruction loss", a pixel-wise L2 distance between images analogous to the Hamming loss, which makes the training relatively easy and very stable.

However, the Gaussian is partly responsible for the VAE's inability to learn sharp distributions.

Indeed it is a known problem that VAEs produce blurry samples , in fact even if the approximate posterior matches exactly the true posterior, which would correspond to the evidence lower-bound being tight, the output of the VAE would still be blurry .

Other examples are autoregressive models such as recurrent neural networks BID28 which factorize naturally as log q θ (x) = i log q θ (x i |x 1 , .., x i−1 ), and PixelCNNs BID34 .

Training autoregressive models using maximum likelihood results in teacher-forcing BID21 : each ground-truth symbol is fed to the RNN, which then has to maximize the likelihood of the next symbol.

Since teacher-forcing induces a lot of supervision, it is possible to learn using maximumlikelihood.

Once again, there are similarities with the Hamming loss because each predicted symbol is compared with its associated ground truth symbol.

However, among other problems, there is a discrepancy between training and generation.

Sampling from q θ would require iteratively sampling each symbol and feeding it back to the RNN, giving the potential to accumulate errors, which is not something that is accounted for during training.

See BID22 and references therein for more principled approaches to sequence prediction with autoregressive models.

In this section, we try to provide insights in order to design the best adversarial divergence for our final task.

After establishing the relationship between structured prediction and generative adversarial networks, we review theoretical results on the choice of objectives in structured prediction, and discuss their interpretation in generative modeling.

We frame the relationship of structured prediction and GANs using the framework of statistical decision theory.

Assume that we are in a world with a set P of possible states and that we have a set A of actions.

When the world is in the state p ∈ P, the cost of playing action a ∈ A is the (statistical) task loss L p (a).

The goal is to play the action minimizing the task loss.

Generative models with Maximum Likelihood.

The set P of possible states is the set of available distributions {p} for the data x. The set of actions A is the set of possible distributions{q θ ; θ ∈ Θ} for the model and the task loss is the negative log-likelihood, DISPLAYFORM0 Structured prediction.

The set P of possible states is the set of available distribution {p} for (x, y).

The set of actions A is the set of prediction functions {h θ ; θ ∈ Θ} and the task loss is the generalization error: DISPLAYFORM1 where : Y × Y × X → R is a structured loss function.

the minimization of a statistical task loss.

One starts from a useful but illdefined final task, and devises criteria that characterize good solutions.

Such criteria are integrated into the statistical task loss, which is the generalization error in structured prediction, and the adversarial divergence in the GAN framework.

The hope is that minimizing the statistical task loss effectively solves the final task.

GANs.

The set P of possible states is the set of available distributions {p} for the data x. The set of actions A is the set of distributions {q θ ; θ ∈ Θ} that the generator can learn, and the task loss is the adversarial divergence DISPLAYFORM2 Under this unified framework, the prediction function h θ is analogous to the generative model q θ , while the choice of the right structured loss can be related to ∆ and to the choice of the discriminator family F which will induce a good adversarial divergence.

We will further develop this analogy in Section 4.2.

As discussed in the introduction, structured prediction and data generation involve a notion of final task which is at the same time crucial and not well defined.

Nevertheless, for both we can try to define criteria which characterize good solutions.

We would like the statistical task loss (introduced in Section 4.1), which corresponds to the generalization error in structured prediction, and the adversarial divergence in generative modeling, to incorporate task-related criteria.

One way to do that is to choose a structured loss that reflects the criteria of interest, or analogously to choose a class of discriminators, like a CNN architecture, such that the resulting adversarial divergence has good invariance properties.

The whole process of building statistical task losses adapted to a final task, using the right structured losses or discriminators, is represented in FIG0 .For many prediction problems, the structured prediction community has engineered structured loss functions which induce properties of interest on the learned predictors.

In machine translation, a commonly considered property of interest is for candidate translations to contain many words in common with the ground-truth; this has given rise to the BLEU score which counts the percentage of candidate words appearing in the ground truth.

In the context of image segmentation, BID35 have compared various structured loss functions which induces different properties on the predicted mask.

In the same vein as structured loss functions, adversarial divergences can be built to induce certain properties on the generated data.

We are more concerned with generating realistic samples than having samples which are very similar with the training set; we actually want to extrapolate some properties of the true distribution from the training set.

For instance, in the DCGAN , the discriminator has a convolutional architecture, which makes it potentially robust to small deformations that would not affect the visual quality of the samples significantly, while still making it able to detect blurry samples, which is aligned with our objective of generating realistic samples.

Intuition on the Flexibility of Losses.

In this section we get insights from the convergence results of in structured prediction.

They show in a specific setting that some "weaker" structured loss functions are easier to learn than some stronger loss functions.

In some sense, their results formalize the intuition in generative modeling that learning with "weaker" divergences is easier ) and more intuitive BID26 than stronger divergences.

In structured prediction, strong losses such as the 0-1 loss are hard to learn with because they do not give any flexibility on the prediction; the 0-1 loss only tells us whether a prediction is correct or not, and consequently does not give any clue about how close the prediction is to the ground truth.

To get enough learning signal, we roughly need as many training examples as the number of possible outputs |Y|, which is exponential in the dimension of y and thus inefficient.

Conversely, weaker losses like the Hamming loss have more flexibility; because they tell us how close a prediction is to the ground truth, less examples are needed to generalize well.

The theoretical results proved by formalize that intuition in a specific setting.

Theory to Back the Intuition.

In a non-parametric setting (details and limitations in Appendix B), formalize the intuition that weaker structured loss functions are easier to optimize.

Specifically, they compare the 0-1 loss 0−1 (y, y ) =1 {y = y } to the Hamming lossHam (y, y ) = 1 T T t=1 1{y t = y t }, when y decomposes as T = log 2 |Y| binary variables (y t ) 1≤t≤T .

They derive a worst case sample complexity needed to obtain a fixed error > 0.

For the 0-1 loss, they obtain a sample complexity of O(|Y|/ 2 ) which is exponential in the dimension of y. However, for the Hamming loss, under certain constraints (see , section on exact calibration functions) they obtain a much better sample complexity of O(log 2 |Y|/ 2 ) which is polynomial in the number of dimensions, whenever certain constraints are imposed on the score function.

Thus their results suggest that choosing the right structured loss, like the weaker Hamming loss, might make training exponentially faster.

Insights and Relation with Adversarial Divergences. 's theoretical results confirm our intuition that weaker losses are easier to optimize, and quantify in a specific setting how much harder it is to learn with strong structured loss functions, like the 0-1 loss, than with weaker ones, like the Hamming loss (here, exponentially harder).

Under the framework of statistical decision theory (introduced Section 4.1), their results can be related to analogous results in generative modeling BID26 showing that it can be easier to learn with weaker divergences than with stronger ones.

In particular, one of their arguments is that distributions with disjoint support can be compared in weaker topologies like the the one induced by the Wasserstein but not in stronger ones like the the one induced by the Jensen-Shannon.

Closest to our work are the following two papers.

BID2 argue that analyzing GANs with a nonparametric (optimal discriminator) view does not really make sense, because the usual nonparametric divergences considered have bad sample complexity.

They also prove sample complexities for parametric divergences.

BID26 prove under some conditions that globally minimizing a neural divergence is equivalent to matching all moments that can be represented within the discriminator family.

They unify parametric divergences with nonparametric divergences and introduce the notion of strong and weak divergence.

However, both those works do not attempt to study the meaning and practical properties of parametric divergences.

In our work, we start by introducing the notion of final task, and then discuss why parametric divergences can be good task losses with respect to usual final tasks.

We also perform experiments to determine properties of some parametric divergences, such as invariance, ability to enforce constraints and properties of interest, as well as the difference with their nonparametric counterparts.

Finally, we unify structured prediction and generative modeling, which could give a new perspective to the community.

The following papers are also related to our work because of one of the following aspects: unifying divergences, analyzing their statistical properties, giving other interpretations of generative modeling, improving GANs, criticizing maximum-likelihood as a objective for generative modeling, and other reasons.

Before the first GAN paper, BID45 unify traditional IPMs, analyze their statistical properties, and propose to view them as classification problems.

Similarly, BID41 show that computing a divergence can be formulated as a classification problem.

Later, BID33 generalize the GAN objective to any adversarial f-divergence.

However, the first papers to actually study the effect of restricting the discriminator to be a neural network instead of any function are the MMD-GAN papers: BID25 ; Dziugaite et al. FORMULA0 ; BID24 ; BID31 and BID6 who give an interpretation of their Figure 2 : Images generated by the network after training with the Sinkorn-Autodiff algorithm on MNIST dataset (left) and CIFAR-10 dataset (right).

One can observe than although the network succeeds in learning MNIST, it is unable to produce convincing and diverse samples on the more complex CIFAR-10.

energy distance framework in terms of moment matching.

BID29 give many interpretations of generative modeling, including moment-matching, divergence minimization, and density ratio matching.

On the other hand, work has been done to better understand the GAN objective in order to improve its stability BID44 .

Subsequently, introduce the adversarial Wasserstein distance which makes training much more stable, and BID18 improve the objective to make it more practical.

Regarding model evaluation, BID47 contains an excellent discussion on the evaluation of generative models, they show in particular that log-likelihood is not a good proxy for the visual quality of samples.

compare parametric adversarial divergence and likelihood objectives in the special case of RealNVP, a generator with explicit density, and obtain better visual results with the adversarial divergence.

Concerning theoretical understanding of learning in structured prediction, some recent papers are devoted to theoretical understanding of structured prediction such as BID10 and BID27 which propose generalization error bounds in the same vein as but with data dependencies.

One contribution of the present paper is to have taken these results from the prior literature and put them in perspective in an attempt to provide a more principled view of the nature and usefulness of parametric divergences, in comparison to traditional divergences.

To the best of our knowledge, we are also the first to make a link between the generalization error of structured prediction and the adversarial divergence in generative modeling.

Importance of Sample Complexity.

Since the sample complexity of the nonparametric Wasserstein is exponential in the dimension (Section 3.1), we check experimentally whether training a generator to minimize the nonparametric Wasserstein distance fails in high dimensions.

We implement the Sinkhorn-AutoDiff algorithm BID14 to compute the entropy-regularized L2-Wasserstein distance between minibatches of training images and generated images.

Figure 2 shows generated samples after training with the Sinkhorn-Autodiff algorithm on both MNIST and CIFAR-10 dataset.

On MNIST, the network manages to produce decent but blurry images.

However, on CIFAR-10, which is a much more complex dataset, the network fails to produce meaningful samples, which would suggest that indeed the nonparametric Wasserstein should not be used for generative modeling when the (effective) dimensionality is high.

This result is to be contrasted with the recent successes in image generation of the parametric Wasserstein BID18 , which also has much better sample complexity than the nonparametric Wasserstein.

Robustness to Transformations.

Intuitively, small rotations should not significantly affect the realism of images, while additive noise should.

We study the robustness of various parametric adversarial divergences to rotations and additive noise by plotting the evolution of the divergence between MNIST and rotated/noisy versions of it, as a function of the amplitude of transformation.

We consider three discriminators (linear, 1-layer-dense, 2-layer-cnn) combined with two formulations, parametric Jensen-Shannon (ParametricJS) and parametric Wasserstein (ParametricW).

Ideally, good divergences should vary smoothly (be robust) with respect to the amplitude of the transformation.

For rotations FIG2 ) and all discriminators except the linear, ParametricJS saturates at its maximal value, even for small values of rotation, whereas the Wasserstein distance varies much more smoothly, which is consistent with the example given by .

The fact that the linear ParametricJS does not saturate for rotations shows that the architecture of the discriminator has a significant effect on the induced parametric adversarial divergence, and confirms that there is a conceptual difference between the true JS and ParametricJS, and even among different ParametricJS.

For additive Gaussian noise FIG2 ), the linear discriminator is unable to distinguish the two distributions (it only sees the means of the distributions), whereas more complex architectures like CNNs do.

In that sense the linear discriminator is too weak for the task, or not strict enough BID26 , which suggests that a better divergence involves trading off between robustness and strength.

Learning High-dimensional Data.

We collect Thin-8, a dataset of about 1500 handwritten images of the digit "8", with a very high resolution of 512 × 512, and augment them with elastic deformations.

Because the pen strokes are relatively thin, we expect any pixel-wise distance to be uninformative, because the images are dominated by background pixels, and because with high probability, any two "8' will intersect on no more than a little area.

We train a convolutional VAE and a WGAN-GP BID18 , henceforth simply denoted GAN, using nearly the same architectures (VAE decoder similar to GAN generator, VAE encoder similar to GAN discriminator), with 16 latent variables, on the following resolutions: 32 × 32, 128 × 128 and 512 × 512.

Generated samples are shown in FIG3 .

Indeed, we observe that the VAE, trained to minimize the evidence lower bound on maximum-likelihood, fails to generate convincing samples in high-dimensions: they are blurry, pixel values are gray instead of being white, and some samples look like the average of many digits.

On the contrary, the GAN can generate sharp and realistic samples even in 512 × 512.

Our hypothesis is that the discriminator learns moments which are easier to match than it is to directly match the training set with maximum likelihood.

Since we were able to perfectly generate high-resolution digits, an additional insight of our experiment is that the main difficulty in generating high-dimensional natural images (like ImageNet and LSUN bedrooms) resides not in high resolution itself, but in the intrinsic complexity of the scenes.

Such complexity can be hidden in low resolution, which might explain recent successes in generating images in low resolution but not in higher ones.

Learning Visual Hyperplanes.

We design the visual hyperplane task to be able to compare VAEs and GANs quantitatively rather than simply inspecting the quality of their generated images.

We create a new dataset by concatenating sets of 5 images from MNIST, such that those digits sum up to 25.

We train a VAE and a WGAN-GP (henceforth simply denoted GAN) on this new dataset (we used 4504 combinations out of the 5631 possible combinations for training).

Both model share the same architecture for generator network and use 200 latent variables.

With the help of a MNIST classifier, we automatically recognize and sum up the digits in each generated sample.

FIG4 shows the distributions of the sums of the digits generated by the VAE and GAN 7 .

We can see that the GAN distribution is more peaked and centered around the target 25, while the VAE distribution is less precise and not centered around the target.

In that respect, the GAN was better than the VAE at capturing the particular aspects and constraints of the data distribution (summing up to 25).

One , and 512×512 (right column).

Note how the GAN samples are always crips and realistic across all resolutions, while the VAE samples tend to be blurry with gray pixel values in high-resolution.

We can also observe some averaging artifacts in the top-right 512x512 VAE sample, which looks like the average of two "8".

More samples can be found in Section C.2 of the Appendix.

and Independent Baseline (gray).

The latter draws digits independently according to their empirical marginal probabilities, which corresponds to fitting independent multinomial distributions over digits using maximum likelihood.

WGAN-GP beats largely both VAE and Indepedent Baseline as it gives a sharper distribution centered in the target sum 25.

possible explanation is that since training a classifier to recognize digits and sum them up is not hard in a supervised setting, it could also be relatively easy for a discriminator to enforce such a constraint.

We gave arguments in favor of using adversarial divergences rather than traditional divergences for generative modeling, the most important of which being the ability to account for the final task.

After linking structured prediction and generative modeling under the framework of statistical decision theory, we interpreted recent results from structured prediction, and related them to the notions of strong and weak divergences.

Moreover, viewing adversarial divergences as statistical task losses led us to believe that some adversarial divergences could be used as evaluation criteria in the future, replacing hand-crafted criteria which cannot usually be exhaustive.

In some sense, we want to extrapolate a few desirable properties into a meaningful task loss.

In the future we would like to investigate how to define meaningful evaluation criteria with minimal human intervention.

In this section, we describe additional advantages and properties of parametric adversarial divergences.

While adversarial divergences are learned and thus potentially much more powerful than traditional divergences, the fact that they are the solution to a hard, non-convex problem can make GANs unstable.

Not all adversarial divergences are equally stable: claimed that the adversarial Wasserstein gives more meaningful learning signal than the adversarial Jensen-Shannon, in the sense that it correlates well with the quality of the samples, and is less prone to mode dropping.

In Section 6 we will show experimentally on a simple setting that indeed the neural adversarial Wasserstein consistently give more meaningful learning signal than the neural adversarial JensenShannon, regardless of the discriminator architecture.

Similarly to the WGAN, the MMD-GAN divergence BID24 was shown to correlate well with the quality of samples and to be robust to mode collapse.

Recently, it was shown that neural adversarial divergences other than the Wasserstein can also be made stable by regularizing the discriminator properly BID20 BID42 .

Maximum-likelihood typically requires computing the density q θ (x), which is not possible for implicit models such as GANs, from which it is only possible to sample.

On the other hand, parametric adversarial divergences can be estimated with reasonable sample complexity (see Section 3.1) only by sampling from the generator, without any assumption on the form of the generator.

This is also true for MMD but generally not the case for the empirical Wasserstein, which has bad sample complexity as stated previously.

Another issue of f-divergences such as the Kullback-Leibler and the Jensen-Shannon is that they are either not defined (Kullback-Leibler) or uninformative (JensenShannon) when p is not absolutely continuous w.r.t.

q θ BID33 , which makes them unusable for learning sharp distributions such as manifolds.

On the other hand, some integral probability metrics, such as the Wasserstein, MMD, or their adversarial counterparts, are well defined for any distributions p and q θ .

In fact, even though the Jensen-Shannon is uninformative for manifolds, the parametric adversarial Jensen-Shannon used in the original GANs BID15 still allows learning realistic samples, even though the process is unstable BID44 .

Although give a lot of insights, their results must be taken with a grain of salt.

In this section we point out the limitations of their theory.

First, their analysis ignores the dependence on x and is non-parametric, which means that they consider the whole class of possible score functions for each given x. Additionally, they only consider convex consistent surrogate losses in their analysis, and they give upper bounds but not lower bounds on the sample complexity.

It is possible that optimizing approximately-consistent surrogate losses instead of consistent ones, or making additional assumptions on the distribution of the data could yield better sample complexities.

C EXPERIMENTAL RESULTS

Here, we compare the parametric adversarial divergences induced by three different discriminators (linear, dense, and CNN) under the WGAN-GP BID18 formulation.

We consider one of the simplest non-trivial generators, in order to factor out optimization issues on the generator side.

The model is a mixture of 100 Gaussians with zero-covariance.

The model density is q θ (x) = 1 K z δ(x − x z ), parametrized by prototypes θ = (x z ) 1≤z≤K .

The generative process consists in sampling a discrete random variable z ∈ {1, ..., K}, and returning the prototype x z .Learned prototypes (means of each Gaussian) are shown in Figure 6 and 7.

The first observation is that the linear discriminator is too weak of a divergence: all prototypes only learn the mean of the training set.

Now, the dense discriminator learns prototypes which sometimes look like digits, but are blurry or unrecognizable most the time.

The samples from the CNN discriminator are never blurry and recognizable in the majority of cases.

Our results confirms that indeed, even for simplistic models like a mixture of Gaussians, using a CNN discriminator provides a better task loss for generative modeling of images.

Figure 6: Some Prototypes learned using linear (left), dense (middle), and CNN discriminator (right).

We observe that with linear discriminator, only the mean of the training set is learned, while using the dense discriminator yields blurry prototypes.

Only using the CNN discriminator yields clear prototypes.

All 100 prototypes can be found in

@highlight

Parametric adversarial divergences implicitly define more meaningful task losses for generative modeling, we make parallels with structured prediction to study the properties of these divergences and their ability to encode the task of interest.