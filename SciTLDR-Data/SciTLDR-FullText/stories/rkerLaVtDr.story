In this work, we present a novel upper bound of target error to address the problem for unsupervised domain adaptation.

Recent studies reveal that a deep neural network can learn transferable features which generalize well to novel tasks.

Furthermore,  Ben-David et al. (2010) provide an upper bound for target error when transferring the knowledge, which can be summarized as minimizing the source error and  distance between marginal distributions simultaneously.

However, common methods based on the theory usually ignore the joint error such that samples from different classes might be mixed together when matching marginal distribution.

And in such case, no matter how we minimize the marginal discrepancy, the target error is not bounded due to an increasing joint error.

To address this problem, we propose a general upper bound taking joint error into account, such that the undesirable case can be properly penalized.

In addition, we utilize constrained hypothesis space to further formalize a tighter bound as well as a novel cross margin discrepancy to measure the dissimilarity between hypotheses which alleviates instability during adversarial learning.

Extensive empirical evidence shows that our proposal outperforms related approaches in image classification error rates on standard domain adaptation benchmarks.

The advent of deep convolutional neural networks (Krizhevsky et al., 2012) brings visual learning into a new era.

However, the performance heavily relies on the abundance of data annotated with ground-truth labels.

Since traditional machine learning assumes a model is trained and verified in a fixed distribution (single domain), where generalization performance is guaranteed by VC theory (N. Vapnik, 2000) , thus it cannot always be applied to real-world problem directly.

Take image classification task as an example, a number of factors, such as the change of light, noise, angle in which the image is pictured, and different types of sensors, can lead to a domain-shift thus harm the performance when predicting on test data.

Therefore, in many practical cases, we wish that a model trained in one or more source domains is also applicable to another domain.

As a solution, domain adaptation (DA) aims to transfer the knowledge learned from a source distribution, which is typically fully labeled into a different (but related) target distribution.

This work focus on the most challenging case, i.e, unsupervised domain adaptation (UDA), where no target label is available.

Ben-David et al. (2010) suggests that target error can be minimized by bounding the error of a model on the source data, the discrepancy between distributions of the two domains, and a small optimal joint error.

Owing to the strong representation power of deep neural nets, many researchers focus on learning domain-invariant features such that the discrepancy of two feature spaces can be minimized.

For aligning feature distributions across domains, mainly two strategies have been substantially explored.

The first one is bridging the distributions by matching all their statistics (Long et al., 2015; Pan et al., 2009) .

The second strategy is using adversarial learning (Goodfellow et al., 2014) to build a minimax game between domain discriminator and feature extractor, where a domain discriminator is trained to distinguish the source from the target while the feature extractor is learned to confuse it simultaneously (Ganin & Lempitsky, 2015; Ganin et al., 2016; Tzeng et al., 2017) .

In spite of the remarkable empirical results accomplished by feature distribution matching schemes, they still suffer from a major limitation: the joint distributions of feature spaces and categories are not well aligned across data domains.

As is reported in Ganin et al. (2016) , such methods fail to generalize in certain closely related source/target pairs, e.g., digit classification adaptation from MNIST to SVHN.

One potential reason is when matching marginal distributions of source and target domains, samples from different classes can be mixed together, where the joint error becomes nonnegligible since no hypothesis can classify source and target at the same time.

This work aims to address the above problem by incorporating joint error to formalize an optimizable upper bound such that the undesired overlap due to a wrong match can be properly penalized.

We evaluate our proposal on several different classification tasks.

In some experimental settings, our method outperforms other methods by a large margin.

The contributions of this work can be summarized as follows:

· We propose a novel upper bound taking joint error into account and theoretically prove that our proposal can reduce to several other methods under certain simplifications.

· We construct a constrained hypothesis space such that a much tighter bound can be obtained during optimization.

· We adopt a novel measurement, namely cross margin discrepancy, for the dissimilarity of two hypotheses on certain domain to alleviate the instability during adversarial learning and provide reliable performance.

The upper bound proposed by Ben-David et al. (2010) invokes numerous approaches focusing on reducing the gap between source and target domains by learning domain-invariant features, which can be achieved through statistical moment matching.

Long et al. (2015; use maximum mean discrepancy (MMD) to match the hidden representations of certain layers in a deep neural network.

Transfer Component Analysis (TCA) (Pan et al., 2011) tries to learn a subspace across domains in a Reproducing Kernel Hilbert Space (RKHS) using MMD that dramatically minimize the distance between domain distributions.

Adaptive batch normalization (AdaBN) modulates the statistics from source to target on batch normalization layers across the network in a parameterfree way.

Another way to learn domain-invariant features is by leveraging generative adversarial network to produce target features that exactly match the source.

Ganin & Lempitsky (2015) relax divergence measurement in the upper bound by a worst case which is equivalent to the maximum accuracy that a discriminator can possibly achieve when distinguishing source from target.

Tzeng et al. (2017) follow this idea but separate the training procedure into classification stage and adversarial learning stage where an independent feature extractor is used for target.

Saito et al. (2017b) explore a tighter bound by explicitly utilizing task-specific classifiers as discriminators such that features nearby the support of source samples will be favored by extractor.

Zhang et al. (2019) introduce margin disparity discrepancy, a novel measurement with rigorous generalization bounds, tailored to the distribution comparison with the asymmetric margin loss to bridge the gap between theory and algorithm.

Methods perform distribution alignment on pixel-level in raw input, which is known as image-to-image translation, are also proposed (Liu & Tuzel, 2016; Bousmalis et al., 2017; Sankaranarayanan et al., 2017; Shrivastava et al., 2016; Hoffman et al., 2018; Murez et al., 2017) .

Distribution matching may not only bring the source and target domains closer, but also mix samples with different class labels together.

Therefore, Saito et al. (2017a); Sener et al. (2016); Zhang et al. (2018) aim to use pseudo-labels to learn target discriminative representations encouraging a lowdensity separation between classes in the target domain (Lee, 2013) .

However, this usually requires auxiliary data-dependent hyper-parameter to set a threshold for a reliable prediction.

Long et al. (2018) present conditional adversarial domain adaptation, a principled framework that conditions the adversarial adaptation models on discriminative information conveyed in the classifier predictions, where the back-propagation of training objective is highly dependent on pseudo-labels.

We consider the unsupervised domain adaptation as a binary classification task (our proposal holds for multi-class case) where the learning algorithm has access to a set of n labeled points {(

.

from the source domain S and a set of m unlabeled points {(x i t ) ∈ X} m i=1 sampled i.i.d.

from the target domain T .

Let f S : X → {0, 1} and f T : X → {0, 1} be the optimal labeling functions on the source and target domains, respectively.

Let (usually 0-1 loss) denotes a distance metric between two functions over a distribution that satisfies symmetry and triangle inequality.

As a commonly used notation, the source risk of hypothesis h : X → {0, 1} is the error w.r.t.

the true labeling function f S under domain S, i.e., S (h) := S (h, f S ).

Similarly, we use T (h) to represent the risk of the target domain.

With these notations, the following bound holds:

For simplicity, we use

The above upper bound is minimized when h = f S , and it is equivalent to T (f S , f T ) owing to the triangle inequality:

Furthermore, we demonstrate in such case, our proposal is equivalent to an upper bound of optimal joint error λ because:

Fig. 1b illustrates a case where common methods fail to penalize the undesirable situation when samples from different classes are mixed together during distribution matching, while our proposal is capable to do so (for simplicity we assume f S takes a specific form, then T (f S , f T ) measures the overlapping area 2 and 5, which is equivalent to the optimal joint error λ).

Since optimal labeling functions f S , f T are not available during training, we shall further relax the upper bound by taking supreme w.r.t f S , f T within a hypothesis space H:

Then minimizing target risk T (h) becomes optimizing a minimax game and since the max-player taking two parameters f 1 , f 2 is too strong, we introduce a feature extractor g to make the min-player stronger.

Applying g to the source and target distributions, the overall optimization problem can be written as: min

However, if we leave H unconstrained, the supreme term can be arbitrary large.

In order to obtain a tight bound, we need to restrict the size of hypothesis space as well as maintain the upper bound.

For f S ∈ H 1 ≤ H and f T ∈ H 2 ≤ H, the following holds:

The constrained subspace for H 1 is trivial as according to its definition, f S must belong to the space consisting of all classifiers for source domain, namely H sc .

However, the constrained subspace for H 2 is a little problematic since we have no access to the true labels of the target domain, thus it is hard to locate f T .

Therefore, the only thing we can do is to construct a hypothesis space for H 2 that most likely contains f T .

As is illustrated in Fig. 1c , when matching distributions of source and target domain, if the ideal case is achieved where the conditional distributions of source and target are perfectly aligned, then it is fare to assume f T ∈ H sc .

However, if the worst case is reached where samples from different class are mixed together, then we tend to believe f T / ∈ H sc .

Considering this, we present two proposals in the following sections based on different constraints.

We assume H 2 is a space where the hypothesis can classify the samples from the source domain with an accuracy of γ ∈ [0, 1], namely H γ sc , such that we can avoid the worst case by choosing a small value for the hyper-parameter γ when a huge domain shift exists.

In practice, it is difficult to actually build such a space and sample from it due to a huge computational cost.

Instead, we use a weighted source risk to constrain the behavior of f 2 as an approximation to the sample from H γ sc , which leads to the final training objective:

Firstly, we build a space consisting of all classifiers for approximate target domain {(

based on pseudo labels which can be obtained by the prediction of h during training procedure, namely Ht c .

Here, we assume H 2 is an intersection between two hypothesis spaces , i.e. Given enough reliable pseudo labels, we can be confident about f T ∈ H 2 .

Analogously, the training objective is given by:

The reason we make such an assumption for H 2 can be intuitively explained by Fig. 2 .

If H 2 = H sc , then f 2 must perfectly classify the source samples, and it is possible that f 2 does not pass through some target samples (shadow are in 2a), especially when two domains differ a lot.

In such case, the feature extractor can move those samples into either side of the decision boundary to reduce the training objective (shadow area) which is not a desired behavior.

With an appropriate constraint (2b), as for the extractor, the only way to reduce the objective (shadow area) is to move those samples (orange) inside of f 2 .

Following the above notations, we consider a score function s(x, y) for multi-class classification where the output indicates the confidence of the prediction on class y. Thus an induced labeling function named l s from X → Y is given by:

As a well-established theory, the margin between data points and the classification surface plays a significant role in achieving strong generalization performance.

In order to quantify into differentiable measurement as a surrogate of 0-1 loss, we introduce the margin theory developed by Koltchinskii & Panchenko (2002) , where a typical form of margin loss can be interpreted as:

We aim to utilize this concept to further improve the reliability of our proposed method by leveraging this margin loss to define a novel measurement of the discrepancy between two hypotheses f 1 , f 2 (e.g. softmax) over a distribution D, namely cross margin discrepancy:

Before further discussion, we firstly construct two distributions D f1 , D f2 induced by f 1 , f 2 respectively, where

Then we consider the case where two hypotheses f 1 and f 2 disagree, i.e. y 1 = l f1 (x) = l f2 (x) = y 2 , and the primitive loss is defined as:

Then the cross margin discrepancy can be viewed as:

(13) which is a sum of the margin loss for f 1 on D f2 and the margin loss for f 2 on D f1 , if we use the logarithm of softmax as the score function.

Thanks to the trick introduced by Goodfellow et al. (2014) to mitigate the burden of exploding or vanishing gradients when performing adversarial learning, we further define a dual form as:

This dual loss resembles the objective of the generative adversarial network, where two hypotheses try to increase the probability of their own prediction and simultaneously decrease the probability of their opponents; whereas the feature extractor is trained to increase the probability of their opponents, such that the discrepancy can be minimized without unnecessary oscillation.

However, a big difference here is when training extractor, GANs usually maximize an alternative term log f 1 (x, y 2 ) + log f 2 (x, y 1 ) instead of directly minimizing log(1 − f 1 (x, y 2 )) + log(1 − f 2 (x, y 1 )) since the original term is close to zero if the discriminator achieves optimum.

In our case, the hypothesis can hardly beat the extractor thus the original form can be more smoothly optimized.

During the training procedure, the two hypotheses will eventually agree on some points (l f1 (x) = l f2 (x) = y) such that we need to define a new form of discrepancy measurement.

Analogously, the primitive loss and its dual form are given by:

Another reason why we propose such a discrepancy measurement is that it helps alleviate instability for adversarial learning.

As is illustrated in Fig. 3b , during optimization of a minimax game, when two hypotheses try to maximize the discrepancy (shadow area), if one moves too fast around the decision boundary such that the discrepancy is actually maximized w.r.t some samples, then these samples can be aligned on either side to decrease the discrepancy by tuning the feature extractor, which is not a desired behavior.

From Fig. 3a , we can see that our proposed cross margin discrepancy is flat for the points around original, i.e. the gradient w.r.t those points nearby the decision boundary will be relatively small, which helps to prevent such failure.

Zhang et al. (2019) propose a novel margin-aware generalization bound based on scoring functions and a new divergence MDD.

The training objective used in MDD can be alternatively interpreted as (here (h, f ) denotes the margin disparity): min

Recall Eq.7, if we set f 1 = f 2 = f and free the constraint of f to any f ∈ H, our proposal degrades exactly to MDD.

As is discussed above, when matching distribution, if and only if the ideal case is achieved, where the conditional distributions of induced feature spaces for source and target perfectly match (which is not always possible), can we assume two optimal labeling functions f S , f T to be identical.

Besides, an unconstrained hypothesis space for f is definitely not helpful to construct a tight bound.

Saito et al. (2017b) propose two task-specific classifiers f 1 , f 2 that are used to separate the decision boundary on source domain, such that the extractor is encouraged to produce features nearby the support of the source samples.

The objective used in MCD can be alternatively interpreted as (here (f 1 , f 2 ) is quantified by L 1 ):

Again, recall Eq.7, if we set γ = 1 and h = f 1 , MCD is equivalent to our proposal.

As is proved in section 3.1, the upper bound is optimized when h = f S .

However, it no longer holds since the upper bound is relaxed by taking supreme to form an optimizabel objective, i.e. setting h = f 1 does not necessarily minimize the objective.

Besides, as we discuss above, a fixed γ = 1, i.e H 2 = H sc lacks generality since we have no idea about where f T might be, such that it is not likely to be applicable to those cases where a huge domain shift exists.

In this experiment, our proposal is assessed in four types of adaptation scenarios by adopting commonly used digits datasets (Fig. 6 in Appendix),i.e.

MNIST (LeCun et al., 1998) , Street View House Numbers (SVHN) (Netzer et al., 2011) , and USPS (Hull, 1994) such that the result could be easily compared with other popular methods.

All experiments are performed in an unsupervised fashion without any kinds of data augmentation.

Details are omitted due to the limit of space (see A.1).

We report the accuracy of different methods in Tab.

1.

Our proposal outperforms the competitors in almost all settings except a single result compared with GPDA (Kim et al., 2019) .

However, their solution requires sampling that increases data size and is equivalent to adding Gaussian noise to the last layer of a classifier, which is considered as a type of augmentation.

Our success partially owes to combining the upper bound with the joint error, especially when optimal label functions differ from each other (e.g. MNIST →SVHN).

Moreover, as most scenarios are relatively easy for adaptation thus we can be more confident about the hypothesis space constraint owing to reliable pseudo-labels, which leads to a tighter bond during optimization.

The results demonstrate our proposal can improve generalization performance by adopting both of these advantages.

Fig. 4a shows that our original proposal is quite sensitive to the hyper-parameter γ.

In short, setting γ = 1 here yields the best performance in most situations, since f S , f T can be quite close after aligning distributions, especially in these easily adapted scenarios.

However, in MNIST → SVHN, setting γ = 0.1 gives the optimum which means that f S , f T are so far away due to a huge domain shift that no extractor is capable of introducing an identical conditional distribution in feature space.

The improvement is not that much, but at least we outperform the directly comparable MCD and show the importance of hypothesis space constraint.

Furthermore, Fig. 4d empirically proves simply minimizing the discrepancy between the marginal distribution does not necessarily lead to a reliable adaptation, which demonstrates the importance of joint error.

In addition, Fig. 4b,Fig.

4c show the superiority of the cross margin discrepancy which accelerates the convergence and provides a slightly better result.

We further evaluate our method on object classification.

The VisDA dataset (Peng et al., 2017) is used here, which is designed for 12-class adaptation task from synthetic object to real object images.

Source domain contains 152,397 synthetic images (Fig. 7a in Appendix), which are generated by rendering 3D CAD models.

Data of the target domain is collected from MSCOCO (Lin et al., 2014) consisting of 55,388 real images (Fig. 7b in Appendix).

Since the 3D models are generated without the background and color diversity, the synthetic domain is quite different from the real domain, which makes it a much more difficult problem than digits adaptation.

Again, this experiment is performed in unsupervised fashion and no data augmentation technique excluding horizontal flipping is allowed.

Details are omitted due to the limit of space (see A.2).

We report the accuracy of different methods in Tab.

2, and find that our proposal outperforms the competitors in all settings.

The image structure of this dataset is more complex than that of digits, yet our method provides reliable performance even under such a challenging condition.

Another key observation is that some competing methods (e.g., DANN, MCD), which can be categorized as distribution matching based on adversarial learning, perform worse than MDD which simply matches statistics, in classes such as plane and horse, while our methods perform better across all classes, which clearly demonstrates the importance of taking the joint error into account.

As for the original proposal (Fig. 5c ), performance drops when relaxing the constraint which actually confuses us.

Because we expect an improvement here since it is unbelievable that f S , f T eventually lie in a similar space judging from the relatively low prediction accuracy.

As for the alternative proposal ( Fig. 5d) , we test the adaptation performance for different η and the prediction accuracy drastically drops when η goes beyond 0.2.

One possible cause is that f 2 and h might almost agree on target domain, such that the prediction of h could not provide more accurate information for the target domain without introducing noisy pseudo labels.

Fig. 5a , Fig. 5b again demonstrate the superiority of cross margin discrepancy and the importance of joint error.

In this work, we propose a general upper bound that takes the joint error into account.

Then we further pursuit a tighter bound with reasonable constraint on the hypothesis space.

Additionally, we adopt a novel cross domain discrepancy for dissimilarity measurement which alleviates the instability during adversarial learning.

Extensive empirical evidence shows that learning an invariant representation is not enough to guarantee a good generalization in the target domain, as the joint error matters especially when the domain shift is huge.

We believe our results take an important step towards understanding unsupervised domain adaptation, and also stimulate future work on the design of stronger adaptation algorithms that manage to align conditional distributions without using pseudo-labels from the target domain.

layer and a 0.5 rate of dropout is conducted.

Nesterov accelerated gradient is used for optimization with a mini-batch size of 32 and an initial learning rate of 10 −3 which decays exponentially.

As for the hyper-parameter, we test for γ = {0.1, 0.5, 0.9, 1} and η = {0, 0.5, 0.8, 0.9}. For a direct comparison, we report the accuracy after 10 epochs.

Office-Home (Venkateswara et al., 2017 ) is a complex dataset (Fig. 8) containing 15,500 images from four significantly different domains: Art (paintings, sketches and/or artistic depictions), Clipart (clip art images), Product (images without background), and Real-world (regular images captured with a camera).

In this experiment, following the protocol from Zhang et al. (2019) , we evaluate our method by fine-tuning a ResNet-50 (He et al., 2015) model pretrained on ImageNet (Deng et al., 2009) .

The model except the last layer combined with a single-layer bottleneck is used as feature extractor and a randomly initialized 2-layer fully-connected network with width 1024 is used as a classifier, where batch normalization is applied to each layer and a 0.5 rate of dropout is conducted.

For optimization, we use the SGD with the Nesterov momentum term fixed to 0.9, where the batch size is 32 and learning rate is adjusted according to Ganin et al. (2016) .

From Tab.

3, we can see the adaptation accuracy of the source-only method is rather low, which means a huge domain shift is quite likely to exist.

In such case, simply minimizing the discrepancy between source and target might not work as the joint error can be increased when aligning distributions, thus the assumption of the basic theory (Ben-David et al., 2010) does not hold anymore.

On the other hand, our proposal incorporates the joint error into the target error upper bound which can boost the performance especially when there is a large domain shift.

Figure 8: Sample images from the Office-Home dataset (Venkateswara et al., 2017) .

Office-31 (Saenko et al., 2010) (Fig. 9 ) is a popular dataset to verify the effectiveness of a domain adaptation algorithm, which contains three diverse domains, Amazon from Amazon website, Webcam by web camera and DSLR by digital SLR camera with 4,652 images in 31 unbalanced classes.

In this experiment, following the protocol from Zhang et al. (2019) , we evaluate our method by fine-tuning a ResNet-50 (He et al., 2015) model pretrained on ImageNet (Deng et al., 2009) .

The model used here is almost identical to the one in Office-Home experiment except a different width 2048 for classifiers.

For optimization, we use the SGD with the Nesterov momentum term fixed to 0.9, where the batch size is 32 and learning rate is adjusted according to Ganin et al. (2016) .

The results on Office-31 are reported in Tab.

4.

As for the tasks D→A and W→A, judging from the adaptation accuracy of those previous methods that do not consider the joint error, it is quite likely that samples from different classes are mixed together when matching distributions.

Our method shows an advantage in such case which demonstrates that the proposal manage to penalize the undesired matching between source and target.

As for the tasks A→W and A→D, our proposal shows relatively high variance and poor performance especially in A→W. One possible reason is that our method depends on building reliable classifiers for the source domain to satisfy the constraint.

However, the Amazon dataset contains a lot of noise (Fig. 10) such that the decision boundary of the source classifiers varies drastically in each iteration during training procedure, which can definitely harm the convergence.

@highlight

joint error matters for unsupervised domain adaptation especially when the domain shift is huge