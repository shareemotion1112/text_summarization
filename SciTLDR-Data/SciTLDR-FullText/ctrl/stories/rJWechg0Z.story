In this work, we face the problem of unsupervised domain adaptation with a novel deep learning approach which leverages our finding that entropy minimization is induced by the optimal alignment of second order statistics between source and target domains.

We formally demonstrate this hypothesis and, aiming at achieving an optimal alignment in practical cases, we adopt a more principled strategy which, differently from the current Euclidean approaches, deploys alignment along geodesics.

Our pipeline can be implemented by adding to the standard classification loss (on the labeled source domain), a source-to-target regularizer that is weighted in an unsupervised and data-driven fashion.

We provide extensive experiments to assess the superiority of our framework on standard domain and modality adaptation benchmarks.

Learning visual representations that are invariant across different domains is an important task in computer vision.

Actually, data labeling is onerous and even impossible in some cases.

It is thus desirable to train a model with full supervision on a source, labeled domain and then learn how to transfer it on a target domain, as opposed to retrain it completely from scratch.

Moreover, the latter stage is actually not possible if the target domain is totally unlabelled: this is the setting we consider in our work.

In the literature, this problem is known as unsupervised domain adaptation which can be regarded as a special semi-supervised learning problem, where labeled and unlabeled data come from different domains.

Since no labels are available in the target domain, source-to-target adaptation must be carried out in a fully unsupervised manner.

Clearly, this is an arguably difficult task since transferring a model across domains is complicated by the so-called domain shift [Torralba & Efros (2011) ].

In fact, while switching from the source to the target, even if dealing with the same K visual categories in both domains, different biases may arise related to several factors.

For instance, dissimilar points of view, illumination changes, background clutter, etc.

In the previous years, a broad class of approaches has leveraged on entropy optimization as a proxy for (unsupervised) domain adaptation, borrowing this idea from semi-supervised learning [Grandvalet & Bengio (2004) ].

By either performing entropy regularization [Tzeng et al. (2015) ; Carlucci et al. (2017) ; Saito et al. (2017) ], explicit entropy minimization [Haeusser et al. (2017) ], or implicit entropy maximization through adversarial training [Ganin & Lempitsky (2015) ; Tzeng et al. (2017) ], this statistical tool has demonstrated to be powerful for adaptation purposes.

Alternatively, there exist methods which try to align the source to the target domain by learning an explicit transformation between the two so that the target data distribution can be matched to the one of the source one [Glorot et al. (2011); Kan et al. (2015) ; Shekhar et al. (2013) ; Gopalan & Li (2011); Gong et al. (2012a) ].

Within this paradigm, correlation alignment minimizes the distance between second order statistics computed in the form of covariance representations between features from the source a [Fernando et al. (2013) ; Sun et al. (2016) ; Sun & Saenko (2016) ].Apparently, correlation alignment and entropy minimization may seem two unrelated and approaches in optimizing models for domain adaptation.

However, in this paper, we will show that this is not the case and, indeed, we claim that the two classes of approaches are deeply intertwined.

In addition to formally discuss the latter aspect, we also obtain a solution for the prickly problem of hyperparameter validation in unsupervised domain adaptation.

Indeed, one can construct a validation set out of source data but the latter is not helpful since not representative of target data.

At the same time, due to the lack of annotations on the target domain, usual (supervised) validation techniques can not be applied.

In summary, this paper brings the following contributions.1.

We explore the two paradigms of correlation alignment and entropy minimization, by formally demonstrating that, at its optimum, correlation alignment attains the minimum of the sum of cross-entropy on the source domain and of the entropy on the target.2.

Motivated by the urgency of penalizing correlation misalignments in practical terms, we observe that an Euclidean penalty, as adopted in [Sun et al. (2016); Sun & Saenko (2016) ], is not taking into account the structure of the manifold where covariance matrices lie in.

Hence, we propose a different loss function that is inspired by a geodesic distance that takes into account the manifold's curvature while computing distances.3.

When aligning second order statistics, a hyper-parameter controls the balance between the reduction of the domain shift and the supervised classification on the source domain.

In this respect, a manual cross-validation of the parameter is not straightforward: doing it on the source domain may not be representative, and it is not possible to do on the target due to the lack of annotations.

Owing to our principled connection between correlation alignment and entropy regularization, we devise an entropy-based criterion to accomplish such validation in a data-driven fashion.4.

We combine the geodesic correlation alignment with the entropy-based criterion in a unique pipeline that we call minimal-entropy correlation alignment.

Through an extensive experimental analysis on publicly available benchmarks for transfer object categorization, we certify the effectiveness of the proposed approach in terms of systematic improvements over former alignment methods and state-of-the-art techniques for unsupervised domain adaptation in general.

The rest of the paper is outlined as follows.

In Section 2, we report the most relevant related work as background material.

Section 3 presents our theoretical analysis which inspires our proposed method for domain adaptation (Section 4).

We report a broad experimental validation in Section 5.

Finally, Section 6 draws conclusions.

In this Section, we will detail the two classes of correlation alignment and entropy optimization methods that are combined by our adaptation technique.

An additional literature review is available in Appendix A.We consider the problem of classifying an image x in a K-classes problem.

To do so, we exploit a bunch of labeled images x 1 , . . .

, x n and we seek for training a statistical classifier that, during inference, provides probabilities for a given test imagex to belong to each of the K classes.

In this work, such classifier is fixed to be a deep multi-layer feed-forward neural network denoted as DISPLAYFORM0 The network f depends upon some parameters/weights θ that are optimized by minimizing over θ the cross-entropy loss function DISPLAYFORM1 In (2), for each image x i , the inner product ·, · computes a similarity measure between the network prediction f (x i ; θ) and the corresponding data label z i , which is a K dimensional one-hot encoding vector.

Precisely, z ik = 1 if x i belongs to the k-th class, being zero otherwise.

Finally, for notational simplicity, let X and Z define the collection all images x i and corresponding labels z i , respectively.

In a classical fully supervised setting, other than minimizing (2), one can also add some weighted additive regularizers to the final loss, such as an L 2 penalty.

But, in the case of domain adaptation, θ should be chosen as to promote a good portability from the source S to the target domain T .Correlation alignment.

In the case of unsupervised domain adaptation, we assume that none of the examples in the target domain is labelled and, therefore, we should perform adaptation at the feature level.

In the case of correlation alignment, we can replace (2) with the following problem DISPLAYFORM2 where we compute the supervised cross-entropy loss between data X S and annotations Z S belonging to the source domain only.

Concurrently, the network parameters θ are modified in order to align the covariance representations DISPLAYFORM3 that are computed through the centering matrix J (see [

Ha Quang et al. (2014); Cavazza et al. (2016) ] for a closed-form) on top of the activations computed at a given layer 1 by the network f (·, θ).

Precisely, A S and A T stack by columns the d-dimensional activations computed from the source and the target domains.

Also, θ is regularized according to the following Euclidean penalization DISPLAYFORM4 Figure 1: Geodesic versus Euclidean distances in the case of a non-zero curvature manifold (as the one of SPD matrices).in terms of the (squared) Frobenius norm · F .

In [Fernando et al. (2013); Sun et al. (2016) ], the aligning transformation is obtained in closed-form.

Despite the latter would attain the perfect correlation matching, it requires matrix inversion and eigendecomposition operations: thus it is not scalable.

As a remedy, in [Sun & Saenko (2016) ], (5) is used a loss for optimizing (3) with stochastic batchwise gradient descent.

Problem 1.

Mathematically, covariance representations (4) are symmetric and positive definite (SPD) matrices belonging to a Riemannian manifold with non-zero curvature BID0 ].

Therefore, measuring correlation (mis)alignments with an Euclidean metric like (5) is arguably suboptimal since it does not capture the inner geometry of the data (see Figure 1 ).Entropy regularization.

The cross entropy H on the source domain and entropy E on the target domain can be optimized as follows: DISPLAYFORM5 where DISPLAYFORM6 In this way, we circumvent the impossibility of optimizing the cross entropy on the target (due to the unavailability of labels on T ), and we replace it with the entropy E(X T ) computed on the softlabels z soft (x t ) = f (x t ; θ), which is nothing but the network predictions [Lee (2013) ].

Empirically, soft-labels increases the confidence of the model related to its prediction.

However, for the purpose of domain adaptation, optimizing FORMULA5 is not enough and, in parallel, ancillary adaptation techniques are invoked.

Specifically, either additional supervision [Tzeng et al. (2015) ], batch normalization [Carlucci et al. (2017) ] or probabilistic walk on the data manifold [Haeusser et al. (2017) ] have been exploited.

As a different setup, a min-max problem can be devised where H(X S , Z S ) is minimized and, at the same time, entropy is maximized within a binary classification of predicting whether a given instance belongs to the source or the target domain.

This is done in [Ganin & Lempitsky (2015) ] and [Tzeng et al. (2017) ] by reversing the gradients and using adversarial training, respectively.

In practical terms, this means that, in addition to the loss function in (6), one needs to carry out other parallel optimizations whose reciprocal balance in influencing the parameters' update is controlled by means of hyper-parameters.

Since the latter have to be grid-searched, a validation set is needed in order to select the hyper-parameters' configuration that corresponds to the best performance on it.

How to select the aforementioned validation set leads to the following point.

Problem 2.

In the case of domain adaptation, cross-validation for hyper-parameter tuning on the source directly is unreasonable because of the domain shift.

In fact, for instance, [Tzeng et al. (2015) ] can do it only by adding supervision on the target and, in [Carlucci et al. (2017) ], cross-validation is performed on the source after the target has been aligned to it.

Since we need λ to be fixed before solving for correlation alignment and since we consider a fully unsupervised adaptation setting, we cannot use any of the previous strategy and, obviously, we are not allowed for supervised cross-validation on the target.

Thus, hyper-parameter tuning is really a problem.

In this work, we combine the two classes of correlation alignment [Sun et al. FORMULA0 FORMULA0 ] in a unique framework.

By doing so, we embrace a more principled approach to align covariance representations (as to tackle Problem 1), while, at the same time, solving Problem 2 with a novel unsupervised and data-driven cross-validation technique.3 MINIMAL-ENTROPY CORRELATION ALIGNMENT

In this section, we deploy a rigorous mathematical connection between correlation alignment and entropy minimization in order to understand the mutual relationships.

The following theorem (see proof in Appendix B) represents the main result.

Theorem 1.

With the notation introduced so far, if θ optimally aligns correlation in (3), then, θ minimizes (6) for every γ > 0.The previous statement certifies that, at its optimum, correlation alignment provides minimal entropy for free.

If one compares FORMULA2 with FORMULA5 , one may notice that, in both cases, we are minimizing H over the source domain S. Therefore, if we assume that H(X S , Z S ) = min, we have a perfect classifier whose predictions on S are extremely confident and correct.

Thus, the predictions are distributed in a very picky manner and, therefore, entropy on the source is minimized.

At the same time, we can minimize the entropy on the target since T is made "indistinguishable" from S after the alignment.

Hence, the target's predictions are distributed in a similar picky way so that entropy on T is minimized as well.

Observation 1.

Since we proved that optimal correlation alignment implies entropy minimization, one may ask whether the converse holds.

That is, if the optimum of (6) gives the optimum of (3).

The answer is negative as it will be clear by the following counterexample.

In fact, we can always minimize the cross entropy on the source with a fully supervised training on S. However, such classifier could be always confident in classifying a target example as belonging to, say, Class 1.

After that, we can deploy a dummy adaptation step that, for whatever target imagex to be classified, we always predict it to be Class 1.

In this case the entropy on the target is clearly minimized since the distribution of the target prediction is a Dirac's delta δ 1k for any class k.

But, obviously, nothing has been done for the sake of adaptation and, in particular, optimal correlation alignment is far from being realized (see Appendix C).In Theorem 1, the assumption of having an optimal correlation alignment is crucial for our theoretical analysis.

However, in practical terms, optimal alignment is also desirable in order to effectively deploy domain adaptation systems.

Moreover, despite the optimal alignment in (3) is able to minimize (6) for any γ > 0, in practice, hyper-parameters need to be cross-validated and this is not an easy task in unsupervised domain adaptation (as we explained in Problem 2).

In the next section, a solution for all these problems will be distilled from our improved knowledge.

Based on the previous remarks, we address the unsupervised domain adaptation problem by training a deep net for supervised classification on S while adding a loss term based on a geodesic distance on the SPD manifold.

Precisely, we consider the (squared) log-Euclidean distance DISPLAYFORM0 where d is the dimension of the activations A S and A T , whose covariances are intended to be aligned, U and V are the matrices which diagonalize C S and C T , respectively, and σ i , µ i , i = 1, ..., d are the corresponding eigenvalues.

The normalization term 1/d 2 accounts for the sum of the d 2 terms in the · 2 F norm, which makes log independent from the size of the feature layer.

The geodesic alignment for correlation is attained by minimizing the problem min θ [H(X S , Z S ) + λ · log (C S , C T )], for some λ > 0.

This allows lo learn good features for classification which, at the same time, do not overfit the source data since they reflect the statistical structure of the target set.

To this end, a geodesic distance accounts for the geometrical structure of covariance matrices better than (3).

In this respect, the following two aspects are crucial.• Problem 1 is addressed by introducing the log-Euclidean distance log between SPD matrices, which is a geodesic distance widely adopted in computer vision [Cavazza et al. (2016); Zhang et al. (2016); Ha Quang et al. (2014; BID1 ; Cavazza et al. FORMULA0 ] when dealing with covariance operators.

The rationale is that, within the many geodesic distances, (8) is extremely efficient because does not require matrix inversions (like the affine one aff (C S , C T ) = log(C S C −1 T ) F ).

Moreover, while shifting from one geodesic distance to another, the gap in performance obtained are negligible, provided the soundness of the metric [Zhang et al. (2016) ].•

As observed in Problem 2, the hyperparameter λ is a critical coefficient to be cross validated.

In fact, a high value of λ is likely to force the network towards learning oversimplified low-rank feature representations.

Despite this may result in perfectly aligned covariances, it could be useless for classification purposes.

On the other hand, a small λ may not be enough to bridge the domain shift.

Motivated by Theorem 1, we select the λ which minimizes the entropy E(X T ) on the target domain.

Indeed, since we proved that H(X S ) is minimized at the same time in both FORMULA2 and FORMULA5 , we can naturally tune λ so that E(X T ) = min.

Note that this entropy-based criterion for λ is totally fair in unsupervised domain adaptation since, as in FORMULA5 , E does not require ground truth target labels to be computed, but only relies on inferred soft-labels.

In summary, we propose the following minimization pipeline for unsupervised domain adaptation, which we name Minimal-Entropy Correlation Alignment (MECA) DISPLAYFORM1 In other words, in (9), we minimize the objective functional H(X S , Z S ) + λ · log (C S , C T ) by gradient descent over θ.

While doing so, we can choose λ by validation, such that the network f (·; θ) is able, at the same time, to attain the minimal entropy on the target domain.

Differentiability.

For a fixed λ, the loss (9) needs to be differentiable in order for the minimization problem to be solved via back-propagation, and its gradients should be calculated with respect to the input features.

However, as (4) shows, C S and C T are polynomial functions of the activations and the same holds when one applies the Euclidean norm · 2 F .

Additionally, since the log function is differentiable over its domain, we can easily see that we can still write down the gradients of the loss (9) in a closed form by exhaustively applying the chain rule over elementary functions that are in turn differentiable.

In practice, this is not even needed, since modern tools for deep learning consist in software libraries for numerical computation whose core abstraction is represented by computational graphs.

Single mathematical operations (e.g., matrix multiplication, summation etc.) are deployed on nodes of a graph, and data flows through edges.

Reverse-mode differentiation takes advantage of the gradients of single operations, allowing training by backpropagation through the graph.

The loss (9) can be easily written (for a fixed λ) in few lines of code by exploiting mathematical operations which are already implemented, together with their gradients, in TensorFlow TM or other libraries 2 .

In this Section we will corroborate our theoretical analysis with a broad validation which certify the correctness of Theorem 1 and the effectiveness of our proposed entropy-based cross-validation for λ in (9).

In addition, by means of a benchmark comparison with state-of-the-art approaches in unsupervised domain adaptation, we will prove the effectiveness of the geodesic versus the Euclidean alignment and, in general, that MECA outperforms many previously proposed methods.

We run the following adaptation experiments.

We use digits from SVHN [Netzer et al. (2011) ] as source and we transfer on MNIST.

Similarly, we transfer from SYN DIGITS [Ganin & Lempitsky (2015) ] to SVHN.

For the object recognition task, we train a model to classify objects on RGB images from NYUD [Silberman et al. (2012) ] dataset and we test on (different) depth images from the same visual categories.

Reproducibility details for both dataset and baselines are reported in Appendix D.

As shown in Theorem 1, correlation alignment and entropy regularization are intertwined.

Despite this result holds at the optimum only, we can actually observe an even stronger linkage.

Precisely, we empirically register that a gradient descent path for correlation alignment induces a gradient descent path for entropy minimization.

In fact, in the top-left part of FIG1 , while running correlation alignment to align source and target with either an Euclidean (red curve) or geodesic penalty (orange curve), we are able to minimize the entropy.

Also, when comparing the two, geodesic provides a lower entropy value than the Euclidean alignment, meaning that our approach is able to better minimize E(X T ).

Interestingly, even if the baseline with no adaptation is able to minimize the entropy as well (blue curve), this is only a matter of overfitting the source.

In fact, the baseline produces a classifier which is overconfidently wrong on the target (as explained in Appendix C) as long as training evolves.

Remember that optimal correlation alignment implies entropy minimization being the converse not true: if we check the alignment of source and target distributions ( FIG1 bottom-left), we see that, with no adaptation (blue curve), the two distributions are increasingly mismatched as long as training proceeds.

Differently, with either Euclidean or geodesic alignments, we are able to match the two and, in order to check the quality of such alignment, we conduct the following experiment.

In FIG1 , right column, we show the plots of target entropy and classification accuracies related to SVHN→MNIST as a function of λ ∈ {0.1, 0.5, 1, 2, 5, 7, 10, 20}. Let us stress that, since we measure distances on the SPD manifold directly, we can conjecture that (8) can achieve a better alignment between covariances than (5).

Actually, if one applies the closed-form solution of [Sun et al. (2016) ] the optimal alignment can be found analytically.

However, due to the required matrix inversions, such approach is not scalable an one needs to backpropagate errors starting from a penalty function in order to train the model.

As one can clearly see in FIG1 (right), Euclidean alignment is performing about 5% worse than our proposed geodesic alignment on SVHN→MNIST.

But, most importantly, in the Euclidean case, the minimal entropy does not correspond to the maximum performance on the target.

Differently, when using the geodesic penalty (8), we see that the λ which minimizes E(X T ) is also the one that gives the maximum performance on the target.

Thus, we can conclude that our geodesic approach is better than the Euclidean one since totally compatible with a data-driven cross-validation strategy for λ, requiring no labels belonging to the target domain.

Additional evidences of the superiority of our proposed geodesic alignment in favor of a classical one are reported in the next Section.

Thereby, our Minimal-Entropy Correlation Alignment (MECA) method is benchmarked against state-of-the-art approaches for unsupervised deep domain adaptation.

In this Section, we benchmark MECA against general state-of-the-art frameworks for unsupervised domain adaptation with deep learning: Domain Separation Network (DSN) BID1 and Domain Transfer Network (DTN) [Taigman et al. (2017) ].

In addition, we also compare with two (implicit) entropy maximization frameworks -Gradient Reversal Layer (GRL) [Ganin & Lempitsky (2015) ] and ADDA [Tzeng et al. (2017) ]

-and with the entropy regularization technique of [Saito et al. (2017) ], which uses a triple classifier (TRIPLE).

Also, we consider the deep Euclidean correlation alignment named Deep CORAL [Sun & Saenko (2016) ].

In order to carry on a comparative analysis, we setup standard baseline architectures which reproduce source only performances (i.e., performance of the models with no adaptation).

More details are provided in Appendix D

Normalized training time λ Accuracy vs. Entropy Geodesic align.

Normalized training time λ Accuracy vs. Entropy Euclidean align.

In all cases, we report the published results from the other competitors, even when they devised more favorable experimental conditions than ours (e.g., DTN exploits the extra data provided with SVHN).

In the case of Deep CORAL, since the published results only cover the (almost saturated) Office dataset, we decided to run our own implementation of the method.

While doing this, in order to cross-validate λ in (3), we tried to balance the magnitudes of the two losses FORMULA1 and FORMULA4 as prescribed in the original work.

However, since this approach does not provide good results, we were forced to cross-validate Deep Coral on the target directly.

Let us remark that, as we show in Section 5.1, our proposed entropy-based cross validation is not always compatible with an Euclidean alignment.

Differently, for MECA, our geodesic approach naturally embeds the entropy-based criterion and, consequently, we are able to maximize the performance on the target with a fully unsupervised and data-dependent cross-validation.

In addition, the classification performance registered by MECA is extremely solid.

In fact, in the worst case we found (SYN→SVHN), MECA is performing practically on par with respect to Deep CORAL, despite for the latter labels on the target are used, being not far from the score of TRIPLE.

This point can be explained with the fact that, for some benchmark datasets, the domain shift is not so prominent -e.g., check the visual similarities between SYN and SVHN datasets in the first two columns of Figure 3 .

In such cases, one can naturally argue that the type of alignment is not so crucial since adaptation is not strictly necessary, and the two types of alignment are pretty equivalent.

This also explains the gap shown by MECA from the state-of-the-art (TRIPLE, 93.1%, which performs better than training on target with our architecture) and, eventually, the fact that the baseline itself is already doing pretty well (87.0%).

As the results certify, MECA is systematically outperforming Deep CORAL: +0.5% on SYN→SVHN, +2.1% on NYUD and +5% on SVHN→MNIST.

Table 1 : Unsupervised domain adaptation with MECA.

Perfomance is measured as normalized accuracy and we compare with general, entropy-related (E) and correlation alignment (C) state-ofthe-art approaches.

§ We also include this experiment exclusively for evaluation purposes.

Let us stress that all methods in comparisons and our proposed MECA exploit labels only from the source domain during training.† A more powerful feature extractor as baseline and uses also extra SVHN data.‡ Results refer to our own Tensorflow TM implementation, with cross-validation on the target.

DISPLAYFORM0 Finally, our proposed MECA is able to improve the previous methods by margin on SVHN→MNIST (+5.0%) and on NYUD as well (+2.6%).

In this paper we carried out a principled connection between correlation alignment and entropy minimization, formally demonstrating that the optimal solution to the former problem gives for free the optimal solution of the latter.

This improved knowledge brought us to two algorithmic advances.

First, we achieved a more effective alignment of covariance operators which guarantees a superior performance.

Second, we derived a novel cross-validation approach for the hyper-parameter λ so that we can obtain the maximum performance on the target, even not having access to its labels.

These two components, when combined in our proposed MECA pipeline, provide a solid performance against state-of-the-art methods for unsupervised domain adaptation.

L.J.P van der Maaten and G.E. Hinton.

Visualizing high-dimensional data using t-sne.

For the problem of (unsupervised) domain adaptation, a first class of methods aims at learning transformations which align feature representations in the source and target sets.

For instance, in [Glorot et al. (2011) ]

auto-encoders are exploited to learn common features.

In [Kan et al. (2015) ], a bi-shifting auto-encoder (BSA) is instead intended to shift source domain samples into target ones and, similarly, other methods approach the same problem by means of techniques based on dictionary learning (as in [Shekhar et al. (2013)] ).

Geodesic methods (such as [Gopalan & Li (2011); Gong et al. (2012a) ] aim at projecting source and target datasets on a common manifold in such a way that the projection already solves the alignment problem.

The approaches [Gong et al. (2012b) ; Gopalan et al. FORMULA0 ] learns a smooth transition between the source and data manifold by means of Principal Components Analysis and Partial Least Squares, respectively.

Inspired by the idea of adapting second order statistics between the two domains, [Sun et al. (2016); Fernando et al. (2013) ] propose a transformation to minimize the distance between the covariances of source and target datasets in order to, ultimately, achieve correlation alignment.

Due to the well known properties of covariance operators, in some cases [Sun et al. (2016) ], the alignment can be written down in closed-form.

But, since the latter operation can be prohibitively expensive in terms of computational cost, Sun & Saenko (2016) implements correlation alignment in an end-to-end fashion by means of backpropagation.

A complementary family of approaches exploit the powerful statistical tool of entropy optimization in order to carry out adaptation.

Indeed, the notion of association [Haeusser et al. (2017) ] is actually implementing explicit entropy minimization [Grandvalet & Bengio (2004) ] to align the target to the source embedding by navigating the data manifold by means of closed cyclic paths that interconnect instances belonging to the same objects' classes.

In parallel, there are cases [Ganin & Lempitsky (2015) ; Tzeng et al. (2017) ] where minimax optimization is responsible for doing the following adversarial training.

One seeks for feature representations that are effective for the primary visual recognition task being at the same time invariant while changing from source to target.

The latter stage is implemented as the attempt of devising a random chance classifier which is asked to detect whether a given feature vector has been computed from a source or target data instance.

Therefore, those approaches are implicitly promoting entropy maximization 3 at the classifier level.

Finally, entropy regularization is accomplished in [Tzeng et al. (2015); Carlucci et al. (2017); Saito et al. (2017) ] as a complementary step to boost adaptation.

Indeed, already established techniques for adaptation such as Batch Normalization [Ioffe & Szegedy (2015) ; Li et al. (2016) ] are applied in low-level layers to align the representations.

On top of that, adaptation is refined at the end of the feature hierarchy by introducing a entropy-based regularizer on the target domain based.

Practically, the latter exploits network's prediction to generate pseudo-labels [Lee FORMULA0

Proof.

By hypothesis, we assume that θ is the optimal hyper-parameter which attains the optimum of (3), which implies DISPLAYFORM0 by the properties of the squared-distance function d.

Let us fix an arbitrary γ > 0 and let us consider DISPLAYFORM1 the objective functional in (6) which rewrites DISPLAYFORM2 while writing down the expression of the cross-entropy function H between ground truth source labels Z S and network's predictions which are also exploited to compute the entropy function E on the target domain.

By hypothesis, since θ is such that H(X S , Z S ) = min, then the thesis will follow if we prove that DISPLAYFORM3 since the minimum of the sum of two functions is achieved when the two addends are minimized separately.

Now, by hypothesis, since we assume the optimal correlation alignment, then, due to the fact that C S = C T , we can assume that the statistical properties of the trained classifier on the source can be transferred to the target with null performance degradation since, basically, we have obtained the way to completely solve the domain shift issue.

This implies that, if we assume that some oracle will provide us the ground truth labels z j for the target domain, we can get that DISPLAYFORM4 for any arbitrary x j in the target domain T .

Note that θ was optimized in a fair manner, by exploiting the labels of the source domain only and the fact that a perfect classification on the target is achieved is a side effect of assuming that we achieved the optimal correlation alignment, making the target data distribution essentially indistinguishable from the source one.

In particular, f (x j ; θ ) is a Dirac's delta function such that f k (x j ; θ ) = 1 if x j belongs to the k-th class and f k (x j ; θ ) = 0 otherwise.

Therefore, we get DISPLAYFORM5 due to the fact that k (x j ; θ ) is a Dirac's delta and since we decompose, for each x j , the summation over k in two parts: when k equals the class of x j , f k (x j ; θ ) log (f k (x j ; θ )) = log 1 = 0 and, in all other cases, the addends vanishes.

Therefore DISPLAYFORM6 Since E(X T ) is a non-negative function, (16) gives the thesis (13) due to the generality of γ

Consider the fully supervised classification problem of optimizing θ for the deep neural network f (·, θ) such that, while comparing network's prediction f (x i , θ) and ground truth annotations z i , relative to the source domain S, we get the problem of DISPLAYFORM0 where, in (17), minimization is carried out on θ.

Now, we can devise a dummy classifier f , depending upon the same exact parameter choice θ such that DISPLAYFORM1 That is, we use on the target the same exact classifier that we trained on the source (with no adaptation).

That is, source data is classified by f based on f , while, when asked to classify an image from the target domain, f will always predict that instance to belong to the first class.

By using the same exact scheme of proof as in Appendix B, we can show that, f achieves the minimal entropy E(X T ) on the target domain T .

This is an evidence for the fact that, although optimal correlation alignment implies minimal entropy, the converse is not true.

Ancillary, it explains why in [Tzeng et al. (2015) ; Carlucci et al. (2017) ], adaptation is effectively carried out with ancillary techniques and entropy regularization it's just a boosting factor as opposed to a factual regularizer for domain adaptation.

Figure 3: Sampled images from the datasets involved in the domain adaptation experiments.

From left to right, SVHN (first column, digits 9, 9, 2 from top to bottom), SYN (second column, digits 3, 9, 7 from top to bottom), NYUD RGB (third column, toilet, sink and garbage-bin classes acquires as RGB), NYUD depth (fourth column, different instances from the same previous classes acquired with the alternative modality) and the well known MNIST dataset (fifth column).

SVHN → MNIST.

This split represents a very realistic domain shift, since SVHN [Netzer et al. (2011) ] (Street-View-House-Numbers) is built with real-world house numbers.

We used the whole training sets of both datasets, following the usual protocol for unsupervised domain adaptation (SVHN's training set contains 73, 257 images).

We also resized MNIST images to 32 × 32 pixels and converted SVHN to grayscale, according to the standard protocol.

NYUD (RGB → depth).

This domain adaptation problem is actually a modality adaptation task and it was recently proposed by Tzeng et al. [Tzeng et al. (2017) ].

The dataset is gathered by cropping out object bounding boxes around instances of 19 classes of the NYUD [Silberman et al. (2012) ] dataset.

It comprises 2,186 labeled source (RGB) images and 2,401 unlabeled target depth images, HHA-encoded [Gupta et al. (2014) ].

Note that these are obtained from two different splits of the original dataset, in order to ensure that the same instance is not seen in both domains.

The adaptation task is extremely challenging, due to the very different nature of the data, the limited number of examples (especially for some classes) and the low resolution anf heterogeneous size of the cropped bounding boxes.

Table 1 , where MECA discriminates better than Deep Coral.

<|TLDR|>

@highlight

A new unsupervised deep domain adaptation technique which efficiently unifies correlation alignment and entropy minimization

@highlight

Improves the correlation alignment approach to domain adaptation by replacing the Euclidean distance with the geodesic Log-Euclidean distance between two covariance matices, and automatically selecting the balancing cost by the entropy on the target domain.

@highlight

Proposal for minimal-entropy correlation alignment, an unsupervised domain adaptation algorithm which links together entropy minimization and correlation alignment methods.