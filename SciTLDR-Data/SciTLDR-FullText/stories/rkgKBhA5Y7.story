Presently the most successful approaches to semi-supervised learning are based on consistency regularization, whereby a model is trained to be robust to small perturbations of its inputs and parameters.

To understand consistency regularization, we conceptually explore how loss geometry interacts with training procedures.

The consistency loss dramatically improves generalization performance over supervised-only training; however, we show that SGD struggles to converge on the consistency loss and continues to make large steps that lead to changes in predictions on the test data.

Motivated by these observations, we propose to train consistency-based methods with Stochastic Weight Averaging (SWA), a recent approach which averages weights along the trajectory of SGD with a modified learning rate schedule.

We also propose fast-SWA, which further accelerates convergence by averaging multiple points within each cycle of a cyclical learning rate schedule.

With weight averaging, we achieve the best known semi-supervised results on CIFAR-10 and CIFAR-100, over many different quantities of labeled training data.

For example, we achieve 5.0% error on CIFAR-10 with only 4000 labels, compared to the previous best result in the literature of 6.3%.

Recent advances in deep unsupervised learning, such as generative adversarial networks (GANs) BID8 , have led to an explosion of interest in semi-supervised learning.

Semisupervised methods make use of both unlabeled and labeled training data to improve performance over purely supervised methods.

Semi-supervised learning is particularly valuable in applications such as medical imaging, where labeled data may be scarce and expensive BID23 .Currently the best semi-supervised results are obtained by consistency-enforcing approaches BID2 BID17 BID31 BID21 BID24 .

These methods use unlabeled data to stabilize their predictions under input or weight perturbations.

Consistency-enforcing methods can be used at scale with state-of-the-art architectures.

For example, the recent Mean Teacher BID31 model has been used with the Shake-Shake BID7 architecture and has achieved the best semi-supervised performance on the consequential CIFAR benchmarks.

This paper is about conceptually understanding and improving consistency-based semi-supervised learning methods.

Our approach can be used as a guide for exploring how loss geometry interacts with training procedures in general.

We provide several novel observations about the training objective and optimization trajectories of the popular ⇧ BID17 and Mean Teacher BID31 consistency-based models.

Inspired by these findings, we propose to improve SGD solutions via stochastic weight averaging (SWA) BID12 , a recent method that averages weights of the networks corresponding to different training epochs to obtain a single model with improved generalization.

On a thorough empirical study we show that this procedure achieves the best known semi-supervised results on consequential benchmarks.

In particular:• We show in Section 3.1 that a simplified ⇧ model implicitly regularizes the norm of the Jacobian of the network outputs with respect to both its inputs and its weights, which in turn encourages flatter solutions.

Both the reduced Jacobian norm and flatness of solutions have been related to generalization in the literature BID29 BID22 BID3 BID27 BID13 BID12 .

Interpolating between the weights corresponding to different epochs of training we demonstrate that the solutions of ⇧ and Mean Teacher models are indeed flatter along these directions ( FIG0 ).•

In Section 3.2, we compare the training trajectories of the ⇧, Mean Teacher, and supervised models and find that the distances between the weights corresponding to different epochs are much larger for the consistency based models.

The error curves of consistency models are also wider ( FIG0 ), which can be explained by the flatness of the solutions discussed in section 3.1.

Further we observe that the predictions of the SGD iterates can differ significantly between different iterations of SGD.• We observe that for consistency-based methods, SGD does not converge to a single point but continues to explore many solutions with high distances apart.

Inspired by this observation, we propose to average the weights corresponding to SGD iterates, or ensemble the predictions of the models corresponding to these weights.

Averaging weights of SGD iterates compensates for larger steps, stabilizes SGD trajectories and obtains a solution that is centered in a flat region of the loss (as a function of weights).

Further, we show that the SGD iterates correspond to models with diverse predictions -using weight averaging or ensembling allows us to make use of the improved diversity and obtain a better solution compared to the SGD iterates.

In Section 3.3 we demonstrate that both ensembling predictions and averaging weights of the networks corresponding to different training epochs significantly improve generalization performance and find that the improvement is much larger for the ⇧ and Mean Teacher models compared to supervised training.

We find that averaging weights provides similar or improved accuracy compared to ensembling, while offering the computational benefits and convenience of working with a single model.

Thus, we focus on weight averaging for the remainder of the paper.• Motivated by our observations in Section 3 we propose to apply Stochastic Weight Averaging (SWA) BID12 to the ⇧ and Mean Teacher models.

Based on our results in Section 3.3 we propose several modifications to SWA in Section 4.

In particular, we propose fast-SWA, which (1) uses a learning rate schedule with longer cycles to increase the distance between the weights that are averaged and the diversity of the corresponding predictions; and (2) averages weights of multiple networks within each cycle (while SWA only averages weights corresponding to the lowest values of the learning rate within each cycle).

In Section 5, we show that fast-SWA converges to a good solution much faster than SWA.• Applying weight averaging to the ⇧ and Mean Teacher models we improve the best reported results on CIFAR-10 for 1k, 2k, 4k and 10k labeled examples, as well as on CIFAR-100 with 10k labeled examples.

For example, we obtain 5.0% error on CIFAR-10 with only 4k labels, improving the best result reported in the literature BID31 ) by 1.3%.

We also apply weight averaging to a state-of-the-art domain adaptation technique BID6 closely related to the Mean Teacher model and improve the best reported results on domain adaptation from CIFAR-10 to STL from 19.9% to 16.8% error.• We release our code at https://github.com/benathi/fastswa-semi-sup 2 BACKGROUND

We briefly review semi-supervised learning with consistency-based models.

This class of models encourages predictions to stay similar under small perturbations of inputs or network parameters.

For instance, two different translations of the same image should result in similar predicted probabilities.

The consistency of a model (student) can be measured against its own predictions (e.g. ⇧ model) or predictions of a different teacher network (e.g. Mean Teacher model).

In both cases we will say a student network measures consistency against a teacher network.

Consistency Loss In the semi-supervised setting, we have access to labeled data DISPLAYFORM0 , and unlabeled data D U = {x DISPLAYFORM1 .

Given two perturbed inputs x 0 , x 00 of x and the perturbed weights w 0 f and w 0 g , the consistency loss penalizes the difference between the student's predicted probablities f (x 0 ; w 0 f ) and the teacher's g(x 00 ; w 0 g ).

This loss is typically the Mean Squared Error or KL divergence: DISPLAYFORM2 The total loss used to train the model can be written as DISPLAYFORM3 where for classification L CE is the cross entropy between the model predictions and supervised training labels.

The parameter > 0 controls the relative importance of the consistency term in the overall loss.

⇧ Model The ⇧ model, introduced in BID17 and BID26 , uses the student model f as its own teacher.

The data (input) perturbations include random translations, crops, flips and additive Gaussian noise.

Binary dropout BID30 ) is used for weight perturbation.

Mean Teacher Model The Mean Teacher model (MT) proposed in BID31 uses the same data and weight perturbations as the ⇧ model; however, the teacher weights w g are the exponential moving average (EMA) of the student weights w f : DISPLAYFORM4 The decay rate ↵ is usually set between 0.9 and 0.999.

The Mean Teacher model has the best known results on the CIFAR-10 semi-supervised learning benchmark BID31 .Other Consistency-Based Models Temporal Ensembling (TE) BID17 uses an exponential moving average of the student outputs as the teacher outputs in the consistency term for training.

Another approach, Virtual Adversarial Training (VAT) BID21 , enforces the consistency between predictions on the original data inputs and the data perturbed in an adversarial direction x 0 = x + ✏r adv , where r adv = arg max r:krk=1 KL[f (x, w)kf (x + ⇠r, w)].

In Section 3.1, we study a simplified version of the ⇧ model theoretically and show that it penalizes the norm of the Jacobian of the outputs with respect to inputs, as well as the eigenvalues of the Hessian, both of which have been related to generalization BID29 BID22 BID4 BID3 .

In Section 3.2 we empirically study the training trajectories of the ⇧ and MT models and compare them to the training trajectories in supervised learning.

We show that even late in training consistency-based methods make large training steps, leading to significant changes in predictions on test.

In Section 3.3 we show that averaging weights or ensembling predictions of the models proposed by SGD at different training epochs can lead to substantial gains in accuracy and that these gains are much larger for ⇧ and MT than for supervised training.

Penalization of the input-output Jacobian norm.

Consider a simple version of the ⇧ model, where we only apply small additive perturbations to the student inputs: x 0 = x + ✏z, z ⇠ N (0, I) with ✏ ⌧ 1, and the teacher input is unchanged: x 00 = x. 1 Then the consistency loss`c ons (Eq. 1) becomes`c ons (w, x, ✏) = kf (w, x + ✏z) f (w, x)k 2 .

Consider the estimatorQ = lim ✏!

0 DISPLAYFORM0 We show in Section A.5 that DISPLAYFORM1 where J x is the Jacobian of the network's outputs with respect to its inputs evaluated at x, k · k F represents Frobenius norm, and the expectation E x is taken over the distribution of labeled and Published as a conference paper at ICLR 2019 unlabeled data.

That is,Q is an unbiased estimator of E x [kJ x k 2 F ] with variance controlled by the minibatch size m. Therefore, the consistency loss implicitly penalizes DISPLAYFORM2 The quantity ||J x || F has been related to generalization both theoretically BID29 and empirically BID22 .

For linear models f (x) = W x, penalizing ||J x || F exactly corresponds to weight decay, also known as L 2 regularization, since for linear models J x = W , and kW k DISPLAYFORM3 is also closely related to the graph based (manifold) regularization in BID34 which uses the graph Laplacian to approximate E x [kr M f k 2 ] for nonlinear models, making use of the manifold structure of unlabeled data.

Isotropic perturbations investigated in this simplified ⇧ model will not in general lie along the data manifold, and it would be more pertinent to enforce consistency to perturbations sampled from the space of natural images.

In fact, we can interpret consistency with respect to standard data augmentations (which are used in practice) as penalizing the manifold Jacobian norm in the same manner as above.

See Section A.5 for more details.

Penalization of the Hessian's eigenvalues.

Now, instead of the input perturbation, consider the weight perturbation w 0 = w + ✏z. Similarly, the consistency loss is an unbiased estimator for DISPLAYFORM4 , where J w is the Jacobian of the network outputs with respect to the weights w. In Section A.6 we show that for the MSE loss, the expected trace of the Hessian of the loss E x [tr(H)] can be decomposed into two terms, one of which is E x [kJ w k 2 F ].

As minimizing the consistency loss of a simplified ⇧ model penalizes DISPLAYFORM5 As pointed out in BID4 and BID3 , the eigenvalues of H encode the local information about sharpness of the loss for a given solution w. Consequently, the quantity tr(H) which is the sum of the Hessian eigenvalues is related to the notion of sharp and flat optima, which has recently gained attention as a proxy for generalization performance (see e.g. BID27 BID13 BID12 .

Thus, based on our analysis, the consistency loss in the simplified ⇧ model encourages flatter solutions.

In the previous section we have seen that in a simplified ⇧ model, the consistency loss encourages lower input-output Jacobian norm and Hessian's eigenvalues, which are related to better generalization.

In this section we analyze the properties of minimizing the consistency loss in a practical setting.

Specifically, we explore the trajectories followed by SGD for the consistency-based models and compare them to the trajectories in supervised training.

We train our models on CIFAR-10 using 4k labeled data for 180 epochs.

The ⇧ and Mean Teacher models use 46k data points as unlabeled data (see Sections A.8 and A.9 for details).

First, in FIG0 we visualize the evolution of norms of the gradients of the cross-entropy term krL CE k and consistency term krL cons k along the trajectories of the ⇧, MT, and standard supervised models (using CE loss only).

We observe that krL Cons k remains high until the end of training and dominates the gradient krL CE k of the cross-entropy term for the ⇧ and MT models.

Further, for both the ⇧ and MT models, krL Cons k is much larger than in supervised training implying that the ⇧ and MT models are making substantially larger steps until the end of training.

These larger steps suggest that rather than converging to a single minimizer, SGD continues to actively explore a large set of solutions when applied to consistency-based methods.

For further understand this observation, we analyze the behavior of train and test errors in the region of weight space around the solutions of the ⇧ and Mean Teacher models.

First, we consider the onedimensional rays (t) = t · w 180 + (1 t)w 170 , t 0, connecting the weight vectors w 170 and w 180 corresponding to epochs 170 and 180 of training.

We visualize the train and test errors (measured on the labeled data) as functions of the distance from the weights w 170 in FIG0 .

We observe that the distance between the weight vectors w 170 and w 180 is much larger for the semi-supervised methods compared to supervised training, which is consistent with our observation that the gradient norms are larger which implies larger steps during optimization in the ⇧ and MT models.

Further, we observe that the train and test error surfaces are much wider along the directions connecting w 170 and w 180 for the consistency-based methods compared to supervised training.

One possible explanation for the increased width is the effect of the consistency loss on the Jacobian of the network and the eigenvalues of the Hessian of the loss discussed in Section 3.1.

We also observe that the test errors of interpolated weights can be lower than errors of the two SGD solutions between which we interpolate.

This error reduction is larger in the consistency models ( FIG0 ).We also analyze the error surfaces along random and adversarial rays starting at the SGD solution w 180 for each model.

For the random rays we sample 5 random vectors d from the unit sphere and calculate the average train and test errors of the network with weights w t1 + sd for s 2 [0, 30].

With adversarial rays we evaluate the error along the directions of the fastest ascent of test or train loss DISPLAYFORM0 ||rL CE || .

We observe that while the solutions of the ⇧ and MT models are much wider than supervised training solutions along the SGD-SGD directions FIG0 , their widths along random and adversarial rays are comparable (Figure 1c, 1d) We analyze the error along SGD-SGD rays for two reasons.

Firstly, in fast-SWA we are averaging solutions traversed by SGD, so the rays connecting SGD iterates serve as a proxy for the space we average over.

Secondly, we are interested in evaluating the width of the solutions that we explore during training which we expect will be improved by the consistency training, as discussed in Section 3.1 and A.6.

We expect width along random rays to be less meaningful because there are many directions in the parameter space that do not change the network outputs BID5 GurAri et al., 2018; BID25 .

However, by evaluating SGD-SGD rays, we can expect that these directions corresponds to meaningful changes to our model because individual SGD updates correspond to directions that change the predictions on the training set.

Furthermore, we observe that different SGD iterates produce significantly different predictions on the test data.

Neural networks in general are known to be resilient to noise, explaining why both MT, ⇧ and supervised models are flat along random directions BID0 .

At the same time neural networks are susceptible to targeted perturbations (such as adversarial attacks).

We hypothesize that we do not observe improved flatness for semi-supervised methods along adversarial rays because we do not choose our input or weight perturbations adversarially, but rather they are sampled from a predefined set of transformations.

Additionally, we analyze whether the larger optimization steps for the ⇧ and MT models translate into higher diversity in predictions.

We define diversity of a pair of models w 1 , w 2 as Diversity( DISPLAYFORM1 ], the fraction of test samples where the predicted labels between the two models differ.

We found that for the ⇧ and MT models, the Diversity(w 170 , w 180 ) is 7.1% and 6.1% of the test data points respectively, which is much higher than 3.9% in supervised learning.

The increased diversity in the predictions of the networks traversed by SGD supports our conjecture that for the ⇧ and MT models SGD struggles to converge to a single solution and continues to actively explore the set of plausible solutions until the end of training.

In Section 3.2, we observed that the ⇧ and MT models continue taking large steps in the weight space at the end of training.

Not only are the distances between weights larger, we observe these models to have higher diversity.

In this setting, using the last SGD iterate to perform prediction is not ideal since many solutions explored by SGD are equally accurate but produce different predictions.

Ensembling.

In Section 3.2 we showed that the diversity in predictions is significantly larger for the ⇧ and Mean Teacher models compared to purely supervised learning.

The diversity of these iterates suggests that we can achieve greater benefits from ensembling.

We use the same CNN architecture and hyper-parameters as in Section 3.2 but extend the training time by doing 5 learning rate cycles of 30 epochs after the normal training ends at epoch 180 (see A.8 and A.9 for details).

We sample random pairs of weights w 1 , w 2 from epochs 180, 183, . . .

, 330 and measure the error reduction from ensembling these pairs of models, C ens ⌘ 1 2 Err(w 1 ) + 1 2 Err(w 2 ) Err (Ensemble(w 1 , w 2 )).

In FIG1 we visualize C ens , against the diversity of the corresponding pair of models.

We observe a strong correlation between the diversity in predictions of the constituent models and ensemble performance, and therefore C ens is substantially larger for ⇧ and Mean Teacher models.

As shown in BID12 , ensembling can be well approximated by weight averaging if the weights are close by.

Weight Averaging.

First, we experiment on averaging random pairs of weights at the end of training and analyze the performance with respect to the weight distances.

Using the the same pairs from above, we evaluate the performance of the model formed by averaging the pairs of weights, DISPLAYFORM0 Note that C avg is a proxy for convexity: if C avg (w 1 , w 2 ) 0 for any pair of points w 1 , w 2 , then by Jensen's inequality the error function is convex (see the left panel of FIG1 ).

While the error surfaces for neural networks are known to be highly non-convex, they may be approximately convex in the region traversed by SGD late into training BID9 .

In fact, in FIG1 , we find that the error surface of the SGD trajectory is approximately convex due to C avg (w 1 , w 2 ) being mostly positive.

Here we also observe that the distances between pairs of weights are much larger for the ⇧ and MT models than for the supervised training; and as a result, weight averaging achieves a larger gain for these models.

In Section 3.2 we observed that for the ⇧ and Mean Teacher models SGD traverses a large flat region of the weight space late in training.

Being very high-dimensional, this set has most of its volume concentrated near its boundary.

Thus, we find SGD iterates at the periphery of this flat region (see FIG1 ).

We can also explain this behavior via the argument of BID20 .

Under certain assumptions SGD iterates can be thought of as samples from a Gaussian distribution centered at the minimum of the loss, and samples from high-dimensional Gaussians are known to be concentrated on the surface of an ellipse and never be close to the mean.

Averaging the SGD iterates (shown in red in FIG1 ) we can move towards the center (shown in blue) of the flat region, stabilizing the SGD trajectory and improving the width of the resulting solution, and consequently improving generalization.

We observe that the improvement C avg from weight averaging (1.2 ± 0.2% over MT and ⇧ pairs) is on par or larger than the benefit C ens of prediction ensembling (0.9 ± 0.2%) The smaller gain from ensembling might be due to the dependency of the ensembled solutions, since they are from the same SGD run as opposed to independent restarts as in typical ensembling settings.

For the rest of the paper, we focus attention on weight averaging because of its lower costs at test time and slightly higher performance compared to ensembling.

In Section 3 we analyzed the training trajectories of the ⇧, MT, and supervised models.

We observed that the ⇧ and MT models continue to actively explore the set of plausible solutions, producing diverse predictions on the test set even in the late stages of training.

Further, in section 3.3 we have seen that averaging weights leads to significant gains in performance for the ⇧ and MT models.

In particular these gains are much larger than in supervised setting.

Stochastic Weight Averaging (SWA) BID12 ) is a recent approach that is based on averaging weights traversed by SGD with a modified learning rate schedule.

In Section 3 we analyzed averaging pairs of weights corresponding to different epochs of training and showed that it improves the test accuracy.

Averaging multiple weights reinforces this effect, and SWA was shown to significantly improve generalization performance in supervised learning.

Based on our results in section 3.3, we can expect even larger improvements in generalization when applying SWA to the ⇧ and MT models.

SWA typically starts from a pre-trained model, and then averages points in weight space traversed by SGD with a constant or cyclical learning rate.

We illustrate the cyclical cosine learning rate schedule in FIG2 (left) and the SGD solutions explored in FIG2 (middle).

For the first``0 epochs the network is pre-trained using the cosine annealing schedule where the learning rate at epoch i is set equal to ⌘(i) = 0.5 · ⌘ 0 (1 + cos (⇡ · i/`0)).

After`epochs, we use a cyclical schedule, repeating the learning rates from epochs [` c,`] , where c is the cycle length.

SWA collects the networks corresponding to the minimum values of the learning rate (shown in green in FIG2 , left) and averages their weights.

The model with the averaged weights w SWA is then used to make predictions.

We propose to apply SWA to the student network both for the ⇧ and Mean Teacher models.

Note that the SWA weights do not interfere with training.

Originally, BID12 proposed using cyclical learning rates with small cycle length for SWA.

However, as we have seen in Section 3.3 FIG1 , left) the benefits of averaging are the most prominent when the distance between the averaged points is large.

Motivated by this observation, we instead use longer learning rate cycles c. Moreover, SWA updates the average weights only once per cycle, which means that many additional training epochs are needed in order to collect enough weights for averaging.

To overcome this limitation, we propose fast-SWA, a modification of SWA that averages networks corresponding to every k < c epochs starting from epoch` c. We can also average multiple weights within a single epoch setting k < 1.Notice that most of the models included in the fast-SWA average (shown in red in FIG2 , left) have higher errors than those included in the SWA average (shown in green in FIG2 , right) since they are obtained when the learning rate is high.

It is our contention that including more models in the fast-SWA weight average can more than compensate for the larger errors of the individual models.

Indeed, our experiments in Section 5 show that fast-SWA converges substantially faster than SWA and has lower performance variance.

We analyze this result theoretically in Section A.7).

We evaluate the ⇧ and MT models (Section 4) on CIFAR-10 and CIFAR-100 with varying numbers of labeled examples.

We show that fast-SWA and SWA improve the performance of the ⇧ and MT models, as we expect from our observations in Section 3.

In fact, in many cases fast-SWA improves on the best results reported in the semi-supervised literature.

We also demonstrate that the preposed fast-SWA obtains high performance much faster than SWA.

We also evaluate SWA applied to a consistency-based domain adaptation model BID6 , closely related to the MT model, for adapting CIFAR-10 to STL.

We improve the best reported test error rate for this task from 19.9% to 16.8%.We discuss the experimental setup in Section 5.1.

We provide the results for CIFAR-10 and CIFAR-100 datasets in Section 5.2 and 5.3.

We summarize our results in comparison to the best previous results in Section 5.4.

We show several additional results and detailed comparisons in Appendix A.2.

We provide analysis of train and test error surfaces of fast-SWA solutions along the directions connecting fast-SWA and SGD in Section A.1.

We evaluate the weight averaging methods SWA and fast-SWA on different network architectures and learning rate schedules.

We are able to improve on the base models in all settings.

In particular, we consider a 13-layer CNN and a 12-block (26-layer) Residual Network BID11 with ShakeShake regularization BID7 , which we refer to simply as CNN and Shake-Shake respectively (see Section A.8 for details on the architectures).

For training all methods we use the stochastic gradient descent (SGD) optimizer with the cosine annealing learning rate described in Section 4.

We use two learning rate schedules, the short schedule with`= 180,`0 = 210, c = 30, similar to the experiments in BID31 , and the long schedule with`= 1500,`0 = 1800, c = 200, similar to the experiments in BID7 .

We note that the long schedule improves the performance of the base models compared to the short schedule; however, SWA can still further improve the results.

See Section A.9 of the Appendix for more details on other hyperparameters.

We repeat each CNN experiment 3 times with different random seeds to estimate the standard deviations for the results in the Appendix.

CIFAR-100 with CNN.

50k+ and 50k+ ⇤ correspond to 50k+500k and 50k+237k ⇤ settings (c) CIFAR-10 with ResNet + Shake-Shake using the short schedule (d) CIFAR-10 with ResNet + Shake-Shake using the long schedule.

We evaluate the proposed fast-SWA method using the ⇧ and MT models on the CIFAR-10 dataset (Krizhevsky) .

We use 50k images for training with 1k, 2k, 4k, 10k and 50k labels and report the top-1 errors on the test set (10k images).

We visualize the results for the CNN and Shake-Shake architectures in FIG3 , and 4d.

For all quantities of labeled data, fast-SWA substantially improves test accuracy in both architectures.

Additionally, in Tables 2, 4 of the Appendix we provide a thorough comparison of different averaging strategies as well as results for VAT BID21 , TE BID16 , and other baselines.

Note that we applied fast-SWA for VAT as well which is another popular approach for semi-supervised learning.

We found that the improvement on VAT is not drastic -our base implementation obtains 11.26% error where fast-SWA reduces it to 10.97% (see Table 2 in Section A.2).

It is possible that the solutions explored by VAT are not as diverse as in ⇧ and MT models due to VAT loss function.

Throughout the experiments, we focus on the ⇧ and MT models as they have been shown to scale to powerful networks such as Shake-Shake and obtained previous state-of-the-art performance.

In Figure 5 (left), we visualize the test error as a function of iteration using the CNN.

We observe that when the cyclical learning rate starts after epoch`= 180, the base models drop in performance due to the sudden increase in learning rate (see FIG2 ).

However, fast-SWA continues to improve while collecting the weights corresponding to high learning rates for averaging.

In general, we also find that the cyclical learning rate improves the base models beyond the usual cosine annealing schedule and increases the performance of fast-SWA as training progresses.

Compared to SWA, we also observe that fast-SWA converges substantially faster, for instance, reducing the error to 10.5% at epoch 200 while SWA attains similar error at epoch 350 for CIFAR-10 4k labels (Figure 5 left) .

We provide additional plots in Section A.2 showing the convergence of the ⇧ and MT models in all label settings, where we observe similar trends that fast-SWA results in faster error reduction.

We also find that the performance gains of fast-SWA over base models are higher for the ⇧ model compared to the MT model, which is consistent with the convexity observation in Section 3.3 and FIG1 .

In the previous evaluations (see e.g. BID23 BID31 ), the ⇧ model was shown to be inferior to the MT model.

However, with weight averaging, fast-SWA reduces the gap between ⇧ and MT performance.

Surprisingly, we find that the ⇧ model can outperform MT after applying fast-SWA with moderate to large numbers of labeled points.

In particular, the ⇧+fast-SWA model outperforms MT+fast-SWA on CIFAR-10 with 4k, 10k, and 50k labeled data points for the Shake-Shake architecture.

Figure 5: Prediction errors of base models and their weight averages (fast-SWA and SWA) for CNN on (left) CIFAR-10 with 4k labels, (middle) CIFAR-100 with 10k labels, and (right) CIFAR-100 50k labels and extra 500k unlabeled data from Tiny Images BID32 .

We evaluate the ⇧ and MT models with fast-SWA on CIFAR-100.

We train our models using 50000 images with 10k and 50k labels using the 13-layer CNN.

We also analyze the effect of using the Tiny Images dataset BID32 as an additional source of unlabeled data.

The Tiny Images dataset consists of 80 million images, mostly unlabeled, and contains CIFAR-100 as a subset.

Following Laine and Aila (2016), we use two settings of unlabeled data, 50k+500k and 50k+237k ⇤ where the 50k images corresponds to CIFAR-100 images from the training set and the +500k or +237k ⇤ images corresponds to additional 500k or 237k images from the Tiny Images dataset.

For the 237k ⇤ setting, we select only the images that belong to the classes in CIFAR-100, corresponding to 237203 images.

For the 500k setting, we use a random set of 500k images whose classes can be different from CIFAR-100.

We visualize the results in FIG3 , where we again observe that fast-SWA substantially improves performance for every configuration of the number of labeled and unlabeled data.

In Figure 5 (middle, right) we show the errors of MT, SWA and fast-SWA as a function of iteration on CIFAR-100 for the 10k and 50k+500k label settings.

Similar to the CIFAR-10 experiments, we observe that fast-SWA reduces the errors substantially faster than SWA.We provide detailed experimental results in Table 3 of the Appendix and include preliminary results using the Shake-Shake architecture in Table 5 , Section A.2.

We have shown that fast-SWA can significantly improve the performance of both the ⇧ and MT models.

We provide a summary comparing our results with the previous best results in the literature in Table 1 , using the 13-layer CNN and the Shake-Shake architecture that had been applied previously.

We also provide detailed results the Appendix A.2.

Table 1 : Test errors against current state-of-the-art semi-supervised results.

The previous best numbers are obtained from BID31 1 , BID24 2 , BID16 3 and BID19 4 .

CNN denotes performance on the benchmark 13-layer CNN (see A.8).

Rows marked † use the Shake-Shake architecture.

The result marked ‡ are from ⇧ + fast-SWA, where the rest are based on MT + fast-SWA.

The settings 50k+500k and 50k+237k ⇤ use additional 500k and 237k unlabeled data from the Tiny Images dataset BID32 where ⇤ denotes that we use only the images that correspond to CIFAR-100 classes.

6 DISCUSSION Semi-supervised learning is crucial for reducing the dependency of deep learning on large labeled datasets.

Recently, there have been great advances in semi-supervised learning, with consistency regularization models achieving the best known results.

By analyzing solutions along the training trajectories for two of the most successful models in this class, the ⇧ and Mean Teacher models, we have seen that rather than converging to a single solution SGD continues to explore a diverse set of plausible solutions late into training.

As a result, we can expect that averaging predictions or weights will lead to much larger gains in performance than for supervised training.

Indeed, applying a variant of the recently proposed stochastic weight averaging (SWA) we advance the best known semi-supervised results on classification benchmarks.

While not the focus of our paper, we have also shown that weight averaging has great promise in domain adaptation BID6 .

We believe that application-specific analysis of the geometric properties of the training objective and optimization trajectories will further improve results over a wide range of application specific areas, including reinforcement learning with sparse rewards, generative adversarial networks BID33 , or semi-supervised natural language processing.

Figure 6: All plots are a obtained using the 13-layer CNN on CIFAR-10 with 4k labeled and 46k unlabeled data points unless specified otherwise.

Left: Test error as a function of distance along random rays for the ⇧ model with 0, 4k, 10k, 20k or 46k unlabeled data points, and standard fully supervised training which uses only the cross entropy loss.

All methods use 4k labeled examples.

Middle: Train and test errors along rays connecting SGD solutions (showed with circles) to SWA solutions (showed with squares) for each respective model.

Right: Comparison of train and test errors along rays connecting two SGD solutions, random rays, and adversarial rays for the Mean Teacher model.

In this section we provide several additional plots visualizing the train and test error along different types of rays in the weight space.

The left panel of Figure 6 shows how the behavior of test error changes as we add more unlabeled data points for the ⇧ model.

We observe that the test accuracy improves monotonically, but also the solutions become narrower along random rays.

The middle panel of Figure 6 visualizes the train and test error behavior along the directions connecting the fast-SWA solution (shown with squares) to one of the SGD iterates used to compute the average (shown with circles) for ⇧, MT and supervised training.

Similarly to BID12 we observe that for all three methods fast-SWA finds a centered solution, while the SGD solution lies near the boundary of a wide flat region.

Agreeing with our results in section 3.2 we observe that for ⇧ and Mean Teacher models the train and test error surfaces are much wider along the directions connecting the fast-SWA and SGD solutions than for supervised training.

In the right panel of Figure 6 we show the behavior of train and test error surfaces along random rays, adversarial rays and directions connecting the SGD solutions from epochs 170 and 180 for the Mean Teacher model (see section 3.2).

In the left panel of FIG5 we show the evolution of the trace of the gradient of the covariance of the loss tr cov(r w L(w)) = Ekr w L(w) Er w L(w)k 2 for the ⇧, MT and supevised training.

We observe that the variance of the gradient is much larger for the ⇧ and Mean Teacher models compared to supervised training.

In the middle and right panels of figure 7 we provide scatter plots of the improvement C obtained from averaging weights against diversity and diversity against distance.

We observe that diversity is highly correlated with the improvement C coming from weight averaging.

The correlation between distance and diversity is less prominent.

In this section we report detailed results for the ⇧ and Mean Teacher models and various baselines on CIFAR-10 and CIFAR-100 using the 13-layer CNN and Shake-Shake.

The results using the 13-layer CNN are summarized in Tables 2 and 3 for CIFAR-10 and CIFAR-100 respectively.

Tables 4 and 5 summarize the results using Shake-Shake on CIFAR-10 and CIFAR-100.

In the tables ⇧ EMA is the same method as ⇧, where instead of SWA we apply Exponential Moving Averaging (EMA) for the student weights.

We show that simply performing EMA for the student network in the ⇧ model without using it as a teacher (as in MT) typically results in a small improvement in the test error.

Figures 8 and 9 show the performance of the ⇧ and Mean Teacher models as a function of the training epoch for CIFAR-10 and CIFAR-100 respectively for SWA and fast-SWA.

FORMULA2 18.19 ± 0.38 13.46 ± 0.30 10.67 ± 0.18 8.06 ± 0.12 5.90 ± 0.03 MT + fast-SWA FORMULA3 17.81 ± 0.37 13.00 ± 0.31 10.34 ± 0.14 7.73 ± 0.10 5.55 ± 0.03 MT + SWA FORMULA3 18.38 ± 0.29 13.86 ± 0.64 10.95 ± 0.21 8.36 ± 0.50 5.75 ± 0.29 MT + fast-SWA (480) 16.84 ± 0.62 12.24 ± 0.31 9.86 ± 0.27 7.39 ± 0.14 5.14 ± 0.07 MT + SWA (480) 17.48 ± 0.13 13.09 ± 0.80 10.30 ± 0.21 7.78 ± 0.49 5.31 ± 0.43 MT + fast-SWA (1200) 15.58 ± 0.12 11.02 ± 0.23 9.05 ± 0.21 6.92 ± 0.07 4.73 ± 0.18 MT + SWA FORMULA2 15.59 ± 0.77 11.42 ± 0.33 9.38 ± 0.28 7.04 ± 0.11 5.11 ± 0.35 ⇧ 21.85 ± 0.69 16.10 ± 0.51 12.64 ± 0.11 9.11 ± 0.21 6.79 ± 0.22 ⇧ EMA 21.70 ± 0.57 15.83 ± 0.55 12.52 ± 0.16 9.06 ± 0.15 6.66 ± 0.20 ⇧ + fast-SWA FORMULA2 20.79 ± 0.38 15.12 ± 0.44 11.91 ± 0.06 8.83 ± 0.32 6.42 ± 0.09 ⇧ + fast-SWA FORMULA3 20.04 ± 0.41 14.77 ± 0.15 11.61 ± 0.06 8.45 ± 0.28 6.14 ± 0.11 ⇧ + SWA FORMULA3 21.37 ± 0.64 15.38 ± 0.85 12.05 ± 0.40 8.58 ± 0.41 6.36 ± 0.55 ⇧ + fast-SWA (480) 19.11 ± 0.29 13.88 ± 0.30 10.91 ± 0.15 7.91 ± 0.21 5.53 ± 0.07 ⇧ + SWA (480) 20.06 ± 0.64 14.53 ± 0.81 11.35 ± 0.42 8.04 ± 0.37 5.77 ± 0.51 ⇧ + fast-SWA FORMULA2 17.23 ± 0.34 12.61 ± 0.18 10.07 ± 0.27 7.28 ± 0.23 4.72 ± 0.04 ⇧ + SWA FORMULA2 17.70 ± 0.25 12.59 ± 0.29 10.73 ± 0.39 7.13 ± 0.23 4.99 ± 0.41 VAT 11.99 VAT + SWA 11.16 VAT+ EntMin 11.26 VAT + EntMin + SWA 10.97 Table 3 : CIFAR-100 semi-supervised errors on test set.

All models are trained on a 13-layer CNN.

The epoch numbers are reported in parenthesis.

The previous results shown in the first section of the table are obtained from BID16 3 .Number of labels 10k 50k 50k + 500k 50k + 237k FORMULA2 34.54 ± 0.48 21.93 ± 0.16 21.04 ± 0.16 21.09 ± 0.12 MT + SWA FORMULA3 35.59 ± 1.45 23.17 ± 0.86 22.00 ± 0.23 21.59 ± 0.22 MT + fast-SWA FORMULA3 34.10 ± 0.31 21.84 ± 0.12 21.16 ± 0.21 21.07 ± 0.21 MT + SWA FORMULA2 34.90 ± 1.51 22.58 ± 0.79 21.47 ± 0.29 21.27 ± 0.09 MT + fast-SWA (1200) 33.62 ± 0.54 21.52 ± 0.12 21.04 ± 0.04 20.98 ± 0.36 DISPLAYFORM0 38.13 ± 0.52 24.13 ± 0.20 24.26 ± 0.15 24.10 ± 0.07 ⇧ + fast-SWA FORMULA2 35.59 ± 0.62 22.08 ± 0.21 21.40 ± 0.19 21.28 ± 0.20 ⇧ + SWA FORMULA3 36.89 ± 1.51 23.23 ± 0.70 22.17 ± 0.19 21.65 ± 0.13 ⇧ + fast-SWA FORMULA3 35.14 ± 0.71 22.00 ± 0.21 21.29 ± 0.27 21.22 ± 0.04 ⇧ + SWA FORMULA2 35.35 ± 1.15 22.53 ± 0.64 21.53 ± 0.13 21.26 ± 0.34 ⇧ + fast-SWA FORMULA2 34.25 ± 0.16 21.78 ± 0.05 21.19 ± 0.05 20.97 ± 0.08 Table 4 : CIFAR-10 semi-supervised errors on test set.

All models use Shake-Shake Regularization (Gastaldi, 2017) + ResNet-26 BID11 .Number of labels 1000 2000 4000 10000 50000Short Schedule (`= 180) MT † BID31 6.28MT (180) 10.2 8.0 7.1 5.8 3.9 MT + SWA (240) 9.7 7.7 6.2 4.9 3.4 MT + fast-SWA (240) 9.6 7.4 6.2 4.9 3.2 MT + SWA (1200) 7.6 6.4 5.8 4.6 3.1 MT + fast-SWA (1200) 7.5 6.3 5.8 4.5 3.1 ⇧ (180) 12.3 9.1 7.5 6.4 3.8 ⇧ + SWA FORMULA3 11.0 8.3 6.7 5.5 3.3 ⇧ + fast-SWA FORMULA3 11.2 8.2 6.7 5.5 3.3 ⇧ + SWA FORMULA2 8.2 6.7 5.7 4.2 3.1 ⇧ + fast-SWA FORMULA2 8.0 6.5 5.5 4.0 3.1Long Schedule (`= 1500) Supervised-only BID7 2.86 MT FORMULA2 7.5 6.5 6.0 5.0 3.5 MT + fast-SWA FORMULA2 6.4 5.8 5.2 3.8 3.4 MT + SWA FORMULA2 6.9 5.9 5.5 4.2 3.2 MT + fast-SWA (3500) 6.6 5.7 5.1 3.9 3.1 MT + SWA (3500) 6.7 5.8 5.2 3.9 3.1 ⇧ (1500) 8.5 7.0 6.3 5.0 3.4 ⇧ + fast-SWA (1700) 7.5 6.2 5.2 4.0 3.1 ⇧ + SWA (1700) 7.8 6.4 5.6 4.4 3.2 ⇧ + fast-SWA (3500) 7.4 6.0 5.0 3.8 3.0 ⇧ + SWA (3500) 7.9 6.2 5.1 4.0 3.0 Table 5 : CIFAR-100 semi-supervised errors on test set.

Our models use Shake-Shake Regularization BID7 ) + ResNet-26 BID11 .Number of labels 10k 50k 50k + 500k 50k + 237k ⇤ TE (CNN) BID16 Figure 8: Test errors as a function of training epoch for baseline models, SWA and fast-SWA on CIFAR-10 trained using 1k, 2k, 4k, and 10k labels for (top) the MT model (bottom) the ⇧ model.

All models are trained using the 13-layer CNN.Figure 9: Test errors versus training epoch for baseline models, SWA and fast-SWA on CIFAR-100 trained using 10k, 50k, 50k+500k, and 50k+237k ⇤ labels for (top) the MT model (bottom) the ⇧ model.

All models are trained using the 13-layer CNN.

The only hyperparameter for the fast-SWA setting is the cycle length c. We demonstrate in FIG0 that fast-SWA's performance is not sensitive to c over a wide range of c values.

We also demonstrate the performance for constant learning schedule.

fast-SWA with cyclical learning rates generally converges faster due to higher variety in the collected weights.

FIG2 , left) at which the learning rate is evaluated.

We use this fixed learning rate for all epochs i `.

The MT model uses an exponential moving average (EMA) of the student weights as a teacher in the consistency regularization term.

We consider two potential effects of using EMA as a teacher: first, averaging weights improves performance of the teacher for the reasons discussed in Sections 3.2, 3.3; second, having a better teacher model leads to better student performance which in turn further improves the teacher.

In this section we try to separate these two effects.

We apply EMA to the ⇧ model in the same way in which we apply fast-SWA instead of using EMA as a teacher and compare the resulting performance to the Mean Teacher.

FIG0 shows the improvement in error-rate obtained by applying EMA to the ⇧ model in different label settings.

As we can see while EMA improves the results over the baseline ⇧ model, the performance of ⇧-EMA is still inferior to that of the Mean Teacher method, especially when the labeled data is scarce.

This observation suggests that the improvement of the Mean Teacher over the ⇧ model can not be simply attributed to EMA improving the student performance and we should take the second effect discussed above into account.

Like SWA, EMA is a way to average weights of the networks, but it puts more emphasis on very recent models compared to SWA.

Early in training when the student model changes rapidly EMA significantly improves performance and helps a lot when used as a teacher.

However once the student model converges to the vicinity of the optimum, EMA offers little gain.

In this regime SWA is a much better way to average weights.

We show the performance of SWA applied to ⇧ model in FIG0 (left).

Since SWA performs better than EMA, we also experiment with using SWA as a teacher instead of EMA.

We start with the usual MT model pretrained until epoch 150.

Then we switch to using SWA as a teacher at epoch 150.

In FIG0 (right), our results suggest that using SWA as a teacher performs on par with using EMA as a teacher.

We conjecture that once we are at a convex region of test error close to the optimum (epoch 150), having a better teacher doesn't lead to substantially improved performance.

It is possible to start using SWA as a teacher earlier in training; however, during early epochs where the model undergoes rapid improvement EMA is more sensible than SWA as we discussed above.

Estimator mean and variance: In the simplified ⇧ model with small additive data perturbations that are normally distributed, z ⇠ N (0, I), DISPLAYFORM0 Taylor expanding`c ons in ✏, we obtain`c ons (w, DISPLAYFORM1 , where J x is the Jacobian of the network outputs f with respect to the input at a particular value of x. Therefore, DISPLAYFORM2 We can now recognize this term as a one sample stochastic trace estimator for tr(J(x i ) T J(x i )) with a Gaussian probe variable z i ; see BID1 for derivations and guarantees on stochastic trace estimators.

DISPLAYFORM3 In general if we have m samples of x and n sampled perturbations for each x, then for a symmetric matrix A with z ik iid ⇠ N (0, I) and independent x i iid ⇠ p(x), the estimatorQ DISPLAYFORM4 2 , (see e.g. BID1 .

DISPLAYFORM5 whereas this does not hold for the opposite ordering of the sum.

DISPLAYFORM6 Plugging in A = J T J and n = 1, we get DISPLAYFORM7 Published as a conference paper at ICLR 2019Non-isotropic perturbations along data manifold Consistency regularization with natural perturbations such as image translation can also be understood as penalizing a Jacobian norm as in Section 3.1.

For example, consider perturbations sampled from a normal distribution on the tangent space, z ⇠ P (x)N (0, I) where P (x) = P (x) 2 is the orthogonal projection matrix that projects down from R d to T x (M), the tangent space of the image manifold at x. Then the consistency regularization penalizes the Laplacian norm of the network on the manifold (with the inherited metric from R d ).

E[z] = 0 and E[zz T ] = P P T (=)P 2 = P which follows if P is an orthogonal projection matrix.

Then, DISPLAYFORM8 We view the standard data augmentations such as random translation (that are applied in the ⇧ and MT models) as approximating samples of nearby elements of the data manifold and therefore differences x 0 x approximate elements of its tangent space.

DISPLAYFORM9 In the following analysis we review an argument for why smaller E x [kJ w k 2 F ], implies broader optima.

To keep things simple, we focus on the MSE loss, but in principle a similar argument should apply for the Cross Entropy and the Error rate.

For a single data point x and one hot vector y with k classes, the hessian of`M SE (w) = kf (x, w) yk 2 can be decomposed into two terms, the Gauss-Newton matrix G = J T w J w and a term which depends on the labels.

DISPLAYFORM10 Thus tr(H) is also the sum of two terms, kJ w k 2 F and ↵.

As the solution improves, the relative size of ↵ goes down.

In terms of random ray sharpness, consider the expected MSE loss, or risk, R MSE (w) = E (x,y) kf (x, w) yk 2 along random rays.

Let d be a random vector sampled from the unit sphere and s is the distance along the random ray.

Evaluating the risk on a random ray, and Taylor expanding in s we have

In the experiments we use two DNN architectures -13 layer CNN and Shake-Shake.

The architecture of 13-layer CNN is described in TAB3 .

It closely follows the architecture used in BID17 BID21 BID31 .

We re-implement it in PyTorch and removed the Gaussian input noise, since we found having no such noise improves generalization.

For Shake-Shake we use 26-2x96d Shake-Shake regularized architecture of BID7 with 12 residual blocks.

We consider two different schedules.

In the short schedule we set the cosine half-period`0 = 210 and training length`= 180, following the schedule used in BID31 in Shake-Shake experiments.

For our Shake-Shake experiments we also report results with long schedule where we set`= 1800,`0 = 1500 following BID7 .

To determine the initial learning rate ⌘ 0 and the cycle length c we used a separate validation set of size 5000 taken from the unlabeled data.

After determining these values, we added the validation set to the unlabeled data and trained again.

We reuse the same values of ⌘ 0 and c for all experiments with different numbers of labeled data for both ⇧ model and Mean Teacher for a fixed architecture (13-layer CNN or Shake-Shake).

For the short schedule we use cycle length c = 30 and average models once every k = 3 epochs.

For long schedule we use c = 200, k = 20.In all experiments we use stochastic gradient descent optimizer with Nesterov momentum BID18 .

In fast-SWA we average every the weights of the models corresponding to every third epoch.

In the ⇧ model, we back-propagate the gradients through the student side only (as opposed to both sides in BID16 ).

For Mean Teacher we use ↵ = 0.97 decay rate in the Exponential Moving Average (EMA) of the student's weights.

For all other hyper-parameters we reuse the values from BID31 unless mentioned otherwise.

Like in BID31 , we use k · k 2 for divergence in the consistency loss.

Similarly, we ramp up the consistency cost over the first 5 epochs from 0 up to it's maximum value of 100 as done in BID31 .

We use cosine annealing learning rates with no learning rate ramp up, unlike in the original MT implementation BID31 .

Note that this is similar to the same hyperparameter settings as in BID31 for ResNet 2 .

We note that we use the exact same hyperparameters for the ⇧ and MT models in each experiment setting.

In contrast to the original implementation in BID31 of CNN experiments, we use SGD instead of Adam.

Understanding Experiments in Sections 3.2, 3.3 We use the 13-layer CNN with the short learning rate schedule.

We use a total batch size of 100 for CNN experiments with a labeled batch size of 50 for the ⇧ and Mean Teacher models.

We use the maximum learning rate ⌘ 0 = 0.1.

For Section 3.2 we run SGD only for 180 epochs, so 0 learning rate cycles are done.

For Section 3.3 we additionally run 5 learning rate cycles and sample pairs of SGD iterates from epochs 180-330 corresponding to these cycles.

We use a total batch size of 100 for CNN experiments with a labeled batch size of 50.

We use the maximum learning rate ⌘ 0 = 0.1.

ResNet + Shake-Shake We use a total batch size of 128 for ResNet experiments with a labeled batch size of 31.

We use the maximum learning rate ⌘ 0 = 0.05 for CIFAR-10.

This applies for both the short and long schedules.

We use a total batch size of 128 with a labeled batch size of 31 for 10k and 50k label settings.

For the settings 50k+500k and 50k+237k ⇤ , we use a labeled batch size of 64.

We also limit the number of unlabeled images used in each epoch to 100k images.

We use the maximum learning rate ⌘ 0 = 0.1.

ResNet + Shake-Shake We use a total batch size of 128 for ResNet experiments with a labeled batch size of 31 in all label settings.

For the settings 50k+500k and 50k+237k ⇤ , we also limit the number of unlabeled images used in each epoch to 100k images.

We use the maximum learning rate ⌘ 0 = 0.1.

This applies for both the short and long schedules.

We apply fast-SWA to the best experiment setting MT+CT+TFA for CIFAR-10 to STL according to BID6 .

This setting involves using confidence thresholding (CT) and also an augmentation scheme with translation, flipping, and affine transformation (TFA).We modify the optimizer to use SGD instead of Adam BID14 and use cosine annealing schedule with`0 = 600,`= 550, c = 50.

We experimented with two fast-SWA methods: averaging weights once per epoch and averaging once every iteration, which is much more frequent that averaging every epoch as in the semi-supervised case.

Interestingly, we found that for this task averaging the weights in the end of every iteration in fast-SWA converges significantly faster than averaging once per epoch and results in better performance.

We report the results in Table 7 .We observe that averaging every iteration converges much faster (600 epochs instead of 3000) and results in better test accuracy.

In our experiments with semi-supervised learning averaging more often than once per epoch didn't improve convergence or final results.

We hypothesize that the improvement from more frequent averaging is a result of specific geometry of the loss surfaces and training trajectories in domain adaptation.

We leave further analysis of applying fast-SWA to domain adaptation for future work.

Implementation Details We use the public code 3 of BID6 to train the model and apply fast-SWA.

While the original implementation uses Adam BID14 , we use stochastic gradient descent with Nesterov momentum and cosine annealing learning rate with`0 = 600,`= 550, c = 100 and k = 100.

We use the maximum learning rate ⌘ 0 = 0.1 and momentum Table 7 : Domain Adaptation from CIFAR-10 to STL.

VADA results are from BID28 and the original SE ⇤ is from BID6 .

SE is the score with our implementation without fast-SWA.

0.9 with weight decay of scale 2 ⇥ 10 4 .

We use the data augmentation setting MT+CF+TFA in Table 1 of BID6 and apply fast-SWA.

The result reported is from epoch 4000.

@highlight

Consistency-based models for semi-supervised learning do not converge to a single point but continue to explore a diverse set of plausible solutions on the perimeter of a flat region. Weight averaging helps improve generalization performance.

@highlight

The paper proposes to apply Stochastic Weight Averaging to the semi-supervised learning context, arguing that the semi-supervised MT/Pi models are especially amenable to SWA and propose fast SWA to speed up training.