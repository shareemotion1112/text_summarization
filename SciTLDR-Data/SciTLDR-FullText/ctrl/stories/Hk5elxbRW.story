The top-$k$ error is a common measure of performance in machine learning and computer vision.

In practice, top-$k$ classification is typically performed with deep neural networks trained with the cross-entropy loss.

Theoretical results indeed suggest that cross-entropy is an optimal learning objective for such a task in the limit of infinite data.

In the context of limited and noisy data however, the use of a loss function that is specifically designed for top-$k$ classification can bring significant improvements.

Our empirical evidence suggests that the loss function must be smooth and have non-sparse gradients in order to work well with deep neural networks.

Consequently, we introduce a family of smoothed loss functions that are suited to top-$k$ optimization via deep learning.

The widely used cross-entropy is a special case of our family.

Evaluating our smooth loss functions is computationally challenging: a na{\"i}ve algorithm would require $\mathcal{O}(\binom{n}{k})$ operations, where $n$ is the number of classes.

Thanks to a connection to polynomial algebra and a divide-and-conquer approach, we provide an algorithm with a time complexity of $\mathcal{O}(k n)$. Furthermore, we present a novel approximation to obtain fast and stable algorithms on GPUs with single floating point precision.

We compare the performance of the cross-entropy loss and our margin-based losses in various regimes of noise and data size, for the predominant use case of $k=5$. Our investigation reveals that our loss is more robust to noise and overfitting than cross-entropy.

In machine learning many classification tasks present inherent label confusion.

The confusion can originate from a variety of factors, such as incorrect labeling, incomplete annotation, or some fundamental ambiguities that obfuscate the ground truth label even to a human expert.

For example, consider the images from the ImageNet data set (Russakovsky et al., 2015) in Figure 1 , which illustrate the aforementioned factors.

To mitigate these issues, one may require the model to predict the k most likely labels, where k is typically very small compared to the total number of labels.

Then the prediction is considered incorrect if all of its k labels differ from the ground truth, and correct otherwise.

This is commonly referred to as the top-k error.

Learning such models is a longstanding task in machine learning, and many loss functions for top-k error have been suggested in the literature.

In the context of correctly labeled large data, deep neural networks trained with cross-entropy have shown exemplary capacity to accurately approximate the data distribution.

An illustration of this phenomenon is the performance attained by deep convolutional neural networks on the ImageNet challenge.

Specifically, state-of-the-art models trained with cross-entropy yield remarkable success on the top-5 error, although cross-entropy is not tailored for top-5 error minimization.

This phenomenon can be explained by the fact that cross-entropy is top-k calibrated for any k (Lapin et al., 2016) , an asymptotic property which is verified in practice in the large data setting.

However, in cases where only a limited amount of data is available, learning large models with cross-entropy can be prone to over-fitting on incomplete or noisy labels.

To alleviate the deficiency of cross-entropy, we present a new family of top-k classification loss functions for deep neural networks.

Taking inspiration from multi-class SVMs, our loss creates a Figure 1 : Examples of images with label confusion, from the validation set of ImageNet.

The top-left image is incorrectly labeled as "red panda", instead of "giant panda".

The bottom-left image is labeled as "strawberry", although the categories "apple", "banana" and "pineapple" would be other valid labels.

The center image is labeled as "indigo bunting", which is only valid for the lower bird of the image.

The right-most image is labeled as a cocktail shaker, yet could arguably be a part of a music instrument (for example with label "cornet, horn, trumpet, trump").

Such examples motivate the need to predict more than a single label per image.margin between the correct top-k predictions and the incorrect ones.

Our empirical results show that traditional top-k loss functions do not perform well in combination with deep neural networks.

We believe that the reason for this is the lack of smoothness and the sparsity of the derivatives that are used in backpropagation.

In order to overcome this difficulty, we smooth the loss with a temperature parameter.

The evaluation of the smooth function and its gradient is challenging, as smoothing increases the naïve time complexity from O(n) to O( n k ).

With a connection to polynomial algebra and a divide-and-conquer method, we present an algorithm with O(kn) time complexity and training time comparable to cross-entropy in practice.

We provide insights for numerical stability of the forward pass.

To deal with instabilities of the backward pass, we derive a novel approximation.

Our investigation reveals that our top-k loss outperforms cross-entropy in the presence of noisy labels or in the absence of large amounts of data.

We further confirm that the difference of performance reduces with large correctly labeled data, which is consistent with known theoretical results.

Top-k Loss Functions.

The majority of the work on top-k loss functions has been applied to shallow models: Lapin et al. (2016) suggest a convex surrogate on the top-k loss; Fan et al. (2017) select the k largest individual losses in order to be robust to data outliers; Chang et al. (2017) formulate a truncated re-weighted top-k loss as a difference-of-convex objective and optimize it with the Concave-Convex Procedure BID2 BID1 propose to use a combination of top-k classifiers and to fuse their outputs.

Closest to our work is the extensive review of top-k loss functions for computer vision by Lapin et al. (2017) .

The authors conduct a study of a number of top-k loss functions derived from cross-entropy and hinge losses.

Interestingly, they prove that for any k, cross-entropy is top-k calibrated, which is a necessary condition for the classifier to be consistent with regard to the theoretically optimal top-k risk.

In other words, cross-entropy satisfies an essential property to perform the optimal top-k classification decision for any k in the limit of infinite data.

This may explain why cross-entropy performs well on top-5 error on large scale data sets.

While thorough, the experiments are conducted on linear models, or pre-trained deep networks that are fine-tuned.

For a more complete analysis, we wish to design loss functions that allow for the training of deep neural networks from a random initialization.

Smoothing.

Smoothing is a helpful technique in optimization BID0 .

In work closely related to ours, Lee & Mangasarian (2001) show that smoothing a binary SVM with a temperature parameter improves the theoretical convergence speed of their algorithm.

Schwing et al. (2012) use a temperature parameter to smooth latent variables for structured prediction.

Lapin et al. (2017) apply Moreau-Yosida regularization to smooth their top-k surrogate losses.

Smoothing has also been applied in the context of deep neural networks.

In particular, BID3 and Clevert et al. (2016) both suggest modifying the non-smooth ReLU activation to improve the training.

Gulcehre et al. (2017) suggest to introduce "mollifyers" to smooth the objective function by gradually increasing the difficulty of the optimization problem.

Chaudhari et al. (2017) add a local entropy term to the loss to promote solutions with high local entropy.

These smoothing techniques are used to speed up the optimization or improve generalization.

In this work, we show that smoothing is necessary for the neural network to perform well in combination with our loss function.

We hope that this insight can also help the design of losses for tasks other than top-k error minimization.3 TOP-K SVM 3.1 BACKGROUND: MULTI-CLASS SVM In order to build an intuition about top-k losses, we start with the simple case of k = 1, namely multi-class classification, where the output space is defined as Y = {1, ..., n}. We suppose that a vector of scores per label s ∈ R n , and a ground truth label y ∈ Y are both given.

The vector s is the output of the model we wish to learn, for example a linear model or a deep neural network.

The notation 1 will refer to the indicator function over Boolean statements (1 if true, 0 if false).Prediction.

The prediction is given by any index with maximal score: DISPLAYFORM0 (1)Loss.

The classification loss incurs a binary penalty by comparing the prediction to the ground truth label.

Plugging in equation (1) , this can also be written in terms of scores s as follows: DISPLAYFORM1 Surrogate.

The loss in equation FORMULA1 is not amenable to optimization, as it is not even continuous in s. To overcome this difficulty, a typical approach in machine learning is to resort to a surrogate loss that provides a continuous upper bound on Λ. Crammer & Singer (2001) suggest the following upper bound on the loss, known as the multi-class SVM loss: DISPLAYFORM2 In other words, the surrogate loss is zero if the ground truth score is higher than all other scores by a margin of at least one.

Otherwise it incurs a penalty which is linear in the difference between the score of the ground truth and the highest score over all other classes.

Rescaling.

Note that the value of 1 as a margin is an arbitrary choice, and can be changed to α for any α > 0.

This simply entails that we consider the cost Λ of a misclassification to be α instead of 1.

Moreover, we show in Proposition 8 of Appendix D.2 how the choices of α and of the quadratic regularization hyper-parameter are interchangeable.

We now generalize the above framework to top-k classification, where k ∈ {1, ..., n − 1}. We use the following notation: for p ∈ {1, ..., n}, s [p] refers to the p-th largest element of s, and s \p to the vector (s 1 , ..., s p−1 , s p+1 , ..., s n ) ∈ R n−1 (that is, the vector s with the p-th element omitted).

The term Y (k) denotes the set of k-tuples with k distinct elements of Y. Note that we use a bold font for a tupleȳ ∈ Y (k) in order to distinguish it from a single labelȳ ∈ Y.Prediction.

Given the scores s ∈ R n , the top-k prediction consists of any set of labels corresponding to the k largest scores: DISPLAYFORM0 Loss.

The loss depends on whether y is part of the top-k prediction, which is equivalent to comparing the k-largest score with the ground truth score: DISPLAYFORM1 Again, such a binary loss is not suitable for optimization.

Thus we introduce a surrogate loss.

Surrogate.

As pointed out in Lapin et al. (2015) , there is a natural extension of the previous multi-class case: DISPLAYFORM2 This loss creates a margin between the ground truth and the k-th largest score, irrespectively of the values of the (k − 1)-largest scores.

Note that we retrieve the formulation of Crammer & Singer (2001) for k = 1.Difficulty of the Optimization.

The surrogate loss l k of equation FORMULA5 suffers from two disadvantages that make it difficult to optimize: (i) it is not a smooth function of s -it is continuous but not differentiable -and (ii) its weak derivatives have at most two non-zero elements.

Indeed at most two elements of s are retained by the (·) [k] and max operators in equation (6).

All others are discarded and thus get zero derivatives.

When l k is coupled with a deep neural network, the model typically yields poor performance, even on the training set.

Similar difficulties to optimizing a piecewise linear loss have also been reported by BID1 in the context of multi-label classification.

We illustrate this in the next section.

We postulate that the difficulty of the optimization explains why there has been little work exploring the use of SVM losses in deep learning (even in the case k = 1), and that this work may help remedy it.

We propose a smoothing that alleviates both issues (i) and (ii), and we present experimental evidence that the smooth surrogate loss offers better performance in practice.

Reformulation.

We introduce the following notation: given a labelȳ ∈ Y, Y (k) y is the subset of tuples from Y (k) that includeȳ as one of their elements.

Forȳ ∈ Y (k) and y ∈ Y, we further define ∆ k (ȳ, y) 1(y / ∈ȳ).

Then, by adding and subtracting the k − 1 largest scores of s \y as well as s y , we obtain: DISPLAYFORM0 We give a more detailed proof of this in Appendix A.1.

Since the margin can be rescaled without loss of generality, we rewrite l k as: DISPLAYFORM1 Smoothing.

In the form of equation FORMULA7 , the loss function can be smoothed with a temperature parameter τ > 0: DISPLAYFORM2 Note that we have changed the notation to use L k,τ to refer to the smooth loss.

In what follows, we first outline the properties of L k,τ and its relationship with cross-entropy.

Then we show the empirical advantage of L k,τ over its non-smooth counter-part l k .Properties of the Smooth Loss.

The smooth loss L k,τ has a few interesting properties.

First, for any τ > 0, L k,τ is infinitely differentiable and has non-sparse gradients.

Second, under mild conditions, when τ → 0 + , the non-maximal terms become negligible, therefore the summations collapse to maximizations and L k,τ → l k in a pointwise sense (Proposition 2 in Appendix A.2).

Third, L k,τ is an upper bound on l k if and only if k = 1 (Proposition 3 in Appendix A.3), but L k,τ is, up to a scaling factor, an upper bound on Λ k (Proposition 4 in Appendix A.4).

This makes it a valid surrogate loss for the minimization of Λ k .Relationship with Cross-Entropy.

We have previously seen that the margin can be rescaled by a factor of α > 0.

In particular, if we scale ∆ by α → 0 + and choose a temperature τ = 1, it can be seen that L 1,1 becomes exactly the cross-entropy loss for classification.

In that sense, L k,τ is a generalization of the cross-entropy loss to: (i) different values of k ≥ 1, (ii) different values of temperature and (iii) higher margins with the scaling α of ∆. For simplicity purposes, we will keep α = 1 in this work.

Influence of the temperature τ on the learning of a DenseNet 40-12 on CIFAR-100.

We confirm that smoothing helps the training of a neural network in FIG1 , where a large enough value of τ greatly helps the performance on the training set.

In FIG1 , we observe that such high temperatures yield gradients that are not sparse.

In other words, with a high temperature, the gradient is informative about a greater number of labels, which helps the training of the model.

We remark that the network exhibits good accuracy when τ is high enough (0.01 or larger).

For τ too small, the model fails to converge to a good critical point.

When τ is positive but small, the function is smooth but the gradients are numerically sparse (see FIG1 ), which suggests that the smoothness property is not sufficient and that non-sparsity is a key factor here.

Experimental evidence suggests that it is beneficial to use L k,τ rather than l k to train a neural network.

However, at first glance, L k,τ may appear prohibitively expensive to compute.

Specifically, there are summations over DISPLAYFORM0 y , which have a cardinality of n k and n k−1 respectively.

For instance for ImageNet, we have k = 5 and n = 1, 000, which amounts to n k

12 terms to compute and sum over for each single sample, thereby making the approach practically infeasible.

This is in stark contrast with l k , for which the most expensive operation is to compute the k-th largest score of an array of size n, which can be done in O(n).

To overcome this computational challenge, we will now reframe the problem and reveal its exploitable structure.

For a vector e ∈ R n and i ∈ {1, .., n}, we define σ i (e) as the sum of all products of i distinct elements of e. Explicitly, σ i (e) can be written as σ i (e) = 1≤j1<...

<ji≤n e j1 ...e ji .

The terms σ i are known as the elementary symmetric polynomials.

We further define σ 0 (e) = 1 for convenience.

We now re-write L k,τ using the elementary symmetric polynomials, which appear naturally when separating the terms that contain the ground truth from the ones that do not: DISPLAYFORM0 DISPLAYFORM1 Note that the application of exp to vectors is meant in an element-wise fashion.

The last equality of equation FORMULA10 reveals that the challenge is to efficiently compute σ k−1 and σ k , and their derivatives for the optimization.

While there are existing algorithms to evaluate the elementary symmetric polynomials, they have been designed for computations on CPU with double floating point precision.

For the most recent work, see Jiang et al. (2016) .

To efficiently train deep neural networks with L k,τ , we need algorithms that are numerically stable with single floating point precision and that exploit GPU parallelization.

In the next sections, we design algorithms that meet these requirements.

The final performance is compared to the standard alternative algorithm in Appendix B.3.

We consider the general problem of efficiently computing (σ k−1 , σ k ).

Our goal is to compute σ k (e), where e ∈ R n and k n. Since this algorithm will be applied to e = exp(s \y /kτ ) (see equation FORMULA10 ), we can safely assume e i = 0 for all i ∈ 1, n .The main insight of our approach is the connection of σ i (e) to the polynomial: DISPLAYFORM0 Indeed, if we expand P to α 0 + α 1 X + ... + α n X n , Vieta's formula gives the relationship:

DISPLAYFORM1 Therefore, it suffices to compute the coefficients α n−k to obtain the value of σ k (e).

To compute the expansion of P , we can use a divide-and-conquer approach with polynomial multiplications when merging two branches of the recursion.

This method computes all (σ i ) 1≤i≤n instead of the only (σ i ) k−1≤i≤k that we require.

Since we do not need σ i (e) for i > k, we can avoid computations of all coefficients for a degree higher than n − k. However, typically k n.

For example, in ImageNet, we have k = 5 and n = 1, 000, therefore we have to compute coefficients up to a degree 995 instead of 1,000, which is a negligible improvement.

To turn k n to our advantage, we notice that σ i (e) = σ n (e)σ n−i (1/e).

Moreover, σ n (e) = n i=1 e i can be computed in O(n).

Therefore we introduce the polynomial: DISPLAYFORM2 Then if we expand Q to β 0 + β 1 X + ... + β n X n , we obtain with Vieta's formula again: DISPLAYFORM3 Subsequently, in order to compute σ k (e), we only require the k first coefficients of Q, which is very efficient when k is small in comparison with n. This results in a time complexity of O(kn) (Proposition 5 in Appendix B.1).

Moreover, there are only O(log(n)) levels of recursion, and since every level can have its operations parallelized, the resulting algorithm scales very well with n when implemented on a GPU (see Appendix B.3.2 for practical runtimes).The algorithm is described in Algorithm 1: step 2 initializes the polynomials for the divide and conquer method.

While the polynomial has not been fully expanded, steps 5-6 merge branches by performing the polynomial multiplications (which can be done in parallel).Step 10 adjusts the coefficients using equation FORMULA3 .

We point out that we could obtain an algorithm with a time complexity of O(n log(k) 2 ) if we were using Fast Fourier Transform for polynomial multiplications in steps 5-6.

Since we are interested in the case where k is small (typically 5), such an improvement is negligible.

Require: e ∈ (R * DISPLAYFORM0 Initialize n polynomials to X + 1 ei (encoded by coefficients) 3: p ← n Number of polynomials 4: while p > 1 do Merge branches with polynomial multiplications 5: DISPLAYFORM1 Polynomial multiplication up to degree k ... DISPLAYFORM2 Polynomial multiplication up to degree k 7: DISPLAYFORM3 Update number of polynomials 9: end while 10: DISPLAYFORM4 Obtaining numerical stability in single floating point precision requires special attention: the use of exponentials with an arbitrarily small temperature parameter is fundamentally unstable.

In Appendix B.2.1, we describe how operating in the log-space and using the log-sum-exp trick alleviates this issue.

The stability of the resulting algorithm is empirically verified in Appendix B.3.3.

A side effect of using Algorithm 1 is that a large number of buffers are allocated for automatic differentiation: for each addition in log-space, we apply log and exp operations, each of which needs to store values for the backward pass.

This results in a significant amount of time spent on memory allocations, which become the time bottleneck.

To avoid this, we exploit the structure of the problem and design a backward algorithm that relies on the results of the forward pass.

By avoiding the memory allocations and considerably reducing the number of operations, the backward pass is then sped up by one to two orders of magnitude and becomes negligible in comparison to the forward pass.

We describe our efficient backward pass in more details below.

First, we introduce the notation for derivatives: DISPLAYFORM0 We now observe that: DISPLAYFORM1 In other words, equation FORMULA5 states that δ j,i , the derivative of σ j (e) with respect to e i , is the sum of product of all (j − 1)-tuples that do not include e i .

One way of obtaining σ j−1 (e \i ) is to compute a forward pass for e \i , which we would need to do for every i ∈ 1, n .

To avoid such expensive computations, we remark that σ j (e) can be split into two terms: the ones that contain e i (which can expressed as e i σ j−1 (e \i )) and the ones that do not (which are equal to σ j (e \i ) by definition).

This gives the following relationship: DISPLAYFORM2 Simplifying equation FORMULA23 using equation FORMULA5 , we obtain the following recursive relationship: DISPLAYFORM3 Since the (σ j (e)) 1≤i≤k have been computed during the forward pass, we can initialize the induction with δ 1,i = 1 and iteratively compute the derivatives δ j,i for j ≥ 2 with equation FORMULA7 .

This is summarized in Algorithm 2.

Require: e, (σ j (e)) 1≤j≤k , k ∈ N * (σ j (e)) 1≤j≤k have been computed in the forward pass 1: δ 1,i = 1 for i ∈ 1, n 2: for j ∈ 1, k do 3:δ j,i = σ j−1 (e) − e i δ j−1,i for i ∈ 1, n 4: end for Algorithm 2 is subject to numerical instabilities (Observation 1 in Appendix B.2.2).

In order to avoid these, one solution is to use equation (16) for each unstable element, which requires numerous forward passes.

To avoid this inefficiency, we provide a novel approximation in Appendix B.2.2: the computation can be stabilized by an approximation with significantly smaller overhead.

Theoretical results suggest that Cross-Entropy (CE) is an optimal classifier in the limit of infinite data, by accurately approximating the data distribution.

In practice, the presence of label noise makes the data distribution more complex to estimate when only a finite number of samples is available.

For these reasons, we explore the behavior of CE and L k,τ when varying the amount of label noise and the training data size.

For the former, we introduce label noise in the CIFAR-100 data set (Krizhevsky, 2009) in a manner that would not perturb the top-5 error of a perfect classifier.

For the latter, we vary the training data size on subsets of the ImageNet data set (Russakovsky et al., 2015) .In all the following experiments, the temperature parameter is fixed throughout training.

This choice is discussed in Appendix D.1.

The algorithms are implemented in Pytorch (Paszke et al., 2017) and are publicly available at https://github.com/oval-group/smooth-topk.

Experiments on CIFAR-100 and ImageNet are performed on respectively one and two Nvidia Titan Xp cards.

Data set.

In this experiment, we investigate the impact of label noise on CE and L 5,1 .

The CIFAR-100 data set contains 60,000 RGB images, with 50,000 samples for training-validation and 10,000 for testing.

There are 20 "coarse" classes, each consisting of 5 "fine" labels.

For example, the coarse class "people" is made up of the five fine labels "baby", "boy", "girl", "man" and "woman".

In this set of experiments, the images are centered and normalized channel-wise before they are fed to the network.

We use the standard data augmentation technique with random horizontal flips and random crops of size 32 × 32 on the images padded with 4 pixels on each side.

We introduce noise in the labels as follows: with probability p, each fine label is replaced by a fine label from the same coarse class.

This new label is chosen at random and may be identical to the original label.

Note that all instances generated by data augmentation from a single image are assigned the same label.

The case p = 0 corresponds to the original data set without noise, and p = 1 to the case where the label is completely random (within the fine labels of the coarse class).

With this method, a perfect top-5 classifier would still be able to achieve 100 % accuracy by systematically predicting the five fine labels of the unperturbed coarse label.

Methods.

To evaluate our loss functions, we use the architecture DenseNet 40-40 from Huang et al. FORMULA1 , and we use the same hyper-parameters and learning rate schedule as in Huang et al. (2017) .

The temperature parameter is fixed to one.

When the level of noise becomes non-negligible, we empirically find that CE suffers from over-fitting and significantly benefits from early stoppingwhich our loss does not need.

Therefore we help the baseline and hold out a validation set of 5,000 images, on which we monitor the accuracy across epochs.

Then we use the model with the best top-5 validation accuracy and report its performance on the test set.

Results are averaged over three runs with different random seeds.

Results.

As seen in TAB0 , L 5,1 outperforms CE on the top-5 testing accuracy when the labels are noisy, with an improvement of over 5% in the case p = 1.

When there is no noise in the labels, CE provides better top-1 performance, as expected.

It also obtains a better top-5 accuracy, although by a very small margin.

Interestingly, L 5,1 outperforms CE on the top-1 error when there is noise, although L 5,1 is not a surrogate for the top-1 error.

For example when p = 0.8, L 5,1 still yields an accuracy of 55.85%, as compared to 35.53% for CE.

This suggests that when the provided label is only informative about top-5 predictions (because of noise or ambiguity), it is preferable to use L 5,1 .

Data set.

As shown in Figure 1 , the ImageNet data set presents different forms of ambiguity and noise in the labels.

It also has a large number of training samples, which allows us to explore different regimes up to the large-scale setting.

Out of the 1.28 million training samples, we use subsets of various sizes and always hold out a balanced validation set of 50,000 images.

We then report results on the 50,000 images of the official validation set, which we use as our test set.

Images are resized so that their smaller dimension is 256, and they are centered and normalized channel-wise.

At training time, we take random crops of 224 × 224 and randomly flip the images horizontally.

At test time, we use the standard ten-crop procedure (Krizhevsky et al., 2012) .We report results for the following subset sizes of the data: 64k images (5%), 128k images (10%), 320k images (25%), 640k images (50%) and finally the whole data set (1.28M − 50k = 1.23M images for training).

Each strict subset has all 1,000 classes and a balanced number of images per class.

The largest subset has the same slight unbalance as the full ImageNet data set.

Methods.

In all the following experiments, we train a ResNet-18 (He et al., 2016) , adapting the protocol of the ImageNet experiment in Huang et al. (2017) .

In more details, we optimize the model with Stochastic Gradient Descent with a batch-size of 256, for a total of 120 epochs.

We use a Nesterov momentum of 0.9.

The temperature is set to 0.1 for the SVM loss (we discuss the choice of the temperature parameter in Appendix D.1).

The learning rate is divided by ten at epochs 30, 60 and 90, and is set to an initial value of 0.1 for CE and 1 for L 5,0.1 .

The quadratic regularization hyper-parameter is set to 0.0001 for CE.

For L 5,0.1 , it is set to 0.000025 to preserve a similar relative weighting of the loss and the regularizer.

For both methods, training on the whole data set takes about a day and a half (it is only 10% longer with L 5,0.1 than with CE).

As in the previous experiments, the validation top-5 accuracy is monitored at every epoch, and we use the model with best top-5 validation accuracy to report its test error.

Probabilities for Multiple Crops.

Using multiple crops requires a probability distribution over labels for each crop.

Then this probability is averaged over the crops to compute the final prediction.

The standard method is to use a softmax activation over the scores.

We believe that such an approach is only grounded to make top-1 predictions.

The probability of a labelȳ being part of the top-5 prediction should be marginalized over all combinations of 5 labels that includeȳ as one of their elements.

This can be directly computed with our algorithms to evaluate σ k and its derivative.

We refer the reader to Appendix C for details.

All the reported results of top-5 error with multiple crops are computed with this method.

This provides a systematic boost of at least 0.2% for all loss functions.

In fact, it is more beneficial to the CE baseline, by up to 1% in the small data setting.

Results.

The results of TAB1 confirm that L 5,0.1 offers better top-5 error than CE when the amount of training data is restricted.

As the data set size increases, the difference of performance becomes very small, and CE outperforms L 5,0.1 by an insignificant amount in the full data setting.

This work has introduced a new family of loss functions for the direct minimization of the top-k error (that is, without the need for fine-tuning).

We have empirically shown that non-sparsity is essential for loss functions to work well with deep neural networks.

Thanks to a connection to polynomial algebra and a novel approximation, we have presented efficient algorithms to compute the smooth loss and its gradient.

The experimental results have demonstrated that our smooth top-5 loss function is more robust to noise and overfitting than cross-entropy when the amount of training data is limited.

We have argued that smoothing the surrogate loss function helps the training of deep neural networks.

This insight is not specific to top-k classification, and we hope that it will help the design of other surrogate loss functions.

In particular, structured prediction problems could benefit from smoothed SVM losses.

In this section, we fix n the number of classes.

We let τ > 0 and k ∈ {1, ..., n − 1}. All following results are derived with a loss l k defined as in equation FORMULA7 : DISPLAYFORM0 A.1 REFORMULATION Proposition 1.

We can equivalently re-write l k as: DISPLAYFORM1 Proof.

DISPLAYFORM2 A.2 POINT-WISE CONVERGENCE Lemma 1.

Let n ≥ 2 and e ∈ R n .

Assume that the largest element of e is greater than its second largest element: DISPLAYFORM3 exp(e i /τ ) = e [1] .Proof.

For simplicity of notation, and without loss of generality, we suppose that the elements of e are sorted in descending order.

Then for i ∈ {2, ..n}, we have e i − e 1 ≤ e 2 − e 1 < 0 by assumption, and thus ∀ i ∈ {2, ..n}, lim DISPLAYFORM4 exp((e i − e 1 )/τ ) = 0.

Therefore: DISPLAYFORM5 exp((e i − e 1 )/τ ) = 1.And thus: DISPLAYFORM6 exp((e i − e 1 )/τ ) = 0.The result follows by noting that: DISPLAYFORM7 exp(e i /τ ) = e 1 + τ log DISPLAYFORM8 exp((e i − e 1 )/τ ) .Proposition 2.

Assume that DISPLAYFORM9 , one can see that max DISPLAYFORM10 Since L k,τ can be written as: DISPLAYFORM11 the result follows by two applications of Lemma 1.

Proposition 3.

L k,τ is an upper bound on l k if and only if k = 1.Proof.

Suppose k = 1.

Let s ∈ R n and y ∈ Y. We introduce y * = argmax y∈Y {∆ 1 (ȳ, y) + sȳ}. Then we have: DISPLAYFORM0 Now suppose k ≥ 2.

We construct an example (s, y) such that L k,τ (s, y) < l k (s, y).

For simplicity, we set y = 1.

Then let s 1 = α, s i = β for i ∈ {2, ..., k + 1} and s i = −∞ for i ∈ {k + 2, ..., n}.The variables α and β are our degrees of freedom to construct the example.

Assuming infinite values simplifies the analysis, and by continuity of L k,τ and l k , the proof will hold for real values sufficiently small.

We further assume that 1 + 1 k (β − α) > 0.

Then can write l k (s, y) as: DISPLAYFORM1 Exploiting the fact that exp(s i /τ ) = 0 for i ≥ k + 2, we have: DISPLAYFORM2 And: DISPLAYFORM3 This allows us to write L k,τ as: DISPLAYFORM4 We introduce x = 1 + 1 k (β − α).

Then we have: DISPLAYFORM5 And: DISPLAYFORM6 For any value x > 0, we can find (α, β) ∈ R 2 such that x = 1 + 1 k (β − α) and that all our hypotheses are verified.

Consequently, we only have to prove that there exists x > 0 such that: DISPLAYFORM7 We show that lim x→∞ ∆(x) < 0, which will conclude the proof by continuity of ∆. DISPLAYFORM8 A.4 BOUND ON PREDICTION LOSS DISPLAYFORM9 Proof.

DISPLAYFORM10 This is a monotonically increasing function of p ≤ q − 1, therefore it is upper bounded by its maximal value at p = q − 1: DISPLAYFORM11 Lemma 3.

Assume that y / ∈ P k (s).

Then we have: DISPLAYFORM12 Proof.

Let j ∈ 0, k − 1 .

We introduce the random variable U j , whose probability distribution is uniform over the set U j {ȳ ∈ Y (k) y :ȳ ∩ P k (s) = j}.

Then V j is the random variable such that V j |U j replaces y from U j with a value drawn uniformly from P k (s).

We denote by V j the set of values taken by V j with non-zero probability.

Since V j replaces the ground truth score by one of the values of P k (s), it can be seen that: DISPLAYFORM13 Furthermore, we introduce the scoring function DISPLAYFORM14 is the set of the k largest scores and y / ∈ P k (s), we have that: DISPLAYFORM15 with probability 1.Therefore we also have that: DISPLAYFORM16 with probability 1.This finally gives us: DISPLAYFORM17 Making the (uniform) probabilities explicit, we obtain: DISPLAYFORM18 To derive the set cardinalities, we rewrite U j and V j as: DISPLAYFORM19 Therefore we have that: DISPLAYFORM20 And: DISPLAYFORM21 Therefore: DISPLAYFORM22 Combining with equation FORMULA1 , we obtain: DISPLAYFORM23 We sum over j ∈ 0, k − 1 , which yields: DISPLAYFORM24 Finally, we note that {U j } 0≤j≤k−1 and {V j } 0≤j≤k−1 are respective partitions of Y (k) DISPLAYFORM25 y , which gives us the final result: DISPLAYFORM26 Proposition 4.

L k,τ is, up to a scaling factor, an upper bound on the prediction loss Λ k : DISPLAYFORM27 Proof.

Suppose that Λ k (s, y) = 0.

Then the inequality is trivial because L k,τ (s, y) ≥ 0.

We now assume that Λ k (s, y) = 1.

Then there exist at least k higher scores than s y .

To simplify indexing, we introduce Z DISPLAYFORM28 y and T k the set of k labels corresponding to the k-largest scores.

By assumption, y / ∈ T k since y is misclassified.

We then write: DISPLAYFORM29 Thanks to Lemma 3, we have: DISPLAYFORM30 Injecting this back into (51): DISPLAYFORM31 And back to the original loss: DISPLAYFORM32 B ALGORITHMS: PROPERTIES & PERFORMANCE B.1 TIME COMPLEXITY Lemma 4.

Let P and Q be two polynomials of degree p and q. The time complexity of obtaining the first r coefficients of P Q is O(min{r, p} min{r, q}).Proof.

The multiplication of two polynomials can be written as the convolution of their coefficients, which can be truncated at degree r for each polynomial.

Proof.

Let N = log 2 (n), or equivalently n = 2 N .

With the divide-and-conquer algorithm, the complexity of computing the k first coefficients of P can be written as: DISPLAYFORM0 Indeed we decompose P = Q 1 Q 2 , with each Q i of degree n/2, and for these we compute their k first coefficients in T ( n 2 ).

Then given the k first coefficients of Q 1 and Q 2 , the k first coefficients of P are computed in O(min{k, n} 2 ) by Lemma 4.

Then we can write: DISPLAYFORM1 .. DISPLAYFORM2 By summing these terms, we obtain T (k, n) = 2 N T (k, 1) + DISPLAYFORM3 .

In loose notation, we have k DISPLAYFORM4 .

Then we can write: DISPLAYFORM5 Thus finally: DISPLAYFORM6 B.2 NUMERICAL STABILITY

In order to ensure numerical stability of the computation, we maintain all computations in the log space: for a multiplication exp(x 1 ) exp(x 2 ), we actually compute and store x 1 + x 2 ; for an addition exp(x 1 ) + exp(x 2 ) we use the "log-sum-exp" trick: we compute m = max{x 1 , x 2 }, and store m + log(exp(x 1 − m) + exp(x 2 − m)), which guarantees stability of the result.

These two operations suffice to describe the forward pass.

Observation 1.

The backward recursion of Algorithm 2 is unstable when e i 1 and e i max p =i {e p }.

To see that, assume that when we compute ( n p=1 e p ) − e i , we make a numerical error in the order of (e.g 10 −5 for single floating point precision).

With the numerical errors, we obtain approximateδ as follows: DISPLAYFORM0 Since e i 1, we quickly obtain unstable results.

Definition 1.

For p ∈ {0, ..., n − k}, we define the p-th order approximation to the gradient as: DISPLAYFORM1 Proposition 6.

If we approximate the gradient by its p-th order approximation as defined in equation FORMULA5 , the absolute error is: DISPLAYFORM2 Proof.

We remind equation FORMULA7 , which gives a recursive relationship for the gradients: DISPLAYFORM3 This can be re-written as: DISPLAYFORM4 We write σ k+p (e \i ) = δ k+p+1,i , and the result follows by repeated applications of equation FORMULA1 for j ∈ {k + 1, k + 2, ..., k + p + 1}.Intuition.

We have seen in Observation 1 that the recursion tends to be unstable for δ j,i when e i is among the largest elements.

When that is the case, the ratio DISPLAYFORM5 decreases quickly with p.

This has two consequences: (i) the sum of equation FORMULA5 is stable to compute because the summands have different orders of magnitude and (ii) the error becomes small.

Unfortunately, it is difficult to upper-bound the error of equation FORMULA5 by a quantity that is both measurable at runtime (without expensive computations) and small enough to be informative.

Therefore the approximation error is not controlled at runtime.

In practice, we detect the instability of δ k,i : numerical issues arise if subtracted terms have a very small relative difference.

For those unstable elements we use the p-th order approximation (to choose the value of p, a good rule of thumb is p 0.2k).

We have empirically found out that this heuristic works well in practice.

Note that this changes the complexity of the forward pass to O((k + p)n) since we need p additional coefficients during the backward.

If p 0.2k, this increases the running time of the forward pass by 20%, which is a moderate impact.

The Summation Algorithm (SA) is an alternative to the Divide-and-Conquer (DC) algorithm for the evaluation of the elementary symmetric polynomials.

It is described for instance in (Jiang et al., 2016) .

The algorithm can be summarized as follows:Implementation.

Note that the inner loop can be parallelized, but the outer one is essentially sequential.

In our implementation for speed comparisons, the inner loop is parallelized and a buffer is pre-allocated for the σ j,i .

Require: e ∈ R n , k ∈ N * 1: σ 0,i ← 1 for 1 ≤ i ≤ n σ j,i = σ j (e 1 , . . .

, e i ) 2: σ j,i ← 0 for i < j Do not define values for i < j (meaningless) 3: σ 1,1 ← e 1 Initialize recursion 4: for i ∈ 2, n do 5: DISPLAYFORM0 σ j,i ← σ j,i−1 + e i σ j−1,i−1 9: end for 10: end for 11: return σ k,n

We compare the execution time of the DC and SA algorithms on a GPU (Nvidia Titan Xp).

We use the following parameters: k = 5, a batch size of 256 and a varying value of n. The following timings are given in seconds, and are computed as the average of 50 runs.

In Table 3 , we compare the speed of Summation and DC for the evaluation of the forward pass.

In TAB3 , we compare the speed of the evaluation of the backward pass using Automatic Differentiation (AD) and our Custom Algorithm (CA) (see Algorithm 2).

Table 3 : Execution time (s) of the forward pass.

The Divide and Conquer (DC) algorithm offers nearly logarithmic scaling with n in practice, thanks to its parallelization.

In contrast, the runtime of the Summation Algorithm (SA) scales linearly with n. n 100 1,000 10,000 100,000 SA 0.006 0.062 0.627 6.258 DC 0.011 0.018 0.024 0.146We remind that both algorithms have a time complexity of O(kn).

SA provides little parallelization (the parallelizable inner loop is small for k n), which is reflected in the runtimes.

On the other hand, DC is a recursive algorithm with O(log(n)) levels of recursion, and all operations are parallelized at each level of the recursion.

This allows DC to have near-logarithmic effective scaling with n, at least in the range {100 − 10, 000}. These runtimes demonstrate the advantage of using Algorithm 2 instead of automatic differentiation.

In particular, we see that in the use case of ImageNet (n = 1, 000), the backward computation changes from being 8x slower than the forward pass to being 3x faster.

We now investigate the numerical stability of the algorithms.

Here we only analyze the numerical stability, and not the precision of the algorithm.

We point out that compensation algorithms are useful to improve the precision of SA but not its stability.

Therefore they are not considered in this discussion.

Jiang et al. (2016) mention that SA is a stable algorithm, under the assumption that no overflow or underflow is encountered.

However this assumption is not verified in our use case, as we demonstrate below.

We consider that the algorithm is stable if no overflow occurs in the algorithm (underflows are not an issue for our use cases).

We stress out that numerical stability is critical for our machine learning context: if an overflow occurs, the weights of the learning model inevitably diverge to infinite values.

To test numerical stability in a representative setting of our use cases, we take a random mini-batch of 128 images from the ImageNet data set and forward it through a pre-trained ResNet-18 to obtain a vector of scores per sample.

Then we use the scores as an input to the SA and DC algorithms, for various values of the temperature parameter τ .

We compare the algorithms with single (S) and double (D) floating point precision.

By operating in the log-space, DC is significantly more stable than SA.

In this experimental setting, DC log is stable in single floating point precision until τ = 10 −36 .

We consider the probability of label i being part of the final top-k prediction.

To that end, we marginalize over all k-tuples that contain i as one of their element.

Then the probability of selecting label i for the top-k prediction can be written as: DISPLAYFORM0 Proposition 7.

The unnormalized probability can be computed as: DISPLAYFORM1 Proof.

DISPLAYFORM2 Finally we can rescale the unnormalized probability p by σ k (exp(s)) since the latter quantity is independent of i. We obtain: DISPLAYFORM3 NB.

We prefer to use d log σ i (exp(s))

ds i rather than dσ i (exp(s)) ds i for stability reasons.

Once the unnormalized probabilities are computed, they can be normalized by simply dividing by their sum.

In this section, we discuss the choice of the temperature parameter.

Note that such insights are not necessarily confined to a top-k minimization: we believe that these ideas generalize to any loss that is smoothed with a temperature parameter.

When the temperature τ has a low value, propositions 3 and 4 suggest that L k,τ is a sound learning objective.

However, as shown in FIG1 , optimization is difficult and can fail in practice.

Conversely, optimization with a high value of the temperature is easy, but uninformative about the learning: then L k,τ is not representative of the task loss we wish to learn.

In other words, there is a trade-off between the ease of the optimization and the quality of the surrogate loss in terms of learning.

Therefore, it makes sense to use a low temperature that still permits satisfactory optimization.

In FIG1 , we have provided the plots of the training objective to illustrate the speed of convergence.

In TAB5 , we give the training and validation accuracies to show the influence of the temperature: The choice of temperature parameter can affect the scale of the loss function.

In order to preserve a sensible trade-off between regularizer and loss, it is important to adjust the regularization hyperparameter(s) accordingly (the value of the quadratic regularization for instance).

Similarly, the energy landscape may vary significantly for a different value of the temperature, and the learning rate may need to be adapted too.

Continuation methods usually rely on an annealing scheme to gradually improve the quality of the approximation.

For this work, we have found that such an approach required heavy engineering and did not provide substantial improvement in our experiments.

Indeed, we have mentioned that other hyper-parameters depend on the temperature, thus these need to be adapted dynamically too.

This requires sensitive heuristics.

Furthermore, we empirically find that setting the temperature to an appropriate fixed value yields the same performance as careful fine-tuning of a pre-trained network with temperature annealing.

We summarize the methodology that reflects the previous insights and that we have found to work well during our experimental investigation.

First, the temperature hyper-parameter is set to a low fixed value that allows for the model to learn on the training data set.

Then other hyper-parameters, such as quadratic regularization and learning rate are adapted as usual by cross-validation on the validation set.

We believe that the optimal value of the temperature is mostly independent of the architecture of the neural network, but is greatly influenced by the values of k and n (see how these impact the number of summands involved in L k,τ , and therefore its scale).small.

If λ = 0, βλ = 0 for any β > 0 and we can choose β as large as needed to make α β arbitrarily small.

Note that we do not need any hypothesis on the norm · , the result makes only use of the positive homogeneity property.

Consequence On Deep Networks.

Proposition 8 shows that for a deep network trained with l k , one can fix the value of α to 1, and treat the quadratic regularization of the last fully connected layer as an independent hyper-parameter.

By doing this rather than tuning α, the loss keeps the same scale which may make it easier to find an appropriate learning rate.

When using the smooth loss, there is no direct equivalent to Proposition 8 because the log-sumexp function is not positively homogeneous.

However one can consider that with a low enough temperature, the above insight can still be used in practice.

In this section, we provide experiments to qualitatively assess the importance of the margin by running experiments with a margin of either 0 or 1.

The following results are obtained on our validation set, and do not make use of multiple crops.

Top-1 Error.

As we have mentioned before, the case (k, τ, α) = (1, 1, 0) corresponds exactly to Cross-Entropy.

We compare this case against the same loss with a margin of 1: (k, τ, α) = (1, 1, 1).

We obtain the following results:Margin Top-1 Accuracy (%) 0 71.03 1 71.15 Table 7 : Influence of the margin parameter on top-1 performance.

Top-5 Error.

We now compare (k, τ, α) = (5, 0.1, 0) and (k, τ, α) = (5, 0.1, 1):Margin Top-5 Accuracy (%) 0 89.12 1 89.45 Table 8 : Influence of the margin parameter on top-5 performance.

In the main paper, we report the average of the scores on CIFAR-100 for clarity purposes.

Here, we also detail the standard deviation of the scores for completeness.

<|TLDR|>

@highlight

Smooth Loss Function for Top-k Error Minimization

@highlight

Proposes using top-k loss with deep models to address the problem of class confusion with similar classes both present or absent of the training dataset.

@highlight

Smoothes the top-k losses.

@highlight

This paper introduces a smooth surrogate loss function for the top-k SVM, for the purpose of plugging the SVM to the deep neural networks.