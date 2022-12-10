Memorization in over-parameterized neural networks can severely hurt generalization in the presence of mislabeled examples.

However, mislabeled examples are to hard avoid in extremely large datasets.

We address this problem using the implicit regularization effect of stochastic gradient descent with large learning rates, which we find to be able to separate clean and mislabeled examples with remarkable success using loss statistics.

We leverage this to identify and on-the-fly discard mislabeled examples using a threshold on their losses.

This leads to On-the-fly Data Denoising (ODD), a simple yet effective algorithm that is robust to mislabeled examples, while introducing almost zero computational overhead.

Empirical results demonstrate the effectiveness of ODD on several datasets containing artificial and real-world mislabeled examples.

Over-parametrized deep neural networks have remarkable generalization properties while achieving near-zero training error (Zhang et al., 2016) .

However, the ability to fit the entire training set is highly undesirable, as a small portion of mislabeled examples in the dataset could severely hurt generalization (Zhang et al., 2016; BID0 .

Meanwhile, an exponential growth in training data size is required to linearly improve generalization in vision tasks BID31 ; this progress could be hindered if there are mislabeled examples within the dataset.

Mislabeled examples are to be expected in large datasets that contain millions of examples.

Webbased supervision produces noisy labels BID17 BID21 ; whereas human labeled datasets sacrifice accuracy for scalability BID16 .

Therefore, algorithms that are robust to various levels of mislabeled examples are warranted in order to further improve generalization for very large labeled datasets.

In this paper, we propose On-the-fly Data Denoising (ODD), a simple and robust method for training with noisy examples based on the implicit regularization effect of stochastic gradient descent.

First, we train residual networks with large learning rate schedules and use the resulting losses to separate clean examples from mislabeled ones.

This is done by identifying examples whose losses exceed a certain threshold.

Reasonable thresholds can be derived from the loss distribution for uniform label noise which does not depend on the amount of mislabeled examples in the dataset.

Finally, we remove these examples from the dataset and continue training until convergence.

Empirically, ODD performs significantly better than previous methods in datasets containing artificial noise (Sections 4.1 and 4.2) or real-world mislabeled examples (Section 4.3), while achieving equal or better accuracy than the state-of-the-art on clean datasets (Sections 4.1 and 4.2).

We further conduct ablation studies to demonstrate that ODD is robust w.r.t hyperparameters and artificial noise levels (Section 4.4).

Our method is also able to detect mislabeled examples in the CIFAR-100 dataset without any additional supervision ( FIG0 ).

The goal of supervised learning is to find a function f ∈ F that describes the probability of a random label vector Y ∈ Y given a random input vector X ∈ X , which has underlying joint distribution P (X, Y ).

Given a loss function (y,ŷ), one could minimize the average of over P :R(f ) = (y, f (x)) dP (x, y)The joint distribution P (X, Y ) is usually unknown, but we could gain access to its samples via a potentially noisy labeling process, such as crowdsourcing BID16 or web queries BID17 .

We denote the training dataset with N examples as DISPLAYFORM0 G represents correctly labeled (clean) examples sampled from P (X, Y ).

B represents mislabeled examples that are not sampled from P (X, Y ), but from another distribution Q(X, Y ); G ∩ B = ∅. We aim to learn the function f from D without knowledge about B, G or their statistics (e.g. |B|).A typical approach is to assume that B = ∅, i.e. all examples are i.i.d.

from P (X, Y ), and minimizing the following objective:R DISPLAYFORM1 However, this is not true if B = ∅ since D is no longer an unbiased population of P (X, Y ).

Moreover, when F is the space of large neural networks with parameters exceeding N , f could fit the entire training dataset (Zhang et al., 2016) , including the mislabeled examples.

This results in undesired behavior of f on inputs of the mislabeled set, let alone outside the training data.

To illustrate the harm of mislabeled examples to generalization, we consider training on CIFAR-10 where some examples are mislabeled uniformly at random.

Compared with training on D, training only on G could decrease validation error from 11.53 to 4.25 if there are 20% mislabeled examples, and from 15.57 to 5.06 if there are 40% mislabeled examples 1 .

Therefore, if we are able to identify examples that belong to G, we could vastly improve generalization on P (X, Y ).

Fortunately, in the case of classification with deep residual networks BID7 , the implicit generalization of stochastic gradient descent (SGD) with large learning rates (e.g. 0.1) can separate examples from G and examples from B via the loss statistics.

We demonstrate this in FIG1 , where we train deep residual networks on CIFAR-100 and ImageNet with different percentages of uniform label noise.

In early stages of training, the loss distributions of clean examples and mislabeled ones have notable statistical distance.

The network starts to fit mislabeled examples when learning rate starts to decrease, which is also crucial for achieving better generalization on clean datasets.

The working of the implicit regularization of gradient descent is by and large an open question that attracts much recent attentions BID23 BID18 BID4 .

Empirically, it has been observed that large learning rates are beneficial for generalization BID15 .

Recent work has shown that the stationary distribution of SGD iterates corresponds to an Ornstein-Uhlenbeck process BID33 with noise proportional to the learning rate BID22 .

Training with large learning rates would then encourage solutions that are more robust to large random perturbations in the parameter space and less likely to overfit to mislabeled examples.

Therefore, given these empirical and theoretical evidence on large learning rate helps generalization, we propose to classify correct and mislabeled examples through the loss statistics, and achieve better generalization by removing the examples that are potentially mislabeled and training on clean examples only.

To improve generalization in practice, one critical problem is to select a reasonable threshold for classification.

High thresholds could include too many examples from B, whereas low thresholds could prune too many examples from G; reasonable thresholds should also adapt to different unknown ratios of mislabeled examples.

Let us first consider the case where Q(Y |X) has the highest entropy, which is the uniform distribution over labels.

From FIG1 , the loss distribution for B is relatively stable with different ratios of |B|/|D|; examples in B are making little progress when learning rate is large.

We propose to characterize the (negative log-likelihood) loss distribution of uniform label noise p n (l) via the following generative procedure: DISPLAYFORM0 where fc(·) is the final (fully connected) layer of the network, relu(x) = max(x, 0) is the Rectified Linear Unit, and k represents a random label from K classes.

This represents the case where the model's prediction is uncorrelated with the labels.

The actual noise distribution could skew to the left if the model overfits to the noise, and skew to the right if the model predicts a label different from the noisy one.

We find that an identity covariance matrix forx is able to explain the noise distribution; this could result from well-conditioned objectives defined via deep residual networks BID7 and careful initialization BID8 ).

We qualitatively demonstrate the validity of our characterization on CIFAR-100 and ImageNet datasets in FIG1 .Therefore, we could define a threshold via the p-th percentile of p n (l); it relates to approximately how much examples in B we would retain if Q(Y |X) is uniform.

In Section 4.4, we show that this method is able to identify different percentages of uniform label noise with high precision.

We can utilize this implicit regularization effect to remove examples that might harm generalization, leading to On-the-fly Data Denoising (ODD), a simple algorithm robust to mislabeled examples:1.

Train all examples with large learning rates for E epochs.2.

Compute the p-th percentile of the distribution in Eq.(1), denoted as T p .3.

Remove examples whose average loss of the past h epochs exceeds T from the dataset.4.

Continue training the remaining examples from epoch E + 1.ODD introduces three hyperparameters: E determines the amount of training that separates clean examples from noisy ones; p determines T p that specifies the trade-off between less noisy examples and more clean examples; h determines the window of averaged loss statistics to reduce variance from data augmentation.

We do not explicitly estimate the portion of noise in the dataset, nor do we assume any specific noise model.

In fact, the threshold T p could be used to accurately predict the portion of uniform noise in the dataset, and works quite well even on other types of label noise; we will demonstrate this in Section 4.

Moreover, ODD is compatible with existing practices for learning rate schedules, such as stepwise BID7 or cosine BID20 .

Implicit Regularization of SGD The generalization of neural networks trained with SGD depend heavily on learning rate schedules BID20 .

It has been proposed that wide local minima 2 could result in better generalization BID10 BID2 BID14 .

Several factors could contribute to wider local optima and better generalization, such as smaller minibatch sizes BID14 , reasonable learning rates BID15 , and longer training time BID11 .

Moreover, solutions that are further away from the initialization may lead to wider local minima and better generalization BID11 .

In the presence of mislabeled examples, changes in optimization landscape BID0 could result in bad local minima (Zhang et al., 2016) , although it is argued that larger batch sizes could mitigate this effect BID27 .Training with Mislabeled Examples One paradigm to robust training with noisy labels involves estimating the noise distribution BID19 or confusion matrix BID30 .

Another line of methods propose to identify and clean the noisy examples through predictions of auxillary networks or via binary predictions BID24 ; the noisy labels are either pruned BID1 or replaced with model predictions BID25 .

Our method is comparable to these approaches, but the key difference is that we leverage the implicit regularization effect of SGD to identify noisy examples.

Other approaches propose to reweigh the examples via a pretrained network BID13 , meta learning BID26 , or surrogate loss functions BID5 Zhang & Sabuncu, 2018) .

Some methods require a set of trusted examples BID35 BID9 .ODD has several appealing properties compared to existing methods.

First, the thresholds for classifying mislabeled examples from ODD do not rely on estimations of the noise confusion matrix.

Next, ODD does not require additional trusted examples.

Finally, ODD removes potentially noisy examples on-the-fly; it has little computational overhead compared to standard SGD training.

We evaluate our method on clean and noisy versions of CIFAR-10, CIFAR-100, ImageNet BID28 and WebVision BID17 datasets.

We use stochastic gradient descent with momentum for training while following standard image preprocessing and data augmentation practices.

We do not consider dropout BID29 or model ensembles BID12 in our experiments.

We use h = 2 for all our ODD experiments; we observe that having h ∈ [2, 5] yields similar results.

We first evaluate our method on the CIFAR-10 and CIFAR-100 datasets, which contain 50,000 training images and 10,000 validation images of size 32 × 32 with 10 and 100 labels respectively.

During training, we follow the data augmentations in (Zagoruyko & Komodakis, 2016) , which performs horizontal flips, takes random crops from 40 × 40 images padded by 4 pixels on each side, and fills missing pixels with reflections of the original images.

In our experiments, we train the wide residual network architecture (WRN-28-10) in (Zagoruyko & Komodakis, 2016) for 200 epochs with a minibatch size of 128, momentum 0.9 and weight decay 5×10 −4 .

We consider a cosine annealing schedule as described in BID20 with DISPLAYFORM0 −5 (no warm restarts), as we observe this schedule outperforms the traditional stepwise schedules on the clean dataset.

We include results for two types of stepwise schedules in Appendix A.1.

We first consider label noise that are agnostic to inputs.

Following Zhang et al. (2016) , We randomly replace a 0%/20%/40%) of the training labels to uniformly random ones, and evaluate generalization error on the clean validation set.

We compare with the following baselines: ORACLE, where the model knows the true identity of clean examples and only trains on them; Empirical Risk Minimization (ERM, BID6 ) which assumes all examples are clean; MENTORNET BID13 , which pretrains an auxiliary model that predicts weights for each example based on its input features; REN BID26 , which optimizes the weight of examples via meta-learning; mixup (Zhang et al., 2017) , a data augmentation approach that trains neural networks on convex combinations of pairs of examples and their labels; and Generalized Cross Entropy (GCE, Zhang & Sabuncu (2018) ) that includes cross-entropy loss and mean absolute error BID5 .

We report the top-1 validation error in TAB0 , where denotes methods trained with knowledge of 1000 additional clean labels.

Notably, ODD significantly outperforms all other algorithms (except for the oracle) when there is artificial noise, and is on-par with ERM even when there is no artificial noise.

On the one hand, this suggests that ODD is able to distinguish the mislabeled examples and improve generalization; on the other hand, it would seem that removing certain examples even in the "clean" dataset does not seem to hinder generalization.

ODD-train ODD-valid ODD prevents overfitting to noise We compare the learning curves of ERM and ODD in FIG2 with a stepwise schedule under 40% label corruption.

ERM easily overfits the random labels when learning rate decreases, whereas ODD manages to continue improving generalization.

We run our methods on three random seeds, and find the examples that are considered mislabeled by all the three instances (598 in total); we demonstrate some examples 3 in FIG0 , which contains ambiguous / wrong labels.

Images from some classes could be harder to label correctly than that from other classes.

To simulate this, we perform experiments on settings where the label noise only comes from certain types of input data.

Specifically, we remove a portion of classes from the dataset (e.g. class 9 in CIFAR-10), and assign the labels of all its examples to the remaining classes randomly (e.g. a class 9 example has a class 0 -8 random label).

This reduces the total number of classes, so on the validation set we only consider the classes that are not removed (e.g. classes 0 -8).

We compare ERM and ODD on datasets with 10% or 20% of the examples mislabeled, and summarize the results in TAB1 .

ODD is still able to significantly outperform ERM under such input-dependent noise.

We evaluate ERM and ODD on a setting without mislabeled examples, but the ratio of classes could vary.

To prevent the model from utilizing the number of examples in a class, we combine multiple classes of CIFAR-100 into a single class, creating the CIFAR-20 and CIFAR-50 tasks.

In CIFAR-50, we combine an even class with an odd class while we remove c% of the examples in the odd class.

In CIFAR-20, we combine 5 classes in CIFAR-100 that belong to the same super-class 4 while we remove c% of the examples in 4 out of 5 classes.

This is performed for both training and validation datasets.

Results for ERM and ODD with p = 10 and E = 75 are shown in TAB2 , where ODD is able to outperform ERM in these settings where the input examples are not uniformly distributed.

We conduct additional experiments on the ImageNet-2012 classification dataset BID28 .

The dataset contains 1.28 million training images and 50,000 validation images from 1,000 classes.

Input-agnostic random noise of 0%, 20%, 40% are considered.

We follow standard data augmentation practices during training, including scale and aspect ratio distortions, random crops, and horizontal flips.

We only use the center 224 × 224 crop for validation.

We train ResNet-50 and ResNet-152 models BID7 with the cosine schedule with initial learning rate 0.1, momentum 0.9, weight decay 10 −4 , 90 training epochs, and report top-1 and top-5 validation errors in TAB3 .

ODD significantly outperforms ERM in terms of both top-1 and top-5 errors on the 20% and 40% mislabeled examples, while being competitive with the clean dataset.

We further verify the effectiveness of our method on a real-world noisy dataset.

The WebVision-2017 dataset BID17 contains 2.4 million of real-world noisy labels, that are crawled from Google and Flickr using the 1,000 labels from the ImageNet-2012 dataset.

We train two architectures, ResNet-50 and Inception ResNet-v2 BID32 with the same procedure in the ImageNet experiments, except for Inception ResNet-v2 we train for 50 epochs and use input images of size 299×299.

We use both WebVision and ImageNet validation sets for 1-crop validation, following the settings in BID13 .

We do not use a pretrained model or additional labeled data from ImageNet during training.

Our ODD method with p = 30 removes in the training set around 9.0% of the total examples with ResNet-50 and 9.3% of the total examples with Inception ResNet-v2 BID32 .

TAB4 suggests that our method is able to outperform both ERM and MENTORNET when the training dataset is noisy, even as we remove a notable portion of examples.

We include more results in Appendix A.3.

In comparison, we removed around 1.1% of examples in ImageNet TAB0 , Appendix A.2); this may suggest that WebVision labels are indeed much noisier.

Sensitivity to p We first evaluate noisy ImageNet classification with ResNet-50 where p ∈ {1, 10, 30, 50, 80} and E = 60 in TAB5 .

A higher p includes more clean examples at the cost of involving more noisy examples.

In the 20% and 40% noisy cases, the optimal trade-off for generalization is at p = 10, yet even when p = 50, the validation errors are still significantly better than ERM.

When there is no artificial noise, generalization of ODD starts to match that of ERM as p ≥ 10.

Therefore, ODD is not very sensitive to p in these cases, and empirically p = 10 represents the best trade-off.

We include results for ResNet-152 in Appendix A.2.Sensitivity to E We evaluate the validation error of ODD on CIFAR with 20% and 40% inputagnoistic label noise where E ∈ {25, 50, 75, 100, 150, 200} (E = 200 is equivalent to ERM).

The results in FIG3 suggest that our method is able to separate noisy and clean examples if E is relatively small where the learning rate is high, but is unable to perform well when the learning rate decreases at later stages of the training.

Sensitivity to the amount of noise Finally, we evaluate the training error of ODD on CIFAR under input-agnostic label noise of {1%, 5%, 10%, 20%, 30%, 40%} with p = 5, E = 50 or 75.

This reflects how much examples exceed the threshold and are identified as noise at epoch E. From FIG4 , we observe that the training error is almost exactly the amount of noise in the dataset, which demonstrates that the loss distribution of noise can be characterized by our threshold regardless of the percentage of noise in the dataset.

We have proposed ODD, a straightforward method for robust training with mislabeled examples.

ODD utilizes the implicit regularization effect of stochastic gradient descent to prune examples that potentially harm generalization.

Empirical results demonstrate that ODD is able to significantly outperform related methods on a wide range of datasets with artificial and real-world mislabeled examples, maintain competitiveness with ERM on clean datasets, as well as detecting mislabeled examples automatically in CIFAR-100.The implicit regularization of stochastic gradient descent opens up other research directions for implementing robust algorithms.

For example, we could consider using a smaller network to remove examples, removing examples not only once but multiple times, retraining from scratch with the denoised dataset, or other data-augmentation approaches such as mixup (Zhang et al., 2017) .

Moreover, it would be interesting to understand the implicit regularization over mislabeled examples from a theoretical viewpoint.

A ADDITIONAL EXPERIMENTAL RESULTS

In addition to the existing experiments, we include results for ORACLE, ERM, ODD with two stepwise annealing schedules.

In the stepwise schedules, the learning rate starts from 0.1 and is then divided by 5 after 60, 120, 160 epochs (stepwise-i, which is used in Zagoruyko & Komodakis (2016) ) or after 100, 150, 175 epochs (stepwise-ii).

We consider cosine schedule in Section 4 because it achieves better generalization performance on the clean dataset.

For the stepwise schedules, we set E to be the epoch at which learning rate begins to decay.

For the cosine schedule, we set E = 100 for CIFAR-10 and E = 50 for CIFAR-100.

We set p = 20 and p = 10 for CIFAR-10 and CIFAR-100 respectively for the clean datasets; p = 10 and p = 5 for noisy datasets.

This is motivated by the fact that CIFAR-10 has less labels, so the threshold has to take into account random labels that happens to be correct. (α = 8.0) 3.39 ± 0.12 6.09 ± 0.27 9.26 ± 0.14 Tables 7 and 8 contain summary of the results.

The cosine learning rate schedule generally outperforms the stepwise schedules.

We note that for the stepwise schedules in ERM, the optimal validation error is achieved when the learning rate just starts to decay to 0.2, after that the model starts to overfit to noise, as demonstrated in FIG2 .

We evaluate precision and recall for examples classified as noise on CIFAR10 and CIFAR100 for different noise levels (1, 5, 10, 20, 30, 40) in FIG5 .

The recall values are around 0.84 to 0.88 where as the precision values range from 0.88 to 0.92.

This demonstrates that ODD is able to achieve good precision/recall with default hyperparameters even at different noise levels.

We include the ImageNet ablation experiments on the hyperparameter p on the ResNet-152 architecture in TAB9 .

Compared to the ResNet-50 experiments, we can draw similar conclusions here: p = 10 generally represents the best trade-off.

We show the percentage of examples discarded by NOISE CLASSIFIER in TAB0 ; the percentage of discarded examples by p = 10 is very close to the actual noise level.

Moreover, the percentage of discarded examples does not vary significantly when we change our architecture from ResNet-50 to ResNet-152.

TAB0 , where we report top-1 and top-5 validation errors on WebVision and ImageNet validation sets respectively, as well as how many examples are discarded by our method at epoch 60.

Similar to the results in ImageNet, generalization performance is generally insensitive to the hyperparameter p, except for p = 1, which discarded 25.3% of the examples.

We use 2 seeds for each experiment setting.

Notice that at each p, WebVision has more examples discarded compared to ImageNet (with 0% artificial noise), which further suggests that it has more mislabeled examples than ImageNet.

Again, the percentage of discarded examples does not vary significantly across different architectures.

We display the examples in CIFAR-100 training set for which our ODD methods identify as noise across 3 random seeds.

One of the most common label such examples have is "leopard"; in fact, 21 of 50 "leopard" examples in the training set are perceived as hard, and we show some of them in Figure 7 .

It turns out that a lot of the "leopard" examples contains images that clearly contains tigers and black panthers (CIFAR-100 has a label corresponding to "tiger").Figure 7: Examples with label "leopard" that are classified as noise.

We also demonstrate random examples from the CIFAR-100 that are identified as noise in Figure 8 and those that are not identified as noise in Figure 9 .

The examples identified as noise often contains multiple objects, and those not identified as noise often contains only one object that is less ambiguous in terms of identity.

Figure 9 : Random CIFAR-100 examples that are not classified as noise.

<|TLDR|>

@highlight

We introduce a fast and easy-to-implement algorithm that is robust to dataset noise.

@highlight

The paper aims to remove potential examples with label noise by discarding the ones with large losses in the training procedure.