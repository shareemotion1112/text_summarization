It is fundamental and challenging to train robust and accurate Deep Neural Networks (DNNs) when semantically abnormal examples exist.

Although great progress has been made, there is still one crucial research question which is not thoroughly explored yet: What training examples should be focused and how much more should they be emphasised to achieve robust learning?

In this work, we study this question and propose gradient rescaling (GR) to solve it.

GR modifies the magnitude of logit vector’s gradient to emphasise on relatively easier training data points when noise becomes more severe, which functions as explicit emphasis regularisation to improve the generalisation performance of DNNs.

Apart from regularisation, we connect GR to examples weighting and designing robust loss functions.

We empirically demonstrate that GR is highly anomaly-robust and outperforms the state-of-the-art by a large margin, e.g., increasing 7% on CIFAR100 with 40% noisy labels.

It is also significantly superior to standard regularisers in both clean and abnormal settings.

Furthermore, we present comprehensive ablation studies to explore the behaviours of GR under different cases, which is informative for applying GR in real-world scenarios.

DNNs have been successfully applied in diverse applications (Socher et al., 2011; Krizhevsky et al., 2012; LeCun et al., 2015) .

However, their success is heavily reliant on the quality of training data, especially accurate semantic labels for learning supervision.

Unfortunately, on the one hand, maintaining the quality of semantic labels as the scale of training data increases is expensive and almost impossible when the scale becomes excessively large.

On the other hand, it has been demonstrated that DNNs are capable of memorising the whole training data even when all training labels are random (Zhang et al., 2017) .

Therefore, DNNs struggle to discern meaningful data patterns and ignore semantically abnormal examples 1 simultaneously (Krueger et al., 2017; Arpit et al., 2017) .

Consequently, it becomes an inevitable demand for DNNs to hold robustness when training data contains anomalies (Larsen et al., 1998; Natarajan et al., 2013; Sukhbaatar & Fergus, 2014; Xiao et al., 2015; Patrini et al., 2017; Vahdat, 2017; Veit et al., 2017; Li et al., 2017) .

Recently, great progress has been made towards robustness against anomalies when training DNNs (Krueger et al., 2017) .

There are three appealing perspectives in terms of their simplicity and effectiveness: 1) Examples weighting.

For example, knowledge distilling from auxiliary models is popular for heuristically designing weighting schemes.

However, it is challenging to select and train reliable auxiliary models in practice (Li et al., 2017; Malach & Shalev-Shwartz, 2017; Jiang et al., 2018; Ren et al., 2018; Han et al., 2018b) .

2) Robust loss functions (Van Rooyen et al., 2015; Ghosh et al., 2017; Zhang & Sabuncu, 2018; Wang et al., 2019b) ; 3) Explicit regularisation techniques (Arpit et al., 2017; .

Although designing robust losses or explicit regularisation is easier and more flexible in practice, the performance is not the optimal yet.

1 One training example is composed of an input and its corresponding label.

A semantically abnormal example means the input is semantically unrelated to its label, which may come from corrupted input or label.

For example, in Figure 3 in the supplementary material: 1) Out-of-distribution anomalies: An image may contain only background or an object which does not belong to any training class; 2) In-distribution anomalies: An image of class a may be annotated to class b or an image may contain more than one semantic object.

Regarding examples weighting, there is a core research question which is not well answered yet:

What training examples should be focused on and how large the emphasis spread should be?

In this work, we present a thorough study of this practical question under different settings.

For better analysis, we propose two basic and necessary concepts: emphasis focus and spread with explicit definition in Sec. 3.2.

They are conceptually introduced as follows:

Emphasis focus.

It is a common practice to focus on harder instances when training DNNs (Shrivastava et al., 2016; Lin et al., 2017) .

When a dataset is clean, it achieves faster convergence and better performance to emphasise on harder examples because they own larger gradient magnitude, which means more information and a larger update step for model's parameters.

However, when severe noise exists, as demonstrated in (Krueger et al., 2017; Arpit et al., 2017) , DNNs learn simple meaningful patterns first before memorising abnormal ones.

In other words, anomalies are harder to fit and own larger gradient magnitude in the later stage.

Consequently, if we use the default sample weighting in categorical cross entropy (CCE) where harder samples obtain higher weights, anomalies tend to be fitted well especially when a network has large enough capacity.

That is why we need to move the emphasis focus towards relatively easier ones, which serves as emphasis regularisation.

Emphasis spread.

We term the weighting variance of training examples emphasis spread.

The key concept is that we should not treat all examples equally, neither should we let only a few be emphasised and contribute to the training.

Therefore, when emphasis focus changes, the emphasis spread should be adjusted accordingly.

We integrate emphasis focus and spread into a unified example weighting framework.

Emphasis focus defines what training examples own higher weights while emphasis spread indicates how large variance over their weights.

Specifically, we propose gradient rescaling (GR), which modifies the magnitude of logit vector's gradient.

The logit vector is the output of the last fully connected (FC) layer of a network.

We remark that we do not design the weighting scheme heuristically from scratch.

Instead, it is naturally motivated by the gradient analysis of several loss functions.

Interestingly, GR can be naturally connected to examples weighting, robust losses, explicit regularisation: 1) The gradient magnitude of logit vector can be regarded as weight assignment that is built-in in loss functions (Gopal, 2016; Alain et al., 2016; Zhang et al., 2018b) .

Therefore, rescaling the gradient magnitude equals to adjusting the weights of examples; 2) A specific loss function owns a fixed gradient derivation.

Adjusting the gradient can be treated as a more direct and flexible way of modifying optimisation objectives; 3) Instead of focusing on harder examples 2 by default, we can adjust emphasis focus to relative easier ones when noise is severe.

GR serves as emphasis regularisation and is different from standard regularisers, e.g., L2 weight decay constraints on weight parameters and Dropout samples neural units randomly (Srivastava et al., 2014) ; GR is simple yet effective.

We demonstrate its effectiveness on diverse computer vision tasks using different net architectures: 1) Image classification with clean training data; 2) Image classification with synthetic symmetric label noise, which is more challenging than asymmetric noise evaluated by (Vahdat, 2017; ; 3) Image classification with real-world unknown anomalies, which may contain open-set noise , e.g., images with only background, or outliers, etc.; 4) Video person re-identification, a video retrieval task containing diverse anomalies.

Beyond, we show that GR is notably better than other standard regularisers, e.g., L2 weight decay and dropout.

Besides, to comprehensively understand GR's behaviours, we present extensive ablation studies.

Main contribution.

Intuitively and principally, we claim that two basic factors, emphasis focus and spread, should be babysat simultaneously when it comes to examples weighting.

To the best of our knowledge, we are the first to thoroughly study and analyse them together in a unified framework.

Aside from examples weighting, robust losses minimisation and explicit regularisation techniques, there are another two main perspectives for training robust and accurate DNNs when anomalies exist:

2 An example's difficulty can be indicated by its loss (Shrivastava et al., 2016; Loshchilov & Hutter, 2016; Hinton, 2007) , gradient magnitude (Gopal, 2016; Alain et al., 2016) , or input-to-label relevance score (Lee et al., 2018) .

The input-to-label relevance score means the probability of an input belonging to its labelled class predicted by a current model.

The difficulty of an example may change as the model learns.

In summary, higher difficulty, larger loss, larger gradient magnitude, and lower input-to-label relevance score are equal concepts.

1) Robust training strategies (Miyato et al., 2018; Guo et al., 2018; Li et al., 2019; Thulasidasan et al., 2019) ; 2) Noise-aware modelling, and alternative label and parameter optimisation are popular when only label noise exists.

Some methods focus on noise-aware modelling for correcting noisy labels or empirical losses (Larsen et al., 1998; Natarajan et al., 2013; Sukhbaatar & Fergus, 2014; Xiao et al., 2015; Vahdat, 2017; Veit et al., 2017; Goldberger & Ben-Reuven, 2017; Han et al., 2018a) .

However, it is non-trivial and time-consuming to learn a noise-aware model, which also requires prior extra information or some specific assumptions.

For example, Masking (Han et al., 2018a ) is assisted by human cognition to speculate the noise structure of noise-aware matrix while (Veit et al., 2017; Li et al., 2017; Lee et al., 2018; Hendrycks et al., 2018) exploit an extra clean dataset, which is a hyper-factor and hard to control in practice.

Some other algorithms iteratively train the model and infer latent true labels Tanaka et al., 2018) .

Those methods have made great progress on label noise.

But they are not directly applicable to unknown diverse semantic anomalies, which covers both out-of-distribution and in-distribution cases.

We note that (Ghosh et al., 2017) proposed some theorems showing that empirical risk minimization is robust when the loss function is symmetric and the noise type is label noise.

However, they are not applicable for deep learning under arbitrary unknown noise: 1) We remark that we target at the problem of diverse or arbitrary abnormal examples, where an input may be out-of-distribution, i.e., not belonging to any training class.

As a result, the symmetric losses custom-designed for label noise are not applicable.

2) GR is independent of empirical loss expressions as presented in Table 1 .

Therefore, one specific loss is merely an indicator of how far we are away from a specific minimisation objective.

It has no impact on the robustness of the learning process since it has no direct influence on the gradient back-propagation.

Similar to the prior work of rethinking generalisation (Zhang et al., 2017) , we need to rethink robust training under diverse anomalies, where the robustness theorems conditioned on symmetric losses and label noise are not directly applicable.

Notation.

We are given N training examples

, where (x i , y i ) denotes i−th sample with input x i ∈ R D and label y i ∈ {1, 2, ..., C}. C is the number of classes.

Let's consider a deep neural network z composed of an embedding network f (·) :

Generally, the linear classifier is the last FC layer which produces the final output of z, i.e., logit vector z ∈ R C .

To obtain probabilities of a sample belonging to different classes, logit vector is normalised by a softmax function:

(1) p(j|x i ) is the probability of x i belonging to class j. A sample's input-to-label relevance score is defined by

In what follows, we will uncover the sample weighting in popular losses: CCE, Mean Absolute Error (MAE) and Generalised Cross Entropy (GCE) (Zhang & Sabuncu, 2018) .

CCE.

The CCE loss with respect to (x i , y i ), and its gradient with respect to z ij are defined as:

Therefore, we have ||

Here we choose L1 norm to measure the magnitude of gradient because of its simpler statistics and computation.

Since we back-propagate ∂L CCE /z i to update the model's parameters, an example's gradient magnitude determines how much impact it has, i.e., its weight w

In CCE, more difficult examples with smaller p i get higher weight.

When it comes to MAE, the loss of (x i , y i ) and gradient with respect to z im are: (a) GR, CCE, MAE, GCE.

We show 3 settings of GR: (β = 2, λ = 0), (β = 8, λ = 0.5) and (β = 12, λ = 1).

Their corresponding emphasis focuses are 0, 0∼0.5 and 0.5.

(b) GR when fixing λ = 0.5 (emphasis focus is within 0∼0.5) or λ = 2 (emphasis focus is within 0.5∼1).

(c) GR when fixing β = 8.

When λ increases, the emphasis focus moves towards 1 and emphasis spread drops.

Figure 1: A sample's weight w i along with its input-to-label relevance score p i .

GR is a unified sample reweighting framework from the perspective of gradient rescaling, where the emphasis focus and spread can be adjusted by choosing proper λ and β in practice.

Better viewed in colour.

Therefore, w

In MAE, those images whose input-to-label relevance scores are 0.5 become the emphasis focus.

GCE.

In GCE, the loss calculation of (x i , y i ) and gradient with respect to logit vector z i are:

where

In this case, the emphasis focus can be adjusted from 0 to 0.5 when q ranges from 0 to 1.

However, in their practice (Zhang & Sabuncu, 2018) , instead of using this naive version, a truncated one is applied:

The loss of an example with p i ≤ 0.5 is constant so that its gradient is zero, which means it is dropped and does not contribute to the training.

The main drawback is that at the initial stage, the model is not well learned so that the predicted p i of most samples are smaller than 0.5.

To address it, alternative convex search is exploited for iterative data pruning and parameters optimisation, making it quite complex and less appealing in practice.

The derivation details of Eq. (2), (3), (4) are presented in Section B of the supplementary material.

A loss function provides supervision information by its derivative with respect to a network's output.

Therefore, there are two perspectives for improving the supervision information: 1) Modifying the loss format to improve its corresponding derivative; 2) Manipulating the gradient straightforwardly.

In this work, we choose to control the gradient, which is more direct and flexible.

According to Eq. (2), (3), (4), the gradients of CCE, MAE and GCE share the same direction.

Our proposal GR unifies them from the gradient perspective.

Being independent of loss formulas, a sample's gradient is rescaled linearly so that its weight is w GR i :

where λ, β are hyper-parameters for controlling the emphasis focus and spread, respectively.

By choosing a larger λ when more anomalies exist, GR regularises examples weighting by moving emphasis focus toward relatively easier training data points, thus embracing noise-robustness.

For clarification, we explicitly define the emphasis focus and spread over training examples: Definition 1 (Emphasis Focus ψ).

The emphasis focus refers to those examples that own the largest weight.

Since an example's weight is determined by its input-to-label relevance score p i , for simplicity, we define the emphasis focus to be an input-to-label score to which the largest weight is assigned, i.e., ψ = arg max

Definition 2 (Emphasis Spread σ).

The emphasis spread is the weight variance over all training instances in a mini-batch, i.e., σ = E((w

, where E(·) denotes the expectation value of a variable.

With these definitions, we differentiate GR with other methods in Table 1 .

We show the sample weighting curves of GR with different settings in Figure 1 .

As shown in Figure 1c , the emphasis spread declines as λ increases.

Therefore, we choose larger β values when λ is larger in Sec. 4.2.1.

Principally, transformation g could be designed as any monotonically increasing function.

Because the non-linear exponential mapping can change the overall weights' variance and relative weights between any two examples, we choose g(·) = exp(·), which works well in our practice.

By integral, the exact loss format is an error function (non-elementary).

We summarise several existing cases as follows (the ellipsis refers to other potential options which can be explored in the future): Let's regard a deep network z as a black box, which produces C logits.

C is the class number.

Then during gradient back-propagation, an example's impact on the update of z is determined by its gradient w.r.t.

the logit vector.

The impact can be decomposed into two factors, i.e., gradient direction and magnitude.

To reduce the impact of a noisy sample, we can either reduce its gradient magnitude or amend its gradient direction.

In this work, inspired by the analysis of CCE, MAE and GCE, which only differ in the gradient magnitude while perform quite differently, leading to a natural motivation that gradient magnitude matters.

That is why we explore rescaling the gradient magnitude as illustrated in Figure 1 .

It is worth studying amending gradient directions in the future.

Datasets.

We test on CIFAR-10 and CIFAR-100 (Krizhevsky, 2009), which contain 10 and 100 classes, respectively.

In CIFAR-10, the training data contains 5k images per class while the test set includes 1k images per class.

In CIFAR-100, there are 500 images per class for training and 100 images per class for testing.

Implementation details.

On CIFAR-10, following (He et al., 2016) , we adopt ResNet-20 and ResNet-56 as backbones so that we can compare fairly with their reported results.

On CIFAR-100, we follow D2L to choose ResNet-44 and compare with its reported results.

We also use an SGD optimiser with momentum 0.9 and weight decay 10 −4 .

The learning rate is initialised with 0.1, and multiplied with 0.1 every 5k iterations.

We apply the standard data augmentation as in (He et al., 2016; : The original images are padded with 4 pixels on every side, followed by a random crop of 32 × 32 and horizontal flip.

The batch size is 128.

Table 2 : Classification accuracies (%) of CCE, and GR on clean CIFAR-10 and CIFAR-100.

λ = 0 means the emphasis focus is 0 where we fix β = 2.

β = 0 means all examples are treated equally.

Backbone CCE GR (λ = 0) GR (β = 0) Results.

Our purpose is to show GR can achieve competitive performance with CCE under clean data to demonstrate its general applicability.

As reported in D2L, all noise-tolerant proposals (Patrini et al., 2017; perform similarly with CCE when training labels are clean.

Therefore we do not present other related competitors here.

Our reimplemented results are shown in Table 2 .

For reference, the reported results in (He et al., 2016) on CIFAR-10 with CCE are 91.3% for ResNet-20 and 93.0% for ResNet-56.

In D2L, the result on CIFAR-100 with ResNet-44 is 68.2%.

Our reimplemented performance of CCE is only slightly different.

For GR, we observe the best performance when emphasis focus is 0, i.e., λ = 0.

Furthermore, it is insensitive to a wide range of emphasis spreads according to our observations in Figure 5 in the supplementary material.

Treating training examples equally.

As shown in Table 2 , we obtain competitive performance by treating all training examples equally when β = 0.

This is quite interesting and motivates us that sample differentiation and reweighting work much better only when noise exists.

Symmetric noise generation.

Given a probability r, the original label of an image is changed to one of the other class labels uniformly following (Tanaka et al., 2018; .

r denotes the noise rate.

Symmetric label noise generally exists in large-scale real-world applications where the dataset scale is so large that label quality is hard to guarantee.

It is also demonstrated in (Vahdat, 2017) that it is more challenging than asymmetric noisy labels Patrini et al., 2017) , which assume that label errors only exist within a predefined set of similar classes.

All augmented training examples share the same label as the original one.

To understand GR well empirically, we explore the behaviours of GR on CIFAR-10 with r = 20%, 40%, 60%, 80%, respectively.

We use ResNet-56 which has larger capacity than ResNet-20.

Design choices.

We mainly analyse the impact of different emphasis focuses for different noise rates.

We explore 5 emphasis focuses by setting β = 0 or different λ: 1) None: β = 0.

There is no emphasis focus since all examples are treated equally; 2) 0: λ = 0; 3) 0∼0.5: λ = 0.5; 4) 0.5: λ = 1; 5) 0.5∼1: λ = 2.

We remark that when λ is larger, the emphasis focus is higher, leading to relatively easier training data points are emphasised.

As shown in Figure 1 , when emphasis focus changes, emphasis spread changes accordingly.

Therefore, to set a proper spread for each emphasis focus, we try 4 emphasis spread and choose the best one 3 to compare the impact of emphasis focus.

Results analysis.

We show the results in Table 3 .

The intact training set serves as a validation set and we observe that its accuracy is always consistent with the final test accuracy.

This motivates us that we can choose our model's hyper-parameters β, λ via a validation set in practice.

We display the training dynamics in Figure 2 .

We summarise our observations as follows: Fitting and generalisation.

We observe that CCE always achieves the best accuracy on corrupted training sets, which indicates that CCE has a strong data fitting ability even if there is severe noise (Zhang et al., 2017) .

As a result, CCE has much worse final test accuracy than most models.

Emphasising on harder examples.

When there exist abnormal training examples, we obtain the worst final test accuracy if emphasis focus is 0, i.e., CCE and GR with λ = 0.

This unveils that in applications where we have to learn from noisy training data, it will hurt the model's generalisation dramatically if we use CCE or simply focus on harder training data points.

Emphasis focus.

When noise rate is 0, 20%, 40%, 60%, and 80%, we obtain the best final test accuracy when λ = 0, λ = 0.5, λ = 1, λ = 2, and λ = 2, respectively.

This demonstrates that when noise rate is higher, we can improve a model's robustness by moving emphasis focus towards relatively less difficult examples with a larger λ, which is informative in practice.

Emphasis spread.

As displayed in Table 3 and Figures 7-10 in the supplementary material, emphasis spread also matters a lot when fixing emphasis focus, i.e., fixing λ.

For example in Table 3 , when λ = 0, although focusing on harder examples similarly with CCE, GR can outperform CCE by modifying the emphasis spread.

As shown in Figures 7-10, some models even collapse and cannot converge if the emphasis spread is not rational.

Implementation details.

We follow the same settings as MentorNet (Jiang et al., 2018) to compare fairly with its reported results.

Optimiser and data augmentation are described in Section 4.1.

Competitors.

FullModel is the standard CCE trained using L2 weight decay and dropout (Srivastava et al., 2014) .

Forgetting (Arpit et al., 2017) searches the dropout parameter in the range of (0.2-0.9).

Self-paced (Kumar et al., 2010), Focal Loss (Lin et al., 2017) , and MentorNet (Jiang et al., 2018) are representatives of example reweighting algorithms.

Reed Soft ) is a weaklysupervised learning method.

All methods use GoogLeNet V1 .

Results.

We compare the results under different noise rates in Table 4 .

GR with fixed hyperparameters β = 8, λ = 0.5 outperforms the state-of-the-art GCE by a large margin, especially when label noise becomes severe.

Better results can be expected when optimising the hyper-parameters for each case.

We remark that FullModel (naive CCE) (Jiang et al., 2018) was trained with L2 weight decay and dropout.

However, GR's regularization effect is much better in both clean and noisy cases.

Figure 6 in the supplementary material.

We have two key observations: 1) When noise rate increases, better generalisation is obtained with higher emphasis focus, i.e., focusing on relatively easier examples; 2) Both overfitting and underfitting lead to bad generalisation.

For example, 'CCE: 0' fits training data much better than the others while 'GR: None' generally fits it unstably or a lot worse.

Better viewed in colour.

Implementation details.

Most baselines have been reimplemented in with the same settings.

Therefore, for direct comparison, we follow exactly their experimental configurations and use ResNet-44 (He et al., 2016) .

Optimiser and data augmentation are described in Section 4.1.

We repeat training and evaluation 5 times where different random seeds are used for generating noisy labels and model's initialisation.

The mean test accuracy and standard deviation are reported.

Competitors.

We compare with D2L , GCE (Zhang & Sabuncu, 2018) , and other baselines reimplemented in D2L: 1) Standard CCE ; 2) Forward (Patrini et al., 2017 ) uses a noise-transition matrix to multiply the network's predictions for label correction; 3) Backward (Patrini et al., 2017) applies the noise-transition matrix to multiply the CCE losses for loss correction; 4) Bootstrapping trains models with new labels generated by a convex combination of the original ones and their predictions.

The convex combination can be soft (Boot-soft) or hard (Boot-hard); 5) D2L achieves noise-robustness from a novel perspective of restricting the dimensionality expansion of learned subspaces during training and is the state-of-the-art; 6) Since GCE outperforms MAE (Zhang & Sabuncu, 2018) , we only reimplement GCE for comparison; 7) SL (Wang et al., 2019c) boosts CCE symmetrically with a noise-robust counterpart, i.e., reverse cross entropy.

Results.

We compare the results of GR and other algorithms in Table 5 .

GR outperforms other competitors by a large margin, especially when label noise is severe, e.g., r = 40% and 60%.

More importantly, we highlight that GR is much simpler without any extra information.

Compared with Forward and Backward, GR does not need any prior knowledge about the noise-transition matrix.

Bootstrapping targets at label correction and is time-consuming.

D2L estimates the local intrinsic dimensionality every b mini-batches and checks the turning point for dimensionality expansion every e epochs.

However, b and e are difficult to choose and iterative monitoring is time-consuming.

Dataset.

Clothing 1M (Xiao et al., 2015) contains 1 million images.

It is an industrial-level dataset and its noise structure is agnostic.

According to (Xiao et al., 2015) , around 61.54% training labels are reliable, i.e., the noise rate is about 38.46%.

There are 14 classes from several online shopping websites.

In addition, there are 50k, 14k, and 10k images with clean labels for training, validation, Table 5 : The accuracies (%) of GR and recent approaches on CIFAR-100.

The results of fixed parameters (β = 8, λ = 0.5) are shown in the second last column.

With a little effort for optimising β and λ, the results and corresponding parameters are presented in the last column.

The trend is consistent with Table 3 : When r raises, we can increase β, λ for better robustness.

The increasing scale is much smaller.

This is because CIFAR-100 has 100 classes so that its distribution of p i (input-to-label relevance score) is different from CIFAR-10 after softmax normalisation.

and testing, respectively.

Here, we follow and compare with existing methods that only learn from noisy training data since we would like to avoid exploiting auxiliary information.

Implementation details.

We train ResNet-50 (He et al., 2016) and follow exactly the same settings as (Patrini et al., 2017; Tanaka et al., 2018) : 1) Initialisation: ResNet-50 is initialised by publicly available model pretrained on ImageNet (Russakovsky et al., 2015) ; 2) Optimisation: A SGD optimiser with a momentum of 0.9 and a weight decay of 10 −3 is applied.

The learning rate starts at 10 −3 and is divided by 10 after 5 epochs.

Training terminates at 10 epochs; 3) Standard data augmentation: We first resize a raw input image to 256 × 256, and then crop it randomly at 224 × 224 followed by random horizontal flipping.

The batch size is 64 due to memory limitation.

Since the noise rate is around 38.46%, we simply set λ = 1, β = 16 following Table 3 when noise rate is 40%.

Competitors.

We compare with other noise-robust algorithms that have been evaluated on Clothing 1M with similar settings: 1) Standard CCE (Patrini et al., 2017) ; 2) Since Forward outperforms Backward on Clothing 1M (Patrini et al., 2017) , we only present the result of Forward; 3) S-adaptation applies an additional softmax layer to estimate the noise-transition matrix (Goldberger & Ben-Reuven, 2017 ); 4) Masking is a human-assisted approach that conveys human cognition to speculate the structure of the noise-transition matrix (Han et al., 2018a) .

5) Label optimisation (Tanaka et al., 2018) learns latent true labels and model's parameters iteratively.

Two regularisation terms are added for label optimisation and adjusted in practice.

Results.

The results are compared in Table 6 .

Under real-world agnostic noise, GR also outperforms the state-of-the-art.

It is worth mentioning that the burden of noise-transition matrix estimation in Forward and S-adaptation is heavy due to alternative optimisation steps, and such estimation is non-trivial without big enough data.

Masking exploits human cognition of a structure prior and reduces the burden of estimation, nonetheless its performance is not competitive.

Similarly, Label Optimisation requires alternative optimisation steps and is time-consuming.

Dataset and evaluation settings.

MARS contains 20,715 videos of 1,261 persons (Zheng et al., 2016) .

There are 1,067,516 frames in total.

Because person videos are collected by tracking and detection algorithms, abnormal examples exist as shown in Figure 3 in the supplementary material.

We remark that there are some anomalies containing only background or an out-of-distribution person.

Exact noise type and rate are unknown.

Following standard settings, we use 8,298 videos of 625 persons for training and 12,180 videos of the other 636 persons for testing.

We report the cumulated matching characteristics (CMC) and mean average precision (mAP) results. (Patrini et al., 2017) and (Wang et al., 2019c) , respectively.

CCE* and GCE* are our reproduced results using the Caffe framework (Jia et al., 2014 Implementation details.

Following (Liu et al., 2017; Wang et al., 2019a) , we train GoogleNet V2 (Ioffe & Szegedy, 2015) and treat a video as an image set, which means we use only appearance information without exploiting latent temporal information.

A video's representation is simply the average fusion of its frames' representations.

The learning rate starts from 0.01 and is divided by 2 every 10k iterations.

We stop training at 50k iterations.

We apply an SGD optimiser with a weight decay of 0.0005 and a momentum of 0.9.

The batch size is 180.

We use standard data augmentation: a 227 × 227 crop is randomly sampled and flipped after resizing an original image to 256 × 256.

Training settings are the same for each method.

We implement GCE with its reported best settings.

At testing, following (Wang et al., 2019a; Movshovitz-Attias et al., 2017; Law et al., 2017) , we first L 2 normalise videos' features and then calculate the cosine similarity between every two of them.

Results.

The results are displayed in Table 7 .

Although DRSA and CAE (Chen et al., 2018) exploit extra temporal information by incorporating attention mechanisms, GR is superior to them in terms of both effectiveness and simplicity.

OSM+CAA (Wang et al., 2019a ) is the only comparable method.

However, OSM+CAA combines CCE and weighted contrastive loss to address anomalies, thus being more complex than GR.

In addition, we highlight that one query may have multiple matching instances in the MARS benchmark.

Consequently, mAP is a more reliable and accurate performance assessment.

GR is the best in terms of mAP.

In Table 8 , we compare our proposed regulariser GR with other standard ones, i.e., L2 weight decay and Dropout (Srivastava et al., 2014) .

We set the dropout rate to 0.2 and L2 weight decay rate to 10 −4 .

For GR, as mentioned in Section 4.2.3, we fix β = 8, λ = 0.5.

Interestingly, Dropout+L2 achieves 52.8% accuracy, which is even better than the state-of-the-art in Table 5 , i.e., D2L with 52.0% accuracy.

However, GR is better than those standard regularisers and their combinations significantly.

GR works best when it is together with L2 weight decay.

Table 8 : Results of GR and other standard regularisers on CIFAR-100.

We set r = 40%, i.e., the label noise is severe but not belongs to the majority.

We train ResNet-44.

We report the average test accuracy and standard deviation (%) over 5 trials.

Baseline means CCE without regularisation.

In this work, we present three main contributions: 1) We analyse and answer a core research question: What training examples should be focused on and how large the emphasis spread should be?

2) We uncover and analyse that two basic factors, emphasis focus and spread, should be babysat simultaneously when it comes to examples weighting.

Consequently, we propose a simple yet effective gradient rescaling framework serving as emphasis regularisation.

3) Extensive experiments on different tasks using different network architectures are reported for better understanding and demonstration of GR's effectiveness, which are also valuable for applying GR in practice. (Zheng et al., 2016) .

Out-of-distribution anomalies: 1) The first image in the 3rd row contains only background and no semantic information at all.

2) The 2nd first image or the last one in the 3rd row may contain a person that does not belong to any person in the training set.

In-distribution anomalies: 1) Some images of deer class are wrongly annotated to horse class.

2) We cannot decide the object of interest without any prior when an image contains more than one object, e.g., some images contain two persons in the 2nd row.

For left and right sides of Eq. (8), we calculate their derivatives w.r.t.

z ij simultaneously.

If j = y i ,

In summary, the derivation of softmax layer is:

B.2 DERIVATION OF CCE According to Eq. (2), we have

Therefore, we obtain (the parameters are omitted for brevity),

B.3 DERIVATION OF MAE According to Eq. (3), we have

Therefore, we obtain

According to Eq. (4), we have

Therefore, we obtain

B.5 DERIVATIVES W.R.T. LOGITS z i

The calculation is based on Eq. (13) and Eq. (11).

If j = y i , we have:

If j = y i , it becomes:

In summary, ∂L CCE /∂z i can be represented as:

otherwise (j = y i ):

In summary, ∂L MAE /∂z i is:

The calculation is based on Eq. (17) and Eq. (11).

If j = y i , we have:

If j = y i , it becomes:

In summary, ∂L GCE /∂z i can be represented as:

C SMALL-SCALE FINE-GRAINED VISUAL CATEGORISATION OF VEHICLES How does GR perform on small datasets, for example, the number of data points is no more than 5,000?

We have tested GR on CIFAR-10 and CIFAR-100 in the main paper.

However, both of them contain a training set of 50,000 images.

For this question, we answer it from different perspectives as follows:

1.

The problem of label noise we study on CIFAR-10 and CIFAR-100 in Section 4.2 is of similar scale.

For example:

• In Table 4 , when noise rate is 80% on CIFAR-10, the number of clean training examples is around 50, 000 × 20% = 5, 000 × 2.

Therefore, this clean set is only two times as large as 5,000.

Beyond, the learning process may be interrupted by other noisy data points.

• In Table 5 , when noise rate is 60% on CIFAR-100, the number of clean training data points is about 50, 000 × 40% = 5, 000 × 4, i.e., four times as large as 5,000.

2.

We compare GR with other standard regularisers on a small-scale fine-grained visual categorisation problem in Table 9 .

Vehicles-10 Dataset.

In CIFAR-100 Krizhevsky (2009), there are 20 coarse classes, including vehicles 1 and 2.

Vehicles 1 contains 5 fine classes: bicycle, bus, motorcycle, pickup truck, and train.

Vehicles 2 includes another 5 fine classes: lawn-mower, rocket, streetcar, tank, and tractor.

We build a small-scale vehicles classification dataset composed of these 10 vehicles from CIFAR-100.

Specifically, the training set contains 500 images per vehicle class while the testing set has 100 images per class.

Therefore, the number of training data points is 5,000 in total.

We evaluate on CIFAR-100, whose 100 classes are grouped into 20 coarse classes.

Every coarse class has 5 fine classes.

Within each coarse class, an image's label is flipped to one of the other four labels uniformly with a probability r. r represents the noise rate.

We set r = 0.2.

The results are displayed in Table 10 .

When GR is used, the performance is better than its counterparts without GR.

The results are shown in Table 11 .

Proposal: Gradient rescaling incorporates emphasis focus (centre/focal point) and emphasis spread, and serves as explicit regularisation in terms of sample reweighting/emphasis.

Finding: When noise rate is higher, we can improve a model's robustness by moving emphasis focus towards relatively less difficult examples.

The more detailed results on CIFAR-100 are shown in Table 12 , which is the supplementary of Table 5 in the main text.

Table 12 : Exploration of GR with different emphasis focuses (centres) and spreads on CIFAR-100 when r = 20%, 40%, 60%, respectively.

This table presents detailed information of optimising λ, β mentioned in Table 5 in the paper.

Specifically, for each λ, we try 5 β values from {2, 4, 6, 8, 10} and select the best one as the final result of the λ.

We report the mean test accuracy over 5 repetitions.

Our key finding is demonstrated again: When r raises, we can increase β, λ for better robustness.

The increasing scale is much smaller than CIFAR-10.

This is because CIFAR-100 has 100 classes so that its distribution of p i (input-to-label relevance score) is different from CIFAR-10 after softmax normalisation.

Figure 2 in the paper.

We have two key observations: 1) When noise rate increases, better generalisation is obtained with higher emphasis focus, i.e., focusing on relatively easier examples; 2) Both overfitting and underfitting lead to bad generalisation.

For example, 'CCE: 0' fits training data much better than the others while 'GR: None' generally fits it unstably or a lot worse.

Better viewed in colour. .

From left to right, the results of four emphasis focuses 0, 0∼0.5, 0.5, 0.5∼1 with different emphasis spreads are displayed in each column respectively.

When λ is larger, β should be larger as displayed in Figure 1c in the paper.

Specifically : 1) when λ = 0: we tried β = 0.5, 1, 2, 4; 2) when λ = 0.5: we tried β = 4, 8, 12, 16; 3) when λ = 1: we tried β = 8, 12, 16, 20; 4) when λ = 2: we tried β = 12, 16, 20, 24.

@highlight

ROBUST DISCRIMINATIVE REPRESENTATION LEARNING VIA GRADIENT RESCALING: AN EMPHASIS REGULARISATION PERSPECTIVE