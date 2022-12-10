The machine learning and computer vision community is witnessing an unprecedented rate of new tasks being proposed and addressed, thanks to the power of deep convolutional networks to find complex mappings from X to Y. The advent of each task often accompanies the release of a large-scale human-labeled dataset, for supervised training of the deep network.

However, it is expensive and time-consuming to manually label sufficient amount of training data.

Therefore, it is important to develop algorithms that can leverage off-the-shelf labeled dataset to learn useful knowledge for the target task.

While previous works mostly focus on transfer learning from a single source, we study multi-source transfer across domains and tasks (MS-DTT), in a semi-supervised setting.

We propose GradMix, a model-agnostic method applicable to any model trained with gradient-based learning rule.

GradMix transfers knowledge via gradient descent, by weighting and mixing the gradients from all sources during training.

Our method follows a meta-learning objective, by assigning layer-wise weights to the source gradients, such that the combined gradient follows the direction that can minimize the loss for a small set of samples from the target dataset.

In addition, we propose to adaptively adjust the learning rate for each mini-batch based on its importance to the target task, and a pseudo-labeling method to leverage the unlabeled samples in the target domain.

We perform experiments on two MS-DTT tasks: digit recognition and action recognition, and demonstrate the advantageous performance of the proposed method against multiple baselines.

Deep convolutional networks (ConvNets) have significantly improved the state-of-the-art for visual recognition, by finding complex mappings from X to Y. Unfortunately, these impressive gains in performance come only when massive amounts of paired labeled data (x, y) s.t.

x ∈ X , y ∈ Y are available for supervised training.

For many application domains, it is often prohibitive to manually label sufficient training data, due to the significant amount of human efforts involved.

Hence, there is strong incentive to develop algorithms that can reduce the burden of manual labeling, typically by leveraging off-the-shelf labeled datasets from other related domains and tasks.

There has been a large amount of efforts in the research community to address adapting deep models across domains BID5 BID16 BID31 , to transfer knowledge across tasks BID17 BID7 BID34 , and to learn efficiently in a few shot manner BID4 BID22 BID23 .

However, most works focus on a single-source and single-target scenario.

Recently, some works BID33 BID19 propose deep approaches for multi-source domain adaptation, but they assume that the source and target domains have shared label space (task).In many computer vision applications, there often exist multiple labeled datasets available from different domains and/or tasks related to the target application.

Hence, it is important and practically valuable that we can transfer knowledge from as many source datasets as possible.

In this work, we formalize this problem as multi-source domain and task transfer (MS-DTT).

Given a set of labeled source dataset, S = {S 1 , S 2 , ..., S k }, we aim to transfer knowledge to a sparsely labeled target dataset T .

Each source dataset S i could come from a different domain compared to T , or from a different task, or different in both domain and task.

We focus on a semi-supervised setting, where only few samples in T have labels.

Most works achieve domain transfer by aligning the feature distribution of source domain and target domain BID15 BID5 BID30 BID19 BID33 .

However, this method could be suboptimal for MS-DTT.

The reason is that in MS-DTT, the distribution of source data p(x Si , y Si ) and target data p(x T , y T ) could be significantly different in both input space and label space, thus simply aligning their input space may generate indiscriminative features for the target classes.

In addition, feature alignment introduces additional layers and loss terms, which require careful design to perform well.

In this work, we propose a generic and scalable method, namely GradMix, for semi-supervised MS-DTT.

GradMix is a model-agnostic method, applicable to any model that uses gradient-based learning rule.

Our method does not introduce extra layers or loss functions for feature alignment.

Instead, we perform knowledge transfer via gradient descent, by weighting and mixing the gradients from all the source datasets during training.

We follow a meta-learning paradigm and model the most basic assumption: the combined gradient should minimize the loss for a set of unbiased samples from the target dataset.

We propose an online method to weight and mix the source gradients at each training iteration, such that the knowledge most useful for the target task is preserved through the gradient update.

Our method can adaptively adjust the learning rate for each mini-batch based on its importance to the target task.

In addition, we propose a pseudo-labeling method based on model ensemble to learn from the unlabeled data in target domain.

We perform extensive experiments on two sets of MS-DTT task, including digit recognition and action recognition, and demonstrate the advantageous performance of the proposed method compared to multiple baselines.

Our code is available at https://www.url.com.

Domain Adaptation.

Domain adaptation seeks to learn from source domain a well-performing model on the target domain, by addressing the domain shift problem BID2 .

Most existing works focus on aligning the feature distribution of the source domain and target domain.

Several works attempt to learn domain-invariant features by minimizing Maximum Mean Discrepancy BID15 BID28 .

Another class of method uses adversarial discriminative models, i.e. learn domain-agnostic representations by maximizing a domain confusion loss BID5 BID30 BID17 .

Recently, multi-source domain adaptation with deep model has been studied.

BID19 use DA-layers BID1 BID13 to minimize the distribution discrepancy of network activations.

BID33 propose multi-way adversarial domain discriminator that minimizes the domain discrepancies between the target and each of the sources.

However, both methods BID19 BID33 assume that the source and target domains have a shared label space.

Transfer Learning.

Transfer learning extends domain adaptation into more general cases, where the source and target domain could be different in both input space and label space BID21 BID32 .

In computer vision, transfer learning has been widely studied to overcome the deficit of labeled data by adapting models trained for other tasks.

With the advance of deep supervised learning, ConvNets trained on large datasets such as ImageNet BID25 have achieved state-of-the-art performance when transfered to other tasks (e.g. object detection BID7 , semantic segmentation BID14 , image captioning BID3 , etc.) by simple fine-tuning.

In this work, we focus on the setting where source and target domains have the same input space and different label spaces.

Meta-Learning.

Meta-learning aims to utilize knowledge from past experiences to learn quickly on target tasks, from only a few annotated samples.

Meta-learning generally seeks performing the learning at a level higher than where conventional learning occurs, e.g. learning the update rule of a learner BID22 , or finding a good initialization point that can be easily fine-tuned BID4 .

Recently BID11 propose a meta-learning method to train models with good generalization ability to novel domains.

Our method follows the meta-learning paradigm that uses validation loss as the meta-objective.

Our method also resembles the example reweighting method by BID24 .

However, they reweight samples in a batch for robust learning against noise, whereas we reweight source domain gradients layer-wise for transfer learning.

We first formally introduce the semi-supervised MS-DTT problem.

We assume that there exists a set of k source domains S = {S 1 , S 2 , ..., S k }, and a target domain T .

Each source domain S i contains N Si images, x Si ∈ X Si , with associated labels y Si ∈ Y Si .

Similarly, the target domain consists of N T unlabeled images, x T ∈ X T , as well as M T labeled images, with associated labels y T ∈ Y T .

We assume that the target domain is only sparsely labeled, i.e. M T N T .

Our goal is to learn a strong target classifier that can predict labels y T given x T .Different from standard domain adaptation approaches that assume a shared label space between source and target domain (Y S = Y T ), we study the problem of joint transfer across domains and tasks.

Each source domain could have a partially overlapping label space with the target domain DISPLAYFORM0 .

However, we presume that at least one source domain should have the same label space as the target domain DISPLAYFORM1

Let Θ denote the network parameters for our model.

We consider a loss function L(x, y; Θ) = f (Θ) to minimize during training.

For deep networks, SGD or its variants are commonly used to optimize the loss functions.

At every step n of training, we forward a mini-batch of samples from each of the source domain DISPLAYFORM0 , and apply back-propagation to calculate the gradients w.r.t the parameters Θ n , ∇f si (Θ n ).

The parameters are then adjusted according to the sum of the source gradients.

For example, for vanilla SGD: DISPLAYFORM1 where α is the learning rate.

In semi-supervised MS-DTT, we also have a small validation set V that contains few labeled samples from the target domain.

We want to learn a set of weights for the source gradients, DISPLAYFORM2 , such that when taking a gradient descent using their weighted combination k i=1 w si ∇f si (Θ n ), the loss on the validation set is minimized: DISPLAYFORM3 DISPLAYFORM4

Calculating the optimal w * requires two nested loops of optimization, which can be computationally expensive.

Here we propose an approximation to the above objective.

At each training iteration n, we do a forward-backward pass using the small validation set V to calculate the gradient, ∇f V (Θ n ).

We take a first-order approximation and assume that adjusting Θ n in the direction of ∇f V (Θ n ) can minimize f V (Θ n ).

Therefore, we find the optimal w * by maximizing the cosine similarity between the combined source gradient and the validation gradient: DISPLAYFORM0 where a, b denotes the cosine similarity between vector a and b. This method is a cheap estimation for the meta-objective, which can also prevent the model from over-fitting to V.Instead of using a global weight value for each source gradient, we propose a layer-wise gradient weighting, where the gradient for each layer of the network are weighted separately.

This enables a finer level of gradient combination.

Specifically, in our MS-DTT setting, the source domains and the target domain share the same parameters up to the last fully-connected (fc) layer, which is task-specific.

Therefore, for each layer l with parameter θ l , and for each source domain S i , we have a corresponding weight w l si .

We can then write Equation 4 as: DISPLAYFORM1 where L is the total number of layers for the ConvNet.

We constrain w l s i ≥ 0 for all i and l, since negative gradient update can usually result in unstable behavior.

To efficiently solve the above constrained non-linear optimization problem, we utilize a sequential quadratic programming method, SLSQP, implemented in NLopt (Johnson).In practice, we normalize the weights for each layer across all source domains so that they sum up to one:w DISPLAYFORM2 Intuitively, certain mini-batches from the source domains contain more useful knowledge that can be transferred to the target domain, whereas some mini-batches contain less.

Therefore, we want to adaptively adjust our training to pay more attention to the important mini-batches.

To this end, we measure the importance score ρ of a mini-batch using the cosine similarity between the optimally combined gradient and the validation gradient: DISPLAYFORM3 Based on ρ, we calculate a scaling term η bounded between 0 and 1: DISPLAYFORM4 where β controls the rate of saturation for η, and γ defines the value of βρ where η = 0.5.

We determine the value of β and γ empirically through experiments.

Finally, we multiply η to the learning rate α, and perform SGD to update the parameters: DISPLAYFORM5 3.5 PSEUDO-LABEL WITH ENSEMBLESIn our semi-supervised MS-DTT setting, there also exists a large set of unlabeled images in the target domain, denoted as DISPLAYFORM6 .

We want to learn target-discriminative knowledge from U. To achieve this, we propose a method to calculated pseudo-labelsŷ T n for the unlabeled images, and construct a pseudo-labeled dataset DISPLAYFORM7 .

Then we leverage S u using the same gradient mixing method as described above.

Specifically, we consider a loss L u (x,ŷ; Θ) to minimize during training, where (x,ŷ) ∈ S u .

At each training iteration n, we sample a mini-batch from S u , calculate the gradient ∇f su (Θ n ), and combine it with the source gradients {∇f si (Θ n )} k i=1 using the proposed layer-wise weighting method.

In order to acquire the pseudo-labels, we perform a first step to train a model using the source domain datasets following the proposed gradient mixing method, and use the learned model to label U. However, the learned model would inevitably create some false pseudo-labels.

Previous studies found that ensemble of models helps to produce more reliable pseudo-labels BID26 BID9 BID29 .

Therefore, in our first step, we train multiple models with different combination of β and γ in Equation 8.

Then we pick the top R models with the best accuracies on the hyper-validation set (R = 3 in our experiments), and use their ensemble to create pseudo-labels.

The difference in hyper-parameters during training ensures that different models learn significantly different sets of weight, hence the ensemble of their prediction is less biased.

Here we propose two approaches to create pseudo-labels, namely hard label and soft label:Hard label.

In this approach, we assume that the pseudo-label is more likely to be correct if all the models can reach an agreement with high confidence.

We assign a pseudo-labelŷ = C to an image x ∈ U, where C is a class number, if the two following conditions are satisfied.

First, all of the R models should predict C as the class with the maximum probability.

Second, for all models, the probability for C should exceed certain threshold, which is set as 0.8 in our experiments.

If these two conditions are satisfied, we will add (x,ŷ) into S u .

During training, the loss L u (x,ŷ; Θ) is the standard cross entropy loss.

Soft label.

Let p r denote the output from the r-th model's softmax layer for an input x, which represents the probability over classes.

We calculate the average of p r across all of the R pre-trained models as the soft pseudo-label for x:ŷ = 1 R R r=1 p r .

Every unlabeled image x ∈ U will be assigned a soft label and added to S u .

During training, let p Θ be the output probability from the model, we want to minimize the KL-divergence between p Θ and the soft pseudo-label for all pairs (x,ŷ) ∈ S u .

Therefore, the loss is L u (x,ŷ; Θ) = D KL (p Θ ,ŷ).For both hard and soft label approach, after getting the pseudo-labels, we train a model from scratch using all available datasets {S i } k i=1 , S u and V. Since the proposed gradient mixing method relies on V to estimate the model's performance on the target domain, we enlarge the size of V to 100 samples per class, by adding hard-labeled images from S u using the method described above.

The enlarged V can represent the target domain with less bias, which helps to calculate better weights on the source gradients, such that the model's performance on the target domain is maximized.

Datasets.

In our experiment we perform MS-DTT across two different groups of data settings, as shown in FIG0 .

First we do transfer learning across different digit domains using MNIST BID10 and Street View House Numbers (SVHN) BID20 .

MNIST is the popular benchmark for handwritten digit recognition, which contains a training set of 60,000 examples, and a test set of 10,000 examples.

SVHN is a real-word dataset consisting of images with colored background and blurred digits.

It has 73,257 examples for training and 26,032 examples for testing.

For our second setup, we study MS-DTT from human activity images in MPII dataset BID0 and human action images from the Web (BU101 dataset) BID18 , to video action recognition using UCF101 BID27 dataset.

MPII dataset consists of 28,821 images covering 410 human activities including home activities, religious activities, occupation, etc.

UCF101 is a benchmark action recognition dataset collected from YouTube.

It consists of 13,320 videos from 101 action categories, captured under various lighting conditions with camera motion and occlusion.

We take the first split of UCF101 for our experiment.

BU101 contains 23,800 images collected from the Web, with the same action categories as UCF101.

It contains professional photos, commercial photos, and artistic photos, which can differ significantly from video frames.

Network and implementation details.

For our first setting, we use the same ConvNet architecture as BID17 , which has 4 Conv layers and 2 fc layers.

We randomly initialize the weights, and train the network using SGD with learning rate α = 0.05, and a momentum of 0.9.

For fine-tuning we reduce the learning rate to 0.005.

For our second setting, we use the ResNet-18 BID6 architecture.

We initialize the network with ImageNet pre-trained weights, which is important for all baseline methods to perform well.

The learning rate is 0.001 for training and 5e−5 for fine-tuning.

Experimental setting.

In this experiment, we define four sets of training data: (1) labeled images of digits 5-9 from the training split of SVHN dataset as the first source S 1 , (2) labeled images of digits 0-4 from the training split of MNIST dataset as the second source S 2 , (3) few labeled images of digits 5-9 from the training split of MNIST dataset as the validation set V, (4) unlabeled images from the rest of the training split of MNIST 5-9 as U. We subsample k examples from each class of MNIST 5-9 to construct the unbiased validation set V. We experiment with k = 2, 3, 4, 5, which corresponds to 10, 15, 20, 25 labeled examples.

Since V is randomly sampled, we repeat our experiment 10 times with different V. In order to monitor training progress and tune hyper-parameters (e.g. α, β, γ), we split out another 1000 labeled samples from MNIST 5-9 as the hyper-validation set.

The hyper-validation set is the traditional validation set, which is fixed across the 10 runs.

Baselines.

We compare the proposed method to multiple baseline methods: (1) Target only: the model is trained using V. (2) Source only: the model is trained using S 1 and S 2 without gradient reweighting.

(3) Fine-tune: the Source only model is fine-tuned using V. (4) MDDA BID19 : Multi-domain domain alignment layers that shift the network activations for each domain using a parameterized transformation equivalent to batch normalization.

(5) DCTN BID33 : Deep Cocktail Network, which uses multi-way adversarial adaptation to align the distribution of multiple source domains and the target domain.

We also evaluate different variants of our model with and without certain component to show its effect: (6) GradMix w/o AdaLR: the method in Section 3.3 without the adaptive learning rate (Section 3.4).

(7) GradMix: the proposed method that uses S 1 , S 2 and V during training. (8) GradMix w/ hard label: using the hard label approach to create pseudo-labels for U, and train a model with all available datasets.

(9) GradMix w/ soft label: using the soft label approach to create pseudo-labels for U, and train a model with all available datasets.

Results.

TAB0 shows the results for methods described above.

We report the mean and standard error of classification accuracy across 10 runs with randomly sampled V. Methods in the upper part of the table do not use the unlabeled target domain data U. Among these methods, the proposed GradMix has the best performance.

If we remove the adaptive learning rate, the accuracy would decrease.

As expected, the performance improves as k increases, which indicates more samples in Results of GradMix using different β and γ when k = 3.

Numbers indicate the test accuracy (%) on MNIST 5-9 (averaged across 10 runs).

The ensemble of the top three models is used to create pseudo-labels.

V can help the GradMix method to better combine the gradients during training.

The lower part of the table shows methods that use all available datasets including S 1 , S 2 , V and U. The proposed GradMix without U can achieve comparable performance with state-of-the-art baselines that use U (MDDA and DCTN).

Using pseudo-label with model ensemble significantly improves performance compared to baseline methods.

Comparing soft label to hard label, the hard label approach achieves better performance.

More detailed results about model ensemble for pseudo-labeling is shown later in the ablation study.

DISPLAYFORM0 Ablation Study.

In this section, we perform multiple ablation experiments to demonstrate the effectiveness of our method and the effect of different hyper-parameters.

First, FIG1 shows two examples of the hyper-validation loss as training proceeds.

We show the loss for the baseline Source only method and the proposed GradMix, where we perform hyper-validation every 100 mini-batches (gradient descents).

In both examples with different k, GradMix achieves a quicker and steadier decrease in the hyper-validation loss.

In TAB1 , we show the results using GradMix with different combination of β and γ when k = 3.

We perform a grid search with β = [5, 6, ..., 10] and γ = [0, 0.1, ..., 0.8].

The accuracy is the highest for β = 10 and γ = 0.6.

The top three models are selected for ensemble to create pseudo-labels for the unlabeled set U.In addition, we perform experiments with different number of models used for ensemble.

TAB3 shows the results for R = 1, 2, 3 across all values of k. R = 1 and R = 2 have comparable performance, whereas R = 3 performs better.

This indicates that using the top three models for ensemble can create more reliable pseudo-labels.

Experimental setting.

In the action recognition experiment, we have four sets of training data similar to the digit recognition experiment, which include (1) S 1 : labeled images from the training split of MPII, (2) S 2 : labeled images from the training split of BU101, (3) V: k labeled video clips per class randomly sampled from the training split of UCF101, (4) U: unlabeled images from the rest of the training split of UCF101.

We experiment with k = 3, 5, 10 which corresponds to 303, 505, 1010 video clips.

Each experiment is run two times with different V. We report the mean accuracy across the two runs for both per-frame classification and per-video classification.

Per-frame classification is the same as doing individual image classification for every frame in the video, and per-video classification is done by averaging the softmax score for all the frames in a video as the video's score.

Baselines.

We compare our method with multiple baselines described in Section 4.2, including (1) Target only, (2) Source only, (3) Fine-tune.

In addition, we evaluate another baseline for knowledge transfer in action recognition, namely (4) EnergyNet BID12 : The ConvNet (ResNet-18) is first trained on MPII and BU101, then knowledge is transfered to UCF101 through spatial attention maps using a Siamese Energy Network.

Results.

TAB4 shows the results for action recognition.

Target only has better performance compared to Source only even for k = 3, which indicates a strong distribution shift between source data and target data for actions in the wild.

For all values of k, the proposed GradMix outperforms baseline methods that use S 1 , S 2 and V for training in both per-frame and per-video accuracy.

GradMix also has comparable performance with MDDA that uses the unlabeled dataset U. The proposed pseudo-label method achieves significant gain in accuracy by assigning hard labels to U and learn target-discriminative knowledge from the pseudo-labeled dataset.

In this work, we propose GradMix, a method for semi-supervised MS-DTT: multi-source domain and task transfer.

GradMix assigns layer-wise weights to the gradients calculated from each source objective, in a way such that the combined gradient can optimize the target objective, measured by the loss on a small validation set.

GradMix can adaptively adjust the learning rate for each mini-batch based on its importance to the target task.

In addition, we assign pseudo-labels to the unlabeled samples using model ensembles, and consider the pseudo-labeled dataset as a source during training.

We validate the effectiveness our method with extensive experiments on two MS-DTT settings, namely digit recognition and action recognition.

GradMix is a generic framework applicable to any models trained with gradient descent.

For future work, we intend to extend GradMix to other problems where labeled data for the target task is expensive to acquire, such as image captioning.

<|TLDR|>

@highlight

We propose a gradient-based method to transfer knowledge from multiple sources across different domains and tasks.

@highlight

This paper proposes to combine the gradients of source domains to help the learning in the target domain. 