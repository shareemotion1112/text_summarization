Learning with a primary objective, such as softmax cross entropy for classification and sequence generation, has been the norm for training deep neural networks for years.

Although being a widely-adopted approach, using cross entropy as the primary objective exploits mostly the information from the ground-truth class for maximizing data likelihood, and largely ignores information from the complement (incorrect) classes.

We argue that, in addition to the primary objective, training also using a complement objective that leverages information from the complement classes can be effective in improving model performance.

This motivates us to study a new training paradigm that maximizes the likelihood of the ground-truth class while neutralizing the probabilities of the complement classes.

We conduct extensive experiments on multiple tasks ranging from computer vision to natural language understanding.

The experimental results confirm that, compared to the conventional training with just one primary objective, training also with the complement objective further improves the performance of the state-of-the-art models across all tasks.

In addition to the accuracy improvement, we also show that models trained with both primary and complement objectives are more robust to single-step adversarial attacks.

Statistical learning algorithms work by optimizing towards a training objective.

A dominant principle for training is to optimize likelihood BID16 , which measures the probability of data given the model under a specific set of parameters.

The popularity of deep neural networks has given rise to the use of cross entropy BID13 as its primary training objective, since minimizing cross entropy is essentially equivalent to maximizing likelihood for disjoint classes.

Cross entropy has become the standard training objective for many tasks including classification BID12 and sequence generation .Let y i ∈ {0, 1} K be the label of the i th sample in one-hot encoded representation andŷ i ∈ [0, 1] K be the predicted probabilities, the cross entropy H(y,ŷ) is defined as: DISPLAYFORM0 whereŷ ig represents the predicted probability of the ground-truth class for the i th sample.

Training with cross entropy as the primary objective aims at findingθ = arg min θ H(y,ŷ), wherê y = h θ (x), h θ is a neural network and x is a sample.

Although training using the cross entropy as The model is ResNet-110, and the "embedding" is the vector representation before taking the softmax operation.

The embedding representation of each sample is projected to two dimensions using t-SNE for visualization purpose.

Compared to (a), the cluster of each class in (b) is "narrower" in terms of intra-cluster distance.

Also, the clusters in (b) seem to have clean and separable boundaries, leading to more accurate and robust classification results.the primary objective has achieved tremendous success, we have observed one limitation: it exploits mostly the information from the ground-truth class as Eq(1) shows; the information from complement classes (i.e., incorrect classes) has been largely ignored, since the predicted probabilities other thanŷ ig are zeroed out due to the dot product calculation with the one-hot encoded y i .

Therefore, for classes other than the ground truth, the model behavior is not explicitly optimized -their predicted probabilities are indirectly minimized whenŷ ig is maximized since the probabilities sum up to 1.

One way to utilize the information from the complement classes is to neutralize their predicted probabilities.

To this end, we propose Complement Objective Training (COT), a new training paradigm that achieves this optimization goal without compromising the model's primary objective.

FIG1 illustrates the comparison between FIG1 : the predicted probabilityŷ from the model trained with just cross entropy as the primary objective, and FIG1 :ŷ from the model trained with both primary and complement objectives.

Training with the complement objective finds the parameters θ that evenly suppress complement classes without compromising the primary objective (i.e., maximizingŷ g ), making the model more confident of the ground-truth class.

Complement objective training requires a function that complements the primary objective.

In this paper, we propose "complement entropy" (defined in Section 2) to complement the softmax cross entropy for neutralizing the effects of complement classes.

The neural net parameters θ are then updated by alternating iteratively between (a) minimizing cross entropy to increaseŷ g , and (b) maximizing complement entropy to neutralizeŷ j =g .

Experimental results (in Section 3) confirm that COT improves the accuracies of the state-of-the-art methods for both (a) the image classification tasks on ImageNet-2012, Tiny ImageNet, CIFAR-10, CIFAR-100, and SVHN, and (b) language understanding tasks on machine translation and speech recognition.

Furthermore, experimental results also show that models trained by COT are more robust to adversarial attacks.

In this section, we first define "Complement Entropy" as the complement objective, and then provide a new training algorithm for updating neural network parameters θ by alternating iteratively between the primary objective and the complement objective.

Conventionally, training with cross entropy as the primary objective aims at maximizing the predicted probability of the ground-truth classŷ g in Eq(1).

As mentioned in the introduction, the proposed COT also maximizes the complement objective for neutralizing the predicted probabilities of the complement classes.

To achieve this, we propose "complement entropy" as the complement objective; complement entropy C(·) is defined to be the average of sample-wise entropies over complement classes in a mini-batch: DISPLAYFORM0 H(·) is the entropy function.

All the symbols and notations used in this paper are summarized in TAB0 .

One thing worth noticing is that this sample-wise entropy is calculated by considering only the complement classes other than the ground-truth class g. The sample-wise predicted probabilitŷ y ij is normalized by one minus the ground-truth probability (i.e., 1−ŷ ig ).

The termŷ ij /(1−ŷ ig ) can be understood as: conditioned on the ground-truth class g not happening, the predicted probability to see the class j for the i th sample.

Since the entropy is maximized when the events are equally likely to occur, optimizing on the complement entropy drivesŷ ij to (1 −ŷ ig )/(K − 1), which essentially neutralizes the predicted probability of complement classes as K grows large.

In other words, maximizing the complement entropy "flattens" the predicted probabilities of complement classeŝ y j =g .

We conjecture that, whenŷ j =g are neutralized, the neural net h θ generalizes better, since it is less likely to have an incorrect class with a sufficiently high predicted probability to "challenge" the ground-truth class.

Given a training procedure using a primary objective, such as softmax cross entropy, one can easily adopt the complement entropy to turn the procedure into a Complement Objective Training (COT).

Algorithm 1 describes the new training mechanism by alternating iteratively between the primary and complement objectives.

At each training step, the cross entropy is first calculated as the loss value to update the model parameters; next, the complement entropy is calculated as the loss value to perform the second update.

Therefore, additional forward and backward propagation are required in each iteration when using the complement objective, making the total training time empirically 1.6 times longer.

One-hot vector representing the label of the i th sample.

DISPLAYFORM0 The predicted probability for each class for the i th sample.

g Index of the ground-truth class.

y ij orŷ ij The j th class (element) of y i orŷ i .

yc Predicted probabilities of of the complement (incorrect) classes.

DISPLAYFORM1 Cross entropy function.

Entropy function.

Complement entropy.

N and K Total number of samples and total number of classes.

Algorithm 1: Training by alternating between primary and complement objectives 1 for t ← 1 to n train steps do 1.

Update parameters by Primary Objective: DISPLAYFORM0

We perform extensive experiments to evaluate COT on tasks in domains ranging from computer vision to natural language understanding and compare it with the baseline algorithms that achieve state-of-the-art in the respective domains.

We also perform experiments to evaluate the robustness of the model trained by COT when attacked by adversarial examples.

For each task, we select a stateof-the-art model that has an open-source implementation (referred to as "baseline") and reproduce their results with the hyper-parameters reported in the paper or code repository.

Our code is available at https://github.com/henry8527/COT.

In theory, the loss values between the primary and the complement objectives can be in different scales; therefore, additional efforts for tuning learning rates might be required for optimizers to achieve the best performance.

Empirically, we find the complement entropy in Eq(2) can be modified as follows to balance the losses between the two objectives: DISPLAYFORM0 where K is the number of classes.

This modification can be treated as the complement entropy C(·) being "normalized" by (K − 1).

For all the experiments conducted in this paper, we use this normalized complement entropy as the complement objective to improve the baselines without further tuning of learning rates.

We consider the following datasets for experiments with image classification: CIFAR-10, CIFAR-100, SVHN, Tiny ImageNet and ImageNet-2012.

For CIFAR-10, CIFAR-100 and SVHN, we choose the following baseline models: ResNet-110 BID7 , PreAct ResNet-18 BID6 , ResNeXt-29 (2×64d) BID25 , WideResNet-28-10 (Zagoruyko & Komodakis, 2016) and DenseNet-BC-121 BID9 ) with a growth rate of 32.

For those five models, we use a consistent set of settings below, which is described in BID7 .

Specifically, the models are trained using SGD optimizer with momentum of 0.9.

Weight decay is set to be 0.0001 and learning rate starts at 0.1, then being divided by 10 at the 100 th and 150 th epoch.

The models are trained for 200 epochs, with mini-batches of size 128.

The only exception here is for training WideResNet-28-10, we follow the settings described in BID26 , and the learning rate is divided by 10 at the 60 th , 120 th and 180 th epoch.

In addition, no dropout BID21 ) is applied to any baseline according to the best practices in BID10 .

For Tiny ImageNet and ImageNet-2012, the baseline models are slightly different: we follow the settings from BID27 , and the details are described in the corresponding paragraphs.

CIFAR-10 and CIFAR-100.

CIFAR-10 and CIFAR-100 are datasets BID11 ) that contain colored natural images of 32x32 pixels, in 10 and 100 classes, respectively.

We follow the baseline settings BID7 to pre-process the datasets; both datasets are split into a training set with 50,000 samples and a testing set with 10,000 samples.

During training, zero-padding, random cropping, and horizontal mirroring are applied to the images with a probability of 0.5.

For the testing images, we use the original images of 32x32 pixels.

A comparison between the models trained using the primary objective and the COT model is illustrated in FIG4 for CIFAR-10 and CIFAR-100 respectively.

We show that COT consistently outperforms the baseline models.

Some of the models, for example, ResNetXt-29, achieves a significant performance boost of 12.5% in terms of classification errors.

For some other models such as WideResNet-28-10 and DenseNet-BC-121, the improvements are not as significant but are still large enough to justify the differences.

Similar conclusions can be observed from the CIFAR-100 dataset.

In addition to the comparisons of the performance, we also present the change of testing errors over the course of the training in FIG4 for the ResNet-110 model.

Following the standard training practice, learning rates drop after the 100 th epoch, which corresponds to a drop in testing errors.

As we can see from the plot, COT outperforms consistently compared to the baseline models when the models are close to the convergence.

The SVHN dataset BID17 consists of images extracted from Google Street View.

We divide the dataset into a set of 73,257 digits for training and a set of 26,032 digits for testing.

When pre-processing the training and validation images, we follow the general practice to normalize pixel values into [-1,1].

TAB3 shows the experimental results and confirms that COT consistently improves the baseline models with the biggest improvement being the ResNet-110 with 11.7% reduction on the error rate.

Baseline COT The improvement over epochs.

Notice that the performance improvement from COT becomes stable after the 100 th epoch due to the learning rate decrease.

The improvement over epochs.

Similar to the trend observed in CIFAR-10, the performance improvement from COT becomes stable after the 100 th epoch due to the learning rate decrease.

Tiny ImageNet.

Tiny ImageNet 1 dataset is a subset of ImageNet BID2 , which contains 100,000 images for training and 10,000 for testing images across 200 classes.

In this dataset, each image is down-sampled to 64x64 pixels from the original 256x256 pixels.

We consider four state-of-the-art models as baselines: ResNet-50, ResNet-101 BID7 , and ResNeXt-101 (32×4d) BID25 .

During training, we follow the standard data-augmentation techniques, such as random cropping, horizontal flipping, and normalization.

For each model, the stride of the first convolution layer is modified to adapt images of size 64x64 BID8 .

For evaluation, the testing data is only augmented with 56x56 central cropping.

The rest of the experimental details are the same as the ones described at the beginning of Section 3.2.

TAB4 provides the experimental results, which demonstrate that COT consistently improves the performance of all baseline models. (Russakovsky et al., 2015) is one of the largest datasets for image classification, which contains 1.3 million images for training and 50,000 images for testing with 1,000 classes.

Random crops and horizontal flips are applied during training BID7 , while images in the testing set use 224x224 center crops (1-crop testing) for data augmentation.

ResNet-50 is selected as the baseline model, and we follow BID5 for the experimental setup: 256 minibatch size, 90 total training epochs, and 0.1 as the initial learning rate starting that is decayed by dividing 10 at the 30 th , 60 th and 80 th epoch.

TAB5 shows (a) the error rate 2 of baseline reported by BID7 and (b) the error rate of baseline model trained by COT, which confirms COT further improves the baseline performance.

Baseline COTResNet-50 Top-1 Error 24.7 24.4

COT is also evaluated on two natural language understanding (NLU) tasks: machine translation and speech recognition.

One distinct characteristic of most NLU tasks is a large number of target classes.

For example, the machine translation dataset used in this paper, IWSLT 2015 English-Vietnamese BID0 , consists of vocabularies of 17,191 English words and 7,709 Vietnamese words.

This necessitates the normalized complement entropy in Eq(3).Machine translation.

Neural machine translation (NMT) has popularized the use of neural sequence models BID1 .

Specifically, we apply COT on the seq2seq model with Luong attention mechanism BID15 on the IWSLT 2015 EnglishVietnamese dataset, which contains 133 thousand translation pairs.

For validation and testing, we use TED tst2012 and TED tst2013, respectively.

For the baseline implementation, we follow the official TensorFlow-NMT implementation 3 .

That is, the number of total training steps is 12,000 and the weight decay starts at the 8,000th step then applied for every 1,000 steps.

We experiment models with both greedy decoder and beam search decoder.

The model trained by COT gives the best testing results when the beam width is 3, while the baseline uses 10 as the best beam width.

TAB6 illustrates the experimental results, showing COT improves testing BLEU scores compared to the baseline NMT model on both greedy decoder and the beam search decoder.

den, 2018) , which consists of 65,000 one-second utterances of 30 different types such as "Yes," "No," "Up," "Down" and "Stop."

Our baseline model is referenced from BID27 .

We apply the same pre-processing steps as shown in the paper, and perform the short-time Fourier transform on the original waveforms first at a sampling rate of 4 kHz to receive the corresponding spectrograms.

We then zero-pad these spectrograms to equalize each sample's length.

For the baseline model, we select VGG-11 BID20 ) and train the model for 30 epochs following the steps in BID27 .

We use SGD optimizer with momentum, and weight decay is 0.0001.

The learning rate starts at 0.0001 and then is divided by 10 at the 10 th and 20 th epoch.

COT improves the baseline by further reducing the error rate by 1.56%, as shown in TAB7 .

An adversarial example is an imperceptibly-perturbed input that results in the model outputting an incorrect answer with high confidence BID23 BID4 .

Prior As shown in FIG2 , the proposed COT generates embeddings where the class boundaries are clear and well-separated.

We believe that the models trained using COT generalize better and are more robust to adversarial attacks.

To verify this conjecture, we conduct experiments of white-box attacks to the models trained by COT.

We consider a common approach of single-step adversarial attacks: Fast Gradient Sign Method (FGSM) BID4 that uses the gradient to determine the direction of the perturbation to apply on an input for creating an adversarial example.

To set up FGSM white-box attacks on a baseline model, adversarial examples are generated using the gradients calculated based on the primary objective (referred to as the "primary gradient") of the baseline model.

For FGSM white-box attacks on COT, adversarial perturbations are generated based on the sum of the primary gradient and the complement gradient (i.e., the gradient calculated from the complement objective), both gradients from the model trained by COT.

In our experiments, the baseline models are the same as in Section 3.2, and the amount of perturbation is limited to a maximum value of 0.1 as described in BID4 when creating adversarial examples.

Furthermore, we also conduct experiments on FGSM transfer attacks, which use the adversarial examples from a baseline model to attack and test the robustness of the model trained by COT.

TAB8 shows the performance of the models on the CIFAR-10 dataset under FGSM white-box and transfer attacks.

Generally, the models trained using COT have lower classification error under both FGSM white-box and transfer attacks, which is an indicator that COT models are more robust to both kinds of attacks.

We also conduct experiments on the basic iterative attacks using I-FGSM BID14 and the corresponding results can be found in Appendix A.We conjecture that since the main goal of the complement gradients is to neutralize the probabilities of incorrect classes (instead of maximizing the probability of the correct class), the complement gradients may "push away" primary gradients when forming adversarial perturbations, which might partially answer why COT is more robust to FGSM white-box attacks compared to the baseline.

Regarding the transfer attacks, only the primary objective of the baseline model is used to calculate the gradients for generating adversarial examples.

In other words, the complement gradients are not considered when generating adversarial examples in the transfer attack, and this might be the reason why models trained by COT are more robust to transfer attacks.

Both conjectures leave a large space for future work: using complement objective to defend against more advanced adversarial attacks.

In this paper, we study Complement Objective Training (COT), a new training paradigm that optimizes the complement objective in addition to the primary objective.

We propose complement entropy as the complement objective for neutralizing the effects of complement (incorrect) classes.

Models trained using COT demonstrate superior performance compared to the baseline models.

We also find that COT makes the models robust to single-step adversarial attacks.

COT can be extended in several ways: first, in this paper, the complement objective is chosen to be the complement entropy.

Non-entropy-based complement objectives should also be considered for future studies, which is left as a straight-line future work.

Secondly, the exploration of COT on broader applications remains as an open research question.

One example would be applying COT on generative models such as Generative Adversarial Networks .

Another example would be using COT on object detection and segmentation.

Finally, in this work, we show using complement objective help defend single-step adversarial attacks; the behavior of COT on more advanced adversarial attacks deserves further investigation and is left as another future work.

A ITERATIVE FAST GRADIENT SIGN METHOD TAB9 shows the performance of the models on the CIFAR-10 dataset under I-FGSM transfer attacks.

Generally, the models trained using COT have lower classification error under I-FGSM transfer attacks.

The number of iteration is set to 10 in the experiment.

<|TLDR|>

@highlight

We propose Complement Objective Training (COT), a new training paradigm that optimizes both the primary and complement objectives for effectively learning the parameters of neural networks.

@highlight

Considers augmenting the cross-entropy objective with "complement" objective maximization, which aims at neutralizing the predicted probabilities of classes other than the ground truth labels.

@highlight

The authors propose a secondary objective for softmax minimization based on evaluating the information gathered from the incorrect classes, leading to a new training approach.

@highlight

Deals with the training of neural networks for classification or sequence generation tasks using across-entropy loss