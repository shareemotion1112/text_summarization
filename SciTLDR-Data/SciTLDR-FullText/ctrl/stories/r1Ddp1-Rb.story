Large deep neural networks are powerful, but exhibit undesirable behaviors such as memorization and sensitivity to adversarial examples.

In this work, we propose mixup, a simple learning principle to alleviate these issues.

In essence, mixup trains a neural network on convex combinations of pairs of examples and their labels.

By doing so, mixup regularizes the neural network to favor simple linear behavior in-between training examples.

Our experiments on the ImageNet-2012, CIFAR-10, CIFAR-100, Google commands and UCI datasets show that mixup improves the generalization of state-of-the-art neural network architectures.

We also find that mixup reduces the memorization of corrupt labels, increases the robustness to adversarial examples, and stabilizes the training of generative adversarial networks.

Large deep neural networks have enabled breakthroughs in fields such as computer vision BID22 , speech recognition , and reinforcement learning BID28 .

In most successful applications, these neural networks share two commonalities.

First, they are trained as to minimize their average error over the training data, a learning rule also known as the Empirical Risk Minimization (ERM) principle BID35 .

Second, the size of these state-of-theart neural networks scales linearly with the number of training examples.

For instance, the network of BID31 used 10 6 parameters to model the 5 · 10 4 images in the CIFAR-10 dataset, the network of BID30 Strikingly, a classical result in learning theory BID36 tells us that the convergence of ERM is guaranteed as long as the size of the learning machine (e.g., the neural network) does not increase with the number of training data.

Here, the size of a learning machine is measured in terms of its number of parameters or, relatedly, its VC-complexity BID16 .This contradiction challenges the suitability of ERM to train our current neural network models, as highlighted in recent research.

On the one hand, ERM allows large neural networks to memorize (instead of generalize from) the training data even in the presence of strong regularization, or in classification problems where the labels are assigned at random .

On the other hand, neural networks trained with ERM change their predictions drastically when evaluated on examples just outside the training distribution BID33 , also known as adversarial examples.

This evidence suggests that ERM is unable to explain or provide generalization on testing distributions that differ only slightly from the training data.

However, what is the alternative to ERM?The method of choice to train on similar but different examples to the training data is known as data augmentation BID29 , formalized by the Vicinal Risk Minimization (VRM) principle BID3 .

In VRM, human knowledge is required to describe a vicinity or neighborhood around each example in the training data.

Then, additional virtual examples can be drawn from the vicinity distribution of the training examples to enlarge the support of the training distribution.

For instance, when performing image classification, it is common to define the vicinity of one image as the set of its horizontal reflections, slight rotations, and mild scalings.

While data augmentation consistently leads to improved generalization BID29 , the procedure is dataset-dependent, and thus requires the use of expert knowledge.

Furthermore, data augmentation assumes that the examples in the vicinity share the same class, and does not model the vicinity relation across examples of different classes.

Contribution Motivated by these issues, we introduce a simple and data-agnostic data augmentation routine, termed mixup (Section 2).

In a nutshell, mixup constructs virtual training examples DISPLAYFORM0 where x i , x j are raw input vectors y = λy i + (1 − λ)y j , where y i , y j are one-hot label encodings (x i , y i ) and (x j , y j ) are two examples drawn at random from our training data, and λ ∈ [0, 1].

Therefore, mixup extends the training distribution by incorporating the prior knowledge that linear interpolations of feature vectors should lead to linear interpolations of the associated targets.

mixup can be implemented in a few lines of code, and introduces minimal computation overhead.

Despite its simplicity, mixup allows a new state-of-the-art performance in the CIFAR-10, CIFAR-100, and ImageNet-2012 image classification datasets (Sections 3.1 and 3.2).

Furthermore, mixup increases the robustness of neural networks when learning from corrupt labels (Section 3.4), or facing adversarial examples (Section 3.5).

Finally, mixup improves generalization on speech (Sections 3.3) and tabular (Section 3.6) data, and can be used to stabilize the training of GANs (Section 3.7).

The source-code necessary to replicate our CIFAR-10 experiments is available at:https://github.com/facebookresearch/mixup-cifar10.To understand the effects of various design choices in mixup, we conduct a thorough set of ablation study experiments (Section 3.8).

The results suggest that mixup performs significantly better than related methods in previous work, and each of the design choices contributes to the final performance.

We conclude by exploring the connections to prior work (Section 4), as well as offering some points for discussion (Section 5).

In supervised learning, we are interested in finding a function f ∈ F that describes the relationship between a random feature vector X and a random target vector Y , which follow the joint distribution P (X, Y ).

To this end, we first define a loss function that penalizes the differences between predictions f (x) and actual targets y, for examples (x, y) ∼ P .

Then, we minimize the average of the loss function over the data distribution P , also known as the expected risk: DISPLAYFORM0 Unfortunately, the distribution P is unknown in most practical situations.

Instead, we usually have access to a set of training data DISPLAYFORM1 , where (x i , y i ) ∼ P for all i = 1, . . .

, n. Using the training data D, we may approximate P by the empirical distribution DISPLAYFORM2 where δ(x = x i , y = y i ) is a Dirac mass centered at (x i , y i ).

Using the empirical distribution P δ , we can now approximate the expected risk by the empirical risk: DISPLAYFORM3 Learning the function f by minimizing (1) is known as the Empirical Risk Minimization (ERM) principle BID35 .

While efficient to compute, the empirical risk (1) monitors the behaviour of f only at a finite set of n examples.

When considering functions with a number parameters comparable to n (such as large neural networks), one trivial way to minimize (1) is to memorize the training data .

Memorization, in turn, leads to the undesirable behaviour of f outside the training data BID33 .

However, the naïve estimate P δ is one out of many possible choices to approximate the true distribution P .

For instance, in the Vicinal Risk Minimization (VRM) principle BID3 , the distribution P is approximated by DISPLAYFORM4 where ν is a vicinity distribution that measures the probability of finding the virtual feature-target pair (x,ỹ) in the vicinity of the training feature-target pair (x i , y i ).

In particular, BID3 considered Gaussian vicinities ν(x,ỹ|x i , DISPLAYFORM5 , which is equivalent to augmenting the training data with additive Gaussian noise.

To learn using VRM, we sample the vicinal distribution to construct a dataset DISPLAYFORM6 , and minimize the empirical vicinal risk: DISPLAYFORM7 The contribution of this paper is to propose a generic vicinal distribution, called mixup: DISPLAYFORM8 where λ ∼ Beta(α, α), for α ∈ (0, ∞).

In a nutshell, sampling from the mixup vicinal distribution produces virtual feature-target vectorsx DISPLAYFORM9 where (x i , y i ) and (x j , y j ) are two feature-target vectors drawn at random from the training data, and λ ∈ [0, 1].

The mixup hyper-parameter α controls the strength of interpolation between feature-target pairs, recovering the ERM principle as α → 0.The implementation of mixup training is straightforward, and introduces a minimal computation overhead.

FIG2 shows the few lines of code necessary to implement mixup training in PyTorch.

Finally, we mention alternative design choices.

First, in preliminary experiments we find that convex combinations of three or more examples with weights sampled from a Dirichlet distribution does not provide further gain, but increases the computation cost of mixup.

Second, our current implementation uses a single data loader to obtain one minibatch, and then mixup is applied to the same minibatch after random shuffling.

We found this strategy works equally well, while reducing I/O requirements.

Third, interpolating only between inputs with equal label did not lead to the performance gains of mixup discussed in the sequel.

More empirical comparison can be found in Section 3.8.What is mixup doing?

The mixup vicinal distribution can be understood as a form of data augmentation that encourages the model f to behave linearly in-between training examples.

We argue that this linear behaviour reduces the amount of undesirable oscillations when predicting outside the training examples.

Also, linearity is a good inductive bias from the perspective of Occam's razor, since it is one of the simplest possible behaviors.

FIG2 shows that mixup leads to decision boundaries that transition linearly from class to class, providing a smoother estimate of uncertainty.

FIG4 illustrate the average behaviors of two neural network models trained on the CIFAR-10 dataset using ERM and mixup.

Both models have the same architecture, are trained with the same procedure, and are evaluated at the same points in-between randomly sampled training data.

The model trained with mixup is more stable in terms of model predictions and gradient norms in-between training samples.

We evaluate mixup on the ImageNet-2012 classification dataset BID27 .

This dataset contains 1.3 million training images and 50,000 validation images, from a total of 1,000 classes.

For training, we follow standard data augmentation practices: scale and aspect ratio distortions, random crops, and horizontal flips BID13 .

During evaluation, only the 224 × 224 central crop of each image is tested.

We use mixup and ERM to train several state-of-the-art ImageNet-2012 classification models, and report both top-1 and top-5 error rates in For all the experiments in this section, we use data-parallel distributed training in Caffe2 1 with a minibatch size of 1,024.

We use the learning rate schedule described in BID13 .

Specifically, the learning rate is increased linearly from 0.1 to 0.4 during the first 5 epochs, and it is then divided by 10 after 30, 60 and 80 epochs when training for 90 epochs; or after 60, 120 and 180 epochs when training for 200 epochs.

For mixup, we find that α ∈ [0.1, 0.4] leads to improved performance over ERM, whereas for large α, mixup leads to underfitting.

We also find that models with higher capacities and/or longer training runs are the ones to benefit the most from mixup.

For example, when trained for 90 epochs, the mixup variants of ResNet-101 and ResNeXt-101 obtain a greater improvement (0.5% to 0.6%) over their ERM analogues than the gain of smaller models such as ResNet-50 (0.2%).

When trained for 200 epochs, the top-1 error of the mixup variant of ResNet-50 is further reduced by 1.2% compared to the 90 epoch run, whereas its ERM analogue stays the same.

We conduct additional image classification experiments on the CIFAR-10 and CIFAR-100 datasets to further evaluate the generalization performance of mixup.

In particular, we compare ERM and mixup training for: PreAct ResNet-18 as implemented in BID25 , WideResNet-28-10 (Zagoruyko & Komodakis, 2016a) as implemented in BID40 , and DenseNet BID20 as implemented in BID37 .

For DenseNet, we change the growth rate to 40 to follow the DenseNet-BC-190 specification from BID20 .

For mixup, we fix α = 1, which results in interpolations λ uniformly distributed between zero and one.

All models are trained on a single Nvidia Tesla P100 GPU using PyTorch 2 for 200 epochs on the training set with 128 examples per minibatch, and evaluated on the test set.

Learning rates start at 0.1 and are divided by 10 after 100 and 150 epochs for all models except WideResNet.

For WideResNet, we follow BID39 and divide the learning rate by 10 after 60, 120 and 180 epochs.

Weight decay is set to 10 .

We do not use dropout in these experiments.

We summarize our results in FIG5 .

In both CIFAR-10 and CIFAR-100 classification problems, the models trained using mixup significantly outperform their analogues trained with ERM.

As seen in FIG5 , mixup and ERM converge at a similar speed to their best test errors.

Note that the DenseNet models in BID20 were trained for 300 epochs with further learning rate decays scheduled at the 150 and 225 epochs, which may explain the discrepancy the performance of DenseNet reported in FIG5 and the original result of BID20 .

extract normalized spectrograms from the original waveforms at a sampling rate of 16 kHz.

Next, we zero-pad the spectrograms to equalize their sizes at 160 × 101.

For speech data, it is reasonable to apply mixup both at the waveform and spectrogram levels.

Here, we apply mixup at the spectrogram level just before feeding the data to the network.

For this experiment, we compare a LeNet BID23 ) and a VGG-11 BID30 architecture, each of them composed by two convolutional and two fully-connected layers.

We train each model for 30 epochs with minibatches of 100 examples, using Adam as the optimizer BID21 .

Training starts with a learning rate equal to 3 × 10 DISPLAYFORM0 and is divided by 10 every 10 epochs.

For mixup, we use a warm-up period of five epochs where we train the network on original training examples, since we find it speeds up initial convergence.

TAB6 shows that mixup outperforms ERM on this task, specially when using VGG-11, the model with larger capacity.

Following , we evaluate the robustness of ERM and mixup models against randomly corrupted labels.

We hypothesize that increasing the strength of mixup interpolation α should generate virtual examples further from the training examples, making memorization more difficult to achieve.

In particular, it should be easier to learn interpolations between real examples compared to memorizing interpolations involving random labels.

We adapt an open-source implementation BID42 to generate three CIFAR-10 training sets, where 20%, 50%, or 80% of the labels are replaced by random noise, respectively.

All the test labels are kept intact for evaluation.

Dropout BID32 ) is considered the state-of-the-art method for learning with corrupted labels BID1 .

Thus, we compare in these experiments mixup, dropout, mixup + dropout, and ERM.

For mixup, we choose α ∈ {1, 2, 8, 32}; for dropout, we add one dropout layer in each PreAct block after the ReLU activation layer between two convolution layers, as suggested in BID39 .

We choose the dropout probability p ∈ {0.5, 0.7, 0.8, 0.9}. For the combination of mixup and dropout, we choose α ∈ {1, 2, 4, 8} and p ∈ {0.3, 0.5, 0.7}. These experiments use the PreAct ResNet-18 model implemented in BID25 .

All the other settings are the same as in Section 3.2.We summarize our results in TAB3 , where we note the best test error achieved during the training session, as well as the final test error after 200 epochs.

To quantify the amount of memorization, we also evaluate the training errors at the last epoch on real labels and corrupted labels.

As the training progresses with a smaller learning rate (e.g. less than 0.01), the ERM model starts to overfit the corrupted labels.

When using a large probability (e.g. 0.7 or 0.8), dropout can effectively reduce overfitting.

mixup with a large α (e.g. 8 or 32) outperforms dropout on both the best and last epoch test errors, and achieves lower training error on real labels while remaining resistant to noisy labels.

Interestingly, mixup + dropout performs the best of all, showing that the two methods are compatible.

One undesirable consequence of models trained using ERM is their fragility to adversarial examples BID33 .

Adversarial examples are obtained by adding tiny (visually imperceptible) perturbations to legitimate examples in order to deteriorate the performance of the model.

The adversarial noise is generated by ascending the gradient of the loss surface with respect to the legitimate example.

Improving the robustness to adversarial examples is a topic of active research.

Among the several methods aiming to solve this problem, some have proposed to penalize the norm of the Jacobian of the model to control its Lipschitz constant BID9 BID6 BID2 BID18 .

Other approaches perform data augmentation by producing and training on adversarial examples BID12 .

Unfortunately, all of these methods add significant computational overhead to ERM.

Here, we show that mixup can significantly improve the robustness of neural networks without hindering the speed of ERM by penalizing the norm of the gradient of the loss w.r.t a given input along the most plausible directions (e.g. the directions to other training points).

Indeed, FIG4 shows that mixup results in models having a smaller loss and gradient norm between examples compared to vanilla ERM.To assess the robustness of mixup models to adversarial examples, we use three ResNet-101 models: two of them trained using ERM on ImageNet-2012, and the third trained using mixup.

In the first set of experiments, we study the robustness of one ERM model and the mixup model against white box attacks.

That is, for each of the two models, we use the model itself to generate adversarial examples, either using the Fast Gradient Sign Method (FGSM) or the Iterative FGSM (I-FGSM) methods BID12 , allowing a maximum perturbation of = 4 for every pixel.

For I-FGSM, we use 10 iterations with equal step size.

In the second set of experiments, we evaluate robustness against black box attacks.

That is, we use the first ERM model to produce adversarial examples using FGSM and I-FGSM.

Then, we test the robustness of the second ERM model and the mixup model to these examples.

The results of both settings are summarized in TAB4 .For the FGSM white box attack, the mixup model is 2.7 times more robust than the ERM model in terms of Top-1 error.

For the FGSM black box attack, the mixup model is 1.25 times more robust than the ERM model in terms of Top-1 error.

Also, while both mixup and ERM are not robust to white box I-FGSM attacks, mixup is about 40% more robust than ERM in the black box I-FGSM setting.

Overall, mixup produces neural networks that are significantly more robust than ERM against adversarial examples in white box and black settings without additional overhead compared to ERM.

ERM GAN mixup GAN (α = 0.2) Figure 5 : Effect of mixup on stabilizing GAN training at iterations 10, 100, 1000, 10000, and 20000.

To further explore the performance of mixup on non-image data, we performed a series of experiments on six arbitrary classification problems drawn from the UCI dataset BID24 .

The neural networks in this section are fully-connected, and have two hidden layers of 128 ReLU units.

The parameters of these neural networks are learned using Adam BID21 with default hyper-parameters, over 10 epochs of mini-batches of size 16.

TAB6 shows that mixup improves the average test error on four out of the six considered datasets, and never underperforms ERM.

Generative Adversarial Networks, also known as GANs , are a powerful family of implicit generative models.

In GANs, a generator and a discriminator compete against each other to model a distribution P .

On the one hand, the generator g competes to transform noise vectors z ∼ Q into fake samples g(z) that resemble real samples x ∼ P .

On the other hand, the discriminator competes to distinguish between real samples x and fake samples g(z).

Mathematically, training a GAN is equivalent to solving the optimization problem DISPLAYFORM0 where is the binary cross entropy loss.

Unfortunately, solving the previous min-max equation is a notoriously difficult optimization problem BID10 , since the discriminator often provides the generator with vanishing gradients.

We argue that mixup should stabilize GAN training because it acts as a regularizer on the gradients of the discriminator, akin to the binary classifier in FIG2 .

Then, the smoothness of the discriminator guarantees a stable source of gradient information to the generator.

The mixup formulation of GANs is: DISPLAYFORM1 ), λ).

Figure 5 illustrates the stabilizing effect of mixup the training of GAN (orange samples) when modeling two toy datasets (blue samples).

The neural networks in these experiments are fullyconnected and have three hidden layers of 512 ReLU units.

The generator network accepts twodimensional Gaussian noise vectors.

The networks are trained for 20,000 mini-batches of size 128 using the Adam optimizer with default parameters, where the discriminator is trained for five iterations before every generator iteration.

The training of mixup GANs seems promisingly robust to hyper-parameter and architectural choices.

mixup is a data augmentation method that consists of only two parts: random convex combination of raw inputs, and correspondingly, convex combination of one-hot label encodings.

However, there are several design choices to make.

For example, on how to augment the inputs, we could have chosen to interpolate the latent representations (i.e. feature maps) of a neural network, and we could have chosen to interpolate only between the nearest neighbors, or only between inputs of the same class.

When the inputs to interpolate come from two different classes, we could have chosen to assign a single label to the synthetic input, for example using the label of the input that weights more in the convex combination.

To compare mixup with these alternative possibilities, we run a set of ablation study experiments using the PreAct ResNet-18 architecture on the CIFAR-10 dataset.

Specifically, for each of the data augmentation methods, we test two weight decay settings (10 which works well for ERM).

All the other settings and hyperparameters are the same as reported in Section 3.2.To compare interpolating raw inputs with interpolating latent representations, we test on random convex combination of the learned representations before each residual block (denoted Layer 1-4) or before the uppermost "average pooling + fully connected" layer (denoted Layer 5).

To compare mixing random pairs of inputs (RP) with mixing nearest neighbors (KNN), we first compute the 200 nearest neighbors for each training sample, either from the same class (SC) or from all the classes (AC).

Then during training, for each sample in a minibatch, we replace the sample with a synthetic sample by convex combination with a random draw from its nearest neighbors.

To compare mixing all the classes (AC) with mixing within the same class (SC), we convex combine a minibatch with a random permutation of its sample index, where the permutation is done in a per-batch basis (AC) or a per-class basis (SC).

To compare mixing inputs and labels with mixing inputs only, we either use a convex combination of the two one-hot encodings as the target, or select the one-hot encoding of the closer training sample as the target.

For label smoothing, we follow BID34 and use 10 as the target for incorrect classes, and 1 − 9 10 as the target for the correct class.

Adding Gaussian noise to inputs is used as another baseline.

We report the median test errors of the last 10 epochs.

Results are shown in TAB8 .From the ablation study experiments, we have the following observations.

First, mixup is the best data augmentation method we test, and is significantly better than the second best method (mix input + label smoothing).

Second, the effect of regularization can be seen by comparing the test error with a small weight decay (10 ).

For example, for ERM a large weight decay works better, whereas for mixup a small weight decay is preferred, confirming its regularization effects.

We also see an increasing advantage of large weight decay when interpolating in higher layers of latent representations, indicating decreasing strength of regularization.

Among all the input interpolation methods, mixing random pairs from all classes (AC + RP) has the strongest regularization effect.

Label smoothing and adding Gaussian noise have a relatively small regularization effect.

Finally, we note that the SMOTE algorithm BID4 does not lead to a noticeable gain in performance.

Data augmentation lies at the heart of all successful applications of deep learning, ranging from image classification BID22 to speech recognition BID14 BID0 .

In all cases, substantial domain knowledge is leveraged to design suitable data transformations leading to improved generalization.

In image classification, for example, one routinely uses rotation, translation, cropping, resizing, flipping BID23 BID30 , and random erasing BID43 to enforce visually plausible invariances in the model through the training data.

Similarly, in speech recognition, noise injection is a prevalent practice to improve the robustness and accuracy of the trained models BID0 .More related to mixup, BID4 propose to augment the rare class in an imbalanced dataset by interpolating the nearest neighbors; BID8 show that interpolation and extrapolation the nearest neighbors of the same class in feature space can improve generalization.

However, their proposals only operate among the nearest neighbors within a certain class at the input / feature level, and hence does not account for changes in the corresponding labels.

Recent approaches have also proposed to regularize the output distribution of a neural network by label smoothing BID34 , or penalizing high-confidence softmax distributions BID26 .

These methods bear similarities with mixup in the sense that supervision depends on multiple smooth labels, rather than on single hard labels as in traditional ERM.

However, the label smoothing in these works is applied or regularized independently from the associated feature values.mixup enjoys several desirable aspects of previous data augmentation and regularization schemes without suffering from their drawbacks.

Like the method of DeVries & Taylor (2017), it does not require significant domain knowledge.

Like label smoothing, the supervision of every example is not overly dominated by the ground-truth label.

Unlike both of these approaches, the mixup transformation establishes a linear relationship between data augmentation and the supervision signal.

We believe that this leads to a strong regularizer that improves generalization as demonstrated by our experiments.

The linearity constraint, through its effect on the derivatives of the function approximated, also relates mixup to other methods such as Sobolev training of neural networks BID7 or WGAN-GP BID15 .

We have proposed mixup, a data-agnostic and straightforward data augmentation principle.

We have shown that mixup is a form of vicinal risk minimization, which trains on virtual examples constructed as the linear interpolation of two random examples from the training set and their labels.

Incorporating mixup into existing training pipelines reduces to a few lines of code, and introduces little or no computational overhead.

Throughout an extensive evaluation, we have shown that mixup improves the generalization error of state-of-the-art models on ImageNet, CIFAR, speech, and tabular datasets.

Furthermore, mixup helps to combat memorization of corrupt labels, sensitivity to adversarial examples, and instability in adversarial training.

In our experiments, the following trend is consistent: with increasingly large α, the training error on real data increases, while the generalization gap decreases.

This sustains our hypothesis that mixup implicitly controls model complexity.

However, we do not yet have a good theory for understanding the 'sweet spot' of this bias-variance trade-off.

For example, in CIFAR-10 classification we can get very low training error on real data even when α → ∞ (i.e., training only on averages of pairs of real examples), whereas in ImageNet classification, the training error on real data increases significantly with α → ∞. Based on our ImageNet and Google commands experiments with different model architectures, we conjecture that increasing the model capacity would make training error less sensitive to large α, hence giving mixup a more significant advantage.mixup also opens up several possibilities for further exploration.

First, is it possible to make similar ideas work on other types of supervised learning problems, such as regression and structured prediction?

While generalizing mixup to regression problems is straightforward, its application to structured prediction problems such as image segmentation remains less obvious.

Second, can similar methods prove helpful beyond supervised learning?

The interpolation principle seems like a reasonable inductive bias which might also help in unsupervised, semi-supervised, and reinforcement learning.

Can we extend mixup to feature-label extrapolation to guarantee a robust model behavior far away from the training data?

Although our discussion of these directions is still speculative, we are excited about the possibilities mixup opens up, and hope that our observations will prove useful for future development.

<|TLDR|>

@highlight

Training on convex combinations between random training examples and their labels improves generalization in deep neural networks