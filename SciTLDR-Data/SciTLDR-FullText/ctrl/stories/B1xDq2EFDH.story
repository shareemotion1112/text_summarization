Despite the impressive performance of deep neural networks (DNNs) on numerous learning  tasks, they still exhibit uncouth behaviours.

One puzzling behaviour is the subtle sensitive reaction of DNNs to various noise attacks.

Such a nuisance has strengthened the line of research around developing and training noise-robust networks.

In this work, we propose a new training regularizer that aims to minimize the probabilistic expected training loss of a DNN subject to a generic Gaussian input.

We provide an efficient and simple approach to approximate such a regularizer for arbitrarily deep networks.

This is done by leveraging the analytic expression of the output mean of a shallow neural network, avoiding the need for memory and computation expensive data augmentation.

We conduct extensive experiments on LeNet and AlexNet on various datasets including MNIST, CIFAR10, and CIFAR100 to demonstrate the effectiveness of our proposed regularizer.

In particular, we show that networks that are trained with the proposed regularizer benefit from a boost in robustness against Gaussian noise to an equivalent amount of performing 3-21 folds of noisy data augmentation.

Moreover, we empirically show on several architectures and datasets that improving robustness against Gaussian noise, by using the new regularizer, can improve the overall robustness against 6 other types of attacks by two orders of magnitude.

Deep neural networks (DNNs) have emerged as generic models that can be trained to perform impressively well in a variety of learning tasks ranging from object recognition (He et al., 2016) and semantic segmentation (Long et al., 2015) to speech recognition and bioinformatics (Angermueller et al., 2016) .

Despite their increasing popularity, flexibility, generality, and performance, DNNs have been recently shown to be quite susceptible to small imperceptible input noise (Szegedy et al., 2014; Moosavi-Dezfooli et al., 2016; Goodfellow et al., 2015) .

Such analysis gives a clear indication that even state-of-the-art DNNs may lack robustness.

Consequently, there has been an ever-growing interest in the machine learning community to study this uncanny behaviour.

In particular, the work of (Goodfellow et al., 2015; Moosavi-Dezfooli et al., 2016) demonstrates that there are systematic approaches to constructing adversarial attacks that result in misclassification errors with high probability.

Even more peculiarly, some noise perturbations seem to be doubly agnostic (Moosavi-Dezfooli et al., 2017) , i.e. there exist deterministic perturbations that can result in misclassification errors with high probability when applied to different networks, irrespective of the input (denoted network and input agnostic).

Understanding this degradation in performance under adversarial attacks is of tremendous importance, especially for real-world DNN deployment, e.g. self-driving cars/drones and equipment for the visually impaired.

A standard and popular means to alleviate this nuisance is noisy data augmentation in training, i.e. a DNN is exposed to noisy input images during training so as to bolster its robustness during inference.

Several works have demonstrated that DNNs can in fact benefit from such augmentation (Moosavi-Dezfooli et al., 2016; Goodfellow et al., 2015) .

However, data augmentation in general might not be sufficient for two reasons.

(1) Particularly with high-dimensional input noise, the amount of data augmentation necessary to sufficiently capture the noise space will be very large, which will increase training time.

(2) Data augmentation with high energy noise can negatively impact the performance on noise-free test examples.

This can be explained by the fundamental trade-off between accuracy and robustness (Tsipras et al., 2018; Boopathy et al., 2019) .

It can also arise from the fact that augmentation forces the DNN to have the same prediction for two vastly different versions of the same input, noise-free and a substantially corrupted version.

Despite the impressive performance of DNNs on various tasks, they have been shown to be very sensitive to certain types of noise, commonly referred to as adversarial examples, particularly in the recognition task (Moosavi-Dezfooli et al., 2016; Goodfellow et al., 2015) .

Adversarial examples can be viewed as small imperceptible noise that, once added to the input of a DNN, its performance is severely degraded.

This finding has incited interest in studying/measuring the robustness of DNNs.

The literature is rich with work that aims to unify and understand the notion of network robustness.

For instance, Szegedy et al. (2014) suggested a spectral stability analysis for a wide class of DNNs by measuring the Lipschitz constant of the affine transformation describing a fully-connected or a convolutional layer.

This result was extended to compute an upper bound for a composition of layers, i.e. a DNN.

However, this measure sets an upper bound on the robustness over the entire input domain and does not take into account the noise distribution.

Later, Fawzi et al. (2017a) defined robustness as the mean support of the minimum adversarial perturbation, which is now the most common definition for robustness.

Not only was robustness studied against adversarial perturbations but also against geometric transformations to the input.

emphasized the independence of the robustness measure to the ground truth class labels and that it should only depend on the classifier and the dataset distribution.

Subsequently, two different metrics to measure DNN robustness were proposed: one for general adversarial attacks and another for noise sampled from uniform distribution.

Recently, Gilmer et al. (2018) showed the trade-off between robustness and test error from a theoretical point of view on a simple classification problem with hyperspheres.

On the other hand, and based on various robustness analyses, several works proposed various approaches in building networks that are robust against noise sampled from well known distributions and against generic adversarial attacks.

For instance, Grosse et al. (2017) proposed a model that was trained to classify adversarial examples with statistical hypothesis testing on the distribution of the dataset.

Another approach is to perform statistical analysis on the latent feature space instead (Li & Li, 2017; Feinman et al., 2017) , or train a DNN that rejects adversarial attacks (Lu et al., 2017) .

Moreover, the geometry of the decision boundaries of DNN classifiers was studied by Fawzi et al. (2017b) to infer a simple curvature test for this purpose.

Using this method, one can restore the orig- Figure 1 : Overview of the proposed graph for training Gaussian robust networks.

The yellow block corresponds to an arbitrary network Φ(., θ) viewed as the composition of two subnetworks separated by a ReLU.

The stream on the bottom computes the output mean µ4 of the network Φ(:, θ) assuming that (i) the noise input distribution is independent Gaussian with variances σ 2 x , and (ii) Ω(. : θ2) is approximated by a linear function.

This evaluation for the output mean is efficient as it only requires an extra forward pass (bottom stream), as opposed to other methods that employ computationally and memory intensive network linearizations or data augmentation.

inal label and classify the input correctly.

Restoring the original input using defense mechanisms, which can only detect adversarial examples, can be done by denoising (ridding it from its adversarial nature) so long as the noise perturbation is well-known and modeled apriori (Zhu et al., 2016) .

A fresh approach to robustness was proposed by Zantedeschi et al. (2017) , where they showed that using bounded ReLUs (if augmented with Gaussian noise) to limit the output range can improve robustness.

A different work proposed to distill the learned knowledge from a deep model to retrain a similar model architecture as a means to improving robustness (Papernot et al., 2016) .

This training approach is one of many adversarial training strategies for robustness Makhzani et al. (2016) .

More closely to our work is (Cisse et al., 2017) , where a new training regularizer was proposed for a large family of DNNs.

The proposed regularizer softly enforces that the upper bound of the Lipshitz constant of the output of the network to be less than or equal to one.

Moreover and very recently, the work of Bibi et al. (2018) has derived analytic expressions for the output mean and covariance of networks in the form of (Affine, ReLU, Affine) under a generic Gaussian input.

This work also demonstrates how a (memory and computation expensive) two-stage linearization can be employed to locally approximate a deep network with a two layer one, thus enabling the application of the derived expressions on the approximated shallower network.

Most prior work requires data augmentation, training new architectures that distill knowledge, or detect adversaries a priori, resulting in expensive training routines that may be ineffective in the presence of several input noise types.

To this end, we address these limitations through our new regularizer that aims to fundamentally tackle Gaussian input noise without data augmentation and, as a consequence, improves overall robustness against other types of attacks.

Background on Network Moments.

Networks with a single hidden layer of the form (Affine, ReLU, Affine) can be written in the functional form g(x) = Bmax (Ax + c 1 , 0 p ) + c 2 .

The max(.) is an element-wise operator, A ∈ R p×n , and B ∈ R d×p .

Thus, g : Bibi et al. (2018) showed that:

Note that µ 2 = Aµ x + c 1 , σ 2 = diag (Σ 2 ), Σ 2 = AΣ x A , Φ and ϕ are the standard Gaussian cumulative (CDF) and density (PDF) functions, respectively.

The vector multiplication and division are element-wise operations.

Lastly, diag(.) extracts the diagonal elements of a matrix into a vector.

For ease of notation, we let

To extend the results of Theorem (1) to deeper models, a two-stage linearization was proposed in Bibi et al. (2018) , where (A, B) and (c 1 , c 2 ) are taken to be the Jacobians and biases of the first order Taylor approximation to the two network functions around a ReLU layer in a DNN.

Refer to Bibi et al. (2018) for more details about this expression and the proposed linearization.

Proposed Robust Training Regularizer.

To propose an alternative to noisy data augmentation to address its drawbacks, one has to realize that this augmentation strategy aims to minimize the expected training loss of a DNN when subjected to noisy input distribution D through sampling.

In fact, it minimizes an empirical loss that approximates this expected loss when enough samples are present during training.

When sampling is insufficient (a drawback of data augmentation in highdimensions), this approximation is too loose and robustness can suffer.

However, if we have access to an analytic expression for the expected loss, expensive data augmentation can be averted.

This is the key motivation of the paper.

Mathematically, the training loss can be modeled as

Here, Φ : R n → R d is any arbitrary network with parameters θ, is the loss function,

are the noise-free data-label training pairs, and α ≥ 0 is a trade off parameter.

While the first term in Equation 1 is the standard empirical loss commonly used for training, the second term is often replaced with its Monte Carlo estimate through data augmentation.

That is, for each training example x i , the second term is approximated with an empirical average ofÑ noisy examples of

This will increase the size of the dataset by a factor ofÑ , which will in turn increase training complexity.

As discussed earlier, network performance on the noise-free examples can also be negatively impacted.

Note that obtaining a closed form expression for the second term in Equation 1 for some of the popularly used losses is more complicated than deriving expressions for the output mean of the network Φ itself, e.g. in Theorem (1).

Therefore, we propose to replace this loss with the following surrogate

Because of Jensen's inequality, Equation 2 is a lower bound to Eq Equation 1 when is convex, which is the case for most popular losses including 2 -loss and cross-entropy loss.

The proposed second term in Equation 2 encourages that the output mean of the network Φ of every noisy example (x i + n) matches the correct class label y i .

This regularizer will stimulate a separation among the output mean of the classes if the training data is subjected to noise sampled from D. Having access to an analytic expression for these means will prompt a simple inexpensive training, where the actual size of the training set is unaffected and augmentation is avoided.

This form of regularization is proposed to replace data augmentation.

While a closed-form expression for the second term of Equation 2 might be infeasible for a general network Φ(.), an expensive approximation can be attained.

In particular, Theorem Equation 1 provides an analytic expression to evaluate the second term in Equation 2, for when D is Gaussian and when the network is approximated by a two-stage linearization procedure as

However, it is not clear how to utilize such a result to regularize networks during training with Equation 2 as a loss.

This is primarily due to the computationally expensive and memory intensive network linearization proposed in Bibi et al. (2018) .

Specifically, the linearization parameters (A, B, c 1 , c 2 ) are a function of the network parameters, θ, which are updated with every gradient descent step on Equation 2; thus, two-stage linearization has to be performed in every θ update step, which is infeasible.

proposes a generic approach to train robust arbitrary networks against noise sampled from an arbitrary distribution D. Since the problem in its general setting is too broad for detailed analysis, we restrict the scope of this work to the class of networks, which are most popularly used and parameterized by θ, Φ(.; θ) :

with ReLUs as nonlinear activations.

Moreover, since random Gaussian noise was shown to exhibit an adversarial nature Bibi et al. (2018) ; Rauber et al. (2017) ; Franceschi et al. (2018) , and it is one of the most well studied noise models for the useful properties it exhibits, we restrict D to the case of Gaussian noise.

In particular, D is independent zero-mean Gaussian noise at the input, i.e.

x , where σ 2 x ∈ R n is a vector of variances and Diag(.) reshapes the vector elements into a diagonal matrix.

Generally, it is still difficult to compute the second term in Equation 2 under Gaussian noise for arbitrary networks.

However, if we have access to an inexpensive approximation of the network, avoiding the computationally and memory expensive network linearization in Bibi et al. (2018) , an approximation to the second term in Equation 2 can be used for efficient robust training directly on θ.

Consider the l th ReLU layer in Φ(.; θ).

the network can be expressed as Φ(.; θ) = Ω(ReLU l (Υ(., θ 1 )); θ 2 ).

Note that the parameters of the overall network Φ(.; θ) is the union of the parameters of the two subnetworks Υ(.; θ 1 ) and Ω(.; θ 2 ), i.e. θ = θ 1 ∪ θ 2 .

Throughout this work and to simplify the analysis, we set l = 1.

With such a choice of l, the first subnetwork Υ(., θ 1 ) is affine with θ 1 = {A, c 1 }.

However, the second subnetwork Ω(., θ 2 ) is not linear in general, and thus, one can linearize Ω(., θ 2 ) at E n∼N (0,Σx) [ReLU 1 (Υ (x i + n; θ 1 ))] = T (µ 2 , σ 2 ) = µ 3 .

Note that µ 3 is the output mean after the ReLU and µ 2 = Ax i + c 1 , since Υ(x i + n; θ 1 ) = A (x i + n) + c 1 .

Both T (., .) and σ 2 are defined in Equation 1.

Thus, linearizing Ω at µ 3 with linearization parameters (B, c 2 ) being the Jacobian of Ω and c 2 = Ω(µ 3 , θ 2 ) − Bµ 3 , we have that, for any point v i close to µ 3 : Ω(v i , θ 2 ) ≈ Bv i + c 2 .

While computing (B, c 2 ) through linearization is generally very expensive, computing the approximation to Equation 2 requires explicit access to neither B nor c 2 .

Note that this second term for l = 1 is given as:

The approximation follows from the assumption that the input to the second subnetwork Ω(.; θ 2 ), i.e.

Or simply, that the input to Ω is close to the mean inputs, i.e. µ 3 , to Ω under Gaussian noise.

The penultimate equality follows from the linearity of the expectation.

As for the last equality, (B, c 2 ) are the linearization parameters of Ω at µ 3 , where c 2 = Ω(µ 3 , θ 2 ) − Bµ 3 by the first order Taylor approximation.

Thus, computing the second term of Equation 2 according to Equation 3 can be simply approximated by a forward pass of µ 3 through the second network Ω. As for computing µ 3 = T (µ 2 , σ 2 ), note that µ 2 = Ax i + c 1 in Equation 3, which is equivalent to a forward pass of x i through the first subnetwork because Υ(., θ 1 ) is linear with θ 1 = {A, c 1 }.

Moreover, since σ 2 = diag (AΣ x A ), we have:

The expression for σ 2 can be efficiently computed by simply squaring the linear parameters in the first subnetwork and performing a forward pass of the input noise variance σ 2 x through Υ without the bias c 1 and taking the element-wise square root.

Lastly, it is straightforward to compute T (µ 2 , σ 2 ) as it is an elementwise function in Equation 1.

The overall computational graph in Figure 1 shows a summary of the computation needed to evaluate the loss in Equation 2 using only forward passes through the two subnetworks Υ and Ω. It is now possible with the proposed efficient approximation of our proposed regularizer in Equation 2 to efficiently train networks on noisy training examples that are corrupted with noise N (0, Σ x ) without any form of prohibitive data augmentation.

In this section, we conduct experiments on multiple network architectures and datasets to demonstrate the effectiveness of our proposed regularizer in training more robust networks, especially in comparison with data augmentation.

To standardize robustness evaluation, we first propose a new unified robustness metric against additive noise from a general distribution D and later specialize it when D is Gaussian.

Lastly, we show that networks trained with our proposed regularizer not only outperform in robustness networks trained with Gaussian augmented data.

Moreover, we show that such networks are also much more magnitudes times robust against other types of attacks.

On the Robustness Evaluation Metric.

While there is a consensus on the definition of robustness in the presence of adversarial attacks, as the smallest perturbation required to fool a network, i.e. to change its prediction, it is not straightforward to extend such a definition to additive noise sampled from a distribution D. In particular, the work of tried to address this difficulty by defining the robustness of a classifier around an example x as the distance between x and the closest Figure 2: General trade-off between accuracy and robustness on LeNet.

We see, in all plots, that the accuracy tends to be negatively correlated with robustness over varying noise levels and amount of augmentation.

Baseline refers to training with neither data augmentation nor our regularizer.

However, it is hard to compare the performance of our method against data augmentation from these plots as we can only compare the robustness of models with similar noise-free testing accuracy.

decision boundary.

However, this definition is difficult to compute in practice and is not scalable, as it requires solving a generally nonconvex optimization problem for every testing example x that may also suffer from poor local minima.

To remedy these drawbacks, we present a new robustness metric for generic additive noise.

Robustness Against Additive Noise.

Consider a classifier Ψ(.) with ψ(x) = arg max i Ψ i (x) as the predicted class label for the example x regardless of the correct class label y i .

We define the robustness on a sample x against a generic additive noise sampled from a distribution D as

Here, the proposed robustness metric D (x) measures the probability of the classifier to preserve the original prediction of the noise-free example ψ(x) after adding noise, ψ(x + n), from distribution D. Therefore, the robustness over a testing dataset T can be defined as the expected robustness over the test dataset: Franceschi et al. (2018) , for ease, we relax Equation 4 from the probability of preserving the prediction score to a 0/1 robustness over mrandomly sampled examples from D. That is, D (x) = 1 means that, among m randomly sampled noise from D added to x, none changed the prediction from ψ(x).

However, if a single example of these m samples changed the prediction from ψ(x), we set D (x) = 0.

Thus, the robustness score is the average of this measure over the testing dataset T .

Robustness Against Gaussian Noise.

For additive Gaussian noise, i.e. D = N (0, Σ x = Diag σ 2 x ), robustness is averaged over a range of testing variances σ 2 x .

We restrict σ x to 30 evenly sampled values in [0, 0.5], where this set is denoted as A 1 .

In practice, this is equivalent to sampling m Gaussian examples for each σ x ∈ A, and if none of the m samples changes the prediction of the classifier ψ from the original noise-free example, the robustness for that sample at that σ x noise level is set to 1 and then averaged over the complete testing set.

Then, the robustness is the average over multiple σ x ∈ A. To make the computation even more efficient, instead of sampling a large number of Gaussian noise samples (m), we only sample a single noise sample with the average energy over D. That is, we sample a single n of norm n 2 = σ x √ n.

This is due to the fact that

Experimental Setup.

In this section, we demonstrate the effectiveness of the proposed regularizer in improving robustness.

Several experiments are performed with our objective Equation 2, where we strike a comparison with data augmentation approaches.

Architecture Details.

The input images in MNIST (gray-scale) and CIFAR (colored) are squares with sides equal to 28 and 32, respectively.

Since AlexNet was originally trained on ImageNet of report results for models with a test accuracy that is at least as good as the accuracy of the baseline with a tolerance: 0%, 0.39%, and 0.75% for MNIST, CIFAR10, CIFAR100, respectively.

Only the models with the highest robustness are presented.

Training with our regularizer can attain similar/better robustness than 21-fold noisy data augmentation on MNIST and CIFAR100, while maintaining a high noise-free test accuracy.

sides equal to 224, we will marginally alter the implementation of AlexNet in TorchVision Marcel & Rodriguez (2010) to accommodate for this difference.

First, we change the number of hidden units in the first fully-connected layer (in LeNet to 4096, AlexNet to 256, LeNet on MNIST to 3136).

For AlexNet, we changed all pooling kernel sizes from 3 to 2 and the padding size of conv1 from 2 to 5.

Second, we swapped each maxpool with the preceding ReLU, which makes training and inference more efficient.

Third, we enforce that the first layer in all the models is a convolution followed by ReLU as discussed earlier.

Lastly, to simplify analysis, we removed all dropout layers.

We leave the details of the optimization hyper-parameters to the appendix.

Results.

For each model and dataset, we compare baseline models, i.e. models trained with noisefree data and without our regularization, with two others: one using data augmentation and another using our proposed regularizer.

Each of the latter has two configurable variables: the level of noise controlled by σ Accuracy vs. Robustness.

We start by demonstrating that data augmentation tends to improve the robustness, as captured by (T ) over the test set, at the expense of decreasing the testing accuracy on the noise-free examples.

Realizing this is essential for a fair comparison, as one would need to compare the robustness of networks that only have similar noise-free testing accuracies.

To this end, we ran 60 training experiments with data augmentation on LeNet with three datasets (MNIST, CIFAR10, and CIFAR100), four augmentation levels (Ñ ∈ {2, 6, 11, 21}), and five noise levels (σ x ∈ A = {0.125, 0.25, 0.325, 0.5, 1.0}).

In contrast, we ran robust training experiments using Equation 2 with the trade-off coefficient α ∈ {0.5, 1, 1.5, 2, 5, 10, 20} on the same datasets, but we extended the noise levels σ x to include the extreme noise regime of σ x ∈ {2, 5, 10, 20}. These noise levels are too large to be used for data augmentation, especially since x ∈ [0, 1] n ; however, as we will see, they are still beneficial for our proposed regularizer.

Figure 2 shows both the testing accuracy and robustness as measured by (T ) over a varying range of training σ x for the data augmentation approach of LeNet on MNIST, CIFAR-10 and CIFAR-100.

It is important to note here that the main goal of these plots is not to compare the robustness score, but rather, to demonstrate a very important trend.

In particular, increasing the training σ x for each approach degrades testing accuracy on noise-free data.

However, the degradation in our approach is much more graceful since the trained LeNet model was never directly exposed to individually corrupted examples during training as opposed to the data augmentation approach.

Note that our regularizer enforces the separation between the expected output prediction analytically.

Moreover, the robustness of both methods consistently improves as the training σ x increases.

This trend holds even on the easiest dataset (MNIST).

Interestingly, models trained with our regularizer enjoy an improvement in testing accuracy over the baseline model.

Such behaviour only emerges with a large factor of augmentation, N = 21, and a small enough training σ x on MNIST.

This indicates that models can benefit from better accuracy with a good approximation of Equation 1 through our proposed objective or through extensive Monte Carlo estimation.

However, as σ x increases, Monte Carlo estimates of the second term in Equation 1 via data augmentation (withÑ = 21) is no longer enough to capture the noise.

Robustness Comparison.

For fair comparison, it is essential to only compare the robustness of networks that achieve similar testing accuracy, since perfect robustness is attainable with a deterministic classifier that assigns the same class label regardless of the input.

In fact, we proposed a unified robustness metric for the reason that most commonly used metrics are disassociated from Table 1 : Gaussian robustness improves overall robustness.

We report the robustness metrics corresponding to various attacks (PGD, LBFGS, FGSM, and DF2), our proposed GNR metric, and the test accuracy ACC for LeNet and AlexNet networks trained on MNIST and CIFAR100 using our proposed regularizer with noise variance σ in training.

Note that σ = 0 corresponds to baseline models trained without our regularizer.

We observe that training networks with our proposed regularizer (designed for additive Gaussian attacks) not only improves the robustness against Gaussian attacks but also against 6 other types of attacks which 4 of them listed here and the others are left for appendix.

the ground-truth labels and only consider model predictions.

Therefore, we filtered out the results from Figure 2 by removing all the experiments that achieved lower test accuracy than the baseline model.

Figure 3 summarizes these results for LeNet.

Now, we can clearly see the difference between training with data augmentation and our approach.

For MNIST (Figure 3a) , we achieved the same robustness as 21-fold data augmentation without feeding the network with any noisy examples during training and while preserving the same baseline accuracy.

Interestingly, for CIFAR10 (Figure 3b ), our method is twice as robust as the best robustness achieved via data augmentation.

Moreover, for CIFAR100 (Figure 3c ), we are able to outperform data augmentation by around 5%.

Finally, for extra validation, we also conducted the same experiments with AlexNet on CIFAR10 and CIFAR100 which can be found in the appendix.

We can see that our proposed regularizer can improve robustness by 15% on CIFAR10 and around 25% on CIFAR100.

It is interesting to note that for CIFAR10, data augmentation could not improve the robustness of the trained models without drastically degrading the testing accuracy on the noise-free examples.

Moreover, it is interesting to observe that the best robustness achieved through data augmentation is even worse than the baseline.

This could be due to the trade-off coefficient α in Equation 1.

Towards General Robustness via Gaussian Robustness.

Here, we investigate whether improving robustness to Gaussian input noise can improve robustness against other types of attacks.

Specifically, we compare the robustness of models trained using our proposed regularizer (robust again Gaussian attacks) with baseline models subject to different types of attacks: Projected Gradient Descent (PGD) and LBFGS attacks Szegedy et al. (2014) , Fast Sign Gradient Method (FGSM) Goodfellow et al. (2015) , and DeepFool L2Attack (DF2) Moosavi-Dezfooli et al. (2016) as provide by Rauber et al. (2017) .

For all these attacks, we report the minimum energy perturbation that can change the network prediction.

We also report our Gaussian Network Robustness (GNR) metric, which is the Gaussian version of Equation 4 along with the testing accuracy (ACC).

We perform experiments on LeNet on MNIST, CIFAR10 and CIFAR100 datasets and on AlexNet on both CI-FAR10 and CIFAR100.

Due to space constraints, we show the robustness results for only LeNet on MNIST and AlexNet of CIFAR100 and leave the rest along with two other types of attacks for the appendix.

Table 1 shows that improving GNR through our data augmentation free regularizer can significantly improve all robustness metrics.

For instance, comparing LeNet trained with our proposed regularizer against LeNet trained without any regularization, i.e. σ = 0, we see that robustness against all types of attacks improves by almost two orders of magnitude, while maintaining a similar testing accuracy.

A similar improvement in performance is consistently present for AlexNet on CIFAR100.

Addressing the sensitivity problem of deep neural networks to adversarial perturbation is of great importance to the machine learning community.

However, building robust classifiers against this noises is computationally expensive, as it is generally done through the means of data augmentation.

We propose a generic lightweight analytic regularizer, which can be applied to any deep neural network with a ReLU activation after the first affine layer.

It is designed to increase the robustness of the trained models under additive Gaussian noise.

We demonstrate this with multiple architectures and datasets and show that it outperforms data augmentation without observing any noisy examples.

A EXPERIMENTAL SETUP AND DETAILS.

All experiments, are conducted using PyTorch version 0.4.1 Paszke et al. (2017) .

All hyperparameters are fixed and Table 2 we report the setup for the two optimizers.

In particular, we use the Adam optimizaer Kingma & Ba (2015) with β 1 = 0.9, β 2 = 0.999, = 10 −8 with amsgrad set to False.

The second optimizer is SGD Loshchilov & Hutter (2017) with momentum=0.9, dampening=0, with Nesterov acceleration.

In each experiment, we randomly split the training dataset into 10% validation and 90% training and monitor the validation loss after each epoch.

If validation loss did not improve for lr patience epochs, we reduce the learning rate by multiplying it by lr factor.

We start with an initial learning rate of lr initial.

The training is terminated only if the validation loss did not improve for loss patience number of epochs or if the training reached 100 epochs.

We report the results of the model with the best validation loss.

In particular, one can observe that with σ large than 0.7 the among of noise is severe even for the human level.

Training on such extreme noise levels will deem data augmentation to be difficult.

We measure the robustness against Gaussian noise by averaging over a range of input noise levels, where at each level for each image, we consider it misclassified if the probability of it being misclassified is greater than a certain threshold.

The final robustness is the average over multiple testing σ x .

This is special case of the more general case in Equation (4).

We then report the area under the curve of the robustness with varying testing σ x as shown in Figure 6 .

The area under this curve thus represents the overall robustness of a given model under several varying input noise standard deviation σ x .

We report the robustness of several architectures over several datasets with and without our trained regularizer.

We show that our proposed efficient regularizer not only improves the robustness against Gaussin noise attacks but againts several other types of attacks.

Table 3 summarizes the types of attacks used for robustness evaluation.

reported models trained with our regularizer on CIFAR10 and CIFAR100 on all training σx are within 1.68% and 4.83% of the baseline accuracy, respectively.

The models trained with the proposed regularizer achieve better robustness than 11-fold and 6-fold noisy data augmentation on CIFAR10 and CIFAR100, respectively.

34.65 35.50 Table 4 : Gaussian robustness improves overall robustness.

We report the robustness metrics corresponding to various attacks (PGD, LBFGS, FGSM, AGA, AUA and DF2), our

proposed GNR metric, and the test accuracy ACC for LeNet and AlexNet networks trained on MNIST, CIFAR10 and CIFAR100 using our proposed regularizer with noise variance σ in training.

Note that σ = 0 corresponds to baseline models trained without our regularizer.

We observe that training networks with our proposed regularizer (designed for additive Gaussian attacks) not only improves the robustness against Gaussian attacks but also against 6 other types of attacks.

<|TLDR|>

@highlight

An efficient estimate to the Gaussian first moment of DNNs as a regularizer to training robust networks.