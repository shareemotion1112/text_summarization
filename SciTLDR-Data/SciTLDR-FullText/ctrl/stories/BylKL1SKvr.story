Deep neural networks trained on a wide range of datasets demonstrate impressive transferability.

Deep features appear general in that they are applicable to many datasets and tasks.

Such property is in prevalent use in real-world applications.

A neural network pretrained on large datasets, such as ImageNet, can significantly boost generalization and accelerate training if fine-tuned to a smaller target dataset.

Despite its pervasiveness, few effort has been devoted to uncovering the reason of transferability in deep feature representations.

This paper tries to understand transferability from the perspectives of improved generalization, optimization and the feasibility of transferability.

We demonstrate that 1) Transferred models tend to find flatter minima, since their weight matrices stay close to the original flat region of pretrained parameters when transferred to a similar target dataset; 2) Transferred representations make the loss landscape more favorable with improved Lipschitzness, which accelerates and stabilizes training substantially.

The improvement largely attributes to the fact that the principal component of gradient is suppressed in the pretrained parameters, thus stabilizing the magnitude of gradient in back-propagation.

3) The feasibility of transferability is related to the similarity of both input and label.

And a surprising discovery is that the feasibility is also impacted by the training stages in that the transferability first increases during training, and then declines.

We further provide a theoretical analysis to verify our observations.

The last decade has witnessed the enormous success of deep neural networks in a wide range of applications.

Deep learning has made unprecedented advances in many research fields, including computer vision, natural language processing, and robotics.

Such great achievement largely attributes to several desirable properties of deep neural networks.

One of the most prominent properties is the transferability of deep feature representations.

Transferability is basically the desirable phenomenon that deep feature representations learned from one dataset can benefit optimization and generalization on different datasets or even different tasks, e.g. from real images to synthesized images, and from image recognition to object detection (Yosinski et al., 2014) .

This is essentially different from traditional learning techniques and is often regarded as one of the parallels between deep neural networks and human learning mechanisms.

In real-world applications, practitioners harness transferability to overcome various difficulties.

Deep networks pretrained on large datasets are in prevalent use as general-purpose feature extractors for downstream tasks (Donahue et al., 2014) .

For small datasets, a standard practice is to fine-tune a model transferred from large-scale dataset such as ImageNet (Russakovsky et al., 2015) to avoid over-fitting.

For complicated tasks such as object detection, semantic segmentation and landmark localization, ImageNet pretrained networks accelerate training process substantially (Oquab et al., 2014; He et al., 2018) .

In the NLP field, advances in unsupervised pretrained representations have enabled remarkable improvement in downstream tasks (Vaswani et al., 2017; Devlin et al., 2019) .

Despite its practical success, few efforts have been devoted to uncovering the underlying mechanism of transferability.

Intuitively, deep neural networks are capable of preserving the knowledge learned on one dataset after training on another similar dataset (Yosinski et al., 2014; Li et al., 2018b; 2019) .

This is even true for notably different datasets or apparently different tasks.

Another line of works have observed several detailed phenomena in the transfer learning of deep networks (Kirkpatrick et al., 2016; Kornblith et al., 2019 ), yet it remains unclear why and how the transferred representations are beneficial to the generalization and optimization perspectives of deep networks.

The present study addresses this important problem from several new perspectives.

We first probe into how pretrained knowledge benefits generalization.

Results indicate that models fine-tuned on target datasets similar to the pretrained dataset tend to stay close to the transferred parameters.

In this sense, transferring from a similar dataset makes fine-tuned parameters stay in the flat region around the pretrained parameters, leading to flatter minima than training from scratch.

Another key to transferability is that transferred features make the optimization landscape significantly improved with better Lipschitzness, which eases optimization.

Results show that the landscapes with transferred features are smoother and more predictable, fundamentally stabilizing and accelerating training especially at the early stages of training.

This is further enhanced by the proper scaling of gradient in back-propagation.

The principal component of gradient is suppressed in the transferred weight matrices, controlling the magnitude of gradient and smoothing the loss landscapes.

We also investigate a common concern raised by practitioners: when is transfer learning helpful to target tasks?

We test the transferability of pretrained networks with varying inputs and labels.

Instead of the similarity between pretrained and target inputs, what really matters is the similarity between the pretrained and target tasks, i.e. both inputs and labels are required to be sufficiently similar.

We also investigate the relationship between pretraining epoch and transferability.

Surprisingly, although accuracy on the pretrained dataset increases throughout training, transferability first increases at the beginning and then decreases significantly as pretraining proceeds.

Finally, this paper gives a theoretical analysis based on two-layer fully connected networks.

Theoretical results consistently justify our empirical discoveries.

The analysis here also casts light on deeper networks.

We believe the mechanism of transferability is the fundamental property of deep neural networks and the in-depth understanding presented here may stimulate further algorithmic advances.

There exists extensive literature on transferring pretrained representations to learn an accurate model on a target dataset.

Donahue et al. (2014) employed a brand-new label predictor to classify features extracted by the pre-trained feature extractor at different layers of AlexNet (Krizhevsky et al., 2012) .

Oquab et al. (2014) showed deep features can benefit object detection tasks despite the fact that they are trained for image classification.

Ge & Yu (2017) introduced a selective joint fine-tuning scheme for improving the performance of deep learning tasks under the scenario of insufficient training data.

The enormous success of the transferability of deep networks in applications stimulates empirical studies on fine-tuning and transferability.

Yosinski et al. (2014) observed the transferability of deep feature representations decreases as the discrepancy between pretrained task and target task increases and gets worse in higher layers.

Another phenomenon of catastrophic forgetting as discovered by Kirkpatrick et al. (2016) describes the loss of pretrained knowledge when fitting to distant tasks.

Huh et al. (2016) delved into the influence of ImageNet pretrained features by pretraining on various subsets of the ImageNet dataset.

Kornblith et al. (2019) further demonstrated that deep models with better ImageNet pretraining performance can transfer better to target tasks.

As for the techniques used in our analysis, Li et al. (2018a) proposed the impact of the scaling of weight matrices on the visualization of loss landscapes.

Santurkar et al. (2018) proposed to measure the variation of loss to demonstrate the stability of loss function.

Du et al. (2019) provided a powerful framework of analyzing two-layer over-parametrized neural networks, with elegant results and no strong assumptions on input distributions, which is flexible for our extensions to transfer learning.

A basic observation of transferability is that tasks on target datasets more similar to the pretrained dataset have better performance.

We delve deeper into this phenomenon by experimenting on a variety of target datasets (Figure 1 ), carried out with two common settings: 1) train only the last layer by fixing the pretrained network as the feature extractor and 2) train the whole network by fine-tuning from the pretrained representations.

Results in Table 1 clearly demonstrate that, for both settings and for all target datasets, the training error converges to nearly zero while the generalization error varies significantly.

In particular, a network pretrained on more similar dataset tends to generalize better and converge faster on the target dataset.

A natural implication is that the knowledge learned from the pretrained networks can only be preserved to different extents for different target datasets.

We substantiate this implication with the following experiments.

To analyze to what extent the knowledge learned from pretrained dataset is preserved, for the fixing setting, we compute the Frobenius norm of the deviation between fine-tuned weight W and pretrained weight W 0 as 1 ??? n W ??? W 0 F , where n denotes the number of target examples (for the fine-tuning setting, we compute the sum of deviations in all layers

Results are shown in Figure 2 .

It is surprising that although accuracy may oscillate, (a)

Figure 2: The deviation of the weight parameters from the pretrained ones in the transfer process to different target datasets.

For all datasets,

can be preserved on target datasets more similar to ImageNet, yielding smaller

Why is preserving pretrained knowledge related to better generalization?

From the experiments above, we can observe that models preserving more transferred knowledge (i.e. yielding smaller

It is reasonable to hypothesize that 1 ??? n W ???W 0 F is implicitly bounded in the transfer process, and that the bound is related to the similarity between pretrained and target datasets (We will formally study this conjecture in the theoretical analysis).

Intuitively, a neural network attempts to fit the training data by twisting itself from the initialization point.

For similar datasets the twist will be mild, with the weight parameters staying closer to the pretrained parameters.

Such property of staying near the pretrained weight is crucial for understanding the improvement of generalization.

Since optimizing deep networks inevitably runs into local minima, a common belief of deep networks is that the optimization trajectories of weight parameters on different datasets will be essentially different, leading to distant local minima.

To justify whether this is true, we compare the weight matrices of training from scratch and using ImageNet pretrained representations in Figure 4 .

Results are quite counterintuitive.

The local minima of different datasets using ImageNet pretraining are closed to each other, all concentrating around ImageNet pretrained weight.

However, the local minima of training from scratch and ImageNet pretraining are way distant, even on the same dataset. .

Surprisingly, weight matrices on the same dataset may be distant at convergence when using different initializations.

On the contrary, even for discrepant datasets, the weight matrices stay close to the initialization when using the same pretrained parameters.

This provides us with a clear picture of how transferred representations improve generalization on target datasets.

Rich studies have indicated that the properties of local minima are directly related to generalization (Keskar et al., 2017; Izmailov et al., 2018) .

Using pretrained representations restricts weight matrices to stay near the pretrained weight.

Since the pretrained dataset is usually sufficiently large and of high-quality, transferring their representations will lead to flatter minima located in large flat basins.

On the contrary, training from scratch may find sharper minima.

To observe this, we adopt filter normalization (Li et al., 2018a) as the visualization tool, and illustrate the loss landscapes around the minima in Figure 3 .

This observation concurs well with the experiments above.

The weight matrices for datasets similar to pretrained ones deviate less from pretrained weights and stay in the flat region.

On more different datasets, the weight matrices have to go further from pretrained weights to fit the data and may run out of the flat region.

A common belief of modern deep networks is the improvement of loss landscapes with techniques such as BatchNorm (Ioffe & Szegedy, 2015) and residual structures (He et al., 2016) .

Li et al. (2018a); Santurkar et al. (2018) validated this improvement when the model is close to convergence.

However, it is often overlooked that loss landscapes can still be messy at the initialization point.

To verify this conjecture, we visualize the loss landscapes centered at the initialization point of the 25th layer of ResNet-50 in Figure 5 .

(Visualizations of the other layers can be found in Appendix B.4.)

ImageNet pretrained networks have much smoother landscape than networks trained with random initialization.

The improvement of loss landscapes at the initialization point directly gives rise to the acceleration of training.

Concretely, transferred features help ameliorate the chaos of loss landscape with improved Lipschitzness in the early stages of training.

Thus, gradient-based optimization method can easily escape from the initial region where the loss is very large.

The properties of loss landscapes influence the optimization fundamentally.

In randomly initialized networks, going in the direction of gradient may lead to large variation in the loss function.

On the contrary, ImageNet pretrained features make the geometry of loss landscape much more predictable, and a step in gradient direction will lead to mild decrease of loss function.

To demonstrate the impact of transferred features on the stability of loss function, we further analyze the variation of loss in the direction of gradient in Figure 6 .

For each step in the training process, we compute the gradient of the loss and measure how the loss changes as we move the weight matrix in that direction.

We can clearly observe that in contrast to networks with transferred features, randomly initialized networks have larger variation along the gradient, where a step along the gradient leads to drastic change in the loss.

Why can transferred features control the magnitude of gradient and smooth the loss landscape?

A natural explanation is that transferred weight matrices provide appropriate transform of gradient in each layer and help stabilize its magnitude.

Note that in deep neural networks, the gradient w.r.t.

each layer is computed through back-propagation by

, where I k i denotes the activation of x i at layer k. The weight matrices W k function as the scaling factor of gradient in back-propagation.

Basically, a randomly initialized weight matrix will multiply the magnitude of gradient by its norm.

In pretrained weight matrices, situation is completely different.

To delve into this, we decompose the gradient into singular vectors and measure the projections of weight matrices in these principal directions.

Results are shown in Figure 7 (c).

During pretraining, the singular vectors of the gradient with large singular values are shrunk in the weight matrices.

Thus, the magnitude of gradient back-propagated through a pretrained layer is controlled.

In this sense, pretrained weight matrices stabilize the magnitude of gradient especially in lower layers.

We visualize the magnitude and scaling of gradient of different layers in ResNet-50 in Figure 7 .

The gradient of randomly initialized networks grows fast with layer numbers during back-propagation while the gradient of ImageNet pretrained networks remains stable.

Note that ResNet-50 already incorporates BatchNorm and skip-connections to improve the gradient flow, and pretrained representations can stabilize the magnitude of gradient substantially even in these modern networks.

We complete this analysis by visualizing the change of landscapes during back-propagation in Section B.4.

Transferring from pretrained representations boosts performance in a wide range of applications.

However, as discovered by He et al. (2018) ; Kornblith et al. (2019) , there still exist cases when pretrained representations provide no help for target tasks or even downgrade test accuracy.

Hence, the conditions on which transfer learning is feasible is an important open problem to be explored.

In this section, we delve into the feasibility of transfer learning with extensive experiments, while the theoretical perspectives are presented in the next section.

We hope our analysis will provide insights into how to adopt transfer learning by practitioners.

As a common practice, people choose datasets similar to the target dataset for pretraining.

However, how can we determine whether a dataset is sufficiently similar to a target dataset?

We verify with experiments that the similarity depends on the nature of tasks, i.e. both inputs and labels matter.

Varying input with fixed labels.

We randomly sample 600 images from the original SVHN dataset, and fine-tune the MNIST pretrained LeNet (LeCun et al., 1998) to this SVHN subset.

For comparison, we pretrain other two models on MNIST with images upside down and Fashion-MNIST (Xiao et al., 2017), respectively.

Note that for all three pretrained models, the dataset sizes, labels, and the number of images per class are kept exactly the same, and thus the only difference lies in the image pixels themselves.

Results are shown in Figure 8 (a).

Compared to training from scratch, MNIST pretrained features improve generalization significantly.

Upside-down MNIST shows slightly worse generalization performance than the original one.

In contrast, fine-tuning from Fashion-MNIST barely improves generalization.

We also compute the deviation from pretrained weight of each layer.

The weight matrices and convolutional kernel deviation of Fashion-MNIST pretraining show no improvement over training from scratch.

A reasonable implication here is that choosing a model pretrained on a more similar dataset in the inputs yields a larger performance gain.

Varying labels with fixed input.

We train a ResNet-50 model on Caltech-101 and then fine-tune it to Webcam (Saenko et al., 2010) .

We train another ResNet-50 to recognize the color of the upper part of Caltech-101 images and fine-tune it to Webcam.

Results in Figure 8 (b) indicate that the latter one provides no improvement over training on Webcam from scratch, while pretraining on standard Caltech-101 significantly boosts performance.

Models generalizing very well on similar images are not transferable to the target dataset with totally different labels.

These experiments challenge the common perspective of similarity between datasets.

The description of similarity using the input (images) themselves is just one point.

Another key factor of similarity is the relationship between the nature of tasks (labels).

This observation is further in line with our theoretical analysis in Section 6.

Currently, people usually train a model on ImageNet until it converges and use it as the pretrained parameters.

However, the final model do not necessarily have the highest transferability.

To see this, we pretrain a ResNet-50 model on Food-101 (Bossard et al., 2014) and transfer it to CUB-200, with results shown in Figure 9 .

During the early epochs, the transferability increases sharply.

As we continue pretraining, although the test accuracy on the pretraining dataset continues increasing, the test accuracy on the target dataset starts to decline, indicating downgraded transferability.

Intuitively, during the early epochs, the model learns general knowledge that is informative to many datasets.

As training goes on, however, the model starts to fit the specific knowledge of the pretrained dataset and even fit noise.

Such dataset-specific knowledge is usually detrimental to the transfer performance.

This interesting finding implies a promising direction for improving the de facto pretraining method: Instead of seeking for a model with higher accuracy only on the pretraining dataset, a more transferable model can be pretrained with appropriate epochs such that the fine-tuning accuracies on a diverse set of target tasks are advantageous.

Algorithms for pretraining should take this point into consideration.

We have shown through extensive empirical analysis that transferred features exert a fundamental impact on generalization and optimization performance, and provided some insights for the feasibility of transfer learning.

In this section, we analyze some of our empirical observations from a theoretical perspective.

We base our analysis on two-layer fully connected networks with ReLU activation and sufficiently many hidden units.

Our theoretical results are in line with the experimental findings.

Denote by ??(??) the ReLU activation function, ?? (z) = max{z, 0}. I{A} is the indicator function, i.e. I{A} = 1 if A is true and 0 otherwise.

[m] is the set of integers ranging from 1 to m. Consider a two-layer ReLU network of m hidden units

and W = (w 1 , ?? ?? ?? , w m ) ??? R d??m as the weight matrix.

We are provided with n Q samples {x Q,i , y Q,i } n Q i=1 drawn i.i.d.

from the target distribution Q as the target dataset and a weight matrix W(P ) pretrained on n P samples {x P,i , y P,i } n P i=1 drawn i.i.d.

from pretrained distribution P .

Suppose x 2 = 1 and |y| ??? 1.

Our goal is transferring the pretrained W(P ) to learn an accurate model W(Q) for the target distribution Q. When training the model on the pretraining dataset, we initialize the weight as: w r (0) ??? N (0, ?? 2 I), a r ???unif ({???1, 1}), where ???r ??? [m] and ?? is a constant.

For both pretraining and fine-tuning, the objective function of the model is the squared loss

Note that a is fixed throughout training and W is updated with gradient descent.

The learning rate is set to ??.

We base our analysis on the theoretical framework of Du et al. (2019) , since it provides elegant results on convergence of two-layer ReLU networks without strong assumptions on the input distributions, facilitating our extension to the transfer learning scenarios.

In our analysis, we use the Gram matrices H ??? P ??? R n P ??n P and H ??? Q ??? R n Q ??n Q to measure the quality of pretrained input and target input as

To quantify the relationship between pretrained input and target input, we define the following Gram matrix H ??? P Q ??? R n P ??n Q across samples drawn from P and Q:

Assume Gram matrices H ??? P and H ??? Q are invertible with smallest eigenvalue ?? P and ?? Q greater than zero.

H ??? P ???1 y P characterizes the labeling function of pretrained tasks.

y P ???Q H ??? P Q H ??? P ???1 y P further transforms the pretrained labeling function to the target labels.

A critical point in our analysis is y Q ??? y P ???Q , which measures the task similarity between target label and transformed label.

To analyze the Lipschitzness of loss function, a reasonable objective is the magnitude of gradient, which is a direct manifestation of the Lipschitz constant.

We analyze the gradient w.r.t.

the activations.

For the magnitude of gradient w.r.t.

the activations, we show that the Lipschitz constant is significantly reduced when the pretrained and target datasets are similar in both inputs and labels.

Theorem 1 (The effect of transferred features on the Lipschitzness of the loss).

Denote by X 1 the activations in the target dataset.

For a two-layer networks with sufficiently large number of hidden unit m defined in Section 6.1, if m ??? poly(n P , n Q , ?? ???1 , ?? ???1

with probability no less than 1 ??? ?? over the random initialization,

This provides us with theoretical explanation of experimental results in Section 4.

The control of Lipschitz constant relies on the similarity between tasks in both input and labels.

If the original target label is similar to the label transformed from the pretrained label, i.e. y Q ??? y P ???Q 2 2 is small, the Lipschitzness of loss function will be significantly improved.

On the contrary, if the pretrained and target tasks are completely different, the transformed label will be discrepant with target label, resulting in larger Lipschitz constant of the loss function and worse landscape in the fine-tuned model.

Recall in Section 3 that we have investigated the weight change W(Q) ??? W(P ) F during training and point out the role it plays in understanding the generalization.

In this section, we show that W(Q)???W(P ) F can be bounded with terms depicting the similarity between pretrained and target tasks.

Note that the Rademacher complexity of the function class is bounded with W(Q)???W(P ) F as shown in the seminal work (Arora et al., 2019) , thus the generalization error is directly related to W(Q) ??? W(P ) F .

We still use the Gram matrices defined in Section 6.1.

Theorem 2 (The effect of transferred features on the generalization error).

For a two-layer networks with m ??? poly(n P , n Q , ?? ???1 , ?? ???1

, with probability no less than 1 ??? ?? over the random initialization,

This result is directly related to the generalization error and casts light on our experiments in Section 5.1.

Note that when training on the target dataset from scratch, the upper bound of

By fine-tuning from a similar pretrained dataset where the transformed label is close to target label, the generalization error of the function class is hopefully reduced.

On the contrary, features pretrained on discrepant tasks do not transfer to classification task in spite of similar images since they have disparate labeling functions.

Another example is fine-tuning to Food-101 as in the experiment of Kornblith et al. (2019) .

Since it is a fine-grained dataset with many similar images, H ??? Q will be more singular than common tasks, resulting in a larger deviation from the pretrained weight.

Hence even transferring from ImageNet, the performance on Food-101 is still far from satisfactory.

Why are deep representations pretrained from modern neural networks generally transferable to novel tasks?

When is transfer learning feasible enough to consistently improve the target task performance?

These are the key questions in the way of understanding modern neural networks and applying them to a variety of real tasks.

This paper performs the first in-depth analysis of the transferability of deep representations from both empirical and theoretical perspectives.

The results reveal that pretrained representations will improve both generalization and optimization performance of a target network provided that the pretrained and target datasets are sufficiently similar in both input and labels.

With this paper, we show that transfer learning, as an initialization technique of neural networks, exerts implicit regularization to restrict the networks from escaping the flat region of pretrained landscape.

In this section, we provide details of the architectures, setup, methods of visualizations in our analysis.

The codes and visualizations are attached with the submission and will be made available online.

We implement all models on PyTorch with 2080Ti GPUs.

For object recognition and scene recognition tasks, we use standard ResNet-50 from torchvision.

ImageNet pretrained models can be found in torchvision, and Places pretrained models are provided by Zhou et al. (2018) .

During fine-tuning we use a batch size of 32 and set the initial learning rate to 0.01 with 0.9 momentum following the protocol of (Li et al., 2018b) .

For fine-tuning, we train the model for 200 epochs.

We decay the learning rate by 0.1 with the time of decay set by cross validation.

In Figure 2 (a) where the pretrained ResNet-50 functions as feature extractor, the downstream classifier is a two-layer ReLU network with Batch-Norm and Leaky-ReLU non-linearity.

The number of hidden unit is 512.

For this task, the backbone ResNet-50 is fixed, with the downstream two-layer classifier trained with momentum SGD.

The learning rate is set to 0.01 with 0.9 momentum, and remains constant throughout training.

For digit recognition tasks, we use LeNet (LeCun et al., 1998) .

The learning rate is also set to 0.01, with 5 ?? 10 ???4 weight decay.

The batch-size is set to 64.

We train the model for 100 epochs.

Fine-tuning.

We follow the protocol of fine-tuning as in the previous paragraphs.

In Tables 1  and 2 , we run all the experiments for 3 times and report their mean and variance.

For Table 2 , the improvement of fine-tuing is calculated with the generalization error of fine-tuning divided by the generalization error of training from scratch.

Visualization of loss landscapes.

We use techniques similar to filter normalization to provide an accurate analysis of loss landscapes (Li et al., 2018a) .

Note that ReLU networks are invariant to the scaling of weight parameters.

To remove this scaling effect, the direction used in visualization should be normalized in a filter-wise way.

Concretely, the axes of each landscape figure are two random Gaussian orthogonal vectors normalized by the scale of each filter in the convolutional layers.

Concretely, suppose the parameter of the center point is ??.

?? i,j denotes the j-th filter of the i-th layer.

Suppose the two unit orthogonal vectors are a and b. Then with filter normalization, a i,j ??? ai,j ai,j ?? i,j and b i,j ??? bi,j bi,j ?? i,j .

For each point (i.e. pixel) (p, q) in the plot, the value is evaluated with g(p, q) = L(f (?? + ??(pa + qb))), where L denotes the loss function, f denotes the neural networks.

?? is a parameter to control the scale of the plot.

In all visualization images of ResNet-50, the resolution is 200 ?? 200, i.e. p = ???100, ???99, ?? ?? ?? , 98, 99 and q = ???100, ???99, ?? ?? ?? , 98, 99.

For additional details of filter normalization, please refer to (Li et al., 2018a) .

?? is set to 0.001, which is of the same order as 10 times the step size in training.

This is a reasonable scale if we want to study the local loss landscape of model using SGD.

For fair comparison between the pretrained landscapes and randomly initialized landscapes, the scale of loss variation in each plot is exactly the same.

The difference of loss value between each contour is 0.05.

When we compute the loss landscape of one layer, the parameters of other layers are fixed.

The gradient is computed based on 256 fixed samples since the gradient w.r.t.

full dataset requires too much computation.

Figure 3 and Figure 10 are centered at the final weight parameters, while others are centered at the initialization point to show the situation when training just starts.

We visualize the loss landscapes on CUB-200, Stanford Cars and Food-101 for multiple times and reach consistent conclusions.

But due to space limitation, we only show the results on one dataset for each experiment in the main paper.

Other results are deferred to Section B.

Computing the eigenvalue of Hessian.

We compute the eigenvalues of Hessian with Hessianvector product and power methods based on the autograd of PyTorch.

A similar implementation is provided by Yao et al. (2018) .

We only list top 20 eigenvalues in limited space.

t-SNE embedding of model parameters.

We put the weight matrices of ResNet-50 in one vector as input.

For faster computation, we pre-compute the distance between parameters of every two models with PyTorch and then use the distance matrix to compute the t-SNE embedding with scikitlearn.

Note that we use the same ImageNet model from torchvision and the same Places model from Zhou et al. (2018) for fine-tuning.

Variation of loss function in the direction of gradient.

Based on the original trajectory of training, we take steps in the direction of gradient from parameters at different steps during training to calculate the maximum changes of loss in that direction.

The step size is set to the size of gradient.

We take 100 steps from the original trajectories to measure the local property of loss landscapes.

We aim to quantify the stability of loss functions and directly show the magnitude of gradient with this experiments on different datasets.

Results on CUB-200 are provided in the main paper, with additional results further provided in Section B. Not that this experiment is inspired by Santurkar et al. (2018) .

We use the similar protocol as Section 3.2 in Santurkar et al. (2018) .

Another protocol is to fix the step size along the gradient and compute the maximum variation of loss.

Results on Stanford Cars with this protocol are provided in Section B.2.

Results for both scenarios are similar.

Figure 11: Variation of the loss in ResNet-50 with ImageNet pretrained weight and random initialization.

We compare the variation of loss function in the direction of gradient during the training process on Stanford Cars dataset.

The variation of pretrained networks is substantially smaller than the randomly initialized one, implying a more desirable loss landscape and more stable optimization.

To validate that the generalization error is indeed improved with pretraining, for each dataset, we list the generalization error and the norm of deviation from the pretrained parameters in Table 2 .

The decreased percentage is calculated by dividing the error reduced in fine-tuning with the error of training from scratch.

Compared to the results of fine-tuning, we observe that ImageNet pretraining improves the generalization performance of general coarse-grained classification tasks significantly, yet the performance boost is smaller for fine-grained tasks which are dissimilar in the sense of task with ImageNet.

Note that, although Stanford Cars and CUB-200 are visually similar to ImageNet, what really matters is the similarity between the nature of tasks, i.e. both images and labels matter.

We visualize the loss landscape of 25-48th layers in ResNet-50 on Food-101 dataset.

We compare the landscapes centered at the initialization point of randomly initialized and ImageNet pretrained networks -see Figure 12 and Figure 13 .

Results are in line with our observations of the magnitude of gradient in Figure 7 .

At higher layers, the landscapes of random initialization and ImageNet pretraining are similar.

However, as the gradient is back-propagated through lower layers, the landscapes of pretrained networks remain as smooth as the higher layers.

In sharp contrast, the landscapes of randomly initialized networks worsen through the lower layers, indicating that the magnitude of gradient is substantially worsened in back-propagation.

Figure 12: Landscapes centered at the initialization point of each layer in ResNet-50 using ImageNet pretrained weight.

The smoothness of landscapes in each layer are nearly identical, indicating a proper scaling of gradient.

Figure 13: Landscapes centered at the initialization point of each layer in ResNet-50 initialized randomly.

At the higher layers, the landscapes tend to be smooth.

However, as the gradient is propagated to lower layers, the landscapes are becoming full of ridges and trenches in spite of the presence of Batch-Norm and skip connections.

To study how transferring pretrained knowledge helps target tasks, we first study the trajectories of weight matrices during pretraining and then analyze its effect as an initialization in target tasks.

Our analysis is based on Du et al. (2019)'s framework for over-parametrized networks.

For the weight matrix W, W(0) denotes the random initialization.

W P (k) denotes W at the kth step of pretraining.

W(P ) denotes the pretrained weight matrix after training K steps.

W Q (k) denotes the weight matrix after K steps of fine-tuning from W(P ).

For other terms, the notation at each step is similar.

We first analyze the pretraining process on the source datasets based on Arora et al. (2019) .

Define a matrix Z P ??? R md??n P which is crucial to analyzing the trajectories of the weight matrix during pretraining,

where I P i,j = I{w i x P,j ??? 0}. Z P (k) denotes the matrix corresponding to W P (k).

Note that the gradient descent is carried out as

where vec (??) denotes concatenating a column of a matrice into a single vector.

Then in the K iterations of pretraining on the source dataset,

(5) The first term is the primary component in the pretrained matrix, while the second and third terms is small under the over-parametrized conditions.

Now following Arora et al. (2019) , the magnitude of these terms can be bounded with probability no less than 1 ??? ??,

(6) Here we also provide lemmas from Du et al. (2019) which are extensively used later.

, with probability at least 1 ??? ?? over the random initialization we have

Lemma 2.

If w 1 , . . . , w m are i.i.d.

generated from N (0, I), then with probability at least 1 ??? ??, the following holds.

For any set of weight vectors w 1 , . . . , w m ??? R d that satisfy for any r ??? [m], w r (0) ??? w r 2 ??? c????0 n 2 R for some small positive constant c, then the matrix H ??? R n??n defined by

Now we start to analyze the influence of pretrained weight on target tasks.

1) We show that during pretraining,

2) Then we analyze u Q (P )???u Q (0) with the properties of H ??? P Q .

3) Standard calculation shows the magnitude of gradient relates closely to u Q (P ) ??? u Q (0), and we are able to find out how is the magnitude of gradient improved.

To start with, we analyze the properties of the matrix H ??? P Q .

We show that under over-parametrized conditions, H ??? P Q is close to the randomly initialized Gram matrix Z P (0) Z Q (P ).

Use H P Q (0) to denote Z P (0) Z Q (0), and H P Q (P ) to denote Z P (0) Z Q (P ).

Lemma 3.

With the same condition as lemma 1, with probability no less than 1 ??? ??,

where

with a small c. Since w r (0) is independent of x Q,i and x Q,i 2 = 1, the distribution of w(0) r x Q,i and w r (0) are the same Gaussian.

Applying Markov's inequality, and noting that

we have with probability no less than 1 ??? ??,

Also note that E[H P Q,ij (0)] = H ??? P Q,ij .

By Hoeffding's inequality, we have with probability at least 1 ??? ??,

Combining equation 12 and equation 11, we have with probability at least 1 ??? ??,

Denote by u Q (P ), u Q (0) the output on the target dataset using weight matrix W(P ) and W 0 respectively.

First, we compute the gradient with respect to the activations,

It is obvious from equation 14 that u Q (P ) ??? u Q (0) should become the focus of our analysis.

To calculate u Q (P ) ??? u Q (0), we need to sort out how the activations change by initializing the target networks with W(P ) instead of W(0).

For each x Q,i , divide r into two sets to quantify the change of variation in activations on the target dataset.

where

For r in S i , we can estimate the size of S i .

Note that

, since the distribution of w(0) r is Gaussian with mean 0 and covariance matrix ?? 2 I. Therefore, taking sum over all i and m and using Markov inequality, with probability at least 1 ??? ?? over the random initialization we have

Thus, this part of activations is the same for W(0) and W(P ) on the target dataset.

For each x Q,i ,

where

The first term is the primary part, while we can show that the second and the third term can be bounded with

where 1 and 2 correspond to each of the second term and third term in equation 18.

Thus, using lemma 1 and the estimation of |S i |, with probability no less than 1 ??? ??,

Now equipped with equation 6, equation 19, equation 20 and lemma 3, we are ready to calculate exactly how much pretrained wight matrix W(P ) help reduce the magnitude of gradient over W(0),

, and u Q (0) 2 are all small values we have estimated above.

Therefore, using Z P (0) F ??? ??? n P and

we can control the magnitude of the perturbation terms under over-parametrized conditions.

Concretely, with probability at least 1 ??? ?? over random initialization,

Substituting these estimations into equation 21 completes the proof of Theorem 1.

In this subsection, we analyze the impact of pretrained weight matrix on the generalization performance.

First, we show that a model will converge if initialized with pretrained weight matrix.

Based on this, we further investigate the trajectories during transfer learning and bound W ??? W(P ) F with the relationship between source and target datasets.

we set the number of hidden nodes m = ???

, and the learning

then with probability at least 1 ??? ?? over the random initialization we have for k = 0, 1, 2, . . .

The following lemma is a direct corollary of Theorem 3 and lamma 1, and is crucial to analysis to follow.

Lemma 4.

Under the same conditions as Theorem 3, with probability at least 1 ??? ?? over the random initialization we have ???r ??? [m], ???k ??? 0,

We have the estimation of w Q,r (k) ??? w r (0) 2 from lemma 1.

From w Q,r (k) ??? w r (0) 2 ??? w Q,r (k) ??? w r (P ) 2 + w r (P ) ??? w r (0) 2 , we can proove lemma 4 by estimating w Q,r (k) ??? w r (P ) 2 .

, and u Q (P ) ??? u Q (0) = Z Q (0) (Z P (0)H ??? P ???1 y P + ) + 1 + 2 .

Substituting lemma 3, equation 6, and equation 22 into u Q (P ) ??? u Q (0) 2 completes the proof.

Now we start to prove Theorem 3 by induction.

We have the following corollary if condition 1 holds, Corollary 1.

If condition 1 holds for k = 0, . . .

, k, for every r ??? [m], with probability at least 1 ??? ??, w Q,r (k) ??? w r (0) 2 ??? 4 ??? n P y P ??? u P (0) 2 ??? m?? P + 4 ??? n Q y Q ??? u Q (P ) 2 ??? m?? Q R .

If k = 0, by definition Condition 1 holds.

Suppose for k = 0, . . .

, k, condition 1 holds and we want to show it still holds for k = k + 1.

The strategy is similar to the proof of convergence on training from scratch.

By classifying the change of activations into two categories, we are able to deal with the ReLU networks as a perturbed version of linear regression.

We define the event A ir = w r (0) x Q,i ??? R , where R =

where we notice max k>0 k 1 ???

The first term is the primary part, while the second and the third are considered perturbations and could be controlled using lemma 6 and equation 38.

since Z Q (k) ??? Z Q (0) F is bounded, the maximum eigenvalue of H ??? Q is ?? ???1

which completes the proof.

<|TLDR|>

@highlight

Understand transferability from the perspectives of improved generalization, optimization and the feasibility of transferability.