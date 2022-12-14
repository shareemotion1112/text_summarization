Data augmentation (DA) has been widely utilized to improve generalization in training deep neural networks.

Recently, human-designed data augmentation has been gradually replaced by automatically learned augmentation policy.

Through finding the best policy in well-designed search space of data augmentation, AutoAugment (Cubuk et al., 2019) can significantly improve validation accuracy on image classification tasks.

However, this approach is not computationally practical for large-scale problems.

In this paper, we develop an adversarial method to arrive at a computationally-affordable solution called Adversarial AutoAugment, which can simultaneously optimize target related object and augmentation policy search loss.

The augmentation policy network attempts to increase the training loss of a target network through generating adversarial augmentation policies, while the target network can learn more robust features from harder examples to improve the generalization.

In contrast to prior work, we reuse the computation in target network training for policy evaluation, and dispense with the retraining of the target network.

Compared to AutoAugment, this leads to about 12x reduction in computing cost and 11x shortening in time overhead on ImageNet.

We show experimental results of our approach on CIFAR-10/CIFAR-100, ImageNet, and demonstrate significant performance improvements over state-of-the-art.

On CIFAR-10, we achieve a top-1 test error of 1.36%, which is the currently best performing single model.

On ImageNet, we achieve a leading performance of top-1 accuracy 79.40% on ResNet-50 and 80.00% on ResNet-50-D without extra data.

Massive amount of data have promoted the great success of deep learning in academia and industry.

The performance of deep neural networks (DNNs) would be improved substantially when more supervised data is available or better data augmentation method is adapted.

Data augmentation such as rotation, flipping, cropping, etc., is a powerful technique to increase the amount and diversity of data.

Experiments show that the generalization of a neural network can be efficiently improved through manually designing data augmentation policies.

However, this needs lots of knowledge of human expert, and sometimes shows the weak transferability across different tasks and datasets in practical applications.

Inspired by neural architecture search (NAS) (Zoph & Le, 2016; Zoph et al., 2017; Zhong et al., 2018a; b; Guo et al., 2018) , a reinforcement learning (RL) (Williams, 1992) method called AutoAugment is proposed by Cubuk et al. (2019) , which can automatically learn the augmentation policy from data and provide an exciting performance improvement on image classification tasks.

However, the computing cost is huge for training and evaluating thousands of sampled policies in the search process.

Although proxy tasks, i.e., smaller models and reduced datasets, are taken to accelerate the searching process, tens of thousands of GPU-hours of consumption are still required.

In addition, these data augmentation policies optimized on proxy tasks are not guaranteed to be optimal on the target task, and the fixed augmentation policy is also sub-optimal for the whole training process.

Figure 1: The overview of our proposed method.

We formulate it as a Min-Max game.

The data of each batch is augmented by multiple pre-processing components with sampled policies {?? 1 , ?? 2 , ?? ?? ?? , ?? M }, respectively.

Then, a target network is trained to minimize the loss of a large batch, which is formed by multiple augmented instances of the input batch.

We extract the training losses of a target network corresponding to different augmentation policies as the reward signal.

Finally, the augmentation policy network is trained with the guideline of the processed reward signal, and aims to maximize the training loss of the target network through generating adversarial policies.

In this paper, we propose an efficient data augmentation method to address the problems mentioned above, which can directly search the best augmentation policy on the full dataset during training a target network, as shown in Figure 1 .

We first organize the network training and augmentation policy search in an adversarial and online manner.

The augmentation policy is dynamically changed along with the training state of the target network, rather than fixed throughout the whole training process like normal AutoAugment (Cubuk et al., 2019) .

Due to reusing the computation in policy evaluation and dispensing with the retraining of the target network, the computing cost and time overhead are extremely reduced.

Then, the augmentation policy network is taken as an adversary to explore the weakness of the target network.

We augment the data of each min-batch with various adversarial policies in parallel, rather than the same data augmentation taken in batch augmentation (BA) (Hoffer et al., 2019) .

Then, several augmented instances of each mini-batch are formed into a large batch for target network learning.

As an indicator of the hardness of augmentation policies, the training losses of the target network are used to guide the policy network to generate more aggressive and efficient policies based on REINFORCE algorithm (Williams, 1992) .

Through adversarial learning, we can train the target network more efficiently and robustly.

The contributions can be summarized as follows:

??? Our method can directly learn augmentation policies on target tasks, i.e., target networks and full datasets, with a quite low computing cost and time overhead.

The direct policy search avoids the performance degradation caused by the policy transfer from proxy tasks to target tasks.

??? We propose an adversarial framework to jointly optimize target network training and augmentation policy search.

The harder samples augmented by adversarial policies are constantly fed into the target network to promote robust feature learning.

Hence, the generalization of the target network can be significantly improved.

??? The experiment results show that our proposed method outperforms previous augmentation methods.

For instance, we achieve a top-1 test error of 1.36% with PyramidNet+ShakeDrop (Yamada et al., 2018 ) on CIFAR-10, which is the state-of-the-art performance.

On ImageNet, we improve the top-1 accuracy of ResNet-50 (He et al., 2016) from 76.3% to 79.4% without extra data, which is even 1.77% better than AutoAugment (Cubuk et al., 2019) .

Common data augmentation, which can generate extra samples by some label-preserved transformations, is usually used to increase the size of datasets and improve the generalization of networks, such as on MINST, CIFAR-10 and ImageNet (Krizhevsky et al., 2012; Wan et al., 2013; Szegedy et al., 2015) .

However, human-designed augmentation policies are specified for different datasets.

For example, flipping, the widely used transformation on CIFAR-10/CIFAR-100 and ImageNet, is not suitable for MINST, which will destroy the property of original samples.

Hence, several works (Lemley et al., 2017; Cubuk et al., 2019; Lin et al., 2019; Ho et al., 2019) have attempted to automatically learn data augmentation policies.

Lemley et al. (2017) propose a method called Smart Augmentation, which merges two or more samples of a class to improve the generalization of a target network.

The result also indicates that an augmentation network can be learned when a target network is being training.

Through well designing the search space of data augmentation policies, AutoAugment (Cubuk et al., 2019) takes a recurrent neural network (RNN) as a sample controller to find the best data augmentation policy for a selected dataset.

To reduce the computing cost, the augmentation policy search is performed on proxy tasks.

Population based augmentation (PBA) (Ho et al., 2019) replaces the fixed augmentation policy with a dynamic schedule of augmentation policy along with the training process, which is mostly related to our work.

Inspired by population based training (PBT) (Jaderberg et al., 2017) , the augmentation policy search problem in PBA is modeled as a process of hyperparameter schedule learning.

However, the augmentation schedule learning is still performed on proxy tasks.

The learned policy schedule should be manually adjusted when the training process of a target network is non-matched with proxy tasks.

Another related topic is Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) , which has recently attracted lots of research attention due to its fascinating performance, and also been used to enlarge datasets through directly synthesizing new images (Tran et al., 2017; Perez & Wang, 2017; Antoniou et al., 2017; Gurumurthy et al., 2017; Frid-Adar et al., 2018) .

Although we formulate our proposed method as a Min-Max game, there exists an obvious difference with traditional GANs.

We want to find the best augmentation policy to perform image transformation along with the training process, rather than synthesize new images.

Peng et al. (2018) also take such an idea to optimize the training process of a target network in human pose estimation.

In this section, we present the implementation of Adversarial AutoAugment.

First, the motivation for the adversarial relation between network learning and augmentation policy is discussed.

Then, we introduce the search space with the dynamic augmentation policy.

Finally, the joint framework for network training and augmentation policy search is presented in detail.

Although some human-designed data augmentations have been used in the training of DNNs, such as randomly cropping and horizontally flipping on CIFAR-10/CIFAR-100 and ImageNet, limited randomness will make it very difficult to generate effective samples at the tail end of the training.

To struggle with the problem, more randomness about image transformation is introduced into the search space of AutoAugment (Cubuk et al., 2019) (described in Section 3.2).

However, the learned policy is fixed for the entire training process.

All of possible instances of each example will be send to the target network repeatedly, which still results in an inevitable overfitting in a long-epoch training.

This phenomenon indicates that the learned policy is not adaptive to the training process of a target network, especially found on proxy tasks.

Hence, the dynamic and adversarial augmentation policy with the training process is considered as the crucial feature in our search space.

Another consideration is how to improve the efficiency of the policy search.

In AutoAugment (Cubuk et al., 2019) , to evaluate the performance of augmentation policies, a lot of child models should be trained from scratch nearly to convergence.

The computation in training and evaluating the performance of different sampled policies can not be reused, which leads to huge waste of computation resources.

In this paper, we propose a computing-efficient policy search framework through reusing prior computation in policy evaluation.

Only one target network is used to evaluate the performance of different policies with the help of the training losses of corresponding augmented instances.

The augmentation policy network is learned from the intermediate state of the target network, which makes generated augmentation policies more aggressive and adaptive.

On the contrary, to combat harder examples augmented by adversarial policies, the target network has to learn more robust features, which makes the training more efficiently.

In this paper, the basic structure of the search space of AutoAugment (Cubuk et al., 2019 ) is reserved.

An augmentation policy is defined as that it is composed by 5 sub-policies, each sub-policy contains two image operations to be applied orderly, each operation has two corresponding parameters, i.e., the probability and magnitude of the operation.

Finally, the 5 best policies are concatenated to form a single policy with 25 sub-policies.

For each image in a mini-batch, only one sub-policy will be randomly selected to be applied.

To compare with AutoAugment (Cubuk et al., 2019) conveniently, we just slightly modify the search space with removing the probability of each operation.

This is because that we think the stochasticity of an operation with a probability requires a certain epochs to take effect, which will detain the feedback of the intermediate state of the target network.

There are totally 16 image operations in our search space, including ShearX/Y, TranslateX/Y, Rotate, AutoContrast, Invert, Equalize, Solarize, Posterize, Contrast, Color, Brightness, Sharpness, Cutout (Devries & Taylor, 2017) and Sample Pairing (Inoue, 2018) .

The range of the magnitude is also discretized uniformly into 10 values.

To guarantee the convergence during adversarial learning, the magnitude of all the operations are set in a moderate range.

1 Besides, the randomness during the training process is introduced into our search space.

Hence, the search space of the policy in each epoch has |S| = (16??10) 10 ??? 1.1??10 22 possibilities.

Considering the dynamic policy, the number of possible policies with the whole training process can be expressed as |S|

#epochs .

An example of dynamically learning the augmentation policy along with the training process is shown in Figure  2 .

We observe that the magnitude (an indication of difficulty) gradually increases with the training process.

In this section, the adversarial framework of jointly optimizing network training and augmentation policy search is presented in detail.

We use the augmentation policy network A(??, ??) as an adversary, which attempts to increase the training loss of the target network F(??, w) through adversarial learning.

The target network is trained by a large batch formed by multiple augmented instances of each batch to promote invariant learning (Salazar et al., 2018) , and the losses of different augmentation policies applied on the same data are used to train the augmentation policy network by RL algorithm.

Considering the target network F(??, w) with a loss function L[F(x, w), y], where each example is transformed by some random data augmentation o(??), the learning process of the target network can be defined as the following minimization problem

where ??? is the training set, x and y are the input image and the corresponding label, respectively.

The problem is usually solved by vanilla SGD with a learning rate ?? and batch size N , and the training procedure for each batch can be expressed as

To improve the convergence performance of DNNs, more random and efficient data augmentation is performed under the help of the augmentation policy network.

Hence, the minimization problem should be slightly modified as

where ?? (??) represents the augmentation policy generated by the network A(??, ??).

Accordingly, the training rule can be rewritten as

where we introduce M different instances of each input example augmented by adversarial policies {?? 1 , ?? 2 , ?? ?? ?? , ?? M }.

For convenience, we denote the training loss of a mini-batch corresponding to the augmentation policy ?? m as

Hence, we have an equivalent form of Equation 4

Note that the training procedure can be regarded as a larger N ?? M batch training or an average over M instances of gradient computation without changing the learning rate, which will lead to a reduction of gradient variance and a faster convergence of the target network Hoffer et al. (2019) .

However, overfitting will also come.

To overcome the problem, the augmentation policy network is designed to increase the training loss of the target network with harder augmentation policies.

Therefore, we can mathematically express the object as the following maximization problem

where

Similar to AutoAugment (Cubuk et al., 2019) , the augmentation policy network is also implemented as a RNN shown in Figure 3 .

At each time step of the RNN controller, the softmax layer will Figure 3 : The basic architecture of the controller for generating a sub-policy, which consists of two operations with corresponding parameters, the type and magnitude of each operation.

When a policy contains Q sub-policies, the basic architecture will be repeated Q times.

Following the setting of AutoAugment (Cubuk et al., 2019) , the number of sub-policies Q is set to 5 in this paper.

predict an action corresponding to a discrete parameter of a sub-policy, and then an embedding of the predicted action will be fed into the next time step.

In our experiments, the RNN controller will predict 20 discrete parameters to form a whole policy.

However, there has a severe problem in jointly optimizing target network training and augmentation policy search.

This is because that non-differentiable augmentation operations break gradient flow from the target network F to the augmentation policy network A Peng et al., 2018) .

As an alternative approach, REINFORCE algorithm (Williams, 1992 ) is applied to optimize the augmentation policy network as

where p m represents the probability of the policy ?? m .

To reduce the variance of gradient ??? ?? J(??), we replace the training loss of a mini-batch L m with L m a moving average over a certain minibatches 2 , and then normalize it among M instances as L m .

Hence, the training procedure of the augmentation policy network can be expressed as

The adversarial learning of target network training and augmentation policy search is summarized as Algorithm 1.

In this section, we first reveal the details of experiment settings.

Then, we evaluate our proposed method on CIFAR-10/CIFAR-100, ImageNet, and compare it with previous methods.

Results in Figure 4 show our method achieves the state-of-the-art performance with higher computing and time efficiency 3 .

Initialization: target network F(??, w), augmentation policy network A(??, ??) Input: input examples x, corresponding labels y 1: for 1 ??? e ??? epochs do 2:

Initialize L m = 0, ???m ??? {1, 2, ?? ?? ?? , M };

Generate M policies with the probabilities {p 1 , p 2 , ?? ?? ?? , p M };

4:

Augment each batch data with M generated policies, respectively; 6:

Update w e,t+1 according to Equation 4; 7:

Update L m through moving average, ???m ??? {1, 2, ?? ?? ?? , M };

Normalize L m among M instances as L m , ???m ??? {1, 2, ?? ?? ?? , M };

10:

Update ?? e+1 via Equation 9; 11: Output w * , ?? *

The RNN controller is implemented as a one-layer LSTM (Hochreiter & Schmidhuber, 1997) .

We set the hidden size to 100, and the embedding size to 32.

We use Adam optimizer (Kingma & Ba, 2015) with a initial learning rate 0.00035 to train the controller.

To avoid unexpected rapid convergence, an entropy penalty of a weight of 0.00001 is applied.

All the reported results are the mean of five runs with different initializations.

CIFAR-10 dataset (Krizhevsky & Hinton, 2009 ) has totally 60000 images.

The training and test sets have 50000 and 10000 images, respectively.

Each image in size of 32 ?? 32 belongs to one of 10 classes.

We evaluate our proposed method with the following models: Wide-ResNet-28-10 (Zagoruyko & Komodakis, 2016), Shake-Shake (26 2x32d) (Gastaldi, 2017) , Shake-Shake (26 2x96d) (Gastaldi, 2017) , Shake-Shake (26 2x112d) (Gastaldi, 2017) , PyramidNet+ShakeDrop (Han et al., 2017; Yamada et al., 2018) .

All the models are trained on the full training set.

The Baseline is trained with the standard data augmentation, namely, randomly cropping a part of 32 ?? 32 from the padded image and horizontally flipping it with a probability of 0.5.

The Cutout (Devries & Taylor, 2017) randomly select a 16 ?? 16 patch of each image, and then set the pixels of the selected patch to zeros.

For our method, the searched policy is applied in addition to standard data augmentation and Cutout.

For each image in the training process, standard data augmentation, the searched policy and Cutout are applied in sequence.

For Wide-ResNet-28-10, the step learning rate (LR) schedule is adopted.

The cosine LR schedule is adopted for the other models.

More details about model hyperparameters are supplied in A.1.

To choose the optimal M , we select Wide-ResNet-28-10 as a target network, and evaluate the performance of our proposed method verse different M , where M ??? {2, 4, 8, 16, 32}. From Figure 5 , we can observe that the test accuracy of the model improves rapidly with the increase of M up to 8.

The further increase of M does not bring a significant improvement.

Therefore, to balance the performance and the computing cost, M is set to 8 in all the following experiments.

In Table 1 , we report the test error of these models on CIFAR-10.

For all of these models, our proposed method can achieve better performance compared to previous methods.

We achieve 0.78% and 0.68% improvement on Wide-ResNet-28-10 compared to AutoAugment and PBA, respectively.

We achieve a top-1 test error of 1.36% with PyramidNet+ShakeDrop, which is 0.1% better than the current state-of-the-art reported in Ho et al. (2019) .

As shown in Figure 6 (a) and 6(b),we further visualize the probability distribution of the parameters of the augmentation policies learned with PyramidNet+ShakeDrop on CIFAR-10 over time.

From Figure 6 (a), we can find that the percentages of some operations, such as TranslateY, Rotate, Posterize, and SampleParing, gradually increase along with the training process.

Meanwhile, more geometric transformations, such as TranslateX, TranslateY, and Rotate, are picked in the sampled augmentation policies, which is different from color-focused AutoAugment (Cubuk et al., 2019) on CIFAR-10.

that large magnitudes gain higher percentages during training.

However, at the tail of training, low magnitudes remain considerable percentages.

This indicates that our method does not simply learn the transformations with the extremes of the allowed magnitudes to spoil the target network.

We also evaluate our proposed method on CIFAR-100, as shown in Table 2 .

As we can observe from the table, we also achieve the state-of-the-art performance on this dataset.

As a great challenge in image recognition, ImageNet dataset (Deng et al., 2009 ) has about 1.2 million training images and 50000 validation images with 1000 classes.

In this section, we directly search the augmentation policy on the full training set and train ResNet-50 (He et al., 2016) , ResNet-50-D (He et al., 2018) and ResNet-200 (He et al., 2016 ) from scratch.

Training details: For the baseline augmentation, we randomly resize and crop each input image to a size of 224 ?? 224, and then horizontally flip it with a probability of 0.5.

For AutoAugment (Cubuk et al., 2019) and our method, the baseline augmentation and the augmentation policy are both used for each image.

The cosine LR schedule is adopted in the training process.

The model hyperparameters on ImageNet is also detailed in A.1.

ImageNet results: The performance of our proposed method on ImageNet is presented in Table 3 .

It can be observed that we achieve a top-1 accuracy 79.40% on ResNet-50 without extra data.

To the best of our knowledge, this is the highest top-1 accuracy for ResNet-50 learned on ImageNet.

Besides, we only replace the ResNet-50 architecture with ResNet-50-D, and achieve a consistent improvement with a top-1 accuracy of 80.00%.

To check the effect of each component in our proposed method, we report the test error of ResNet-50 on ImageNet the following augmentation methods in Table 4 .

??? Baseline: Training regularly with the standard data augmentation and step LR schedule.

??? Fixed: Augmenting all the instances of each batch with the standard data augmentation fixed throughout the entire training process.

??? Random:

Augmenting all the instances of each batch with randomly and dynamically generated policies.

??? Ours: Augmenting all the instances of each batch with adversarial policies sampled by the policy network along with the training process.

From the table, we can find that Fixed can achieve 0.99% error reduction compared to Baseline.

This shows that a large-batch training with multiple augmented instances of each mini-batch can indeed improve the generalization of the model, which is consistent with the conclusion presented in Hoffer et al. (2019) .

In addition, the test error of Random is 1.02% better than Fixed.

This indicates that augmenting batch with randomly generated policies can reduce overfitting in a certain extent.

Furthermore, our method achieves the best test error of 20.60% through augmenting samples with adversarial policies.

From the result, we can conclude that these policies generated by the policy network are more adaptive to the training process, and make the target network have to learn more robust features.

Computing Cost: The computation in target network training is reused for policy evaluation.

This makes the computing cost in policy search become negligible.

Although there exists an increase of computing cost in target network training, the total computing cost in training one target network with augmentation policies is quite small compared to prior work.

Time Overhead: Since we just train one target network with a large batch distributedly and simultaneously, the time overhead of the large-batch training is equal to the regular training.

Meanwhile, the joint optimization of target network training and augmentation policy search dispenses with the process of offline policy search and the retraining of a target network, which leads to a extreme time overhead reduction.

In Table 5 , we take the training of ResNet-50 on ImageNet as an example to compare the computing cost and time overhead of our method and AutoAugment.

From the table, we can find that our method is 12?? less computing cost and 11?? shorter time overhead than AutoAugment.

To further show the higher efficiency of our method, the transferability of the learned augmentation policies is evaluated in this section.

We first take a snapshot of the adversarial training process of ResNet-50 on ImageNet, and then directly use the learned dynamic augmentation policies to regularly train the following models: Wide-ResNet-28-10 on CIFAR-10/100, ResNet-50-D on ImageNet and ResNet200 on ImageNet.

Table 6 presents the experimental results of the transferability.

From the table, we can find that a competitive performance can be still achieved through direct policy transfer.

This indicates that the learned augmentation policies transfer well across datasets and architectures.

However, compared to the proposed method, the policy transfer results in an obvious performance degradation, especially the transfer across datasets.

In this paper, we introduce the idea of adversarial learning into automatic data augmentation.

The policy network tries to combat the overfitting of the target network through generating adversarial policies with the training process.

To oppose this, robust features are learned in the target network, which leads to a significant performance improvement.

Meanwhile, the augmentation policy search is performed along with the training of a target network, and the computation in network training is reused for policy evaluation, which can extremely reduce the search cost and make our method more computing-efficient.

<|TLDR|>

@highlight

We introduce the idea of adversarial learning into automatic data augmentation to improve the generalization  of a targe network.

@highlight

A technique called Adversarial AutoAugment which dynamically learns good data augmentation policies during training using an adversarial approach.