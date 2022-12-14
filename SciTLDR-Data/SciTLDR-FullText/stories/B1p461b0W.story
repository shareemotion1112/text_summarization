Deep neural networks trained on large supervised datasets have led to impressive results in recent years.

However, since well-annotated datasets can be prohibitively expensive and time-consuming to collect, recent work has explored the use of larger but noisy datasets that can be more easily obtained.

In this paper, we investigate the behavior of deep neural networks on training sets with massively noisy labels.

We show on multiple datasets such as MINST, CIFAR-10 and ImageNet that successful learning is possible even with an essentially arbitrary amount of noise.

For example, on MNIST we find that accuracy of above 90 percent is still attainable even when the dataset has been diluted with 100 noisy examples for each clean example.

Such behavior holds across multiple patterns of label noise, even when noisy labels are biased towards confusing classes.

Further, we show how the required dataset size for successful training increases with higher label noise.

Finally, we present simple actionable techniques for improving learning in the regime of high label noise.

Deep learning has proven to be powerful for a wide range of problems, from image classification to machine translation.

Typically, deep neural networks are trained using supervised learning on large, carefully annotated datasets.

However, the need for such datasets restricts the space of problems that can be addressed.

This has led to a proliferation of deep learning results on the same tasks using the same well-known datasets.

Carefully annotated data is difficult to obtain, especially for classification tasks with large numbers of classes (requiring extensive annotation) or with fine-grained classes (requiring skilled annotation).

Thus, annotation can be expensive and, for tasks requiring expert knowledge, may simply be unattainable at scale.

To address this limitation, other training paradigms have been investigated to alleviate the need for expensive annotations, such as unsupervised learning BID11 , self-supervised learning BID16 BID23 and learning from noisy annotations (Joulin et al., 2016; BID15 BID22 .

Very large datasets (e.g., BID7 ; BID19 ) can often be attained, for example from web sources, with partial or unreliable annotation.

This can allow neural networks to be trained on a much wider variety of tasks or classes and with less manual effort.

The good performance obtained from these large noisy datasets indicates that deep learning approaches can tolerate modest amounts of noise in the training set.

In this work, we take this trend to an extreme, and consider the performance of deep neural networks under extremely low label reliability, only slightly above chance.

We envision a future in which arbitrarily large amounts of data will easily be obtained, but in which labels come without any guarantee of validity and may merely be biased towards the correct distribution.

The key takeaways from this paper may be summarized as follows:??? Deep neural networks are able to learn from data that has been diluted by an arbitrary amount of noise.

We demonstrate that standard deep neural networks still perform well even on training sets in which label accuracy is as low as 1 percent above chance.

On MNIST, for example, performance still exceeds 90 percent even with this level of label noise (see Figure 1 ).

This behavior holds, to varying extents, across datasets as well as patterns of label noise, including when noisy labels are biased towards confused classes.??? A sufficiently large training set can accommodate a wide range of noise levels.

We find that the minimum dataset size required for effective training increases with the noise level.

A large enough training set can accommodate a wide range of noise levels.

Increasing the dataset size further, however, does not appreciably increase accuracy.??? Adjusting batch size and learning rate can allow conventional neural networks to operate in the regime of very high label noise.

We find that label noise reduces the effective batch size, as noisy labels roughly cancel out and only a small learning signal remains.

We show that dataset noise can be partly compensated for by larger batch sizes and by scaling the learning rate with the effective batch size.

Learning from noisy data.

Several studies have investigated the impact of noisy datasets on machine classifiers.

Approaches to learn from noisy data can generally be categorized into two groups:In the first group, approaches aim to learn directly from noisy labels and focus on noise-robust algorithms, e.g., BID0 BID3 Joulin et al. (2016) ; BID8 ; BID13 ; BID14 ; BID20 .

The second group comprises mostly label-cleansing methods that aim to remove or correct mislabeled data, e.g., BID1 .

Methods in this group frequently face the challenge of disambiguating between mislabeled and hard training examples.

To address this challenge, they often use semi-supervised approaches by combining noisy data with a small set of clean labels BID27 .

Some approaches model the label noise as conditionally independent from the input image BID15 BID17 and some propose image-conditional noise models BID22 BID24 .

Our work differs from these approaches in that we do not aim to clean the training dataset or propose new noise-robust training algorithms.

Instead, we study the behavior of standard neural network training procedures in settings with massive label noise.

We show that even without explicit cleaning or noise-robust algorithms, neural networks can learn from data that has been diluted by an arbitrary amount of label noise.

Analyzing the robustness of neural networks.

Several investigative studies aim to improve our understanding of convolutional neural networks.

One particular stream of research in this space seeks to investigate neural networks by analyzing their robustness.

For example, show that network architectures with residual connections have a high redundancy in terms of parameters and are robust to the deletion of multiple complete layers during test time.

Further, BID18 investigate the robustness of neural networks to adversarial examples.

They show that even for fully trained networks, small changes in the input can lead to large changes in the output and thus misclassification.

In contrast, we are focusing on non-adversarial noise during training time.

Within this stream of research, closest to our work are studies that focus on the impact of noisy training datasets on classification performance (e.g., BID17 BID20 ; BID26 ).

In these studies an increase in noise is assumed to decrease not only the proportion of correct examples, but also their absolute number.

In contrast to these studies, we separate the effects and show in ??4 that a decrease in the number of correct examples is more destructive to learning than an increase in the number of noisy labels.

In this work, we are concerned with scenarios of abundant data of very poor label quality, i.e., the regime in which falsely labeled training examples vastly outnumber correctly labeled examples.

In particular, our experiments involve observing the performance of deep neural networks on multiclass classification tasks as label noise is increased.

To formalize the problem, we denote the number of original training examples by n. To model the amount of noise, we dilute the dataset by adding ?? noisy examples to the training set for each original training example.

Thus, the total number of noisy labels in the training set is ??n.

Note that by varying the noise level ??, we do not change the available number of original examples.

Thus, even in the presence of high noise, there is still appreciable data to learn from, if we are able to pick it out.

This is in contrast to previous work (e.g., BID17 ; BID20 ; BID26 ), in which an increase in noise also implies a decrease in the absolute number Figure 1: Performance on MNIST as different amounts of noisy labels are added to a fixed training set of clean labels.

We compare a perceptron, MLPs with 1, 2, and 4 hidden layers, and a 4-layer ConvNet.

Even with 100 noisy labels for every clean label the ConvNet still attains a performance of 91%.

Performance on CIFAR-10 as different amounts of noisy labels are added to a fixed training set of clean labels.

We tested ConvNets with 4 and 6 layers, and a ResNet with 101 layers.

Even with 10 noisy labels for every clean label the ResNet still attains a performance of 85%.

of correct examples.

In the following experiments we investigate three different types of noise: uniform label-swapping, structured label-swapping, and out-of-vocabulary examples.

A key assumption in this paper is that unreliable labels are better modeled by an unknown stochastic process rather than by the output of an adversary.

This is a natural assumption for data that is pulled from the environment, in which antagonism is not to be expected in the noisy annotation process.

Deep neural networks have been shown to be exceedingly brittle to adversarial noise patterns BID18 .

In this work, we demonstrate that even massive amounts of non-adversarial noise present far less of an impediment to learning.

As a first experiment, we will show that common training procedures for neural networks are resilient even to settings where correct labels are outnumbered by labels sampled uniformly at random at a ratio of 100 to 1.

For this experiment we focus on the task of image classification and work with three commonly used datasets, MNIST (LeCun et al., 1998) , CIFAR-10 (Krizhevsky & Hinton, 2009) and ImageNet BID2 ).In FIG1 we show the classification performance with varying levels of label noise.

For MNIST, we vary the ratio ?? of randomly labeled examples to cleanly labeled examples from 0 (no noise) to 100 (only 11 out of 101 labels are correct, as compared with 10.1 for pure chance).

For the more challenging dataset CIFAR-10, we vary ?? from 0 to 10.

For the most challenging dataset ImageNet, we let ?? range from 0 to 5.

We compare various architectures of neural networks: multilayer perceptrons with different numbers of hidden layers, convolutional networks (ConvNets) with different numbers of convolutional layers, and residual networks (ResNets) with different numbers of layers BID4 .

We evaluate performance after training on a test dataset that is free from noisy labels.

Full details of our experimental setup are provided in ??3.4.Our results show that, remarkably, it is possible to attain over 90 percent accuracy on MNIST, even when there are 100 randomly labeled images for every cleanly labeled example, to attain over 85 percent accuracy on CIFAR-10 with 10 random labels for every clean label, and to attain over 70 percent top-5 accuracy on ImageNet with 5 random labels for every clean label.

Thus, in this highnoise regime, deep networks are able not merely to perform above chance, but to attain accuracies that would be respectable even without noise.

Further, we observe from Figures 1 and 2 that larger neural network architectures tend also to be more robust to label noise.

On MNIST, the performance of a perceptron decays rapidly with in- Figure 4: Illustration of uniform and structured noise models.

In the case of structured noise, the order of false labels is important; we tested decreasing order of confusion, increasing order of confusion, and random order.

The parameter ?? parameterizes the degree of structure in the noise.

It defines how much more likely the second most likely class is over chance.creasing noise (though it still attains 40 percent accuracy, well above chance, at ?? = 100).

The performance of a multilayer perceptron drops off more slowly, and the ConvNet is even more robust.

Likewise, for CIFAR-10, the accuracy of the residual network drops more slowly than that of the smaller ConvNets.

This observation provides further support for the effectiveness of ConvNets and ResNets in particular for applications where noise tolerance may be important.

We have seen that neural networks are extremely robust to uniform label noise.

However, label noise in datasets gathered from a natural environment is unlikely to follow a perfectly uniform distribution.

In this experiment, we investigate the effects of various forms of structured noise on the performance of neural networks.

Figure 4 illustrates the procedure used to model noise structure.

In the uniform noise setting, as illustrated on the left side of Figure 4 , correct labels are more likely than any individual false label.

However, overall false labels vastly outnumber correct labels.

We denote the likelihood over chance for a label to be correct as .

Note that = 1/(1 + ??), where ?? is the ratio of noisy labels to certainly correct labels.

To induce structure in the noise, we bias noisy labels to certain classes.

We introduce the parameter ?? to parameterize the degree of structure in the noise.

It defines how much more likely the second most likely class is over chance.

With ?? = 0 the noise is uniform, whereas for ?? = 1 the second most likely class is equally likely as the correct class.

The likelihood for the remaining classes is scaled linearly, as illustrated in Figure 4 on the right.

We investigate three different setups for structured noise: labels biased towards easily confused classes, towards hardly confused classes and towards random classes.

FIG4 shows the results on MNIST for the three different types of structured noise, as ?? varies from 0 to 1.

In this experiment, we train 4-layer ConvNets on a dataset that is diluted with 20 noisy labels for each clean label.

We vary the order of false labels so that, besides the correct class, labels are assigned most frequently to (1) those most often confused with the correct class, (2) those least often confused with it, and (3) in a random order.

We determine commonly confused labels by training the network repeatedly on a small subset of MNIST and observing the errors it makes on a test set.

The results show that deep neural nets are robust even to structured noise, as long as the correct label remains the most likely by at least a small margin.

Generally, we do not observe large differences between the different models of noise structure, only that bias towards random classes seems to hurt the performance a little more than bias towards confused classes.

This result might help explain why we often observe quite good results from real world noisy datasets, where label noise is more likely to be biased towards related and confusing classes.

In the preceding experiments, we diluted the training sets with noisy examples drawn from the same dataset; i.e., falsely labeled examples were images from within other categories of the dataset.

In (1) "confusing order" (highest probability for the most confusing label), (2) "reverse confusing order", and (3) random order.

We interpolate between uniform noise, ?? = 0, and noise so highly skewed that the most common false label is as likely as the correct label, ?? = 1.

Except for ?? ??? 1, performance is similar to uniform noise. :

Performance on CIFAR-10 for varying amounts of noisy labels.

Noisy training examples are drawn from (1) CIFAR-10 itself, but mislabeled uniformly at random, (2) CIFAR-100, with uniformly random labels, and (3) white noise with mean and variance chosen to match those of CIFAR-10.

Noise drawn from CIFAR-100 resulted in only half the drop in performance observed with noise from CIFAR-10 itself, while white noise examples did not appreciable affect performance.natural scenarios, however, noisy examples likely also include categories not included in the dataset that have erroneously been assigned labels within the dataset.

Thus, we now consider two alternative sources for noisy training examples.

First, we dilute the training set with examples that are drawn from a similar but different dataset.

In particular, we use CIFAR-10 as our training dataset and dilute it with examples from CIFAR-100, assigning each image a category from CIFAR-10 at random.

Second, we also consider a dilution of the training set with "examples" that are simply white noise; in this case, we match the mean and variance of pixels within CIFAR-10 and again assign labels uniformly at random.

FIG5 shows the results obtained by a six-layer ConvNet on the different noise sources for varying levels of noise.

We observe that both alternative sources of noise lead to better performance than the noise originating from the same dataset.

For noisy examples drawn from CIFAR-100, performance drops only about half as much as when noise originates from CIFAR-10 itself.

This trend is consistent across noise levels.

For white noise, performance does not drop regardless of noise level; this is in line with prior work that has shown that neural networks are able to fit random input BID26 .

This indicates the scenarios considered in Experiments 1 and 2 represent in some sense a worst case.

In natural scenarios, we may expect massively noisy datasets to fall somewhere in between the cases exemplified by CIFAR-10 and CIFAR-100.

That is, some examples will be relevant but mislabeled.

However, it is likely that many examples will not be from any classes under consideration and therefore will influence training less negatively.

In fact, it is possible that such examples might increase accuracy, if the erroneous labels reflect underlying similarity between the examples in question.

All models are trained with AdaDelta (Zeiler, 2012) as optimizer and a batch size of 128.

For each level of label noise we train separate models with different learning rates ranging from 0.01 to 1 and pick the learning rate that results in the best performance.

Generally, we observe that the higher the label noise, the lower the optimal learning rate.

We investigate this trend in detail in ??5.

There seems to be a critical amount of clean training data required to successfully train the networks.

This threshold increases as the noise level rises.

For example, at ?? = 10, 2,000 clean labels are needed to attain 90% performance, while at ?? = 50, 10,000 clean labels are needed.

In Experiments 1 and 2, noisy labels are drawn from the same dataset as the labels guaranteed to be correct.

This involves drawing the same example many times from the dataset, giving it the correct label once, and in every other instance picking a random label according to the noise distribution in question.

We show in Figure 7 that performance would have been comparable had we been able to draw noisy labels from an extended dataset, instead of repeating images.

Specifically, we train a convolutional network on a subset of MNIST, with 2,500 certainly correct labels and with noisy labels drawn either with repetition from this set of 2,500 or without repetition from the remaining examples in the MNIST dataset.

The results are essentially identical between repeated and unique examples, supporting our setup in the preceding experiments.

Underlying the ability of deep networks to learn from massively noisy data is the size of the data in question.

It is well-established, see e.g., BID2 , that traditional deep learning relies upon large datasets.

We will now see how this is particularly true of noisy datasets.

In Figure 8 , we compare the performance of a ConvNet on MNIST as the size of the training set varies.

We also show the performance of the same ConvNet trained on MNIST diluted with noisy labels sampled uniformly.

We show how the performance of the ConvNet varies with the number of cleanly labeled training examples.

For example, for the blue curve of ?? = 10 and 1,000 clean labels, the network is trained on 11,000 examples: 1,000 cleanly labeled examples and 10,000 with random labels.

Generally, we observe that independent of the noise level the networks benefit from more data and that, given sufficient data, the networks reach similar results.

Further, the results indicate that there seems to be a critical amount of clean training data that is required to successfully train the networks.

This critical amount of clean data depends on the noise level; in particular, it increases as the noise level rises.

Since performance rapidly levels off past the critical threshold the main requirement for the clean training set is to be of sufficient size.

It is because of the critical amount of required clean data that we have not attempted to train networks for ?? 100.

The number of correct examples needed to train such a network might rise above the 60,000 provided in the MNIST dataset.

In a real-world dataset, the amount of (noisy) data available for training is likely not to be the limiting factor.

Rather, considerations such as training time and learning rate may play a more important role, as we discuss in the following section.

In the preceding sections, our results were obtained by training neural networks with fixed batch size and running a parameter search to pick the optimal learning rate.

We now look in more detail into how the choice of hyperparameters affects learning on noisy datasets.

First, we investigate the effect of the batch size on the noise robustness of neural network training.

In Figure 9 , we compare the performance of a simple 2-layer ConvNet on MNIST with increasing noise, as batch size varies from 32 to 256.

We observe that increasing the batch size provides greater robustness to noisy labels.

One reason for this behavior could be that, within a batch, gradient updates from randomly sampled noisy labels cancel out, while gradients from correct examples that are marginally more frequent sum together and contribute to learning.

By this logic, large batch sizes would be more robust to noise since the mean gradient over a larger batch is closer to the gradient for correct labels.

All other experiments in this paper are performed with a fixed batch size of 128.We may also consider the theoretical case of infinite batch size, in which gradients are averaged over the entire space of possible inputs at each training step.

While this is often impossible to perform in practice, we can simulate such behavior by an auxiliary loss function.

In classification tasks, we are given an input x and aim to predict the class f (x) ??? {1, 2, . . .

, m}. The value f (x) is encoded within a neural network by the 1-hot vector y(x) such that DISPLAYFORM0 for 1 ??? k ??? m. Then, the standard cross-entropy loss over a batch X is given by: DISPLAYFORM1 where?? is the predicted vector and ?? X denotes the expected value over the batch X. We assume that?? is normalized (e.g. by the softmax function) so that the entries sum to 1.For a training set with noisy labels, we may consider the label f (x) given in the training set to be merely an approximation to the true label f 0 (x).

Consider the case of n training examples, and ??n noisy labels that are sampled uniformly at random from the set {1, 2, . . .

, m}. Then, f (x) = f 0 (x) with probability 1 1+?? , and otherwise it is 1, 2, . . .

, m, each with probability ?? m(1+??) .

As batch size increases, the expected value over the batch X is approximated more closely by these probabilities.

In the limit of infinite batch size, equation FORMULA1 takes the form of a noisy loss function H ?? : DISPLAYFORM2 We can therefore compare training using the cross-entropy loss with ??n noisy labels to training using the noisy loss function H ?? without noisy labels.

The term on the right-hand side of (3) represents the noise contribution, and is clearly minimized where?? k are all equal.

As ?? increases, this contribution is weighted more heavily against ??? log?? f0(x) X , which is minimized at??(x) = y(x).We show in Figure 9 the results of training our 2-layer ConvNet on MNIST with the noisy loss function H ?? , simulating ??n noisy labels with infinite batch size.

We can observe that the network's accuracy does not decrease as ?? increases.

This can be explained by the observation that an increasing ?? is merely decreasing the magnitude of the true gradient, rather than altering its direction.

Our observations indicate that increasing noise in the training set reduces the effective batch size, as noisy signals roughly cancel out and only small learning signal remains.

We show that increasing the batch size is a simple practical means to mitigate the effect of noisy training labels.

It has become common practice in training deep neural networks to scale the learning rate with the batch size.

In particular, it has been shown that the smaller the batch size, the lower the optimal learning rate BID9 .

In our experiments, we have observed that noisy labels reduce the effective batch size.

As such, we would expect that lower learning rates perform better than large learning rates as noise increases.

Figure 10 shows the performance of a 4-layer ConvNet trained with different learning rates on CIFAR-10 for varying label noise.

As expected, we observe that the optimal learning rate decreases as noise increases.

For example, the optimal learning rate for the clean dataset is 1, while, with the introduction of noise, this learning rate becomes unstable.

To sum up, we observe that increasing label noise reduces the effective batch size.

We have shown that the effect of label noise can be partly counterbalanced for by a larger training batch size.

Now, we see that one can additionally scale the learning rate to compensate for any remaining change in effective batch size induced by noisy labels.

In this paper, we have considered the behavior of deep neural networks on training sets with very noisy labels.

In a series of experiments, we have demonstrated that learning is robust to an essentially arbitrary amount of label noise, provided that the number of clean labels is sufficiently large.

We have further shown that the threshold required for clean labels increases as the noise level does.

Finally, we have observed that noisy labels reduce the effective batch size, an effect that can be mitigated by larger batch sizes and downscaling the learning rate.

It is worthy of note that although deep networks appear robust to even high degrees of label noise, clean labels still always perform better than noisy labels, given the same quantity of training data.

Further, one still requires expert-vetted test sets for evaluation.

Lastly, it is important to reiterate that our studies focus on non-adversarial noise.

Our work suggests numerous directions for future investigation.

For example, we are interested in how label-cleaning and semi-supervised methods affect the performance of networks in a high-noise regime.

Are such approaches able to lower the threshold for training set size?

Finally, it remains to translate the results we present into an actionable trade-off between data annotation and acquisition costs, which can be utilized in real world training pipelines for deep networks on massive noisy data.

@highlight

We show that deep neural networks are able to learn from data that has been diluted by an arbitrary amount of noise.