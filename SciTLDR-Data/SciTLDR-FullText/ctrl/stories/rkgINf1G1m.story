The softmax function is widely used to train deep neural networks for multi-class classification.

Despite its outstanding performance in classification tasks, the features derived from the supervision of softmax are usually sub-optimal in some scenarios where Euclidean distances apply in feature spaces.

To address this issue, we propose a new loss, dubbed the isotropic loss, in the sense that the overall distribution of data points is regularized to approach the isotropic normal one.

Combined with the vanilla softmax, we formalize a novel criterion called the isotropic softmax, or isomax for short, for supervised learning of deep neural networks.

By virtue of the isomax, the intra-class features are penalized by the isotropic loss while inter-class distances are well kept by the original softmax loss.

Moreover, the isomax loss does not require any additional modifications to the network, mini-batches or the training process.

Extensive experiments on classification and clustering are performed to demonstrate the superiority and robustness of the isomax loss.

Recent years have witnessed significant progress in image classification tasks with convolution neural networks (CNN) BID13 ; .

For classification problems, the softmax is a suitable criterion for supervised learning since it is capable of training network parameters to generate discriminative features for hyperplanes to distinguish different classes.

Due to its end-to-end characteristic, CNN is amenable to learning such that we only need to feed the network with plenty of training samples.

Therefore, the softmax is the most fundamental classifier applied in architectures of deep learning.

However, there are still defects of the softmax loss 1 .

Features extracted by convolution layers work best only with softmax classifier.

When we apply these feature vectors in other tasks such as image retrieval with k-nearest neighbors (k-NN) or clustering with K-means, the results are usually suboptimal, as shown in FIG0 .

To separate different classes is the sole purpose of softmax classifier, and it does not ensure that the distances (generally Euclidean distances) within the same class are smaller than inter-class ones.

In order to extract better features not only for classification powered by the softmax classifier but for other tasks using distances of feature vectors, many approaches have been proposed in the past years.

One way is adding some new loss terms to the original softmax loss.

The center loss penalizes distances of training samples to their corresponding class centers, thus enhancing the compactness of each class.

Since this loss cannot extend the distances between class centers, the result depends heavily on the center initialization of all classes.

Also, a relatively small batch size will seriously affect the performance of the algorithm, because centers cannot be accurately calculated with limited examples in a batch, especially for datasets with plenty of classes.

The contrastive-center loss BID17 combines the center loss and the contrastive loss BID22 together to penalize distances of samples in the same classes and enlarge inter-class distances.

It works well on the CIFAR-10 classification task BID10 and the LFW verification task BID7 .

To gain good performance, however, this loss needs to carefully select data batches for training too.

Some other strategies aim at solving the problem in a different way.

The triplet loss BID18 comes up with a novel method.

Instead of exploiting the softmax loss, they remove the final logits layer of network and directly minimize Euclidean distances between anchors and positive samples while maximizing distances of anchors to negative samples in triplets.

However, the number of different triplets are much more than that of training samples, leading to that selecting proper triplets is crucial for the triplet loss.

Or inappropriate triplets will result in slow convergence.

What's more, the scalar margin in the triplet loss and the learning strategy influence the final performance of the model as well.

To sum up, the principle of the triplet loss is straightforward and plausible, but the parameters and the "semi-hard triplets mining" approach make this algorithm hard to implement.

Methods mentioned above are all based on supervised optimization, meaning that they all depend on sample labels to learn discriminative features.

In this paper, we propose a simple approach, named isotropic normalization, that reshapes data distribution towards easy classification without the aid of labels.

We firstly analyze the distribution of features extracted by CNNs supervised by the softmax loss.

Elliptical shapes of feature distributions lead to intra-class distances even greater than inter-class distances, indicating that the softmax loss needs to be improved for tasks using feature distances.

Then we attempt to modify the feature distribution according to the global distribution itself rather than information from sample labels.

Combined with the vanilla softmax, we propose a new loss, called the isotropic softmax (isomax for short) loss.

For the isomax, the intra-class distances of the features can be well minimized by the isotropic loss, and at the same time, the vanilla softmax loss ensures the inter-class separability.

We perform extensive experiments with different networks and different datasets to illustrate the effectiveness, simplicity and portability of our method.

In this section, we introduce the softmax loss and analyze its limitations with visualization of 2-D features.

According to the feature distribution, we show why these features are not the optimal option in tasks that use feature distances.

Then we present our approach to ameliorating the feature distribution.

Suppose that we have N training samples of n classes.

Let x i ??? R d denote the i-th image feature vector and y i its label.

Then softmax loss can be formulated as DISPLAYFORM0 For typical neural networks, f denotes the output of a fully connected layer with weights W (a parameter matrix with n columns) and bias b. Then the softmax loss can be rewritten as Equation FORMULA1 where W j denotes the j-th column of W .

Since W j and x i are all vectors, the inner product of them can be formulated as W j x i cos(?? j ), where ?? j is the angle of W j and x i .

After formulating the softmax loss to an inner product form with a factor cos(??), it is clear that the softmax classifier is prone to enforce examples in a class have similar ?? and different ?? for different classes.

This property of the softmax leads to the thin elliptical shape of feature distribution for each class with respect to the global center.

DISPLAYFORM1

To analyze the feature distribution, it is a feasible way to visualize data points in a 2-D plane or 3-D space.

MNIST LeCun et al. (1995) is a hand-written digit dataset of 0-9 in 10 classes, with 60,000 training samples and 10,000 testing samples.

Image size of both subsets is 28??28.

Since MNIST is such a simple dataset that even we reduce the dimension of feature vectors to 2, the softmax classifier is applicable to maintain 98% accuracy.

So we choose it as our first example to describe the feature distribution.

The network architecture is presented in TAB1 .

We train the network with 60,000 training samples without data argumentation, and randomly sample 1000 testing images in 10,000 testing set.

Since the dimension of feature vectors is 2, we can plot features of testing samples, as shown in FIG2 (a).

It is evident that minimizing the softmax loss leads to the separability of different classes.

But for some samples, especially those near to the center of overall distribution, inter-class distances are even smaller than intra-class distances.

Figure 2 (a) directly shows overall distribution under supervision of softmax loss.

The thin shape of feature distribution of a certain class is the inevitable result of the softmax loss according to our previous analysis, which causes the large intra-class distances and small inter-class distances.

Apart from the softmax loss, if we can find a new "force" which can reshape the thin distribution to a "medium build" FIG2 (c)) , the inter-class distances will be kept by the softmax loss while the intra-class distances will be guaranteed by the new force.

For a certain class i, we can regard features of this class as a normal distribution N (?? i , ?? i ) where ?? i is the mean of this class and ?? i is the covariance matrix.

Then there exits an orthogonal matrix U and an eigenvalue-diagonal matrix D satisfying ?? i = U DU .

Diagonal values in D can describe the shape of the normal distribution.

If these eigenvalues are almost the same (with small variance), then the distribution of features is like an isotropic normal one.

This analysis inspires us to devise a simple algorithm to optimize the shape of each class.

In this section, we describe our approach in detail.

The relevant classification criteria are discussed as well.

Inspired by FIG2 (c) and the discussion of isotropic multivariate normal distribution, the spatial distribution of each class will be more compact if we penalize features such that N (?? i , ?? i ) of class i will be like an isotropic one, say that the variances of different dimensions are approximately uniform.

Here we present the isotropic loss that action on data points with the complementary effect compared with the softmax function, writing that DISPLAYFORM0 wherex denotes the center of DISPLAYFORM1 , L I denotes our new isotropic loss.

In simple words, the loss we attempt to minimize is the unbiased variance estimate of distance x i ???x 2 2 .

Specially, take d=2 in FIG2 for instance.

The L I loss will push points near to the circle of radiusD andx for the circle center.

It is worth mentioning that this method is essentially different from feature normalization after training.

We perform this isotropic regularizer during training andD varies, meaning that the shape of each class will be deformed with softmax supervision and isotropic regularization towards easy classification during iterations.

Obviously, it is impractical and inefficient to calculate the L I loss function for all training examples in each iteration.

However, we can compute the estimates ofx andD within a batch if we randomly select samples in a training batch.

The randomness of the data and the globality of the loss function guarantee the plausibility of such manipulation.

Thanks to the decoupling of the isotropic loss and class labels, our loss remains unchanged from the whole training set to a batch.

Details are shown in Algorithm 1.

Input: A batch of feature B={x 1 , ..., x m } produced by deep neural networks Output: isotropic loss L I for a mini-batch The total joint loss named the isotropic softmax (isomax) loss is formulated as DISPLAYFORM0 DISPLAYFORM1 where L S is the softmax loss and ?? controls the trade-off between the isotropic loss and the softmax loss.

A small ?? may be not enough to form an isotropic distribution while a large ?? will restrain the supervision of the softmax loss, making features indistinguishable near the hypersphere.

In the following section, we will discuss the influence of different ??.

As mentioned above, the loss is the unbiased variance estimate of distance x i ???x 2 2 , which is differentiable to x i , and x i is the output of the last hidden layer of CNNs.

According to chain rule, our loss term is differentiable to parameters of networks.

SGD Bottou (2010), Adam BID9 , RMSProp Tieleman & Hinton (2012) or other learning methods for neural networks can be used to minimize the isomax loss.

As an unsupervised loss term, the optimization of the isotropic loss requires no conditions about training batches or class labels.

All we need to do is combining this new loss with the original softmax loss and training the network directly via backpropagation.

To find out the hyper-parameter ??, we first examine the 2-D distribution of MNIST.

After 20,000 step training, the feature distribution of 1,000 test samples are shown in FIG3 .

We can see that the result is much like what we expect when ??=0.005 or ??=0.05.

But for a large ??, the force of the isotropic loss is so powerful that the softmax loss loses its ability to shape classes.

During training, we find that even very small ?? is able to cut down the isotropic loss quickly, the only influence of ?? is the speed of convergence.

But if ?? > 0.1, not only the training procedure converges slowly, but the result is also unacceptable.

This threshold is suitable for almost all networks and datasets we try.

In following sections, unless otherwise specified, the value of ?? is set to 0.05.

Since our loss tends to normalize features to a hypersphere surface, one alternative way may be simply normalizing features to a hypersphere, specifically a circle for 2-D features FIG2 ).

It needs no additional operation or loss term during training and may work well in low dimensions, but it does not mean that feature normalization will work in higher dimensions.

The reason is that in high-dimensional space, the hyperplanes of the softmax classifier is much more complex.

If we normalize the features to a hypersphere after training, many features from different classes would overlap together on the hyperspherical surface.

In fact, isotropic loss, as a part of isomax loss, is not a normalization operation.

It only tries to gather the feature points around the sphere surface, which is a slow process.

So features cannot be directly stacked together.

Moreover, even if features from different classes become close to each other around the sphere surface, softmax loss, as a another supervision in isomax loss, is capable of pushing different feature points away again.

Therefore our method is totally different from feature normalization after training.

There are some related works with the same purpose as ours such as the triplet loss BID18 and the center loss .

We will analyze the properties of compared methods in this section.

TAB3 concisely shows the merits of each method.

We also discuss Batch Normalization BID8 since it is an operation on feature distributions too, though for different purpose.

Center loss is an excellent work to solve the problem of large intra-class distances.

Jointly supervised by the center loss and the softmax loss, the features of same class can gather together.

Equation FORMULA6 gives the formulation DISPLAYFORM0 where c yi is the center of the i-th sample class in a batch.

Centers are learned in each iteration and each batch.

Though the authors of this work give some methods to avoid perturbations of centers, centers are still difficult to determine, especially when the samples from one class are limited in a batch.

We train models of MNIST with the center loss and the isomax loss both twice and the experimental conditions are all the same except for the different loss terms.

Results are shown in Figure 4 .

It is obvious that once a center is not well established, the features of this class will gather near this center, incurring that the overall distribution of associated features is unfavorable for classification.

Instead the results of our isomax loss are more stable and robust.

What's more, the fluctuating convergence and the additional calculation of centers make the training time longer.

In our algorithm, the isotropic loss is an unsupervised loss.

This loss function learns a better distribution directly from distribution itself without using labels of training data.

All we need is randomly selecting samples from the training set like what we do with the softmax loss.

The triplet loss is employed to supervise the learning of an Euclidean embedding per image.

The loss is formulated as DISPLAYFORM0 where f (x) is an embedding generated by CNNs, the selected anchor x a i , the positive sample x p i , and the negative sample x n i constitute a triplet.

The triplet loss directly minimizes the Euclidean distance of samples from the same class and maximizes the distance of samples from the different class until reaching a selection margin ??.

For the triplet loss, the learning strategy and the triplet selection are critical for performance.

By many practices, we find that the model is sensitive to the learning rate and the learning method.

Also, the converging speed is much lower than that of the model supervised by the softmax loss.

Due to the massive combination of triplets and the learning tricks it requires, the results we get are not so good.

To sum up, the training with the triplet loss is a laborious task.

Batch Normalization is a method to reduce internal covariate shift in data BID19 .

They introduce ?? (k) and ?? (k) for each activation x (k) in a d-dimensional feature x, and transform original feature x=( DISPLAYFORM0 By adding Batch-Norm layer (BN layer) to a certain layer in networks, each dimension of the feature of this layer is transformed according to Equation 7.

Training samples in a batch are used for feature distribution transformation, which seems similar to our approach.

Though batch normalization and our method both aim at distribution transformation, they are totally different from operation and functionality.

According to Equation 7, the normalization operation is along the axis of one dimension of features.

Batch normalization constrains the values of each dimension of features within a reasonable range, while our method normalizes Euclidean distances of feature vectors with respect to the global center of data distribution.

To make it clearer, we interpret a simplified example by letting ?? (k) =1 and ?? (k) =0.

As each dimension is normalized, batch normalization transforms features inside a hypersphere.

Our loss term tries to push features onto the spherical surface, thus reducing the variance of distances to the distribution center (spherical center).

Batch normalization prevents the training from getting stuck.

As for our approach, a final feature map with small intra-class distance and large inter-class distance is our target.

In fact, these two seemingly similar operations are mutually independent.

If we use batch normalization for the last hidden layer, the x in Algorithm 1 will be replaced by the output y of batch normalization.

In the following experiments in the next section, batch normalization is applied together with our new loss.

Briefly in summary, batch normalization is a local normalization on a certain feature map to improve the training of the network.

Ours is a global geometric supervision on the overall feature distribution of the last hidden layer to acquire better representations.

We evaluate the isomax on three tasks: image classification, feature clustering and face verification.

Experiments are performed on four datasets: MNIST, CIFAR-10, a subset of ILSVRC2012 Deng et al. (2012 with 200 classes, and CASIA-WebFace BID22 .

We divide CASIA-WebFace into training and testing sets with a ratio of 2:1.

These four datasets are very different in data attributes, data volumes, and class numbers.

For the classification task, the testing sets of these four datasets are all used.

For the clustering task, the Face Recognition Grand Challenge (FRGC) BID16 dataset is also used for testing.

For face verification, we use the Labeled Faces in the Wild (LFW) benchmark dataset.

To compare fairly for each dataset, we train three models under different supervision: the softmax loss, the center-softmax joint loss and our isomax loss using the same training methodology and network architecture.

For verification task, we also train a model with the triplet loss since it achieves state-of-the-art performance in LFW.

All the experiments are implemented with TensorFlow BID0 .

For MNIST, the network we use is shown in TAB1 .

For CIFAR-10, inspired by the architecture of VGG-net BID20 , we design our network in this way: 3 cascaded convolution blocks followed by a dropout Hinton et al. FORMULA0 layer and then a fully-connected layer .

There are 4 cascaded convolution layers with size of 3??3 (stride=1) and a max-pooling layer with size of 2??2 (stride=2) in each block.

Filter numbers in blocks are 64, 96 and 128.

We do not use data augmentation on these two datasets but use a dropout of 0.8.

The initial learning rate is 0.1 and decays with an exponential rate of 0.96 every 1000 steps.

The Adagrad method is employed for optimization.

Batch size is 128.

For this dataset, there are 200 classes and about 1300 images in each class.

The Inception-ResNetv1 network BID24 is applied.

We do not use the input size of 299??299 according to the original paper, but 160??160 for simplicity of calculation.

In training phase, we resize the short edge of image to 182, and randomly crop a 160??160 window.

The random adjustment of hue, brightness, contrast and saturation is applied.

In testing phase, we resize the short edge to 160 and crop a square sub-image in the center.

We harness Adagrad and a decaying learning rate that starts at 0.05 and decays with an exponential rate of 0.94 every two epochs.

Batch size is 128.

CASIA-WebFace contains 0.49M labeled face images from over 10,575 individuals.

We firstly exploit MTCNN Zhang et al. (2016) to align face images based on 5 points.

Then two-thirds of images from each individual (about 0.32M totally) are randomly selected to train the InceptionResNet-v1 network.

During training, we resize images to 182??182 and randomly crop a 160??160 window, while for testing we directly resize images to 160??160.

Random left-to-right flipping is also used for training.

The training method we use is RMSProp with decay of 0.9 and = 1.0.

Batch size is 512.

In fact, this relatively large batch size is chosen for center loss since centers need to be updated in a batch and the number of classes of CASIA dataset is too large.

We use an initial learning rate of 0.1, divided by 10 after 50 epochs and 65 epochs.

The softmax loss is our baseline, which is designed for image classification.

So we firstly evaluate our model on this kind of task.

To ensure that our modification to the original softmax loss has no negative impact on the softmax classifier, we firstly evaluate the accuracy on four datasets.

Then, to illustrate the superiority of our model in distance-based tasks, we also report the k-NN classification results using feature vectors attained by the supervision of compared losses.

TAB5 shows the experimental results.

We observe that our algorithm does not deteriorate the performance of the softmax classifier, while in k-NN classification tasks, our proposed algorithm performs better, especially for datasets with many classes.

For these datasets, it is more essential to place features of different classes in a finite feature space properly.

To show details of training, we take the CASIA-WebFace dataset as an example, plotting the curve of accuracy during training process in FIG4 .Another interesting evaluation we analyze here is shown in Figure 6 .

In three contrast experiments, we monitor the value of center loss, which is only minimized as a loss term in the experiment of center loss.

However, for each dataset, the isomax loss makes the center loss much lower, though we do not take the initiative to optimize it.

As the center loss is a measure of intra-class distance, this proves that, by reshaping the overall global distribution, our algorithm can reduce the distance within a class more effectively.

In this section, we test the performance of different losses when associated neural networks are attacked by adversarial samples 2 .

To do so, we employ the Fast Gradient Sign Method to generate the adversarial samples BID4 , say DISPLAYFORM0 where controls the degree of adversary.

Small epsilon means slight perturbation and weak attacking.

The gradient in the above formula is derived from the derivative of the softmax loss for each of three compared models.

The experiments are performed on MNIST and CAFAR-10.

The experimental procedures are kept the same with ones in section 4.2.

From the results shown in Table 4 , we can see that the isomax loss consistently outperforms the other two losses.

It is worthing noting that the superiority of the isomax loss is significant for = 0.003 on MNIST.

The underlying reason is presumably that classes are still maintained separable due to the more compact isotropic distribution for the isomax loss under the slight perturbation whereas the distribution of classes for the softmax and center losses might be 0 20000 40000 60000 80000 100000Step Table 4 : Classification accuracy on MNIST and CIFAR-10 for adversarial attacking.

The adversarial samples are produced with the fast gradient sign method (FGSM) with different .

messy, especially near boundaries.

To our surprise, however, the center loss performs the worst on CIFAR-10, implying its weak robustness under adversarial perturbation.

Compact intra-class features can be used in clustering tasks.

We evaluate K-means and agglomerative clustering (ward linkage) performance of features under different supervision signals.

For MNIST, CIFAR-10, and ILSVRC-sub, 10,000 images from their corresponding testing set are used for clustering.

As for the face model, we use 12,776 face images of 466 different identities from FRGC Phillips et al. (2005) dataset.

For evaluation, 12 pre-trained models mentioned above are employed, i.e. 3 different losses for each of 4 datasets.

Normalized Mutual Information (NMI) BID21 is selected as the clustering performance evaluation metric.

The results in TAB8 show the consistent superiority of our algorithm.

We also evaluate the performance of our algorithm on a widely used verification benchmark LFW BID7 .

The dataset contains 13,233 faces from 5749 individuals with different poses and expressions.

We use the models trained on 0.32M CASIA-WebFace mentioned above.

We also implement a model trained with triplet loss with the same training data and network.

There is no overlap between our training data and LFW dataset.

We use MTCNN Zhang et al. (2016) to align the face images of LFW like what we do on CASIA-WebFace.

Instructed by the protocol for unrestricted with labeled outside data Huang & Learned-Miller (2014), we report the result of verification performance of 6,000 pairs of faces in TAB9 and FIG5 .

We only train our model with 0.32M outside face images on a single network, so our purpose is not to improve state-of-the-art accuracy but a good comparison of isomax loss, center loss, softmax loss and triplet loss under the same condition.

Our proposed method is more efficient and suitable in verification task compared with softmax loss baseline and also outperforms center loss and triplet loss under the same condition.

Since the dataset we use for training has more than 10,000 classes, making centers for each class difficult to determine in a mini-batch.

So the model with center loss surpasses the model with softmax loss by a small margin while ours outperforms it by a relatively large margin.

Triplet loss is another baseline we want to compare with.

Due to its long training process, we partially plot it in FIG5 .

The accuracy of triplet loss finally plateaus at 96.88%, which is lower than the accuracy reported in the original paper.

The small amount of training data and complex training tricks that we do not tune well may be the reasons.

In this paper, we propose a new isotropic loss together with the original softmax loss named isomax loss.

With the joint supervision signal, CNNs generate more isotropic feature distribution for each class, and the Euclidean distance in the class decreases.

The 2-D visualization and extensive experiments on different datasets for different tasks show the effectiveness of our approach.

Comparison to other related works illustrates the advantage of our method on tasks using feature distance.

<|TLDR|>

@highlight

The discriminative capability of softmax for learning feature vectors of objects is effectively enhanced by virture of isotropic normalization on global distribution of data points.