In this paper, we propose a novel regularization method, RotationOut, for neural networks.

Different from Dropout that handles each neuron/channel independently, RotationOut regards its input layer as an entire vector and introduces regularization by randomly rotating the vector.

RotationOut can also be used in convolutional layers and recurrent layers with a small modification.

We further use a noise analysis method to interpret the difference between RotationOut and Dropout in co-adaptation reduction.

Using this method, we also show how to use RotationOut/Dropout together with Batch Normalization.

Extensive experiments in vision and language tasks are conducted to show the effectiveness of the proposed method.

Codes will be available.

Dropout (Srivastava et al., 2014 ) has proven to be effective for preventing overfitting over many deep learning areas, such as image classification (Shrivastava et al., 2017) , natural language processing (Hu et al., 2016) and speech recognition (Amodei et al., 2016) .

In the years since, a wide range of variants have been proposed for wider scenarios, and most related work focus on the improvement of Dropout structures, i.e., how to drop.

For example, drop connect (Wan et al., 2013) drops the weights instead of neurons, evolutional dropout (Li et al., 2016) computes the adaptive dropping probabilities on-the-fly, max-pooling dropout (Wu & Gu, 2015) drops neurons in the max-pooling kernel so smaller feature values have some probabilities to to affect the activations.

These Dropout-like methods process each neuron/channel in one layer independently and introduce randomness by dropping.

These architectures are certainly simple and effective.

However, randomly dropping independently is not the only method to introduce randomness.

Hinton et al. (2012) argues that overfitting can be reduced by preventing co-adaptation between feature detectors.

Thus it is helpful to consider other neurons' information when adding noise to one neuron.

For example, lateral inhibition noise could be more effective than independent noise.

In this paper, we propose RotationOut as a regularization method for neural networks.

RotationOut regards the neurons in one layer as a vector and introduces noise by randomly rotating the vector.

Specifically, consider a fully-connected layer with n neurons: x ∈ R n .

If applying RotationOut to this layer, the output is Rx where R ∈ R n×n is a random rotation matrix.

It rotates the input with random angles and directions, bringing noise to the input.

The noise added to a neuron comes not only from itself, but also from other neurons.

It is the major difference between RotationOut and Dropout-like methods.

We further show that RotationOut uses the activations of the other neurons as the noise to one neuron so that the co-adaptation between neurons can be reduced.

RotationOut uses random rotation matrices instead of unrestricted matrices because the directions of feature vectors are important.

Random rotation provides noise to the directions directly.

Most neural networks use dot product between the feature vector and weight vector as the output.

The network actually learns the direction of the weights, especially when there is a normalization layer (e.g. Batch Normalization (Ioffe & Szegedy, 2015) or Weight Normalization (Salimans & Kingma, 2016) ) after the weight layer.

Random rotation of feature vecoters introduces noise into the angle between the feature and the weight, making the learning of weights directions more stable.

Sabour et al. (2017) also uses the orientation of feature vectors to represent the instantiation parameters in capsules.

Another motivation for rotating feature vectors comes from network dissection.

Bau et al. (2017) finds that random rotations of a learned representation can destroy the interpretability which is axis-aligned.

Thus random rotating the feature during training makes the network more robust.

Even small rotations can be a strong regularization.

We study how RotationOut helps prevent neural networks from overfitting.

Hinton et al. (2012) introduces co-adaptation to interpret Dropout but few literature give a clear concept of co-adaptation.

In this paper,we provide a metric to approximate co-adaptations and derive a general formula for noise analysis.

Using the formula, we prove that RotationOut can reduce co-adaptations more effectively than Dropout and show how to combine Dropout and Batch Normalization together.

In our experiments, RotationOut can achieve results on par with or better than Dropout and Dropoutlike methods among several deep learning tasks.

Applying RotationOut after convolutional layers and fully connected layers improves image classification accuracy of ConvNet on CIFAR100 and ImageNet datasets.

On COCO datasets, RotationOut also improves the generalization of object detection models.

For LSTM models, RotationOut can achieve competitive results with existing RNN dropout method for speech recognition task on Wall Street Journal (WSJ) corpus.

The main contributions of this paper are as follows: We propose RotationOut as a regularization method for neural networks which is different from existing Dropout-like methods that operate on each neuron independently.

RotationOut randomly rotates the feature vector and introduces noise to one neuron with other neurons' information.

We present a theoretical analysis method for general formula of noise.

Using the method, we answer two questions: 1) how noise-based regularization methods reduce co-adaptions and 2) how to combine noise-based regularization methods with Batch Normalization.

Experiments in vision and language tasks are conducted to show the effectiveness of the proposed RotationOut method.

Related Work Dropout is effective for fully connected layers.

When applied to convolution layers, it is less effective.

Ghiasi et al. (2018) argues that information about the input can still be sent to the next layer even with dropout, which causes the networks to overfit (Ghiasi et al., 2018) .

SpatialDropout (Tompson et al., 2015) drops the entire channel from the feature map.

Shake-shake regularization (Gastaldi, 2017) drops the residual branches.

Cutout (DeVries & Taylor, 2017) and Dropblock (Ghiasi et al., 2018 ) drop a continuois square region from the inputs/feature maps.

Applying standard dropout to recurrent layers also results in poor performance (Zaremba et al., 2014; Labach et al., 2019) , since the noise caused by dropout at each time step prevents the network from retaining long-term memory.

Gal & Ghahramani (2016) ; Moon et al. (2015) ; Merity et al. (2017) generate a dropout mask for each input sequence, and keep it the same at every time step so that memory can be retained.

Batch Normalization (BN) (Ioffe & Szegedy, 2015) accelerates deep network training.

It is also a regularization to the network, and discourage the strength of dropout to prevent overfitting (Ioffe & Szegedy, 2015) .

Many modern ConvNet architectures such as ResNet (He et al., 2016) and DenseNet (Huang et al., 2017) do not apply dropout in convolutions.

Li et al. (2019) is the first to argue that it is caused by the a variance shift.

In this paper, we use the noise analysis method to further explore this problem.

There is a lot of work studying rotations in networks.

Rotations on the images (Lenc & Vedaldi, 2015; Simard et al., 2003) are important data augmentation methods.

There are also studies about rotation equivalence.

Worrall et al. (2017) uses an enriched feature map explicitly capturing the underlying orientations.

Marcos et al. (2017) applies multiple rotated versions of each filter to the input to solve problems requiring different responses with respect to the inputs' rotation.

The motivations of these work are different from ours.

The most related work is network dissection (Bau et al., 2017) .

They discuss the impact on the interpretability of random rotations of learned features, showing that rotation in training can be a strong regularization.

In this section, we first introduce the formulation of RotationOut.

Next, we use linear models to demonstrate how RotationOut helps for regularization.

In the last part, we discuss the implementation of RotationOut in neural networks.

A rotation in D dimension is represented by the product between a rotation matrix R ∈ R D×D and the feature vector x ∈ R n .

The complexity for random rotation matrix generation and the matrix multiplication are both O(D 2 ), which would be less efficient than Dropout with O(D) complexity.

We consider a special case that uses Givens rotations (Anderson, 2000) to construct random rotation matrices to reduce the complexity.

Let D = 2d be an even number, and P = [n 1 , n 2 , · · · , n 2d ] be a permutation of {1, 2, · · · , D}. A rotation matrix can be generated by function M (θ, P ) = {r ij } ∈ R D×D :

Here P l represents the l th element of P where 1 ≤ l ≤ d. See Appendix A.1 for some examples of such rotation matrices.

Suppose we sample the angle θ from zero-centered distributions, e.g., truncated Gaussian distribution or uniform distribution and sample the permutation P from P,the set of all permutations of {1, 2, · · · , D}, with equal probability.

The RotationOut operator R can be generated using the function M (P, θ):

Here 1/ cos θ is a normalization term and R is not a rotation matrix strictly speaking.

The random operator generated from Equation 2 have some good properties.

1) The noise is zero centered:

2) For any vector x and any random permutation P , the angle between x and Rx is determined by angle θ:

The complexity for random rotation matrix generation and the matrix multiplication are both O(D).

Permutation P draws the rotation direction and angel θ draws the rotation angle.

As an analogy, permutation P is similar to the dropout mask widely used in RNN dropout.

There exists 2

, thus the diversity of random rotation in Equation 1 is sufficient for network training.

Angle θ is similar to the percentage of dropped neurons in Dropout, and the distribution of θ controls the regularization strength. (Srivastava et al., 2014) used the multiplier's variance to compare Bernoulli dropout and Gaussian dropout.

Following this setting, RotationOut is equivalent to Bernoulli Dropout with the keeping rate p and Gaussian dropout with variance

Reviewing the formulation of the random rotation matrix, it arranges all D dimensions of the input into d pairs randomly, and rotates the two dimension vectors with angle θ in each pair.

Suppose u and v are two dimensions/neurons in one pair, the outputs of u and v after RotationOut are

The noise of u comes from v and the noise of v comes from u since θ is random.

Note that the pairs are randomly arranged, thus RotationOut uses all other dimensions/neurons as the noise for one dimension/neuron of the feature vector.

With RotationOut, the neurons are trained to work more independently since one neuron has to regard the activation of other neurons as noise.

Thus the co-adaptations are reduced.

Consider Gaussian dropout, the outputs are u = u+u , v = v+v where E = 0, E 2 = E θ tan 2 θ.

The difference between Gaussian dropout and RotationOut is the source of noise, i.e., the Gaussian dropout noise for one neuron comes from itself while the RotationOut noise comes from other neurons.

First we consider a simple case of applying RotationOut to the classical problem of linear regression.

be the dataset where x i ∈ R D , y i ∈ R. Linear regression tries to find the weight

for each x i .

The objective function becomes:

Denote

To compare RotationOut with Dropout with keep rate p, we suppose E θ tan 2 θ = (1 − p)/p = λ.

Equation 4 reduces to:

Details see Appendix A.2.

Solutions to Equation 5 (LR with Rotation) and the mirror problem with dropout (Srivastava et al., 2014) are :

Therefore, linear regression with RotationOut and Dropout are equivalent to ridge regression with different regularization terms.

Set λ = 1 (Dropout rate p = 0.5) for simplicity.

LR with Dropout doubles the diagonal elements of X T X to make the problem numerical stable.

LR with RotationOut is more close to ridge regression:

The condition number of Equation 7 and the LR with RotationOut problem is up bounded by D − 1.

For the Dropout case, if some data dimensions have extremely small variances, both X T X and diag(X T X) are ill-conditioned.

LR with Dropout problem has unbounded condition number.

Next we consider an m-way classification model of logistic regression.

The input is x ∈ R D and the weights are W = [w 1 , w 2 , · · · , w m ] ∈ R m×D .

The probability that the input belongs to the k category is:

In Equation 8, θ i denotes the angel between x and w i .

Assume that the length of each weights w i are very close, the input x belongs to the k category if x is most close to w k in angle.

Consider a hard sample case that θ i <

θ j are the two smallest weight-data angles.

But θ i and θ j are very close: θ i ≈ θ j , i.e., the data are close to the decision boundary.

The model should classify the data correctly but could make mistakes if there is some noise.

Applying RotationOut, the angle between the data and the weights can be changed, and the new angles can be θ i > θ j .

To classify the data correctly, there should be a gap between θ i and θ j .

In other words, the decision boundary changed from θ i < θ j to θ i < θ j − Θ where Θ is a positive constant that depends on the regularization.

Thus RotationOut can be regarded as a margin-based hard sample mining.

Here we provide an intuitive understanding of how Dropout with low keep rates leads to lower performance.

Randomly zeroing units, Dropout method also rotates the feature vector.

A lower keep rate results in a bigger rotation angle:

Consider the last hidden layer in neural networks, it is similar to logistic regression on the features.

If one feature x is most close to w k , it belongs to the k th .

A lower keep rate Dropout would rotate the feature with a bigger angle, and the Dropout output can be most close to another weight with higher probability, which may hurts the training.

Consider a neural network with L hidden layers.

Let x l , y l , and W l denote the vector of inputs, the vector of output before activation, and the weights for the layer l. Let R be generated from Equation 2 and a be the activation function, for example Rectified Linear Unit (ReLU).

The MLP feed-forward operation with RotationOut in training time can be:

We rotate the zero-centered features and then add the expectation back.

The reasons will be explained later.

Here we give an intuitive understanding.

If features are not zero-centered, we do not know the exact regularization strength.

Suppose all features elements are in one interval, say 1 < x < 2.

The angle between any two feature vectors is a sharp angle.

In this case a rotation angle of π/4 would be too big.

It is the same for Dropout.

The regularization strength is influenced by the mean value of features which we may not know.

At test time, the RotationOut operation is removed.

Consider 2D case for example, the input for 2D convolutional layers are three dimensional: number of channels C, width H and height W :

We regard each x hw as a feature vector with semantic information for each position (h, w), and apply rotation to each position.

As Ghiasi et al. (2018) argued, the convolutional feature maps are spatially correlated, so information can still flow through convolutional layers if features are dropped out randomly.

Similarly, if we rotate feature vectors in different positions with random directions, random directions offset each other and result in no rotation.

So we rotate all feature vectors with the same directions but different angles.

The operation on convolutional feature maps can be:

The operation for general convolutional networks are very similar.

Also note that RotationOut can combined with DropBlock (Ghiasi et al., 2018) easily: only rotating features in a continuous block.

Experiments show that the combination can get extra performance gain.

As mentioned in Section 3.1, the rotation directions defined by P is similar to the dropout mask in RNN drops.

RotationOut can also be used in recurrent networks following Equation 11.

In this section, we first study the general formula of adding noise.

Using the formula, we show how introducing randomness/noise helps reduce co-adaptations and why RotationOut is more efficient than the vanilla dropout.

Strictly speaking, the co-adaptations describe the dependence between neurons.

The mutual information between two neurons may be the best metric to define co-adaptations.

To compute mutual information, we need the exact distributions of neurons, which are generally unknown.

So we consider the correlation coefficient to evaluate co-adaptations, which only need the first and second moment.

Moreover, if we assume the distributions of neurons are Gaussian, correlation coefficient and mutual information are equivalent in co-adaptations evaluation.

The ideal situation is that Σ = diagΣ, i.e., the neurons are mutually independent.

We define the co-adaptations as the distance between Σ and Σ = diag(Σ).

Here trace(Σ) is a normalization term that defines the regularization strength.

Let x be the out of x with arbitrary noise (e.g. Dropout or RotationOut).

We assume that the noise should follow two assumptions: 1) zero-center: E[ x|x] = x; 2) non-trivial: Var[ x|x] = O (avoid that x always equals to x).

Consider the law of total variance, we have:

Let x Drop be the out of x after Dropout with drop rate p, and x Rot be the out of x after RotationOut with E θ tan 2 θ = (1 − p)/p, we have Lemma 1 (proof see Appendix A.3):

We can compute the co-adaptations of x (Assume c = 0):

Under zero-center assumption, Dropout with keep rate p reduces co-adaptation by p times, and the equivalent RotationOut reduces co-adaptation by p −

For Dropout-and other dropout-like methods, they add noise to different neurons independently, so cov( x i , x j |x) = 0.

The only term to reduce correlation coefficients in Equation 16 is

Under out non-trivial noise assumption, V ar[ x j |x] is always positive.

Thus non-trivial noise can always reduce co-adaptations.

For RotationOut, there is another term to reduce correlation coefficients:

In addition to increasing the uncertainty of each neuron as Dropout does, RotationOut can also reduce the correlation between two neurons.

In other words, inhibition noise.

Here we explain why we need a zero-center assumption and rotate the zero-centered features in Section 2.3.

Equation 14 and 16 show that the non-zero mean value can further reduce the coadaptations.

If we do not know the exact mean value, we do not know the exact regularization strength.

Suppose the neurons x ∼ N (0, 1) follow a normal distribution, and we apply Dropout on the ReLU activations y = ReLU(x).

With a keep rate 0.9, Dropout reduces the co-adaptations by 0.86 times, while Dropout reduces the co-adaptations by 0.61 times with a keep rate 0.7, which is a non-linear mapping and influenced by the mean value.

We rotate/drop the zero-centered features so that the regularization strength is independent with the mean value.

In this section, we evaluate the performance of RotationOut for image classification, object detection, and speech recognition.

First, we conduct detailed ablation studies with CIFAR100 dataset.

Next, we compare RotationOut with other regularization techniques using more data and higher resolution.

We test on two tasks: image classification on ILSVRC dataset and object detection on COCO dataset.

The CIFAR100 dataset consists of 60,000 colour images of size 32 × 32 pixels and 100 classes.

The official version of the dataset is split into a training set with 50,000 images and a test set with 10,000 images.

We conduct image classification experiments on the dataset.

Our focus is on the regularization abilities, so the experiment settings for different regularization techniques are the same.

We follow the setting from He et al. (2016) .

The network inputs are 32 × 32 and normalized using per-channel mean and standard deviation.

The data augmentation methods are as follows: first zero-pad the images with 4 pixels on each side to obtain a 40 × 40 pixel image, then randomly crop a 32 × 32 pixel image, and finally mirror the images horizontally with 50% probability.

For all of these experiments, we use the same optimizer: training for 64k iterations with batches of 128 images using SGD, momentum of 0.9, and weight decay of 1e-5.

We start with a learning rate of 0.1, divide it by 10 at 32k and 48k iterations, and terminate training at 64k iterations.

For each run, we record the best validation accuracy and the avergae validation accuracy of the last 10 epochs.

Each experiment is repeated 5 times and we report the top 1 (best and avergae) validation accuracy as "mean ± standard deviation" of the 5 runs.

We compare the regularization abilities of RotationOut and Dropout on two classical architectures: ResNet110 from He et al. (2016) and WideResNet28-10 from Zagoruyko & Komodakis (2016).

ResNet110 is a deep but not so wide architecture using 18 × 3 BasicBlocks (Zagoruyko & Komodakis, 2016) in three residual stages.

The feature map sizes are {32, 16, 8} respectively and the numbers of filters are {16, 32, 64} respectively.

WideResNet28-10 is a wide but not so deep architecture using 4 × 3 BasicBlocks in three residual stages.

The feature map sizes are {32, 16, 8} respectively and the numbers of filters are {160, 320, 640} respectively.

For ResNet110, we only apply RotationOut or Dropout (with the same rate) to all convolutional layers in the third residual stages.

FOr WideResNet28-10, we apply RotationOut or Dropout (with the same keep rate) to all convolutional layers in the second and third residual stages since WideResNet28-10 has much more parameters.

As mentioned ealier, we can use different distributions to generate θ.

and the regularization strength is controlled by E tan θ 2 = 1/p − 1.

We compare RotationOut with the corresponding Dropout.

We tried different distributions and found that the performance difference is very small.

We report the results of Gaussian distributions here.

Table 1 shows the results on CIFAR100 dataset with two architectures.

Table 1a and 1b are the results for ResNet110.

Table 1c and 1d are the results for WideResNet28-10.

Results in the same row compare the regularization abilities of Dropout and the equivalent keep rate RotationOut.

We can find dropping too many neurons is less effective and may hurt training.

Since WideResNet28-10 has much more parameters, the best performance is from a heavier regularization.

ImageNet Classification.

The ILSVRC 2012 classification dataset contains 1.2 million training images and 50,000 validation images with 1,000 categories.

We following the training and test schema as in He et al., 2016 ) but train the model for 240 epochs.

The learning rate is decayed by the factor of 0.1 at 120, 190 and 230 epochs.

We apply RotationOut with with normal distribution of tangent E tan θ 2 = 1/4 to convolutional layers in Res3 and Res4 as well as the last fully connected layer.

As mentioned earlier, RotationOut is easily combined with DropBlock idea.

We rotate features in a continuous block size of 7 × 7 in Res3 and 3 × 3 in Res4.

Table 2 shows the results of some state of the art methods and our results.

Our results are average over 5 runs.

Results of other methods are from Ghiasi et al. (2018) , and also regularize on Res3 and Res4.

Our result is significantly better than Dropout and SpatialDropout.

By using the DropBlock idea, RotationOut can get competitive results compared with state of the art methods and get a 2.07% improvement compared with the baseline.

ResNet-50 (He et al., 2016) 76.51 ± 0.07 93.20 ± 0.05 ResNet-50 + dropout(kp=0.7) (Srivastava et al., 2014) 76.80 ± 0.04 93.41 ± 0.04 ResNet-50 + DropPath(kp=0.9) (Larsson et al., 2016) 77.10 ± 0.08 93.50 ± 0.05 ResNet-50 + SpatialDropout(kp=0.9) (Tompson et al., 2015) 77.41 ± 0.04 93.74 ± 0.02 ResNet-50 + Cutout (DeVries & Taylor, 2017) 76.52 ± 0.07 93.21 ± 0.04 ResNet-50 + DropBlock(kp=0.9) (Ghiasi et al., 2018) 78.

COCO Object Detection.

Our proposed method can also be used in other vision tasks, for example Object Detection on MS COCO (Lin et al., 2014) .

In this task, we use RetinaNet (Lin et al., 2017) as the detection method and apply RotationOut to the ResNet backbone.

We use the same hyperparameters as in ImageNet classification.

We follow the implementation details in (Ghiasi et al., 2018) : resize images between scales [512, 768] and then crop the image to max dimension 640.

The model are initialized with ImageNet pretraining and trained for 35 epochs with learning decay at 20 and 28 epochs.

We set α = 0.25 and γ = 1.5 for focal loss, a weight decay of 0.0001, a momentum of 0.9 and a batch size of 64.

The model is trained on COCO train2017 and evaluated on COCO val2017.

We compare our result with DropBlock (Ghiasi et al., 2018) as table 3 shows.

Due to limited computing resources, we finetune the model from PyTorch library's pretraining ImageNet classification models while DropBlock method trained the model from scratch.

We think it is fair to compare DropBlock method since the initialization does not help increase the results as showed in the first two rows.

Our RotationOut can still have additional 0.3 AP based on the DropBlock result.

In this work, we introduce RotationOut as an alternative for dropout for neural network.

RotationOut adds continuous noise to data/features and keep the semantics.

We further establish an analysis of noise to show how co-adaptations are reduced in neural network and why dropout is more effective than dropout.

Our experiments show that applying RotationOut in neural network helps training and increase the accuracy.

Possible direction for further work is the theoretical analysis of co-adaptations.

As discussed earlier, the proposed correlation analysis is not optimal.

It cannot explain the difference between standard Dropout and Gaussian dropout.

Also it can not ex-plain some methods such as Shake-shake regularization.

Further work on co-adaptation analysis can help better understand noise-based regularization methods.

One example of such a matrix that rotates the (1, 3) dimensions and (2, 4) dimensions can be:

In Section 2, we mentioned the complexity of RotationOut is O(D).

It is because we can avoid matrix multiplications to get Rx.

For example, let the R be the operator generated by Equation 17, we have:

The sparse matrix in Equation 18 is similar to a combine of permutation matrix, and we do not need matrix multiplications to get the output.

The output can be get by slicing and an elementwise multiplication: x[3, 4, 1, 2] * [−1, 1, 1, −1].

Recall that E R = I, the marginalizing linear regression expression:

From Lemma one, we have Var

Write the second term of Equation 19 in the matrix form, we can get Equation 5.

The Dropout form is trivial.

We consider the RotationOut equation.

Denote x i as the i th term of x. The probability distribution of each element of x Rot is:

The joint distribution of each two elements of x Rot is:

So we have:

A.4 DROPOUT BEFORE BATCH NORMALIZATION Dropout changes the variance of a specific neuron when transferring the network from training to inference.

However, BN requires a consistent statistical variance.

The variance inconsistency (variance shift) in training and inference leads to unstable numerical behaviors and more erroneous predictions when applying Dropout-before BN.

We can easily understand this using Equation 13.

If a Dropout layer is applied right before a BN layer.

In training time, the BN layer records the diagonal element of Var[ x] as the running variance and uses them in inference.

However, the actul variance in inference should be the diagonal element of Var [x] which is small than the recorded running variance (train variance).

Li et al. (2019) argues:

P1 Instead of using Dropout, a more variance-stable form Uout can be used to mitigate the problem:

.

P2 Instead of applying Dropout-a (Figure 1) , applying Dropout-b can mitigate the problem.

P3 In Dropout-b, let r be the ratio between train variance and test variance.

Expanding the input dimension of weight layer D can mitigate the problem: D → ∞, r → 1.

Figure 1: Two types of Dropout.

The weight layer can be convolutional or fully connected layer.

We revisit these propositions and discuss how to mitigate the problem.

For Proposition 1, Uout is unlikely to mitigate the problem.

The Uout noise to different neurons are independent, so the variance shift is the only term to reduce co-adaptations in Equation 16.

Though Uout is variancestable, it provides less regularization, which is equivalent to Dropout with a higher keep rate.

Let w be i th row of of W and assume w i is uniformly distributed on the unit ball.

Since the length of w expands the training and testing variance with the same proportion, it does not affect the ratio between training and testing variance, and we can assume the length of w is fixed.

Though the expected variance shift is the same, the variance of the shift is different.

Let r(w) be the ratio between the training variance and the testing variance:

We have the following observation:

Observation.

If c > 0 which is the case that the activation function is ReLU.

The ratio in Dropout-b is more centered:

Sample n weights to make the weight layer W , the maximum ratio in Dropout-a is bigger than the maximum ratio in Dropout-b with high probability: max

According to this observation, Proposition 2 and 3 are basically right but might not be precise.

Dropout-b does help mitigate the problem but there might be other reasons.

The expected variance shift is the same in Dropout-a and b: D → ∞, r 1.

Dropout-b has more stable variance shift among different dimensions.

Dropout-a is more likely to have very big training/testing variance ratio, leading to more serious unstable numerical behaviour.

The ratio is fixed to be 1/p for any weights, i.e. Var w [r a (w)] = 0.

It leads to fewer unstable numerical behaviour since there is no extreme variance shift ratio, and we can modify BN layer's validation mode (reduce the running variance by 1/p times).

Zero-centered Dropout-a can be one solution to mitigate the variance shift problem.

We verified this claim on the CIFAR100 dataset using ResNet110.

We apply Dropout between the convolutions of all residual blocks in the third residual stage (18 dropout layers are added).

We test three types of Dropout with a keep rate of 0.5: 1) Dropout-a-centered, 2) Dropout-b) and 3) Dropout-b-centered.

Following (Li et al., 2019) , the experiments are conducted by following three steps: 1) Calculate the running variance of all BN layers in training mode.

It is the the training variance.

2) Calculate the running variance of all BN layers in testing mode.

It is the the testing variance.

Data augmentation and the dataloader are also kept to ensure that every possible detail for calculating neural variances remains exactly the same with training. .

For dropout-a-center, we reduce the running variance by 1/p times (We also tried this for the other two dropout, but the results are not better).

The obtained ratio r measures the variance shift between training and testing mode.

A smaller ratio r is better.

The results are averaged over 3 runs and shown in Figure   A .5 EXPERIMENT IN SPEECH RECOGNITION We show that our RotationOut can also help train LSTMs.

We conduct an Auto2Text experiment on the WSJ (Wall Street Journal) dataset (Paul & Baker, 1992) .

The dataset is a database with 80 hours T ×L where T is the length and L is the feature dimension for one time step.

The labels are character-based words.

We use a fourlayer bidirectional LSTM network to design a CTC (Connectionist temporal classification) Graves et al. (2006) model.

The input dimension, hidden dimension and output dimension of the four-layer bidirectional LSTM network are 40, 512, 137 respectively.

We use Adam optimizer with learning rate 1e-3, weight decay 1e-5 and batch size 32, and train the model for 80 epochs and reduce the learning rate by 5x at epoch 40.

We report the edit distance between our prediction and ground truth on the "eval92" test set.

Table 4 shows the performance of different regularization methods.

A.6 RETHINKING SMALL BATCHSIZE BATCHNORMALIZATION BN also introduces noise to the neurons by using the batch mean and variance.

The noise to different neurons/channels are independent, so the effect of BN's noise is similar to Dropout.

It is widely believed that the noise causes BN performance to decrease with small batch size (Wu & He, 2018; Luo et al., 2018) .

However, Dropout usually decrease the performance when the keep rate is very low.

We study the effect of BN's noise and argue that BN is not a linear operation.

The nonlinearity increases when the batch size decreases, which is also one reason for the small batch size BN's performance drop. (

The batch normalization operation records a running mean µ B and running variance σ 2 B to be used in testing:

We want to check whether the test mode formula can be a good estimation of the training mode formula.

Suppose we have a batch of data {x k } B k=1 .

Denote:

x k , σ

Let the function in 28 be f (x, B).

Easy to know that it is not a linear function (but BN assumes it should be y = x!).

Suppose the data follows normal distribution, we can plot f (x, B) by Monte Carlo sampling: Figure 3 shows that BN is not a linear operation.

The nonlinearity increases when the batch size decreases.

It is another important reason for the small batch size BN's performance drop.

To validate our conlusion, we propose cross normalization:

For each data in the batch, cross normalization uses the sample mean and variance except itself to comute its normalization mean and variance.

In this case, the expectation of operation on any data is striclty linear in expection, but it uses less data.

We do not intend to propose a better alternative for BN but want to check whether the nonlinearity is an important issue for BN when batch size is small.

If cross normalization can outperform BN in batch size case, then the nonlinearity is definately an important issue.

On CIFAR100 dataset, following the settings in our ablation study, ResNet50 with cross normalization has lower test loss when the batch size is 8 and 16.

But the test accuracy is almost the same in terms of 95% confidence interval since cross normalization leads to higher variance.

<|TLDR|>

@highlight

We propose a regularization method for neural network and a noise analysis method

@highlight

This paper proposes a new regularization method to mitigate the overfitting issue of deep neural networks by rotating features with a random rotation matrix to reduce co-adaptation.

@highlight

This paper proposes a novel regularization method for training neural networks, which adds noise neurons in an inter-dependent fashion.