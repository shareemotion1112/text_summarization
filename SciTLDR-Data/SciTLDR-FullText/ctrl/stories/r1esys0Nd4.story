Using higher order knowledge to reduce training data has become a popular research topic.

However, the ability for available methods to draw effective decision boundaries is still limited: when training set is small, neural networks will be biased to certain labels.

Based on this observation, we consider constraining output probability distribution as higher order domain knowledge.

We design a novel algorithm that jointly optimizes output probability distribution on a clustered embedding space to make neural networks draw effective decision boundaries.

While directly applying probability constraint is not effective, users need to provide additional very weak supervisions: mark some batches that have output distribution greatly differ from target probability distribution.

We use experiments to empirically prove that our model can converge to an accuracy higher than other state-of-art semi-supervised learning models with less high quality labeled training examples.

Probability is an abstract measure on how a certain event occurs independent of features of the events.

Knowing how likely a certain event occurs, people leverages such prior knowledge to their decision making.

For example, doctors know certain diseases are rare, even if they are told in terms of probabilities instead of "training examples".

Based on this knowledge, they make less predictions on these diseases than those common ones.

Do neural networks behave in a similar way?

Unfortunately, the answer is no. When we train a multi-layer perceptron(MLP) for MNIST classifier BID10 ) with limited labelled examples, the output distribution can be extremely biased in favor of some of the labels.

In Figure 1a , we compare the predicted number of labels with ground truth.

While the training accuracy is 1.0, the model clearly overfits to those training examples and leave labels between training data points undefined in high dimensional feature space.

As we plot the last hidden layer of a MLP trained with 50 labelled MNIST data as shown in Figure 1b , we find neural networks fail to learn the decision boundary correctly from a limited number of examples.

Thus, it is natural to consider introducing output label probability distribution as higher order knowledge when we train neural networks.

Different from traditional logical constraints BID22 ) or functional constraints BID18 , we propose a novel embedding space probabilistic constraint.

Because of the sparsity of high dimensional feature space with only a few labeled examples, we perform our probabilistic constraint on neural network's embedding space, which is constructed unsupervisedly by projecting data into low dimensional space through autoencoder.

Based on observation by BID21 , BID23 , embedding space preserves information of separations of different label clusters.

In the embedding space, we pool softmax activation (a) Strong imbalanced output distribution of labels when training set is limited (b) Chaotic embedding space in the hidden layer of the classifier trained with 50 labelled examples Figure 1 : Limited training data cannot train neural networks to learn accurate decision boundaries outputs and optimize towards target distribution.

By training with very few high quality labelled examples and marking on batches that have output distribution greatly different from target probability distribution, we use experiments to empirically prove that our model can converge to a high accuracy faster than state-of-art semi-supervised learning methods.

Weak Supervision Current supervised representation learning algorithms have gained great success on various tasks in computer vision BID4 , BID8 ), natural language processing BID19 , BID1 ) with little domain knowledge, but they require large quantity and high-quality labels for training.

Thus, there is a growing trend of research that address this problem by transferring knowledge learned from different datasets BID2 , BID17 ) or introducing higher level knowledge.

In this work, we consider incomplete weak supervision problem BID24 ).

A typical incomplete supervision problem BID3 ) is formulated as following: with a dataset {X, Y } that consists of labeled dataset X 1 = {X 1 , y 1 } and unlabeled dataset X 2 = {X 2 , y 2 }, where {y 2 } is not visible during training.

|X 1 | |X 2 |.

This problem can usually be tackled by state-of-art semi-supervised learning algorithms like AtlasRBF BID13 ), Neural Rendering Model BID6 )or LadderNet BID14 or using novel approaches such as logical constraints BID22 ).

While they still rely on certain amount of high quality labeled data, while in this work, we further decrease the number of labeled data needed for convergence.

Learning With Constraints Learning with constraints takes various higher order domain knowledge into the optimization of neural networks.

Based on domain knowledge, different constraints are effective on different tasks.

For example, BID12 uses linear constraints on the output space and optimizes the training objective as a biconvex optimization for linear models to perform dense pixelwise semantic segmentation.

Frameworks such as semantic loss by BID22 and logical loss by BID7 specify logic rules when training neural networks.

BID18 propose a novel framework that one can learn physical or causal relationship without labels.

In this work, we consider the case where limited labeled examples lead to biased output distribution.

Different from these arithmetic or logical constraints, we consider placing an output probability constraint.

In this section, we state our problem formulation and describe the proposed algorithm and architecture for this problem.

Higher-order Knowledge Formulation Based on the incomplete weak supervision defined in Section 2, we specify our introduced higher order knowledge.

We assume from our domain knowledge of output probability distribution, the model can acquire a set Q = {(k, P(Y = k) + } k???{Y }) .

One thing to note is that domain knowledge distribution Q does not necessarily cover all k ??? Y .

We use ??? R drawn from Gaussian distribution such that ??? N (0, ?? 2 ) to reflect the variance of human domain knowledge from true Y .

We use ?? = 0.05 throughout this paper.

We need a training algorithm A({X 1 , y 1 }, X 2 , Q}) such that it trains a multi-layer perceptron DISPLAYFORM0 , where W i is model's weights, ?? is an nonlinear function like ReLU, and ?? is the softmax function.

This algorithm minimizes the loss function : {Y, f (X)} ??? ??? R.Batchwise Probability Constraint Following from this problem formulation, we define a loss term c : R 2 ??? ??? R to regulate the output distribution.

We regard a single update batch {X , Y } ??? {X, Y } with size c as the unit of output probability distribution.

Inspired by BID11 and BID5 , the activation of final softmax layer of classifier ??(??) can reflect neural network's confidence towards a certain label.

Instead of performing counting the arguments of the maxima for all labels, which is not inefficient, we consider calculating the mean pooling of all the activation outputs BID20 ).

It can be written mathematically as DISPLAYFORM1 Here we use f k to denote the softmax activation of the kth label.

This potentially improves the accuracy of detecting low confidence or out-of-distribution examples.

A basic flowchart of our mechanism can be found in FIG0 .It is natural to use Kullback-Leibler (KL) divergence as a metric of our output distribution different from the reference domain knowledge probability distribution Q, that is, DISPLAYFORM2 One may notice the probability of labels in the batch does not always reflect the domain knowledge distribution Q. That is, DISPLAYFORM3 For the simplicity of this work, we assume additional but very weak supervision on identifying some of those batches and using different but noisy batch probability distribution.

However, this supervision can be easily done through at-a-glance (abriel Ryan (2018)) supervision or auto-regressive algorithms similar to BID15 .

Our proposed algorithm and its convergence analysis can be found in Appendix A.

In order to use existing unlabeled data to draw decision boundaries, we propose to jointly optimize this probability constrained classifier with an embedding space regularizer.

Embedding is a lower dimensional form that structurally preserves data from original hyper-dimensional space.

In our case, we treat a hyperparameter ith hidden layer of perceptron E(x) as our embedding space, where DISPLAYFORM0 , where the dimension of E(x) should be much smaller than dimension of input x. BID23 propose using unsupervised loss can preserve information of separations between different label clusters.

Thus, we adopt the structure of decoder of autoencoder and define a multi-layer neural network D(??) as a decoder of our embedding space.

For a single batch {X , Y }, our loss function for training a separation-preserving embedding space by reconstructing the original input, that is, DISPLAYFORM1 General Framework Our proposed method uses unsupervised loss r to construct an embedding in low dimensional space, uses limited labeled data to identify the cluster location in the embedding space by original classification loss original , and uses domain knowledge of output probability distribution to determine the actual decision boundaries.

Then our updating loss function is DISPLAYFORM2 , where ?? 1 , ?? 2 and ?? 3 are hyperparameter constants.

Experiment setup We evaluate our proposed embedding space probabilistic constraint in semisupervised learning setting.

Using the similar base multilayer perceptron model as in BID14 and BID22 All the experiments are repeated five times with different seeds.

We add an additional embedding layer with width 40, and the decoder has a symmetric architecture as the feed forward neural network.

Model Description To guarantee that our comparison focuses on output probability distribution instead of one single instance's label, we train our models with batch size 128.

We experiment our model under different level of constraints.

Datasetwise probability constraints assumes the target output should be all 10%, and the noisy datasetwise probability constraints adds a random noise drawn from N (0, 0.3) to simulate user's knowledge.

Also, we use batchwise probability constraint, which assumes we know the probability of labels in every batch, as an upper bound for our algorithm.

We compare our model with other state-of-art semi-supervised learning models BID13 , BID14 ), and logical constraint model BID22 .

Since we require more human supervision than other semi-supervised learning models, we use their results to demonstrate our model can converge to high accuracy with much less high quality labeled examples.

We also choose other baselines models without both losses and without embedding loss to show the benefit of our architecture.

Accuracy/# of labelled per class 3 5 10 all AtlasRBF BID13 73.58 ?? 0.95 84.28 ?? 0.21 91.54 ?? 0.13 98.20 ?? 0.25 Ladder Net BID14 Table 3 : Semi-supervised learning on CIFAR 10 dataset algorithms, we need far fewer high quality training examples to reach high accuracy.

Thus, we conclude jointly optimizing the output constraint with hypothesis can draw a decision boundary with smaller labelled training data than other state-of-art methods.

Our focus is to show the power of a very weak labelling method without high quality labelling technique.

We leave it as a future research direction to design an auto-regressive algorithm that requires less supervisions.

In addition, since our formulation sums all the activation functions together as a measure of confidence instead of counting-based probabilities, this allows us to use it as a future direction on confidence of classification in semi-supervised learning setting.

In Bootstrap phase, we require users to provide an entry supervision mark {??}. It marks entries that has unusually high probabilities of occurrence from target probability distribution.

We need this extra supervision because reliance on model's own probabilistic judgement might converge to a wrong model.

Feeding additional but weak supervision can give model enough information to judge if current probability distribution has such unusual batches.

We also claim that we can bound it by concentration inequalities, as shown in Theorem 1.

The probability of this mark happens can be bounded by Theorem 1.Theorem 1 For a single batch {X, y} with size c, on a label k with domain knowledge output probability Q(k).

For all > 0, we have DISPLAYFORM0 2 )Proof of Theorem 1.1 This claim directly follows from Hoeffding inequality, which we state as following: For independent bounded random variables X i ??? [a i , b i ] and for all t > 0, which has a ??? X ??? b, for all , we have DISPLAYFORM1 The proof can be found in BID16 .In our case, our random variable is P(|Y = k), ranging from [0, 1], while the target distribution, by higher knowledge, is E[P(|Y = k)] = Q(k).

With batch size c, we have DISPLAYFORM2 We state our bootstrap algorithm in Algorithm 1.

When we found a marked batch with higher probability, we redistribute our new target distribution based on its original target distribution.

When an output probability is marked, we rescale the target distribution of the label q marked to adapt the higher-than-usual probability by Equation 2.

A detailed derivation can be found in Appendix B.2.

DISPLAYFORM3 Algorithm 1: Bootstrap Probabilistic Constraint Data: Training batch {X, y} ??? R c??m ?? R n , entry supervision mark {??}, error threshold , Domain knowledge output distribution Q ??? R n , neural network f Feed Forward f (X) and pool output activations by DISPLAYFORM4 Train the network f with {X, y, q} by Equation 3;The boundary between bootstrap phase and auto-regressive phase is a hyper-parameter.

In this work, we use validation set to observe the accuracy of current model.

When the accuracy on validation set is larger than 70%, we enter auto-regressive phase.

A convergence plot example can be found in FIG1 .

DISPLAYFORM5 Train the network f with {X, y, q} by loss function from Equation 3; Lemma 1 Let f be a feed forward neural network with softmax as the last layer's activation function.

Then given a batch {X, y} ??? R c??m ?? R n , the mean-pooling ??? = 1 c c i=0 f (X i ) is a probability distribution.

Proof of Lemma 1.1 Without loss of generality, we write function f (x) as ??(??) ??? g(x) for some function g(x).

Then for all x ??? X, f (x) is in range of ??(??), that is, ???y ??? R n , f (x) = ??(y).

Then for a vector z ??? R n , for any label i, DISPLAYFORM6 Then we perform mean-pooling on ??(y) i , we have DISPLAYFORM7 The pooled output activation ??? i is trivially larger than 0, and since ??(z) i < 1 for all z, DISPLAYFORM8 Then we conclude it fullfills all the axiom of a probability distribution.

Since ??? is a probability distribution by lemma 1, we can apply KL divergence to ??? from our target distribution Q. With specified constraints set K, with f k (x) for the specific dimension of k ??? K, we have DISPLAYFORM9 For a single training example x j ??? X, the features for DISPLAYFORM10 ..x j n } are dependent, so we cannot apply statistical bounds by entries.

However, when we compare training examples in the same batch, that is, x j 1 and x j+1 1 , they are independent.

As a result, we can still use the Proof of Theorem 1.1.

When we would like to bound DISPLAYFORM11 In this case, we find a safe margin d that controls the tradeoff between human supervision and training batch accuracy.

Directly applying the upper bound is empirically fine when we perform our experiments.

This section includes main implementation details not included in main text.

MNIST MNIST dataset BID10 ) is a dataset of handwritten digits, which has 60000 training images and 10000 testing images.

For each of the image, it is a grey-scale 28 ?? 28 matrix that belongs to 10 classes from 1 to 10.FASHION a dataset of clothes that possess similar structure as MNIST, which has 60000 training images and 10000 testing images.

For each of the image, it is a grey-scale 28 ?? 28 matrix that belongs to 10 classes for different types of clothes.

CIFAR-10 CIFAR-10 (Krizhevsky FORMULA10 ) is a dataset that contains colored images with size 32 ?? 32.

Each image belongs to one of ten classes like dog, cat, car.

The training set has 50000 images and the testing set has 10000 images.

Selection In order to make the probability the same for all classes, we keep the same number of images among 10 classes.

We choose the number as the minimum number of examples out of all classes.

For example, for MNIST dataset, it has numbers of examples 5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851 respectively from 0 to 1.

Since 5421 is the minimum for all labels, we choose 5421 as the number of examples we use.

Based on this threshold, we choose the training examples randomly from the dataset.

To make examples from different classes balanced, we choose randomly (number of labeled examples/number of classes) number of examples.

Processing We first normalize all images scale from 0 to 1.

Then we have 50% of chance to perform two operations on training examples to prevent overfitting of the training set: adding Gaussian noise and cropping.

For Gaussian noise, we add to the image another matrix of random noise drawn from Gaussian distribution with mean 0 and standard deviation 0.3.

We crop the image by three pixels.

For example, if the original image has pixel 28 ?? 28, our cropped image has 25 ?? 25.

Multi-layer Perceptron We evaluate our proposed embedding space probabilistic constraint in semi-supervised learning setting.

Using the similar base multilayer perceptron model as in BID14 and BID22 with layer of the size 784-1000-500-250-250-250-40-10, except adding a layer with width 40 as embedding layer, we feed the output of the embedding layer to another multilayer deocoder with size 40-250-250-1000-784.

We also perform batch normalization and dropout with a dropout rate 50% to increase robustivity of our model.

To compare fairly with other state-of-art models, we adopt a basic 10-layer architecture similar to BID22 for classification.

For every three layers, we use one convolution layer with activation function ReLU, one 2 ?? 2 max-pool layer with stride 2.

We insert one layer of dropout with a dropout rate 0.5 and one batch normalization layer as we described in multi-layer perceptron.

For the decoder, we use a symmetric decoder.

For every three convolution layers, we use a upsampling layer.

We insert batch normalization layer and dropout layer between them.

The embedding space has a vector with dimension 50.

Detailed hyperparameter is as following: we use Stochastic Gradient Descent(SGD) to update our neural network with a learning rate 10e ??? 4.

For simplicity, we choose ?? 1 , ?? 2 and ?? 3 to be 1.

A rule of thumb is to make ?? 1 original ??? ?? 2 c .

When they possess similar values, they can converge to the solution quickly.

The convergence of our algorithm can be shown in FIG1 .In the cold start session, with batch size 128, we choose to find batch output probabilities p / ??? [0.02, 0.25].

Empirically this covers 98% of data, we can safely set them with probability 0.1, while the rest but obvious data, we set them a manual probability, in our case, we set them 0.02 and 0.28.

<|TLDR|>

@highlight

We introduce an embedding space approach to constrain neural network output probability distribution.

@highlight

This paper introduces a method to perform semi-supervised learning with deep neural networks, and the model achieves relatively high accuracy, given a small training size.

@highlight

This paper incorporates label distribution into model learning when a limited number of training instances is available, and proposes two techniques for handling the problem of output label distribution being wrongly biased.