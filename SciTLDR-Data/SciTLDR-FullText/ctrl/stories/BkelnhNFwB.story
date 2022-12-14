In the last few years, deep learning has been tremendously successful in many applications.

However, our theoretical understanding of deep learning, and thus the ability of providing principled improvements, seems to lag behind.

A theoretical puzzle concerns the ability of deep networks to predict well despite their intriguing apparent lack of generalization: their classification accuracy on the training set is not a proxy for their performance on a test set.

How is it possible that training performance is independent of testing performance?

Do indeed deep networks require a drastically new theory of generalization?

Or are there measurements based on the training data that are predictive of the network performance on future data?

Here we show that when performance is measured appropriately, the training performance is in fact predictive of expected performance, consistently with classical machine learning theory.

Is it possible to decide the prediction performance of a deep network from its performance in training -as it is typically the case for shallower classifiers such as kernel machines and linear classifiers?

Is there any relationship at all between training and test performances?

Figure 1a shows that when the network has more parameters than the size of the training set -which is the standard regime for deep nets -the training classification error can be zero and is very different from the testing error.

This intriguing lack of generalization was recently highlighted by the surprising and influential observation (Zhang et al. (2016) ) that the same network that predicts well on normally labeled data (CIFAR10), can fit randomly labeled images with zero classification error in training while its test classification error is of course at chance level, see Figure 1b .

The riddle of large capacity and good predictive performance led to many papers, with a variety of claims ranging from "This situation poses a conceptual challenge to statistical learning theory as traditional measures of model complexity struggle to explain the generalization ability of large artificial neural networks... " Zhang et al. (2016) , to various hypotheses about the role of flat minima Keskar et al. (2016) ; Dinh et al. (2017) ; Chaudhari et al. (2016) , about SGD Chaudhari & Soatto (2017) ; Zhang et al. (2017) and to a number of other explanations (e.g. Belkin et al. (2018) ; Martin & Mahoney (2019) ) for such unusual properties of deep networks.

We start by defining some key concepts.

We call "loss" the measure of performance of the network f on a training set S = x 1 , y 1 , ?? ?? ?? , x N , y N .

The most common loss optimized during training for binary classification is the logistic loss L(f ) = 1 N N n=1 ln(1 + e ???ynf (xn) ).

We call classification "error" 1 N N n=1 H(???y n f (x n )), where y is binary and H is the Heaviside function with H(???yf (x)) = 1 if ???yf > 0 which correspond to wrong classification.

There is a close relation between the logistic loss and the classification error: the logistic loss is an upper bound for the classification error.

Thus minimizing the logistic loss implies minimizing the classification error.

The criticism in papers such as Zhang et al. (2016) refers to the classification error.

However, training minimizes the logistic loss.

As a first step it seems therefore natural to look at whether logistic loss in training can be used as a proxy for the logistic loss at testing.

The second step follows from the following observation.

The logistic loss can always be made arbitrarily small for separable data (when f (x n )y n > 0, ???n) by scaling up the value of f and in fact it can be shown that the norm of the weights of f grows monotonically with time

during gradient descent.

Notice that multiplying f by any positive number does not affect its sign and thus does not change its behavior at classification.

The second step is then to consider the logistic loss of the normalized networkf with f = ??f , with ?? = ?? 1 ...?? K where ?? i is the Frobenius norm of the weight matrix of layer i.

Now we show that by measuring the performance of a trained network on the training set in terms of the quantities described above, classical generalization holds: the performance of the network on the training set becomes a good qualitative proxy for the expected future performance of the network.

The precise procedure involves normalizing the output of the trained network by dividing it by a single number -the product of the Frobenius norms of the weight matrices at each layer -and then measuring the cross-entropy loss.

Figure 2a) shows the cross entropy test loss plotted versus the training loss for networks with the same architecture trained with different initializations (this is a way to get networks with different test performance).

There is no clear relation between training and test cross-entropy loss in the same way that there is not between classification error in training and testing.

Normalizing each network separately yields Figure 2b ): now the test loss is tightly predicted by the training loss.

We emphasize that normalization does not change the classification performance of the networks.

In fact, since (binary) classification depends only on the sign of f (x), it is not changed by normalization to any ball.

The figure suggests that the empirical cross-entropy loss off on the training set predicts rather well how networks will perform on a test set.

Impressively, the same normalization process correctly predicts the test loss when the training is on randomly labeled data.

This apparent lack of predictivity of the training performance on randomly labeled data was at the core of Zhang et al. criticism of machine learning theory (Zhang et al. (2016) ).

Figure 2 shows that it is in fact possible to gauge just from the training performance that the network trained on randomly labeled examples will have chance performance on a test set even if its classification error in training is always zero.

As shown in Figure 2 the data point corresponding to the randomly trained network still satisfies approximately the same linear relationship, as explained by the classical theory.

Thus, while Figure 2 shows that the normalized train cross-entropy can predict the performance of the of the normalized test cross-entropy, Figure 4

We define a deep network with K layers with the usual elementwise scalar activation functions ??(z) : R ??? R as the set of functions

where the input is x ??? R d , the weights are given by the matrices W k , one per layer, with matching dimensions.

We use the symbol W as a shorthand for the set of W k matrices k = 1, ?? ?? ?? , K. For simplicity we consider here the case of binary classification in which f takes scalar values, implying that the last layer matrix W K is W K ??? R 1,K l and the labels are y n ??? {???1, 1}. There are no biases apart form the input layer where the bias is instantiated by one of the input dimensions being a constant.

The activation function in this paper is the ReLU activation.

We denote the network as f = f (W 1 , ?? ?? ?? , W K ; x) where x is the input and the weight matrices W k are the parameters.

We callf the network with normalized weights matrices,

Consider different asymptotic minima of the empirical loss obtained with the same network architecture on the same training set by minimization of the cross-entropy loss with different initial conditions (see Appendix).

We obtain different test losses, depending on initial conditions.

The question is whether their test performance can be predicted from empirical properties measured only on the training set.

1 used in our experiments are multiclass problems.

In this theory section we discuss for simplicity the simpler case of binary classification.

Exactly the same steps in the theoretical arguments below apply to the cross-entropy loss (because Equations 1,3 apply, see Appendix).

Consider the structure of the loss for a deep network.

The "positive homogeneity" property of ReLU networks implies the following property:

where W K = ?? kWK and ||W K || = 1 and ?? 1 ?? ?? ?? ?? K = ?? .

1 MNIST and CIFAR100 results are in the appendix This property is valid for layer-wise normalization under any norm.

We emphasize again that

have the same classification performance on any data set.

Furthermore, during gradient descent on separable data -most data sets are separable by overparametrized deep networks -it can be shown (see Poggio et al. (2018) ) that the ?? factor continue to increase with time towards infinity, driving an exponential type loss to zero without affecting the classification performance, which only depends on the sign of y n f (x n )???n.

As we discussed earlier, it seems that different networks corresponding to different empirical minimizers 2 could be evaluated in terms of their normalized formf (x).

The intuition is similar to the linear case in which the classifier w T x depends on the unit vectorw = w |w| while |w| diverges to infinity Soudry et al. (2017) ; Poggio et al. (2018) .

2 In general an overparametrized network may have a large number of global minima, see for instance .

To assessf (x) we compute its logistic loss (which is the special case of cross-entropy in the binary case)L

Of course for separable data (y n f (x n ) > 0, ???n) the loss off is larger than the loss of f since the negative exponent is smaller.

Figure 2 uses L 2 for layer-wise normalization.

We are now ready to explain the results of Figure 2 in terms of classical machine learning theory.

A typical generalization bound that holds with probability at least (1 ??? ??), ???f ??? F has the form Bousquet et al. (2003) :

where

is the expected loss, R N (F) is the empirical Rademacher average of the class of functions F measuring its complexity; c 1 , c 2 are constants that reflect the Lipschitz constant of the loss function and the architecture of the network.

We use the bound in Equation 3 and the key observation that the Rademacher complexity satisfies the property,

because of homogeneity of the networks.

Then, the bound on the cross-entropy loss for the unnormalized network gives

sinceL(f ) ??? 0.

Considering the corresponding bound for the cross-entropy loss of the normalized network scaled with any desired scaling R gives

In our experiments we find that R N (F) is small for the value of N of our datasets and R N (F) is large.

Equation 5 implies then that the unnormalized test loss will be quite different from zero and thus different from the unnormalized train loss.

On the other hand, in Equation 6 the terms

2N are small, implying that the normalized test loss is very close to the normalized train loss.

Thus Equation 6 with R = 1 shows that L(f ) is bounded byL(f ), predicting a bound in terms of a linear relationship with slope one and a small offset between L(f ) andL(f ).

The prediction is verified experimentally (see Figure 2 ).

Notice that both homogeneity of the ReLU network and separability, that is very small cross-entropy loss, are key assumptions in our argument.

The first applies to networks with a linear or ReLU activation but not to networks with other activation functions.

The second is usually satisfied by overparametrized networks.

Thus Equation 6 shows that the training cross-entropy loss is a very good proxy of the cross-entropy loss at testing, implying generalization for a relatively small number of examples N .

Notice that for all the networks in Figure 2 , classification performance at training is perfect and that scaling does not change the sign of the networks outputs and therefore their behaviour in terms of classification.

In particular, Figure 2 shows that performance on the randomly labeled training set when measured for the normalized network is bad (despite classification being at zero error) predicting correctly bad performance on the test set.

As we mentioned, Figure 4 b shows that there is a good correlation between crossentropy loss and classification performance: empirically the ranking between the different classifiers is mostly the same for crossentropy vs classification loss. (2017) and of the upper bounds Equations 3 and 6.

Lower bounds, however, are not available.

As a consequence, the theory does not guarantee that among two (normalized) networks, the one with lower cross-entropy loss in training will always have a lower classification error at test.

This difficulty is not specific to deep networks.

It is common to approaches using a surrogate loss function.

The empirical evidence however supports the claim that there is a roughly monotonic relationship between training (and testing) loss of the normalized network and its expected classification error: Figure 4b shows an approximately monotonic relation between normalized test cross-entropy loss and test classification error.

The linear relationship we found means that the generalization error of Equation 3 is small once the complexity of the space of deep networks is "dialed-down" by normalization.

It also means that, as expected from the theory of uniform convergence, the generalization gap decreases to zero for increasing size of the training set (see Figure 1 ).

Thus there is indeed asymptotic generalization -defined as training loss converging to test loss when the number of training examples grows to infinity -in deep neural networks, when appropriately measured.

The title in Zhang et al. (2016) "Understanding deep learning requires rethinking generalization" seems to suggest that deep networks are so "magical" to be beyond the reach of existing machine learning theory.

This paper shows that this is not the case.

On the other hand, the generalization gap for the classification error and for the unnormalized cross-entropy is expected to be small only for much larger N (N must be significantly larger than the number of parameters).

However, consistently with classical learning theory, the cross-entropy loss at training predicts well the cross-entropy loss at test when the complexity of the function space is reduced by appropriate normalization.

For the normalized case with R = 1 this happens in our data sets for a relatively "small" number N of training examples as shown by the linear relationship of Figure 2 .

The classical analysis of ERM algorithms studies their asymptotic behavior for the number of data N going to infinity.

In this limiting regime, N > W where W is the fixed number of weights; consistency (informally the expected error of the empirical minimizer converges to the best in the class) and generalization (the empirical error of the minimizer converges to the expected error of the minimizer) are equivalent.

This note implies that there is indeed asymptotic generalization and consistency in deep networks.

However, it has been shown that in the case of linear regression, for instance with kernels, there are situations -depending on the kernel and the data -in which there is simultaneously interpolation of the training data and good expected error.

This is typically when W > N and corresponds to the limit for ?? = 0 of regularization, that is the pseudoinverse.

It is likely that deep nets may have a similar regime, in which case the implicit regularization described here, with its asymptotic generalization effect, is just an important prerequisite for a full explanation for W > N -as it is the case for kernel machines under the square loss.

The results of this paper strongly suggested that the complexity of the normalized network is controlled by the optimization process.

In fact a satisfactory theory of the precise underlying implicit regularization mechanism has now been proposed Soudry et al. (2017) As expected, the linear relationship we found holds in a robust way for networks with different architectures, different data sets and different initializations.

Our observations, which are mostly relevant for theory, yield a recommendation for practitioners: it is better to monitor during training the empirical "normalized" cross-entropy loss instead of the unnormalized cross-entropy loss actually minimized.

The former matters in terms of stopping time and predicts test performance in terms of cross-entropy and ranking of classification error.

More significantly for the theory of Deep Learning, this paper confirms that classical machine learning theory can describe how training performance is a proxy for testing performance of deep networks.

While the theoretical explanation in the main text applies to the case of binary classification, the extension to multi-class case follows straightforwardly.

Recall some definitions for neural networks with multiple outputs.

Let C be the number of classes -the neural network is then a vector f (W ; x) ??? R C .

The component fj(W ; x) denotes the j-th output.

The dataset is again composed of examples xn ??? R d and labels are now yn ??? [C].

Note that nothing here changes in regards to homogeneity, and we again can define a normalized network

The main theoretical arguments depend on the generalization bounds of the form

As the right hand side depends on the neural networks, which do not change in any substantial way, all that remains is understanding the multi-class loss.

To transform the outputs of the network into probabilities, the Softmax function is used

The cross-entropy loss is then defined simply a??

It's very easy to see that this reduces to the logistic loss in the binary case.

Classification now depends only on the margin ??n = fy n (W ; x) ??? max j =yn {fj(W ; x)} -if ??n > 0 then the example is correctly classified.

This means that, again, classification only cares about the sign of the margin and not the normalization of the neural network.

One final property of note is that for separable data, the loss monotonically decreases with increasing ??.

To see this, let us write ??nj = fy n (W ; x) ??? fj(W ; x), which is a positive quantity in the separable case.

Additionally define gn = ??? log j =yn e ????? nj = ??? log j =yn e ??????? nj , which is clearly a monotonic function of ?? if all ??nj > 0.

We can now rewrite the Cross-entropy loss a??

which implies that we can drive the loss to 0 by increasing ?? ??? ???.

Top The top left graph shows testing vs training cross-entropy loss for networks trained on the same data sets but with different initializations.

The top right graph shows the testing vs training loss for the same networks, normalized by dividing each weight by the Frobenius norm of its layer.

Notice that all points have zero classification error at training.

The red point on the top right refers to a network trained on the same CIFAR data set but with randomized labels.

It shows zero classification error at training and test error at chance level.

The top line is a square-loss regression of slope 1 with positive intercept.

The bottom line is the diagonal at which training and test loss are equal.

The networks are 3-layer networks; the first layer is convolutional, 64 filters of size 5x5, stride 2, padding 2, no bias, ReLU activation; the second layer is also convolutional, 64 filters of size 5x5, stride 2, padding 2, no bias, ReLU activation; the third layer is fully connected, input size 64*8*8, output size 10, no bias, softmax activation.

The training is on the CIFAR-10 dataset with 50k training examples, 10k testing examples.

The network used for the point in red was trained similarly except the testing set and training set labels were randomized.

No data augmentation was performed, but data were normalized to mean (

The bottom graphs show similar experiments for ResNet-56 with 56 layers, demonstrating that our observations hold for state-of-the-art networks (testing errors ranging from 7% to 9%; on this 10-classes classification problem chance performance is 90% error).

We again emphasize that the performance in classification of normalized and unnormalized networks is exactly the same.

The normalization in the case of ResNet was performed by using the norm of the output of each network on one example from CIFAR-10, because we found it computationally difficult to directly evaluate the effective norm of the residual layers.

The networks were trained for 200 epochs with learning rate = 0.01 for the first 100 epochs and learning rate = 0.001 for the second 100 epochs.

SGD was used with batch size of 128 and shuffled training data with random crop and horizontal flip data augmentation.

First we start with a common observation: even when two networks have the same architecture, same optimization meta parameters and same training loss, they usually have different test performances (i.e. error and loss), because the stochastic nature of the minimization process leads to convergence to different minima among the many existing in the loss landscape ; .

With standard settings the differences are usually small (though significant, as shown later).

We use therefore two approaches to magnify the effect:

??? Initialize networks with different levels of "random pretraining": the network is pretrained for a specified number of epochs on "corrupted" training data -the labels of a portion of the examples are swapped with each other in a random fashion.

??? Initialize the weights of the networks with different standard deviations of a diagonal Gaussian distribution.

As it turns out, different standard deviations yield different test performance.

Similar techniques have been used previously Neyshabur et al. (2017); Liang et al. (2017) .

We show the results of "random pretraining" with networks on CIFAR-10 ( Figure 5 ) and CIFAR-100 ( Figure  7 ) and initialization with different standard deviations on CIFAR-10 ( Figure 6 ) and CIFAR-100 (Figure 8 ).

Our observations are that the linear relationship between train loss and test loss for normalized networks hold in a robust way under several conditions:

??? Independence from Initialization: The linear relationship is independent of whether the initialization is via pretraining on randomly labeled natural images or whether it is via larger initialization, as shown by Figures 11 and 12.

??? Independence from Network Architecture: The linear relationship of the test loss and train loss does not depend on the network architectures we tried.

??? Independence from Data Set: Figures 10, 11 , 12, 9 show the linear relationship on CIFAR10 while Figures 17 and 18 show the linear relationship on CIFAR100.

??? ??? Normalization is independent of training loss: Figure 10 shows that networks with different cross-entropy training losses (which are sufficiently small to guarantee zero classification error), once normalized, show the same linear relationship between train loss and test loss.

One way to formalize the upper bound on classification error by the logistic loss is to consider the excess classification risk R(f ) ??? R * , where R(f ) is the classification error associated with f and R * is the Bayes error Bartlett et al. (2003) .

Let us call as before L(f ) the expected cross-entropy loss and L * the optimal expected cross entropy loss.

Then the following bound holds for binary classification in terms of the so-called ??-transform of the logistic loss :

where the ?? function for the logistic is similar to the ?? for the exponential loss which is ??(x) = 1 ??? ??? 1 ??? x 2 .

The key point here is that ?? ???1 is monotonic: minimizing the logistic or cross-entropy loss implies minimizing the classification error.

Our arguments imply that among two unnormalized minimizers of the exponential loss that achieve the same given small empirical lossL = , the minimizer with higher product of the norms ??1, ?? ?? ?? , ??K has the higher capacity and thus the highest expected loss L. Experiments support this claim, see Figure 13 .

Notice the linear relationship of test loss with increasing capacity on the top right panels of Figure 11 , 12, 17, 18.

Product of L2 Norms of Weights from All Layers (Unormalized Net) Figure 13 shows that when the capacity of a network (as measured by the product norm of the layers) increases, so does the test error.

Neyshabur et al. (2017); Liang et al. (2017)

This section shows figures replicating the main results on the MNIST data set.

Figures 15 and 16 show that the linear relationship holds after normalization on the MNIST data set.

Figure 16 shows the linear relationship holds after adding the point trained only on random labels.

This section is about replicating the main results on CIFAR-100.

Figure 7 shows how different test performance can be obtained with pretraining on random labels while Figure 8 shows that different increasing initializations are also effective.

Figures 17 and 18 show that the linear relationship holds after normalization, regardless of whether the training was done with pretraining on random labels or with large initialization.

This section is about replicating the main results with ResNets (see main text figures).

To avoid using the product of layerwise norms -whose form is not obvious for ResNets -we normalized the different architectures using their output on the same single image, exploting the fact that all the networks only differ in their initialization (for this reason we did not plot the RL point because this shortcut to normalization does not apply in the case of a different training set with random labels).

While in this article we use the term generalization when the offset in the difference between training and test losses is small, the technical definition of "generalization" just requires that the offset decreases with increasing N .

This means that for all f ??? F, limN?????? |L(f ) ???L(f )| ??? 0, since in general we expect the Rademacher complexity to decrease as N increases.

As Figure 19 shows, |L(f ) ???L(f )| does in fact decrease with increasing N (notice that L(f ) ???L(f ) around N = 50000 for normalization to the unit L2 ball of our network).

These arguments do not depend on the specific form of the Rademacher complexity.

A typical margin bound for classification Bartlett & Shawe-Taylor (1998) is

where ?? is the margin, L binary (f ) is the expected classification error, Lsurr(f ) is the empirical loss of a surrogate loss such as the logistic or the exponential.

For a point x, the margin is ?? ??? y??f (x).

Since RN (F) ??? ??, the margin bound says that the classification error is bounded by 1 ?? that is by the value off on the "support vectors" -that is on the xi, yi s.t arg minn ynf (xn).

We have looked at data showing the test classification error versus the inverse of the margin.

The data are consistent with the margin bound in the sense that the error increases with the inverse of the margin and can be bounded by a straight line with appropriate slope and intercept.

Since the bound does not directly provide slope and intercept, the data do not represent a very convincing evidence.

Why are all the cross-entropy loss values close to chance (e.g. ln 10 ??? 2.3 for a 10 class data set) in the plots for convnets -bit not for ResNets -showing the linear relationship?

This is because most of the (correct) outputs of the normalized neural networks are close to zero as shown by Figure 20 .

We would expect the norm of the network to be appromimately bounded by |f (W ; x)| |W ||x| = |x|; the data x is usually pre-processed to have mean 0 and a standard deviation of 1.

In fact, for the MNIST experiments, the average value f (x) of the most likely class according to the normalized neural network is 0.026683 with a standard deviation 0.007144.

This means that significant differences, directly reflecting the predicted class of each point, are between 0.019539 and 0.033827.

This in turn implies that the exponentials in the cross-entropy loss are all very close to 1.

Before normalization (which of course does not affect the classification performance), the average value f (x) of the most likely class was 60.564373 with standard deviation 16.214078.

The model is a 3-layer convolutional ReLU network with the first two layers containing 24 filters of size 5 by 5; the final layer is fully connected; only the first layer has biases.

There is no pooling.

The network is overparametrized: it has 154, 464 parameters (compared to 50, 000 training examples).

The model is the same 3-layer convolutional ReLU network as in section L.1.1 except it had 34 units.

The network was still overparametrized: it has 165, 784 parameters (compared to 50, 000 training examples).

The model is a 5-layer convolutional ReLU network with (with no pooling).

It has in the five layers 32, 64, 64, 128 filters of size 3 by 3; the final layer is fully connected; batch-normalization is used during training.

The network is overparametrized with about 188, 810 parameters (compared to 50, 000 training examples).

<|TLDR|>

@highlight

Contrary to previous beliefs, the training performance of deep networks, when measured appropriately, is predictive of test performance, consistent with classical machine learning theory.