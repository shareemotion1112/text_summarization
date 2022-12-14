Deep neural networks have been tremendously successful in a number of tasks.

One of the main reasons for this is their capability to automatically learn representations of data in levels of abstraction, increasingly disentangling the data as the internal transformations are applied.

In this paper we propose a novel regularization method that penalize covariance between dimensions of the hidden layers in a network, something that benefits the disentanglement.

This makes the network learn nonlinear representations that are linearly uncorrelated, yet allows the model to obtain good results on a number of tasks, as demonstrated by our experimental evaluation.

The proposed technique can be used to find the dimensionality of the underlying data, because it effectively disables dimensions that aren't needed.

Our approach is simple and computationally cheap, as it can be applied as a regularizer to any gradient-based learning model.

A good data representation should ultimately uncover underlying factors in the raw data while being useful for a model to solve some task.

Deep neural networks learn representations that are increasingly abstract in deeper layers, disentangling the causes of variation in the underlying data BID1 .

Formal definitions of disentanglement are lacking, although Ver BID17 ; BID0 both use the total correlation as a measure of disentanglement.

Inspired by this, we consider a simpler objective: a representation disentangles the data well when its components do not correlate, and we explore the effects of penalizing this linear dependence between different dimensions in the representation.

Ensuring independence in the representation space results in a distribution that is factorizable and thus easy to model BID8 BID14 .We propose a novel regularization scheme that penalizes the cross-correlation between the dimensions of the learned representations, and helps artificial neural networks learn disentangled representations.

The approach is very versatile and can be applied to any gradient-based machine learning model that learns its own distributed vector representations.

A large body of literature have been published about techniques for learning non-linear independent representations BID12 BID5 BID3 , but in comparison our approach is simpler, and does not impose restrictions on the model used.

The proposed technique penalizes representations with correlated activations.

It strongly encourages the model to find the dimensionality of the data, and thus to disable superfluous dimensions in the resulting representations.

The experimental evaluation on synthetic data verifies this: the model is able to learn all useful dimensions in the data, and after convergence, these are the only ones that are active.

This can be of great utility when pruning a network, or to decide when a network needs a larger capacity.

The disabling of activations in the internal representation can be viewed as (and used for) dimensionality reduction.

The proposed approach allows for interpretability of the activations computed in the model, such as isolating specific underlying factors.

The solution is computationally cheap, and can be applied without modification to many gradient-based machine learning models that learns distributed representations.

Moreover, we present an extensive experimental evaluation on a range of tasks on different data modalities, which shows that the proposed approach disentangles the data well; we do get uncorrelated components in the resulting internal representations, while retaining the performance of the models on their respective task.

Figure 1: When data is distributed along non-linear manifolds, a linear model cannot describe the data well (left).

However, with a non-linear model (right), it is possible to capture the variations of the data in a more reasonable way and unfold it into a compact orthogonal representation space.

The main contributions of this work include: L ?? regularization, a novel approach penalizing the covariance between dimensions in a representation (see Section 2).

The regularizer encourages a model to use the minimal number of dimensions needed in the representation.

The approach is computationally cheap and can be applied without any restrictions on the model.

The experimental evaluation shows how different models can benefit from using L ?? regularization.

From autoencoders on synthetic data to deep convolutional autoencoders trained on CIFAR-10, we show that L ?? helps us learn uncorrelated and disentangled representations (see Section 3).

We present a novel regularizer based on the covariance of the activations in a neural network layer over a batch of examples.

The aim of the regularizer is to penalize the covariance between dimensions in the layer to decrease linear correlation.

The covariance regularization term (L ?? ) for a layer, henceforth referred to as the coding layer, is computed as DISPLAYFORM0 where p is the dimensionality of the coding layer, DISPLAYFORM1 is the element wise L1 matrix norm of C, and C ??? R p??p is the sample covariance of the activations in the coding layer over N examples DISPLAYFORM2 Further, H = [h 1 ; ...; h N ] is a matrix of all activations in the batch, 1 N is an N -dimensional column vector of ones, andh is the mean activation.

As L ?? has the structure of a regularizer, it can be applied to most gradient based models without changing the underlying architecture.

In particular, L ?? is simply computed based on select layers and added to the error function, e.g. Loss = Error + ??L ??

This section describes the experimental evaluation performed using L ?? regularization on different models in various settings, from simple multi-layer perceptron-based models using synthetic data (see Section 3.2 and 3.3) to convolutional autoencoders on real data (see Section 3.4).

However, before describing the experiments in detail we define the metrics that will be used to quantify the results.

A number of different metrics are employed in the experiments to measure different aspects of the results.

Pearson correlation report the normalized linear correlation between variables ??? [???1, 1] where 0 indicates no correlation.

To get the total linear correlation between all dimensions in the coding layer the absolute value of each contribution is averaged.

DISPLAYFORM0 Covariance/Variance Ratio (CVR) Though mean absolute Pearson correlation measure the quantity we are interested in it becomes ill defined when the variance of one (or both) of the variables approaches zero.

To avoid this problem we define a related measure where all variances are summed for each term.

Hence, as long as some dimension has activity the measure remains well defined.

More precise, the CVR score is computed as: DISPLAYFORM1 where ||C|| 1 is defined as in Equation 2.

The intuition behind CVR is simply to measure the fraction of all information that is captured in a linear uncorrelated fashion within the coding layer.

Utilized Dimensions (UD) UD is the number of dimensions that needs to be kept to retain a set percentage, e.g. 90% in the case of UD 90% , of the total variance.

This measure has the advantage that the dimension of the underlying data does not need to be known a priori.

The purpose of this experiment is to investigate if it is possible to disentangle independent data that has been projected to a higher dimension using a random projection, i.e. we would like to find the principal components of the original data.

The model we employ in this experiment is an auto encoder consisting of a linear p = 10 dimensional coding layer and a linear outputlayer.

The model is trained using the proposed covariance regularization L ?? on the coding layer.

The data is generated by sampling a d = 4 dimensional vector of independent features z ??? N (0, ??), where ?? ??? R d??d is constrained to be non-degenerate and diagonal.

However, before the data is fed to the autoencoder it is pushed through a random linear transformation x = ???z.

The goal of the model is to reconstruct properties of z in the coding layer while only having access to x.

The model is trained on 10000 iid random samples for 10000 epochs.

9 experiments were performed with different values for the regularization constant ??.

The first point on each curve (in Figure 2 and 3) is ?? = 0, i.e. no regularization, followed by 8 points logarithmically spaced between 0.001 and Figure 2 : In this figure we compare the amount of residual linear correlation after training the model with L ?? and L 1 regularization respectively, measured in MAPC (left) and CVR (right).

The first point on each curve corresponds to ?? = 0, i.e. no regularization, followed by 8 points logarithmically spaced between 0.001 and 1.

All scores are averaged over 10 experiments using a different random projection (???).Figure 3: The resulting dimensionality the coding layer after training the model with L ?? and L 1 regularization respectively, measured in TdV (left) and UD 90% (right).

The first point on each curve corresponds to ?? = 0, i.e. no regularization, followed by 8 points logarithmically spaced between 0.001 and 1.All scores are averaged over 10 experiments using a different random projection (???).1.

Each experiment is repeated 10 times using a different random projection ??? and the average is reported.

The result of the experiment is reported using all four metrics defined in Section 3.1.

The result in terms of MAPC and CVR is reported in Figure 2 .

The first thing to notice is that L ?? consistently lead to lower correlation while incurring less MSE penalty compared to L 1 .

Further, looking at the MAPC it is interesting to notice that it is optimal for a very small values of L ?? .

This is because higher amounts of L ?? leads to lowering of the dimensionality of the data, see Figure 3 , which in turn yields unpredictable Pearson correlation scores between these inactivated neurons.

However, this effect is compensated for in CVR for which L ?? quickly converges towards the optimal value of one, which in turn indicates no presence of linear correlation.

Turning the attention to dimensionality reduction, Figure 3 shows that L ?? consistently outperform L 1 .

Further, looking closer at the TdV score, L ?? is able to compress the data almost perfectly, i.e. TdV=1, at a very small MSE cost while L 1 struggle even when accepting a much higher MSE cost.

Further, the UD 90% scores again show that L ?? achieves a higher compression at lower MSE cost.

In this instance the underlying data was of 4 dimensions which L ?? quickly achieves.

At higher amounts of L ?? the dimensionality even locationally fall to 3, however, this is because the threshold is set to 90%.

In Section 3.2 we showed that we can learn a minimal orthogonal representation of data that is generated to ensure that each dimension is independent.

However, in reality it is not always possible to encode the necessary information, to solve the problem at hand, in an uncorrelated coding layer, e.g. the data illustrated in Figure 1 would first need a non linear transform before the coding layer.

However, using a deep network it should be possible to learn such a nonlinear transformation that enables uncorrelated features in higher layers.

To test this in practice on a problem that has this property but still is small enough to easily understand we turn to the XOR problem.

It is well known that the XOR problem can be solved by a neural network of one hidden layer consisting of a minimum of two units.

However, instead of providing this minimal structure we would like the network to discover it by itself during training.

Hence, the model used is intentionally overspecified consisting of two hidden layers of four logistic units each followed by a one dimensional logistic output layer.

The model was trained on XOR examples, e.g. [1,0]=1, in a random order until convergence with L ?? applied to both hidden layers and added to the cost function after scaling it with ?? = 0.2.

Figure 4 the model was able to learn the optimal structure of exactly 2 dimensions in the first layer and one dimension in the second.

Further, as expected, the first layer do encode a negative covariance between the two active units while the second layer is completely free from covariance.

Note that, even though the second hidden layer is not the output of the model it does encode the result in that one active neuron.

For comparison, see Figure 5 for the same model trained without L ?? .

Figure 4: Covariance matrix (left) and spectrum (right) of the hidden layers of a feed forward neural network trained with L ?? regularization to solve the XOR problem.

Layer one (top) has learned to utilize unit zero and three while keeping the rest constant, and in layer two only unit two is utilized.

This learned structure is the minimal solution to the XOR problem.

Convolutional autoencoders have been used to learn features for visual input and for layer-wise pretraining for image classification tasks.

Here, we will see that it is possible to train a deep convolutional autoencoder on real-world data and learn representations that have low covariance, while retaining the reconstruction quality.

To keep it simple, the encoder part of the model used two convolutional layers and two fully connected layers, with a total of roughly 500.000 parameters in the whole model.

The regularization was applied to the coding layer which has 84 dimensions, giving a bottleneck effect.

The model was trained and evaluated on the CIFAR-10 dataset BID11 , containing 32x32 pixel colour images tagged with 10 different classes.

The model was trained on 45,000 images, while 5,000 were set aside for validation, and 10,000 make out the test set.

We compare the results from using L ?? regularization with L1 regularization and with no regularization at all.

The autoencoder was trained with a batch size of 100, using the Adam optimizer BID7 with an initial learning rate of 0.001.

Training was run until the MSE score on the validation set stopped improving 1 .

The regularization parameter ?? was chosen to be 0.08, for a reasonable trade-off between performance and covariance/variance ratio.

The reported scores in Table 1 and Figure 6 are averages from training the model five times with different initialization.

The results (see Table 1 ) show that the high-level features become more disentangled and has a lower CVR (6.56) using L ?? regularization.

Without regularization, the score is 20.00, and with L1 regularization the score is 4.03.

The model with L ?? regularization obtains a reconstruction error (MSE) of 0.0398, roughly the same as without regularization (0.0365), both of which are much better than using L1 regularization, with an MSE of 0.0569.

Figure 6 shows the CVR score plotted against the MSE, illustrating that the L ?? technique leads to more disentangled representations while retaining a better MSE score.

As you increase the regularization factor both L ?? regularization pushes down the CVR quickly, while retaining an MSE error that is almost constant.

L1 regularization also pushes the model towards learning representation with lower CVR, although slower, and while worsening the MSE error.

The UD 90% results show that L ?? encourages representations that concentrate the variation, and the model constantly learns representations with lower UD 90% score than using L1.

With ?? > 0.08, the MSE, the CV R, and the UD 90% all becomes much worse when using L1 regularization, while the L ?? seems to continue smoothly to improve CV R and UD 90% , as the MSE starts to grow.

Disentanglement is important in learned representations.

Different notions of independence have been proposed as useful criteria to learn disentangled representations, and a large body of work has been dedicated to methods that learn such representations.

Principal component analysis (PCA; BID13 ) is a technique that fits a transformation of the (possibly correlated) input into a space of lower dimensionality of linearly uncorrelated variables.

Nonlinear extensions of PCA include neural autoencoder models BID10 , using a network layout with three hidden layers and with a bottleneck in the middle coding layer, forcing the network to learn a lower-dimensional representation.

Self-organizing maps BID9 and kernelbased models BID15 have also been proposed for nonlinear PCA.Independent component analysis (ICA; BID6 ) is a set of techniques to learn additive components of the data with a somewhat stronger requirement of statistical independence.

A number of approaches have been made on non-linear independent components analysis, BID12 BID5 .

While ICA has a somewhat stronger criterion on the resulting representations, the approaches are generally more involved.

BID3 proposed a method to train a neural network to transform data into a space with independent components.

Using the substitution rule of differentiation as a motivation, they learn bijective transformations, letting them use the neural transformation both to compute the transformed hidden state, to sample from the distribution over the hidden variables, and get a sample in the original data space.

The authors used a fixed factorial distribution as prior distribution (i.e. a distribution with independent dimensions), encouraging the model to learn independent representations.

The model is demonstrated as a generative model for images, and for inpainting (sampling a part of the image, when the rest of it is given).

BID0 connected the properties of disentanglement and invariance in neural networks to information theoretic properties.

They argue that having invariance to nuisance factors in a network requires that its learned representations to carry minimal information.

They propose using the information bottleneck Lagrangian as a regularizer for the weights.

Our approach is more flexible and portable, as it can be applied as a regularization to learn uncorrelated components in any gradient-based model that learns internal representations.

BID2 showed that it is possible to adversarial training to make a generative network learn a factorized, independent distribution p(z).

The independence criterion (mutual information) makes use of the Kullback-Leibler divergence between the joint distribtion p(z) (represented by the generator network) and the product of the marginals (which is not explicitly modelled).

In this paper, the authors propose to resample from the joint distribution, each time picking only the value for one of the components z i , and let that be the sample from the marginal for that component, p(z i ).

A discriminator (the adversary) is simultaneously trained to distinguish the joint from the product of the marginals.

One loss function is applied to the output of the discriminator, and one measures the reconstruction error from a decoder reconstructing the input from the joint.

BID16 considers a reinforcement learning setting where there is an environment with which one can interact during training.

The authors trained one policy ?? i (a|s) for each dimension i of the representation, such that the policy can interact with the environment and learn how to modify the input in a way that modifies the representation only at dimension i, without changing any other dimensions.

The approach is interesting because it is a setting similar to humans learning by interaction, and this may be an important learning setting for agents in the future, but it is also limited to the setting where you do have the interactive environment, and cannot be applied to other settings discussed above, whereas our approach can.

In this paper, we have presented L ?? regularization, a novel regularization scheme based on penalizing the covariance between dimensions of the internal representation learned in a hierarchical model.

The proposed regularization scheme helps models learn linearly uncorrelated variables in a non-linear space.

While techniques for learning independent components follow criteria that are more strict, our solution is flexible and portable, and can be applied to any feature-learning model that is trained with gradient descent.

Our method has no penalty on the performance on tasks evaluated in the experiments, while it does disentangle the data.

@highlight

We propose a novel regularization method that penalize covariance between dimensions of the hidden layers in a network.

@highlight

This paper presents a regularization mechanism which penalizes covariance between all dimensions in the latent representation of a neural network in order to disentangle the latent representation