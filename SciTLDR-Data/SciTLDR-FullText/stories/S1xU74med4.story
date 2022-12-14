The ResNet and the batch-normalization (BN) achieved high performance even when only a few labeled data are available.

However, the reasons for its high performance are unclear.

To clear the reasons, we analyzed the effect of the skip-connection in ResNet and the BN on the data separation ability, which is an important ability for the classification problem.

Our results show that, in the multilayer perceptron with randomly initialized weights, the angle between two input vectors converges to zero in an exponential order of its depth, that the skip-connection makes this exponential decrease into a sub-exponential decrease, and that the BN relaxes this sub-exponential decrease into a reciprocal decrease.

Moreover, our analysis shows that the preservation of the angle at initialization encourages trained neural networks to separate points from different classes.

These imply that the skip-connection and the BN improve the data separation ability and achieve high performance even when only a few labeled data are available.

The architecture of a neural network heavily affects its performance especially when only a few labeled data are available.

The most famous example of one such architecture is the convolutional neural network (CNN) BID6 .

Even when convolutional layers of CNN were randomly initialized and kept fixed and only the last fully-connected layer was trained, it achieved a competitive performance compared with the traditional CNN BID5 BID14 .

Recent other examples are the ResNet BID3 and the batch-normalization (BN) BID4 .

The ResNet and the BN are widely used in few-shot learning problems and achieved high performance BID8 BID9 .One reason for the success of neural networks is that their architectures enable its feature vector to capture prior knowledge about the problem.

The convolutional layer of CNN enable its feature vector to capture statistical properties of data such as the shift invariance and the compositionality through local features, which present in images BID13 .

However, effects of the skip-connection in ResNet and the BN on its feature vector are still unclear.

To clear the effects of the skip-connection and the BN, we analyzed the transformations of input vectors by the multilayer perceptron, the ResNet, and the ResNet with BN.

Our results show that the skip-connection and the BN preserve the angle between input vectors.

This preservation of the angle is a desirable ability for the classification problem because the last output layer should separate points from different classes and input vectors in different classes have a large angle BID11 BID10 .

Moreover, our analysis shows that the preservation of the angle at initialization encourages trained neural networks to separate points from different classes.

These imply that the skip-connection and the BN improve the data separation ability and achieve high performance even when only a few labeled data are available.

We consider the following L layers neural networks, which transform an input vector x ??? R D into a new feature vector h L ??? R D through layers.

Let h 0 = x and ??(??) = max(0, ??) be the ReLU activation function.

ResNet BID12 BID1 : DISPLAYFORM0 ResNet with batch-normalization (BN): DISPLAYFORM1 where the expectation is taken under the distribution of input vectors in the mini-batch of the stochastic gradient descent (SGD).

Without loss of generality, we assume that the variance of input vectors in the mini-batch is one, Var ( DISPLAYFORM2 We analyzed the average behavior of these neural networks when the weights were randomly initialized as follows.

In the MLP, the weights were initialized by the He initialization BID2 because the activation function is the ReLU function.

DISPLAYFORM3 In the ResNet and the ResNet with BN, the first internal weights were initialized by the He initialization, but the second internal weights were initialized by the Xavier initialization BID0 because the second internal activation function is the identity.

DISPLAYFORM4

We analyzed the transformation of input vectors through hidden layers of the neural networks.

Now we define the quantity studied in this paper.

, we define the angle and the cosine similarity, DISPLAYFORM0 where DISPLAYFORM1 is the length of the feature vector and DISPLAYFORM2 is the inner product between the pair of the feature vectors.

Note that the expectation is taken under the probability distribution of initial weights.

We derived the recurrence relation of the angle TAB0 .

Its plot FIG0 shows that the MLP contracts the angle between input vectors, which is an undesirable property for the classification DISPLAYFORM0 DISPLAYFORM1 Figure 2: Transformation of the angle between input vectors.

We plotted the mean and the standard deviation of the angle over 10 randomly initialized parameters.problem, and that the skip-connection in ResNet and the BN relax this contraction.

Numerical simulations (Fig.2) on the MNIST dataset (LeCun et al., 1998) validated our analysis.

TAB0 gives us the clear interpretation how the skip-connection in ResNet and the BN preserve the angle between input vectors.

The ReLU activation function contracts the angle because the ReLU activation function truncates negative value of its input.

The skip-connection bypasses the ReLU activation function and thus reduces the effect of the ReLU activation function to the half.

Moreover, the BN reduces the effect of the ReLU activation function to the reciprocal of the depth.

We derived the dynamics of the angle through layers (Table 2) by applying the recurrence relation of the angle TAB0 iteratively and using the fact that, if ?? is small, arccos(??(??)) can be well approximated by the linear function, a ?? ?? where a < 1 is constant.

Table 2 shows that, in the MLP with randomly initialized weights, the angle between input vectors converges to zero in an exponential order of its depth, that the skip-connection in ResNet makes this exponential decrease into a sub-exponential decrease, and that the BN relaxes this sub-exponential decrease into a reciprocal decrease.

In other words, the skip-connection in ResNet and the BN preserve the angle between input vectors.

Numerical simulation (Fig.3) on the MNIST dataset validated our analysis.

Table 2 : Dynamics of the angle through layers.

DISPLAYFORM0 Figure 3: Dynamics of the angle.

We plotted the mean and the standard deviation of the angle over 10 randomly initialized parameters.

A desirable ability of the neural network for the classification problem is to separate points from different classes.

However, our results show that the randomly initialized neural networks contract the angle between input vectors from different classes.

Our analysis provide us with an insight how training tackle this problem.

We can show that the cosine similarity c l+1 (n, m) is proportional to DISPLAYFORM0 where ?? is a parameter we can control by training.

Its plot FIG1 implies that training makes small angles smaller and large angles larger by taking the extreme value of ?? like 0 or ??.

In order to validate this insight, we stacked the softmax layer on top of an 1 layer MLP and trained this model by the SGD with 100 labeled examples in the MNIST dataset.

Fig.5 shows the change of the angles of feature vectors by training, which validated our insight.

The above discussion also shows the relationship between training and the preservation of the angle.

The angle of feature vectors at high layer of the initialized MLP is small, which implies that training doesn't take extreme value of ?? and doesn't separate points from different classes.

On the other hand, the skip-connection and the BN preserve the angle between input vectors even at high layer.

Thus, training takes extreme value of ?? and separates points from different classes.

Numerical simulations (Fig.6) , which is the same as the previous one, validated our insight.

The ResNet and the BN achieved high performance even when only a few labeled data are available.

To clear the reasons for its high performance, we analyzed effects of the skip-connection in ResNet and the BN on the transformation of input vectors through layers.

Our results show that the skip-connection and the BN preserve the angle between input vectors, which is a desirable ability for the classification problem.

Moreover, our analysis shows that the preservation of the angle at initialization encourages trained neural networks to separate points from different classes.

These results imply that the skip-connection and the BN improve the data separation ability and achieve high performance even when only a few labeled data are available.

@highlight

The Skip-connection in ResNet and the batch-normalization improve the data separation ability and help to train a deep neural network.