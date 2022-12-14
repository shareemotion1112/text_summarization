In this paper, we first identify \textit{angle bias}, a simple but remarkable phenomenon that causes the vanishing gradient problem in a multilayer perceptron (MLP) with sigmoid activation functions.

We then propose \textit{linearly constrained weights (LCW)} to reduce the angle bias in a neural network, so as to train the network under the constraints that the sum of the elements of each weight vector is zero.

A reparameterization technique is presented to efficiently train a model with LCW by embedding the constraints on weight vectors into the structure of the network.

Interestingly, batch normalization (Ioffe & Szegedy, 2015) can be viewed as a mechanism to correct angle bias.

Preliminary experiments show that LCW helps train a 100-layered MLP more efficiently than does batch normalization.

Neural networks with a single hidden layer have been shown to be universal approximators BID6 BID8 .

However, an exponential number of neurons may be necessary to approximate complex functions.

A solution to this problem is to use more hidden layers.

The representation power of a network increases exponentially with the addition of layers BID17 BID2 .

A major obstacle in training deep nets, that is, neural networks with many hidden layers, is the vanishing gradient problem.

Various techniques have been proposed for training deep nets, such as layer-wise pretraining BID5 , rectified linear units BID13 BID9 , variance-preserving initialization BID3 , and normalization layers BID7 BID4 .In this paper, we first identify the angle bias that arises in the dot product of a nonzero vector and a random vector.

The mean of the dot product depends on the angle between the nonzero vector and the mean vector of the random vector.

We show that this simple phenomenon is a key cause of the vanishing gradient in a multilayer perceptron (MLP) with sigmoid activation functions.

We then propose the use of so-called linearly constrained weights (LCW) to reduce the angle bias in a neural network.

LCW is a weight vector subject to the constraint that the sum of its elements is zero.

A reparameterization technique is presented to embed the constraints on weight vectors into the structure of a neural network.

This enables us to train a neural network with LCW by using optimization solvers for unconstrained problems, such as stochastic gradient descent.

Preliminary experiments show that we can train a 100-layered MLP with sigmoid activation functions by reducing the angle bias in the network.

Interestingly, batch normalization BID7 can be viewed as a mechanism to correct angle bias in a neural network, although it was originally developed to overcome another problem, that is, the internal covariate shift problem.

Preliminary experiments suggest that LCW helps train deep MLPs more efficiently than does batch normalization.

In Section 2, we define angle bias and discuss its relation to the vanishing gradient problem.

In Section 3, we propose LCW as an approach to reduce angle bias in a neural network.

We also present a reparameterization technique to efficiently train a model with LCW and an initialization method for LCW.

In Section 4, we review related work; mainly, we examine existing normalization techniques from the viewpoint of reducing the angle bias.

In Section 5, we present empirical results that show that it is possible to efficiently train a 100-layered MLP by reducing the angle bias using LCW.

Finally, we conclude with a discussion of future works.

We introduce angle bias by using the simple example shown in FIG1 .

FIG1 (a) is a heat map representation of matrix W ??? R 100??100 , each of whose elements is independently drawn from a uniform random distribution in the range (???1, 1).

Matrix A ??? R 100??100 is also generated randomly, and its elements range from 0 to 1, as shown in FIG1 (b).

We multiply W and A to obtain the matrix shown in FIG1 (c).

Unexpectedly, a horizontal stripe pattern appears in the heat map of W A although both W and A are random matrices.

This pattern is attributed to the angle bias that is defined as follows: Definition 1.

P ?? is an m dimensional probability distribution whose expected value is ??1 m , where ?? ??? R and 1 m is an m dimensional vector whose elements are all one.

Proposition 1.

Let a be a random vector in R m that follows P ?? .

Given w ??? R m such that w > 0, the expected value of w ?? a is |??| ??? m w cos ?? w , where ?? w is the angle between w and 1 m .

where E(x) denotes the expected value of random variable x.

Definition 2.

From Proposition 1, the expected value of w ?? a depends on ?? w as long as ?? = 0.

The distribution of w ?? a is then biased depending on ?? w ; this is called angle bias.

In FIG1 , if we denote the i-th row vector of W and the j-th column vector of A by w i and a j , respectively, a j follows P ?? with ?? = 0.5.

The i-th row of W A is biased according to the angle between w i and 1 m , because the (i, j)-th element of W A is the dot product of w i and a j .

Note that if the random matrix A has both positive and negative elements, W A also shows a stripe pattern as long as each column vector of A follows P ?? with ?? = 0.We can generalize Proposition 1 for any m dimensional distributionP, instead of P ?? , as follows: DISPLAYFORM0 Let?? be a random vector that follows an m dimensional probability distributionP whose expected value is?? ??? R m .

Given w ??? R m such that w > 0, it follows that DISPLAYFORM1 where?? w is the angle between w and??.

Proof.

The proof is the same as that of Proposition 1.Proposition 2 states that the distribution of w ???? is biased according to?? w unless ?? = 0.

(1) and BID21 .

Repeating the operations through multiple layers, the variance of ?? l i and a l i will shrink to small values.

We illustrate the effect of angle bias in an MLP by using the CIFAR-10 dataset BID10 ) that includes a set of 32 ?? 32 color (RGB) images.

Each sample in CIFAR-10 is considered an input vector with 32 ?? 32 ?? 3 = 3072 real values, in which each variable is scaled into the range [???1, 1].

We consider an MLP with sigmoid activation functions that has 10 hidden layers with m = 128 neurons in each layer.

The weights of the MLP are initialized according to BID3 .

We randomly took 100 samples from the dataset and input them into the MLP.

FIG2 shows the activation pattern in layers BID19 BID21 5, 7 , and 9 on the selected samples.

Please note that the activation in Layer 1 corresponds to a 1 i in Equation (1) , that is, Layer 1 is the layer after the input layer.

We see stripe patterns in the layers other than Layers 1 in FIG2 that are caused by angle bias.

In Layer 9, the activation value of each neuron is almost constant regardless of the input.

In contrast, no stripe pattern appears in Layer 1, because each element of the input vector is scaled into the range [???1, 1] and its mean value is near zero; this corresponds to the case in which ?? ??? 0 in Proposition 2.

(3) , for each sample 1 .

FIG6 shows boxplot summaries of ?? l i on the first ten neurons in layers 1, 3, 5, 7, and 9, in which the 1%, 25%, 50%, 75%, and 99% quantiles are displayed as whiskers or boxes.

We see the mean of ?? l i are biased according to the neurons in the layers other than Layer 1.

We also see that the variance of ?? l i shrink through layers.

Next, we consider an MLP with ReLU activation functions that has 50 hidden layers with m = 128 neurons in each layer.

The weights are initialized according to BID3 .

Figure 4 shows the activation pattern in layers BID19 10, 20, 30 , and 40 on the randomly selected samples.

We see stripe patterns in the layers other than Layer 1 that are caused by the angle bias.

Figure 5 shows boxplot summaries of ?? l i on the first ten neurons in layers 1, 10, 20, 30, and 40 .

We see that the mean of ?? l i are biased according the neurons in the layers other than Layer 1.

We also see that the variance of ?? l i shrink through layers, but the shrinking rate is much moderate compared to that in FIG6 .

This is because ReLU projects a preactivation vector into the unbounded region [0, +???) m and the activation vector is less likely to concentrate on a specific region.

Under the effect of angle bias, the activation of neurons in deeper layers are almost constant regardless of the input in an MLP with sigmoid activation functions, as shown in FIG2 .

It indicates that ??? a 0 L = 0, where L is a loss function that is defined based on the output of the MLP and ??? a 0 L means the gradient with respect to the input vector a 0 .

From Equation (2) , we have DISPLAYFORM0 DISPLAYFORM1 Assuming that w Equation FORMULA4 , with l = 1, indicating that the gradients of weights in the first layer are vanished.

From Equation (1) , with l = 1, FORMULA3 and FORMULA4 , with l = 2, under the assumption that w DISPLAYFORM2 DISPLAYFORM3 If we use rectified linear activation instead of sigmoid activation, the gradients of weights are less likely to vanish, because ??? a 0 L will seldom be exactly zero.

However, the rate of each neuron being active 2 is biased, because the distribution of preactivation z l i is biased.

If a neuron is always active, it behaves as an identity mapping.

If a neuron is always inactive, it is worthless because its output is always zero.

Such a phenomenon is observed in deep layers in Figure 4 .

As discussed in BID1 , the efficiency of the network decreases in this case.

In this sense, angle bias may reduce the efficiency of a network with rectified linear activation.

There are two approaches to reduce angle bias in a neural network.

The first one is to somehow make the expected value of the activation of each neuron near zero, because angle bias does not occur if ?? = 0 from Proposition 2.

The second one is to somehow regularize the angle between w .

In this section, we propose a method to reduce angle bias in a neural network by using the latter approach.

We introduce W LC as follows: DISPLAYFORM0 The following holds for w ??? W LC :Proposition 3.

Let a be an m dimensional random variable that follows P ?? .

Given w ??? W LC such that w > 0, the expected value of w ?? a is zero. . .

, m) will likely be more similar to each other.

The activation vector in layer l, each of whose elements is given by Equation (1) , is then expected to follow P ?? .

Therefore, if the input vector a 0 follows P ?? , we can inductively reduce the angle bias in each layer of an MLP by using weight vectors that are included in W LC .

We call weight vector w DISPLAYFORM1

We built an MLP with sigmoid activation functions of the same size as that used in Section 2.2.1, but whose weight vectors are replaced with LCWs.

We applied the minibatch-based initialization described in Section 3.3.

FIG8 shows the activation pattern in layers 1, 3, 5, 7, and 9 of the MLP with LCW on the randomly selected samples that are used in FIG2 .

When compared with FIG2 , we see no stripe pattern in FIG8 .

The neurons in Layer 9 respond differently to each input sample; this means that a change in the input leads to a different output.

Therefore, the network output changes if we adjust the weight vectors in Layer 1, that is, the gradients of weights in Layer 1 do not vanish in FIG8 .

FIG9 shows boxplot summaries of ?? l i on the first ten neurons in layers 1, 3, 5, 7, and 9 of the MLP with LCW.

We see that the angle distributes around 90??? on each neuron in each layer.

This indicates that the angle bias is resolved in the calculation of z l i by using LCW.

FIG10 shows the activation pattern in layers of the MLP with LCW after 10 epochs training.

A slight stripe pattern is visible in FIG10 , but neurons in each layer react differently to each input.

FIG11 shows boxplot summaries of ?? l i of the MLP after 10 epochs training.

We see that the mean of ?? l i is slightly biased according to the neurons.

However, the variance of ?? l i do not shrink even in deeper layers.

We built an MLP with ReLU activation functions of the same size as that used in Section 2.2.2, whose weight vectors are replaced with LCWs.

We applied the minibatch-based initialization described in Section 3.3.

FIG1 shows the activation pattern in layers BID19 10, 20, 30 , and 40 of the MLP with LCW.

When compared with Figure 4 , we see no stripe pattern in FIG1 .

FIG1 shows boxplot summaries of ?? l i on the first ten neurons in layers BID19 10, 20, 30 , and 40 of the MLP with LCW.

We can observe that the angle bias is resolved by using LCW in the MLP with ReLU activation functions.

A straightforward way to train a neural network with LCW is to solve a constrained optimization problem, in which a loss function is minimized under the condition that each weight vector is included in W LC .

Although several methods are available to solve such constrained problems, for example, the gradient projection method BID11 , it might be less efficient to solve a constrained optimization problem than to solve an unconstrained one.

We propose a reparameterization technique that enables us to train a neural network with LCW by using a solver for unconstrained optimization.

We can embed the constraints on the weight vectors into the structure of the neural network by reparameterization.

DISPLAYFORM0 where I m???1 is the identity matrix of order (m ??? 1) ?? (m ??? 1).

In the experiments in Section 5, we used B m in Equation FORMULA9 .

We also tried an orthonormal basis of W LC as B m ; however, there was little difference in accuracy.

It is worth noting that the proposed reparameterization can be implemented easily and efficiently by using modern frameworks for deep learning using GPUs.

By introducing LCW, we can reduce the angle bias in z l i in Equation (2) , which mainly affects the expected value of z l i .

It is also important to regularize the variance of z l i , especially when the sigmoid activation is used, because the output of the activation will likely saturate when the variance of z l i is too large.

We apply an initialization method by which the variance of z l i is regularized based on a minibatch of samples.

This type of initialization has also been used in previous studies BID12 and BID16 .

We conducted preliminary experiments using the CIFAR-10 dataset, the CIFAR-100 dataset BID10 , and the SVHN dataset BID14 .

These experiments are aimed not at achieving state-of-the-art results but at investigating whether we can train a deep model by reducing the angle bias and empirically evaluating the performance of LCW in comparison to that of BN and WN.Network structure We used MLPs with the cross-entropy loss function.

Each network has 32 ?? 32 ?? 3 = 3072 input neurons and 10 output neurons, and it is followed by a softmax layer.

We refer to an MLP that has L hidden layers and M neurons in each hidden layer as MLP(L, M ).

Either a sigmoid activation function or a rectified linear activation function was used.

MLP LCW denotes an MLP in which each weight vector is replaced by LCW.

MLP BN denotes an MLP in which the preactivation of each neuron is normalized by BN.

MLP WN denotes an MLP whose weight vectors are reparametrized by WN.Initialization Plain MLP and MLP BN were initialized using the method proposed in BID3 .

MLP LCW was initialized using the minibatch-based method described in Section 3.3 with ?? z = 0.5.

MLP WN was initialized according to BID16 .Optimization MLPs were trained using a stochastic gradient descent with a minibatch size of 128 for 100 epochs.

The learning rate starts from 0.1 and is multiplied by 0.95 after every two epochs.

The experiments were performed on a system running Ubuntu 16.04 LTS with NVIDIA R Tesla R K80 GPUs.

We implemented LCW using PyTorch version 0.1.12.

We implemented BN using the torch.nn.BatchNorm1d module in PyTorch.

We implemented WN by ourselves using PyTorch 3 .

We first consider MLPs with sigmoid activation functions.

FIG1 shows the convergence and computation time for training MLPs with CIFAR-10 dataset.

FIG1 (a) shows that the training accuracy of the plain MLP(100, 128) is 10% throughout the training, because the MLP output is insensible to the input because of the angle bias, as mentioned in Section 2.2 BID22 .

By contrast, MLP LCW or MLP BN is successfully trained, as shown in FIG1 (a), indicating that the angle bias is a crucial obstacle to training deep MLPs with sigmoid activation functions.

MLP LCW achieves a higher rate of increase in the training accuracy compared to MLP BN in FIG1 (a) , (d) , and (g).

As described in Section 4, WN itself cannot reduce the angle bias, but the bias is reduced immediately after the initialization of WN.

From FIG1 (a) and (d) , we see that deep MLPs with WN are not trainable.

These results suggest that starting with weight vectors that do not incur angle bias is not sufficient to train deep nets.

It is important to incorporate a mechanism that reduces the angle bias during training, such as LCW or BN.The computational overhead of training of MLP LCW (100, 128) is approximately 55% compared to plain MLP(100, 128), as shown in FIG1 (b); this is much lower than that of MLP BN (100, 128) .

The overhead of MLP WN is large compared to that of MLP BN , although it contradicts the claim of BID16 .

We think this is due to the implementation of these methods.

The BN module we used in the experiments consists of a specialized function developed by GPU vendors, whereas the WN module was developed by ourselves.

In this sense, the overhead of LCW may be improved by a more sophisticated implementation.

In terms of the test accuracy, MLP LCW has peaks around 20 epochs, as shown in FIG1 (c), (f), and (i).

We have no clear explanation for this finding, and further studies are needed to investigate the generalizability of neural networks.

Experimental results with SVHN and CIFAR-100 datasets are reported in Section B in the appendix.

We have experimented with MLPs with rectified linear activation functions.

In our experiments, we observed that the plain MLP with 20 layers and 256 neurons per layer was successfully trained.

However, the training of MLP LCW of the same size did not proceed at all, regardless of the dataset used in our experiment; in fact, the output values of the network exploded after a few minibatch updates.

We have investigated the weight gradients of the plain MLP and MLP LCW .

FIG1 shows boxplot summaries of the weight gradients in each layer of both models, in which the gradients are evaluated by using a minibatch of CIFAR-10 immediately after the initialization.

By comparing FIG1 (a) and FIG1 (b), we find an exponential increase in the distributions of the weight gradients of MLP LCW in contrast to the plain MLP.

Because the learning rate was the same for every layer in our experiments, this exponential increase of the gradients might hamper the learning of MLP LCW .

The gradients in a rectifier network are sums of path-weights over active paths BID1 .

The exponential increase of the gradients therefore implies an exponential increase of active paths.

As discussed in Section 2.3, we can prevent neurons from being always inactive by reducing the angle bias, which we think caused the exponential increase in active paths.

We need further studies to make MLP LCW with rectified linear activation functions trainable.

Possible directions are to apply layer-wise learning rates or to somehow regularize the distribution of the weight gradients in each layer of MLP LCW , which we leave as future work.

In this paper, we have first identified the angle bias that arises in the dot product of a nonzero vector and a random vector.

The mean of the dot product depends on the angle between the nonzero vector and the mean vector of the random vector.

In a neural network, the preactivation value of a neuron is biased depending on the angle between the weight vector of the neuron and the mean of the activation vector in the previous layer.

We have shown that such biases cause a vanishing gradient in a neural network with sigmoid activation functions.

To overcome this problem, we have proposed linearly constrained weights to reduce the angle bias in a neural network; these can be learned efficiently by the reparameterization technique.

Preliminary experiments suggest that reducing the angle bias is essential to train deep MLPs with sigmoid activation functions.

@highlight

We identify angle bias that causes the vanishing gradient problem in deep nets and propose an efficient method to reduce the bias.