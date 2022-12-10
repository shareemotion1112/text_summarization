Artificial neural networks are built on the basic operation of linear combination and non-linear activation function.

Theoretically this structure can approximate any continuous function with three layer architecture.

But in practice learning  the parameters of such network can be hard.

Also the choice of activation function can greatly impact the performance of the network.

In this paper we are proposing to replace the basic linear combination operation with non-linear operations that do away with the need of additional non-linear activation function.

To this end we are proposing the use of elementary  morphological operations (dilation and erosion) as the basic operation in neurons.

We show that these networks (Denoted as Morph-Net) with morphological operations can approximate any smooth function requiring less number of parameters than what is necessary for normal neural networks.

The results show that our network perform favorably when compared with similar structured network.

We have carried out our experiments on  MNIST, Fashion-MNIST, CIFAR10 and CIFAR100.

In artificial neural networks, the basic building block is an artificial neuron or perceptron that simply computes the linear combination of the input BID22 .

It is usually followed by a non-linear activation function to model the non-linearity of the output.

Although the neurons are simple in nature, when connected together they can approximate any continuous function of the input BID4 .

This has been successfully utilized in solving different real world problems like image classification (Krizhevsky et al., 2012) , semantic segmentation BID13 and image generation BID6 .

While these models are quite powerful in nature, their efficient training can be hard in general BID12 and they need support of specials techniques, such as batch normalization BID5 and dropout BID24 , in order to achieve better generalization capabilities.

Their training time also depends on the choice of activation function BID15 .In this paper we are proposing new building blocks for building networks similar to neural network.

Here, instead of the linear combination operation of the artificial neurons, we use a non-linear operation that eliminates the need of additional activation function while requiring a small number of neurons to attain same performance or better.

More specifically, We use morphological operations (i.e. dilation and erosion) as the elementary operation of the neurons in the network.

Our contribution in this paper is building a network with these operations that has the following properties.1.

Networks built with with dilation-erosion neurons followed by linear combination can approximate any continuous function given enough dilation/erosion neurons.

2.

As dilation and erosion operation are non-linear by themselves, requirement of separate non-linear activation function is eliminated.

3.

The use of dilation-erosion operation greatly increases number of possible decision boundaries.

As a result, complex decision boundaries can be learned using small number of parameters.

The rest of the paper is organized as follows.

Section 2 describes the prior work on morphological neural network.

In Section 3, we introduce our proposed network and prove its capabilities theoretically.

We further demonstrate its capabilities empirically on a few benchmark datasets in Section 4.

Lastly Section 6 concludes the paper.

Morphological neuron was first introduced by BID1 in their effort to learn the structuring element of dilation operation in images.

Similar effort has been made to learn the structuring elements in a more recent work by BID14 .

Use of morphological neurons in a more general setting was first proposed by BID18 .

They restricted the network to a single layer architecture and focused only on binary classification task.

To classify the data, these networks use two axis parallel hyperplanes as the decision boundary.

This single layer architecture of BID18 has been extended to two layer architecture by BID25 .

This two layer architecture is able to learn multiple axis parallel hyperplanes, and therefore is able to solve arbitrary binary classification task.

But, in general the decision boundaries may not be axis parallel, as a result this two layer network may need to learn a large number of hyperplanes to achieve good results.

So, one natural extension is to incorporate the option to rotate the hyperplanes.

Taking a cue from this idea, BID0 proposed to learn a rotational matrix that rotates the input before trying to classify the data using axis parallel hyperplanes.

In a separate work by BID20 the use of L 1 and L ∞ norm has been proposed as a replacement of the max/min operation of dilation and erosion in order to smooth the decision boundaries.

BID19 first introduced the dendritic structure of biological neurons to the morphological neurons.

This new structure creates hyperbox based decision boundaries instead of hyperplanes.

The authors have proved that with hyperboxes any compact region can be estimated, therefore any two class classification problems can be solved.

A generalization of this structure to the multiclass case has also been done by BID21 .

BID26 had proposed a new type of structure called morphological perceptrons with competitive neurons, where the output is computed in winner-take-all strategy.

This is modelled using the argmax operator and this allows the network to learn more complex decision boundaries.

Later BID23 proposed a new training strategy to train this model with competitive neurons.

The non-differentiability of the max-min operations has forced the researchers to propose specialized training procedures for their models.

So, a separate line of research has attempted to modify these networks so that gradient descent based optimizer can be used for training.

Pessoa & Maragos (2000) have combined the classical perceptron with the morphological perceptron.

The output of each node is taken as the convex combination of the classical and the morphological perceptron.

Although max/min operation is not differentiable, they have proposed methodology to circumvent this problem.

They have shown that this network can perform complex classification tasks.

Morphological neurons have also been employed for regression task.

de A. Arajo (2012) has utilized network architecture similar to morphological perceptrons with competitive learning to forecast stock markets.

The argmax operator is replaced with a linear function so that the network is able to regress forecasts.

The use of linear activation function enables the use of gradient descent for training which is not possible with the argmax operator.

For morphological neurons with dendritic structure BID31 had proposed to replace the argmax operator with a softmax function.

This overcomes the problem of gradient computation and therefore gradient descent is employed to train the network.

So, this retains the hyperbox based boundaries of the dendritic networks, but facilitates easy training with gradient descent.

In this section we introduce the basic components and structure of our network and establish its approximation power.

Dilation and Erosion are two basic operations of our proposed network.

Given an input x ∈ R d and some structuring element s ∈ R d+1 , dilation (⊕) and erosion ( ) neurons computes the following two functions respectively The 0 is appended to the input x to take care of the 'bias'.

Here we try to learn the structuring element (s).

Note that erosion operation can also be written in the following form.

DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 3.2 GRADIENT OF DILATION AND EROSION Artificial neural networks are trained using back-propagation.

To be able to use the dilation/erosion neurons as a drop-in replacement of the artificial neurons, we must be able to compute the gradient of this operation.

Here we show the gradient of the dilation operation.

For erosion the gradient computation will be similar.

FIG0 shows the computational graph model of dilation operation and its gradient flow.

Lets assume L is the loss, we are trying to optimize.

We need to compute

to be able to update the structuring element.

This can be computed using the chain rule as follows.

DISPLAYFORM0 Here ∂L ∂z + will be the input gradient to this node which will be available from the node following this node.

We need to compute ∂z + ∂s .

Now as s ∈ R d+1 we can write the following.

DISPLAYFORM1 Now for each ∂z + ∂si the gradient is computed as follows, ∂z DISPLAYFORM2

The Dense Morphological Net or 'DenMo-Net', in short, that we propose here is a simple feed forward network with some dilation and erosion neurons followed by classical artificial neurons ( Figure 2 ).

We call the layer of dilation and erosion neurons as the dilation-erosion layer and the following layer as the linear combination layer.

Let's assume the dilation-erosion layer contains n dilation neurons and m erosion neurons, followed by c neurons in the linear combination layer.

Let x ∈ R d is the input to the network.

Let z + i and z − j be the output of i th dilation neuron and j th erosion node, respectively.

Then we can write, DISPLAYFORM0 DISPLAYFORM1 where, s + i and s − j are the structuring elements of the i th dilation neuron and j th erosion neuron respectively.

Note that i ∈ {1, 2, . . .

, n} and j ∈ {1, 2, . . .

, m}. The final output from a node of the linear combination layer is computed in the following way.

DISPLAYFORM2 where ω + i and ω − j are the weights of the artificial neuron in the linear combination layer.

In following subsection we show that g(x) can approximate any continuous function f : DISPLAYFORM3 Figure 2: Single Layer DenMo-Net with n dilation and m erosion neuron and c output neurons

Here we show that with the linear combination of dilation and erosion, any function can be approximated, and the approximation error decreases with increase in the number of neurons in the dilation-erosion layer.

Before that we need to describe some concepts.

Definition 1 (k-order Hinge Function BID29 ) A k-order hinge function consists of (k + 1) hyperplanes continuously joined together.

it is defined by the following equation, DISPLAYFORM0 Definition 2 (d-order hinging hyperplanes (d-HH) BID29 ) A d-order hinging hyperplanes (d-HH) is defined as the sum of multi-order hinge function as follows, DISPLAYFORM1 DISPLAYFORM2 From BID29 the following can be said about hinging hyperplanes.

Proposition 1 For any given positive integer d and arbitrary continuous piece-wise linear function DISPLAYFORM3 This says that any continuous piece-wise linear function of d variables can be written as an d-HH, i.e. the sum of multi-order hinge functions.

Now to show that our network can approximate any continuous functions, we show the following.

DISPLAYFORM4 is sum of multi-order hinge functions.

The proof of this lemma is given in Appendix A. Basically we show that g(x) can written as the sum of l hinge functions in the following form.

DISPLAYFORM5 where l = m + n (number of neurons in the dilation-erosion layer), α i ∈ {1, −1} and φ i (x)'s are d-order hinge function.

Proposition 2 (Stone-Weierstrass approximation theorem) Let C be a compact domain (C ⊂ R d ) and f : C → R a continuous function.

Then there exists a continuous piece wise linear function g such that for all x ∈ C, |f (x) − g(x)| < for some > 0.Theorem 1 (Universal approximation) Only a single dilation-erosion layer followed by a linear combination layer can approximate any continuous smooth function provided there are enough nodes in dilation erosion-layer.

From lemma 1 we know that our DenMo-Net with of n dilation and m erosion neurons followed by a linear combination layer computes g(x), which is a sum of multi-order hinge functions.

Now from proposition 1 we get that any continuous piecewise linear function can be written by a finite sum of multi-order hinge function.

Now from Proposition 2 we can say that any continuous function can be well approximated by a piecewise linear function.

In general if l → ∞ then → 0.

If we increase the number of neurons in the dilation-erosion layer the approximation error decreases.

Therefore, we can say that a DenMo-Net with enough dilation and erosion neurons can approximate any continuous function.

The DenMo-Net we have defined above learns the following function, DISPLAYFORM0 Where each φ(x) is collection of multiple hyperplanes joined together.

Therefore the number of hyperplanes learned by the network with l neurons in the dilation-erosion layer is much more than l. Each morphological neuron allows only one of the inputs to pass through because of max / min operation after addition with the structuring element.

So, effectively each neuron in the dilation-erosion layer chooses one component of the d-dimensional input vector.

Depending on which component is being chosen, the final linear combination layer computes the hyperplane by taking either all the components of the input or only some of them (when a subset of input components is chosen more than once in the dilation-erosion layer).

Note that this choice depends on the input and the structuring element together.

For a network with d dimensional input data and l neurons (l ≥ d) in the dilation-erosion layer, theoretically (d + 1) l − 1 hyperplanes can be formed in d dimension.

Out of the all possible planes only DISPLAYFORM1 l−d planes can span anywhere in the d dimensional space.

Therefore, increasing the number of neurons in the dilation-erosion layer exponentially increases the possible number of hyperplanes, i.e., the decision boundaries.

This implies that, using only a small number of neurons, complex decision boundaries can be learned.

Here we empirically validate the power of our DenMo-Net and demonstrate its advantages in comparison with other networks like artificial neural networks with different activation functions i.e. tanh (NN-tanh) and ReLU (NN-ReLU) and Maxout network BID3 .

As our network is defined with all possible connections between two consecutive layers, we have compared with only similar structured networks.

We have chosen the maxout network for comparison, because it uses the max function as a replacement of the activation function but with added nodes to compute the maximum.

The experiments have been carried out on a toy dataset with two concentric circles for visualizing the decision boundaries and also on benchmark datasets like MNIST BID11 , Fashion-MNIST BID30 , CIFAR-10 and CIFAR-100 BID8 ).

For all the tasks we have used categorical cross entropy as the loss and in the last layer softmax function is used.

In the training phase, all the networks have been optimized using Adam (learning rate= 0.001, β 1 = 0.9, β 2 = 0.999) optimizer BID7 with mini batches of size 32.

In all the experiments, we have used same number of dilation and erosion neurons in dilation-erosion layer unless otherwise stated.

For visualizing the decision boundaries learned by the classifiers, we have generated data on two concentric circles belonging to two different classes with center at the origin.

We compare the results when only two neurons are taken in the hidden layer in all the networks.

It is observed that classical neural network fails to classify this data with two hidden neurons as it learns one hyperplane per one hidden neuron.

The boundaries learned by the network with ReLU activation function (NNReLU) is shown in FIG1 .

The result of maxout network is better (87.17% training accuracy) as it introduces extra parameters with max function to achieve non-linearity.

In the maxout layer we have taken maximum among h = 2 features.

As we see in the figure 3b the network learns (2 * h =) 4 straight lines when trying to classify these data.

For the same data and two neurons in dilation-erosion layer, our DenMo-Net has learned 6 lines to form the decision boundary (figure 3c).Although from equation 14 we can say that we get at most 8 lines, only two of them can be placed anywhere in the 2D space while others are parallel to the axes.

For this reason, we are getting two slanted lines and the remaining lines are parallel to the axes.

The classification accuracy achieved by the networks along with their number of parameters is reported in table 1.

The difference in the accuracy clearly shows the power of DenMo-Net.

MNIST dataset (LeCun et al., 1998) contains gray scale images of hand written numbers (0-9) of size 28 × 28.

It has 60,000 training images and 10,000 test images.

Since our network does not support two dimensional input, we have converted each image to a column vector (in row major order) before giving it as input.

The network we use follows the structure we have previously defined: input layer, dilation-erosion layer and linear combination layer computing the output.

As in this dataset we had to distinguish between 10 classes of images, 10 neurons are taken in the output layer.

In TAB1 we have shown the accuracy on test data after training the network for 150 epochs with different number of nodes (l) in the dilation-erosion layer.

The change of test accuracy over the epochs is shown in FIG2 .

It is seen that increasing number of nodes in the dilation-erosion layer helps to increase non-linearity, and thus it results in better accuracy on test data.

We get test average accuracy of 98.43% after training 3 times with the DenMo-Net of 200 dilation and 200 erosion neurons TAB2 up to 400 epochs.

We have also experimented when only dilation, only erosion and both type of neurons were used in the dilation-erosion layer FIG2 ).

We see that using both erosion only and both dilation and erosion neurons giving better accuracy.

The Fashion-MNIST dataset BID30 has been proposed with the aim of replacing the popular MNIST dataset.

Similar to the MNIST dataset this also contains 28 × 28 images of 10 classes and 60,000 training and 10,000 testing samples.

While MNIST is still a popular choice for benchmarking classifiers, the authors' claim that MNIST is too easy and does not represent the modern computer vision tasks.

This dataset aims to provide the accessibility of the MNIST dataset while posing a more challenging classification task.

For the experiment, we have converted the images to a column vector similar to what we have done for the MNIST dataset.

We have taken 400 dilation and 400 erosion nodes in the dilation-erosion layer for this experiment.

We have trained the network separately 3 times up to 300 epochs.

The reported test accuracy TAB2 is the average of the 3 runs.

We see that our method gives better results.

CIFAR-10 ( BID8 ) is natural image dataset with 10 classes.

It has 50,000 training and 10,000 test images.

Each of them is a color image of size 32 × 32.

The images are converted to column vector before they are fed to the DenMo-Net.

For all the networks we compare with, the experiments have been conducted with keeping the number of neurons same in the hidden layer.

For maxout network each hidden neuron have two extra nodes over which the maximum is computed.

In table 4 we have reported the average test accuracy obtained over three run of 150 epochs.

The change of accuracy over epochs is also shown in figure 5a when number of hidden neurons is 600.

As it can be seen from both the table and the figure that DenMo-Net achieves the best accuracy in all the cases.

Maxout network lags behind even with more number of parameters.

This BID30 happens because our network is able to learn more hyperplanes with number of parameters similar to normal artificial neural networks.

When using only a single type of neurons in our network, we see a different result for this dataset FIG3 ).

The network takes time to learn with only erosion neurons.

The situation improves a little when using only dilation neurons.

When using both type of morphological neurons, the network is able to perform better by leveraging the power of both the operations.

CIFAR-100 BID8 ) is a image dataset similar to CIFAR-10 but with 100 classes with 600 images in each.

There are 500 training and 100 testing images for each class.

The training has been done similar to what is done for CIFAR-10.

Network has been trained with batch size 100.

We have reported the average test accuracy of 3 run with 100 epochs each in table 5.

The change of test accuracy over the epochs is plotted in FIG4 .

The results show trend similar to what is observed in other dataset.

DenMo-Net is giving better result with comparable number of trainable parameters and trains much faster.

When the type of neurons is restricted in our network, the results are similar to what we have seen for CIFAR-10 dataset FIG4 ).

Using only erosion neurons, the networks lags behind in terms of accuracy.

But interestingly, using erosion only the network is able to perform better than using both type of morphological neurons.

In our network we are learning the structuring element and the weights of the linear combination layer using gradient descent method.

While learning the structuring elements we may encounter two kinds of problem.

Dilation/erosion operation involves max/min operation.

As shown in Section 3.2, the gradient of the loss with respect to the structuring element contains a single non-zero element.

Therefore, only a single element of the structuring element is updated.

So, the learning can be slow.

On the other hand, some values of the structuring element may not get updated at all.

We have defined the network and have shown its properties when only three layers are employed in the network.

Although it may seem stacking multiple of these layers would translate to better results, the results are showing the opposite.

Straight-forward stacking of the layers we have used in our network can give rise to two kinds of network.

Type-I Multiple dilation-erosion layer, followed by a single linear combination layer at the end.

Type-II Dilation-Erosion layer followed by a linear combination layer.

Then another dilationerosion layer followed by one more linear combination layer and so on.

For the network of Type-I, it can be argued that the network is performing some combination of opening and closing operation, and their linear combination.

As there are dilation-erosion (DE) layers back to back, the problem of gradient propagation is amplified.

As a result it takes much more time to train than single layer architecture TAB5 ).Similar explanation doesn't work for Type-II networks.

From FIG5 we see that the network has tendency to overfit.

We believe its understanding requires further exploration.

In this paper we have proposed a new class of networks that uses both normal and morphological neurons.

These network consists of three layers only: input layer, dilation-erosion layer with dilation and erosion neurons followed by linear combination layer giving the output of the network with normal artificial neurons.

We have done our analysis using this three layer network only, but its deeper version can also be explored.

We have shown that this three layer architecture can approximate any sufficiently smooth function without requiring any non-linear activation function.

These networks are able to learn a large number of hyperplanes with very few neurons in the dilation-erosion layer thereby providing superior results compared to other networks with three layer architecture.

The improved results could also be the result of 'feature selection' by the max/min operator in the dilation erosion layer.

In this work we have only worked with fully connected layers, i.e. a node in a layer is connected to all the nodes in the previous layer.

This type of connectivity is not very efficient for image data where architectures with convolution layers perform better.

So, extending this work to the case where a structuring element operates by sliding over the whole image, should be the next logical step.

APPENDIX A PROOF OF LEMMA 1 From equation 9 we have DISPLAYFORM0 Now this equation can be rewritten as follows DISPLAYFORM1 where s + ik and s − ik denote the k th component of the i th structuring element of dilation and erosion neurons, respectively.

The above equation can be further expressed in the following form, DISPLAYFORM2 Where DISPLAYFORM3 are define in the following way DISPLAYFORM4 Now, without any loss of generality we can write equation 17 as follows DISPLAYFORM5 where DISPLAYFORM6 Finally, we can rewrite equation 18 as DISPLAYFORM7 where l = m + n, α i ∈ {1, −1} and φ i (x)'s are of the following form DISPLAYFORM8 DISPLAYFORM9 In equation 20, v DISPLAYFORM10 represents sum of multi-oder hinge function.

However, it may be noted that taking l ≥ d results hinge hyper planes which can span any where in d dimensional input space.

We can assume there are l 1 and l 2 number of terms where α = 1 and α = −1 respectively, then DISPLAYFORM11 where l 1 + l 2 = l and φ i (x), φ i (x) is of same form as equation 20.

Threfore can write, DISPLAYFORM12 DISPLAYFORM13 where k i ∈ {1, 2, .., d + 1}∀i.

In equation 24 we are taking maximum of (d + 1) l1 terms.

Similarly we can derive same expression for Proof From equation 19, without any loss of generality we can assume there are t 1 and t 2 number of terms where α = 1 and α = −1 respectively, then DISPLAYFORM14 where t 1 + t 2 = l and φ i (x), φ i (x) are of same form as equation 20.As sum of PWL functions is also a PWL function, hence each t1 i=1 φ i (x) and t2 i=1 φ i (x) and PWL.

Now, if t 1 > 0, from Proposition 3 we can conclude that g(x) is PWL linear function since difference of two continuous PWL function is PWL function .

If t 1 = 0 then g(x) becomes PWL concave function.

Hence, can say g(x) is PWL function.

It may be noted that if l < d then PWL hyperplane will be in parallel to at least one of the axis.

Taking l ≥ d results PWL hyperplane which may span anywhere in d dimensional space.

Theorem 2 (Universal approximation) Using only a single dilation-erosion layer followed by a linear combination layer any continuous function can be approximated.

From proposition 2 we get that any continuous function can be well approximated by a PWL function with an error bound of .

Now from lemma 2 we know that our DenMo-Net with of n dilation and m erosion neurons followed by a linear combination layer computes a PWL function.

Hence we can say our network can approximate any continuous function.

In general if we increase the neurons in the dilation-erosion layer, number of affine function in g(x) (equation 18) increases and the error bound → 0 as the number of nodes in dilation-erosion layer increases.

<|TLDR|>

@highlight

Using mophological operation (dilation and erosion) we have defined a class of network which can approximate any continious function. 

@highlight

This paper proposes to replace standard RELU/tanh units with a combination of dilation and erosion operations, observing that the new operator creates more hyper-planes and has more expressive power.

@highlight

The authors introduce Morph-Net, a single layer neural network where the mapping is performed using morphological dilation and erosion.