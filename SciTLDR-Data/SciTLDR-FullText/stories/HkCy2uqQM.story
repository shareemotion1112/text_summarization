Complex-value neural networks are not a new concept, however, the use of real-values has often been favoured over complex-values due to difficulties in training and accuracy of results.

Existing literature ignores the number of parameters used.

We compared complex- and real-valued neural networks using five activation functions.

We found that when real and complex neural networks are compared using simple classification tasks, complex neural networks perform equal to or slightly worse than real-value neural networks.

However, when specialised architecture is used, complex-valued neural networks outperform real-valued neural networks.

Therefore, complex–valued neural networks should be used when the input data is also complex or it can be meaningfully to the complex plane,  or when the network architecture uses the structure defined by using complex numbers.

In recent years complex numbers in neural networks are increasingly frequently used.

ComplexValued neural networks have been sucessfully applied to a variety of tasks specifically in signal processing where the input data has a natural interpretation in the complex domain.

In most publications complex-valued neural networks are compared to real-valued architectures.

We need to ensure that these architectures are comparable in their ability to approximate functions.

A common metric for their capacity are the number of real-valued parameters.

The number of parameters of complex-valued neural networks are rarely studied aspects.

While complex numbers increase the computational complexity, their introduction also assumes a certain structure between weights and input.

Hence, it is not sufficient to increase the number of parameters.

Even more important than in real-valued networks is the choice of activation function for each layer.

We test 5 functions: identity or no activation function, rectifier linear unit, hyperbolic tangent, magnitude, squared magnitude.

This paper explores the performance of complex-valued multi-layer perceptrons (MLP) with varying depth and width in consideration of the number of parameters and choice of activation function on benchmark classification tasks.

In section 2 we will give an overview of the past and current developments in the applications of complex-valued neural networks.

We shortly present the multi-layer perceptron architecture in section 3 using complex numbers and review the building blocks of complex-valued network.

In section 4 we consider the multi-layer perceptron with respect to the number of real-valued parameters in both the complex and real case.

We construct complex MLPs with the same number of units in each layer.

We propose two methods to define comparable networks: A fixed number of real-valued neurons per layer or a fixed budget of real-valued parameters.

In the same section we also consider the structure that is assumed by introducing complex numbers into a neural network.

We present the activation function to be used in our experiments in section 5.

In section 6 we present our experiments and their settings.

Section 7 discuss the results of different multi-layer perceptrons on MNIST digit classification, CIFAR-10 image classification, CIFAR-100 image classification, Reuters topic classification and bAbI question answering.

We identify a general direction of why and how to use complex-valued neural networks.

The idea of artificial neural networks with complex-valued input, complex-valued weights and complex-valued output was proposed in the 1970s BID0 .

A complex-valued backpropogation algorithm to train complex multi-layer networks was proposed in the 1990s by several authors BID2 BID10 BID4 .

In the 2000s complex neural networks, like real-valued neural networks, have been successfully applied to a variety of tasks.

These tasks included the processing and analysis of complex-valued data or data with an intuitive mapping to complex numbers.

Particularly, signals in their wave form were used as input data to complex-valued neural networks BID7 ).Another natural application of complex numbers are complex convolutions BID3 , since they have an application in both image and signal processing.

While real convolutions are widely used in deep neural networks for image processing, complex convolution can replace realvalued convolutions BID14 BID5 BID12 BID6 .The properties of complex numbers and matrices introduce constraints into deep learning models.

Introduced by BID1 and developed further by Wisdom et al. (2016) recurrent networks, which constrain their weights to be unitary, reduce the impact of the vanishing or exploding gradient problem.

More recently complex numbers have been (re)discovered by a wider audience and used in approaches to other tasks like embedding learning BID15 BID13 , knowledge base completion BID16 or memory networks BID9 .Despite their success in signal processing tasks, complex-valued neural networks have been less popular than their real-valued counter-parts.

This may be due to training and reports of varying results in related tasks.

The training process and architecture design are less intuitive, which stems from difficulties in differentiability of activation functions in the complex plane (Zimmermann et al., 2011; BID8 BID11 ).An aspect that has received little attention is an appropriate comparison of real-and complex-valued neural networks.

Many publications ignore the number of parameters all together (?), consider only the number of parameters of the entire model (?) or do not distinguish in complex-or real-valued parameters (?).

While the latter is most confusing for the reader, all three problems lead to an inappropriate comparison of the overall performance.

There exists a significant body of work on exploring deep learning architectures for real-valued neural networks.

Deep complex-valued neural networks are still to be explored.

Previous work has also shown the significance of the activation, not only for the training and gradient computation, but also for the accuracy.

Therefore, the width, depth and the choice of activation function need to be considered together.

We aim to fill this gap by systematically exploring the performance of multi-layered architectures on simple classification taks.

Many fundamental building blocks of neural networks can be used in the complex domain by replacing real-valued parameters with complex parameters.

However, there are some differences in training complex-valued neural networks.

We introduce the building blocks and consider differences in structure and training.

While the convolution on the complex plane using complex-valued filters is natural, it has been investigated in related literature (see section 2).

In this work we focus on layers consisting of complex-valued neurons as building blocks and their use in multi-layer architecture.

We define a complex-valued neuron analogous to its real-valued counter-part.

In consequence we can use projection onto a complex weight matrix to realise complex-numbered embeddings.

The complex valued neuron can be defined as: DISPLAYFORM0 with the (real or complex) input x ∈ ¼ n , complex weight w ∈ ¼ n and complex bias b ∈ ¼. Arranging m neurons into a layer: DISPLAYFORM1 Similarly, we can define the projection onto a complex matrix if the input x is a projector (e.g. one-hot vector).The activation function φ in all of the above definitions can be a real function φ : ¼ → or complex function φ : ¼ →

¼, but the function always acts on a complex variable.

We will consider the choice of the non-linear activation function φ in more detail in section 5.The loss function J should be a real function J : ¼ → or J :→ .

Since there is no total ordering on the field of complex numbers, because the i 2 = −1, a complex-valued function may lead to added difficulties in training.

To be able to interpret the output of the last layer as probability one, can use an additional activation function.

Thus the activation of the output layer is sigmoid(φ(z)) resp.

so f tmax(φ(z)) with φ : ¼ → and is used as a real number in a classical loss function (e.g. cross entropy).Both activation and loss functions are not always complex-differentiable.

Hence, the training process in the complex domain differs.

Similar to a real function, a complex function f : DISPLAYFORM2 A complex-valued function of one or more complex variables that is entire and complex differentiable is called holomorphic.

While in the real-valued case the existence of a limit is sufficient for differentiability, the complex definition in equation 3 implies a stronger property.

We map ¼ to 2 to illustrate this point.

A complex function f (x + iy) = u(x, y) + iv(x, y) with real-differentiable functions u(x, y) and v(x, y) is complex-differentiable if they satisfy the Cauchy-Riemann equations: DISPLAYFORM3 We simply separate a complex number z ∈ ¼ into two real numbers z = x + iy.

For f to be holomorphic, the limit not only needs to exist for the two functions u(x, y) and v(x, y), but they must also satisfy the Cauchy-Riemann equations.

That also means that a function can be nonholomorphic (not complex-differentiable) in z, but still be analytic in its parts x, y. That is exactly if the two functions are real-differentiable, but do not satisfy the Cauchy-Riemann equations.

To be able to apply the chain rule, the basic principle of backpropagation, to non-holomorphic functions, we exploit the fact that many non-holymorphic functions, are still differentiable in their real and imaginary parts.

We consider the complex function f to be a function of z and its complex conjugatez.

Effectively, we choose a different basis for our partial derivatives.

DISPLAYFORM4 These derivatives are a consequence of the Wirtinger calculus (or CR-calculus).

With the new basis we are able allow the application of the chain rule to non-holomorphic functions for multiple complex variables z i : DISPLAYFORM5 4 Real and Complex ParametersIn this section we discuss complex-and real-valued parameters in consideration of the number of parameters per layer and the structure assumed by complex numbers.

Any complex number z = x + iy = r * e iϕ can be represented by two real numbers: the real part Re(z) = x and the imaginary part Im(z) = y or as length or magnitude |z| = x 2 + y 2 = r and a phase ϕ. Effectively the number of real parameters of each layer is doubled: DISPLAYFORM6 The number of (real-valued) parameters is a metric of capacity or the ability to approximate functions.

Too many and a neural network tends to overfit the data, too few and the neural network tends to underfit.

For a comparison of architectures the real-valued parameters per layer should be equal (or at least as close as possible) in the real architecture and its complex counter part.

This ensures that models have the same capacity and a comparison shows the performance difference due to the added structure, not due to varying capacity.

Consider the number of parameters in a fully-connected layer in the real case and in the complex case.

Let n be the input dimension and m the number of neurons.

DISPLAYFORM7 (7) For a multi-layer perceptron with k the number of hidden layers, and output dimension c the number of real-valued parameters without bias is given by: DISPLAYFORM8 At first glance designing comparable multi-layer neural network architectures, i.e. they have the same number of real-valued parameters in each layer, is trivial.

Halving the number of neurons in every layer will not achieve a comparison, because the number of neurons define the output dimensions of the layer and the following layer's input dimension.

We adressed this problem by choosing MLP architectures with an even number of hidden layers k and choose the number of neurons per layer to be alternating between m and m 2 .

Thus we receive the same number of real parameters in each layer compared to a real-valued network.

As an example, let us consider the dimensions of outputs and weights in k = 4.

For the real-valued case: DISPLAYFORM9 where m i is the number of (complex or real) neurons of the i-th layer.

The equivalent with m complex-valued neurons would be: DISPLAYFORM10 Another approach to the design of comparable architectures is to work with a parameter budget.

Given a fixed budget of real parameters p we can define real or complex multi-layer perceptron with an even number k of hidden layers such that the network's parameters are within that budget.

All k + 2 layers have the same number of real-valued neurons m or complex-valued neurons m ¼ .

DISPLAYFORM11 DISPLAYFORM12 Despite the straight forward use and representation in neural networks, complex numbers define an additional structure compared to real-valued networks.

This interaction of the two parts becomes particularly apparant if we consider operations on complex numbers to be composed of the real and imaginary part or magnitude and phase: DISPLAYFORM13 with complex numbers z 1 = a + ib, z 2 = c + id.

In an equivalent representation with Euler's constant e iϕ = cos(ϕ) + isin(ϕ) the real parts do not interact.z 1 z 2 = (r 1 e iϕ 1 )(r 2 e iϕ 2 ) = (r 1 r 2 e iϕ 1 +ϕ 2 ), DISPLAYFORM14 Complex parameters increase the computational complexity of a neural network, since more operations are required.

Instead of a single real-valued multiplication operation, up to four real multplication and two real additions are required.

Depending on the implementation and the representation, this can be significantly reduced.

Nevertheless, it is not sufficient to double the numbers of real parameters per layer to achieve the same effect as in complex-valued neural networks.

This is also illustrated expressing a complex number z = a + ib ∈ ¼ as 2 × 2 matrices M in the ring of M 2 ( ): DISPLAYFORM15 An augmented representation of an input x allows the represention of complex matrix multiplication with an weight matrix W as larger real matrix multiplication: DISPLAYFORM16 This added structure, however, also means that architecture design needs to be reconsidered.

A deep learning architecture which performs well with real-valued parameters, may not work for complexvalued parameters and vice versa.

In later sections we experimentally investigate what consequences this particular structure has for the overall performance of a model.

In any neural network, real or complex, an important decision is the choice of non-linearity.

With the same number of parameters in each layer, we are able to study the effects activation functions have on the overall performance.

The Liouville Theorem states that any bounded holomorphic function f : ¼to¼ (that is differentiable on the entire complex plane) must be constant.

Hence, we need to choose unbounded and/or non-holomorphic activation functions.

We chose the identity function to investigate the performance of complex models assuming a function which is linearly separable in the complex parameters by not introducing a non-linearity into the model.

The hyperbolic tangents is a well-studied function and defined for both complex and real numbers.

The rectifier linear unit is a well-studied function and illustrates the separate application of an activation function.

The magnitude and squared magnitude functions are chosen to map complex numbers to real numbers.• Identity (or no activation function): DISPLAYFORM0 • Hyperbolic tangent: DISPLAYFORM1 • Rectifier linear unit (relU): DISPLAYFORM2 • Magnitude squared: DISPLAYFORM3 • Magnitude (or complex absolute): DISPLAYFORM4 In the last layer we apply an activation function φ : ¼ → before using the softmax or sigmoid to use in a receive a real loss.

Note that the squared magnitude allows a more efficient implementation than the magnitude.

Thus we change the activation function of the last layer to: DISPLAYFORM5 Applying the two functions in the opposite order as in |sigmoid(z)| 2 and |so f tmax(z)| 2 does not return probabilities from the last layer of a network and would take away the direct interpretability of the models output.

To compare real and complex-valued neural networks and their architecture we chose various classification tasks and defined experiments.

The settings are as follows:• Experiment 1: We tested multi-layer perceptrons (MLP) with with k = 0, 2, 4, 8 hidden layers, fixed width of units in each layer in real-valued architectures and alternating 64, 32 units in complex-valued architectures (see section 4), no fixed budget, applied to Reuters topic classification , MNIST digit classification, CIFAR-10 Image classification, CIFAR-100 image classification.

Reuters topic classification and MNIST digit classification use 64 units per layer, CIFAR-10 and CIFAR-100 use 128 units per layer.

All tested activation functions are introduced in 5.• Experiment 2: We tested multi-layer perceptrons (MLP) with fixed budget of 500,000 realvalued parameters, no fixed width, applied to MNIST digit classification, CIFAR-10 Image classification, CIFAR-100 image classification and Reuters topic classification.

All tested activation functions introduced are in section 5.

Used sigmoid(|z| 2 ) function for the gates.• Experiment 3: We tested the Memory Network architecture introduced by BID17 as complex-valued network in two versions -one below and one above parameter budget of the real-valued network.

We used the bAbI question answering tasks with one, two and three supporting facts.

Activation functions in each layer were defined by the original publication.

The network used a recurrent layer, which defined by replacing the real-valued weight matrices with complex weight matrices.

For all of our experiments we used the weight initialisation discussed by BID14 .

However, to reduce the impact of the initialisation we ran each model at least 10 times.

The larger memory networks were initialised 30 times.

All models were trained over 100 epochs with an Adam optimisation.

We used categorical or binary cross entropy as a loss function for all of our experiments and models.

We used sigmoid(|z| 2 ) or so f tmax(|z| 2 ) as activation function for the last layer of the complex models.7 Results and Discussion TAB1 , 3, 4 show the results for experiment 1.

Generally, the performance of complex and real neural network in this setting is similar, altough the complex valued neural network tends to perform slightly worse.

We found that the best choice for the activation function for the complex neural network is relu applied separatly to the imaginary and real parts.

Suprisingly the hyperbolic tangents tanh and squared magnitude |z| 2 perform significantly worse Tables 5, 6 , 7, 8 show the results for experiment 2.

Similar to experiment 1 the results show that the best choice for the activation function for the complex neural network is relu applied separatly to the imaginary and real parts.

In both experiments with the depth of the architecture the performance of the complex neural network decreases significantly.

These experiments illustrate that an increased width per layers outperforms an increased depth in classification tasks.

This is true for the real and the complex case.

Table 9 shows the results for experiment 3.

For a single supporting fact in the bABi data set realvalued neural network.

In the first bABi task the real-valued version outperforms the two complex version of the memory network.

In the more diffcult tasks with two or three supporting facts both, the small and large version, of the complex-valued neural network outperform the real-valued versiondespite the reduce number of parameters.

We made the observation that the assumed structure with introducing complex numbers into neural networks has a regularising effect on the training procedure if used in combination with real-valued input.

We also found that complex-valued neural networks are more sensitive towards their initialisation than real-valued neural networks.

Overall the complex-valued neural networks do not perform as well as expected.

This may be due to the nature of the chosen tasks or the simple architecture of a multi-layer perceptron.

Complex neural networks should be used if the data is naturally in the complex domain or can be mapped to complex numbers.

The architecture should be selected to respect the expected structure complex numbers introduce to the network.

In conclusion the architecture needs to reflect the interaction of the real and imaginary part.

If the structure is ignored, the model will not perform as well as the corresponding real-valued network.

Table 5 : Test accuaracy of multi-layer perceptron conisting of k + 2 dense layers with an overall budget of 500, 000 real-valued parameters on MNIST digit classification.

Selected from best of 10 runs with each run 100 epochs to converge.

Table 6 : Test accuaracy of multi-layer perceptron conisting of k + 2 dense layers with an overall budget of 500, 000 real-valued parameters on Reuters topic classification.

Selected from best of 10 runs with each run 100 epochs to converge.

Table 7 : Test accuaracy of multi-layer perceptron conisting of k + 2 dense layers with an overall budget of 500, 000 real-valued parameters on CIFAR-10 image classification.

Selected from best of 10 runs with each run 100 epochs to converge.

Table 8 : Test accuaracy of multi-layer perceptron conisting of k + 2 dense layers with an overall budget of 500, 000 real-valued parameters on CIFAR-100 image classification.

Selected from best of 10 runs with each run 100 epochs to converge.

identity 0.0000 0.0000 tanh 0.0000 0.0000 relU 0.0000 0.0000 |z| 2 0.0000 0.0000 |z| 0.0000 0.0000 Table 9 : Test accuaracy of Memory Networks BID17 in complex and real version on the first three bAbI tasks.

Selected from best of 30 runs with each run 100 epochs to converge.

@highlight

Comparison of complex- and real-valued multi-layer perceptron with respect to the number of real-valued parameters.