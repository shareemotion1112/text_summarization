To provide principled ways of designing proper Deep Neural Network (DNN) models, it is essential to understand the loss surface of DNNs under realistic assumptions.

We introduce interesting aspects for understanding the local minima and overall structure of the loss surface.

The parameter domain of the loss surface can be decomposed into regions in which activation values (zero or one for rectified linear units) are consistent.

We found that, in each region, the loss surface have properties similar to that of linear neural networks where every local minimum is a global minimum.

This means that every differentiable local minimum is the global minimum of the corresponding region.

We prove that for a neural network with one hidden layer using rectified linear units under realistic assumptions.

There are poor regions that lead to poor local minima, and we explain why such regions exist even in the overparameterized DNNs.

Deep Neural Networks (DNNs) have achieved state-of-the-art performances in computer vision, natural language processing, and other areas of machine learning .

One of the most promising features of DNNs is its significant expressive power.

The expressiveness of DNNs even surpass shallow networks as a network with few layers need exponential number of nodes to have similar expressive power (Telgarsky, 2016) .

The DNNs are getting even deeper after the vanishing gradient problem has been solved by using rectified linear units (ReLUs) BID12 .

Nowadays, RELU has become the most popular activation function for hidden layers.

Leveraging this kind of activation functions, depth of DNNs has increased to more than 100 layers BID7 .Another problem of training DNNs is that parameters can encounter pathological curvatures of the loss surfaces prolonging training time.

Some of the pathological curvatures such as narrow valleys would cause unnecessary vibrations.

To avoid these obstacles, various optimization methods were introduced (Tieleman & Hinton, 2012; BID9 .

These methods utilize the first and second order moments of the gradients to preserve the historical trends.

The gradient descent methods also have a problem of getting stuck in a poor local minimum.

The poor local minima do exist (Swirszcz et al., 2016) in DNNs, but recent works showed that errors at the local minima are as low as that of global minima with high probability BID4 BID2 BID8 BID14 Soudry & Hoffer, 2017) .In case of linear DNNs in which activation function does not exist, every local minimum is a global minimum and other critical points are saddle points BID8 .

Although these beneficial properties do not hold in general DNNs, we conjecture that it holds in each region of parameters where the activation values for each data point are the same as shown in FIG0 .

We prove this for a simple network.

The activation values of a node can be different between data points as shown in FIG0 , so it is hard to apply proof techniques used for linear DNNs.

The whole parameter space is a disjoint union of these regions, so we call it loss surface decomposition.

Using the concepts of loss surface decomposition, we explain why poor local minima do exist even in large networks.

There are poor local minima where gradient flow disappears when using the ReLU (Swirszcz et al., 2016) .

We introduce another kind of poor local minima where the loss is same as that of linear regression.

To be more general, we prove that for each local minimum in a network, there exists a local minimum of the same loss in the larger network that is constructed by adding a node to that network.

DISPLAYFORM0 T .

In each region, activation values are the same.

There are six nonempty regions.

The parameters on the boundaries hit the non-differentiable point of the rectified linear unit.

Loss surface of deep linear networks have the following interesting properties: 1) the function is non-convex and non-concave, 2) every local minimum is a global minimum, 3)

every critical point that is not a global minimum is a saddle point BID8 .

This means that there is no poor local minima problem when using gradient descent methods, but such properties do not hold for nonlinear networks.

We conjecture that these properties hold if activation values are fixed, and we prove it for a simple network.

The loss surface of DNNs can be decomposed into regions in terms of activation values as illustrated in FIG0 .

Let D be a dataset {(x 1 , y 1 ), (x 2 , y 2 ), ..., (x N , y N )} with x i ∈ R n and y i ∈ R. We define a network with one hidden layer as follows: DISPLAYFORM0 The model parameters are W ∈ R h×n , v ∈ R h , b ∈ R h , and c ∈ R where h is the number of hidden nodes.

Let θ = [vec(W ), v, b, c] T collectively denote vectorized form of all the model parameters.

The activation function σ(x) = max(x, 0) is a rectified linear unit, and we abuse notation by generalizing it as an element-wise function for multidimensional inputs.

Alternatively, the network can be expressed in terms of the activation values: DISPLAYFORM1 where DISPLAYFORM2 T is a vector of the binary activation values a ij ∈ {0, 1} of i-th data point x i , and A = (a 1 , a 2 , ..., a N ) is a collection of all activation values for a given dataset D. We fix the activation values of the function g A (x i , θ) regardless of real activation values to find out the interesting properties.

The real model f (x i , θ) agrees with g A (x i , θ) only if A is same as the real activation values in the model.

Before we introduce a definition of the activation region, we denote w A simple example of a non-differentiable local minimum for a dataset x 1 = −2, y 1 = −3, x 2 = +2, y 2 = −1.

In this example, a network is defined by f (x) = w 2 σ(w 1 x) and w 1 is fixed to one.

The non-differentiable local minima exist in a line w 1 = 0 which is a boundary of the two regions.

Note that if DISPLAYFORM3 We consider a general loss function called squared error loss: DISPLAYFORM4 The following lemma state that the local curvatures of DISPLAYFORM5 Lemma 2.2 For any differentiable point θ ∈ R A , the θ is a local minimum (saddle point) in L f (θ) if and only if it is a local minimum (saddle point) in L g A (θ).

The function g A (x i , θ) of fixed activation values A has properties similar to that of linear neural networks.

If all activation values are one, then the function g A (x i , θ) is identical to a linear neural network.

In other cases, some of the parameters are inactive.

The proof becomes tricky since inactive parameters are different for each data point.

In case of the simple network g A (x i , θ), we can convert it into a convex function in terms of other variables.

DISPLAYFORM0 where p j = v j w j and q j = v j b j .

The v j is a j-th scalar value of the vector v and a ij is an activation value on a j-th hidden node of a i-th data point.

DISPLAYFORM1 is a convex function in terms of p j , q j , and c.

Note that for any p j and q j , there exist θ that forms them, so the following lemma holds.

DISPLAYFORM2 Now we introduce the following theorem describing the important properties of the function L g A (θ).Theorem 2.5 The function L g A (θ) has following properties: 1) it is non-convex and non-concave except for the case that activation values are all zeros, 2) every local minimum is a global minimum, 3) every critical point that is not a global minimum is a saddle point.

A function f (x, y) = (xy − 1) 2 is not convex, since it has a saddle point at x = y = 0.

Similarly, the L g A (θ) is a quadratic function of v j b j , so it is non-convex and nonconcave.

If activation values are all zeros, then DISPLAYFORM0 , so the global minima are critical points.

In other critical points, at least one of the gradients along p j or q j is not zero.

If a critical point satisfies ∇ pj L = 0 (or ∇ qj L = 0), then it is a saddle point with respect to w T j and v j (or b j and v j ).

The detailed proof is in the appendix.

To distinguish between the global minimum of L g A (θ) and L f (θ), we introduce subglobal minimum: DISPLAYFORM1 Some of the subglobal minima may not exist in the real loss surface L f (θ).

For this kind of regions, there only exist saddle points and the parameter would move to another region by gradient descent methods without getting stuck into local minima.

Since the parameter space is a disjoint union of the activation regions, the real loss surface L f (θ) is a piecewise combination of L g A (θ).

Using Lemma 2.2 and Theorem 2.5, we conclude as follows: DISPLAYFORM2 The function L f (θ) has following properties: 1) it is non-convex and non-concave, 2) every differentiable local minimum is a subglobal minimum, 3) every critical point that is not a subglobal minimum is a saddle point.

We explicitly distinguish differentiable and non-differentiable local minima.

The non-differentiable local minima can exist as shown in FIG2 .

In this section, we answer why poor local minima do exist even in large networks.

There are parameter points where all the activation values are zeros eliminating gradient flow (Swirszcz et al., 2016) .

This is a well-known region that forms poor and flat local minima.

We introduce another kind of poor region called linear region and show that it always forms poor local minima when a dataset is nonlinear.

In a more general setting, we prove that a network has every local minimum of the narrower networks of the same number of layers.

There always exists a linear region where all activation values are one, and its subglobal minima stay in that region.

This subglobal minimum results in an error which is same as that of linear regression, so if given dataset is nonlinear the error would be poor.

We can easily spot a linear region by manipulating biases to satisfy w T j x i + b j > 0.

One way of achieving this is by selecting b j as: DISPLAYFORM0 To say that the model can get stuck in the linear region, it is necessary to find the subglobal minima in that region.

If f (x i , θ) is linear, then it is of form u T , y 2 = 2, x 3 = [1, 3] T , y 3 = 3.

The network has one hidden layer and no biases.

We increased the number of hidden nodes from one to four.

DISPLAYFORM1

The ratio of the poor regions decreases as the size of the network grows.

We show it numerically by identifying all subglobal minima of a simple network.

For the MNIST (LeCun, 1998), we estimated subglobal minima of randomly selected activation values and compared with the rich regions.

Training a neural network is known to be NP-Complete BID1 , due to the nonconvexity and infinite parameter space of DNNs.

The number of possible combination of activation values has the complexity of O(2 N h ) for f (x i , θ), so we restricted the experiments to a small size of hidden layers and datasets to find all subglobal minima in a reasonable time.

Consider the Equation 4 again.

The subglobal minimum is a solution of the convex optimization for L g A (θ).

To compute optimal parameters, we need to solve linear equations DISPLAYFORM0 , and ∇ c L g A = 0.

For simplicity, we assume that biases are removed, then the gradient ∇

pj L g A is as follows: DISPLAYFORM1 DISPLAYFORM2 As a result, the linear equation to solve is as follows: The leftmost matrix in the Equation 7 is a square matrix.

If it is not full rank, we compute a particular solution.

FIG4 shows four histograms of the poor subglobal minima for the different number of hidden nodes.

As shown in the histograms, gradient descent based methods are more likely to avoid poor subglobal minima in larger networks.

It also shows that the subglobal minima arise from the smaller networks.

Intuitively speaking, adding a node provides a downhill path to the previous poor subglobal minima without hurting the rich subglobal minima in most cases.

DISPLAYFORM3

For more realistic networks and datasets, we conducted experiments on MNIST.

We used networks of two hidden layers consisting of 2k and k nodes respectively.

The networks use biases, softmax outputs, cross entropy loss, mini-batch size of 100, and Adam Optimizer BID9 .

Assuming that the Corollary 2.7 holds for multilayer networks, the subglobal minima can be estimated by gradient descent methods.

It is impossible to compute all of them, so we randomly selected various combinations of activation values with P (a = 1) = P (a = 0) = 0.5.

Then we removed rectified linear units and multiplied the fixed activation values as follows: DISPLAYFORM0 where h A is the output of the second hidden layer.

The rich subglobal minima were estimated by optimizing the real networks since it would end up in one of the subglobal minima that exist in the real loss surface.

The experiments were repeated for 100 times, and then we computed mean and standard deviation.

The results are shown in TAB0 and it implies that most of the regions in the large networks are rich, whereas the small networks have few rich regions.

In other words, it is more likely to end up in a rich subglobal minimum in larger networks.5 RELATED WORKS BID0 proved that linear networks with one hidden layer have the properties of the Theorem 2.5 under minimal assumptions.

Recently, BID8 proved that it also holds for deep linear networks.

Assuming that the activation values are drawn from independent Bernoulli distribution, a DNN can be mapped to a spin-glass Ising model in which the number of local minima far from the global minima diminishes exponentially with the size of the network BID2 .

Under same assumptions in BID2 , the effect of nonlinear activation values disappears by taking expectation, so nonlinear networks satisfy the same properties of linear networks BID8 .Nonlinear DNNs usually do not encounter any significant obstacles on a single smooth slope path BID6 BID4 explained that the training error at local minima seems to be similar to the error at the global minimum which can be understood via random matrix theory.

The volume of differentiable sub-optimal local minima is exponentially vanishing in comparison with the same volume of global minima under infinite data points (Soudry & Hoffer, 2017) .

Although a number of specific example of local minima can be found in DNNs (Swirszcz et al., 2016) , it seems plausible to state that most of the local minima are near optimal.

As the network width increases, we are more likely to meet a random starting point from which there is a continuous, strictly monotonically decreasing path to a global minimum BID14 .

Similarly, the starting point of the DNNs approximate a rich family of hypotheses precisely BID3 .

Another explanation is that the level sets of the loss become connected as the network is increasingly overparameterized BID5 .

These works are analogous to our resultsshowing that the parameters would end up in one of the subglobal minima which are similar to the global minima.

We conjecture that the loss surface is a disjoint union of activation regions where every local minimum is a subglobal minimum.

Using the concept of loss surface decomposition, we studied the existence of poor local minima and experimentally investigated losses of subglobal minima.

However, the structure of non-differentiable local minima is not yet well understood yet.

These non-differentiable points exist within the boundaries of the activation regions which can be obstacles when using gradient descent methods.

Further work is needed to extend knowledge about the local minima, activation regions, their boundaries.

Let θ ∈ R A be a differentiable point, so it is not in the boundaries of the activation regions.

This implies that w T j x i + b j = 0 for all parameters.

Without loss of generality, we assume w T j x i + b j < 0.

Then there exist > 0 such that w T j x i + b j + < 0.

This implies that small changes in the parameters for any direction does not change the activation region.

Since L f (θ) and L g A (θ) are equivalent in the region R A , the local curvatures of these two function around the θ are also the same.

Thus, the θ is a local minimum (saddle point) in L f (θ) if and only if it is a local minimum (saddle point) in L g A (θ).

DISPLAYFORM0 is a linear transformation of p j , q j , and c, the DISPLAYFORM1 2 is convex in terms of p j , q j , and c. Summation of convex functions is convex, so the lemma holds.

A.3 PROOF OF THEOREM 2.5(1) Assume that activation values are not all zeros, and then consider the following Hessian matrix evaluated from v j and b j for some non-zero activation values a ij > 0: DISPLAYFORM2 Let v j = 0 and b j = 0, then two eigenvalues of the Hessian matrix are as follows: DISPLAYFORM3 There exist c > 0 such that g A (x i , θ) > y i for all i. If we choose such c, then DISPLAYFORM4 ∂vj ∂bj > 0 which implies that two eigenvalues are positive and negative.

Since the Hessian matrix is not positive semidefinite nor negative semidefinite, the function L g A (θ) is non-convex and non-concave.(2, 3) We organize some of the gradients as follows: DISPLAYFORM5 We select a critical point θ * where ∇ wj L g A (θ * ) = 0, ∇ vj L g A (θ * ) = 0, ∇ bj L g A (θ * ) = 0, and ∇ c L g A (θ * ) = 0 for all j.

Case 1) Assume that ∇ pj L g A (θ * ) = 0 and ∇ qj L g A (θ * ) = 0 for all j.

These points are global minima, since ∇ c L g A (θ * ) = 0 and L g A (θ) is convex in terms of p j , q j , and c.

Case 2) Assume that there exist j such that ∇ pj L g A (θ DISPLAYFORM6 There exist an element w * in w j such that ∇ vj ∇ w * L g A (θ * ) = 0.

Consider a Hessian matrix evaluated from w * and v j .

Analogous to the proof of (1), this matrix is not positive semidefinite nor negative semidefinite.

Thus θ * is a saddle point.

Case 3) Assume that there exist j such that ∇ qj L g A (θ * ) = 0.

Since ∇ bj L g A (θ * ) = v j ∇ qj L g A (θ * ) = 0, the v j is zero.

Analogous to the Case 2, a Hessian matrix evaluated from b j and v j is not positive semidefinite nor negative semidefinite.

Thus θ * is a saddle point.

As a result, every critical point is a global minimum or a saddle point.

Since L g A (θ) is a differentiable function, every local minimum is a critical point.

Thus every local minimum is a global minimum.

<|TLDR|>

@highlight

The loss surface of neural networks is a disjoint union of regions where every local minimum is a global minimum of the corresponding region.