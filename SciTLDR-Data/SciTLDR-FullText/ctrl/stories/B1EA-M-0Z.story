It has long been known that a single-layer fully-connected neural network with an i.i.d.

prior over its parameters is equivalent to a Gaussian process (GP), in the limit of infinite network width.

This correspondence enables exact Bayesian inference for infinite width neural networks on regression tasks by means of evaluating the corresponding GP.

Recently, kernel functions which mimic multi-layer random neural networks have been developed, but only outside of a Bayesian framework.

As such, previous work has not identified that these kernels can be used as covariance functions for GPs and allow fully Bayesian prediction with a deep neural network.



In this work, we derive the exact equivalence between infinitely wide, deep, networks and GPs with a particular covariance function.

We further develop a computationally efficient pipeline to compute this covariance function.

We then use the resulting GP to perform Bayesian inference for deep neural networks on MNIST and CIFAR-10.

We observe that the trained neural network accuracy approaches that of the corresponding GP with increasing layer width, and that the GP uncertainty is strongly correlated with trained network prediction error.

We further find that test performance increases as finite-width trained networks are made wider and  more  similar  to  a  GP,  and  that  the  GP-based  predictions  typically  outperform  those  of  finite-width  networks.

Finally  we  connect  the  prior  distribution over weights and variances in our GP formulation to the recent development of signal propagation in random neural networks.

Deep neural networks have emerged in recent years as flexible parametric models which can fit complex patterns in data.

As a contrasting approach, Gaussian processes have long served as a traditional nonparametric tool for modeling.

An equivalence between these two approaches was derived in BID17 , for the case of one layer networks in the limit of infinite width.

Neal (1994a) further suggested that a similar correspondence might hold for deeper networks.

Consider a deep fully-connected neural network with i.i.d.

random parameters.

Each scalar output of the network, an affine transformation of the final hidden layer, will be a sum of i.i.d.

terms.

As we will discuss in detail below, in the limit of infinite width the Central Limit Theorem 1 implies that the function computed by the neural network (NN) is a function drawn from a Gaussian process (GP).

In the case of single hidden-layer networks, the form of the kernel of this GP is well known BID17 BID25 ).This correspondence implies that if we choose the hypothesis space to be the class of infinitely wide neural networks, an i.i.d.

prior over weights and biases can be replaced with a corresponding GP prior over functions.

As noted by BID25 , this substitution enables exact Bayesian inference for regression using neural networks.

The computation requires building the necessary covariance matrices over the training and test sets and straightforward linear algebra computations.

In light of the resurgence in popularity of neural networks, it is timely to revisit this line of work.

We delineate the correspondence between deep and wide neural networks and GPs and utilize it for Bayesian training of neural networks on regression tasks.

Our work touches on aspects of GPs, Bayesian learning, and compositional kernels.

The correspondence between infinite neural networks and GPs was first noted by BID17 b) .

BID25 computes analytic GP kernels for single hidden-layer neural networks with error function or Gaussian nonlinearities and noted the use of the GP prior for exact Bayesian inference in regression.

BID6 discusses several routes to building deep GPs and observes the degenerate form of kernels that are composed infinitely many times -a point we will return to Section 3.2 -but they do not derive the form of GP kernels as we do.

BID10 also discusses constructing kernels equivalent to infinitely wide deep neural networks, but their construction does not go beyond two hidden layers with nonlinearities.

Related work has also appeared outside of the GP context but in compositional kernel constructions.

BID2 derives compositional kernels for polynomial rectified nonlinearities, which includes the Sign and ReLU nonlinearities, and can be used in GPs; our manner of composing kernels matches theirs, though the context is different.

BID4 extends the construction of compositional kernels to neural networks whose underlying directed acyclic graph is of general form.

They also prove, utilizing the formalism of dual activations, that compositional kernels originating from fully-connected topologies with the same nonlinearity become degenerate when composed infinitely many times.

In a different context than compositional kernels, BID19 ; BID24 study the same underlying recurrence relation for the specific case of fully-connected networks and bounded nonlinearities.

They distinguish regions in hyperparameter space with different fixed points and convergence behavior in the recurrence relations.

The focus in these works was to better understand the expressivity and trainability of deep networks.

Drawing inspiration from the multi-layer nature of deep neural networks, there is a line of work considering various approaches to stacking GPs, such as deep GPs BID15 ; BID3 ; BID11 ; BID6 ; BID1 ), which can give rise to a richer class of probabilistic models beyond GPs.

This contrasts with our work, where we study GPs that are in direct correspondence with deep, infinitely wide neural networks.

BID14 has recently explored the performance of GP models with deep kernels given in BID2 , implemented with scalable approximations.

However, they do not discuss the equivalence between deep neural networks and GPs with compositional kernels, which constitutes a conceptual contribution of our work.

Furthermore, we note that the GP kernels in our work are more general than the compositional kernel construction outlined in BID2 in two respects: (i) we are not limited to rectified polynomials but can deal with general nonlinearities, and (ii) we consider two additional hyperparameters in the kernels, which would correspond to the weight and bias parameter variances in a neural network.

Finally, BID8 connects dropout in deep neural networks with approximate Bayesian inference in deep GPs.

Another series of recent works BID28 a) ; BID0 ), termed deep kernel learning, utilize GPs with base kernels which take in features produced by a deep multilayer neural network, and train the resulting model end-to-end.

Our work differs from these in that our GP corresponds to a multilayer neural network.

Additionally, our GP kernels have many fewer parameters, and these parameters correspond to the hyperparameters of the equivalent neural network.

We begin by specifying the form of a GP which corresponds to a deep, infinitely wide neural network -hereafter referred to as the Neural Network GP (NNGP) -in terms of a recursive, deterministic computation of the kernel function.

The prescription is valid for generic pointwise nonlinearities in fully-connected feedforward networks.

We develop a computationally efficient method (Section 2.5) to compute the covariance function corresponding to deep neural networks with fixed hyperparameters.

In this work, as a first proof of concept of our NNGP construction, we focus on exact Bayesian inference for regression tasks, treating classification as regression on class labels.

While less principled, least-squares classification performs well BID23 and allows us to compare exact inference via a GP to prediction by a trained neural network on well-studied tasks (MNIST and CIFAR-10 classification).

Note that it is possible to extend GPs to softmax classification with cross entropy loss BID26 ; BID21 ), which we aim to investigate in future work.

We conduct experiments making Bayesian predictions on MNIST and CIFAR-10 (Section 3) and compare against NNs trained with standard gradient-based approaches.

The experiments explore different hyperparameter settings of the Bayesian training including network depth, nonlinearity, training set size (up to and including the full dataset consisting of tens of thousands of images), and weight and bias variance.

Our experiments reveal that the best NNGP performance is consistently competitive against that of NNs trained with gradient-based techniques, and the best NNGP setting, chosen across hyperparameters, often surpasses that of conventional training (Section 3, TAB0 ).

We further observe that, with increasing network width, the performance of neural networks with gradient-based training approaches that of the NNGP computation, and that the GP uncertainty is strongly correlated with prediction error.

Furthermore, the performance of the NNGP depends on the structure of the kernel, which can be connected to recent work on signal propagation in networks with random parameters BID24 .

We begin by specifying the correspondence between GPs and deep, infinitely wide neural networks, which hinges crucially on application of the Central Limit Theorem.

We review the single-hidden layer case (Section 2.2) before moving to the multi-layer case (Section 2.3).

Consider an L-hidden-layer fully-connected neural network with hidden layers of width N l (for layer l) and pointwise nonlinearities ??.

Let x ??? R din denote the input to the network, and let z L ??? R dout denote its output.

The ith component of the activations in the lth layer, post-nonlinearity and postaffine transformation, are denoted x l i and z l i respectively.

We will refer to these as the post-and pre-activations. (We let x 0 i ??? x i for the input, dropping the Arabic numeral superscript, and instead use a Greek superscript x ?? to denote a particular input ??).

Weight and bias parameters for the lth layer have components W l ij , b l i , which are independent and randomly drawn, and we take them all to have zero mean and variances ?? 2 w /N l and ?? 2 b , respectively.

GP(??, K) denotes a Gaussian process with mean and covariance functions ??(??), K(??, ??), respectively.

We briefly review the correspondence between single-hidden layer neural networks and GPs BID17 b); BID25 ).

The ith component of the network output, z 1 i , is computed as, DISPLAYFORM0 where we have emphasized the dependence on input x. Because the weight and bias parameters are taken to be i.i.d., the post-activations x 1 j , x 1 j are independent for j = j .

Moreover, since z 1 i (x) is a sum of i.i.d terms, it follows from the Central Limit Theorem that in the limit of infinite width N 1 ??? ???, z 1 i (x) will be Gaussian distributed.

Likewise, from the multidimensional Central Limit Theorem, any finite collection of {z DISPLAYFORM1 } will have a joint multivariate Gaussian distribution, which is exactly the definition of a Gaussian process.

Therefore we conclude that z DISPLAYFORM2 , a GP with mean ?? 1 and covariance K 1 , which are themselves independent of i. Because the parameters have zero mean, we have that ?? 1 (x) = E z 1 i (x) = 0 and, DISPLAYFORM3 where we have introduced C(x, x ) as in BID17 ; it is obtained by integrating against the distribution of W 0 , b 0 .

Note that, as any two z 1 i , z 1 j for i = j are joint Gaussian and have zero covariance, they are guaranteed to be independent despite utilizing the same features produced by the hidden layer.

The arguments of the previous section can be extended to deeper layers by induction.

We proceed by taking the hidden layer widths to be infinite in succession (N 1 ??? ???, N 2 ??? ???, etc.) as we continue with the induction, to guarantee that the input to the layer under consideration is already governed by a GP.

In Appendix C we provide an alternative derivation in terms of Bayesian marginalization over intermediate layers, which does not depend on the order of limits, in the case of a Gaussian prior on the weights.

A concurrent work BID5 further derives the convergence rate towards a GP if all layers are taken to infinite width simultaneously, but at different rates.

Suppose that z l???1 j is a GP, identical and independent for every j (and hence x l j (x) are independent and identically distributed).

After l ??? 1 steps, the network computes DISPLAYFORM0 As before, z l i (x) is a sum of i.i.d.

random terms so that, as N l ??? ???, any finite collection {z DISPLAYFORM1 By induction, the expectation in Equation 4 is over the GP governing z DISPLAYFORM2 , but this is equivalent to integrating against the joint distribution of only z l???1 i (x) and z l???1 i (x ).

The latter is described by a zero mean, two-dimensional Gaussian whose covariance matrix has distinct entries DISPLAYFORM3 , and K l???1 (x , x ).

As such, these are the only three quantities that appear in the result.

We introduce the shorthand DISPLAYFORM4 to emphasize the recursive relationship between K l and K l???1 via a deterministic function F whose form depends only on the nonlinearity ??.

This gives an iterative series of computations which can be performed to obtain K L for the GP describing the network's final output.

For the base case DISPLAYFORM5 ; we can utilize the recursion relating K 1 and K 0 , where DISPLAYFORM6 In fact, these recurrence relations have appeared in other contexts.

They are exactly the relations derived in the mean field theory of signal propagation in fully-connected random neural networks BID19 BID24 ) and also appear in the literature on compositional kernels BID2 BID4 ).

For certain activation functions, Equation 5 can be computed analytically BID2 BID4 ).

In the case of the ReLU nonlinearity, it yields the well-known arccosine kernel BID2 ) whose form we reproduce in Appendix B. When no analytic form exists, it can instead be efficiently computed numerically, as described in Section 2.5.

Here we provide a short review of how a GP prior over functions can be used to do Bayesian inference; see e.g. BID21 for a comprehensive review of GPs.

Given a dataset DISPLAYFORM0 , we wish to make a Bayesian prediction at test point x * using a distribution over functions z(x).

This distribution is constrained to take values z ??? (z 1 , ..., z n ) on the training inputs x ??? (x 1 , ..., x n ) and, DISPLAYFORM1 where t = (t 1 , ..., t n ) T are the targets on the training set, and P (t|z) corresponds to observation noise.

We will assume a noise model consisting of a Gaussian with variance ?? 2 centered at z.

If the conditions of Section 2.2 or 2.3 apply, our choice of prior over functions implies that z 1 , ..., z n , z * are n + 1 draws from a GP and z * , z|x * , x ??? N (0, K) is a multivariate Gaussian whose covariance matrix has the form DISPLAYFORM2 where the block structure corresponds to the division between the training set and the test point.

DISPLAYFORM3 As is standard in GPs, the integral in Equation 7 can be done exactly, resulting in z DISPLAYFORM4 where I n is the n ?? n identity.

The predicted distribution for z * |D, x * is hence determined from straightforward matrix computations, yet nonetheless corresponds to fully Bayesian training of the deep neural network.

The form of the covariance function used is determined by the choice of GP prior, i.e. the neural network model class, which depends on depth, nonlinearity, and weight and bias variances.

We henceforth resume placing a superscript L as in K L to emphasize the choice of depth for the compositional kernel.

Given an L-layer deep neural network with fixed hyperparameters, constructing the covariance matrix K L for the equivalent GP involves computing the Gaussian integral in Equation 4 for all pairs of training-training and training-test points, recursively for all layers.

For some nonlinearities, such as ReLU, this integration can be done analytically.

However, to compute the kernel corresponding to arbitrary nonlinearities, the integral must be performed numerically.

The most direct implementation of a numerical algorithm for K L would be to compute integrals independently for each pair of datapoints and each layer.

This is prohibitively expensive and costs O n 2 g L(n 2 train + n train n test ) , where n 2 g is the sampling density for the pair of Gaussian random variables in the 2D integral and n train , n test are the training and test set sizes, respectively.

However, by careful pipelining, and by preprocessing all inputs to have identical norm, we can improve this cost to O n 2 g n v n c + L(n 2 train + n train n test ) , where n v and n c are sampling densities for a variance and correlation grid, as described below.

In order to achieve this, we break the process into several steps:1.

Generate: pre-activations u = [???u max , ?? ?? ?? , u max ] consisting of n g elements linearly spaced between ???u max and u max ; variances s = [0, ?? ?? ?? , s max ] with n v linearly spaced elements, where s max < u 2 max ; and correlations c = (???1, ?? ?? ?? , 1) with n c linearly spaced elements.

Note that we are using fixed, rather than adaptive, sampling grids to allow operations to be parallelized and reused across datapoints and layers.2.

Populate a matrix F containing a lookup table for the function F ?? in Equation 5.

This involves numerically approximating a Gaussian integral, in terms of the marginal variances s and correlations c. We guarantee that the marginal variance is identical for each datapoint, by preprocessing all datapoints to have identical norm at the input layer, so the number of entries in the lookup table need only be n v n c .

These entries are computed as 2 : DISPLAYFORM0 3.

For every pair of datapoints x and x in layer l, compute K l (x, x ) using Equation 5.Approximate the function DISPLAYFORM1 Step 2, where we interpolate into s using the value of K l???1 (x, x), and interpolate into c using DISPLAYFORM2 , due to data preprocessing to guarantee constant norm.4.

Repeat the previous step recursively for all layers.

Bilinear interpolation has constant cost, so this has cost O L(n 2 train + n train n test ) .This computational recipe allows us to compute the covariance matrix for the NNGP corresponding to any well-behaved nonlinearity ??.

All computational steps above can be implemented using accelerated tensor operations, and computation of K L is typically faster than solving the system of linear equations in Equation 8-9.

Figure 6 illustrates the close agreement between the kernel function computed numerically (using this approach) and analytically, for the ReLU nonlinearity.

It also illustrates the angular dependence of the kernel and its evolution with increasing depth.

Finally, note that the full computational pipeline is deterministic and differentiable.

The shape and properties of a deep network kernel are purely determined by hyperparameters of the deep neural network.

Since GPs give exact marginal likelihood estimates, this kernel construction may allow principled hyperparameter selection, or nonlinearity design, e.g. by gradient ascent on the log likelihood w.r.t.

the hyperparameters.

Although this is not the focus of current work, we hope to return to this topic in follow-up work.

An open source implementation of the algorithm is available at https://github.com/brainresearch/nngp.

We compare NNGPs with SGD 3 trained neural networks on the permutation invariant MNIST and CIFAR-10 datasets.

The baseline neural network is a fully-connected network with identical width at each hidden layer.

Training is on the mean squared error (MSE) loss, chosen so as to allow direct comparison to GP predictions.

Formulating classification as regression often leads to good results BID22 .

Future work may involve evaluating the NNGP on a cross entropy loss using the approach in BID26 BID21 .

Training used the Adam optimizer BID13 ) with learning rate and initial weight/bias variances optimized over validation error using the Google Vizier hyperparameter tuner BID9 .

Dropout was not used.

In future work, it would be interesting to incorporate dropout into the NNGP covariance matrix using an approach like that in BID24 .

For the study, nonlinearities were chosen to be either rectified linear units (ReLU) or hyperbolic tangent (Tanh).

Class labels were encoded as a one-hot, zero-mean, regression target (i.e., entries of -0.1 for the incorrect class and 0.9 for the correct class).

We constructed the covariance kernel numerically for ReLU and Tanh nonlinearities following the method described in Section 2.5.Performance: We find that the NNGP often outperforms trained finite width networks.

See TAB0 and FIG0 .

The NNGP often outperforms finite width networks, and neural network performance more closely resembles NNGP performance with increasing width.

Test accuracy and mean squared error on MNIST and CIFAR-10 dataset are shown for the best performing NNGP and best performing SGD trained neural networks for given width. '

NN-best' denotes the best performing (on the validation set) neural network across all widths and trials.

Often this is the neural network with the largest width.

We additionally find the performance of the best finite-width NNs, trained with a variant of SGD, approaches that of the NNGP with increasing layer width.

This is interesting from at least two, potentially related, standpoints.(1) NNs are commonly believed to be powerful because of their ability to do flexible representation learning, while our NNGP uses fixed basis functions; nonetheless, in our experiments we find no salient performance advantage to the former.

(2) It hints at a possible relationship between SGD and Bayesian inference in certain regimes -were the neural networks trained in a fully Bayesian fashion, rather than by SGD, the approach to NNGP in the large width limit would be guaranteed.

There is recent work suggesting that SGD can implement approximate Bayesian inference BID16 under certain assumptions.

The similarity of the performance of the widest NN in FIG0 with the NNGP suggests that the limit of infinite network width, which is inherent to the GP, is far from being a disadvantage.

Indeed, in practice it is found that the best generalizing NNs are in fact the widest.

To support this, in FIG1 we show generalization gap results from an experiment in which we train 180 fully-connected networks with five hidden layers on CIFAR-10 with a range of layer widths.

For this experiment, we trained the networks using a standard cross entropy loss rather than MSE, leading to a slight difference in performance.

Uncertainty: One benefit in using a GP is that, due to its Bayesian nature, all predictions have uncertainty estimates (Equation 9).

For conventional neural networks, capturing the uncertainty in a model's predictions is challenging BID7 .

In the NNGP, every test point has an explicit estimate of prediction variance associated with it (Equation 9).

In our experiments, we observe that the NNGP uncertainty estimate is highly correlated with prediction error (Figure 3 ).

commonly approach a functionally uninteresting fixed point with depth l ??? ???, in that K ??? (x, x ) becomes a constant or piecewise constant map.

We now briefly relate our ability to train NNGPs with the convergence of K l (x, x ) to the fixed-point kernel.

We will be particularly interested in contextualizing our results in relation to BID19 For the Tanh nonlinearity, there are two distinct phases respectively called the "ordered" phase and the "chaotic" phase that can be understood as a competition between the weights and the biases of the network.

A diagram showing these phases and the boundary between them is shown in Figure 4a .

In the ordered phase, the features obtained by propagating an input through the each layer of the recursion become similar for dissimilar inputs.

Fundamentally, this occurs because the different inputs share common bias vectors and so all inputs end up just approaching the random bias.

In this case the covariance K l (x, x ) ??? q * for every pair of inputs x, x , where q * is a constant that depends only on ?? 2 w and ?? 2 b .

All inputs have unit correlation asymptotically with depth.

By contrast in the chaotic phase the weight variance ?? 2 w dominates and similar inputs become dissimilar with depth as they are randomly projected by the weight matrices.

In this case, the covariance K l (x, x ) ??? q * for x = x but q * c * for x = x .

Here c * < 1 is the fixed point correlation.

In each of these regimes, there is also a finite depth-scale ?? which describes the characteristic number of layers over which the covariance function decays exponentially towards its fixed point form.

Exactly at the boundary between these two regimes is a line in (?? FORMULA0 that this approach to the fixed-point covariance fundamentally bounded whether or not neural networks could successfully be trained.

It was shown that initializing networks on this line allowed for significantly deeper neural networks to be trained.

For ReLU networks a similar picture emerges, however there are some subtleties due to the unbounded nature of the nonlinearity.

In this case for all ?? 2 w and ?? 2 b , K ??? (x, x ) = q * for all x, x and every point becomes asymptotically correlated.

Despite this, there are again two phases: a "bounded" phase in which q * is finite (and nonzero) and an unbounded phase in which q * is either infinite or zero.

As in the Tanh case there are depth scales that control the rate of convergence to these fixed points and therefore limit the maximum trainable depth.

The phase diagram for the ReLU nonlinearity is also shown in Figure 4b .In a striking analogy with the trainability of neural networks, we observe that the performance of the NNGP appears to closely track the structure from the phase diagram, clearly illustrated in Figure 4 .

Indeed, we see that as for hyperparameter settings that are far from criticality, the GP is unable to train and we encounter poor test set performance.

By contrast, near criticality we observe that our models display high accuracy.

Moreover, we find that the accuracy appears to drop more quickly away from the phase boundary with increase in depth L of the GP kernel, K L .

To understand this effect we note that information about data will be available to our model only through the difference DISPLAYFORM0 However, as the depth gets larger, this difference becomes increasingly small and at some point can no longer be represented due to numerical precision.

At this point our test accuracy begins to quickly degrade to random chance.

By harnessing the limit of infinite width, we have specified a correspondence between priors on deep neural networks and Gaussian processes whose kernel function is constructed in a compositional, but fully deterministic and differentiable, manner.

Use of a GP prior on functions enables exact Bayesian inference for regression from matrix computations, and hence we are able to obtain predictions and uncertainty estimates from deep neural networks without stochastic gradient-based training.

The performance is competitive with the best neural networks (within specified class of fully-connected models) trained on the same regression task under similar hyperparameter settings.

While we were able to run experiments for somewhat large datasets (sizes of 50k), we intend to look into scalability for larger learning tasks, possibly harnessing recent progress in scalable GPs BID20 ; BID12 ).

In our experiments, we observed the performance of the optimized neural network appears to approach that of the GP computation with increasing width.

Whether gradient-based stochastic optimization implements an approximate Bayesian computation is an interesting question BID16 .

Further investigation is needed to determine if SGD does approximately implement Bayesian inference under the conditions typically employed in practice.

Additionally, the NNGP provides explicit estimates of uncertainty.

This may be useful in predicting model failure in critical applications of deep learning, or for active learning tasks where it can be used to identify the best datapoints to hand label.

A DRAWS FROM AN NNGP PRIOR FIG5 illustrates the nature of the GP prior for the ReLU nonlinearity by depicting samples of 1D functions z(x) drawn from a ReLU GP, GP(0, K L ), with fixed depth L = 10 and (?? Figure 6: The angular structure of the kernel and its evolution with depth.

Also illustrated is the good agreement between the kernel computed using the methods of Section 2.5 (blue, starred) and the analytic form of the kernel (red).

The depth l in K l runs from l = 0, ..., 9 (flattened curves for increasing l), and (?? In the main text, we noted that the recurrence relation Equation 5 can be computed analytically for certain nonlinearities.

In particular, this was computed in BID2 for polynomial rectified nonlinearities.

For ReLU, the result including the weight and bias variance is DISPLAYFORM0 To illustrate the angular form of K l (x, x ) and its evolution with l, in Figure 6 we plot K l (??) for the ReLU nonlinearity, where ?? is the angle between x and x with norms such that ||x|| 2 = ||x || 2 = d in .

We observe a flattening of the angular structure with increase in depth l, as predicted from the understanding in Section 3.2.

Simultaneously, the figure also illustrates the good agreement between the kernel computed using the numerical implementation of Section 2.5 (blue, starred) and the analytic arccosine kernel, Equation 11 (red), for a particular choice of hyperparameters (??

In this section, we present an alternate derivation of the equivalence between infinitely wide deep neural networks and Gaussian process by marginalization over intermediate layers.

For this derivation, we take the weight and bias parameters to be drawn from independent Gaussians, with zero mean and appropriately scaled variance.

We are interested in finding the distribution p(z L |x) over network outputs z L ??? R dout??B , conditioned on network inputs x ??? R din??B , for input dimensionality d in , output dimensionality d out , and dataset size B. Intervening layers will have width N l , z l ??? R N l+1 ??B for L > l > 0.

We define the second moment matrix (here post-nonlinearity) for each layer l to be DISPLAYFORM0 Our approach is to think of intermediate random variables corresponding to these second moments defined above.

By definition, K l only depends on z l???1 .

In turn, the pre-activations z l are described by a Gaussian process conditioned on the second moment matrix K l , DISPLAYFORM1 where DISPLAYFORM2 This correspondence of each layer to a GP, conditioned on the layer's second moment matrix, is exact even for finite width N l because the parameters are drawn from a Gaussian.

Altogether, this justifies the graphical model depicted in Figure 7 .We will write p(z L |x) as an integral over all the intervening second moment matrices DISPLAYFORM3 This joint distribution can be decomposed as DISPLAYFORM4 The directed decomposition in Equation 16 holds because DISPLAYFORM5 Figure 7: Graphical model for neural network's computation.

The sum in Equation 12 for l > 0 is a sum over i.i.d.

terms.

As N l grows large, the Central Limit Theorem applies, and p K l |K l???1 converges to a Gaussian with variance that shrinks as 1 N l .

Further, in the infinite width limit it will go to a delta function, DISPLAYFORM6 with F (??) defined as in FIG5 .

Similarly, the dependence of K 0 on x can be expressed as a delta function, DISPLAYFORM7 So, in the limit of infinite width, z L |x is described by a Gaussian process with kernel DISPLAYFORM8

We outline details of the experiments for Section 3.

For MNIST we use a 50k/10k/10k split of the training/validation/test dataset.

For CIFAR-10, we used a 45k/5k/10k split.

The validation set was used for choosing the best hyperparameters and evaluation on the test set is reported.

For training neural networks hyperparameters were optimized via random search on average 250 trials for each choice of (n train , depth, width, nonlinearity).Random search range: Learning rate was sampled within (10 ???4 , 0.2) in log-scale, weight decay constant was sampled from (10 ???8 , 1.0) in log-scale, ?? w ??? [0.01, 2.5], ?? b ??? [0, 1.5] was uniformly sampled and mini-batch size was chosen equally among [16, 32, 64, 128, 256] .For the GP with given depth and nonlinearity, a grid of 30 points evenly spaced from 0.1 to 5.0 (for ?? 2 w ) and 30 points evenly spaced from 0 to 2.0 (for ?? Computation time: We report computation times for NNGP experiments.

The grid generation with took 440-460s with 6 CPUs for n g = 501, n v = 501, n c = 500, which was amortized over all the experiments.

For full (50k) MNIST, constructing K DD for each layer took 90-140s (depending on CPU generation) running on 64 CPUs.

Solving linear equations via Cholesky decomposition took 180-220s for 1000 test points.

For all the experiments we used pre-computed lookup tables F with n g = 501, n v = 501, n c = 500, and s max = 100.

Default value for the target noise ?? 2 was set to 10 ???10 and was increased by factor of 10 when Cholesky decomposition failed while solving Equation 8 and 9.

We refer to BID21 for standard numerically stable implementation of GP regression.

Here we include more results from experiments described in Section 3.Uncertainty: Relationship between the target MSE and the GP's uncertainty estimate for smaller training set size is shown in Figure 8 .

0.5566 GP-3-3.48-1.52 0.5558

<|TLDR|>

@highlight

We show how to make predictions using deep networks, without training deep networks.