We show that the output of a (residual) CNN with an appropriate prior over the weights and biases is a GP in the limit of infinitely many convolutional filters, extending similar results for dense networks.

For a CNN, the equivalent kernel can be computed exactly and, unlike "deep kernels", has very few parameters: only the hyperparameters of the original CNN.

Further, we show that this kernel has two properties that allow it to be computed efficiently; the cost of evaluating the kernel for a pair of images is similar to a single forward pass through the original CNN with only one filter per layer.

The kernel equivalent to a 32-layer ResNet obtains 0.84% classification error on MNIST, a new record for GP with a comparable number of parameters.

Convolutional Neural Networks (CNNs) have powerful pattern-recognition capabilities that have recently given dramatic improvements in important tasks such as image classification BID13 .

However, as CNNs are increasingly being applied in real-world, safety-critical domains, their vulnerability to adversarial examples BID27 BID15 , and their poor uncertainty estimates are becoming increasingly problematic.

Bayesian inference is a theoretically principled and demonstrably successful BID26 BID7 framework for learning in the face of uncertainty, which may also help to address the problems of adversarial examples BID9 .

Unfortunately, Bayesian inference in CNNs is extremely difficult due to the very large number of parameters, requiring highly approximate factorised variational approximations BID1 BID8 , or requiring the storage BID16 ) of large numbers of posterior samples (Welling & Teh, 2011; BID19 .Other methods such as those based on Gaussian Processes (GPs) are more amenable to Bayesian inference, allowing us to compute the posterior uncertainty exactly BID24 .

This raises the question of whether it might be possible to combine the pattern-recognition capabilities of CNNs with exact probabilistic computations in GPs.

Two such approaches exist in the literature.

First, deep convolutional kernels (Wilson et al., 2016 ) parameterise a GP kernel using the weights and biases of a CNN, which is used to embed the input images into some latent space before computing their similarity.

The CNN parameters of the resulting kernel then have to be optimised by gradient descent.

However, the large number of kernel parameters in the CNN reintroduces the risk of overconfidence and overfitting.

To avoid this risk, we need to infer a posterior over the CNN kernel parameters, which is as difficult as directly inferring a posterior over the parameters of the original CNN.

Second, it is possible to define a convolutional GP BID22 or a Furthermore, we show that two properties of the GP kernel induced by a CNN allow it to be computed very efficiently.

First, in previous work it was necessary to compute the covariance matrix for the output of a single convolutional filter applied at all possible locations within a single image BID22 , which was prohibitively computationally expensive.

In contrast, under our prior, the downstream weights are independent with zero-mean, which decorrelates the contribution from each location, and implies that it is necessary only to track the patch variances, and not their covariances.

Second, while it is still necessary to compute the variance of the output of a convolutional filter applied at all locations within the image, the specific structure of the kernel induced by the CNN means that the variance at every location can be computed simultaneously and efficiently as a convolution.

Finally, we empirically demonstrate the performance increase coming from adding translationinvariant structure to the GP prior.

Without computing any gradients, and without augmenting the training set (e.g. using translations), we obtain 0.84% error rate on the MNIST classification benchmark, setting a new record for nonparametric GP-based methods.

For clarity of exposition, we will treat the case of a 2D convolutional NN.

The result applies straightforwardly to nD convolutions, dilated convolutions and upconvolutions ("deconvolutions"), since they can be represented as linear transformations with tied coefficients (see Fig. 1 ).

The network takes an arbitrary input image X of height H (0) and width DISPLAYFORM0 Each row, which we denote x 1 , x 2 , . . .

, x C (0) , corresponds to a channel of the image (e.g. C (0) = 3 for RGB), flattened to form a vector.

The first activations A (1) (X) are a linear transformation of the inputs.

For i ??? {1, . . .

, C(1) }: DISPLAYFORM1 We consider a network with L hidden layers.

The other activations of the network, from A (2) (X) up to A (L+1) (X), are defined recursively: i,j corresponds to applying the filter to the ??th convolutional patch of the channel x j .

DISPLAYFORM2

represents the flattened jth channel of the image that results from applying a convolutional filter to ??(A ( ) (X)).The structure of the pseudo-weight matrices W where it does, as illustrated in Fig. 1 .The outputs of the network are the last activations, A (L+1) (X).

In the classification or regression setting, the outputs are not spatially extended, so we have H (L+1) = D (L+1) = 1, which is equivalent to a fully-connected output layer.

In this case, the pseudo-weights W Finally, we define the prior distribution over functions by making the filters U ( ) i,j and biases b ( ) i be independent Gaussian random variables (RVs).

For each layer , channels i, j and locations within the filter x, y: DISPLAYFORM0 Note that, to keep the activation variance constant, the weight variance is divided by the number of input channels.

The weight variance can also be divided by the number of elements of the filter, which makes it equivalent to the NN weight initialisation scheme introduced by BID10 .

We follow the proofs by BID18 and BID20 to show that the output of the CNN described in the previous section, A (L+1) , defines a GP indexed by the inputs, X. Their proof BID18 proceeds by applying the multivariate Central Limit Theorem (CLT) to each layer in sequence, i.e. taking the limit as N (1) ??? ???, then N (2) ??? ??? etc, where N ( ) is the number of hidden units in layer .

By analogy, we sequentially apply the multivariate CLT by taking the limit as the number of channels goes to infinity, i.e. C(1) ??? ???, then C (2) ??? ??? etc.

While this is the simplest approach to taking the limits, other potentially more realistic approaches also exist BID20 .The fundamental quantity we consider is a vector formed by concatenating the feature maps (or equivalently channels), a ( ) DISPLAYFORM0 This quantity (and the following arguments) can all be extended to the case of countably finite numbers of input points.

Induction base case.

For any pair of data points, X and X the feature-maps corresponding to the jth channel, aj (X, X ) have a multivariate Gaussian joint distribution.

This is because each element is a linear combination of shared Gaussian random variables: the biases, b DISPLAYFORM1 where 1 is a vector of all-ones.

While the elements within a feature map display strong correlations, different feature maps are independent and identically distributed (iid) conditioned on the data (i.e. a( FORMULA1 i (X, X ) and a Induction step.

Consider the feature maps at the th layer, a ( ) j (X, X ), to be iid multivariate Gaussian RVs (i.e. for j = j , a ( ) j (X, X ) and a ( ) j (X, X ) are iid).

Our goal is to show that, taking the number of channels at layer to infinity (i.e. C ( ) ??? ???), the same properties hold at the next layer (i.e. all feature maps, a ( +1) i (X, X ), are iid multivariate Gaussian RVs).

Writing eq. (2) for two training examples, X and X , we obtain, DISPLAYFORM2 We begin by showing that a ( +1) i (X, X ) is a multivariate Gaussian RV.

The first term is multivariate Gaussian, as it is a linear function of b DISPLAYFORM3 , which is itself iid Gaussian.

We can apply the multivariate CLT to show that the second term is also Gaussian, because, in the limit as C ( ) ??? ???, it is the sum of infinitely many iid terms: a ( ) j (X, X ) are iid by assumption, and W ( +1) i,j are iid by definition.

Note that the same argument applies to all feature maps jointly, so all elements of A ( +1) (X, X ) (defined by analogy with eq. 4) are jointly multivariate Gaussian.

Following BID18 , to complete the argument, we need to show that the output feature maps are iid, i.e. a DISPLAYFORM4 To show that they are independent, remember that a ( +1) i (X, X ) and a ( +1) i (X, X ) are jointly Gaussian, so it is sufficient to show that they are uncorrelated, and we can show that they are uncorrelated because the weights, W ( +1) i,j are independent with zero-mean, eliminating any correlations that might arise through the shared RV, ??(a ( ) j (X, X )).

In the appendix, we consider the more complex case where we take limits simultaneously.

Here we derive a computationally efficient kernel corresponding to the CNN described in the previous section.

It is surprising that we can compute the kernel efficiently because the feature maps, Published as a conference paper at ICLR 2019 a ( ) i (X), display rich covariance structure due to the shared convolutional filter.

Computing and representing these covariances would be prohibitively computationally expensive.

However, in many cases we only need the variance of the output, e.g. in the case of classification or regression with a final dense layer.

It turns out that this propagates backwards through the convolutional network, implying that for every layer, we only need the "diagonal covariance" of the activations: the covariance between the corresponding elements of a ( ) DISPLAYFORM0

A GP is completely specified by its mean and covariance (kernel) functions.

These give the parameters of the joint Gaussian distribution of the RVs indexed by any two inputs, X and X .

For the purposes of computing the mean and covariance, it is easiest to consider the network as being written entirely in index notation, DISPLAYFORM0 where and + 1 denote the input and output layers respectively, j and i ??? {1, . . .

, C ( +1) } denote the input and output channels, and ?? and ?? ??? {1, . . .

, H ( +1) D ( +1) } denote the location within the input and output channel or feature-maps.

The mean function is thus easy to compute DISPLAYFORM1 i,j,??,?? have zero mean, and W ( +1) i,j,??,?? are independent of the activations at the previous layer, ??(A ( ) j,?? (X)).

Now we show that it is possible to efficiently compute the covariance function.

This is surprising because for many networks, we need to compute the covariance of activations between all pairs of locations in the feature map (i.e. C A DISPLAYFORM2 and this object is extremely high-dimensional, DISPLAYFORM3 However, it turns out that we only need to consider the "diagonal" covariance, (i.e. we only need C A DISPLAYFORM4 This is true at the output layer (L + 1): in order to achieve an output suitable for classification or regression, we use only a single output location H (L+1) = D (L+1) = 1, with a number of "channels" equal to the number of of outputs/classes, so it is only possible to compute the covariance at that single location.

We now show that, if we only need the covariance at corresponding locations in the outputs, we only need the covariance at corresponding locations in the inputs, and this requirement propagates backwards through the network.

Formally, as the activations are composed of a sum of terms, their covariance is the sum of the covariances of all those underlying terms, DISPLAYFORM5 As the terms in the covariance have mean zero, and as the weights and activations from the previous layer are independent, DISPLAYFORM6 using Eq. (13), or some other nonlinearity.

DISPLAYFORM7 +1) }; using Eq. (11).

6: end for 7: Output the scalar K DISPLAYFORM8 The weights are independent for different channels: DISPLAYFORM9 can eliminate the sums over j and ?? : DISPLAYFORM10 The ??th row of W DISPLAYFORM11 is zero for indices ?? that do not belong to its convolutional patch, so we can restrict the sum over ?? to that region.

We also define v (1) g (X, X ), to emphasise that the covariances are independent of the output channel, j. The variance of the first layer is DISPLAYFORM12 And we do the same for the other layers, DISPLAYFORM13 where DISPLAYFORM14 is the covariance of the activations, which is again independent of the channel.

The elementwise covariance in the right-hand side of Eq. (11) can be computed in closed form for many choices of ?? if the activations are Gaussian.

For each element of the activations, one needs to keep track of the 3 distinct entries of the bivariate covariance matrix between the inputs, K DISPLAYFORM0 For example, for the ReLU nonlinearity (??(x) = max(0, x)), one can adapt BID5 in the same way as Matthews et al. (2018a, section 3 ) to obtain DISPLAYFORM1 where ?? ( ) DISPLAYFORM2

We now have all the pieces for computing the kernel, as written in Algorithm 1.Putting together Eq. (11) and Eq. FORMULA1 gives us the surprising result that the diagonal covariances of the activations at layer + 1 only depend on the diagonal covariances of the activations at layer .

This is very important, because it makes the computational cost of the kernel be within a constant factor of the cost of a forward pass for the equivalent CNN with 1 filter per layer.

Thus, the algorithm is more efficient that one would naively think.

A priori, one needs to compute the covariance between all the elements of a ( ) Furthermore, the particular form for the kernel (eq. 1 and eq. 2) implies that the required variances and covariances at all required locations can be computed efficiently as a convolution.

DISPLAYFORM0

The induction step in the argument for GP behaviour from Sec. 2.2 depends only on the previous activations being iid Gaussian.

Since all the activations are iid Gaussian, we can add skip connections between the activations of different layers while preserving GP behaviour, e.g. A ( +1) and A DISPLAYFORM0 where s is the number of layers that the skip connection spans.

If we change the NN recursion (Eq. 2) to DISPLAYFORM1 then the kernel recursion (Eq. 11) becomes DISPLAYFORM2 This way of adding skip connections is equivalent to the "pre-activation" shortcuts described by BID11 .

Remarkably, the natural way of adding residual connections to NNs is the one that performed best in their empirical evaluations.

We evaluate our kernel on the MNIST handwritten digit classification task.

Classification likelihoods are not conjugate for GPs, so we must make an approximation, and we follow BID18 , in re-framing classification as multi-output regression.

The training set is split into N = 50000 training and 10000 validation examples.

The regression targets Y ??? {???1, 1} N ??10 are a one-hot encoding of the example's class: y n,c = 1 if the nth example belongs to class c, and ???1 otherwise.

Training is exact conjugate likelihood GP regression with noiseless targets Y BID24 .

First we compute the N ??N kernel matrix K xx , which contains the kernel between every pair of examples.

Then we compute K ???1 xx Y using a linear system solver.

The test set has N T = 10000 examples.

We compute the N T ?? N matrix K x * x , the kernel between each test example and all the training examples.

The predictions are given by the row-wise maximum of K x * x K ???1 xx Y. For the "ConvNet GP" and "Residual CNN GP", (Table 1) we optimise the kernel hyperparameters by random search.

We draw M random hyperparameter samples, compute the resulting kernel's performance in the validation set, and pick the highest performing run.

The kernel hyperparameters are: ?? 2 b , ?? 2 w ; the number of layers; the convolution stride, filter sizes and edge behaviour; the nonlinearity (we consider the error function and ReLU); and the frequency of residual skip connections (for Residual CNN GPs).

We do not retrain the model on the validation set after choosing hyperparameters.

Table 1 : MNIST classification results.

#samples gives the number of kernels that were randomly sampled for the hyperparameter search.

"ConvNet GP" and "Residual CNN GP" are random CNN architectures with a fixed filter size, whereas "ResNet GP" is a slight modification of the architecture by BID11 .

Entries labelled "SGD" used stochastic gradient descent for tuning hyperparameters, by maximising the likelihood of the training set.

The last two methods use parametric neural networks.

The hyperparameters of the ResNet GP were not optimised (they were fixed based on the architecture from He et al., 2016b).The "ResNet GP" (Table 1) is the kernel equivalent to a 32-layer version of the basic residual architecture by BID10 .

The differences are: an initial 3 ?? 3 convolutional layer and a final dense layer instead of average pooling.

We chose to remove the pooling because computing its output variance requires the off-diagonal elements of the filter covariance, in which case we could not exploit the efficiency gains described in Sec. 3.3.We found that the despite it not being optimised, the 32-layer ResNet GP outperformed all other comparable architectures (Table 1) , including the NNGP in BID18 , which is state-ofthe-art for non-convolutional networks, and convolutional GPs (van der Wilk et al., 2017; BID14 .

That said, our results have not reached state-of-the-art for methods that incorporate a parametric neural network, such as a standard ResNet (Chen et al., 2018) and a Gaussian process with a deep neural network kernel BID3 .To check whether the GP limit is applicable to relatively small networks used practically (with of the order of 100 channels in the first layers), we randomly sampled 10, 000 32-layer ResNets, with 3, 10, 30 and 100 channels in the first layers, and, following the usual practice for ResNets we increase the number the number of hidden units when we downsample the feature maps.

The probability density plots show a good match around 100 channels ( FIG6 , which matches a more sensitive graphical procedure based on quantile-quantile plots FIG6 .

Notably, even for only 30 channels, the moments match closely FIG6 ).

For comparison, typical ResNets use from 64 BID10 ) to 192 (Zagoruyko & Komodakis, 2016 ) channels in their first layers.

We believe that this is because the moment propagation equations only require the Gaussianity assumption for propagation through the relu, and presumably this is robust to non-Gaussian input activations.

Computational efficiency.

Asymptotically, computing the kernel matrix takes O(N 2 LD) time, where L is the number of layers in the network and D is the dimensionality of the input, and inverting the kernel matrix takes O(N 3 ).

As such, we expect that for very large datasets, inverting the kernel matrix will dominate the computation time.

However, on MNIST, N 3 is only around a factor of 10 larger than N 2 LD.

In practice, we found that it was more expensive to compute the kernel matrix than to invert it.

For the ResNet kernel, the most expensive, computing K xx , and K xx * for validation and test took 3h 40min on two Tesla P100 GPUs.

In contrast, inverting K xx and computing validation and test performance took 43.25 ?? 8.8 seconds on a single Tesla P100 GPU.

Van der Wilk et al. BID22 ) also adapted GPs to image classification.

They defined a prior on functions f that takes an image and outputs a scalar.

First, draw a function g ??? GP(0, k p (X, X )).

Then, f is the sum of the output of g applied to each of the convolutional patches.

Their approach is also inspired by convolutional NNs, but their kernel k p is applied to all pairs of patches of X and X .

This makes their convolutional kernel expensive to evaluate, requiring , and 100 channels in their first layers.

A Comparison of the empirical and limiting probability densities.

B A more sensitive test of Gaussianity is a quantile-quantile plot, which shows converges with 100 channels.

C The moments (variances and covariances) for 100 training inputs shows gives a good match for all numbers of channels.inter-domain inducing point approximations to remain tractable.

The kernels in this work, directly motivated by the infinite-filter limit of a CNN, only apply something like k p to the corresponding pairs of patches within X and X (Eq. 10).

As such, the CNN kernels are cheaper to compute and exhibit superior performance (Table 1) , despite the use of an approximate likelihood function.

BID14 define a prior over functions by stacking several GPs with van der Wilk's convolutional kernel, forming a "Deep GP" BID6 .

In contrast, the kernel in this paper confines all hierarchy to the definition of the kernel, and the resulting GPs is shallow.

BID3 improved deep kernel learning.

The inputs to a classic GP kernel k (e.g. RBF) are preprocessed by applying a feature extractor g (a deep NN) prior to computing the kernel: k deep (X, X ) := k(g(X; ??), g(X , ??)).

The NN parameters are optimised by gradient ascent using the likelihood as the objective, as in standard GP kernel learning (Rasmussen & Williams, 2006, Chapter 5) .

Since deep kernel learning incorporates a state-of-the-art NN with over 10 6 parameters, we expect it to perform similarly to a NN applied directly to the task of image classification.

At present both CNNs and deep kernel learning display superior performance to the GP kernels in this work.

However, the kernels defined here have far fewer parameters (around 10, compared to their 10 6 ).

Borovykh (2018) also suggests that a CNN exhibits GP behaviour.

However, they take the infinite limit with respect to the filter size, not the number of filters.

Thus, their infinite network is inapplicable to real data which is always of finite dimension.

Finally, there is a series of papers analysing the mean-field behaviour of deep NNs and CNNs which aims to find good random initializations, i.e. those that do not exhibit vanishing or exploding gradients or activations BID25 Yang & Schoenholz, 2017) .

Apart from their very different focus, the key difference to our work is that they compute the variance for a single training-example, whereas to obtain the GPs kernel, we additionally need to compute the output covariances for different training/test examples (Xiao et al., 2018) .

We have shown that deep Bayesian CNNs with infinitely many filters are equivalent to a GP with a recursive kernel.

We also derived the kernel for the GP equivalent to a CNN, and showed that, in handwritten digit classification, it outperforms all previous GP approaches that do not incorporate a parametric NN into the kernel.

Given that most state-of-the-art neural networks incorporate structure (convolutional or otherwise) into their architecture, the equivalence between CNNs and GPs is potentially of considerable practical relevance.

In particular, we hope to apply GP CNNs in domains as widespread as adversarial examples, lifelong learning and k-shot learning, and we hope to improve them by developing efficient multi-layered inducing point approximation schemes.

The key technical issues in the proof (and the key differences between BID18 BID21 arise from exactly how and where we take limits.

In particular, consider the activations as being functions of the activities at the previous layer, DISPLAYFORM0 Now, there are two approaches to taking limits.

First, both our argument in the main text, and the argument in BID18 is valid if we are able to take limits "inside" the network, DISPLAYFORM1 However, BID20 b) argue that is preferable to take limits "outside" the network.

In particular, BID21 take the limit with all layers simultaneously, DISPLAYFORM2 where C ( ) = C ( ) (n) goes to infinity as n ??? ???. That said, similar technical issues arise if we take limits in sequence, but outside the network.

In the main text, we follow BID18 in sequentially taking the limit of each layer to infinity (i.e. C(1) ??? ???, then C (2) ??? ??? etc.).

This dramatically simplified the argument, because taking the number of units in the previous layer to infinity means that the inputs from that layer are exactly Gaussian distributed.

However, BID21 argue that the more practically relevant limit is where we take all layers to infinity simultaneously.

This raises considerable additional difficulties, because we must reason about convergence in the case where the previous layer is finite.

Note that this section is not intended to stand independently: it is intended to be read alongside BID21 , and we use several of their results without proof.

Mirroring Definition 3 in BID21 , we begin by choosing a set of "width" functions, C ( ) (n), for ??? {1, . . .

, L} which all approach infinity as n ??? ???. In BID21 , these functions described the number of hidden units in each layer, whereas here they describe the number of channels.

Our goal is then to extend the proofs in BID21 (in particular, of theorem 4), to show that the output of our convolutional networks converge in distribution to a Gaussian process as n ??? ???, with mean zero and covariance given by the recursion in Eqs. (10 -12).The proof in BID21 has three main steps.

First, they use the Cram??r-Wold device, to reduce the full problem to that of proving convergence of scalar random variables to a Gaussian with specified variance.

Second, if the previous layers have finite numbers of channels, then the channels a ( ) j (X) and a ( ) j (X ) are uncorrelated but no longer independent, so we cannot apply the CLT directly, as we did in the main text.

Instead, they write the activations as a sum of exchangeable random variables, and derive an adapted CLT for exchangeable (rather than independent) random variables BID0 .

Third, they show that moment conditions required by their exchangeable CLT are satisfied.

To extend their proofs to the convolutional case, we begin by defining our networks in a form that is easier to manipulate and as close as possible to Eq. (21-23) in BID21 , DISPLAYFORM0 DISPLAYFORM1 where, DISPLAYFORM2 The first step is to use the Cram??r-Wold device (Lemma 6 in BID21 , which indicates that convergence in distribution of a sequence of finite-dimensional vectors is equivalent to convergence on all possible linear projections to the corresponding real-valued random variable.

Mirroring Eq. 25 in BID21 , we consider convergence of random vectors, f DISPLAYFORM3 where L ??? X ?? N ?? {1, . . .

, H ( ) D ( ) } is a finite set of tuples of data points and channel indicies, i, and indicies of elements within channels/feature maps, ??. The suffix [n] indicates width functions that are instantiated with input, n. Now, we must prove that these projections converge in distribution a Gaussian.

We begin by defining summands, as in Eq. 26 in BID21 , DISPLAYFORM4 such that the projections can be written as a sum of the summands, exactly as in Eq. 27 in BID21 , DISPLAYFORM5 Now we can apply the exchangeable CLT to prove that T ( ) (L, ??) [n] converges to the limiting Gaussian implied by the recursions in the main text.

To apply the exchangeable CLT, the first step is to mirror Lemma 8 in BID21 , in showing that for each fixed n and ??? {2, . . .

, L + 1}, the summands, ?? ( ) j (L, ??) [n] are exchangeable with respect to the index j.

In particular, we apply de Finetti's theorem, which states that a sequence of random variables is exchangeable if and only if they are i.i.d.

conditional on some set of random variables, so it is sufficient to exhibit such a set of random variables.

Mirroring Eq. 29 in BID21 , we apply the recursion, k,?? (x)[n] : k ??? {1, . . .

, C ( ???2) }, ?? ??? {1, . . .

, H ( ???2) D ( ???2) }, x ??? L X , where L X is the set of input points in L.The exchangeable CLT in Lemma 10 in BID21 indicates that T ( ) (L, ??) [n] converges in distribution to N 0, ?? 2 * if the summands are exchangeable (which we showed above), and if three conditions hold, DISPLAYFORM6 Condition a) follows immediately as the summands are uncorrelated and zero-mean.

Conditions b) and c) are more involved as convergence in distribution in the previous layers does not imply convergence in moments for our activation functions.

We begin by considering the extension of Lemma 20 in BID21 , which allow us to show conditions b) and c) above, even in the case of unbounded but linearly enveloped nonlinearities (Definition 1 in BID21 .

Lemma 20 states that the eighth moments of f (t)i,?? (x)[n] are bounded by a finite constant independent of n ??? N. We prove this by induction.

The base case is trivial, as f where g ( ???1) j???{1,...,C ( ???1) (n)},???????th patch (x)[n] is the set of post-nonlinearities corresponding to j ??? {1, . . .

, C ( ???1) (n)} and ?? ??? ??th patch.

Following BID21 , observe that, .

The x-axis gives GP prediction for the label probability.

The points give corresponding proportion of test points with that label, and the bars give the proportion of training examples in each bin.

DISPLAYFORM7

@highlight

We show that CNNs and ResNets with appropriate priors on the parameters are Gaussian processes in the limit of infinitely many convolutional filters.