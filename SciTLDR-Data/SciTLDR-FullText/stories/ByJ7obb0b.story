Training methods for deep networks are primarily variants on stochastic gradient descent.

Techniques that use (approximate) second-order information are rarely used because of the computational cost and noise associated with those approaches in deep learning contexts.

However, in this paper, we show how feedforward deep networks exhibit a low-rank derivative structure.

This low-rank structure makes it possible to use second-order information without needing approximations and without incurring a significantly greater computational cost than gradient descent.

To demonstrate this capability, we implement Cubic Regularization (CR) on a feedforward deep network with stochastic gradient descent and two of its variants.

There, we use CR to calculate learning rates on a per-iteration basis while training on the MNIST and CIFAR-10 datasets.

CR proved particularly successful in escaping plateau regions of the objective function.

We also found that this approach requires less problem-specific information (e.g. an optimal initial learning rate) than other first-order methods in order to perform well.

Gradient-based optimization methods use derivative information to determine intelligent search directions when minimizing a continuous objective function.

The steepest descent method is the most basic of these optimization techniques, but it is known to converge very slowly in ill-conditioned systems.

Even outside of these cases, it still only has a linear rate of convergence.

Newton's method is a more sophisticated approach -one that uses second-order derivative information, which allows the optimizer to model the error surface more accurately and thus take more efficient update steps.

When it converges, it does so quadratically, but Newton's method also has limitations of its own.

Firstly, it does not scale well: it can be very expensive to calculate, store, and invert the objective function Hessian.

Secondly, the method may fail if the Hessian is indefinite or singular.

A variety of methods have been developed to try and appropriate the strengths of each approach while avoiding their weaknesses.

The conjugate gradient method, for example, uses only firstorder information but uses the history of past steps taken to produce a better convergence rate than steepest descent.

Quasi-Newton methods, on the other hand, approximate the Hessian (or its inverse) using first-order information and may enforce positive-definiteness on its approximation.

Other approaches like trust region methods use second-order information without requiring convexity.

For further information about gradient-based optimization, see BID15 .Deep learning (DL) provides a set of problems that can be tackled with gradient-based optimization methods, but it has a number of unique features and challenges.

Firstly, DL problems can be extremely large, and storing the Hessian, or even a full matrix approximation thereto, is not feasible for such problems.

Secondly, DL problems are often highly nonconvex.

Thirdly, training deep networks via mini-batch sampling results in a stochastic optimization problem.

Even if the necessary expectations can be calculated (in an unbiased way), the variance associated with the batch sample calculations produces noise, and this noise can make it more difficult to perform the optimization.

Finally, deep networks consist of the composition of analytic functions whose forms are known.

As such, we can calculate derivative information analytically via back-propagation (i.e. the chain rule).

These special characteristics of DL have motivated researchers to develop training methods specifically designed to overcome the challenges with training a deep neural network.

One such approach is layer-wise pretraining BID1 , where pretraining a neural network layer-by-layer encourages the weights to initialize close to a optimal minimum.

Transfer learning (Yosinski et al., 2014) works by a similar mechanism, relying on knowledge gained through previous tasks to encourage nice training on a novel task.

Outside of pretraining, a class of optimization algorithms have been specifically designed for training deep networks.

The Adam, Adagrad, and Adamax set of algorithms provide examples of using history-dependent learning rate adjustment BID6 .

Similarly, Nesterov momentum provides a method for leveraging history dependence in stochastic gradient descent BID17 .

One could possibly argue that these methods implicitly leverage second order information via their history dependence, but the stochastic nature of mini-batching prevents this from becoming explicit.

Some researchers have sought to use second-order information explicitly to improve the training process.

Most of these methods have used an approximation to the Hessian.

For example, the L-BFGS method can estimate the Hessian (or its inverse) in a way that is feasible with respect to memory requirements; however, the noise associated with the sampling techniques can either overwhelm the estimation or require special modifications to the L-BFGS method to prevent it from diverging BID3 .

There have been two primary ways to deal with this: subsampling BID3 BID13 and mini-batch reuse BID16 BID12 .

Subsampling involves updating the Hessian approximation every L iterations rather than every iteration, as would normally be done.

Mini-batch reuse consists of using the same minibatch on subsequent iterations when calculating the difference in gradients between those two iterations.

These approximate second-order methods typically have a computational cost that is higher than, though on the same order of, gradient descent, and that cost can be further reduced by using a smaller mini-batch for the Hessian approximation calculations than for the gradient calculation BID2 .

There is also the question of bias: it is possible to produce unbiased low-rank Hessian approximations BID11 , but if the Hessian is indefinite, then quasi-Newton methods will prefer biased estimates -ones that are positive definite.

Other work has foregone these kinds of Hessian approximations in favor of using finite differences BID10 ).

In this paper, we prove, by construction, that the first and second derivatives of feedforward deep learning networks exhibit a low-rank, outer product structure.

This structure allows us to use and manipulate second-order derivative information, without requiring approximation, in a computationally feasible way.

As an application of this low-rank structure, we implement Cubic Regularization (CR) to exploit Hessian information in calculating learning rates while training a feedforward deep network.

Finally, we show that calculating learning rates in this fashion can improve existing training methods' ability to exit plateau regions during the training process.

Second-order derivatives are not widely used in DL, and where they are used, they are typically estimated.

These derivatives can be calculated analytically, but this is not often done because of the scalability constraints described in Section 1.1.

If we write out the first and second derivatives, though, we can see that they have a low-rank structure to them -an outer product structure, in fact.

When a matrix has low rank (or less than full rank), it means that the information contained in that matrix (or the operations performed by that matrix) can be fully represented without needing to know every entry of that matrix.

An outer product structure is a special case of this, where an mxn matrix A can be fully represented by two vectors A = uv T .

We can then calculate, store, and use secondorder derivatives exactly in an efficient manner by only dealing with the components needed to represent the full Hessians rather than dealing with those Hessians themselves.

Doing this involves some extra calculations, but the storage costs are comparable to those of gradient calculations.

In this section, we will illustrate the low-rank structure for a feedforward network, of arbitrary depth and layer widths, consisting of ReLUs in the hidden layers and a softmax at the output layer.

A feedforward network with arbitrary activation functions has somewhat more complicated derivative formulae, but those derivatives still exhibit a low-rank structure.

That structure also does not depend on the form of the objective function or whether a softmax is used, and it is present for convolutional and recurrent layers as well.

The complete derivations for these cases are given in Appendix B.In our calculations, we make extensive use of index notation with the summation convention BID5 .

In index notation, a scalar has no indices (v), a vector has one index (v as v i or v i ), a matrix has two (V as V ij , V i j , or V ij ), and so on.

The summation convention holds that repeated indices in a given expression are summed over unless otherwise indicated.

For example, DISPLAYFORM0 The pair of indices being summed over will often consist of a superscript and a subscript; this is a bookkeeping technique used in differential geometry, but in this context, the subscripting or superscripting of indices will not indicate covariance or contravariance.

We have also adapted index notation slightly to suit the structure of deep networks better: indices placed in brackets (e.g. the k in v (k),j ) are not summed over, even if repeated, unless explicitly indicated by a summation sign.

A tensor convention that we will use, however, is the Kronecker delta: ?? ij , ?? i j , or ?? ij .

The Kronecker delta is the identity matrix represented in index notation: it is 1 for i = j and 0 otherwise.

The summation convention can sometimes be employed to simplify expressions containing Kronecker deltas.

For example, ?? DISPLAYFORM1 Let us consider a generic feedforward network with ReLU activation functions in n hidden layers, a softmax at the output layer, and categorical cross-entropy as the objective function (defined in more detail in Appendix B. The first derivatives, on a per-sample basis, for this deep network are DISPLAYFORM2 where f is the per-sample objective function, v (k),j is the vector output of layer k, u , see Appendix B. In calculating these expressions, we have deliberately left ???f ???p j unevaluated.

This keeps the expression relatively simple, and programs like TensorFlow BID0 can easily calculate this for us.

Leaving it in this form also preserves the generality of the expression -there is no low-rank structure contained in ???f ???p j , and the low-rank structure of the network as a whole is therefore shown to be independent of the objective function and whether or not a softmax is used.

In fact, as long as Equation 13 holds, any sufficiently smooth function of p j may be used in place of a softmax without disrupting the low-rank structure.

The one quantity that needs to be stored here is ?? (n,k),j i for k = 1, 2, . . .

, n ??? 1; it will be needed in the second derivative calculations.

Note, however, that this is roughly the same size as the gradient itself.

We can now see the low-rank structure: DISPLAYFORM3 is the outer product (or tensor product) of the vectors ???f ???p i and v (n),j , and DISPLAYFORM4 (which ends up being a rank-1 tensor) and v (k???1),j .

The index notation makes the outer product structure clear.

It is important to note that this low-rank structure only exists for each sample -a weighted sum of low-rank matrices is not necessarily (and generally, will not be) low rank.

In other words, even if the gradient of f is low rank, the gradient of the expectation, F = E [f ], will not be, because the gradient of F is the weighted sum of the gradients of f .

The second-order objective function derivatives are then DISPLAYFORM5 DISPLAYFORM6 Calculating all of these second derivatives requires the repeated use of DISPLAYFORM7 Evaluating that Hessian is straightforward given knowledge of the activation functions and objective used in the network, and storing it is also likely not an issue as long as the number of categories is small relative to the number of weights.

For example, consider a small network with 10 categories and 1000 weights.

In such a case, DISPLAYFORM8 ???p 2 would only contain 100 entries -the gradient would be 10 times larger.

We now find that we have to store ?? (n,k),i j values in order to calculate the derivatives.

In DISPLAYFORM9 ???w 2 , we also end up needing ?? (r,k),i j for r = n. In a network with n hidden layers, we would then have DISPLAYFORM10 matrices to store.

For n = 10, this would be 45, for n = 20, this would be 190, and so on.

This aspect of the calculations does not seem to scale well, but in practice, it is relatively simple to work around.

It is still necessary to store ?? DISPLAYFORM11 , r < n, only actually shows up in one place, and thus it is possible to calculate each ?? (r,k),i j matrix, use it, and discard it without needing to store it for future calculations.

The key thing to note about these second derivatives is that they retain a low-rank structure -they are now tensor products (or the sums of tensor products) of matrices and vectors.

For example, DISPLAYFORM12 With these expressions, it would be relatively straightforward to extract the diagonal of the Hessian and store or manipulate it as a vector.

The rank of the weighted sum of low rank components (as occurs with mini-batch sampling) is generally larger than the rank of the summed components, however.

As such, manipulating the entire Hessian may not be as computationally feasible; this will depend on how large the mini-batch size is relative to the number of weights.

The low rank properties that we highlight here for the Hessian exist on a per-sample basis, as they did for the gradient, and therefore, the computational savings provided by this approach will be most salient when calculating scalar or vector quantities on a sample-by-sample basis and then taking a weighted sum of the results.

In principle, we could calculate third derivatives, but the formulae would likely become unwieldy, and they may require memory usage significantly greater than that involved in storing gradient information.

Second derivatives should suffice for now, but of course if a use arose for third derivatives, calculating them would be a real option.

Thus far, we have not included bias terms.

Including bias terms as trainable weights would increase the overall size of the gradient (by adding additional variables), but it would not change the overall low-rank structure.

Using the calculations provided in Appendix B, it would not be difficult to produce the appropriate derivations.

Cubic Regularization (CR) is a trust region method that uses a cubic model of the objective function: DISPLAYFORM0 at the j-th iteration, where H j is the objective function Hessian and s j = x ??? x j .

The cubic term makes it possible to use information in the Hessian without requiring convexity, and the weight ?? j on that cubic term can have its own update scheme (based on how well m (s j ) approximates f BID7 ).

Solving for an optimal s j value then involves finding the root of a univariate nonlinear equation BID14 .

CR is not commonly used in deep learning; we have seen only one example of CR applied to machine learning BID7 and no examples with deep learning.

This is likely the case because of two computationally expensive operations: calculating the Hessian and solving for s j .

We can overcome the first by using the lowrank properties described above.

The second is more challenging, but we can bypass it by using CR to calculate a step length (i.e. the learning rate) for a given search direction rather than calculating the search direction itself.

Our approach in this paper is to use CR as a metamethod -a technique that sits on top of existing training algorithms.

The algorithm calculates a search direction, and then CR calculates a learning rate for that search direction.

For a general iterative optimization process, this would look like x j+1 = x j + ?? j g j , where g j is the search direction (which need not be normalized), ?? j is the learning rate, and the subscript refers to the iteration.

With the search direction fixed, m would then be a cubic function of ?? at each iteration.

Solving ???m ????? = 0 as a quadratic equation in ?? then yields DISPLAYFORM0 If we assume that g T ???f < 0 (i.e. g is a descent direction), then ?? is guaranteed to be real.

Continuing under that assumption, of the two possible ?? values, we choose the one guaranteed to be positive.

The sampling involved in mini-batch training means that there are a number of possible ways to get a final ?? j g j result.

One option would be to calculate E [?? j g j ].

This would involve calculating an ?? value with respect to the search direction produced by each sample point and then averaging the product ??g over all of the sample points.

Doing this should produce an unbiased estimate of ?? j g j , but in practice, we found that this approach resulted in a great deal sampling noise and thus was not effective.

The second approach would calculate DISPLAYFORM1 To do this, we would calculate an ?? value with respect to the search direction produced by each sample point, as in the first option, calculate an average ?? value, and multiply the overall search direction by that average.

This approach, too, suffered from excessive noise.

In the interest of reducing noise and increasing simplicity, we chose a third option: once the step direction had been determined, we considered that fixed, took the average of g T Hg and g T ???f over all of the sample points to produce m (??) and then solved for a single ?? j value.

This approach was the most effective of the three.

To test CR computationally, we created deep feedforward networks using ReLU activations in the hidden layers, softmax in the output layer, and categorical cross-entropy as the error function; we then trained them on the MNIST (LeCun et al., 1998) and CIFAR-10 ( BID8 data sets.

This paper shows results from networks with 12 hidden layers, each 128 nodes wide.

For the purposes of this paper, we treat network training strictly as an optimization process, and thus we are not interested in network performance measures such as accuracy and validation error -the sole consideration is minimizing the error function presented to the network.

As we consider that minimization progress, we will also focus on optimization iteration rather than wall clock time: the former indicates the behaviour of the algorithm itself, whereas the latter is strongly dependent upon implementation (which we do not want to address at this juncture).

Overall computational cost per iteration matters, and we will discuss it, but it will not be our primary interest.

Further implementation details are found in Appendix A. Figure 1 shows an example of CR (applied on top of SGD).

In this case, using CR provided little to no benefit.

The average learning rate with CR was around 0.05 (a moving average with a period of 100 is shown in green on the learning rate plot both here and in the rest of the paper), which was close to our initial choice of learning rate.

This suggests that 0.02 was a good choice of learning rate.

Another reason the results were similar, though, is that the optimization process did not run into any plateaus.

We would expect CR to provide the greatest benefit when the optimization gets stuck on a plateau -having information about the objective function curvature would enable the algorithm to increase the learning rate while on the plateau and then return it to a more typical value once it leaves the plateau.

To test this, we deliberately initialized our weights so that they lay on a plateau: the objective function is very flat near the origin, and we found that setting the network weights to random values uniformly sampled between 0.1 and -0.1 was sufficient.(a) Error (SGD in blue, SGD with CR in red) (b) Learning rate (calculated rate in red, period-100 moving average in green)Figure 3: Cubic Regularization (CR) applied to Stochastic Gradient Descent (SGD) on the CIFAR-10 Dataset; initial learning rate = 0.01, ?? = 1000 Figure 2 shows the results of SGD with and without CR when stuck on a plateau.

There, we see a hundred-fold increase in the learning rate while the optimization is on the plateau, but this rate drops rapidly as the optimization exits the plateau, and once it returns to a more normal descent, the learning rate also returns to an average of about 0.05 as before.

The CR calculation enables the training process to recognize the flat space and take significantly larger steps as a result.

Applying CR to SGD when training on CIFAR-10 ( Figure 3 ) produced results similar to those seen on MNIST.We then considered if this behaviour would hold true on other training algorithms: we employed CR with Adagrad BID4 and Adadelta (Zeiler, 2012)on MNIST.

The results were similar.

CR did not provide a meaningful difference when the algorithms performed well, but when those algorithms were stuck on plateaus, CR increased the learning rate and caused the algorithms to exit the plateau more quickly than they otherwise would have (Figures 4 and 5) .

The relative magnitudes of those increases were smaller than for SGD, but Adagrad and Adadelta already incorporate some adaptive learning rate behaviour, and good choices for the initial learning rate varied significantly from algorithm to algorithm.

We also used a larger value for ?? to account for the increased variability due to those algorithms' adaptive nature.

The result with Adadelta showed some interesting learning rate changes: the learning rate calculated by CR dropped steadily as the algorithm exited the plateau, but it jumped again around iteration 1200 as it apparently found itself in a flat region of space.

We see this CR approach as an addition to, not a replacement for, existing training methods.

It could potentially replace existing methods, but it does not have to in order to be used.

Because of the low-rank structure of the Hessian, we can use CR to supplement existing optimizers that do not explicitly leverage second order information.

The CR technique used here is most useful when the optimization is stuck on a plateau prior to convergence: CR makes it possible to determine whether the optimization has converged (perhaps to a local minimum) or is simply bogged down in a flat region.

It may eventually be possible to calculate a search direction as well as a step length, which would likely be a significant advancement, but this would be a completely separate algorithm.

We found that applying CR to Adagrad and Adadelta provided the same kinds of improvements that applying CR to SGD did.

However, using CR with Adam BID6 did not provide gains as it did with the other methods.

Adam generally demonstrates a greater degree of adaptivity than Adagrad or Adadelta; in our experiments, we found that Adam was better than Adagrad or Adadelta in escaping the plateau region.

We suspect that trying to overlay an additional calculated learning rate on top of the variable-specific learning rate produced by Adam may create interference in both sets of learning rate calculations.

Analyzing each algorithm's update scheme in conjunction with the CR calculations could provide insight into the nature and extent of this interference, and provide ways to further improve both algorithms.

In future work, though, it would not be difficult to adapt the CR approach to calculate layer-or variable-specific learning rates, and doing that could address this problem.

Calculating a variable-specific learning rate would essentially involve rescaling each variable's step by the corresponding diagonal entry in the Hessian; calculating a layer-specific learning rate would involve rescaling the step of each variable in that layer by some measure of the block diagonal component of the Hessian corresponding to those variables.

The calculations for variable-specific learning rates with CR are given in Appendix B.There are two aspects of the computational cost to consider in evaluating the use of CR.

The first aspect is storage cost.

In this regard, the second-order calculations are relatively inexpensive (comparable to storing gradient information).

The second aspect is the number of operations, and the second-order calculations circumvent the storage issue by increasing the number of operations.

The number of matrix multiplications involved in calculating the components of Equation 9, for example, scales quadratically with the number of layers (see the derivations in Appendix B).

Although the number of matrix multiplications will not change with an increase in width, the cost of na??ve matrix multiplication scales cubically with matrix size.

That being said, these calculations are parallelizable and as such, the effect of the computation cost will be implementation-dependent.

A significant distinction between CR and methods like SGD has to do with the degree of knowledge about the problem required prior to optimization.

SGD requires an initial learning rate and (usually) a learning rate decay scheme; an optimal value for the former can be very problem-dependent and may be different for other algorithms when applied to the same problem.

For CR, it is necessary to specify ??, but optimization performance is relatively insensitive to this -order of magnitude estimates seem to be sufficient -and varying ?? has a stronger affect on the variability of the learning rate than it does on the magnitude (though it does affect both).

If the space is very curved, the choice of ?? matters little because the step size determination is dominated by the curvature, and if the space if flat, it bounds the step length.

It is also possible to employ an adaptive approach for updating ?? BID7 ), but we did not pursue that here.

Essentially, using CR is roughly equivalent to using the optimal learning rate (for SGD).

In this paper, we showed that feedforward networks exhibit a low-rank derivative structure.

We demonstrate that this structure provides a way to represent the Hessian efficiently; we can exploit this structure to obtain higher-order derivative information at relatively low computational cost and without massive storage requirements.

We then used second-order derivative information to implement CR in calculating a learning rate when supplied with a search direction.

The CR method has a higher per-iteration cost than SGD, for example, but it is also highly parallelizable.

When SGD converged well, CR showed comparable optimization performance (on a per-iteration basis), but the adaptive learning rate that CR provided proved to be capable of driving the optimization away from plateaus that SGD would stagnate on.

The results were similar with Adagrad and Adadelta, though not with Adam.

CR also required less problem-specific knowledge (such as an optimal initial learning rate) to perform well.

At this point, we see it as a valuable technique that can be incorporated into existing methods, but there is room for further work on exploiting the low-rank derivative structure to enable CR to calculate search directions as well as step sizes.

Starting at a point far from the origin resulted in extremely large derivative and curvature values (not to mention extremely large objective function values), and this could sometimes cause difficulties for the CR method.

This was easy to solve by choosing an initialization point relatively near the origin; choosing an initialization relatively near the origin also provided a significantly better initial objective function value.

We initialized the networks' weights to random values between an upper and lower bound: to induce plateau effects, we set, the bounds to ??0.1, otherwise, we set them to ??0.2.All of the networks used a mini-batch size of 32 and were implemented in TensorFlow BID0 .

The initial learning rate varied with network size; we chose learning rates that were large and reasonable but perhaps not optimal, and for optimization algorithms with other parameters governing the optimization, we used the default TensorFlow values for those parameters.

For the learning rate decay, we used an exponential decay with a decay rate of 0.95 per 100 iterations.

The ?? value used is specified along with the initial learning rate for each network's results.

This value was also not optimized but was instead set to a reasonable power of 10.B LOW-RANK DERIVATIONS FOR DEEP NETWORKS B.1 FEEDFORWARD NETWORK WITH RELU ACTIVATIONS TAB1 provides a nomenclature for our deep network definition.

Equations 10-16 define a generic feedforward network with ReLU activation functions in the hidden layers, n hidden layers, a softmax at the output layer, and categorical cross-entropy as the objective function.

DISPLAYFORM0 A (z) = max (z, 0) (11) DISPLAYFORM1 DISPLAYFORM2 The relevant first derivatives for this deep network are DISPLAYFORM3 where there is no summation over j in Equation FORMULA13 .

We now define several intermediate quantities to simplify the derivation process: DISPLAYFORM4 DISPLAYFORM5 where there is no summation over j in Equations 19 and 20.

We can now complete our calculations of the first derivatives.

DISPLAYFORM6 DISPLAYFORM7 We then start our second derivative calculations by considering some intermediate quantities: DISPLAYFORM8

Convolutional and recurrent layers preserve the low-rank derivative structure of the fully connected feedforward layers considered above, and we will show this in the following sections.

Because we are only considering a single layer of each, we calculate the derivatives of the layer outputs with respect to the layer inputs -in a larger network, those derivatives will be necessary for calculating total derivatives via back-propagation.

We can define a convolutional layer as DISPLAYFORM0 where x i j is the layer input, ?? is the vertical stride, ?? is the horizontal stride, A is the activation function, and v s t is the layer output.

A convolutional structure can make the expressions somewhat complicated when expressed in index notation, but we can simplify matters by using the simplification z with no summation over s and t in any of the expressions above.

Using the simplification with z sl tk makes it significantly easier to see the low rank structure in these derivatives, but that structure is still noticeable without the simplification.

The conditional form of the expressions is more complicated, but it is also possible to see how the derivatives relate to w DISPLAYFORM1 where t indicates the number of times that the recursion has been looped through.

If we inspect this carefully, we can actually see that this is almost identical to the hidden layers of the feedforward network: they are identical if we stipulate that the weights of the feedforward network are identical at each layer (i.e. w

@highlight

We show that deep learning network derivatives have a low-rank structure, and this structure allows us to use second-order derivative information to calculate learning rates adaptively and in a computationally feasible manner.