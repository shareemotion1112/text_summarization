Interpolation of data in deep neural networks has become a subject of significant research interest.

We prove that over-parameterized single layer fully connected autoencoders do not merely interpolate, but rather, memorize training data: they produce outputs in (a non-linear version of) the span of the training examples.

In contrast to fully connected autoencoders, we prove that depth is necessary for memorization in convolutional autoencoders.

Moreover, we observe that adding nonlinearity to deep convolutional autoencoders results in a stronger form of memorization: instead of outputting points in the span of the training images, deep convolutional autoencoders tend to output individual training images.

Since convolutional autoencoder components are building blocks of deep convolutional networks, we envision that our findings will shed light on the important question of the inductive bias in over-parameterized deep networks.

As deep convolutional neural networks (CNNs) become ubiquitous in computer vision thanks to their strong performance on a range of tasks (Goodfellow et al., 2016) , recent work has begun to analyze the role of interpolation (perfectly fitting training data) in such networks BID0 Zhang et al., 2017) .

These works show that deep overparametrized networks can interpolate training data even when the labels are random.

For an overparameterized model, there are typically infinitely many interpolating solutions.

Thus it is important to characterize the inductive bias of an algorithm, i.e., the properties of the specific solution chosen by the training procedure.

In this paper we study autoencoders (Goodfellow et al., 2016) , i.e. maps ψ : R d → R d that are trained to satisfy DISPLAYFORM0 Autoencoders are typically trained by solving arg min DISPLAYFORM1 by gradient descent over a parametrized function space Ψ.There are many interpolating solutions to the autoencoding problem in the overparametrized setting.

We characterize the inductive bias as memorization when the autoencoder output is within the span of the training data and strong memorization when the output is close to one of the training examples for almost any input.

Studying memorization in the context of autoencoders is relevant since (1) components of convolutional autoencoders are building blocks of many CNNs; (2) layerwise pre-training using autoencoders is a standard technique to initialize individual layers of CNNs to improve training (Belilovsky et al., 2019; Bengio et al., 2007; Erhan et al., 2010) ; and (3) autoencoder architectures are used in many image-to-image tasks such as image segmentation, image impainting, etc. (Ulyanov et al., 2017) .

While the results in this paper hold generally for autoencoders, we concentrate on image data, since this allows identifying memorization by visual inspection of the input and output.

To illustrate the memorization phenomenon, consider linear single layer fully connected autoencoders.

This autoencoding problem can be reduced to linear regression (see Appendix A).

It is well-known that solving overparametrized linear regression by gradient descent initialized at zero converges to the minimum norm solution (see, e.g., Theorem 6.1 in (Engl et al., 1996) ).

This minimum norm solution translated to the autoencoding setting corresponds to memorization of the training data: after training the autoencoder, any input image is mapped to an image that lies in the span of the training set.

In this paper, we prove that the memorization property extends to nonlinear single layer fully connected autoencoders.

We proceed to show that memorization extends to deep (but not shallow) convolutional autoencoders.

As a striking illustration of this phenomenon consider FIG0 .

After training a U-Net architecture (Ronneberger et al., 2015) , which is commonly used in image-to-image tasks (Ulyanov et al., 2017) , on a single training image, any input image is mapped to the training image.

Related ideas were concurrently explored for autoencoders trained on a single example in (Zhang et al., 2019) .The main contributions of this paper are as follows.

Building on the connection to linear regression, we prove that single layer fully connected nonlinear autoencoders produce outputs in the "nonlinear" span (see Definition 2) of the training data.

Interestingly, we show in Section 3 that in contrast to fully connected autoencoders, shallow convolutional autoencoders do not memorize training data, even when adding filters to increase the number of parameters.

In Section 4, we observe that our memorization results for linear CNNs carry over to nonlinear CNNs.

Further, nonlinear CNNs demonstrate a strong form of memorization: the trained network outputs individual training images rather than just combinations of training images.

We end with a short discussion in Section 5.

Appendices E, F, G, and H provide additional details concerning the effect of downsampling, early stopping, and initialization on memorization in linear and nonlinear convolutional autoencoders.

In this section, we characterize memorization properties of nonlinear single layer fully connected autoencoders initialized at zero.

A nonlinear single layer fully connected autoencoder satisfies φ(Ax DISPLAYFORM0 where φ is a non-linear function (such as the sigmoid function) that acts element-wise with x DISPLAYFORM1 In the following, we provide a closed form solution for the matrix A when initialized at A (0) = 0 and computed using gradient descent on the mean squared error loss, i.e. DISPLAYFORM2 Let φ −1 (y) be the pre-image of y ∈ R of minimum 2 norm and for each j ∈ {1, 2, . . .

d} let DISPLAYFORM3 In the following, we provide three mild assumptions that are often satisfied in practice under which a closed form formula for A can be derived in the nonlinear overparameterized setting.

Assumption 1.

For all j ∈ {1, 2, . . .

, d} it holds that DISPLAYFORM4 (c) φ satisfies one of the following conditions: DISPLAYFORM5 (2) if φ −1 (x j ) > 0 , then φ is strictly concave and monotonically increasing on [0, φ −1 (x j )];(3) if φ −1 (x j ) < 0 , then φ is strictly convex and monotonically increasing on [φ −1 (x j ), 0]; DISPLAYFORM6 Assumption (a) typically holds for un-normalized images.

Assumption (b) is satisfied for example when using a minmax scaling of the images.

Assumption (c) holds for many nonlinearities used in practice including the sigmoid and tanh functions.

To prove memorization for overparametrized nonlinear single layer fully connected autoencoders, we first show how to reduce the non-linear setting to the linear setting.

Theorem 1.

Let n < d (overparametrized setting).

Under Assumption 1, solving (1) to achieve φ(Ax (i) ) ≈ x (i) using a variant of gradient descent (with an adaptive learning rate as described in Supplementary Material B) initialized at A (0) = 0 converges to a solution A (∞) that satisfies the linear system A (∞) DISPLAYFORM7 The proof is presented in Supplementary Material B. Given our empirical observations using a constant learning rate, we suspect that the adaptive learning rate used for gradient descent in the proof is not necessary for the result to hold.

As a consequence of Theorem 1, the single layer nonlinear autoencoding problem can be reduced to a linear regression problem.

This allows us to define a memorization property for nonlinear systems by introducing nonlinear analogs of an eigenvector and the span.

Definition 1 (φ-eigenvector).

Given a matrix A ∈ R d×d and element-wise nonlinearity φ, a vector u ∈ R d is a φ-eigenvector of A with φ-eigenvalue λ if φ(Au) = λu.

Definition 2 (φ-span).

Given a set of vectors U = {u 1 , . . .

u r } with u i ∈ R d and an element-wise nonlinearity φ, let φ −1 (U ) = {φ −1 (u 1 ) . . .

φ −1 (u r )}.

The nonlinear span of U corresponding to φ (denoted φ-span(U )) consists of all vectors φ(v) such that v ∈ span(φ −1 (U )).The following corollary characterizes memorization for nonlinear single layer fully connected autoencoders.

Corollary (Memorization in non-linear single layer fully connected autoencoders).

Let n < d (overparametrized setting) and let A (∞) be the solution to (1) using a variant of gradient descent with an adaptive learning rate initialized at A (0) = 0.

Then under Assumption 1, rank(A (∞) ) = dim(span(X)); in addition, the training examples DISPLAYFORM8 Proof.

Let S denote the covariance matrix of the training examples and let r := rank(S).

It then follows from Theorem 1 and the minimum norm solution of linear regression that rank(A (∞) ) ≤ r. Since in the overparameterized setting, A (∞) achieves 0 training error, the training examples satisfy φ(A (∞) x (i) ) = x (i) for all 1 ≤ i ≤ n, which implies that the examples are φ-eigenvectors with eigenvalue 1.

Hence, it follows that rank(A (∞) ) ≥ r and thus rank(A (∞) ) = r. Lastly, since the φ-eigenvectors are the training examples, it follows that φ(A (∞) y) ∈ φ-span(X) for any y ∈ R d .

In contrast to single layer autoencoders discussed in the previous section, we now show that shallow linear convolutional autoencoders in general do not memorize training data even in the overparametrized setting; hence depth is necessary for memorization in convolutional autoencoders.

For the following discussion of convolutional autoencoders, let the training samples be images in R s×s .

While all our results also hold for color images, we dropped the color channel to simplify notation.

Theorem 2.

A single filter convolutional autoencoder with kernel size k and k−1 2 zero padding trained to autoencode an image x ∈ R s×s using gradient descent on the mean squared error loss learns a rank s 2 solution.

The proof is presented in Supplementary Material C. The main ingredient of the proof is the construction of the matrix A to represent a linear convolutional autoencoder.

An algorithm for obtaining A for any linear convolutional autoencoder is presented in Supplementary Material D. Theorem 2 implies that even in the overparameterized setting, a single layer single filter convolutional autoencoder will not memorize training data.

For example, a network with a kernel of size 5 and a single training image of size s = 2 is overparametrized, since the number of parameters is 25 while the input has dimension 4.

However, in contrast to the non-convolutional setting, Theorem 2 implies that the rank of the learned solution is 4, which exceeds the number of training examples; i.e., memorization does not occur.

As explained in the following, this contrasting behavior stems from the added constraints imposed on the matrix A through convolutions, in particular the zeros forced by the structure of the matrix.

A concrete example illustrating this constraint is provided in Supplementary Material D.We now prove that these forced zeros prevent memorization in single layer single filter convolutional autoencoders.

The following lemma shows that a single layer matrix with just one forced zero cannot memorize arbitrary inputs.

Lemma 1.

A single layer linear autoencoder, represented by a matrix A ∈ R d×d with a single forced zero entry cannot memorize an arbitrary v ∈ R d .The proof follows directly from the fact that in the linear setting, memorization corresponds to projection onto the training example and thus cannot have a zero in a fixed data-independent entry.

Since single layer single filter convolutional autoencoders have forced zeros, Lemma 1 shows that these networks cannot memorize arbitrary inputs.

Next, we show that shallow convolutional autoencoders still contain forced zeros regardless of the number of filters that are used in the intermediate layers.

Theorem 3.

At least s − 1 layers are required for memorization (regardless of the number of filters per layer) in a linear convolutional autoencoder with filters of kernel size 3 applied to images of size s ×

s.

This lower bound follows by analyzing the forced zero pattern of A s−1 , which corresponds to the operator for the s − 1 layer network.

Importantly, Theorem 3 shows that adding filters cannot make up for missing depth, i.e., overparameterization through depth rather than filters is necessary for memorization in convolutional autoencoders.

The following corollary emphasizes this point.

Corollary.

A 2-layer linear convolutional autoencoder with filters of kernel size 3 and stride 1 for the hidden representation cannot memorize images of size 4 × 4 or larger, independently of the number of filters.

This shows that depth is necessary for memorization in convolutional autoencoders.

In Appendix E, we provide empirical evidence that depth is sufficient for memorization, and refine the lower bound from 3 to a lower bound layers needed to identify memorization in linear convolutional autoencoders.

While the number of layers needed for memorization are large according to this lower bound, in Appendix F, we show empirically that downsampling through strided convolution allows a network to memorize with far fewer layers.

We now provide evidence that our observations regarding memorization in linear convolutional autoencoders extend to the nonlinear setting.

In FIG2 , we observe that a downsampling nonlinear convolutional autoencoder with leaky ReLU activations (described in FIG13 ) strongly memorizes 10 examples, one from each class of CIFAR10.

That is, given a new test example from CIFAR10, samples from a standard Gaussian, or random sized color squares, the model outputs an image visually similar to one of the training examples instead of a combination of training examples.

This is in contrast to deep linear convolutional autoencoders; for example, in FIG2 , we see that training the linear model from 3a leads to the model outputting linear combinations of the training examples.

These results suggest that for deep nonlinear convolutional autoencoders the training examples are strongly attractive fixed points.

This paper identified the mechanism behind memorization in autoencoders.

While it is well-known that linear regression converges to a minimum norm solution when initialized at zero, we tied this phenomenon to memorization in non-linear single layer fully connected autoencoders, showing that they produce output in the nonlinear span of the training examples.

Furthermore, we showed that convolutional autoencoders behave quite differently since not every overparameterized convolutional autoencoder memorizes.

Indeed, we showed that overparameterization by adding depth or downsampling is necessary and empirically sufficient for memorization in the convolutional setting, while overparameterization by extending the number of filters in a layer does not lead to memorization.

Interestingly, we observed empirically that the phenomenon of memorization is pronounced in the non-linear setting, where nearly arbitrary input images are mapped to output images that are visually identifiable as one of the training images rather than a linear combination thereof as in the linear setting.

While the exact mechanism for this strong form of memorization in the non-linear setting still needs to be understood, this phenomenon is reminiscent of FastICA in Independent Component Analysis (Hyvrinen & Oja, 1997) or more general non-linear eigenproblems (Belkin et al., 2018b) , where every "eigenvector" (corresponding to training examples in our setting) of certain iterative maps has its own basin of attraction.

We conjecture that increasing the depth may play the role of increasing the number of iterations in those methods.

Since the use of deep networks with near zero initialization is the current standard for image classification tasks, we expect that our memorization results also carry over to these application domains.

We note that memorization is a particular form of interpolation (zero training loss) and interpolation has been demonstrated to be capable of generalizing to test data in neural networks and a range of other methods (Zhang et al., 2017; Belkin et al., 2018a) .

Our work could provide a mechanism to link overparameterization and memorization with generalization properties observed in deep convolutional networks.

Belilovsky, E., Eickenberg, M., and Oyallon, E.

In the following, we analyze the solution when using gradient descent to solve the autoencoding problem for the system DISPLAYFORM0 and the gradient with respect to the parameters A is DISPLAYFORM1 .

Hence gradient descent with learning rate γ > 0 will proceed according to the equation: DISPLAYFORM2 Now suppose that A (0) = 0, then we can directly solve the recurrence relation for t > 0, namely DISPLAYFORM3 Note that S is a real symmetric matrix, and so it has eigendecomposition S = QΛQ T where Λ is a diagonal matrix with eigenvalue entries λ 1 ≥ λ 2 ≥ . . .

≥ λ r (where r is the rank of S).

Then: DISPLAYFORM4 , then we have that: DISPLAYFORM5 which is the minimum norm solution.

In the following, we present the proof of Theorem 2 from the main text.

Proof.

As we are using a fully connected network, the rows of the matrix A can be optimized independently during gradient descent.

Thus without loss of generality, we only consider the convergence of the first row of the matrix A denoted DISPLAYFORM0 .The loss function for optimizing row A 1 is given by: DISPLAYFORM1 Our proof involves using gradient descent on L but with a different adaptive learning rate per example.

That is, let γ (t) i be the learning rate for training example i at iteration t of gradient descent.

Without loss of generality, fix j ∈ {1, . . .

, d}. The gradient descent equation for parameter a j is: DISPLAYFORM2 To simplify the above equation, we make the following substitution γ DISPLAYFORM3 i.e., the adaptive component of the learning rate is the reciprocal of φ (A1 x (i) ) (which is nonzero due to monotonicity conditions on φ).

Note that we have included the negative sign so that if φ is monotonically decreasing on the region of gradient descent, then our learning rate will be positive.

Hence the gradient descent equation simplifies to DISPLAYFORM4 Before continuing, we briefly outline the strategy for the remainder of the proof.

First, we will use assumption (c) and induction to upper bound the sequence (φ(A DISPLAYFORM5 j ) with a sequence along a line segment.

The iterative form of gradient descent along the line segment will have a simple closed form and so we will obtain a coordinate-wise upper bound on our sequence of interest A (t)1 .

Next, we show that our upper bound given by iterations along the selected line segment is in fact a coordinate-wise least upper bound.

Then we show that A (t) 1 is a coordinate-wise monotonically increasing function, meaning that it must converge to the least upper bound established prior.

Without loss of generality assume, DISPLAYFORM6 since the right hand side is just the line segment joining points (0, φ(0)) and (φ −1 (x DISPLAYFORM7 , which must be above the function φ(x) if the function is strictly convex.

To simplify notation, we write DISPLAYFORM8 .

Now that we have established a linear upper bound on φ, consider a sequence B DISPLAYFORM9 but with updates: DISPLAYFORM10 Now if we let γ i = γ si , then we have DISPLAYFORM11 which is the gradient descent update equation with learning rate γ for the first row of the parameters B in solving DISPLAYFORM12 Since gradient descent for a linear regression initialized at 0 converges to the minimum norm solution (see Appendix A), we obtain that B DISPLAYFORM13 Next, we wish to show that B (t) j is a coordinate-wise upper bound for A (t)1 .

To do this, we first select L such that DISPLAYFORM14 Then, we proceed by induction to show the following: DISPLAYFORM15 To simplify notation, we follow induction for a 2.

We have that: a DISPLAYFORM16 Hence we have A 3.

Now for t = 2, DISPLAYFORM17 However, we know that B DISPLAYFORM18 1 , A DISPLAYFORM19 1 since the on the interval [0, φ −1 (x 1 )], φ is bounded above by the line segments with endpoints (0, φ(0)) and (φ −1 (x DISPLAYFORM20 ).

Now for the second component of induction, we have: DISPLAYFORM21 To simplify the notation, let: DISPLAYFORM22 DISPLAYFORM23 Thus, we have DISPLAYFORM24 Inductive Hypothesis: We now assume that for t = k, DISPLAYFORM25 InductiveStep: Now we consider t = k + 1.

Since b DISPLAYFORM26 Consider now the difference between b DISPLAYFORM27 and a DISPLAYFORM28 where the first inequality comes from the fact that −sA DISPLAYFORM29 is a point on the line that upper bounds φ on the interval [0, φ −1 (x 1 )], and the second inequality comes from the fact that each x (i) j < 1.

Hence, with a learning rate of DISPLAYFORM30 we obtain that c DISPLAYFORM31 > 0 as desired.

Hence, the first component of the induction is complete.

To fully complete the induction we must show that DISPLAYFORM32 We proceed as we did in the base case: DISPLAYFORM33 To simplify the notation, let DISPLAYFORM34 and thus DISPLAYFORM35 This completes the induction argument and as a consequence we obtain c (t) l > 0 and DISPLAYFORM36 ≤ L for all integers t ≥ 2 and for 1 ≤ l, j ≤ d.

i is an upper bound for a (t) i given learning rate γ ≤ 1 nLd .

By symmetry between the rows of A, we have that, the solution given by solving the system Bx (i) = φ −1 (x (i) ) for 1 ≤ i ≤ n using gradient descent with constant learning rate is an entry-wise upper bound for the solution given by solving φ(Ax (i) ) = x (i) for 1 ≤ i ≤ n using gradient descent with adaptive learning rate per training example when DISPLAYFORM0 Now, since the entries of B (t) are bounded and since they are greater than the entries of A (t) for the given learning rate, it follows from the gradient update equation for A that the sequence of entries of A (t) are monotonically increasing from 0.

Hence, if we show that the entries of B (∞) are least upper bounds on the entries of A (t) , then it follows that the entries of A (t) converge to the entries of B (∞) .Suppose for the sake of contradiction that the least upper bound on the sequence a DISPLAYFORM1 for 1 ≤ i ≤ n. Since we are in the overparameterized setting, at convergence A DISPLAYFORM2 1 .

This implies that B (∞) 1 DISPLAYFORM3 under φ.

However, we know that B DISPLAYFORM4 This completes the proof and so we conclude that A ( t) converges to the solution given by autoencoding the linear system Ax (i) = φ −1 x (i) for 1 ≤ i ≤ n using gradient descent with constant learning rate.

In the following, we present the proof for Theorem 3 from the main text.

Proof.

A single convolutional filter with kernel size k and k−1 2 zero padding operating on an image of size s × s can be equivalently written as a matrix operating on a vectorized zero padded image of size (s + k − 1) 2 .

Namely, if C 1 , C 2 , . . .

C k 2 are the parameters of the convolutional filter, then the layer can be written as the matrix DISPLAYFORM0 . . . . . .

DISPLAYFORM1 and R r:t denotes a right rotation of R by t elements.

Now, training the convolutional layer to autoencode example x using gradient descent is equivalent to training R to fit s 2 examples using gradient descent.

Namely, R must satisfy Rx = x 1 , Rx l:1 = x 2 , . . .

Rx l:(s+k−1)(s−1)+s−1 = x s 2 where x T l:t denotes a left rotation of x T by t elements.

As in the proof for Theorem 1, we can use the general form of the solution for linear regression using gradient descent from Appendix A to conclude that the rank of the resulting solution will be s 2 .

In this section, we present how to extract a matrix form for convolutional and nearest neighbor upsampling layers.

We first present how to construct a block of this matrix for a single filter in Algorithm 1.

To construct a matrix for multiple filters, one need only apply the provided algorithm to construct separate matrix blocks for each filter and then concatenate them.

We first provide an example of how to convert a single layer convolutional network with a single filter of kernel size 3 into a single matrix for 3 × 3 images.

First suppose we have a 3 × 3 image x as input, which is shown vectorized below: DISPLAYFORM0 Next, let the parameters below denote the filter of kernel size 3 that will be used to autoencode the above example: DISPLAYFORM1 We now present the matrix form A for this convolutional filter such that A multiplied with the vectorized version of x will be equivalent to applying the convolutional filter above to the image x (the general algorithm to perform this construction is presented in Algorithm 1).

DISPLAYFORM2 Importantly, this example demonstrates that the matrix corresponding to a convolutional layer has a fixed zero pattern.

It is this forced zero pattern we use to prove that depth is required for memorization in convolutional autoencoders.

In downsampling autoencoders, we will also need to linearize the nearest neighbor upsampling operation.

We provide the general algorithm to do this in Algorithm 2.

Here, we provide a simple example for an upsampling layer with scale factor 2 operating on a vectorized zero padded 1 × 1 image: Table 1 .

Linear convolutional autoencoders with a varying number of layers and filters were initialized close to zero and trained on 2 normally distributed images of size 3 × 3.

Memorization does not occur in any of the examples (memorization would be indicated by the spectrum containing two eigenvalues that are 1 and the remaining eigenvalues being close to 0).

Increasing the number of filters per layer has minimal effect on the spectrum.

DISPLAYFORM3 DISPLAYFORM4 The resulting output is a zero padded upsampled version of the input.

While Theorem 3 provided a lower bound on the depth required for memorization, Table 1 shows that the depth predicted by this bound is not sufficient.

In each experiment, we trained a linear convolutional autoencoder to encode 2 randomly sampled images of size 3×3 with a varying number of layers and filters per layer.

The first 3 rows of Table 1 show that the lower bound from Theorem 3 is not sufficient for memorization (regardless of overparameterization through filters) since memorization would be indicated by a rank 2 solution (with the third eigenvalue close to zero).

In fact, the remaining rows of Table 1 show that even 8 layers are not sufficient for memorizing two images of size 3 × 3.

layers (as predicted by our heuristic lower bound) with a single filter per layer, initialized with each parameter as close to zero as possible, memorize training examples of size s × s similar to a single layer fully connected system.

The bracket notation in the spectrum indicates that the magnitude of the remaining eigenvalues in the spectrum is below the value in the brackets.

Next we provide a heuristic bound to determine the depth needed to observe memorization (denoted by "Heuristic Lower Bound Layers" in TAB4 ).

Theorem 3 and Table 1 suggest that the number of filters per layer does not have an effect on the rank of the learned solution.

We thus only consider networks with a single filter per layer with kernel size 3.

It follows from Section 2 that overparameterized single layer fully connected autoencoders memorize training examples when initialized at 0.

Hence, we can obtain a heuristic bound on the depth needed to observe memorization in linear convolutional autoencoders with a single filter per layer based on the number of layers needed for the network to have as many parameters as a fully connected network.

The number of parameters in a single layer fully connected linear network operating on vectorized images of size s×s is s 4 .

Hence, using a single filter per layer with kernel size 3, the network needs s 4 9 layers to achieve the same number of parameters as a fully connected network.

This leads to a heuristic lower bound of s 4 9 layers for memorization in linear convolutional autoencoders operating on images of size s × s.

In TAB4 , we investigate the memorization properties of networks that are initialized with parameters as close to zero as possible with the number of layers given by our heuristic lower bound and one filter of kernel size 3 per layer.

The first 6 rows of the table show that all networks satisfying our heuristic lower bound have memorized a single training example since the spectrum consists of a single eigenvalue that is 1 and remaining eigenvalues with magnitude less than ≈ 10 −2 .

Similarly, the spectra in the last 3 rows indicate that networks satisfying our heuristic lower bound also memorize multiple training examples, thereby suggesting that our bound is relevant in practice.

The experimental setup was as follows: All networks were trained using gradient descent with a learning rate of 10 for f ilterIndex ← 0 to f − 1 do 6:for kernelIndex ← 0 to 8 do

rowIndex ← kernelIndex mod 3 + paddedSize C ← zeros matrix of size ((resized + 2) 2 , f · paddedSize 2 )12:index ← resized + 2 + 1

for shif t ← 0 to resized − 1 do

nextBlock ← zeros matrix of size (resized, f · paddedSize 2 )15: DISPLAYFORM0 for rowShif t ← 1 to resized − 1 do return C 24: end function until the loss became less than 10 −6 (to speed up training, we used Adam (Kingma & Ba, 2015) with a learning rate of 10 −4 when the depth of the network was greater than 10).

For large networks with over 100 layers (indicated by an asterisk in TAB4 ), we used skip connections between every 10 layers, as explained in (He et al., 2016) , to ensure that the gradients can propagate to earlier layers.

TAB4 shows the resulting spectrum for each experiment, where the eigenvalues were sorted by there magnitudes.

The bracket notation indicates that all the remaining eigenvalues have magnitude less than the value provided in the brackets.

Interestingly, our heuristic lower bound also seems to work for deep networks that have skip connections, which are commonly used in practice.

The experiments in TAB4 indicate that over 200 layers are needed for memorization of 7 × 7 images.

In the next section, we discuss how downsampling can be used to construct much smaller convolutional autoencoders that memorize training examples.

To gain intuition for why downsampling can trade off depth to achieve memorization, consider a convolutional autoencoder that downsamples input to 1 × 1 representations through non-unit strides.

Such extreme downsampling makes a convolutional autoencoder equivalent to a fully connected network; hence given the results in Section 2 such downsampling convolutional networks are expected to memorize.

This is illustrated in FIG13 : The network uses strides of size 2 to progressively downsample to a 1 × 1 representation of a CIFAR10 input image.

Training the network on two images from CIFAR10, the rank of the learned solution is exactly 2 with the top eigenvalues being 1 and the corresponding eigenvectors being linear combinations of the training images.

In this case, using the default PyTorch initialization was sufficient in forcing each parameter to be close to zero.

Memorization using convolutional autoencoders is also observed with less extreme forms of downsampling.

In fact, we observed that downsampling to a smaller representation and then operating on the downsampled representation index ← outputSize + 1 DISPLAYFORM0 for f ilterIndex ← 0 to f − 1 do 6:for rowIndex ← 1 to s do 7:for scaleIndex ← 0 to scale − 1 do 8:for columnIndex ← 0 to s do 9:row ← zeros vector of size (f (s + 2) 2 )10: DISPLAYFORM1 for repeatIndex ← 0 to scale − 1 do 12: return U 22: end function with depth provided by our heuristic bound established in Section E also leads to memorization.

As an example, consider the network in FIG14 operating on images from CIFAR10 (size 32 × 32).

This network downsamples a 32 × 32 CIFAR10 image to a 4 × 4 representation after layer 1.

As suggested by our heuristic lower bound for 4×4 images (see TAB4 ) we use 29 layers in the network.

Figure 4b indicates that this network indeed memorized the image by producing a solution of rank 1 with eigenvalue 1 and corresponding eigenvector being the dog image.

DISPLAYFORM2

Non-downsampling AutoencodersWe start by investigating whether the heuristic bound on depth needed for memorization that we have established for linear convolutional autoencoders carries over to nonlinear convolutional autoencoders.

Example.

Consider a deep nonlinear convolutional autoencoder with a single filter per layer of kernel size 3, 1 unit of zero padding, and stride 1 followed by a leaky ReLU (Xu et al., 2015) activation that is initialized with parameters as close to 0 as possible.

In TAB4 we reported that its linear counterpart memorizes 4 × 4 images with 29 layers.

Figure 5 shows that also the corresponding nonlinear network with 29 layers can memorize 4 × 4 images.

While the spectrum can be used to prove memorization in the linear setting, since we are unable to extract a nonlinear equivalent of the spectrum for these networks, we can only provide evidence for memorization by visual inspection.

This example suggests that our results on depth required for memorization in deep linear convolutional autoencoders carry over to the nonlinear setting.

In fact, when training on multiple examples, we observe that memorization is of a stronger form in the nonlinear case.

Consider the example in Figure 6 .

We see that given new test examples, a nonlinear convolutional autoencoder with 5 layers trained on 2 × 2 images outputs individual training examples instead of combinations of training examples.

Memorization with Early Stopping.

In all examples discussed so far, we trained the autoencoders to achieve nearly 0 error (less than 10 −6 ).

In this section, we provide empirical evidence suggesting that the phenomenon of memorization is robust in the sense of appearing early in train- ing, well before full convergence.

The examples in FIG16 using the network architecture shown in FIG13 (where the nonlinear version is created by adding Leaky ReLU activation after every convolutional layer) illustrate this phenomenon.

Both linear and nonlinear convolutional networks (that satisfy the heuristic conditions for memorization discussed in Sections 3 and F) show memorization throughout the training process.

As illustrated in FIG16 , networks in which training was terminated early, map a new given input to the current representation of the training examples.

As shown in FIG16 , the nonlinear autoencoder trained to autoencode two images from CIFAR10 clearly outputs the learned representations when given arbitrary test examples.

As shown in FIG16 , memorization is evident throughout training also in the linear setting, although the outputs are noisier than in the nonlinear setting.

Initialization at Zero is Necessary for Memorization.

Section 2 showed that linear fully connected autoencoders initialized at zero memorize training examples by learning the minimum norm solution.

Since in the linear setting the distance to the span of the training examples remains constant when minimizing the autoencoder loss regardless of the gradient descent algorithm used, non-zero initialization does not result in memorization.

Hence, to see memorization, we require that each parameter of an autoencoder be Figure 5 .

A 29 layer network with a single filter of kernel size 3, 1 unit of zero padding, and stride 1 followed by a leaky ReLU activation per layer initialized with every parameter set to 10 −1 memorizes 4 × 4 images.

Our training image consists of a white square in the upper left hand corner and the test examples contain pixels drawn from a standard normal distribution.

Figure 6 .

A 5 layer nonlinear network strongly memorizes 2 × 2 images.

The network has a single filter of kernel size 3, 1 unit of zero padding, and stride 1 followed by a leaky ReLU activation per layer with Xavier Uniform initialization.

The network also has skip connections between every 2 layers.

The training images are orthogonal: one with a white square in the upper left corner and one with a white square in the lower right corner.

The test examples contain pixels drawn from a standard normal distribution.

initialized as close to zero as possible (while allowing for training).We now briefly discuss how popular initialization techniques such as Kaiming uniform/normal (He et al., 2015) , Xavier uniform/normal (Glorot & Bengio, 2010) , and default PyTorch initialization (Paszke et al., 2017) relate to zero initialization.

In general, we observe that Kaiming uniform/normal initialization leads to an output with a larger 2 norm as compared to a network initialized using Xavier uniform/normal or PyTorch initializations.

Thus, we do not expect Kaiming uniform/normal initialized networks to present memorization as clearly as the other initialization schemes.

That is, for linear convolutional autoencoders, we expect these networks to converge to a solution further from the minimum nuclear norm solution and for nonlinear convolutional autoencoders, we expect these networks to produce noisy versions of the training examples when fed arbitrary inputs.

This phenomenon is demonstrated experimentally in the examples in FIG17 .

FIG13 (modified with Leaky ReLU activations after each convolutional layer) behaves when initialized using Xavier uniform/normal and Kaiming uniform/normal strategies.

We also give the 2 norm of the output for the training example prior to training.

Consistent with our predictions, the Kaiming uniform/normal strategies have larger norms and the output for arbitrary inputs shows that memorization is noisy.

<|TLDR|>

@highlight

 We identify memorization as the inductive bias of interpolation in overparameterized fully connected and convolutional auto-encoders. 