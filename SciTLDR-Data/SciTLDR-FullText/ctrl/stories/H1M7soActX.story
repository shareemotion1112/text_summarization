Understanding the behavior of  stochastic gradient descent (SGD) in the context of deep neural networks has raised lots of concerns recently.

Along this line, we theoretically study a general form of gradient based optimization dynamics with unbiased noise, which unifies SGD and standard Langevin dynamics.

Through investigating this general optimization dynamics, we analyze the behavior of SGD on escaping from minima and its regularization effects.

A novel indicator is derived to characterize the efficiency of escaping from minima through measuring the alignment of noise covariance and the curvature of loss function.

Based on this indicator, two conditions are established to show which type of noise structure is superior to isotropic noise in term of escaping efficiency.

We further show that the anisotropic noise in SGD satisfies the two conditions, and thus helps to  escape from sharp and poor minima effectively, towards more stable and flat minima that typically generalize well.

We verify our understanding through comparing this anisotropic diffusion with full gradient descent plus isotropic diffusion (i.e. Langevin dynamics) and other types of position-dependent noise.

As a successful learning algorithm, stochastic gradient descent (SGD) was originally adopted for dealing with the computational bottleneck of training neural networks with large-scale datasets BID0 .

Its empirical efficiency and effectiveness have attracted lots of attention.

And thus, SGD and its variants have become standard workhorse for learning deep models.

Besides the aspect of empirical efficiency, recently, researchers started to analyze the optimization behaviors of SGD and its impacts on generalization.

The optimization properties of SGD have been studied from various perspectives.

The convergence behaviors of SGD for simple one hidden layer neural networks were investigated in BID13 BID1 .

In non-convex settings, the characterization of how SGD escapes from stationary points, including saddle points and local minima, was analyzed in BID3 BID10 BID8 .On the other hand, in the context of deep learning, researchers realized that the noise introduced by SGD impacts the generalization, thanks to the research on the phenomenon that training with a large batch could cause a significant drop of test accuracy BID11 .

Particularly, several works attempted to investigate how the magnitude of the noise influences the generalization during the process of SGD optimization, including the batch size and learning rate BID7 BID5 BID2 BID9 .

Another line of research interpreted SGD from a Bayesian perspective.

In BID14 BID2 , SGD was interpreted as performing variational inference, where certain entropic regularization involves to prevent overfitting.

And the work BID21 tried to provide an understanding based on model evidence.

These explanations are compatible with the flat/sharp minima argument BID6 BID11 , since Bayesian inference tends to targeting the region with large probability mass, corresponding to the flat minima.

However, when analyzing the optimization behavior and regularization effects of SGD, most of existing works only assume the noise covariance of SGD is constant or upper bounded by some constant, and what role the noise structure of stochastic gradient plays in optimization and generalization was rarely discussed in literature.

In this work, we theoretically study a general form of gradient-based optimization dynamics with unbiased noise, which unifies SGD and standard Langevin dynamics.

By investigating this general dynamics, we analyze how the noise structure of SGD influences the escaping behavior from minima and its regularization effects.

Several novel theoretical results and empirical justifications are made.1.

We derive a key indicator to characterize the efficiency of escaping from minima through measuring the alignment of noise covariance and the curvature of loss function.

Based on this indicator, two conditions are established to show which type of noise structure is superior to isotropic noise in term of escaping efficiency;2.

We further justify that SGD in the context of deep neural networks satisfies these two conditions, and thus provide a plausible explanation why SGD can escape from sharp minima more efficiently, converging to flat minima with a higher probability.

Moreover, these flat minima typically generalize well according to various works BID6 BID11 BID16 BID22 .

We also show that Langevin dynamics with well tuned isotropic noise cannot beat SGD, which further confirms the importance of noise structure of SGD; 3.

A large number of experiments are designed systematically to justify our understanding on the behavior of the anisotropic diffusion of SGD.

We compare SGD with full gradient descent with different types of diffusion noise, including isotropic and positiondependent/independent noise.

All these comparisons demonstrate the effectiveness of anisotropic diffusion for good generalization in training deep networks.

The remaining of the paper is organized as follows.

In Section 2, we introduce the background of SGD and a general form of optimization dynamics of interest.

We then theoretically study the behaviors of escaping from minima in Ornstein-Uhlenbeck process in Section 3, and establish two conditions for characterizing the noise structure that affects the escaping efficiency.

In Section 4, we show that the noise of SGD in the context of deep learning meets the two conditions, and thus explains its superior efficiency of escaping from sharp minima over other dynamics with isotropic noise.

Various experiments are conducted for verifying our understanding in Section 5, and we conclude the paper in Section 6.

In general, supervised learning usually involves an optimization process of minimizing an empirical loss over training data, DISPLAYFORM0 denotes the training set with N i.i.d.

samples, the prediction function f is often parameterized by ?? ??? R D , such as deep neural networks.

And (??, ??) is the loss function, such as mean squared error and cross entropy, typically corresponding to certain negative log likelihood.

Due to the over parameterization and non-convexity of the loss function in deep networks, there exist multiple global minima, exhibiting diverse generalization performance.

We call those solutions generalizing well good solutions or minima, and vice versa.

Gradient descent and its stochastic variants A typical approach to minimize the loss function is gradient descent (GD), the dynamics of which in each iteration t is, ?? t+1 = ?? t ??? ?? t g 0 (?? t ), where g 0 (?? t ) = ??? ?? L(?? t ) denotes the full gradient and ?? t denotes the learning rate.

In non-convex optimization, a more useful kind of gradient based optimizers act like GD with an unbiased noise, including gradient Langevin dynamics (GLD), ?? t+1 = ?? t ??? ?? t g 0 (?? t ) + ?? t t , t ??? N (0, I), and stochastic gradient descent (SGD), during each iteration t of which, a minibatch of training samples with size m are randomly selected, with index set B t ??? {1, 2, . . .

, N }, and a stochastic gradient is evaluated based on the chosen minibatch,g(?? t ) = i???Bt ??? ?? (f (x i ; ?? t ), y i )/m, which is an unbiased estimator of the full gradient g 0 (?? t ).

Then, the parameters are updated with some learning rate ?? t as ?? t+1 = ?? t ??? ?? tg (?? t ).

Denote g(??) = ??? ?? ((f (x; ??), y), the gradient for loss with a single data point (x, y), and assume that the size of minibatch is large enough for the central limit theorem to hold, and thusg(?? t ) follows a Gaussian distribution BID14 , DISPLAYFORM1 (1) Note that the covariance matrix ?? depends on the model architecture, dataset and the current parameter ?? t .

Now we can rewrite the update of SGD as, DISPLAYFORM2 Inspired by GLD and SGD, we may consider a general kind of optimization dynamics, namely, gradient descent with unbiased noise, DISPLAYFORM3 For small enough constant learning rate ?? t = ??, the above iteration in Eq. (3) can be treated as the numerical discretization of the following stochastic differential equation BID9 BID2 , DISPLAYFORM4 Considering ???? 2 t ?? t as the coefficient of noise term, existing works BID7 BID9 studied the influence of noise magnitude of SGD on generalization, i.e. ???? 2 t = ??/m.

In this work, we focus on studying the benefits of anisotropic structure of ?? t in SGD helping escape from minima by bridging the covariance matrix with the Hessian of the loss surface, and its implicit regularization effects on generalization, especially in deep learning context.

For the purpose of eliminating the influence of the noise magnitude, we constrain it to be a constant when studying different structures of noise covariance.

The noise magnitude could be evaluated as the expectation of the squared norm of the noise vector, DISPLAYFORM5 Thus, we introduce the following constraint, DISPLAYFORM6 From the statistical physics point of view, Tr(???? 2 t ?? t ) characterizes the kinetic energy (Gardiner), thus it is natural to force the energy to be unchanging, otherwise it is trivial that the higher the energy is, the less stable the system is.

For simplicity, we absorb ???? 2 t into ?? t , denoting ???? 2 t ?? t as ?? t .

If not pointed out, the subscript t of matrix ?? t is omitted to emphasize that we are fixing t and discussing the varying structure of ??.

For a general loss function L(??) = E X X (??) (the expectation could be either population or empirical), where X denotes data example and ?? denoted parameters to be optimized, under suitable smoothness assumptions, the SDE associated with the gradient variant optimizer as shown in Eq. (4) can be written as follows BID9 BID2 BID8 , with little abuse of notation, DISPLAYFORM0 Let L 0 = L(?? 0 ) be one of the minimal values of L(??), then for a fixed t small enough (such that DISPLAYFORM1 It is natural to measure the escaping efficiency using E[L t ??? L 0 ] since it characterizes the increase of the potential, i.e., the increase of the loss L. And also note that L t ??? L 0 ??? 0, for any ?? > 0, the escaping probability DISPLAYFORM2 where H t denotes the Hessian of L(?? t ) at ?? t .We provide the proof in Appendix, and the same for the other propositions.

The escaping efficiency for general processes is hard to analyze due to the intractableness of the integral in Eq. (8).

However, we may consider the second-order approximation locally near the minima ?? 0 , where DISPLAYFORM3 .

Without losing generality, we suppose ?? 0 = 0.

Further, suppose that H is a positive definite matrix and the diffusion covariance ?? t = ?? is constant for t. Then the SDE FORMULA7 becomes an Ornstein-Uhlenbeck process, DISPLAYFORM4 Proposition 2 (Escaping efficiency of Ornstein-Uhlenbeck process).

For Ornstein-Uhlenbeck process (9), with t small enough, the escaping efficiency from minimum ?? 0 = 0 is, DISPLAYFORM5 Inspired by Proposition 1 and Proposition 2, we propose Tr (H??) as an empirical indicator measuring the efficiency for a stochastic process escaping from minima.

Now we turn to analysis which kind of noise covariance structure ?? will benefit escaping sharp minima, under the constraint Eq. (6).Firstly, for the isotropic loss surface, i.e., DISPLAYFORM6 Tr ??, which is invariant under the constraint that Tr ?? is constant (Eq. FORMULA6 ).

Thus it is only nontrivial to study the impact of noise structure when the Hessian of loss surface is anisotropic.

Secondly, H and ?? being semi-positive definite, to achieve the maximum of Tr(H??) under constraint (6), ?? should be DISPLAYFORM7 , where ?? 1 , u 1 are the maximal eigenvalue and corresponding unit eigenvector of H. Note that the rank-1 matrix ?? * is highly anisotropic.

More generally, the following Proposition 3 characterizes one kind of anisotropic noise significantly outperforming isotropic noise in order of number of parameters D, given H is ill-conditioned.

Proposition 3 (The benefits of anisotropic noise).

With semi-positive definite H and ??, assume DISPLAYFORM8 DISPLAYFORM9 (2) ?? is "aligned" with H. Let u i be the corresponding unit eigenvector of eigenvalue ?? i , for some projection coefficient a > 0, DISPLAYFORM10 then we have the benefit of the anisotropic noise over the isotropic one in term of escaping efficiency, which can be characterized by the follow ratio, DISPLAYFORM11 where?? = Tr ?? D I denotes the covariance of isotropic noise, to meet the constraint Eq. (6).To give some geometric intuitions on the left hand side of Eq. FORMULA2 , let the maximal eigenvalue and its corresponding unit eigenvector of ?? be ?? 1 , v 1 , then the right hand side has a lower bound as u DISPLAYFORM12 Thus if the maximal eigenvalues of H and ?? are aligned in proportion, ?? 1 / Tr ?? ??? a 1 ?? 1 / Tr H, and the angle of their corresponding unit eigenvectors is close to zero, u 1 , v 1 ??? a 2 , the second condition Eq. (12) in Proposition 3 holds for a = a 1 a 2 .Typically, in the scenario of modern deep neural networks, due to the over-parameterization, Hessian and the gradient covariance are usually ill-conditioned and anistropic near minima, as shown by BID19 and BID2 .

Thus the first condition in Eq. (11) usually holds for deep neural networks, and we further justify it by experiments in Section 5.3.

Therefore, in the following section, we turn to focus on how the gradient covariance, i.e. the covariance of SGD noise meets the second condition of Proposition 3 in the context of deep neural networks.

In this section, we mainly investigate the anisotropic structure of gradient covariance in SGD, and explore its connection with the Hessian of loss surface.

Around the true parameter According to the classic statistical theory (Pawitan, 2001, Chap. 8) , for population loss L(??) = E X (??), with being the negative log likelihood, when evaluating at the true parameter ?? * , there is the exact equivalence between the Hessian H of the population loss and Fisher information matrix F , DISPLAYFORM0 In practice, with the assumptions that the sample size N is large enough (i.e. indicating asymptotic behavior) and suitable smoothness conditions, when the current parameter ?? t is not far from the ground truth, Fisher is close to Hessian.

Thus we can obtain the following approximate equality between gradient covariance and Hessian, DISPLAYFORM1 The first approximation is due to the dominance of noise over the mean of gradient in the later stage of SGD optimization, which has been shown in BID20 .

A similar experiment as BID20 has been conducted to demonstrate this observation, which is left in Appendix due to the limit of space.

In the following, we theoretically characterize the closeness between ?? and H in the context of one hidden layer neural networks; and show that the gradient covariance introduced by SGD indeed has more benefits than isotropic one in term of escaping from minima, provided some assumptions.

One hidden layer neural network with fixed output layer parameters For binary classification neural network with one hidden layer in classic setups (with softmax and cross-entropy loss), we have following results to globally bound Fisher and Hessian with each other.

Proposition 4 (The relationship between Fisher and Hessian in one hidden layer neural network).

Consider the binary classification problem with data {(x i , y i )} i???I , y ??? {0, 1}, and typical (either population or empirical) loss as DISPLAYFORM2 , where f denotes the output of neural network, and ?? denotes the cross-entropy loss with softmax, DISPLAYFORM3 If: (1) the neural network f is with one hidden layer and piece-wise linear activation.

And the parameters of output layer are fixed during training; (2) the optimization happens on a set U such that, f (x; ??) ??? (???C, C), ????? ??? U, ???x, i.e., the output of the classifier is bounded during optimization.

Then, we have the following relationship between (either population or empirical) Fisher F and Hessian H almost everywhere: DISPLAYFORM4 A B means that (B ??? A) is semi-positive definite.

There are a few remarks on Proposition 4.

Firstly, as shown in BID1 , the considered neural networks in Proposition 4 are non-convex and have multiple minima, and thus it is still nontrivial to consider the escaping from minima.

Secondly, the Proposition 4 holds in both population and empirical sense, since the proof does not distinguish the two circumstances.

Thirdly, the bound between F and H holds "globally" in the set U where the output f is bounded, rather than merely around the true global minima as discussed previously.

By Proposition 4, the following relationship between gradient covariance and Hessian could be derived.

Proposition 5 (The relationship between gradient covariance and Hessian in one hidden layer neural network).

Assume the conditions in Proposition 4 hold, then for some small ?? > 0 and for ?? close enough to minima ?? * (local or global), DISPLAYFORM5 holds for any positive eigenvalue ?? and its corresponding unit eigenvector u of Hessian H.As a direct corollary of Proposition 5, for such neural networks, the second condition Eq. (12) in Proposition 3 holds in a very loose sense.

Therefore, based on the discussion on population loss around the true parameters and one hidden layer neural network with fixed output layer parameters, given the ill-conditioning of H due to the over-parameterization of modern deep networks, according to Proposition 3, we can conclude the noise structure of SGD helps to escape from sharp minima much faster than the dynamics with isotropic noise, and converge to flatter solutions with a high probability.

These flat minima typically generalize well BID6 BID11 BID16 BID22 .

Thus, we attribute such properties of SGD on its better generalization performance comparing to GD, GLD and other dynamics with isotropic noise BID7 BID5 BID11 .In the following, we conduct a series of experiments systematically to verify our understanding on the behavior of escaping from minima and its regularization effects for different optimization dynamics.

To better understanding the behavior of anisotropic noise different from isotropic ones, we introduce dynamics with different kinds of noise structure to empirical study with, as shown on TAB0 .

DISPLAYFORM0 ??t is adjusted to make ??t t share the same expected norm as that of SGD DISPLAYFORM1 The covariance diag(??t) is the diagonal of the covariance of SGD noise.

DISPLAYFORM2 .

??i, vi are the first k leading eigenvalues and corresponding eigenvalues of the covariance of SGD noise, respectively. (A low rank approximation of ?? sgd t ) GLD Hessian t ??? N 0,Ht H t is a low rank approximation of the Hessian matrix of loss L(??) by its the first k leading eigenvalues and corresponding eigenvalues.

GLD 1st eigven (H) t ??? N 0, ??1u1u DISPLAYFORM3 ??1, u1 are the maximal eigenvalue and its corresponding unit eigenvector of the Hessian matrix of loss L(??t).

We design a 2-D toy example L(w 1 , w 2 ) with two basins, a small one and a large one, corresponding to a sharp and flat minima, (1, 1) and (???1, ???1), respectively, both of which are global minima.

Please refer to Appendix for the detailed constructions.

We initialize the dynamics of interest with the sharp minimum (w 1 , w 2 ) = (1, 1), and run them to study their behaviors escaping from this sharp minimum.

To explicitly control the noise magnitude, we only conduct experiments on GD, GLD const, GLD diag, GLD leading (with k = 2 = D in TAB0 , or in other words, the exactly covariance of SGD noise), GLD Hessian (k = 2) and GLD 1st eigven(H).

And we adjust ?? t in each dynamics to force their noise to share the same expected squared norm as defined in Eq. (6).

Figure 1(a) shows the trajectories of the dynamics escaping from the sharp minimum (1, 1) towards the flat one (???1, ???1), while Figure 1(b) presents the success rate of escaping for each dynamic during 100 repeated experiments.

As shown in Figure 1 , GLD 1st eigvec(H) achieves the highest success rate, indicating the fastest escaping speed from the sharp minimum.

The dynamics with anisotropic noise aligned with Hessian well, including GLD 1st eigvec(H), GLD Hessian and GLD leading, greatly outperform GD, GLD const with isotropic noise, and GLD diag with noise poorly aligned with Hessian.

These experiments are consistent with our theoretical analysis on Ornstein-Uhlenbeck process shown Proposition 2 and 3, demonstrating the benefits of anisotropic noise for escaping from sharp minima.

We empirically show that in one hidden layer neural network with fixed output layer parameters, the anisotropic noise induced by SGD indeed helps escape from sharp minima more efficiently than isotropic noise.

Three networks are trained to binary classify 1, 000 linearly separable two-dimensional points.

The number of hidden nodes for each network varies in {20, 200, 2000}. We plot the empirical indicator Tr (H??) in FIG3 .

We can easily observe that as the increase of the number of hidden nodes, the ratio

is enlarged significantly, which is consistent with the Eq. (13) described in Proposition 3.

In this part, we conduct a series of experiments in real deep learning scenarios to demonstrate the behavior of SGD noise and its implicit regularization effects.

We construct a noisy training set based on FashionMNIST dataset 1 .

Concretely, the training set consist of 1000 images with correct labels, and another 200 images with random labels.

All the test data are with clean labels.

A small LeNet-like network is utilized such that the spectrum decomposition over gradient covariance matrix and Hessian matrix are computationally feasible.

The network consists of two convolutional layers and two fully-connected layers, with 11, 330 parameters in total.

We firstly run the standard gradient decent for 3000 iterations to arrive at the parameters ?? * GD near the global minima with near zero training loss and 100% training accuracy, which are typically sharp minima that generalize poorly BID16 .

And then all other compared methods are initialized with ?? * GD and run for optimization with the same learning rate ?? t = 0.07 and same batch size m = 20 (if needed) for fair comparison 2 .Verification of SGD noise satisfying the conditions in Proposition 3 To see whether the noise of SGD in real deep learning circumstance satisfies the two conditions in Proposition 3, we run SGD optimizer initialized from ?? * GD , i.e. the sharp minima found by GD.

FIG4 shows the first 400 eigenvalues of Hessian at ?? * GD , from which we see that the 140th eigenvalue has already decayed to about 1% of the first eigenvalue.

Note that Hessian H ??? R D??D , D = 11330, thus H around ?? * GD approximately meets the ill-conditioning requirement in Proposition 3.

FIG4 shows the projection coefficient estimated by?? = u T 1 ??u1 Tr H ??1 Tr ?? along the trajectory of SGD.

The plot indicates that the projection coefficient is in a descent scale comparing to D 2d???1 , thus satisfying the second condition in Proposition 3.

Therefore, Proposition 3 ensures that SGD would escape from minima ?? * GD faster than GLD in order of O(D 2d???1 ), as shown in FIG4 (c).

An interesting observation is that in the later stage of SGD optimization, Tr(H??) becomes significantly (10 7 times) smaller than in the beginning stage, implying that SGD has already converged to minima being almost impossible to escape from.

This phenomenon demonstrates the reasonability to employ Tr(H??) as an empirical indicator for escaping efficiency.

Behaviors of different dynamics escaping from minima and its generalization effects To compare the different dynamics on escaping behaviors and generalization performance, we run dynamics initialized from the sharp minima ?? * GD found by GD.

The settings for each compared method are as follows.

The hyperparameter ?? 2 for GLD const has already been tuned as optimal (?? = 0.001) by grid search.

For GLD leading, we set k = 20 for comprising the computational cost and approximation accuracy.

As for GLD Hessian, to reduce the expensive evaluation of such a huge Hessian in each iteration, we set k = 20 and update the Hessian every 10 iterations.

We adjust ?? t in GLD dynamic, GLD Hessian and GLD 1st eigvec(H) to guarantee that they share the same expected squred noise norm defined in Eq. (6) as that of SGD.

And we measure the expected sharpness of different minima as E ?????N (0,?? 2 I) L(?? + ??) ??? L(??), as defined in BID16 , Eq. FORMULA7 ).

The results are shown in FIG5 .As shown in FIG5 , SGD, GLD 1st eigvec(H), GLD leading and GLD Hessian successfully escape from the sharp minima found by GD, while GLD, GLD dynamic and GLD diag are trapped in the minima.

This demonstrates that the methods with anisotropic noise "aligned" with loss curvature can help to find flatter minima that generalize well.

We also provide experiments on standard CIFAR-10 with VGG11 in Appendix.

DISPLAYFORM0 , and ?? = 0.01, the expectation is computed by average on 1000 times sampling.

We theoretically investigate a general optimization dynamics with unbiased noise, which unifies various existing optimization methods, including SGD.

We provide some novel results on the behaviors of escaping from minima and its regularization effects.

A novel indicator is derived for characterizing the escaping efficiency.

Based on this indicator, two conditions are constructed for showing what type of noise structure is superior to isotropic noise in term of escaping.

We then analyze the noise structure of SGD in deep learning and find that it indeed satisfies the two conditions, thus explaining the widely know observation that SGD can escape from sharp minima efficiently toward flat minina that generalize well.

Various experimental evidence supports our arguments on the behavior of SGD and its effects on generalization.

Our study also shows that isotropic noise helps little for escaping from sharp minima, due to the highly anisotropic nature of landscape.

This indicates that it is not sufficient to analyze SGD by treating it as an isotropic diffusion over landscape (Zhang et al., 2017; BID15 .

A better understanding of this out-of-equilibrium behavior BID2 ) is on demand.

Taking expectation with respect to the distribution of ?? t , DISPLAYFORM0 for the expectation of Brownian motion is zero.

Thus the solution of EY t is, DISPLAYFORM1

Proof.

Without losing generality, we assume that L 0 = 0.For multivariate Ornstein-Uhlenbeck process, when ?? 0 = 0 is an constant, ?? t follows a multivariate Gaussian distribution (??ksendal, 2003) .Consider change of variables ?? ??? ??(??, t) = e Ht ?? t .

Here, for symmetric matrix A, DISPLAYFORM0 where ?? 1 , . . . , ?? n and U are the eigenvalues and eigenvector matrix of A. Note that with this notation, DISPLAYFORM1 Applying Ito's lemma, we have DISPLAYFORM2 which we can integrate form 0 to t to get DISPLAYFORM3 The expectation of ?? t is zero.

And by Ito's isometry (??ksendal, 2003) , the covariance of ?? t is, DISPLAYFORM4 DISPLAYFORM5 The proof is finished.

Proof.

Firstly compute the gradients and Hessian of ??, DISPLAYFORM0 And note the Gauss-Newton decomposition for functions with the form of L = ?? ??? f , DISPLAYFORM1 Since the output layer parameters for f is fixed and the activation functions are piece-wise linear, f (x; ??) is a piece-wise linear function on its parameters ??.

Therefore ??? 2 f ????? 2 = 0, a.e., and H = E (x,y) DISPLAYFORM2 It is easy to check that e DISPLAYFORM3 .

Thus, DISPLAYFORM4 A.5 PROOF OF PROPOSITION 5Proof.

For simplicity, we define g := ??? , g 0 := ???L = E??? .The gradient covariance and Fisher has the following relationship, DISPLAYFORM5 Hence, DISPLAYFORM6 Therefore, with the condition DISPLAYFORM7 , we have DISPLAYFORM8 Tr F e ???2?? , for ?? small enough.

On the other hand, Proposition 4 indicates that e ???C F H e C F , which means, DISPLAYFORM9 .

Therefore, for ??, u being a positive eigenvalue and the corresponding unit eigenvector of H, we have DISPLAYFORM10 B ADDITIONAL EXPERIMENTS B.1 DOMINANCE OF NOISE OVER GRADIENT FIG6 shows the comparison of gradient mean and the expected norm of noise during training using SGD.

The dataset and model are same as the experiments of FashionMNIST in main paper, or as in Section C.2.

From FIG6 , we see that in the later stage of SGD optimization, noise indeed dominates gradient.

These experiments are implemented by TensorFlow 1.5.0.

FIG7 shows the first 50 iterations of FashionMNIST experiments in main paper.

We observe that SGD, GLD 1st eigvec(H), GLD Hessian and GLD leading successfully escape from the sharp minima found by GD, while GLD diag, GLD dynamic, GLD const and GD do not.

These experiments are implemented by TensorFlow 1.5.0.

Dataset Standard CIFAR-10 dataset without data augmentation.

Model Standard VGG11 network without any regularizations including dropout, batch normalization, weight decay, etc.

The total number of parameters of this network is 9, 750, 922.Training details Learning rates ?? t = 0.05 are fixed for all optimizers, which is tuned for the best generalization performance of GD.

The batch size of SGD is m = 100.

The noise std of GLD constant is ?? = 10 ???3 , which is tuned to best.

Due to computational limitation, we only conduct experiments on GD, GLD const, GLD dynamic, GLD diag and SGD.Estimation of Sharpness The sharpness are estimated by DISPLAYFORM0 with M = 100 and ?? = 0.01.Experiments Similar experiments are conducted as in main paper for CIFAR-10 and VGG11, as shown in FIG8 .

The observations and conclusions consist with main paper.

These experiments are implemented by PyTorch 0.3.0.

Note that ?? is the inverse of the Hessian of the quadric form generalizeing the sharp minima.

And the 3-dimensional plot of the loss surface is shown in FIG9 .Hyperparameters All learning rates are equal to 0.005.

All dynamics concerned are tuned to share the same expected square norm, 0.01.

The number of iteration during one run is 500.These experiments are implemented by PyTorch 0.3.0.

Dataset Our training set consists of 1200 examples randomly sampled from original FashionM-NIST training set, and we further specify 200 of them with randomly wrong labels.

The test set is same as the original FashionMNIST test set.

Model Network architecture: input ??? conv1 ??? max_pool ??? ReLU ??? conv2 ??? max_pool ??? ReLU ??? fc1 ??? ReLU ??? fc2 ??? output.

Both two convolutional layers use 5 ?? 5 kernels with 10 channels and no padding.

The number of hidden units between fully connected layers are 50.

The total number of parameters of this network are 11, 330.

??? GD: Learning rate ?? = 0.1.

We tuned the learning rate (in diffusion stage) in a wide range of {0.5, 0.2, 0.15, 0.1, 0.09, 0.08, . . .

, 0.01} and no improvement on generalization.??? GLD constant: Learning rate ?? = 0.07, noise std ?? = 10 ???3 .

We tuned the noise std in range of {10 ???1 , 10 ???2 , 10 ???3 , 10 ???4 , 10 ???5 } and no improvement on generalization.??? GLD dynamic: Learning rate ?? = 0.07.??? GLD diagnoal: Learning rate ?? = 0.07.??? GLD leading: Learning rate ?? = 0.07, number of leading eigenvalues k = 20, batchsize m = 20.

We first randomly divide the training set into 60 mini batches containing 20 examples, and then use those minibatches to estimate covariance matrix.??? GLD Hessian: Learning rate ?? = 0.07, number of leading eigenvalues = 20, update frequence f = 10.

Do to the limit of computational resources, we only update Hessian matrix every 10 iterations.

But add Hessian generated noise every iteration.

And to the same reason, we simplily set the coefficent of Hessian noise to Tr H/m Tr ??, to avoid extensively tuning of hyperparameter.

??? GLD 1st eigvec(H): Learning rate ?? = 0.07, as for GLD Hessian, and we set the coefficient of noise to ?? 1 /m Tr ??, where ?? 1 is the first eigenvalue of H. These experiments are implemented by TensorFlow 1.5.0.

<|TLDR|>

@highlight

We provide theoretical and empirical analysis on the role of anisotropic noise introduced by stochastic gradient on escaping from minima.