Deep neural networks (DNNs) typically have enough capacity to fit random data by brute force even when conventional data-dependent regularizations focusing on the geometry of the features are imposed.

We find out that the reason for this is the inconsistency between the enforced geometry and the standard softmax cross entropy loss.

To resolve this, we propose a new framework for data-dependent DNN regularization, the Geometrically-Regularized-Self-Validating neural Networks (GRSVNet).

During training, the geometry enforced on one batch of features is simultaneously validated on a separate batch using a validation loss consistent with the geometry.

We study  a particular case of GRSVNet, the Orthogonal-Low-rank Embedding (OLE)-GRSVNet, which is capable of producing highly discriminative features residing in orthogonal low-rank subspaces.

Numerical experiments show that OLE-GRSVNet outperforms DNNs with conventional regularization when trained on real data.

More importantly, unlike conventional DNNs, OLE-GRSVNet refuses to memorize random data or random labels, suggesting it only learns intrinsic patterns by reducing the memorizing capacity of the baseline DNN.

It remains an open question why DNNs, typically with far more model parameters than training samples, can achieve such small generalization error.

Previous work used various complexity measures from statistical learning theory, such as VC dimension (Vapnik, 1998) , Radamacher complexity BID1 , and uniform stability BID2 BID10 , to provide an upper bound for the generalization error, suggesting that the effective capacity of DNNs, possibly with some regularization techniques, is usually limited.

However, the experiments by Zhang et al. (2017) showed that, even with data-independent regularization, DNNs can perfectly fit the training data when the true labels are replaced by random labels, or when the training data are replaced by Gaussian noise.

This suggests that DNNs with data-independent regularization have enough capacity to "memorize" the training data.

This poses an interesting question for network regularization design: is there a way for DNNs to refuse to (over)fit training samples with random labels, while exhibiting better generalization power than conventional DNNs when trained with true labels?

Such networks are very important because they will extract only intrinsic patterns from the training data instead of memorizing miscellaneous details.

One would expect that data-dependent regularizations should be a better choice for reducing the memorizing capacity of DNNs.

Such regularizations are typically enforced by penalizing the standard softmax cross entropy loss with an extra geometric loss which regularizes the feature geometry BID8 Zhu et al., 2018; Wen et al., 2016) .

However, regularizing DNNs with an extra geometric loss has two disadvantages: First, the output of the softmax layer, usually viewed as a probability distribution, is typically inconsistent with the feature geometry enforced by the geometric loss.

Therefore, the geometric loss typically has a small weight to avoid jeopardizing the minimization of the softmax loss.

Second, we find that DNNs with such regularization can still perfectly (over)fit random training samples or random labels.

The reason is that the geometric loss (because of its small weight) is ignored and only the softmax loss is minimized.

This suggests that simply penalizing the softmax loss with a geometric loss is not sufficient to regularize DNNs.

Instead, the softmax loss should be replaced by a validation loss that is consistent with the enforced geometry.

More specifically, every training batch B is split into two sub-batches, the geometry batch B g and the validation batch B v .

The geometric loss l g is imposed on the features of B g for them to exhibit a desired geometric structure.

A semi-supervised learning algorithm based on the proposed feature geometry is then used to generate a predicted label distribution for the validation batch, which combined with the true labels defines a validation loss on B v .

The total loss on the training batch B is then defined as the weighted sum l = l g + λl v .

Because the predicted label distribution on B v is based on the enforced geometry, the geometric loss l g can no longer be neglected.

Therefore, l g and l v will be minimized simultaneously, i.e., the geometry is correctly enforced (small l g ) and it can be used to predict validation samples (small l v ).

We call such DNNs Geometrically-Regularized-Self-Validating neural Networks (GRSVNets).

See FIG0 for a visual illustration of the network architecture.

GRSVNet is a general architecture because every consistent geometry/validation pair can fit into this framework as long as the loss functions are differentiable.

In this paper, we focus on a particular type of GRSVNet, the Orthogonal-Low-rank-Embedding-GRSVNet (OLE-GRSVNet).

More specifically, we impose the OLE loss (Qiu & Sapiro, 2015) on the geometry batch to produce features residing in orthogonal subspaces, and we use the principal angles between the validation features and those subspaces to define a predicted label distribution on the validation batch.

We prove that the loss function obtains its minimum if and only if the subspaces of different classes spanned by the features in the geometry batch are orthogonal, and the features in the validation batch reside perfectly in the subspaces corresponding to their labels (see FIG0 ).

We show in our experiments that OLE-GRSVNet has better generalization performance when trained on real data, but it refuses to memorize the training samples when given random training data or random labels, which suggests that OLE-GRSVNet effectively learns intrinsic patterns.

Our contributions can be summarized as follows:• We proposed a general framework, GRSVNet, to effectively impose data-dependent DNN regularization.

The core idea is the self-validation of the enforced geometry with a consistent validation loss on a separate batch of features.• We study a particular case of GRSVNet, OLE-GRSVNet, that can produce highly discriminative features: samples from the same class belong to a low-rank subspace, and the subspaces for different classes are orthogonal.•

OLE-GRSVNet achieves better generalization performance when compared to DNNs with conventional regularizers.

And more importantly, unlike conventional DNNs, OLEGRSVNet refuses to fit the training data (i.e., with a training error close to random guess) when the training data or the training labels are randomly generated.

This implies that OLE-GRSVNet never memorizes the training samples, only learns intrinsic patterns.

Many data-dependent regularizations focusing on feature geometry have been proposed for deep learning BID8 Zhu et al., 2018; Wen et al., 2016) .

The center loss (Wen et al., 2016) produces compact clusters by minimizing the Euclidean distance between features and their class centers.

LDMNet (Zhu et al., 2018) extracts features sampling a collection of low dimensional manifolds.

The OLE loss BID8 Qiu & Sapiro, 2015) increases inter-class separation and intra-class similarity by embedding inputs into orthogonal low-rank subspaces.

However, as mentioned in Section 1, these regularizations are imposed by adding the geometric loss to the softmax loss, which, when viewed as a probability distribution, is typically not consistent with the desired geometry.

Our proposed GRSVNet instead uses a validation loss based on the regularized geometry so that the predicted label distribution has a meaningful geometric interpretation.

The way in which GRSVNets impose geometric loss and validation loss on two separate batches of features extracted with two identical baseline DNNs bears a certain resemblance to the siamese network architecture BID4 used extensively in metric learning BID3 BID6 BID7 Schroff et al., 2015; Sun et al., 2014) .

The difference is, unlike contrastive loss BID6 and triplet loss (Schroff et al., 2015) in metric learning, the feature geometry is explicitly regularized in GRSVNets, and a representation of the geometry, e.g., basis of the low-rank subspace, can be later used directly for the classification of test data.

Our work is also related to two recent papers (Zhang et al., 2017; BID0 addressing the memorization of DNNs.

Zhang et al. (2017) empirically showed that conventional DNNs, even with data-independent regularization, are fully capable of memorizing random labels or random data.

BID0 argued that DNNs trained with stochastic gradient descent (SGD) tend to fit patterns first before memorizing miscellaneous details, suggesting that memorization of DNNs depends also on the data itself, and SGD with early stopping is a valid strategy in conventional DNN training.

We demonstrate in our paper that when data-dependent regularization is imposed in accordance with the validation, GRSVNets will never memorize random labels or random data, and only extracts intrinsic patterns.

An explanation of this phenomenon is provided in Section 4.

DISPLAYFORM0

As pointed out in Section 1, the core idea of GRSVNet is to self-validate the geometry using a consistent validation loss.

To contextualize this idea, we study a particular case, OLE-GRSVNet, where the regularized feature geometry is orthogonal low-rank subspaces, and the validation loss is defined by the principal angles between the validation features and the subspaces.

The OLE loss was originally proposed by Qiu & Sapiro (2015) .

Consider a K-way classification problem.

DISPLAYFORM0 Let X c denote the submatrix of X formed by inputs of the c-th class.

Qiu & Sapiro (2015) proposed to learn a linear transformation T : R d → R d that maps data from the same class X c into a low-rank subspace, while mapping the entire data X into a high-rank linear space.

This is achieved by solving: DISPLAYFORM1 where · * is the matrix nuclear norm, which is a convex lower bound of the rank function on the unit ball in the operator norm (Recht et al., 2010) .

The norm constraint T 2 = 1 is imposed to avoid the trivial solution T = 0.

It is proved by Qiu & Sapiro (2015) that the OLE loss (1) is always nonnegative, and the global optimum value 0 is obtained if TX c ⊥TX c , ∀c = c .

BID8 later used OLE loss as a data-dependent regularization for deep learning.

Given a baseline DNN that maps a batch of inputs X into the features Z = Φ(X; θ), the OLE loss on Z is DISPLAYFORM2 The OLE loss is later combined with the standard softmax loss for training, and we will henceforth call such network "softmax+OLE." Softmax+OLE significantly improves the generalization performance, but it suffers from two problems because of the inconsistency between the softmax loss and the OLE loss: First, the learned features no longer exhibit the desired geometry of orthogonal low-rank subspaces.

Second, as will be shown in Section 4, softmax+OLE is still capable of memorizing random data or random labels, i.e., it has not reduced the memorizing capacity of DNNs.

We will now explain how to incorporate OLE loss into the GRSVNet framework.

First, let us better understand the geometry enforced by the OLE loss by stating the following theorem.

DISPLAYFORM0 .e., the column spaces of Z c and Z c are orthogonal.

The proof of Theorem 1, as well as those of the remaining theorems, is detailed in the Appendix.

Note that Theorem 1, which ensures that the OLE loss is minimized if and only if features of different classes are orthogonal, is a much stronger result than that by Qiu & Sapiro (2015) .

We then need to define a validation loss l v that is consistent with the geometry enforced by l g .

A natural choice would be the principal angles between the validation features and the subspaces spanned by {Z c } K c=1 .

Now we detail the architecture for OLE-GRSVNet.

Given a baseline DNN, we split every training batch X ∈ R d×|B| into two sub-batches, the geometry batch X g ∈ R d×|Bg| and the validation batch X v ∈ R d×|Bv| , which are mapped by the same baseline DNN into features Z g = Φ(X g ; θ) and DISPLAYFORM1 ) is imposed on the geometry batch to ensure span(Z For any feature z = Φ(x; θ) ∈ Z v in the validation batch, its projection onto the subspace span(Z g c ) is proj c (z) = U c U * c z. The cosine similarity between z and proj c (z) is then defined as the (unnormalized) probability of x belonging to class c, i.e., DISPLAYFORM2 where a small ε is chosen for numerical stability.

The validation loss for x is then defined as the cross entropy between the predicted distributionŷ = (ŷ 1 , . . .

,ŷ K ) T ∈ R K and the true label y ∈ {1, . . .

, K}. More specifically, let Y v ∈ R 1×|Bv| andŶ v ∈ R K×|Bv| be the collection of true labels and predicted label distributions on the validation batch, then the validation loss is defined as DISPLAYFORM3 where δ y is the Dirac distribution at label y, and H(·, ·) is the cross entropy between two distributions.

The empirical loss l on the training batch X is then defined as DISPLAYFORM4 See FIG0 for a visual illustration of the OLE-GRSVNet architecture.

Because of the consistency between l g and l v , we have the following theorem: Theorem 2.

For any λ > 0, and any geometry/validation splitting of X = [X g , X v ] satisfying X v contains at least one sample for each class, the empirical loss function defined in (5) is always nonnegative.

l(X, Y) = 0 if and only if both of the following conditions hold true: Figure 2: Training and testing accuracy of different networks on the SVHN dataset with random labels or random data (Gaussian noise).

Note that softmax, sotmax+wd, and softmax+OLE can all perfectly (over)fit the random training data or training data with random labels.

However, OLE-GRSVNet refuses to fit the training data when there is no intrinsically learnable patterns.• The features of the geometry batch belonging to different classes are orthogonal, i.e., Moreover, if l < ∞, then rank(span(Z g c )) ≥ 1, ∀c, i.e., Φ(·; θ) does not trivially map data into 0.

DISPLAYFORM5 Remark: The requirement that λ > 0 is crucial in Theorem 2, because otherwise the network can map every input into 0 and achieve the minimum.

This is validated in our numerical experiments.

Before delving into the implementation details of OLE-GRSVNet, we first present two toy experiments to illustrate our proposed framework.

We use VGG-11 (Simonyan & Zisserman, 2014) as the baseline architecture, and compare the performance of the following four DNNs: (a) The baseline network with a softmax classifier (softmax).

(b) VGG-11 with weight decay (softmax+wd).

(c) VGG-11 regularized by penalizing the softmax loss with the OLE loss (softmax+OLE) (d) OLE-GRSVNet.

We first train these four DNNs on the Street View House Numbers (SVHN) dataset with the original data and labels without data augmentation.

The test accuracy and the PCA embedding of the learned test features are shown in FIG0 .

OLE-GRSVNet has the highest test accuracy among the comparing DNNs.

Moreover, because of the consistency between the geometric loss and the validation loss, the test features produced by OLE-GRSVNet are even more discriminative than softmax+OLE: features of the same class reside in a low-rank subspace, and different subspaces are (almost) orthogonal.

Note that in FIG0 , features of only four classes out of ten (though ideally it should be three) have nonzero 3D embedding (Theorem 2).Next, we train the same networks, without changing hyperparameters, on the SVHN dataset with either (a) randomly generated labels, or (b) random training data (Gaussian noise).

We train the DNNs for 800 epochs to ensure their convergence, and the learning curves of training/testing accuracy are shown in Figure 2 .

Note that the baseline DNN, with either data-independent or conventional data-dependent regularization, can perfectly (over)fit the training data, while OLE-GRSVNet refuses to memorize the training data when there are no intrinsically learnable patterns.

In another experiment, we generate three classes of one-dimensional data in R 10 : the data points in the i-th class are i.i.d.

samples from the Gaussian distribution with the standard deviation in the i-th coordinate 50 times larger than other coordinates.

Each class has 500 data points, and we randomly shuffle the class labels after generation.

We then train a multilayer perceptron (MLP) with 128 neurons in each layer for 2000 epochs to classify these low dimensional data with random labels.

We found out that only three layers are needed to perfectly classify these data when using a softmax classifier.

However, after incrementally adding more layers to the baseline MLP, we found out that OLE-GRSVNet still refuses to memorize the random labels even for 100-layer MLP.

This further suggests that OLE-GRSVNet refuses to memorize training data by brute force when there is no intrinsic patterns in the data.

A visual illustration of this experiment is shown in the Appendix.

We provide an intuitive explanation for why OLE-GRSVNet can generalize very well when given true labeled data but refuses to memorize random data or random labels.

By Theorem 2, we know that OLE-GRSVNet obtains its global minimum if and only if the features of every random training batch exhibit the same orthogonal low-rank-subspace structure.

This essentially implies that OLEGRSVNet is implicitly conducting O(N |B| )-fold data augmentation, where N is the number of training data, and |B| is the batch size, while conventional data augmentation by the manipulation of the inputs, e.g., random cropping, flipping, etc., is typically O(N ).

This poses a very interesting question: Does it mean that OLE-GRSVNet can also memorize random data if the baseline DNN has exponentially many model parameters?

Or is it because of the learning algorithm (SGD) that prevents OLE-GRSVNet from learning a decision boundary too complicated for classifying random data?

Answering this question will be the focus of our future research.

Most of the operations in the computational graph of OLE-GRSVNet FIG0 ) explained in Section 3 are basic matrix operations.

The only two exceptions are the OLE loss (Z g → l g ((Z g ))) and the SVD (Z g → (U 1 , . . .

, U K )).

We hereby specify their forward and backward propagations.

According to the definition of the OLE loss in (2), we only need to find a (sub)gradient of the nuclear norm to back-propagate the OLE loss.

The characterization of the subdifferential of the nuclear norm is explained by Watson (1992) .

More specifically, assuming m ≥ n for simplicity, let U ∈ R m×m , Σ ∈ R m×n , V ∈ R n×n be the SVD of a rank-s matrix A. DISPLAYFORM0 ) ] be the partition of U, V, respectively, where U (1) ∈ R m×s and V (1) ∈ R n×s , then the subdifferential of the nuclear norm at A is: DISPLAYFORM1 where · 2 is the spectral norm.

Note that to use (6), one needs to identify the rank-s column space of A, i.e., span(U (1) ) to find a subgradient, which is not necessarily easy because of the existence of numerical error.

BID8 intuitively truncated the numerical SVD with a small parameter chosen a priori to ensure the numerical stability.

We show in the following theorem using the backward stability of SVD that such concern is, in theory, not necessary.

DISPLAYFORM2 and δU 2 , δV 2 , δA 2 are all O(ε), where ε is the machine error.

If rank(A) = s ≤ n, and the smallest singular value DISPLAYFORM3 However, in practice we did observe that using a small threshold (10 −6 in this work) to truncate the numerical SVD can speed up the convergence, especially in the first few epochs of training.

With the help of Theorem 3, we can easily find a stable subgradient of the OLE loss in (2).

Unlike the computation of the subgradient in Theorem 3, we have to threshold the singular vectors of Z g c , because the desired output U c should be an orthonormal basis of the low-rank subspace span(Z g c ).In the forward propagation, we threshold the singular vectors U c such that the smallest singular value is at least 1/10 of the largest singular value.

As for the backward propagation, one needs to know the Jacobian of SVD, which has been explained by BID9 .

Typically, for a matrix A ∈ R n×n , computing the Jacobian of the SVD of A involves solving a total of O(n 4 ) 2 × 2 linear systems.

We have not implemented the backward propagation of SVD in this work because this involves technical implementation with CUDA API.

In our current implementation, the node (U 1 , . . .

, U K ) is detached from the computational graph during back propagation, i.e., the validation loss l v is only propagated back through the path l v →Ŷ v →

Z v → θ.

Our rational is this: The validation loss l v can be propagated back through two paths: DISPLAYFORM0 The first path will modify θ so that Z v c moves closer to U c , while the second path will move U c closer to Z v c .

Cutting off the second path when computing the gradient might decrease the speed of convergence, but numerical experiments suggest that the training process is still well-behaved under such simplification.

With such simplification, the only extra computation is the SVD of a mini-batch of features, which is negligible (<5%) when compared to the time of training the baseline network.

In this section, we demonstrate the superiority of OLE-GRSVNet when compared to conventional DNNs in two aspects: (a) It has greater generalization power when trained on true data and true labels.

(b) Unlike conventionally regularized DNNs, OLE-GRSVNet refuses to memorize the training samples when given random training data or random labels.

We use similar experimental setup as in Section 4.

The same four modifications to three baseline architectures (VGG-11,16,19 (Simonyan & Zisserman, 2014) DISPLAYFORM0 The performance of the networks are tested on the following datasets:• MNIST.

The MNIST dataset contains 28 × 28 grayscale images of digits from 0 to 9.

There are 60,000 training samples and 10,000 testing samples.

No data augmentation was used.• SVHN.

The Street View House Numbers (SVHN) dataset contains 32 × 32 RGB images of digits from 0 to 9.

The training and testing set contain 73,257 and 26,032 images respectively.

No data augmentation was used.• CIFAR.

This dataset contains 32 × 32 RGB images of ten classes, with 50,000 images for training and 10,000 images for testing.

We use "CIFAR+" to denote experiments on CIFAR with data augmentation: 4 pixel padding, 32 × 32 random cropping and horizontal flipping.

All networks are trained from scratch with the "Xavier" initialization BID5 .

SGD with Nesterov momentum 0.9 is used for the optimization, and the batch size is set to 200 (a 100/100 split for geometry/validation batch is used in OLE-GRSVNet).

We set the initial learning rate to 0.01, and decrease it ten-fold at 50% and 75% of the total training epochs.

For the experiments with true labels, all networks are trained for 100, 160 epochs for MNIST, SVHN, respectively.

For CIFAR, we train the networks for 200, 300, 400 epochs for VGG-11, VGG16, VGG-19, respectively.

In order to ensure the convergence of SGD, all networks are trained for 800 epochs for the experiments with random labels.

The mean accuracy after five independent trials is reported.

The weight decay parameter is always set to µ = 10 −4 .

The weight for the OLE loss in "softmax+OLE" is chosen according to BID8 .

More specifically, it is set to 0.5 for MNIST and SVHN, 0.5 for CIFAR with VGG-11 and VGG-16, and 0.25 for CIFAR with VGG-19.

For OLE-GRSVNet, the parameter λ in (5) is determined by cross-validation.

More specifically, we set λ = 10 for MNIST, λ = 5 for SVHN and CIFAR with VGG-11 and VGG-16, and λ = 1 for CIFAR with VGG-19.

Table 1 reports the performance of the networks trained on the original data with real or randomly generated labels.

The numbers without parentheses are the percentage accuracies on the test data when networks are trained with real labels, and the numbers enclosed in parentheses are the accuracies on the training data when given random labels.

Accuracies on the training data with real labels Table 1 : Testing or training accuracies when trained on training data with real or random labels.

The numbers without parentheses are the percentage accuracies on the testing data when networks are trained with real labels.

The numbers enclosed in parentheses are the accuracies on the training data when networks are trained with random labels.

The mean accuracy after five independent trials is reported.

This suggests that OLE-GRSVNet outperforms conventional DNNs on the testing data when trained with real labels.

Moreover, unlike conventional DNNs, OLE-GRSVNet refuses to memorize the training data when trained with random labels.

(always 100%) and accuracies on the test data with random labels (always close to 10%) are omitted from the table.

As we can see, similar to the experiment in Section 4, when trained with real labels, OLE-GRSVNet exhibits better generalization performance than the competing networks.

But when trained with random labels, OLE-GRSVNet refuses to memorize the training samples like the other networks because there are no intrinsically learnable patterns.

This is still the case even if we increase the number of training epochs to 2000.We point out that by combining different regularization and tuning the hyperparameters, the test error of conventional DNNs can indeed be reduced.

For example, if we combine weight decay, conventional OLE regularization, batch normalization, data augmentation, and increase the learning rate from 0.01 to 0.1, the test accuracy of CIFAR can be pushed to 91.02%.

However, this does not change the fact that such network can still perfectly memorize training samples when given random labels.

This corroborates the claim by Zhang et al. (2017) that conventional regularization appears to be more of a tuning parameter instead of playing an essential role in reducing network capacity.

We proposed a general framework, GRSVNet, for data-dependent DNN regularization.

The core idea is the self-validation of the enforced geometry on a separate batch using a validation loss consistent with the geometric loss, so that the predicted label distribution has a meaningful geometric interpretation.

In particular, we study a special case of GRSVNet, OLE-GRSVNet, which is capable of producing highly discriminative features: samples from the same class belong to a low-rank subspace, and the subspaces for different classes are orthogonal.

When trained on benchmark datasets with real labels, OLE-GRSVNet achieves better test accuracy when compared to DNNs with different regularizations sharing the same baseline architecture.

More importantly, unlike conventional DNNs, OLE-GRSVNet refuses to memorize and overfit the training data when trained on random labels or random data.

This suggests that OLE-GRSVNet effectively reduces the memorizing capacity of DNNs, and it only extracts intrinsically learnable patterns from the data.

Although we provided some intuitive explanation as to why GRSVNet generalizes well on real data and refuses overfitting random data, there are still open questions to be answered.

For example, what is the minimum representational capacity of the baseline DNN (i.e., number of layers and number of units) to make even GRSVNet trainable on random data?

Or is it because of the learning algorithm (SGD) that prevents GRSVNet from learning a decision boundary that is too complicated for random samples?

Moreover, we still have not answered why conventional DNNs, while fully capable of memorizing random data by brute force, typically find generalizable solutions on real data.

These questions will be the focus of our future work.

It suffices to prove the case when K = 2, as the case for larger K can be proved by induction.

In order to simplify the notation, we restate the original theorem for K = 2:Theorem.

Let A ∈ R N ×m and B ∈ R N ×n be matrices of the same row dimensions, and [A, B] ∈ R N ×(m+n) be the concatenation of A and B. We have DISPLAYFORM0 Moreover, the equality holds if and only if A * B = 0, i.e., the column spaces of A and B are orthogonal.

Proof.

The inequality (8) and the sufficient condition for the equality to hold is easy to prove.

More specifically, DISPLAYFORM1 Moreover, if A * B = 0, then DISPLAYFORM2 where |A| = (A * A) 1 2 .

Therefore, DISPLAYFORM3 Next, we show the necessary condition for the equality to hold, i.e., DISPLAYFORM4 DISPLAYFORM5 | be a symmetric positive semidefinite matrix.

We DISPLAYFORM6 Let DISPLAYFORM7 be the orthonormal eigenvectors of |A|, |B|, respectively.

Then DISPLAYFORM8 Similarly, DISPLAYFORM9 Suppose that [A, B] * = A * + B * , then DISPLAYFORM10 Therefore, both of the inequalities in this chain must be equalities, and the first one being equality only if G = 0.

This combined with the last equation in FORMULA2 implies DISPLAYFORM11 APPENDIX B PROOF OF THEOREM 2Proof.

First, l is defined in equation FORMULA8 as DISPLAYFORM12 The nonnegativity of l g (Z g ) is guaranteed by Theorem 1.

The validation loss l v (Y v ,Ŷ v ) is also nonnegative since it is the average (over the validation batch) of the cross entropy losses: DISPLAYFORM13 Therefore l = l g + λl v is also nonnegative.

Next, for a given λ > 0, l(X, Y) obtains its minimum value zero if and only if both l g (Z g ) and l v (Y v ,Ŷ v ) are zeros.• By Theorem 1, l g (Z g ) = 0 if and only if span(Z g c )⊥ span(Z g c ), ∀c = c .•

According to (19), l v (Y v ,Ŷ v ) = 0 if and only ifŷ(x) = δ y , ∀x ∈ X v , i.e., for every x ∈ X v c , its feature z = Φ(x; θ) belongs to span(Z g c ).At last, we want to prove that if λ > 0, and X v contains at least one sample for each class, then rank(span(Z g c )) ≥ 1 for any c ∈ {1, . . .

, K}. If not, then there exists c ∈ {1, . . .

, K} such that rank(span(Z g c )) = 0.

Let x ∈ X v be a validation datum belonging to class y = c. The predicted probability of x belonging to class c is defined in (3): DISPLAYFORM14 Thus we have DISPLAYFORM15

@highlight

we propose a new framework for data-dependent DNN regularization that can prevent DNNs from overfitting random data or random labels.