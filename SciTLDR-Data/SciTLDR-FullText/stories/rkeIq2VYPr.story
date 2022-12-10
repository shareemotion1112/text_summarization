Determinantal point processes (DPPs) is an effective tool to deliver diversity on multiple machine learning and computer vision tasks.

Under deep learning framework, DPP is typically optimized via approximation, which is not straightforward and has some conflict with diversity requirement.

We note, however, there has been no deep learning paradigms to optimize DPP directly since it involves matrix inversion which may result in highly computational instability.

This fact greatly hinders the wide use of DPP on some specific objectives where DPP serves as a term to measure the feature diversity.

In this paper, we devise a simple but effective algorithm to address this issue to optimize DPP term directly expressed with L-ensemble in spectral domain over gram matrix, which is more flexible than learning on parametric kernels.

By further taking into account some geometric constraints, our algorithm seeks to generate valid sub-gradients of DPP term in case when the DPP gram matrix is not invertible (no gradients exist in this case).

In this sense, our algorithm can be easily incorporated with multiple deep learning tasks.

Experiments show the effectiveness of our algorithm, indicating promising performance for practical learning problems.

Diversity is desired in multiple machine learning and computer vision tasks (e.g., image hashing (Chen et al., 2017; Carreira-Perpinán & Raziperchikolaei, 2016) , descriptor learning , metric learning (Mishchuk et al., 2017) and video summarization (Sharghi et al., 2018; Liu et al., 2017) ), in which sub-sampled points or learned features need to spread out through a specific bounded space.

Originated from quantum physics, determinantal point processes (DPP) have shown its power in delivering such properties Kulesza & Taskar, 2011b) .

Compared with other diversity-oriented techniques (e.g., entropy (Zadeh et al., 2017) and orthogonality ), DPP shows its superiority as it incorporates only one single metric and delivers genuine diversity on any bounded space Affandi et al., 2013; Gillenwater et al., 2012) .

Therefore, DPP has been utilized in a large body of diversity-oriented tasks.

In general, sample points from a DPP tend to distribute diversely within a bounded space A .

Given a positive semi-definite kernel function κ : A × A → R, the probability of a discrete point set X ⊂ A under a DPP with kernel function κ can be characterized as:

where L is a |X | × |X | matrix with entry L ij = κ(x i , x j ) and x i , x j ∈ X .

L is called L-ensemble.

Note that A is a continuous space, whereas X is finite.

In the Hilbert space associated with κ, larger determinant implies larger spanned volume, thus the mapped points tend not to be similar or linearly dependent.

DPP can be viewed from two perspectives: sampling and learning.

A comprehensive introduction to mathematical fundamentals of DPP for sampling from a discrete space can be found in .

Based on this, a line of works has been proposed (Kulesza & Taskar, 2011a; Kang, 2013; Hennig & Garnett, 2016) .

In this paper, we concentrate on learning DPPs.

In learning of DPP, the term det(L) is typically treated as a singleton diversity measurement and is extended to learning paradigms on continuous space (Chao et al., 2015; Kulesza & Taskar, 2010; Affandi et al., 2014) .

There are generally two lines of strategies to learn DPPs:

Approximation.

This type of methods is to convert DPP into a simpler format which can ease and stabilize the computation.

low-rank approximation proves powerful in easing the computational burden (Gartrell et al., 2017) , in which the gram matrix is factorized as L = BB where B ∈ n×m with m n. This decomposition can also reduce the complexity which is originally a cubic time of |L|.

Kulesza & Taskar (2011b) explicitly expressed the kernel with κ(x, y) = σ 1 σ 2 δ(x) δ(y), where σ measures the intrinsic quality of the feature and δ(·) is function mapping input x to a feature space.

In this sense, the pairwise similarity is calculated in Euclidean feature space with cosine distance.

Elfeki et al. (2019) suggest approximating a given distribution by approximating the eigenvalues of the corresponding DPP.

As such, the computation can be eased and become stable.

Following this, DPP is also applied on some visual tasks, such as video summarization (Sharghi et al., 2018) , ranking (Liu et al., 2017) and image classification (Xie et al., 2017) .

It can be noted that the approximation is not straightforward for DPP, thus cannot fully deliver the diversity property (e.g. resulting in rank-deficiency).

Direct optimization.

While the aforementioned methods optimize DPP with specific approximation, a series of efforts also seek to optimize the DPP term directly (Gillenwater et al., 2014; Mariet & Sra, 2015; Bardenet & Titsias, 2015) .

In this setting, the whole gram matrix L corresponding to the pairwise similarity among features is updated directly, which allows accommodating more flexible feature mapping functions rather than an approximation.

Gillenwater et al. (2014) proposed an Expectation-Maximization algorithm to update marginal kernel DPP K = L(L + I) −1 , together with a baseline K-Ascent derived from projected gradient ascent (Levitin & Polyak, 1966) .

Mariet & Sra (2015) extended DPP from a fixed-point perspective and Bardenet & Titsias (2015) proposed to optimize DPP upon a lower bound in variational inference fashion.

A key problem of such line of works is that the computation is not differentiable, making it difficult to be used in deep learning frameworks.

To the best of our knowledge, there is no previous method incorporating DPP as a feature-level diversity metric in deep learning.

A key difficulty in doing so is that the calculation of the gradient of det(L) involves matrix inversion, which can be unstable and inaccurate in GPUs.

Though KAscent seems to be a naive rule, it still needs explicit matrix inversion in the first step before the projection procedure.

This fact greatly hinders the tight integration of DPP with deep networks.

Some alternative methods seek to reach diversity under more constrained settings.

For example, resorted to a global pairwise orthogonality constraint in hyper-sphere and Zadeh et al. (2017) employed statistical moments to measure the diversity.

However, compared with DPP, such measurements are unable to fully characterize diversity in an arbitrary bounded space.

In this paper, rather than providing more efficient DPP solvers, we concentrate on delivering a feasible feature-level DPP integration under the deep learning framework.

To this end, we revisit the spectral decomposition of DPP and propose a sub-gradient generation method which can be tightly integrated with deep learning.

Our method differs from either approximation or direct optimization by introducing a "differentiable direct optimization" procedure, thus can produce genuinely diverse features in continuous bounded space.

Our method is stable and scalable to the relatively large dataset with a specific mini-batch sampling strategy, which is verified by several experiments on various tasks.

Notations: Bold lower case x and bold upper case K represent vector and matrix, respectively.

det(·) and Tr(·) calculate the determinant and trace of a matrix, respectively.

A ⊗ B is the element-wise product of matrices A and B. |X | and |x| measure the cardinality of a finite set X and the L 2 length of a vector x, respectively.

x, y calculates the inner product of the two vectors.

x = diag(X) transforms a diagonal matrix X into its vector form x, and vice versa.

We refer "positive semi-definite" and "positive definite" to PSD and PD, respectively.

Denote the real numbers.

2.1 DETERMINANTAL POINT PROCESS L-ensemble expression of DPP requires L to be PSD, whereas kernel expression further constrains K < I (each eigenvalue of K is less than 1).

A conversion from L to K can thus be written as

, which is the marginal normalization constant given a specific L. While there is always conversion from L to K, the inverse may not exist .

In practice, one may construct L-ensemble first, then normalize it into a marginal kernel.

This fact may give rise to the difficulty of deep networks.

Since a conversion from K to L might not exist, the network needs carefully adjusting the gradients under specific constraints to ensure the updated L to be valid.

As L and K share the same eigenvectors v i , a pair of L and K holds the relation:

where λ i is the ith eigenvalue.

It is seen that such conversion is not straightforward to be directly integrated with deep learning framework.

Therefore, we optimize ensemble L directly in this paper.

We briefly introduce Gaussian kernel in this section, which works on Hilbert space with infinite dimension.

Mercer's theorem Friedman et al. (2001) ensures the PSD properties when constructing new kernels with existing ones under a specific procedure.

Such procedure is also employed in multiple kernel learning paradigms (Affandi et al., 2014; Kulesza & Taskar, 2011b; Chao et al., 2015) , which is out of the scope of this paper.

A Gaussian kernel is defined as κ(

where σ is a controlling parameter.

Thus an L-ensemble matrix becomes

According to the definition, L ii = 1 and for any element in the matrix we have L ij ∈ (0, 1].

With Gaussian kernel, we have a nice property 0 ≤ det(L) ≤ 1.

This can be easily verified by applying geometric inequality to the eigenvalues of L. Although not tight, this property shows that the determinant value with Gaussian kernel is bounded.

This fact inspires one version of our algorithm detailed in the next section.

Throughout this paper, our discussion is based on the Gaussian kernel unless specified.

Given vectorized inputs I i ∈ R h where i = 1, ..., n, our goal is to learn a map f such that the features x i = f (I i ) can spread out within a bounded feature space x i ∈ S. Hereafter we refer space to an Euclidean bounded space (e.g., [−1, 1] d ) without loss of generality.

Given any loss function J, the chain rule of gradient involving DPP is written as:

where X refer to the features before DPP layer.

While calculating ∂J/∂ det(L) and ∂L/∂X is straightforward, the main difficulty lies on the calculation of ∂ det(L)/∂L. We will discuss the calculation of this term under two case: 1) When the inversion L −1 can be stably obained, we will derive the gradient of DPP det(L) on Sec 3.1; When L is not invertible or L −1 is difficult to calculate, we give the procedure to handle the case by generating valid sub-gradient in Sec 3.2.

Since our objective is to diverse features, det(L) will serve as a (partial) objective term to be directly maximized.

With kernel κ, a DPP regularization term seeks to maximize the possibility of a feature configuration x i , i = 1, ..., n. As this possibility is proportional to det(L), the objective is max det(L).

This can become a regularization term where diversity is required.

Thus with a general loss function L G , our aim is to solve min L G − λ 1 det(L), with the controlling parameter λ 1 ≥ 0.

For the time being, we assume that kernel matrix L is invertible (we will discuss the case when L is not invertible in the next section), hence L −1 exists.

Without loss of generality, we discuss the gradient of the determinant equipped with Gaussian kernel.

For other kernels the derivation is analogous.

According to Eq (??), L ij can be further factorized as:

where x ij is the jth dimension of feature x i .

Using chain rule, the derivative of det(L) w.r.t.

x il can be written as:

where on the ijth position of

the corresponding element is:

Eq (6) can be more compactly expressed as:

where M (il) is such a matrix that, except for the ith column and row, all resting elements are 0s.

Besides, the ijth and jith elements of

.

In summary, Eq (5) can be simplified as:

To ease the computation and fully utilize the chain rule in deep learning architecture, we peel the DPP loss into two layers, and the corresponding gradient product can be expressed as:

While we can use existing package to obtain ∂L/∂x reliably, the way to stably calculate ∂ det(L)/∂L becomes essential.

We will detail in the next section once the term is hard to calculate.

The calculation of the gradient ∂ det L/∂L involves computing the inverse matrix L −1 .

However, the kernel matrix L is not always invertible.

This situation happens iff there exists at least a pair of features x i and x j such that x i = x j .

In this case, there exist two identical columns/rows of L and the 0 eigenvalue results in the non-invertibility.

This phenomenon is sometimes caused by Relu function, which can map different input values onto an identical one.

Even when all features are distinct, the numerical precision (typically on float number in GPU) may also lead to failure.

We occasionally observed that GPU calculation of L −1 reports error even no eigenvalue is 0.

One may imagine a naive replacement of matrix inverse with the pseudo-inverse, which can be applied on singular matrices.

However, pseudo-inverse will keep the zero eigenvalues intact (still rank-deficiency), and the back-propagated gradient will play no part to increase the determinant value (both 0 before and after updates).

To address this, we first diverge to consider the objective of DPP max det(L).

Since DPP term seeks to maximize the determinant, for a configuration

can be a valid ascending direction.

Thus we give the following definition: We see if a proper sub-gradientL can be found at det(L) = 0, back-propagation procedure in deep learning can consequently perform calculation usingL. To obtain suchL, we first note that L can be eigen-decomposed as following since it is symmetric and PSD:

where U is the orthogonal eigenvector matrix and Λ's diagonal elements are the corresponding eigenvalues.

As L has zero eigenvalues, the rank of Λ is lower than the dimension of L. We sort all eigenvalue into descending order to k = (σ 1 , ..., σ q , 0, ..., 0), where q < n. We then employ a simple yet effective amplification procedure by amplifying any eigenvalue smaller than ∆ to ∆. The amplified eigenvalues are nowk = (σ 1 , ..., σ s , ∆, ..., ∆), where s ≤ q. Let the diagonalized amplified eigenvalue matrix beΛ (w.r.t.

k), then the modified matrix with small positive determinant can be written as:L

For any > 0, we can choose a sufficiently small ∆ such that det(L) < .

Thus the continuity of this procedure is guaranteed.

The differenceL =L − L can be viewed as a proper ascending direction w.r.t.

L, as by addingL, det(L +L) becomes above 0 as well as arbitrarily small.

It is trivial to prove thatL is a sub-gradient on a neighbor of L, thusL is also a proper sub-gradient sufficing Definition 3.1.

This procedure is summarized in Algorithm 1 and is termed as DPPSG.

Intuitively, once encountering an identical or too close feature pair x i and x j , this procedure tries to enhance the diversity by separating them apart from each other.

Inspired by geometric inequality, we provide an improved version of the algorithm taking into account the property of Gaussian kernel.

First it easy to show that the function i σ i is concave in the feasible set i σ i = n (diagonal of Gaussian gram matrices are 1s, thus trace is n) and the maximal objective is reached out iff σ i = 1.

Therefore, any point b = (1 − θ)(σ 1 , ..., σ n ) + θ(1, ..., 1) will increase the objective i σ i .

By letting θ being a small value, the proper sub-gradient becomes Udiag(b − σ)U , where σ = (σ 1 , ..., σ n ).

This version of update differs from DPPSG as it generates sub-gradients under geometric constraints.

The method is summarized in Algorithm 2 and is termed as DPPSG*.

During implementation, the irregularity of L is examined to determine whether to adopt a normal back-propagation (in Sec 3.1) or sub-gradient (in Sec 3.2).

This can be done by verifying if the determinant value in the forward pass is less than a pre-defined small value β.

This proper subgradient based back-propagation method can be used to integrate to deep learning framework with other objectives involving matrix determinant.

We emphasize that our method is different from the line of gradient-projection based methods, such as K-Ascent.

While projection-based methods calculate the true gradient then project it back to a feasible set, our methods generate proper subgradient directly.

Without explicitly computing matrix inversion, sub-gradients, in this case, is more feasible for deep learning framework.

We employed a balanced sampling strategy for each mini-batch.

Assuming the batch size is n and there are c classes in total, in each mini-batch the distribution of samples generally follows the whole training sample distribution on c classes.

This strategy is considered to utilize the intrinsic diversity of the original data.

Besides, mini-batch sampling can constrain the overhead of DPP computation depending only on the batch size, which can be viewed as a constant in practice.

Practically, the features are always required to lie in a bounded space.

This is essential in some applications as a bounded space is more controllable.

Especially, sometimes one may demand that the features should suffice to a pre-defined distribution P. This bounding requirement is crucial to the objective of DPP since maximizing determinant tends to draw feature points infinitely apart from each other.

A naive method to achieve this is to truncate the features or using barrier functions.

However, these methods will result in irregularly dense distribution on the learned feature space boundary.

To overcome this issue, we employ Wasserstein GAN (WGAN) Arjovsky et al. (2017) to enforce the features mapped to a specific distribution P. As we do not focus on WGAN in this paper, readers are referred to Arjovsky et al. (2017) for more details.

To this end, we randomly sample n 1 pointsx i from the distribution P under balanced sampling, which are treated as positive samples.

The generator f (·) takes a feature as input and outputs the corresponding embedding.

Denote the discriminator h(·) (which is also the mapping from input to feature).

Then the WGAN loss for discriminator is:

According to the Arjovsky et al. (2017), we incorporate the generator loss

In this section, we conduct two experiments.

One is about metric learning and image hashing on MNIST and CIFAR to verify the effectiveness of our method, while another is for local descriptor retrieval task based on HardNet (Mishchuk et al., 2017) .

For the first test.

MNIST This simple dataset is suitable to reveal the geometric properties of the features on various tasks.

We test the image retrieval task equipped with contrastive loss

where L(i) indicates the label of the ith feature and x i is the learnt feature.

We employ a simple network structure for MNIST.

This network consists of 3 convolutional layers (Conv) followed by 2 fully connected layers (FC).

Batch normalization (Ioffe & Szegedy, 2015) is applied on each layer.

The filter number of each Conv are 32, 32 and 64, respectively.

The sizes of the filter are identically 5 × 5.

For the first Conv, we employ maxpooling.

For the other 2 Convs, average pooling is adopted.

The dimensions of the last FCs are 200 and 2 (for 2D visualization).

The performance can be found in Table 1 and the feature distribution is visualized in Figure 1 .

From Table 1 , it is observed that the performance on retrieval task can be enhanced by adding the DPP and WGAN regularization terms.

We see that DPP term can enhance the retrieval performance by avoiding feature points from concentrating too much.

In this sense, the learned map around the separating boundary can be much smoother.

As retrieval task typically requires the existence of top-k inter-class samples rather than concentrating property, the DPP term is more preferable.

In Figure  1 (c), we see that the feature points generally fall into the pre-defined space [−1, 1] 2 .

The utility of such space is high without sacrificing the retrieval performance.

Typically, DPPSG* is slightly superior to DPPSG.

Thus in the following test, we only report the performance under DPPSG* setting (termed as DPP* for short).

We conduct image hashing on CIFAR-10 which seeks to produce binary code for images.

To this end, we follow the binary hashing code generation procedure in Lin et al. (2015) which is activated by a Sigmoid function.

The number of neurons in the second last layer equals to the number of bits of the hashing codes.

It is anticipated that DPP regularization can enhance the utility in binary code space since the code can spread out 2 .

We test two lengths of binary code (12 and 16).

We visualize the 16-bit feature distribution using TSNE (Maaten & Hinton, 2008) in Fig. 2 (a) and (b) , and the binary code histogram comparison in (c).

The quantitative results are summarized in Table 2 .

As Lin et al. (2015) jointly solve binary code generation and classification, we report both retrieval performance (mAP) and classification performance (Acc).

We see our method can significantly enhance the binary space utility while keeping the performance almost intact.

We employ all the convolutional layers in VGG-19 (Simonyan & Zisserman, 2014) as the base and discard its final fully connected layers.

Thus the output size of this base VGG-19 network is 1 × 1 × 512.

We concatenate 3 fully connected layers with ReLU activation on each after that with dimensions 512, 100 and 20, respectively.

Contrastive loss is applied on the 20-dimensional space.

We train the whole network from scratch.

Aside from mAP, we also report top-k average precision (Precision-k) and the Wasserstein distance to the pre-defined distribution (Gap to P).

The performance on coarse (20 classes) and fine (100 classes) levels can be found in Table 3 .

In either setting, we see that DPP+WGAN significantly outperform the baseline.

Thus we infer that the DPP term can serve as a regularization not only for the feature itself but also for the smoothness of the mapping.

Since the DPP term avoids the features from concentrating too much, the learned mapping should also be from a smoother function family.

Batch size VS.

performance We study how batch size influences the performance with DPP regularization.

To this end, we report the performance on CIFAR-100 100-class retrieval with different batch sizes.

The results are shown in Table 2 .

Generally, with larger batch size, the algorithm can reach out better mAP.

We note the computational efficiency of DPP sub-gradients is high, which adds very slight overhead (even with 500 batch size) to each iteration of common back-propagation under contrastive loss, which can be neglected.

Precision-k (%) Gap to P k 10 20 50 10 20 50 On coarse (20) Table 4 : Image hashing on CIFAR-10.

"Acc" is the classification accuracy.

This test utilizes the UBC Phototour dataset (Brown & Lowe, 2007) , which consists of three subsets (Liberty, Notre Dame, and Yosemite) with around 400k 64 × 64 local patches for each.

We follow the protocol in Mishchuk et al. (2017) to treat two subsets as the training set and the third one as the testing set.

As each pair of matched image patches includes only two patches, there is no need to apply balanced sampling in this test.

We simply add DPP regularization term to the objective of state-of-the-art algorithm HardNet (Mishchuk et al., 2017) .

The batch size is 512.

We report FPR (false positive rate) and FDR (false discovery rate) following Mishchuk et al. (2017); Han et al. (2015) .

Results are summarized in Table 5 .

Several baselines are selected for comparison (i.e. SIFT (Lowe, 1999) , MatchNet (Han et al., 2015) , TFeat-M (Balntas et al., 2016) , L2Net (Tian et al., 2017) and HardNet (Mishchuk et al., 2017) ).

As the authors improved HardNet after the NeurIPS submission, we also compare with the latest version (termed as HardNet+).

We only conduct our method under DPPSG* setting and name our method HardDPP.

We see that with DPP regularization, the performance of HardNet can be further enhanced.

Note that in HardNet there is no WGAN integrated as the mapped features lie in the surface of a hyper unit sphere.

While the sampling strategy of HardNet emphasizes the embedding behavior near the margin, DPP regularization can further focus on global feature distribution.

Table 5 : Performance of UBC Phototour comparison.

Notre, Yose and Lib are short for "Notre Dame", "Yosemite" and "Liberty", respectively.

Following HardNet Mishchuk et al. (2017) , we report FPR at true positive rate at 95%.

The best results are in bold.

In this paper, we investigated the problem of learning diverse features via a determinantal point process under deep learning framework.

To overcome the instability in computing the gradient which involves the matrix inverse, we developed an efficient and reliable procedure called proper spectral sub-gradient generation.

The generated proper sub-gradient can replace the true gradient and performs well in applications.

We also considered how to constrain the features into a bounded space, since in such a way one can ensure the behavior of the network more predictable.

To this end, we further incorporated Wasserstein GAN into our framework.

Together, DPP+WGAN showed significant performance on both some common criteria and feature space utility.

A APPENDIX

MNIST Some parameters are set as follows: α = 5, λ 1 = 10 3 , λ 2 = 10 6 , margin µ = 0.8, variance for Gaussian kernel σ = 0.2 and ∆ = 10 −7 .

During the training, the batch size is set to 200.

In each iteration of DPP and WGAN training, we uniformly sample 2, 000 adversarial points from the space [−1, 1] 2 .

We adopt RMSprop and the learning rate is 10 −4 for all tests.

In the testing stage, we sample 2, 000 points from [−1, 1] 2 and calculate the Wasserstein distance with all the testing samples.

This procedure is conducted 10 times and the mean distance is reported.

CIFAR-10 image hashing The parameters in the hashing related experiments are used as following: variance for Gaussian Kernel σ = 2, the coefficients for the loss term of DPP is λ 1 = 10 2 and for the loss term of discriminator and generator in WGAN is 10 and 1 respectively.

The batch size is set to 500 and the learning rate is initialized to 0.01 with a changing rate of 0.1 at every 150 epoch.

The total number of epoch is set to 350 and we adopt the Adam optimizer to update our model.

The parameter setting is as follows: α = 1, λ 1 = 10 3 , λ 2 = 10 3 , margin µ = 0.8, variance for Gaussian kernel σ = 0.2 and ∆ = 10 −6 .

The rest of the settings are the same as those of MNIST test.

A.2 CRITERIA PRECISION-k AND MAP-k For image retrieval task, we adopt the top-k mean average precision (abbreviated as mAP-k) to evaluate the performance.

We also present the top-k average precision (abbreviated as Precision-k), which is calculated as:

where b is the corresponding class and I is the indicator function:

Thus mAP-k is the reweighted version of Precision-k:

A.3 OVERHEAD OF DPP

Calculating SVD or matrix inversion on a large number of features can be time consuming.

However, in our setting, we employed a common practice in deep learning -mini-batch -to avoid such computation on a whole batch.

We can conclude that mini-batch strategy can limit the computational cost such that the extra overhead of DPP is only dependent on the batch size (thus other parts of the networks have no impact on this overhead).

Therefore, although the complexity of our method is O(n 3 ), n only corresponds to the batch size rather than whole sample number in our setting, which is much more manageable in practice.

We report the average overhead comparison on CIFAR-10 hashing task with varying batch sizes (100, 200, 250, 400 and 500) on a GTX 1080 GPU as in Table  6 Table 6 : Overhead of a single batch and a DPP calculation on CIFAR-10 hashing task with varying batch size.

Time is in seconds.

where "overhead-all" and "overhead-DPP" refer to the average time cost (s) for a single batch on all the computation and only DPP computation (both forward and backward), respectively.

We can conclude that, compared to other computation, the extra overhead of DPP is small (even in a simple network as CIFAR-10 hashing).

Besides, a batch size up to 500 is considered to be sufficient in most of the applications.

In practice, we did not employ any trick to reduce such overhead (since it is out of the papers focus) but simply utilized standard functions provided by Pytorch.

For MNIST verification test, we employed a simple backbone.

The structure of the backbone is {conv 1(5 × 5)+maxpool+conv 2(5 × 5)+avepool+conv 3(5 × 5)+avepool+fully con1(200-d)+fully con2(2-d)+fully con3(10-d)+contrastive loss} .

We add DPP and WGAN regularization to the features at "fully con2" layer, which is 2-dimensional thus better for visualization.

For CIFAR-10 image hashing task, we employ the same network structure as a high-cited method DCH (Lin et al., 2015) .

DPP and WGAN loss is applied on the second last fully connected layer (the dimension of this layer corresponds to the length of digits in the hashing code).

For CIFAR-100 metric learning task, we employ all the convolutional layers of VGG19, concatenated with 3 fully connected layers (with 512, 100 and 20 dimension).

DPP and WGAN loss, together with contrastive loss, is applied on the final fully connected layer (20-dimension).

The network is trained from scratch without any pre-training.

A.5 DEGRADATION ON CIFAR-10 IMAGE HASHING For the performance degradation with DPP on hashing task, we can take Figure 2 (c) as an example to explain.

We see that original DCH features concentrate on several digits (generally 10 digits corresponding to 10 classes), while DPP features diffuse to almost the whole discrete space.

In this sense, if one retrieves the k-th closest hashing code, DCH can find the hashing code with a small searching radius.

However, one has to greatly enlarge the search radius for k-th closest code in DPP feature space since the distribution is much more even.

In this sense, DPP will inevitably causes degradation since large searching radius will more likely to reach a code in other class.

Therefore, we think "utility vs mAP" is an intrinsic conflict and needs to reach a trade-off.

@highlight

We proposed a specific back-propagation method via proper spectral sub-gradient to integrate determinantal point process to deep learning framework.