In this paper, we propose a novel kind of kernel, random forest kernel, to enhance the empirical performance of MMD GAN.

Different from common forests with deterministic routings, a probabilistic routing variant is used in our innovated random-forest kernel, which is possible to merge with the CNN frameworks.

Our proposed random-forest kernel has the following advantages: From the perspective of random forest, the output of GAN discriminator can be viewed as feature inputs to the forest, where each tree gets access to merely a fraction of the features, and thus the entire forest benefits from ensemble learning.

In the aspect of kernel method, random-forest kernel is proved to be characteristic, and therefore suitable for the MMD structure.

Besides, being an asymmetric kernel, our random-forest kernel is much more flexible, in terms of capturing the differences between distributions.

Sharing the advantages of CNN, kernel method, and ensemble learning, our random-forest kernel based MMD GAN obtains desirable empirical performances on CIFAR-10, CelebA and LSUN bedroom data sets.

Furthermore, for the sake of completeness, we also put forward comprehensive theoretical analysis to support our experimental results.

Generative adversarial nets (GANs; Goodfellow et al., 2014) are well-known generative models, which largely attribute to the sophisticated design of a generator and a discriminator which are trained jointly in an adversarial fashion.

Nowadays GANs are intensely used in a variety of practical tasks, such as image-to-image translation (Tang et al., 2019; Mo et al., 2019) ; 3D reconstruction (Gecer et al., 2019) ; video prediction (Kwon & Park, 2019) ; text-to-image generation (Zhu et al., 2019) ; just to name a few.

However, it's well-known that the training of GANs is a little tricky, see e.g. (Salimans et al., 2016) .

One reason of instability of GAN training lies in the distance used in discriminator to measure the divergence between the generated distribution and the target distribution.

For instance, concerning with the Jensen-Shannon divergence based GANs proposed in Goodfellow et al. (2014) , points out that if the generated distribution and the target distribution are supported on manifolds where the measure of intersection is zero, Jensen-Shannon divergence will be constant and the KL divergences be infinite.

Consequently, the generator fails to obtain enough useful gradient to update, which undermines GAN training.

Moreover, two non-overlapping distributions may be judged to be quite different by the Jensen-Shannon divergence, even if they are nearby with high probability.

As a result, to better measure the difference between two distributions, Integral Probability Metrics (IPM) based GANs have been proposed.

For instance, utilizes Wasserstein distance in GAN discriminator, while Li et al. (2017) adopts maximum mean discrepancy (MMD), managing to project and discriminate data in reproducing kernel Hilbert space (RKHS).

To mention, the RKHS with characteristic kernels including Gaussian RBF kernel (Li et al., 2017) and rational quadratic kernel (Bińkowski et al., 2018) has strong power in the discrimination of two distributions, see e.g. (Sriperumbudur et al., 2010) .

In this paper, inspired by non-linear discriminating power of decision forests, we propose a new type of kernel named random-forest kernel to improve the performance of MMD GAN discriminator.

In order to fit with back-propagation training procedure, we borrow the decision forest model with stochastic and differentiable decision trees from Kontschieder et al. (2015) in our random-forest kernel.

To be specific, each dimension of the GAN discriminator outputs is randomly connected to one internal node of a soft decision forest, serving as the candidate to-be-split dimension.

Then, the tree is split with a soft decision function through a probabilistic routing.

Other than the typical decision forest used in classification tasks where the value of each leaf node is a label, the leaf value of our random forest is the probability of a sample x i falling into a certain leaf node of the forest.

If the output of the discriminator is denoted as h θ N (x i ) and the probability output of the t-th tree is denoted as µ t (h θ N (x i ); θ F ), the random forest kernel k RF can be formulated as

where T is the total number of trees in the forest, θ N and θ F denote the parameters of the GAN discriminator and the random forest respectively.

Recall that random forest and deep neural networks are first combined in Kontschieder et al. (2015) , where differentiable decision tree model and deep convolutional networks are trained together in an end-to-end manner to solve classification tasks.

Then, Shen et al. (2017) extends the idea to label distribution learning, and Shen et al. (2018) makes further extensions in regression regime.

Moreover, Zuo & Drummond (2017) , Zuo et al. (2018) and Avraham et al. (2019) also introduce deep decision forests.

Apart from the typical ensemble method that averages the results across trees, they aggregate the results by multiplication.

As for the combination of random forest and GAN, Zuo et al. (2018) introduce forests structure in GAN discriminator, combining CNN network and forest as a composited classifier, while Avraham et al. (2019) uses forest structure as one of non-linear mapping functions in regularization part.

On the other hand, in the aspect of relationship between random forest and kernel method, Breiman (2000) initiates the literature concerning the link.

He shows the fact that a purely random tree partition is equivalent to a kernel acting on the true margin, of which form can be viewed as the probability of two samples falling into the same terminal node.

Shen & Vogelstein (2018) proves that random forest kernel is characteristic.

Some more theoretical analysis can be found in Davies & Ghahramani (2014) , Arlot & Genuer (2014) , Scornet (2016) .

However, despite their theoretical breakthroughs, forest decision functions used in these forest kernels are non-differentiable hard margins rather than differentiable soft ones, and thus cannot be directly used in back propagation regime.

To the best of our knowledge, MMD GAN with our proposed random-forest kernel is the first to combine random forest with deep neural network in the form of kernel MMD GAN.

Through theoretical analysis and numerical experiments, we evaluate the effectiveness of MMD GAN with our random-forest kernel.

From the theoretical point of view, our random-forest kernel enjoys the property of being characteristic, and the gradient estimators used in the training process of random-forest kernel GAN are unbiased.

In numerical experiments, we evaluate our random-forest kernel under the setting of both the original MMD GAN (Li et al., 2017) and the one with repulsive loss (Wang et al., 2019) .

Besides, we also compare our random-forest kernel with Gaussian RBF kernel (Li et al., 2017) , rational quadratic kernel (Bińkowski et al., 2018) , and bounded RBF kernel (Wang et al., 2019) .

As a result, MMD GAN with our random-forest kernel outperforms its counterparts with respect to both accuracy and training stability.

This paper is organized as follows.

First of all, we introduce some preliminaries of MMD GAN in Section 2.

Then we review the concept of deep random forest and show how it is embedded within a CNN in 3.1.

After that, random-forest kernels and MMD GAN with random-forest kernels are proposed in 3.2 and 3.3 respectively.

Besides, the training techniques of MMD GAN with random-forest kernel are demonstrated in Section 3.4 and the theoretical results are shown in Section 3.5.

Eventually, Section 4 presents the experimental setups and results, including the comparison between our proposed random-forest kernel and other kernels.

In addition, all detailed theoretical proofs are included in the Appendices.

The generative model captures the data distribution P X , by building a mapping function G : Z → X from a prior noise distribution P Z to data space.

While the discriminative model D : X → R is used to distinguish generated distribution P Y from real data distribution P X .

Taking X, X ∼ P X and Y, Y ∼ P Y := P G (Z) where Y := G(Z) and Y := G(Z ), the squared MMD is expressed as

The loss of generator and discriminator in MMD GAN proposed in Li et al. (2017) is:

Wang et al. (2019) proposed MMD GAN with repulsive loss, where the objective functions for G and D are:

we can write an unbiased estimator of the squared MMD in terms of k as

When k is a characteristic kernel, we have MMD 2 [P X , P Y ] ≥ 0 with equality applies if and only if P X = P Y .

The best-known characteristic kernels are gaussian RBF kernel and rational quadratic kernel (Bińkowski et al., 2018) .

In this section, we review a stochastic and differentiable variant of random forest and how it is embedded within a deep convolutional neural network proposed in Kontschieder et al. (2015) .

Then we propose random-forest kernel and we apply it in MMD GAN.

We illustrate the advantages of our random-forest kernel, show the training technique of MMD GAN with random-forest kernel, and study its theoretical properties.

Suppose that a random forest consists of T ∈ N random trees.

For the t-th tree in the forest, t ∈ {1, . . . , T }, we denote N t := {d t j } |Nt| j=1 as the set of its internal nodes and if T trees have the same structure, then we have |N t | = |N |, see Figure 1 .

Furthermore, we denote L t as the set of its leaf nodes and θ t F as the parameters of the t-th tree.

Here we introduce the routing function µ t (x; θ t F ) which indicates the probability of the sample x falling into the -th leaf node of the t-th tree.

In order to provide an explicit form for the routing function µ t (x; θ t F ), e.g. the thick black line in Figure 1 , we introduce the following binary relations that depend on the tree structure: d t j is true, if belongs to the left subtree of node d ) be the decision function of the j-th internal node in the t-th tree, that is the probability of the sample x falling into the left child of node d t j in the t-th tree.

Then, µ t can be expressed by these relations as

where R t denotes the unique path from node 1 to node of the t-th tree.

Figure 1: Example of a random forest with T trees: blue nodes denote the internal nodes N := {d1, ..., d7} while purple nodes are the leaf nodes L := { 1, · · · , 8}. The black thick path illustrates the route of a sample x falling into the 6 leaf node of the t-th tree.

Now, let us derive the explicit form of the decision function p t j (x; θ t F ).

Here, to utilize the power of deep learning, we consider using the convolutional neural network to construct decision functions of random forest.

To be specific, given the parameter θ N trained from a CNN network, we denote h(·; θ N ) as the d -dimension output of a convolutional neural network, which is the unit of the last fullyconnected layer in the CNN, and h i (·; θ N ) is the i-th element of the CNN output.

We denote C : {1, . . .

, T |N |} → {1, . . . , d } as the connection function, which represents the connection between the internal node d t j and the former CNN output h i .

Note that during the whole training process, the form of the connection function C is not changed and every internal node d t j is randomly assigned to an element h C(T (t−1)+j) (·; θ).

If we choose the sigmoid function σ(x) = (1+e −x ) −1 as the decision function, and let the parameters of the t-th tree be θ

For example, we have the probability p Every leaf node in each tree has a unique road R t from node 1 to node with length |R t | = log 2 (|L t |).

Then, for the every leaf node of the t-th tree, we have

where Lf denotes the set of all left son nodes of its father node.

Here, we propose the random-forest kernel as follows:

Definition 1 (Random-Forest Kernel) Let x, y be a pair of kernel input, let θ t F = (w t , b t ) denotes the weights and bias of the t-th tree of the random forest, and θ F := (θ t F ) T t=1 .

The random-forest kernel can be defined as

where L t denotes the set of leaf nodes in the t-th tree,

We write

and introduce the objective functions of MMD GAN with random-forest kernel by

where y = G ψ (z), z is noise vector, and R is the regularizer of random-forest kernel (the detail is shown in Section 3.4).

In addition, the objective functions of MMD GAN with repulsive loss are

Random-forest kernel MMD GAN enjoys the following advantages:

• Our proposed random-forest kernel used in MMD GAN benefits from ensemble learning.

From the perspective of random forest, the output of MMD GAN discriminator h(·; θ N ) can be viewed as feature inputs to the forest.

To mention, each tree only gets access to merely a fraction of the features by random connection functions, and thus the entire forest benefits from ensemble learning.

• Our random-forest kernel MMD GAN enjoys the advantages of three powerful discriminative methods, which are CNN, kernel method, and ensemble learning.

To be specific, CNN is good at extracting useful features from images; Kernel method utilize RKHS for discrimination; Ensemble learning utilizes the power of randomness and ensemble.

• Our proposed random-forest kernel has some good theoretical properties.

In one aspect, random-forest kernel is proved to be characteristic in Shen & Vogelstein (2018) .

In another, in Section 3.5, the unbiasedness of the gradients of MMD GAN with random-forest kernel is proved.

In Frosst & Hinton (2017) , the authors mention that the tree may get stuck on plateaus if internal nodes always assign the most of probability to one of its subtree.

The gradients will vanish because the gradients of the logistic-type decision function will be very closed to zero.

In order to stabilize the training of random-forest kernel and avoid the stuck on bad solutions, we add penalty that encourage each internal node to split in a balanced style as Frosst & Hinton (2017) does, that is, we penalize the cross entropy between the desired 0.5, 0.5 average probabilities of falling into two subtrees and the actual average probability α, 1 − α.

The actual average probability of the i-th internal node α i is

, where P i (x) is the routing probability of x from root node to internal node i, p i (x) is the probability of x falling into the left subtree of the i-th internal node, and Ω is the collection of mini-batch samples.

Then, the formulation of the regularizer is:

where λ is exponentially decayed with the depth of d of the internal node by multiplying the coefficient 2 −d , for the intuition that less balanced split in deeper internal node may increase the non-linear discrimination power.

When training random-forest kernel, a mini-batch of real samples X and generated pictures Y are both fed into the discriminator, and then k(X, X), k(X, Y ) and k(Y, Y ) are calculated, where k := k RF • h(·; θ N ).

Here, to notify, we find that the Ω in the regularizer formulation does matter in forest-kernel setting.

It's better to calculate α i and R(Ω) in the case of Ω = X, Ω = Y , Ω = X ∪ Y respectively, and then sum up three parts of regularizer as final regularizer R.

Therefore, the formulation of regularizer R added in the training of random-forest kernel is

In this subsection, we present our main theoretical results.

Theorem 2 (Unbiasedness) Let X be the true data on X with the distribution P X and Z be the noise on Z with the distribution P Z satisfying E P X X α < ∞ and E P Z Z α < ∞ for some α ≥ 1.

Moreover, let G ψ : Z → X be a generator network, h θ N : X → R d be a discriminator network, k RF be the random-forest kernel, and θ D := (θ N , θ F ) be the parameters of the GAN discriminator.

Then, for µ-almost all θ D ∈ R |θ D | and ψ ∈ R |ψ| , there holds

In other words, during the training process of Random-Forest Kernel MMD GAN, the estimated gradients of MMD with respect to the parameters ψ and θ D are unbiased, that is, the expectation and the differential are exchangeable.

In this section, we evaluate our proposed random-forest kernel in the setting of MMD GAN in (Li et al., 2017) and the MMD GAN with repulsive loss (Wang et al., 2019) .

To illustrate the efficacy of our random-forest kernel, we compare our random-forest kernel with Gaussian kernel (Li et al., 2017) , rational quadratic kernel (Bińkowski et al., 2018) As is shown in Figure 3 , the shapes of the Gaussian RBF kernel and the rational quadratic kernel are both symmetric.

However, the local structure of random-forest kernel (w.r.t reference points except 70-dimensional zero vector) is asymmetric and very complex.

The asymmetry and complexity of random-forest kernel may be helpful to discriminate two distributions in MMD GAN training.

For dataset Cifar10 and dataset LSUN bedroom, DCGAN (Radford et al., 2016) architecture with hyper parameters from Miyato et al. (2018) is used for both generator and discriminator; and for dataset CelebA, we use a 5-layer DCGAN discriminator and a 10-layer ResNet generator.

Further details of the network architecture are given in Appendix A.3.

We mention that in all experiments, batch normalization (Ioffe & Szegedy, 2015) is used in the generator and spectral normalization (Miyato et al., 2018 ) is used in the discriminator.

The hyper-parameter details of kernels used in the discriminator are shown in Appendix A.1.

For the sake of comparison with forest kernel, the dimension of discriminator output layer s is set to be 70 for random-forest kernel and to be 16 for other kernels following the previous setting of Bińkowski et al. (2018); Wang et al. (2019) .

We set the initial learning rate 10 −4 and decrease the learning rate by coefficient 0.8 in iteration 30000, 60000, 90000, and 120000.

Adam optimizer (Kingma & Ba, 2015) is used with momentum parameters β 1 = 0.5 and β 2 = 0.999.

The batch size of each model is 64.

All models were trained for 150000 iterations on CIFAR-10, CelebA, and LSUN bedroom datasets, with five discriminator updates per generator update.

The following three metrics are used for quantitative evaluation: Inception score (IS) (Salimans et al., 2016) , Fréchet inception distance (FID) (Heusel et al., 2017) , and Kernel inception distance (KID) (Bińkowski et al., 2018) .

In general, higher IS and Lower FID, KID means better quality.

However, outside the dataset Imagenet, the metric IS has some problem, especially for datasets celebA and LSUN bedroom.

Therefore, for inception score, we only report the inception score of CIFAR-10.

Quantitative scores are calculated based on 50000 generator samples and 50000 real samples.

We compare our proposed random-forest kernel with mix-rbf kernel and mix-rq kernel in the setting of the MMD GAN loss, and compare our proposed random-forest kernel with rbf-b kernel in the setting with MMD GAN repulsive loss.

The Inception Score, the Fréchet Inception Distance and the Kernel Inception Distance of applying different kernels and different loss functions on three benchmark datasets are shown in table 1.

We find that, in the perspective of the original MMD GAN loss, our newly proposed random-forest kernel shows better performance than the mix-rbf kernel and the mix-rq kernel in CIFAR-10 dataset and LSUN bedroom dataset; and in the perspective of the repulsive loss, the performance of our newly proposed random-forest kernel is comparable or better than the rbf-b kernel.

The efficacy of our newly proposed random-forest kernel is shown under the setting of both MMD GAN loss and MMD GAN repulsive loss.

Some randomly generated pictures of model learned with various kernels and two different loss functions are visualized in Appendix D. The formulation of our proposed random-forest kernel is

In experiments, we take the number of trees in the forest T = 10 and the depth of trees dep = 3.

Thus, the total number of internal nodes is 70.

To notify, in general, the parameters

are trainable, where

and N is the set of every internal nodes of a tree.

However, for the sake of experimental simplification, we fix each w

where Σ = {2, 5, 10, 20, 40, 80}, and rational quadratic kernel (Bińkowski et al., 2018) with mixture of kernel scale α, that is,

where A = {0.2, 0.5, 1, 2, 5}.

In the setting of the MMD GAN with repulsive loss proposed in Wang et al. (2019), we compare our forest kernel with bounded RBF kernel (Wang et al., 2019) , that is,

with σ = 1, b l = 0.25 and b u = 4.

In figure 3 , we compare the contours of three different kernels (the detail of kernels is shown in A.1).

We directly plot the filled contours of 2-dimensional Gaussian kernel and rational quadratic kernel with reference to (0, 0); As for random-forest kernel with T = 10 and dep = 3, where the input dimension is 70, the steps of a 2-dimensional visualization are as follows:

1) We randomly generate 25000 points from the uniform distribution U[−1, 1] 70 and set (0.5, . . .

, 0.5) ∈ R 70 as the reference point.

To notify, if the reference point is 70-dimensional zero vector, the values of random-forest kernel will be constant; 2) We calculate the 25000 output values of random-forest kernel; 3) We transform 70-dimensional 25000 randomly generated points and reference point together by t-SNE to 2-dimensional points; 4) We show the filled contour of the neighborhood of transformed reference point.

We try to visualize the local structure of random-forest kernel.

In the experiments of the CIFAR-10 and LSUN bedroom datasets, we use the DCGAN architecture following Miyato et al. (2018) , and for the experiments of the CelebA dataset, we use a 5-layer DCGAN discriminator and a 10-layer ResNet generator as in Bińkowski et al. (2018) .

The first few layers of the ResNet generator consist of a linear layer and 4 residual blocks as in Gulrajani et al. (2017) .

The network architecture details are shown in Table 2 and 3.

Table 2 : Network architecture used in the image generation on CIFAR-10 dataset and LSUN bedroom dataset.

In terms of the shape parameter h and w in the table, we take h = w = 4 for CIFAR-10 and h = w = 8 for LSUN bedroom.

As for the output dimension of discriminator s, we take s = 70 for random-forest kernel and s = 16, 70 for other kernels.

To notify, 16 is the setting in Bińkowski et al.

In Section B, we will show the main propositions used to prove the Theorem 2.

To be specific, in Section B.1, we represent neural networks as computation graphs.

In Section B.2, we consider a general class of piecewise analytic functions as the non-linear part of neural networks.

In Section B.3, we prove the Lipschitz property of the whole discriminators.

In Section B.4, we discover that for P X almost surely, the network is not differential for its parameters θ N and ψ.

Fortunately, we prove that the measure of bad parameters set is zero.

In Section C, we will show the explicit proof of main propositions in Section B and Theorem 2.

Historical attempts to scale up GANs using CNNs to model images have been unsuccessful.

The original CNN architecture is made up of convolution, non-linear and pooling.

Now for our model, we adopt the deconvolution (Zeiler & Fergus, 2014) net to generate the new data with spatial upsampling.

Moreover, batch normalization (Ioffe & Szegedy, 2015) is a regular method which stabilizes learning by normalizing the input to each unit to have zero mean and unit variance.

Furthermore, relu functions are used both in generator and discriminator networks as non-linear part.

Here we avoid spatial pooling such as max-pooling and global average pooling.

Throughout this paper, we always denote by

the output of a fully connected layer, where d represents the number of neurons in the output.

The general feed-forward networks including CNN and FC can be formulated as a directed acyclic computation graph G consisting of L + 1 layers, with a root node i = 0 and a leaf node i = L. For a fixed node i, we use the following notations:

π(i): the set of parent nodes of i; j < i: j is a parent node of i.

Each node i > 0 corresponds to a function f i that receives a R d π(i) -valued input vector, which is the concatenation of the outputs of each layer in π(i), and outputs a R di -valued vector, where

According to the construction of the graph G, the feed-forward network that factorizes with functions f i recursively can therefore be defined by h 0 = X, and for all 0 < i ≤ L,

where h π(i) is the concatenation of the vectors h j , j ∈ π(i).

Here, the functions f i can be of the following different types:

) is a linear operator on the weights W i , e.g. convolutions, then the functions f i are of the linear form

(ii) Non-linear: Such functions f i including ReLU, max-pooling and ELU, have no learnable weights, can potentially be non-differentiable, and only satisfy some weak regularization conditions, see Definition 3 and the related Examples 2, 3, and 4.

In the following, we denote by I the set of nodes i such that f i is non-linear, and its complement by I c := {1, . . .

, L} \ I, that is, the set of all of all linear modules.

We write θ N as the concatenation of parameters

where |θ N | denotes the total number of parameters of θ N .

Moreover, the feature vector of the network corresponds to the output neurons G L of the last layer L and will be denoted by

where the subscript θ stands for the parameters of the network.

If X is random, we will use h θ N (X) to denote explicit dependence on X, and otherwise when X is fixed, it will be omitted.

Throughout this paper, in the construction of neural networks, we only consider activation functions that are piecewise analytic which can be defined as follows:

Definition 3 (Piecewise Analytic Functions) Let {f i } i∈I be non-linear layers in neural networks, then f i is said to be an piecewise analytic function if there exists a partition of R d π(i) with J i pieces, that is,

The sets D The following examples show that in practice, the vast majority of deep networks satisfy the conditions in the above definition, and therefore are piecewise analytic.

Example 1 (Sigmoid) Let f i outputs the sigmoid activation function on the inputs.

Then we need no partition, and hence there exist

corresponding to the whole space;

(ii) J i = 1 real analytic function f

Here, we mention that the case in Example 1 corresponds to most of the differentiable activation functions used in deep learning, more examples include the softmax, hyperbolic tangent, and batch normalization functions.

On the other hand, in the case that the functions are not differentiable, as is shown in the following examples, many commonly used activation functions including the ReLU activation function are at least piecewise analytic.

Besides the ReLU activation function, other activation functions, such as the Max-Pooling and the ELU (Clevert et al., 2016 ) also satisfy the form of Definition 3 are therefore piecewise analytic.

In this section, we investigate the Lipschitz property of the proposed discriminative networks, which is formulated as follows:

Proposition 4 Let θ D be the parameters of discriminators and B r (θ D ) ⊂ R |θ D | be the ball with center θ D and radius r ∈ (0, ∞).

Then, for all θ D ∈ B r (θ D ) and all x ∈ R d , there exists a regular function c(x) with

In the analysis of section B.3, we intentionally ignore the situation when samples fall into the "bad sets", where the network is not differentiable for data sets x with nonzero measure.

And in proposition 5, we show that the measure of the so called "bad sets" is zero.

To better illustrate this proposition, we first denote some notations as follows:

as the set of input vectors x ∈ R d such that h θ N (x) is not differentiable with respect to θ N at the point θ N,0 .

Then we call

the set of critical parameters, where the network is not differentiable for data sets x with nonzero measure.

Proposition 5 Let the set Θ P X be as in equation 8.

Then, for any distribution P X , we have µ(Θ P X ) = 0.

To prove Proposition 4, we first introduce Lemma 6 and Lemma 7.

Lemma 6 describes the growth and Lipschitz properties of general networks including convolutional neural networks and fully connected neural networks introduced in Section B.1.

Lemma 6 Let h θ N be the neural networks defined as in Section B.1.

Then there exist continuous functions a, b : R |θ N | → R and α, β :

Proof [of Lemma 6]

We proceed the proof by induction on the nodes of the network.

Obviously, for i = 0, the inequalities hold with b 0 = 0, a 0 = 1, β 0 = 0 and α 0 = 0.

Then, for the induction step, let us fix an index i. Assume that for all x ∈ R d and all

where

are some continuous functions.

Moreover, concerning with the Lipschitz property, we have

where we used the notations

(ii) If i ∈ I c , that is, i is not a linear layer, here we only consider the sigmoid and ReLU functions.

We first show that both of them are Lipschitz continuous.

Concerning the sigmoid function, σ(x) = 1 1 + e −x we obviously have

for all x ∈ R. Consequently, the sigmoid function is Lipschitz continuous with Lipschitz constant |σ| 1 := 1/4.

Next, for the ReLU function,

we have for all x ∈ R,

Therefore, the ReLU function is Lipschitz continuous with Lipschitz constant |σ| 1 := 1.

Thus, non-linear layer f i is Lipschitz continuous with Lipschitz constant M := |f i | 1 .

Consequently, by recursion, we obtain continuous functions

Next, we investigate the growth conditions and Lipschitz property of the random forest.

For the ease of notations, we write the function µ (T ) as

Moreover, it is easily seen that T · |N | equals the number of internal nodes in the random forest.

Lemma 7 Let h(x; θ N ) be the input vector of random trees and θ F := (w, b) where w and b denote the weights and bias of the random forests, respectively.

Then, for all h ∈ R d and θ F , θ F ∈ R 2T |N | , there exist continuous functions c 1 , c 2 , and constants c 3 , c 4 , c 5 such that

µ ≤ 1

Taking the summation on both sides of the above inequality with respect to all of the nodes in the random forest, that is, w.r.t.

∈ L, we obtain

where the second inequality follows from the Cauchy-Schwartz inequality and the third inequality is due to the fact that the number that the nodes p i in the random forest assigned to the corresponding node h j equals T |N |/d or T |N |/d + 1, which are less than |L|/d .

Now, we show the Lipschitz properties equation 12 and equation 13 of the random forest.

From the equivalent form equation 3 concerning the value of the leaf node, we easily see that µ can be written as a product of probability functions p i or 1 − p i .

Therefore, without loss of generality, we can assume µ are of the product form

For a fixed t = 1, . . .

, T , recall that T t denotes the connection function of the t-th random tree.

Then, the Lipschitz property of the sigmoid function and the continuously differentiability of the linear transform yield

Then, equation 14 together with equation 15 implies

where the last inequality follows from the Cauchy-Schwartz inequality.

Consequently, concerning the random forest, we obtain

where the second inequality is again due to then Cauchy-Schwartz inequality.

Analogously, for a fixed t = 1, . . .

, T and any i, we have

and consequently we obtain

and for the random forest, there holds

which completes the proof.

The next proposition presents the growth condition and the Lipschitz property of the composition of the neural network and the random forest.

Lemma 8 Let B r (θ D ) ⊂ R |θ D | be the ball with center θ D and radius r ∈ (0, ∞).

Then, for all θ D ∈ B r (θ D ), all x ∈ R d , there exist continuous functions c 6 , c 7 , c 8 , and c 9 such that

Proof [of Lemma 8] Combining equation 9 in Lemma 6 with equation 11 in Lemma 7, we obtain the growth condition of the form

Concerning the Lipschitz property, using equation 12 and equation 13, we get

where the last inequality follows from equation 9 and equation 10 established in Proposition 6.

With the concatenation θ D := (θ N , θ F ) := (θ N , w, b) we obtain

and thus the assertion is proved.

Proof [of Proposition 4]

First we give the growth conditions for the linear kernel.

Let k be the linear kernel k L (x, y) := x, y , then we have

If we denote u := (x, y), then the above linear kernel k L (x, y) can be written a as a univariate function

Therefore we have K is continuously differentiable satisfying the following growth conditions:

Let us define the function f :

Then we have f (0) = K(v) and f (1) = K(u).

Moreover, f is differentiable with derivative

The growth condition equation 18 implies

Using the mean value theorem, we obtain that for all u, v ∈ R L , there holds

Proposition 8 tells us that

holds for some continuous functions c 6 , c 7 , c 8 , and c 9 , which are also bounded by certain constant B > 0 on the ball B r (θ D ).

Some elementary algebra shows that

Since t → (1+t 1/2 ) 2 is concave on t ≥ 0, Jensen's inequality together with the moment assumption E x 2 < ∞ implies

and also E(1 + x )

< ∞ by the moment assumption.

Therefore, the regular function c(x) defined by c(

is integrable and thus our assertion is proved.

To reach the proof of Proposition 5, we first introduce Lemma 9 and Lemma 11.

Let i be a fixed node.

To describe paths through the network's computational graph, we need to introduce the following notations:

A(i) := {i | i is an ancestor of i};

Obviously, we always have ∂i ⊆ ¬i and ¬i = ∂i ∪ ¬π(i).

We define a backward trajectory starting from node i by

The set of all backward trajectories for node i will be denoted by Q(i).

Lemma 9 Let i be a fixed node in the network graph.

If θ N ∈ R |θ N | \ ∂S ¬i , then there exist a constant η > 0 and a trajectory q ∈ Q(i) such that for all θ N ∈ B η (θ N ), there holds

where f q is the real analytic function on R |θ N | with the same structure as h θ N , only replacing each nonlinear f i with the analytic function f

Proof [of Lemma 9]

We proceed by induction on the nodes of the network.

If i = 0, we obviously have h 0 θ N = x, which is real analytic on R |θ N | .

For the induction step we assume that the assertion holds for ¬π(i) and let θ N ∈ R |θ N | \ ∂S ¬i .

Then there exist a constant η > 0 and a trajectory q ∈ Q(π(i)) such that for all θ N ∈ B η (θ N ), there holds

with f q : R |θ N | → R being a real analytic function.

(i) If θ N ∈ S ∂i , then there exists a sufficiently small constant η > 0 such that B η (θ N ) ∩ S ∂i = ∅.

Therefore, there exists a j ∈ {1, . . .

, J i } such that for all θ N ∈ B η (θ N ), there holds

where f j i is one of the real analytic functions in the definition of f i .

Then, equation 19 implies that for all θ N ∈ B min{η,η } (θ N ), there holds

(ii) Consider the case θ N ∈ S ∂i .

By assumption, we have θ N ∈ ∂S ∂i .

Then there exists a small enough constant η > 0 such that B η (θ N ) ⊂ S ∂i .

If we denote

Now we show by contradiction that for η small enough, there holds

To this end, we assume that there exists a sequence of parameter and index pairs (θ N,n , p n ) such that p n ∈ A c , θ N,n ∈ S pn , and θ N,n → θ N as n → ∞. Since A c is finite, there exists a constant subsequence {p ni } ⊂ {p n } and some constant p 0 ∈ A c with p ni = p 0 for all i.

Then the continuity of the network and g p0 imply that S p0 is a closed set and consequently we obtain θ N ∈ S p0 by taking the limit, which contradicts the fact that θ N ∈ p∈A c S p .

Therefore, for η small enough, there holds B η (θ N ) ⊆ p∈A S p , which contradicts the assumption θ N ∈ ∂S ∂i .

Consequently, there exists a j ∈ {1, . . .

, J i } satisfying equation 20.

By setting

, where ⊕ denotes concatenation, then for for all θ N ∈ B min{η,η } (θ N ), there holds

and the assertion is proved.

Now, for a fixed p = (i, j, s) ∈ P, we denote the set of network parameters θ N that lie on the boundary of p by

where the functions g j i,s are as in Definition 3.

As usual, the boundary of the set S p is denoted by ∂S p and the set of the boundaries is denoted by

Finally, for the ease of notations, if P ⊂ P, we write

Obviously, we have θ N ∈ R m \ ∂S ¬π(i) .

To prove Lemma 11, we need the following lemma which follows directly from Mityagin (2015) and hence we omit the proof.

Lemma 10 Let θ N → F (θ N ) : R |θ N | → R be a real analytic function and define

Then we have either µ(M) = 0 or F = 0.

Lemma 11 Let the set of boundaries ∂S P be as in equation 21.

Then we have

Proof [of Lemma 11]

We proceed the proof by induction.

For i = 0, we obviously have ∂S ¬0 = ∅ and therefore µ(∂S ¬0 ) = 0.

For the induction step let us assume that

For s = (p, q), the pair of an index p ∈ ∂i and a trajectory q ∈ Q(i), we define

where the analytic function f q is defined as in Proposition 9.

Then we prove by contradiction that for any θ N ∈ ∂S ∂i \ ∂S ¬π(i) , there exists an s ∈ ∂i × Q(i) such that θ N ∈ M s and µ(M s ) = 0.

To this end, let θ N ∈ ∂S ∂i \ ∂S ¬π(i) , then for small enough η > 0, there holds

.

By Proposition 9, there exists a trajectory q ∈ Q(π(i)) such that for all θ N ∈ B η (θ N ), there holds

Moreover, since θ N ∈ ∂S ∂i , there exists an index p ∈ ∂i such that g p (h

This means that for s = (p, q), we have θ N ∈ M s .

Therefore, we have

where

Combing equation 24 and equation 25, we obtain

By the assumption µ(∂S ¬π(i) ) = 0, we conclude that µ(∂S ¬i ) = 0.

Since for the last node L, we have ¬L = P and therefore µ(∂S P ) = 0.

Note that the random forest µ (T ) (·) can be considered as the composition of affine transformations and sigmoid functions and hence are always continuously differentiable with respect to θ F , we only need to investigate whether the neural network h θ N (x) is differentiable with respect to θ N .

For a fixed x ∈ R d , we write

as the set of parameters for which the network is not differentiable.

Lemma 12 Let the set Θ x be as in equation 26.

Then, for any x ∈ R d , we have

Proof [of Lemma 12] Let the boundary set ∂S P be defined as in equation 22 and θ N,0 ∈ Θ x .

Obviously, we have θ 0 ∈ S P .

We proceed the proof of the inclusion Θ x ⊆ ∂S P by contradiction.

To this end, we assume that θ D,0 ∈ ∂S P .

When Lemma 9 applied to the output layer, there exist some η > 0 and a real analytic function f (θ N ) such that for all θ N ∈ B η (θ N,0 ), there holds

Consequently, the network is differentiable at θ N,0 , contradicting the fact that θ N,0 ∈ Θ x .

Therefore, we have Θ x ⊆ ∂S P and hence µ(Θ x ) = 0 since µ(∂S P ) = 0 according to Lemma 11.

Proof [of Proposition 5] Let the sets N (θ N ), Θ P X , and Θ x be as in equation 7, equation 8, and equation 26, respectively.

Consider the sets S 1 = {(θ N , x) ∈ R |θ N | × R d | θ N ∈ Θ P X and x ∈ N (θ N )},

Since the network is continuous and not differentiable, Theorem I in Piranian (1966) implies that the sets S 1 and S 2 are measurable.

Obviously, we have S 1 ⊂ S 2 and therefore we obtain ν(S 1 ) ≤ ν(S 2 ), where ν := P X ⊗ µ. On the one hand, Fubini's theorem implies

By Lemma 12, we have µ(Θ x ) = 0 and therefore ν(S 2 ) = 0 and hence ν(S 1 ) = 0.

On the other hand, Fubini's theorem again yields

By the definition of Θ P X we have P X (N (θ N )) > 0 for all θ N ∈ Θ P X .

Therefore, ν(S 1 ) = 0 implies that µ(Θ P X ) = 0.

C.3 PROOFS OF SECTION 3.5

To prove Theorem 2, we also need the following lemma.

Lemma 13 Let (θ n ) n∈N be a sequence in R m converging towards θ 0 , i.e., θ n = θ 0 as n → ∞. Moreover, let f : R m → R be a function and g be a vector in R m .

If

holds for all sequences (θ n ) n∈N , then f is differentiable at θ 0 with differential g.

Proof [of Lemma 13]

The definition of a differential tell us that g is the differential of f at θ 0 , if

By the sequential characterization of limits, we immediately obtain the assertion.

Proof [of Theorem 2] Consider the following augmented networks:

(θ D ,ψ) (Z, Z ) = (µ

Without loss of generality, in the following, we only consider the network h

(θ D ,ψ) (X, Z) with inputs from P X ⊗ P Z , which satisfies E P X ⊗P Z (X, Z) 2 < ∞.

By the expression of definition 1, we have

where we denote µ (T ) θ D (x) := µ (T ) (h θ N (x); θ F ).

Due to the linear kernel k L (x, y) := x, y , there holds )) is differentiable at λ 0 for P X -almost all x and P Z -almost all z.

Then, according to Proposition 5, this statement holds for µ-almost all θ D ∈ R |θ D | and all ψ ∈ R |ψ| .

Consider a sequence (λ n ) n∈N that converges to λ 0 , then there exists a δ > 0 such that λ n − λ 0 < δ for all n ∈ N. For a fixed u = (x, z) ∈ R d × R |z| , Proposition 4 states that there exists a regular function c(u) with E P X c(X) < ∞ such that |K(h λn (u)) − K(h λ0 (u))| ≤ c(u) λ n − λ 0 and consequently we have |∂ λ K(h λ0 (u))| ≤ c(u)

for P X ⊗ P Z -almost all u ∈ R |u| .

For n ∈ N, we define the sequence g n (x) by g n (u) = |K(h λn (u)) − K(h λ0 (u)) − ∂ λ K(h λ0 (u))(λ n − λ 0 )| λ n − λ 0 .

Obviously, the sequence g n (x) converges pointwise to 0 and is bounded by the integrable function 2c(u).

By the dominated convergence theorem, see e.g., Theorem 2.24 in Folland (1999) , we have

Moreover, for n ∈ N, we define the sequenceg n (x) bỹ

Clearly, the sequenceg n (x) is upper bounded by E P X ⊗P Z g n (u) and therefore converges to 0.

By Lemma 13, E P X ⊗P Z [K(h λ (u))] is differentiable at λ 0 with differential

Since similar results as above hold also for the networks h (2) and h (3) , and Lemma 6 in Gretton et al. (2012) states that MMD 2 u is unbiased, our assertion follows then from the linearity of the form of MMD D SAMPLES OF GENERATED PICTURES Generated samples on the datasets CIFAR-10, CelebA, and LSUN bedroom are shown in Figure  4 , 5, and 6, respectively.

<|TLDR|>

@highlight

Equip MMD GANs with a new random-forest kernel.