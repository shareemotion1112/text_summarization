One can substitute each neuron in any neural network with a kernel machine and obtain a counterpart powered by kernel machines.

The new network inherits the expressive power and architecture of the original but works in a more intuitive way since each node enjoys the simple interpretation as a hyperplane (in a reproducing kernel Hilbert space).

Further, using the kernel multilayer perceptron as an example, we prove that in classification, an optimal representation that minimizes the risk of the network can be characterized for each hidden layer.

This result removes the need of backpropagation in learning the model and can be generalized to any feedforward kernel network.

Moreover, unlike backpropagation, which turns models into black boxes, the optimal hidden representation enjoys an intuitive geometric interpretation, making the dynamics of learning in a deep kernel network simple to understand.

Empirical results are provided to validate our theory.

Any neural network (NN) can be turned into a kernel network (KN) by replacing each artificial neuron (McCulloch & Pitts, 1943) , i.e., learning machine of the form f (x) = σ(w x + b), with a kernel machine, i.e., learning machine of the form f (x) = w, φ(x) + b with kernel function k(x, y) = φ(x), φ(y) .

This combination of connectionism and kernel method enables the learning of hierarchical, distributed representations with kernels.

In terms of training, similar to NN, KN can be trained with backpropagation (BP) (Rumelhart et al., 1986) .

In the context of supervised learning, the need for BP in learning a deep architecture is caused by the fact that there is no explicit target information to tune the hidden layers (Rumelhart et al., 1986) .

Moreover, BP is usually computationally intensive and can suffer from vanishing gradient.

And most importantly, BP results in hidden representations that are notoriously difficult to interpret or assess, turning deep architectures into "black boxes".The main theoretical contribution of this paper is the following: Employing the simplest feedforward, fully-connected KN as an example, we prove that in classification and under certain losses, the optimal representation for each hidden layer that minimizes the risk of the network can be explicitly characterized.

This result removes the need for BP and makes it possible to train the network in a feedforward, layer-wise fashion.

And the same idea can be generalized to other feedforward KNs.

The layer-wise learning algorithm gives the same optimality guarantee as BP in the sense that it minimizes the risk.

But the former is much faster and evidently less susceptible to vanishing gradient.

Moreover, the quality of learning in the hidden layers can be directly assessed during or after training, providing more information about the model to the user.

For practitioners, this enables completely new model selection paradigms.

For example, the bad performance of the network can now be traced to a certain layer, allowing the user to debug the layers individually.

Most importantly, the optimal representation for each hidden layer enjoys an intuitive geometric interpretation, making the learning dynamics in a deep KN more transparent than that in a deep NN.

A simple acceleration method that utilizes the "sparse" nature of the optimal hidden representations is proposed to further reduce computational complexity.

Empirical results on several computer vision benchmarks are provided to demonstrate the competence of the model and the effectiveness of the greedy learning algorithm.

Figure 1: (a) Any NN (left, presented in the usual weight-nonlinearity abstraction) can be abstracted as a "graph" (right) with each node representing a neuron and each edge the input-output relationship between neurons.

If a node receives multiple inputs, we view its input as a vector in some Euclidean space, as indicated by the colored rectangles.

Under this abstraction, each neuron (f (x) = σ(w x + b)) can be directly replaced by a kernel machine (f (x) = w, φ(x) + b with kernel k(x, y) = φ(x), φ(y) ) mapping from the same Euclidean space into the real line without altering the architecture and functionality of the model.

(b) Illustration for layer-wise optimality drifting away from network-optimality.

Consider a two-layer network and let T 1 , T 2 be the target function of the first and second layer, respectively.

If the first layer creates error, which is illustrated by F (1) (x) being far away from T 1 (x), the composed solution F (2) • F (1) on the right is better than that on the left and hence the F (2) on the right corresponds to the network-wise optimality of the second layer.

But the F (2) on the left is clearly a better estimate to the layer-wise optimality T 2 if the quality of estimation is measured by the supremum distance.

In this section, we discuss how to build a KN using a given NN.

First, the generic approach is described in FIG4 .

Note that KN inherits the expressive power of the original NN since a kernel machine is a universal function approximator under mild conditions (Park & Sandberg, 1991; Micchelli et al., 2006) and the two models share the same architecture.

However, KN works in a more intuitive way since each node is a simple linear model in a reproducing kernel Hilbert space (RKHS).We now concretely define the KN equivalent of an l-layer Multilayer Perceptron (MLP), which we shall refer to as the kernel MLP (kMLP).1 Given a random sample (x n , y n ) N n=1 , where (x n , y n ) ∈ X 1 × Y ⊂ R d0 × R, denote (x n ) N n=1 as S and (y n ) N n=1 as Y S for convenience.

For i = 1, 2, . . .

, l, consider kernel k (i) : X i × X i → R, X i ⊂ R di−1 (for i > 1, d i−1 is determined by the width of the i − 1 th layer).

k (i) (x, y) = φ (i) (x), φ (i) (y) Hi , where φ (i) is a mapping into RKHS H i .For i ≥ 1, the i th layer in a kMLP, denoted F (i) , is an array of d i kernel machines: DISPLAYFORM0 2 , . . .

, f (i) di ), a d i -tuple.

Let F (0) be the identity map on R d0 , each f DISPLAYFORM1 j , where the α DISPLAYFORM2 ∈ R are the learnable parameters.

2 The set of mappings DISPLAYFORM3 j ∈ R for all admissible n, j, i} defines an l-layer kMLP.

In the rest of this paper, we shall restrict our discussions to this kMLP.

We now specify the assumptions that we impose on all kernels considered in this paper.

First, we consider real, continuous, symmetric kernels only and we call a kernel positive semidefinite (PSD) or positive definite (PD) if for any S, the kernel matrix defined as (G) mn = k(x m , x n ) is PSD or PD, respectively.

We shall always assume that any kernel considered is at least PSD and that DISPLAYFORM0 It is straightforward to check using Cauchy-Schwarz inequality that the first condition implies max x,y∈Xi k (i) (x, y) = c.

For each fixed x ∈ X i , we assume that DISPLAYFORM1 x -Lipschitz with respect to the Euclidean metric on DISPLAYFORM2 , which we assume to be finite.

The following notations will be used whenever convenient: We use the shorthand DISPLAYFORM3 (1) (S) and the same is true with S substituted by any x. Throughout this paper, notations such as F (i) can either be used to denote a set of functions or a specific function in some set depending on the context.

Also, when there is no confusion, we shall suppress the dependency of any loss function on the example for brevity, i.e., for a loss function , instead of writing (f (x), y), we shall write (f ).

To simplify discussion, we shall restrict ourselves to binary classification (Y = {+1, −1}) and directly give the result on classification with more than two classes in the end.

A generalization to regression is left as future work.

Again, we only focus on kMLP although the idea can be directly generalized to all feedforward KNs.

We now discuss the layer-wise learning algorithm, beginning by addressing the difficulties with training a deep architecture layer-by-layer.

There are two fundamental difficulties with learning a deep architecture layer-wise.

First, the hidden layers do not have supervision (labels) to learn from.

And it depends on BP to propagate supervision from the output backward (Rumelhart et al., 1986) .

We shall prove that for kMLP, one can characterize the optimal target representation for each hidden layer, which induces a risk for that layer.

The target is optimal in the sense that it minimizes the risk of the subsequent layer and eventually that of the network if all layers succeed in learning their optimal representations.

This optimal representation defines what we call "layer-wise optimality".The other difficulty with layer-wise learning is that for any given hidden layer, when the upstream layers create error, layer-wise optimality may not coincide with "network-wise optimality", i.e., the solution of this layer that eventually leads to the composed solution that minimizes the risk of the network in this suboptimal case.

Indeed, when a hidden layer creates error, the objective of any layer after it becomes learning a solution that is a compromise between one that is close to the layer-wise optimality and one that prevents the error from the "bad" layer before it from "getting through" easily.

And the best compromise is the network-wise optimality.

The two solutions may not coincide, as shown in the toy example in FIG4 .

Clearly, we would like to always learn the network-wise optimality at each layer, but the learner is blind to it if it is only allowed to work on one layer at a time.

By decomposing the overall error of the network into error at each layer, we prove that in fact, network-wise optimality is learnable for each hidden layer even in a purely layer-wise fashion and that the proposed layer-wise algorithm learns network-wise optimality at each layer.

We now address the first difficulty in layer-wise learning.

The basic idea is first described in Section 4.2.1.

Then we provide technical results in Section 4.2.2 and Section 4.2.3 to fill in the details.

DISPLAYFORM0 and a loss function l defined for this network which induces a risk that we wish to minimize: R l = E l (F).

BP views this problem in the following way: R l is a function of F. The learner tries to find an F that minimizes R l using the random sample S with labels Y S according to some learning paradigm such as Empirical Risk Minimization (ERM) or Structural Risk Minimization (SRM) (Vapnik, 2000; Shalev-Shwartz & Ben-David, 2014) .

S is considered as fixed in the sense that it cannot be adjusted by the learner.

Alternatively, one can view R l as a function of F (l) and the learner tries to find an F (l) minimizing R l using random sample S l−1 with labels Y S according to some learning paradigm, where S l−1 := DISPLAYFORM1 The advantage is that the learner has the freedom to learn both the function F (l) and the random sample S l−1 .

And since S l−1 determines the decision of the learning paradigm, which then determines R l , R l is now essentially a function of both F (l) and S l−1 : DISPLAYFORM2 The key result is that independently of the actual learning of F (l) , one can characterize the sufficient condition on S l−1 under which R l , as a function of S l−1 , is minimized, as we shall prove.

In other words, the "global minimum" of R l w.r.t.

S l−1 can be explicitly identified prior to any training.

This gives the optimal S l−1 , which we denote as S l−1 .Moreover, the characterization of S l−1 gives rise to a new loss function l−1 and thus also a new risk R l−1 that is a function of DISPLAYFORM3 .

Consequently, the same reasoning would allow us to deduce S l−2 before the learner learns F (l−1) .

And this analysis can be applied to each layer, eventually leading to a greedy learning algorithm that sequentially learns DISPLAYFORM4 (1) * , in that order, where the asterisk on the superscript indicates that the corresponding layer has been learned and frozen.

The layer-wise learning algorithm provides a framework that enjoys great flexibility.

To be specific, one could stop the above analysis at any layer i, then learn layers i + 1, . . .

, l in a greedy fashion but still learn layers 1, . . .

, i together with BP.

Thus, it is easy to see that BP can be brought under this framework as a special case.

Nevertheless, in later text, we shall stay on the one end of the spectrum where each layer is learned individually for clarity.

We now present the formal results that give the optimal hidden representations.

By the reasoning above, the analysis starts from the last hidden layer (layer l − 1) and proceeds backward.

To begin with, we need to approximate the true classification error R l since it is not computable.

To this end, we first review a well-known complexity measure.

Definition 4.1 (Gaussian complexity (Bartlett & Mendelson, 2002) ).

Let P be a probability distribution on a metric space X and suppose x 1 , . . .

, x N are independent random elements distributed as P. Let F be a set of functions mapping from X into R. Definê DISPLAYFORM0 where g 1 , . . . , g N are independent standard normal random variables.

The Gaussian complexity of DISPLAYFORM1 Intuitively, Gaussian complexity quantifies how well elements in a given function class can be correlated with a noise sequence of length N , i.e., the g n (Bartlett & Mendelson, 2002) .

Based on this complexity measure, we have the following bound on the expected classification error.

DISPLAYFORM2 , with probability at least 1 − δ and for any N ∈ N, every function DISPLAYFORM3 , the empirical hinge loss.

Given the assumptions on k (l) , for any F , we have DISPLAYFORM4 Without loss of generality, we shall set hyperparameter γ = 1.

We now characterize S l−1 .

Note that for a given f (l) , A = w f (l) H l is the smallest nonnegative real number such that f (l) ∈ F l,A and it is immediate that this gives the tightest bound in Theorem 4.2.

DISPLAYFORM5 , where τ is any positive constant satisfying τ < 2(c − a) min(κ, 1 − κ).

Denote as S l−1 any representation satisfying DISPLAYFORM6 for all pairs of x + , x − from distinct classes in S and all pairs of x, x from the same class.

Suppose the learning paradigm returns f (l) under this representation.

Let S DISPLAYFORM7 The optimal representation S l−1 , characterized by Eq. 1, enjoys a straightforward geometric interpretation: Examples from distinct classes are as distant as possible in the RKHS whereas examples from the same class are as concentrated as possible (see proof (C) of Lemma 4.3 for a rigorous justification).

Intuitively, it is easy to see that such a representation is the "easiest" for the classifier.

The conditions in Eq. 1 can be concisely summarized in an ideal kernel matrix G defined as DISPLAYFORM8 And to have the l − 1 th layer learn S l−1 , it suffices to train it to minimize some dissimilarity measure between G and the kernel matrix computed from k (l) and F (l−1) (S), which we denote G l−1 .

Empirical alignment (Cristianini et al., 2002) , L 1 and L 2 distances between matrices can all serve as the dissimilarity measure.

To simplify discussion, we let the dissimilarity measure be the DISPLAYFORM9 This specifiesR l−1 (F (l−1) ) as the sample mean of ( l−1 (F (l−1) , (x m , y m ), (x n , y n ))) N m,n=1 and R l−1 as the expectation of l−1 over (X 1 , Y )×(X 1 , Y ).

Note that due to the boundedness assumption on k (l) , l−1 ≤ 2 max(|c|, |a|).

S l−2 , . . .

, S 1 Similar to Section 4.2.2, we first need to approximate R l−1 .

DISPLAYFORM0 , where F l−1 is a given hypothesis class.

There exists an absolute constant C > 0 such that for any N ∈ N, with probability at least 1 − δ, DISPLAYFORM1 We are now in a position to characterize S l−2 .

For the following lemma only, we further assume that k (l−1) (x, y), as a function of (x, y), depends only on and strictly decreases in x − y 2 for all x, y ∈ X l−1 with k (l−1) (x, y) > a, and that the infimum inf x,y∈X l−1 k (l−1) (x, y) = a is attained in X l−1 at all x, y with x − y 2 ≥ η.

Also assume that inf x,y∈X l−1 ; x−y 2<η ∂k (l−1) (x, y)/∂ x − y 2 = ι (l−1) is defined and is positive.

Consider an F mapping S into X l−1 , let DISPLAYFORM2 , it is immediate that DISPLAYFORM3 is the smallest nonnegative real number such that f Under review as a conference paper at ICLR 2019Lemma 4.5 (optimal S l−2 ).

Given a learning paradigm minimizingR l−1 ( DISPLAYFORM4 , where τ is any positive constant satisfying τ < 2d l−1 (c − a)ψι (l−1) .

Denote as S l−2 any representation satisfying DISPLAYFORM5 for all pairs of x + , x − from distinct classes in S and all pairs of x, x from the same class.

Suppose the learning paradigm returns DISPLAYFORM6 under this representation.

Let S • l−2 be another representation under which the learning paradigm returns DISPLAYFORM7 achieves zero loss on at least one pair of examples from distinct classes, then for any N ∈ N, DISPLAYFORM8 Applying this analysis to the rest of the hidden layers, it is evident that the i th layer, i = 1, 2, ..., l − 1, should be trained to minimize the difference between G and the kernel matrix computed with k DISPLAYFORM9 and DISPLAYFORM10 Generalizing to classification with more than two classes requires no change to the algorithm since the definition of G is agnostic to the number of classes involved in the classification task.

Also note that the sufficiency of expanding the kernel machines of each layer on the training sample (see Section 2) for the learning objectives in Lemma 4.3 and Lemma 4.5 is trivially justified since the generalized representer theorem directly applies (Schölkopf et al., 2001 ).

Now since the optimal representation is consistent across layers, the dynamics of layer-wise learning in a kMLP is clear: The network maps the random sample sequentially through layers, with each layer trying to map examples from distinct classes as far as possible in the RKHS while keeping examples from the same class in a cluster as concentrated as possible.

In other words, each layer learns a more separable representation of the sample.

Eventually, the output layer works as a classifier on the final representation and since the representation would be "simple" after the mappings of the lower layers, the learned decision boundary would generalize better to unseen data, as suggested by the bounds above.

We now discuss how to design a layer-wise learning algorithm that learns network-wise optimality at each layer.

A rigorous description of the problem of layer-wise optimality drifting away from network-wise optimality and the search for a solution begins with the following bound on the total error of any two consecutive layers in a kMLP.Lemma 4.6.

For any i = 2, . . .

, l, let the target function and the approximation function be T i , F (i) * : DISPLAYFORM0 DISPLAYFORM1 By applying the above bound sequentially from the input layer to the output, we can decompose the error of an arbitrary kMLP into the error of the layers.

This result gives a formal description of the problem: The hypothesis with the minimal norm minimizes the propagated error from upstream, but evidently, this hypothesis is not necessarily close to the layer-wise optimality T i .Moreover, this bound provides the insight needed for learning network-wise optimality individually at each layer: For the i th layer, i ≥ 2, searching for network-wise optimality amounts to minimizing the r.h.s.

of Eq. 2.

Lemma 4.3 and Lemma 4.5 characterized T i for i < l and learning objectives that bound i were provided earlier in the text accordingly.

Based on those results, the solution that minimizes the new learning objectiveR i ( DISPLAYFORM2

, where τ > 0 is a hyperparameter, provides a good approximate to the minimizer of the r.h.s.

of Eq. 2 if, of course, τ is chosen well.

Thus, taking this as the learning objective of the i th layer produces a layer-wise algorithm that learns network-wise optimality at this layer.

Note that for BP, one usually also needs to heuristically tune the regularization coefficient for weights as a hyperparameter.

There is a natural method to accelerate the upper layers (all but the input layer): The optimal representation F (S) is sparse in the sense that φ(F (x m )) = φ(F (x n )) if y m = y n and φ(F (x m )) = φ(F (x n )) if y m = y n (see the proof (C) of Lemma 4.3).

Since a kernel machine built on this representation of the given sample is a function in the RKHS that is contained in the span of the image of the sample, retaining only one example from each class would result in exactly the same hypothesis class because trivially, we have { DISPLAYFORM0 Thus, after training a given layer, depending on how close the actual kernel matrix is to the ideal one, one can (even randomly) discard a large portion of centers for kernel machines of the next layer to speed up the training of it without sacrificing performance.

As we will later show in the experiments, randomly keeping a fraction of the training sample as centers for upper layers produces performance comparable to or better than that obtained with using the entire training set.

The idea of combining connectionism with kernel method was initiated by Cho & Saul (2009) .

In their work, an "arc cosine" kernel was so defined as to imitate the computations performed by a one-layer MLP.

Zhuang et al. (2011) extended the idea to arbitrary kernels with a focus on MKL, using an architecture similar to a two-layer kMLP.

As a further generalization, Zhang et al. FORMULA8 independently proposed kMLP and the KN equivalent of CNN.

However, they did not extend the idea to any arbitrary NN.

Scardapane et al. (2017) proposed to reparameterize each nonlinearity in an NN with a kernel expansion, resulting in a network similar to KN but is trained with BP.

There are other works aiming at building "deep" kernels using approaches that are different in spirit from those above.

Wilson et al. (2016) proposed to learn the covariance matrix of a Gaussian process using an NN in order to make the kernel "adaptive".

This idea also underlies the now standard approach of combining a deep NN with an SVM for classification, which was first explored by Huang & LeCun (2006) and Tang (2013) .

Such an interpretation can be given to KNs as well, as we point out in Appendix B.5.

Mairal et al. (2014) proposed to learn hierarchical representations by learning mappings of kernels that are invariant to irrelevant variations in images.

Much works have been done to improve or substitute BP in learning a deep architecture.

Most aim at improving the classical method, working as add-ons for BP.

The most notable ones are perhaps the unsupervised greedy pre-training techniques proposed by Hinton et al. (2006) and Bengio et al. (2007) .

Among works that try to completely substitute BP, none provided a comparable optimality guarantee in theory as that given by BP.

Fahlman & Lebiere (1990) pioneered the idea of greedily learn the architecture of an NN.

In their work, each new node is added to maximize the correlation between its output and the residual error signal.

Several authors explored the idea of approximating error signals propagated by BP locally at each layer or each node (Bengio, 2014; Carreira-Perpinan & Wang, 2014; Lee et al., 2015; Balduzzi et al., 2015; Jaderberg et al., 2016) .

Kulkarni & Karande (2017) proposed to train NN layer-wise using an ideal kernel matrix that is a special case of that in our work.

No theoretical results were provided to justify its optimality for NN.

Zhou & Feng (2017) proposed a BP-free deep architecture based on decision trees, but the idea is very different from ours.

Raghu et al. (2017) attempted to quantify the quality of hidden representations toward learning more interpretable deep architectures, sharing a motivation similar to ours.

We compared kMLP learned using the proposed greedy algorithm with other popular deep architectures including MLP, Deep Belief Network (DBN) (Hinton & Salakhutdinov, 2006) and Stacked Autoencoder (SAE) (Vincent et al., 2010) , with the last two trained using a combination of unsupervised greedy pre-training and standard BP (Hinton et al., 2006; Bengio et al., 2007) .

Note that we only focused on comparing with the standard, generic architectures because kMLP, as the KN equivalent of MLP, does not have a specialized architecture or features designed for specific application domains.

Several optimization and training techniques were applied to the MLPs to boost performance.

These include Adam (Kingma & Ba, 2014) , RMSProp (Tieleman & Hinton, 2012) , dropout (Srivastava et al., 2014) and batch normalization (BN) (Ioffe & Szegedy, 2015) .

kMLP accelerated using the proposed method (kMLP FAST ) was also compared.

For these models, we randomly retained a subset of the centers of each upper layer before its training.

As for the benchmarks used, rectangles, rectangles-image and convex are binary classification datasets, mnist (10k) and mnist (10k) rotated are variants of MNIST (Larochelle et al., 2007; LeCun et al., 2010) .

And fashion-mnist is the Fashion-MNIST dataset (Xiao et al., 2017) .

To further test the proposed layer-wise learning algorithm and the acceleration method, we compared greedily-trained kMLP with MLP and kMLP trained using BP (Zhang et al., 2017) using the standard MNIST (LeCun et al., 2010) .

Two popular acceleration methods for kernel machines were also compared on the same benchmark, including using a parametric representation (i.e., for each node in a kMLP, f (x) = k(w, x), w learnable) (kMLP PARAM ) and using random Fourier features (kMLP RFF ) (Rahimi & Recht, 2008) .

More details for the experiments can be found in Appendix A 4 .From TAB0 , we see that the performance of kMLP is on par with some of the most popular and most mature deep architectures.

In particular, the greedily-trained kMLPs compared favorably with their direct NN equivalents, i.e., the MLPs, even though neither batch normalization nor dropout was used for the former.

These results also validate our earlier theoretical results on the layer-wise learning algorithm, showing that it indeed has the potential to be a substitute for BP with an equivalent optimality guarantee.

Results in TAB0 further demonstrate the effectiveness of the greedy learning scheme.

For both the single-hidden-layer and the two-hidden-layer kMLPs, the layer-wise algorithm consistently outperformed BP.

It is worth noting that the proposed acceleration trick, despite being extremely simple, is clearly very effective and even produced models outperforming the original ones.

This shows that kMLP together with the greedy learning scheme can be of practical interest even when dealing with the massive data sets in today's machine learning.

Last but not least, we argue that it is the practical aspects that makes the greedy learning framework promising.

Namely, this framework of learning makes deep architectures more transparent and intuitive, which can serve as a tentative step toward more interpretable, easy-to-understand models with strong expressive power.

Also, new design paradigms are now possible under the layer-wise framework.

For example, each layer can now be "debugged" individually.

Moreover, since learning becomes increasingly simple for the upper layers as the representations become more and more well-behaved, these layers are usually very easy to set up and also converge very fast during training.

The first data set, known as rectangles, has 1000 training images, 200 validation images and 50000 test images.

The learning machine is required to tell if a rectangle contained in an image has a larger width or length.

The location of the rectangle is random.

The border of the rectangle has pixel value 255 and pixels in the rest of an image all have value 0.

The second data set, rectangles-image, is the same with rectangles except that the inside and outside of the rectangle are replaced by an image patch, respectively.

rectangles-image has 10000 training images, 2000 validation images and 50000 test images.

The third data set, convex, consists of images in which there are white regions (pixel value 255) on black (pixel value 0) background.

The learning machine needs to distinguish if the region is convex.

This data set has 6000 training images, 2000 validation images and 50000 test images.

The fourth data set contains 10000 training images, 2000 validation images and 50000 test images taken from MNIST.

The fifth is the same as the fourth except that the digits have been randomly rotated.

Sample images from the data sets are given in FIG3 The experimental setup for the greedily-trained kMLPs is as follows, kMLP-1 corresponds to a one-hidden-layer kMLP with the first layer consisting of 15 to 150 kernel machines using the same Gaussian kernel (k(x, y) = e − x−y 2 /σ 2 ) and the second layer being a single or ten (depending on the number of classes) kernel machines using another Gaussian kernel.

Note that the Gaussian kernel does not satisfy the condition that the infimum a is attained (see the extra assumptions before Lemma 4.5), but for practical purposes, it suffices to set the corresponding entries of the ideal kernel matrix to some small value.

For all of our experiments, we set (G ) mn = 1 if y m = y n and 0 otherwise.

Hyperparameters were selected using the validation set.

The validation set was then used in final training only for early-stopping based on validation error.

For the standard MNIST and Fashion-MNIST, the last 5000 training examples were held out as validation set.

For other datasets, see (Larochelle et al., 2007) .

kMLP-1 FAST is the same kMLP for which we accelerated by randomly choosing a fraction of the training set as centers for the second layer after the first had been trained.

The kMLP-2 and kMLP-2 FAST are the two-hidden-layer kMLPs, the second hidden layers of which contained 15 to 150 kernel machines.

We used Adam (Kingma & Ba, 2014) as the optimization algorithm of the layer-wise scheme.

Although some of the theoretical results presented earlier in the paper were proved under certain losses, we did not notice a significant performance difference between using L 1 , L 2 and empirical alignment as loss function for the hidden layers.

And neither was such difference observed between using hinge loss and cross-entropy for the output layer.

This suggests that these results may be proved in more general settings.

To make a fair comparison with the NN models, the overall loss functions of all models were chosen to be the cross-entropy loss.

Settings of all the kMLPs trained with BP can be found in (Zhang et al., 2017) .

Note that because it is extremely time/memory-consuming to train kMLP with BP without any acceleration method, to make training possible, we could only randomly use 10000 examples from the entire training set of 55000 examples as centers for the kMLP-2 (BP) from TAB0 .We compared kMLP with a one/two-hidden-layer MLP (MLP-1/MLP-2), a one/three-hidden-layer DBN (DBN-1/DBN-3) and a three-hidden-layer SAE (SAE-3).

For these models, hyperparameters were also selected using the validation set.

For the MLPs, the sizes of the hidden layers were chosen from the interval [25, 700] .

All hyperparameters involved in Adam, RMSProp and BN were set to the suggested default values in the corresponding papers.

If used, dropout and BN was added to each hidden layer, respectively.

For DBN-3 and SAE-3, the sizes of the three hidden layers varied in intervals [500, 3000] , [500, 4000] and [1000, 6000], respectively.

DBN-1 used a much larger hidden layer than DBN-3 to obtain comparable performance.

A simple calculation shows that the total numbers of parameters in the kMLPs were fewer than those in the corresponding DBNs and SAEs by orders of magnitude in all experiments.

Like in the training for the kMLPs, the validation set were also reserved for early-stopping in final training.

The DBNs and SAEs had been pre-trained unsupervisedly before the supervised training phase, following the algorithms described in (Hinton et al., 2006; Bengio et al., 2007) .

More detailed settings for these models were reported in (Larochelle et al., 2007) .

In this section, we provide some further analysis on kMLP and the layer-wise learning algorithm.

Namely, in Appendix B.1, we give a bound on the Gaussian complexity of an l-layer kMLP, which describes the intrinsic model complexity of kMLP.

In particular, the bound describes the relationship between the depth/width of the model and the complexity of its hypothesis class, providing useful information for model selection.

In Appendix B.2, we give a constructive result stating that the dissimilarity measure being optimized at each hidden layer will not increase as training proceeds from the input layer to the output.

This also implies that a deeper kMLP performs at least as well as its shallower counterparts in minimizing any loss function they are trained on.

In Appendix B.3, a result similar to Lemma 4.3 is provided, stating that the characterization for the optimal representation can be made much simpler if one uses a more restricted learning paradigm.

In fact, in contrast to Lemma 4.3, both necessary and sufficient conditions can be determined under the more restricted setting.

In Appendix B.4, we provide a standard, generic method to estimate the Lipschitz constant of a continuously differentiable kernel, as this quantity has been repeatedly used in many of our results in this paper.

In Appendix B.5, we state some advantages of kMLP over classical kernel machines.

In particular, empirical results are provided in Appendix B.5.1, in which a two-layer kMLP consistently outperforms the classical Support Vector Machine (SVM) (Cortes & Vapnik, 1995) as well as several SVMs enhanced by Multiple Kernel Learning (MKL) algorithms (Bach et al., 2004; Gönen & Alpaydın, 2011) .

We first give a result on the Gaussian complexity of a two-layer kMLP.

Lemma B.1.

Given kernel k : DISPLAYFORM0 where the x ν are arbitrary examples from X 2 .

DISPLAYFORM1 where Ω is a given hypothesis class that is closed under negation, i.e., if DISPLAYFORM2 If the range of some element in Ω contains 0, we have DISPLAYFORM3 The above result can be easily generalized to kMLP with an arbitrary number of layers.

Lemma B.2.

Given an l-layer kMLP, for each f DISPLAYFORM4 1 ≤ A i and let d l = 1.

Denote the class of functions implemented by this kMLP as F l , we have DISPLAYFORM5 Proof.

It is trivial to check that the hypothesis class of each layer is closed under negation and that there exists a function in each of these hypothesis classes whose range contains 0.

Then the result follows from repeatedly applying Lemma B.1.

Lemma B.3.

For i ≥ 2, assume k (i) is PD and fix layers 1, 2, . . .

, i − 1 at arbitrary states F (1) , F (2) , . . .

, F (i−1) .

Let the loss functions i , i−1 be the same up to their domains, and denote both i and i−1 as .

Suppose layer i is trained with a gradient-based algorithm to minimize the loss (F (i) ).

Denote the state of layer i after training by DISPLAYFORM0 Calculation for this initialization is specified in the proof.

For i = 1, under the further assumption that DISPLAYFORM1 is the identity map on X 1 .Remark B.3.1.

For the greedily-trained kMLP, Lemma B.3 applies to the hidden layers and implicitly requires that k (i+1) = k (i) since the loss function for layer i, when viewed as a function of DISPLAYFORM2 ) and can be rewritten as DISPLAYFORM3 ).

Since Lemma B.3 assumes to be the same across layers (otherwise it does not make sense to compare between layers), this forces DISPLAYFORM4 Further, if k (i+1) and k (i) have the property that k(x, y) = k(x,ȳ), wherex,ȳ denote the images of x, y under an embedding of R p into R q (p ≤ q) defined by the identity map onto a p-dimensional subspace of R q , then the condition DISPLAYFORM5 This lemma states that for a given kMLP, when it has been trained upto and including the i th hidden layer, the i + 1 th hidden layer can be initialized in such a way that the value of its loss function will be lower than or equal to that of the i th hidden layer after training.

In particular, the actual hidden representation "converges" to the optimal represetation as training proceeds across layers.

On the other hand, when comparing two kMLPs, this result implies that the deeper kMLP will not perform worse in minimizing the loss function than its shallower counterpart.

In deep learning literature, results analogous to Lemma B.3 generally state that in the hypothesis class of a NN with more layers, there exists a hypothesis that approximates the target function nontrivially better than any hypothesis in that of another shallower network (Sun et al., 2016) .

Such an existence result for kMLP can be easily deduced from the earlier bound on its Gaussian complexity (see Lemma B.2).

However, these proofs of existence do not guarantee that such a hypothesis can always be found through learning in practice, whereas Lemma B.3 is constructive in this regard.

Nevertheless, one should note that Lemma B.3 does not address the risk R = E .

Instead, it serves as a handy result that guarantees fast convergence of upper layers during training in practice.

The following lemma states that if we are willing to settle with a more restricted learning paradigm, the necessary and sufficient condition that guarantees the optimality of a representation can be characterized and is simpler than that described in Lemma 4.3.

The setup for this lemma is the same as that of Lemma 4.3 except that the assumption that the numbers of examples from the two classes are equal is not needed.

Lemma B.4.

Consider a learning paradigm that minimizesR l (f (l) ) + τ G N (F l,A ) using represen- DISPLAYFORM0 is minimized over all linearly separable representations if and only if the representation F (S) satisfies DISPLAYFORM1 for all pairs of x + , x − from distinct classes in S.

In general, for a continuously differentiable function f : R → R with derivative f and any a, b ∈ R, a < b, we have DISPLAYFORM0 This simple result can be used to bound the Lipschitz constant of a continuously differentiable kernel.

For example, for Gaussian kernel k : DISPLAYFORM1 , we have ∂k(x, y)/∂y = 2(x − y)k(x, y)/σ 2 .

Hence for each fixed x ∈ X, k(x, y) is Lipschitz in y with Lipschitz constant bounded by sup y∈X 2(x − y)k(x, y)/σ 2 .

In practice, X is always compact and can be a rather small subspace of some Euclidean space after normalization of data, hence this would provide a reasonable approximation to the Lipschitz constant of Gaussian kernel.

There are mainly two issues with classical kernel machines besides their usually high computational complexity.

First, despite the fact that under mild conditions, they are capable of universal function approximation (Park & Sandberg, 1991; Micchelli et al., 2006) and that they enjoy a very solid mathematical foundation BID0 , kernel machines are unable to learn multiple levels of distributed representations (Bengio et al., 2013 ), yet learning representations of this nature is considered to be crucial for complicated artificial intelligence (AI) tasks such as computer vision, natural language processing, etc. (Bengio, 2009; LeCun et al., 2015) .

Second, in practice, performance of a kernel machine is usually highly dependent on the choice of kernel since it governs the quality of the accessible hypothesis class.

But few rules or good heuristics exist for this topic due to its extremely task-dependent nature.

Existing solutions such as MKL (Bach et al., 2004; Gönen & Alpaydın, 2011) view the task of learning an ideal kernel for the given problem to be separate from the problem itself, necessitating either designing an ad hoc kernel or fitting an extra trainable model on a set of generic base kernels, complicating training.kMLP learns distributed, hierarchical representations because it inherits the architecture of MLP.To be specific, first, we see easily that the hidden activation of each layer, i.e., F (i) (x) ⊂ R di , is a distributed representation (Hinton, 1984; Bengio et al., 2013) .

Indeed, just like in an MLP, each layer of a kMLP consists of an array of identical computing units (kernel machines) that can be activated independently.

Further, since each layer in a kMLP is built on top of the previous layer in exactly the same way as how the layers are composed in an MLP, the hidden representations are hierarchical (Bengio et al., 2013) .Second, kMLP naturally combines the problem of learning an ideal kernel for a given task and the problem of learning the parameters of its kernel machines to accomplish that task.

To be specific, kMLP performs nonparametric kernel learning alongside learning to perform the given task.

Indeed, for kMLP, to build the network one only needs generic kernels, but each layer F (i) can be viewed as a part of a kernel of the form DISPLAYFORM0 The fact that each F (i) is learnable makes this kernel "adaptive", mitigating to some extent any limitation of the fixed generic kernel k (i+1) .

The training of layer i makes this adaptive kernel optimal as a constituent part of layer i + 1 for the task the network was trained for.

And it is always a valid kernel if the generic kernel k (i+1) is.

Note that this interpretation has been given in a different context by Huang & LeCun (2006) and Bengio et al. (2013) , we include it here only for completeness.

We now compare a single-hidden-layer kMLP using simple, generic kernels with SVMs enhanced by MKL algorithms that used significantly more kernels to demonstrate the ability of kMLP to automatically learn task-specific kernels out of standard ones.

The standard SVM and seven other SVMs enhanced by popular MKL methods were compared (Zhuang et al., 2011) , including the classical convex MKL (Lanckriet et al., 2004) with kernels learned using the extended level method proposed in (Xu et al., 2009) Eleven binary classification data sets that have been widely used in MKL literature were split evenly for training and test and were all normalized to zero mean and unit variance prior to training.

20 runs with identical settings but random weight initializations were repeated for each model.

For each repetition, a new training-test split was selected randomly.

For kMLP, all results were achieved using a greedily-trained, one-hidden-layer model with the number of kernel machines ranging from 3 to 10 on the first layer for different data sets.

The second layer was a single kernel machine.

All kernel machines within one layer used the same Gaussian kernel, and the two kernels on the two layers differed only in kernel width σ.

All hyperparameters were chosen via 5-fold cross-validation.

As for the other models compared, for each data set, SVM used a Gaussian kernel.

For the MKL algorithms, the base kernels contained Gaussian kernels with 10 different widths on all features and on each single feature and polynomial kernels of degree 1 to 3 on all features and on each single feature.

For 2LMKL INF , one Gaussian kernel was added to the base kernels at each iteration.

Each base kernel matrix was normalized to unit trace.

For L p MKL, p was selected from {2, 3, 4}. For MKM, the degree parameter was chosen from {0, 1, 2}. All hyperparameters were selected via 5-fold cross-validation.

From TAB4 , kMLP compares favorably with other models, which validates our claim that kMLP learns its own kernels nonparametrically hence can work well even without excessive kernel parameterization.

Performance difference among models can be small for some data sets, which is expected since they are all rather small in size and not too challenging.

Nevertheless, it is worth noting that only 2 Gaussian kernels were used for kMLP, whereas all other models except for SVM used significantly more kernels.

Proof of Lemma 4.3.

Throughout this proof we shall drop the layer index l for brevity.

Given that the representation satisfies Eq. 1, the idea is to first collect enough information about the returned f = (w , b ) such that we can computeR(f ) + τ w H and then show that for any other F (S) satisfying the condition in the lemma, suppose the learning paradigm returns f = (w , b ) ∈ F A , thenR(f ) + τ w H ≥R(f ) + τ w H .

We now start the formal proof.

First, note that in the optimal representation, i.e., an F (S) such that Eq. 1 holds, it is easy to see that φ(F (x − )) − φ(F (x + )) H is maximized over all representations for all x − , x + .Moreover, note that given the representation is optimal, we have φ(F (x)) = φ(F (x )) if y = y and φ(F (x)) = φ(F (x )) if y = y : Indeed, by Cauchy-Schwarz inequality, for all x, x ∈ S, k(F (x), F (x )) = φ(F (x)), φ(F (x )) H ≤ φ(F (x)) H φ(F (x )) H and the equality holds if and only if φ(F (x)) = pφ(F (x )) for some real constant p. Using the assumption on k, namely, that φ(F (x)) H = √ c for all F (x), we further conclude that the equality holds if and only if p = 1.

And the second half of the claim follows simply from c > a. Thus, all examples from the + and − class can be viewed as one vector φ(F (x + )) and φ(F (x − )), respectively.

The returned hyperplane f cannot pass both F (x + ) and F (x − ), i.e., f (F (x + )) = 0 and f (F (x − )) = 0 cannot happen simultaneously since if so, first subtract b , rotate while keeping w H unchanged and add some suitable b to get a new f such that f (F (x − )) < 0 and f (F (x + )) > 0, then it is easy to see thatR(f ) + τ w H <R(f ) + τ w H .

But by construction of the learning paradigm, this is not possible.

Now suppose the learning paradigm returns an f such that DISPLAYFORM0 First note that for an arbitrary θ F,w , ζ is less than or equal to 2 since one can always adjust b such that y + f (F (x + )) = y − f (F (x − )) without changing ζ and hence having a larger ζ will not further reduceR(f ), which is 0 when ζ = 2, but will result in a larger w H according to Eq. 4.

On the other hand, θ F,w must be 0 since this gives the largest ζ with the smallest w H .

Indeed, if the returned f does not satisfy θ F,w = 0, one could always shift, rotate while keeping w H fixed and then shift the hyperplane back to produce another f with θ F,w = 0 and this f results in a larger ζ if ζ < 2 or the same ζ if ζ = 2 but a smaller w H by rescaling.

Hencê DISPLAYFORM1 Together with what we have shown earlier, we conclude that 2 ≥ ζ > 0.

Then for some t ∈ R, we haveR DISPLAYFORM2 First note that we can choose t freely while keeping w fixed by changing b .

If κ = 1/2, we havê DISPLAYFORM3 Evidently, the last two cases both result inR(f DISPLAYFORM4 If κ > 1/2,R(f ) decreases in t hence t must be 1 for f , which impliesR(f ) = (1 − κ)(2 − ζ).

Similarly, if κ < 1/2, t = ζ − 1 and henceR(f ) = κ(2 − ζ).

DISPLAYFORM5 , which increases in t and hence t = 1 and DISPLAYFORM6 , this combination of κ and t contradicts the optimality assumption of f .

DISPLAYFORM7 , where the second equality is becauseR(f ) decreases in t. Again, κ > 1/2 leads to a contradiction.

Combining all cases, we havê DISPLAYFORM8 which, by the assumption on τ , strictly decreases in ζ over (0, 2] .

Hence the returned f must satisfy ζ = 2, which impliesR(f ) = 0 and we havê DISPLAYFORM9 Now, for any other F (S), suppose the learning paradigm returns f .

Let x w + , x w − be the pair of examples with the largest f (F (x + )) − f (F (x − )).

We have DISPLAYFORM10 where we have used the assumption that there exists DISPLAYFORM11 This proves the desired result.

Lemma C.1.

Suppose f 1 ∈ F 1 , . . .

, f d ∈ F d are elements from sets of real-valued functions defined on all of X 1 , X 2 , . . .

, X m , where FIG4 , . . .

, f d (x 1 ), f 1 (x 2 ), . . . , f d (x m ), y), where ω : R md × Y → R + ∪ {0} is bounded and L-Lipschitz for each y ∈ Y with respect to the Euclidean metric on R md .

Let ω • F = {ω • f : f ∈ F}. Denote the Gaussian complexity of F i on X j as G j N (F i ), if the F i are closed under negation, i.e., for all i, if f ∈ F i , then −f ∈ F i , we have DISPLAYFORM12 DISPLAYFORM13 In particular, for all j, if the x j n upon which the Gaussian complexities of the F i are evaluated are sets of i.i.d.

random elements with the same distribution, we have G DISPLAYFORM14 This lemma is a generalization of a result on the Gaussian complexity of Lipschitz functions on R k from (Bartlett & Mendelson, 2002) .

And the technique used in the following proof is also adapted from there.

Proof.

For the sake of brevity, we prove the case where m = 2.

The general case uses exactly the same technique except that the notations would be more cumbersome.

Let F be indexed by A. Without loss of generality, assume |A| < ∞. Define DISPLAYFORM15 ω(f α,1 (x n ), . . .

, f α,d (x n ), y n )g n ; DISPLAYFORM16 (f α,i (x n )g n,i + f α,i (x n )g N +n,i ), where α ∈ A, the (x n , x n ) are a sample of size N from X 1 × X 2 and g 1 , . . .

, g N , g 1,1 , . . .

, g 2N,d are i.i.d.

standard normal random variables.

Let arbitrary α, β ∈ A be given, define X α − X β 2 2 = E(X α − X β ) 2 , where the expectation is taken over the g n .

Define Y α − Y β 2 2 similarly and we have DISPLAYFORM17 ω(f α,1 (x n ), . . .

, f α,d (x n ), y n ) − ω(f β,1 (x n ), . . . , f β,d (x n ), y n ) DISPLAYFORM18 .

By Slepian's lemma (Pisier, 1999) and since the F i are closed under negation, DISPLAYFORM19 Taking the expectation of the x n and x n on both sides, we have DISPLAYFORM20 Proof of Lemma 4.4.

Normalize l−1 to [0, 1] by dividing 2 max(|c|, |a|).

Then the loss function becomes l−1 F (l−1) , (x m , y m ), (x n , y n ) = 1 2 max(|c|, |a|) k (l) F (l−1) (x m ), F (l−1) (x n ) − (G ) mn .For each fixed (G ) mn , l−1 F (l−1) , (x m , y m ), (x n , y n ) − l−1 F (l−1) , (x m , y m ), (x n , y n ) DISPLAYFORM21 2 max(|c|, |a|) DISPLAYFORM22 max(|c|, |a|) DISPLAYFORM23 Hence l−1 is L (l) / max(|c|, |a|)-Lipschitz in (F (l−1) (x m ), F (l−1) (x n )) with respect to the Euclidean metric on R Proof of Lemma 4.6.

First, it is trivial that the so-defined s metric is indeed a metric.

In particular, it satisfies the triangle inequality.

For i = 2, . . .

, l, Proof of Lemma B.1.

Since Ω and F 2 are both closed under negation, we havê DISPLAYFORM24 which proves that, as a function of F , A achieves its minimum if and only if F maximizes φ(F (x + )) − φ(F (x − )) H .

Since arg max where we have used the assumption on k, namely, that k(x, x) = φ(x), φ(x) H = φ(x) 2 H = c, for all x. It immediately follows that any minimizer F of A must minimize k(F (x + ), F (x − )) for all pairs of examples from opposite classes.

This proves the desired result.

<|TLDR|>

@highlight

We combine kernel method with connectionist models and show that the resulting deep architectures can be trained layer-wise and have more transparent learning dynamics. 