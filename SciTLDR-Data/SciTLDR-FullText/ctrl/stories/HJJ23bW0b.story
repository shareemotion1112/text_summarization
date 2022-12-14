Learning to predict complex time-series data is a fundamental challenge in a range of disciplines including Machine Learning, Robotics, and Natural Language Processing.

Predictive State Recurrent Neural Networks (PSRNNs) (Downey et al.) are a state-of-the-art approach for modeling time-series data which combine the benefits of probabilistic filters and Recurrent Neural Networks into a single model.

PSRNNs leverage the concept of Hilbert Space Embeddings of distributions (Smola et al.) to embed predictive states into a Reproducing Kernel Hilbert Space, then estimate, predict, and update these embedded states using Kernel Bayes Rule.

Practical implementations of PSRNNs are made possible by the machinery of Random Features, where input features are mapped into a new space where dot products approximate the kernel well.

Unfortunately PSRNNs often require a large number of RFs to obtain good results, resulting in large models which are slow to execute and slow to train.

Orthogonal Random Features (ORFs) (Choromanski et al.) is an improvement on RFs which has been shown to decrease the number of RFs required for pointwise kernel approximation.

Unfortunately, it is not clear that ORFs can be applied to PSRNNs, as PSRNNs rely on Kernel Ridge Regression as a core component of their learning algorithm, and the theoretical guarantees of ORF do not apply in this setting.

In this paper, we extend the theory of ORFs to Kernel Ridge Regression and show that ORFs can be used to obtain Orthogonal PSRNNs (OPSRNNs), which are smaller and faster than PSRNNs.

In particular, we show that OPSRNN models clearly outperform LSTMs and furthermore, can achieve accuracy similar to PSRNNs with an order of magnitude smaller number of features needed.

Learning to predict temporal sequences of observations is a fundamental challenge in a range of disciplines including machine learning, robotics, and natural language processing.

There exist a wide variety of approaches to modeling time series data, however recurrent neural networks (RNNs) have emerged as the clear frontrunner, achieving state-of-the-art performance in applications such as speech recognition (Heigold et al., 2016) , language modeling BID5 , translation (Cho et al., 2014b) , and image captioning BID13 .Predictive State Recurrent Neural Networks (PSRNNs) are a state-of-the-art RNN architecture recently introduced by Downey et al. (2017) that combine the strengths of probabilistic models and RNNs in a single model.

Specifically PSRNNs offer strong statistical theory, globally consistent model initializations, and a rich functional form which is none-the-less amenable to refinement via backpropogation through time (BPTT).

Despite consisting of a simple bi-linear operations, PSRNNs have been shown to significantly outperform more complex RNN architectures (Downey et al., 2017) , such as the widely used LSTMs BID3 and GRUs (Cho et al., 2014a) .PSRNNs leverage the concept of Hilbert Space embeddings of distributions BID10 to embed predictive states into a Reproducing Kernel Hilbert Space (RKHS), then estimate, predict, and update these embedded states using Kernel Bayes Rule (KBR) BID10 .

Because PSRNNs directly manipulate (kernel embeddings of) distributions over observations, they can be initialized via a globally consistent method-of-moments algorithm which reduces to a series of linear ridge regressions.

Practical implementations of PSRNNs are made possible by the machinery of Random Features (RFs): input features are mapped into a new space where dot products approximate the kernel well BID7 .

RFs are crucial to the success of PSRNNs, however PSRNNs often require a significant number of RFs in order to obtain good results.

And, unfortunately, the number of required RFs grows with the dimensionality of the input, resulting in models which can be large, slow to execute, and slow to train.

One technique that has proven to be effective for reducing the required number of RFs for kernel machines is Orthogonal Random Features (ORFs) BID14 .

When using ORFs, the matrix of RFs is replaced by a properly scaled random orthogonal matrix, resulting in significantly decreased kernel approximation error.

A particularly nice feature of ORFs is that BID14 Choromanski et al., 2017) prove that using ORFs results in a guaranteed improvement in pointwise kernel approximation error when compared with RFs.

Unfortunately the guarantees in BID14 are not directly applicable to the PSRNN setting.

PSRNNs first obtain a set of model parameters via ridge regression, then use these model parameters to calculate inner products in RF space.

This "downstream" application of RFs goes beyond the results proven in BID14 and Choromanski et al. (2017) .

Hence it is not clear whether or not ORF can be applied to obtain an improvement in the PSRNN setting.

In this work, we show that ORFs can be used to obtain OPSRNNs: PSRNNs initialized using ORFs which are smaller, faster to execute and train than PSRNNs initialized using conventional unstructured RFs.

We theoretically analyze the orthogonal version of the KRR algorithm that is used to initialize OPSRNNs.

We show that orthogonal RNNs lead to kernel algorithms with strictly better spectral properties and explain how this translates to strictly smaller upper bounds on failure probabilities regarding KRR empirical risk.

We compare the performance of OPSRNNs with that of LSTMs as well as conventional PSRNNs on a number of robotics tasks, and show that OPSRRNs are consistently superior on all tasks.

In particular, we show that OPSRNN models can achieve accuracy similar to PSRNNs with an order of magnitude smaller number of features needed.

Orthogonal random features were introduced in BID14 as an alternative to the standard approach for constructing random feature maps to scale kernel methods BID6 .

Several other structured constructions were known before BID1 Hinrichs & Vyb??ral, 2011; BID11 BID15 Choromanski & Sindhwani, 2016; Choromanska et al., 2016; Bojarski et al., 2017) , however these were motivated by computational and space complexity gains and led to weaker accuracy guarantees.

In contrast to this previous work, orthogonal random features were proposed to improve accuracy of the estimators relying on them.

Such an improvement was theoretically and experimentally verified, but only for pointwise kernel approximation BID14 Choromanski et al., 2017) and for specific kernels (such as Gaussian for dimensionality large enough, as well as dot-product and angular kernels).

It was not clear whether these pointwise gains translate to downstream guarantees for algorithms relying on kernels (for instance kernel ridge regression), even though there was some partial empirical evidence that this might be the case (in Choromanski et al. (2017) orthogonal random features were experimentally tested to provide more accurate approximation of the groundtruth kernel matrix in terms of the Frobenius norm error).

Even for the pointwise estimators and for the selected kernels, the guarantees were given only with the use of second moment methods (variance computation) and thus did not lead to strong concentration results with exponentially small probabilities of failure, which we obtain in this paper.

To the best of our knowledge, we are the first to apply orthogonal random features via kernel ridge regression for recurrent neural networks.

There is however vast related literature on orthogonal recurrent neural networks, where the matrices are enforced to be orthogonal or initialized to be random orthogonal.

Probably some of the most exploited recent directions are unitary evolution RNN architectures (Arjovsky et al., 2016) , where orthogonal and unitary matrices are used to address a key problem of RNN training -vanishing or exploding gradients.

Related results are presented in Henaff et al. (2016) , BID8 (orthogonal initialization for training acceleration), Ganguli et al. (2008) and BID12 .

Most of these results do not provide any strict theoretical guarantees regarding the superiority of the orthogonal approach.

Even though these approaches are only loosely related to our work, there is a common denominator: orthogonality, whether applied in our context or the aforementioned ones, seems to to be responsible for disentangling in (deep) representations BID0 .

Our rigorous theoretical analysis shows that this phenomenon occurs for the orthogonal KRR that is used as a subroutine of OPSRNNs, but the general mechanism is still not completely understood from the theoretical point of view.3 PREDICTIVE STATE RECURRENT NEURAL NETWORKS PSRNNs (Downey et al., 2017) are a recently developed RNN architecture which combine the ideas of predictive state (Boots et al., 2013) and RFs.

The key advantage of PSRNNs is that their state update can be interpreted in terms of Bayes Rule.

This allows them to be initialized as a fully functional model via a consistent method of moments algorithm, in contrast to conventional RNN architectures which are initialized at random.

Below we describe PSRNNs in more detail.

We pay particular attention to how PSRNNS utilize RFs, which will be replaced with ORFs in OPSRNNs.

The explicit construction of ORFs is given in Section 4.A predictive state BID4 ) is a set of expectations of features of future observations.

A predictive state is best understood in contrast with the more typical latent state: predictive states are distributions over known functions of observations, whereas latent states are distributions over unobserved latent quantities.

Formally, a predictive state is a vector q t = E[f t | h t ], where f t = f (o t:t+k???1 ) are features of future observations and h t = h(o 1:t???1 ) are features of historical observations.

PSRNNs use a predictive state, but embed it in a Reproducing Kernel Hilbert Space (RKHS) using RFs: Let k f , k h , k o be translation invariant kernels BID7 ) defined on f t , h t , and o t respectively.

Define projections ?? t = RF (f t ), ?? t = RF (h t ), and DISPLAYFORM0 PSRNN model parameters consist of an initial state q 1 and a 3-mode update tensor W .

The PSRNN state update equation is: DISPLAYFORM1 Figure 1: PSRNN Update, displayed on the left as a neural network and on the right as an equation

PSRNNs can be initialized using the Two Stage Regression (2SR) approach of Hefny et al. (2015) .

This approach is fast, statistically consistent, and reduces to simple linear algebra operations.

In 2SR q 1 and W are learned by solving three Kernel Ridge Regression problems in two stages.

Ridge regression is required in order to obtain a stable model, as it allows us to minimize the destabilizing effect of rare events while preserving statistical consistency.

In stage one we regress from past ?? t to future ?? t , and from past ?? t to the outer product of shifted future ?? := ?? t+1 with observations ?? t .

Let X ?? be the matrix whose tth column is ?? t , X ?? the matrix whose tth column is ?? t , and X ??????? be the matrix whose tth column is ?? t ??? ?? t : DISPLAYFORM0 Using W ??|?? and W ????|?? we obtain predictive state estimates X ??|?? and X ????|?? at each time step: DISPLAYFORM1 In stage two we regress from C ??|?? to C ????|?? to obtain the model weights W : DISPLAYFORM2 where ?? ??? R is the ridge parameter and I is the identity matrix and 1 is a column vector of ones.

Once the state update parameters have been learned via 2SR we train a kernel ridge regression model to predict ?? t from q t .

The 2SR algorithm is provably consistent in the realizable setting, meaning that in the limit we are guaranteed to recover the true model parameters.

Unfortunately this result relies on exact kernel values, while scalable implementations work with approximate kernel values via the machinery of RFs.

In practice we often require a large number of RFs in order to obtain a useful model.

This can result in large models which are slow to execute and slow to train.

We now introduce ORFs and show that they can be used to obtain smaller, faster models.

The key challenge with applying ORFs to PSRNNs is extending the theoretical guarantees of BID14 to the kernel ridge regression setting.

We explain here how to construct orthogonal random features to approximate values of kernels defined by the prominent family of radial basis functions and consequently, conduct kernel ridge regression for the OPSRNN model.

A class of RBF kernels K (RBFs in shorthand) is a family of functions: DISPLAYFORM0 , for z = x???y 2 , where ?? : R ??? R is a fixed positive definite function (not parametrized by n).

An important example is the class of Gaussian kernels.

Every RBF kernel K is shift-invariant, thus in particular its values can be described by an integral via Bochner's Theorem BID6 : DISPLAYFORM1 where ?? K ??? M(R n ) stands for some finite Borel measure.

Some commonly used RBF kernels K together with the corresponding functions ?? and probability density functions for measures ?? K are given in Table 2 .

The above formula leads straightforwardly to the standard unbiased MonteCarlo estimator of K(x, y) given as: K(x, y) = ?? m,n (x)?? m,n (y), where a random embedding ?? m,n : R n ??? R 2m is given as: DISPLAYFORM2 vectors w i ??? R n are sampled independently from ?? K and m stands for the number of random features used.

In this scenario we will often use the notation w iid i , to emphasize that different w i s are sampled independently from the same distribution.1 Note that we can train a regression model to predict any quantity from the state NamePositive-definite function ?? Probability density function DISPLAYFORM3 Figure 2: Common RBF kernels, the corresponding functions ??, and probability density functions (here: w = (w 1 , ..., w n ) ).For a datasets X , random features provide an equivalent description of the original kernel via the linear kernel in the new dataset ??(X ) = {??(x) : x ??? X } obtained by the nonlinear transformation ?? and lead to scalable kernel algorithms if the number of random features needed to accurately approximate kernel values satisfies: m N = |X |.Orthogonal random features are obtained by replacing a standard mechanism of constructing vectors w i and described above with the one where the sequence (w 1 , ..., w m ) is sampled from a "related" joint distribution ?? ort K,m,n on R n ?? ... ?? R n satisfying the orthogonality condition, namely: with probability p = 1 different vectors w i are pairwise orthogonal.

Since in practice we often need m > n, the sequence ( m??n , performing Gram-Schmidt orthogonalization and then renormalizing the rows such that the length of the renormalized row is sampled from the distribution from which w iid i s are sampled.

Thus the Gram-Schmidt orthogonalization is used just to define the directions of the vectors.

From now on, we will call such a joint distribution continuous-orthogonal.

The fact that for RBF kernels the marginal distributions are exactly ?? K and thus, kernel estimator is still unbiased, is a direct consequence of the isotropicity of distributions fom which directions of vectors w iid i are sampled.

For this class of orthogonal estimators we prove strong theoretical guarantees showing that they lead to kernel ridge regression models superior to state-of-the-art ones based on vectors w iid i .

Another class of orthogonal architectures considered by us is based on discrete matrices.

We denote by D a random diagonal matrix with nonzero entries taken independently and uniformly at random from the two-element set {???1, +1}. Furthermore, we will denote by H a Hadamard matrix obtained via Kronecker-products (see: Choromanski et al. (2017) ).

An m-vector sample from the discreteorthogonal joint distribution is obtained by taking m first rows of matrix G defined as G HAD = HD 1 ?? ... ?? HD k for: fixed k > 0, independent copies D i of D and then renormalizing each row in exactly the same way as we did it for continuous-orthogonal joint distributions.

Note that G HAD is a product of orthogonal matrices, thus its rows are also orthogonal.

The advantage of a discrete approach is that it leads to a more time and space efficient method for computing random feature maps (with the use of Fast Walsh-Hadamard Transform; notice that the Hadamard matrix does not even need to be stored explicitly).

This is not our focus in this paper though.

Accuracywise discrete-orthogonal distributions lead to slightly biased estimators (the bias is a decreasing function of the dimensionality n).

However as we have observed, in practice they give as accurate PSRNN models as continuous-orthogonal distributions, consistently beating approaches based on unstructured random features.

One intuitive explanation of that phenomenon is that even though in that setting kernel estimators are biased, they are still characterized by much lower variance than those based on unstructured features.

We leave a throughout theoretical analysis of discreteorthogonal joint distributions in the RNN context to future work.

DISPLAYFORM4

In this section we extend the theoretical guarantees of BID14 to give rigorous theoretical analysis of the initialization phase of OPSRNN.

Specifically, we provide theoretical guarantees for kernel ridge regression with orthogonal random features, showing that they provide strictly better spectral approximation of the ground-truth kernel matrix than unstructured random features.

As a corollary, we prove that orthogonal random features lead to strictly smaller empirical risk of the model.

Our results go beyond second moment guarantees and enable us to provide the first exponentially small bounds on the probability of a failure for random orthogonal transforms.

Before we state our main results, we will introduce some basic notation and summarize previous results.

Assume that labeled datapoints (x i , y i ), where x i ??? R n , y i ??? R for i = 1, 2, ..., are generated as follows: y i = f * (x i ) + ?? i , where f * : R n ??? R is a function that the model aims to learn, and ?? i for i = 1, 2, ... are independent Gaussians with zero mean and standard deviation ?? > 0.

The empirical risk of the estimator f : R n ??? R is defined as follows: DISPLAYFORM0 where N stands for a dataset size.

By f * vec ??? R N we denote a vector whose j th entry is f * (x j ).

Denote by f KRR a kernel ridge regression estimator applying exact kernel method (no random feature map approximation).

Assume that we analyze kernel K : R n ?? R n ??? R with the corresponding kernel matrix K. It is a well known result (Alaoui & Mahoney, 2015; Bach, 2013) that the empirical risk of f KRR is given by the formula: DISPLAYFORM1 where ?? > 0 stands for the regularization parameter and I N ??? R N ??N is an identity matrix.

Denote by f KRR an estimator based on some random feature map mechanism and by K the corresponding approximate kernel matrix.

The expression that is used in several bounds on the empirical risk for kernel ridge regression (see for instance Avron et al. (2017) ) is the modified version of the above formula for R(f KRR ), namely: DISPLAYFORM2 .

To measure how similar to the exact kernel matrix (in terms of spectral properties) a kernel matrix obtained with random feature maps is, we use the notion of ???-spectral approximation (Avron et al., 2017) .

DISPLAYFORM3 It turns out that one can upper-bound the risk R( f KRR ) for the estimator f KRR in terms of the ??? parameter if matrix K + ??N I N is a ???-spectral approximation of the matrix K + ??N I N , as the next result (Avron et al., 2017) shows:Theorem 1.

Suppose that K 2 ??? 1 and that matrix K + ??N I N obtained with the use of random features is a ???-spectral approximation of matrix K + ??N I N .

Then the empirical risk R( f KRR ) of the estimator f KRR satisfies: DISPLAYFORM4

Consider the following RBF kernels, that we call smooth RBFs.

As we show next, Gaussian kernels are smooth.

Definition 2 (smooth RBFs).

We say that the class of RBF kernels defined by a fixed ?? : R ??? R (different elements of the class corresponds to different input dimensionalities) and with associated sequence of probabilistic measures {?? 1 , ?? 2 , ...} (?? i ??? M(R i )) is smooth if there exists a nonincreasing function f : R ??? R such that f (x) ??? 0 as x ??? ??? and furthermore the k th moments of random variables X n = w , where w ??? ?? n satisfy for every n, k ??? 0: DISPLAYFORM0 Many important classes of RBF kernels are smooth, in particular the class of Gaussian kernels.

This follows immediately from the well-known fact that for Gaussian kernels the above k th moments are given by the following formula: DISPLAYFORM1 n 2 ???1)! for n > 1.

Our main result is given below and shows that orthogonal random features lead to tighter bounds on ??? for the spectral approximation of K + ??N I N .

Tighter bounds on ???, as Theorem 1 explains, lead to tighter upper bounds also on the empirical risk of the estimator.

We will prove it for the setting where each structured block consists of a fixed number l > 1 of rows (note that many independent structured blocks are needed if m > n), however our experiments suggest that the results are valid also without this assumption.

Theorem 2 (spectral approximation).

Consider a smooth RBF (in particular Gaussian kernel).

Let ??? iid denote the smallest positive number such that K iid + ??N I N is a ???-approximation of K + ??N I N , where K iid is an approximate kernel matrix obtained by using unstructured random features.

Then for any a > 0, DISPLAYFORM2 where: p iid N,m is given as: DISPLAYFORM3 2 for some universal constant C > 0, m is the number of random features used, ?? min is the smallest singular value of K + ??N I N and N is dataset size.

If instead orthogonal random features are used then for the corresponding spectral parameter ??? ort the following holds: We see that both constructions lead to exponentially small (in the number of random features m used) probabilities of failure, however the bounds are tighter for the orthogonal case.

An exact formula on p ort N,m can be derived from the proof that we present in the Appendix, however for clarity we do not give it here.

DISPLAYFORM4 Theorem 2 combined with Theorem 1 lead to risk bounds for the kernel ridge regression model based on random unstructured and random orthogonal features.

We use the notation introduced before and obtain the following: Theorem 3.

Under the assumptions of Theorem 1 and Theorem 2, the following holds for the kernel ridge regression risk and any c > 0 if m-dimensional unstructured random feature maps are used to approximate a kernel: DISPLAYFORM5 , where a c is given as: DISPLAYFORM6 and the probability is taken with respect to the random choices of features.

If instead random orthogonal features are used, we obtain the following bound: DISPLAYFORM7 As before, since for large n function p ort N,m satisfies p ort N,m < p iid N,m , for orthogonal random features we obtain strictly smaller upper bounds on the failure probability regarding empirical risk than for the state-of-the-art unstructured ones.

In practice, as we will show in the experimental section, we see gains also in the regimes of moderate dimensionalities n.

In section 5 we extended the theoretical guarantees for ORFs to the case of the initialization phase of OPSRNNs.

In this section we confirm these results experimentally and show that they imply better performance of the entire model by comparing the performance of PSRNNs with that of OPSRNNs on a selection of robotics time-series datasets.

Since OPSRNN models obtained via continuousorthogonal and discrete-orthogonal joint sampling (see: Section 4) gave almost the same results, presented OPSRNN-curves are for the continuous-orthogonal setting.

We now describe the datasets and model hyperparameters used in our experiments.

All models were implemented using the Tensorflow framework in Python.

We use the following datasets in our experiments:??? Swimmer We consider the 3-link simulated swimmer robot from the open-source package OpenAI gym.2 .

The observation model returns the angular position of the nose as well as the (2D) angles of the two joints, giving in a total of 5 features.

We collect 25 trajectories from a robot that is trained to swim forward (via the cross entropy with a linear policy), with a train/test split of 20/5.??? Mocap A Human Motion Capture dataset consisting of 48 skeletal tracks from three human subjects collected while they were walking.

The tracks have 300 time steps each, and are from a Vicon motion capture system.

We use a train/test split of 40/8.

There are 22 total features consisting of the 3D positions of the skeletal parts (e.g., upper back, thorax, clavicle).??? Handwriting This is a digital database available on the UCI repository (Alpaydin & Alimoglu, 1998) created using a pressure sensitive tablet and a cordless stylus.

Features are x and y tablet coordinates and pressure levels of the pen at a sampling rate of 100 milliseconds giving a total of 3 features.

We use 25 trajectories with a train/test split of 20/5.??? Moving MNIST Pairs of MNIST digits bouncing around inside of a box according to ideal physics.

http://www.cs.toronto.edu/??nitish/unsupervised_video/.

Each video is 64x64 pixels single channel (4096 features) and 20 frames long.

We use 1000 randomly selected videos, split evenly between train and test.

In two-stage regression we use history (similarly future) features consisting of the past (next) 2 observations concatenated together.

We use a ridge-regression parameter of 10 (???2) (this is consistent with the values suggested in Boots et al. FORMULA1 ; Downey et al. FORMULA1 ).

The kernel width is set to the median pairwise (Euclidean) distance between neighboring data points.

We use a fixed learning rate of 0.1 for BPTT with a BPTT horizon of 20.

We use a single layer PSRNN.We optimize and evaluate all models with respect to the Mean Squared Error (MSE) of one step predictions (this should not be confused with the MSE of the pointwise kernel approximation which does not give the downstream guarantees we are interested in here).

This means that to evaluate the model we perform recursive filtering on the test set to produce states, then use these states to make predictions about observations one time step in the future.

In our first experiment we examine the effectiveness of Orthogonal RF with respect to learning a good PSRNN via 2SR.

In figure 3 we compare the MSE for a PSRNN learned via Orthogonal RF with that of one learned using Standard RF for varying numbers of random features.

Note that these models were initialized using 2SR but were not refined using BPTT.

We see that in all cases when the ratio of RF to input dimension is small Orthogonal RF significantly outperforms Standard RF.

This difference decreases as the number of RF increases, with both approaches resulting in similar MSE for large RF to input ratios.

In our second experiment we examine the effectiveness of Orthogonal RF with respect to learning a good PSRNN via 2SR initialization combined with refinement via BPTT.

In figure 4 we compare the MSE for a PSRNN learned via Orthogonal RF with that of one learned using Standard RF over a number of epochs of BPTT.

We see that on all datasets, for both Orthogonal RF and Standard RF, MSE decreases as the number of epochs increases.

However it is interesting to note that in all datasets Orthogonal RF converges to a better MSE than Standard RF.

These results demonstrate the effectiveness of Orthogonal RF as a technique for improving the performance of downstream applications.

First we have shown that Orthogonal RF can offer significant performance improvements for kernel ridge regression, specifically in the context of the 2SR algorithm for PSRNNs.

Furthermore we have shown that not only does the resulting model have lower error, it is also a better initialization for the BPTT gradient descent procedure.

In other words, using a model initialization based on orthogonal RF results in BPTT converging to a superior final model.

While the focus of these experiments was to compare the performance of PSRNNs and OPSRNNs, for the sake of completeness we also include error plots for LSTMs.

We see that OPSRNNs significantly outperform LSTMs on all data sets.

We showed how structured orthogonal constructions can be effectively integrated with recurrent neural network based architectures to provide models that consistently achieve performance superior to the baselines.

They also provide significant compression, achieving similar accuracy as PSRNNs with an order of magnitude smaller number of features needed.

Furthermore, we gave the first theoretical guarantees showing that orthogonal random features lead to exponentially small bounds on the failure probability regarding empirical risk of the kernel ridge regression model.

The latter one is an important component of the RNN based architectures for state prediction that we consider in this paper.

Finally, we proved that these bounds are strictly better than for the standard nonorthogonal random feature map mechanism.

Exhaustive experiments conducted on several robotics task confirm our theoretical findings.

We will use the notation from the main body of the paper.

We will assume that a dataset X = {x 1 , ..., x N } under consideration is taken from a ball B of a fixed radius r (that does not depend on data dimensionality n and dataset size N ) and center x 0 .

We begin with the following lemma: DISPLAYFORM0 Consider a randomized kernel estimator K with a corresponding random feature map: ?? m,n : R n ??? R 2m and assume that for any fixed i, j ??? {1, ..., N } the followig holds for any c > 0: DISPLAYFORM1 for some fixed function g : R ??? R. Then with probability at least 1 ??? N 2 g(c), DISPLAYFORM2 ??min , where ?? min stands for the minimal singular value of K + ??I N .Proof.

Denote K + ??N I N = V ?? 2 V, where an orthonormal matrix V ??? R N ??N and a diagonal matrix ?? ??? R N ??N define the eigendecomposition of K + ??N I N .

Following Avron et al. FORMULA1 , we notice that in order to prove that K + ??I N is a ???-spectral approximation of K + ??I N , it suffices to show that: DISPLAYFORM3 From basic properties of the spectral norm 2 and the Frobenius norm F we have: DISPLAYFORM4 The latter probability is equal to DISPLAYFORM5 Furthermore, since V is an isometry matrix, we have: DISPLAYFORM6 ??min .

Thus we have: DISPLAYFORM7 Now notice that from the union bound we get: DISPLAYFORM8 Therefore the probability that DISPLAYFORM9 and that completes the proof.

Our goal right now is to compute function g from Lemma 1 for random feature maps constructed according to two procedures: the standard one based on independent sampling and the orthogonal one, where marginal distributions corresponding to the joint distribution (w 1 , ..., w m ) are the same, but vectors w i are conditioned to be orthogonal.

We start with a standard random feature map mechanism.

Note first that from basic properties of the trigonometric functions we conclude that for any two vectors x, y ??? R n , the random feature map approximation of the RBF kernel K(x, y) which is of the form K(x, y) = |?? m,n (x) ?? m,n (y) can be equivalently rewritten as: K(x, y) = 1 m m i=1 cos(w i z) for z = x ??? y.

This is true for any joint distribution (w 1 , ..., w m ).

Lemma 2.

If mapping ?? m,n is based on the standard mechanism of independent sampling then one can take as function g from Lemma 1 a function given by the following formula: g(x) = e ???Cmx 2 for some universal constant C > 0.Proof.

Notice first that by the remark above, we get: Zi ] for a parameter t > 0 (that is then being optimized),as it is the case for the standard Chernoff's argument.

Now, since variables Z i are independent, we obtained: DISPLAYFORM10 DISPLAYFORM11 .

Thus, if we can prove that for the continuous-orthogonal distribution we have: DISPLAYFORM12 , then we complete the proof of Theorem 2 (note that the marginal distributions of Z i are the same for both: standard mechanism based on unstructured random features and the one based on continuous-orthogonal sampling of the m-tuple of n-dimensional vectors).

This is what we prove below.

Lemma 3.

Fix some z ??? R n and t > 0.

For a sample (w ort 1 , ..., w ort m ) from the continuousorthogonal distribution the following holds for n large enough: DISPLAYFORM13 Proof.

Since different blocks of vectors w i used to construct the orthogonal feature map are independent (the number of blocks is greater than one if m > n), it suffices to prove the inequality just for one block.

Thus from now on we will focus just on one block and thus without loss of generality we will assume that m ??? n.

where FORMULA44 Thus we need to prove that DISPLAYFORM0 Using Taylor expansion for e x , we conclude that it sufficies to prove that: DISPLAYFORM1 i.e. that: DISPLAYFORM2 where: DISPLAYFORM3 By applying the trigonometric formula: DISPLAYFORM4 we get: DISPLAYFORM5 where ??? 1 stands for vector-addition operator and ??? ???1 stands for vector-subtraction operator.

Note that without loss of generality we can assume that s j1 = s j1+j2 = ... = +1, since for other configurations we obtain a random variable of the same distribution.

Consider a fixed configuration (s 1 , s 2 , ..., s j1+...+jm ) and the corresponding term of the sum above that can be rewritten in the compressed way as: DISPLAYFORM6 21) for some n 1 , ..., n m ??? Z. Without loss of generality, we can assume that n 1 , ..., n m ??? N, since the distribution of the random variables under consideration does not change if n i is replaced with ???n i .

Without loss of generality we can also assume that there exists i ??? {1, ..., m} such that n i > 0, since otherwise the corresponding term F is equal to 0.Denote by R 1 , ..., R m the set of independent random variables, where each is characterized by the distribution which is the distribution of vectors w Therefore, by expanding cos(x) using Taylor expansion, we get: DISPLAYFORM7 where DISPLAYFORM8 It is easy to see that for odd k we have: A(n, k) = 0.

We obtain: DISPLAYFORM9 The following technical fact will be useful:Lemma 4.

Expression A(k, n) is given by the following formula: DISPLAYFORM10 which can be equivalently rewritten (by computing the integrals explicitly) as: DISPLAYFORM11 where: ??(n = 2) = 2 if n = 2 and ??(n = 2) = 1 otherwise and: ??(n = 2) = 1 if n = 2 and ??(n = 2) = 2 otherwise.

In particular, the following is true: DISPLAYFORM12 We will use that later.

Note that v i v j ??? v z.

Therefore we obtain: DISPLAYFORM13 where DISPLAYFORM14 and ?? = i,j???{1,...,m} n i n j R i R j v i v j .Note that E[(R 2 ) k???1 ??] = 0 since E[(v i v j )] = 0 and furthermore, directions v 1 , ..., v m are chosen independently from lengths R 1 , ..., R m .

Therefore we have: DISPLAYFORM15 where: DISPLAYFORM16 Now let us focus on a single term ?? = E[(R 2 ) k???l ?? l ] for some fixed l ??? 3.Note that the following is true: DISPLAYFORM17 where the maximum is taken over i 1 , j 1 , ..., i l , j l ??? {1, ..., m} such that i s = j s for s = 1, ..., l.

Note first that: DISPLAYFORM18 Let us focus now on the expression max i1,j1,...,i l ,j l E[|v i1 v j1 | ?? ... ?? |v i l v j l |].We will prove the following upper bound on max i1,j1,...,i l ,j l E[|v i1 v j1 | ??

... ?? |v i l v j l |].Lemma 5.

The following is true: DISPLAYFORM19 n ??? ??? n log(n) ) l + l(e ??? log 2 (n) 4 + e ??? log 2 (n) 2).Proof.

Note that from the isotropicity of Gaussian vectors we can conclude that each single |v is v js | is distributed as: .

Thus, by taking: a = ??? n log(n), x = log(n) and applying union bound, we conclude that with probability at least 1 ??? e ??? log 2 (n) 4 ??? e ??? log 2 (n) 2 a fixed random variable |v is v js | satisfies: |v is v js | ??? log(n) ??? n??? ??? n log(n).

Thus, by the union bound we conclude that for any fixed i 1 , j 1 , ..., i l , j l random variable |v i1 v j1 | ?? ... ?? |v i l v j l | satisfies: DISPLAYFORM20 ) l with probability at least 1 ??? l(e ??? log 2 (n) 4 + e ??? log 2 (n) 2).

Since |v i1 v j1 | ?? ... ?? |v i l v j l | is upper bounded by one, we conclude that: ).

DISPLAYFORM21 Using Lemma 5, we can conclude that: DISPLAYFORM22 Therefore we have: DISPLAYFORM23 Thus we can conclude that for n large enough: DISPLAYFORM24 Thus we get: DISPLAYFORM25 where: DISPLAYFORM26 B satisfies: B = Ak and A is given as: DISPLAYFORM27

<|TLDR|>

@highlight

Improving Predictive State Recurrent Neural Networks via Orthogonal Random Features

@highlight

Proposes improving the performances of Predicitve State Recurrent Neural Networks by considering Orthogonal Random Features.

@highlight

The paper tackles the problem of training predictive state recurrent neural networks and makes two contributions.