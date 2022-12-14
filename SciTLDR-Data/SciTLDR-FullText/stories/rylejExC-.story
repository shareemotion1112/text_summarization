Graph convolutional networks (GCNs) are powerful deep neural networks for graph-structured data.

However, GCN computes nodes' representation recursively from their neighbors, making the receptive field size grow exponentially with the number of layers.

Previous attempts on reducing the receptive field size by subsampling neighbors do not have any convergence guarantee, and their receptive field size per node is still in the order of hundreds.

In this paper, we develop a preprocessing strategy and two control variate based algorithms to further reduce the receptive field size.

Our algorithms are guaranteed to converge to GCN's local optimum regardless of the neighbor sampling size.

Empirical results show that our algorithms have a similar convergence speed per epoch with the exact algorithm even using only two neighbors per node.

The time consumption of our algorithm on the Reddit dataset is only one fifth of previous neighbor sampling algorithms.

Graph convolution networks (GCNs) BID1 generalize convolutional neural networks (CNNs) (LeCun et al., 1995) to graph structured data.

The "graph convolution" operation applies same linear transformation to all the neighbors of a node, followed by mean pooling.

By stacking multiple graph convolution layers, GCNs can learn nodes' representation by utilizing information from distant neighbors.

GCNs have been applied to semi-supervised node classification BID1 , inductive node embedding (Hamilton et al., 2017a) , link prediction (Kipf & Welling, 2016; BID1 and knowledge graphs (Schlichtkrull et al., 2017) , outperforming multi-layer perceptron (MLP) models that do not use the graph structure and graph embedding approaches (Perozzi et al., 2014; Tang et al., 2015; Grover & Leskovec, 2016 ) that do not use node features.

However, the graph convolution operation makes it difficult to train GCN efficiently.

A node's representation at layer L is computed recursively by all its neighbors' representations at layer L − 1.

Therefore, the receptive field of a single node grows exponentially with respect to the number of layers, as illustrated in FIG0 .

Due to the large receptive field size, BID1 proposed training GCN by a batch algorithm, which computes the representation for all the nodes altogether.

However, batch algorithms cannot handle large scale datasets because of their slow convergence and the requirement to fit the entire dataset in GPU memory.

Hamilton et al. (2017a) made an initial attempt on developing stochastic algorithms to train GCNs, which is referred as neighbor sampling (NS) in this paper.

Instead of considering all the neighbors, they randomly subsample D (l) neighbors at the l-th layer.

Therefore, they reduce the receptive field size to l D (l) , as shown in FIG0 (b).

They found that for two layer GCNs, keeping D (1) = 10 and D (2) = 25 neighbors can achieve comparable performance with the original model.

However, there is no theoretical guarantee on the predictive performance of the model learnt by NS comparing with the original algorithm.

Moreover, the time complexity of NS is still D(1) D (2) = 250 times larger than training an MLP, which is unsatisfactory.

In this paper, we develop novel stochastic training algorithms for GCNs such that D (l) can be as low as two, so that the time complexity of training GCN is comparable with training MLPs.

Our methods are built on two techniques.

First, we propose a strategy which preprocesses the first graph convolution layer, so that we only need to consider all neighbors within L−1 hops instead of L hops.

This is significant because most GCNs only have L = 2 layers BID1 ; Hamilton et al., 2017a) .

Second, we develop two control variate (CV) based stochastic training algorithms.

We show that our CV-based algorithms have lower variance than NS, and for GCNs without dropout, our algorithm provably converges to a local optimum of the model regardless of D (l) .We empirically test on six graph datasets, and show that our techniques significantly reduce the bias and variance of the gradient from NS with the same receptive field size.

Our algorithm with D (l) = 2 achieves the same predictive performance with the exact algorithm in comparable number of epochs on all the datasets, while the training time is 5 times shorter on our largest dataset.

We now briefly review graph convolutional networks (GCNs) BID1 and the neighbor sampling (NS) algorithm (Hamilton et al., 2017a) .

The original GCN was presented in a semi-supervised node classification task BID1 .

We follow this setting throughout this paper.

Generalization of GCN to other tasks can be found in Kipf & Welling (2016); BID1 Schlichtkrull et al. (2017) and Hamilton et al. (2017b) .

In the node classification task, we have an undirected graph G = (V, E) with V = |V| vertices and E = |E| edges, where each vertex v consists of a feature vector x v and a label y v .

The label is only observed for some vertices V L and we want to predict the label for the rest vertices V U := V\V L .

The edges are represented as a symmetric V × V adjacency matrix A, where A v,v is the weight of the edge between v and v , and the propagation matrix P is a normalized version of A:Ã = A + I,D vv = v Ã vv , and P =D DISPLAYFORM0 .

A graph convolution layer is defined as DISPLAYFORM1 where H (l) is the activation matrix in the l-th layer, whose each row is the activation of a graph node.

H (0) = X is the input feature matrix, W (l) is a trainable weight matrix, σ(·) is an activation function, and Dropout p (·) is the dropout operation (Srivastava et al., 2014) with keep probability p.

, where f (·, ·) can be the square loss, cross entropy loss, etc., depending on the type of the label.

When P = I, GCN reduces to a multi-layer perceptron (MLP) model which does not use the graph structure.

Comparing with MLP, GCN is able to utilize neighbor information for node classification.

We define n(v, L) as the set of all the L-neighbors of node v, i.e., the nodes that are reachable from v within L hops.

It is easy to see from FIG0 that in an L-layer GCN, a node uses the information from all its L-neighbors.

This makes GCN more powerful than MLP, but also complicates the stochastic training, which utilizes an approximated gradient ∇L ≈ DISPLAYFORM0 v ), where V B ⊂ V L is a minibatch of training data.

The large receptive field size | ∪ v∈V B n(v, L)| per minibatch leads to high time complexity, space complexity and amount of IO.

See Table 1 for the average number of 1-and 2-neighbors of our datasets.

We introduce alternative notations to help compare different algorithms.

Let DISPLAYFORM0 v , we focus on studying how u v is computed based on node v's neighbors.

To keep notations simple, we omit all the subscripts and tildes, and exchange the ID of nodes such TAB4 : Number of vertexes, edges, average number of 1-and 2-neighbors per node for each dataset.

Undirected edges are counted twice and self-loops are counted once.

Reddit is already subsampled to have a max degree of 128 following Hamilton et al. (2017a) .

DISPLAYFORM1 DISPLAYFORM2 | is the number of neighbors.

We get the propagation rule u = D v=1 p v h v , which is used interchangeably with the matrix form U (l) = PH (l) .

To reduce the receptive field size, Hamilton et al. (2017a) propose a neighbor sampling (NS) algorithm.

On the l-th layer, they randomly choose D (l) neighbors for each node, and develop an estimator u N S of u based on Monte-Carlo approximation DISPLAYFORM0 In this way, they reduce the receptive field size from DISPLAYFORM1 Neighbor sampling can also be written in a matrix form as DISPLAYFORM2 whereP (l) is a sparser unbiased estimator of P , i.e., EP DISPLAYFORM3 used for testing and for computing stochastic gradient DISPLAYFORM4 CV,v ) during training.

The NS estimator u N S is unbiased.

However it has a large variance, which leads to biased prediction and gradients after the non-linearity in subsequent layers.

Due to the biased gradients, training with NS does not converge to the local optimum of GCN.

When D (l) is moderate, NS may has some regularization effect like dropout (Srivastava et al., 2014) , where it drops neighbors instead of features.

However, for the extreme ease D (l) = 2, the neighbor dropout rate is too high to reach high predictive performance, as we will see in Sec. 5.4.

Intuitively, making prediction solely depends on one neighbor is inferior to using all the neighbors.

To keep comparable prediction performance with the original GCN, Hamilton et al. (2017a) use relatively large D(1) = 10 and D (2) = 25.

Their receptive field size D(1) × D (2) = 250 is still much larger than MLP, which is 1.

We first present a technique to preprocess the first graph convolution layer, by approximating ADropout p (X) with Dropout p (AX).

The model becomes DISPLAYFORM0 This approximation does not change the expectation because E ADropout p (X) = E Dropout p (AX) , and it does not affect the predictive performance, as we shall see in Sec. 5.1.The advantage of this modification is that we can preprocess U (0) = P H (0) = P X and takes Uas the new input.

In this way, the actual number of graph convolution layers is reduced by one -the first layer is merely a fully connected layer instead of a graph convolution one.

Since most GCNs only have two graph convolution layers BID1 Hamilton et al., 2017a) , this gives a significant reduction of the receptive field size from the number of DISPLAYFORM1 The numbers are reported in Table 1 .

We now present two novel control variate based estimators that have smaller variance as well as stronger theoretical guarantees than NS.

We assume that the model does not have dropout for now and will address dropout in Sec. DISPLAYFORM0 where v is a random neighbor, and ∆h v = h v −h v .

For the ease of presentation, we assume that we only use the latest activation of one neighbor, while the implementation also include the node itself besides the random neighbor, so D (l) = 2.

Using historical activations is cheap because they need not to be computed recursively using their neighbors' activations, as shown in FIG0 .

Unlike NS, we apply Monte-Carlo approximation on v p v ∆h v instead of v p v h v .

Since we expect h v andh v to be close, ∆h v will be small and u CV should have a smaller variance than u N S .

Particularly, if the model weight is kept fixed,h v should be eventually equal with h v , so that Ripley, 2009, Chapter 5) , which has zero mean and large correlation with u N S , to reduce its variance.

We refer this stochastic approximation algorithm as CV, and we will formally analyze the variance and prove the convergence of the training algorithm using CV for stochastic gradient in subsequent sections.

DISPLAYFORM1 In matrix form, CV computes the approximate predictions as follows, where we explicitly write down the iteration number i and add the subscript CV to the approximate activations DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 CV,i,v stores the latest activation of node v on layer l computed before time i. Formally, let m (l) i ∈ R V ×V be a diagonal matrix, and (m DISPLAYFORM5 After finishing one iteration we update historyH with the activations computed in that iteration as Eq. (6).

With dropout, the activations H are no longer deterministic.

They become random variables whose randomness come from different dropout configurations.

Therefore, ∆h v = h v −h v is not necessarily small even if h v andh v have the same distribution.

We develop another stochastic approximation algorithm, control variate for dropout (CVD), that works well with dropout.

Our method is based on the weight scaling procedure (Srivastava et al., 2014) to approximately compute the mean DISPLAYFORM0 That is, along with the dropout model, we can run a copy of the model with no dropout to obtain the mean µ v , as illustrated in FIG0 .

With the mean, we can obtain a better stochastic approximation by separating the mean and variance DISPLAYFORM1 whereμ v is the historical mean activation, obtained by storing µ v instead of h v , and ∆µ = µ v −μ v .

u CV D an unbiased estimator of u because the term √ Dp v (h v − µ v ) has zero mean, and the Monte-Carlo approximation DISPLAYFORM2 is made by assuming h v 's to be independent Gaussians, which we will soon clarify.

The pseudocodes for CV and CVD are in Appendix E.

We analyze their variance in a simple independent Gaussian case, where we assume that activations are Gaussian random variables Table 2 : Variance of different algorithms in the independent Gaussian case.

Manning (2013) .

Without loss of generality, we assume that all the activations h v are one dimensional.

We also assume that all the activations h 1 , . . .

, h D and historical activationsh 1 , . . .

,h D are independent, where the historical DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 We introduce a few more notations.

∆µ v and ∆s Table 2 , where the derivations can be found in Appendix C.

We decompose the variance as two terms: variance from Monte-Carlo approximation (VMCA) and variance from dropout (VD).If the model has no dropout, the activations have zero variance, i.e., s v =s v = 0, and the only source of variance is VMCA.

We want VMCA to be small.

As in Table 2 , the VMCA for the exact estimator is 0.

For the NS estimator, VMCA is DISPLAYFORM3 2 , whose magnitude depends DISPLAYFORM4

Similarly, VMCA for both CV and CVD estimators is DISPLAYFORM0 , which is likely because ∆µ v should be smaller than µ v .

Since CV and CVD estimators have the same VMCA, we adopt the CV estimator for models without dropout, due to its simplicity.

The VD of the exact estimator is s 2 , which is overestimated by both NS and CV.

NS overestimates VD by D times, and CV has even larger VD.

Meanwhile, the VD of the CVD estimator is the same as the exact estimator, indicating CVD to be the best estimator for models with dropout.

Besides smaller variance, CV also has stronger theoretical guarantees than NS.

We can show that during testing, CV's prediction becomes exact after a few testing epochs.

For models without dropout, we can further show that training using the stochastic gradients obtained by CV converges to GCN's local optimum.

We present these results in this section and Sec. 4.5.

Note that the analysis does not need the independent Gaussian assumption.

Given a model W , we compare the exact predictions (Eq. 1) and CV's approximate predictions (Eq. 5,6) during testing, which uses the deterministic weight scaling procedure.

To make predictions, we run forward propagation by epochs.

In each epoch, we randomly partition the vertex set V as I minibatches V 1 , . . .

, V I and in the i-th iteration, we run a forward pass to compute the prediction for nodes in V i .

Note that in each epoch we scan all the nodes instead of just testing nodes, to ensure that the activation of each node is computed at least once per epoch.

The following theorem reveals the connection of the exact predictions and gradients, and their approximate versions by CV.Theorem 1.

For a fixed W and any i > LI we have: (1) (Exact Prediction) The activations computed by CV are exact, i.e., Z (l) DISPLAYFORM0 Theorem 1 shows that at testing time, we can run forward propagation with CV for L epoches and get the exact prediction.

This outperforms NS, which cannot recover the exact prediction.

Comparing with directly making exact predictions by a batch algorithm, CV is more scalable because it does not need to load the entire graph into memory.

The proof can be found in Appendix A.

The following theorem shows that for a model without dropout, training using CV's approximated gradients converges to a local optimum of GCN, regardless of the neighbor sampling size D (l) .

Therefore, we can choose arbitrarily small D (l) without worrying about the convergence.

Theorem 2.

Assume that (1) all the activations are ρ-Lipschitz, (2) the gradient of the cost func- DISPLAYFORM0 is the inner product of matrix A and matrix B. We randomly run SGD for R ≤ N iterations, where DISPLAYFORM1 .

Then, for the updates W i+1 = W i − γ i g CV (W i ) and step sizes DISPLAYFORM2 }, there exists constants K 1 and K 2 which are irrelevant with N , s.t.

∀N > LI, DISPLAYFORM3 The proof can be found in Appendix B. Particularly, DISPLAYFORM4 Therefore, our algorithm converges to a local optimum.

Finally we discuss the time complexity of different algorithms.

We decompose the time complexity as sparse time complexity for sparse-dense matrix multiplication such as PH (l) , and dense time complexity for dense-dense matrix multiplication such as U (1) times higher than NS.Our implementation is similar as BID1 .

We store the node features in the main memory, without assuming that they fit in GPU memory as Hamilton et al. (2017a) , which makes our implementation about 2 times slower than theirs.

We keep the histories in GPU memory for efficiency since they are only LH < K dimensional.

We examine the variance and convergence of our algorithms empirically on six datasets, including Citeseer, Cora, PubMed and NELL from BID1 and Reddit, PPI from Hamilton et al.(2017a), as summarized in Table 1 .

To measure the predictive performance, we report Micro-F1 for the multi-label PPI dataset, and accuracy for all the other multi-class datasets.

We use the same model architectures with previous papers but slightly different hyperparameters (see Appendix D for the details).

We repeat the convergence experiments 10 times on Citeseer, Cora, PubMed and NELL, and 5 times on Reddit and PPI.

The experiments are done on a Titan X (Maxwell) GPU.

We first examine the approximation in Sec. 3 that switches the order of dropout and aggregating the neighbors.

Let M0 be the original model (Eq. 1) and M1 be our approximated model (Eq. 3), we compare three settings: (1) M0, D (l) = ∞ is the exact algorithm without any neighbor sampling.

(2) M1+PP, D (l) = ∞ changes the model from M0 to M1.

Preprocessing does not affect the training for DISPLAYFORM0 uses NS with a relatively large number of neighbors.

In Table 3 we can see that all the three settings performs similarly, i.e., our approximation does not affect the predictive performance.

Therefore, we use M1+PP, D (l) = 20 as the exact baseline in following convergence experiments because it is the fastest among these three settings.

We now study how fast our algorithms converge with a very small neighbor sampling size D (l) = 2.

We compare the following algorithms: (1) Exact, which is M1+PP, D We first validate Theorem 2, which states that CV+PP converges to a local optimum of Exact, for models without dropout, regardless of D (l) .

We disable dropout and plot the training loss with respect to number of epochs as FIG1 .

We can see that CV+PP can always reach the same training loss with Exact, which matches the conclusion of Theorem 2.

Meanwhile, NS and NS+PP have a higher training loss because their gradients are biased.

Next, we compare the predictive accuracy obtained by the model trained by different algorithms, with dropout turned on.

We use different algorithms for training and the same Exact algorithm for testing, and report the validation accuracy at each training epoch.

The result is shown in FIG5 .

We find that CVD+PP is the only algorithm that is able to reach comparable validation accuracy with Exact on all datasets.

Furthermore, its convergence speed with respect to the number of epochs is comparable with Exact despite its D (l) is 10 times smaller.

Note that CVD+PP performs much better than Exact on the PubMed dataset; we suspect it finds a better local optimum.

Meanwhile, simper algorithms CV+PP and NS+PP work acceptably on most of the datasets.

CV+PP reaches a comparable accuracy with Exact for all datasets except PPI.

NS+PP works slightly worse but the final validation accuracy is still within 2%.

These algorithms can be adopted if there is no strong need for predictive performance.

We however emphasize that exact algorithms must be used for making predictions, as we will show in Sec. 5.4.

Finally, the algorithm NS without preprocessing works much worse than others, indicating the significance of our preprocessing strategy.

TAB4 reports the average number of epochs, time, and total number of floating point operations to reach a given 96% validation accuracy on the largest Reddit dataset.

Sparse and dense computations are defined in Sec. 4.6.

We found that CVD+PP is about 5 times faster than Exact due to the significantly reduced receptive field size.

Meanwhile, simply setting D (l) = 2 for NS does not converge to the given accuracy.

We compare the quality of the predictions made by different algorithms, using the same model trained by Exact in Fig. 4 .

As Thm.

1 states, CV reaches the same testing accuracy as Exact, while NS and NS+PP perform much worse.

Testing using exact algorithms (CV or Exact) corresponds to the weight scaling algorithm for dropout (Srivastava et al., 2014) .Finally, we compare the average bias and variance of the gradients per dimension for first layer weights relative to the weights' magnitude in Fig. 5 .

For models without dropout, the gradient of CV+PP is almost unbiased.

For models with dropout, the bias and variance of CV+PP and CVD+PP are ususally smaller than NS and NS+PP, as we analyzed in Sec. 4.3.

The large receptive field size of GCN hinders its fast stochastic training.

In this paper, we present a preprocessing strategy and two control variate based algorithms to reduce the receptive field size.

Our algorithms can achieve comparable convergence speed with the exact algorithm even the neighbor sampling size D (l) = 2, so that the per-epoch cost of training GCN is comparable with training MLPs.

We also present strong theoretical guarantees, including exact prediction and convergence to GCN's local optimum, for our control variate based algorithm.

DISPLAYFORM0 H (l+1) DISPLAYFORM1 After one more epoch, all the activations h (l+1)CV,i,v are computed at least once for each v, soH DISPLAYFORM2 for all i > (l + 2)I. By induction, we know that after LI steps, we havē DISPLAYFORM3 2.

We omit the time subscript i and denote DISPLAYFORM4 CV,v ).

By back propagation, the approximated gradients by CV can be computed as follows DISPLAYFORM5 where • is the element wise product and σ (Z DISPLAYFORM6 CV ) is the element-wise derivative.

Similarly, denote DISPLAYFORM7 v ), the exact gradients can be computed as follows DISPLAYFORM8 Applying EP = EP (1) ,...,P (L) to both sides of Eq. 8, and utilizing DISPLAYFORM9 we have DISPLAYFORM10 Comparing Eq. 10 and Eq. 9 we get DISPLAYFORM11

We proof Theorem 2 in 3 steps:1.

Lemma 1: For a sequence of weights W(1) , . . .

, W (N ) which are close to each other, CV's approximate activations are close to the exact activations.

(1) , . . .

, W (N ) which are close to each other, CV's gradients are close to be unbiased.3.

Theorem 2: An SGD algorithm generates the weights that changes slow enough for the gradient bias goes to zero, so the algorithm converges.

The following proposition is needed in our proof DISPLAYFORM0 is the number of columns of the matrix A.• DISPLAYFORM1 Proof.

DISPLAYFORM2 Proposition 2.

There are a series of T inputs X 1 , . . .

, X T , X CV,1 , . . . , X CV,T and weights W 1 , . . .

, W T feed to an one-layer GCN with CV DISPLAYFORM3 and an one-layer exact GCN DISPLAYFORM4 2.

X CV,i − X CV,j ∞ < and X CV,i − X i ∞ < for all i, j ≤ T and > 0.Then there exists some K > 0, s.t.

, H CV,i − H CV,j ∞ < K and H CV,i − H i ∞ < K for all I < i, j ≤ T , where I is the number of iterations per epoch.

Proof.

Because for all i > I, the elements ofX CV,i are all taken from previous epochs, i.e., X CV,1 , . . .

, X CV,i−1 , we know that DISPLAYFORM5 By triangular inequality, we also know DISPLAYFORM6 Since X CV,1 ∞ , . . .

, X CV,T ∞ are bounded, X CV,i ∞ is also bounded for i >

I. Then, DISPLAYFORM7 and DISPLAYFORM8 The following lemma bounds CV's approximation error of activations Lemma 1.

Given a sequence of model weights W 1 , . . .

, W T .

If W i − W j ∞ < , ∀i, j, and all the activations are ρ-Lipschitz, there exists K > 0, s.t.

, DISPLAYFORM9 Proof.

We prove by induction.

Because DISPLAYFORM10 Repeatedly apply Proposition B.1 for L − 1 times, we get the intended results.

The following lemma bounds the bias of CV's approximate gradient Lemma 2.

Given a sequence of model weights W 1 , . . .

, W T , if DISPLAYFORM0 2.

all the activations are ρ-Lipschitz, 3.

the gradient of the cost function ∇ z f (y, z) is ρ-Lipschitz and bounded, then there exists K > 0, s.t.

, DISPLAYFORM1 Proof.

By Lipschitz continuity of ∇ z f (y, z) and Lemma 1, there exists K > 0, s.t.

, DISPLAYFORM2 By Eq. 9, Eq. 10 and Lemma 1, we have DISPLAYFORM3 By induction we know that for l = 1, . . .

, L there exists K, s.t.

, DISPLAYFORM4 Again by Eq. 9, Eq. 10, and Lemma 1, DISPLAYFORM5 Finally, DISPLAYFORM6

Proof.

This proof is a modification of Ghadimi & Lan (2013) , but using biased stochastic gradients instead.

We assume the algorithm is already warmed-up for LI steps with the initial weights W 0 , so that Lemma 2 holds for step i > 0.

DISPLAYFORM0 By smoothness we have DISPLAYFORM1 Consider the sequence of LI + 1 weights W i−LI , . . .

, W i .

DISPLAYFORM2 By Lemma 2, there exists K > 0, s.t.

DISPLAYFORM3 where DISPLAYFORM4 Taking EP ,V B to both sides of Eq. 14 we have DISPLAYFORM5 Summing up the above inequalities and re-arranging the terms, we obtain, DISPLAYFORM6 Dividing both sides by DISPLAYFORM7 .

DISPLAYFORM8 Particularly, when N → ∞, we have E R∼P R EP ,V B ∇L(W R ) 2 = 0, which implies that the gradient is asymptotically unbiased.

We test 3-layer GCNs on the Reddit dataset.

The settings are the same with 2-layer GCNs in Sec. 5.3.

To ensure the exact algorithm can run in a reasonable amount of time, we subsample the graph so that the maximum degree is 10.

The convergence result is shown as FIG7 , which is similar with the two-layer models.

The time consumption to reach 0.94 testing accuracy is shown in TAB6 .

We justify the independent Gaussian assumption in Sec. 4.3 by showing that for a 2-layer GCN with the first layer pre-processed, the neighbor's activations are independent.

Without loss of generality, suppose that we want to compute z Assumption 1 is not GCN-specific and is discussed in Wang & Manning (2013), we now prove assumption 2 by the following lemma.

Lemma 3.

If a and b are independent random variables, then their transformations f 1 (a) and f 2 (b) are independent.

Because for any event A and B, P (f 1 (a) ∈ f 1 (A), f 2 (b) ∈ f 2 (B)) = P (a ∈ A, b ∈ B) = P (a ∈ A)P (b ∈ B) = P (f 1 (a) ∈ f 1 (A))P (f 2 (B) ∈ f 2 (B)), where f 1 (A) = {f 1 (a)|a ∈ A} and f 2 (B) = {f 2 (b)|b ∈ B}. v and hv are independent.

The result can be further generalized to deeper models.

If the receptive fields of two nodes does not overlap, they should be independent.where the indices i, j ∈n (l) (v).

Then, we compute the average correlation of all pairs of neighbors i = j.

AvgCorr (l,v,d) := 1 n (l) (v) ( n (l) (v) − 1) i =j Corr (l,v,d ) ij , and define the average neighbor correlation on layer l as AvgCorr (l,v,d) averaged over all the nodes v and dimensions d.

We report the average feature correlation and the average neighbor correlation per layer, on the Citeseer, Cora, PubMed and PPI datasets.

These quantities are too expensive to compute for NELL and Reddit.

On each dataset, we train a GCN with 10 graph convoluation layers until early stopping criteria is met, and compute the average feature correlation and the average neighbor correlation for layer 1 to 9.

We are not interested in the correlation on layer 10 because there are no more graph convolutional layers after it.

The result is shown as FIG10 .

As analyzed in Sec. G.1, the average neighbor correlation is close to zero on the first layer, but it is not exactly zero due to the finite sample size for computing the empirical covariance.

There is no strong tendency of increased correlation as the number of layers increases, after the third layer.

The average neighbor correlation and the average feature correlation remain on the same order of magnitude, so bringing correlated neighbors does not make the activations much more correlated than the MLP case (Wang & Manning, 2013) .

Finally, both correlations are much smaller than one.

@highlight

A control variate based stochastic training algorithm for graph convolutional networks that the receptive field can be only two neighbors per node.