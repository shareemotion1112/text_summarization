Determinantal Point Processes (DPPs) provide an elegant and versatile way to sample sets of items that balance the point-wise quality with the set-wise diversity of selected items.

For this reason, they have gained prominence in many machine learning applications that rely on subset selection.

However, sampling from a DPP over a ground set of size N is a costly operation, requiring in general an O(N^3) preprocessing cost and an O(Nk^3) sampling cost for subsets of size k. We approach this problem by introducing DppNets: generative deep models that produce DPP-like samples for arbitrary ground sets.

We develop an inhibitive attention mechanism based on transformer networks that captures a notion of dissimilarity between feature vectors.

We show theoretically that such an approximation is sensible as it maintains the guarantees of inhibition or dissimilarity that makes DPP so powerful and unique.

Empirically, we demonstrate that samples from our model receive high likelihood under the more expensive DPP alternative.

Selecting a representative sample of data from a large pool of available candidates is an essential step of a large class of machine learning problems: noteworthy examples include automatic summarization, matrix approximation, and minibatch selection.

Such problems require sampling schemes that calibrate the tradeoff between the point-wise quality -e.g.

the relevance of a sentence to a document summary -of selected elements and the set-wise diversity 1 of the sampled set as a whole.

Determinantal Point Processes (DPPs) are probabilistic models over subsets of a ground set that elegantly model the tradeoff between these often competing notions of quality and diversity.

Given a ground set of size N , DPPs allow for O(N 3 ) sampling over all 2 N possible subsets of elements, assigning to any subset S of a ground set Y of elements the probability DISPLAYFORM0 where L ∈ R N ×N is the DPP kernel and L S = [L ij ] i,j∈S denotes the principal submatrix of L indexed by items in S. Intuitively, DPPs measure the volume spanned by the feature embedding of the items in feature space (Figure 1 ).

BID31 to model the distribution of possible states of fermions obeying the Pauli exclusion principle, the properties of DPPs have since then been studied in depth BID19 BID6 , see e.g.).

As DPPs capture repulsive forces between similar elements, they arise in many natural processes, such as the distribution of non-intersecting random walks BID22 , spectra of random matrix ensembles BID37 BID13 , and zerocrossings of polynomials with Gaussian coefficients BID20 ).

More recently, DPPs have become a prominent tool in machine learning due to their elegance and tractability: recent applications include video recommendation BID10 , minibatch selection BID46 , and kernel approximation BID28 BID35 .However, the O(N 3 ) sampling cost makes the practical application of DPPs intractable for large datasets, requiring additional work such as subsampling from Y, structured kernels (Gartrell et al., (a) (b) (c) φ i φ j Figure 1 : Geometric intuition for DPPs: let φ i , φ j be two feature vectors of Φ such that the DPP kernel verifies L = ΦΦ T ; then P L ({i, j}) ∝ Vol(φ i , φ j ).

Increasing the norm of a vector (quality) or increasing the angle between the vectors (diversity) increases the volume spanned by the vectors BID25 , Section 2.2.1).2017; BID34 , or approximate sampling methods BID2 BID27 BID0 .

Nonetheless, even such methods require significant pre-processing time, and scale poorly with the size of the dataset.

Furthermore, when dealing with ground sets with variable components, pre-processing costs cannot be amortized, significantly impeding the application of DPPs in practice.

These setbacks motivate us to investigate the use of more scalable models to generate high-quality, diverse samples from datasets to obtain highly-scalable methods with flexibility to adapt to constantly changing datasets.

Specifically, we use generative deep models to approximate the DPP distribution over a ground set of items with both fixed and variable feature representations.

We show that a simple, carefully constructed neural network, DPPNET, can generate DPP-like samples with very little overhead, while maintaining fundamental theoretical properties of DPP measures.

Furthermore, we show that DPPNETs can be trivially employed to sample from a conditional DPP (i.e. sampling S such that A ⊆ S is predefined) and for greedy mode approximation.

• We introduce DPPNET, a deep network trained to generate DPP-like samples based on static and variable ground sets of possible items.• We derive theoretical conditions under which the DPPNETs inherit the DPP's log-submodularity.• We show empirically that DPPNETs provide an accurate approximation to DPPs and drastically speed up DPP sampling.

DPPs belong to the class of Strongly Rayleigh (SR) measures; these measures benefit from the strongest characterization of negative association between similar items; as such, SR measures have benefited from significant interest in the mathematics community BID39 BID5 BID4 BID32 and more recently in machine learning BID3 BID27 BID35 .

This, combined with their tractability, makes DPPs a particularly attractive tool for subset selection in machine learning, and is one of the key motivations for our work.

The application of DPPs to machine learning problems spans fields from document and video summarization BID14 BID9 , recommender systems BID47 BID10 and object retrieval BID1 to kernel approximation BID28 , neural network pruning BID33 , and minibatch selection BID46 .

BID48 developed DPP priors for encouraging diversity in generative models and BID40 showed that DPPs accurately model inhibition in neural spiking data.

In the general case, sampling exactly from a DPP requires an initial eigendecomposition of the kernel matrix L, incurring a O(N 3 ) cost.

In order to avoid this time-consuming step, several approximate sampling methods have been derived; BID0 approximate the DPP kernel during sampling; more recently, results by BID2 followed by BID27 showed that DPPs are amenable to efficient MCMC-based sampling methods.

return S Exact methods that significantly speed up sampling by leveraging specific structure in the DPP kernel have also been developed BID34 BID12 .

Of particular interest is the dual sampling method introduced in BID25 : if the DPP kernel can be composed as an inner product over a finite basis, i.e. there exists a feature matrix Φ ∈ R N ×D such that the DPP kernel is given by L = ΦΦ , exact sampling can be done in DISPLAYFORM0 However, MCMC sampling requires variable amounts of sampling rounds, which is unfavorable for parallelization; dual DPP sampling requires an explicit feature matrix Φ. Motivated by recent work on modeling set functions with neural networks BID45 BID11 , we propose instead to generate approximate samples via a generative network; this allows for simple parallelization while simultaneously benefiting from recent improvements in specialized architectures for neural network models (e.g. parallelized matrix multiplications).

We furthermore show that, extending the abilities of dual DPP sampling, neural networks may take as input variable feature matrices Φ and sample from non-linear kernels L.

In this section, we build up a framework that allows the O(N 3 ) computational cost associated with DPP sampling to be addressed via approximate sampling with a neural network.

Given a positive semi-definite matrix L ∈ R N ×N , we take P L to represent the distribution modeled by a DPP with kernel L over the power set of DISPLAYFORM0

Although the elegant quality/diversity tradeoff modeled by DPPs is a key reason for their recent success in many different applications, they benefit from other properties that make them particularly well-suited to machine learning problems.

We now focus on how these properties can be maintained with a deep generative model with the right architecture.

BID7 .

Although conditioning comes at the cost of an expensive matrix inversion, this property make DPPs well-suited to applications requiring diversity in conditioned sets, such as basket completion for recommender systems.

Standard deep generative models such as (Variational) Auto-Encoders BID23 (VAEs) and Generative Adversarial Networks BID15 ) (GANs) would not enable simple conditioning operations during sampling.

Instead, we develop a model that given an input set S, returns a prediction vector v ∈ R N such that DISPLAYFORM0 where Y ∼ P L : in other words, v i is the marginal probability of item i being included in the final set, given that S is a subset of the final set.

Mathematically, we can compute v i as DISPLAYFORM1 for i ∈ S BID25 ; for i ∈ S, we simply set v i = 0.With this architecture, we sample a set via Algorithm 1, which allows for trivial basket-completion type conditioning operations.

Furthermore, Algorithm 1 can be modified to implement a greedy sampling algorithm without any additional cost.

Log-submodularity.

As mentioned above, DPPs are included in the larger class of Strongly Rayleigh (SR) measures over subsets.

Although being SR is a delicate property, which is maintained by only few operations BID5 ), log-submodularity 3 (which is implied by SR-ness) is more robust, as well as a fundamental property in discrete optimization BID44 BID8 BID16 and machine learning BID9 .

Crucially, we show in the following that (log)-submodularity can be inherited by a generative model trained on a log-submodular distribution: THEOREM 1.

Let P be a strictly submodular function over subsets of Y, and Q be a function over the same space such that DISPLAYFORM2 where D TV indicates the total variational distance.

Then Q is also submodular.

COROLLARY 1.1.

Let P L be a strictly log-submodular DPP over Y and DPPNET be a network trained on the DPP probabilities p(S), with a loss function of the form p − q where · is a norm and p ∈ R 2 N (resp.

q) is the probability vector assigned by the DPP (resp.

the DPPNET) to each subset of Y. Let α = max x ∞=1 1 x .

If DPPNET converges to a loss smaller than DISPLAYFORM3 its generative distribution is log-submodular.

The result follows directly from Thm.

3 and the equivalence of norms in finite dimensional spaces.

REMARK 1.

Cor.

1.1 is generalizable to the KL divergence loss D KL (P Q) via Pinsker's inequality.

For this reason, we train our models by minimizing the distance between the predicted and target probabilities, rather than optimizing the log-likelihood of generative samples under the true DPP.Leveraging the sampling path.

When drawing samples from a DPP, the standard DPP sampling algorithm (Kulesza & Taskar, 2012, Alg.

1) generates the sample as a sequence, adding items one after the other until reaching a pre-determined size 4 , similarly to Alg.

1.

We take advantage of this by recording all intermediary subsets generated by the DPP when sampling training data: in practice, instead of training on n subsets of size k, we train on kn subsets of size 0, . . .

, k − 1.

Thus, our model is very much like an unrolled recurrent neural network.

In the simplest setting, we may wish to draw many samples over a ground set with a fixed feature embedding.

In this case, we wish to model a DPP with a fixed kernel via a generative neural network.

Specifically, we consider a fixed DPP with kernel L and wish to obtain a generative model such that DISPLAYFORM0 More generally, in many cases we may care about sampling from a DPP over a ground set of items that varies: this may be the case for example with a pool of products that are available for sale at a given time, or social media posts with a relevance that varies based on context.

To leverage the speed-up provided by dual DPP sampling, we can only sample from the DPP with kernel given by L = ΦΦ ; for more complex kernels, we once again incur the O(N 3 ) cost.

Furthermore, training a static neural network for each new feature embedding may be too costly.

Instead, we augment the static DPPNET to include the feature matrix Φ representing the ground set of all items as input to the network.

Specifically, we draw inspiration for the dot-product attention introduced in BID41 .

In the original paper, the attention mechanism takes as input 3 matrices: the keys K, the values V , and the query Q. Attention is computed as DISPLAYFORM1 where d is the dimension of each query/key: the inner product acts as a proxy to the similarity between each query and each key.

Finally, the reweighted value matrix AV is fed to the trainable neural network.

DISPLAYFORM2 Figure 2: Transformer network architecture for sampling from a variable ground set.

Here, the feature representation of items in the input set S acts as the query Q ∈ R k×d ; the feature representation Φ ∈ R N ×d of our ground set is both the keys and the values.

In order for the attention mechanism to make sense in the framework of DPP modeling, we make two modifications to the attention in BID41 ):• We want our network to attend to items that are dissimilar to the query (input subset): for each item i in the input subset S, we compute its pairwise dissimilarity to each item in Y as the vector DISPLAYFORM3 • Instead of returning this k × N matrix D of dissimilarities d i , we return a vector a ∈ R N in the probability simplex such that a j ∝ i∈S D ij .

This allows us to have a fixed-size input to the neural network, and simultaneously enforces the desirable property that similarity to a single item is enough to disqualify an element from the ground set.

Note that we could also return D in the form of a N × N matrix, but this would be counterproductive to speeding up DPP sampling.

Putting everything together, our attention vector a is computed via the inhibitive attention mechanism DISPLAYFORM4 where represents the row-wise multiplication operator; this vector can be computed in O(kDN ) time.

The attention component of the neural network finally feeds the element-wise multiplication of each row of V with a to the feed-forward component.

Given Φ and a subset S, the network is trained as in the static case to learn the marginal probabilities of adding any item in Y to S under a DPP with a kernel L dependent on Φ. In practice, we set L to be an exponentiated quadratic kernel L ij = exp(−β φ i − φ j 2 ) constructed with the features φ i .REMARK 2.

Dual sampling for DPPs as introduced in BID25 ) is efficient only when sampling from a DPP with kernel L = ΦΦ ; for non-linear kernels, a low-rank decomposition of L(Φ) must first be obtained, which in the worst case requires O(N 3 ) operations.

In comparison, the dynamic DPPNET can be trained on any DPP kernel, while only requiring Φ as input.

To evaluate DPPNET, we look at its performance both as a proxy for a static DPP (Section 4.1) and as a tool for generating diverse subsets of varying ground sets (Section 4.2).

Our models are trained with TensorFlow, using the Adam optimizer.

Hyperparameters are tuned to maximize the normalized log-likelihood of generated subsets.

We compare DPPNET to DPP performance as well as two additional baselines:• UNIF: Uniform sampling over the ground set.• k-MEDOIDS: The k-medoids clustering algorithm (Hastie et al., 2001, 14.3.10) , applied to items in the ground set, with distance between points computed as the same distance metric used by the DPP.

Conversely to k-means, k-MED uses data points as centers for each cluster.

We use the negative log-likelihood (NLL) of a subset under a DPP constructed over the ground set to evaluate the subsets obtained by all methods.

This choice is motivated by the following considerations: DPPs have become a standard way of measuring and enforcing diversity over subsets of data in machine learning, and b) to the extent of our knowledge, there is no other standard method to benchmark the diversity of a selected subset that depends on specific dataset encodings.

We begin by analyzing the performance of a DPPNET trained on a DPP with fixed kernel over the unit square.

This is motivated by the need for diverse sampling methods on the unit hypercube, motivated by e.g. quasi-Monte Carlo methods, latin hypercube sampling BID36 and low discrepancy sequences.

The ground set consists of the 100 points lying at the intersections of the 10 × 10 grid on the unit square.

The DPP is defined by setting its kernel L to L ij = exp(− x i − x j 2 2 /2).

As the DPP kernel is fixed, these experiments exclude the inhibitive attention mechanism.

We report the performance of the different sampling methods in FIG2 .

Visually FIG2 ) and quantitively FIG2 ), DPPNET improves significantly over all other baselines.

The NLL of DPPNET samples is almost identical to that of true DPP samples.

Furthermore, greedily sampling the mode from the DPPNET achieves a better NLL than DPP samples themselves.

Numerical results are reported in TAB0 .

We evaluate the performance of DPPNETs on varying ground set sizes through the MNIST (LeCun & Cortes, 2010), CelebA BID30 , and MovieLens BID17 datasets.

For MNIST and CelebA, we generate feature representations of length 32 by training a Variational Auto-Encoder BID23 on the dataset 5 ; for MovieLens, we obtain a feature vector for each movie by applying nonnegative matrix factorization the rating matrix, obtaining features of length 10.

Experimental results presented below were obtained using feature representations obtained via the test instances of each dataset.

The DPPNET is trained based on samples from DPPs with a linear kernel for MovieLens and with an exponentiated quadratic kernel for the image datasets.

Bandwidths were set to β = 0.0025 for MNIST and β = 0.1 for CelebA in order to obtain a DPP average sample size ≈ 20: recall that for a DPP with kernel L, the expected sample size is given by the formula For MNIST, FIG3 shows images selected by the baselines and the DPPNET, chosen among 100 digits with either random labels or all identical labels; visually, DPPNET and DPP samples provide a wider coverage of writing styles.

However, the NLL of samples from DPPNET decay significantly, whereas the DPPNET mode continues to maintain competitive performance with DPP samples.

DISPLAYFORM0 Numerical results for MNIST are reported in Table 2 ; additionally to the previous baselines, we also consider two further ways of generating subsets.

INHIBATTN samples items from the multinomial distribution generated by the inhibitive attention mechanism only (without the subsequent neural network).

NOATTN is a pure feed-forward neural network without attention; after hyper-parameter tuning, we found that the best architecture for this model consisted in 6 layers of 585 neurons each.

Table 2 reveals that both the attention mechanism and the subsequent neural network are crucial to modeling DPP samples.

Strikingly, DPPNET performs significantly better than other baselines even on feature matrices drawn from a single class of digits (Table 2) , despite the training distribution over feature matrices being much less specialized.

This implies that DPPNET sampling for dataset summarization may be leveraged to focus on sub-areas of datasets that are identified as areas of interest.

Numerical results for CelebA and MovieLens are reported in TAB2 , confirming the modeling ability of DPPNETs.

Finally, we verify that DPPNET allows for significantly faster sampling by running DPP and DPPNET sampling for subsets of size 20 drawn from a ground set of size 100 with both a standard DPP Table 2 : NLL (mean ± standard error) under the true DPP of samples drawn uniformly, according to the mode of the DPPNET, and from the DPP itself.

We sample subsets of size 20; for each class of digits we build 25 feature matrices Φ from encodings of those digits, and for each feature matrix we draw 25 different samples.

Bolded numbers indicate the best-performing (non-DPP) sampling method.

TAB0 49.2 ± 0.1 52.2 ± 0.1 60.5 ± 0.1 49.8 ± 0.0 50.7 ± 0.1 51.0 ± 0.1 50.4 ± 0.1 51.6 ± 0.1 51.5 ± 0.1 50.9 ± 0.1 52.7 ± 0.1 UNIF 51.6 ± 0.1 54.9 ± 0.1 65.1 ± 0.1 51.5 ± 0.1 52.9 ± 0.1 53.3 ± 0.1 52.4 ± 0.1 54.6 ± 0.1 55.1 ± 0.1 53.3 ± 0.1 56.2 ± 0.1 MEDOIDS 51.0 ± 0.1 55.1 ± 0.1 65.0 ± 0.1 51.5 ± 0.0 52.9 ± 0.1 53.1 ± 0.1 52.4 ± 0.0 54.4 ± 0.1 55.1 ± 0.1 53.2 ± 0.1 56.1 ± 0.1 INHIBATTN 51.3 ± 0.1 54.7 ± 0.1 65.0 ± 0.1 51.4 ± 0.1 52.8 ± 0.1 53.0 ± 0.1 52.1 ± 0.1 54.5 ± 0.1 54.9 ± 0.1 53.2 ± 0.1 55.9 ± 0.1 NOATTN 51.4 ± 0.1 54.9 ± 0.1 65.4 ± 0.1 51.5 ± 0.1 52.9 ± 0.1 53.3 ± 0.1 52.2 ± 0.1 54.6 ± 0.1 55.2 ± 0.1 53.3 ± 0.1 56.1 ± 0.1 DPPNET MODE 48.6 ± 0.2 53.6 ± 0.3 63.6 ± 0.4 50.8 ± 0.2 51.4 ± 0.3 51.6 ± 0.4 51.8 ± 0.3 52.8 ± 0.3 52.7 ± 0.4 50.9 ± 0.3 55.0 ± 0.4 BID29 according to root mean squared error (RMSE) and wallclock time.

We observe that subsets selected by DPPNET achieve comparable and lower RMSE than a DPP and the MCMC method respectively while being significantly faster.and DPPNET (using the MNIST architecture).

Both methods were implemented in graph-mode TensorFlow.

Sampling batches of size 32, standard DPP sampling costs 2.74 ± 0.02 seconds; DPPNET sampling takes 0.10 ± 0.001 seconds, amounting to an almost 30-fold speed improvement.

As a final experiment, we evaluate DPPNET's performance on a downstream task for which DPPs have been shown to be useful: kernel reconstruction using the Nyström method BID38 BID43 .

Given a positive semidefinite matrix K ∈ R N ×N , the Nyström method DISPLAYFORM0 where K † denotes the pseudoinverse of K and K ·,S (resp.

K S,· ) is the submatrix of K formed by its rows (resp.

columns) indexed by S. The Nyström method is a popular method to scale up kernel methods and has found many applications in machine learning (see e.g. (Bac; She; Fow; Tal)).

Importantly, the approximation quality directly depends on the choice of subset S. Recently, DPPs have been shown to be a competitive approach for selecting S BID35 BID28 .

Following the approach of BID28 , we evaluate the quality of the kernel reconstruction by learning a regression kernel K on a training set, and reporting the prediction error on the test set using the Nyström reconstructed kernelK. Additionally to the full DPP, we also compare DPPNET to the MCMC sampling method with quadrature acceleration BID28 c) FIG4 reports our results on the Ailerons dataset 6 also used in BID28 .

We start with a ground set size of 1000 and compute the resulting root mean squared error (RMSE) of the regression using various sized subsets selected by sampling from a DPP, the MCMC method of BID29 , using the full ground set and DPPNET.

FIG4 reports the runtimes for each method.

We note that while all methods were run on CPU, DPPNet is more amenable to acceleration using GPUs.

We introduced DPPNETs, generative networks trained on DPPs over static and varying ground sets which enable fast and modular sampling in a wide variety of scenarios.

We showed experimentally on several datasets and standard DPP applications that DPPNETs obtain competitive performance as evaluated in terms of NLLs, while being amenable to the extensive recent advances in speeding up computation for neural network architectures.

Although we trained our models on DPPs on exponentiated quadratic and linear kernels; we can train on any kernel type built from a feature representations of the dataset.

This is not the case for dual DPP exact sampling, which requires that the DPP kernel be L = ΦΦ for faster sampling.

DPPNETs are not exchangeable: that is, two sequences i 1 , . . .

, i k and σ(i 1 ), . . . , σ(i k ) where σ is a permutation of [k], which represent the same set of items, will not in general have the same probability under a DPPNET.

Exchangeability can be enforced by leveraging previous work BID45 ; however, non-exchangeability can be an asset when sampling a ranking of items.

Our models are trained to take as input a fixed-size subset representation; we aim to investigate the ability to take a variable-length encoding as input as future work.

The scaling of the DPPNET's complexity with the ground set size also remains an open question.

However, standard tricks to enforce fixed-size ground sets such as sub-sampling from the dataset may be applied to DPPNETs.

Similarly, if further speedups are necessary, sub-sampling from the ground set -a standard approach for DPP sampling over very large set sizes -can be combined with DPPNET sampling.

In light of our results on dataset sampling, the question of whether encoders can be trained to produce encodings conducive to dataset summarization via DPPNETs seems of particular interest.

Assuming knowledge of the (encoding-independent) relative diversity of a large quantity of subsets, an end-to-end training of the encoder and the DPPNET simultaneously may yield interesting results.

Finally, although Corollary 1.1 shows the log-submodularity of the DPP can be transferred to a generative model, understanding which additional properties of training distributions may be conserved through careful training remains an open question which we believe to be of high significance to the machine learning community in general.

A MAINTAINING LOG-SUBMODULARITY IN THE GENERATIVE MODEL THEOREM 2.

Let p be a strictly submodular distribution over subsets of a ground set Y, and q be a distribution over the same space such that DISPLAYFORM0 Then q is also submodular.

Proof.

In all the following, we assume that S, T are subsets of a ground set Y such that S = T and S, T ∈ {∅, Y} (the inequalities being immediate in these corner cases).

For the MNIST encodings, the VAE encoder consists of a 2d-convolutional layer with 64 filters of height and width 4 and strides of 2, followed by a 2d convolution layer with 128 filters (same height, width and strides), then by a dense layer of 1024 neurons.

The encodings are of length 32.

CelebA encodings were generated by a VAE using a Wide Residual Network BID44 ) encoder with 10 layers and filter-multiplier k = 4, a latent space of 32 full-covariance Gaussians, and a deconvolutional decoder trained end-to-end using an ELBO loss.

In detail, the decoder architecture consists of a 16K dense layer followed by a sequence of 4 × 4 convolutions with [512, 256, 128, 64] filters interleaved with 2× upsampling layers and a final 6 × 6 convolution with 3 output channels for each of 5 components in a mixture of quantized logistic distributions representing the decoded image.

<|TLDR|>

@highlight

We approximate Determinantal Point Processes with neural nets; we justify our model theoretically and empirically.