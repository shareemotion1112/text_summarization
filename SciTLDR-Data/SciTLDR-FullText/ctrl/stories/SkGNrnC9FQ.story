We propose a novel framework for combining datasets via alignment of their associated intrinsic dimensions.

Our approach assumes that the two datasets are sampled from a common latent space, i.e., they measure equivalent systems.

Thus, we expect there to exist a natural (albeit unknown) alignment of the data manifolds associated with the intrinsic geometry of these datasets, which are perturbed by measurement artifacts in the sampling process.

Importantly, we do not assume any individual correspondence (partial or complete) between data points.

Instead, we rely on our assumption that a subset of data features have correspondence across datasets.

We leverage this assumption to estimate relations between intrinsic manifold dimensions, which are given by diffusion map coordinates over each of the datasets.

We compute a correlation matrix between diffusion coordinates of the datasets by considering graph (or manifold) Fourier coefficients of corresponding data features.

We then orthogonalize this correlation matrix to form an isometric transformation between the diffusion maps of the datasets.

Finally, we apply this transformation to the diffusion coordinates and construct a unified diffusion geometry of the datasets together.

We show that this approach successfully corrects misalignment artifacts, and allows for integrated data.

In biology and other natural science settings we often have the problem that data are measured from the same system but with different sensors or in different days where sensors are calibrated differently.

This is often termed batch effect in biology and can include, for example, drastic variations between subjects, experimental settings, or even times of day when an experiment is conducted.

In such settings it is important to globally and locally align the datasets such that they can be combined for effective further analysis.

Otherwise, measurement artifacts may dominate downstream analysis.

For instance, clustering the data will group samples by measurement time or sensor used rather than by biological or meaningful differences between datapoints.

Recent works regard the two datasets as views of the same system and construct a multiview diffusion geometry but all of them require at least partial bijection, if not full one, between views BID11 BID12 BID24 Tuia & Camps-Valls, 2016) .

Other work directly attempt to match data points directly in ambient space, or by local data geometry and these can be very sensitive to differences in sampling density rather than data geometry BID10 .

Here, we present a principled approach called harmonic alignment to for correct this type of effect based on the manifold assumption.

The manifold assumption holds that high dimensional data originates from an intrinsically low dimensional smoothly varying space that is mapped via nonlinear functions to observable high dimensional measurements.

Thus, we assume that the datasets are from transformed versions of the same low dimensional manifold.

We learn the manifolds separately from the two datasets using diffusion geometric approaches and then find an isometric transformation to map from one manifold to the other.

Note that we are not aligning points to points.

Indeed there may be sampling differences and density differences in the data.

However, our manifold learning approach uses an anisotropic kernel that detects the geometry of the data to align rather than point-by-point matching which is done in other methods.

Our method involves first embedding each dataset separately into diffusion components, and then finding an isometric transformation that aligns these diffusion representations.

To find such transformation, we utilize the duality between diffusion coordinates and geometric harmonics that act as generalized Fourier harmonics in the graph space.

The diffusion components are eigenvectors of a Markov-normalized data diffusion operator, whose eigenvalues indicate frequency of the eigenvector.

We attempt to find a transformation from one set of eigenvectors to another, via feature correspondences in the data.

While datapoint correspondences may be difficult or impossible to obtain since many biological measurements are destructive, feature correspondences are often available.

For instance single-cell measurements of cells from the same device, thus containing counts for the same genes, albeit affected by batch differences.

Thus when corresponding features are transformed via the graph Fourier transform (GFT) into diffusion coordinates, the representations should be similar, with potentially small frequency-proximal perturbations.

For instance, slowly varying features across the manifold should be load to low-frequency eigenvectors of the Markov matrix.

This insight allows us to create a correlation matrix between the eigenvectors of one dataset to another based on correlation between feature loadings to the eigenvectors.

However, since we know that eigenvectors represent frequency harmonics, we need not compute the entire correlation matrix but rather only the near-diagonal values.

This implies that the two manifolds must be perturbed such that low and high frequency eigenvector space are similar.

We then find a linear transformation between that maps the eigenvectors of one space into those of the other that maximizes these correlations by orthogonalizing this matrix.

This transformation allows us to align the two datasets with each other.

Finally, given an aligned representation, we build a robust unified diffusion geometry that is invariant batch effects and sample-specific artifacts and low-pass filter this geometry to denoise the unified manifold.

Thus, in addition to aligning the manifolds our method denoises the manifolds as well.

We demonstrate the results of our method on artificial manifolds created from rotated MNIST digits, corrupted MNIST digits, as well as on single-cell biological data measuring peripheral blood cells.

In each case our method successfully aligns the manifolds such that they have appropriate neighbors within and across the two datasets.

We show an application to transfer learning where a lazy classifier trained on one dataset is applied to the other dataset after alignment.

Further, comparisons with recently developed methods such as the MNN-based method of BID10 show significant improvements in performance and denoising ability.

A typical and effective assumption in machine learning is that high dimensional data originates from an intrinsic low dimensional manifold that is mapped via nonlinear functions to observable high dimensional measurements; this is commonly referred to as the manifold assumption.

Formally, let M d be a hidden d dimensional manifold that is only observable via a collection of n d nonlinear functions f 1 , . . .

, f n : DISPLAYFORM0 n from which data is collected.

Conversely, given a dataset X = {x 1 , . . .

, x N } ⊂ R n of high dimensional observations, manifold learning methods assume its data points originate from a sampling DISPLAYFORM1 . .

, n, and aim to learn a low dimensional intrinsic representation that approximates the manifold geometry of M d .A popular approach towards this manifold learning task is to construct diffusion geometry from data , and embed data into diffusion coordinates, which provide a natural global coordinate system derived from Laplace operator over manifold geometries, as explained in Section 2.1.

However, this approach, as well as other manifold learning ones, implicitly assumes that the feature functions {f j } n j=1 represent data collection technologies (e.g., sensors or markers) that operate in a consistent manner on all samples in Z. While this assumption may be valid in some fields, biological data collection (and more generally, data collection in empirical sciences) is often highly affected by a family of phenomena known as batch effects, which introduce nonnegligible variance between different data batches due to various uncontrollable factors.

These include, for example, drastic variations between subjects, experimental settings, or even times of day when an experiment is conducted.

Therefore, in such settings, one should consider a collection of S samples {X (s) } S s=1 , each originating from feature functions {f (s) j } n j=1 that aim to measure the same quantities in the data, but are also affected by sample-dependent artifacts.

While each sample can be analyzed to find its intrinsic structure, their union into a single dataset X = S s=1 X (s) often yields an incoherent geometry biased by batch effects, where neither the relations between samples or within each sample can be clearly seen.

To address such artifacts, and constrct a unified geometry of multiple batches (i.e., samples or datasets) together, we propose to first embed each batch separately in diffusion coordinates, and then find an isometric transformation that aligns these diffusion representations.

In order to find such transformation, we utilize the duality between diffusion coordinates and geometric harmonics that act as generalized Fourier harmonics, as shown in graph signal processing BID20 .

As explained in Section 2.2, this duality allows us to capture cross-batch relations between diffusion coordinates, and orthogonalize the resulting matrix to provide a map between batch-specific diffusion representations.

Finally, given an aligned representation, we build a robust unified diffusion geometry that is invariant to both batch effects and batch-specific artifacts.

While our approach generalizes naturally to any number of batches, for simplicity, we focus our formulation here on the case of two batches.

The first step in our approach is to capture the intrinsic geometry of each batch X (s) using the diffusion maps method from , which non-linearly embeds the data in a new coordinate system (i.e., diffusion coordinates) that is often considered as representing a data manifold or more generally a diffusion geometry over the data.

The diffusion maps construction starts by considering local similarities defined via a kernel K(x, y), x, y ∈ X (s) that capture local neighborhoods in the data.

We note that a popular choice for K is the Gaussian kernel e − x−y 2 σ , where σ > 0 is interpreted as a user-configurable neighborhood size.

Next, these similarities are normalized to defined transition probabilities p(x, y) = DISPLAYFORM0 that are organized in an N × N row stochastic matrix P that describes a Markovian diffusion process over the intrinsic geometry of the data.

Finally, a diffusion map is defined by taking the eigenvalues 1 = µ 1 ≥ µ 2 ≥ · · · ≥ µ N and (corresponding) eignevectors {φ j } N j=1 of P, and mapping each data point DISPLAYFORM1 T , where t represents a diffusion-time (i.e., number of transitions considered in the diffusion process).

In this work, we denote the diffusion map for the entire dataset DISPLAYFORM2 t .

We refer the reader to for further details and mathematical derivation, but note that in general, as t increases, most of the eigenvalue weights µ t j , j = 1, . . .

, N , become numerically negligible, and thus truncated diffusion map coordinates (i.e., using only nonnegligible ones) can be used for dimensionality reduction purposes.

Much work has been done in various fields on applications of diffusion maps as a whole, as well as individual diffusion coordinates (i.e., eigenvectors of P ), in data analysis BID8 BID2 BID14 BID1 BID9 .

In particular, as discussed in Coifman & Lafon FORMULA10 and BID15 , the diffusion coordinates are closely related to eigenvectors of Laplace operators on manifolds, as well as their discretizations as eigenvectors of graph Laplacians, which were studied previously, for example, in BID3 .

Indeed, the similarities measured in K can be considered as determining edge weights of a graph structure defined over the data.

Formally, we define this graph by considering every data point in X as a vertex on the graph, and then defining weighted edges between them via an N × N adjacency matrix W with DISPLAYFORM3 , and thus it can be verified that the eigenvectors of L can be written as ψ j = D 1/2 φ j , with corresponding eigenvalues λ j = 1 − µ j .

It should be noted that if data is uniformly sampled from a manifold (as considered in BID3 , these two sets of eigenvectors coincide and the diffusion coordinates can be considered as Laplacian eigenvectors (or eigenfunctions, in continuous settings).A central tenet of graph signal processing is that the Laplacian eigenfunctions {ψ j } N j=1 can be regarded as generalized Fourier harmonics BID20 , i.e., graph harmonics.

In- DISPLAYFORM4 Compute the anisotropic weight matrix W (s) (Section 2.2) and degree matrix D3:Construct the normalized graph Laplacian L (s) and its truncated eigensystem DISPLAYFORM5 Compute the diffusion map DISPLAYFORM6 The spectral domain wavelet transform tensorĤ DISPLAYFORM7 .

11: Embed E using a Gaussian kernel to obtain L (Y ) .deed, a classic result in spectral graph theory shows that the discrete Fourier basis can be derived as Laplacian eigenvectors of the ring graphs (see, e.g. Olfati-Saber, 2007, Proposition 10).

Based on this interpretation, a graph Fourier transform (GFT) is defined on graph signals (i.e., functions f : X (s) → R over the vertices of the graph) asf (λ j ) = f, ψ j , j = 1, . . .

, N , similar to the definition of the classic discrete Fourier transform (DFT).

Further, we can also write the GFT in terms of the diffusion coordinates asf (λ k ) = f, D 1/2 φ j , given their relation to graph harmonics.

Therefore, up to appropriate weighting, the diffusion coordinates can conceptually be interpreted as intrinsic harmonics of the data, and conversely, the graph harmonics can be considered (conceptually) as intrinsic coordinates of data manifolds.

In Section 2.2, we leverage this duality between coordinates and harmonics in order to capture relations between data manifolds of individual batches, and then them in Section 2.3 to align their intrinsic coordinates and construct a unified data manifold over them.

We now turn our attention to considering the relation between two batches X (s1) , X (s2) via their their intrinsic data manifold structure, as it is captured by diffusion coordinates or, equivalently, graph harmonics.

We note that, as discussed extensively in , a naïve construction of an intrinsic data graph with a Gaussian kernel (as described, for simplicity, in Section 2.1) may be severely distorted by density variations in the data.

Such distortion would detrimental in our case, as it would the resulting diffusion geometry and its harmonic structure would not longer reflect a stable (i.e., batch-invariant) intrinsic "shape" of the data.

Therefore, we follow the normalization suggested in there to separate data geometry from density, and define a graph structure (i.e., adjacency matrix) over each batch via an anistotropic kernel given by DISPLAYFORM0 where K is the previously defined Gaussian kernel.

This graph structure is then used, as previously described, to construct the intrinsic harmonic structure given by {ψ DISPLAYFORM1 on each batch.

While the intrinsic geometry constructed by our graph structures should describe similar "shapes" for the two datasets, there is no guarantee that their computed intrinsic coordinates will match.

Indeed, it is easy to see how various permutations of these coordinates can be obtained if some eigenvalues have multiplicities greater than one (i.e., their monotonic order is no longer deterministic), but even beyond that, in practical settings batch effects often result in various misalignments (e.g., rotations or other affine transformations) between derived intrinsic coordinates.

Therefore, to properly recover relations between multiple batches, we aim to quantify relations between their coordinates, or more accurately, between their graph harmonics.

We note that if we even a partial overlap between data points in the two batches, this task would be trivially enabled by taking correlations between these harmonics.

However, given that here we assume a setting without such predetermined overlap, we have to rely on other properties that are independent of individual data points.

To this end, we now consider the feature functions {f DISPLAYFORM2 and our initial assumption that corresponding functions aim to measure equivalent quantities in the batches (or datasets).

Therefore, while they may differ in the original raw form, we expect their expression over the intrinsic structure of the data (e.g., as captured by GFT coefficients) to correlate, at least partially, between batches.

Therefore, we use this property to compute crossbatch correlations between graph harmonics based on the GFT of corresponding data features.

To formulate this, it is convenient to extend the definition of the GFT from functions (or vectors) to matrices, by slight abuse of the inner product notation, asX DISPLAYFORM3 , where X consists of data features as columns and Ψ (s) has graph harmonics as columns (both with rows representing data points).Notice that the resulting Fourier matrixX (s) , for each batch, no longer depends on individual data points, and instead it expresses the graph harmonics in terms of data features.

Therefore, we can now use this matrix to formulate a cross-batch harmonic correlations by considering inner products between rows of these matrices.

Further, we need not consider all ther correlations between graph harmonics, since we also have access to their corresponding frequency information, expressed via the associated Laplacian eigenvalues {λ DISPLAYFORM4 j=1 .

Therefore, instead of computing correlations between every pair of harmonics across batches, we only consider them within local frequency bands, defined via appropriate graph filters, as exlpained in the following.

Let g(t) be a smooth window defined on the interval [−0.5, 0.5] as g(t) = sin 0.5π cos (πt) 2 .Then, by translating this window along the along the real line, we obtain τ equally spaced wavelet windows that can be applied to the eigenvalues λ (s) j in order to smoothly partition the spectrum of each data graph.

This construction is known as the itersine filter bank, which can be shown to be a tight frame BID18 .

The resulting windows g ξi (λ) are centered at frequencies Ξ = {ξ 1 , . . .

, ξ τ }.

The generating function for these wavelets ensures that each g ξi halfway overlaps with g ξi+1 .

This property implies that there are smooth transitions between the weights of consecutive frequency bands.

Furthermore, as a tight frame, this filterbank has the property that τ i=1 h ξi (λ) = 1 for any eigenvalue.

This choice ensures that any filtering we do using the filter bank G = {h ξi } τ i=1 (λ) will behave uniformly across the spectrum.

Together, these two properties imply that cross-batch correlations between harmonics within and between bands across the respective batch spectra will be robust.

To obtain such bandlimited correlations we construct the following filterbank tensorĤ , which we refer to as the harmonic (cross-batch) correlation matrix.

This step, when combined with the half-overlaps discussed above, allows flexibility in aligning harmonics across bands, which is demonstrated in practice in Section 2.3.

DISPLAYFORM5

Given the harmonic correlation matrix M (s1,s2) , we now define an isometric transformation between the intrinsic coordinate systems of the two data manifolds.

Such transformation ensures our alignment fits the two coordinate systems together without breaking the rigid structure of each batch, thus preserving their intrinsic structure.

To formulate such transformation, we recall that isometric transformations are given by orthogonal matrices, and thus we can rephrase our task as finding the best approximation of M (s1,s2) by an orthogonal matrix.

Such approximation is a well studied problem, dating back to BID19 , which showed that it can be obtained directly the singular value decomposition M = U SV T by taking T (s1,s2) = U V T .Finally, given the isometric transformation defined by T (s1,s2) , we can now align of the data manifolds of two batches, and define a unified intrinsic coordinate system for the entire data.

While such alignment could equivalently be phrased in terms of diffusion coordinates {φ DISPLAYFORM0 j=1 , we opt here for the latter, as it relates more directly to the computed harmonic correlations.

Therefore, we construct the transformed embedding matrix E as DISPLAYFORM1 .( FORMULA10 where we drop the superscript for T, as they are clear from context,Λ ( s) are diagonal matrices that consists of the nonzero Laplacian eigenvalues of each view, andΨ consist of the corresponding eigenvectors (i.e., harmonics) as its columns.

We note that the truncated of zero eigenvalues correspond to zero frequencies (i.e., flat constant harmonics), and therefore they only encode global shifts that we anyway aim to remove in the alignment process.

Accordingly, this truncation is also applied to the harmonic correlation matrix M (s1,s2) prior to its orthogonalization.

Finally, we note that this construction is equivalent to the diffusion map, albeit using a slightly different derivation of a discretized heat kernel (popular, for example, in graph signal processing works such as BID21 ), with the parameter t again serving an analogous purpose to diffusion time.

As a proof of principle we first demonstrate harmonic alignment of two circular manifolds.

To generate these manifolds, we rotated two different MNIST examples of the digit '3' 360 degrees and sampled a point for each degree (See FIG2 .

As we noted in section 2.2, the manifold coordinates obtained by diffusion maps are invariant to the phase of the data.

In this example it is clear that each '3' manifold is out of phase with the other.

FIG2 demonstrates the simple rotation that is learned by harmonic alignment between the two embeddings.

On the left side, we see the out-of-phase embeddings.

Taking nearest neighbors in this space illustrates the disconnection between the embeddings: nearest neighbors are only formed for within-sample points.

After alignment, however, we see that the embeddings are in phase with each other because nearest neighbors in the aligned space span both samples and are in the same orientation with each other.

Next, we assessed the ability of harmonic alignment in recovering k-neighborhoods after random feature corruption (figure 2).To do this, we drew random samples from MNIST X(1) and X (2) of N (1) = N (2) = 1000.

Then, for each trial in this experiment we drew 784 2 samples from a unit normal distribution to create a 784 × 784 random matrix.

We orthogonalized this matrix to yield the corruption matrix O 0 .

To vary the amount of feature corruption, we then randomly substituted 0.01 * p * 784 columns from I to make O p .

Right multiplication of X (2) by this matrix yielded corrupted images with only p% preserved pixels (figure 2b, 'corrupted').

To assess the recovery of k-neighborhoods, we then performed a lazy classification on X (2) O p by only using the labels of its neighbors in X (1) .

The results of this experiment, performed for p = {0, 5, 10, . . .

95, 100} are reported in figure 2a.

For robustness, at each p we sampled three different non-overlapping pairs X(1) and X (2) and for each pair we sampled three O p matrices each with random identity columns, for a total of nine trials per p.

In general, unaligned, mutual nearest neighbors (MNN), and harmonic alignment with any filter set cannot recover k-neighborhoods under total corruption; 10% serves as a baseline accuracy that results from the rotation having 10% overlap in the manifold space.

On the other hand, for small filter choices we observe that harmonic alignment quickly recovers ≥ 80% accuracy and outperforms MNN and unaligned classifications consistently except under high correspondence.

Next we examined the ability of harmonic alignment to reconstruct corrupted data (figure 2b).

We performed the same corruption procedure as before with p = 25 and selected 10 examples of each digit in MNIST.

We show the ground truth from X (2) and the corrupted result X (2) O 25 in figure 2b.

Then, a reconstruction was performed by setting each pixel in a new image to the dominant class average of the ten nearest X(1) neighbors.

In the unaligned case we see that most examples turn into smeared fives or ones; this is likely the intersection formed by X(1) and X (2) O 25 that accounts for the 10% baseline accuracy in figure 2a.

On the other hand, the reconstructions produced by harmonic alignment resemble their original input examples.

In figure 3 we compare the runtime, k-nn accuracy, and transfer learning capabilities of our method with two other contemporary alignment methods.

First, we examine the unsupervised algorithm proposed by BID25 for generating weight matrices between two different samples.

The algorithm first creates a local distance matrix of size k around each point and its four nearest neighbors.

Then it computes an optimal match between k-nn distance matrices of each pair of points in X(1) and X (2) by comparing all k!

permutations of the k-nn matrices and computing the minimal frobenius norm between such permuted matrices.

We report runtime results for k = 5, as k = 10 failed to complete after running for 8 hours.

Because the W ang&M ahadevan (2009) method merely gives a weight matrix that can be used with separate algorithms for computing the final features, we report accuracy results using their implementation.

Regardless of input size, we were unable to recover k-neighborhoods for datasets with 35% uncorrupted columns (figure 3b) despite the method's computational cost (figure 3a).A more scalable approach to manifold alignment has emerged recently in the computational biology BID10 and X (2) of N (1) = N (2) = 1000 points were sampled from MNIST.

X (2) was then distorted by a 784 × 784 corruption matrix O p for various identity percentages p (see section 3.2).

Subsequently, a lazy classification scheme was used to classify points in X (2) O p using a nearest neighbor vote from X Transfer learning performance.

For each ratio, 1,000 uncorrupted, labeled digits were sampled from MNIST.

1,000, 2,000, 4,000, and 8,000 (x-axis) unlabeled points were sampled and corrupted with 35% column identity.

The mean of three iterations of lazy classification for each method is reported.

compute mapping between datasets based on the assumption that if two points are truly neighbors they will resemble each other in both datasets.

Because this approach amounts to building a knearest neighbors graph for each dataset and then choosing a set of neighbors between each dataset, MNN scales comparably to our method (figure 3a.

Additionally, MNN is able to recover 20-30% of k-neighborhoods when only 35% of features match (figure 3b); this is an improvement over Wang but is substantially lower than what harmonic alignment achieves.

We note that the performance of harmonic alignment was correlated with input size whereas MNN did not improve with more points.

An interesting use of manifold alignment algorithms is transfer learning.

In this setting, an algorithm is trained to perform well on one dataset, and the goal is to extend the algorithm to the other dataset after alignment.

In figure 3c we explore the utility of harmonic alignment in transfer learning and compare it to MNN and the method proposed by BID25 .In this experiment, we first draw 1000 uncorrupted examples of MNIST digits, and construct a diffusion map to use as our training set.

Next, we took 65% corrupted unlabeled points (see section 3.2) in batches of 1, 000, 2, 000, 4, 000, 8, 000 as a test set to perform lazy classification on using the labels from the uncorrupted examples.

At 1:8 test:training sample sizes, Harmonic alignment outperformed Wang and MNN by aligning upto 60% correct k-neighborhoods.

In addition to showing the use of manifold alignment in transfer learning, this demonstrates the robustness of our algorithm to dataset imbalances.

To illustrate the need for robust manifold alignment in computational biology, we turn to a simple real-world example obtained from BID0 FIG5 ).

This dataset was collected by mass cytometry (CyTOF) of peripheral blood mononuclear cells (PBMC) from patients who contracted dengue fever.

Subsequently, the Montgomery lab at Yale University experimentally introduced these PBMCs to Zika virus strains.

The canonical response to dengue infection is upregulation of interferon gamma (IFNγ) BID6 BID5 BID4 .

During early immune response, IFNγ works in tandem with acute phase cytokines such as tumor necrosis factor α to induce febrile response and inhibit viral replication BID16 .

In the PBMC dataset, we thus expect to see upregulation of these two cytokines together, which we explore in 4.In FIG5 , we show the relationship between IFNγ and TNFα without denoising.

Note that there is a substantial difference between the IFNγ distributions of sample 1 and sample 2 (Earth Mover's Distance (EMD) = 2.699).

In order to identify meaningful relationships in CyTOF data, it is common to denoise it first.

We used a graph filter to denoise the cytokine data .

The results of this denoising are shown in FIG5 .

This procedure introduced more technical artifacts by enhancing the difference between batches, as seen by the increased EMD (3.127) between the IFNγ distributions of both patients.

This is likely due to a substantial connectivity difference between the two batch subgraphs in the total graph.

Next, we performed harmonic alignment of the two patient profiles.

We show the results of this in FIG5 .

Harmonic alignment corrected the difference between IFNγ distributions and restored the canonical correlation of IFNγ and TNFα.

This example illustrates the utlity of harmonic alignment for biological data, where it can be used for integrated analysis of data collected across different experiments, patients, and time points.

We presented a novel method for aligning or batch-normalizing two datasets that involves learning and aligning their intrinsic manifold dimensions.

Our method leverages the fact that common or corresponding features in the two datasets should have similar harmonics on the graph of the data.

Our harmonic alignment method finds an isometric transformation that maximizes the similarity of frequency harmonics of common features.

Results show that our method successfully aligns artifi- cially misaligned as well as biological data containing batch effect.

Our method has the advantages that it aligns manifold geometry and not density (and thus is insensitive to sampling differences in data) and further our method denoises the datasets to obtain alignments of significant manifold dimensions rather than noise.

Future applications of harmonic alignment can include integration of data from different measurement types performed on the same system, where features have known correlations.

<|TLDR|>

@highlight

We propose a method for aligning the latent features learned from different datasets using harmonic correlations.

@highlight

Proposes using feature correspondences to preform manifold alignment between batches of data from the same samples to avoid the collection of noisy measurements.