Unsupervised bilingual dictionary induction (UBDI) is useful for unsupervised machine translation and for cross-lingual transfer of models into low-resource languages.

One approach to UBDI is to align word vector spaces in different languages using Generative adversarial networks (GANs) with linear generators, achieving state-of-the-art performance for several language pairs.

For some pairs, however, GAN-based induction is unstable or completely fails to align the vector spaces.

We focus on cases where linear transformations provably exist, but the performance of GAN-based UBDI depends heavily on the model initialization.

We show that the instability depends on the shape and density of the vector sets, but not on noise; it is the result of local optima, but neither over-parameterization nor changing the batch size or the learning rate consistently reduces instability.

Nevertheless, we can stabilize GAN-based UBDI through best-of-N model selection, based on an unsupervised stopping criterion.

A word vector space -also sometimes referred to as a word embedding -associates similar words in a vocabulary with similar vectors.

Learning a projection of one word vector space into another, such that similar words -across the two word embeddings -are associated with similar vectors, is useful in many contexts, with the most prominent example being the alignment of vocabularies of different languages.

This is a key step in machine translation of low-resource languages ).

An embedding of English words may associate thoughtful, considerate, and gracious with similar vectors, for example, but for English-Icelandic translation, it would be useful to have access to a cross-lingual word embedding space in which hugulsamur (lit.: 'thoughtful') was also associated with a similar vector.

Such joint embeddings of words across languages can also be used to extract bilingual dictionaries.

Projections between word vector spaces have typically been learned from dictionary seeds.

In seminal papers such as BID22 and BID11 , these seeds would comprise thousands of words, but BID31 showed that we can learn reliable projections from as little as 50 words.

and BID15 subsequently showed that the seed can be replaced with just words that are identical across languages; and BID1 showed that numerals can also do the job, in some cases; both proposals removing the need for an actual dictionary.

Even more recently, a handful of papers have proposed an entirely unsupervised approach to projecting word vector spaces onto each other, based on Generative Adversarial Networks (GANs) BID12 .

We present the core idea behind such approaches in ??3, but briefly put, GANs are used to learn a linear transformation to minimize the divergence between a target distribution (say the Icelandic embeddings) and a source distribution (the English embeddings projected into the Icelandic space).The possibility of unsupervised bilingual dictionary induction (UBDI) has seemingly removed the data bottleneck in machine translation, evoking the idea that we can now learn to translate without human supervision ).

Yet, it remains an open question whether the initial, positive results extrapolate to real-world scenarios of learning translations between low-resource language pairs.

recently presented results suggesting that UBDI is challenged by some language pairs exhibiting very different morphosyntactic properties, as well as when the monolingual corpora are very different.

In this paper, we identify easy, hard, and impossible instances of GAN-based UBDI, and apply a simple test for discriminating between them.

The hard cases exhibit instability, i.e. their success depends heavily on initialization.

We set up a series of experiments to investigate these hard cases.

Our contributions We introduce a distinction between easy, hard, and impossible alignment problems over pairs of word vector spaces and show that a simple linearity test can be used to tell these cases apart.

We show that the impossible cases are caused not necessarily by linguistic differences, but rather by properties of the corpora and the embedding algorithms.

We also show that in the hard cases, the likelihood of being trapped in local minima depends heavily on the shape and density of the vector sets, but not on noise.

Changes in the number of parameters, batch size, and learning rate do not alleviate the instability.

Yet, using an unsupervised model selection method over N different initializations to select the best generators, leads to a 6.74% average error reduction over standard MUSE.Structure of the paper ??2 presents MUSE BID6 , an approach to GAN-based UBDI.

Here we also discuss theoretical results from the GAN literature, relevant to our case, and show a relation to a common point set registration method.

In ??3, we use a test based on Procrustes Analysis to discriminate between easy, hard, and impossible cases, discussing its relation with tests of isomorphism and isospectrality.

We then focus on the hard cases, where linear transformations provably exist, but GANs exhibit considerable instability.

Through a series of experiments, we analyze what affects the instability of GAN-based UBDI.

??4 presents our unsupervised best-of-N model selection method for stabilizing GAN-based UBDI.

In this section, we discuss the dynamics of GAN-based UBDI and how the training behavior of GANs can help us understand their limitations as applied to UBDI.

Two families of approaches to UBDI exist: using GANs BID3 BID6 BID34 and using iterative closest point BID16 .

We focus on GAN-based UBDI, and more specifically on MUSE BID6 , but at the end of this section we establish a relation between the two families of algorithms.

A GAN consists of a generator and a discriminator.

The generator G is trained to fool the discriminator D. The generator can be any differentiable function; in MUSE, it is a linear transform ???. Let e ??? E be an English word vector, and f ??? F a French word vector, both of dimensionality d. The goal of the generator is then to choose ??? ??? R d??d such that ???E has a distribution close to F .

The discriminator is a map D w : X ??? {0, 1}, implemented in MUSE as a multi-layered perceptron.

The objective of the discriminator is to discriminate between vector spaces F and ???E. During training, the model parameters ??? and w are optimized using stochastic gradient descent by alternately updating the parameters of the discriminator based on the gradient of the discriminator loss and the parameters of the generator based on the gradient of the generator loss, which, by definition, is the inverse of the discriminator loss.

The loss function used in MUSE and in our experiments below is cross-entropy.

In each iteration, we sample N vectors e ??? E and N vectors f ??? F and update the discriminator parameters w according to DISPLAYFORM0 Theoretically, the optimal parameters are a solution to the min-max problem: DISPLAYFORM1 If a generator wins the game against an ideal discriminator on a very large number of samples, then F and ???E can be shown to be close in Jensen-Shannon divergence, and thus the model has learned the true data distribution.

This result, referring to the distributions of the data, p data , and the distribution, p g , G is sampling from, is from BID12 : If G and D have enough capacity, and at each step of training, the discriminator is allowed to reach its optimum given G, and p g is updated so as to improve the criterion DISPLAYFORM2 then p g converges to p data .

This result relies on a number of assumptions that do not hold in practice.

The generator in MUSE, which learns a linear transform ???, has very limited capacity, for example, and we are updating ??? rather than p g .

In practice, therefore, during training, MUSE alternates between k steps of optimizing the discriminator and one step of optimizing the generator.

Another common problem with training GANs is that the discriminator loss quickly drops to zero, when there is no overlap between p g and p data (Arjovsky et al., 2017) ; but note that in our case, the discriminator is initially presented with IE and F , for which there is typically no trivial solution, since the embedding spaces are likely to overlap.

We show in ??4 that discriminator and generator loss are poor model selection criteria, however; instead we propose a simple criterion based on cosine similarities between nearest neighbors in the learned alignment.

From ???E and F , we can extract a bilingual dictionary using nearest neighbor queries, i.e., by asking what is the nearest neighbor of ???E in F , or vice versa.

MUSE uses a normalized nearest neighbor retrieval method to reduce the influence of hubs BID24 BID9 .

The method is called cross-domain similarity local scaling (CSLS) and used to expand high-density areas and condense low-density ones.

The mean similarity of a source language embedding ???e to its k nearest neighbors in the target language (k = 10 suggested) is defined as ?? DISPLAYFORM3 where cos is the cosine similarity.

?? F (f i ) is defined in an analogous manner for every i. CSLS(e, f i ) is then calculated as 2cos(e, DISPLAYFORM4 .

MUSE uses an unsupervised validation criterion based on CSLS.

The translations of the top 10k most frequent words in the source language are obtained with CSLS and average pairwise cosine similarity is computed over them.

This metric is considered indicative of the closeness between the projected source space and the target space, and is found to correlate well with supervised evaluation metrics.

After inducing a bilingual dictionary, E d and F d , by querying ???E and F with CSLS, MUSE performs a refinement step based on the Procrustes algorithm BID26 , whereby the singular value decomposition of DISPLAYFORM5 The idea of minimizing nearest neighbor similarity for unsupervised model selection is also found in point set registration and lies at the core of iterative closest point (ICP) optimization BID4 .

ICP typically minimizes the ?? 2 distance (mean squared error) between nearest neighbor pairs.

The ICP optimization algorithm works by assigning each transformed vector to its nearest neighbor and then computing the new relative transformation that minimizes the cost function with respect to this assignment.

ICP can be shown to converge to local optima BID4 , in polynomial time BID10 .

ICP easily gets trapped in local optima, however, exact algorithms only exist for two-and three-dimensional point set registration, and these algorithms are slow BID33 .

Generally, it holds that the optimal solution to the GAN min-max problem is also optimal for ICP.

To see this, note that a GAN minimizes the Jensen-Shannon divergence between F and ???E. The optimal solution to this is F = ???E. As sample size goes to infinity, this means the L 2 loss in ICP goes to 0.

In other words, ICP loss is minimal if an optimal solution to the UBDI min-max problem is found.

ICP was independently proposed for UBDI in BID16 .

They report their method only works using PCA initialization.

We explored PCA initialization for MUSE, but observed the opposite effect, namely that PCA initialization leads to a degradation in performance.

A function ??? from E to F is a linear transformation if ???(f + g) = ???(f ) + ???(g) and ???(kf ) = k???(f ) for all elements f, g of E, and for all scalars k. An invertible linear transformation is called an isomorphism.

The two vector spaces E and F are called isomorphic, if there is an isomorphism from E to F .

Equivalently, if the kernel of a linear transformation between two vector spaces of the same dimensionality contains only the zero vector, it is invertible and hence an isomorphism.

Most work on supervised or unsupervised alignment of word vector spaces relies on the assumption that they are approximately isomorphic, i.e., isomorphic after removing a small set of vertices BID22 BID3 BID34 BID6 .

In this section, show that word vector spaces are not necessarily approximately isomorphic.

We will refer to cases of non-approximately isomorphic word vector spaces as impossible cases.

The possible cases can be further divided into easy and hard cases; corresponding to the cases where GAN-based UBDI is stable and unstable (i.e., performance is highly dependent on initialization), respectively.

It is not difficult to see why hard cases may arise when using GANs for unsupervised alignment of vector spaces.

One example of a hard (but not impossible) problem instance is the case of two smoothly populated vector spaces on unit spheres.

In this case, there is an infinite set of equally good linear transformations (rotations) that achieve the same training loss.

Similarly, for two binary-valued, n-dimensional vector spaces with one vector in each possible position.

Here the number of local optima would be 2 n , but since the loss is the same in each of them the loss landscape is highly non-convex, and the basin of convergence is therefore very small BID33 .

The chance of aligning the two spaces using gradient descent optimization would be 1 2 n .

In other words, minimizing the Jensen-Shannon divergence between the word vector distributions, even in the easy case, is not always guaranteed to uncover an alignment between translation equivalents.

From the above, it follows that alignments between linearly alignable vector spaces cannot always be learned using UBDI methods.

In ??3.1 , we test for approximate isomorphism to decide whether two vector spaces are linearly alignable.

??3.2-3.3 are devoted to analyzing when alignments between linearly alignable vector spaces can be learned.

In our experiments in ??3 and 4, Bengali and Cebuano embeddings are pretrained by FastText; 1 all others are trained using FastText on Polyglot.2 In the experiments in ??5, we use FastText embeddings pretrained on Wiki and Common Crawl data.

3 If not indicated otherwise, we use MUSE with default parameters BID6 .

Procrustes fit ) is a simple linearity test, which, as we find, captures the dynamics of GAN-based UBDI well.

Compared to isomorphism and isospectrality tests, Procrustes fit is inexpensive and can be run with bigger dictionary seeds.

Procrustes fit The idea behind this test is to apply a Procrustes analysis (see ??2) on a sizeable dictionary seed (5000 tokens), to measure the training fit.

Since U V T E = F if and only if E and F are isomorphic, the Procrustes fit tests the linear alignability between two embedding spaces exists.

We can correlate the Procrustes fit measure with the performance of UBDI.

While UBDI is motivated by cases where dictionary seeds are not available, and Procrustes fit relies on dictionary seeds, a strong correlation can act as a sanity check on UBDI, as well as a tool to help us understand its limitations.

The relationship between Procrustes fit and UBDI performance is presented in FIG0 and shows a very strong correlation.

One immediate conclusion is that the poor UBDI performance on languages such as Bengali and Cebuano is not a result of GANs being a poor estimator of the linear transforms, but rather a result of there not being a good linear transform from English into these languages.

Isomorphism and isospectrality We briefly compare Procrustes fit to two similarity measures for nearest neighbor graphs of vector spaces, introduced in graph of a word vector space is obtained by adding edges between any word vertex and its nearest neighbor.

Note that only cycles of length 2 are possible in a nearest neighbor graph.

Two nearest neighbor graphs are graph isomorphic if they contain the same number of vertices connected in the same way.

Two isomorphic vector spaces have isomorphic nearest neighbor graphs, but not vice versa.

We say that the nearest neighbor graphs are k-subgraph isomorphic if the nearest neighbor graphs for the most frequent k words (in the source language and their translations) are isomorphic.

There are exact algorithms, e.g., VF2 BID7 , for checking whether two nearest neighbor graphs are graph isomorphic.

These algorithms do not scale easily to graphs with hundreds of thousands of nodes, however.

Also, the algorithms do not identify approximate isomorphism, unless run on all subgraphs with k vertices removed.

Such tests are therefore impractical.

instead introduce a spectral metric based on eigenvalues of the Laplacian of the nearest neighbor graphs, similar to metrics used for graph matching problems in computer vision BID25 and biology BID20 .

The metric quantifies to what extent the nearest neighbor graphs are isospectral.

Note that (approximately) isospectral graphs need not be (approximately) isomorphic, but (approximately) isomorphic graphs are always (approximately) isospectral.

Let A 1 and A 2 be the adjacency matrices of the nearest neighbor graphs G 1 and G 2 of our two word embeddings, respectively.

Let L 1 = D 1 ??? A 1 and L 2 = D 2 ??? A 2 be the Laplacians of the nearest neighbor graphs, where D 1 and D 2 are the corresponding diagonal matrices of degrees.

We then compute the eigensimilarity of the Laplacians of the nearest neighbor graphs, L 1 and L 2 .

For each graph, we find the smallest k such that the sum of the k largest Laplacian eigenvalues is <90% of the Laplacian eigenvalues.

We take the smallest k of the two, and use the sum of the squared differences between the largest k Laplacian eigenvalues DISPLAYFORM0 .

Note that ??? = 0 means the graphs are isospectral, and the metric goes to infinite.

Thus, the higher ??? is, the less similar the graphs (i.e., their Laplacian spectra).

Isospectrality varies with Procrustes fit; to see this, we show that DISPLAYFORM1 F , in this case ??? = I. Two isomorphic graphs also have the same set of sorted eigenvalues, i.e., DISPLAYFORM2 In general, it holds that if we add an edge to a graph G, to form G , its spectrum changes monotonically BID29 .

Since the Procrustes fit evaluates the nearest neighbor graph, it follows that a change in the nearest neighbor graph leading to a drop in Procrustes fit will also lead to a drop in eigenvector similarity.

However, isomorphism and isospectrality tests are computationally expensive, and in practice, we have to sample subgraphs and run the tests on multiple subgraphs, which leads to a poor approximation of the similarities of the two embedding graphs.

In practice, Procrustes fit, k-subgraph isomorphism, and k-subgraph isospectrality thus all rely on a dictionary.

The tests are therefore not diagnostic tools, but means to understand the dynamics of UBDI.

Procrustes fit is more discriminative (since vector space isomorphism entails nearest neighbor graph isomorphism, not vice versa) and computationally more efficient.

In our experiments, it also correlates much better with UBDI performance (MUSE in Table 2 ; the correlation coefficient is 96%, compared to 0% for k-subgraph isomorphism (not listed in Table 2 ), and -27% for k-subgraph isospectrality with k = 10.Observation 1 Impossible cases are not (solely) the result of linguistic differences, but also of corpus characteristics.

English-Bengali and English-Cebuano are not linearly alignable according to our Procrustes fit tests.

There can be two explanations for such an observation: linguistic differences between the two languages or variance in the monolingual corpora for Bengali and Cebuano, i.e. noise and little support per word.

We test for this by applying the Procrustes fit test to the word vector spaces of Bengali and a higher resource related language, Hindi.

The Procrustes fit for Bengali-Hindi is even lower than for compared to 46.25, respectively) .

This finding is surprising as we would expect Bengali and Hindi to align well due to their relatedness.

The result thus suggests that the Bengali embeddings are of insufficient quality, which can largely explain the poor alignment found by the GAN.

This is further supported by follow-up experiments we ran aligning a word vector space for English and a word vector space induced from scrambled English sentences (learned on two different 10% samples of Wikipedia), which can be thought of as a sample from a synthetic language that completely diverges from English in its syntactic properties.5 GAN-based UBDI was able to near-perfectly recover the word identities without supervision, showing that its success is not easily impeded by linguistic differences.

Observation 2 Impossible cases can also be the result of the inductive biases of the underlying word embedding algorithms.

One observation made in BID6 is that the performance of MUSE degrades a little when using alternative embedding algorithms, but that alignment is still possible.

We, however, observe that this is not the case if using different, monolingual embedding algorithms, i.e., if using FastText for English and Hyperwords for Spanish.

While such embeddings are still linearly alignable (as verified by computing their Procrustes fits), GAN-based UBDI consistently fails on such cases.

This also holds for the case of aligning FastText for English and Hyperwords for English, as observed in BID14 .

In order to better understand the dynamics of GAN-based UBDI in hard cases, i.e., when the GAN suffers from local minima, we introduce three ablation transformations, designed to control for properties of the word vector spaces: unit length normalization, PCA-based pruning, and noising.

The results of GAN-based UBDI after applying these transforms are reported in Table 2 .Observation 3 GAN-based UBDI becomes more unstable and performance deteriorates with unit length normalization.

This ablation transform performs unit length normalization (ULN) of all vectors x, i.e., x = x ||x|| 2 , and is often used in supervised bilingual dictionary induction BID32 BID1 .

We use this transform to project word vectors onto a sphere -to control for shape information.

If vectors are distributed smoothly over two spheres, there is no way to learn an alignment in the absence of dictionary seed; in other words, if UBDI is unaffected by this transform, UBDI learns from density information alone.

While supervised methods are insensitive to or benefit from ULN, we find that UBDI is very sensitive to such normalization (see Table 2 , M-unit).

We verify that supervised alignment is not affected by ULN by checking the Procrustes fit ( ??3.1), which remains unchanged under this transformation.

Observation 4 GAN-based UBDI becomes more unstable and performance deteriorates with PCA pruning.

In order to control for density, we apply PCA to our word vector spaces, reducing them to 25 dimensions, and prune our vocabularies to remove density clusters by keeping all but one of the nearest neighbors vectors on an integer grid.

This removes about 10% of our vocabularies.

We then apply UBDI to the original vectors for the remaining words.

This smoothening of the embeddings results in highly unstable and reduced performance (see Table 2 , M-PCA).

In other words, density information, while less crucial than shape information, is important for the stability of UBDI, possibly by reducing the chance of getting stuck in local optima.

This is in contrast with the results on using ICP for UBDI in BID16 Table 2 : Main experiments; average performance and stability across 10 runs.

We consider a P@1 score below 1% a fail.

MUSE is the MUSE system with default parameters.

Ablation transforms: M-unit uses unit length normalization to evaluate the impact of shape; M-PCA uses PCA-based pruning to evaluate the impact of density; M-noise uses 25% random vectors injected in the target language space to evaluate the impact of noise.

M-discr uses discriminator loss for model selection, as a baseline for M-cosine; M-cosine uses our model selection criterion.

The macro-averaged error reduction of M-cosine over MUSE for the HARD languages is 7%; and 4% across all language pairs.

using PCA initialization with 50 dimensions.

We ran experiments with 25, 50, and 100 dimensions, with or without pruning, observing significant drops in performance across the board.

Observation 5 GAN-based UBDI is largely unaffected by noise injection.

We add 25% random vectors, randomly sampled from a hypercube bounding the vector set.

GAN-based UBDI results are not consistently affected by noise injection (see Table 2 , M-noise).

This is because the injected vectors rarely end up in the seed dictionaries used for the Procrustes analysis step.

In follow-up experiments on Greek and Hungarian, we find that GAN-based UBDI gets stuck in local optima in hard cases, and over-parameterization, increasing batch size or decreasing learning rate does not help.

Observation 6 In the hard cases, GAN-based UBDI gets stuck in local optima.

In cases where linear alignment is possible, but UBDI is unstable, the model might get stuck in a local optimum, which is the result of the discriminator loss not increasing in the region around the current discriminator model.

We analyze the discriminator loss in these areas by plotting it as a function of the generator parameters for the failure cases of two of the hard alignment cases, namely English-Greek and English-Hungarian.

We plot the loss surface along its intersection with a line segment connecting two sets of parameters BID13 BID21 .

In our case, we interpolate between the model induced by GAN-based UBDI and the (oracle) model obtained using supervised Procrustes analysis.

Result are shown in FIG2 .

The green loss curves represent the current discriminator's loss along all the generators between the current generator and the generator found by Procrustes analysis.

In all cases, we see that while performance (P@1 and mean cosine similarity) goes up, there is an initial drop in the discriminator loss, which suggests there is no learning signal in this direction for GAN-based UBDI.

This is along a line segment representing the shortest path from the failed generator to the oracle generator, of course; linear interpolation provides no guarantee there are no almost-as-short paths with plenty of signal.

A more sophisticated sampling method is to sample along two random direction vectors BID13 ; BID21 .

We used an alternative strategy of sampling from normal distributions with fixed variance that were orthogonal to the line segment.

We observed the same pattern, leading us to the conclusion that instability is caused by local optima.

Observation 7 Over-parameterization does not consistently help in the hard cases.

Recent work has observed that over-parameterization leads to smoother loss landscapes and makes optimization easier BID5 .

We experiment with widening our discriminators to smoothen the loss landscape, but results are inconsistent: for Hungarian, this made GAN-based UBDI more stable; for Greek, less stable (see FIG2 ).Observation 8 Changing the batch size or the learning rate to hurt the discriminator also does not help.

Previous work has shown that large learning rate and small batch size contribute towards SGD finding flatter minima BID17 , but in our experiments, we are interested in the discriminator not ending up in flat regions, where there is no signal to update the generator.

We therefore experiment with smaller learning rate and larger batch sizes.

The motivation behind both is decreasing the scale of random fluctuations in the SGD dynamics BID27 BID2 , enabling the discriminator to explore narrower regions in the loss landscape.

See FIG2 for results.

Increasing the batch size or varying the learning rate (up or down) clearly comes at a cost, and it seems the MUSE default hyperparameters are close to optimal.

In this section, we compare two unsupervised model selection criteria.

We train three models with different random seeds in parallel and use the selection criterion to select one of these models to train for the remaining epochs.

The first criterion is the discriminator loss during training, which is used in BID8 , for example.

In contrast, we propose to use the mean cosine similarity between all translations predicted by the CSLS method (see ??2), which was used as an unsupervised stopping criterion by BID6 .Observation 9 In the hard cases, model selection with cosine similarity can stabilize GAN-based UBDI.

As we see in Table 2 , the selection criterion based on discriminator loss (M-discr) increases the instability of UBDI, leading to 4/10 failed alignments for Greek compared to 2/10 without model selection, for example.

Cosine similarity (M-cosine) in contrast leads to perfectly stable UBDI.

Note that if the probability of getting stuck in a local optimum that leads to a poor alignment is ??, using n random restarts and oracle model selection we increase the probability of finding a good alignment to 1 ??? (1 ??? ??) n .

In our experiments, n = 3.

Some pairs of word vector spaces are not alignable based on distributional information alone.

For other pairs, GANs can be used to induce such an alignment, but the degree of instability is very susceptible to the shape and density of the word vector spaces, albeit not to noise.

Instability is caused by local optima, but not remedied by standard techniques such as over-parameterization, increasing the batch size or decreasing the learning rate.

We propose an unsupervised model selection criterion that enables stable learning, leading to a~7% error reduction over MUSE, and present further observations about the alignability of word vector distributions.

<|TLDR|>

@highlight

An empirical investigation of GAN-based alignment of word vector spaces, focusing on cases, where linear transformations provably exist, but training is unstable.