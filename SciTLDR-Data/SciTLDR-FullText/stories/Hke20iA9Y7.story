We study the problem of learning similarity functions over very large corpora using neural network embedding models.

These models are typically trained using SGD with random sampling of unobserved pairs, with a sample size that grows quadratically with the corpus size, making it expensive to scale.

We propose new efficient methods to train these models without having to sample unobserved pairs.

Inspired by matrix factorization, our approach relies on adding a global quadratic penalty and expressing this term as the inner-product of two generalized Gramians.

We show that the gradient of this term can be efficiently computed by maintaining estimates of the Gramians, and develop variance reduction schemes to improve the quality of the estimates.

We conduct large-scale experiments that show a significant improvement both in training time and generalization performance compared to sampling methods.

We consider the problem of learning a similarity function h : X × Y → R, that maps each pair of items, represented by their feature vectors (x, y) ∈ X × Y, to a real number h(x, y), representing their similarity.

We will refer to x and y as the left and right feature vectors, respectively.

Many problems can be cast in this form: In natural language processing, x represents a context (e.g. a bag of words), y represents a candidate word, and the target similarity measures the likelihood to observe y in context x BID14 BID16 BID13 .

In recommender systems, x represents a user query, y represents a candidate item, and the target similarity is a measure of relevance of item y to query x, e.g. a movie rating BID0 , or the likelihood to watch a given movie BID12 Rendle, 2010) .

Other applications include image similarity, where x and y are pixel-representations of images BID5 BID6 Schroff et al., 2015) , and network embedding models BID10 Qiu et al., 2018) , where x and y are nodes in a graph and the similarity is whether an edge connects them.

A popular approach to learning similarity functions is to train an embedding representation of each item, such that items with high similarity are mapped to vectors that are close in the embedding space.

A common property of such problems is that only a small subset of all possible pairs X × Y is present in the training set, and those examples typically have high similarity.

Training exclusively on observed examples has been demonstrated to yield poor generalization performance.

Intuitively, when trained only on observed pairs, the model places the embedding of a given item close to similar items, but does not learn to place it far from dissimilar ones (Shazeer et al., 2016; Xin et al., 2017) .

Taking into account unobserved pairs is known to improve the embedding quality in many applications, including recommendation BID12 BID1 and word analogy tasks (Shazeer et al., 2016) .

This is often achieved by adding a low-similarity prior on all pairs, which acts as a repulsive force between all embeddings.

But because it involves a number of terms quadratic in the corpus size, this term is computationally intractable (except in the linear case), and it is typically optimized using sampling: for each observed pair in the training set, a set of random unobserved pairs is sampled and used to compute an estimate of the repulsive term.

But as the corpus size increases, the quality of the estimates deteriorates unless the sample size is increased, which limits scalability.

In this paper, we address this issue by developing new methods to efficiently estimate the repulsive term, without sampling unobserved pairs.

Our approach is inspired by matrix factorization models, which correspond to the special case of linear embedding functions.

They are typically trained using alternating least squares BID12 , or coordinate descent methods BID2 , which circumvent the computational burden of the repulsive term by writing it as a matrix-inner-product of two Gramians, and computing the left Gramian before optimizing over the right embeddings, and viceversa.

Unfortunately, in non-linear embedding models, each update of the model parameters induces a simulateneous change in all embeddings, making it impractical to recompute the Gramians at each iteration.

As a result, the Gramian formulation has been largely ignored in the non-linear setting, where models are instead trained using stochastic gradient methods with sampling of unobserved pairs, see BID7 .

Vincent et al. (2015) were, to our knowledge, the first to attempt leveraging the Gramian formulation in the non-linear case.

They consider a model where only one of the embedding functions is non-linear, and show that the gradient can be computed efficiently in that case.

Their result is remarkable in that it allows exact gradient computation, but this unfortunately does not generalize to the case where both embedding functions are non-linear.

Contributions We propose new methods that leverage the Gramian formulation in the non-linear case, and that, unlike previous approaches, are efficient even when both left and right embeddings are non-linear.

Our methods operate by maintaining stochastic estimates of the Gram matrices, and using different variance reduction schemes to improve the quality of the estimates.

We perform several experiments that show these methods scale far better than traditional sampling approaches on very large corpora.

We start by reviewing preliminaries in Section 2, then derive the Gramian-based methods and analyze them in Section 3.

We conduct large-scale experiments on the Wikipedia dataset in Section 4, and provide additional experiments in the appendix.

All the proofs are deferred to Appendix A.

We consider models that consist of two embedding functions u : R d ×X → R k and v : R d ×Y → R k , which map a parameter vector 1 θ ∈ R d and feature vectors x, y to embeddings u(θ, x), v(θ, y) ∈ R k .

The output of the model is the dot product 2 of the embeddings h θ (x, y) = u(θ, x), v(θ, y) , where ·, · denotes the usual inner-product on R k .

Low-rank matrix factorization is a special case, in which the left and right embedding functions are linear in x and y. Figure 1 illustrates a non-linear model, in which each embedding function is given by a feed-forward neural network.

3 We denote the training set by T = {(x i , y i , s i ) ∈ X × Y × R} i∈{1,...,n} , where x i , y i are the feature vectors and s i is the target similarity for example i. To make notation more compact, we will use u i (θ), v i (θ) as a shorthand for u(θ, x i ), v(θ, y i ), respectively.

As discussed in the introduction, we also assume that we are given a low-similarity prior p ij ∈ R for all pairs (i, j) ∈ {1, . . .

, n} 2 .

Given a differentiable scalar loss function : R × R → R, the objective function is given by DISPLAYFORM0 where the first term measures the loss on observed data, the second term penalizes deviations from the prior, and λ is a positive hyper-parameter that trades-off the two terms.

To simplify the discussion, we will assume a uniform zero prior p ij as in BID12 , the general case is treated in Appendix B.To optimize this objective, existing methods rely on sampling to approximate the second term, and are usually referred to as negative sampling or candidate sampling, see BID7 BID1 for a survey.

Due to the double sum in (1), the quality of the sampling estimates degrades as the corpus size increases, which can significantly increase training times.

This can be alleviated by increasing the sample size, but does not scale to very large corpora.

DISPLAYFORM1 Figure 1: A dot-product embedding model for a similarity function on X × Y.

A different approach to solving (1), widely popular in matrix factorization, is to rewrite the double sum as the inner product of two Gram matrices.

Let us denote by U θ ∈ R n×k the matrix of all left embeddings such that u i (θ) is the i-th row of U θ , and similarly for V θ ∈ R n×k .

Then denoting the matrix inner-product by A, B = i,j A ij B ij , we can rewrite the double sum in (1) as: DISPLAYFORM0 Now, using the adjoint property of the inner product, we have DISPLAYFORM1 and if we denote by u ⊗ u the outer product of a vector u by itself, and define the Gram matrices DISPLAYFORM2 the penalty term becomes DISPLAYFORM3 The Gramians are k × k PSD matrices, where k, the dimension of the embedding space, is much smaller than n -typically k is smaller than 1000, while n can be arbitrarily large.

Thus, the Gramian formulation (4) has a much lower computational complexity than the double sum formulation (2), and this reformulation is at the core of alternating least squares and coordinate descent methods BID12 BID2 , which operate by computing the exact Gramian for one side, and solving for the embeddings on the other.

However, these methods do not apply in the non-linear setting due to the implicit dependence on θ, as a change in the model parameters simultaneously changes all embeddings on both sides, making it intractable to recompute the Gramians at each iteration, so the Gramian formulation has not been used when training non-linear models.

In the next section, we show that it can in fact be leveraged in the non-linear case.

In order to leverage the Gramian formulation in the non-linear case, we start by rewriting the objective function (1) in terms of the Gramians defined in (3).

Let DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 .

Intuitively, for each example i, −∇f i (θ) pulls the embeddings u i and v i close to each other (assuming a high similarity s i ), while −∇g i (θ) creates a repulsive force between u i and all embeddings {v j } j∈{1,...,n} , and between v i and all {u j } j∈{1,...,n} , see Appendix D for further discussion, and illustration of the effect of this term.

While the Gramians are expensive to recompute at each iteration, we can maintain PSD estimateŝ DISPLAYFORM3 .

Then the gradient of g(θ) can be approximated by the gradient (w.r.t.

θ) of DISPLAYFORM4 as stated in the following proposition.

DISPLAYFORM5 is an unbiased estimate of ∇g(θ).In a mini-batch setting, one can further averageĝ i over a batch of examples i ∈ B (which we do in our experiments), but we will omit batches to keep the notation concise.

Next, we propose several methods for computing Gramian estimatesĜ u ,Ĝ v , and discuss their tradeoffs.

Since each Gramian can be written as a sum of rank-one terms, e.g. DISPLAYFORM6 , a simple unbiased estimate can be obtained by sampling one term (or a batch) from this sum.

We improve on this by using different variance reduction methods, which we discuss in the next two sections.

Our first method is inspired by the stochastic average gradient (SAG) method (Roux et al., 2012; BID9 Schmidt et al., 2017) , which reduces the variance of the gradient estimates by maintaining a cache of individual gradients, and estimating the full gradient using this cache.

Since each Gramian is a sum of outer-products (see equation FORMULA4 ), we can apply the same technique to estimate Gramians.

For all i ∈ {1, . . . , n}, letû i ,v i be a cache of the left and right embeddings respectively.

We will denote by a superscript (t) the value of a variable at iteration t. LetŜ DISPLAYFORM0 i , which corresponds to the Gramian computed with the current caches.

At each iteration t, an example i is drawn uniformly at random and the estimate of the Gramian is given bŷ DISPLAYFORM1 and similarly forĜ DISPLAYFORM2 v .

This is summarized in Algorithm 1, where the model parameters are updated using SGD (line 10), but this update can be replaced with any first-order method.

Here β can take one of the following values: β = 1 n , following SAG (Schmidt et al., 2017) , or β = 1, following SAGA BID9 .

The choice of β comes with trade-offs that we briefly discuss below.

We will denote the cone of positive semi-definite k × k matrices by S DISPLAYFORM3 While taking β = 1 gives an unbiased estimate, note that it does not guarantee that the estimates remain in S k + .

In practice, this can cause numerical issues, but can be avoided by projectingĜ u ,Ĝ v on S k + , using their eigenvalue decompositions.

The per-iteration cost of maintaining the Gramian estimates is O(k) to update the caches, O(k 2 ) to update the estimatesŜ u ,Ŝ v ,Ĝ u ,Ĝ v , and O(k 3 ) for projecting on S k + .

Given the small size of the embedding dimension k, O(k 3 ) remains tractable.

The memory cost is O(nk), since each embedding needs to be cached (plus a negligible O(k 2 ) for storing the Gramian estimates).

This makes SAGram much less expensive than applying the original SAG(A) methods, which require maintaining caches of the gradients, this would incur a O(nd) memory cost, where d is the number of parameters of the model, and can be orders of magnitude larger than the embedding dimension k. However, O(nk) can still be prohibitively expensive when n is very large.

In the next section, we propose a different method which does not incur this additional memory cost.

DISPLAYFORM4 Update Gramian estimates (i ∼ Uniform(n)) 8: DISPLAYFORM5 Update model parameters then update caches (i ∼ Uniform(n)) 10: DISPLAYFORM6

To derive the second method, we reformulate problem (1) as a two-player game.

The first player optimizes over the parameters of the model θ, the second player optimizes over the Gramian estimateŝ G u ,Ĝ v ∈ S k + , and they seek to minimize the respective losses DISPLAYFORM0 whereĝ i is defined in FORMULA10 , and · F denotes the Frobenius norm.

To justify this reformulation, we can characterize its first-order stationary points, as follows.

DISPLAYFORM1 + is a first-order stationary point for (9) if and only if θ is a first-order stationary point for problem FORMULA0 DISPLAYFORM2 Several stochastic first-order dynamics can be applied to problem (9), and Algorithm 2 gives a simple instance where each player implements SGD with a constant learning rate.

In this case, the updates of the Gramian estimates (line 7) have a particularly simple form, since ∇Ĝ DISPLAYFORM3 and similarly forĜ v .

One advantage of this form is that each update performs a convex combination between the current estimate and a rank-1 PSD matrix, thus guaranteeing that the estimates remain in S k + , without the need to project.

The per-iteration cost of updating the estimates is O(k 2 ), and the memory cost is O(k 2 ) for storing the Gramians, which are both negligible.

DISPLAYFORM4 Update model parameters (i ∼ Uniform) 8: DISPLAYFORM5 The update (10) can also be interpreted as computing an online estimate of the Gramian by averaging rank-1 terms with decaying weights, thus we call the method Stochastic Online Gramian.

Indeed, we have by induction on t,Ĝ DISPLAYFORM6 Intuitively, averaging reduces the variance of the estimator but introduces a bias, and the choice of the hyper-parameter α trades-off bias and variance.

The next proposition quantifies this tradeoff under mild assumptions.

DISPLAYFORM7 The first assumption simply bounds the variance of single-point estimates, while the second bounds the distance between two consecutive Gramians, a reasonable assumption, since in practice the changes in Gramians vanish as the trajectory θ (τ ) converges.

In the limiting case α = 1,Ĝu reduces to a single-point estimate, in which case the bias (12) vanishes and the variance (11) is maximal, while smaller values of α decrease variance and increase bias.

This is confirmed in our experiments, as discussed in Section 4.2.

We conclude this section by showing that candidate sampling methods (see BID7 BID1 for recent surveys) can be reinterpreted in terms of the Gramian formulation (4).

These methods work by approximating the double-sum in (1) using a random sample of pairs.

Suppose a batch of pairs (i, j) ∈ B × B is sampled 4 , and the double sum is approximated bỹ DISPLAYFORM0 where µ i , ν j are the inverse probabilities of sampling i, j respectively (to guarantee that the estimate is unbiased).

Then applying a similar transformation to Section 2.2, one can show that DISPLAYFORM1 which is equivalent to computing two batch-estimates of the Gramians.

Implementing existing methods using (14) rather than (13) can decrease their computional complexity in the large batch regime, for the following reason: the double-sum formulation (13) involves a sum of |B||B | dot products of vectors in R k , thus computing its gradient costs O(k|B||B |).

On the other hand, the Gramian formulation FORMULA0 is the inner product of two k × k matrices, each involving a sum over the batch, thus computing its gradient costs O(k 2 max(|B|, |B |)), which is cheaper when the batch size is larger than the embedding dimension k, a common situation in practice.

With this formulation, the advantage of SOGram and SAGram becomes clear, as they use more embeddings to estimate Gramians (by caching or online averaging) than would be possible using candidate sampling.

In this section, we conduct large-scale experiments on the Wikipedia dataset (Wikimedia Foundation).

Additional experiments on MovieLens BID11 are given in Appendix F.

Datasets We consider the problem of learning the intra-site links between Wikipedia pages.

Given a pair of pages (x, y) ∈ X × X , the target similarity is 1 if there is a link from x to y, and 0 otherwise.

Here a page is represented by a feature vector x = (x id , x ngrams , x cats ), where x id is (a one-hot encoding of) the page URL, x ngrams is a bag-of-words representation of the set of n-grams of the page's title, and x cats is a bag-of-words representation of the categories the page belongs to.

Note that the left and right feature spaces coincide in this case, but the target similarity is not necessarily symmetric (the links are directed edges).

We carry out experiments on subsets of the Wikipedia graph corresponding to three languages: Simple English, French, and English, denoted respectively by simple, fr, and en.

These subgraphs vary in size, and Table 1 shows some basic statistics for each set.

Each set is partitioned into training and validation using a (90%, 10%) split.

Table 1 : Corpus sizes for each training set.

Models We train non-linear embedding models consisting of a two-tower neural network as in Figure 1 , where the left and right embedding functions map, respectively, the source and destination page features.

The two embedding networks have the same structure: the input feature embeddings are concatenated then mapped through two hidden layers with ReLU activations.

The input embeddings are shared between the two networks, and their dimensions are 50 for simple, 100 for fr, and 120 for en.

Training methods The model is trained using a squared error loss, (s, s ) = 1 2 (s − s ) 2 , optimized using SAGram, SOGram, and as baseline, SGD with candidate sampling, using different sampling strategies.

The experiments reported in this section use a learning rate η = 0.01, a penalty coefficient λ = 10, and batch size 1024.

These parameters correspond to the best performance of the baseline methods; we report additional results with different hyper-parameter settings in Appendix E. For SAGram and SOGram, a batch B is used in the Gramian updates (line 8 in Algorithm 1 and line 6 in Algorithm 2, where we use a sum of rank-1 terms over the batch), and another batch B is used in the model parameter update.

For the sampling baselines, the double sum is approximated by all pairs in the cross product (i, j) ∈ B × B , and for efficiency, we implement them using the Gramian formulation as discussed in Section 3.3, since we operate in a regime where the batch size is an order of magnitude larger than the embedding dimension k. In the first baseline method, uniform, items are sampled uniformly from the vocabulary (all pages are sampled with the same probability).

The other baseline methods implement importance sampling similarly to BID3 ; BID14 : in linear, the probability is proportional to the number of occurrences of the page in the training set, and in sqrt, the probability is proportional to the square root of the number of occurrences.

DISPLAYFORM0 DISPLAYFORM1

In the first set of experiments, we evaluate the quality of the Gramian estimates using each method.

In order to have a meaningful comparison, we fix a trajectory of model parameters (θ (t) ) t∈{1,...,T } , and evaluate how well each method tracks the true Gramians G u (θ (t) ), G v (θ (t) ) on that common trajectory.

This experiment is done on Wikipedia simple (the smallest of the datasets) so that we can compute the exact Gramians by periodically computing the embeddings u i (θ (t) ), v i (θ (t) ) on the full training set at a given time t. We report in FIG3 the estimation error for each method, measured by the normalized Frobenius distance DISPLAYFORM0 .

In FIG3 , we can observe that both variants of SAGram yield the best estimates, and that SOGram yields better estimates than the baselines.

Among the baseline methods, importance sampling (both linear and sqrt) perform better than uniform.

We also vary the batch size to evaluate its impact: increasing the batch size from 128 to 1024 improves the quality of all estimates, as expected, but it is worth noting that the estimates of SOGram with |B| = 128 have comparable quality to baseline estimates with |B| = 1024.In Appendix E, we show that a similar effect can be observed for gradient estimates, and we make a formal connection between Gramian and gradient estimation errors.

In FIG3 , we evaluate the bias-variance tradeoff discussed in Section 3.2, by comparing the estimates of SOGram with different learning rates α.

We observe that higher values of α suffer from higher variance which persists throughout the trajectory.

A lower α reduces the variance but introduces a bias, which is mostly visible during the early iterations.

In order to evaluate the impact of the Gramian estimation quality on training speed and generalization, we compare the validation performance of SOGram to the sampling baselines, on each dataset (we do not use SAGram due to its prohibitive memory cost for corpus sizes of 1M or more).

The models are trained with a fixed time budget of 20 hours for simple, 30 hours for fr and 50 hours for en.

We estimate the mean average precision (MAP) at 10, by scoring, every 5 minutes, left items in the validation set against 50K random candidates (exhaustively scoring all candidates is prohibitively expensive, but this gives a reasonable approximation).

The results are reported in FIG4 .

Compared to the sampling baselines, SOGram exhibits faster training and better validation performance across all sampling strategies.

TAB3 The improvement on simple is modest (between 4% and 10%), which can be explained by the relatively small corpus size (85K unique pages), in which case candidate sampling with a large batch size already yields decent estimates.

On the larger corpora, we obtain more significant improvements: between 9% and 15% on fr and between 9% and 19% on en.

It's interesting to observe that the best performance is consistently achieved by SOGram with linear importance sampling, even though linear performs slightly worse than other strategies in the baseline.

SOGram also has a significant impact on training speed: if we measure the time it takes for SOGram to exceed the final validation performance of each baseline method, this time is a small fraction of the total budget.

In our experiments, this fraction is between 10% and 17% for simple, between 23% and 30% for fr, and between 16% and 24% for en.

Additional numerical results are provided in Appendix E, where we evaluate the impact of other parameters, such as the effect of the batch size |B|, the learning rate η, and the Gramian learning rate α.

For example, we show that the relative improvement of SOGram compared to the baselines is even larger when using smaller batches, and its generalization performance is more robust to the choice of batch size and learning rate.

We showed that the Gramian formulation commonly used in low-rank matrix factorization can be leveraged for training non-linear embedding models, by maintaining estimates of the Gram matrices and using them to estimate the gradient.

By applying variance reduction techniques to the Gramians, one can improve the quality of the gradient estimates, without relying on large sample size as is done in traditional sampling methods.

This leads to a significant impact on training time and generalization quality, as indicated by our experiments.

While we focused on problems with very large vocabulary size, where traditional approaches are inefficient, it will be interesting to evaluate our methods on other applications such as word-analogy tasks BID14 Schnabel et al. (2015) .

Another direction of future work is to extend this formulation to a larger family of penalty functions, such as the spherical loss family studied in (Vincent et al., 2015; BID8

Proof of Proposition 1.

Starting from the expression (6) of g(θ) = G u (θ), G v (θ) , and applying the chain rule, we have DISPLAYFORM0 where J u (θ) denotes the Jacobian of G u (θ), an order-three tensor given by DISPLAYFORM1 and DISPLAYFORM2 Observing DISPLAYFORM3 , and applying the chain rule, we have DISPLAYFORM4 where J u,i (θ) is the Jacobian of u i (θ) ⊗ u i (θ), and DISPLAYFORM5 an similarly for J v,i .

We conclude by taking expectations in (16) and using assumption thatĜ u ,Ĝ v are independent of i.

u , we have, DISPLAYFORM0 which is a sum of matrices in the PSD cone S k + .Proof of Proposition 3.

Denoting by (F t ) t≥0 the filtration generated by the sequence (θ (t) ) t≥0 , and taking conditional expectations in (8), we have DISPLAYFORM1 + is a first-order stationary point of the game if and only if DISPLAYFORM2 The second and third conditions simply states that ∇Ĝ FORMULA0 is equivalent toĜ u = G u (θ) (and similarly, (19) is equivalent toĜ v = G v (θ)).

Using the expression (15) of ∇g, we get that (17-19) is equivalent to ∇f (θ) + λ∇g(θ) = 0.

DISPLAYFORM3 Proof of Proposition 5.

We start by proving the first bound (11).

As stated in Section 3.2, we have, by induction on t,Ĝ DISPLAYFORM4 And by definition ofḠ (t) , we haveḠ DISPLAYFORM5 ) are zero-mean random variables.

Thus, taking the second moment, and using the first assumption (which simply states that the variance of ∆ (τ ) u is bounded by σ 2 ), we have DISPLAYFORM6 which proves the first inequality (11).To prove the second inequality, we start from the definition ofḠ DISPLAYFORM7 u : DISPLAYFORM8 where the first equality uses that fact that DISPLAYFORM9 Focusing on the first term, and bounding G DISPLAYFORM10 u F ≤ (t − τ )δ by the triangle inequality, we get DISPLAYFORM11 Combining FORMULA2 and FORMULA0 , we get the desired inequality (12).

So far, we have assumed a uniform zero prior to simplify the notation.

In this section, we relax this assumption.

Suppose that the prior is given by a low-rank matrix P = QR , where Q, R ∈ R n×k P .

In other words, the prior for a given pair (i, j) is given by the dot product of two vectors p ij = q i , r j .

In practice, such a low-rank prior can be obtained, for example, by first training a simple low-rank matrix approximation of the target similarity matrix.

Given this low-rank prior, the penalty term (2) becomes DISPLAYFORM0 where c = Q Q, R R is a constant that does not depend on θ.

Here, we used a superscript P in g P to disambiguate the zero-prior case.

Now, if we define weighted embedding matrices DISPLAYFORM1 Finally, if we maintain estimatesĤ u ,Ĥ v of H u (θ), H v (θ), respectively (using the methods proposed in Section 3), we can approximate ∇g P (θ) by the gradient of DISPLAYFORM2 Proposition 1 and Algorithms 1 and 2 can be generalized to the low-rank prior case by adding updates forĤ u ,Ĥ v , and by using expression (22) ofĝ P i when computing the gradient estimate.

DISPLAYFORM3 Proof.

Similar to the proof of Proposition 1.The generalized versions of SAGram and SOGram are stated below, where the differences compared to the zero prior case are highlighted.

Note that, unlike the Gramian matrices, the weighted embedding matrices H u , H v are not symmetric, thus we do not project their estimates.

Algorithm 3 SAGram (Stochastic Average Gramian) with low-rank prior 1: Input: Training data {(x i , y i , s i )} i∈{1,...,n} , low-rank priors {q i , r i } i∈{1,...,n} 2: Initialization phase DISPLAYFORM4 Update weighted embedding estimates 11: DISPLAYFORM5 Update model parameters then update caches (i ∼ Uniform(n)) 14: DISPLAYFORM6 Algorithm 4 SOGram (Stochastic Online Gramian) with low-rank prior 1: Input: Training data {(x i , y i , s i )} i∈{1,...,n} , low-rank priors {q i , r i } i∈{1,...,n} 2: Initialization phase DISPLAYFORM7 Update weighted embedding estimates 9: DISPLAYFORM8 Update model parameters (i ∼ Uniform(n)) 11: DISPLAYFORM9 In additional to using a non-uniform prior, it can also be desirable to use non-uniform weights in the penalty term, for example to balance the contribution of frequent and infrequent items to the penalty term.

We discuss how to adapt our algorithms to the non-uniform weights case.

Suppose that the penalty function is given by DISPLAYFORM10 where a i , b j are positive left and right weights, respectively.

Here we used a superscript W in g W to disambiguate the uniform-weight case.

Then using a similar transformation to Section 2.2, we can rewrite g W as follows: DISPLAYFORM11 i.e. g W is the inner-product of two weighted Gramians.

Both SAGram and SOGram can be generalized to this case, by maintaining estimates of the weighted Gramians, one simply needs to scale the contribution of each term u i ⊗ u i by the appropriate embedding weight a i (and similarly for the right embedding).Remark Here we discussed the case of a rank-one weight matrix, i.e. when the unobserved weight matrix can be written as W = a ⊗ b for a given left and right weight vectors a, b. The weight matrix cannot be arbitrary (as specifying n 2 individual weights is prohibitively expensive in many applications such as the experiments of this paper), thus one needs a consice description of the weights matrix.

One such description is the sum of a sparse and low-rank matrix, and one can generalize SAGram and SOGram to this case: the sparse part of the weight matrix can be optimized explicitly, and the low-rank part can be optimized using weighted Gramians, by generalizing the argument of the previous paragraph.

In this section, we briefly discuss different interpretations of the Gramian inner-product g(θ).

Starting from the expression (4) of g(θ) and the definition (3) of the Gram matrices, we have DISPLAYFORM0 which is a quadratic form in the left embeddings u i (and similarly for v j , by symmetry).

In particular, the partial derivative of the Gramian term with respect to an embedding u i is DISPLAYFORM1 .

Thus the gradient of g(θ) with respect to u i is an average of scaled projections of u i on each of the right embeddings v j , and moving in the direction of the negative gradient simply moves u i away from regions of the embedding space with a high density of right embeddings.

This corresponds to the intuition discussed in the introduction: the purpose of the g(θ) term is precisely to push left and right embeddings away from each other, to avoid placing embeddings of dissimilar items near each other, a phenomenon referred to as folding of the embedding space (Xin et al., 2017) .

In order to illustrate the effect of this term on the embedding distributions, we visualize, in FIG6 , the distribution of the inner product u i (θ (t) ), v j (θ (t) ) , for random pairs (i, j), and for observed pairs (i = j), and how these distributions change as t increases.

The plots are generated for the Wikipedia en model described in Section 4, trained with SOGram (α = 0.01), with two different values of the penalty coefficient, λ = 10 −2 and λ = 10.

In both cases, the distribution for observed pairs remains concentrated around values close to 1, as one expects (recall that the target similarity is 1 for observed pairs, i.e. pairs of connected pages in the Wikipedia graph).

The distributions for random pairs, however, are very different: with λ = 10, the distribution quickly concentrates around a value close to 0, while with λ = 10 −2 the distribution is more flat, and a large proportion of pairs have a high inner-product.

This indicates that with a lower λ, the model is more likely to fold, i.e. place embeddings of unrelated items near each other.

This is consistent with the validation MAP, reported in FIG7 .

With λ = 10 −2 , the validation MAP increases very slowly, and remains two orders of magnitude smaller than the model trained with λ = 10.

The figure also shows that when λ is too large (λ = 103 ), the model is over-regularized and the MAP decreases.

To conclude this section, we note that our methods also apply to a related regularizer introduced in BID1 , called Global Orthogonal Regularization.

The authors argue that when learning feature embedding representations, spreading out the embeddings is helpful for generalization, and propose to match the second moment of each embedding distribution with that of the uniform distribution.

Formally, and using our notation, they use the penalty term max(g u (θ), 1/k)+max(g v (θ), 1/k), where k is the embedding dimension, g u (θ) = 1 n 2 n i=1 n j=1 u i , u j 2 , and similarly for g v .

They optimize this term using candidate sampling.

We can also apply the same Gramian transformation as in Section 2.2 to write g DISPLAYFORM2 , and we can similarly apply SAGram and SOGram to estimate both Gramians.

Formally, the difference here is that one would penalize the inner-product of each Gramian with itself, instead of the inner-product of the two.

One advantage of this regularizer is that it applies to a broader class of models, as it does not require the output of the model to be the dot-product of two embedding functions.

The experiments in Section 4 indicate that our methods give better estimates of the Gramians, and a natural question is how this affects gradient estimation quality.

First, one can make a formal connection between the two.

Since DISPLAYFORM0 the estimation error of the gradient with respect to the left embeddings u is DISPLAYFORM1 This last expression can be interpreted as a Frobenius norm of the right Gramian estimation error G v −Ĝ v , weighted by the left Gramian G u , thus the gradient error is closely related to the Gramian error.

FIG8 shows the gradient estimation quality on Wikipedia simple, measured by the normalized squared norm DISPLAYFORM2 .

The results are similar to the Gramian estimation errors reported in FIG3 .

Comparing the different baselines methods on simple FIG4 , we observe that uniform sampling performs better than sqrt, despite having a worse Gramian estimate according to FIG3 .

One possible explanation is that the sampling distribution affects both the quality of the Gramian estimates, and the frequency at which the item embeddings are updated, which in turn affects the MAP.

In particular, tail items are updated more frequently under uniform than other distributions, and this may have a positive impact on the MAP.

In addition to the experiments of Section 4, we also evaluated the effect of the Gramian learning rate α on the quality of the Gramian esimates and generalization performance on Wikipedia en.

FIG9 shows the validation MAP of the SOGram method for different values of α (together with the basline for reference).

This reflects the bias-variance tradeoff dicussed in Proposition 5: with a lower α, progress is initially slower (due to the bias introduced in the Gramian estimates), but the final performance is better.

Given a limited training time budget, this suggests that a higher α can be preferable.

We also evaluate the quality of the Gramian estimates, but due to the large vocabulary size in en, computing the exact Gramians is no longer feasible, so we approximate it using a large sample of 1M embeddings.

The results are reported in FIG10 , which shows the normalized Frobenius distance between the Gramian estimatesĜ u and (the large sample approximation of) the true Gramian G u .

The results are similar to the experiment on simple: with a lower α, the estimation error is initially high, but decays to a lower value as training progresses, which can be explained by the bias-variance tradeoff discussed in Proposition 5.

The tradeoff is affected by the trajectory of the true Gramians: smaller changes in the Gramians (captured by the parameter δ in Proposition 5) induce a smaller bias.

In particular, changing the learning rate η of the main algorithm can affect the performance of the Gramian estimates by affecting the rate of change of the true Gramians.

To investiage this effect, we ran the same experiment with two different learning rates, η = 0.01 as in Section 4, and a lower learning rate η = 0.002.

The errors converge to similar values in both cases, but the error decay occurs much faster with smaller η, which is consistent with our analysis.

In this section, we explore the effect of the batch size |B| and learning rate η on the performance of SOGram compared to the baselines.

We ran the Wikipedia en experiment with different values of these hyperparameters, and report the final validation MAP in TAB6 , which correspond to batch size 128 and 512 respectively.

We can make several observations.

First, the best performance is consistently achieved by SOGram with learning rate α = 0.001.

Second, the relative improvement compared to the baseline is, in general, larger for smaller batch sizes.

This can be explained intuitively by the fact that because of online averaging, the quality of the Gramian estimates with SOGram suffers less than with the sampling baseline.

Finally, we can also observe that the final performance also seems more robust to the choice of batch size and learning rate, compared to the baseline.

For example, with the larger learning rate η = 0.02, the performance degrades for all methods, but the drop in performance for the baseline is much more significant than for the SOGram methods.

In this section, we report experiments on a regression task on MovieLens.

Dataset The MovieLens dataset consists of movie ratings given by a set of users.

In our notation, the left features x represent a user, the right features y represent an item, and the target similarity is the rating of movie y by user x. The data is partitioned into a training and a validation set using a (80%-20%) split.

Table 5 gives a basic description of the data size.

Note that it is comparable to the simple dataset in the Wikipedia experiments.

Dataset # users # movies # ratings MovieLens 72K 10K 10M Table 5 : Corpus size of the MovieLens dataset.

Model We train a two-tower neural network model, as described in Figure 1 , where each tower consists of an input layer, a hidden layer, and output embedding dimension k = 35.

The left tower takes as input a one-hot encoding of a unique user id, and the right tower takes as input one-hot encodings of a unique movie id, the release year of the movie, and a bag-of-words representation of the genres of the movie.

These input embeddings are concatenated and used as input to the right tower.

Methods The model is trained using a squared loss (s, s ) = 1 2 (s − s ) 2 , using SOGram with different values of α, and sampling as a baseline.

We use a learning rate η = 0.05, and penalty coefficient λ = 1.

We measure mean average precision on the trainig set and validation set, following the same procedure described in Section 4.

The results are given in FIG12 .Results The results are similar to those reported on the Wikipedia simple dataset, which is comparable in corpus size and number of observations to MovieLens.

The best validation mean average precision is achieved by SOGram with α = 0.1 (for an improvement of 2.9% compared to the sampling baseline), despite its poor performance on the training set, which indicates that better estimation of g(θ) induces better regularization.

The impact on training speed is also remarkable in this case, SOGram with α = 0.1 achieves a better validation performance in under 1 hour of training than the sampling baseline in 6 hours.

@highlight

We develop efficient methods to train neural embedding models with a dot-product structure, by reformulating the objective function in terms of generalized Gram matrices, and maintaining estimates of those matrices.