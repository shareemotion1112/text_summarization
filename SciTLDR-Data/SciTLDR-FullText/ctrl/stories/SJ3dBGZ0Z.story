Log-linear models models are widely used in machine learning, and in particular are ubiquitous in deep learning architectures in the form of the softmax.

While exact inference and learning of these requires linear time, it can be done approximately in sub-linear time with strong concentrations guarantees.

In this work, we present LSH Softmax, a method to perform sub-linear learning and inference of the softmax layer in the deep learning setting.

Our method relies on the popular Locality-Sensitive Hashing to build a well-concentrated gradient estimator, using nearest neighbors and uniform samples.

We also present an inference scheme in sub-linear time for LSH Softmax using the Gumbel distribution.

On language modeling, we show that Recurrent Neural Networks trained with LSH Softmax perform on-par with computing the exact softmax while requiring sub-linear computations.

Deep neural networks have achieved impressive successes in tasks spanning vision BID9 BID16 , language BID3 , speech BID6 BID27 and videos BID1 .

While these models can vastly differ in architecture, activation functions, and presence of recurrence, they (almost) all share a common trait: the softmax layer.

The softmax layer, or log-linear model, is a widely used model in machine learning and statistics that transforms a feature vector into a distribution over the output space, modeling log-probabilities as a linear function of the feature vector.

For example, in object classification, the softmax layer at the end of a deep convolutional network transforms a feature vector into a probability distribution over classes for the image; in language modeling using recurrent neural networks, it maps the hidden state to a distribution over next words.

While parameterizing for logits offers modeling flexibility, inference and learning have linear runtime in the number of classes.

Indeed, both of these require computing the un-normalized probability for every class to compute the partition function and retrieve an actual probability distribution.

Problems with large output spaces arise naturally in many areas like natural language processing (NLP), where the output space is a language's vocabulary and can be on the order of hundreds of thousands of elements BID15 ; BID12 .

This can also occur in computer vision BID14 when attempting tag prediction on massive, weakly-labeled datasets such as Flickr100M BID31 .Many solutions have been proposed to address this bottleneck, all revolving around two themes: approximation of the softmax probabilities or computation of exact probabilities for an approximate model.

Canonical examples of the former are importance sampling (IS) or noise contrastive estimation (NCE; BID8 ).

Instead of computing probabilities over the whole output space, these methods compute the softmax over a smaller, sampled vocabulary and re-weight the probabilities, providing an unbiased estimator.

An illustration of the latter is Hierarchical Softmax BID24 , where the output classes are first clustered such that you only need to compute the softmax over a smaller output space.

While the former is an unbiased estimate, it comes with no concentration guarantees, and it is often more art than science to craft proposal distributions which will provide low-variance estimators.

The latter, while efficient, requires carefully hand-crafted clustering of the output space, at the risk of making mistakes from which there is no recovery.

More recently, estimators based on nearest neighbor search have been proposed for inference and learning in log-linear models BID25 BID26 .

These estimators hinge on Maximum Inner Product Search using Locality-Sensitive to retrieve the largest logits of the distribution and account for the tail with uniformly sampled classes.

They boast strong theoretical guarantees and well-established concentration bounds.

However, they were constrained to toy settings and not directly applicable to real-world, large-scale, machine learning.

In this work, we build upon these estimators to make them amenable to deep learning practitioners, without losing any theoretical guarantees.

We first show how they can be extended to be usable within training of deep learning models, then present our efficient implementation, adapted to deep learning hardware and frameworks.

Finally, we show the applicability and efficiency of our method by evaluating on a real-world task: language modeling.

We show significant perplexity gains against competing methods with significant speed-ups.

Our contributions are as follows:• We present a new deep learning layer, LSH Softmax, an efficient replacement for the softmax layer based on Locality-Sensitive Hashing and the Gumbel distribution, for any deep learning architecture, with strong theoretical guarantees for sub-linear learning and inference.• We provide details for efficient implementation on deep learning hardware (GPUs) and modern deep learning frameworks BID0 BID19 ).•

Empirically, we show, on several datasets, that training and sampling from LSH Softmax performs similarly to an exact softmax while requiring significantly less FLOPS.

In this section, we first provide a quick overview of Neural Networks and the most popular classification layer, the softmax layer.

We then present the Gumbel distribution BID7 and introduce Locality-Sensitive Hashing BID11 , both of which our estimator is built upon for inference and learning.

Notationally, X is the input space, e.g. X R d and Y is a discrete output space: Y {1, . . .

, C}.

Feedforward Networks Neural networks models are built hierarchically by applying linear and non-linear transformations in alternating fashion.

Formally, given input x ∈ X , an m-layer neural network with σ(·) non-linearity transforms x into h defined as: DISPLAYFORM0 {W i } i≤m and {b i } i≤m are learned weights of the network.

σ(·) denotes an element-wise nonlinearity such as ReLU (max(·, 0)) or sigmoid ((1 + exp(−·)) −1 ).Recurrent Networks Recurrent Neural Networks (RNN) are an extension of the previous setting to arbitrarily long sequences by keeping an internal state h t .

Formally, given an input sequence (x 1 , . . . , x T ), it can be written as a dynamical system of the form: DISPLAYFORM1 where U and V are learnable weight matrices.

In practice, this parametrization is not wellconditioned for optimization as it can be subject to vanishing or exploding gradients and in practice the Longer Short Term Memory (LSTM; BID10 ) is preferred.

In both cases, these outputs are then given as input to a softmax layer which produces a distribution over the output space Y. In the rest of this work, we denote by φ the parameters of the neural network.

The softmax layer is the common name given to a log-linear model for multi-classification at the end of a neural network.

Let us consider the multi-classification setting with inputs in X and outputs in Y. Given a feature vector ψ(x) and C weight vectors {θ c } c≤C , the softmax layer parameterizes the following distribution: DISPLAYFORM0 In the particular case of neural networks, p(y|x; θ, φ) ∝ exp(h T θ i ).

{h T θ i } i≤C are called the logits.

It is important to note that computing the distribution over the output space, for inference or learning, requires O(C) operations.

For the rest of this work, θ denotes the parameters of the softmax whereas φ denotes the parameters of the neural network (producing the feature).

First introduced by BID7 , the Gumbel distribution is defined by the following cumulative distribution function: p(G < s) = exp(− exp(−s)).

More practically, one can sample from the Gumbel distribution by first sampling U ∼ U[0, 1] and returning G = − log(− log(U )).

This distribution is particularly useful as it casts sampling as optimization.

Theorem 1 (Maddison et al. FORMULA0 ).

Let {y i } i≤C be un-normalized probabilities (or logits) over Y and let {G i } i≤C be i.i.d Gumbel variables.

Then: DISPLAYFORM0

Nearest neighbor search is a task that arises in many fields, such as information retrieval.

Given a fixed set of vectors S and a distance, this task consists of, given any incoming query q, returning the vectors closest to the query according to the specified distance.

In this work, we will be interested in the Maximum Inner Product Search (MIPS) task.

Let S = {s 1 , . . .

, s N } be a subset of R d .

Given a query q ∈ R d , MIPS aims at retrieving arg max s∈S q T s.

This requires Θ(N ) operations as one has to compute the dot-product of q with all elements of S.In the case where we assume that, for a given set S, it is needed to retrieve the nearest neighbor for a large numbers of queries, we can achieve amortized sub-linear time.

This problem is commonly addressed with space partitioning techniques, such as Locality-Sensitive Hashing (LSH; BID11 ).

LSH leverages hashing to reduce the number of candidate vectors to evaluate, based on the idea that similar vectors will hash in the same bucket.

We have the following result: Theorem 2 BID11 ).

Given a set S of size N , a similarity measure d(·, ·) and a family of hash functions H s.t.

for S > T and p > q: DISPLAYFORM0 we can construct a data structure s.t.

given an incoming query q, a nearest neighbor can be retrieved, with high probability, in sub-linear time O(N ρ log N ) with ρ log p log q < 1.Recent work builds on top of LSH to either reduce the number of tables BID18 , or utilize more expressive hash functions BID2 .

A common family of hash is the hyperplane hash, i.e. for v ∼ N (0, I), h v (x) = sign v T x , also called Signed Random Projections BID4 .

For the rest of this work, we denote b the number of hashing bits (equivalently, the number of random vectors) per table, and L the number of tables.

In this section, we show how we can apply Theorem 3.5 of BID26 to enable sublinear learning of softmax parameters in the context of deep models, i.e. where both weights and inputs can change.

This is crucial for real-world use.

Deep learning models for both classification BID16 and generation BID22 are often trained with a maximum-likelihood objective.

Formally, given a training pair (x, y) ∈ X × Y, one aims at maximizing log p(y|x; θ, φ), where θ ∈ Θ and φ ∈ Φ are respectively the parameters of the softmax and of the neural network.

To optimize this model, the usual method is to use back-propagation BID28 to differentiate and then perform stochastic gradient descent (SGD; BID17 ) on θ and φ.

Let's denote by f (x; φ) h the feature vector given as input to the softmax.

Given our notation, the objective is written as (x, y, θ, φ) DISPLAYFORM0 For backpropagation, we need to compute the gradient of w.r.t to both θ and h -the gradient w.r.t.

h is then passed down to compute the gradient w.r.t.

φ.

DISPLAYFORM1 with DISPLAYFORM2 Computing these gradients clearly requires O(|Y|) operations.

In practice, this constitutes a major bottleneck for large output spaces.

BID26 shows how to compute expectation in in sub-linear time, with a well-concentrated estimator using an LSH structure.

Intuitively, we can build a good estimate of the partition function by retrieving the largest logits (using LSH) and accounting for the tail with uniform samples.

Applying this result, we can compute the expectations necessary to compute the softmax gradients in sub-linear time.

This is described in Theorem 3.

Theorem 3 (LSH Softmax for Learning).

Let h = f (x; φ) be input to a softmax layer with parameters {θ c } c≤C and define (x, y, θ, φ) as previously.

Given S, the k-nearest neighbors of h in {θ c } c≤C and T , l uniform samples from {1, . . .

, C} − S, let us define: DISPLAYFORM3 DISPLAYFORM4 These estimators are well concentrated: i.e. for , δ > 0, if k = l = O n 2 3 1 1 δ , then with probability greater than 1 − δ: DISPLAYFORM5 While Theorem 3 provides computation of the gradients in sub-linear time, it is only usable in a setting where the weights ({θ i } i≤C ) are not updated.

Indeed, querying nearest neighbors in sublinear time assumes that an appropriate data structure (here LSH) was built in advance.

However, when training deep models, we are required to update the weights at every training step.

This necessitates online updating of the LSH structure.

To maintain the sub-linear runtime, we perform these updates in a sparse manner.

We describe in Algorithm 1 how this estimator can be used in a training loop, with weight updating and sparse LSH updates.

Proposition 4.

The softmax computations described in Algorithm 1 run in sub-linear time.

, n iters number of training iterations.

Initialize θ and φ Initialize the MIPS structure with {θ i } i≤|V| .

for j ≤ n iters doSample an example (x, y) from D. DISPLAYFORM0 Find S, k-nearest-neighbors of h using the MIPS.

Define T as l indexes uniformly sampled from Y − S. DISPLAYFORM1 Pass downĝ h for back-propagation.

Re-hash the updated vectors (at most (k + l)) into the right buckets.

end forProof.

The softmax computations can be split into three parts: retrieving nearest neighbors, computing the forward/backward passes, and rehashing updated vectors.

With a sub-linear MIPS such as LSH, the first part is guaranteed to be sub-linear.

For the second part, computing the partition function and the entire gradient estimator requires computing a finite number of sums over DISPLAYFORM2 3 ) terms, which is sub-linear.

The third part consists of re-hashing updated vectors.

Re-hashing a vector is a constant operation (consisting of b × L dot-products) and thus, given that only a sub-linear number of vectors are updated, re-hashing is sub-linear.

In the last section, we presented a method to speed-up training time based on an LSH data structure.

In addition to these training time gains, LSH Softmax can be utilized for computational gains at inference time as well.

While MAP inference can be easily derived from the MIPS structure, sampling from the conditional distribution is often required (e.g. to generate diverse sentences in language modeling or machine translation).

These gains can be crucial for large-scale deployment.

This is a direct application of BID26 that once again leverages a MIPS structure and the Gumbel distribution.

By lazily evaluating Gumbel noise, once can devise an inference scheme which allows to sample from log-linear models in sub-linear time.

Theorem 5 (LSH Softmax for Inference).

We reuse the same notations as the once in Theorem 3.

We define t − log(− log(1 − l/C)).

Let {G i } i≤k be k samples from the Gumbel distribution.

We then proceed to sample m ∼ Binomial(C, l/C), and sample T , m points from Y − S with associated Gumbels {G i } i≤m s.t.

each G i are larger than t. Let us define: DISPLAYFORM0 Let , δ > 0, we then have the two following results:1.

For k = l ≥ log 1 δ ,ŷ is a sample from p(y|x; θ, φ) with probability greater than 1 − δ.2.

This inference scheme runs in sub-linear time.

Proof.

BID26 We denote by p Gumbel (·|h; θ) the implicit distribution over Y provided by this inference scheme.

While we can sample from p Gumbel , we note that the likelihood is intractable.

We also emphasize that this scheme can be utilized for any softmax model, regardless of the training method.

Recent successes of deep neural networks hinge on their efficient implementation on specialized hardware: Graphics Processor Units (GPU), which enables training of large models in reasonable time.

Often, methods with theoretically faster runtime are dismissed by practitioners because of their incompatibility with the hardware, rendering them hard to implement efficiently and ultimately not widely used.

In this section, we first detail how our method is indeed amenable to GPU implementation and can amount to wall-clock gains in practice, and explain why LSH Softmax is easy to implement in the context of modern deep learning frameworks who often provide a gradient computation API.GPU Implementation Standard LSH implementations consist of three steps: DISPLAYFORM0 b , retrieve candidates in each of the L tables.

Let us denote C q the number of candidates retrieved.3.

Distances: Given those candidates {x 1 , . . .

, x Cq } ⊂ R d , compute the distances {q T x i }

i≤Cq and only return the closest one.

It is also important to note that deep learning models are often trained using minibatch optimization; let us describe how each of these steps can be computed efficiently and in the minibatch setting.

The first step is amenable to the GPU setting; a batch of queries {q i } i≤m ⊂ R d can be represented by Q ∈ R m×d .

Given that the hyperplanes are similarly presented in matrix form i.e. H ∈ R d×(b×L) , the hashing step is equivalent to sign (Q · H) ∈ {0, 1} m×(b×L) .

This is the type of operations that GPUs excel at: matrix-multiply followed by element-wise function.

The second step, while not as compatible with GPU, is still massively parallelizable using multithreading on CPU.

Given the computed signatures, one can run parallelism at the query level (i.e. each thread retrieves candidates for a given query), rendering that step efficient.

It also allows for more memory-efficient look-up such as BID18 .The last operation is, once again, very amenable to GPU.

It simply consists of a gather (i.e. building a matrix with the appropriate indexes from the candidates) into a 3-d tensor.

Indeed, after the previous step, the LSH structure returns m lists of s candidates, and the gather step returns the appropriate vectors from the vocabulary into a 3-d tensor of shape R m×s×d .

As the batched queries can be also seen as a 3-d tensor R m×d×1 , computing the exact distances then reduces to a batch matrix-multiply which is a very efficient operation on GPU.Software Implementation Another crucial point for practitioners is the ability to rely on frameworks automatically providing gradients, such as BID0 BID19 , to implement deep learning models; this abstracts away the need to write down the exact gradients which can be both cumbersome and error-prone.

An additional advantage of our estimator is that it can be effortlessly implemented in these frameworks.

Indeed, given logits computed over the nearest-neighbors and the additional uniformly sampled indexes, one can compute the estimate of the partition function and thus an estimate of the loss.

Computing the gradient estimators now reduces to differentiating this loss, which can be very simply done using the framework's differentiation API.

After having presented our new layer LSH Softmax, we now proceed to show its applicability and efficiency in a real-world setting for deep learning practitioners, specifically towards language modeling.

We first show that our method significantly outperforms approximate softmax baselines while performing within 20% of the performance of the exact softmax.

We then provide a computational comparison.

While we evaluate our method on NLP tasks, we want to emphasize that it is directly applicable to other domains, such as vision.

However, public vision benchmark datasets with large output spaces require significantly more computational resources (e.g. 98 GPU nodes for 8 days for Flickr100M BID31 ) which is outside the scope of this paper.

Language modeling is the task of, given a sequence of words (w 1 , . . .

, w T ) in a vocabulary V, estimating p(w 1 , . . . , w T ) = t≤T p(w t |w <t ).

Substantial work has been done to model these distributions using non-parametric n-gram counts with additional smoothing techniques, but can fail to model long histories because of an exponential number of sequences.

Recently, parametric models using RNNs have shown impressive success on these tasks BID22 .

In this setting, large output spaces arise naturally, as the vocabulary size can range from 10 4 to 10 6 .

We first describe our experimental protocol, and then report perplexity (ppl) of LSH Softmax against a set of baselines on this task for several datasets.

Datasets We evaluate our method on three standard datasets for Language Modeling with varying number of characters and vocabulary size:• Penn TreeBank (PTB): We follow the pre-processing described by BID22 , which results in 929k training tokens, 73k validation and 82k test tokens with a 10k vocabulary size.• Text8 is a dataset consisting of the first 100 millions characters of Wikipedia, and has a vocabulary size of 44k.

This dataset has been used recently in the context of language modeling BID33 .

We use the 90M first words for training and split the remaining between the validation and test set.• Wikitext-2.

First introduced in BID21 , this is a selected corpus of Wikipedia articles.

It has a vocabulary size of 33k and contains 217k tokens.

As previously, we split between a training, validation and testing set.

Baselines We evaluate the performance of models trained with (1) exact softmax i.e. computed over the entire output space, (2) Biased Importance Sampled softmax (BIS), as presented in BID12 , which consists of sub-sampling the vocabulary according to a proposal distribution based on unigram counts, and (3) Negative Sampling (NS), proposed in , equivalent to (BIS) with a uniform distribution, (4) standard Importance Sampling BID15 and ( Our models are trained using SGD using gradient clipping, with an initial learning rate of 20.

This learning rate is annealed when the validation perplexity plateaus.

Our models are trained for 40 epochs for PTB, 3 epochs for Text8, 25 epochs for Wikitext-2.

With the notations of Theorem 3, for LSH Softmax, we choose k = 10 |V| and l = |V|.

For the IS and NS baselines, we choose to sample k + l classes from the output space for a fair comparison.

We choose the number of bits per signature b log 2 |V| and choose L, number of tables, to have sufficient recall for the MIPS task.

We report perplexity for a fixed architecture but comparing different softmax evaluations; we present both learning curves and perplexity on each set.

We report the perplexity of all trained models using the exact probabilities i.e. the full softmax.

Perplexities are reported in Table 1 and learning curves in FIG1 .

We see that LSH Softmax consistently outperforms the approximate baselines by a fair margin while performing a similar number of operations, showcasing the strength of this estimator.

We also observe from the training curves that approximate methods' performances tend to plateau, as IS and NS cannot target the proper classes to push down.

In constrast, LSH Softmax does not.

Having established that the proposed estimator performs very well on real-world tasks, we now proceed to evaluate the computation gains.

It is important to note that for models with large output spaces, the softmax computation can amount to about 80% of the total computation (Joulin et 2016; BID13 ; we thus choose to only evaluate computational gains in the softmax layer.

We evaluate our method in CPU, with a batch size of 1, to have an accurate estimation of the ratio of FLOPS.

We report both speed-up and validation perplexity (ppl) relative difference with the exact softmax for LSH Softmax and NS.

Note that NS requires the same number of operations as importance sampling (IS) but outperforms it in all tasks.

Additionally, we show the speed-ups one can achieve on the One Billion Word dataset BID5 , whose ppl was not evaluated due to computational constraints.

We report the results in Table 2 .

We observe that, while faster, NS performs significantly worse than LSH Softmax.

Furthermore, its performance deteriorates significantly when increasing the size of the output space, contrary to LSH Softmax which always performs in the same relative range.

Table 2 : LSH Softmax performs closest to the exact softmax and handily outperforms importance sampling based methods with no concentration guarantees.

In recent years, MIPS-based estimators for log-linear models have been explored in the literature.

BID32 propose retrieving the largest logits using LSH and estimating the Softmax using only those classes.

Their method is encompassed in ours by simply setting l to 0.

However, we note that not accounting for the tail can lead to highly biased gradients.

Indeed, BID26 show that, using only the top-k largest values leads to significantly worse performance.

In a similar direction, BID30 propose using LSH at each layer and only retaining the largest activations which can be viewed as a form of adaptive dropout.

This work differs with ours in two ways: first of all, their paper provides no theoretical guarantees and secondly, they focus on reducing memory footprint which is not the aim of our work.

Finally, BID29 proposed using the LSH structure as a proposal distribution to evaluate the Softmax.

While unbiased and efficient, their method does not offer any concentration guarantees and the estimator can have arbitrarily bad variance.

In this work, we presented LSH Softmax, a softmax approximation layer for large output spaces with sub-linear learning and inference cost (in the number of states) and strong theoretical guarantees.

We showcased both its applicability and efficiency by evaluating LSH on a common NLP task, language modeling.

On several datasets for this task, we report perplexity closest to exact training among all baselines, as well as significant speed-ups.

Our hope is that, for any architecture, this layer could be chosen in lieu of softmax, when the output space is sufficiently large to warrant the approximation.

To that end, we plan to release source-code with the camera-ready version.

<|TLDR|>

@highlight

we present LSH Softmax, a softmax approximation layer for sub-linear learning and inference with strong theoretical guarantees; we showcase both its applicability and efficiency by evaluating on a real-world task: language modeling.