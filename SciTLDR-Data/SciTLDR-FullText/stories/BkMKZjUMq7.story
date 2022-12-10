While deep neural networks are a highly successful model class, their large memory footprint puts considerable strain on energy consumption, communication bandwidth, and storage requirements.

Consequently, model size reduction has become an utmost goal in deep learning.

Following the classical bits-back argument, we encode the network weights using a random sample, requiring only a number of bits corresponding to the Kullback-Leibler divergence between the sampled variational distribution and the encoding distribution.

By imposing a constraint on the Kullback-Leibler divergence, we are able to explicitly control the compression rate, while optimizing the expected loss on the training set.

The employed encoding scheme can be shown to be close to the optimal information-theoretical lower bound, with respect to the employed variational family.

On benchmarks LeNet-5/MNIST and VGG-16/CIFAR-10, our approach yields the best test performance for a fixed memory budget, and vice versa, it achieves the highest compression rates for a fixed test performance.

Traditional approaches to model compression usually rely on three main techniques: pruning, quantization and coding.

For example, Deep Compression BID3 proposes a pipeline employing all three of these techniques in a systematic manner.

From an information-theoretic perspective, the central routine is coding, while pruning and quantization can be seen as helper heuristics to reduce the entropy of the empirical weight-distribution, leading to shorter encoding lengths BID11 .

Also, the recently proposed Bayesian Compression BID9 falls into this scheme, despite being motivated by the so-called bits-back argument BID7 which theoretically allows for higher compression rates.1 While the bits-back argument certainly motivated the use of variational inference in Bayesian Compression, the downstream encoding is still akin to Deep Compression (and other approaches).

In particular, the variational distribution is merely used to derive a deterministic set of weights, which is subsequently encoded with Shannonstyle coding.

This approach, however, does not fully exploit the coding efficiency postulated by the bits-back argument.1 Recall that the bits-back argument states that, assuming a large dataset and a neural network equipped with a weight-prior p, the effective coding cost of the network weights is KL(q||p) = Eq[log q p ], where q is a variational posterior.

However, in order to realize this effective cost, one needs to encode both the network weights and the training targets, while it remains unclear whether it can also be achieved for network weights alone.

In this paper, we step aside from the pruning-quantization pipeline and propose a novel coding method which approximately realizes bits-back efficiency.

In particular, we refrain from constructing a deterministic weight-set but rather encode a random weight-set from the full variational posterior.

This is fundamentally different from first drawing a weight-set and subsequently encoding it -this would be no more efficient than previous approaches.

Rather, the coding scheme developed here is allowed to pick a random weight-set which can be cheaply encoded.

By using results from BID4 , we show that such a coding scheme always exists and that the bits-back argument indeed represents a theoretical lower bound for its coding efficiency.

Moreover, we propose a practical scheme which produces an approximate sample from the variational distribution and which can indeed be encoded with this efficiency.

Since our algorithm learns a distribution over weightsets and derives a random message from it, while minimizing the resulting code length, we dub it Minimal Random Code Learning (MIRACLE).

All preceding works BID2 BID9 BID10 BID1 essentially use the following coding scheme, or a (sometimes sub-optimal) variant of it.

After a deterministic weight-set w * has been obtained, involving potential pruning and quantization techniques, one interprets w * as a sequence of i.i.d.

variables and assumes the coding distribution (i.e. a dictionary) DISPLAYFORM0 , where δ x denotes the Dirac delta at x. According to Shannon's source coding theorem BID11 , w * can be coded with no less than N H[p ] nats (H denotes the Shannon entropy), which is asymptotically achieved by Huffman coding BID8 , like in BID3 .

However, note that the Shannon lower bound can also be written as DISPLAYFORM1 where we set p (w) = i p (w i ).

Thus, these Shannon-style coding schemes are in some sense optimal, when the variational family is restricted to point-measures, i.e. deterministic weights.

By extending the variational family to comprise more general distributions q, the coding length KL(q||p) could potentially be drastically reduced.

In the following, we develop one such method which exploits the uncertainty represented by q in order to encode a random weight-set with short coding length.

Consider the scenario where we want to train a neural network but our memory budget is constrained.

As illustrated in the previous section, a variational approach offers -in principle -a simple and elegant solution.

Now, similar to BID9 , we first fix a suitable network architecture, select an encoding distribution p and a parameterized variational family q φ for the network weights w.

We consider, however, a slightly different variational objective related to the β-VAE BID6 in order to be able to constrain the compression size using the penalty factor β: DISPLAYFORM0 This objective directly reflects our goal of achieving both a good training performance (loss term) and being able to represent our model with a short code (model complexity), at least according to the bits-back argument.

After obtaining q φ by maximizing (2), a weight-set drawn from q φ will perform comparable to a deterministically trained network, since the variance of the negative loss term will be comparatively small to the mean. , and since the KL term regularizes the model.

Thus, our declared goal is to draw a sample from q φ such that this sample can be encoded as efficiently as possible.

It turns out that the expected message length E[|M |] that allows for sampling q φ is bounded by the mutual information between the data D and the weights w BID4 BID5 : DISPLAYFORM1 Harsha et al. FORMULA1 provide a constructive proof that this lower-bound can be well approximated using a variant of rejection sampling.

However, this algorithm is in fact intractable, because it requires keeping track of the acceptance probabilities over the whole sample domain.

We propose a method to produce an approximate sample from q φ that can be cheaply encoded.

First, K = exp(KL(q φ ||p)) samples are drawn from p, using the shared random generator.

Subsequently, we craft a discrete proxy distributionq, which has support only on these K samples, and where the probability mass for each sample is proportional to the importance weights a k = q φ (w k ) p(w k ) .

Finally, we draw a sample fromq and return its index k * .

Since any number 0 ≤ k * < K can be easily encoded with KL(q φ ||p) nats, we achieve our aimed coding efficiency.

Decoding the sample is easy: simply draw the k * th sample w k * from the shared random generator (e.g. by resetting the random seed).

While this algorithm is remarkably simple and easy to implement, it can be shown that it produces a close-to unbiased sample from q φ BID0 BID5 .Furthermore, an immediate caveat is that the number K of required samples grows exponentially in KL(q φ ||p), which is clearly infeasible for encoding a practical neural network.

To deal with this issue, the weights are randomly split into groups each with a small, fixed allowance of nats such that drawing exp(KL(q φblock ||p block )) ≈ 10 6 samples can be done efficiently.

The experiments 2 were conducted on two common benchmarks, LeNet-5 on MNIST and VGG-16 on CIFAR-10, using a Gaussian distribution with diagonal covariance matrix for q φ .

As baselines, we used three recent state-of-the-art methods, namely Deep Compression BID3 , Weightless encoding BID10 and Bayesian Compression BID9 .

The performance of the baseline methods are quoted from their respective source materials.

We see that MIRACLE is Pareto-better than the competitors: for a given test error rate, we achieve better compression, while for a given model size we achieve lower test error FIG0 ).

In this paper we followed through the philosophy of the bits-back argument for the goal of coding model parameters.

Our algorithm is backed by solid recent information-theoretic insights, yet it is simple to implement.

We demonstrated that it outperforms the previous state-of-the-art.

An important question remaining for future work is how efficient MIRACLE can be made in terms of memory accesses and consequently for energy consumption and inference time.

There lies clear potential in this direction, as any single weight can be recovered by its group-index and relative index within each group.

By smartly keeping track of these addresses, and using pseudo-random generators as algorithmic lookup-tables, we could design an inference machine which is able to directly run our compressed models, which might lead to considerable savings in memory accesses.

@highlight

This paper proposes an effective coding scheme for neural networks that encodes a random set of weights from a variational distribution.