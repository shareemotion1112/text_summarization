While deep neural networks are a highly successful model class, their large memory footprint puts considerable strain on energy consumption, communication bandwidth, and storage requirements.

Consequently, model size reduction has become an utmost goal in deep learning.

A typical approach is to train a set of deterministic weights, while applying certain techniques such as pruning and quantization, in order that the empirical weight distribution becomes amenable to Shannon-style coding schemes.

However, as shown in this paper, relaxing weight determinism and using a full variational distribution over weights allows for more efficient coding schemes and consequently higher compression rates.

In particular, following the classical bits-back argument, we encode the network weights using a random sample, requiring only a number of bits corresponding to the Kullback-Leibler divergence between the sampled variational distribution and the encoding distribution.

By imposing a constraint on the Kullback-Leibler divergence, we are able to explicitly control the compression rate, while optimizing the expected loss on the training set.

The employed encoding scheme can be shown to be close to the optimal information-theoretical lower bound, with respect to the employed variational family.

Our method sets new state-of-the-art in neural network compression, as it strictly dominates previous approaches in a Pareto sense: On the benchmarks LeNet-5/MNIST and VGG-16/CIFAR-10, our approach yields the best test performance for a fixed memory budget, and vice versa, it achieves the highest compression rates for a fixed test performance.

With the celebrated success of deep learning models and their ever increasing presence, it has become a key challenge to increase their efficiency.

In particular, the rather substantial memory requirements in neural networks can often conflict with storage and communication constraints, especially in mobile applications.

Moreover, as discussed in BID4 , memory accesses are up to three orders of magnitude more costly than arithmetic operations in terms of energy consumption.

Thus, compressing deep learning models has become a priority goal with a beneficial economic and ecological impact.

Traditional approaches to model compression usually rely on three main techniques: pruning, quantization and coding.

For example, Deep Compression BID5 proposes a pipeline employing all three of these techniques in a systematic manner.

From an information-theoretic perspective, the central routine is coding, while pruning and quantization can be seen as helper heuristics to reduce the entropy of the empirical weight-distribution, leading to shorter encoding lengths BID15 .

Also, the recently proposed Bayesian Compression BID13 falls into this scheme, despite being motivated by the so-called bits-back argument BID8 which theoretically allows for higher compression rates.1 While the bits-back argument certainly motivated the use of variational inference in Bayesian Compression, the downstream encoding is still akin to Deep Compression (and other approaches).

In particular, the variational distribution is merely used to derive a deterministic set of weights, which is subsequently encoded with Shannonstyle coding.

This approach, however, does not fully exploit the coding efficiency postulated by the bits-back argument.

In this paper, we step aside from the pruning-quantization pipeline and propose a novel coding method which approximately realizes bits-back efficiency.

In particular, we refrain from constructing a deterministic weight-set but rather encode a random weight-set from the full variational posterior.

This is fundamentally different from first drawing a weight-set and subsequently encoding it -this would be no more efficient than previous approaches.

Rather, the coding scheme developed here is allowed to pick a random weight-set which can be cheaply encoded.

By using results from BID6 , we show that such an coding scheme always exists and that the bits-back argument indeed represents a theoretical lower bound for its coding efficiency.

Moreover, we propose a practical scheme which produces an approximate sample from the variational distribution and which can indeed be encoded with this efficiency.

Since our algorithm learns a distribution over weightsets and derives a random message from it, while minimizing the resulting code length, we dub it Minimal Random Code Learning (MIRACLE).From a practical perspective, MIRACLE has the advantage that it offers explicit control over the expected loss and the compression size.

This is distinct from previous techniques, which require tedious tuning of various hyper-parameters and/or thresholds in order to achieve a certain coding goal.

In our method, we can simply control the KL-divergence using a penalty factor, which directly reflects the achieved code length (plus a small overhead), while simultaneously optimizing the expected training loss.

As a result, we were able to trace the trade-off curve for compression size versus classification performance ( FIG4 ).

We clearly outperform previous state-of-the-art in a Pareto sense: For any desired compression rate, our encoding achieves better performance on the test set; vice versa, for a certain performance on the test set, our method achieves the highest compression.

To summarize, our main contributions are:• We introduce MIRACLE, an innovative compression algorithm that exploits the noise resistance of deep learning models by training a variational distribution and efficiently encodes a random set of weights.• Our method is easy to implement and offers explicit control over the loss and the compression size.• We provide theoretical justification that our algorithm gets close to the theoretical lower bound on the encoding length.• The potency of MIRACLE is demonstrated on two common compression tasks, where it clearly outperforms previous state-of-the-art methods for compressing neural networks.

In the following section, we discuss related work and introduce required background.

In Section 3 we introduce our method.

Section 4 presents our experimental results and Section 5 concludes the paper.

There is an ample amount of research on compressing neural networks, so that we will only discuss the most prominent ones, and those which are related to our work.

An early approach is Optimal Brain Damage (LeCun et al., 1990) which employs the Hessian of the network weights in order to determine whether weights can be pruned without significantly impacting training performance.

A related but simpler approach was proposed in BID4 , where small weights are truncated to zero, alternated with re-training.

This simple approach yielded -somewhat surprisingly -networks which are one order of magnitude smaller, without impairing performance.

The approach was refined into a systematic pipeline called Deep Compression, where magnitude-based weight pruning is followed by weight quantization (clustering weights) and Huffman coding BID10 .

While a variational posterior.

However, in order to realize this effective cost, one needs to encode both the network weights and the training targets, while it remains unclear whether it can also be achieved for network weights alone.its compression ratio (∼ 50×) has been surpassed since, many of the subsequent works took lessons from this paper.

HashNet proposed by BID1 also follows a simple and surprisingly effective approach: They exploit the fact that training of neural networks is resistant to imposing random constraints on the weights.

In particular, they use hashing to enforce groups of weights to share the same value, yielding memory reductions of up to 64× with gracefully degrading performance.

Weightless encoding by BID14 demonstrates that neural networks are resilient to weight noise, and exploits this fact for a lossy compression algorithm.

The recently proposed Bayesian Compression BID13 ) uses a Bayesian variational framework and is motivated by the bits-back argument BID8 .

Since this work is the closest to ours, albeit with important differences, we discuss Bayesian Compression and the bits-back argument in more detail.

The basic approach is to equip the network weights w with a prior p and to approximate the posterior using the standard variational framework, i.e. maximize the evidence lower bound (ELBO) for a given dataset DISPLAYFORM0 w.r.t.

the variational distribution q φ , parameterized by φ.

The bits-back argument BID8 ) establishes a connection between the Bayesian variational framework and the Minimum Description Length (MDL) principle BID3 .

Assuming a large dataset D of input-target pairs, we aim to use the neural network to transmit the targets with a minimal message, while the inputs are assumed to be public.

To this end, we draw a weight-set w * from q φ , which has been obtained by maximizing FORMULA0 ; note that knowing a particular weight w * set conveys a message of length H[q φ ] (H refers to the Shannon entropy of the distribution).

The weight-set w * is used to encode the residual of the targets, and is itself encoded with the prior distribution p, yielding a message of length DISPLAYFORM1 .

This message allows the receiver to perfectly reconstruct the original targets, and consequently the variational distribution q φ , by running the same (deterministic) algorithm as used by the sender.

Consequently, with q φ at hand, the receiver is able to retrieve an auxiliary message encoded in w * .

When subtracting the length of this "free message" from the original E q φ [log p] nats, 2 we yield a net cost of KL(q φ ||p) = E q φ [log q φ p ] nats for encoding the weights, i.e. we recover the ELBO (1) as negative MDL BID8 .In BID9 BID2 coding schemes were proposed which practically exploited the bits-back argument for the purpose of coding data.

However, it is not clear how these free bits can be spent solely for the purpose of model compression, as we only want to store a representation of our model, while discarding the training data.

Therefore, while Bayesian Compression is certainly motivated by the bits-back argument, it actually does not strive for the postulated coding efficiency KL(q φ ||p).

Rather, this method imposes a sparsity inducing prior distribution to aid the pruning process.

Moreover, high posterior variance is translated into reduced precision which constitutes a heuristic for quantization.

In the end, Bayesian Compression merely produces a deterministic weight-set w * which is encoded similar as in preceding works.

In particular, all previous approaches essentially use the following coding scheme, or a (sometimes sub-optimal) variant of it.

After a deterministic weight-set w * has been obtained, involving potential pruning and quantization techniques, one interprets w * as a sequence of i.i.d.

variables, taking values from a finite alphabet.

Then one assumes the coding distribution DISPLAYFORM2 where δ x denotes the Kronecker delta at x. According to Shannon's source coding theorem BID15 , w * can be coded with no less than N H[p ] nats, which is asymptotically achieved by Huffman coding, like in BID5 .

Note that the Shannon lower bound can be written as DISPLAYFORM3 where we have set p (w) = i p (w i ).

Thus, these Shannon-style coding schemes are in some sense optimal, when the variational family is restricted to point-measures, i.e. deterministic weights.

By extending the variational family to comprise more general distributions q, the coding length KL(q||p) could be drastically reduced.

In the following, we develop such a method which exploits the uncertainty represented by q in order to encode a random weight-set with short coding length.

Consider the scenario where we want to train a neural network but our memory budget is constrained to C nats.

As illustrated in the previous section, a variational approach offers -in principle -a simple and elegant solution.

Before we proceed, we note that we do not consider our approach to be a strictly Bayesian one, but rather based on the MDL principle, although these two are of course highly related BID3 .

In particular, we refer to p as an encoding distribution rather than a prior, and moreover we will use a framework akin to the β-VAE BID7 which better reflects our goal of efficient coding.

The crucial difference to the β-VAE being that we encode parameters rather than data.

Now, similar to BID13 , we first fix a suitable network architecture, select an encoding distribution p and a parameterized variational family q φ for the network weights w.

We consider, however, a slightly different variational objective related to the β-VAE: DISPLAYFORM0 This objective directly reflects our goal of achieving both a good training performance (loss term) and being able to represent our model with a short code (model complexity), at least according to the bits-back argument.

After obtaining q φ by maximizing (3), a weight-set drawn from q φ will perform comparable to a deterministically trained network, since the variance of the negative loss term will be comparatively small to the mean, and since the KL term regularizes the model.

Thus, our declared goal is to draw a sample from q φ such that this sample can be encoded as efficiently as possible.

This problem can be formulated as the following communication problem.

Alice observes a training data set (X, Y ) = D drawn from an unknown distribution p(D).

She trains a variational distribution q φ (w) by optimizing (3) for a given β using a deterministic algorithm.

Subsequently, she wishes to send a message M (D) to Bob, which allows him to generate a sample distributed according to q φ .

How long does this message need to be?The answer to this question depends on the unknown data distribution p(D), so we need to make an assumption about it.

Since the variational parameters φ depend on the realized dataset D, we can interpret the variational distribution as a conditional distribution q(w|D) := q φ (w), giving rise to the joint q(w, D) = q(w|D)p(D).

Now, our assumption about p(D) is that q(w|D)p(D) dD = p(w), that is, the variational distribution q φ yields the assumed encoding distribution p(w), when averaged over all possible datasets.

Note that this a similar strong assumption as in a Bayesian setting, where we assume that the data distribution is given as p(D) = p(D|w)p(w)dw.

In this setting, it follows immediately from the data processing inequality BID6 ) that in expectation the message length |M | cannot be smaller than KL(q φ ||p): DISPLAYFORM1 where I refers to the mutual information and in the third inequality we applied the data processing inequality for Markov chain D → M →

w. As discussed by BID6 , the inequal- DISPLAYFORM2 can be very loose.

However, as they further show, the message length can be brought close to the lower bound, if Alice and Bob are allowed to share a source of randomness:Theorem 3.1 BID6 ) Given random variables D, w and a random string R, let a protocol Π be defined via a message function M (D, R) and a decoder function w(M, R), DISPLAYFORM3 be the expected message length for data D, and let the minimal expected message length be defined as DISPLAYFORM4 where Π The results of BID6 establish a characterization of the mutual information in terms of minimal coding a conditional sample.

For our purposes, Theorem 3.1 guarantees that in principle DISPLAYFORM5 draw a sample w k * ∼q 7:return w k * , k * 8: end procedure there is an algorithm which realizes near bits-back efficiency.

Furthermore, the theorem shows that this is indeed a fundamental lower bound, i.e. that such an algorithm is optimal for the considered setting.

To this end, we need to refer to a "common ground", i.e. a shared random source R, where w.l.o.g.

we can assume that this source is an infinite list of samples from our encoding distribution p.

In practice, this can be realized via a pseudo-random generator with a public seed.

While BID6 provide a constructive proof using a variant of rejection sampling (see Appendix A), this algorithm is in fact intractable, because it requires keeping track of the acceptance probabilities over the whole sample domain.

Therefore, we propose an alternative method to produce an approximate sample from q φ , depicted in Algorithm 1.

This algorithm takes as inputs the trained variational distribution q φ and the encoding distribution p.

We first draw K = exp(KL(q φ ||p)) samples from p, using the shared random generator.

Subsequently, we craft a discrete proxy distributioñ q, which has support only on these K samples, and where the probability mass for each sample is proportional to the importance weights a k = q φ (w k ) p(w k ) .

Finally, we draw a sample fromq and return its index k * and the sample w k * itself.

Since any number 0 ≤ k * < K can be easily encoded with KL(q φ ||p) nats, we achieve our aimed coding efficiency.

Decoding the sample is easy: simply draw the k * th sample w k * from the shared random generator (e.g. by resetting the random seed).While this algorithm is remarkably simple and easy to implement, there is of course the question of whether it is a correct thing to do.

Moreover, an immediate caveat is that the number K of required samples grows exponentially in KL(q φ ||p), which is clearly infeasible for encoding a practical neural network.

The first point is addressed in the next section, while the latter is discussed in Section 3.3, together with other practical considerations.

The proxy distributionq in Algorithm 1 is based on an importance sampling scheme, as its probability masses are defined to be proportional to the usual importance weights DISPLAYFORM0 .

Under mild assumptions (q φ , p continuous; a k < ∞) it is easy to verify thatq converges to q φ in distribution for K → ∞; thus in the limit, Algorithm 1 samples from the correct distribution.

However, since we collect only K = exp(KL(q φ ||p)) samples in order to achieve a short coding length,q will be biased.

Fortunately, it turns out that K is just in the right order for this bias to be small.

DISPLAYFORM1 .

Furthermore, let f (w) be a measurable function and ||f || q φ = E q φ [f 2 ] be its 2-norm under q φ .

Then it holds that DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 Perform stochastic gradient update of L O 19: DISPLAYFORM5 else 23: DISPLAYFORM6 DISPLAYFORM7 which is precisely the importance sampling estimator for unnormalized distributions (denoted as J n in BID0 ), i.e. their Theorem 1.2 directly yields Theorem 3.2.

Note that the term e − t /4 decays quickly with t, and, since log q φ /p is typically concentrated around its expected value KL(q||p), the second term in (8) also quickly becomes negligible.

Thus, roughly speaking, Theorem 3.2 establishes that E q φ [f ] ≈ Eq[f ] with high probability, for any measurable function f .

This is in particular true for the function f (w) = log p(D|w) − β log q φ (w) p(w) .

Note that the expectation of this function is just the variational objective (3) we optimized to yield q φ in the first place.

Thus, DISPLAYFORM8 , replacing q φ byq is well justified.

Thereby, any sample ofq can trivially be encoded with KL(q φ ||p) nats, and decoded by simple reference to a pseudo-random generator.

Note that according to Theorem 3.2 we should actually take a number of samples somewhat larger than exp(KL(q φ ||p)) in order to make sufficiently small.

In particular, the results in BID0 ) also imply that a too small number of samples will typically be quite off the targeted expectation (for the worst-case f ).

However, although our choice of number of samples is at a critical point, in our experiments this number of samples yielded very good results.

In this section, we describe the application of Algorithm 1 within a practical learning algorithmMinimal Random Code Learning (MIRACLE) -depicted in Algorithm 2.

For both q φ and p we used Gaussians with diagonal covariance matrices.

For q φ , all means and standard deviations constituted the variational parameters φ.

The mean of p was fixed to zero, and the standard deviation was shared within each layer of the encoded network.

These shared parameters of p where learned jointly with q φ , i.e. the encoding distribution was also adapted to the task.

This choice of distributions allowed us to use the reparameterization trick for effective variational training and furthermore, KL(q φ ||p) can be computed analytically.

Since generating K = exp(KL(q φ ||p)) samples is infeasible for any reasonable KL(q φ ||p), we divided the overall problem into sub-problems.

To this end, we set a global coding goal of C nats and a local coding goal of C loc nats.

We randomly split the weight vector w into B = C C loc equally sized blocks, and assigned each block an allowance of C loc nats.

For example, fixing C loc to 11.09 nats ≈ 16 bits, corresponds to K = 65536 samples which need to be drawn per block.

We imposed block-wise KL constraints using block-wise penalty factors β b , which were automatically annealed via multiplication/division with (1 + β ) during the variational updates (see Algorithm 2).

Note that the random splitting into B blocks can be efficiently coded via the shared random generator, and only the number B needs communicated.

Before encoding any weights, we made sure that variational learning had converged by training for a large number of iterations I 0 = 10 4 .

After that, we alternated between encoding single blocks and updating the variational distribution not-yet coded weights, by spending I intermediate variational iterations.

To this end, we define a variational objective L O w.r.t.

to blocks which have not been coded yet, while weights of already encoded blocks were fixed to their encoded value.

Intuitively, this allows to compensate for poor choices in earlier encoded blocks, and was crucial for good performance.

Theoretically, this amounts to a rich auto-regressive variational family q φ , as the blocks which remain to be updated are effectively conditioned on the weights which have already been encoded.

We also found that the hashing trick BID1 further improves performance (not depicted in Algorithm 2 for simplicity).

The hashing trick randomly conditions weights to share the same value.

While BID1 apply it to reduce the entropy, in our case it helps to restrict the optimization space and reduces the dimensionality of both p and q φ .

We found that this typically improves the compression rate by a factor of ∼ 1.5×.

The experiments 3 were conducted on two common benchmarks: LeNet-5 on MNIST and VGG-16 on CIFAR-10.

As baselines we used three recent state-of-the-art methods, namely Deep Compression BID5 , Weightless encoding BID14 and Bayesian Compression BID13 .

The performance of the baseline methods are quoted from their respective source materials.

BID11 with the default learning rate (10 −3 ) and we set β0 = 10 −8 and β = 5 × 10 −5 .

For VGG, the means of the weights were initialized using a pretrained model.

4 We recommend applying the hashing trick mainly to reduce the size of the largest layers.

In particular, we applied the hashing trick was to layers 2 and 3 in LeNet-5 to reduce their sizes by 2× and 64× respectively and to layers 10-16 in VGG to reduce their sizes 8×.

The local coding goal C loc was fixed at 20 bits for LeNet-5 and it was varied between 15 and 5 bits for VGG (B was kept constant).

For the number of intermediate variational updates I, we used I = 50 for LeNet-5 and I = 1 for VGG, in order to keep training time reasonable (≈ 1 day on a single NVIDIA P100 for VGG).The performance trade-offs (test error rate and compression size) of MIRACLE along with the baseline methods and the uncompressed model are shown in FIG4 and TAB1 .

For MIRACLE we can easily construct the Pareto frontier, by starting with a large coding goal C (i.e. allowing a large coding length) and successively reducing it.

Constructing such a Pareto frontier for other methods is delicate, as it requires re-tuning hyper-parameters which are often only indirectly related to the compression size -for MIRACLE it is directly reflected via the KL-term.

We see that MIRACLE is Pareto-better than the competitors: for a given test error rate, we achieve better compression, while for a given model size we achieve lower test error.

In this paper we followed through the philosophy of the bits-back argument for the goal of coding model parameters.

The basic insight here is that restricting to a single deterministic weight-set and aiming to coding it in a classic Shannon-style is greedy and in fact sub-optimal.

Neural networks -and other deep learning models -are highly overparameterized, and consequently there are many "good" parameterizations.

Thus, rather than focusing on a single weight set, we showed that this fact can be exploited for coding, by selecting a "cheap" weight set out of the set of "good" ones.

Our algorithm is backed by solid recent information-theoretic insights, yet it is simple to implement.

We demonstrated that the presented coding algorithm clearly outperforms previous state-of-the-art.

An important question remaining for future work is how efficient MIRACLE can be made in terms of memory accesses and consequently for energy consumption and inference time.

There lies clear potential in this direction, as any single weight can be recovered by its block-index and relative index within each block.

By smartly keeping track of these addresses, and using pseudo-random generators as algorithmic lookup-tables, we could design an inference machine which is able to directly run our compressed models, which might lead to considerable savings in memory accesses.

This is shown by proving that q(w) − p i (w) ≤ q(w)(1 − p(w)) i for i ∈ N.In order to bound the encoding length, one has to first show that if the accepted sample has index i * , then E[log i * ] ≤ KL(q||p) + O(1) .Following this, one can employ the prefix-free binary encoding of BID16 .

Let l(n) be the length of the encoding for n ∈ N using the encoding scheme proposed by BID16 .

Their method is proven to have |l(n)| = log n + 2 log log(n + 1) + O(1), from which the upper bound follows: DISPLAYFORM0

@highlight

This paper proposes an effective method to compress neural networks based on recent results in information theory.