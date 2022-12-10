Most distributed machine learning (ML) systems store a copy of the model parameters locally on each machine to minimize network communication.

In practice, in order to reduce synchronization waiting time, these copies of the model are not necessarily updated in lock-step, and can become stale.

Despite much development in large-scale ML, the effect of staleness on the learning efficiency is inconclusive, mainly because it is challenging to control or monitor the staleness in complex distributed environments.

In this work, we study the convergence behaviors of a wide array of ML models and algorithms under delayed updates.

Our extensive experiments reveal the rich diversity of the effects of staleness on the convergence of ML algorithms and offer insights into seemingly contradictory reports in the literature.

The empirical findings also inspire a new convergence analysis of SGD in non-convex optimization under staleness, matching the best-known convergence rate of O(1/\sqrt{T}).

With the advent of big data and complex models, there is a growing body of works on scaling machine learning under synchronous and non-synchronous 1 distributed execution BID8 BID11 BID29 .

These works, however, point to seemingly contradictory conclusions on whether non-synchronous execution outperforms synchronous counterparts in terms of absolute convergence, which is measured by the wall clock time to reach the desired model quality.

For deep neural networks, BID2 ; BID8 show that fully asynchronous systems achieve high scalability and model quality, but others argue that synchronous training converges faster BID1 BID5 .

The disagreement goes beyond deep learning models: ; BID49 ; BID26 ; BID31 ; BID41 empirically and theoretically show that many algorithms scale effectively under non-synchronous settings, but BID36 ; ; demonstrate significant penalties from asynchrony.

The crux of the disagreement lies in the trade-off between two factors contributing to the absolute convergence: statistical efficiency and system throughput.

Statistical efficiency measures convergence per algorithmic step (e.g., a mini-batch), while system throughput captures the performance of the underlying implementation and hardware.

Non-synchronous execution can improve system throughput due to lower synchronization overheads, which is well understood BID1 BID4 BID2 .

However, by allowing various workers to use stale versions of the model that do not always reflect the latest updates, non-synchronous systems can exhibit lower statistical efficiency BID1 BID5 .

How statistical efficiency and system throughput trade off in distributed systems, however, is far from clear.

The difficulties in understanding the trade-off arise because statistical efficiency and system throughput are coupled during execution in distributed environments.

Non-synchronous executions are in general non-deterministic, which can be difficult to profile.

Furthermore, large-scale experiments 2 RELATED WORK Staleness is reported to help absolute convergence for distributed deep learning in BID2 ; BID8 ; and has minimal impact on convergence BID31 BID6 BID51 BID32 .

But BID1 ; BID5 show significant negative effects of staleness.

LDA training is generally insensitive to staleness BID44 BID47 BID7 , and so is MF training BID48 BID33 BID4 BID49 .

However, none of their evaluations quantifies the level of staleness in the systems.

By explicitly controlling the staleness, we decouple the distributed execution, which is hard to control, from ML convergence outcomes.

We focus on algorithms that are commonly used in large-scale optimization BID11 BID1 BID8 , instead of methods specifically designed to minimize synchronization BID39 BID43 BID20 .

Non-synchronous execution has theoretical underpinning BID30 BID49 BID31 BID41 .

Here we study algorithms that do not necessarily satisfy assumptions in their analyses.

We study six ML models and focus on algorithms that lend itself to data parallelism, which a primary approach for distributed ML.

Our algorithms span optimization, sampling, and black box variational inference.

TAB1 summarizes the studied models and algorithms.

Simulation Model.

Each update generated by worker p needs to be propagated to both worker p's model cache and other worker's model cache.

We apply a uniformly random delay model to these updates that are in transit.

Specifically, let u each worker p (including p itself), our delay model applies a delay r BID40 BID9 BID23 BID16 BID12 , and dataset BID24 BID34 BID27 BID14 BID42 in our study.

η denotes learning rate, which, if not specified, are tuned empirically for each algorithm and staleness level, β1, β2 are optimization hyperparameters (using common default values).

α, β in LDA are Dirichlet priors for document topic and word topic random variables, respectively.

Convolutional Neural Networks (CNNs) have been a strong focus of large-scale training, both under synchronous BID11 BID5 BID3 and non-synchronous BID2 BID8 BID1 training.

We consider residual networks with 6n + 2 weight layers BID15 .

The networks consist of 3 groups of n residual blocks, with 16, 32, and 64 feature maps in each group, respectively, followed by a global pooling layer and a softmax layer.

The residual blocks have the same construction as in BID15 .

We measure the model quality using test accuracy.

For simplicity, we omit data augmentation in our experiments.

Deep Neural Networks (DNNs) are neural networks composed of fully connected layers.

Our DNNs have 1 to 6 hidden layers, with 256 neurons in each layer, followed by a softmax layer.

We use rectified linear units (ReLU) for nonlinearity after each hidden layer BID38 .

Multiclass Logistic Regression (MLR) is the special case of DNN with 0 hidden layers.

We measure the model quality using test accuracy.

Matrix factorization (MF) is commonly used in recommender systems and have been implemented at scale BID48 BID33 BID4 BID49 BID22 .

Let D ∈ R M ×N be a partially filled matrix, MF factorizes D into two factor matrices L ∈ R M ×r and R ∈ R N ×r (r min(M, N ) is the user-defined rank).

The 2 -penalized optimization problem is: DISPLAYFORM0 where || · || F is the Frobenius norm and λ is the regularization parameter.

We partition observations D to workers while treating L, R as shared model parameters.

We optimize MF via SGD, and measure model quality by training loss defined by the objective function above.

Latent Dirichlet Allocation (LDA) is an unsupervised method to uncover hidden semantics ("topics") from a group of documents, each represented as a bag of tokens.

LDA has been scaled under non-synchronous execution BID0 BID33 BID47 with great success.

Further details are provided in Appendix.

Variational Autoencoder (VAE) is commonly optimized by black box variational inference, which can be considered as a hybrid of optimization and sampling methods.

The inputs to VAE training include two sources of stochasticity: the data sampling x and samples of random variable .

We measure the model quality by test loss.

We use DNNs with 1∼3 layers as the encoders and decoders in VAE, in which each layer has 256 units furnished with rectified linear function for non-linearity.

The model quality is measured by the training objective value, assuming continuous input x and isotropic Gaussian prior p(z) ∼ N (0, I).

We use batch size 32 for CNNs, DNNs, MLR, and VAEs 34 .

For MF, we use batch size of 25000 samples, which is 2.5% of the MovieLens dataset (1M samples).

We study staleness up to s = 50 on 8 workers, which means model caches can miss updates up to 8.75 data passes.

For LDA we use DISPLAYFORM0 as the batch size, where D is the number of documents and P is the number of workers.

We study staleness up to s = 20, which means model caches can miss updates up to 2 data passes.

We measure time in terms of the amount of work performed, such as the number of batches processed.

Convergence Slowdown.

Perhaps the most prominent effect of staleness on ML algorithms is the slowdown in convergence, evident throughout the experiments.

FIG1 shows the number of batches needed to reach the desired model quality for CNNs and DNNs/MLR with varying network depths and different staleness (s = 0, ..., 16).

FIG1 (d) show that convergence under higher level of staleness requires more batches to be processed in order to reach the same model quality.

This additional work can potentially be quite substantial, such as in FIG1 where it takes up to 6x more batches compared with settings without staleness (s = 0).

It is also worth pointing out that while there can be a substantial slowdown in convergence, the optimization still reaches desirable models under most cases in our experiments.

When staleness is geometrically distributed FIG3 ), we observe similar patterns of convergence slowdown.

We are not aware of any prior work reporting slowdown as high as observed here.

This finding has important ramifications for distributed ML.

Usually, the moderate amount of workload increases due to parallelization errors can be compensated by the additional computation resources and higher system throughput in the distributed execution.

However, it may be difficult to justify spending large 3 Non-synchronous execution allows us to use small batch sizes, eschewing the potential generalization problem with large batch SGD BID21 BID35 .

4 We present RNN results in the Appendix.amount of resources for a distributed implementation if the statistical penalty is too high, which should be avoided (e.g., by staleness minimization system designs or synchronous execution).Model Complexity.

FIG1 also reveals that the impact of staleness can depend on ML parameters, such as the depths of the networks.

Overall we observe that staleness impacts deeper networks more than shallower ones.

This holds true for SGD, Adam, Momentum, RMSProp, Adagrad FIG1 , and other optimization schemes, and generalizes to other numbers of workers (see Appendix) 5 .This is perhaps not surprising, given the fact that deeper models pose more optimization challenges even under the sequential settings BID10 BID15 , though we point out that existing literature does not explicitly consider model complexity as a factor in distributed ML BID31 BID11 .

Our results suggest that the staleness level acceptable in distributed training can depend strongly on the complexity of the model.

For sufficiently complex models it may be more advantageous to eliminate staleness altogether and use synchronous training.

Algorithms' Sensitivity to Staleness.

Staleness has uneven impacts on different SGD variants.

Fig. 2 shows the amount of work (measured in the number of batches) to reach the desired model quality for five SGD variants.

Fig. 2 (d)(e)(f) reveals that while staleness generally increases the number of batches needed to reach the target test accuracy, the increase can be higher for certain algorithms, such as Momentum.

On the other hand, Adagrad appear to be robust to staleness 6 .

Our finding is consistent with the fact that, to our knowledge, all existing successful cases applying non-synchronous training to deep neural networks use SGD BID8 BID2 .

In contrast, works reporting subpar performance from non-synchronous training often use momentum, such as RMSProp with momentum BID1 and momentum BID5 .

Our results suggest that these different outcomes may be partly driven by the choice of optimization algorithms, leading to the seemingly contradictory reports of whether non-synchronous execution is advantageous over synchronous ones.

Effects of More Workers.

The impact of staleness is amplified by the number of workers.

In the case of MF, Fig. 3(b) shows that the convergence slowdown in terms of the number of batches (normalized by the convergence for s = 0) on 8 workers is more than twice of the slowdown on 4 workers.

For example, in Fig. 3 (b) the slowdown at s = 15 is ∼3.4, but the slowdown at the same staleness level on 8 workers is ∼8.2.

Similar observations can be made for CNNs (Fig. 3 ).

This can be explained by the fact that additional workers amplifies the effect of staleness by (1) generating updates that will be subject to delays, and (2) missing updates from other workers that are subject to delays.

Fig. 3 (c)(d) show the convergence curves of LDA with different staleness levels for two settings varying on the number of workers and topics.

Unlike the convergence curves for SGD-based algorithms (see Appendix), the convergence curves of Gibbs sampling are highly smooth, even under high staleness and a large number of workers.

This can be attributed to the structure of log likelihood objective function BID12 .

Since in each sampling step we only update the count statistics based on a portion of the corpus, the objective value will generally change smoothly.

Staleness levels under a certain threshold (s ≤ 10) lead to convergence, following indistinguishable log likelihood trajectories, regardless of the number of topics (K = 10, 100) or the number of workers (2-16 workers, see Appendix).

Also, there is very minimal variance in those trajectories.

However, for staleness beyond a certain level (s ≥ 15), Gibbs sampling does not converge to a fixed point.

The convergence trajectories are distinct and are sensitive to the number of topics and the number of workers.

There appears to be a "phase transition" at a certain staleness level that creates two distinct phases of convergence behaviors 7 .

We believe this is the first report of a staleness-induced failure case for LDA Gibbs sampling.

Fig. 3 (e)(f), VAEs exhibit a much higher sensitivity to staleness compared with DNNs ( FIG1 ).

This is the case even considering that VAE with depth 3 has 6 weight layers, which has a comparable number of model parameters and network architecture to DNNs with 6 layers.

We hypothesize that this is caused by the additional source of stochasticity from the sampling procedure, in addition to the data sampling process.

We now provide theoretical insight into the effect of staleness on the observed convergence slowdown.

We focus on the challenging asynchronous SGD (Async-SGD) case, which characterizes the neural network models, among others.

Consider the following nonconvex optimization problem DISPLAYFORM0 where f i corresponds to the loss on the i-th data sample, and the objective function is assumed to satisfy the following standard conditions:Assumption 1.

The objective function F in the problem (P) satisfies:1.

Function F is continuously differentiable and bounded below, i.e., inf x∈R d F (x) > −∞; 2.

The gradient of F is L-Lipschitz continuous.

Notice that we allow F to be nonconvex.

We apply the Async-SGD to solve the problem (P).

Let ξ(k) be the mini-batch of data indices sampled from {1, . . .

, n} uniformly at random by the algorithm at iteration k, and |ξ(k)| is the mini-batch size.

Denote mini-batch gradient as ∇f ξ(k) (x k ) := i∈ξ(k) ∇f i (x k ).

Then, the update rule of Async-SGD can be written as DISPLAYFORM1 where η k corresponds to the stepsize, τ k denotes the delayed clock and the maximum staleness is assumed to be bounded by s. This implies that DISPLAYFORM2 The optimization dynamics of Async-SGD is complex due to the nonconvexity and the uncertainty of the delayed updates.

Interestingly, we find that the following notion of gradient coherence provides insights toward understanding the convergence property of Async-SGD.Definition 1 (Gradient coherence).

The gradient coherence at iteration k is defined as DISPLAYFORM3 Parameter µ k captures the minimum coherence between the current gradient ∇F (x k ) and the gradients along the past s iterations 8 .

Intuitively, if µ k is positive, then the direction of the current gradient is well aligned to those of the past gradients.

In this case, the convergence property induced by using delayed stochastic gradients is close to that induced by using synchronous stochastic gradients.

Note that Definition 1 only requires the gradients to be positively correlated over a small number of iterations s, which is often very small (e.g. <10 in our experiments).

Therefore, Definition 1 is not a global requirement on optimization path.

Even though neural network's loss function is non-convex, recent studies showed strong evidences that SGD in practical neural network training encourage positive gradient coherence BID28 BID32 .

This is consistent with the findings that the loss surface of shallow networks and deep networks with skip connections are dominated by large, flat, nearly convex attractors around the critical points BID28 BID21 , implying that the degree of non-convexity is mild around critical points.

We show in the sequel that µ k > 0 through most of the optimization path, especially when the staleness is minimized in practice by system optimization FIG3 .

Our theory can be readily adapted to account for a limited amount of negative µ k (see Appendix), but our primary interest is to provide a quantity that is (1) easy to compute empirically during the course of optimization 9 , and (2) informative for the impact of staleness and can potentially be used to control synchronization levels.

We now characterize the convergence property of Async-SGD.

Theorem 1.

Let Assumption 1 hold.

Suppose for some µ > 0, the gradient coherence satisfies µ k ≥ µ for all k and the variance of the stochastic gradients is bounded by σ 2 > 0.

Choose stepsize DISPLAYFORM4 .

Then, the iterates generated by the Async-SGD satisfy DISPLAYFORM5 8 Our gradient coherence bears similarity with the sufficient direction assumption in BID19 .

However, sufficient direction is a layer-wise and fixed delay, whereas our staleness is a random variable that is subject to system level factors such as communication bandwidth 9 It can be approximated by storing a pre-selected batch of data on a worker.

The worker just needs to compute gradient every T mini-batches to obtain approximate ∇F (x k ), ∇F (xt) in Definition 1.

TAB1 .

Shaded region is 1 standard deviation over 3 runs.

For computational efficiency, we approximate the full gradient ∇F (x k ) by gradients on a fixed set of 1000 training samples D f ixed and use ∇D f ixed F (x k ).

(c) The number of batches to reach 71% test accuracy on CIFAR10 for ResNet8-32 using 8 workers and SGD under geometric delay distribution (details in Appendix).We refer readers to Appendix for the the proof.

Theorem 1 characterizes several theoretical aspects of Async-SGD.

First, the choice of the stepsize η k = µ sL √ k is adapted to both the maximum staleness and the gradient coherence.

Intuitively, if the system encounters a larger staleness, then a smaller stepsize should be used to compensate the negative effect.

On the other hand, the stepsize can be accordingly enlarged if the gradient coherence along the iterates turns out to be high.

In this case, the direction of the gradient barely changes along the past several iterations, and a more aggressive stepsize can be adopted.

In summary, the choice of stepsize should trade-off between the effects caused by both the staleness and the gradient coherence.

depths optimized by SGD using 8 workers.

The x-axis m is defined in FIG3 Furthermore, Theorem 1 shows that the minimum gradient norm decays at the rate O( DISPLAYFORM6 ), implying that the Async-SGD converges to a stationary point provided a positive gradient coherence, which we observe empirically in the sequel.

On the other hand, the bound in Eq. (1) captures the trade-off between the maximum staleness s and the gradient coherence µ. Specifically, minimizing the right hand side of Eq. (1) with regard to the maximum staleness s yields the optimal choice s * = σµ log T L(F (x0)−infx F (x)) , i.e., a larger staleness is allowed if the gradients remain to be highly coherent along the past iterates.

Empirical Observations.

Theorem 1 suggests that more coherent gradients along the optimization paths can be advantageous under non-synchronous execution.

FIG3 shows the cosine similarity sim(a, b) := a·b a b between gradients along the convergence path for CNNs and DNNs 10 .

We observe the followings: (1) Cosine similarity improves over the course of convergence FIG3 ).

Except the highest staleness during the early phase of convergence, cosine similarity remains positive 11 .

In practice the staleness experienced during run time can be limited to small staleness , which minimizes the likelihood of negative gradient coherence during the early phase.(2) FIG4 shows that cosine similarity decreases with increasing CNN model complexity.

Theorem 1 implies that lower gradient coherence amplifies the effect of staleness s through the factor s µ 2 in Eq. (1).

This is consistent with the convergence difficulty encountered in deeper models FIG1 .

In this work, we study the convergence behaviors under delayed updates for a wide array of models and algorithms.

Our extensive experiments reveal that staleness appears to be a key governing parameter in learning.

Overall staleness slows down the convergence, and under high staleness levels the convergence can progress very slowly or fail.

The effects of staleness are highly problem 10 Cosine similarity is closely related to the coherence measure in Definition 1.

11 Low gradient coherence during the early part of optimization is consistent with the common heuristics to use fewer workers at the beginning in asynchronous training.

BID31 also requires the number of workers to follow DISPLAYFORM0 where K is the iteration number.dependent, influenced by model complexity, choice of the algorithms, the number of workers, and the model itself, among others.

Our empirical findings inspire new analyses of non-convex optimization under asynchrony based on gradient coherence, matching the existing rate of O(1/ √ T ).Our findings have clear implications for distributed ML.

To achieve actual speed-up in absolute convergence, any distributed ML system needs to overcome the slowdown from staleness, and carefully trade off between system throughput gains and statistical penalties.

Many ML methods indeed demonstrate certain robustness against low staleness, which should offer opportunities for system optimization.

Our results support the broader observation that existing successful nonsynchronous systems generally keep staleness low and use algorithms efficient under staleness.

A.1 PROOF OF THEOREM 1Theorem 2.

Let Assumption 1 hold.

Suppose the gradient coherence µ k is lower bounded by some µ > 0 for all k and the variance of the stochastic gradients is upper bounded by some σ 2 > 0.

DISPLAYFORM0 .

Then, the iterates generated by the Async-SGD satisfy DISPLAYFORM1 Proof.

By the L-Lipschitz property of ∇F , we obtain that for all k DISPLAYFORM2 Taking expectation on both sides of the above inequality and note that the variance of the stochastic gradient is bounded by σ 2 , we further obtain that DISPLAYFORM3 Telescoping the above inequality over k from 0 to T yields that DISPLAYFORM4 Rearranging the above inequality and note that DISPLAYFORM5 Note that the choice of stepsize guarantees that η k µ − > 0 for all k. Thus, we conclude that DISPLAYFORM6 where the last inequality uses the fact that DISPLAYFORM7 into the above inequality and simplifying, we finally obtain that DISPLAYFORM8 A.2 HANDLING NEGATIVE GRADIENT COHERENCE IN THEOREM 1Our assumption of positive gradient coherence (GC) is motivated by strong empirical evidence that GC is largely positive FIG3 in the main text).

Contrary to conventional wisdom, GC generally improves when approaching convergence for both SGD and Adam.

Furthermore, in practice, the effective staleness for any given iteration generally concentrates in low staleness for the non-stragglers .When some µ k are negative at some iterations, in eq. 11 in the Appendix we can move the negative terms in k η k µ k to the right hand side and yield a higher upper bound (i.e., slower convergence).

This is also consistent with empirical observations that higher staleness lowers GC and slows convergence.

A.3 EXPONENTIAL DELAY DISTRIBUTION.We consider delays drawn from geometric distribution (GD), which is the discrete version of exponential distribution.

For each iterate we randomly select a worker to be the straggler with large mean delay (p = 0.1), while all other non-straggler workers have small delays.

The non-straggler delay is drawn from GD with p chosen to achieve the same mean delay as in the uniform case (after factoring in straggler) in the main text.

The delay is drawn per worker for each iteration, and thus a straggler's outgoing updates to all workers suffer the same delay.

FIG3 (c) in the main text shows the convergence speed under the corresponding staleness s with the same mean delay (though s is not a parameter in GD).

It exhibits trends analogous to FIG1 in the main text: staleness slows convergence substantially and overall impacts deeper networks more.

We present additional results for DNNs.

Fig. 6 shows the number of batches, normalized by s = 0, to reach convergence using 1 hidden layer and 1 worker under varying staleness levels and batch sizes.

Overall, the effect of batch size is relatively small except in high staleness regime (s = 32).

Fig. 7 shows the number of batches to reach convergence, normalized by s = 0 case, for 5 variants of SGD using 1 worker.

The results are in line with the analyses in the main text: staleness generally leads to larger slow down for deeper networks than shallower ones.

SGD and Adagrad are more robust to staleness than Adam, RMSProp, and SGD with momentum.

In particular, RMSProp exhibit high variance in batches to convergence (not shown in the normalized plot) and thus does not exhibit consistent trend.

The results are consistent with the observations and analyses in the main text, namely, that having more workers amplifies the effect of staleness.

We can also observe that SGDS is more robust to staleness than Adam, and shallower networks are less impacted by staleness.

In particular, note that staleness sometimes accelerates convergence, such as in FIG6 .

This is due to the implicit momentum created by staleness .A.5 LDA AND ADDITIONAL RESULTS FOR LDAIn LDA each token w ij (j-th token in the i-th document) is assigned with a latent topic z ij from totally K topics.

We use Gibbs sampling to infer the topic assignments z ij .

The Gibbs sampling step involves three sets of parameters, known as sufficient statistics: FORMULA7 Figure 6 : The number of batches to reach 95% test accuracy using 1 hidden layer and 1 worker, respectively normalized by s = 0.vector φ w ∈ R K where φ wk is the number of topic assignments to topic k = 1, ..., K for word (vocabulary) w across all documents; (3)φ ∈ R K whereφ k = W w=1 φ wk is the number of tokens in the corpus assigned to topic k. The corpus (w ij , z ij ) is partitioned to workers, while φ w andφ are shared model parameters.

We measure the model quality using log likelihood.

We present additional results of LDA under different numbers of workers and topics in FIG12 and FIG1 .

These panels extends Fig. 3(c)(d) in the main text.

See the main text for experimental setup and analyses and experimental setup.

We show the convergence curves for MF under different numbers of workers and staleness levels in FIG1 .

It is evident that higher staleness leads to a higher variance in convergence.

Furthermore, the number of workers also affects variance, given the same staleness level.

For example, MF with 4 workers incurs very low standard deviation up to staleness 20.

In contrast, MF with 8 workers already exhibits a large variance at staleness 15.

The amplification of staleness from increasing number of Figure 7: The number of batches to reach 92% test accuracy using DNNs with varying numbers of hidden layers under 1 worker.

We consider several variants of SGD algorithms (a)-(e).

Note that with depth 0 the model reduces to MLR, which is convex.

The numbers are averaged over 5 randomized runs.

We omit the result whenever convergence is not achieved within the experiment horizon (77824 batches), such as SGD with momentum at depth 6 and s = 32.

Recurrent Neural Networks (RNNs) are widely used in recent natural language processing tasks.

We consider long short-term memory (LSTM) BID18 applied to the language modeling task, using a subset of Penn Treebank dataset (PTB) BID34 containing 5855 words.

The dataset is pre-processed by standard de-capitalization and tokenization.

We evaluate the impact of staleness for LSTM with 1 to 4 layers, with 256 neurons in each layer.

The maximum length for each sentence is 25.

Note that 4 layer LSTM is about 4x more model parameters than the 1 layer LSTM, which is the same ratio between ResNet32 and Resnet 8.

We use batch size 32 similar to other experiments.

We consider staleness s = 0, 4, 8, 16 on 8 workers.

The model quality is measured in perplexity.

FIG1 shows the number of batches needed to reach the desired model quality for RNNs with varying network depths.

We again observe that staleness impacts deeper network variants more than shallower counterparts, which is consistent with our observation in CNNs and DNNs.

<|TLDR|>

@highlight

Empirical and theoretical study of the effects of staleness in non-synchronous execution on machine learning algorithms.