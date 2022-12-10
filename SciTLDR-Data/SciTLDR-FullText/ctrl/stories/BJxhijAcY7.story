Training neural networks on large datasets can be accelerated by distributing the workload over a network of machines.

As datasets grow ever larger, networks of hundreds or thousands of machines become economically viable.

The time cost of communicating gradients limits the effectiveness of using such large machine counts, as may the increased chance of network faults.

We explore a particularly simple algorithm for robust, communication-efficient learning---signSGD.

Workers transmit only the sign of their gradient vector to a server, and the overall update is decided by a majority vote.

This algorithm uses 32x less communication per iteration than full-precision, distributed SGD.

Under natural conditions verified by experiment, we prove that signSGD converges in the large and mini-batch settings, establishing convergence for a parameter regime of Adam as a byproduct.

Aggregating sign gradients by majority vote means that no individual worker has too much power.

We prove that unlike SGD, majority vote is robust when up to 50% of workers behave adversarially.

The class of adversaries we consider includes as special cases those that invert or randomise their gradient estimate.

On the practical side, we built our distributed training system in Pytorch.

Benchmarking against the state of the art collective communications library (NCCL), our framework---with the parameter server housed entirely on one machine---led to a 25% reduction in time for training resnet50 on Imagenet when using 15 AWS p3.2xlarge machines.

The most powerful supercomputer in the world is currently a cluster of over 27,000 GPUs at Oak Ridge National Labs (TOP500, 2018).

Distributed algorithms designed for such large-scale systems typically involve both computation and communication: worker nodes compute intermediate results locally, before sharing them with their peers.

When devising new machine learning algorithms for distribution over networks of thousands of workers, we posit the following desiderata: D1 fast algorithmic convergence; D2 good generalisation performance;

D4 robustness to network faults.

When seeking an algorithm that satisfies all four desiderata D1-4, inevitably some tradeoff must be made.

Stochastic gradient descent (SGD) naturally satisfies D1-2, and this has buoyed recent advances in deep learning.

Yet when it comes to large neural network models with hundreds of millions of parameters, distributed SGD can suffer large communication overheads.

To make matters worse, any faulty SGD worker can corrupt the entire model at any time by sending an infinite gradient, meaning that SGD without modification is not robust.

A simple algorithm with aspirations towards all desiderata D1-4 is as follows: workers send the sign of their gradient up to the parameter server, which aggregates the signs and sends back only the majority decision.

We refer to this algorithm as SIGNSGD with majority vote.

All communication to and from the parameter server is compressed to one bit, so the algorithm certainly gives us D3.

What's more, in deep learning folklore sign based methods are known to perform well, indeed inspiring the popular RMSPROP and ADAM optimisers BID3 , giving hope for D1.

As far as robustness goes, aggregating gradients by a majority vote denies any individual worker too much power, suggesting it may be a natural way to achieve D4.In this work, we make the above aspirations rigorous.

Whilst D3 is immmediate, we provide the first convergence guarantees for SIGNSGD in the mini-batch setting, providing theoretical grounds for D1.

We show how theoretically the behaviour of SIGNSGD changes as gradients move from high to low signal-to-noise ratio.

We also extend the theory of majority vote to show that it achieves a notion of Byzantine fault tolerance.

A distributed algorithm is Byzantine fault tolerant BID5 if its convergence is robust when up to 50% of workers behave adversarially.

The class of adversaries we consider contains interesting special cases, such as robustness to a corrupted worker sending random bits, or a worker that inverts their gradient estimate.

Though our adversarial model is not the most general, it is interesting as a model of network faults, and so gives us D4.Next, we embark on a large-scale empirical validation of our theory.

We implement majority vote in the Pytorch deep learning framework, using CUDA kernels to bit pack sign tensors down to one bit.

Our results provide experimental evidence for D1-D4.

Comparing our framework to NCCL (the state of the art communications library), we were able to speed up Imagenet training by 25% when distributing over 7 to 15 AWS p3.2xlarge machines, albeit at a slight loss in generalisation.

Finally, in an interesting twist, the theoretical tools we develop may be brought to bear on a seemingly unrelated problem in the machine learning literature.

BID13 proved that the extremely popular ADAM optimiser in general does not converge in the mini-batch setting.

This result belies the success of the algorithm in a wide variety of practical applications.

SIGNSGD is equivalent to a special case of ADAM, and we establish the convergence rate of mini-batch SIGNSGD for a large class of practically realistic objectives.

Therefore, we expect that these tools should carry over to help understand the success modes of ADAM.

Our insight is that gradient noise distributions in practical problems are often unimodal and symmetric because of the Central Limit Theorem, yet BID13 's construction relies on bimodal noise distributions.

For decades, neural network researchers have adapted biologically inspired algorithms for efficient hardware implementation.

Hopfield (1982) , for example, considered taking the sign of the synaptic weights of his memory network for readier adaptation into integrated circuits.

This past decade, neural network research has focused on training feedforward networks by gradient descent (LeCun et al., 2015) .

It is natural to ask what practical efficiency may accompany simply taking the sign of the backpropagated gradient.

In this section, we explore related work pertaining to this question.

Deep learning: whilst stochastic gradient descent (SGD) is the workhorse of machine learning BID15 , algorithms like RMSPROP BID17 and ADAM (Kingma & Ba, 2015) are also extremely popular neural net optimisers.

These algorithms have their Algorithm 1 SIGNUM with majority vote, the proposed algorithm for distributed optimisation.

Good default settings for the tested machine learning problems are η = 0.0001 and β = 0.9, though tuning is recommended.

All operations on vectors are element-wise.

Setting β = 0 yields SIGNSGD.

Require: learning rate η > 0, momentum constant β ∈ [0, 1), weight decay λ ≥ 0, mini-batch size n, initial point x held by each of M workers, initial momentum v m ← 0 on m th worker repeat on m th worker DISPLAYFORM0 aggregate sign momenta push sign(V ) to each worker broadcast majority vote on every workerx ← x − η(sign(V ) + λx) update parameters until convergence roots in the RPROP optimiser BID14 , which is a sign-based method similar to SIGNSGD except for a component-wise adaptive learning rate.

Non-convex optimisation: parallel to (and oftentimes in isolation from) advances in deep learning practice, a sophisticated optimisation literature has developed.

BID10 proposed cubic regularisation as an algorithm that can escape saddle points and provide guaranteed convergence to local minima of non-convex functions.

This has been followed up by more recent works such as NATASHA BID2 ) that use other theoretical tricks to escape saddle points.

It is still unclear how relevant these works are to deep learning, since it is not clear to what extent saddle points are an obstacle in practical problems.

We avoid this issue altogether and satisfy ourselves with establishing convergence to critical points.

Gradient compression: prior work on gradient compression generally falls into two camps.

In the first camp, algorithms like QSGD BID0 , TERNGRAD BID20 and ATOMO use stochastic quantisation schemes to ensure that the compressed stochastic gradient remains an unbiased approximation to the true gradient.

These works are therefore able to bootstrap existing SGD convergence theory.

In the second camp, more heuristic algorithms like 1BITSGD BID16 and deep gradient compression (Lin et al., 2018) pay less attention to theoretical guarantees and focus more on practical performance.

These algorithms track quantisation errors and feed them back into subsequent updates.

The commonality between the two camps is an effort to, one way or another, correct for bias in the compression.

SIGNSGD with majority vote takes a different approach to these two existing camps.

In directly employing the sign of the stochastic gradient, the algorithm unabashedly uses a biased approximation of the stochastic gradient.

BID8 and BID4 provide theoretical and empirical evidence that signed gradient schemes can converge well in spite of their biased nature.

Their theory only applies in the large batch setting, meaning the theoretical results are less relevant to deep learning practice.

Still BID4 showed promising experimental results in the mini-batch setting.

An appealing feature of majority vote is that it naturally leads to compression in both directions of communication between workers and parameter server.

As far as we are aware, all existing gradient compression schemes lose compression before scattering results back to workers.

Byzantine fault tolerant optimisation: the problem of modifying SGD to make it Byzantine fault tolerant has recently attracted interest in the literature BID21 .

For example, BID5 proposed KRUM, which operates by detecting and excluding outliers in the gradient aggregation.

BID1 propose BYZANTINESGD which instead focuses on detecting and eliminating adversaries.

Clearly both these strategies incur overheads, and eliminating adversaries precludes the possibility that they might reform.

El BID9 point out that powerful adversaries may steer convergence to bad local minimisers.

We see majority vote as a natural way to protect against less malign faults such as network errors, and thus satisfy ourselves with convergence guarantees to critical points without placing guarantees on their quality.

: Gradient distributions for resnet18 on Cifar-10 at mini-batch size 128.

At the start of epochs 0, 1 and 5, we do a full pass over the data and collect the gradients for three randomly chosen weights (left, middle, right) .

In all cases the distribution is close to unimodal and symmetric.

We aim to develop an optimisation theory that is relevant for real problems in deep learning.

For this reason, we are careful about the assumptions we make.

For example, we do not assume convexity because neural network loss functions are typically not convex.

Though we allow our objective function to be non-convex, we insist on a lower bound to enable meaningful convergence results.

Assumption 1 (Lower bound).

For all x and some constant f * , we have objective value f (x) ≥ f * .Our next two assumptions of Lipschitz smoothness and bounded variance are standard in the stochastic optimisation literature (Allen-Zhu, 2017).

That said, we give them in a component-wise form.

This allows our convergence results to encode information not just about the total noise level and overall smoothness, but also about how these quantities are distributed across dimension.

Assumption 2 (Smooth).

Let g(x) denote the gradient of the objective f (.) evaluated at point x.

Then ∀x, y we require that for some non- DISPLAYFORM0 Assumption 3 (Variance bound).

Upon receiving query x ∈ R d , the stochastic gradient oracle gives us an independent, unbiased estimateg that has coordinate bounded variance: DISPLAYFORM1 for a vector of non-negative constants σ : DISPLAYFORM2 Our final assumption is non-standard.

We assume that the gradient noise is unimodal and symmetric.

Clearly, Gaussian noise is a special case.

Note that even for a moderate mini-batch size, we expect the central limit theorem to kick in rendering typical gradient noise distributions close to Gaussian.

See FIG2 for noise distributions measured whilst training resnet18 on Cifar-10.

Assumption 4 (Unimodal, symmetric gradient noise).

At any given point x, each component of the stochastic gradient vectorg(x) has a unimodal distribution that is also symmetric about the mean.

Showing how to work with this assumption is a key theoretical contribution of this work.

Combining Assumption 4 with an old tail bound of Gauss (1823) yields Lemma 1, which will be crucial for guaranteeing mini-batch convergence of SIGNSGD.

As will be explained in Section 3.3, this result also constitutes a convergence proof for a parameter regime of ADAM.

This suggests that Assumption 4 may more generally be a theoretical fix for Reddi et al. (2018) 's non-convergence proof of mini-batch ADAM, a fix which does not involve modifying the ADAM algorithm itself.

With our assumptions in place, we move on to presenting our theoretical results, which are all proved in Appendix C. Our first result establishes the mini-batch convergence behaviour of SIGNSGD.

We will first state the result and make some remarks.

We provide intuition for the proof in Section 3.3.Theorem 1 (Non-convex convergence rate of mini-batch SIGNSGD).

Run the following algorithm for K iterations under Assumptions 1 to 4: DISPLAYFORM0 Set the learning rate, η, and mini-batch size, n, as DISPLAYFORM1 Let H k be the set of gradient components at step k with large signal-to-noise ratio DISPLAYFORM2 .

We refer to DISPLAYFORM3 as the 'critical SNR'.

Then we have DISPLAYFORM4 where N = K is the total number of stochastic gradient calls up to step K. Remark 2: the gradient appears as a mixed norm: an 1 norm for high SNR components, and a weighted 2 norm for low SNR compoenents.

Remark 3: we wish to understand the dimension dependence of our bound.

We may simplify matters by assuming that, during the entire course of optimisation, every gradient component lies in the low SNR regime.

FIG3 shows that this is almost true when training a resnet18 model.

In this limit, the bound becomes: DISPLAYFORM5 Further assume that we are in a well-conditioned setting, meaning that the variance is distributed uniformly across dimension (σ DISPLAYFORM6 , and every weight has the same smoothness constant (L i = L).

σ 2 is the total variance bound, and L is the conventional Lipschitz smoothness.

These are the quantities which appear in the standard analysis of SGD.

Then we get DISPLAYFORM7 The factors of dimension d have conveniently cancelled.

This illustrates that there are problem geometries where mini-batch SIGNSGD does not pick up an unfavourable dimension dependence.

Intuitively, the convergence analysis of SIGNSGD depends on the probability that a given bit of the sign stochastic gradient vector is incorrect, or P[sign(g i ) = sign(g i )].

Lemma 1 provides a bound on this quantity under Assumption 4 (unimodal symmetric gradient noise).Lemma 1 BID4 ).

Letg i be an unbiased stochastic approximation to gradient component g i , with variance bounded by σ 2 i .

Further assume that the noise distribution is unimodal and symmetric.

Define signal-to-noise ratio S i := |gi| σi .

Then we have that DISPLAYFORM0 otherwise which is in all cases less than or equal to DISPLAYFORM1 The bound characterises how the failure probability of a sign bit depends on the signal-to-noise ratio (SNR) of that gradient component.

Intuitively as the SNR decreases, the quality of the sign estimate should degrade.

The bound is important since it tells us that, under conditions of unimodal symmetric gradient noise, even at extremely low SNR we still have that DISPLAYFORM2 2 .

This means that even when the gradient is very small compared to the noise, the sign stochastic gradient still tells us, on average, useful information about the true gradient direction, allowing us to guarantee convergence as in Theorem 1.Without Assumption 4, the mini-batch algorithm may not converge.

This is best appreciated with a simple example.

Consider a stochastic gradient componentg with bimodal noise: g = 50 with probability 0.1; −1 with probability 0.9.The true gradient g = E[g] = 4.1 is positive.

But the sign gradient sign(g) is negative with probability 0.9.

Therefore SIGNSGD will tend to move in the wrong direction for this noise distribution.

Note that SIGNSGD is a special case of the ADAM algorithm BID3 .

To see this, set β 1 = β 2 = = 0 in ADAM, and the ADAM update becomes: DISPLAYFORM3 This correspondence suggests that Assumption 4 should be useful for obtaining mini-batch convergence guarantees for ADAM.

Note that when BID13 construct toy divergence examples for ADAM, they rely on bimodal noise distributions which violate Assumption 4.We conclude this section by noting that without Assumption 4, SIGNSGD can still be guaranteed to converge.

The trick is to use a "large" batch size that grows with the number of iterations.

This will ensure that the algorithm stays in the high SNR regime where the failure probability of the sign bit is low.

This is the approach taken by both BID8 and BID4 .

We will now study SIGNSGD's robustness when distributed by majority vote.

We model adversaries as machines that may manipulate their stochastic gradient as follows.

Definition 1 (Blind multiplicative adversary).

A blind multiplicative adversary may manipulate their stochastic gradient estimateg t at iteration t by element-wise multiplyingg t with any vector v t of their choice.

The vector v t must be chosen before observingg t , so the adversary is 'blind'.

Some interesting members of this class are:(i) adversaries that arbitrarily rescale their stochastic gradient estimate;(ii) adversaries that randomise the sign of each coordinate of the stochastic gradient;(iii) adversaries that invert their stochastic gradient estimate.

SGD is certainly not robust to rescaling since an adversary could set the gradient to infinity and corrupt the entire model.

Our algorithm, on the other hand, is robust to all adversaries in this class.

For ease of analysis, here we derive large batch results.

We make sure to give results in terms of sample complexity N (and not iteration number K) to enable fair comparison with other algorithms.

Theorem 2 (Non-convex convergence rate of majority vote with adversarial workers).

Run algorithm 1 for K iterations under Assumptions 1 to 4.

Switch off momentum and weight decay (β = λ = 0).

Set the learning rate, η, and mini-batch size, n, for each worker as DISPLAYFORM0 Assume that a fraction α < 1 2 of the M workers behave adversarially according to Definition 1.

Then majority vote converges at rate: DISPLAYFORM1 where N = K 2 is the total number of stochastic gradient calls per worker up to step K.The result is intuitive: provided there are more machines sending honest gradients than adversarial gradients, we expect that the majority vote should come out correct on average.

Remark 1: if we switch off adversaries by setting the proportion of adversaries α = 0, this result reduces to Theorem 2 in BID4 .

In this case, we note the DISPLAYFORM2 variance reduction that majority vote obtains by distributing over M machines, similar to distributed SGD.Remark 2: the convergence rate degrades as we ramp up α from 0 to 1 2 .

Remark 3: from an optimisation theory perspective, the large batch size is an advantage.

This is because when using a large batch size, fewer iterations and rounds of communication are theoretically needed to reach a desired accuracy, since only √ N iterations are needed to reach N samples.

But from a practical perspective, workers may be unable to handle such a large batch size in a timely manner.

It should be possible to extend the result to the mini-batch setting by combining the techniques of Theorems 1 and 2, but we leave this for future work.

For our experiments, we distributed SIGNUM (Algorithm 1) by majority vote.

SIGNUM is the momentum counterpart of SIGNSGD, where each worker maintains a momentum and transmits the sign momentum to the parameter server at each step.

The addition of momentum to SIGNSGD is proposed and studied in BID3 BID4 .We built SIGNUM with majority vote in the Pytorch deep learning framework BID11 using the Gloo (2018) communication library.

Unfortunately Pytorch and Gloo do not natively support 1-bit tensors, therefore we wrote our own compression code to bit-pack a sign tensor down to an efficient 1-bit representation.

We obtained a performance boost by fusing together smaller tensors, which saved on compression and communication costs.

We train resnet50 on Imagenet distributed over 7 to 15 AWS p3.2xlarge machines.

Top: increasing the number of workers participating in the majority vote shows a similar convergence speedup to distributed SGD.

But in terms of wall-clock time, majority vote training is roughly 25% faster for the same number of epochs.

Bottom: in terms of generalisation accuracy, majority vote shows a slight degradation compared to SGD.

Perhaps a better regularisation scheme can fix this.

Figure 6: Training QRNN across three p3.16xlarge machines on WikiText-103.

Each machine uses a batch size of 240.

For ADAM, the gradient is aggregated with NCCL.

SIGNUM with majority vote shows some degradation compared to ADAM, although an epoch is completed roughly three times faster.

This means that after 2 hours of training, SIGNUM attains a similar perplexity to ADAM.

Increasing the per-worker batch size improved SIGNUM's performance (see Appendix A), and increasing it beyond 240 may further improve SIGNUM's performance.

Note: the test perplexity beats training perplexity because dropout was applied during training but not testing.

Figure 7: Left: comparing convergence of majority vote to QSGD BID0 .

resnet18 is trained on Cifar-10 across M = 3 machines, each at bach size 128.

1-bit QSGD stochastically snaps gradient components to {0, ±1}. 2-way refers to the compression function Q(.) being applied in both directions of communication: machine i sends Q(g i ) to the server and gets Q( We used majority vote to train resnet50 distributed across 7 AWS p3.2xlarge machines.

Adversaries invert their sign stochastic gradient.

Left: all experiments are run at identical hyperparameter settings, with weight decay switched off for simplicity.

The network still learns even at 43% adversarial.

Right: at 43% adversarial, learning became slightly unstable.

We decreased the learning rate for this setting, and learning stabilised.

BID5 .

We train resnet18 on Cifar-10 across 7 workers, each at batch size 64.

Momentum and weight decay are switched off for simplicity, and for majority vote we divide the learning rate by 10 at epoch 100.

Negative adversaries multiply their stochastic gradient estimate by −10.

Random adversaries multiply their stochastic gradient estimate by 10 and then randomise the sign of each coordinate.

For MULTI-KRUM, we use the maximum allowed security level of f = 2.

Notice that MULTI-KRUM fails catastrophically once the number of adversaries exceeds the security level, whereas majority vote fails more gracefully.

We test against SGD distributed using the state of the art NCCL (2018) communication library.

NCCL provides an efficient implementation of allreduce.

Our framework is often 4× faster in communication (including the cost of compression) than NCCL, as can be seen in Figure 4 .

Further code optimisation should bring the speedup closer to the ideal 32×.

We first benchmark majority vote on the Imagenet dataset.

We train a resnet50 model and disitribute learning over 7 to 15 AWS p3.2xlarge machines.

These machines each contain one Nvidia Tesla V100 GPU, and AWS lists the connection speed between machines as "up to 10 Gbps".

Results are plotted in Figure 5 .

Per epoch, distributing by majority vote is able to attain a similar speedup to distributed SGD.

But per hour majority vote is able to process more epochs than NCCL, meaning it can complete the 80 epoch training job roughly 25% faster.

In terms of overall generalisation, majority vote reaches a slightly degraded test set accuracy.

We hypothesise that this may be fixed by inventing a better regularisation scheme or tuning momentum, which we did not do.

In Figure 6 we compare majority vote to ADAM (distributed by NCCL) for training QRNN (Bradbury et al., 2017) on WikiText-103.

Majority vote completes an epoch roughly 3 times faster than ADAM, but it reaches a degraded accuracy so that the overall test perplexity after 2 hours ends up being similar.

In Figure 7 we show that majority vote has superior convergence to the 'theory' version of QSGD that BID0 develop.

Convergence is similar for the 'max' version that BID0 use in their experiments.

Additional results are given in Appendix A.

In this section we test the robustness of SIGNUM with majority vote to Byzantine faults.

Again we run tests on the Imagenet dataset, training resnet50 across 7 AWS p3.2xlarge machines.

Our adversarial workers take the sign of their stochastic gradient calculation, but send the negation to the parameter server.

Our results are plotted in FIG8 .

In the left hand plot, all experiments were carried out using hyperparameters tuned for the 0% adversarial case.

Weight decay was not used in these experiments to simplify matters.

We see that learning is tolerant of up to 43% (3 out of 7) machines behaving adversarially.

The 43% adversarial case was slightly unstable FIG8 , but re-tuning the learning rate for this specific case stabilised learning FIG8 .In FIG9 we compare majority vote to MULTI-KRUM BID5 ) with a security level of f = 2.

When the number of adversaries exceeds f , MULTI-KRUM fails catastrophically in our experiments, whereas SIGNSGD fails more gracefully.

Note that MULTI-KRUM requires 2f + 2 < M , therefore f = 2 is the maximum possible security level for these experiments with M = 7 workers.

We have analysed the theoretical and empirical properties of a very simple algorithm for distributed, stochastic optimisation.

We have shown that SIGNSGD with majority vote aggregation is robust and communication efficient, whilst its per-iteration convergence rate is competitive with SGD for training large-scale convolutional neural nets on image datasets.

We believe that it is important to understand this simple algorithm before going on to devise more complex learning algorithms.

An important takeaway from our theory is that mini-batch SIGNSGD should converge if the gradient noise is Gaussian.

This means that the performance of SIGNSGD may be improved by increasing the per-worker mini-batch size, since this should make the noise 'more Gaussian' according to the Central Limit Theorem.

We will now give some possible directions for future work.

Our implementation of majority vote may be further optimised by breaking up the parameter server and distributing it across machines.

This would prevent a single machine from becoming a communication bottleneck as in our experiments.

Though our framework speeds up Imagenet training, we still have a test set gap.

Future work could attempt to devise new regularisation schemes for signed updates to close this gap.

Promising future work could also explore the link between SIGNSGD and model compression.

In this section, we perform theoretical calculations of the number of bits sent per iteration in distributed training.

We compare SIGNSGD using majority vote aggregation to two forms of QSGD BID0 .

These calculations give the numbers in the table in Figure 7 .The communication cost of SIGNSGD with majority vote is trivially 2M d bits per iterations, since at each iteration M machines send d-dimensional sign vectors up to the server, and the server sends back one d-dimensional sign vector to all M machines.

There are two variants of QSGD given in BID0 .

The first we refer to as L2 QSGD which is the version developed in the theory section of BID0 .

The second we refer to as max QSGD which is the version actually used in their experiments.

For each version we compute the number of bits sent for the highest compression version of the algorithm, which is a ternary quantisation (snapping gradient components into {0, ±1}).

We refer to this as 1-bit QSGD.

The higher precision versions of QSGD will send more bits per iteration.1-bit L2 QSGD takes a gradient vector g and snaps i th coordinate g i to sign(g i ) with probability DISPLAYFORM0 and sets it to zero otherwise.

Therefore the expected number of bits set to ±1 is bounded by DISPLAYFORM1 To send a vector compressed in this way, for each non-zero component 1 bit is needed to send the sign and log d bits are needed to send the index.

Therefore sending a vector compressed by 1-bit L2 QSGD requires at most DISPLAYFORM2 In the experiments in Figure 7 we see that the '2-way' version of 1-bit L2 QSGD (which recompresses the aggregated compressed gradients) converges very poorly.

Therefore it makes sense to use the 1-way version where the aggregated compressed gradient is not recompressed.

A sensible way to enact this is to have each of the M workers broadcast their compressed gradient vector to all other workers.

This has a cost of (M − 1) The final algorithm to characterise is 1-bit max QSGD.

1-bit max QSGD takes a gradient vector g and snaps i th coordinate g i to sign(g i ) with probability |gi| g ∞ and sets it to zero otherwise.

As noted in BID0 , there are no sparsity guarantees for this algorithm, so compression will generally be much lower than for 1-bit L2 QSGD.

DISPLAYFORM3 It is easy to see that 1-bit max QSGD requires no more than O(d) bits to compress a d-dimensional vector, since 2d bits can always store d numbers in {0, ±1}. To see that we can't generally do better than O(d) bits, notice that 1-bit max QSGD leaves sign vectors invariant, and thus the compressed form of a sign vector requires exactly d bits.

The natural way to enact 1-bit max QSGD is with a two-way compression where the M workers each send an O(d)-bit compressed gradient up to the server, and the server sends back an O(d)-bit compressed aggregated result back to the M workers.

This gives a number of bits sent per iteration of O(M d).For very sparse vectors 1-bit max QSGD will compress much better than indicated above.

For a vector g with a single non-zero entry, 1-bit max QSGD will set this entry to 1 and keep the rest zero, thus requiring only log d bits to send the index of the non-zero entry.

But it is not clear whether these extremely sparse vectors appear in deep learning problems.

In the experiments in Figure 7 , 1-bit max QSGD led to compressed vectors that were 5× more compressed than SIGNSGD-in our experimental setting this additional improvement turned out to be small relative to the time cost of backpropagation.

Lemma 1 BID4 ).

Letg i be an unbiased stochastic approximation to gradient component g i , with variance bounded by σ 2 i .

Further assume that the noise distribution is unimodal and symmetric.

Define signal-to-noise ratio S i := |gi| σi .

Then we have that DISPLAYFORM0 otherwise which is in all cases less than or equal to 1 2 .Proof.

Recall Gauss' inequality for unimodal random variable X with mode ν and expected squared deviation from the mode τ 2 (Gauss, 1823; BID12 : DISPLAYFORM1 otherwise By the symmetry assumption, the mode is equal to the mean, so we replace mean µ = ν and variance DISPLAYFORM2 otherwise Without loss of generality assume that g i is negative.

Then applying symmetry followed by Gauss, the failure probability for the sign bit satisfies: DISPLAYFORM3

Theorem 1 (Non-convex convergence rate of mini-batch SIGNSGD).

Run the following algorithm for K iterations under Assumptions 1 to 4: DISPLAYFORM0 Set the learning rate, η, and mini-batch size, n, as DISPLAYFORM1 Let H k be the set of gradient components at step k with large signal-to-noise ratio DISPLAYFORM2 .

We refer to DISPLAYFORM3 as the 'critical SNR'.

Then we have DISPLAYFORM4 where N = K is the total number of stochastic gradient calls up to step K.Proof.

First let's bound the improvement of the objective during a single step of the algorithm for one instantiation of the noise.

I[.] is the indicator function, g k,i denotes the i th component of the true gradient g(x k ) andg k is a stochastic sample obeying Assumption 3.First take Assumption 2, plug in the algorithmic step, and decompose the improvement to expose the stochasticity-induced error: DISPLAYFORM5 Next we find the expected improvement at time k + 1 conditioned on the previous iterate.

DISPLAYFORM6 By Assumption 4 and Lemma 1 we have the following bound on the failure probability of the sign: DISPLAYFORM7 otherwise Substituting this in, we get that DISPLAYFORM8 Interestingly a mixture between an 1 and a variance weighted 2 norm has appeared.

Now substitute in the learning rate schedule, and we get: DISPLAYFORM9 Now extend the expectation over the randomness in the trajectory and telescope over the iterations: DISPLAYFORM10 Finally, rearrange and substitute in N = K to yield the bound DISPLAYFORM11

Theorem 2 (Non-convex convergence rate of majority vote with adversarial workers).

Run algorithm 1 for K iterations under Assumptions 1 to 4.

Switch off momentum and weight decay (β = λ = 0).

Set the learning rate, η, and mini-batch size, n, for each worker as DISPLAYFORM0 Assume that a fraction α < 1 2 of the M workers behave adversarially according to Definition 1.

Then majority vote converges at rate: DISPLAYFORM1 where N = K 2 is the total number of stochastic gradient calls per worker up to step K.Proof.

We need to bound the failure probability of the vote.

We can then use this bound to derive a convergence rate.

We will begin by showing this bound is worst when the adversary inverts the signs of the sign stochastic gradient.

Given an adversary from the class of blind multiplicative adversaries (Definition 1), the adversary may manipulate their stochastic gradient estimateg t into the form v t ⊗g t .

Here v t is a vector of the adversary's choosing, and ⊗ denotes element-wise multiplication.

The sign of this quantity obeys:sign(v t ⊗g t ) = sign(v t ) ⊗ sign(g t ).Therefore, the only thing that matters is the sign of v t , and rescaling attacks are immediately nullified.

For each component of the stochastic gradient, the adversary must decide (without observing g t , since the adversary is blind) whether or not they would like to invert the sign of that component.

We will now show that the failure probability of the vote is always larger when the adversary decides to invert (by setting every component of sign(v t ) to −1).

Our analysis will then proceed under this worst case.

For a gradient component with true value g, let random variable Z ∈ [0, M ] denote the number of correct sign bits received by the parameter server.

For a given adversary, we may decompose Z into the contribution from that adversary and a residual term X from the remaining workers (both regular and adversarial):Z(sign(v)) = X + I[sign(v)sign(g) = sign(g)],whereg is the adversary's stochastic gradient estimate for that component, v is the adversary's chosen scalar for that component, and I is the 0-1 indicator function.

We are considering Z to be a function of sign(v).But by Assumption 4 and Lemma 1, we see that I[+1 × sign(g) = sign(g)] is a Bernoulli random variable with success probability p ≥ 1 2 .

On the other hand, I[−1×sign(g) = sign(g)] is a Bernoulli random variable with success probability q = 1 − p ≤ 1 2 .

<|TLDR|>

@highlight

Workers send gradient signs to the server, and the update is decided by majority vote. We show that this algorithm is convergent, communication efficient and fault tolerant, both in theory and in practice.

@highlight

Presents a distributed implementation of signSGD with majority vote as aggregation.