The question why deep learning algorithms generalize so well has attracted increasing research interest.

However, most of the well-established approaches, such as hypothesis capacity, stability or sparseness, have not provided complete explanations (Zhang et al., 2016; Kawaguchi et al., 2017).

In this work, we focus on the robustness approach (Xu & Mannor, 2012), i.e., if the error of a hypothesis will not change much due to perturbations of its training examples, then it will also generalize well.

As most deep learning algorithms are stochastic (e.g., Stochastic Gradient Descent, Dropout, and Bayes-by-backprop), we revisit the robustness arguments of Xu & Mannor, and introduce a new approach – ensemble robustness – that concerns the robustness of a population of hypotheses.

Through the lens of ensemble robustness, we reveal that a stochastic learning algorithm can generalize well as long as its sensitiveness to adversarial perturbations is bounded in average over training examples.

Moreover, an algorithm may be sensitive to some adversarial examples (Goodfellow et al., 2015) but still generalize well.

To support our claims, we provide extensive simulations for different deep learning algorithms and different network architectures exhibiting a strong correlation between ensemble robustness and the ability to generalize.

Deep Neural Networks (DNNs) have been successfully applied in many artificial intelligence tasks, providing state-of-the-art performance and a remarkably small generalization error.

On the other hand, DNNs often have far more trainable model parameters than the number of samples they are trained on and were shown to have a large enough capacity to memorize the training data BID26 .

The findings of Zhang et al. suggest that classical explanations for generalization cannot be applied directly to DNNs and motivated researchers to look for new complexity measures and explanations for the generalization deep neural networks BID2 BID16 BID0 BID13 .

However, in this work, we focus on a different approach to study generalization of DNNs, i.e., the connection between the robustness of a deep learning algorithm and its generalization performance.

Xu & Mannor have shown that if an algorithm is robust (i.e., its empirical loss does not change dramatically for perturbed samples), its generalization performance can also be guaranteed.

However, in the context of DNNs, practitioners observe contradicting evidence between these two attributes.

On the one hand, DNNs generalize well, and on the other, they are fragile to adversarial perturbation on the inputs BID21 BID8 .

Nevertheless, algorithms that try to improve the robustness of learning algorithms have been shown to improve the generalization of deep neural networks.

Two examples are adversarial training, i.e., generating adversarial examples and training on them BID21 BID8 BID18 , and Parseval regularization BID5 , i.e., minimizing the Lifshitz constant of the network to guarantee low robustness.

While these meth-ods minimize the robustness implicitly, their empirical success Indicates a connection between the robustness of an algorithm and its ability to generalize.

To solve this contradiction, we revisit the robustness argument in BID23 and present ensemble robustness, to characterize the generalization performance of deep learning algorithms.

Our proposed approach is not intended to give tight bounds for general deep learning algorithms, but rather to pave the way for addressing the question: how can deep learning perform so well while being fragile to adversarial examples?

Answering this question is difficult, yet we present evidence in both theory and simulation suggesting that ensemble robustness explains the generalization performance of deep learning algorithms.

Ensemble robustness concerns the fact that a randomized algorithm (e.g., Stochastic Gradient Descent (SGD), Dropout BID19 , Bayes-by-backprop BID3 , etc.) produces a distribution of hypotheses instead of a deterministic one.

Therefore, ensemble robustness takes into consideration robustness of the population of the hypotheses: even though some hypotheses may be sensitive to perturbation on inputs, an algorithm can still generalize well as long as most of the hypotheses sampled from the distribution are robust on average.

BID13 took a different approach and claimed that deep neural networks could generalize well despite nonrobustness.

However, our definition of ensemble robustness together with our empirical findings suggest that deep learning methods are typically robust although being fragile to adversarial examples.

Through ensemble robustness, we prove that the following holds with a high probability: randomized learning algorithms can generalize well as long as its output hypothesis has bounded sensitiveness to perturbation in average (see Theorem 1).

Specified for deep learning algorithms, we reveal that if hypotheses from different runs of a deep learning method perform consistently well in terms of robustness, the performance of such deep learning method can be confidently expected.

Moreover, each hypothesis may be sensitive to some adversarial examples as long as it is robust on average.

Although ensemble robustness may be difficult to compute analytically, we demonstrate an empirical estimate of ensemble robustness and investigate the role of ensemble robustness via extensive simulations.

The results provide supporting evidence for our claim: ensemble robustness consistently explains the generalization performance of deep neural networks.

Furthermore, ensemble robustness is measured solely on training data, potentially allowing one to use the testing examples for training and selecting the best model based on its ensemble robustness.

BID23 proposed to consider model robustness for estimating generalization performance for deterministic algorithms, such as for SVM BID25 and Lasso BID24 .

They suggest using robust optimization to construct learning algorithms, i.e., minimizing the empirical loss concerning the adversarial perturbed training examples.

Introducing stochasticity to deep learning algorithms has achieved great success in practice and also receives theoretical investigation.

BID10 analyzed the stability property of SGD methods, and Dropout BID19 was introduced as a way to control over-fitting by randomly omitting subsets of features at each iteration of a training procedure.

Different explanations for the empirical success of dropout have been proposed, including, avoiding over-fitting as a regularization method BID1 BID22 BID12 and explaining dropout as a Bayesian approximation for a Gaussian process BID7 .

Different from those works, this work will extend the results in BID23 to randomized algorithms, to analyze them from an ensemble robustness perspective.

Robustness and ensemble robustness share some similarities with stability, a related yet different property of learning algorithms that also guarantees generalization.

An algorithm is stable if it produces an output hypothesis that is not sensitive to the sampling of the empirical data set.

In more detail, if a training example is replaced with another example from the same distribution, the training error will not change much; see BID4 for more details, and BID6 for a discussion on randomized algorithms.

We emphasize that robustness and stability are different properties; robustness concerns global modifications of the training data while stability is more local in that sense.

Moreover, robustness concerns attributes of a single hypothesis, while stability concerns two (one for the original data set and one for the modified one).

Finally, a learning algorithm may be both stable and robust, e.g., SVM BID25 , or robust but not stable, e.g., Lasso Regression BID24 .Adversarial examples for deep neural networks were first introduced in BID21 , while some recent works propose to utilize them as a regularization technique for training deep models BID8 BID9 BID18 .

However, all of those works attempt to find the "worst case" examples in a local neighborhood of the original training data and are not focused on measuring the global robustness of an algorithm nor on studying the connection between robustness and generalization.

In this work, we investigate the generalization property of stochastic learning algorithms in deep neural networks, by establishing their PAC bounds.

In this section, we provide some preliminary facts that are necessary for developing the approach of ensemble robustness.

After introducing the problem setup we are interested in, we in particular highlight the inherent randomness of deep learning algorithms and give a formal description of randomized learning algorithms.

Then, we briefly review the relationship between robustness and generalization performance established in BID23 .Problem setup We now introduce the learning setup for deep neural networks, which follows a standard one for supervised learning.

More concretely, we have Z and H as the sample set and the hypothesis set respectively.

The training sample set s = {s 1 , . . . , s n } consists of n i.i.d.

samples generated by an unknown distribution µ, and the target of learning is to obtain a neural network that minimizes expected classification error over the i.i.d.

samples from µ. Throughout the paper, we consider the training set s with a fixed size of n.

We denote the learning algorithm as A, which is a mapping from Z n to H. We use A : s → h s to denote the learned hypothesis given the training set s.

We consider the loss function (h, z) whose value is nonnegative and upper bounded by M .

Let L(·) and emp (·) denote the expected error and the training error for a learned hypothesis h s , i.e., DISPLAYFORM0 We are going to characterize the generalization error |L(h s )− emp (h s )| of deep learning algorithms in the following section.

Randomized algorithms Most of modern deep learning algorithms are in essence randomized ones, which map a training set s to a distribution of hypotheses ∆(H) instead of a single hypothesis.

For example, running a deep learning algorithm A with dropout for multiple times will produce different hypotheses which can be deemed as samples from the distribution ∆(H).

Therefore, before proceeding to analyze the performance of deep learning, we provide a formal definition of randomized learning algorithms here.

Definition 1 (Randomized Algorithms).

A randomized learning algorithm A is a function from Z n to a set of distributions of hypotheses ∆(H), which outputs a hypothesis h s ∼ ∆(H) with a probability π s (h).When learning with a randomized algorithm, the target is to minimize the expected empirical loss for a specific output hypothesis h s , similar to the ones in (1).

Here is the loss incurred by a specific output hypothesis by one instantiation of the randomized algorithm A.Examples of the internal randomness of a deep learning algorithm A include dropout rate (the parameter for a Bernoulli distribution for randomly masking certain neurons), random shuffle among training samples in SGD, the initialization of weights for different layers, to name a few.

BID23 established the relation between algorithmic robustness and generalization for the first time.

An algorithm is robust if the following holds: if two samples are close to each other, their associated losses are also close.

For being self-contained, we here briefly review the algorithmic robustness and its induced generalization guarantee.

Definition 2 (Robustness, BID23 DISPLAYFORM0 , such that the following holds for all s ∈ Z n : DISPLAYFORM1 Based on the above robustness property of algorithms, Xu et al. BID23 prove that a robust algorithm also generalizes well.

Motivated by their results, Shaham et al. BID18 proposed adversarial training algorithm to minimize the empirical loss over synthesized adversarial examples.

However, those results cannot be applied for characterizing the performance of modern deep learning models well.

To explain the proper performance of deep learning, one needs to understand the internal randomness of deep learning algorithms and the population performance of the multiple possible hypotheses.

Intuitively, a single output hypothesis cannot be robust to adversarial perturbation on training samples and the deterministic robustness argument in BID23 ) cannot be applied here.

Fortunately, deep learning algorithms output the hypothesis sampled from a distribution of hypotheses.

Therefore, even if some samples are not "nice" for one specific hypothesis, they aren't likely to fail most of the hypothesis from the produced distribution.

Thus, deep learning algorithms generalize well.

Such intuition motivates us to introduce the concept of ensemble robustness that is defined over the distribution of output hypotheses of a deep learning algorithm.

, such that the following holds for all s ∈ Z n : DISPLAYFORM0 Here the expectation is taken w.r.t.

the internal randomness of the algorithm A.An algorithm with strong ensemble robustness can provide good generalization performance in expectation w.r.t.

the generated hypothesis, as stated in the following theorem.

We note that the proofs for all the theorems that we present in this section can be found supplementary material.

Also, the supplementary material holds an additional proof for the special case of Dropout.

Theorem 1.

Let A be a randomized algorithm with (K,¯ (n)) ensemble robustness over the training set s, with |s| = n. Let ∆(H) ← A : s denote the output hypothesis distribution of A. Then for any δ > 0, with probability at least 1 − δ with respect to the random draw of the s and h ∼ ∆(H), the following holds: DISPLAYFORM1 Note that in the above theorem, we hide the dependency of the generalization bound on K in ensemble robustness measure¯ (n).

Generally, there is a trade-off between¯ (n) and K, the larger K is, the smaller¯ (n) is due to the finer partition.

This tradeoff is more evident in the bound of Theorem 2, see also BID23 .

Studying the asymptotics of¯ (n) is hard for general algorithms and deep networks, yet, it can be done for simpler learning algorithms.

For example, for linear SVM, (n) is equivalent to the covering number BID25 .Due to space limitations, all the technical lemmas and details of the proofs throughout the paper are deferred to supplementary material.

Theorem 1 leads to the following corollary which gives a way to minimize expected loss directly.

Corollary 1.

Let A be a randomized algorithm with (K,¯ (n)) ensemble robustness.

Let C 1 , . . .

, C K be a partition of Z, and write z 1 ∼ z 2 if z 1 , z 2 fall into the same C k .

If the training sample s is generated by i.i.d.

draws from µ, then with probability at least 1 − δ, the following holds over h ∈ H DISPLAYFORM2 Corollary 1 suggests that one can minimize the expected error of a deep learning algorithm effectively through minimizing the empirical error over the training samples s i perturbed in an adversarial way.

In fact, such an adversarial training strategy has been exploited in BID8 BID18 .Theorem 2.

Let A be a randomized algorithm with (K,¯ (n)) ensemble robustness over the training set s, where |s| = n. Let ∆(H) denote the output hypothesis distribution of the algorithm A on the training set s. Suppose following variance bound holds: DISPLAYFORM3 Then for any δ > 0, with probability at least 1 − δ with respect to the random draw of the s and h ∼ ∆(H), we have DISPLAYFORM4 Theorem 2 implies that Ensemble robustness is a "weaker" requirement for the model compared with Robustness proposed in BID23 .

To see this, consider the trade-off between the expectation and variance of ensemble robustness on two extreme examples.

When α = 0, we do not allow any variance in the output of the algorithm A. Thus, A reduces to a deterministic one.

To achieve the above upper bound, it is required that the output hypothesis satisfies max z∈Ci | (h, s i )− (h, z)| ≤ (n).

However, due to the intriguing property of deep neural networks BID21 , the deterministic model robustness measure (n) (ref.

Definition 2) is usually large.

In contrast, when the hypotheses variance α can be large enough, there are multiple possible output hypotheses from the distribution ∆(H).

We fix the partition of Z as C 1 , . . .

, C K .

Then, DISPLAYFORM5 Therefore, allowing certain variance on produced hypotheses, a randomized algorithm can tolerate the non-robustness of some hypotheses to certain samples.

As long as the ensemble robustness is small, the algorithm can still perform well.

Indeed, in the following section we demonstrate through simulations that generalization of deep learning models is more correlated with ensemble robustness than robustness.

This section is devoted to simulations for quantitatively and qualitatively demonstrating how ensemble robustness of a deep learning method explains its performance.

We first introduce our experiment settings and implementation details.

Data sets We conduct simulations on two benchmarks.

MNIST, a dataset of handwritten digit images (28x28) with 50,000 training samples and 10,000 test samples BID14 .

NotM-NIST 1 , a "mnist like database" containing font glyphs for the letters A through J (10 classes).

The training set contains 367,440 samples and 18,724 testing examples.

The images (for both data sets) were scaled such that each pixel is in the range [0, 1].

We note that we did not use the cross-validation data.

Network architecture and parameter setting Without explicit explanation, we use multi-layer perceptrons throughout the simulations.

All networks we examined are composed of three fully connected layers, each of which is followed by a rectified linear unit on top.

The output of the last fully-connected layer is fed to a 10-way softmax.

To avoid the bias brought by specific network architecture on our observations, we sample at random the number of units in each layer (uniformly over {400, 800, 1200} units) and the learning rate (uniformly over [0.005, 0.05] for SGD, and uniformly over [0.05, 0.5] for Bayes-by-backprop).

Finally, we used a mini-batch of 128 training examples at a time.

Compared algorithms We evaluate and compare ensemble robustness as well as the generalization performance for following 4 deep learning algorithms.

(1) Explicit ensembles, i.e., using a stochastic algorithm to train different members of the ensemble by running the algorithm multiple times with different seeds.

In practice, this was implemented using SGD as the stochastic algorithm, trained to minimize the cross-entropy loss.

(2) Implicit ensembles, i.e., learning a probability distribution on the weights of a neural network and sampling ensemble members from it.

This was implemented with the Bayes-by-backprop BID3 algorithm, a recent approach for training Bayesian Neural Networks.

It uses backpropagation to learn a probability distribution on the weights of a neural network by minimizing the expected lower bound on the marginal likelihood (or the variational free energy).

Methods 3 and 4 correspond for adding adversarial training BID21 BID8 BID18 to the ensemble methods, where the magnitude of perturbation is measured by its 2 norm and is sampled uniformly over {0.1, 0.3, 0.5} to avoid sampling bias.

From now on, a specific configuration will refer to a unique set of these parameters (algorithm type, network width, learning rate and perturbation norm).

We now present simulations that empirically validate Theorem 1, i.e., that the ensemble robustness of a DNN (measured on the training set) is highly correlated with its generalization performance.

But empirically evaluating ensemble robustness of deep neural nets is hard for two reasons.

.

To deal with this challenge, we use adversarial examples, and define K = n partitions such implicitly, such that each partition contains a small 2 ball around each training example.

We then approximate the loss change in each partition using the adversarial example, i.e., approximating the maximal loss in the partition using the adversarial example.

While this approximation is loose, we will soon show that empirically, it is correlated with generalization.

We emphasize that under this partition, there is no violation of the i.i.d assumption for general stochastic algorithms, but it is violated in the case of adversarial training (since the adversarial examples used for training are not sampled i.i.d).

Despite the latter observation, we measured even stronger correlation for these algorithms.

Second, ensemble robustness involves taking an expectation over all the possible output hypothesis.

Hence it is computationally intractable to measure ensemble robustness for deep learning algorithms exactly.

In this simulation, we take the empirical average of robustness to adversarial perturbation from 5 different hypotheses of the same learning algorithm as its ensemble robustness.

In the case of the SGD variants, for each configuration, we collect an ensemble of output hypotheses by repeating the training procedures using the same configuration while using different random seeds.

In the case of the Bayes-by-backprop methods, the algorithm explicitly outputs a distribution over output hypothesis, so we simply sample the networks from the learned weight distribution.

In particular, we aim to empirically demonstrate that a deep learning algorithm with stronger ensemble robustness presents better generalization performance (Theorem 1).

Recall the definition of ensemble robustness in Definition 3, another obstacle in calculating ensemble robustness is to find the most adversarial perturbation ∆s (or equivalently the most adversarial example z = s + ∆s) for a specific training sample s ∈ s within a partition set C i .

We therefore employ an approximate search strategy for finding the adversarial examples.

More concretely, we optimize the following first-order Taylor expansion of the loss function as a surrogate for finding the adversarial example: DISPLAYFORM0 with a pre-defined magnitude constraint r on the perturbation ∆s i .

In the simulations, we vary the magnitude r in order to calculate the empirical ensemble robustness at different perturbation levels.

We then calculate the empirical ensemble robustness by averaging the difference between the loss of the algorithm on the training samples and the adversarial samples output by the method in FORMULA11 : DISPLAYFORM1 with T = 10 denoting the size of the ensemble.

We emphasize that¯ (n) (Theorem 1) and the empirical approximation¯ (n) emp measure the non robustness of an algorithm, i.e., an algorithm is more robust if¯ (n) is smaller.

The generalization performance of different learning algorithms and different networks compared with the empirical ensemble robustness on MNIST is given in FIG0 .

Notice that the x-axis corresponds to the empirical ensemble robustness (Equation 3), and the y-axis corresponds to the test error.

Examining FIG0 we observe a high correlation between ensemble robustness and generalization for all learning algorithms, i.e., algorithms that are more robust (have lower¯ (n)) generalize better on this data set.

FIG1 in the appendix presents similar results on the notMNIST dataset, although we observe lower (yet positive) correlation for the Bayes-by-backprop algorithm in this case.

These observations support our claim on the relation between ensemble robustness and algorithm generalization performance in Theorem 1.We also compare ensemble robustness with robustness on MNIST in Table 1 , where robustness is measured similarly to ensemble robustness using Equation 3 but with T = 1 (while T = 10 for ensemble robustness).

Indeed, we observe that averaging over instances of the same algorithm, exhibits a higher correlation between generalization and robustness, i.e., ensemble robustness is a better estimation of the generalization performance than standard robustness.

In this paper, we investigated the generalization ability of stochastic deep learning algorithm based on their ensemble robustness; i.e., the property that if a testing sample is similar to a training sample, then its loss is close to the training error.

We established both theoretically and experimentally evidence that ensemble robustness of an algorithm, measured on the training set, indicates its generalization performance well.

Moreover, our theory and experiments suggest that DNNs may be robust (and generalize) while being fragile to specific adversarial examples.

Measuring ensemble robustness of stochastic deep learning algorithms may be computationally prohibitive as one needs to sample several output hypotheses of the algorithm.

Thus, we demonstrated that by learning the probability distribution of the weights of a neural network explicitly, e.g., via variational methods such as Bayes-by-backprop, we can still observe a positive correlation between robustness and generalization while using fewer computations, making ensemble robustness feasible to measure.

As a direct consequence, one can potentially measure the generalization error of an algorithm without using testing examples.

In future work, we plan to further investigate if ensemble robustness can be used for model selection instead of cross-validation (and hence, increasing the training set size), in particular in problems that have a small training set.

A different direction is to study the resilience of deep learning methods to adversarial attacks BID17 .

BID20 recently showed that ensemble methods are useful as a mean to defense against adversarial attacks.

However, they only considered implicit ensemble methods which are computationally prohibitive.

As our simulations show that explicit ensembles are robust as well, we believe that they are likely to be a useful defense strategy while reducing computational cost.

Finally, Theorem 2 suggests that a randomized algorithm can tolerate the non-robustness of some hypotheses to certain samples; this may help to explain Proposition 1 in BID13 : "For any dataset, there exist arbitrarily unstable non-robust algorithms such that has a small generalization gap".

We leave this intuition for future work.

In this section, we illustrate how ensemble robustness can well characterize the performance of various training strategies of deep learning.

In particular, we take the dropout as a concrete example.

Dropout is a widely used technique for optimizing deep neural network models.

We demonstrate that dropout is a random scheme to perturb the algorithm.

During dropout, at each step, a random fraction of the units are masked out in a round of parameter updating.

Assumption 1.

We assume the randomness of the algorithm A is parametrized by r = (r 1 , . . .

, r L ) ∈ R where r l , l = 1, . . .

, L are random elements drawn independently.

For a deep neural network consisting of L layers, the random variable r l is the dropout randomness for the l-th layer.

The next theorem establishes the generalization performance for the neural network with dropout training.

Theorem 3 (Generalization of Dropout Training).

Consider an L-layer neural network trained by dropout.

Let A be an algorithm with (K,¯ (n)) ensemble robustness.

Let ∆(H) denote the output hypothesis distribution of the randomized algorithm A on a training set s. Assume there exists a β > 0 such that, DISPLAYFORM0 with r and t only differing in one element.

Then for any δ > 0, with probability at least 1 − δ with respect to the random draw of the s and h ∼ ∆(H), DISPLAYFORM1 Theorem 3 also establishes the relation between the depth of a neural network model and the generalization performance.

It suggests that when using dropout training, controlling the variance β of the empirical performance over different runs is important: when β converges at the rate of L −3/4 , increasing the layer number L will improve the performance of a deep neural network model.

However, simply making L larger without controlling β does not help.

Therefore, in practice, we usually use voting from multiple models to reduce the variance and thus decrease the generalization error .

Also, when dropout training is applied for more layers in a neural network model, smaller variance of the model performance is preferred.

This can be compensated by increasing the size of training examples or ensemble of multiple models.

Lemma 1.

For a randomized learning algorithm A with (K,¯ (n)) uniform ensemble robustness, and loss function such that 0 ≤ (h, z) ≤ M , we have, DISPLAYFORM0 where we use P s to denote the probability w.r.t.

the choice of s, and |s| = n.

Proof.

Given a random choice of training set s with cardinality of n, let N i be the set of index of points of s that fall into the C i .

Note that (|N 1 |, . . .

, |N K |) is an i.i.d.

multinomial random variable with parameters n and (µ(C 1 ), . . . , µ(C K )).

The following holds by the Breteganolle-Huber-Carol inequality: DISPLAYFORM1 (c) DISPLAYFORM2 Here the inequalities (a) and (b) are due to triangle inequality, (c) is from the definition of ensemble robustness and the fact that the loss function is upper bounded by M , and (d) holds with a probability greater than 1 − δ.

Lemma 2.

For a randomized learning algorithm A with (K,¯ (n)) uniform ensemble robustness, and loss function such that 0 ≤ (h, z) ≤ M , we have, DISPLAYFORM3 Proof.

Let N i be the set of index of points of s that fall into the C i .

Note that (|N 1 |, . . .

, |N K |) is an i.i.d.

multinomial random variable with parameters n and (µ( DISPLAYFORM4 We then bound the term H as follows.

DISPLAYFORM5 Then we have, DISPLAYFORM6 To analyze the generalization performance of deep learning with dropout, following lemma is central.

Lemma 3 (Bounded difference inequality BID15 ).

Let r = (r 1 , . . .

, r L ) ∈ R be L independent random variables (r l can be vectors or scalars) with r l ∈ {0, 1} m l .

Assume that the function f : R L → R satisfies: DISPLAYFORM7 f (r (l) ) − f ( r (l) ) ≤ c l , ∀l = 1, . . .

, L, whenever r (l) and r (l) differ only in the l-th element.

Here, c l is a nonnegative function of l. Then, for every > 0, DISPLAYFORM8 .

Proof of Theorem 1.

Now we proceed to prove Theorem 1.

Using Chebyshev's inequality, Lemma 2 leads to the following inequality:Pr s {|L(h) − emp (h)| ≥ |h} ≤ nM E s max s∈s,z∼s | (h, s) − (h, z)| + 2M 2 n 2 .By integrating with respect to h, we can derive the following bound on the generalization error: and applying Lemma 3 we obtain (note that s is independent of r) P r {R(s, r) − E r R(s, r) ≥ |s} ≤ exp − 2 2Lβ 2 .

DISPLAYFORM0 We also have E s P r {R(s, r) − E r R(s, r) ≥ } = E s P r {R(s, r) − E r R(s, r) ≥ |s} ≤ exp DISPLAYFORM1 Setting the r.h.s.

equal to δ and writing as a function of δ, we have that with probability at least 1 − δ w.r.t.

the random sampling of s and r:R(s, r) − E r R(s, r) ≤ β 2L log(1/δ).

E r R(s, r) ≤¯ (n) + 2K ln 2 + 2 ln(1/δ) n holds with probability greater than 1−δ.

Observe that the above two inequalities hold simultaneously with probability at least 1 − 2δ.

Combining those inequalities and setting δ = δ/2 gives R(s, r) ≤ β 2L log(1/δ) +¯ (n) + 2K ln 2 + 2 ln(2/δ) n .

<|TLDR|>

@highlight

Explaining the generalization of stochastic deep learning algorithms, theoretically and empirically, via ensemble robustness

@highlight

This paper presents an adaptation of the algorithmic robustness of Xu&Mannor'12 and presents learning bounds and an experimental showing correlation between empirical ensemble robustness and generalization error. 

@highlight

Proposes a study of the generalization ability of deep learning algorithms using an extension of notion of stability called ensemble robustness and gives bounds on generalization error of a randomized algorithm in terms of stability parameter and provides empirical study attempting to connect theory with practice.

@highlight

The paper studied the generalization ability of learning algorithms from the robustness viewpoint in a deep learning context