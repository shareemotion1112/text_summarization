We introduce MTLAB, a new algorithm for learning multiple related tasks with strong theoretical guarantees.

Its key idea is to perform learning sequentially over the data of all tasks, without interruptions or restarts at task boundaries.

Predictors for individual tasks are derived from this process by an additional online-to-batch conversion step.



By learning across task boundaries, MTLAB achieves a sublinear regret of true risks in the number of tasks.

In the lifelong learning setting, this leads to an improved generalization bound that converges with the total number of samples across all observed tasks, instead of the number of examples per tasks or the number of tasks independently.

At the same time, it is widely applicable: it can handle finite sets of tasks, as common in multi-task learning, as well as stochastic task sequences, as studied in lifelong learning.

In recent years, machine learning has become a core technology in many commercially relevant applications.

One observation in this context was that real-world learning tasks often do not occur in isolation, but rather as collections or temporal sequences of many, often highly related tasks.

Examples include click-through rate prediction for online ads, personalized voice recognition for smart devices, or handwriting recognition of different languages.

Multi-task learning BID3 has been developed exactly to handle such situations.

It is based on an intuitive idea that sharing information between tasks should help the learning process and therefore lead to improved prediction quality.

In practice, however, this is not guaranteed and multi-task learning can even lead to a reduction of prediction quality, so called negative transfer.

The question when negative transfer occurs and how it can be avoided has triggered a surge of research interest to better understanding the theoretical properties of multi-task learning, as well as related research areas, such as lifelong learning BID1 BID9 , where more and more tasks occur sequentially, and task curriculum learning , where the order in which to learn tasks needs to be determined.

In this work, we describe a new approach to multi-task learning that has strong theoretical guarantees, in particular improving the rate of convergence over some previous work.

Our core idea is to decouple the process of predictor learning from the task structure.

This is also the main difference of our approach to previous work, which typically learned one predictor for each task.

We treat the available data for all tasks as parts of a single large online-learning problem, in which individual tasks simply correspond to subsets of the data stream that is processed.

To obtain predictors for the individual tasks, we make use of online-to-batch conversion methods.

We name the method MTLAB (multi-task learning across boundaries).Our main contribution is a sublinear bound on the task regret of MTLAB with true risks.

As a corollary, we show that MTLAB improves the existing convergence rates in the case of lifelong learning.

From the regret-type bounds, we derive high probability bounds on the expected risk of each task, which constitutes a second main contribution of our work.

For real-world problems, not all tasks might be related to all previous ones.

Our third contribution is a theoretically well-founded, yet practical, mechanism to avoid negative transfer in this case: we show that by splitting the set of tasks into homogeneous groups and using MTLAB to learn individual predictors on each of the resulting subsequences of samples, one obtains the same strong guarantees for each of the learned predictors while avoiding negative transfer.

In this section we present the main notation and introduce the MTLAB approach to information transfer between tasks.

We face a sequence of tasks k 1 , . . .

, k n , . . . , where each k t from a task environment K, and the sequence is a random realization of a stochastic process over K. Note that this general formulation includes the situations most commonly studied in the literature: the case of finitely many fixed tasks (in which case the distribution over the tasks sequence is a delta peak) and the lifelong learning setting with i.i.d.

BID1 BID9 or non-i.i.d.

tasks .All tasks share the same input set X , output set Y, and hypothesis set H. Each task k t , however, has its own associated joint probability distribution, D t , over X ?? Y, conditioned on k t .

Whenever we observe a task k t , we receive a set S t = {(x t,i , y t,i )} mt i=1 sampled i.i.d.

from the task distribution D t , and we are given a loss function, DISPLAYFORM0 that measures the quality of predictions.

Alternatively, one can assume that all tasks share the same, a priori known, loss function.

Learning a task k t means to identify a hypothesis h ??? H with as small as possible per-task risk er t (h), which is defined as DISPLAYFORM1 The PAC-Bayes framework, originated in BID7 BID12 , studies the performance of stochastic (Gibbs) predictors.

A stochastic predictor is defined by a probability distribution Q over the hypotheses set.

For any Gibbs predictor with a distribution Q we define the corresponding true risk of a predictor as DISPLAYFORM2 As described in the introduction, we do not require that data for all tasks is available at the same time.

Instead, we adopt an online learning protocol for tasks: at step t we observe the dataset S t for task k t , and we output the distributionQ t .

Our first goal is, at any step n, to bound the regret of a learned sequence of predictorsQ 1 , . . .

,Q n with respect to any fixed reference distribution Q from some set, ???, of distributions, i.e. DISPLAYFORM0 Note that the regret is defined using true risks, that we do not observe, in contrast to empirical risks.

This makes the problem setting very different from the traditional online learning where the empirical performance is considered.

The main idea of MTLAB is to run an online learning algorithm on the samples from all tasks, essentially ignoring the task structure of the problem, and then use a properly defined online-to-batch conversion to obtain predictors for the individual tasks.

In this paper, we work with Input: decision set ???, initial distribution P , learning rate ?? Initialization: Q 1,0 = P At any time point t = 1, 2, . . . : BID4 run on the level of samples.

Let P be some prior distribution over H. We set Q 1,0 = P and, once we receive a dataset S t = {(x t,1 , y t,1 ), . . .

, (x t,mt , y t,mt )} on step t, we compute a sequence of predictors Q t,i each being a solution to DISPLAYFORM1 DISPLAYFORM2 for all i = 1, . . .

, m t with ?? > 0.

Afterwards, the algorithm outputs a predictorQ t = 1 mt mt i=1 Q t,i for task t, and sets Q t+1,0 = Q t,mt , to be used as a starting distribution for the next task.

We call the above procedure MTLAB (multi-task learning across task boundaries) and summarize it in Figure 1 .

Our first main result is a regret bound for the true risks of the sequence of distributions that it produces.

Theorem 1.

Letm = n/( n t=1 1/m t ) be the harmonic mean of m 1 , . . .

, m n and let P be a fixed prior distribution that is chosen independently of the data.

The predictors produced by MTLAB satisfy with probability 1 ??? ?? (over the random training sets) uniformly over Q ??? ??? DISPLAYFORM3 Corollary.

Set ?? = m n .

Then, with probability 1 ??? ??, it holds uniformly over DISPLAYFORM4 To put this result into perspective, we compare it to the average regret bounds given in BID0 , where the goal is to find the best possible data representation for tasks.

Even though the settings are a bit different, it gives a good idea of the qualitative nature of our result.

BID0 provides O( DISPLAYFORM5 ) bound (if all tasks are of the same size m) that can be sometimes improved to O( DISPLAYFORM6 In either case, convergence happen only in the regime when the number of tasks and the amount of data for each task both tend to infinity.

In contrast to this, the right hand side of inequality (5) converges to zero even if only one of the two quantities grows, so in particular for the most common case that the number of tasks grows to infinity, but the amount of data per task remains bounded.

The examples of real-world implementations of MTLAB are provided in the supplementary material.

We obtain further insight into the behavior of MTLAB by comparing it to the situation in which each task is learned independently.

A more traditional PAC-Bayes bound (e.g. BID6 ) states that with probability 1 ??? ?? the following inequality holds for all Q DISPLAYFORM0 (6) This inequality suggests a learning algorithm, namely to minimize the upper bound with respect to Q. In principle, MTLAB is based on a similar objective, but it acts on the sample level and it automatically provides relevant prior distributions for each task.

Thereby it is able to achieve better guarantee than one could get by combining separate bounds of the form (6) for multiple tasks.

The bound of Theorem 1 holds for any stochastic process over the tasks.

In particular, it holds in special case where tasks are sampled independently from a hyper distribution over the task environment, which is usually called lifelong learning BID1 BID9 .

In this setting, we have a fixed distribution T over K, and the sequence k 1 , . . .

, k n is an i.i.d.

sample from this distribution.

One can then define the lifelong risk as DISPLAYFORM0 where D k and k are the distribution and loss function for a task k, respectively.

The risk of the Gibbs predictor is then DISPLAYFORM1 . .

,Q n be the output of MTLAB, then we define the corresponding batch solution asQ n = 1 n n t=1Q t and observe DISPLAYFORM2 Using Theorem 1 we obtain the following guarantee.

Theorem 2.

In the lifelong learning setting, if we run MT-LAB with ?? = ???m ??? n , for any fixed prior distribution P that is chosen independently from the data, with probability 1 ??? ?? DISPLAYFORM3 Typical results for this setting, such as shown in BID9 BID5 BID0 , show the additive convergence rate O( DISPLAYFORM4 , which goes to zero only in the case of infinite data and infinite tasks.

In contrast, the generalization error for MTLAB converges in the most realistic scenario of finite data per task and increasing number of tasks.

The results of the previous section provide guarantees on MTLAB's multi-task regret.

In this section we compliment those results by presenting a modification that provides guarantees for individual risks of each task.

The detailed proofs of all statements can be found in the supplementary material.

As a start, let us consider a bound that can be obtained immediately from Theorem 1.

We make use of the following notion of relatedness between tasks that is commonly used in the field of domain adaptation BID2 .

Definition 1.

For a fixed hypothesis class H, the discrepancy between tasks k i and k j is defined as DISPLAYFORM0 The following theorem is an immediate corollary of Theorem 1.

Theorem 3.

Let P be a fixed prior distribution that is chosen independently of the data.

LetQ t be a sequence of predictors produced by MTLAB run with ?? = m n and let Q n = 1 n n t=1Q t .

Then the following inequality holds with probability 1 ??? ??, uniformly over Q ??? ??? DISPLAYFORM1 DISPLAYFORM2 This bound resembles the guarantees typical in the setting of learning from drifting distributions BID8 .

It converges if 1 n n i=1 disc(k i , k n ) ??? 0 with n, so if either tasks are identical to each other, or if tasks get suitably more similar on average with growing n.

This is a good example of possible negative transfer: when the previous tasks are not related to the current one as measured by the discrepancies, the average discrepancy term will prevent the bound from convergence.

The main question is if we can avoid the negative transfer and improve upon the bound of Theorem 3 in the case when 1 n n i=1 disc(k i , k n ) does not vanish over time.

Consider, for example, a simple case of two alternating tasks, i.e. DISPLAYFORM3 If we split the sequence of tasks into two subsequences, one for tasks with even and one for tasks with odd indices, and then run MT-LAB separately for each sequence, we could nevertheless guarantee the convergence of the error rate for the resulting procedure.

Unfortunately, it is rather easy to construct examples in which convergence to zero is not achievable, even with the best possible split of the sequence of tasks into subsequences.

Consequently, we redefine our goal to prove error rates that converge below a given threshold ??.

We present an online algorithm, MTLAB.MS (for MTLAB with Multiple Sequences), that splits the tasks into subsequences on the fly given some distance dist(k i , k j ) between tasks.

MTLAB.MS keeps a representative task for each subsequence, and we use the distances to the representatives to decide which subsequence to extend with the new task, or if a new subsequence needs to be initialized.

Pseudo-code for MTLAB.MS is provided in Algorithm 2.

The notationQ, P = MTLAB(S, P ) denotes a single run of MTLAB that takes a dataset S, runs its learning procedure starting from distribution P and outputs two distributions: the final distribution P to be used in the subsequent runs and the aggregate distributionQ that is a final predictor for the task.

Further notation used are: I n are the indices of the tasks in the subsequence chosen at step n, s n = |I n | is the size of this subsequence,m n is the harmonic average of the sizes of tasks in the chosen subsequence and ?? n is the learning rate of MTLAB associated with the chosen subsequence.

The following theorem shows that if MTLAB.MS could be run with the task discrepancies as distances, it would, for any given threshold ??, yield subsequences with generalization error below ??.

Theorem 4.

Let P be a fixed prior distribution that is chosen independently of the data.

If we run MTLAB.MS with dist(k i , k j ) = disc(k i , k j ), we get with probability 1 ??? ??, DISPLAYFORM4 This theorem works when the transfer algorithm uses a fixed learning rate ?? for each subsequence.

It is possible to prove a similar statement for the case when the parameters are optimized for the length of each subsequence using the machinery developed in BID13 .

However, the final statement gets more complicated and adds little to the discussions in the current paper.

Therefore, we leave this extension for future work.

Input: task distance dist, prior distribution P , threshold ?? Initialization: set of representative tasks R = ??? set of priors P = ??? At any time point t = 1, 2, . . .

:??? receive dataset S t .??? set I = {r ??? R : dist(k r , k t ) ??? ??} ??? if I = ??? then -add t to the set of representatives R -set P(t) = P ??? choose the closest representatives r = argmin DISPLAYFORM5 ??? run the transfer algorithm: DISPLAYFORM6 ??? set P(r ) = P ??? outputQ t Figure 2 .

MTLAB.MS algorithm Theorem 4 confirms that it is possible to avoid effects of negative transfer by carefully choosing the tasks we transfer knowledge from at each step.

MTLAB.MS is a computationally efficient way of doing this.

In practice, however, the true discrepancy values are unknown.

The most direct method to determine the right subsequence for each task is to estimate the discrepancies from the data and use the estimates in the MTLAB.MS algorithm.

In the supplementary material we detail two approaches for discrepancy estimation: a) using a part of the labelled training data and b) using separate unlabelled datasets.

In both cases it is possible to prove the statements similar to Theorem 4.

We introduced a new and widely applicable algorithm for sequentially learning of multiple tasks.

By performing learning across tasks boundaries it is able to achieve a sublinear regret bound and improves the convergence rates in the lifelong learning scenario.

MTLAB's way of not interrupting or restarting the learning process at task boundaries results in faster convergence rates than what can be achieved by learning individual predictors for each task: in particular, the generalization error decreases with the product of the number of tasks and the number of samples per task, instead of separately in each of these quantities.

We also introduced a mechanism for the situation when the tasks to be learned are not all related to each other.

We show that by constructing suitable subsequences of task, the convergence properties can hold even in this case.

<|TLDR|>

@highlight

A new algorithm for online multi-task learning that learns without restarts at the task borders