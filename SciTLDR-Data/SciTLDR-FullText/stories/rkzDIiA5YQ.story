Distributed optimization is vital in solving large-scale machine learning problems.

A widely-shared feature of distributed optimization techniques is the requirement that all nodes complete their assigned tasks in each computational epoch before the system can proceed to the next epoch.

In such settings, slow nodes, called stragglers, can greatly slow progress.

To mitigate the impact of stragglers, we propose an online distributed optimization method called Anytime Minibatch.

In this approach, all nodes are given a fixed time to compute the gradients of as many data samples as possible.

The result is a variable per-node minibatch size.

Workers then get a fixed communication time to average their minibatch gradients via several rounds of consensus, which are then used to update primal variables via dual averaging.

Anytime Minibatch prevents stragglers from holding up the system without wasting the work that stragglers can complete.

We present a convergence analysis and analyze the wall time performance.

Our numerical results show that our approach is up to 1.5 times faster in Amazon EC2 and it is up to five times faster when there is greater variability in compute node performance.

The advent of massive data sets has resulted in demand for solutions to optimization problems that are too large for a single processor to solve in a reasonable time.

This has led to a renaissance in the study of parallel and distributed computing paradigms.

Numerous recent advances in this field can be categorized into two approaches; synchronous Dekel et al. (2012) ; Duchi et al. (2012) ; Tsianos & Rabbat (2016) ; Zinkevich et al. (2010) and asynchronous Recht et al. (2011); Liu et al. (2015) .

This paper focuses on the synchronous approach.

One can characterize synchronization methods in terms of the topology of the computing system, either master-worker or fully distributed.

In a master-worker topology, workers update their estimates of the optimization variables locally, followed by a fusion step at the master yielding a synchronized estimate.

In a fully distributed setting, nodes are sparsely connected and there is no obvious master node.

Nodes synchronize their estimates via local communications.

In both topologies, synchronization is a key step.

Maintaining synchronization in practical computing systems can, however, introduce significant delay.

One cause is slow processing nodes, known as stragglers Dean et al. (2012) ; Yu et al. (2017) ; Tandon et al. (2017) ; Lee et al. (2018) ; Pan et al. (2017) ; S. Dutta & Nagpurkar (2018) .

A classical requirement in parallel computing is that all nodes process an equal amount of data per computational epoch prior to the initiation of the synchronization mechanism.

In networks in which the processing speed and computational load of nodes vary greatly between nodes and over time, the straggling nodes will determine the processing time, often at a great expense to overall system efficiency.

Such straggler nodes are a significant issue in cloud-based computing systems.

Thus, an important challenge is the design of parallel optimization techniques that are robust to stragglers.

To meet this challenge, we propose an approach that we term Anytime MiniBatch (AMB).

We consider a fully distributed topologyand consider the problem of stochastic convex optimization via dual averaging Nesterov (2009); Xiao (2010) .

Rather than fixing the minibatch size, we fix the computation time (T ) in each epoch, forcing each node to "turn in" its work after the specified fixed time has expired.

This prevents a single straggler (or stragglers) from holding up the entire network, while allowing nodes to benefit from the partial work carried out by the slower nodes.

On the other hand, fixing the computation time means that each node process a different amount of data in each epoch.

Our method adapts to this variability.

After computation, all workers get fixed communication time (T c ) to share their gradient information via averaging consensus on their dual variables, accounting for the variable number of data samples processed at each node.

Thus, the epoch time of AMB is fixed to T + T c in the presence of stragglers and network delays.

We analyze the convergence of AMB, showing that the online regret achieves O( √m ) performance, which is optimal for gradient based algorithms for arbitrary convex loss Dekel et al. (2012) .

In here, m is the expected sum number of samples processed across all nodes.

We further show an upper bound that, in terms of the expected wall time needed to attain a specified regret, AMB is O( √ n − 1) faster than methods that use a fixed minibatch size under the assumption that the computation time follows an arbitrary distribution where n is the number of nodes.

We provide numerical simulations using Amazon Elastic Compute Cloud (EC2) and show that AMB offers significant acceleration over the fixed minibatch approach.

This work contributes to the ever-growing body of literature on distributed learning and optimization, which goes back at least as far as Tsitsiklis et al. (1986) , in which distributed first-order methods were considered.

Recent seminal works include Nedic & Ozdaglar (2009) , which considers distributed optimization in sensor and robotic networks, and Dekel et al. (2012) , which considers stochastic learning and prediction in large, distributed data networks.

A large body of work elaborates on these ideas, considering differences in topology, communications models, data models, etc.

Duchi et al. (2012) ; Tsianos et al. (2012) ; Shi et al. (2015) ; Xi & Khan (2017) .

The two recent works most similar to ours are Tsianos & Rabbat (2016) and Nokleby & Bajwa (2017) , which consider distributed online stochastic convex optimization over networks with communications constraints.

However, both of these works suppose that worker nodes are homogeneous in terms of processing power, and do not account for the straggler effect examined herein.

The recent work Pan et al. (2017) ; Tandon et al. (2017) ; S. Dutta & Nagpurkar (2018) proposed synchronous fixed minibatch methods to mitigate stragglers for master-worker setup.

These methods either ignore stragglers or use redundancy to accelerate convergence in the presence of stragglers.

However, our approach in comparison to Pan et al. (2017) ; Tandon et al. (2017) ; S. Dutta & Nagpurkar (2018) utilizes work completed by both fast and slow working nodes, thus results in faster wall time in convergence.

In this section we outline our computation and optimization model and step through the three phases of the AMB algorithm.

The pseudo code of the algorithm is provided in App.

A. We defer discussion of detailed mathematical assumptions and analytical results to Sec. 4.We suppose a computing system that consists of n compute nodes.

Each node corresponds to a vertex in a connected and undirected graph G(V, E) that represents the inter-node communication structure.

The vertex set V satisfies |V | = n and the edge set E tells us which nodes can communicate directly.

Let N i = {j ∈ V : (i, j) ∈ E, i = j} denote the neighborhood of node i.

The collaborative objective of the nodes is to find the parameter vector w ∈ W ⊆ R d that solves DISPLAYFORM0 The expectation E x [·] is computed with respect to an unknown probability distribution Q over a set X ⊆ R d .

Because the distribution is unknown, the nodes must approximate the solution in (1) using data points drawn in an independent and identically distributed (i.i.d.) manner from Q. Nesterov (2009); Dekel et al. (2012) as its optimization workhorse and averaging consensus Nokleby & Bajwa (2017); Tsianos & Rabbat (2016) to facilitate collaboration among nodes.

It proceeds in epochs consisting of three phases: compute, in which nodes compute local minibatches; consensus, in which nodes average their dual variables together; and update, in which nodes take a dual averaging step with respect to the consensus-averaged dual variables.

We let t index each epoch, and each node i has a primal variable w i (t) ∈ R d and dual variable z i (t) ∈ R d .

At the start of the first epoch, t = 1, we initialize all primal variables to the same value w(1) as

and all dual variables to zero, i.e., z i (1) = 0 ∈ R d .

In here, h : W → R is a 1-strongly convex function.

Compute Phase: All workers are given T fixed time to compute their local minibatches.

During each epoch, each node is able to compute b i (t) gradients of f (w, x), evaluated at w i (t) where the data samples x i (t, s) are drawn i.i.d.

from Q. At the end of epoch t, each node i computes its local minibatch gradient: DISPLAYFORM0 As we fix the compute time, the local minibatch size b i (t) is a random variable.

Let b(t) := n i=1 b i (t) be the global minibatch size aggregated over all nodes.

This contrasts with traditional approaches in which the minibatch is fixed.

In Sec. 4 we provide a convergence analysis that accounts for the variability in the amount of work completed by each node.

In Sec. 5, we presents a wall time analysis based on random local minibatch sizes.

Consensus Phase: Between computational epochs each node is given a fixed amount of time, T c , to communicate with neighboring nodes.

The objective of this phase is for each node to get (an approximation of) the following quantity: DISPLAYFORM1 The first term,z(t), is the weighted average of the previous dual variables.

The second, g(t) , is the average of all gradients computed in epoch t.

The nodes compute this quantity approximately via several synchronous rounds of average consensus.

Each node waits until it hears from all neighbors before starting a consensus round.

As we have fixed communication time T c , the number of consensus rounds r i (t) varies across workers and epochs due to random network delays.

Let P be a positive semi-definite, doubly-stochastic matrix (i.e., all entries of P are non-negative and all row-and column-sums are one) that is consistent with the graph G (i.e., DISPLAYFORM2 .

At the start of the consensus phase, each node i shares its message m DISPLAYFORM3 As long as G is connected and the second-largest eigenvalue of P is strictly less than unity, the iterations are guaranteed to converge to the true average.

For finite r i (t), each node will have an error in its approximation.

Instead of (4), at the end of the rounds of consensus, node i will have DISPLAYFORM4 where ξ i (t) is the error.

We use D (ri(t)) {y j } j∈V , i to denote the distributed averaging affected by r i (t) rounds of consensus.

Thus, DISPLAYFORM5 We note that the updated dual variable z i (t + 1) is a normalized version of the distributed average solution, normalized by DISPLAYFORM6 Update Phase: After distributed averaging of dual variables, each node updates its primal variable as DISPLAYFORM7 where ·, · denotes the standard inner product.

As will be discussed further in our analysis, in this paper we assume h : W → R to be a 1-strongly convex function and β(t) to be a sequence of positive non-decreasing parameters, i.e., β(t) ≤ β(t + 1).

We also work in Euclidean space where h(w) = w 2 is a typical choice.

In this section we analyze the performance of AMB in terms of expected regret.

As the performance is sensitive to the specific distribution of the processing times of the computing platform used, we first present a generic analysis in terms of the number of epochs processed and the size of the minibatches processed by each node in each epoch.

Then in Sec. 5, in order to illustrate the advantages of AMB, we assert a probabilistic model on the processing time and analyze the performance in terms of the elapsed "wall time" .

We assume that the feasible space W ∈ R d of the primal optimization variable w is a closed and bounded convex set where D = max w,u∈W w − u .

Let · denote the 2 norm.

We assume the objective function f (w, x) is convex and differentiable in w ∈ W for all x ∈ X. We further assume that f (w, x) is Lipschitz continuous with constant L, i.e. DISPLAYFORM0 Let ∇f (w, x) be the gradient of f (w, x) with respect to w. We assume the gradient of f (w, x) is Lipschitz continuous with constant K, i.e., ∇f (w, x) − ∇f (w, x) ≤ K w −w , ∀ x ∈ X, and ∀ w,w ∈ W.As mentioned in Sec. 3, DISPLAYFORM1 where the expectation is taken with respect to the (unknown) data distribution Q, and thus ∇F (w) = E[∇f (w, x)].

We also assume that there exists a constant σ that bounds the second moment of the norm of the gradient so that DISPLAYFORM2 Let the global minimum be denoted w * := arg min w∈W F (w).

First we bound the consensus errors.

Let z(t) be the exact dual variable without any consensus errors at each node DISPLAYFORM0 The following Lemma bounds the consensus errors, which is obtained using (Tsianos & Rabbat, 2016 , Theorem 2) DISPLAYFORM1 i (t) be the output after r rounds consensus.

Let λ 2 (P ) be the second eigenvalue of the matrix P and let ≥ 0, then DISPLAYFORM2 if the number of consensus rounds satisfies DISPLAYFORM3 We characterize the regret after τ epochs, averaging over the data distribution but keeping a fixed "sample path" of per-node minibatch sizes b i (t).

We observe that due to the time spent in communicating with other nodes via consensus, each node has computation cycles that could have been used to compute more gradients had the consensus phase been shorter (or nonexistent).

To model this, let a i (t) denote the number of additional gradients that node i could have computed had there been no consensus phase.

This undone work does not impact the system performance, but does enter into our characterization of the regret.

Let c i (t) = b i (t) + a i (t) be the total number of gradients that node i had the potential to compute during the t-th epoch.

Therefore, the total potential data samples processed in the t-th epoch is c(t) = n i=1 c i (t).

After τ epochs the total number of data points that could have been processed by all nodes in the absence of communication delays is DISPLAYFORM4 An important quantity is the ratio of total potential computations in each epoch to that actually completed.

Define the maximum such minibatch "skewness" as DISPLAYFORM5 It turns out that it is important to compute this skewness across epochs (i.e., c(t + 1) versus b(t)) in order to bound the regret via a telescoping sum. [Details can be found in the supplementary material.]In practice, a i (t) and b i (t) (and therefore c i (t)) depend on latent effects, e.g., how many other virtual machines are co-hosted on node i, and therefore we model them as random variables.

We bound the expected regret for a fixed sample path of a i (t) and b i (t).

The sample paths of importance are c tot (τ ) = {c i (t)} i∈V,t∈ [τ ] and b tot (τ ) = {b i (t)} i∈V,t∈ [τ ] , where we introduce c tot and b tot for notational compactness.

Define the average regret after τ epochs as DISPLAYFORM6 where the expectation is taken with respect the the i.i.d.

sampling from the distribution Q. Then, we have the following bound on R(τ ).Theorem 2 Suppose workers collectively processed m samples after τ epochs, cf. (15), minibatch skewness parameter γ, cf.

FORMULA0 , and let c max = max t∈[τ ] c(t), c avg = (1/τ ) τ t=1 c(t) and δ = max {t,t }∈{1,τ −1} |c(t)−c(t )| be the maximum, average, and variation across c(t).

Further, suppose the averaging consensus has additive accuracy , cf.

Lemma 1.

Then, the expected regret is DISPLAYFORM7 Theorem 2 is proved in App.

B of the supplementary material.

We now make a few comments about this result.

First, recall that the expectation is taken with respect to the data distribution, but holds for any sample path of minibatch sizes.

Further, the regret bound depends only on the summary statistics c max , c avg , δ, and γ.

These parameters capture the distribution of the processing speed at each node.

Further, the impact of consensus error, which depends on the communication speed relative to the processing speed of each node, is summarized in the assumption of uniform accuracy on the distributed averaging mechanism.

Thus, Theorem 2 is a sample path result that depends only coarsely on the distribution of the speed of data processing.

Next, observe that the dominant term is the final one, which scales in the aggregate number of samples m. The first term is approximately constant, only scaling with the monotonically increasing β and c max parameters.

The terms containing characterizes the effect of imperfect consensus, which can be reduced by increasing the number of rounds of consensus.

The effect of variability across c(t) is reflected in the terms containing the c max , c avg and δ parameters.

If perfect consensus were achieved ( = 0) then all components of the final term that scales in √ m would disappear except for the term that contains the minibatch skewness parameter γ.

It is through this term that the amount of useful computation performed in each epoch (b i (t) ≤ c i (t)) enters the result.

In the special case of constant minibatch size c max = c avg and δ = 0, we have the following corollary.

Corollary 3 If c(t) = c for all t ∈ [τ ] and the consensus error ≤ 1/c, then the expected regret is DISPLAYFORM8

We can translate Theorem 2 and Cor.

3 to a regret bound averaged over the sample path.

Since the summary statistics c max , c avg , δ, and γ are sufficient to bound the regret, we assert a joint distribution p over these terms rather than over the sample path b tot (τ ), c tot (τ ).

For the following result, we need only specify several moments of the distribution.

In Sec. 5 we will take the further step of choosing a specific distribution p. DISPLAYFORM0 ] so thatm = τc is the expected total work that can be completed in τ epochs.

Also, let DISPLAYFORM1 If averaging consensus has additive accuracy , then the expected regret is bounded by DISPLAYFORM2 Theorem 4 is proved in App.

F of the supplementary material.

Note that this expected regret is over both the i.i.d.

choice of data samples and the i.i.d.

choice of (b(t), c(t)) pairs.

Corollary 5 If ≤ 1/c, the expected regret is DISPLAYFORM3 Remark 1 Note that by letting = 0, we can immediately find the results for master-worker setup.

In the preceding section we studied regret as a function of the number of epochs.

The advantages of AMB is the reduction of wall time.

That is, AMB can get to same convergence in less time than fixed minibatch approaches.

Thus, in this section, we caracterize the wall time performance of AMB.In AMB, each epoch corresponds to a fixed compute time T .

As we have already commented, this contrasts with fixed minibatch approaches where they have variable computing times.

We refer "Fixed MiniBatch" methods as FMB.

To gain insight into the advantages of AMB, we develop an understanding of the regret per unit time.

We consider an FMB method in which each node computes computes b/n gradients, where b is the size of the global minibatch in each epoch.

Let T i (t) denote the amount of time taken by node i to compute b/n gradients for FMB method.

We make the following assumptions: DISPLAYFORM0 The time T i (t) follows an arbitrary distribution with the mean µ and the variance σ 2 .

Further, T i (t) is identical across node index i and epoch index t. .

Assumption 2 If node i takes T i (t) seconds to compute b/n gradients in the t-th epoch, then it will take nT i (t)/b seconds to compute one gradient.

Lemma 6 Let Assumptions 1 and 2 hold.

Let the FMB scheme have a minibatch size of b. Letb be the expected minibatch size of AMB.

Then, if we fix the computation time of an epoch in AMB to DISPLAYFORM1 Lemma 6 is proved in App.

G and it shows that the expected minibatch size of AMB is at least as big as FMB if we fix T = (1 + n/b)µ. Thus, we get same (or better) expected regret bound.

Next, we show that AMB achieve this in less time.

Theorem 7 Let Assumptions 1 and 2 hold.

Let T = (1 + n/b)µ and minibatch size of FMB is b. Let S A and S F be the total compute time across τ epochs of AMB and FMB, respectively, then DISPLAYFORM2 The proof is given in App.

G. Lemma 6 and Theorem 7 show that our method attains the same (or better) bound on the expected regret that is given in Theorem 4 but is at most 1 + σ/µ √ n − 1 faster than traditional FMB methods.

In Bertsimas et al. FORMULA1 , it was shown this bound is tight and there is a distribution that achieves it.

In our setup, there are no analytical distributions that exactly match with finishing time distribution.

Recent papers on stragglers Lee et al. FORMULA0 ; S. Dutta & Nagpurkar (2018) use the shifted exponential distribution to model T i (t).

The choice of shifted exponential distribution is motivated by the fact that it strikes a good balance between analytical tractability and practical behavior.

Based on the assumption of shifted exponential distribution, we show that AMB is O(log(n)) faster than FMB.

This result is proved in App.

H.

To evaluate the performance of AMB and compare it with that of FMB, we ran several experiments on Amazon EC2 for both schemes to solve two different classes of machine learning tasks: linear regression and logistic regression using both synthetic and real datasets.

In this section we present error vs. wall time performance using two experiments.

Additional simulations are given in App.

I

We solved two problems using two datasets: synthetic and real.

Linear regression problem was solved using synthetic data.

The element of global minimum parameter, w * ∈ R d , is generated from the multivariate normal distribution N (0, I).

The workers observe a sequence of pairs (x i (s), y i (s)) where s is the time index, data DISPLAYFORM0 .

The aim of all nodes is to collaboratively learn the true parameter w * .

The data dimension is d = 10 5 .For the logistic regression problem, we used the MNIST images of numbers from 0 to 9.

Each image is of size 28 × 28 pixels which can be represented as a 784-dimensional vector.

We used MNIST training dataset that consists of 60,000 data points.

The cost function is the cross-entropy function J DISPLAYFORM1 where x is the observed data point sampled randomly from the dataset, y is the true label of DISPLAYFORM2 is the indicator function and P(y = i|x) is the predicted probability that y = i given the observed data point x which can be calculated using the softmax function.

In other words, P(y = i|x) = e wix / j e wj x .

The aim of the system is to collaboratively learn the parameter w ∈ R c×d , where c = 10 classes and d = 785 the dimension (including the bias term) that minimizes the cost function while streaming the inputs x online.

We tested the performance of AMB and FMB schemes using fully distributed setup.

We used a network consisting of n = 10 nodes, in which the underlying network topology is given in FIG4 of App.

I.1.

In all our experiments, we used t2.micro instances and ami-6b211202, a publicly available To ensure a fair comparison between the two schemes, we ran both algorithms repeatedly and for a long time and averaged the performance over the same duration.

We also observed that the processors finish tasks much faster during the first hour or two before slowing significantly.

After that initial period, workers enter a steady state in which they keep their processor speed relatively constant except for occasional bursts.

We discarded the transient behaviour and considered the performance during the steady-state.

We ran both AMB and FMB in a fully distributed setting to solve the linear regression problem.

In FMB, each worker computed b = 6000 gradients.

The average compute time during the steady-state phase was found to be 14.5 sec.

Therefore, in AMB case, the compute time for each worker was set to be T = 14.5 sec. and we set T c = 4.5 sec. Workers are allowed r = 5 average rounds of consensus to average their calculated gradients.

Figure 1(a) plots the error vs. wall time, which includes both computation and communication times.

One can notice AMB clearly outperforms FMB.

In fact, the total amount of time spent by FMB to finish all the epochs is larger than that spent by AMB by almost 25% as shown in FIG1 (a) (e.g., the error rate achieved by FMB after 400 sec. has already been achieved by AMB after around 300 sec.).

We notice, both scheme has the same average inter-node communication times.

Therefore, when ignoring inter-node communication times, this ratio increases to almost 30%.

In here we perform logistic regression using n = 10 distributed nodes.

The network topology is as same as above.

The per-node fixed minibatch in FMB is b/n = 800 while the fixed compute time in AMB is T = 12 sec. and the communication time T c = 3 sec. As in the linear regression experiment above, the workers on average go through r = 5 round of consensus.

Figures 1(b) shows the achieved cost vs. wall clock time.

We observe AMB outperforms FMB by achieving the same error rate earlier.

In fact, FIG1 (b) demonstrates that AMB is about 1.7 times faster than FMB.

For instance, the cost achieved by AMB at 150 sec. is almost the same as that achieved by FMB at around 250 sec.

We proposed a distributed optimization method called Anytime MiniBatch.

A key property of our scheme is that we fix the computation time of each distributed node instead of minibatch size.

Therefore, the finishing time of all nodes are deterministic and does not depend on the slowest processing node.

We proved the convergence rate of our scheme in terms of the expected regret bound.

We performed numerical experiments using Amazon EC2 and showed our scheme offers significant improvements over fixed minibatch schemes.

A AMB ALGORITHM

The pseudocode of the Anytime Minibatch scheme operating in a distributed setting is given in Algorithm 1.

Line 2 is for initialization purpose.

Lines 3 − 8 corresponds to the compute phase during which each node i calculates b i (t) gradients.

The consensus phase steps are given in lines 9 − 21.

Each node first averages the gradients (line 9) and calculates the initial messages m i (t) it will share with its neighbours (line 10).

Lines 14 − 19 corresponds to the communication rounds that results in distributed averaging of the dual variable z i (t + 1) (line 21).

Finally, line 22 represents the update phase in which each node updates its primal variable w i (t + 1).For the hub-and-spoke configuration, one can easily modify the algorithm as only a single consensus round is required during which all workers send their gradients to the master node which calculates z(t + 1) and w(t + 1) followed by a communication from the master to the workers with the updated w(t + 1).

DISPLAYFORM0 initialize g i (t) = 0, b i (t) = 0 3: DISPLAYFORM1 while current_time DISPLAYFORM2 receive input start consensus rounds 10: DISPLAYFORM3 DISPLAYFORM4 11: DISPLAYFORM5 DISPLAYFORM6

w i (t + 1) = arg min w∈W w, z i (t + 1) + β(t + 1)h(w)23: end for B PROOF OF THEOREM 2In this section, we prove Theorem 2.

There are three factors impacting the convergence of our scheme; first is that gradient is calculated with respect to f (w, x) rather than directly computing the exact gradient ∇ w F (w), the second factor is the errors due to limited consensus rounds, and the last factor is that we have variable sized minibatch size over epochs.

We bound these errors to find the expected regret bound with respect to a sample path.

Let w(t) be the primal variable computed using the exact dual z(t), cf.

12: DISPLAYFORM0 From (Tsianos & Rabbat, 2016, Lemma 2), we have DISPLAYFORM1 Recall that z i (t) is the dual variable after r rounds of consensus.

The last step is due to Lemma 1.

Let X(t) be the total set of samples processed by the end of t-th epoch: DISPLAYFORM2 Let E[·] denote the expectation over the data set X(τ ) where we recall τ is the number of epochs.

Note that conditioned on X(t − 1) the w i (t) and x i (t, s) are independent according to equation 7.

Thus, DISPLAYFORM3 where equation 25 is due to equation 10.

From equation 17 we have DISPLAYFORM4 Now, we add and subtract F (w(t)) from equation 17 to get DISPLAYFORM5 Note that equation 28 and equation 29 are due to equation 8 and equation 24.

Now, we bound the first term in the following Lemma, which is proved in App.

C. DISPLAYFORM6 where DISPLAYFORM7 In equation 31, the first term is a constant, which depends on the initialization.

The fourth and the sixth terms are due to consensus errors and the fifth term is due to noisy gradient calculation.

The second and the last term E[ψ] are due to variable minibatch sizes.

Now, the total regret can be obtained by using Lemma 8 in equation 30 DISPLAYFORM8 Define γ = max t∈{1,τ −1} DISPLAYFORM9 In App.

D, we bound DISPLAYFORM10 1 α(t) and DISPLAYFORM11 1 α(t)β(t) 2 terms.

Using them, we have DISPLAYFORM12 Now we bound E[ψ].

Using δ = max {t,t }∈{1,τ −1} |c(t) − c(t )| in equation 32, we can write DISPLAYFORM13 By substituting equation 37 in equation 36 DISPLAYFORM14 By rearranging terms DISPLAYFORM15 , then from equation 15 µτ = m and we substitute DISPLAYFORM16 This completes the proof of Theorem 2.C PROOF OF LEMMA 8Note that g(t) is calculated with respect to w i (t) by different nodes in equation 3.

Letḡ(t) be the minibatch calculated with respect to w(t) (given in equation 23) by all the nodes.

DISPLAYFORM17 Note that there are two types of errors in computing gradients.

The first is common in any gradient based methods.

That is, the gradient is calculated with respect to the function f (w, x), which is based on the data x instead of being a direct evaluation of ∇ w F (w).

We denote this error as q(t): DISPLAYFORM18 The second error results from the fact that we use g(t) instead ofḡ(t).

We denote this error as r(t): DISPLAYFORM19 Lemma 9 The following four relations hold DISPLAYFORM20 The proof of Lemma 9 is given in App.

E. Let l t (w) be the first order approximation of F (w) at w(t): DISPLAYFORM21 Letl t (w) be an approximation of l t (w) by replacing ∇ w F (w(t)) with g(t)l t (w) = F (w(t)) + g(t), w − w(t) (45) = F (w(t)) + ∇ w F (w(t)), w − w(t) + q(t), w − w(t) + r(t), w − w(t) (46) = l t (w) + q(t), w − w(t) + r(t), w − w(t) .Note that equation 46 follows since g(t) = q(t) + r(t) + ∇ w F (w(t)).

By using the smoothness of F (w), we can write DISPLAYFORM22 The last step is due to the Cauchy-Schwarz inequality.

Let α(t) = β(t) − K. We add and subtract α(t) w(t + 1) − w(t) 2 /2 to find DISPLAYFORM23 Note that DISPLAYFORM24 Similarly, we have that DISPLAYFORM25 Using equation 49, equation 50, and β(t) = K + α(t) in equation 48 we have DISPLAYFORM26 The following Lemma gives a relation between w(t) andl t (w(t))Lemma 10 The optimization stated in equation 23 is equivalent to DISPLAYFORM27 By using the result (Dekel et al., 2012 , Lemma 8), we have DISPLAYFORM28 t (w(t + 1)) + (β(t))h(w(t + 1)) DISPLAYFORM29

Use equation 51 in equation 53 and substituting in β(t) = K + α(t) we get DISPLAYFORM0 where equation 54 is due to the fact that α(t + 1) ≥ α(t).

Now, we use β(t) = K + α(t), multiply by c(t + 1) and rewrite DISPLAYFORM1

Summing from t = 1 to τ − 1 we get DISPLAYFORM0

Let ψ be the last two terms, i.e., DISPLAYFORM0 Then, using Lemma 10 DISPLAYFORM1 By substituting in equation 45 we continue DISPLAYFORM2 where equation 57 is due to convexity of F (w), i.e., DISPLAYFORM3 .

Adding and subtracting terms we find that DISPLAYFORM4 Taking the expectation with respect to X(τ − 1) DISPLAYFORM5 We use the bounds in Lemma 9 to get DISPLAYFORM6 We rewrite by rearranging terms DISPLAYFORM7 Now we bound E [ψ] .

From equation 56 we find DISPLAYFORM8 where equation 59 is due to Lemma 10, equation 60 is simple substitution of equation 45, and the last step is due to convexity of F (w).

Now, we take the expectation over data samples X(τ − 1) DISPLAYFORM9 DISPLAYFORM10 where Lemma 9 is used in equation 62 and the last step is due to equation 64.

This completes the proof of Lemma 8.

We know β(t) = K + α(t).

Let α(t) = t µ .

Then, we have DISPLAYFORM0 Similarly, DISPLAYFORM1 E PROOF OF LEMMA 9Note that the expectation with respect to x s (t) DISPLAYFORM2 Also we use the fact that gradient and expectation operators commutes DISPLAYFORM3 Bounding E[ q(t), w * − w(t) ] and E[ q(t) 2 ] follows the same approach as in (Dekel et al., 2012, Appendix A.1) or Tsianos & Rabbat (2016) .

Now, we find E[ r(t), w DISPLAYFORM4 DISPLAYFORM5 where equation 68 is due to the Cauchy-Schwarz inequality and equation 69 due to equation 9 and D = max w,u∈W w − u .

Using equation 24 DISPLAYFORM6 Now we find E[ r(t) 2 ].

DISPLAYFORM7 F PROOF OF THEOREM 4By definition DISPLAYFORM8 where c i (t) is the total number of gradients computed at the node i in the t-th epoch.

We assume c i (t) is independent across network and is independent and identically distributed according to some processing time distribution p across epochs.

DISPLAYFORM9 Let α(t) = t/c.

Now take expectation over the c(t) to get DISPLAYFORM10 The last step is due to the fact that c(t + 1) and b(t) are independent since these are in two different epochs.

Further E p [E[ψ|c(t)]] = 0.

After further simplification through the use of Appendix D, we get DISPLAYFORM11 Taking the expectation over c(t) in equation 30, we have DISPLAYFORM12 By definition DISPLAYFORM13 Thenm = E p =cτ .

By substitutingm and rearranging we find that DISPLAYFORM14 G PROOF OF THEOREM 7Proof: Consider an FMB method in which each node computes b/n gradients per epoch, with T i (t) denoting the time taken to complete the job.

Also consider AMB with a fixed epoch duration of T .

The number of gradient computations completed by the i-th node in the t-th epoch is DISPLAYFORM15 Therefore, the minibatch size b(t) computed in AMB in the t-th epoch is DISPLAYFORM16 Taking the expectation over the distribution of T i (t) in FORMULA10 , and applying Jensen's inequality, we find that DISPLAYFORM17 where E p [T i (t)] = mu.

Fixing the computing time to T = (1 + n/b)µ we find that E p [b(t)]

≥ b, i.e., the expected minibatch of AMB is at least as large as the minibatch size b used in the FMB.The expected computing time for τ epochs in our approach is DISPLAYFORM18 In contrast, in the FMB approach the finishing time of the tth epoch is max i∈[n]

T i (t).

Using the result of Arnold & Groeneveld (1979); Bertsimas et al. (2006) we find that DISPLAYFORM19 where σ is the standard deviation of T i (t).

Thus τ epochs takes expected time DISPLAYFORM20 Taking the ratio of the two finishing times we find that DISPLAYFORM21 For parallelization to be meaningful, the minibatch size should be much larger than number of nodes and hence b n. This means (1 + n/b) ≈ 1 for any system of interest.

Thus, DISPLAYFORM22 This completes the proof of Theorem 7.

The shifted exponential distribution is given by DISPLAYFORM0 where λ ≥ 0 and ζ ≥ 0.

The shifted exponential distribution models a minimum time (ζ) to complete a job, and a memoryless balance of processing time thereafter.

The λ parameter dictates the average processing speed, with larger λ indicating faster processing.

The expected finishing time is DISPLAYFORM1 By using order statistics, we can find DISPLAYFORM2 and thus τ epochs takes expected time DISPLAYFORM3 Taking the ratio of the two finishing times we find that DISPLAYFORM4 For parallelization to be meaningful we must have much more data than nodes and hence b n.

This means that the first factor in the denominator will be approximately equal to one for any system of interest.

Therefore, in the large n regime, DISPLAYFORM5 which is order-log(n) since the product λζ is fixed.

In this section, we present additional details regarding the numerical results of Section 6 of the main paper as well as some new results.

In Appendix I.1, we detail the network used in Section 6 and, for a point of comparison, implement the same computations in a master-worker network topology.

In Appendix I.2, we model the compute times of the nodes as shifted exponential random variables and, under this model, present results contrasting AMB and FMB performance for the linear regression problem.

In Appendix I.3 we present an experimental methodology for simulating a wide variety of straggler distributions in EC2.

By running background jobs on some of the EC2 nodes we slow the foreground job of interest, thereby simulating a heavily-loaded straggler node.

Finally, in Appendix I.4, we present another experiment in which we also induce stragglers by forcing the nodes to make random pauses between two consecutive gradient calculations.

We present numerical results for both settings as well, demonstrating the even greater advantage of AMB versus FMB when compared to the results presented in Section 6.

As there was not space in the main text, in FIG4 we diagram the connectivity of the distributed computation network used in Section 6.

The second largest eigenvalue of the P matrix corresponding to this network, which controls the speed of consensus, is 0.888.In Section 6, we presented results for distributed logistic regression in the network depicted in FIG4 .

Another network topology of great interest is the hub-and-spoke topology wherein a central master node is directly connected to a number of worker nodes, and worker nodes are only indirectly connected via the master.

We also ran the MNIST logistic regression experiments for this topology.

In our experiments there were 20 nodes total, 19 workers and one master.

As in Sec.6 we used t2.micro instances and ami-62b11202 to launch the instances.

We set the total batch size used in FMB to be b = 3990 so, with n = 19 worker each worker calculated b/n = 210 gradients per batch.

Working with this per-worker batch size, we found the average EC2 compute time per batch to be 3 sec.

Therefore, we used a compute time of T = 3 sec. in the AMB scheme while the communication time of T c = 1 sec. Figure 3 plots the logistical error versus wall clock time for both AMB and FMB in the master-worker (i.e., hub-and-spoke) topology.

We see that the workers implementing AMB far outperform those implementing FMB.

In this section, we model the speed of each worker probabilistically.

Let T i (t) denote the time taken by worker i to calculate a total of 600 gradients in the t-th epoch.

We assume T i (t) follows a shifted exponential distribution and is independent and identically distributed across nodes (indexed by i) and across computing epochs (indexed by t).

The probability density function of the shifted exponential is p Ti(t) (z) = λe −λ(z−ζ) .

The mean of this distribution is µ = ζ + λ −1 and its variance is λ −2 .

Conditioned on T i (t) we assume that worker i makes linear progress through the dataset.

In other words, worker i takes kT i (t)/600 seconds to calculate k gradients.

(Note that our model allows k to exceed 600.)

In the simulation results we present we choose λ = 2/3 and ζ = 1.

In the AMB scheme, node i computes b i (t) = 600T /T i (t) gradients in epoch t where T is the fixed computing time allocated.

To ensure a fair comparison between FMB and AMB, T is chosen according to Thm.

7.

This means that E[b(t)]

≥ b where b(t) = i b i (t) and b is the fixed minibatch size used by FMB.

Based on our parameter choices, T = (1 + n/b)µ = (1 + n/b) (λ −1 + ζ) = 2.5.

generate 20 sample paths; each sample path is a set {T i (t)} for i ∈ {1, . . .

, 20} and t ∈ {1, . . .

20}.At the end of each of the 20 computing epoch we conduct r = 5 rounds of consensus.

As can be observed in Fig. 4 , for all 20 sample paths AMB outperforms FMB.

One can also observe that there for neither scheme is there much variance in performance across sample paths; there is a bit more for FMB than for AMB.

Due to this small variability, in the rest of this discussion we pick a single sample path to plot results for.

Figures 5a and 5b help us understand the performance impact of imperfect consensus on both AMB and on FMB.

In each we plot the consensus error for r = 5 rounds of consensus and perfect consensus (r = ∞).

In FIG7 we plot the error versus number of computing epochs while in FIG7 we plot it versus wall clock time.

In the former there is very little difference between AMB and FMB.

This is due to the fact that we have set the computation times so that the expected AMB batch size equals the fixed FMB batch size.

On the other hand, there is a large performance gap between the schemes when plotted versus wall clock time.

It is thus in terms of real time (not epoch count) where AMB strongly outperforms FMB.

In particular, AMB reaches an error rate of 10 −3 in less than half the time that it takes FMB (2.24 time faster, to be exact).

In this section, we introduce a new experimental methodology for studying the effect of stragglers.

In these experiments we induce stragglers amongst our EC2 micro.t2 instances by running background jobs.

In our experiments, there were 10 compute nodes interconnected according to the topology of FIG4 .

The 10 worker nodes were partitioned into three groups.

In the first group we run two background jobs that "interfere" with the foreground (AMB or FMB) job.

The background jobs we used were matrix multiplication jobs that were continuously performed during the experiment.

This first group will contain the "bad" straggler nodes.

In the second group we run a single background job.

These will be the intermediate stragglers.

In the third group we do not run background jobs.

These will be the non-stragglers.

In our experiments, there are three bad stragglers (workers 1, 2, and 3), two intermediate stragglers (workers 4 and 5), and five non-stragglers (workers 6-10).We first launch the background jobs in groups one and two.

We then launch the FMB jobs on all nodes at once.

By simultaneously running the background jobs and FMB, the resources of nodes in the first two groups are shared across multiple tasks resulting in an overall slowdown in their computing.

The slowdown can be clearly observed in Figure 6a which depicts the histogram of the FMB compute times.

The count ("frequency") is the number of jobs (fixed mini batches) completed as a function of the time it took to complete the job.

The third (fast) group is on the left, clustered around 10 seconds per batch, while the other two groups are clustered at roughly 20 and 30 seconds.

Figure 6b depicts the same experiment as performed with AMB: first launching the background jobs, and then launching AMB in parallel on all nodes.

In this scenario compute time is fixed, so the histogram plots the number of completed batches completed as a function of batch size.

In the AMB experiments the bad straggler nodes appear in the first cluster (centered around batch size of 230) while the faster nodes appear in the clusters to the right.

In the FMB histogram per-worker batch size was fixed to 585 while in the AMB histograms the compute time was fixed to 12 sec.

We observe that these empirical results confirm the conditionally deterministic aspects of our statistical model of Appendix I.2.

This was the portion of the model wherein we assumed that nodes make linear progress conditioned on the time it takes to compute one match.

In Figure 6a , we observe it takes the non-straggler nodes about 10 seconds to complete one fixed-sized minibatch.

It takes the intermediate nodes about twice as long.

Turning to the AMB plots we observe that, indeed, the intermediate stragglers nodes complete only about 50% of the work that the non-straggler nodes do in the fixed amount of time.

Hence this "linear progress" aspect of our model is confirmed experimentally.

Figure 7 illustrates the performance of AMB and FMB on the MNIST regression problem in the setting of EC2 with induced stragglers.

As can be observed by comparing these results to those presented in FIG1 of Section 6, the speedup now effected by AMB over FMB is far larger.

While in FIG1 the AMB was about 50% faster than FMB it is now about twice as fast.

While previously AMB effect a reduction of 30% in the time it took FMB to hit a target error rate, the reduction now is about 50%.

Generally as the variation amongst stragglers increases we will see a corresponding improvement in AMB over FMB.

We conducted another experiment on a high-performance computing (HPC) platform that consists of a large number of nodes.

Jobs submitted to this system are scheduled and assigned to dedicated nodes.

Since nodes are dedicated, no obvious stragglers exist.

Furthermore, users of this platform do not know which tasks are assigned to which node.

This means that we were not able to use the same approach for inducing stragglers on this platform as we used on EC2.

In EC2, we ran background simulations on certain nodes to slow them down.

But, since in this HPC environment we cannot tell where our jobs are placed, we are not able to place additional jobs on a subset of those same nodes to induce stragglers.

Therefore, we used a different approach for inducing stragglers as we now explain.

First, we ran the MNIST classification problem using 51 nodes: one master and 50 worker nodes where workers nodes were divided into 5 groups.

After each gradient calculation (in both AMB and FMB), worker i pauses its computation before proceeding to the next iteration.

The duration of the pause of the worker in epoch t after calculating the s-th gradient is denoted by T i (t, s).

We modeled the T i (t, s) as independent of each other and each T i (t, s) is drawn according to the normal distribution N (µ j , σ 2 j ) if worker i is in group j ∈ [5].

If T i (t, s) < 0, then there is no pause and the worker starts calculating the next gradient immediately.

Groups with larger µ j model worse stragglers and larger σ 2 j models more variance in that straggler's delay.

In AMB, if the remaining time to compute gradients is less than the sampled T i (t, s), then the duration of the pause is the remaining time.

In other words, the node will not calculate any further gradients in that epoch but will pause till the end of the compute phase before proceeding to consensus rounds.

In our experiment, we chose (µ 1 , µ 2 , µ 3 , µ 4 , µ 5 ) = (5, 10, 20, 35, 55) and σ 2 j = j 2 .

In the FMB experiment, each worker calculated 10 gradients leading to a fixed minibatch size b = 500 while in AMB each worker was given a fixed compute time, T = 115 msec.

which resulted in an empirical average minibatch size b ≈ 504 across all epochs.

Figures 8a and 8b respectively depict the histogram of the compute time (including the pauses) for FMB and the histogram of minibatch sizes for AMB obtained in our experiment.

In each histogram, five distinct distributions can be discerned, each representing one of the five groups.

Notice that the fastest group of nodes has the smallest average compute time (the leftmost spike in FIG11 ) and the largest average minibatch size (the rightmost distribution in FIG11 ).In FIG12 , we compare the logistic regression performance of AMB with that of FMB for the MNIST data set.

Note that AMB achieves its lowest cost in 2.45 sec while FMB achieves the same cost only at 12.7 sec. In other words, the convergence rate of AMB is more than five times faster than that of FMB.

@highlight

Accelerate distributed optimization by exploiting stragglers.