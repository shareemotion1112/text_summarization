We address the efficiency issues caused by the straggler effect in the recently emerged federated learning, which collaboratively trains a model on decentralized non-i.i.d. (non-independent and identically distributed) data across massive worker devices without exchanging training data in the unreliable and heterogeneous networks.

We propose a novel two-stage analysis on the error bounds of general federated learning, which provides practical insights into optimization.

As a result, we propose a novel easy-to-implement federated learning algorithm that uses asynchronous settings and strategies to control discrepancies between the global model and delayed models and adjust the number of local epochs with the estimation of staleness to accelerate convergence and resist performance deterioration caused by stragglers.

Experiment results show that our algorithm converges fast and robust on the existence of massive stragglers.

Distributed machine learning has received increasing attention in recent years, e.g., distributed stochastic gradient descent (DSGD) approaches (Gemulla et al., 2011; Lan et al., 2017) and the well-known parameter server paradigm (Agarwal & Duchi, 2011; Li et al., 2013; 2014) .

However, these approaches always suffer from communication overhead and privacy risk (McMahan et al., 2017) .

Federated learning (FL) (Konečnỳ et al., 2016 ) is proposed to alleviate the above issues, where a subset of devices are randomly selected, and training data in devices are locally kept when training a global model, thus reducing communication and protecting user privacy.

Furthermore, FL approaches are dedicated to a more complex context with 1) non-i.i.d. (Non-independent and identically distributed), unbalanced and heterogeneous data in devices, 2) constrained computing resources with unreliable connections and unstable environments (McMahan et al., 2017; Konečnỳ et al., 2016) .

Typically, FL approaches apply weight averaging methods for model aggregation, e.g., FedAvg (McMahan et al., 2017) and its variants (Sahu et al., 2018; Wang et al., 2018; Kamp et al., 2018; Leroy et al., 2019; Nishio & Yonetani, 2019) .

Such methods are similar to the synchronous distributed optimization domain.

However, synchronous optimization methods are costly in synchronization (Chen et al., 2018) , and they are potentially inefficient due to the synchrony even when collecting model updates from a much smaller subset of devices (Xie et al., 2019b) .

Besides, waiting time for slow devices (i.e., stragglers or stale workers) is inevitable due to the heterogeneity and unreliability as mentioned above.

The existence of such devices is proved to affect the convergence of FL (Chen et al., 2018) .

To address this problem, scholars propose asynchronous federated learning (AFL) methods (Xie et al., 2019a; Mohammad & Sorour, 2019; Samarakoon et al., 2018) that allow model aggregation without waiting for slow devices.

However, asynchrony magnifies the straggler effect because 1) when the server node receives models uploaded by the slow workers, it probably has already updated the global model for many times, and 2) real-world data are usually heavy-tailed in distributed heterogeneous devices, where the rich get richer, i.e., the straggler effect accumulates when no adjustment operations in stale workers, and eventually it affects the convergence of the global model.

Furthermore, dynamics in AFL brings more challenges in parameter tuning and speed-accuracy trade-off, and the guidelines for designing efficient and stale-robust algorithms in this context are still missing.

Contributions Our main contributions are summarized as follows.

We first establish a new twostage analysis on federated learning, namely training error decomposition and convergence analysis.

To the best of our knowledge, it is the first analysis based on the above two stages that address the optimization roadmap for the general federated learning entirely.

Such analysis provides insight into designing efficient and stale-robust federated learning algorithms.

By following the guidelines of the above two stages, we propose a novel FL algorithm with asynchronous settings and a set of easy-to-implement training strategies.

Specifically, the algorithm controls model training by estimating the model consistency and dynamically adjusting the number of local epochs on straggle workers to reduce the impact of staleness on the convergence of the global model.

We conduct experiments to evaluate the efficiency and robustness of our algorithm on imbalanced and balanced data partitions with different proportions of straggle worker nodes.

Results show that our approach converges fast and robust on the existence of straggle worker nodes compared to the state-of-the-art solutions.

Related Work Our work is targeting the AFL and staleness resilience approaches in this context.

Straggler effect (also called staleness) is one of the main problems in the similar asynchronous gradient descent (Async-SGD) approaches, which has been discussed by various studies and its remedies have been proposed (Hakimi et al., 2019; Lian et al., 2015; Chen et al., 2016; Cui et al., 2016; Chai et al., 2019; Zheng et al., 2017; Dai et al., 2018; Hakimi et al., 2019) .

However, these works are mainly targeting the distributed Async-SGD scenarios, which is different from FL as discussed in the previous section.

Existing FL solutions that address the straggler effect are mainly consensus-based.

Consensus mechanisms are introduced where a threshold metric (i.e., control variable) is computed, and only the workers who satisfy this threshold are permitted to upload their model (Chen et al., 2018; Smith et al., 2017; Nishio & Yonetani, 2019) .

Thus it significantly reduces the number of communications and updates model without waiting for straggle workers.

However, current approaches are mainly focusing on synchronized FL.

Xie et al. (2019a) propose an AFL algorithm which uses a mixing hyperparameter to adaptively control the trade-off between the convergence speed and error reduction on staleness.

However, this work and above mentioned FL solutions only consider the staleness caused by network delay instead of imbalanced data size in each worker and only evaluate on equal size of local data, which is inconsistent with the real-world cases.

Our approach is similar to (Xie et al., 2019a) , but instead we adaptively control the number of local epochs combined with the approximation of staleness and model discrepancy, and prove the performance guarantee on imbalanced data partitions.

We illustrate our approach in the rest of this paper.

We first summarize the general form of FL.

Generally, an FL system consists of M distributed worker nodes (e.g., mobile phones) and a server node.

The goal is training a global model across these worker nodes without uploading local data.

Each worker node employs the same machine learning model, and an optimizer (e.g., stochastic gradient descent) to iteratively optimize the loss function of the local model.

At t-th communication round, the server node uses an aggregation operator (e.g., averaging) to aggregate the local models uploaded by worker nodes, and broadcasts the aggregated global model to workers.

We use

mi to present local data points in worker node i, where m i is the size of data points in this worker.

The whole dataset χ = i X (i) , where i ∈ {1, 2, 3, ..., M }.

We assume that X (i) X (j) = ∅ for i = j, and apparently, the total size of data m = M i=1 m i .

We denote the model in worker node i by ω i ∈ R d , and the objective function of worker node i by where g(·) is the user-defined aggregation function, and ξ t is a vector which describes the settings of activated workers, such as worker ID, number of local epochs, and the learning rate.

Here and thereafter, we use g(ω t ) to represent g(ω t , ξ t ) for convenience.

We denote update term as h(·), a userdefined function which represents the model parameter differences between the collected models from activated worker nodes and previous global model, and

Here τ i is the time when worker node i received the global model

In this section, we aim to design an efficient and robust FL algorithm.

To do so, we first establish a twostage analysis, and finally, propose our new FL algorithm by combining the insights provided by the two stages.

Stage 1: Traning Error Decomposition.

We first discuss the main errors of the general FL.

We assume that each worker node has a local optimal model ω * i = arg min F i (ω).

Then at the communication round t, we define the global error as

where · is L 2 norm.

For worker node i, two terms in the right-hand side of inequality 7 respectively represent 1) initialization and local error: the error between the local model at communication round t and the optimal local model (the well known empirical risk).

Here, the initialization error (i.e., the error between the initial model and local model at communication round t) partially contributes to the first term.

2) local-global error: the error between optimal local models and optimal global solution, which is a constant given a specific learning task.

Figure 1 illustrates these errors.

Usually, the error between the initial model and the optimal global model is greater than the local-global error, and thus at the early stage of training, the first term is greater than the second term in the right-hand side of inequality 7.

Therefore, reducing the initialization error and the local error at the beginning of model training can reduce the global error ω t − ω * .

Afterward, when initialization and local error is minimized, the local-global error dominates the global error.

However, as we mentioned previously, the local-global error is a constant that can not be solved directly.

Therefore, we need a more sophisticated analysis to reach ω * since 7 is no longer appropriate to guide the optimization other than the early stage of FL training.

Following the above analysis, we analyze the convergence bounds of the general FL (Eq. 6) on the rest of the training stages other than the early stage.

Stage 2: Convergence Analysis.

First, we make the following assumptions on the objective functions: Assumption 1.

Smoothness.

For all i in {1, 2, 3, ..., M } and given constant β, the objective function F (ω) and F i (ω) are β-smooth, i.e.,

Assumption 2.

The first and second moment conditions.

The objective function F (ω) and the aggregation operation g(ω t ) satisfy the following:

2 .

E(·) is abbreviation of E ξt (·) which denotes the expected value w.r.t.

the distribution of the random variable ξ t given t.

Assumption 3.

Strong convexity.

For all i in {1, 2, 3, ..., M } and given constant c, the objective function F (ω) and F i (ω) are c-strong convex, i.e.,

Theorem 1.

Convergence for strongly-convex problems.

When c and β in assumption 1 and 3 satisfy c ≤ β, we can set the step size η t =η, where

, and L G = 1 + δ 2 G .

Withη, the upper error bound of global model satisfies:

The proof of theorem 1 is provided in appendix A.1.

Theorem 1 gives an error bound for the general form of model aggregation without assuming that g(ω t ) should come from ∇F (ω t ).

Note that the scalars δ and δ G are equal to 1 when g(ω t ) is the unbiased estimation of ∇F (ω t ).

However, current convergence bound in theorem 1 is too loose, and it can be further optimized by introducing controlled local epoch settings.

We assume that ∇F i (ω

Then we can extend theorem 1 with the local epochĒ.

Theorem 2.

Convergence with selected local epoch.

We use δ 0 and L 0 to represent the scalar δ and L in assumption 2 whenĒ = 1, and all worker nodes are assumed to participate in model training, then under 9, 8 can be rewritten as

The theorem 2 gives us the error bound of FL with the selected number of local epochs for stronglyconvex problems.

The proof of theorem 2 is provided in appendix A.2.

The right-hand side of theorem 2 implies the dynamics of hyper-parameters in local models for efficiency and robustness trade-off.

In a general FL algorithm, e.g., FedAvg, the model settings in worker nodes are always predefined and kept fixed, which may lead to a higher variance.

We now discuss such dynamics and practical insights on designing efficient and robust FL algorithms.

Selection of local epochs.

We discuss how to reduce the global error and communication round simultaneously for general FL.

From the second term of the right-hand side of 10, we can see that theorem 2 yields to linear convergence when E(

.

In this condition, to quickly reduce the global error, we can reduce the second term of the right-hand side of 10 by increasing the local epochĒ while reducing the communication round t.

Therefore, we can dynamically assign each worker with a bigger number of local epoch while reducing the communication round.

Asynchronous acceleration with stragglers.

We discuss why asynchronous strategies are needed in FL.

We rearrange 10 as:

When t increases, and we fixĒ andη, the global error only depends on L 0 .

L 0 can be controlled by sampling more worker nodes within a communication round.

Specifically, we compare n-worker participation with M -worker participation for model aggregation at the server node.

When we select n workers out of M , L 0 increases according to assumption 2(c) since the variance increases.

Thus, to get the same precision, we decreaseη, while it significantly slows the convergence speed.

However, in practice, waiting for all the workers to be ready is time-consuming.

Thus, we can introduce asynchronous mechanisms in model aggregation without waiting for all workers.

Robust training on straggle workers.

We discuss how to reduce the global error for FL on the existence of stragglers.

As we mentioned above, asynchronous strategies can accelerate model training by reducing the waiting time at each communication round.

However, the straggler effect is magnified by asynchrony, as discussed in section 1.

Stale workers accumulate their staleness, which increases the variances and affects the convergence of the global model.

A practical strategy to tame such effect is increasing the number of local epoch under the considerations that when the distributions of local data are far away from the global data, we use more epochs to train from the local data.

However, the divergence of these local epoch numbers between stale and non-stale workers may affect variance adversely, and we can adjust the number of local epoch with the normalized epochs from all workers to reduce such variance.

Theorem 3.

Convergence for non-convex problems With the Assumption 1 and 2, we can select a step size

The expected error bound satisfies:

Theorem 3 is similar to theorem 2 that the first term of the right-hand side of (12) does not decrease by iterative training.

Note that the above remarks are also applicable to theorem 3.

We provide proof of theorem 3 in appendix A.3.

Proposed Algorithm.

Under the guidance of the above analysis and the practical insights discussed above, we propose a fast and stale-robust AFL algorithm.

Algorithm 1 and 2 illustrate the processes in worker nodes and the server node, respectively.

H(t) is a predefined function at communication round t which determines how long should the server node waiting for the updated models from workers.

H(t) can be used to control the accuracy-speed trade-off.

The training processes on the server node can be divided into two stages, i.e., the initial stage and the converging stage.

We switch the stages by estimating the consistency of model updates.

Set t ← t + 1.

During ∆t time, receive the triplet (ω i , τ i , E i ) from any worker i.

Update ω t with ω i with τ i = t − 1 using 5.

Broadcast (ω t , t) to each worker.

Calculate U using 13.

9: until U ≤ 0.1 10: Broadcast start flag to each worker.

11: repeat // the converging stage.

Set t ← t + 1, ∆t ← H(t).

During ∆t time, receive the triplet (ω i , τ i , E i ) from any worker i.

Update ω t by 6 and 14.

Set E i ← mean(E) * s. 5: end if 6: Set τ i ← t, E i ← 0.

7: for e in 1, 2, 3, ..., E i do 8:

Randomly divide X (i) with batch size B i .

Update ω i by using opt for each batch.

Send triplet (ω i , τ i , E i ) to the server.

14:

e ← e − 1.

end if 18: end for Definition 1.

Update consistency.

The model update consistency of n worker nodes is the similarities between worker models at communication round t, i.e.,

U is consistent with the global error in inequality 7, and in algorithm 1 we empirically set 0.1 as the threshold to switch from the initial stage and the converging stage of the global model training.

At the initial stage of global model training, we use a bigger local epoch to accelerate training time as discussed above, and repeat this process until U ≤ 0.1.

After the initial stage, we define the update term as

ϕi ϕ is the above mentioned normalized local epochĒ with

, and

ϕ is the regularization term where ϕ = n i=1 ϕ i .

Finally, we define a stale-related penalty function of ϕ i as:

Here, ω t+1 f resh is the average model of worker nodes with τ i = t. The key processes of worker nodes are 1) estimating its staleness level, and 2) assign the number of local epoch using mean(E) in the received triplet from the server node and the previously estimated staleness level.

In the next section, we evaluate the performance of our algorithms.

We evaluate the performance of our approach on both imbalanced and balanced data partitions with the existence of stale worker nodes.

Experiment Settings.

We conduct experiments on Fashion-MNIST (Xiao et al., 2017) and CIFAR-10 ( Krizhevsky et al., 2009 ) to test the accuracy of our approach on 100 simulated workers, where 60 workers are stale.

We use 55,000 on Fashion-MNIST and 50,000 on CIFAR-10 for training and 10,000 for testing. [0, 1] normalization is used in the data preprocessing.

We conduct all experiments on CPU devices.

We use a light-weight convolutional neural network (CNN) model, which is suitable for mobile edge devices.

It has 4 convolutional layers that use 3 × 3 kernels with the size of 32, 64, 64, 128.

Rectified linear unit (ReLU) is used in each convolutional layer, and every two convolutional layers are followed by a 2 × 2 max-pooling layer and a dropout of 50%.

Finally, we use a 512-unit dense layer with ReLU and a dropout of 50% and an output layer with softmax.

We use an SGD optimizer with a learning rate of 0.01.

We set batch size as 50, and the initial number of local epochsĒ as 50.

We randomly split the data size in each worker node ranging from 2 to 2122 with a standard deviation of 480 on CIFAR-10, and 9 to 2157 with a standard deviation of 540 on Fashion-MNIST.

For the balanced cases we randomly assign each worker with 500 samples.

The communication speed of nodes is divided into ten levels ranging from 100 milliseconds to 1 second, and the 60 stale workers are assigned with bigger levels (6-10).

Finally, we set H(t) = 0.4s.

We compare the performance of our proposed method with four approaches: 1) FedAvg (McMahan et al., 2017) (synchronized).

We set the sampling rate C = 0.1 FedProx (Sahu et al., 2018) (synchronized).

We set C = 0.1, µ = 1 as the best parameters provided in their paper.

Results and Analysis.

Figure 2 shows the performance of our proposed algorithm and four baselines.

Our method converge faster compared to all the baselines, and the convergence is promised with 60% stale workers.

Furthermore, the whole upload times of our method do not increase with the same level of accuracy.

From the experiment results on Fashion-MNIST, we can see that our method has the same accuracy level on test data compared with synchronized approach such as FedAvg.

We can also see that on imbalanced data partitions (i.e., more realistic FL scenarios), our method is faster and more stable compared to other baselines.

Finally, we can clearly see the stage transition from the initial training stage to the converging stage (e.g., the transitions in imbalanced cases in figure 2(b) and (d)), which validates the efficiency of our approach.

Figure 3 shows the performance of our method with different proportion of stale nodes in 1,000 global communication rounds.

Our method outperforms the AFL baseline (i.e., FedAsync) in both accuracy and loss, and when the proportion of stale workers is less than 80%, our method outperforms the synchronized FL baseline (i.e., FedAvg).

In this paper, we propose a new two-stage analysis on federated learning, and inspired by such analysis, we propose a novel AFL algorithm that accelerates convergence and resists performance deterioration caused by stragglers simultaneously.

Experimental results show that our approach converges two times faster than baselines, and it can resist the straggler effect without sacrificing accuracy and communication.

As a byproduct, our approach improves the generalization ability of neural network models.

We will theoretically analyze it in future work.

Besides, while not the focus of our work, security and privacy are essential concerns in federated learning, and as the future work, we can apply various security methods to our approach.

Furthermore, besides the stale- We respectively test the performance with 20%, 60%, 80%, and 90% of stale workers.

The green dotted line is FedAvg which waits all selected workers.

resistance ability, the discrepancy estimation in our method also has the potential ability to resist malicious attacks to the worker nodes such as massive Byzantine attacks, which has been addressed in (Bagdasaryan et al., 2018; Li et al., 2019; Muñoz-González et al., 2019) .

We will analyze and evaluate such ability in future work.

A.1 PROOF OF THEOREM 1 Lemma 1.

Under the assumption 1, we can get:

Proof:

Under the assumption 1, for any ω and ω , we have:

Then using 18, we have:

Taking expectations in 19 w.r.t the distribution of ξ t , we complete the proof.

Lemma 2.

Under the assumption 1 and 2, we can get:

Proof: Using assumption 2(b) and 2(c), we have:

Then using 21, assumption 2(b) and lemma 1, we have:

We can easily get Lemma 2 by rearranging 22.

Then we prove Theorem 1 under the assumption 1, 2, 3.

First, we define

Function F is a quadratic model relevant toω.

Then it has the minimal value when all the partial derivatives are 0.

That is

Then, when we selectω =ω

for 23, we get the minimal of 23, which is

From assumption 3, we have

which is equivalent to

Then from Lemma 2, when a fixedη ≤ δ βL G is selected, we have:

And using 26, we have

Subtracting F (ω * ) from both sides and moving F (ω t ) from left to right, we get

Taking the whole expectations and rearranging 29, we obtain

Substracting the constantη βL 2cδ from both sides of 30, we have

The left hand side of 31 is a geometric series with common ratio 1 − η t δc, then we complete the proof.

We first prove 9.

Assume ∇F i (ω

Let ∇F i (ω t i ) = a and ν∇F i (ω

Since a+b a > 1 and ν ≤ 1, we have h(ν) min = h(0) = 0, and a ≈ b when η t is small.

We know that the smaller ν is, the smaller Res t i is.

Then we consider the situation that E i = E = 1, and define ∇F (ω t ) E(g E=1 (ω t )) ≥ δ 0 ∇F (ω t ) 2 and

Then sum all the form of 38 from 1 to t. We have

Besides, we can easily understand that F inf ≤ E(F (ω t+1 )), because F inf is the minimal value of F .

Then we have

By rearrange 40, we have

Dividing t from both sides of 41, we get

Then using equation 9, we complete the proof.

We conduct additional experiments to evaluate stale-robustness of our algorithm on CIFAR-10 based on the settings in section 4.

We visualize the impact of different staleness levels at different communication rounds with cosine angles (i.e., discrepancies) between the update terms (i.e., update directions of local models) of stale workers and fresh workers in figure 4 .

The results show that our method (in the first row) effectively adjusts the update direction of the reversed stale nodes while angles of stale nodes reverse with FedAvg compared to our algorithm, which shows the robustness of our method.

Figure 4: Impact visualization of different levels of staleness using cosine angles between the update terms defined in section 2 of fresh nodes (40 out of 100) and stale nodes (60 out of 100 worker nodes) on CIFAR-10 at different communication round.

The blue numbers represent the staleness levels by using the differences of version numbers of models between the stale nodes and the fresh nodes.

E.g., the staleness level is 10 at this communication round means that the fresh nodes has updated 10 more versions compared to the stale nodes.

<|TLDR|>

@highlight

We propose an efficient and robust asynchronous federated learning algorithm on the existence of stragglers