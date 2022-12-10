Large mini-batch parallel SGD is commonly used for distributed training of deep networks.

Approaches that use tightly-coupled exact distributed averaging based on AllReduce are sensitive to slow nodes and high-latency communication.

In this work we show the applicability of Stochastic Gradient Push (SGP) for distributed training.

SGP uses a gossip algorithm called PushSum for approximate distributed averaging, allowing for much more loosely coupled communications which can be beneficial in high-latency or high-variability scenarios.

The tradeoff is that approximate distributed averaging injects additional noise in the gradient which can affect the train and test accuracies.

We prove that SGP converges to a stationary point of smooth, non-convex objective functions.

Furthermore, we validate empirically the potential of SGP.

For example, using 32 nodes with 8 GPUs per node to train ResNet-50 on ImageNet, where nodes communicate over 10Gbps Ethernet, SGP completes 90 epochs in around 1.5 hours while AllReduce SGD takes over 5 hours, and the top-1 validation accuracy of SGP remains within 1.2% of that obtained using AllReduce SGD.

Deep Neural Networks (DNNs) are the state-of-the art machine learning approach in many application areas, including image recognition (He et al., 2016) and natural language processing (Vaswani et al., 2017) .

Stochastic Gradient Descent (SGD) is the current workhorse for training neural networks.

The algorithm optimizes the network parameters, x, to minimize a loss function, f (·), through gradient descent, where the loss function's gradients are approximated using a subset of training examples (a mini-batch).

DNNs often require large amounts of training data and trainable parameters, necessitating non-trivial computational requirements (Wu et al., 2016; Mahajan et al., 2018) .

There is a need for efficient methods to train DNNs in large-scale computing environments.

A parallel version of SGD is usually adopted for large-scale, distributed training (Goyal et al., 2017; Li et al., 2014) .

Worker nodes compute local mini-batch gradients of the loss function on different subsets of the data, and then calculate an exact inter-node average gradient using either the ALLRE-DUCE communication primitive, in synchronous implementations (Goyal et al., 2017) , or using a central parameter server, in asynchronous implementations (Dean et al., 2012) .

Using a parameter server to aggregate gradients introduces a potential bottleneck and a central point of failure (Lian et al., 2017) .

The ALLREDUCE primitive computes the exact average gradient at all workers in a decentralized manner, avoiding issues associated with centralized communication and computation.

However, exact averaging algorithms like ALLREDUCE are not robust in high-latency or highvariability platforms, e.g., where the network bandwidth may be a significant bottleneck, because they involve tightly-coupled, blocking communication (i.e., the call does not return until all nodes have finished aggregating).

Moreover, aggregating gradients across all the nodes in the network can introduce non-trivial computational overhead when there are many nodes, or when the gradients themselves are large.

This issue motivates the investigation of a decentralized and inexact version of SGD to reduce the overhead associated with distributed training.

There have been numerous decentralized optimization algorithms proposed and studied in the control-systems literature that leverage consensus-based approaches to aggregate information; see the recent survey Nedić et al. (2018) and references therein.

Rather than exactly aggregating gradi-ents (as with ALLREDUCE), this line of work uses less-coupled message passing algorithms which compute inexact distributed averages.

Most previous work in this area has focused on theoretical convergence analysis assuming convex objectives.

Recent work has begun to investigate their applicability to large-scale training of DNNs (Lian et al., 2017; Jiang et al., 2017) .

However, these papers study methods based on communication patterns which are static (the same at every iteration) and symmetric (if i sends to j, then i must also receive from j before proceeding).

Such methods inherently require blocking and communication overhead.

State-of-the-art consensus optimization methods build on the PUSHSUM algorithm for approximate distributed averaging (Kempe et al., 2003; Nedić et al., 2018) , which allows for non-blocking, time-varying, and directed (asymmetric) communication.

Since SGD already uses stochastic mini-batches, the hope is that an inexact average mini-batch will be as useful as the exact one if the averaging error is sufficiently small relative to the variability in the stochastic gradient.

This paper studies the use of Stochastic Gradient Push (SGP), an algorithm blending SGD and PUSHSUM, for distributed training of deep neural networks.

We provide a theoretical analysis of SGP, showing it converges for smooth non-convex objectives.

We also evaluate SGP experimentally, training ResNets on ImageNet using up to 32 nodes, each with 8 GPUs (i.e., 256 GPUs in total).

Our main contributions are summarized as follows:• We provide the first convergence analysis for Stochastic Gradient Push when the objective function is smooth and non-convex.

We show that, for an appropriate choice of the step size, SGP converges to a stationary point at a rate of O 1/ √ nK , where n is the number of nodes and K is the number of iterations.• In a high-latency scenario, where nodes communicate over 10Gbps Ethernet, SGP runs up to 3× faster than ALLREDUCE SGD and exhibits 88.6% scaling efficiency over the range from 4-32 nodes.• The top-1 validation accuracy of SGP matches that of ALLREDUCE SGD for up to 8 nodes (64 GPUs), and remains within 1.2% of ALLREDUCE SGD for larger networks.• In a low-latency scenario, where nodes communicate over a 100Gbps InfiniBand network supporting GPUDirect, SGP is on par with ALLREDUCE SGD in terms of running time, and SGP exhibits 92.4% scaling efficiency.• In comparison to other synchronous decentralized consensus-based approaches that require symmetric messaging, SGP runs faster and it produces models with better validation accuracy.

Problem formulation.

We consider the setting where a network of n nodes cooperates to solve the stochastic consensus optimization problem DISPLAYFORM0 ,...,n 1 n n i=1 E ξi∼Di F i (x i ; ξ i ) subject to x i = x j , ∀i, j = 1, . . .

, n.

Each node has local data following a distribution D i , and the nodes wish to cooperate to find the parameters x of a DNN that minimizes the average loss with respect to their data, where F i is the loss function at node i. Moreover, the goal codified in the constraints is for the nodes to reach agreement (i.e., consensus) on the solution they report.

We assume that nodes can locally evaluate stochastic gradients ∇F (x i ; ξ i ), ξ i ∼ D i , but they must communicate to access information about the objective functions at other nodes.

Distributed averaging.

The problem described above encompasses distributed training based on data parallelism.

There a canonical approach is large mini-batch parallel stochastic gradient descent: for an overall mini-batch of size nb, each node computes a local stochastic mini-batch gradient using b samples, and then the nodes use the ALLREDUCE communication primitive to compute the average gradient at every node.

Let f i (x i ) = E ξi∼Di F i (x i ; ξ i ) denote the objective at node i, and let DISPLAYFORM1 , averaging gradients via ALLREDUCE provides an exact stochastic gradient of f .

Typical implementations of ALLREDUCE have each node send and receive 2 n−1 n B bytes, where B is the size (in bytes) of the tensor being reduced, and involve 2 log 2 (n) communication steps (Rabenseifner, 2004) .

Moreover, ALLREDUCE is a blocking primitive, meaning that no node will proceed with local computations until the primitive returns.

Approximate distributed averaging.

In this work we explore the alternative approach of using a gossip algorithm for approximate distributed averaging-specifically, the PUSHSUM algorithm.

Gossip algorithms typically use linear iterations for averaging.

For example, let y (0) i ∈ R n be a vector at node i, and consider the goal of computing the average vector DISPLAYFORM2 at all nodes.

Stack the initial vectors into a matrix Y (0) ∈ R n×d with one row per node.

Typical gossip iterations have the form DISPLAYFORM3 n×n is referred to as the mixing matrix.

This corresponds to the update y DISPLAYFORM4 at node i. To implement this update, node i only needs to receive messages from other nodes j for which p (k)i,j = 0, so it will be appealing to use sparse P (k) to reduce communications.

Drawing inspiration from the theory of Markov chains (Seneta, 1981) , the mixing matrices P (k) are designed to be column stochastic.

Then, under mild conditions (e.g., ensuring that information from every node eventually reaches all other nodes) one can show that lim K→∞ K k=0 P (k) = π1 , where π is the ergodic limit of the chain and 1 is a vector with all entries equal to 1.

Consequently, the gossip iterations converge to a limit Y (∞) = π 1 Y (0) ; i.e., the value at node i converges to DISPLAYFORM5 j .

When the matrices P (k) are symmetric, it is straightforward to design the algorithm so that π i = 1/n for all i by making P (k) doubly stochastic.

However, symmetric Phas strong practical ramifications, such as requiring care in the implementation to avoid deadlocks.

The PUSHSUM algorithm only requires that P (k) be column-stochastic, and not necessarily symmetric (so node i may send to node j, but not necessarily vice versa).

Instead, one additional scalar parameter w (k) i is maintained at each node.

The parameter is initialized to w (0) i = 1 for all i, and updated using the same linear iteration, w (k+1) = P (k) w (k) .

Consequently, the parameter con- DISPLAYFORM6 = π i n at node i. Thus each node can recover the average of the initial vectors by computing the de-biased ratio y DISPLAYFORM7 .

In practice, we stop after a finite number of gossip iterations K and compute y DISPLAYFORM8 .

The distance of the de-biased ratio to the exact average can be quantified in terms of properties of the matrices DISPLAYFORM9 i,j > 0} denote the sets of nodes that i transmits to and receives from, respectively, at iteration k. If we use B bytes to represent the vector y

Algorithm description.

The stochastic gradient push (SGP) method for solving equation 1 is obtained by interleaving one local stochastic gradient descent update at each node with one iteration of PUSHSUM.

Each node maintains three variables: the model parameters x (k) i at node i, the scalar PUSHSUM weight w (k) i , and the de-biased parameters z DISPLAYFORM0 can be initialized to any arbitrary value as long as x DISPLAYFORM1 i .

Pseudocode is shown in Alg.

1.

Each node performs a local SGD step (lines 2-4) followed by one step of PUSHSUM for approximate distributed averaging (lines 5-8).Note that the gradients are evaluated at the de-biased parameters z (k) i in line 3, and they are then used to update x (k) i , the PUSHSUM numerator, in line 4.

All communication takes place in line 5, and each message contains two parts, the PUSHSUM numerator and denominator.

In particular, node i controls the values p (k) j,i used to weight the values in messages it sends.

Require: DISPLAYFORM0 Sample new mini-batch ξ DISPLAYFORM1 Compute a local stochastic mini-batch gradient at z DISPLAYFORM2 We are mainly interested in the case where the mixing matrices P (k) are sparse in order to have low communication overhead.

However, we point out that when the nodes' initial values are identical, DISPLAYFORM3 , and every entry of P (k) is equal to 1/n, then SGP is mathematically equivalent to parallel SGD using ALLREDUCE.

Please refer to appendix A for pratical implementation details, including how we design mixing matrices P (k) .Theoretical guarantees.

SGP was first proposed and analyzed in (Nedić & Olshevsky, 2016) assuming the local objectives f i (x) are strongly convex.

Here we provide convergence results in the more general setting of smooth, non-convex objectives.

We make the following three assumptions: DISPLAYFORM4 Note that this assumption implies that function f (x) is also L-smooth.

There exist finite positive constants σ 2 and ζ 2 such that DISPLAYFORM0 DISPLAYFORM1 Thus σ 2 bounds the variance of stochastic gradients at each node, and ζ 2 quantifies the similarity of data distributions at different nodes.3. (Mixing connectivity) To each mixing matrix P (k) we can associate a graph with vertex set {1, . . .

, n} and edge set DISPLAYFORM2 i,j > 0}; i.e., with edges (i, j) from j to i if i receives a message from j at iteration k. Assume that the graph with edge set DISPLAYFORM3 is strongly connected and has diameter at most ∆ for every l ≥ 0.

To simplify the discussion, we assume that every column of the mixing matrices P (k) has at most D non-zero entries.

Let DISPLAYFORM4 i .

Under similar assumptions, Lian et al. (2017) define that a decentralized algorithm for solving equation 1 converges if, for any > 0, it eventually satisfies DISPLAYFORM5 Our first result shows that SGP converges in this sense.

Theorem 1.

Suppose that Assumptions 1-3 hold, and run SGP for K iterations with step-size γ = n/K. Let f * = min x f (x) and assume that f * > −∞. There exist constants C > 0 and q ∈ (0, 1) which depend on B, n, and ∆ such that if the total number of iterations satisfies DISPLAYFORM6 where DISPLAYFORM7 The proof is given in Appendix C, where we also provide precise expressions for the constants C and q. The proof of Theorem 1 builds on an approach developed in Lian et al. (2017) .

Theorem 1 shows that, for a given number of nodes n, by running a sufficiently large number of iterations K (roughly speaking, Ω(n), which is reasonable for distributed training of DNNs) and choosing the step-size γ as prescribed, then the criterion equation 5 is satisfied with a number of iterations K = Ω(1/n 2 ).

That is, we achieve a linear speedup in the number of nodes.

Theorem 1 shows that the average of the nodes parameters, x (k) , converges, but it doesn't directly say anything about the parameters at each node.

In fact, we can show a stronger result.

Theorem 2.

Under the same assumptions as in Theorem 1, DISPLAYFORM8 The proof is also given in Appendix C.

This result shows that as K grows, the de-biased variables z DISPLAYFORM9 converge to the node-wise average x (k) , and hence the de-biased variables at each node also converge to a stationary point.

Note that for fixed n and large K, the 1/ √ nK term will dominate the other factors.

A variety of approaches have been proposed to accelerate distributed training of DNNs, including quantizing gradients BID0 Wen et al., 2007) and performing multiple local SGD steps at each node before averaging (McMahan et al., 2017) .

These approaches are complementary to the tradeoff we consider in this paper, between exact and approximate distributed averaging.

Similar to using PUSHSUM for averaging, both quantizing gradients and performing multiple local SGD steps before averaging can also be seen as injecting additional noise into SGD, leading to a trade off between training faster (by reducing communication overhead) and potentially obtaining a less accurate result.

Combining these approaches (quantized, inexact, and infrequent averaging) is an interesting direction for future work.

For the remainder of this section we review related work applying consensus-based approaches to large-scale training of DNNs.

Blot et al. (2016) report initial experimental results on small-scale experiments with an SGP-like algorithm.

Jin et al. (2016) make a theoretical connection between PUSHSUM-based methods and Elastic Averaging SGD (Zhang et al., 2015) .

Relative to those previous works, we provide the first convergence analysis for a PUSHSUM-based method in the smooth non-convex case.

Lian et al. (2017) and Jiang et al. (2017) study synchronous consensus-based versions of SGD.

However, unlike PUSHSUM, those methods involve symmetric message passing (if i sends to j at iteration k, then j also sends to i before both nodes update) which is inherently blocking.

Consequently, these methods are more sensitive to high-latency communication settings, and each node generally must communicate more per iteration, in comparison to PUSHSUM-based SGP where communication may be directed (i can send to j without needing a response from i).

The decentralized parallel SGD (D-PSGD) method proposed in Lian et al. (2017) produces iterates whose node-wise average, x (k) , is shown to converge in the sense of equation 5.

Our proof of Theorem 1, showing the convergence of SGP in the same sense, adapts some ideas from their analysis and also goes beyond to show that, since the values at each node converge to the average, the individual values at each node also converge to a stationary point.

We compare SGP with D-PSGD experimentally in Section 5 below and find that although the two methods find solutions of comparable accuracy, SGP is consistently faster.

Jin et al. (2016) and Lian et al. (2018) study asynchronous consensus-based methods for training DNNs.

Lian et al. (2018) analyzes an asynchronous version of D-PSGD and proves that its node-wise averages also converge to a stationary point.

In general, these contributions focusing on asynchrony can be seen as orthogonal to the use of a PUSHSUM based protocol for consensus averaging.

Next, we compare SGP with ALLREDUCE SGD, and D-PSGD (Lian et al., 2017) , an approximate distributed averaging baseline relying on doubly-stochastic gossip.

We run experiments on a large-scale distributed computing environment using up to 256 GPUs.

Our results show that when communication is the bottleneck, SGP is faster than both SGD and D-PSGD.

SGP also outperforms D-PSGD in terms of validation accuracy, while achieving a slightly worse accuracy compared to SGD when using a large number of compute nodes.

Our results also highlight that, in a setting where communication is efficient (e.g., over InfiniBand), doing exact averaging through ALLRE-DUCE SGD remains a competitive approach.

We run experiments on 32 DGX-1 GPU servers in a high-performance computing cluster.

Each server contains 8 NVIDIA Volta-V100 GPUs.

We consider two communication scenarios: in the high-latency scenario the nodes communicate over a 10 Gbit/s Ethernet network, and in the lowlatency scenario the nodes communicate over 100 Gbit/s InfiniBand, which supports GPUDirect RDMA communications.

To investigate how each algorithm scales, we run experiments with 4, 8, 16, and 32 nodes (i.e., 32, 64, 128, and 256 GPUs).We adopt the 1000-way ImageNet classification task (Russakovsky et al., 2015) as our experimental benchmark.

We train a ResNet-50 (He et al., 2016) following the experimental protocol of Goyal et al. (2017) , using the same hyperparameters with the exception of the learning rate schedule in the 32 node experiment for SGP and D-PSGD.

In the experiments, we also modify SGP to use Nesterov momentum.

In our default implementation of SGP, each node sends and receives to one other node at each iteration, and this destination changes from one iteration to the next.

Please refer to appendix A for more information about our implementation, including how we design/implement the sequence of mixing matrices P (k) .All algorithms are implemented in PyTorch v0.5 (Paszke et al.) .

To leverage the highly efficient NVLink interconnect within each server, we treat each DGX-1 as one node in all of our experiments.

In our implementation of SGP, each node computes a local mini-batch in parallel using all eight GPUs using a local ALLREDUCE, which is efficiently implemented via the NVIDIA Collective Communications Library.

Then inter-node averaging is accomplished using PUSHSUM either over Ethernet or InfiniBand.

In the low-latency experiments, we leverage GPUDirect to directly send/receive messages between GPUs on different nodes and avoid transferring the model back to host memory.

In the high-latency experiments this is not possible, so the model is transferred to host memory after the local ALLREDUCE, and then PUSHSUM messages are sent over Ethernet.

We consider the high-latency scenario where nodes communicate over 10Gbit/s Ethernet.

With a local mini-batch size of 256 samples per node (32 samples per GPU), a single Volta DGX-1 server can perform roughly 4.384 mini-batches per second.

Since the ResNet-50 model size is roughly 100MBytes, transmitting one copy of the model per iteration requires 3.5 Gbit/s.

Thus in the highlatency scenario the problem, if a single 10 Gbit/s link must carry the traffic between more than two pairs of nodes, then communication clearly becomes a bottleneck.

Comparison with synchronous approaches.

We first compare SGP with other synchronous and decentralized approaches.

FIG1 (a) shows the validation curves when training on 4 and 32 nodes (additional training and validation curves for all the training runs can be found in B.1).

Note that when we increase the number of nodes n, we also decrease the total number of iterations K to K/n following Theorem 1 (see Figure B. 3).

For any number of nodes used in our experiments, we observe that SGP consistently outperforms D-PSGD and ALLREDUCE SGD in terms of total training time in this scenario.

In particular for 32 nodes, SGP training time takes less than 1.6 hours while D-PSGD and ALLREDUCE SGD require roughly 2.6 and 5.1 hours.

Appendix B.2 provides experimental evidence that all nodes converge to models with a similar training and validation accuracy when using SGP.

Figure 1 (c) reports the best validation accuracy for the different training runs.

While they all start around the same value, the accuracy of D-PSGD and SGP decreases as we increase the number of nodes.

In the case of SGP, we see its performance decrease by 1.2% relative to SGD on 32 nodes.

We hypothesize that this decrease is due to the noise introduced by approximate distributed averaging.

We will see below than changing the connectivity between the nodes can ameliorate this issue.

We also note that the SGP validation accuracy is better than D-PSGD for larger networks.

Comparison with asynchronous approach.

The results in TAB1 provide a comparison between the aforementioned synchronous methods and AD-PSGD (Lian et al., 2018), a state-ofart asynchronous method.

AD-PSGD is an asynchronous implementation of the doubly-stochastic method D-PSGD, which relies on doubly-stochastic averaging.

All methods are trained for exactly 90 epochs, therefore, the time-per-iteration is a direct reflection of the total training time.

Training using AD-PSGD does not degrade the accuracy (relative to D-PSGD), and provides substantial speedups in training time.

Relative to SGP, the AD-PSGD method runs slightly faster at the expense of lower validation accuracy (except in the 32 nodes case).

In general, we emphasize that this asynchronous line of work is orthogonal, and that by combining the two approaches (leveraging the PUSHSUM protocol in an asynchronous manner), one can expect to further speed up SGP.

We leave this as a promising line of investigation for future work.

We now investigate the behavior of SGP and ALLREDUCE SGD over InfiniBand 100Gbit/s, following the same experimental protocol as in the Ethernet 10Gbit/s case.

In this scenario which is not On this low-latency interconnect, SGD and SGP obtain similar timing and differ at most by 21ms per iteration ( For experiments running at this speed (less than 0.31 seconds per iteration), timing could be impacted by other factors such as data loading.

To better isolate the effects of data-loading, we run additional experiments on 32, 64, and 128 GPUs where we first copied the data locally on every node; see Appendix B.3 for more details.

In that setting, the time-per-iteration of SGP remains approximately constant as we increase the number of nodes in the network, while the time for AllReduceSGD increases with more nodes.

Next we investigate the impact of the communication graph topology on the SGP validation performance using Ethernet 10Gbit/s.

In the limit of a fully-connected communication graph, SGD and SGP are strictly equivalent (see section 3).

By increasing the number of neighbors in the graph, we expect the accuracy of SGP to improve (approximate averages are more accurate) but the communication time required for training will increase.

In FIG5 , we compare the training and validation accuracies of SGP using a communication graph with 1-neighbor and 2-neighbors with D-PSGD and SGD on 32 nodes.

By increasing the number of neighbors to two, SGP achieves better training/validation accuracy (from 74.8/75.0 to 75.6/75.4) and gets closer to final validation achieves by SGD (77.0/76.2).

Increasing the number of neighbors also increases the communication, hence the overall training time.

SGP with 2 neighbors completes training in 2.1 hours and its average time per iteration increases by 27% relative to SGP with one neighbor.

Nevertheless, SGP 2-neighbors is still faster than SGD and D-PSGD, while achieving better accuracy than SGP 1-neighbor.

6 CONCLUSION DNN training often necessistates non-trivial computational requirements leveaging distributed computing resources.

Traditional parallel versions of SGD use exact averaging algorithms to parallelize the computation between nodes, and induce additional parallelization overhead as the model and network sizes grow.

This paper proposes the use of Stochastic Gradient Push for distributed deep learning.

The proposed method computes in-exact averages at each iteartion in order to improve scaling efficiency and reduce the dependency on the underlying network topology.

SGP converges to a stationary point at an O 1/ √ nK rate in the smooth and non-convex case, and proveably achieves a linear speedup (in iterations) with respect to the number of nodes.

Empirical results show that SGP can be up to 3× times faster than traditional ALLREDUCE SGD over high-latency interconnect, matches the top-1 validation accuracy up to 8 nodes (64GPUs), and remains within 1.2% of the top-1 validation accuracy for larger-networks.

For the SGP experiments we use a time-varying directed graph to represent the inter-node connectivity.

Thinking of the nodes as being ordered sequentially, according to their rank, 0, . . .

, n − 1, 1 each node periodically communicates with peers that are 2 0 , 2 1 , . . .

, 2 log 2 (n−1) hops away.

FIG1 shows an example of a directed 8-node exponential graph.

Node 0's 2 0 -hop neighbour is node 1, node 0's 2 1 -hop neighbour is node 2, and node 0's 2 2 -hop neighbour is node 4.In the one-peer-per-node experiments, each node cycles through these peers, transmitting, only, to a single peer from this list at each iteration.

E.g., at iteration k, all nodes transmit messages to their 2 0 -hop neighbours, at iteration k + 1 all nodes transmit messages to their 2 1 -hop neighbours, an so on, eventually returning to the beginning of the list before cycling through the peers again.

This procedure ensures that each node only sends and receives a single message at each iteration.

By using full-duplex communication, sending and receiving can happen in parallel.

In the two-peer-per-node experiments, each node cycles through the same set of peers, transmitting to two peers from the list at each iteration.

E.g., at iteration k, all nodes transmit messages to their 2 0 -hop and 2 1 -hop neighbours, at iteration k + 1 all nodes transmit messages to their 2 1 -hop and 2 2 neighbours, an so on, eventually returning to the beginning of the list before cycling through the peers again.

Similarly, at each iteration, each node also receives, in a full-duplex manner, two messages from some peers that are unknown to the receiving node ahead of time.

Thereby performing the send and receive operations in parallel.

.

Based on the description above, in the one-peer-per-node experiments, each node sends to one neighbor at every iteration, and so each column of P (k) has exactly two nonzero entries, both of which are equal to 1/2.

The diagonal entries p (k) i,i = 1/2 for all i and k. At time step k, each node sends to a neighbor that is 2 k mod log 2 (n−1) hops away.

Thus, with h k = 2 k mod log 2 (n−1) , we get that DISPLAYFORM0 Note that, with this design, in fact each node sends to one peer and receives from one peer at every iteration, so the communication load is balanced across the network.

In the two-peer-per-node experiments, the definition is similar, but now there will be three non-zero entries in each column of P (k) , all of which will be equal to 1/3; these are the diagonal, and the entries corresponding to the two neighbors to which the node sends at that iteration.

In addition, each node will send two messages and receive two messages at every iteration, so the communication load is again balanced across the network.

Undirected exponential graph.

For the D-PSGD experiments we use a time-varying undirected bipartite exponential graph to represent the inter-node connectivity.

Odd-numbered nodes send messages to peers that are 2 1 − 1, 2 2 − 1, . . .

, 2 log 2 (n−1) − 1 (even-numbered nodes), and wait to a receive a message back in return.

Each odd-numbered node cycles through the peers in the list in a similar fashion to the one-peer-per-node SGP experiments.

Even-numbered nodes wait to receive a message from some peer (unknown to the receiving node ahead of time), and send a message back in return.

We adopt these graphs to be consistent with the experimental setup used in Lian et al. FORMULA1 and Lian et al. (2018) .Note also that these graphs are all regular, in that all nodes have the same number of in-coming and out-going connections.

Decentralized averaging errors.

To further motivate our choice of using the directed exponential graph with SGP, let us forget about optimization for a moment and focus on the problem of distributed averaging, described in Section 2, using the PUSHSUM algorithm.

Recall that each node i starts with a vector y (0) i , and the goal of the agents is to compute the average y = DISPLAYFORM1 as its ith row.

Let DISPLAYFORM2 .

The worst-case rate of convergence can be related to the second-largest singular value of P (k−1:0) Nedić et al. (2018) .

In particular, after k iterations we have DISPLAYFORM3 where λ 2 (P (k−1:0) ) denotes the second largest singular value of P (k−1:0) .For the scheme proposed above, cycling deterministically through neighbors in the directed exponential graph, one can verify that after k = log 2 (n − 1) iterations, we have λ 2 (P (k−1:0) ) = 0, so all nodes exactly have the average.

Intuitively, this happens because the directed exponential graph has excellent mixing properties: from any starting node in the network, one can get to any other node in at most log 2 (n) hops.

For n = 32 nodes, after 5 iterations averaging has converged using this strategy.

In comparison, if one were to cycle through edges of the complete graph (where every node is connected to every other node), then for n = 32, after 5 consecutive iterations one would have still have λ 2 (P (k−1:0) ) ≈ 0.6; i.e., nodes could be much further from the average (and hence, much less well-synchronized).Similarly, one could consider designing the matrices P (k) in a stochastic manner, where each node randomly samples one neighbor to send to at every iteration.

If each node samples a destination uniformly from its set of neighbors in the directed exponential graph, then Eλ 2 (P (k−1:0) ) ≈ 0.4, and if each node randomly selected a destination uniformly among all other nodes in the network (i.e., randomly from neighbors in the complete graph), then Eλ 2 (P (k−1:0) ) ≈ 0.2.

Thus, random schemes are still not as effective at quickly averaging as deterministically cycling through neighbors in the directed exponential graph.

Moreover, with randomized schemes, we are no longer guaranteed that each node receives the same number of messages at every iteration, so the communication load will not be balanced as in the deterministic scheme.

The above discussion focused only on approximate distributed averaging, which is a key step within decentralized optimization.

When averaging occurs less quickly, this also impacts optimization.

Specifically, since nodes are less well-synchronized (i.e., further from a consensus), each node will be evaluating its local mini-batch gradient at a different point in parameter space.

Averaging these points (rather than updates based on mini-batch gradients evaluated at the same point) can be seen as injecting additional noise into the optimization process, and in our experience this can lead to worse performance in terms of train and generalization errors.

In all of our experiments, we minimize the number of floating-point operations performed in each iteration, k, by using the mixing weights DISPLAYFORM0 for all i, j = 1, 2, . . .

, n. In words, each node assigns mixing weights uniformly to all of its outneighbors in each iteration.

Recalling our convention that each node is an in-and out-neighbor of itself, it is easy to see that this choice of mixing-weight satisfies the column-stochasticity property.

It may very well be that there is a different choice of mixing-weights that lead to better spectral properties of the gossip algorithm; however we leave this exploration for future work.

We denote node i's uniform mixing weights at time t by p (k) i -dropping the other subscript, which identifies the receiving node.

To maximize the utility of the resources available on each server, each node (occupying a single server exclusively) runs two threads, a gossip thread and a computation thread.

The computation thread executes the main logic used to train the local model on the GPUs available to the node, while the communication thread is used for inter-node network I/O. In particular, the communication thread is used to gossip messages between nodes.

When using Ethernet-based communication, the nodes communicate their parameter tensors over CPUs.

When using InifiniBand-based communication, the nodes communicate their parameter tensors using GPUDirect RDMA, thereby avoiding superfluous device to pinned-memory transfers of the model parameters.

Each node initializes its model on one of its GPUs, and initializes its scalar push-sum weight to 1.

At the start of training, each node also allocates a send-and a receive-communication-buffer in pinned memory on the CPU (or equivalently on a GPU in the case of GPUDirect RDMA communication).In each iteration, the communication thread waits for the send-buffer to be filled by the computation thread; transmits the message in the send-buffer to its out-neighbours; and then aggregates any newly-received messages into the receive-buffer.

In each iteration, the computation thread blocks to retrieve the aggregated messages in the receivebuffer; directly adds the received parameters to its own model parameters; and directly adds the received push-sum weights to its own push-sum weight.

The computation thread then converts the model parameters to the de-biased estimate by dividing by the push-sum weight; executes a forwardbackward pass of the de-biased model in order to compute a stochastic mini-batch gradient; converts the model parameters back to the biased estimate by multiplying by the push-sum weight; and applies the newly-computed stochastic gradients to the biased model.

The updated model parameters are then multiplied by the mixing weight, p (k) i , and asynchronously copied back into the send-buffer for use by the communication thread.

The push-sum weight is also multiplied by the same mixing weight and concatenated into the send-buffer.

In short, gossip is performed on the biased model parameters (push-sum numerators); stochastic gradients are computed using the de-biased model parameters; stochastic gradients are applied back to the biased model parameters; and then the biased-model and the push-sum weight are multiplied by the same uniform mixing-weight and copied back into the send-buffer.

When we "apply the stochastic gradients" to the biased model parameters, we actually carry out an SGD step with nesterov momentum.

For the 32, 64, and 128 GPU experiments we use the same exact learning-rate, schedule, momentum, and weight decay as those suggested in (Goyal et al., 2017) for SGD.

In particular, we use a reference learning-rate of 0.1 with respect to a 256 sample batch, and scale this linearly with the batch-size; we decay the learning-rate by a factor of 10 at epochs 30, 60, 80; we use a nesterov momentum parameter of 0.9, and we use weight decay 0.0001.

For the 256 GPU experiments, we decay the learning-rate by a factor of 10 at epochs 40, 70, 85, and we use a reference learning-rate of 0.0375.

In the 256 GPU experiment with two peers-per-node, we revert to the original learning-rate and schedule.

Require: Initialize γ > 0, m ∈ (0, 1), DISPLAYFORM0 Sample new mini-batch ξ DISPLAYFORM1 Compute a local stochastic mini-batch gradient at z The shaded region shows the maximum and minimum error attained at different nodes in the same experiment.

Although there is non-trivial variability across nodes early in training, all nodes eventually converge to similar validation errors, achieving consensus in the sense that they represent the same function.

Figure B .3 reports the training and validation accuracy of SGP when using a high-latency interconnect.

As we scale up the number of nodes n, we scale down the total number of iterations K to K/n following Theorem 1.

In particular, 32-node runs involves 8 times fewer global iterations than 4-node runs.

We additionally report the total number of iterations and the final performances in Table 3 .

While we reduce the total number iterations by a factor of 8 when going from 4 to 32 nodes, the validation accuracy and training accuracy of the 32 node runs remain within 1.7% and 2.6%, respecively, of the validation and training accuracy achieved by the 4-node runs (and remains within the 1.2% of ALLREDUCE SGD accuracies).

Table 3 : Total number of iterations and final training and validation performances when training a Resnet50 on ImageNet using SGP over Ethernet 10Gbit/s.

DISPLAYFORM2

Here, we investigate the performance variability across nodes during training for SGP.

In figure B .4, we report the minimum, maximum and mean error across the different nodes for training and vali- dation.

In an initial training phase, we observe that nodes have different validation errors; their local copies of the Resnet-50 model diverge.

As we decrease the learning, the variability between the different nodes diminish and the nodes eventually converging to similar errors.

This suggests that all models ultimately represent the same function, achieving consensus.

To better isolate the effects of data-loading, we ran experiments on 32, 64, and 128 GPUs, where we first copied the data locally on every node.

In that setting, we observe in Figure B .5 that the time-per-iteration of SGP remains approximately constant as we increase the number of nodes in the network, while the time for ALLREDUCE SGD increases.

Figure B.6 highlights SGP input images throughput as we scale up the number of cluster node on both Ethernet 10Gbit/s and Infiniband 100Gbit/s.

SGP exhibits 88.6% scaling efficiency on Ethernet 10Gbit/s and 92.4% on InfiniBand and stay close to the ideal scaling in both cases.

In addition Figure (c) shows that SGP exhibit better scaling as we increase the network size and is more robust to high-latency interconnect.

Our convergence rate analysis is divided into three main parts.

In the first one (subsection C.1) we present upper bounds for three important expressions that appear in our computations.

In subsection C.2 we focus on proving the important for our analysis Lemma 8 based on which we later build the proofs of our main Theorems.

Finally in the third part (subsection C.3) we provide the proofs for Theorems 1 and 2.Preliminary results.

In our analysis two preliminary results are extensively used.

We state them here for future reference.• Let a, b ∈ R. Since (a − b) 2 ≥ 0, it holds that DISPLAYFORM0 Thus, x y ≤ ( x 2 + y 2 )/2.• Let r ∈ (0, 1) then from the summation of geometric sequence and for any K ≤ ∞ it holds that DISPLAYFORM1 Matrix Representation.

The presentation of stochastic gradient push (Algorithm 1) was done from node i's perspective for all i ∈ [n].

Note however, that the update rule of SGP at the k th iteration can be viewed from a global viewpoint.

To see this let us define the following matrices (concatenation of the values of all nodes at the k th iteration): DISPLAYFORM2 Using the above matrices, the 6 th step of SGP (Algorithm 1) can be expressed as follows 2 : DISPLAYFORM3 where DISPLAYFORM4 T is the transpose of matrix P k with entries: DISPLAYFORM5 Recall that we also have DISPLAYFORM6 Bound for the mixing matrices.

Next we state a known result from the control literature studying consensus-based optimization which allows us to bound the distance between the de-biased parameters at each node and the node-wise average.

Recall that we have assumed that the sequence of mixing matrices P (k) are B-strongly connected.

A directed graph is called strongly connected if every pair of vertices is connected with a directed path (i.e., following the direction of edges), and the B-strongly connected assumption is that the graph with edge set (l+1)B−1 k=lB E (k) is strongly connected, for every l ≥ 0.We have also assumed that for all k ≥ 0, each column of P (k) has D non-zero entries, and the diameter of the graph with edge set (l+1)B−1 k=lB E (k) has diameter at most ∆. Based on these assumptions, after ∆B consecutive iterations, the product DISPLAYFORM7 has no non-zero entries.

Moreover, every entry of DISPLAYFORM8 Lemma 3.

Suppose that Assumption 3 (mixing connectivity) holds.

Let λ = 1 − nD −∆B and let q = λ 1/(∆B+1) .

Then there exists a constant DISPLAYFORM9 , and x i (0) , such that, for all i = 1, 2, . . .

, n and k ≥ 0, DISPLAYFORM10 This particular lemma follows after a small adaptation to Theorem 1 in BID1 and its proof is based on Wolfowitz (1963) .

Similar bounds appear in a variety of other papers, including Nedić & Olshevsky (2016).

Lemma 4 (Bound of stochastic gradient).

We have the following inequality under Assumptions 1 and 2: DISPLAYFORM0 Proof.

DISPLAYFORM1 Lemma 5.

Let Assumptions 1-3 hold.

Then, DISPLAYFORM2 Proof.

DISPLAYFORM3 Thus, using the above expressions of a, b and c we have that Q (k) i ≤ E(a 2 +b 2 +c 2 +2ab+2bc+2ac).

Let us now obtain bounds for all of these quantities: DISPLAYFORM4 The expression b 1 is bounded as follows: DISPLAYFORM5 Thus, DISPLAYFORM6 where in the first inequality above we use the fact that for q ∈ (0, 1), we have q k < 1 1−q , ∀k > 0.

By identical construction we have DISPLAYFORM7 Now let us bound the products 2ab, 2ac and 2bc.

DISPLAYFORM8 By similar procedure, DISPLAYFORM9 Finally, DISPLAYFORM10 DISPLAYFORM11 By combining all of the above bounds together we obtain: DISPLAYFORM12 After grouping terms together and using the upper bound of Lemma 4, we obtain DISPLAYFORM13 This completes the proof.

Having found a bound for the quantity Q (k)i , let us know present a lemma for bounding the quantity DISPLAYFORM14 Lemma 6.

Let Assumptions 1-3 hold and let us define DISPLAYFORM15 Proof.

Using the bound for Q (k) i let us first bound its average across all nodes M DISPLAYFORM16 At this point note that for any λ ∈ (0, 1), non-negative integer K ∈ N, and non-negative sequence DISPLAYFORM17 Similarly, DISPLAYFORM18 DISPLAYFORM19 Now by summing from k = 0 to K − 1 and using the bounds of FORMULA18 and FORMULA18 we obtain: DISPLAYFORM20 By rearranging: DISPLAYFORM21 Dividing both sides with DISPLAYFORM22 (1 − q) 2 completes the proof.

The goal of this section is the presentation of Lemma 8.

It is the main lemma of our convergence analysis and based on which we build the proofs of Theorems 1 and 2.Let us first state a preliminary lemma that simplifies some of the expressions that involve expectations with respect to the random variable ξi .

Lemma 7.

Under the definition of our problem and the Assumptions 1-3 we have that: DISPLAYFORM0 where in the last equality the inner product becomes zero from the fact that E ξ DISPLAYFORM1 Before present the proof of next lemma let us define the conditional expectation E[ DISPLAYFORM2 [·].

The expectation in this expression is with respect to the random choice ξ (k) i for node i ∈ [n] at the k th iteration.

In other words, F k denotes all the information generated by the stochastic gradient-push algorithm by time t, i.e., all the x DISPLAYFORM3 i ) for k = 1, . . . , t. In addition, we should highlight that the choices of random variables ξ k i ∼ D i , ξ k j ∼ D j at the step t of the algorithm, are independent for any two nodes i = j ∈ [n].

This is also true in the case that the two nodes follow the same distribution D = D i = D j .Lemma 8.

Let Assumptions 1-3 hold and let DISPLAYFORM4 Here C > 0 and q ∈ (0, 1) are the two non-negative constants defined in Lemma 3.

Let {X k } ∞ k=0be the random sequence produced by (9) (Matrix representation of Algorithm 1).

Then, DISPLAYFORM5 By rearranging: DISPLAYFORM6 n .By defining DISPLAYFORM7 the proof is complete.

DISPLAYFORM8 Let us now substitute in the above expression γ = n K .

This can be done due to the lower bound (see equation 6) on the total number of iterations K where guarantees that n K ≤ min (1 − q) 2 60L 2 C 2 , 1 . DISPLAYFORM9 Using again the assumption on the lower bound (6) of the total number of iterations K, the last two terms of the above expression are bounded by the first term.

Thus, DISPLAYFORM10 C.3.2 PROOF OF THEOREM 2Proof.

From Lemma 6 we have that: DISPLAYFORM11 Using the assumptions of Theorem 1 and stepsize γ = n K : DISPLAYFORM12 where the Big O notation swallows all constants of our setting n, L, σ, ζ, C, q, DISPLAYFORM13 where again the Big O notation swallows all constants of our setting n, L, σ, ζ, C, q, DISPLAYFORM14

<|TLDR|>

@highlight

For distributed training over high-latency networks, use gossip-based approximate distributed averaging instead of exact distribute averaging like AllReduce.

@highlight

The authors propose using gossip algorithms as a general method of computing approximate average over a set of workers approximately

@highlight

The paper proves the convergence of SGP for nonconvex smooth functions and shows the SGP can achieve a significant speed-up in the low-latency environment without sacrificing too much predictive performance. 