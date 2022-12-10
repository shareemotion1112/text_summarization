The recently presented idea to learn heuristics for combinatorial optimization problems is promising as it can save costly development.

However, to push this idea towards practical implementation, we need better models and better ways of training.

We contribute in both directions: we propose a model based on attention layers with benefits over the Pointer Network and we show how to train this model using REINFORCE with a simple baseline based on a deterministic greedy rollout, which we find is more efficient than using a value function.

We significantly improve over recent learned heuristics for the Travelling Salesman Problem (TSP), getting close to optimal results for problems up to 100 nodes.

With the same hyperparameters, we learn strong heuristics for two variants of the Vehicle Routing Problem (VRP), the Orienteering Problem (OP) and (a stochastic variant of) the Prize Collecting TSP (PCTSP), outperforming a wide range of baselines and getting results close to highly optimized and specialized algorithms.

Imagine yourself travelling to a scientific conference.

The field is popular, and surely you do not want to miss out on anything.

You have selected several posters you want to visit, and naturally you must return to the place where you are now: the coffee corner.

In which order should you visit the posters, to minimize your time walking around?

This is the Travelling Scientist Problem (TSP).You realize that your problem is equivalent to the Travelling Salesman Problem (conveniently also TSP).

This seems discouraging as you know the problem is (NP-)hard (Garey & Johnson, 1979) .

Fortunately, complexity theory analyzes the worst case, and your Bayesian view considers this unlikely.

In particular, you have a strong prior: the posters will probably be laid out regularly.

You want a special algorithm that solves not any, but this type of problem instance.

You have some months left to prepare.

As a machine learner, you wonder whether your algorithm can be learned?Motivation Machine learning algorithms have replaced humans as the engineers of algorithms to solve various tasks.

A decade ago, computer vision algorithms used hand-crafted features but today they are learned end-to-end by Deep Neural Networks (DNNs).

DNNs have outperformed classic approaches in speech recognition, machine translation, image captioning and other problems, by learning from data (LeCun et al., 2015) .

While DNNs are mainly used to make predictions, Reinforcement Learning (RL) has enabled algorithms to learn to make decisions, either by interacting with an environment, e.g. to learn to play Atari games (Mnih et al., 2015) , or by inducing knowledge through look-ahead search: this was used to master the game of Go (Silver et al., 2017) .The world is not a game, and we desire to train models that make decisions to solve real problems.

These models must learn to select good solutions for a problem from a combinatorially large set of potential solutions.

Classically, approaches to this problem of combinatorial optimization can be divided into exact methods, that guarantee finding optimal solutions, and heuristics, that trade off optimality for computational cost, although exact methods can use heuristics internally and vice versa.

Heuristics are typically expressed in the form of rules, which can be interpreted as policies to make decisions.

We believe that these policies can be parameterized using DNNs, and be trained to obtain new and stronger algorithms for many different combinatorial optimization problems, similar to the way DNNs have boosted performance in the applications mentioned before.

In this paper, we focus on routing problems: an important class of practical combinatorial optimization problems.

The promising idea to learn heuristics has been tested on TSP BID4 .

In order to push this idea, we need better models and better ways of training.

Therefore, we propose to use a powerful model based on attention and we propose to train this model using REINFORCE with a simple but effective greedy rollout baseline.

The goal of our method is not to outperform a nonlearned, specialized TSP algorithm such as Concorde BID0 .

Rather, we show the flexibility of our approach on multiple (routing) problems of reasonable size, with a single set of hyperparameters.

This is important progress towards the situation where we can learn strong heuristics to solve a wide range of different practical problems for which no good heuristics exist.

The application of Neural Networks (NNs) for optimizing decisions in combinatorial optimization problems dates back to Hopfield & Tank (1985) , who applied a Hopfield-network for solving small TSP instances.

NNs have been applied to many related problems (Smith, 1999) , although in most cases in an online manner, starting 'from scratch' and 'learning' a solution for every instance.

More recently, (D)NNs have also been used offline to learn about an entire class of problem instances.

Vinyals et al. (2015) introduce the Pointer Network (PN) as a model that uses attention to output a permutation of the input, and train this model offline to solve the (Euclidean) TSP, supervised by example solutions.

Upon test time, their beam search procedure filters invalid tours.

BID4 introduce an Actor-Critic algorithm to train the PN without supervised solutions.

They consider each instance as a training sample and use the cost (tour length) of a sampled solution for an unbiased Monte-Carlo estimate of the policy gradient.

They introduce extra model depth in the decoder by an additional glimpse (Vinyals et al., 2016) at the embeddings, masking nodes already visited.

For small instances (n = 20), they get close to the results by Vinyals et al. (2015) , they improve for n = 50 and additionally include results for n = 100.

Nazari et al. (2018) replace the LSTM encoder of the PN by element-wise projections, such that the updated embeddings after state-changes can be effectively computed.

They apply this model on the Vehicle Routing Problem (VRP) with split deliveries and a stochastic variant.

Dai et al. (2017) do not use a separate encoder and decoder, but a single model based on graph embeddings.

They train the model to output the order in which nodes are inserted into a partial tour, using a helper function to insert at the best possible location.

Their 1-step DQN (Mnih et al., 2015) training method trains the algorithm per step and incremental rewards provided to the agent at every step effectively encourage greedy behavior.

As mentioned in their appendix, they use the negative of the reward, which combined with discounting encourages the agent to insert the farthest nodes first, which is known to be an effective heuristic (Rosenkrantz et al., 2009) .

Nowak et al. (2017) train a Graph Neural Network in a supervised manner to directly output a tour as an adjacency matrix, which is converted into a feasible solution by a beam search.

The model is non-autoregressive, so cannot condition its output on the partial tour and the authors report an optimality gap of 2.7% for n = 20, worse than autoregressive approaches mentioned in this section.

Kaempfer & Wolf (2018) train a model based on the Transformer architecture (Vaswani et al., 2017) that outputs a fractional solution to the multiple TSP (mTSP).

The result can be seen as a solution to the linear relaxation of the problem and they use a beam search to obtain a feasible integer solution.

Independently of our work, Deudon et al. (2018) presented a model for TSP using attention in the OR community.

They show performance can improve using 2OPT local search, but do not show benefit of their model in direct comparison to the PN.

We use a different decoder and improved training algorithm, both contributing to significantly improved results, without 2OPT and additionally show application to different problems.

For a full discussion of the differences, we refer to Appendix B.4.

We define the Attention Model in terms of the TSP.

For other problems, the model is the same but the input, mask and decoder context need to be defined accordingly, which is discussed in the Appendix.

We define a problem instance s as a graph with n nodes, where node i ∈ {1, . . .

, n} is represented by features x i .

For TSP, x i is the coordinate of node i and the graph is fully connected (with selfconnections) but in general, the model can be considered a Graph Attention Network (Velickovic et al., 2018) and take graph structure into account by a masking procedure (see Appendix A).

We define a solution (tour) π = (π 1 , . . .

, π n ) as a permutation of the nodes, so π t ∈ {1, . . .

n} and π t = π t ∀t = t .

Our attention based encoder-decoder model defines a stochastic policy p(π|s) for selecting a solution π given a problem instance s.

It is factorized and parameterized by θ as p θ (π|s) = n t=1 p θ (π t |s, π 1:t−1 ).(1)The encoder produces embeddings of all input nodes.

The decoder produces the sequence π of input nodes, one node at a time.

It takes as input the encoder embeddings and a problem specific mask and context.

For TSP, when a partial tour has been constructed, it cannot be changed and the remaining problem is to find a path from the last node, through all unvisited nodes, to the first node.

The order and coordinates of other nodes already visited are irrelevant.

To know the first and last node, the decoder context consists (next to the graph embedding) of embeddings of the first and last node.

Similar to BID4 , the decoder observes a mask to know which nodes have been visited.

DISPLAYFORM0 The embeddings are updated using N attention layers, each consisting of two sublayers.

We denote with h ( ) i the node embeddings produced by layer ∈ {1, .., N }.

The encoder computes an aggregated embeddinḡ h (N ) of the input graph as the mean of the final node embeddings h DISPLAYFORM1 .

Both the node embeddings h (N ) i and the graph embeddingh (N ) are used as input to the decoder.

Attention layer Following the Transformer architecture (Vaswani et al., 2017) , each attention layer consist of two sublayers: a multi-head attention (MHA) layer that executes message passing between the nodes and a node-wise fully connected feed-forward (FF) layer.

Each sublayer adds a skip-connection (He et al., 2016) and batch normalization (BN) (Ioffe & Szegedy, 2015 ) (which we found to work better than layer normalization BID1 ): DISPLAYFORM2 The layer index indicates that the layers do not share parameters.

The MHA sublayer uses M = 8 heads with dimensionality DISPLAYFORM3 , and the FF sublayer has one hidden (sub)sublayer with dimension 512 and ReLu activation.

See Appendix A for details.

Decoding happens sequentially, and at timestep t ∈ {1, . . .

n}, the decoder outputs the node π t based on the embeddings from the encoder and the outputs π t generated at time t < t. During decoding, we augment the graph with a special context node (c) to represent the decoding context.

The decoder computes an attention (sub)layer on top of the encoder, but with messages only to the context node for efficiency.1 The final probabilities are computed using a single-head attention mechanism.

See FIG1 for an illustration of the decoding process.

The decoder takes as input the graph embedding and node embeddings.

At each time step t, the context consist of the graph embedding and the embeddings of the first and last (previously output) node of the partial tour, where learned placeholders are used if t = 1.

Nodes that cannot be visited (since they are already visited) are masked.

The example shows how a tour π = (3, 1, 2, 4) is constructed.

Best viewed in color.

Context embedding The context of the decoder at time t comes from the encoder and the output up to time t. As mentioned, for the TSP it consists of the embedding of the graph, the previous (last) node π t−1 and the first node π 1 .

For t = 1 we use learned d h -dimensional parameters v l and v f as input placeholders: DISPLAYFORM0 Here [·, ·, ·] is the horizontal concatenation operator and we write the DISPLAYFORM1 (c) to indicate we interpret it as the embedding of the special context node (c) and use the superscript (N ) to align with the node embeddings h , but we only compute a single query q (c) (per head) from the context node (we omit the (N ) for readability): DISPLAYFORM2 We compute the compatibility of the query with all nodes, and mask (set u (c)j = −∞) nodes which cannot be visited at time t. For TSP, this simply means we mask the nodes already visited: DISPLAYFORM3 Here d k = dh M is the query/key dimensionality (see Appendix A).

Again, we compute u (c)j and v i for M = 8 heads and compute the final multi-head attention value for the context node using equations 12-14 from Appendix A, but with (c) instead of i. This mechanism is similar to our encoder, but does not use skip-connections, batch normalization or the feed-forward sublayer for maximal efficiency.

The result h DISPLAYFORM4 is similar to the glimpse described by BID4 .Calculation of log-probabilities To compute output probabilities p θ (π t |s, π 1:t−1 ) in equation 1, we add one final decoder layer with a single attention head (M = 1 so d k = d h ).

For this layer, we only compute the compatibilities u (c)j using equation 6, but following BID4 we clip the result (before masking!) within [−C, C] (C = 10) using tanh: DISPLAYFORM5 We interpret these compatibilities as unnormalized log-probabilities (logits) and compute the final output probability vector p using a softmax (similar to equation 12 in Appendix A): DISPLAYFORM6 4 REINFORCE WITH GREEDY ROLLOUT BASELINE Section 3 defined our model that given an instance s defines a probability distribution p θ (π|s), from which we can sample to obtain a solution (tour) π|s.

In order to train our model, we define the loss DISPLAYFORM7 the expectation of the cost L(π) (tour length for TSP).

We optimize L by gradient descent, using the REINFORCE (Williams, 1992) gradient estimator with baseline b(s): DISPLAYFORM8 A good baseline b(s) reduces gradient variance and therefore increases speed of learning.

A simple example is an exponential moving average b(s) = M with decay β.

Here M = L(π) in the first iteration and gets updated as M ← βM +(1−β)L(π) in subsequent iterations.

A popular alternative is the use of a learned value function (critic)v(s, w), where the parameters w are learned from the observations (s, L(π)).

However, getting such actor-critic algorithms to work is non-trivial.

We propose to use a rollout baseline in a way that is similar to self-critical training by Rennie et al. (2017) , but with periodic updates of the baseline policy.

It is defined as follows: b(s) is the cost of a solution from a deterministic greedy rollout of the policy defined by the best model so far.

Motivation The goal of a baseline is to estimate the difficulty of the instance s, such that it can relate to the cost L(π) to estimate the advantage of the solution π selected by the model.

We make the following key observation: The difficulty of an instance can (on average) be estimated by the performance of an algorithm applied to it.

This follows from the assumption that (on average) an algorithm will have a higher cost on instances that are more difficult.

Therefore we form a baseline by applying (rolling out) the algorithm defined by our model during training.

To eliminate variance we force the result to be deterministic by selecting greedily the action with maximum probability.

Determining the baseline policy As the model changes during training, we stabilize the baseline by freezing the greedy rollout policy p θ BL for a fixed number of steps (every epoch), similar to freezing of the target Q-network in DQN (Mnih et al., 2015) .

A stronger algorithm defines a stronger baseline, so we compare (with greedy decoding) the current training policy with the baseline policy at the end of every epoch, and replace the parameters θ BL of the baseline policy only if the improvement is significant according to a paired t-test (α = 5%), on 10000 separate (evaluation) instances.

If the baseline policy is updated, we sample new evaluation instances to prevent overfitting.

Analysis With the greedy rollout as baseline b(s), the function L(π) − b(s) is negative if the sampled solution π is better than the greedy rollout, causing actions to be reinforced, and vice versa.

This way the model is trained to improve over its (greedy) self.

We see similarities with selfplay improvement (Silver et al., 2017) : sampling replaces tree search for exploration and the model is rewarded if it yields improvement ('wins') compared to the best model.

Similar to AlphaGo, the evaluation at the end of each epoch ensures that we are always challenged by the best model.

si ← RandomInstance() ∀i ∈ {1, . . .

, B}

πi ← SampleRollout(si, p θ ) ∀i ∈ {1, . . .

, B} 7: Efficiency Each rollout constitutes an additional forward pass, increasing computation by 50%.

However, as the baseline policy is fixed for an epoch, we can sample the data and compute baselines per epoch using larger batch sizes, allowed by the reduced memory requirement as the computations can run in pure inference mode.

Empirically we find that it adds only 25% (see Appendix B.5), taking up 20% of total time.

If desired, the baseline rollout can be computed in parallel such that there is no increase in time per iteration, as an easy way to benefit from an additional GPU.

DISPLAYFORM0

We focus on routing problems: we consider the TSP, two variants of the VRP, the Orienteering Problem and the (Stochastic) Prize Collecting TSP.

These provide a range of different challenges, constraints and objectives and are traditionally solved by different algorithms.

For the Attention Model (AM), we adjust the input, mask, decoder context and objective function for each problem (see Appendix for details and data generation) and train on problem instances of n = 20, 50 and 100 nodes.

For all problems, we use the same hyperparameters: those we found to work well on TSP.Hyperparameters We initialize parameters Uniform( DISPLAYFORM0 , with d the input dimension.

Every epoch we process 2500 batches of 512 instances (except for VRP with n = 100, where we use 2500 × 256 for memory constraints).

For TSP, an epoch takes 5:30 minutes for n = 20, 16:20 for n = 50 (single GPU 1080Ti) and 27:30 for n = 100 (on 2 1080Ti's).

We train for 100 epochs using training data generated on the fly.

We found training to be stable and results to be robust against different seeds, where only in one case (PCTSP with n = 20) we had to restart training with a different seed because the run diverged.

We use N = 3 layers in the encoder, which we found is a good trade-off between quality of the results and computational complexity.

We use a constant learning rate η = 10 −4 .

Training with a higher learning rate η = 10 −3 is possible and speeds up initial learning, but requires decay (0.96 per epoch) to converge and may be a bit more unstable.

See Appendix B.5.

With the rollout baseline, we use an exponential baseline (β = 0.8) during the first epoch, to stabilize initial learning, although in many cases learning also succeeds without this 'warmup'.

Our code in PyTorch (Paszke et al., 2017) is publicly available.

Decoding strategy and baselines For each problem, we report performance on 10000 test instances.

At test time we use greedy decoding, where we select the best action (according to the model) at each step, or sampling, where we sample 1280 solutions (in < 1s on a single GPU) and report the best.

More sampling improves solution quality at increased computation.

In TAB1 we compare greedy decoding against baselines that also construct a single solution, and compare sampling against baselines that also consider multiple solutions, either via sampling or (local) search.

For each problem, we also report the 'best possible solution': either optimal via Gurobi (2018) (intractable for n > 20 except for TSP) or a problem specific state-of-the-art algorithm.

Run times Run times are important but hard to compare: they can vary by two orders of magnitude as a result of implementation (Python vs C++) and hardware (GPU vs CPU).

We take a practical view and report the time it takes to solve the test set of 10000 instances, either on a single GPU (1080Ti) or 32 instances in parallel on a 32 virtual CPU system (2 × Xeon E5-2630).

This is conservative: our model is parallelizable while most of the baselines are single thread CPU implementations which cannot parallelize when running individually.

Also we note that after training our run time can likely be reduced by model compression (Hinton et al., 2015) .

In TAB1 we do not report running times for the results which were reported by others as they are not directly comparable but we note that in general our model and implementation is fast: for instance BID4 report 10.3s for sampling 1280 TSP solutions (K80 GPU) which we do in less than one second (on a 1080Ti).

For most algorithms it is possible to trade off runtime for performance.

As reporting full trade-off curves is impractical we tried to pick reasonable spots, reporting the fastest if results were similar or reporting results with different time limits (for example we use Gurobi with time limits as heuristic).

Travelling Salesman Problem (TSP) For the TSP, we report optimal results by Gurobi, as well as by Concorde BID0 ) (faster than Gurobi as it is specialized for TSP) and LKH3 (Helsgaun, 2017), a state-of-the-art heuristic solver that empirically also finds optimal solutions in time comparable to Gurobi.

We compare against Nearest, Random and Farthest Insertion, as well as Nearest Neighbor, which is the only non-learned baseline algorithm that also constructs a tour directly in order (i.e. is structurally similar to our model).

For details, see Appendix B.3.

Additionally we compare against the learned heuristics in Section 2, most importantly BID4 , as well as OR Tools reported by BID4 and Christofides + 2OPT local search reported by Vinyals (2015) .

Results for Dai et al. (2017) are (optimistically) computed from the optimality gaps they report on 15-20, 40-50 and 50-100 node graphs, respectively.

Using a single greedy construction we outperform traditional baselines and we are able to achieve significantly closer to optimal results than previous learned heuristics (from around 1.5% to 0.3% above optimal for n = 20).

Naturally, the difference with BID4 gets diluted when sampling many solutions (as with many samples even a random policy performs well), but we still obtain significantly better results, without tuning the softmax temperature.

For completeness, we also report results from running the Encode-Attend-Navigate (EAN) code 3 which is concurrent work by Deudon et al. FORMULA10 (for details see Appendix B.4).

Our model outperforms EAN, even if EAN is improved with 2OPT local search.

Appendix B.5 presents the results visually, including generalization results for different n.

Vehicle Routing Problem (VRP) In the Capacitated VRP (CVRP) (Toth & Vigo, 2014) , each node has a demand and multiple routes should be constructed (starting and ending at the depot), such that the total demand of the nodes in each route does not exceed the vehicle capacity.

We also consider the Split Delivery VRP (SDVRP), which allows to split customer demands over multiple routes.

We implement the datasets described by Nazari et al. (2018) and compare against their Reinforcement Learning (RL) framework and the strongest baselines they report.

Comparing greedy decoding, we obtain significantly better results.

We cannot directly compare our sampling (1280 samples) to their beam search with size 10 (they do not report sampling or larger beam sizes), but note that our greedy method also outperforms their beam search in most (larger) cases, getting (in <1 second/instance) much closer to LKH3 (Helsgaun, 2017), a state-of-the-art algorithm which found best known solutions to CVRP benchmarks.

See Appendix C.4 for greedy example solution plots.

Orienteering Problem (OP) The OP (Golden et al., 1987) is an important problem used to model many real world problems.

Each node has an associated prize, and the goal is to construct a single tour (starting and ending at the depot) that maximizes the sum of prizes of nodes visited while being shorter than a maximum (given) length.

We consider the prize distributions proposed in Fischetti et al. (1998) : constant, uniform (in Appendix D.4), and increasing with the distance to the depot, which we report here as this is the hardest problem.

As 'best possible solution' we report Gurobi (intractable for n > 20) and Compass, the recent state-of-the-art Genetic Algorithm (GA) by Kobeaga et al. (2018) , which is only 2% better than sampling 1280 solutions with our method (objective is maximization).

We outperform a Python GA 4 (which seems not to scale), as well the construction phase of the heuristic by Tsiligirides (1984) (comparing greedy or 1280 samples) which is structurally similar to the one learned by our model.

OR Tools fails to find feasible solutions in a few percent of the cases for n > 20.Prize Collecting TSP (PCTSP) In the PCTSP BID3 , each node has not only an associated prize, but also an associated penalty.

The goal is to collect at least a minimum total prize, while minimizing the total tour length plus the sum of penalties of unvisited nodes.

This problem is difficult as an algorithm has to trade off the penalty for not visiting a node with the marginal cost/tour length of visiting (which depends on the other nodes visited), while also satisfying the minimum total prize constraint.

We compare against OR Tools with 10 or 60 seconds of local search, as well as open source C++ 5 and Python 6 implementations of Iterated Local Search (ILS).

Although the Attention Model does not find better solutions than OR Tools with 60s of local search, it finds almost equally good results in significantly less time.

The results are also within 2% of the C++ ILS algorithm (but obtained much faster), which was the best open-source algorithm for PCTSP we could find.

Stochastic PCTSP (SPCTSP) The Stochastic variant of the PCTSP (SPCTSP) we consider shows how our model can deal with uncertainty naturally.

In the SPCTSP, the expected node prize is known upfront, but the real collected prize only becomes known upon visitation.

With penalties, this problem is a generalization of the stochastic k-TSP (Ene et al., 2018 ).

Since our model constructs a tour one node at the time, we only need to use the real prizes to compute the remaining prize constraint.

By contrast, any algorithm that selects a fixed tour may fail to satisfy the prize constraint so an algorithm must be adaptive.

As a baseline, we implement an algorithm that plans a tour, executes part of it and then re-optimizes using the C++ ILS algorithm.

We either execute all node visits (so planning additional nodes if the result does not satisfy the prize constraint), half of the planned node visits (for O(log n) replanning iterations) or only the first node visit, for maximum adaptivity.

We observe that our model outperforms all baselines for n = 20.

We think that failure to account for uncertainty (by the baselines) in the prize might result in the need to visit one or two additional nodes, which is relatively costly for small instances but relatively cheap for larger n. Still, our method is beneficial as it provides competitive solutions at a fraction of the computational cost, which is important in online settings.

Figure 3 : Held-out validation set optimality gap as a function of the number of epochs for the Attention Model (AM) and Pointer Network (PN) with different baselines (two different seeds).

Figure 3 compares the performance of the TSP20 Attention Model (AM) and our implementation of the Pointer Network (PN) during training.

We use a validation set of size 10000 with greedy decoding, and compare to using an exponential (β = 0.8) and a critic (see Appendix B.1) baseline.

We used two random seeds and a decaying learning rate of η = 10 −3 × 0.96 epoch .

This performs best for the PN, while for the AM results are similar to using η = 10 −4 (see Appendix B.5).

This clearly illustrates how the improvement we obtain is the result of both the AM and the rollout baseline: the AM outperforms the PN using any baseline and the rollout baseline improves the quality and convergence speed for both AM and PN.

For the PN with critic baseline, we are unable to reproduce the 1.5% reported by BID4 (also when using an LSTM based critic), but our reproduction is closer than others have reported (Dai et al., 2017; Nazari et al., 2018) .

In TAB1 we compare against the original results.

Compared to the rollout baseline, the exponential baseline is around 20% faster per epoch, whereas the critic baseline is around 13% slower (see Appendix B.5), so the picture does not change significantly if time is used as x-axis.

In this work we have introduced a model and training method which both contribute to significantly improved results on learned heuristics for TSP and additionally learned strong (single construction) heuristics for multiple routing problems, which are traditionally solved by problem-specific approaches.

We believe that our method is a powerful starting point for learning heuristics for other combinatorial optimization problems defined on graphs, if their solutions can be described as sequential decisions.

In practice, operational constraints often lead to many variants of problems for which no good (human-designed) heuristics are available such that the ability to learn heuristics could be of great practical value.

Compared to previous works, by using attention instead of recurrence (LSTMs) we introduce invariance to the input order of the nodes, increasing learning efficiency.

Also this enables parallelization, for increased computational efficiency.

The multi-head attention mechanism can be seen as a message passing algorithm that allows nodes to communicate relevant information over different channels, such that the node embeddings from the encoder can learn to include valuable information about the node in the context of the graph.

This information is important in our setting where decisions relate directly to the nodes in a graph.

Being a graph based method, our model has increased scaling potential (compared to LSTMs) as it can be applied on a sparse graph and operate locally.

Scaling to larger problem instances is an important direction for future research, where we think we have made an important first step by using a graph based method, which can be sparsified for improved computational efficiency.

Another challenge is that many problems of practical importance have feasibility constraints that cannot be satisfied by a simple masking procedure, and we think it is promising to investigate if these problems can be addressed by a combination of heuristic learning and backtracking.

This would unleash the potential of our method, already highly competitive to the popular Google OR Tools project, to an even larger class of difficult practical problems.

A ATTENTION MODEL DETAILS FIG4 : Illustration of weighted message passing using a dot-attention mechanism.

Only computation of messages received by node 1 are shown for clarity.

Best viewed in color.

Attention mechanism We interpret the attention mechanism by Vaswani et al. FORMULA9 as a weighted message passing algorithm between nodes in a graph.

The weight of the message value that a node receives from a neighbor depends on the compatibility of its query with the key of the neighbor, as illustrated in FIG4 .

Formally, we define dimensions d k and d v and compute the key k i ∈ R dk , value v i ∈ R dv and query q i ∈ R dk for each node by projecting the embedding h i : DISPLAYFORM0 From the queries and keys, we compute the compatibility u ij ∈ R of the query q i of node i with the key k j of node j as the (scaled, see Vaswani et al. FORMULA9 ) dot-product: DISPLAYFORM1 In a general graph, defining the compatibility of non-adjacent nodes as −∞ prevents message passing between these nodes.

From the compatibilities u ij , we compute the attention weights a ij ∈ [0, 1] using a softmax: DISPLAYFORM2 Finally, the vector h i that is received by node i is the convex combination of messages v j : DISPLAYFORM3 Multi-head attention As was noted by Vaswani et al. (2017) and Velickovic et al. (2018) , it is beneficial to have multiple attention heads.

This allows nodes to receive different types of messages from different neighbors.

Especially, we compute the value in equation 13 M = 8 times with different parameters, using DISPLAYFORM4 We denote the result vectors by h im for m ∈ 1, . . .

, M .

These are projected back to a single d h -dimensional vector using DISPLAYFORM5 The final multi-head attention value for node i is a function of h 1 , . . .

, h n through h im : DISPLAYFORM6 Feed-forward sublayer The feed-forward sublayer computes node-wise projections using a hidden (sub)sublayer with dimension d ff = 512 and a ReLu activation: DISPLAYFORM7 Batch normalization We use batch normalization with learnable d h -dimensional affine parameters w bn and b bn : DISPLAYFORM8 Here denotes the element-wise product and BN refers to batch normalization without affine transformation.

The critic network architecture uses 3 attention layers similar to our encoder, after which the node embeddings are averaged and processed by an MLP with one hidden layer with 128 neurons and ReLu activation and a single output.

We used the same learning rate as for the AM/PN in all experiments.

For all TSP instances, the n node locations are sampled uniformly at random in the unit square.

This distribution is chosen to be neither easy nor artificially hard and to be able to compare to other learned heuristics.

This section describes details of the heuristics implemented for the TSP.

All of the heuristics construct a single tour in a single pass, by extending a partial solution one node at the time.

Nearest neighbor The nearest neighbor heuristic represents the partial solution as a path with a start and end node.

The initial path is formed by a single node, selected randomly, which becomes the start node but also the end node of the initial path.

In each iteration, the next node is selected as the node nearest to the end node of the partial path.

This node is added to the path and becomes the new end node.

Finally, after all nodes are added this way, the end node is connected with the start node to form a tour.

In our implementation, for deterministic results we always start with the first node in the input, which can be considered random as the instances are generated randomly.

Farthest/nearest/random insertion The insertion heuristics represent a partial solution as a tour, and extends it by inserting nodes one node at the time.

In our implementation, we always insert the node using the cheapest insertion cost.

This means that when node i is inserted, the place of insertion (between adjacent nodes j and k in the tour) is selected such that it minimizes the insertion costs d ji + d ik − d jk , where d ji , d ik and d jk represent the distances from node j to i, i to k and j to k, respectively.

The different variants of the insertion heuristic vary in the way in which the node which is inserted is selected.

Let S be the set of nodes in the partial tour.

Nearest insertion inserts the node i that is nearest to (any node in) the tour: DISPLAYFORM0 Farthest insertion inserts the node i such that the distance to the tour (i.e. the distance from i to the nearest node j in the tour) is maximized: DISPLAYFORM1 Random insertion inserts a random node.

Similar to nearest neighbor, we consider the input order random so we simply insert the nodes in this order.

, 2017) .

There are important differences to this paper:• As 'context' for the decoder, Deudon et al. (2018) use the embeddings of the last K = 3 visited nodes.

We use only the last (e.g. K = 1) node but add the first visited node (as well as the graph embedding), since the first node is important (it is the destination) while the order of the other nodes is irrelevant as we explain in Section 3.• Deudon et al. FORMULA10 use a critic as baseline (which also uses the Transformer architecture).We also experiment with using a critic (based on the Transformer architecture), but found that using a rollout baseline is much more effective (see Section 5).• Deudon et al. FORMULA10 report results with sampling 128 solutions, with and without 2OPT local search.

We report results without 2OPT, using either a single greedy solution or sampling 1280 solutions and additionally show how this directly improves performance compared to BID4 .•

By adding 2OPT on top of the best sampled solution, Deudon et al. (2018) show that the model does not produce a local optimum and results can improve by using a 'hybrid' approach of a learned algorithm with local search.

This is a nice example of combining learned and traditional heuristics, but it is not compared against using the Pointer Network BID4 to eliminate rotation symmetry whereas we directly input node coordinates.• Additionally to TSP, we also consider two variants of VRP, the OP with different prize distributions and the (stochastic) PCTSP.We want to emphasize that this is independent work, but for completeness we include a full emperical comparison of performance.

Since the results presented in the paper by Deudon et al. FORMULA10 are not directly comparable, we ran their code 7 and report results under the same circumstances: using greedy decoding and sampling 1280 solutions on our test dataset (which has exactly the same generative procedure, e.g. uniform in the unit square).

Additionally, we include results of their model with 2OPT, showing that (even without 2OPT) final performance of our model is better.

We use the hyperparameters in their code, but increase the batch size to 512 and number of training steps to 100 × 2500 = 250000 for a fair comparison (this increased the performance of their model).

As training with n = 100 gave out-of-memory errors, we train only on n = 20 and n = 50 and (following Deudon et al. FORMULA10 ) report results for n = 100 using the model trained for n = 50.

The training time as well as test run times are comparable.

Hyperparameters We found in general that using a larger learning rate of 10 −3 works better with decay but may be unstable in some cases.

A smaller learning rate 10 −4 is more stable and does not require decay.

This is illustrated in Figure 6 , which shows validation results over time using both 10 −3 and 10 −4 with and without decay for TSP20 and TSP50 (2 seeds).

As can be seen, without decay the method has not yet fully converged after 100 epochs and results may improve even further with longer training.

TAB3 shows the results in absolute terms as well as the relative optimality gap compared to Gurobi, for all runs using seeds 1234 and 1235 with the two different learning rate schedules.

We did not run final experiments for n = 100 with the larger learning rate as we found training with the smaller learning rate to be more stable.

It can be seen that in most cases the end results with different learning rate schedules are similar, except for the larger models (N = 5, N = 8) where some of the runs diverged using the larger learning rate.

Experiments with different number of layers N show that N = 3 and N = 5 achieve best performance, and we find N = 3 is a good trade-off between quality of the results and computational complexity (runtime) of the model.

Generalization We test generalization performance on different n than trained for, which we plot in Figure 5 in terms of the relative optimality gap compared to Gurobi.

The train sizes are indicated with vertical marker bars.

The models generalize when tested on different sizes, although quality degrades as the difference becomes bigger, which can be expected as there is no free lunch (Wolpert & Macready, 1997) .

Since the architectures are the same, these differences mean the models learn to specialize on the problem sizes trained for.

We can make a strong overall algorithm by selecting the trained model with highest validation performance for each instance size n (marked in Figure 5 by the red bar).

For reference, we also include the baselines, where for the methods that perform search or sampling we do not connect the dots to prevent cluttering and to make the distinction with methods that consider only a single solution clear.

The Capacitated Vehicle Routing Problem (CVRP) is a generalization of the TSP in which case there is a depot and multiple routes should be created, each starting and ending at the depot.

In our graph based formulation, we add a special depot node with index 0 and coordinates x 0 .

A vehicle (route) has capacity D > 0 and each (regular) node i ∈ {1, . . .

n} has a demand 0 < δ i ≤ D. Each route starts and ends at the depot and the total demand in each route should not exceed the capacity, so i∈Rj δ i ≤ D, where R j is the set of node indices assigned to route j. Without loss of generality, we assume a normalizedD = 1 as we can use normalized demandsδ i = δi D .

The Split Delivery VRP (SDVRP) is a generalization of CVRP in which every node can be visited multiple times, and only a subset of the demand has to be delivered at each visit.

Instances for both CVRP and SDVRP are specified in the same way: an instance with size n as a depot location x 0 , n node locations x i , i = 1 . . .

n and (normalized) demands 0 <δ i ≤ 1, i = 1 . . .

n.

We follow Nazari et al. (2018) in the generation of instances for n = 20, 50, 100, but normalize the demands by the capacities.

The depot location as well as n node locations are sampled uniformly at random in the unit square.

The demands are defined asδ i = δi D n where δ i is discrete and sampled uniformly from {1, . . .

, 9} and D 20 = 30, D 50 = 40 and D 100 = 50.

Encoder In order to allow our Attention Model to distinguish the depot node from the regular nodes, we use separate parameters W 0 of the depot node.

Additionally, we provide the normalized demand δ i as input feature (and adjust the size of parameter W x accordingly): DISPLAYFORM0 Capacity constraints To facilitate the capacity constraints, we keep track of the remaining demandsδ i,t for the nodes i ∈ {1, . . .

n} and remaining vehicle capacityD t at time t. At t = 1, these are initialized asδ i,t =δ i andD t = 1, after which they are updated as follows (recall that π t is the index of the node selected at decoding step t): DISPLAYFORM1 If we do not allow split deliveries,δ i,t will be either 0 orδ i for all t.

Decoder context The context for the decoder for the VRP at time t is the current/last location π t−1 and the remaining capacityD t .

Compared to TSP, we do not need placeholders if t = 1 as the route starts at the depot and we do not need to provide information about the first node as the route should end at the depot: DISPLAYFORM2 Masking The depot can be visited multiple times, but we do not allow it to be visited at two subsequent timesteps.

Therefore, in both layers of the decoder, we change the masking for the depot j = 0 and define u (c)0 = −∞ if (and only if) t = 1 or π t−1 = 0.

The masking for the nodes depends on whether we allow split deliveries.

Without split deliveries, we do not allow nodes to be visited if their remaining demand is 0 (if the node was already visited) or exceeds the remaining capacity, so for j = 0 we define u (c)j = −∞ if (and only if)δ i,t = 0 orδ i,t >D t .

With split deliveries, we only forbid delivery when the remaining demand is 0, so we define u (c)j = −∞ if (and only if)δ i,t = 0.Split deliveries Without split deliveries, the remaining demandδ i,t is either 0 orδ i , corresponding to whether the node has been visited or not, and this information is conveyed to the model via the masking of the nodes already visited.

However, when split deliveries are allowed, the remaining demandδ i,t can take any value 0 ≤δ i,t ≤δ i .

This information cannot be included in the context node as it corresponds to individual nodes.

Therefore we include it in the computation of the keys and values in both the attention layer (glimpse) and the output layer of the decoder, such that we compute queries, keys and values using: DISPLAYFORM3 Here we DISPLAYFORM4 ) parameter matrices and we defineδ i,t = 0 for the depot i = 0.

Summing the projection of both h i andδ i,t is equivalent to projecting the concatenation DISPLAYFORM5 However, using this formulation we only need to compute the first term once (instead for every t) and by the weight initialization this puts more importance onδ i,t initially (which is otherwise just 1 of d h + 1 = 129 input values).Training For the VRP, the length of the output of the model depends on the number of times the depot is visited.

In general, the depot is visited multiple times, and in the case of SDVRP also some regular nodes are visited twice.

Therefore the length of the solution is larger than n, which requires more memory such that we find it necessary to limit the batch size B to 256 for n = 100 (on 2 GPUs).

To keep training times tractable and the total number of parameter updates equal, we still process 2500 batches per epoch, for a total of 0.64M training instances per epoch.

For FORMULA2 8 by Helsgaun (2017)

we build and run their code with the SPECIAL parameter as specified in their CVRP runscript 9 .

We perform 1 run with a maximum of 10000 trials, as we found performing 10 runs only marginally improves the quality of the results while taking much more time.

C.4 EXAMPLE SOLUTIONS Figure 7 shows example solutions for the CVRP with n = 100 that were obtained by a single construction using the model with greedy decoding.

These visualizations give insight in the heuristic that the model has learned.

In general we see that the model constructs the routes from the bottom to the top, starting below the depot.

Most routes are densely packed, except for the last route that has to serve some remaining (close to each other) customers.

In most cases, the node in the route that is farthest from the depot is somewhere in the middle of the route, such that customers are served on the way to and from the farthest nodes.

In some cases, we see that the order of stops within some individual routes is suboptimal, which means that the method will likely benefit from simple further optimizations on top, such as a beam search, a post-processing procedure based on local search (e.g. 2OPT) or solving the individual routes using a TSP solver.

DISPLAYFORM0 Figure 7: Example greedy solutions for the CVRP (n = 100).

Edges from and to depot omitted for clarity.

Legend order/coloring and arcs indicate the order in which the solution was generated.

Legends indicate the number of stops, the used and available capacity and the distance per route.

In the Orienteering Problem (OP) each node has a prize ρ i and the goal is to maximize the total prize of nodes visited, while keeping the total length of the route below a maximum length T .

This problem is different from the TSP and the VRP because visiting each node is optional.

Similar to the VRP, we add a special depot node with index 0 and coordinates x 0 .

If the model selects the depot, we consider the route to be finished.

In order to prevent infeasible solutions, we only allow to visit a node if after visiting that node a return to the depot is still possible within the maximum length constraint.

Note that it is always suboptimal to visit the depot if additional nodes can be visited, but we do not enforce this knowledge.

The depot location as well as n node locations are sampled uniformly at random in the unit square.

For the distribution of the prizes, we consider three different variants described by Fischetti et al. (1998) , but we normalize the prizes ρ i such that the normalized prizesρ i are between 0 and 1.Constant ρ i =ρ i = 1.

Every node has the same prize so the goal becomes to visit as many nodes as possible within the length constraint.

DISPLAYFORM0 .

Every node has a prize that is (discretized) uniform.

DISPLAYFORM1 , where d 0i is the distance from the depot to node i. Every node has a (discretized) prize that is proportional to the distance to the depot.

This is designed to be challenging as the largest prizes are furthest away from the depot (Fischetti et al., 1998) .The maximum length T n for instances with n nodes (and a depot) is chosen to be (on average) approximately half of the length of the average TSP tour for uniform TSP instances with n nodes 10 .

This idea is that this way approximately (a little more than) half of the nodes can be visited, which results in the most difficult problem instances (Vansteenwegen et al., 2011) .

This is because the number of possible node selections n k is maximized if k = n 2 and additionally determining the actual path is harder with more nodes selected.

We set fixed maximum lengths T 20 = 2, T 50 = 3 and T 100 = 4 instead of adjusting the constraint per instance, such that for some instances more or less nodes can be visited.

Note that T n has the same unit as the node coordinates x i , so we do not normalize them.

Encoder Similar to the VRP, we use separate parameters for the depot node embedding.

Additionally, we provide the node prizeρ i as input feature: DISPLAYFORM0 Max length constraint In order to satisfy the max length constraint, we keep track of the remaining max length T t at time t. Starting at t = 1, T 1 = T .

Then for t > 0, T is updated as DISPLAYFORM1 Here d πt−1,πt is the distance from node π t−1 to π t and we conveniently define π 0 = 0 as we start at the depot.

Decoder context The context for the decoder for the OP at time t is the current/last location π t−1 and the remaining max length T t .

Similar to VRP, we do not need placeholders if t = 1 as the route starts at the depot and we do not need to provide information about the first node as the route should end at the depot.

We do not need to provide information on the prizes gathered as this is irrelevant for the remaining decisions.

The context is defined as: DISPLAYFORM2 Masking In the OP, the depot node can always be visited so is never masked.

Regular nodes are masked (i.e. cannot be visited) if either they are already visited or if they cannot be visited within the remaining length constraint: Tsiligirides Tsiligirides (1984) describes a heuristic procedure for solving the OP.

It consists of sampling 3000 tours through a randomized construction procedure and applies local search on top.

DISPLAYFORM3 The randomized construction part of the heuristic is structurally exactly the same as the heuristic learned by our model, but with a manually engineered function to define the node probabilities.

We implement the construction part of the heuristic and compare it to our model (either greedy or sampling 1280 solutions), without the local search (as this can also be applied on top of our model).

The final heuristic used by Tsiligirides (1984) uses a formula with multiple terms to define the probability that a node should be selected, but by tuning the weights the form with only one simple term works best, showing the difficulty of manually defining a good probability distribution.

In our terms, the heuristic defines a score s i for each node at time t as the prize divided by the distance from the current node π t−1 , raised to the 4th power: DISPLAYFORM4 Let S be the set with the min(4, n − (t − 1)) unvisited nodes with maximum score s i .

Then the node probabilities p i at time t are defined as DISPLAYFORM5 OR Tools For the Google OR Tools implementation, we modify the formulation for the CVRP 13 :• We replace the Manhattan distance by the Euclidian distance.• We set the number of vehicles to 1.• For each individual node i, we add a Disjunction constraint with {i} as the set of nodes, and a penalty equal to the prizeρ i .

This allows OR tools to skip node i at a costρ i .•

We replace the capacity constraint by a maximum distance.

constraint • We remove the objective to minimize the length.

We multiply all float inputs by 10 7 and round to integers.

Note that OR Tools computes penalties for skipped nodes rather than gains for nodes that are visited.

The problem is equivalent, but in order to compare the objective value against our method, we need to add the constant sum of all penalties iρ i to the OR Tools objective.

Table 3 displays the results for the OP with constant and uniform prize distributions.

The results are similar to the results for the prize distribution based on the distance to the depot, although by the calculation time for Gurobi it is confirmed that indeed constant and uniform prize distributions are easier.

In the Prize Collecting TSP (PCTSP) each node has a prize ρ i and an associated penalty β i .

The goal is to minimize the total length of the tour plus the sum of penalties for nodes which are not visited, while collecting at least a given minimum total prize.

W.l.o.g.

we assume the minimum total prize is equal to 1 (as prizes can be normalized).

This problem is related to the OP but inverts the goal (minimizing tour length given a minimum total prize to collect instead of maximizing total prize given a maximum tour length) and additionally adds penalties.

Again, we add a special depot node with index 0 and coordinates x 0 and if the model selects the depot, the route is finished.

In the PCTSP, it can be beneficial to visit additional nodes, even if the minimum total prize constraint is already satisfied, in order to avoid penalties.

The depot location as well as n node locations are sampled uniformly at random in the unit square.

Similar to the OP, we select the distribution for the prizes and penalties with the idea that for difficult instances approximately half of the nodes should be visited.

Additionally, neither the prize nor the penalty should dominate the node selection process.

Prizes We consider uniformly distributed prizes.

If we sample prizes ρ i ∼ Uniform(0, 1), then E(ρ i ) = 1 2 , and the expected total prize of any subset of n 2 nodes (i.e. half of the nodes) would be n 4 .

Therefore, if S is the set of nodes that is visited, we require that i∈S ρ i ≥ n 4 , or equivalently i∈Sρ i ≥ 1 whereρ i = ρ i · 4 n is the normalized prize.

Note that it can be the case that n i=1ρ i < 1, in which case the prize constraint may be violated but it is only allowed to return to the depot after all nodes have been visited.

Penalties If penalties are too small, then node selection is determined almost entirely by the minimum total prize constraint.

If penalties are too large, we will always visit all nodes, making the minimum total prize constraint obsolete.

We argue that in order for the penalties to be meaningful, they should contribute a term in the objective approximately equal to the total length of the tour.

If L n is the expected TSP tour length with n nodes, we try to achieve this by sampling 14 .

This means that we should sample β i ∼ Uniform(0, 4 · K n n ), but empirically we find thatβ i ∼ Uniform(0, 3 · K n n ) works better, which means that the prizes and penalties are balanced as the minimum total prize constraint is sometimes binding and sometimes not.

DISPLAYFORM0

Encoder Again, we use separate parameters for the depot node embedding.

Additionally, we provide the node prizeρ i and the penaltyβ i as input features: Minimum prize constraint In order to satisfy the minimum total prize constraint, we keep track of the remaining total prize P t to collect at time t. At t = 1, P 1 = 1 (as we normalized prizes).

Then for t > 0, P is updated as

For the SPCTSP, we assume that the real prize collectedρ * i at each node only becomes known when visiting the node, andρ i = E [ρ * i ] is the expected prize.

We assume the real prizes follow a uniform distribution, soρ * i ∼ Uniform(0, 2ρ i ).

In order to apply the Attention Model to the Stochastic PCTSP, the only change we need is that we use the realρ * i to update the remaining prize to collect P t in equation 31: P t+1 = max(0, P t −ρ * πt ).We could theoretically use the model trained for PCTSP without retraining, but we choose to retrain.

This way the model could (for example) learn that if it needs to gather a remaining (normalized) prize of 0.1, it might prefer to visit a node with expected prize 0.2 over a node with expected prize 0.1 as the first real prize will be ≥ 0.1 with probability 75% (uniform prizes) whereas the latter only with 50% and thus has a probability of 50% to not satisfy the constraint.

Instead of sampling the real prizes online, we already sample them when creating the dataset but keep them hidden to the algorithm.

This way, when using a rollout baseline, both the greedy rollout baseline as well as the sample (rollout) from the model use the same real prizes, such that any difference between the two is not a result of stochasticity.

This can be seen as a variant of using Common Random Numbers for variance reduction (Glasserman & Yao, 1992) .

For the SPCTSP, it is not possible to formulate an exact model that constructs a tour offline (as any tour can be infeasible with nonzero probability) and an algorithm that computes the optimal decision online should take into account an infinite number of scenarios.

As a baseline we implement a strategy that:1.

Plans a tour using the expected prizesρ i 2.

Executes part of the tour (not returning to the depot), observing the real prizesρ * i 3.

Computes the remaining total prize that needs to be collected 4.

Computes a new tour (again using expected prizesρ i ), starting from the last node that was visited, through nodes that have not yet been visited and ending at the depot 5.

Repeats the steps (2) -(4) above until the minimum total prize has been collected or all nodes have been visited 6.

Returns to the depot Planning of the tours using deterministic prizes means we need to solve a (deterministic) PCTSP, for which we use the ILS C++ algorithm as this was the strongest algorithm for PCTSP (for large n).

Note that in (4), we have a variant of the PCTSP where we do not have a single depot, but rather separate start and end points, whereas the ILS C++ implementation assumes starting and ending at a single depot.

However, as the ILS C++ implementation uses a distance matrix, we can effectively plan with a start and end node by defining the distance from the 'depot' to node j as the distance from the start node (the last visited node) to node j, whereas we leave the distance from node j to the depot/end node unchanged (so the distance matrix becomes asymmetrical).

Additionally, we remove all nodes (rows/columns in the distance matrix) that have already been visited from the problem.

We consider three variants that differ in the number of nodes that are visited before replanning the tour, for a tradeoff between adaptivity and run time:1.

All nodes in the planned tour are visited (except the final return to the depot).

We only need to replan and visit additional nodes if the constraint is not satisfied, otherwise we return to the depot.

@highlight

Attention based model trained with REINFORCE with greedy rollout baseline to learn heuristics with competitive results on TSP and other routing problems

@highlight

Presents an attention-based approach to learning a policy for solving TSP and other routing-type combinatorial optimzation problems.

@highlight

This paper trys to learn heuristics for solving combinatorial optimisation problems