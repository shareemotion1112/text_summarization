Neuromorphic hardware tends to pose limits on the connectivity of deep networks that one can run on them.

But also generic hardware and software implementations of deep learning run more efficiently for sparse networks.

Several methods exist for pruning connections of a neural network after it was trained without connectivity constraints.

We present an algorithm, DEEP R, that enables us to train directly a sparsely connected neural network.

DEEP R automatically rewires the network during supervised training so that connections are there where they are most needed for the task, while its total number is all the time strictly bounded.

We demonstrate that DEEP R can be used to train very sparse feedforward and recurrent neural networks on standard benchmark tasks with just a minor loss in performance.

DEEP R is based on a rigorous theoretical foundation that views rewiring as stochastic sampling of network configurations from a posterior.

Network connectivity is one of the main determinants for whether a neural network can be efficiently implemented in hardware or simulated in software.

For example, it is mentioned in Jouppi et al. (2017) that in Google's tensor processing units (TPUs), weights do not normally fit in on-chip memory for neural network applications despite the small 8 bit weight precision on TPUs.

Memory is also the bottleneck in terms of energy consumption in TPUs and FPGAs (Han et al., 2017; Iandola et al., 2016) .

For example, for an implementation of a long short term memory network (LSTM), memory reference consumes more than two orders of magnitude more energy than ALU operations (Han et al., 2017) .

The situation is even more critical in neuromorphic hardware, where either hard upper bounds on network connectivity are unavoidable (Schemmel et al., 2010; Merolla et al., 2014) or fast on-chip memory of local processing cores is severely limited, for example the 96 MByte local memory of cores in the SpiNNaker system (Furber et al., 2014) .

This implementation bottleneck will become even more severe in future applications of deep learning when the number of neurons in layers will increase, causing a quadratic growth in the number of connections between them.

Evolution has apparently faced a similar problem when evolving large neuronal systems such as the human brain, given that the brain volume is dominated by white matter, i.e., by connections between neurons.

The solution found by evolution is convincing.

Synaptic connectivity in the brain is highly dynamic in the sense that new synapses are constantly rewired, especially during learning (Holtmaat et al., 2005; Stettler et al., 2006; BID0 BID2 .

In other words, rewiring is an integral part of the learning algorithms in the brain, rather than a separate process.

We are not aware of previous methods for simultaneous training and rewiring in artificial neural networks, so that they are able to stay within a strict bound on the total number of connections throughout the learning process.

There are however several heuristic methods for pruning a larger network (Han et al., 2015b; a; BID4 Yang et al., 2015; Srinivas & Babu, 2015) , that is, the network is first trained to convergence, and network connections and / or neurons are pruned only subsequently.

These methods are useful for downloading a trained network on neuromorphic hardware, but not for on-chip training.

A number of methods have been proposed that are capable of reducing connectivity during training BID4 Jin et al., 2016; Narang et al., This training goal was explored by Welling & Teh (2011) , BID3 , and BID6 where it was shown that gradient descent in combination with stochastic weight updates performs Markov Chain Monte Carlo (MCMC) sampling from the posterior distribution.

In this paper we extend these results by (a) allowing the algorithm also to sample the network structure, and (b) including a hard posterior constraint on the total number of connections during the sampling process.

We define the training goal as follows:produce samples θ with high probability in p * (θ) = 0 if θ violates the constraint DISPLAYFORM0 where Z is a normalizing constant.

The emerging learning dynamics jointly samples from a posterior distribution over network parameters θ and constrained network architectures.

In the next section we introduce the algorithm and in Section 4 we discuss the theoretical guarantees.

The DEEP R algorithm: In many situations, network connectivity is strictly limited during training, for instance because of hardware memory limitations.

Then the limiting factor for a training algorithm is the maximal connectivity ever needed during training.

DEEP R guarantees such a hard limit.

DEEP R achieves the learning goal (1) on network configurations, that is, it not only samples the network weights and biases, but also the connectivity under the given constraints.

This is achieved by introducing the following mapping from network parameters θ to network weights w:A connection parameter θ k and a constant sign s k ∈ {−1, 1} are assigned to each connection k. If θ k is negative, we say that the connection k is dormant, and the corresponding weight is w k = 0.

Otherwise, the connection is considered active, and the corresponding weight is w k = s k θ k .

Hence, each θ k encodes (a) whether the connection is active in the network, and (b) the weight of the connection if it is active.

Note that we use here a single index k for each connection / weight instead of the more usual double index that defines the sending and receiving neuron.

This connectioncentric indexing is more natural for our rewiring algorithms where the connections are in the focus rather than the neurons.

Using this mapping, sampling from the posterior over θ is equivalent to sampling from the posterior over network configurations, that is, the network connectivity structure and the network weights.

while number of active connections lower than K do 7 select a dormant connection k with uniform probability and activate it; 8 θ k ← 0 9 end 10 end Algorithm 1: Pseudo code of the DEEP R algorithm.

ν k is sampled from a zero-mean Gaussian of unit variance independently for each active and each update step.

Note that the gradient of the error E X,Y * (θ) is computed by backpropagation over a mini-batch in practice.

DEEP R is defined in Algorithm 1.

Gradient updates are performed only on parameters of active connections (line 3).

The derivatives of the error function ∂ ∂θ k E X,Y * (θ) can be computed in the usual way, most commonly with the backpropagation algorithm.

Since we consider only classification problems in this article, we used the cross-entropy error for the experiments in this article.

The third term in line 3 (−ηα) is an 1 regularization term, but other regularizers could be used as well.

A conceptual difference to gradient descent is introduced via the last term in line 3.

Here, noise √ 2ηT ν k is added to the update, where the temperature parameter T controls the amount of noise and ν k is sampled from a zero-mean Gaussian of unit variance independently for each parameter and each update step.

The last term alone would implement a random walk in parameter space.

Hence, the whole line 3 of the algorithm implements a combination of gradient descent on the regularized error function with a random walk.

Our theoretical analysis shows that this random walk behavior , test classification accuracy after training for various connectivity levels (middle) and example test accuracy evolution during training (bottom) for a standard feed forward network trained on MNIST (A) and a CNN trained on CIFAR-10 (B).

Accuracies are shown for various algorithms.

Green: DEEP R; red: soft-DEEP R; blue: SGD with initially fixed sparse connectivity; dashed gray: SGD, fully connected.

Since soft-DEEP R does not guarantee a strict upper bound on the connectivity, accuracies are plotted against the highest connectivity ever met during training (middle panels).

Iteration number refers to the number of parameter updates during training.has an important functional consequence, see the paragraph after the next for a discussion on the theoretical properties of DEEP R.The rewiring aspect of the algorithm is captured in lines 4 and 6-9 in Algorithm (1).

Whenever a parameter θ k becomes smaller than 0, the connection is set dormant, i.e., it is deleted from the network and no longer considered for updates (line 4).

For each connection that was set to the dormant state, a new connection k is chosen randomly from the uniform distribution over dormant connections, k is activated and its parameter is initialized to 0.

This rewiring strategy (a) ensures that exactly K connections are active at any time during training (one initializes the network with K active connections), and (b) that dormant connections do not need any computational demands except for drawing connections to be activated.

Note that for sparse networks, it is efficient to keep only a list of active connections and none for the dormant connections.

Then, one can efficiently draw connections from the whole set of possible connections and reject those that are already active.

Rewiring in fully connected and in convolutional networks: We first tested the performance of DEEP R on MNIST and CIFAR-10.

For MNIST, we considered a fully connected feed-forward network used in Han et al. (2015b) to benchmark pruning algorithms.

It has two hidden layers of 300 and 100 neurons respectively and a 10-fold softmax output layer.

On the CIFAR-10 dataset, we used a convolutional neural network (CNN) with two convolutional followed by two fully connected layers.

For reproducibility purposes the network architecture and all parameters of this CNN were taken from the official tutorial of Tensorflow.

On CIFAR-10, we used a decreasing learning rate and a cooling schedule to reduce the temperature parameter T over iterations (see Appendix A for details on all experiments).For each task, we performed four training sessions.

First, we trained a network with DEEP R. In the CNN, the first convolutional layer was kept fully connected while we allowed rewiring of the second convolutional layer.

Second, we tested another algorithm, soft-DEEP R, which is a simplified version of DEEP R that does however not guarantee a strict connectivity constraint (see Section 4 for a description).

Third, we trained a network in the standard manner without any rewiring or pruning to obtain a baseline performance.

Finally, we trained a network with a connectivity that was randomly chosen before training and kept fixed during the optimization.

The connectivity was however not completely random.

Rather each layer received a number of connections that was the same as the number found by soft-DEEP R. The performance of this network is expected to be much better than a network where all layers are treated equally.

FIG1 shows the performance of these algorithms on MNIST (panel A) and on CIFAR-10 (panel B).

DEEP R reaches a classification accuracy of 96.2 % when constrained to 1.3 % connectivity.

To evaluate precisely the accuracy that is reachable with 1.0 % connectivity, we did an additional experiment where we doubled the number of training epochs.

DEEP R reached a classification accuracy of 96.3% (less than 2 % drop in comparison to the fully connected baseline).

Training on fixed random connectivity performed surprisingly well for connectivities around 10 %, possibly due to the large redundancy in the MNIST images.

Soft-DEEP R does not guarantee a strict upper bound on the network connectivity.

When considering the maximum connectivity ever seen during training, soft-DEEP R performed consistently worse than DEEP R for networks where this maximum connectivity was low.

On CIFAR-10, the classification accuracy of DEEP R was 84.1 % at a connectivity level of 5 %.

The performance of DEEP R at 20 % connectivity was close to the performance of the fully connected network.

To study the rewiring properties of DEEP R, we monitored the number of newly activated connections per iteration (i.e., connections that changed their status from dormant to active in that iteration).

We found that after an initial transient, the number of newly activated connections converged to a stable value and remained stable even after network performance has converged, see Appendix B.Rewiring in recurrent neural networks: In order to test the generality of our rewiring approach, we also considered the training of recurrent neural networks with backpropagation through time (BPTT).

Recurrent networks are quite different from their feed forward counterparts in terms of their dynamics.

In particular, they are potentially unstable due to recurrent loops in inference and training signals.

As a test bed, we considered an LSTM network trained on the TIMIT data set.

In our rewiring algorithms, all connections were potentially available for rewiring, including connections to gating units.

From the TIMIT audio data, MFCC coefficients and their temporal derivatives were computed and fed into a bi-directional LSTMs with a single recurrent layer of 200 cells followed by a softmax to generate the phoneme likelihood (Graves & Schmidhuber, 2005) , see Appendix A.We considered as first baseline a fully connected LSTM with standard BPTT without regularization as the training algorithm.

This algorithm performed similarly as the one described in Greff et al. (2017) .

It turned out however that performance could be significantly improved by including a regularizer in the training objective.

We therefore considered the same setup with 2 regularization (cross-validated).

This setup achieved a phoneme error rate of 28.3 %.

We note that better results have been reported in the literature using the CTC cost function and deeper networks (Graves et al., 2013) .

For the sake of easy comparison however, we sticked here to the much simpler setup with a medium-sized network and the standard cross-entropy error function.

We found that connectivity can be reduced significantly in this setup with our algorithms, see FIG2 .

Both algorithms, DEEP R and soft-DEEP R, performed even slightly better than the fully connected baseline at connectivities around 10 %, probably due to generalization issues.

DEEP R outperformed soft-DEEP R at very low connectivities and it outperformed BPTT with fixed random connectivity consistently at any connectivity level considered.

Comparison to algorithms that cannot be run on very sparse networks: We wondered how much performance is lost when a strict connectivity constraint has to be taken into account during training as compared to pruning algorithms that only achieve sparse networks after training.

To this end, we compared the performance of DEEP R and soft-DEEP R to recently proposed pruning algorithms: 1 -shrinkage (Tibshirani, 1996; BID4 and the pruning algorithm proposed by Han et al. (2015b) .

1 -shrinkage uses simple 1 -norm regularization and finds network solutions with a connectivity that is comparable to the state of the art BID4 Yu et al., 2012) .

We chose this one since it is relatively close to DEEP R with the difference that it does not implement rewiring.

The pruning algorithm from Han et al. FORMULA0 is more complex and uses a projection of network weights on a 0 constraint.

Both algorithms prune connections starting from the fully connected network.

The hyper-parameters such as learning rate, layer size, and weight decay coefficients were kept the same in all experiments.

We validated by an extensive parameter search that these settings were good settings for the comparison algorithms, see Appendix A.Results for the same setups as considered above (MNIST, CIFAR-10, TIMIT) are shown in FIG3 .

Despite the strict connectivity constraints, DEEP R and soft-DEEP R performed slightly better than the unconstrained pruning algorithms on CIFAR-10 and TIMIT at all connectivity levels considered.

On MNIST, pruning was slightly better for larger connectivities.

On MNIST and TIMIT, pruning and 1 -shrinkage failed completely for very low connectivities while rewiring with DEEP R or soft-DEEP R still produced reasonable networks in this case.

One interesting observation can be made for the error rate evolution of the LSTM on TIMIT FIG3 .

Here, both 1 -shrinkage and pruning induced large sudden increases of the error rate, possibly due to instabilities induced by parameter changes in the recurrent network.

In contrast, we observed only small glitches of this type in DEEP R. This indicates that sparsification of network connectivity is harder in recurrent networks due to potential instabilities, and that DEEP R is better suited to avoid such instabilities.

The reason for this advantage of DEEP R is however not clear.

Transfer learning is supported by DEEP R: If the temperature parameter T is kept constant during training, the proposed rewiring algorithms do not converge to a static solution but explore continuously the posterior distribution of network configurations.

As a consequence, rewiring is expected to adapt to changes in the task in an on line manner.

If the task demands change in an online learning setup, one may hope that a transfer of invariant aspects of the tasks occurs such that these aspects can be utilized for faster convergence on later tasks (transfer learning).

To verify this hypothesis, we performed one experiment on the MNIST dataset where the class to which each output neuron should respond to was changed after each training epoch (class-shuffled MNIST task).

FIG4 shows the performance of a network trained with DEEP R in the class-shuffled MNIST task.

One can observe that performance recovered after each shuffling of the target classes.

More importantly, we found a clear trend of increasing classification accuracy even across shuffles.

This indicates a form of transfer learning in the network such that information about the previous tasks ) and 1 -shrinkage (Tibshirani, 1996; BID4 .

A, B) Accuracy against the connectivity for MNIST (A) and CIFAR-10 (B).

For each algorithm, one network with a decent compromise between accuracy and sparsity is chosen (small gray boxes) and its connectivity across training iterations is shown below.

C) Performance on the TIMIT dataset.

D) Phoneme error rates and connectivities across iteration number for representative training sessions.(i.e., the previous target-shuffled MNIST instances) was preserved in the network and utilized in the following instances.

We hypothesized for the reason of this transfer that early layers developed features that were invariant to the target shuffling and did not need to be re-learned in later task instances.

To verify this hypothesis, we computed the following two quantities.

First, in order to quantify the speed of parameter dynamics in different layers, we computed the correlation between the layer weight matrices of two subsequent training epoch FIG4 ).

Second, in order to quantify the speed of change of network dynamics in different layers, we computed the correlation between the neuron outputs of a layer in subsequent epochs FIG4 ).

We found that the correlation between weights and layer outputs increased across training epochs and were significantly larger in early layers.

This supports the hypothesis that early network layers learned features invariant to the shuffled coding convention of the output layer.

The theoretical analysis of DEEP R is somewhat involved due to the implemented hard constraints.

We therefore first introduce and discuss here another algorithm, soft-DEEP R where the theoretical treatment of convergence is more straight forward.

In contrast to standard gradient-based algorithms, this convergence is not a convergence to a particular parameter vector, but a convergence to the target distribution over network configurations.

DISPLAYFORM0 Algorithm 2: Pseudo code of the soft-DEEP R algorithm.

θ min < 0 is a constant that defines a lower boundary for negative θ k s.

The soft-DEEP R algorithm is given in Algorithm 2.

Note that the updates for active connections are the same as for DEEP R (line 3).

Also the mapping from parameters θ k to weights w k is the same as in DEEP R. The main conceptual difference to DEEP R is that connection parameters continue their random walk when dormant (line 7).

Due to this random walk, connections will be re-activated at random times when they cross zero.

Therefore, soft-DEEP R does not impose a hard constraint on network connectivity but rather uses the 1 norm regularization to impose a soft-constraint.

Since dormant connections have to be simulated, this algorithm is computationally inefficient for sparse networks.

An approximation could be used where silent connections are re-activated at a constant rate, leading to an algorithm very similar to DEEP R. DEEP R adds to that the additional feature of a strict connectivity constraint.

The central result for soft-DEEP R has been proven in the context of spiking neural networks in BID6 in order to understand rewiring in the brain from a functional perspective.

The same theory however also applies to standard deep neural networks.

To be able to apply standard mathematical tools, we consider parameter dynamics in continuous time.

In particular, consider the following stochastic differential equation (SDE) DISPLAYFORM0 where β is the equivalent to the learning rate and DISPLAYFORM1 denotes the gradient of the log parameter posterior evaluated at the parameter vector θ t at time t. The term dW k denotes the infinitesimal updates of a standard Wiener process.

This SDE describes gradient ascent on the log posterior combined with a random walk in parameter space.

We show in Appendix C that the unique stationary distribution of this parameter dynamics is given by DISPLAYFORM2 Since we considered classification tasks in this article, we interpret the network output as a multinomial distribution over class labels.

Then, the derivative of the log likelihood is equivalent to the derivative of the negative cross-entropy error.

Together with an 1 regularization term for the prior, and after discretization of time, we obtain the update of line 3 in Algorithm 2 for non-negative parameters.

For negative parameters, the first term in Eq. (2) vanishes since the network weight is constant zero there.

This leads to the update in line 7.

Note that we introduced a reflecting boundary at θ min < 0 in the practical algorithm to avoid divergence of parameters (line 8).Convergence properties of DEEP R: A detailed analysis of the stochastic process that underlies the algorithm is provided in Appendix D. Here we summarize the main findings.

Each iteration of DEEP R in Algorithm 1 consists of two parts: In the first part (lines 2-5) all connections that are currently active are advanced, while keeping the other parameters at 0.

In the second part (lines 6-9) the connections that became dormant during the first step are randomly replenished.

To describe the connectivity constraint over connections we introduce the binary constraint vector c which represents the set of active connections, i.e., element c k of c is 1 if connection k is allowed to be active and zero else.

In Theorem 2 of Appendix D, we link DEEP R to a compound Markov chain operator that simultaneously updates the parameters θ according to the soft-DEEP R dynamics under the constraint c and the constraint vector c itself.

The stationary distribution of this Markov chain is given by the joint probability DISPLAYFORM3 where C(θ, c) is a binary function that indicates compatibility of θ with the constraint c and p * (θ) is the tempered posterior of Eq. (3) which is left stationary by soft-DEEP R in the absence of constraints.

p C (c) in Eq. (4) is a uniform prior over all connectivity constraints with exactly K synapses that are allowed to be active.

By marginalizing over c, we obtain that the posterior distribution of DEEP R is identical to that of soft-DEEP R if the constraint on the connectivity is fulfilled.

By marginalizing over θ, we obtain that the probability of sampling a network architecture (i.e. a connectivity constraint c) with DEEP R and soft-DEEP R are proportional to one another.

The only difference is that DEEP R exclusively visits architectures with K active connections (see equation FORMULA4 in Appendix D for details).In other words, DEEP R solves a constraint optimization problem by sampling parameter vectors θ with high performance within the space of constrained connectivities.

The algorithm will therefore spend most time in network configurations where the connectivity supports the desired network function, such that, connections with large support under the objective function (1) will be maintained active with high probability, while other connections are randomly tested and discarded if found not useful.

Related Work: de Freitas et al. FORMULA2 considered sequential Monte Carlo sampling to train neural networks by combining stochastic weight updates with gradient updates.

Stochastic gradient updates in mini-batch learning was considered in Welling & Teh (2011) , where also a link to the true posterior distribution was established.

BID3 proposed a momentum scheme and temperature annealing (for the temperature T in our notation) for stochastic gradient updates, leading to a stochastic optimization method.

DEEP R extends this approach by using stochastic gradient Monte Carlo sampling not only for parameter updates but also to sample the connectivity of the network.

In addition, the posterior in DEEP R is subject to a hard constraint on the network architecture.

In this sense, DEEP R performs constrained sampling, or constrained stochastic optimization if the temperature is annealed.

Patterson & Teh (2013) considered the problem of stochastic gradient dynamics constrained to the probability simplex.

The methods considered there are however not readily applicable to the problem of constraints on the connection matrix considered here.

Additionally, we show that a correct sampler can be constructed that does not simulate dormant connections.

This sampler is efficient for sparse connection matrices.

Thus, we developed a novel method, random reintroduction of connections, and analyzed its convergence properties (see Theorem 2 in Appendix D).

We have presented a method for modifying backprop and backprop-through-time so that not only the weights of connections, but also the connectivity graph is simultaneously optimized during training.

This can be achieved while staying always within a given bound on the total number of connections.

When the absolute value of a weight is moved by backprop through 0, it becomes a weight with the opposite sign.

In contrast, in DEEP R a connection vanishes in this case (more precisely: becomes dormant), and a randomly drawn other connection is tried out by the algorithm.

This setup requires that, like in neurobiology, the sign of a weight does not change during learning.

Another essential ingredient of DEEP R is that it superimposes the gradient-driven dynamics of each weight with a random walk.

This feature can be viewed as another inspiration from neurobiology (Mongillo et al., 2017 ).

An important property of DEEP R is that -in spite of its stochastic ingredient -its overall learning dynamics remains theoretically tractable: Not as gradient descent in the usual sense, but as convergence to a stationary distribution of network configurations which assigns the largest probabilities to the best-performing network configurations.

An automatic benefit of this ongoing stochastic parameter dynamics is that the training process immediately adjusts to changes in the task, while simultaneously transferring previously gained competences of the network (see FIG4

Implementations of DEEP R are freely available at github.com/guillaumeBellec/deep rewiring.

Choosing hyper-parameters for DEEP R: The learning rate η is defined for each task independently (see task descriptions below).

Considering that the number of active connections is given as a constraint, the remaining hyper parameters are the regularization coefficient α and the temperature T .

We found that the performance of DEEP R does not depend strongly on the temperature T .

Yet, the choice of α has to be done more carefully.

For each dataset there was an ideal value of α: one order of magnitude higher or lower typically lead to a substantial loss of accuracy.

In MNIST, 96.3% accuracy under the constraint of 1% connectivity was achieved with α = 10 −4and T chosen so that T = η 2 10 −12 .

In TIMIT, α = 0.03 and T = 0 (higher values of T could improve the performance slightly but it did not seem very significant).

In CIFAR-10 a different α was assigned to each connectivity matrix.

To reach 84.1% accuracy with 5% connectivity we used in each layer from input to output α = [0, 10 −7 , 10 −6 , 10 −9 , 0].

The temperature is initialized with DISPLAYFORM0 18 and decays with the learning rate (see paragraph of the methods about CIFAR-10).

The main difference between soft-DEEP R and DEEP R is that the connectivity is not given as a global constraint.

This is a considerable drawback if one has strict constraint due to hardware limitation but it is also an advantage if one simply wants to generate very sparse network solutions without having a clear idea on the connectivities that are reachable for the task and architecture considered.

In any cases, the performance depends on the choice of hyper-parameters α, T and θ min , but alsounlike in DEEP R -these hyper parameters have inter-dependent relationships that one cannot ignore (as for DEEP R, the learning rate η is defined for each task independently).

The reason why soft-DEEP R depends more on the temperature is that the rate of re-activation of connections is driven by the amplitude of the noise whereas they are decoupled in DEEP R. To summarize the results of an exhaustive parameter search, we found that √ 2T η should ideally be slightly below α.

In general high θ min leads to high performance but it also defines an approximate lower bound on the smallest reachable connectivity.

This lower bound can be estimated by computing analytically the stationary distribution under rough approximations and the assumption that the gradient of the likelihood is zero.

If p min is the targeted lower connectivity bound, one needs θ min ≈ − DISPLAYFORM0 For MNIST we used α = 10 −5 and T = η α 2 18 for all data points in FIG1 panel A and a range of values of θ min to scope across different ranges of connectivity lower bounds.

In TIMIT and CIFAR-10 we used a simpler strategy which lead to a similar outcome, we fixed the relationships: α = 3 2 DISPLAYFORM1 3 θ min and we varied only α to produce the solutions shown in FIG1 Re-implementing pruning and 1 -shrinkage: To implement 1 -shrinkage (Tibshirani, 1996; Collins & Kohli, 2014), we applied the 1 -shrinkage operator θ ← relu (|θ| − ηα) sign(θ) after each gradient descent iteration.

The performance of the algorithm is evaluated for different α varying on a logarithmic scale to privilege a sparse connectivity or a high accuracy.

For instance for MNIST in FIG3 .A we used α of the form 10 − n 2 with n going from 4 to 12.

The optimal parameter was n = 9.

We implemented the pruning described in Han et al. (2015b) .

This algorithm uses several phases: training -pruning -training, or one can also add another pruning iteration: training -pruning -training -pruning -training.

We went for the latter because it increased performance.

Each "training" phase is a complete training of the neural network with 2 -regularization 1 .

At each "pruning" phase, the standard deviation of weights within a weight matrix w std is computed and all active weights with absolute values smaller than qw std are pruned (q is called the quality parameter).

Grid search is pruning quality factor q pruning quality factor q 1.5 1.5Figure 5: Hyper-parameter search for the pruning algorithm according to Han et al. (2015b) .

Each point of the grid represents a weight decay coefficient -quality factor pair.

The number and the color indicate the performance in terms of accuracy (left) or connectivity (right).

The red rectangle indicates the data points that were used in FIG3 .used to optimize the 2 -regularization coefficient and quality parameter.

The results for MNIST are reported in Figure 5 .MNIST: We used a standard feed forward network architecture with two hidden layers with 200 neurons each and rectified linear activation functions followed by a 10-fold softmax output.

For all algorithms we used a learning rate of 0.05 and a batch size of 10 with standard stochastic gradient descent.

Learning stopped after 10 epochs.

All reported performances in this article are based on the classification error on the MNIST test set.

The official tutorial for convolutional networks of tensorflow 2 is used as a reference implementation.

Its performance out-of-the-box provides the fully connected baseline.

We used the values given in the tutorial for the hyper-parameters in all algorithms.

In particular the layer-specific weight decay coefficients that interact with our algorithms were chosen from the tutorial for DEEP R, soft-DEEP R, pruning, and 1 -shrinkage.

In the fully connected baseline implementation, standard stochastic gradient descent was used with a decreasing learning rate initialized to 1 and decayed by a factor 0.1 every 350 epochs.

Training was performed for one million iterations for all algorithms.

For soft-DEEP R, which includes a temperature parameter, keeping a high temperature as the weight decays was increasing the rate of re-activation of connections.

Even if intermediate solutions were rather sparse and efficient the solutions after convergence were always dense.

Therefore, the weight decay was accompanied by annealing of the temperature T .

This was done by setting the temperature to be proportional to the decaying η.

This annealing was used for DEEP R and soft-DEEP R.TIMIT: The TIMIT dataset was preprocessed and the LSTM architecture was chosen to reproduce the results from Greff et al. (2017) .

Input time series were formed by 12 MFCC coefficients and the log energy computed over each time frame.

The inputs were then expanded with their first and second temporal derivatives.

There are 61 different phonemes annotated in the TIMIT dataset, to report an error rate that is comparable to the literature we performed a standard grouping of the phonemes to generate 39 output classes (Lee & Hon, 1989; Graves et al., 2013; Greff et al., 2017) .

As usual, the dialect specific sentences were excluded (SA files).

The phoneme error rate was computed as the proportion of misclassified frames.

A validation set and early stopping were necessary to train a network with dense connectivity matrix on TIMIT because the performance was sometimes unstable and it suddenly dropped during training as seen in FIG3 for 1 -shrinkage.

Therefore a validation set was defined by randomly selecting 5% of the training utterances.

All algorithms were trained for 40 epochs and the reported test error rate is the one at minimal validation error.

To accelerate the training in comparison the reference from Greff et al. FORMULA0 we used mini-batches of size 32 and the ADAM optimizer (Kingma & Ba (2014) ).

This was also an opportunity to test the performance of DEEP R and soft-DEEP R with such a variant of gradient descent.

The learning rate was set to 0.01 and we kept the default momentum parameters of ADAM, yet we found that changing the parameter (as defined in Kingma & Ba (2014) ) from 10 −8 to 10 −4 improved the stability of fully connected networks during training in this recurrent setup.

As we could not find a reference that implemented 1 -shrinkage in combination with ADAM, we simply applied the shrinkage operator after each iteration of ADAM which might not be the ideal choice in theory.

It worked well in practice as the minimal error rate was reached with this setup.

The same type of 1 regularization in combination with ADAM was used for DEEP R and soft-DEEP R which lead to very sparse and efficient network solutions.

Initialization of connectivity matrices: We found that the performance of the networks depended strongly on the initial connectivity.

Therefore, we followed the following heuristics to generate initial connectivity for DEEP R, soft-DEEP R and the control setup with fixed connectivity.

First, for the connectivity matrix of each individual layer, the zero entries were chosen with uniform probability.

Second, for a given connectivity constraint we found that the learning time increased and the performance dropped if the initial connectivity matrices were not chosen carefully.

Typically the performance dropped drastically if the output layer was initialized to be very sparse.

Yet in most networks the number of parameters is dominated by large connectivity matrices to hidden layers.

A basic rule of thumb that worked in our cases was to give an equal number of active connections to the large and intermediate weight matrices, whereas smaller ones -typically output layers -should be densely connected.

We suggest two approaches to refine this guess: One can either look at the statistics of the connectivity matrices after convergence of DEEP R or soft-DEEP R, or, if possible, the second alternative is to initialize once soft-DEEP R with a dense matrix and observe the connectivity matrix after convergence.

In our experiments the connectivities after convergence were coherent with the rule of thumb described above and we did not need to pursue intensive search for ideal initial connectivity matrices.

For MNIST, the number of parameters in each layer was 235k, 30k and 1k from input to output.

Using our rule of thumb, for a given global connectivity p 0 , the layers were respectively initialized with connectivity 0.75p 0 , 2.3p 0 and 22.8p 0 .For CIFAR-10, the baseline network had two convolutional layers with filters of shapes 5×5×3×64 and 5×5×64×64 respectively, followed by two fully connected layer with weight matrices of shape 2304 × 384 and 384 × 192.

The last layer was then projected into a softmax over 10 output classes.

The numbers of parameters per connectivity matrices were therefore 5k, 102k, 885k, 738k and 2k from input to output.

The connectivity matrices were initialized with connectivity 1, 8p 0 , 0.8p 0 , 8p 0 , and 1.For TIMIT, the connection matrix from the input to the hidden layer was of size 39 × 800, the recurrent matrix had size 200 × 800 and the size of the output matrix was 200 × 39.

Each of these three connectivity matrices were initialized with a connectivity of 3p 0 , p 0 , and 10p 0 respectively.

Initialization of weight matrices: For CIFAR-10 the initialization of matrix coefficients was given by the reference implementation.

For MNIST and TIMIT, the weight matrices were initialized with θ = 1 √ nin N (0, 1)c where n in is the number of afferent neurons, N (0, 1) samples from a centered gaussian with unit variance and c is a binary connectivity matrix.

It would not be good to initialize the parameters of all dormant connections to zero in soft-DEEP R. After a single noisy iteration, half of them would become active which would fail to initialize the network with a sparse connectivity matrix.

To balance out this problem we initialized the parameters of dormant connections uniformly between the clipping value θ min and zero in soft-DEEP R. FIG4 The experiment provided in FIG4 is a variant of our MNIST experiment where the target labels were shuffled after every training epoch.

To make the generalization capability of DEEP R over a small number of epochs visible, we enhanced the noise exploration by setting a batch to 1 so that the connectivity matrices were updated at every time step.

Also we used new to layer l = 1 (brown), l = 2 (orange), and the output layer (l = 3, gray) per iteration.

Note that these layers have quite different numbers of potential connections K (l) .

C) Same as panel B but the number of newly activated connections are shown relative to the number of potential connections in the layer (values in panel C are smoothed with a boxcar filter over X iterations).

a larger network with 400 neurons in each hidden layer.

The remaining parameters were similar to those used previously: the connectivity was constrained to 1% and the connectivity matrices were initialized with respective connectivities: 0.01, 0.01, and 0.1.

The parameters of DEEP R were set to η = 0.05, α = 10 −5 and T = η B REWIRING DURING TRAINING ON MNIST FIG7 shows the rewiring behavior of DEEP R per network layer for the feed-forward neural network trained on MNIST and the training run indicated by the small gray box around the green dot in FIG1 .

Since it takes some iterations until the weights of connections that do not contribute to a reduction of the error are driven to 0, the number of newly established connections K

new in layer l is small for all layers initially.

After this initial transient, the number of newly activated connections stabilized to a value that is proportional to the total number of potential connections in the layer FIG1 .

DEEP R continued to rewire connections even late in the training process.

Here we provide additional details on the convergence properties of the soft-DEEP R parameter update provided in Algorithm 2.

We reiterate here Eq. (2): DISPLAYFORM0 Discrete time updates can be recovered from the set of SDEs (5) by integration over a short time period ∆t DISPLAYFORM1 where the learning rate η is given by η = β ∆t.

We prove that the stochastic parameter dynamics Eq. (5) converges to the target distribution p * (θ) given in Eq. (3).

The proof is analogous to the derivation given in BID6 .

We reiterate the proof here for the special case of supervised learning.

The fundamental property of the synaptic sampling dynamics Eq. FORMULA10 is formalized in Theorem 1 and proven below.

Before we state the theorem, we briefly discuss its statement in simple terms.

Consider some initial parameter setting θ 0 .

Over time, the parameters change according to the dynamics (5).

Since the dynamics include a noise term, the exact value of the parameters θ(t) at some time t > 0 cannot be determined.

However, it is possible to describe the exact distribution of parameters for each time t. We denote this distribution by p FP (θ, t), where the "FP" subscript stands for "Fokker-Planck" since the evolution1 T , i.e., p * (θ) is the only solution for which ∂ ∂t p FP (θ, t) becomes 0, which completes the proof.

The updates of the soft-DEEP R algorithm (Algorithm 2) can be written as DISPLAYFORM2 Eq. FORMULA12 is a special case of the general discrete parameter dynamics (6).

To see this we apply Bayes' rule to expand the derivative of the log posterior into the sum of the derivatives of the prior and the likelihood: DISPLAYFORM3 such that we can rewrite Eq. (6) DISPLAYFORM4 To include automatic network rewiring in our deep learning model we adopt the approach described in BID6 .

Instead of using the network parameters θ directly to determine the synaptic weights of network N , we apply a nonlinear transformation w k = f (θ k ) to each connection k, given by the function DISPLAYFORM5 where s k ∈ {1, −1} is a parameter that determines the sign of the connection weight and γ > 0 is a constant parameter that determines the smoothness of the mapping.

In the limit of large γ Eq. FORMULA0 converges to the rectified linear function DISPLAYFORM6 such that all connections with θ k < 0 are not functional.

Using this, the gradient of the log-likelihood function FORMULA14 can be written as DISPLAYFORM7 DISPLAYFORM8 where σ(x) = 1 1+e −x denotes the sigmoid function.

The error gradient DISPLAYFORM9 can be computed using standard Error Backpropagation Neal (1992); BID9 .Theorem 1 requires that Eq. (12) is twice differentiable, which is true for any finite value for γ.

In our simulations we used the limiting case of large γ such that dormant connections are actually mapped to zero weight.

In this limit, one approaches the simple expression DISPLAYFORM10 Thus, the gradient (13) vanishes for dormant connections (θ k < 0).

Therefore changes of dormant connections are independent of the error gradient.

This leads to the parameter updates of the soft-DEEP R algorithm given by Eq. (8).

The term √ 2T η ν k results from the diffusion term W k integrated over ∆t, where ν k is a Gaussian random variable with zero mean and unit variance.

The term −ηα results from the exponential prior distribution p S (θ) (the 1 -regularization).

Note that this prior is not differentiable at 0.

In (8) we approximate the gradient by assuming it to be zero at θ k = 0 and below.

Thus, parameters on the negative axis are only driven by a random walk and parameter values might therefore diverge to −∞. To fix this problem we introduced a reflecting boundary at θ min (parameters were clipped at this value).

Another potential solution would be to use a different prior distribution that also effects the negative axis, however we found that Eq. (8) produces very good results in practice.

Here we provide additional details to the convergence properties of the DEEP R algorithm.

To do so we formulate the algorithm in terms of a Markov chain that evolves the parameters θ and the connectivity constraints (listed in Algorithm 3).

Each application of the Markov transition operators corresponds to one iteration of the DEEP R algorithm.

We show that the distribution of parameters and network connectivities over the iterations of DEEP R converges to the stationary distribution Eq. (4) that jointly realizes parameter vectors θ and admissible connectivity constraints.

Each iteration of DEEP R corresponds to two update steps, which we formally describe in Algorithm 3 using the Markov transition operators T θ and T c and the binary constraint vector c ∈ {0, 1} M over all M connections of the network with elements c k , where c k = 1 represents an active connection k. c is a constraint on the dynamics, i.e., all connections k for which c k = 0 have to be dormant in the evolution of the parameters.

The transition operators are conditional probability distributions from which in each iteration new samples for θ and c are drawn for given previous values θ and c .

The transition operator T θ (θ|θ , c ) updates all parameters θ k for which c k = 1 (active connections) and leaves the parameters θ k at their current value for c k = 0 (dormant connections).

The update of active connections is realized by advancing the SDE (2) for an arbitrary time step ∆t (line 3 of Algorithm 3).

2.

Connectivity update: for all parameters θ k that are dormant, set c k = 0 and randomly select an element c l which is currently 0 and set it to 1.

This corresponds to line 3 of Algorithm 3 and is realized by drawing a new c from T c (c|θ).The constraint imposed by c on θ is formalized through the deterministic binary function C(θ, c) ∈ {0, 1} which is 1 if the parameters θ are compatible with the constraint vector c and 0 otherwise.

This is expressed as (with ⇒ denoting the Boolean implication): DISPLAYFORM0 The constraint C(θ, c) is fulfilled if all connections k with c k = 0 are dormant (θ k < 0).Note that the transition operator T c (c|θ) depends only on the parameter vector θ.

It samples a new c with uniform probability among the constraint vectors that are compatible with the current set of parameters θ.

We write the number of possible vectors c that are compatible with θ as µ(θ), given by the binomial coefficient (the number of possible selections that fulfill the constraint of new active connections) DISPLAYFORM1 where |c| denotes the number of non-zero elements in c and χ is the set of all binary vectors with exactly K elements of value 1.

Using this we can define the operator T c (c|θ) as: DISPLAYFORM2 where δ denotes the vectorized Kronecker delta function, with δ(0) = 1 and 0 else.

Note that Eq. (16) assigns non-zero probability only to vectors c that are zero for elements k for which θ k < 0 is true (assured by the second term).

In addition vectors c have to fulfill |c| = K. Therefore, sampling from this operator introduces randomly new connection for the number of missing ones in θ.

This process models the connectivity update of Algorithm 3.The transition operator T θ (θ|θ , c ) in Eq. (34) evolves the parameter vector θ under the constraint c, i.e., it produces parameters confined to the connectivity constraint.

By construction this operator has a stationary distribution that is given by the following Lemma.

Lemma 1.

Let T θ (θ|θ , c) be the transition operator of the Markov chain over θ which is defined, as the integration of the SDE written in Eq. (2) over an interval ∆t for active connections (c k = 1), and as the identity for the remaining dormant connections (c k = 0).

Then it leaves the following distribution p * (θ|c) invariant DISPLAYFORM3 where θ ∈c denotes the truncation of the vector θ to the active connections (c k = 1), thus p * (θ / ∈c < 0) is the probability that all connections outside of c are dormant according to the posterior, and p * (θ) is the posterior (see Theorem 1).The proof is divided into two sub proofs.

First we show that the distribution defined as p DISPLAYFORM4 , second we will show that this normalization constant has to be equal to p * (θ / ∈c < 0).

In coherence with the notation θ ∈c we will use verbally that θ k is an element of c if c k = 1.Proof.

To show that the distribution defined as p DISPLAYFORM5 To do so we will show that both p * (θ |c) and T factorizes in terms that depend only on θ ∈c or on θ / ∈c and thus we will be able to separate the integral over θ as the product of two simpler integrals.

We first study the distribution p * (θ ∈c |c).

Before factorizing, one has to notice a strong property of this distribution.

Let's partition the tempered posterior distribution p * (θ ) over the cases when the constraint is satisfied or not DISPLAYFORM6 when we multiply individually the first and the second term with C(c, θ), C(c, θ) can be replaced by its binary value and the second term is always null.

It remains that DISPLAYFORM7 seeing that one can rewrite the condition C(c, θ) = 1 as the condition on the sign of the random variable θ / ∈c < 0 (note that in this inequality c is a deterministic constant and θ / ∈c is a random variable) DISPLAYFORM8 We can factorize the conditioned posterior as p DISPLAYFORM9 .

But when the dormant parameters are negative θ / ∈c < 0, the active parameters θ ∈c do not depend on the actual value of the dormant parameters θ / ∈c , so we can simplify the conditions of the first factor further to obtain DISPLAYFORM10 We now study the operator T θ .

It factorizes similarly because it is built out of two independent operations: one that integrates the SDE over the active connections and one that applies identity to the dormant ones.

Moreover all the terms in the SDE which evolve the active parameters θ / ∈c are independent of the dormant ones θ / ∈c as long as we know they are dormant.

Thus, the operator T θ splits in two DISPLAYFORM11 To finally separate the integration over θ as a product of two integrals we need to make sure that all the factor depend only on the variable θ ∈c or only on θ / ∈c .

This might not seem obvious but even the conditioned probability p * (θ ∈c |θ / ∈c < 0) is a function of θ ∈c because in the conditioning θ / ∈c < 0, θ / ∈c refers to the random variable and not to a specific value over which we integrate.

As a result the double integral is equal to the product of the two integrals DISPLAYFORM12 We can now study the two integrals separately.

The second integral over the parameters θ / ∈c is simpler because by construction the operator T θ is the identity DISPLAYFORM13 There is more to say about the first integral over the active connections θ ∈c .

The operator T θ (θ ∈c |θ / ∈c , c) integrates over the active parameters θ ∈c the same SDE as before with the difference that the network is reduced to a sparse architecture where only the parameters θ ∈c are active.

We want to find the relationship between the stationary distribution of this new operator and p * (θ) that is written in the integral which is defined in equation FORMULA4 as the tempered posterior of the dense network.

In fact, the tempered posterior of the dense network marginalized and conditioned over the dormant connections p * (θ ∈c |θ / ∈c < 0) is equal to the stationary distribution of T θ (θ ∈c |θ / ∈c , c) (i.e. of the SDE in the sparse network).

To prove this, we detail in the following paragraph that the drift in the SDE evolving the sparse network is given by the log-posterior of the dense network condition on θ / ∈c < 0 and using Theorem 1, we will conclude that p * (θ ∈c |θ / ∈c < 0) is the stationary distribution of T θ (θ ∈c |θ / ∈c , c).

We write the prior and the likelihood of the sparse network as function of the prior and the likelihood p S with p N of the dense network.

The likelihood in the sparse network is defined as previously with the exception that the dormant connections are given zero-weight w k = 0 so it is equal to p N (X, Y * |θ ∈c , θ / ∈c < 0).

The difference between the prior that defines soft-DEEP R and the prior of DEEP R remains in the presence of the constraint.

When considering the sparse network defined by c the constraint is satisfied and the prior of soft-DEEP R marginalized over the dormant connections p S (θ ∈c ) is the prior of the sparse network with p S defined as before.

As this prior is connection-specific (p S (θ i ) independent of θ j ), this implies that p S (θ ∈c ) is independent of the dormant connection, and the prior p S (θ ∈c ) is equal to p S (θ ∈c |θ / ∈c < 0).

Thus, we can write the posterior of the sparse network which is by definition proportional to the product DISPLAYFORM14 .

Looking back to the definition of the posterior of the dense network this product is actually proportional to posterior of the dense network conditioned on the negativity of dormant connections p * (θ ∈c |θ / ∈c < 0, X, Y * ).

The posterior of the sparse network is therefore proportional to the conditioned posterior of the dense network but as they both normalize to 1 they are actually equal.

Writing down the new SDE, the diffusion term √ 2T βdW k remains unchanged, and the drift term is given by the gradient of the log-posterior log p * (θ ∈c |θ / ∈c < 0, X, Y * ).

Applying Theorem 1 to this new SDE, we now confirm that the tempered and conditioned posterior of the dense network p * (θ ∈c |θ / ∈c < 0) is left invariant by the SDE evolving the sparse network.

As T θ (θ ∈c |θ / ∈c , c) is the integration for a given ∆t of this SDE, it also leaves p * (θ ∈c |θ / ∈c < 0) invariant.

This yields θ ∈c T θ (θ ∈c |θ ∈c , c)p * (θ ∈c |θ / ∈c < 0)dθ ∈c = p * (θ ∈c |θ / ∈c < 0)As we simplified both integrals we arrived at θ T θ (θ|θ , c)p DISPLAYFORM15 Replacing the right-end side with equation FORMULA2 we conclude θ T θ (θ|θ , c)p * (θ |c)dθ = p * (θ|c)We now show that the normalization constant L(c) is equal to p * (θ / ∈c < 0).Proof.

Using equation FORMULA2 , as p * (θ|c) normalizes to 1 the normalization constant is equal to DISPLAYFORM16 By factorizing the last factor in the integral we have that p * (θ, θ / ∈c < 0) = p * (θ ∈c |θ / ∈c < 0)p * (θ / ∈c |θ / ∈c < 0)p * (θ / ∈c < 0)The last term does not depend on the value θ because θ / ∈c refers here to the random variable and the first two term depend either on θ ∈c or θ / ∈c .

Plugging the previous equation into the computation of L(c) and separating the integrals we have DISPLAYFORM17 Due to Lemma 1, there exists a distribution π(θ | c) of the following form which is left invariant by the operator T θ DISPLAYFORM18 where L(c) is a normalizer and where π(θ) is some distribution over θ that may not obey the constraint C(θ, c).

This will imply a very strong property on the compound operator which evolves both θ and c. To form T the operators T θ and T c are performed one after the other so that the total update can be written in terms of the compound operator DISPLAYFORM19 Applying the compound operator T given by Eq. (34) corresponds to advancing the parameters for a single iteration of Algorithm 3.Using these definitions a general theorem can be enunciated for arbitrary distributions π(θ | c) of the form (33).

The following theorem states that the distribution of variable pairs c and θ that is left stationary by the operator T is the product of Eq. (33) and a uniform prior p C (c) over the constraint vectors which have K active connections.

This prior is formally defined as DISPLAYFORM20 with χ as defined in (15) .

The theorem to analyze the dynamics of Algorithm 3 can now be written as Theorem 2.

Let T θ (θ|θ , c) be the transition operator of a Markov chain over θ and let T c (c|θ) be defined by Eq. (16).

Under the assumption that T θ (θ|θ , c) has a unique stationary distribution π(θ|c), that verifies Eq. (33), then the Markov chain over θ and c with transition operator DISPLAYFORM21 leaves the stationary distribution DISPLAYFORM22 invariant.

If the Markov chain of the transition operator T θ (θ|θ , c ) is ergodic, then the stationary distribution is also unique.

Proof.

Theorem 2 holds for T c in combination with any operator T θ that updates θ that can be written in the form (33).

We prove Theorem 2 by proving the following equality to show that T leaves (37) invariant: DISPLAYFORM23 Where p * (θ) is defined as previously as the tempered posterior of the dense network which is left invariant by soft-DEEP R according to Theorem 1.

The prior p C (c) in Eq. (43) assures that only constraints c with exactly K active connections are selected.

By Theorem 2 the stationary distribution (43) is also unique.

By inserting the result of Lemma 1, Eq. (17) we recover Eq. 4 of the main text.

Interestingly, by marginalizing over θ, we can show that the network architecture identified by c is sampled by algorithm 3 from the probability distribution DISPLAYFORM24 The difference between the formal algorithm 3 and the actual implementation of DEEP R are that T c keeps the dormant connection parameters constant, whereas in DEEP R we implement this by setting connections to 0 as they become negative.

We found the process used in DEEP R works very well in practice.

The reason why we did not implement algorithm 3 in practice is that we did not want to consume memory by storing any parameter for dormant connections.

This difference is obsolete from the view point of the network function for a given (θ, c) pair because nor negative neither strictly zero θ have any influence on the network function.

This difference might seem problematic to consider that the properties of convergence to a specific stationary distribution as proven for algorithm 3 extends to DEEP R. However, both the theorem and the implementation are rather unspecific regarding the choice of the prior on the negative sides θ < 0.We believe that, with good choices of priors on the negative side, the conceptual and quantitative difference between the distribution explored by 3 and DEEP R are minor, and in general, algorithm 3 is a decent mathematical formalization of DEEP R for the purpose of this paper.

@highlight

The paper presents Deep Rewiring, an algorithm that can be used to train deep neural networks when the network connectivity is severely constrained during training.

@highlight

An approach to implement deep learning directly on sparsely connected graphs, allowing networks to be trained efficiently online and for fast and flexible learning.

@highlight

The authors provide a simple algorithm capable of training with limited memory