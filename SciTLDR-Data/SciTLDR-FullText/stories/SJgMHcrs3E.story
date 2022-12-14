Catastrophic forgetting in neural networks is one of the most well-known problems in continual learning.

Previous attempts on addressing the problem focus on preventing important weights from changing.

Such methods often require task boundaries to learn effectively and do not support backward transfer learning.

In this paper, we propose a meta-learning algorithm which learns to reconstruct the gradients of old tasks w.r.t.

the current parameters and combines these reconstructed gradients with the current gradient to enable continual learning and backward transfer learning from the current task to previous tasks.

Experiments on standard continual learning benchmarks show that our algorithm can effectively prevent catastrophic forgetting and supports backward transfer learning.

The ability to learn continually without forgetting previously learned skills is crucial to artificial general intelligence (AGI) BID3 .

Addressing catastrophic forgetting in artificial neural networks (ANNs) has been the top priority of continual learning research.

Notable attempts on solving the problem include Elastic Weight Consolidation (EWC) by BID2 and the follow up work on Synaptic Intelligence (SI) by BID6 , and Memory Aware Synapse (MAS) by BID0 .

These algorithms share the same core idea: preventing important parameters from deviating from their old (presumably better) values.

In order to achieve that, EWC-like algorithms compute the importance of each parameter w.r.t.

each task in the sequence and for each old task, a regularization term is added to the loss of the new task to prevent that task from being catastrophically forgotten.

The regular-Preliminary work.

Under review by the International Conference on Machine Learning (ICML).

Do not distribute.

ization term for task T (i) in EWC-like algorithms takes the following form: DISPLAYFORM0 where λ (i) controls the relative importance of task i to the current task, θ is the current parameters, θ (i) * is the parameters found at the end of the training of T (i) , and ω DISPLAYFORM1 j is the importance of parameter θ 1.

The regularizer in Eqn.

1 prevent changes to important parameters regardless of the effect of these changes.

Unless θ DISPLAYFORM2 is the optimal value for the j-th parameter, either increasing or decreasing its value will result in better performance on task i. Keeping θ close to θ (i) * only prevent the network from catastrophically forgetting T (i) but cannot help the network to leverage the information from the current task T (k) , k > i to improve its performance on T (i) and other previous tasks.

In other words, regularizers of the form in Eqn.

1 do not support backward transfer learning.2.

The number of old parameter and importance vectors, θ * and ω, grows linearly with the number of tasks, making EWC-like algorithms not scalable to a large number of tasks.

BID5 proposed the online EWC algorithm which maintains only one copy of θ * and ω.

The sizes of θ * and ω are equal to that of the network.

Therefore, the memory requirement of online EWC is still considerably large for large networks.

To address these limitations of EWC-like algorithms, we propose a meta learning algorithm which:1.

Learns to approximate the gradient of a task w.r.t.

the current parameters from the current parameters 2.

Combines the approximated gradients of old tasks w.r.t.

the current parameters and the current task's gradient to result in an update that improves the performance of the network on all tasks.

By combining the gradients, our algorithm exploits the similarity between the current task and previous tasks to enable backward transfer learning.

As described in section 2.2 and 5.2, the size of a meta-network is typically orders of magnitude smaller than that of the main network and metanetworks for different tasks can be distilled into a single meta-network in an online manner.

That significantly reduces the memory requirement of our method.

In the next section, we introduce our learning to learn algorithm for continual learning.

Experiments are presented in section 3.

Conclusions and future work are located in section 4 and 5, respectively.

Let us consider a continual learning problem with a learner f (x; θ) : DISPLAYFORM0 , DISPLAYFORM1 , and T loss functions DISPLAYFORM2 To avoid clutter, we remove the input of the loss function L, and DISPLAYFORM3 In joint learning settings, data from all tasks is available to the learner.

The parameter θ is updated using the average of gradients from all tasks: DISPLAYFORM4 where α is the learning rate, and DISPLAYFORM5 Generally, updating θ with δ will improve the performance of f on all T tasks.

In continual learning settings, at task t+1, the learner cannot access to D (i) , i = 1, ..., t and cannot compute ∇ θ L (i) , i = 1, ..., t. The update at task t + 1 is computed from ∇ θ L (t+1)only.

When θ is updated with this gradient, f 's performance on T (t+1) will be improved while f 's performance on tasks T (i) , i = 1, ..., t might be catastrophically damaged.1 The analysis here still applies to the case where mini-batches are used because the expectation of DISPLAYFORM6 To address this problem, we propose the following meta learning algorithm.

During task i, we train meta-network DISPLAYFORM7 In subsequent tasks, h (i) is used to reconstruct the gradient of task i w.r.t.

the current parameters without having to access to D (i) .

More concretely, h (i) learns to map the parameter to the corresponding gradient: DISPLAYFORM8 When the main network f is trained on a new task DISPLAYFORM9 Section 2.3 introduces several ways to combine predicted gradients with the current gradient to prevent catastrophic forgetting and enable backward transfer learning.

For our method to work when optimizers other than SGD is used to train the main network, ∇ θ L (i) should be replaced with the update vector produced by the optimizer.

Because a real world neural network typically contains tens of thousands to billions of parameters, the naive way of training h would require an astronomically large number of samples of θ and ∇ to cover a very high dimensional space.

A fully connected meta-network h also need to be extremely large to receive a very high dimensional input and produce a very high dimensional output.

To circumvent the problem, we follow the coordinate-wise approach proposed by BID1 where each coordinate is processed independently.

h is a neural network that takes in a 1-dimensional input and produces 1-dimensional output DISPLAYFORM0 The procedure is applied to all coordinates in θ.

In our experiments, hs are MLPs and are trained to minimize the Euclidean distance between h(θ j ; φ) and ∇ j for all θ j in θ.

h could be modified to process more inputs such as the position of parameter in the network or the previous values of θ j .

It is also possible for h to process a small set of related parameters simultaneously, e.g. parameters in the same filter of a CNN.

However, we leave these variations for future work.

Let us consider a pair of gradients ∇ (k) and DISPLAYFORM0 and ∇

j have different signs and α is small enough, then updating f with ∇ (k) j will improve the network's performance on task k and damage its performance on task i. If they have the same sign then the update will improve the performance on both tasks.

That intuition leads to the following rule to create an update vector from a pair of gradients: DISPLAYFORM0 At task t+1, an update vector δ can be produced by applying the above rule the pair between ∇ (t+1) and all other gradients∇ (i) , i = 1, ..., t. When t is large, that method usually results in a sparse update vector.

In practice, we apply the rule to the pair ∇ (t+1) ,∇ 1:t where∇ 1:t = 1 t t i=1∇(i) .

Updating the main network with δ will improve the performance on task t + 1 and will likely to improve the performance on tasks 1, ..., t.

The update vector δ contains information that are common between task t + 1 and previous tasks.

Updating f with δ transfers information from the current task to previous tasks.

δ is the medium for backward transfer learning in our algorithm.

We tested our algorithm on the Permuted MNIST dataset BID2 ).

To better demonstrate the effect of backward transfer learning, we train each task for only 2000 iterations to prevent the main network from reaching its maximum performance.

The result is shown in FIG1 .The network in FIG1 suffers from catastrophic forgetting problem: the performance on old tasks decrease rapidly when new tasks are trained.

The network trained with our algorithm FIG1 ) does not suffer from catastrophic forgetting: the performance on old tasks is maintained or even improved when new tasks are trained.

The performance improvement on old tasks suggests that our algorithm has backward transfer learning capability.

We also note the forward transfer learning phenomenon in FIG1 : the starting accuracy of a later task is higher than that of former ones.

In this paper, we present a meta learning algorithm for continual learning.

Experiments on Permuted MNIST dataset show that our algorithm is effective in preventing catastrophic forgetting and is capable of supporting backward transfer learning.

To make our algorithm works without task boundaries, we need to detect boundaries automatically.

The simplest way to detect task boundaries is to look at the loss of the main network.

That way, however, does not work well when different tasks uses different loss functions or have different input, output scales.

We propose to use detect task boundaries using the loss of the meta-networks.

Different tasks

@highlight

We propose a meta learning algorithm for continual learning which can effectively prevent catastrophic forgetting problem and support backward transfer learning.