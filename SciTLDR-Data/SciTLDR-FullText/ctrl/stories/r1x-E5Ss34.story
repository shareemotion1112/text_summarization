Catastrophic forgetting poses a grand challenge for continual learning systems, which prevents neural networks from protecting old knowledge while learning new tasks sequentially.

We propose a Differentiable Hebbian Plasticity (DHP) Softmax layer which adds a fast learning plastic component to the slow weights of the softmax output layer.

The DHP Softmax behaves as a compressed episodic memory that reactivates existing memory traces, while creating new ones.

We demonstrate the flexibility of our model by combining it with existing well-known consolidation methods to prevent catastrophic forgetting.

We evaluate our approach on the Permuted MNIST and Split MNIST benchmarks, and introduce Imbalanced Permuted MNIST — a dataset that combines the challenges of class imbalance and concept drift.

Our model requires no additional hyperparameters and outperforms comparable baselines by reducing forgetting.

A key aspect of human intelligence is the ability to continually adapt and learn in dynamic environments, a characteristic which is challenging to embed into artificial intelligence.

Recent advances in machine learning (ML) have shown tremendous improvements in various problems, by learning to solve one complex task very well, through extensive training on large datasets with millions of training examples or more.

Most of the ML models that we use during deployment assume that the real-world is stationary, where in fact it is non-stationary and the distribution of acquired data changes over time.

Therefore, after learning is complete, and these models are fine-tuned with new data, performance degrades with respect to the original data.

This phenomenon *Work done during an internship at Uber AI.

† Work done while at Google Brain.

known as catastrophic forgetting or catastrophic interference BID17 BID7 serves to be a crucial problem for deep neural networks (DNNs) that are tasked with continual learning BID26 or lifelong learning (Thrun & Mitchell, 1995) .

In this learning paradigm, the goal is to adapt and learn consecutive tasks without forgetting how to perform previously learned tasks.

Some of the real-world applications that typically require this kind of learning include perception for autonomous vehicles, recommender systems, fraud detection, etc.

In most supervised learning methods, DNN architectures require independent and identically distributed (iid) samples from a stationary training distribution.

However, for ML systems that require continual learning in the real-world, the iid assumption is easily violated when: (1) There is concept drift or class imbalance in the training data distribution.(2) Data representing all scenarios in which the learner is expected to perform are not initially available.

In such situations, DNNs face the "stability-plasticity dilemma" BID6 BID0 .

This presents a continual learning challenge for models that need to balance plasticity (integrate new knowledge) and stability (preserve existing knowledge).Two major theories have been proposed to explain a human's ability to perform continual learning.

The first theory is inspired by synaptic consolidation in the mammalian neocortex BID5 where a subset of synapses are rendered less plastic and therefore preserved for a longer timescale.

The second theory is the complementary learning systems (CLS) theory BID16 BID23 BID12 , which suggests that humans extract high-level structural information and store it in a different brain area while retaining episodic memories.

Here, we extend the work on differentiable plasticity BID18 BID19 to a continual learning setting and develop a model that is capable of adapting quickly to changing environments as well as consolidating previous knowledge by selectively adjusting the plasticity of synapses.

We modify the traditional softmax layer and propose to augment the slow weights with a set of plastic weights implemented using Differentiable Hebbian Plasticity (DHP).

The model's slow weights learn deep representations of data and the fast weights implemented with DHP learn to quickly "auto-associate" the class labels to representations.

We also demonstrate the flexibility of our model by combining it with recent task-specific synaptic consolidation based methods to overcoming catastrophic forgetting such as elastic weight consolidation BID11 BID28 , synaptic intelligence (Zenke et al., 2017) and memory aware synapses .

Our model unifies core concepts from Hebbian plasticity, synaptic consolidation and CLS theory to enable rapid adaptation to new unseen data, while consolidating synapses and leveraging compressed episodic memories to remember previous knowledge and mitigate catastrophic forgetting.

Plastic Neural Networks: One of the major theories that have been proposed to explain a human's ability to learn continually is Hebbian learning BID8 , which suggests that learning and memory are attributed to weight plasticity, that is, the modification of the strength of existing synapses according to variants of Hebb's rule BID24 Song et al., 2000; BID22 .Recent approaches in the meta-learning literature have shown that we can incorporate fast weights into a neural network BID21 BID25 .

BID21 augmented fully-connected (FC) layers preceding the softmax with a matrix of fast weights.

Here, the fast weights were implemented with non-trainable Hebbian learning-based associative memory.

BID25 proposed a softmax layer that can improve learning of rare classes by interpolating between Hebbian updates and stochastic gradient descent (SGD) updates on the output layer using an arbitrarily engineered scheduling scheme.

BID19 proposed differentiable plasticity, which uses SGD to optimize the plasticity of each synaptic connection composed of a slow weight and a plastic (fast) weight.

Although this approach served to be a powerful new method for training neural networks, it was mainly demonstrated on RNNs for solving simple tasks.

Overcoming Catastrophic Forgetting: This work leverages two biologically inspired strategies to overcome the catastrophic forgetting problem: 1) Task-specific Synaptic Consolidation -Protecting old knowledge by dynamically adjusting the synaptic strengths to consolidate and retain memories.

2) CLS Theory -A dual memory system where, structural knowledge is acquired through slow learning via the neocortex and rapid learning via the hippocampus.

There have been several notable works inspired by taskspecific synaptic consolidation for overcoming catastrophic forgetting BID11 Zenke et al., 2017; BID1 .

All of these approaches propose a method to estimate the importance of each parameter or synapse, Ω k , where the least plastic synapses can retain memories for a long time and the more plastic synapses are considered less important.

The Ω k and network parameters θ k are updated online or after learning task T n .

Therefore, when learning new task T n , a regularizer is added to the original loss function L n (θ), so that we dynamically adjust the plasticity w.r.t.

Ω k and prevent any changes to the important parameters of previously learned tasks: DISPLAYFORM0 where θ n−1 k are the learned network parameters after training on the previous n − 1 tasks and λ is a hyperparameter for the regularizer to control the amount of forgetting.

In Elastic Weight Consolidation (EWC), BID11 use the diagonal values of an approximated Fisher information matrix for Ω k , and it is computed offline after training on a task is completed.

BID28 proposed an online variant of EWC to improve scalability by ensuring the computational cost of the regularization term does not grow with the number of tasks.

Zenke et al. (2017) proposed an online method called Synaptic Intelligence (SI) for computing the parameter importance where, Ω k is the cumulative change in individual synapses over the entire training trajectory on a given task.

Memory Aware Synapses (MAS) from BID1 measures Ω k by the sensitivity of the learned function to a perturbation in the parameters and use the cumulative change in individual synapses on the squared L2-norm of the penultimate layer.

There have been numerous approaches based on CLS principles involving pseudo-rehersal (Robins, 1995; BID2 BID3 , episodic replay BID15 BID14 and generative replay (Shin et al., 2017; Wu et al., 2018) .

However, in our work, we are primarily interested in neuroplasticity techniques inspired from CLS theory for representing memories.

Hinton & Plaut (1987) showed how each synaptic connection can be composed of a fixed weight where slow learning stores long-term knowledge and a fast weight for temporary associative memory.

Recent research in this vein has included replacing soft attention mechanism with fast weights in RNNs BID4 , the Hebbian Softmax layer BID25 , augmenting the FC layer with a fast weights matrix BID21 , differentiable plasticity BID19 and neuromodulated differentiable plasticity BID20 .

However, all of these methods were focused on rapid learning on simple tasks or meta-learning over a distribution of tasks.

Furthermore, they did not examine learning a large number of new tasks while, alleviating catastrophic forgetting in continual learning.

In our model, each synaptic connection in the softmax layer has two weights: 1) The slow weights, θ ∈ R m×d , where m is the number of units in the final hidden layer.

2) A Hebbian plastic component of the same cardinality as the slow weights, composed of the plasticity coefficient, α, and the Hebbian trace, Hebb.

The α is a scaling parameter for adjusting the magnitude of the Hebb.

Hebb accumulates the mean activations of the penultimate layer for each target label in the mini-batch {y 1:B } of size B which are denoted byh ∈ R 1×m (refer to Algorithm 1).

Given the activation of each neuron in h at the pre-synaptic connection i, the unnormalized log probabilities z at the post-synaptic connection j can be more formally computed using Eq. 2.

Then, the softmax function is applied on z to obtain the desired logitsŷ thus,ŷ = softmax(z).

The η parameter in Eq. 3 is a "learning rate" that learns how quickly to acquire new experiences into the plastic component.

The η parameter also acts as a decay term to prevent instability caused by a positive feedback loop in the Hebbian traces.

DISPLAYFORM0 The network parameters α i,j , η and θ i,j are optimized by gradient descent as the model is trained sequentially on different tasks in the continual learning setup.

Hebb is initialized to zero only at the start of learning the first task T 1 and is automatically updated based on Algorithm 1 in the forward pass during training.

Specifically, the Hebbian update for the active class c in y 1:B is computed on line 6.

This Hebbian update BID8 , where w i,j is the change in weight at connection i, j and a k i , a k j denote the activation levels of neurons i and j, respectively, for the k th input.

Therefore, in our model, w =h the Hebbian weight update, a i = h the hidden activations of the last hidden layer, a j = y the active target class in y 1:B and N = s the number of inputs for the corresponding class in y 1:B (see Algorithm 1).

Across the model's lifetime, we only update Hebb during training and during test time, we use the most recent Hebbian traces to make predictions.

The plastic component learns rapidly and performs sparse parameter updates to quickly store memory traces for each recent experience without interference from other similar recent experiences.

Furthermore, the hidden activations corresponding to the same active class are accumulated into one vectorh, thus forming a compressed episodic memory in the Hebb to reflect individual episodic memory traces.

This method improves learning of rare classes and speeds up binding of class labels to deep representations of the data.

Updated Loss:

Following the existing work for overcoming catastrophic forgetting such as EWC, Online EWC, SI and MAS (see Eq. 1), we regularize the loss L n (θ, α, η) and update the synaptic importance parameters of the network in an online manner.

We rewrite Eq. 1 to obtain Eq. 4 and show that the network parameters θ i,j are the weights of the connections between pre-and post-synaptic activity, as seen in Eq. 2.

DISPLAYFORM1 We adapt these existing consolidation approaches to our model and only compute the synaptic importance parameters on the slow weights of the network.

The plastic part of our model can alleviate catastrophic forgetting of learned classes by optimizing the plasticity of the synaptic connections.

We tested our continual learning approach on the Permuted MNIST, Imbalanced Permuted MNIST and Split MNIST benchmarks.

We evaluated the methods based on the average classification accuracy on all previously learned tasks.

To establish a baseline for comparison of well-known synaptic consolidation methods, we trained neural networks with Online EWC, SI and MAS, respectively, on all tasks in a sequential manner.

In the Permuted MNIST and Imbalanced Permuted benchmarks we trained a multi-layered perceptron (MLP) network on a sequence of 10 tasks using plain SGD.

Detailed descriptions of the hyperparameters and training setups for all benchmarks can be found in Appendix A.Permuted MNIST: In this benchmark, all of the MNIST pixels are permuted differently for each task with a fixed random permutation.

Although the output domain is constant, the input distribution changes between tasks thus, there exists a concept drift.

Figure 1 shows the average test accuracy as new tasks are learned.

The network with DHP Softmax alone showed significant improvement in its ability to alleviate catastrophic forgetting across all tasks compared to the baseline finetuned vanilla MLP network we refer to as Finetune in Figure 1 .

Then we compared the performance with and without DHP Softmax using the synaptic consolidation methods.

We find our DHP Softmax with synaptic consolidation maintains a higher test accuracy after T 10 tasks than without DHP Softmax for all variants.

Figure 1 .

The average test accuracy on a sequence of Permuted MNIST tasks Tn=1:10.

The average test accuracy after T10 tasks is given in the legend.

Error bars correspond to SE on 10 trials.

This benchmark is identical to the Permuted MNIST benchmark but, now each task is an imbalanced distribution.

The statistics of the class distribution in each task are presented in Appendix A.2, Table 1 .

Figure 2 shows the average test accuracy as new tasks are learned.

We see that DHP Softmax achieves 80.85% after learning 10 tasks, thus providing significant improvement over the standard neural network baseline of 76.4%.

The significance of the compressed episodic memory mechanism in the Hebbian traces is more apparent in this benchmark because the plastic component allows rare classes that are encountered infrequently to be remembered for a longer period of time.

We find that DHP Softmax with MAS achieves 88.8%; outperforming all other methods and across all tasks.

Split MNIST: A sequence of T n=1:5 tasks are generated by splitting the original MNIST training dataset into binary classification problems (0/1, 2/3, 4/5, 6/7, 8/9), making the output spaces disjoint between tasks.

Similar to Zenke et al. (2017) , we trained a multi-headed MLP network on a sequence of 5 tasks.

We compute the cross entropy loss at the softmax output layer only for the digits present in the current task, T n .

We observe that DHP Softmax provides a 4.7% improvement on test performance compared to a finetuned MLP network (Figure 3) .

Also, combining DHP Softmax with task-specific consolidation consistently improves performance across all tasks T n=1:5 .

Figure 3 .

The average test accuracy on a sequence of 5 binary classification problems (0/1, 2/3, 4/5, 6/7, 8/9) from the original MNIST dataset.

The average test accuracy after learning T5 tasks is given in the legend.

Error bars refer to the SE on 10 trials.

We have shown that the problem of catastrophic forgetting in continual learning environments can be alleviated by adding compressed episodic memory in the softmax layer through DHP.

DHP Softmax alone showed noticeable improvement across all benchmarks when compared to a neural network with a traditional softmax layer.

We demonstrated the flexibility of our model where, in addition to DHP Softmax, we can regularize the slow weights using EWC, SI or MAS to improve a model's ability to alleviate catastrophic forgetting.

The approach where we combine DHP Softmax and MAS consistently leads to overall superior results compared to other baseline methods on several benchmarks.

This gives a strong indication that Hebbian plasticity enables neural networks to learn continually and remember distant memories, thus reducing catastrophic forgetting when learning from sequential datasets in dynamic environments.

For the Imbalanced Permuted MNIST experiments shown in Figure 2 , the regularization hyperparameter λ for each of the task-specific consolidation methods is λ = 400 for Online EWC BID28 , λ = 1.0 for SI (Zenke et al., 2017) and λ = 0.1 for MAS .

In SI, the damping parameter, ξ, was set to 0.1.

Similar to the Permuted MNIST benchmark, to find the best hyperparameter combination for each of these synaptic consolidation methods, we performed a grid search using a task sequence determined by a single seed.

Across all experiments, we maintained the the same random probabilities detemined by a single seed to artificially remove training samples from each class.

The hyperparameters of the synaptic consolidation methods (i.e. Online EWC, SI and MAS) remain the same with and without DHP Softmax, and the plastic components are not regularized.

<|TLDR|>

@highlight

Hebbian plastic weights can behave as a compressed episodic memory storage in neural networks; improving their ability to alleviate catastrophic forgetting in continual learning.