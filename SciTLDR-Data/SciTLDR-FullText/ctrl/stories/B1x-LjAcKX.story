This paper proposes a novel approach to train deep neural networks by unlocking the layer-wise dependency of backpropagation training.

The approach employs additional modules called local critic networks besides the main network model to be trained, which are used to obtain error gradients without complete feedforward and backward propagation processes.

We propose a cascaded learning strategy for these local networks.

In addition, the approach is also useful from multi-model perspectives, including structural optimization of neural networks, computationally efficient progressive inference, and ensemble classification for performance improvement.

Experimental results show the effectiveness of the proposed approach and suggest guidelines for determining appropriate algorithm parameters.

In recent days, deep learning has been remarkably advanced and successfully applied in numerous fields BID14 .

A key mechanism behind the success of deep neural networks is that they are capable of extracting useful information progressively through their layered structures.

It is an increasing trend that more and more complex deep neural network structures are developed in order to solve challenging real-world problems, e.g., BID7 .

Training of deep neural networks is based on backpropagation in most cases, which basically works in a sequential and synchronous manner.

During the feedforward pass, the input data is processed through the hidden layers to produce the network output; during the feedback pass, the error gradient is propagated back through the layers to update each layer's weight parameters.

Therefore, training of each layer has dependency on all the other layers, which causes the issue of locking BID9 .

This is undesirable in some cases, e.g., a system consisting of several interacting models, a model distributed across multiple computing nodes, etc.

There have been attempts to remove the locking constraint.

In BID0 , the method of auxiliary coordinates (MAC) is proposed.

It replaces the original loss minimization problem with an equality-constrained optimization problem by introducing an auxiliary variable for each data and each hidden unit.

Then, solving the problem is formulated as iteratively solving several sub-problems independently.

A similar approach using the alternating direction method of multipliers (ADMM) is proposed in BID17 .

It also employs an equality-constrained optimization but with different auxiliary variables, so that resulting sub-problems have closed form solutions.

However, these methods are not scalable to deep learning architectures such as convolutional neural networks (CNNs).The method proposed in BID9 , called decoupled neural interface (DNI), directly synthesizes estimated error gradients, called synthetic gradients, using an additional small neural network for training a layer's weight parameters.

As long as the synthetic gradients are close to the actual backpropagated gradients, each layer does not need to wait until the error at the output layer is backpropagated through the preceding layers, which allows independent training of each layer.

However, this method suffers from performance degradation when compared to regular backpropagation BID2 .

The idea of having additional modules supporting the layers of the main model is also adopted in BID2 , where the additional modules are trained to approximate the main model's outputs instead of error gradients.

Due to this, however, the method does not resolve the issue of update locking, and in fact, the work does not intend to design a non-sequential learning algorithm.

BID9 and the proposed local critic training.

The black, green, and blue arrows indicate feedforward passes, an error gradient flow, and loss comparison, respectively.

In this paper, we propose a novel approach for non-sequential learning, called local critic training.

The key idea is that additional modules besides the main neural network model are employed, which we call local critics, in order to indirectly deliver error gradients to the main model for training without backpropagation.

In other words, a local critic located at a certain layer group is trained in such a way that the derivative of its output serves as the error gradient for training of the corresponding layers' weight parameters.

Thus, the error gradient does not need to be backpropagated, and the feedforward operations and gradient-descent learning can be performed independently.

Through extensive experiments, we examine the influences of the network structure, update frequency, and total number of local critics, which provide not only insight into operation characteristics but also guidelines for performance optimization of the proposed method.

In addition to the capability of implementing training without locking, the proposed approach can be exploited for additional important applications.

First, we show that applying the proposed method automatically performs structural optimization of neural networks for a given problem, which has been a challenging issue in the machine learning field.

Second, a progressive inference algorithm using the network trained with the proposed method is presented, which can adaptively reduce the computational complexity during the inference process (i.e., test phase) depending on the given data.

Third, the network trained by the proposed method naturally enables ensemble inference that can improve the classification performance.

The basic idea of the proposed approach is to introduce additional local networks, which we call local critics, besides the main network model, so that they eventually provide estimates of the output of the main network.

Each local critic network can serve a group of layers of the main model by being attached to the last layer of the group.

The proposed architecture is illustrated in FIG0 , where f i is the ith layer group (containing one or more layers), h i is the output of f i , and h N is the final output of the main model having N layer groups: DISPLAYFORM0 ) c i is the local critic network for f i , which is expected to approximate h N based on h i , i.e., DISPLAYFORM1 Then, this can be used to approximate the loss function of the main network, L N = l(h N , y), which is used to train f i , by DISPLAYFORM2 Under review as a conference paper at ICLR 2019 DISPLAYFORM3 where y is the training target and l is the loss function such as cross-entropy or mean-squared error.

Then, the error gradient for training f i is obtained by differentiating L i with respect to h i , i.e., DISPLAYFORM4 which can be used to train the weight parameters of f i , denoted by θ i , via a gradient-descent rule: DISPLAYFORM5 where η is a learning rate.

Note that the final layer group h N does not require a local critic network and can be trained using the regular backproagation because the final output of the main network is directly available.

Therefore, the update of f i does not need to wait until its output h i propagates till the end of the main network and the error gradient is backpropagated; it can be performed when the operations from (2) to (5) are done.

For c i , we usually use a simple model so that the operations through c i are simpler than those through f i+1 till f N .While the dependency of f i on f j (j > i) during training is resolved in this way, there still exists the dependency of c i on f j (j > i), because training c i requires its ideal target, i.e., h N , which is available from f N only after the feedforward pass is complete.

In order to resolve this problem, we use an indirect, cascaded approach, where c i is trained so that its training loss targets the training loss for c i+1 1 : DISPLAYFORM6 In other words, training of c i can be performed once the loss for c i+1 is available.

FIG0 compares the proposed architecture with the existing DNI approach that also employs local networks besides the main network to resolve the issue of locking BID9 .

In DNI, the local network c i directly estimates the error gradient, i.e., DISPLAYFORM7 so that each layer group of the main model can be updated without waiting for the forward and backward propagations in the subsequent layers.

And, to update c i , the error gradient for f i+1 estimated by c i+1 is backpropagated through f i+1 and is used as the (estimated) target for c i .

Therefore, all the necessary computations in the forward and backward passes can be locally confined.

The performance of the two methods will be compared in Section 3.

In many cases, determining an appropriate structure of neural networks for a given problem is not straightforward.

This is usually done through trial-and-error, which is extremely time-consuming.

There have been studies to automate the structural optimization process BID1 BID5 BID12 BID15 , but this issue still remains very challenging.

In deep learning, the problem of structural optimization is even more critical.

Large-sized networks may easily show overfitting.

Even if large networks may produce high accuracy, they take significantly large amounts of memory and computation, which is undesirable especially for resourceconstrained cases such as embedded and mobile systems.

Therefore, it is highly desirable to find an optimal network structure that is sufficiently small while the performance is kept reasonably good.

During local critic training, each local critic network is trained to estimate the output of the main network eventually.

Therefore, once the training of the proposed architecture finishes, we obtain different networks that are supposed to have similar input-output mappings but have different structures and possibly different accuracy, i.e., multiple sub-models and one main model (see Figure 2b ).

Here, a sub-model is composed of the layers on the path from the input to a certain hidden layer DISPLAYFORM0 and its local critic network.

Among the sub-models, we can choose one as a structure-optimized network by considering the trade-off relationship between the complexity and performance.

It is worth mentioning that our structural optimization approach can be performed instantly after training of the model, whereas many existing methods for structural optimization require iterative search processes, e.g., BID19 .

We propose another simple but effective way to utilize the sub-models obtained by the proposed approach for computational efficiency, which we call progressive inference.

Although small submodels (e.g., sub-model 1) tend to show low accuracy, they would still perform well for some data.

For such data, we do not need to perform the full feedforward pass but can take the classification decision by the sub-models.

Thus, the basic idea of the progressive inference is to finish inference (i.e., classification) with a small sub-model if its confidence on the classification result is high enough, instead of completing the full feedforward pass with the main model, which can reduce the computational complexity.

Here, the softmax outputs for all classes are compared and the maximum probability is used as the confidence level.

If it is higher than a threshold, we take the decision by the sub-model; otherwise, the feedforward pass continues.

The proposed progressive inference method is summarized in Algorithm 1 2 .

In recent deep learning systems, it is popular to use ensemble approaches to improve performance in comparison to single models, where multiple networks are combined for producing final results, e.g., BID6 ; BID16 .

The sub-models and main model obtained by applying the proposed local critic training approach can be used for ensemble inference.

FIG2 depicts how the sub-models and the main model can work together to form an ensemble classifier.

We take the simplest way to combine them, i.e., summation of the networks' outputs.

We conduct extensive experiments to examine the performance of the proposed method in various aspects.

We use the CIFAR-10 and CIFAR-100 datasets BID11 ) with data augmentation.

We employ a VGG-like CNN architecture with batch normalization and ReLU activation functions, which is shown in Figure 2a .

Note that this structure is the same to that used in BID2 .

It has three local critic networks, thus four layer groups that can be trained independently are formed (i.e., N =4).

The local critic networks are also CNNs, and their structures are kept as 2 Our method shares some similarity with the anytime prediction scheme BID13 BID8 that produces outputs according to the given computational budget.

However, ours does not require particular network structures (such as multi-scale dense network BID8 or FractalNet BID13 ) but works with generic CNNs.

simple as possible in order to minimize the computational complexity for computing the estimated error gradient given by (5).We use the stochastic gradient descent with a momentum of 0.9 for the main network and the Adam optimization with a fixed learning rate of 10 −4 for the local networks.

The L2 regularization is used with 5 × 10 −4 for the main network.

For the loss functions in (3) and FORMULA6 , the cross-entropy and the L1 loss are used, respectively, which is determined empirically.

The batch size is set to 128, and the maximum training iteration is set to 80,000.

The learning rate for the main network is initialized to 0.1 and dropped by an order of magnitude after 40,000 and 60,000 iterations.

The Xavier method is used for initialization of the network parameters.

All experiments are performed using TensorFlow.

We conduct all the experiments five times with different random seeds and report the average accuracy.

Figure 3 shows how the loss values of the main network and each local critic network, i.e., L i in (3), evolve with respect to the training iteration.

The graphs show that the local critic networks successfully learn to approximate the main network's loss with high accuracy during the whole training process.

The local critic network farthest from the output side (i.e., L 1 ) shows larger loss values than the others, which is due to the inaccuracy accumulated through the cascaded approximation.

The classification performance of the proposed local critic training approach is evaluated in Table 1 .

For comparison, the performance of the regular backpropagation, DNI BID9 , and critic training BID2 ) is also evaluated.

Although the critic training method is not for removing update locking, we include its result because it shares some similarity with our approach, i.e., additional modules to estimate the main network's output.

In all three methods, each additional module is composed of a convolutional layer and an output layer.

In the case of the proposed method, we test different numbers of local critic networks.

Figure 2a shows the structure Table 1 : Average test accuracy (%) of backpropagation (BP), DNI BID9 , critic training BID2 , and proposed local critic training (LC).

The numbers of local networks used are shown in the parentheses.

The standard deviation values are also shown.

with three local critic networks.

When only one local network is used, it is located at the place of LC2 in Figure 2a .

When five local networks are used, they are placed after every two layers of the main network.

When compared to the result of backpropagation, the proposed approach successfully decouples training of the layer groups at a small expense of accuracy decrease (note that the performance of the proposed method can be made closer to that of backpropagation using different structures, as will be shown in Tables 2 and FIG2 ).

The degradation of the accuracy and standard deviation of our method is larger for CIFAR-100, which implies that the influence of gradient estimation is larger for more complex problems.

When more local critic networks are used, the accuracy tends to decrease more due to higher reliance on predicted gradients rather than true gradients, while more layer groups can be trained independently.

Thus, there exists a trade-off between the accuracy and unlocking effect.

The DNI method shows poor performance as in BID2 ).

The proposed method shows performance improvement by 0.4% and 0.9% over the critic training method, both with three local networks, for the two datasets, respectively, which are found to be statistically significant using Mann-Whitney tests at a significance level of 0.05.

This shows the efficacy of the cascaded learning scheme of the local networks in our method.

We examine the influence of the structures of the local critic networks in our method.

Two aspects are considered, one about the influence of the overall complexity of the local networks and the other about the relative complexities of the local networks for good performance.

For this, we change the number of convolutional layers in each local critic network, while keeping the other structural parameters unchanged.

The results for various structure combinations of the three local critic networks are shown in TAB1 .

As the number of convolutional layers increase for all local networks (the first three cases in the table), the accuracy for CIFAR-100 slightly increases from 69.91% (with one convolutional layer) to 70.02% (three convolutional layers) and 70.34% (five convolutional layers), whereas for CIFAR-10 the accuracy slightly decreases when five convolutional layers are used.

A more complex local network can learn better the target input-output relationship, which leads to the performance improvement for CIFAR-100.

For CIFAR-10, on the other hand, the network structure with five convolutional layers seems too complex compared to the data to learn, which causes the performance drop.

Next, the numbers of layers of the local networks are adjusted differently in order to investigate which local networks should be more complex for good performance.

The results are shown in the last four columns of TAB1 .

Overall, it is more desirable to use more complex structures for the local networks closer to the input side of the main model.

For instance, LC1 and LC3 are supposed to learn the relationship from h 1 to h 4 and that from h 3 to h 4 , respectively.

More layers are involved from h 1 to h 4 in the main network, so the mapping that LC1 should learn would be more complicated, requiring a network structure with sufficient modeling capability.

A way to increase the efficiency of the proposed approach is to update the local critic networks not at every iteration but only periodically.

This may degrade the accuracy but has two benefits.

First, the amount of computation required to update the local networks can be reduced.

Second, the burden of the communication between the layer groups also can be reduced.

These benefits will be more significant when the local networks have larger sizes.

For the default structure shown in Figure 2a , we compare different update frequency in TAB2 .

It is noticed that the accuracy only slightly decreases as the frequency decreases.

When the update frequency is a half of that for the main network (i.e., 1/2), the accuracy drops by 0.48% and 1.92% for the two datasets, respectively.

Then, the decrease of the accuracy is only 0.56% for CIFAR-10 and 1.60% for CIFAR-100 when the update frequency decreases from 1/2 to 1/5.

TAB3 compares the performance of the sub-models, and TAB4 shows the complexities of the sub-models in terms of the amount of computation for a feedforward pass and the number of weight parameters.

A larger network (e.g., sub-model 3) shows better performance than a smaller network (e.g., sub-model 1), which is reasonable due to the difference in learning capability with respect to the model size.

The largest sub-model (sub-model 3) shows similar accuracy to the main model (92.29% vs. 92.39% for CIFAR-10 and 67.54% vs. 69.91% for CIFAR-100), while the complexity is significantly reduced.

For CIFAR-10, the computational complexity in terms of the number of floating-point operations (FLOPs) and the memory complexity are reduced to only about 30% (15.72 to 4.52 million FLOPs, and 7.87 to 2.26 million parameters), as shown in TAB4 .

If an absolute accuracy reduction of 1.86% (from 92.39% to 90.53%) is allowed by taking sub-model 2, the reduction of complexity is even more remarkable, up to about one ninth.

In addition, the table also shows the accuracy of the networks that have the same structures with the sub-models but are trained using regular backpropagation.

Surprisingly, such networks do not easily reach accuracy comparable to that of the sub-models obtained by local critic training, particularly for smaller networks (e.g., 74.46% vs. 85.24% with sub-model 1 for CIFAR-10).

We think that joint training of the sub-models in local critic training helps them to find better solutions than those reached by independent regular backpropagation.

Therefore, these results demonstrate that a structurally optimized network can be obtained at a cost of a small loss in accuracy by local critic training, which may not be attainable by trial-and-error with backpropagation.

We apply the progressive inference algorithm shown in Algorithm 1 to the trained default network for CIFAR-10 with the threshold set to 0.9 or 0.95.

The results are shown in TAB5 .

The feedforward pass ends at different sub-models for different test data, and the average FLOPs over all test data are shown.

When the threshold is 0.9, with only a slight loss of accuracy (92.39% to 91.18%), the computational complexity is reduced significantly, which is only 18.45% of that of the main model.

When the threshold increases to 0.95, the accuracy loss becomes smaller (only 0.64%), while the complexity reduction remains almost the same (19.40% of the main model's complexity).

The results of ensemble inference using the sub-models and main model are shown in FIG2 .

Using an ensemble of the three sub-models, we observe improved classification accuracy (92.68% and 70.86% for the two datasets, respectively) in comparison to the main model.

The performance is further enhanced by an ensemble of both the three sub-models and the main model (92.79% and 71.86%).

The improvement comes from the complementarity among the models, particularly between the models sharing a smaller number of layers.

For instance, we found that sub-model 3 and the main model tend to show coincident classification results for a large portion of test data, so their complementarity is not significant; on the other hand, more data are classified differently by sub-model 1 and the main model, where we mainly observe performance improvement.

Instead of the simple summation, there could be a better method to combine the models, which is left for future work.

In this paper, we proposed the local critic training approach for removing the inter-layer locking constraint in training of deep neural networks.

In addition, we proposed three applications of the local critic training method: structural optimization of neural networks, progressive inference, and ensemble classification.

It was demonstrated that the proposed method can successfully train CNNs with local critic networks having extremely simple structures.

The performance of the method was also evaluated in various aspects, including effects of structures and update intervals of local critic networks and influences of the sizes of layer groups.

Finally, it was shown that structural optimization, progressive inference, and ensemble classification can be performed directly using the models trained with the proposed approach without additional procedures.

A ADDITIONAL RESULTS

In BID2 , a method to minimize not only the loss of the network output but also its derivative is proposed, called Sobolev training, and applied to the critic training algorithm.

We also conduct an experiment to use the Sobolev training method in our proposed algorithm.

The results are shown in TAB6 .

In comparison to the performance shown in Table 1 , we do not observe significant difference overall.

In addition, we test the deep supervision algorithm BID18 , which also has additional modules connected to intermediate layers.

The table shows that its performance is not significantly different from that of backpropagation.

BID2 , and deep supervision BID18 .

We examine the effectiveness of the proposed method for larger networks than those used in Section 3.

For this, ResNet-50 and ResNet-101 BID6 are trained with backpropagation or the proposed method using three local critic networks.

The results shown in TAB8 have a similar trend to those in Table 1 with slight performance improvement in most cases, which confirm that the proposed method works successfully for relatively complex networks.

In addition, we experiment using the ImageNet dataset BID4 , which is much larger and more complex than CIFAR-10 and CIFAR-100.

The results for ResNet-50 in TAB9 show that the proposed method can also work well for large datasets.

In order to analyze the trained networks, we obtain the representational dissimilarity matrix (RDM) BID10 from each layer of the networks trained by backpropagation and the proposed method.

For each of 400 samples from CIFAR-10, the activation of each layer is recorded, and the correlation between two samples is measured, which is shown in FIG3 .

In the figure, clear diagonal-blocks indicate that samples from the same class have highly correlated representations (e.g., the last layer).

Overall, block-diagonal patterns become clear at the last layers for all cases.

However, the figure also shows that the two training methods result in networks showing qualitatively different characteristics in their internal representations.

In particular, the layers at which local critic networks are attached (e.g., layer 5 in LC (1) and LC (3), and layer 6 in LC (5)) show relatively distinguishable blockdiagonal patterns in comparison to those of the network trained by backpropagation.

These layers in the proposed method act not only as intermediate layers of the main network but also as near-final layers of the sub-models, and thus are forced to learn class-specific representations to some extent.

C LEARNING DYNAMICS

As a way to investigate the learning dynamics of the proposed method, the loss values at each local critic network (L i ) for individual data over the training iterations are examined BID3 .

Figure 3 visualizes the loss values for 400 sampled data of CIFAR-10 (arranged in 2D) when three local critic networks are used.

For visualization, the same results are shown twice, once sorted by labels ( Figure 6a ) and once sorted by L 4 at iteration 20000 (Figure 6b ).

At the early stage of learning, the loss values at the local critic networks (L 1 to L 3 ) are largely different from those at the main network (L 4 ), with only slight similarity (e.g., the blue region at iteration 50 in Figure 6a ).

At the later stage, however, all the losses similarly converged to small values for most of the samples (at iteration 20000 in Figure 6b ).

We showed the test performance of sub-models in TAB3 .

In addition, we show their test accuracy over training iterations for CIFAR-10 in Figure 7 .

In particular, faster convergence in the case of local critic training than backpropagation is observed in Figure 7a .

<|TLDR|>

@highlight

We propose a new learning algorithm of deep neural networks, which unlocks the layer-wise dependency of backpropagation.

@highlight

An alternative training paradigm for DNIs in which the auxiliary module is trained to approximate directly the final output of the original model, offering side benefits.

@highlight

Describes a method of training neural networks without update locking.