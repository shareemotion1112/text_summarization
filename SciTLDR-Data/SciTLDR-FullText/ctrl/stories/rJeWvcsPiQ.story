Deep neural networks have achieved outstanding performance in many real-world applications with the expense of huge computational resources.

The DenseNet, one of the recently proposed neural network architecture, has achieved the state-of-the-art performance in many visual tasks.

However, it has great redundancy due to the dense connections of the internal structure, which leads to high computational costs in training such dense networks.

To address this issue,  we design a reinforcement learning framework to search for efficient DenseNet architectures with layer-wise pruning (LWP) for different tasks, while retaining the original advantages of DenseNet, such as feature reuse, short paths, etc.

In this framework, an agent evaluates the importance of each connection between any two block layers, and prunes the redundant connections.

In addition, a novel reward-shaping trick is introduced to make DenseNet reach a better trade-off between accuracy and float point operations (FLOPs).

Our experiments show that DenseNet with LWP is more compact and efficient than existing alternatives.

Multi-scale DenseNet (Huang et al., 2017a) and CondenseNet(Huang et al., 2018) , have shown that 23 there exists high redundancy in DenseNet.

Neural architecture search (NAS) has been successfully 24 applied to design model architectures for image classification and language models (Liu et al., 2018; 25 Zoph & Le, 2016; Pham et al., 2018; Liu et al., 2017a; Brock et al., 2017) .

However, none of these 26 NAS methods are efficient for DenseNet due to the dense connectivity between layers.

It is thus 27 interesting and important to develop an adaptive strategy for searching an on-demand neural network 28 structure for DenseNet such that it can satisfy both computational budget and inference accuracy 29 requirement.

To this end, we propose a layer-wise pruning method for DenseNet based on reinforcement learning.

Our scheme is that an agent learns to prune as many as possible weights and connections while 32 maintaining good accuracy on validation dataset.

Our agent learns to output a sequence of actions 33 and receives reward according to the generated network structure on validation datasets.

Additionally, 34 our agent automatically generates a curriculum of exploration, enabling effective pruning of neural 35 networks.

Submitted to 32nd Conference on Neural Information Processing Systems (NIPS 2018).

Do not distribute.

Suppose the DenseNet has L layers, the controller needs to make K (equal to the number of layers in 38 dense blocks) decisions.

For layer i, we specify the number of previous layers to be connected in the 39 range between 0 and n i (n i = i).

All possible connections among the DenseNet constitute the action 40 space of the agent.

However, the time complexity of traversing the action space is O( DISPLAYFORM0 which is NP-hard and unacceptable for DenseNet (Huang et al., 2017b DISPLAYFORM1 where f denotes the controller parameterized with θ c .

The j-th entry of the output vector o i , denoted

by o ij ∈ [0, 1], represents the likelihood probability of the corresponding connection between the 53 i-th layer and the j-th layer being kept.

The action a i ∈ {0, 1} ni is sampled from Bernoulli(o i ).

a ij = 1 means keeping the connection, otherwise dropping it.

Finally, the probability distribution of 55 the whole neural network architecture is formed as: DISPLAYFORM0 The reward function is designed for each sample and not only considers the prediction correct or not, 57 but also encourages less computation: obtaining the feedback from the child network, we define the following expected reward: DISPLAYFORM1 DISPLAYFORM2 To maximize Eq (4) and accelerate policy gradient training over θ c , we utilize the advantage actor-62 critic(A2C) with an estimation of state value function V (s; θ v ) to derive the gradients of J(θ c ) as: DISPLAYFORM3 The entire training procedure is divided into three stages: curriculum learning, joint training and 66 training from scratch and they are well defined in Appendix 4.1.

Algorithm 1 shows the complete 67 recipe for layer-wise pruning.

3 Experiment and conclusion

The results on CIFAR are reported in which takes much search time complexity and needs more parameters but gets higher test error.

We can also observe the results on CIFAR-100 from the last t layers and keeps the policy of the remaining K − t layers consistent with the vanilla DenseNet.

As t ≥ K, all block layers are involved in the decision making process.

Training from scratch.

After joint training, several child networks can be sampled from the policy 146 distribution π(a|s, θ c ) and we select the child network with the highest reward to train from scratch,

and thus better experimental results have been produced.

We summarize the entire process in Algorithm 1.

The pseudo-code for layer-wise pruning.

Input: Training dataset Dt; Validation dataset Dv; Pretrained DenseNet.

Initialize the parameters θc of the LSTM controller and θv of the value network randomly.

Set epochs for curriculum learning, joint training and training from scratch to M cl , M jt and M f s respectively and sample Z child networks.

Output: The optimal child network 1: //Curriculum learning 2: DISPLAYFORM0 if t < K − t then 5: DISPLAYFORM1 end if 10:Sample a from Bernoulli(o) 11:DenseNet with policy makes predictions on the training dataset Dt 12:Calculate feedback R(a) with Eq (3) 13:Update parameters θc and θv 14: end for 15: //Joint training 16: for t = 1 to M jt do 17:Simultaneously train DenseNet and the controller 18: end for 19: for t = 1 to Z do 20:Sample a child network from π(a|s, θc) 21:Execute the child network on the validation dataset Dv 22:Obtain feedback R (t) (a) with Eq (

In this section, we argue that our proposed methods can learn more compact neural network architec-

ture by analyzing the number of input channel in DenseNet layer and the connection dependency 152 between a convolution layer with its preceding layers.

In FIG3

The input channel is 0 means this layer is dropped so that the block layers is reduced from 36 to FIG3 right, the x, y axis define the target layer t and source layer s.

The small square at position (s, t) represents the connection dependency of target layer t on source that the values of these small square connecting the same target layer t are almost equal which means 167 the layer t almost has the same dependency on different preceding layers.

Naturally, we can prove 168 that the child network learned from vanilla DenseNet is quite compact and efficient.

<|TLDR|>

@highlight

Learning to Search Efficient DenseNet with Layer-wise Pruning