Claims from the fields of network neuroscience and connectomics suggest that topological models of the brain involving complex networks are of particular use and interest.

The field of deep neural networks has mostly left inspiration from these claims out.

In this paper, we propose three architectures and use each of them to explore the intersection of network neuroscience and deep learning in an attempt to bridge the gap between the two fields.

Using the teachings from network neuroscience and connectomics, we show improvements over the ResNet architecture, we show a possible connection between early training and the spectral properties of the network, and we show the trainability of a DNN based on the neuronal network of C.Elegans.

Training only the edge-weights of the C.Elagans neuronal network still resulted in 55.7% top-1 accuracy on the MNIST dataset.4.

Retaining of the small-world topology throughout the training phase.

Specifics: Small-worldness of the topology measured by the small-world propensity metric used for weighted graphs was > 0.53 for all epochs of the training period for the MNIST dataset and the C.Elgenans neuronal network.

There exists a formidable body of work in network neuroscience BID2 and connectomics BID4 that deals with topological modeling of the brain's anatomical and functional networks.

Some of the main claims from this field are:1.

Complex networks with small world topology BID13 serve as an attractive model for the organization of brain anatomical and functional networks because a small-world topology can support both segregated/specialized and distributed/integrated information processing BID1 2.

Human brain structural and functional networks follow small-world configuration and this small-world model captures individual cognition and exhibits physiological basis.

BID10 3.

Small-world, modules and hubs are present during the mid-gestation period.

Early brain network topology can predict later behavioral and cognitive performance. (Zhao et al., 2018; BID13 It is therefore somewhat striking that the field of deep neural networks, with all its neuro-biologically inspired building blocks 1 , has mostly left the topology story out.

There have been some nascent attempts worth a mention 1 These sentences appear verbatim in BID9 : The convolutional and pooling layers in ConvNets are directly inspired by the classic notions of simple cells and complex cells in visual neuroscience, and the overall architecture is reminiscent of the here.

In BID11 ) the authors investigated multilayered feed-forward networks trained with back propagation and found that the networks with the small-world connectivity resulted in lower learning error and lower training time when benchmarked with networks of regular or random connectivity.

We note in passing that some specifics of the experimentation used in this paper have been challenged in BID5 .

Continuing with small-world feedforward neural networks theme, the authors in BID3 obtained better results for a diabetes diagnosis dataset when compared with plain-vanilla fully connected feed-forward neural networks.

But in the post-AlexNet era, ever with this seemingly cambrian-explosion of neural network architectures 2 , none seem to be inspired by the ideas prevalent in the domain of brain connectomics.

In this paper, we showcase a series of experimental attempts we've made to bridge the gap between the two communities (namely connectomics/network-neuroscience and deep learning).

While it is not yet clear as to what deep neural network module would serve to be proverbial node in the complex network framework (we initially tried ResNet modules -See RogueNet section below), we firmly believe that the recent work by BID14 serves as a truly elegant template and is worthy as the de facto choice for further modeling.

The rest of the paper is organized as follows: In section 2, we present RogueNet.

A DNN model architecture with small-world topological properties where we used a ResNet module to be a node.

In section-3, we present two architectures, RamanujanNet and C.ElgansNet based on the node model proposed in BID14 .

In section-4, we conclude the paper and present the current directions of research we are currently pursuing to extend this work.

Our proposed RogueNet architecture expands upon the PreActivation ResNet architecture in which ReLu activations and batch normalization are performed before convolutions BID7 .

We introduce random additive connections between ResNet modules in a global manner according to a connectivity pattern resembling a Watts-Strogatz small world network.

The additive skip connections are combined with the activation from a given ResNet module using a convolutional gating mechanism inspired by Highway Net-LGN-V1-V2-V4-IT hierarchy in the visual cortex ventral pathway.

When ConvNet models and monkeys are shown the same picture, the activations of high-level units in the ConvNet explains half of the variance of random sets of 160 neurons in the monkeys inferotemporal cortex.

http://www.asimovinstitute.org/ neural-network-zoo/ to visualize the growing neuralnetwork zoo (1x1) works BID12 .

The base architecture which we apply random connections to is composed of Pre-Activation ResNet modules.

These modules come in two flavors as proposed by He et al., the first of which is referred to as a 'basic' module which contains two convolutional layers, and the second of which is referred to as a 'bottleneck' module which contains three convolutional layers BID6 .

The details of each of these are presented in TAB0 .The experiments presented here rely primarily on the use of basic ResNet modules, and we compare RogueNet architectures consisting of 34 layers (excluding gating layers) to ResNet34, which similarly has 34 layers.

Each ResNet module takes input from the previous ResNet module or layer and skip connections from previous layers in the network pass activations which are added to it before being passed to a gating mechanism and being sent to one or more later layers in the network.

The connectivity pattern of the skip connections is a Watts Strogatz random network with mean edge density and rewiring probability set such that it has small world properties.

Our neuron architecture can be seen in Figure 3 The gating mechanism used in the RogueNet architecture is inspired by Highway layers, introduced by Srivastava et al. BID12 .

The original formulation for Highway layers is expressed as DISPLAYFORM0 where H(x, W H ) is an arbitrary fully connected layer with input x and parameters W H .

T (x, W T ) is specified to be a fully connected layer parameterized by W T and a bias vector with input x with the sigmoid activation function.

Our gating mechanism differs from this in that we use convolutional layers, and our transformation H(x, ·) sums the output and the activations from the incoming skip connections.

Mathematically, this is described as follows DISPLAYFORM1 where L is the index of the current layer and S is the set of indices of layers which have a skip connection to layer L. X j where j refers to the index of a particular layer, is the activation from the gating mechanism from layer j.

In this case, T conv (x L , ·) is a 1 by 1 convolution with a bias term and the sigmoid activation function.

Intuitively, this layer learns how much the incoming skip connection activations should be weighted relative to the activation from the previous module.

We use the same training procedure for each of the RogueNet experiments, including our comparison to ResNet.

We trained for 200 epochs on the CIFAR-10 dataset using SGD with a learning rate of 0.1, momentum set to 0.9, weight decay set to 0.0005, and a batch size of 128.

We randomly re-compute the connectivity pattern of the skip connections once per epoch according to the Watts Strogatz recipe for generating random Watts Strogatz graphs.

This process is parameterized by a mean edge density, K, and a rewiring probability, β, which were fixed to be K = 4 and β = 0.08.

K was found empirically, and β was estimated based on the value of K by generating random graphs with various values of β and choosing the β yielding the greatest difference in the normalized average clustering coefficient and the average path length, which are properties of small world networks.

An example result of this computation can be seen in FIG1

In our RogueNet experiments, we explored various configurations of using pre-activations, freezing the highway layer, and shuffling the random skip connections.

We compare our top performing configurations with ResNet34 using the same training procedure on CIFAR-10.

We find that our best configuration involves using pre-activations, using trainable parameters in the gating mechanisms, and shuffling the skip connections once per epoch.

This and the results of the rest of the configurations can be found in We propose an architecture which leverages the spectral properties of the connectivity pattern in the graph, based on the RandWire architecture proposed by Xie et al. BID14 .

In their work, they propose an architecture composed of three randomly connected graph modules, each with random connections between nodes consisting of a single convolutional layer.

In their work, they propose the use of Watts Strogatz, Erdos Renyi, and Barabasi Albert graphs, where a different random graph is used for each of the three modules.

We propose the use of expander graphs where the same graph is replicated across the three modules.

We experiment with expander graphs with various spectral gaps generated by sampling random K-regular graphs and computing the spectral gap.

In our experiments, each graph consists of 36 nodes and 64 edges.

We train the RamaujanNet architecture using 9 different graph topologies with various spectral gaps for 1 epoch each on MNIST and ImageNette using the Adam optimizer with a learning rate of 0.001 BID8 .

We evaluated the first epoch accuracy across randomly sampled K-regular graphs with various spectral gaps.

We discovered that the first epoch accuracy was positively correlated with spectral gap across the MNIST and Imagenette datasets (fas).

The results for these can be found in FIG3 .

Here, we show that the Pearson correlation between first epoch accuracy and spectral gap on MNIST is 0.55, and the Pearson correlation on the Imagenette dataset is 0.65.

We propose another architecture which is also based on RandWire, but this time using only a single graph based module instead of three such consecutive modules as described by BID14 .

In this architecture, we choose our participant graph topology to be a small world network found in nature, namely the neuronal network of C.Elegans.

In this series of experiments, we trained a neural network using this architecture on MNIST, KMNIST, and Fashion MNIST.

An additional experiment was conducted in which we froze every parameter including the final fully connected softmax layer except those corresponding to edge weights in the C.Elegans small world network, and tracked the evolution of small world propensity of the graph over training.

We train the C.ElegansNet architecture with 10 random initializations for 2 epochs each on MNIST, KMNIST, and Fashion MNIST using the Adam optimizer with a learning rate of 0.001 BID8 ).

For our experiment in which we froze all other parameters except for the network edge weights, we trained on MNIST for 20 epochs.

Additionally, we tracked the small world propensity of the C.Elegans graph with the learned edge weights over training.

Before computing the small world propensity, we standardized the neural network parameters and applied the sigmoid activation function to ensure positivity while transforming the parameters in a monotonic way.

We discovered that the C.ElegansNet architecture was able to consistently achieve competitive results on MNIST, KM-NIST, and Fashion MNIST.

Figure 4 shows the distribution of our results compiled from 10 training trials on each dataset.

The mean test accuracy on MNIST was 99%, while we achieved 93% on KMNIST, and 90% on Fashion MNIST.Additionally, upon freezing all parameters traditionally associated with deep neural networks leaving the edge weights as the only trainable parameters, we found that the C.ElegansNet architecture achieved 55.7% accuracy on MNIST.

The loss and accuracy curves for this experiment can be seen in Figure 5 .

For this experiment we also tracked the small world propensity of the C.Elegans graph with the learned edge weights over training and found that it does not significantly change throughout training.

We have demonstrated three distinct approaches to applying work from network neurosciences and connectomics to deep learning.

Our experiments show improvements over ResNet by the inclusion of skip connections which follow a connectivity pattern with small world properties, a possible connection between early training performance and spectral gap when using expander graphs as the participant graph topology with the node model proposed by BID14 , and the trainability of a DNN based on the neuronal network of C.Elegans with and without freezing the parameters of the convolutional and fully connected layers.

In future work, we will examine the impact of other spectral properties of the graph topologies used both in the architectures we proposed and in the RandWire architecture proposed by BID14 .

Additionally, we will explore parameter efficient connectivity patterns which could achieve similar performance to related networks with more parameters TAB0 Deep connectomics networks Figure 5 .

Performance of C.ElegansNet with all parameters frozen except the C.Elegans graph edge weights.

@highlight

We explore the intersection of network neurosciences and deep learning. 