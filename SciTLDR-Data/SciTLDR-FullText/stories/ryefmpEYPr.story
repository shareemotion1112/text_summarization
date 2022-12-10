Deep neural networks have demonstrated unprecedented success in various knowledge management applications.

However, the networks created are often very complex, with large numbers of trainable edges which require extensive computational resources.

We note that many successful networks nevertheless often contain large numbers of redundant edges.

Moreover, many of these edges may have negligible contributions towards the overall network performance.

In this paper, we propose a novel iSparse framework and experimentally show, that we can sparsify the network, by 30-50%, without impacting the network performance.

iSparse leverages a novel edge significance score, E, to determine the importance of an edge with respect to the final network output.

Furthermore, iSparse can be applied both while training a model or on top of a pre-trained model, making it a  retraining-free approach - leading to a minimal computational overhead.

Comparisons of iSparse against PFEC, NISP, DropConnect, and Retraining-Free on benchmark datasets show that iSparse leads to effective network sparsifications.

Deep neural networks (DNNs), particularly convolutional neural networks (CNN), have shown impressive success in many applications, such as facial recognition (Lawrence et al., 1997) , time series analysis (Yang et al., 2015) , speech recognition (Hinton et al., 2012) , object classification (Liang & Hu, 2015) , and video surveillance (Karpathy & et.

at., 2014) .

As the term "deep" neural networks implies, this success often relies on large networks, with large number of trainable edges (weights) (Huang et al., 2017; Zoph et al., 2018; He et al., 2016; Simonyan & Zisserman, 2015) .

While a large number of trainable edges help generalize the network for complex and diverse patterns in large-scale datasets, this often comes with enormous computation cost to account for the non-linearity of the deep networks (ReLU, sigmoid, tanh) .

In fact, DNNs owe their recent success to hardware level innovations that render the immense computational requirements practical (Ovtcharov & et.

al., 2015; Matthieu Courbariaux et al., 2015) .

However, the benefits of hardware solutions and optimizations that can be applied to a general purpose DNN or CNN are limited and these solutions are fast reaching their limits.

This has lead to significant interest in networkspecific optimization techniques, such as network compression (Choi & et.

al., 2018) , pruning (Li et al., 2016; Yu et al., 2018) , and regularization (Srivastava & et.

al., 2014; Wan et al., 2013) , aim to reduce the number of edges in the network.

However, many of these techniques require retraining the pruned network, leading to the significant amount of computational waste.

Many successful networks nevertheless often contain large numbers of redundant edges.

Consider for example, the weights of sample network shown in Figure 1a .

As we see here, the weight distribution is centered around zero and has significant number of weights with insignificant contribution to the network output.

Such edges may add noise or non-informative information leading to reduction in the network performance. (Denil et al., 2013; Ashouri et al., 2018; Yu et al., 2018) has shown that it is possible to predict 95% network parameters while only learning 5% parameters.

Sparsification techniques can generally be classified into neuron/kernel sparcification (Li et al., 2016; Yu et al., 2018) and edge/weight sparcification techniques (Wan et al., 2013 ; Ashouri et al., Figure 1 : Overview of weight distribution and model accuracies for MNIST dataset 2018): (Li et al., 2016) proposed to eliminate neurons that have low l2-norms of their weights, whereas (Yu et al., 2018) proposed a neuron importance score propagation (NISP) technique where neuron importance scores (using Roffo & et.

al. (2015) -See Equation 5 ) are propagated from the output layer to the input layer in a back-propagation fashion.

Drop-out (Srivastava & et.

al., 2014) technique instead deactivates neuron activations at random.

As an edge sparsification technique, DropConnect (Wan et al., 2013 ) selects edges to be sparsified randomly. (Ashouri et al., 2018) showed that the network performance can be maintained by eliminating insignificant weights without modifying the network architecture.

Following these works, we argue network sparsification can be a very effective tool for reducing sizes and complexities of DNNs and CNNs, without any significant loss in accuracy.

However, we also argue that edge weights cannot be used "as is" for pruning the network.

Instead, one needs to consider the significance of each edge within the context of their place in the network (Figure 2 ): "Two edges in a network with the same edge weight may have different degrees of contributions to the final network output" and in this paper, we show that it is possible to quantify significance of each edge in the network, relative to their contributions to the final network output and use these measures significance to minimize the redundancy in the network by sparsifying the weights that contributes insignificantly to network.

We, therefore, propose a novel iSparse framework, and experimentally show, that we can sparsify the network, by almost 50%, without impacting the network performance.

The key contributions of our proposed work are as follows:

• Output-informed quantification of the significance of network parameters: Informed by the final layer network output, iSparse computes and propagates edge significant scores that measure the importance of each edge with respect to the model output (Section 3).

• Retraining-free network sparsification (Sparsify-with): The proposed iSparse framework is robust to edge sparsification and can maintain network performance without having to retraining the network.

This implies that one can apply iSparse on pre-trained networks, on-the-fly, to achieve the desired level of sparsification (Section 3.3) • Sparsification during training (Train-with): iSparse can also be used as a regularizer during the model training allowing for learning of sparse networks from scratch (Section 4).

As the sample results in Figure 1b shows, iSparse is able to achieve 30-50% sparsification with minimal impact on model accuracy.

More detailed experimental comparisons (See Section 5) of iSparse against PFEC, NISP, Retraining-Free and DropConnect on benchmark datasets illustrated that iSparse leads to more effective network sparsifications.

A neural network is a sequence of layers of neurons to help learn (and remember) complex nonlinear patterns in a given dataset (Grossberg, 1988) .

Recently, deep neural networks (DNNs), and particularly CNNs, which leverage recent hardware advances to increase the number of layers in the

-.*,)$#*)#*/0#)10+%22#).*3.

* Figure 2 : Overview of iSparse sparsification, considering the n i 's contribution to overall output rather than only between n i and n j neurons network to scales that were not practical until recently, (Lawrence et al., 1997; Yang et al., 2015; Hinton et al., 2012; Liang & Hu, 2015; Karpathy & et.

at., 2014) have shown impressive success in several data analysis and machine learning applications.

A typical CNN consists of (1) feature extraction layers are responsible for learning complex patterns in the data and remember them through layer weights.

, by training for a weight matrix W l R m l ×n l (see Section 3.1 for further details); (2) activation layers, which help capture non-linear patterns in the data through activation functions (σ) which maps the output from a feature extraction layer to a nonlinear space ( ReLU and softmax are commonly used activation functions); and (3) pooling layers, (sampling) which up-or down-sample the intermediate data in the network.

The training process of a neural network often comprises of two key stages: (1) forward-propagation (upstream) maps the input data, X, to an output variable,Ŷ .

At each layer, we haveŶ The number of trainable parameters in a deep network can range from as low as tens of thousands (LeCun et al., 1999) to hundreds of millions (Simonyan & Zisserman, 2015) (Table 1 in Section 5).

The three order increase in the number trainable parameter may lead to parameters being redundant or may have negligible contribution to the overall network output.

This redundancy and insignificance of the network parameters has led to advancements in network regularization, by introducing dynamic or informed sparsification in the network.

These advancements can be broadly classified into two main categories: parameter pruning and parameter regularization.

In particular, pruning focuses on compressing the network by eliminating the redundant or insignificant parameters. (Han et al., 2015; Han & et.

al., 2016) aims to prune the parameters with near-zero weights inspired from l 1 and l 2 regularization (Tibshirani, 1996; Tikhonov, 1963) . (Li et al., 2016) choose to filter out convolutional kernel with minimum weight values in given layer.

Recently, (Yu et al., 2018) minimizes the change in final network performance by eliminating the neuron that have minimal impact on the network output by leveraging neuron importance score (N L ) (See Section 5.3) -computed using Inf-FS (Roffo & et.

al., 2015) .

More complex approaches have been proposed to tackle the problem of redundancy in the network through weight quantization. (Rastegari & et.

al., 2016) propose to the quantize the inputs and output activations of the layers in a CNN by using step function and also leveraging the binary operation by using the binary weights opposed to the real-values weights. (Chen & et.

al., 2015) focuses on low-level mobile hardware with limited computational power, and proposed to leverage the inherent redundancy in the network for using hashing functions to compress the weights in the network. (Bahdanau et al., 2014; Woo et al., 2018) showed that the each input feature to a given layer in the network rarely have the same importance, therefore, learning there individual importance (attention) helps improve the performance of the network.

More recently, (Garg & Candan, 2019) has shown that input data informed deep networks can provide high-performance network configurations.

In this paper, we rely on output information for identifying and eliminating insignificant parameters from the network, without having to update the edge weights or retraining the network.

As discussed in Section 1, in order to tackle complex inputs, deep neural networks have gone increasingly deeper and wider.

This design strategy, however, often results in large numbers of insignificant edges 1 (weights), if not redundant.

In this section, we describe the proposed iSparse, framework which quantifies the significance of each individual connection in the network with respect to the overall network output to determine the set of edges that can be sparsified to alleviate network redundancy and eliminate insignificant edges.

iSparse aims to determine the significance of the edges in the network to make informed sparsification of the network.

A typical neural network, N , can be viewed as a sequential arrangement of convolutional (C) and fully connected (F) layers:

, here, X is the input, L is the total number of layers in the network and L {C, F} , s.t., any given layer,

where, Y l is the input to the layer (s.t.

Y l =Ŷ l−1 and for l = 1, Y 1 = X) and σ l , W l , and B l are the activation function, weight, and bias respectively.

Note that, if the l th layer has m l neurons and the (l − 1) th layer has n l neurons ,

Given this formulation, the problem of identifying insignificant edges can be formulated as the problem of generating a sequence of binary mask matrices, M 1 , . . .

, M L , that collectively represents whether any given edge in the network is sparsified (0) or not (1):

Let M l be a mask matrix as defined in Equation 2, and M l can be expanded as

where each M l,i,j ∈ {0, 1} corresponds to an edge e l,i,j in the network.

Our goal in the paper is to develop an edge significant score measure to help set the binary value of M l,i,j for each edge in the network.

More specifically, we aim to associate a non-negative real valued number, E l,i,j ≥ 0, to each edge in the network, s.t.

Here, τ l (θ l ) represents the lowest significance of the θ l % of the most significant edges in the layer l. Intuitively, given a target sparsification rate, θ l , we rank all the edges based on their edge significance scores and keep only the highest scoring θ l % of the edges by setting their mask values to 1.

As we have seen in Figure 1a , the (signed) weight distribution of the edges in a layer is often centered around zero, with large numbers of edges having weights very close to 0.

As we also argued in the Introduction, such edges can work counter-intuitively and add noise or non-informative information leading to reduction in the network performance.

In fact, several existing works, such as (Ashouri et al., 2018) , relies on these weights for eliminating insignificant edges without having to retrain the network architecture.

However, as we also commented in the Introduction, we argue that edge weights should not be used alone for sparsifying the network.

Instead, one needs to consider each edge within the context of their place in the network: Two edges in a network with the same edge weight may have different degrees of contributions to the final network output.

Unlike existing works, iSparse takes this into account when selecting the edges to be sparsified (Figure 3 ).

Figure 3: A sample network architecture and its sparsification using Retraining-free (Ashouri et al., 2018) and iSparse; here node labels indicate input to the node; edge labels [0,1] indicate the edge weights; and edge labels between parentheses indicate edge contribution

More specifically, let W + l be the absolute positive of the weight matrix,W l , for edges in l th layer.

We compute the corresponding edge significance score matrix, E l , as

where, N l represents the neuron significance scores 2 , N l,1 through N l,n l , and " " represents the scalar multiplication between edge weights and neuron scores.

N l,i , denotes the significance of the i th input neuron to the l th connection layer of the network, which itself is defined recursively, based on the following layer in the network, using the conventional dot product:

Note that N l can be expanded as

Above, N L denotes the neuron scores of the final output layer, and N L is defined using infinite feature selection (Roffo & et.

al., 2015; Yu et al., 2018) as

x is the number of input samples and n is the number of output neurons) to determine neuron importance score with respect to the the final network output.

Given the above, the edge score (Equation 5) can be rewritten as

Note that the significance scores of edges in layer l considers not only the weights of the edges, but also the weights of all downstream edges between these edges and the final output layer.

As noted in Section 3.1, the binary values in the masking matrix M l depends on τ l (θ l ), which represents the lowest significance of the θ l % of the most significant edges in the layer 3 : therefore, given a target sparsification rate, θ l , for layer l, we rank all the edges based on their edge significance scores and keep only the highest scoring θ l % of the edges by setting their mask values to 1.

Note that, once an edge is sparsified, change in its contribution is not propagated back to the layers earlier in the network relative to the sparsified edge.

Having determined the insignificant edges with respect to the final layer output, represented in form of the mask matrix, M l (described in Section 3.1), the next step is to integrate this mask matrix in the layer itself.

To achieve this, iSparse extends the layer l (Equation 1) to account for the corresponding mask matrix (M l ): where, * represents the element-wise multiplication between the matrices W l and M l .

Intuitively, M l facilitates introduction of informed sparsity in the layer by eliminating edges that do not contribute significantly to the final output layer.

In the previous section, we discussed the computation of edge significance scores on a pre-trained network, such as of pre-trained ImageNet models, and the use of these scores for network sparsification.

In this section, we highlight that iSparse can also be integrated directly within the the training process.

To achieve this, the edge significance score is computed for every trainable layer in the network using the strategy described in Section 3.2 and the mask matrix is updated using Equation 4.

Furthermore, the back-propagation rule, described in Section 2 , is updated to account for the mask matrices:

where, W l are the updated weights, W l original weights, η is the learning rate, and Err l is the error recorded by as the divergence in between ground truth (Y l ) and model predictions (Ŷ l ) as

Note that, we argue that any edge that does not contribute towards the final model output, must not be included in the back-propagation.

Therefore, we mask the error as Err l * M l .

In this section, we experimentally evaluate of the proposed iSparse framework using LeNet and VGG architectures (See Section 5.2) and compare it against the approaches, such as PFEC, NISP, and DropConnect (see Section 5.3).

We implemented iSparse in Python environment (3.5.2) using Keras Deep Learning Library (2.2.4-tf) (Chollet et al., 2015) with TensorFlow Backend (1.14.0).

All experiments were performed on an Intel Xeon E5-2670 2.3 GHz Quad-Core Processor with 32GB RAM equipped with Nvidia Tesla P100 GPU with 16 GiB GDDR5 RAM with CUDA-10.0 and cuDNN v7.6.4.

4 .

In this paper, without loss of generality, we leverage LeNet-5 (LeCun et al., 1999) and VGG-16 (Simonyan & Zisserman, 2015) as the baseline architectures to evaluate sparsification performance on different benchmark image classification datasets and for varying degrees of edge sparsification.

In this section, we present an overview of these architectures (See Table 1 ).

LeNet-5: Designed for recognizing handwritten digits, LeNet-5 is simple network with 5 trainable (2 convolution and 3 dense) and 2 non-trainable layers using average pooling with tanh and softmax as the hidden and output activation.

LeNet's simplicity has made it a common benchmark for datasets recorded in constrained environments, such as MNIST (LeCun et al., 1998) , FMNIST (Xiao et al., 2017) , COIL (Nene et al., 1996a; b) , and NORB (LeCun et al., 2004) .

VGG-16: VGG (Simonyan & Zisserman, 2015) 's, a 16 layer network with 13 convolution and 3 dense layers, with interleaved 5 max-pooling layers.

VGG leverages ReLU as the hidden activation to overcome the problem of vanishing gradient, as opposed to tanh.

Given the ability of VGG network to learn the complex pattern in the real-world dataset, we use the network on benchmark datasets, such as CIFAR10/20/100 (Krizhevsky, 2009 ), SVHN (Netzer & et.

al., 2011) , GTSRB (Stallkamp & et.

al., 2012) , and ImageNet (Deng et.

al., 2009) .

Table 1 reports the number of trainable parameters (or weights) for each model/data set pair considered in the experiments.

We compared iSparse against several state-of-the-art network sparsification techniques: DropConnect (Wan et al., 2013 ) is a purely random approach, where edges are randomly selected for sparsification.

Retraining-free (Ashouri et al., 2018) considers each layer independently and sparsifies insignificant weights in the layer, without accounting for the final network output contribution.

PFEC (Li et al., 2016 ) is a kernel pruning strategy that aims to eliminate neurons that have low impact on the overall model accuracy.

In order to determine the impact, PFEC computes the l2-norms of the weights of the neurons and ranks them, separately, for each layer.

NISP (Yu et al., 2018) proposes a neuron importance score propagation (NISP) technique where neuron importance scores are propagated from the output layer to the input layer in a back-propagation fashion.

Figure 7: Mask matrices for the LeNet network conv 2 layer for MNIST data (sparsification factor = 50%): dark regions indicate the edges that have been marked for sparsification; in (e) iSparse, the arrows point to those edges that are subject to different pruning decision from retrain-free in(d) (green arrows point to edges that are kept in iSparse instead of being pruned and red arrows point to edges that are sparsified in iSparse instead of being kept)

In Figure 4 , we first present top-1 and top-5 classification results for ImageNet dataset for VGG-16 network.

As we see in the Figure 4 , iSparse provides the highest robustness to the degree of sparsification in the network.

In particular, with iSparse , the network can be sparsified by 50% with ≤ 6% drop in accuracy for top-1 and ≤ 2% drop in accuracy for top-5 classification, respectively.

In contrast, the competitors, see larger drops in accuracy.

The closest competitor, Retrain-free, suffers a loss in accuracy of ∼ 16% and ∼ 6% for top-1 and top-5 classification, respectively.

The other competitors suffer significant accuracy drops after a mere 10-20% sparsification.

Figures 6a and 6b show the top-1 classification accuracy results for other models and data sets.

As we see here, the above pattern holds for all configurations considered: iSparse provides the best robustness.

It is interesting to note that DropConnect, NISP, and PFEC see especially drastic drops in accuracy for the VGG-16 network and especially on the CIFAR data.

This is likely because, VGG-CIFAR is already relatively sparse (20% > sparsity as opposed to ∼ 7% for VGG-ImageNet and < 1% for LeNet) and these three techniques are not able to introduce additional sparseness in a robust manner.

In contrast, iSparse is able to introduce significant additional sparsification with minimal impact on accuracy.

Figure 7 provides the mask matrices created by the different algorithms to visual illustrate the key differences between the competitors.

As we see in this figure, PFEC and NISP, both sparsify input neurons.

Consequently, their effect is to mask out entire columns from the weight matrix and this prevents these algorithms to provide fine grain adaption during sparsification.

DropConnect selects individual edges for sparsification, but only randomly and this prevents the algorithm to provide sufficiently high robustness.

Retrain-free and iSparse both select edges in an fine-grained manner: retrain-free uses relies on edge-weights, whereas iSparse complements edge-weight with an edge significance measure that accounts for each edges contribution to the final output within the overall network.

As we see in Figure 7 (d) and (e), this results in some differences in the corresponding mask matrices, and these differences are sufficient to provide significant boost in accuracy.

Tables 2 present accuracy results for the scenarios where iSparse (iS) is used to sparsify the model during the training process.

The table also considers DropConnect (DC) and Retrain-Free (RF), as alternatives.

As we see in the table, for both network architectures, under most sparsification rates, the output informed sparsification approach underlying iSparse leads to networks with the highest classification accuracies.

In this section, we study the effect of the variations in network elements.

In particular, we compare the performance of iSparse (iS) against DropConnect (DC) and Retraining-Free (RF) for different hidden activation functions and network optimizers.

Table 3 presents classification performances for networks that rely on different activation functions (tanh and ReLU) and for optimizers (Adam and RMSProp).

As we see in these two tables, iSparse remains the alternative which provides the best classification accuracy under different activation/optimization configurations.

We next investigate the performance of iSparseunder different orders in which the network layers are sparsified.

In particular, we considerthree sparsification orders: (a) input-to-output layer order: this is the most intuitive approach as it does not require edge significance scores to be revised based on sparsified edges in layers closer to the input; (b) output-to-input layer-order: in this case, edges in layers closer to the network output are sparsified first -but, this implies that edge significance scores are updated in the earlier layers in the network to account for changes in the overall edge contributions to the network; (c) random layer order: in this case, to order of the layers to be sparsified is selected randomly.

Figure 8 presents the sparsification results for different orders, data sets, and sparsification rates.

As we see in the figure, the performance of iSparse is not sensitive to the sparsification order of the network layers.

In Figure 5 , we investigate the impact of edge sparsification on the classification time.

As we see in this Figure, edge sparsification rate has a direct impact on the classification time of the resulting model.

When we consider that iSparse allows for ∼ 30 − 50% edge sparsification without any major impact on classification accuracies, this indicates that iSparse has the potential to provide significant performance gains.

What is especially interesting to note in Figure 5 is that, while all three sparsification methods, iSparse, DropConnect, and Retraining-Free, all have the same number of sparsified edges for a given sparsification factor, the proposed iSparse approach leads to the least execution times among the three alternatives.

We argue that this is because the output informed sparsification provided by iSparse allows for more efficient computations in the sparsified space 5 .

In this paper, we proposed iSparse, a novel output-informed, framework for edge sparsification in deep neural networks (DNNs).

In particular, we propose a novel edge significance score that quantifies the significance of each edge in the network relative to its contribution to the final network output.

iSparse leverages this edge significance score to minimize the redundancy in the network by sparsifying those edges that contribute least to the final network output.

Experiments, with 11 benchmark datasets and using two well-know network architectures have shown that the proposed iSparse framework enables 30 − 50% network sparsification with minimal impact on the model classification accuracy.

Experiments have also shown that the iSparse is highly robust to variations in network elements (activation and model optimization functions) and that iSparse provides a much better accuracy/classification-time trade-off against competitors.

@highlight

iSparse eliminates irrelevant or insignificant network edges with minimal impact on network performance by determining edge importance w.r.t. the final network output. 