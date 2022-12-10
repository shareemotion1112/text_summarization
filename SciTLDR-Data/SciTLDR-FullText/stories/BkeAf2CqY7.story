As an emerging field, federated learning has recently attracted considerable attention.

Compared to distributed learning in the datacenter setting, federated learning has more strict constraints on computate efficiency of the learned model and communication cost during the training process.

In this work, we propose an efficient federated learning framework based on variational dropout.

Our approach is able to jointly learn a sparse model while reducing the amount of gradients exchanged during the iterative training process.

We demonstrate the superior performance of our approach on achieving significant model compression and communication reduction ratios with no accuracy loss.

Federated Learning is an emerging machine learning approach that has recently attracted considerable attention due to its wide range of applications in mobile scenarios BID18 BID12 BID24 .

It enables geographically distributed devices such as mobile phones to collaboratively learn a shared model while keeping the training data on each phone.

This is different from standard machine learning approach which requires all the training data to be centralized in a server or in a datacenter.

As such, federated learning enables distributing the knowledge across phones without sharing users' private data.

Federated Learning uses some form of distributed stochastic gradient descent (SGD) and requires a parameter server to coordinate the training process.

The server initializes the model and distributes it to all the participating devices.

In each distributed SGD iteration, each device computes the gradients of the model parameters using its local data.

The server aggregates the gradients from each device, averages them, and sends the averaged gradients back.

Each device then updates the model parameters using the averaged gradients.

In such manner, each device benefits from obtaining a better model than the one trained only on the locally stored private data.

While federated learning shares some common features with distributed learning in the datacenter setting BID3 BID16 since they both use distributed SGD as the core training technique, federated learning has two more strict constraints which datacenter setting does not have:Model Constraint: Compared to datacenters, mobile devices have much less compute resources.

This requires the final model learned in the federated learning setting to be computationally efficient so that it can efficiently run on mobile devices.

Communication Constraint: In datacenters, communication between the server and working nodes during SGD is conducted via Gbps Ethernet or InfiniBand network with even higher bandwidth BID26 .

In contrast, communication in the federated learning setting relies on wireless networks such as 4G and Wi-Fi.

Both uplink and downlink bandwidths of those wireless networks are at Mbps scale, which is much lower than the Gbps scale in the datacenter setting.

The limited bandwidth in the federated learning setting illustrates the necessity of reducing the communication cost to accelerate the training process.

In this work, we propose an efficient federated learning framework that meets both model and communication constraints.

Our approach is inspired by variational dropout BID10 .

Our key idea is to jointly and iteratively sparsify the parameters of the shared model to be learned as well as the gradients exchanged between the server and the participating devices during the distributed SGD training process.

By sparsifying parameters, only important parameters are kept, and the final model learned thus becomes computationally efficient run on mobile devices.

By sparsifying gradients, only important gradients are transmitted, and the communication cost is thus significantly reduced.

We examine the performance of our framework on three deep neural networks and five datasets that fit the federated learning setting and are appropriate to be deployed on resource-limited mobile devices.

Our experiment results show that our framework is able to achieve significant model compression and communication reduction ratios with no accuracy loss.

Federated Learning.

As an emerging field, federated learning has recently attracted a lot of attention.

BID18 developed a federated learning approach based on iterative model averaging to tackle the statistical challenge, especially for mobile setting.

BID24 proposed a distributed learning framework based on multi-task learning and demonstrated their approach is able to address the statistical challenge and is robust to stragglers.

BID12 developed structured and sketched update techniques to reduce uplink communication cost in federated learning.

In comparison, our work -to the best of our knowledge -represents the first federated learning framework that achieves model sparsification and communication reduction in a unified manner.

Gradient Compression.

Our work is related to gradient compression in distributed SGD.

One line of research is to use low-bit gradient representation to reduce the gradient communication overhead.

In BID26 , the authors proposed TernGrad which only used three numerical levels to represent gradients.

In BID2 , the authors presented QSGD that allows dynamic trade-off between accuracy and communication cost by choosing bit width of gradients.

As another line of research, some previous works aimed to reduce communication cost by transmitting only a small portion of gradients.

For example, BID11 proposed using the structured update to sparsify the gradients.

BID1 and BID17 used magnitudes as indicators to measure the importance of gradients and proposed to only transmit gradients whose magnitudes are larger than a threshold.

In our work, we use an additional dropout parameter as the importance indicator, and transmit gradients based on the values of the dropout parameters stochastically.

Model Sparsification.

Our work is also related to model sparsification.

Model sparsification aims to reduce the computational intensity of deep neural networks by pruning out redundant model parameters.

In BID5 , the authors developed a parameter pruning method that can sparsify deep neural networks by an order of magnitude without losing the prediction accuracy.

The novelty of our proposed approach lies at integrating model sparsification and communication reduction into a single optimization framework based on training data distributed at different devices.

Bayesian Neural Networks.

Our work is inspired by Bayesian neural networks BID22 .

Conventional neural networks learn scalar weights via maximum likelihood estimation.

In comparison, a Bayesian neural network associates each weight with a distribution.

Having a distribution instead of a single scalar value takes account of uncertainty for weight estimates and is thus more robust to overfitting BID7 .Consider a Bayesian neural network parameterized by weight matrix w. Let p(w) denote the prior distribution of w. Given a dataset D with N data samples (x n , y n ) for n ∈ [1, N ], the posterior distribution of w is p(w|D) = p(D|w)p(w)/p(D).

Unfortunately, p(w|D) is generally computationally intractable, and thus approximation approaches are needed.

One popular approximation approach is variational inference, which uses a parametric distribution q φ (w) to approximate the true posterior distribution p(w|D) BID9 .

The parameters φ of q φ (w) are learned by maximizing the variational lower bound L(φ) defined as follows BID10 : DISPLAYFORM0 The first term on the right side measures the predictive performance of q φ (w).

The second term is the KL-divergence between q φ (w) and p(w), which regularizes q φ (w) to be close to p(w).Variational Dropout.

Dropout is one of the most effective regularization methods for training deep neural networks BID25 Figure 1 : Left: The standard federated learning framework.

The gradients transmitted from each device to the central server is non-sparse, and the shared model learned from the distributed private data is non-sparse.

Right: The proposed efficient federated learning framework based on unified gradient and model sparsification.

The gradients transmitted from each device to the central server is sparse, and the shared model learned from the distributed private data is sparse.noises are added to the weights with a fixed dropout rate p BID25 .

In BID10 , the authors connected dropout with Bayesian neural networks, and showed that a different dropout rate p ij can be learned for each individual weight w ij in w if the Gaussian noise DISPLAYFORM1 As a result, each weight w ij in w has the following Gaussian distribution parameterized by φ ij = (θ ij , α ij ): DISPLAYFORM2 where α ij θ 2 ij is the variance of the Gaussian distribution, and α ij is enforced to be greater than zero.

Given q φij (w ij ) in (2), the first term in (1) can be approximated by Monte Carlo estimation BID10 .

To compute the second term in (1), the authors in (Kingma et al., 2015) adopted a prior distribution p(w) which makes D KL (q φ (w)||p(w)) only depend on α ij .

As a result, the second term in (1) can be approximated by DISPLAYFORM3 where C is a constant and σ is the sigmoid function 1 1 + e −x .

Sparsifying Bayesian Neural Networks via Variational Dropout.

Given that variational dropout allows learning different dropout rates for different weights, the authors in BID19 showed that variational dropout can be used to sparsify Bayesian neural networks by pruning the weights whose learned dropout rates are high while still achieving comparable predictive accuracies comparing to the unpruned ones.

This is because a large dropout rate corresponds to large noise in the weight.

The large noise causes the value of this weight to be random and unbounded, which will corrupt the model prediction.

As a result, pruning this weight out would be beneficial to the prediction performance of the learned neural network.

Our work is inspired by this key finding.

We adopt variational dropout to sparsify neural networks in the federated learning setting.

We describe the details of our approach in the following section.

Standard Federated Learning Framework.

Figure 1 (left) illustrates the standard federated learning framework based on synchronous SGD (we leave asynchronous SGD for future work).

The framework consists of a parameter server and a number of geographically distributed devices.

In each distributed SGD iteration, each device computes the gradients of the model parameters based on its local data, and sends the gradients (non-sparse) to the server.

The server averages the gradients aggregated from every device and sends the averaged gradients (non-sparse) to all devices.

Each device then uses the received averaged gradients to update its local model (non-sparse).

N devices with local private datasets D1, D2, D3, . . .

, D N .

Base model parameter θ 0 , Dropout parameter α 0 , Learning rate η, the threshold value T.

A sparse model w shared by N decentralized devices.

DISPLAYFORM0 1: for each iteration do:2:for i = 1 : N do:3:Device i selects a mini-batch bi from local private dataset Di.

Device i computes the sparse gradients DISPLAYFORM0 , where M (αi) is a 0-1 binary matrix, and each entry M (αij ) is put to 0 if αij > T otherwise 1.

Device i uploads (g θ i , gα i ) to the central server.

The central sever receives the uploaded model parameter gradients and computes their average: DISPLAYFORM0 7:for i = 1 : N do:8:Device i pulls averaged gradients from the central server and updates parameters: DISPLAYFORM1 Given that the standard federated learning framework suffers from both model and communication constraints described in the introduction section, we propose an efficient federated learning framework (Figure 1 (right) ) that meets both model and communication constraints.

Our Algorithm.

Algorithm 1 formulates our proposed federated learning framework based on variational dropout.

Let N denote the number of devices, and DISPLAYFORM2 .., D N be the dataset stored inside the N devices, respectively.

The goal of our proposed federated learning framework is to have all these N devices to collectively learn a sparse model at the final stage while reducing the communication cost during the training process.

Before the training process starts, the server initializes a base model parameterized by (θ 0 , α 0 ), sets the learning rate to be η, and distributes the base model and the learning rate to all the N devices.

As such, all the devices have the same initial parameters to start with.

In practice, α 0 are initialized with random values; θ 0 can be initialized with either random values or the pretrained weights from other dataset such as ImageNet BID4 ).During each distributed SGD iteration, the i th device first selects a mini-batch b i from its local private dataset D i , and computes the gradients of the loss function in (1), including model parameter gradient ∇ θi L i and dropout parameter gradient ∇ αi L i , where θ i , α i are the parameters of the i th device and L i is the variational lower bound of the i th device.

The iterative exchange of gradients (∇ θi L i , ∇ αi L i ) would incur massive communication overhead and becomes the communication bottleneck in standard federated learning framework.

To reduce the communication overhead, our key idea is to utilize the dropout parameters α to identify important gradients to transmit to the server (Line 4).

As explained in the preliminaries section, a large dropout rate corresponds to large noise in the weight.

The large noise causes the value of this weight to be random and unbounded, which will corrupt the model prediction.

As a result, pruning this weight out would be beneficial to the prediction performance of the learned neural network.

Recall DISPLAYFORM3 A large dropout rate p ij leads to a large dropout parameter α ij .

As such, the dropout parameter α ij can be viewed as an importance indicator of its corresponding model parameter θ ij , where higher value α ij corresponds to less important θ ij .Inspired by this property, the i th device is able to identify important gradients (g θi , g αi ) by: DISPLAYFORM4 where M (α i ) is a (0,1)-matrix and represents Hadamard product.

Each entry M (α ij ) in M (α i ) is put to 0 if α ij > T otherwise 1.As shown in the experiment section, M (α i ) will become highly sparse during the training process and thus (g θi , g θi ) will also be highly sparse.

It should be emphasized that although we upload a pair of sparse gradients (g θi , g αi ) (Figure 1 (right) ) compared to non-sparse gradients g wi in the standard federated learning framework (Figure 1 (left) ), due to highly sparse (g θi , g αi ), the total number of exchanged gradients is significantly reduced.

The remaining steps are the same with the standard federated learning framework except that we upload g αi and downloadḡ α .

The server averages the uploaded gradients denoted as (ḡ θ ,ḡ α ) and sends back to the devices.

Finally, each device downloads the averaged parameter gradients from the server and updates the model parameters with the preset learning rate η.

Algorithm Variants.

During our experiments, we have observed two key insights.

First, even though α and θ are independent variables, α is empirically negatively correlated with θ, which is consistent with the observation in BID19 .

Second, θ ij is suppressed by large α ij and it can hardly grow back unless D KL in FORMULA0 is removed.

Based on these observations, we make some variants from Algorithm 1 to further reduce the communication cost.

Specifically, each device is forced to optimize α locally.

It neither uploads the gradients of α to the server nor downloads the gradients of α from the server.

The rationale behind this strategy is that since α is associated with θ which is synchronized every iteration during SGD, α will thus be forced to be almost the same across all the devices.

Thus we can only need to upload gradients of important model parameters g θ and omit the gradient of the dropout parameters.

In our experiments, we have implemented both Algorithm 1 and the variants described above.

Our experimental results on variants show better performance on communication cost reduction while achieving convergence and the same test accuracies as Algorithm 1 and in our experimental section we only show the results on the algorithm variants.

The experimental results on Algorithm 1 is shown in the supplementary material.

Datasets and Deep Neural Networks.

We evaluate the performance of our framework on three deep neural networks and five datasets that fit the federated learning setting.

Specifically, we examined CifarNet ( Protocol.

We use the standard distributed learning framework illustrated in Figure 1 (left) as the baseline.

We implement the baseline and our proposed framework using PyTorch BID21 and conduct experiments on NVIDIA 1080Ti GPUs.

All the experimental settings are the same for the baseline and our proposed framework.

In all experiments, we set the mini-batch size to 128 and used Adam BID8 as the SGD optimizer.

Each dataset is randomly and evenly divided into N non-overlapping parts for N devices.

To achieve state-of-the-art accuracy, all the deep neural networks were pre-trained on ImageNet BID4 ) before they are trained on each of the five datasets.

This setup also emulates the practical use scenario of federated learning in real-world settings where devices are loaded with pre-trained models before they participate in federated learning when new data comes in.

For fair comparison, we run both the baseline and our proposed framework to full convergence on each dataset.

We first examine the Top-1 test accuracy and the model sparsity achieved by our framework when 4 devices are included.

TAB1 lists the top-1 accuracy achieved by the baseline (i.e., non-sparse) and our framework (i.e., sparse) as well as the the model sparsity (defined as the percentage of non-zero weights) for each dataset.

Overall, our framework is able to sparsify the model with 1.4% to 7.2% non-zero weights across three different models trained on five different datasets.

The sparse models have achieved comparable accuracy compared to the non-sparse ones.

In the second set of experiments, we demonstrate the benefits of our framework in reducing communication costs during training.

FIG1 plots the communication cost curves of both the baseline (red dashed line) and our framework (blue solid line) during training on five datasets.

The horizontal axis represents the epoch number during training.

The vertical axis represents the percentage of gradients transmitted during both uploading and downloading stages.

We have two observations from the results.

First, our framework takes more epochs than the baseline to converge.

For example, as shown in FIG1 (a), it takes 20 epochs for our framework to converge while it takes 10 epochs for the baseline.

This is because compared to sending all the gradients during each SGD iteration, sending a sparse version of the gradients leads to a larger variance.

Second, the percentage of gradients transmitted drops very quickly at the beginning of the training process.

This indicates that it only takes a small number of SGD iterations for our framework to learn a model with a significantly high sparsity.

In other words, it demonstrates the efficiency of our framework on learning sparse models in the federated setting.

Given the fact that our approach takes more epochs to converge, the key question we need to answer is whether our approach can reduce the communication costs by enough to offset the overhead caused by the extra epochs.

To answer this question, TAB2 lists the data communication volumes of the baseline (i.e., non-sparse) and our approach (i.e., sparse) for each dataset.

The data communication volume is the total transmission data volume (including both uploading and downloading stages) accumulated from all the SGD iterations until convergence during training.

The value of the data communication volume can be seen as the area under the communication cost curves depicted in FIG1 .

As shown in TAB2 , our approach not only reduces the communication costs by enough to offset the overhead caused by the extra epochs, but also cuts the communication costs to 12.2% to 22.2% of the non-sparse framework across datasets.

The success in reducing the communication costs by this large is attributed to our approach's fast sparsification speed illustrated in FIG1 .

In fact, the transmission of sparse gradients include both gradient value itself and its corresponding index.

However, the corresponding index overhead is negligible due to the high sparsity, and we omit it in calculating the communication cost BID0 .

In this experiment, we examine the scalability of our framework when extending to larger number of devices.

Figure 3 shows the performance of accuracy when extending to 8 and 16 devices, and Figure 4 shows the percentage of non-zero weights, and the percentage of communication cost when extending to 8 and 16 devices.

We have the following observations from the results.

First, our framework doesn't incur any accuracy loss for 8 and 16 devices, demonstrating its robustness across different models and datasets.

Second, we observe that our framework scales well when the number of devices increases to 8 and 16.

Specifically, with 8 devices, our framework achieves 3.3% to 7.7% non-zero weights and and cuts the communication costs to 14.7% to 24.3% compared to the non-sparse model.

With 16 devices, our framework achieves 4.4% to 8.4% non-zero weights and cuts the communication costs to 16.1% to 26.3% compared to the non-sparse model.

We also observe that both percentage of non-zero weight and communication reduction increase slightly when the number of devices increases.

This is due to the way we set up our experiments in which we randomly and evenly divided each dataset into N non-overlapping parts for N devices.

As the number of devices increases, the data size allocated to each device decreases.

As a consequence, the data distribution at each device becomes less similar, which makes it challenging to learn a global shared model with the same high sparsity without accuracy loss.

In our experiments, we focused on maintaining the accuracy and thus compromised with a slightly increased percentage of non-zero weights and communication reduction.

In previous experiments, the model parameters θ are pretrained with ImageNet, and in this section we will display the performance of our proposed framework with random initialization weights.

TAB3 are the accuracy, percentage of non-zero weights and percentage of communication cost for 4, 8, and 16 devices with random initialization weights, respectively.

Firstly, we see that the accuracy has decreased 2-3% compared to models using the pretrained weights.

Secondly, the percentage of the non-zero weights and total communication cost has decreased accordingly compared to the model with pretrained weights.

Starting from the random initialization has some drawbacks for the Bayesian dropout training since lots of weights will be pruned away in the early stage BID19 , before the model could possibly learn some useful representations of the data.

As a result, the model sparsity will be higher while the accuracy will be lower.

At the same time, the total communication cost will be lower compared to pretrained model due to the higher model sparsity.

In this paper, we have presented a novel federated learning framework based on variational dropout for efficiently learning deep neural networks from distributed data.

Our experiments on diverse deep network architectures and datasets show that our framework achieves high model sparsity with 2.9% to 7.2% non-zero weights and cuts the communication costs from 12.0% to 18.2% with no accuracy loss.

Our experiments also show that our framework scales well when the number of devices increases.

While this work focuses on the synchronous distributed SGD setting, we plan to examine the performance of our framework in the asynchronous distributed SGD setting as our future work.

We also plan to explore the potential of our framework in further compressing the model and reducing the communication cost by combining with the orthogonal gradient quantization approaches.

We are grateful to NVIDIA Corporation for the donation of GPUs to support our research.

In this section, we compare the performance of Algorithm 1 with different settings.

The first setting is shown as our experiment section that model using the pretrained weights from Imagenet.

The second setting is that we train our neural network from random initialization weights which is also shown in experiment section.

The third setting is that we share the gradients of α using the pretrained weights.

The accuracy performance comparison of these settings for 4, 8 and 16 devices are summarized in TAB6 , TAB7 .

To be noted here, in our experimental setting, we don't not share α in order to further reduce communication cost for both the pretrained model and random initialized model.

We have two observations.

Firstly, the accuracy of the setting that shares α using the pretrained weights has the same accuracy of the setting using the pretrained weights without sharing α.

Secondly, if the model is initialized with random weights, the accuracy will drop 2-3% compared to other two settings using pretrained weights.

In this section, we report the percentage of non-zero weights for these three settings described above, and the results are shown as FIG3 .We discover that the percentage of non-zero weights of the setting that shares α using the pretrained weights has the same level with the one uisng pretrained weights without sharing α.

If we train the model from random initialized weights, the percentage of non-zero weights is lower than the two other settings, it means that we will obtain a more sparse model after training.

However, the model accuracy will also be lower compared to other two settings.

In this section, we report the total communication percentage of these three settings described above, and the results are shown as FIG4 .First, we find that if the pretraied model shares α during the training, then the total communication percentage is nearly twice as much as the pretrained model without sharing.

The communication cost is doubled because we share a pair of gradients (g θi , g αi ) compared to sharing g θi only.

Second, if we train our model from random initialized weights, the total communication cost in the minimal among the three settings.

It is reasonable because the percentage of non-zero weights from random initialized weights is the lowest and thus the total communication exchange during training is also the lowest.

@highlight

a joint model and gradient sparsification method for federated learning

@highlight

Applies variational dropout to reduce the communication cost of distributed training of neural networks, and does experiments on mnist, cifar10 and svhn datasets. 

@highlight

The authors propose an algorithm that reduces communication costs in federated learning by sending sparse gradients from device to server and back.

@highlight

Combines distributed optimization algorithm with variational dropout to sparsify the gradients sent to master server from local learners.