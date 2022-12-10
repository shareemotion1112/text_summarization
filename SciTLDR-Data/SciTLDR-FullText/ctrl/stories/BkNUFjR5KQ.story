Recent years have witnessed two seemingly opposite developments of deep convolutional neural networks (CNNs).

On one hand, increasing the density of CNNs by adding cross-layer connections achieve higher accuracy.

On the other hand, creating sparsity structures through regularization and pruning methods enjoys lower computational costs.

In this paper, we bridge these two by proposing a new network structure with locally dense yet externally sparse connections.

This new structure uses dense modules, as basic building blocks and then sparsely connects these modules via a novel algorithm during the training process.

Experimental results demonstrate that the locally dense yet externally sparse structure could acquire competitive performance on benchmark tasks (CIFAR10, CIFAR100, and ImageNet) while keeping the network structure slim.

Under the inspiration of bridging these two trends and search more efficient network structures, our paper explores methods which directly introduce sparsity into network structure thus avoid pruningafter-training strategy.

In neural science, papers (e.g. BID4 BID35 BID10 ) concentrating on the brain structure reveal that neuron connections in brain perform a locally dense but externally sparse property as paper BID4 shown, that the closer two regions are, the denser the connections between them will be.

Visual cortex papers BID2 show that while sensory information arrives at the cortex, it is fed up through hierarchy regions from primary area V1 up to higher areas such as V2, V4 and IT.

Inside of each cor-tex layer, tightly packed pyramidal cells consist of basic locally dense structures in the brain.

While our brain has been trained over time, internal densely connected modules will form a few long distance and cross-level connections to transfer information to higher hierarchy.

Modular structures have shown vital importance in our brain behaviors such as specializing in information processing BID9 , performing focal functions BID1 , and supporting complex neural dynamics.

In this case, instead of creating local density by pruning redundancy on the trained model, we perform local density by prefixing untrained dense modules as tightly packed neuron cell in the human brain and let it evolving both the weights of itself and the sparse connection between them via training.

Since DenseNet has reached theoretical densest connection status, we use a similar dense block structure with growth rate k, but only with very narrow channels in each block.

The growth rate k BID16 ) is a hyper parameter in Densely Connected structures, which denotes growth rate of the input feature map scale when network goes deeper.

Previous methods constructing neural modules with structural sparsity (e.g. BID39 BID34 ) are mostly empirically constructing the sparse connection between modules.

To give more convincing guidance of forming sparse connections, we design a genetic training strategy to search an optimized connection matrix.

This algorithm treats the connection matrix as the gene, and only reserves mutated individual with the best performance among others.

Actually, this strategy consistently changes the input feature groups during training process, and by always counting new feature distribution in, this strategy could take similar effect as drop-out methods, thus make the model robust.

Moreover, besides merely creating parallel connections between modules, our algorithm could create long-distance connections between input module and output module by a transit layer.

The experiment results demonstrate that evolving locally dense but externally sparse connections could maintain competitive performance on benchmark image datasets while using compared slim network structures.

By comparison experiments, we reveal contribution proportion on the final performance of each specific connection, and by that give the principle of design sparse connections between dense modules.

The main contribution of this paper is as follows:• We enhance the hierarchical structure by utilizing the property of locally dense but externally sparse connections.• Instead of empirically constructing module connections, we design an evolving training algorithm to search optimized connection matrix.•

We let each module choose output flow globally rather than simply creating parallel streams between modules so that the feature could flow to final layer through various depth.• We give a detailed analysis of how different sparse connections and different module properties will contribute to the final performance.

Moreover, We reveal contribution proportion on the final performance of each connection and each module by several contrast experiments, and by that give principle of design sparse connections between dense modules.

Network architectures are becomming denser.

The exploration of network architectures has been an important foundation for all Deep Learning tasks.

At the early period of deep learning, increasing the depth of a network might promise a good result as the network structure varied from LeNet to VGG (e.g. BID20 BID18 BID29 ).

Since people realize that the increasing depth of the network amplifies problem such as over-fitting and gradient-vanishing BID3 , parallel structures (e.g. BID40 ) BID34 ) and densely connected layers BID16 have been introduced to increase network capacity.

As DenseNet reaches the densest connection method inside each dense block, we refer to the dense block in this paper while constructing internal densely connected modules.

Although our paper does not merely concentrate on highest benchmark accuracy, but also hierarchy structure and global sparsity, we still acquire competitive result on benchmark datasets using slim network structure.

Deep neural network compression.

Besides increasing model capacity, deep neural network compression is another activate domain concentrating on acquire slim model by eliminating network redundancy.

These methods could be roughly summarized as three basic aspects as fol-lows: 1.

Numerical approximation of kernels, which includes binarization BID7 BID25 , quantization BID41 , weight sharing or coding method BID11 and mainly use numerical method to approximate kernel with smaller scale; 2.

Sparse regularization on kernels, which mainly prune connections based on regularization on kernels, such as weights/channel pruning BID15 BID23 and structure sparsity learning BID22 ; 3.

Decomposition of kernels, which mainly use smaller groups of low-rank kernel instead of a larger whole kernel, such as BID8 BID17 BID6 and BID39 .

These papers mostly put an emphasis on model sparsity rather than capacity.

Our paper combines the global sparsity and locally dense feature together to maintain high capacity while making the network structure slim and separable.

Evloving algorithm on nerual network.

Many early works have developed methods that evolve both topologies and weights BID0 BID5 BID24 BID32 .

Most of them implement in the area of reinforcement learning.

Evolutionary methods for searching better network structures have risen again recently on reinforcement domain BID36 BID28 .

Also, it still shows great potential for image classification BID26 .

Google has proposed a state-of-the-art deep neural network structure NasNet BID42 , and reaches the best performance so far by searching the best architecture on large scale.

However, the huge scale of these networks with the searching parameters method still remains a problem.

Our paper emphasizes on structural density & sparsity.

The evolving algorithm is only used to search sparse connections during the training process.

Convolution operation could be understood as the projection between different channels of feature maps.

Channel-wise mapping between input and output feature map could be expressed as connections.

In convolution operation, the kernel could be written as: j * i * m * n, where j, i denotes output channels and input channels, M * N denotes the size of filter W .

In order to separately represents the connection between each channel pair (i, j) and illustrate concepts of 'local', we use Frobenius norm representation in Eq.(1) of each filter to represent the importance of channels as in FIG0 : Separately represents connections between output channels and input channels.

Brightened part means the F-norm between these specific channels is significantly large.

Dark area shows that the model has significant channel wise redundancy.

DISPLAYFORM0 Under this representation, we could calculate feature map of typical convolution kernel and show in FIG0 .

As a convolution kernel could be considered as mapping input feature from i channels to j channels, dark parts suggest that norm of a size m * n filter is compared small, which is also called redundancy in network compression domain.

Besides, inspired by the brain structure shown in neural science papers, making kernels locally dense connected could significantly save parameters.

In this case, the kernel could be decomposed as it shows in FIG1 .

Obviously, this decomposition method sacrifice a large number of connections in each layer.

In order to maintain high model capacity after decomposition, we create sparse connections between these modules as below. illustrates two small kernels with shape 3 * 2 * m * n after ideal decomposition.

Especially, grey color denotes the connections between channels has been cut off.

Note that under this example, decomposition saved 18 * m * n parameters.

To create locally density, different from the traditional method that eliminates redundancy by pruning channels on a pretrained kernel, we would like the modularity forming along training process perusing.

In that case, there exist two major ways to form local density, the first is placing L2 regularization in loss function to regulate weights distributed along diagonal in the adjacent matrix, the second and what we have chosen is to prefix some densely connected modules and explore the sparse connections between them.

In order to acquire locally density both in depth and width wise, we stack several 'narrow' convolution kernels into a dense module as shown in Fig. 3 .

This structure also uses a bottleneck layer with growth rate k BID16 which consists of sequential layers {BN -1*1conv -BN -3*3conv} (with zero padding 1).

The connection strategy between bottleneck layers is also densely connected, and the connectivity could be presented as DISPLAYFORM0 , where H l represents nonlinear operation on feature map in layer l and (x 0 , x 1 , x 2 , ...x n ) represents the concatenation of all previous layers.

It should be noticed that inside each dense module, the feature map size is constant, but channels will grow rapidly as layer depth and growth rate increase.

To control the model scale, we use a transit layer introduced in DenseNet BID16 to reduce channels of output feature map to half of original number.

In this paper, we take a densely connected module as a basic element.

Figure 3 : A structure example for a prefix dense module as shown above, where yellow layer represents several densely connect bottleneck layers (it means all output has a direct connection to the output layer).

The detailed structure used in a bottleneck layer shown left.

After the final layer, the green layer represents a transition layer to control the feature map size.

Dense blocks depth in our experiment usually varied from 6-20 layers.

While dealing with sparse connections between modules, a significant problem for us is that feature map size will decrease after it flows through a dense module.

In order to create sparse connections, here, we firstly figure out connection methods within the same distance, secondly we figure out methods of dealing long distance connections; finally, we could represent sparse connections in the matrix.

To figure out the influence of the connection method, we implement a contrast experiment as it shows in Experiment of connection method part 4.1.

Experiment results demonstrate that although concatenation method caused a larger model, its accuracy on CIFAR10 does not show an absolute advantage over the addition method.

Moreover, since we attempt to apply an evolution method to find optimized external connections, the concatenating need more operations when changing the input features.

In that case, we select the addition method for sparse connections.

Long Distance Connections: As we mentioned above, the feature map size will change since it flows through dense modules.

In order to make it possible for making long-distance connections between different depth, we use a transfer layer with {1*1conv -average pooling} structure to fit the feature map into the dense module requirement.

Notice that {1*1conv} layer reform the feature map channels while average pooling changes the feature map size to fit requirement.

It should be noticed that, in this way, for each module, they could have various network depth.

Represent sparse connection: For better analysis of sparse connections, we use the adjacent matrix to represent connections as FIG2 .

If there exists a connection, we set element value correspond to that index in connection matrix to be 1, otherwise 0.

Here we could simply define the density as DISPLAYFORM0 sum(Cmax) , where C i denotes the current connection matrix, C max denotes the connection matrix under the fully connected condition, sum() means the summation value of all elements in the matrix.

In this paper, we only used directed graphs and down sampling connections, so the lower left of the matrix should always be zero.

Fig. (b) , red rectangle area denotes connections with distance 1, green rectangle denotes connections with distance 2, blue area denotes connections with distance 3 3.4 EVOLUTION ALGORITHM TO SEARCH IMPORTANT CONNECTIONS One crucial problem in creating sparse topology connections is that there has not been a convincing theory on what could be called an efficient connection.

In that case, we decide to make the neural network searching optimized sparse connection by itself.

In this paper we use a genetic algorithm BID30 ) to search the proper connections.

We encoding connection matrix as the gene for genetic algorithm.

In each iteration, the genetic algorithm generate several new 'individuals' with genes from mutation of the best 'individual' in last iteration.

The set of generated 'individuals' is called 'population'.

Genetic algorithm evolves by select best performance individual Encoding: Inspired by the genetic algorithm, evolving methods need to have a good encoding to describe object features.

Here we take the adjacent matrix to represent connection topology during training.

In implementation details, we use a connection list to attach each module to avoid wasting storage.

Initial state: As we do not use pre-trained modules, we randomly initialize the weight value of modules at the first iteration of the training process.

Since a deep neural network needs a long time to train, restricted to our computation capacity, we set the population between 2 to 3 individuals.

For the connection matrix of the initial state, we set it only have parallel direct connections as shown in Initial state denotes the initial connections P .

As we set before first iteration P best = P init , based on P best we generate 2 individual below.

All together these 3 individual form the population to be trained simultaneously in iteration 1.

Then, we choose the individual with the best performance, and based on that we form population for iteration 2.

Follow this principle we maintain network evolving.

Evolution Strategy: We define the connection matrix of the initial individual state as P init ; best performance individual of the previous iteration as P best , and others as P i at beginning of each iteration, the evolution of connections could be defined as: DISPLAYFORM1 where we choose P best as input of mutation function G, then generate several mutation individuals P 1 , P 2 ... based on P best .

Then we treat the set of P best , P 1 , P 2 ... as population in this iteration.

It means the best performance individual will remain to next iteration, and based on it we mutate new individuals.

What exactly mutation function G does is that based on the input connection matrix, randomly pick two possible connections and change the connectivity of it.

It means that, if we randomly pick an unconnected connection, we set it connected, and for already connected connection, we set it disconnected.

Different from methods used in the NEAT algorithm BID32 ) which forces connections denser over time, our strategy has a larger probability to become denser if density is less than 0.5, and it has a larger probability to become sparser if density is large than 0.5.After the population of each iteration has been generated, we need to separately train each individual for a complete epoch and make it a fair comparison between each individual.

In implementing detailwise, before start training, we set a checkpoint for all status and parameters and make sure that all individuals under comparison start from checkpoints.

After the training process, only the individual with the best performance will remain, and based on that, we can generate the population of the next iteration.

The whole process shows in Algorithm 1 and Fig P init ← Initial Connection Matrix 3: DISPLAYFORM2 for n iterations do 5: DISPLAYFORM3 checkpoint ← Model at P best 7:for k iterations do 8: DISPLAYFORM4 train P k 10:if P k .accuracy > P best .accuracy then 11: DISPLAYFORM5 end if

end for

end for

Return P best 16: end procedure

We firstly do a contrast experiment on Concatenation vs. Addition method to figure out which connection method we will use.

As the test object is the connection method, we prefix a group of sparse connections and control all other training strategy and environment exactly the same, then separately train the network on the CIFAR10 dataset.

We run our experiments on NVIDIA K80 GPU device.

The test result is shown as FIG5 .

Fig. (b) denotes an example of a random chosen P 1 and Fig (a) denotes the train&test curve correspond to it.

Fig. (c) shows the comparison result on three random chosen situation.

We could observe that the addition method only have a negligible difference with the concatenation method.

Although the curve of addition method seems to have more fluctuations, it only has a negligible difference (we use the difference in highest accuracy on the test set to represent difference) with the concatenation method.

As we mentioned before, the addition method is faster and more convenient for changeable feature map size.

We choose addition method in later experiments.

It should be also noticed that, the accuracy step jumps in the figures are caused by learning rate change for all experiments in this section.

As we use the same learning rate change strategy mentioned in section 4.2 for all experiments, all step jumps in our experiments happen at the same position.

For prefixed dense modules, we set it with 4 different depth, where each depth has 3 modules.

The total of 12 modules has the growth rate of 32, the modules in depth 1,2,3,4 respectively have 6,12,24,16 layers.

Then we run several sparse connection evolving algorithms also training on CI-FAR10 dataset on NVIDIA AWS P3.x2large instance.

We set the total iteration number to be 160, with weight decay of 5e-4.

We use SDG with momentum 0.9 for gradient decsent.

The learning rate strategy is the same as most of the papers that during epoch 0-90 the learning rate is 0.1; during 90-140 learning rate is 0.01; and during 140-160 learning rate is 0.001.

It should be noticed that changing the learning rate will lead to accuracy 'step jumps' such as FIG5 shows.

It's a common phenomenon.

Restricted to our computation power, we set the number of individuals generated in each iteration to be 2.

The training curve of P best shown as According to the repeatable experiments, we could see that although randomness of forming the first generation of populations may lead to variation and fluctuation in the early period of the testing performance curve, the training curve will finally converge to the same trend.

This shows the feasibility of our algorithm.

Based on these experiments we found that the optimized connection matrix is not unique to achieve good performance.

However, we could still find some similarity between those connection matrices in the experiment (Fig. 8 ) which could reach high accuracy.

It denotes that the modules with shallow depth are more likely to form a long-distance connection, which means the distance between the input feature map and output are shorten under that situation.

This perfectly fits a current trend observed by various other papers BID16 BID27 BID13 BID6 BID33 BID6 that skip/direct connections are important.above.

The result shown in FIG8 .

Clearly, the networks with smaller growth rate have higher test accuracy and more flatten curve shape compared to those with larger growth rates at the earlier period of training.

It means that the modules with smaller scale are easier to train while evolving sparse connections.

We can also see that although modules with smaller growth rates converge really fast and could get a good result after 90 epoch, the final test accuracy is not as high as those modules with larger growth rate.

This phenomenon, in fact, proves an empirical conclusion that neural network redundancy is a necessary part of achieving high performance on test accuracy.

However, experiment results also demonstrate that the network redundancy is not the 'larger the better'.

As it shows in FIG8 , after the growth rate is larger than 32, the test accuracy will not increase anymore.

It is also rational because if the capacity of each module is too large, an unstable input feature may make the network harder to train.

In another side, the increasing growth rate, which leads to the increasing of model scale, increases the risk of over-fitting.

Although our paper emphasizes on how sparse connections will change the model performance, we still give performance scores on the benchmark dataset as shown in Tab1, Tab2.

Since the aim of this paper is to obtain slim structures while keeping the model's capacity and achieve separable network structures, the test accuracy on both ImageNet and CIFAR is not that high compared to the state-of-the-art model.

However, we still get a competitive result on both datasets.

After the evolving training algorithm gives optimal sparse connections, we wonder which sets of connections play a more important role in the whole network flow.

We separately cut off one sparse connection each time and test the remaining accuracy on CIFAR10 dataset.

Then we come up with a matrix that suggests how much accuracy decreasing results from losing each connection as shown in FIG0 In experiment results, the red rectangle area denotes the direct connections; the green and blue rectangle area denote the long-distance connections.

According to the accuracy loss distribution, local and direct connections are of vital importance for a neural network.

It is rational because the deep learning method needs a compared invariant forward and backward feature flow path for propagation.

We could also see the accuracy loss is larger along the diagonal to the high left of the matrix.

It means that connections with shallow depth perform a more important role in conduct features/patterns than deeper connections.

It is also rational because the shallower connections simultaneously mean the features that flow through such connections have not been extract to some level of abstraction.

In FIG0 , each column denotes how many connections are attached to this module.

Contrast experiment suggests that: 1.

The connections between shallow modules are more important than deeper and long-distance connections.

2.

The local connections contribute a base test accuracy, and the long-distance connections will contribute more on increase accuracy by small steps based on the baseline accuracy.

3.

The more connections a module has as input, the more robust the module will be when cutting off some of the connections.

DISPLAYFORM0

In this paper, we firstly create locally dense and externally sparse structures by prefixing some dense modules and add sparse connections between them.

Experiment results demonstrate that evolving sparse connections could reach competitive results on benchmark datasets.

In order to give properties of these biologically plausible structures, we apply several sets of contrast experiments as shown in Experiment.

By equally changing the input feature groups of each module during the whole training process, this strategy could alleviate the risk of the weights being trapped in local optimal point.

Same to most of the related works, redundancy of each dense module is not 'the larger the better', where the test accuracy will first increase within the growth rate increases, but finally drop while the growth is above some threshold.

The combination of being dense and being sparse is an interesting area, and the internal dense and externally sparse structure also coincide with the modularity in human brain.

We prove the feasibility of these structures and give a simple algorithm to search best connections.

We also noticed that the connection matrix is not unique for reaching good performance.

We will concentrate on revealing the relationship between these similar connection matrices and the representing features behind it.

In this case, we may acquire state of the art performance on other datasets and tasks in our future work.

Moreover, as these structures have various direct paths between input and output, separating a network into several small networks without any accuracy loss is also a promising topic.

<|TLDR|>

@highlight

In this paper, we explore an internal dense yet external sparse network structure of deep neural networks and analyze its key properties.