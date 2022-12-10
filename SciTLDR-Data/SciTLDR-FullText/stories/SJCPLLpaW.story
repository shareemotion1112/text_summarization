DeePa is a deep learning framework that explores parallelism in all parallelizable dimensions to accelerate the training process of convolutional neural networks.

DeePa optimizes parallelism at the granularity of each individual layer in the network.

We present an elimination-based algorithm that finds an optimal parallelism configuration for every layer.

Our evaluation shows that DeePa achieves up to 6.5× speedup compared to state-of-the-art deep learning frameworks and reduces data transfers by up to 23×.

Training convolutional neural networks (CNNs) is increasingly compute-intensive and timeconsuming.

It takes days or even weeks to train deep CNNs from scratch BID10 BID12 BID9 BID11 .

Existing deep learning frameworks such as TensorFlow, PyTorch, and Caffe2 parallelize the training process onto multiple processors (usually GPUs) using image parallelism 1 dividing the entire image dataset into batches with the same number of images and assigning each batch to a dedicated processor.

The standard parallelization of CNN training only exploits image parallelism.

However, other dimensions can also parallelize the training process.

For example, in CNNs for 2D images, data is commonly organized as 4-dimensional tensors (i.e., image, height, width, channel).

The image dimension includes an index for each image in the input dataset.

The height and width dimensions specify a position in an image.

For a particular position, the channel dimension 2 indexes different neurons for that position.

Exploring these other parallelizable dimensions can potentially reduce the compute time and data transfer cost when training CNNs (see Section 2).

Moreover, different layers in a CNN may prefer different parallelism configurations for achieving optimal performance.

We propose DeePa, a deep learning framework that explores parallelism in all parallelizable dimensions to accelerate the training of CNNs.

To the best of our knowledge, DeePa is the first system that models and exploits the parallelism of neural networks at the granularity of each individual layer.

To generate a parallelism configuration for each layer, DeePa uses an elimination-based algorithm that automatically finds the configuration with the best estimated performance.

The main contributions of this paper are:• We present DeePa, a deep learning framework that explores parallelism in all parallelizable dimensions to accelerate the training of CNNs.• The parallelization strategy is selected at the granularity of each individual layer.• We present an elimination-based algorithm for finding the parallelism configuration with optimal estimated performance for each layer.• Our evaluation shows that, compared to state-of-the-art deep learning frameworks (e.g., TensorFlow and PyTorch), DeePa achieves 6.5×, 1.9×, and 1.5× speedup for AlexNet, TAB0 Conv3x3 in an Inception-v3 module respectively .

The performance improvement comes from reducing overall data transfers, automatically overlapping computation with data movement, and accelerating computation throughput.

This work is motivated by the following observations.

Convolutional layers generally consume the bulk of the training time in CNNs, and parallelizing training in different data dimensions results in significantly different performance.

Figure 1 shows the relative speed of training six different convolutional layers from AlexNet, VGG-16, and Inception-v3.

The properties of the convolutional layers are shown in TAB0 .

For each convolutional layer, we tried parallelizing the computation in each individual parallelizable dimension as well as combinations of different parallelizable dimensions, and we report the performance of the standard parallelization over images along with the worst and best parallelization strategies we discovered.

Figure 1 shows that different parallelism configurations result in very different performance, and image parallelism generally achieves suboptimal performance.

Therefore, exploring parallelism in other dimensions can potentially accelerate the training of convolutional layers.

Different parallelization strategies can also result in significantly different amounts of data movement.

FIG3 shows an example of parallelizing the first fully-connected layer of VGG-16 on two GPUs in different dimensions.

In image parallelism FIG3 , each GPU processes a batch of images and computes the gradient for the entire fully-connected layer.

This requires each GPU to synchronize the gradients for the entire fully-connected layer (shown as the shadow rectangles) after each step.

An alternative approach FIG3 ) parallelizes in the channel dimension by assigning a subset of the output channels to each GPU.

As a result, different GPUs compute the gradients for disjoint subsets of the fully-connected layer, which eliminates transferring the fully-connected layer but introduces additional data transfers for input tensors (shown as the shadow rectangles).

For this particular case, using parallelism in the channel dimension reduces data transfer costs by 12×.

When processing a batch of images, increasing the number of workers does not always improve overall execution time, due to the data transfer overhead to synchronize gradients across different workers.

FIG1 shows the per-step training time for three different layers in Inception-v3 for a batch size of 512 images on up to 16 GPUs.

The training time includes forward processing, backward propagation, and gradient aggregation.

The figure shows that different layers in a neural network may prefer different hardware configurations, and there is no single configuration that is optimal for all layers.

For example, the third layer performs best on 16 GPUs while the last layer performs best on 4 GPUs.

Thus, a parallelism configuration includes both selecting the data dimensions to be parallelized and the number of parallel workers (or, equivalently, the number of subsets into which the data is partitioned).

Width Height C h a n n e l (a) (n=1, c=1, h=1, w=4)Width Height C h a n n e l (b) (n=1, c=1, h=4, w=1)Width Height C h a n n e l (c) (n=1, c=4, h=1, w=1)Width Height C h a n n e l (d) (n=1, c=1, h=2, w=2) Similar to TensorFlow and PyTorch, DeePa uses computation graphs to describe dependencies between operations.

In a computation graph G = (V, E), each node n ∈ V is an operation (e.g., a convolution or matrix-multiply), and each directed edge (u, v) ∈ E is a tensor that is an output of u and an input of v.

One key difference between DeePa and TensorFlow or PyTorch is that each node in the DeePa computation graph also includes a configuration that describes how the corresponding operation is parallelized across different workers.

For each parallelizable dimension (i.e., image, height, width, and channel), the configuration includes an integer that describes the degree of parallelism in that dimension.

For a configuration, the product of the integers over all dimensions is the number of workers needed to process the operation in that configuration.

FIG4 demonstrates some example configurations that explore parallelism in a single dimension as well as combinations of different dimensions.

DeePa assumes equal partitioning in each dimension.

As a result, each worker receives the same size input, which provides well-balanced workload distribution in our experiments.

For each node in the computation graph, its configuration describes how the output tensor is divided onto multiple workers.

Each worker computes a disjoint subset of the output tensor, and thus each worker can process the operation in parallel without data dependencies.

Given a node's configuration, DeePa calculates the input sets for each worker and automatically schedules proper data transfers between operations.

DeePa also provides three additional functions:• For each node v and configuration c, v.compute(c) estimates the time to process the corresponding operation under the parallelism configuration c. This includes both the forward processing and back propagation time and is estimated by running the operation in that configuration multiple times on the device and measuring the average execution time.• For each edge e = (u, v), e.xfer(c u , c v ) estimates the time to transfer the input tensor e to each worker, using the size of the data to be moved and the known communication bandwidth.

Note that e.xfer(c u , c v ) is zero if u and v have the same configuration (i.e., c u = c v ), in which case no data is transferred.

As with compute(), we precompute the xfer() function for each edge in the graph by calculating the overall data transfer size for all possible source and destination configurations.• For each node v and configuration c, v.update(c) estimates the time to update parameters for the corresponding operation.

We use the data transfer time to approximate the update time, since the data transfer time is much longer than the compute time for updating parameters.

Note that different configurations can have significantly different update time, as described in Section 2.2.A global configuration g includes a parallelism configuration for each node in a computation graph: g(v) describes the parallelism configuration for node v. Using the functions defined above, we can model the per-step execution time for a computation graph: DISPLAYFORM0 Cost(g, (V, E)) estimates the per-step execution time if the computation graph (V, E) is parallelized using global configuration g. This execution time includes forwarding processing, backward propagation, and gradient aggregation.

Equation 1 expresses the problem of finding the configuration for each individual node as a global optimization problem.

(a) Node elimination.

We now describe our algorithm for finding a global configuration that minimizes Equation 1.

In DeePa, each node can select any of a fixed (but large) set of parallelism configurations.

Therefore the number of potential global configurations is exponential in the number of nodes in a computation graph, which makes it impractical to enumerate all global configurations for deep CNNs such as VGG-16 and Inception-v3.

However, the CNNs we have seen in practice exhibit strong locality: each node is only connected to a few nodes with similar depth in a computation graph.

Based on this observation, we use the following two elimination strategies to iteratively simplify the computation graph while preserving the globally optimal configuration.

Node elimination.

For each node w with a single in-edge e 1 = (u, w) and a single out-edge e 2 = (w, v), we remove node w and the two edges e 1 and e 2 from the graph and insert a new edge e = (u, v) (shown in FIG7 ).

The xfer() function for node e is e .xfer(c u , c v ) = min DISPLAYFORM0 Note that because we have precomputed the xfer() function for edges in the original graph, we can similarly compute the xfer() function for the transitive edge added by a node elimination; i.e., we use dynamic programming to compute the optimal configuration for node w for every possible choice of configurations for nodes u and v. For CNNs with a linear computation graph (e.g., AlexNet and VGG-16), node elimination is sufficient to reduce the original graph to a graph with only 2 nodes.

Edge elimination.

For two edges with the same source and destination node (i.e., e 1 = (u, v) and e 2 = (u, v)), we can remove e 1 and e 2 from the graph and insert a new edge e = (u, v) (shown in FIG7 ).

The xfer() function for node e is e .xfer(c u , DISPLAYFORM1 Concat (c) After edge elimination.

As with node elimination, we compute the xfer() function for e using the already computed xfer() functions for e 1 and e 2 .

FIG9 shows how DeePa iteratively eliminates nodes and edges for an Inception-v3 module.

The full Inception-v3 computation graph has 120 nodes, which DeePa reduces to a 2-node graph.

DeePa iteratively uses node and edge eliminations to simplify a computation graph until neither elimination can be applied.

DeePa then enumerates all global configurations for the final graph and chooses the one that minimizes the Cost function in Equation 1.After deciding the configuration for each node in the final graph, DeePa then decides the configuration for the eliminated nodes by undoing the node and edge eliminations in reverse order.

When undoing a node elimination for node w, DeePa selects the configuration that minimizes Equation 2 for node w. After undoing all eliminations, DeePa has a configuration for every node in the original graph.

In Appendix A.1, we prove that our algorithm finds an optimal global configuration.

In our experiments, DeePa finds an optimal configuration for parallelizing the largest CNN we have worked with, Inception-v3, on 16 GPUs in about 100ms.

We found that it is non-trivial to parallelize the training of CNNs in the height, width, and channel dimensions in existing frameworks (e.g., TensorFlow, PyTorch, and Caffe2), and none provides an interface for controlling per-operation parallelism.

We implemented DeePa in Legion BID3 , a high-performance parallel runtime for distributed heterogeneous architectures, and use cuDNN BID4 and cuBLAS (cub, 2016) as the underlying libraries for processing neural network operations.

The following Legion features significantly simplify our implementation for DeePa.

First, Legion supports high-dimensional partitioning that allows us to parallelize any operation in any combination of the dimensions.

Second, Legion allows DeePa to control parallelism at the granularity of each operation.

Third, Legion allows fine-grain control over the placement of data in memory.

Fourth, Legion's asynchronous tasking model makes it easy to exploit task as well as image parallelism.

We also include two critical optimizations that help achieve good performance.

Overlapping computation with data transfers.

DeePa manages the gradients of each operation separately and transfers an operation's gradients as long as its back propagation is completed.

We have found that this can effectively hide the data transfer overhead for gradient synchronization.

As a result, the synchronous training performance matches asynchronous training in DeePa, which allows users to use synchronous training with its better algorithmic efficiency.

Distributing parameter servers.

Existing frameworks use parameter servers to store and update variables for a CNN model.

Parameter servers are located in CPU memory in TensorFlow and PyTorch.

Because DeePa manages the parameters for each operation separately, DeePa can opportunistically distribute the parameter server onto the GPU memories whenever possible.

This eliminates data transfers for operations whose gradients and parameter server are located on the same GPU and transforms all GPU to CPU copies into faster GPU to GPU copies.

To the best of our knowledge, DeePa is the first deep learning framework that controls and optimizes the parallelism of neural networks in all dimensions at the granularity of each operation.

Existing frameworks such as TensorFlow BID2 ), Caffe2 (Caf, 2016 , and PyTorch (Pyt, 2017) use image parallelism to distribute the training of CNNs and only explore parallelism in the image dimension.

The standard image parallelism configuration keeps a replica of the entire network on each worker, which results in large data transfers for synchronizing the gradients in each step.

BID7 uses model parallelism that assigns each operation to a dedicated processor for training Inception-v3.

It uses a reinforcement learning algorithm to optimize the placement of each operation on a GPU device.

The learned device placement on 4 GPUs achieves 19% speedup compared to single GPU performance.

However, parallelism in each operation is not explored.

Krizhevsky FORMULA4 introduces "one weird trick" (OWT) that combines image parallelism with model parallelism to accelerate the distributed training of AlexNet, which efficiently reduces the data transfer cost compared to the standard image parallelism configuration.

In Section 7.1.2, we show that DeePa further reduces the overall data transfers for AlexNet by 3× and the per-step training time by 2.3× compared to OWT.

BID5 empirically shows no loss of accuracy for training ResNet-50 on the ImageNet dataset with a large minibatch size of 8192 images 3 .

It uses the standard image parallelism configuration to distribute the training onto 256 GPUs and includes a number of optimizations for reducing communication overhead.

As communication is a bottleneck in distributed deep learning, we believe our techniques for reducing data transfers can substantially benefit training on large numbers of GPUs.

We use AlexNet BID6 , VGG-16 BID9 , and Inceptionv3 BID11 as benchmark CNNs and use the ImageNet dataset BID8 as the input.

For each CNN, we compare the performance of DeePa against TensorFlow, PyTorch, and OWT.

We implement OWT in DeePa by restricting all convolutional and pooling layers to use image parallelism and all fully-connected layers to use model parallelism.

We conduct a detailed case study for training the three CNNs on a 16-GPU machine, with two Intel 10-core E5-2680 Xeon processors, 256 GB main memory, and 16 NVIDIA Tesla K80 GPUs 4 .

We use all 16 GPUs for training each CNN model with a minibatch size of 512 images.

As a result, each GPU processes a batch of 32 images in the image parallelism configuration.

DeePa uses the search algorithm in Section 4 to find the optimal parallelism configurations, which requires 0.7, 1.1, and 4.8 seconds for AlexNet, VGG-16, and Inception-v3, respectively.

Figure 7 shows the synchronous training throughput for a minibatch size of 512 images on 16 GPUs.

When DeePa uses image parallelism for all operations, DeePa achieves competitive performance compared to the best of TensorFlow and PyTorch.

The OWT approach that uses model parallelism for fully-connected layers speeds up the training throughput by 1.4×, 1.2×, and 1.07× compared to image parallelism using DeePa.

The best configurations found by DeePa achieve 6.5×, 1.9×, and 1.5× speedup compared to TensorFlow and PyTorch.

Three main optimizations in DeePa achieve most of the performance benefit over the other frameworks.

First, DeePa significantly reduces data transfers in each step, as shown in Figure 8 .

Compared to image parallelism, the OWT approach reduces data transfers by 1.05-8.4×.

However, the best configuration used by DeePa further reduces data transfers by 1.2-2.7× compared to OWT.

Second, the optimization for overlapping computation with data transfers (described in Section 5) effectively hides data transfer latency and achieves better GPU utilization.

The grey bars in Figure 7 illustrate DeePa's performance when the overlap optimization is disabled, which shows that overlapping computation with data transfers can improve the training throughput by 10%-30%.

Third, DeePa also improves performance by exploring parallelism in the height and width dimensions (see Section 7.1.3).

We describe the best configurations discovered for AlexNet, VGG-16, and Inception-v3 in Sections 7.1.2 to 7.1.4.

The best configurations have several similarities.

First, for the beginning layers with large height/width dimensions and small channel dimensions, DeePa uses image parallelism on all available GPUs, since the data transfers for synchronizing gradients are much smaller than the data transfers for moving tensors between operations.

Second, deeper layers in CNNs tend to have smaller height/width dimensions and larger channel dimensions.

As a result, the cost for moving tensors between different operations decreases, while the cost for synchronizing gradients increases.

DeePa adaptively reduces the number of GPU workers for these layers to reduce the expensive data transfers for synchronizing gradients at the cost of introducing cheaper data transfers for moving tensors.

Third, DeePa uses model parallelism on a small number of GPU workers for fully-connected layers, because synchronizing gradients and moving tensors are both much more expensive than the compute time for fully-connected layers.

DeePa reduces the data transfers for synchronizing gradients and moving tensors at the cost of using fewer GPUs.

Figure 9 : The global configuration for parallelizing AlexNet on 16 GPU workers.

Figure 9 shows the global configuration for AlexNet on 16 GPU workers.

Note that DeePa selects the parallelism configuration that optimizes the performance for each layer.

TAB2 lists the cost for different configurations of the first fully-connected layer.

The standard image parallelism configuration eliminates the cost for transferring the input tensors but introduces additional data transfers for synchronizing gradients.

The OWT approach completely eliminates gradient synchronization at the cost of replicating the input tensors on every GPU worker.

The configuration chosen by DeePa only uses 2 GPU workers for training the first fully-connected layer, which prolongs the compute time but significantly reduces the cost for both transferring input tensors and synchronizing gradients.

As a result, DeePa reduces the total cost by 5× compared to other approaches.

DeePa uses image parallelism for all convolutional and pooling layers, because the additional data transfer cost introduced by transforming configurations outweighs any performance benefits.

Configurations:{n=16, c=1, h=1,w=1} {n=1, c=1, h=2,w=2} {n=1, c=4} {n=1, c=2} DeePa uses similar configurations for parallelizing the fully-connected layers in VGG-16 FIG11 .

In addition, DeePa also uses a different configuration to cooperatively accelerate the last three convolutional layers (the yellow node in FIG11 ).

TAB4 lists the cost for different parallelism configurations for the last three convolutional layers.

The configuration with optimal total cost uses only four GPU workers for the last three convolutional layers to reduce data transfers for synchronizing gradients.

DeePa also exploits parallelism in the height and width dimensions to further reduce the compute time.

Configurations:

{n=16, c=1, h=1, w=1} {n=8, c=1, h=1, w=1} {n=1, c=4} {n=1, c=2}Figure 11: The global configuration for parallelizing Inception-v3 on 16 GPU workers.

Each module is shown as a single node for simplicity.

The Inception-v3 model has multiple Inception modules BID11 .

Each module has several branches of convolutional and pooling layers, which are then concatenated as the output tensor of the module.

Figure 11 shows the global configuration for Inception-v3.

DeePa uses different configurations to parallelize different branches for the InceptionE1 module, as shown in FIG1 .

We found that this configuration reduces data transfers by 30% in InceptionE1 and InceptionE2 and reduces overall data transfers by 20%.

The minibatch size plays an important rule on the performance of CNNs.

FIG3 compares DeePa, PyTorch, and TensorFlow with different minibatch sizes.

All three networks were trained on 16 Tesla K80 GPUs on a single node, as described in Section 7.1.

We were not able to train VGG-16 and Inception-v3 with a minibatch size of 2048 images, because the required metadata size exceeds the aggregate memory capacity of the 16 GPUs.

FIG3 shows that, DeePa achieves constant speedups compared to PyTorch and TensorFlow for various minibatch sizes.

In particular, DeePa achieves 4.6-6.5×, 1.6-1.9×, and 1.2-1.5× speedup for AlexNet, VGG-16, and Inception-v3, respectively.

We evaluate the scalability of different frameworks by comparing their training throughput with different number of GPUs and compute nodes.

The experiments were performed on a GPU cluster with 4 nodes, each of which is equipped with two Intel 10-core E5-2600 Xeon processors, 256G main memory, and four NVIDIA Tesla P100 GPUs.

GPUs on the same node are connected by NVLink, and nodes are connected over 100Gb/s EDR Infiniband.

FIG4 shows the performance comparison among DeePa, PyTorch, and TensorFlow for weakscaling.

DeePa achieves competitive performance compared to PyTorch and TensorFlow for training on a single GPU, in which all three frameworks place all operations on a single GPU.

For training on 4 GPUs on a single node, DeePa achieves 3.1×, 1.6×, and 1.3× speedup for AlexNet, VGG-16, and Inception-v3, respectively.

DeePa achieves even better performance speedups for trainings on multiple nodes, where the data transfer time becomes a larger component of the per-iteration training time.

For training on 4 nodes, DeePa achieves 8.1×, 3.2×, and 1.8× speedup for AlexNet, VGG-16, and Inception-v3, respectively.

We have presented DeePa, a deep learning framework that explores parallelism in all parallelizable dimensions to accelerate the training of CNNs.

DeePa optimizes the parallelism configuration chosen at the granularity of individual layers.

DeePa achieves up to 6.5× for training CNNs and reduces overall data transfers by up to 23× compared to state-of-the-art deep learning frameworks.

Proof.

The Cost function is defined in Equation 1.

Let g be any configuration.

We first compute the difference between Cost(g, (V, E)) and Cost(g, (V , E )).

DISPLAYFORM0 =w.compute(g(w)) + w.update(g(w)) DISPLAYFORM1 Now assume g is an optimal configuration for (V, E).

Then we have w.compute(g(w)) + w.update(g(w)) + e 1 .xfer(g(u), g(w)) + e 2 .xfer(g(w), g(v)) = min cw {w.compute(c w ) + w.update(c w ) + e 1 .xfer(g(u), c w ) + e 2 .xfer(c w , g(v))}Therefore, g is an optimal configuration of (V , E ).

For the other direction, note that if g is an optimal configuration of (V , E ), then it can be extended to an optimal configuration of (V, E) by adding the node w with the same minimal assignment.

For a computation graph G(V, E), applying an edge elimination on e 1 = (u, v) and e 2 = (u, v) results in a modified graph G = (V, E ), where E = E − e 1 − e 2 + e and e = (u, v).

We prove that Cost(g, (V, E)) = Cost(g, (V, E )) for any global configuration g of (V, E).

DISPLAYFORM0 , where (V, E ) is the modified graph of (V, E) after an edge elimination.

Proof.

We compute the difference between Cost(g, (V, E)) and Cost(g, (V, E )).

DISPLAYFORM1 The last equation uses Equation 3.

The overlap optimization in Section 5 is motivated by BID5 , which performs gradient aggregation in parallel with back propagation to scale synchronous training to large number of GPUs.

We extend their design and implementation by also enabling the optimization for asynchronous training in DeePa.

We show profiling results for visualizing the performance bottlenecks in different parallelism approaches.

The experiment was performed on a single node with four Tesla P100 GPUs (as described in Section 7.3).

We enable overlapping computation with data transfers (described in Section 5) in this experiment.

FIG7 shows the profiling results for training VGG-16 on 4 GPUs with different parallelism configurations.

Note that DeePa with image parallelism achieves 10% higher training throughput compared to PyTorch and TensorFlow, as shown in FIG4 .

FIG7 shows that all GPUs are highly utilized during forward and backward passes, as indicated by the tight packing of tasks in the timeline.

However, the image parallelism approach requires moving 4GB of metadata in every iteration, which cannot be fully overlapped with back propagation, therefore the image parallelism approach has a performance gap between iterations (shown as the white space on the GPU timelines).

(a) Image parallelism.

(b) DeePa's parallelism configuration.

FIG7 shows the profiling of the optimal parallelism configuration chosen by DeePa, which uses image parallelism on 4 GPUs for all convolutional layers and pooling layers and uses model parallelism on 2 GPUs for the fully connected layers.

Therefore, the training with the optimal configuration includes data transfers for each fully connected layers, which adds small performance gaps at the end of the forward pass and the beginning of the backward pass (shown as the small white space on the GPU timelines).

However, the optimal configuration reduces the per-iteration data transfers from 4GB to 490MB, which effectively hides data transfer overhead and achieves better GPU utilization.

As a result, the optimal configuration reduces the per-iteration training time from 0.34 seconds to 0.24 seconds.

We compare the performance of DeePa, PyTorch, and TensorFlow on the ImageNet-22K dataset BID8 ) that contains 21,841 different categories (the ImageNet dataset used in Section 7 contains 1,000 catagories).

The last fully-connected layer in AlexNet, VGG-16, and Inception-v3 originally have 1,000 neurons followed by a 1,000-way softmax layer.

To train the three networks on the ImageNet-22K dataset, we change the last fully-connected layer to have 21,841 neurons and use a 21,841-way softmax layer at the end.

The modified networks were trained on 16 Tesla K80 GPUs on a single node with a minibatch size of 512 images.

FIG9 compares the training throughput and per-iteration data transfers among DeePa, PyTorch, and TensorFlow on the ImageNet and ImageNet-22K datasets.

FIG9 shows that, on the ImageNet-22K dataset, the training throughput of PyTorch and TensorFlow is reduced by 20%-45%, while DeePa's throughput falls off by 3%, compared to training on the original ImageNet dataset.

FIG9 compares the per-iteration data transfers between image parallelism and the global configurations used by DeePa.

Using image parallelism increases the data transfers in each iteration by 5-10GB, while DeePa only increases the per-iteration data transfers by 40MB.

As a result, for training on the ImageNet-22K dataset, DeePa reduces the per-iteration data transfers by 3.7-44.5× compared to image parallelism.

@highlight

To the best of our knowledge, DeePa is the first deep learning framework that controls and optimizes the parallelism of CNNs in all parallelizable dimensions at the granularity of each layer.