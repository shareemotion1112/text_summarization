This paper addresses the scalability challenge of architecture search by formulating the task in a differentiable manner.

Unlike conventional approaches of applying evolution or reinforcement learning over a discrete and non-differentiable search space, our method is based on the continuous relaxation of the architecture representation, allowing efficient search of the architecture using gradient descent.

Extensive experiments on CIFAR-10, ImageNet, Penn Treebank and WikiText-2 show that our algorithm excels in discovering high-performance convolutional architectures for image classification and recurrent architectures for language modeling, while being orders of magnitude faster than state-of-the-art non-differentiable techniques.

Discovering state-of-the-art neural network architectures requires substantial effort of human experts.

Recently, there has been a growing interest in developing algorithmic solutions to automate the manual process of architecture design.

The automatically searched architectures have achieved highly competitive performance in tasks such as image classification BID35 BID36 BID13 a; BID26 and object detection BID36 .The best existing architecture search algorithms are computationally demanding despite their remarkable performance.

For example, obtaining a state-of-the-art architecture for CIFAR-10 and ImageNet required 2000 GPU days of reinforcement learning (RL) BID36 or 3150 GPU days of evolution BID26 .

Several approaches for speeding up have been proposed, such as imposing a particular structure of the search space BID13 a) , weights or performance prediction for each individual architecture (Brock et al., 2018; Baker et al., 2018) and weight sharing/inheritance across multiple architectures BID0 BID24 Cai et al., 2018; Bender et al., 2018) , but the fundamental challenge of scalability remains.

An inherent cause of inefficiency for the dominant approaches, e.g. based on RL, evolution, MCTS BID20 , SMBO BID12 or Bayesian optimization BID9 , is the fact that architecture search is treated as a black-box optimization problem over a discrete domain, which leads to a large number of architecture evaluations required.

In this work, we approach the problem from a different angle, and propose a method for efficient architecture search called DARTS (Differentiable ARchiTecture Search).

Instead of searching over a discrete set of candidate architectures, we relax the search space to be continuous, so that the architecture can be optimized with respect to its validation set performance by gradient descent.

The data efficiency of gradient-based optimization, as opposed to inefficient black-box search, allows DARTS to achieve competitive performance with the state of the art using orders of magnitude less computation resources.

It also outperforms another recent efficient architecture search method, ENAS BID24 .

Notably, DARTS is simpler than many existing approaches as it does not involve controllers BID35 Baker et al., 2017; BID36 BID24 BID33 , hypernetworks (Brock et al., 2018) or performance predictors BID12 ), yet it is generic enough handle both convolutional and recurrent architectures.

The idea of searching architectures within a continuous domain is not new BID27 Ahmed & Torresani, 2017; BID30 BID28 , but there are several major distinctions.

While prior works seek to fine-tune a specific aspect of an architecture, such as filter shapes or branching patterns in a convolutional network, DARTS is able to learn high-performance architecture building blocks with complex graph topologies within a rich search space.

Moreover, DARTS is not restricted to any specific architecture family, and is applicable to both convolutional and recurrent networks.

In our experiments (Sect.

3) we show that DARTS is able to design a convolutional cell that achieves 2.76 ± 0.09% test error on CIFAR-10 for image classification using 3.3M parameters, which is competitive with the state-of-the-art result by regularized evolution BID26 obtained using three orders of magnitude more computation resources.

The same convolutional cell also achieves 26.7% top-1 error when transferred to ImageNet (mobile setting), which is comparable to the best RL method BID36 .

On the language modeling task, DARTS efficiently discovers a recurrent cell that achieves 55.7 test perplexity on Penn Treebank (PTB), outperforming both extensively tuned LSTM BID17 and all the existing automatically searched cells based on NAS BID35 and ENAS BID24 .Our contributions can be summarized as follows:• We introduce a novel algorithm for differentiable network architecture search based on bilevel optimization, which is applicable to both convolutional and recurrent architectures.• Through extensive experiments on image classification and language modeling tasks we show that gradient-based architecture search achieves highly competitive results on CIFAR-10 and outperforms the state of the art on PTB.

This is a very interesting result, considering that so far the best architecture search methods used non-differentiable search techniques, e.g. based on RL BID36 or evolution BID26 BID13 ).•

We achieve remarkable efficiency improvement (reducing the cost of architecture discovery to a few GPU days), which we attribute to the use of gradient-based optimization as opposed to non-differentiable search techniques.• We show that the architectures learned by DARTS on CIFAR-10 and PTB are transferable to ImageNet and WikiText-2, respectively.

The implementation of DARTS is available at https://github.com/quark0/darts 2 DIFFERENTIABLE ARCHITECTURE SEARCH We describe our search space in general form in Sect.

2.1, where the computation procedure for an architecture (or a cell in it) is represented as a directed acyclic graph.

We then introduce a simple continuous relaxation scheme for our search space which leads to a differentiable learning objective for the joint optimization of the architecture and its weights (Sect.

2.2).

Finally, we propose an approximation technique to make the algorithm computationally feasible and efficient (Sect.

2.3).

Following BID36 ; BID26 ; BID12 BID25 , we search for a computation cell as the building block of the final architecture.

The learned cell could either be stacked to form a convolutional network or recursively connected to form a recurrent network.

A cell is a directed acyclic graph consisting of an ordered sequence of N nodes.

Each node x (i) is a latent representation (e.g. a feature map in convolutional networks) and each directed edge (i, j) is associated with some operation o (i,j) that transforms x (i) .

We assume the cell to have two input nodes and a single output node.

For convolutional cells, the input nodes are defined as the cell outputs in the previous two layers BID36 .

For recurrent cells, these are defined as the input at the current step and the state carried from the previous step.

The output of the cell is obtained by applying a reduction operation (e.g. concatenation) to all the intermediate nodes.

Each intermediate node is computed based on all of its predecessors: A special zero operation is also included to indicate a lack of connection between two nodes.

The task of learning the cell therefore reduces to learning the operations on its edges.

DISPLAYFORM0

Let O be a set of candidate operations (e.g., convolution, max pooling, zero) where each operation represents some function o(·) to be applied to x (i) .

To make the search space continuous, we relax the categorical choice of a particular operation to a softmax over all possible operations: DISPLAYFORM0 where the operation mixing weights for a pair of nodes (i, j) are parameterized by a vector α (i,j) of dimension |O|.

The task of architecture search then reduces to learning a set of continuous variables α = α (i,j) , as illustrated in FIG0 .

At the end of search, a discrete architecture can be obtained by replacing each mixed operationō (i,j) with the most likely operation, i.e., o (i,j) = argmax o∈O α (i,j) o .

In the following, we refer to α as the (encoding of the) architecture.

After relaxation, our goal is to jointly learn the architecture α and the weights w within all the mixed operations (e.g. weights of the convolution filters).

Analogous to architecture search using RL BID35 BID36 BID24 or evolution BID13 BID26 where the validation set performance is treated as the reward or fitness, DARTS aims to optimize the validation loss, but using gradient descent.

Denote by L train and L val the training and the validation loss, respectively.

Both losses are determined not only by the architecture α, but also the weights w in the network.

The goal for architecture search is to find α * that minimizes the validation loss L val (w * , α * ), where the weights w * associated with the architecture are obtained by minimizing the training loss w * = argmin w L train (w, α * ).This implies a bilevel optimization problem (Anandalingam & Friesz, 1992; Colson et al., 2007) with α as the upper-level variable and w as the lower-level variable: DISPLAYFORM1 DISPLAYFORM2 The nested formulation also arises in gradient-based hyperparameter optimization BID16 BID22 BID2 , which is related in a sense that the architecture α could be viewed as a special type of hyperparameter, although its dimension is substantially higher than scalar-valued hyperparameters such as the learning rate, and it is harder to optimize.

Create a mixed operationō DISPLAYFORM0 (ξ = 0 if using first-order approximation) 2.

Update weights w by descending ∇ w L train (w, α) Derive the final architecture based on the learned α.

Evaluating the architecture gradient exactly can be prohibitive due to the expensive inner optimization.

We therefore propose a simple approximation scheme as follows: DISPLAYFORM0 where w denotes the current weights maintained by the algorithm, and ξ is the learning rate for a step of inner optimization.

The idea is to approximate w * (α) by adapting w using only a single training step, without solving the inner optimization (equation 4) completely by training until convergence.

Related techniques have been used in meta-learning for model transfer BID1 , gradientbased hyperparameter tuning BID15 and unrolled generative adversarial networks BID19 .

Note equation 6 will reduce to ∇ α L val (w, α) if w is already a local optimum for the inner optimization and thus ∇ w L train (w, α) = 0.The iterative procedure is outlined in Alg.

1.

While we are not currently aware of the convergence guarantees for our optimization algorithm, in practice it is able to reach a fixed point with a suitable choice of ξ 1 .

We also note that when momentum is enabled for weight optimisation, the one-step unrolled learning objective in equation 6 is modified accordingly and all of our analysis still applies.

Applying chain rule to the approximate architecture gradient (equation 6) yields DISPLAYFORM1 where w = w − ξ∇ w L train (w, α) denotes the weights for a one-step forward model.

The expression above contains an expensive matrix-vector product in its second term.

Fortunately, the complexity can be substantially reduced using the finite difference approximation.

Let be a small scalar 2 and DISPLAYFORM2 Evaluating the finite difference requires only two forward passes for the weights and two backward passes for α, and the complexity is reduced from O(|α||w|) to O(|α| + |w|).First-order Approximation When ξ = 0, the second-order derivative in equation 7 will disappear.

In this case, the architecture gradient is given by ∇ α L val (w, α), corresponding to the simple heuristic of optimizing the validation loss by assuming the current w is the same as w * (α).

This leads to some speed-up but empirically worse performance, according to our experimental results in TAB0 .

In the following, we refer to the case of ξ = 0 as the first-order approximation, and refer to the gradient formulation with ξ > 0 as the second-order approximation.

To form each node in the discrete architecture, we retain the top-k strongest operations (from distinct nodes) among all non-zero candidate operations collected from all the previous nodes.

The strength of an operation is defined as DISPLAYFORM0 .

To make our derived architecture comparable with 1 A simple working strategy is to set ξ equal to the learning rate for w's optimizer.

2 We found = 0.01/ ∇ w L val (w , α) 2 to be sufficiently accurate in all of our experiments.

DISPLAYFORM1 .

The analytical solution for the corresponding bilevel optimization problem is (α * , w * ) = (1, 1), which is highlighted in the red circle.

The dashed red line indicates the feasible set where constraint equation 4 is satisfied exactly (namely, weights in w are optimal for the given architecture α).

The example shows that a suitable choice of ξ helps to converge to a better local optimum.

those in the existing works, we use k = 2 for convolutional cells BID36 BID12 BID26 and k = 1 for recurrent cells BID24 .The zero operations are excluded in the above for two reasons.

First, we need exactly k non-zero incoming edges per node for fair comparison with the existing models.

Second, the strength of the zero operations is underdetermined, as increasing the logits of zero operations only affects the scale of the resulting node representations, and does not affect the final classification outcome due to the presence of batch normalization BID8 .

Our experiments on CIFAR-10 and PTB consist of two stages, architecture search (Sect.

3.1) and architecture evaluation (Sect.

3.2).

In the first stage, we search for the cell architectures using DARTS, and determine the best cells based on their validation performance.

In the second stage, we use these cells to construct larger architectures, which we train from scratch and report their performance on the test set.

We also investigate the transferability of the best cells learned on CIFAR-10 and PTB by evaluating them on ImageNet and WikiText-2 (WT2) respectively.

We include the following operations in O: 3 × 3 and 5 × 5 separable convolutions, 3 × 3 and 5 × 5 dilated separable convolutions, 3 × 3 max pooling, 3 × 3 average pooling, identity, and zero.

All operations are of stride one (if applicable) and the convolved feature maps are padded to preserve their spatial resolution.

We use the ReLU-Conv-BN order for convolutional operations, and each separable convolution is always applied twice BID36 BID26 BID12 .Our convolutional cell consists of N = 7 nodes, among which the output node is defined as the depthwise concatenation of all the intermediate nodes (input nodes excluded).

The rest of the setup follows BID36 ; BID12 BID26 , where a network is then formed by stacking multiple cells together.

The first and second nodes of cell k are set equal to the outputs of cell k − 2 and cell k − 1, respectively, and 1 × 1 convolutions are inserted as necessary.

Cells located at the 1/3 and 2/3 of the total depth of the network are reduction cells, in which all the operations adjacent to the input nodes are of stride two.

The architecture encoding therefore is (α normal , α reduce ), where α normal is shared by all the normal cells and α reduce is shared by all the reduction cells.

Detailed experimental setup for this section can be found in Sect.

A.1.1.

Our set of available operations includes linear transformations followed by one of tanh, relu, sigmoid activations, as well as the identity mapping and the zero operation.

The choice of these candidate operations follows BID35 ; BID24 .

Each architecture snapshot is re-trained from scratch using the training set (for 100 epochs on CIFAR-10 and for 300 epochs on PTB) and then evaluated on the validation set.

For each task, we repeat the experiments for 4 times with different random seeds, and report the median and the best (per run) validation performance of the architectures over time.

As references, we also report the results (under the same evaluation setup; with comparable number of parameters) of the best existing cells discovered using RL or evolution, including NASNet-A (Zoph et al., 2018) (2000 GPU days), AmoebaNet-A (3150 GPU days) BID26 and ENAS (0.5 GPU day) BID24 .function, as done in the ENAS cell BID24 .

The rest of the cell is learned.

Other settings are similar to ENAS, where each operation is enhanced with a highway bypass BID34 and the cell output is defined as the average of all the intermediate nodes.

As in ENAS, we enable batch normalization in each node to prevent gradient explosion during architecture search, and disable it during architecture evaluation.

Our recurrent network consists of only a single cell, i.e. we do not assume any repetitive patterns within the recurrent architecture.

Detailed experimental setup for this section can be found in Sect.

A.1.2.

To determine the architecture for final evaluation, we run DARTS four times with different random seeds and pick the best cell based on its validation performance obtained by training from scratch for a short period (100 epochs on CIFAR-10 and 300 epochs on PTB).

This is particularly important for recurrent cells, as the optimization outcomes can be initialization-sensitive FIG2 .To evaluate the selected architecture, we randomly initialize its weights (weights learned during the search process are discarded), train it from scratch, and report its performance on the test set.

We note the test set is never used for architecture search or architecture selection.

Detailed experimental setup for architecture evaluation on CIFAR-10 and PTB can be found in Sect.

A.2.1 and Sect.

A.2.2, respectively.

Besides CIFAR-10 and PTB, we further investigated the transferability of our best convolutional cell (searched on CIFAR-10) and recurrent cell (searched on PTB) by evaluating them on ImageNet (mobile setting) and WikiText-2, respectively.

More details of the transfer learning experiments can be found in Sect.

A.2.3 and Sect.

A.2.4.

BID33 3.54 39.8 96 8 RL AmoebaNet-A BID26 3.34 ± 0.06 3.2 3150 19 evolution AmoebaNet-A + cutout BID26 ) † 3.12 3.1 3150 19 evolution AmoebaNet-B + cutout BID26 2.55 ± 0.05 2.8 3150 19 evolution Hierarchical evolution BID13 3.75 ± 0.12 15.7 300 6 evolution PNAS BID12 3.41 ± 0.09 3.2 225 8 SMBO ENAS + cutout BID24 2.89 4.6 0.5 6 RL ENAS + cutout BID24 3.00 ± 0.14 3.3 1.5 7 gradient-based DARTS (second order) + cutout 2.76 ± 0.09 3.3 4 7 gradient-based * Obtained by repeating ENAS for 8 times using the code publicly released by the authors.

The cell for final evaluation is chosen according to the same selection protocol as for DARTS.

† Obtained by training the corresponding architectures using our setup.

‡ Best architecture among 24 samples according to the validation error after 100 training epochs.

Table 2 : Comparison with state-of-the-art language models on PTB (lower perplexity is better).

Note the search cost for DARTS does not include the selection cost (1 GPU day) or the final evaluation cost by training the selected architecture from scratch (3 GPU days).

BID18 60.7 58.8 24 --manual LSTM + skip connections BID17 60.9 58.3 24 --manual LSTM + 15 softmax experts BID31 BID23 publicly released by the authors.

† Obtained by training the corresponding architecture using our setup.

‡ Best architecture among 8 samples according to the validation perplexity after 300 training epochs.

BID36 27.2 8.7 5.3 488 2000 RL NASNet-C BID36 27.5 9.0 4.9 558 2000 RL AmoebaNet-A BID26 25.5 8.0 5.1 555 3150 evolution AmoebaNet-B BID26 26.0 8.5 5.3 555 3150 evolution AmoebaNet-C BID26 24.3 7.6 6.4 570 3150 evolution PNAS BID12 25.

The CIFAR-10 results for convolutional architectures are presented in TAB0 .

Notably, DARTS achieved comparable results with the state of the art BID36 BID26 while using three orders of magnitude less computation resources (i.e. 1.5 or 4 GPU days vs 2000 GPU days for NASNet and 3150 GPU days for AmoebaNet).

Moreover, with slightly longer search time, DARTS outperformed ENAS BID24 ) by discovering cells with comparable error rates but less parameters.

The longer search time is due to the fact that we have repeated the search process four times for cell selection.

This practice is less important for convolutional cells however, because the performance of discovered architectures does not strongly depend on initialization FIG2 .Alternative Optimization Strategies To better understand the necessity of bilevel optimization, we investigated a simplistic search strategy, where α and w are jointly optimized over the union of the training and validation sets using coordinate descent.

The resulting best convolutional cell (out of 4 runs) yielded 4.16 ± 0.16% test error using 3.1M parameters, which is worse than random search.

In the second experiment, we optimized α simultaneously with w (without alteration) using SGD, again over all the data available (training + validation).

The resulting best cell yielded 3.56 ± 0.10% test error using 3.0M parameters.

We hypothesize that these heuristics would cause α (analogous to hyperparameters) to overfit the training data, leading to poor generalization.

Note that α is not directly optimized on the training set in DARTS.

Table 2 presents the results for recurrent architectures on PTB, where a cell discovered by DARTS achieved the test perplexity of 55.7.

This is on par with the state-of-the-art model enhanced by a mixture of softmaxes BID31 , and better than all the rest of the architectures that are either manually or automatically discovered.

Note that our automatically searched cell outperforms the extensively tuned LSTM BID17 , demonstrating the importance of architecture search in addition to hyperparameter search.

In terms of efficiency, the overall cost (4 runs in total) is within 1 GPU day, which is comparable to ENAS and significantly faster than NAS BID35 .It is also interesting to note that random search is competitive for both convolutional and recurrent models, which reflects the importance of the search space design.

Nevertheless, with comparable or less search cost, DARTS is able to significantly improve upon random search in both cases (2.76 ± 0.09 vs 3.29 ± 0.15 on CIFAR-10; 55.7 vs 59.4 on PTB).

TAB4 show that the cell learned on CIFAR-10 is indeed transferable to ImageNet.

It is worth noticing that DARTS achieves competitive performance with the state-of-the-art RL method BID36 while using three orders of magnitude less computation resources.

TAB6 shows that the cell identified by DARTS transfers to WT2 better than ENAS, although the overall results are less strong than those presented in Table 2 for PTB.

The weaker transferability between PTB and WT2 (as compared to that between CIFAR-10 and ImageNet) could be explained by the relatively small size of the source dataset (PTB) for architecture search.

The issue of transferability could potentially be circumvented by directly optimizing the architecture on the task of interest.

BID4 -68.9 --manual LSTM BID18 69.1 66.0 33 -manual LSTM + skip connections BID17 69.1 65.9 24 -manual LSTM + 15 softmax experts BID31 66

We presented DARTS, a simple yet efficient architecture search algorithm for both convolutional and recurrent networks.

By searching in a continuous space, DARTS is able to match or outperform the state-of-the-art non-differentiable architecture search methods on image classification and language modeling tasks with remarkable efficiency improvement by several orders of magnitude.

There are many interesting directions to improve DARTS further.

For example, the current method may suffer from discrepancies between the continuous architecture encoding and the derived discrete architecture.

This could be alleviated, e.g., by annealing the softmax temperature (with a suitable schedule) to enforce one-hot selection.

It would also be interesting to explore performance-aware architecture derivation schemes based on the one-shot model learned during the search process.

A EXPERIMENTAL DETAILS A.1 ARCHITECTURE SEARCH A.1.1 CIFAR-10Since the architecture will be varying throughout the search process, we always use batch-specific statistics for batch normalization rather than the global moving average.

Learnable affine parameters in all batch normalizations are disabled during the search process to avoid rescaling the outputs of the candidate operations.

To carry out architecture search, we hold out half of the CIFAR-10 training data as the validation set.

A small network of 8 cells is trained using DARTS for 50 epochs, with batch size 64 (for both the training and validation sets) and the initial number of channels 16.

The numbers were chosen to ensure the network can fit into a single GPU.

We use momentum SGD to optimize the weights w, with initial learning rate η w = 0.025 (annealed down to zero following a cosine schedule without restart BID14 ), momentum 0.9, and weight decay 3 × 10 −4 .

We use zero initialization for architecture variables (the α's in both the normal and reduction cells), which implies equal amount of attention (after taking the softmax) over all possible ops.

At the early stage this ensures weights in every candidate op to receive sufficient learning signal (more exploration).

We use Adam BID10 as the optimizer for α, with initial learning rate η α = 3 × 10 −4 , momentum β = (0.5, 0.999) and weight decay 10 −3 .

The search takes one day on a single GPU 3 .A.1.2 PTB For architecture search, both the embedding and the hidden sizes are set to 300.

The linear transformation parameters across all incoming operations connected to the same node are shared (their shapes are all 300 × 300), as the algorithm always has the option to focus on one of the predecessors and mask away the others.

Tying the weights leads to memory savings and faster computation, allowing us to train the continuous architecture using a single GPU.

Learnable affine parameters in batch normalizations are disabled, as we did for convolutional cells.

The network is then trained for 50 epochs using SGD without momentum, with learning rate η w = 20, batch size 256, BPTT length 35, and weight decay 5 × 10 −7 .

We apply variational dropout BID3 of 0.2 to word embeddings, 0.75 to the cell input, and 0.25 to all the hidden nodes.

A dropout of 0.75 is also applied to the output layer.

Other training settings are identical to those in BID18 ; BID31 .

Similarly to the convolutional architectures, we use Adam for the optimization of α (initialized as zeros), with initial learning rate η α = 3 × 10 −3 , momentum β = (0.9, 0.999) and weight decay 10 −3 .

The search takes 6 hours on a single GPU.

A large network of 20 cells is trained for 600 epochs with batch size 96.

The initial number of channels is increased from 16 to 36 to ensure our model size is comparable with other baselines in the literature (around 3M).

Other hyperparameters remain the same as the ones used for architecture search.

Following existing works BID24 BID36 BID12 BID26 , additional enhancements include cutout (DeVries & Taylor, 2017) , path dropout of probability 0.2 and auxiliary towers with weight 0.4.

The training takes 1.5 days on a single GPU with our implementation in PyTorch BID21 .

Since the CIFAR results are subject to high variance even with exactly the same setup BID13 , we report the mean and standard deviation of 10 independent runs for our full model.

To avoid any discrepancy between different implementations or training settings (e.g. the batch sizes), we incorporated the NASNet-A cell BID36 and the AmoebaNet-A cell BID26 into our training framework and reported their results under the same settings as our cells.

<|TLDR|>

@highlight

We propose a differentiable architecture search algorithm for both convolutional and recurrent networks, achieving competitive performance with the state of the art using orders of magnitude less computation resources.