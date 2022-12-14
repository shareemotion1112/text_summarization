Convolutional Neural Networks (CNNs) are computationally intensive, which limits their application on mobile devices.

Their energy is dominated by the number of multiplies needed to perform the convolutions.

Winograd’s minimal filtering algorithm (Lavin, 2015) and network pruning (Han et al., 2015) can reduce the operation count, but these two methods cannot be straightforwardly combined — applying the Winograd transform fills in the sparsity in both the weights and the activations.

We propose two modifications to Winograd-based CNNs to enable these methods to exploit sparsity.

First, we move the ReLU operation into the Winograd domain to increase the sparsity of the transformed activations.

Second, we prune the weights in the Winograd domain to exploit static weight sparsity.

For models on CIFAR-10, CIFAR-100 and ImageNet datasets, our method reduces the number of multiplications by 10.4x, 6.8x and 10.8x respectively with loss of accuracy less than 0.1%, outperforming previous baselines by 2.0x-3.0x.

We also show that moving ReLU to the Winograd domain allows more aggressive pruning.

Deep Convolutional Neural Networks (CNNs) have shown significant improvement in many machine learning applications.

However, CNNs are compute-limited.

Their performance is dominated by the number of multiplies needed to perform the convolutions.

Moreover, the computational workload of CNNs continues to grow over time.

BID16 proposed a CNN model with less than 2.3 × 10 7 multiplies for handwritten digit classification.

Later, BID13 developed AlexNet, an ImageNet-winning CNN with more than 1.1 × 10 9 multiplies.

In 2014, ImageNetwinning and runner up CNNs increased the number of multiplies to 1.4 × 10 9 BID24 ) and 1.6 × 10 10 BID22 respectively.

Despite the powerful representational ability of large scale CNNs, their computational workload prohibits deployment on mobile devices.

Two research directions have been explored to address the problem.

BID14 proposed using Winograd's minimal filtering algorithm BID25 to reduce the number of multiplies needed to perform 3 × 3 kernel convolutions.

On the other end, pruning the model BID5 and exploiting the dynamic sparsity of activations due to ReLU also reduces the required multiplies.

Unfortunately, the above two directions are not compatible: the Winograd transformation fills in the zeros in both the weights and the activations FIG0 ) -eliminating the gain from exploiting sparsity.

Thus, for a pruned network, Winograd's algorithm actually increases the number of multiplies; the loss of sparsity more than offsets the reduced operation count.

In this paper, we introduce two modifications to the original Winograd-based convolution algorithm to eliminate this problem.

First, we move the ReLU operation to be after the Winograd transform to also make the activations sparse at the point where the multiplies are performed.

Second, we prune the weights after (rather than before) they are transformed.

Thus, the weights are sparse when the elementwise multiply is performed -reducing the operation count.

Together, these two modifications enable the gains of Winograd's algorithm and of exploiting sparsity to be combined.

We open-source our code and models at https://github.com/xingyul/Sparse-Winograd-CNN.

Linear Algebra property in Convolution: Previous research proposes using the linear algebra property of convolution to reduce the number of multiplies by trading additions for multiplies.

BID3 convert convolution into matrix multiplies and utilize the linear algebra property at the sub-matrix block level.

This approach achieves a 47% saving in multiplies.

BID14 exploits the element-level linear algebra property of convolution, i.e. Winograd's minimal filtering algorithm BID25 .

This approach reduces the number of multiplies by 2.25× to 4×, depending on the image patch size used in the algorithm.

Winograd's algorithm is also used in a state-of-the-art deep learning library, cuDNN BID2 , to improve computation efficiency.

Model Compression: Model compression reduces the number of multiplies of CNNs by pruning network parameters BID15 BID8 and exploiting weight sparsity.

BID5 proposed learning the sparsity pattern of network weights by eliminating weights whose absolute value is less than an empirical threshold.

This approach can prune the convolutional layers of the model to only 30% − 50% of the original size and reduce the number of multiplies required.

Liu et al. (2017) first proposed pruning and re-training the weights in Winograd domain for conventional Winograd convolution.

Li et al. (2017) later showed promising results on large datasets and reported 90% sparsity in the Winograd parameters of AlexNet with less than 0.1% accuracy loss.

Dynamic Activation Sparsity: The ReLU non-linearity sets activations whose values are negative to zero, causing dynamic sparsity in activations.

Model compression can work in tandem with dynamic activation sparsity and reduce multiplication workload.

BID5 showed that exploiting sparsity of both weights and activations can reduce the number of multiplies by 4 − 11×.

BID11 further proposed to manually set a small positive ReLU threshold at test time to exploit greater sparsity in activation without losing testing accuracy.

Research in novel architectures also led to optimizations for deep learning accelerators to exploit the sparsity in activations.

BID6 proposed using a Leading Non-zero Detection unit (LNZD) for their fully-connected layer accelerator to efficiently skip zeros in input activations.

BID1 proposed a similar mechanism for a convolution layer accelerator.

We first introduce the conventional Winograd convolution and show how sparsity of weights or activations is lost during the dataflow of the algorithm.

We then present the novel Winograd-ReLU CNN architecture.

It preserves sparsity in both weights and activations before multiplies are performed and significantly reduces the computational workload.

The basic block of the conventional Winograd convolution algorithm works on an p×p patch (denoted by d) extracted with stride of (p − 2) × (p − 2) from an H × W input feature map.

With "valid" padding, the p×p patch is convolved with a 3×3 kernel (denoted by g) to produce an (p−2)×(p−2) output patch (denoted by S).

The output patches are assembled into an output feature map.

Input activation patch d and kernel g (spatial-domain activation and weights) are transformed using matrices B and G to be B T dB and GgG T (Winograd-domain activation and weights) respectively, both with shape p × p. After element-wise product in Winograd-domain, the output activation S is obtained using matrix A (equation (1) ).

Matrices B, G and A are p-specific.

When p = 4, B and A consists of 1, −1 and 0, so the multiplication with B and A only requires addition.

It reduces the number of multiplies from 9(p − 2) 2 to p 2 .

Lavin FORMULA1 gives details of the algorithm.

DISPLAYFORM0 Spatial Baseline Network: When using a "vanilla" pruned network, as introduced by BID5 , a ReLU non-linear operation is performed by the previous layer on spatial-domain input d and spatial-domain weight g is pruned.

The output activation patch S is obtained from equation FORMULA1 .

This is illustrated in FIG0 (a) for p = 4.

Though g and d may both be sparse due to pruning and ReLU respectively, the element-wise multiply is dense due to G(·)G T and B(·)B T transformations filling the spatial-domain zeros.

Sparsity does not reduce the number of multiplies in Winograd's algorithm.

DISPLAYFORM1 Winograd Native Pruned Network:

When using the Winograd-domain pruned network introduced by Liu et al. (2017) and BID17 , the spatial-domain input d is ReLU-ed by the previous layer while the Winograd-domain weight GgG T is pruned.

The output activation patch S is obtained from equation (3).

The algorithm when p = 4 is also illustrated in FIG0 (b).

Though Winograd-domain weights are sparse due to pruning, Winograd-domain activations are still dense due to B(·)B T transforms.

The sparsity in spatial activations due to ReLU does not reduce the number of multiplies.

DISPLAYFORM2

To address the above problems, we introduce the Winograd-ReLU Network.

Instead of applying ReLU to the activations in the spatial domain, we apply ReLU to the activations in the Winograd domain, as in equation FORMULA3 and FIG0 (c).

The ReLU operation zeros all negative transformed activations, reducing the number of multiplies in the Winograd domain.

DISPLAYFORM0 In the Winograd-ReLU CNN, we eliminate the spatial-domain kernel entirely.

Because this ReLU is really associated with the previous layer, we perform this transformed ReLU starting with the second layer.

We point out that the proposed new CNN architecture is not mathematically equivalent to the vanilla CNN nor the conventional Winograd CNN.

Due to the change of network architecture, the training and pruning should also be changed.

Our method operates in three phases: dense training, pruning, and retraining.

Dense training: we train a dense p × p kernel directly in the transform domain.

The transformed kernel is initialized and trained directly by back-propagation through the inverse transform -eliminating the need to maintain a kernel in the spatial domain or to transform a spatial kernel.

Pruning: we prune the transformed kernel by computing the threshold t required to achieve a desired pruning rate r and setting all weights whose absolute value less than t to zero.

In our experiments, we used the same r for all Winograd-ReLU layers.

Because sensitivity varies from layer to layer, we expect that better performance could be achieved by varying the pruning rate r i for each layer i.

Re-training: we re-train the model using a "sparsity mask" to force the weights that were pruned to remain zero.

The sparsity mask is computed during the pruning step and is kept constant during re-training.

The gradient of the network's loss, L, with respect to the input activation and Winograd weights can be derived using the chain rule.

Equation FORMULA4 shows the calculation of input activation gradient ∇ d L and Winograd weight gradient ∇ GgG T L using the loss gradient passed from upstream layers DISPLAYFORM1 4 EXPERIMENTS We applied the methodology described above to several different CNNs on different datasets.

The original network models are chosen such that the majority of the convolution layers have 3 × 3 kernels.

This ensures the largest portion of layers can be converted to Winograd convolution layers and ReLU be put in Winograd domain.

We used image classification datasets of different scales: CIFAR-10, CIFAR-100 BID12 ) and ImageNet 2012 BID21 .

For network architectures, we chose VGG-nagadomi (Nagadomi, 2014), ConvPool-CNN-C model BID23 and a variation of ResNet-18 BID9 ) respectively on three datasets.

Using the Tensorflow BID0 ) framework, we trained the spatial baseline CNN, corresponding conventional Winograd CNN, and Winograd-ReLU CNN models from scratch.

Then the three models are iteratively pruned and re-trained.

For a specific dataset, we used the same data augmentation for the training of all models on the dataset.

We used VGG-nagadomi (Nagadomi, 2014) on the CIFAR-10 dataset.

VGG-nagadomi is a lightweight version of VGGNet BID22 .

It contains 8 convolution layers with 3×3 kernels.

The best reported validation set accuracy it achieves on CIFAR-10 is 93.31% (Nagadomi, 2014).

We trained three models from scratch.

The corresponding conventional Winograd CNN model and Winograd-ReLU CNN model can achieve validation set accuracy of 93.30% and 93.43% respectively.

The first convolution layer is most sensitive to pruning and we set its density to a constant of 80%.

We iteratively pruned and re-trained other convolution layers with density from 80% down to 20%.

Figure 2: Test accuracy vs density for the three models in FIG0 on VGG-nagadomi.

Figure 2 shows test accuracy as a function of weight density for the three models.

The two baseline models can only be pruned to 60% density before accuracy falls significantly (> 0.1%).

Our Winograd-ReLU CNN model can be pruned to 40% density before falling to the same accuracy.

TAB1 shows the input activation density and compares the workloads for each pruned convolution layer in three models.

Pruning two baseline models reduces the convolution layer workload by 5.1× and 3.7× 1 respectively.

Pruning the Winograd-ReLU model reduces the convolution layer workload by 13.3×, a 2.6× and 3.6× improvement respectively over the two baselines.

The improvement of overall network workload reduction is 2.2× and 3.0× respectively over two baselines.1 All Winograd CNN model workload reduction results include the intrinsic 2.25× reduction.

We used the ConvPool-CNN-C (Springenberg et al., 2015) model on on the CIFAR-100 dataset.

ConvPool-CNN-C contains 9 convolution layers, out of which 7 have 3 × 3 kernels.

We trained three models from scratch.

The spatial baseline CNN model and conventional Winograd CNN model can achieve single model validation accuracy of 69.34% and 69.32% respectively.

The corresponding Winograd-ReLU network model can achieve validation set accuracy of 69.75%.

We pruned the first convolution layer to a constant density of 80%.

We iteratively pruned and re-trained the other layers to densities from 80% down to 20%.

Figure 3: Test accuracy vs density for the three models in FIG0 on ConvPool-CNN-C. Figure 3 shows the accuracy as a function of density for spatial baseline and Winograd-ReLU models.

The spatial-baseline and Winograd-ReLU models can be pruned to 60% density without significant (> 0.1%) loss of accuracy.

In contrast, the conventional Winograd CNN model can only be pruned to 70% density.

At a given density, the Winograd-ReLU model has the highest accuracy.

TAB3 shows the input activation density and compares the workloads for each pruned convolution layer in three models.

Pruning two baseline models reduces the convolution layer workload by 3.5× and 3.2× respectively.

Pruning the Winograd-ReLU model reduces the workload by 7.1×, a 2.1× and 2.2× improvement respectively over the two baselines.

The improvement of overall network workload reduction is 2.0× and 2.2× respectively over two baselines.

DISPLAYFORM0

We used a variation of the full pre-activation version BID10 of ResNet-18 BID9 on the ImageNet 2012 dataset.

We used this version because it performs the best among various ResNet versions and its structure suits our Winograd-ReLU approach -its ReLU units are located before convolutions in the residual modules.

The variation is different from original ResNet-18 by replacing all 2 × 2-stride 3 × 3 convolution layers with a 2 × 2 max-pooling layer followed by a 1 × 1-stride 3 × 3 convolution layer.

Such difference ensure most of convolution layers can be converted to Winograd convolution layer.

Another difference is that it doesn't have the last max pooling layer so the last group of residual modules has spatial size of 14 × 14, in order to keep the spatial size even instead of odd.

This setting suits Winograd convolution with p = 4 best in that even spatial size is required for even p values.

We trained three models from scratch.

Figure 4 : Top-1 and top-5 validation accuracy vs density for three models on a variation of ResNet-18.

Figure 4 shows the accuracy as a function of density for three models.

The spatial baseline CNN model and conventional Winograd CNN model can be pruned to 60% and 50% respectively without significant (> 0.1%) loss of top-1 or top-5 accuracy.

The Winograd-ReLU model can be pruned much further, to 30%/35% density without significant (> 0.1%) loss of top-1/top-5 accuracy.

At these densities, top-1 accuracies are 66.53%, 66.45% and 66.61% for three models respectively, with a dense spatial baseline of 66.67%; top-5 accuracies are 87.29%, 87.30% and 87.35% for three models respectively, with a dense spatial baseline of 87.42%.

TAB5 shows the input activation density and compares the workloads for each pruned convolution layer in three models.

Pruning the two baseline models reduces the convolution layer workload by 5.1× and 4.5× respectively.

Pruning the Winograd-ReLU model reduces the workload by 13.2×, a 2.6× and 2.9× improvement respectively over the two baselines.

The improvement of overall network workload reduction is 2.3× and 2.6× respectively over two baselines.

In this section, we summarize the experiment results and compare the three models in terms of a) weight and activation dimensions and b) the dynamic density of activations.

We then visualize the kernels to illustrate the pattern of the proposed Winograd-ReLU model kernel.

DISPLAYFORM0

In a convolutional neural network, a convolution-ReLU pair acts as a classifier on a spatial patch of an input feature.

The dimension of the space being classified is the total number of elements passing through the ReLU layer.

The decision boundaries of the classifier are determined by the weights.

Insufficient non-zero weights or insufficient activations results in too simple a decision boundary and causes accuracy loss.

Experimental results have shown that Winograd-ReLU CNN can reach the same accuracy as both vanilla spatial baseline CNN and conventional Winograd CNN without pruning, and that WinogradReLU CNN is more robust to aggressive pruning.

In this subsection we provide an explanation for the latter observation from the aspect of activation and weight dimensions.

We provide a summary on dimensions in Table 4 .

Table 4 : Comparison of ReLU dimension and weight dimension in three types of networks.

Assume the convolution-ReLU pair operates on input activation of spatial size of H × W and the number of input and output channels are C and K respectively.

Spatial Baseline CNN BID5 Winograd native pruned CNN (Li et al., 2017) We can see that our Winograd-ReLU architecture has an advantage on the dimensions of weights and activations over other two models.

This means Winograd-ReLU CNNs classify on a higher dimension with more complex decision boundaries, which forms a stronger representational ability in high dimensional image feature space.

DISPLAYFORM0

As is shown in the ImageNet results in the previous section, dynamic activation density of spatial baseline CNN model varies significantly among layers.

Layers at earlier stages typically have higher density in activation than later stages.

In Winograd-ReLU CNN model, the dynamic activation densities vary little among layers and are all close to 50%.An explanation is that the nature of image convolution ensures activations d to be spatially smooth.

Thus, due to the structure of matrix B BID14 , 15 of 16 elements in the 4 × 4 matrix of Winograd-domain activation patch B T · d · B have a mean close to zero.

This benefits classification within a patch since ReLU layer is most powerful when half of activations are positive.

We visualize the kernels of the proposed Winograd-ReLU model.

We selected the first 6 input and output channels of layer res2a_2a of ResNet-18 at three different pruning densities.

Unlike spatial domain kernels, Winograd-ReLU kernels do not show clear physical meanings such as edge or corner detectors.

However, we observe that values of the (2, 2) elements (from top-left, 1-based indices) in each kernel are typically distinct in a kernel and are most likely kept during aggressive pruning.

A possible reason for this is that the (2, 2) elements of Winograd-domain activation in a 4 × 4 patch are special: interested readers can calculate B T · d · B symbolically and will realize that (2, 2) elements are the only elements that are transformed with a linear combination of only adding and no subtraction.

In a spatially smooth activation patch, this means the (2, 2) elements are the ones and the only ones with a non-zero mean.

We have shown that we can combine the computational savings of sparse weights and activations with the savings of the Winograd transform by making two modifcations to conventional CNNs.

To make the weights sparse at the point of multiplication, we train and prune the weights in the transform domain.

This simple approach does not reduce the workload with respect to spatial pruning, though, so we move the ReLU non-linear operation after the Winograd transform to make the activations sparse at the point of multiplication.

Moving ReLU to the Winograd domain also allows the weights to be more aggressively pruned without losing accuracy.

With a 2 × 2 output patch (p = 4), the net result is a reduction of 10.4×, 6.8× and 10.8× in computation on three datasets: CIFAR-10, CIFAR-100 and ImageNet.

We plan to extend this work in the following directions.

First, we expect that even greater savings on computation can be realized by using larger patch sizes (e.g., p = 6), and there may be benefit in exploring different Winograd transformation matrices (B,G and A).

Second, we expect that using different pruning rates r i for each network layer will help maintain accuracy and improve overall workload reduction.

Finally, we expect that combining our Winograd-ReLU network with other network simplification techniques, e.g. quantization of weights and/or activations BID4 BID18 BID20 , will reduce the energy of computation even further.

<|TLDR|>

@highlight

Prune and ReLU in Winograd domain for efficient convolutional neural network