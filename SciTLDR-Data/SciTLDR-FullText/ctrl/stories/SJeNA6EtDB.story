Modern deep neural networks (DNNs) require high memory consumption and large computational loads.

In order to deploy DNN algorithms efficiently on edge or mobile devices, a series of DNN compression algorithms have been explored, including the line of works on factorization methods.

Factorization methods approximate the weight matrix of a DNN layer with multiplication of two or multiple low-rank matrices.

However, it is hard to measure the ranks of DNN layers during the training process.

Previous works mainly induce low-rank through implicit approximations or via costly singular value decomposition (SVD) process on every training step.

The former approach usually induces a high accuracy loss while the latter prevents DNN factorization from efficiently reaching a high compression rate.

In this work, we propose SVD training, which first applies SVD to decompose DNN's layers and then performs training on the full-rank decomposed weights.

To improve the training quality and convergence, we add orthogonality regularization to the singular vectors, which ensure the valid form of SVD and avoid gradient vanishing/exploding.

Low-rank is encouraged by applying sparsity-inducing regularizers on the singular values of each layer.

Singular value pruning is applied at the end to reach a low-rank model.

We empirically show that SVD training can significantly reduce the rank of DNN layers and achieve higher reduction on computation load under the same accuracy, comparing to not only previous factorization methods but also state-of-the-art filter pruning methods.

The booming development in deep learning models and applications has enabled beyond human performance in tasks like large-scale image classification (Krizhevsky et al., 2012; He et al., 2016; Hu et al., 2018; Huang et al., 2017) , object detection (Redmon et al., 2016; Liu et al., 2016; He et al., 2017) , and semantic segmentation (Long et al., 2015; Chen et al., 2017) .

Such high performance, however, comes with a high price of large memory consumption and computation load.

For example, a ResNet-50 model needs approximately 4G floating-point operations (FLOPs) to classify a color image of 224 ?? 224 pixels.

The computation load can easily expand to tens or even hundreds of GFLOPs for detection or segmentation models using state-of-the-art (SOTA) DNNs as backbones (Canziani et al., 2016 ).

This is a major challenge that prevents the deployment of modern DNN models on resource-constrained platforms, such as phones, smart sensors, and drones.

Model compression techniques for DNN models have been extensively studied.

Some successful methods include element-wise pruning (Han et al., 2015; Liu et al., 2015; Zhang et al., 2018) , structural pruning (Wen et al., 2016; Luo et al., 2017; Li et al., 2019) , quantization (Liu et al., 2018; Wang et al., 2019) , and factorization (Jaderberg et al., 2014; Zhang et al., 2015; Yang et al., 2015; Xu et al., 2018) .

Among these methods, quantization and element-wise pruning can effectively reduce model's memory consumption, but require specific hardware to realize efficient computation.

Structural pruning reduces the computation load by removing redundant filters or channels.

However, the complicated structures adopted in some modern DNNs (i.e., ResNet or DenseNet) enforce strict constraints on the input/output dimension of certain layers.

This requires additional filter grouping during the pruning and filter rearranging after the pruning to make the pruned structure valid (Wen et al., 2017a; Ding et al., 2019) .

Factorization method approximates the weight matrix of a layer with a multiplication of two or more low-rank matrices.

It by nature keeps the input/output dimension of a layer unchanged, and therefore the resulted decomposed network can be supported by any common DNN computation architectures, without additional grouping and post-processing.

The previous investigation show that it is feasible to approximate the weight matrices of a pretrained DNN model with the multiplication of low-rank matrices, but it may greatly degrade the performance (Jaderberg et al., 2014; Zhang et al., 2015; .

Some other methods attempt to manipulate the "directions" of filters to implicitly reduce the rank of weight matrices (Wen et al., 2017b; Li et al., 2019) .

However, the difficulties in training and the implicitness of rank representation prevent these methods from reaching a high compression rate.

Nuclear norm regularizer (Xu et al., 2018) has been used to directly reduce the rank of weight matrices.

Optimizing the nuclear norm requires back propagation through singular value decomposition (SVD).

Applying such a numerical process on every training step is inefficient and unstable.

Our work aims to explicitly achieve a low-rank DNN network during the training without applying SVD on every step.

In particular, we propose SVD training by training the weight matrix of each layer in the form of its full-rank SVD.

The weight matrix is decomposed into the matrices of left-singular vectors, singular values and right-singular vectors, and the training is done on the decomposed variables.

Furthermore, two techniques are proposed to induce low-rank while maintaining high performance during the SVD training: (1) Singular vector orthogonality regularization which keeps the singular vector matrices close to unitary through the training.

It mitigates gradient vanishing/exploding during the training, and provide a valid form of SVD to guarantee the effective rank reduction.

(2) Singular value sparsification which applies sparsity-inducing regularizers on the singular values during the training to induce low-rank.

The low-rank model is finally achieved through singular value pruning.

We evaluate the individual contribution of each technique as well as the overall performance when putting them together via ablation studies.

Results show that the proposed method constantly beats SOTA factorization and structural pruning methods on various tasks and model structures.

Approximating a weight matrix with the multiplication of low-rank matrices is a straightforward idea for compressing DNNs.

Early works in this field focus on designing the matrix decomposition scheme, so that the operation of a pretrained network layer can be closely approximated with cascaded low-rank layers (Jaderberg et al., 2014; Zhang et al., 2015; Yang et al., 2015) .

Notably, Zhang et al. (2015) propose a channel-wise decomposition, which uses SVD to decompose a convolution layer with a kernel size w ?? h into two consecutive layers with kernel sizes w ?? h and 1 ?? 1, respectively.

The computation reduction can be achieved by exploiting the channel-wise redundancy, e.g., channels with smaller singular values in both decomposed layers are removed.

Similarly, Jaderberg et al. (2014) propose to decompose a convolution layer into two consecutive layers with less channels in between.

They further utilize the spatial-wise redundancy to reduce the size of convolution kernels in the decomposed layers to 1 ?? h and w ?? 1, respectively.

These methods provide a closed-form decomposition for each layer.

However, the weights of the pretrained model may not be low-rank by nature, so the manually imposed low-rank after decomposition inevitably leads to high accuracy loss as the compression ratio increases (Xu et al., 2018) .

Methods have been proposed to reduce the rank of weight matrices during training process in order to achieve low-rank decomposition with low accuracy loss.

Wen et al. (2017b) induce low rank by applying an "attractive force" regularizer to increase the correlation of different filters in a certain layer.

Ding et al. (2019) achieve a similar goal by optimizing with "centripetal SGD," which moves multiple filters towards a set of clustering centers.

Both methods can reduce the rank of the weight matrices without performing actual low-rank decomposition during the training.

However, the rank representations in these methods are implicit, so the regularization effects are weak and may lead to sharp performance decrease when seeking for a high speedup.

On the other hand, Xu et al. (2018) explicitly estimate and reduce the rank throughout the training by adding Nuclear Norm (defined as the sum of all singular values) regularizer to the training objective.

This method requires performing SVD to compute and optimize the Nuclear Norm of each layer on every optimization step.

Since the complexity of the SVD operation is O(n 3 ) and the gradient computation through SVD is not straightforward (Giles, 2008) , performing SVD on every step is time consuming and leads to instability in the training process.

To explicitly achieve a low-rank network without performing decomposition on each training step, Tai et al. (2015) propose to directly train the network from scratch in the low-rank decomposed form.

Since decomposition typically doubles the number of layers in the network, directly training the decomposed network may be difficult due to gradient vanishing or exploding.

Tai et al. (2015) tackle this problem by adding batch normalization (Ioffe & Szegedy, 2015) between decomposed layers.

However, adding batch normalization breaks the theoretical guarantee that the decomposed layers can approximate the operation of the original layer and will largely hurt the inference efficiency.

Moreover, the low-rank decomposed training scheme used in this line of works requires setting the rank of each layer before the training (Tai et al., 2015; Ioffe & Szegedy, 2015) .

The manually chosen low rank may not lead to the optimal compression and will make the optimization harder as lower rank implies lower model capacity (Xu et al., 2018) .

Building upon previous works, we combine the ideas of decomposed training and trained low-rank in this work.

As shown in Figure 1 , the model will first be trained in a decomposed form through the full-rank SVD training, then undergoes singular value pruning for rank reduction, and finally be finetuned for further accuracy recovery.

As we will explain in Section 3.1, the model will be trained in the form of the spatial-wise (Jaderberg et al., 2014) or channel-wise decomposition (Zhang et al., 2015) to avoid the time consuming SVD.

Unlike the training procedure proposed by Tai et al. (2015) , we will train the decomposed model in its full-rank to preserve the model capacity.

During the SVD training, we apply orthogonality regularization to the singular vector matrices and sparsity-inducing regularizers to the singular values of each layer, the details of which will be discussed in Section 3.2 and 3.3, respectively.

Section 3.4 will elaborate the full objective of the SVD training and the overall model compression pipeline.

This method is able to achieve optimal compression rate by inducing low-rank through training without the need for performing decomposition on every training step.

In this work, we propose to train the neural network in its low-rank decomposition form, where each layer is decomposed into two consecutive layers via SVN, without introducing additional operations in between.

For a fully connected layer, the weight W is a 2-D matrix with dimension W ??? R m??n .

Following the form of SVD, W can be directly decomposed into three variables U , V , s as U diag(s)V T , with dimension U ??? R m??r , V ??? R n??r and s ??? R r .

Both U and V shall be unitary matrices.

In the full-rank setting where r = min(m, n), W can be exactly reconstructed as

T .

For a neural network, this is equivalent to decomposing a layer with weight W into two consecutive layers with weight

For a convolution layer, the kernel K can be represented as a 4-D tensor with dimension K ??? R n??c??w??h .

Here n, c, w, h represent the numbers of filters, the number of input channels, the width and the height of the filter respectively.

This work mainly focuses on the channel-wise decomposition method (Zhang et al., 2015) and the spatial-wise decomposition method (Jaderberg et al., 2014) to decompose the convolution layer, as these methods have shown their effectiveness in previous CNN decomposition research.

For channel-wise decomposition, K is first reshaped to a 2-D matrixK ??? R n??cwh .K is then decomposed with SVD into U ??? R n??r , V ??? R cwh??r and s ??? R r , where U and V are unitary matrices and r = min(n, cwh).

The original convolution layer is therefore decomposed into two consecutive layers with kernels

.

Spatial-wise decomposition shares a similar process as the channel-wise decomposition.

The major difference is that K is now reshaped toK ??? R nw??ch and then decomposed into U ??? R nw??r , V ??? R ch??r , and s ??? R r with r = min(nw, ch).

The resulting decomposed layers would have kernels K 1 ??? R r??c??1??h and K 2 ??? R n??r??w??1 respectively.

Zhang et al. (2015) and Jaderberg et al. (2014) 's works theoretically show that the decomposed layers can exactly replicate the function of the original convolution layer in the full-rank setting.

Therefore training the decomposed model at full-rank should achieve a similar accuracy as training the original model.

During the SVD training, for each layer we use the variables from the decomposition, i.e., U , s, V , instead of the original kernel K or weight W as the trainable variables in the network.

The forward pass will be executed by converting the U , s, V into a form of the two consecutive layers as demonstrated above, and the back propagation and optimization will be done directly with respect to the U , s, V of each layer.

In this way, we can access the singular value s directly without performing the time-consuming SVD on each step.

Note that U and V need to be orthogonal for efficient rank reduction via SVD, but this is not naturally induced by the decomposed training process.

Also, Training with the decomposed variables may aggravate gradient vanishing/exploding during the optimization.

Therefore we add orthogonality regularization to U and V to tackle the aforementioned problems, as discussed in Section 3.2.

Rank reduction is induced by adding sparsity-inducing regularizers to the s of each layer, which will be discussed in Section 3.3.

In a standard SVD procedure, the resulted U and V should be orthogonal by construction, which provides theoretical guarantee for the low-rank approximation.

However, U and V in each layer are treated as free trainable variables in the decomposed training process, so the orthogonality may not hold.

Without the orthogonal property, it is unsafe to prune s even if it reaches a small value, because the corresponding singular vectors in U and V may have high energy and induce a large difference to the result.

To make the form of SVD valid and enable effective rank reduction via singular value pruning, we introduce an orthogonality regularization loss to U and V as:

where || ?? || F is the Frobenius norm of matrix and r is the rank of U and V .

Note that the ranks of U and V are same given their definition in the decomposed training procedure.

Adding the orthogonality loss in Equation (1) to the total loss function forces U s and V s of all the layers close to be orthogonal matrices.

Meanwhile, one layer in the original network is converted to two consecutive layers in the decomposed training, therefore doubles the number of layers.

As aforementioned by Ioffe & Szegedy (2015) , this may worsen the problem of exploding or vanishing gradient during the optimization, degrading the performance of the achieved model.

Since the proposed orthogonality loss can keep all the columns of U and V to have the L 2 norms close to 1, adding it to the decomposed training objective can effectively mitigate the exploding or vanishing gradient, therefore achieving high accuracy.

The accuracy gain brought by training with the orthogonality loss will be discussed in our ablation study in Section 4.1.

With orthogonal singular vector matrices, reducing the rank of the decomposed network is equivalent to making the singular value vector s of each layer sparse.

Although the sparsity of a vector is directly represented by its L 0 norm, it is hard to optimize the norm through gradient-based methods.

Inspired by the recent works in DNN pruning (Liu et al., 2015; Wen et al., 2016) , we use differentiable sparsity-inducing regularizer to make more elements in s closer to zero, and apply post-train pruning to make the singular value vector sparse.

For the choice of the sparsity-inducing regularizer, the L 1 norm has been commonly applied in feature selection (Tibshirani, 1996) and DNN pruning (Wen et al., 2016) .

The L 1 regularizer takes the form of L 1 (s) = i |s i |, which is both almost everywhere differentiable and convex, making it friendly for optimization.

Moreover, applying L 1 regularizer on the singular value s is equivalent to regularizing with the nuclear norm of the original weight matrix, which is a popular approximation of the rank of a matrix (Xu et al., 2018) .

However, the L 1 norm is proportional to the scaling of parameters, i.e., ||??W || 1 = ??||W || 1 , with a non-negative constant ??.

Therefore, minimizing the L 1 norm of s will shrink all the singular values simultaneously.

In such a situation, some singular values that are close to zero after training may still contain a large portion of the matrix's energy.

Pruning such singular values may undermine the performance of the neural network.

To mitigate the proportional scaling problem of the L 1 regularizer, previous works in compressed sensing have been using Hoyer regularizer to induce sparsity in solving non-negative matrix factorization (Hoyer, 2004) and blind deconvolution (Krishnan et al., 2011) , where the Hoyer regularizer shows superior performance comparing to other methods.

The Hoyer regularizer is formulated as

which is the ratio of the L 1 norm and the L 2 norm of a vector (Krishnan et al., 2011) .

It can be easily seen that the Hoyer regularizer is almost everywhere differentiable and scale-invariant.

The differentiable property implies that the Hoyer regularizer can be easily optimized as part of the objective function.

The scale-invariant property shows that if we apply the Hoyer regularizer to s, the total energy will be retained as the singular values getting sparser.

Therefore most of the energy will be kept within the top singular values while the rest getting close to zero.

This makes Hoyer regularizer attractive in our training process.

The effectiveness of the L 1 regularizer and the L H regularizer is explored and compared in Section 4.2.

With the analysis above, we propose the overall objective function of the decomposed training as:

Here L T is the training loss computed on the model with decomposed layers.

L o denotes the orthogonality loss provided in Equation (1), which is calculated on the singular vector matrices U l and V l of layer l and added up over all D layers.

L s is the sparsity-inducing regularization loss, applying to the vector of singular values s l of each layer.

We explore the use of both the L 1 regularizer and the L H regularizer in Equation (2) as L s in this work.

?? s and ?? o are the decay parameters for the sparsity-inducing regularization loss and the orthogonality loss respectively, which are hyperparameters of the proposed training process.

?? o can be chosen as a large positive number to enforce the orthogonality of singular vectors, and ?? s can be modified to explore the tradeoff between accuracy and FLOPs of the achieved low-rank model.

As shown in Figure 1 , the low-rank decomposed network will be achieved through a three-stage process of full-rank SVD training, singular value pruning and low-rank finetuning.

First we train a full-rank decomposed network using the objective function in Equation (3).

Training at full rank enables the decomposed model to easily reach the performance of the original model, as there is no capacity loss during the full-rank decomposition.

With the help of the sparsity-inducing regularizer,

where e ??? [0, 1] is a predefined energy threshold.

We use the same threshold for all the layers in our experiments.

When e is small enough, the singular values in set K and the corresponding singular vectors can be removed safely with negligible performance loss.

The pruning step will dramatically reduce the rank of the decomposed layers.

For a convolution layer with kernel K ??? R n??c??w??h , if we can reduce the rank of the decomposed layers to r, the number of FLOPs for the convolution will be reduced by (n+chw)r nchw or (nw+ch)r nchw when channel-wise or spatial-wise decomposition is applied, respectively.

The resulted low-rank model will then be finetuned with ?? s set to zero for further performance recovery.

In this section, we first perform ablation studies on the importance of the singular vector orthogonality regularization and the choice of singular value sparsity regularizers.

The studies use ResNet models (He et al., 2016) on the CIFAR-10 dataset (Krizhevsky & Hinton, 2009 ).

We then apply the proposed decomposed training method on various DNN models on the CIFAR-10 dataset and the ImageNet ILSVRC-2012 dataset (Russakovsky et al., 2015) .

Different hyperparameters are used to explore the accuracy-FLOPs trade-off induced by the proposed method.

Our results constantly stay above the Pareto frontier of previous works.

Here we demonstrate the importance of adding the singular value orthogonality loss to the decomposed training process.

We separately train two decomposed model with the same optimizer and hyperparameters, one with the orthogonality loss of ?? o = 1.0 and the other with ?? o = 0.

No sparsity-inducing regularizer is applied to the singular values in this set of experiments.

The experiments are conducted on ResNet-56 and ResNet-110 models, both trained under channel-wise decomposition and spatial-wise decomposition.

The CIFAR-10 dataset is used for training and testing.

As shown in Table 1 , the orthogonality loss enables the decomposed model to achieve similar or even better accuracy comparing to that of the original full model.

On the contrary, training the decomposed model without the orthogonality loss will cause around 2% accuracy loss.

With the proposed decomposed training method, there are two main factors related to the final compression rate and the performance of the compressed model: the decomposition method for the DNN layers and the choice of sparsity-inducing regularizers for the singular values.

As mentioned in Section 3.1 and Section 3.3, we mainly consider the channel-wise and the spatial-wise decomposition method, with the L 1 and the Hoyer regularizer.

In this section, we comprehensively explore the accuracy-compression rate tradeoff of ResNet models in various depths and under different configurations by changing the strength of the sparsity-inducing regularizer (?? s in Equation (3)).

From the results shown in Figure 2 , we make the following observations:

Channel-wise decomposition works as well as spatial-wise decomposition in deeper neural networks Here we compare the accuracy-#FLOPs tradeoff tendency of the channel-wise decomposition (red) and the spatial-wise decomposition (blue), both with the Hoyer regularizer.

The spatialwise decomposition shows a large advantage comparing to the channel-wise decomposition in the experiments with a shallower network like ResNet-20 or ResNet-32.

However, with a deeper network like ResNet-110, these two decomposition methods perform similarly.

Spatial-wise decomposition can utilize both spatial-wise redundancy and channel-wise redundancy, while the channel-wise decomposition utilizes channel-wise redundancy only.

The observations indicate that as networks get deeper, the channel-wise redundancy will become a dominant factor comparing to the spatialwise redundancy.

This corresponds to the fact that deeper layers in modern DNN typically have significantly more channels than shallower layers, resulting in significant channel-wise redundancy.

Hoyer achieves higher speedup under low accuracy loss comparing to the L 1 regularizer Here we compare the effect of the L 1 regularizer (magenta) and the Hoyer regularizer (red), both with spatial decomposition.

As shown in Figure 2 , the tradeoff tendency of the L 1 regularizer constantly demonstrates a larger slope than that of the Hoyer regularizer.

Under low accuracy loss, the Hoyer regularizer achieves a higher compression rate comparing to that of the L 1 regularizer.

However, if we are aiming for extremely high compression rate while allowing higher accuracy loss, the L 1 regularizer can have a better performance.

One possible reason for the difference in tendency is that the L 1 regularizer will make all the singular values small through the training process, while the Hoyer regularizer will maintain the total energy of the singular values during the training, focusing more energy in larger singular values.

Therefore more singular values can be removed from the decomposed model trained with the Hoyer regularizer without significantly hurting the performance of the model, resulting in higher compression rate at low accuracy loss.

But it would be harder to keep most of the energy in a tiny amount of singular values than simply making everything closer to zero, therefore the L 1 regularizer may perform better in the case of extremely high speedup.

We apply the proposed SVD training framework on the ResNet-20, ResNet-32, ResNet-56 and ResNet-110 models on the CIFAR-10 dataset as well as the ResNet-50 model on the ImageNet ILSVRC-2012 dataset to compare the accuracy-#FLOPs tradeoff with previous methods.

The hyperparameter choices of these models can be found in Appendix A. Here we mainly compare our method with state-of-the-art low-rank compression methods like TRP (Xu et al., 2018) and C-SGD (Ding et al., 2019) , as well as recent filter pruning methods like NISP (Yu et al., 2018) , SFP (He et al., 2018) and CNN-FCF (Li et al., 2019) .

The results of different models are shown in Figure 3 .

As analyzed in Section 4.2, the spatial-wise decomposition methods achieves significantly higher compression rate than the channel-wise decomposition in shallower networks, while similar performance can be achieved when compressing a deeper model.

Thus we compare the results of only the spatial-wise decomposition against previous works for ResNet-20 and ResNet-32.

For other deeper networks, we report the results for both channel-wise and spatial-wise decomposition.

As most of the previous works focus on compressing the model with a small accuracy loss, here we use the Hoyer regularizer for the singular values sparsity, as it can achieve a better compression rate than the L 1 norm under low accuracy loss (see Section 4.2).

We use multiple strength for the Hoyer regularizer to explore the accuracy-#FLOPs tradeoff, in order to compare against previous works with different accuracy levels.

As shown in Figure 3 , our proposed method can constantly achieve higher FLOPs reduction with less accuracy loss comparing to previous methods on different models and datasets.

These comparison results prove that the proposed SVD training and singular value pruning scheme can effectively compress modern deep neural networks through low-rank decomposition.

In this work, we propose the SVD training framework, which incorporates the full-rank decomposed training and singular value pruning to reach low-rank DNNs with minor accuracy loss.

We apply SVD to decompose each DNN layer before the training and directly train with the decomposed singular vectors and singular values, so we can keep an explicit measure of layers' ranks without performing the SVD on each step.

Orthogonality regularizers are applied to the singular vectors during the training to keep the decomposed layers in a valid SVD form.

And sparsity-inducing regularizers are applied to the singular values to explicitly induce low-rank layers.

Thorough experiments are done to analyse each proposed technique.

We demonstrate that the orthogonality regularization on singular vector matrices is crucial to the performance of the decomposed training process.

For decomposition methods, we find that the spatial-wise method performs better than channel-wise in shallower networks while the performances are similar for deeper models.

For the sparsity-inducing regularizer, we show that higher compression rate can be achieved by Hoyer regularizer comparing to that of the L 1 regularizer under low accuracy loss.

We further apply the proposed method to various depth of ResNet models on both CIFAR-10 and ImageNet dataset, where we find the accuracy-#FLOPs tradeoff achieved by the proposed method constantly stays above the Pareto frontier of previous methods, including both factorization and structural pruning methods.

These results prove that this work provides an effective way for learning low-rank deep neural networks.

1943-1955, 2015.

A EXPERIMENT SETUPS Our experiments are done on the CIFAR-10 dataset (Krizhevsky & Hinton, 2009 ) and the ImageNet ILSVRC-2012 dataset (Russakovsky et al., 2015) .

We access both datasets via the API provided in the "TorchVision" Python package.

As recommended in the PyTorch tutorial, we normalize the data and augment the data with random crop and random horizontal flip before the training.

We use batch size 100 to train CIFAR-10 model and use 256 for the ImageNet model.

For all the models on CIFAR-10, both the full-rank SVD training and the low-rank finetuning are trained for 164 epochs.

The learning rate is set to 0.001 initially and decayed by 0.1 at epoch 81 and 122.

For models on ImageNet, the full-rank SVD training is trained for 90 epochs, with initial learning rate 0.1 and learning rate decayed by 0.1 every 30 epochs.

The finetuning is done for 60 epochs, starting at learning rate 0.01 and decay by 0.1 at epoch 30.

We use pretrained full-rank decomposed model (trained with the orthogonality regularizer but without sparsity-inducing regularizer) to initialize the SVD training.

SGD optimizer with momentum 0.9 is used for optimizing all the models, with weight decay 5e-4 for CIFAR-10 models and 1e-4 for ImageNet models.

The accuracy reported in the experiment is the best testing accuracy achieved during the finetuning process.

During the SVD training, the decay parameter of the orthogonality regularizer ?? o is set to 1.0 for both channel-wise and spatial-wise decomposition on CIFAR-10.

On ImageNet, ?? o is set to 10.0 for channel-wise decomposition and 5.0 for spatial-wise decomposition.

The decay parameter ?? s for the sparsity-inducing regularizer and the energy threshold used for singular value pruning are altered through different set of experiments to fully explore the accuracy-#FLOPs tradeoff.

In most cases, the energy threshold is selected through a line search, where we find the highest percentage of energy that can be pruned without leading to a sudden accuracy drop.

The ?? s and the energy thresholds used in each set of the experiments are reported alongside the experiment results in Appendix B.

In this section we list the exact data used to plot the experiment result figures in Section 4.

The results of our proposed method with various choice of decomposition method and sparsity-inducing regularizer tested on the CIFAR-10 dataset are listed in Table 2 .

All of these data points are visualized in Figure 2 to compare the tradeoff tendency.

As discussed in Section 4.3, the results of spatialwise decomposition with the Hoyer regularizer for ResNet-20 and ResNet-32 are shown in Figure 3 to compare with previous methods.

The results of both channel-wise and spatial-wise decomposition with the Hoyer regularizer are compared with previous methods in Figure 3 for ResNet-56 and ResNet-110.

The results of our method for the ResNet-50 model tested on the ImageNet dataset are listed in The baseline results of previous works on compressing CIFAR-10 and ImageNet models used for comparison in Figure 3 are listed in Table 4 and Table 5 respectively.

As there are a large amount of previous works in this field, we only list the results of the most recent works here to show the state-of-the-art Pareto frontier.

Therefore we choose state of the art low-rank compression methods like TRP (Xu et al., 2018) and C-SGD (Ding et al., 2019) , as well as recent filter pruning methods like NISP (Yu et al., 2018) , SFP (He et al., 2018) and CNN-FCF (Li et al., 2019) as the baseline to compare our results against.

<|TLDR|>

@highlight

Efficiently inducing low-rank deep neural networks via SVD training with sparse singular values and orthogonal singular vectors.

@highlight

This paper introduces an approach to network compression by encouraging the weight matrix in each layer to have a low rank and explicitly factorizing the weight matrices into an SVD-like factorization for treatment as new parameters.

@highlight

Proposal to parametrize each layer of a deep neural network, before training, with a low-rank matrix decomposition, accordingly replace convolutions with two consecutive convolutions, and then train the decomposed method.