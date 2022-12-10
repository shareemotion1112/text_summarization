Quantization of a neural network has an inherent problem called accumulated quantization error, which is the key obstacle towards ultra-low precision, e.g., 2- or 3-bit precision.

To resolve this problem, we propose precision highway, which forms an end-to-end high-precision information flow while performing the ultra-low-precision computation.

First, we describe how the precision highway reduce the accumulated quantization error in both convolutional and recurrent neural networks.

We also provide the quantitative analysis of the benefit of precision highway and evaluate the overhead on the state-of-the-art hardware accelerator.

In the experiments, our proposed method outperforms the best existing quantization methods while offering 3-bit weight/activation quantization with no accuracy loss and 2-bit quantization with a 2.45 % top-1 accuracy loss in ResNet-50.

We also report that the proposed method significantly outperforms the existing method in the 2-bit quantization of an LSTM for language modeling.

Energy-efficient inference of neural networks is becoming increasingly important in both servers and mobile devices (e.g., smartphones, AR/VR devices, and drones).

Recently, there have been active studies on ultra-low-precision inference using 1 to 4 bits BID21 BID9 BID27 BID25 BID28 BID2 BID15 BID1 and their implementations on CPU and GPU BID23 , and dedicated hardware BID19 BID22 .

However, as will be explained in section 5.2, the existing quantization methods suffer from a problem called accumulated quantization error where large quantization errors get accumulated across layers, making it difficult to enable ultra-low precision in deep neural networks.

In order to address this problem, we propose a novel concept called precision highway where an end-to-end path of high-precision information reduces the accumulated quantization error thereby enabling ultra-low-precision computation.

Our proposed work is similar to recent studies BID15 BID2 which propose utilizing pre-activation residual networks, where skip connections are kept in full precision while the residual path performs low-precision computation.

Compared with these works, our proposed method offers a generalized concept of high-precision information flow, namely, precision highway, which can be applied to not only the pre-activation convolutional networks but also both the post-activation convolutional and recurrent neural networks.

Our contributions are as follows.• We propose a novel idea of network-level approach to quantization, called precision highway and quantitatively analyze its benefits in terms of the propagation of quantization errors and the difficulty of convergence in training based on the shape of loss surface.• We provide the detailed analysis of the energy and memory overhead of precision highway based on the state-of-the-art hardware accelerator model.

According to our experiments, the overhead is negligible while offering significant improvements in accuracy.• We apply precision highway to both convolution and recurrent networks.

We report a 3-bit quantization of ResNet-50 without accuracy loss and a 2-bit quantization with a very small accuracy loss.

We also provide the sub 4-bit quantization results of long short-term memory (LSTM) for language modeling.2 RELATED WORK BID16 presented an int8 quantization method that selects an activation truncation threshold to minimize the Kullback-Leibler divergence between the distributions of the original and quantized data.

BID12 proposed a quantization scheme that enables integer-arithmetic only matrix multiplications (practically, 8-bit quantization for neural networks).

These methods are implemented on existing CPUs or GPUs BID11 ACL) .

BID9 presented a binarization method and demonstrated the performance benefit on a GPU.

BID21 proposed a binary network called XNOR-Net in which a weight-binarized AlexNet gives the same accuracy as a full-precision one.

BID25 presented DoReFa-Net, which applies tanh-based weight quantization and bounded activation.

BID26 proposed a balanced quantization that attempts to balance the population of values on quantization levels.

BID7 proposed utilizing full precision for internal cell states in the LSTM because of their wide value distributions.

This work is similar to ours in that high-precision data are selectively utilized to improve the quantized network.

Our difference is proposing a network-level end-to-end flow of high-precision activation.

Recently, BID28 presented 4-bit quantization with ResNet-50.

They adopt Dorefa-net style weight quantization with static bounded activation, and improve accuracy by adopting multi-step quantization and knowledge distillation during fine-tuning.

proposed a trade-off between the number of channels and precision.

Clustering-based methods have the potential to further reduce the precision BID5 .

However, they require a lookup table and full-precision computation, which makes them less hardware-friendly.

Recently, BID2 a) and BID15 proposed utilizing full-precision on the skip connections in pre-activation residual networks 1 .

Compared with those works, our proposed idea has a salient difference in that it offers a network-level solution and demonstrates that the end-to-end flow of high-precision information is crucial.

In addition, our method is not limited to pre-activation residual networks, but general enough to be applied to both post-activation convolutional and recurrent neural networks.

In precision highway, we build a path from the input to output of a network to enable the end-toend flow of high-precision activation, while performing low-precision computation.

Our proposed method was motivated (1) by a residual network where the signal, i.e., the activation/gradient in a forward/backward pass, can be directly propagated from one block to another BID6 and (2) by the LSTM, which provides an uninterrupted gradient flow across time steps via the inter-cell state path BID4 .

Our proposed method focuses instead on improving the accuracy of quantized network by providing an end-to-end high-precision information flow.

In this section, we first describe the precision highway in the cases of residual network (section 3.1) and recurrent neural network (section 3.2).

Then, we discuss practical issues to be addressed before application to other networks in section 3.3.

In the case of a residual network, we can form a precision highway by making high-precision skip connections.

In this subsection, we explain how high-precision skip connections can be constructed to reduce the accumulated quantization error.

Conv BN [ , ] + ReLU Conv BN [ , ] + ReLU [ , ] + ReLU DISPLAYFORM0 ReLU [ , ] + ReLU BN ReLU [ , ] + ReLU + (c) [ , ] + ReLU + Figure , k-bit linear quantization in range from 0 to 1) is applied to all of the activations after the activation function.

In the figure, thick (thin) arrows represent high-precision (low-precision) activations.

As the figure shows, the input of a residual block is first quantized, and the quantized input (x + e in the figure), which contains the quantization error e, enters both the skip connection and residual path.

The output of a residual block, y, is calculated as follows:

where F () represents a residual function (typically, 2 or 3 consecutive convolutional layers).

For simplicity of explanation, we assume that F (x + e) can be decomposed into F (x) + e r , where e r represents the resulting quantization error of the residual path incurred by the quantization operations on the residual path as well as the quantization error in the input, e. As the equation shows, output y has two quantization error terms, that of residual path, e r , and that of the skip connection, e.

Figure 1 (b) shows our idea of high-precision skip connection.

Compared with Figure 1 (a), the difference is the location of the first quantization operation in the residual block.

In Figure 1 (b), quantization is applied only to the residual path after the bifurcation to the residual path and skip connection.

As shown in the figure, the skip connection now becomes a thick arrow, i.e., a high-precision path.

The proposed idea gives the output of the residual block as follows: DISPLAYFORM0 As Equation 2 shows, the proposed idea eliminates the quantization error of skip connection e.

Thus, only the quantization error of the residual path e r remains in the output of the residual block.

Note that all of the input activations of the residual path are kept in low precision.

It enables us to perform low-precision convolution operations in the residual path.

We keep high-precision activation only on the skip connection and utilize it only for the element-wise addition.

As will be shown in our experiments, the overhead of computation and memory access cost is small since the element-wise addition is much less expensive than the convolution on the residual path, and the low-precision activation is accessed for the computation on the residual path.

As will be shown later, our method gives a smaller quantization error, and the gap between the quantization error of the existing method and that of ours becomes wider across layers.

Because of the reduction of the accumulated quantization error, the proposed method offers much better accuracy than the state-of-the-art methods with an ultra-low precision of 2 and 3 bits.

Note also that, as shown in figure 1 (c), our idea can be applied to other types of residual blocks, including the full pre-activation residual block BID6 as proposed in some recent works BID1 BID15 .

However, our idea is general in that it is applicable to recurrent networks as well as post-activation convolutional networks.

Especially, our proposed idea is advantageous over the existing ones since hardware accelerators tend to be designed assuming as the input non-negative input activations enabled by ReLU activation functions BID19 .

Contrary to the existing works BID1 BID15 , we provide a detailed analysis of the effect of precision highway.

FIG1 illustrates how the precision highway can be constructed on the LSTM BID4 .

In time step t, the LSTM cell takes, as an input, new input x t , along with the results of the previous time step, output h t−1 and cell state c t−1 .

First, it calculates four intermediate signals: i (input gate), f (forget gate), g (gate gate), and o (output gate).

Then, it produces two results, c t and h t , as follows:

where σ represents a sigmoid function, the element-wise multiplication, W the weight matrix, and b the bias.

In the conventional LSTM operation, as FIG1 shows, the quantization (gray box denoted by Q k with the output value range as the superscript) is applied to all of the activations before computation.

The results of a time step, c t and h t , are calculated based on such inputs with quantization errors.

More specifically, cell state c t is calculated with the quantized, i.e., low-precision, inputs of c t−1 , f , i, and g. Thus, cell state c t accumulates the quantization errors of those inputs.

In addition, output h t also accumulates the quantization errors from its inputs, c t and o. Then, they are propagated to the next time steps.

Thus, we have the problem of accumulated quantization error across the time steps.

Such an accumulation of quantization error will prevent us from achieving ultra-low precision.

Figure 2 (b) shows how we can build the precision highway in the LSTM cell.

The figure shows that the quantization operation is applied only to the inputs of matrix multiplication (a circle denoted with × in the figure) .

Thus, all of the other operations and their input activations are in high precision.

Specifically, when calculating c t , the inputs are not quantized, which reduces the accumulation of quantization error on c t .

The computation of h t can also reduce the accumulation of quantization error by utilizing high-precision inputs.

The construction of such a precision highway allows us to propagate high-precision information, i.e., cell states c t and outputs h t , across time steps.

Note that we benefit from low-precision computation by performing low-precision matrix multiplications (in Equations 3a-3d), which dominate the total computation cost.

In our proposed method, all of the element-wise multiplications in Equations 3e and 3f are performed in high precision.

However, the overhead of this high-precision element-wise multiplications is negligible compared with the matrix multiplication in Equations 3a-3d.

In addition, this method can be applied to other types of recurrent neural networks.

For instance, the GRU BID3 can be equipped with a precision highway, in a way similar to that shown in FIG1 , by keeping high-precision output h t while performing low-precision matrix multiplications and high-precision element-wise multiplications.

In order to generalize our proposed idea to other networks in real applications, we need to address the following issues.

First, in the case of feed-forward networks with identity path, our precision highway idea is applicable regardless of pre-activation or post-activation structure.

We can exploit the benefit of reduced precision by applying quantization in front of matrix multiplications, while maintain the accuracy by handing the identity path in high precision.

Second, in the case of non-residual feed-forward networks, the precision highway can be constructed by equipping them with additional skip connections.

In the case of networks with multiple candidates for the precision highway, e.g., DenseNet, which has multiple parallel skip connections BID8 , we need to address a new problem of selecting skip connections to form a precision highway, which is left for future work.

In this section, we describe weight quantization and fine-tuning for weight/activation quantization.

Figure 3 illustrates that a Laplace distribution can well fit the distributions of weights in full-precision trained networks.

Thus, we propose modeling the weight distribution with Laplace distribution and selecting quantization levels for weights based on a Laplace distribution.

Given a distribution of weights and a target precision of k bits, e.g., 2 bits, the quantization levels are determined as follows.

First, the quantization levels for k bits are pre-computed for the normalized Laplace distribution.

We determine quantization levels that minimize L2 error on the normalized Laplace distribution.

For instance, in case of the 2-bit quantization, the error is minimized when four quantization levels are placed evenly with a spacing of 1.53 µ, where µ is the mean of the absolute value of weights.

Given the distribution of weights and the pre-calculated quantization levels on the normalized Laplace distribution for the given k bits, we determine the real quantization levels by multiplying the pre-computed quantization levels and the mean of the absolute value of weights.

Our proposed weight quantization is similar to the one in BID1 .

Compared to it, ours is simpler in that only Laplace distribution model is utilized, and our experiments show that the precision highway together with the proposed simple weight quantization gives outstanding results.

Our quantization is applied during fine-tuning after training a full-precision network.

As the baseline, we adopt the fine-tuning procedure in BID28 , where we perform incremental/progressive quantization.

In contrast to BID28 , we first quantize activations and then weights in an incremental quantization.

In addition, for each precision configuration, we perform teacher-student training to improve the quantized network BID28 .As the teacher network, we utilize a deeper full-precision network, e.g., ResNet-101, compared to the student network, e.g., quantized ResNet-50.

Note that, during fine-tuning, we apply quantization in forward pass while updating full-precision weights during backward pass.

We implemented the proposed method in PyTorch and Caffe2.

We use two types of trained neural networks, ResNet-18/50 for ImageNet and an LSTM for language modeling BID24 BID20 BID10 .

We evaluate 4-, 3-, and 2-bit quantizations for the networks.

For ResNet, we did test with single center crop of 256x256 resized image.

We compare our proposed method with the state-of-the-art methods BID25 BID28 BID2 a; BID15 .

Note that, for the teacher-student training, we use the same teacher network for both the baseline method (our implementation) BID28 and ours.

We also evaluate the effects of increasing the number of channels to recover from accuracy loss due to quantization.

As in the previous works BID9 BID26 BID28 BID2 a; BID15 , we do not apply quantization to the first and last layers.

The LSTM has 2 layers and 300 cells on each layer.

We used the Penn Treebank dataset and evaluated the perplexity per word.

We compared the state-of-the-art method in BID7 and our proposed method.

Figure 4 shows the quantization errors across layers in ResNet-50 when applying the state-of-the-art 4-bit quantization to activations.

We prepared, from the same initial condition, two activation-quantized networks (one with precision highway and the other with low precision skip connection) where weights are not modified and only activations are quantized to 4 bits.

As the metric of the quantization error, we utilize a metric based on the cosine similarity between the activation tensor of corresponding layer in the full-precision and quantized networks, respectively.

As the figure shows, in the existing method, the quantization errors become larger for deeper layers.

It is because the quantization error generated in each layer is propagated and accumulated across layers.

We call this accumulated quantization error.

The accumulated errors become larger with more aggressive quantization, e.g., 2 bits, and cause poor performance, i.e., 4.8 % drop BID28 ) from the top-1 accuracy of the full-precision ResNet-50 for ImageNet classification.

The accumulation of quantization errors is an inherent characteristic of a quantized network in both feed-forward and feed-back networks.

In the case of a recurrent neural network, the quantization errors are propagated across time steps.

As shown in Figure 4 , our proposed precision highway significantly reduces the accumulated quantization errors, which enables 3-bit quantization without accuracy drop and much better accuracy in 2-bit quantization than the existing methods.

Figure 5 visualizes the complexity of loss surface depending on the existence of precision highway.

We obtained the figures by applying the method proposed by Li et al. BID14 .

Each figure represents loss surface seen from the local minimum we obtained from the training, i.e., the weight vector of the final trained model.

The origin of the figure at (0, 0) corresponds to the weight vector of the local minimum.

As shown in the figure 5 (d), the precision highway gives better loss surface (having lower and smoother surface near the minimum point and steep and simple surface elsewhere) than the existing quantization method.

This characteristic helps stochastic gradient descent (SGD) method to quickly converge to a good local minimum offering better accuracy than the existing method.

BID25 7.6 9.8 Zhuang's BID28 -4.8 PACT BID2 5.8 4.7 PACT new BID1 3.4 2.7 Bi-Real FIG1 12.9 - TAB0 shows the accuracy of 2-bit quantization for ResNet-18/50.

We evaluate each of our proposed methods, Laplace, teacher, and highway, as shown in the table.

When the highway box is unchecked the skip connection is branched after the quantization and when the teacher box is unchecked, we use the conventional cross-entropy loss.

Compared with the full-precision accuracy, our 2-bit quantization (when all the methods were applied) gives a top-1 accuracy of 73.55 %, which is within 2.45 % of the full-precision accuracy and much better than the state-of-the-art method (Zhuang's 70.8 %) having a top-1 accuracy loss of 4.8 %.

Note that Zhuang's implemented all the methods, incremental/progressive quantization and teacher-student training, in BID28 .

We presents the accuracy results of our own implementations of Zhuang's method under the same amount of training time.

Zhuang's (ours) implemented only incremental and progressive methods while Zhuang's + Teacher utilized our teacher network.

Zhuang's, and PACT) .

PACT new and Bi-Real utilize high-precision skip connections on pre-activation resiudal networks.

Thus, they show comparable results to ours 2 .

Note that our results in the table are obtained from the conventional post-activation residual network, which demonstrates the generality of our proposed precision highway.

As will be shown below for the LSTM, our proposed method is generally applied to recurrent networks as well as feed forward ones.

TAB4 shows the impact of the precision of the precision highway.

We obtained the results by varying the highway precision (without retraining) after obtaining the results with the full-precision highway.

The table shows that 2-bit quantization with the 8-bit highway gives only 0.09 % and 0.40 % drops in the top-1 accuracy for ResNet-18 and ResNet-50, respectively, from that of the 2-bit quantization with the full-precision highway.

Most importantly, our 3-bit quantization (with the 8-bit highway) gives the same accuracy as the full-precision network, i.e., 76.08 % in ResNet-50, which means that our proposed method reduces the precision of the ResNet-50 from 4 bits with BID28 down to 3 bits even with the 8-bit highway.

TAB5 shows the effects of two times wider channel under 2-bit quantization.

We first doubled the number of channels in ResNet-18 and ResNet-50, and then quantized them with our methods.

As the table shows, the wide ResNets give better accuracy than the full-precision ones even for 2-bit quantization, i.e., 73.80 % (77.35 %) in TAB5 vs. 70.15 % (76.00 %) of the full precision in TAB0 for .

It would be worth investigating how to minimize the channel size while meeting the full-precision accuracy with ultra-low precision, which is left for future work.

BID7 ) in perplexity for 2-bit quantization.

FIG4 shows the chip area cost and energy consumption of ResNet-18 at different levels of precision on the state-of-the-art hardware accelerator BID0 .

The accelerator is synthesized at 65 nm, 250 MHz, and 1.0 V. Each processing element (PE) consists of a multiply-accumulate (MAC) unit and local buffers.

The PEs share global on-chip 2 MB static random access memory (SRAM) at 16-bit precision and the size of which is adjusted proportional to the precision.

As the figure shows, the reduced precision offers significant reduction in chip area, e.g., 82.3 % reduction from 16 to 3 bits and energy consumption, e.g., 73.1 % from 16 to 3 bits.

In the 2-bit case where the overhead of precision highway is the largest, the precision highway incurs only 3.9 % additional energy consumption due to the high-precision data while offering 4.1 % better accuracy than the case that precision highway is not adopted.

The accelerator is already equipped large internal buffer for partial sum accumulation.

Thus, precision highway incurs additional energy consumption mainly on the accesses to on-chip SRAM and main memory (dynamic random access memory, DRAM).

TAB8 compares the number of operations in three neural networks used in our experiments.

The table explains why the high-precision operations incur such a small overhead in energy consumption.

As the table shows, it is because the frequency of high-precision operations is much smaller than that of low-precision operations.

For instance, the 2-bit LSTM network has one high-precision (in 32 bits) element-wise multiplication for every 800 2-bit multiplications.

In this paper, we proposed the concept of end-to-end precision highway which can be applied to both feedforward and feedback networks and enable ultra-low precision in deep neural networks.

The proposed precision highway reduces quantization errors by keeping high-precision activation from the input to output of the network with small computation costs.

We described how it reduces the accumulated quantization error and presented quantitative analyses in terms of accuracy and hardware cost as well as training characteristics.

Our experiments showed that the proposed method outperforms the state-of-the-art methods in the 3-and 2-bit quantizations of ResNet-18/50 and 2-bit quantization of an LSTM model.

We believe that our work will serve as a step toward mixed precision networks for computational efficiency.

@highlight

precision highway; a generalized concept of high-precision information flow for sub 4-bit quantization 

@highlight

Investigates the problem of neural network quantization by employing an end-to-end precision highway to reduce the accumulated quantization error and enable ultra-low precision in deep neural networks. 

@highlight

This paper studies methods to improve the performance of quantized neural networks

@highlight

This paper proposes to keep a high activation/gradient flow in two kinds of networks structures, ResNet and LSTM.