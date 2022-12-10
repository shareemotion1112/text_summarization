Most deep neural networks (DNNs) require complex models to achieve high performance.

Parameter quantization is widely used for reducing the implementation complexities.

Previous studies on quantization were mostly based on extensive simulation using training data.

We choose a different approach and attempt to measure the per-parameter capacity of DNN models and interpret the results to obtain insights on optimum quantization of parameters.

This research uses artificially generated data and generic forms of fully connected DNNs, convolutional neural networks, and recurrent neural networks.

We conduct memorization and classification tests to study the effects of the number and precision of the parameters on the performance.

The model and the per-parameter capacities are assessed by measuring the mutual information between the input and the classified output.

We also extend the memorization capacity measurement results to image classification and language modeling tasks.

To get insight for parameter quantization when performing real tasks, the training and test performances are compared.

Deep neural networks (DNNs) have achieved impressive performance on various machine learning tasks.

Several DNN architectures are known, and the most famous ones are fully connected DNNs (FCDNNs), convolutional neural networks (CNNs), and recurrent neural networks (RNNs).It is known that neural networks do not need full floating-point precision for inference BID10 BID16 BID23 .

A 32-bit floating-point parameter can be reduced to 8-bit, 4-bit, 2-bit, or 1-bit, but this can incur performance degradation.

Therefore, precision should be optimized, which is primarily conducted by extensive computer simulations using training data.

This not only takes much time for optimization but also can incorrectly predict the performance in real environments when the characteristics of input data are different from the training data.

In this study, we attempt to measure the capacity of DNNs, including FCDNN, CNN, and RNN, using a memorization and classification task that applies random binary input data.

The per-parameter capacities of various models are estimated by measuring the mutual information between the input data and the classification output.

Then, the fixed-point performances of the models are measured to determine the relation between the quantization sensitivity and the per-parameter capacity.

The memorization capacity analysis results are extended to real models for performing image classification and language modeling, by which the parameter quantization sensitivity is compared between memorization and generalization tasks.

The contributions of this paper are as follows.• We experimentally measure the memorization capacity of DNNs and estimate the perparameter capacity.

The capacity per parameter is between 2.3 bits to 3.7 bits, according to the network structure, which is FCDNN, CNN, or RNN.

The value is fairly independent of the model size.• We show that the performance of the quantized networks is closely related to the capacity per parameter, and FCDNNs show the most resilient quantization performance while RNNs suffer most from parameter quantization.

The network size hardly effects the quantization performance when DNN models are trained to use full capacity.•

We explain that severe quantization, such as binary or ternary weights, can be employed without much performance degradation when the networks are in the over-parameter region.• We suggest the sufficient number of bits for representing weights of neural networks, which are approximately 6 bits, 8 bits, and 10 bits for FCDNNs, CNNs, and RNNs, respectively.

This estimate of the number of bits for implementing neural networks is very important considering that many accelerators are designed without any specific training data or applications.• The study with real-models shows that neural networks are more resilient to quantization when performing generalization tasks than conducting memorization.

Thus, the optimum bits obtained with the memorization tasks are conservative and safe estimate when solving real problems.

The paper is organized as follows.

In Section 2, previous works on neural network capacity and fixedpoint optimization are briefly presented.

Section 3 explains the capacity measurement methods for DNN models.

Section 4 presents parameter capacity measurement results for FCDNNs, CNNs, and RNNs.

The quantization performances measured on DNNs are presented in Section 5.

Concluding remarks follow in Section 6.

The capacity of neural networks has been studied since the early days of DNN research.

Although the capacity can be defined in many ways, it is related to the learnability of networks.

The capacity of networks is shown as the number of uncorrelated random samples that can be memorized BID8 .

A single-layer perceptron with n parameters can memorize at least 2n random samples BID11 .

In other words, the network can always construct a hyperplane with n parameters that divides 2n samples.

Additionally, the capacity of a three-layer perceptron is proportional to the number of parameters BID0 .

Recently, RNNs were trained with random data to measure the capacity per parameter BID6 .

Our study is strongly motivated by this research, and extends it to the quantization performance interpretation of generic DNN models, including FCDNN, CNN, and RNN.

Recent studies have showed that neural networks have a generalization ability even if the expressive capacity of the model is sufficiently large BID33 BID2 .

In this paper, we also discuss the effect of network quantization when performing generalization tasks.

Early works on neural network quantization usually employed 16-bit parameters obtained by directly quantizing the floating-point numbers BID10 .

Recently, a retraining technique was developed to improve the performance of quantized networks BID16 BID23 .

Retraining-based quantization was applied to CNN and RNN models, showing superior performance compared to directly quantized ones BID1 BID31 .

Many studies attempting extreme quantization have been published, such as 2-bit ternary BID16 BID21 BID35 , 1-bit binary weight quantization, and XNOR networks BID7 BID28 .

Some aggressive model compression techniques also employed vector quantization or table look-up BID12 BID4 .

However, not all CNNs show the same quantization performance.

For example, AlexNet BID20 shows almost the same performance with only 1-bit quantized parameters.

However, the same quantization technique incurs a very severe performance loss when applied to ResNet BID28 .

A previous study shows that large sized networks are more resilient to severe quantization than smaller ones .

Theoretical works and many practical implementation optimization techniques have been studied BID13 BID17 BID19 BID30 BID24 .

Recent work increases the number of network parameters to preserve the performance under low-precision quantization BID26 .

Our works are not targeted to a specific data or model, but introduce the general understanding of parameter quantization.

We assess the network capacity of DNN models using a random data memorization and classification task BID6 .

In this task, N random binary vectors, X, are generated and each is randomly and uniformly assigned to the output label Y .

The size of the binary vector depends on the DNN model.

For FCDNN, the input X is a one dimensional vector whose size is determined by the hidden layer dimension.

In CNN, the input needs to be a 2-D or 3-D tensor.

Input samples of CNNs are generated by concatenating and reshaping random binary vectors.

During the training process, the DNN is trained to correctly predict the label, which is 0 or 1, of the random input X. As the number of input data size, N , increases, the classification accuracy drops because of the limited memorization capacity.

Note that the accuracy for the memorization task refers to the training performance after convergence because there is no proper test dataset for random training samples.

The capacity is measured using the mutual information, defined as a measure of the amount of information that one random variable contains about another random variable (Cover & BID9 .

The mutual information of a trained network with N input samples is calculated as follows: DISPLAYFORM0 where p is the mean classification accuracy for all samples under trained parameter θ.

The network capacity is defined as DISPLAYFORM1 The accuracy, p, may vary depending on the training method of the model.

We find N and p that maximize the mutual information of the networks by iteratively training the models.

This optimization employs both grid search-and Bayesian optimization-based hyper-parameter tuning BID5 .

The optimization procedure consists of three stages.

First, we try to find the largest input data size whose accuracy is slightly lower than 1.

Second, we perform a grid search to determine the boundary values of the hyper-parameters.

The searched hyper-parameters can include initialization, optimizer, initial learning rate, learning rate decay factor, batch size, and optimizer variables.

Finally, we conduct hyper-parameter tuning within the search space using Scikit-learn library BID27 .

We add the number of training samples N as a hyper-parameter and use the mutual information of Eq. FORMULA0 as the metric for the optimization.

Quantization of model parameters perturbs the trained network, therefore, fixed-point training or retraining with full-precision backpropagation is usually needed BID16 BID21 BID7 BID34 .

However, the performance of the quantized networks does not always meet that of the floating-point models, even after retraining.

This suggests that model capacity is reduced by quantization, especially when the number of bits used is very small.

In this research, we observe the memorization capacity degradation caused by quantization in generic FCDNN, CNN, and RNN models.

The uniform quantization is used for the sake of convenient arithmetic, and the same step size is assigned to each layer in the FCDNN, each kernel in the CNN, or each weight matrix in the LSTM layer.

The bias values are not quantized, because they have a large dynamic range.

It is important to note that the weights connected to the output are not quantized, because their optimum bit-widths depend on the number of labels in the output.

Quantization is performed from floating-point to 8-bit, 6-bit, 5-bit, 4-bit, 3-bit, and 2-bit precision, in sequence.

Retraining is performed after every quantization, but requires only a small number of epochs, because only fine-tuning is needed BID16 .

We compare the generalization performance of floating-point and fixed-point DNNs by visualizing the loss surface.

Loss is measured by applying Gaussian random noise to the parameters of the trained network as shown in Eq. (3).

DISPLAYFORM0 Here, L(θ) is the loss according to the network parameters.

The distribution of weights may vary depending on the model size and learning method.

We apply the normalized filter noise to the θ noise for fair comparison on different models BID22 .We employ two real networks, one is for image classification with CIFAR-10 dataset and the other is language modeling with Penn Tree Bank (PTB) dataset.

One large and one small model are trained for these networks.

We quantize those networks with the precision of 8, 6, 4, and 2 bits and analyze the variation of the surface according to the precision.

θ noise is added to the quantized parameters.

In order to reduce the error due to randomness of noise, all loss values are measured with 10 different trials and the average values are plotted.

The capacities of FCDNNs, CNNs, and RNNs are measured via the memorization task explained in Section 3.1.

The models used for the test employ floating-point parameters.

The training data for FCDNNs is a 1-D vector of size n in .

N input data are used as for the training data.

The output, Y , is the randomly assigned label, either 0 or 1, for each input.

Thus, inputs, X and Y , are represented as X ∈ {0, 1} N ×nin and Y ∈ {0, 1} N , respectively.

The input data dimension, n in , should be larger than log 2 N so that no overlapped data is contained among N input data.

In the experiments for FCDNNs, the input vector dimension, n in , is chosen to be equal to the number of units in the hidden layer.

We conduct experiments for FCDNNs with hidden layer dimensions of 32, 64, 128, and 256, and with hidden layer depths of 1, 2, 3, and 4.

The initialization method chosen is the 'He' initialization BID14 and gradients are updated following the rule in SGD, with momentum, which shows the best performance in our grid search.

The initial learning rate for hyper parameter tuning is chosen between 0.001 and 0.05 on the log scale.

The decay factor and momentum are set to have even distance values in the linear scale between 0.1 and 0.5 and between 0.6 and 0.99, respectively.

For each model, experiments are conducted to measure the accuracy of memorization while increasing the size of the input data, N .

Note that only the training error is measured in this memorization task, because there is no unseen data.

Experimental results are based upon the best accuracy obtained when attempted with different hyper parameters.

The capacity of the model is estimated according to Eq. (1), where p is the training accuracy.

The experimentally obtained memorization capacities of the FCDNN models are presented in FIG1 , where depths of 1, 2, 3, and 4, and widths of 32, 64, 128, and 256 are used.

When the number of hidden layers is the same, the amount of data that can be almost perfectly memorized quadruples when the dimension of the hidden layer is doubled.

This means that the memorization capacity is linearly proportional to the number of parameters.

Similarly, the FCDNN models with 2, 3, or 4 hidden layer depths can memorize 2, 3, or 4 times the input data as compared to the single layer DNN, respectively.

FIG2 shows the memorization accuracy and the mutual information obtained using Eq. (1) on the FCDNN.

The model is composed of three layers and the hidden layer of size 64.

Here, we find that the amount of mutual information steadily increases as the input data size grows.

However, it begins to drop as the input size grows farther, and the memorization accuracy drops.

By analyzing the accuracy trend of the model, it is possible to distinguish the input data size into three regions: the over-parameterized, the maximum performance, and the under-parameterized sections, as shown in FIG2 .

For example, if the model is trained to memorize only 10,000 data, it can be regarded as over-parameterized.

The number of data that can be memorized by maximally utilizing all the parameters is between 30,000 and 40,000.

In over-parameterized regions, performance can be maintained, even if the capacity of the networks is reduced.

The per-parameter capacity of FCDNNs is shown in FIG2 .

Regardless of the width or depth, one parameter has a capacity of 1.7 to 2.5 bits, and FCDNNs have an average of 2.3-bit capacity per parameter.

This result is consistent with theoretical study BID11 BID0 .

The total capacity of the model may be interpreted as an optimal storage that can store a maximum of random binary samples BID11 BID3 .

The capacity of CNNs is also measured via a similar memorization task.

CNNs can have a variety of structures according to the number of channels, the size of the kernels, and the number of layers.

The kernel size of CNNs in this test are either (3 × 3) or (5 × 5), which are the same for all layers, the number of convolution layers from 3 to 9.

The dimensions of the inputs are n height = n width = 32 and n channel = 1 for all experiments.

Three max-pooling operations are applied to reduce the number of parameters in the fully connected layer.

CNN structures used in our experiments are shown in Supplementary materials.

The CNN models contain not only convolution layers but also fully connected layers.

Thus, the per-parameter capacity for convolution layers is calculated after subtracting the capacity for fully connected layers from the measured total capacity.

We assume the per-parameter capacity of the fully connected layer as 2.3 bits to calculate the capacity for convolution layers.

As shown in FIG2 , the convolution layers have the per-parameter capacity of between 2.86 and 3.09 except the smallest model, which is higher than that of FCDNNs.

The average capacity per parameter of the tested models is 3.0 bits.

Results show that the per-parameter capacity of CNNs is higher than that of FCDNNs, even when CNNs memorize uncorrelated data.

Note that one parameter of FCDNNs is used only once for each inference.

However, the parameter of CNNs is used multiple times.

This parameter-sharing nature of CNNs seems to increase the amount of information that one parameter can store.

It has been shown that the various structures of RNNs all have similar capacity per parameter of 3 to 6 bits BID6 .

We train RNNs with a dataset with no sequence correlation to show the capacity of the parameters.

The random input dataset is composed of inputs, X ∈ {0, 1} N ×nseq×nin and labels Y ∈ {0, 1} N , which are uniformly set to 0 or 1.

The training loss is calculated using the cross-entropy of the label at the output of the last step.

We train RNNs with a single LSTM layer of 32-D. The input dimension, n in , is also 32-D and the amount of unrolling sequence, n seq , is five-step.

It has been reported that unrolling of five-step almost saturates the performance in this setup BID6 .

We apply 5 input random vectors, X 0 , X 1 , X 2 , X 3 , and X 4 , each with 32-D, and assign one label to this 160-D vector at the last time step.

The error propagates from the last step only, and the outputs at intermediate time-steps are ignored.

The number of parameters in the network is 8,386.

In this case, the maximum mutual information is obtained when the number of samples is 32K, and the memorization accuracy is 99.52 %.

Therefore, the per-parameter capacity of the model is 3.7 bits.

The RNN shows higher per-parameter capacity than FCDNNs and CNNs.

We have shown that FCDNNs, CNNs, and RNNs have different per-parameter capacities.

According to the parameter-data ratio, a trained DNN can be an over-parameterized, max-capacity, or underparameterized model.

Thus, we can assume that the DNN performance under quantization would depend on not only the network structure, such as FCDNN, CNN, or RNN, but also the parameter-data ratio.

The experiments are divided into two cases.

The first is to measure performance degradation via quantization precision when each model is in the maximum capacity region.

The second analyzes performance when the models are in the over-parameterized region.

When the FCDNN, CNN, and RNN are trained to have the maximum memorization capacity, the performances with parameter quantization are shown in Fig. 3(a) .

The FCDNN, CNN, and RNN models are shown.

The fixed-point performances of two FCDNNs, two CNNs, and two RNNs are illustrated.

With 6-bit parameter quantization, the FCDNN shows no accuracy drop.

However, those for CNNs and RNNs are 5 % and 18 %, respectively.

Because the RNN contains the largest amount of information at each parameter, the loss caused by parameter quantization seems to be the most severe.

We also find that there is no decline in performance until the parameter precision is lowered to 6-bit for FCDNNs, 8-bit for CNNs, and 10-bit for RNNs, even when all models use full capacity.

Next, we show the fixed-point performance of DNNs when they are trained to be in the overparameterized region.

Note that the per-parameter capacity is lowered in the over-parameterized region.

We conducted simulation with half size of the maximum number of data that can be memorized.

For example, an FCDNN used for the measurement has 3 hidden layers with a hiddenlayer dimension of 128; the capacity of the corresponding model is about 2 17 bits.

The network is FORMULA6 FC FORMULA0 FC FORMULA1 FC (64) FC FORMULA0 FC FORMULA1 (a) (64) LSTM FORMULA0 LSTM FORMULA1 LSTM FORMULA0 (c) over-parameterized when the number of memorized samples is 2 16 .

Fig. 3(b) shows that the FCDNN model memorizes all samples even with 4-bit parameter quantization when the model uses half of the capacity.

Also, over-parmeterized model is less sensitive to bit-precision on CNNs and RNNs.

The performances of fixed-point DNNs with the number of samples are shown in Fig. 4 .

The result shows that DNNs are more robust when the networks are more over-parameterized.2

We have assessed the required precision of networks for performing memorization tasks.

The memorization test only uses the training data that are artificially generated.

However, most neural networks should conduct more than memorization because the test data are not seen during the training.

In this section, we analyze the effects of network quantization for performing real tasks.

We train two different sized CNN models with CIFAR-10 data.

The structures of the two models are as follows: DISPLAYFORM0 The size of kernels of both models is (3×3), 16C represents a convolution layer with 16 channels and 128F C means a fully connected layer with 128-dimension.

The number of parameters is 0.22M for the small model and 3.5M for the large one.

Both models were trained with the same hyper-parameter setting.

To analyze the impact of network quantization on the test performance, we plot the loss and accuracy surfaces of floating-point and quantized CNN models in Fig. 5 .

For simplicity, the results of floatingpoint and 2-bit fixed-point CNN are given.

Please refer to the Supplementary materials for other results.

When applying the training data that may have been memorized during the training phase, the large model shows indifferent performance surface regardless of the parameter precision.

But, for the small model, the 2-bit model shows quite degraded performance when compared to the floating-point network.

However, the test accuracy of small 2-bit model is not much lowered.

We can notice that the loss surface of the 2-bit model shown in Fig. 5 is much wider than that of the floating-point model.

This result is consistent with recent studies on generalization BID18 BID22 .

This observation suggests that the quantized networks are more resilient when performing generalization tasks.

Thus, the required precision of the network obtained with the memorization task can be considered a conservative estimate.

Fig. 6 shows the training and test data based performance of fixed-point CNN and RNN on real data.

T iny model has the same structure as the small model, but reducing the number of channels by half and the size of the fully connected layers by a quarter.

The RNNs are trained for language modeling with Penn Treebank corpus BID25 , and the models consists of two LSTM layers with the same dimension.

Here, both for CNN and RNN models, we can confirm that large networks are more robust to quantization.

Also, the networks need more parameter precision when conducting memorization tasks using the training set, rather than solving real problems using the test set.

We have measured fixed-point DNN performance on real tasks and results are shown in FIG7 .

FCDNN models are trained with MNIST dataset, ResNet BID15 models are trained using ILSVRC-2012 dataset BID29 and RNN based word-level language models (WLMs) are designed using PTB dataset.

Experimented FCDNNs and RNNs are composed of two FC layers and two LSTM layers with the same dimension, respectively.

4-bit quantized FCDNNs shows almost same performance compared to the floating-point networks even when the number of neurons are only 8.

Performances are preserved up to 6 bits on ResNets and RNN WLMs.

Their resiliency to quantization increases as networks become larger.

Quantization of parameters is a straightforward way of reducing the complexity of DNN implementations, especially when VLSI or special purpose neural processing engines are used.

Our study employed simulations on varying sizes of generic forms of DNN models.

Memorization tests using random binary input data were conducted to determine the total capacity by measuring the mutual information.

Our simulation results show that the per-parameter capacity is not sensitive to the model size, but is dependent on the structure of the network models, such as FCDNN, CNN, and RNN.

The maximum per-parameter capacities of FCDNNs, CNNs, and RNNs are approximately 2.3 bits, 3.0 bits, and 3.7 bits per parameter.

Thus, RNNs have the tendency of demanding more bits when compared to FCDNNs.

We quantized DNNs under various capacity-utilization regions and showed that the capacity of parameters are preserved up to 6 bits, 8 bits, and 10 bits on FCDNNs, CNN, and RNNs, respectively.

The performance of the quantized networks was also tested with image classification and language modeling tasks.

The results show that networks need more parameter precision when conducting memorization tasks, rather than inferencing with unseen data.

Thus, the precision obtained through the memorization test can be considered a conservative estimate in implementing neural networks for solving real problems.

This research not only gives valuable insights on the capacity of parameters but also provides practical strategies for training and optimizing DNN models.

DISPLAYFORM0 In FCDNNs, each weight matrix, W l between two layers, demands |h l−1 | × |h l | weights, where |h| is the number of units for the layer, l.

CNNs, which are popular for image processing, usually receive 2-dimensional (D) or 3-D input data, whose size is much larger than the filter or kernel size.

The set of weights between layers is referred to as the 'kernel' and the output is referred to as the 'feature map'.

Because the input size is usually much larger than the kernel size, the CNN parameters are reused many times.

A kernel slides over the input feature map and produces an output feature map, and the sliding step is determined by the stride, s.

The convolution weights is denoted as W l ∈ R k l,h ×k l,w ×n l−1 ×n l and feature map of the layer as C l ∈ R c l,h ×c l,w ×n l .

k l,h and k l,w are height and width of each kernel and c l,h , c l,w are height and width of the feature map in layer l, respectively.

n l is the number of feature map in the layer l.

CNNs can have a variety of structures according to the number of channels, the size of the kernels, and the number of layers.

We attempted to produce a general setting for CNNs.

The experiments models are shown in Table.

1.

We constructed CNNs to have twice the number of feature maps and half the height/width after pooling.

Also, to minimize side-effects by fully connected layers, the output feature of the last convolution layer is flattened and directly propagated to the sof tmax layer.

The capacity per parameter is measured only for parameters in convolution layers, by subtracting capacity of the fully connected layer.

RNNs have a feedback structure that reflects the information in the previous steps when processing sequence data.

RNNs are composed of one or multiple recurrent layers, and each layer computes the output, y t , and the hidden state, h t , using the previous hidden state, h t−1 , and the input, x t .We use LSTM as the recurrent layers, showing stable performance in various applications.

The mutual information equation of a network can be obtained as follows.

We first re-write our random variables X, Y , andŶ .

DISPLAYFORM0 DISPLAYFORM1 where f (θ, X i ) is the predict of a network when the input is X i .

Under our experimental setting, both X and Y have uniform random distribution.

Note that X and Y are independent as well as Y i and Y j when i = j. Therefore, DISPLAYFORM2 And we use the network's average accuracy p as a probability of Y i =Ŷ i , so that DISPLAYFORM3 Finally, the equation is derived as: DISPLAYFORM4

The loss surfaces of fixed-point CNNs are shown in Fig. 8 and Fig. 9 .

The models are trained with CIFAR-10 dataset.

The loss surfaces of RNN LMs for PTB dataset are also shown in FIG1 and FIG1 .

The performance degradation on test dataset is lower than the degradation on training dataset in all experiments.

<|TLDR|>

@highlight

We suggest the sufficient number of bits for representing weights of DNNs and the optimum bits are conservative when solving real problems.