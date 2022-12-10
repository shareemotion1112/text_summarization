Deep neural networks (DNNs) continue to make significant advances, solving tasks from image classification to translation or reinforcement learning.

One aspect of the field receiving considerable attention is efficiently executing deep models in resource-constrained environments, such as mobile or embedded devices.

This paper focuses on this problem, and proposes two new compression methods, which jointly leverage weight quantization and distillation of larger teacher networks into smaller student networks.

The first method we propose is called quantized distillation and leverages distillation during the training process, by incorporating distillation loss, expressed with respect to the teacher, into the training of a student network whose weights are quantized to a limited set of levels.

The second method,  differentiable quantization, optimizes the location of quantization points through stochastic gradient descent, to better fit the behavior of the teacher model.

We validate both methods through experiments on convolutional and recurrent architectures.

We show that quantized shallow students can reach similar accuracy levels to full-precision teacher models, while providing order of magnitude compression, and inference speedup that is linear in the depth reduction.

In sum, our results enable DNNs for resource-constrained environments to leverage architecture and accuracy advances developed on more powerful devices.

Background.

Neural networks are extremely effective for solving several real world problems, like image classification BID20 BID10 , translation BID33 , voice synthesis BID27 or reinforcement learning BID26 BID30 .

At the same time, modern neural network architectures are often compute, space and power hungry, typically requiring powerful GPUs to train and evaluate.

The debate is still ongoing on whether large models are necessary for good accuracy.

It is known that individual network weights can be redundant, and may not carry significant information, e.g. BID9 .

At the same time, large models often have the ability to completely memorize datasets ), yet they do not, but instead appear to learn generic task solutions.

A standing hypothesis for why overcomplete representations are necessary is that they make learning possible by transforming local minima into saddle points BID6 or to discover robust solutions, which do not rely on precise weight values BID13 BID16 .If large models are only needed for robustness during training, then significant compression of these models should be achievable, without impacting accuracy.

This intuition is strengthened by two related, but slightly different research directions.

The first direction is the work on training quantized neural networks, e.g. BID5 ; BID29 ; BID14 ; BID35 ; BID24 ; BID28 ; BID41 , which showed that neural networks can converge to good task solutions even when weights are constrained to having values from a set of integer levels.

The second direction aims to compress already-trained models, while preserving their accuracy.

To this end, various elegant compression techniques have been proposed, e.g. BID9 ; BID15 ; BID34 ; BID8 ; BID25 , which combine quantization, weight sharing, and careful coding of network weights, to reduce the size of state-of-the-art deep models by orders of magnitude, while at the same time speeding up inference.

Both these research directions are extremely active, and have been shown to yield significant compression and accuracy improvements, which can be crucial when making such models available on embedded devices or phones.

However, the literature on compressing deep networks focuses almost exclusively on finding good compression schemes for a given model, without significantly altering the structure of the model.

On the other hand, recent parallel work BID3 BID12 introduces the process of distillation, which can be used for transferring the behaviour of a given model to any other structure.

This can be used for compression, e.g. to obtain compact representations of ensembles BID12 .

However the size of the student model needs to be large enough for allowing learning to succeed.

A model that is too shallow, too narrow, or which misses necessary units, can result in considerable loss of accuracy BID31 .In this work, we examine whether distillation and quantization can be jointly leveraged for better compression.

We start from the intuition that 1) the existence of highly-accurate, full-precision teacher models should be leveraged to improve the performance of quantized models, while 2) quantizing a model can provide better compression than a distillation process attempting the same space gains by purely decreasing the number of layers or layer width.

While our approach is very natural, interesting research questions arise when these two ideas are combined.

Contribution.

We present two methods which allow the user to compound compression in terms of depth, by distilling a shallower student network with similar accuracy to a deeper teacher network, with compression in terms of width, by quantizing the weights of the student to a limited set of integer levels, and using less weights per layer.

The basic idea is that quantized models can leverage distillation loss BID12 , the weighted average between the correct targets (represented by the labels) and soft targets (represented by the teacher's outputs).We implement this intuition via two different methods.

The first, called quantized distillation, aims to leverage distillation loss during the training process, by incorporating it into the training of a student network whose weights are constrained to a limited set of levels.

The second method, which we call differentiable quantization, takes a different approach, by attempting to converge to the optimal location of quantization points through stochastic gradient descent.

We validate both methods empirically through a range of experiments on convolutional and recurrent network architectures.

We show that quantized shallow students can reach similar accuracy levels to full-precision and deeper teacher models on datasets such as CIFAR and ImageNet (for image classification) and OpenNMT and WMT (for machine translation), while providing up to order of magnitude compression, and inference speedup that is linear in the depth.

Related Work.

Our work is a special case of knowledge distillation BID3 BID12 , in which we focus on techniques to obtain high-accuracy students that are both quantized and shallower.

More generally, it can be seen as a special instance of learning with privileged information, e.g. BID32 ; BID37 , in which the student is provided additional information in the form of outputs from a larger, pre-trained model.

The idea of optimizing the locations of quantization points during the learning process, which we use in differentiable quantization, has been used previously in BID21 ; BID19 BID40 , although in the different context of matrix completion and recommender systems.

Using distillation for size reduction is mentioned in BID12 , for distilling ensembles.

To our knowledge, the only other work using distillation in the context of quantization is , which uses it to improve the accuracy of binary neural networks on ImageNet.

We significantly refine this idea, as we match or even improve the accuracy of the original full-precision model: for example, our 4-bit quantized version of ResNet18 has higher accuracy than full-precision ResNet18 (matching the accuracy of the ResNet34 teacher): it has higher top-1 accuracy (by >15%) and top-5 accuracy (by >7%) compared to the most accurate model in .

We start by defining a scaling function sc : R n → [0, 1], which normalizes vectors whose values come from an arbitrary range, to vectors whose values are in [0, 1] .

Given such a function, the general structure of the quantization functions is as follows: DISPLAYFORM0 where sc −1 is the inverse of the scaling function, andQ is the actual quantization function that only accepts values in [0, 1].

We always assume v to be a vector; in practice, of course, the weight vectors can be multi-dimensional, but we can reshape them to one dimensional vectors and restore the original dimensions after the quantization.

Scaling.

There are various specifications for the scaling function; in this paper, we will use linear scaling, e.g. BID11 , that is sc(v) = v−β α , with α = max i v i − min i v i and β = min i v i which results in the target values being in [0, 1] , and the quantization function DISPLAYFORM1 Bucketing.

One problem with this formulation is that an identical scaling factor is used for the whole vector, whose dimension might be huge.

Magnitude imbalance can result in a significant loss of precision, where most of the elements of the scaled vector are pushed to zero.

To avoid this, we will use bucketing, e.g. BID1 , that is, we will apply the scaling function separately to buckets of consecutive values of a certain fixed size.

The trade-off here is that we obtain better quantization accuracy for each bucket, but will have to store two floating-point scaling factors for each bucket.

We characterize the compression comparison in Section 5.

The functionQ can also be defined in several ways.

We will consider both uniform and non-uniform placement of quantization points.

Uniform Quantization.

We fix a parameter s ≥ 1, describing the number of quantization levels employed.

Intuitively, uniform quantization considers s + 1 equally spaced points between 0 and 1 (including these endpoints).

The deterministic version will assign each (scaled) vector coordinate v i to the closest quantization point, while in the stochastic version we perform rounding probabilistically, such that the resulting value is an unbiased estimator of v i , of minimal variance.

Formally, the uniform quantization function with s + 1 levels is defined aŝ DISPLAYFORM2 where ξ i is the rounding function.

For the deterministic version, we define k i = sv i − v i s and set DISPLAYFORM3 while for the stochastic version we will set ξ i ∼ Bernoulli(k i ).

Note that k i is the normalized distance between the original point v i and the closest quantization point that is smaller than v i and that the vector components are quantized independently.

Non-Uniform Quantization.

Non-uniform quantization takes as input a set of s quantization points {p 1 , . . .

, p s } and quantizes each element v i to the closest of these points.

For simplicity, we only define the deterministic version of this function.

In this section we list some interesting mathematical properties of the uniform quantization function.

Clearly, stochastic uniform quantization is an unbiased estimator of its input, i.e. DISPLAYFORM0 What interests us is applying this function to neural networks; as the scalar product is the most common operation performed by neural networks, we would like to study the properties of Q(v) T x, where v is the weight vector of a certain layer in the network and x are the inputs.

We are able to show that DISPLAYFORM1 where ε is a random variable that is asymptotically normally distributed, i.e. This means that quantizing the weights is equivalent to adding to the output of each layer (before the activation function) a zero-mean error term that is asymptotically normally distributed.

The variance of this error term depends on s. This connects quantization to work advocating adding noise to intermediary activations of neural networks as a regularizer BID7 and to BID2 , which investigates the connection between adding noise to a network weights and the network generalization properties.

We plan to investigate this connection in more detail in future work.

The context is the following: given a task, we consider a trained state-of-the-art deep model solving it-the teacher, and a compressed student model.

The student is compressed in the sense that 1) it is shallower than the teacher; and 2) it is quantized, in the sense that its weights are expressed at limited bit width.

The strategy, as for standard distillation BID3 BID12 is for the student to leverage the converged teacher model to reach similar accuracy.

We note that distillation has been used previously to obtain compact high-accuracy encodings of ensembles BID12 ; however, we believe this is the first time it is used for model compression via quantization.

Given this setup, there are two questions we need to address.

The first is how to transfer knowledge from the teacher to the student.

For this, the student will use the distillation loss, as defined by BID12 , as the weighted average between two objective functions: cross entropy with soft targets, controlled by the temperature parameter T , and the cross entropy with the correct labels.

We refer the reader to BID12 for the precise definition of distillation loss.

The second question is how to employ distillation loss in the context of a quantized neural network.

An intuitive approach is to rely on projected gradient descent, where a gradient step is taken as in full-precision training, and then the new parameters are projected to the set of valid solutions.

Critically, we accumulate the error at each projection step into the gradient for the next step.

One can think of this process as if collecting evidence for whether each weight needs to move to the next quantization point or not.

Crucially, the error accumulation prevents the algorithm from getting stuck in the current solution if gradients are small, which would occur in a naive projected gradient approach.

This is similar to the approach taken by BinaryConnect technique, with some differences.

also examines these dynamics in detail.

Compared to BinnaryConnect, we use distillation rather than learning from scratch, hence learning more efficiently.

We also do not restrict ourselves to binary representation, but rather use variable bit-width quantization functions and bucketing, as defined in Section 2.An alternative view of this process, illustrated in FIG2 , is that we perform the SGD step on the full-precision model, but computing the gradient on the quantized model, expressed with respect to the distillation loss.

See Algorithm 1 for details.

We introduce differentiable quantization as a general method of improving the accuracy of a quantized neural network, by exploiting non-uniform quantization point placement.

In particular, we are going to use the non-uniform quantization function defined in Section 2.1.

Experimentally, we have Let w be the network weights 3: loop 4: DISPLAYFORM0 Run forward pass and compute distillation loss l(w q )

Run backward pass and compute DISPLAYFORM0 Update original weights using SGD in full precision w = w − ν · ∂l(w q ) ∂w q 8: Finally quantize the weights before returning: w q ← quant function(w, s) 9: return w q quantize quantize quantize found little difference between stochastic and deterministic quantization in this case, and therefore will focus on the simpler deterministic quantization function here.

DISPLAYFORM1 Let p = (p 1 , . . .

, p s ) be the vector of quantization points, and let Q(v, p) be our quantization function, as defined previously.

Ideally, we would like to find a set of quantization points p which minimizes the accuracy loss when quantizing the model using Q(v, p).

The key observation is that to find this set p, we can just use stochastic gradient descent, because we are able to compute the gradient of Q with respect to p.

A major problem in quantizing neural networks is the fact that the decision of which p i should replace a given weight is discrete, hence the gradient is zero: ∂Q(v, p) ∂v = 0, almost everywhere.

This implies that we cannot backpropagate the gradients through the quantization function.

To solve this problem, typically a variant of the straight-through estimator is used, see e.g. BID4 BID14 .

On the other hand, the model as a function of the chosen p i is continuous and can be differentiated; the gradient of Q(v, p) i with respect to p j is well defined almost everywhere, and it is simply DISPLAYFORM2 where α i is i-th element of the scaling factor, assuming we are using a bucketing scheme.

If no bucketing is used, then α i = α for every i. Otherwise it changes depending on which bucket the weight v i belongs to.

Therefore, we can use the same loss function we used when training the original model, and with Equation (6) and the usual backpropagation algorithm we are able to compute its gradient with respect to the quantization points p.

Then we can minimize the loss function with respect to p with the standard SGD algorithm.

See Algorithm 2 for details.

Note on Efficiency.

Optimizing the points p can be slower than training the original network, since we have to perform the normal forward and backward pass, and in addition we need to quantize the weights of the model and perform the backward pass to get to the gradients w.r.t.

p.

However, in our experience differential quantization requires an order of magnitude less iterations to converge to a good solution, and can be implemented efficiently.

Let w be the networks weights and p the initial quantization points 3: loop 4: DISPLAYFORM3 Run forward pass and compute loss l(w q )

Run backward pass and compute DISPLAYFORM0 Use equation 6 to compute DISPLAYFORM1 Update quantization points using SGD or similar: DISPLAYFORM2 Weight Sharing.

Upon close inspection, this method can be related to weight sharing BID9 .

Weight sharing uses a k-mean clustering algorithm to find good clusters for the weights, adopting the centroids as quantization points for a cluster.

The network is trained modifying the values of the centroids, aggregating the gradient in a similar fashion.

The difference is in the initial assignment of points to centroids, but also, more importantly, in the fact that the assignment of weights to centroids never changes.

By contrast, at every iteration we re-assign weights to the closest quantization point, and use a different initialization.

While the loss is continuous w.r.t.

p, there are indirect effects when changing the way each weight gets quantized.

This can have drastic effect on the learning process.

As an extreme example, we could have degeneracies, where all weights get represented by the same quantization point, making learning impossible.

Or diversity of p i gets reduced, resulting in very few weights being represented at a really high precision while the rest are forced to be represented in a much lower resolution.

To avoid such issues, we rely on the following set of heuristics.

Future work will look at adding a reinforcement learning loss for how the p i are assigned to weights.

Choose good starting points.

One way to initialize the starting quantization points is to make them uniformly spaced, which would correspond to use as a starting point the uniform quantization function.

The differentiable quantization algorithm needs to be able to use a quantization point in order to update it; therefore, to make sure every quantization point is used we initialize the points to be the quantiles of the weight values.

This ensures that every quantization point is associated with the same number of values and we are able to update it.

Redistribute bits where it matters.

Not all layers in the network need the same accuracy.

A measure of how important each weight is to the final prediction is the norm of the gradient of each weight vector.

So in an initial phase we run the forward and backward pass a certain number of times to estimate the gradient of the weight vectors in each layer, we compute the average gradient across multiple minibatches and compute the norm; we then allocate the number of points associated with each weight according to a simple linear proportion.

In short we estimate DISPLAYFORM0 where l is the loss function,v is the vector of weights in a particular layer and DISPLAYFORM1 = ∂l ∂vi and we use this value to determine which layers are most sensitive to quantization.

When using this process, we will use more than the indicated number of bits in some layers, and less in others.

We can reduce the impact of this effect with the use of Huffman encoding, see Section 5; in any case, note that while the total number of points stays constant, allocating more points to a layer will increase bit complexity overall if the layer has a larger proportion of the weights.

Use the distillation loss.

In the algorithm delineated above, the loss refers to the loss we used to train the original model with.

Another possible specification is to treat the unquantized model as the teacher model, the quantized model as the student, and to use as loss the distillation loss between the outputs of the unquantized and quantized model.

In this case, then, we are optimizing our quantized model not to perform best with respect to the original loss, but to mimic the results of the unquantized model, which should be easier to learn for the model and provide better results.

Hyperparameter optimization.

The algorithm above is an optimization problem very similar to the original one.

As usual, to obtain the best results one should experiment with hyperparameters optimization, and different variants of gradient descent.

We now analyze the space savings when using b bits and bucket size of k. Let f be the size of full precision weights (32 bit) and let N be the size of the "vector" we are quantizing.

Full precision requires f N bits, while the quantized vector requires bN + 2f N k . (We use b bits per weight, plus the scaling factors α and β for every bucket).

The size gain is therefore g(b, k; f ) = kf kb+2f .

For differentiable quantization, we also have to store the values of the quantization points.

Since this number does not depend on N , the amount of space required is negligible and we ignore it for simplicity.

As an example, at 256 bucket size, using 2 bits per component yields 14.2× space savings w.r.t.

full precision, while 4 bits yields 7.52× space savings.

At 512 bucket size, the 2 bit savings are 15.05×, while 4 bits yields 7.75× compression.

Huffman encoding.

To save additional space, we can use Huffman encoding to represent the quantized values.

In fact, each quantized value can be thought as the pointer to a full precision value; in the case of non uniform quantization is p k , in the case of uniform quantization is k/s.

We can then compute the frequency for every index across all the weights of the model and compute the optimal Huffman encoding.

The mean bit length of the optimal encoding is the amount of bits we actually use to encode the values.

This explains the presence of fractional bits in some of our size gain tables from the Appendix.

We emphasize that we only use these compression numbers as a ballpark figure, since additional implementation costs might mean that these savings are not always easy to translate to practice BID9 .

Methods.

We will begin with a set of experiments on smaller datasets, which allow us to more carefully cover the parameter space.

We compare the performance of the methods described in the following way: we consider as baseline the teacher model, the distilled model and a smaller model: the distilled and smaller models have the same architecture, but the distilled model is trained using distillation loss on the teacher, while the smaller model is trained directly on targets.

Further, we compare the performance of Quantized Distillation and Differentiable Quantization.

In addition, we will also use PM ("post-mortem") quantization, which uniformly quantizes the weights after training without any additional operation, with and without bucketing.

All the results are obtained with a bucket size of 256, which we found to empirically provide a good compression-accuracy trade-off.

We refer the reader to Appendix A for details of the datasets and models.

CIFAR-10 Experiments.

For image classification on CIFAR-10, we tested the impact of different training techniques on the accuracy of the distilled model, while varying the parameters of a CNN architecture, such as quantization levels and model size.

TAB1 contains the results for full-precision training, PM quantization with and without bucketing, as well as our methods.

The percentages on the left below the student models definition are the accuracy of the normal and the distilled model respectively (trained with full precision).

More details are reported in table 11 in the appendix.

We also tried an additional model where the student is deeper than the teacher, where we obtained that the student quantized to 4 bits is able to achieve significantly better accuracy than the teacher, with a compression factor of more than 7×.We performed additional experiments for differentiable quantization using a wide residual network BID38 ) that gets to higher accuracies; see table 3.Overall, quantized distillation appears to be the method with best accuracy across the whole range of bit widths and architectures.

It outperforms PM significantly for 2bit and 4bit quantization, achieves accuracy within 0.2% of the teacher at 8 bits on the larger student model, and relatively minor accuracy loss at 4bit quantization.

Differentiable quantization is a close second on all experiments, but it has much faster convergence.

Further, we highlight the good accuracy of the much simpler PM quantization method with bucketing at higher bit width (4 and 8 bits).CIFAR-100 Experiments.

Next, we perform image classification with the full 100 classes.

Here, we focus on 2bit and 4bit quantization, and on a single student architecture.

The baseline architecture is a wide residual network with 28 layers, and 36.5M parameters, which is state-of-the-art for its depth on this dataset.

The student has depth and width reduced by 20%, and half the parameters.

It is chosen so that reaches the same accuracy as the teacher model when distilled at full precision.

Accuracy results are given in TAB4 .

More details are reported in TAB2 , in the appendix.

The results confirm the trend from the previous dataset, with distilled and differential quantization preserving accuracy within less than 1% at 4bit precision.

However, we note that accuracy loss is catastrophic at 2bit precision, probably because of reduced model capacity.

We note that differentiable quantization is able to best recover accuracy for this harder task.

OpenNMT Experiments.

The OpenNMT integration test dataset (Ope) consists of 200K train sentences and 10K test sentences for a German-English translation task.

To train and test models we use the OpenNMT PyTorch codebase BID17 .

We modified the code, in particular by adding the quantization algorithms and the distillation loss.

As measure of fit we will use perplexity and the BLEU score, the latter computed using the multi-bleu.perl code from the moses project (mos).Our target models consist of an embedding layer, an encoder consisting of n layers of LSTM, a decoder consisting of n layers of LSTM, and a linear layer.

The decoder also uses the global attention mechanism described in BID23 .

For the teacher network we set n = 2, for a total of 4 LSTM layers with LSTM size 500.

For the student networks we choose n = 1, for a total of 2 LSTM layers.

We vary the LSTM size of the student networks and for each one, we compute the distilled model and the quantized versions for varying bit width.

Results are summarized in TAB5 .

The BLEU scores below the student model refer to the BLEU scores of the normal and distilled model respectively (trained with full precision).

Details about the resulting size of the models are reported in table 23 in the appendix.

A reasonable intuition would be that recurrent neural networks should be harder to quantize than convolutional neural networks, as quantization errors do not average out when executing repeatedly through the same cell, but accumulate.

Results contradict this intuition.

In particular, medium and large-sized students are able to essentially recover the same scores as the teacher model on this dataset.

Perhaps surprisingly, bucketing PM and quantized distillation perform equally well for 4bit quantization.

As expected, cell size is an important indicator for accuracy, although halving both cell size and the number of layers can be done without significant loss.

WMT13 Experiments.

We run a similar LSTM architecture as above for the WMT13 dataset BID18 (1.7M sentences train, 190K sentences test) and we provide additional experiments for quantized distillation technique, see TAB6 .

We note that, on this large dataset, PM quantization does not perform well, even with bucketing.

On the other hand, quantized distillation with 4bits of precision has higher BLEU score than the teacher, and similar perplexity.

The ImageNet Dataset.

We also experiment with ImageNet using the ResNet architecture BID10 .

In the first experiment, we use a ResNet34 teacher, and a student ResNet18 student model.

Experiments quantizing the standard version of this student resulted in an accuracy loss of around 4%, and hence we experiment with a wider model, which doubles the number of filters for each convolutional layer.

We call this 2xResNet18.

This is in line with previous work on wide ResNet architectures BID38 , wide students for distillation BID3 , and wider quantized networks BID25 .

We also note that, in line with previous work on this dataset BID41 BID25 , we do not quantize the first and last layers of the models, as this can hurt accuracy.

After 62 epochs of training, the quantized distilled 2xResNet18 with 4 bits reaches a validation accuracy of 73.31%.

Surprisingly, this is higher than the unquantized ResNet18 model (69.75%), and has virtually the same accuracy as the ResNet34 teacher.

In terms of size, this model is more than 2× smaller than ResNet18 (but has higher accuracy), and is 4× smaller than ResNet34, and about 1.5× faster on inference, as it has fewer layers.

This is state-of-the-art for 4bit models with 18 layers; to our knowledge, no such model has been able to surpass the accuracy of ResNet18.We re-iterated this experiment using a 4-bit quantized 2xResNet34 student transferring from a ResNet50 full-precision teacher.

We obtain a 4-bit quantized student of almost the same accuracy, which is 50% shallower and has a 2.5× smaller size.

Distillation Loss versus Normal Loss.

One key question we are interested in is whether distillation loss is a consistently better metric when quantizing, compared to standard loss.

We tested this for CIFAR-10, comparing the performance of quantized training with respect to each loss.

At 2bit precision, the student converges to 67.22% accuracy with normal loss, and to 82.40% with distillation loss.

At 4bit precision, the student converges to 86.01% accuracy with normal loss, and to 88.00% with distillation loss.

On OpenNMT, we observe a similar gap: the 4bit quantized student converges to 32.67 perplexity and 15.03 BLEU when trained with normal loss, and to 25.43 perplexity (better than the teacher) and 15.73 BLEU when trained with distillation loss.

This strongly suggests that distillation loss is superior when quantizing.

For details, see Section A.4.1 in the Appendix.

Impact of Heuristics on Differentiable Quantization.

We also performed an in-depth study of how the various heuristics impact accuracy.

We found that, for differentiable quantization, redistributing bits according to the gradient norm of the layers is absolutely essential for good accuracy; quantiles and distillation loss also seem to provide an improvement, albeit smaller.

Due to space constraints, we defer the results and their discussion to Section A.4.2 of the Appendix.

Inference Speed.

In general, shallower students lead to an almost-linear decrease in inference cost, w.r.t.

the depth reduction.

For instance, in the CIFAR-10 experiments with the wide ResNet models, the teacher forward pass takes 67.4 seconds, while the student takes 43.7 seconds; roughly a 1.5x speedup, for 1.75x reduction in depth.

On the ImageNet test set using 4 GPUs (data-parallel), a forward pass takes 263 seconds for ResNet34, 169 seconds for ResNet18, and 169 seconds for our 2xResNet18.

(So, while having more parameters than ResNet18, it has the same speed because it has the same number of layers, and is not wide enough to saturate the GPU.

We note that we did not exploit 4bit weights, due to the lack of hardware support.)

Inference on our model is 1.5 times faster, while being 1.8 times shallower, so here the speedup is again almost linear.

We have examined the impact of combining distillation and quantization when compressing deep neural networks.

Our main finding is that, when quantizing, one can (and should) leverage large, accurate models via distillation loss, if such models are available.

We have given two methods to do just that, namely quantized distillation, and differentiable quantization.

The former acts directly on the training process of the student model, while the latter provides a way of optimizing the quantization of the student so as to best fit the teacher model.

Our experimental results suggest that these methods can compress existing models by up to an order of magnitude in terms of size, on small image classification and NMT tasks, while preserving accuracy.

At the same time, we note that distillation also provides an automatic improvement in inference speed, since it generates shallower models.

One of our more surprising findings is that naive uniform quantization with bucketing appears to perform well in a wide range of scenarios.

Our analysis in Section 2.2 suggests that this may be because bucketing provides a way to parametrize the Gaussian-like noise induced by quantization.

Given its simplicity, it could be used consistently as a baseline method.

In our experimental results, we performed manual architecture search for the depth and bit width of the student model, which is time-consuming and error-prone.

In future work, we plan to examine the potential of reinforcement learning or evolution strategies to discover the structure of the student for best performance given a set of space and latency constraints.

The second, and more immediate direction, is to examine the practical speedup potential of these methods, and use them together and in conjunction with existing compression methods such as weight sharing BID9 and with existing low-precision computation frameworks, such as NVIDIA TensorRT, or FPGA platforms.

The model used to train CIFAR10 is the one described in BID31 with some minor modifications.

We use standard data augmentation techniques, including random cropping and random flipping.

The learning rate schedule follows the one detailed in the paper.

The structure of the models we experiment with consists of some convolutional layers, mixed with dropout layers and max pooling layers, followed by one or more linear layers.

The model used are defined in TAB8 .

The c indicates a convolutional layer, mp a max pooling layer, dp a dropout layer, fc a linear (fully connected) layer.

The exponent indicates how many consecutive layers of the same type are there, while the number in front of the letter determines the size of the layer.

In the case of convolutional layers is the number of filters.

All convolutional layers of the teacher are 3x3, while the convolutional layers in the smaller models are 5x5.

Teacher model 76c 2 -mp-dp-126c 2 -mp-dp-148c 4 -mp-dp-1200fc-dp-1200fc Smaller model 1 75c-mp-dp-50c 2 -mp-dp-25c-mp-dp-500fc-dp Smaller model 2 50c-mp-dp-25c 2 -mp-dp-10c-mp-dp-400fc-dp Smaller model 3 25c-mp-dp-10c 2 -mp-dp-5c-mp-dp-300fc-dpFollowing the authors of the paper, we don't use dropout layers when training the models using distillation loss.

Distillation loss is computed with a temperature of T = 5.

TAB9 reports the accuracy of the models trained (in full precision) and their size.

TAB1 reports the accuracy achieved with each method, and table 11 reports the optimal mean bit length using Huffman encoding and resulting model size.

We also performed an experiment with a deeper student model.

The architecture is 76c 3 -mp-dp126c3 -mp-dp-148c 5 -mp-dp-1000fc-dp-1000fc-dp-1000fc (following the same notation as in For our CIFAR100 experiments, we use the same implementation of wide residual networks as in our CIFAR10 experiments.

The wide factor is a multiplicative factor controlling the amount of filters in each layer; for more details please refer to the original paper BID38 .

We train for 200 epochs with an initial learning rate of 0.1.

As mentioned in the main text, we use the openNMT-py codebase.

We slightly modify it to add distillation loss and the quantization methods proposed.

We mostly use standard options to train the model; in particular, the learning rate starts at 1 and is halved every epoch starting from the first epoch where perplexity doesn't drop on the test set.

We train every model for 15 epochs.

Distillation loss is computed with a temperature of T = 1.

For the WMT13 datasets, we run a similar architecture.

We ran all models for 15 epochs; the smaller model overfit with 15 epochs, so we ran it for 5 epochs instead.

In this section we highlight the positive effects of using distillation loss during quantization.

We take models with the same architecture and we train them with the same number of bits; one of the models is trained with normal loss, the other with the distillation loss with equal weighting between soft cross entropy and normal cross entropy (that is, it is the quantized distilled model).

TAB2 shows the results on the CIFAR10 dataset; the models we train have the same structure as the Smaller model 1, see Section A.1.

TAB2 shows the results on the openNMT integration test dataset; the models trained have the same structure of Smaller model 1, see Section A.3.

Notice that distillation loss can significantly improve the accuracy of the quantized models.

These results suggest that quantization works better when combined with distillation, and that we should try to take advantage of this whenever we are quantizing a neural network.

To test the different heuristics presented in Section 4.2, we train with differentiable quantization the Smaller model 1 architecture specified in Section A.1 on the cifar10 dataset.

The same model is trained with different heuristics to provide a sense of how important they are; the experiments is performed with 2 and 4 bits.

Results suggests that when using 4 bits, the method is robust and works regardless.

When using 2 bits, redistributing bits according to the gradient norm of the layers is absolutely essential for this method to work ; quantiles starting point also seem to provide an small improvement, while using distillation loss in this case does not seem to be crucial.

In this section we will prove some results about the uniform quantization function, including the fact that is asymptotically normally distributed, see subsection B.1 below.

Clearly, we refer to the stochastic version, see Section 2.1.

We first start proving the unbiasedness ofQ; DISPLAYFORM0 Then it is immediate that DISPLAYFORM1 Bounds on second and third momentWe will write out bounds onQ; the analogous bounds on Q are then straightforward.

For convenience, let us calll i = v i s s 3B.1 ASYMPTOTIC NORMALITY Most of neural networks operations are scalar product computation.

Therefore, the scalar product of the quantized weights and the inputs is an important quantity: DISPLAYFORM2 We already know from section B that the quantization function is unbiased; hence we know that DISPLAYFORM3 with ε n is a zero-mean random variable.

We will show that ε n tends in distribution to a normal random variable.

To prove asymptotic normality, we will use a generalized version of the central limit theorem due to Lyapunov: Theorem B.1 (Lyapunov Central Limit Theorem).

Let {X 1 , X 2 , . . . } be a sequence of independent random variables, each with finite expected value µ i and variance σ DISPLAYFORM4 is satisfied, then DISPLAYFORM5 We can now state the theorem: Theorem B.2.

Let v, x be two vectors with n elements.

Let Q be the uniform quantization function with s levels defined in 2.1 and define s Proof.

Using the same notation as theorem B.1, let X i = Q(v i )x i , µ i = E[X i ] = v i x i .

We already mentioned in 2.1 that these are independent random variables.

We will show that the Lyapunov condition holds with δ = 1.

@highlight

Obtains state-of-the-art accuracy for quantized, shallow nets by leveraging distillation. 

@highlight

Proposes small and low-cost models by combining distillation and quantization for vision and neural machine translation experiments

@highlight

This paper presents a framework of using the teacher model to help the compression for the deep learning model in the context of model compression.