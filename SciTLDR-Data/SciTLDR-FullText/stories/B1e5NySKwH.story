Low bit-width integer weights and activations are very important for efficient inference, especially with respect to lower power consumption.

We propose to apply Monte Carlo methods and importance sampling to sparsify and quantize pre-trained neural networks without any retraining.

We obtain sparse, low bit-width integer representations that approximate the full precision weights and activations.

The precision, sparsity, and complexity are easily configurable by the amount of sampling performed.

Our approach, called Monte Carlo Quantization (MCQ), is linear in both time and space, while the resulting quantized sparse networks show minimal accuracy loss compared to the original full-precision networks.

Our method either outperforms or achieves results competitive with methods that do require additional training on a variety of challenging tasks.

Developing novel ways of increasing the efficiency of neural networks is of great importance due to their widespread usage in today's variety of applications.

Reducing the network's footprint enables local processing on personal devices without the need for cloud services.

In addition, such methods allow for reducing power consumption -also in data centers.

Very compact models can be fully stored and executed on-chip in specialized hardware like for example ASICs or FPGAs.

This reduces latency, increases inference speed, improves privacy concerns, and limits bandwidth cost.

Quantization methods usually require re-training of the quantized model to achieve competitive results.

This leads to an additional cost and complexity.

The proposed method, Monte Carlo Quantization (MCQ), aims to avoid retraining by approximating the full-precision weight and activation distributions using importance sampling.

The resulting quantized networks achieve close to the full-precision accuracy without any kind of additional training.

Importantly, the complexity of the resulting networks is proportional to the number of samples taken.

First, our algorithm normalizes the weights and activations of a given layer to treat them as probability distributions.

Then, we randomly sample from the corresponding cumulative distributions and count the number of hits for every weight and activation.

Finally, we quantize the weights and activations by their integer count values, which form a discrete approximation of the original continuous values.

Since the quality of this approximation relies entirely on (quasi)random sampling, the accuracy of the quantized model is directly dependent on the amount of sampling performed.

Thus, accuracy may be traded for higher sparsity and speed by adjusting the number of samples.

On the challenging tasks of image classification, language modeling, speech recognition, and machine translation, our method outperforms or is competitive with existing quantization methods that do require additional training.

The computational cost of neural networks can be reduced by pruning redundant weights or neurons, which has been shown to work well Mocanu et al., 2018; LeCun et al., 1990) .

Alternatively, the precision of the network weights and activations may be lowered, potentially introducing sparsity.

Using low precision computations to reduce the cost and sparsity to skip computations allows for efficient hardware implementations Venkatesh et al., 2017) .

This is the approach used in this paper.

BinaryConnect proposed training with binary weights, while XNOR-Net (Rastegari et al., 2016) and BNN (Hubara et al., 2016) extended this binarization to activations as well.

TWN (Li et al., 2016) proposed ternary quantization instead, increasing model expressiveness.

Similarly, TTQ (Zhu et al., 2016) used ternary weights with a positive and negative scaling learned during training.

LR-Net (Shayer et al., 2017 ) made use of both binary and ternary weights by using stochastic parameterization while INQ (Zhou et al., 2017) constrained weights to powers of two and zero.

FGQ (Mellempudi et al., 2017) categorized weights in different groups and used different scaling factors to minimize the element-wise distance between full and low-precision weights.

Wang et al. (2019) used the hardware accelerator's feedback to perform hardware-aware quantization using reinforcement learning.

Zhang et al. (2018) jointly trained quantized networks and respective quantizers.

Reagen et al. (2017) used Bloomier filters to compactly encode network weights.

Similarly, quantization techniques can also be applied in the backward pass.

Therefore, some previous work quantized not only weights and activations but also the gradients to augment training performance (Zhou et al., 2016; Gupta et al., 2015; Courbariaux et al., 2014) .

In particular, RQ (Louizos et al., 2018) propose a differentiable quantization procedure to allow for gradient-based optimization using discrete values and Wu et al. (2018) recently proposed to discretize weights, activations, gradients, and errors both at training and inference time.

These quantization techniques have great benefits and have shown to successfully reduce the computation requirements compared to full-precision models.

However, all the aforementioned methods require re-training of the quantized network to achieve close to full-precision accuracy, which can introduce significant financial and environmental cost (Strubell et al., 2019) .

On the other hand, our method instantly quantizes pre-trained neural networks with minimal accuracy loss as compared to their full-precision counterparts without any kind of additional training.

Neural networks make extensive use of randomization and random sampling techniques.

Examples are random initialization of network weights, stochastic gradient descent (Robbins & Monro, 1951) , regularization techniques such as Dropout (Srivastava et al., 2014) and DropConnect (Wan et al., 2013) , data augmentation and data shuffling, recurrent neural networks' regularization (Merity et al., 2017a) , or the generator's noise input on generative adversarial networks (Goodfellow et al., 2014) .

Many state-of-the-art networks use ReLU (Nair & Hinton, 2010) , which has interesting properties such as scale-invariance.

This enables a scaling factor to be propagated through all network layers without affecting the network's original output.

This principle can be used to normalize network values, such as weights and activations, as further described in Section 3.1.

After normalization, these values can be treated as probabilities, which enables the simulation of discrete probability densities to approximate the corresponding full-precision, continuous distributions (Section 3.2).

Assuming the exclusive use of the ReLU activation function in the hidden layers, the scale-invariance property of the ReLU activation function allows for arbitrary scaling of the weights or activations without affecting the network's output.

Given weights w l???1,i,j connecting the i-th neuron in layer l ??? 1 to the j-th neuron in layer l, where i ??? [0, N l???1 ??? 1] and j ??? [0, N l ??? 1], with N l???1 and N l the number of neurons of layer l ???1 and l, respectively.

Let a l,j be the j-th activation in the l-th layer and f ??? R + : a l,j = max 0,

Biases and incoming weights for neuron j in layer l may then be normalized by f = w l???1,j 1 = N l???1 ???1 i=0 |w l???1,i,j |, enabling weights to be seen as a probability distribution over all connections to a neuron.

A similar procedure could be used to normalize all activations a l,j of layer l.

Propagating these scaling factors forward layer by layer results in a single scalar (per output), which converts the outputs of the normalized network to the same range as the original network.

This technique allows for the usage of integer weights and activations throughout the entire network without requiring rescaling or conversion to floating point at every layer.

Taking advantage of the normalized network, we can simulate discrete probability densities by constructing a probability density function (PDF) and then sampling from the corresponding cumulative density function (CDF).

The number of references of a weight is then the quantized integer approximation of the continuous value.

For simplicity, the following discussion shows the quantization procedure for weights; activations can be quantized in the same way at inference time.

Without loss of generality, given n weights, assuming n???1 k=0 |w k | = w 1 = 1 and defining a partition of the unit interval by P m := m k=1 |w k | we have the following partitions:

Then, given N uniformly distributed samples x i ??? [0, 1), we can approximate the weight distribution as follows:

where j i ??? {0, . . . , n ??? 1} is uniquely determined by P ji???1 ??? x i < P ji .

One can further improve this sampling process by using jittered equidistant sampling.

Thus, given a random variable ?? ??? [0, 1), we generate N uniformly distributed samples x i ??? [0, 1) such that

The combination of equidistant samples and a random offset improves the weight approximation, as the samples are more uniformly distributed.

The variance of different sampling seeds is discussed in the Appendix.

Our approach builds on the aforementioned ideas of network normalization and quantization using random sampling to quantize an entire pre-trained full-precision neural network.

As before, we focus on weight quantization; online activation quantization is discussed in Section 4.4.

Our method, called Monte Carlo Quantization (MCQ), consists of the following steps, which are executed layer by layer:

(1) Create a probability density function (PDF) for all N l,w weights of layer l such that

(2) Perform importance sampling on the weights based on their magnitude by sampling from the corresponding cumulative density function (CDF) and counting the number of hits per weight (Section 4.2).

(3) Replace each weight with its quantized integer value, i.e. its hit count, to obtain a low bit-width, integer weight representation (Section 4.3).

The pseudo-code for our method is shown in Algorithm 1 of the Appendix.

Figure 1 illustrates both the normalization and importance sampling processes for a layer with 10 weights and 1 sample per weight, i.e. K = 1.0.

Performing normalization neuron-wise, as introduced in Section 3.1 may result in an inferior approximation, especially when the number of weights to sample from is small, as for example in convolutional layers with a small number of filters or input channels.

To mitigate this, we propose to normalize all neurons simultaneously in a layer-wise manner.

This has the additional advantage that samples can be redistributed from low-importance neurons to high-importance neurons (according to and uniformly sample from the corresponding CDF (c).

The sampling process produces quantized integer network weights based on the number of hits per weight (d).

Note that since weights 7, 8, and 9 were not hit, sparsity is introduced which can be exploited by hardware accelerators.

some metric), resulting in an increased level of sparsity.

Additionally, there is more opportunity for global optimization, so the overall weight distribution approximation improves as well.

We use the 1-norm of all weights of a given layer l as the scaling factor f used to perform weight normalization.

Thus, each normalized weight can be seen as a probability with respect to all connections between layer l ??? 1 and layer l, instead of a single neuron.

This layer-wise normalization technique is similar to Weight Normalization (Salimans & Kingma, 2016) , which decouples the neuron weight vector magnitude from its direction.

As introduced in Section 3.2, we generate ternary samples (hit positive weight, hit negative weight, or no hit), and count such hits during the sampling process.

Note that even though the individual samples are ternary, the final quantized values may not be, because a single weight can be sampled multiple times.

For jittered sampling, we use one random offset per layer, with a number of samples N = K ?? N values , where K ??? R + is a user-specified parameter to control the number of samples and N values represents the number of weights of a given layer.

By varying K, the computational cost of sampling can be traded off better approximation (more bits per weight) of the original weight distribution, leading to higher accuracy.

In our experiments, K is set the same for all network layers.

One simple modification to enhance the quality of the discrete approximation is to sort the continuous values prior to creating the PDF.

Applying sorting mechanisms to Monte Carlo schemes has been shown to be beneficial in the past (L'Ecuyer et al., 2008; .

Sorting groups smaller values together in the overall distribution.

Since we are using a uniform sampling strategy, smaller weights are then sampled less often, which results in both higher sparsity and a better quantized approximation of the larger weights in practice.

This effect is particularly significant on smaller layers with fewer weights.

Since the quantized integer weights span a different range of values than the original weights, and biases remain unchanged, care must be taken to ensure the activations of each neuron are calculated correctly.

After the integer multiply-accumulate (MAC) operation, the result must then be scaled by f N before adding the bias.

This requires the storage of one floating point scaling value per layer.

However, weights are stored as low bit-width integers and the computational cost is greatly reduced since the MAC operations use low-precision integers only instead of floating point numbers.

The number of bits required for the weights B W l ??? N, for layer l and its quantized weights Q(w l,i ), corresponds to the bit amount needed to represent the highest hit count during sampling, including its sign: B W l = 1 + log 2 (max 0???i???Nw???1 |Q(w l,i )|) + 1.

Alternatively, positive and negative weights could be separated into two sets.

While weights are quantized offline, i.e. after training and before inference, activations are quantized online during inference time using the same procedure as weight quantization previously described.

Thus, in the normalization step (Section 4.1), all N l,a activations of a given layer l are treated as a probability distribution over the output features, such that N l,a ???1 j=0 |a l,j | = 1.

Then, in the importance sampling step (Section 4.2), activations are sub-sampled using possibly different relative sampling amounts, i.e. K, than the ones used for the weights (we use the same K for both weights and activations in all of our experiments).

The required number of bits B A l for the quantized activations Q(a l,j ) can also be calculated similarly as described in Section 4.3, although no additional bit sign is required when using ReLU since all activations are non-negative.

The proposed method is extensively evaluated on a variety of tasks: for image classification we use CIFAR-10 ( Krizhevsky & Hinton, 2009) , SVHN (Netzer et al., 2011) , and ImageNet (Deng et al., 2009) , on multiple models each.

We further evaluate MCQ on language modeling, speech recognition, and machine translation, to assess the preformance of MCQ across different task domains.

Due to the automatic quantization done by MCQ, some layers may be quantized to lower or higher levels than others.

We indicate the quantization level for the whole network by the average number of bits, e.g. '8w-32a' means that on average 8 bits were used for weights and 32 bits for activations on each layer.

Many works note that quantizing the first or last network layer reduces accuracy significantly Zhou et al., 2016; Li et al., 2016) .

We use footnotes 1 , 2 , and 3 to denote the special treatment of first or last layers respectively.

For MCQ we report the results with both quantized and full-precision first layer.

We do not quantize Batch Normalization layers as the parameters are fixed after training and can be incorporated into the weights and biases (Wu et al., 2018) .

Tables 1 to 4 show the accuracy difference ??? between the quantized and full-precision models.

For other compared works this difference is calculated using the baseline models reported in each of the respective works.

We didn't perform any search over random sampling seeds for MCQ's results.

The best accuracies on VGG-7, VGG-14, and ResNet-20 produced by our method using K = 1.0 on CIFAR-10 are shown in Table 1 .

We refer to the Appendix for model and training details.

MCQ outperforms or shows competitive results showing minimal accuracy loss on all tested models against the compared methods that require network re-training.

The full-precision baselines for BNN (Hubara et al., 2016) and XNOR-Net (Rastegari et al., 2016) are from BC as these works use the same model.

Similarly, BWN (Rastegari et al., 2016) 's results on VGG-7 are the ones reported in TWN (Li et al., 2016 ) since they did not report the baseline in the original paper.

Figure 2 shows the effects of varying the amount of sampling, i.e. using K ??? [0.1...2.0].The average percentage of used weights/activations per layer and corresponding bit-widths of the final quantized model is also presented on each graph.

We observe a rapid increase of the accuracy even when sparsity levels are high on all tested models.

For SVHN, the tested models are identical to the compared methods.

Models B, C, and D have the same architecture as Model A but with a 50%, 75%, and 87.5% reduction in the number of filters in each convolutional layer, respectively (Zhou et al., 2016) .

We refer to the Appendix for further model and training details.

Table 2 shows MCQ's results for several models on SVHN using K = 1.0.

On bigger models, i.e. VGG-7* and Model A, we see minimal accuracy loss when compared to the full-precision baselines.

For the smaller models, we observe a slight accuracy degradation as model size decreases due to the reduction in the sample size, resulting in a poorer approximation.

However, we used only about 4 bits per weight/activation for such models.

Thus, increasing the number of samples would improve Figure 2: Results of quantizing both weights and activations on CIFAR-10 using different sampling amounts.

The quantized models reach close to full-precision accuracy at around half the sample size while using only around half the weights and one-third of the activations of the full-precision models.

accuracy while still maintaining a low bit-width.

Figure 3 illustrates the consequences of varying the number of samples.

Less samples are required than on CIFAR-10 for bigger models to achieve close to full-precision accuracy.

Potentially this is because layers have a larger number of weights and activations, so a larger sample size reduces quantization noise since the important values being more likely to be better approximated.

For ImageNet, we evaluate MCQ on AlexNet, ResNet-18, and ResNet-50 using the pre-trained models provided by Pytorch's model zoo (Paszke et al., 2017) ).

Table 3 shows the results on ImageNet with K = 5.0 for the different models.

The results shown for DoReFa, BWN, TWN (Zhou et al., 2016; Rastegari et al., 2016; Li et al., 2016) are the ones reported in TTQ (Zhu et al., 2016) .

Figure 4 shows the accuracy of the quantized model when using different sample sizes, i.e., K ??? [0.25, ..., 5.0].

We observe that more sampling is required to achieve a close to full-precision model accuracy on ImageNet.

On this dataset, sorting the CDF before sampling didn't result in any improvements, so reported results are without sorting.

All the quantized models achieve close to full-precision accuracy, though more samples are required than for the previous datasets resulting in a higher required bit-width.

To assess the robustness of MCQ, we further evaluate MCQ on several models in natural language and speech processing.

We evaluate language modeling on Wikitext-103 using a Transformer-based model (Baevski & Auli, 2018) and Wikitext-2 using a 2-layer LSTM (Zhao et al., 2019) , speech recognition on VCTK using Deepspeech2 (Amodei et al., 2015) , and machine translation on WMT-14 English-to-French using a Transformer (Ott et al., 2018) .

Additional details are provided in the Appendix.

Table 4 shows the comparison to full-precision models for these various tasks. (1W-32A) +0.14 ----??? BNN (1W-1A) - Results of quantizing both weights and activations on SVHN using different sampling amounts.

The quantized VGG-7* model reaches close to full-precision accuracy using around 0.5 samples per weight/activation, requiring around 8 bits and using 22% of the weights of the original model, with 22% nonzero activations.

Model A, B, C, and D are less redundant models that require more sampling to achieve close to full-precision accuracy.

The experimental results show the performance of MCQ on multiple models, datasets, and tasks, demonstrated by the minimal loss of accuracy compared to the full-precision counterparts.

MCQ either outperforms or is competitive to other methods that require additional training of the quantized network.

Moreover, the trade-off between accuracy, sparsity, and bit-width can be easily controlled by adjusting the number of samples.

Note that the complexity of the resulting quantized network is proportional to the number of samples in both space and time.

One limitation of MCQ, however, is that it often requires a higher number of bits to represent the quantized values.

On the other hand, this sampling-based approach directly translates to a good approximation of the real full-precision values without any additional training.

Recently Zhao et al. (2019) proposed to outlier channel splitting, which is orthogonal work to MCQ and could be used to reduce the bit-width required for the highest hit counts.

There are several paths that could be worth following for future investigations.

In the importance sampling stage, using more sophisticated metrics for importance ranking, e.g. approximation of the Hessian by Taylor expansion could be beneficial (Molchanov et al., 2016) .

Automatically selecting optimal sampling levels on each layer could lead to a lower cost since later layers seem to tolerate more sparsity and noise.

For efficient hardware implementation, it's important that the quantized Figure 4: Results of quantizing both weights and activations on ImageNet using different sampling amounts.

All quantized models reach close to full-precision accuracy at K = 3.

Table 4 : Evaluation of MCQ on language modeling, speech recognition, and machine translation.

All quantized models reach close to full precision performance.

Note that, as opposed to the image classification task, we did not study different sampling amounts nor the effect of quantization on specific network layers.

A more in-depth analysis could then help to achieve close to full-precision accuracy at a lower bit-width on these additional models.

network can be executed using integer operations only.

Bias quantization and rescaling, activation rescaling to prevent overflow or underflow, and quantization of errors and gradients for efficient training leave room for future work.

In this work, we showed that Monte Carlo sampling is an effective technique to quickly and efficiently convert floating-point, full-precision models to integer, low bit-width models.

Computational cost and sparsity can be traded for accuracy by adjusting the number of sampling accordingly.

Our method is linear in both time and space in the number of weights and activations, and is shown to achieve similar results as the full-precision counterparts, for a variety of network architectures, datasets, and tasks.

In addition, MCQ is very easy to use for quantizing and sparsifying any pre-trained model.

It requires only a few additional lines of code and runs in a matter of seconds depending on the model size, and requires no additional training.

The use of sparse, low-bitwidth integer weights and activations in the resulting quantized networks lends itself to efficient hardware implementations.

A ALGORITHM An overview of the proposed method is given in Algorithm 1.

Input: Pre-trained full-precision network Output: Quantized network with integer weights for K=0 to L-1 do

//

Update layer's precision B W K ??? 1 + f loor(log 2 (max(abs(W K )))) + 1 ; end Algorithm 1: Monte Carlo Quantization (MCQ) on network weights.

L represents the number of trainable layers, K indicates the percentage of samples to be sampled per weight.

The process is performed equivalently for quantizing activations at inference time.

Our algorithm is linear in both time and space in the number of weights and activations.

When using integer weights, care has to be taken to avoid overflows in the activations.

For that, activations can be scaled using a dynamically computed shifting factor as in Wu et al. (2018) .

With Monte Carlo sampling, since we know the expected value of the next-layer activations, we can scale accordingly.

With the activation equation presented in Section 3.1 and N I connections from the input layer to every neuron in the second layer:

With

The activations of a neuron need to be scaled by its number of inputs (the receptive field F in ), multiplied with the number of samples per weight and the number of samples per activation.

This is also valid for neurons in convolutional layers, where the receptive field is 3D, e.g. 3 ?? 3 ?? 128.

Moreover, care must be taken to scale biases correctly, by taking both the scaling of weights and activations into account:

We trained our full-precision baseline models on the CIFAR-10 dataset Krizhevsky & Hinton (2009) , consisting of 50000 training samples.

We evaluated both our full-precision and quantized models similarly on the rest of the 10000 testing samples.

The full-precision VGG-7 (2??128C3???M P 2???2?? 256C3???M P 2???2??512C3???M P 2???1024F C ???Sof tmax) and VGG-14 (2??64C3???M P 2???2?? 128C3???M P 2???3??256C3???M P 2???3??512C3???M P 2???3??512C3???M P 2???1024F C???Sof tmax) models were trained using the code at https://github.com/bearpaw/pytorch-classification.

Each was trained for 300 epochs with the Adam optimizer, with a learning rate starting at 0.1 and decreased by factor 10 at epochs 150 and 225, batch size of 128, and weights decay of 0.0005.

The ResNet-20 model uses the standard configuration described , with 64, 128 and 256 filters in the respective residual blocks.

We used more filters to increase the number of available weights in the first block to sample from.

This could be similarly performed by sampling more on this specific model to reduce the accuracy loss.

The ResNet-20 model is trained using the same hyperparameter settings as the VGG models.

We trained our full-precision baseline models on the Street View House Numbers (SVHN) dataset Netzer et al. (2011) , consising of 73257 training samples.

We evaluated both our full-precision and quantized models similarly using the 26032 testing samples provided in this dataset.

The fullprecision VGG-7* model (2 ?? 64C3 ??? M P 2 ??? 2 ?? 128C3 ??? M P 2 ??? 2 ?? 256C3 ??? M P 2 ??? 1024F C ??? Sof tmax) was trained for 164 epochs, using the Adam optimizer with learning rate starting at 0.001 and divided by 10 at epochs 80 and 120, weight decay 0.001, and batch size 200.

Models A (48C3 ??? M P 2 ??? 2 ?? 64C3 ??? M P 2 ??? 3 ?? 128C3 ??? M P 2 ??? 512C3 ??? Sof tmax), B, C, and D were trained using the code at https://github.com/aaron-xichen/pytorch-playground and the same hyperparameter settings as VGG-7* but trained for 200 epochs.

We evaluated both our full-precision and quantized models similarly on the validation set of the ILSVRC12 classification dataset Deng et al. (2009) , consisting of 50K valida-tion images.

The full-precision pre-trained models are taken from Pytorch's model zoo https://pytorch.org/docs/stable/torchvision/models.html (Paszke et al., 2017) .

CSTR's VCTK Corpus (Centre for Speech Technology Voice Cloning Toolkit) includes speech data uttered by 109 native speakers of English with various accents, where each speaker reads out about 400 sentences, most of which were selected from a newspaper.

The evaluated model uses 2 convolutional layers and 5 GRU layers of 768 hidden units, using code from https://github.com/SeanNaren/deepspeech.pytorch (Veaux et al., 2017) .

The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia.

Compared to the preprocessed version of Penn Treebank (PTB), WikiText-2 is over 2 times larger and WikiText-103 is over 110 times larger.

The WikiText dataset also features a far larger vocabulary and retains the original case, punctuation and numbers -all of which are removed in PTB.

As it is composed of full articles, the dataset is well suited for models that can take advantage of long term dependencies.

The WikiText-2 model was a 2-layer LSTM with 650 hidden neurons, and an embedding size of 400.

It was trained using the setup and code at https://github.com/salesforce/awdlstm-lm (Merity et al., 2017b) .

The WikiText-102 model was a pretrained model available at https://github.com/pytorch/fairseq/tree/master/examples/language model, along with evaluation code (Baevski & Auli, 2018 ).

The dataset is WMT14 English-French, cmobining data from several other corpuses, amongst others the Europarl corpus, the News Commentary corpus, and the Common Crawl corpus (Machacek & Bojar, 2014) .

The model was a pretrained model available at https://github.com/pytorch/fairseq/tree/master/examples/scaling nmt, along with evaluation code (Ott et al., 2018) .

Figures 5, 6, and 7 show the effects of varying the amounts of sampling when quantizing only the weights.

E QUANTIZING ACTIVATIONS ONLY Figures 8, 9 , and 10 show the effects of varying the amounts of sampling when quantizing only the activations.

We observe less sampling is required to achieve full-precision accuracy when quantizing only the activations when compared to quantizing the weights only.

In a small experiment on CIFAR-10, we observe that using different sampling seeds can result in up to a ??? 0.5% absolute variation in accuracy of the different quantized networks (Figure 11) .

Grid searching over several sampling seeds may then be beneficial to achieve a better quantized model in the end, depending on the use-case.

@highlight

Monte Carlo methods for quantizing pre-trained models without any additional training.