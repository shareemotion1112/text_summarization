Depthwise separable convolutions reduce the number of parameters and computation used in convolutional operations while increasing representational efficiency.

They have been shown to be successful in image classification models, both in obtaining better models than previously possible for a given parameter count (the Xception architecture) and considerably reducing the number of parameters required to perform at a given level (the MobileNets family of architectures).

Recently, convolutional sequence-to-sequence networks have been applied to machine translation tasks with good results.

In this work, we study how depthwise separable convolutions can be applied to neural machine translation.

We introduce a new architecture inspired by Xception and ByteNet, called SliceNet, which enables a significant reduction of the parameter count and amount of computation needed to obtain results like ByteNet, and, with a similar parameter count, achieves better results.

In addition to showing that depthwise separable convolutions perform well for machine translation, we investigate the architectural changes that they enable: we observe that thanks to depthwise separability, we can increase the length of convolution windows, removing the need for filter dilation.

We also introduce a new super-separable convolution operation that further reduces the number of parameters and computational cost of the models.

In recent years, sequence-to-sequence recurrent neural networks (RNNs) with long short-term memory (LSTM) cells BID7 have proven successful at many natural language processing (NLP) tasks, including machine translation BID18 BID2 BID4 .

In fact, the results they yielded have been so good that the gap between human translations and machine translations has narrowed significantly BID23 and LSTM-based recurrent neural networks have become standard in natural language processing.

Even more recently, auto-regressive convolutional models have proven highly effective when applied to audio BID19 , image BID20 ) and text generation BID11 .

Their success on sequence data in particular rivals or surpasses that of previous recurrent models BID11 BID6 .

Convolutions provide the means for efficient non-local referencing across time without the need for the fully sequential processing of RNNs.

However, a major critique of such models is their computational complexity and large parameter count.

These are the principal concerns addressed within this work: inspired by the efficiency of depthwise separable convolutions demonstrated in the domain of vision, in particular the Xception architecture BID5 and MobileNets (Howard et al., 2017) , we generalize these techniques and apply them to the language domain, with great success.

We present a new convolutional sequence-to-sequence architecture, dubbed SliceNet, and apply it to machine translation tasks, achieving results that surpass all previous reported experiments except for the recent Transformer model (Vaswani et al., 2017) .

Our architecture features two key ideas:??? Inspired by the Xception network BID5 , our model is a stack of depthwise separable convolution layers with residual connections.

Such an architecture has been previously shown to perform well for image classification.

We also experimented with using grouped convolutions (or "sub-separable convolutions") and add even more separation with our new super-separable convolutions.??? We do away with filter dilation in our architecture, after exploring the trade-off between filter dilation and larger convolution windows.

Filter dilation was previously a key component of successful 1D convolutional architectures for sequence-to-sequence tasks, such as ByteNet BID11 and WaveNet (van den Oord et al., 2016a), but we obtain better results without dilation thanks to separability.

The depthwise separable convolution operation can be understood as related to both grouped convolutions and the "inception modules" used by the Inception family of convolutional network architectures, a connection explored in Xception BID5 .

It consists of a depthwise convolution, i.e. a spatial convolution performed independently over every channel of an input, followed by a pointwise convolution, i.e. a regular convolution with 1x1 windows, projecting the channels computed by the depthwise convolution onto a new channel space.

The depthwise separable convolution operation should not be confused with spatially separable convolutions, which are also often called "separable convolutions" in the image processing community.

Their mathematical formulation is as follow (we use to denote the element-wise product): DISPLAYFORM0 Thus, the fundamental idea behind depthwise separable convolutions is to replace the feature learning operated by regular convolutions over a joint "space-cross-channels realm" into two simpler steps, a spatial feature learning step, and a channel combination step.

This is a powerful simplification under the oft-verified assumption that the 2D or 3D inputs that convolutions operate on will feature both fairly independent channels and highly correlated spatial locations.

A deep neural network forms a chain of differentiable feature learning modules, structured as a discrete set of units, each trained to learn a particular feature.

These units are subsequently composed and combined, gradually learning higher and higher levels of feature abstraction with increasing depth.

Of significance is the availability of dedicated feature pathways that are merged together later in the network; this is one property enabled by depthwise separable convolutions, which define independent feature pathways that are later merged.

In contrast, regular convolutional layers break this creed by learning filters that must simultaneously perform the extraction of spatial features and their merger into channel dimensions; an inefficient and ineffective use of parameters.

Grouped convolutions (or "sub-separable convolutions") are an intermediary step between regular convolutions and depthwise separable convolutions.

They consist in splitting the channels of an input into several non-overlapping segments (or "groups"), performing a regular spatial convolution over each segment independently, then concatenating the resulting feature maps along the channel axis.

Convolution type Parameters and approximate floating point operations per position The theoretical justifications for replacing regular convolution with depthwise separable convolution, as well as the strong gains achieved in practice by such architectures, are a significant motivation for applying them to 1D sequence-to-sequence models.

DISPLAYFORM1 The key gains from separability can be seen when comparing the number of parameters (which in this case corresponds to the computational cost too) of separable convolutions, group convolutions, and regular convolutions.

Assume we have c channels and filters (often c = 1000 or more) and a receptive field of size k (often k = 3 but we will use k upto 63).

The number of parameters for a regular convolution, separable convolution, and group convolution with g groups is: DISPLAYFORM2

As can be seen above, the size (and cost) of a separable convolution with c channels and a receptive field of size k is k ?? c + c 2 .

When k is small compared to c (as is usuallty the case) the term c 2 dominates, which raises the question how it could be reduced.

We use the idea from group convolutions and the recent separable-LSTM paper BID12 to further reduce this size by factoring the final 1 ?? 1 convolution, and we call the result a super-separable convolution.

We define a super-separable convolution (denoted SuperSC) with g groups as follows.

Applied to a tensor x, we first split x on the depth dimension into g groups, then apply a separable convolution to each group separately, and then concatenate the results on the depth dimension.

DISPLAYFORM0 where x 1 , . . . , x g is x split on the depth axis and W Note that a super-separable convolution doesn't allow channels in separate groups to exchange information.

To avoid making a bottleneck of this kind, we use stack super-separable convolutions in layer with co-prime g. In particular, in our experiments we always alternate g = 2 and g = 3.

Filter dilation, as introduced in BID24 , is a technique for aggregating multiscale information across considerably larger receptive fields in convolution operations, while avoiding an explosion in parameter count for the convolution kernels.

It has been presented in (Kalchbrenner et al., When dilated convolution layers are stacked such that consecutive layers' dilation values have common divisors, an issue similar to the checkerboard artifacts in deconvolutions BID14 appears.

Uneven filter coverage results in dead zones where filter coverage is reduced (as displayed in the plaid-like appearance of FIG1 in BID24 ).

Choosing dilation factors that are co-prime can indeed offer some relief from these artifacts, however, it would be preferable to do away with the necessity for dilation entirely.

The purpose of filter dilation is to increase the receptive field of the convolution operation, i.e. the spatial extent from which feature information can be gathered, at a reasonable computational cost.

A similar effect would be achieved by simply using larger convolution windows.

Besides, the use of larger windows would avoid an important shortcoming of filter dilation, unequal convolutional coverage of the input space.

Notably, the use of depthwise separable convolutions in our network in place of regular convolutions makes each convolution operation significantly cheaper (we are able to cut the number of non-embedding model parameters by half), thus lifting the computational and memory limitations that guided the development of filter dilation in the first place.

In our experiments, we explore the trade-off between using lower dilation rates and increasing the size of the convolution windows for our depthwise separable convolution layers.

In contrast to the conclusions drawn in WaveNet and ByteNet, we find that the computational savings brought on by depthwise separable convolutions allow us to do away with dilation entirely.

In fact, we observe no benefits of dilations: our best models feature larger filters and no dilation (see TAB1 ).

A comparison of the parameter count for different convolution operations is found in TAB0 .

Here we present the model we use for our experiments, called SliceNet in reference to the way separable convolutions operate on channel-wise slices of their inputs.

Our model follows the convolutional autoregressive structure introduced by ByteNet BID11 ), WaveNet (van den Oord et al., 2016a and PixelCNN (van den Oord et al., 2016b) .

Inputs and outputs are embedded into the same feature depth, encoded by two separate sub-networks and concatenated before being fed into a decoder that autoregressively generates each element of the output.

At each step, the autoregressive decoder produces a new output prediction given the encoded inputs and the encoding of the existing predicted outputs.

The encoders and the decoder (described in Section 3.3) are constructed from stacks of convolutional modules (described in Section 3.1) and attention (described in Section 3.2) is used to allow the decoder to get information from the encoder.

To perform local computation, we use modules of convolutions with ReLU non-linearities and layer normalization.

A module of convolutions gets as input a tensor of shape [sequence length, feature channels] and returns a tensor of the same shape.

Each step in our module consist of three components: a ReLU activation of the inputs, followed by a depthwise separable convolution, followed by layer normalization.

Layer normalization BID1 acts over the h hidden units of the layer below, computing layer-wise statistics and normalizing accordingly.

These normalized units are then scaled and shifted by scalar learned parameters G and B respectively, producing the final units to be activated by a non-linearity: DISPLAYFORM0 where the sum are taken only over the last (depth) dimension of x, and G and B are learned scalars.

A complete convolution step with kernel size K and dilation D is defined as:ConvStep DISPLAYFORM1 where W p and W d are fresh sets of trainable weights that we omit from the notation for clarity.

The convolutional steps are composed into modules by stacking them and adding residual connections as depicted in FIG1 .

We use stacks of four convolutional steps with two skip-connections between the stack input and the outputs of the second and fourth convolutional steps: DISPLAYFORM2 ConvModule(x) = dropout(hidden 4 (x), 0.5) during training hidden 4 (x) otherwiseFigure 2: The ConvModule architecture described in Section 3.1.

We vary the convolution sizes and dilations; see Section 3 for details on the architecture and Section 5 for the variations we study.

ConvModules are used in stacks in our module, the output of the last feeding into the next.

We denote a stack with n modules by ConvModule n .

For attention, we use a simple inner-product attention that takes as input two tensors: source of shape [m, depth] and target of shape [n, depth].

The attention mechanism computes the feature vector similarities at each position and re-scales according to the depth: DISPLAYFORM0 To allow the attention to access positional information, we add a signal that carries it.

We call this signal the timing, it is a tensor of any shape [k, depth] defined by concatenating sine and cosine functions of different frequencies calculated upto k: timing(t, 2d) = sin(t/10000 2d/depth ) timing(t, 2d + 1) = cos(t/10000 2d/depth ) Our full attention mechanism consists of adding the timing signal to the targets, performing two convolutional steps, and then attending to the source: DISPLAYFORM1 (attention 1 (t)))

As previously discussed, the outputs of our model are generated in an autoregressive manner.

Unlike RNNs, autoregressive sequence generation depends not only on the previously generated output, but potentially all previously generated outputs.

This notion of long term dependencies has proven highly effect in NMT before.

By using attention, establishing long term dependencies has been shown to significantly boost task performance of RNNs for NMT BID3 .

Similarly, a convolutional autoregressive generation scheme offer large receptive fields over the inputs and past outputs, capable of establishing these long term dependencies.

Below we detail the structure of the InputEncoder, IOMixer and Decoder.

The OutputEmbedding simply performs a learning-embedding look-up.

We denote the concatenation of tensors a and b along the d DISPLAYFORM0 InputEncoder(x) = ConvModule 6 (x + timing) DISPLAYFORM1 Decoder(x) = AttnConvModule 4 (x, InputEncoder(inputs))

Machine translation using deep neural networks achieved great success with sequence-to-sequence models BID18 BID2 BID4 ) that used recurrent neural networks (RNNs) with long short-term memory (LSTM, BID7 ) cells.

The basic sequence-to-sequence architecture is composed of an RNN encoder which reads the source sentence one token at a time and transforms it into a fixed-sized state vector.

This is followed by an RNN decoder, which generates the target sentence, one token at a time, from the state vector.

While a pure sequence-to-sequence recurrent neural network can already obtain good translation results BID18 BID4 , it suffers from the fact that the whole input sentence needs to be encoded into a single fixed-size vector.

This clearly manifests itself in the degradation of translation quality on longer sentences and was overcome in BID2 by using a neural model of attention.

We use a simplified version of this neural attention mechanism in SliceNet, as introduced above.

Convolutional architectures have been used to obtain good results in word-level neural machine translation starting from BID10 and later in BID13 .

These early models used a standard RNN on top of the convolution to generate the output.

The state of this RNN has a fixed size, and in the first one the sentence representation generated by the convolutional network is also a fixed-size vector, which creates a bottleneck and hurts performance, especially on longer sentences, similarly to the limitations of RNN sequence-to-sequence models without attention BID18 BID4 ) discussed above.

Fully convolutional neural machine translation without this bottleneck was first achieved in BID9 and BID11 .

The model in BID9 ) (Extended Neural GPU) used a recurrent stack of gated convolutional layers, while the model in BID11 ) (ByteNet) did away with recursion and used left-padded convolutions in the decoder.

This idea, introduced in WaveNet (van den Oord et al., 2016a), significantly improves efficiency of the model.

The same technique is used in SliceNet as well, and it has been used in a number of neural translation models recently, most notably in BID6 where it is combined with an attention mechanism in a way similar to SliceNet.

Depthwise separable convolutions were first studied by Sifre BID17 ) during a 2013 internship at Google Brain, and were first introduced in an ICLR 2014 presentation BID21 .

In 2016, they were demonstrated to yield strong results on large-scale image classification in Xception BID5 , and in 2017 they were shown to lead to small and parameter-efficient image classification models in MobileNets BID8 .

We design our experiments with the goal to answer two key questions:??? What is the performance impact of replacing convolutions in a ByteNet-like model with depthwise separable convolutions?

???

What is the performance trade-off of reducing dilation while correspondingly increasing convolution window size?In addition, we make two auxiliary experiments:??? One experiment to test the performance of an intermediate separability point in-between regular convolutions and full depthwise separability: we replace depthwise separable convolutions with grouped convolutions (sub-separable convolutions) with groups of size 16.??? One experiment to test the performance impact of our newly-introduced super-separable convolutions compared to depthwise separable convolutions.

We evaluate all models on the WMT English to German translation task and use newstest2013 evaluation set for this purpose.

For two best large models, we also provide results on the standard test set, newstest2014, to compare with other works.

For tokenization, we use subword units, and follow the same tokenization process as BID15 .

All of our experiments are implemented using the TensorFlow framework BID0 .

A comparison of our different models in terms of parameter count and Negative Log Perplexity as well as per-token Accuracy on our task are provided in TAB1 .

The parameter count (and computation cost) of the different types of convolution operations used was already presented in TAB0 .

Our experimental results allow us to draw the following conclusions:??? Depthwise separable convolutions are strictly superior to regular convolutions in a ByteNetlike architecture, resulting in models that are more accurate while requiring fewer parameters and being computationally cheaper to train and run.

BID11 23.8 -GNMT BID23 24.6 278 M ConvS2S BID6 25.1 -GNMT+Mixture of Experts BID16 26.0 8700 M Transformer (Vaswani et al., 2017) 28.4 213 M Table 3 : Performance of our larger models compared to best published results.groups as small as possible, tending to full depthwise separable convolutions) is preferable in this setup, this further confirming the advantages of depthwise separable convolutions.??? The need for dilation can be completely removed by using correspondingly larger convolution windows, which is made computationally tractable by the use of depthwise separable convolutions.??? The newly-introduced super-separable convolution operation seems to offer an incremental performance improvement.

Finally, we run two larger models with a design based on the conclusions drawn from our first round of experiments: a SliceNet model which uses depthwise separable convolutions and a SliceNet model which uses super-separable convolutions, with significantly higher feature depth in both cases.

We achieve results that surpass all previously reported models except for the recent Transformer (Vaswani et al., 2017) , as shown in Table 3 , where we also include previously reported results for comparison.

For getting the BLEU, we used a beam-search decoder with a beam size of 4 and a length penalty tuned on the evaluation set (newstest2013).

In this work, we introduced a new convolutional architecture for sequence-to-sequence tasks, called SliceNet, based on the use of depthwise separable convolutions.

We showed how this architecture achieves results beating not only ByteNet but also the previous best Mixture-of-Experts models while using over two times less (non-embedding) parameters and floating point operations than ByteNet.

Additionally, we have shown that filter dilation, previously thought to be a key component of successful convolutional sequence-to-sequence architectures, was not a requirement.

The use of depthwise separable convolutions makes much larger convolution window sizes possible, and we found that we could achieve the best results by using larger windows instead of dilated filters.

We have also introduced a new type of depthwise separable convolution, the super-separable convolution, which shows incremental performance improvements over depthwise separable convolutions.

Our work is one more point on a significant trendline started with Xception and MobileNets, that indicates that in any convolutional model, whether for 1D or 2D data, it is possible to replace convolutions with depthwise separable convolutions and obtain a model that is simultaneously cheaper to run, smaller, and performs a few percentage points better.

This trend is backed by both solid theoretical foundations and strong experimental results.

We expect our current work to play a significant role in affirming and accelerating this trend.

We only experimented on translation, but we expect that our results will apply to other sequence-to-sequence tasks and we hope to see depthwise separable convolutions replace regular convolutions in more and more use cases in the future.

<|TLDR|>

@highlight

Depthwise separable convolutions improve neural machine translation: the more separable the better.

@highlight

This paper proposes to use depthwise separable convolution layers in a fully convolutional neural machine translation model, and introduces a new super-separable convolution layer which further reduces computational cost.