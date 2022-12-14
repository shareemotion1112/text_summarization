Deep learning yields great results across many fields, from speech recognition, image classification, to translation.

But for each problem, getting a deep model to work well involves research into the architecture and a long period of tuning.



We present a single model that yields good results on a number of problems spanning multiple domains.

In particular, this single model is trained concurrently on ImageNet, multiple translation tasks, image captioning (COCO dataset), a speech recognition corpus, and an English parsing task.



Our model architecture incorporates building blocks from multiple domains.

It contains convolutional layers, an attention mechanism, and sparsely-gated layers.



Each of these computational blocks is crucial for a subset of the tasks we train on.

Interestingly, even if a block is not crucial for a task, we observe that adding it never hurts performance and in most cases improves it on all tasks.



We also show that tasks with less data benefit largely from joint training with other tasks, while performance on large tasks degrades only slightly if at all.

Recent successes of deep neural networks have spanned many domains, from computer vision BID16 to speech recognition BID7 and many other tasks.

Convolutional networks excel at tasks related to vision, while recurrent neural networks have proven successful at natural language processing tasks, e.g., at machine translation BID31 BID2 .

But in each case, the network was designed and tuned specifically for the problem at hand.

This limits the impact of deep learning, as this effort needs to be repeated for each new task.

It is also very different from the general nature of the human brain, which is able to learn many different tasks and benefit from transfer learning.

The natural question arises:Can we create a unified deep learning model to solve tasks across multiple domains?The question about multi-task models has been studied in many papers in the deep learning literature.

Natural language processing models have been shown to benefit from a multi-task approach a long time ago BID5 , and recently multi-task machine translation models (MinhThang Luong, 2015) have even been shown to exhibit zero-shot learning when trained on multiple languages (Melvin Johnson, 2016) .

Speech recognition has also been shown to benefit from multi-task training BID27 , as have some vision problems, such as facial landmark detection BID36 .

But all these models are trained on other tasks from the same domain: translation tasks are trained with other translation tasks, vision tasks with other vision tasks, speech tasks with other speech tasks.

Multi-modal learning has been shown to improve learned representations in the unsupervised setting BID22 and when used as a-priori known unrelated tasks BID24 .

But no competitive multi-task multi-modal model has been proposed, so the above question remains unanswered.

In this work, we take a step toward positively answering the above question by introducing the MultiModel architecture, a single deep-learning model that can simultaneously learn multiple tasks from various domains.

Concretely, we train the MultiModel simultaneously on the following 8 corpora:Code available at redacted.

(1) WSJ speech corpus (Consortium et al., 1994 ), used for sentence-level speech recognition.(2) ImageNet dataset BID25 , used for image classification.

(3) COCO image captioning dataset BID17 , used for image captioning.

(4) WSJ parsing dataset BID18 , used for constituency parsing.

These corpora were chosen as they are commonly used for machine learning the respective tasks: speech-to-text, image classification, captioning, parsing and translation.

The model learns all of these tasks and achieves good performance: not state-of-the-art at present, but above many task-specific models studied in recent past (see the Section 3 for details).

FIG0 illustrates some decodes taken directly from the model: it is clear that it can caption images, categorize them, translate to French and German and construct parse trees.

While the MultiModel is only a first step and will be improved in the future, two key insights are crucial to making it work at all and are our main contributions.

Small modality-specific sub-networks convert into a unified representation and back from it.

To allow training on input data of widely different sizes and dimensions, such as images, sound waves and text, we need sub-networks to convert inputs into a joint representation space.

We call these sub-networks modality nets as they are specific to each modality (images, speech, text) and define transformations between these external domains and a unified representation.

We design modality nets to be computationally minimal, promoting heavy feature extraction and ensuring that the majority of computation is performed within the domain-agnostic body of the model.

Since our model is auto-regressive, modality nets need to both convert the inputs into the unified representation and later convert from this representation into the output space.

Two design decisions were important:??? The unified representation is variable-size.

While a fixed-size representation is tempting and easier to implement, it creates a bottleneck and limits the performance of the model.

??? Different tasks from the same domain share modality nets.

We avoid creating a sub-network for every task, and prefer only to create one for every input modality.

For example, all translation tasks share the same modality-net (and vocabulary), no matter for which language pair.

This encourages generalization across tasks and allows to add new tasks on the fly.

Computational blocks of different kinds are crucial for good results on various problems.

The body of the MultiModel incorporates building blocks from mutiple domains.

We use depthwiseseparable convolutions, an attention mechanism, and sparsely-gated mixture-of-experts layers.

These blocks were introduced in papers that belonged to different domains and were not studied before on tasks from other domains.

For example, separable convolutions were introduced in the Xception Figure 2 : The MultiModel, with modality-nets, an encoder, and an autoregressive decoder.architecture BID4 and were not applied to text or speech processing before.

On the other hand, the sparsely-gated mixture-of-experts BID29 had been introduced for language processing tasks and has not been studied on image problems.

We find that each of these mechanisms is indeed crucial for the domain it was introduced, e.g., attention is far more important for languagerelated tasks than for image-related ones.

But, interestingly, adding these computational blocks never hurts performance, even on tasks they were not designed for.

In fact we find that both attention and mixture-of-experts layers slightly improve performance of MultiModel on ImageNet, the task that needs them the least.

The MultiModel consists of a few small modality-nets, an encoder, I/O mixer, and an autoregressive decoder, as depicted in Figure 2 .

As already said above, the encoder and decoder are constructed using 3 key computational blocks to get good performance across different problems:(1) Convolutions allow the model to detect local patterns and generalize across space.(2) Attention layers allow to focus on specific elements to improve performance of the model.

(3) Sparsely-gated mixture-of-experts gives the model capacity without excessive computation cost.

We start by describing the architecture of each of these 3 blocks and then introduce the encoder, decoder and the architecture of our modality-nets.

To perform local computation, we use blocks of convolutions with ReLU non-linearities and normalization.

A block of convolutions gets as input a tensor of shape [batch size, sequence length, feature channels] and returns a tensor of the same shape, processed as follows.

For convolution operations, we use depthwise separable convolutions, studied for images in BID4 , in a way similar to BID12 .

Depthwise separable convolutions are a parameterand computationally-efficient variant of the traditional convolution.

They are defined by a convolution on each feature channel separately, followed by a pointwise convolution to project to the desired feature depth.

We refer the reader to BID4 for a complete definition; here we will denote a depthwise separable convolution with weights W h??w corresponding to f kernels of size h ?? w applied to an input tensor x with stride s and dilated by a factor d (see BID35 ) as SepConv d,s,f (W, x).

Note that subscripts for stride, dilation and output size are omitted when dilation d or stride s are equal to 1, or output size f is equal to the input's feature depth.

We use convolutions in blocks that consist of three components: a ReLU activation of the inputs, followed by a SepConv, followed by layer normalization.

Layer normalization BID1 acts over the h hidden units of the layer below, computing layer-wise statistics for each batch example and normalizing accordingly.

These normalized units are then scaled and shifted by scalar learned parameters G and B respectively, producing the final units to be activated by a non-linearity.

The complete convolution step is therefore defined as: DISPLAYFORM0 The convolutional steps are composed into blocks by stacking them and adding residual connections BID9 as depicted in FIG3 .

We use stacks of four convolutional blocks with two skip-connections between the stack input and the outputs of the second and fourth convolutional steps, and with the first two having 3 ?? 1 kernels and the next two having 15 ?? 1 kernels, with the final one dilated by 8 to provide a wide receptive field.

We also add 40% dropout at the end of each block, so the complete block is defined as follows: DISPLAYFORM1

For attention, we use a multi-head dot-product attention mechanism inspired by BID2 and similar to (Ashish Vaswani, 2017) , as depicted in FIG3 .

The inputs to the attention layer are two tensors: a source tensor and a target tensor both with the shape [batch size, sequence length, feature channels]

The target tensor is additively composed with a timing signal and mixed using two convolutional blocks.

This mixed tensor is then self-attended using a multi-head dot-product attention, which is a dot-product attention with inputs split into g = 8 separate tensors representing each attention head, as shown in FIG3 .

The timing signals are the main difference between this attention mechanism and the ones used previously.

They allow this content-based attention to focus based on their position.

They are constructed by concatenating sine and cosine curves: The source tensor is finally passed through two different pointwise convolutions to generate the memory keys K and values V and the query keys, memory keys and memory values are used to apply the attention mechanism between the self-attended target and the source (see FIG3 ).

DISPLAYFORM0

We use sparsely-gated mixture-of-experts layers of the same kind as introduced in BID29 : A mixture-of-experts layer consists of a number of simple feed-forward neural networks (experts) and a trainable gating network which selects a sparse combination of the experts to process each input.

We refer the reader to BID29 for details as we use exactly the architecture described there.

In particular, during training we select k = 4 experts out of the whole expert pool and add the additional load-balancing cost as in BID29 .

In each of the two mixture-of-experts layers in our model, we use a pool of 240 experts when training on 8 problems jointly, and 60 experts when training on each problem separately.

The body of the MultiModel consists of 3 parts: the encoder that only processes the inputs, the mixer that mixes the encoded inputs with previous outputs (autoregressive part), and a decoder that processes the inputs and the mixture to generate new outputs.

The encoder, mixer and decoder are structured similarly to previous fully convolutional sequence to sequence models such as ByteNet or WaveNet (van den Oord et al., 2016), but differ in the computational blocks that are used.

We depict their architecture in FIG3 .

As can be seen there, the encoder consists of 6 repeated convolutional blocks (described before) with a mixture-of-experts layer in the middle.

The mixer consists of an attention block and 2 convolutional blocks.

The decoder consists of 4 blocks of convolutions and attention, with a mixture-of-experts layer in the middle.

Crucially, the convolutions in the mixer and decoder are padded on the left, so they can never access any information in the future.

This allows the model to be autoregressive, and this convolutional autoregressive generation scheme offers large receptive fields over the inputs and past outputs, which are capable of establishing long term dependencies.

To allow the decoder to produce outputs for different tasks even with the same modality, we always start decoding with a command-token, such as To-English or To-Parse-Tree.

We learn an embedding vector corresponding to each of the tokens during training.

We have 4 modality nets, for language (text data), images, audio, and categorical data.

For all predictions, we use the cross-entropy loss, per subword-unit on text, per category on classification.

Our language-based data is all tokenized using the same vocabulary with 8k subword-units, following the method from BID28 .

The language input modality takes a sequence of tokens ending in a termination token.

This sequence of tokens is mapped to the correct dimensionality for the body using a learned embedding.

On the output side, the language modality takes the decoded output of the body and performs a learned linear mapping, followed by a Sof tmax, resulting in a probability distribution over the token vocabulary.

DISPLAYFORM0

The image input modality is analogous to the Xception entry flow BID4 .

The input image's feature depth is gradually deepened using residual convolution blocks which we call ConvRes and define as follows: DISPLAYFORM0 DISPLAYFORM1

The categorical output modality is analogous to the Xception exit flow BID4 .

If the network inputs are two-dimensional data such as image or spectral audio data, then the one-dimensional output from the model body is first reshaped into two-dimensions again, followed by progressive down-sampling: DISPLAYFORM0 GlobalAvgP ool denotes a mean taken across all spatial and temporal dimensions.

We accept audio input in the form of a 1-dimensional waveform over time BID8 BID23 BID26 or as a 2-dimensional spectrogram.

Both the waveform and spectral input modalities use a stack of 8 ConvRes blocks from the ImageInputM odality (Section 2.5.2).

The i th block has the form: l i = ConvRes(l i???1 , 2 i ).

The spectral modality does not perform any striding along the frequency bin dimension, preserving full resolution in the spectral domain.

The modalities of the MultiModel allows to perform a training step on a batch of data from any of the 8 tasks we consider.

For example, when making a training step on a batch of translation data, only the language modality sub-network will be activated.

Training will then update the parameters of the language modality and all shared parameters, i.e., those in input encoder, mixer and decoder.

MultiModel can be trained on a single machine, but we used distributed training for the multi-task runs.

When training jointly on 8 tasks, we had a separate worker training on each task, while the shared parameters of the model were on a parameter server and were updated asynchronously.

When training on a single task, we used only a single worker training for a similar number of steps.

In all training runs report below we used the same set of hyper-parameters and the Adam optimizer BID15 with gradient clipping.

We will release the implementation as open-source together with the details of our setup and all used hyper-parameters.

The MultiModel architecture draws from eariler encoder-decoder architectures applied to neural machine translation.

Earlier sequence-to-sequence models for translation BID31 BID2 BID32 Vaswani et al., 2017) .memory cells BID10 ).

Convolutional architectures yielded good results on word-level neural machine translation starting from BID13 and later in BID20 .

These early models used a standard RNN on top of the convolution to generate the output and had a bottleneck there that hurt performance, especially on longer sentences, similarly to the limitations of RNN sequence-to-sequence models without attention BID31 .

Fully convolutional neural machine translation without this bottleneck was presented in BID11 .

The model in BID11 ) (Extended Neural GPU) used a recurrent stack of gated convolutional layers, while the model in ) (ByteNet) did away with recursion and used left-padded convolutions in the decoder.

This idea, introduced in WaveNet (van den and also used in MultiModel (see above) significantly improves efficiency.

Depthwise separable convolutions were first studied by Sifre BID30 ) and later they were used to get good results on large-scale image classification with Xception BID4 .

We implemented the MultiModel architecture described above using TensorFlow and trained it in a number of configurations.

We focused our experiments so as to answer the following questions:(1) How far is the MultiModel trained on 8 tasks simultaneously from state-of-the-art results?(2) How does training on 8 tasks simultaneously compare to training on each task separately?(3) How do the different computational blocks discussed above influence different tasks?In answering the above questions, we don't always consider all 8 problems.

Especially the 4 translation problems behave very similarly, so we decided to not include them all in each comparison but we focused on the more varied problems instead.

To answer question (1) , we compare the performance of the 8-problem MultiModel with state-of-theart results in Table 1 .

We use the standard top-5 accuracy metric for ImageNet and the standard BLEU metric for translation (scored with MOSES on newstest2014 while newstest2013 was used as the development set).

We did not invest much time yet in tuning hyper-parameters of the MultiModel, so we believe that the difference seen there will become much smaller with more tuning.

The results we achieve are similar to the ones task-specific models get without heavy tuning, e.g., on English-French translation we improve on the recent Extended Neural GPU results BID11 .To answer question (2) , we compare the MultiModel trained jointly with MultiModel trained separately just on a single task.

Since we are comparing different instantiations of the same model, we report two internal metrics: the negative log-perplexity and per-token accuracy (measured on the development set).

As can be seen from the results in TAB3 , the joint 8-problem model performs similarly to single-model on large tasks, and better, sometimes significantly, on tasks where less data is available, such as parsing.

The large improvement on parsing seen in TAB3 is not that surprising taking into account the large number of text data in translation tasks.

But we were curious if training parsing just with ImageNet, a seemingly unrelated task, would also bring any improvements.

This is indeed the case, as can be seen in Table 3 .

The difference in performance is significant, and since we use both dropout and early stopping, we conjecture that it is not related to over-fitting.

Rather, it seems, there are computational primitives shared between different tasks that allow for some transfer learning even between such seemingly unrelated tasks as ImageNet and parsing.

Table 3 : Results on training parsing alone, with ImageNet, and with 8 other tasks.

We report log-perplexity, per-token accuracy, and the percentage of fully correct parse trees.

To answer question (3), we check how training without the mixture-of-experts layers or without the attention mechanism influences performance on different problems.

Since both these mechanisms were designed with machine translation in mind, we check the English-French translation.

But we also include ImageNet, since this is the problem that stands the least to benefit from those blocks.

In fact, one could expect that removing these blocks will improve performance on ImageNet alone if they were truly useless for this task.

In contrast, we see in TAB5 that these blocks either don't affect or slightly improve performance.

This leads us to conclude that mixing different computation blocks is in fact a good way to improve performance on many various tasks.

<|TLDR|>

@highlight

Large scale multi-task architecture solves ImageNet and translation together and shows transfer learning.