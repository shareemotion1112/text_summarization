Transformers have achieved state-of-the-art results on a variety of natural language processing tasks.

Despite good performance, Transformers are still weak in long sentence modeling where the global attention map is too dispersed to capture valuable information.

In such case, the local/token features that are also significant to sequence modeling are omitted to some extent.

To address this problem, we propose a Multi-scale attention model (MUSE) by concatenating attention networks with convolutional networks and position-wise feed-forward networks to explicitly capture local and token features.

Considering the parameter size and computation efficiency, we re-use the feed-forward layer in the original Transformer and adopt a lightweight dynamic convolution as implementation.

Experimental results show that the proposed model achieves substantial performance improvements over Transformer, especially on long sentences, and pushes the state-of-the-art from 35.6 to 36.2 on IWSLT 2014  German to English translation task,  from 30.6 to 31.3 on  IWSLT 2015 English to Vietnamese translation task.

We also reach the state-of-art performance on  WMT 2014 English to French translation dataset, with a BLEU score of 43.2.

In recent years, Transformer has been remarkably adept at sequence learning tasks like machine translation (Vaswani et al., 2017; Dehghani et al., 2018 ), text classification (Devlin et al., 2018; , language modeling (Sukhbaatar et al., 2019b; , etc.

It is solely based on an attention mechanism that captures global dependencies between input tokens, dispensing with recurrence and convolutions entirely.

The key idea of the self-attention mechanism is updating token representations based on a weighted sum of all input representations.

However, recent research (Tang et al., 2018) has shown that the Transformer has surprising shortcomings in long sequence learning, exactly because of its use of self-attention.

As shown in Figure  1 (a), in the task of machine translation, the performance of Transformer drops with the increase of the source sentence length, especially for long sequences.

The reason is that the attention can be over-concentrated and disperse, as shown in Figure 1 (b), and only a small number of tokens are represented by attention.

It may work fine for shorter sequences, but for longer sequences, it causes insufficient representation of information and brings difficulty for the model to comprehend the source information intactly.

In recent work, local attention that constrains the attention to focus on only part of the sequences (Child et al., 2019; Sukhbaatar et al., 2019a ) is used to address this problem.

However, it costs self-attention the ability to capture long-range dependencies and also does not demonstrate effectiveness in sequence to sequence learning tasks.

To build a module with both inductive bias of local and global context modelling in sequence to sequence learning, we hybrid self-attention with convolution and present Parallel multi-scale attention called MUSE.

It encodes inputs into hidden representations and then applies self-attention and depth-separable convolution transformations in parallel.

The convolution compensates for the in- The left figure shows that the performance drops largely with the increase of sentence length on the De-En dataset.

The right figure shows the attention map from the 3-th encoder layer.

As we can see, the attention map is too dispersed to capture sufficient information.

For example, "[EOS]", contributing little to word alignment, is surprisingly over attended.

sufficient use of local information while the self-attention focuses on capturing the dependencies.

Moreover, this parallel structure is highly extensible, i.e., new transformations can be easily introduced as new parallel branches, and is also favourable to parallel computation.

The main contributions are summarized as follows:

• We find that the attention mechanism alone suffers from dispersed weights and is not suitable for long sequence representation learning.

The proposed method tries to address this problem and achieves much better performance on generating long sequence.

• We propose a parallel multi-scale attention and explore a simple but efficient method to successfully combine convolution with self-attention all in one module.

• MUSE outperforms all previous models with same training data and the comparable model size, with state-of-the-art BLEU scores on three main machine translation tasks.

• The proposed method enables parallel representation learning.

Experiments show that the inference speed can be increased by 31% on GPUs.

Like other sequence-to-sequence models, MUSE also adopts an encoder-decoder framework.

The encoder takes a sequence of word embeddings (x 1 , · · · , x n ) as input where n is the length of input.

It transfers word embeddings to a sequence of hidden representation z = (z 1 , · · · , z n ).

Given z, the decoder is responsible for generating a sequence of text (y 1 , · · · , y m ) token by token.

The encoder is a stack of N MUSE modules.

Residual mechanism and layer normalization are used to connect two adjacent layers.

The decoder is similar to encoder, except that each MUSE module in the decoder not only captures features from the generated text representations but also performs attention over the output of the encoder stack through additional context attention.

Residual mechanism and layer normalization are also used to connect two modules and two adjacent layers.

The key part in the proposed model is the MUSE module, which contains three main parts: selfattention for capturing global features, depth-wise separable convolution for capturing local features, and a position-wise feed-forward network for capturing token features.

The module takes the output of (i − 1) layer as input and generates the output representation in a fusion way:

where "Attention" refers to self-attention, "Conv" refers to dynamic convolution, "Pointwise" refers to a position-wise feed-forward network.

The followings list the details of each part.

Figure 2: Multi-scale attention hybrids point-wise transformation, convolution, and self-attention to learn multi-scale sequence representations in parallel.

We project convolution and self-attention into the same space to learn contextual representations.

We also propose MUSE-simple, a simple version of MUSE, which generates the output representation similar to the MUST model except for that it dose not the include convolution operation:

2.1 ATTENTION MECHANISM FOR GLOBAL CONTEXT REPRESENTATION Self-attention is responsible for learning representations of global context.

For a given input sequence X, it first projects X into three representations, key K, query Q, and value V .

Then, it uses a self-attention mechanism to get the output representation:

Where W O , W Q , W K , and W V are projection parameters.

The self-attention operation σ is the dot-production between key, query, and value pairs:

Note that we conduct a projecting operation over the value in our self-attention mechanism V 1 = V W V here.

We introduce convolution operations into MUSE to capture local context.

To learn contextual sequence representations in the same hidden space, we choose depth-wise convolution (Chollet, 2017) (we denote it as DepthConv in the experiments) as the convolution operation because it includes two separate transformations, namely, point-wise projecting transformation and contextual transformation.

It is because that original convolution operator is not separable, but DepthConv can share the same point-wise projecting transformation with self-attention mechanism.

We choose dynamic convolution (Wu et al., 2019a) , the best variant of DepthConv, as our implementation.

Each convolution sub-module contains multiple cells with different kernel sizes.

They are used for capturing different-range features.

The output of the convolution cell with kernel size k is:

where W V and W out are parameters, W V is a point-wise projecting transformation matrix.

The Depth conv refers to depth convolution in the work of Wu et al. (2019a) .

For an input sequence X, the output O is computed as:

where d is the hidden size.

Note that we conduct the same projecting operation over the input in our convolution mechanism V 2 = XW V here with that in self-attention mechanism.

Shared projection To learn contextual sequence representations in the same hidden space, the projection in the self-attention mechanism V 1 = V W V and that in the convolution mechanism V 2 = XW V is shared.

because the shared projection can project the input feature into the same hidden space.

If we conduct two independent projection here:

are two parameter matrices, we call it as separate projection.

We will analyze the necessity of applying shared projection here instead of separate projection.

We introduce a gating mechanism to automatically select the weight of different convolution cells.

To learn token level representations, MUSE concatenates an self-attention network with a positionwise feed-forward network at each layer.

Since the linear transformations are the same across different positions, the position-wise feed-forward network can be seen as a token feature extractor.

where W 1 , b 1 , W 2 , and b 2 are projection parameters.

We evaluate MUSE on four machine translation tasks.

This section describes the datasets, experimental settings, detailed results, and analysis.

The WMT 2014 English-French translation dataset, consisting of 36M sentence pairs, is adopted as a big dataset to test our model.

We use the standard split of development set and test set.

We use newstest2014 as the test set and use newstest2012 +newstest2013 as the development set.

Following Gehring et al. (2017) , we also adopt a joint source and target BPE factorization with the vocabulary size of 40K.

For medium dataset, we borrow the setup of Vaswani et al. (2017) and adopt the WMT 2014 English-German translation dataset which consists of 4.5M sentence pairs, the BPE vocabulary size is set to 32K.

The test and validation datasets we used are the same as Vaswani et al. (2017) .

IWSLT De-En and En-Vi datasets Besides, we perform experiments on two small IWSLT datasets to test the small version of MUSE with other comparable models.

The IWSLT 2014 German-English translation dataset consists of 160k sentence pairs.

We also adopt a joint source and target BPE factorization with the vocabulary size of 32K.

The IWSLT 2015 English-Vietnamese translation dataset consists of 133K training sentence pairs.

For the En-Vi task, we build a dictionary including all source and target tokens.

The vocabulary size for English is 17.2K, and the vocabulary size for the Vietnamese is 6.8K.

For fair comparisons, we only compare models reported with the comparable model size and the same training data.

We do not compare Wu et al. (2019b) because it is an ensemble method.

We build MUSE-base and MUSE-large with the parameter size comparable to Transformer-base and Transformer-large.

We adopt multi-head attention (Vaswani et al., 2017) as implementation of selfattention in MUSE module.

The number of attention head is set to 4 for MUSE-base and 16 for MUSE-large.

We also add the network architecture built by MUSE-simple in the similar way into the comparison.

MUSE consist of 12 residual blocks for encoder and 12 residual blocks for decoder, the dimension is set to 384 for MUSE-base and 768 for MUSE-large.

The hidden dimension of non linear transformation is set to 768 for MUSE-base and 3072 for MUSE-large.

The MUSE-large is trained on 4 Titan RTX GPUs while the MUSE-base is trained on a single NVIDIA RTX 2080Ti GPU.

The batch size is calculated at the token level, which is called dynamic batching (Vaswani et al., 2017) .

We adopt dynamic convolution as the variant of depth-wise separable convolution.

We tune the kernel size on the validation set.

For convolution with a single kernel, we use the kernel size of 7 for all layers.

In case of dynamic selected kernels, the kernel size is 3 for small kernels and 15 for large kernels for all layers.

The training hyper-parameters are tuned on the validation set.

Vaswani et al. (2017) , we use Adam optimizer with a learning rate of 0.001.

We use the warmup mechanism and invert the learning rate decay with warmup updates of 4K.

For the De-En dataset, we train the model for 20K steps with a batch size of 4K.

The parameters are updated every 4 steps.

The dropout rate is set to 0.4.

For the En-Vi dataset, we train the model for 10K steps with a batch size of 4K.

The parameters are also updated every 4 steps.

The dropout rate is set to 0.3.

We save checkpoints every epoch and average the last 10 checkpoints for inference.

During inference, we adopt beam search with a beam size of 5 for De-En, En-Fr and En-Vi translation tasks.

The length penalty is set to 0.8 for En-Fr according to the validation results, 1 for the two small datasets following the default setting of Ott et al. (2019) .

We do not tune beam width and length penalty but use the setting reported in Vaswani et al. (2017) .

The BLEU 1 metric is adopted to evaluate the model performance during evaluation.

As shown in Table 1 , MUSE outperforms all previously models on En-De and En-Fr translation, including both state-of-the-art models of stand alone self-attention (Vaswani et al., 2017; Ott et al., 2018) , and convolutional models (Gehring et al., 2017; Wu et al., 2019a) .

This result shows that either self-attention or convolution alone is not enough for sequence to sequence learning.

The proposed parallel multi-scale attention improves over them both on En-De and En-Fr.

En-De En-Fr ConvSeq2seq (Gehring et al., 2017) 25.2 40.5 SliceNet 26.1 -Transformer (Vaswani et al., 2017) 28.4 41.0 Weighted Transformer (Ahmed et al., 2017) 28.9 41.4 Layer-wise Coordination (He et al., 2018) 29.1 -Transformer (relative position) (Shaw et al., 2018) 29.2 41.5 Transformer (Ott et al., 2018) 29 Relative position or local attention constraints bring improvements over origin self-attention model, but parallel multi-scale outperforms them.

MUSE can also scale to small model and small datasets, as depicted in Table 2 , MUSE-base pushes the state-of-the-art from 35.7 to 36.3 on IWSLT De-En translation dataset.

It is shown in Table 1 and Table 2 that MUSE-simple which contains the basic idea of parallel multi-scale attention achieves state-of-the-art performance on three tasks.

In this subsection we compare MUSE and its variants on IWSLT 2015 De-En translation to answer the question.

Does concatenating self-attention with convolution certainly improve the model?

To bridge the gap between point-wise transformation which learns token level representations and self-attention which learns representations of global context, we introduce convolution to enhance our multi-scale attention.

As we can see from the first experiment group of Table 3 , convolution is important in the parallel multi-scale attention.

However, it is not easy to combine convolution and self-attention in one module to build better representations on sequence to sequence tasks.

As shown in the first line of both second and third group of Table 3 , simply learning local representations by using convolution or depth-wise separable convolution in parallel with self-attention harms the performance.

Furthermore, combining depth-wise separable convolution (in this work we choose its best variant dynamic convolution as implementation) is even worse than combining convolution.

Conv and self-attention?

We conjecture that convolution and self-attention both learn contextual sequence representations and they should share the point transformation and perform the contextual transformation in the same hidden space.

We first project the input to a hidden representation and perform a variant of depth-wise convolution and self-attention transformations in parallel.

The fist two experiments in third group of Table 3 show that validating the utility of sharing Projection in parallel multi-scale attention, shared projection gain 1.4 BLEU scores over separate projection, and bring improvement of 0.5 BLEU scores over MUSE-simple (without DepthConv).

How much is the kernel size?

Comparative experiments show that the too large kernel harms performance both for DepthConv and convolution.

Since there exists self-attention and point-wise transformations, simply applying the growing kernel size schedule proposed in SliceNet doesn't work.

Thus, we propose to use dynamically selected kernel size to let the learned network decide the kernel size for each layer.

Parallel multi-scale attention brings time efficiency on GPUs The underlying parallel structure (compared to the sequential structure in each block of Transformer) allows MUSE to be efficiently computed on GPUs.

For example, we can combine small matrices into large matrices, and while it does not reduce the number of actual operations, it can be better paralleled by GPUs to speed up computation.

Concretely, for each MUSE module, we first concentrate W Q , W K , W V of selfattention and W 1 of point feed-forward transformation into a single encoder matrix W Enc , and then perform transformation such as self-attention, depth-separable convolution, and nonlinear transformation, in parallel, to learn multi-scale representations in the hidden layer.

W O , W 2 , W out can also be combined a single decoder matrix W Dec .

The decoder of sequence to sequence architecture can be implemented similarly.

In Table 4 , we conduct comparisons to show the speed gains with the aforementioned implementation, the batch size is set to one sample per batch to simulate online inference environment.

Under the settings, where the numbers of parameters are similar for MUSE and Transformer, about 31% increase in inference speed can be obtained.

The experiments use MUSE with 6 MUSE-simple modules and Transformer with 6 base blocks.

The hidden size is set to 512.

It is worth noticing that for the MUSE structure used in the main experiments, ideally a similar speedup can be witnessed if the computing device is powerful enough.

However, such is not the case in our preliminary experiments.

We also need to point out the implementation is far from fully optimized and the results are only meant to demonstrate the feasibility of the procedure.

Inference Speed (tokens/s) Transformer 132 MUSE 173 Acceleration 31% Parallel multi-scale attention generates much better long sequence As demonstrated in Figure 3 , MUSE generates better sequences of various length than self-attention, but it is remarkably adept at generate long sequence, e.g. for sequence longer than 100, MUSE is two times better.

Lower layers prefer local context and higher layers prefer more contextual representations MUSE contains multiple dynamic convolution cells, whose streams are fused by a gated mechanism.

The weight for each dynamic cell is a scalar.

Here we analyze the weight of different dynamic convolution cells in different layers.

Figure 4 shows that as the layer depth increases, the weight of dynamic convolution cells with small kernel sizes gradually decreases.

It demonstrates that lower layers prefer local features while higher layers prefer global features.

It is corresponding to the finding in Ramachandran et al. (2019) .

MUSE not only gains BLEU scores, but also generates more reasonable sentences and increases the translation quality.

We conduct the case study on the De-En dataset and the cases are shown in Table 5 .

In case 1, although the baseline transformer translates many correct words according to the source sentence, the translated sentence is not fluent at all.

It indicates that Transformer does not capture the relationship between some words and their neighbors, such as "right" and "clap".

By contrast, MUSE captures them well by combining local convolution with global selfattention.

In case 2, the cause adverbial clause is correctly translated by MUSE while transformer misses the word "why" and fails to translate it.

Sequence to sequence learning is an important task in machine learning.

It evolves understanding and generating sequence.

Machine translation is the touchstone of sequence to sequence learning.

Traditional approaches usually adopt long-short term memory networks (Sutskever et al., 2014; Ma et al., 2018) to learn the representation of sequences.

However, these models either are built upon auto-regressive structures requiring longer encoding time or perform worse on real-world natural language processing tasks.

Recent studies explore convolutional neural networks (CNN) (Gehring et al., 2017) or self-attention (Vaswani et al., 2017) to support high-parallel sequence modeling and does not require auto-regressive structure during encoding, thus bringing large efficiency improvements.

They are strong at capturing local or global dependencies.

There are several studies on combining self-attention and convolution.

However, they do not surpass both convectional and self-attention mechanisms.

Sukhbaatar et al. (2019b) propose to augment convolution with self attention by directly concentrating them in computer vision tasks.

However, as demonstrated in Table 3 there method does not work for sequence to sequence learning task.

Since state-of-the-art models on question answering tasks still consist on self-attention and do no adopt ideas in QAnet (Yu et al., 2018) .

Both self-attention (Ott et al., 2018) and convolution (Wu et al., 2019a) outperforms Evolved transformer by near 2 BLEU scores on En-Fr translation.

It seems that learning global and local context through stacking self-attention and convolution layers does not beat either self-attention or convolution models.

In contrast, the proposed parallel multiscale attention outperforms previous convolution or self-attention based models on main translation tasks, showing its effectiveness for sequence to sequence learning.

Although the self-attention mechanism has been prevalent in sequence modeling, we find that attention suffers from dispersed weights especially for long sequences, resulting from the insufficient local information.

To address this problem, we present Parallel Multi-scale Attention (MUSE) and MUSE-simple.

MUSE-simple introduces the idea of parallel multi-scale attention into sequence to sequence learning.

And MUSE fuses self-attention, convolution, and point-wise transformation together to explicitly learn global, local and token level sequence representations.

Especially, we find from empirical results that the shared projection plays important part in its success, and is essential for our multiscale learning.

Beyond the inspiring new state-of-the-art results on three machine translation datasets, detailed analysis and model variants also verify the effectiveness of MUSE.

In future work, we would like to explore the detailed effects of shared projection on contextual representation learning.

We are exited about future of parallel multi-scale attention and plan to apply this simple but effective idea to other tasks including image and speech.

A.1 CASE STUDY

Source wenn sie denken, dass die auf der linken seite jazz ist und die, auf der rechten seite swing ist, dann klatschen sie bitte.

Target if you think the one on the left is jazz and the one on the right is swing, clap your hands.

Transformer if you think it's jazz on the left, and those on the right side of the swing are clapping, please.

MUSE if you think the one on the left is jazz, and the one on the right is swing, please clap.

Case 2 Source und deswegen haben wir uns entschlossen in berlin eine halle zu bauen, in der wir sozusagen die elektrischen verhältnisse der insel im maßstab eins zu drei ganz genau abbilden können.

Target and that's why we decided to build a hall in berlin, where we could precisely reconstruct, so to speak, the electrical ratio of the island on a one to three scale.

Transformer and so in berlin, we decided to build a hall where we could sort of map the electrical proportions of the island at scale one to three very precisely.

and that's why we decided to build a hall in berlin, where we can sort of map the electric relationship of the island at the scale one to three very precisely.

Table 5 : Case study on the De-En dataset.

The blue bolded words denote the wrong translation and red bolded words denote the correct translation.

In case 1, transformer fails to capture the relationship between some words and their neighbors, such as "right" and "clap".

In case 2, the cause adverbial clause is correctly translated by MUSE while transformer misses the word "why" and fails to translate it.

<|TLDR|>

@highlight

This paper propose a new model  which combines multi scale information for sequence to sequence learning.