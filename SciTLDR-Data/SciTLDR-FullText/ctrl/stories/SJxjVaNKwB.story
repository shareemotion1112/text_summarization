The recent development of Natural Language Processing (NLP) has achieved great success using large pre-trained models with hundreds of millions of parameters.

However, these models suffer from the heavy model size and high latency such that we cannot directly deploy them to resource-limited mobile devices.

In this paper, we propose MobileBERT for compressing and accelerating the popular BERT model.

Like BERT, MobileBERT is task-agnostic; that is, it can be universally applied to various downstream NLP tasks via fine-tuning.

MobileBERT is a slimmed version of BERT-LARGE augmented with bottleneck structures and a carefully designed balance between self-attentions and feed-forward networks.

To train MobileBERT, we use a bottom-to-top progressive scheme to transfer the intrinsic knowledge of a specially designed Inverted Bottleneck BERT-LARGE teacher to it.

Empirical studies show that MobileBERT is 4.3x smaller and 4.0x faster than original BERT-BASE while achieving competitive results on well-known NLP benchmarks.

On the natural language inference tasks of GLUE, MobileBERT achieves 0.6 GLUE score performance degradation, and 367 ms latency on a Pixel 3 phone.

On the SQuAD v1.1/v2.0 question answering task, MobileBERT achieves a 90.0/79.2 dev F1 score, which is 1.5/2.1 higher than BERT-BASE.

The NLP community has witnessed a revolution of pre-training self-supervised models.

These models usually have hundreds of millions of parameters.

They are trained on huge unannotated corpus and then fine-tuned for different small-data tasks (Peters et al., 2018; Radford et al., 2018; Devlin et al., 2018; Radford et al., 2019; Yang et al., 2019) .

Among these models, BERT (Devlin et al., 2018) , which stands for Bidirectional Encoder Representations from Transformers (Vaswani et al., 2017) , shows substantial accuracy improvements compared to training from scratch using annotated data only.

However, as one of the largest models ever in NLP, BERT suffers from the heavy model size and high latency, making it impractical for resource-limited mobile devices to deploy the power of BERT in mobile-based machine translation, dialogue modeling, and the like.

There have been some works that task-specifically distill BERT into compact models (Turc et al., 2019; Tang et al., 2019; Sun et al., 2019; Tsai et al., 2019) .

To the best of our knowledge, there is not yet any work for building a task-agnostic lightweight pre-trained model, that is, a model that can be fine-tuned on downstream NLP tasks just like what the original BERT does.

In this paper, we propose MobileBERT to fill this gap.

In practice, task-agnostic compression of BERT is desirable.

Task-specific compression needs to first fine-tune the original large BERT model into task-specific teachers and then distill.

Such a process is way more complicated and costly than directly fine-tuning a task-agnostic compact model.

At first glance, it may seem straightforward to obtain a task-agnostic compact version of BERT.

For example, one may just take a narrower or shallower architecture of BERT, and then train it with a prediction loss together with a distillation loss (Turc et al., 2019; Sun et al., 2019) .

Unfortunately, empirical results show that such a straightforward approach results in significant accuracy loss (Turc et al., 2019) .

This may not be that surprising.

It aligns with a well-known observation that shallow networks usually do not have enough representation power while narrow and deep networks are difficult to train.

Our MobileBERT is designed to be as deep as BERT LARGE while each layer is made much narrower via adopting bottleneck structures and balancing between self-attentions and feed- MobileBERT is trained by progressively transferring knowledge from IB-BERT.

forward networks (Figure 1 ).

To train MobileBERT, we use a bottom-to-top progressive scheme to transfer the intrinsic knowledge of a specially designed Inverted Bottleneck BERT LARGE (IB-BERT) teacher to it.

As a pre-trained NLP model, MobileBERT is both storage efficient (w.r.t model size) and computationally efficient (w.r.t latency) for mobile and resource-constrained environments.

Experimental results on several NLP tasks show that while being 4.3?? smaller and 4.0?? faster, MobileBERT can still achieve competitive results compared to BERT BASE .

On the natural language inference tasks of GLUE, MobileBERT can have only 0.6 GLUE score performance degradation with 367 ms latency on a Pixel 3 phone.

On the SQuAD v1.1/v2.0 question answering task, MobileBERT obtains 90.3/80.2 dev F1 score which is 1.5/2.1 higher than BERT BASE .

2 RELATED WORK 2.1 BERT BERT takes the embedding of source tokens as input.

Each building block of BERT contains one Multi-Head self-Attention (MHA) module (Vaswani et al., 2017) and one Feed-Forward Network (FFN) module, which are connected by skip connections.

The MHA module allows the model to jointly attend to information from different subspaces, while the position-wise FFN consists of a two-layer linear transformation with gelu activation (Hendrycks & Gimpel, 2016) , which increase the representational power of the model.

Figure 1 (a) illustrates the original BERT architecture.

In the pre-training stage, BERT is required to predict the masked tokens in sentences (mask language modeling task), as well as whether one sentence is the next sentence of the other (next sentence prediction task).

In the fine-tuning stage, BERT is further trained on task-specific annotated data.

Exploiting knowledge transfer to compress model size was first proposed by Bucilu et al. (2006) .

The idea was then adopted in knowledge distillation (Hinton et al., 2015) , which requires the smaller student network to mimic the class distribution output of the larger teacher network.

Fitnets (Romero et al., 2014) make the student mimic the intermediate hidden layers of the teacher to train narrow and deep networks.

Luo et al. (2016) show that the knowledge of the teacher can also be obtained from the neurons in the top hidden layer.

Similar to our proposed progressive knowledge transfer scheme, Yeo et al. (2018) proposed a sequential knowledge transfer scheme to distill knowledge from a deep teacher into a shallow student in a sequential way.

Zagoruyko & Komodakis (2016) proposed to transfer the attention maps of the teacher on images.

proposed to transfer the similarity of hidden states and word alignment from an autoregressive Transformer teacher to a non-autoregressive student.

Recently, knowledge transfer for BERT has attracted much attention.

Researchers have distilled BERT into smaller pre-trained BERT models (Turc et al., 2019) , an extremely small bi-directional LSTM Tang et al. (2019) , and smaller models on sequence labeling tasks (Tsai et al., 2019) .

Sun et al. (2019) distill BERT into shallower students through knowledge distillation and an additional knowledge transfer of hidden states on multiple intermediate layers.

In contrast to these works, we only use knowledge transfer in the pre-training stage and do not require a fine-tuned teacher for task-specific knowledge in the down-stream tasks.

Moreover, compared to patient knowledge distillation (Sun et al., 2019) which transfers knowledge for all intermediate layers simultaneously to alleviate over-fitting in down-stream task fine-tuning, we design a novel progressive knowledge transfer which eases the pre-training of our compact MobileBERT.

The pre-training of BERT is challenging.

This problem becomes more severe when we pre-train a compact BERT model from scratch (Frankle & Carbin, 2018) .

To tackle this problem, we propose a bottom-to-top progressive knowledge transfer scheme.

Specifically, we first train a wider teacher network that is easier to optimize, and then progressively train the student network from bottom to top, requiring it to mimic the teacher network layer by layer.

In our algorithm, the student and the teacher can be any multi-head attention encoder such as Transformer (Vaswani et al., 2017) , BERT or XLNet .

We take BERT as an example in the following description.

The progressive knowledge transfer is divided into L stages, where L is the number of layers.

Figure  2 illustrates the diagram and algorithm of progressive knowledge transfer.

The idea of progressive transfer is that when training the ( +1) th layer of the student, the th layer is already well-optimized.

As there are no soft target distributions that can be used for the intermediate states of BERT, we propose the following two knowledge transfer objectives, i.e., feature map transfer and attention transfer, to train the student network.

Particularly, we assume that the teacher and the student have the same 1) feature map size, 2) the number of layers, and 3) the number of attention heads.

Since each layer in BERT merely takes the output of the previous layer as input, the most important thing in progressively training the student network is that the feature maps of each layer should be as close as possible to those of the teacher, i.e., well-optimized.

In particular, the mean squared error between the normalized feature maps of the student and the teacher is used as the objective:

where is the index of layers, T is the sequence length, and N is the feature map size.

The layer normalization is added to stabilize the layer-wise training loss.

We also minimize two statistics discrepancies on mean and variance in feature map transfer:

where ?? and ?? 2 represents mean and variance, respectively.

Our empirical studies show that minimizing the statistics discrepancy is helpful when layer normalization is removed from BERT to reduce inference latency (see more discussions in Section 4.3).

The attention mechanism greatly boosts the performance of NLP and becomes a crucial building block in Transformer and BERT.

Many papers (Clark et al., 2019; Jawahar et al., 2019) words.

This motivates us to use self-attention maps from the well-optimized teacher to help the training of the student in augmentation to the feature map transfer.

In particular, we minimize the KL-divergence between the per-head self-attention distributions of the teacher and the student:

where A is the number of attention heads.

Our final progressive knowledge transfer loss L P KT for the th stage is a linear combination of the objectives stated above.

As shown in the right panel of Figure 2 , we progressively train each layer of the student by minimizing the knowledge transfer loss.

In other words, when we train the th layer, we freeze all the trainable parameters in the layers below.

We can somewhat soften the training process as follows.

When training a layer, we further tune the lower layers with a small learning rate rather than entirely freezing them.

Freezing the lower layers can be regarded as a special case of this softened process with the learning rate being zero.

There is no knowledge transfer for the beginning embedding layer and the final classifier.

They are are the same for the student and teacher.

After the progressive knowledge transfer, we further pre-train MobileBERT until convergence.

We use a linear combination of the original masked language modeling (MLM) loss, next sentence prediction (NSP) loss, and the new knowledge distillation loss as our pre-training distillation loss:

where [N ] is the set of masked tokens, P tr (i) and P st (i) are two predicted distributions respectively from the teacher and student model on the masked tokens, and ?? is a hyperparameter in (0, 1).

We do not perform knowledge distillation on the next sentence prediction (NSP) task as it has been shown to be unimportant .

In this section, we present the MobileBERT architecture and the underlining design principle, i.e., how to exploit the benefits of the proposed progressive knowledge transfer.

MobileBERT is a much slimmed version of BERT LARGE .

As illustrated in Figure 1 (c), to align its feature maps with the teacher's, it is augmented with the bottleneck modules (He et al., 2016) , which have additional shortcut connections outside the original non-linear modules.

Through the bottleneck modules, MobileBERT can increase the dimension of its block outputs by a linear transformation, while decreasing the dimension of its block inputs by another linear transformation.

So the intra-block hidden size (hidden size of the original non-linear modules) stays unchanged.

Symmetrically, to align with the student's feature maps, we can also place the inverted bottleneck modules (Sandler et al., 2018) in the BERT LARGE teacher (Figure 1b) .

We refer this variant of BERT LARGE as IB-BERT.

Through the inverted bottleneck modules, we can effectively reduce the feature map size of the teacher without losing its representational power.

We may either only use bottleneck for the student or only the inverted bottleneck for the teacher to align their feature maps.

However, when using both of them, we have a chance to search for a better feature map size for the teacher and student to obtain a more compact student model while not hurting the performance of the teacher.

A problem introduced by the bottleneck structure of MobileBERT is that the balance between selfattentions and feed-forward networks is broken.

In original BERT, the ratio of the parameter numbers in self-attentions and feed-forward networks is always 1:2.

But in the bottleneck structure, the inputs to the self-attentions are from wider feature maps (of inter-block size), while the inputs to the feed-forward networks are from narrower bottlenecks (of intra-block size).

This results in that the self-attentions in MobileBERT will contain more parameters than normally.

Therefore, we propose to use stacked feed-forward networks in MobileBERT to re-balance it.

As illustrated in 1(c), each MobileBERT layer contains one self-attention but several stacked feed-forward networks.

By model latency analysis 1 , we find that layer normalization and gelu activation accounted for a considerable proportion of total latency.

Therefore, we replace them with new operations in our MobileBERT.

Remove layer normalization We replace the layer normalization of a n-channel hidden state h with an element-wise linear transformation:

where ??, ?? ??? R n and ??? denotes the Hadamard product.

Please note that NoNorm has different properties from LayerNorm even in test mode since the original layer normalization is not a linear operation for a batch of vectors.

Use relu activation We replace the gelu activation with simpler relu activation.

We conduct extensive experiments to search good model settings for the IB-BERT teacher and the MobileBERT student.

We replace the original embedding table by a 3-convolution from a smaller embedding table with embedding size 128 to keep the number of embedding parameters in different model settings the same.

We start with SQuAD v1.1 dev F1 score as the metric to measure the performance of different model settings.

Since BERT pre-training is time and resource consuming, in the architecture search stage, we only train each model for 125k steps with 2048 batch size, which halves the training schedule of original BERT (Devlin et al., 2018; You et al., 2019) .

Architecture Search of the Teacher As shrinking the inter-block size can effectively compress the model while maintaining its representational power (Sandler et al., 2018) , our design philosophy for the teacher model is to use as small inter-block hidden size (feature map size) as possible as long as there is no accuracy loss.

Under this guideline, we design experiments to manipulate the inter-block size of a BERT LARGE -sized IB-BERT, and the results are shown in the left panel of Table  1 with labels (a)-(e).

As can be seen, decreasing the inter-block hidden size doesn't damage the performance of BERT until the inter-block size is smaller than 512.

As a result, we choose the IB-BERT LARGE with its inter-block hidden size being 512 as the teacher model.

One may wonder whether we can also shrink the intra-block hidden size of the teacher, as this may bridge the gap between the student and teacher (Mirzadeh et al., 2019) .

We conduct experiments and the results are shown in the left panel of Table 1 with labels (f)-(i).

We can see that when the intra-block hidden size is reduced, the model performance is dramatically worse.

This means that the intra-block hidden size, which represents the representation power of non-linear modules, plays a crucial role in BERT.

Therefore, unlike the inter-block hidden size, we do not shrink the intrablock hidden size of our teacher model.

Besides, by comparing (a) and (f) in Table 1 , we can see that reducing the number of heads from 16 to 4 does not harm the performance of BERT.

This is in line with the observation in the recent literature (Michel et al., 2019; Voita et al., 2019) .

Architecture Search of the Student We seek a compression ratio of 4?? for BERT BASE , so we design a set of MobileBERT models all with approximately 25M parameters but different ratios of the parameter numbers in MHA and FFN to select a good student model.

The right part of Table  1 shows our experimental results.

They have different balances between self-attentions and feedforward networks.

From the table, we can see that the model performance reaches the peak when the ratio of parameters in MHA and FFN is 0.4 ??? 0.6.

This may justify why the original Transformer chooses the parameter ratio of self-attention and feed-forward networks to 0.5.

We choose the architecture with 128 intra-block hidden size and 4 stacked FFNs as the student model in consideration of model accuracy and training efficiency.

We also accordingly set the number of attention heads in the teacher model to 4 in preparation for the progressive knowledge transfer.

Table  2 demonstrates the model settings of our IB-BERT LARGE teacher and MobileBERT student.

Following BERT (Devlin et al., 2018) , we use the BooksCorpus (Zhu et al., 2015) and English Wikipedia as our pre-training data.

To make the IB-BERT LARGE teacher reach the same accuracy as original BERT LARGE , we train IB-BERT LARGE on 256 TPU v3 chips for 500k steps with a batch size of 4096 and LAMB optimizer (You et al., 2019) .

For MobileBERT, we also use the same training schedule.

Besides, progressive knowledge transfer of MobileBERT over 24 layers takes 240k steps, so that each layer of MobileBERT is trained for 10k steps.

For the downstream tasks, all reported results are obtained by simply fine-tuning MobileBERT just like what the original BERT does.

To fine-tune the pre-trained models, we search the optimization hyperparameters in a search space including different batch sizes (16/32/48), learning rates ((1-10) * e-5), and the number of epochs (2-10).

The search space is different from the original BERT because we find that MobileBERT usually needs a larger learning rate and more training epochs in fine-tuning.

We select the model for testing according to their performance on the development (dev) set.

The General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2018 ) is a collection of 9 natural language understanding tasks.

We briefly describe these tasks in Appendix F. Following BERT (Devlin et al., 2018) , we use the final hidden vector corresponding to the first input token as model output, and introduced a new linear classification layer for the final predictions.

We submit the predictions of MobileBERT and MobileBERT without operational optimizations to the online test evaluation system 2 of GLUE to get the test results.

We compare MobileBERT with BERT BASE and a few other state-of-the-art pre-BERT models on the GLUE leaderboard: OpenAI GPT (Radford et al., 2018) and ELMo (Peters et al., 2018) .

We also compare with a recent work on compressing BERT: BERT-PKD (Sun et al., 2019) .

The results are listed in Table 3 .

3 We can see that our MobileBERT is quite competitive with the original BERT BASE .

It outperforms BERT BASE a bit on QNLI and RTE tasks, while the overall GLUE score performance gap is only 0.6.

Moreover, It outperform the strong OpenAI GPT baseline by 0.8 GLUE score with 4.3?? smaller model size.

We also find that the introduced operational optimizations hurt the model performance a bit.

Without these optimizations, MobileBERT can even outperform BERT BASE by 0.2 GLUE score.

SQuAD is a large-scale reading comprehension datasets.

SQuAD1.1 (Rajpurkar et al., 2016) only contains questions that always have an answer in the given context, while SQuAD2.0 (Rajpurkar et al., 2018) contains unanswerable questions.

Following BERT (Devlin et al., 2018) , we treat questions that do not have an answer as having an answer span with start and end at the sentence classification token to fine-tune a MobileBERT on SQuAD2.0.

We evaluate MobileBERT only on the SQuAD dev datasets, as there is nearly no single model submission on SQuAD test leaderboard 4 .

We compare our MobileBERT with BERT BASE and a strong baseline DocQA (Clark & Gardner, 2017) .

As shown in Table 4 , MobileBERT outperforms a large margin over BERT BASE and DocQA.

We notice that MobileBERT also outperforms BERT BASE on QNLI, a question-answering GLUE task.

This may be due to that since we search the model settings on SQuAD, MobileBERT may be over-fitted to question answering tasks.

We perform an ablation study to investigate how each component of MobileBERT contributes to its performance on the dev data of a few GLUE tasks with diverse characteristics.

To accelerate the experiment process, we halve the original pre-training schedule in the ablation study.

We conduct a set of ablation experiments with regard to Attention Transfer (AT), Feature Map Transfer (FMT) and Pre-training Distillation (PD).

The operational OPTimizations (OPT) are removed in these experiments.

Moreover, to investigate the effectiveness of the proposed novel architecture of MobileBERT, we compare MobileBERT with two compact BERT models from Turc et al. (2019) .

For a fair comparison, we also design our own BERT baseline BERT SMALL* , which is the best model setting we can find with roughly 25M parameters under the original BERT architecture.

The detailed model setting of BERT SMALL* can be found in Table 2 .

Besides these experiments, to verify the performance of MobileBERT on real-world mobile devices, we export the models with Tensorflow Lite 5 APIs and measure the inference latencies on a single large core of a Pixel 3 phone with a fixed sequence length of 128.

The results are listed in Table 5 .

We first can see that the propose Feature Map Transfer contributes most to the performance improvement of MobileBERT, while Attention Transfer and Pre-training Distillation also play positive roles.

As expected, the proposed operational OPTimizations hurt the model performance a bit, but it brings a crucial speedup of 1.68??.

In architecture comparison, we find that although specifically designed for progressive knowledge transfer, our MobileBERT architecture alone is still quite competitive.

It outperforms BERT SMALL * and BERT SMALL on all compared tasks, while outperforming the 1.7?? sized BERT MEDIUM on the SST-2 task.

Finally, we can L1 H1  L1 H2  L1 H3  L1 H4  L12 H1  L12 H2  L12 H3  L12 H4 MobileBERT ( find that although augmented with the powerful progressive knowledge transfer, our MobileBERT still degrades greatly when compared to the IB-BERT LARGE teacher.

We visualize the attention distributions of the 1 st and the 12 th layers of a few models in Figure  3 for further investigation.

The proposed attention transfer can help the student mimic the attention distributions of the teacher very well.

Surprisingly, we find that the attention distributions in the attention heads of "MobileBERT(bare)+PD+FMT" are exactly a re-order of those of "Mobile-BERT(bare)+PD+FMT+AT" (also the teacher model), even if it has not been trained by the attention transfer objective.

This phenomenon indicates that multi-head attention is a crucial and unique part of the non-linearity of BERT.

Moreover, it can explain the minor improvements of Attention Transfer in ablation table 5, since the alignment of feature maps lead to the alignment of attention distributions.

We have presented MobileBERT which is a task-agnostic compact variant of BERT.

It is built upon a progressive knowledge transfer method and a conjugate architecture design.

Standard model compression techniques including quantization (Shen et al., 2019) and pruning (Zhu & Gupta, 2017) can be applied to MobileBERT to further reduce the model size as well as the inference latency.

In addition, although we have utilized low-rank decomposition for the embedding layer, it still accounts for a large part in the final model.

We believe there is a big room for extremely compressing the embedding table (Khrulkov et al., 2019; May et al., 2019) .

Layer-wise pre-training of neural networks can be dated back to Deep Belief Networks (DBN) (Hinton et al., 2006) and stacked auto-encoders (Vincent et al., 2008) .

Bengio et al. (2007) showed that the unsupervised pre-training of DBN helps to mitigate the difficult optimization problem of deep networks by better initializing the weights of all layers.

Although they made essential breakthrough in the application of neural networks, they are widely considered to be obsolete.

A more popular way today is to train deep neural networks in an end-to-end fashion.

However, Glasmachers (2017) recently showed that end-to-end learning can sometimes be very inefficient.

In this paper, we propose a progressive knowledge transfer scheme to combine the best of both worlds.

Compared to previous layer-wise methods, we use a well-optimized wider teacher to guide the layer-wise pre-training of the narrower student, rather than a greedy layer-wise unsupervised way, which makes better use of labels and rewards.

Our method also tackle the difficult training problem of end-to-end training from scratch.

While much recent research has focused on improving efficient Convolutional Neural Networks (CNN) for mobile vision applications (Iandola et al., 2016; Howard et al., 2017; Zhang et al., 2017; Sandler et al., 2018; , they are usually tailored for CNN.

Popular lightweight operations such as depth-wise convolution (Howard et al., 2017) cannot be directly applied to Transformer or BERT.

In the NLP literature, the most relevant work can be group LSTMs (Kuchaiev & Ginsburg, 2017; Gao et al., 2018) , which employs the idea of group convolution (Zhang et al., 2017; into Recurrent Neural Networks (RNN).

Recently, compressing or accelerating Transformer or BERT has attracted much attention.

apply Block-Term Tensor Decomposition on the self-attention modules of Transformer and achieve a compression of 2.5 on the machine translation task, but they don't consider how to compress the feed-forward networks, which constrains the compression ratio.

Lample et al. (2019) use structured memory layers to replace feed-forward networks in BERT and get better perplexity by half the computation, but they cannot compress the model size.

Compared to these work, Mobile-BERT reduces overheads in both self-attentions and feed-forward networks of BERT by bottleneck structure, while achieves efficiency with regard to both storage and computation.

We evaluate the effectiveness of our two operational optimizations for MobileBERT introduced in Section 4.3: replacing layer normalization (LayerNorm) with NoNorm and replacing gelu activation with relu activation.

We use the same experimental setting as in Section 5.5, where the models are exported to Tensorflow Lite format and evaluated on a single large core of a Pixel 3 phone with a fixed sequence length of 128.

From Table 6 , we can see that both NoNorm and relu are very effective in reducing the latency of MobileBERT, even if these two operational optimizations do not reduce FLOPS.

This reveals the gap between the real-world inference latency and the theoretical computation overhead (i.e., FLOPS).

(??, ??, ??, ??) are hyperparameters to balance the different loss terms.

Specifically, we use ?? = 1, ?? = 100, ?? = 5000, ?? = 5 in our all experiments.

Pre-train MobileBERT For a fair comparison with original BERT, we follow the same preprocessing scheme as BERT, where we mask 15% of all WordPiece (Kudo & Richardson, 2018) tokens in each sequence at random and use next sentence prediction.

Please note that MobileBERT can be potentially further improved by several training techniques recently introduced, such as span prediction or removing next sentence prediction objective .

We leave it for future work.

In pre-training distillation, the hyperparameter ?? is used to balance the original masked language modeling loss and the distillation loss.

Following (Kim & Rush, 2016) , we set ?? to 0.5.

We notice that recently there is an unpublished work 6 that also propose a task-agnosticly compressed BERT, called DistilBERT.

Basically, DistilBERT is a 6-layer truncated BERT BASE , which is distilled from BERT BASE on unannotated data with masked language modeling target.

The distillation process of DistilBERT is quite similar to the pre-training distillation described in Section 3.4.

In comparison, in this paper, we propose a pair of conjugate architectures to help knowledge transfer and design a progressive knowledge transfer scheme which transfers the intrinsic knowledge of intermediate layers from the teacher to the student in a bottom-to-top progressive way.

In this section, we provide a brief description of the tasks in the GLUE benchmark (Wang et al., 2018) .

CoLA The Corpus of Linguistic Acceptability (Warstadt et al., 2018 ) is a collection of English acceptability judgments drawn from books and journal articles on linguistic theory.

The task is to predict whether an example is a grammatical English sentence and is evaluated by Matthews correlation coefficient (Matthews, 1975) .

SST-2 The Stanford Sentiment Treebank (Socher et al., 2013 ) is a collection of sentences from movie reviews and human annotations of their sentiment.

The task is to predict the sentiment of a given sentence and is evaluated by accuracy.

MRPC The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005 ) is a collection of sentence pairs automatically extracted from online news sources.

They are labeled by human annotations for whether the sentences in the pair are semantically equivalent.

The performance is evaluated by both accuracy and F1 score.

The Semantic Textual Similarity Benchmark (Cer et al., 2017 ) is a collection of sentence pairs drawn from news headlines, video and image captions, and natural language inference data.

Each pair is human-annotated with a similarity score from 1 to 5.

The task is to predict these scores and is evaluated by Pearson and Spearman correlation coefficients.

<|TLDR|>

@highlight

We develop a task-agnosticlly compressed BERT, which is 4.3x smaller and 4.0x faster than BERT-BASE while achieving competitive performance on GLUE and SQuAD.