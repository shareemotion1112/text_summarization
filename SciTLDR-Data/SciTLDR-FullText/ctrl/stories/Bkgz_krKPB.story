Large-scale pre-trained language model, such as BERT, has recently achieved great success in a wide range of language understanding tasks.

However, it remains an open question how to utilize BERT for text generation tasks.

In this paper, we present a novel approach to addressing this challenge in a generic sequence-to-sequence (Seq2Seq) setting.

We first propose a new task, Conditional Masked Language Modeling (C-MLM), to enable fine-tuning of BERT on target text-generation dataset.

The fine-tuned BERT (i.e., teacher) is then exploited as extra supervision to improve conventional Seq2Seq models (i.e., student) for text generation.

By leveraging BERT's idiosyncratic bidirectional nature, distilling the knowledge learned from BERT can encourage auto-regressive Seq2Seq models to plan ahead, imposing global sequence-level supervision for coherent text generation.

Experiments show that the proposed approach significantly outperforms strong baselines of Transformer on multiple text generation tasks, including machine translation (MT) and text summarization.

Our proposed model also achieves new state-of-the-art results on the IWSLT German-English and English-Vietnamese MT datasets.

Large-scale pre-trained language model, such as ELMo (Peters et al., 2018) , GPT (Radford et al., 2018) and BERT (Devlin et al., 2019) , has become the de facto first encoding step for many natural language processing (NLP) tasks.

For example, BERT, pre-trained with deep bidirectional Transformer (Vaswani et al., 2017) via masked language modeling and next sentence prediction, has revolutionized the state of the art in many language understanding tasks, such as natural language inference (Bowman et al., 2015) and question answering (Rajpurkar et al., 2016) .

However, beyond common practice of fine-tuning BERT for language understanding , applying BERT to language generation still remains an open question.

Text generation aims to generate natural language sentences conditioned on certain input, with applications ranging from machine translation (Cho et al., 2014; Bahdanau et al., 2015) , text summarization (Nallapati et al., 2016; Gehring et al., 2017; Chen & Bansal, 2018) ), to image captioning Xu et al., 2015; Gan et al., 2017) .

In this paper, we study how to use BERT for better text generation, which to the best of our knowledge is still a relatively unexplored territory.

Intuitively, as BERT is learned with a generative objective via Masked Language Modeling (MLM) during the pre-training stage, a natural assumption is that this training objective should have learned essential, bidirectional, contextual knowledge that can help enhance text generation.

Unfortunately, this MLM objective is not auto-regressive, which encumbers its direct application to auto-regressive text generation in practice.

In this paper, we tackle this challenge by proposing a novel and generalizable approach to distilling knowledge learned in BERT for text generation tasks.

We first propose a new Conditional Masked Language Modeling (C-MLM) task, inspired by MLM but requiring additional conditional input, which enables fine-tuning pre-trained BERT on a target dataset.

In order to extract knowledge from the fine-tuned BERT and apply it to a text generation model, we leverage the fine-tuned BERT as a teacher model that generates sequences of word probability logits for the training samples, and treat the text generation model as a student network, which can effectively learn from the teacher's outputs for imitation.

The proposed approach improves text generation by providing a good estimation on the word probability distribution for each token in a sentence, consuming both the left and the right context, the exploitation of which encourages conventional text generation models to plan ahead.

Text generation models are usually trained via Maximum Likelihood Estimation (MLE), or teacher forcing : at each time step, it maximizes the likelihood of the next word conditioned on its previous ground-truth words.

This corresponds to optimizing one-step-ahead prediction.

As there is no explicit signal towards global planning in the training objective, the generation model may incline to focusing on local structure rather than global coherence.

With our proposed approach, BERT's looking into the future ability can act as an effective regularization method, capturing subtle long-term dependencies that ensure global coherence and in consequence boost model performance on text generation.

An alternative way to leverage BERT for text generation is to initialize the parameters of the encoder or decoder of Seq2Seq with pre-trained BERT, and then fine-tuning on the target dataset.

However, this approach requires the encoder/decoder to have the same size as BERT, inevitably making the final text generation model too large.

Our approach, on the other hand, is modular and compatible to any text-generation model, and has no restriction on the model size (e.g., large or small) or model architecture (e.g., LSTM or Transformer).

The main contributions of this work are three-fold.

(i) We present a novel approach to utilizing BERT for text generation.

The proposed method induces sequence-level knowledge into the conventional one-step-ahead and teacher-forcing training paradigm, by introducing an effective regularization term to MLE training loss. (ii) We conduct comprehensive evaluation on multiple text generation tasks, including machine translation, text summarization and image captioning.

Experiments show that our proposed approach significantly outperforms strong Transformer baselines and is generalizable to different tasks. (iii) The proposed model achieves new state-of-the-art on both IWSLT14 German-English and IWSLT15 English-Vietnamese datasets.

Pre-trained Language Models Prior to pre-trained language model, word embeddings (Mikolov et al., 2013; Pennington et al., 2014; Bojanowski et al., 2017) were widely used for NLP tasks.

Recently, CoVe (McCann et al., 2017) introduced (conditional) language models pre-trained on paired machine translation corpus.

ELMo (Peters et al., 2018) learned a contextual language model on a large corpus with bidirectional RNN.

GPT (Radford et al., 2018) used unidirectional Transformer to achieve better contextualized word representation.

By fine-tuning pre-trained language models, ULMFit (Howard & Ruder, 2018 ) also achieved promising results on text classification.

In our study, we focus on BERT due to its superior performance on multiple language understanding tasks.

However, different from previous work exploiting BERT for language understanding tasks, here we aim to apply BERT to text generation.

To the best of our knowledge, this is still a relatively unexplored space.

The proposed approach is also model-agnostic and can be applied to other pretrained language models as well.

There has been some recent attempt on applying BERT to text generation.

Specifically, Lample & Conneau (2019) trained cross-lingual MLM and demonstrated promising results for cross-lingual natural language inference and unsupervised neural machine translation (NMT) .

Wang & Cho (2019) formulated BERT as a Markov Random Field LM and showed preliminary results on unsupervised text generation with improved diversity.

Zhang et al. (2019a) utilized an encoder with BERT and a two-stage decoder for text summarization.

Song et al. (2019) proposed Masked Seq2Seq (MASS) pre-training, demonstrating promising results on unsupervised NMT, text summarization and conversational response generation.

Concurrent with our work, Ghazvininejad et al. (2019) proposed a similar conditional MLM for constant-time translation, and Yang et al. (2019) studied how to fine-tune BERT for NMT.

Our approach is novel in the sense that we do not directly use the parameters of BERT in the Seq2Seq model.

Instead, BERT acts as an effective regularization to the MLE training loss, by proactively injecting future information for predicting the present.

Right-to-Left Generation Our work also shares a high-level intuition with those approaches that try to regularize left-to-right generative models with a right-to-left counterpart.

Specifically, trained a separate reverse NMT and performed joint decoding at inference time to enforce agreement between forward and reverse models.

Twin Networks (Serdyuk et al., 2018 ) used a backward RNN jointly trained with a forward RNN decoder by matching their hidden states.

Zhang et al. (2019b) further extended the idea to Transformer with joint training, so that the forward and the backward models iteratively improve each other.

Our proposed approach stems from a similar intuition.

However, we focus on using pre-trained language model such as BERT to regularize an auto-regressive generation model.

Knowledge Distillation Our method shares the same loss formulation as Knowledge Distillation (KD) proposed in Bucilu et al. (2006) ; Hinton et al. (2015) ; Kim & Rush (2016) , where a smaller student model is trained on soft labels provided by a larger teacher model.

More recently, applied KD to multilingual NMT, and Sun et al. (2019) proposed patient KD for BERT model compression.

Compared with these previous studies, where both the teacher and the student are trained on the same task, our approach is different in the sense that the BERT teacher is not designed to perform the student's generation task.

We focus on using KD to leverage the learned knowledge of BERT for text generation, while previous work mostly focused on model compression.

In this section, we present our proposed approach to distilling the knowledge in BERT for text generation in generic sequence-to-sequence (Seq2Seq) setting.

We first review Seq2Seq learning in Section 3.1, and then describe the proposed approach in Section 3.2 and 3.3.

Seq2Seq learning aims to generate a sequence of discrete output Y = (y 1 , . . .

, y N ) of length N , conditioned on a sequence of discrete input X = (x 1 , . . .

, x M ) of length M .

A Seq2Seq model learns parameters θ to estimate the conditional likelihood P θ (Y |X), typically trained via Maximum Likelihood Estimation (MLE), or equivalently, minimizing the cross-entropy loss as follows:

where each conditional probability can be calculated via an attention-based recurrent neural network (RNN) (Bahdanau et al., 2015; , Transformer (Vaswani et al., 2017) , or any other neural sequence-generation models.

This generic Seq2Seq learning framework is the state of the art on a wide range of text generation tasks.

Using modern deep neural networks, the conditional probabilities can be readily modeled as a sequence of classifications over the word vocabulary.

However, during training, in order to generate the t-th token y t , the model only sees a partial sentence y 1:t−1 from the ground-truth training data.

Intuitively, it is reasonable to assume that a bidirectional model can be more informative than a leftto-right generation model, since additional context from the right (or future) is also incorporated to predict the current word.

Unfortunately, this additional information is not utilized in a standard Seq2Seq model, since it can only be trained in a left-to-right manner, where the future context is masked out to prevent each word from indirectly "seeing itself ".

To compensate this singledirectional limitation of Seq2Seq setting, we propose a new conditional language model (C-MLM) to enable the fine-tuning of BERT on target generation task, in hope that the fine-tuned bidirectional BERT can be utilized for better text generation.

BERT (Devlin et al., 2019 ) is a deep bidirectional Transformer trained via Masked Language Modeling (MLM).

2 In a similar setting, where the input is a sequence pair (X, Y ), 3 15% of the tokens are randomly masked.

Formally, we denote the masked token sets as X m and Y m , and the disjoint counterpart (i.e., the unmasked tokens) as X u and Y u , respectively.

The trained BERT model aims to estimate the joint probability:

where i and j denote the number of masked tokens in X and Y , respectively.

Each x m ∈ X m , and each y m ∈ Y m .

Eqn.

(2) can be trained with the standard word-level cross-entropy loss.

We aim to marry MLM pre-training with Seq2Seq learning, to leverage bidirectional language model for text generation.

To this end, we propose a conditional-MLM, a variant of MLM that allows further fine-tuning of pre-trained BERT on target dataset.

For example, for machine translation, X and Y represent the source and the target sentence, respectively.

We first concatenate them together and randomly mask 15% of the tokens only in Y , then train the network to model the joint probability

The above C-MLM objective is similar to the conditional language modeling (LM) objective in Eqn.

(1), but conditional LM only permits predicting a word based on its left context.

C-MLM is also related to Masked Seq2Seq (MASS) pre-training Song et al. (2019) .

However, in MASS, the encoder takes a sentence with randomly masked fragment (several consecutive tokens) as input, and the decoder tries to predict this masked fragment, which is different from our model design.

The final goal is also different: MASS focuses on Seq2Seq pre-training, while we focus on leveraging BERT for text generation.

In our experiments, we observe that the C-MLM task can obtain high accuracy and good generalization on word prediction.

However, it is not feasible to generate sequential output directly from C-MLM.

Instead, we use knowledge distillation to distill the knowledge learned from the fine-tuned BERT into a Seq2Seq model for direct text generation, which will be explained in the next sub-section.

Our inspiration springs from the observation that the probability distribution of the masked word y m t is estimated using both y u 1:t−1 and y u t+1:N from Y u .

In other words, the distribution for a given word P (y m t |X, Y u ) contains information from both backward and forward contexts, which is a desirable benefit for providing sequence-level global guidance.

This probability distribution can be considered as soft targets for a text generation model to mimic from, which potentially contains more useful and fine-grained information than the usual hard-assigned, one-hot label, therefore enhancing conventional left-to-right generation models to look into the future.

In a knowledge distillation setting, the BERT model can be considered as a teacher, while the Seq2Seq model acts as a student.

Specifically, the Seq2Seq model can be trained with the following objective function:

where P φ (y t ) is the soft target estimated by the fine-tuned BERT with learned parameters φ, and V denotes the output vocabulary.

Note that φ is fixed during the distillation process.

An illustration of this learning process is provided in Figure 1 , which aims to match the word probability distribution P θ (y t ) provided by the student with P φ (y t ) provided by the teacher (i.e., distillation).

To further improve the Seq2Seq student model, hard-assigned labels are also utilized.

the final model is trained with the following compound objective:

where α is a hyper-parameter for tuning the relative importance of the two training targets: soft estimation from fine-tuned BERT, and ground-truth hard label.

Note that our proposed approach only has a minimal requirement on the architecture of the incorporated Seq2Seq model.

As long as the model is trained to estimate word-level probability as in Eqn.

(1), it can be trained jointly with the proposed objective function Eqn.

(5).

At a higher level, the additional loss term L bidi can be interpreted as a sequence-level objective function.

Our auto-regressive (or causal) model θ tries to predict the probability distribution that matches the estimation the bidirectional teacher model predicts, hence encouraging the planning of future (right context) for generation.

In this section, we describe our experiments on two well-studied text generation tasks: machine translation, and abstractive text summarization.

Machine Translation We consider two relatively small-scale datasets, IWSLT15 EnglishVietnamese (En-Vi, 113k training samples) and IWSLT14 German-English (De-En, 160k training samples), and one medium-scale dataset, WMT14 English-German (En-De, 4.5M training samples).

For IWSLT15 En-Vi, we use the pre-processed dataset provided by .

We use tst2012 as dev set and test on tst2013.

For IWSLT14 De-En, we follow the pre-processing steps and the same train/dev/test split as in Wu et al. (2019) .

For WMT14 En-De, we follow the preprocessing steps in Vaswani et al. (2017) for fair comparison.

We use newstest2013 as the dev set and newstest2014 as the test set.

We report BLEU scores (Papineni et al., 2002) for evaluation of MT performance following the Moses script.

Abstractive Summarization For summarization, we conduct experiments on the Gigaword summarization dataset (Rush et al., 2015) .

Note that the original train/valid/test split of Gigaword is 3.8M/190k/2k.

In our experiments, we observed severe distribution mismatch between the validation and test data.

See Table 4 , 5, and Sec. 4.3 for detailed discussion.

Therefore, we further sampled 5k/5k dev/test-dev splits from the validation set and tuned hyper-parameters on the dev set only.

We report ROUGE scores (Lin, 2004) on test-dev for the evaluation of our proposed approach, and include results on the standard test split for the comparison with prior work.

Training and Hyper-parameters Our implementation is based on the PyTorch (Paszke et al., 2017) version of OpenNMT (Klein et al., 2018) seq2seq toolkit.

We use the 'base' model of 6-layer Transformer with 512-hidden 8-head attention blocks and 2048-hidden feed-forward layer for all experiments, with label smoothing regularization (LSR) (Szegedy et al., 2016) of 0.1.

We batch examples with similar sequence length, and count batch size by the number of tokens.

For MT we use the pre-trained BERT-base-multilingual-cased model, and for summarization we use BERTbase-uncased as the starting point of BERT fine-tuning.

5 We use the corresponding pre-trained byte-pair-encoding (Sennrich et al., 2016) shipped together with the BERT model for tokenization.

For all training methods of all Transformer models, the learning rate schedule is set to lr = η · d −0.5 model · min(step −0.5 , step · warmup steps −1.5 ), where d model = 512 is the attention representation size (Vaswani et al., 2017) .

For all BERT fine-tuning, we follow Devlin et al. (2019) use a triangular learning rate schedule with maximum learning rate η.

The parameters are updated with the Adam optimizer (Kingma & Ba, 2015) .

In the distillation stage, we pre-compute BERT's prediction logits of the training data using top-K distillation to reduce computation overhead and memory footprint, where K is set to 8 across all the experiments.

We also tune the temperature T for the sof tmax applied at the teacher's logits.

For the detailed values of the hyper-parameters for each experiment, please refer to the supplementary material.

We found it necessary to train longer with L bidi , since it is still improving after the step at which the baseline Transformer starts to plateau.

At inference time, we use beam search with beam size 4 and length penalty (Wu et al., 2016 ) of 0.6 across all the models.

All the hyperparameters are tuned on the development set.

Note that we tuned our Transformer baseline to achieve higher scores than the reference implementation on each dataset with default hyper-parameters (in most cases comparable to the state-of-the-art).

We first validate our proposed text generation approach on machine translation task.

Experimental results are summarized in Table 1, 2 and 3, which show that our model significantly improves over the strong Transformer baseline across all three datasets.

Note that our baseline is the 'base' model of Transformer, which has 44M trainable parameters, and the reference implementation by Wu et al. (2019) is Transformer (big) with 176M trainable parameters.

For IWSLT German-English translation, our method improves over the Transformer baseline by 1.54 BLEU points, and achieves new state of the art.

Our approach outperforms previously-reported results such as ConvS2S+MRT, a convolutional-based model (Gehring et al., 2017) with minimum risk training , and Lightweight and Dynamic Convolution (Wu et al., 2019) .

Note that Wu et al. (2019) also tuned checkpoint averaging, which creates a soft ensemble effect.

And their model has roughly the same amount of parameters as Transformer (big).

For IWSLT English-Vietnamese translation, since most prior work experimented with RNN models, we also report RNN-based results here.

This also suggests that our method is model-agnostic.

Our best model outperforms Seq2Seq-OT (Chen et al., 2019 ) that utilizes optimal transport for sequencelevel training, as well as the ELMo and CVT results reported in .

8 For WMT14

6 Different from the original KD, we do not apply the same temperature on the student.

In our preliminary experiment we found high T of Seq2Seq results in much worse performance.

We hypothesize the low-entropy nature of conditioned text generation is not suitable for temperature scaling.

7 Parameter counts exclude word embedding and final linear projection, which mostly depends on the vocabulary size.

BERT-base has 86M trainable parameters.

8 The CVT results used a much larger RNN and CNN-based character embedding, as well as a customized structure.

Therefore, we did not try to use RNN to match their results.

English-German translation, our method still improves over the well-tuned Transformer baseline.

We also report the scores of Transformer (big) and state-of-the-art Dynamic Convolution model (Wu et al., 2019) for reference.

Table 4 and Table 5 show the results of our approach on abstractive summarization task, where R-1, R-2, and R-L denote F 1 scores of ROUGE-1, ROUGE-2, and ROUGE-L, respectively.

Our method shows improvement on all the metrics, as shown in Table 4 .

We observe that the performance on test set is much lower, which suggests that the distribution in the test set is very different from that in the validation set, as mentioned in Section 4.1.

When we manually checked the test set data, we found many corrupted examples such as short input articles, meaningless text, and dominating unknown words.

Given that the official test split contains only 1,951 noisy examples, we believe that our results on the dev/test-dev sets are more reliable.

R-1 R-2 R-L Dev

On the test split, our best model is comparable to state-of-the-art models that use much more complex architectures specifically designed for summarization.

CGU (Lin et al., 2018) augmented convolutional gating units.

FTSum g (Cao et al., 2018b) leveraged extra information extraction and dependency parsing features.

E2T cnn (Amplayo et al., 2018) utilized entities provided by an external entity linking system.

Re 3 Sum (Cao et al., 2018a) carefully designed a retrieve-and-rerank pipeline with human-written soft templates.

Despite that our model has no summarization-specific model design, we still achieve comparable performance to these models on all the metrics.

There are several possible factors that could contribute to the performance gain: additional parameters of BERT, extra data (pretraining corpus) of BERT, and the bidirectional nature.

To better understand the key contributions of our method, we conduct an ablation study described in the following.

We finetune 2 extra teachers: BERT sm and BERT l2r .

For BERT sm , we use a smaller BERT (6 layers) for C-MLM finetuning, which has approximately the same number of parameters as Transformer-base 9 .

For BERT l2r , we use the full BERT model but finetune it using left-to-right LM as in the conventional Seq2Seq model.

Next, we apply the proposed KD method to train the Transformer on En-Vi and De-En MT tasks.

Results are shown in Table 6 .

BERT sm still works well though the full BERT provides further improvement.

On the other hand, BERT l2r slightly hurts the performance.

We hypothesize that it generates noisy learning targets for the student, hence the performance drop.

Empirically, we show that the bidirectional knowledge could be more important than the extra parameters, while the pre-trained weights remain useful for more stable C-MLM training.

We next analyze the effect of our proposed approach on different output lengths.

We plot the BLEU scores on MT w.r.t.

different output generation lengths N on the development set.

10 Results are provided in Figure 2 .

For IWSLT German-English dataset (Figure 2 : Left), we can see a shared trend that the proposed L bidi objective gains higher BLEU points on longer translation pairs.

For WMT English-German (Figure 2 : Middle), we can see that although the proposed method performs much worse when the output sentences are very short, it achieves relatively consistent improvement on longer cases, hence resulting in overall BLEU improvement.

For IWSLT English-Vietnamese (Figure 2 : Right), we see a similar trend when the length N > 24.

In Table 7 , we show some translation examples on IWSLT German-English dataset.

In the first example, the baseline Transformer cannot recover from 'with' and 'of ', which renders the full sentence not making much sense.

"

I started reading with..." would make sense from the left context; however, if the model also considers the right context "the age of two", the word 'with' would be assigned with lower probability by the soft labels provided by the BERT teacher.

Even though at test-time the model cannot 'look ahead', the soft-targets at training-time prevents the over-confidence of the model on one-hot label; hence the better generalization at the test-time.

Similarly, other examples show that our model can generate text more coherently w.r.t.

the context on the right (underlined in Table 7 ), thus making more accurate and natural translation.

In this work, we propose a novel and generic approach to utilizing pre-trained language models to improve text generation without explicit parameter sharing, feature extraction, or augmenting with auxiliary tasks.

Our proposed Conditional MLM mechanism leverages unsupervised language models pre-trained on large corpus, and then adapts to supervised sequence-to-sequence tasks.

Our distillation approach indirectly influences the text generation model by providing soft-label distributions only, hence is model-agnostic.

Experiments show that our model improves over strong Transformer baselines on multiple text generation tasks such as machine translation and abstractive summarization, and achieves new state-of-the-art on some of the translation tasks.

For future work, we will explore the extension of Conditional MLM to multimodal input such as image captioning.

We run all experiments on single GPU of NVIDIA Titan RTX or V100 except for WMT En-De we use 4 V100s for training.

Note that for large batch sizes that do not fit in GPU memory, we use the gradient accumulation tricks as in .

Batch sizes are counted in number of tokens.

Note that all the hyper-parameters are tuned on the development set only.

IWSLT De-En For C-MLM fine-tuning, we train for 100k steps with 5k warmup steps, η = 5 · 10 −5 , and batch size of 16k tokens.

For baseline model, we train for 50k steps with 4k warmup steps and batch size of 6k tokens.

The learning rate η is set to 1.

For the proposed model, we train for 100k steps with 8k warmup steps and batch size of 6k tokens.

The learning rate η is set to 2, α = 0.5, and T = 10.

Seq2Seq model uses dropout (Srivastava et al., 2014) of 0.3 in both cases.

IWSLT En-Vi For C-MLM fine-tuning and baseline Transformer, the hyper-parameters are identical to that of IWSLT De-En.

For the proposed model, we train for 100k steps with 8k warmup steps and batch size of 6k tokens.

The learning rate η is set to 2, α = 0.1, and T = 5.

Dropout is still 0.1.

WMT En-De For C-MLM fine-tuning, we train for 100k steps with 5k warmup steps, η = 5 · 10 −5 , and batch size of 512k tokens.

For baseline model, we train for 30k steps with 4k warmup steps and batch size of 384k tokens.

The learning rate η is set to 4.

Since this is our largest dataset and training is slow, for the proposed model we use the baseline Transformer to initialize the Seq2Seq student.

For the proposed model, we continue training for 50k steps with 4k warmup steps and batch size of 64k tokens.

The learning rate η is set to 2, α = 0.1, and T = 5.

Seq2Seq model uses dropout of 0.1 in both cases.

Gigaword For C-MLM fine-tuning, we train for 100k steps with 5k warmup steps, η = 5·10 −5 , and batch size of 64k tokens.

For baseline model, we train for 50k steps with 4k warmup steps and batch size of 40k tokens.

The learning rate η is set to 1.

For the proposed model, we train for 70k steps with 4k warmup steps and batch size of 36k tokens.

The learning rate η is set to 2, α = 0.1, and T = 10.

Seq2Seq model uses dropout of 0.1 in both cases.

We show Gigaword summarization examples in Table 9 and extra En-DE generation examples in Reference it would be immoral to leave these young people with a climate system spiraling out of control .

Transformer it would be immoral to let these young people leave a climate system that was out of control . (44.6) Ours it would be immoral to leave these young people with a climate system out of control .

Table 8 : Qualitative examples from IWSLT German-English translation.

Numbers inside the parenthesis are sentence-level BLEU scores.

Red word is where the baseline Transformer makes a mistake without considering the possible future phrase and fails to recover.

On the other hand, our model makes the right decision at the blue word, hence generates more coherent sentence.

Please refer to Section 4.5 in the main paper for detailed explanation.

Reference china offers tax exemptions for laid-off workers Transformer china encourages laid-off workers to seek employment Ours china offers tax exemptions to laid-off workers Reference swiss police arrest britons who allegedly ran rental car racket Transformer three britons arrested in swiss luxury hotel Ours swiss police arrest three britons in rental car racket case Reference south korea stocks extend declines as kia concerns intensify Transformer south korean stocks fall for #th time in # days ; kia leads Ours south korean stocks fall as kia troubles intensify Table 9 : Qualitative examples from the Gigaword summarization dataset.

Baseline model suffers from early mistakes.

Our model generates more coherent summaries.

<|TLDR|>

@highlight

We propose a model-agnostic way to leverage BERT for text generation and achieve improvements over Transformer on 2 tasks over 4 datasets.