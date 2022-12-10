Multilingual machine translation, which translates multiple languages with a single model, has attracted much attention due to its efficiency of offline training and online serving.

However, traditional multilingual translation usually yields inferior accuracy compared with the counterpart using individual models for each language pair, due to language diversity and model capacity limitations.

In this paper, we propose a distillation-based approach to boost the accuracy of multilingual machine translation.

Specifically, individual models are first trained and regarded as teachers, and then the multilingual model is trained to fit the training data and match the outputs of individual models simultaneously through knowledge distillation.

Experiments on IWSLT, WMT and Ted talk translation datasets demonstrate the effectiveness of our method.

Particularly, we show that one model is enough to handle multiple languages (up to 44 languages in our experiment), with comparable or even better accuracy than individual models.

Neural Machine Translation (NMT) has witnessed rapid development in recent years BID1 BID26 BID36 BID8 BID34 BID32 BID31 BID12 , including advanced model structures BID8 BID34 and human parity achievements .

While conventional NMT can well handle single pair translation, training a separate model for each language pair is resource consuming, considering there are thousands of languages in the world 1 .

Therefore, multilingual NMT BID17 BID5 BID13 ) is developed which handles multiple language pairs in one model, greatly reducing the offline training and online serving cost.

Previous works on multilingual NMT mainly focus on model architecture design through parameter sharing, e.g., sharing encoder, decoder or attention module BID5 or sharing the entire models BID17 BID13 .

They achieve comparable accuracy with individual models (each language pair with a separate model) when the languages are similar to each other and the number of language pairs is small (e.g., two or three).

However, when handling more language pairs (dozens or even hundreds), the translation accuracy of multilingual model is usually inferior to individual models, due to language diversity.

It is challenging to train a multilingual translation model supporting dozens of language pairs while achieving comparable accuracy as individual models.

Observing that individual models are usually of higher accuracy than the multilingual model in conventional model training, we propose to transfer the knowledge from individual models to the multilingual model with knowledge distillation, which has been studied for model compression and knowledge transfer and well matches our setting of multilingual translation.

It usually starts by training a big/deep teacher model (or ensemble of multiple models), and then train a small/shallow student model to mimic the behaviors of the teacher model, such as its hidden representation BID39 BID29 , its output probabilities BID16 BID6 or directly training on the sentences generated by the teacher model in neural machine translation BID19 ).

The student model can (nearly) match the accuracy of the cumbersome teacher model (or the ensemble of multiple models) with knowledge distillation.

In this paper, we propose a new method based on knowledge distillation for multilingual translation to eliminate the accuracy gap between the multilingual model and individual models.

In our method, multiple individual models serve as teachers, each handling a separate language pair, while the student handles all the language pairs in a single model, which is different from the conventional knowledge distillation where the teacher and student models usually handle the same task.

We first train the individual models for each translation pair and then we train the multilingual model by matching with the outputs of all the individual models and the ground-truth translation simultaneously.

After some iterations of training, the multilingual model may get higher translation accuracy than the individual models on some language pairs.

Then we remove the distillation loss and keep training the multilingual model on these languages pairs with the original log-likelihood loss of the ground-truth translation.

We conduct experiments on three translation datasets: IWSLT with 12 language pairs, WMT with 6 language pairs and Ted talk with 44 language pairs.

Our proposed method boosts the translation accuracy of the baseline multilingual model and achieve similar (or even better) accuracy as individual models for most language pairs.

Specifically, the multilingual model with only 1/44 parameters can match or surpass the accuracy of individual models on the Ted talk datasets.

Given a set of bilingual sentence pairs D = {(x, y) ∈ X ×Y}, an NMT model learns the parameter θ by minimizing the negative log-likelihood − (x,y)∈D log P (y|x; θ).

P (y|x; θ) is calculated based on the chain rule Ty t=1 P (y t |y <t , x; θ), where y <t represents the tokens preceding position t, and T y is the length of sentence y.

The encoder-decoder framework BID1 BID26 BID33 BID36 BID8 BID34 is usually adopted to model the conditional probability P (y|x; θ), where the encoder maps the input to a set of hidden representations h and the decoder generates each target token y t using the previous generated tokens y <t as well as the representations h.

NMT has been extended from the translation of a single language pair to multilingual translation BID4 BID25 BID5 BID17 BID13 , considering the large amount of languages pairs in the world.

Some of these works focus on how to share the components of the NMT model among multiple language pairs.

BID4 use a shared encoder but different decoders to translate the same source language to multiple target languages.

BID25 use the combination of multiple encoders and decoders, with one encoder for each source language and one decoder for each target language respectively, to translate multiple source languages to multiple target languages.

BID5 share the attention mechanism but use different encoders and decoders for multilingual translation.

Similarly, design the neural interlingua, which is an attentional LSTM encoder to bridge multiple encoders and decoders for different language pairs.

In BID17 and BID13 , multiple source and target languages are handled with a universal model (one encoder and decoder), with a special tag in the encoder to determine which target language to translate.

In BID10 BID17 and BID27 , multilingual translation is leveraged to boost the accuracy of low-resource language pairs with better model structure or training mechanism.

It is observed that when there are dozens of language pairs, multilingual NMT usually achieves inferior accuracy compared with its counterpart which trains an individual model for each language pair.

In this work we propose the multilingual distillation framework to boost the accuracy of multilingual NMT, so as to match or even surpass the accuracy of individual models.

The early adoption of knowledge distillation is for model compression BID2 , where the goal is to deliver a compact student model that matches the accuracy of a large teacher model or the ensemble of multiple models.

Knowledge distillation has soon been applied to a variety of tasks, including image classification BID16 BID7 BID37 BID0 BID23 , speech recognition BID16 and natural language processing BID19 BID6 .

Recent works BID7 BID37 even demonstrate that student model can surpass the accuracy of the teacher model, even if the teacher model is of the same capacity as the student model.

BID40 propose the mutual learning to enable multiple student models to learn collaboratively and teach each other by knowledge distillation, which can improve the accuracy of those individual models.

BID0 propose online distillation to improve the scalability of distributed model training and the training accuracy.

In this paper, we develop the multilingual distillation framework for multilingual NMT.

Our work differs from BID40 and BID0 in that they collaboratively train multiple student models with codistillation, while we use multiple teacher models to train a single student model, the multilingual NMT model.

As mentioned, when there are many language pairs and each pair has enough training data, the accuracy of individual models for those language pairs is usually higher than that of the multilingual model, given that the multilingual model has limited capacity comparing with the sum of all the individual models.

Therefore, we propose to teach the multilingual model using the individual models as teachers.

Here we first describe the idea of knowledge distillation in neural machine translation for the case of one teacher and one student, and then introduce our method in the multilingual setting with multiple teachers (the individual models) and one student (the multilingual model).

Denote D = {(x, y) ∈ X × Y} as the bilingual corpus of a language pair.

The log-likelihood loss (cross-entropy with one-hot label) on corpus D with regard to an NMT model θ can be formulated as follows: DISPLAYFORM0 where T y is the length of the target sentence, |V | is the vocabulary size of the target language, y t is the t-th target token, 1{·} is the indicator function that represents the one-hot label, and P (·|·) is the conditional probability with model θ.

In knowledge distillation, the student (with model parameter θ) not only matches the outputs of the ground-truth one-hot label, but also to the probability outputs of the teacher model (with parameter θ T ).

Denote the output distribution of the teacher model for token y t as Q(y t |y <t , x; θ T ).

The cross entropy between two distributions serves as the distillation loss: DISPLAYFORM1 Q{y t = k|y <t , x; θ T } log P (y t = k|y <t , x; θ).( DISPLAYFORM2 is no longer the original one-hot label, but teacher's output distribution which is more smooth by assigning non-zero probabilities to more than one word and yields smaller variance in gradients BID16 .

Then the total loss function becomes DISPLAYFORM3 where λ is the coefficient to trade off the two loss terms.

Let L denote the total number of language pairs in our setting, superscript l ∈ [L] denote the index of language pair, D l denote the bilingual corpus for the l-th language pair, θ M denote the parameters of the (student) multilingual model, and θ l I denote the parameters of the (teacher) individual model for l-th language pair.

Therefore, L NLL (D; θ M ) denotes the log-likelihood loss on training data D, and DISPLAYFORM0 ) denotes the total loss on training data D l , which consists of the original log-likelihood loss and the distillation loss by matching to the outputs from the teacher model θ l I .

The multilingual distillation process is summarized in Algorithm 1.

As can be seen in Line 1, our algorithm takes pretrained individual models for each language pair as inputs.

Note that those models can be pretrained using the same datasets {D l } L l=1 or different datasets, and they can share the same network structure as the multilingual model or use different architectures.

For simplification, in our experiments, we use the same datasets to pretrain the individual models and they share the same architecture as the multilingual model.

In Line 8-9, the multilingual model learns from both the ground-truth data and the individual models with loss L ALL when its accuracy has not surpassed the individual model for a certain threshold τ (which is checked in Line 15-19 every T check steps according to the accuracy in validation set); otherwise, the multilingual model only learns from the ground-truth data using the original log-likelihood loss L NLL (in Line 10-11).

and pretrained individual models {θ DISPLAYFORM0 for L language pairs, learning rate η, total training steps T , distillation check step T check , threshold τ of distillation accuracy.

2: Initialize: Randomly initialize multilingual model θ M .

Set current training step DISPLAYFORM1 Randomly sample a mini-batch of sentence pairs ( DISPLAYFORM2 Compute and accumulate the gradient on loss DISPLAYFORM3 DISPLAYFORM4 end for

end if 20: end while

Selective Distillation Considering that distillation from a bad teacher model is likely to hurt the student model and thus result in inferior accuracy, we selectively use distillation in the training process, as shown in Line 15-19 in Algorithm 1.

When the accuracy of multilingual model surpasses the individual model for the accuracy threshold τ on a certain language pair, we remove the distillation loss and just train the model with original negative log-likelihood loss for this pair.

Note that in one iteration, one language may not uses the distillation loss; it is very likely in later iterations that this language will be distilled again since the multilingual model may become worse than the teacher model for this language.

Therefore, we call this mechanism as selective distillation.

We also verify the effectiveness of the selective distillation in experiment part (Section 4.3).Top-K Distillation It is burdensome to load all the teacher models in the GPU memory for distillation considering there are dozens or even hundreds of language pairs in the multilingual setting.

Alternatively, we first generate the output probability distribution of each teacher model for the sentence pairs offline, and then just load the top-K probabilities of the distribution into memory and normalize them so that they sum to 1 for distillation.

This can reduce the memory cost again from the scale of |V | (the vocabulary size) to K. We also study in Section 4.3 that top-K distribution can result in comparable or better distillation accuracy than the full distribution.

We test our proposed method on three public datasets: IWSLT, WMT, and Ted talk translation tasks.

We first describe experimental settings, report results, and conduct some analyses on our method.

Datasets We use three datasets in our experiment.

IWSLT: We collect 12 languages↔English translation pairs from IWSLT evaluation campaign 2 from year 2014 to 2016.

WMT: We collect 6 languages↔English translation pairs from WMT translation task 3 .

Ted Talk: We use the common corpus of TED talk which contains translations between multiple languages BID38 .

We select 44 languages in this corpus that has sufficient data for our experiments.

More descriptions about the three datasets can be found in Appendix (Section 1).

We also list the language code according to ISO-639-1 standard 4 for the languages used in our experiments in Appendix (Section 2).

All the sentences are first tokenized with moses tokenizer 5 and then segmented into subword symbols using Byte Pair Encoding (BPE) BID30 .

We learn the BPE merge operations across all the languages and keep the output vocabulary of the teacher and student model the same, to ensure knowledge distillation.

We use the Transformer BID34 as the basic NMT model structure since it achieves state-of-the-art accuracy and becomes a popular choice for recent NMT researches.

We use the same model configuration for individual models and the multilingual model.

For IWSLT and Ted talk tasks, the model hidden size d model , feed-forward hidden size d ff , number of layer are 256, 1024 and 2, while for WMT task, the three parameters are 512, 2048 and 6 respectively considering its large scale of training data.

Training and Inference For the multilingual model training, we up sample the data of each language to make all languages have the same size of data.

The mini batch size is set to roughly 8192 tokens.

We train the individual models with 4 NVIDIA Tesla V100 GPU cards and multilingual models with 8 of them.

We follow the default parameters of Adam optimizer BID21 and learning rate schedule in BID34 .

For the individual models, we use 0.2 dropout, while for multilingual models, we use 0.1 dropout according to the validation performance.

For knowledge distillation, we set T check = 3000 steps (nearly two training epochs), the accuracy threshold τ = 1 BLEU score, the distillation coefficient λ = 0.5 and the number of teacher's outputs K = 8 according to the validation performance.

During inference, we decode with beam search and set beam size to 4 and length penalty α = 1.0 for all the languages.

We evaluate the translation quality by tokenized case sensitive BLEU BID28 with multi-bleu.pl 6 .

Our codes are implemented based on fairseq 7 and we will release the codes once the paper is published.

Table 2 : BLEU scores of English→12 languages on the IWLST dataset.

The BLEU scores in () represent the difference between the multilingual model and individual models.

∆ represents the improvements of our multi-distillation method over the multi-baseline.

Results on IWSLT Multilingual NMT usually consists of three settings: many-to-one, one-tomany and many-to-many.

As many-many translation can be bridged though many-to-one and oneto-many setting, we just conduct the experiments on many-to-one and one-to-many settings.

We first show the results of 12 languages→English translations on the IWLST dataset are shown in TAB2 .

There are 3 methods for comparison: 1) Individual, each language pair with a separate model; 2) Multi-Baseline, the baseline multilingual model, simply training all the language pairs in one model; 3) Multi-Distillation, our multilingual model with knowledge distillation.

We have several observations.

First, the multilingual baseline performs worse than individual models on most languages.

The only exception is the languages with small training data, which benefit from data augmentation in multilingual training.

Second, our method outperforms the multilingual baseline for all the languages, demonstrating the effectiveness of our framework for multilingual NMT.

More importantly, compared with the individual models, our method achieves similar or even better accuracy (better on 10 out of 12 languages), with only 1/12 model parameters of the sum of all individual models.

One-to-many setting is usually considered as more difficult than many-to-one setting, as it contains different target languages which is hard to handle.

Here we show how our method performs in oneto-many setting in Table 2 .

It can be seen that our method can maintain the accuracy (even better on most languages) compared with the individual models.

We still improve over the multilingual baseline by nearly 1 BLEU score, which demonstrates the effectiveness of our method.

Table 3 : BLEU scores of 6 languages→English on the WMT dataset.

The BLEU scores in () represent the difference between the multilingual model and individual models.

∆ represents the improvements of our multi-distillation method over the multi-baseline.

Table 4 : BLEU scores of English→ 6 languages on the WMT dataset.

Results on WMT The results of 6 languages→English translations on the WMT dataset are reported in Table 3 .

It can be seen that the multi-baseline model performs worse than the individual models on 5 out of 6 languages, while in contrast, our method performs better on all the 6 languages.

Particularly, our method improves the accuracy of some languages with more than 2 BLEU scores over individual models.

The results of one-to-many setting on WMT dataset are reported in Table 4 .

It can be seen that our method outperforms the multilingual baseline by more than 1 BLEU score on nearly all the languages.

Table 5 : BLEU scores improvements of our method over the individual models (∆ 1 ) and multibaseline model (∆ 2 ) on the 44 languages→English in the Ted talk dataset.

Results on Ted Talk Now we study the effectiveness of our method on a large number of languages.

The experiments are conducted on the 44 languages→English on the Ted talk dataset.

Due to the large number of languages and space limitations, we just show the BLEU score improvements of our method over individual models and the multi-baseline for each language in Table 5 , and leave the detailed experiment results to Appendix (Section 3).

It can be seen that our method can improve over the multi-baseline for all the languages, mostly with more than 1 BLEU score improvements.

Our method can also match or even surpass individual models for most languages, not to mention that the number of parameters of our method is only 1/44 of that of the sum of 44 individual models.

Our method achieves larger improvements on some languages, such as Da, Et, Fi, Hi and Hy, than others.

We find this is correlated with the data size of the languages, which are listed in Appendix TAB2 .

When a language is of smaller data size, it may get more improvement due to the benefit of multilingual training.

In this section, we conduct thorough analyses on our proposed method for multilingual NMT.Selective Distillation We study the effectiveness of the selective distillation (discussed in Section 3.3) on the Ted talk dataset, as shown in Table 6 .

We list the 16 languages on which the two methods (selective distillation, and distillation all the time) that have difference bigger than 0.5 in terms of BLEU score.

It can be seen that selective distillation performs better on 13 out of 16 languages, with large BLEU score improvements, which demonstrates the effectiveness of the selective distillation.

Table 6 : BLEU scores of selective distillation (our method) and distillation all the time during the training process on the Ted talk dataset.

In our experiments, the student model just matches the top-K output distribution of the teacher model, instead of the full distribution, in order to reduce the memory cost.

We analyze whether there is accuracy difference between the top-K distribution and the full distribution.

We conduct experiments on IWSLT dataset with varying K (from 1 to |V |, where |V | is the vocabulary size), and just show the BLEU scores on the validation set of De-En translation due to space limitation, as illustrated in Table 7 .

It can be seen that increasing K from 1 to 8 will improve the accuracy, while bigger K will bring no gains, even with the full distribution (K = |V |).

Table 7 : BLEU scores on De-En translation with varying Top-K distillation on the IWSLT dataset.

Back Distillation In our current distillation algorithm, we fix the individual models and use them to teach and improve the multilingual model.

After such a distillation process, the multilingual model outperforms the individual models on most of the languages.

Then naturally, we may wonder whether this improved multilingual model can further be used to teach and improve individual models through knowledge distillation.

We call such a process back distillation.

We conduct the experiments on the IWSLT dataset, and find that the accuracy of 9 out of 12 languages gets improved, Table 8 : BLEU score improvements of the individual models with back distillation on the IWSLT dataset.as shown in TAB2 .

The other 3 languages (He, Pt, Zh) cannot get improvements because the improved multilingual model performs very close to individual models, as shown in TAB2 .Comparison with Sequence-Level Knowledge Distillation We conduct experiments to compare the word-level knowledge distillation (the exact method used in our paper) with sequence-level knowledge distillation BID20 ) on IWSLT dataset.

As shown in TAB10 , sequencelevel knowledge distillation results in consistently inferior accuracy on all languages compared with word-level knowledge distillation used in our work.

TAB2 : BLEU score improvements of the individual models with back distillation on the IWSLT dataset.

Generalization Analysis Previous works BID37 BID22 have shown that knowledge distillation can help a model generalize well to unseen data, and thus yield better performance.

We analyze how distillation in multilingual setting helps the model generalization.

Previous studies BID18 BID3 demonstrate the relationship between model generalization and the width of local minima in loss surface.

Wider local minima can make the model more robust to small perturbations in testing.

Therefore, we compare the generalization capability of the two multilingual models (our method and the baseline) by perturbing their parameters.

Specifically, we perturb a model θ as θ i (σ) = θ i +θ * N (0, σ 2 ), where θ i is the i-th parameter of the model,θ is the average of all the parameters in θ.

We sample from the normal distribution N with standard variance σ and larger σ represents bigger perturbation on the parameter.

We conduct the analyses on the IWSLT dataset and vary σ ∈ [0. 05, 0.1, 0.15, 0.2, 0.25, 0.3] .

FIG0 shows the loss curve in the test set with varying σ.

As can be seen, while both the two losses increase with the increase of σ, the loss of the baseline model increases quicker than our method.

We also show three test BLEU curves on three translation pairs FIG0 : Ar-En, FIG0 : Cs-En, FIG0 : De-En, which are randomly picked from the 12 languages pairs on the IWSLT dataset).

We observe that the BLEU score of the multilingual baseline drops quicker than our method, which demonstrates that our method helps the model find wider local minima and thus generalize better.

In this work, we have proposed a distillation-based approach to boost the accuracy of multilingual NMT, which is usually of lower accuracy than the individual models in previous works.

Experiments on three translation datasets with up to 44 languages demonstrate the multilingual model based on our proposed method can nearly match or even outperform the individual models, with just 1/N model parameters (N is up to 44 in our experiments).In the future, we will conduct more deep analyses about how distillation helps the multilingual model training.

We will apply our method to larger datasets and more languages pairs (hundreds or even thousands), to study the upper limit of our proposed method.

We give a detailed description about the IWSLT,WMT and Ted Talk datasets used in experiments.

IWSLT: We collect 12 languages↔English translation pairs from IWSLT evaluation campaign 8 from year 2014 to 2016.

Each language pair contains roughly 80K to 200K sentence pairs.

We use the official validation and test sets for each language pair.

The data sizes of the training set for each language↔English pair are listed in TAB2 TAB2 : The training data size on the 12 languages↔ English on the IWSLT dataset.

WMT: We collect 6 languages↔English translation pairs from WMT translation task 9 .

We use 5 language↔English translation pairs from WMT 2016 dataset: Cs-En, De-En, Fi-En, Ro-En, RuEn and one other translation pair from WMT 2017 dataset: Lv-En.

We use the official released validation and test sets for each language pair.

The training data sizes of each language↔English pair are shown in the TAB2 : The training data size on the 6 languages↔English on the WMT dataset.

Ted Talk: We use the common corpus of TED talk which contains translations between multiple languages BID38 10 .

We select 44 languages in this corpus that has sufficient data for our experiments.

We use the official validation and test sets for each language pair.

The data sizes of the training set for each language↔English pair are listed in TAB2 2 LANGUAGE NAME AND CODEThe language names and their corresponding language codes according to ISO 639-1 standard 11 are listed in TAB2 .

@highlight

We proposed a knowledge distillation based method to boost the accuracy of multilingual neural machine translation.

@highlight

A many-to-one multilingual neural machine translation model that first training separate models for each language pair then performs distillation.

@highlight

The paper aims at training a machine translation model by augmenting the standard cross-entropy loss with a distillation component based on individual (single-language-pair) teacher models.