Recent developments in natural language representations have been accompanied by large and expensive models that leverage vast amounts of general-domain text through self-supervised pre-training.

Due to the cost of applying such models to down-stream tasks, several model compression techniques on pre-trained language representations have been proposed (Sun et al., 2019; Sanh, 2019).

However, surprisingly,  the simple baseline of just pre-training and fine-tuning compact models has been overlooked.

In this paper, we first show that pre-training remains important in the context of smaller architectures, and fine-tuning pre-trained compact models can be competitive to more elaborate methods proposed in concurrent work.

Starting with pre-trained compact models, we then explore transferring task knowledge from large fine-tuned models through standard knowledge distillation.

The resulting simple, yet effective and general algorithm, Pre-trained Distillation, brings further improvements.

Through extensive experiments, we more generally explore the interaction between pre-training and distillation under two variables that have been under-studied: model size and properties of unlabeled task data.

One surprising observation is that they have a compound effect even when sequentially applied on the same data.

To accelerate future research, we will make our 24 pre-trained miniature BERT models publicly available.

Self-supervised learning on a general-domain text corpus followed by end-task learning is the twostaged training approach that enabled deep-and-wide Transformer-based networks (Vaswani et al., 2017) to advance language understanding (Devlin et al., 2018; Yang et al., 2019b; Sun et al., 2019b; .

However, state-of-the-art models have hundreds of millions of parameters, incurring a high computational cost.

Our goal is to realize their gains under a restricted memory and latency budget.

We seek a training method that is well-performing, general and simple and can leverage additional resources such as unlabeled task data.

Before considering compression techniques, we start with the following research question: Could we directly train small models using the same two-staged approach?

In other words, we explore the idea of applying language model (LM) pre-training and task fine-tuning to compact architectures directly.

This simple baseline has so far been overlooked by the NLP community, potentially based on an underlying assumption that the limited capacity of compact models is capitalized better when focusing on the end task rather than a general language model objective.

Concurrent work to ours proposes variations of the standard pre-training+fine-tuning procedure, but with limited generality (Sun et al., 2019a; Sanh, 2019) .

We make the surprising finding that pre-training+fine-tuning in its original formulation is a competitive method for building compact models.

For further gains, we additionally leverage knowledge distillation (Hinton et al., 2015) , the standard technique for model compression.

A compact student is trained to recover the predictions of a highly accurate teacher.

In addition to the posited regularization effect of these soft labels (Hinton et al., 2015) , distillation provides a means of producing pseudo-labels for unlabeled data.

By regarding LM pre-training of compact models as a student initialization strategy, we can take advantage of both methods.

The resulting algorithm is a sequence of three standard training operations: masked LM (MLM) pre-training (Devlin et al., 2018) , task-specific distillation, and optional fine-tuning.

From here on, we will refer to it as Pre-trained Distillation (PD) ( Figure 1 ).

As we will show in Get loss L ??? ??? y P ??? (y|x) log P ?? (y|x) In a controlled study following data and model architecture settings in concurrent work (Section 4), we show that Pre-trained Distillation outperforms or is competitive with more elaborate approaches which use either more sophisticated distillation of task knowledge (Sun et al., 2019a) or more sophisticated pre-training from unlabeled text (Sanh, 2019) .

The former distill task knowledge from intermediate teacher activations, starting with a heuristically initialized student.

The latter fine-tune a compact model that is pre-trained on unlabeled text with the help of a larger LM teacher.

One of the most noteworthy contributions of our paper are the extensive experiments that examine how Pre-trained Distillation and its baselines perform under various conditions.

We investigate two axes that have been under-studied in previous work: model size and amount/quality of unlabeled data.

While experimenting with 24 models of various sizes (4m to 110m parameters) and depth/width trade-offs, we observe that pre-trained students can leverage depth much better than width; in contrast, this property is not visible for randomly-initialized models.

For the second axis, we vary the amount of unlabeled data, as well as its similarity to the labeled set.

Interestingly, Pretrained Distillation is more robust to these variations in the transfer set than standard distillation.

Finally, in order to gain insight into the interaction between LM pre-training and task-specific distillation, we sequentially apply these operations on the same dataset.

In this experiment, chaining the two operations performs better than any one of them applied in isolation, despite the fact that a single dataset was used for both steps.

This compounding effect is surprising, indicating that pre-training and distillation are learning complementary aspects of the data.

Given the effectiveness of LM pre-training on compact architectures, we will make our 24 pretrained miniature BERT models publicly available in order to accelerate future research.

Our high-level goal is to build accurate models which fit a given memory and latency budget.

There are many aspects to explore: the parametric form of the compact model (architecture, number of parameters, trade-off between number of hidden layers and embedding size), the training data (size, distribution, presence or absence of labels, training objective), etc.

Since an exhaustive search over this space is impractical, we fix the model architecture to bidirectional Transformers, known to be suitable for a wide range of NLP tasks (Vaswani et al., 2017; Devlin et al., 2018) .

The rest of this section elaborates on the training resources we assume to have at our disposal.

The teacher is a highly accurate but large model for an end task, that does not meet the resource constraints.

Prior work on distillation often makes use of an ensemble of networks (Hinton et al., 2015) .

For faster experimentation, we use a single teacher, without making a statement about the best architectural choice.

In Section 4, the teacher is pre-trained BERT BASE fine-tuned on labeled end-task data.

In Section 6, we use BERT LARGE instead.

Students are compact models that satisfy resource constraints.

Since model size qualifiers are relative (e.g., what is considered small in a data center can be impractically large on a mobile device), Labeled data (D L ) is a set of N training examples {(x 1 , y 1 ), ..., (x N , y N )}, where x i is an input and y i is a label.

For most NLP tasks, labeled sets are hard to produce and thus restricted in size.

Unlabeled transfer data (D T ) is a set of M input examples of the form {x 1 , ..., x M } sampled from a distribution that is similar to but possibly not identical to the input distribution of the labeled set.

During distillation, the teacher transfers knowledge to the student by exposing its label predictions for instances x m .

D T can also include the input portion of labeled data D L instances.

Due to the lack of true labels, such sets are generally easier to produce and consequently larger than labeled ones.

Note, however, that task-relevant input text is not readily available for key tasks requiring paired texts such as natural language inference and question answering, as well as domain-specific dialog understanding.

In addition, for deployed systems, input data distribution shifts over time and existing unlabeled data becomes stale ).

Unlabeled language model data (D LM ) is a collection of natural language texts that enable unsupervised learning of text representations.

We use it for unsupervised pre-training with a masked language model objective (Devlin et al., 2018) .

Because no labels are needed and strong domain similarity is not required, these corpora are often vast, containing thousands of millions of words.

The distinction between the three types of datasets is strictly functional.

Note they are not necessarily disjunct.

For instance, the same corpus that forms the labeled data can also be part of the unlabeled transfer set, after its labels are discarded.

Similarly, corpora that are included in the transfer set can also be used as unlabeled LM data.

Pre-trained Distillation (PD) (Figure 1 ) is a general, yet simple algorithm for building compact models that can leverage all the resources enumerated in Section 2.

It consists of a sequence of three standard training operations that can be applied to any choice of architecture:

1.

Pre-training on D LM .

A compact model is trained with a masked LM objective (Devlin et al., 2018) , capturing linguistic phenomena from a large corpus of natural language texts.

2.

Distillation on D T .

This well-read student is now prepared to take full advantage of the teacher expertise, and is trained on the soft labels (predictive distribution) produced by the teacher.

As we will show in Section 6.2, randomly initialized distillation is constrained by the size and distribution of its unlabeled transfer set.

However, the previous pre-training step mitigates to some extent the negative effects caused by an imperfect transfer set.

3. (Optional) fine-tuning on D L .

This step makes the model robust to potential mismatches between the distribution of the transfer and labeled sets.

We will refer to the two-step algorithm as PD, and to the three-step algorithm as PDF.

Step

Patient-KD Sanh (2019) BERT BASE truncated + LM-KD Fine-tuning Our test results are evaluated on the GLUE server, using the model that performed best on dev.

For anchoring, we also provide our results for MLM pre-training followed by fine-tuning (PF) and cite results from Sun et al. (2019a) for BERT BASE truncated and fine-tuned (TF).

The meta score is computed on 6 tasks only, and is therefore not directly comparable to the GLUE leaderboard.

While we are treating our large teachers as black boxes, it is worth noting that they are produced by pre-training and fine-tuning.

Since the teacher could potentially transfer the knowledge it has obtained via pre-training to the student through distillation, it is a priori unclear whether pre-training the student would bring additional benefits.

As Section 6.2 shows, pre-training students is surprisingly important, even when millions of samples are available for transfer.

There are concurrent efforts to ours aiming to leverage both pre-training and distillation in the context of building compact models.

Though inspired by the two-stage pre-training+fine-tuning approach that enabled deep-and-wide architectures to advance the state-of-the-art in language understanding, they depart from this traditional method in several key ways.

Patient Knowledge Distillation (Sun et al., 2019a ) initializes a student from the bottom layers of a deeper pre-trained model, then performs task-specific patient distillation.

The training objective relies not only on the teacher output, but also on its intermediate layers, thus making assumptions about the student and teacher architectures.

In a parallel line of work, DistilBert (Sanh, 2019) applies the same truncation-based initialization method for the student, then continues its LM pre-training via distillation from a more expensive LM teacher, and finally fine-tunes on task data.

Its downside is that LM distillation is computationally expensive, as it requires a softmax operation over the entire vocabulary to compute the expensive LM teacher's predictive distribution.

A common limitation in both studies is that the initialization strategy constrains the student to the teacher embedding size.

Table 2 summarizes the differences between concurrent work and Pre-trained Distillation (PD).

To facilitate direct comparison, in this section we perform an experiment with the same model architecture, sizes and dataset settings used in the two studies mentioned above.

We perform Pretrained Distillation on a 6-layer BERT student with task supervision from a 12-layer BERT BASE teacher, using embedding size 768 for both models.

For distillation, our transfer set coincides with the labeled set (D T = D L ).

Table 3 reports results on the 6 GLUE tasks selected by Sun et al. (2019a) and shows that, on average, PD performs best.

For anchoring, we also provide quality numbers for pre-training+fine-tuning (PF), which is surprisingly competitive to the more elaborate alternatives in this setting where D T is not larger than D L .

Remarkably, PF does not compromise generality or simplicity for quality.

Its downside is, however, that it cannot leverage unlabeled task data and teacher model predictions.

Given these positive results, we aim to gain more insight into Pre-trained Distillation.

We perform extensive analyses on two orthogonal axes-model sizes and properties of unlabeled data, thus departing from the settings used in Section 4.

All our models follow the Transformer architecture (Vaswani et al., 2017) and input processing used in BERT (Devlin et al., 2018) .

We denote the number of hidden layers as L and the hidden embedding size as H, and refer to models by their L/H dimensions.

We always fix the number of self-attention heads to H/64 and the feed-forward/filter size to 4H.

The end-task models are obtained by stacking a linear classifier on top of the Transformer architectures.

The teacher, BERT LARGE , has dimensions 24L/1024H and 340M parameters.

We experiment with 24 student models, with sizes and relative latencies listed in Table 1 .

The most expensive student, Transformer BASE , is 3 times smaller and 1.25 times faster than the teacher; the cheapest student, Transformer SMALL , is 77 times smaller and 65 times faster.

For readability, we report results on a selection of 5 students, but verify that all conclusions hold across the entire 24-model grid.

We select three baselines for Pre-trained Distillation that can provide insights into the contributions made by each of its constituent operations.

Basic Training (Figure 4a ) is the standard supervised learning method: a compact model is trained directly on the labeled set.

Knowledge Distillation (Figure 4b ) (Bucil?? et al., 2006; Hinton et al., 2015) (or simply "distillation") transfers information from a highly-parameterized and accurate teacher model to a more compact and thus less expressive student.

For classification tasks, distillation exposes the student to soft labels, namely the class probabilities produced by the teacher p l = softmax(z l /T ), where p l is the output probability for class l, z l is the logit for class l, and T is a constant called temperature that controls the smoothness of the output distribution.

The softness of the labels enables better generalization than the gold hard labels.

For each end task, we train: (i) a teacher obtained by fine-tuning pre-trained BERT LARGE (24L/1024H) on the labeled dataset (note teachers do not learn from the transfer set), and (ii) 24 students of various sizes.

Students are always distilled on the soft labels produced by the teacher with a temperature of 1 2 .

Pre-training+Fine-tuning (Figure 4c ) (Dai & Le, 2015; Devlin et al., 2018) , or simply PF, leverages large unlabeled general-domain corpora to pre-train models that can be fine-tuned for end tasks.

Following BERT, we perform pre-training with the masked LM (MLM) and next sentence objectives (collectively referred to as MLM + from here on).

The resulting model is fine-tuned on end-task labeled data.

While pre-training large models has been shown to provide substantial benefits, we are unaware of any prior work systematically studying its effectiveness on compact architectures.

Unlabeled Transfer Data (DT )

MNLI (390k) NLI* (1.3m samples) RTE (2.5k) NLI* (1.3m samples) SST-2 (68k) Movie Reviews* (1.7m samples) Book Reviews (50k) Book Reviews* (8m samples) The tasks and associated datasets are summarized in Table 4 .

Sentiment classification aims to classify text according to the polarities of opinions it contains.

We perform 3-way document classification on Amazon Book Reviews (He & McAuley, 2016) .

Its considerable size (8m) allows us to closely follow the standard distillation setting, where there is a large number of unlabeled examples for transfer.

Additionally, we test our algorithm on SST-2 (Socher et al., 2013) , which is a binary sentence classification task, and our results are directly comparable with prior work on the GLUE leaderboard (Wang et al., 2018) .

We use whole documents from Amazon Movie Reviews (1.7m) as unlabeled transfer data (note that SST-2 consists of single sentences).

Natural language inference involves classifying pairs of sentences (a premise and a hypothesis) as entailment, contradiction, or neutral.

This task is representative of the scenario in which proxy data is non-trivial to gather (Gururangan et al., 2018) .

We chose MNLI (Williams et al., 2018) as our target dataset.

Since strictly in-domain data is difficult to obtain, we supplement D T with two other sentence-pair datasets: SNLI (Bowman et al., 2015) and QQP (Chen et al., 2018) .

Textual entailment is similar to NLI, but restricted to binary classification (entailment vs nonentailment).

The most popular RTE dataset (Bentivogli et al., 2009 ) is two orders of magnitude smaller than MNLI and offers an extreme test of robustness to the amount of transfer data.

In this section, we conduct experiments that help us understand why Pre-trained Distillation is successful and how to attribute credit to its constituent operations.

As later elaborated in Section 7, earlier efforts to leverage pre-training in the context of compact models simply feed pre-trained (possibly contextual) input representations into randomly-initialized students (Hu et al., 2018; Chia et al., 2018; Tang et al., 2019) .

Concurrent work initializes shallowand-wide students from the bottom layers of their deeper pre-trained counterparts (Yang et al., 2019a; Sun et al., 2019a) .

The experiments below indicate these strategies are suboptimal, and that LM pre-training is necessary in order to unlock the full student potential.

Is it enough to pre-train word embeddings?

No.

In order to prove that pre-training Transformer layers is important, we compare two flavors of Pre-trained Distillation 3 : PD with pre-trained word embeddings and PD with pre-trained word embeddings and Transformer layers.

We produce wordpiece embeddings by pre-training one-layer Transformers for each embedding size.

We then discard the single Transformer layer and keep the embeddings to initialize our students.

For MNLI (Figure 5 ), less than 24% of the gains PD brings over distillation can be attributed to the pre-trained word embeddings (for Transformer TINY , this drops even lower, to 5%).

The rest of the benefits come from additionally pre-training the Transformer layers.

Students initialized via LM pre-training (green) outperform those initialized from the bottom layers of 12-layer pre-trained models (gray).

When only word embeddings are pre-trained (red), performance is degraded even further.

Is it worse to truncate deep pre-trained models?

Yes, especially for shallow students.

Given that pre-training is an expensive process, an exhaustive search over model sizes in the pursuit of the one that meets a certain performance threshold can be impractical.

Instead of pre-training all (number of layers, embedding size) combinations of students, one way of short-cutting the process is to pre-train a single deep (e.g. 12-layer) student for each embedding size, then truncate it at various heights.

Figure 5 shows that this can be detrimental especially to shallow architectures; Transformer TINY loses more than 73% of the pre-training gains over distillation.

As expected, losses fade away as the number of layers increases.

What is the best student for a fixed parameter size budget?

As a rule of thumb, prioritize depth over width, especially with pre-trained students.

Figure 6 presents a comparison between 24 student model architectures on SST-2, demonstrating how well different students utilize model capacity.

They are sorted first by the hidden size, then by the number of layers.

This roughly corresponds to a monotonic increase in the number of parameters, with a few exceptions for the largest students.

The quality of randomly initialized students (i.e. basic training and distillation) is closely correlated with the number of parameters.

With pre-training (i.e. PD and PF), we observe two intuitive findings: (1) pre-trained models are much more effective at using more parameters, and (2) pre-trained models are particularly efficient at utilizing depth, as indicated by the sharp drops in performance when moving to wider but shallower models.

This is yet another argument against initialization via truncation: for instance, truncating the bottom two layers of BERT BASE would lead to a suboptimal distribution of parameters: the 2L/768H model (39.2m parameters) is dramatically worse than e.g. 6L/512H (35.4m parameters).

In the previous section, we presented empirical evidence for the importance of the initial LM pretraining step.

In this section, we show that distillation brings additional value, especially in the presence of a considerably-sized transfer set, and that fine-tuning ensures robustness when the unlabeled data diverges from the labeled set.

Comparison to analysis baselines First, we quantify how much Pre-trained Distillation improves upon its constituent operations applied in isolation.

We compare it against the baselines established in Section 5.1 (basic training, distillation, and pre-training+fine-tuning) on the three NLP tasks Amazon Book Reviews Figure 7 : Comparison against analysis baselines.

Pre-trained Distillation out-performs all baselines: pretraining+fine-tuning, distillation, and basic training over five different student sizes.

Pre-training is performed on a large unlabeled LM set (BookCorpus & English Wikipedia).

Distillation uses the task-specific unlabeled transfer sets listed in Table 4 .

Teachers are pre-trained BERTLARGE, fine-tuned on labeled data.

described in Section 5.2.

We use the BookCorpus (Zhu et al., 2015) and English Wikipedia as our unlabeled LM set, following the same pre-training procedure as Devlin et al. (2018) .

Figure 7 confirm that PD outperforms these baselines, with particularly remarkable results on the Amazon Book Reviews corpus, where Transformer MINI recovers the accuracy of the teacher at a 31x decrease in model size and 16x speed-up.

Distillation achieves the same performance with Transformer BASE , which is 10x larger than Transformer MINI .

Thus PD can compress the model more effectively than distillation.

On RTE, Pre-trained Distillation improves Transformer TINY by more than 5% absolute over the closest baseline (pre-training+fine-tuning) and is the only method to recover teacher accuracy with Transformer BASE .

It is interesting to note that the performance of the baseline systems is closely related to the size of the transfer set.

For the sentence-pair tasks such as MNLI and RTE, where the size of the transfer set is moderate (1.3m) and slightly out-of-domain (see Table 4 ), pre-training+fine-tuning out-performs distillation across all student sizes, with an average of 12% for MNLI and 8% on RTE.

Interestingly, the order is inverted on Amazon Book Reviews, where the large transfer set (8m) is strictly indomain: distillation is better than pre-training+fine-tuning by an average of 3%.

On the other hand, Pre-trained Distillation is consistently best in all cases.

We will examine the robustness of Pre-trained Distillation in the rest of the section.

Robustness to transfer set size It is generally accepted that distillation is reliant upon a large transfer set.

For instance, distillation for speech recognition is performed on hundreds of millions of data points (Li et al., 2014; Hinton et al., 2015) .

We reaffirm this statement through experiments on Amazon Book Reviews in Figure 8 , given that Amazon Book Reviews have the biggest transfer set.

Distillation barely recovers teacher accuracy with the largest student (Transformer BASE ), using the entire 8m transfer set.

When there is only 1m transfer set, the performance is 4% behind the teacher model.

In contrast, PD achieves the same performance with Transformer MINI on 5m instances.

In other words, PD can match the teacher model with 10x smaller model and 1.5x less transfer data, compared to distillation.

Robustness to domain shift To the best of our knowledge, there is no prior work that explicitly studies how distillation is impacted by the mismatch between training and transfer sets (which we will refer to as domain shift).

Many previous distillation efforts focus on tasks where the two sets come from the same distribution (Romero et al., 2014; Hinton et al., 2015) , while others simply acknowledge the importance of and strive for a close match between them (Bucil?? et al., 2006) .

We provide empirical evidence that out-of-domain data degrades distillation and that our algorithm is more robust to mismatches between D L and D T .

We measure domain shift using the Spearman rank correlation coefficient (which we refer to as Spearman or simply S), introduced as a general metric in (Spearman, 1904) and first used as a corpus similarity metric in (Johansson et al., 1989) .

To compute corpus similarity, we follow the procedure described in (Kilgarriff & Rose, 1998) : for two datasets X and Y , we compute the corresponding frequency ranks F X and F Y of their most verify that distillation requires a large transfer set: 8m instances are needed to match the performance of the teacher using TransformerBASE.

PD achieves the same performance with TransformerMINI, on a 5m transfer set (10x smaller, 13x faster, 1.5x less data).

common n = 100 words.

For each of these words, the difference d between ranks in F X and F Y is computed.

The final statistic is given by the following formula:

To measure the effect of domain shift, we again experiment on the Amazon Book Reviews task.

Instead of varying the size of the transfer sets, this time we keep size fixed (to 1.7m documents) and vary the source of the unlabeled text used for distillation.

Transfer set domains vary from not task-related (paragraphs from Wikipedia with S=0.43), to reviews for products of unrelated category (electronics reviews with S=0.52), followed by reviews from a related category (movie reviews with S=0.76), and finally in-domain book reviews (S=1.0).

Results in Figure 9 show a direct correlation between accuracy and the Spearman coefficient for both distillation and PD.

When S drops to 0.43, distillation on D T is 1.8% worse than basic training on D L , whereas PD suffers a smaller loss over pre-training+fine-tuning, and a gain of about 1.5% when a final fine-tuning step is added.

When reviews from an unrelated product are used as a transfer set (S=0.52), PD obtains a much larger gain from learning from the teacher, compared to distillation.

PD outperforms the baselines even when we pre-train and distill on the same dataset (DLM = DT = NLI*).

We investigate the interaction between pretraining and distillation by applying them sequentially on the same data.

We compare the following two algorithms: Pre-training+Fine-tuning with D LM = X and Pre-trained Distillation with D LM = D T = X. Any additional gains that the latter brings over the former must be attributed to distillation, providing evidence that the compound effect still exists.

For MNLI, we set D LM = D T = NLI* and continue the experiment above by taking the students pre-trained on D LM = NLI* and distilling them on D T = NLI*. As shown in Figure  10 , PD is better than PF by 2.2% on average over all student sizes.

Note that even when pretraining and then distilling on the same data, PD outperforms the two training strategies applied in isolation.

The two methods are thus learning different linguistic aspects, both useful for the end task.

Pre-training Decades of research have shown that unlabeled text can help learn language representations.

Word embeddings were first used (Mikolov et al., 2013; Pennington et al., 2014) , while subsequently contextual word representations were found more effective (Peters et al., 2018) .

Most recently, research has shifted towards fine-tuning methods (Radford et al., 2018; Devlin et al., 2018; Radford et al., 2019) , where entire large pre-trained representations are fine-tuned for end tasks together with a small number of task-specific parameters.

While feature-based unsupervised representations have been successfully used in compact models (Johnson & Zhang, 2015; Gururangan et al., 2019) , inter alia, the pretraining+fine-tuning approach has not been studied in depth for such small models.

Learning compact models In this work we built on model compression (Bucil?? et al., 2006) and its variant knowledge distillation (Hinton et al., 2015) .

Other related efforts introduced ways to transfer more information from a teacher to a student model, by sharing intermediate layer activations (Romero et al., 2014; Yim et al., 2017; Sun et al., 2019a) .

We experimented with related approaches, but found only slight gains which were dominated by the gains from pre-training and were not complementary.

Prior works have also noted the unavailability of in-domain large-scale transfer data and proposed the use of automatically generated pseudo-examples (Bucil?? et al., 2006; Kimura et al., 2018) .

Here we showed that large-scale general domain text can be successfully used for pre-training instead.

A separate line of work uses pruning or quantization to derive smaller models (Han et al., 2016; Gupta et al., 2015) .

Gains from such techniques are expected to be complementary to PD.

Distillation with unsupervised pre-training Early efforts to leverage both unsupervised pretraining and distillation provide pre-trained (possibly contextual) word embeddings as inputs to students, rather than pre-training the student stack.

For instance, Hu et al. (2018) use ELMo embeddings, while (Chia et al., 2018; Tang et al., 2019) use context-independent word embeddings.

Concurrent work initializes Transformer students from the bottom layers of a 12-layer BERT model (Yang et al., 2019a; Sun et al., 2019a; Sanh, 2019) .

The latter continues student LM pre-training via distillation from a more expensive LM teacher.

For a different purpose of deriving a single model for multiple tasks through distillation, Clark et al. (2019) use a pre-trained student model of the same size as multiple teacher models.

However, none of the prior work has analyzed the impact of unsupervised learning for students in relation to the model size and domain of the transfer set.

<|TLDR|>

@highlight

Studies how self-supervised learning and knowledge distillation interact in the context of building compact models.

@highlight

Investigates training compact pre-trained language models via distillation and shows that using a teacher for distilling a compact student model performs better than directly pre-training the model.

@highlight

This submission shows that pre-training a student directly on masked language modeling is better than distillation, and the best is to combine both and distill from that pre-trained student model.