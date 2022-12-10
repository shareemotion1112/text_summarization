Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks.

However, at some point further model increases become harder due to GPU/TPU memory limitations, longer training times, and unexpected model degradation.

To address these problems,  we present two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT.

Comprehensive empirical evidence shows that our proposed methods lead to models that scale much better compared to the original BERT.

We also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs.

As a result, our best model establishes new state-of-the-art results on the GLUE, RACE, and SQuAD benchmarks while having fewer parameters compared to BERT-large.

Full network pre-training (Dai & Le, 2015; Radford et al., 2018; Howard & Ruder, 2018 ) has led to a series of breakthroughs in language representation learning.

Many nontrivial NLP tasks, including those that have limited training data, have greatly benefited from these pre-trained models.

One of the most compelling signs of these breakthroughs is the evolution of machine performance on a reading comprehension task designed for middle and high-school English exams in China, the RACE test (Lai et al., 2017) : the paper that originally describes the task and formulates the modeling challenge reports then state-of-the-art machine accuracy at 44.1%; the latest published result reports their model performance at 83.2% ; the work we present here pushes it even higher to 89.4%, a stunning 45.3% improvement that is mainly attributable to our current ability to build high-performance pretrained language representations.

Evidence from these improvements reveals that a large network is of crucial importance for achieving state-of-the-art performance .

It has become common practice to pre-train large models and distill them down to smaller ones (Sun et al., 2019; Turc et al., 2019) for real applications.

Given the importance of model size, we ask: Is having better NLP models as easy as having larger models?

An obstacle to answering this question is the memory limitations of available hardware.

Given that current state-of-the-art models often have hundreds of millions or even billions of parameters, it is easy to hit these limitations as we try to scale our models.

Training speed can also be significantly hampered in distributed training, as the communication overhead is directly proportional to the number of parameters in the model.

We also observe that simply growing the hidden size of a model such as BERT-large can lead to worse performance.

Table 1 and Fig. 1 show a typical example, where we simply increase the hidden size of BERT-large to be 2x larger and get worse results with this BERT-xlarge model.

Model Hidden Size Parameters RACE (Accuracy) BERT-large 1024 334M 72.0% BERT-large (ours) 1024 334M 73.9% BERT-xlarge (ours) 2048 1270M 54.3% Table 1 :

Increasing hidden size of BERT-large leads to worse performance on RACE.

Existing solutions to the aforementioned problems include model parallelization (Shoeybi et al., 2019) and clever memory management (Chen et al., 2016; .

These solutions address the memory limitation problem, but not the communication overhead and model degradation problem.

In this paper, we address all of the aforementioned problems, by designing A Lite BERT (ALBERT) architecture that has significantly fewer parameters than a traditional BERT architecture.

ALBERT incorporates two parameter reduction techniques that lift the major obstacles in scaling pre-trained models.

The first one is a factorized embedding parameterization.

By decomposing the large vocabulary embedding matrix into two small matrices, we separate the size of the hidden layers from the size of vocabulary embedding.

This separation makes it easier to grow the hidden size without significantly increasing the parameter size of the vocabulary embeddings.

The second technique is cross-layer parameter sharing.

This technique prevents the parameter from growing with the depth of the network.

Both techniques significantly reduce the number of parameters for BERT without seriously hurting performance, thus improving parameter-efficiency.

An ALBERT configuration similar to BERT-large has 18x fewer parameters and can be trained about 1.7x faster.

The parameter reduction techniques also act as a form of regularization that stabilizes the training and helps with generalization.

To further improve the performance of ALBERT, we also introduce a self-supervised loss for sentence-order prediction (SOP).

SOP primary focuses on inter-sentence coherence and is designed to address the ineffectiveness of the next sentence prediction (NSP) loss proposed in the original BERT.

As a result of these design decisions, we are able to scale up to much larger ALBERT configurations that still have fewer parameters than BERT-large but achieve significantly better performance.

We establish new state-of-the-art results on the well-known GLUE, SQuAD, and RACE benchmarks for natural language understanding.

Specifically, we push the RACE accuracy to 89.4%, the GLUE benchmark to 89.4, and the F1 score of SQuAD 2.0 to 92.2.

Learning representations of natural language has been shown to be useful for a wide range of NLP tasks and has been widely adopted (Mikolov et al., 2013; Le & Mikolov, 2014; Dai & Le, 2015; Peters et al., 2018; Radford et al., 2018; 2019) .

One of the most significant changes in the last two years is the shift from pre-training word embeddings, whether standard (Mikolov et al., 2013; Pennington et al., 2014) or contextualized (McCann et al., 2017; Peters et al., 2018) , to full-network pre-training followed by task-specific fine-tuning (Dai & Le, 2015; Radford et al., 2018; .

In this line of work, it is often shown that larger model size improves performance.

For example, show that across three selected natural language understanding tasks, using larger hidden size, more hidden layers, and more attention heads always leads to better performance.

However, they stop at a hidden size of 1024.

We show that, under the same setting, increasing the hidden size to 2048 leads to model degradation and hence worse performance.

Therefore, scaling up representation learning for natural language is not as easy as simply increasing model size.

In addition, it is difficult to experiment with large models due to computational constraints, especially in terms of GPU/TPU memory limitations.

Given that current state-of-the-art models often have hundreds of millions or even billions of parameters, we can easily hit memory limits.

To address this issue, Chen et al. (2016) propose a method called gradient checkpointing to reduce the memory requirement to be sublinear at the cost of an extra forward pass.

propose a way to reconstruct each layer's activations from the next layer so that they do not need to store the intermediate activations.

Both methods reduce the memory consumption at the cost of speed.

In contrast, our parameter-reduction techniques reduce memory consumption and increase training speed.

The idea of sharing parameters across layers has been previously explored with the Transformer architecture (Vaswani et al., 2017) , but this prior work has focused on training for standard encoderdecoder tasks rather than the pretraining/finetuning setting.

Different from our observations, Dehghani et al. (2018) show that networks with cross-layer parameter sharing (Universal Transformer, UT) get better performance on language modeling and subject-verb agreement than the standard transformer.

Very recently, Bai et al. (2019) propose a Deep Equilibrium Model (DQE) for transformer networks and show that DQE can reach an equilibrium point for which the input embedding and the output embedding of a certain layer stay the same.

Our observations show that our embeddings are oscillating rather than converging.

Hao et al. (2019) combine a parameter-sharing transformer with the standard one, which further increases the number of parameters of the standard transformer.

ALBERT uses a pretraining loss based on predicting the ordering of two consecutive segments of text.

Several researchers have experimented with pretraining objectives that similarly relate to discourse coherence.

Coherence and cohesion in discourse have been widely studied and many phenomena have been identified that connect neighboring text segments (Hobbs, 1979; Halliday & Hasan, 1976; Grosz et al., 1995) .

Most objectives found effective in practice are quite simple.

Skipthought and FastSent (Hill et al., 2016) sentence embeddings are learned by using an encoding of a sentence to predict words in neighboring sentences.

Other objectives for sentence embedding learning include predicting future sentences rather than only neighbors (Gan et al., 2017) and predicting explicit discourse markers (Jernite et al., 2017; Nie et al., 2019) .

Our loss is most similar to the sentence ordering objective of Jernite et al. (2017) , where sentence embeddings are learned in order to determine the ordering of two consecutive sentences.

Unlike most of the above work, however, our loss is defined on textual segments rather than sentences.

BERT uses a loss based on predicting whether the second segment in a pair has been swapped with a segment from another document.

We compare to this loss in our experiments and find that sentence ordering is a more challenging pretraining task and more useful for certain downstream tasks.

Concurrently to our work, also try to predict the order of two consecutive segments of text, but they combine it with the original next sentence prediction in a three-way classification task rather than empirically comparing the two.

In this section, we present the design decisions for ALBERT and provide quantified comparisons against corresponding configurations of the original BERT architecture .

The backbone of the ALBERT architecture is similar to BERT in that it uses a transformer encoder (Vaswani et al., 2017) with GELU nonlinearities (Hendrycks & Gimpel, 2016) .

We follow the BERT notation conventions and denote the vocabulary embedding size as E, the number of encoder layers as L, and the hidden size as H. Following Devlin et al. (2019), we set the feed-forward/filter size to be 4H and the number of attention heads to be H/64.

There are three main contributions that ALBERT makes over the design choices of BERT.

Factorized embedding parameterization.

In BERT, as well as subsequent modeling improvements such as XLNet and RoBERTa , the WordPiece embedding size E is tied with the hidden layer size H, i.e., E ≡ H. This decision appears suboptimal for both modeling and practical reasons, as follows.

From a modeling perspective, WordPiece embeddings are meant to learn context-independent representations, whereas hidden-layer embeddings are meant to learn context-dependent representations.

As experiments with context length indicate , the power of BERT-like representations comes from the use of context to provide the signal for learning such context-dependent representations.

As such, untying the WordPiece embedding size E from the hidden layer size H allows us to make a more efficient usage of the total model parameters as informed by modeling needs, which dictate that H E.

From a practical perspective, natural language processing usually require the vocabulary size V to be large.

1 If E ≡ H, then increasing H increases the size of the embedding matrix, which has size V × E. This can easily result in a model with billions of parameters, most of which are only updated sparsely during training.

Therefore, for ALBERT we use a factorization of the embedding parameters, decomposing them into two smaller matrices.

Instead of projecting the one-hot vectors directly into the hidden space of size H, we first project them into a lower dimensional embedding space of size E, and then project it to the hidden space.

By using this decomposition, we reduce the embedding parameters from

.

This parameter reduction is significant when H E. We choose to use the same E for all word pieces because they are much more evenly distributed across documents compared to whole-word embedding, where having different embedding size (Grave et al. (2017) ; Baevski & Auli (2018) ; ) for different words is important.

Cross-layer parameter sharing.

For ALBERT, we propose cross-layer parameter sharing as another way to improve parameter efficiency.

There are multiple ways to share parameters, e.g., only sharing feed-forward network (FFN) parameters across layers, or only sharing attention parameters.

The default decision for ALBERT is to share all parameters across layers.

All our experiments use this default decision unless otherwise specified We compare this design decision against other strategies in our experiments in Sec. 4.5.

Similar strategies have been explored by Dehghani et al. (2018) (Universal Transformer, UT) and Bai et al. (2019) (Deep Equilibrium Models, DQE) for Transformer networks.

Different from our observations, Dehghani et al. (2018) show that UT outperforms a vanilla Transformer.

Bai et al. (2019) show that their DQEs reach an equilibrium point for which the input and output embedding of a certain layer stay the same.

Our measurement on the L2 distances and cosine similarity show that our embeddings are oscillating rather than converging.

Figure 2 shows the L2 distances and cosine similarity of the input and output embeddings for each layer, using BERT-large and ALBERT-large configurations (see Table 2 ).

We observe that the transitions from layer to layer are much smoother for ALBERT than for BERT.

These results show that weight-sharing has an effect on stabilizing network parameters.

Although there is a drop for both metrics compared to BERT, they nevertheless do not converge to 0 even after 24 layers.

This shows that the solution space for ALBERT parameters is very different from the one found by DQE.

Inter-sentence coherence loss.

In addition to the masked language modeling (MLM) loss , BERT uses an additional loss called next-sentence prediction (NSP).

NSP is a binary classification loss for predicting whether two segments appear consecutively in the original text, as follows: positive examples are created by taking consecutive segments from the training corpus; negative examples are created by pairing segments from different documents; positive and negative examples are sampled with equal probability.

The NSP objective was designed to improve performance on downstream tasks, such as natural language inference, that require reasoning about the relationship between sentence pairs.

However, subsequent studies found NSP's impact unreliable and decided to eliminate it, a decision supported by an improvement in downstream task performance across several tasks.

We conjecture that the main reason behind NSP's ineffectiveness is its lack of difficulty as a task, as compared to MLM.

As formulated, NSP conflates topic prediction and coherence prediction in a single task 2 .

However, topic prediction is easier to learn compared to coherence prediction, and also overlaps more with what is learned using the MLM loss.

We maintain that inter-sentence modeling is an important aspect of language understanding, but we propose a loss based primarily on coherence.

That is, for ALBERT, we use a sentence-order prediction (SOP) loss, which avoids topic prediction and instead focuses on modeling inter-sentence coherence.

The SOP loss uses as positive examples the same technique as BERT (two consecutive segments from the same document), and as negative examples the same two consecutive segments but with their order swapped.

This forces the model to learn finer-grained distinctions about discourse-level coherence properties.

As we show in Sec. 4.6, it turns out that NSP cannot solve the SOP task at all (i.e., it ends up learning the easier topic-prediction signal, and performs at randombaseline level on the SOP task), while SOP can solve the NSP task to a reasonable degree, presumably based on analyzing misaligned coherence cues.

As a result, ALBERT models consistently improve downstream task performance for multi-sentence encoding tasks.

We present the differences between BERT and ALBERT models with comparable hyperparameter settings in Table 2 .

Due to the design choices discussed above, ALBERT models have much smaller parameter size compared to corresponding BERT models.

For example, ALBERT-large has about 18x fewer parameters compared to BERT-large, 18M versus 334M.

If we set BERT to have an extra-large size with H = 2048, we end up with a model that has 1.27 billion parameters and under-performs (Fig. 1) .

In contrast, an ALBERT-xlarge configuration with H = 2048 has only 60M parameters, while an ALBERT-xxlarge configuration with H = BERT  base  108M  12  768  768  False  large  334M  24  1024  1024  False  xlarge  1270M  24  2048  2048  False   ALBERT   base  12M  12  768  128  True  large  18M  24  1024  128  True  xlarge  60M  24  2048  128  True  xxlarge  235M  12  4096  128  True   Table 2 : The configurations of the main BERT and ALBERT models analyzed in this paper.

4096 has 233M parameters, i.e., around 70% of BERT-large's parameters.

Note that for ALBERTxxlarge, we mainly report results on a 12-layer network because a 24-layer network (with the same configuration) obtains similar results but is computationally more expensive.

This improvement in parameter efficiency is the most important advantage of ALBERT's design choices.

Before we can quantify this advantage, we need to introduce our experimental setup in more detail.

To keep the comparison as meaningful as possible, we follow the BERT setup in using the BOOKCORPUS and English Wikipedia for pretraining baseline models.

These two corpora consist of around 16GB of uncompressed text.

We format our inputs as "

, where x 1 = x 1,1 , x 1,2 · · · and x 2 = x 1,1 , x 1,2 · · · are two segments.

We always limit the maximum input length to 512, and randomly generate input sequences shorter than 512 with a probability of 10%.

Like BERT, we use a vocabulary size of 30,000, tokenized using SentencePiece (Kudo & Richardson, 2018) as in XLNet .

We generate masked inputs for the MLM targets using n-gram masking , with the length of each n-gram mask selected randomly.

The probability for the length n is given by p(n) = 1/n N k=1 1/k We set the maximum length of n-gram (i.e., n) to be 3 (i.e., the MLM target can consist of up to a 3-gram of complete words, such as "White House correspondents").

All the model updates use a batch size of 4096 and a LAMB optimizer with learning rate 0.00176 (You et al., 2019) .

We train all models for 125,000 steps unless otherwise specified.

Training was done on Cloud TPU V3.

The number of TPUs used for training ranged from 64 to 1024, depending on model size.

The experimental setup described in this section is used for all of our own versions of BERT as well as ALBERT models, unless otherwise specified.

To monitor the training progress, we create a development set based on the development sets from SQuAD and RACE using the same procedure as in Sec. 4.1.

We report accuracies for both MLM and sentence classification tasks.

Note that we only use this set to check how the model is converging; it has not been used in a way that would affect the performance of any downstream evaluation, such as via model selection.

Following Yang et al. (2019) and , we evaluate our models on three popular benchmarks: The General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2018) , two versions of the Stanford Question Answering Dataset (SQuAD; , and the ReAding Comprehension from Examinations (RACE) dataset (Lai et al., 2017) .

For completeness, we provide description of these benchmarks in Appendix A.1.

As in , we perform early stopping on the development sets, on which we report all comparisons except for our final comparisons based on the task leaderboards, for which we also report test set results.

We are now ready to quantify the impact of the design choices described in Sec. 3, specifically the ones around parameter efficiency.

The improvement in parameter efficiency showcases the most important advantage of ALBERT's design choices, as shown in Table 3 : with only around 70% of BERT-large's parameters, ALBERT-xxlarge achieves significant improvements over BERT-large, as measured by the difference on development set scores for several representative downstream tasks: SQuAD v1.1 (+1.9%), SQuAD v2.0 (+3.1%), MNLI (+1.4%), SST-2 (+2.2%), and RACE (+8.4%).

We also observe that BERT-xlarge gets significantly worse results than BERT-base on all metrics.

This indicates that a model like BERT-xlarge is more difficult to train than those that have smaller parameter sizes.

Another interesting observation is the speed of data throughput at training time under the same training configuration (same number of TPUs).

Because of less communication and fewer computations, ALBERT models have higher data throughput compared to their corresponding BERT models.

The slowest one is the BERT-xlarge model, which we use as a baseline.

As the models get larger, the differences between BERT and ALBERT models become bigger, e.g., ALBERT-xlarge can be trained 2.4x faster than BERT-xlarge.

Table 3 : Dev set results for models pretrained over BOOKCORPUS and Wikipedia for 125k steps.

Here and everywhere else, the Avg column is computed by averaging the scores of the downstream tasks to its left (the two numbers of F1 and EM for each SQuAD are first averaged).

Next, we perform ablation experiments that quantify the individual contribution of each of the design choices for ALBERT.

Table 4 shows the effect of changing the vocabulary embedding size E using an ALBERT-base configuration setting (see Table 2 ), using the same set of representative downstream tasks.

Under the non-shared condition (BERT-style), larger embedding sizes give better performance, but not by much.

Under the all-shared condition (ALBERT-style), an embedding of size 128 appears to be the best.

Based on these results, we use an embedding size E = 128 in all future settings, as a necessary step to do further scaling.

Table 5 presents experiments for various cross-layer parameter-sharing strategies, using an ALBERT-base configuration (Table 2 ) with two embedding sizes (E = 768 and E = 128).

We compare the all-shared strategy (ALBERT-style), the not-shared strategy (BERT-style), and intermediate strategies in which only the attention parameters are shared (but not the FNN ones) or only the FFN parameters are shared (but not the attention ones).

Table 4 : The effect of vocabulary embedding size on the performance of ALBERT-base.

The all-shared strategy hurts performance under both conditions, but it is less severe for E = 128 (-1.5 on Avg) compared to E = 768 (-2.5 on Avg).

In addition, most of the performance drop appears to come from sharing the FFN-layer parameters, while sharing the attention parameters results in no drop when E = 128 (+0.1 on Avg), and a slight drop when E = 768 (-0.7 on Avg).

There are other strategies of sharing the parameters cross layers.

For example, We can divide the L layers into N groups of size M , and each size-M group shares parameters.

Overall, our experimental results shows that the smaller the group size M is, the better the performance we get.

However, decreasing group size M also dramatically increase the number of overall parameters.

We choose all-shared strategy as our default choice.

Table 5 : The effect of cross-layer parameter-sharing strategies, ALBERT-base configuration.

We compare head-to-head three experimental conditions for the additional inter-sentence loss: none (XLNet-and RoBERTa-style), NSP (BERT-style), and SOP (ALBERT-style), using an ALBERTbase configuration.

Results are shown in The results on the intrinsic tasks reveal that the NSP loss brings no discriminative power to the SOP task (52.0% accuracy, similar to the random-guess performance for the "None" condition).

This allows us to conclude that NSP ends up modeling only topic shift.

In contrast, the SOP loss does solve the NSP task relatively well (78.9% accuracy), and the SOP task even better (86.5% accuracy).

Even more importantly, the SOP loss appears to consistently improve downstream task performance for multi-sentence encoding tasks (around +1% for SQuAD1.1, +2% for SQuAD2.0, +1.7% for RACE), for an Avg score improvement of around +1%.

In this section, we check how depth (number of layers) and width (hidden size) affect the performance of ALBERT.

Table 7 shows the performance of an ALBERT-large configuration (see Table 2 ) using different numbers of layers.

Networks with 3 or more layers are trained by fine-tuning using the parameters from the depth before (e.g., the 12-layer network parameters are fine-tuned from the checkpoint of the 6-layer network parameters).

4 Similar technique has been used in Gong et al. (2019) .

If we compare a 3-layer ALBERT model with a 1-layer ALBERT model, although they have the same number of parameters, the performance increases significantly.

However, there are diminishing returns when continuing to increase the number of layers: the results of a 12-layer network are relatively close to the results of a 24-layer network, and the performance of a 48-layer network appears to decline.

Table 7 : The effect of increasing the number of layers for an ALBERT-large configuration.

A similar phenomenon, this time for width, can be seen in Table 8 for a 3-layer ALBERT-large configuration.

As we increase the hidden size, we get an increase in performance with diminishing returns.

At a hidden size of 6144, the performance appears to decline significantly.

We note that none of these models appear to overfit the training data, and they all have higher training and development loss compared to the best-performing ALBERT configurations.

Table 8 : The effect of increasing the hidden-layer size for an ALBERT-large 3-layer configuration.

The speed-up results in Table 3 indicate that data-throughput for BERT-large is about 3.17x higher compared to ALBERT-xxlarge.

Since longer training usually leads to better performance, we perform a comparison in which, instead of controlling for data throughput (number of training steps), we control for the actual training time (i.e., let the models train for the same number of hours).

In Table 9 , we compare the performance of a BERT-large model after 400k training steps (after 34h of training), roughly equivalent with the amount of time needed to train an ALBERT-xxlarge model with 125k training steps (32h of training).

Table 9 : The effect of controlling for training time, BERT-large vs ALBERT-xxlarge configurations.

After training for roughly the same amount of time, ALBERT-xxlarge is significantly better than BERT-large: +1.5% better on Avg, with the difference on RACE as high as +5.2%.

In Section 4.7, we show that for ALBERT-large (H=1024), the difference between a 12-layer and a 24-layer configuration is small.

Does this result still hold for much wider ALBERT configurations, such as ALBERT-xxlarge (H=4096)?

Number of layers SQuAD1.

Table 10 : The effect of a deeper network using an ALBERT-xxlarge configuration.

The answer is given by the results from Table 10 .

The difference between 12-layer and 24-layer ALBERT-xxlarge configurations in terms of downstream accuracy is negligible, with the Avg score being the same.

We conclude that, when sharing all cross-layer parameters (ALBERT-style), there is no need for models deeper than a 12-layer configuration.

The experiments done up to this point use only the Wikipedia and BOOKCORPUS datasets, as in .

In this section, we report measurements on the impact of the additional data used by both XLNet and RoBERTa .

Fig. 3a plots the dev set MLM accuracy under two conditions, without and with additional data, with the latter condition giving a significant boost.

We also observe performance improvements on the downstream tasks in Table 11 , except for the SQuAD benchmarks (which are Wikipedia-based, and therefore are negatively affected by out-of-domain training material).

Table 11 : The effect of additional training data using the ALBERT-base configuration.

We also note that, even after training for 1M steps, our largest models still do not overfit to their training data.

As a result, we decide to remove dropout to further increase our model capacity.

The plot in Fig. 3b shows that removing dropout significantly improves MLM accuracy.

Intermediate evaluation on ALBERT-xxlarge at around 1M training steps (Table 12 ) also confirms that removing dropout helps the downstream tasks.

There is empirical (Szegedy et al., 2017) and theoretical evidence showing that a combination of batch normalization and dropout in Convolutional Neural Networks may have harmful results.

To the best of our knowledge, we are the first to show that dropout can hurt performance in large Transformer-based models.

However, the underlying network structure of ALBERT is a special case of the transformer and further experimentation is needed to see if this phenomenon appears with other transformer-based architectures or not.

Table 12 : The effect of removing dropout, measured for an ALBERT-xxlarge configuration.

The results we report in this section make use of the training data used by , as well as the additional data used by and Yang et al. (2019) .

We report state-of-the-art results under two settings for fine-tuning: single-model and ensembles.

In both settings, we only do single-task fine-tuning

.

Following , on the development set we report the median result over five runs.

The single-model ALBERT configuration incorporates the best-performing settings discussed: an ALBERT-xxlarge configuration (Table 2 ) using combined MLM and SOP losses, and no dropout.

The checkpoints that contribute to the final ensemble model are selected based on development set performance; the number of checkpoints considered for this selection range from 6 to 17, depending on the task.

For the GLUE (Table 13 ) and RACE (Table 14) benchmarks, we average the model predictions for the ensemble models, where the candidates are fine-tuned from different training steps using the 12-layer and 24-layer architectures.

For SQuAD (Table 14) , we average the prediction scores for those spans that have multiple probabilities; we also average the scores of the "unanswerable" decision.

Both single-model and ensemble results indicate that ALBERT improves the state-of-the-art significantly for all three benchmarks, achieving a GLUE score of 89.4, a SQuAD 2.0 test F1 score of 92.2, and a RACE test accuracy of 89.4.

The latter appears to be a particularly strong improvement, a jump of +17.4% absolute points over BERT , +7.6% over XLNet , +6.2% over RoBERTa , and 5.3% over DCMI+ , an ensemble of multiple models specifically designed for reading comprehension tasks.

Our single model achieves an accuracy of 86.5%, which is still 2.4% better than the state-of-the-art ensemble model.

Table 13 : State-of-the-art results on the GLUE benchmark.

For single-task single-model results, we report ALBERT at 1M steps (comparable to RoBERTa) and at 1.5M steps.

The ALBERT ensemble uses models trained with 1M, 1.5M, and other numbers of steps.

While ALBERT-xxlarge has less parameters than BERT-large and gets significantly better results, it is computationally more expensive due to its larger structure.

An important next step is thus to speed up the training and inference speed of ALBERT through methods like sparse attention and block attention (Shen et al., 2018 ).

An orthogonal line of research, which could provide additional representation power, includes hard example mining (Mikolov et al., 2013) efficient language modeling training .

Additionally, although we have convincing evidence that sentence order prediction is a more consistently-useful learning task that leads to better language representations, we hypothesize that there could be more dimensions not yet captured by the current self-supervised training losses that could create additional representation power for the resulting representations.

RACE RACE is a large-scale dataset for multi-choice reading comprehension, collected from English examinations in China with nearly 100,000 questions.

Each instance in RACE has 4 candidate answers.

Following prior work , we use the concatenation of the passage, question, and each candidate answer as the input to models.

Then, we use the representations from the "[CLS]" token for predicting the probability of each answer.

The dataset consists of two domains: middle school and high school.

We train our models on both domains and report accuracies on both the development set and test set.

A.2 HYPERPARAMETERS Hyperparameters for downstream tasks are shown in Table 15 .

We adapt these hyperparameters from , , and Yang et al. (2019

<|TLDR|>

@highlight

A new pretraining method that establishes new state-of-the-art results on the GLUE, RACE, and SQuAD benchmarks while having fewer parameters compared to BERT-large. 