Work on the problem of contextualized word representation—the development of reusable neural network components for sentence understanding—has recently seen a  surge of progress centered on the unsupervised pretraining task of language modeling with methods like ELMo (Peters et al., 2018).

This paper contributes the first large-scale systematic study comparing different pretraining tasks in this context, both as complements to language modeling and as potential alternatives.

The primary results of the study support the use of language modeling as a pretraining task and set a new state of the art among comparable models using multitask learning with language models.

However, a closer look at these results reveals worryingly strong baselines and strikingly varied results across target tasks, suggesting that the widely-used paradigm of pretraining and freezing sentence encoders may not be an ideal platform for further work.

InputFigure 1: Our common model design: During pretraining, we train the shared encoder and the task-specific model for each pretraining task.

We then freeze the shared encoder and train the task-specific model anew for each target evaluation task.

Tasks may involve more than one sentence.

State-of-the-art models for natural language processing (NLP) tasks like translation, question answering, and parsing include components intended to extract representations for the meaning and contents of each input sentence.

These sentence encoder components are typically trained directly for the target task at hand.

This approach can be effective on data rich tasks and yields human performance on some narrowly-defined benchmarks BID35 BID13 , but it is tenable only for the few NLP tasks with millions of examples of training data.

This has prompted interest in pretraining for sentence encoding: There is good reason to believe it should be possible to exploit outside data and training signals to effectively pretrain these encoders, both because they are intended to primarily capture sentence meaning rather than any task-specific skill, and because we have seen dramatic successes with pretraining in the related domains of word embeddings and image encoders BID46 .More concretely, four recent papers show that pretrained sentence encoders can yield very strong performance on NLP tasks.

First, McCann et al. (2017) show that a BiLSTM encoder from a neural machine translation (MT) system can be effectively reused elsewhere.

BID16 , , and BID33 show that various kinds of encoder pretrained in an unsupervised fashion through generative language modeling (LM) are effective as well.

Each paper uses its own evaluation methods, though, making it unclear which pretraining task is most effective or whether multiple pretraining tasks can be productively combined; in the related setting of sentence-to-vector encoding, multitask learning with multiple labeled datasets has yielded a robust state of the art BID39 .

This paper attempts to systematically address these questions.

We train reusable sentence encoders on 17 different pretraining tasks, several simple baselines, and several combinations of these tasks, all using a single model architecture and procedure for pretraining and transfer, inspired by ELMo.

We then evaluate each of these encoders on the nine target language understanding tasks in the GLUE benchmark BID41 , yielding a total of 40 sentence encoders and 360 total trained models.

We then measure correlation in performance across target tasks and plot learning curves evaluating the effect of training data volume on each pretraining and target tasks.

Looking to the results of this experiment, we find that language modeling is the most effective single pretraining task we study, and that multitask learning during pretraining can offer further gains and a new state-of-the-art among fixed sentence encoders.

We also, however, find reasons to worry that ELMo-style pretraining, in which we pretrain a model and use it on target tasks with no further fine-tuning, is brittle and seriously limiting: (i) Trivial baseline representations do nearly as well as the best pretrained encoders, and the margins between substantially different pretraining tasks can be extremely small. (ii) Different target tasks differ dramatically on what kinds of pretraining they benefit most from, and multitask pretraining is not sufficient to circumvent this problem and offer general-purpose pretrained encoders.

Work toward learning reusable sentence encoders can be traced back at least as far as the multitask model of BID7 , but has seen a recent surge in progress with the successes of CoVe BID25 , ULMFit BID16 , ELMo , and the Transformer LM BID33 .

However, each uses a different model and dataset from the others, so while these works serve as existence proofs that effective reusable sentence encoders are possible, they do not address the question of what task or tasks should be used to create them.

The revival of interest in sentence encoder pretraining is recent enough that relatively little has been done to understand the relative merits of these models, though two exceptions stand out.

In unpublished work, Zhang & Bowman (2018) offer an analysis of the relative strengths of translation and language modeling using a single architecture and training dataset.

They find that encoders trained as language models reliably uncover the most syntactic structure, even when they are trained on a strict subset of the data used for a comparable translation model.

Peters et al. offer a deeper investigation of model design issues for ELMo, showing that all of the standard architectures for sentence encoding can be effectively pretrained with broadly similar performance, and that all learn reasonably good representations of the morphological and syntactic properties of sentences.

There has been a great deal of work on sentence-to-vector encoding, a setting in which the pretrained encoder produces a fixed-size vector representation for each input sentence BID10 BID20 BID14 BID8 BID45 .

These vectors are potentially useful for tasks that require fast similarity-based matching of sentences, but using them to replace sentence encoders trained in the conventional way on a given target text classification task does not reliably yield state-of-the art performance on that task BID39 .Multitask representation learning in NLP in general has been well studied, and again can be traced back at least as far as BID7 .

For example, BID23 show promising results from the combination of translation and parsing, BID39 show the benefits of multitask learning in sentence-to-vector encoding, and BID0 and BID4 offer studies of when multitask learning is helpful for lower-level NLP tasks.

Our main experiment compares encoders pretrained on a large number of tasks and task combinations, where a task is a dataset-objective function pair.

This section lists these tasks, which we select either to serve as baselines or because they have shown promise in outside prior work, especially prior work on sentence-to-vector encoding.

Appendix A includes additional details on how we implemented some of these tasks, and names tasks we evaluated but left out.

Random Encoder Our primary baseline is equivalent to pretraining on a task with zero examples.

Here, we randomly initialize a sentence encoder and use it directly with no further training.

This baseline works well, yielding scores far above those of a bag-of-words encoder.1 This surprising result matches results seen recently with ELMo-like models by Zhang & Bowman (2018) and earlier work on Reservoir Computing.

This baseline is especially strong because our model contains a skip connection from the input of the shared encoder to its output, allowing the task-specific model to directly see our word representations, or, in experiments where we use a pretrained ELMo model as our input layer, ELMo's contextual word representations.

We use the nine tasks included with GLUE as pretraining tasks: acceptability classification with CoLA BID42 ; binary sentiment classification with SST BID38 ; semantic similarity with the MSR Paraphrase Corpus (MRPC; BID11 , the Quora Question Pairs 2 (QQP), and STS-Benchmark (STS; BID3 ; and textual entailment with the Multi-Genre NLI Corpus (MNLI BID44 , RTE 1, 2, 3, and 5 (RTE; Dagan et al., 2006, et seq.) , and data from SQuAD (QNLI, BID34 and the Winograd Schema Challenge (WNLI, BID21 recast as entailment in the style of BID43 .

MNLI is the only task with substantial prior work in this area, as it was found to be highly effective as a pretraining strategy by BID8 and BID39 .

Other tasks are included to represent a broad sample of labeling schemes commonly used in NLP.

We train language models on two datasets: WikiText-103 (WP, BID26 and 1 Billion Word Language Model Benchmark (BWB, BID5 , which are used by ULMFit BID16 and ELMo respectively.

Translation We train MT models on two datasets: WMT14 English-German BID1 and WMT17 English-Russian BID2 .

SkipThought Our SkipThought model BID20 BID40 ) is a sequence-tosequence model that reads a sentence from WikiText-103 running text and attempts to decode the following sentence from that text.

We train our DisSent model BID17 BID29 to read two separate clauses that appear in WikiText-103 connected by a discourse marker such as and, but, or so and predict the identity of the discourse marker.

Reddit These models reconstruct comment threads from reddit.com using a dataset of about 18M comment-response pairs collected from 2008-2011 by BID45 .

We consider two settings: A classification task in which the model makes a binary prediction about whether a candidate response is the actual response to a given comment, and a sequence-to-sequence task in the model attempts to generate the true response to a comment.

We implement our models using the AllenNLP toolkit BID12 , aiming to build the simplest architecture that could be reasonably expected to perform well on the target tasks under study.

3 The design of the models roughly follows that used in the GLUE baselines and ELMo.

The core of our model is a two-layer 1024D bidirectional LSTM.

We feed the word representations to the biLSTM and take the sequence of hidden states from the top-level LSTM as the contextual representation.

The downstream task-specific model sees both the top-layer hidden states of this model and, through a skip connection, the input representations for each word.

All of our models use the pretrained character-level convolutional neural network (CNN) word encoder from ELMo .

This encoder acts as a standard input layer which uses no information beyond the word, and allows us to avoid potentially the difficult issues surrounding unknown word handling in transfer learning.

In some experiments, we use the full pretrained ELMo model as an input handler, yielding a form of multitask learning in which the lower layers of the overall model (ELMo) are pretrained on language modeling, and the higher layers (our shared encoder) are pretrained on some additional task or tasks.

We choose to use this pretrained model because it represents a larger model with more extensive tuning than we have the resources to produce ourselves.

We compare pretraining tasks in this setting to understand how well they complement large-scale language model pretraining, and we additionally train our own language models to directly compare between language modeling and other pretraining methods.

We follow the standard practice of training a set of scalar weights of ELMo's three layers.

We use one set of weights to supply input to the shared encoder, and an additional set for each target task to use in the skip connection.

We use only ELMo and not the similarly-situated CoVe, as BID41 showed CoVe to be less effective on the GLUE tasks.

Evaluation and Per-Task Models The GLUE benchmark BID41 ) is an open-ended shared task competition and evaluation toolkit for reusable sentence encoders, and we use it as our primary vehicle for evaluation.

GLUE is a set of nine classification or regression tasks over sentences and sentence pairs spanning a range of dataset sizes, paired with private test data and an online leaderboard.

GLUE offers a larger set of tasks than evaluated by ELMo or CoVe while omitting more expensive paragraph-level tasks, allowing us to evaluate a substantially larger number of experiments with available compute resources.

To evaluate the shared encoder, we use the following procedure: We freeze the pretrained encoder and, for each of the nine tasks in the GLUE benchmark, separately train a target-task model on the representations produced by the encoder.

We then evaluate each of these models on the validation or test set of the corresponding task using the standard metric(s) for that task, and report the resulting scores and the overall average GLUE scores, which weight each task equally.

For single-sentence target tasks (CoLA, SST) and sentence-pair tasks with smaller training datasets (MRPC, RTE, WNLI) we train a linear projection over the output states of the shared encoder, max-pool over those projected states, and feed the results to a one-hidden-layer classifier MLP.

For smaller sentence pair-tasks, we perform these steps on both sentences and use the heuristic matching feature vector [h 1 ; h 2 ; h 1 · h 2 ; h 1 − h 2 ] in the MLP, following BID28 .For the remaining sentence-pair tasks (MNLI, QNLI, QQP, STS), we use an attention mechanism between all pairs of words, followed by a 512D ×2 BiLSTM with max-pooling over time, following the basic mechanism used in BiDAF BID37 .

This is followed by heuristic matching and a final MLP, as above.

Appendices A and B present additional details on the task specific models.

Pretraining Task Models For pretraining on GLUE tasks, we use the architecture described above, except that we do not use an attention mechanism, as early results indicated that this hurt cross-task transfer performance.

For consistency with other experiments when pretraining on a GLUE task, we reinitialize the task-specific parameters between pretraining and target-task training.

Several of the outside (non-GLUE) pretraining tasks involve sentence pair classification.

For these, we use the same non-attentive architecture as for the larger GLUE tasks.

For LM, to prevent information leakage across directions and LSTM layers, we follow the broad strategy used by ELMo: We train separate forward and backward two-layer LSTM language models, and concatenate the outputs during target task training.

For sequence-to-sequence pretraining tasks (MT, SkipThought, Reddit), we use an LSTM decoder with a single layer.

We also investigate three sets of tasks for multitask pretraining: all GLUE tasks, all outside (non-GLUE) pretraining tasks, and all pretraining tasks.

Because ELMo representations are computed with the full context and so cannot be used as the input to downstream unidirectional language models, we exclude language modeling from multitask runs that use ELMo.

At each update during multitask learning, we randomly sample a single task with probability proportional to its training data size raised to the power of 0.75.

This sampling rate is meant to balance the risks of overfitting small-data tasks and underfitting large ones, and performed best in early exper-iments.

More extensive experiments with methods like this are shown in Appendix C. We perform early stopping based on an unweighted average of the pretraining tasks' validation metrics.

For validation metrics like perplexity that decrease from high starting values during training, we include the transformed metric 1 − m 250 in our average, tuning the constant 250 in early experiments.

Optimization We train our models with the AMSGrad optimizer (Reddi et al., 2018)-a variant of Adam BID19 .

We perform early stopping at pretraining time and target task training time using the respective dev set performances.

Typical experiments, including pretraining one encoder and training the nine associated target-task models, take 1-5 days to complete on an NVIDIA P100 GPU.

See Appendix B for more details.

Hyperparameter Tuning Appendix B describes our chosen hyperparameter values.

As our primary experiment required more than 100 GPU-days on NVIDIA P100 GPUs to run-not counting debugging or learning curves-we did not have the resources for extensive hyperparameter tuning.

Instead of carefully tuning our shared and task-specific models on a single pretraining task in a way that might bias results toward that task, we simply chose commonly-used values for most hyperparameters.

The choice not to tune limits our ability to diagnose the causes of poor performance when it occurs, and we invite readers to further refine our models using the public code.5 RESULTS TAB0 shows results on the GLUE dev set for all our pretrained encoders, each with and without the pretrained ELMo BiLSTM layers ( E ).

The N/A baselines are untrained encoders with random intialization.

The Single-Task baselines are aggregations of results from nine GLUE runs: The result in this row for a given GLUE task uses the encoder pretrained on only that task.

For consistency with other runs, we treat the pretraining task and the target task as two separate tasks in all cases (including here) and give them separate task-specific parameters, despite the fact that they use identical data.

We use S and C to distinguish the sequence-to-sequence and classification versions of the Reddit task, respectively.

To comply with GLUE's limits on test set access, we evaluated only three of our pretrained encoders on test data.

These reflect our best models with and without the use of the pretrained ELMo encoder, and with and without the use of GLUE data during pretraining.

For discussion of our limited hyperparameter tuning, see above.

For roughly-comparable GLUE results in prior work, see BID41 or https://www.gluebenchmark.com; we omit them here in the interest of space.

The limited size of a US Letter page prevent us from including these baselines in this table.

As of writing, the best test result using a comparable frozen pretrained encoder is 68.9 from BID41 for a model similar to our GLUE E multitask model, and the best overall result is 72.8 from BID33 with a model that is fine-tuned in its entirety for each target task.

While not feasible to run each setting multiple times, we estimate the variance of the GLUE score by re-running the random encoder and MNLI pretraining setups with and without ELMo with different random seeds.

Across five runs, we recorded σ = 0.4 for the random encoder (N/A in table), and σ = 0.2 for MNLI E .

This variation is substantial but not so high as to render results meaningless.

For the explicitly adversarial WNLI dataset (based on the Winograd Schema Challenge; BID21 , only one of our models reached even the most frequent class performance of 56.3.

In computing average and test set performances, we replace model predictions with the most frequent label to simulate the better performance achievable by choosing not to model that task.

Looking to other target tasks, the grammar-related CoLA task benefits dramatically from ELMo pretraining: The best result without language model pretraining is less than half the result achieved with such pretraining.

In contrast, the meaning-oriented textual similarity benchmark STS sees good results with several kinds of pretraining, but does not benefit substantially from the use of ELMo.

Comparing pretraining tasks in isolation without ELMo, language modeling performs best, followed by MNLI.

The remaining pretraining tasks yield performance near that of the random baseline.

Even when training directly on each target task (Single-Task in table), we get less than a one point gain over this simple baseline.

Adding ELMo yielded improvements in performance across all pretraining tasks.

MNLI and English-German translation perform best in this setting, with SkipThought, Reddit classification, and DisSent also outperforming the ELMo-augmented random baseline.

With ELMo, a multitask model performs best, but without it, all three multitask models are tied or outperformed by models trained on one of their constituent tasks, suggesting that our approach to multitask learning is not reliably able to produce models that productively use the knowledge taught by each training task.

However, of the two non-ELMo models that perform best on the development data, the multitask model generalizes better than the single-task model on test data for tasks like STS where the test set contains new out-of-domain data.

TAB1 presents an alternative view of the results of the main experiment TAB0 : The table shows the correlations between pairs of tasks over the space of pretrained encoders.

These reflect the degree to which knowing the performance of one target task with some encoder will allow us to predict the performance of the other target task with that same encoder.

Many correlations are low, suggesting that different tasks benefit from different forms of pretraining to a substantial degree, and mirroring the observation that no one pretraining task yields good performance on all target tasks.

As noted above, the models that tended to perform best overall also overfit the WNLI training set most, leading to a negative correlation between WNLI and overall GLUE score.

STS also shows a negative correlation, likely due to the observation that it does not benefit from ELMo pretraining.

In contrast, CoLA shows a strong 0.93 correlation with the overall GLUE scores, but has weak or negative correlations with many tasks-the use of ELMo or LM pretraining dramatically improves CoLA performance, but most other forms of pretraining have little effect.

FIG1 shows two types of learning curves.

The first set measures performance on the overall GLUE metric for encoders trained to convergence on each pretraining task with varying amounts of data.

The second set focuses on three pretrained encoders and measures performance on each GLUE target task separately with varying amounts of target task data.

Looking at pretraining tasks in isolation (top left), most tasks improve slightly as the amount of pretraining data increases, with the LM and MT tasks showing the most promising combination of slope and maximum performance.

Combining these pretraining tasks with ELMo (top right) yields a less interpretable result: the relationship between training data volume and performance becomes weaker, and some of the best results reported in this paper are achieved by models that combine pretrained ELMo with restricted-data versions of other pretraining tasks like MNLI and QQP.Looking at target task performance as target task training data volume varies, we see that all tasks benefit from increasing data quantities, with no obvious diminishing returns, and that most tasks see a constant improvement in performance across data volumes from the use of pretraining, either with ELMo (center) or with multitask learning (right).

Results on the GLUE Diagnostic Set From GLUE's auxiliary diagnostic analysis dataset, we find that ELMo and other forms of unsupervised pretraining helps on examples that involve world knowledge and lexical-semantic knowledge, and less so on examples that highlight complex sentence structures.

See TAB5 in Appendix D for more details.

This paper presents a systematic comparison of tasks and task-combinations for the pretraining of sentence-level BiLSTM encoders like those seen in ELMo and CoVe.

With 40 pretraining tasks and task combinations (not counting many more ruled out early) and nine target tasks, this represents a far more comprehensive study than any seen on this problem to date.

Our chief positive results are perhaps unsurprising: Language modeling works well as a pretraining task, and no other single task is consistently better.

Multitask pretraining can produce results better than any single task can, and sets a new state-of-the-art among comparable models.

Target task performance continues to improve with the addition of more language model data, even at large scales, suggesting that further work scaling up language model pretraining is warranted.

However, a closer look at our results suggests that the pretrain-and-freeze paradigm that underlies ELMo and CoVe might not be a sound platform for future work: Some trivial baselines do strikingly well, the margins between pretraining tasks are small, and some pretraining configurations (such as MNLI E ) yield better performance with less data.

This suggests that we may be nearing an upper bound on the performance that can be reached with methods like these.

In addition, different tasks benefit from different forms of pretraining to a striking degree-with correlations between target tasks often low or negative-and multitask pretraining tasks fail to reliably produce models better than their best individual components.

This suggests that if truly generalpurpose sentence encoders are possible, our current methods cannot produce them.

While further work on language modeling seems straightforward and worthwhile, the author(s) of this paper believe that the future of this line of work will require a better understanding of the ways in which neural network target task models can benefit from outside knowledge and data, and new methods for pretraining and transfer learning to allow them to do so.

DisSent To extract discourse model examples from the WikiText-103 corpus BID26 , we follow the procedure described in BID29 by extracting clause-pairs that follow specific dependency relationships within the corpus (see Figure 4 in BID29 .

We use the Stanford Parser BID6 distributed in Stanford CoreNLP version 3.9.1 to identify the relevant dependency arcs.

Reddit Response Prediction The Reddit classification task requires a model to select which of two candidate replies to a comment is correct.

Since the dataset from BID45 contains only real comment-reply pairs, we select an incorrect distractor reply for each correct reply by permuting each minibatch.

Alternative Tasks Any large-scale comparison like the one attempted in this paper is inevitably incomplete.

Among the thousands of publicly available NLP datasets, we also performed initial trial experiments on several datasets for which we were not able to reach development-set performance above that of the random encoder baseline in any setting.

These include image-caption matching with MSCOCO BID22 , following BID18 ; the small-to-medium-data textunderstanding tasks collected in NLI format by BID32 ; ordinal common sense inference (Zhang et al., 2017) ; POS tagging on the Penn Treebank BID24 ; and supertagging on CCGBank BID15 .

See Section 4 for general comments on hyperparameter tuning.

Validation We evaluate on the validation set for the current training task or tasks every 1,000 steps, except where noted otherwise for small-data target tasks.

During multitask learning, we multiply this interval by the number of tasks, evaluating every 9,000 steps during GLUE multitask training, for example.

Optimizer We use AMSGrad BID36 .

During pretraining, we use a learning rate of 1e-4 for classification and regression tasks, and 1e-3 for text generation tasks.

During target-task training, we use a learning rate of 3e-4 for all tasks.

Learning Rate Decay We multiply the learning rate by 0.5 whenever validation performance fails to improve for more than 4 validation checks.

We stop training if the learning rate falls below 1e-6.Early Stopping We maintain a saved checkpoint reflecting the best validation result seen so far.

We stop training if we see no improvement after more than 20 validation checks.

After training, we use the last saved checkpoint.

Regularization We apply dropout with a drop rate of 0.2 after the input layer (the character CNN or ELMo), after each LSTM layer, and after each MLP layer in the task-specific classifier or regressor.

For small-data target tasks, we increase MLP dropout to 0.4 during target-task training.

Preprocessing We use Moses tokenizer for encoder inputs, and set a maximum sequence length of 40 tokens.

There is no input vocabulary, as we use ELMo's character-based input layer.

For English text generation tasks, we use the Moses tokenizer to tokenize our data, but use a wordlevel output vocabulary of 20,000 types for tasks that require text generation.

For translation tasks, we use BPE tokenization with a vocabulary of 20,000 types.

For all sequence-to-sequence tasks we train word embeddings on the decoder side.

Target-Task-Specific Parameters To ensure that baseline performance for each target task is competitive, we find it necessary to use slightly different models and training regimes for larger and smaller target tasks.

We used partially-heuristic tuning to separate GLUE tasks into big-, mediumand small-data groups, giving each group its own heuristically chosen task-specific model specifications.

Exact values are shown in Table 3 .

Table 3 : Hyperparameter settings for target-task models and target-task training.

Attention is always disabled when pretraining on GLUE tasks.

STS has a relatively small training set, but consistently patterns with the larger tasks in its behavior.

Sequence-to-Sequence Models We found attention to be helpful for the SkipThought and Reddit pretraining tasks but not for machine translation, and report results for these configurations.

We use the max-pooled output of the encoder to initialize the hidden state of the decoder, and the size of this hidden state is equal to the size of the output of our shared encoder.

We reduce the dimension of the output of the decoder by half via a linear projection before the output softmax layer.

Our multitask learning experiments have three somewhat distinctive properties: (i) We mix tasks with very different amounts of training data-at the extreme, under 1,000 examples for WNLI, and over 1,000,000,000 examples from LM BWB. (ii) Our goal is to optimize the quality of the shared encoder, not the performance of any one of the tasks in the multitask mix. (iii) We mix a relatively large number of tasks, up to eighteen at once in some conditions.

These conditions make it challenging but important to avoid overfitting or underfitting any of our tasks.

Relatively little work has been done on this problem, so we conduct a small experiment here.

All our experiments use the basic paradigm of randomly sampling a new task to train on at each step, and we experiment with two hyperparameters that can be used to control over-and underfitting: The probability with which we sample each task and the weight with which we scale the loss for each task.

Our experiments follow the setup in Appendix B, and do not use the ELMo BiLSTM.Task Sampling We consider several approaches to determine the probability with which to sample a task during training, generally making this probability a function of the amount of data available for the task.

For task i with training set size N i , the probability is DISPLAYFORM0 where a is a constant.

Loss Scaling At each update, we scale the loss of a task with weight DISPLAYFORM1 Experiments For task sampling, we run experiments with multitask learning on the full set of nine GLUE tasks, as well as three subsets: single sentence tasks (S1: SST, CoLA), similarity and paraphrase tasks (S2: MRPC, STS, QQP), and inference tasks (S3: WNLI, QNLI, MNLI, RTE).

The results are shown in TAB3 .We also experiment with several combinations of task sampling and loss scaling methods, using only the full set of GLUE tasks.

The results are shown in TAB4 .While no combination of methods consistently offers dramatically better performance than any other, we observe that it is generally better to apply only one of non-uniform sampling and nonuniform loss scaling at a time rather than apply both simultaneously, as they provide roughly the same effect.

Following encouraging results from earlier pilot experiments, we use power 0.75 task sampling and uniform loss scaling in the multitask learning experiments shown in TAB0 .

TAB5 , below, shows results on the four coarse-grained categories of the GLUE diagnostic set for all our pretraining experiments.

This set consists of about 1000 expert-constructed examples in NLI While no model achieves near-human performance, the use of ELMo and other forms of unsupervised pretraining appears to be helpful on examples that highlight world knowledge and lexicalsemantic knowledge, and less so on examples that highlight complex logical reasoning patterns or alternations in sentence structure.

This relative weakness on sentence structure is somewhat surprising given the finding in Zhang & Bowman (2018) that language model pretraining is helpful for tasks involving sentence structure.

@highlight

We compare many tasks and task combinations for pretraining sentence-level BiLSTMs for NLP tasks. Language modeling is the best single pretraining task, but simple baselines also do well.