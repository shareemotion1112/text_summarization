The current state-of-the-art end-to-end semantic role labeling (SRL) model is a deep neural network architecture with no explicit linguistic features.

However, prior work has shown that gold syntax trees can dramatically improve SRL, suggesting that neural network models could see great improvements from explicit modeling of syntax.

In this work, we present linguistically-informed self-attention (LISA): a new neural network model that combines  multi-head self-attention with multi-task learning across dependency parsing, part-of-speech, predicate detection and SRL.

For example, syntax is incorporated by training one of the attention heads to attend to syntactic parents for each token.

Our model can predict all of the above tasks, but it is also trained such that if a high-quality syntactic parse is already available, it can be beneficially injected at test time without re-training our SRL model.

In experiments on the CoNLL-2005 SRL dataset LISA achieves an increase of 2.5 F1 absolute over the previous state-of-the-art on newswire with predicted predicates and more than 2.0 F1 on out-of-domain data.

On ConLL-2012 English SRL we also show an improvement of more than 3.0 F1, a 13% reduction in error.

Semantic role labeling (SRL) extracts a high-level representation of meaning from a sentence, labeling e.g. who did what to whom.

Explicit representations of such semantic information have been shown to improve results in challenging downstream tasks such as dialog systems BID63 BID14 , machine reading BID8 BID65 and machine translation BID36 BID5 .Though syntax was long considered an obvious prerequisite for SRL systems BID34 BID51 , recently deep neural network architectures have surpassed syntacticallyinformed models BID69 BID25 BID60 , achieving state-of-the art SRL performance with no explicit modeling of syntax.

Still, recent work BID53 BID25 indicates that neural network models could see even higher performance gains by leveraging syntactic information rather than ignoring it.

BID25 indicate that many of the errors made by a strong syntax-free neural-network on SRL are tied to certain syntactic confusions such as prepositional phrase attachment, and show that while constrained inference using a relatively low-accuracy predicted parse can provide small improvements in SRL accuracy, providing a gold-quality parse leads to very significant gains.

incorporate syntax from a highquality parser BID31 using graph convolutional neural networks BID32 , but like BID25 they attain only small increases over a model with no syntactic parse, and even perform worse than a syntax-free model on out-of-domain data.

These works suggest that though syntax has the potential to improve neural network SRL models, we have not yet designed an architecture which maximizes the benefits of auxiliary syntactic information.

In response, we propose linguistically-informed self-attention (LISA): a model which combines multi-task learning BID12 with stacked layers of multi-head self-attention BID64 trained to act as an oracle providing syntactic parses to downstream parameters tasked with predicting semantic role labels.

Our model is endto-end: earlier layers are trained to predict prerequisite parts-of-speech and predicates, which are supplied to later layers for scoring.

The model is trained such that, as syntactic parsing models improve, providing high-quality parses at test time can only improve its performance, allowing the model to benefit maximally from improved parsing models without requiring re-training.

Unlike previous work, we encode each sentence only once, predict its predicates, part-of-speech tags and syntactic parse, then predict the semantic roles for all predicates in the sentence in parallel, leading to exceptionally fast training and decoding speeds: our model matches state-of-the art accuracy in less than one quarter the training time.

In extensive experiments on the CoNLL-2005 and CoNLL-2012 datasets, we show that our linguistically-informed models consistently outperform the syntax-free state-of-the-art for SRL models with predicted predicates.

On CoNLL-2005, our single model out-performs the previous state-of-the-art single model on the WSJ test set by nearly 1.5 F1 points absolute using its own predicted parses, and by 2.5 points using a stateof-the-art parse (Dozat and Manning, 2017) .

On the challenging out-of-domain Brown test set, our model also improves over the previous state-ofthe-art by more than 2.0 F1.

On CoNLL-2012, our model gains 1.4 points with its own parses and more than 3.0 points absolute over previous work: 13% reduction in error.

Our single models also out-perform state-of-the-art ensembles across all datasets, up to more than 1.4 F1 over a strong fivemodel ensemble on CoNLL-2012.

Our goal is to design an efficient neural network model which makes use of linguistic information as effectively as possible in order to perform endto-end SRL.

LISA achieves this by combining: (1) Multi-task learning across four related tasks; (2) a new technique of supervising neural attention to predict syntactic dependencies; and (3) careful conditioning of different parts of the model on gold versus predicted annotations during training.

Figure 1 depicts the overall architecture of our model.

To first encode rich token-level representations, our neural network model takes word embeddings as input, which are passed through stacked convolutional, feed-forward and multihead self-attention layers BID64 to efficiently produce contextually encoded token embeddings (Eqns.

1-4).

We choose this combination of network components because we found it to perform better than LSTM, CNN or self-attention layers alone in terms of speed-accuracy Pareto efficiency in initial experiments.

To predict semantic role labels, the contextually encoded tokens are projected to distinct predicate and role embeddings ( §2.4), and each predicted predicate is scored with the sequence's role representations using a bilinear model (Eqn.

5), producing per-label scores for BIO-encoded semantic role labels for each token and each semantic frame in the sequence entirely in parallel.

To incorporate syntax, one self-attention head is trained to attend to each token's syntactic parent, allowing the model to use this attention head as an oracle for syntactic dependencies.

We encourage the model to use this syntactic information as much as possible by giving subsequent layers access to a gold parse oracle during training, allowing either the predicted parse attention or an externally predicted parse to be used at test time.

We introduce this syntactically-informed self-attention in more detail in §2.2.We integrate part-of-speech and predicate information into earlier layers by re-purposing representations closer to the input to predict predicates and part-of-speech (POS) tags ( §2.3).

We simplify optimization and benefit from shared statistical strength derived from highly correlated POS and predicates by treating tagging and predicate detection as a single task, performing multi-class classification into the joint Cartesian product space of POS and predicate labels.

The model is trained end-to-end by maximum likelihood using stochastic gradient descent ( §2.5).

The input to the network is a sequence X of T token representations x t .

Each token representation is the sum of a fixed (pre-trained) and learned (randomly initialized) word embedding.

In the case where we feed a predicate indicator embedding p t as input to the network, we concatenate that representation with the word embedding to give the final token embedding.

h -1 Figure 1 : Word embeddings are input to k CNN layers.

This output is passed to (1) a joint POS/predicate classifier and (2) j layers of multi-head selfattention.

One attention head is trained to attend to parse parents.

A bilinear operation scores distinct predicate and role representations to produce BIOencoded SRL predictions.

DISPLAYFORM0 These token representations are then the input to a series of width-3 stacked convolutional layers with residual connections BID24 , producing contextually embedded token representations c (k) t at each layer k. We denote the kth convolutional layer as C (k) .

Let r(·) denote the leaky ReLU activation function BID39 , and let LN (·) denote layer normalization BID2 , then starting with input x t , the final CNN output is given by the recurrence: DISPLAYFORM1 We use leaky ReLU activations to avoid dead activations and vanishing gradients BID26 , whereas layer normalization reduces covariate shift between layers BID28 without requiring distinct train-and testtime operations.

We then feed this representation as input to a series of residual multi-head self-attention layers with feed-forward connections in the style of the encoder portion of the Transformer architecture of BID64 .

This architecture allows each token to observe long-distance context from the entire sentence like an LSTM, but unlike an LSTM, representations for all tokens can be computed in parallel at each layer.

We first project 1 the output of the convolutional layers to a representation c (p) t that is the same size as the output of the self-attention layers and add a positional encoding vector computed as a deterministic sinusoidal function of t, following BID64 .

We then apply the selfattention layers to this projected representation, applying layer normalization after each residual connection.

Denoting the jth self-attention layer as T (j) (·), the output of that layer s (j) t , and h as the number of attention heads at each layer, the 1 All of our linear projections include bias terms, which we omit in this exposition for the sake of clarity.following recurrence applied to initial input c DISPLAYFORM2 gives our final token representations s DISPLAYFORM3 with which we perform a weighted sum of the value vectors a value vh for each other token v to compose a new token representation for each attention head.

The representations for each attention head are concatenated into a single vector a t .

We feed this representation through a multi-layer perception, add it to the initial representation and apply layer normalization to give the final output of selfattention layer j: DISPLAYFORM4 2.2 Syntactically-informed self-attention Typically, neural attention mechanisms are left on their own to learn to attend to relevant inputs.

Instead, we propose training the self-attention to attend to specific tokens corresponding to the syntactic structure of the sentence as a mechanism for passing linguistic knowledge to later layers.

Specifically, we train with an auxiliary objective on one attention head which encourages that head to attend to each token's parent in a syntactic dependency tree.

We use the attention weights a th between token t and each other token q in the sequence as the distribution over possible heads for token t: P (q = head(t) | X ) = a thq , where we define the root token as having a self-loop.

This attention head thus emits a directed graph 2 where each token's head is the token to which the attention assigns the highest weight.

This attention head now becomes an oracle for syntax, denoted P, providing a dependency parse to downstream layers.

This model not only predicts its own dependency arcs, but allows for the injection of auxiliary parse information at test time by simply swapping out the oracle given by a th to one produced by e.g. a state-of-the-art parser.

In this way, our model can benefit from improved, external parsing models without re-training.

Unlike typical multi-task models, ours maintains the ability to leverage external syntactic information.

Unfortunately, this parsing objective does not maximize the model's ability to use the syntactic information for predicting semantic role labels in later layers.

Though one would expect model accuracy to increase significantly if injecting e.g. gold dependency arcs into the learned attention head at test time, we find that without specialized training this is not the case: Without the training described below, fixing P to gold parses at test time improves SRL F1 over predicted parses by 0.3 points, whereas the F1 increases by 7.0 when the model is trained with our technique.

3 Injecting high-accuracy predicted parses follows the same trend.

We hypothesize that the model is limited by the poor representations to which it has access during early training.

When training begins, the model observes randomly initialized attention rather than strong syntactic information, even in the head which will be trained to provide it with such information.

Thus rather than learning to look to this head for syntax, the model learns to encode that information itself, like a model which was trained with no explicit syntax at all.

Prior work BID68 , has alleviated this problem by pre-training the parameters of earlier tasks before initiating the training of later tasks.

However, optimization in this setting becomes computationally expensive and complicated, especially as the number of auxiliary tasks increases, and when using adaptive techniques for stochastic gradient descent such as Adam BID30 .To alleviate this problem, during training we 2 In most but not all cases, the head emits a tree, but we do not currently enforce it.3 CoNLL-2012.

CoNLL-2005 yields similar results.clamp P to the gold parse (P G ) when using its representation for later layers, while still training a th to predict syntactic heads.

We find that this vastly improves the model's ability to leverage the parse information encoded in P at test time.

Our approach is essentially an extension of teacher forcing BID66 to MTL.

Though a large body of work suggests that, by closing the gap between observed data distributions during train and test, training on predicted rather than gold labels leads to improved test-time accuracy BID21 BID52 BID15 BID22 BID13 BID6 BID4 , our simple approach works surprisingly well; we leave more advanced scheduled sampling techniques to future work.

We also share the parameters of lower layers in our model to predict POS tags and predicates.

Following He et al. FORMULA1 , we focus on the end-toend setting, where predicates must be predicted on-the-fly.

Since we also train our model to predict syntactic dependencies, it is beneficial to give the model some knowledge of POS information.

While much previous work employs a pipelined approach to both POS tagging for dependency parsing and predicate detection for SRL, we take a multi-task learning (MTL) approach BID12 , sharing the parameters of earlier layers in our SRL model with a joint POS and predicate detection objective.

Since POS is a strong predictor of predicates, 4 and the complexity of training a multi-task model increases with the number of tasks, we combine POS tagging and predicate detection into a joint label space: for each POS tag TAG in the training data which co-occurs with a predicate, we add a label of the form TAG:PREDICATE.Specifically, we experiment with feeding a lower-level representation, r t , which may be either c (k) t , the output of the convolutional layers, or s (1) t , the output of the first self-attention layer, to a linear classifier.

We compute locally-normalized probabilities using the softmax function: P (z t | X ) ∝ exp(r t ), where z t is a label in the joint space.

We apply this supervision at earlier lay-ers following prior work BID54 BID23 .

Our final goal is to predict semantic roles for each predicate in the sequence.

We score each predicate 5 against each token in the sequence using a bilinear operation, producing per-label scores for each token for each predicate, with predicates and syntax determined by oracles V and P.First, we project each token representation s (j) t to a predicate-specific representation s pred t and a role-specific representation s role t .

We then provide these representations to a bilinear transformation U for scoring.

So, the role label scores s f t for the token at index t with respect to the predicate at index f (i.e. token t and frame f ) are given by: DISPLAYFORM0 which can be computed in parallel across all semantic frames in an entire minibatch.

We calculate a locally normalized distribution over role labels for token t in frame f using the softmax function: DISPLAYFORM1 At test time, we perform constrained decoding using the Viterbi algorithm to emit valid sequences of BIO tags, using unary scores s f t and the transition probabilities given by the training data.

We maximize the sum of the likelihoods of the individual tasks, entrusting the network to learn parameters which model the complex coupling between tasks, rather than explicitly modeling structure in the output labels: DISPLAYFORM0 where λ is a penalty on the syntactic attention loss.

Note that as described in §2.2, the terms for the syntactically-informed attention and joint predicate/POS prediction are conditioned only on the input sequence X , whereas the SRL component is conditioned on gold predicates V G and gold parse structure P G during training.

We train the model using Nadam (Dozat, 2016) SGD combined with the learning rate schedule in BID64 .

In addition to MTL, we regularize our model using element-wise and word dropout BID55 BID20 and parameter averaging.

We use gradient clipping to avoid exploding gradients BID7 BID45 .

Our models are implemented in TensorFlow BID0 with source code and models to be released upon publication.

Additional details on optimization and hyperparameters are included in Appendix A.

Early approaches to SRL BID50 BID56 BID29 BID61 focused on developing rich sets of linguistic features as input to a linear model, often combined with complex constrained inference e.g. with an ILP BID51 .

BID59 showed that constraints could be enforced more efficiently using a clever dynamic program for exact inference.

BID57 modeled syntactic parsing and SRL jointly, and BID35 jointly modeled SRL and CCG parsing.

BID19 were among the first to use a neural network model for SRL, a CNN over word embeddings which failed to out-perform non-neural models.

FitzGerald et al. (2015) successfully employed neural networks by embedding lexicalized features and providing them as factors in the model of BID59 .More recent neural models are syntax-free.

BID69 , and BID25 Some work has incorporated syntax into neural models for SRL.

BID53 incorporate syntax by embedding dependency paths, and similarly encode syntax using a graph CNN over a predicted syntax tree, out-performing models without syntax on CoNLL-2009.

However, both models are at risk of over-fitting to or otherwise inheriting the flaws of the predictions upon which they are trained.

Indeed, report that their model does not out-perform a similar syntaxfree model on out-of-domain data.

Syntactically-informed self-attention is similar to the concurrent work of BID37 , who use edge marginals produced by the matrixtree algorithm as attention weights for document classification and natural language inference.

MTL BID12 ) is popular in NLP.

Collobert et al. FORMULA1 multi-task part-of-speech, chunking, NER and SRL.

BID68 jointly train a dependency parser and POS tagger.

Søgaard and Goldberg (2016) train a multitask model for POS, chunking and CCG tagging.

BID23 built a single, jointly trained model for POS, chunking, parsing, semantic relatedness and entailment, using a special regularization scheme to facilitate training.

BID9 and Alonso and Plank (2017) investigate different combinations of NLP tagging tasks including POS, chunking and FrameNet semantics (Baker, 2014).

BID38 enhance a machine translation model by multitasking with parsing.

MTL has also been applied to semantic dependency parsing: BID58 multi-task with a syntax-based tagging objective while BID46 train on three semantic dependency frameworks.

The question of training on gold versus predicted labels is closely related to learning to search BID21 BID52 BID13 and scheduled sampling BID6 , with applications in NLP to sequence labeling and transition-based parsing BID15 BID22 BID4 .

We believe more sophisticated approaches extending these techniques to MTL could improve LISA in future work.

We present results on the CoNLL-2005 shared task BID11 and the CoNLL-2012 English subset of OntoNotes 5.0 BID49 , achieving state-of-the-art results for a single model with predicted predicates on both corpora.

In all experiments, we initialize with pre-trained GloVe word embeddings BID47 , hyperparameters that resulted in the best performance on the validation set were selected via a small grid search, and models were trained for a maximum of 7 days on one TitanX GPU using early stopping on the validation set.

6 For CoNLL-2005 we convert constituencies to dependencies using the Stanford head rules v3.5 (de Marneffe and Manning, 2008) and for CoNLL-2012 we use ClearNLP (Choi and Palmer, 2012b), following previous work.

A detailed description of hyperparameter settings and data pre-processing can be found in Appendix A.For both datasets, we compare our best models (LISA G ) to three strong sets of baselines: the syntax-free deep LSTM model of BID25 which was the previous state-of-the-art model for SRL with predicted predicates, both as an ensemble of five models (PoE) and as a single model (single); an ablation of our own self-attention model where we don't incorporate any syntactic information (SA), and another ablation where we do train with syntactically-informed self-attention, but where downstream layers in the model are conditioned on the predicted attention weights (i.e. dynamic oracle, D) rather than the gold parse (G) during training (LISA D ).We demonstrate that our models can benefit from injecting state-of-the-art predicted parses at test time (+D&M) by setting the attention oracle to parses predicted by Dozat and Manning (2017) , the state-of-the-art dependency parser for English PTB and winner of the 2017 CoNLL shared task BID67 .

In all cases, using these parses at test time improves performance.

We also evaluate our model using the gold syntactic parse at test time (+Gold), to provide an upper bound for the benefit that syntax could have for SRL using LISA.

These experiments show that despite LISA's strong performance, there remains substantial room for improvement.

In §4.4 we perform detailed analysis comparing SRL models us-6 Our best reported CoNLL-2012 model was trained for just under 6 days, though it matched BID25 ing gold and predicted parses to better understand where syntax provides the most benefit to SRL, and what remains to be improved.

We first report the unlabeled attachment scores (UAS) of our parsing models on the CoNLL-2005 and 2012 SRL test sets (Table 1) .

Dozat and Manning (2017) achieves the best scores, obtaining state-of-the-art results on the CoNLL-2012 split of OntoNotes in terms of UAS, followed by LISA G then LISA D .

7 We still see SRL accuracy improvements despite our relatively low parser UAS from LISA's predicted parses, but the difference in accuracy likely explains the large increase in SRL we see from decoding with D&M parses.

TAB3 reports precision, recall and F1 on the CoNLL-2012 test set.

Our SA model already performs strongly without access to syntax, out-performing the single model of BID25 but under-performing their ensemble.

Adding syntactically-informed training to the self- 7 The previous best score we know of is 92.5 attained by Mate BID10 , as reported in BID18 .

attention increases over the model without syntax, achieving about the same score using dynamic versus gold parse oracles for downstream layers during training.

When evaluating using an injected parse, we see that (1) a large increase of more than 1.5 F1 absolute for LISA G and (2) this increase is markedly larger than for LISA D .

With the injected D&M parse, our single models impressively outperform the ensemble.

We also report predicate detection precision, recall and F1 TAB5 ).

Our models obtain much higher scores than BID25 on this task, likely explaining improvements of our basic SA model over theirs.

Like BID25 , our model achieves much higher precision than recall, indicative of the model memorizing predicate words from the training data.

Interestingly, our SA model out-performs syntax-infused models by a small margin.

We hypothesize that this could be due to asking the LISA models to learn to predict more tasks, taking some model capacity away from predicate detection.

TAB7 lists precision, recall and F1 on the CoNLL-2005 test sets.

Unlike on CoNLL-2012, our SA baseline does not out-perform BID25 .

This is likely due to their predicate detection scores being closer to ours on this data (Table 5).

Interestingly, unlike on CoNLL-2012 we see a distinct improvement between LISA G and LISA D in models which use LISA parses: LISA G training leads to improved SRL scores by more than 1 F1 absolute using LISA-predicted parses.

Similar to CoNLL-2012, we see very little improvement from adding D&M parses at test-time with the dynamic oracle, whereas we obtain the highest score of all when using D&M parses combined with LISA G , demonstrating that our training technique markedly improves LISA's ability to leverage improved parses at test time.

Our best single models out-perform the ensemble of (Table 5) : all our models out-perform the baseline in terms of F1.

In the case of CoNLL-2005 BID25 attains higher recall, especially on the Brown test set, while our model achieves higher precision.

We report only LISA G since there is little difference across *SA models.

LISA in its current form does not perform as well when gold predicates are given at test time.

Table 6 presents LISA G performance with predicate indicator embeddings provided on the input.

On neither test set does our model using LISA parses out-perform the state-of-the-art.

With D&M parses, our models out-perform BID25 , but not BID60 .We attribute this behavior to two factors.

First, the models of BID25 and BID60 are larger than our models.

8 .

Our models were designed to predict predicates, and we found the current model size sufficient for good performance in this setting.

Second, our model encodes each sequence only once, while the works to which we compare re-encode the sequence anew for each predicate.

Since our model predicts its own predicates using a shared sentence encoding, it is impossible to encode sequences in this way.

We also do not enforce that the model assign the correct predicate label during decoding, leading to incorrect predicate predictions despite gold predicate inputs.

For example, in a challenging sentence which contains two distinct semantic frames with the identical predicate continued, our model incorrectly predicts both tokens as predicates in one of the frames.

With more careful modeling toward gold predicates, our technique could be improved for this setting.

Still, LISA shows impressive performance when gold predicates are not available, as when using SRL in the wild.

In §4.2 and §4.3 we observed that while LISA performs well with state-of-the-art predicted syntax, it still sees a large gain across all datasets of 4-5 F1 points absolute when provided with gold syntax trees.

In order to better understand the nature of these improvements, we perform a detailed model analysis based on that of BID25 First, we compare the impact of Viterbi decoding with LISA, D&M, and gold syntax trees TAB10 , finding the same trends across both datasets.

While Viterbi decoding makes a larger difference over greedy decoding with LISA parses than with D&M, we find that Viterbi has the exact same impact for D&M and gold parses: Gold parses provide no improvement over state-of-the-art predicted parses in terms of BIO label consistency.

We also assess SRL F1 as a function of sentence length.

In FIG3 we see that providing LISA with gold parses is particularly helpful for sentences longer than 10 tokens.

This likely directly follows from the tendency of syntactic parsers to perform worse on longer sentences.

Next, we compare SRL error types.

Following BID25 , we apply a series of corrections to model predictions in order to understand which error types the gold parse resolves: e.g. Fix Labels fixes labels on spans which match gold boundaries, whereas Merge Spans merges adjacent predicted spans into a gold span.

9 In Figure 3 we see that much of the performance gap between the gold and predicted parses is due to span boundary errors (Merge Spans, Split Spans and Fix Span Boundary), which supports the hypothesis proposed by BID25 that incorporating syntax could be particularly helpful for resolving these errors.

BID25 that these errors are due mainly to prepositional phrase (PP) attachment mistakes.

We also find this to be the case: Figure 4 shows a breakdown of split/merge corrections by phrase type.

Though the number of corrections decreases substantially across phrase types, the proportion of corrections attributed to PPs remains the same (approx.

50%) even after providing the correct PP attachment to the model, indicating that PP span boundary mistakes are due not only to parse mistakes, but are a fundamental difficulty for SRL.

We present linguistically-informed self-attention: a new multi-task neural network model that effectively incorporates rich linguistic information for semantic role labeling.

LISA out-performs the state-of-the-art on two benchmark SRL datasets, including out-of-domain, while training more than 4× faster.

Future work will explore improving LISA's parsing accuracy, developing better training techniques and adapting to more tasks.

Following previous work BID25 , we evaluate our models on the CoNLL-2012 data split BID49 of OntoNotes 5.0 BID27 .

10 This dataset is drawn from seven domains: newswire, web, broadcast news and conversation, magazines, telephone conversations, and text from the bible.

The text is annotated with gold part-of-speech, syntactic constituencies, named entities, word sense, speaker, co-reference and semantic role labels based on the PropBank guidelines BID44 .

Propositions may be verbal or nominal, and there are 41 distinct semantic role labels, excluding continuation roles and including the predicate.

We processed the data as follows: We convert the semantic proposition and role segmentations to BIO boundary-encoded tags, resulting in 129 distinct BIO-encoded tags (including continuation roles).

We initialize word embeddings with 100d pre-trained GloVe embeddings trained on 6 billion tokens of Wikipedia and Gigaword BID47 .

Following the experimental setup for parsing from Choi et al. FORMULA1 , we convert constituency structure to dependencies using the ClearNLP dependency converter BID17 , use automatic part-of-speech tags assigned by the ClearNLP tagger BID16 , and exclude single-token sentences in our parsing evaluation.

10 We constructed the data split following instructions at: BID11 ) is based on the original PropBank corpus BID44 , which labels the Wall Street Journal portion of the Penn TreeBank corpus (PTB) BID42 with predicateargument structures, plus a challenging out-ofdomain test set derived from the Brown corpus (Francis and Kučera, 1964) .

This dataset contains only verbal predicates, though some are multiword verbs, and 28 distinct role label types.

We obtain 105 SRL labels including continuations after encoding predicate argument segment boundaries with BIO tags.

We evaluate the SRL performance of our models using the srl-eval.pl script provided by the CoNLL-2005 shared task, 11 which computes segment-level precision, recall and F1 score.

We also report the predicate detection scores output by this script.

For CoNLL-2005 we train the same parser as for CoNLL-2012 except on the typical split of the WSJ portion of the PTB using Stanford dependencies (de Marneffe and Manning, 2008) and POS tags from the Stanford CoreNLP left3words model BID62 .

We train on WSJ sections 02-21, use section 22 for development and section 23 for test.

This corresponds to the same train/test split used for propositions in the CoNLL-2005 dataset, except that section 24 is used for development rather than section 22.

We train the model using the Nadam (Dozat, 2016) algorithm for adaptive stochastic gradient descent (SGD), which combines Adam BID30 SGD with Nesterov momentum BID43 .

We additionally vary the learning rate lr as a function of an initial learning rate lr 0 and the current training step step as described in Vaswani which increases the learning rate linearly for the first warm training steps, then decays it proportionally to the inverse square root of the step number.

We found this learning rate schedule essential for training the self-attention model.

We only update optimization moving-average accumulators for parameters which receive gradient updates at a given step.

12 In all of our experiments we used initial learning rate 0.04, β 1 = 0.9, β 2 = 0.98, = 1 × 10 −12 and dropout rates of 0.33 everywhere.

We use four self-attention layers made up of 8 attention heads each with embedding dimension 64, and two CNN layers with filter size 1024.

The size of all MLP projections: In the feed-forward portion of self-attention, predicate and role representations, and representation used for joint partof-speech/predicate classification is 256.

We train with warm = 4000 warmup steps and clip gradient norms to 5.

<|TLDR|>

@highlight

Our combination of multi-task learning and self-attention, training the model to attend to parents in a syntactic parse tree, achieves state-of-the-art CoNLL-2005 and CoNLL-2012 SRL results for models using predicted predicates.