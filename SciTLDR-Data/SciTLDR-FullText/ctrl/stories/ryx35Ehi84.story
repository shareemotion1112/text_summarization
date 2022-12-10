Analysis methods which enable us to better understand the   representations and functioning of neural models of language are   increasingly needed as deep learning becomes the dominant approach   in NLP.

Here we present two methods based on Representational   Similarity Analysis (RSA) and Tree Kernels (TK) which allow us to   directly quantify how strongly the information encoded in neural   activation patterns corresponds to information represented by   symbolic structures such as syntax trees.

We first validate our   methods on the case of a simple synthetic language for arithmetic   expressions with clearly defined syntax and semantics, and show that   they exhibit the expected pattern of results.

We then apply our methods to   correlate neural representations of English sentences with their   constituency parse trees.

Analysis methods which allow us to better understand the representations and functioning of neural models of language are increasingly needed as deep learning becomes the dominant approach to natural language processing.

A popular technique for analyzing neural representations involves predicting information of interest from the activation patterns, typically using a simple predictive model such as a linear classifier or regressor.

If the model is able to predict this information with high accuracy, the inference is that the neural representation encodes it.

We refer to these as diagnostic models.

One important limitation of this method of analysis is that it is only easily applicable to relatively simple types of target information, which are amenable to be predicted via linear regression or classification.

Should we wish to decode activation patterns into a structured target such as a syntax tree, we would need to resort to complex structure prediction algorithms, running the risk that the analytic method becomes no simpler than the actual neural model.

Here we introduce an alternative approach based on correlating neural representations of sentences and structured symbolic representations commonly used in linguistics.

Crucially, the correlation is in similarity space rather than in the original representation space, removing most constraints on the types of representations we can use.

Our approach is an extension of the Representational Similarity Analysis (RSA) method, initially introduced by BID19 in the context of understanding neural activation patterns in human brains.

In this work we propose to apply RSA to neural representations of strings from a language on one side, and to structured symbolic representations of these strings on the other side.

To capture the similarities between these symbolic representations, we use a tree kernel, a metric to compute the proportion of common substructures between trees.

This approach enables straightforward comparison of neural and symbolic-linguistic representations.

Furthermore, we introduce RSA REGRESS , a similarity-based analytic method which combines features of RSA and of diagnostic models.

We validate both techniques on neural models which process a synthetic language for arithmetic expressions with a simple syntax and semantics and show that they behave as expected in this controlled setting.

We further apply our techniques to two neural models trained on English text, Infersent BID9 and BERT BID13 , and show that both models encode a substantial amount of syntactic information compared to random models and simple bag-of-words representations; we also show that according to our metrics syntax is most salient in the intermediate layers of BERT.

The dominance of deep learning models in NLP has brought an increasing interest in techniques to analyze these models and gain insight into how they encode linguistic information.

For an overview of analysis techniques, see BID3 .

The most widespread family of techniques are diagnostic models, which use the internal activations of neural networks trained on a particular task as input to another predictive model.

The success of such a predictive model is then interpreted as evidence that the predicted information has been encoded by the original neural model.

The approach has also been called auxiliary task BID0 , decoding BID1 , diagnostic classifier BID16 or probing BID10 .Diagnostic models have used a range of predictive tasks, but since their main purpose is to help us better understand the dynamics of a complex model, they themselves need to be kept simple and interpretable.

This means that the predicted information in these techniques is typically limited to simple class labels or values, as opposed to symbolic, structured representations of interest to linguists such as syntactic trees.

In order to work around this limitation BID29 present a method for probing complex structures via a formulation named edge probing, where classifiers are trained to predict various lexical, syntactic and semantic relations between representation of word spans within a sentence.

Another important consideration when analyzing neural encodings is the fact that a randomly initialized network will often show non-random activation patterns.

The reason for this depends on each particular case, but may involve the dynamics of the network itself as well as features of the input data.

For a discussion of this issue in the context of diagnostic models see BID33 .Alternative approaches have been proposed to analyzing neural models of language.

For example, BID25 train a language model and parallel recurrent models for POS, semantic and topic tagging, and measure the correlation between the neural representations of the language model and the taggers.

Others modify the neural architecture itself to make it more interpretable: Croce et al.(2018) adapt layerwise relevance propagation BID2 to Kernel-based Deep Architectures BID11 in order to retrieve examples which motivate model decisions.

A vector representation for a given structured symbolic input is built based on kernel evaluations between the input and a subset of training examples known as landmarks, and the network decision is then traced back to the landmarks which had most influence on it.

In our work we also use kernels between symbolic structures, but rather than building a particular interpretable model we propose a general analytical framework.

BID19 present RSA as a variant of pattern-information analysis, to be applied for understanding neural activation patterns in human brains, for example syntactic computations BID30 or sensory cortical processing BID32 .

The core idea is to find connections between data from neuroimaging, behavioral experiments and computational modeling by correlating representations of stimuli in each of these representation spaces via their pairwise (dis)similarities.

RSA has also been used for measuring similarities between neuralnetwork representation spaces (e.g. BID6 BID7 .

For extending RSA to a structured representation space, we need a metric for measuring (dis)similarity between two structured representations.

Kernels provide a suitable framework for this purpose: BID8 introduce convolutional kernels for syntactic parse trees as a metric which quantifies similarity between trees as the number of overlapping tree fragments between them, and introduce a polynomial time algorithm to compute these kernels; BID21 propose an efficient algorithm for computing tree kernels in linear average running time.

When developing techniques for analyzing neural network models of language, several studies have used synthetic data from artificial languages.

Using synthetic language has the advantage that its structure is well-understood and the complexity of the language and the statistical characteristics of the generated data can be carefully controlled.

The tradition goes back to the first generation of connectionist models of language BID14 BID15 .

More recently, BID26 and BID28 both use contextfree grammars to generate data, and train RNNbased models to identify matching numbers of opening and closing brackets (so called Dyck languages).

The task can be learned, but BID26 report that the models fail to generalize to longer sentences.

BID23 also show that with extensive training and the appropriate curriculum, LSTMs trained on synthetic language can learn compositional interpretation rules.

Nested arithmetic languages are also appealing choices since they have an unambiguous hierarchical structure and a clear compositional semantic interpretation (i.e. the value of the arithmetic expression).

BID16 train RNNs to calculate the value of such expressions and show that they perform and generalize well to unseen strings.

They apply diagnostic classifiers to analyze the strategy employed by the RNN model.

RSA finds connections between data from two different representation spaces.

Specifically, for each representation type we compute a matrix of similarities between pairs of stimuli.

Pairs of these matrices are then subject to second-order analysis by extracting their upper triangulars and computing a correlation coefficient between them.

Thus for a set of objects X, given a similarity function s k for a representation k, the function S k which computes the representational similarity matrix is defined as: DISPLAYFORM0 and the RSA score between representations k and l for data X is the correlation (such as Pearson's correlation coefficient r) between the upper triangulars S k (X) and S l (X), excluding the diagonals.

Structured RSA We apply RSA to neural representations of strings from a language on one side, and to structured symbolic representations of these strings on the other side.

The structural properties are captured by defining appropriate similarity functions for these symbolic representations; we use tree kernels for this purpose.

A tree kernel measures the similarity between a pair of tree structures by computing the number of tree fragments they share.

BID8 introduce an algorithm for efficiently computing this quantity; a tree fragment in their formulation is a set of connected nodes subject to the constraint that only complete production rules are included.

Following BID8 , we calculate the tree kernel between two trees T 1 and T 2 as: DISPLAYFORM1 where n 1 and n 2 are the complete sets of tree fragments in T 1 and T 2 , respectively, and the function C(n 1 , n 2 , λ) is calculated as shown in figure 2.

The parameter λ is used to scale the relative importance of tree fragments with their size.

Lower values of this parameter discount larger tree fragments in the computation of the kernel; the value 1 does not do any discounting.

See FIG0 for the illustration of the effect of the value of λ on the kernel.

We work with normalized kernels: given a function K which computes the raw count of tree fragments in common between trees t 1 and t 2 , the normalized tree kernel is defined as: FIG1 shows the complete set of tree fragments which the tree kernel implicitly computes for an example syntax tree.

DISPLAYFORM2 DISPLAYFORM3 Figure 2: Dynamic programming formula for computing a convolution kernel, after BID8 .

Here nc(n) is the number of children of a given (sub)tree, and ch(n, i) is its i th child; prod(n) is the production of node n, and preterm(n) is true if n is a preterminal node.

RSA REGRESS Basic RSA measures correlation between similarities in two different representations globally, i.e. how close they are in their totality.

In contrast, diagnostic models answer a more specific question: to what extent a particular type of information can be extracted from a given representation.

For example, while for a particular neural encoding of sentences it may be possible to predict the length of the sentence with high accuracy, the RSA between this representation and the strings represented only by their length may be relatively small in magnitude, since the neural representation may be encoding many other aspects of the input in addition to its length.

We introduce RSA REGRESS , a method which shares features of both classic RSA as well as the diagnostic model approach.

Like RSA it is based on two similarity functions s k and s l specific to two different representations k and l. But rather than computing the square matrices S k (X) and S l (X) for a set of objects X, we sample a reference set of objects R to act as anchor points, and then embed the objects of interest X in the representation space k via the representational similarity function σ k defined as: 1 DISPLAYFORM4 Likewise for representation l, we calculate σ l for the same set of objects X. The rows of the two resulting matrices contain two different views of the objects of interest, where the dimensions of each view indicate the degree of similarity for a particular reference anchor point.

We can now fit a multivariate linear regression model to map between the two views: DISPLAYFORM5 where k is the source and l is the target view, and MSE is the mean squared error.

The success of this model can be seen as an indication of how predictable representation l is from representation k. Specifically, we use a cross-validated Pearson's correlation between predicted and true targets for an L 2 -penalized model.

Evaluation of analysis methods for neural network models is an open problem.

One frequently resorts to largely qualitative evaluation: checking whether the conclusions reached via a particular approach have face validity and match pre-existing intuitions.

However pre-existing intuitions are often not reliable when it comes to complex neural models applied to also very complex natural language data.

It is helpful to simplify one part of the overall system and apply the analytic technique of interest on a neural model which processes a simple and well-understood synthetic language.

As our first case study, we use a simple language of arithmetic expressions.

Here we first describe the language and its syntax and semantics, and then introduce neural recurrent models which process these expressions.

DISPLAYFORM0

Our language consists of expressions which encode addition and subtraction modulo 10.

Consider the example expression ((6+2)-(3+7)).In order to evaluate the whole expression, each parenthesized sub-expression is evaluated modulo 10: in this case the left sub-expression evaluates to 8, the right one to 0 and the whole expression to 8.

TAB0 gives the context-free grammar which generates this language, and the rules for semantic evaluation.

Figure 4 shows the syntax tree for the example expression according to this grammar.

This language lacks ambiguity, has a small vocabulary (14 symbols) and simple semantics, while at the same time requiring the processing of hierarchical structure to evaluate its expressions.

2Generating expressions In order to generate expressions in L we use the recursive function GENERATE defined in Algorithm 1.

The function receives two input parameters: the branching probability p and the decay factor d. In the recursive call to GENERATE in lines 4 and 5 the probability p is divided by the decay factor.

Larger values of d lead to the generation of smaller expressions.

Within the branching path in line 6 the operator is selected uniformly at random, and likewise in the non-branching path in line 9 the digit is sampled uniformly.2 The grammar is more complex than strictly needed in order to facilitate the computation of the Tree Kernel, which assumes each vocabulary symbol is expanded from a pre- DISPLAYFORM0 Syntax tree of the expression ((6+2)-(3+7)).Algorithm 1 Recursive function for generating an expression of language L. DISPLAYFORM1 if branch then 4: DISPLAYFORM2 end if 12: end function

We define three recurrent models which process the arithmetic expressions from language L. Each of them is trained to predict a different target, related either to the syntax of the language or to its semantics.

We use these models as a testbed for validating our analytical approaches.

All these models share the same recurrent encoder architecture, based on LSTM BID15 .Encoder The encoder consists of a trainable embedding lookup table for the input symbols, and a single-layer LSTM.

The state of the hidden layer of the LSTM at the last step in the sequence is used as a representation of the input expression.

SEMANTIC EVALUATION This model consists of the encoder as described above, which passes its representation of the input to a multi-layer perceptron component with a single output neuron.

It is trained to predict the value of the input expression, with mean squared error as the loss function.

In order to perform this task we would expect that the model needs to encode the hierarchical structerminal node.

ture of the expression to some extent while also encoding the result of actually carrying out the operations of semantic evaluation.

TREE DEPTH This model is similar to SEMAN-TIC EVALUATION but is trained to predict the depth of the syntax tree corresponding to the expression instead of its value.

We expect this model to need to encode a fair amount of hierarchical information, but it can completely ignore the semantics of the language, including the identity of the digit symbols.

INFIX-TO-PREFIX This model uses the encoder to create a representation of the input expression, which it then decodes in its prefix form.

For example, the expression ((6+2)-(3+7)) is converted to (-(+62)(+37)).

The decoder is an LSTM trained as a conditional language model, i.e. its initial hidden state is the output of the encoder and its input at each step is the embedding of previous output symbol.

The loss function is categorical cross-entropy.

We would expect this model to encode the hierarchical structure in some form as well as the identity of the digit symbols, but it can ignore the compositional semantics of the language.

We use RSA to correlate the neural encoders from Section 4.2 with reference syntactic and semantic information about the arithmetic expressions.

For the neural representations we use cosine distance as the dissimilarity metric.

The reference representations and their associated dissimilarity metrics are described below.

Semantic value This is simply the value to which each expression evaluates, also used as the target of the SEMANTIC EVALUATION model.

As a measure of dissimilarity we use the absolute difference between values, which ranges from 0 to 9.Tree depth This is the depth of the syntax tree for each expression, also used as the target of the TREE DEPTH model.

We use the absolute difference as the dissimilarity measure.

The dissimilarity is minimum 0 and has no upper bound, but in our data the typical maximum value is around 7.Tree kernel This is an estimate of similarity between two syntax trees based on the number of tree fragments they share, as described in Section 3.

The normalized tree kernel metric ranges between 0 and 1, which we convert to dissimilarity by subtracting it from 1.The semantic value and tree depth correlates are easy to investigate with a variety of analytic methods including diagnostic models; we include them in our experiments as a point of comparison.

We use the tree kernel representation to evaluate structured RSA for a simple synthetic language.

We implement the neural models in PyTorch 1.0.0.

We use the following model architecture: encoder embedding layer size 64, encoder LSTM size 128, for the regression models, MLP with 1 hidden layer of size 256; for the sequence-to-sequence model the decoder hyper-parameters are the same as the encoder.

The symbols are predicted via a linear projection layer from hidden state, followed by a softmax.

Training proceeds following a curriculum: we first train on 100,000 batches of size 32 of random expressions sampled with decay d = 2.0, followed by 200,000 batches with d = 1.8 and finally 400,000 batches with d = 1.5.

We optimize with Adam with learning rate 0.001.

We report results on expressions sampled with d = 1.5.

See FIG2 for the distribution of expression sizes for these values of d. We report all results for two conditions: randomly initialized, and trained, in order to quantify the effect of learning on the activation patterns.

The trained model is chosen by saving model weights during training every 10,000 batches and selecting the weights with the smallest loss on 1,000 held-out validation expressions.

Results are reported on separate test data consisting of 2,000 expressions and 200 reference expressions for RSA REGRESS embedding.

TAB2 shows the results of our experiments, where each row shows a different encoder type and each column a different target task.

Semantic value and tree depth As a first sanity check, we would like to see whether the RSA techniques show the same pattern captured by the diagnostic models.

As expected, both diagnostic and RSA scores are the highest when the objective function used to train the encoder and the analytical reference representations match: for example, the SEMANTIC EVALUATION encoder scores high on the semantic value reference, both for the diagnostic model and the RSA.

Furthermore, the scores for the value and depth reference representation according to the diagnostic model and according to RSA REGRESS are in agreement.

The scores according to RSA in some cases show a different picture.

This is expected, as RSA answers a substantially different question than the other two approaches: it looks at how the whole representations match in their similarity structure, whereas both the diagnostic model and RSA REGRESS focus on the part of the representation that encodes the target information the strongest.

Tree Kernel We can use both RSA and RSA REGRESS for exploring whether the hidden activations encode any structural representation of syntax: this is evident in the scores yielded by the TK reference representations.

As expected, the highest scores for both methods are gained when using INFIX-TO-PREFIX encodings, the task that relies the most on the hierarchical structure of an input string.

RSA REGRESS yields the secondhighest score for TREE DEPTH encodings, which also depend on aspects of tree structure.

The overall pattern for the TK with different values of the discounting parameter λ is similar, even though the absolute values of the scores vary.

What is unexpected is the results for the random encoder, which we turn to next.

The non-random nature of the activation patterns of randomly initialized models (e.g., BID33 is also strongly in evidence in our results.

For example the random encoder has quite a high score for diagnostic regression on tree depth.

Even more striking is the fact that the random encoder has substantial negative RSA score for the Tree Kernel: thus, expression pairs more similar according to the Tree Kernel are less similar according to the random encoder, and vice-versa.

When applying RSA we can inspect the full correlation pattern via a scatter-plot of the dissimilarities in the reference and encoder representations.

FIG3 shows the data for the random encoder and the Tree Kernel representations.

As can be seen, the negative correlation for the random encoder is due to the fact that according to the Tree Kernel, expression pairs tend to have high dissimilarities, while according to the random encoder's activations they tend to have overall low dissimilarities.

For the trained INFIX-TO-PREFIX encoder the dissimilarities are clearly positively correlated with the TK dissimilarities.

Thus the raw correlation value for the trained encoder is a biased estimate of the effect of learning, as learning has to overcome the initially substantial negative correlation: a better estimate is the difference between scores for the learned and random model.

It is worth noting that the same approach would be less informative for the diagnostic model approach or for RSA REGRESS .

For a regression model the correlation scores will be positive, and when taking the difference between learned and random scores, they may cancel out, even though a particular information may be predictable from the random activations in a completely different way than from the learned activations.

This is what we see for the RSA REGRESS scores for random vs. INFIX-TO-PREFIX the scores partially cancel out, and given the pattern in FIG3 it is clear that subtracting them is misleading.

It is thus a good idea to complement the RSA REGRESS score with the plain RSA correlation score in order to obtain a full picture of how learning affects the neural representations.

Overall, these results show that RSA REGRESS can be used to answer the same sort of questions as the diagnostic model.

It has the added advantage of being also easily applicable to structured symbolic representations, while the RSA scores and the full RSA correlation pattern provides a complementary source of insight into neural representations.

Encouraged by these findings, we next apply both RSA and RSA REGRESS to representations of natural language sentences.

Here we use our proposed RSA-based techniques to compare tree-structure representations of natural language sentences with their neural representations captured by sentence embeddings.

Such embeddings are often provided by NLP systems trained on unlabeled text, using variants of a language modeling objective (e.g. BID24 , next and previous sentence prediction BID18 BID20 , or discourse based objectives BID22 BID17 .

Alternatively they can be either fully trained or fine-tuned on annotated data using a task such as natural language inference BID9 .

In our experiments we use one of each type of encoders.

Bag of words As a baseline we use a classic bag of words model where a sentence is represented by a vector of word counts.

We do not exclude any words and use raw, unweighted word counts.

Infersent This is the supervised model described in BID9 based on a bidirectional LSTM trained on natural language inference.

We use the infersent2 model with pretrained fastText BID5 word embeddings.

3 We also test a randomly initialized version of this model, including random word embeddings.

BERT This is an unsupervised model based on the Transformer architecture BID31 ) trained on a cloze-task and next-sentence prediction BID13 .

We use the Pytorch version of the large 24-layer model (bert-large-uncased).

4 We also test a randomly initialized version of this model.

Data We use a sample of data from the English Web Treebank (EWT) BID4 which contains a mix of English weblogs, newsgroups, email, reviews and question-answers manually annotated for syntactic constituency structure.

We use the 2,002 sentences corresponding to the development section of the EWT Universal Dependencies BID27 , plus 200 sentences from the training section as reference sentences when fitting RSA REGRESS .Tree Kernel Prior to computing the Tree Kernel scores we delexicalize the constituency trees by replacing all terminals (i.e. words) with a single placeholder value X. This ensures that only syntactic structure, and not lexical overlap, contributes to kernel scores.

We compute kernels for the values of λ ∈ {1, 1 2 }.

Embeddings For the BERT embeddings we use the vector associated with the first token (CLS) for a given layer.

For Infersent, we use the default max-pooled representation.

Fitting When fitting RSA REGRESS we use L2-penalized multivariate linear regression.

We report the results for the value of the penalty = 10 n , for n ∈ {−3, −2, −1, 0, 1, 2}, with the highest 10-fold cross-validated Pearson's r between target and predicted similarity-embedded vectors.

TAB4 shows the results of applying RSA and RSA REGRESS on five different sentence encoders, using the Tree Kernel reference.

Results are reported using two different values for the Tree Kernel parameter λ.

As can be seen, with λ = 1 2 , all the encoders show a substantial RSA correlation with the parse trees.

The highest scores are achieved by the trained Infersent and BERT, but even Bag of Words and untrained versions of Infersent and BERT show a sizeable correlation with syntactic trees according to both RSA and RSA REGRESS .When structure matching is strict (λ = 1), only trained BERT and Infersent capture syntactic information according to RSA; however, RSA REGRESS still shows moderate correlation for BoW and the untrained versions of BERT and Infersent.

Thus RSA REGRESS is less sensitive to the value of λ than RSA since changing it from 1 2 to 1 does not alter results in a qualitative sense.

FIG4 shows how RSA and RSA REGRESS scores change when correlating Tree Kernel estimates with embeddings from different layers of BERT.

For trained models, scores peak between layers 15-22 (depending on metric and λ) and decline thereafter, which indicates that the final layers are increasingly dedicated to encoding aspects of sentences other than pure syntax.

We present two RSA-based methods for correlating neural and syntactic representations of language, using tree kernels as a measure of similarity between syntactic trees.

Our results on arithmetic expressions confirm that both versions of structured RSA capture correlations between different representation spaces, while providing complementary insights.

We apply the same techniques to English sentence embeddings, and show where and to what extent each representation encodes syntactic information.

The proposed methods are general and applicable not just to constituency trees, but given a similarity metric, to any symbolic representation of linguistic structures including dependency trees or Abstract Meaning Representations.

We plan to explore these options in future work.

A toolkit with the implementation of our methods is available at https://github.com/gchrupala/ursa.

<|TLDR|>

@highlight

Two methods based on Representational Similarity Analysis (RSA) and Tree Kernels (TK) which directly quantify how strongly information encoded in neural activation patterns corresponds to information represented by symbolic structures.