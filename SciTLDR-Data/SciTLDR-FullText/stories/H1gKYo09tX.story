The ability to generate natural language sequences from source code snippets has a variety of applications such as code summarization, documentation, and retrieval.

Sequence-to-sequence (seq2seq) models, adopted from neural machine translation (NMT), have achieved state-of-the-art performance on these tasks by treating source code as a sequence of tokens.

We present code2seq: an alternative approach that leverages the syntactic structure of programming languages to better encode source code.

Our model represents a code snippet as the set of compositional paths in its abstract syntax tree (AST) and uses attention to select the relevant paths while decoding.

We demonstrate the effectiveness of our approach for two tasks, two programming languages, and four datasets of up to 16M examples.

Our model significantly outperforms previous models that were specifically designed for programming languages, as well as general state-of-the-art NMT models.

An interactive online demo of our model is available at http://code2seq.org.

Our code, data and trained models are available at http://github.com/tech-srl/code2seq.

Modeling the relation between source code and natural language can be used for automatic code summarization BID2 , documentation BID19 , retrieval BID1 , and even generation BID7 BID28 BID40 BID14 BID24 .

In this work, we consider the general problem of generating a natural language sequence from a given snippet of source code.

A direct approach is to frame the problem as a machine translation problem, where the source sentence is the sequence of tokens in the code and the target sentence is a corresponding natural language sequence.

This approach allows one to apply state-of-the-art neural machine translation (NMT) models from the sequence-to-sequence (seq2seq) paradigm BID23 BID39 , yielding state-ofthe-art performance on various code captioning and documentation benchmarks BID19 BID2 BID22 ) despite having extremely long source sequences.

We present an alternative approach for encoding source code that leverages the syntactic structure of programming languages: CODE2SEQ.

We represent a given code snippet as a set of compositional paths over its abstract syntax tree (AST), where each path is compressed to a fixed-length vector using LSTMs BID17 .

During decoding, CODE2SEQ attends over a different weighted average of the path-vectors to produce each output token, much like NMT models attend over token representations in the source sentence.

We show the effectiveness of our code2seq model on two tasks: (1) code summarization (Figure 1a) , where we predict a Java method's name given its body, and (2) code captioning (Figure 1b) , where we predict a natural language sentence that describes a given C# snippet.

Code summarization in Java:Code captioning in C#: DISPLAYFORM0 Figure 1: Example of (a) code summarization of a Java code snippet, and (b) code captioning of a C# code snippet, along with the predictions produced by our models.

The highlighted paths in each example are the top-attended paths in each decoding step.

Because of space limitations we included only the top-attended path for each decoding step, but hundreds of paths are attended at each step.

Additional examples are presented in Appendix B and Appendix C.On both tasks, our CODE2SEQ model outperforms models that were explicitly designed for code, such as the model of BID2 and CodeNN BID19 , as well as TreeLSTMs BID38 and state-of-the-art NMT models BID23 BID39 .

To examine the importance of each component of the model, we conduct a thorough ablation study.

In particular, we show the importance of structural encoding of code, by showing how our model yields a significant improvement over an ablation that uses only token-level information without syntactic paths.

To the best of our knowledge, this is the first work to directly use paths in the abstract syntax tree for end-to-end generation of sequences.

An Abstract Syntax Tree (AST) uniquely represents a source code snippet in a given language and grammar.

The leaves of the tree are called terminals, and usually refer to user-defined values which represent identifiers and names from the code.

The non-leaf nodes are called nonterminals and represent a restricted set of structures in the language, e.g., loops, expressions, and variable declarations.

For example, Figure 2c shows a partial AST for the code snippet of Figure 2a .

Names (such as num) and types (such as int) are represented as values of terminals; syntactic structures such as variable declaration (VarDec) and a do-while loop (DoStmt) are represented as nonterminals.

Given the AST of a code snippet, we consider all pairwise paths between terminals, and represent them as sequences of terminal and nonterminal nodes.

We then use these paths with their terminals' values to represent the code snippet itself.

For example, consider the two Java methods of Figure 2 .

Both of these methods count occurrences of a character in a string.

They have exactly the same functionality, although a different implementation, and therefore different surface forms.

If these snippets are encoded as sequences of tokens, the recurring patterns that suggest the common method name might be overlooked.

However, a structural observation reveals syntactic paths that are common to both methods, and differ only in a single node of a Do-while statement versus a For statement.

This example shows the effectiveness of a syntactic encoding of code.

Such an encoder can generalize much better to unseen examples because the AST normalizes a lot of the surface form variance.

Since our encoding is compositional, the encoder can generalize even if the paths are not identical (e.g., a For node in one path and a While in the other).Since a code snippet can contain an arbitrary number of such paths, we sample k paths as the representation of the code snippet.

To avoid bias, k new paths are sampled afresh in every training iteration.

In Section 5 we show that this runtime-sampling provides regularization and improves results compared to sampling the same k paths for each example in advance.

Formally, we use C to denote a given snippet of code.

Every training iteration, k pairs of terminals are uniformly sampled from within the AST of C. DISPLAYFORM0 , where l j is the length of the jth path.

Our model follows the standard encoder-decoder architecture for NMT (Section 3.1), with the significant difference that the encoder does not read the input as a flat sequence of tokens.

Instead, the encoder creates a vector representation for each AST path separately (Section 3.2).

The decoder then attends over the encoded AST paths (rather than the encoded tokens) while generating the target sequence.

Our model is illustrated in FIG0 .

Contemporary NMT models are largely based on an encoder-decoder architecture BID23 , where the encoder maps an input sequence of tokens x = (x 1 , ..., x n ) to a sequence of continuous representations z = (z 1 , ..., z n ).

Given z, the decoder then generates a sequence of output tokens y = (y 1 , ..., y m ) one token at a time, hence modeling the conditional probability: p (y 1 , ..., y m |x 1 , ..., x n ).At each decoding step, the probability of the next target token depends on the previously generated token, and can therefore be factorized as: DISPLAYFORM0 In attention-based models, at each time step t in the decoding phase, a context vector c t is computed by attending over the elements in z using the decoding state h t , typically computed by an LSTM.

DISPLAYFORM1 The context vector c t and the decoding state h t are then combined to predict the current target token y t .

Previous work differs in the way the context vector is computed and in the way it is combined with the current decoding state.

A standard approach BID23 is to pass c t and h t through a multi-layer perceptron (MLP) and then predict the probability of the next token using softmax: DISPLAYFORM2

Given a set of AST paths {x 1 , ..., x k }, our goal is to create a vector representation z i for each path DISPLAYFORM0 We represent each path separately using a bi-directional LSTM to encode the path, and sub-token embeddings to capture the compositional nature of the terminals' values (the tokens).Path Representation Each AST path is composed of nodes and their child indices from a limited vocabulary of up to 364 symbols.

We represent each node using a learned embedding matrix E nodes and then encode the entire sequence using the final states of a bi-directional LSTM: DISPLAYFORM1 Token Representation The first and last node of an AST path are terminals whose values are tokens in the code.

Following BID0 , we split code tokens into subtokens; for example, a token with the value ArrayList will be decomposed into Array and List.

This is somewhat analogous to byte-pair encoding in NMT BID35 , although in the case of programming languages, coding conventions such as camel notation provide us with an explicit partition of each token.

We use a learned embedding matrix E subtokens to represent each subtoken, and then sum the subtoken vectors to represent the full token: DISPLAYFORM2 The LSTM decoder may also predict subtokens at each step (e.g. when generating method names), although the decoder's subtoken embedding matrix will be different.

Combined Representation To represent the path x = v 1 ...v l , we concatenate the path's representation with the token representations of each terminal node, and apply a fully-connected layer: DISPLAYFORM3 where value is the mapping of a terminal node to its associated value, and W in is a (2d path + 2d token ) ?? d hidden matrix.

Decoder Start State To provide the decoder with an initial state, we average the combined representations of all the k paths in the given example: DISPLAYFORM4 Unlike typical encoder-decoder models, the order of the input random paths is not taken into account.

Each path is encoded separately and the combined representations are aggregated with mean pooling to initialize the decoder's state.

This represents the given source code as a set of random paths.

Attention Finally, the decoder generates the output sequence while attending over all of the combined representations z 1 , ...z k , similarly to the way that seq2seq models attend over the source symbols.

The attention mechanism is used to dynamically select the distribution over these k combined representations while decoding, just as an NMT model would attend over the encoded source tokens.

We evaluate our model on two code-to-sequence tasks: summarization (Section 4.1), in which we predict Java methods' names from their bodies, and captioning (Section 4.2), where we generate natural language descriptions of C# code snippets.

Although out of the focus of this work, in Section 4.3 we show that our model also generates Javadocs more accurately than an existing work.

We thus demonstrate that our approach can produce both method names and natural language outputs, and can encode a code snippet in any language for which an AST can be constructed (i.e., a parser exists).Setup The values of all of the parameters are initialized using the initialization heuristic of BID16 .

We optimize the cross-entropy loss BID32 BID33 Choice of k We experimented with different values of k, the number of sampled paths from each example (which we set to 200 in the final models).

Lower values than k = 100 showed worse results, and increasing to k > 300 did not result in consistent improvement.

In practice, k = 200 was found to be a reasonable sweet spot between capturing enough information while keeping training feasible in the GPU's memory.

Additionally, since the average number of paths in our Java-large training set is 220 paths per example, a number as high as 200 is beneficial for some large methods.

In this task, we predict a Java method's name given its body.

As was previously observed BID2 BID5 , this is a good benchmark because a method name in open-source Java projects tends to be succinct and precise, and a method body is often a complete logical unit.

We predict the target method name as a sequence of sub-tokens, e.g., setMaxConnectionsPerServer is predicted as the sequence "set max connections per server".

The target sequence length is about 3 on average.

We adopt the measure used by BID2 and BID5 , who measured precision, recall, and F1 score over the target sequence, case insensitive.

Data We experiment with this task across three datsets.

In these datasets, we always train across multiple projects and predict on distinct projects:Java-small -Contains 11 relatively large Java projects, originally used for 11 distinct models for training and predicting within the scope of the same project BID2 .

We use the same data, but train and predict across projects: we took 9 projects for training, 1 project for validation and 1 project as our test set.

This dataset contains about 700K examples.

Java-med -A new dataset of the 1000 top-starred Java projects from GitHub.

We randomly select 800 projects for training, 100 for validation and 100 for testing.

This dataset contains about 4M examples and we make it publicly available.

Java-large -A new dataset of the 9500 top-starred Java projects from GitHub that were created since January 2007.

We randomly select 9000 projects for training, 250 for validation and 300 for testing.

This dataset contains about 16M examples and we make it publicly available.

More statistics of our datasets can be found in Appendix A.Baselines We re-trained all of the baselines on all of the datasets of this task using the original implementations of the authors.

We compare CODE2SEQ to the following baselines: BID2 , who used a convolutional attention network to predict method names; syntactic paths with Conditional Random Fields (CRFs) BID4 ; code2vec BID5 ; and a TreeL-STM BID38 encoder with an LSTM decoder and attention on the input sub-trees.

Additionally, we compared to three NMT baselines that read the input source code as a stream of tokens: 2-layer bidirectional encoder-decoder LSTMs (split tokens and full tokens) with global attention BID23 , and the Transformer BID39 , which achieved state-of-the-art results for translation tasks.

We put significant effort into strengthening the NMT baselines in order to provide a fair comparison:(1) we split tokens to subtokens, as in our model (e.g., HashSet ??? Hash Set) -this was shown to improve the results by about 10 F1 points TAB1 ; (2) we deliberately kept the original casing of the source tokens since we found it to improve their results; and (3) during inference, we replaced generated UNK tokens with the source tokens that were given the highest attention.

For the 2-layer BiLSTM we used embeddings of size 512, an encoder and a decoder of 512 units each, and the default hyperparameters of OpenNMT .

For the Transformer, we used their original hyperparameters BID39 .

This resulted in a Transformer model with 169M parameters and a BiLSTM model with 134M parameters, while our code2seq model had only 37M.

Performance TAB1 shows the results for the code summarization task.

Our model significantly outperforms the baselines in both precision and recall across all three datasets, demonstrating that there is added value in leveraging ASTs to encode source code.

Our model improves over the best baselines, BiLSTM with split tokens, by between 4 to 8 F1 points on all benchmarks.

BiLSTM with split tokens consistently scored about 10 F1 points more than BiLSTM with full tokens, and for this reason we included only the split token Transformer and TreeLSTM baselines.

Our model outperforms ConvAttention BID2 , which was designed specifically for this task; Paths+CRFs BID4 , which used syntactic features; and TreeLSTMs.

Although TreeLSTMs also leverage syntax, we hypothesize that our syntactic paths capture long distance relationships while TreeLSTMs capture mostly local properties.

An additional comparison to code2vec on the code2vec dataset can be found in Appendix A. Examples for predictions made by our model and each of the baselines can be found in Appendix C and at http://code2seq.org.

BID15 encoded code using Graph Neural Networks (GNN), and reported lower performance than our model on Java-large without specifying the exact F1 score.

They report slightly higher results than us on Java-small only by extending their GNN encoder with a subtoken-LSTM (BILSTM+GNN??? LSTM); by extending the Transformer with GNN (SELFATT+GNN???SELFATT); or by extending their LSTM decoder with a pointer network (GNN???LSTM+POINTER).

All these extensions can be incorporated into our model as well.

Data Efficiency ConvAttention BID2 performed even better than the Transformer on the Java-small dataset, but could not scale and leverage the larger datasets.

Paths+CRFs showed very poor results on the Java-small dataset, which is expected due to the sparse nature of their paths and the CRF model.

When compared to the best among the baselines (BiLSTM with split tokens), our model achieves a relative improvement of 7.3% on Java-large, but as the dataset becomes smaller, the larger the relative difference becomes: 13% on Java-med and 22% on Java-small; when compared to the Transformer, the relative improvement is 23% on Java-large and 37% on Java-small.

These results show the data efficiency of our architecture: while the data-hungry NMT baselines require large datasets, our model can leverage both small and large datasets.

Sensitivity to input length We examined how the performance of each model changes as the size of the test method grows.

As shown in Figure 4 , our model is superior to all examined baselines across all code lengths.

All models give their best results for short snippets of code, i.e., less than 3 lines.

As the size of the input code increases, all examined models show a natural descent, and show stable results for lengths of 9 and above.

2-layer BiLSTMs TreeLSTM BID38 Transformer BID39 code2vec BID5 Figure 4: F1 score compared to the length of the input code.

This experiment was performed for the code summarization task on the Java-med test set.

All examples having more than 30 lines were counted as having 30 lines.

For this task we consider predicting a full natural language sentence given a short C# code snippet.

We used the dataset of CodeNN BID19 , which consists of 66,015 pairs of questions and answers from StackOverflow.

They used a semi-supervised classifier to filter irrelevant examples and asked human annotators to provide two additional titles for the examples in the test set, making a total of three reference titles for each code snippet.

The target sequence length in this task is about 10 on average.

This dataset is especially challenging as it is orders of magnitude smaller than the code summarization datasets.

Additionally, StackOverflow code snippets are typically short, incomplete at times, and aim to provide an answer to a very specific question.

We evaluated using BLEU score with smoothing, using the same evaluation scripts as BID19 .Baselines We present results compared to CodeNN, TreeLSTMs with attention, 2-layer bidirectional LSTMs with attention, and the Transformer.

As before, we provide a fair comparison by splitting tokens to subtokens, and replacing UNK during inference.

We also include numbers from baselines used by BID19 .Results TAB2 summarizes the results for the code captioning task.

Our model achieves a BLEU score of 23.04, which improves by 2.51 points (12.2% relative) over CodeNN, whose authors introduced this dataset, and over all the other baselines, including BiLSTMs, TreeLSTMs and the Transformer, which achieved slightly lower results than CodeNN.

Examples for predictions made by our model and each of the baselines can be found in Appendix F. These results show that when the training examples are short and contain incomplete code snippets, our model generalizes better to unseen examples than a shallow textual token-level approach, thanks to its syntactic representation of the data.

Although TreeLSTMs also represent the data syntactically, the TreeLSTM baseline achieved lower scores.

Although the task of generating code documentation is outside the focus of this work, we performed an additional comparison to BID18 .

They trained a standard seq2seq model by using the linearized AST as the source sequence and a Javadoc natural language sentence as the target sequence.

While they originally report a BLEU score of 38.17, we computed their BLEU score using prediction logs provided us by the authors and obtained a BLEU score of 8.97, which we find more realistic.

Training our model on the same dataset as Hu et al., matching LSTM sizes, and using the same script on our predictions yields a BLEU score of 14.53, which is a 62% relative gain over the model of BID18 .

This shows that our structural approach represents code better than linearizing the AST and learning it as a sequence.

To better understand the importance of the different components of our model, we conducted an extensive ablation study.

We varied our model in different ways and measured the change in performance.

These experiments were performed for the code summarization task, on the validation set of the Java-med dataset.

We examined several alternative designs:1.

No AST nodes -instead of encoding an AST path using an LSTM, take only the first and last terminal values to construct an input vector 2.

No decoder -no sequential decoding; instead, predict the target sequence as a single symbol using a single softmax layer.

3.

No token splitting -no subtoken encoding; instead, embed the full token.

4.

No tokens -use only the AST nodes without using the values associated with the terminals.

No attention -decode the target sequence given the initial decoder state, without attention.

6.

No random -no re-sampling of k paths in each iteration; instead, sample in advance and use the same k paths for each example throughout the training process.

TAB3 shows the results of these alternatives.

As seen, not encoding AST nodes resulted in a degradation especially in the precision: a decrease of 5.16 compared to 4.30 for the recall.

It is quite surprising that this ablation was still better than the baselines TAB1 : for example, the Transformer can implicitly capture pairs of tokens using its self-attention mechanism.

However, not all tokens are AST leaves.

By focusing on AST leaves, we increase the focus on named tokens, and effectively ignore functional tokens like brackets, parentheses, semicolons, etc.

Transformers can (in theory) capture the same signal, but perhaps they require significantly more layers or a different optimization to actually learn to focus on those particular elements.

The AST gives us this information for free without having to spend more transformer layers just to learn it.

Additionally, for practical reasons we limited the length of the paths to 9 .

This leads to pairs of leaves that are close in the AST, but not necessarily close in the sequence.

In contrast, the Transformer's attention is effectively skewed towards sequential proximity because of the positional embeddings.

Using a single prediction with no decoder reduces recall by more than one-third.

This shows that the method name prediction task should be addressed as a sequential prediction, despite the methods' relatively short names.

Using no token splitting or no tokens at all drastically reduces the score, showing the significance of encoding both subtokens and syntactic paths.

Despite the poor results of no tokens, it is still surprising that the model can achieve around half the score of the full model, as using no tokens is equivalent to reasoning about code which has no identifier names, types, APIs, and constant values, which can be very difficult even for a human.

The no attention experiment shows the contribution of attention in our model, which is very close in its relative value to the contribution of attention in seq2seq models BID23 .

The no random experiment shows the positive contribution of sampling k different paths afresh on every training iteration, instead of using the same sample of paths from each example during the entire training.

This approach provides data-level regularization that further improves an already powerful model.

Another visualization can be found in Appendix D.

The growing availability of open source repositories creates new opportunities for using machine learning to process source code en masse.

Several papers model code as a sequence of tokens BID19 BID2 BID22 , characters BID10 , and API calls BID29 .

While sometimes obtaining satisfying results, these models treat code as a sequence rather than a tree.

This necessitates implicit relearning of the (predefined) syntax of the programming language, wasting resources and reducing accuracy.

Code representation models that use syntactic information have usually been evaluated on relatively easier tasks, which mainly focus on "filling the blanks" in a given program BID4 BID3 or semantic classification of code snippets BID5 .

Moreover, none of the models that use syntactic relations are compositional, and therefore the number of possible syntactic relations is fixed either before or after training, a process which results in a large RAM and GPU memory consumption.

The syntactic paths of BID4 are represented monolithically, and are therefore limited to only a subset of the paths that were observed enough times during training.

As a result, they cannot represent unseen relations.

In contrast, by representing AST paths node-by-node using LSTMs, our model can represent and use any syntactic path in any unseen example.

Further, our model decodes the output sequence step-by-step while attending over the input paths, and can thus generate unseen sequences, compared to code2vec BID5 , which has a closed vocabulary.

BID26 were the first to generate sequences by leveraging the syntax of code.

They performed a line-by-line statistical machine translation (SMT) to translate Python code to pseudocode.

Our tasks are different, and we cannot assume an alignment between elements in the input and the output; our tasks take a whole code snippet as their input, and produce a much shorter sequence as output.

Additionally, a conceptual advantage of our model over line-by-line translation is its ability to capture multiline patterns in the source code.

These multiline patterns are often very useful for the model and get the most attention (Figure 1a) .

A recent work BID18 generates comments from code.

There is a conceptual difference between our approaches: BID18 linearize the AST, and then pass it on to a standard seq2seq model.

We present a new model, in which the encoder already assumes that the input is tree-structured.

When training our model on their dataset, we improve over their BLEU score by 62% (Section 4.3).

BID3 represent code with Gated Graph Neural Networks.

Nodes in the graph represent identifiers, and edges represent syntactic and semantic relations in the code such as "ComputedFrom" and "LastWrite".

The edges are designed for the semantics of a specific programming language, for a specific task, and require an expert to devise and implement.

In contrast, our model has minimal assumptions on the input language and is general enough not to require either expert semantic knowledge or the manual design of features.

Our model can therefore be easily implemented for various input languages.

BID8 used graph-convolutional networks for machine translation of natural languages.

BID27 encoded code using Tree-RNNs to propagate feedback on student code; and BID12 used Tree-RNNs for a tree-to-tree translation of code into another programming language.

We presented a novel code-to-sequence model which considers the unique syntactic structure of source code with a sequential modeling of natural language.

The core idea is to sample paths in the Abstract Syntax Tree of a code snippet, encode these paths with an LSTM, and attend to them while generating the target sequence.

We demonstrate our approach by using it to predict method names across three datasets of varying sizes, predict natural language captions given partial and short code snippets, and to generate method documentation, in two programming languages.

Our model performs significantly better than previous programming-language-oriented works and state-of-the-art NMT models applied in our settings.

We believe that the principles presented in this paper can serve as a basis for a wide range of tasks which involve source code and natural language, and can be extended to other kinds of generated outputs.

To this end, we make all our code, datasets, and trained models publicly available.

Comparison to code2vec on their dataset We perform an additional comparison to code2vec BID5 ) on their proposed dataset.

As shown in TAB4 , code2vec achieves a high F1 score on that dataset.

However, our model achieves an even higher F1 score.

The poorer performance of code2vec on our dataset is probably due to its always being split to train/validation/test by project, whereas the dataset of code2vec is split by file.

In the code2vec dataset, a file can be in the training set, while another file from the same project can be in the test set.

This makes their dataset significantly easier, because method names "leak" to other files in the same project, and there are often duplicates in different files of the same project.

This is consistent with BID3 , who found that splitting by file makes the dataset easier than by project.

We decided to take the stricter approach, and not to use their dataset (even though our model achieves better results on it), in order to make all of our comparisons on split-by-project datasets.

TAB5 shows some statistics of our used datasets.

Figure 6 contains examples from our test set for the code captioning task in C#, along with the prediction of our model and each of the baselines.

FIG3 shows a timestep-by-timestep example, with the symbol decoded at each timestep and the top-attended path at that step.

The width of the path is proportional to the attention it was given by the model (because of space limitations we included only the top-attended path for each decoding step, but hundreds of paths are attended at each step).

Figure 8 contains examples from our test set for the code summarization task in C#, along with the prediction of our model and each of the baselines.

The presented predictions are made by models that were trained on the same Java-large dataset.

D CODE CAPTIONING RESULTS Figure 9 shows a bar chart of the BLEU score of our model and the baselines, in the code captioning task (predicting natural language descriptions for C# code snippets).

The numbers are the same as in TAB2 .

Figure 10 shows a bar chart of the F1 score of our model and the baselines, in the code summarization task (predicting method names in Java).

The numbers are the F1 columns from TAB1 F ABLATION STUDY RESULTS Figure 11 shows a bar chart of the relative decrease in precision and recall for each of the ablations described in Section 5 and presented in TAB3 .

Prediction ConvAttention BID2 add Paths+CRFs BID4 call code2vec BID5 log response 2-layer BiLSTM (no token splitting) handle request 2-layer BiLSTM report child request Transformer add child TreeLSTM BID38 add child Gold:add child request code2seq (this work) add child request public static int _

_

_

_

__(int value) { return value <= 0 ? 1 : value >= 0x40000000 ? 0x40000000 : 1 << (32 -Integer.numberOfLeadingZeros(value -1)); }

Prediction ConvAttention BID2 get Paths+CRFs BID4 test bit inolz code2vec BID5 multiply 2-layer BiLSTM (no token splitting) next power of two 2-layer BiLSTM { (replaced UNK) Transformer get bit length TreeLSTM BID38 get Gold:find next positive power of two code2seq (this work) get power of two BID4 i code2vec BID5 to big integer 2-layer BiLSTM (no token splitting) generate prime 2-layer BiLSTM generate prime number Transformer generate TreeLSTM BID38 probable prime Gold:choose random prime code2seq (this work) generate prime number public boolean _

_

_

_

__(Set<String> set, String value) { for (String entry : set) { if (entry.equalsIgnoreCase(value)) { return true; } } return false; }

Prediction ConvAttention BID2 is Paths+CRFs BID4 equals code2vec BID5 contains ignore case 2-layer BiLSTM (no token splitting) contains ignore case 2-layer BiLSTM contains Transformer contains TreeLSTM BID38 contains ignore case Gold:contains ignore case code2seq (this work) contains ignore case

ConvAttention BID2 Paths+CRFs BID4 code2vec BID5 ) 2-layer BiLSTM (no token splitting) 2-layer BiLSTM TreeLSTM BID38 Transformer BID39 code2seq (this work)

@highlight

We leverage the syntactic structure of source code to generate natural language sequences.

@highlight

Presents a method for generating sequences from code by parsing and producing a syntax tree

@highlight

This paper introduces an AST-based encoding for programming code and shows its effectiveness in the tasks of extreme code summarization and code captioning.

@highlight

This paper presents a new code-to-sequence model that leverages the syntactic structure of programming languages to encode source code snippets and then decode them to natural language