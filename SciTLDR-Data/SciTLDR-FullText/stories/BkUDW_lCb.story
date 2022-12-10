The digitization of data has resulted in making datasets available to millions of users in the form of relational databases and spreadsheet tables.

However, a majority of these users come from diverse backgrounds and lack the programming expertise to query and analyze such tables.

We present a system that allows for querying data tables using natural language questions, where the system translates the question into an executable SQL query.

We use a deep sequence to sequence model in wich the decoder uses a simple type system of SQL expressions to structure the output prediction.

Based on the type, the decoder either copies an output token from the input question using an attention-based copying mechanism or generates it from a fixed vocabulary.

We also introduce a value-based loss function that transforms a distribution over locations to copy from into a distribution over the set of input tokens to improve training of our model.

We evaluate our model on the recently released WikiSQL dataset and show that our model trained using only supervised learning significantly outperforms the current state-of-the-art Seq2SQL model that uses reinforcement learning.

The IT revolution of the past few decades has resulted in a large-scale digitization of data, making it accessible to millions of users in the form of databases and spreadsheet tables.

Despite advances in designing new high-level programming languages and user interfaces, querying and analyzing such tables usually still requires users to write small programs in languages such as SQL or Excel, which is unfortunately beyond the programming expertise of a majority of end-users BID8 .

Thus, building effective semantic parsers that can translate natural language questions into executable programs has been a long-standing goal to improve end-user data accessibility BID22 BID30 BID20 BID15 BID9 .Recent work has shown that recurrent neural networks with attention and copying mechanisms BID4 BID18 BID13 can be used effectively to build successful semantic parsers.

Notably, BID32 recently introduced the state-ofthe-art Seq2SQL model for question to SQL translation in the supervised setting, where programs are explicitly provided with their corresponding questions.

The Seq2SQL model shows that using separate decoders for different parts of a query (i.e., aggregation operation, target column, and where predicates) increases prediction accuracy, and reinforcement learning further improves the model by allowing it to learn semantically equivalent queries beyond supervision.

In this paper, we present a new encoder-decoder model as an extension of the attentional seq2seq model for natural language to SQL program translation and a training approach that is capable of learning the model in an effective and stable manner.

FIG0 shows an example table-question pair and how our system generates the answer by executing the synthesized SQL program.

First, we present a simple type system to control the decoding mode at each decoding step (cf.

Sect.

2).

Based on the SQL grammar, a decoder cell is specialized to either select a token from the SQL built-in vocabulary, generate a pointer over the table header and the input question to copy a table column, or generate a pointer to copy a constant from the user's question.

The type system allows us to have a fine-grain control over the decoding process while retaining the simplicity of the sequence structure, as opposed to designing multiple decoders for different language components or adding extra controllers for expansion of production rules .

FIG0 .

The model encodes table columns as well as the user question with a bidirectional LSTM and then decodes the hidden state with a typed LSTM, where the decoding action for each cell is statically determined.

Second, we constructed an objective function that allows us to effectively train our model to copy correct values (cf.

Sect.

3).

Training copying decoders can be challenging when the value to be copied appears in multiple places in the input (i.e. both in the question and the table headers).

Our solution to the problem is to use a new value-based loss function that transfers the distribution over the pointer locations in the input into a distribution over the set of tokens observed in the input, by summing up the probabilities of the same vocabulary value appearing at different input indices.

Our results show that our training strategy performs better than alternatives (e.g., direct supervision on pointers).

Our approach is very robust and consistently converges to high-accuracy models starting from random initializations.

We have evaluated our approach on the recently released WikiSQL dataset BID32 , a corpus consisting of over 80,000 natural language question and pairs.

Our results in Sect.

4 show that our model can significantly outperform the current state-of-the-art Seq2SQL model BID32 , without requiring a reinforcement learning refinement phase (59.5% vs 48.3% for exact syntactic match and 65.1% vs 59.4% for execution accuracy).

Also, with a series of ablation experiments, we analyze the influence of different components of our model on the overall results.

We generate SQL queries from questions using an RNN-based encoder-decoder model with attention and copying mechanisms BID25 BID7 BID32 .

However, we use the known structure of SQL to statically determine the "type" of output of a decoding step while generating the SQL query.

For example, we know from the grammar that the third token (after the aggregation function) of the query is always a column name specifying the aggregated column.

Thus, when decoding, as shown in FIG1 , we statically determine the type of the token to generate based on its decoding time stamp, and then use a specialized decoder to generate the output: if we have to produce a column name or a constant, we enforce the use of a copying mechanism, otherwise we project the hidden state to a built-in vocabulary to obtain a built-in SQL operator.

This means that we only need to maintain a small built-in decoder vocabulary (sized 17) for all operators.

Our encoder is a bidirectional recurrent neural network (RNN) using Long Short-Term Memory (LSTM) cells.

As input tokens, we use the concatenation of the table header (i.e., the column names) of the queried table and the user query, i.e., X = [x DISPLAYFORM0 ].

This concatenation allows the model to learn how to compute a joint representation for both columns and the input query.

We use |X | to represent the input sequence length (equal to C + Q).Token Embedding To handle the large number of different tokens in the input query, we combine a pre-trained character n-gram embedding and a pre-trained global word embedding.

For a token x , we compute its embedding emb e (x ) as the concatenation of its word embedding and the average embeddings of all n-gram features contained in x, in the same way as BID32 .

Formally, if W word is a pre-trained word model, x [i, j] is the character sequence from i to j in x , W n-gram is a pre-trained n-gram for the n-gram feature set V , and N x is the number of n-gram features contained in the word, then DISPLAYFORM1 We use the pre-trained n-gram model by BID10 and the GloVe embedding (Pennington et al., 2014) for words; both are set untrainable to avoid over-fitting.

Bidirectional RNN We feed embedded tokens into a bidirectional RNN composed of LSTM cells C e,fw , C e,bw , computing DISPLAYFORM2 and will use the sequence O e = [o DISPLAYFORM3 e,bw ) as the learned representation of token x (k) for the attention and copying mechanisms of our decoder.

We initialize the forward encoder with hidden states h

Output Grammar Our model uses types abstracted from the grammar of the target language to improve the decoding performance.

Concretely, we know that the subset of SQL necessary to answer WikiSQL Questions can be represented using the following grammar, in which t refers to the name of the table being queried, c refers to a column name in the table, and v refers to any open world string or number that may be used in the query: DISPLAYFORM0 A consequence of this observation is that we can, based on the tokens generated so far, determine the "type" of the next token to generate.

For example, after generating the two tokens "Select Id", we know that the following token must be one of the column names from the queried table.

We found it sufficient to distinguish three different cases by types:τ V The output is a token from the terminals V = {Select, From, Where, Id, Max, Min, Count, Sum, Avg, And, =, >, ≥, <, ≤, <END>, <GO>} of our grammar.

τ C The output has to be a column name, which will be copied from either the table header or the question section of X .

Note that the column required for the correct query may not be mentioned explicitly in the question.

τ Q The output is a constant that would be copied from the question section of I.Since the SQL grammar can be written in regular expression form as "Select s c From t Where (c op v) * ", the output types can be described as DISPLAYFORM1 We can then use the type of the output token we want to generate to specialize the decoder.

Decoder RNN We use a standard RNN, based on an LSTM cell with attention over O e to generate the target program O. Notably, we initialize the decoder from both the final hidden states h DISPLAYFORM2 e,fw and the hidden states h DISPLAYFORM3 e,bw generated at index C, the index of the end of the table header in X .

This state forwarding strategy allows the decoder to directly access the encoding of column names to improve decoding accuracy.

Using i DISPLAYFORM4 d to denote the input (resp.

output, hidden state) of the LSTM cell at decoding step k, we define three different output layers for our three output types: DISPLAYFORM5 and O e α (k) is the attention vector.

Then, the input to the next decoder cell is i DISPLAYFORM6 , the concatenation of the token embedding and the attention vector, where the embedding function emb d is a trainable embedding for built-in SQL operators.

τ C , τ Q We use the same approach to compute the attention mask α (k) .

However, instead ofprojecting O e α (k) to obtain the output, the model generates o DISPLAYFORM7 by copying a token v from the input sequence X .

The index l of the token to copy is calculated by l = argmax ([α (k,1) . . .

α (k,|X |) ]), the one with the highest attention value, and the decoder DISPLAYFORM8 d is set to x l .

For the τ Q decoder, only the question part of X is considered.

The input i DISPLAYFORM9 to the next decoder cell reuses the embedding of the copied token, and is computed as the concatenation i DISPLAYFORM10 ) of the token embedding and the attention vector.

As all different decoder types consume and produce similar values, they could easily be exchanged or extended if more types need to be supported.

The advantage of this construction is that only a very small output vocabulary of SQL operators needs to be considered, whereas all other values are obtained through copying.

The model is trained from question-SQL program pairs (X , Y ), where Y = [y BID33 , . . .

, y (|Y |) ] is a sequence representing the ground truth program for question X .

Different typed decoder cells in our model are trained with different loss functions.

τ V loss: This is the standard RNN case, i.e. the loss for an output token is the cross-entropy of the one-hot encoding of the target token and the distribution over the decoder vocabulary V: DISPLAYFORM0 In this case, our objective is to copy a correct token from the input into the output.

As the original input-output pair does not explicitly contain any pointers, we first need to find an index DISPLAYFORM1 In practice, there are often multiple such indices, i.e., the target token appears several times in the input query (e.g., both as a column name supplied from the table information and as part of the user question).

We define two loss functions for this case and evaluate both.• Pointer-based loss: We pick the smallest λ k with y (k) = x (λ k ) and compute the loss as cross entropy between this index and the chosen index, i.e., DISPLAYFORM2 • Value-based loss: While loss pntr C trains the network to generate the correct output sequence, it restricts the model to only point to the first occurrence in the input sequence.

In contrast, we can allow the decoder to choose any one of the input tokens with the correct value.

For that, we define a value-based loss functions that transforms the computed distribution over locations into a distribution over the set of tokens in the input.

We considered to strategies for this: -Max Transfer: This strategy calculates the probability of copying a token v in the input as the maximum probability of pointers that point to token v: DISPLAYFORM3 This strategy calculates the probability of copying a token v in the input vocabulary as the sum of probabilities of pointers that point to token v: DISPLAYFORM4 For both strategies, we calculate the loss function by: DISPLAYFORM5 When training with the sum-transfer loss function, we adapt the outputs of the τ Q and τ C decoder cells to be the tokens with the highest transferred probabilities, computed by DISPLAYFORM6 sum (v)), so that decoding results are consistent with the training goal.

The overall loss for a target output sequence O can then be computed as the sum of the appropriate loss functions for each individual output token o (k) .

We evaluate our model on WikiSQL dataset BID32 ) by comparing it with prior work and our model with different sub-components to analyze their contributions.

We use the sequence version of the WikiSQL dataset with the default train/dev/test split.

Besides question-query pairs, we also use the tables in the dataset to preprocess the dataset.

Preprocessing We first preprocess the dataset by running both tables and question-query pairs through Stanford Stanza using the script included with the WikiSQL dataset, which normalizes punctuation and cases of the dataset.

We further normalize each question based on its corresponding table: for table entries and columns occurring in questions or queries, we normalize their format to be consistent with the table.

This process aims to eliminate inconsistencies caused by different whitespace, e.g. for a column named "country (endonym)" in the table, we normalize its occurrences as "country ( endonym )" in the question to "country (endonym)" so that they are consistent with the entity in table.

Note that we restrict our normalization to only whitespace, comma (','), period ('.') and word permutations to avoid over-processing.

We do not edit tokens: e.g., a phrase "office depot" occurring in a question or a query will not be normalized into "the office depot" even if the latter occurs as a table entry.

Similarly, "california district 10th" won't be normalized to "california 10th", and "citv" won't be normalized to "city".

We also treat each occurrence of a column name or a table entry in questions as a single word for embedding and copying (instead of copying multiple times for multi-word names/constants).Dataset After preprocessing, we filter the training set by removing pairs whose ground truth solution contains constants not mentioned in the question, as our model requires the constants to be copied from the question.

We train and tune our model only on the filtered training and filtered dev set, but we report our evaluation on the full dev and test sets.

We obtain 59,845 (originally 61,297) training pairs, 8,928 (originally 9,145) dev pairs and 17,283 test pairs (the test set is not filtered).Column Annotation We annotate table entry mentions in the question with their corresponding column name iff the table entry mentioned uniquely belongs to one column of the table.

The purpose of this annotation is to bridge special column entries and their column information that cannot be learned elsewhere.

For example, if an entity "rocco mediate" in the question only appears in the "player" column in the table, we annotate the question by concatenating the column name in front of the entity (resulting in "player rocco mediate").

This process resembles the entity linking technique used by , but in a conservative and deterministic way.

We use the pre-trained n-gram embedding by BID10 (100 dimensions) and the GloVe word embedding (100 dimension) by BID21 ; each token is embedded into a 200 dimensional vector.

Both the encoder and decoder are 3-layer bidirectional LSTM RNNs with hidden states sized 100.

The model is trained with question-query pairs with a batch size of 200 for 100 epochs.

During training, we clip gradients at 10 and add gradient noise with η = 0.3, γ = 0.55 to stabilize training BID17 .

The model is implemented in Tensorflow and trained using the Adagrad optimizer BID5 .

Table 1 shows the results of our model with the best performance on the dev set, compared against the augmented pointer model and Seq2SQL model (with RL), both by BID32 .

We report both the accuracy computed with exact syntax match (Acc syn ) and the accuracy based on query execution result (Acc ex ).

Since syntactically different queries can be equivalent on the table (e.g., queries with different predicate orders compared to the ground truth), the execution accuracy in all cases is higher than the corresponding syntax accuracy.

Our best model achieves 61.0% on the filtered dev set, and it is trained with our value-based loss with sum-transfer strategy.

Table 1 : Dev and test accuracy of the model, where Acc syn refers to syntax accuracy and Acc ex refers to execution accuracy.

While the overall results show that our model significantly improves over prior work, we now analyze different sub-components of our model individually to better understand their contribution to the overall performance.

We ran four sets of abalation tests on our model, running each model 5 times.

All model variances are based on the model described in Sect.

4.1 with same hyper-parameters, and the model accuracy on the (filtered) development set during training is plotted in FIG4 .• Type-based decoding: We compare our model with and without type-driven specialization of the decoder cell in FIG4 .

For the untyped model, we directly concatenate all SQL operators in the front of table header and set all decoder cells to copy mode.

The result shows that while types do not significantly improve model performance (with an average improvement 1.4%), they allow the model to stabilize within fewer epochs.

Additionally, we also observed that typed decoders increase the training speed per epoch by approximately ∼23%.• Loss function: We compare the three training objectives and corresponding decoding strategies described in Sect.

3 in FIG4 .

The results show that the sum-transfer strategy significantly improves training stability and model accuracy compared to other strategies typically used in pointer models.

Notably, while the value-based loss with max-transfer strategy outperforms the pointer-based loss in its best runs (with an accuracy of 56.4%), its performance differs greatly between runs and is very sensitive to the chosen initialization.

The results also show that overly constraining the model by only allowing the model to only choose columns from the header and not from their mentions in questions (as in the pointer loss) can have negative impact on the model performance.

• Column Annotation: We study the effect of performing column annotation during preprocessing in FIG4 .

We observe that the model accuracy drops by 7.5% if trained and tested on questions without column annotation.

The result suggests that deterministically linking entities with their column can benefit the model and incorporating entity linking provides an important performance boost.

On the other hand, the results indicate that typed decoding and the value-based loss function alone already reach ∼52.5% accuracy on unannotated questions, beating the Seq2SQL baseline.• Embedding Method: Finally, we study different input token embeddings in FIG4 : untrainable n-gram + GloVe embedding (untrainable in the plot), trainable embedding with n-gram + GloVe initialization (fixed-init) and trainable embedding with random initialization (random-init).

Our results show that incorporating prior knowledge through untrainable embeddings can effectively prevent over-fitting.

To better understand the source of erroneous results, we classify errors made by our model by the part of a query (aggregation function, select column, or predicates) that was incorrectly predicted.

Among the 6,024 incorrectly predicted cases, 32.0% cases use a wrong aggregation function, 47.1% cases copied the wrong column name, and 51.1% cases contain mistakes in predicates (27.6% cases made multiple mistakes).

Notably, most cases with wrong predicates are due to selecting a wrong column to compare to.

Such cases are typically caused by the correct column name is not mentioned in the question (e.g., the questions contains 'best', but the respective column is called 'rank') or because multiple columns with similar names exist (e.g., 'team 1', 'team 2').

These errors suggest that the model lacks understanding of the knowledge presented in the table, and that embedding the table content together with the question BID27 could potentially improve the model.

That our model does not support multiple pointer headers and no external vocabulary for decoding constant results in 13.1% wrong predictions (e.g., our model cannot generate 'intisar field' from 'field of intisar' in the question or generate 'score 4-4' from the question 'which team wins by 4-4?'), which suggests that extending the model with multiple constant pointers per slot or introducing an extra decoding layer for constant rewriting could potentially improve the model.

Finally, we do not directly train our model to learn syntactically different but semantically equivalent program.

62.2% among all wrong queries yield a run-time error or return None during execution.

This suggests that training our model with an reinforcement loop to explicitly punish ill-formed queries and reward semantically equivalent ones BID32 could further improve results.

Semantic Parsing Nearest to our work, mapping natural language to logic forms has been extensively studied in natural language processing research BID31 BID1 BID2 BID26 BID11 BID12 BID23 are closely related neural semantic parsers adopting tree-based decoding that also utilize grammar production rules as decoding constraints.

However, our model foregoes the complexity of generating a full parse tree and never produces non-terminal nodes, and instead retains the simplicity of a sequence decoder.

This makes it substantially easier to implement and train, as the sequence model requires no explicit controller for production rule selection.

To our knowledge, our model is also the first to use target token type information to specialize the decoder to a mode in which it copies from a type-compatible, restricted set of input tokens.

Pointer Networks Pointer and copy networks enhance RNNs with the ability to reuse input tokens, and they have been successfully used in interactive conversation BID7 , geometric problems BID25 and program generation BID32 .

Our model differs from previous approaches in that we use types to explicitly restrict locations in the input to point to; furthermore, we developed a new training objective to handle pointer aliases.

Program Induction / Synthesis Program induction BID24 BID18 BID6 BID29 aims to induce latent programs for question answering; on the other hand, program synthesis models BID32 BID19 aim to generate explicit programs and execute the program to obtain answer.

Our model follows the line of neural program synthesis models and trains directly with question program pairs.

Orthogonal Approaches Entity linking BID3 BID27 ) is a technique used to link knowledge between the encoding sequence and knowledge base (e.g., table, document) in semantic parsing that is orthogonal to the neural encoder decoder model.

This technique can potentially be used to address our limitation in our deterministic column annotation process.

Besides, reinforcement learning BID32 allows the model to freely learn semantically equivalent solutions to user questions, and can be combined with our model to further improve its accuracy.

We presented a new sequence to sequence based neural architecture to translate natural language questions over tables into executable SQL queries.

Our approach uses a simple type system to guide the decoder to either copy a token from the input using a pointer-based copying mechanism or generate a token from a finite vocabulary.

We presented a sum-transfer value based loss function that transforms a distribution over pointer locations into a distribution over token values in the input to efficiently train the architecture.

Our evaluation on the WikiSQL dataset showed that our model significantly outperforms the current state-of-the-art Seq2SQL model.

@highlight

We present a type-based pointer network model together with a value-based loss method to effectively train a neural model to translate natural language to SQL.

@highlight

The paper claims to develop a novel method to map natural language queries to SQL by using a grammar to guide decoding and using a new loss function for pointer / copy mechanism