We present a sequence-to-action parsing approach for the natural language to SQL task that incrementally fills the slots of a SQL query with feasible actions from a pre-defined inventory.

To account for the fact that typically there are multiple correct SQL queries with the same or very similar semantics, we draw inspiration from syntactic parsing techniques and propose to train our sequence-to-action models with non-deterministic oracles.

We evaluate our models on the WikiSQL dataset and achieve an execution accuracy of 83.7% on the test set, a 2.1% absolute improvement over the models trained with traditional static oracles assuming a single correct target SQL query.

When further combined with the execution-guided decoding strategy, our model sets a new state-of-the-art performance at an execution accuracy of 87.1%.

Many mission-critical applications in health care, financial markets, and business process management store their information in relational databases BID10 BID22 BID16 .

Users access that information using a query language such as SQL.

Although expressive and powerful, SQL is difficult to master for non-technical users.

Even for an expert, writing SQL queries can be challenging as it requires knowing the exact schema of the database and the roles of various entities in the query.

Hence, a long-standing goal has been to allow users to interact with the database through natural language BID0 BID24 .The key to achieving this goal is understanding the semantics of the natural language statements and mapping them to the intended SQL.

This problem, also known as NL2SQL, was previously understudied largely due to the availability of annotation.

Without paired natural language statement and SQL query, a weak supervision approach may be adopted which reduces supervision from annotated SQL queries to answers BID19 .

This is a more difficult learning problem.

Therefore only with recent release of a number of large-scale annotated NL2SQL datasets BID36 BID6 , we start to see a surge of interest in solving this problem.

Existing NL2SQL approaches largely fall into two categories: sequence-to-sequence style neural "machine translation " systems BID36 BID5 and sets of modularized models with each predicting a specific part of the SQL queries BID32 BID34 .

The former class suffer from the requirement of labeling a single ground truth query while multiple semantically equivalent queries exist for each intent.

For example, as noticed by BID36 , the ordering of filtering conditions in a query does not affect execution but affects generation.

To account for this, techniques such as reinforcement learning have been used on top of those sequenceto-sequence models.

The second class of models employ a sequence-to-set approach: they first predict table columns present in the query and then independently predict the rest for each column.

This avoids the ordering issue, but makes it harder to leverage inter-dependencies among conditions.

In this work, we develop a sequence-to-action parsing approach (Section 3) for the NL2SQL problem.

It incrementally fills the slots of a SQL query with actions from an inventory designed for this task.

Taking inspiration from training oracles in incremental syntactic parsing BID8 , we further propose to use non-deterministic oracles (Section 4) for training the incremental parsers.

These oracles permit multiple correct action continuations from a partial parse, thus are able to account for the logical form variations.

Our model combines the advantage of a sequence-to-sequence model that captures inter-dependencies within sequence of predictions and a SELECT`Height (ft)??? HERE Name="Willis Tower" AND Location="Chicago" DISPLAYFORM0 What is the height of Willis Tower in Chicago?Figure 1: Our running example.

The input is a natural language question and a table schema, and the output is an executable SQL query.

Table contents are shown here, but unknown to our models.modularized model that avoids any standarized linearization of the logical forms.

We evaluate our models on the WikiSQL dataset and observe a performance improvement of 2.1% when comparing non-deterministic oracles with traditional static oracles.

We further combine our approach and the execution-guided decoding strategy ) and achieve a new state-of-the-art performance with 87.1% test execution accuracy.

Experiments on a filtered ATIS dataset in addition confirm that our models can be applied to other NL2SQL datasets.

Given an input natural language question, our goal is to generate its corresponding SQL query.

In the following and throughout the paper, we use the WikiSQL dataset BID36 as our motivating example.

However, it should be noted that our approach is generally applicable to other NL2SQL data, with proper choice of an action inventory and redesign of parser states.

Figure 1 shows an example.

The SQL structure of the WikiSQL dataset queries is restricted and always follows the template SELECT agg selcol WHERE col op val (AND col op val) * .

Here, selcol is a single table column and agg is an aggregator (e.g., COUNT, SUM, empty).

The WHERE segment is a sequence of conjunctive filtering conditions.

Each op is a filtering operator (e.g., =) and the filtering value val is mentioned in the question.

Although the dataset comes with a "standard" linear ordering of the conditions, the order is actually irrelevant given the semantics of AND.Throughout the paper we denote the input to the parser as x. It consists of a natural language question w with tokens w i and a single table schema c with column names c j .

A column name c j can have one or more tokens.

The parser needs to generate an executable SQL query y as its output.

Given an input x, the generation of a structured output y is broken down into a sequence of parsing decisions.

The parser starts from an initial state and incrementally takes actions according to a learned policy.

Each action advances the parser from one state to another, until it reaches one of the terminal states, where we may extract a complete logical form y.

We take a probabilistic approach to model the policy.

It predicts a probability distribution over the valid set of subsequent actions given the input x and the running decoding history.

The goal of training such an incremental semantic parser is then to optimize this parameterized policy.

Formally, we let P ?? (y|x) = P ?? (a|x), where ?? is model parameters.

Execution of the action sequence a = {a 1 , a 2 , . . . , a k } leads the parser from the initial state to a terminal state that contains the parsing result y.

Here we assume that each y has only one corresponding action sequence a, an assumption that we will revisit in Section 4.

The probability of action sequence is further factored as the product of incremental decision probabilities: DISPLAYFORM0 During inference, instead of attempting to enumerate over the entire output space and find the highest scoring a * = arg max a P ?? (a|x), our decoder takes a greedy approach: at each intermediate step, it picks the highest scoring action according to the policy: a * i = arg max ai P ?? (a i |x, a * <i ).

Resulting state after taking the action at state p Parameter representation In the following subsections, we define the parser states and the inventory of actions, followed by a description of our encoder-decoder neural-network model architecture.

DISPLAYFORM0

We first look at a structured representation of a full parse corresponding to the example in Figure 1 : Table 1 .

The action CONDVAL selects a span of text w i:j from the input question w. In practice, this leads to a large number of actions, quadratic in the length of the input question, so we break down CONDVAL into two consecutive actions, one selecting the starting position w i and the other selecting the ending position w j for the span.

At the end of the action sequence, we append a special action END that terminates the parsing process and brings the parser into a terminal state.

As an example, the query in Figure 1 translates to an action sequence of { AGG(NONE), SELCOL(c 3 ), CONDCOL(c 1 ), CONDOP(=), CONDVAL(w 5:6 ), CONDCOL(c 2 ), CONDOP(=), CONDVAL(w 8:8 )}.

DISPLAYFORM0 The above definitions assume all the valid sequences to have the form of AGG SELCOL (CONDCOL CONDOP CONDVAL) * END.

This guarantees that we can extract a complete logical form from each terminal state.

For other data with different SQL structure, a redesign of action inventory and parser states is required.

We first assume that we have some context-sensitive representations r The main component of our decoder is to model a probability distribution P ?? (a|x, a <i ) over potential parser actions a conditioned on input x and past actions a <i .

It has two main challenges:(1) there is no fixed set of valid parser actions: it depends on the input and the current parser state; (2) the parser decision is context-dependent: it relies on the decoding history and the information embedded in the input question and column headers.

We adopt an LSTM-based decoder framework and address the first challenge through individual scoring of actions.

The model scores each candidate action a as s a and uses a softmax function to normalize the scores into a probability distribution.

At time step i, we denote the current decoder hidden state as h DEC iand model the score of a in the form of a bilinear function: DISPLAYFORM0 , where r A a is a vector representation of the action a and is modeled as the concatenation of the action embedding and the parameter representation.

The form of the latter is given in Table 1 .The dependencies between the parser decisions and the input question and column headers are captured through a dot-product attention mechanism BID20 .

The input to the first layer of our decoder LSTM at time step i + 1 is a concatenation of the output action representation r A ai from previous time step i, a question attention vector e W i , and a column header attention vector e DISPLAYFORM1

Now we return to the context-sensitive representations r W i and r C j .

Ideally, these representations should be both intra-context-sensitive, i.e. aware of information within the sequences, and intersequence-dependent, i.e. utilizing knowledge about the other sequences.

These intuitions are reflected in our model design as intra-sequence LSTMs, self-attention and cross-serial attention.

Our model architecture is illustrated in FIG1 .

Each word w i is first mapped to its embedding, and then fed into a bi-directional LSTM (bi-LSTM) that associates each position with a hidden state h W i .

For column headers, since each column name can have multiple words, we apply word embedding lookup and bi-LSTM for each column name, and use the final hidden state from the bi-LSTM as the initial representation for the column name.

Next, we apply self-attention BID27 to contextualize this initial representation into h C j .

After obtaining these intra-contextsensitive representations h W i and h C j , we use cross-serial dot-product attention BID20 to get a weighted average of h

Previously, we assumed that each natural language question has a single corresponding SQL query, and each query has a single underlying correct action sequence.

However, these assumptions do not hold in practice.

One well-observed example is the ordering of the filtering conditions in the WHERE clause.

Reordering of those conditions leads to different action sequences.

Furthermore, we identify another source of ambiguity in section 4.2, where a question can be expressed by different SQL queries with the same execution results.

These queries are equivalent from an end-user perspective.

For both cases, we obtain multiple correct "reference" transition sequences for each training instance and there is no single target policy for our model to mimic during training.

To solve this, we draw inspiration from syntactic parsing and define non-deterministic oracles BID8 ) that allow our parser to explore alternative correct action sequences.

In contrast, the training mechanism we discussed in Section 3 is called static oracles.

We denote the oracle as O that returns a set of correct continuation actions O(x, a <t ) at time step t. Taking any action from the set can lead to some desired parse among a potentially large set of correct results.

The training objective for each instance L x is defined as: DISPLAYFORM0 where a <i denotes the sequence of actions a 1 , . . .

, a i???1 and a i = arg max a???O(x,a<i) s a , the most confident correct action to take as decided by the parser during training.

When O is a static oracle, it always contains a single correct action.

In that scenario, Equation 1 is reduced to a na??ve crossentropy loss.

When O is non-deterministic, the parser can be exposed to different correct action sequences and it is no longer forced to conform to a single correct action sequence during training.

Training a text-to-SQL parser is known to suffer from the so-called "order-matters" issue.

The filtering conditions of the SQL queries do not presume any ordering.

However, an incremental parser must linearize queries and thus impose a pre-defined order.

A correct prediction that differs from a golden labeling in its ordering of conditions then may not be properly rewarded.

Prior work has tackled this issue through reinforcement learning BID36 ) and a modularized sequenceto-set solution BID32 .

The former lowers optimization stability and increases training time, while the latter complicates model design to capture inter-dependencies among clauses: information about a predicted filtering condition is useful for predicting the next condition.

We leverage non-deterministic oracles to alleviate the "order-matters" issue.

Our model combines the advantage of an incremental approach to leverage inter-dependencies among clauses and the modularized approach for higher-quality training signals.

Specifically, at intermediate steps for predicting the next filtering condition, we accept all possible continuations, i.e. conditions that have not been predicted yet, regardless of their linearized positions.

For the example in Figure 1 , in addition to the transition sequence we gave in Section 3.1, our non-deterministic oracles also accept CONDCOL(c 2 ) as a correct continuation of the second action.

If our model predicts this action first, it will continue predicting the second filtering condition before predicting the first.

In preliminary experiments, we observed that a major source of parser errors on the development set is incorrect prediction of implicit column names.

Many natural language queries do not explicitly mention the column name of the filtering conditions.

For example, the question in Figure 1 does not mention the column name "Name".

Similarly, a typical question like "What is the area of Canada?" does not mention the word "country".

For human, such implicit references make natural language queries succinct, and the missing information can be easily inferred from context.

But for a machine learning model, they pose a huge challenge.

We leverage the non-deterministic oracles to learn the aforementioned implicit column name mentions by accepting the prediction of a special column name, ANYCOL.

During execution, we expand such predictions into disjunction of filtering conditions applied to all columns, simulating the intuition why a human can easily locate a column name without hearing it from the query.

For the example in Figure 1 , in addition to the action CONDCOL(c 1 ), we also allow an alternative prediction CONDCOL(ANYCOL).

When the latter appears in the query (e.g. ANYCOL="Willis Tower"), we expand it into a disjunctive clause (Rank="Willis Tower" OR Name="Willis Tower" OR ...).

With our non-deterministic oracles, when column names can be unambiguously resolved using the filtering values, we accept both ANYCOL and the column name as correct actions during training, allowing our models to predict whichever is easier to learn.

In our experiments, we use the default train/dev/test split of the WikiSQL dataset.

We evaluate our models trained with both the static oracles and the non-deterministic oracles on the dev and test split.

We report both logical form accuracy (i.e., exact match of SQL queries) and execution accuracy (i.e., the ratio of predicted SQL queries that result in the same answer after execution).

The execution accuracy is the metric that we aim to optimize.

We largely follow the preprocessing steps in prior work of BID5 .

Before the embedding layer, only the tokens which appear at least twice in the training data are retained in the vocabulary, the rest are assigned a special "UNK" token.

We use the pre-trained GloVe embeddings BID23 , and allow them to be fine-tuned during training.

Embeddings of size 16 are used for the actions.

We further use the type embeddings for the natural language queries and column names following BID34 : for each word w i , we have a discrete feature indicating whether it appears in the column names, and vice versa for c j .

These features are embedded into 4-dimensional vectors and are concatenated with word embeddings before being fed into the biLSTMs.

The encoding bi-LSTMs have a single hidden layer with size 256 (128 for each direction).

The decoder LSTM has two hidden layers each of size 256.

All the attention connections adopt the dot-product form as described in Section 3.2.For the training, we use a batch size of 64 with a dropout rate of 0.3 to help with the regularization.

We use Adam optimizer BID14 with the default initial learning rate of 0.001 for the parameter update.

Gradients are clipped at 5.0 to increase stability in training.

The main results are presented in TAB4 .

Our model trained with static oracles achieves comparable results with the current state-of-the-art Coarse2Fine BID5 and MQAN (McCann et al., 2018) models.

On top of this strong model, using non-deterministic oracles during training leads to a large improvement of 2.1% in terms of execution accuracy.

The significant drop in the logical form accuracy is expected, as it is mainly due to the use of ANYCOL option for the column choice: the resulting SQL query may not match the original annotation.

We further separate the contribution of "order-matters" and ANYCOL for the non-deterministic oracles.

When our non-deterministic oracles only address the "order-matters" issue as described in Section 4.1, the model performance stays roughly the same compared with the static-oracle model.

We hypothesize that it is because the ordering variation presented in different training instances is already rich enough for a vanilla sequence-to-action model to learn well.

Adding ANYCOL to the oracle better captures the implicit column name mentions and has a significant impact on the performance, increasing the execution accuracy from 81.8% to 83.7%.Our incremental parser uses a greedy strategy for decoding, i.e. picking the highest scoring action predicted by the policy.

A natural extension is to expand the search space using beam search decoding.

We further incorporate the execution-guided strategy Table 3 : Execution accuracy (%) and decoding speed of our models on the test set of WikiSQL, with varying decoding beam size.

The notation "+ EG (k)" is as in TAB4 .errors and empty results.

The key insight is that a partially generated output can already be executed using the SQL engine against the database, and the execution results can be used to guide the decoding.

The decoder maintains a state for the partial output, which consists of the aggregation operator, selection column and the completed filtering conditions until that stage in decoding.

After every action, the execution-guided decoder retains the top-k scoring partial SQL queries free of runtime exceptions and empty output.

At final stage, the query with the highest likelihood is chosen.

With k = 5, the execution-guided decoder on top of our previous best-performing model achieves an execution accuracy of 87.1% on the test set, setting a new state of the art.

We also report the performance of the static oracle model with execution-guided decoding in Table 3 .

It comes closely to the performance of the non-deterministic oracle model, but requires a larger beam size, which translates to an increase in the decoding time.

To test whether our model can generalize to other datasets, we perform experiments with the ATIS dataset BID25 BID3 .

ATIS has more diverse SQL structures, including queries on multiple tables and nested queries.

To be compatible with our task setting, we only retain examples in the ATIS dataset that are free of nested queries, containing only AND operations and no INNER JOIN operators.

We perform table joins and create a single table to be included in the input to our models along with the natural language question.

The reduced dataset consists of 933 examples, with 714/93/126 examples in the train/dev/test split, respectively.

Our models trained with the static and non-deterministic oracles (without ANYCOL) achieve accuracy of 67.5% and 69.1% on the test set, respectively.

The improvement gained from using nondeterministic oracles during training validates our previous hypothesis: ATIS is a much smaller dataset compared with WikiSQL, therefore explicitly addressing "order-matters" helps here.

We didn't apply ANYCOL due to the nature of ATIS data.

WikiSQL, introduced by BID36 , is the first large-scale dataset with annotated pairs of natural language queries and their corresponding SQL forms on a large selection of table schemas.

While its coverage of SQL syntax is weaker than previous datasets such as ATIS BID25 BID3 and GeoQuery BID35 , WikiSQL is highly diverse in its questions, BID29 BID32 BID34 BID5 BID21 .NL2SQL is a special case of semantic parsing.

The task of semantic parsing maps natural language to a logical form representing its meaning, and has been studied extensively by the natural language processing community (see Liang 2016 for a survey).

The choice of meaning representation is usually task-dependent, including lambda calculus BID31 , lambda dependency-based compositional semantics (Liang, 2013, ??-DCS) , and SQL BID36 .

Neural semantic parsing, on the other hand, views semantic parsing as a sequence generation problem.

It adapts deep learning models such as those introduced by BID26 ; BID1 ; BID28 .

Combined with data augmentation BID13 BID12 or reinforcement learning BID36 , sequence-to-sequence with attention and copying has already achieved state-of-the-art results on many datasets including WikiSQL.The meaning representation in semantic parsing usually has strict grammar syntax, as opposed to target sentences in machine translation.

Thus, models are often constrained to output syntactically valid results.

BID4 propose models that generate tree outputs through hierarchical decoding and models that use sketches to guide decoding, but they do not explicitly deal with grammar constraints.

In contrast, BID33 and directly utilize grammar productions during decoding.

Training oracles have been extensively studied for the task of syntactic parsing, where incremental approaches are common BID8 .

For syntactic parsing, due to the more structurally-constrained nature of the task and clearly-defined partial credits for evaluation, dynamic oracles allow the parsers to find optimal subsequent actions even if they are in some sub-optimal parsing states BID7 BID9 BID2 .

In comparison, non-deterministic oracles are defined for the optimal parsing states that have potential to reach a perfect terminal state.

To the best of our knowledge, our work is the first to explore non-deterministic training oracles for incremental semantic parsing.

In this paper, we introduce a sequence-to-action incremental parsing approach for the NL2SQL task.

With the observation that multiple SQL queries can have the same or very similar semantics corresponding to a given natural language question, we propose to use non-deterministic oracles during training.

On the WikiSQL dataset, our model trained with the non-deterministic oracles achieves an execution accuracy of 83.7%, which is 2.3% higher than the current state of the art.

We also discuss using execution-guided decoding in combination with our model.

This leads to a further improvement of 3.4%, achieving a new state-of-the-art 87.1% execution accuracy on the test set.

To the best of our knowledge, our work is the first to use non-deterministic oracles for training incremental semantic parsers.

Designing such non-deterministic oracles requires identification of multiple correct transition sequences for a given training instance, and an algorithm that decides the possible continuations for any intermediate state that will lead to one of the desired terminal states.

We have shown promising results for WikiSQL and filtered ATIS dataset and it would be interesting to extend our work to other more complex NL2SQL tasks and to other semantic parsing domains.

<|TLDR|>

@highlight

We design incremental sequence-to-action parsers for text-to-SQL task and achieve SOTA results. We further improve by using non-deterministic oracles to allow multiple correct action sequences. 