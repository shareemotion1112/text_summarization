When translating natural language questions into SQL queries to answer questions from a database, contemporary semantic parsing models struggle to generalize to unseen database schemas.

The generalization challenge lies in (a) encoding the database relations in an accessible way for the semantic parser, and (b) modeling alignment between database columns and their mentions in a given query.

We present a unified framework, based on the relation-aware self-attention mechanism,to address schema encoding, schema linking, and feature representation within a text-to-SQL encoder.

On the challenging Spider dataset this framework boosts the exact match accuracy to 53.7%, compared to 47.4% for the previous state-of-the-art model unaugmented with BERT embeddings.

In addition, we observe qualitative improvements in the model’s understanding of schema linking and alignment.

The ability to effectively query databases with natural language has the potential to unlock the power of large datasets to the vast majority of users who are not proficient in query languages.

As such, a large body of research has focused on the task of translating natural language questions into queries that existing database software can execute.

The release of large annotated datasets containing questions and the corresponding database SQL queries has catalyzed progress in the field, by enabling the training of supervised learning models for the task.

In contrast to prior semantic parsing datasets (Finegan-Dollak et al., 2018) , new tasks such as WikiSQL (Zhong et al., 2017) and Spider (Yu et al., 2018b) pose the real-life challenge of generalization to unseen database schemas.

Every query is conditioned on a multi-table database schema, and the databases do not overlap between the train and test sets.

Schema generalization is challenging for three interconnected reasons.

First, any text-to-SQL semantic parsing model must encode a given schema into column and table representations suitable for decoding a SQL query that might involve any of the given columns or tables.

Second, these representations should encode all the information about the schema, including its column types, foreign key relations, and primary keys used for database joins.

Finally, the model must recognize natural language used to refer to database columns and tables, which might differ from the referential language seen in training.

The latter challenge is known as schema linking -aligning column/table references in the question to the corresponding schema columns/tables.

While the question of schema encoding has been studied in recent literature (Bogin et al., 2019b) , schema linking has been relatively less explored.

Consider the example in Figure 1 .

It illustrates the challenge of ambiguity in linking: while "model" in the question refers to car_names.model rather than model_list.model, "cars" actually refers to both cars_data and car_names (but not car_makers) for the purpose of table joining.

To resolve the column/table references properly, the semantic parser must take into account both the known schema relations (e.g. foreign keys) and the question context.

Prior work (Bogin et al., 2019b) addressed the schema representation problem by encoding the directed graph of foreign key relations among the columns with a graph neural network.

While effective, this approach has two important shortcomings.

First, it does not contextualize schema encoding with the question, thus making it difficult for the model to reason about schema linking after both the column representations and question word representations have been built.

Second, it limits information propagation during schema encoding to predefined relations in the schema such as foreign keys.

The advent of self-attentional mechanisms in natural language processing (Vaswani et al., 2017) shows that global reasoning is crucial to building effective representations of relational structures.

However, we would like any global reasoning to also take into account the aforementioned predefined schema relations.

In this work, we present a unified framework, called RAT-SQL, 1 for encoding relational structure in the database schema and a given question.

It uses relation-aware self-attention to combine global reasoning over the schema entities and question words with structured reasoning over predefined schema relations.

We then apply RAT-SQL to the problems of schema encoding and schema linking.

As a result, we obtain 53.7% exact match accuracy on the Spider test set.

At the time of writing, this result is the state of the art among models unaugmented with pretrained BERT embeddings.

In addition, we experimentally demonstrate that RAT-SQL enables the model to build more accurate internal representations of the question's true alignment with schema columns and tables.

Semantic parsing of natural language to SQL queries recently surged in popularity thanks to the creation of two new multi-table datasets with the challenge of schema generalization -WikiSQL (Zhong et al., 2017) and Spider (Yu et al., 2018b) .

Schema encoding is not as challenging in WikiSQL as in Spider thanks to the lack of multi-table relations.

Schema linking is relevant for both tasks but also more challenging in Spider due to the richer natural language expressiveness and less restricted SQL grammar observed in it.

Indeed, the state of the art semantic parser on WikiSQL achieves a test set accuracy of 91.8%, significantly higher than the state of the art on Spider.

The recent state-of-the-art models evaluated on Spider use various attentional architectures for question/schema encoding and AST-based structural architectures for query decoding.

IRNet (Guo et al., 2019) encodes the question and schema separately with LSTM and self-attention respectively, augmenting them with custom type vectors for schema linking.

They further use the AST-based decoder of Yin and Neubig (2017) to decode a query in an intermediate representation (IR) that exhibits higher-level abstraction structure than SQL.

Bogin et al. (2019b) encode the schema with a graph neural network and a similar grammar-based decoder.

Both approaches highlight the importance of schema encoding and schema linking, but design separate feature engineering techniques to augment word vectors (as opposed to relations between words and columns) to resolve it.

In contrast, the relational framework of RAT-SQL provides a unified way to encode arbitrary relational information among the inputs.

Concurrently with this work, Bogin et al. (2019a) published Global-GNN, a different approach to schema linking for Spider which applies global reasoning between question words and schema columns/tables.

Global reasoning is implemented by gating the graph neural network that computes the representation of schema elements using question token representations.

This conceptually differs from RAT-SQL in two important ways: (a) question word representations influence the schema representations but not vice versa, and (b) like in other GNN-based encoding approaches, message propagation is limited to the schema-induced edges such as foreign key relations.

In contrast, our Table 1 to reduce clutter.

relation-aware transformer mechanism allows encoding arbitrary relations between question words and schema elements explicitly, and these representations are computed jointly using self-attention.

We use the same formulation of relation-aware self-attention as Shaw et al. (2018) .

However, that work only applied it to sequences of words in the context of machine translation, and as such, their set of relation types only encoded the relative distance between two words.

We extend their work and show that relation-aware self-attention can effectively encode more complex relationships that exist within an unordered sets of elements (in this case, columns and tables within a database schema as well as relations between the schema and the question).

To the best of our knowledge, this is the first application of relation-aware self-attention to joint representation learning with both predefined and softly induced relations in the input structure.

We now describe the RAT-SQL framework and its application to the problems of schema encoding and linking.

First, we formally define the text-to-SQL semantic parsing problem and its components.

Then, we introduce the relation-aware self-attention mechanism, our framework for jointly encoding relational structure between the question and the schema.

Finally, we present our implementation of schema linking in the RAT-SQL framework.

Given a natural language question Q and a schema S = C, T for a relational database, our goal is to generate the corresponding SQL P .

Here the question Q = q 1 . . .

q |Q| is a sequence of words, and the schema consists of columns C = {c 1 , . . .

, c |C| } and tables T = t 1 , . . . , t |T | .

Each column name c i contains words c i,1 , . . .

, c i,|ci| and each table name t i contains words t i,1 , . . .

, t i,|ti| .

The desired program P is represented as an abstract syntax tree T in the context-free grammar of SQL.

Some columns in the schema are primary keys, used for uniquely indexing the corresponding table, and some are foreign keys, used to reference a primary key column in a different table.

As described in Section 1, we would like to softly bias our schema encoding mechanism toward these predefined relations.

In addition, each column has a type τ such as number or text.

Schema linking aims at finding the alignment between question words and mentioned columns or tables.

It's a crucial step for a parser to generate the right columns and tables in SQL.

We model the latent alignment explicitly using an alignment matrix (Section 3.6), which is softly biased towards some string-match based relations, as inspired by Guo et al. (2019) .

To support reasoning about relationships between schema elements in the encoder, we begin by representing the database schema using a directed graph G, where each node and edge has a label.

We represent each table and column in the schema as a node in this graph, labeled with the words in the name; for columns, we prepend the type of the column to the label.

For each pair of nodes x and y in the graph, Table 1 describes when there exists an edge from x to y and the label it should have.

Figure 2 illustrates an example graph (although not all edges and labels are shown).

Tree-structured decoder Self-attention layers (c) The decoder, choosing a column (Section 3.7)

Figure 3: Overview of the stages of our approach.

We now obtain an initial representation for each of the nodes in the graph, as well as for the words in the input question.

For the graph nodes, we use a bidirectional LSTM (BiLSTM) over the words contained in the label.

We concatenate the output of the initial and final time steps of this LSTM to form the embedding for the node.

For the question, we also use a bidirectional LSTM over the words:

where each of the BiLSTM functions first lookup word embeddings for each of the input tokens.

The LSTMs do not share any parameters.

At this point, we have representations c init i , t init i , and q init i .

Similar to encoders used in some previous papers, these initial representations are independent of each other (uninfluenced by which other columns or tables are present).

Now, we would like to imbue these representations with the information in the schema graph.

We use a form of self-attention (Vaswani et al., 2017) that is relation-aware (Shaw et al., 2018) to achieve this goal.

In one step of relation-aware self-attention, we begin with an input x of n elements (where x i ∈ R dx ) and transform each x i into y i ∈ R dx .

We follow the formulation described in Shaw et al. (2018) :

where FC is a fully-connected layer, 1 ≤ h ≤ H, and W

The r ij terms encode the relationship between the two elements x i and x j in the input.

We explain how we obtain r ij in the next part.

Application Within Our Encoder At the start, we construct the input x of |c| + |t| + |q| elements using c init i , t init i , and q init i :

We then apply a stack of N relation-aware self-attention layers, where N is a hyperparameter.

The weights of the encoder layers are not tied; each layer has its own set of weights.

After processing through the stack of N encoder layers, we obtain Description of edge types present in the directed graph created to represent the schema.

An edge exists from source node x ∈ S to target node y ∈ S if the pair fulfills one of the descriptions listed in the table, with the corresponding label.

Otherwise, no edge exists from x to y.

Column Column SAME-TABLE x and y belong to the same table.

FOREIGN-KEY-COL-F x is a foreign key for y. FOREIGN-KEY-COL-R y is a foreign key for x.

Table PRIMARY-KEY-F x is the primary key of y.

x is a column of y (but not the primary key).

Table  Column PRIMARY-KEY-R y is the primary key of x. BELONGS-TO-R y is a column of x (but not the primary key).

Table  Table   FOREIGN -KEY-TAB-F Table x has a foreign key column in y. FOREIGN-KEY-TAB-R Same as above, but x and y are reversed.

FOREIGN-KEY-TAB-B x and y have foreign keys in both directions.

We use c We define a discrete set of possible relation types, and map each type to an embedding to obtain r V ij and r K ij .

We need a value of r ij for every pair of elements in x. In the subsequent sections, we describe the set of relation types we used.

If x i and x j both correspond to nodes in G (i.e. each is either a column or table) with an edge from x i to x j , then we use the label on that edge (possibilities listed in Table 1 ) for r ij .

However, this is not sufficient to obtain r ij for every pair of i and j. The graph G has no nodes corresponding to the question words, not every pair of schema nodes has an edge between them, and there is no self-edges (for when i = j).

As such, we add more types beyond what is defined in Table 1: • If i = j, then COLUMN-IDENTITY or TABLE-IDENTITY.

•

• x i ∈ question, x j ∈ column ∪ table; or x i ∈ column ∪ table, x j ∈ question: see Section 3.6.

• Otherwise, one of COLUMN-COLUMN, COLUMN -TABLE, TABLE-COLUMN, or TABLE-TABLE.

3.6 SCHEMA LINKING To aid the model with aligning column/table references in the question to the corresponding schema columns/tables, we furthermore define relation types which indicate when parts of the question textually match the names of the columns and tables.

Specifically, for all n-grams of length 1 to 5 in the question, we determine (1) whether it exactly matches the name of a column/table (exact match); or (2) whether the n-gram is a subsequence of the name of a column/table (partial match).

Therefore, for the case where x i ∈ question, x j ∈ column ∪ table; or x i ∈ column ∪ table, x j ∈ question, we set r ij to QUESTION-COLUMN-M, QUESTION-TABLE-M, COLUMN-QUESTION-M or TABLE-QUESTION-M depending on the type of x i and x j .

M is one of EXACTMATCH, PARTIALMATCH, or NOMATCH.

In the end, we add 2 + 5 + (4 × 3) + 4 types (one term per bullet in Section 3.5) beyond the 10 in Table 1 , for a total of 33 types.

Memory-Schema Alignment Matrix Our intuition suggests that the columns and tables which occur in the SQL P will generally have a corresponding reference in the natural language question (for example, "cars" and "cylinders" in Figure 1 ).

To capture this intuition in the model, we apply relation-aware attention as a pointer mechanism between every memory element in y and all the columns/tables to compute explicit alignment matrices L col ∈ R |y|×|C| and L tab ∈ R |y|×|T | :

The memory-schema alignment matrix is expected to resemble the real discrete alignments, therefore should respect certain constraints like sparsity.

For example, the question word "model" in Figure 1 should be aligned with car_names.model rather than model_list.model or model_-list.model_id.

To further bias the soft alignment towards the real discrete structures, we add an auxiliary loss to encourage sparsity of the alignment matrix.

Specifically, for a column/table that is mentioned in the SQL query, we treat the model's current belief of the best alignment as the ground truth.

Then we use a cross-entropy loss, referred as alignment loss, to strengthen the model's belief:

where Rel(C) and Rel(T ) denote the set of relevant columns and tables that appear in the SQL P .

Once we have obtained an encoding of the input, we used the decoder from Yin and Neubig (2017) to generate the SQL P .

The decoder generates P as an abstract syntax tree in depth-first traversal order, by using an LSTM to output a sequence of decoder actions that (i) expand the last generated node in the tree according to the grammar, called APPLYRULE; or when necessary to complete the last node, (ii) chooses a column or table from the schema, called SELECTCOLUMN and SELECTTABLE.

Formally, we have the following:

where y is the final encoding of the question and schema from the previous section, and a <t are all previous actions.

We update the LSTM's state in the following way: m t , h t = f LSTM ([a t−1 z t h pt a pt n ft ], m t−1 , h t−1 ) where m t is the LSTM cell state, h t is the LSTM output at step t, a t−1 is the embedding of the previous action, p t is the step corresponding to expanding the parent AST node of the current node, and n ft is the embedding of the current node type.

We obtain z t using multi-head attention (with 8 heads) on h t−1 over y.

is a 2-layer MLP with a tanh non-linearity.

For SELECTCOLUMN, we computẽ

and similarly for SELECTTABLE.

We implemented our model using PyTorch (Paszke et al., 2017) .

During preprocessing, the input of questions, column names and table names are tokenized and lemmatized with the StandfordNLP toolkit .

Within the encoder, we use GloVe (Pennington et al., 2014) word embeddings, held fixed in training except for the 50 most common words in the training set.

All word embeddings have dimension 300.

The bidirectional LSTMs have hidden size 128 per direction, and use the recurrent dropout method of Gal and Ghahramani (2016) with rate 0.2.

We stack 8 relation-aware self-attention layers on top of the bidirectional LSTMs.

Within the relation-aware self-attention layers, we set d x = d z = 256, H = 8, and use dropout with rate 0.1.

The position-wise feed-forward network has inner layer dimension 1024.

Inside the decoder, we use rule embeddings of size 128, node type embeddings of size 64, and a hidden size of 512 inside the LSTM with dropout rate 0.21.

We used the Adam optimizer (Kingma and Ba, 2014) with β 1 = 0.9, β 2 = 0.999, and = 10 −9 , which are defaults in PyTorch.

During the first warmup_steps = max_steps/20 steps of training, we linearly increase the learning rate from 0 to 7.4 × 10 −4 .

Afterwards, the learning rate is annealed to 0, with formula 10 −3 (1 − step−warmup_steps max_steps−warmup_steps ) −0.5 .

For all parameters, we used the default initialization method in PyTorch.

We use a batch size of 20 and train for up to 40,000 steps.

We use the Spider dataset (Yu et al., 2018b ) for all our experiments.

As described by Yu et al. (2018b) , the training data contains 8,659 examples, including 1,659 examples (questions and queries, with the accompanying schemas) from the Restaurants (Popescu et al., 2003; Tang and Mooney, 2000) , GeoQuery (Zelle and Mooney, 1996) , Scholar (Iyer et al., 2017) , Academic (Li and Jagadish, 2014) , Yelp and IMDB (Yaghmazadeh et al., 2017) datasets.

As Yu et al. (2018b) make the test set accessible only through an evaluation server, we perform most evaluations (other than the final accuracy measurement) using the development set.

It contains 1,034 examples, with databases and schemas distinct from those in the training set.

We report results using the same metrics as Yu et al. (2018a) : exact match accuracy on all examples, as well as divided by difficulty levels specified in the dataset.

As in previous work, these metrics do not measure the model's performance on generating values within the queries.

In Table 2a we show accuracy on the (hidden) test set for RAT-SQL and compare to all other approaches that are at or near state-of-the-art (according to the official dataset leaderboard).

RAT-SQL outperforms all other methods that, like RAT-SQL, are not augmented with BERT embeddings.

It even comes within 1.3% of beating the best BERT-augmented model.

Since the typical improvement achieved by BERT augmentation is about 7% for all models, we are hopeful that adding such augmentation to RAT-SQL will also lead to state-of-the-art performance among BERT models.

We also provide a breakdown of the accuracy by difficulty in Table 2b .

As expected, performance drops with increasing difficulty.

The overall generalization gap between development and test was strongly affected by the significant drop in accuracy (15%) on the extra hard questions.

Table 2c shows an ablation study without RAT-based schema linking relations.

Schema linking makes a statistically significant improvement to accuracy (p<0.001).

The full Figure 4 : Alignment between the question "For the cars with 4 cylinders, which model has the largest horsepower" and the database car_1 schema (columns and tables).

model accuracy here differs from Table 2a because the latter shows the best single model from a hyper-parameter sweep (submitted for test evaluation) and the former gives the mean over ten runs.

Alignment Recall from Section 3 that we explicitly represent the alignment between question words and table columns which is used during decoding for column selection.

The existence of the alignment matrix provides a mechanism for the model to align words to columns, but the additional terms in the loss encourage it to actually act like an alignment.

In our final model, the alignment loss terms do not make a difference in overall accuracy.

This is surprising to us because in earlier development, the alignment loss did improve the model (statistically significantly, from 53.0% to 55.4%).

We hypothesize that hyper-parameter tuning that caused us to increase encoding depth also eliminated the need for explicit supervision of alignment.

An accurate alignment representation has other benefits as well, such as identifying question words to copy when a constant is needed (not part of the Spider dataset evaluation).

In Figure 4 we show the alignment generated by our model on an example from the development set.

3 For the three key words that reference columns ("cylinders", "model", "horsepower"), the alignment matrix correctly identifies their corresponding column (cylinders, model, horsepower) and the table (cars_data) except it mistakenly aligns "model" to cars_data also instead of to car_names.

The word "cars" aligns to the primary key of the cars_data table.

Despite the abundance of research in semantic parsing of text to SQL, many contemporary models struggle to learn good representations for a given database schema as well as to properly link column/table references in the question.

These problems are related: to encode & use columns/tables from the schema, the model must reason about their role in the context of a given question.

In this work, we present a unified framework for addressing the schema encoding and linking challenges.

Thanks to relation-aware self-attention, it jointly learns schema and question word representations based on their alignment with each other and predefined schema relations.

Empirically, the RAT framework allows us to gain significant state of the art improvement on textto-SQL parsing.

Qualitatively, it provides a way to combine predefined hard schema relations and inferred soft self-attended relations in the same encoder architecture.

We foresee this joint representation learning being beneficial in many learning tasks beyond text-to-SQL, as long as the input has predefined structure.

A THE NEED FOR SCHEMA LINKING One natural question is how often does the decoder fail to select the correct column, even with the schema encoding and linking improvements we have made.

To answer this, we conducted an oracle experiment (see Table 3 ).

For "oracle sketch", at every grammar nonterminal the decoder is forced to make the correct choice so the final SQL sketch exactly matches that of the correct answer.

The rest of the decoding proceeds as if the decoder had made the choice on its own.

Similarly, "oracle cols" forces the decoder to output the correct column or table at terminal productions.

With both oracles, we see an accuracy of 99.4% which just verifies that our grammar is sufficient to answer nearly every question in the data set.

With just "oracle sketch", the accuracy is only 70.9%, which means 73.5% of the questions that RAT-SQL gets wrong and could get right have incorrect column or table selection.

Similarly, with just "oracle cols", the accuracy is 67.6%, which means that 82.0% of the questions that RAT-SQL gets wrong have incorrect structure.

In other words, most questions have both column and structure wrong, so both problems will continue to be important to work on for the future.

@highlight

State of the art in complex text-to-SQL parsing by combining hard and soft relational reasoning in schema/question encoding.