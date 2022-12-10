The problem of verifying whether a textual hypothesis holds based on the given evidence, also known as fact verification, plays an important role in the study of natural language understanding and semantic representation.

However, existing studies are mainly restricted to dealing with unstructured evidence (e.g., natural language sentences and documents, news, etc), while verification under structured evidence, such as tables, graphs, and databases, remains unexplored.

This paper specifically aims to study the fact verification given semi-structured data as evidence.

To this end, we construct a large-scale dataset called TabFact with 16k Wikipedia tables as the evidence for 118k human-annotated natural language statements, which are labeled as either ENTAILED or REFUTED.

TabFact is challenging since it involves both soft linguistic reasoning and hard symbolic reasoning.

To address these reasoning challenges, we design two different models: Table-BERT and Latent Program Algorithm (LPA).

Table-BERT leverages the state-of-the-art pre-trained language model to encode the linearized tables and statements into continuous vectors for verification.

LPA parses statements into LISP-like programs and executes them against the tables to obtain the returned binary value for verification.

Both methods achieve similar accuracy but still lag far behind human performance.

We also perform a comprehensive analysis to demonstrate great future opportunities.

Verifying whether a textual hypothesis is entailed or refuted by the given evidence is a fundamental problem in natural language understanding (Katz & Fodor, 1963; Van Benthem et al., 2008) .

It can benefit many downstream applications like misinformation detection, fake news detection, etc.

Recently, the first-ever end-to-end fact-checking system has been designed and proposed in Hassan et al. (2017) .

The verification problem has been extensively studied under different natural language tasks such as recognizing textual entailment (RTE) (Dagan et al., 2005) , natural language inference (NLI) (Bowman et al., 2015) , claim verification (Popat et al., 2017; Hanselowski et al., 2018; Thorne et al., 2018) and multimodal language reasoning (NLVR/NLVR2) (Suhr et al., 2017; .

RTE and NLI view a premise sentence as the evidence, claim verification views passage collection like Wikipedia 1 as the evidence, NLVR/NLVR2 views images as the evidence.

These problems have been previously addressed using a variety of techniques including logic rules, knowledge bases, and neural networks.

Recently large-scale pre-trained language models (Devlin et al., 2019; Peters et al., 2018; Yang et al., 2019; Liu et al., 2019) have surged to dominate the other algorithms to approach human performance on several textual entailment tasks (Wang et al., 2018; .

However, existing studies are restricted to dealing with unstructured text as the evidence, which would not generalize to the cases where the evidence has a highly structured format.

Since such structured evidence (graphs, tables, or databases) are also ubiquitous in real-world applications like database systems, dialog systems, commercial management systems, social networks, etc, we argue that the fact verification under structured evidence forms is an equivalently important yet underexplored problem.

Therefore, in this paper, we are specifically interested in studying fact verification with semi-structured Wikipedia tables (Bhagavatula et al., 2013) 2 as evidences owing to its structured and ubiquitous nature (Jauhar et al., 2016; Zhong et al., 2017; Pasupat & Liang, 2015) .

To this end, we introduce a large-scale dataset called TABFACT, which consists of 118K manually annotated statements with regard to 16K Wikipedia tables, their relations are classified as ENTAILED and REFUTED 3 .

The entailed and refuted statements are both annotated by human workers.

With some examples in Figure 1 , we can clearly observe that unlike the previous verification related problems, TABFACT combines two different forms of reasoning in the statements, (i) Linguistic Reasoning: the verification requires semantic-level understanding.

For example, "John J. Mcfall failed to be re-elected though being unopposed." requires understanding over the phrase "lost renomination ..." in the table to correctly classify the entailment relation.

Unlike the existing QA datasets (Zhong et al., 2017; Pasupat & Liang, 2015) , where the linguistic reasoning is dominated by paraphrasing, TABFACT requires more linguistic inference or common sense. (ii) Symbolic Reasoning: the verification requires symbolic execution on the table structure.

For example, the phrase "There are three Democrats incumbents" requires both condition operation (where condition) and arithmetic operation (count).

Unlike question answering, a statement could contain compound facts, all of these facts need to be verified to predict the verdict.

For example, the "There are ..." in Figure 1 requires verifying three QA pairs (total count=5, democratic count=2, republic count=3).

The two forms of reasoning are interleaved across the statements making it challenging for existing models.

In this paper, we particularly propose two approaches to deal with such mixed-reasoning challenge: (i) Table-BERT, this model views the verification task completely as an NLI problem by linearizing a table as a premise sentence p, and applies state-of-the-art language understanding pre-trained model to encode both the table and statements h into distributed representation for classification.

This model excels at linguistic reasoning like paraphrasing and inference but lacks symbolic reasoning skills. (ii) Latent Program Algorithm, this model applies lexical matching to find linked entities and triggers to filter pre-defined APIs (e.g. argmax, argmin, count, etc).

We adopt bread-first-search with memorization to construct the potential program candidates, a discriminator is further utilized to select the most "consistent" latent programs.

This model excels at the symbolic reasoning aspects by executing database queries, which also provides better interpretability by laying out the decision rationale.

We perform extensive experiments to investigate their performances: the best-achieved accuracy of both models are reasonable, but far below human performance.

Thus, we believe that the proposed table-based fact verification task can serve as an important new benchmark towards the goal of building powerful AI that can reason over both soft linguistic form and hard symbolic forms.

To facilitate future research, we released all the data, code with the intermediate results.

First, we follow the previous Table-based Q&A datasets (Pasupat & Liang, 2015; Zhong et al., 2017) to extract web tables (Bhagavatula et al., 2013) with captions from WikiTables 4 .

Here we filter out overly complicated and huge tables (e.g. multirows, multicolumns, latex symbol) and obtain 18K relatively clean tables with less than 50 rows and 10 columns.

For crowd-sourcing jobs, we follow the human subject research protocols 5 to pay Amazon Mechanical Turk 6 workers from the native English-speaking countries "US, GB, NZ, CA, AU" with approval rates higher than 95% and more than 500 accepted HITs.

Following WikiTableQuestion (Pasupat & Liang, 2015) , we provide the annotators with the corresponding table captions to help them better understand the background.

To ensure the annotation quality, we develop a pipeline of "positive two-channel annotation" → "negative statement rewriting" → "verification", as described below.

To harvest statements of different difficulty levels, we design a two-channel collection process: Low-Reward Simple Channel: the workers are paid 0.45 USD for annotating one Human Intelligent Task (HIT) that requires writing five statements.

The workers are encouraged to produce plain statements meeting the requirements: (i) corresponding to a single row/record in the table with unary fact without involving compound logical inference. (ii) mention the cell values without dramatic modification or paraphrasing.

The average annotation time of a HIT is 4.2 min.

High-Reward Complex Channel: the workers are paid 0.75 USD for annotating a HIT (five statements).

They are guided to produce more sophisticated statements to meet the requirements: (i) involving multiple rows in the tables with higher-order semantics like argmax, argmin, count, difference, average, summarize, etc. (ii) rephrase the table records to involve more semantic understanding.

The average annotation time of a HIT is 6.8 min.

The data obtained from the complex channel are harder in terms of both linguistic and symbolic reasoning, the goal of the two-channel split is to help us understand the proposed models can reach under different levels of difficulty.

As suggested in (Zellers et al., 2018) , there might be annotation artifacts and conditional stylistic patterns such as length and word-preference biases, which can allow shallow models (e.g. bag-ofwords) to obtain artificially high performance.

Therefore, we design a negative rewriting strategy to minimize such linguistic cues or patterns.

Instead of letting the annotators write negative statements from scratch, we let them rewrite the collected entailed statements.

During the annotation, the workers are explicitly guided to modify the words, phrases or sentence structures but retain the sentence style/length to prevent artificial cues.

We disallow naive negations by adding "not, never, etc" to revert the statement polarity in case of obvious linguistic patterns.

To control the quality of the annotation process, we review a randomly sampled statement from each HIT to decide whether the whole annotation job should be rejected during the annotation process.

Specifically, a HIT must satisfy the following criteria to be accepted: (i) the statements should contain neither typos nor grammatical errors. (ii) the statements do not contain vague claims like might, few, etc. (iii) the claims should be explicitly supported or contradicted by the table without requiring additional knowledge, no middle ground is permitted.

After the data collection, we redistribute all the annotated samples to further filter erroneous statements, the workers are paid 0.05 USD per statement to decide whether the statement should be rejected.

The criteria we apply are similar: no ambiguity, no typos, explicitly supported or contradictory.

Through the post-filtering process, roughly 18% entailed and 27% refuted instances are further abandoned due to poor quality.

Table 1 : Basic statistics of the data collected from the simple/complex channel and the division of Train/Val/Test Split in the dataset, where "Len" denotes the averaged sentence length.

Inter-Annotator Agreement: After the data collection pipeline, we merged the instances from two different channels to obtain a diverse yet clean dataset for table-based fact verification.

We sample 1000 annotated (table, statement) pairs and re-distribute each to 5 individual workers to re-label them as either ENTAILED or REFUTED.

We follow the previous works (Thorne et al., 2018; Bowman et al., 2015) to adopt the Fleiss Kappa (Fleiss, 1971) as an indicator, where Fleiss κ =p c −pe 1−pe is computed from from the observed agreementp c and the agreement by chancep e .

We obtain a Fleiss κ = 0.75, which indicates strong inter-annotator agreement and good-quality.

Dataset Statistics: As shown in Table 1 , the amount of data harvested via the complex channel slightly outnumbers the simple channel, the averaged length of both the positive and negative samples are indistinguishable.

More specifically, to analyze to which extent the higher-order operations are included in two channels, we group the common higher-order operations into 8 different categories.

As shown in Figure 2 , we sample 200 sentences from two different channels to visualize their distribution.

We can see that the complex channel overwhelms the simple channel in terms of the higher-order logic, among which, count and superlatives are the most frequent.

We split the whole data roughly with 8:1:1 into train, validation 7 , and test splits and shows their statistics in Table 1 .

Each table with an average of 14 rows and 5-6 columns corresponds to 2-20 different statements, while each cell has an average of 2.1 words.

In the training split, the positive instances slightly outnumber the negative instances, while the validation and test split both have rather balanced distributions over positive and negative instances.

With the collected dataset, we now formally define the table-based fact verification task: the dataset is comprised of triple instances (T, S, L) consisting of a table T, a natural language statement S = s 1 , · · · , s n and a verification label L ∈ {0, 1}. The table T = {T i,j |i ≤ R T , j ≤ C T } has R T rows and C T columns with the T ij being the content in the (i, j)-th cell.

T ij could be a word, a number, a phrase or even a natural language sentence.

The statement S describes a fact to be verified against the content in the table T. If it is entailed by T, then L = 1, otherwise the label L = 0.

Figure 1 shows some entailed and refuted examples.

During training, the model and the learning algorithm are presented with K instances like (T, S, L) K k=1 from the training split.

In the testing stage, the model is presented with (T, S) K k=1 and supposed to predict the label asL. We measure the performance by the prediction accuracy Acc =

Before building the model, we first perform entity linking to detect all the entities in the statements.

Briefly, we first lemmatize the words and search for the longest sub-string matching pairs between statements and table cells/captions, where the matched phrases are denoted as the linked entities.

To focus on statement verification against the table, we do not feed the caption to the model and simply mask the phrases in the statements which links to the caption with placeholders.

The details of the entity linker are listed in the Appendix.

We describe our two proposed models as follows.

In this approach, we formulate the table fact verification as a program synthesis problem, where the latent program algorithm is not given in TABFACT.

Thus, it can be seen as a weakly supervised learning problem as discussed in Liang et al. (2017); Lao et al. (2011) .

Under such a setting, we propose to break down the verification into two stages: (i) latent program search, (ii) discriminator ranking.

In the first program synthesis step, we aim to parse the statement into programs to represent its semantics.

We define the plausible API set to include roughly 50 different functions like min, max, count, average, filter, and and realize their interpreter with Python-Pandas.

Each API is defined to take arguments of specific types (number, string, bool and view (e.g sub-table)) to output specifictype variables.

During the program execution, we store the generated intermediate variables to different-typed caches N , R, B, V (Num, Str, Bool, View).

At each execution step, the program can fetch the intermediate variable from the caches to achieve semantic compositionality.

In order to shrink the search space, we follow NSM (Liang et al., 2017) to use trigger words to prune the API set and accelerate the search speed.

The definitions of all API, trigger words can be found in the Appendix.

The comprehensive the latent program search procedure is summarized in Algorithm 1, Algorithm 1 Latent Program Search with Comments 1: Initialize Number Cache N , String Cache R, Bool Cache B, View Cache V → ∅ 2: Push linked numbers, strings from the given statement S into N , R, and push T into V 3: Initialize the result collector P → ∅ and an empty program trace P = ∅ 4: Initialize the Queue Q = [(P, N , R, B, V)], we use Q to store the intermediate states 5: Use trigger words to find plausible function set F, for example, more will trigger Greater function.

6: while loop over time t = 1 → MAXSTEP do: 7:

while loop over function set f ∈ F do: 9:

if arguments of f are in the caches then 10:

Pop out the required arguments arg1, arg2, · · · , argn for different cachess.

Execute A = f (arg1, · · · , argn) and concatenate the program trace P .

if Type(A)=Bool then 13:

if N = S = B = ∅ then 14:

P.push((P, A))

# The program P is valid since it consumes all the variables.

15: P = ∅ # Collect the valid program P into set P and reset P 16: else 17:

B.push(A) # The intermediate boolean value is added to the bool cache 18:

Q.push((P, N , R, B, V))

# Add the refreshed state to the queue again 19:

if Type(A) ∈ {Num, Str, View} then 20:

if N = S = B = ∅ then 21: P = ∅;break # The program ends without consuming the cache, throw it.

22: else 23: push A into N or S or V # Add the refreshed state to the queue for further search 24:

Q.push((P, N , R, B, V)) 25: Return the triple (T, S, P) # Return (Table, Statement, Program Set) and the searching procedure is illustrated in Figure 3 .

After we collected all the potential program candidates P = {(P 1 , A 1 ), · · · , (P n , A n )} for a given statement S (where (P i , A i ) refers to i-th candidate), we need to learn a discriminator to identify the "appropriate" traces from the set from many erroneous and spurious traces.

Since we do not have the ground truth label about such discriminator, we use a weakly supervised training algorithm by viewing all the label-consistent programs as positive instances {P i |(P i , A i ); A i = L} and the label-inconsistent program as negative instances {P i |(P i , A i ); A i = L} to minimize the cross-entropy of discriminator p θ (S, P ) with the weakly supervised label.

Specifically, we build our discriminator with a Transformer-based two-way encoder (Vaswani et al., 2017) , where the statement encoder encodes the input statement S as a vector Enc S (S) ∈ R n×D with dimension D, while the program encoder encodes the program P = p 1 , · · · , p m as another vector Enc P (P ) ∈ R m×D , we concatenate these two vectors and feed it into a linear projection layer

There are more democrats than republicans in the election.

V1=Filter(T, incumbent==democratic))

Figure 4: The diagram of Table- BERT with horizontal scan, two different linearizations are depicted.

as the relevance between S and P with weight

At test time, we use the discriminator p θ to assign confidence p θ (S, P ) to each candidate P ∈ P, and then either aggregate the prediction from all hypothesis with the confidence weights or rank the highest-confident hypothesis and use their outputs as the prediction.

In this approach, we view the table verification problem as a two-sequence binary classification problem like NLI or MPRC (Wang et al., 2018) by linearizing a table T into a sequence and treating the statement as another sequence.

Since the linearized table can be extremely long surpassing the limit of sequence models like LSTM, Transformers, etc.

We propose to shrink the sequence by only retaining the columns containing entities linked to the statement to alleviate such a memory issue.

In order to encode such sub-table as a sequence, we propose two different linearization methods, as is depicted in Figure 4 .

(i) Concatenation: we simply concatenate the table cells with [SEP] tokens in between and restart position counter at the cell boundaries; the column name is fed as another type embedding to the input layer.

Such design retains the table information in its machine format.

(ii) Template: we adopt simple natural language templates to transform a table into a "somewhat natural" sentence.

Taking the horizontal scan as an example, we linearize a table as "row one's game is 51; the date is February; ..., the score is 3.4 (ot).

row 2 is ...".

The isolated cells are connected with punctuations and copula verbs in a language-like format.

After obtaining the linearized sub-tableT, we concatenate it with the natural language statement S and prefix a [CLS] token to the sentence to obtain the sequence-level representation

768 from pre-trained BERT (Devlin et al., 2019) .

The representation is further fed into multi-layer perceptron f M LP to obtain the entailment probability p θ (T, S) = σ(f M LP (H)), where σ is the sigmoid function.

We finetune the model θ (including the parameters of BERT and MLP) to minimize the binary cross entropy L(p θ (T, S), L) on the training set.

At test time, we use the trained BERT model to compute the matching probability between the (table, statement) pair, and classify it as ENTAILED statement when p θ (T, S) is greater than 0.5.

In this section, we aim to evaluate the proposed methods on TABFACT.

Besides the standard validation and test sets, we also split the test set into a simple and a complex partition based on the channel from which they were collected.

This facilitates analyzing how well the model performs under different levels of difficulty.

Additionally, we also hold out a small test set with 2K samples for human evaluation, where we distribute each (table, statement) pair to 5 different workers to approximate human judgments based on their majority voting, the results are reported in Table 2 NSM We follow Liang et al. (2017) to modify their approach to fit the setting of TABFACT.

Specifically, we adopt an LSTM as an encoder and another LSTM with copy mechanism as a decoder to synthesize the program.

However, without any ground truth annotation for the intermediate programs, directly training with reinforcement learning is difficult as the binary reward is underspecified, which is listed in Table 2 as "NSM w/ RL".

Further, we use LPA as a teacher to search the top programs for the NSM to bootstrap and then use reinforcement learning to finetune the model, which achieves reasonable performance on our dataset listed as "NSM w/ ML + RL".

Table- BERT We build Table- BERT based on the open-source implementation of BERT 8 using the pre-trained model with 12-layer, 768-hidden, 12-heads, and 110M parameters trained in 104 languages.

We use the standard BERT tokenizer to break the words in both statements and tables into subwords and join the two sequences with a [SEP] token in between.

The representation corresponding to [CLS] is fed into an MLP layer to predict the verification label.

We finetune the model on a single TITAN X GPU with a mini-batch size of 6.

The best performance is reached after about 3 hours of training (around 10K steps).

We implement and compare the following variants of the Table- BERT model including (i) Concatenation vs. Template: whether to use natural language templates during linearization. (ii) Horizontal vs. Vertical: scan direction in linearization.

LPA We run the latent program search in a distributed fashion on three 64-core machines to generate the latent programs.

The search terminates once the buffer has more than 50 traces or the path length is larger than 7.

The average search time for each statement is about 2.5s.

For the discriminator model, we design two transformer-based encoders (3 layers, 128-dimension hidden embedding, and 4 heads at each layer) to encode the programs and statements, respectively.

The variants of LPA models considered include (i) Voting: assign each program with equal weight and vote without the learned discriminator. (ii) Weighted-Voting: compute a weighted-sum to aggregate the predictions of all latent programs with the discriminator confidence as the weights. (iii) Ranking: rank all the hypotheses by the discriminator confidence and use the top-rated hypothesis as the output. (Caption) means feeding the caption as a sequence of words to the discriminator during ranking.

Preliminary Evaluation In order to test whether our negative rewriting strategy eliminates the artifacts or shallow cues, we also fine-tune a pre-trained BERT (Devlin et al., 2019) to classify the statement S without feeding in table information.

The result is reported as "BERT classifier w/o Table" in Table 2 , which is approximately the majority guess and reflects the effectiveness of the rewriting strategy.

Before presenting the experiment results, we first perform a preliminary study to evaluate how well the entity linking system, program search, and the statement-program discriminator perform.

Since we do not have the ground truth labels for these models, we randomly sample 100 samples from the dev set to perform the human study.

For the entity linking, we evaluate the precision of correctly linked entities and the recall of entities that should be linked.

For the latent program search, we evaluate whether the "true" programs are included in the candidate set P and report the recall score.

For discriminator, under the cases where the "true" program lies in the candidate set, we use the trained model to select the top K hypothesis and calculate the HITS@K accuracy (the chance of correct program being included in the top K candidates).

Please note that the discriminator can also select a spurious program that happens to obtain the same label as ground truth, but this does not count as a hit.

These preliminary case study results are reported in Table 3 Table 3 : Case Study results on different components, including the entity linking accuracy, systematic search recall, and discriminator accuracy.

Results We report the performance of different methods as well as human performance in Table 2 .

First of all, we observe that the naive serialized model fails to learn anything effective (same as the Majority Guess).

It reveals the importance of template when using the pre-trained BERT (Devlin et al., 2019) model: the "natural" connection words between individual cells is able to unleash the power of the large pre-trained language model and enable it to perform reasoning on the structured table form.

Such behavior is understandable given the fact that BERT is pre-trained on purely natural language corpora.

In addition, we also observe that the horizontal scan excels the vertical scan because it better captures the convention of human expression.

Among different LPA methods, we found that LPA-Ranking performs the best since it can better suppress the spurious programs than the voting-based algorithm.

As suggested in Table 3 , the current LPA method is upper bounded by 77% (recall of "true" program hypothesis), but the real accuracy (65%) is still far from that.

Diving into specific cases to examine the performance of discriminator, we found that only 17% "true" programs are ranked at the top Table 3 .

We hypothesize that the weakly supervised learning of the discriminator is the main bottleneck for LPA.

By comparing the performance of simple-channel with complex-channel split, we observe a significant accuracy drop (≈ 20%), which reveals the weakness of existing models in dealing with higher-ordered semantics.

Besides, we observe that Table- BERT exhibits instability during training, after the model achieves the reported ceiling performance, its performance will degrade gradually.

Additionally, it also exhibits poor consistency as it can miss some simple cases but hit hard cases.

These two major weaknesses are yet to be solved in the future study.

In contrast, LPA behaves much more consistently and provides a clear latent rationale for its decision.

But, such a pipeline system requires laborious handcrafting of API operations and is also very sensitive to the entity linking accuracy.

Both methods have pros and cons; how to combine them still remains an open question.

Natural Language Inference & Reasoning: Modeling reasoning and inference in human language is a fundamental and challenging problem towards true natural language understanding.

There has been extensive research on RTE in the early years (Dagan et al., 2005) and more recently shifted to NLI (Bowman et al., 2015; Williams et al., 2017) .

NLI seeks to determine whether a natural language hypothesis h can be inferred from a natural language premise p.

With the surge of deep learning, there have been many powerful algorithms like the Decomposed Model (Parikh et al., 2016) , Enhanced-LSTM (Chen et al., 2017) and BERT (Devlin et al., 2019) .

Besides the textual evidence, NLVR (Suhr et al., 2017) and NLVR2 (Suhr et al., 2019) have been proposed to use images as the evidence for statement verification on multi-modal setting.

Our proposed fact verification task is closely related to these inference tasks, where our semi-structured table can be seen as a collection of "premises" exhibited in a semi-structured format.

Our proposed problem hence could be viewed

(1) Figure 5 : The two uniqueness of Table- based fact verification against standard QA problems.

as the generalization of NLI under the semi-structured domain. , 2013) .

However, in these Q&A tasks, the question types typically provide strong signals needed for identifying the type of answers, while TABFACT does not provide such specificity.

The uniqueness of TABFACT lies in two folds: 1) a given fact is regarded as a false claim as long as any part of the statement contains misinformation.

Due to the conjunctive nature of verification, a fact needs to be broken down into several sub-clauses or (Q, A) pairs to separate evaluate their correctness.

Such a compositional nature of verification problem makes it more challenging than standard QA setting.

On one hand, the model needs to recognize the multiple QA pairs and their relationship.

On the other hand, the multiple sub-clauses make the semantic form longer and logic inference harder than the standard QA setting.

2) some facts cannot even be handled using semantic forms, as they are driven by linguistic inference or common sense.

In order to verify these statements, more inference techniques have to be leveraged to enable robust verification.

We visualize the above two characteristics of TABFACT in Figure 5 . , 2017; 2018; Agarwal et al., 2019) .

The proposed TABFACT serves as a great benchmark to evaluate the reasoning ability of different neural reasoning models.

Specifically, TABFACT poses the followng challenges: 1) spurious programs (i.e., wrong programs with the true returned answers): since the program output is only a binary label, which can cause serious spurious problems and misguide the reinforcement learning with the under-specified binary rewards.

2) decomposition: the model needs to decompose the statement into sub-clauses and verify the sub-clauses one by one, which normally requires the longer logic inference chains to infer the statement verdict.

3) linguistic reasoning like inference and paraphrasing.

This paper investigates a very important yet previously under-explored research problem: semistructured fact verification.

We construct a large-scale dataset and proposed two methods, Table- BERT and LPA, based on the state-of-the-art pre-trained natural language inference model and program synthesis.

In the future, we plan to push forward this research direction by inspiring more sophisticated architectures which can perform both linguistic and symbolic reasoning.

We list all the trigger words for different functions in Figure 8 Trigger Function 'average' average 'difference ', 'gap', 'than', 'separate' diff 'sum', 'summation', 'combine', 'combined', 'total', 'add', 'all', 'there are' ddd, sum 'not', 'no', 'never', "didn't", "won't", "wasn't", "isn't,"haven't", "weren't", "won't", 'neither', 'none', 'unable, 'fail', 'different', 'outside', 'unable', 'fail' not_eq, not_within, Filter_not_eq, none 'not', 'no', 'none' none 'first', 'top', 'latest', 'most' first 'last', 'bottom', 'latest', 'most' last 'RBR', 'JJR', 'more', 'than', 'above', 'after' filter_greater, greater 'RBR', 'JJR', 'less', 'than', 'below', 'under' filter_less, less 'all', 'every', 'each' all_eq, all_less, all_greater, ['all', 'every', 'each'] , ['not', 'no', 'never', "didn't", "won't", "wasn't"] 2.

Negation: the negation operation refers to sentences like "xxx did not get the best score", "xxx has never obtained a score higher than 5".

3.

Superlative: the superlative operation refers to sentences like "xxx achieves the highest score in", "xxx is the lowest player in the team".

4.

Comparative: the comparative operation refers to sentences like "xxx has a higher score than yyy".

Ordinal: the ordinal operation refers to sentences like "the first country to achieve xxx is xxx", "xxx is the second oldest person in the country".

6.

Unique: the unique operation refers to sentences like "there are 5 different nations in the tournament, ", "there are no two different players from U.S" 7.

All: the for all operation refers to sentences like "all of the trains are departing in the morning", "none of the people are older than 25." 8.

None: the sentences which do not involve higher-order operations like "xxx achieves 2 points in xxx game", "xxx player is from xxx country".

Before we quantitatively demonstrate the error analysis of the two methods, we first theoretically analyze the bottlenecks of the two methods as follows:

Symbolic We first provide a case in which the symbolic execution can not deal with theoretically in Figure 9 .

The failure cases of symbolic are either due to the entity link problem or function coverage problem.

For example, in the given statement below, there is no explicit mention of "7-5, 6-4" cell.

Therefore, the entity linking model fails to link to this cell content.

Furthermore, even though we can successfully link to this string, there is no defined function to parse "7-5, 6-5" as "won two games" because it requires linguistic/mathematical inference to understand the implication from the string.

Such cases are the weakness of symbolic reasoning models.

Jordi Arrese achieves better score in 1986 than in 1985.

Jordi Arrese won both of the final games in 1986.

Table- BERT model seems to have no coverage problem as long as it can feed the whole table content.

However, due to the template linearization, the table is unfolded into a long sequence as depicted in Figure 10 .

The useful information, "clay" are separated in a very long span of unrelated words.

How to grasp such a long dependency and memorize the history information poses a great challenge to the Table-

Jordi Arrese played all of his games on clay surface.

Given the table titled "Jordi Arrese", in row one, the outcome is runner-up, the date is 1985, … , the surface is clay ….

…

… , In row two, the outcome is … , the surface is clay.

In row three, the outcome is …, … the surface is clay.

The three "Clay" are separated by more over 20 words Statistics Here we pick 200 samples from the validation set which only involve single semantic and divide them into different categories.

We denote the above-mentioned cases as "linguistic inference", and the sentences which only describe information from one row as "Trivial", the rest are based on their logic operation like Aggregation, Superlative, Count, etc.

We visualize the accuracy of LPA and Table-BERT in Figure 11 .

From which we can observe that the statements with linguistic inference are much better handled with BERT model, while LPA achieves an accuracy barely higher than random guess.

The BERT model can deal with trivial cases well as it uses a horizontal scan order.

In contrast, the LPA model outperforms BERT on higher-order logic cases, especially when the statement involves operations like Count and Superlative.

Error Analysis of LPA/ Table-BERT   Table- BERT LPA Figure 11 : The error analysis of two different models D REASONING DEPTH Given that our LPA has the breadth to cover a large semantic space.

Here we also show the reasoning depth in terms of how many logic inference steps are required to tackle verify the given claims.

We visualize the histogram in Figure 12 and observe that the reasoning steps are concentrated between 4 to 7.

Such statistics indicate the difficulty of fact verification in our TABFACT dataset. (2017) does not.

Therefore, we particularly design ablation annotation tasks to compare the annotation quality between w/ and w/o Wikipedia title as context.

We demonstrate a typical example in Figure 13 , where a Wiki table 9 aims to describe the achievements of a tennis player named Dennis, but itself does not provide any explicit hint about "Tennis Player Dennis".

Unsurprisingly, the sentence fluency and coherence significantly drops without such information.

Actually, a great portion of these Wikipedia tables requires background knowledge (like sports, celebrity, music, etc) to understand.

We perform a small user study to measure the fluency of annotated statements.

Specifically, we collected 50 sentences from both annotation w/ and w/o title context and randomly shuffle them as pairs, which are distributed to the 8 experts without telling them their source to compare the language fluency.

It turns out that the experts ubiquitously agree that the statements with Wikipedia titles are more human-readable.

Therefore, we argue that such a context is necessary for annotators to understand the background knowledge to write more fluent sentences.

On the other end, we also hope to minimize the influence of the textual context in the table-based verification task, therefore, we design an annotation criterion: the Wikipedia title is provided to the workers during the annotation, but they are explicitly banned from bring any unrelated background information other than the title into the annotation.

As illustrated in Figure 13 , the title only acts as a placeholder in the statements to make it sound more natural.

Richard Dennis Ralston (born July 27, 1942, an In order to harvest fake statements without statistical cues, we also provide detailed instructions on how to re-write the "fake" statements.

During the annotation, we hire 8 experts to perform sanity checks on each of the HIT to make sure that the annotated dataset is clean and meets our requirements.

You are given a table with its wikipedia source, your job is to compose non-trivial statements supported by the table.

-"Trivial": the sentence can be easily generated by looking only a certain row without understanding the Rejected ("Trivial") examples:

1.

In the TV series "The Island", Derrick Kosinski is a male character.

(Easy!

You can simply look into first row to produce this sentence.)

2.

Derrick Kosinski has the placing of winner in the TV series.

3.

Kenny Santucci is from original season of "Fresh Meat".

https://s3.amazonaws.com/mturk_bulk/hits/370501562/uNGk1Dz1zM48BZI6mALfxA.html 3/4

First Read the following table, then write five diverse non-trivial facts for this given table: Please write a non-trivial statement, minimum 9 words Please write a non-trivial statement, minimum 9 words

Please write a non-trivial statement, minimum 9 words

Please write a non-trivial statement, minimum 9 words

Please write a non-trivial statement, minimum 9 words

@highlight

We propose a new dataset to investigate the entailment problem under semi-structured table as premise

@highlight

This paper proposes a new dataset for table-based fact verification and introduces methods for the task.

@highlight

The authors propose the problem of fact verification with semi-structured data sources such as tables, create a new dataset, and evaluate baseline models with variations.