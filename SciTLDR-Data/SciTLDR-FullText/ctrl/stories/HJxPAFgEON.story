We consider the problem of weakly supervised structured prediction (SP) with reinforcement learning (RL) – for example, given a database table and a question, perform a sequence of computation actions on the table, which generates a response and  receives  a  binary  success-failure  reward.

This  line  of  research  has  been successful by leveraging RL to directly optimizes the desired metrics of the SP tasks – for example, the accuracy in question answering or BLEU score in machine translation.

However, different from the common RL settings, the environment dynamics is deterministic in SP, which hasn’t been fully utilized by the model-freeRL methods that are usually applied.

Since SP models usually have full access to the environment dynamics, we propose to apply model-based RL methods, which rely on planning as a primary model component.

We demonstrate the effectiveness of planning-based SP with a Neural Program Planner (NPP), which, given a set of candidate programs from a pretrained search policy, decides which program is the most promising considering all the information generated from executing these programs.

We evaluate NPP on weakly supervised program synthesis from natural language(semantic parsing) by stacked learning a planning module based on pretrained search policies.

On the WIKITABLEQUESTIONS benchmark, NPP achieves a new state-of-the-art of 47.2% accuracy.

Numerous results from natural language processing tasks have shown that Structured Prediction (SP) can be cast into a reinforcement learning (RL) framework, and known RL techniques can give formal performance bounds on SP tasks BID3 BID13 BID0 .

RL also directly optimizes task metrics, such as, the accuracy in question answering or BLEU score in machine translation, and avoids the exposure bias problem when compaired to maximum likelihood training that is commonly used in SP BID13 BID12 .However, previous works on applying RL to SP problems often use model-free RL algorithms (e.g., REINFORCE or actor-critic) and fail to leverage the characteristics of SP, which are different than typical RL tasks, e.g., playing video games BID9 or the game of Go BID15 .

In most SP problems conditioned on the input x, the environment dynamics, except for the reward signal, is known, deterministic, reversible, and therefore can be searched.

This means that there is a perfect model 1 of the environment, which can be used to apply model-based RL methods that utilize planning 2 as a primary model component.

Take semantic parsing BID1 BID11 as an example, semantic parsers trained by RL such as Neural Semantic Machine (NSM) BID8 typically rely on beam search for inference -the program with the highest probability in beam is used for execution and generating answer.

However, the policy, which is used for beam search, may not be 1 A model of the environment usually means anything that an agent can use to predict how the environment will respond to its actions BID17 .2 planning usually refers to any computational process that takes a model as input and produces or improves a policy for interacting with the modeled environment BID17 able to assign the highest probability to the correct program.

This limitation is due to the policy predicting locally normalized probabilities for each possible action based on the partially generated program, and the probability of a program is a product of these local probabilities.

For example, when applied to the WEBQUESTIONSSP task, NSM made mistakes with two common patterns: (1) the program would ignore important information in the context; (2) the generated program does not execute to a reasonable output, but still receives high probability (spurious programs).

Resolving this issue requires using the information of the full program and its execution output to further evaluate its quality based on the context, which can be seen as planning.

This can be observed in Figure 4 where the model is asked a question "Which programming is played the most?".

The full context of the input table (shown in TAB0 ) contains programming for a television station.

The top program generated by a search policy produces the wrong answer, filtering by a column not relevant to the question.

If provided the correct contextual features, and if allowed to evaluate the full program forward and backward through time, we observe that a planning model would be able to better evaluate which program would produce the correct answer.

To handle errors related to context, we propose to train a value function to compute the utility of each token in a program.

This utility is evaluated by considering the program and token probability as well as the attention mask generated by the sequence-to-sequence (seq2seq) model for the underlying policy.

We also introduce beam and question context with a binary feature representing overlap from question/program and program/program, such as how many programs share a token at a given timestep.

In the experiments, we found that applying a planner that uses a learned value function to re-rank the candidates in the beam can significantly and consistently improve the accuracy.

On the WIKITABLEQUESTIONS benchmark, we improve the state-of-the-art by 0.9%, achieving an accuracy of 47.2%.

2.1 WIKITABLEQUESTIONS WIKITABLEQUESTIONS BID11 contains tables extracted from Wikipedia and question-answer pairs about the tables.

See TAB0 as an example.

There are 2,108 tables and 18,496 question-answer pairs split into train/dev/test set.

Each table can be converted into a directed graph that can be queried, where rows and cells are converted to graph nodes while column names become labeled directed edges.

For the questions, we use string match to identify phrases that appear in the table.

We also identify numbers and dates using the CoreNLP annotation released with the dataset.

The task is challenging in several aspects.

First, the tables are taken from Wikipedia and cover a wide range of topics.

Second, at test time, new tables that contain unseen column names appear.

Third, the table contents are not normalized as in knowledge-bases like Freebase, so there are noises and ambiguities in the table annotation.

Last, the semantics are more complex comparing to previous datasets like WEBQUESTIONSSP BID19 .

It requires multiple-step reasoning using a large set of functions, including comparisons, superlatives, aggregations, and arithmetic operations BID11 .

See BID8 for more details about the functions.

We adopt the NSM framework open sourced in the Memory Augmented Policy Optimization paper (MAPO) BID8 , which combines (1) a neural "programmer", which is a seq2seq model augmented by a key-variable memory that can translate a natural language utterance to a program as a sequence of tokens, and (2) a symbolic "computer", which is an Lisp interpreter that implements a domain specific language with built-in functions and provides code assistance by eliminating syntactically or semantically invalid choices.

For the Lisp interpreter, it added functions according to BID20 BID10 for WIKITABLEQUESTIONS, refer to BID8 for further detail of the open-sourced implementation.

Same as BID8 we consider the problem of contextual program synthesis with weak supervision -i.e., no correct action sequence a is given as part of the training data, and training needs to solve the hard problem of exploring a large program space.

However, we will focus on improving decision making (planning) giving a pretrained search policy, while previous work mainly focus on learning the search policies.

The problem of weakly supervised contextual program synthesis can be formulated as: generating a program a by using a parametric mapping function,â " f θ pxq, where θ denotes the model parameters.

The quality of a generated programâ is measured in terms of a scoring or reward function Rpâ; x, yq with y being the expected correct answer.

The reward function evaluates a program by executing it on a real environment and comparing the emitted output against the correct answer.

For example, the reward may be binary that is 1 when the output equals the answer and 0 otherwise.

We assume that the context x includes both a natural language input and an environment, for example an interpreter or a database, on which the program will be executed.

Given a dataset of context-answer pairs, tpx i , y i qu N i"1 , the goal is to find an optimal parameter θ˚that parameterizes a mapping of x Ñ a with maximum empirical return on a heldout test set.

In this study we will further decompose the policy f θ into the stacking of a search model s φ pxq, which produces a set of candidate programs B given an environment x, and a value model v ω pa; x, Bq, which assigns a score s to program a given the environment and all the candidate programs.

Therefore, θ " rφ; ωs and f θ pxq « argmax DISPLAYFORM0 The search model s φ can be learnt by optimizing a conditional distribution π φ pa | xq that assigns a probability to each program given the context.

That is, π φ is a distribution over the countable set of all possible programs, denoted A. Thus @a P A : π φ pa | xq ě 0 and ř aPA π φ pa | xq " 1.

Then, to synthesize candidate programs for a novel context, one may find the most likely programs under the distribution π φ via exact or approximate inference such as beam search.

B " s φ pxq « argmax B aPA π φ pa | xq .

π φ is typically an autoregressive model such as a recurrent neural network: (e.g. BID5 ) π φ pa | xq " ś |a| i"t π φ pa t | a ăt , xq ,where a ăt " pa 1 , . . .

, a t´1 q denotes a prefix of the program a. In the absence of ground truth programs, policy gradient techniques (such as REINFORCE BID18 ) present a way to optimize the parameters of a stochastic policy π φ via optimization of expected return.

Given a training dataset of context-answer pairs, tpx l , y l qu N l"1 , the training objective is O ER pθq " E a"π φ pa|xq Rpa; x, yq.

Decision-time planning typically relies on value network BID14 trained to predict the true reward.

In the next section, however, we introduce a max-margin training objective for the value model v ω , which optimizes to rank rewarded programs higher than non-rewarded programs.

We now introduce NPP by first describing the architecture of v ω -a seq2seq model which goes over candidate program answer pairs and the final score of a candidate program is simply the sum of its token scores (Section 4.1).

Secondly we describe the program token representation, which considers the program, question and beam context which are used to denote the utility of all tokens within a program (Section 4.2).

Finally, we describe a training procedure that is based on max-margin/ranking objective on candidate programs given a question (Section 4.3).

Here we introduce the NPP architecture FIG0 ) in the context of semantic parsing, but the framework should be broadly applicable to applying RL in structured predictions.

Given a pre-trained search policy π φ and environment x, NPP first unrolls the policy with beam search to generate candidate programs (plans) B " s φ pxq.

Then it scores each program a considering token a t at every step and global statistics among all programs B. a t is represented as a context feature vector C t (details in Section 4.2).

To capture the sequential inputs the scoring component is a seq2seq model which goes over candidate program answer pairs and assigns preference scores to each program/answer token.

We implement a bi-directional recurrent network with LSTM Hochreiter & Schmidhuber (1997b) ; BID4 cells as the first layer of our planner C LSTM " LSTMpCq.

The LSTM hidden state at each step is fed to a one dimensional convolutional layer with kernel size 3, in order to capture inner function scoring as most functions are of size 3-5, as a feature extractor DISPLAYFORM0 Finally we calculate the score per token by feeding into a single node hyperbolic tangent activation layer to compute the score per token v t " TanhpO CNN t q of token a t .

The final score of a candidate Program token probability according to the search policy π φ t agree countNumber of candidate programs having token a t at position t program v ω paq " ř t"1..

T v t is simply the sum of its token scores.

The choice of simply summing token level scores makes the score very understandable (details in Section 4.2).

FIG1 gives implementation details of the value model.

To better score tokens based on the overall context of the environment we represent each token with a set of context features C t " rq tok ; q attn ; p prob ; t prob ; t agree s as in TAB1 .

q attn is the softmax attention across question tokens, which helps to discern which part of the question is of most importance to the model when generating the current given token.

Together q tok and q attn represents the alignment between program and query.

t prob and p prob are the probability of token a t and program a assigned by the search policy π φ , which represent the decisions from the search model.

t agree is the number of candidate programs having token a t at position t. Access to information such as t agree is only available to NPP as it can only be used after all the candidate programs have been generated.

We formulate NPP training as a learning to rank problem -optimizing pairwise ranking among candidate programs B. Given a training dataset of context-answer pairs, tpx l , y l qu N l"1 , the training objective is DISPLAYFORM0 where σpvq " 1{p1`e´vq is the sigmoid function, and v l,i " v ω pa l,i ; x l , s φ px l qq is the estimated value of a l,i , the i-th program generated for context x l .We compare NPP training in two setups: a single MAPO setup, and a stack learning setup.

For the single MAPO setup the queries used to produce a pretrained MAPO model are also used to train the NPP model.

The dev and test queries are used for ablation study and final evaluation.

Since the NPP training queries are already used to train the MAPO model, the candidate programs are biased towards better reward (compared to those candidate generated for unseen queries).

This setup causes NPP to learn from a different distribution as the intended dev/test data.

Surprisingly NPP is still able to improve the prediction of MAPO as will be discussed in Section 5.3.To overcome the issue with single MAPO setup we also generate NPP training data with a stacked learning setup.

First the train and dev queries are merged and splitted into K equal portions, and with Leave-One-Out (LOO) scheme they form K train/dev sets.

Then K MAPO models are trained on K train sets.

Finally we use each of the K MAPO s to produce a stack learning dataset by running these models on their respective dev set.

The stack learning dataset is further splited for train/dev purposes for NPP.

In this way, each NPP training query is decoded by a MAPO model, which has never seen this query before, and therefore avoid the bias in training data generation.

Our empirical study is based on the semantic parsing benchmark, WIKITABLEQUESTIONS BID11 .

To focus on studying the planning part of the problem we assume that the search policy is pretrained using MAPO BID8 , and NPP is trained to rescore given candidate programs produced by MAPO.

Additionally we show that stacked learning BID2 is helpful in correctly training a planner given pre-trained policy models.

Datasets.

WIKITABLEQUESTIONS BID11 contains tables extracted from Wikipedia and question-answer pairs about the tables.

See TAB0 as an example.

There are 2,108 tables and 18,496 question-answer pairs.

We follow the construction in BID11 for converting a table into a directed graph that can be queried, where rows and cells are converted to graph nodes while column names become labeled directed edges.

For the questions, we use string match to identify phrases that appear in the table.

We also identify numbers and dates using the CoreNLP annotation released with the dataset.

Baselines.

We compare NPP to BID8 , the current state of the art on the WIKITABLE-QUESTIONS dataset.

MAPO relies on beam search to find candidate programs, and uses the program probability according to the policy to determine the program to execute.

MAPO manages to achieve 46.3% accuracy on this task when using an ensemble of size 10.

We aim to show that NPP can improve on single MAPO as well as the ensemble of MAPO models.

Training Details.

We set the stacked learning parameter K " 5 for all our experiments.

We set the batch size to be equal to 16.

We use Adam optimizer for training with a learning rate 10´3.

We choose a size of 64 nodes for the LSTM (which becomes 128 as it is bidirectional).

The CNN consists of 32 filters with kernel size 3.

All the hyper parameters are tuned on the development set and trained for 10 epochs.

Ensemble Details.

We formulate the ensemble of K MAPO models with NPP as the sum of normalized NPP scores under different MAPO models: Let Φ " tφ k u K k"1 be K MAPO models to be ensembled.

We define the ensembled score of a program a under context x as v ω,Φ pa; xq " DISPLAYFORM0 wherev ω pxq is the average score for programs in beam s k φ pxq DISPLAYFORM1 and v 1 ω backs-off v ω tov ω pxq whenever a is not in beam DISPLAYFORM2 Table 3: Feature ablation study on the dev set with a mean of 5 runs on a single MAPO setup.

DISPLAYFORM3

In order to evaluate the effectiveness of our proposed programs token representations, we present a feature ablation test in Figure 3 .

We can see that the program probability p prob produced by the search policy is the most important feature, providing the foundation to NPP scoring.

The program agreement feature t agree is also very useful.

We often observe cases for which beam B produces program with similar tokens that are not highly valued by the underlying model.

By utilizing this feature, we more strongly consider programs which were repeatedly searched by s φ .

t agree also helps to identify the programs that are very similar throughout most of the sequence to learn their divergence and grade their utility.

Question referencing features such as q tok provide significant importance in providing the program with query level context, ensuring we are filtering or selecting values based on the query context.

While the help from attention feature q attn is not significant.

We evaluate NPP's impact on MAPO under three different settings, in each of which NPP consistently improves the precision of MAPO.First we consider a single MAPO trained on a single train-dev data split.

Similar to other RL models in the literature, MAPO training is a procedure with big variances.

Even trained on the same data split multiple times with different random seed gives big variances in accuracy of 0.3% (dev) and 0.5% (test).

We use the MAPO model to decode on its own train, dev and test data, in order to generate parallel splits for NPP training.

Training and applying NPP this way is able to improve precision, despite of the exposure bias in the NPP training data.

However, it does not improve on the variances.

We next consider MAPO models that were trained and evaluated on separate train/dev splits created with a Leave-One-Out (LOO) scheme.

As described in Section 4.3 we also use these splits to generate a stacked learning dataset for NPP to avoid the data bias problem.

We can see that with different data splits MAPO has significantly higher variances on the dev set (1.1%), which is a drawback of RL models in general.

Stacked learning helps NPP to improve precision more significantly (1.3% for dev and 1.1% for test).

It also helps to reduce the variances to 0.2% on both dev and test.

Finally, we consider ensembled MAPO settings, which produces the previous state of the art result.

We use the same NPP model trained from the stacked learning setting, and apply it to an ensemble of either 5, or 10 MAPO models from BID8 .

We can see that when applied to 5 MAPO ensemble, NPP can still improve the precision by 1.1%.

When applied to 10 MAPO ensemble, NPP can improves the precision by 0.9%.

Since the score of a program is the sum of its token scores, it is easy to visualize how NPP plan and select the correct program to execute.

We observed that there are two common situations in which MAPO chooses the wrong program from the beam -selecting a spurious program over the semantically correct program and executing the incorrect table filtering or column selection.

NPP aims to overcome these non optimal decisions by taking advantage of the larger time horizon and other programs discovered so far.

For example NPP may reward earlier program tokens based on program tokens chosen much later on due to the bi-directional recurrent network.

We first investigate how NPP demotes spurious programs.

MAPO may produce programs which return the correct answer but are not semantically correct.

NPP helps solve this by scoring semantically correct programs higher in the beam.

An example is shown in Figure 3 for the question "What venue was the latest match played at?" when referring to a soccer (football) player given a table of his matches.

The top program in beam proposed by MAPO was to first filter all rows for the competition string, which is incorrect considering the context of the table is only competitions.

NPP is able to reevaluate the program given the full context.

Although the first function (filter_in) is typically used to filter the table for the correct row/column.

NPP learns that in this situation it is better to find the last of all rows using the function last.

NPP, re-evaluates the first function of the new best program as being high in utility, and scores all tokens within this function higher than the tokens in the incorrect program.

We then investigate programs from MAPO which produce wrong answers.

An example is shown in Figure 4 which is based on TAB0 .

MAPO assigns a higher probability to a program in beam that filters on an incorrect column.

Because NPP knows program token overlap with the query as well as the attention matrix, it is able to better understand the question and grade the program which is more closely related to the question.

In addition to this we notice that the convolutional layer grades full functions within their context, given a kernel of size 3 the parenthesis at the beginning of program already receives a higher NPP score which we interpret as the overall score of executing the function..

Reinforcement learning applied to structured prediction suffers from limited use of the world model as well as not being able to consider future and past program context when generating a sequence.

To overcome these limitations we proposed Neural Program Planner (NPP) which is a planning step after candidate program generation.

We show that an additional planning model can better evaluate overall structure value.

When applied to a difficult SP task NPP improves state of the art by 0.9% and allows intuitive analysis of its scoring model per program token.

A MORE NPP SCORING DETAILS

<|TLDR|>

@highlight

A model-based planning component improves RL-based semantic parsing on WikiTableQuestions.