Recurrent neural networks (RNNs) sequentially process data by updating their state with each new data point, and have long been the de facto choice for sequence modeling tasks.

However, their inherently sequential computation makes them slow to train.

Feed-forward and convolutional architectures have recently been shown to achieve superior results on some sequence modeling tasks such as machine translation, with the added advantage that they concurrently process all inputs in the sequence, leading to easy parallelization and faster training times.

Despite these successes, however, popular feed-forward sequence models like the Transformer fail to generalize in many simple tasks that recurrent models handle with ease, e.g. copying strings or even simple logical inference when the string or formula lengths exceed those observed at training time.

We propose the Universal Transformer (UT), a parallel-in-time self-attentive recurrent sequence model which can be cast as a generalization of the Transformer model and which addresses these issues.

UTs combine the parallelizability and global receptive field of feed-forward sequence models like the Transformer with the recurrent inductive bias of RNNs.

We also add a dynamic per-position halting mechanism and find that it improves accuracy on several tasks.

In contrast to the standard Transformer, under certain assumptions UTs can be shown to be Turing-complete.

Our experiments show that UTs outperform standard Transformers on a wide range of algorithmic and language understanding tasks, including the challenging LAMBADA language modeling task where UTs achieve a new state of the art, and machine translation where UTs achieve a 0.9 BLEU improvement over Transformers on the WMT14 En-De dataset.

Convolutional and fully-attentional feed-forward architectures like the Transformer have recently emerged as viable alternatives to recurrent neural networks (RNNs) for a range of sequence modeling tasks, notably machine translation BID8 Vaswani et al., 2017) .

These parallel-in-time architectures address a significant shortcoming of RNNs, namely their inherently sequential computation which prevents parallelization across elements of the input sequence, whilst still addressing the vanishing gradients problem as the sequence length gets longer BID15 .

The Transformer model in particular relies entirely on a self-attention mechanism BID23 BID20 to compute a series of context-informed vector-space representations of the symbols in its input and output, which are then used to predict distributions over subsequent symbols as the model predicts the output sequence symbol-by-symbol.

Not only is this mechanism straightforward to parallelize, but as each symbol's representation is also directly informed by all other symbols' representations, this results in an effectively global receptive field across the whole sequence.

This stands in contrast to e.g. convolutional architectures which typically only have a limited receptive field.

Notably, however, the Transformer with its fixed stack of distinct layers foregoes RNNs' inductive bias towards learning iterative or recursive transformations.

Our experiments indicate that this inductive The Universal Transformer repeatedly refines a series of vector representations for each position of the sequence in parallel, by combining information from different positions using self-attention (see Eqn 2) and applying a recurrent transition function (see Eqn 4) across all time steps 1 ??? t ??? T .

We show this process over two recurrent time-steps.

Arrows denote dependencies between operations.

Initially, h 0 is initialized with the embedding for each symbol in the sequence.

h t i represents the representation for input symbol 1 ??? i ??? m at recurrent time-step t. With dynamic halting, T is dynamically determined for each position (Section 2.2).bias may be crucial for several algorithmic and language understanding tasks of varying complexity: in contrast to models such as the Neural Turing Machine BID12 , the Neural GPU BID17 or Stack RNNs , the Transformer does not generalize well to input lengths not encountered during training.

In this paper, we introduce the Universal Transformer (UT), a parallel-in-time recurrent self-attentive sequence model which can be cast as a generalization of the Transformer model, yielding increased theoretical capabilities and improved results on a wide range of challenging sequence-to-sequence tasks.

UTs combine the parallelizability and global receptive field of feed-forward sequence models like the Transformer with the recurrent inductive bias of RNNs, which seems to be better suited to a range of algorithmic and natural language understanding sequence-to-sequence problems.

As the name implies, and in contrast to the standard Transformer, under certain assumptions UTs can be shown to be Turing-complete (or "computationally universal", as shown in Section 4).In each recurrent step, the Universal Transformer iteratively refines its representations for all symbols in the sequence in parallel using a self-attention mechanism BID23 BID20 , followed by a transformation (shared across all positions and time-steps) consisting of a depth-wise separable convolution BID4 BID18 or a position-wise fully-connected layer (see FIG0 .

We also add a dynamic per-position halting mechanism BID11 , allowing the model to choose the required number of refinement steps for each symbol dynamically, and show for the first time that such a conditional computation mechanism can in fact improve accuracy on several smaller, structured algorithmic and linguistic inference tasks (although it marginally degraded results on MT).Our strong experimental results show that UTs outperform Transformers and LSTMs across a wide range of tasks.

The added recurrence yields improved results in machine translation where UTs outperform the standard Transformer.

In experiments on several algorithmic tasks and the bAbI language understanding task, UTs also consistently and significantly improve over LSTMs and the standard Transformer.

Furthermore, on the challenging LAMBADA text understanding data set UTs with dynamic halting achieve a new state of the art.

The Universal Transformer (UT; see FIG1 ) is based on the popular encoder-decoder architecture commonly used in most neural sequence-to-sequence models Vaswani et al., 2017) .

Both the encoder and decoder of the UT operate by applying a recurrent neural network to the representations of each of the positions of the input and output sequence, respectively.

However, in contrast to most applications of recurrent neural networks to sequential data, the UT does not recur over positions in the sequence, but over consecutive revisions of the vector representations of each position (i.e., over "depth").

In other words, the UT is not computationally bound by the number of symbols in the sequence, but only by the number of revisions made to each symbol's representation.

In each recurrent time-step, the representation of every position is concurrently (in parallel) revised in two sub-steps: first, using a self-attention mechanism to exchange information across all positions in the sequence, thereby generating a vector representation for each position that is informed by the representations of all other positions at the previous time-step.

Then, by applying a transition function (shared across position and time) to the outputs of the self-attention mechanism, independently at each position.

As the recurrent transition function can be applied any number of times, this implies that UTs can have variable depth (number of per-symbol processing steps).

Crucially, this is in contrast to most popular neural sequence models, including the Transformer (Vaswani et al., 2017) or deep RNNs, which have constant depth as a result of applying a fixed stack of layers.

We now describe the encoder and decoder in more detail.

Given an input sequence of length m, we start with a matrix whose rows are initialized as the d-dimensional embeddings of the symbols at each position of the sequence H 0 ??? R m??d .

The UT then iteratively computes representations H t at step t for all m positions in parallel by applying the multi-headed dot-product self-attention mechanism from Vaswani et al. (2017) , followed by a recurrent transition function.

We also add residual connections around each of these function blocks and apply dropout and layer normalization BID26 BID1 ) (see FIG1 for a simplified diagram, and FIG5 in the Appendix A for the complete model.).More specifically, we use the scaled dot-product attention which combines queries Q, keys K and values V as follows DISPLAYFORM0 where d is the number of columns of Q, K and V .

We use the multi-head version with k heads, as introduced in (Vaswani et al., 2017) , DISPLAYFORM1 where DISPLAYFORM2 and we map the state H t to queries, keys and values with affine projections using learned parameter matrices DISPLAYFORM3 At step t, the UT then computes revised representations H t ??? R m??d for all m input positions as follows DISPLAYFORM4 where DISPLAYFORM5 where LAYERNORM() is defined in BID1 , and TRANSITION() and P t are discussed below.

Depending on the task, we use one of two different transition functions: either a separable convolution BID4 or a fully-connected neural network that consists of a single rectified-linear activation function between two affine transformations, applied position-wise, i.e. individually to each row of A t .

P t ??? R m??d above are fixed, constant, two-dimensional (position, time) coordinate embeddings, obtained by computing the sinusoidal position embedding vectors as defined in (Vaswani et al., 2017) for the positions 1 ??? i ??? m and the time-step 1 ??? t ??? T separately for each vector-dimension 1 ??? j ??? d, and summing: A complete version can be found in Appendix A. The Universal Transformer with dynamic halting determines the number of steps T for each position individually using ACT BID11 .

DISPLAYFORM6 DISPLAYFORM7 After T steps (each updating all positions of the input sequence in parallel), the final output of the Universal Transformer encoder is a matrix of d-dimensional vector representations H T ??? R m??d for the m symbols of the input sequence.

The decoder shares the same basic recurrent structure of the encoder.

However, after the self-attention function, the decoder additionally also attends to the final encoder representation H T of each position in the input sequence using the same multihead dot-product attention function from Equation 2, but with queries Q obtained from projecting the decoder representations, and keys and values (K and V ) obtained from projecting the encoder representations (this process is akin to standard attention BID2 ).Like the Transformer model, the UT is autoregressive BID10 .

Trained using teacher-forcing, at generation time it produces its output one symbol at a time, with the decoder consuming the previously produced output positions.

During training, the decoder input is the target output, shifted to the right by one position.

The decoder self-attention distributions are further masked so that the model can only attend to positions to the left of any predicted symbol.

Finally, the per-symbol target distributions are obtained by applying an affine transformation O ??? R d??V from the final decoder state to the output vocabulary size V , followed by a softmax which yields an (m??V )-dimensional output matrix normalized over its rows: DISPLAYFORM0 To generate from the model, the encoder is run once for the conditioning input sequence.

Then the decoder is run repeatedly, consuming all already-generated symbols, while generating one additional distribution over the vocabulary for the symbol at the next output position per iteration.

We then typically sample or select the highest probability symbol as the next symbol.

In sequence processing systems, certain symbols (e.g. some words or phonemes) are usually more ambiguous than others.

It is therefore reasonable to allocate more processing resources to these more ambiguous symbols.

Adaptive Computation Time (ACT) BID11 ) is a mechanism for dynamically modulating the number of computational steps needed to process each input symbol 1 Note that T here denotes time-step T and not the transpose operation.

Table 1 : Average error and number of failed tasks (> 5% error) out of 20 (in parentheses; lower is better in both cases) on the bAbI dataset under the different training/evaluation setups.

We indicate state-of-the-art where available for each, or '-' otherwise.(called the "ponder time") in standard recurrent neural networks based on a scalar halting probability predicted by the model at each step.

Inspired by the interpretation of Universal Transformers as applying self-attentive RNNs in parallel to all positions in the sequence, we also add a dynamic ACT halting mechanism to each position (i.e. to each per-symbol self-attentive RNN; see Appendix C for more details).

Once the per-symbol recurrent block halts, its state is simply copied to the next step until all blocks halt, or we reach a maximum number of steps.

The final output of the encoder is then the final layer of representations produced in this way.

We evaluated the Universal Transformer on a range of algorithmic and language understanding tasks, as well as on machine translation.

We describe these tasks and datasets in more detail in Appendix D.

The bAbi question answering dataset consists of 20 different tasks, where the goal is to answer a question given a number of English sentences that encode potentially multiple supporting facts.

The goal is to measure various forms of language understanding by requiring a certain type of reasoning over the linguistic facts presented in each story.

A standard Transformer does not achieve good results on this task 2 .

However, we have designed a model based on the Universal Transformer which achieves state-of-the-art results on this task.

To encode the input, similar to BID14 , we first encode each fact in the story by applying a learned multiplicative positional mask to each word's embedding, and summing up all embeddings.

We embed the question in the same way, and then feed the (Universal) Transformer with these embeddings of the facts and questions.

As originally proposed, models can either be trained on each task separately ("train single") or jointly on all tasks ("train joint").

Table 1 summarizes our results.

We conducted 10 runs with different initializations and picked the best model based on performance on the validation set, similar to previous work.

Both the UT and UT with dynamic halting achieve state-of-the-art results on all tasks in terms of average error and number of failed tasks 3 , in both the 10K and 1K training regime (see Appendix E for breakdown by task).To understand the working of the model better, we analyzed both the attention distributions and the average ACT ponder times for this task (see Appendix F for details).

First, we observe that the attention distributions start out very uniform, but get progressively sharper in later steps around the correct supporting facts that are required to answer each question, which is indeed very similar to how humans would solve the task.

Second, with dynamic halting we observe that the average ponder time (i.e. depth of the per-symbol recurrent processing chain) over all positions in all samples in the test data for tasks requiring three supporting facts is higher (3.8??2.2) than for tasks requiring only two (3.1??1.1), which is in turn higher than for tasks requiring only one supporting fact (2.3??0.8).

This indicates that the model adjusts the number of processing steps with the number of supporting facts required to answer the questions.

Finally, we observe that the histogram of ponder times at different positions is more uniform in tasks requiring only one supporting fact compared to two and three, and likewise for tasks requiring two compared to three.

Especially for tasks requiring three supporting facts, many positions halt at step 1 or 2 already and only a few get transformed for more steps (see for example FIG3 .

This is particularly interesting as the length of stories is indeed much higher in this setting, with more irrelevant facts which the model seems to successfully learn to ignore in this way.

Similar to dynamic memory networks BID19 , there is an iterative attention process in UTs that allows the model to condition its attention over memory on the result of previous iterations.

Appendix F presents some examples illustrating that there is a notion of temporal states in UT, where the model updates its states (memory) in each step based on the output of previous steps, and this chain of updates can also be viewed as steps in a multi-hop reasoning process.

Next, we consider the task of predicting number-agreement between subjects and verbs in English sentences BID21 .

This task acts as a proxy for measuring the ability of a model to capture hierarchical (dependency) structure in natural language sentences.

We use the dataset provided by BID21 and follow their experimental protocol of solving the task using a language modeling training setup, i.e. a next word prediction objective, followed by calculating the ranking accuracy of the target verb at test time.

We evaluated our model on subsets of the test data with different task difficulty, measured in terms of agreement attractors -the number of intervening nouns with the opposite number from the subject (meant to confuse the model).

For example, given the sentence The keys to the cabinet 4 , the objective during training is to predict the verb are (plural).

At test time, we then evaluate the ranking accuracy of the agreement attractors: i.e. the goal is to rank are higher than is in this case.

Our results are summarized in TAB0 .

The best LSTM with attention from the literature achieves 99.18% on this task BID33 , outperforming a vanilla Transformer BID29 .

UTs significantly outperform standard Transformers, and achieve an average result comparable to the current state of the art (99.2%).

However, we see that UTs (and particularly with dynamic halting) perform progressively better than all other models as the number of attractors increases (see the last row, ???).

The LAMBADA task BID22 ) is a language modeling task consisting of predicting a missing target word given a broader context of 4-5 preceding sentences.

The dataset was specifically designed so that humans are able to accurately predict the target word when shown the full context, but not when only shown the target sentence in which it appears.

It therefore goes beyond language modeling, and tests the ability of a model to incorporate broader discourse and longer term context when predicting the target word.

The task is evaluated in two settings: as language modeling (the standard setup) and as reading comprehension.

In the former (more challenging) case, a model is simply trained for next-word prediction on the training data, and evaluated on the target words at test time (i.e. the model is trained to predict all words, not specifically challenging target words).

In the latter setting, introduced by Chu et al. BID5 , the target sentence (minus the last word) is used as query for selecting the target word from the context sentences.

Note that the target word appears in the context 81% of the time, making this setup much simpler.

However the task is impossible in the remaining 19% of the cases.

The results are shown in TAB1 .

Universal Transformer achieves state-of-the-art results in both the language modeling and reading comprehension setup, outperforming both LSTMs and vanilla Transformers.

Note that the control set was constructed similar to the LAMBADA development and test sets, but without filtering them in any way, so achieving good results on this set shows a model's strength in standard language modeling.

Our best fixed UT results used 6 steps.

However, the average number of steps that the best UT with dynamic halting took on the test data over all positions and examples was 8.2??2.1.

In order to see if the dynamic model did better simply because it took more steps, we trained two fixed UT models with 8 and 9 steps respectively (see last two rows).

Interestingly, these two models achieve better results compared to the model with 6 steps, but do not outperform the UT with dynamic halting.

This leads us to believe that dynamic halting may act as a useful regularizer for the model via incentivizing a smaller numbers of steps for some of the input symbols, while allowing more computation for others.

We trained UTs on three algorithmic tasks, namely Copy, Reverse, and (integer) Addition, all on strings composed of decimal symbols ('0'-'9').

In all the experiments, we train the models on sequences of length 40 and evaluated on sequences of length 400 BID17 train UTs using positions starting with randomized offsets to further encourage the model to learn position-relative transformations.

Results are shown in TAB3 .

The UT outperforms both LSTM and vanilla Transformer by a wide margin on all three tasks.

The Neural GPU reports perfect results on this task BID17 ), however we note that this result required a special curriculum-based training protocol which was not used for other models.

As another class of sequence-to-sequence learning problems, we also evaluate UTs on tasks indicating the ability of a model to learn to execute computer programs, as proposed in BID34 .

These tasks include program evaluation tasks (program, control, and addition), and memorization tasks (copy, double, and reverse).We use the mix-strategy discussed in BID34 to generate the datasets.

Unlike BID34 , we do not use any curriculum learning strategy during training and we make no use of target sequences at test time.

TAB4 present the performance of an LSTM model, Transformer, and Universal Transformer on the program evaluation and memorization tasks, respectively.

UT achieves perfect scores in all the memorization tasks and also outperforms both LSTMs and Transformers in all program evaluation tasks by a wide margin.

We trained a UT on the WMT 2014 English-German translation task using the same setup as reported in (Vaswani et al., 2017) in order to evaluate its performance on a large-scale sequence-to-sequence task.

Results are summarized in Table 7 .

The UT with a fully-connected recurrent transition function (instead of separable convolution) and without ACT improves by 0.9 BLEU over a Transformer and 0.5 BLEU over a Weighted Transformer with approximately the same number of parameters BID0 .

Universal Transformer small 26.8 Transformer base (Vaswani et al., 2017) 28.0 Weighted Transformer base BID0 28.4 Universal Transformer base 28.9 Table 7 : Machine translation results on the WMT14 En-De translation task trained on 8xP100 GPUs in comparable training setups.

All base results have the same number of parameters.

When running for a fixed number of steps, the Universal Transformer is equivalent to a multi-layer Transformer with tied parameters across all its layers.

This is partly similar to the Recursive Transformer, which ties the weights of its self-attention layers across depth BID13 5 .

However, as the per-symbol recurrent transition functions can be applied any number of times, another and possibly more informative way of characterizing the UT is as a block of parallel RNNs (one for each symbol, with shared parameters) evolving per-symbol hidden states concurrently, generated at each step by attending to the sequence of hidden states at the previous step.

In this way, it is related to architectures such as the Neural GPU BID17 and the Neural Turing Machine BID12 .

UTs thereby retain the attractive computational efficiency of the original feedforward Transformer model, but with the added recurrent inductive bias of RNNs.

Furthermore, using a dynamic halting mechanism, UTs can choose the number of processing steps based on the input data.

The connection between the Universal Transformer and other sequence models is apparent from the architecture: if we limited the recurrent steps to one, it would be a Transformer.

But it is more interesting to consider the relationship between the Universal Transformer and RNNs and other networks where recurrence happens over the time dimension.

Superficially these models may seem closely related since they are recurrent as well.

But there is a crucial difference: time-recurrent models like RNNs cannot access memory in the recurrent steps.

This makes them computationally more similar to automata, since the only memory available in the recurrent part is a fixed-size state vector.

UTs on the other hand can attend to the whole previous layer, allowing it to access memory in the recurrent step.

Given sufficient memory the Universal Transformer is computationally universal -i.e.

it belongs to the class of models that can be used to simulate any Turing machine, thereby addressing a shortcoming of the standard Transformer model 6 .

In addition to being theoretically appealing, our results show that this added expressivity also leads to improved accuracy on several challenging sequence modeling tasks.

This closes the gap between practical sequence models competitive on large-scale tasks such as machine translation, and computationally universal models such as the Neural Turing Machine or the Neural GPU BID12 BID17 , which can be trained using gradient descent to perform algorithmic tasks.

To show this, we can reduce a Neural GPU to a Universal Transformer.

Ignoring the decoder and parameterizing the self-attention module, i.e. self-attention with the residual connection, to be the identity function, we assume the transition function to be a convolution.

If we now set the total number of recurrent steps T to be equal to the input length, we obtain exactly a Neural GPU.

Note that the last step is where the Universal Transformer crucially differs from the vanilla Transformer whose depth cannot scale dynamically with the size of the input.

A similar relationship exists between the Universal Transformer and the Neural Turing Machine, whose single read/write operations per step can be expressed by the global, parallel representation revisions of the Universal Transformer.

In contrast to these models, however, which only perform well on algorithmic tasks, the Universal Transformer also achieves competitive results on realistic natural language tasks such as LAMBADA and machine translation.

Another related model architecture is that of end-to-end Memory Networks BID27 .

In contrast to end-to-end memory networks, however, the Universal Transformer uses memory corresponding to states aligned to individual positions of its inputs or outputs.

Furthermore, the Universal Transformer follows the encoder-decoder configuration and achieves competitive performance in large-scale sequence-to-sequence tasks.

This paper introduces the Universal Transformer, a generalization of the Transformer model that extends its theoretical capabilities and produces state-of-the-art results on a wide range of challenging sequence modeling tasks, such as language understanding but also a variety of algorithmic tasks, thereby addressing a key shortcoming of the standard Transformer.

The Universal Transformer combines the following key properties into one model:Weight sharing: Following intuitions behind weight sharing found in CNNs and RNNs, we extend the Transformer with a simple form of weight sharing that strikes an effective balance between inductive bias and model expressivity, which we show extensively on both small and large-scale experiments.

Conditional computation: In our goal to build a computationally universal machine, we equipped the Universal Transformer with the ability to halt or continue computation through a recently introduced mechanism, which shows stronger results compared to the fixed-depth Universal Transformer.

We are enthusiastic about the recent developments on parallel-in-time sequence models.

By adding computational capacity and recurrence in processing depth, we hope that further improvements beyond the basic Universal Transformer presented here will help us build learning algorithms that are both more powerful, data efficient, and generalize beyond the current state-of-the-art.

The code used to train and evaluate Universal Transformers is available at https: //github.com/tensorflow/tensor2tensor BID31 .

With respect to their computational power, the key difference between the Transformer and the Universal Transformer lies in the number of sequential steps of computation (i.e. in depth).

While a standard Transformer executes a total number of operations that scales with the input size, the number of sequential operations is constant, independent of the input size and determined solely by the number of layers.

Assuming finite precision, this property implies that the standard Transformer cannot be computationally universal.

When choosing a number of steps as a function of the input length, however, the Universal Transformer does not suffer from this limitation.

Note that this holds independently of whether or not adaptive computation time is employed but does assume a non-constant, even if possibly deterministic, number of steps.

Varying the number of steps dynamically after training is enabled by sharing weights across sequential computation steps in the Universal Transformer.

An intuitive example are functions whose execution requires the sequential processing of each input element.

In this case, for any given choice of depth T , one can construct an input sequence of length N > T that cannot be processed correctly by a standard Transformer.

With an appropriate, input-length dependent choice of sequential steps, however, a Universal Transformer, RNNs or Neural GPUs can execute such a function.

Here, we provide some additional details on the bAbI, subject-verb agreement, LAMBADA language modeling, and learning to execute (LTE) tasks.

The bAbi question answering dataset consists of 20 different synthetic tasks 7 .

The aim is that each task tests a unique aspect of language understanding and reasoning, including the ability of: reasoning from supporting facts in a story, answering true/false type questions, counting, understanding negation and indefinite knowledge, understanding coreferences, time reasoning, positional and size reasoning, path-finding, and understanding motivations (to see examples for each of these tasks, please refer to Table 1 in ).There are two versions of the dataset, one with 1k training examples and the other with 10k examples.

It is important for a model to be data-efficient to achieve good results using only the 1k training examples.

Moreover, the original idea is that a single model should be evaluated across all the tasks (not tuning per task), which is the train joint setup in Table 1 , and the tables presented in Appendix E.

Subject-verb agreement is the task of predicting number agreement between subject and verb in English sentences.

Succeeding in this task is a strong indicator that a model can learn to approximate syntactic structure and therefore it was proposed by BID21 as proxy for assessing the ability of different models to capture hierarchical structure in natural language.

Two experimental setups were proposed by BID21 for training a model on this task: 1) training with a language modeling objective, i.e., next word prediction, and 2) as binary classification, i.e. predicting the number of the verb given the sentence.

In this paper, we use the language modeling objective, meaning that we provide the model with an implicit supervision and evaluate based on the ranking accuracy of the correct form of the verb compared to the incorrect form of the verb.

In this task, in order to have different levels of difficulty, "agreement attractors" are used, i.e. one or more intervening nouns with the opposite number from the subject with the goal of confusing the model.

In this case, the model needs to correctly identify the head of the syntactic subject that corresponds to a given verb and ignore the intervening attractors in order to predict the correct form of that verb.

Here are some examples for this task in which subjects and the corresponding verbs are in boldface and agreement attractors are underlined:

The boy smiles.

One attractor:The number of men is not clear.

The ratio of men to women is not clear.

The ratio of men to women and children is not clear.

The LAMBADA task BID22 ) is a broad context language modeling task.

In this task, given a narrative passage, the goal is to predict the last word (target word) of the last sentence (target sentence) in the passage.

These passages are specifically selected in a way that human subjects are easily able to guess their last word if they are exposed to a long passage, but not if they only see the target sentence preceding the target word 8 .

Here is a sample from the dataset:

"

Yes, I thought I was going to lose the baby."

"I was scared too," he stated, sincerity flooding his eyes.

"

You were?" "Yes, of course.

Why do you even ask?" "This baby wasn't exactly planned for."

Target sentence:"Do you honestly think that I would want you to have a __

_

_

_

_

__?

" Target word: miscarriageThe LAMBADA task consists in predicting the target word given the whole passage (i.e., the context plus the target sentence).

A "control set" is also provided which was constructed by randomly sampling passages of the same shape and size as the ones used to build LAMBADA, but without filtering them in any way.

The control set is used to evaluate the models at standard language modeling before testing on the LAMBADA task, and therefore to ensure that low performance on the latter cannot be attributed simply to poor language modeling.

The task is evaluated in two settings: as language modeling (the standard setup) and as reading comprehension.

In the former (more challenging) case, a model is simply trained for the next word prediction on the training data, and evaluated on the target words at test time (i.e. the model is trained to predict all words, not specifically challenging target words).

In this paper, we report the results of the Universal Transformer in both setups.

LTE is a set of tasks indicating the ability of a model to learn to execute computer programs and was proposed by BID34 .

These tasks include two subsets: 1) program evaluation tasks (program, control, and addition) that are designed to assess the ability of models for understanding numerical operations, if-statements, variable assignments, the compositionality of operations, and more, as well as 2) memorization tasks (copy, double, and reverse).The difficulty of the program evaluation tasks is parameterized by their length and nesting.

The length parameter is the number of digits in the integers that appear in the programs (so the integers are chosen uniformly from [1, length]), and the nesting parameter is the number of times we are allowed to combine the operations with each other.

Higher values of nesting yield programs with deeper parse trees.

For instance, here is a program that is generated with length = 4 and nesting = 3.

j=8584 for x in range(8): j+=920 b=(1500+j) print((b+7567)) Target: 25011

<|TLDR|>

@highlight

We introduce the Universal Transformer, a self-attentive parallel-in-time recurrent sequence model that outperforms Transformers and LSTMs on a wide range of sequence-to-sequence tasks, including machine translation.

@highlight

Proposes a new model UT, based on the Transformer model, with added recurrence and dynamic halting of the recurrence.

@highlight

This paper extends Transformer by recursively applying a multi-head self-attention block, rather than stacking multiple blocks in the vanilla Transformer.