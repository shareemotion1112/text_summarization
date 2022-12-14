Long Short-Term Memory (LSTM) units have the ability to memorise and use long-term dependencies between inputs to generate predictions on time series data.

We introduce the concept of modifying the cell state (memory) of LSTMs using rotation matrices parametrised by a new set of trainable weights.

This addition shows significant increases of performance on some of the tasks from the bAbI dataset.

In the recent years, Recurrent Neural Networks (RNNs) have been successfully used to tackle problems with data that can be represented in the shape of time series.

Application domains include Natural Language Processing (NLP) (translation BID12 , summarisation BID9 , question answering and more), speech recogition BID5 BID3 ), text to speech systems BID0 , computer vision tasks BID13 BID16 , and differentiable programming language interpreters BID10 BID11 ).An intuitive explanation for the success of RNNs in fields such as natural language understanding is that they allow words at the beginning of a sentence or paragraph to be memorised.

This can be crucial to understanding the semantic content.

Thus in the phrase "The cat ate the fish" it is important to memorise the subject (cat).

However, often later words can change the meaning of a senstence in subtle ways.

For example, "The cat ate the fish, didn't it" changes a simple statement into a question.

In this paper, we study a mechanism to enhance a standard RNN to enable it to modify its memory, with the hope that this will allow it to capture in the memory cells sequence information using a shorter and more robust representation.

One of the most used RNN units is the Long Short-Term Memory (LSTM) BID7 .

The core of the LSTM is that each unit has a cell state that is modified in a gated fashion at every time step.

At a high level, the cell state has the role of providing the neural network with memory to hold long-term relationships between inputs.

There are many small variations of LSTM units in the literature and most of them yield similar performance BID4 .The memory (cell state) is expected to encode information necessary to make the next prediction.

Currently the ability of the LSTMs to rotate and swap memory positions is limited to what can be achieved using the available gates.

In this work we introduce a new operation on the memory that explicitly enables rotations and swaps of pairwise memory elements.

Our preliminary tests show performance improvements on some of the bAbI tasks compared with LSTM based architectures.

In this section we introduce the idea of adding a new set of parameters for the RNN cell that enable rotation of the cell state.

The following subsection shows how this is implemented in the LSTM unit.

One of the key innovations of LSTMs was the introduction of gated modified states so that if the gate neuron i is saturated then the memory c i (t ??? 1) would be unaltered.

That is, c i (t ??? 1) ??? c i (t) with high accuracy.

The fact that the amplification factor is very close to 1 prevents the memory vanishing or exploding over many epochs.

To modify the memory, but retain an amplification factor of 1 we take the output after appling the forget and add gates (we call it d t ), and apply a rotation matrix U to obtain a modified memory c t = Ud t .

Note that, for a rotation matrix U T U = I so that d t = c t .We parametrise the rotation by a vector of angles DISPLAYFORM0 where W rot is a weight matrix and b rot is a bias vector which we learn along with the other parameters.

x is the vector of our concatenated inputs (in LSTMs given by concatenating the input for the current timestep with the output from the previous time step).A full rotation matrix is parametrisable by n(n ??? 1)/2 parameters (angles).

Using all of these would introduce a huge number of weights, which is likely to over-fit.

Instead, we have limited ourselves to considering rotations between pairs of inputs d i (t) and d i+1 (t).

Exploring more powerful sets of rotations is currently being investigated.

Our rotation matrix is a block-diagonal matrix of 2D rotations DISPLAYFORM1 where the cell state is of size n. Our choice of rotations only needs n/2 angles.

In this section we show how to add memory rotation to the LSTM unit.

The rotation is applied after the forget and add gates and before using the current cell state to produce an output.

The RotLSTM equations are as follows: DISPLAYFORM0 DISPLAYFORM1 where W {f,i,o,rot,c} are weight matrices, b {f,i,o,rot,c} are biases (Ws and bs learned during training), h t???1 is the previous cell output, h t is the output the cell produces for the current timestep, similarly c t???1 and c t are the cell states for the previous and current timestep, ??? is element-wise multiplication and [??, ??] is concatenation.

U as defined in Equation 2, parametrised by u t .

Figure 1 shows a RotLSTM unit in detail.

Assuming cell state size n, input size m, the RotLSTM has n(n + m)/2 extra parameters, a 12.5% increase (ignoring biases).

Our expectation is that we can decrease n without harming performance and the rotations will enforce a better representation for the cell state.

To empirically evaluate the performance of adding the rotation gate to LSTMs we use the toy NLP dataset bAbI with 1000 samples per task.

The bAbI dataset is composed of 20 different tasks of various difficulties, starting from easy questions based on a single supporting fact (for example: DISPLAYFORM0 x is the concatenation of h t???1 and x t in the diagram (green and blue lines).

Note that this differs from a regular LSTM by the introduction of the network producing angles u t and the rotation module marked U. In the diagram input size is 4 and cell state size is 3.John is in the kitchen.

Where is John?

A: Kitchen) and going to more difficult tasks of reasoning about size (example: The football fits in the suitcase.

The box is smaller than the football.

Will the box fit in the suitcase?

A: yes) and path finding (example: The bathroom is south of the office.

The bathroom is north of the hallway.

How do you go from the hallway to the office?

A: north, north).

A summary of all tasks is available in Table 2 .

We are interested in evaluating the behaviour and performance of rotations on RNN units rather than beating state of the art.

We compare a model based on RotLSTM with the same model based on the traditional LSTM.

All models are trained with the same hyperparameters and we do not perform any hyperparameter tuning apart from using the sensible defaults provided in the Keras library and example code BID2 .For the first experiment we train a LSTM and RotLSTM based model 10 times using a fixed cell state size of 50.

In the second experiment we train the same models but vary the cell state size from 6 to 50 to assess whether the rotations help our models achieve good performance with smaller state sizes.

We only choose even numbers for the cell state size to make all units go through rotations.

The model architecture, illustrated in FIG0 , is based on the Keras example implementation 1 .

This model architecture, empirically, shows better performance than the LSTM baseline published in .

The input question and sentences are passed thorugh a word embedding layer (not shared, embeddings are different for questions and sentences).

The question is fed into an RNN which produces a representation of the question.

This representation is concatenated to every word vector from the story, which is then used as input to the second RNN.

Intuitively, this helps the second RNN (Query) to focus on the important words to answer the question.

The output of the second RNN is passed to a fully connected layer with a softmax activation of the size of the dictionary.

The answer is the word with the highest activation.

The categorical cross-entropy loss function was used for training.

All dropout layers are dropping 30% of the nodes.

The train-validation dataset split used was 95%-5%.

The optimizer used was Adam with learning rate 0.001, no decay, ?? 1 = 0.9, ?? 2 = 0.999, = 10 ???8 .

The training set was randomly shuffled before every epoch.

All models were trained for 40 epochs.

After every epoch the model performance was evaluated on the validation and training sets, and every 10 epochs on the test set.

We set the random seeds to the same number for reproducibility and ran the experiments 10 times with 10 different random seeds.

The source code is available at https://goo.gl/ Eopz2C 2 .

In this subsection we compare the the performance of models based on the LSTM and RotLSTM units on the bAbI dataset.

Applying rotations on the unit memory of the LSTM cell gives a slight improvement in performance overall, and significant improvements on specific tasks.

Results are shown in TAB0 .

The most significant improvements are faster convergence, as shown in Figure 3 , and requiring smaller state sizes, illustrated in FIG2 .On tasks 1 (basic factoid), 11 (basic coreference), 12 (conjunction) and 13 (compound coreference) the RotLSTM model reaches top performance a couple of epochs before the LSTM model consistently.

The RotLSTM model also needs a smaller cell state size, reaching top performance at state size 10 to 20 where the LSTM needs 20 to 30.

The top performance is, however, similar for both models, with RotLSTM improving the accuracy with up to 2.5%.The effect is observed on task 18 (reasoning about size) at a greater magnitude where the RotLSTM reaches top performance before epoch 20, after which it plateaus, while the LSTM model takes 40 epochs to fit the data.

The training is more stable for RotLSTM and the final accuracy is improved by 20%.

The RotLSTM reaches top performance using cell state 10 and the LSTM needs size 40.

Similar performance increase for the RotLSTM (22.1%) is observed in task 5 (three argument relations), reaching top performance around epoch 25 and using a cell state of 50.

Task 7 (counting) shows a similar behaviour with an accuracy increase of 14% for RotLSTM.Tasks 4 (two argument relations) and 20 (agent motivation) show quicker learning (better performance in the early epochs) for the RotLSTM model but both models reach their top performance after the same amount of traning.

On task 20 the RotLSTM performance reaches top accuracy using state size 10 while the LSTM incremetally improves until using state size 40 to 50.Signs of overfitting for the RotLSTM model can be observed more prominently than for the LSTM model on tasks 15 (basic deduction) and 17 (positional reasoning).Our models, both LSTM and RotLSTM, perform poorly on tasks 2 and 3 (factoid questions with 2 and 3 supporting facts, respectively) and 14 (time manipulation).

These problem classes are solved very well using models that look over the input data many times and use an attention mechanism that allows the model to focus on the relevant input sentences to answer a question BID14 BID8 .

Our models only look at the input data once and we do not filter out irrelevant information.

A limitation of the models in our experiments is only applying pairwise 2D rotations.

Representations of past input can be larger groups of the cell state vector, thus 2D rotations might not fully exploit the benefits of transformations.

In the future we hope to explore rotating groups of elements and multi-dimensional rotations.

Rotating groups of elements of the cell state could potentially also force the models to learn a more structured representation of the world, similar to how forcing a model to learn specific representations of scenes, as presented in BID6 , yields semantic representations of the scene.

Rotations also need not be fully flexible.

Introducing hard constraints on the rotations and what groups of parameters can be rotated might lead the model to learn richer memory representations.

Future work could explore how adding such constraints impacts learning times and final performance on different datasets, but also look at what constraints can qualitatively improve the representation of long-term dependencies.

In this work we presented prelimiary tests for adding rotations to simple models but we only used a toy dataset.

The bAbI dataset has certain advantages such as being small thus easy to train many models on a single machine, not having noise as it is generated from a simulation, and having a wide range of tasks of various difficulties.

However it is a toy dataset that has a very limited vocabulary and lacks the complexity of real world datasets (noise, inconsistencies, larger vocabularies, more complex language constructs, and so on).

Another limitation of our evaluation is only using text, specifically question answering.

To fully evaluate the idea of adding rotations to memory cells, in the future, we aim to look into incorporating our rotations on different domains and tasks including speech to text, translation, language generation, stock prices, and other common problems using real world datasets.

Tuning the hyperparameters of the rotation models might give better insights and performance increases and is something we aim to incorporate in our training pipeline in the future.

A brief exploration of the angles produced by u and the weight matrix W rot show that u does not saturate, thus rotations are in fact applied to our cell states and do not converge to 0 (or 360 degress).

A more in-depth qualitative analysis of the rotation gate is planned for future work.

Peeking into the activations of our rotation gates could help understand the behaviour of rotations and to what extent they help better represent long-term memory.

A very successful and popular mutation of the LSTM is the Gated Recurrent Unit (GRU) unit BID1 .

The GRU only has an output as opposed to both a cell state and an output and uses fewer gates.

In the future we hope to explore adding rotations to GRU units and whether we can obtain similar results.

We have introduced a novel gating mechanism for RNN units that enables applying a parametrised transformation matrix to the cell state.

We picked pairwise 2D rotations as the transformation and shown how this can be added to the popular LSTM units to create what we call RotLSTM.

Figure 3: Accuracy comparison on training, validation (val) and test sets over 40 epochs for LSTM and RotLSTM models.

The models were trained 10 times and shown is the average accuracy and in faded colour is the standard deviation.

Test set accuracy was computed every 10 epochs.

We trained a simple model using RotLSTM units and compared them with the same model based on LSTM units.

We show that for the LSTM-based architetures adding rotations has a positive impact on most bAbI tasks, making the training require fewer epochs to achieve similar or higher accuracy.

On some tasks the RotLSTM model can use a lower dimensional cell state vector and maintain its performance.

Significant accracy improvements of approximatively 20% for the RotLSTM model over the LSTM model are visible on bAbI tasks 5 (three argument relations) and 18 (reasoning about size).

@highlight

Adding a new set of weights to the LSTM that rotate the cell memory improves performance on some bAbI tasks.