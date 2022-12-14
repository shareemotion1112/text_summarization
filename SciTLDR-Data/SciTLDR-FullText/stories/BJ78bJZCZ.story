Recurrent Neural Networks architectures excel at processing sequences by modelling dependencies over different timescales.

The recently introduced Recurrent Weighted Average (RWA) unit captures long term dependencies far better than an LSTM on several challenging tasks.

The RWA achieves this by applying attention to each input and computing a weighted average over the full history of its computations.

Unfortunately, the RWA cannot change the attention it has assigned to previous timesteps, and so struggles with carrying out consecutive tasks or tasks with changing requirements.

We present the Recurrent Discounted Attention (RDA) unit that builds on the RWA by additionally allowing the discounting of the past.

We empirically compare our model to RWA, LSTM and GRU units on several challenging tasks.

On tasks with a single output the RWA, RDA and GRU units learn much quicker than the LSTM and with better performance.

On the multiple sequence copy task our RDA unit learns the task three times as quickly as the LSTM or GRU units while the RWA fails to learn at all.

On the Wikipedia character prediction task the LSTM performs best but it followed closely by our RDA unit.

Overall our RDA unit performs well and is sample efficient on a large variety of sequence tasks.

Many types of information such as language, music and video can be represented as sequential data.

Sequential data often contains related information separated by many timesteps, for instance a poem may start and end with the same line, a scenario which we call long term dependencies.

Long term dependencies are difficult to model as we must retain information from the whole sequence and this increases the complexity of the model.

A class of model capable of capturing long term dependencies are Recurrent Neural Networks (RNNs).

A specific RNN architecture, known as Long Short-Term Memory (LSTM) BID13 , is the benchmark against which other RNNs are compared.

LSTMs have been shown to learn many difficult sequential tasks effectively.

They store information from the past within a hidden state that is combined with the latest input at each timestep.

This hidden state can carry information right from the beginning of the input sequence, which allows long term dependencies to be captured.

However, the hidden state tends to focus on the more recent past and while this mostly works well, in tasks requiring equal weighting between old and new information LSTMs can fail to learn.

A technique for accessing information from anywhere in the input sequence is known as attention.

The attention mechanism was introduced to RNNs by BID2 for neural machine translation.

The text to translate is first encoded by a bidirectional-RNN producing a new sequence of encoded state.

Different locations within the encoded state are focused on by multiplying each of them by an attention matrix and calculating the weighted average.

This attention is calculated for each translated word.

Computing the attention matrix for each encoded state and translated word combination provides a great deal of flexibility in choosing where in the sequence to attend to, but the cost of computing these matrices grows as a square of the number of words to translate.

This cost limits this method to short sequences, typically only single sentences are processed at a time.

The Recurrent Weighted Average (RWA) unit, recently introduced by BID17 , can apply attention to sequences of any length.

It does this by only computing the attention for each input once and computing the weighted average by maintaining a running average up to the current timestep.

Their experiments show that the RWA performs very well on tasks where information is needed from any point in the input sequence.

Unfortunately, as it cannot change the attention it assigns to previous timesteps, it performs poorly when asked to carry out multiple tasks within the same sequence, or when asked to predict the next character in a sample of text, a task in which new information is more important than old.

We introduce the Recurrent Discounted Attention (RDA) unit, which extends the RWA by allowing it to discount the attention applied to previous timesteps.

As this adjustment is applied to all previous timesteps at once, it continues to be efficient.

It performs very well both at tasks requiring equal weighting over all information seen and at tasks in which new information is more important than old.

The main contributions of this paper are as follows:1.

We analyse the Recurrent Weighted Average unit and show that it cannot output certain simple sequences.2.

We propose the Recurrent Discounted Attention unit that extends the Recurrent Weighted Average by allowing it to discount the past.3.

We run extensive experiments on the RWA, RDA, LSTM and GRU units and show that the RWA, RDA and GRU units are well suited to tasks with a single output, the RDA performs best on the multiple sequence copy task while the LSTM unit performs better on the Hutter Prize Wikipedia dataset.

Our paper is setout as follows: we present the analysis of the RWA (sections 3 and 4) and propose the RDA (section 5).

The experimental results (section 6), discussion (section 7) and conclusion follow (section 8).

Recently many people have worked on using RNNs to predict the next character in a corpus of text.

BID19 first attempted this on the Hutter Prize Wikipedia datasets using the MRNN archtecture.

Since then many architectures BID8 BID4 BID14 BID18 BID21 BID12 BID5 and regularization techniques BID1 BID16 have achieved impressive performance on this task, coming close to the bit-per-character limits bespoke compression algorithms have attained.

Many of the above architectures are very complex, and so the Gated Recurrent Unit (GRU) is a much simpler design that achieves similar performance to the LSTM.

Our experiments confirm previous literature BID3 ) that reports it performing very well.

Attention mechanisms have been used in neural machine translation by BID2 .

BID20 experimented with hard-attention on image where a single location is selected from a multinomial distribution.

BID20 introduced the global and local attention to refer to attention applied to the whole input and hard attention applied to a local subset of the input.

An idea related to attention is the notion of providing additional computation time for difficult inputs.

BID9 introduce shows that this yields insight into the distribution of information in the input data itself.

Several RNN architectures have attempted to deal with long term dependencies by storing information in an external memory BID10 .

At each timestep the Recurrent Weighted Average model uses its current hidden state h t???1 and the input x t to calculate two quantities:1.

The features z t of the current input: DISPLAYFORM0 where u t is an unbounded vector dependent only on the input x t , and tanh g t is a bounded vector dependent on the input x t and the hidden state h t???1 .

Notation: W are weights, b are biases, (??) is matrix multiplication, and is the elementwise product.2.

The attention a t to pay to the features z t : DISPLAYFORM1 The hidden state h t is then the average of of the features z t , weighted by the attention a t , and squashed through the hyperbolic tangent function: DISPLAYFORM2 This is implemented efficiently as a running average: DISPLAYFORM3 where n t is the numerator and d t is the denominator of the average.

The RWA shows superior experimental results compared to the LSTM on the following tasks:1.

Classifying whether a sequence is longer than 500 elements.2.

Memorising short random sequences of symbols and recalling them at any point in the subsequent 1000 timesteps.3.

Adding two numbers spaced randomly apart on a sequence of length 1000.4.

Classifying MNIST images pixel by pixel.

All of these tasks require combining the full sequence of inputs into a single output.

It makes perfect sense that an average over all timesteps would perform well in these tasks.

On the other hand, we can imagine tasks where an average over all timesteps would not work effectively:1.

Copying many input sequences from input to output.

It will need to forget sequences once they have been output.2.

Predicting the next character in a body of text.

Typically, the next character depends much more on recent characters than on those from the beginning of the text.

All of these follow from the property that d t is monotonically increasing in t, which can be seen from a t > 0 and d t = d t???1 + a t .

As d t becomes larger, the magnitude of a t must increase to change the value of h t .

This means that it becomes harder and harder to change the value of h t to the point where it almost becomes fixed.

In the specific case of outputting the sequence h t = ???1 t c we can show that a t must grow geometrically with time.

Lemma 1 Let the task be to output the sequence h t = ???1 t c for 0 < c < 1.

Let h t be defined by the equations of the Recurrent Weighted Average, and let z t be bounded and f h be a continuous, monotonically increasing surjection from R ??? (???1, 1).

Then, a t grows geometrically with increasing t.

Corollary 2 If a t is also bounded then it cannot grow geometrically for all time and so the RWA cannot output the sequence h t = ???1 t c.

Corollary 2 suggests that the Recurrent Weighted Average may not actually be Turing Complete.

Overall, these properties suggest the the RWA is a good choice for tasks with a single result, but not for sequences with multiple results or tasks that require forgetting.

The RDA uses its current hidden state h t???1 and the input x t to calculate three quantities:1.

The features z t of the current input are calculated identically to the RWA: DISPLAYFORM0 The attention a t to pay to the features: z t DISPLAYFORM1 Here we generalize attention to allow any function f a which is non-negative and monotonically increasing.

If we choose f a = exp, then we recover the RWA attention function.3.

The discount factor ?? t to apply to the previous values in the average DISPLAYFORM2 where ?? is the sigmoid/logistic function defined as ??( DISPLAYFORM3 We use these values to calculate a discounted moving average.

This discounting mechanism is crucial in remediating the RWA's inability to forget the past DISPLAYFORM4 Here we generalize RWA further by allowing f h to be any function, and we also introduce a final transformation to the hidden state h t to produce the output DISPLAYFORM5 The attention function f a (x) is a non-negative monotonically increasing function of x. There are several possible choices:??? f a (x) = e x -This is used in the RWA.??? f a (x) = max(0, x) -Using a ReLU allows the complete ignoring of some timesteps with linearly increasing attention for others.??? f a (x) = ln(1 + e x ) -The softplus function is a smooth approximation to the ReLU.??? f a (x) = ??(x) -Using the sigmoid limits the maximum attention an input can be given.

The domain of the hidden activation function f h is the average nt dt .

This average is bounded by the minimum and maximum values of z t .

Possible choices of f h include: DISPLAYFORM6 ) -This is used in the RWA.

We observed that the range of nt dt mostly remained in the linear domain of tanh centred around 0, suggesting that using this was unneccessary.??? f h ( nt dt ) = nt dt -The identity is our choice for f h in the RDA.Possible choices for the output function f o are:??? f o (h t ) = h t -The RWA uses the identity as its hidden state has already been transformed by tanh.??? f o (h t ) = tanh(h t ) -The output can be squashed between [???1, 1] using tanh.

We ran experiments to investigate the following questions: We provide plots of the training process in Appendix B.

For all tasks except the Wikipedia character prediction task, we use 250 recurrent units.

Weights are initialized using Xavier initialization BID7 and biases are initialized to 0, except for forget gates and discount gates which are initialized to 1 (Gers, Schmidhuber, and Cummins, 2000).

We use mini-batches of 100 examples and backpropagate over the full sequence length.

We train the models using Adam BID15 with a learning rate of 0.001.

Gradients are clipped between -1 and 1.For the Wikipedia task, we use a character embedding of 64 dimensions, followed by a single layer of 1800 recurrent units, and a final softmax layer for the character prediction.

We apply truncated backpropagation every 250 timesteps, and use the last hidden state of the sequence as the initial hidden state of the next sequence to approximate full backpropagation.

All of our experiments are implemented in TensorFlow BID0 Table 4 : MNIST permuted test set accuracy.??? Using a ReLU for the attention function f a almost always fails to train.

Using a Softplus for f a is much more stable than a ReLU.

However, it doesn't perform as well as using sigmoid or exponential attention.??? Exponential attention performs well in all tasks, and works best with the tanh output function f o (h t ) = tanh(h t ).

We refer to this as RDA-exp-tanh.??? Sigmoid attention performs well in all tasks, and works best with the identity output function f o (h t ) = h t .

We refer to this as RDA-sigmoid-id.??? It is difficult to choose between RDA-exp-tanh and RDA-sigmoid-id.

RDA-exp-tanh often trains faster, but it sometimes diverges with NaN errors during training.

RDA-sigmoid-id trains slower but is more stable, and tends to have better loss.

We include results for both of them.

Here we investigate whether sequences with a single task can be performed as well with the RDA as with the RWA.Each of the four tasks detailed below require the RNN to save some or all of the input sequence before outputting a single result many steps later.1.

Addition -The input consists of two sequences.

The first is a sequence of numbers each uniformly sampled from [0, 1], and the second consists of all zeros except for two ones which indicate the two numbers of the first sequence to be added together. (Table 1) 2.

Classify length -A sequence of length between 1 and 1000 is input.

The goal is to classify whether the sequence is longer than 500.

All RNN architectures could learn their initial hidden state for this task, which improved performance for all of them.

TAB1 3.

MNIST -The task is supervised classification of MNIST digits.

We flatten the 28x28 pixel arrays into a single 784 element sequence and use RNNs to predict the digit class labels.

This task challenges networks' ability to learn long-range dependencies, as crucial pixels are present at the beginning, middle and end of the sequence.

We implement two variants of this task:(a) Sequential -the pixels are fed in from the top left to the bottom right of the image.

TAB2 BID19 1.60 GF-LSTM BID4 1.58 Grid-LSTM BID14 1.47 MI-LSTM (Wu et al., 2016) 1.44 Recurrent Memory Array Structures (Rocki, 2016a) 1.40 HyperNetworks BID12 1.35 LayerNorm HyperNetworks BID12 1.34 Recurrent Highway Networks BID21 1 Table 7 : Bits per character on the Hutter Prize Wikipedia test set (b) Permuted -the pixels of the image are randomly permuted before the image is fed in.

The same permutation is applied to all images. (Table 4) 4.

Copy -The input sequence starts with randomly sampled symbols.

The rest of the input is blanks except for a single recall symbol.

The goal is to memorize the starting symbols and output them when prompted by the recall symbol.

All other output symbols must be blank.

TAB4 6.4 MULTIPLE SEQUENCE COPY TASK Here we investigate whether the different RNN units can cope with doing the same task repeatedly.

The tasks consists of multiple copying tasks all within the same sequence.

Instead of having the recall symbol randomly placed over the whole sequence it always appears a couple of steps after the sequence being memorized.

This gives room for 50 consecutive copying tasks in a length 1000 input sequence.

TAB5 6.5 WIKIPEDIA CHARACTER PREDICTION TASKThe standard test for RNN models is character-level language modelling.

We evaluate our models on the Hutter Prize Wikipedia dataset enwik8, which contains 100M characters of 205 different symbols including XML markup and special characters.

We split the data into the first 90M characters for the training set, the next 5M for validation, and the final 5M for the test set. (Table 7) 7 DISCUSSION We start our discussion by describing the performance of each individual unit.

Our analysis of the RWA unit showed that it should only work well on the single task sequences and we confirm this experimentally.

It learns the single sequence tasks quickly but is unable to learn the multiple sequence copy task and Wikipedia character prediction task.

Our experiments show that the RDA unit is a consistent performer on all types of tasks.

As expected, it learns single task sequences slower than the RWA but it actually achieves better generalization on the MNIST test sets.

We speculate that the cause of this improvement is because the ability to forget effectively allows it to compress the information it has previously processed, or perhaps discounting the past should be considered as changing the attention on the past and the RDA is able to vary its attention on previous inputs based on later inputs.

On the multiple sequence copy task the RDA unit was far superior to all other units learning three times as fast as the LSTM and GRU units.

On the Wikipedia character prediction task the RDA unit performed respectably, achieving a better compression rate than the GRU but worse than the LSTM.The LSTM unit learns the single task sequences slower than all the other units and often fails to learn at all.

This is surprising as it is often used on these tasks as a baseline against which other archtectures are compared.

On the multiple sequence copy task it learns slowly compared to the RDA units but solves the task.

The Wikipedia character prediction task is where it performs best, learning much faster and achieving better compression than the other units.

The GRU unit works very well on single task sequences often learning the fastest and achieving excellent generalization on the MNIST test sets.

On the multiple sequence copy task it has equal performance to the LSTM.

On the Wikipedia character prediction task it performs worse than the LSTM and RDA units but still achieves a good performance.

We now look at how our results show that different neural network architectures are suited for different tasks.

For our single output tasks the RWA, RDA and GRU units work best.

Thus for similar real work applications such as encoding a molecule into a latent representation, classification of genomic sequences, answering questions or language translation, these units should be considered before LSTM units.

However, our results are yet to be verified in these domains.

For sequences that contain an unknown number of independent tasks the RDA unit should be used.

For the Wikipedia character prediction task the LSTM performs best.

Therefore we can't recommend RWA, RDA or GRU units on this or similar tasks.

We analysed the Recurrent Weighted Average (RWA) unit and identified its weakness as the inability to forget the past.

By adding this ability to forget the past we arrived at the Recurrent Discounted Attention (RDA).

We implemented several varieties of the RDA and compared them to the RWA, LSTM and GRU units on several different tasks.

We showed that in almost all cases the RDA should be used in preference to the RWA and is a flexible RNN unit that can perform well on all types of tasks.

We also determined which types of tasks were more suited to each different RNN unit.

For tasks involving a single output the RWA, RDA and GRU units performed best, for the multiple sequence copy task the RDA performed best, while on the Wikipedia character prediction task the LSTM unit performed best.

We recommend taking these results into account when choosing a unit for real world applications.

A MATHEMATICAL PROOFS Lemma 1 Let the task be to output the sequence h t = ???1 t c for 0 < c < 1.

Let h t be defined by the equations of the Recurrent Weighted Average, and let z t be bounded and f h be a continuous, monotonically increasing surjection from R ??? FIG0 .

Then, a t grows geometrically with increasing t.

Given that the activation f h is a continuous, monotonically increasing surjection from R to (???1, 1), we know that there are two values x + and x ??? such that f (x + ) = c and f (x ??? ) = ???c.

Define DISPLAYFORM0 Then for every even integer i, we have DISPLAYFORM1 From the definitions of n t and d t we have DISPLAYFORM2 Substituting n i = d i x + and rearranging yields DISPLAYFORM3 where |z| max = max{|z|} and |x| max = max{|x + |, |x ??? |}. From the definition of d t we have DISPLAYFORM4 As d t grows geometrically, then so does a t .Corollary 2 If a t is also bounded then it cannot grow geometrically for all time and so the RWA cannot output the sequence h t = ???1 t cProof Assume the RWA can output the sequence h t = ???1 t c. As a t grows geometrically, it is unbounded, but this is a contradiction.

We include figures of the loss function learning curves during training.

In the case of MNIST, we report on validation accuracy instead.

These experiments provide evidence that the two flavours of the RDA unit consistently perform close to the best across a broad range of tasks.

Figures are best viewed in colour.

@highlight

We introduce the Recurrent Discounted Unit which applies attention to any length sequence in linear time

@highlight

This paper proposes the Recurrent Discounted Attention (RDA), an extension to Recurrent Weighted Average (RWA) by adding a discount factor.

@highlight

Extends the recurrent weight average to overcome the limitation of the original method while maintaining its advantage and proposes the method of using Elman nets as the base RNN