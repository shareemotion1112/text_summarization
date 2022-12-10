In this work, we first conduct mathematical analysis on the memory, which is defined as a function that maps an element in a sequence to the current output, of three RNN cells; namely, the simple recurrent neural network (SRN), the long short-term memory (LSTM) and the gated recurrent unit (GRU).

Based on the analysis, we propose a new design, called the extended-long short-term memory (ELSTM), to extend the memory length of a cell.

Next, we present a multi-task RNN model that is robust to previous erroneous predictions, called the dependent bidirectional recurrent neural network (DBRNN), for the sequence-in-sequenceout (SISO) problem.

Finally, the performance of the DBRNN model with the ELSTM cell is demonstrated by experimental results.

The recurrent neural network (RNN) has proved to be an effective solution for natural language processing (NLP) through the advancement in the last three decades BID8 BID11 BID2 BID1 .

At the cell level of a RNN, the long short-term memory (LSTM) BID10 and the gated recurrent unit (GRU) are often adopted by a RNN as its low-level building cell.

Being built upon these cells, various RNN models have been proposed to solve the sequence-in-sequence-out (SISO) problem.

To name a few, there are the bidirectional RNN (BRNN) BID14 , the encoder-decoder model BID15 BID16 BID0 and the deep RNN BID12 .

Although the LSTM and the GRU were designed to enhance the memory length of RNNs and avoid the gradient vanishing/exploding issue BID10 BID13 BID3 , a good understanding of their memory length is still lacking.

Here, we define the memory of a RNN model as a function that maps an element in a sequence to current output.

The first objective of this research is to analyze the memory length of three RNN cells -the simple RNN (SRN) BID8 BID11 , the long short-term memory (LSTM) and the gated recurrent unit (GRU).

This will be conducted in Sec. 2.

Such analysis is different to the investigation of gradient vanishing/exploding problem in a sense that gradient vanishing/exploding problem happens during the training process, the memory analysis is, however, done on a trained RNN model.

Based on the understanding from the memory analysis, we propose a new design, called the extended-long short-term memory (ELSTM), to extend the memory length of a cell in Sec.3.As to the macro RNN model, one popular choice is the BRNN.

Since the elements in BRNN output sequences should be independent of each other BID14 , the BRNN cannot be used to solve dependent output sequence problem alone.

Nevertheless, most language tasks do involve dependent output sequences.

The second choice is the encoder-decoder system, where the attention mechanism has been introduced BID16 BID0 to improve its performance furthermore.

As shown later in this work, the encoder-decoder system is not an efficient learner.

Here, to take advantages of both the encoder-decoder and the BRNN and overcome their drawbacks, we propose a new multitask model called the dependent bidirectional recurrent neural network (DBRNN), which will be elaborated in Sec. 4.

Furthermore, we conduct a series of experiments on the part of speech (POS) tagging and the dependency parsing (DP) problems in Sec. 5 to demonstrate the performance of the DBRNN model with the ELSTM cell.

Finally, concluding remarks are given and future research direction is pointed out in Sec. 6.

For a large number of NLP tasks, we are concerned with finding semantic patterns from the input sequence.

It was shown by BID8 that the RNN builds an internal representation of semantic patterns.

The memory of a cell characterizes its ability to map an input sequence of certain length into such a representation.

More rigidly, we define the memory as a function that maps an element in a sequence to the current output.

So the memory capability of a RNN is not only about whether an element can be mapped into current output, but also how this mapping takes place.

It was reported by BID9 that a SRN only memorized sequences of length between 3-5 units while a LSTM could memorize sequences of length longer than 1000 units.

In this section, we study the memory of the SRN, LSTM and GRU.

Here, for the ease of analysis, we use Elman's SRN model BID8 with linear hidden state activation function and non-linear output activation function since such cell model is mathematically tractable and performance-wise equivalent to BID11 and Tensorflow's variations.

The SRN is described by the following two equations: DISPLAYFORM0 DISPLAYFORM1 where subscript t is the index of the time unit, W c ∈ R N ×N is the weight matrix for hidden state vector c t−1 ∈ R N , W in ∈ R N ×M is the weight matrix of input vector X t ∈ R M , h t ∈ R N and f (·) is an element-wise non-linear function.

Usually, f (·) is a hyperbolic-tangent or a sigmoid function.

Throughout this paper, we omit the bias terms by putting them inside the corresponding weight matrices.

By induction, c t can be rewritten as DISPLAYFORM2 where c 0 is the initial internal state of the SRN.

Typically, we set c 0 = 0.

Then, Eq. (3) becomes DISPLAYFORM3 Let λ max be the largest singular value of W c .

Then, we have DISPLAYFORM4 Here, we are only interested in the case of memory decay when λ max < 1.

Hence, the contribution of X k , k < t, to h t decays at least in form of λ |t−k| max .

We conclude that SRN's memory decays at least exponentially with its memory length |t − k|.

By following the work of BID10 , we plot the diagram of a LSTM cell in FIG0 .

In this figure, φ, σ and ⊗ denote the hyperbolic tangent function, the sigmoid function and the multiplication operation, respectively.

All of them operate in an element-wise fashion.

The LSTM has an input gate, an output gate, a forget gate and a constant error carousal (CEC) module.

Mathematically, the LSTM cell can be written as DISPLAYFORM5 DISPLAYFORM6 where c t ∈ R N , column vector I t ∈ R (M +N ) is a concatenation of the current input, X t ∈ R M , and the previous output, h t−1 ∈ R N (i.e., I DISPLAYFORM7 and W in are weight matrices for the forget gate, the input gate, the output gate and the input, respectively.

Under the assumption c 0 = 0, the hidden state vector of the LSTM can be derived by induction as DISPLAYFORM8 By setting f (·) in Eq. (2) to a hyperbolic-tangent function, we can compare outputs of the SRN and the LSTM below: DISPLAYFORM9 LSTM: h DISPLAYFORM10 We see from the above that W t−k c and t j=k+1 σ(W f I j ) play the same memory role for the SRN and the LSTM, respectively.

DISPLAYFORM11 As given in Eqs. FORMULA4 and FORMULA0 , the impact of input I k on output h t in the LSTM lasts longer than that of input X k in the SRN.

This is the case if an appropriate weight matrix, W f , of the forget gate is selected.

The GRU was originally proposed for neural machine translation .

It provides an effective alternative for the LSTM.

Its operations can be expressed by the following four equations: DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 where X t , h t , z t and r t denote the input vector, the hidden state vector, the update gate vector and the reset gate vector, respectively, and W z , W r , W , are trainable weight matrices.

Its hidden state is also its output, which is given in Eq. FORMULA0 .

If we simplify the GRU by setting U z , U r and U to zero matrices, then we can obtain the following simplified GRU system: DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 For the simplified GRU with the initial rest condition, we can derive the following by induction: DISPLAYFORM6 By comparing Eqs. FORMULA8 and FORMULA0 , we see that the update gate of the simplified GRU and the forget gate of the LSTM play the same role.

One can control the memory decay behavior of the GRU by choosing the weight matrix, W z , of the update gate carefully.

As discussed above, the LSTM and the GRU have longer memory by introducing the forget and the update gates, respectively.

However, from Eq. 10 and Eq. 11, it can be seen that the impact of proceeding element to the current output at time step t still fades quickly due to the presence of forget gate and update gate.

And as we will show in the ELSTM design, this does not have to be the case.

In this section, we attempt to design extended-long short-term memory (ELSTM) cells and propose two new cell models:• ELSTM-I: the extended-long short-term memory (ELSTM) with trainable input weight vector s i ∈ R N , i = 1, · · · , t − 1, where weights s i and s j (with i = j) are independent.• ELSTM-II: the ELSTM-I with no forget gate.

The ELSTM-I cell can be described by DISPLAYFORM0 DISPLAYFORM1 where b ∈ R N is a trainable bias vector.

The ELSTM-II cell can be written as DISPLAYFORM2 DISPLAYFORM3 As shown above, we introduce scaling factor, s i , i = 1, · · · , t − 1, to the ELSTM-I and the ELSTM-II to increase or decrease the impact of input I i in the sequence.

To prove that the proposed ELSTM-I has longer memory than LSTM, we first derive the closed form expression of h t , which is: DISPLAYFORM4 We then pick s k such that: DISPLAYFORM5 Compare Eq. 25 with Eq. 11, we conclude that ELSTM-I has longer memory than LSTM.

As a matter of fact, s k plays a similarly role as the attention score in various attention models such as BID16 .

The impact of proceeding elements to the current output can be adjusted (either increase or decrease) by s k .

The memory capability of ELSTM-II can be proven in a similarly fashion, so even ELSTM-II does not have forget gate, it is capable in attending to or forgetting a particular position of a sequence as ELSTM-I through the scaling factor.

The major difference between the ELSTM-I and the ELSTM-II is that fewer parameters are used in the ELSTM-II than those in the ELSTM-I. The numbers of parameters used by different RNN cells are compared in TAB0 , where X t ∈ R M , h t ∈ R N and t = 1, · · · , T .

Although the number of parameters of ELSTM depends on the maximum length of a sequence in practice, the memory overhead required is limited.

ELSTM-II requires less number of parameters than LSTM for typical lengthed sequence.

From Table.

1, to double the number of parameters as compare to an ordinary LSTM, the length of a sentence needs to be 4 times the size of the word embedding size and number of cells put together.

That is, in the case of BID15 with 1000 word embedding and 1000 cells, the sentence length needs to be 4 × (1000 + 1000) = 8000!

In practice, most NLP problems whose input involves sentences, the length will be typically less than 100.

In our experiment, sequence to sequence with attention BID16 for maximum sentence length 100 (other model settings please refer to TAB1 ), ELSTM-I parameters uses 75M of memory, ELSTM-II uses 69.1M, LSTM uses 71.5M, and GRU uses 65.9M.

Through GPU parallelization, the computational time for all four cells are almost identical with 0.4 seconds per step time on a GeForce GTX TITAN X GPU.

We investigate the macro RNN model and propose a multitask model called dependent BRNN (DBRNN) in this section.

The model is tasked to predict a output sequence DISPLAYFORM0 , where T and T are the length of the input and output sequence respectively.

Our proposal is inspired by the pros and cons of two RNN modelsthe bidirectional RNN (BRNN) model BID14 and the encoder-decoder model .

In the following, we will first examine the BRNN and the encoder-decoder in Sec. 4.1 and, then, propose the DBRNN in Sec. 4.2.

BRNN is modeling the conditional probability density function: DISPLAYFORM0 ).

This output is a combination of the output of a forward and a backward RNN.

Due to this bidirectional design, the BRNN can fully utilize the information of the entire input sequence to predict each individual output element.

On the other hand, the BRNN does not utilize the predicted output in predicting Y t .

This makes elements in the predicted sequenceŶ t = argmax Yt P (Y t |{X i } T t=1 ) independent of each other.

).

However, the encoder-decoder model is vulnerable to previous erroneous predictions in the forward path.

Recently, the BRNN has been introduced in the encoder by BID0 , yet this design still does not address the erroneous prediction problem.

Being motivated by observations in Sec. 4.1, we propose a multitask RNN model called DBRNN to fulfill the following objectives: DISPLAYFORM0 DISPLAYFORM1 where W f and W b are trainable weights.

DISPLAYFORM2 ).

The DBRNN has three learning objectives: the target sequence for the forward RNN prediction, the reversed target sequence for the backward RNN prediction, and, finally, the target sequence for the bidirectional prediction.

The DBRNN model is shown in FIG2 .

It consists of a lower and an upper BRNN branches.

At each time step, the input to the forward and the backward parts of the upper BRNN is the concatenated forward and backward outputs from the lower BRNN branch.

The final bidirectional prediction is the pooling of both the forward and backward predictions.

We will show later that this design will make DBRNN robust to previous erroneous predictions.

DISPLAYFORM3 where c denotes the cell hidden state and l denotes the lower BRNN.

The final output, h t , of the lower BRNN is the concatenation of the output, h f t , of the forward RNN and the output, h b t , of the backward RNN.

Similarly, the upper BRNN generates the final output p t as DISPLAYFORM4 where u denotes the upper BRNN.

To generate forward predictionŶ There are three errors: prediction error ofŶ f t denoted by e f , prediction error ofŶ b t denoted by e b and prediction error ofŶ t denoted by e. To train this network, e f is back propagated through time to the upper forward RNN and the lower BRNN, e b is back propagated through time to the upper backward RNN and the lower BRNN, and e is back propagated through time to the entire model.

To show that DBRNN is more robust to previous erroneous predictions than one-directional models, we compare the cross entropy of them as follows: DISPLAYFORM5 where K is the total number of classes (e.g. the size of vocabulary for language tasks).

p t is the ground truth distribution which is an one-hot vector such that: p tk = I(p tk = k ), ∀k ∈ 1, ..., K, where I is the indicator function, k is the ground truth label of the tth output.p t is the predicted distribution.

From Eq. 26, l can be further expressed as: DISPLAYFORM6 DISPLAYFORM7 We can pick W It is worthwhile to compare the DBRNN and the solution in BID5 .

Both of them have a bidirectional design for the output.

However, there exist three main differences.

First, the DBRNN is a general design for the sequence-in-sequence-out (SISO) problem without being restricted to dependency parsing.

The target sequences in trainingŶ f t ,Ŷ b t andŶ t are the same for the DBRNN.

In contrast, the solution in BID5 has different target sequences.

Second, the attention mechanism is used by BID5 but not in the DBRNN.

Third, The encoder-decoder design is adopted by in BID5 but not in the DBRNN.

We conduct experiments on two problems: part of speech (POS) tagging and dependency parsing (DP).

The POS tagging task is an easy one which requires shorter memory while the DP task needs much longer memory and has more complex relations between the input and the output.

In the experiments, we compare the performance of five RNN models under two scenarios: 1) I t = X t , and 2) I The training dataset used for both problems are from the Universal Dependency 2.0 English branch (UD-English).

It contains 12543 sentences and 14985 unique tokens.

The test dataset for both experiments is from the test English branch (gold, en.conllu) of CoNLL 2017 shared task development and test data.

In the experiment, the lengths of the input and the target sequences are fixed.

Sequences longer than the maximum length will be truncated.

If the sequence is shorter than the fixed length, a special pad symbol will be used to pad the sequence.

Similar technique called bucketing is also used for some popular models such as BID15 .

The input to the POS tagging and the DP problems are the stemmed and lemmatized sequences (column 3 in CoNLL-U format).

The target sequence for POS tagging is the universal POS tag (column 4).

The target sequence for DP is the interleaved dependency relation to the headword (relation, column 8) and its position (column 7).

As a result, the length of the actual target sequence (rather than the preprocessed fixed-length sequence) for DP is twice of the length of the actual input sequence.

The input is first fed into a trainable embedding layer BID4 before it is sent to the actual network.

TAB1 shows the detailed network and training specifications.

It is important to point out that we do not finetune network parameters or apply any engineering trick for the best possible performance since our main goal is to compare the performance of the LSTM, GRU, ELSTM-I and ELSTM-II four cells under various macro-models.

The results of the POS tagging problem with I t = X t and I BID14 88.49 82.84 79.14 Seq2seq BID15 25.83 24.87 31.43 Seq2seq with Attention BID16 27 The results of the DP problem with I t = X t and I TAB5 , respectively.

The ELSTM-I and ELSTM-II cells perform better than the LSTM and the GRU cells.

Among all possible combinations, the sequence-to-sequence with attention combined with ELSTM-I has the best performance.

It has an accuracy of 60.19% and 66.72% for the former and the latter, respectively.

Also, the basic RNN often outperforms BRNN for the DP problem as shown in TAB5 .

This can be explained by that the basic RNN can access the entire input sequence when predicting the latter half of the output sequence since the target sequence is twice as long as the input.

The other reason is that the BRNN can easily overfit when predicting the headword position.

We see from Tables 3 -6 that the two DBRNN models outperform both BRNN and sequence-tosequence (without attention) in both POS tagging and DP problems regardless of used cells.

This shows the superiority of introducing the expert opinion pooling from both the input and the predicted output.

DISPLAYFORM0 Furthermore, the proposed ELSTM-I and ELSTM-II outperform the LSTM and the GRU by a significant margin for complex language tasks.

This demonstrates that the scaling factor in the ELSTM-I and the ELSTM-II does help the network retain longer memory with better attention.

ELSTMs even outperform BID5 , which is designed specifically for DP.

For the POS tagging problem, the ELSTM-I and the ELSTM-II do not perform as well as the GRU or the LSTM.

This is probably due to the shorter memory requirement of this simple task.

The ELSTM cells are over-parameterized and, as a result, they converge slower and tend to overfit the training data.

The ELSTM-I and the ELSTM-II perform particularly well for sequence-to-sequence (with and without attention) model.

The hidden state c t of the ELSTMs is more expressive in representing patterns over a longer distance.

Since the sequence-to-sequence design relies on the expressive power of the hidden state, the ELSTMs do have an advantage.

We compare the convergence behavior of I t = X t and I DISPLAYFORM1 with the LSTM, the ELSTM-I and the ELSTM-II cells for the DP problem in FIG6 .

We see that the ELSTM-I and the ELSTM-II do not behave very differently between I t = X t and I

The memory decay behavior of the LSTM and the GRU was investigated and explained by mathematical analysis.

Although the memory of the LSTM and the GRU fades slower than that of the SRN, it may not be long enough for complicated language tasks such as dependency parsing.

To enhance the memory length, two cells called the ELSTM-I and the ELSTM-II were proposed.

Furthermore, we introduced a new RNN model called the DBRNN that has the merits of both the BRNN and the encoder-decoder.

It was shown by experimental results that the ELSTM-I and ELSTM-II outperforms other designs by a significant margin for complex language tasks.

The DBRNN design is superior to BRNN as well as sequence-to-sequence models for both simple and complex language tasks.

There are interesting issues to be further explored.

For example, is the ELSTM cell also helpful in more sophisticated RNN models such as the deep RNN?

Is it possible to make the DBRNN deeper and better?

They are left for future study.

<|TLDR|>

@highlight

A recurrent neural network cell with extended-long short-term memory and a multi-task RNN model for sequence-in-sequence-out problems