Neural networks are known to produce unexpected results on inputs that are far from the training distribution.

One approach to tackle this problem is to detect the samples on which the trained network can not answer reliably.

ODIN is a recently proposed method for out-of-distribution detection that does not modify the trained network and achieves good performance for various image classification tasks.

In this paper we adapt ODIN for sentence classification and word tagging tasks.

We show that the scores produced by ODIN can be used as a confidence measure for the predictions on both in-distribution and out-of-distribution datasets.

The majority of the sentences in English-LinES treebank are from literature.

English-EWT dataset is 89 larger and is more diverse.

The datasets are described in

Let s be a sentence from one of the datasets, and w 1 , . . .

, w M be the words.

The embedding of 97 the m-th word of the sentence will be x m = W e hash(w m ).

We apply bidirectional LSTM on the DISPLAYFORM0 .

For sentiment analysis we apply a dense layer on the 99 concatenation of the last states of the two LSTMs: DISPLAYFORM1 is a cross-entropy: loss(s) = ce(S(f sc (s), 1)), where DISPLAYFORM2 is the modified 101 softmax function, T is the temperature scaling parameter, and C is the number of classes.

For POS tagging we apply a dense layer on every hidden state: and ODIN(s) = max S(f sc (x), T ), wherex = x + sign(∇ x Sŷ(x))), whereŷ = argmax S(x, 1).

DISPLAYFORM3

Here (perturbation magnitude) and T (temperature) are hyperparameters, which are chosen based 109 on the OOD detection performance on the development sets.

For POS tagging, the gradient in the

ODIN score formula is applied to the mean of word-level probability maximums.

TAB3 shows the results for OOD detection and Table 4 shows the rank correlation coefficients for

PbThreshold and ODIN methods.

The role of the temperature scaling and input perturbations All our experiments confirm the 129 observation from BID3 ] that temperature scaling improves out-of-distribution detection.

The effect of higher temperatures saturates when T reaches thousands (Figure 1) .

The positive effect 131 of the perturbations on the inputs is visible for sentiment analysis, but not for POS tagging.

We

Ranking of the sentences ODIN is clearly better than PbThreshold according to Spearman's rank 141 correlation coefficient for POS tagging tasks (Table 4) .

For a neural network trained on en-LinES,

ODIN scores are a good indicator how the network will perform on OOD samples.

It is a much ODIN) .

and T for ODIN are determined based on the development sets of en-LinES (ID) and en-EWT (OOD).

The size of a circle is proportional to the number of samples that fall into that bucket.

Ideally, accuracy scores for the i-th bucket should be higher than for the (i − 1)-th bucket, and y coordinates of the three circles for each bucket should be the same.

In this work we have adapted ODIN out-of-distribution detection method on sentence classification 163 and sequence tagging tasks.

We showed that as an OOD detector it performs consistently better than 164 for the PbThreshold baseline.

Additionally, we attempted to quantify how well the scores produced 165 by these methods can be used as confidence scores for the predictions of neural models.

There are many other OOD detection methods that have yet to be tested on NLP tasks.

On the other 167 hand, our analysis notably doesn't cover sequence-to-sequence tasks.

We have shown that the usage

<|TLDR|>

@highlight

A recent out-of-distribution detection method helps to measure the confidence of RNN predictions for some NLP tasks