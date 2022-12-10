In this paper, we present Neural Phrase-based Machine Translation (NPMT).

Our method explicitly models the phrase structures in output sequences using Sleep-WAke Networks (SWAN), a recently proposed segmentation-based sequence modeling method.

To mitigate the monotonic alignment requirement of SWAN, we introduce a new layer to perform (soft) local reordering of input sequences.

Different from existing neural machine translation (NMT) approaches, NPMT does not use attention-based decoding mechanisms.

Instead, it directly outputs phrases in a sequential order and can decode in linear time.

Our experiments show that NPMT achieves superior performances on IWSLT 2014 German-English/English-German and IWSLT 2015 English-Vietnamese machine translation tasks compared with strong NMT baselines.

We also observe that our method produces meaningful phrases in output languages.

A word can be considered as a basic unit in languages.

However, in many cases, we often need a phrase to express a concrete meaning.

For example, consider understanding the following sentence, "machine learning is a field of computer science".

It may become easier to comprehend if we segment it as " [machine learning] [is] [a field of] [computer science]", where the words in the bracket '[]' are regarded as "phrases".

These phrases have their own meanings, and can often be reused in other contexts.

The goal of this paper is to explore the use of phrase structures aforementioned for neural networkbased machine translation systems BID22 BID0 .

To this end, we develop a neural machine translation method that explicitly models phrases in target language sequences.

Traditional phrase-based statistical machine translation (SMT) approaches have been shown to consistently outperform word-based ones (Koehn et al., 2003; Koehn, 2009; BID15 .

However, modern neural machine translation (NMT) methods BID22 BID0 do not have an explicit treatment on phrases, but they still work surprisingly well and have been deployed to industrial systems BID31 BID28 .

The proposed Neural Phrase-based Machine Translation (NPMT) method tries to explore the advantages from both kingdoms.

It builds upon Sleep-WAke Networks (SWAN), a segmentation-based sequence modeling technique described in BID25 , where segments (or phrases) are automatically discovered given the data.

However, SWAN requires monotonic alignments between inputs and outputs.

This is often not an appropriate assumption in many language pairs.

To mitigate this issue, we introduce a new layer to perform (soft) local reordering on input sequences.

Experimental results show that NPMT outperforms attention-based NMT baselines in terms of the BLEU score BID19 on IWSLT 2014 German-English/English-German and IWSLT 2015 English-Vietnamese translation tasks.

We believe our method is one step towards the full integration of the advantages from neural machine translation and phrase-based SMT.

This paper is organized as follows.

Section 2 presents the neural phrase-based machine translation model.

Section 3 demonstrates the usefulness of our approach on several language pairs.

We conclude our work with some discussions in Section 4.

We first give an overview of the proposed NPMT architecture and some related work on incorporating phrases into NMT.

We then describe the two key building blocks in NPMT: 1) SWAN, and 2) the soft reordering layer which alleviates the monotonic alignment requirement of SWAN.

In the context of machine translation, we use "segment" and "phrase" interchangeably.

Figure 1(a) shows the overall architecture of NPMT.

The input sequence is first turned into embedding representations and then they go through a (soft) reordering layer (described below in Section 2.3).

We then pass these "reordered" activations to the bi-directional RNN layers, which are finally fed into the SWAN layer to directly output target language in terms of segments (or phrases).

While it is possible to replace bi-directional RNN layers with other layers BID8 , in this paper, we have only explored this particular setting to demonstrate our proposed idea.

There have been several works that propose different ways to incorporate phrases into attentionbased neural machine translation, such as BID23 BID26 ; BID6 .

These approaches typically use predefined phrases (obtained by external methods, e.g., phrase-based SMT) to guide or modify the existing attention-based decoder.

The major difference from our approach is that, in NPMT, we do not use attention-based decoding mechanisms, and our phrase structures for the target language are automatically discovered from the training data.

Another line of related work is the segment-to-segment neural transduction model (SSNT) , which shows promising results on a Chinese-to-English translation task under a noisy channel framework BID30 .

In SSNT, the segments are implicit, and the monotonic alignments between the inputs and outputs are achieved using latent variables.

The latent variables are marginalized out during training using dynamic programming.

BID25 .

Symbol $ indicates the end of a segment.

Given a sequence of inputs x1, . . .

, x5, which is from the outputs from the bi-directional RNN of FIG0 (a), SWAN emits one particular segmentation of y1:3 = π(a1:5), where DISPLAYFORM0 Here x1 wakes (emitting segment a1) and x4 wakes (emitting segment a4) while x2, x3 and x5 sleep (emitting empty segments a2, a3 and a5 respectively).

Here we review the SWAN model proposed in BID25 .

SWAN defines a probability distribution for the output sequence given an input sequence.

It models all valid output segmentations of the output sequence as well as the monotonic alignments between the output segments and the input sequence.

Empty segments are allowed in the output segmentations.

It does not make any assumption on the lengths of input or output sequence.

Assume input sequence for SWAN is x 1:T , which is the outputs from bi-directional RNN of FIG0 (a), and output sequence is y 1:T .

Let S y denote the set containing all valid segmentations of y 1:T , with the constraint that the number of segments in a segmentation is the same as the input sequence length, T .

Let a t denote a segment or phrase in the target sequence.

Empty segments are allowed to ensure that we can correctly align segment a t to input element x t .

Otherwise, we might not have a valid alignment for the input and output pair.

See FIG1 for an example of the emitted segmentation of y 1:T .

The probability of the sequence y 1:T is defined as the sum of the probabilities of all the segmentations in S y {a 1:T : π(a 1:T ) = y 1:T }, DISPLAYFORM0 where the p(a t |x t ) is the segment probability given input element x t , which is modeled using a recurrent neural network (RNN) with an additional softmax layer.

π(·) is the concatenation operator and the symbol $, end of a segment, is ignored in the concatenation operator π(·).

(An empty segment, which only contains $ will thus be ignored as well.)

SWAN can be also understood via a generative model, 1.

For t = 1, ..., T : (a) Given an initial state of x t , sample words from RNN until we reach an end of segment symbol $.

This gives us a segment a t .

2.

Concatenate {a 1 , ..., a T } to obtain the output sequence via π(a 1:T ) = y 1:T .Since there are more than one way to obtain the same y 1:T using the generative process above, the probability of observing y 1:T is obtained by summing over all possible ways, which is Eq. 1.Note that |S y | is exponentially large, direct summation quickly becomes infeasible when T or T is not small.

Instead, BID25 developed an exact dynamic programming algorithm to tackle the computation challenges.

3 The key idea is that although the number of possible segmentations is exponentially large, the number of possible segments is polynomial-O(T 2 ).

In other words, it is possible to first compute all possible segment probabilities, p(a t |x t ), ∀a t , x t , and then use dynamic programming to calculate the output sequence probability p(y 1:T |x 1:T ) in Eq. (1).

The feasibility of using dynamic programming is due to a property of segmentations-a segmentation of a subsequence is also part of the segmentation of the entire sequence.

In practice, a maximum length L for a segment a t is enforced to reduce the computational complexity, since the length of useful segments is often not very long.

BID25 also discussed a way to carry over information across segments using a separate RNN, which we will not elaborate here.

We refer the readers to the original paper for the algorithmic details.

SWAN defines a conditional probability for an output sequence given an input one.

It can be used in many sequence-to-sequence tasks.

In practice, a sequence encoder like a bi-directional RNN can be used to process the raw input sequence (like speech signals or source language) to obtain x 1:T that is to be passed into SWAN for decoding.

For example, BID25 demonstrated the usefulness of SWAN in the context of speech recognition.

Greedy decoding for SWAN is straightforward.

We first note that p(a t |x t ) is modeled as an RNN with an additional softmax layer.

Given each p(a t |x t ), ∀t ∈ 1, . . .

, T , is independent of each other, we can run the RNN in parallel to produce an output segment (possibly empty) for each p(a t |x t ).

We then concatenate these output segments to form the greedy decoding of the entire output sequence.

The decoding satisfies the non-autoregressive property BID9 and the decoding complexity is O(T L).

See BID25 for the algorithmic details of the beam search decoder.

We finally note that, in SWAN (thus in NPMT), only output segments are explicit; input segments are implicitly modeled by allowing empty segments in the output.

This is conceptually different from the traditional phrase-based SMT where both inputs and outputs are phrases (or segments).

We leave the option of exploring explicit input segments as future work.

SWAN assumes a monotonic alignment between the output segments and the input elements.

For speech recognition experiments in BID25 , this is a reasonable assumption.

However, for machine translation, this is usually too restrictive.

In neural machine translation literature, attention mechanisms were proposed to address alignment problems BID0 BID20 BID24 .

But it is not clear how to apply a similar attention mechanism to SWAN due to the use of segmentations for output sequences.

One may note that in NPMT, a bi-directional RNN encoder for the source language can partially mitigate the alignment issue for SWAN, since it can access every source word.

However, from our empirical studies, it is not enough to obtain the best performance.

Here we augment SWAN with a reordering layer that does (soft) local reordering of the input sequence.

This new model leads to promising results on the IWSLT 2014 German-English/English-German, and IWSLT 2015 EnglishVietnamese machine translation tasks.

One additional advantage of using SWAN is that since SWAN does not use attention mechanisms, decoding can be done in parallel with linear complexity, as now we remove the need to query the entire input source for every output word BID20 BID9 .We now describe the details of the local reordering layer shown in FIG2 .

Denote the input to the local reordering layer by e 1:T , which is the output from the word embedding layer of FIG0 , and the output of this layer by h 1:T , which is fed as inputs to the bi-directional RNN of FIG0 .

We compute h t as DISPLAYFORM0 where σ(·) is the sigmoid function, and 2τ + 1 is the local reordering window size.

weight of e t−τ +i through the gate σ w T i [e t−τ ; . . . ; e t ; . . . ; e t+τ ] .

The final output h t is a weighted linear combination of the input elements, e t−τ , . . .

, e t , . . . , e t+τ , in the window followed by a nonlinear transformation by the tanh(·) function.

Figure 3(b) illustrates how local reordering works.

Here we want to (softly) select an input element from a window given all information available in this window.

Suppose we have two adjacent windows, (e 1 , e 2 , e 3 ) and (e 2 , e 3 , e 4 ).

If e 3 gets the largest weight (e 3 is picked) in the first window and e 2 gets the largest weight (e 2 is picked) in the second window, e 2 and e 3 are effectively reordered.

Our layer is different from the attention mechanism BID0 BID20 BID24 in following ways.

First, we do not have a query to begin with as in standard attention mechanisms.

Second, unlike standard attention, which is top-down from a decoder state to encoder states, the reordering operation is bottom-up.

Third, the weights {w i } 2τ i=0 capture the relative positions of the input elements, whereas the weights are the same for different queries and encoder hidden states in the attention mechanism (no positional information).

The reordering layer performs locally similar to a convolutional layer and the positional information is encoded by a different parameter w i for each relative position i in the window.

Fourth, we do not normalize the weights for the input elements e t−τ , . . .

, e t , . . . , e t+τ .

This provides the reordering capability and can potentially turn off everything if needed.

Finally, the gate of any position i in the reordering window is determined by all input elements e t−τ , . . .

, e t , . . . , e t+τ in the window.

We provide a visualizing example of the reordering layer gates that performs input swapping in Appendix A.One related work to our proposed reordering layer is the Gated Linear Units (GLU) which can control the information flow of the output of a traditional convolutional layer.

But GLU does not have a mechanism to decide which input element from the convolutional window to choose.

From our experiments, neither GLU nor traditional convolutional layer helped our NPMT.

Another related work to the window size of the reordering layer is the distortion limit in traditional phrase-based statistical machine translation methods BID2 .

Different window sizes restrict the context of each position to different numbers of neighbors.

We provide an empirical comparison of different window sizes in Appendix B.

In this section, we evaluate our model on the IWSLT 2014 German-English BID3 , IWSLT 2014 English-German, and IWSLT 2015 English-Vietnamese BID4 machine translation tasks.

We note that, in this paper, we limit the applications of our model to relatively small datasets to demonstrate the usefulness of our method.

We plan to conduct more large scale experiments in future work.

MIXER BID21 20.73 21.83 LL BID27 22.53 23.87 BSO BID27 23.83 25.48 LL BID1 25

We evaluate our model on the German-English machine translation track of the IWSLT 2014 evaluation campaign BID3 .

The data comes from translated TED talks, and the dataset contains roughly 153K training sentences, 7K development sentences, and 7K test sentences.

We use the same preprocessing and dataset splits as in Ranzato et al. We report our IWSLT 2014 German-English experiments using one reordering layer with window size 7, two layers of bi-directional GRU encoder (Gated recurrent unit, BID5 ) with 256 hidden units, and two layers of unidirectional GRU decoder with 512 hidden units.

We add dropout with a rate of 0.5 in the GRU layer.

We choose GRU since baselines for comparisons were using GRU.

The maximum segment length is set to 6.

Batch size is set as 32 (per GPU) and the Adam algorithm BID13 ) is used for optimization with an initial learning rate of 0.001.

For decoding, we use greedy search and beam search with a beam size of 10.

As reported in BID18 ; BID1 , we find that penalizing candidate sentences that are too short was required to obtain the best results.

We add the middle term of Eq. (3) to encourage longer candidate sentences.

All hyperparameters are chosen based on the development set.

NPMT takes about 2-3 days to run to convergence (40 epochs) on a machine with four M40 GPUs.

The results are summarized in Table 1 .

In addition to previous reported baselines in the literature, we also explored the best hyperparameter using the same model architecture (except the reordering layer) using sequence-to-sequence model with attention as reported as LL * of Table 1 .NPMT achieves state-of-the-art results on this dataset as far as we know.

Compared to the supervised sequence-to-sequence model, LL BID1 , NPMT achieves 2.4 BLEU gain in the greedy setting and 2.25 BLEU gain using beam-search.

Our results are also better than those from the actor-critic based methods in BID1 .

But we note that our proposed method is orthogonal to the actor-critic method.

So it is possible to further improve our results using the actor-critic method.

We also run the following two experiments to verify the sources of the gain.

The first is to add a reordering layer to the original sequence-to-sequence model with attention, which gives us BLEU scores of 25.55 (greedy) and 26.91 (beam search).

Since the attention mechanism and reordering layer capture similar information, adding the reordering layer to the sequence-to-sequence model with attention does not improve the performance.

The second is to remove the reordering layer from NPMT, which gives us BLEU scores of 27.79 (greedy) and 29.28 (beam search).

This shows that the reordering layer and SWAN are both important for the effectiveness of NPMT. .

target ground truth there are tens of thousands of machines around the world that make small pieces of dna -30 to 50 letters -in length -and it 's a UNK process , so the longer you make the piece , the more errors there are .

Table 2 : Examples of German-English translation outputs with their segmentations.

We label the indexes of the words in the source sentence and we use those indexes to indicate where the output segment is emitted.

For example, in greedy decoding results, " i word1, . . .

, wordm" denotes i-th word in the source sentence emits words word1, . . .

, wordm during decoding (assuming monotonic alignments).

The "•" represents the segment boundary in the target output.

See Figure 4 for a visualization of row 1 in this table.

In greedy decoding, we can estimate the average segment length 4 for the output.

The average segment length is around 1.4-1.6, indicating phrases with more than one word are being decoded.

Figure 4 shows an example of the input and decoding results with NPMT.

We can observe phraselevel translation being captured by the model (e.g., "danke" → "thank you").

The model also knows when to sleep before outputting a phrase (e.g., "das" → "$").

We use the indexes of words in the source sentence to indicate where the output phrases are from.

Table 2 shows some sampled exam-ples.

We can observe there are many informative segments in the decoding results, e.g., "tens of thousands of", "the best thing", "a little", etc.

There are also mappings from phrase to phrase, word to phrases, and phrase to word in the examples.

Following the analysis, we show the most frequent phrase mappings in Appendix C.We also explore an option of adding a language-model score during beam search as the traditional statistical machine translation does.

This option might not make much sense in attention-based approaches, since the decoder itself is usually a neural network language model.

In SWAN, however, there is no language models directly involved in the segmentation modeling, 5 and we find it useful to have an external language model during beam search.

We use a 4th-order language model trained using the KenLM implementation BID11 for English target training data.

So the final beam search score we use is DISPLAYFORM0 where we empirically find that λ 1 = 1.2 and λ 2 = 0.2 give good performance, which are tuned on the development set.

The results with the external language model are denoted by NPMT+LM in Table 1 .

If no external language models are used, we set λ 2 = 0.

This scoring function is similar to the one for speech recognition in .

We also evaluate our model on the opposition direction, English-German, which translates from a more segmented text to a more inflectional one.

Following the setup in Section 3.1, we use the same dataset with the opposite source and target languages.

We use the same model architecture, optimization algorithm and beam search size as the German-English translation task.

NPMT takes about 2-3 days to run to convergence (40 epochs) on a machine with four M40 GPUs.

Given there is no previous sequence-to-sequence attention model baseline for this setup, we create a strong one and tune hyperparameters on the development set.

The results are shown in TAB4 .

Based on the development set, we set λ 1 = 1 and λ 2 = 0.15 in Eq. (3).

Our model outperforms sequence-to-sequence model with attention by 2.46 BLEU and 2.49 BLEU in greedy and beam search cases.

We can also use a 4th-order language model trained using the KenLM implementation for German target training data, which further improves the performance.

Some sampled examples are shown in TAB6 .

Several informative segments/phrases can be found in the decoding results, e.g., "some time ago" → "vor enniger zeit".

In this section, we evaluate our model on the IWSLT 2015 English to Vietnamese machine translation task.

The data is from translated TED talks, and the dataset contains roughly 133K training sentence pairs provided by the IWSLT 2015 Evaluation Campaign BID4 .

Following the same preprocessing steps in ; BID20 We use one reordering layer with window size 7, two layers of bi-directional LSTM (Long shortterm memory, BID12 ) encoder with 512 hidden units, and three layers of unidirectional LSTM decoder with 512 hidden units.

We add dropout with a rate of 0.4 in the LSTM layer.

We choose LSTM since baselines for comparisons were using LSTM.

The maximum segment length is set to 7.

Batch size is set as 48 (per GPU) and the Adam algorithm BID13 is used for optimization with an initial learning rate of 0.001.

For decoding, we use greedy decoding and beam search with a beam size of 10.

The results are shown in TAB8 .

Based on the development set, we set λ 1 = 0.7 and λ 2 = 0.15 in Eq. (3).

NPMT takes about one day to run to convergence (15 epochs) on a machine with 4 M40 GPUs.

Our model outperforms sequence-tosequence model with attention by 1.41 BLEU and 1.59 BLEU in greedy and beam search cases.

We also use a 4th-order language model trained using the KenLM implementation for Vietnamese target training data, which further improves the BLEU score.

Note that our reordering layer relaxes the monotonic assumption as in BID20 and is able to decode in linear time.

Empirically we outperform models with monotonic attention.

Table 6 shows some sampled examples.

Hard monotonic BID20 23.00 -Luong & Manning FORMULA1 FIG4 , we show an example that translates from "can you translate it ?" to "können man esübersetzen ?", where the mapping between words are as follows: "can → können", "you → man", "translate →übersetzen", "it → es" and "?

→ ?".

Note that the example needs to be reordered from "translate it" to "esübersetzen".

Each row of FIG4 represents a window of size 7 that is centered at a source sentence word.

The values in the matrix represent the gate values for the corresponding words.

The gate values will later be multiplied with the embedding e t−τ +i of Eq. (2) and contribute to the hidden vector h t .

The y-axis represents the word/phrases emitted from the corresponding position.

We can observe that the gates mostly focus on the central word since the first part of the sentence only requires monotonic alignment.

Interestingly, the model outputs "$" (empty) when the model has the word "translate" in the center of the window.

Then, the model outputs "es" when the model encounters "it".

Finally, in the last window (top row), the model not only has a large gate value to the center input "?", but the model also has a relatively large gate value to the word "translate" in order to output the translation "übersetzen ?".

This shows an example of the reordering effect achieved by using the gating mechanism of the reordering layer.

In this section, we examine the effect of window sizes in the reordering layer.

Following the setup in Section 3.2, we evaluate the performance of different window sizes on the IWSLT 2014 EnglishGerman translation task.

TAB9 summarizes the results.

We can observe that the performance reaches the peak with a windows size of 7.

With a window size of 5, the performance drops 0.88 BLEU in greedy decoding and 0.72 BLEU using beam search.

It suggests that the context window is not large enough to properly perform reordering.

When the window sizes are 9 and 11, we do not observe further improvements.

It might be because the translation between English and German mostly requires local word reordering.

Following the examples of Table 2 , we analyze the decoding results on the test set of the GermanEnglish translation task.

Given we do not have explicit input segments in NPMT, we assume input words that emit "$" symbol are within the same group as the next non-'$' word.

For example, in Figure 4 , input words "das beste" are considered as an input segment.

We then can aggregate all the input, output segments (phrases) and sort them based on the frequency.

Tables C and C show UNK → the UNK in der → in der UNK → the UNK in diesem → in this und → and und → , and UNK .

→ .

ein UNK →

a UNK die welt → the world UNK → UNK das → this is UNK , → , das UNK → the UNK ist es → it 's aber → but das, → that 's , die → that eine UNK → a UNK " .

→ . " " → " UNK → a UNK ist .

→ .

in UNK → in UNK ein paar → a few ist → is ich →

i think in den →

in den UNK → the UNK gibt es → there 's der → of es →

it was ist , → , wissen sie →

you know der welt →

the world von → of dies →

this is sind .

→ .

in diesem → in this die frage → the question mit → with es → there 's , wenn → if dem UNK → the UNK haben wir →

we have Table 8 : German-English phrase mapping results.

We show the top 10 input, output phrase mappings in five categories ("One" stands for single word and "Many" stands for multiple words.).

In the last column, Many →

Many * , we remove the phrases with the "UNK" word as the "UNK" appears often.

Phrases with 3 words Phrases with 4 words auf der ganzen → all over the auf der ganzen → a little bit of gibt eine menge → a lot of weiß nicht , was → what 's going to be dann hat er→

he doesn 't have tun , das wir →

we can 't do , die man → you can do tat , das ich →

i didn 't do das können wir →

we can do that zu verbessern , die → that can be done Table 9 : German-English longer phrase mapping results.

We show the top 5 input, output phrase mappings for two categories: input and output phrases with three words, and input and output phrases with four words.

@highlight

Neural phrase-based machine translation with linear decoding time