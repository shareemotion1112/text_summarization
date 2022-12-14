Structural planning is important for producing long sentences, which is a missing part in current language generation models.

In this work, we add a planning phase in neural machine translation to control the coarse structure of output sentences.

The model first generates some planner codes, then predicts real output words conditioned on them.

The codes are learned to capture the coarse structure of the target sentence.

In order to learn the codes, we design an end-to-end neural network with a discretization bottleneck, which predicts the simplified part-of-speech tags of target sentences.

Experiments show that the translation performance are generally improved by planning ahead.

We also find that translations with different structures can be obtained by manipulating the planner codes.

When human speaks, it is difficult to ensure the grammatical or logical correctness without any form of planning.

Linguists have found evidence through speech errors or particular behaviors that indicate speakers are planning ahead BID16 .

Such planning can happen in discourse or sentence level, and sometimes we may notice it through inner speech.

In contrast to human, a neural machine translation (NMT) model does not have the planning phase when it is asked to generate a sentence.

Although we can argue that the planning is done in the hidden layers, however, such structural information remains uncertain in the continuous vectors until the concrete words are sampled.

In tasks such as machine translation, a source sentence can have multiple valid translations with different syntactic structures.

As a consequence, in each step of generation, the model is unaware of the "big picture" of the sentence to produce, resulting in uncertainty of word prediction.

In this research, we try to let the model plan the coarse structure of the output sentence before decoding real words.

As illustrated in FIG0 , in our proposed framework, we insert some planner codes into the beginning of the output sentences.

The sentence structure of the translation is governed by the codes.

An NMT model takes an input sentence X and produce a translation Y .

Let S Y denotes the syntactic structure of the translation.

Indeed, the input sentence already provides rich information about the target-side structure S Y .For example, given the Spanish sentence in FIG0 , we can easily know that the translation will have a noun, a pronoun and a verb.

Such obvious structural information does not have uncertainty, and thus does not require planning.

In this example, the uncertain part is the order of the noun and the pronoun.

Thus, we want to learn a set of planner codes C Y to disambiguate such uncertain information about the sentence structure.

By conditioning on the codes, we can potentially increase the effectiveness of beam search as the search space is properly regulated.

In this work, we use simplified POS tags to annotate the structure S Y .

We learn the planner codes by putting a discretization bottleneck in an end-to-end network that reconstructs S Y with both X and C Y .

The codes are merged with the target sentences in the training data.

Thus, no modification to the NMT model is required.

Experiments show the translation performance is generally improved with structural planning.

More interestingly, we can control the structure of output sentences by manipulating the planner codes.

In this section, we first extract the structural annotation S Y by simplifying the POS tags.

Then we explain the code learning model for obtaining the planner codes.

To reduce uncertainty in the decoding phase, we want a structural annotation that describes the "big picture" of the sentence.

For instance, the annotation can tell whether the sentence to generate is in a "NP VP" order.

The uncertainty of local structures can be efficiently solved by beam search or the NMT model itself.

In this work, we extract such coarse structural annotations S Y through a simple two-step process that simplifies the POS tags of the target sentence:1.

Remove all tags other than "N", "V", "PRP", "," and ".".

Note that all tags begin with "N" (e.g. NNS) are mapped to "N", and tags begin with "V" (e.g. VBD) are mapped to "V".

The following list gives an example of the process:Input: He found a fox behind the wall.

Step 1: PRP V N N .Step 2: PRP V N .Note that many other annotations can also be considered to represent the syntactic structure, which is left for future work to explore.

Next, we learn the planner codes C Y to remove the uncertainty of the sentence structure S Y when producing a translation.

For simplicity, we use the notion S and C to replace S Y and C Y in this section.

DISPLAYFORM0 Architecture of the code learning model.

The discretization bottleneck is shown as the dashed lines.

We first compute the discrete codes C 1 , .., C N based on simplified POS tags S 1 , ..., S T : DISPLAYFORM1 DISPLAYFORM2 where the tag sequence S 1 , ..., S T is firstly encoded using a backward LSTM BID4 .

E(??) denotes the embedding function.

Then, we compute a set of vectorsC 1 , ...,C N , which are latterly discretized in to approximated one-hot vectors C 1 , ..., C N using Gumbel-Softmax trick BID5 BID10 .

We then combine the information from X and C to initialize a decoder LSTM that sequentially predicts S 1 , ..., S T : DISPLAYFORM3 DISPLAYFORM4 where [C 1 , ..., C N ] denotes a concatenation of N one-hot vectors.

Note that only h t is computed with a forward LSTM.

Both f enc and f dec are affine transformations.

Finally, we predict the probability of emitting each tag S t with DISPLAYFORM5 The architecture of the code learning model is depicted in Fig. 2 , which can be seen as a sequence auto-encoder with an extra context input X to the decoder.

The parameters are optimized with crossentropy loss.

Once the code learning model is trained, we can obtain the planner codes C for all target sentences in the training data using the encoder part.

The training data of machine translation dataset is composed of (X, Y ) sentence pairs.

With the planner codes C Y we obtained, our training data now becomes a list of (X, C Y ; Y ) pairs.

As shown in FIG0 , we connect the planner codes and target sentence with a " eoc " token.

With the modified dataset, we train a regular NMT model.

We use beam search when decoding sentences, thus the planner codes are searched before emitting real words.

The codes are removed from the translation results during evaluation.

Recently, some methods are proposed to improve the syntactic correctness of the translations.

BID19 restricts the search space of the NMT decoder using the lattice produced by a Statistical Machine Translation system.

BID2 takes a multi-task approach, letting the NMT model to parse a dependency tree and combine the parsing loss with the original loss.

Several works further incorporate the targetside syntactic structures explicitly.

BID12 interleaves CCG supertags with normal output words in the target side.

Instead of predicting words, Aharoni and Goldberg (2017) trains a NMT model to generate linearized constituent parse trees.

BID20 proposed a model to generate words and parse actions simultaneously.

The word prediction and action prediction are conditioned on each other.

However, none of the these methods plan the structure before translation.

Similar to our code learning approach, some works also learn the discrete codes for different purposes.

Shu and Nakayama (2018) compresses the word embeddings by learning the concept codes to represent each word.

BID7 breaks down the dependency among words with shorter code sequences.

The decoding can be faster by predicting the shorter artificial codes.

We evaluate our models on IWSLT 2014 Germanto-English task BID1 and ASPEC Japanese-to-English task BID13 , containing 178K and 3M bilingual pairs respectively.

We use Kytea BID15 to tokenize Japanese texts and moses toolkit BID8 for other languages.

Using bytepair encoding BID17

In the code learning model, all hidden layers have 256 hidden units.

The model is trained using Nesterov's accelerated gradient (NAG) BID14 for maximum 50 epochs with a learning rate of 0.25.

We test different settings of code length N and the number of code types K. The information capacity of the codes will be N log K bits.

In TAB1 , we evaluate the learned codes for different settings.

S y accuracy evaluates the accuracy of correctly reconstructing S y with the source sentence X and the code C y .

C y accuracy reflects the chance of guessing the correct code C y given X.We can see a clear trade-off between S Y accuracy and C Y accuracy.

When the code has more capacity, it can recover S Y more accurately, however, resulting in a lower probability for the NMT model to guess the correct code.

We found the setting of N = 2, K = 4 has a balanced trade-off.

To make a strong baseline, we use 2 layers of bidirectional LSTM encoders with 2 layers of LSTM decoders in the NMT model.

The hidden layers have 256 units for IWSLT De-En task and 1000 units for ASPEC Ja-En task.

We apply Key-Value Attention (Miller et al., 2016) in the first decoder layer.

Residual connection BID3 ) is used to combine the hidden states in two decoder layers.

Dropout is applied everywhere outside of the recurrent function with a drop rate of 0.2 .

To train the NMT models, we also use the NAG optimizer

Model BLEU(%) BS=1 BS=3 BS=5 with a learning rate of 0.25, which is annealed by a factor of 10 if no improvement of loss value is observed in 20K iterations.

Best parameters are chosen on a validation set.

As shown in TAB3 , by conditioning the word prediction on the generated planner codes, the translation performance is generally improved over a strong baseline.

The improvement may be the result of properly regulating the search space.

However, when we apply greedy search on JaEn dataset, the BLEU score is much lower compared to the baseline.

We also tried to beam search the planner codes then switch to greedy search, but the results are not significantly changed.

We hypothesize that it is important to simultaneously explore multiple candidates with drastically different structures on Ja-En task.

By planning ahead, more diverse candidates can be explored, which improves beam search but not greedy search.

If so, the results are in line with a recent study BID9 that shows the performance of beam search depends on the diversity of candidates.

Instead of letting the beam search to decide the planner codes, we can also choose the codes manually.

Table 3 gives an example of the candidate translations produced by the model when conditioning on different planner codes.input AP no katei ni tsuite nobeta. (Japanese) code 1 <c4> <c1> <eoc> the process of AP is described .code 2 <c1> <c1> <eoc> this paper describes the process of AP .code 3 <c3> <c1> <eoc> here was described on process of AP .code 4 <c2> <c1> <eoc> they described the process of AP .

Table 3 : Example of translation results conditioned on different planner codes in Ja-En task <c1> <c1> <c1> <c2> <c1> <c3> <c1> <c4> <c2> <c1> <c2> <c2> <c2> <c3> <c2> <c4> <c3> <c1> <c3> <c2> <c3> <c3> <c3> <c4> <c4> <c1> <c4> <c2> <c4> <c3> <c4> <c4> 4% 8% 12%

Figure 3: Distribution of assigned planner codes for English sentences in ASPEC Ja-En datasetAs shown in Table 3 , we can obtain translations with drastically different structures by manipulating the codes.

The results show that the proposed method can be useful for sampling paraphrased translations with high diversity.

The distribution of the codes learned for 3M English sentences in ASPEC Ja-En dataset is shown in Fig. 3 .

We found the code "<c1> <c1>" is assigned to 20% of the sentences, whereas "<c4> <c3>" is not assigned to any sentence.

The skewed distribution may indicate that the capacity of the codes is not fully exploited, and thus leaves room for further improvement.

Instead of learning discrete codes, we can also directly predict the structural annotations (e.g. POS tags), then translate based on the predicted structure.

However, as the simplified POS tags are also long sequences, the error of predicting the tags will be propagated to word generation.

In our experiments, doing so degrades the performance by around 8 BLEU points on IWSLT dataset.

In this paper, we add a planning phase in neural machine translation, which generates some planner codes to control the structure of the output sentence.

To learn the codes, we design an end-to-end neural network with a discretization bottleneck to predict the simplified POS tags of target sentences.

Experiments show that the proposed method generally improves the translation performance.

We also confirm the effect of the planner codes, by being able to sample translations with drastically different structures using different planner codes.

The planning phase helps the decoding algorithm by removing the uncertainty of the sentence structure.

The framework described in this paper can be extended to plan other latent factors, such as the sentiment or topic of the sentence.

@highlight

Plan the syntactic structural of translation using codes