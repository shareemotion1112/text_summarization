Like language, music can be represented as a sequence of discrete symbols that form a hierarchical syntax, with notes being roughly like characters and motifs of notes like words.

Unlike text however, music relies heavily on repetition on multiple timescales to build structure and meaning.

The Music Transformer has shown compelling results in generating music with structure (Huang et al., 2018).

In this paper, we introduce a tool for visualizing self-attention on polyphonic music with an interactive pianoroll.

We use music transformer as both a descriptive tool and a generative model.

For the former, we use it to analyze existing music to see if the resulting self-attention structure corroborates with the musical structure known from music theory.

For the latter, we inspect the model's self-attention during generation, in order to understand how past notes affect future ones.

We also compare and contrast the attention structure of regular attention to that of relative attention (Shaw et al., 2018, Huang et al., 2018), and examine its impact on the resulting generated music.

For example, for the JSB Chorales dataset, a model trained with relative attention is more consistent in attending to all the voices in the preceding timestep and the chords before, and at cadences to the beginning of a phrase, allowing it to create an arc.

We hope that our analyses will offer more evidence for relative self-attention as a powerful inductive bias for modeling music.

We invite the reader to explore our video animations of music attention and to interact with the visualizations at https://storage.googleapis.com/nips-workshop-visualization/index.html.

Attention is a cornerstone in neural network architectures.

It can be the primary mechanism for constructing a network, such as in the self-attention based Transformer, or serve as a secondary mechanism for connecting parts of a model that would otherwise be far apart or different modalities of varying dimensionalities.

Attention also offers us an avenue for visualizing the inner workings of a model, often to illustrate alignments BID3 .

For example in machine translation, the Transformer uses attention to build up both context and alignment while in the LSTM-based seq2seq models, attention eases the word alignment between source and target sentences.

For both types, attention gives points us to where a model is looking when translating BID6 BID0 .

For example in speech recognition, attention aligns different modalities from spectograms to phonemes BID1 .In contrast to the above domains, there is less "groundtruth" in what should be attended to in a creative domain such as music.

Moreover, in contrast to encoder-decoder models where attention serves as alignment, in language modeling self-attention serves to build context, to retrieve relevant information from the past to predict the future.

Music theory gives us some insight of the motivic, harmonic, temporal dependencies across a piece, and attention could be a lens in showing their relevance in a generative setting, i.e. does the model have to pay attention to this previous motif to generate the new note?

Music Transformer, based on self-attention BID6 , has been shown to be effective in modeling music, being able to generate sequences with repetition on multiple timescales (motifs and phrases) with long-term coherence BID2 .

In particular, the use of relative attention improved sample quality and allowed the model generalize beyond lengths observed during training time.

Why does relative attention help?

More generally, how does the attention structure look like on these models?In this paper, we introduce a tool for visualizing self-attention on music with an interactive pianoroll.

We use Music Transformer as both a descriptive tool and a generative model.

For the former, we use it to analyze existing music to see if the resulting self-attention structure corroborates with musical structure known from music theory.

For the latter, we inspect the model's self-attention during generation, in order to understand how past notes affect future ones.

We explore music attention on two music datasets, JSB Chorales and Piano-e-Competition.

The former are Chorale harmonizations, and we see attention keeping track of the harmonic progression and also voice-leading.

The latter are virtuosic classical piano music and attention looks back on previous motifs and gestures.

We show for JSB Chorales the heads in multihead-attention distribute and focus on different temporal regions.

Moreover, we compare and contrast the attention structure of regular attention to that of relative attention, and examine its impact on the resulting generated music.

For example, for the JSB Chorales dataset, a model trained with relative attention is more consistent in attending to all the voices in the preceding timestep and the many chords before, and at cadences to the beginning of a phrase, allowing it to create an arc.

In contrast, regular attention often becomes a "local" model only attending to the most recent history, resulting in certain voice repeating the same note for a long duration, perhaps due to overconfidence.

We take a language-modeling approach to training generative models for symbolic music.

Hence we represent music as a sequence of discrete tokens, with the vocabulary determined by the dataset.

The JSB Chorale dataset consists of four-part scored choral music, which can be represented as a pianoroll like representation with rows being pitch and columns being time discretized to sixteenth notes.

It is serialized in raster-scan fashion when consumed by a language model.

For the Piano-e-Competition dataset we use the performance encoding BID4 which consists of a vocabulary of 128 NOTE_ON events, 128 NOTE_OFFs, 100 TIME_SHIFTs allowing for expressive timing at 10ms and 32 VELOCITY bins for expressive dynamics.

The Transformer BID6 is a sequence model based primarily on self-attention.

Multiple heads are typically used to allow the model to focus on different parts of the history.

These are supported by first splitting the queries Q, keys K, and values V into h parts on the depth d dimension.

FIG0 shows the scaled dot-product attention for a single head.

Regular attention consists of only the Q h K h term, while relative attention adds S rel to modulate the attention logits based on pairwise distances between queries and keys.

DISPLAYFORM0 We adopt S rel = Skew(Q h E h ) as in BID2 , where E h are learned embeddings for every possible pairwise distance.

The attention outputs for each head are concatenated and linearly transformed to get Z, a L by D dimensional matrix, where L is the length of the input sequence.

FIG0 shows a full-view of our visualization tool for exploring self-attention 2 .

The arcs, inspired by BID7 , connect the current query (highlighted by the pink playhead) to earlier parts of the piece.

Each head bears a different color, and the thickness of the lines give the attention weights.

The user can choose to see a subset of the attention arcs either by specifying the top n number of arcs or by specifying a threshold at which attention weights lower then that would not be shown.

Our tool also supports animation, which allows us to inspect if a certain phenomena is consistent throughout a piece, and not just for certain timesteps.

FIG0 shows that some heads focus on the immediate past, some further back, nicely distributed in time.

This maybe due to relative attention explicitly modulating attention based on pairwise distance.

The left shows how on the bottom layer the attention is dense, while the right shows on the top layer each position is already a summary and hence the model only needs to attend to less positions.

When trained on JSB Chorales, regular attention failed to align the voices, causing one voice to repeat the same note (left on FIG2 ), while relative attention generated samples with musical phrasing.

To compare, we use relative attention (pink) and regular attention (green) to analyze the same JSB Chorale, by feeding the piece through the models and recording their attention weights.

FIG3 shows a drastic difference in how regular attention only focuses on the immediate past and the beginning of the piece, while relative attention attends to the entire passage.

FIG4 shows a sample generated by Transformer with relative attention trained on the Piano-eCompetition dataset.

The top shows a passage with right-hand "triangular" motifs and the model attends to the runs to learn the scale and also peaks to know when to change directions.

The bottom shows the same passage with the query being on the left-hand, and the attention focuses more on the left-hand chords and also the right-hand when they coincide with the left-hand.

On the left panel, the query is a left-hand note and attention is more focused on the bottom half compared to the right which is right-hand note, with more attention on the top half.

We presented a visualization tool for seeing and exploring music self-attention in context of music sequences.

We have shown some preliminary observations and we hope this it the beginning to furthering our understanding in how these models learn to generate music.

<|TLDR|>

@highlight

Visualizing the differences between regular and relative attention for Music Transformer.