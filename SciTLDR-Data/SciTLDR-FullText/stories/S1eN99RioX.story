We study the problem of cross-lingual voice conversion in non-parallel speech corpora and one-shot learning setting.

Most prior work require either parallel speech corpora or enough amount of training data from a target speaker.

However, we convert an arbitrary sentences of an arbitrary source speaker to target speaker's given only one target speaker training utterance.

To achieve this, we formulate the problem as learning disentangled speaker-specific and context-specific representations and follow the idea of [1] which uses Factorized Hierarchical Variational Autoencoder (FHVAE).

After training FHVAE on multi-speaker training data, given arbitrary source and target speakers' utterance, we estimate those latent representations and then reconstruct the desired utterance of converted voice to that of target speaker.

We use multi-language speech corpus to learn a universal model that works for all of the languages.

We investigate the use of a one-hot language embedding to condition the model on the language of the utterance being queried and show the effectiveness of the approach.

We conduct voice conversion experiments with varying size of training utterances and it was able to achieve reasonable performance with even just one training utterance.

We also investigate the effect of using or not using the language conditioning.

Furthermore, we visualize the embeddings of the different languages and sexes.

Finally, in the subjective tests, for one language and cross-lingual voice conversion, our approach achieved moderately better or comparable results compared to the baseline in speech quality and similarity.

Variational Autoencoder proposed by Hsu et al BID0 .

Let a dataset D consist of N seq i. DISPLAYFORM0 2 .

Thus, joint probability with a sequence X i is: DISPLAYFORM1 This is illustrated in FIG0 .

For inference, we use variational inference to approximate the true 93 posterior and have: DISPLAYFORM2 Since sequence variational lower bound can be decomposed to segment variational lower bound, we 95 can use batches of segment instead of sequence level to maximize: are shown.

In all subplots, the female and male embedding cluster locations are clearly separated.

DISPLAYFORM3

Furthermore, the plot shows that the speaker embeddings of unique speakers fall near the same 147 location.

One phenomenon that we notice is that the speaker embeddings for different languages and 148 gender fall to different locations for VAE-UNC, however, they fall closer to each other in VAE-CND.

This might be due to the conditioning on language improving the representation ability of the model.

Furthermore, we investigate the phonetic context embedding Z 1 for a sentence for four English test

To subjectively evaluate voice conversion performance, we performed two perceptual tests.

The

To evaluate the speaker similarity of the converted utterances, we conducted a same-different speaker 181 similarity test [32] .

In this test, listeners heard two stimuli A and B with different content, and 182 were then asked to indicate whether they thought that A and B were spoken by the same, or by two 183 different speakers, using a five-point scale comprised of +2 (definitely same), +1 (probably same), 0

@highlight

We use a Variational Autoencoder to separate style and content, and achieve voice conversion by modifying style embedding and decoding. We investigate using a multi-language speech corpus and investigate its effects.