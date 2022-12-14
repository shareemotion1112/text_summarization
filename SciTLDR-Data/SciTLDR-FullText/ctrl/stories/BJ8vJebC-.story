Character-based neural machine translation (NMT) models alleviate out-of-vocabulary issues, learn morphology, and move us closer to completely end-to-end translation systems.

Unfortunately, they are also very brittle and easily falter when presented with noisy data.

In this paper, we confront NMT models with synthetic and natural sources of noise.

We find that state-of-the-art models fail to translate even moderately noisy texts that humans have no trouble comprehending.

We explore two approaches to increase model robustness: structure-invariant word representations and robust training on noisy texts.

We find that a model based on a character convolutional neural network is able to simultaneously learn representations robust to multiple kinds of noise.

Humans have surprisingly robust language processing systems that can easily overcome typos, misspellings, and the complete omission of letters when reading BID34 .

A particularly extreme and comical exploitation of our robustness came years ago in the form of a popular meme: "Aoccdrnig to a rscheearch at Cmabrigde Uinervtisy, it deosn't mttaer in waht oredr the ltteers in a wrod are, the olny iprmoetnt tihng is taht the frist and lsat ltteer be at the rghit pclae." A person's ability to read this text comes as no surprise to the psychology literature.

BID38 found that this robustness extends to audio as well.

They experimented with playing parts of audio transcripts backwards and found that it did not affect comprehension.

BID35 found that in noisier settings reading comprehension only slowed by 11%.

BID27 found that the common case of swapping letters could often go unnoticed by the reader.

The exact mechanisms and limitations of our understanding system are unknown.

There is some evidence that we rely on word shape BID26 , that we can switch between whole word recognition and piecing together words from letters BID36 BID33 , and there appears to be no evidence that the first and last letter positions are required to stay constant for comprehension.

1 In stark contrast, neural machine translation (NMT) systems, despite their pervasive use, are immensely brittle.

For instance, Google Translate produces the following unintelligible translation for a German version of the above meme: 2 "After being stubbornly defiant, it is clear to kenie Rlloe in which Reiehnfogle is advancing the boulders in a Wrot that is integral to Sahce, as the utterance and the lukewarm boorstbaen stmimt."While typos and noise are not new to NLP, our systems are rarely trained to explicitly address them, as we instead hope that the relevant noise will occur in the training data.

Despite these weaknesses, the move to character-based NMT is important.

It helps us tackle the long tailed distribution of out-of-vocabulary words in natural language, as well as reduce computation load of dealing with large word embedding matrices.

NMT models based on characters and other sub-word units are able to extract stem and morphological information to generalize to unseen words and conjugations.

They perform very well in practice on a range of languages BID44 BID53 .

In many cases, these models actually discover an impressive amount of morphological information about a language BID2 .

Unfortunately, training (and testing) on clean data makes models brittle and, arguably, unfit for broad deployment.

Figure 1 shows how the performance of two state-of-the-art NMT systems degrades when translating German to English as a function of the percent of German words modified.

Here we show three types of noise: 1) Random permutation of the word, 2) Swapping a pair of adjacent letters, and 3) Natural human errors.

We discuss these types of noise and others in depth in section 4.2.

The important thing to note is that even small amounts of noise lead to substantial drops in performance.

Figure 1: Degradation of Nematus and char2char BID21 performance as noise increases.

To address these trends and investigate the effects of noise on NMT, we explore two simple strategies for increasing model robustness: using structure-invariant representations and robust training on noisy data, a form of adversarial training BID49 BID14 .

We find that a character CNN representation trained on an ensemble of noise types is robust to all kinds of noise.

We shed some light on the model ability to learn robust representations to multiple types of noise, and point to remaining difficulties in handling natural noise.

Our goal is two fold: 1) initiate a conversation on robust training and modeling techniques in NMT, and 2) promote the creation of better and more linguistically accurate artificial noise to be applied to new languages and tasks.

The growing literature on adversarial examples has demonstrated how dangerous it can be to use brittle machine learning systems so pervasively in the real world BID4 BID49 BID14 BID28 .

Small changes to the input can lead to dramatic failures of deep learning models BID49 BID14 .

In the machine vision field, changes to the input image that are indistinguishable by humans can lead to misclassification.

This leads to potential for malicious attacks using adversarial examples.

An important distinction is often drawn between white-box attacks, where adversarial examples are generated with access to the model parameters, and black-box attacks, where examples are generated without such access BID30 BID43 BID29 BID23 .While more common in the vision domain, recent work has started exploring adversarial examples for NLP.

A few white-box attacks have employed the fast gradient sign method BID14 or other techniques to find important text edit operations BID31 BID41 BID11 .

Others have considered black-box adversarial examples for text classification BID12 or NLP evaluation BID17 .

BID15 evaluated character-based models on several types of noise in morphological tagging and MT, and observed similar trends to our findings.

Finally, BID40 designed a character-level recurrent neural network that can better handle the particular kind of noise present in the meme mentioned above by modeling spelling correction.

Here we devise simple methods for generating adversarial examples for NMT.

We do not assume any access to the NMT models' gradients, instead relying on synthetic and naturally occurring language errors to generate noise.

The other side of the coin is to improve models' robustness to adversarial examples BID13 BID9 BID37 BID7 .

Adversarial trainingincluding adversarial examples in the training data -can improve a model's ability to cope with such examples at test time BID49 BID14 .

This kind of defense is sensitive to the type of adversarial examples seen in training, but can be made more robust by ensemble adversarial training -training on examples transfered from multiple pre-trained models BID50 .

We explore ensemble training by combining multiple types of noise at training time, and observe similar increased robustness in the machine translation scenario.

Training on and for adversarial noise is an important extension of earlier work on creating robustness in neural networks by incorporating noise to a network's representations, data, or gradients.

Training with noise can provide a form of regularization BID5 and ensure the model is exposed to samples outside the training distribution BID24 .

The rise of end-to-end models in neural machine translation has led to recent interest in understanding how these models operate.

Several studies investigated the ability of such models to learn linguistic properties at morphological BID51 BID2 , syntactic BID47 BID43 , and semantic levels BID3 .

The use of characters or other sub-word units emerges as an important component in these models.

Our work complements previous studies by presenting such NMT systems with noisy examples and exploring methods for increasing their robustness.

We experiment with three different NMT systems with access to character information at different levels.

First, we use the fully character-level model of BID21 .

This is a sequence-tosequence model with attention BID0 that is trained on characters to characters (char2char).

It has a complex encoder with convolutional, highway, and recurrent layers, and a standard recurrent decoder.

See BID21 for architecture details.

This model was shown to have excellent performance on the German???English and Czech???English language pairs.

We use the pre-trained German/Czech???English models.

Second, we use Nematus , a popular NMT toolkit that was used in topperforming contributions in shared MT tasks in WMT BID45 and IWSLT (JunczysDowmunt & Birch, 2016) .

It is another sequence-to-sequence model with several architecture modifications, especially operating on sub-word units using byte-pair encoding (BPE) BID44 .

We experimented with both their single best and ensemble BPE models, but saw no significant difference in their performance under noise, so we report results with their single best WMT models for German/Czech???English.

Finally, we train an attentional sequence-to-sequence model with a word representation based on a character convolutional neural network (charCNN).

This model retains the notion of a word but learns a character-dependent representation of words.

It was shown to perform well on morphologically-rich languages BID20 BID1 BID8 , thanks to its ability to learn morphologically-informative representations BID2 ).

The charCNN model has two long short-term memory BID16 layers in the encoder and decoder.

A CNN over characters in each word replaces the word embeddings on the encoder side (for simplicity, the decoder is word-based).

We use 1000 filters with a width of 6 characters.

The character embedding size is set to 25.

The convolutions are followed by Tanh and max-pooling over the length of the word BID20 .

We train charCNN with the implementation in BID19 ; all other settings are kept to default values.

We use the TED talks parallel corpus prepared for IWSLT 2016 BID6 for testing all of the NMT systems, as well as for training the charCNN models.

We follow the official training/development/test splits.

All texts are tokenized with the Moses tokenizer.

TAB1 summarizes statistics on the TED talks corpus.

Since we do not have access to a parallel corpus with natural noise, we instead harvest naturally occurring errors (typos, misspellings, etc.) from available corpora of edits to build a look-up table of possible lexical replacements.

In this work, we restrict ourselves to single word replacements, but several of the corpora below also provide access to phrase replacements.

French BID25 collected Wikipedia edit histories to form the Wikipedia Correction and Paraphrase Corpus (WiCoPaCo).

They found the bulk of edits were due to incorrect diacritics, choosing the wrong homophone, and incorrect grammatical conjugation.

German Our German data combines two projects: RWSE Wikipedia Revision Dataset BID54 and The MERLIN corpus of language learners BID52 .

These corpora were created to measure spelling difficulty and test models of contextual fitness.

Unfortunately, the datasets are quite small so we have combined them here.

Czech Our Czech errors come from manually annotated essays written by non-native speakers (??ebesta et al., 2017) .

Here, the authors found an incredibly diverse set of errors, and therefore phenomena of interest: capitalization, incorrectly replacing voiced and voiceless consonants (e.g. z/s, g/k), missing palatalization (mat??e/matce), error in valence, pronominal reference, inflection, colloquial forms, and so forth.

Their analysis gives us the best insight into how difficult it would be to synthetically generate truly natural errors.

We found similarly rich errors in German (Section 7.2).We insert these errors into the source-side of the parallel data by replacing every word in the corpus with an error if one exists in our dataset.

When there is more than one possible replacement to choose we sample uniformly.

Words for which there is no error are kept as is.

TAB2 shows the number of words for which we were able to collect errors in each language, and the average number of errors per word.

Despite the small size of the German and Czech datasets, we are able to replace up to half of the words in the corpus with errors.

Due to the small size of the German and Czech datasets these percentages decrease for longer words (> 4 characters) to 25% and 32%, respectively.

In addition to naturally collected sources of error, we also experiment with four types of synthetic noise: Swap, Middle Random, Fully Random, and Keyboard Typo.

According to a study from Cambridge university, it doesn't matter which order letters in a word are, the only important thing is that the first and the last letter appear in their correct place.char2char Cambridge Universttte is one of the most important features of the Cambridge Universttten , which is one of the most important features of the Cambridge Universttten .

Luat eienr Stduie der Cambrant Unievrstilt splashed it kenie Rlloe in welcehr Reiehnfogle the Buhcstbaen in eniem Wred vorkmomen, die eingzie whcene Sahce ist, DSAs der ertse und der lettze Buhcstbaen stmimt .

charCNN According to the <unk> of the Cambridge University , it 's a little bit of crude oil in a little bit of recycling , which is a little bit of a cool cap , which is a little bit of a strong cap , that the fat and the <unk> bites is consistent .Swap :

Swap The simplest source of noise is swapping two letters (e.g. noise???nosie).

This is common when typing quickly and is easily implemented.

We perform one swap per word, but do not alter the first or last letters.

For this reason, this noise is only applied to words of length ??? 4.Middle Random : Mid Following the claims of the previously discussed meme, we randomize the order of all the letters in a word except for the first and last (noise???nisoe).

Again, by necessity, this means we do not alter words shorter than four characters.

Fully Random :

Rand As we are unaware of any strong results on the importance of the first and last letters we also include completely randomized words (noise???iones).

This is a particularly extreme case, but we include it for completeness.

This type of noise is applied to all words.

Keyboard Typo : Key Finally, using the traditional keyboards for our languages, we randomly replace one letter in each word with an adjacent key (noise???noide).

This type of error should be much easier than the random settings as most of the word is left intact, but does introduce a completely new character which will often break the templates a system has learned to rely on.

TAB3 shows BLEU scores of models trained on clean (Vanilla) texts and tested on clean and noisy texts.

All models suffer a significant drop in BLEU when evaluated on noisy texts.

This is true for both natural noise and all kinds of synthetic noise.

The more noise in the text, the worse the translation quality, with random scrambling producing the lowest BLEU scores.

The degradation in translation quality is especially severe in light of humans' ability to understand noisy texts.

To illustrate this, consider the noisy text in TAB4 .

Humans are quite good at understanding such scrambled texts in a variety of languages.

4 We also verified this by obtaining a translation from a German native-speaker, unfamiliar with the meme.

As shown in the table, the speaker had no trouble understanding and translating the sentence properly.

In contrast, the state-ofthe-art systems (char2char and Nematus) fail on this text.

One natural question is if robust spell checkers trained on human errors are sufficient to address this performance gap.

To test this, we ran texts with and without natural errors through Google Translate.

We then used Google's spell-checkers to correct the documents.

We simply accepted the first suggestion for every detected mistake detected, and report results in TAB5 .We found that in French and German, there was often only a single predicted correction and this corresponds to roughly +5 or more in BLEU.

In Czech, however, there was often a large list of possible conjugations and changes, likely indicating that a rich grammatical model would be necessary to predict the correction.

It is also important to note the substantial drops from vanilla text even with spell check.

This suggests that natural noise cannot be easily addressed by existing tools.6 DEALING WITH NOISE

The three NMT models are all sensitive to word structure.

The char2char and charCNN models both have convolutional layers on character sequences, designed to capture character n-grams.

The model in Nematus is based on sub-word units obtained with BPE.

It thus relies on character order within and across sub-word units.

All these models are therefore sensitive to types of noise generated by character scrambling (Swap, Mid, and Rand).

Can we improve model robustness by adding invariance to these kinds of noise?

Perhaps the simplest such model is to take the average character embedding as a word representation.

This model, referred to as meanChar, first generates a word representation by averaging character embeddings, and then proceeds with a word-level encoder similar to the charCNN model.

The meanChar model is by definition insensitive to scrambling, although it is still sensitive to other kinds of noise (Key and Nat).

Table 6 (first row) shows the results of meanChar models trained on vanilla texts and tested on noisy texts (the results on vanilla texts are by definition equal to those on scrambled texts).

Overall, the average character embedding proves to be a pretty good representation for translating scrambled texts: while performance drops by about 7 BLEU points below charCNN on vanilla French and German, it is much better than charCNN's performance on scrambled texts (compare to TAB3 ).

The results of meanChar on Czech are much worse, possibly due to its more complex morphology.

However, the meanChar model performance degrades quickly on other kinds of noise as the model trained on vanilla texts was not designed to handle Nat and Key types of noise.

To increase model robustness we follow a black-box adversarial training scenario, where the model is presented with adversarial examples that are generated without direct access to the model BID30 BID43 BID23 BID29 BID17 .

We replace the original training set with a noisy training set, where noise is introduced according to the description in Section 4.2.

The noisy training set has exactly the same number of sentences and words as the training set.

We have one fixed noisy training set per each noise type.

As shown in Table 6 (second block), training on noisy text can lead to improved performance.

The meanChar models trained on Key perform well on Key in French, but not in the other languages.

The models trained on Nat perform well in French and German, but not in Czech.

Overall, training the meanChar model on noisy text does not appear to consistently increase its robustness to different kinds of noise.

The meanChar model however was not expected to perform well on nonscrambling types of noise.

Next we test whether the more complicated charCNN model is more robust to different kinds of noise, by training on noisy texts.

The results are shown in TAB6 .In general, charCNN models that are trained on a specific kind of noise perform well on the same kind of noise at test time (results in bold).

All models also maintain a fairly good quality on vanilla texts.

The robust training is sensitive to the kind of noise.

Among the scrambling methods (Swap/Mid/Rand), more noise helps in training: models trained on random noise can still translate Swap/Mid noise, but not vice versa.

The three broad classes of noise (scrambling, Key, Nat) are not mutually-beneficial.

Models trained on one do not perform well on the others.

In particular, only models trained on natural noise can reasonably translate natural noise at test time.

We find this result indicates an important difference between computational models and human performance, since humans can decipher random letter orderings without explicit training of this form.

Next, we test whether we can increase training robustness by exposing the model to multiple types of noise during training.

Our motivation is to see if models can perform well on more than one kind of noise.

We therefore mix up to three kinds of noise by sampling a noise method uniformly at random for each sentence.

We then train a model on the mixed noisy training set and test it on both vanilla and (unmixed) noisy versions of the test set.

We find that models trained on mixed noise are slightly worse than models trained on unmixed noise.

However, the models trained on mixed noise are robust to the specific types of noise they were trained on.

In particular, the model trained on a mix of Rand, Key, and Nat noise is robust to all noise kinds.

Even though it is not the best on any one kind of noise, it achieves the best result on average.

This model is also able to translate the scrambled meme reasonably well:"According to a study of Cambridge University, it doesn't matter which technology in a word is going to get the letters in a word that is the only important thing for the first and last letter.

"

The charCNN model was able to perform well on all kinds of noise by training on a mix of noise types.

In particular, it performed well on scrambled characters even though its convolutions should be sensitive to the character order, as opposed to meanChar which is by definition invariant to character order.

How then can charCNN learn to be robust to multiple kinds of noise at the same time?

We speculate that different convolutional filters learn to be robust to different kinds of noise.

A convolutional filter can in principle capture a mean (or sum) operation by employing equal or close to equal weights.

To test this, we analyze the weights learned by charCNN models trained under four conditions: three models trained each on completely scrambled words (Rand), keyboard typos (Key), and natural human errors (Nat), as well as an ensemble model trained on a mix of Rand+Key+Nat kinds of noise.

For each model, we compute the variance across the filter width (6 characters) for each one of the 1000 filters and for each one out of 25 character embedding dimensions.

Intuitively, this variance captures how much a particular filter learns a uniform vs. non-uniform combination of characters.

Then we average the variances across the 1000 filters.

This yields 25 averaged variances, one for each character embedding dimension.

Low average variance means that different filters tend to learn similar behaviors, while high average variance means that they learn different patterns.

FIG0 shows a box plot of these averages for our three languages and four training conditions.

Clearly, the variances of the weights learned by the Rand model are much smaller than those of the weights learned by any other setting.

This makes sense as with random scrambling there are no patterns to detect in the data, so filters resort to close to uniform weights.

In contrast, the Key and Nat settings introduce a large set of new patterns for the CNNs to try and learn, leading to high variances.

Finally, the ensemble model trained on mixed noise appears to be in the middle as it tries to capture both the uniform relationships of Rand and the more diverse patterns of Nat + Key.

Moreover, the variance of variances (size of the box) is smallest in the Rand setting, larger in the mixed noise model, and largest in Key and Nat.

This indicates that filters for different character embedding dimensions are more different from one another in Key and Nat models.

In contrast, in the Rand model, the variance of variances is close to zero, indicating that in all character embedding dimensions the learned weights are of small variance; they do similar things, that is, the model learned to reproduce a representation similar to the meanChar model.

The ensemble model again seems to find a balance between Rand and Key/Nat.

Natural noise appears to be very different from synthetic noise.

None of the models that were trained only on synthetic noise were able to perform well on natural noise.

We manually analyzed a small sample (~40 examples) of natural noise from the German dataset.

We found that the most common sources of noise are phonetic or phonological phenomena in the language (34%) and character omissions (32%).

The rest are incorrect morphological conjugations of verbs, key swaps, character insertions, orthographic variants, and other errors.

TAB7 shows examples of these kinds of noise.

The most common types of natural noise -phonological and omissions -are not directly captured by our synthetic noise generation, and demonstrate that good synthetic errors will likely require more explicit phonemic and linguistic knowledge.

This discrepancy helps explain why the models trained on synthetic noise were not particularly successful in translating natural noise.

In this work, we have shown that character-based NMT models are extremely brittle and tend to break when presented with both natural and synthetic kinds of noise.

We investigated methods for increasing their robustness by using a structure-invariant word representation and by ensemble training on adversarial examples of different kinds.

We found that a character-based CNN can learn to address multiple types of errors that are seen in training.

However, we observed rich characteristics of natural human errors that cannot be easily captured by existing models.

Future work might investigate using phonetic and syntactic structure to generate more realistic synthetic noise.

We believe that more work is necessary in order to immune NMT models against natural noise.

As corpora with natural noise are limited, another approach to future work is to design better NMT architectures that would be robust to noise without seeing it in the training data.

New psychology results on how humans cope with natural noise might point to possible solutions to this problem.

<|TLDR|>

@highlight

CharNMT is brittle

@highlight

This paper investigates the impact of character-level noise on 4 different neural machine translation systems

@highlight

This paper empirically investigates the performance of character-level NMT systems in the face of character-level noise, both synthesized and natural.

@highlight

This paper investigates the impact of noisy input on Machine Translation and tests ways to make NMT models more robust