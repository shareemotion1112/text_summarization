Recent work has shown that contextualized word representations derived from neural machine translation (NMT) are a viable alternative to such from simple word predictions tasks.

This is because the internal understanding that needs to be built in order to be able to translate from one language to another is much more comprehensive.

Unfortunately, computational and memory limitations as of present prevent NMT models from using large word vocabularies, and thus alternatives such as subword units (BPE and morphological segmentations) and characters have been used.

Here we study the impact of using different kinds of units on the quality of the resulting representations when used to model syntax, semantics, and morphology.

We found that while representations derived from subwords are slightly better for modeling syntax, character-based representations are superior for modeling morphology and are also more robust to noisy input.

Recent years have seen the rise of deep neural networks and the subsequent rise of representation learning based on network-internal activations.

Such representations have been shown useful when addressing various problems from fields such as image recognition , speech recognition BID2 , and natural language processing (NLP) BID30 .

The central idea is that the internal representations trained to solve an NLP task could be useful for other tasks as well.

For example, word embeddings learned for a simple word prediction task in context, word2vec-style BID31 , have now become almost obligatory in state-ofthe-art NLP models.

One issue with such word embeddings is that the resulting representation is context-independent.

Recently, it has been shown that huge performance gains can be achieved by contextualizing the representations, so that the same word could have a different embedding in different contexts.

This is best achieved by changing the auxiliary task, e.g., the ElMo model learns contextualized word embeddings from a language modeling task, using LSTMs BID37 .More recently, it has been shown that complex tasks such as neural machine translation can yield superior representations BID29 .

This is because the internal understanding of the input language that needs to be built by the network in order to be able to translate from one language to another needs to be much more comprehensive compared to what would be needed for a simple word prediction task.

Such representations have yielded state-of-the-art results for tasks such as sentiment analysis, textual entailment, and question answering.

Unfortunately, computational and memory limitations as of present prevent neural machine translation (NMT) models from using large-scale vocabularies, typically limiting them to 30-50k words .

This is a severe limitation, as most NLP applications need to handle vocabularies of millions of words, e.g., word2vec BID31 , GloVe BID36 and FastText BID32 offer pre-trained embeddings for 3M, 2M, and 2.5M words/phrases, respectively.

The problem is typically addressed using byte-pair encoding (BPE), where words are segmented into pseudo-word character sequences based on frequency BID43 .

A somewhat less popular solution is to use characters as the basic unit of representation BID8 .

In the case of morphologically complex languages, another alternative is to reduce the vocabulary by using unsupervised morpheme segmentation BID6 ).The impact of using different units of representation in NMT models has been studied in previous work BID27 BID10 BID8 Lee et al., 2017, among others) , but the focus has been exclusively on the quality of the resulting translation output.

However, it remains unclear what input and output units should be chosen if we are primarily interested in representation learning.

Here, we aim at bridging this gap by evaluating the quality of NMT-derived embeddings originating from units of different granularity when used for modeling morphology, syntax, and semantics (as opposed to end tasks such as sentiment analysis and question answering).

Our contributions can be summarized as follows:• We study the impact of using words vs. characters vs. BPE units vs. morphological segments on the quality of representations learned by NMT models when used to model morphology, syntax, and semantics.• We further study the robustness of these representations with respect to noise.• We make practical recommendations based on our results.

We found that while representations derived from morphological segments are better for modeling syntax, character-based ones are superior for morphology and are also more robust to noise.

Representation analysis aims at demystifying what is learned inside the neural network black-box.

This includes analyzing word and sentence embeddings BID1 BID39 BID12 Conneau et al., 2018, among others) , RNN states (Qian et al., 2016a; BID44 BID52 BID50 , and NMT representations BID44 BID4 , as applied to morphological BID39 BID49 , semantic BID39 and syntactic BID28 BID47 BID9 tasks.

While previous work focused on words, here we compare units of different granularities.

Subword translation units aim at reducing vocabulary size and OOV rate.

NMT researchers have used BPE units BID43 , morphological segmentation BID6 , characters , and hybrid units BID27 BID10 .

There have also been comparisons between subword units in the context of NMT BID42 .

Unlike this work, here we focus on representation learning rather than on translation quality.

Robustness to noise is an important aspect in machine learning.

It has been studied for various machine learning models BID46 BID15 , including NLP models BID34 BID41 BID11 BID13 BID21 , and character-based NMT models BID17 BID3 .

Unlike the above work, we compare robustness to noise for units of different granularity.

Moreover, we focus on representation learning rather than translation.

Our methodology is inspired by research on interpreting neural network (NN) models.

A typical framework involves extracting feature representations from different components (e.g., encoder/decoder) of a trained model and then training a classifier to make predictions for an auxiliary task.

The performance of the trained classifier is considered to be a proxy for judging the quality of the extracted representations with respect to the particular auxiliary task.

Formally, for each input word x i we extract the LSTM hidden states from each layer of the encoder/decoder.

We concatenate the representations of layers and we use them as feature vector z i for the auxiliary task.

We train a logistic regression classifier by minimizing the cross-entropy loss: DISPLAYFORM0 is the probability that word x i is assigned label l.

The weights θ ∈ R D×L are learned with gradient descent.

Here D is the dimensionality of the latent representations z i and L is the size of the label set for P.

We consider four representation units: words, byte-pair encoding (BPE) units, morphological units, and characters.

TAB0 shows an example of each representation unit.

BPE splits words into symbols (a symbol is a sequence of characters) and then iteratively replaces the most frequent sequences of symbols with a new merged symbol.

In essence, frequent character n-gram sequences merge to form one symbol.

The number of merge operations is controlled by a hyper-parameter OP, which directly affects the granularity of segmentation: a high value of OP means coarse segmentation and a low value means fine-grained segmentation.

For morphologically segmented units, we use an unsupervised morphological segmenter, Morfessor BID45 .

Note that although BPE and Morfessor segment words at a similar level of granularity, the segmentation generated by Morfessor is linguistically motivated.

For example, it splits the gerund shooting into base verb shoot and the suffix ing.

Compare this to the BPE segmentation sho + oting, which has no linguistic justification.

On the extreme, the fully character-level units treat each word as a sequence of characters.

Extracting Activations for Subword and Character Units Previous work on analyzing NMT representations has been limited to the analysis of word representations only, 1 where there is a oneto-one mapping from input units (words) and their NMT representations (hidden states) to their linguistic annotations (e.g., morphological tags).

In the case of subword-based systems, each word may be split into multiple subword units, and each unit has its own representation.

It is less trivial to define which representations should be evaluated when predicting a word-level property.

We consider two simple approximations to estimate a word representation from subword representations:

(i) Average: for each word, average the activation values of all the subwords (or characters) comprising it.

In the case of a bi-directional encoder, we concatenate the averages from the forward and the backward activations of the encoder on the subwords (or characters) that represent the current word. (ii) Last: consider the activation of the last subword (or character) as the representation of the word.

For the bi-directional encoder, we concatenate the forward encoder's activation on the last subword unit with the backward encoder's activation on the first subword unit.

This formalization allows us to analyze character-and subword-based representations at the word level via prediction tasks.

Such kind of analysis has not been performed before.

We choose three fundamental NLP tasks that serve as a good representative of various properties inherent in a language, ranging from morphology (word structure), syntax (grammar) and semantics (meaning).

In particular, we experiment with morphological tagging for German, Czech, Russian and English 2 languages, lexical semantics tagging for English and German languages, and syntactic tagging via CCG supertagging for English language.

TAB1 shows an example sentence with annotations of each task.

The morphological tags capture word structure, semantic tags show semantic property, and syntax tags (CCG super tags) captures global syntactic information locally at the lexical level.

For example in TAB1 , -the morphological tag VBZ for the word "receives", marks that it is a verb with non-third person singular present property, the semantic tag ENS describes a present simple event category, and the syntactic tag S[dcl]\NP)/NP indicates that the preposition "in" attaches to the verb.

Recent studies have shown that small perturbations in the input can cause significant deterioration in the performance of the deep neural networks.

Here, we evaluate the robustness of various representations under noisy input conditions.

We use corpora of real errors harvested by BID3 .

The errors contain a good mix of typos, misspellings, and other kinds of errors.

In addition, we create data with synthetic noise.

We induced two kinds of errors: i) Swap and Middle.

Swap is a common error which occurs when neighboring characters are mistakenly swapped (e.g., word → wodr).

In Middle errors, the order of the first and the last characters of a word are preserved while the middle characters are randomly shuffled BID40 ) (e.g., example→ eaxmlpe).

We corrupt (using swap or middle) or replace (using real errors corpora) n% words randomly in each test sentence.

We then re-extract feature vectors for the erroneous words in a sentence and re-evaluate the prediction capability of these embeddings on the linguistic tasks.

Data and Languages: We trained NMT systems for 4 language pairs: German-English, CzechEnglish, Russian-English and English-German, using data made available through the two popular machine translation campaigns, namely, WMT BID5 and IWSLT BID7 .

The MT models were trained using a concatenation of NEWS and TED training data.

We used official TED testsets (testsets-11-13) to report translation quality BID35 .

The morphological classifiers were trained and tested on a concatenation of NEWS and TED testsets, which were automatically tagged as described in the next section.

Semantic and syntactic classifiers were trained and tested on existing annotated corpora.

Statistics are shown in TAB2 .Taggers: We used RDRPOST (Nguyen et al., 2014) to annotate data for the classifier.

For semantic tagging, we used the the Groningen Parallel Meaning Bank BID0 .

The tags TAB2 for statistics.

We used seq2seq-attn BID23 to train 2-layered attentional long short-term memory (LSTM) BID18 encoder-decoder systems with bidirectional encoder.

We used 500 dimensions for both word embeddings and LSTM states.

We trained systems with SGD for 20 epochs and used the final model for generating features for the classifier.

We trained the systems in both *-to-English and English-to-* directions and analyze the representations from both encoder and decoder.

To analyze the encoder-side, we fix the decoder-side with BPE-based embeddings and train the source-side with word/BPE/Morfessor/char units.

Similarly, to analyze the decoder-side, we train the encoder representation with BPE units and vary the decoder side with different input units.

Our motivation for this setup is to analyze representations in isolation keeping the other half of the network static across different settings.

We use 50k BPE operations and limit the vocabulary of all systems to 50k.

The word/BPE/Morfessor/characterbased systems were trained with sentence lengths of 80/100/100/400, respectively.

The classifier is a logistic regression whose input is either hidden states in word-based models, or Last or Average representations in character-and subword-based models.

Since we concatenate forward and backward states from all layers, this ends up being 2000/1000 dimensions when classifying the encoder/decoder: 500 dimensions×2 layers×2 directions (1 for decoder).

The classifiers are trained for 10 epochs.

The encoder models are trained with BPE as target and the decoder models with BPE as a source.

We now present the results of using representations learned from different input units on the task of predicting morphology, semantics and syntax.

For subword and character units, we found that the activation of the last subword/character unit of a word consistently better than using the average of all activations.

So we present the results using Last method only and discuss this more later.

FIG0 summarizes the results for predicting morphological tags with representations learned using different units.

The character-based representations consistently outperformed other representations on all language pairs while the word-based representations achieved the lowest accuracy.

The differences are more significant in the case of languages with relatively complex morphology, Czech and Russian.

We see a difference of up to 14% in favor of using character-based representations when compared with the word-based representations.

The improvement is minimal in the case of English (1.2%), which is a morphologically simpler language.

Comparing subword units as obtained using Morfessor and BPE, we found Morfessor to give much better morphological tagging performance especially in the case of morphologically rich languages, Czech and Russian.

This is due to the linguistically motivated segmentations which are helpful in for learning language morphology.

We further investigated whether the performance difference between various representation is due to the difference in modeling infrequent and out-of-vocabulary words.

TAB4 shows the OOVs rate of each language which is higher for morphologically rich languages.

Figure 2 shows that the gap between different representations is inversely related to the frequency of the word in the training data: character-based models perform much better than others on less frequent and OOV words.

Decoder Representations: Next, we used the decoder representations from the English-to-* models.

We saw a similar performance trend as in the case of encoder-side representations, character units performed the best while word units performed the worst.

Also morphological units performed better than the BPE-based units.

Comparing encoder representation with decoder representation, it is interesting to see that in several cases the decoder-side representations performed better than the encoder-side representations, even though they are trained using a uni-directional LSTM only.

Since we did not see any difference in trend between encoder and decoder side representations, we only present the encoder side results in the later part of the paper.

Figure 3 summarizes the results on the semantic tagging task.

On English, subword-based (BPE and Morfessor) representations and character-based representation achieve comparable results.

However, for German, BPE-based representations performed better than the other representations.

These results contrast to morphology prediction results, where character-based representations were consistently better compared to their subword-based counterparts.

The final property we evaluate is CCG super-tagging, reflecting syntactic knowledge.

Here we only have English tags, so we evaluate encoder representations from English→German models, trained with words, characters, and subwords.

We found that morphologically segmented representation units perform the best while words and BPE-based representations perform comparable.

The characters-based representations lag behind, though the difference between accuracy is small compared to the morphological tagging results.

4 It is noteworthy that characters perform below both words and subwords here, contrary to their superior performance on the task of morphology.

We will return to this point in the discussion in Section 6.

We now evaluate the robustness of the representations towards noise.

We induce errors in the testsets by corrupting 25% of the words in each sentence using different error types (synthetic or real noise), as described in Section 3.3.

We extract the representations of the noisy testsets and re-evaluate the classifiers.

FIG2 shows the performance on each task.

Evidently, characters yield much better performance on all tasks and for all languages, showing minimal drop in the accuracy, in contrast to earlier results where they did not outperform subword units 5 on the task of syntactic tagging.

This result shows character-based representations are more robust towards noise compared to others.

Surprisingly in a few cases, BPE-based representations performed even worst than word-based representations, e.g. in the case of Syntactic tagging (80.3 vs. 81.1).

We hypothesize that BPE segments a noisy word into two known subword units that may have no close relationship with the actual word.

Using representations of wrong subword units resulted in a significant drop in performance.

We further investigated the robustness of each classifier by increasing the percentage of noise in the test data and found that the difference in representation quality stays constant across BPE and

Comparing Performance Across Tasks Character-based representations outperformed in the case of morphological tagging; BPE-based representations performed better than others in the semantic tagging task for German (and about the same in English); and Morfessor performed slightly better than others for syntax.

Syntactic tagging requires knowledge of the complete sentence.

Splitting a sentence into characters substantially increases the length (from 50 words in a sentence to 250 characters on average) of the sentence.

The character-based models lack in capturing long distance dependencies, which could be a reason for their low performance in this task.

Similarly, in case of morphological tagging, the information about the morphology of a word is dependent on the surrounding words plus internal information (root, morphemes etc.) presents in the word.

The character-based system has access to all of this information which results in high tagging performance.

Morfessor performed better than BPE in the morphological tagging task because its segments are linguistically motivated units (segmented into root + morphemes), making the information about the word morphology explicit in the representation.

In comparison, BPE solely focuses on the frequency of characters occurring together in the corpus and can yield linguistically incorrect units.

TAB3 summarizes the translation performance of each system.

In most of the cases, the subword-based systems perform better than the word-based and character-based systems.

However, this is not true in the case of using their representations as feature in the core NLP tasks.

For example, we found that character-based representations perform better than others in the morphological tagging task.

On an additional note, BPE-based representations although perform better for some tasks, are sensitive to noise.

Their ability to segment any unknown words into two known subwords result in less reliable systems.

Notably, the translation performance of the BPE-based system falls below the character-based system even with 10% noise only.

The variation in the performance of the representations reflect that they may be learning different aspects of the language.

To investigate whether representations are complementary to each other, we train the classifier on their concatenation.

TAB5 summarizes the results on the morphological tagging task.

The performance of the classifier improved in all combinations of representations while the best results are achieved using all three units together.

We studied the impact of using different representation units -words, characters, BPE units, and morphological segments on the representations learned by NMT.

Unlike previous work, which targeted end tasks such as sentiment analysis and question answering, here we focused on modeling morphology, syntax and semantics.

We found that (i) while representations derived from subwords units are slightly better for modeling syntax, (ii) character representations are distinctly better for modeling morphology, and (iii) are also more robust to noise in contrast to subword representations, (iv) and that using all representations together works best.

Based on our findings, we conjecture that although BPE segmentation is a de-facto standard in building state-of-the-art NMT systems, the underlying representations it yields are suboptimal for external tasks.

Character-based representations provide a more viable and robust alternative in this regard, followed by morphological segmentation.

In future work, we plan to explore specialized character-based architectures for NMT.

We further want to study how different units affect representation quality in non-recurrent models such as the Transformer BID48 and in convolutional architectures BID14 .A SUPPLEMENTARY MATERIAL

@highlight

We study the impact of using different kinds of subword units on the quality of the resulting representations when used to model syntax, semantics, and morphology.