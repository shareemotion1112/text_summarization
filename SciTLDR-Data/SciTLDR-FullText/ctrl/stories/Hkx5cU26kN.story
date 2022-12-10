Recent pretrained sentence encoders achieve state of the art results on language understanding tasks, but does this mean they have implicit knowledge of syntactic structures?

We introduce a grammatically annotated development set for the Corpus of Linguistic Acceptability (CoLA; Warstadt et al., 2018), which we use to investigate the grammatical knowledge of three pretrained encoders, including the popular OpenAI Transformer (Radford et al., 2018) and BERT (Devlin et al., 2018).

We fine-tune these encoders to do acceptability classification over CoLA and compare the models’ performance on the annotated analysis set.

Some phenomena, e.g. modification by adjuncts, are easy to learn for all models, while others, e.g. long-distance movement, are learned effectively only by models with strong overall performance, and others still, e.g. morphological agreement, are hardly learned by any model.

The effectiveness and ubiquity of pretrained sentence embeddings for natural language understanding has grown dramatically in recent years.

Recent sentence encoders like OpenAI's Generative Pretrained Transformer (GPT; Radford et al., 2018) and BERT (Devlin et al., 2018) achieve the state of the art on the GLUE benchmark (Wang et al., 2018) .

Among the GLUE tasks, these stateof-the-art systems make their greatest gains on the acceptability task with the Corpus of Linguistic Acceptability (CoLA; Warstadt et al., 2018) .

CoLA contains example sentences from linguistics publications labeled by experts for grammatical acceptability, and written to show subtle grammatical features.

Because minimal syntactic differences can separate acceptable sentences from unacceptable ones (What did Bo write a book about?

/ *What was a book about written by Bo?), and acceptability classifiers are more reliable when trained on GPT and BERT than on recurrent models, it stands to reason that GPT and BERT have better implicit knowledge of syntactic features relevant to acceptability.

Our goal in this paper is to develop an evaluation dataset that can locate which syntactic features that a model successfully learns by identifying the syntactic domains of CoLA in which it performs the best.

Using this evaluation set, we compare the syntactic knowledge of GPT and BERT in detail, and investigate the strengths of these models over the baseline BiLSTM model published by Warstadt et al. (2018) .

The analysis set includes expert annotations labeling the entire CoLA development set for the presence of 63 fine-grained syntactic features.

We identify many specific syntactic features that make sentences harder to classify, and many that have little effect.

For instance, sentences involving unusual or marked argument structures are no harder than the average sentence, while sentences with long distance dependencies are hard to learn.

We also find features of sentences that accentuate or minimize the differences between models.

Specifically, the transformer models seem to learn long-distance dependencies much better than the recurrent model, yet have no advantage on sentences with morphological violations.

Sentence Embeddings Robust pretrained word embeddings like word2vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014) have been extemely successful and widely adopted in machine learning applications for language understanding.

Recent research tries to reproduce this success at the sentence level, in the form of reusable sentence embeddings with pretrained weights.

These rep- Table 1 : A random sample of sentences from the CoLA development set, shown with their original acceptability labels (= acceptable, *=unacceptable) and with a subset of our new phenomenon-level annotations.resentations are useful for language understanding tasks that require a model to classify a single sentence, as in sentiment analysis and acceptability classification; or a pair of sentences, as in paraphrase detection and natural language inference (NLI); or that require a model to generate text based on an input text, as in question-answering.

Early work in this area primarily uses recurrent models like Long Short-Term Memory (Hochreiter and Schmidhuber, 1997, LSTM) networks to reduce variable length sequences into fixed-length sentence embeddings.

Current state of the art sentence encoders are pretrained on language modeling or related tasks with unlabeled-data.

Among these, ELMo (Peters et al., 2018 ) uses a BiLSTM architecture, while GPT (Radford et al., 2018) and BERT (Devlin et al., 2018) use the Transformer architecture (Vaswani et al., 2017) .

Unlike most earlier approaches where the weights of the encoder are frozen after pretraining, the last two fine-tune the encoder on the downstream task.

With additional fine-tuning on secondary tasks like NLI, these are the top performing models on the GLUE benchmark (Phang et al., 2018) .

The evaluation and analysis of sentence embeddings is an active area of research.

One branch of this work uses probing tasks which can reveal how much syntactic information a sentence embedding encodes about, for instance, tense and voice (Shi et al., 2016) , sentence length and word content (Adi et al., 2017) , or syntactic depth and morphological number (Conneau et al., 2018) .Related work indirectly probes features of sentence embeddings using language understanding tasks with custom datasets manipulating specific grammatical features.

Linzen et al. (2016) uses several tasks including acceptability classification of sentences with manipulated verbal inflection to investigate whether LSTMs can identify violations in subject-verb agreement, and therefore a (potentially long distance) syntactic dependency.

Ettinger et al. (2018) test whether sentence embeddings encode the scope of negation and semantic roles using semi-automatically generated sentences exhibiting carefully controlled syntactic variation.

Kann et al. (2019) also semiautomatically generate data and use acceptability classification to test whether word and sentence embeddings encode information about verbs and their argument structures.

CoLA & Acceptability Classification The Corpus of Linguistic Acceptability (Warstadt et al., 2018 ) is a dataset of 10k example sentences including expert annotations for grammatical acceptability.

The sentences are example sentences taken from 23 theoretical linguistics publications, mostly about syntax, including undergraduate textbooks, research articles, and dissertations.

Such example sentences are usually labeled for acceptability by their authors or a small group of native English speakers.

A small random sample of the CoLA development set (with our added annotations) can be seen in Table 1 .Within computational linguistics, the acceptability classification task has been explored in var-ious settings.

Lawrence et al. (2000) train RNNs to do acceptability classification over sequences of POS tags corresponding to example sentences from a syntax textbook.

Wagner et al. (2009) also train RNNs, but using naturally occurring sentences that have been automatically manipulated to be unacceptable.

Lau et al. (2016) predict acceptability from language model probabilities, applying this technique to sentences from a syntax textbook, and sentences which were translated round-trip through various languages.

Lau et al. attempt to model gradient crowdsourced acceptability judgments, rather than binary expert judgments.

This reflects an ongoing debate about whether binary expert judgments like those in CoLA are reliable (Gibson and Fedorenko, 2010; Sprouse and Almeida, 2012) .

We remain agnostic as to the role of binary judgments in linguistic theory, taking the expert judgments in CoLA at face value.

However, Warstadt et al. (2018) measure human performance on a subset of CoLA (see TAB5 ), finding that new human annotators, while not in perfect agreement with the judgments in CoLA, still outperform the best neural network models by a wide margin.

We introduce a grammatically annotated version of the entire CoLA development set to facilitate detailed error analysis of acceptability classifiers.

These 1043 sentences are expert-labeled for the presence of 63 minor grammatical features organized into 15 major features.

Each minor feature belongs to a single major feature.

A sentence belongs to a major feature if it belongs to one or more of the relevant minor features.

The Appendix includes descriptions of each feature along with examples and the criteria used for annotation.

The 63 minor features and 15 major features are illustrated in TAB1 .

Considering minor features, an average of 4.31 features is present per sentence (SD=2.59).

The average feature is present in 71.3 sentences (SD=54.7).

Turning to major features, the average sentence belongs to 3.22 major features (SD=1.66), and the average major feature is present in 224 sentences (SD=112).

Every sentence is labeled with at least one feature.

The sentences were annotated manually by one of the authors, who is a PhD student with extensive training in formal linguistics.

The features were developed in a trial stage, in which the annotator performed a similar annotation with different annotation schema for several hundred sentences from CoLA not belonging to the development set.

Here we briefly summarize the feature set in order of the major features.

Many of these constructions are well-studied in syntax, and further background can be found in textbooks such as BID0 and Sportiche et al. (2013) .Simple This major feature contains only one minor feature, SIMPLE, including sentences with a syntactically simplex subject and predicate.

Pred(icate) These three features correspond to predicative phrases, including copular constructions, small clauses (I saw Bo jump), and resultatives/depictives (Bo wiped the table clean).Adjunct These six features mark various kinds of optional modifiers.

This includes modifiers of NPs (The boy with blue eyes gasped) or VPs (The cat meowed all morning), and temporal (Bo swam yesterday) or locative (Bo jumped on the bed).Argument types These five features identify syntactically selected arguments, differentiating, for example, obliques (I gave a book to Bo), PP arguments of NPs and VPs (Bo voted for Jones), and expletives (It seems that Bo left).Argument Alternations These four features mark VPs with unusual argument structures, including added arguments (I baked Bo a cake) or dropped arguments (Bo knows), and the passive (I was applauded).Imperative This contains only one feature for imperative clauses (Stop it!).Bind These are two minor features, one for bound reflexives (Bo loves himself), and one for other bound pronouns (Bo thinks he won).Question These five features apply to sentences with question-like properties.

They mark whether the interrogative is an embedded clause (I know who you are), a matrix clause (Who are you?), or a relative clause (Bo saw the guy who left); whether it contains an island out of which extraction is unacceptable (*What was a picture of hanging on the wall?); or whether there is pied-piping or a multiword wh-expressions (With whom did you eat?).

S-Syntax These seven features mark various unrelated syntactic constructions, including dislocated phrases (The boy left who was here earlier); movement related to focus or information structure (This I've gotta see ); coordination, subordinate clauses, and ellipsis (I can't); or sentencelevel adjuncts (Apparently, it's raining).Determiner These four features mark various determiners, including quantifiers, partitives (two of the boys), negative polarity items (I *do/don't have any pie), and comparative constructions.

Violations These three features apply only to unacceptable sentences, and only ones which are ungrammatical due to a semantic or morphological violation, or the presence or absence of a single salient word.

We wish to emphasize that these features are overlapping and in many cases are correlated, thus not all results from using this analysis set will be independent.

We analyzed the pairwise Matthews Correlation Coefficient (MCC; Matthews, 1975) of the 63 minor features (giving 1953 pairs), and of the 15 major features (giving 105 pairs).

MCC is a special case of Pearson's r for Boolean variables.

1 These results are summarized in TAB3 .

Regarding the minor features, 60 pairs had a correlation of 0.2 or greater, 17 had a correlation of 0.4 or greater, and 6 had a correlation of 0.6 or greater.

None had an anti-correlation of greater magnitude than -0.17.

Turning to the major features, 6 pairs had a correlation of 0.2 or greater, and 2 had an anti-correlation of greater magnitude than -0.2.We can see at least three reasons for these observed correlations.

First, some correlations can be attributed to overlapping feature definitions.

For instance, EXPLETIVE arguments (e.g. There are birds singing) are, by definition, non-canonical arguments, and thus are a subset of ADD ARG.

However, some added arguments, such as benefactives (Bo baked Mo a cake), are not expletives.

Second, some correlations can be attributed to grammatical properties of the relevant constructions.

For instance, QUESTION and AUX are correlated because main-clause questions in English require subject-aux inversion and in many cases the insertion of auxiliary do (Do lions meow?).

Third, some correlations may be a consequence of the sources sampled in CoLA and the phenomena they focus on.

For instance, the unusually high correlation of EMB-Q and ELLIPSIS/ANAPHOR can be attributed to (Chung et al., 1995) , which is an article about the sluicing construction involving ellipsis of an embedded interrogative (e.g. I saw someone, but I don't know who).Finally, two strongest anti-correlations between major features are between SIMPLE and the two features related to argument structure, ARGU-MENT TYPES and ARG ALTERN.

This follows from the definition of SIMPLE, which excludes any sentence containing a large number or unusual configuration of arguments.

We train MLP acceptability classifiers for CoLA on top of three sentence encoders: (1) the CoLA baseline encoder with ELMo-style embeddings, (2) OpenAI GPT, and (3) BERT.

We use publicly available sentence encoders with pretrained weights.

2 LSTM encoder: CoLA baseline The CoLA baseline model is the sentence encoder with the highest performance on CoLA from Warstadt et al. The encoder uses a BiLSTM, which reads the sentence word-by-word in both directions, with maxpooling over the hidden states.

Similar to ELMo (Peters et al., 2018) , the inputs to the BiLSTM are the hidden states of a language model (only a forward language model is used in contrast with ELMo).

The encoder is trained on a real/fake discrimination task which requires it to identify whether a sentence is naturally occurring or automatically generated.

We train acceptability classifiers on CoLA using the CoLA baselines codebase with 20 random restarts, following the original authors' transfer-learning approach: The sentence encoder's weights are frozen, and the sentence embedding serves as input to an MLP with a single hidden layer.

All hyperparameters are held constant across restarts.

Transformer encoders: GPT and BERT In contrast with recurrent models, GPT and BERT use a self attention mechanism which combines representations for each (possibly non-adjacent) pair of words to give a sentence embedding.

GPT is trained using a standard language modeling task, while BERT is trained with masked language modeling and next sentence prediction tasks.

For each encoder, we use the jiant toolkit 3 to train 20 random restarts on CoLA feeding the pretrained models published by these authors into a single output layer.

Following the methods of the original authors, we fine-tune the encoders during training on CoLA.

All hyperparameters are held constant across restarts.

The overall performance of the three sentence encoders is shown in TAB5 .

Performance on CoLA is measured using MCC (Warstadt et al., 2018) .

We present the best single restart for each encoder, the mean over restarts for an encoder, and the result of ensembling the restarts for a given encoder, i.e. taking the majority classification for a given sentence, or the majority label of acceptable if tied.

4 For BERT results, we exclude 5 out of the 20 restarts because they were degenerate (MCC=0).

Across the board, BERT outperforms GPT, which outperforms the CoLA baseline.

However, BERT and GPT are much closer in performance than they are to CoLA baseline.

While ensemble performance exceeded the average for BERT and GPT, it did not outperform the best single model.

The results for the major features and minor features are shown in FIG1 , respectively.

For each feature, we measure the MCC of the sentences including that feature.

We plot the mean of these results across the different restarts for each model, and error bars mark the mean ±1 standard deviation.

For the VIOLATIONS features, MCC is technically undefined because these features only contain unacceptable sentences.

We report MCC in these cases by including for each feature a single acceptable example that is correctly classified by all models.

Comparison across features reveals that the presence of certain features has a large effect on performance, and we comment on some overall patterns below.

Within a given feature, the effect of model type is overwhelmingly stable, and resembles the overall difference in performance.

However, we observe several interactions, i.e. specific features where the relative performance of models does not track their overall relative performance.

Comparing Features Among the major features FIG1 , performance is universally highest on the SIMPLE sentences, and is higher than each model's overall performance.

Though these sentences are simple, we notice that the proportion of ungrammatical ones is on par with the entire dataset.

Otherwise we find that a model's performance on sentences of a given feature is on par with or lower than its overall performance, reflecting the fact that features mark the presence of unusual or complex syntactic structure.

Performance is also high (and close to overall performance) on sentences with marked argument structures (ARGUMENT TYPES and ARG(UMENT) ALT(ERNATION)).

While these models are still worse than human (overall) per- formance on these sentences, this result indicates that argument structure is relatively easy to learn.

Comparing different kinds of embedded content, we observe higher performance on sentences with embedded clauses (major feature=COMP CLAUSE) embedded VPs (major feature=TO-VP) than on sentences with embedded interrogatives (minor features=EMB-Q, REL CLAUSE).

An exception to this trend is the minor feature NO C-IZER, which labels complement clauses without a complementizer (e.g. I think that you're crazy).

Low performance on these sentences compared to most other features in COMP CLAUSE might indicate that complementizers are an important syntactic cue for these models.

As the major feature QUESTION shows, the difficulty of sentences with question-like syntax applies beyond just embedded questions.

Excluding polar questions, sentences with question-like syntax almost always involve extraction of a wh-word, creating a long-distance dependency between the wh-word and its extraction site, which may be difficult for models to recognize.

The most challenging features are all related to VIOLATIONS.

Low performance on INFL/AGR VIOLATIONS, which marks morphological violations (He washed yourself, This is happy), is especially striking because a relatively high proportion (29%) of these sentences are SIMPLE.

These models are likely to be deficient in encoding morphological features is that they are word level models, and do not have direct access sub-word information like inflectional endings, which indicates that these features are difficult to learn effectively purely from lexical distributions.

Finally, unusual performance on some features is due to small samples, and have a high standard deviation, suggesting the result is unreliable.

This includes CP SUBJ, FRAG/PAREN, IMPERATIVE, NPI/FCI, and COMPARATIVE.Comparing Models Comparing within-feature performance of the three encoders to their overall performance, we find they have differing strengths and weaknesses.

BERT stands out over other models in DEEP EMBED, which includes challenging sentences with doubly-embedded, as well as in several features involving extraction (i.e. longdistance dependencies) such as VP+EXTRACT and INFO-STRUC.

The transformer models show evidence of learning long-distance dependencies better than the CoLA baseline.

They outperform the CoLA baseline by an especially wide margin on BIND:REFL, which all involves establishing a dependency between a reflexive and its antecedent (Bo tries to love himself).

They also have a large advantage in DISLOCATION, in which expressions are separated from their dependents (Bo practiced on the train an important presentation).

The advantage of BERT and GPT may be due in part to their use of the transformer architecture.

Unlike the BiLSTM used by the CoLA baseline, the transformer uses a self-attention mechanism that associates all pairs of words regardless of distance.

In some cases models showed surprisingly good or bad performance, revealing possible idiosyncrasies of the sentence embeddings they output.

For instance, the CoLA baseline performs on par with the others on the major feature ADJUNCT, especially considering the minor feature PARTICLE (Bo looked the word up).Furthermore, all models struggle equally with sentences in VIOLATION, indicating that the advantages of the transformer models over the CoLA baseline does not extend to the detection of morphological violations (INFL/AGR VIOLATION) or single word anomalies (EXTRA/MISSING EXPR).

For comparison, we analyze the effect of sentence length on acceptability classifier performance.

The results are shown in FIG3 .

The results for the CoLA baseline are inconsistent, but do drop off as sentence length increases.

For BERT and GPT, performance decreases very steadily with length.

Exceptions are extremely short sentences (length 1-3), which may be challenging due to insufficient information; and extremely long sentences, where we see a small (but somewhat unreliable) boost in BERT's performance.

BERT and GPT are generally quite close in performance, except on the longest sentences, where BERT's performance is considerably better.

Using a new grammatically annotated analysis set, we identify several syntactic phenomena that are predictive of good or bad performance of current state of the art sentence encoders on CoLA.

We also use these results to develop hypotheses about why BERT is successful, and why transformer models outperform sequence models.

Our findings can guide future work on sentence embeddings.

A current weakness of all sentence encoders we investigate, including BERT, is the identification of morphological violations.

Future engineering work should investigate whether switching to a character-level model can mitigate this problem.

Additionally, transformer models appear to have an advantage over sequence models with long-distance dependencies, but still struggle with these constructions relative to more local phenomena.

It stands to reason that this performance gap might be widened by training larger or deeper transformer models, or training on longer or more complex sentences.

This analysis set can be used by engineers interested in evaluating the syntactic knowledge of their encoders.

Finally, these findings suggest possible controlled experiments that could confirm whether there is a causal relation between the presence of the syntactic features we single out as interesting and model performance.

Our results are purely correlational, and do not mark whether a particular construction is crucial for the acceptability of the sentence.

Future experiments following Ettinger et al. (2018) and Kann et al. (2019) can semi-automatically generate datasets manipulating, for example, length of long-distance dependencies, inflectional violations, or the presence of interrogatives, while controlling for factors like sentence length and word choice, in order determine the extent to which these features impact the quality of sentence embeddings.

(1) Included a. John owns the book. (37) b. Park Square has a festive air. (131)

c. *Herself likes Mary's mother.

FORMULA0 (2) Excluded a. Bill has eaten cake.

b. I gave Joe a book.

A.

These are sentences including the verb be used predicatively.

Also, sentences where the object of the verb is itself a predicate, which applies to the subject.

Not included are auxiliary uses of be or other predicate phrases that are not linked to a subject by a verb.

These sentences involve predication of a nonsubject argument by another non-subject argument, without the presence of a copula.

Some of these cases may be analyzed as small clauses. (see Sportiche et al., 2013, pp.

189-193)

These are adjuncts modifying noun phrases.

Adjuncts are (usually) optional, and they do not change the category of the expression they modify.

Single-word prenominal adjectives are excluded, as are relative clauses (this has another category).

These are adjuncts of VPs and NPs that specify a time or modify tense or aspect or frequency of an event.

Adjuncts are (usually) optional, and they do not change the category of the expression they modify.

These are adjuncts of VPs and NPs not described by some other category (with the exception of (6-7)), i.e. not temporal, locative, or relative clauses.

Adjuncts are (usually) optional, and they do not change the category of the expression they modify.

Prepositional Phrase arguments of NPs or APs are individual-denoting arguments of a noun or adjective which are marked by a proposition.

Arguments are selected for by the head, and they are (generally) not optional, though in some cases they may be omitted where they are understood or implicitly existentially quantified over.

Prepositional arguments introduced with by.

Usually, this is the (semantic) subject of a passive verb, but in rare cases it may be the subject of a nominalized verb.

Arguments are usually selected for by the head, and they are generally not optional.

In this case, the argument introduced with by is semantically selected for by the verb, but it is syntactically optional.

See Adger (2003, p.190) and Collins (2005) . (22 Expletives, or dummy arguments, are semantically inert arguments.

The most common expletives in English are it and there, although not all occurrences of these items are expletives.

Arguments are usually selected for by the head, and they are generally not optional.

In this case, the expletive occupies a syntactic argument slot, but it is not semantically selected by the verb, and there is often a syntactic variation without the expletive.

See Adger (2003, p.170-172) and Kim and Sells (2008, p.82-83 The passive voice is marked by the demotion of the subject (either complete omission or to a byphrase) and the verb appearing as a past participle.

In the stereotypical construction there is an auxiliary be verb, though this may be absent.

See Kim and Sells (2008, p.175-190) et al. (2013, p.163-186) and Sag et al. (2003, p.203-226) .

A.7.2 Binding:Other (Binding of Other Pronouns) These are cases in which a non-reflexive pronoun appears along with its antecedent.

This includes donkey anaphora, quantificational binding, and bound possessives, among other bound pronouns.

See Sportiche et al. (2013, p.163-186) and Sag et al. (2003, p.203-226) .

These are sentences in which the matrix clause is interrogative (either a wh-or polar question).

See Adger (2003, pp.282-213) , Kim and Sells (2008, pp.193-222), and Carnie (2013, p.315-350 Relative clauses are noun modifiers appearing with a relativizer (either that or a wh-word) and an associated gap.

See Kim and Sells (2008, p.223-244

<|TLDR|>

@highlight

We investigate the implicit syntactic knowledge of sentence embeddings using a new analysis set of grammatically annotated sentences with acceptability judgments.