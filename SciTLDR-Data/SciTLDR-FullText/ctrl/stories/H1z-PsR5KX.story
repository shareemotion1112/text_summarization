Neural machine translation (NMT) models learn representations containing substantial linguistic information.

However, it is not clear if such information is fully distributed or if some of it can be attributed to individual neurons.

We develop unsupervised methods for discovering important neurons in NMT models.

Our methods rely on the intuition that different models learn similar properties, and do not require any costly external supervision.

We show experimentally that translation quality depends on the discovered neurons, and find that many of them capture common linguistic phenomena.

Finally, we show how to control NMT translations in predictable ways, by modifying activations of individual neurons.

Neural machine translation (NMT) systems achieve state-of-the-art results by learning from large amounts of example translations, typically without additional linguistic information.

Recent studies have shown that representations learned by NMT models contain a non-trivial amount of linguistic information on multiple levels: morphological BID4 BID7 , syntactic BID31 , and semantic BID16 .

These studies use trained NMT models to generate feature representations for words, and use these representations to predict certain linguistic properties.

This approach has two main limitations.

First, it targets the whole vector representation and fails to analyze individual dimensions in the vector space.

In contrast, previous work found meaningful individual neurons in computer vision BID34 BID36 Bau et al., 2017, among others) and in a few NLP tasks BID18 BID27 BID25 .

Second, these methods require external supervision in the form of linguistic annotations.

They are therefore limited by available annotated data and tools.

In this work, we make initial progress towards addressing these limitations by developing unsupervised methods for analyzing the contribution of individual neurons to NMT models.

We aim to answer the following questions:• How important are individual neurons for obtaining high-quality translations?• Do individual neurons in NMT models contain interpretable linguistic information?• Can we control MT output by intervening in the representation at the individual neuron level?To answer these questions, we develop several unsupervised methods for ranking neurons according to their importance to an NMT model.

Inspired by work in machine vision BID22 , we hypothesize that different NMT models learn similar properties, and therefore similar important neurons should emerge in different models.

To test this hypothesis, we map neurons between pairs of trained NMT models using several methods: correlation analysis, regression analysis, and SVCCA, a recent method combining singular vectors and canonical correlation analysis BID28 .

Our mappings yield lists of candidate neurons containing shared information across models.

We then evaluate whether these neurons carry important information to the NMT model by masking their activations during testing.

We find that highly-shared neurons impact translation quality much more than unshared neurons, affirming our hypothesis that shared information matters.

Given the list of important neurons, we then investigate what linguistic properties they capture, both qualitatively by visualizing neuron activations and quantitatively by performing supervised classification experiments.

We were able to identify neurons corresponding to several linguistic phenomena, including morphological and syntactic properties.

Finally, we test whether intervening in the representation at the individual neuron level can help control the translation.

We demonstrate the ability to control NMT translations on three linguistic properties-tense, number, and gender-to varying degrees of success.

This sets the ground for controlling NMT in desirable ways, potentially reducing system bias to properties like gender.

Our work indicates that not all information is distributed in NMT models, and that many humaninterpretable grammatical and structural properties are captured by individual neurons.

Moreover, modifying the activations of individual neurons allows controlling the translation output according to specified linguistic properties.

The methods we develop here are task-independent and can be used for analyzing neural networks in other tasks.

More broadly, our work contributes to the localist/distributed debate in artificial intelligence and cognitive science BID13 by investigating the important case of neural machine translation.

Much recent work has been concerned with analyzing neural representations of linguistic units, such as word embeddings BID20 BID26 , sentence embeddings BID0 BID12 BID6 , and NMT representations at different linguistic levels: morphological BID4 , syntactic BID31 , and semantic BID16 BID5 .

These studies follow a common methodology of evaluating learned representations on external supervision by training classifiers or measuring other kinds of correlations BID3 .

Thus they are limited to the available supervised annotation.

In addition, these studies do not typically consider individual dimensions.

In contrast, we propose intrinsic unsupervised methods for detecting important neurons based on correlations between independently trained models.

A similar approach was used to analyze vision models BID22 , but to the best of our knowledge these ideas were not applied to NMT or other NLP models before.

In computer vision, individual neurons were shown to capture meaningful information BID34 BID36 BID2 .

Even though some doubts were cast on the importance of individual units BID24 , recent work stressed their contribution to predicting specific object classes via masking experiments similar to ours BID37 .

A few studies analyzed individual neurons in NLP.

For instance, neural language models learn specific neurons that activate on brackets BID18 , sentiment BID27 , and length BID25 .

Length-specific neurons were also found in NMT BID30 ), but generally not much work has been devoted to analyzing individual neurons in NMT.

We aim to address this gap.

Much recent work on analyzing NMT relies on supervised learning, where NMT representations are used as features for predicting linguistic annotations (see Section 2).

However, such annotations may not be available, or may constrain the analysis to a particular scheme.

Instead, we propose to use different kinds of correlations between neurons from different models as a measure of their importance.

Suppose we have M such models and let h Figure 1: An illustration of the correlation methods, showing how to compute the score for one neuron using each of the methods.

Here the number of models is M = 3, each having four neurons.

We consider four methods for ranking neurons, based on correlations between pairs of models.

Our hypothesis is that different NMT models learn similar properties, and therefore similar important neurons emerge in different models, akin to neural vision models BID22 .

Our methods capture different levels of localization/distributivity, as described next.

See Figure 1 for illustration.

The maximum correlation (MaxCorr) of neuron x m i looks for the highest correlation with any neuron in all other models: DISPLAYFORM0 where ρ(x, y) is the Pearson correlation coefficient between x and y. We then rank the neurons in model m according to their MaxCorr score.

We repeat this procedure for every model m. This score looks for neurons that capture properties that emerge strongly in two separate models.

Neurons in model m are ranked according to their MinCorr score.

This tries to find neurons that are well correlated with many other models, even if they are not the overall most correlated ones.

Regression ranking We perform linear regression (LinReg) from the full representation of another model x m to the neuron x m i .

Then we rank neurons by the regression mean squared error.

This attempts to find neurons whose information might be distributed in other models.

SVCCA Singular vector canonical correlation analysis (SVCCA) is a recent method for analyzing neural networks BID28 .

In our implementation, we perform PCA on each model's representations x m and take enough dimensions to account for 99% of the variance.

For each pair of models, we obtain the canonically correlated basis, and rank the basis directions by their CCA coefficients.

This attempts to capture information that may be distributed in less dimensions than the whole representation.

In this case we get a ranking of directions, rather than individual neurons.

We want to verify that neurons ranked highly by the unsupervised methods are indeed important for the NMT models.

We consider quantitative and qualitative techniques for verifying their importance.

Erasing Neurons We test importance of neurons by erasing some of them during translation.

Erasure is a useful technique for analyzing neural networks BID21 .

Given a ranked list of neurons π, where π(i) is the rank of neuron x i , we zero-out increasingly more neurons according to the ranking π, starting from either the top or the bottom of the list.

Our hypothesis is that erasing neurons from the top would hurt translation performance more than erasing from the bottom.

Concretely, we first run the entire encoder as usual, then zero out specific neurons from all source hidden states {h 1 , . . .

, h n } before running the decoder.

For MaxCorr, MinCorr, and LinReg, we zero out individual neurons.

To erase k directions found by SVCCA, we instead project the embedding E (corresponding to all activations of a given model over a dataset) onto the space spanned by the non-erased directions: DISPLAYFORM0 , where C is the CCA projection matrix with the first or last k columns removed.

This corresponds to erasing from the top or bottom.

Supervised Verification While our focus is on unsupervised methods for finding important neurons, we also utilize supervision to verify our results.

Importantly, these experiments are done post-hoc, after having a candidate neuron to examine.

Since training a supervised classifier on every neuron is costly, we instead report simple metrics that can be easily computed.

Specifically, we sometimes report the expected conditional variance of neuron activations conditioned on some property.

In other cases we found it useful to estimate a Gaussian mixture model (GMM) for predicting a label and measure its prediction quality.

The number of mixtures in the GMM is set according to the number of classes in the predicted property (e.g. 2 mixtures when predicting tokens inside or outside of parentheses), and its parameters are estimated using the mean and variance of the neuron activation conditioned on each class.

We obtain linguistic annotations with Spacy: spacy.io.Visualization Interpretability of machine learning models remains elusive (Lipton, 2016), but visualizing can be an instructive technique.

Similar to previous work analyzing neural networks in NLP BID11 BID18 BID17 , we visualize activations of neurons and observe interpretable behavior.

We will illustrate this with example heatmaps below.

Data We use the United Nations (UN) parallel corpus BID38 for all experiments.

We train models from English to 5 languages: Arabic, Chinese, French, Russian, and Spanish, as well as an English-English auto-encoder.

For each target language, we train 3 models on different parts of the training set, each with 500K sentences.

In total, we have 18 models.

This setting allows us to compare models trained on the same language pairs but different training data, as well as models trained on different language pairs.

We evaluate on the official test set.

MT training We train 500 dimensional 2-layer LSTM encoder-decoder models with attention BID1 .

In order to study both word and sub-word properties, we use a word representation based on a character convolutional neural network (charCNN) as input to both encoder and decoder, which was shown to learn morphology in language modeling and NMT BID19 BID4 .3 While we focus here on recurrent NMT, our approach can be applied to other models like the Transformer BID33 , which we leave for future work.

FIG2 shows erasure results using the methods from Section 3.1, on an English-Spanish model.

For all four methods, erasing from the top hurts performance much more than erasing from the bottom.

This confirms our hypothesis that neurons ranked higher by our methods have a larger impact on translation quality.

Comparing erasure with different rankings, we find similar patterns with MaxCorr, MinCorr, and LinReg: erasing the top ranked 10% (50 neurons) degrades BLEU by 15-20 points, while erasing the bottom 10% neurons only hurts by 2-3 points.

In contrast, erasing SVCCA directions results in rapid degradation: 15 BLEU point drop when erasing 1% (5) of the top directions, and poor performance when erasing 10% (50).

This indicates that top SVCCA directions capture very important information in the model.

We analyze these top neurons and directions in the next section, finding that top SVCCA directions focus mostly on identifying specific words.

FIG3 shows the results of MaxCorr when erasing neurons from top and bottom, using models trained on three language pairs.

In all cases, erasing from the top hurts performance more than erasing from the bottom.

We found similar trends with other language pairs and ranking methods.

What kind of information is captured by the neurons ranked highly by each of our ranking methods?

Previous work found specific neurons in NMT that capture position of words in the sentence BID30 .

Do our methods capture similar properties?

Indeed, we found that many top neurons capture position.

For instance, TAB0 shows the top 10 ranked neurons from an English-Spanish model according to each of the methods.

The table shows the percent of variance in neuron activation that is eliminated by conditioning on position in the sentence, calculated over the test set.

Similarly, it shows the percent of explained variance by conditioning on the current token identity.

We observe an interesting difference between the ranking methods.

LinReg and especially SVCCA, which are both computed by using multiple neurons, tend to find information determined by the identity of the current token.

MinCorr and to a lesser extent MaxCorr tend to find position information.

This suggests that information about the current token is often distributed in multiple neurons, which can be explained by the fact that tokens carry multiple kinds of linguistic information.

In contrast, position is a fairly simple property that the NMT encoder can represent in a small number of neurons.

That fact that many top MinCorr neurons capture position suggests that this kind of information is captured in multiple models in a similar way.

Neurons that activate on specific tokens or capture position in the sentence are important in some of the methods, as shown in the previous section.

But they are not highly ranked in all methods and are also less interesting from the perspective of capturing language information.

In this section, we investigate several linguistic properties by measuring predictive capacity and visualizing neuron activations.

The supplementary material discusses more properties.

Further analysis of linguistically interpretable neurons is available in BID8 .Parentheses TAB1 shows top neurons from each model for predicting that tokens are inside/outside of parentheses, quotes, or brackets, estimated by a GMM model.

Often, the parentheses neuron is unique (low scores for the 2nd best neuron), suggesting that this property tends to be relatively localized.

Generally, neurons that detect parentheses were ranked highly in most models by the MaxCorr method, indicating that they capture important patterns in multiple networks.

Tense We annotated the test data for verb tense (with Spacy) and trained a GMM model to predict tense from neuron activations.

FIG6 shows activations of a top-scoring neuron (0.66 F 1 ) from the English-Arabic model on the first 5 test sentences.

It tends to activate positively (red color) on present tense ("recognizes", "recalls", "commemorate") and negatively (blue color) on past tense ("published", "disbursed", "held").

These results are obtained with a charCNN representation, which is sensitive to common suffixes like "-ed", "-es".

However, this neuron also detects irregular past tense verbs like "held", suggesting that it captures context in addition to sub-word information.

The neuron also makes some mistakes by activating weakly positively on nouns ending with "s" ("videos", "punishments"), presumably because it gets confused with the 3rd person present tense.

Similarly, it activates positively on "Spreads", even though it functions as a noun in this context.

TAB2 shows correlations of neurons most correlated with this tense neuron, according to MaxCorr.

All these neurons are highly predictive of tense: all but 3 are in the top 10 and 8 out of 15 (non-auto-encoder) neurons have the highest F 1 score for predicting tense.

The auto-encoder English models are an exception, exhibiting much lower correlations with the English-Arabic tense neuron.

This suggests that tense emerges in a "real" NMT model, but not in an auto-encoder that only learns to copy.

Interestingly, English-Chinese models have somewhat lower correlated neurons with the tense neuron, possibly due to the lack of explicit tense marking in Chinese.

The encoder does not need to pay as much attention to tense when generating representations for the decoder.

Other Properties We found many more linguistic properties by visualizing top neurons ranked by our methods, especially with MaxCorr.

We briefly mention some of these here and provide more details and quantitative results in the appendix.

We found neurons that activate on numbers, dates, adjectives, plural nouns, auxiliary verbs, and more.

We also investigated noun phrase segmentation, a compositional property above the word level, and found high-scoring neurons (60-80% accuracy) in every network.

Many of these neurons were ranked highly by the MaxCorr method.

In contrast, other methods did not rank such neurons very highly.

See TAB4 in the appendix for the full results.

Some neurons have quite complicated behavior.

For example, when visualizing neurons highly ranked by MaxCorr we found a neuron that activates on numbers in the beginning of a sentence, but not elsewhere (see FIG15 in the appendix).

It would be difficult to conceive of a supervised prediction task which would capture this behavior a-priori, without knowing what to look for.

Our unsupervised methods are flexible enough to find any neurons deemed important by the NMT model, without constraining the analysis to properties for which we have supervised annotations.

In this section, we explore a potential benefit of finding important neurons with linguistically meaningful properties: controlling the translation output.

This may be important for mitigating biases in neural networks.

For instance, gender stereotypes are often reflected in automatic translations, as the following motivating examples from Google Translate demonstrate.

The Turkish sentences (1a, 2a) have no gender information-they can refer to either male or female.

But the MT system is biased to think that doctors are usually men and nurses are usually women, so its generated translations (1b, 2b) represent these biases.

If we know the correct gender from another source such as metadata, we may want to encourage the system to output a translation with the correct gender.

We make here a modest step towards this goal by intervening in neuron activations to induce a desired translation.

We conjecture that if a given neuron matters to the model, then we can control the translation in predictable ways by modifying its activations.

To do this, we first encode the source sentence as usual.

Before decoding, we set the activation of a particular neuron in the encoder state to a value α, which is a function of the mean activations over a particular property (defined below).

To evaluate our ability to control the translation, we design the following protocol:1.

Tag the source and target sentences in the development set with a desired property, such as gender (masculine/feminine).

We use Spacy for these tags.2.

Obtain word alignments for the development set using an alignment model trained on 2 million sentences of the UN data.

We use fast align BID10 ) with default settings.3.

For every neuron in the encoder, predict the target property on the word aligned to its source word activations using a supervised GMM model.

4.

For every word having a desired property, modify the source activations of the top k neurons found in step 3 and generate a modified translation.

The modification value is defined as α = µ 1 + β(µ 1 − µ 2 ), where µ 1 and µ 2 are mean activations of the property we modify from and to, respectively (e.g. modifying gender from masculine to feminine), and β is a hyperparameter.5.

Tag the output translation and word-align it to the source.

Declare success if the source word was aligned to a target word with the desired property value (e.g. feminine).6.1 RESULTS FIG10 shows translation control results in an English-Spanish model.

We report success rate-the percentage of cases where the word was aligned to a target word with the desired property-and the effect on BLEU scores, when varying α.

Our tense control results are the most successful, with up to 67% success rate for changing past-to-present.

Modifications generally degrade BLEU, but the loss at the best success rate is not large (2 BLEU points).

Appendix A.2 provides more tense results.

Controlling other properties seems more difficult, with the best success rate for controlling number at 37%, using the top 5 number neurons.

Gender is the most difficult to control, with a 21% success rate using the top 5 neurons.

Modifying even more neurons did not help.

We conjecture that these properties are more distributed than tense, which makes controlling them more difficult.

Future work can explore more sophisticated methods for controlling multiple neurons simultaneously.

We provide examples of controlling translation of number, gender, and tense.

While these are cherrypicked examples, they illustrate that the controlling procedure can work in multiple properties and languages.

We discuss language-specific patterns below.

Chinese -/-50 DISPLAYFORM0 Number TAB3 shows translation control results for a number neuron from an English-Spanish model, which activates negatively/positively on plural/singular nouns.

The translation changes from plural to singular as we increase the modification α.

We notice that using too high α values yields nonsense translations, but with correct number: transitioning from the plural adjective particulares ("particular") to the singular adjectiveútil ("useful") .

In between, we see a nice transition between plural and singular translations.

Interestingly, the translations exhibit correct agreement between the modified noun and its adjectives and determines, e.g., Las partes interesadas vs. La parte interesada.

This is probably due to the strong language model in the decoder.

Gender TAB3 shows examples of controlling gender translation for a gender neuron from the same model, which activates negatively/positively on masculine/feminine nouns.

The translations of "parties" and "questions" change from masculine to feminine synonyms as we increase the modification α.

Generally, we found it difficult to control gender, as also suggested by the relatively low success rate.

Tense TAB3 shows examples of controlling tense when translating from English to five target languages.

In all language pairs, we are able to change the translation from past to present by modifying the activation of the tense neurons from the previous section TAB2 .

Occasionally, modifying the activation on a single word leads to a change in phrasing; in Arabic the translation changes to "the efforts that the authorities invest".

In Spanish, we find a transition from past (apoyó) to imperfect (apoyaba) to present (apoya).

Interestingly, in Chinese, we had to use a fairly large α value (in absolute terms), consistent with the fact that tense is not usually marked in Chinese.

In fact, our modification generates a Chinese expression (正在) that is used to express an action in progress, similar to English "-ing", resulting in the meaning "is supporting".

Neural machine translation models learn vector representations that contain linguistic information while being trained solely on example translations.

In this work, we developed unsupervised methods for finding important neurons in NMT, and evaluated how these neurons impact translation quality.

We analyzed several linguistic properties that are captured by individual neurons using quantitative prediction tasks and qualitative visualizations.

We also designed a protocol for controlling translations by modifying neurons that capture desired properties.

Our analysis can be extended to other NMT components (e.g. the decoder) and architectures BID14 BID33 , as well as other tasks.

We believe that more work should be done to analyze the spectrum of localized vs. distributed information in neural language representations.

We would also like to expand the translation control experiments to other architectures and components (e.g. the decoder), and to develop more sophisticated ways to control translation output, for example by modifying representations in variational NMT architectures BID35 BID32 .

Our code is publicly available as part of the NeuroX toolkit BID9 .

A.1 NOUN PHRASE SEGMENTATION TAB4 shows the top neurons from each network by accuracy when classifying interior, exterior, or beginning of a noun phrase.

The annotations were obtained with Spacy.

We found high-scoring neurons (60-80% accuracy) in every network.

Many of these neurons were ranked highly by the MaxCorr ranking methods.

In contrast, other correlation methods did not rank such neurons very highly.

Thus there is correspondence between a high rank by our intrinsic unsupervised measure MaxCorr and the neuron's capacity to predict external annotation.

We provide additional translation control results.

TAB5 shows the tense results using the best modification value from FIG10 .

We report the number of times the source word was aligned to a target word which is past or present, or to multiple words that include both or neither of these tenses.

The success rate is the percentage of cases where the word was aligned to a target word with the desired tense.

By modifying the activation of only one neuron (the most predictive one), we were able to change the translation from past to present in 67% of the times and vice-versa in 49% of the times.

In many other cases, the tense was erased, that is, the modified source word was not aligned to any tensed word, which is a partial success.

Noun phrases We visualize the top scoring neuron (79%) from an English-Spanish model in FIG13 .

Notice how the neuron activates positively (red color) on the first word in the noun phrases, but negatively (blue color) on the rest of the noun phrase (e.g. "Regional" in "Regional Service Centre").

Dates and Numbers FIG14 shows activations of neurons capturing dates and numbers.

These neurons were ranked highly (top 30) by MaxCorr when ranking an English-Arabic model trained with charCNN representations.

We note that access to character information leads to many neurons capturing sub-word information such as years (4-digit numbers).

The first neuron is especially sensitive to month names ("May", "April").

The second neuron is an approximate year-detector: it is sensitive to years ("2015") as well as other tokens with four digits ("7439th", "10.15").

List items FIG15 shows an interesting case of a neuron that is sensitive to the appearance of two properties simultaneously: position in the beginning of the sentence and number format.

Notice that it activates strongly (negatively) on numbers when they open a sentence but not in the middle of the sentence.

Conversely, it does not activate strongly on non-number words that open a sentence.

This neuron aims to capture patterns of opening list items.

In the main experiments in the paper, we have used models trained on different parts of the training data, as well as on different language pairs.

However, our methodology can be applied to any collection of models that we think should exhibit correlations in their neurons.

We have verified that this approach works with model checkpoints from different epochs of the same training run.

Concretely, we computed MaxCorr scores for the last checkpoint in two models-English-Spanish and English-Arabic-when comparing to other checkpoints.

In both cases, we found highly correlated neurons across checkpoints, especially in the last few checkpoints.

We also observed that erasing the top neurons hurt translation performance more than erasing the bottom neurons, similar to the findings in Section 5.1.

Moreover, we noticed a significant overlap between the top ranked neurons in this case and the ones found by correlating with other models, as in the rest of the paper.

In particular, for the English-Spanish model, we found that 8 out of 10 and 34 out of 50 top ranked neurons are the same in these two rankings.

For the English-Arabic model, we found a similar behavior (7 out of 10 and 33 out of 50 top ranked neurons are the same).

This indicates that our method may be applied to different checkpoints as well.

3.5% 96% Punctuation/conjunctions: ",", ";", "Also", "also", "well".

120 0.094% 84% Plural noun detector.

Best F1-score (0.85) for retrieving plural nouns.

269 0.1% 80% Spanish noun gender detector.

Very positive for "islands", "activities", "measures" -feminine.

Very negative for "states", "principles", "aspects" -masculine.

0.44% 89% Prepositions: "of", "or", "United", "de".

1861.6% 69% Conjunctions: "also", "therefore", "thus", "alia".

244 54% 15% Position.

Conjunctions: "and", "or", "well", "addition".

139 0.86% 93% Punctuation/conjunctions: ",", ".", "-", "alia".

494 3.5% 96% Punctuation/conjunctions: ",", ";", "also", "well".

342 88% 7.9% Position.

228 0.38% 96% Possibly determiners: ""s", "the", "this", "on", "that".

3171.5% 83% Indefinite determiners: "(", "one", "a", "an".

367 0.44% 89% Prepositions. "of", "for", "United", "de", "from", "by", "about".

106 0.25% 92% Possibly determiners: "that", "this", "which", "the".

383 67% 6.5% Position.

485 64% 10% Position.

186 1.6% 69% Conjunctions. "also", "therefore", "thus", "alia".

272 2% 73% Tokens that mean "in other words": "(", "namely", "i.e.", "see", "or".

124 77% 48% Position.

480 70% 12% Position.

1871.1% 87% Unknown: "them", "well", "be", "would", "remain".

201 0.14% 73% Tokens that mean "regarding": "on", "in", "throughout", "concerning", "regarding".

67 0.27% 71% Unknown: "united", "'s", "by", "made", "from".

154 63% 17% Position.

72 0.32% 89% Verbs suggesting equivalence: "is", "was", "are", "become", "constitute", "represent".

Position Token Comments 86% 26% Position 1.6% 90% Detects "the".

7.5% 85% Conjunctions: "and", "well", "or".

20% 79% Determiners: "the", "this", "these", "those".

1.1% 89% Possibly conjunctions: negative for "and", "or", "nor", positive for "been", "into", "will".

10% 76% Punctuation/conjunctions: positive for ",", ";", "." "-", negative for "and".

30% 57% Possibly verbs: "been", "will", "be", "shall".

24% 55% Possibly date detector.

23% 60% Possibly adjective detector.

18% 63% Unknown.

4.5% 88% Punctuation: ".

", ",", ";" 9.8% 69% Forms of "to be": "is", "will", "shall", "would", "are".

1.7% 77% Combined dates/prepositions/parentheses: negative for "in", "at", ".

", positive for dates and in quotes/parentheses/brackets.

Noisy.

16% 25% Activates for a few words after "and".

14% 63% Possibly plural noun detector.

0.8% 73% Spanish noun gender detector.

11% 61% Possibly singular noun detector.

13% 58% Possibly possessives: "its", "his", "their".

1.4% 73% Spanish noun gender detector.

5.6% 53% Unknown.

<|TLDR|>

@highlight

Unsupervised methods for finding, analyzing, and controlling important neurons in NMT

@highlight

This paper presents unsupervised approaches to discovering important neurons in neural machine translation systems and analyzes linguistic properties controlled by those neurons.

@highlight

Unsupervised methods for ranking neurons in machine translation where important neurons are thus identified and used to control the MT output.