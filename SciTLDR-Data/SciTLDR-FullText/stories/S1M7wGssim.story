Neural machine translation (NMT) models learn representations containing substantial linguistic information.

However, it is not clear if such information is fully distributed or if some of it can be attributed to individual neurons.

We develop unsupervised methods for discovering important neurons in NMT models.

Our methods rely on the intuition that different models learn similar properties, and do not require any costly external supervision.

We show experimentally that translation quality depends on the discovered neurons, and find that many of them capture common linguistic phenomena.

Finally, we show how to control NMT translations in predictable ways, by modifying activations of individual neurons.

of the i-th neuron in the encoder of the m-th model for the t-th word.1 These may be models

We consider four methods for ranking neurons, based on correlations between pairs of models.

Our 86 hypothesis is that different NMT models learn similar properties, and therefore similar important 87 neurons emerge in different models, akin to neural vision models (Li et al., 2016b) .

Our methods

capture different levels of localization/distributivity, as described next.

See Figure 1 for illustration.

Figure 1: An illustration of the correlation methods, showing how to compute the score for one neuron using each of the methods.

Here the number of models is M = 3, each having four neurons.

are well correlated with many other models, even if they are not the overall most correlated ones.

Regression ranking We perform linear regression (LinReg) from the full representation of an-

other model x m to the neuron x m i .

Then we rank neurons by the regression mean squared error.

This attempts to find neurons whose information might be distributed in other models.

SVCCA Singular vector canonical correlation analysis (SVCCA) is a recent method for analyzing 103 neural networks (Raghu et al., 2017 coefficients.

This attempts to capture information that may be distributed in less dimensions than 107 the whole representation.

In this case we get a ranking of directions, rather than individual neurons.

We want to verify that neurons ranked highly by the unsupervised methods are indeed important for 110 the NMT models.

We consider quantitative and qualitative techniques for verifying their importance.

Erasing Neurons We test importance of neurons by erasing some of them during translation.

Erasure is a useful technique for analyzing neural networks (Li et al., 2016a) .

Given a ranked list of 113 neurons π, where π(i) is the rank of neuron x i , we zero-out increasingly more neurons according to 114 the ranking π, starting from either the top or the bottom of the list.

Our hypothesis is that erasing 115 neurons from the top would hurt translation performance more than erasing from the bottom.

Concretely, we first run the entire encoder as usual, then zero out specific neurons from all source 117 hidden states {h 1 , . . .

, h n } before running the decoder.

For MaxCorr, MinCorr, and LinReg,

we zero out individual neurons.

To erase k directions found by SVCCA, we instead project the 119 embedding E (corresponding to all activations of a given model over a dataset) onto the space 120 spanned by the non-erased directions: us to compare models trained on the same language pairs but different training data, as well as 138 models trained on different language pairs.

We evaluate on the official test set.

DISPLAYFORM0

MT training We train 500 dimensional 2-layer LSTM encoder-decoder models with atten- next section, finding that top SVCCA directions focus mostly on identifying specific words.157 FIG5 shows the results of MaxCorr when erasing neurons from top and bottom, using models 158 trained on three language pairs.

In all cases, erasing from the top hurts performance more than 159 erasing from the bottom.

We found similar trends with other language pairs and ranking methods.

What kind of information is captured by the neurons ranked highly by each of our ranking methods?

Previous work found specific neurons in NMT that capture position of words in the sentence (Shi activation that is eliminated by conditioning on position in the sentence, calculated over the test set.

Similarly, it shows the percent of explained variance by conditioning on the current token identity.

We observe an interesting difference between the ranking methods.

LinReg and especially SVCCA,

which are both computed by using multiple neurons, tend to find information determined by the 170 identity of the current token.

MaxCorr and (especially) MinCorr tend to find position information.

This suggests that information about the current token is often distributed in multiple neurons, which 172 can be explained by the fact that tokens carry multiple kinds of linguistic information.

In contrast,

position is a fairly simple property that the NMT encoder can represent in a small number of neurons.

tense ("published", "disbursed", "held").

These results are obtained with a charCNN representation,

which is sensitive to common suffixes like "-ed", "-es".

However, this neuron also detects irregular 193 past tense verbs like "held", suggesting that it captures context in addition to sub-word information.

The neuron also makes some mistakes by activating weakly positively on nouns ending with "s"

("videos", "punishments"), presumably because it gets confused with the 3rd person present tense.

TAB5 shows correlations of neurons most correlated with this tense neuron, according to does not need to pay as much attention to tense when generating representations for the decoder.

Other Properties We found many more linguistic properties by visualizing top neurons ranked 205 by our methods, especially with MaxCorr.

We found neurons that activate on numbers, dates, 206 adjectives, plural nouns, auxiliary verbs, prepositions, and more.

We do not include a detailed 207 discussion for lack of space, and instead briefly discuss noun phrase segmentation, a compositional 208 property above the word level.

We obtained noun phrase segmentation (using Spacy) and classified 209 tokens as inside, outside, or beginning of a noun phrase (IOB scheme), and found high-scoring 210 neurons (60-80% accuracy) in every network.

Many of these neurons were ranked highly by the 211 MaxCorr method.

In contrast, other methods did not rank such neurons very highly.

We visualize the top scoring neuron (79%) from an English-Spanish model below.

Notice how the

In this section, we explore a potential benefit of finding important neurons with linguistically The Turkish sentences (1a, 2a) have no gender information-they can refer to either male or female.

But the MT system is biased to think that doctors are usually men and nurses are usually women,

so its generated translations (1b, 2b) represent these biases.

If we know the correct gender from

another source such as metadata, we may want to encourage the system to output a translation with 227 the correct gender.

We conjecture that if a given neuron matters to the model, then we can control the translation by 229 modifying its activations.

To do this, we first encode the source sentence as usual.

Before decoding,

we set the activation of a particular neuron in the encoder state to a value α (defined below).

To 231 evaluate our ability to control the translation, we design the following protocol: 232 1.

Tag the source and target sentences in the development set with a desired property, such as gender

(masculine/feminine).

We use Spacy for these tags.

to 67% success rate for changing past-to-present.

Modifications generally degrade BLEU, but the 249 loss at the best success rate is not large (2 BLEU points).

Controlling other properties seems more difficult, with the best success rate for controlling number 251 at 37%, using the 5 top number neurons.

Gender is the most difficult to control, with a 21% success 252 rate using the 5 top neurons.

Modifying even more neurons did not help.

We conjecture that these

properties are more distributed than tense, which makes controlling them more difficult.

Future 254 work can explore more sophisticated methods for controlling multiple neurons simultaneously.

We provide examples of controlling translation of number, gender, and tense.

While these are cherry-257 picked, they illustrate that the controlling procedure can work in multiple properties and languages.

Number TAB6 shows translation control results for a number neuron from an English-Spanish 259 model, which activates negatively/positively on plural/singular nouns.

The translation changes from 260 plural to singular as we increase the modification α.

We notice that using too high α values yields 261 nonsense translations, but with correct number: transitioning from the plural adjective particulares

("particular") to the singular adjectiveútil ("useful"), with valid translations in between.

Gender TAB6 shows examples of controlling gender translation for a gender neuron from the 264 same model, which activates negatively/positively on masculine/feminine nouns.

The translations 265 change from masculine to feminine synonyms as we increase the modification α.

Generally, we

found it difficult to control gender, as also suggested by the relatively low success rate.

Tense TAB6 shows examples of controlling tense when translating from English to five target 268 languages.

In all language pairs, we are able to change the translation from past to present by 269 modifying the activation of the tense neurons from the previous section TAB5 ).

In Spanish, we 270 find a transition from past to imperfect to present.

Interestingly, in Chinese, we had to use a fairly 271 large α value (in absolute terms), consistent with the fact that tense is not usually marked in Chinese.

We developed unsupervised methods for finding important neurons in NMT, and evaluated how 274 these neurons impact translation quality.

We analyzed several linguistic properties that are captured

by individual neurons using quantitative prediction tasks and qualitative visualizations.

We also 276 designed a protocol for controlling translations by modifying neurons that capture desired properties.

Our analysis can be extended to other NMT components (e.g. the decoder) and architec-278 tures (Gehring et al., 2017; Vaswani et al., 2017) , as well as other datasets from different domains,

and even other NLP tasks.

We believe that more work should be done to analyze the spectrum of lo-280 calized vs. distributed information in neural language representations.

We would also like to develop 281 more sophisticated ways to control translation output, for example by modifying representations in 282 variational NMT architectures (Zhang et al., 2016; Su et al., 2018

@highlight

Unsupervised methods for finding, analyzing, and controlling important neurons in NMT

@highlight

This work proposes finding "meaningful" neurons in Neural Machine Translation models by ranking based on correlation between pairs of models, different epochs, or different datasets, and proposes a controlling mechanism for the models.