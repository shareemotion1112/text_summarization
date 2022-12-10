Although there are more than 65,000 languages in the world, the pronunciations of many phonemes sound similar across the languages.

When people learn a foreign language, their pronunciation often reflect their native language's characteristics.

That motivates us to investigate how the speech synthesis network learns the pronunciation when multi-lingual dataset is given.

In this study, we train the speech synthesis network bilingually in English and Korean, and analyze how the network learns the relations of phoneme pronunciation between the languages.

Our experimental result shows that the learned phoneme embedding vectors are located closer if their pronunciations are similar across the languages.

Based on the result, we also show that it is possible to train networks that synthesize English speaker's Korean speech and vice versa.

In another experiment, we train the network with limited amount of English dataset and large Korean dataset, and analyze the required amount of dataset to train a resource-poor language with the help of resource-rich languages.

every speaker in the trained model could speak both English and Korean fluently.

We investigated 37 whether the phoneme embeddings from different languages are learned meaningful representations.

We found phonemes with similar pronunciation tend to stay closer than the others even across the 39 different languages.

From these results, we thought that the cross-lingual model would be possible to 40 generalize for a language with scarce amount of data when there is another language with abundant 41 data.

We trained cross-lingual TTS models while differing the amount of data for a resource-scarce 42 language.

Then we computed and compared character error rate (CER) of generated speeches from 43 each model by crowd-sourced human dictation.

To summarize, the contributions of this study are as follows: 45 1.

We successfully trained a cross-lingual multi-speaker TTS model using English and Korean 46 data in which no bilingual speaker is included.

3.

We showed how much data of a language is required to train a TTS model when we have a 50 large amount of data from another language.

DISPLAYFORM0 The 4 Spectrogram of generated speech using English phonemes and the nearest Korean phonemes using crowd-sourcing platform, and the average CER of the transcriptions is reported in TAB2 .

Also, the generated samples from each model are posted in the demo page.

The symbols in the parentheses are IPA symbols.

The symbol '-' in the parentheses denotes that there

was no IPA symbol for that phoneme.184 TAB2 aa (a) AE2 (ae) AY0 (aI) AW0 (aU) AW1 (aU) DISPLAYFORM0 The symbols in the parentheses are IPA symbols.

DISPLAYFORM1

<|TLDR|>

@highlight

Learned phoneme embeddings of multilingual neural speech synthesis network could represent relations of phoneme pronunciation between the languages.