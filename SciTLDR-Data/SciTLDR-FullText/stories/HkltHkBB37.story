We present a simple idea that allows to record a speaker in a given language and synthesize their voice in other languages that they may not even know.

These techniques open a wide range of potential applications such as cross-language communication, language learning or automatic video dubbing.

We call this general problem multi-language speaker-conditioned speech synthesis and we present a simple but strong baseline for it.



Our model architecture is similar to the encoder-decoder Char2Wav model or Tacotron.

The main difference is that, instead of conditioning on characters or phonemes that are specific to a given language, we condition on a shared phonetic representation that is universal to all languages.

This cross-language phonetic representation of text allows to synthesize speech in any language while preserving the vocal characteristics of the original speaker.

Furthermore, we show that fine-tuning the weights of our model allows us to extend our results to speakers outside of the training dataset.

Our approach is to build a model able to generate speech in multiple languages.

The model is trained 23 with multiple speakers to let the model be aware of the variations between speakers and also to 24 disentangle speech content from speaker identity.

Once the model is trained, we bias the generation 25 process so that it sounds like a specific speaker.

This speaker doesn't have to be in the training data.

Our work builds upon recent developments in neural network based speech synthesis [Sotelo et al., 28 2017 , Ping et al., 2017 , Shen et al., 2017 , Van Den Oord et al., 2016 languages to a universal representation.

Our model is able to accomplish zero-shot accent transfer, which is very similar to zero-shot machine 42 translation, done by grounding the input from different languages to a common neural representation 43 space, followed by decoding in the audio space [Johnson et al., 2016] .

The training data consists of audio-transcript pairs.

The transcript is translated into its IPA equivalent 61 before being fed to the model and the audio is transformed into an intermediate representation (e.g.

WORLD vocoder parameters or spectrogram).

Each speaker within the training dataset only speaks 63 a single language.

However, at synthesis time, we are able to take any combination of speaker and 64 language, and produce natural sounding speech in the voice of the speaker and in the accent matching 65 that of the language.

speakers.

Crucially, we apply a smaller learning rate to the encoder and decoder parts of the models,

and a higher one for the speaker embedding.

This improved speaker fidelity considerably.

After fine-tuning, the model is able to generate any text in any language 1 with the new speaker's 76 vocal identity.

We conduct experiments on our models trained in two distinct settings.

First, we train our model 79 with data in two languages (Bilingual Model).

Second, we train our model with data in six languages

(Multilingual Model).

For these experiments, we used several datasets.

We used an internal English dataset composed of 82 approximately 20000 speakers, with about 10 utterances per speaker.

We also used the TIMIT dataset

[ Garofolo et al., 1993] and DIMEx100 [Pineda, 2009] .

DIMEx100 is a Spanish dataset composed of 84 100 Spanish native speakers, with about 60 2-seconds utterances per speaker.

For all the experiments we provide audio samples 2 rather than an exhaustive quantitative analysis.

language.

We show that the model is able to generate in any language for any speaker in the dataset.

The model also shows robust performance on new, out-of-sample speakers after the fine-tuning step 97 (see FIG4 .

@highlight

We present a simple idea that allows to record a speaker in a given language and synthesize their voice in other languages that they may not even know.