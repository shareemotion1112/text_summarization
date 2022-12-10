We perform an in-depth investigation of the suitability of self-attention models for character-level neural machine translation.

We test the standard transformer model, as well as a novel variant in which the encoder block combines information from nearby characters using convolution.

We perform extensive experiments on WMT and UN datasets, testing both bilingual and multilingual translation to English using up to three input languages (French, Spanish, and Chinese).

Our transformer variant consistently outperforms the standard transformer at the character-level and converges faster while learning more robust character-level alignments.

Most existing Neural Machine Translation (NMT) models operate on the word or subword-level.

Often, these models are memory inefficient because of large vocabulary size.

Character-level models (Lee et al., 2017; Cherry et al., 2018) instead work directly on raw characters, resulting in a more compact language representation, while mitigating out-of-vocabulary (OOV) problems (Luong and Manning, 2016) .

They are especially suitable for multilingual translation, where multiple languages can be modelled using the same character vocabulary.

Multilingual training can lead to improvements in the overall performance without any increase in model complexity (Lee et al., 2017) .

It also circumvents the need to train separate models for each language pair.

Models based on self-attention have achieved excellent performance on a number of tasks including machine translation (Vaswani et al., 2017) and representation learning (Devlin et al., 2019; Yang et al., 2019) .

Despite the success of these models, no previous work has considered their suitability for character-level translation, with the In this work, we perform an in-depth investigation of the suitability of self-attention models for character-level translation.

We consider two models: the standard transformer from (Vaswani et al., 2017) ; as well as a novel variant, which we call the convtransformer (Figure 1 , Section 3).

The latter uses convolution to facilitate interactions among nearby character representations.

We evaluate these models on both bilingual and multilingual translation to English, using up to three input languages: French (FR), Spanish (ES), and Chinese (ZH).

We compare their translation performance on close (e.g., FR and ES) and on distant (e.g., FR and ZH) input languages (Section 5.1), and we analyze their learned character alignments (Section 5.2).

We find that self-attention models work surprisingly well for character-level translation, performing competitively with equivalent subword-level models while requiring up to 60% fewer parameters.

At the character-level, the convtransformer performs better than the standard transformer, converging faster and producing more robust alignments.

Fully character-level translation was first tackled in (Lee et al., 2017) , who proposed a recurrent encoder-decoder model similar to the one in (Bahdanau et al., 2015) .

Their encoder combines convolutional layers with max pooling and highway layers to construct intermediate representations of segments of nearby characters.

Their decoder network autoregressively generates the output translation one character at a time, utilizing attention on the encoded representations.

Lee et al. (2017) 's approach showed promising results on multilingual translation in particular.

Without any architectural modifications, training on multiple source languages yielded performance improvements while also acting as a regularizer.

Multilingual training of character-level models is possible not only for languages that have almost identical character vocabularies, such as French and Spanish, but even for distant languages for which a mapping to a common character-level representation can be made, for example through latinizing Russian (Lee et al., 2017) or Chinese (Nikolov et al., 2018) .

More recently, (Cherry et al., 2018) perform an in-depth comparison between different characterand subword-level models.

They show that, given sufficient computational time and model capacity, character-level models can outperform subwordlevel models, due to their greater flexibility in processing and segmenting the input and output sequences.

The transformer (Vaswani et al., 2017) is an attention-driven encoder-decoder model that has achieved state-of-the-art performance on a number of sequence modelling tasks in NLP.

Instead of using recurrence, the transformer uses only feedforward layers based on self-attention.

The standard transformer architecture consists of six stacked encoder layers that process the input using selfattention and six decoder layers that autoregressively generate the output sequence.

Intuitively, attention as an operation is not as meaningful for encoding characters as it is for words.

However, recent work on language modelling (Al-Rfou et al., 2019) has surprisingly shown that attention can be very effective for modelling characters, raising the question how well the transformer would work on character-level bilingual and multilingual translation, and what architectures would be suitable for this task.

These are the questions this paper sets out to investigate.

To facilitate character-level interactions in the transformer, we propose a modification of the standard architecture which we call the convtransformer.

In this architecture, we use the same decoder as the standard transformer, but we adapt each encoder block to include an additional subblock.

The sub-block (Figure 1, b) , inspired from (Lee et al., 2017) , consists of three parallel 1D convolutional layers.

We use separate context window sizes of 3, 5 and 7 for each convolutional layer in order to resemble character interactions of different levels of granularity, similar to subwordor word-level.

Finally, we fuse the representations using an additional convolutional layer, resulting in an output dimensionality that is identical to the input dimensionality.

Therefore, in contrast to (Lee et al., 2017) , who use max pooling to compress the input character sequence into segments of characters, here we leave the resolution unchanged, for both transformer and convtransformer models.

For additional flexibility, we add a residual connection (He et al., 2016) from the input to the output of the convolutional block.

Datasets.

We conduct experiments on two datasets.

First, we use the WMT15 DE→EN dataset, on which we test different model configurations and compare our results to previous work on character-level translation.

We follow the preprocessing in (Lee et al., 2017) and use the newstest-2014 dataset for testing.

Second, we conduct our main experiments using the United Nations Parallel Corporus (UN) (Ziemski et al., 2016) , for two reasons: (i) UN contains a large number of parallel sentences from six languages, allowing us to conduct multilingual experiments; (ii) all sentences in the corpus are from the same domain.

We construct our training corpora by randomly sampling one million sentence pairs from the FR, ES, and ZH parts of the UN dataset, targeting translation to English.

To construct multilin- Table 2 : BLEU scores on the UN dataset, for different input training languages (first column), and evaluated on three different test sets (t-FR, t-ES and t-ZH).

The target language is always English.

#P is the number of training pairs.

The best overall results for each language are in bold.

gual datasets, we combine the respective bilingual datasets (e.g., FR→EN and ES→EN) and shuffle them.

In order to ensure all languages share the same character vocabulary, we latinize the Chinese dataset using the Wubi encoding method, following (Nikolov et al., 2018) .

For testing, we use the original UN test sets provided for each pair.

Tasks.

Our experiments are designed as follows: (i) bilingual scenario, in which we train a model with a single input language; (ii) multilingual scenario, in which we input two or three languages at the same time without providing any language identifiers to the models or increasing their parameters.

We test combining input languages that can be considered as more similar in terms of syntax and vocabulary (e.g. FR and ES) as well as more distant (e.g., ES and ZH).

Model comparison.

In Table 1 , we compare the BLEU performance (Papineni et al., 2002) of different character-level architectures trained on the WMT dataset.

For reference, we include the recurrent character-level model from (Lee et al., 2017) , as well as transformers trained on the subword level, using a vocabulary of 50k byte-pair encoding (BPE) tokens.

All models were trained on four Nvidia GTX 1080X GPUs for 20 epochs.

We find character-level training to be 3 to 5 times slower than subword-level training, due to much longer sequence lengths.

However, the standard transformer trained at the character-level already achieves very strong performance, outperforming the model from (Lee et al., 2017) .

Character-level transformers also perform competitively with equivalent BPE models while requiring up to 60% fewer parameters.

Our convtransformer variant performs on par with the standard transformer on this dataset.

Multilingual experiments.

In Table 2 , we report our BLEU results on the UN dataset using the 6-layer transformer/convtransformer models.

All models were trained for 30 epochs.

Multilingual models are evaluated on translation from all possible input languages to English.

The convtransformer consistently outperforms the transformer on this dataset, with a gap of up to 2.3 BLEU on bilingual translation (ZH→EN) and up to 2.6 BLEU on multilingual translation (FR+ZH→EN).

Training multilingual models on similar input languages (FR + ES→EN) leads to improved performance for both languages, which is consistent with (Lee et al., 2017) .

Training on distant languages can surprisingly still be effective, for example, the models trained on FR+ZH→EN outperform the models trained just on FR→EN, however they perform worse than the bilingual models trained on ZH→EN.

Thus, distant-language training seems only to be helpful when the input language is closer to the target translation language (which is English here).

The convtransformer is about 30% slower to train than the transformer, however, as shown in Figure 2 , the convtransformer reaches compara- ble performance in less than half of the number of epochs, leading to an overall training speedup compared to the transformer.

To gain a better understanding of the multilingual models, we analyze their learned character alignments as inferred from the model attention probabilities.

For each input language (e.g., FR), we compare the alignments learned by each of our multilingual models (e.g., FR + ES → EN model) to the alignments learned by the corresponding bilingual model (e.g., FR → EN).

Our intuition is that the bilingual models have the greatest flexibility to learn high-quality alignments because they are not distracted by other input languages.

Multilingual models, by contrast, might learn lower quality alignments because either (i) the architecture is not robust enough for multilingual training; or (ii) the languages are too dissimilar to allow for effective joint training, prompting the model to learn alternative alignment strategies to accommodate for all languages.

We quantify the alignments using canonincal correlation analysis (CCA) (Morcos et al., 2018) .

First, we sample 500 random sentences from each of our UN testing datasets (FR, ES, or ZH) and then produce alignment matrices by extracting the encoder-decoder attention from the last layer of each model.

We use CCA to project each alignment matrix to a common vector space and infer the correlation.

We conduct the analysis on our transformer and convtransformer models separately.

Our results are in Figure 3 .

For similar source and target languages (e.g., the FR+ES→EN model), we observe strong positive correlation to the bilingual models, indicating that alignments can be simultaneously learned.

When introducing a distant source language (ZH) in the training, we observe a drop in correlation, for FR and ES, and an even bigger drop for ZH.

This is in line with our BLEU results from Section 5.1 suggesting that multilingual training of distant languages is more challenging.

The convtransformer is more robust to the introduction of a distant language than the transformer (p < 0.005 for FR and ES inputs, according to a one-way ANOVA test).

We performed a detailed investigation of the utility of self-attention models for character-level translation, testing the standard transformer architecture, as well as a novel variant augmented by convolution in the encoder to facilitate information propagation across characters.

Our experiments show that self-attention performs very well on characterlevel translation, performing competitively with subword-level models, while requiring fewer parameters.

Training on multiple input languages is also effective and leads to improvements across all languages when the source and target languages are similar.

When the languages are different, we observe a drop in performance, in particular for the distant language.

In future work, we will extend our analysis to include additional source and target languages from different language families, such as more Asian languages.

We will also work towards improving the training efficiency of character-level models, which is one of their main bottlenecks.

A Example model outputs Tables 3, 4 and 5 contain example translations produced by our different bilingual and multilingual models trained on the UN datasets.

In Figures 4,5, 6 and 7 we plot example alignments produced by our different bilingual and multilingual models trained on the UN datasets, always testing on translation from FR to EN.

The alignments are produced by extracting the encoderdecoder attention produced by the last decoder layer of our transformer/convtransformer models.

We observe some patterns: (i) for bilingual translation (Figure 4) , the convtransformer has a sharper weight distribution on the matching characters and words than the transformer; (ii) for multilingual translation of close languages (FR+ES→EN, Figure 5 ), both transformer and convtransformer are able to preserve the word alignments, but the alignments produced by the convtransformer appear to be slightly less noisy; (iii) for multilingual translation of distant languages (FR+ZH→EN, Figure 6 ), the character alignments of the transformer become visually much noisier and concentrated on a few individual chracters and many word alignments dissolve, while the convtransformer character alignments remain more spread out and word alignment is much better preserved.

This is another indication that the convtransformer is more robust for multilingual translation of distant languages. (iv) for multilingual translation with three inputs, where two of the three languages are close (FR+ES+ZH→EN, Figure 7 ), we observe a similar pattern, with the word alignments being better preserved by the convtransformer.

source Pour que ce cadre institutionnel soit efficace, il devra remédier aux lacunes en matière de réglementation et de mise en oeuvre qui caractérisentà ce jour la gouvernance dans le domaine du développement durable.

reference For this institutional framework to be effective, it will need to fill the regulatory and implementation deficit that has thus far characterized governance in the area of sustainable development.

FR→EN transformer

To ensure that this institutional framework is effective, it will need to address regulatory and implementation gaps that characterize governance in sustainable development.

convtransformer

In order to ensure that this institutional framework is effective, it will have to address regulatory and implementation gaps that characterize governance in the area of sustainable development.

To ensure that this institutional framework is effective, it will need to address gaps in regulatory and implementation that characterize governance in the area of sustainable development.

convtransformer

In order to ensure that this institutional framework is effective, it will be necessary to address regulatory and implementation gaps that characterize governance in sustainable development so far.

To ensure that this institutional framework is effective, gaps in regulatory and implementation that have characterized governance in sustainable development to date.

convtransformer For this institutional framework to be effective, it will need to address gaps in regulatory and implementation that characterize governance in the area of sustainable development.

To ensure that this institutional framework is effective, it will need to address regulatory and implementation gaps that are characterized by governance in the area of sustainable development.

convtransformer If this institutional framework is to be effective, it will need to address gaps in regulatory and implementation that are characterized by governance in the area of sustainable development.

source Estamos convencidos de que el futuro de la humanidad en condiciones de seguridad, la coexistencia pacífica, la tolerancia y la reconciliación entre las naciones se verán reforzados por el reconocimiento de los hechos del pasado.

reference

We strongly believe that the secure future of humanity, peaceful coexistence, tolerance and reconciliation between nations will be reinforced by the acknowledgement of the past.

ES→EN transformer

We are convinced that the future of humanity in conditions of security, peaceful coexistence, tolerance and reconciliation among nations will be strengthened by recognition of the facts of the past.

convtransformer

We are convinced that the future of humanity under conditions of safe, peaceful coexistence, tolerance and reconciliation among nations will be reinforced by the recognition of the facts of the past.

We are convinced that the future of mankind under security, peaceful coexistence, tolerance and reconciliation among nations will be strengthened by the recognition of the facts of the past.

convtransformer

We are convinced that the future of humanity in safety, peaceful coexistence, tolerance and reconciliation among nations will be reinforced by the recognition of the facts of the past.

We are convinced that the future of humanity in safety, peaceful coexistence, tolerance and reconciliation among nations will be strengthened by the recognition of the facts of the past.

convtransformer

We are convinced that the future of humanity in safety, peaceful coexistence, tolerance and reconciliation among nations will be strengthened by the recognition of the facts of the past.

We are convinced that the future of mankind in safety, peaceful coexistence, tolerance and reconciliation among nations will be strengthened by the recognition of the facts of the past.

convtransformer

We are convinced that the future of mankind in security, peaceful coexistence, tolerance and reconciliation among nations will be strengthened by the recognition of the facts of the past.

source ZH 利用专家管理农场对于最大限度提高生产率和灌溉水使用效率也是重要的。 source ZH tjh|et fny|pe tp|gj pei|fnrt cf|gf jb|dd bv|ya rj|ym tg|u|yx t iak|ivc|ii wgkq0|et uqt|yx bn j tgj|s r .

reference EN The use of expert farm management is also important to maximize land productivity and efficiency in the use of irrigation water.

ZH→EN transformer

The use of expert management farms is also important for maximizing productivity and irrigation use.

convtransformer

The use of experts to manage farms is also important for maximizing efficiency in productivity and irrigation water use.

The use of expert management farms is also important for maximizing productivity and efficiency in irrigation water use.

convtransformer

The use of expert management farms is also important for maximizing productivity and irrigation water efficiency.

The use of expert farm management is also important for maximizing productivity and irrigation water use efficiency.

convtransformer

The use of expert management farms to maximize efficiency in productivity and irrigation water use is also important.

The use of expert management farms is also important for maximizing productivity and irrigation water use.

convtransformer It is also important that expert management farms be used to maximize efficiency in productivity and irrigation use.

@highlight

We perform an in-depth investigation of the suitability of self-attention models for character-level neural machine translation.