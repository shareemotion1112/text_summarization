The transformer is a state-of-the-art neural translation model that uses attention to iteratively refine lexical representations with information drawn from the surrounding context.

Lexical features are fed into the first layer and propagated through a deep network of hidden layers.

We argue that the need to represent and propagate lexical features in each layer limits the model’s capacity for learning and representing other information relevant to the task.

To alleviate this bottleneck, we introduce gated shortcut connections between the embedding layer and each subsequent layer within the encoder and decoder.

This enables the model to access relevant lexical content dynamically, without expending limited resources on storing it within intermediate states.

We show that the proposed modification yields consistent improvements on standard WMT translation tasks and reduces the amount of lexical information passed along the hidden layers.

We furthermore evaluate different ways to integrate lexical connections into the transformer architecture and present ablation experiments exploring the effect of proposed shortcuts on model behavior.

Since it was first proposed, the transformer model BID28 ) has quickly established itself as a popular choice for neural machine translation, where it has been found to deliver state-ofthe-art results on various translation tasks BID3 .

Its success can be attributed to the model's high parallelizability allowing for significantly faster training compared to recurrent neural networks , superior ability to perform lexical disambiguation, and capacity for capturing long-distance dependencies on par with existing alternatives BID26 .Recently, several studies have investigated the nature of features encoded within individual layers of neural translation models BID1 BID2 .

One central finding reported in this body of work is that, within current architectures, different layers prioritize different information types.

As such, lower layers appear to predominantly perform morphological and syntactic processing, whereas semantic features reach their highest concentration towards the top of the layer stack.

One necessary consequence of this distributed learning is that different types of information encoded within input representations received by the translation model have to be transported to the layers specialized in exploiting them.

Within the transformer encoder and decoder alike, information exchange proceeds in a strictly sequential manner, whereby each layer attends over the output of the immediately preceding layer, complemented by a shallow residual connection.

For input features to be successfully propagated to the uppermost layers, the translation model must therefore store them in its intermediate representations until they can be processed.

By retaining lexical content, the model is unable to leverage its full representational capacity for learning new information from other sources, such as the surrounding sentence context.

We refer to this limitation as the representation bottleneck.

To alleviate this bottleneck, we propose extending the standard transformer architecture with lexical shortcuts which connect the embedding layer with each subsequent self-attention sub-layer in both encoder and decoder.

The shortcuts are defined as gated skip connections, allowing the model to access relevant lexical information at any point, instead of propagating it upwards from the embedding layer along the hidden states.

We evaluate the resulting model's performance on multiple language pairs and varying corpus sizes, showing a consistent improvement in translation quality over the unmodified transformer baseline.

Moreover, we examine the distribution of lexical information across the hidden layers of the transformer model in its standard configuration and with added shortcut connections.

The presented experiments provide quantitative evidence for the presence of a representation bottleneck in the standard transformer and its reduction following the integration of lexical shortcuts.

While our experimental efforts are centered around the transformer, the proposed components are compatible with other multi-layer NMT architectures.

The contributions of our work are therefore as follows:1.

We propose the use of lexical shortcuts as a simple strategy for alleviating the representation bottleneck in neural machine translation models.2.

We demonstrate significant improvements in translation quality across multiple language pairs as a result of equipping the transformer with lexical shortcut connections.3.

We report a positive impact of our modification on the model's ability to perform word sense disambiguation.4.

We conduct a series of ablation studies, showing that shortcuts are best applied to the self-attention mechanism in both encoder and decoder.2 Proposed Method 2.1 Background:

The transformerAs defined in BID28 , the transformer is comprised of two sub-networks, the encoder and the decoder.

The encoder coverts the received source language sentence into a sequence of continuous representations containing translation-relevant features.

The decoder, on the other hand, generates the target language sequence, whereby each translation step is conditioned on the encoder's output as well as the translation prefix produced up to that point.

Both encoder and decoder are composed of a series of identical layers.

Each encoder layer contains two sub-layers: A self-attention mechanism and a position-wise fully connected feed-forward network.

Within the decoder, each layer is extended with a third sub-layer responsible for attending over the encoder's output.

In each case, the attention mechanism is implemented as multihead, scaled dot-product attention, which allows the model to simultaneously consider different context sub-spaces.

Additionally, residual connections between neighboring layers are employed to aid with signal propagation.

In order for the dot-product attention mechanism to be effective, its inputs first have to be projected into a common representation sub-space.

This is accomplished by multiplying the input arrays H S and H T by one of the three weight matrices K, V , and Q, as shown in Eqn.

1-3, producing attention keys, values, and queries, respectively.

In case of multi-head attention, each head is assigned its own set of keys, values, and queries with the associated learned projection weights.

DISPLAYFORM0 In case of encoder-to-decoder attention, H T corresponds to the final encoder states, whereas H S is the context vector generated by the preceding self-attention sub-layer.

For self-attention, on the other hand, all three operations are given the output of the preceding layer as their input.

Eqn.

4 defines attention as a function over the projected representations.

DISPLAYFORM1 To prevent the magnitude of the pre-softmax dot-product from becoming too large, it is divided by the square root of the total key dimensionality d k .

Finally, the translated sequence is obtained by feeding the output of the decoder through a softmax layer and sampling from the produced distribution over target language tokens.

Given that the attention mechanism represents the primary means of establishing parameterized connections between the different layers within the transformer, it is well suited for the re-introduction of lexical content.

We achieve this by adding gated connections between the embedding layer and each subsequent self-attention sub-layer within the encoder and the decoder, as shown in FIG0 .To ensure that lexical features are compatible with the learned hidden representations, the retrieved embeddings are projected into the appropriate latent space, by multiplying them with the layer-specific weight matrices W K SC l and W V SC l .

We account for the potentially variable importance of lexical features by equipping each added connection with a binary gate inspired by the Gated Recurrent Unit BID5 .

Functionally, our lexical shortcuts are therefore reminiscent of highway connections proposed in BID25 .

DISPLAYFORM0 DISPLAYFORM1 After situating the outputs of the immediately preceding layer H l−1 and the embeddings E within a shared representation space (Eqn.

5-8), the relevance of lexical information for the current attention step is estimated by comparing lexical and latent features, followed by the addition of a bias term b (Eqn.

9-10).

The respective attention key arrays are denoted as K SC l and K l , while V SC l and V l represent the corresponding value arrays.

The result is then fed through a sigmoid function to obtain the lexical relevance weight r, used to combine both sets of features by calculating their weighted sum (Eqn.

11-12), where denotes element-wise multiplication.

Next, the obtained key and value arrays K l and V l are passed to the multi-head attention function instead of the original K l and V l .In an alternative formulation of the model, we concatenate E and H l−1 before the initial linear projection, splitting the result in two halves along the feature dimension and leaving the rest of the shortcut definition unchanged.

This reduces Eqn.

5-8 to Eqn.

13-14, and enables the model to select relevant information by directly inter-relating lexical and hidden features.

As such, both K SC l and K l encode a mixture of embedding and hidden features, as do the corresponding value arrays.

We refer to this step as 'feature-fusion' 1 .

FIG1 provides a high-level illustration of how lexical in-formation is integrated into the attention inputs.

DISPLAYFORM2 Other than the immediate accessibility of lexical information, one potential benefit afforded by the introduced shortcuts is the improved gradient flow during back-propagation.

As noted in BID11 , the addition of skip connections between individual layers of a deep neural network results in an implicit 'deep supervision' effect BID14 , which aids the training process.

In case of our modified transformer, this corresponds to the embedding layer receiving its learning signal from the model's overall optimization objective as well as from each layer it is connected to, making the model easier to train.

To evaluate the efficacy of the proposed approach, we re-implement the transformer model and extend it by applying lexical shortcuts to each selfattention layer in the encoder and decoder.

Our code is publicly available to aid the reproduction of the reported results.

2 Details regarding our model configurations, data pre-processing, and training setup are given in the appendix (A.1-A.2).

We investigate the potential benefits of lexical shortcuts on 5 WMT translation tasks: German → English (DE→EN), English → German (EN→DE), English → Russian (EN→RU), English → Czech (EN→CS), and English → Finnish (EN→FI).

Our choice is motivated by the differences in training data size as well as by the typological diversity of the target languages.

To make our findings comparable to related work, we train EN↔DE models on the WMT14 news translation data which encompasses ∼4.5M sentence pairs.

EN→RU models are trained on the WMT17 version of the news translation task, consisting of ∼24.8M sentence pairs.

For EN→CS and EN→FI, we use the respective WMT18 parallel training corpora, with the former containing ∼50.4M and the latter ∼3.2M sentence pairs.

Throughout training, model performance is validated on newstest2013 for EN↔DE, newstest2016 for EN→RU, and on newstest2017 for 2 Link withheld to preserve anonymity.

EN→CS and EN→FI.

Final model performance is reported on multiple tests sets from the news domain for each direction.

The results of our translation experiments are summarized in TAB1 .

To ensure their comparability, we evaluate translation quality using sacre-BLEU BID16 .

As such, our baseline performance diverges from that reported in BID28 .

We address this by evaluating our EN→DE models using the scoring script from the tensor2tensor toolkit 3 BID27 ) on the tokenized model output, and list the corresponding BLEU scores in the first column of Table 1 .Our evaluation shows that the introduction of lexical shortcuts consistently improves translation quality of the transformer model across different test-sets and language pairs, outperforming transformer-BASE by 0.5 BLEU on average.

With feature-fusion, we see even stronger improvements, yielding total performance gains over transformer-BASE of up to 1.4 BLEU for EN→DE (averaging to 1.0), and 0.8 BLEU on average for the other 4 translation directions.

We furthermore observe that the relative improvements from the addition of lexical shortcuts are substantially smaller for transformer-BIG compared to transformer-BASE.

One potential explanation for this drop in effectiveness is the increased size of the latent representations the wider model is able to learn.

It is possible that the larger hidden state size of transformer-BIG widens the representation bottleneck, thus reducing the benefits of dynamic lexical access.

It is also worth noting that transformer-BASE, when equipped with lexical connections, performs comparably to the standard transformer-BIG, despite containing ∼2/3 of its parameters and being only marginally slower to train than our transformer-BASE implementation.

An overview of model sizes and training speed is provided in the supplementary material (A.1).Concerning the examined language pairs, the average increase in BLEU is lowest for DE→EN (0.6 BLEU) and highest for EN→RU (1.1 BLEU).

While we do not have conclusive evidence for why this is the case, one possible explanation could be the difference in language topology.

Of the tar- get languages we consider, English is the morphologically weakest one, where individual words do not carry much inflectional information.

As such, features encoded by (sub-)words in isolation may be less useful for the translation task than the larger sentence context aggregated within the hidden states.

We expect this to result in a less pronounced representation bottleneck with fewer lexical features propagated across the network, diminishing the contribution of added shortcuts.

To further investigate the role of lexical connections within the transformer, we perform a thorough analysis of the models' internal representations and learning behaviour.

The following analysis is based on the model incorporating lexical shortcuts as well as feature-fusion, due to its superior performance.

The proposed approach is motivated by the hypothesis that the transformer retains lexical features within its individual layers, which limits its capacity for learning and representing other types of relevant information.

Direct connections to the embedding layer alleviate this by providing the model with access to lexical features at each processing step, reducing the need to propagate them along hidden states.

To investigate whether this is indeed the case, we perform a probing study, where we estimate the amount of lexical content present within each hidden state of the encoder and decoder.

We examine the internal representations learned by our models by modifying the probing technique introduced in BID1 .

Specifically, we train a separate lexical classifier for each layer of a frozen translation model.

Each classifier receives hidden states extracted from the respective transformer layer 4 and is tasked with reconstructing the sub-word corresponding to the position of each hidden state.

Encoder-specific classifiers learn to reconstruct sub-words in the source sentence, whereas classifiers trained on decoder states are trained to reconstruct target sub-words.

The accuracy of each classifier on a withheld test set is assumed to be indicative of the lexical content encoded by the associated transformer layer.

We expect classification error to be low if the evaluated representations predominantly store information propagated upwards from the embeddings at the same position and to increase proportional to the amount of information drawn from Based on the classifier results, it appears that immediate access to lexical information does indeed alleviate the representation bottleneck by reducing the extent to which (sub-)word-level content is retained across encoder and decoder layers.

The effect is consistent across multiple language pairs, supporting its generality.

Additionally, to examine whether lexical retention depends on the specific properties of the input tokens, we track classification accuracy conditioned on partof-speech tags and sub-word frequencies.

While we do not discover a pronounced effect of either category on classification accuracy, we present a summary of our findings as part of the supplementary material for future reference (A.3).Another observation arising from this analysis is that the decoder retains fewer lexical features beyond its initial layers than the encoder.

This may be due to the decoder having to represent information it receives from the encoder in addition to target-side content, necessitating a lower rate of lexical feature retention.

Even so, by adding shortcut connections we can increase the dissimilarity between the embedding layer and the subsequent layers of the decoder, indicating a noticeable reduction in the retention and propagation of lexical features along the decoder's hidden states.

A similar trend can be observed when evaluating layer similarity directly, which we accomplish by calculating the cosine similarity between the embeddings and the hidden states of each trans- former layer.

Echoing our findings so far, the addition of lexical shortcuts reduces layer similarity relative to the baseline transformer for both encoder and decoder.

The corresponding visualizations are also provided in the appendix (A.3).

If the absolute amount of lexical information propagated along layers is independent of model size, shortcuts may benefit smaller models more.

We put this hypothesis to test by scaling down the standard transformer, halving the size of its embeddings, hidden states, and feed-forward sublayers.

TAB3 shows that, on average, improvements to translation quality are comparable for the small and standard transformer (1.0 BLEU for both).

A possible explanation is that halving the embedding size effectively halves the amount of lexical features the model can access and propagate upwards from the embedding layer, leaving the overall width of the bottleneck unchanged.

Nonetheless, the exact interaction between the scale of a model and the information encoded in its hidden states remains to be fully explored.

Interestingly, basic lexical shortcuts are more effective in the small model, whereas feature-fusion offers more benefit to the standard model, implying that their relative contributions may be complementary and dependent on model size.

Overall, the presented experiments support the existence of a representation bottleneck in NMT models as one explanation for the efficacy of the proposed lexical shortcut connections.

Until now, we focused on applying shortcuts to self-attention as a natural re-entry point for lexical content.

However, previous studies suggest that providing the decoder with direct access to source sentences can improve translation adequacy, by conditioning each translation step on the relevant source words BID15 .To investigate whether the proposed method can confer a similar benefit to the transformer, we apply shortcut connections to decoder-to-encoder attention, replacing or adding to shortcuts feeding into self-attention.

Formally, this equates to fixing E to E enc in Eqn.

5-6.

As can be seen from Table 4, while integrating shortcut connections into the decoder-to-encoder attention improves upon the base transformer, the improvement is smaller than when we modify self-attention.

Furthermore, combining both methods yields worse translation quality than either one does in isolation, indicating that the observed improvements are not complementary.

We therefore conclude that lexical shortcuts are most beneficial to self-attention.

A related question is whether the encoder and decoder benefit from the addition of lexical shortcuts to self-attention equally.

We explore this by disabling shortcuts in either sub-network and comparing the so obtained translation models to one with intact connections.

FIG4 illustrates that best translation performance is obtained by en- abling shortcuts in both encoder and decoder.

This also improves training stability, as compared to the decoder-only ablated model.

The latter may be explained by our use of tied embeddings which receive a stronger training signal from shortcut connections due to 'deep supervision', as this may bias learned embeddings against the sub-network lacking improved lexical connectivity.

While adding shortcuts improves translation quality, it is not obvious whether this is predominantly due to improved accessibility of lexical content, rather than increased connectivity between network layers, as suggested in BID7 .

To isolate the importance of lexical information, we equip the transformer with non-lexical shortcuts connecting each layer n to layer n − 2, e.g. layer 6 to layer 4.

5 As a result, the number of added connections and parameters is kept identical to lexical shortcuts, while lexical accessibility is reduced.

Test-BLEU reported in TAB5 suggests that while non-lexical shortcuts improve over the baseline model, they perform noticeably worse than lexical connections.

Therefore, the increase in translation quality associated with lexical shortcuts is not solely attributable to a better signal flow or the increased number of trainable parameters.

Beyond the effects of lexical shortcuts on the transformer's learning dynamics, we are interested in how widening the representation bottleneck affects the properties of the produced translations.

One challenging problem in translation which in-tuitively should benefit from the model's increased capacity for learning information drawn from sentence context is word-sense disambiguation.

We examine whether the addition of lexical shortcuts aids disambiguation by evaluating our trained DE→EN models on the ContraWSD corpus BID19 .

The contrastive dataset is constructed by paring source sentences with multiple translations, varying the translated sense of selected source nouns between translation candidates.

A competent model is expected to assign a higher probability to the translation hypothesis containing the appropriate word-sense.

While the standard transformer provides a very strong baseline for the disambiguation task, we nonetheless see improvements as a result of adding direct connections to the embedding layer.

While our baseline model reaches an accuracy rating of 88.8%, equipping it with lexical shortcuts further improves the score to 89.5%.

Within recent literature, proposals have been made for how the standard transformer architecture may be extended, including adaptive model depth BID6 , layer-wise transparent attention for the effective training of deeper models , and a more effective exploitation of features learned by the deep network BID7 .

Our investigation bears strongest resemblance to the latter, concurrent work by introducing additional connectivity to the model.

However, rather than establishing new connections between layers indiscriminately, we explicitly seek to facilitate the accessibility of lexical information throughout the model, as this reduces the need to represent and propagate lexical features along hidden states, thereby increasing the model's capacity for learning novel information.

As a result, our proposed shortcut connections are sparser, simpler, and more efficient.

Another line of research from which we draw inspiration concerns itself with the analysis of the internal dynamics and learned representations within deep neural networks BID12 BID23 BID18 .

Here, BID1 and BID2 serve as our primary points of reference by providing a thorough and principled investigation of the extent to which neural translation models are capable of learning linguistic properties from raw text.

While the role of lexical features in NMT has not received widespread attention, BID15 note that improving accessibility of source words by the decoder benefits translation quality in low-resource settings.

Our view of the transformer as a model learning to refine input representations through the repeated application of attention is consistent with the iterative estimation paradigm introduced in BID8 .

According to this interpretation, given a stack of connected layers sharing the same dimensionality and interlinked through highway or residual connections, the initial layer generates a rough version of the stack's final output, which is iteratively refined by successive layers, e.g. by enriching localized features with information drawn from the surrounding context.

The results of our probing studies support this analysis of the transformer, further suggesting that different layers not only refine input features but also learn entirely new information given sufficient capacity, as evidenced by the decrease in similarity between embeddings and hidden states with increasing model depth.

In this paper, we have proposed a simple yet effective method for widening the representation bottleneck in the transformer by introducing lexical shortcuts.

Our modified models achieve up to 1.4 BLEU (0.9 BLEU on average) improvement on 5 standard WMT datasets, at a small cost in computing time and model size.

Our analysis suggests that lexical connections are useful to both encoder and decoder, and remain effective when included in smaller models.

Moreover, the addition of shortcuts noticeably reduces the similarity of hidden states to the initial embeddings, indicating that dynamic lexical access aids the network in learning novel, diverse information.

We also performed ablation studies comparing different shortcut variants and demonstrated that one effect of lexical shortcuts is an improved WSD capability.

The presented findings offer new insights into the nature of information encoded by the transformer layers, supporting the iterative refinement view of feature learning.

In future work, we intend to explore other ways to better our understanding of the refinement process and to help translation models learn more diverse and meaningful internal representations.

The majority of our experiments is conducted using the transformer-BASE configuration, with the number of encoder and decoder layers set to 6 each, embedding and attention dimensionality to 512, number of attention heads to 8, and feedforward sub-layer dimensionality to 2048.

We tie the encoder embedding table with the decoder embedding table and the pre-softmax projection matrix to speed up training, following BID17 .

All trained models are optimized using Adam (Kingma and Ba, 2014) adhering to the learning rate schedule described in BID28 .

We set the number of warm-up steps to 4000 for the baseline model, increasing it to 6000 and 8000 when adding lexical shortcuts and feature-fusion, respectively, so as to accommodate the increase in parameter size.

We also evaluate the effect of lexical shortcuts on the larger transformer-BIG model, limiting this set of experiments to EN→DE due to computational constraints.

In this more expensive configuration, the baseline model employs 16 attention heads, with attention, embedding, and feedforward dimensionality doubled to 1024, 1024, and 4096.

Warm-up period for all big models is set to 16,000 steps.

For our probing experiments, the classifiers used are simple feed-forward networks with a single hidden layer consisting of 512 units, dropout BID24 with p = 0.5, and a ReLU non-linearity.

All models are trained concurrently on four Nvidia P100 Tesla GPUs using synchronous data parallelization.

Delayed optimization BID20 ) is employed to simulate batch sizes of 25,000 tokens, to be consistent with BID28 .

Each transformer-BASE model is trained for a total of 150,000 updates, while our transformer-BIG experiments are stopped after 300,000 updates.

Validation is performed every 4000 steps, as is check-pointing.

Training base models takes ∼43 hours, while the addition of shortcut connections increases training time up to ∼46 hours (∼50 hours with feature-fusion).

Table 5 details the differences in parameter size and training speed for the different transformer configurations.

Parameters are given in thousands, while speed is averaged over the entire training duration.

Validation-BLEU is calculated using multibleu-detok.pl 6 on a reference which we pre-and post-process following the same steps as for the models' inputs and outputs.

All reported test-BLEU scores were obtained by averaging the final 5 checkpoints for transformer-BASE and final 16 for transformer-BIG.

We tokenize, clean, and truecase each training corpus using scripts from the Moses toolkit 7 , and apply byte-pair encoding BID22 to counteract the open vocabulary issue.

Cleaning is skipped for validation and test sets.

For EN↔DE and EN→RU we limit the number of BPE merge operations to 32,000 and set the vocabulary threshold to 50.

For EN→CS and EN→FI, the number of merge operations is set to 89,500 with a vocabulary threshold of 50, following 8 .

In each case, the BPE vocabulary is learned jointly over the source and target language, which necessitated an additional transliteration step for the pre-processing of Russian data 9 .

Cosine similarity scores between the embedding layer and each successive layer in transformer-BASE and its variant equipped with lexical shortcuts are summarized in FIG6 .For our fine-grained probing studies, we evaluated classification accuracy conditioned of partof-speech tags and sub-word frequencies.

For the former, we first parse our test-sets with TreeTagger BID21 , projecting tags onto the constituent sub-words of each annotated word.

For frequency-based evaluation, we divide sub-words into ten equally-sized frequency bins, with bin 1 containing the least frequent sub-words and bin 10 containing the most frequent ones.

We do not observe any immediately obvious, significant effects of either POS or frequency on the retention of lexical features.

While classification accuracy is notably low for infrequent sub-words, this can be attributed to the limited occurrence of the corresponding transformer states in the classifier's training data.

Evaluation for EN→DE models is done on newstest2014, while newstest2017 is used for EN→RU models.

We also investigated the activation patterns of the lexical shortcut gates.

However, despite their essential status for the successful training of transformer variants equipped with lexical connections, we were unable to discern any distinct patterns in the activations of the individual gates, which tend to prioritize lexical and hidden features to an equal degree regardless of training progress or (sub-)word characteristics.

@highlight

Equipping the transformer model with shortcuts to the embedding layer frees up model capacity for learning novel information.