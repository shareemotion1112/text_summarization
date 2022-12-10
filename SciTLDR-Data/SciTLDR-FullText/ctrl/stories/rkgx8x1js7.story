Multimodal sentiment analysis is a core research area that studies speaker sentiment expressed from the language, visual, and acoustic modalities.

The central challenge in multimodal learning involves inferring joint representations that can process and relate information from these modalities.

However, existing work learns joint representations using multiple modalities as input and may be sensitive to noisy or missing modalities at test time.

With the recent success of sequence to sequence models in machine translation, there is an opportunity to explore new ways of learning joint representations that may not require all input modalities at test time.

In this paper, we propose a method to learn robust joint representations by translating between modalities.

Our method is based on the key insight that translation from a source to a target modality provides a method of learning joint representations using only the source modality as input.

We augment modality translations with a cycle consistency loss to ensure that our joint representations retain maximal information from all modalities.

Once our translation model is trained with paired multimodal data, we only need data from the source modality at test-time for prediction.

This ensures that our model remains robust from perturbations or missing target modalities.

We train our model with a coupled translation-prediction objective and it achieves new state-of-the-art results on multimodal sentiment analysis datasets: CMU-MOSI, ICT-MMMO, and YouTube.

Additional experiments show that our model learns increasingly discriminative joint representations with more input modalities while maintaining robustness to perturbations of all other modalities.

Sentiment analysis, which involves identifying a speaker's opinion, is a core research problem in machine learning and natural language processing.

However, language-based sentiment analysis through words, phrases, and their compositionality was found to be insufficient for inferring affective content from spoken opinions BID33 , which contain rich nonverbal behaviors in addition to verbal text.

As a result, there has been a recent push towards using machine learning methods to learn joint representations from additional behavioral cues present in the visual and acoustic modalities.

This research field has become known as multimodal sentiment analysis and extends the conventional textbased definition of sentiment analysis to a multimodal setup.

For example, BID21 explore the additional acoustic modality while BID61 use the language, visual, and acoustic modalities present in monologue videos to predict sentiment.

The abundance of multimodal data has led to the creation of multimodal datasets, such as CMU-MOSI BID66 and ICT-MMMO BID61 , as well as deep multimodal models that are highly effective at learning discriminative joint multimodal representations BID29 BID57 BID7 .

Existing work learns joint representations using multiple modalities as input with neural networks BID28 , graphical models BID33 or geometric classifiers BID66 .

However, this results in joint representations that are sensitive to noisy or missing modalities at test time.

To address this problem, we draw inspirations from the recent success of sequence to sequence models for unsupervised representation learning BID55 .

We propose the Multimodal Cyclic Translation Network model (MCTN) to learn robust joint multimodal representations by translating between modalities.

FIG0 illustrates these translations between the language, visual and acoustic modalities.

Our method is based on the key insight that translation from a source modality S to a target modality T results in an intermediate representation that captures joint information between modalities S and T .

MCTN extends this insight using a cyclic translation loss involving both forward translations from source to target, and backward translations from the predicted target back to the source modality.

Together, we call these multimodal cyclic translations to ensure that the learned joint representations capture maximal information from both modalities.

We also propose a hierarchical MCTN to learn joint representations between a source modality and multiple target modalities.

MCTN is trainable end-to-end with a coupled translation-prediction loss which consists of (1) the cyclic translation loss, and (2) a prediction loss to ensure that the learned joint representations are task-specific.

Another advantage of MCTN is that once trained with paired multimodal data (S, T ), we only need data from the source modality S at test time to infer the joint representation and sentiment prediction.

As a result, MCTN is completely robust to test-time perturbations on target modality T and missing modalities.

Even though translation and generation of videos, audios, and text are difficult BID27 , our experiments show that the learned joint representations can help for discriminative tasks: MCTN achieves new state-of-the-art results on multimodal sentiment analysis using the CMU-MOSI, ICT-MMMO, and YouTube public datasets.

Additional experiments show that MCTN learns increasingly discriminative joint representations with more input modalities while maintaining robustness to all target modalities.

Early work on sentiment analysis focused primarily on written text BID39 BID50 .

Recently, multimodal sentiment analysis has gained more research interest BID4 since learning joint representation of multiple modalities is a challenging task.

Earlier work simply concatenated the input features BID36 BID25 .

Recently, several neural models have also been proposed to learn joint representations BID7 BID64 BID8 .

For example, BID28 presented a multistage approach to learn hierarchical multimodal representations.

The Tensor Fusion Network BID63 and the Low-rank Multimodal Fusion model BID30 presented methods based on Cartesian-products to model unimodal, bimodal and trimodal interactions.

In addition to purely supervised approaches, generative methods based on Generative Adversarial Networks (GANs) BID16 have been used to learn joint distributions between two or more modalities BID13 BID26 .

Another method involves using conditional generative models BID32 BID22 BID38 to translate one modality to another.

Generative-discriminative objectives have been used to learn either joint BID43 BID23 or factorized BID57 representations.

Our work takes into account the sequential dependency of modality translations and also explores the effect of a cyclic translation loss on modality translations.

Finally, there has been some progress on accounting for noisy or missing modalities at test time.

BID54 proposed using Deep Boltzmann Machines to model the joint distribution over multimodal data.

Sampling from the conditional distributions allow for inference of missing modalities.

BID51 trained Restricted Boltzmann Machines to minimize the variation of information between modalityspecific latent variables.

Models based on autoencoders BID56 , adversarial learning BID6 , or multiple kernel learning BID31 have also been proposed for these tasks.

It was also found that training with missing or noisy modalities can improve the robustness of joint representations BID36 .

These methods approximately infer the missing modalities before prediction, leading to possible error compounding.

On the other hand, MCTN remains fully robust to other modalities during testing.

In this section, we describe our approach for learning joint multimodal representations through modality translations.

Notation: A multimodal dataset consists of data X = (X l , X v , X a ) from the language, visual, and acoustic modalities respectively.

It is indexed by n segments X = (X 1 , X 2 , ..., X n ) where DISPLAYFORM0 The labels for these n segments are denoted as y = (y 1 , y 2 , ..., y n ), y i ∈ R. Many datasets are easily synchronized by aligning the input based on the boundaries of each word and zero-padding each segment to obtain time-series data of the same length BID28 .

The ith segment is given by DISPLAYFORM1 ) where w i ( ) stands for the th word and L is the length of each segment.

To accompany the language features, we also have a sequence of visual features DISPLAYFORM2

We define learning a joint representation between two modalities X S and X T as learning a parametrized function f θ that returns an embedding DISPLAYFORM0 From there, another function g w is learned that predicts the label given this joint representation:ŷ = g w (E ST ).Most work follows this framework during both training and testing BID28 BID30 BID57 BID64 .

During training, the parameters θ and w are learned by empirical risk minimization over paired multimodal data and labels in the training set (X S tr , X T tr , y tr ): DISPLAYFORM1 for a suitable choice of loss function y over the labels (tr denotes training set).

During testing, paired multimodal data in the test set (X S te , X T te ) are used to infer the label (te denotes test set): S and the target modality X T .

The joint representation E S⇆T is obtained via a cyclic translation between X S and X T .

Next, the joint representation E S⇆T is used for sentiment prediction.

The model is trained end-to-end with a coupled translation-prediction objective.

At test time, only the source modality X S is required.

DISPLAYFORM2

Multimodal Cyclic Translation Network (MCTN) is a neural model that learns robust joint representations by modality translations.

FIG2 shows a detailed description of MCTN for two modalities.

Our method is based on the key insight that translation from a source modality X S to a target modality X T results in a representation that captures joint information between modalities X S and X T , but using only the source modality X S as input.

To ensure that our model learns joint representations that retain maximal information from all modalities, we use a cycle consistency loss BID67 during modality translation.

This method can also be seen as a variant of back-translation which has been recently applied to style transfer BID46 BID67 and unsupervised machine translation BID24 .

We use back-translation in a multimodal setup where we encourage our translation model to learn informative joint representations but with only the source modality as input.

The cycle consistency loss for modality translation starts by decomposing function f θ into two parts: an encoder f θe and a decoder f θ d .

The encoder takes in X S as input and returns a joint embedding E S→T : DISPLAYFORM0 which the decoder then transforms into target modality X T : DISPLAYFORM1 following which the decoded modality T is translated back into modality S: DISPLAYFORM2 The joint representation is learned by using a Sequence to Sequence (Seq2Seq) model with attention BID3 that translates source modality X S to a target modality X T .

While Seq2Seq models have been predominantly used for machine translation, we extend its usage to the realm of multimodal machine learning.

The Seq2Seq model consists of an encoder network and a decoder network, each parametrized as Recurrent Neural Networks (RNNs).

The encoder maps the source modality X S into an embedded representation E S→T .

Using a recurrent network, the hidden state output of each time step is based on the previous hidden state along with the input sequence DISPLAYFORM3 (7) The encoder's output is the concatenation of all hidden states of the encoding RNN, DISPLAYFORM4 where L is the length of the source modality X S .The decoder maps the representation E S→T into the target modality X T .

This is performed by decoding each token X T t at a time based on E S→T and all previous decoded tokens, which is formulated as DISPLAYFORM5 MCTN accepts variable-length inputs of X S and X T , and is trained to maximize the translational condition probability p(X T X S ).

The best translation sequence is then given bŷ DISPLAYFORM6 While there are other search algorithms such as random sampling and greedy search that can be used for decoding each token BID35 , we use the traditional beam search approach BID55 .To obtain the joint representation for prediction, we found that simply using one of the translated representations was sufficient for good performance (⇆ denotes multimodal cyclic translations): E S⇆T = E S→T .

E S⇆T is used for prediction via a recurrent neural network,ŷ = g w (E S⇆T ).

Training is performed with paired multimodal data and labels in the training set (X S tr , X T tr , y tr ).

We evaluate the forward translation loss DISPLAYFORM0 and the cycle consistency loss DISPLAYFORM1 for suitable choices of loss functions X T and X S .

We use the Mean Squared Error (MSE) between the ground-truth and translated modalities.

Finally, the prediction loss L p is DISPLAYFORM2 (12) for a loss function y over the labels.

Equations FORMULA13 , BID10 , and (12) are evaluated using the training set and MCTN can be trained end-toend with a coupled translation-prediction DISPLAYFORM3 where L p is the prediction loss, L c is the cyclic translation loss, and λ t , λ t are weighting hyperparameters.

MCTN parameters are learned by minimizing this objective function DISPLAYFORM4 Parallel multimodal data is not required at test time.

Inference is performed using only the source modality X S te : DISPLAYFORM5 This is possible because the encoder f θ * e has been trained to translate the source modality X S into a joint representation E S⇆T that captures information from both source and target modalities.

Intuitively, the translation model learns to predict target modalities through an informative joint representation.

E S⇆T1 is obtained via a cyclic translation between X S and X T1 , then further translated into X T2 .

Next, the joint representation of all three modalities, E (S⇆T1)→T2 , is used for sentiment prediction.

The model is trained end-to-end with a coupled translation-prediction objective.

At test time, only the source modality X S is required.

We extend the MCTN hierarchically to learn joint representations from more than two modalities.

FIG3 shows the case for three modalities.

The hierarchical MCTN starts with a source modality X S and two target modalities X T1 and X T2 .

To learn joint representations, two levels of translations are performed.

The first level learns a joint representation from X S and X T1 using multimodal cyclic translations as defined previously.

At the second level, a joint representation is learned hierarchically by translating the first representation E S→T1 into X T2 .

For more than three modalities, the modality translation process can be repeated hierarchically.

TwoSeq2Seq models are used in the hierarchical MCTN for three modalities.

Denote these as encoder-decoder pairs (f 1 θe , f 1 θ d ) and (f 2 θe , f 2 θ d

A multimodal cyclic translation is first performed between source modality X S and the first target modality X T1 .

The consists of the forward translation: DISPLAYFORM0 following which the decoded modality X T1 is translated back into modality X S : DISPLAYFORM1 A second hierarchical Seq2Seq model is applied on the time-distributed outputs of the encoder f 1 θe : DISPLAYFORM2 DISPLAYFORM3 The joint representation between modalities X S , X T1 and X T2 is now E (S⇆T1)→T2 and is used for sentiment prediction via a recurrent neural network.

Training the hierarchical MCTN involves computing a cycle consistent loss for modality T 1 , given by L t1 and L c1 .

We do not use a cyclic translation loss when translating from E S⇆T1 to X T2 since the ground truth E S⇆T1 is unknown, and so only the translation loss L t2 is computed.

The final objective for hierarchical MCTN is given by DISPLAYFORM4 We emphasize that for MCTN with three modalities, only a single source modality X S is required at test time.

Therefore, MCTN has a significant advantage over existing models since it is robust to noise or missing target modalities.

In this section, we describe our experimental methodology to evaluate the joint representations learned by MCTN.

Datasets: We use the CMU-MOSI dataset which contains 2199 video segments each with a sentiment label in the range from −3 to +3.

−3 indicates strongly negative sentiment, +3 indicates strongly positive sentiment, and 0 indicates neutral sentiment.

CMU-MOSI is subject to much research BID57 BID64 BID7 and the current state of the art is achieved by BID28 with a binary accuracy of 78.4%.

We additionally perform experiments on the ICT-MMMO BID61 and YouTube BID33 datasets.

These datasets consist of online review videos annotated for sentiment.

Features: Following previous work BID28 , GloVe word embeddings BID41 , Facet BID19 and CO-VAREP BID11 features are extracted for the language, visual and acoustic modalities respectively.

BID1 Forced alignment is performed using P2FA BID62 to obtain word utterance times and we align the visual and acoustic features by computing their average over each word utterance interval.

BID18 .

For details on all baselines, please refer to the supplementary.

This section discusses several research questions and presents our experimental results.

Step 1Step FORMULA4 TAB2 , we observe that adding more modalities improves performance, indicating that the joint representations learned are leveraging the information from more input modalities.

This also implies that cyclic translations are a viable method to learn joint representations from multiple modalities since little information is lost from adding more modality translations.

Another observation is that using language as the source modality always leads to the best performance, which is intuitive since the language modality contains the most information towards sentiment BID63 .In addition, we visually inspect the joint representations learnt from MCTN as we add more modalities during training.

The joint representations for each video segment in CMU-MOSI are extracted from the best performing model for each number of modalities and then projected into two dimensions via the t-SNE algorithm BID58 .

Each point is colored red or blue depending on whether the video segment is annotated for positive or negative sentiment.

From FIG5 , we observe that the joint representations become increasingly separable as the more modalities are added when the MCTN is trained.

This is consistent with increasing discriminative performance as seen in TAB2 .Ablation Studies: We devise the following ablation models to test each design decision in MCTN: the use of cyclic translations, shared Seq2Seq models, modality ordering, and hierarchical structure.

For bimodal MCTN, we design the following ablation models shown in the left half of FIG6 : (a) is our proposed MCTN between X S and X T , (b) is the MCTN based on translation from X S to X T without cyclic translations, (c) does not use cyclic translations but rather performs two independent translations between X S and X T , (d) is the pair of MCTN models with different inputs (of the same modality pair) and then using the concatenation of the joint representations E S→T and E T →S as the final embeddings.

For trimodal MCTN, we design the following ablation models shown in the right half of FIG6 : (e) is the proposed hierarchical MCTN between X S , X T1 and X T2 , (f) is the MCTN based on translation from X S to X T1 without cyclic translations, (g) is extended from (d) which does not use cyclic translations but rather performs two independent translations between X S and X T1 , and finally, (h) does not perform a first level of cyclic translation but directly translates the concatenated modality pair [X S , X T1 ] into X T2 .

The bimodal and trimodal results are shown in TAB4 .

Only the model in FIG6 (a) employs cyclic translations and they outperform the other baselines.

We make a similar observation for hierarchical MCTN: FIG6 (e) with cyclic translations outperforms the trimodal baselines (f), (g) and (h).

The gap for the trimodal case is especially large.

This implies that using cyclic translations is crucial in learning joint representations.

Our intuition is that using cyclic translations: (1) encourages the model to enforce symmetry between the joint representations from source and target modalities, and (2) ensures that the joint representation retains maximal information from all modalities.

DISPLAYFORM0 MCTN Bi FIG6 ) FIG6 (c), which uses one Seq2Seq model for translations with FIG6 (d), which uses two separate Seq2Seq models: one for forward and one for backward translation.

We observe from TAB4 that (c) > (d), so using one model with shared parameters is better.

This is also true for hierarchical MCTN: (f) > (g).

We hypothesize that this is because training two Seq2Seq models requires more data and is prone to overfitting.

Q5: What is the impact when varying source and target modalities for cyclic translations?

As shown in Tables 2 and 3, we observe that language contributes most towards the joint representations.

For bimodal cases, combining language with visual is generally better than combining language with audio.

For hierarchical MCTN, presenting language as the source modality leads to the best performance, and a first level of cyclic translations between language and visual is better than between language and acoustic.

On the other hand, only translating between visual and acoustic modalities dramatically decreases performance.

Further adding language as a target modality for hierarchical MCTN will not help much as well.

Overall, language is still the most important modality for multimodal sentiment analysis and must be used as the source modality during translations.

Q6: What is the impact of using two levels of hierarchical translations instead of one level for three modalities?

Our hierarchical MCTN is shown in FIG6 (e).

In FIG6 (h), we concatenate two modalities as input and use only one phase of translation.

From TAB4 , we observe that (e) > (h): both levels of modality translations are important in the hierarchical MCTN.

We believe that representation learning is easier when the task is broken down recursively: using two translations each between a single pair of modalities, rather than a single translation between all modalities.

DISPLAYFORM1

To conclude, this paper investigated learning joint representations via cyclic modality translations from source to target modalities.

During testing, we only need the source modality for prediction which ensures that our model remains robust from noisy or missing target modalities.

We demonstrate that cyclic translations and seq2seq models are especially useful for learning joint multimodal representations.

In addition to achieving state-of-the-art results on three datasets, our model learns increasingly discriminative representations with more input modalities while maintaining robustness to all target modalities.

Our approach presents several exciting areas for future work, such as: 1) combining our approach with the transformer architecture BID59 for modality translations, 2) exploring pretrained deep language models BID12 BID42 for translations, as well as 3) extending our translation model to work other multimodal tasks involving language and raw speech signals (prosody), videos with multiple speakers (diarization), and combinations of static and temporal data (i.e. image captioning).

Here we present extra details on feature extraction for the language, visual and acoustic modalities.

Language: We used 300 dimensional Glove word embeddings trained on 840 billion tokens from the common crawl dataset BID41 .

These word embeddings were used to embed a sequence of individual words from video segment transcripts into a sequence of word vectors that represent spoken text.

Visual: The library Facet BID19 is used to extract a set of visual features including facial action units, facial landmarks, head pose, gaze tracking and HOG features BID68 .

These visual features are extracted from the full video segment at 30Hz to form a sequence of facial gesture measures throughout time.

Acoustic: The software COVAREP BID11 is used to extract acoustic features including 12 Mel-frequency cepstral coefficients, pitch tracking and voiced/unvoiced segmenting features BID14 , glottal source parameters BID9 BID15 BID0 BID2 BID1 , peak slope parameters and maxima dispersion quotients BID20 .

These visual features are extracted from the full audio clip of each segment at 100Hz to form a sequence that represent variations in tone of voice over an audio segment.

We perform forced alignment using P2FA BID62 to obtain the exact utterance time-stamp of each word.

This allows us to align the three modalities together.

Since words are considered the basic units of language we use the interval duration of each word utterance as one time-step.

We acquire the aligned video and audio features by computing the expectation of their modality feature values over the word utterance time interval BID28 .

We also implement the Stacked, (EF-SLSTM) BID17 , Bidirectional (EF-BLSTM) BID49

We present the full results across all baseline models in TAB4 .

MCTN using all modalities achieves new start-of-the-art results on binary classification accuracy, F1 score, and MAE on the CMU-MOSI dataset for multimodal sentiment analysis.

State-of-the-art results are also achieved on the ICT-MMMO and YouTube datasets TAB8 .

These results are even more impressive considering that MCTN only uses the language modality during testing, while other baseline models use all three modalities.

<|TLDR|>

@highlight

We present a model that learns robust joint representations by performing hierarchical cyclic translations between multiple modalities.

@highlight

This paper presents the Multimodal Cyclic Translation Network (MCTN) and evaluates it for multimodal sentiment analysis.