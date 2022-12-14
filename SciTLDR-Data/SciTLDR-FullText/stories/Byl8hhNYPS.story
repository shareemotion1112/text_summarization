Though visual information has been introduced for enhancing neural machine translation (NMT), its effectiveness strongly relies on the availability of large amounts of bilingual parallel sentence pairs with manual image annotations.

In this paper, we present a universal visual representation learned over the monolingual corpora with image annotations, which overcomes the lack of large-scale bilingual sentence-image pairs, thereby extending image applicability in NMT.

In detail, a group of images with similar topics to the source sentence will be retrieved from a light topic-image lookup table learned over the existing sentence-image pairs, and then is encoded as image representations by a pre-trained ResNet.

An attention layer with a gated weighting is to fuse the visual information and text information as input to the decoder for predicting target translations.

In particular, the proposed method enables the visual information to be integrated into large-scale text-only NMT in addition to the multimodel NMT.

Experiments on four widely used translation datasets, including the WMT'16 English-to-Romanian, WMT'14 English-to-German, WMT'14 English-to-French, and Multi30K, show that the proposed approach achieves significant improvements over strong baselines.

Visual information has been introduced for neural machine translation in some previous studies (NMT) Barrault et al., 2018; Ive et al., 2019) though the contribution of images is still an open question (Elliott, 2018; Caglayan et al., 2019) .

Typically, each bilingual (or multilingual) parallel sentence pair is annotated manually by one image describing the content of this sentence pair.

The bilingual parallel corpora with manual image annotations are used to train a multimodel NMT model by an end-to-end framework, and results are reported on a specific data set, Multi30K .

One strong point of the multimodel NMT model is the ability to use visual information to improve the quality of the target translation.

However, the effectiveness heavily relies on the availability of bilingual parallel sentence pairs with manual image annotations, which hinders the image applicability to the NMT.

As a result, the visual information is only applied to the translation task over a small and specific multimodel data set Multi30K , but not to large-scale text-only NMT (Bahdanau et al., 2014; Gehring et al., 2017; Vaswani et al., 2017) and low-resource text-only NMT (Fadaee et al., 2017; Lample et al., 2018; .

In addition, because of the high cost of annotation, the content of one bilingual parallel sentence pair is only represented by a single image, which is weak in capturing the diversity of visual information.

The current situation of introducing visual information results in a bottleneck in the multimodel NMT, and is not feasible for text-only NMT and low-resource NMT.

In this paper, we present a universal visual representation (VR) method 1 relying only on image-monolingual annotations instead of the existing approach that depends on image-bilingual annotations, thus breaking the bottleneck of using visual information in NMT.

In detail, we transform the existing sentence-image pairs into topic-image lookup table from a small-scale multimodel data set Multi30K.

During the training and decoding process, a group of images with similar topic to the source sentence will be retrieved from the topic-image lookup table learned by the term frequency-inverse document frequency, and thus is encoded as image representations by a pretrained ResNet (He et al., 2016) .

A simple and effective attention layer is then designed to fuse the image representations and the original source sentence representations as input to the decoder for predicting target translations.

In particular, the proposed approach can be easily integrated into the text-only NMT model without annotating large-scale bilingual parallel corpora.

The proposed method was evaluated on four widely-used translation datasets, including the WMT'16 Englishto-Romanian, WMT'14 English-to-German, WMT'14 English-to-French, and Multi30K which are standard corpora for NMT and multi-modal machine translation (MMT) evaluation.

Experiments and analysis show effectiveness.

In summary, our contributions are primarily three-fold:

1.

We present a universal visual representation method that overcomes the shortcomings of the bilingual (or multilingual) parallel data with manual image annotations for MMT.

2.

The proposed method enables the text-only NMT to use the multimodality of visual information without annotating the existing large scale bilingual parallel data.

3. Experiments on different scales of translation tasks verified the effectiveness and generality of the proposed approach.

Building fine-grained representation with extra knowledge is an important topic in language modeling b; a) , among which adopting visual modality could potentially benefit the machine with more comprehensive perception of the real world.

Inspired by the studies on the image description generation (IDG) task (Mao et al., 2014; Elliott et al., 2015; Venugopalan et al., 2015; Xu et al., 2015) , a new shared translation task for multimodel machine translation was addressed by the machine translation community .

In particular, the released dataset Multi30K includes 29,000 multilingual (English, German, and French) parallel sentence pairs with image annotations Barrault et al., 2018) .

Subsequently, there has been a rise in the number of studies (Caglayan et al., 2016; Calixto et al., 2016; Huang et al., 2016; Libovick??? & Helcl, 2017; Helcl et al., 2018) .

For example, proposed a doubly-attentive multi-modal NMT model to incorporate spatial visual features, improving the translation performance.

Compared with spatial-visual features, further incorporated global image features as words in the source sentence and to enhance the encoder or decoder hidden state.

In contrast, some recent studies indicated that the visual modality is either unnecessary (Zhang et al., 2017) or only marginally beneficial (Gr??nroos et al., 2018) .

More recently, Ive et al. (2019) showed that visual information is only needed in particular cases, such as for ambiguous words where the textual context is not sufficient.

However, these approaches only center around a small and specific Multi30K data set to build multimodel NMT model, which hinders image applicability to NMT.

The reason would be the high cost of image annotations, resulting potentially in the image information not being adequately discovered.

We believe that the capacity of MMT has not yet been excavated sufficiently and there is still a long way to go before the potential of MMT is fully discovered.

In this work, we seek to break this constraint and enable visual information to benefit NMT, especially text-only NMT.

return TF-IDF dictionary F 9: end procedure 10: procedure LOOKUP(S, E, F)

for For each pair {T i , e i } ??? zip{S, E} do

Rank and pick out the top-w "topic" words in the sentence according to the TF-IDF score in the dictionary F, and each sentence is reformed as T = {t 1 , t 2 , . . .

, t w }

Pair the w words with the corresponding image e i

for For each word t j in T do In this section, we will introduce the proposed universal visual representation method.

Generally, the default input setting of the MMT is a sentence-image pair.

Our basic intuition is to transform the existing sentence-image pairs into topic-image lookup table 2 , which assumes the topic words in a sentence should be relevant to the paired image.

Consequently, a sentence can possess a group of images by retrieving the topic-image lookup table.

To focus on the major part of the sentence and suppress the noise such as stopwords and low-frequency words, we design a filtering method to extract the "topic" words of the sentence through the term frequency-inverse document frequency (TF-IDF) 3 inspired by Chen et al. (2019) .

Specifically, given an original input sentence X = {x 1 , x 2 , . . .

, x I } of length I and its paired image e, X is first filtered by a stopword list 4 and then the sentence is treated as a document g. We then compute TF-IDF T I i,j for each word x i in g,

where o i,j represents the number of occurrences of the word x i in the input sentence g, |G| the total number of source language sentences in the training data, and |j : x i ??? g| the number of source sentences including word x i in the training data.

We then select the top-w high TF-IDF words as the new image description T = {t 1 , t 2 , . . .

, t w } for the input sentence X. After preprocessing, each filtered sentence T is paired with an image e, and each word t i ??? T is regarded as the topic word for image e.

After processing the whole corpus (i.e., Multi30K), we form a topic-image lookup table Q as described in Algorithm 1, in which each topic word t i would be paired with dozens of images.

Image Retrieval For input sentence, we first obtain its topic words according to the text preprocessing method described above.

Then we retrieve the associated images for each topic word dog is playing in the snow dog (1, 512) playing (1,531) snow (439) (a) a black dog and a spotted dog are fighting (b) a dog is running in the snow (c) a dog is playing with a hose (d) a family playing on a tractor on a beautiful day (e) two people working on removing snow from a roof (f) a black dog and a white dog are standing on snow corpus (29,000) Figure 1 : Illustration of the proposed visual retrieval.

from the lookup table Q and group all the retrieved images together to form an image list G. We observe that an image might be associated with multiple topic words so that it would occur multiple times in the list G. Therefore, we sort the images according to the frequency of occurrences in G to maintain the total number of images for each sentence at m. In the left block, we show six examples of sentence-image pairs in which the topic words are in boldface.

Then we process the corpus using the topic-image transformation method demonstrated above and obtain the topic-image lookup table.

For example, the word dog is associated with 1,512 images.

For an input source sentence, we obtain the topic words (in boldface) using the same preprocessing.

Then we retrieve the corresponding images from the lookup table for each topic word.

Now we have a list of images, and some images appear multiple times as they have multiple topics (like the boxed image in Figure 1 ).

So we sort the retrieved image list by the count of occurrence to pick out the top-m images that cover the most topics of the sentence.

At test time, the process of getting images is done using the image lookup table built by the training set, so we do not need to use the images from the dev and test sets in Multi30K dataset 6 .

Intuitively, we do not strictly require the manual alignment of the word (or concept) and image, but rely on the co-occurrence of topic word and image, which is simpler and more general.

In this way, we call our method as universal visual retrieval.

In this section, we introduce the proposed universal visual representation (VR) method for NMT.

The overview of the framework of our proposed method is shown in Figure 2 .

In the state-of-the-art Transformer-based NMT (Vaswani et al., 2017) , source information is encoded as source representation by an SAN-based encoder with multiple layers.

Specifically, the encoder is composed of a stack of L identical layers, each of which includes two sub-layers.

The first sublayer is a self-attention module, whereas the second is a position-wise fully connected feed-forward network.

A residual connection (He et al., 2016) is applied between the two sub-layers, and then 5 More examples are provided in the Appendix A.1.

6 The lookup table can be easily adapted to a wide range of other NLP tasks even without any paired image, and therefore opens our proposed model to generalization.

a layer normalization (Ba et al., 2016 ) is performed.

Formally, the stack of learning the source representation is organized as follows:

where ATT l (??), LN(??), and FFN l (??) are the attention module, layer normalization, and the feedforward network for the l-th identical layer, respectively.

After retrieval as described in Section 3, each original sentence X = {x 1 , x 2 , . . .

, x I } is paired with m images E = {e 1 , e 2 , . . .

, e m } retrieved from the topic-image lookup table Q. First, the source sentence X={x 1 , x 2 , . . .

, x I } is fed into the encoder (Eq.2) to learn the source sentence representation H L .

Second, the images E ={e 1 , e 2 , . . . , e m } are the inputs to a pre-trained ResNet (He et al., 2016) followed by a feed forward layer to learn the source image representation textM ??? R m??2048 .

Then, we apply an attention mechanism 7 to append the image representation to the text representation:

where {K M , V M } are packed from the learned source image representation M.

7 We used single head here for simplicity.

Intuitively, NMT aims to produce a target word sequence with the same meaning as the source sentence rather than a group of images.

In other words, the image information may play an auxiliary effect during the translation prediction.

Therefore, we compute ?? ??? [0, 1] to weight the expected importance of source image representation for each source word:

where W ?? and U ?? are model parameters.

We then fuse H L and H to learn an effective source representation:

Finally, H is fed to the decoder to learn a dependent-time context vector for predicting target translation.

Note that there is a single aggregation layer to fuse image and text information.

The proposed method was evaluated on four widely-used translation datasets, including WMT'16 English-to-Romanian (EN-RO), WMT'14 English-to-German (EN-DE), WMT'14 English-toFrench (EN-DE), and Multi30K which are standard corpora for NMT and MMT evaluation.

1) For the EN-RO task, we experimented with the officially provided parallel corpus: Europarl v7 and SETIMES2 from WMT'16 with 0.6M sentence pairs.

We used newsdev2016 as the dev set and newstest2016 as the test set.

2) For the EN-DE translation task, 4.43M bilingual sentence pairs of the WMT14 dataset were used as training data, including Common Crawl, News Commentary, and Europarl v7.

The newstest2013 and newstest2014 datasets were used as the dev set and test set, respectively.

3) For the EN-FR translation task, 36M bilingual sentence pairs from the WMT14 dataset were used as training data.

Newstest12 and newstest13 were combined for validation and newstest14 was used as the test set, following the setting of Gehring et al. (2017) .

4) The Multi30K dataset contains 29K English???{German, French} parallel sentence pairs with visual annotations.

The 1,014 English???{German, French} sentence pairs visual annotations are as dev set.

The test sets are test2016 and test2017 with 1,000 pairs for each.

Image Retrieval Implementation We used 29,000 sentence-image pairs from Multi30K to build the topic-image lookup table.

We segmented the sentences using the same BPE vocabulary as that for each source language.

We selected top-8 (w = 8) high TF-IDF words, and the default number of images m was set 5.

The detailed case study is shown in Section 6.2.

After preprocessing, we had about 3K topic words, associated with a total of 10K images for retrieval.

Image features were extracted from the averaged pooled features of a pre-trained ResNet50 CNN (He et al., 2016) .

This led to feature maps V ??? R 2048 .

Baseline Our baseline was text-only Transformer (Vaswani et al., 2017) .

We used six layers for the encoder and the decoder.

The number of dimensions of all input and output layers was set to 512 and 1024 for base and big models.

The inner feed-forward neural network layer was set to 2048.

The heads of all multi-head modules were set to eight in both encoder and decoder layers.

For Multi30K dataset, we further evaluated a multimodal baseline (denoted as MMT) where each source sentence was paired with an original image.

The other settings were the same as our proposed model.

The byte pair encoding algorithm was adopted, and the size of the vocabulary was set to 40,000.

In each training batch, a set of sentence pairs contained approximately 4096??4 source tokens and 4096??4 target tokens.

During training, the value of label smoothing was set to 0.1, and the attention dropout and residual dropout were p = 0.1.

We used Adam optimizer (Kingma & Ba, 2014) of 1,000 batches on the dev set.

For the Multi30K dataset, we trained the model up to 10,000 steps, and the training was early-stopped if dev set BLEU score did not improve for ten epochs.

For the EN-DE, EN-RO, and EN-FR tasks, following the training of 200,000 batches, the model with the highest BLEU score of the dev set was selected to evaluate the test sets.

During the decoding, the beam size was set to five.

All models were trained and evaluated on a single V100 GPU.

Multi-bleu.perl 8 was used to compute case-sensitive 4-gram BLEU scores for all test sets.

The signtest (Collins et al., 2005 ) is a standard statistical-significance test.

In addition, we followed the model configurations of Vaswani et al. (2017) to train Big models for WMT EN-RO, EN-DE, and EN-FR translation tasks.

All experiments were conducted with fairseq 9 (Ott et al., 2019) .

The analysis in Section 6 is conducted on base models.

Table 1 shows the results for the WMT'14 EN-DE, EN-FR, and WMT'16 EN-RO translation tasks.

Our implemented Transformer (base/big) models showed similar BLEU scores with the original Transformer (Vaswani et al., 2017) , ensuring that the proposed method can be evaluated over strong baseline NMT systems.

As seen, the proposed +VR significantly outperformed the baseline Transformer (base), demonstrating the effectiveness of modeling visual information for text-only NMT.

In particular, the effectiveness was adapted to the translation tasks of the three language pairs which have different scales of training data, verifying that the proposed approach is a universal method for improving the translation performance.

Our method introduced only 1.5M and 4.0M parameters for base and big transformers, respectively.

The number is less than 3% of the baseline parameters as we used the fixed image embeddings from the pre-trained ResNet feature extractor.

Besides, the training time was basically the same as the baseline model (Section 6.4).

In addition, the proposed method was also evaluated for MMT on the multimodel dataset, Multi30K.

Results in Table 2 show that our model also outperformed the transformer baseline.

Compared with the results in text-only NMT, we find that the image presentation gave marginal contribution, which was consistent with the findings in previous work (Zhang et al., 2017; Gr??nroos et al., 2018; Caglayan et al., 2019) .

The most plausible reason might be that the sentences in Multi30K are so simple, short, and repetitive that the source text is sufficient to perform the translation (Caglayan et al., 2019; Ive et al., 2019) .

This verifies our assumption of the current bottleneck of MMT due to the limitation of Multi30K and shows the necessity of our new setting of transferring multimodality into more standard and mature text-only NMT tasks.

Table 2 : Results from the test2016 and test2017 for the MMT task.

Del denotes the deliberation network in (Ive et al., 2019) .

is the official baseline (text-only NMT) on WMT17-Multi30K 2017 test data.

Trans. is short for transformer and MMT is the multimodal baseline described in Section 5.2.

Because we used the same model for test2016 and test2017 evaluation, the numbers of parameters are the same.

The contribution of the lookup table could be two folds: 1) the content connection of the sentences and images; 2) the topic-aware co-occurrence of similar images and sentences.

There are cases when paired images are not accurately related to the given sentence.

A simple solution is to set a threshold heuristically for the TF-IDF retrieval to filter out the "improper" images.

However, we maintain the specific number of the images in this work because of the second potential benefits of the cooccurrence, by taking images as diverse topic information.

According to Distributional Hypothesis (Harris, 1954) which states that, words that occur in similar contexts tend to have similar meanings, we are inspired to extend the concept in multimodal world, the sentences with similar meanings would be likely to pair with similar even the same images.

Therefore, the consistent images (with similar topic) could play the role of topic or type hints for similar sentence modeling.

This is also very similar to the idea of word embedding by taking each image as a "word".

Because we use the average pooled output of ResNet, each image is represented as 2400d vector.

For all the 29,000 images, we have an embedding layer with size (29000, 2400).

The "content" of the image is regarded as the embedding initialization.

It indeed makes effects, but the capacity of the neural network is not up to it.

In contrast, the mapping from text word to the index in the word embedding is critical.

Similarly, the mapping of sentence to image in image embedding would be essential, i.e., the similar sentences (with the same topic words) tend to map the same or similar image.

To verify the hypotheses, we replace our ResNet features with 1) Shuffle: shuffle the image features but keep the lookup table; 2) Random Init: randomly initialize the image embedding but keep the lookup table; 3) Random Mapping: randomly retrieve unrelated images.

The BLEU scores are on EN-RO are 33.53, 33,28, 32.14, respectively.

The results of 1-2 are close to the proposed VR (33.78) and outperform the baseline (32.66), which shows that the content of images would not be very important.

The ablation 3) gives a lower result, which verifies the necessity of the mapping especially the topic relationship.

To evaluate the influence of the number of paired images m, we constrained m in {0, 1, 3, 5, 7, 9, 15, 20, 30} for experiments on the EN-RO test set, as shown in Figure 4 .

When m = 0, the model is the baseline NMT model, whose BLEU score was lower than all the models with images.

As the number of images increases, the BLEU score also increased at the beginning (from 32.66 to 33.78) and then slightly decreased when m exceeds 5.

The reason might be that too many images for a sentence would have greater chance of noise.

Therefore, we set m = 5 in our models.

The number of sentence-image pairs to create the lookup table could also make effects.

We randomly split the pairs of Multi30K into the proportion in [0.1, 0.3, 0.5, 0.7, 0.9] , the corresponding BLEU scores for 33.44, 34.01, 34.06, 33.80] .

Furthermore, we also evaluate the performance by adding external sentence-pairs from the training set of MS COCO image caption dataset (Lin et al., 2014) .

The BLEU scores are 33.55 and 33.71 respectively for COCO only and Multi30K+COCO.

These results indicate that a modest number of pairs would be beneficial.

In our model, the weight ?? of the gated aggregation method was learned automatically to measure the importance of the visual information.

We compared by manually setting the weight ?? into scalar values in {0.1, 0.3, 0.5, 0.7, 0.9} for experiments on the EN-RO test set.

Figure 5 shows that all models with manual ?? outperformed the baseline Trans. (base), indicating the effectiveness of image information.

In contrast, they were inferior to the performance of our model.

This means that the degree of dependency for image information varies for each source sentence, indicating the necessity of automatically learning the gating weights of image representations.

There are mainly two extra computation cost using our method, including 1) obtaining image data for sentences and 2) learning image representations, which are negligible compared with training a NMT model.

The time of obtaining image data for MT sentences for EN-RO dataset is less than 1 minute using GPU.

The lookup table is formed as the mapping of token (only topic words) index to image id.

Then, the retrieval method is applied as the tensor indexing from the sentence token indices (only topic words) to image ids, which is the same as the procedure of word embedding.

The retrieved image ids are then sorted by frequency.

Learning image representations takes about 2 minutes for all the 29,000 images in Multi30K using 6G GPU memory for feature extraction and 8 threads of CPU for transforming images.

The extracted features are formed as the "image embedding layer" with the size of (29000, 2400) for quick accessing in neural network.

This work presents a universal visual representation method for neural machine translation relying on monolingual image annotations, which breaks the restraint of heavy dependency on bilingual sentence-image pairs in the current multimodal NMT setting.

In particular, this method enables visual information to be applied to large-scale text-only NMT through a topic-image lookup.

We hope this work sheds some light for future MMT research.

In the future, we will try to adopt the proposed method to other tasks.

a man walks by a silver vehicle an elderly woman pan frying food in a kitchen small boy carries a soccer ball on a field man (6,675)

woman (3, 484) food (342) Retrieved Images for Sentences Figure 5 : Examples of the topic-image lookup table and retrieved images for sentences in Multi30K dataset.

We only show six images for each topic or sentence for instance.

The topics in each sentence are in boldface.

@highlight

This work proposed a universal visual representation for neural machine translation (NMT) using retrieved images with similar topics to source sentence,  extending image applicability in NMT.