Chinese text classification has received more and more attention today.

However, the problem of Chinese text representation still hinders the improvement of Chinese text classification, especially the polyphone and the homophone in social media.

To cope with it effectively, we propose a new structure, the Extractor, based on attention mechanisms and design novel attention networks named Extractor-attention network (EAN).

Unlike most of previous works, EAN uses a combination of a word encoder and a Pinyin character encoder instead of a single encoder.

It improves the capability of Chinese text representation.

Moreover, compared with the hybrid encoder methods, EAN has more complex combination architecture and more reducing parameters structures.

Thus, EAN can take advantage of a large amount of information that comes from multi-inputs and alleviates efficiency issues.

The proposed model achieves the state of the art results on 5 large datasets for Chinese text classification.

Recently, Chinese text classification, as an important task of Chinese natural language processing (NLP), is extensively applied in many fields.

Deep learning has gotten great results on Chinese text classification.

However, the relevant studies are still insufficient compared with English, especially the method of Chinese text representation or encoding.

It is considered to be closely related to the result of Chinese text classification models.

Specifically, there are some issues in previous representation methods: (i) The word embedding (Le & Mikolov (2014) ; Mikolov et al. (2013) ; Pennington et al. (2014) ) is the most common method to represent the text, but it may become less effective when processing texts with the ambiguous word boundary such as Chinese texts. (ii) The character embedding (Zhang et al. (2015) ) can avoid the word segment.

However, using Pinyin characters loses the ideographic ability of Chinese characters, and using Chinese characters requires more training data because there are thousands of Chinese characters that are often used in daily life. (iii) Both the word embedding and the Chinese character embedding are hard to encode some intricate Chinese language phenomena about pronunciations, such as the polyphone and the homophone.

We notice that humans have associated the word or character with the corresponding pronunciation and remembered them in the process of learning the language.

Thus, when humans read texts in daily life, they spontaneously associate with the corresponding voices.

It is very difficult for computers and usually ignored by traditional text classification method.

Moreover, using the voice can cope with some representation issues of Chinese characters or words better, The polyphone and the homophone are 2 typical examples.

The former means different pronunciations and meanings are from the same character, and the latter means the same pronunciations are from different characters, which are usually used to represent similar meanings in social media.

And inspired by recent multimedia domain methods (Gu et al. (2018) ), the extra audio information can obtain better results.

However, large amounts of corresponding audio data are required difficultly.

Pinyin can precisely express the pronunciation by no more than 6 letters and is easily generated from texts, and it also solves representation issues of Chinese characters or words.

There are some typical examples that illustrate these points in detail.

Table 1 shows an example (sentence1) of the homophone of social medias.

There is a homophone "鸭梨山大" , the pronunciation

wǒ zhǐ néng shuō dōng xī shì hǎo dōng xī，1 hào de dìng dān，6 hào cái dào，dìng dān shì liǎng jiàn，yī jiàn yùn dá，yī jiàn zhōng tōng，yùn dá 2 tiān dào，zhōng tōng 6 tiān dào，shāng jiā wán fēn kāi sòng，zhēn shì ràng mǎi jiā yā lí shān dà！ sentence2: 大学英语六级考试：优选真题 标准模拟 没有王长喜好 好 好 用，后悔了 dà xué yīng yǔ liù jí kǎo shì：yōu xuǎn zhēn tí biāo zhǔn mó nǐ méi yǒu wáng zhǎng xǐ hǎo yòng，hòu huǐ le sentence3:

有一点点小(我个人的喜好 好 好)，勉强吧 yǒu yī diǎn diǎn xiǎo (wǒ gè rén de xǐ hào )，miǎn qiǎng ba (Pinyin) and the meaning of which are the same as "压力山大".

Table 1 also shows some examples (sentence2 and sentence3) of the polyphone of social medias.

The pronunciation (Pinyin) and the meaning of "好" are different in two sentences.

Besides,"hào" can represent "好" in sentence3 or "号" in sentence1.

In fact, it can represent the pronunciation of dozens of Chinese characters.

By those examples, we foucs on some points: In Chinese texts, some intricate language phenomena about pronunciations relatively easier to be recognized by a simple Pinyin encoder than by a complex Chinese character or word encoder.

And most of language phenomena about glyph are the opposite.

Based on the above points, we propose a new hybrid encoder (including word encoder and Pinyin character encoder) network to obtain better results for Chinese text classification, we call it Extractor-attention network (EAN).

Inspired by Transformer (Vaswani et al. (2017) ), we also propose a new structure named the Extractor.

The Extractor includes a multi-head self-attention mechanism with separable convolution layers (Chollet (2017) ).

In EAN, the Extractor is used to encode the information of Pinyin.

Besides, it is repeatedly used to combine word encoder with Pinyin encoder.

Compared with previous hybrid encoder methods, our method has relatively simple encoders and a complicated combination part, which uses a deep self-attention mechanism.

It makes EAN assign weights between features extracting by each encoder more accurately and avoid huge feature maps.

Moreover, we use pooling layers for downsampling and separable convolution layers to compress parameters.

Therefore, the Extractor network represent the Chinese text well, improve the classification accuracy, and the computational cost is relatively cheap.

The experimental results show that our model outperforms all baseline models on all datasets, and has fewer parameters in comparison to similar works.

Our primary contributions (i) Inspired by human language learning and reading, we design a novel method to solve the text representation issue of Chinese text classification, especially the language phenomena about pronunciations such as the polyphone and the homophone.

To the best of our knowledge, this is the first time that a hybrid encoding method including Pinyin has been used to solve those language phenomena expression problem. (ii) We propose a new attention architecture named the Extractor to experss Chinese texts information.

Besides, to better represent Chinese texts, we design a new hybird encoder method EAN based on the Extractor.

We also propose a complex attention method to combine word encoder with Pinyin encoder effectively, which can commendably balance the amount of information transmitted by 2 encoders. (iii) Our method is able to surpass previous methods.

It can get the state of the art results on public datasets.

Today deep neural networks have been widely used in text classification.

Compared with traditional methods Pang et al. (2002) , these methods do not rely on hand-crafted features.

Deep learning usually represents or encodes texts as feature vectors and classifies them.

The first step of text representation is to convert texts to low dimension vectors.

The embedding methods are often utilized in this process.

These methods include pre-trained word embedding (Le & Mikolov (2014); Mikolov et al. (2013) ; Pennington et al. (2014) ), character embedding (Conneau et al. (2017); Zhang et al. (2015) ), and word embedding without pre-trained (Blunsom et al. (2014) (Hochreiter & Schmidhuber (1997) ), can obtain good results in capturing sequence features.

Both CNN-based methods (Conneau et al. (2017) ; Blunsom et al. (2014) ; Kim (2014) ; Kim et al. (2016) ; Zhang et al. (2015) ) and RNN-based methods (Tang et al. (2015) ) have achieved outstanding accomplishments in text classification.

Besides, the attention mechanisms are usually used in NLP to capture relatively more critical features.

Some of them are based on RNN (Gu et al. (2018) ; Yang et al. (2016) ) or CNN (Gu et al. (2018) ).

use CNN within the attention mechanism.

Others are based solely on attention mechanisms such as Vaswani et al. (2017) , which performs exceptionally well in many tasks of NLP.

It means that using attention mechanisms entirely without CNN or RNN to represent texts is perfectly feasible.

At last, there is a softmax (multiclass classification) or sigmoid (binary or multi-label classification) classifier.

Sometimes full-connection layers may be added in front of it such as Zhang & LeCun (2017) .

Compared with the mainstream English text classification, the most significant difference of Chinese text classification is the text presentation approach.

Sometimes they are not different except for embedding (Conneau et al. (2017) ; ; Zhang et al. (2015) ).

Moreover, Shi et al. (2015) propose the Radical embedding which is similar to Mikolov et al. (2013) but uses Chinese radicals instead of words.

Zhuang et al. (2017; utilize Chinese character strokes and multi-layers CNN to represent Chinese text.

Liu et al. (2017) propose the visual character embedding that creates an image for the characters and employs CNN to process the image.

The experiments show that it performs well in different languages data including Chinese, Japanese, and Korean.

Chinese Pinyin has gained popularity among relative researchers in recent years.

It is used in character embedding (Conneau et al. (2017) ; Zhang et al. (2015) ) at the primary stage.

Then Pinyin is regarded as the Chinese word in pre-trained word embedding methods or is combined with the Chinese word as training data (sometimes also including the Chinese character).

Zhang & LeCun (2017) propose a variety of encoder methods of Chinese, Japanese, Korean and English.

These methods mainly consist of differently simple encoding of character, word, and romanization word. propose a multi-channel CNN for Chinese sentiment analysis.

The channels include word, character, and Pinyin.

This model shows that the combination always performs better than using the Chinese word or Pinyin alone.

Those Chinese text classification methods have gotten good results in different datasets.

However, there are still some disadvantages: Some methods (Liu et al. (2017) ; Zhang & LeCun (2017) ) are relatively straightforward so that not do well in lengthy and complicated text data, and some methods (Conneau et al. (2017) ) have quite a few parameters result in relatively inefficiency.

Our Extractor-based method can be divided into several parts: the word encoder, the Pinyin encoder, the combination part, and a classifier.

We illustrate these parts in the following sections.

Besides, the attention mechanism is also repeatedly employed inside and outside of the Extractor, and thus we call this method the Extractor-attention network (EAN).

The method architecture is shown in Figure  1 .

In EAN, Batch Normalization (BN) (Ioffe & Szegedy (2015) ) is used after all convolutional layers.

The activation function is the rectified linear unit (RELU) for all convolutional layers and full-connected layers.

In the embedding part,the pre-trained word embedding method is employed like most text classification methods.

There are 3 consecutive operations after it: Gaussian noise, dropout, and BN.

They preclude overfitting or making the model converged faster.

A single separable CNN (Chollet (2017)) layer is placed at the end. (2015)).

LN indicates Layer Normalization (Ba et al. (2016) ).

The Pinyin encoder, which consists of 2 parts: the character embedding part and an Extractor, is designed to represent audio information from Chinese texts data and avoid the issue of word segment accuracy.

It can supply some information which is difficult to be extracted from Chinese texts, especially the polyphone and the homophone.

In Pinyin character embedding part, the embedding layer is similar to Zhang & LeCun (2017) .

The characters consist of Pinyin letters, digits and punctuations.

We use a Gaussian distribution to initialize the embedding weights.

Therefore, the Gaussian noise operation is not used.

After the embedding layer, we employ BN, a combination of separable CNN and max pooling, and dropout.

The Extractor is composed of an attention block and an extraction block, as roughly shown in Figure  1 .

The residual connection ) is employed in each block, which can alleviate gradient issues, speed up training, and strengthen feature propagation.

Layer Normalization (LN) (Ba et al. (2016) ) is applied after the residual connection.

The attention block The attention block extracts features by assigning weights to itself.

Some attention structures that include self-attention and multi-head attention have gotten great results in many NLP tasks, especially the Transformer (Vaswani et al. (2017) ).

Thus, nonlinear multi-head self-attention structure is employed in the attention block to enhance the representation ability of the model.

The original linear operation of multi-head attention is replaced by the separable CNN.

Compared with linear operations such as fully-connection layers, CNN is more capable of capturing local and position-invariance features.

Besides, CNN is a faster computation due to parallel-processing friendly, peculiarly separable CNN with fewer parameters.

These properties are required for Chinese text representation and classification.

The input of this block is the output of the Pinyin character embedding part.

Define Q, K, V as the matrixes which consist of queries, keys, and values, respectively.

In self-attention, Q, K, V are the identical matrixes of size l × d, where l is the input length, d is the number of the input channels.

To obtain different attention functions, different representation subspaces should be generalized.

In order to achieve it, we have:

Where n is the number of heads, Separable Conv1D is the separable 1D convolution function, and Q :,:,i , K :,:,i , V :,:,i ∈ R l×d k are the i-th matrix of Q s , K s , V s , respectively.

Define d k is the number of channels of Q :,:,i , K :,:,i , V :,:,i :

For each head H i , the Scaled Dot-Product Attention is employed to capture the internal relationship:

All the heads are concatenated, then processed by a separable CNN layer.

Define P as the output of the CNN, and it is also the output of block:

The extraction block The extraction block compresses the feature maps and further extracts features.

Compared with the word embedding, there is no word boundary issue in the Pinyin character embedding.

However, the Pinyin character embedding requires a much longer length than the word embedding.

Thus, the feature maps of Pinyin encoder may be too large to be processed efficiently.

The filtration of feature maps is employed to alleviate this problem, which is why the extraction block is designed.

At first, a downsampling operation by max pooling is used to primarily reduce feature maps of the output of the attention block.

To further extract the relative spatial information and introduce more nonlinear transformation, 2-layers separable CNN is used after the max pooling layer.

By this block, the feature maps become narrow.

The key problem of the hybrid encoder method lies in combining the encoders.

Traditional combination methods often use the simple features concatenation ) or the complicated encoders (including attention structures) with straightforward features combination (Amplayo et al. (2018)).

We choose relatively more uncomplicated encoders and more complex combination ways to avoid redundancy and overmuch parameters.

The combination part concatenates the output of word and Pinyin encoder at the first step.

And then the Extractor is employed repeatedly to extract long-term dependencies and global features.

Besides, Extractors can effectively reduce feature maps.

The Extractor structure is similar to that of Pinyin encoder.

Finally, a Scaled Dot-Product Attention is employed to weight the output of the final Extractor by the self-attention scores.

We do not choose the global max pooling layer or the flatten layer, because the global max pooling layer is coarse, and the flatten layer has too many parameters.

Define X ∈ R d l ×dx is the output matrix of the final Extractor.

d l is the input length of X, d x is the number of channels of X.The self-attention scores A can be computed by X:

The final hybrid representation f is the sum of weighted features by A:

Where A:, i ∈ A, X:, i ∈ X. f is the output of the combination part, and is also the input of the classifier.

The classifier is the final part, which consists of 1 or 3 full-connected layers and a softmax layer.

The dropout and BN are used after each full-connected layer in this part.

Zhang & LeCun (2017) .

Specifically, Ifeng and Chinanews are news topic classification datasets.

Dianping, JD.b, and JD.f are sentiment classification datasets on user review.

All datasets are Chinese text datasets.

The summary statistics of datasets are shown in Table 2 .

We selected 10K documents from the training data for use as the validation set on each dataset.

Baselines We compared EAN with various methods, including EmbedNet, GlyphNet, OnehotNet, Linear model (multinomial logistic regression), fastText (Joulin et al. (2017) ), and the EAN without the hybrid encoder (removing the concatenation layer and Pinyin encoder).

All experiments data of those baseline methods come from Zhang & LeCun (2017) .

We will omit an exhaustive background description of the baseline methods and refer readers to Zhang & LeCun (2017) .

Besides, to comfortably compare the parameters between EAN and other methods, especially the hybrid methods, we design a comparison baseline based on EAN.

All the Extractor are replaced by the Transformer encoder structure (Multi-Head Attention and Feed-Forward Networks).

Thus, we name it TAN.

We tested TAN with hybrid encoder and TAN with word encoder.

We also compared the parameters with some text classification model including Bi-BloSAN ) (the result are cited from Yu et al. (2017) ) and VDCNN (Conneau et al. (2017) ).

Setup In word embedding, we employed Jieba, a word segmentation package, to process Chinese texts and used the SGNS vectors (Wikipedia-zh (Word + Ngram)) by as the embedding initialization.

In Pinyin character embedding, we obtained the Pinyin texts by the pypinyin package.

The character embedding weights were initialized from a Gaussian distribution with an initial mean of 0 and a standard deviation of 0.05.

The dropout rate of both embeddings was 0.2 or 0.5.

The dimension of word embedding was 300, and that of Pinyin characters was 256.

Empirically, A separable convolutional layer of 256 convolutions of size 3 was employed in the word and Pinyin character encoder.

We used an Extractor in Pinyin encoder and 3 Extractors in combination part.

All Extractors owned the same setup: There were 256 input channels, 4 heads in the attention block.

Thus, separable convolutional layers of 64 convolutions of size 3 were applied to generate Q :,:,i , K :,:,i , V :,:,i , and a separable convolutional layer of 256 convolutions of size 3 were used at the end of the attention block.

The max pooling with size 3 and stride 2 was used in the extraction block.

It is similar to the max pooling in Pinyin character encoder.

After max pooling, there were 2 separable convolutional layers with the setup as same as that in encoder parts.

We used 1 or 3 fullconnected layers with size 256 and dropout rate 0.2 or 0.5 in the classifier.

Moreover, we employed Adam optimizer with an initial learning rate of 0.001.

The loss function was the cross-entropy.

All experiments were implemented using Keras and were performed on GPU 1080Ti.

Testing Error Rates The testing error rate results are shown in Table 3 .

Due to the page limit, we list the best results of their variations with different hyperparameters.

The results of EAN with the hybrid encoder or with the single word encoder are better than the state-of-the-art baseline methods (including TAN) on all datasets.

It shows that EAN excels in the accuracy of Chinese text classification and the Extractor is very powerful to capture long-range dependencies or global features.

And the EAN with the hybrid encoder performs better than EAN with the single encoder on all data sets, which proves the advantage of the hybrid encoder.

We also observe that TAN with the hybrid We design 2 extra combination methods to prove the effectiveness of our combination method, which is shown in Figure 2 .

The concatenate combination method remove Extractors and self-attention after the concatenate layer (simple features concatenation), and the Extractor+Concatenate (E+C) combination method place Extractors after the word encoder and the Pinyin encoder respectively (the complicated encoders with straightforward features combination).

The Concatenate+Extractor (C+E) combination method is our combination method.

The results are shown in Table 4 .

In Table 4 rows 1, 2, and 4, we observe that the C+E combination method is better than other combination methods, and the E+C combination method is better than the concatenate combination method.

It means that the simple features concatenation without attention structures is relatively difficult to capture the associations across encoders, and the straightforward encoders with complicated features combination may work better in comparison to the complicated encoders with straightforward features combination.

The results of TAN are very close to those of EAN, but most results of letter are better than those of former.

The model which obtain the optimal results contained concatenate+Extractor method, 3 Extractors(Transformers) with 4 heads.

Table 5 .

There are diiferent parameters of EAN due to diiferent word and character lengths on different datasets, especially Concatenate and E+C combation methods.

The parameters of EAN (C+E) are fewer than other models, which shows the excellent property of the Extractor.

specifically, the parameters of EAN (C+E) are fewer than the parameters of TAN because we use the separable CNN to replace the linear operation such as full-connection layers and employ downsampling operation like the max pooling layer to compress feature maps.

In fact, feature maps are halved in each extraction block of the Extractor.

Moreover, the parameters of EAN (C+E) are much fewer than EAN (Concatenate) and EAN (E+C).

It means our combination method is computationally relatively cheaper.

Thus, as we mentioned before, the feature maps of EAN are narrow but enough to obtain a good result.

It can alleviate the efficiency problem of the hybrid encoder method such as too many parameters or too slow speed.

This paper proposes a novel attention network, the Extractor-attention network (EAN), for Chinese text classification.

Compared to the traditional Chinese text classification methods using only word encoder, our approach uses hybrid encoder including words and Pinyin characters, which takes full advantage of the extra Pinyin information to improve the performance.

Moreover, there is a new structure named the Extractor in our work, reduces the number of parameters in EAN and makes it excellent to extract feature.

Thus, EAN obtains the state of the art results on 5 public Chinese text classification datasets.

Finally, we also analyze the effects of different encoders structures on the method.

<|TLDR|>

@highlight

We propose a novel attention networks with the hybird encoder to solve the text representation issue of Chinese text classification, especially the language phenomena about pronunciations such as the polyphone and the homophone.

@highlight

This paper proposes an attention-based model consisting of the word encoder and Pinyin encoder for the Chinese text classification task, and extends the architecture for the Pinyin character encoder.

@highlight

Proposal for an attention network where both word and pinyin are considered for Chinese representation, with improved results shown in several datasets for text classification.