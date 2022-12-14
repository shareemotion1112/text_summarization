Existing attention mechanisms, are mostly item-based in that a model is trained to attend to individual items in a collection (the memory) where each item has a predefined, fixed granularity, e.g., a character or a word.

Intuitively, an area in the memory consisting of multiple items can be worth attending to as a whole.

We propose area attention: a way to attend to an area of the memory, where each area contains a group of items that are either spatially adjacent when the memory has a 2-dimensional structure, such as images, or temporally adjacent for 1-dimensional memory, such as natural language sentences.

Importantly, the size of an area, i.e., the number of items in an area or the level of aggregation, is dynamically determined via learning, which can vary depending on the learned coherence of the adjacent items.

By giving the model the option to attend to an area of items, instead of only individual items, a model can attend to information with varying granularity.

Area attention can work along multi-head attention for attending to multiple areas in the memory.

We evaluate area attention on two tasks: neural machine translation (both character and token-level) and image captioning, and improve upon strong (state-of-the-art) baselines in all the cases.

These improvements are obtainable with a basic form of area attention that is parameter free.

In addition to proposing the novel concept of area attention, we contribute an efficient way for computing it by leveraging the technique of summed area tables.

Attentional mechanisms have significantly boosted the accuracy on a variety of deep learning tasks BID0 BID10 BID20 .

They allow the model to selectively focus on specific pieces of information, which can be a word in a sentence for neural machine translation BID0 BID10 or a region of pixels in image captioning BID20 BID13 ).An attentional mechanism typically follows a memory-query paradigm, where the memory M contains a collection of items of information from a source modality such as the embeddings of an image or the hidden states of encoding an input sentence, and the query q comes from a target modality such as the hidden state of a decoder model.

In recent architectures such as Transformer BID15 , self-attention involves queries and memory from the same modality for either encoder or decoder.

Each item in the memory has a key and value (k i , v i ), where the key is used to compute the probability a i regarding how well the query matches the item (see TAB3 ).

DISPLAYFORM0 The typical choices for f att include dot products qk i BID10 and a multilayer perceptron BID0 .

The output O M q from querying the memory M with q is then calculated as the sum of all the values in the memory weighted by their probabilities (see Equation 2), which can be fed to other parts of the model for further calculation.

During training, the model learns to attend to specific piece of information, e.g., the correspondance between a word in the target sentence and a word in the source sentence for translation tasks.

DISPLAYFORM1 Attention mechanisms are typically designed to focus on individual items in the entire memory, where each item defines the granularity of what the model can attend to.

For example, it can be a character for a character-level translation model, a word for a word-level model or a grid cell for an image-based model.

Such a construction of attention granularity is predetermined rather than learned.

While this kind of item-based attention has been helpful for many tasks, it can be fundamentally limited for modeling complex attention distribution that might be involved in a task.

In this paper, we propose area attention, as a general mechanism for the model to attend to a group of items in the memory that are structurally adjacent.

In area attention, each unit for attention calculation is an area that can contain one or more than one item.

Each of these areas can aggregate a varying number of items and the granularity of attention is thus learned from the data rather than predetermined.

Note that area attention subsumes item-based attention because when an area contains a single item, it is equivalent to regular attention mechanisms.

Area attention can be used along multi-head attention BID15 .

With each head using area attention, multi-head area attention allows the model to attend to multiple areas in the memory.

As we show in the experiments, the combination of both achieved the best results.

Extensive experiments with area attention indicate that area attention outperforms regular attention on a number of recent models for two popular tasks: machine translation (both token and character-level translation on WMT'14 EN-DE and EN-FR), and image captioning (trained on COCO and tested for both in-domain with COCO40 and out-of-domain captioning with Flickr 1K).

These models involve several distinct architectures, such as the canonical LSTM seq2seq with attention BID10 and the encoder-decoder Transformer BID15 BID13 .

Item-grouping has been brought up in a number of language-specific tasks.

Ranges or segments of a sentence, beyond individual tokens, have been often considered for problems such as dependency parsing or constituency parsing in natural language processing.

Recent works BID18 BID14 BID6 ) represent a sentence segment by subtracting the encoding of the first token from that of the last token in the segment, assuming the encoder captures contextual dependency of tokens.

The popular choices of the encoder are LSTM BID18 BID14 or Transformer BID6 .

In contrast, the representation of an area (or a segment) in area attention, for its basic form, is defined as the mean of all the vectors in the segment where each vector does not need to carry contextual dependency.

We calculate the mean of each area of vectors using subtraction operation over a summed area table BID17 ) that is fundamentally different from the subtraction applied in these previous works.

Lee et al. proposed a rich representation for a segment in coreference resolution tasks BID7 , where each span (segment) in a document is represented as a concatenation of the encodings of the first and last words in the span, the size of the span and an attention-weighted sum of the word embeddings within the span.

Again, this approach operates on encodings that have already captured contextual dependency between tokens, while area attention we propose does not require each item to carry contextual or dependency information.

In addition, the concept of range, segment or span that is proposed in all the above works addresses their specific task, rather than aiming for improving general attentional mechanisms.

Previous work have proposed several methods for capturing structures in attention calculation.

For example, Kim et al. used a conditional random field to directly model the dependency between items, which allows multiple "cliques" of items to be attended to at the same time BID5 .

Niculae and Blondel approached the problem, from a different angle, by using regularizers to encourage attention to be placed onto contiguous segments BID11 .

In image captioning tasks, Pedersoli et al. enabled a model to attend to object proposals on an image BID12 while You et al. applied attention to semantic concepts and visual attributes that are extracted from an image BID21 .Compared to these previous works, area attention we propose here does not require to train a special network or sub-network, or use an additional loss (regularizer) to capture structures.

It allows a model to attend to information at a varying granularity, which can be at the input layer where each item might lack contextual information, or in the latent space.

It is easy to apply area attention to existing single or multi-head attention mechanisms.

By enhancing Transformer, an attention-based architecture, BID15 with area attention, we achieved state-of-art results on a number of tasks.

An area is a group of structurally adjacent items in the memory.

When the memory consists of a sequence of items, a 1-dimensional structure, an area is a range of items that are sequentially (or temporally) adjacent and the number of items in the area can be one or multiple.

Many language-related tasks are categorized in the 1-dimensional case, e.g., machine translation or sequence prediction tasks.

In Figure 1 , the original memory is a 4-item sequence.

By combining the adjacent items in the sequence, we form area memory where each item is a combination of multiple adjacent items in the original memory.

We can limit the maximum area size to consider for a task.

In Figure 1 , the maximum area size is 3.original memory area memory query 1-item areas 2-item areas 3-item area3Figure 1: An illustration of area attention for the 1-dimensional case.

In this example, the memory is a 4-item sequence and the maximum size of an area allowed is 3.When the memory contains a grid of items, a 2-dimensional structure, an area can be any rectangular region in the grid (see Figure 2 ).

This resembles many image-related tasks, e.g., image captioning.

Again, we can limit the maximum size allowed for an area.

For a 2-dimensional area, we can set the maximum height and width for each area.

In this example, the original memory is a 3x3 grid of items and the maximum height and width allowed for each area is 2.As we can see, many areas can be generated by combining adjacent items.

For the 1-dimensional case, the number of areas that can be generated is |R| = (L ??? S)S + (S + 1)S/2 where S is the maximum size of an area and L is the length of the sequence.

For the 2-dimensional case, there are an quadratic number of areas can be generated from the original memory: DISPLAYFORM0 where L v and L h are the height and width of the memory grid and H and W are the maximum height and width allowed for a rectangular area.

To be able to attend to each area, we need to define the key and value for each area that contains one or multiple items in the original memory.

As the first step to explore area attention, we define the key of an area, ?? i , simply as the mean vector of the key of each item in the area.

DISPLAYFORM1 original memory area memory query 1x1 areas 1x2 areas 2x1 areas 2x2 areasFigure 2: An illustration of area attention for the 2-dimensional case.

In this example, the memory is a 3x3 grid and the dimension allowed for an area is 2x2.where |r i | is the size of the area r i .

For the value of an area, we simply define it as the the sum of all the value vectors in the area.

DISPLAYFORM2 With the keys and values defined, we can use the standard way for calculating attention as discussed in Equation 1 and Equation 2.

Note that this basic form of area attention (Eq.3 and Eq.4) is parameterfree-it does not introduce any parameters to be learned.

Alternatively, we can derive a richer representation of each area by using features other than the mean of the key vectors of the area.

For example, we can consider the standard deviation of the key vectors within each area.

DISPLAYFORM0 We can also consider the height and width of each area, h i ,1 ??? h i ??? H and w i ,1 ??? w i ??? W , as the features of the area.

To combine these features, we use a multi-layer perceptron.

To do so, we treat h i and w i as discrete values and project them onto a vector space using embedding (see Equation 6 and 7).

DISPLAYFORM1 DISPLAYFORM2 where 1(h i ) and 1(w i ) are the one-hot encoding of h i and w i , and E h ??? R H??S and E w ??? R W ??S are the embedding matrices.

S is the depth of the embedding.

We concatenate them to form the representation of the shape of an area.

DISPLAYFORM3 We then combine them using a single-layer perceptron followed by a linear transformation (see Equation 9 ).

DISPLAYFORM4 where ?? is a nonlinear transformation such as ReLU, and DISPLAYFORM5 DISPLAYFORM6 A is the maximum size of an area, which is S in the one dimensional case and W H in the 2-dimensional case.

This is computationally expensive in comparison to the attention computed on the original memory, which is O(|M |).

To address the issue, we use summed area table, an optimization technique that has been used in computer vision for computing features on image areas BID17 .

It allows constant time to calculate a summation-based feature in each rectangular area, which allows us to bring down the time complexity to O(|M |A)-We will report on the actual time cost in our experimental section.

Summed area table is based on a pre-computed integral image, I, which can be computed in a single pass of the memory (see Equation 10).

Here let us focus on the area value calculation for a 2-dimensional memory because a 1-dimensional memory is just a special case with the height of the memory grid as 1.

DISPLAYFORM7 where x and y are the coordinates of the item in the memory.

With the integral image, we can calculate the key and value of each area in constant time.

The sum of all the vectors in a rectangular area can be easily computed as the following FORMULA0 ).

DISPLAYFORM8 where v x1,y1,x2,y2 is the value for the area located with the top-left corner at (x 1 , y 1 ) and the bottomright corner at (x 2 , y 2 ).

By dividing v x1,y1,x2,y2 with the size of the area, we can easily compute ?? x1,y1,x2,y2 .

Based on the summed area table, ?? DISPLAYFORM9 The core component for computing these quantities is to be able to quickly compute the sum of vectors in each area after we obtain the integral image DISPLAYFORM10 (H???h)??(W ???w) ; Fill tensor with value h for the height of each area; Vector mean ??, standard deviation ?? and sum U as well as height S h and width S w of each area.

1 Acquire U , S h and S w using Algorithm 1 with input G; 2 Acquire U using Algorithm 1 with input G G where is for element-wise multiplication; 3 ?? ??? U S where is for element-wise division; 4 ?? ??? U S ; 5 ?? ??? ?? ??? ?? ?? ; 6 return ??, ??, U , as well as S h and S w .

DISPLAYFORM11

We experimented with area attention on two important tasks: neural machine translation (including both token and character-level translation) and image captioning, where attention mechanisms have been a common component in model architectures for these tasks.

The architectures we investigate involves several popular encoder and decoder choices, such as LSTM BID4 and Transformer BID15 .

The attention mechansims in these tasks include both self attention and encoder-decoder attention.

Transformer has recently BID15 established the state of art performance on WMT 2014 English-to-German and English-to-French tasks, while LSTM with encoder-decoder attention has been a popular choice for neural machine translation tasks.

We use the same dataset as the one used in BID15 in which the WMT 2014 English-German dataset contains about 4.5 million English-German sentence pairs, and the English-French dataset has about 36 million English-French sentence pairs BID19 .

A token is either a byte pair BID1 or a word piece BID19 as in the original Transformer experiments.

Transformer heavily uses attentional mechanisms, including both self-attention in the encoder and the decoder, and attention from the decoder to the encoder.

We vary the configuration of Transformer to investigate how area attention impacts the model.

In particular, we investigated the following variations of Transformer: Tiny (#hidden layers=2, hidden size=128, filter size=512, #attention heads=4), Small (#hidden layers=2, hidden size=256, filter size=1024, #attention heads=4), Base (#hidden layers=6, hidden size=512, filter size=2048, #attention heads=8) and Big (#hidden layers=6, hidden size=1024, filter size=4096, #attention heads=16).During training, sentence pairs were batched together based on their approximate sequence lengths.

All the model variations except Big uses a training batch contained a set of sentence pairs that amount to approximately 32,000 source and target tokens and were trained on one machine with 8 NVIDIA P100 GPUs for a total of 250,000 steps.

Given the batch size, each training step for the Transformer Base model, on 8 NVIDIA P100 GPUs, took 0.4 seconds for Regular Attention, 0.5 seconds for the basic form of Area Attention (Eq.3 and Eq.4), 0.8 seconds for Area Attention using multiple features (Eq.9 and Eq.4).For Big, due to the memory constraint, we had to use a smaller batch size that amounts to roughly 16,000 source and target tokens and trained the model for 600,000 steps.

Each training step took 0.5 seconds for Regular Attention, 0.6 seconds for the basic form of Area Attention (Eq.3 and 4), 1.0 seconds for Area Attention using multiple features (Eq.9 and 4).

Similar to previous work, we used the Adam optimizer with a varying learning rate over the course of training-see BID15 for details.

We applied area attention to each of the Transformer variation, with the maximum area size of 5 to both encoder and decoder self-attention, and the encoder-decoder attention in the first two layers.

We found area attention consistently improved Transformer on all the model variations (see TAB3 For EN-FR, the performance of Transformer Big with regular attention-a baseline-does not match what was reported in the Transformer paper BID15 , largely due to a different batch size and the different number of training steps used, although area attention still outperformed the baseline consistently.

On the other hand, area attention with Transformer Big achieved BLEU 29.68 on EN-DE that improved upon the state-of-art result of 28.4 reported in BID15 ) with a significant margin.

We used a 2-layer LSTM for both encoder and decoder.

The encoder-decoder attention is based on multiplicative attention where the alignment of a query and a memory key is computed as their dot product BID10 .

We vary the size of LSTM and the number of attention heads to investigate how area attention can improve LSTM with varying capacity on translation tasks.

The purpose is to observe the impact of area attention on each LSTM configuration, rather than for a comparison with Transformer.

Because LSTM requires sequential computation along a sequence, it trains rather slow compared to Transformer.

To improve GPU utilization we increased data parallelism by using a much larger batch size than training Transformer.

We trained each LSTM model on one machine with 8 NVIDIA P100.

For a model has 256 or 512 LSTM cells, we trained it for 50,000 steps using a batch size that amounts to approximately 160,000 source and target tokens.

When the number of cells is 1024, we had to use a smaller batch size with roughly 128,000 tokens, due to the memory constraint, and trained the model for 625,000 steps.

In these experiments, we used the maximum area size of 2 and the attention is computed from the output of the decoder's top layer to that of the encoder.

Similar to what we observed with Transformer, area attention consistently improves LSTM architectures in all the conditions (see TAB4 ).

Compared to token-level translation, character-level translation requires the model to address significantly longer sequences, which are a more difficult task and often less studied.

We speculate that the ability to combine adjacent characters, as enabled by area attention, is likely useful to improve a regular attentional mechanisms.

Likewise, we experimented with both Transformer and LSTM-based architectures for this task.

We here used the same dataset, and the batching and training strategies as the ones used in the token-level translation experiments.

Transformer has not been used for character-level translation tasks.

We found area attention consistently improved Transformer across all the model configurations.

The best result we found in the literature is BLEU = 22.62 reported by BID19 .

We achieved BLEU = 26.65 for the English-to-German character-level translation task and BLEU = 34.81 on the English-to-French character-level translation task.

Note that these accuracy gains are based on the basic form of area attention (see Eq.3 and Eq.4), which does not add any additional trainable parameters to the model.

Similarly, we tested LSTM architectures on the character-level translation tasks.

We found area attention outperformed the baselines in most conditions (see TAB6 ).

The improvement seems more substantial when a model is relatively small.

Image captioning is the task to generate natural language description of an image that reflects the visual content of an image.

This task has been addressed previously using a deep architecture that features an image encoder and a language decoder BID20 BID13 .

The image encoder typically employs a convolutional net such as ResNet BID3 to embed the images and then uses a recurrent net such as LSTM or Transformer BID13 to encode the image based on these embeddings.

For the decoder, either LSTM BID20 or Transformer BID13 has been used for generating natural language descriptions.

In many of these designs, attention mechanisms have been an important component that allows the decoder to selectively focus on a specific part of the image at each step of decoding, which often leads to better captioning quality.

In this experiment, we follow a champion condition in the experimental setup of BID13 that achieved state-of-the-art results.

It uses a pre-trained Inception-ResNet to generate 8 ?? 8 image embeddings, a 6-layer Transformer for image encoding and a 6-layer Transformer for decoding.

The dimension of Transformer is 512 and the number of heads is 8.

We intend to investigate how area attention improves the captioning accuracy, particularly regarding self-attention and encoder-decoder attention computed off the image, which resembles a 2-dimensional case for using area attention.

We also vary the maximum area size allowed to examine the impact.

Similar to BID13 , we trained each model based on the training & development sets provided by the COCO dataset BID9 , which as 82K images for training and 40K for validation.

Each of these images have at least 5 groudtruth captions.

The training was conducted on a distributed learning infrastructure BID2 with 10 GPU cores where updates are applied asynchronously across multiple replicas.

We then tested each model on the COCO40 BID9 and the Flickr 1K BID22 test sets.

Flickr 1K is out-of-domain for the trained model.

For each experiment, we report CIDEr BID16 and ROUGE-L BID8 metrics.

For both metrics, higher number means better captioning accuracy-the closer distances between the predicted and the groundtruth captions.

Similar to the previous work BID13 , we report the numerical values returned by the COCO online evaluation server 3 for the COCO C40 test set (see TAB7 ), In the benchmark model, a regular multi-head attention is used.

We then experimented with several variations by adding area attention with different maximum area sizes to the first 2 layers of the image encoder self-attention and encoder-decoder (caption-to-image) attention, which both are a 2-dimensional area attention case.

2 ?? 2 stands for the maximum area size 2 by 2 and 3 ?? 3 for 3 by 3.

For the 2 ?? 2 case, an area can be 1 by 1, 2 by 1, 1 by 2, and 2 by 2 as illustrated in Figure 2 .

3 ?? 3 allows more area shapes.

We found models with area attention outperformed the benchmark on both CIDEr and ROUGE-L metrics with a large margin.

The models with 2 ?? 2 Eq.3 and 3 ?? 3 Eq.3 are parameter free-they do not use any additional parameters beyond the benchmark model.

3 ?? 3 achieved the best results overall.

3 ?? 3 Eq. 9 adds a small fraction of the number of parameters to the benchmark model and did not seem to improve on the parameter-free version of area attention, although it still outperformed the benchmark.

In this paper, we present a novel attentional mechanism by allowing the model to attend to areas as a whole.

An area contains one or a group of items in the memory to be attended.

The items in the area are either spatially adjacent when the memory has 2-dimensional structure, such as images, or temporally adjacent for 1-dimensional memory, such as natural language sentences.

Importantly, the size of an area, i.e., the number of items in an area or the level of aggregation, can vary depending on the learned coherence of the adjacent items, which gives the model the ability to attend to information at varying granularity.

Area attention contrasts with the existing attentional mechanisms that are itembased.

We evaluated area attention on two tasks: neural machine translation and image captioning, based on model architectures such as Transformer and LSTM.

On both tasks, we obtained new state-of-the-art results using area attention.

@highlight

The paper presents a novel approach for attentional mechanisms that can benefit a range of tasks such as machine translation and image captioning.

@highlight

This paper extends the current attention models from word level to the combination of adjacent words, by applying the models to items made from merged adjacent words.