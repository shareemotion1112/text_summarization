Natural language processing (NLP) models often require a massive number of parameters for word embeddings, resulting in a large storage or memory footprint.

Deploying neural NLP models to mobile devices requires compressing the word embeddings without any significant sacrifices in performance.

For this purpose, we propose to construct the embeddings with few basis vectors.

For each word, the composition of basis vectors is determined by a hash code.

To maximize the compression rate, we adopt the multi-codebook quantization approach instead of binary coding scheme.

Each code is composed of multiple discrete numbers, such as (3, 2, 1, 8), where the value of each component is limited to a fixed range.

We propose to directly learn the discrete codes in an end-to-end neural network by applying the Gumbel-softmax trick.

Experiments show the compression rate achieves 98% in a sentiment analysis task and 94% ~ 99% in machine translation tasks without performance loss.

In both tasks, the proposed method can improve the model performance by slightly lowering the compression rate.

Compared to other approaches such as character-level segmentation, the proposed method is language-independent and does not require modifications to the network architecture.

Word embeddings play an important role in neural-based natural language processing (NLP) models.

Neural word embeddings encapsulate the linguistic information of words in continuous vectors.

However, as each word is assigned an independent embedding vector, the number of parameters in the embedding matrix can be huge.

For example, when each embedding has 500 dimensions, the network has to hold 100M embedding parameters to represent 200K words.

In practice, for a simple sentiment analysis model, the word embedding parameters account for 98.8% of the total parameters.

As only a small portion of the word embeddings is selected in the forward pass, the giant embedding matrix usually does not cause a speed issue.

However, the massive number of parameters in the neural network results in a large storage or memory footprint.

When other components of the neural network are also large, the model may fail to fit into GPU memory during training.

Moreover, as the demand for low-latency neural computation for mobile platforms increases, some neural-based models are expected to run on mobile devices.

Thus, it is becoming more important to compress the size of NLP models for deployment to devices with limited memory or storage capacity.

In this study, we attempt to reduce the number of parameters used in word embeddings without hurting the model performance.

Neural networks are known for the significant redundancy in the connections BID8 .

In this work, we further hypothesize that learning independent embeddings causes more redundancy in the embedding vectors, as the inter-similarity among words is ignored.

Some words are very similar regarding the semantics.

For example, "dog" and "dogs" have almost the same meaning, except one is plural.

To efficiently represent these two words, it is desirable to share information between the two embeddings.

However, a small portion in both vectors still has to be trained independently to capture the syntactic difference.

Following the intuition of creating partially shared embeddings, instead of assigning each word a unique ID, we represent each word w with a code C w = (C an integer number in [1, K] .

Ideally, similar words should have similar codes.

For example, we may desire C dog = (3, 2, 4, 1) and C dogs = (3, 2, 4, 2).

Once we have obtained such compact codes for all words in the vocabulary, we use embedding vectors to represent the codes rather than the unique words.

More specifically, we create M codebooks E 1 , E 2 , ..., E M , each containing K codeword vectors.

The embedding of a word is computed by summing up the codewords corresponding to all the components in the code as DISPLAYFORM0 where DISPLAYFORM1 w -th codeword in the codebook E i .

In this way, the number of vectors in the embedding matrix will be M × K, which is usually much smaller than the vocabulary size.

FIG0 gives an intuitive comparison between the compositional approach and the conventional approach (assigning unique IDs).

The codes of all the words can be stored in an integer matrix, denoted by C. Thus, the storage footprint of the embedding layer now depends on the total size of the combined codebook E and the code matrix C.Although the number of embedding vectors can be greatly reduced by using such coding approach, we want to prevent any serious degradation in performance compared to the models using normal embeddings.

In other words, given a set of baseline word embeddingsẼ(w), we wish to find a set of codesĈ and combined codebookÊ that can produce the embeddings with the same effectiveness asẼ(w).

A safe and straight-forward way is to minimize the squared distance between the baseline embeddings and the composed embeddings as DISPLAYFORM2 where |V | is the vocabulary size.

The baseline embeddings can be a set of pre-trained vectors such as word2vec BID29 or GloVe BID34 embeddings.

In Eq. 3, the baseline embedding matrixẼ is approximated by M codewords selected from M codebooks.

The selection of codewords is controlled by the code C w .

Such problem of learning compact codes with multiple codebooks is formalized and discussed in the research field of compressionbased source coding, known as product quantization BID16 and additive quantization BID1 BID28 .

Previous works learn compositional codes so as to enable an efficient similarity search of vectors.

In this work, we utilize such codes for a different purpose, that is, constructing word embeddings with drastically fewer parameters.

Due to the discreteness in the hash codes, it is usually difficult to directly optimize the objective function in Eq. 3.

In this paper, we propose a simple and straight-forward method to learn the codes in an end-to-end neural network.

We utilize the Gumbel-softmax trick BID27 BID15 to find the best discrete codes that minimize the loss.

Besides the simplicity, this approach also allows one to use any arbitrary differentiable loss function, such as cosine similarity.

The contribution of this work can be summarized as follows:• We propose to utilize the compositional coding approach for constructing the word embeddings with significantly fewer parameters.

In the experiments, we show that over 98% of the embedding parameters can be eliminated in sentiment analysis task without affecting performance.

In machine translation tasks, the loss-free compression rate reaches 94% ∼ 99%.• We propose a direct learning approach for the codes in an end-to-end neural network, with a Gumbel-softmax layer to encourage the discreteness.• The neural network for learning codes will be packaged into a tool 1 .

With the learned codes and basis vectors, the computation graph for composing embeddings is fairly easy to implement, and does not require modifications to other parts in the neural network.

Existing works for compressing neural networks include low-precision computation BID38 BID14 BID7 BID0 , quantization BID6 BID10 BID44 , network pruning BID22 BID11 BID9 BID40 and knowledge distillation BID13 .

Network quantization such as HashedNet BID6 forces the weight matrix to have few real weights, with a hash function to determine the weight assignment.

To capture the non-uniform nature of the networks, DeepCompression BID10 groups weight values into clusters based on pre-trained weight matrices.

The weight assignment for each value is stored in the form of Huffman codes.

However, as the embedding matrix is tremendously big, the number of hash codes a model need to maintain is still large even with Huffman coding.

Network pruning works in a different way that makes a network sparse.

Iterative pruning BID9 prunes a weight value if its absolute value is smaller than a threshold.

The remaining network weights are retrained after pruning.

Some recent works BID36 BID43 ) also apply iterative pruning to prune 80% of the connections for neural machine translation models.

In this paper, we compare the proposed method with iterative pruning.

The problem of learning compact codes considered in this paper is closely related to learning to hash BID39 BID21 BID25 , which aims to learn the hash codes for vectors to facilitate the approximate nearest neighbor search.

Initiated by product quantization BID16 , subsequent works such as additive quantization BID1 explore the use of multiple codebooks for source coding, resulting in compositional codes.

We also adopt the coding scheme of additive quantization for its storage efficiency.

Previous works mainly focus on performing efficient similarity search of image descriptors.

In this work, we put more focus on reducing the codebook sizes and learning efficient codes to avoid performance loss.

BID17 utilizes an improved version of product quantization to compress text classification models.

However, to match the baseline performance, much longer hash codes are required by product quantization.

This will be detailed in Section 5.2.

Concurrent to this work, also explores the similar idea and obtained positive results in language modeling tasks.

Also, BID35 tried to reduce dimension of embeddings using PCA.To learn the codebooks and code assignment, additive quantization alternatively optimizes the codebooks and the discrete codes.

The learning of code assignment is performed by Beam Search algorithm when the codebooks are fixed.

In this work, we propose a straight-forward method to directly learn the code assignment and codebooks simutaneously in an end-to-end neural network.

Some recent works BID41 BID24 BID42 in learning to hash also utilize neural networks to produce binary codes by applying binary constrains (e.g., sigmoid function).

In this work, we encourage the discreteness with the Gumbel-Softmax trick for producing compositional codes.

As an alternative to our approach, one can also reduce the number of unique word types by forcing a character-level segmentation.

BID18 proposed a character-based neural language model, which applies a convolutional layer after the character embeddings.

BID3 propose to use char-gram as input features, which are further hashed to save space.

Generally, using characterlevel inputs requires modifications to the model architecture.

Moreover, some Asian languages such as Japanese and Chinese retain a large vocabulary at the character level, which makes the character-based approach difficult to be applied.

In contrast, our approach does not suffer from these limitations.

In this section, we formally describe the compositional coding approach and analyze its merits for compressing word embeddings.

The coding approach follows the scheme in additive quantization BID1 .

We represent each word w with a compact code C w that is composed of M components such that DISPLAYFORM0 , which also indicates that M log 2 K bits are required to store each code.

For convenience, K is selected to be a number of a multiple of 2, so that the codes can be efficiently stored.

If we restrict each component C i w to values of 0 or 1, the code for each word C w will be a binary code.

In this case, the code learning problem is equivalent to a matrix factorization problem with binary components.

Forcing the compact codes to be binary numbers can be beneficial, as the learning problem is usually easier to solve in the binary case, and some existing optimization algorithms in learning to hash can be reused.

However, the compositional coding approach produces shorter codes and is thus more storage efficient.

As the number of basis vectors is M × K regardless of the vocabulary size, the only uncertain factor contributing to the model size is the size of the hash codes, which is proportional to the vocabulary size.

Therefore, maintaining short codes is cruicial in our work.

Suppose we wish the model to have a set of N basis vectors.

Then in the binary case, each code will have N/2 bits.

For the compositional coding approach, if we can find a M × K decomposition such that M × K = N , then each code will have M log 2 K bits.

For example, a binary code will have a length of 256 bits to support 512 basis vectors.

In contrast, a 32 × 16 compositional coding scheme will produce codes of only 128 bits.

To support N basis vectors, a binary code will have N/2 bits and the embedding computation is a summation over N/2 vectors.

For the compositional approach with M codebooks and K codewords in each codebook, each code has M log 2 K bits, and the computation is a summation over M vectors.

A comparison of different coding approaches is summarized in TAB0 .

We also report the number of basis vectors required to compute an embedding as a measure of computational cost.

For the conventional approach, the number of vectors is identical to the vocabulary size and the computation is basically a single indexing operation.

In the case of binary codes, the computation for constructing an embedding involves a summation over N/2 basis vectors.

For the compositional approach, the number of vectors required to construct an embedding vector is M .

Both the binary and compositional approaches have significantly fewer vectors in the embedding matrix.

The compositional coding approach provides a better balance with shorter codes and lower computational cost.

LetẼ ∈ R |V |×H be the original embedding matrix, where each embedding vector has H dimensions.

By using the reconstruction loss as the objective function in Eq. 3, we are actually finding DISPLAYFORM0 Therefore, the problem of learning discrete codes C w can be converted to a problem of finding a set of optimal one-hot vectors d DISPLAYFORM1 where G k is a noise term that is sampled from the Gumbel distribution − log(− log(Uniform[0, 1])), whereas τ is the temperature of the softmax.

In our model, the vector α i w is computed by a simple neural network with a single hidden layer as DISPLAYFORM2 DISPLAYFORM3 In our experiments, the hidden layer h w always has a size of M K/2.

We found that a fixed temperature of τ = 1 just works well.

The Gumbel-softmax trick is applied to α i w to obtain d i w .

Then, the model reconstructs the embedding E(C w ) with Eq. 5 and computes the reconstruction loss with Eq. 3.

The model architecture of the end-to-end neural network is illustrated in FIG1 , which is effectively an auto-encoder with a Gumbel-softmax middle layer.

The whole neural network for coding learning has five parameters (θ, b, θ , b , A).Once the coding learning model is trained, the code C w for each word can be easily obtained by applying argmax to the one-hot vectors d For general NLP tasks, one can learn the compositional codes from publicly available word vectors such as GloVe vectors.

However, for some tasks such as machine translation, the word embeddings are usually jointly learned with other parts of the neural network.

For such tasks, one has to first train a normal model to obtain the baseline embeddings.

Then, based on the trained embedding matrix, one can learn a set of task-specific codes.

As the reconstructed embeddings E(C w ) are not identical to the original embeddingsẼ(w), the model parameters other than the embedding matrix have to be retrained again.

The code learning model cannot be jointly trained with the machine translation model as it takes far more iterations for the coding layer to converge to one-hot vectors.

In our experiments, we focus on evaluating the maximum loss-free compression rate of word embeddings on two typical NLP tasks: sentiment analysis and machine translation.

We compare the model performance and the size of embedding layer with the baseline model and the iterative pruning method BID9 .

Please note that the sizes of other parts in the neural networks are not included in our results.

For dense matrices, we report the size of dumped numpy arrays.

For the sparse matrices, we report the size of dumped compressed sparse column matrices (csc matrix) in scipy.

All float numbers take 32 bits storage.

We enable the "compressed" option when dumping the matrices, without this option, the file size is about 1.1 times bigger.

To learn efficient compact codes for each word, our proposed method requires a set of baseline embedding vectors.

For the sentiment analysis task, we learn the codes based on the publicly available GloVe vectors.

For the machine translation task, we first train a normal neural machine translation (NMT) model to obtain task-specific word embeddings.

Then we learn the codes using the pre-trained embeddings.

We train the end-to-end network described in Section 4 to learn the codes automatically.

In each iteration, a small batch of the embeddings is sampled uniformly from the baseline embedding matrix.

The network parameters are optimized to minimize the reconstruction loss of the sampled embeddings.

In our experiments, the batch size is set to 128.

We use Adam optimizer BID19 ) with a fixed learning rate of 0.0001.

The training is run for 200K iterations.

Every 1,000 iterations, we examine the loss on a fixed validation set and save the parameters if the loss decreases.

We evenly distribute the model training to 4 GPUs using the nccl package, so that one round of code learning takes around 15 minutes to complete.

Dataset:

For sentiment analysis, we use a standard separation of IMDB movie review dataset BID26 , which contains 25k reviews for training and 25K reviews for testing purpose.

We lowercase and tokenize all texts with the nltk package.

We choose the 300-dimensional uncased GloVe word vectors (trained on 42B tokens of Common Crawl data) as our baseline embeddings.

The vocabulary for the model training contains all words appears both in the IMDB dataset and the GloVe vocabulary, which results in around 75K words.

We truncate the texts of reviews to assure they are not longer than 400 words.

Model architecture: Both the baseline model and the compressed models have the same computational graph except the embedding layer.

The model is composed of a single LSTM layer with 150 hidden units and a softmax layer for predicting the binary label.

For the baseline model, the embedding layer contains a large 75K × 300 embedding matrix initialized by GloVe embeddings.

For the compressed models based on the compositional coding, the embedding layer maintains a matrix of basis vectors.

Suppose we use a 32 × 16 coding scheme, the basis matrix will then have a shape of 512 × 300, which is initialized by the concatenated weight matrices [A 1 ; A 2 ; ...; A M ] in the code learning model.

The embedding parameters for both models remain fixed during the training.

For the models with network pruning, the sparse embedding matrix is finetuned during training.

Training details: The models are trained with Adam optimizer for 15 epochs with a fixed learning rate of 0.0001.

At the end of each epoch, we evaluate the loss on a small validation set.

The parameters with lowest validation loss are saved.

Results: For different settings of the number of components M and the number of codewords K, we train the code learning network.

The average reconstruction loss on a fixed validation set is summarized in the left of TAB2 .

For reference, we also report the total size (MB) of the embedding layer in the right table, which includes the sizes of the basis matrix and the hash table.

We can see that increasing either M or K can effectively decrease the reconstruction loss.

However, setting M to a large number will result in longer hash codes, thus significantly increase the size of the embedding layer.

Hence, it is important to choose correct numbers for M and K to balance the performance and model size.

We also show the results using normalized product quantization (NPQ) BID17 .

We quantize the filtered GloVe embeddings with the codes provided by the authors, and train the models based on the quantized embeddings.

To make the results comparable, we report the codebook size in numpy format.

For our proposed methods, the maximum loss-free compression rate is achieved by a 16 × 32 coding scheme.

In this case, the total size of the embedding layer is 1.23 MB, which is equivalent to a compression rate of 98.4%.

We also found the classification accuracy can be substantially improved with a slightly lower compression rate.

The improved model performance may be a byproduct of the strong regularization.

Dataset:

For machine translation tasks, we experiment on IWSLT 2014 German-to-English translation task BID4 and ASPEC English-to-Japanese translation task BID31 .

The IWSLT14 training data contains 178K sentence pairs, which is a small dataset for machine translation.

We utilize moses toolkit BID20 to tokenize and lowercase both sides of the texts.

Then we concatenate all five TED/TEDx development and test corpus to form a test set containing 6750 sentence pairs.

We apply byte-pair encoding BID37 to transform the texts to subword level so that the vocabulary has a size of 20K for each language.

For evaluation, we report tokenized BLEU using "multi-bleu.perl".The ASPEC dataset contains 300M bilingual pairs in the training data with the automatically estimated quality scores provided for each pair.

We only use the first 150M pairs for training the models.

The English texts are tokenized by moses toolkit whereas the Japanese texts are tokenized by kytea BID33 .

The vocabulary size for each language is reduced to 40K using byte-pair encoding.

The evaluation is performed using a standard kytea-based post-processing script for this dataset.

Model architecture: In our preliminary experiments, we found a 32 × 16 coding works well for a vanilla NMT model.

As it is more meaningful to test on a high-performance model, we applied several techniques to improve the performance.

The model has a standard bi-directional encoder composed of two LSTM layers similar to BID2 .

The decoder contains two LSTM layers.

Residual connection BID12 with a scaling factor of 1/2 is applied to the two decoder states to compute the outputs.

All LSTMs and embeddings have 256 hidden units in the IWSLT14 task and 1000 hidden units in ASPEC task.

The decoder states are firstly linearly transformed to 600-dimensional vectors before computing the final softmax.

Dropout with a rate of 0.2 is applied everywhere except the recurrent computation.

We apply Key-Value Attention BID30 to the first decoder, where the query is the sum of the feedback embedding and the previous decoder state and the keys are computed by linear transformation of encoder states.

Training details: All models are trained by Nesterov's accelerated gradient BID32 with an initial learning rate of 0.25.

We evaluate the smoothed BLEU BID23 ) on a validation set composed of 50 batches every 7,000 iterations.

The learning rate is reduced by a factor of 10 if no improvement is observed in 3 validation runs.

The training ends after the learning rate is reduced three times.

Similar to the code learning, the training is distributed to 4 GPUs, each GPU computes a mini-batch of 16 samples.

We firstly train a baseline NMT model to obtain the task-specific embeddings for all in-vocabulary words in both languages.

Then based on these baseline embeddings, we obtain the hash codes and basis vectors by training the code learning model.

Finally, the NMT models using compositional coding are retrained by plugging in the reconstructed embeddings.

Note that the embedding layer is fixed in this phase, other parameters are retrained from random initial values.

The experimental results are summarized in TAB5 .

All translations are decoded by the beam search with a beam size of 5.

The performance of iterative pruning varies between tasks.

The loss-free compression rate reaches 92% on ASPEC dataset by pruning 90% of the connections.

However, with the same pruning ratio, a modest performance loss is observed in IWSLT14 dataset.

For the models using compositional coding, the loss-free compression rate is 94% for the IWSLT14 dataset and 99% for the ASPEC dataset.

Similar to the sentiment analysis task, a significant performance improvement can be observed by slightly lowering the compression rate.

Note that the sizes of NMT models are still quite large due to the big softmax layer and the recurrent layers, which are not reported in the table.

Please refer to existing works such as BID43 TAB6 , we show some examples of learned codes based on the 300-dimensional uncased GloVe embeddings used in the sentiment analysis task.

We can see that the model learned to assign similar codes to the words with similar meanings.

Such a code-sharing mechanism can significantly reduce the redundancy of the word embeddings, thus helping to achieve a high compression rate.

Besides the performance, we also care about the storage efficiency of the codes.

In the ideal situation, all codewords shall be fully utilized to convey a fraction of meaning.

However, as the codes are automatically learned, it is possible that some codewords are abandoned during the training.

In extreme cases, some "dead" codewords can be used by none of the words.

To analyze the code efficiency, we count the number of words that contain a specific subcode in each component.

FIG5 gives a visualization of the code balance for three coding schemes.

Each column shows the counts of the subcodes of a specific component.

In our experiments, when using a 8 × 8 coding scheme, we found 31% of the words have a subcode "0" for the first component, while the subcode "1" is only used by 5% of the words.

The assignment of codes is more balanced for larger coding schemes.

In any coding scheme, even the most unpopular codeword is used by about 1000 words.

This result indicates that the code learning model is capable of assigning codes efficiently without wasting a codeword.

The results show that any codeword is assigned to more than 1000 words without wasting.

In this work, we propose a novel method for reducing the number of parameters required in word embeddings.

Instead of assigning each unique word an embedding vector, we compose the embedding vectors using a small set of basis vectors.

The selection of basis vectors is governed by the hash code of each word.

We apply the compositional coding approach to maximize the storage efficiency.

The proposed method works by eliminating the redundancy inherent in representing similar words with independent embeddings.

In our work, we propose a simple way to directly learn the discrete codes in a neural network with Gumbel-softmax trick.

The results show that the size of the embedding layer was reduced by 98% in IMDB sentiment analysis task and 94% ∼ 99% in machine translation tasks without affecting the performance.

Our approach achieves a high loss-free compression rate by considering the semantic inter-similarity among different words.

In qualitative analysis, we found the learned codes of similar words are very close in Hamming space.

As our approach maintains a dense basis matrix, it has the potential to be further compressed by applying pruning techniques to the dense matrix.

The advantage of compositional coding approach will be more significant if the size of embedding layer is dominated by the hash codes.

A APPENDIX: SHARED CODESIn both tasks, when we use a small code decomposition, we found some hash codes are assigned to multiple words.

Table 6 lists some samples of shared codes with their corresponding words from the sentiment analysis task.

This phenomenon does not cause a problem in either task, as the words only have shared codes when they have almost the same sentiments or target translations.shared code words 4 7 7 0 4 7 1 1 homes cruises motel hotel resorts mall vacations hotels 6 6 7 1 4 0 2 0 basketball softball nfl nascar baseball defensive ncaa tackle nba 3 7 3 2 4 3 3 0 unfortunately hardly obviously enough supposed seem totally ... 4 6 7 0 4 7 5 0 toronto oakland phoenix miami sacramento denver minneapolis ...

7 7 6 6 7 3 0 0 yo ya dig lol dat lil bye Table 6 : Examples of words sharing same codes when using a 8 × 8 code decomposition B APPENDIX: SEMANTICS OF CODES In order to see whether each component captures semantic meaning.

we learned a set of codes using a 3 x 256 coding scheme, this will force the model to decompose each embedding into 3 vectors.

In order to maximize the compression rate, the model must make these 3 vectors as independent as possible.

Table 7 : Some code examples using a 3 × 256 coding scheme.

As we can see from Table 7 , we can transform "man/king" to "woman/queen" by change the subcode "210" in the first component to "232".

So we can think "210" must be a "male" code, and "232" must be a "female" code.

Such phenomenon can also be observed in other words such as city names.

<|TLDR|>

@highlight

Compressing the word embeddings over 94% without hurting the performance.