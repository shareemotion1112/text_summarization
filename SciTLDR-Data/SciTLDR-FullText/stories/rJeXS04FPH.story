For sequence models with large word-level vocabularies, a majority of network parameters lie in the input and output layers.

In this work, we describe a new method, DeFINE, for learning deep word-level representations efficiently.

Our architecture uses a hierarchical structure with novel skip-connections which allows for the use of low dimensional input and output layers, reducing total parameters and training time while delivering similar or better performance versus existing methods.

DeFINE can be incorporated easily in new or existing sequence models.

Compared to state-of-the-art methods including adaptive input representations, this technique results in a 6% to 20% drop in perplexity.

On WikiText-103, DeFINE reduces total parameters of Transformer-XL by half with minimal impact on performance.

On the Penn Treebank, DeFINE improves AWD-LSTM by 4 points with a 17% reduction in parameters,  achieving comparable performance to state-of-the-art methods with fewer parameters.

For machine translation, DeFINE improves a Transformer model by 2% while simultaneously reducing total parameters by 26%

Neural models for NLP tasks, such as language modeling and machine translation, require large vocabularies for generality (Chelba et al., 2013; Bahdanau et al., 2015; Luong et al., 2015; Merity et al., 2017) .

These models often employ a similar architecture: words, represented as one-hot vectors, are mapped to a dense continuous space; they are then processed by a context model; finally, the contextualized representations are mapped back to a vocabulary-sized vector for computing next-token probabilities.

A language modeling example is shown in Figure 1a .

The mapping in the first and last steps often uses a shared learned look-up table, referred to as an embedding layer, which takes every word in the vocabulary to a fixed m-dimensional vector.

One drawback of this approach is that the number of parameters in the embedding layer increases as the vocabulary size grows, limiting us to small values of m over large vocabularies.

Researchers have sought to improve the efficiency of the embedding layer by assigning lower frequency words smaller dimensional vectors, however, significant parameter reductions come at the cost of performance (Morin & Bengio, 2005; Grave et al., 2017a; Baevski & Auli, 2019) .

In all these approaches, word embedding is approximated with a linear function from words to vectors.

In this work, we introduce DEep Factorized INput word Embeddings (DeFINE) for neural sequence modeling.

DeFINE approximates the complicated word embedding function with far fewer parameters compared to standard methods.

DeFINE allows for lower-dimensional input and output mappings in sequence models, reducing their computational burden without reducing performance.

The representations produced by DeFINE are more powerful than those of other factorization techniques and even standard embedding layers.

To accomplish this, DeFINE leverages a hierarchical group transformation (HGT) that learns deep representations efficiently and effectively.

HGT connects different subsets of the input using sparse and dense connections.

To improve the flow of information, DeFINE introduces a new skip-connection that establishes a direct link with the input layer at every level of its hierarchy, allowing gradient to flow back directly to the input via multiple paths.

DeFINE replaces standard word embedding layers, leaving the rest of the model untouched, and so it can be used with a wide variety of sequence modeling architectures.

Figure 1 shows how we incorporate DeFINE with Transformer-XL (Dai et al., 2019) , a state-of-the-art Transformer-based language model and the resulting reduction in total parameters.

Figure 1: With DeFINE, Transformer-XL learns input (embedding) and output (classification) representations in low n-dimensional space rather than high m-dimensional space, thus reducing parameters significantly while having a minimal impact on the performance.

Our experiments show that both LSTM-and Transformer-based sequence models benefit from the use of DeFINE.

On the Wikitext-103 dataset, an LSTM-based language model with DeFINE provides a 9 point improvement over a full capacity model while using half as many parameters.

When combined with adaptive input (Baevski & Auli, 2019) and output (Grave et al., 2017a) representations, DeFINE improves the performance by about 3 points across LSTM-based (see Table 1a ) and Transformer-XL-based (see Table 2 ) language models with a minimal increase in training parameters.

Computation time at inference is unaffected.

1 Incorporating DeFINE into the popular AWD-LSTM language model (Merity et al., 2018b) without finetuning results in a test perplexity of 54.2 on the Penn Treebank dataset, outperforming both the original and fine-tuned AWD-LSTM models as well as Transformer-XL and MoS .

For machine translation, DeFINE improves the efficiency of a Transformer model (Vaswani et al., 2017) by 26% while maintaining translation quality.

We provide substantive experiments which detail the impact of our architecture decisions and demonstrate the effectiveness of DeFINE across models of varying capacities.

Many sequence modeling tasks -including language modeling and machine translation -have a large vocabulary.

As a consequence, the majority of a model's parameters are located in the input (or embedding) and the output (or classification) layers.

To reduce the computational load presented by these layers, Press & Wolf (2017) and Inan et al. (2017) introduce an effective mechanism called weight-tying that enables learning input and output representations jointly while significantly reducing the number of network parameters.

To further reduce the computational load from these layers, factorization-based methods, such as projective embeddings (Acharya et al., 2019; Shu & Nakayama, 2017; Dai et al., 2019) , grouped embeddings (Chen et al., 2016; Grave et al., 2017a; Goodman, 2001; Mnih & Hinton, 2009; Morin & Bengio, 2005) , and slim embeddings , have been proposed.

Projective embeddings approximate a large embedding matrix with two smaller matrices while grouped embeddings cluster input tokens by frequency and assign different capacities to different clusters using projective embedding methods.

We note that projective embeddings is a special case of grouped embeddings when the number of clusters is one.

The adaptive input method of Baevski & Auli (2019) generalizes projective and grouped embedding methods and proposes a factorization method that allows for faster, memory-efficient end-to-end training while providing similar or better benefits compared to existing post-training methods which require a pretrained embedding matrix (Chen et al., 2018; Shu & Nakayama, 2017) .

Unlike projective and grouped embeddings, extends group transformation (Kuchaiev & Ginsburg, 2017; Mehta et al., 2018) with the shuffling algorithm of Fisher & Yates (1943) to factorize these layers.

DeFINE is orthogonal to these factorization methods; our empirical results in Section 4 show improved performance compared to these methods alone.

Recent advances in sequence modeling, such as Transformers and multi-layer RNNs, demonstrate the power of deep architectures in NLP (Jozefowicz et al., 2016; Vaswani et al., 2017; Merity et al., 2018a) .

But while significant attention has been given to modeling the interactions between words with deep architectures (e.g. ELMo (Peters et al., 2018) and BERT (Devlin et al., 2019) ), contextfree word representations are typically modeled with only corpus statistics (Pennington et al., 2014) or a single linear transformation McCann et al., 2017) .

Character-level models (Kim et al., 2016 ) also effect deep representations of words as a convolution over characters, however these models often require more capacity to deliver performance comparable to word-level models (Baevski & Auli, 2019) .

Still, DeFINE can be used to learn deep representations of a variety of token types, including words, characters, or byte-pair encodings (Sennrich et al., 2015) .

Word embedding is often treated as simple function of a one-hot vector to a dense continuous space.

The embedding layer can thus be thought of as a wide, shallow network consisting of a single linear transformation.

At its heart, the function that this network approximates (call it f ) takes a word from its orthographic form to a representation of those of its syntactic and semantic properties which are relevant for modeling an arbitrary number of contexts in which the word can occur.

Most NLP research assumes a simple embedding layer can sufficiently approximate the intractable function f .

We hypothesize that, due to the complexity of f , a shallow network would require exceptional capacity to learn a good approximation.

Time and data constraints prohibit learning such a high capacity shallow network.

We propose, based on recent theoretical results of Liang & Srikant (2017) , 2 that a deeper network can approximate f with significantly fewer parameters than a shallow network.

The validity of this assumption is evidenced by our experimental results in Section 4.

In this work, we introduce DeFINE, an effective way of learning deep word-level representations in high-dimensional space with a minimum of additional parameters.

Our method is based on a Map-Expand-Reduce (MER) principle, described in Section 3.1, that first maps an input word to a low dimensional embedding vector, then transforms it to a high-dimensional space using a computationally efficient hierarchical group transformation (HGT, Section 3.2), which is sketched in Figure  2c .

The resultant vector is then transformed to a low-dimensional space.

Over the course of these transformations, we make use of a new connectivity pattern that establishes a direct link between the input and output layers (Figure 3) , promoting feature reuse, and improving gradient flow (Section 3.3).

The output layer of DeFINE can then be used in place of a traditional embedding as an input to sequence modeling tasks.

We detail the various aspects of the architecture below.

The first step in MER, Map, is similar to standard sequence models.

Every input word in the vocabulary V is mapped to a fixed dimensional vector e i ∈ R n×1 .

However, in our case, the value of n is small (say 64 or 128, compared to typical dimensions of 400 or more).

The next step, Expand, takes e i as an input and applies a hierarchical group transformation (HGT) to produce a very highdimensional vectorê i ∈ R k×1 , where k >> n. Unlike a stack of fully connected layers, HGT learns deep representations efficiently from different subsets of the input using sparse and dense connections.

The last step, Reduce, projects the vectorê i to a lower dimensional space to produce the final embedding vector e o ∈ R m×1 for a given input word.

The dimensions of e o can be matched to contextual representation models, such as LSTMs or Transformers, allowing DeFINE to serve as an input layer for these models.

Figure 2 .

Here, N is the total number of layers, n l and k l are the input and output dimensions of l-th layer, g l is the number of groups in l-th layer, and g is the fixed number of groups in group linear transforms.

We introduce a hierarchical group transformation (HGT), sketched in Figure 2c , to learn deep wordlevel representations efficiently.

HGT comprises of a stack of N layers.

At each layer, HGT uses a different number of groups that allow it learn representations from different subsets of input.

HGT starts with g max groups at the first layer and then subsequently decreases the number of groups by a factor of 2 at each level.

This hierarchical grouping mechanism sparsifies the connections in fully connected (or linear) layers and allows us to learn representations efficiently with fewer parameters.

Similar to a stack of fully connected layers, the N -th layer in HGT has access to every input element of the first layer through multiple paths, thereby, allowing it to learn effective representations.

Group linear transformations (GLT), originally introduced to improve the efficiency of the LSTM, also sparsify the connections in fully connected layers and significantly reduce computational costs (Kuchaiev & Ginsburg, 2017; Mehta et al., 2018) .

However, if we stack multiple GLT layers, the outputs of certain group are only derived from a small fraction of the input, thus learning weak representations.

The hierarchical grouping mechanism in HGT allows the N -th layer to obtain input data from multiple paths, enabling HGT to learn stronger representations.

A comparison of different transformations is given in Figure 2 .

We can see that HGT is both efficient and has better access to the input.

Note that linear and group linear transforms are special cases of HGT when g l = 1 and g l = g (fixed), respectively.

To transform e i ∈ R n×1 toê i ∈ R k×1 , HGT first samples the space between n and k linearly to construct N intermediate layers of increasing dimensionality.

Therefore, the output vector produced by l + 1-th layer will have higher dimensionality than the l-th layer.

Assume that the linearly spaced vector dimensions are divisible by g max , we transform e i toê i as follows:

where g l = max gmax 2 l−1 , 1 , W l are the weights learned at l-th layer, and F G is a group transformation function defined in Mehta et al. (2018) .

Group transformation splits the input into g groups, each of which is processed independently using a linear transformation.

The output of these groups are then concatenated to produce final output.

See Section A.1 for details.

The DeFINE unit is composed of HGT transformations that are designed using the MER principle.

Though HGT layers are an efficient approximation to computationally expensive fully connected layers, they might impede training as the depth N of the DeFINE unit grows.

Residual connections (He et al., 2016 ) have proved to be very effective at mitigating this issue, however, such connections are difficult to implement in HGT because the input and output dimensions of each layer are different.

To maximize the flow of information and facilitate training with deeper DeFINE units, we introduce a simple new skip-connection that establishes a direct link between any layer in HGT with the input e i .

Figure 3 visualizes the DeFINE unit with a depth of two (N =2).

To enable the sparse connections in HGT to have access to the input e i and the output of the previous layer (ê

Figure 3: The DeFINE unit with N = 2 that uses HGT to learn word-level representations efficiently and a direct connection with the input to maximize the flow of information.

the input and the output into g l groups using a split layer.

The chunked input and output vectors are then mixed such that the first chunk of the input and the first chunk of the l − 1-th layer's output are put together as the input for the first group transformation in the l-th layer, and so on until g l inputs have been constructed.

The resultant vector is then fed to l-th layer.

This mechanism promotes input feature reuse efficiently.

Additionally, it establishes a direct link with the input e i , allowing gradient to flow back to the input via multiple paths and resulting in improved performance.

The DeFINE unit can be easily integrated with any new or existing sequence models.

Sequence models typically consist of a stack of an input layer (embedding or adaptive input layer), a contextual model (e.g. LSTM or Transformer), and a classification layer (a fully-connected or adaptive softmax).

Since DeFINE learns deep word-level representations, we can easily stack it immediately after the input.

An example is shown in Figure 1 , where DeFINE is integrated with Transformer-XL, a state-of-the-art language model.

DeFINE enables the use of relatively lower dimensions in the input layer, thus reducing network parameters.

The input word-level representations, e i ,ê i , and e o , that a neural model learns for each word are independent of other words.

This allows us to create another independent look-up table (after training a model) that caches the mapping between the input word and the output of the DeFINE unit (e o ), resulting in a mechanism that allows to skip the computations of the DeFINE unit at inference time.

We demonstrate the performance of DeFINE on two sequence modeling tasks: language modeling (Section 4.1) and machine translation (Section 4.2).

We compare the performance of DeFINE with existing factorization methods in Section 4.3.

We also provide ablations in Section 4.4 to show the effectiveness of our design decisions.

Throughout this section, we use the following notation: n, k, and m are dimensions of e i ,ê i , and e o respectively, and N represents depth of DeFINE.

In this section, we study the performance of our models with LSTM-and Transformer-based language models on two datasets: WikiText-103 (Merity et al., 2017) Results of LSTM-based language models: Table 1 summarizes the results of LSTM-based language models.

Though the adaptive input (Baevski & Auli, 2019) and output (Grave et al., 2017a) methods are effective and reduce the number of parameters significantly, our method further improves performance by about 3 points while learning only 1.25% (or 0.4 million) more parameters.

It is important to note that the computational complexity of models in R2 and R3 is the same because our method allows caching outputs of DeFINE for use at inference (see Section 3.4).

When we scale the depth of DeFINE from 3 to 11 layers (Table 1b) 3 , the performance improves by a further 6 points, delivering competitive performance to existing RNN-based methods with fewer parameters (e.g. 1/3 as many parameters as Merity et al. (2018a) ).

The performance of our model is better than existing methods such as Dauphin et al. (2017) and Bai et al. (2018) .

Table 2 compares the performance of Transformer-XL, a state-of-the-art Transformer-based model, with and without DeFINE.

Table 2a shows our method is able to attain similar performance to Dai et al. (2019) while learning 10M fewer parameters.

It is interesting to note that DeFINE enables us to reduce the computational burden from the input and output layers by a large amount with minimal impact on performance.

With DeFINE, the performance of Transformer-XL drops only by about 2 points while the number of parameters are reduced by 50%.

For similar reduction in the number of parameters, the performance of original Transformer-XL drops by 5 points, suggesting the proposed method for learning word-level representations is effective.

Table 2b highlights the fact that Transformer-XL with DeFINE is able to achieve comparable perplexity to a standard Transformer-XL with projective embeddings while using significantly fewer parameters. (Dai et al., 2019) that linearly projects the vector e i to a dimension of m = 384.

Except the row marked with that uses inner model dimension of 2100, all other rows uses an inner model dimension of 1920.

Best number in each group is highlighted in red while overall best numbers are marked in bold.

Table 2a shows that adding DeFINE significantly improves results with low overhead; Table 2b shows the parameter reduction using DeFINE for similar performance.

Data and models: The Penn Treebank dataset (Marcus et al., 1994) contains about 929K/74K/82K tokens in its train, validation, and test sets respectively.

It has a vocabulary size of about 10K.

Following recent works, we use the processed version provided by Mikolov et al. (2010) .

To evaluate the effectiveness of our model, we compare to AWD-LSTM (Merity et al., 2018b) .

Our model replaces the embedding layer in AWD-LSTM with DeFINE unit with the following settings: n = 128, k = 1024, N = 7, and m = 400.

We use the same hyper-parameters and PyTorch version as the original AWD-LSTM.

Results: Results are summarized in Table 1c .

The proposed method improves the performance of AWD-LSTM by 4 points while simultaneously reducing the number of parameters by 4 million.

Without any finetuning, AWD-LSTM + DeFINE achieves comparable performance to state-of-theart methods, including Transformer-XL, with fewer parameters.

Data and models: We use the WMT 2014 English-German (EN-DE) dataset (Luong et al., 2015) for training.

Following Vaswani et al. (2017) , we encode the sentences using byte-pair encoding (Britz et al., 2017) and use newstest2014 and newstest2017 as validation and test sets, respectively.

We integrate DeFINE with the state-of-the-art Transformer model (Vaswani et al., 2017) with following parameters: n = 128, k = 1024, m = 512, and N = 3.

We use the implementation in OpenNMT-py (Klein et al., 2017) for training and evaluation with the recommended hyper-parameters.

Table 4 : Performance comparison of different sequence models with different factorization methods.

Projective and adaptive factorization method refers to methods in Dai et al. (2019) and Baevski & Auli (2019) , respectively.

For language modeling, performance is measured by perplexity; for machine translation, BLEU is used.

Table 4 compares the performance of different factorization methods for different sequence models.

With DeFINE, the performance and efficiency of sequence models improves across different tasks.

This is likely because the output of DeFINE more closely approximates the correlation pattern of a standard embedding layer compared to other embeddings (see Figure 4 and Appendix B).

Furthermore, we see that strong correlations between dimensions in the mapping layer of DeFINE are reduced over the course of the expansion layers (see Figures 8, 9 , and 10 in Appendix).

Figure 11 in Appendix shows that groups within an expansion layer of DeFINE are not correlated, suggesting these matrices are learning different representations of their input.

In this section, we provide an analysis of our design choices using an LSTM-based language model.

In our ablations, we choose LSTM-over Transformer-based language models because they are less sensitive to hyper-parameters and can be trained on a single GPU.

We use the same hyper-parameters for training as described in Section 4.1.1, specifically N = 7, n = 384, k = 1024, and m = 384.

Impact of different transformations: Table 5 summarizes our results.

HGT is as effective as linear transformation while learning two million fewer parameters.

Compared to group linear transform (GLT), HGT improves perplexity by about 5 points while learning a similar number of parameters.

Furthermore, when we establish a direct connection with the input (see Section 3.2 for details), the performance further improves by 2.9 points with a minimal impact on number of parameters, suggesting that DeFINE learns good representations.

Impact of scaling depth (N ) and width (k): Table 6 summarizes the results of our scaling experiments.

For the same value of k, the performance of the language model improves with the increase in the depth N .

However, when we scale the width k for a fixed value of depth N , the performance does not improve.

This is likely because, as we increase the size of k, more neurons are receiving their input from the same subset of dimensions and thus learning many redundant parameters.

DeFINE with different connections: Table 7a demonstrates the impact of residual connections in DeFINE.

In order to facilitate residual connections inside DeFINE, we fix the dimension of each layerê l i in DeFINE to be k 2 instead of linearly spanning from n to k. We can clearly see that the proposed skip-connections are more effective.

In the MER strategy (Section 3.1), we project the highdimensional vector to a low-dimensional space before feeding it to a contextual model, such as an LSTM.

We empirically found that the performance with and without this reduction step is similar, however, a model without the reduction step learns more parameters (Table 7b ).

DeFINE uses a deep, hierarchical, sparse network with new skip connections to learn better word embeddings efficiently.

Sequence models with DeFINE (e.g. Transformer and LSTM) perform comparably or better with state-of-the-art methods with fewer parameters.

Our experiments show that the proposed architectural decisions each contribute to the effectiveness of the DeFINE unit.

We believe neural sequence models with DeFINE can be further improved with extended hyperparameter search, similar to Melis et al. (2018) .

In future work, we will apply DeFINE to other sequence modeling tasks.

For instance, we believe that pretrained language model architectures such as ELMo and BERT can benefit from incorporating DeFINE to improve efficiency and performance.

Another direction is to use the components of DeFINE -specifically MER, HGT, and mixing layers -in neural architecture search processes.

We have shown the promise of these components here, but a thorough architecture search may discover more optimal configurations in the large search space defined by the depth, grouping, and connectivity parameters.

To produce an output y ∈ R m×1 from an input x ∈ R n×1 and weight matrix W ∈ R n g × m g , F G first chunks the input x into g groups and then concatenates the chunked parts to producex ∈ R g× n g .x is then multiplied with weight matrix W to produceŷ =x · W ∈ R g× m g .

The resultant vectorŷ is then flattened to produce y. When g = 1, we obtain the linear transform.

Block level diagrams of different variants of DeFINE are given in Figure 5 .

Figure 5a stacks transformation layer F G (Eq. 1) and is the same as HGT in Figure 2c .

Figure 5b adds a residual connection to Figure 5a .

Figure 5c is the same as Figure 3 while Figure 5d is the same as Figure 5c , but without split and mixer functionality.

For training LSTM-based language models, we use a single NVIDIA GTX 1080 Ti GPU with 11 GB GPU memory while for training Transformer-XL, we used four GeForce RTX 2080 Ti GPUs, each with 11 GB of GPU memory (as recommended by authors).

Following recent works, including Merity et al. (2018a) and Baevski & Auli (2019) , we use adaptive inputs as a mapping function in DeFINE and adaptive softmax for classification for our experiments with RNN-based sequence models.

We also tie weights between the adaptive inputs and outputs.

For Transformer-XL Dai et al. (2019) , we use projective embeddings (as done by authors).

We train our models using PyTorch (v1.2).

For LSTM-based language models, we use similar hyper-parameters as Merity et al. (2018a) which are summarized in Section 8.

A.4 PERFORMANCE OF TRANSFORMER-XL ON WIKITEXT-103 Figure 6 plots the validation perplexity of Transformer-XL on the WikiText-103 as a function of training steps.

We can see that DeFINE enables Transformer-XL to deliver similar performance with fewer parameters.

Computing correlation map: Let us say that we have an arbitrary look-up table E ∈ R V×m that maps every word in vocabulary V to a m-dimensional vector space.

We compute the correlation map M as: M = E T · E ∈ R m×m .

4 If the correlation map is identity, then it suggests that the mdimensions in E are independent.

To encode better contextual representations among words using context models such as LSTMs and Transformers, embedding dimensions should be independent.

Can DeFINE approximate the standard embedding layer?

Figure 7 visualizes the correlation maps of embeddings learned using a standard embedding layer (top row), projective embeddings (Acharya et al., 2019; Dai et al., 2019 ) (middle row), and DeFINE embeddings (bottom row) at different values of n, where n is the dimension of mapping layer in DeFINE.

Compared to projective embeddings, DeFINE is able to approximate the standard embedding layer efficiently and effectively (see Table 2 for efficiency and performance comparison).

Furthermore, we provide layer-wise comparison for DeFINE at different values of n in Figures 8,  9 , and 10.

The mapping layer in DeFINE is in low-dimensional space and has correlations.

As we learn deeper representations using DeFINE, these correlations are reduced and we obtain a correlation matrix similar to a standard embedding layer.

This suggests that DeFINE is effective in approximating the standard embedding layer.

Importantly, the groups at different expansion layers in DeFINE are independent (see Figure 11 ), suggesting these matrices are learning different representations of their input.

Projective Embedding Output n = 64 n = 128 n = 256

DeFINE Output n = 64 n = 128 n = 256

@highlight

DeFINE uses a deep, hierarchical, sparse network with new skip connections to learn better word embeddings efficiently. 

@highlight

This paper describes a new method for learning deep word-level representations efficiently by using a hierarchical structure with skip-connections for the use of low dimensional input and output layers.