Fine-tuning with pre-trained models has achieved exceptional results for many language tasks.

In this study, we focused on one such self-attention network model, namely BERT, which has performed well in terms of stacking layers across diverse language-understanding benchmarks.

However, in many downstream tasks, information between layers is ignored by BERT for fine-tuning.

In addition, although self-attention networks are well-known for their ability to capture global dependencies, room for improvement remains in terms of emphasizing the importance of local contexts.

In light of these advantages and disadvantages, this paper proposes SesameBERT, a generalized fine-tuning method that (1) enables the extraction of global information among all layers through Squeeze and Excitation and (2) enriches local information by capturing neighboring contexts via Gaussian blurring.

Furthermore, we demonstrated the effectiveness of our approach in the HANS dataset, which is used to determine whether models have adopted shallow heuristics instead of learning underlying generalizations.

The experiments revealed that SesameBERT outperformed BERT with respect to GLUE benchmark and the HANS evaluation set.

In recent years, unsupervised pretrained models have dominated the field of natural language processing (NLP).

The construction of a framework for such a model involves two steps: pretraining and fine-tuning.

During pretraining, an encoder neural network model is trained using large-scale unlabeled data to learn word embeddings; parameters are then fine-tuned with labeled data related to downstream tasks.

Traditionally, word embeddings are vector representations learned from large quantities of unstructured textual data such as those from Wikipedia corpora (Mikolov et al., 2013) .

Each word is represented by an independent vector, even though many words are morphologically similar.

To solve this problem, techniques for contextualized word representation (Peters et al., 2018; Devlin et al., 2019) have been developed; some have proven to be more effective than conventional word-embedding techniques, which extract only local semantic information of individual words.

By contrast, pretrained contextual representations learn sentence-level information from sentence encoders and can generate multiple word embeddings for a word.

Pretraining methods related to contextualized word representation, such as BERT (Devlin et al., 2019) , OpenAI GPT (Radford et al., 2018) , and ELMo (Peters et al., 2018) , have attracted considerable attention in the field of NLP and have achieved high accuracy in GLUE tasks such as single-sentence, similarity and paraphrasing, and inference tasks .

Among the aforementioned pretraining methods, BERT, a state-of-the-art network, is the leading method that applies the architecture of the Transformer encoder, which outperforms other models with respect to the GLUE benchmark.

BERT's performance suggests that self-attention is highly effective in extracting the latent meanings of sentence embeddings.

This study aimed to improve contextualized word embeddings, which constitute the output of encoder layers to be fed into a classifier.

We used the original method of the pretraining stage in the BERT model.

During the fine-tuning process, we introduced a new architecture known as Squeeze and Excitation alongside Gaussian blurring with symmetrically SAME padding ("SESAME" hereafter).

First, although the developer of the BERT model initially presented several options for its use, whether the selective layer approaches involved information contained in all layers was unclear.

In a previous study, by investigating relationships between layers, we observed that the Squeeze and Excitation method (Hu et al., 2018) is key for focusing on information between layer weights.

This method enables the network to perform feature recalibration and improves the quality of representations by selectively emphasizing informative features and suppressing redundant ones.

Second, the self-attention mechanism enables a word to analyze other words in an input sequence; this process can lead to more accurate encoding.

The main benefit of the self-attention mechanism method is its high ability to capture global dependencies.

Therefore, this paper proposes the strategy, namely Gaussian blurring, to focus on local contexts.

We created a Gaussian matrix and performed convolution alongside a fixed window size for sentence embedding.

Convolution helps a word to focus on not only its own importance but also its relationships with neighboring words.

Through such focus, each word in a sentence can simultaneously maintain global and local dependencies.

We conducted experiments with our proposed method to determine whether the trained model could outperform the BERT model.

We observed that SesameBERT yielded marked improvement across most GLUE tasks.

In addition, we adopted a new evaluation set called HANS , which was designed to diagnose the use of fallible structural heuristics, namely the lexical overlap heuristic, subsequent heuristic, and constituent heuristic.

Models that apply these heuristics are guaranteed to fail in the HANS dataset.

For example, although BERT scores highly in the given test set, it performs poorly in the HANS dataset; BERT may label an example correctly not based on reasoning regarding the meanings of sentences but rather by assuming that the premise entails any hypothesis whose words all appear in the premise (Dasgupta et al., 2018) .

By contrast, SesameBERT performs well in the HANS dataset; this implies that this model does not merely rely on heuristics.

In summary, our final model proved to be competitive on multiple downstream tasks.

Most related studies have used pretrained word vectors (Mikolov et al., 2013; Pennington et al., 2014) as the primary components of NLP architectures.

This is problematic because word vectors capture semantics only from a word's surrounding text.

Therefore, a vector has the same embedding for the same word in different contexts, even though the word's meaning may be different.

Pretrained contextualized word representations overcome the shortcomings of word vectors by capturing the meanings of words with respect to context.

ELMo (Peters et al., 2018) can extract contextsensitive representations from a language model by using hidden states in stacked LSTMs.

Generative pretraining (Radford et al., 2018) uses the "Transformer encoder" rather than LSTMs to acquire textual representations for NLP downstream tasks; however, one limitation of this model is that it is trained to predict future left-to-right contexts of a unidirectional nature.

BERT (Devlin et al., 2019) involves a masked language modeling task and achieves high performance on multiple natural language-understanding tasks.

In BERT architecture, however, because the output data of different layers encode a wide variety of information, the most appropriate pooling strategy depends on the case.

Therefore, layer selection is a challenge in learning how to apply the aforementioned models.

The Squeeze and Excitation method was introduced by Hu et al. (2018) , who aimed to enhance the quality of representations produced by a network.

Convolutional neural networks traditionally use convolutional filters to extract informative features from images.

Such extraction is achieved by fusing the spatial and channel-wise information of the image in question.

However, the channels of such networks' convolutional features have no interdependencies with one another.

The network weighs each of its channels equally during the creation of output feature maps.

Through Squeeze and Excitation, a network can take advantage of feature recalibration and use global information to emphasize informative features and suppress less important ones.

The self-attention network relies on an attention mechanism to capture global dependencies without considering their distances by calculating all the positions in an input sequence.

Our Gaussianblurring method focuses on learning local contexts while maintaining a high ability to capture longrange dependencies.

Localness modeling was considered a learnable form of Gaussian bias in which a central position and dynamic window are predicted alongside intermediate representations in a neural network.

However, instead of using Gaussian bias to mask the logit similarity of a word, we performed Gaussian bias in the layer after the embedding layer to demonstrate that performing element-wise operations in this layer can improve the model performance.

A recent study investigated whether neural network architectures are prone to adopting shallow heuristics to achieve success in training examples rather than learning the underlying generalizations that need to be captured.

For example, in computer vision, neural networks trained to recognize objects are misled by contextual heuristics in cases of monkey recognition (Wang et al., 2017) .

For example, in the field of natural language inference (NLI), a model may predict a label that contradicts the input because the word "not", which often appears in examples of contradiction in standard NLI training sets, is present (Naik et al., 2018; Carmona et al., 2018) .

In the present study, we aimed to make SesameBERT robust with respect to all training sets.

Consequently, our experiments used HANS datasets to diagnose some fallible structural heuristics presented in this paper .

We focused on BERT, which is the encoder architecture of a multilayer Transformer (Vaswani et al., 2017) , featuring some improvements.

The encoder consists of L encoder layers, each containing two sublayers, namely a multihead self-attention layer and a feed-forward network.

The multihead mechanism runs through a scaled dot product attention function, which can be formulated by querying a dictionary entry with key value pairs (Miller et al., 2016) .

The self-attention input consists of a query Q ??? R l??d , a key K ??? R l??d , and a value V ??? R l??d , where l is the length of the input sentence, and d is the dimension of embedding for query, key and value.

For subsequent layers, Q, K, V comes from the output of the previous layer.

The scaled dot product attention (Vaswani et al., 2017 ) is defined as follows:

The output represents the multiplication of the attention weights A and the vector v, where A = sof tmax(

The attention weights A i,j enabled us to better understand about the importance of the i-th key-value pairs with respect to the j-th query in generating the output (Bahdanau et al., 2015) .

During fine-tuning, We used the output encoder layer from the pretrained BERT model to create contextualized word embeddings and feed these embeddings into the model.

Although several methods have been developed for extracting contextualized embeddings from various layers, we believed that these methods had substantial room for improvement.

Therefore, we used Squeeze and Excitation to solve the aforementioned problem.

In this study, we proposed the application of Squeeze and Excitation (Hu et al., 2018) ; its application to the output of the encoder layer was straightforward once we realized that the number of channels was equivalent to the number of layers.

Therefore, we intended to use the term channels and layers interchangeably.

First, we defined U :,:,k as the output of the k-th encoder layer, for all 1 ??? k ??? n. We wanted to acquire global information from between the layers before feeding the input into the classifier; therefore, we concatenated all the output from each encoder layer to form the feature maps U ??? R l??d??n .

In the squeeze step, by using global average pooling on the kth layer, we were able to squeeze the global spatial information into a layer descriptor.

In other words, we set the kth layer's output of the squeeze function as Z :,:,k .

Figure 1: We extracted the output from each layer of the encoders and concatenated all the layers to form a three-dimensional tensor U. We then performed Squeeze f sq (U) and Excitation f ex (f sq (U)) to obtain the weight of each output layer.

Finally, we fed the weighted average of all layers into the classifier.

In this work we employed n = 12 attention layers.

In the excitation step, we aimed to fully capture layer-wise dependencies.

This method uses the layer-wise output of the squeeze operation f sq to modulate interdependencies of all layers.

Excitation is a gating mechanism with a sigmoid activation function that contains two fully connected layers.

Let W 1 and W 2 be the weights of the first and second fully connected layers, respectively, and let r be the bottleneck in the layer excitation that encodes the layer-wise dependencies; therefore, W 1 ??? R n?? n r , and W 2 ??? R n r ??n .

The excitation function f ex :

where z is the vector squeezed from tensor Z.

Finally, we rescaled the output Z :,:,k by multiplying it by s k .

The rescaled output is deonted as u k .

The scaling function f scale is defined as follows:

We concatenated all rescaled outputs from all encoder layers to form our rescaled feature maps u. The architecture is shown in Figure 1 .

We then extracted layers from the rescaled feature maps, or calculated a weighted average layer u avg .

Given an input sequence X = {x 1 , x 2 , ..., x l } ??? R l??d , the model transformed it into queries Q, keys K, and values V , where Q, K, and V ??? R l??d .

Multihead attention enabled the model to jointly attend to information from different representation subspaces at different positions.

Thus, the three types of representations are split into h subspaces of size

where

To capture the local dependency related to each word, we first used a predefined fixed window size k to create a Gaussian blur g, where g ??? R k :

where ?? refers to the standard deviation.

Several Gaussian-blurring strategies are feasible for applying convolutional operations to attention outputs.

Figure 2: Diagram of a one-dimensional Gaussian blur kernel, which was convoluted through the input dimension.

This approach enabled the central word to acquire information concerning neighboring words with weights proportional to the Gaussian distribution.

The first strategy focuses on each attention output O h .

We restrict?? h i,j,: to a local scope with a fixed size k centered at the position i and dimension j, where 1 ??? j ??? d, and k can be any odd number between 1 and l, expressed as follows:

We then enhance the localness of?? h i,j,: through a parameter-free 1D convolution operation with g.

The final attention output is O h , which is the dot product between the Gaussian kernel and the corresponding input array elements at every position of??

where * is defined as a convolution operation, as illustrated in Figure 2 .

Another option focuses on values V. We applied the aforementioned method again but restrict V h to a local scope.

The final attention output O h is denoted as follows:

The difference between the present method and the method of performing Gaussian blurring on attention outputs and values is that the method of performing Gaussian blurring on attention outputs and values places greater emphasis on the interaction of cross-query vectors, whereas the present method focuses on cross-value vectors.

Finally, the outputs of the h attention heads are concatenated to form the final output representation O:

where O ??? R l??d .

The multihead mechanism enables each head to capture distinct linguistic input properties .

Furthermore, because our model is based on BERT, which builds an encoder framework with a stack of 12 layers, we were able to apply locality modeling to all layers through Squeeze and Excitation.

Therefore, we expected that the global information and local properties captured by all layers could be exploited.

We evaluated the proposed SesameBERT model by conducting multiple classification tasks.

For comparison with the results of a previous study on BERT (Devlin et al., 2019) , we reimplemented the BERT model in TensorFlow in our experiments.

1 In addition, we set most of the parameters to be identical to those in the original BERT model, namely, batch size: 16, 32, learning rate: 5e-5, 3e-5, 2e-5, and number of epochs: 3, 4.

All of the results in this paper can be replicated in no more than 12 hours by a graphics processing unit with nine GLUE datasets.

We trained all of the models in the same computation environment with an NVIDIA Tesla V100 graphics processing unit.

GLUE benchmark is a collection of nine natural language-understanding tasks, including questionanswering, sentiment analysis, identification of textual similarities, and recognition of textual entailment .

GLUE datasets were employed because they are sets of tools used to evaluate the performance of models for a diverse set of existing NLU tasks.

The datasets and metrics used for the experiments in this study are detailed in the appendix A.

We used a new evaluation set, namely the HANS dataset, to diagnose fallible structural heuristics presented in a previous study based on syntactic properties.

More specifically, models might apply accurate labels not based on reasoning regarding the meanings of words but rather by assuming that the premise entails any hypothesis whose words all appear in the premise (Dasgupta et al., 2018; Naik et al., 2018) .

Furthermore, an instance that contradicts the lexical overlap heuristics in MNLI is likely too rare to prevent a model from learning heuristics.

Models may learn to assume that a label is contradictory whenever a negation word is contained in the premise but not the hypothesis .

Therefore, whether a model scored well on a given test set because it relied on heuristics can be observed.

For example, BERT performed well on MNLI tasks but poorly on the HANS dataset; this finding suggested that the BERT model employs the aforementioned heuristics.

The main difference between the MNLI and HANS datasets is their numbers of labels.

The MNLI dataset has three labels, namely Entailment, Neutral, and Contradiction.

In the HANS dataset, instances labeled as Contradiction or Neutral are translated into non-entailment.

Therefore, this dataset has only two labels: Entailment and Non-entailment.

The HANS dataset targets three heuristics, namely Lexical overlap, Subsequence, and Constituent, with more details in appendix B. This dataset not only serves as a tool for measuring progress in this field but also enables the visualization of interpretable shortcomings in models trained using MNLI.

This subsection provides the experiment results of the baseline model and the models trained using our proposed method.

We performed Gaussian blurring on attention outputs in the experiment.

In addition, we employed a batch size of 32, learning rates of 3e-5, and 3 epochs over the data for all GLUE tasks.

We fine-tuned the SesameBERT model through 9 downstream tasks in the datasets.

For each task, we performed fine-tuning alongside Gaussian blur kernel sigmas 1e-2, 1e-1, 3e-1, and 5e-1 and selected that with the most favorable performance in the dev set.

Because GLUE datasets do not distribute labels for test sets, we uploaded our predictions to the GLUE server for evaluation.

The results are presented in Table 1 ; GLUE benchmark is provided for reference.

In most tasks, our proposed method outperformed the original BERT-Base model (Devlin et al., 2019) .

For example, in the RTE and AX datasets, SesameBERT yielded improvements of 1.2% and 1.6%, respectively.

We conducted experiments on GLUE datasets to test the effects of Gaussian blurring alongside BERT on the value layer and context layer.

Table 2 shows the degrees of accuracy in the dev set.

The performance of Gaussian blurring with respect to self-attention layers varied among cases.

Table 1 : Test results in relation to the GLUE benchmark.

The metrics for these tasks, shown in appendix A, were calculated using a GLUE score.

We compared our SesameBERT model with the original BERT-Base model, ELMo (Peters et al., 2018) and OpenAI GPT (Radford et al., 2018 et al. (2019) demonstrated that different layers vary in terms of their abilities to distinguish and capture neighboring positions and global dependency between words.

We evaluated the weights learned from all layers.

These weights indicated that a heavier weight represents greater importance.

The results are shown in appendix C. Because the lower layer represents word embeddings that are deficient in terms of context (Baosong Yang, 2018) , the self-attention model in the lower layer may need to encode representations with global context and may struggle to learn localness.

Table 3 shows the degree of accuracy predicted by each extracted attention output layer method.

The results indicated that the lower layers had lower accuracy.

We performed three ablation studies.

First, we examined the performance of our method without blurring; we observed that Squeeze and Excitation helped the higher layer.

This trend suggested that higher layers benefit more than do lower layers from Squeeze and Excitation.

Second, we analyzed the effect of Gaussian blurring on the context layer.

The results revealed that the method with blurring achieved higher accuracy in lower layers.

We assumed that capturing short-range dependencies among neighboring words in lower layers is an effective strategy.

Even if self-attention models capture long-range dependencies beyond phrase boundaries in higher layers, modeling localness remains a helpful metric.

Finally, we observed the direct effects of SesameBERT.

Although our proposed architecture performed poorly in lower layers, it outperformed the other methods in higher layers.

This finding indicated that in higher layers, using Squeeze and Excitation alongside Gaussian blurring helps self-attention models to capture global information in all layers.

We trained both BERT and SesameBERT on the MNLI-m dataset to evaluate their classification accuracy.

Similar to the results of another study (Devlin et al., 2019) , BERT achieved 84.6% accuracy, which is higher than that of SesameBERT, as shown in Table 1 .

In the HANS dataset, we explored the effects of two models on each type of heuristic.

The results are presented in Figure 3 ; we first examined heuristics for which the label was Entailment.

We can see that both models performed well; they assigned the correct labels almost 100% of the time, as we had expected them to do after adopting the heuristics targeted by HANS.

Next, we evaluated the heuristics labeled as Non-entailment.

BERT performed poorly for all three cases, meaning that BERT assigned correct labels based on heuristics instead of applying the correct rules of inference.

By contrast, our proposed method performed almost three times as well as BERT in the case of "Lexical overlap".

BERT performed poorly in all three cases in Right; this indicated that the model had adopted shallow heuristics rather than learning the latent information that it intended to capture.

This paper argues that capturing local contexts for self-attention networks with Gaussian blurring can prevent models from easily adopting heuristics.

Although our models performed poorly in cases of "Subsequence" and "Constituent", both of these heuristics may be hierarchical cases of the lexical overlap heuristic, meaning that the performance of this hierarchy would not necessarily match the performance of our models .

This paper proposes a fine-tuning approach named SesameBERT based on the pretraining model BERT to improve the performance of self-attention networks.

Specifically, we aimed to find highquality attention output layers and then extract information from aspects in all layers through Squeeze and Excitation.

Additionally, we adopted Gaussian blurring to help capture local contexts.

Experiments using GLUE datasets revealed that SesameBERT outperformed the BERT baseline model.

The results also revealed the weight distributions of different layers and the effects of applying different Gaussian-blurring approaches when training the model.

Finally, we used the HANS dataset to determine whether our models were learning what we wanted them to learn rather than using shallow heuristics.

We highlighted the use of lexical overlap heuristics as an advantage over the BERT model.

SesameBERT could be further applied to prevent models from easily adopting shallow heuristics.

A DESCRIPTIONS OF GLUE DATASETS

<|TLDR|>

@highlight

We proposed SesameBERT, a generalized fine-tuning method that enables the extraction of global information among all layers through Squeeze and Excitation and enriches local information by capturing neighboring contexts via Gaussian blurring.