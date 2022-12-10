Character-level language modeling is an essential but challenging task in Natural Language Processing.

Prior works have focused on identifying long-term dependencies between characters and have built deeper and wider networks for better performance.

However, their models require substantial computational resources, which hinders the usability of character-level language models in applications with limited resources.

In this paper, we propose a lightweight model, called Group-Transformer, that reduces the resource requirements for a Transformer, a promising method for modeling sequence with long-term dependencies.

Specifically, the proposed method partitions linear operations to reduce the number of parameters and computational cost.

As a result, Group-Transformer only uses 18.2\% of parameters compared to the best performing LSTM-based model, while providing better performance on two benchmark tasks, enwik8 and text8.

When compared to Transformers with a comparable number of parameters and time complexity, the proposed model shows better performance.

The implementation code will be available.

Character-level language modeling has become a core task in the field of natural language processing (NLP) such as classification (Zhang et al., 2015) , sequence tagging (Guo et al., 2019a) , question answering (He & Golub, 2016) , and recognition (Baek et al., 2019; Hwang & Sung, 2016) , with its simplicity on generating text and its adaptability to other languages.

Along with the development of deep learning in NLP, using recurrent neural networks (RNNs) have been a standard way to solve the problem for many years.

Recently, however, a new architecture, Transformer (Vaswani et al., 2017) , have shown promise in addressing this problem and have achieved breakthroughs in general language modeling (Al-Rfou et al., 2019; Dai et al., 2019) .

Though this technique has achieved incredible successes, it has led to the huge size of Transformerbased models due to building deeper and wider networks.

Transformer-XL (Dai et al., 2019) and GPT-2 , for instance, contain 277M and 1542M parameters, respectively.

This trend toward a large size model for performance is not suitable for edge device applications, which require small memory sizes, such as optical character reader (OCR) and speech to text (STT), and for auto-correction and auto-completion applications that need fast real-time responsiveness.

To tackle this issue, choosing an appropriately efficient strategy becomes more crucial, especially in the real-world application which requires not only good performance but a lightweight model.

In this paper, we introduce a lightweight transformer for character-level language modeling.

Our method is one of the factorization methods in that it separates the standard linear layer in transformer architecture using group-wise linear operation and makes sparse connectivity between linear transformations.

The proposed model is referred to as Group-Transformer since it is inspired by the group convolution approaches (Zhang et al., 2018; Sandler et al., 2018) that have effectively compressed huge image processing models for usability on mobile devices.

While the group strategy reduces parameters and calculations in the proposed modules, its mutually exclusive calculation for the multiple groups compromises performance, caused by the information loss of inter-group correlations.

To compensate for this problem, we added two inter-group operations that share a common feature over groups for the group attention layer and linking features in different groups for the group feed-forward layer.

By modeling the inter-group information flows, Group-Transformer becomes performant as well as lightweight.

We conducted extensive experiments on two benchmark datasets, enwik8 and text8, and found that Group-Transformer with 6M parameters outperformed all LSTM-based models with under 35M parameters.

Furthermore, Group-Transformer shows better performance when compared against Transformers with a comparable number of parameters.

We provide further analysis to identify the contributions of our proposed modules in detail.

To the best of our knowledge, Group-Transformer is the first attempt to build a lightweight Transformer with the group strategy.

Since Transformer has become a promising model for diverse NLP tasks, there have been attempts to improve its efficiency with two majority approaches.

The first is to restrict dependencies between input tokens to reduce superfluous pair-wise calculations Guo et al., 2019b; Sukhbaatar et al., 2019a) .

The approach provides time efficiency during inference, but it does not address the heavy parameterization of Transformer.

The second approach is to develop a lightweight network architecture while maintaining the properties of Transformer.

For example, Tay et al. (2019) utilize quaternion algebra to build lightweight modules for Transformer.

They also use the factorize the components of the embedding layer, but the expression power can be limited by the connection of factorized components based on the quaternion principle.

Another such approach (Sukhbaatar et al., 2019b) combined the multi-head attention and point-wise feed-forward layer to devise a unified module with fewer parameters.

Despite these attempts on architectural changes, their models still struggle to provide a lightweight language model with nearly still 30M parameters.

In this work, we describe a lightweight transformer with less than 10M parameters, which is extremely small when compared against previous character-level language models.

A group strategy has attracted much attention recently to compress many large and deep state-of-theart convolutional neural networks (CNNs) (Krizhevsky et al., 2012; Szegedy et al., 2015; Chollet, 2017) .

For example, when the group strategy is applied to a standard linear layer with a weight W ∈ R I×O , the feature map is partitioned into G groups.

As a result, the layer is replaced by G small linear layers where each holds a weight W ∈ R (I/G)×(O/G) , leading to a significant parameter reduction.

Although intuitively appealing, it has been reported that applying the group strategy to the model often leads to huge performance degradation, since the features in different groups cannot interact with each other.

To overcome this problem, ShuffleNet (Zhang et al., 2018) proposed channel shuffle operation to make interactions between different groups.

This kind of consideration has also been applied to recurrent neural networks (RNNs).

Kuchaiev & Ginsburg (2017) proposed group-wise RNN as a special form of ensembled RNNs.

But, they did not consider the interactions between different groups.

Inspired by the ShuffleNet, Gao et al. (2018) combined the shuffling idea into the group-wise RNN and achieved promising results.

In this work, we adopt the group strategy and build the group-wise operations suitable for Transformer architecture.

3 GROUP-TRANSFORMER Figure 1a shows the overall architecture of Group-Transformer.

It consists of a group embedding (bottom grey box), which embeds a character into grouped features, a group attention (yellow box), which contains attention modules to identify dependencies in the time domain, and a group feedforward layer (green box), which re-configures the grouped features.

As can be seen, when an input character is given, Group-Transformer converts the input into multiple group representation (blue dots and red dots), processes and merges them to predict the next character.

Figure 1b and 1c show group-wise information flow (blue and red arrows), and inter-group information flow (grey arrow) in the sub-modules.

Without the inter-group information flows, the grouped features are processed independently.

We observed that inter-group modeling ensures that the groups become aware of the others and prevents different groups hold the same information.

The following subsections describe architectural details of the sub-modules and their relations.

Group embedding layer identifies a set of embeddings to represent a token.

The idea of representing a sentence, word or even character using a set of vectors can widely be found in many NLP models that embed input tokens by concatenating (or summing) its embedding and its sub-units' embeddings (Verwimp et al., 2017; Bojanowski et al., 2017; Zhou et al., 2019) .

Similarly, we assume a single character c to be represented with G vector representations of groups, that is,

When a character is given, the group embedding layer retrieves a corresponding set of vectors and passes it to the following group attention layer.

Through this paper, we describe a process at a single time step.

The attention mechanism identifies dependencies between features in the time domain and combines the information of them.

It contains three steps; (1) identifying queries, keys, and values, (2) retrieving relative features at different times, and (3) transforming the attended feature into the input domain (Vaswani et al., 2017) .

The main focus of this paper is to apply a group strategy to the feature space of Transformer.

Thus, we let the second step be identical to those of the original Transformer and focused on the first and the third steps.

Figure 1b explains the architecture of group attention.

The multi-head attention module represents the second step, the under operations identify the queries for the first step, and the upper operations transform the attention output for the third step.

We note that we do not represent the key and value for the multi-head attention block in the figure because they are possible to come from another source domain.

The group attention processes the grouped features with intra-group operations (white boxes) and inter-group operations (grey boxes).

Let x = [x 1 , ..., x G ] be a set of input vectors where x g ∈ R Dgroup for the group g. Since the multi-head attention contains H attention modules for a single group, group attention first calculates query q gh for a group g and its head h as the below,

where W q-intra gh ∈ R Dgroup×(Dgroup/H) and W q-inter gh ∈ R Dgroup×(Dgroup/H) are linear weights to describe an intra-group (white boxes) and an inter-group (grey box) combinations, respectively.

In the formula, the first term on the right-hand side identifies a specific feature for the head h in the group g and the second term determines head-wise features that allow the grouped features to share a common expression retrieving other features in a different time.

When comparing with the fully connected linear layer over the groups, the approach restricts the connection between the groups, so requires a fewer number of parameters and calculations.

For the key k gh and the value v gh , we use fully connected layers by using all group pairs g and g;

Dgroup×(Dgroup/H) are linear weights.

As we mentioned, since the keys and the values can be defined from the other source domain, we use the same formula of the original Transformer, pursuing the universality of the proposed module.

The identified headed elements are used for connecting features in the time domain.

In this step, position encoding (Vaswani et al., 2017) has an important role for the features to be aware of their position in an input sequence.

In this paper, we apply the relative positional encoding, which describes a long-length character sequence effectively.

By following Dai et al. (2019) , we define the attention score map with the relative positional information and the attention mechanism determines the attended feature a gh of the head h in the group g.

The multiple heads [a g1 , ..., a gH ] in the group g are combined as the below;

where W o-intra gh ∈ R (Dgroup/H)×Dgroup and W o-inter gh ∈ R (Dgroup/H)×Dgroup are linear weights for combining intra-group and inter-group information, respectively.

As can be seen, the final output is determined with a specific feature from its own group and a shared feature from whole groups.

This step utilizes the same mechanism used to identify the queries with the same objective spreading group-wise information from multi-headed attention to all groups.

These intra-group and inter-group modelings mainly contribute to reducing the number of parameters and calculations.

Finally, the inputs x g are added into the output o g asx g = x g + o g for a residual connection.

Group feed-forward layer re-configures the outputs of the attention module,x g , by applying groupwise operation at each position.

Figure 1c shows the architecture of the proposed module.

As can be seen, the groups are shuffled (grey box) and support each other.

The group-wise features are processed with two linear transformations and one non-linear activation.

As the original module does, the linear layers in our module transpose the input feature into a high dimensional space with non-linear activation and transform the output back into the input space.

The group feed-forward layer can be formally explained as follows.

Given G input features [x 1 , ...,x G ], group feed-forward layer transposes the grouped features into a high dimensional space as follows;ȳ

where

Dgroup×D G are linear weights for mapping intra-group and inter-group information into theD G -dimensional space, relatively bigger than D group dimension.

Here, we introduce a low-rank matrix approximation on the inter-group transformation matrix W f1-inter g g .

Modeling interactions between the groups requires the G × G numbers of weights, as well as the multiple weights for the group g, transpose the group into the high dimensional space for the target group g .

If designing a fully connected weight for all groups like the original Transformer, the feed-forward layer still holds heavyweights and expensive calculations.

To reduce the overburden, we factorize the matrix W Sainath et al. (2013) and Novikov et al. (2015) .

The newly introduced dimension M is smaller than D group , and thus the number of parameters and calculation is reduced proportionally with the ratio between M and D group .

In this paper, we set M as D group /G to control the dimension relatively with the number of the groups.

Interestingly, such matrix factorization can be modeled efficiently with a group-wise linear transformation and a shuffle trick as shown in Figure 1c .

Please refer to the appendix for the detail of the shuffle trick.

Finally, a group-wise linear transformation is applied upon the high-dimensional feature as follow;

where W f2 g ∈ RD G ×Dgroup is a linear weight.

For a residual connection, each grouped input feature is added into the output of the group feed-forward layer;ŷ g =x g + y g .

Here, we describe the efficiency of Group-Transformer in view of the number of parameters and required computational costs.

When considering the original transformer, its required numbers of parameters are 4 * O(D 2 model ) for its attention module (query, key, value, and output linear) and 2 * O(D modelDmodel ) for its feed-forward module, whereD model is a bottleneck dimension.

Group-Transformer pursues to reduce the number of parameters by splitting the hidden state into multiple groups and processing them group-wisely.

When we set the total dimension over groups as

for group attention and 3 G * O(D modelDmodel ) for group feed-forward module.

The number of groups is increasing, the resources is decreasing.

Appendix B provides the detailed required resources of all sub-modules and comparisons with those of the original Transformer.

We demonstrate the efficiency of the proposed Group-Transformer with two popular benchmark datasets, enwik8 and text8.

The enwik8 dataset contains 100M of English Wikipedia texts with 204 unique characters including alphabets, non-Latin and special characters.

In comparison, the text8 dataset provides 100MB of pre-processed texts only with 27 unique characters by filtering superfluous content, such as tables, citations, and punctuation, and by replacing the non-Latin characters with spelled-out equivalents (i.e., "15" to "one five").

For a fair comparison with previous works, we used the training/dev/test splits defined by Mahoney (2011) for both enwik8 and text8.

Most of the experimental settings follow those of Dai et al. (2019) , where the difference lies in the hyperparameters that influence the size of the model.

We set the number of layers L as 9, and we fixed the total size of feature D model for a single character as 256 and the total numbers of heads as 8 while the number of groups are explored in {2,4}. For the regularization of the model, we applied layer normalization (Ba et al., 2016) independently over groups and dropout layers upon the outputs of the group attention and the group feed-forward layer with the probability p = 0.1.

The length of the feed sequence was 512 with the cached 512-length for the previous sequence (Dai et al., 2019) .

We use the Adam optimizer with a learning rate of 2.5e-4, β 1 of 0.9, β 2 of 0.999, a batch size of 22, the number of iterations of 400,000, and the best model on the validation set is chosen.

The implementation code will be available for the other details.

We compare the Group-Transformer against existing character-level language models using under 50M parameters in Table 1 .

The prior models are grouped according to their methodologies, including "LSTM," and "Transformer".

We observe that the Group-Transformer outperforms the LSTM models with under 30M parameters and that the 2 Group-Transformer attains the best performance against all prior LSTM models on the enwik8 dataset.

When compared to Transformers, we observe The number of groups can be interpreted as a hyper-parameter affecting the model size.

Figure 2 shows the effectiveness of three hyper-parameters such as the number of layers, the size of hidden dimension, and the number of groups.

The default model used Transformer-XL (Dai et al., 2019) with L = 9, H model = 8, D model = 256, andD model = 4 * D model , and then we reduced the three hyper-parameters.

When making the model thinner or shallower, the performances of the model become worse, but the required resources are getting lower.

When comparing ours with two reduction methods, the group strategy shows better performances than the models requiring comparable resources.

This experiment proved that the feature grouping methods, the main idea of this paper, is more efficient to reduce the model size and the time complexity than tuning other model parameters.

Group-Transformer includes two modules utilizing group-wise operations and inter-group modeling.

We conduct ablation studies to identify the contributions of the proposed modules and inter-group operations.

Table 3 : Ablation study on the proposed modules, group attention and group feed-forward layer.

Table 3 shows the module-wise impact on the number of parameters and performance.

For a fair comparison, we set the baseline model to a reduced Transformer-XL (Dai et al., 2019) of less than 8M parameters, and can gradually reduce the model size by replacing the attention and the feedforward layer with Group-Transformer module selectively.

When replacing the feed-forward layer with Group-Transformer module, we observe that the number of parameters in all cases decreases more efficiently than replacing the attention module.

Interestingly, when replacing both modules, the degradation is lower than the sum of the individual performance losses, but the sum of the individuals' reduces the required resources.

This result demonstrates more efficiency of concurrently using both group-wise modules.

H1 H2 H3 H4 H5 H6 H7 H8  H1  H2  H3  H4  H5  H6  H7  H8   G1   G2   H1 H2 H3 H4 H5 H6 H7 H8   G1   G2   H1 H2 H3 H4 H5 H6 H7 H8   G1   G2   G3   G4   H1 H2 H3 H4 H5 H6 H7 H3 H4 H5 H6 H7 H8  H1  H2  H3  H4  H5  H6  H7  H8   G1   G2   H1 H2 H3 H4 H5 H6 H7 H8   G1   G2   H1 H2 H3 H4 H5 H6 H7 H8   G1   G2   G3   G4   H1 H2 H3 H4 H5 H6 H7 H3 H4 H5 H6 H7 H8  H1  H2  H3  H4  H5  H6  H7  H8   G1   G2   H1 H2 H3 H4 H5 H6 H7 H8   G1   G2   H1 H2 H3 H4 H5 H6 H7 H8   G1   G2   G3   G4   H1 H2 H3 H4 H5 H6 H7 H3 H4 H5 H6 H7 H8  H1  H2  H3  H4  H5  H6  H7  H8   G1   G2   H1 H2 H3 H4 H5 H6 H7 H8   G1   G2   H1 H2 H3 H4 H5 H6 H7 H8   G1   G2   G3   G4   H1 H2 H3 H4 H5 H6 H7 In addition to this, we also investigate the influence of inter-group operations in our model.

When the inter-group operations are removed (grey boxes in Figure 1b and 1c) , we observed the performance degradation on 2-Group-Transformer by 0.028 bpc and 4-Group-Transformer by 0.051 bpc.

These gaps are relatively huge when compared to the performance gap between Transformer-XL and Group Transformers in Table 3 .

The results re-emphasize the importance of inter-group modeling in GroupTransformer.

Figure 3 shows the similarity patterns between the multi-head attention of our models ((a) and (c)) and the ablation models without the inter-group operations ((b) and (d)).

As can be seen, the multi-head attention map from the model without inter-group operations shows high similarities among different groups, while the proposed model shows the opposite.

These similarity patterns imply that the model cannot fully take advantage of multi-head attention, which is designed to attend multiple positions of content, without the proposed inter-group operation.

Recently, remarkable progress has been made in character-level language modeling by Transformer.

The advantage of Transformer lies in its effectiveness in modeling long-term dependencies between characters.

However, the models have been developed with a huge number of parameters, and the inference of them has required an expensive computational cost.

We argue that big models cannot be used in a limited computational environment.

Group-Transformer has been developed to prove the effectiveness of Transformer in a lightweight setting.

We have grouped features and proposed group-wise operations to reduce the number of parameters and time complexity of Transformer.

In addition, to fully realize the advantage of the original Transformer, we have connected the groups to interact with each other.

When applying Group-Transformer on enwik8 and text8, we found that Group-Transformer only with 6M parameters achieves better performances than LSTM-based models holding over 30M parameters.

Further analysis has proved the effectiveness of the group strategy to reduce computational resources.

Julian Georg Zilly, Rupesh Kumar Srivastava, Jan Koutník, and Jürgen Schmidhuber.

Recurrent highway networks.

In Proceedings of the 34th International Conference on Machine LearningVolume 70, pp.

4189-4198, 2017.

Here, we describe the shuffle trick used for the inter-group interaction in the group feed-forward layer;ȳ

where

M ×Dgroup are linear weights used in the low rank matrix factorization.

To explain the relationship from the shuffle trick, we describes the operations in the group feed-forward layer in a bottom-up way.

When we applying group-wise linear operations on the input features [x 1 , ...,x G ], the outputs are formed as [

By splitting each element into G groups and the shuffle operation perturbs the outputs as follows;

. . .

. . .

. . .

. . .

. . .

. . .

where the Shuffle operation transposes the first and second dimensions of G × G × M matrix and W

Dgroup×M is a linear weight describing information flow from the group g to the group g. Finally, a linear transformation with a weight W

. . .

where the Flatten operation vectorizes G×M matrix to G * M vector.

Therefore, the outputs (ȳ

In this section, we compare Group-Transformer from the original transformer in views of the numbers of parameters.

For a common expression, we denote D model ,D model , H as the feature size, the filter size in the feed-forward layer and the number of heads for the original transformer, and we set the feature size of a group as D group = D model /G, the filter sizeD group =D model /G, the number of heads in a group as H group = H/G for Group-Transformer.

In this calculation, we set the filter size as four times bigger than D model .

The multi-head attention of the original transformer uses 4D 2 model of parameters for the query, the key, the value, and the output.

The feature size for the multiple head is usually set as D model /H where H is the number of the heads.

Therefore, all transformations in the module is conducted for a D model -dimensional input feature to identify a D model -dimensional feature.

model .

When the number of groups is 2, the number of parameters of group attention is the same with those of the original transformer.

However, when the number of groups increases to 4 or 8, the number of the parameters decreases to 75% or 62.5% of the original module.

A point-wise feed-forward layer of the original transformer requires 8D

To see the effectiveness of our models on real-world applications, we have performed two generative tasks; word completion and sentence completion.

The former is to generate a word and the latter is to conclude a sentence when a given character sequence is in-complete.

Table 4 shows the generated top 20 words to conclude the in-complete character sequence, "pr".

Although our 6M and 4M Group-Transformers showed relatively lower scores (bpc) on the quantitative evaluations, as can be seen, the model still produces all real or plausible English words without a significant quality gap from the Transformer-XL with 41M parameters.

Seed: mary was not permitted to see them or to speak in her ···(abbreviate) ···proof of guilt if authentic the inquiry reached the conclusion that nothing was proven from the start this could have been pr ··· (Truth) predicted Transformer-XL (41M, 1.17 bpc on text8) proven, proved, proof, presented, proposed, probably, prevented, preceded, predicted, presumed, praised, preserved, problematic, preferred, present, previously, precisely, printed, produced, profound 2 Group-Transformer (6M, 1.26 bpc on text8) proven, proof, proved, proposed, present, previously, presented, preserved, printed, probably, practically, produced, prepared, prohibited, predicted, progressively, profound, primarily, problematic, practical 4 Group-Transformer (4M, 1.30 bpc on text8) proven, present, proposed, preserved, presented, previously, proved, practiced, produced, prepared, printed, probably, practically, provided, properly, presumed, praised, presently, prevented, primarily

The proposed method is focused on developing character-level language models, but the model can be applied to other NLP tasks.

When it comes to the word-level language modeling, compressing the word embedding layer becomes the most important part for designing a lightweight language model.

Therefore, we set a embedding dimension as 500 and adjusted the number of layers and the hidden dimension for the models to have the same number of parameters (4.5M).

Specifically, we set the bottleneck dimension as 4 times larger than the hidden dimension and follows other experimental settings of Dai et al. (2019) .

Table 7 : Performance comparison between the numbers of groups under the similar number of parameters.

We denote "L" and "D" as the number of layers and the hidden dimension, respectively.

We investigated the effects of all grouping methods in group attention.

Table 8 shows that the benefit on the parameter size is marginal compared to the performance drop when the number of grouping operations is increased.

On the other hand, there is a relatively small performance gap between grouping targets under the same number of grouping operations.

@highlight

This paper proposes a novel lightweight Transformer for character-level language modeling, utilizing group-wise operations.