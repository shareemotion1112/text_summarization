Neural models achieved considerable improvement for many natural language processing tasks, but they offer little transparency, and interpretability comes at a cost.

In some domains, automated predictions without justifications have limited applicability.

Recently, progress has been made regarding single-aspect sentiment analysis for reviews, where the ambiguity of a justification is minimal.

In this context, a justification, or mask, consists of (long) word sequences from the input text, which suffice to make the prediction.

Existing models cannot handle more than one aspect in one training and induce binary masks that might be ambiguous.

In our work, we propose a neural model for predicting multi-aspect sentiments for reviews and generates a probabilistic multi-dimensional mask (one per aspect) simultaneously, in an unsupervised and multi-task learning manner.

Our evaluation shows that on three datasets, in the beer and hotel domain, our model outperforms strong baselines and generates masks that are: strong feature predictors, meaningful, and interpretable.

Neural networks have become the standard for many natural language processing tasks.

Despite the significant performance gains achieved by these complex models, they offer little transparency concerning their inner workings.

Thus, they come at the cost of interpretability (Jain & Wallace, 2019).

In many domains, automated predictions have a real impact on the final decision, such as treatment options in the field of medicine.

Therefore, it is important to provide the underlying reasons for such a decision.

We claim that integrating interpretability in a (neural) model should supply the reason of the prediction and should yield better performance.

However, justifying a prediction might be ambiguous and challenging.

Prior work includes various methods that find the justification in an input text -also called rationale or mask of a target variable.

The mask is defined as one or multiple pieces of text fragments from the input text.

1 Each should contain words that altogether are short, coherent, and alone sufficient for the prediction as a substitute of the input (Lei et al., 2016) .

Many works have been applied to single-aspect sentiment analysis for reviews, where the ambiguity of a justification is minimal.

In this case, we define an aspect as an attribute of a product or service (Giannakopoulos et al., 2017) , such as Location or Cleanliness for the hotel domain.

There are three different methods to generate masks: using reinforcement learning with a trained model (Li et al., 2016b) , generating rationales in an unsupervised manner and jointly with the objective function (Lei et al., 2016) , or including annotations during training (Bao et al., 2018; Zhang et al., 2016) .

However, these models generate justifications that are 1) only tailored for one aspect, and 2) expressed as a hard (binary) selection of words.

A review text reflects opinions about multiple topics a user cares about (Musat et al., 2013) .

It appears reasonable to analyze multiple aspects with a multi-task learning setting, but a model must be trained as many times as the number of aspects.

A hard assignment of words to aspects might lead to ambiguities that are difficult to capture with a binary mask: in the text "The room was large, clean and close to the beach.", the word "room" refers to the aspects Room, Cleanliness and Location.

Finally, collecting human-provided rationales at scale is expensive and thus impractical.

In this work, we study interpretable multi-aspect sentiment classification.

We describe an architecture for predicting the sentiment of multiple aspects while generating a probabilistic (soft) multi-dimensional mask (one dimension per aspect) jointly, in an unsupervised and multi-task learning manner.

We show that the induced mask is beneficial for identifying simultaneously what parts of the review relate to what aspect, and capturing ambiguities of words belonging to multiple aspects.

Thus, the induced mask provides fine-grained interpretability and improves the final performance.

Traditionally interpretability came at a cost of reduced accuracy.

In contrast, our evaluation shows that on three datasets, in the beer and hotel domain, our model outperforms strong baselines and generates masks that are: strong feature predictors, meaningful, and interpretable compared to attention-based methods and a single-aspect masker.

We show that it can be a benefit to 1) guide the model to focus on different parts of the input text, and 2) further improve the sentiment prediction for all aspects.

Therefore, interpretabilty does not come at a cost anymore.

The contributions of this work can be summarized as follow:

??? We propose a Multi-Aspect Masker (MAM), an end-to-end neural model for multi-aspect sentiment classification that provides fine-grained interpretability in the same training.

Given a text review as input, the model generates a probabilistic multi-dimensional mask, with one dimension per aspect.

It predicts the sentiments of multiple aspects, and highlights long sequences of words justifying the current rating prediction for each aspect; ??? We show that interpretability does not come at a cost: our final model significantly outperforms strong baselines and attention models, both in terms of performance and mask coherence.

Furthermore, the level of interpretability is controllable using two regularizers; ??? Finally, we release a new dataset for multi-aspect sentiment classification, which contains 140k reviews from TripAdvisor with five aspects, each with its corresponding rating.

Developing interpretable models is of considerable interest to the broader research community, even more pronounced with neural models (Kim et al., 2015; Doshi-Velez & Kim, 2017) .

Many works analyzed and visualized state activation (Karpathy et al., 2015; Li et al., 2016a; Montavon et al., 2018) , learned sparse and interpretable word vectors (Faruqui et al., 2015b; a; Herbelot & Vecchi, 2015) or analyzed attention (Clark et al., 2019; Jain & Wallace, 2019) .

Our work differs from these in terms of what is meant by an explanation.

Our system identifies one or multiple short and coherent text fragments that -as a substitute of the input text -are sufficient for the prediction.

Attention models (Vaswani et al., 2017; Yang et al., 2016; Lin et al., 2017) have been shown to improve prediction accuracy, visualization, and interpretability.

The most popular and widely used attention mechanism is soft attention (Bahdanau et al., 2015) over hard (Luong et al., 2015) and sparse ones (Martins & Astudillo, 2016) .

According to Jain & Wallace (2019); Serrano & Smith (2019) , standard attention modules noisily predict input importance; the weights cannot provide safe and meaningful explanations.

Our model differs in two ways from attention mechanisms: our loss includes two regularizers to favor long word sequences for interpretability; the normalization is not done over the sequence length.

Review multi-aspect sentiment classification is sometimes seen as a sub-problem (Wang et al., 2010; McAuley et al., 2012; Pappas & Popescu-Belis, 2014) , by utilizing heuristic-based methods or topic models.

Recently, neural models achieved significant improvements with less feature engineering.

Yin et al. (2017) built a hierarchical attention model with aspect representations by using a set of manually defined topics.

Li et al. (2018) extended this work with user attention and additional features such as overall rating, aspect, and user embeddings.

The disadvantage of these methods is their limited interpretability, as they rely on many features in addition to the review text.

The idea of including human rationales during training has been explored in (Zhang et al., 2016; Marshall et al., 2015; Bao et al., 2018) .

Although they have been shown to be beneficial, they are expensive to collect and might vary across annotators.

In our work, no annotation is used.

The work most closely related to ours is Li et al. (2016b) and Lei et al. (2016) .

Both generate hard rationales and address single-aspect sentiment classification.

Their model must be trained separately for each aspect, which leads to ambiguities.

Li et al. (2016b) developed a post-training method that removes words from a review text until another trained model changes its prediction.

Lei et al. (2016) provides a model that learns an aspect sentiment and its rationale jointly, but hinders the performance and relies on assumptions on the data, such as a small correlation between aspect ratings.

In contrast, our model: 1) supports multi-aspect sentiment classification, 2) generates soft multidimensional masks in a single training; 3) the masks provide interpretability and improve the performance significantly.

Let X be a review composed of L words x 1 , x 2 , ..., x L and Y the target A-dimensional sentiment vector, corresponding to the different rated aspects.

Our proposed model, called Multi-Aspect Masker, is composed of three components: 1) a Masker module that computes a probability distribution over aspects for each word, resulting in A + 1 different masks (including one for not-aspect); 2) an Encoder that learns a representation of a review conditioned on the induced masks; 3) a Classifier that predicts the target variables.

The overall model architecture is shown in Figure 1 .

Our framework generalizes for other tasks, and each neural module is interchangeable with other models.

The Masker first computes a hidden representation h for each word x in the input sequence, using their word embeddings e 1 , e 2 , ..., e L .

Many sequence models could realize this task, such as recurrent, attention, or convolution neural networks.

In our case, we chose a convolutional network because it led to a smaller model, faster training, and empirically, performed similarly to recurrent models.

Let a i denote the i th aspect for i = 1, ..., A, and a 0 the not-aspect case, because many words can be irrelevant to every aspect.

We define M ??? R (A+1) , the aspect distribution of the input word x as:

Because we have categorical distributions, we cannot directly sample P (M |x ) and backpropagate the gradient through this discrete generation process.

Instead, we model the variable m ai using the Straight Through Gumbel Softmax (Jang et al., 2017; Maddison et al., 2017) , to approximate sampling from a categorical distribution.

We model the parameters of each Gumbel Softmax distribution M with a single-layer feedforward neural network followed by applying a log softmax, which induces the log-probabilities of the th distribution: ?? = log(softmax(W h + b)).

W and b are shared across all tokens, to have a constant number of parameters with respect to the sequence length.

We control the sharpness of the distributions with the temperature parameter ?? .

Compared to attention mechanisms, the word importance is a probability distribution over the targets:

T t=0 P (m at |x ) = 1, instead of a normalization over the sequence length,

We weight the word embeddings by their importance towards an aspect a i with the induced submasks, such that

Thereafter, each modified embedding E ai is fed into the Encoder block.

Note that E a0 is ignored because M a0 only serves to absorb probabilities of words that are insignificant to every aspect.

The Encoder module includes a convolutional neural network, for the same reasons as earlier, followed by a max-over-time pooling to obtain a fixed-length feature vector.

It produces the hidden representation h ai for each aspect a i .

To exploit commonalities and differences among aspects, we share the weights of the encoders for all E ai .

Finally, the Classifier block contains for each aspect a i a two-layer feedforward neural networks followed by a softmax layer to predict the sentiment?? ai .

Multi-Aspect Masker Trained on sent and no constraint Trained on sent with ?? p , sel , and cont i stayed at daulsol in september 2013 and could n't have asked for anymore for the price ! !

it is a great location ....

only 2 minutes walk to jet , space and sankeys with a short drive to ushuaia .

the hotel is basic but cleaned daily and i did nt have any problems at all with the bathroom or kitchen facilities .

the lady at reception was really helpful and explained everything we needed to know .....

even when we managed to miss our flight she let us stay around and use the facilities until we got on a later flight .

there are loads of restaurants in the vicinity and supermarkets and shops right outside .

i loved these apartments so much that i booked to come back for september 2014 ! !

can not wait :)

Aspect Changes: 30 i stayed at daulsol in september 2013 and could n't have asked for anymore for the price ! !

it is a great location ....

only 2 minutes walk to jet , space and sankeys with a short drive to ushuaia .

the hotel is basic but cleaned daily and i did nt have any problems at all with the bathroom or kitchen facilities .

the lady at reception was really helpful and explained everything we needed to know .....

even when we managed to miss our flight she let us stay around and use the facilities until we got on a later flight .

there are loads of restaurants in the vicinity and supermarkets and shops right outside .

i loved these apartments so much that i booked to come back for september 2014 ! !

can not wait :)

Aspect Changes: 5 Masks lead to mostly long sequences describing clearly each aspect (one switch per aspect), while attention to many short and interleaving sequences (30 changes between aspects), where most relate to noise or multiple aspects.

Highlighted words correspond to the highest aspect-attention scores above 1/L (i.e., more important than a uniform distribution), and the aspect a i maximizing P (m ai |x ).

The first objective to optimize is the sentiment loss, represented with the cross-entropy between the true aspect sentiment label y ai and the prediction?? ai :

Training Multi-Aspect Masker to optimize sent will lead to meaningless sub-masks M ai , as the model tends to focus on certain key-words.

Consequently, we guide the model to produce long and meaningful sequences of words, as shown in Figure 2 .

We propose two regularizers: the first controls the number of selected words, and the second favors consecutive words belonging to the

same aspect.

For the first term sel , we calculate the probability p sel of tagging a word as aspect and then compute the cross-entropy with a parameter ?? p .

The hyper-parameter ?? p can be interpreted as the prior on the number of selected words among all aspects, which corresponds to the expectation of Binomial(p sel ), as the optimizer will try to minimize the difference between p sel and ?? p .

The second regularizer discourages aspect transition between two consecutive words, by minimizing the mean variation of two consecutive aspect distributions.

We generalize the formulation in Lei et al. (2016) , from a hard binary single-aspect selection, to a soft probabilistic multi-aspect selection.

Finally, we train our Multi-Aspect Masker in an end-to-end manner, and optimize the final loss M AM = sent +?? sel ?? sel +?? cont ??

cont , where ?? sel and ?? cont control the impact of each constraint.

In this section, we assess our model on two dimensions: the predictive performance and the quality of the induced mask.

We first evaluate Multi-Aspect Masker on the multi-aspect sentiment classification task.

In a second experiment, we measure the quality of the induced sub-masks using aspect sentence-level annotations, and an automatic topic model evaluation method.

5 million beer reviews from BeerAdvocat.

Each contains multiple sentences describing various beer aspects: Appearance, Smell, Palate, and Taste; users also provided a five-star rating for each aspect.

Lei et al. (2016) modified the dataset to suit the requirements of their method.

5 As a consequence, the obtained subset, composed of 280k reviews, does not reflect the real data distribution: it contains only the first three aspects, and the sentiment correlation between any pair of aspects is decreased significantly (27.2% on average, instead of 71.8% originally).

We denote this version as the Filtered Beer dataset, and the original one as the Full Beer dataset.

To evaluate the robustness of models across domains, we crawled 140k hotel reviews from TripAdvisor.

Each review contains a five-star rating for each aspect: Service, Cleanliness, Value, Location, and Room.

The average correlation between aspects is high (63.0% on average).

Compared to beer reviews, hotel reviews are longer, noisier, and less structured, as shown in Appendix A.3.

As in Bao et al. (2018) , we binarize the problem: ratings at three and above are labeled as positive and the rest as negative.

We further divide the datasets into 80/10/10 for train, development, and test subsets (more details in Appendix A.1).

We compared our Multi-Aspect Masker (MAM) with various baselines.

We first used a simple baseline, Sentiment Majority, that reports the majority sentiment across aspects, as the sentiment correlation between aspects might be high (see Section 4.1).

Because this information is not available at testing, we trained a model to predict the majority sentiment of a review using Wang & Manning (2012) .

The second baseline we used is a shared encoder followed by A classifiers, that we denote Emb + Enc + Clf.

This model does not offer any interpretability.

We extended it with a shared attention mechanism (Bahdanau et al., 2015) after the encoder, noted A Shared , that provides a coarse-grained interpretability: for all aspects, the network focuses on the same words in the input.

Our final goal is to achieve the best performance and provide fine-grained interpretability: to visualize what sequences of words a model focuses on and to predict the aspect sentiment predictions.

To this end, we included other baselines: two trained separately for each aspect and two trained with a multi-aspect sentiment loss.

We employed for the first ones: the well-known NB-SVM of Wang & Manning (2012) for sentiment analysis tasks, and the Single Aspect-Mask (SAM) model from Lei et al. (2016) , each trained separately for each aspect.

The two last methods are composed of a separate encoder, attention mechanism, and classifier for each aspect.

We utilized two types of attention mechanism: additive (Bahdanau et al., 2015) , and sparse (Martins & Astudillo, 2016) .

We call each variant Multi Aspect-Attentions (MAA) and Multi Aspect-Sparse-Attentions (MASA).

Diagrams for the baselines can be found in Appendix A.5.

In this section, we enquire whether fine-grained interpretability can become a benefit rather than a cost.

We group the models and baselines in three different levels of interpretability:

??? None:

we cannot identify what parts of the review are important for the prediction;

??? Coarse-grained: we can identify what parts of the reviews were important to predict all aspect sentiments, without knowing what part corresponds to what aspect;

??? Fine-grained: for each aspect, we can identify what parts are used to predict its sentiment.

Overall F1 scores (macro and for each aspect A i ) for the controlled-environment Filtered Beer (where there are assumptions on the data distribution) and the real-world Full Beer dataset are shown in Table 1 and Table 2 .

We find that our Multi-Aspect Masker (MAM) model 8 , with 1.7 to 2.1 times fewer parameters than aspect-wise attention models ( 6 + 7 ), performs better on average than all other baselines on both datasets, and provides fine-grained interpretability.

For the synthetic Filtered Beer dataset, MAM achieves a significant improvement of at least 0.36 macro F1 score, and 0.05 for the Full Beer one.

To demonstrate that the induced sub-masks M a1 , ..., M a A are 1) meaningful for other models to improve final predictions, and 2) bring fine-grained interpretability, we extracted and concatenated the masks to the word embeddings, resulting in contextualized embeddings (Peters et al., 2018) .

We trained a simple Encoder-Classifier with the contextualized embeddings 9 , which has approximately 1.5 times fewer parameters than MAM.

We achieved a macro F1 score absolute improvement of 0.34 compared to MAM, and 1.43 compared to the non-contextualized variant for the Filtered Beer dataset; for the Full Beer one, the performance increases by 0.39 and 2.49 respectively.

Similarly, each individual aspect A i F1 score of MAM is improved to a similar extent.

We provide in Appendix A.3.1 and A.3.2 visualizations of reviews with the computed sub-masks M a1 , ..., M a A and attentions by different models.

Not only do sub-masks enable the reach of higher performance; they better capture parts of reviews related to each aspect compared to other methods.

Both NB-SVM 4 and SAM 5 , offering fine-grained interpretability and trained separately for each aspect, significantly underperform compared to the Encoder-Classifier 1 .

This result is expected: the goal of SAM is to provide rationales at the price of performance (Lei et al., 2016) , and NB-SVM might not perform well because of its simplicity.

Shared attention models ( 2 + 3 ) perform similarly to the Encoder-Classifier 1 , but provide only coarse-grained interpretability.

However, in the Full Beer dataset, SAM 5 obtains better results than the Encoder-Classifier baseline 1 and NB-SVM 4 , which is outperformed by all other models.

It might be counterintuitive that SAM performs better, but we claim that its behavior comes from the high correlation between aspects: SAM selects words that should belong to aspect a i to predict the sentiment of aspect a j (a i = a j ).

Moreover, in Section 4.5, we show that a single-aspect mask from SAM cannot be employed for interpretability.

Finally, Sentiment Majority 0 is outperformed by a large margin by all other models in the Filtered Beer dataset, because of the low sentiment correlation between aspects.

However, in the realistic dataset Full Beer, Sentiment Majority obtains higher score and performs better than NB-SVM 4 .

These results emphasize the ease of the Filtered Beer dataset compared to the Full Beer one, because of the assumptions not holding in the real data distribution.

On the Hotel dataset, the learned mask M from Multi-Aspect Masker 8 is again meaningful, by increasing the performance and adding interpretability.

The Encoder-Classifier with contextualized embeddings 9 outperforms all other models significantly, with an absolute macro F1 score improvement of 0.49.

Moreover, it achieves the best individual F1 score for each aspect A 1 , ..., A 5 .

Visualizations of reviews, with masks and attentions, are available in Appendix A.3.3.

The interpretability comes from the long sequences that MAM identifies, unlike attention models.

In comparison, SAM 5 lacks coverage and suffers from ambiguity due to the high correlation between aspects.

We observe that Multi-Aspect Masker 8 performs slightly worse than aspect-wise atten- tion models ( 7 + 8 ), with 2.5 times fewer parameters.

We emphasize that using the induced masks in the Encoder-Classifier 9 already achieves the best performance.

The Single Aspect-Mask 5 obtains the lowest relative macro F1 score of the three datasets: a difference of ???3.27; ???2.6 and ???2.32 for the Filtered Beer and Full Beer dataset respectively.

This proves that the model is not meant to provide rationales and increase the performance simultaneously.

Finally, we show that learning soft multi-dimensional masks along training objectives achieves strong predictive results, and using these to create contextualized word embeddings and train a baseline model with, provides the best performance across the three datasets.

In these experiments, we verify that Multi-Aspect Masker generates induced masks M a1 , ..., M a A that, in addition to improving performance, are meaningful and can be interpreted by humans.

Evaluating justifications that have short and coherent pieces of text is challenging because there is no gold standard provided with reviews.

McAuley et al. (2012) provided 994 beer reviews with aspect sentence-level annotations, although our model computes masks at a finer level.

Each sentence of the (2016), we used trained models on beer reviews and extracted a similar number of selected words.

We show that the generated sub-masks M a1 , M a2 , M a3 obtained with Multi-Aspect Masker (MAM) correlate best with human judgment.

Table 4 presents the precision of the masks and attentions computed on sentence-level aspect annotations.

We reported results of the models in Lei et al. (2016) : SVM, the Single Aspect-Attention (SAA) and Single Aspect-Mask (SAM) -trained separately for each aspect because they find hard justifications for a single aspect.

In comparison to SAM, our MAM model obtains significant higher precisions with an average of +1.13.

Interestingly, SVM and attention models perform poorly compared with mask models: especially MASA that focuses only on a couple of words due to the sparseness of the attention (examples in Appendix A.3.1).

In addition to evaluating masks with human annotations, we computed their semantic interpretability for each dataset.

According to Aletras & Stevenson (2013); Lau et al. (2014) , NPMI (Bouma, 2009 ) is a good metric for qualitative evaluation of topics, because it matches human judgment most closely.

However, the top-N topic words, used for evaluation, are often selected arbitrarily.

To alleviate this problem, we followed Lau & Baldwin (2016): we computed the topic coherence over several cardinalities N , and report all the results, as well as the average; the authors claim the mean leads to a more stable and robust evaluation.

More details are available in Appendix A.4.

We show that generated masks by MAM obtains the highest mean NPMI and, on average, superior results in all datasets (17 out of 21 cases), while only needing a single training.

Results are shown in Table 5 .

For the Hotel and Full Beer datasets, where reviews reflect the real data distribution, our model significantly outperforms SAM and attention models for N ??? 20.

For smaller N , MAM obtains higher scores in four out of six cases, and for these two, the difference is only below 0.003.

For the controlled-environment Filtered Beer dataset, MAM still performs better for N ??? 15, although the differences are smaller, and is beat by SAM for N ??? 10.

However, SAM obtains poor results in all other cases of all datasets and must be trained as many times as the number of aspects.

We show the top words for each aspect and a human evaluation in Appendix A.4.

Generally, our model finds better sets of words among the three datasets compared with other methods.

In this work, we propose Multi-Aspect Masker, an end-to-end neural network architecture to perform multi-aspect sentiment classification for reviews.

Our model predicts aspect sentiments while generating a probabilistic (soft) multi-dimensional mask (one dimension per aspect) simultaneously, in an unsupervised and multi-task learning manner.

We showed that the induced mask is beneficial to guide the model to focus on different parts of the input text and to further improve the sentiment prediction for all aspects.

Our evaluation shows that on three datasets, in the beer and hotel domain, our model outperforms strong baselines and generates masks that are: strong feature predictors, meaningful, and interpretable compared to attention-based methods and a single-aspect masker.

Nikolaos Aletras and Mark Stevenson.

Evaluating topic coherence using distributional semantics.

In (Nair & Hinton, 2010) .

We used the 200-dimensional pretrained word embeddings of Lei et al. (2016) for beer reviews.

For the hotel domain, we trained word2vec (Mikolov et al., 2013) on a large collection of hotel reviews with an embedding size of 300.

We used dropout (Srivastava et al., 2014) of 0.1, clipped the gradient norm at 1.0 if higher, added L2-norm regularizer with a regularization factor of 10 ???6 and trained using early stopping with a patience of three iterations.

We used Adam (Kingma & Ba, 2015) for training with a learning rate of 0.001, ?? 1 = 0.9, and ?? 2 = 0.999.

The temperature ?? for Gumbel-Softmax distributions was fixed at 0.8.

The two regularizer terms and the prior of our model are ?? sel = 0.03, ?? cont = 0.04, and ?? p = 0.11 for the Filtered Beer dataset; ?? sel = 0.03, ?? cont = 0.03, and ?? p = 0.15 for the Full Beer dataseet; and ?? sel = 0.02, ?? cont = 0.02 and ?? p = 0.10 for the Hotel dataset.

We ran all experiments for a maximum of 50 epochs with a batch-size of 256 and a Titan X GPU.

For the model of Lei et al. (2016) , we reused the code from the authors.

We randomly sampled reviews from each dataset and computed the masks and attentions of four models: our Multi-Aspect Masker model (MAM), the Single Aspect-Mask method (SAM) of Lei et al. (2016) and two attention models with additive and sparse attention, called Multi AspectAttentions (MAA) and Multi Aspect-Sparse-Attentions (MASA) respectively (more details in Section 4.2).

Each color represents an aspect and the shade its confidence.

All models generate soft attentions or masks besides SAM, which does hard masking.

Samples for the Filtered Beer, Full Beer and Hotel dataset are shown below.

Multi Aspect-Masks (Ours) a : ruby red brown in color .

fluffy off white single -finger head settles down to a thin cap .

coating thin lacing all over the sides on the glass .

s : some faint burnt , sweet malt smells , but little else and very faint .

t : taste is very solid for a brown .

malts and some sweetness .

maybe some toffee , biscuit and burnt flavors too .

m : decent carbonation is followed by a medium bodied feel .

flavor coats the tongue for a very satisfying and lasting finish .

d : an easy drinker , as a good brown should be .

my wife is a big brown fan , so i 'll definitely be grabbing this one for her again .

a solid beer for any time of the year . served : in a standard pint glass .

Single Aspect-Mask (Lei et al., 2016) a : ruby red brown in color .

fluffy off white single -finger head settles down to a thin cap .

coating thin lacing all over the sides on the glass .

s : some faint burnt , sweet malt smells , but little else and very faint .

t : taste is very solid for a brown .

malts and some sweetness .

maybe some toffee , biscuit and burnt flavors too .

m : decent carbonation is followed by a medium bodied feel .

flavor coats the tongue for a very satisfying and lasting finish .

d : an easy drinker , as a good brown should be .

my wife is a big brown fan , so i 'll definitely be grabbing this one for her again .

a solid beer for any time of the year . served : in a standard pint glass .

Multi Aspect-Attentions a : ruby red brown in color .

fluffy off white single -finger head settles down to a thin cap .

coating thin lacing all over the sides on the glass .

s : some faint burnt , sweet malt smells , but little else and very faint .

t : taste is very solid for a brown .

malts and some sweetness .

maybe some toffee , biscuit and burnt flavors too .

m : decent carbonation is followed by a medium bodied feel .

flavor coats the tongue for a very satisfying and lasting finish .

d : an easy drinker , as a good brown should be .

my wife is a big brown fan , so i 'll definitely be grabbing this one for her again .

a solid beer for any time of the year . served : in a standard pint glass .

Multi Aspect-Sparse-Attentions a : ruby red brown in color .

fluffy off white single -finger head settles down to a thin cap .

coating thin lacing all over the sides on the glass .

s : some faint burnt , sweet malt smells , but little else and very faint .

t : taste is very solid for a brown .

malts and some sweetness .

maybe some toffee , biscuit and burnt flavors too .

m : decent carbonation is followed by a medium bodied feel .

flavor coats the tongue for a very satisfying and lasting finish .

d : an easy drinker , as a good brown should be .

my wife is a big brown fan , so i 'll definitely be grabbing this one for her again .

a solid beer for any time of the year . served : in a standard pint glass .

Figure 3: Our model MAM highlights all the words corresponding to aspects.

SAM only highlights the most crucial information, but some words are missing out, and one is ambiguous.

MAA and MASA fail to identify most of the words related to the aspect Appearance, and only a few words have high confidence, resulting in noisy labeling.

Additionally, MAA considers words belonging to the aspect Taste whereas the Filtered Beer dataset does not include it in the aspect set.

Multi Aspect-Masks (Ours) a-crystal clear gold , taunt fluffy three finger white head that holds it own very well , when it falls it falls to a 1/2 " ring , full white lace on glass s-clean , crisp , floral , pine , citric lemon t-crisp biscuit malt up front , hops all the way through , grassy , lemon , tart yeast at finish , hop bitterness through finish m-dry , bubbly coarse , high carbonation , light bodied , hops leave impression on palette .

d-nice hop bitterness , good flavor , sessionable , recommended , good brew

Single Aspect-Mask (Lei et al., 2016) a-crystal clear gold , taunt fluffy three finger white head that holds it own very well , when it falls it falls to a 1/2 " ring , full white lace on glass s-clean , crisp , floral , pine , citric lemon t-crisp biscuit malt up front , hops all the way through , grassy , lemon , tart yeast at finish , hop bitterness through finish m-dry , bubbly coarse , high carbonation , light bodied , hops leave impression on palette .

d-nice hop bitterness , good flavor , sessionable , recommended , good brew

Multi Aspect-Attentions a-crystal clear gold , taunt fluffy three finger white head that holds it own very well , when it falls it falls to a 1/2 " ring , full white lace on glass s-clean , crisp , floral , pine , citric lemon t-crisp biscuit malt up front , hops all the way through , grassy , lemon , tart yeast at finish , hop bitterness through finish m-dry , bubbly coarse , high carbonation , light bodied , hops leave impression on palette .

d-nice hop bitterness , good flavor , sessionable , recommended , good brew

Multi Aspect-Sparse-Attentions a-crystal clear gold , taunt fluffy three finger white head that holds it own very well , when it falls it falls to a 1/2 " ring , full white lace on glass s-clean , crisp , floral , pine , citric lemon t-crisp biscuit malt up front , hops all the way through , grassy , lemon , tart yeast at finish , hop bitterness through finish m-dry , bubbly coarse , high carbonation , light bodied , hops leave impression on palette .

d-nice hop bitterness , good flavor , sessionable , recommended , good brew Figure 4 : MAM finds the exact parts corresponding to the aspect Appearance and Palate while covering most of the aspect Smell.

SAM identifies key-information without any ambiguity, but lacks coverage.

MAA highlights confidently nearly all the words while having some noise for the aspect Appearance.

MASA selects confidently only most predictive words.

Multi Aspect-Masks (Ours) sa 's harvest pumpkin ale 2011 .

had this last year , loved it , and bought 6 harvest packs and saved the pumpkins and the dunkel 's ... not too sure why sa dropped the dunkel , i think it would make a great standard to them .

pours a dark brown with a 1 " bone white head , that settles down to a thin lace across the top of the brew .

smells of the typical pumpkin pie spice , along with a good squash note .

tastes just like last years , very subtle , nothing over the top .

a damn good pumpkin ale that is worth seeking out .

when i mean everything is subtle i mean everything .

nothing is overdone in this pumpkin ale , and is a great representation of the original style .

mouthfeel is somewhat thick , with a pleasant coating feel .

overall , i loved it last year , and i love it this year .

do n't get me wrong , its no pumpking , but this is a damn fine pumpkin ale that could hold its own any day among all the others .

i would rate this as my 4th favorite pumpkin ale to date .

i 'm not sure why the bros rated it so low , but do n't take their opinion , make your own !

Appearance Smell Palate Taste Single Aspect-Mask (Lei et al., 2016) sa 's harvest pumpkin ale 2011 . had this last year , loved it , and bought 6 harvest packs and saved the pumpkins and the dunkel 's ... not too sure why sa dropped the dunkel , i think it would make a great standard to them .

pours a dark brown with a 1 " bone white head , that settles down to a thin lace across the top of the brew .

smells of the typical pumpkin pie spice , along with a good squash note .

tastes just like last years , very subtle , nothing over the top .

a damn good pumpkin ale that is worth seeking out .

when i mean everything is subtle i mean everything .

nothing is overdone in this pumpkin ale , and is a great representation of the original style .

mouthfeel is somewhat thick , with a pleasant coating feel .

overall , i loved it last year , and i love it this year .

do n't get me wrong , its no pumpking , but this is a damn fine pumpkin ale that could hold its own any day among all the others .

i would rate this as my 4th favorite pumpkin ale to date .

i 'm not sure why the bros rated it so low , but do n't take their opinion , make your own !

Appearance Smell Palate Taste Multi Aspect-Attentions sa 's harvest pumpkin ale 2011 .

had this last year , loved it , and bought 6 harvest packs and saved the pumpkins and the dunkel 's ... not too sure why sa dropped the dunkel , i think it would make a great standard to them .

pours a dark brown with a 1 " bone white head , that settles down to a thin lace across the top of the brew .

smells of the typical pumpkin pie spice , along with a good squash note .

tastes just like last years , very subtle , nothing over the top .

a damn good pumpkin ale that is worth seeking out .

when i mean everything is subtle i mean everything .

nothing is overdone in this pumpkin ale , and is a great representation of the original style .

mouthfeel is somewhat thick , with a pleasant coating feel .

overall , i loved it last year , and i love it this year .

do n't get me wrong , its no pumpking , but this is a damn fine pumpkin ale that could hold its own any day among all the others .

i would rate this as my 4th favorite pumpkin ale to date .

i 'm not sure why the bros rated it so low , but do n't take their opinion , make your own !

Appearance Smell Palate Taste Multi Aspect-Sparse-Attentions sa 's harvest pumpkin ale 2011 .

had this last year , loved it , and bought 6 harvest packs and saved the pumpkins and the dunkel 's ... not too sure why sa dropped the dunkel , i think it would make a great standard to them .

pours a dark brown with a 1 " bone white head , that settles down to a thin lace across the top of the brew .

smells of the typical pumpkin pie spice , along with a good squash note .

tastes just like last years , very subtle , nothing over the top .

a damn good pumpkin ale that is worth seeking out .

when i mean everything is subtle i mean everything .

nothing is overdone in this pumpkin ale , and is a great representation of the original style .

mouthfeel is somewhat thick , with a pleasant coating feel .

overall , i loved it last year , and i love it this year .

do n't get me wrong , its no pumpking , but this is a damn fine pumpkin ale that could hold its own any day among all the others .

i would rate this as my 4th favorite pumpkin ale to date .

i 'm not sure why the bros rated it so low , but do n't take their opinion , make your own !

Figure 5 : MAM can identify accurately what parts of the review describe each aspect.

Due to the high imbalance and correlation, MAA provides very noisy labels, while MASA highlights only a few important words.

We can see that SAM is confused and performs a poor selection.

Multi Aspect-Masks (Ours) 75cl bottle shared with larrylsb , pre -grad .

bright , hazy gold with a big white head .

the flavor has bursting fruit and funky yeast with tropical and peach standing out .

the flavor has the same intense fruitiness , with a funky , lightly tart edge , and a nice hop balance .

dry and refreshing on the tongue .

medium bodied with perfect carbonation that livens up the palate .

this was just beautiful stuff that i 'm already craving more of .

Single Aspect-Mask (Lei et al., 2016) 75cl bottle shared with larrylsb , pre -grad .

bright , hazy gold with a big white head .

the flavor has bursting fruit and funky yeast with tropical and peach standing out .

the flavor has the same intense fruitiness , with a funky , lightly tart edge , and a nice hop balance .

dry and refreshing on the tongue .

medium bodied with perfect carbonation that livens up the palate .

this was just beautiful stuff that i 'm already craving more of .

Multi Aspect-Attentions 75cl bottle shared with larrylsb , pre -grad .

bright , hazy gold with a big white head .

the flavor has bursting fruit and funky yeast with tropical and peach standing out .

the flavor has the same intense fruitiness , with a funky , lightly tart edge , and a nice hop balance .

dry and refreshing on the tongue .

medium bodied with perfect carbonation that livens up the palate .

this was just beautiful stuff that i 'm already craving more of .

Multi Aspect-Sparse-Attentions 75cl bottle shared with larrylsb , pre -grad .

bright , hazy gold with a big white head .

the flavor has bursting fruit and funky yeast with tropical and peach standing out .

the flavor has the same intense fruitiness , with a funky , lightly tart edge , and a nice hop balance .

dry and refreshing on the tongue .

medium bodied with perfect carbonation that livens up the palate .

this was just beautiful stuff that i 'm already craving more of .

i stayed at daulsol in september 2013 and could n't have asked for anymore for the price ! !

it is a great location ....

only 2 minutes walk to jet , space and sankeys with a short drive to ushuaia .

the hotel is basic but cleaned daily and i did nt have any problems at all with the bathroom or kitchen facilities .

the lady at reception was really helpful and explained everything we needed to know .....

even when we managed to miss our flight she let us stay around and use the facilities until we got on a later flight .

there are loads of restaurants in the vicinity and supermarkets and shops right outside .

i loved these apartments so much that i booked to come back for september 2014 ! !

can not wait :)

Single Aspect-Mask (Lei et al., 2016) i stayed at daulsol in september 2013 and could n't have asked for anymore for the price ! !

it is a great location ....

only 2 minutes walk to jet , space and sankeys with a short drive to ushuaia .

the hotel is basic but cleaned daily and i did nt have any problems at all with the bathroom or kitchen facilities .

the lady at reception was really helpful and explained everything we needed to know .....

even when we managed to miss our flight she let us stay around and use the facilities until we got on a later flight .

there are loads of restaurants in the vicinity and supermarkets and shops right outside .

i loved these apartments so much that i booked to come back for september 2014 ! !

can not wait :)

Multi Aspect-Attentions i stayed at daulsol in september 2013 and could n't have asked for anymore for the price ! !

it is a great location ....

only 2 minutes walk to jet , space and sankeys with a short drive to ushuaia .

the hotel is basic but cleaned daily and i did nt have any problems at all with the bathroom or kitchen facilities .

the lady at reception was really helpful and explained everything we needed to know .....

even when we managed to miss our flight she let us stay around and use the facilities until we got on a later flight .

there are loads of restaurants in the vicinity and supermarkets and shops right outside .

i loved these apartments so much that i booked to come back for september 2014 ! !

can not wait :)

Multi Aspect-Sparse-Attentions i stayed at daulsol in september 2013 and could n't have asked for anymore for the price ! !

it is a great location ....

only 2 minutes walk to jet , space and sankeys with a short drive to ushuaia .

the hotel is basic but cleaned daily and i did nt have any problems at all with the bathroom or kitchen facilities .

the lady at reception was really helpful and explained everything we needed to know .....

even when we managed to miss our flight she let us stay around and use the facilities until we got on a later flight .

there are loads of restaurants in the vicinity and supermarkets and shops right outside .

i loved these apartments so much that i booked to come back for september 2014 ! !

can not wait :) Figure 7 : MAM emphasizes consecutive words, identifies important spans while having a small amount of noise.

SAM focuses on certain specific words and spans, but labels are ambiguous.

The MAA model highlights many words, ignores a few important key-phrases, and labels are noisy when the confidence is not high.

MASA provides noisier tags than MAA.

Multi-Aspect Masker (Ours) stayed at the parasio 10 apartments early april 2011 .

reception staff absolutely fantastic , great customer service .. ca nt fault at all !

we were on the 4th floor , facing the front of the hotel ..

basic , but nice and clean .

good location , not too far away from the strip and beach ( 10 min walk ) .

however .. do not go out alone at night at all !

i went to the end of the street one night and got mugged ..

all my money , camera ..

everything ! got sratches on my chest which has now scarred me , and i had bruises at the time .

just make sure you have got someone with you at all times , the local people are very renound for this . went to police station the next day ( in old town ) and there was many english in there reporting their muggings from the day before .

shocking ! !

apart from this incident ( on the first night we arrived :( )

we had a good time in the end , plenty of laughs and everything is very cheap !

beer -1euro !

fryups -2euro .

would go back again , but maybe stay somewhere else closer to the beach ( sol pelicanos etc ) ..

this hotel is next to an alley called ' muggers alley '

Single Aspect-Mask (Lei et al., 2016) stayed at the parasio 10 apartments early april 2011 .

reception staff absolutely fantastic , great customer service .. ca nt fault at all !

we were on the 4th floor , facing the front of the hotel ..

basic , but nice and clean .

good location , not too far away from the strip and beach ( 10 min walk ) .

however .. do not go out alone at night at all !

i went to the end of the street one night and got mugged ..

all my money , camera ..

everything ! got sratches on my chest which has now scarred me , and i had bruises at the time .

just make sure you have got someone with you at all times , the local people are very renound for this . went to police station the next day ( in old town ) and there was many english in there reporting their muggings from the day before .

shocking ! !

apart from this incident ( on the first night we arrived :( )

we had a good time in the end , plenty of laughs and everything is very cheap !

beer -1euro !

fryups -2euro .

would go back again , but maybe stay somewhere else closer to the beach ( sol pelicanos etc ) ..

this hotel is next to an alley called ' muggers alley '

Multi Aspect-Attentions stayed at the parasio 10 apartments early april 2011 .

reception staff absolutely fantastic , great customer service .. ca nt fault at all !

we were on the 4th floor , facing the front of the hotel ..

basic , but nice and clean .

good location , not too far away from the strip and beach ( 10 min walk ) .

however .. do not go out alone at night at all !

i went to the end of the street one night and got mugged ..

all my money , camera ..

everything ! got sratches on my chest which has now scarred me , and i had bruises at the time .

just make sure you have got someone with you at all times , the local people are very renound for this . went to police station the next day ( in old town ) and there was many english in there reporting their muggings from the day before .

shocking ! !

apart from this incident ( on the first night we arrived :( )

we had a good time in the end , plenty of laughs and everything is very cheap !

beer -1euro !

fryups -2euro .

would go back again , but maybe stay somewhere else closer to the beach ( sol pelicanos etc ) ..

this hotel is next to an alley called ' muggers alley '

Multi Aspect-Sparse-Attentions stayed at the parasio 10 apartments early april 2011 .

reception staff absolutely fantastic , great customer service .. ca nt fault at all !

we were on the 4th floor , facing the front of the hotel ..

basic , but nice and clean .

good location , not too far away from the strip and beach ( 10 min walk ) .

however .. do not go out alone at night at all !

i went to the end of the street one night and got mugged ..

all my money , camera ..

everything ! got sratches on my chest which has now scarred me , and i had bruises at the time .

just make sure you have got someone with you at all times , the local people are very renound for this . went to police station the next day ( in old town ) and there was many english in there reporting their muggings from the day before .

shocking ! !

apart from this incident ( on the first night we arrived :( )

we had a good time in the end , plenty of laughs and everything is very cheap !

beer -1euro !

fryups -2euro .

would go back again , but maybe stay somewhere else closer to the beach ( sol pelicanos etc ) ..

this hotel is next to an alley called ' muggers alley '

Figure 8: Our MAM model finds most of the important span of words with a small amount of noise.

SAM lacks coverage but identifies words where half are correctly tags and the others ambiguous.

MAA partially correctly highlights words for the aspects Service, Location, and Value while missing out the aspect Cleanliness.

MASA confidently finds a few important words.

For each model, we computed the probability distribution of words per aspect by using the induced sub-masks M a1 , ..., M a A or attention values.

Given an aspect a i and a set of top-N words w N ai , the Normalized Pointwise Mutual Information (Bouma, 2009) coherence score is:

Top words of coherent topics (i.e., aspects) should share a similar semantic interpretation and thus, interpretability of a topic can be estimated by measuring how many words are not related.

For each aspect a i and word w having been highlighted at least once as belonging to aspect a i , we computed the probability P (w|a i ) on each dataset and sorted them in decreasing order of P (w|a i ).

Unsurprisingly, we found that the most common words are stop words such as "a" and "it", because masks are mostly word sequences instead of individual words.

To gain a better interpretation of the aspect words, we followed the procedure in McAuley et al. (2012): we first computed averages across all aspect words for each word w: b w = 1 |A| |A| i=1 P (w|a i ), which generates a general distribution that includes words common to all aspects.

The final word distribution per aspect is computed by removing the general distribution:P (w|a i ) = P (w|a i ) ??? b w .

After generating the final word distribution per aspect, we picked the top ten words and asked two human annotators to identify intruder words, i.e., words not matching the corresponding aspect.

We show in subsequent tables the top ten words for each aspect, where red denotes all words identified as unrelated to the aspect by the two annotators.

Generally, our model finds better sets of words across the three datasets compared with other methods.

Additionally, we observe that the aspects can be easily recovered given its top words.

Table 7 : Top ten words for each aspect from the Filtered Beer dataset, learned by various models.

Red denotes intruders according to two human annotators.

For the three aspects, MAM has only one word considered as an intruder, followed by MASA with SAM (two) and MAA (six).

Top-10 Words Appearance SAM head color white brown dark lacing pours amber clear black MASA head lacing lace retention glass foam color amber yellow cloudy MAA nice dark amber pours black hazy brown great cloudy clear MAM (Ours) head color lacing white brown clear amber glass black retention Smell SAM sweet malt hops coffee chocolate citrus hop strong smell aroma MASA smell aroma nose smells sweet aromas scent hops malty roasted MAA taste smell aroma sweet chocolate lacing malt roasted hops nose MAM (Ours) smell aroma nose smells sweet malt citrus chocolate caramel aromas Palate SAM mouthfeel smooth medium carbonation bodied watery body thin creamy full MASA mouthfeel medium smooth body nice m-feel bodied mouth beer MAA carbonation mouthfeel medium overall smooth finish body drinkability bodied watery MAM (Ours) mouthfeel carbonation medium smooth body bodied drinkability good mouth thin

@highlight

Neural model predicting multi-aspect sentiments and generating a probabilistic multi-dimensional mask simultaneously. Model outperforms strong baselines and generates masks that are: strong feature predictors, meaningful, and interpretable.