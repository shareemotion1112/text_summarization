Contextualized word representations such as ELMo and BERT have become the de facto starting point for incorporating pretrained representations for downstream NLP tasks.

In these settings, contextual representations have largely made obsolete their static embedding predecessors such as Word2Vec and GloVe.

However, static embeddings do have their advantages in that they are straightforward to understand and faster to use.

Additionally, embedding analysis methods for static embeddings are far more diverse and mature than those available for their dynamic counterparts.

In this work, we introduce simple methods for generating static lookup table embeddings from existing pretrained contextual representations and demonstrate they outperform Word2Vec and GloVe embeddings on a variety of word similarity and word relatedness tasks.

In doing so, our results also reveal insights that may be useful for subsequent downstream tasks using our embeddings or the original contextual models.

Further, we demonstrate the increased potential for analysis by applying existing approaches for estimating social bias in word embeddings.

Our analysis constitutes the most comprehensive study of social bias in contextual word representations (via the proxy of our distilled embeddings) and reveals a number of inconsistencies in current techniques for quantifying social bias in word embeddings.

We publicly release our code and distilled word embeddings to support reproducible research and the broader NLP community.

Word embeddings (Bengio et al., 2003; Collobert & Weston, 2008; Collobert et al., 2011) have been a hallmark of modern natural language processing (NLP) for several years.

Pretrained embeddings in particular have seen widespread use and have experienced parallel and complementary innovations alongside neural networks for NLP.

Advances in embedding quality in part have come from integrating additional information such as syntax (Levy & Goldberg, 2014b; Li et al., 2017) , morphology (Cotterell & Schütze, 2015) , subwords (Bojanowski et al., 2017) , subcharacters (Stratos, 2017; Yu et al., 2017) and, most recently, context (Peters et al., 2018; Devlin et al., 2019) .

As a consequence of their representational potential, pretrained word representations have seen widespread adoption across almost every task in NLP and reflect one of the greatest successes of both representation learning and transfer learning for NLP (Ruder, 2019b) .

The space of pretrained word representations can be partitioned into static vs. dynamic embeddings methods.

Static methods such as Word2Vec (Mikolov et al., 2013) , GloVe (Pennington et al., 2014), and FastText (Bojanowski et al., 2017) yield representations that are fixed after training and generally associate a single vector with a given word in the style of a lookup table.

While subsequent work addressed the fact that words may have multiple senses and should have different representations for different senses (Pilehvar & Collier, 2016; Lee & Chen, 2017; Pilehvar et al., 2017; Athiwaratkun & Wilson, 2017; Camacho-Collados & Pilehvar, 2018) , fundamentally these methods cannot easily adapt to the inference time context in which they are applied.

This contrasts with contextual, or dynamic, methods such as CoVe (McCann et al., 2017) , ELMo (Peters et al., 2018) , and BERT (Devlin et al., 2019) , which produce vector representations for a word conditional on the inference time context in which it appears.

Given that dynamic representations are arguably more linguistically valid, more expressive (static embeddings are a special-case of dynamic embeddings that are optimally ineffective at being dynamic), and have yielded significant empirical improvements (Wang et al., 2019b; a; Ruder, 2019a) , it would seem that static embeddings are outdated.

Static embeddings, however, have significant advantages over dynamic embeddings with regard to speed, computational resources, and ease of use.

These benefits have important implications for time-sensitive systems, resource-constrained settings or environmental concerns (Strubell et al., 2019) , and broader accessibility of NLP technologies 1 .

As a consequence of this dichotomy between static and dynamic representations and their disparate benefits, we propose in this work a simple yet effective mechanism for converting from dynamic representations to static representations.

We begin by demonstrating that our method when applied to pretrained contextual models (BERT, GPT-2, RoBERTa, XLNet, DistilBERT) yields higher quality static embeddings than Word2Vec and GloVe when evaluated intrinsically on four word similarity and word relatedness datasets.

Further, since our procedure does not rely on specific properties of the pretrained contextual model, it can be applied as needed to generate ever-improving static embeddings that will track advances in pretrained contextual word representations.

Our approach offers the hope that high-quality embeddings can be maintained in both settings given their unique advantages and appropriateness in different settings.

At the same time, we show that by distilling static embeddings from their dynamic counterparts, we can then employ the more comprehensive arsenal of embedding analysis tools that have been developed in the static embedding setting to better understand the original contextual embeddings.

As an example, we employ methods for identifying gender, racial, and religious bias (Bolukbasi et al., 2016; Garg et al., 2018; Manzini et al., 2019) to our distilled representations and find that these experiments not only shed light on the properties of our distilled embeddings for downstream use but can also serve as a proxy for understanding existing biases in the original pretrained contextual representations.

Our large-scale and exhaustive evaluation of bias further reveals dramatic inconsistencies in existing measures of social bias and highlights sizeable discrepancies in the bias estimates obtained for distilled embeddings drawn from different pretrained models and individual model layers.

In this work, we study pretrained word embeddings, primarily of the static variety.

As such, we focus on comparing our embeddings against existing pretrained static embeddings that have seen widespread adoption.

We identify Word2Vec and GloVe as being the most prominent static embeddings currently in use and posit that these embeddings have been frequently chosen not only because of their high quality representations but also because lookup tables pretrained on large corpora are publicly accessible and easy to use.

Similarly, in considering contextual models to distill from, we begin with BERT as it has been the most prominent in downstream use among the growing number of alternatives (e.g. ELMo (Peters et al., 2018) , GPT (Radford et al., 2018) , BERT (Devlin et al., 2019) , Transformer-XL , GPT-2 (Radford et al., 2019) , XLNet , RoBERTa , and DistilBERT (Sanh, 2019) ) though we provide similar analyses for several of the other models (GPT-2, XLNet, RoBERTa, DistilBERT) and more comprehensively address them in the appendices.

We primarily report results for the bert-base-uncased model and include complete results for the bert-large-uncased model in the appendices as well.

In order to use a contextual model like BERT to compute a single context-agnostic representation for a given word w, we define two operations.

The first is subword pooling: the application of a pooling mechanism over the subword representations generated for w in context c to compute a single representation for w in c, i.e. {w 1 c , . . . , w k c } → w c .

Beyond this, we define context combination to be the mapping from representations w c1 , . . . , w cn of w in different contexts c 1 , . . . , c n to a single static embedding w that is agnostic of context.

The tokenization procedure for BERT can be decomposed into two steps: performing a simple word-level tokenization and then potentially deconstructing a word into multiple subwords, yielding w 1 , . . .

, w k such that cat(w 1 , . . .

, w k ) = w where cat(·) indicates concatenation.

In English, the subword tokenization algorithm is WordPiece (Wu et al., 2016) .

As a consequence, the decomposition of a word into subwords is the same across contexts and the subwords can be unambiguously associated with their source word.

Therefore, any given layer of the model outputs vectors w 1 c , . . . , w k c .

We consider four potential pooling mechanisms to compute w c given these vectors:

min(·) and max(·) are element-wise min and max pooling, mean(·) indicates mean pooling, i.e.

|X | and last(·) indicates selecting the last vector, w k c .

In order to convert contextual representations into static ones, we describe two methods of specifying contexts c 1 , . . .

, c n and then combining the resulting representations w c1 , . . . , w cn .

Decontextualized -For a word w, we use a single context where c 1 = w. That is, we feed the single word w by itself into the pretrained contextual model and consider the resulting vector to be the representation (applying subword pooling if the word is split into multiple subwords).

Aggregated -Observing that the Decontextualized strategy may be presenting an unnatural input to the pretrained encoder which may have never encountered w by itself without a surrounding phrase or sentence, we instead consider ways of combining the representations for w in multiple contexts.

In particular, we sample n sentences from a large corpus D, each of which contains the word w, and compute the vectors w c1 , . . .

, w cn .

Then, we apply a pooling strategy to yield a single representation that aggregates the representations across the n contexts as is shown in Equation 2.

To assess the representational quality of our static embeddings, we evaluate on several word similarity and word relatedness datasets (see §A.2 for additional commentary).

We consider 4 such datasets: RG65 (Rubenstein & Goodenough, 1965) , WS353 (Agirre et al., 2009 ), SIMLEX999 (Hill et al., 2015) and SIMVERB3500 (Gerz et al., 2016) .

Taken together, these datasets contain 4917 examples and contain a vocabulary V of 2005 unique words.

Each example is a pair of words (w 1 , w 2 ) with a gold-standard annotation (provided by one or more humans depending on the dataset) of how semantically similar or how semantically related w 1 and w 2 are.

A word embedding is evaluated by the relative correctness of its ranking of the similarity/relatedness of all examples in a dataset with respect to the gold-standard ranking using the Spearman ρ coefficient.

Embedding predictions are computed using cosine similarity as in Equation 3:

We begin by studying how the choices of f and g 2 impact the performance of embeddings distilled from bert-base-uncased.

In Figure 1 , we show the performance on all four datasets of the resulting static embeddings where embeddings computed using the Aggregated strategy are pooled over N = 100000 sentences.

Here, N is the number of total contexts for all words (see §A.4).

Across all four datasets, we see that g = mean is the best performing pooling mechanism within the Aggregated strategy and also outperforms the Decontexualized strategy by a substantial margin.

Fixing g = mean, we further observe that mean pooling at the subword level also performs best.

We further find that this trend that f = mean, g = mean is optimal among the 16 possible pairs consistently holds for almost all pretrained contextual models we considered.

If we further consider the impacts of N as shown in Table 1 , we see that performance for both bert-base-uncased and bert-large-uncased tends to steadily increase for all datasets with increasing N (and this trend holds for the 7 other pretrained models).

In particular, in the largest setting with N = 1000000, the bert-large-uncased embeddings distilled from the best performing layer for each dataset dramatically outperform both Word2Vec and GloVe.

However, this can be seen as an unfair comparison given that we are selecting the layer for specific datasets.

As the middle band of table shows, we can fix a layer and still outperform both Word2Vec and Glove.

Beyond the benefits of using a larger N , Table 1 reveals an interesting relationship between N and the best-performing layer.

In Figure 1 , there is a clear preference towards the first quarter of the model's layers (layers 0-3) with a sharp drop-off in performance immediately thereafter (we see a similar preference for the first quarter in models with a different number of layers, e.g. Figure 3 , Figure 10 ) .

Given that our intrinsic evaluation is centered on lexical semantic understanding, this appears to be largely consistent with the findings of Liu et al. (2019a) ; Tenney et al. (2019) .

However, as we pool over a larger number of contexts, we see that the best-performing layer monotonically (with a single exception) shifts to be later and later within the pretrained model.

What this indicates is that since the later layers did not perform better for smaller values of N , these layers demonstrate greater variance with respect to the layer-wise distributional mean and reducing this variance helps in our evaluation 3 .

This may have implications for downstream use, given that later layers of the model are generally preferred by downstream practitioners and it is precisely these layers where we see the greatest variance.

Accordingly, combining our stable static embeddings from layer with the contextual example-specific embeddings also from layer of the pretrained model as was suggested in Peters et al. (2018) may be a potent strategy in downstream settings.

In general, we find these results suggest there may be merits towards further work studying the unification of static and dynamic methods.

Along with a trend towards later layers for larger values of N , we see a similar preference towards later layers as we consider each column of results from left to right.

In particular, while the datasets are ordered chronologically 4 , each dataset was explicitly introduced as an improvement over its predecessors (perhaps transitively, see §A.3).

While it is unclear from our evaluation as to what differences in the examples in each dataset may cause this behavior, we find this correlation with dataset difficulty and layer-wise optimality to be intriguing.

In particular, we see that SIMVERB3500 which contains verbs primarily (as opposed to nouns or adjectives which dominate the other datasets) tends to yield the best performance for embeddings distilled from the intermediary layers of the model (most clear for bert-large-uncased).

Remarkably, we find that most tendencies we observe generalize well to all other pretrained models we study (specifically the optimality of f = mean, g = mean, the improved performance for larger N , and the layer-wise tendencies with respect to N and dataset).

In Table 2 , we summarize the results of all models employing the Aggregated strategy with f = mean, g = mean and N = 100000 contexts.

Surprisingly, despite the fact that many of these models perform approximately equally on many downstream evaluations, we observe that their corresponding distilled embeddings perform radically differently even when the same distillation procedure is applied.

These results can be interpreted as suggesting that some models learn better lexical semantic representations whereas others learn other behaviors such as context representation and semantic composition more accurately.

More generally, we argue that these results warrant reconsideration of analyses performed on only one pretrained model as they may not generalize to other pretrained models even when the models considered have (nearly) identical Transformer architectures.

A noteworthy result in Table 2 is that of DistilBert-6 which outperforms BERT-12 on three out of the four datasets despite being distilled using knowledge distillation (Ba & Caruana, 2014; Hinton et al., 2015) from BERT-12.

Analogously, RoBERTa, which was introduced as a direct improvement over BERT, does not reliably outperform the corresponding BERT models when comparing the derived static embeddings.

Table 2 : Performance of static embeddings from different pretrained models on word similarity and word relatedness tasks.

f and g are set to mean for all models, N = 100000, and (#) indicates the layer the embeddings are distilled from.

Bold indicates best performing embeddings for a given dataset of those depicted.

Bias is a complex and highly relevant topic in developing representations and models in machine learning and natural language processing.

In this context, we study the social bias encoded within static word representations.

As Kate Crawford argued for in her NIPS 2017 keynote, while studying individual models is important given that specific models may propagate, accentuate, or diminish biases in different ways, studying the representations that serve as the starting point and that are shared across models (which are used for possibly different tasks) allows for more generalizable understanding of bias (Barocas et al., 2017) .

In this work, we simultaneously consider multiple axes of social bias (i.e. gender, race, and religion) and multiple proposed methods for computationally quantifying these biases.

We do so precisely because we find that existing NLP literature has primarily prioritized gender (which may be a technically easier setting) and because we find that different computational specifications of bias that evaluate the same social phenomena yield different results.

As a direct consequence, we strongly caution that the results should be taken with respect to the definitions of bias being applied.

Further, we note that an embedding which receives low bias scores cannot be assumed to be (nearly) unbiased, rather that under existing definitions the embedding exhibits low bias and perhaps additional more nuanced definitions are needed.

Bolukbasi et al. (2016) introduced a definition for computing gender bias which assumes access to a set P = {(m 1 , f 1 ), . . .

, (m n , f n )} of (male, female) word pairs where m i and f i only differ in gender (e.g. 'men' and 'women').

They compute a gender direction g:

where E(·) is the embedding function, ";" indicates horizontal concatenation/stacking and [0] indicates taking the first principal component.

Then, given a set N of target words that we are interested in evaluating the bias with respect to, Bolukbasi et al. (2016) specifies the bias as:

This definition is only inherently applicable to binary bias settings, i.e. where there are exactly two protected classes, but still is difficult to apply to binary settings beyond gender as constructing a set P can be challenging.

Similarly, multi-class generalizations of this bias definition are also difficult to propose due to the issue of constructing k-tuples that only differ in the underlying social attribute.

This definition also assumes the first principal component is capable of explaining a large fraction of the variance.

Garg et al. (2018) introduced a different definition for computing binary bias that is not restricted to gender, which assumes access to sets A 1 = {m 1 , · · · , m n } and A 2 = {f 1 , · · · , f n } of representative words for each of the two protected classes.

For each class, µ i = mean w∈Ai E(w) is computed.

Garg et al. (2018) computes the bias in the following ways:

Compared to the definition of Bolukbasi et al. (2016) , these definitions may be more general as constructing P is strictly more difficult than constructing A 1 , A 2 (as P can always be split into two such sets but the reverse is not generally true) and Garg et al. (2018) 's definition does not rely on the first principal component explaining a large fraction of the variance.

However, unlike the first definition, Garg et al. (2018) computes the bias in favor of/against a specific class (meaning if N = {'programmer', 'homemaker'} and 'programmer' was equally male-biased as 'homemaker' was female-biased, then under the definition of Garg et al. (2018) , there would be no bias in aggregate).

For the purposes of comparison, we adjust their definition by taking the absolute value of each term in the mean over N .

Manzini et al. (2019) introduced a definition for quantifying multi-class bias which assumes access to sets A 1 , . . .

, A k of representative words as in Garg et al. (2018) .

They quantify the bias as 5 :

Similar to the adjustment made for the Garg et al. (2018) definition, we again take the absolute value of each term in the mean over N .

Figure 2: Layer-wise bias of distilled BERT-12 embeddings for f = mean, g = mean, N = 100000 Left: Gender, Center: Race, Right: Religion Table 3 : Social bias within static embeddings from different pretrained models with respect to a set of professions N prof .

Parameters are set as f = mean, g = mean, N = 100000 and the layer of the pretrained model used in distillation is X 4 .

Lowest bias in a particular column is denoted in bold.

Inspired by the results of Nissim et al. (2019) , in this work we transparently report social bias in existing static embeddings as well as the embeddings we compute.

In particular, we exhaustively report the bias for all 3542 valid (pretrained model, layer, social attribute, bias definition) 4-tuples which describe all combinations of static embeddings and bias measures referenced in this work.

We specifically report results for binary gender (male, female), two-class religion (Christianity, Islam) and three-class race (white, Hispanic, and Asian), directly following Garg et al. (2018) .

These results are by no means intended to be comprehensive with regards to the breadth of bias socially and only address a restricted class of social biases which notably does not include the important class of intersectional biases.

The types of biases being evaluated for are taken with respect to specific word lists (which are sometimes subjective albeit being peer-reviewed) that serve as exemplars and with respect to definitions of bias grounded in the norms of the United States.

Beginning with bert-base-uncased, we report the layer-wise bias across all (attribute, definition) pairs in Figure 2 .

What we immediately observe is that for any given social attribute, there is a great deal of variation across the layers in the quantified amount of bias.

Further, while we are unsurprised that different bias measures for the same social attribute assign different absolute scores, we observe that they also do not agree in relative judgments.

For gender, we observe that the bias estimated by the definition of Manzini et al. (2019) steadily increases before peaking at the penultimate layer and slightly decreasing thereafter.

In contrast, under bias GARG-EUC we see a distribution with two peaks corresponding to layers at the start or end of the pretrained contextual model with lower bias observed in the intermediary layers.

For estimating the same quantity, bias GARG-COS is mostly uniform across the layers (though the scale of the axes visually lessens the variation displayed).

Similarly, in looking at the religious bias, we see similar inconsistencies with the bias increasing monotonically from layers 2 through 8 under bias MANZINI , decreasing monotonically under bias GARG-EUC , and remaining roughly constant under bias GARG-COS .

In general, while the choice of N (and the choice of A i in the gender bias case) does affect the absolute bias estimates under any given definition, we find that the general trends in the bias across layers are approximately invariant under these choices for a specific definition.

Taken together, our analysis suggests a concerning state of affairs regarding bias quantification measures for (static) word embeddings.

In particular, while estimates are seemingly stable to some types of choices regarding word lists, bias scores for a particular word embedding are tightly related to the definition being used and existing bias measures are markedly inconsistent with each other.

We find this has important consequences beyond understanding the social biases in our representations.

Concretely, we argue that without certainty regarding the extent to which embeddings are biased, it is impossible to properly interpret the meaningfulness of debiasing procedures (Bolukbasi et al., 2016; Zhao et al., 2018a; b; Sun et al., 2019) as we cannot reliably estimate the bias in the embeddings both before and after the procedure.

This is further compounded with the existing evidence that current intrinsic measures of social bias may not handle geometric behavior such as clustering (Gonen & Goldberg, 2019) .

In light of the above, next we compare bias estimates across different pretrained models in Table 3 .

Given the conflicting scores assigned by different definitions, we retain all definitions along with all social attributes in this comparison.

However, we only consider target words given by N prof for visual clarity as well as due to the aforementioned stability to the choice of N , with the results for adjectives provided in Table 8 .

We begin by noting that since we do not perform preprocessing to normalize embeddings, the scores using bias GARG-EUC are not comparable (and may not have been proper to compare in the layer-wise case either) as they are sensitive to the absolute norms of the embeddings which cannot be expected to be similar across models 6 .

Further, we note that bias BOLUKBASI may not be a reliable indicator as similar to Zhao et al. (2019a) , we find that the first principal component explains less than 35% of the variance in the majority of the static embeddings distilled from contextual models.

Of the two bias definitions not mentioned thus far, we find that all distilled static embeddings have substantially higher scores under bias MANZINI but generally lower scores under bias GARG-COS when compared to Word2Vec and GloVe.

Interestingly, we see that under bias MANZINI both GPT-2 and RoBERTa embedding consistently get high scores across social attributes when compared to other distilled embeddings but under bias GARG-COS they receive the lowest scores among distilled embeddings.

Ultimately, given the aforementioned issues regarding the reliability of bias measures, it is difficult to arrive at a clear consensus of the comparative bias between our distilled embeddings and prior static embeddings.

What our analysis does resolutely reveal is a pronounced and likely problematic effect of existing bias definitions on the resulting bias scores.

Distilled Static Representations.

Recently, Akbik et al. (2019) introduced an approach similar to our Aggregated strategy where representations are gradually aggregated across instances in a dataset during training to model global information.

Between epochs, the memory of past instances is reset and during testing, inference-time instances are added into the memory.

In that work, the computed static embeddings are an additional feature that is used to achieve the state-of-the-art on several NER datasets.

Based on our results, we believe their approach could be further improved by different decisions in pretrained model and layer choice.

Their results may be explained by the (desirable) variance reduction we observe in pooling over many contexts.

Additionally, since they only pool over instances in an online fashion within an epoch, the number of contexts is relatively small in their approach as compared to ours which may help to explain why they find that min or max pooling perform slightly better than mean pooling as the choice for g. May et al. (2019) proposes a different approach to convert representations from sentence encoders into static embeddings as a means for applying the WEAT (Caliskan et al., 2017) implicit bias tests to a sentence encoder.

In their method, a single semantically-bleached sentence is synthetically constructed from a template and then fed into the encoder to compute a static embedding for the word of interest.

We argue that this approach may inherently not be appropriate for quantifying bias in sentence encoders 7 in the general case as sentence encoders are trained on semantically-meaningful sentences and semantically-bleached constructions are not representative of this distribution.

Moreover, the types of templated constructions presented heavily rely on deictic expressions and therefore are difficult to adapt for certain syntactic categories such as verbs (as would be required for the SimVerb3500 dataset especially) without providing arguments for the verb.

These concerns are further exacerbated by our findings given the poor representational behavior seen in our Decontextualized embeddings which have similar deficiencies with their static embeddings and the poor representational behavior when we pool over relatively few semantically-meaningful contexts using the Aggregated strategy (e.g. our results for N = 10000 which is still 50 instances per word on average and is much more than the single instance they consider).

We believe our quantification of bias as a result can be taken as a more faithful estimator of bias in sentence encoders.

Concurrently, Hu et al. (2019) considers a similar approach towards diachronic sense modelling.

In particular, given a word, they find its senses and example sentences of each sense in the Oxford English Dictionary and use these to compute static embeddings using the Aggregated strategy with the last layer of bert-base-uncased and n i upper-bounded at 10.

Given our results, their performance could likely be improved by pooling over more sentences, using bert-large-uncased, and considering layer choice as their task heavily relies on lexical understanding which seems to be better captured in earlier layers of the model than the last one.

Since they require sense annotations for their setting (and the number of example sentences in a dictionary for a sense is inherently constrained), our findings also suggest that additional sense-annotated or weakly sense-annotated sentences would be beneficial.

Lightweight Pretrained Representations.

Taken differently, our approach can be seen as a method for integrating pretraining in a more lightweight fashion.

Model compression (LeCun et al., 1990; Frankle & Carbin, 2019) and knowledge distillation (Ba & Caruana, 2014; Hinton et al., 2015) are well-studied techniques in machine learning that have been recently applied for similar purposes.

In particular, several concurrent approaches have been proposed to yield lighter pretrained sentence encoders and contextual word representations (Gururangan et al., 2019; Sanh, 2019; Tsai et al., 2019; Jiao et al., 2019) .

Our approach along with these recent approaches yield representations that are more appropriate for resource-constrained settings such as on-device models for mobile phones , for real-time settings where we require low-latency and short inference times, and for users that may not have access to GPU or TPU computational resources (Tsai et al., 2019) .

Additionally, this line of work is particularly timely given the emergent concerns of the environmental impact/harm of training and using increasingly large models in NLP (Strubell et al., 2019) , machine learning (Li et al., 2016; Canziani et al., 2016) , and the broader AI community (Schwartz et al., 2019) .

Bias.

Social bias in NLP has been primarily evaluated in three ways: (a) using geometric similarity between embeddings (Bolukbasi et al., 2016; Garg et al., 2018; Manzini et al., 2019) , (b) adapting psychological association tests (Caliskan et al., 2017; May et al., 2019) , and (c) considering down-stream behavior 2018a; 2019a; Stanovsky et al., 2019) 8 .

In relation to this body of work, our bias evaluation is in the style of (a) as we are interested in intrinsic bias in embeddings and considers (potentially) multi-class social bias in the lens of gender, race, and religion whereas prior work has primarily focused on gender.

Additionally, while most of the work on bias in embeddings has considered the static embedding setting, recent work has considered sentence encoders and contextual models.

Zhao et al. (2019a) considers gender bias in ELMo when applied to NER and Kurita et al. (2019) extends these results by considering not only NER but also bias using WEAT by leveraging the masked language modeling objective of BERT.

Similarly, Basta et al. (2019) considers intrinsic gender bias using ELMo by studying gender-swapped sentences.

When compared to these approaches, we study a broader class of biases under more than one bias definition and consider more than one model.

Further, while these approaches generally neglect reporting bias values for different layers of the model, we show this is crucial as bias is not uniformly distributed throughout model layers and downstream practitioners often do not use the last layer of deep Transformer models (Liu et al., 2019a; Tenney et al., 2019; Zhao et al., 2019b) 9 .

Pretrained contextual word representations have quickly gained traction in the NLP community, largely because of the flurry of empirical successes that have followed since their introduction.

For downstream practitioners, our work suggests several simple (e.g. subword pooling mechanism choice) and more sophisticated (e.g. layer choice, benefits of variance reduction by using multiple contexts) strategies that may yield better downstream performance.

Additionally, some recent models have combined static and dynamic embeddings (Peters et al., 2018; Bommasani et al., 2019; Akbik et al., 2019) and our representations may support drop-in improvements in these settings.

Beyond furthering efforts in representation learning, this work introduces a new approach towards the understanding of contextual word representations via proxy analysis.

In particular, while in this work we choose to study social bias, similar analyses toward other forms of interpretability and understanding would be valuable.

Additionally, post-processing approaches that go beyond analysis such as dimensionality reduction may be particularly intriguing given that this is often challenging to do within large multi-layered networks like BERT (Sanh, 2019) but has been successfully done for static embeddings (Nunes & Antunes, 2018; Mu & Viswanath, 2018; Raunak et al., 2019) .

Future work may also consider the choice of the corpus D from which contexts are drawn.

In particular, we believe choosing D to be drawn from the target domain for some downstream task may serve as an extremely lightweight domain adaptation strategy.

Additionally, in this work we choose to provide contexts of sentence length in order to facilitate regularity in the comparison across models.

But for some models, such as Transformer-XL or XLNet which are trained with memories to handle larger contexts, better performance may be achieved by using larger contexts.

In this work, we propose simple but effective procedures for converting contextual word representations into static word embeddings.

When applied to pretrained models like BERT, we find the resulting embeddings outperform Word2Vec and GloVe substantially under intrinsic evaluation and provide insights into the pretrained model.

We further demonstrate the resulting embeddings are more amenable to (existing) embedding analysis methods and report the extent of various social biases (gender, race, religion) across a number of measures.

Our large-scale analysis furnishes several findings with respect to social bias encoded in popular pretrained contextual representations via the proxy of our embeddings and has implications towards the reliability of existing protocols for quantifying bias in word embeddings.

All data, code, visualizations (and code to produce to them), and distilled word embeddings will be publicly released.

Additional reproducibility details are provided in Appendix A. URL http://papers.nips.cc/paper/ 6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddi pdf.

Rishi Bommasani, Arzoo Katiyar, and Claire Cardie.

SPARSE: Structured prediction using argument-relative structured encoding.

In this work, we chose to conduct intrinsic evaluation experiments that focused on word similarity and word relatedness.

We did not consider the related evaluation of lexical understanding via word analogies as they have been shown to decompose into word similarity subtasks (Levy & Goldberg, 2014a) and there are significant concerns about the validity of these analogies tests (Nissim et al., 2019) .

We acknowledge that word similarity and word relatedness tasks have also been heavily scrutinized (Faruqui et al., 2016; Gladkova & Drozd, 2016) .

A primary concern is that results are highly sensitive to (hyper)parameter selection (Levy et al., 2015) .

In our setting, where the parameters of the embeddings are largely fixed based on which pretrained models are publicly released and where we exhaustively report the impact of most remaining parameters, we find these concerns to still be valid but less relevant.

To this end, prior work has considered various preprocessing operations on static embeddings such as clipping embeddings on an elementwise basis (Hasan & Curry, 2017) when performing intrinsic evaluation.

We chose not to study these preprocessing choices as they create discrepancies between the embeddings used in intrinsic evaluation and those used in downstream tasks (where this form of preprocessing is generally not considered) and would have added additional parameters implicitly.

Instead, we directly used the computed embeddings from the pretrained model with no changes throughout this work.

A.3 REPRESENTATION QUALITY DATASET TRENDS Rubenstein & Goodenough (1965) introduced a set of 65 noun-pairs and demonstrated strong correlation (exceeding 95%) between the scores in their dataset and additional human validation.

Miller & Charles (1991) introduced a larger collection of pairs which they argued was an improvement over RG65 as it more faithfully addressed semantic similarity.

Agirre et al. (2009) followed this work by introducing a even more pairs that included those of Miller & Charles (1991) as a subset and again demonstrated correlations with human scores exceeding 95%.

Hill et al. (2015) argued that SIMLEX999 was an improvement in coverage over RG65 and more correctly quantified semantic similarity as opposed to semantic relatedness or association when compared to WS353.

Beyond this, SIMVERB3500 was introduced by Gerz et al. (2016) to further increase coverage over all predecessors.

Specifically, it shifted the focus towards verbs which had been heavily neglected in the prior datasets which centered on nouns and adjectives.

We used PyTorch (Paszke et al., 2017) throughout this work with the pretrained contextual word representations taken from the HuggingFace pytorch-transformers repository 13 .

Tokenization for each model was conducted using its corresponding tokenizer, i.e. results for GPT2 use the GPT2Tokenizer in pytorch-transformers.

For simplicity, throughout this work, we introduce N as the total number of contexts used in distilling with the Aggregated strategy.

Concretely, N = wi∈V n i where V is the vocabulary used (generally the 2005 words in the four datasets considered).

As a result, in finding contexts, we filter for sentences in D that contain at least one word in V. We choose to do this as this requires a number of candidate sentences upper bounded with respect to the most frequent word in V as opposed to filtering for a specific value for n which requires a number of sentences scaling in the frequency of the least frequent word in V. The N samples from D for the Aggregated strategy were sampled uniformly at random.

Accordingly, as the aforementioned discussion suggests, for word w i , the number of examples n i which contain w i scales in the frequency of w i in the vocabulary being used.

As a consequence, for small values of N , it is possible that rare words would have no examples and computing a representation w using the Aggregated strategy would be impossible.

In this case, we back-offed to using the Decontextualized representation for w i .

Given this concern, in the bias evaluation, we fix n i = 20 for every w i .

In initial experiments, we found the bias results to be fairly stable when choosing values n i ∈ {20, 50, 100}. The choice of n i would correspond to N = 40100 (as the vocabulary size was 2005) in the representation quality section in some sense (however this assumes a uniform distribution of word frequency as opposed to a Zipf distribution).

The embeddings in the bias evaluation are drawn from layer X 4 using f = mean, g = mean as we found these to be the best performing embeddings generally across pretrained models and datasets in the representational quality evaluation.

The set of gender-paired tuples P were taken from Bolukbasi et al. (2016) .

In the gender bias section, P for definitions involving sets A i indicates that P was split into equal-sized sets of male and female work.

For the remaining gender results, the sets described in §G.3 were used.

The various attribute sets A i and target sets N j were taken from Garg et al. (2018) which can be further sourced to a number of prior works in studying social bias.

We remove any multi-word terms from these lists.

B BERT-LARGE Table 8 : Social bias within static embeddings from different pretrained models with respect to a set of adjectives, N adj .

Parameters are set as f = mean, g = mean, N = 100000 and the layer of the pretrained model used in distillation is X 4 .

<|TLDR|>

@highlight

A procedure for distilling contextual models into static embeddings; we apply our method to 9 popular models and demonstrate clear gains in representation quality wrt Word2Vec/GloVe and improved analysis potential by thoroughly studying social bias.