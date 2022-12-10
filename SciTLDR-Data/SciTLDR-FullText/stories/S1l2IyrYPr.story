Recent improvements in large-scale language models have driven progress on automatic generation of syntactically and semantically consistent text for many real-world applications.

Many of these advances leverage the availability of large corpora.

While training on such corpora encourages the model to understand long-range dependencies in text, it can also result in the models internalizing the social biases present in the corpora.

This paper aims to quantify and reduce biases exhibited by language models.

Given a conditioning context (e.g. a writing prompt) and a language model, we analyze if (and how) the sentiment of the generated text is affected by changes in values of sensitive attributes (e.g. country names, occupations, genders, etc.) in the conditioning context, a.k.a.

counterfactual evaluation.

We quantify these biases by adapting individual and group fairness metrics from the fair machine learning literature.

Extensive evaluation on two different corpora (news articles and Wikipedia) shows that state-of-the-art Transformer-based language models exhibit biases learned from data.

We propose embedding-similarity and sentiment-similarity regularization methods that improve both individual and group fairness metrics without sacrificing perplexity and semantic similarity---a positive step toward development and deployment of fairer language models for real-world applications.

Text representation learning methods (word and sentence encoders) trained on large unlabeled corpora are widely used in the development of natural language processing systems (Mikolov et al., 2013; Pennington et al., 2014; Peters et al., 2018; Devlin et al., 2018) .

Progress in this area has led to consistent improvements of model performances on many downstream tasks.

However, recent studies have found that both context-free and context-dependent word embedding models contain human-like semantic biases, including gender and race (Bolukbasi et al., 2016; Caliskan et al., 2017; Zhao et al., 2019) .

Zhao et al. (2018a) provide an insight into this phenomenon by showing that web corpora contain biases (e.g., gender) which are inherited by models trained on these datasets.

In this work, we focus on language models which have been shown to exhibit systematic biases (Lu et al., 2018; Bordia & Bowman, 2019; Qian et al., 2019) .

We train a Transformer-based language model (Vaswani et al., 2017; on two large corpora: Wikipedia articles from Wikitext-103 (Merity et al., 2016) and news articles from the English-language news corpus from .

1 We analyze systematic variations in sentiment scores of the text generated by the language model given a conditioning context, under different instantiations of control variables (e.g. country names, occupations, and person names) in the context.

In a counterfactual experiment, we find that sentiment scores for the text generated by this language model vary substantially as we change the control variables in the context.

We propose two approaches to reduce counterfactual sentiment biases based on the concept of embedding similarity or sentiment similarity.

In the first method, we encourage hidden states of the conditioning context to be similar irrespective of the instantiations of the control variables in the context.

In the second method, we regularize the difference between sentiment scores of various instantiations of the control variables.

Experiments with counterfactual conditioning demonstrate that both of these methods reduce sentiment biases while retaining the generation capability of the language model, as measured by perplexity and semantic similarity.

While specifying optimal model fairness behavior is difficult, our method provides a framework to address various fairness specifications and an important step toward the deployment of fairer language models.

Our main contributions in this paper are:

• We demonstrate systematic counterfactual sentiment biases in large-scale language models.

• We present methods to quantify these biases by adopting individual and group fairness metrics from the fair machine learning literature.

• We propose embedding and sentiment similarity-based methods for training language models to be invariant to certain transformations of their inputs.

• We empirically demonstrate the efficacy of these methods to reduce counterfactual sentiment biases of language models.

We use a sentiment classifier as a proxy to measure biases in this paper.

We note that the classifier itself is not perfect and might exhibit some biases.

We leave investigations of an unbiased evaluator to future work.

Language models.

Given an article x composed of n tokens (x 1 , · · · , x n ), a language model estimates the probability p(x) of x occurring in natural language under the assumption that the joint probability factorizes over the tokens as follows:

p(x i |x 1:i−1 )

where the prefix x 1:i−1 := (x 1 , · · · , x i−1 ) for convenience.

Once a language model is learned, the model can be used to generate sequences that capture long-range dependencies (Graves, 2013) .

By using the conditional probability p(x i |x 1:i−1 ), we sample the next token x i given a prefix (or conditioning inputs) x 1:i−1 .

Then we can iteratively use the generated token x i along with the previous prompt as the conditioning inputs to generate the next token x i+1 using p(x i+1 |x 1:i ).

We use Transformer-based models (Vaswani et al., 2017) to learn the probability p(x i |x 1:i−1 ), which has been demonstrated to scale to large self-supervised models with outstanding performance in generation quality and representation learning, including BERT (Devlin et al., 2018) , GPT-2 , MT-DNN (Liu et al., 2019) , XLNet and many others.

Bias in Natural Language Processing Systems.

Besides learning to favor language of the authors' demographic group (Hovy & Søgaard, 2015) , NLP models pick up on a variety of cultural associations and undesirable social biases (Caliskan et al., 2017) .

Systematic imbalances were observed across NLP tasks, e.g. as gender bias in coreference resolution (Zhao et al., 2018a; Rudinger et al., 2018) , visual semantic role labeling , image captioning (Hendricks et al., 2018) , or in text classification (Dixon et al., 2018; Garg et al., 2019) .

Concretely in sentiment analysis, Kiritchenko & Mohammad (2018) found systematic biases with respect to race and gender across more than 200 systems.

For word embeddings, occupational gender bias has been identified and addressed by measuring projections onto linear gender-related subspaces of word representations (Bolukbasi et al., 2016; Lemoine et al., 2018; Zhao et al., 2018b; Bordia & Bowman, 2019) .

Gonen & Goldberg (2019) however pointed out limitations to this approach: bias in word embeddings appear indirectly in other ways, even after minimizing linear projections onto gender-related subspaces.

Bias in Language Modeling.

Rather than debiasing word embeddings, Lu et al. (2018) proposed counterfactual data augmentation as a remedy to occupation-specific gender biases, and found that it can much better retain model performance than debiasing word embeddings, especially in language modeling.

Qian et al. (2019) on the other hand regularize a generative language model to predict similar log-probabilities for either option of a gendered word pair.

Zhao et al. (2019) and Basta et al. (2019) demonstrate gender bias in pretrained language modeling representations (ELMo), which translates into downstream tasks, but do not consider language generated by the ELMo language model.

In contrast to these prior works on debiasing language models, we probe language models' generated output using a sentiment analysis system.

We do not rely on gendered word pairs for data augmentation or for approximating linear gender subspaces.

Furthermore, prior work mostly considers only comparatively small language modeling training sets.

In contrast, we investigate bias in Transformer-based models with a similar number of parameters to GPT-2.

Our models are trained on English news articles from the WMT-19 news translation challenge, which contains 40GB of text, as well as WikiText-103, with more than 100 million tokens.

A fundamental group fairness definition is "equality of odds", which requires false positive and false negative prediction rates to be equal across demographic subgroups (Hardt et al., 2016) .

However, this definition of group fairness can be superficially satisfied through postprocessing methods at a potential cost on individual fairness, which requires similar individuals to be treated similarly (Dwork et al., 2012) , as well as other statistical fairness metrics.

Furthermore, ignoring the data generating causal graph of the problem may lead to "corrective discrimination", that is, discrimination caused by the very procedure to enforce statistical fairness criteria.

Hence causal inference tools are leveraged in fairness research to deal with these problems that may occur in satisfying statistical fairness criteria.

Similar to individual fairness, counterfactual fairness requires same model predictions before and after intervention on sensitive attributes in data generating causal graphs (Kusner et al., 2017; Kilbertus et al., 2017) .

In our problem setting, we consider the counterfactual fairness goal using a causal graph representing the text generation model with input features, latent features, model outputs and predictions as nodes of the graph.

We aim towards counterfactual fairness by de-biasing the learned representation of inputs in the latent space of the text generative model, contributing to a family of methods to learn fair representations (Beutel et al., 2017; Zemel et al., 2013; Creager et al., 2019; Edwards & Storkey, 2016; Louizos et al., 2016) and enforcing independence between sensitive attributes and prediction outputs (Calders et al., 2009; Lemoine et al., 2018; Jiang et al., 2019) .

Motivating Examples.

To illustrate the problem of biased sentiment, we condition a large-scale language model (for model details see Section 5) with the prefix "You are a/an <occupation>, and you", with the same random seeds using "accountant" and "designer" as occupation.

We sample 1,000 sentences with both prefixes and measure the sentiment scores of the generated sentences.

In Fig. 1 , we observe systematic sentiment differences in the generated output.

In Table 1 , we present some generated examples with large sentiment difference.

The systematic difference in the sentiment distribution, further exemplified in these particular generated sentences, demonstrates that there exists a bias in sentiment with respect to a counterfactual change of occupation in the given context.

To further quantify this problem and reduce the biases, we illustrate the problem formulation and our proposed approaches below.

Fairness Specification.

Given a predefined specification on a set of sensitive attribute variables C (e.g., occupations, genders, or countries), we would like to reduce their counterfactual sentiment biases in language models for every sensitive attribute variable A ∈ C. We let A be the set of possible values of the variable A, and use a to denote a particular value of A (e.g. A = {female, male}, a = female).

For each input sequence x containing sensitive tokens φ(a) (such as φ(a)={he, his, him, husband, Paul} for a = male), we generate a counterfactual inputx to x by replacing all occurrences of each sensitive token in φ(a) with the corresponding token in φ(ã), whereã is another sensitive attribute randomly chosen from the set A \ {a}, and leaving all other non-sensitive tokens of x unchanged.

Given a fixed/pre-defined sentiment classifier f s and a pretrained language model LM , so that the random variable LM (x) is a sentence sampled from the language model conditioned on x, define the random variable S(x) = f s (LM (x)) to be the generated sentence sentiment score in [0, 1] , and denote its distribution by P S (x).

For binary sentiment classification, typically we compute predictionŷ = S > τ given a decision threshold τ .

Figure 2: Sentiment score histogram using "You are a/an <Occupation>, and you" as an input to a language model trained with our proposed method.

Table 1 : Generated samples with counterfactual inputs using a baseline language model.

One fundamental fairness concept is "demographic parity", which requires equal positive classification rates across subgroups, i.e. p(ŷ | A = a) = p(ŷ | A = a ) for any sensitive attributes a, a .

We also measure deviation from it, "demographic disparity", by differences between the subgroup positive rates Dwork et al., 2012) ).

Applying this concept to measuring fairness between counterfactual pairs, demographic disparity is the difference between positive sentiment rates of S(x) and

However, often we do not want our fairness goal to be dependent on a predetermined decision threshold τ , since τ may be user-defined or simply not known at training time.

We require the raw output distributions P S (x) and P S (x) to match -instead of the binary predictionŷ, which is called "Strong Demographic Parity" (Jiang et al., 2019) .

We also extend the deviation measurement by computing statistical disparity averaged over uniformly random choices of

where U denotes the random uniform distribution.

This quantity is equal to the Wasserstein-1 distance between distributions P S (x) and P S (x) (Jiang et al., 2019) ,

Sentiment bias by counterfactual evaluation is then the Wasserstein-1 distance between output sentiment distributions P S of the original input x and its counterfactualx.

Thus our counterfactual fairness specification for sentiment biases, i.e. counterfactual sentiment bias, is

for any sensitive attribute a ∈ A and a chosen threshold > 0.

This fairness formulation also expresses individual fairness which requires similar individuals to be treated similarly (Dwork et al., 2012) , provided that similarity is defined by having the same non-sensitive tokens.

Note that this specification addresses the output distribution of a generative model, in which it differs from prior work on specifications in NLP models which concern individual predictions of discriminative models (Huang et al., 2019; Jia et al., 2019) .

Fairness Evaluation.

For each sensitive variable A ∈ C, we measure the individual fairness and group fairness metrics from distributions of sentiment scores P S on the evaluation set in the following way.

Individual Fairness Metric.

Based on the fairness property of the Wasserstein-1 distance (Eq. 1), we compute Average Individual Fairness by averaging Wasserstein-1 distance between the sentiment score distribution of every evaluation sentence P S (x) and each of its counterfactual sentence P S (x) across all M templates 2 for sensitive variable A. Formally, this is

where the inner sum is over all

unordered pairs of distinct a,ã in A. a,ã are the sensitive attributes of x m , x m respectively.

Group Fairness Metric.

The evaluation sentences are separated into |A| = K disjoint subgroups, assigning a sentence to group a if it contains sensitive tokens from φ(a).

For example, when sensitive variable A = gender, we have K = 2 for A = {male, female} and φ(male) = {he, his, him, husband, Paul, . . .}.

For each subgroup a ∈ A, we measure the Wasserstein-1 distance between the sentiment distribution of all generated sentences of inputs from this subgroup, denoted P a S , and that over the entire evaluation set, denoted P * S .

Then we report the sum of all subgroup Wasserstein-1 distances as the Total Group Fairness metric, i.e.,

Given an input prefix x 1:i with i tokens,

where the token x i ∈ φ(a) is associated with a group a of a sensitive attribute (e.g., countries, names, occupations), we construct a perturbed prefix by replacing x i with a tokenx i ∈ φ(ã) from a different groupã, where fairness between the two groups should be maintained.

We obtain a perturbed prefixx 1:i = (x 1:i−1 ,x i ).

To train the language model towards reducing counterfactual sentiment bias, we want to ensure that the language model produces similar sentiment distributions for the two prefixes.

Specifically, we would like the Wasserstein-1 distance between the sentiment distributions of generated sentences, P S (x 1:i ) and P S (x 1:i ), to be small, as shown in Eq. 2.

In practice, it is prohibitively expensive to sample a distribution of generated sequences for every x 1:i andx 1:i Instead, we use hidden features from the language model as a proxy to represent the distribution of future generated sequences, since p(x i+1 , x i+2 , · · · |x 1:i ) and p(x i+1 , x i+2 , · · · |x 1:i ) depend on the hidden states of the language model conditioned on x 1:i andx 1:i , respectively.

We explore two approaches: Fairness through embedding similarity and Fairness through sentiment similarity by exploiting the hidden states of the language model.

Given an L-layer transformer based language model with an input x 1:i , we let h(

Fairness through embedding similarity.

In this approach, we want to make sure the embedding h (j) (x 1:i ) and h (j) (x 1:i ) are close enough, since the joint probabilities p(x i+1 , x i+2 , · · · |x 1:i ) and p(x i+1 , x i+2 , · · · |x 1:i ) are determined by the embedding.

We call it the "embedding similarity" approach.

We define the fairness loss as a distance between the embeddings, denoted as d(h(x 1:i ), h(x 1:i ).

We consider using the cosine distance:

summary" of embedding layer features, and α j is the weight of h (j) (x).

Typically, the embedding in earlier layers captures word-level information and embedding in later layers represents more high-level semantics (Tenney et al., 2019 case, since we want to capture high-level semantics (e.g., sentiments), we use the average over the last 2 layers' embedding as the extracted featuresh(x) (L s = L − 2, α L−1 = 0.5, α L = 0.5).

We find that averaging too many layers can make the difference betweenh(x) and h(x 1:i )) very small, reducing the effectiveness of regularization.

The main drawback of enforcing embedding similarity is that this regularization can be too strong, as we require the hidden representations (and thus the joint probabilities) to be as close as possible: in the worst case, the model can learn to ignore individual members and generate the same texts for all of them.

Despite being completely fair in this extreme case, model performance may suffer since the generated text should contextually depend on x i orx i .

Fairness through sentiment similarity.

To overcome the above-mentioned drawback, we propose an alternative method for eliminating sentiment biases using sentiment classifiers.

Instead of measuring d(h(x 1:i ), h(x 1:i )) directly, we first apply the same sentiment classifier f s to both h(x 1:i ) and h(x 1:i ), and measure d(f s (h(x 1:i )), f s (h(x 1:i ))) instead.

Note that the output of f s can be multi-dimensional (e.g., a hidden layer in the sentiment classifier), and we can measure the distance via cosine similarity.

The classifier f s can be seen as a projection from h(x) to a subspace that ideally only contains sentiment related information.

If such a perfect projection exists, we can regularize the sentiment difference between the two inputs without affecting the model's perplexity.

The detailed implementation of f s is introduced in Section 5.1.

On one hand, this classifier-based sentiment similarity approach avoids the strong regularization in enforcing embedding similarity and can potentially produce better language models with lower perplexity on test sets.

On the other hand, the effectiveness of this method is correlated with the quality of the sentiment classifier (or sentiment "projection").

Implementation -Three-Step Curriculum Training.

We use a three-step curriculum training scheme to implement the proposed embedding similarity, sentiment similarity approaches.

First, we train a language model using regular cross-entropy loss for predicting the next token given all the previous tokens, as done in typical language training setting; a good validation perplexity ensures a relatively good hidden feature space has been learned.

Second, using this language model, we train a sentiment classifier f s (e.g., a simple multilayer perceptron (MLP)) using the extracted features from the language model; since sentiment labels are generally unavailable for large-scale corpus, we label a subset of training data with Google Cloud sentiment analysis API.

3 Third, we continue language model training with the addition of fairness loss L fairness based on "embedding similarity" or "sentiment similarity" with a regularization parameter λ, and in the meanwhile the language model is still trained on regular negative log-likelihood (NLL) or cross-entropy loss (L LM ) on predicting the next token of unperturbed input x. The loss function for an input sequence x is:

We refer the third step as "debiasing step", which is illustrated in Figure 3 .

The second and third steps may be repeated if desired.

To reflect recent advancements in language modeling, we train two TransformerXL language models similar in scale to GPT-2 (Radford et al., 2019) on a medium-scale corpus of Wikipedia articles, WikiText-103, and a large-scale corpus of English new articles, from the WMT-19 document-level translation task, which we will refer to as WMT-19.

4 We do not use the pre-trained GPT-2 models themselves, for which the training data is not publicly available.

The wikitext103 dataset (Merity et al., 2016) consists of 28,591 articles and over 100 million tokens extracted from high quality Wikipedia articles.

We use 28,471 articles for training, 60 articles for validation and 60 articles for tests.

WMT-19 consists of 14,635,198 English news articles; we take the last 10,000 for evaluation with 1,000 for validation and the final 9,000 articles as a test set.

On the WikiText-103 dataset, we train a TransformerXL language model composed of 18-layer transformers with an embedding size of 1024, 8 attention heads, and 257M parameters.

The model achieved 17.06 perplexity on the validation set.

On the WMT-19 dataset, we train a language model composed of 48 layer transformers with an embedding size of 1024, comprising 2,125 million parameters.

The model achieved 17.46 perplexity on the validation set.

For both models, we train a 3-layer MLP network with hidden layer size 128 as the sentiment classifier f s for sentiment feature projection.

Labels for sentence sentiment are generated using the Google Cloud sentiment analysis API.

As it does not generate perfect labels we only keep sentences with relatively high sentiment scores (normalized scores close to 0 or 1) to reduce noise in label generation.

The sentiment classifier achieves over 98% test accuracy on both datasets.

Sensitive groups and attributes.

To measure the counterfactual sentiment biases in language models, we examine three categories of sensitive attributes: Country, Occupation, and Name.

Country contains 10 representative countries and Occupation contains 29 common occupations; for Country or Occupation, sensitive tokens φ(a) are always a singleton containing either the country name or the occupation.

For Name, we consider gender as the sensitive attribute and sensitive tokens for both subgroups φ(A = male) and φ(A = female) contain 17 different common names.

All attributes are detailed in Appendix A.

Sentence templates.

For each category of sensitive attributes, we design a set of M = 10 templates to evaluate the counterfactual sentiment biases.

Each template is a sentence prefix with length i m , m ∈ [M ] containing a placeholder that will be replaced by a sensitive token in φ(a) for each sensitive attribute value a ∈ A. In other words, for each template we complete it by inputting the appropriate sensitive token for every a ∈ A, forming a prefix x 1:im which is used as a conditioned input to the language model.

We apply an external sentiment classifier f s on the generated sentences and sample 1000 sentences conditioned on each input prefix.

All templates are described in Appendix A.

Since it is impractical to evaluate each generated sentence manually, we evaluate the generated sentences using both Google Cloud sentiment API and a simpler, counting-based sentiment classifier.

We design the counting-based sentiment classifier by simply counting the number of positive opinion words p and the number of negative opinion words n (Hu & Liu, 2004) and define the sentiment scores as p/(p + n) and 0.5 if no opinion words exist.

The counting-based sentiment classifier is introduced because the sentiment API is a blackbox model and may itself contain bias, as researchers have discovered in many existing automatic sentiment analysis systems (Kiritchenko & Mohammad, 2018) .

The simple counting-based method, while being less accurate, is less prone to giving biased judgments as it does not contain sensitive attributes and only contains opinion words.

Furthermore, since we use the same sentiment API to create the sentiment label of the training data for creating the sentiment projection, it is better to use a different metric to gauge sentiment and avoid overfitting a specific sentiment analysis system.

As mentioned in Section 3, we report average individual fairness (Eq. 3), and total group fairness (Eq. 4) for Country, Occupation and Name detailed above.

Trade-off between relevance and fairness.

We found that the model could generate irrelevant sentences if trained using a very large debiasing regularization parameter.

In this case, the model is "fair" in the sense that it completely ignores the sensitive attributes.

However this deteriorates the original language model's performance, and we expect the model to ideally capture semantics given by these attributes.

Thus, it is important to evaluate the trade-off between generation quality and fairness.

We use three metrics for this purpose.

First, we report the perplexity on the whole test set and the perplexity on a subset of the test set that includes articles with at least one sensitive attribute.

The perplexity on a whole test set reflects the language model performance overall.

Given the sensitive attributes only exist in a small fraction of test data, we report perplexity over a subset of test set specifically to examine the language model performance related to the sensitive attributes.

Second, we measure the semantic similarity using an universal sentence encoder (Cer et al., 2018) .

We calculate the cosine similarity between the embedding of the attribute word and the generated sentences.

We define a generated sentence to be similar if the cosine similarity is above a given threshold (set to 0.2 empirically).

We report semantic similarity ratio as a proxy on whether the generated sentences capture the original semantics.

Note we empirically find it is helpful to measure whether models generate irrelevant sentences when there is a large semantic similarity ratio drop (e.g. >20%) compared to baseline language models.

Smaller semantic similarity ratio difference might not reflect obvious semantic changes in generation quality.

Model Selection.

We train language models using both embedding-similarity and semanticsimilarity losses with different regularization strengths.

Based on the losses in the validation set, we report λ = {10, 100} for embedding-similarity and λ = {100, 1000} for sentiment-similarity on WMT-19.

On WikiText-103, we report λ = {1, 10} for embedding-similarity and λ = {10, 100} for sentiment-similarity.

Note that it is unlikely that our models overfit the templates -during the training process (see Figure 3) , we do not add these templates explicitly to the dataset.

In Tables 2 and 3 , we report the performance on WMT-19 and WikiText-103 dataset, respectively.

Each fairness metric is evaluated twice using the sentiment API and counting-based sentiment scores.

We can observe that the proposed approaches achieve reduced bias in both individual fairness and group fairness metrics.

For each method, we report the performance of two models with two different regularization parameters for the fairness loss.

A larger regularization produces a model with less bias; however the semantic similarity scores also reduces slightly.

We can balance the trade-off between model performance by choosing different regularization parameters.

A very strong regularization (not shown in Tables 2 and 3 ) will produce a model that generates almost identical texts (under the same random seed) given different countries, names or occupations in the prefix.

We give an example of generated text in this situation in Appendix C.

We observe that our proposed methods can retain a similar level of perplexity on the subset of test set containing sensitive attributes (PPL s ).

Since we do not further train our baseline model on this subset, with the additional epochs of the debiasing step, subset perplexity (PPL s ) can sometimes improve a little bit, while reducing counterfactual sentiment biases under individual fairness and group fairness measure.

Note the perplexity on the full test set (PPL) is almost unaffected by our proposed methods, which can be potentially related to the use of a small learning rate during the debiasing step and the use of small regularization parameters.

In most settings, we found that the sentiment-similarity method performs slightly better -when semantic similarities are similar, models trained using sentiment-similarity regularization achieve better fairness metrics (e.g. Emb.

Sim.

λ = 100 versus Sent.

Sim λ = 1000 in Country of Table 2 ).

When fairness scores are similar, sentiment-similarity regularization achieves better semantic similarity (e.g., Emb.

Sim.

λ = 10 versus Sent.

Sim.

λ = 1000 in Occupation of Table 2 ; Emb.

Sim.

λ = 100 versus Sent.

Sim.

λ = 1000 in Name of Table 2.)

Table 2 : Performance for language models trained on WMT-19, where "PPL" and "PPL s " represent the perplexity at the BPE level on the full test set and the subset of the test set that contains the sensitive attributes, respectively.

"Semantic Sim." lists sentence similarity ratios, and "I. F." and "G. F." indicate average individual fairness and total group fairness, respectively.

Metrics with superscript c are based on the counting-based sentiment classifier; otherwise they use sentence sentiments from the sentiment API.

Note that except for "Semantic Sim.", lower numbers are better.

Table 3 : Performance for language models trained on WikiText-103, where "PPL" and "PPL s " represent the perplexity at the word level on the full test set and the subset of the test set that contains the sensitive attributes, respectively.

"Semantic Sim." lists sentence similarity ratios, and "I. F." and "G. F." indicate average individual fairness and total group fairness, respectively.

Metrics with superscript c are based on the counting-based sentiment classifier; otherwise they use sentence sentiments from the sentiment API.

Note that except for "Semantic Sim." lower numbers are better.

Comparing between Tables 2 and 3 , we can observe the degree of bias (average individual fairness and group fairness) in WikiText-103 is overall smaller than WMT-19, possibly reflecting the characteristics of data source (Wikipedia articles vs. news articles), and the sensitivity/quality of the language models with smaller model sizes and a smaller dataset .

Finally, in Table 4 , we randomly sample some sentences from the generated examples to show some qualitative examples.

Note we fix the random seed for each model and only change the attribute in template.

baseline "are an ordinary accountant... probably not.

However, no one -and this has led me to question the checklist for a public kitchen diet -has the money to turn to another morbid office worker." accountant embed-sim.

"do not currently make money.

As the cost of public service bonuses has now risen, so has the demand for private finance." sent-sim.

"pay a burden of millions.

So why do you govern like an accountant, and how do you keep it safe?

We argue that the costs of managing other people's money in a way you would expect from old-fashioned. ."

baseline "are an ordinary, smart young Twitter fan.

Your name wasn't mentioned on the first piece of Jones Bros." designer embed-sim.

"do ski for every variety set.

The Elsa chance!" sent-sim.

"may hatch your old lake.

So before you leave, commit to preferring a lakeside resort -keep it listsgarten.com.

If last month's ITA entries flip out, you'd hope it would flip out."

As large-scale language models are increasingly deployed for real-world applications, developing methods for assessing and mitigating bias with respect to sensitive attributes may be an increasingly important area of inquiry for facilitating pro-social outcomes.

Recent work on bias in language models has made significant progress in this direction (Lu et al., 2018; Qian et al., 2019; Bordia & Bowman, 2019) , but most work to date has focused on comparatively smaller-scale language models.

In this paper, we study counterfactual sentiment biases in large-scale transformer-based language models.

We evaluate and quantify the presence of biases in terms of both individual fairness and group fairness metrics.

We have demonstrated that our proposed embedding-similarity and sentiment-similarity based methods reduce the counterfactual sentiment biases, while maintaining similar perplexity and generation semantics.

While specifying optimal model fairness behavior is difficult, our method provides a framework to address various fairness specifications and an important step toward the deployment of fairer language models.

For future work, the proposed framework could be extended to study counterfactual biases given other specifications (e.g. religion, ethnicity, age, or multiple-attribute cross-subgroups) that requires fairness guarantees, and could be used with other predefined measures, such as an emotion classifier.

We provide additional experimental details for training and evaluating the models in this section.

Language model training (step 1 of curriculum training).

For WMT-19, we train our model on 128 TPUv3 cores using Adam optimizer with a learning rate of 2.5 × 10 −4 , batch size of 256 and a total of 5 × 10 5 training steps; for WikiText-103, we train our model on 128 TPUv3 cores using Adam optimizer with a learning rate of 2.5 × 10 −4 , batch size 512 and a total of 2.5 × 10 5 training steps.

For both datasets, we use a sequence length of 512 per batch, and we keep the states (embeddings) for the latest 512 tokens in transformer.

Language model debiasing (step 3 of curriculum training).

Since the language model has achieved good validation perplexity in step 1, we decrease learning rate and use a smaller number of training steps in this step.

For both datasets, we reduce learning rate to 2.5 × 10 −5 ; we train WMT-19 for 5 × 10 4 steps, and train WikiText103 for 2.5 × 10 4 steps for debiasing.

For this step, we only use 16 TPUv3 cores and reduce batch size to 16 and 32 for WMT-19 and WikiText-103, respectively.

Due to the decrease of step size in this step, we found that sometimes language model perplexity improves after step 3, despite adding the additional fairness loss.

Sample Generation.

We sample 1000 sentences per template given a specified sensitive attribute to estimate the fairness metrics.

The total number of samples generated is huge as we have 10 templates per category and in each category we can have tens of sensitive attributes.

Throughout the sampling experiments, we sample sentences with 50 tokens and we remove unfinished sentences determined by period or new-line symbol.

We sample with temperature of 1.0.

In this section we demonstrate a model trained with too large embedding similarity regularization.

Under the same random seed, the model produces almost identical outputs for different occupations, and the text generated is irrelevant to the context given by occupations ("sheriff" or "designer").

This model achieves very low semantic similarity score.

This example shows an extreme for trading off between fairness and performance, and it also shows the importance of using a semantic score to guide model selection.

Figure 4 , we report semantic similarity scores and individual fairness for models under different regularization strengths in the WMT-19 Country category (corresponding to Table 2 ).

We can observe that the sentiment similarity based models achieve higher semantic similarity scores than embedding similarity based models at a similar level of individual fairness.

On the other hand, with similar semantic similarity scores, the sentiment similarity based models achieve better individual fairness than embedding similarity based models.

For both proposed approaches, we improve the individual fairness significantly compared to the baseline model.

In addition to the sentiment biases discussed in this paper, we can also observe some gender biases in occupation, relevant to some findings in Solaiman et al. (2019) .

Specifically, using templates 2 and 3 in the country category, "My wife/husband just got an exciting new job in <Country>. Starting next week , she/he will be", we count occupation words (Zhao et al., 2018a) in the generated samples across all the countries using a WMT-19 baseline language model.

Among the 10,000 generated sentences, we filter out occupation that occurs less than 5 times and we report the counts in in Fig 5.

We can observe the model has gender biases towards some occupations such as "editor", "teacher", "guard", "CEO", and "secretary".

We demonstrate the models capture the distinction between the counterfactual attributes by showing some examples of distinct words in the generated samples.

Specifically we define the distinct words w for category a between categories a and b as arg max w p(w|a)/p(w|b).

In Table 9 , we show some examples between several pair of categories and the top 10 distinct words.

@highlight

We reduce sentiment biases based on counterfactual evaluation of text generation using language models.

@highlight

This paper measures sentiment bias in language models as reflected by text generated by the models, and adds other objective terms to the usual language modeling objective to reduce bias.

@highlight

This paper proposes to evaluate bias in pre-trained language models by using a fixed sentiment system and tests several different prefix templates.

@highlight

A method based on semantic simiilarity and a method based on sentiment similarity to debias the neural language models trained from large datasets.