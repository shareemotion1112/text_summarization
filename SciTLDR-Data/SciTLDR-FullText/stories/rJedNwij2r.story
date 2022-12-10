Real-world Relation Extraction (RE) tasks are challenging to deal with, either due to limited training data or class imbalance issues.

In this work, we present Data Augmented Relation Extraction (DARE), a simple method to augment training data by properly finetuning GPT2 to generate examples for specific relation types.

The generated training data is then used in combination with the gold dataset to train a BERT-based RE classifier.

In a series of experiments we show the advantages of our method, which leads in improvements of up to 11 F1 score points compared to a strong baseline.

Also, DARE achieves new state-of-the-art in three widely used biomedical RE datasets surpassing the previous best results by 4.7 F1 points on average.

Relation Extraction (RE) is the task of identifying semantic relations from text, for given entity mentions in it.

This task, along with Named Entity Recognition, has become increasingly important recently due to the advent of knowledge graphs and their applications.

In this work, we focus on supervised RE (Zeng et al., 2014; Lin et al., 2016; Wu et al., 2017; Verga et al., 2018) , where relation types come from a set of predefined categories, as opposed to Open Information Extraction approaches that represent relations among entities using their surface forms (Banko et al., 2007; Fader et al., 2011) .

RE is inherently linked to Natural Language Understanding in the sense that a successful RE model should manage to capture adequately well language structure and meaning.

So, almost inevitably, the latest advances in language modelling with Transformer-based architectures (Radford et al., 2018a; Devlin et al., 2018; Radford et al., 2018b) have been quickly employed to also deal with RE tasks (Soares et al., 2019; Lin et al., 2019; Shi and Lin, 2019; Papanikolaou et al., 2019) .

These recent works have mainly leveraged the discriminative power of BERT-based models to improve upon the state-of-the-art.

In this work we take a step further and try to assess whether the text generating capabilities of another language model, GPT-2 (Radford et al., 2018b) , can be applied to augment training data and deal with class imbalance and small-sized training sets successfully.

Specifically, given a RE task we finetune a pretrained GPT-2 model per each relation type and then use the resulting finetuned models to generate new training samples.

We then combine the generated data with the gold dataset and finetune a pretrained BERT model (Devlin et al., 2018) on the resulting dataset to perform RE.

We conduct extensive experiments, studying different configurations for our approach and compare DARE against two strong baselines and the stateof-the-art on three well established biomedical RE benchmark datasets.

The results show that our approach yields significant improvements against the rest of the approaches.

To the best of our knowledge, this is the first attempt to augment training data with GPT-2 for RE.

In Table 1 we show some generated examples with GPT-2 models finetuned on the datasets that are used in the experiments (refer to Section 4).

In the following, we provide a brief overview of related works in Section 2, we then describe our approach in Section 3, followed by our experimental results (Section 4) and the conclusions (Section 5).

Relation Extraction is usually modelled as a text classification task, therefore most approaches to deal with class imbalance or limited data in RE follow the ones from text classification.

A number of approaches have been followed in the literature in order to tackle these challenges.

One direction is to deal with imbalance at the classifier level, by penalizing misclassification errors differently for each class, depending on the class frequency (Lewis et al., 2004; Zhou and Liu, 2005) or by explicitly adjusting prior class probabilities (Lawrence et al., 1998) .

Another popular direction relies on either undersampling the majority class(es) or oversampling the minority one(s), transforming the training data with the aim to balance it.

One of the simplest approaches, random majority undersampling, simply removes a random portion of examples from majority classes such that per class training examples are roughly equal (Japkowicz and Stephen, 2002 ).

An improved version of the previous method, balanced bagging (Hido et al., 2009) , employs bagging of classifiers that have been trained with random majority undersampling.

Oversampling approaches for textual data have been somehow limited as opposed to those for image data (Wong et al., 2016; Fawzi et al., 2016; Wang and Perez, 2017; Frid-Adar et al., 2018) , since text semantics depend inherently on the exact order or structure of word tokens.

A simple approach is to replace words or phrases with their synonyms (Zhang et al., 2015) .

Chen et al. (2011) have employed topic models to generate additional training examples by sampling from the topic-word and document-topic distributions.

Ratner et al. (2016) have proposed a data augmentation framework that employs transformation operations provided by domain experts, such as a word swap, to learn a sequence generation model.

Kafle et al. (2017) have used both a template-based method as well as an LSTM-based approach to generate new samples for visual question answering.

A similar method to our approach has been proposed by Sun et al. (2019a) who presented a framework to deal successfully with catastrophic forgetting in language lifelong learning (LLL).

Specifically and given a set of tasks in the framework of LLL, they finetune GPT-2 to learn to solve a task and generate training samples at the same time for that task.

At the beginning of training a new task, the model generates some pseudo samples of previous tasks to train alongside the data of the new task, therefore avoiding catastrophic forgetting.

Our work falls into the oversampling techniques for text, but our focus is RE.

Importantly, we do not need any domain expertise, templates, synonym thesaurus or training a model from scratch, which makes our approach easily adaptable to any domain, with relatively low requirements in resources.

In this section we present briefly the GPT-2 model and then introduce in detail our approach.

GPT-2 (Radford et al., 2018b ) is a successor of the GPT language model (Radford et al., 2018a) .

Both models are deep neural network architectures using the Transformer (Vaswani et al., 2017) , pre-trained on vast amounts of textual data.

Both models are pre-trained with a standard language modelling objective, that is to predict the next word token given k previously seen word tokens.

This is achieved by maximizing the following likelihood:

where Θ are the neural network parameters.

The authors have gradually provided publicly four different flavours of GPT-2, with 124M, 355M, 774M and 1558M parameters respectively.

In our experiments we use the second largest model (774M), since it seems to represent a good compromise between accuracy and hardware requirements 1 .

where Θ are the parameters of the model.

In this work we employ a RE classifier based on a pretrained BERT language model.

This classifier follows the same principle followed by Devlin et al. (2018) , using a special token (CLS) for classification.

The only modification is that we mask entity mentions with generic entity types, i.e., $EN-TITY A$ or $ENTITY B$. It should be noted that the method that we introduce here is not classifier specific, so any other classifier can be used instead.

To generate new training data, we split the D dataset into c subsets where each D c subset contains only examples from relation type c. Subsequently, we finetune GPT-2 on each D c for five epochs and then prompt each resulting finetuned model to generate new sentences, filtering out sentences that do not contain the special entity masks or that are too small (less than 8 tokens).

The generated sequences are combined for all relation types into a dataset Dsynth.

Subsequently, we build an ensemble of RE classifiers, each of them being finetuned on a subset 1 https://openai.com/blog/gpt-2-1-5b-release/ of Dsynth and the whole D, such that the perrelation type generated instances are equal to the number of gold instances for that relation, multiplied by ratio, i.e., |Dsynth c | = |D c | * r.

In our experiments we have set r = 1.0 (refer to Section 4.6 for a short study of its influence).

Algorithm 1 illustrates our method.

We would like to note that in early experiments, we also experimented with finetuning over the whole D, by adding a special token to the beginning of each sentence that encoded the relation type, e.g., <0>: or <1>:.

Then during generation, we would prompt the model with the different special tokens and let it generate a training instance from the respective relation type.

This approach though did not prove effective leading to worse results than just using gold data, primarily because frequent classes "influenced" more GPT-2 and the model was generating many incorrectly labeled samples.

In this section we present the empirical evaluation of our method.

We first describe the experimental setup, the datasets used, the baselines against which we evaluate DARE and subsequently present the experiments and report the relevant results.

In all experiments we used the second largest GPT-2 model (774M parameters).

All experiments were carried out on a machine equipped with a GPU V100-16GB.

For the implementation, We have used HuggingFace's Transformers library (Wolf et al., 2019) .

To finetune GPT-2 we employed Adam as the optimizer, a sequence length of 128, a batch size of 4 with gradient accumulation over 2 batches (being equivalent to a batch size of 8) and a learning rate of 3e − 5.

In all datasets and for all relation types we finetuned for 5 epochs.

For generation we used a temperature of 1.0, fixed the top-k parameter to 5 and generated sequences of up to 100 word tokens.

An extensive search for the above optimal hyperparameter values is left to future work.

Since all of our datasets are from the biomedical domain, we found out empirically (see Section 4.4 for the relevant experiment) that it was beneficial to first finetune a GPT-2 model on 500k PubMed abstracts, followed by a second round of finetuning per dataset, per relation type.

As a RE classifier we have used in all cases a pre-trained BERT model (the large uncased model) which we finetuned on either the gold or the gold+generated datasets.

We used the AdamW optimizer (Loshchilov and Hutter, 2017) , a sequence length of 128, a batch size of 32 and a learning rate of 2e − 5, We finetuned for 5 epochs, keeping the best model with respect to the validation set loss.

Also, we used a softmax layer to output predictions and we assigned a relation type to each instance si as follows:

where c ∈ L and 0 < t < 1 is a threshold that maximizes the micro-F score on the validation set.

For DARE, in all experiments we train an ensemble of twenty classifiers, where each classifier has been trained on the full gold set and a sub-sample of the generated data.

In this way, we manage to alleviate the effect of potential noisy generated instances.

To evaluate DARE, we employ three RE datasets from the biomedical domain, their statistics being provided in Table 2 .

The BioCreative V CDR corpus (Li et al., 2016) contains chemical-disease relations.

The dataset is a binary classification task with one relation type, chemical induces disease, and annotations are at the document level, having already been split into train, development and test splits.

For simplicity, we followed the work of Papanikolaou et al. (2019) and considered only intra-sentence relations.

We have included the dataset in our GitHub repository to ease replication.

In the following, we dub this dataset as CDR.

The DDIExtraction 2013 corpus (Segura Bedmar et al., 2013) contains MedLine abstracts and DrugBank documents describing drug-drug interactions.

The dataset has four relation types and annotations are at the sentence level.

The dataset is provided with a train and test split for both MedLine and DrugBank instances.

Following previous works, we concatenated the two training sets into one.

Also, we randomly sampled 10% as a development set.

In the following this dataset will be referred to as DDI2013.

The BioCreative VI-ChemProt corpus (Krallinger et al., 2017) covers chemical-protein interactions, containing five relation types, the vast majority of them being at the sentence level.

The dataset comes with a train-development-test split.

In the following we will refer to it as ChemProt.

The above datasets suffer both from class imbalance and limited number of positives, for example the rarest relation type in DDI2013 has only 153 instances in the training set, while the respective one in ChemProt has only 173 data points.

Therefore, we consider two baselines that are suited for such scenarios, the balanced bagging approach and the class weighting method, both described in Section 2.

Both baselines have as a base classifier the one described in Section 4.1.

Also, in both cases we consider an ensemble of ten models 2 .

Finally, for the class weighting approach we set each class's weight as

with min being the rarest class.

Since all our datasets come from the biomedical domain, we hypothesized that a first round of finetuning GPT-2 on in-domain data could be beneficial as opposed to directly employing the vanilla GPT-2 model.

We designed a short experiment using the CDR dataset to test this hypothesis.

To clarify, any of the two models would be then finetuned per relation type to come up with the final GPT-2 models that would generate the new training examples.

Table 3 illustrates the results of this experiment.

As we expect, this first round of finetuning proves significantly favourable.

We note that when inspecting the generated examples from the vanilla GPT-2 model, there was often the case that generated sentences contained a peculiar mix of news stories with the compound-disease relations.

In this experiment we would like to see what is the effect of our method when dealing with great imbalance, i.e., datasets with very few positive samples.

To that end, we consider the CDR dataset and sample different numbers of positive examples from the dataset (50, 250, 500, 1000 and all positives) and combine them with all the negative instances.

The resulting five datasets are used to train either a balanced bagging ensemble or DARE.

In Figure 1 , we show the results, averaging across five different runs.

In all cases our approach has a steady, significant advantage over the balanced bagging baseline, their difference reaching up to 11 F1 score points when only few positives (≤ 250) are available.

As we add more samples, the differences start to smooth out as expected.

These results clearly illustrate that DARE can boost the predictive power of a classifier when dealing with few positive samples, by cheaply generating training data of arbitrary sizes.

Our next experiment focuses in studying the effect of different sizes of generated data on DARE's performance.

As explained, our method relies on finetuning GPT-2 to generate examples for each relation type that will, ideally, come from the same distribution as the ones from the gold training data.

Nevertheless, we should expect that this procedure will not be perfect, generating also noisy samples.

As mentioned previously, we try to alleviate this effect by training an ensemble of classifiers, each trained on the whole gold and a part of the generated dataset.

An important question that arises therefore, is to determine the optimal ratio of generated examples to include in each classifier.

If too few, the improvements will be insignificant, if too many we risk to have the model being influenced by the noise.

In order to get an empirical insight to the above question we design a short experiment using the CDR dataset, for different sizes of generated data.

As gold set, we consider a random subset of 1,000 positive examples and all negatives, to make more prominent the effect of class imbalance.

In Figure 2 we show the results for five different generated data sizes.

Interestingly, adding more data does not necessarily boost classifier performance, since the noisy patterns in the generated data seem to influence more the classifier than those in the gold data.

In the following, we choose a ratio = 1, adding for each relation type a number of generated instances equal to the number of gold instances.

It should be noted that we are not limited in the total generated data that we will use since we can finetune an arbitrary number of classifiers on combinations of the gold data and subsets of the generated data.

Taking into account the previous observations, we proceed to compare DARE against the SOTA and the two previously described baselines.

Table 4 describes the results.

For the multi-class datasets we report the micro-F score in order to make our results comparable with previous works.

Also, in Appendix A, in Tables 5 and 6 we report the per class results for DARE against the state-of-the-art and the class weighting baseline, for the two multiclass datasets in order to ease comparison with past or future works.

Comparing DARE against the state-of-the-art, we observe a steady advantage of our method across all datasets, ranging from 3 to 8 F1 points.

These results are somehow expected, since we employ BERT-large as our base classifier which has proven substantially better than Convolutional (CNN) or Recurrent neural networks (RNN) across a variety of tasks (Devlin et al., 2018) .

In CDR, Papanikolaou et al. (2019) have used BioBERT(Lee et al., 2019) which is based on a BERT base cased model, while we use BERT large uncased, in ChemProt, Peng et al. (2018) use ensembles of SVM, CNN and RNN models while in DDI2013 Sun et al. (2019b have used hybrid CNN-RNN models.

When observing results for the baselines, we notice that they perform roughly on par.

DARE is better from 2 to 5 F1 points against the baselines, an improvement that is smaller that that against the state-of-the-art, but still statistically significant in all cases.

Overall, and in accordance with the results from the experiment in Section 4.5, we observe that DARE manages to leverage the GPT-2 automatically generated data, to steadily improve upon the state-of-the-art and two competitive baselines.

We have presented DARE, a novel method to augment training data in Relation Extraction.

Given a gold RE dataset, our approach proceeds by finetuning a pre-trained GPT-2 model per relation type and then uses the finetuned models to generate new training data.

We sample subsets of the synthetic data with the gold dataset to finetune an ensemble of RE classifiers that are based on BERT.

On a series of experiments we show empirically that our method is particularly suited to deal with class imbalance or limited data settings, recording improvements up to 11 F1 score points over two strong baselines.

We also report new state-of-the-art performance on three biomedical RE benchmarks.

Our work can be extended with minor improvements on other Natural Language Understanding tasks, a direction that we would like to address in future work.

@highlight

Data Augmented Relation Extraction with GPT-2