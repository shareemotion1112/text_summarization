Recent pretrained transformer-based language models have set state-of-the-art performances on various NLP datasets.

However, despite their great progress, they suffer from various structural and syntactic biases.

In this work, we investigate the lexical overlap bias, e.g., the model classifies two sentences that have a high lexical overlap as entailing regardless of their underlying meaning.

To improve the robustness, we enrich input sentences of the training data with their automatically detected predicate-argument structures.

This enhanced representation allows the transformer-based models to learn different attention patterns by focusing on and recognizing the major semantically and syntactically important parts of the sentences.

We evaluate our solution for the tasks of natural language inference and grounded commonsense inference using the BERT, RoBERTa, and XLNET models.

We evaluate the models' understanding of syntactic variations, antonym relations, and named entities in the presence of lexical overlap.

Our results show that the incorporation of predicate-argument structures during fine-tuning considerably improves the robustness, e.g.,  about 20pp on discriminating different named entities, while it incurs no additional cost at the test time and does not require changing the model or the training procedure.

Transformer-based language models like BERT (Devlin et al., 2019) , XLNET (Yang et al., 2019) , and RoBERTa (Liu et al., 2019) achieved stateof-the-art performances on various NLP datasets including those of natural language inference (NLI) (Condoravdi et al., 2003; Dagan et al., 2006) , and grounded commonsense reasoning (GCI) (Zellers et al., 2018) .

1 Natural language inference is the task of determining whether the hypothesis entails, contradicts, or is neutral to the given premise.

Grounded commonsense reasoning, as it is defined by the SWAG dataset (Zellers et al., 2018) , is the task of reasoning about what is happening and predict what might come next given a premise that is a partial description about a situation.

Despite their great progress on individual datasets, pretrained language models suffer from various biases, including lexical overlap (McCoy et al., 2019b) .

For instance, given the premise "Neil Armstrong was the first man who landed on the Moon", the model may recognize the sentence "Moon was the first man who landed on the Neil Armstrong" as an entailing hypothesis or a plausible ending because it has a high lexical overlap with the premise.

In this paper, we enhance the text of the input sentences of the training data, which is used for fine-tuning the pretrained language model on the target task, with automatically detected predicateargument structures.

Predicate-argument structures identify who did what to whom for each sentence.

The motivation of using predicate-argument structures is to provide a higher-level abstraction over different surface realizations of the same underlying meaning.

As a result, they can help the model to focus on the more important parts of the sentence and abstract away from the less relevant details.

We show that adding this information during fine-tuning considerably improves the robustness of the examined models against various adversarial settings including those that evaluate models' understanding of syntactic variations, antonym relations, and named entities in the presence of high lexical overlap.

Our solution imposes no additional cost over the linguistic-agnostic counterpart at the test time since it does not require predicateargument structures for the test data.

Besides, compared to existing methods for handling the lexical overlap bias Clark et al., 2019; Mahabadi and Henderson, 2019) , it does not require introducing new models or training procedures and the model's complexity remains unchanged.

The contributions of this work are as follows:

1.

We provide three adversarial evaluation sets for the SWAG dataset to evaluate the lexical overlap bias.

These adversarial test sets evaluate the model's understanding of syntactic variation, antonym relation, and named entities.

The performance of all the examined models drops substantially on these datasets.

We will release the datasets to encourage the community to develop models that better capture the semantics of the task instead of relying on surface features.

2.

We propose a simple solution for improving the robustness against the lexical overlap bias by adding predicate-argument structures to the fine-tuning data.

Our solution results in no additional cost during the test time, it does not require oracle predicate-argument structures, and it also does not require any changes in the model or the training procedure.

We will release the augmented training data for MultiNLI and SWAG training data.

The findings of this work include:

• While lexical overlap is a known bias for NLI, we show that models that are fine-tuned on SWAG are more prone to this bias.

• The RoBERTa model performs the best on all adversarial test sets and is therefore more robust against the lexical overlap bias.

• Among the examined evaluation settings, discriminating different named entities in the presence of high lexical overlap is the most challenging.

The best accuracy, i.e., the accuracy of the RoBERTa-large model fine-tuned with augmented training data, is 59%.

• Previous work showed that pretrained transformer-based language models capture various linguistic phenomena, e.g., POS tags, syntax, named entities, and predicate-argument structures, without explicit supervision (Hewitt and Manning, 2019; Tenney et al., 2019) .

Yet, our work shows that explicit incorporation of such information is beneficial for improving robustness.

Overcoming the Lexical Overlap Bias.

This bias is investigated for NLI.

The existing solutions for tackling this bias include using debiasing methods Clark et al., 2019; Mahabadi and Henderson, 2019) .

All these models use a separate model to recognize the training examples that contain the bias.

They then use various approaches to either not learn from biased examples or down-weight their importance during training.

The resulting improvements from these methods on the HANS dataset, using the BERT model, is on-par as those reported in this work.

Our proposed solution, on the other hand, makes the model itself, e.g., BERT, more robust against the lexical overlap bias by learning better attention patterns and does not require a separate model for recognizing or skipping biased examples during training.

The use of linguistic information in recent neural models is not very common.

The use of such information has been mainly investigated for tasks in which there is a clear relation between the linguistic features and the target task.

For instance, various neural models use syntactic information for the task of semantic role labeling (SRL) (Roth and Lapata, 2016; Marcheggiani and Titov, 2017; Strubell et al., 2018; Swayamdipta et al., 2018) , which is closely related to syntactic relations, i.e., some arcs in the syntactic dependency tree can be mirrored in semantic dependency relations.

Marcheggiani and Titov (2017) build a graph representation from the input text using their corresponding dependency relations and use graph convolutional networks (GCNs) to process the resulting graph for SRL.

They show that the incorporation of syntactic relations improves the in-domain but decreases the out-of-domain performance.

Similarly, Cao et al. (2019) and Dhingra et al. (2018) incorporate linguistic information, i.e., coreference relations, in their model and show improvements in in-domain evaluations.

Strubell et al. (2018) use linguistic information, i.e., dependency parse, part-of-speech tags, and predicates for SRL using a transformer-based encoder (Vaswani et al., 2017) .

They make use of this linguistic information by (1) using multi-task learning, and (2) supervising the neural attention of the transformer model to predict syntactic dependencies.

They use gold syntax information during training and predicted information during the test time.

Their model substantially improves both indomain and out-of-domain performance in SRL.

However, these results are then outperformed by a simple BERT model without using any additional linguistic information (Shi and Lin, 2019) .

Moosavi and Strube (2018) examine the use of various linguistic features, e.g., syntactic dependency relations and gender and number information, as additional input features to a neural coreference resolver.

They show that using informative linguistic features substantially improves the generalization of the examined model.

All the above approaches require additional linguistic information, e.g., syntax, both during the training and the test time.

Swayamdipta et al. (2018) , on the other hand, only make use of the additional syntactic information during training.

They use multi-task learning by considering syntax parsing as an auxiliary task and minimizing the combination of the losses of the main and auxiliary tasks.

They use syntactic information for the tasks of SRL and coreference resolution.

They show that this information slightly improves the in-domain performance.

In this work, we do not change the loss function and only augment the input sentences of the training data.

The advantage of our solution is that it does not require any changes in the model or its training objective.

It can be applied to all the transformer-based models without changing the training procedure.

Predicate-Argument Structures.

Predicate-argument structures have been used for improving the performance of downstream tasks like machine translation (Liu and Gildea, 2010; Bazrafshan and Gildea, 2013) , reading comprehension (Berant et al., 2014; Wang et al., 2015) , and dialogue systems (Tur et al., 2005; Chen et al., 2013) .

However, these approaches are based on pre-neural models.

The proposed model by Marcheggiani et al. (2018) for neural machine translation is a sample neural model that incorporates predicate-argument structures.

Unlike this work, Marcheggiani et al. (2018) incorporate these linguistic structures at the model-level.

They add two layers of semantic GCNs on top of a standard encoder, e.g., convolutional neural network or bidirectional LSTM.

The Premise: A man in a black polo shirt is sitting in front of an electronic drum set.

Correct ending: The tutorial starts by showing each part of the drum set up close.

semantic structures are used for determining nodes and edges in the GCNs.

In this work, however, we incorporate these structures at the input level, and only for the training data.

Therefore, we can use the state-of-the-art models without any changes.

Overall, this work differs from the related work because (1) it evaluates the use of predicateargument structures for improving the robustness of state-of-the-art models for the tasks of NLI and GCI, and (2) it uses these structures at the input level to extend raw inputs, (3) it only employs this information during training, and (4) it requires no changes in the model or the training procedure.

3 Experimental Setup 3.1 Tasks Grounded Commonsense Inference.

Given a premise that is a partial description about a situation, GCI is the task of reasoning about what is happening and predicting what might come next.

SWAG models this task as a multiple choice answer selection, in which the premise is given and the correct and three incorrect endings are presented as candidate answers.

Figure 1 shows a sample premise and its correct ending from SWAG.

Natural Language Inference.

Given a premise and a hypothesis, NLI is the task of determining whether the hypothesis entails, contradicts, or is neutral to the premise.

For instance, the hypothesis in Figure 2 entails the given premise.

For the experiments of this paper, we use MultiNLI (Williams et al., 2018) dataset, which is the largest available dataset for NLI.

Premise: As spacecraft commander for Apollo XI, the first manned lunar landing mission, Armstrong was the first man to walk on the Moon.

"

That's one small step for a man, one giant leap for mankind."

With these historic words, man's dream of the ages was fulfilled.

Hypothesis:

Neil Armstrong was the first man who landed on the Moon.

Premise: A woman is packing a suitcase.

Hypothesis: A suitcase is packing a woman.

Premise: A lot of people are sitting on terraces in a big field and people is walking in the entrance of a big stadium.

Ending: A lot of people are standing on terraces in a big field and people is walking in the entrance of a big stadium.

In this section, we describe the adversarial datasets that we use to evaluate the robustness of the model against the lexical overlap bias.

We created three different adversarial datasets based on the SWAG development set for evaluating the lexical overlap bias.

These datasets evaluate the model's understanding of (1) syntactic variations, (2) antonym relations, and (3) named entities in the presence of high lexical overlap.

Syntactic Variations.

In this evaluation set, premises which contain subject-verb-object structures are taken from the SWAG development set.

We then construct a new negative ending by swapping the subject and object of the premise and replace one of the existing negative endings with the new one.

This dataset includes 20 006 samples.

2 Figure 3 contains an example of this test set.

Antonym Relations.

In this test set, we create a new negative ending by replacing the first verb of the premise (from the SWAG development set) with its antonym.

We use WordNet for the antonym relations.

This adversarial setting is also common in NLI, e.g., (Naik et al., 2018; Glockner et al., 2018) .

Figure 4 shows a sample premise and its corresponding incorrect ending that is created based on antonym relations.

This set contains 7476 samples.

Named Entities.

In order to evaluate the capability of the examined models in discriminating different named entities, we create a new adversarial dataset in which a new incorrect ending is Premise:

The reflection he sees is Harrison Ford as someone Solo winking back at him.

Ending: The reflection he sees is Eve as someone Solo winking back at him.

created by replacing one of the named entities of the premise with an unrelated named entity, i.e., "Eve".

Figure 5 shows an example of this adversarial set.

This test set contains 190 samples.

We use the Stanford named entity recognizer (Finkel et al., 2005) for determining the named entities.

For the adversarial evaluation of natural language inference, we use the Heuristic Analysis for NLI Systems (HANS) dataset (McCoy et al., 2019b) .

Sentence pairs in HANS include various forms of lexical overlap, which are created based on various syntactic variations, namely lexical overlap, subsequence, and constituent.

In the lexical overlap subset, all words of the hypothesis appear in the premise.

The subsequence subset contains hypotheses which are a contiguous subsequence of their corresponding premise.

Finally, in the constituent subset, hypotheses are a complete subtree of the premise.

Constituent is a special case of the subsequence heuristic, and they are both special cases of lexical overlap.

Figure 6 includes an example for each of these three subsets.

We incorporate predicate-argument structures of sentences by augmenting the raw text of each input sentence with its corresponding predicatearguments.

This way, no change is required in the model's architecture.

For the main experiments, we use the ProbBank-style semantic role labeling model of Shi and Lin (2019) 3 , which has the state-of-the-art results on the CoNLL-2009 dataset, to get predicate-argument structures.

We specify the beginning of the augmentation by the [PRD] special token that indicates that the next tokens are the detected predicate.

We then specify the ARG0 and ARG1 arguments, if any, with [AG0] and [AG1] special tokens, respectively.

The end of the detected predicate-argument structure is also specified by the [PRE] special token.

If more than one predicate is detected for a sentence, they would all be added at the end of the input sentence.

Figure 7 shows an example for an augmented sentence.

For our experiments, we use BERT, XLNET, and RoBERTa.

BERT (Devlin et al., 2018) is jointly trained on a masked language modeling task and a next sentence prediction task.

It is pre-trained on the BookCorpus and English Wikipedia.

XLNET (Yang et al., 2019 ) is trained with a permutationbased language modeling objective for capturing bidirectional contexts.

The XLNet-base model is trained with the same data as BERT-base.

The RoBERTa model (Liu et al., 2019) has the same architecture as BERT.

However, it is trained with dynamic masking and without the next sentence prediction task.

It is also trained using larger batchsize, vocabulary size, and training data.

We use the Huggingface Transformers library (Wolf et al., 2019) 4 , and we initialize the models with bert-base-uncased, roberta-base, and xlnet-base-cased, respectively.

We finetune each of the above models on MultiNLI and SWAG training data for the NLI and GCI experiments, respectively.

We report all the results in two different settings including (1) original: in which the model is fine-tuned on the original training data, and (2) augmented: in which the input sentences of the bert_base_srl.jsonnet 4 https://github.com/huggingface/ transformers training data are extended with their corresponding predicate-argument structures.

Except for the fine-tuning data, the other settings are exactly the same for both original and augmented experiments.

Please note that we only augment the training data of the target task that is required for fine-tuning, and not the training data of the underlying language model.

In this section, we evaluate the impact of augmenting the training data with predicate-argument structures for both in-domain and adversarial evaluations on the grounded commonsense reasoning and natural language inference tasks.

Table 2 shows the results of the examined models, based on both original and augmented settings, on the SWAG development set (in-domain) and the adversarial evaluation sets.

From the results of Table 2 , we observe that:

1.

While the examined models achieve humanlevel performance on in-domain evaluation, their performance drops drastically on the adversarial sets, e.g., below the random baseline on the antonym and named entities evaluation sets.

This shows that all these models overly rely on shallow heuristics such as word overlap for predicting whether two sentences contain successive events.

2.

Discriminating different named entities is the most challenging adversarial evaluation.

3.

The augmentation of training data with predicate-argument structures slightly decreases the performance on in-domain evaluations.

However, it significantly improves the robustness of all models in all three adversarial evaluation sets, i.e., from 8 to 22 points and on average by 13pp.

4.

While the augmentation of the training data improves the performance on the examined adversarial evaluation sets, there is still room for improvement, i.e., the highest accuracy on the named entities adversarial set is 43.99.

5.

RoBERTa has the highest performance in both original and augmented experiments and is, therefore, more robust against the lexical overlap bias.

Table 1 : Impact of data augmentation on the HANS dataset for the "entailment" and "non-entailment" labels.

All models are fine-tuned on MultiNLI training data.

The highest accuracy for each subset is boldfaced.

Table 2 : Accuracy of the examined models using the original vs. augmented SWAG training data.

Table 3 and Table 1 show the results of the examined models on the MultiNLI development set and HANS, respectively.

The main challenge in HANS is the detection of non-entailment labels because, as a result of the dataset creation artifacts in MultiNLI, most of the samples that have high lexical overlap are labeled as entailment.

Therefore, models that are trained on this dataset tend to classify most of Table 3 : Accuracy on MultiNLI development sets when the models are fine-tuned on original vs. augmented training data.

such samples as entailment.

Based on the results, fine-tuning the models on the augmented data slightly (0.2pp-0.6pp) decreases the performance on the original dataset.

The exception is the results of the XLNET model on the match subset of MultiNLI in which the performance increases slightly.

However, it improves the performance of BERT and XLNET models on the NLI adversarial subsets.

The improvements are not as large as those of GCI, e.g., 5pp in NLI vs. 22pp in GCI.

We hypothesize that models that are fine-tuned on the NLI dataset recognize predicateargument structures better than those trained on the SWAG dataset.

As mentioned by McCoy et al. (2019a) , the results on the HANS dataset can vary by a large margin using different random seeds during fine-tuning, e.g., BERT accuracy on the lexical overlap subset Figure 8 : BERT attention weights on an example from the HANS dataset based on original (top weights) and augmented (bottom weights) training.

Attention weights are visualized using BertViz (Vig, 2019) .

They highlight the attention between the hypothesis and premise words and for the predicate-argument structures of the hypothesis.

can vary from 6% to 54%.

For reproducibility, all the results in this paper are reported using the default random seed in Huggingface Transformers.

We have evaluated the impact of data augmentation on the BERT model using different random seeds.

The results on the SWAG adversarial test sets do not change notably using different random seeds.

However, using a different random seed, the improvement of augmented compared to original experiment on the HANS lexical overlap subset and for the non-entailment label can increase to 21.3pp.

5 .

Figure 8 shows the difference of the BERT attention weights, using BertViz 6 (Vig, 2019) , on an example from the HANS dataset.

In this example, the premise and hypothesis are "The senators supported the secretary in front of the doctor." and "The doctor supported the senators.", respectively.

For instance, for the predicate "supported" in the hypothesis, the BERT model that is trained on augmented data (bottom subfigure), has high attention weights on "senators", "supported", and "secretary", while for original the attention weights of this predicate are more distributed.

Similarly, for the subject "doctor" in the hypothesis, augmented mainly attends to the corresponding subject in the 5 The model is always trained using the same random seed for both original and augmented experiments 6 https://github.com/jessevig/bertviz premise, i.e., "senators".

5.1 Are predicate-argument useful for large models as well?

The examined language models have two variations, base and large models.

For instance, the RoBERTa-base model contains 125M parameters while RoBERTa-large contains 355M parameters.

In this section, we examine whether the addition of predicate-argument structures still improves the robustness of large models.

For the experiments of this section, we use RoBERTa and the GCI datasets.

The results are shown in Table 4 .

As we see from the results, while RoBERTa-large model has higher performance on all GCI evaluation sets compared to RoBERTa-base, the addition of predicate-argument structures still considerably improves the performance on adversarial evaluation sets, i.e., 7pp-27pp.

Table 4 : Accuracy of RoBERTa-large on the GCI adversarial sets.

In all the experiments of this paper, we use the predicted predicate-argument structures of a stateof-the-art semantic role labeling system, i.e., Shi and Lin (2019) .

In this section, we examine the impact of the used SRL system to see how the resulting errors in SRL impact the results.

Therefore, in this section we use OpenIE (Angeli et al., 2015) for augmenting the training data instead of the stateof-the-art SRL model.

OpenIE is a less accurate but more efficient tool for extracting relations from the sentences.

Table 5 report the results of the RoBERTa-base model on GCI evaluation sets when the model is fine-tuned on the augmented data with relations that are extracted using OpenIE.

As we see, the use of different models to extract predicate-argument structures does not considerably impact the resulting robustness.

Both augmentations considerably improve the robustness against the lexical overlap, even though one is less accurate than the other.

Table 5 : Accuracy of the RoBERTa-base model on the GCI adversarial sets when the training data is augmented using OpenIE.

Based on our experiments, the addition of the predicate-argument structures to the test data does not have a benefit for pretrained transformer-based models.

The reason is that the addition of predicateargument structures to the training data helps the transformer models to learn a different attention pattern.

However, this is not possible for other neural models.

Table 6 shows the results of the ESIM model (Chen et al., 2017) , when it is trained and tested on the original vs. augmented SWAG training data.

As we see, (1) the performance of the ESIM model is considerably lower than the pretrained language models on the adversarial datasets for the lexical overlap, and (2) the addition of predicate-argument structures to the ESIM training and test data results in an improvement in the robustness.

However, the improvements are not as large as those of

In this paper, we propose a solution to improve the robustness of the state-of-the-art NLP models, i.e., BERT, XLNET, and RoBERTa, against the lexical overlap bias.

We improve the model robustness by extending the input sentences with their corresponding predicate-argument structures.

The addition of these structures helps the transformer model to better recognize the major semantically and syntactically important parts of the sentences and learns more informative attention patterns accordingly.

Our finding, regarding the benefit of explicit incorporation of predicate-argument structures, is despite the fact that transformer-based models already captures various linguistic phenomena, including predicate-argument structures (Tenney et al., 2019) .

Our proposed solution (1) results in considerable improvements in the robustness, e.g., 20pp in accuracy, (2) incurs no additional cost during the test time, (3) does not require ant change in the model or the training procedure, and (4) works with noisy predicate-argument structures.

We evaluate the effectiveness of our solution on the task of natural language inference and grounded commonsense reasoning.

However, since our solution only includes enhancing the training examples, it is not limited to a specific task and it is applicable to other tasks and datasets that suffer from this bias, e.g., paraphrase identification (Zhang et al., 2019) , and question answering (Jia and Liang, 2017) .

We will release the new adversarial evaluation sets for the lexical overlap bias as well as the augmented training data for MultiNLI ans SWAG datasets upon the publication.

<|TLDR|>

@highlight

Enhancing the robustness of pretrained transformer models against the lexical overlap bias by extending the input sentences of the training data with their corresponding predicate-argument structures 