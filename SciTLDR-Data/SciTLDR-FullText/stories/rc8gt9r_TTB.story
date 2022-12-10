Textual entailment (or NLI) data has proven useful as pretraining data for tasks requiring language understanding, even when building on an already-pretrained model like RoBERTa.

The standard protocol for collecting NLI was not designed for the creation of pretraining data, and it is likely far from ideal for this purpose.

With this application in mind we propose four alternative protocols, each aimed at improving either the ease with which annotators can produce sound training examples or the quality and diversity of those examples.

Using these alternatives and a simple MNLIbased baseline, we collect and compare five new 9k-example training sets.

Our primary results are largely negative, with none of these new methods showing major improvements in transfer learning.

However, we make several observations that should inform future work on NLI data, such as that the use of automatically provided seed sentences for inspiration improves the quality of the resulting data on most measures, and all of the interventions we investigated dramatically reduce previously observed issues with annotation artifacts.

The task of natural language inference (NLI; also known as textual entailment) has been widely used as an evaluation task when developing new methods for language understanding tasks, but it has recently become clear that high-quality NLI data can be useful in transfer learning as well.

Several recent papers have shown that training large neural network models on natural language inference data, then fine-tuning them for other language understanding tasks often yields substantially better results on those target tasks (Conneau et al., 2017; Subramanian et al., 2018) .

This result holds even when starting from large models like BERT (Devlin et al., 2019) that have already been pretrained extensively on unlabeled data (Phang et al., 2018; Clark et al., 2019; Liu et al., 2019b) .

The largest general-purpose corpus for NLI, and the one that has proven most successful in this setting, is the Multi-Genre NLI Corpus (MNLI Williams et al., 2018) .

MNLI was designed for use in a benchmark task, rather than as a resource for use in transfer learning and as far as we know, it was not developed on the basis of any kind of deliberate experimentation.

Further, data collected under MNLI's data collection protocol has known issues with annotation artifacts which make it possible to perform much better than chance using only one of the sentences in each pair (Tsuchiya, 2018; Gururangan et al., 2018; Poliak et al., 2018) .

This work begins to ask what would be involved in collecting a similar dataset that is explicitly designed with transfer learning in mind.

In particular, we consider four potential changes to the original MNLI data collection protocol that are designed to improve either the ease with which annotators can produce sound examples, or the quality and diversity of those examples, and evaluate their effects on transfer.

We collect a baseline dataset of about 10k examples that follows the MNLI protocol with our annotator pool, followed by four additional datasets of the same size which isolate each of our candidate changes.

We then compare all five in a set of transfer learning experiments that look at our ability to use each of these datasets to improve performance on the eight downstream language understanding tasks in the SuperGLUE (Wang et al., 2019b) benchmark.

All five of our datasets are consistent with the task definition that was used in MNLI, which is in turn based on the definition introduced by .

In this task, each example consists of a pair of short texts, called the premise and the hypothesis.

The model is asked to read both texts and make a three-way classification decision:

Given the premise, would a reasonable person infer that hypothesis must be true (entailment), infer that that it must be false (contradiction), or decide that there is not enough information to make either inference (neutral).

While it is certainly not clear that this framing is optimal for pretraining, we leave a more broad-based exploration of task definitions for future work.

Our BASE data collection protocol ( Figure 1 ) follows MNLI closely in asking annotators to read a premise sentence and then write three corresponding hypothesis sentences in empty text boxes corresponding to the three different labels (entailment, contradiction, and neutral).

When an annotator follows this protocol, they produce three sentence pairs at once, all sharing a single premise.

Our PARAGRAPH protocol tests the effect of supplying annotators with complete paragraphs, rather than sentences, as premises.

Longer texts offer the potential for discourse-level inferences, the addition of which should yield a dataset which is more difficult, more diverse, and less likely to contain trivial artifacts.

However, reading full paragraphs adds a potential cost in added annotator time and effort, which could potentially be better spent constructing more sentence-level examples.

Our EDITPREMISE and EDITOTHER protocols test the effect of pre-filling a single seed text in each of the three text boxes that annotators are asked to fill out.

By reducing the raw amount of typing required, this could allow annotators to produce good examples more quickly.

By encouraging them to keep the three sentences similar, it could also encourage minimal-pair-like examples that minimize artifacts.

We test two variants of this idea: One uses a copy of the premise sentence as a seed text and the second retrieves a new sentence from an existing corpus that is similar to the premise sentence, and uses that.

Our CONTRAST protocol tests the effect of adding artificial constraints on the kinds of hypothesis sentences annotators can write.

Giving annotators difficult and varying constraints could encourage creativity and prevent annotators from falling into repeating ruts or patterns in their writing that could lead to easier, more repetitive data.

However, as with the use of longer contexts in BASE, this protocol risks substantially slowing the annotation process.

We experiment with a procedure inspired by that used to create the language-andvision dataset NLVR2 (Suhr et al., 2019) , in which in which annotators must write sentences that are valid entailments (or contradictions) for a given premise, but not valid entailments for a second, similar, distractor premise.

In evaluations on transfer learning with the SuperGLUE benchmark, all of these four methods offer substantial improvements in transfer ability over a plain RoBERTa model, but that only EDITOTHER and CONTRAST offering consistent improvements over BASE, and only by very small margins.

While this is largely a negative result for our primary focus on transfer, we also observe that all four of these methods are able to produce data of comparable subjective quality while significantly reducing the incidence of previously reported annotation artifacts, and that PARAGRAPH, EDITPREMISE, and EDITOTHER all accomplish this without significantly increasing the time cost of annotation.

The observation that NLI data can be effective in pretraining was first reported for SNLI (Bowman et al., 2015) and MNLI by Conneau et al. (2017) on models pretrained from scratch on NLI data.

This finding was replicated in the setting of multitask pretraining by Subramanian et al. (2018) .

This was later extended to the context of intermediate training-where a model is pretrained on unlabeled data, then on relatively abundant labeled data (MNLI), and finally scarce task specific labeled data-by Phang et al. (2018) , Clark et al. (2019) , Liu et al. (2019a) , Yang et al. (2019) , and Liu et al. (2019b) across a range of large pretrained models models and target language understanding tasks.

Similar results have been observed with transfer from the SocialIQA corpus (Sap et al., 2019) to target tasks centered on common sense.

A small body of work including Mou et al. (2016) , Bingel and Søgaard (2017) and Wang et al. (2019a) Table 1 : Randomly selected examples from the datasets under study.

Neither the MNLI training set nor any of our collected data are filtered for quality in any way, and errors or debatable judgments are common in both. explored the empirical landscape of which supervised NLP tasks can offer effective pretraining for other supervised NLP tasks.

Existing NLI datasets have been built using a wide range of strategies: FraCaS (Cooper et al., 1996) and several targeted evaluation sets were constructed manually by experts from scratch.

The RTE challenge corpora (Dagan et al., 2006, et seq.) primarily used expert annotations on top of existing premise sentences.

SICK (Marelli et al., 2014) was created using a structured pipeline centered on asking crowdworkers to edit sentences in prescribed ways.

MPE (Lai et al., 2017 ) uses a similar strategy, but constructs unordered sets of sentences for use as premise.

SNLI (Bowman et al., 2015) introduced the method, used in MNLI, of asking crowdworkers to compose labeled hypotheses for a given premise.

SciTail (Khot et al., 2018) and SWAG (Zellers et al., 2018) used domain-specific resources to pair existing sentences as potential entailment pairs, with SWAG additionally using trained models to identify examples worth annotating.

There has been little work directly evaluating and comparing these many methods.

In that absence, we focus on the SNLI/MNLI approach, because it has been shown to be effective for the collection of pretraining data and because its reliance on only crowdworkers and unstructured source text makes it simple to scale.

Two recent papers have investigated other methods that could augment the base MNLI protocol we study here.

ANLI (Nie et al., 2019) collects new examples following this protocol, but adds an incentive for crowdworkers to produce sentence pairs on which a baseline system will perform poorly.

Kaushik et al. (2019) introduce a method for expanding an already-collected dataset by making small edits to existing examples that change their labels, motivated by the same desire that motivates our EDITPREMISE and EDITOTHER protocols: to produce minimally-different minimal pairs with differing labels.

Both of these papers offer methodological changes that are potentially complementary to the changes we investigate here, and neither evaluates the impact of their methods on transfer learning.

Since ANLI is large and roughly comparable with MNLI, we include it in our transfer evaluations here.

The basic interface for our tasks is similar to that used for SNLI and MNLI: We provide a premise from a preexisting text source and ask human annotators to provide three hypothesis sentences: one that says something true about the fact or situation in the prompt (entailment), one that says something that may or may not be true about the fact or situation in the prompt (neutral), and one that definitely does not say something true about the fact or situation in the prompt (contradiction).

BASE In this baseline, modeled closely on the protocol used for MNLI, we show annotators a premise sentence and ask them to provide compose one new sentence for each label.

PARAGRAPH Here, we use the same instructions as BASE, but with full paragraphs, rather than single sentences, as the supplied premises.

EDITPREMISE Here, we pre-fill the three text boxes with editable copies of the premise sentence, and ask annotators to edit each text field to compose sentences that conform to the same three requirements used in BASE.

Annotators are permitted to delete the pre-filled text.

EDITOTHER Here, we follow the same procedure as EDITPREMISE, but rather than pre-filling the premise as a seed sentence, we instead use a similarity search method to retrieve a different sentence from the same source corpus that is similar to the premise.

We hypothesize that the additional variation offered by these added sentences could improve the creativity and diversity of the resulting examples.

CONTRAST Here, we again retrieve a second sentence that is similar to the premise, but we display it as a second premise rather than using it to seed an editable text box.

We then ask annotators to compose two new sentences: One sentence must be true only about the fact or situation in the first premise (that is, contradictory or neutral with respect to the second premise).

The other sentence must be false only about the fact or situation in the first premise (and true or neutral with respect to the second premise).

This yields an entailment pair and a contradiction pair, both of which use only the first premises, with the second premise serving only as a constraint on the annotation process.

We could not find a sufficiently intuitive way to collect neutral sentence pairs under this protocol, and opted to use only two classes rather than increase the difficulty of an already unintuitive task.

MNLI uses the small but stylistically diverse OpenANC corpus (Ide and Suderman, 2006) as its source for premise sentences, but uses nearly every available sentence from its non-technical sections.

To avoid re-using premises, we instead draw on English Wikipedia.

1 1 We use the 2019-06-20 downloadable version, extract the plain text with Apertium's WikiExtractor feature (Forcada Similarity Search The EDITOTHER and CON-TRAST protocols require pairs of related sentences as their inputs.

To construct these, we assemble a heuristic sentence-matching system intended to generate pairs of highly similar sentences that can be minimally edited to construct entailments or contradictions:

Given a premise, we retrieve its closest 10k nearest neighbors according to dot-product similarity over Universal Sentence Encoder (Cer et al., 2018) embeddings.

Using a parser and an NER system, we then select those neighbors which share a subject noun phrase in common with the premise (dropping premises for which no such neighbors exist).

From those filtered neighbors, we retrieve the single non-identical neighbor that has the highest overlap with the premise in both raw tokens and entity mentions, preferring sentences with similar length to the hypothesis.

We start data collection for each protocol with a pilot of 100 items, which are not included in the final datasets.

We use these to refine task instructions and to provide feedback to our annotator pool on the intended task definition.

We continue to provide regular feedback throughout the annotation process to clarify ambiguities in the protocols and to discourage the use of systematic patternssuch as consistently composing shorter hypotheses for entailments than for contradictions-that could make the resulting data artificially easy.

Annotators are allowed to skip prompts which they deem unusable for any reason.

These generally involve either non-sentence strings that were mishandled by our sentence tokenizer or premises with inaccessible technical language.

Skip rates ranged from about 2.5% for EDITOTHER to about 10% for CONTRAST (which can only be completed when the two premises are both comprehensible and sufficiently different from one another).

A pool of 19 professional annotators located in the United States worked on our tasks, with about ten working on each.

As a consequence of this relatively small annotation team, many annotators worked under more than one protocol, which we ran consecutively.

This introduces a modest confound into our results, in that annotators start the later tasks having seen somewhat more feedback.

All annotators, though, see substantial feedback et al., 2011), sentence-tokenize it with SpaCy (Honnibal and Montani, 2017) , and randomly sample sentences (or paragraphs) for annotation.

from a pilot phase before starting each task.

This confound presentation prevents us from perfectly isolating the differences between protocols, but we argue that these results nonetheless form an informative case study.

Using each protocol, we collect at least 10k examples and split them into exactly 9k training examples and at least 1k validation examples, all to be released upon acceptance.

Table 1 shows randomly chosen examples.

As we are investigating data collection protocols for pretraining and do not use any kind of second-pass quality control (motivated by work like Khetan et al., 2018), we do not collect a test set and do not recommend these datasets for system evaluation.

Hypotheses are mostly fluent, full sentences that adhere to prescriptive writing conventions for US English.

In constructing hypotheses, annotators often reuse words or phrases from the premise, but rearrange them, alter their inflectional forms, or substitute synonyms or antonyms.

Hypothesis sentences tend to differ from premise sentences both grammatically and stylistically.

Table 2 shows some simple statistics on the collected text.

Our clearest observation here is that the two methods that use seed sentences tend to yield longer hypotheses and tend not to show a clear correlation between hypothesis-premise token overlap and label (as measured by the standard deviations in unique tokens).

CONTRAST tends to produce shorter hypotheses.

Annotator Time Annotators completed each of the five protocols at a similar rate, taking 3-4 minutes per prompt.

This goes against our expectations that the longer premises in PARAGRAPH should substantially slow the annotation process, and that the pre-filled text in EDITPREMISE and EDITOTHER should speed annotation.

Since the relatively complex CONTRAST produces only two sentence pairs per prompt rather than three, it yields fewer examples per minute.

Table 3 shows the three words in each dataset that are most strongly associated with each label, using the smoothed PMI method of Gururangan et al. (2018) .

We also include results for a baseline: a 9k-example sample from the government documents single-genre section of MNLI, which is meant to to be maximally comparable to the single-genre datasets we collect.

BASE shows similar associations to the original MNLI, but all four of our interventions reduce these associations.

The use of longer contexts or seed sentences in particular largely eliminates the strong association between negation and contradiction seen in MNLI, and no new strong associations appear to take its place.

Our experiments generally compare models trained in ten different settings: Each of the five 9k-example training sets introduced in this paper; the full 393k-example MNLI training set; the full 1.1m-example ANLI training set (which combines the SNLI training set, the MNLI training set, and the newly-collected ANLI training examples); 2 9k- Table 3 : The top three words most associated with specific labels in each dataset, sorted by the PMI between the word and the label.

The counts column shows how many of the instances of each word occur in hypotheses matching the specified label.

We compare the two-class CONTRAST with a two-class version of MNLI Gov.

example samples from the MNLI training set and from the combined ANLI training set, meant to control for the size differences between these existing datasets and our baselines; and finally a 9k-example sample from the government section of the MNLI training set, meant to control (as much as possible) for the difference between our singlegenre Wikipedia datasets and MNLI's relatively diverse text.

Our models are trained starting from pretrained RoBERTa (large variant; Liu et al., 2019b) or XLNet (large, cased; Yang et al., 2019) .

RoBERTa represented the state of the art on most of our target tasks as of the launch of our experiments, and XLNet is competitive with RoBERTa on most tasks, but offers a natural replication, as well as the advan- Table 4 : NLI modeling experiments with RoBERTa, reporting results on the validation sets for MNLI and for the task used for training each model (Self), and the GLUE diagnostic set.

We compare the two-class CON-TRAST with a two-class version of MNLI.

tage that it can be used to better compare models trained on our data with models trained on ANLI data: ANLI was collected with a model-in-the-loop procedure using RoBERTa that makes it difficult to interpret RoBERTa results.

We run our expemients using the jiant toolkit (Wang et al., 2019d) , which implements the Super-GLUE tasks, MNLI, and ANLI, and in turn uses transformers (Wolf et al., 2019) , AllenNLP (Gardner et al., 2017) , and PyTorch (Paszke et al., 2017) .

To make it possible to train these large models on single consumer GPUs, we use small-batch (b = 4) training and a maximum total sequence length of 128 word pieces.

We train for up to 2 epochs for the very large ReCoRD, 10 epochs for the very small CB, COPA, and WSC, and 4 epochs for the remaining tasks.

Except where noted, all results reflect the median final performance from three random restarts of training.

Direct NLI Evaluations As a preliminary sanity check, Table 4 shows the results of evaluating models trained in each of the settings described above on their own validation sets, on the MNLI validation set, and on the expert-constructed GLUE diagnostic set (Wang et al., 2019c) .

As NLI classifiers trained on CONTRAST cannot produce the neutral labels used in MNLI, we evaluate them separately, and compare them with two-class variants of the MNLI models.

Our BASE data yields a model that performs somewhat worse than a comparable MNLI Gov. 9k model, both on their respective validation sets and on the full MNLI validation set.

This suggests, at Table 5 : Results from RoBERTa hypothesis-only NLI classifiers on the vaidation sets for MNLI and for the datasets used in training.

least tentatively, that the new annotations are less reliable than those in MNLI.

This is disconcerting, but does not interfere with our key comparisons.

The main conclusion we draw from these results is that none of the first three interventions improve performance on the out-of-domain GLUE diagnostic set, suggesting that they do not help in the collection of data that is both difficult and consistent with the MNLI label definitions.

We also observe that the newer ANLI data yields worse performance than MNLI on the out-of-domain evaluation data when we control for dataset size.

To further investigate the degree to which our hypotheses contain artifacts that reveal their labels, Table 5 shows results with single-input versions of our models trained on hypothesis-only versions of the datasets under study and evaluated on the datasets' validation sections.

Our first three interventions, especially EDIT-PREMISE, show much lower hypothesis-only performance than BASE.

This adds further evidence, alongside our PMI results, that these interventions reduce the presence of such artifacts.

While we do not have a direct baseline for the two-class CON-TRAST in this experiment, the fact that it shows substantially lower performance than MNLI 9k is consistent with the encouraging results seen above.

Transfer Evaluations For our primary evaluation, we use the training sets from our datasets in STILTs-style intermediate training (Phang et al., 2018) : We fine-tune a large pretrained model on our collected data using standard fine-tuning procedures, then fine-tune a copy of the resulting model again on each of the target evaluation datasets we use.

We then measure the aggregate performance of the resulting models across those evaluation datasets.

We evaluate on the tasks in the SuperGLUE benchmark (Wang et al., 2019b) : BoolQ (Clark et al., 2019) , MultiRC (Khashabi et al., 2018) , and (Zhang et al., 2018) , CommitmentBank (CB; De Marneffe et al., 2019) , Choice of Plausible Alternatives (COPA; Roemmele et al., 2011), Recognizing Textual Entailment (RTE; Bar Haim et al., 2006; Giampiccolo et al., 2007; Bentivogli et al., 2009) , the Winograd Schema Challenge (WSC; Levesque et al., 2012), and WiC (Pilehvar and Camacho-Collados, 2019) , in addition to a broad-coverage RTE diagnostic set (AX b ) and WinoGender RTE (AX g ; Poliak et al., 2018) .

These tasks were selected to be difficult for BERT but relatively easy for nonexpert humans, and are meant to replace the largelysaturated GLUE benchmark (Wang et al., 2019c) .

SuperGLUE does not include labeled test data, and does not allow for substantial ablation analyses on its test sets.

Since we have no single final model whose performance we aim to show off, we do not evaluate on the test sets.

We also neither use any auxiliary WSC-format data when training our WSC model (as in Kocijan et al., 2019 ) nor artificially modify the task format.

As has been observed elsewhere, we do not generally reach above-chance performance on that task without these extra techniques.

Results are shown in Table 6 .

3 Intermediate training with any of our five datasets yields models that transfer better than the plain RoBERTa or XLNet baseline, but we do not see consistent improvements over MNLI Gov. 9k.

We also replicate the previously established result that NLI data is broadly helpful for transfer, with the large combined ANLI training set showing improvements over plain RoBERTa on six of eight tasks and simultaneously reducing the variance in results across restarts.

We note, though, that our best overall result uses only 9k NLI training examples, suggesting either that this size is enough to maximize the gains available through NLI pretraining, or Table 6 : Model performance on the SuperGLUE validation and diagnostic sets.

The Avg.

column shows the overall SuperGLUE score-an average across the eight primary tasks, weighting each task equally-as a mean and standard deviation across three restarts.

CONTRAST yield consistent improvements in transfer performance over BASE, and these improvements are small and not reliably reflected by any one target task.

We take this to be a largely negative result: While there are informative trends, we cannot confidently claim that any intervention yields improvements in the degree to which the resulting data can be used for transfer.

Our chief results on transfer learning are negative: None of our four interventions consistently improve upon the base MNLI data collection protocol by more than a marginal degree, though we see suggestive evidence that methods that supply annotators with retrieved non-premise seed sentences for inspiration offer small improvements.

However, we also observe that all four of our interventions, and especially the use of longer contexts or pre-filled seed sentences, help reduce the prevalence of artifacts in the generated hypotheses that reveal the label, and the use of longer premises or seed sentences in particular do this without increasing the time cost of annotation.

This suggests that these methods may be valuable in the collection of high-quality evaluation data, if combined with additional validation methods to ensure high human agreement with the collected labels.

The need and opportunity that motivated this work remains compelling: Human-annotated data like MNLI has already proven itself as a valuable tool in teaching machines general-purpose skils for language understanding, and discovering ways to more effectively build and use such data could further accelerate the field's already fast progress toward robust, general-purpose language understanding technologies.

Further work along this line of research could productively follow a number of directions: General work on incentive structures and task design for crowdsourcing could help to address more general questions about how to collect data that is simultaneously creative and consistently labeled.

Machine learning methods work on transfer learning could help to better understand and exploit the effects that drive the successes we have seen with NLI data so far.

Finally, there remains room for further empirical work investigating the kinds of task definitions and data collection protocols most likely to yield positive transfer.

@highlight

We propose four new ways of collecting NLI data. Some help slightly as pretraining data, all help reduce annotation artifacts.