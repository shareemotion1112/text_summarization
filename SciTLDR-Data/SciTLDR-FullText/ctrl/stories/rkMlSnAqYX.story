Recognizing the relationship between two texts is an important aspect of natural language understanding (NLU), and a variety of neural network models have been proposed for solving NLU tasks.

Unfortunately, recent work showed that the datasets these models are trained on often contain biases that allow models to achieve non-trivial performance without possibly learning the relationship between the two texts.

We propose a framework for building robust models by using adversarial learning to encourage models to learn latent, bias-free representations.

We test our approach in a Natural Language Inference (NLI) scenario, and show that our adversarially-trained models learn robust representations that ignore known dataset-specific biases.

Our experiments demonstrate that our models are more robust to new NLI datasets.

Recognizing the relationship between two texts is a significant aspect of general natural language understanding (NLU) BID2 .

Natural Language Inference (NLI) is often used to gauge a model's ability to understand such a relationship between two texts BID11 BID12 .

In NLI, a model is tasked with determining whether a hypothesis (the animal moved) would likely be inferred from a premise (a black cat ran).

The development of new large-scale datasets has led to a flurry of various neural network architectures for solving NLI.

However, recent work has found that many NLI datasets contain biases that enable hypothesis-only models -models that are given access to the hypothesis alone -to perform surprisingly well without possibly learning the relationship between two texts.

For instance, annotation artifacts and statistical irregularities in the popular Stanford Natural Language Inference dataset (SNLI) BID5 allowed hypothesis-only models to perform at double the majority class baseline, and at least 5 other recent NLI datasets contain similar biases BID21 BID43 BID54 .

We will use the terms "artifacts" and "biases" interchangeably.

The existence of annotation artifacts in large-scale NLI datasets is detrimental for making progress in deep learning research for NLU.

How can we trust the performance of top models if it is possible to infer the relationship without even looking at the premise?

Solutions to this concern are so far unsatisfactory: constructing new datasets BID50 ) is costly and may still result in other artifacts; filtering "easy" examples and defining a harder subset is useful for evaluation purposes BID21 , but difficult to do on a large scale that will enable training; and compiling adversarial examples BID18 ) is informative but again limited by scale or diversity.

Furthermore, these solutions do not address a lingering question: can we develop models that will generalize well despite many NLI datasets containing specific hypothesis-only biases?Inspired by domain-adversarial training of neural networks BID16 BID17 , we propose two architectures (Figure 1 ) that enable a model to perform well on other NLI datasets regardless of what annotation artifacts exist in the training corpus's hypotheses.

While learning to classify the relationship between two texts, we simultaneously use adversarial learning to discourage our model from using dataset-specific biases.

In this way, the resulting representations contain fewer biases, and the model is encouraged to learn the relationship between the two texts.

Our experiments demonstrate that our architectures generate sentence representations that are more robust to annotation artifacts, and also transfer better: when trained on one dataset and evaluated on another, they perform better than a non-adversarial model in 9 out of 12 target datasets.

The methodology can also be extended to other NLU tasks, and we outline the necessary changes to our architectures in the conclusion.

To our knowledge, this is the first study that explores methods to ignore hypothesis-only biases when training NLI models.

Annotation artifacts or biases were found in multiple NLU datasets and tasks.

Early work on NLI, also known as recognizing textual entailment (RTE), found biases that allowed models to perform relatively well by focusing on syntactic clues alone BID51 BID55 .

More recently, Gururangan et al. (2018) found such artifacts in SNLI and its multi-genre extension, MNLI BID58 .

BID54 observed similar findings in SNLI and SICK, a sentence similarity dataset, and BID43 reported similar results on six datasets, including those previous three.

Some of the biased annotations that were found include the use of negation words ("not", "nothing") for cases of contradiction or approximate words ("some", "various") for entailment.

BID43 discussed how faulty dataset-construction methods may be responsible for these biases.

Other NLU datasets also exhibit biases.

In a story cloze completion setup, BID49 obtained a high performance by only considering the candidate endings, without even looking at the story context.

In the ROC stories dataset BID33 , stylistic features such as length or the use of certain words are predictive of the correct ending BID48 BID6 .

A similar phenomenon was observed in reading comprehension, where systems performed non-trivially well by only using the final sentence in the passage, or by ignoring the passage altogether BID27 .

Finally, multiple studies found non-trivial performance in visual question answering by using only the question, without any access to the image, due to biases in the question text BID24 BID20 BID25 BID1 .

Neural networks are notoriously sensitive to adversarial examples, primarily in the machine vision field BID52 BID19 , but also in NLP tasks like machine translation BID14 BID3 BID22 and reading comprehension BID23 BID46 BID35 .

A common approach to improving robustness is to train the model on data including adversarial examples BID52 BID19 .

However, this method may not generalize well to new types of adversarial examples BID59 BID53 BID3 .

Domain-adversarial neural networks aim to increase robustness to domain change, by learning to be oblivious to the domain BID17 .

This approach requires knowledge of the domain at training time, which makes transfer learning more difficult.

Our method relies on a similar idea, but we learn to ignore latent annotation artifacts.

Hence, we do not require direct supervision in the form of a domain label.

Others have attempted to remove biases from learned representations.

BID4 successful removed gender biases from word embeddings.

BID30 removed sensitive information like sex and age from text representations, and obtained improved representations especially in out-of-domain scenarios.

However, other work suggests that removing such attributes from text representations may be difficult BID15 .

In contrast to this line of work, our final goal is not the removal of such attributes per se; instead, we strive for more robust representations that better transfer to other datasets, similar to BID30 .

Very recent work has focused on applying adversarial learning to NLI.

BID32 generate adversarial examples that do not conform to logical rules and then regularize models based on those examples.

Similarly, BID26 incorporate external linguistic resources and use a GAN-style framework to adversarially train robust NLI models.

Unlike these works, we do not use external resources and we are interested in removing specific biases that allow hypothesis-only models to perform well.

Let (P, H) denote a premise-hypothesis pair, and let f : S → v denote an encoder that maps a sentence S to a vector representation v, and g : v →

y a classifier that maps a vector representation v to an output label y. Our baseline NLI architecture (Figure 1a) contains the following components: Figure 1: Illustration of (a) the baseline NLI architecture, and two proposed adversarial architectures: (b) a double-classifier adds an adversarial hypothesis-only classifier, (c) a single-classifier is trained adversarially with a random premise, and otherwise in the normal manner.

Upward and downward arrows correspond to forward and backward propagation.

Green or red arrows respectively mean that the gradient sign is kept as is or reversed.

Gray arrow indicates that the gradient is blocked and not back-propagated.• A premise encoder f P that maps the premise P to a vector representation p.• A hypothesis encoder f H that maps the hypothesis H to a vector representation h.• A classifier g NLI that maps a premise-hypothesis vector representation to an output y.

In this model, the premise and hypothesis are each encoded with separate encoders, f P and f H , into p and h respectively.

1 Then their representations are combined into [p; h] 2 and fed to a classifier g NLI , which predicts an output y. If f P is not used, a model should no longer be able to successfully perform NLI.

However, models without f P achieve non-trivial results, indicating the existence of biases in datasets' hypotheses BID21 BID43 BID54 .To overcome such biases that may limit the ability to transfer models across NLI datasets, we design two kinds of models for robust NLI, a single-classifier model and a double-classifier model.

Both models aim to encourage the hypothesis representations to be free of biases via adversarial learning.

The double-classifier model, illustrated in Figure 1b , is similar to our baseline model but includes an adversarial classifier g Hypoth that maps the hypothesis representation h to an output y. The crucial aspect of this model is the interaction between the NLI classifier, g NLI , and the hypothesis classifier, g Hypoth .

The NLI classifier is trained to minimize the following objective: DISPLAYFORM0 where L(ỹ, y) is the cross-entropy loss.

In the forward step, we feed the premise and hypothesis to their respective encoders, f P and f H , and forward their joint representation [p; h] to the classifier.

In the backward step, gradients are back-propagated from the classifier into the premise and hypothesis encoders, f P and f H , in the normal fashion.

In contrast, the hypothesis classifier g Hypoth is adversary with respect to the input.

It is also trained to minimize the cross-entropy loss, but its forward/backward propagation is different.

In the forward step, we feed the hypothesis into its encoder f H and forward its representation h to the classifier.

In the backward step, we first back-propagate gradients through the classifier, as before.

However, before back-propagating to the encoder, we reverse the gradients BID16 .

This simple step aims to discourage the model from learning patterns that may be useful for classification when considering only the hypothesis.

The adversary minimizes the following objective: DISPLAYFORM1 where GRL λ is a gradient reversal layer.

To control the interplay between g NLI and g Hypoth we set two hyper-parameters: λ Loss controls the importance of the adversarial loss function, and λ Enc controls the weight of the adversarial update by multiplying the gradients after reversing them.

This is implemented by the scaled gradient reversal layer, GRL λ .

The final loss function is defined by: DISPLAYFORM2 Limitation and a cryptographic perspective The double-classifier model is conceptually simple.

However, it has a potential limitation because of the separate adversarial classifier.

In theory, it is possible that the adversary and the hypothesis encoder will co-adapt during training, such that the hypothesis representation still contains biased information but the adversary cannot utilize it.

Thus we may be fooled to think that the biases were removed, while in fact they are encoded in a way that is accessible to the normal classifier but not to the adversary.

A similar situation arises in neural cryptography BID0 , where an encryptor Alice and a decryptor Bob communicate while an adversary Eve tries to eavesdrop on their communication.

Alice and Bob are analogous to the hypothesis encoder f H and the normal classifier g NLI , while Eve is analogous to the adversary g Hypoth .

Secret communication is analogous to solving NLI without using biases.

In their asymmetric encryption experiments, Abadi & Andersen observed seemingly secret communication, which on closer look the adversary was able to eavesdrop on.

Here, if the adversarial classifier does not perform well, we might be tricked into thinking that the encoded representation does not contain any biases, while in fact they are still hidden in the representation.

Our next architecture aims to prevent such a situation from possibly happening by folding the NLI and adversarial classifiers into a single network.

The single-classifier model also includes the premise and hypothesis encoders, f P and f H .

However, it only has one classifier, g, which acts as both a normal NLI classifier and an adversarial hypothesis classifier.

Having only one classifier aims at reducing the risk that a separate adversary would give a false impression of success, learning to not use hidden biases in the hypothesis representations.

Since g must also do well on NLI with normal training, it is less likely to ignore hidden biases.

To achieve this, we consider two modes of operation.

In the normal mode, we get a premisehypothesis pair from the training data, feed them through their encoders, forward to the classifier, and back-propagate in the normal fashion.

In the adversarial mode, we get a premise-hypothesis from the training data and replace the premise with a random one.

In the forward step, we feed the new pair to the encoders and classifier as before, predicting the entailment decision corresponding to the original premise-hypothesis pair.

In the backward step, we first back-propagate through the classifier as usual.

Then, we block the premise encoder and only back-propagate to the hypothesis encoder.

In this case, we reverse the gradients going into the hypothesis encoder, as in the doubleclassifier.

This procedure is shown in Figure 1c .

The adversarial loss function is defined as: DISPLAYFORM0 Here GRL 0 implements gradient blocking on the premise encoder by using the identity function in the forward step and a zero gradient during the backward step.

At the same time, GRL λ reverses the gradient going into the hypothesis encoder and scales it by λ Enc , as before.

The single-classifier model has the advantage of a simpler architecture, avoiding the need to train two different classifiers, which may result in failed adversarial learning as described before.

However, it has a more complicated training regime, in choosing random examples.

In practice, we set a hyper-parameter λ Rand ∈ [0, 1] that controls what fraction of the examples are random.

The final loss function combines the two operation modes with a random variable z ∼ Bernoulli(λ Rand ): DISPLAYFORM1

Data To determine how well our proposed architectures enable a model to perform well on NLI datasets despite the high presence of the annotation artifacts in the training corpus, we use a total of 11 NLI datasets -the 10 datasets that BID43 investigated in their hypothesis-only study plus GLUE's diagnostic test set that was carefully constructed to not contain hypothesisbiases BID56 .

The most popular recent NLI datasets are arguably the Stanford Natural Language Inference (SNLI) BID5 dataset and its successor, the Multi-genre Natural Language Inference (MNLI) BID58 dataset.

These datasets are human elicited: they were created by having human generate a corresponding hypothesis for a given premise and NLI label.

Since SNLI is known to contain significant annotation artifacts/biases BID21 BID43 ), we will demonstrate the robustness of our methods by training our adversarial models on SNLI, and evaluating on all other datasets.

We also evaluate on SNLI-hard, a subset of the test set that is thought to contain fewer biases BID21 .The second category of NLI datasets we consider are human-judged datasets that used automatic methods to pair context and hypothesis sentences and then relied on humans to label the pairs: Scitail , ADD-ONE-RTE BID38 , Johns Hopkins Ordinal Commonsense Inference (JOCI) BID62 , Multiple Premise Entailment (MPE) BID29 , and Sentences Involving Compositional Knowledge (SICK) BID31 .

We also consider datasets that are automatically recast from existing NLU datasets into NLI.

We use the three datasets recast by BID57 to evaluate a number of semantic phenomena: FrameNet+ (FN+) (Pavlick et al., 2015) , Definite Pronoun Resolution (DPR) BID44 , Semantic Proto-Roles (SPR) BID45 .As many of these target datasets have different label spaces than SNLI, we define a mapping from our models' predictions to each target dataset's labels.

These mappings are available in Appendix A.1.

We adopt InferSent's method BID9 for learning sentence representations from NLI data as our basic NLI architecture.

In InferSent, each sentence is encoded by a bidirectional long short-term memory (LSTM) network whose hidden states are max-pooled.

The premise and hypothesis representations are combined via a method introduced by BID34 and passed to a one hidden layer neural network.

3 We chose this model because it works well and its architecture is representative of many NLI models.

However, our methodology can be applied to other models as well.

We follow the InferSent training regime, using SGD with an initial learning rate of 0.1.

See Appendix B.1 for hyper-parameter settings and more details.

TAB0 reports the results of our proposed architectures compared to the non-adversarial baseline model.

In each case, we tune the hyper-parameters on each target dataset's development set and evaluate on the corresponding test set.

4 The double-classifier model outperforms the baseline in 9 of the 12 target datasets (∆ > 0), though most of the improvements are small.

The single-classifier only outperforms the baseline in 5 datasets, 4 of which are cases where the double-classifier also outperformed the baseline.

These gains are much larger than the gains of the double-classifier.

The fact that the two architectures agree to a large extent on which datasets benefit from adversarial training is a validation of our basic approach.

As our results improve on the target datasets, we note that the double classifier models' performance on SNLI does not drastically decrease, even when the improvement on the target dataset is large (for example, the SPR case).

For these models, the performance on SNLI drops by just an average of 1.11 (0.65 STDV).

For the single-classifier, there is a large decrease on SNLI for many of the adversarial models as the models drop by an average of 11.19 (12.71 STDV).

For these models, when we see large improvement on a target dataset, we often see a large drop on SNLI.

For example, on ADD-ONE-RTE, the single classifier outperforms the baseline by roughly 17% but performs almost 50% lower than the baseline on SNLI.

A priori, we expect the adversarial models to benefit most when there are either no biases or biases that differ from ones in the training data.

Indeed, both our adversarial architectures obtain improved results on the GLUE diagnostic test set, which was designed to be bias-free.

We do not see improvements on the SNLI hard subset BID21 , indicating that it may still have biases not identified by the authors, a possibility they also acknowledge.

To estimate the amount of bias that differ in the datasets, we compare the hypothesis-only results from BID43 with a hypothesis-only model trained on SNLI and tested on the target datasets.

Since the results drop significantly below the majority class baseline (MAJ) on all but one dataset (Figure 4 , Appendix C.2), we believe that these target NLI datasets contain different biases than those in SNLI.

The largest difference is on SPR where the hypothesis-only model trained on SNLI performs over 50% worse than when trained on SPR.

On MNLI, this hypothesis-only model performs 10% above MAJ, compared to the roughly 20% when trained on MNLI, suggesting that MNLI contains similar biases as SNLI.

This may explain why our adversarial models only slightly outperform our baseline on MNLI.

The hypothesis-only model of BID43 did not outperform MAJ on DPR, ADD-ONE-RTE, SICK, and MPE.

We observe improvements with adversarially trained models that are tested on all these datasets, to varying degrees (from 0.45 on MPE to 31.11 on SICK).

On the other hand, we also see improvements on datasets with biases (high performance of hypothesis-only model), most noticeably on SPR.

The only exception seems to be SCITAIL, where we do not improve although it has different biases than SNLI.

However, when we strengthen the adversary (in the analysis below), the double-classifier outperforms the baseline.

Our results demonstrates that our approach is robust to many datasets with different types of biases.

Fine-tuning on target datasets Our main goal is to determine whether adversarial learning can allow a model to perform well across multiple datasets by ignoring dataset-specific artifacts.

In turn, we did not update the models' parameters on other datasets.

However, what if we are given different amounts of training data for a new NLI dataset?

Is our adversarial approach still helpful?

To answer these questions, we updated four existing models, on increasing sizes of training data from different target datasets (namely, MNLI and SICK).

The four models are (1) our baseline model trained on SNLI, (2) an adversarial double classifier trained on SNLI, (3) an adversarial single classifier trained on SNLI, and (4) our baseline model trained on the target dataset (MNLI/SICK).

Both MNLI and SICK have the same label spaces as SNLI, allowing us to hold that variable constant.

We use SICK because our adversarial models achieved good gains on it TAB0 .

We also use MNLI, even though we saw small gains there, because MNLI's training set is large, allowing us to consider a wide range of different training set sizes.

6 FIG1 shows the results on the dev sets.

In MNLI, there is little to no gain from adversarial pre-training compared to non-adversarial pre-training.

This is expected, as we saw relatively small gains with the adversarial models, and can be explained by SNLI and MNLI having similar biases.

In SICK, adversarial pre-training is better in most data regimes.

The single-classifier is especially helpful, as it is the first to beat the model without pre-training (after using 25% of the training data).

Stronger adversary Does a stronger adversary improve the quality of the transferred representations?

Intuitively, we expect more adversarial training to hurt performance on the original data, where biases are useful, but improve on target data, which may have different or no biases.

To test this, we trained adversarial models with larger hyper-parameter values on the adversary (see details in Appendix B.1).

While there are other ways to make adversaries stronger, such as increasing the number or size of hidden layers BID15 , we are especially interested in the effect of these hyper-parameters as they control the trade-off between normal and adversarial training.

TAB1 shows the results of double-classifier models with a stronger adversary.

As expected, performance on SNLI test sets decreases more, but many of the other datasets benefit from a stronger adversary (compared with TAB0 ).

As for the single-classifier model, we found large drops in quality even in our basic configurations (Appendix C.3), so we do not increase its strength further.

Hidden biases in the representation Recall our motivation for the single-classifier model: we were concerned that the learned hypothesis representations may seem to be bias-free (low adversarial performance), while in fact there are still biases hidden in them.

In the cryptographic analogy, it might appear that Alice and Bob communicate secretly (akin to solving NLI without biases), while in fact their communication can be decrypted (has biases).

Indeed, BID0 found that, "upon resetting and retraining Eve, the retrained adversary was able to decrypt messages nearly as well as Bob was".

We perform an analogous experiment here: given a trained adversarial model, we freeze the hypothesis encoder f H and retrain a new, hypothesis-only classifier.

We evaluate its quality to determine whether the (frozen) hypothesis representations have hidden biases.

FIG2 shows the results on SNLI's dev set.

A few trends can be noticed.

First, we confirm that, in the double-classifier case FIG2 , the adversary is indeed trained to perform poorly on the task (orange line), while the normal NLI classifier (blue line) performs much better.

However, as suspected, retraining a classifier on frozen hypothesis representations (green line) leads to improved performance.

In fact, the retrained classifier performs close to the fully trained hypothesis-only baseline from BID43 , indicating that the hypothesis representation still contains biases.

Interestingly, we found that even a frozen random encoder captures biases in the hypothesis, as a classifier trained on it performs fairly well (63.26%), and far above the majority baseline (34.28%).

One reason might be that just the word embeddings (which are pre-trained) contain significant information that propagates even through a random encoder.

Others have also found that random encodings contain non-trivial information BID10 BID60 .

Relatedly, Indicator words Certain words in SNLI are more correlated with specific entailment labels than others, especially with negation words ("not", "nobody", "no") correlated with contradiction BID21 BID43 ).

Here we investigate whether our adversarial models make predictions that are less impacted by such biases.

For each of the most biased contradiction words in SNLI, we computed the probability that an adversarial model (or the baseline model) predicts an example as contradiction, given that the hypothesis has the word.

Table 3 shows the top 5 examples in the training set (Appendix C.4 has more examples).

The single-classifier with λ Rand = 0.4, λ Enc = 1 predicts contradiction much less frequently than the baseline on examples with these words.

This configuration was the strongest adversarial model that still performed reasonably well on SNLI (Appendix C.3).

With these hyper-parameters, the single-classifier appears to remove some of the biases learned by the baseline.

We also provide two other adversarial configurations that do not show such an effect, to illustrate that this behavior highly depends on the adversarial hyper-parameters.

Biases in annotations are a major source of concern for the quality of NLI datasets and systems.

In this paper, we presented a solution for combating annotation biases based on adversarial learning.

We designed two architectures that discourage the hypothesis encoder from learning the biases, and instead obtain a more unbiased representation.

We empirically evaluated our approach in a transfer learning scenario, where we found our models to perform better than a non-adversarial baseline on a range of datasets.

We also investigated what biases remain in the latent representations.

The methodology developed in this work can be extended to deal with biases in other NLU tasks, where one is concerned with finding the relationship between two objects.

For example, in Reading Comprehension, a question is being asked about a passage; in story cloze completion, an ending is judged with respect to a context; and in Visual Question Answering, a question is asked about an image.

In all these cases, the second element (question, ending, and question, respectively) may contain biases.

Our adversarial architectures naturally apply to any model that relies on encoding this biased element, and may help remove such biases from the latent representation.

We hope to encourage such investigation in the broader research community.

TAB1 we consider the range {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}. In each dataset, we choose the best-performing model on the development set and report its quality on the test set.

We follow the InferSent training regime, using SGD with an initial learning rate of 0.1.

See BID9 for details.

One drawback of using MNLI is that gold labels for the test set are not publicly available.

We remove 10K examples from the training set and treat them as our development set.

Here, we use the MNLI matched development set as our test set since we assume that the new dataset, that we would like our robust NLI models to generalize well on, contains consistent domains and genres.

C ADDITIONAL RESULTS C.1 PERFORMANCE ON SNLI WITH ADVERSARIAL TRAINING TAB4 shows the results of transferring representations to new datsets.

These results complement TAB0 .

In each dataset, we tune the adversarial hyper-parameters on the dev set and report test set accuracies.

We also give the performance on the SNLI test set.

Figure 4 shows the results of several baselines in multiple datasets: the majority baseline and the hypothesis-only baselines trained on either SNLI or each dataset's training set, and evaluated on each target dataset's test set.

Notice that the hypothesis-only model trained on SNLI drops in performance when evaluated on other models.

When the hypothesis-only model trained on SNLI is tested on the target datasets, the model performs below the majority baseline (except for MNLI), indicating that the biases in SNLI's hypotheses do not occur in the other datasets' hypotheses.

This is not surprising since these datasets contain different types of biases owing to noisy generation methods (FN+) BID41 , social and cognitive biases (SNLI & MNLI) BID43 , or even "non-uniform distributions from original dataset that have been recast" (SPR) BID42 .Figure 4: Majority (Maj) and hypothesis-only baselines, trained on each dataset (Hyp Self) or on SNLI (Hyp SNLI) baselines.

The differences determine whether each dataset has different types of biases in the hypothesis than the biases in SNLI's hypotheses.

Here we provide cross-validation results with different settings of our hyper-parameters.

FIG3 shows the dev set results with different configurations of the single classifier.

Notice that performance degrades quickly when we increase the fraction of random premises (large λ Rand ).

In contrast, the results with the double classifier ( FIG3 ) are more stable.

TAB5 shows the top 20 indicator words in SNLI, to complement the discussion in Section 6.

<|TLDR|>

@highlight

Adversarial learning methods encourage NLI models to ignore dataset-specific biases and help models transfer across datasets.

@highlight

The paper proposes an adversarial setup to mitigate annotation artifacts in natural language inference data

@highlight

This paper presents a method for removing bias of a textual entailment model through an adversarial training objective. 