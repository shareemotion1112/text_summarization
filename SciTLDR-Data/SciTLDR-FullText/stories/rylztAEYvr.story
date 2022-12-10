Many challenging prediction problems, from molecular optimization to program synthesis, involve creating complex structured objects as outputs.

However, available training data may not be sufficient for a generative model to learn all possible complex transformations.

By leveraging the idea that evaluation is easier than generation, we show how a simple, broadly applicable, iterative target augmentation scheme can be surprisingly effective in guiding the training and use of such models.

Our scheme views the generative model as a prior distribution, and employs a separately trained filter as the likelihood.

In each augmentation step, we filter the model's outputs to obtain additional prediction targets for the next training epoch.

Our method is applicable in the supervised as well as semi-supervised settings.

We demonstrate that our approach yields significant gains over strong baselines both in molecular optimization and program synthesis.

In particular, our augmented model outperforms the previous state-of-the-art in molecular optimization by over 10% in absolute gain.

Deep architectures are becoming increasingly adept at generating complex objects such as images, text, molecules, or programs.

Many useful generation problems can be seen as translation tasks, where the goal is to take a source (precursor) object such as a molecule and turn it into a target satisfying given design characteristics.

Indeed, molecular optimization of this kind is a key step in drug development, though the adoption of automated tools remains limited due to accuracy concerns.

We propose here a simple, broadly applicable meta-algorithm to improve translation quality.

Translation is a challenging task for many reasons.

Objects are complex and the available training data pairs do not fully exemplify the intricate ways in which valid targets can be created from the precursors.

Moreover, precursors provided at test time may differ substantially from those available during training -a scenario common in drug development.

While data augmentation and semisupervised methods have been used to address some of these challenges, the focus has been on either simple prediction tasks (e.g., classification) or augmenting data primarily on the source side.

We show, in contrast, that iteratively augmenting translation targets significantly improves performance on complex generation tasks in which each precursor corresponds to multiple possible outputs.

Our iterative target augmentation approach builds on the idea that it is easier to evaluate candidate objects than to generate them.

Thus a learned predictor of target object quality (a filter) can be used to effectively guide the generation process.

To this end, we construct an external filter and apply it to the complex generative model's sampled translations of training set precursors.

Candidate translations that pass the filter criteria become part of the training data for the next training epoch.

The translation model is therefore iteratively guided to generate candidates that pass the filter.

The generative model can be viewed as an adaptively tuned prior distribution over complex objects, with the filter as the likelihood.

For this reason, it is helpful to apply the filter at test time as well, or to use the approach transductively 1 to adapt the generation process to novel test cases.

The approach is reminiscent of self-training or reranking approaches employed with some success for parsing (McClosky et al., 2006; Charniak et al., 2016) .

However, in our case, it is the candidate generator that is complex while the filter is relatively simple and remains fixed during the iterative process.

We demonstrate that our meta-algorithm is quite effective and consistent in its ability to improve translation quality in the supervised setting.

On a program synthesis task (Bunel et al., 2018) , under the same neural architecture, our augmented model outperforms their MLE baseline by 8% and their RL model by 3% in top-1 generalization accuracy (in absolute measure).

On molecular optimization (Jin et al., 2019a) , their sequence to sequence translation baseline, when combined with our target data augmentation, achieves a new state-of-the-art result and outperforms their graph based approach by over 10% in success rate.

Their graph based methods are also improved by iterative target augmentation with more than 10% absolute gain.

The results reflect the difficulty of generation in comparison to evaluation; indeed, the gains persist even if the filter quality is reduced somewhat.

Source side augmentation with unlabeled precursors (the semi-supervised setting) can further improve results, but only when combined with the filter in the target data augmentation framework.

We provide ablation experiments to empirically highlight the effect of our method and also offer some theoretical insights for why it is effective.

Molecular Optimization The goal of molecular optimization is to learn to modify compounds so as to improve their chemical properties.

Jaques et al. (2017) ; You et al. (2018) ; Popova et al. (2018) used reinforcement learning approaches, while Jin et al. (2019a; b) formulated this problem as graphto-graph translation and significantly outperformed previous methods.

However, their performance remains imperfect due to the limited size of given training sets.

Our work uses property prediction models to check whether generated molecules have desired chemical properties.

Recent advances in graph convolutional networks (Duvenaud et al., 2015; Gilmer et al., 2017) have provided effective solutions to predict those properties in silico.

In this work, we use an off-the-shelf property prediction model (Yang et al., 2019) to filter proposed translation pairs during data augmentation.

Program Synthesis Program synthesis is the task of generating a program (using domain-specific language) based on given input-output specifications (Bunel et al., 2018; Gulwani, 2011; Devlin et al., 2017) .

One can check a generated program's correctness by simply executing it on each input and verifying its output.

Indeed, Zhang et al. (2018) ; Chen et al. (2019) leverage this idea in their respective decoding procedures, while also using structural constraints on valid programs.

Semi-supervised Learning Our method is related to various approaches in semi-supervised learning.

In image and text classification, data augmentation and label guessing (Berthelot et al., 2019; Xie et al., 2019) are commonly applied to obtain artificial labels for unlabeled data.

In machine translation, Norouzi et al. (2016) sample new targets from a stationary distribution in order to match the model distribution to the exponentiated payoff distribution centered at a single target sentence.

Back-translation (Sennrich et al., 2015; Edunov et al., 2018) creates extra translation pairs by using a backward translation system to translate unlabeled sentences from a target language into a source language.

In contrast, our method works in the forward direction because many translation tasks are not symmetric.

Moreover, our data augmentation is carried out over multiple iterations, in which we use the augmented model to generate new data for the next iteration.

In syntactic parsing, our method is closely related to self-training (McClosky et al., 2006) .

They generate new parse trees from unlabeled sentences by applying an existing parser followed by a reranker, and then treat the resulting parse trees as new training targets.

However, their method is not iterative, and their reranker is explicitly trained to operate over the top k outputs of the parser; in contrast, our filter is independent of the generative model.

In addition we show that our approach, which can be viewed as iteratively combining reranking and self-training, is theoretically motivated and can improve the performance of highly complex neural models in multiple domains.

Co-training (Blum & Mitchell, 1998) and tri-training (Zhou & Li, 2005; Charniak et al., 2016 ) also augment a parsing dataset by adding targets on which multiple baseline models agree.

Instead of using multiple learners, our method uses task-specific constraints to select correct outputs.

Our iterative target augmentation framework can be applied to any conditional generation task with task-specific constraints.

For example, molecular optimization (Jin et al., 2019a; b) is the task of transforming a given molecule X into another compound Y with improved chemical properties, Figure 1 : Illustration of our data generation process in the program synthesis setting.

Given an input-output specification, we first use our generation model to generate candidate programs, and then select correct programs using our external filter.

Images of input-output specification and the program A are from Bunel et al. (2018) .

for attempt in 1, . . .

, C do Train model on D.

14:

for epoch in 1, . . . , n 2 do Iterative target augmentation 15:

while constraining Y to remain similar to X. Program synthesis (Bunel et al., 2018; Chen et al., 2019) is the task of generating a program Y satisfying input specification X; for example, X may be a set of input-output test cases which Y must pass.

Without loss of generality, we formulate the generation task as a translation problem.

For a given input X, the model learns to generate an output Y satisfying the constraint c. The proposed augmentation framework can be applied to any translation model M trained on an existing dataset

As illustrated in Figure 1 , our method is an iterative procedure in which each iteration consists of the following two steps:

Step: Let D t be the training set at iteration t. To construct each next training set D t+1 , we feed each input X i ∈ D (the original training set, not D t ) into the translation model up to C times to sample C candidate translations

We take the first K distinct translations for each X i satisfying the constraint c and add them to D t+1 .

When we do not find K distinct valid translations, we simply add the original translation Y i to D t+1 .

• Training Step: We continue to train the model M t over the new training set D t+1 for one epoch.

The above training procedure is summarized in Algorithm 1.

As the constraint c is known a priori, we can construct an external filter to remove generated outputs that violate c during the augmentation step.

At test time, we also use this filter to screen predicted outputs.

To propose the final translation of a given input X, we have the model generate up to L outputs until we find one satisfying the constraint c. If all L attempts fail for a particular input, we just output the first of the failed attempts.

Finally, as an additional improvement, we observe that the augmentation step can be carried out for unlabeled inputs X that have no corresponding Y .

Thus we can further augment our training dataset in the transductive setting by including test set inputs during the augmentation step, or in the semi-supervised setting by simply including unlabeled inputs.

We provide here some theoretical motivation for our iterative target augmentation framework.

For simplicity, we consider an external filter c X,Y that is a binary indicator function representing whether output Y satisfies the desired constraint in relation to input X. In other words, we would like to generate Y such that Y ∈ B(X) = {Y |c X,Y = 1}. If the initial translation model P (0) (Y |X) serves as a reasonable prior distribution over outputs, we could simply "invert" the filter and use

as the ideal translation model.

While this posterior calculation is typically not feasible but could be approximated through samples, it relies heavily on the appropriateness of the prior (model prior to augmentation).

Instead, we go a step further and iteratively optimize our parametrically defined prior translation model P θ (Y |X).

Note that the resulting prior can become much more concentrated around acceptable translations.

We maximize the log-likelihood that candidate translations satisfy the constraints implicitly encoded in the filter

In many cases there are multiple viable outputs for any given input X. The training data may provide only one (or none) of them.

Therefore, we treat the output structure Y as a latent variable, and expand the inner term of Eq. (2) as

Since the above objective involves discrete latent variables Y , we propose to maximize Eq.(5) using the standard EM algorithm (Dempster et al., 1977) , especially its incremental, approximate variant.

The target augmentation step in our approach is a sampled version of the E-step where the posterior samples are drawn with rejection sampling guided by the filter.

The number of samples K controls the quality of approximation to the posterior.

3 The additional training step based on the augmented targets corresponds to a generalized M-step.

More precisely, let P (t) θ (Y |X) be the current translation model after t epochs of augmentation training.

In epoch t + 1, the augmentation step first samples C different candidates for each input X using the old model P (t) parameterized by θ (t) , and then removes those which violate the constraint c, interpretable as samples from the current posterior

As a result, the training step maximizes the EM auxiliary objective via stochastic gradient descent:

We train the model with multiple iterations and show empirically that model performance indeed keeps improving as we add more iterations.

The EM approach is likely to converge to a different and better-performing translation model than the initial posterior calculation discussed above.

We demonstrate the broad applicability of iterative target augmentation by applying it to two tasks of different domains: molecular optimization and program synthesis.

Figure 2: Illustration of molecular optimization.

Molecules can be modeled as graphs, with atoms as nodes and bonds as edges.

Here, the task is to train a translation model to modify a given input molecule into a target molecule with higher drug-likeness (QED) score.

The constraint has two components: the output Y must be highly drug-like, and must be sufficiently similar to the input X.

The goal of molecular optimization is to learn to modify molecules so as to improve their chemical properties.

As illustrated in Figure 2 , this task is formulated as a graph-to-graph translation problem.

Similar to machine translation, the training set is a set of molecular pairs {(X, Y )}.

X is the input molecule (precursor) and Y is a similar molecule with improved chemical properties.

Each molecule in the training set D is further labeled with its property score.

Our method is well-suited to this task because the target molecule is not unique: each precursor molecule can be modified in many different ways to optimize its properties.

External Filter The constraint for this task contains two parts: 1) the chemical property of Y must exceed a certain threshold β, and 2) the molecular similarity between X and Y must exceed a certain threshold δ.

The molecular similarity sim(X, Y ) is defined as Tanimoto similarity on Morgan fingerprints (Rogers & Hahn, 2010) , which measures structural overlap between two molecules.

In real world settings, ground truth values of chemical properties are often evaluated through experimental assays, which are too expensive and time-consuming to run for iterative target augmentation.

Therefore, we construct an in silico property predictor F 1 to approximate the true property evaluator F 0 .

To train this property prediction model, we use the molecules in the training set and their labeled property values.

The predictor F 1 is parameterized as a graph convolutional network and trained using the Chemprop package (Yang et al., 2019) .

During data augmentation, we use F 1 to filter out molecules whose predicted property is under the threshold β.

We follow the evaluation setup of Jin et al. (2019b) for two molecular optimization tasks:

1.

QED Optimization: The task is to improve the drug-likeness (QED) of a given compound X.

The task is to optimize biological activity against the dopamine type 2 receptor (DRD2).

The similarity constraint is sim(X, Y ) ≥ 0.4 and the property constraint is DRD2(Y ) ≥ 0.5, where DRD2(Y ) ∈ [0, 1] is the predicted probability of biological activity given by the model from Olivecrona et al. (2017) .

We treat the output of the in silico evaluators from Bickerton et al. (2012) and Olivecrona et al. (2017) as ground truth, and we use them only during test-time evaluation.

Evaluation Metrics.

During evaluation, we are interested both in the probability that the model will find a successful modification for a given molecule, as well as the diversity of the successful modifications when there are multiple.

Table 1 : Performance of different models on QED and DRD2 optimization tasks.

Italicized models with + are augmented with iterative target augmentation.

We emphasize that iterative target augmentation remains critical to performance in the semi-supervised and transductive settings; data augmentation without an external filter instead decreases performance.

1.

Success: The fraction of molecules X for which any of the outputs Y 1 . . .

Y Z meet the required similarity and property constraints (specified previously for each task).

This is our main metric.

2. Diversity: For each molecule X, we measure the average Tanimoto distance (defined as 1 − sim(Y i , Y j )) between pairs within the set of successfully translated compounds among Y 1 . . .

Y Z .

If there are one or fewer successful translations then the diversity is 0.

We average this quantity across all test molecules.

We consider the following two model architectures from Jin et al. (2019a) to show that our augmentation scheme is not tied to specific neural architectures.

1. VSeq2Seq, a sequence-to-sequence translation model generating molecules by their SMILES string (Weininger, 1988) .

2.

HierGNN, a hierarchical graph-to-graph architecture that achieves state-of-the-art performance on the QED and DRD2 tasks, outperforming VSeq2Seq by a wide margin.

We apply our iterative augmentation procedure to the above two models, generating up to K = 4 new targets per precursor during each epoch of iterative target augmentation.

Additionally, we evaluate our augmentation of VSeq2Seq in a transductive setting, as well as in a semi-supervised setting where we provide 100K additional source-side precursors from the ZINC database (Sterling & Irwin, 2015) .

Full hyperparameters are in Appendix A.

As shown in Table 1 , our iterative augmentation paradigm significantly improves the performance of VSeq2Seq and HierGNN.

On both datasets, the translation success rate increases by over 10% in absolute terms for both models.

In fact, VSeq2Seq+, our augmentation of the simple VSeq2Seq model, outperforms the non-augmented version of HierGNN.

This result strongly confirms our hypothesis about the inherent challenge of learning translation models in data sparse scenarios.

Moreover, we find that adding more precursors during data augmentation further improves the VSeq2Seq model.

On the QED dataset, the translation success rate improves from 89.0% to 92.6% by just adding test set molecules as precursors (VSeq2Seq+, transductive).

When instead adding 100K presursors from the external ZINC database, the performance further increases to 95.0% (VSeq2Seq+, semisupervised).

We observe similar improvements for the DRD2 task as well.

Beyond accuracy gain, our augmentation strategy also improves the diversity of generated molecules.

For instance, on the DRD2 dataset, our approach yields 100% relative gain in terms of output diversity.

Although the property predictor used in data augmentation is different from the ground truth property evaluator used at test time, the difference in evaluators does not derail the overall training process.

Here we analyze the influence of the quality of the property predictor used in data augmentation.

Specifically, we rerun our experiments using less accurate predictors in the property-predicting component of our external filter.

We obtain these less accurate predictors by undertraining Chemprop and decreasing its hidden dimension.

For comparison, we also report results with the oracle property predictor which is the ground truth property evaluator.

As shown in Figure 3 , on the DRD2 dataset, we are able to maintain strong performance despite using predictors that deviate significantly from the ground truth.

This implies that our framework can potentially be applied to other properties that are harder to predict.

On the QED dataset, our Table 2 : Ablation analysis of filtering at training and test time.

"Train" indicates a model whose training process uses data augmentation according to our framework.

"Test" indicates a model that uses the external filter at prediction time to discard candidate outputs which fail to pass the filter.

The evaluation for VSeq2Seq(no-filter) is conducted after 10 augmentation epochs, as the best validation set performance only decreases over the course of training.

method is less tolerant to inaccurate property prediction because the property constraint is much tighter -it requires the QED score of an output Y to be in the range [0.9, 1.0].

Importance of External Filtering Our full model (VSeq2Seq+) uses the external filter during both training and testing.

We further experiment with Vseq2seq(test), a version of our model trained without data augmentation but which uses the external filter to remove invalid outputs at test time.

As shown in Table 2 , VSeq2Seq(test) performs significantly worse than our full model trained under data augmentation.

Similarly, a model VSeq2Seq(train) trained with the data augmentation but without the prediction time filtering also performs much worse than the full model.

In addition, we run an augmentation-only version of the model without an external filter.

This model (referred to as VSeq2Seq(no-filter) in Table 2 ) augments the data in each epoch by simply using the first K distinct candidate translations for each precursor X in the training set, without using the external filter at all.

In addition, we provide this model with the 100K unlabeled precursors from the semi-supervised setting.

Nevertheless, we find that the performance of this model steadily declines from that of the bootstrapped starting point with each data augmentation epoch.

Thus the external filter is necessary to prevent poor targets from leading the model training astray.

In program synthesis, the source is a set of input-output specifications for the program, and the target is a program that passes all test cases.

Our method is suitable for this task because the target program is not unique.

Multiple programs may be consistent with the given input-output specifications.

The external filter is straightforward for this task: we simply check whether the generated output passes all test cases.

Note that at evaluation time, each instance contains extra held-out input-output test cases; the program must pass these in addition to the given test cases in order to be considered correct.

When we perform prediction time filtering, we do not use held-out test cases in our filter.

Top-1 Generalization MLE (Bunel et al., 2018) 71.91 MLE + RL + Beam Search (Bunel et al., 2018) 77.12

80.17 Table 3 : Model performance on Karel program synthesis task.

MLE+ is our augmented version of the MLE model (Bunel et al., 2018) .

Our task is based on the educational Karel programming language (Pattis, 1981) used for evaluation in Bunel et al. (2018) and Chen et al. (2019) .

Commands in the Karel language guide a robot's actions in a 2D grid, and may include for loops, while loops, and conditionals.

Figure 1 contains an example.

We follow the experiment setup of Bunel et al. (2018) .

Evaluation Metrics.

The evaluation metric is top-1 generalization.

This metric measures how often the model can generate a program that passes the input-output test cases on the test set.

At test time, we use our model to generate up to L candidate programs and select the first one to pass the input-output specifications (not including held-out test cases).

Models and Baselines.

Our main baseline is the MLE baseline from Bunel et al. (2018) .

This model consists of a CNN encoder for the input-output grids and a LSTM decoder along with a handcoded syntax checker.

It is trained to maximize the likelihood of the provided target program.

Our model is the augmentation of this MLE baseline by our iterative target augmentation framework.

As with molecular optimization, we generate up to K = 4 new targets per precursor during each augmentation step.

Additionally, we compare against the best model from Bunel et al. (2018) , which finetunes the same MLE architecture using an RL method with beam search to estimate gradients.

We use the same hyperparameters as the original MLE baseline; see Appendix A for details.

Table 3 shows the performance of our model in comparison to previous work.

Our model (MLE+) outperforms the base MLE model in Bunel et al. (2018) model by a wide margin.

Moreover, our model outperforms the best reinforcement learning model (RL + Beam Search) in Bunel et al. (2018) , which was trained to directly maximize the generalization metric.

This demonstrates the efficacy of our approach in the program synthesis domain.

Since our augmentation framework is complementary to architectural improvements, we hypothesize that other techniques, such as execution based synthesis (Chen et al., 2019) , can benefit from our approach as well.

In this work, we have presented an iterative target augmentation framework for generation tasks with multiple possible outputs.

Our approach is theoretically motivated, and we demonstrate strong empirical results on both the molecular optimization and program synthesis tasks, significantly outperforming baseline models on each task.

Moreover, we find that iterative target augmentation is complementary to architectural improvements, and that its effect can be quite robust to the quality of the external filter.

Finally, in principle our approach is applicable to other domains as well.

Our augmented models share the same hyperparameters as their baseline counterparts in all cases.

For the VSeq2Seq model we use batch size 64, embedding and hidden dimension 300, VAE latent dimension 30, and an LSTM with depth 1 (bidirectional in the encoder, unidirectional in the decoder).

For models using iterative target augmentation, n 1 is set to 5 and n 2 is set to 10, while for the baseline models we train for 20 epochs (corresponding to n 1 = 20, n 2 = 0).

The HierGNN model shares the same hyperparameters as in Jin et al. (2019a) .

For the training time and prediction time filtering parameters, we set K = 4, C = 200, and L = 10 for both the QED and DRD2 tasks.

For the Karel program synthesis task, we use the same hyperparameters as the MLE baseline model in Bunel et al. (2018) .

We use a beam size of 64 at test time, the same as the MLE baseline, but simply sample programs from the decoder distribution when running iterative target augmentation during training.

The baseline model is trained for 100 epochs, while for the model employing iterative target augmentation we train as normal for n 1 = 15 epochs followed by n 2 = 50 epochs of iterative target augmentation.

Due to the large size of the full training dataset, in each epoch of iterative augmentation we use 1 10 of the dataset, so in total we make 5 passes over the entire dataset.

For the training time and prediction time filtering parameters, we set K = 4, C = 50, and L = 10.

In Table 4 we provide the training, validation, and test set sizes for all of our tasks.

For each task we use the same splits as our baselines.

Training Set Validation Set Test Set  QED  88306  360  800  DRD2 34404  500  1000  Karel  1116854  2500  2500   Table 4 : Number of source-target pairs in training, validation, and test sets for each task.

In Figure 5 , we provide the validation set performance per iterative target augmentation epoch for our VSeq2Seq+ model on both the QED and DRD2 tasks.

The corresponding figure for the MLE+ model on the Karel task is in the main text in Figure 4 .

In our molecular optimization tasks, we experiment with the effect of modifying K, the number of new targets added per precursor during each training epoch.

In all other experiments we have used K = 4.

Since taking K = 0 corresponds to the base non-augmented model, it is unsurprising that performance may suffer when K is too small.

However, as shown in Table 5 , at least in molecular optimization there is relatively little change in performance for K much larger than 4.

We also experiment with a version of our method which continually grows the training dataset by keeping all augmented targets, instead of discarding new targets at the end of each epoch.

We chose the latter version for our main experiments due to its closer alignment to our EM motivation.

However, we demonstrate in Table 6 that performance gains from continually growing the dataset are small to insignificant in our molecular optimization tasks.

Table 6 : Performance of our proposed augmentation scheme, VSeq2Seq+, compared to an alternative version (VSeq2Seq+, keep-targets) which keeps all generated targets and continually grows the training dataset.

In Table 7 we provide the same ablation analysis that we provided in the main text for molecular optimization, demonstrating that both training time iterative target augmentation as well as prediction time filtering are beneficial to model performance.

However, we note that even MLE(train), our model without prediction time filtering, outperforms the best RL method from Bunel et al. (2018) .

Table 7 : Ablation analysis of filtering at training and test time.

"Train" indicates a model whose training process uses data augmentation according to our framework.

"Test" indicates a model that uses the external filter at prediction time to discard candidate outputs which fail to pass the filter.

Note that MLE and MLE(test) are based on an MLE checkpoint which underperforms the published result from Bunel et al. (2018) by 1 point, due to training for fewer epochs.

@highlight

We improve generative models by proposing a meta-algorithm that filters new training data from the model's outputs.