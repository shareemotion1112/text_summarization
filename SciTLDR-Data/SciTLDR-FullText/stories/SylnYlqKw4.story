It can be challenging to train multi-task neural networks that outperform or even match their single-task counterparts.

To help address this, we propose using knowledge distillation where single-task models teach a multi-task model.

We enhance this training with teacher annealing, a novel method that gradually transitions the model from distillation to supervised learning, helping the multi-task model surpass its single-task teachers.

We evaluate our approach by multi-task fine-tuning BERT on the GLUE benchmark.

Our method consistently improves over standard single-task and multi-task training.

Building a single model that jointly learns to perform many tasks effectively has been a longstanding challenge in Natural Language Processing (NLP).

However, applying multi-task NLP remains difficult for many applications, with multitask models often performing worse than their single-task counterparts BID30 BID1 BID25 .

Motivated by these results, we propose a way of applying knowledge distillation BID3 BID0 BID14 so that single-task models effectively teach a multi-task model.

Knowledge distillation transfers knowledge from a "teacher" model to a "student" model by training the student to imitate the teacher's outputs.

In "born-again networks" BID10 , the teacher and student have the same neural architecture and model size, but surprisingly the student is able to surpass the teacher's accuracy.

Intuitively, distillation is effective because the teacher's output distribution over classes provides more training signal than a one-hot label; BID14 suggest that teacher outputs contain "dark knowledge" capturing additional information about training examples.

Our work extends born-again networks to the multi-task setting.

We compare Single→Multi 1 born-again distillation with several other variants (Single→Single and Multi→Multi), and also explore performing multiple rounds of distillation (Single→Multi→Single→Multi) .

Furthermore, we propose a simple teacher annealing method that helps the student model outperform its teachers.

Teacher annealing gradually transitions the student from learning from the teacher to learning from the gold labels.

This method ensures the student gets a rich training signal early in training but is not limited to only imitating the teacher.

Our experiments build upon recent success in self-supervised pre-training BID7 BID28 and multi-task fine-tune BERT BID8 to perform the tasks from the GLUE natural language understanding benchmark BID41 .

Our training method, which we call Born-Again Multi-tasking (BAM) 2 , consistently outperforms standard single-task and multi-task training.

Further analysis shows the multi-task models benefit from both better regu- 1 We use Single→Multi to indicate distilling single-task "teacher" models into a multi-task "student" model.

2 Code is available at https://github.com/ google-research/google-research/tree/ master/bam larization and transfer between related tasks.

Multi-task learning for neural networks in general BID4 and within NLP specifically BID6 BID24 has been widely studied.

Much of the recent work for NLP has centered on neural architecture design: e.g., ensuring only beneficial information is shared across tasks BID21 or arranging tasks in linguistically-motivated hierarchies BID37 BID13 BID34 .

These contributions are orthogonal to ours because we instead focus on the multi-task training algorithm.

Distilling large models into small models BID19 BID26 or ensembles of models into single models BID20 BID22 has been shown to improve results for many NLP tasks.

There has also been some work on using knowledge distillation to aide in multi-task learning.

In reinforcement learning, knowledge distillation has been used to regularize multi-task agents BID27 BID39 .

In NLP, BID38 distill singlelanguage-pair machine translation systems into a many-language system.

However, they focus on multilingual rather than multi-task learning, use a more complex training procedure, and only experiment with Single→Multi distillation.

Concurrently with our work, several other recent works also explore fine-tuning BERT using multiple tasks BID29 BID23 BID18 .

However, they use only standard transfer or multi-task learning, instead focusing on finding beneficial task pairs or designing improved task-specific components on top of BERT.

Model.

All of our models are built on top of BERT BID8 .

This model passes byte-pairtokenized BID35 input sentences through a Transformer network BID40 , producing a contextualized representation for each token.

The vector corresponding to the first input token 3 c is passed into a task-specific classifier.

For classification tasks, we use a standard softmax layer: softmax(W c).

For regression tasks, we normalize the labels so they are between 0 and 1 and then use a size-1 NN layer with a sigmoid activation: sigmoid(w T c).

In our multi-task models, all of the model parameters are shared across tasks except for these classifiers on top of BERT, which means less than 0.01% of the parameters are task-specific.

Following BERT, the token embeddings and Transformer are initialized with weights from a self-supervised pre-training phase.

Training.

Single-task training is performed as in BID8 .

For multi-task training, examples of different tasks are shuffled together, even within minibatches.

The summed loss across all tasks is minimized.

We use DISPLAYFORM0 } to denote the training set for a task τ and f τ (x, θ) to denote the outputs for task τ produced by a neural network with parameters θ on the input x (for classification tasks this is a distribution over classes).

Standard supervised learning trains θ to minimize the loss on the training set: DISPLAYFORM1 where for classification tasks is usually crossentropy.

Knowledge distillation trains the model to instead match the predictions of a teacher model with parameters θ : DISPLAYFORM2 Note that our distilled networks are "born-again" in that the student has the same model architecture as the teacher, i.e., all of our models have the same prediction function f τ for each task.

For regression tasks, we train the student to minimize the L2 distance between its prediction and the teacher's instead of using cross-entropy loss.

Intuitively, knowledge distillation improves training because the full distribution over labels provided by the teacher provides a richer training signal than a one-hot label.

See BID10 for a more thorough discussion.

Multi-Task Distillation.

Given a set of tasks T , we train a single-task model with parameters θ τ on each task τ .

For most experiments, we use the single-task models to teach a multi-task model with parameters θ: DISPLAYFORM3 However, we experiment with other distillation strategies as well.

Teacher Annealing.

In knowledge distillation, the student is trained to imitate the teacher.

This raises the concern that the student may be limited by the teacher's performance and not be able to substantially outperform the teacher.

To address this, we propose teacher annealing, which mixes the teacher prediction with the gold label during training.

Specifically, the term in the summation becomes DISPLAYFORM4 where λ is linearly increased from 0 to 1 throughout training.

Early in training, the model is mostly distilling to get as useful of a training signal as possible.

Towards the end of training, the model is mostly relying on the gold-standard labels so it can learn to surpass its teachers.

Data.

We use the General Language Understanding Evaluation (GLUE) benchmark BID41 , which consists of 9 natural language understanding tasks on English data.

Tasks cover textual entailment (RTE and MNLI) question-answer entailment (QNLI), paraphrase (MRPC), question paraphrase (QQP), textual similarity (STS), sentiment (SST-2), linguistic acceptability (CoLA), and Winograd Schema (WNLI).Training Details.

Rather than simply shuffling the datasets for our multi-task models, we follow the task sampling procedure from , where the probability of training on an example for a particular task τ is proportional to |D τ | 0.75 .

This ensures that tasks with very large datasets don't overly dominate the training.

We also use the layerwise-learning-rate trick from BID16 .

If layer 0 is the NN layer closest to the output, the learning rate for a particular layer d is set to BASE LR · α d (i.e., layers closest to the input get lower learning rates).

The intuition is that pre-trained layers closer to the input learn more general features, so they shouldn't be altered much during training.

Hyperparameters.

For single-task models, we use the same hyperparameters as in the original BERT experiments except we pick a layerwiselearning-rate decay α of 1.0 or 0.9 on the dev set for each task.

For multi-task models, we train the model for longer (6 epochs instead of 3) and with a larger batch size (128 instead of 32), using α = 0.9 and a learning rate of 1e-4.

All models use the BERT-Large pre-trained weights.

Reporting Results.

Dev set results report the average score (Spearman correlation for STS, Matthews correlation for CoLA, and accuracy for the other tasks) on all GLUE tasks except WNLI, for which methods can't outperform a majority baseline.

Results show the median score of at least 20 trials with different random seeds.

We find using a large number of trials is essential because results can vary significantly for different runs.

For example, standard deviations in score are over ±1 for CoLA, RTE, and MRPC for multi-task models.

Single-task standard deviations are even larger.

Main Results.

We compare models trained with single-task learning, multi-task learning, and several varieties of distillation in TAB1 .

While standard multi-task training improves over single-task training for RTE (likely because it is closely related to MNLI), there is no improvement on the other tasks.

In contrast, Single→Multi knowledge distillation improves or matches the performance of the other methods on all tasks except STS, the only regression task in GLUE.

We believe distillation does not work well for regression tasks because there is no distribution over classes passed on by the teacher to aid learning.

The gain for Single→Multi over Multi is larger than the gain for Single→Single over Single, suggesting that distillation works particularly well in combination with multi-task learning.

Interestingly, Single→Multi works substantially better than Multi→Multi distillation.

We speculate it may help that the student is exposed to a diverse set of teachers in the same way ensembles benefit from a diverse set of models, but future work is required to fully understand this phenomenon.

In addition to the models reported in the table, we also trained Single→Multi→Single→Multi models.

However, the difference with Single→Multi was not statistically significant, suggesting there is little value in multiple rounds of distillation.

Avg.

CoLA Williams et al. (2018) g constructed from SQuAD BID31 h BID11 BERT-Base BID8 78.5 BERT-Large BID8 80.5 BERT on STILTs BID29 82.0 MT-DNN BID23 82 Overall, a key benefit of our method is robustness: while standard multi-task learning produces mixed results, Single→Multi distillation consistently outperforms standard single-task and multitask training.

We also note that in some trials single-task training resulted in models that score quite poorly (e.g., less than 91 for QQP or less than 70 for MRPC), while the multi-task models have more dependable performance.

DISPLAYFORM0 Test Set Results.

We compare against recent work by submitting to the GLUE leaderboard.

We use Single→Multi distillation.

Following the procedure used by BERT, we train multiple models and submit the one with the highest average dev set score to the test set.

BERT trained 10 models for each task (80 total); we trained 20 multi-task models.

Results are shown in TAB3 .Our work outperforms or matches existing published results that do not rely on ensembling.

However, due to the variance between trials dis-cussed under "Reporting Results," we think these test set numbers should be taken with a grain of salt, as they only show the performance of individual training runs (which is further complicated by the use of tricks such as dev set model selection).

We believe significance testing over multiple trials would be needed to have a definitive comparison.

Single-Task Fine-Tuning.

A crucial difference distinguishing our work from the STILTs, Snorkel MeTaL, and MT-DNN KD methods in TAB3 is that we do not single-task fine-tune our model.

That is, we do not further train the model on individual tasks after the multi-task training finishes.

While single-task fine-tuning improves results, we think to some extent it defeats the purpose of multi-task learning: the result of training is one model for each task instead of a model that can perform all of the tasks.

Compared to having many single-task models, a multi-task model is simpler to deploy, faster to run, and arguably more scientifically interesting from the perspective of building general language-processing systems.

We evaluate the benefits of single-task finetuning in TAB5 .

Single-task fine-tuning initializes models with multi-task-learned weights and then performs single-task training.

Hyperparameters are the same as for our single-task models except we use a smaller learning rate of 1e-5.

While single-task fine-tuning unsurprisingly improves results, the gain on top of Single→Multi distillation is small, reinforcing the claim that distillation provides many of the benefits of singletask training while producing a single unified model instead of many task-specific models.

Ablation Study.

We show the importance of teacher annealing and the other training tricks in improve scores.

Using pure distillation without teacher annealing (i.e., fixing λ = 0) performs no better than standard multi-task learning, demonstrating the importance of the proposed teacher annealing method.

Comparing Combinations of Tasks.

Training on a large number of tasks is known to help regularize multi-task models BID32 .

A related benefit of multi-task learning is the transfer of learned "knowledge" between closely related tasks.

We investigate these two benefits by comparing several models on the RTE task, including one trained with a very closely related task (MNLI, a much large textual entailment dataset) and one trained with fairly unrelated tasks (QQP, CoLA, and SST).

We use Single→Multi distillation (Single→Single in the case of the RTE-only model).

Results are shown in TAB8 .

We find both sets of auxiliary tasks improve RTE performance, suggesting that both benefits are playing a role in improving multi-task models.

Interestingly, RTE + MNLI alone slightly outperforms the model performing all tasks, perhaps because training on MNLI, which has a very large dataset, is already enough to sufficiently regularize the model.

We have shown that Single→Multi distillation combined with teacher annealing produces results consistently better than standard single-task or multi-task training.

Achieving robust multi-task gains across many tasks has remained elusive in previous research, so we hope our work will make multi-task learning more broadly useful within NLP.

However, with the exception of closely related tasks with small datasets (e.g., MNLI helping RTE), the overall size of the gains from our multi-task method are small compared to the gains provided by transfer learning from self-supervised tasks (i.e., BERT).

It remains to be fully understood to what extent "self-supervised pre-training is all you need" and where transfer/multi-task learning from supervised tasks can provide the most value.

@highlight

distilling single-task models into a multi-task model improves natural language understanding performance