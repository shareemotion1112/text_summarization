Automatic Essay Scoring (AES) has been an active research area as it can greatly reduce the workload of teachers and prevents subjectivity bias .

Most recent AES solutions apply deep neural network (DNN)-based models with regression, where the neural neural-based encoder learns an essay representation that helps differentiate among the essays and the corresponding essay score is inferred by a regressor.

Such DNN approach usually requires a lot of expert-rated essays as training data in order to learn a good essay representation for accurate scoring.

However, such data is usually expensive and thus is sparse.

Inspired by the observation that human usually scores an essay by comparing it with some references, we propose a Siamese framework called Referee Network (RefNet) which allows the model to compare the quality of two essays by capturing the relative features that can differentiate the essay pair.

The proposed framework can be applied as an extension to regression models as it can capture additional relative features on top of internal information.

Moreover, it intrinsically augment the data by pairing thus is ideal for handling data sparsity.

Experiment shows that our framework can significantly improve the existing regression models and achieve acceptable performance even when the training data is greatly reduced.

Automatic Essay Scoring (AES) is the technique to automatically score an essay over some specific marking scale.

AES has been an eye-catching problem in machine learning due to its promising application in education.

It can free tremendous amount of repetitive labour, boosting the efficiency of educators.

Apart from automation, computers also prevail human beings in consistency, thus eliminate subjectivity and improve fairness in scoring.

Attempts in AES started as early as Project Essay Grade (PEG) (Page, 1967; 2003) , when the most prevalent methods relied on hand-crafted features engineered by human experts.

Recent advances in neural networks bring new possibilities to AES.

Several related works leveraged neural networks and achieved decent results (Dong et al., 2017; Taghipour & Ng, 2016; Tay et al., 2018; Liu et al., 2019) .

As is shown in Figure 1 , these approaches generally follow the 'representation + regression' scheme where a neural network reads in the text embeddings and generates a high level representation that will be fed to some regression model for a score.

However, such model requires a large amount of expert-rated essays for training.

In reality, collecting such dataset is expensive.

Therefore, data sparsity remains a knotty problem to be solved.

Inspired by the observation that human raters usually score an essay by comparing it to a set of references, we propose to leverage the pairwise comparisons for scoring instead of regression.

The goal of the model is shifted from predicting the score directly to comparing two essays, and the final score will be determined by comparing new essays with known samples.

In order to achieve this, we designed a Siamese network called Referee Network (RefNet) and corresponding scoring algorithms.

RefNet is a framework so that it can use various representation encoders as backbones.

What's more, though this model is designed to capture mutual features, it can also benefit from essay internal information via transfer learning.

Scoring essays by comparison has various benefits.

First, RefNet is incredibly strong in dealing with data sparsity problem.

Essays are paired with each other to form the training data for RefNet, which significantly augmented the data size.

Experiments show that our model achieve acceptable performance even when the training data is radically reduced, while regression models are subject to drastic performance degeneration.

Second, unlike end-to-end black-box models, our system scores an essay by comparing it with a set of labeled anchors, providing transparency to a certain degree during inference process.

Last but not least, with information in both internal and mutual perspective, RefNet can have better insight into the quality of essays.

Our contributions can be summarized as follows:

• We designed Referee Network (RefNet), a simple but effective model to compare two essays, and Majority Probability Voting Algorithm to infer the score from pairwise comparison results.

To the best of our knowledge, it is the first time a Siamese neutral network is used in AES.

• Our model intrinsically solves the problem of data sparsity.

It achieves acceptable performance even when the training data is greatly reduced, while regression models are impaired a lot.

Its efficacy in few-shot learning makes it an ideal solution for real applications where labelled data is usually limited.

• RefNet exploits a new realm of information, mutual relationship, by pairwise comparison.

With transfer learning, it also leverages internal features captured by regression.

Moreover, RefNet can be applied as an extension to various regression models and consistently improve the performance.

Existing AES solutions fall into two categories depending on how essays are represented: featureengineered models and end-to-end models.

In Yannakoudakis et al. (2011) , 9 handcrafted features are fed into a Support Vector Machine(SVM).

Those features range from simple ones like script length to elaborated ones like phrase structure rules and grammatical relation distance measure.

However, no matter how many features are used, it cannot develop expressive representations of the essays.

Besides, extracting handcrafted features often relies on other models, resulting in highly coupled systems that are slow and sophisticated.

Recent models based on neural networks can capture the features automatically.

Taghipour & Ng (2016) uses a self-trained look-up table for embedding and investigates a variety of neural networks, among which LSTM yields the best performance.

Several variations of neural network were also explored: Dong et al. (2017) applies CNN with attention pooling on sentence level and LSTM with attention pooling on essay level.

Tay et al. (2018) attached LSTM to a special network architecture called SkipFlow, which models the relationship between snapshots of hidden states of LSTM as it reads.

All of these methods achieved decent results, which proved the effectiveness of neural network in this problem.

However, several problems remind to be solved.

The first one is that no significant improvement over vanilla ones has been observed.

The second problem is the scarcity of labeled data casts doubt on whether elaboration of neural networks has reached its bottleneck.

A stark contrast to researchers' delicate designs in generating better representations is the rudimentary regression methods used to obtain the final prediction.

Of course, a regression layer satisfies the fundamental need of this task: mapping a representation to a scalar.

However, considering how complicated the relationship between essay contents and its score can be, the capability of such simple method is questionable.

Pairwise difference between essays, on the contrary, is intuitively more understandable and, according to Yannakoudakis et al. (2011) , outperforms mere regression in experiment.

Instead of directly mapping to the grade, learning rank preference is used to explicitly model the grade relationships between essays.

Yannakoudakis et al. used a special kind of SVM which outputs a real number with the -intensive loss function.

However, its only distinction from regression is including the degree of misclassification as part of the loss function so that the SVM can maximize the difference between closely-ranked data pairs.

It is more an amendment for regression than a new approach and cannot fully release the power the of comparison based methods.

In the last section, we argue that the potential of ranking preference methods in AES has not yet been fully exploited.

In this paper, we will try to exhaust the information in representations using a novel approach.

As depicted in Figure 1 , to score an essay in our system, Referee Network will compare it with known samples.

Then probability majority voting will be used to infer the score from pairwise rank preference.

Details will be elaborated in this section.

We use the same scheme as Liu et al. (2019) to obtain the embeddings at sentence level.

It is proven that Transformers, an attention mechanism that learns the contextual relations between words, can effectively make use of the textual information.

The model Bidirectional Encoder Representations from Transformers (BERT) which is built on this this approach achieved state-of-the-art results on various downstream tasks such as reading comprehension (Devlin et al., 2018) .

Therefore, we directly retrieve the word embeddings from pre-trained BERT model.

For a sentence with l words, BERT will output a set of low-dimensional representations s = {w 0 , w 1 , ..., w l , w l+1 }, where w i (1 ≤ i ≤ l) denotes the embedding for the i th word, w 0 is a special tag CLS for classification tasks and w l+1 is a SEP tag for separating sentences.

We use the average of the hidden states of the penultimate layer along the time axis to represent a sentence:

Here the superscript -2 means that the embedding is retrieved from the second last layer in BERT.

We use the penultimate layer instead of the final one because according to Liu et al. (2019) , the representations in the final layer fit too closely to the pre-training tasks including masked language modelling and next sentence prediction.

Embeddings in the penultimate layer are flexible enough for the model to fit to our AES task while at the same time semantically meaningful.

3.2.1 NETWORK ARCHITECTURE As is mentioned above, we will not built a model that directly infers the score via any kind of regression.

Instead, we built a model that compares essays, i.e takes two essays as input and outputs the rank preference.

In order to achieve this goal, we borrowed the idea from Siamese Network Koch et al. (2015) and designed the Referee Network (RefNet).

As shown in Figure 2 , RefNet is actually embarrassingly simple.

The pair of essays, which are both in the form of a set of sentences embeddings, will be encoded into essay representations through a backbone network.

The backbone network can be anything that outputs a vector from a matrix with certain dimension, in our case, the size of BERT outputs.

In this paper, we will try the following backbones:

• Average Embeddings.

For each input essay e = {s 1 , s 2 , s 3 , ..., s m }, it will just compute the average of all sentences r = (

• One layer of Simple Recurrent Neural Networks (RNN).

• One layer of Long Short-Term Memory (LSTM).

Due to the prevalence of RNN and LSTM, we shall skip their descriptions here.

Details of these two network architectures can be found in Sherstinsky (2018) .

Note that for a pair of essays [e 1 , e 2 ] ∈ R 2m×d , the RNN and LSTM backbones will generate two sets of hidden states {{h

m }}, and we will use only use the state at the final time step because its superiority over average state over time has been presented by Liu et al. (2019) .

Then the representations of the two essays will be concatenated along their first dimension r = [h

After that, one fully connected layer with leaky linear rectifier activation will be applied to extract higher level features.

Finally, another fully connected layer with two units together with softmax activation will give a rank preference, indicating with essay is better written:

In this paper, we denote RefNet as R(e i , e j ) in mathematical formulas.

To keep things simple, we simplify the output of R(e i , e j ) as a scalar in [0,1], indicating the probability that essay e i is better written than e j .

To train RefNet, we firstly pair each essay with those with different scores within the same prompt to form essay pairs.

Pairing is not conducted across prompts because essays from different prompts are not really comparable due to disparate writing requirements and scoring scales.

We do not pair the essays with the same score because one essay can hardly be exactly as good as another: among identically scores essays, one can further distinguish one may be better than another.

What's more, the inconsistent scoring schemes make concept of equal quality even more vague.

In the ASAP dataset, for instance, the scale can be as wide as 0-60 or as narrow as 0-3.

As a result, two essays with the same score in 0-3 scale may have drastically different scores in 0-60 scale.

Therefore, it is foreseeable that requiring RefNet to categorize identically rated essays will frustrate the model during training and impair the performance.

After pairing, RefNet is trained by minimizing the cross entropy between model output and true values.

We hope to make full advantage of the essay representations by exploiting both their internal and mutual information.

And what enables RefNet to use internal information is transfer learning.

As Figure  3 presents, we firstly train the backbone network with conventional end-to-end schemes where the model outputs the score directly by regression.

After being trained in this pseudo task, the parameters in the LSTM are transferred to RefNet and fine tuned for essay comparison, which is the real task.

In this way, what is learned in regression is more or less retained in the comparison model.

After comparing the test essay against known samples, we designed an algorithm called probability majority voting to infer its final score.

Suppose there are n i anchors for score notch N i .

Each anchor is paired with the testing input x and fed into our referee network for comparison.

By taking the average referee output over all the anchors, we get the probability of how likely that the essays at

Two aspects are considered to form the voting criteria.

First, according to the model, the test input is inferior than p i percent of anchors with score label N i .

Intuitively, if we choose the anchors with higher scores, the test essay will also be beaten.

Therefore, depending on how large p i is, it will more or less add to the likelihood that [N min , N i ) contains the correct answer.

In specific, votes of value p i will be given to all the scores in [N min , N i ).

Similarly, the test input is also superior than 1 − p i percent of anchors with score label N i .

Thus, a vote with value 1 − p i is added to

The second consideration is that we hope the p for the predicted score is close to 0.5, as one score does not appear correct if most of its anchors are better or worse than the input essay.

Therefore, we will penalize the distance from 0.5 by subtracting |p i − 0.5| from the notch's own votes.

Formally, total pairwise comparison result for some essay is {N 1 : p 1 , N 2 : p 2 , ..., N k : p k }, where k is the total number of score notches.

Without loss of generality, we assume that N 1 < N 2 < ... < N k .

The total votes for score notch N i is:

Our final prediction is naturally the score notch with the highest votes.

If more than one scores are the highest, we will take the average of all such scores and round to the nearest integer:

In this section, we will firstly present some background information regarding the dataset and performance metrics.

Then we will show the performance of our model in two different tasks and compare with other models.

After that, we will conduct ablation studies analyze the results.

We use Automated Student Assessment Prize(ASAP) dataset 1 , which is the standard dataset for developing and evaluating AES systems.

It is composed of 12976 essays in 8 prompts (see Table 1 for detailed statistics), where prompt 1,2,7,8 are persuasive, narrative or expository essays while the rest are Source Dependent Response questions.

In our experiments, we use the same 5-fold split as Taghipour & Ng (2016) did for fair comparisons.

In this data split scheme, 60% of data is used for training, 20% for validation and the rest for testing.

Similar to (Dong et al., 2017; Taghipour & Ng, 2016; Tay et al., 2018; Liu et al., 2019) , we use Quadratic Weighted Kappa (QWK), which is the official standard for ASAP dataset, as our evaluation metric.

The QWKs are computed for each prompt in its original scale respectively.

All the experiments are conducted over all 5 folds and the the average of each prompt's QWKs over all folds is calculated as the performance measure for the model.

We firstly trained our model on the whole dataset and observed that RefNet tends to overfit when all the training data is used.

It is not supervising as the total training data is amplified by approximately 300 times after pairing, which means that the model will see each essay around 300 times within each epoch.

As a result, the model can overfit before the first epoch is finished.

To solve this problem, training samples are dropped randomly before training.

To minimize the potentially negative impact, we set up different dropping rate for different essay pairs.

As Figure 4 suggests, when training on the whole dataset, RefNet can easily give accurate comparisons between two essays with contrasting scores.

However, distinguishing the pairs that have similar scores is much more challenging.

Therefore, the pairs with larger score difference are more likely to be dropped while those with smaller difference is more likely to be kept.

Different dropping rate are shown in Table 4 .2.

After the data adjustment, approximately only 15% of training set are kept.

The first three blocks of Table 3 compares the of regression and RefNet with the same backbone.

One can clearly see that RefNet constantly outperforms conventional regression approaches.

Notice that our method offers the greatest improvement in for RNN backbone, which is the weakest of the three.

That is because scoring by comparison is not very sensitive to the quality of representations.

As long as the representation makes sense, RefNet is able to boost the performance to a high level.

We also compare our model with the ensembled neural networks in Taghipour & Ng (2016) and SkipFlow LSTM networks proposed by Tay et al. (2018) .

The results show that we achieved the state-of-the-art average QWK score, and our edge is particularly large in prompt 8 which has much less data but more score notches than other prompts.

In this task, we created two mini-ASAP datasets by excerpting 25% and 10% of essays from the original training and validation dataset respectively while testing data remains the same.

Noticing that the essays under each prompt are distributed evenly in the original train test split, we gave special attention to this issue and our mini-ASAP dataset is balanced among different prompts.

And we will not use the drop training data as the excessive data problem no longer exists.

Table 4 we can see that with regression, the model will suffer from major performance degradation when the training data is reduced.

On the contrary, RefNet is much more robust to scarce data.

Even after 90% of data is dropped, RefNet can still offer high quality predictions.

Several ablation tests are conducted to study the effects of individual components.

In the first experiment, we remove the transfer learning and try to train RefNet from starch to test its efficacy.

In the second experiment, data adjustment is disabled and RefNet is trained on the whole dataset.

Finally, we hope to compare the performance of regression and pairwise methods under the same representations.

We fix the transferred backbone parameters and train the fully connected comparison layers in RefNet only and see how QWK varies.

All the ablation tests use LSTM as backbone.

Table 5 , the pairwise ranking approach achieved 0.754 on the fixed embeddings learnt from regression task, which is higher than pure regression performance 0.729 in Table  3 .

It shows that by taking mutual information into account, scoring by pairwise comparison can consistently improve the performance.

Second, internal information is also exploited by transfer learning.

With information from both internal and mutual perspective, RefNet can have better insight into the quality of essays.

Besides, considering the complexity of the task, it is hard to train RefNet from scratch.

Results in Table 5 show that the performance of RefNet without transfer learning degenerates by 2.48%.

Third, RefNet is naturally invulnerable to cross-prompt noise.

In reality, dataset of a single prompt is rarely large enough, so hybrid dataset such as ASAP dataset are commonly used.

However, the hybrid dataset consisting of essays of different scoring range, written by students at different levels or with different backgrounds may be not self-consistent.

For regression, the scores from different score range should be carefully aligned before training on the whole dataset because end-to-end method is sensitive to labels.

However, such alignment can hardly be achieved.

Current works just linearly rescale whatever original score range to [0, 1] , bringing noise to the system.

In contrast, RefNet sees only the relative relation between essays within the same prompt, avoiding possible cross-prompt noise.

RefNet shows even larger edge over regression in few-shot learning problems.

On one hand, RefNet is intrinsically suitable for few-shot learning problems as the pairing operation can amplify the training data by one or multiple levels of magnitude.

What's more, unlike some common data augmentation techniques such as random transforms, no noise is introduced in pairing.

On the other hand, the 'representation + regression' mechanism is highly data demanding.

First, though both approaches needs to somehow extract the features from texts, predicting by regression imposes higher requirement in expressing those features in a numeric and explicit form.

Second, since basic models such as LSTM and the features that can be learnt by those models have been well exploited, researchers may have to resort to deeper and more elaborated features to push the performance, resulting in more complicated models with more parameters.

Both factors makes endto-end model unable to be fully trained in small datasets.

In this paper we present Referee Network, a framework for automatic essay scoring using pairwise comparisons.

We demonstrate that RefNet is expert in solving data sparsity problems.

It can retain the performance at a high level even when the training data is significantly reduced, which outperforms regression models by a significant margin.

We also show that RefNet can improve conventional regression models by leveraging the additional mutual information between representations.

With only vanilla backbones, our model is able to obtain state-of-the-art results.

Even if the essay representations are fixed and have mediocre quality, our model can still boost up the scoring accuracy.

Furthermore, the capacity of RefNet can go far beyond this context as it is an extendable framework that can be used with any kind of representation encoders.

Besides the simple backbones we tried in this paper, one can by all means utilize more complicated and better performing models as backbones.

In this way, the performance of AES systems can always be pushed to a new record.

@highlight

Automatically score essays on sparse data by comparing new essays with known samples with Referee Network. 