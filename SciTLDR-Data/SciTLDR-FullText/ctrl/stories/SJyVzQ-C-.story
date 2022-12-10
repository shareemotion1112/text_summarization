Recurrent neural networks (RNNs) are important class of architectures among neural networks useful for language modeling and sequential prediction.

However, optimizing RNNs is known to be harder compared to feed-forward neural networks.

A number of techniques have been proposed in literature to address this problem.

In this paper we propose a simple technique called fraternal dropout that takes advantage of dropout to achieve this goal.

Specifically, we propose to train two identical copies of an RNN (that share parameters) with different dropout masks while minimizing the difference between their (pre-softmax) predictions.

In this way our regularization encourages the representations of RNNs to be invariant to dropout mask, thus being robust.

We show that our regularization term is upper bounded by the expectation-linear dropout objective which has been shown to address the gap due to the difference between the train and inference phases of dropout.

We evaluate our model and achieve state-of-the-art results in sequence modeling tasks on two benchmark datasets - Penn Treebank and Wikitext-2.

We also show that our approach leads to performance improvement by a significant margin in image captioning (Microsoft COCO) and semi-supervised (CIFAR-10) tasks.

Recurrent neural networks (RNNs) like long short-term memory (LSTM; BID4 ) networks and gated recurrent unit (GRU; BID0 ) are popular architectures for sequence modeling tasks like language generation, translation, speech synthesis, and machine comprehension.

However, they are harder to optimize compared to feed-forward networks due to challenges like variable length input sequences, repeated application of the same transition operator at each time step, and largely-dense embedding matrix that depends on the vocabulary size.

Due to these optimization challenges in RNNs, the application of batch normalization and its variants (layer normalization, recurrent batch normalization, recurrent normalization propagation) have not been as successful as their counterparts in feed-forward networks , although they do considerably provide performance gains.

Similarly, naive application of dropout BID23 has been shown to be ineffective in RNNs BID27 .

Therefore, regularization techniques for RNNs is an active area of research.

To address these challenges, BID27 proposed to apply dropout only to the nonrecurrent connections in multi-layer RNNs.

Variational dropout BID2 ) uses the same dropout mask throughout a sequence during training.

DropConnect BID25 applies the dropout operation on the weight matrices.

Zoneout BID7 ), in a similar spirit with dropout, randomly chooses to use the previous time step hidden state instead of using the current one.

Similarly as a substitute for batch normalization, layer normalization normalizes the hidden units within each sample to have zero mean and unit standard deviation.

Recurrent batch normalization applies batch normalization but with unshared mini-batch statistics for each time step BID1 .In this paper we propose a simple regularization based on dropout that we call fraternal dropout, where we minimize an equally weighted sum of prediction losses from two identical copies of the same LSTM with different dropout masks, and add as a regularization the 2 difference between the predictions (pre-softmax) of the two networks.

We analytically show that our regularization objective is equivalent to minimizing the variance in predictions from different i.i.d.

dropout masks; thus encouraging the predictions to be invariant to dropout masks.

We also discuss how our regularization is related to expectation linear dropout BID10 , Π-model BID8 and activity regularization BID16 , and empirically show that our method provides non-trivial gains over these related methods which we explain furthermore in our ablation study (Section 5).

Dropout is a powerful regularization for neural networks.

It is usually more effective on densely connected layers because they suffer more from overfitting compared with convolution layers where the parameters are shared.

For this reason dropout is an important regularization for RNNs.

However, dropout has a gap between its training and inference phase since the latter phase assumes linear activations to correct for the factor by which the expected value of each activation would be different BID10 .

In addition, the prediction of models with dropout generally vary with different dropout mask.

However, the desirable property in such cases would be to have final predictions be invariant to dropout masks.

As such, the idea behind fraternal dropout is to train a neural network model in a way that encourages the variance in predictions under different dropout masks to be as small as possible.

Specifically, consider we have an RNN model denoted by M(θ) that takes as input X, where θ denotes the model parameters.

DISPLAYFORM0 Note that a generalization of our approach would be to minimize the difference between the predictions of the two networks with different data/model augmentations.

However, in this paper we focus on using different dropout masks and experiment mainly with RNNs 2 .3 RELATED WORK 3.1 RELATION TO EXPECTATION LINEAR DROPOUT (ELD) BID10 analytically showed that the expected error (over samples) between a model's expected prediction over all dropout masks, and the prediction using the average mask, is upper bounded.

Based on this result, they propose to explicitly minimize the difference (we have adapted their regularization to our notations), DISPLAYFORM1 Specifically, this is achieved by feed-forwarding the input twice through the network, with and without dropout mask, and minimizing the main network loss (with dropout) along with the regularization term specified above (but without back-propagating the gradients through the network without dropout).

The goal of BID10 is to minimize the network loss along with the expected difference between the prediction from individual dropout mask and the prediction from the expected dropout mask.

We note that our regularization objective is upper bounded by the expectation-linear dropout regularization as shown below (proof in the appendix).

DISPLAYFORM2 This result shows that minimizing the ELD objective indirectly minimizes our regularization term.

Finally as indicated above, they apply the target loss only on the network with dropout.

In fact, in our own ablation studies (see Section 5) we find that back-propagating target loss through the network (without dropout) makes optimizing the model harder.

However, in our setting, simultaneously backpropagating target loss through both networks yields both performance gain as well as convergence gain.

We believe convergence is faster for our regularization because network weights are more likely to get target based updates from back-propagation in our case.

This is especially true for weight dropout BID25 since in this case dropped weights do not get updated in the training iteration.

BID8 propose Π-model with the goal of improving performance on classification tasks in the semi-supervised setting.

They propose a model similar to ours (considering the equivalent deep feed-forward version of our model) except they apply target loss only on one of the networks and use time-dependent weighting function ω(t) (while we use constant κ mT ).

The intuition in their case is to leverage unlabeled data by using them to minimize the difference in prediction between the two copies of the network with different dropout masks.

Further, they also test their model in the supervised setting but fail to explain the improvements they obtain by using this regularization.

We note that in our case we analytically show that minimizing our regularizer (also used in Π-model) is equivalent to minimizing the variance in the model predictions (Remark 1).

Furthermore, we also show the relation of our regularizer to expectation linear dropout (Proposition 1).

In Section 5, we study the effects of target based loss on both networks, which is not used in the Π-model.

We find that applying target loss on both the networks leads to significantly faster convergence.

Finally, we bring to attention that temporal embedding (another model proposed by BID8 , claimed to be a better version of Π-model for semi-supervised, learning) is intractable in natural language processing applications because storing averaged predictions over all of the time steps would be memory exhaustive (since predictions are usually huge -tens of thousands values).

On a final note, we argue that in the supervised case, using a time-dependent weighting function ω(t) instead of a constant value κ mT is not needed.

Since the ground truth labels are known, we have not observed the problem mentioned by BID8 , that the network gets stuck in a degenerate solution when ω(t) is too large in earlier epochs of training.

We note that it is much easier to search for an optimal constant value, which is true in our case, as opposed to tuning the time-dependent function.

Similarity to Π-model makes our method related to other semi-supervised works, mainly BID20 and BID21 .

Since semi-supervised learning is not a primary focus of this paper, we refer to BID8 for more details.

Another way to address the gap between the train and evaluation mode of dropout is to perform Monte Carlo sampling of masks and average the predictions during evaluation, and this has been used for feed-forward networks.

We find that this technique does not work well for RNNs.

The details of these experiments can be found in the appendix.

In the case of language modeling we test our model 3 on two benchmark datasets -Penn Tree-bank (PTB) dataset BID12 and WikiText-2 (WT2) dataset BID14 .

We use preprocessing as specified by BID17 (for PTB corpus) and Moses tokenizer BID6 (for the WT2 dataset).For both datasets we use the AWD-LSTM 3-layer architecture described in BID15 4 which we call the baseline model.

The number of parameters in the model used for PTB is 24 million as compared to 34 million in the case of WT2 because WT2 has a larger vocabulary size for which we use a larger embedding matrix.

Apart from those differences, the architectures are identical.

When we use fraternal dropout, we simply add our regularization on top of this baseline model.

Influenced by BID13 , our goal here is to make sure that fraternal dropout outperforms existing methods not simply because of extensive hyper-parameter grid search but rather due to its regularization effects.

Hence, in our experiments we leave a vast majority of hyper-parameters used in the baseline model BID13 unchanged i.e. embedding and hidden states sizes, gradient clipping value, weight decay and the values used for all dropout layers (dropout on the word vectors, the output between LSTM layers, the output of the final LSTM, and embedding dropout).

However, a few changes are necessary:• the coefficients for AR and TAR needed to be altered because fraternal dropout also affects RNNs activation (as explained in Subsection 5.3) -we did not run grid search to obtain the best values but simply deactivated AR and TAR regularizers; • since fraternal dropout needs twice as much memory, batch size is halved so the model needs approximately the same amount of memory and hence fits on the same GPU.The final change in hyper-parameters is to alter the non-monotone interval n used in nonmonotonically triggered averaged SGD (NT-ASGD) optimizer Polyak & Juditsky (1992); BID11 ; BID13 .

We run a grid search on n ∈ {5, 25, 40, 50, 60} and obtain very similar results for the largest values (40, 50 and 60) in the candidate set.

Hence, our model is trained longer using ordinary SGD optimizer as compared to the baseline model BID13 .We evaluate our model using the perplexity metric and compare the results that we obtain against the existing state-of-the-art results.

The results are reported in Table 1 .

Our approach achieves the state-of-the-art performance compared with existing benchmarks.

To confirm that the gains are robust to initialization, we run ten experiments for the baseline model with different seeds (without fine-tuning) for PTB dataset to compute confidence intervals.

The average best validation perplexity is 60.64 ± 0.15 with the minimum value equals 60.33.

The same for test perplexity is 58.32 ± 0.14 and 58.05, respectively.

Our score (59.8 validation and 58.0 test perplexity) beats ordinal dropout minimum values.

We also perform experiments using fraternal dropout with a grid search on all the hyper-parameters and find that it leads to further improvements in performance.

The details of this experiment can be found in section 5.5.

In the case of WikiText-2 language modeling task, we outperform the current state-of-the-art using the perplexity metric by a significant margin.

Due to the lack of computational power, we run a single training procedure for fraternal dropout on WT2 dataset because it is larger than PTB.

In this experiment, we use the best hyper-parameters found for PTB dataset (κ = 0.1, non-monotone interval n = 60 and halved batch size; the rest of the hyper-parameters are the same as described in BID13 for WT2).

The final results are presented in TAB2 .

We also apply fraternal dropout on an image captioning task.

We use the well-known show and tell model as a baseline 5 BID24 .

We emphasize that in the image captioning task, the image encoder and sentence decoder architectures are usually learned together.

Since we want to focus on the benefits of using fraternal dropout in RNNs we use frozen pretrained ResNet-101 (He TAB4 .We argue that in this task smaller κ values are optimal because the image captioning encoder is given all information in the beginning and hence the variance of consecutive predictions is smaller that in unconditioned natural language processing tasks.

Fraternal dropout may benefits here mainly due to averaging gradients for different mask and hence updating weights more frequently.

In this section, the goal is to study existing methods closely related to ours -expectation linear dropout BID10 , Π-model BID8 and activity regularization BID16 .

All of our experiments for ablation studies, which apply a single layer LSTM, use the same hyper-parameters and model architecture 6 as BID13 .

The relation with expectation-linear dropout BID10 has been discussed in Section 2.

Here we perform experiments to study the difference in performance when using the ELD regularization versus our regularization (FD).

In addition to ELD, we also study a modification (ELDM) of ELD which applies target loss to both copies of LSTMs in ELD similar to FD (notice in their case they only have dropout on one LSTM).

Finally we also evaluate a baseline model without any of these regularizations.

The learning dynamics curves are shown in FIG0 .

Our regularization performs better in terms of convergence compared with other methods.

In terms of generalization, we find that FD is similar to ELD, but baseline and ELDM are much worse.

Interestingly, looking at the train and validation curves together, ELDM seems to be suffering from optimization problems.

Since Π-model Laine & Aila (2016) is similar to our algorithm (even though it is designed for semi-supervised learning in feed-forward networks), we study the difference in performance with Π-model 7 both qualitatively and quantitatively to establish the advantage of our approach.

First, we run both single layer LSTM and 3-layer AWD-LSTM on PTB task to check how their model compares with ours in the case of language modeling.

The results are shown in FIG0 and 2.

We find that our model converges significantly faster than Π-model.

We believe this happens because we back-propagate the target loss through both networks (in contrast to Π-model) that leads to weights getting updated using target-based gradients more often.

6 We use a batch size of 64, truncated back-propagation with 35 time steps, a constant zero state is provided as the initial state with probability 0.01 (similar to BID13 ), SGD with learning rate 30 (no momentum) which is multiplied by 0.1 whenever validation performance does not improve ever during 20 epochs, weight dropout on the hidden to hidden matrix 0.5, dropout every word in a mini-batch with probability 0.1, embedding dropout 0.65, output dropout 0.4 (final value of LSTM), gradient clipping of 0.25, weight decay 1.2 × 10 −6 , input embedding size of 655, the input/output size of LSTM is the same as embedding size (655) and the embedding weights are tied BID5 BID19 .

7 We use a constant function ω(t) = κ mT as a coefficient for Π-model (similar to our regularization term).

Hence, the focus of our experiment is to evaluate the difference in performance when target loss is backpropagated through one of the networks (Π-model) vs. both (ours).

Additionally, we find that tuning a function instead of using a constant coefficient is infeasible.

Even though we designed our algorithm specifically to address problems in RNNs, to have a fair comparison, we compare with Π-model on a semi-supervised task which is their goal.

Specifically, we use the CIFAR-10 dataset that consists of 32 × 32 images from 10 classes.

Following the usual splits used in semi-supervised learning literature, we use 4 thousand labeled and 41 thousand unlabeled samples for training, 5 thousand labeled samples for validation and 10 thousand labeled samples for test set.

We use the original ResNet-56 BID3 architecture.

We run grid search on κ ∈ {0.05, 0.1, 0.15, 0.2}, dropout rates in {0.05, 0.1, 0.15, 0.2} and leave the rest of the hyperparameters unchanged.

We additionally check importance of using unlabeled data.

The results are reported in Table 4 .

We find that our algorithm performs at par with Π-model.

When unlabeled data is not used, fraternal dropout provides slightly better results as compared to traditional dropout.

The authors of BID16 study the importance of activity regularization (AR) 8 and temporal activity regularization (TAR) in LSTMs given as, DISPLAYFORM0 where h t ∈ R d is the LSTM's output activation at time step t (hence depends on both current input z t and the model parameters θ).

Notice that AR and TAR regularizations are applied on the output of 8 We used m · h t 2 2 , where m is the dropout mask, in our actual experiments with AR because it was implemented as such in the original paper's Github repository BID15 Table 4 : Ablation study: Accuracy on altered (semi-supervised) CIFAR-10 dataset for ResNet-56 based models.

We find that our algorithm performs at par with Π-model.

When unlabeled data is not used traditional dropout hurts performance while fraternal dropout provides slightly better results.

It means that our methods may be beneficial when we lack data and have to use additional regularizing methods.

Figure 4: Ablation study: Train (left) and validation (right) perplexity on PTB word level modeling with single layer LSTM (10M parameters).

These curves study the learning dynamics of the baseline model, temporal activity regularization (TAR), prediction regularization (PR), activity regularization (AR) and fraternal dropout (FD, our algorithm).

We find that FD both converges faster and generalizes better than the regularizers in comparison.the LSTM, while our regularization is applied on the pre-softmax output pt i ; θ) 2 2 to further rule-out any gains only from 2 regularization.

Based on this grid search, we pick the best model on the validation set for all the regularizations, and additionally report a baseline model without any of these four mentioned regularizations.

The learning dynamics is shown in Figure 4 .

Our regularization performs better both in terms of convergence and generalization compared with other methods.

Average hidden state activation is reduced when any of the regularizer described is applied (see FIG2 .

We confirm that models trained with fraternal dropout benefit from the NT-ASGD fine-tuning step (as also used in BID15 ).

However, this is a very time-consuming practice and since different hyper-parameters may be used in this additional part of the learning procedure, the probability of obtaining better results due to the extensive grid search is higher.

Hence, in our experiments we use the same fine-tuning procedure as implemented in the official repository (even fraternal dropout was not used).

We present the importance of fine-tuning in Table 5 .

Table 5 : Ablation study: Importance of fine-tuning for AWD-LSTM 3-layer model.

Perplexity for the Penn Treebank and WikiText-2 language modeling tasks.

Hyper-parameter Possible values batch size [10, 20, 30 , 40] non-monotone interval [5, 10, 20, 40, 60 , 100] κ -FD or ELD strength U (0, 0.3) weight decay U (0.6 × 10 −6 , 2.4 × 10 −6 )

We perform extensive grid search for the baseline model from Subsection 4.1 (an AWD-LSTM 3-layer architecture) trained with either fraternal dropout or expectation linear dropout regularizations, to further contrast the performance of these two methods.

The experiments are run without fine-tuning on the PTB dataset.

In each run, all five dropout rates are randomly altered (they are set to their original value, as in BID15 , multiplied by a value drawn from the uniform distribution on the interval [0.5, 1.5]) and the rest of the hyper-parameters are drawn as shown in TAB7 .

As in Subsection 4.1, AR and TAR regularizers are deactivated.

Together we run more than 400 experiments.

The results are presented in TAB8 .

Both FD and ELD perform better than the baseline model that instead uses AR and TAR regularizers.

Hence, we confirm our previous finding (see Subsection 5.3) that both FD and ELD are better.

However, as found previously for smaller model in Subsection 5.1, the convergence of FD is faster than that of ELD.

Additionally, fraternal dropout is more robust to different hyper-parameters choice (more runs performing better than the baseline and better average for top performing runs).

In this paper we propose a simple regularization method for RNNs called fraternal dropout that acts as a regularization by reducing the variance in model predictions across different dropout masks.

We show that our model achieves state-of-the-art results on benchmark language modeling tasks along with faster convergence.

We also analytically study the relationship between our regularization and expectation linear dropout BID10 .

We perform a number of ablation studies to evaluate our model from different aspects and carefully compare it with related methods both qualitatively and quantitatively.

A well known way to address the gap between the train and evaluation mode of dropout is to perform Monte Carlo sampling of masks and average the predictions during evaluation (MC-eval), and this has been used for feed-forward networks.

Since fraternal dropout addresses the same problem, we would like to clarify that it is not straight-forward and feasible to apply MC-eval for RNNs.

In feed-forward networks, we average the output prediction scores from different masks.

However, in the case RNNs (for next step predictions), there is more than one way to perform such evaluation, but each one is problematic.

They are as follows:

Consider that we first make the prediction at time step 1 using different masks by averaging the prediction score.

Then we use this output to feed as input to the time step 2, then use different masks at time step 2 to generate the output at time step 2, and so on.

But in order to do so, because of the way RNNs work, we also need to feed the previous time hidden state to time step 2.

One way would be to average the hidden states over different masks at time step 1.

But the hidden space can in general be highly nonlinear, and it is not clear if averaging in this space is a good strategy.

This approach is not justified.

Besides, this strategy as a whole is extremely time consuming because we would need to sequentially make predictions with multiple masks at each time step.

Let's consider that we use a different mask each time we want to generate a sequence, and then we average the prediction scores, and compute the argmax (at each time step) to get the actual generated sequence.

In this case, notice it is not guaranteed that the predicted word at time step t due to averaging the predictions would lead to the next word (generated by the same process) if we were to feed the time step t output as input to the time step t + 1.

For example, with different dropout masks, if the probability of 1 st time step outputs are: I 40%), he (30%), she (30%), and the probability of the 2nd time step outputs are: am (30%), is (60%), was (10%).

Then the averaged prediction score followed by argmax will result in the prediction "I is", but this would be incorrect.

A similar concern applies for output predictions varying in temporal length.

Hence, this approach can not be used to generate a sequence (it has to be done by by sampling a mask and generating a single sequence).

However, this approach may be used to estimate the probability assigned by the model to a given sequence.

Nonetheless, we run experiments on the PTB dataset using MC-eval (the results are summarized in TAB10 ).

We start with a simple comparison that compares fraternal dropout with the averaged mask and the AWD-LSTM 3-layer baseline with a single fixed mask that we call MC1.

The MC1 model performs much worse than fraternal dropout.

Hence, it would be hard to use MC1 model in practice because a single sample is inaccurate.

We also check MC-eval for a larger number of models (MC50) (50 models were used since we were not able to fit more models simultaneously on a single GPU).

The final results for MC50 are worse than the baseline which uses the averaged mask.

For comparison, we also evaluate MC10.

Note that no fine-tuning is used for the above experiments.

The fraternal dropout method is general and may be applied in feed-forward architectures (as shown in Subsection 5.2 for CIFAR-10 semisupervised example).

However, we believe that it is more powerful in the case of RNNs because:1.

Variance in prediction accumulates among time steps in RNNs and since we share parameters for all time steps, one may use the same κ value at each step.

In feed-forward networks the layers usually do not share parameters and hence one may want to use different κ values for different layers (which may be hard to tune).

The simple way to alleviate this problem is to apply the regularization term on the pre-softmax predictions only (as shown in the paper) or use the same κ value for all layers.

However, we believe that it may limit possible gains.2.

The best performing RNN architectures (state-of-the-art) usually use some kind of dropout (embedding dropout, word dropout, weight dropout etc.), very often with high dropout rates (even larger than 50% for input word embedding in NLP tasks).

However, this is not true for feed-forward networks.

For instance, ResNet architectures very often do not use dropout at all (probably because batch normalization is often better to use).

It can be seen in the paper (Subsection 5.2, semisupervised CIFAR-10 task) that when unlabeled data is not used the regular dropout hurts performance and using fraternal dropout seems to improve just a little.3.

On a final note, the Monte Carlo sampling (a well known method that adresses the gap betweem the train and evaluation mode of dropout) can not be easily applied for RNNs and fraternal dropout may be seen as an alternative.

To conclude, we believe that when the use of dropout benefits in a given architecture, applying fraternal dropout should improve performance even more.

As mentioned before, in image recognition tasks, one may experiment with something what we would temporarily dub fraternal augmentation (even though dropout is not used, one can use random data augmentation such as random crop or random flip).

Hence, one may force a given neural network to have the same predictions for different augmentations.

<|TLDR|>

@highlight

We propose to train two identical copies of an recurrent neural network (that share parameters) with different dropout masks while minimizing the difference between their (pre-softmax) predictions.

@highlight

Presents Fraternal dropout as an improvement over Expectation-linear dropout in terms of convergence, and demonstrates the utility of Fraternal dropout on a number of tasks and datasets.