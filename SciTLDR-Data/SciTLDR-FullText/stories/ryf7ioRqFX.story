Recurrent neural networks are known for their notorious exploding and vanishing gradient problem (EVGP).

This problem becomes more evident in tasks where the information needed to correctly solve them exist over long time scales, because EVGP prevents important gradient components from being back-propagated adequately over a large number of steps.

We introduce a simple stochastic algorithm (\textit{h}-detach) that is specific to LSTM optimization and targeted towards addressing this problem.

Specifically, we show that when the LSTM weights are large, the gradient components through the linear path (cell state) in the LSTM computational graph get suppressed.

Based on the hypothesis that these components carry information about long term dependencies (which we show empirically), their suppression can prevent LSTMs from capturing them.

Our algorithm\footnote{Our code is available at https://github.com/bhargav104/h-detach.} prevents gradients flowing through this path from getting suppressed, thus allowing the LSTM to capture such dependencies better.

We show significant improvements over vanilla LSTM gradient based training in terms of convergence speed, robustness to seed and learning rate, and generalization using our modification of LSTM gradient on various benchmark datasets.

Recurrent Neural Networks (RNNs) BID25 ; BID4 ) are a class of neural network architectures used for modeling sequential data.

Compared to feed-forward networks, the loss landscape of recurrent neural networks are much harder to optimize.

Among others, this difficulty may be attributed to the exploding and vanishing gradient problem BID8 BID2 BID24 which is more severe for recurrent networks and arises due to the highly ill-conditioned nature of their loss surface.

This problem becomes more evident in tasks where training data has dependencies that exist over long time scales.

Due to the aforementioned optimization difficulty, variants of RNN architectures have been proposed that aim at addressing these problems.

The most popular among such architectures that are used in a wide number of applications include long short term memory (LSTM, BID9 ) and gated recurrent unit (GRU, Chung et al. (2014) ) networks, which is a variant of LSTM with forget gates BID5 .

These architectures mitigate such difficulties by introducing a linear temporal path that allows gradients to flow more freely across time steps.

BID0 on the other hand try to address this problem by parameterizing a recurrent neural network to have unitary transition matrices based on the idea that unitary matrices have unit singular values which prevents gradients from exploding/vanishing.

Among the aforementioned RNN architectures, LSTMs are arguably most widely used (for instance they have more representational power compared with GRUs BID31 ) and it remains a hard problem to optimize them on tasks that involve long term dependencies.

Examples of such tasks are copying problem BID2 BID24 , and sequential MNIST (Le Figure 1 : The computational graph of a typical LSTM.

Here we have omitted the inputs x i for convenience.

The top horizontal path through the cell state units c t s is the linear temporal path which allows gradients to flow more freely over long durations.

The dotted blue crosses along the computational paths denote the stochastic process of blocking the flow of gradients though the h t states (see Eq 2) during the back-propagation phase of LSTM.

We call this approach h-detach.

et al., 2015) , which are designed in such a way that the only way to produce the correct output is for the model to retain information over long time scales.

The goal of this paper is to introduce a simple trick that is specific to LSTM optimization and improves its training on tasks that involve long term dependencies.

To achieve this goal, we write out the full back-propagation gradient equation for LSTM parameters and split the composition of this gradient into its components resulting from different paths in the unrolled network.

We then show that when LSTM weights are large in magnitude, the gradients through the linear temporal path (cell state) get suppressed (recall that this path was designed to allow smooth gradient flow over many time steps).

We show empirical evidence that this path carries information about long term dependencies (see section 3.5) and hence gradients from this path getting suppressed is problematic for such tasks.

To fix this problem, we introduce a simple stochastic algorithm that in expectation scales the individual gradient components, which prevents the gradients through the linear temporal path from being suppressed.

In essence, the algorithm stochastically prevents gradient from flowing through the h-state of the LSTM (see figure 1) , hence we call it h-detach.

Using this method, we show improvements in convergence/generalization over vanilla LSTM optimization on the copying task, transfer copying task, sequential and permuted MNIST, and image captioning.

We begin by reviewing the LSTM roll-out equations.

We then derive the LSTM back-propagation equations and by studying its decomposition, identify the aforementioned problem.

Based on this analysis we propose a simple stochastic algorithm to fix this problem.

LSTM is a variant of traditional RNNs that was designed with the goal of improving the flow of gradients over many time steps.

The roll-out equations of an LSTM are as follows, DISPLAYFORM0 where denotes point-wise product and the gates f t , i t , o t and g t are defined as, DISPLAYFORM1 Here c t and h t are the cell state and hidden state respectively.

Usually a transformation φ(h T ) is used as the output at time step t (Eg.

next word prediction in language model) based on which we can compute the loss t := (φ(h t )) for that time step.

An important feature of the LSTM architecture is the linear recursive relation between the cell states c t as shown in Eq. 1.

This linear path allows gradients to flow easily over long time scales.

This however is one of the components in the full composition of the LSTM gradient.

As we will show next, the remaining components that are a result of the other paths in the LSTM computational graph are polynomial in the weight matrices W gh , W f h , W ih , W oh whose order grows with the number of time steps.

These terms cause an imbalance in the order of magnitude of gradients from different paths, thereby suppressing gradients from linear paths of LSTM computational graph in cases where the weight matrices are large.

In this section we derive the back-propagation equations for LSTM network and by studying its composition, we identify a problem in this composition.

The back-propagation equation of an LSTM can be written in the following form.

Theorem 1 Fix w to be an element of the matrix DISPLAYFORM0 Then z t = (A t + B t )z t−1 .

In other words, DISPLAYFORM1 where all the symbols used to define A t and B t are defined in notation 1 in appendix.

To avoid unnecessary details, we use a compressed definitions of A t and B t in the above statement and write the detailed definitions of the symbols that constitute them in notation 1 in appendix.

Nonetheless, we now provide some intuitive properties of the matrices A t and B t .The matrix A t contains components of parameter's full gradient that arise due to the cell state (linear temporal path) described in Eq.(1) (top most horizontal path in figure 1 ).

Thus based on the above analysis, we identify the following problem with the LSTM gradient: when the LSTM weights are large, the gradient component through the cell state paths (A t ) get suppressed compared to the gradient components through the other paths (B t ) due to an imbalance in gradient component magnitudes.

We recall that the linear recursion in the cell state path was introduced in the LSTM architecture BID9 as an important feature to allow gradients to flow smoothly through time.

As we show in our ablation studies (section 3.5), this path carries information about long term dependencies in the data.

Hence it is problematic if the gradient components from this path get suppressed.

We now propose a simple fix to the above problem.

Our goal is to manipulate the gradient components such that the components through the cell state path (A t ) do not get suppressed when the components through the remaining paths (B t ) are very large (described in the section 2.2).

Thus it would be helpful to multiply B t by a positive number less than 1 to dampen its magnitude.

In Algorithm 1 we propose a simple trick that achieves this goal.

A diagrammatic form of algorithm 1 is shown in Figure 1 .

In simple words, our algorithm essentially blocks gradients from flowing through each of the h t states independently with a probability 1 − p, where p ∈ [0, 1] is a tunable hyper-parameter.

Note the subtle detail in Algorithm 1 (line 9) that the loss t at any time step t is a function of h t which is not detached.

Algorithm 1 Forward Pass of h-detach Algorithm DISPLAYFORM0 t ← loss(φ(h t ))10: DISPLAYFORM1 We now show that the gradient of the loss function resulting from the LSTM forward pass shown in algorithm 1 has the property that the gradient components arising from B t get dampened.

DISPLAYFORM2 T andz t be the analogue of z t when applying h-detach with probability 1 − p during back-propagation.

Then, DISPLAYFORM3 where ξ t , ξ t−1 , . . . , ξ 2 are i.i.d.

Bernoulli random variables with probability p of being 1, and w, A t and B t and are same as defined in theorem 1.The above theorem shows that by stochastically blocking gradients from flowing through the h t states of an LSTM with probability 1 − p, we stochastically drop the B t term in the gradient components.

The corollary below shows that in expectation, this results in dampening the B t term compared to the original LSTM gradient.

DISPLAYFORM4 Finally, we note that when training LSTMs with h-detach, we reduce the amount of computation needed.

This is simply because by stochastically blocking the gradient from flowing through the h t hidden states of LSTM, less computation needs to be done during back-propagation through time (BPTT).

This task requires the recurrent network to memorize the network inputs provided at the first few time steps and output them in the same order after a large time delay.

Thus the only way to solve this task is for the network to capture the long term dependency between inputs and targets which requires gradient components carrying this information to flow through many time steps.

We follow the copying task setup identical to BID0 (described in appendix).

Using their data generation process, we sample 100,000 training input-target sequence pairs and 5,000 validation pairs.

We use cross-entropy as our loss to train an LSTM with hidden state size 128 for a maximum of 500-600 epochs.

We use the ADAM optimizer with batch-size 100, learning rate 0.001 and clip the gradient norms to 1.

Figure 2: Validation accuracy curves during training on copying task using vanilla LSTM (left) and LSTM with h-detach with probability 0.25 (middle) and 0.5 (right).

Top row is delay T = 100 and bottom row is delay T = 300.

Each plot contains multiple runs with different seeds.

We see that for T = 100, even the baseline LSTM is able to reach ∼ 100% accuracy for most seeds and the only difference we see between vanilla LSTM and LSTM with h-detach is in terms of convergence.

T = 300 is a more interesting case because it involves longer term dependencies.

In this case we find that h-detach leads to faster convergence and achieves ∼ 100% validation accuracy while being more robust to the choice of seed.

Figure 2 shows the validation accuracy plots for copying task training for T = 100 (top row) and T = 300 (bottom row) without h-detach (left), and with h-detach (middle and right).

Each plot contains runs from the same algorithm with multiple seeds to show a healthy sample of variations using these algorithms.

For T = 100 time delay, we see both vanilla LSTM and LSTM with hdetach converge to 100% accuracy.

For time delay 100 and the training setting used, vanilla LSTM is known to converge to optimal validation performance (for instance, see BID0 ).

Nonetheless, we note that h-detach converges faster in this setting.

A more interesting case is when time decay is set to 300 because it requires capturing longer term dependencies.

In this case, we find that LSTM training without h-detach achieves a validation accuracy of ∼ 82% at best while a number of other seeds converge to much worse performance.

On the other hand, we find that using h-detach with detach probabilities 0.25 and 0.5 achieves the best performance of 100% and converging quickly while being reasonably robust to the choice of seed.

Having shown the benefit of h-detach in terms of training dynamics, we now extend the challenge of the copying task by evaluating how well an LSTM trained on data with a certain time delay generalizes when a larger time delay is used during inference.

This task is referred as the transfer copying task BID9 .

Specifically, we train the LSTM architecture on copying task with delay T = 100 without h-detach and with h-detach with probability 0.25 and 0.5.

We then evaluate the accuracy of the trained model for each setting for various values of T > 100.The results are shown in table 1.

We find that the function learned by LSTM when trained with h-detach generalize significantly better on longer time delays during inference compared with the LSTM trained without h-detach.

This task is a sequential version of the MNIST classification task BID17 .

In this task, an image is fed into the LSTM one pixel per time step and the goal is to predict the label after the last pixel is fed.

We consider two versions of the task: one is which the pixels are read in order (from left to right and top to bottom), and one where all the pixels are permuted in a random but fixed order.

We call the second version the permuted MNIST task or pMNIST in short.

The setup used for this experiment is as follows.

We use 50000 images for training, 10000 for validation and 10000 for testing.

We use the ADAM optimizer with different learning rates-0.001,0.0005 and 0.0001, and a fixed batch size of 100.

We train for 200 epochs and pick our final model based on the best validation score.

We use an LSTM with 100 hidden units.

For h-detach, we do a hyperparameter search on the detach probability in {0.1, 0.25, 0.4, 0.5}. For both pixel by pixel MNIST and pMNIST, we found the detach hyper-parameter of 0.25 to perform best on the validation set for both MNIST and pMNIST.

DISPLAYFORM0 On the sequential MNIST task, both vanilla LSTM and training with h-detach give an accuracy of 98.5%.

Here, we note that the convergence of our method is much faster and is more robust to the different learning rates of the ADAM optimizer as seen in FIG1 .

Refer to appendix (figure 6) for experiments with multiple seeds that shows the robustness of our method to initialization.

In the pMNIST task, we find that training LSTM with h-detach gives a test accuracy of 92.3% which is an improvement over the regular LSTM training which reaches an accuracy of 91.

BID16 97.0 82.0 uRNN BID0 95.1 91.4 Zoneout BID15 -93.1 IndRNN BID18 99 96 h-detach (ours) 98.5 92.3 BID29 and Soft Attention (Xu et al., 2015) and train the LSTM in these models with and without h-detach.

We now evaluate h-detach on an image captioning task which involves using an RNN for generating captions for images.

We use the Microsoft COCO dataset BID19 which contains 82,783 training images and 40,504 validation images.

Since this dataset does not have a standard split for training, validation and test, we follow the setting in BID12 which suggests a split of 80,000 training images and 5,000 images each for validation and test set.

We use two models to test our approach-the Show&Tell encoder-decoder model BID29 which does not employ any attention mechanism, and the 'Show, Attend and Tell' model (Xu et al., 2015) , which uses soft attention.

For feature extraction, we use the 2048-dimensional last layer feature vector of a residual network (Resnet He et al. (2015) ) with 152 layers which was pretrained on ImageNet for image classification.

We use an LSTM with 512 hidden units for caption generation.

We train both the Resnet and LSTM models using the ADAM optimizer (Kingma & Ba, 2014) with a learning rate of 10 −4 and leave the rest of the hyper-parameters as suggested in their paper.

We also perform a small hyperparameter search where we find the optimial value of the h-detach parameter.

We considered values in the set {0.1, 0.25, 0.4, 0.5} and pick the optimal value based on the best validation score.

Similar to BID26 , we early stop based on the validation CIDEr scores and report BLEU-1 to BLEU-4, CIDEr, and Meteor scores.

The results are presented in table 3.

Training the LSTM with h-detach outperforms the baseline LSTM by a good margin for all the metrics and produces the best BLEU-1 to BLEU-3 scores among all the compared methods.

Even for the other metrics, except for the results reported by BID21 , we beat all the other methods reported.

We emphasize that compared to all the other reported methods, h-detach is extremely simple to implement and does not add any computational overhead (in fact reduces computation).

In this section, we first study the effect of removing gradient clipping in the LSTM training and compare how the training of vanilla LSTM and our method get affected.

Getting rid of gradient clipping would be insightful because it would confirm our claim that stochastically blocking gradients through the hidden states h t of the LSTM prevent the growth of gradient magnitude.

We train both models on pixel by pixel MNIST using ADAM without any gradient clipping.

The validation accuracy curves are reported in FIG2 for two different learning rates.

We notice that removing gradient clipping causes the Vanilla LSTM training to become extremely unstable.

h-detach on the Figure 5: Validation accuracy curves for copying task T=100 (left) and pixel by pixel MNIST (right) using LSTM such that gradient is stochastically blocked through the cell state (the probability of detaching the cell state in this experiment is mentioned in sub-titles.).

Blocking gradients from flowing through the cell state path of LSTM (c-detach) leads to significantly worse performance compared even to vanilla LSTM on tasks that requires long term dependencies.

This suggests that the cell state path carry information about long term dependencies.other hand seems robust to removing gradient clipping for both the learning rates used.

Additional experiments with multiple seeds and learning rates can be found in figure 8 in appendix.

Second, we conduct experiments where we stochastically block gradients from flowing through the cell state c t instead of the hidden state h t and observe how the LSTM behaves in such a scenario.

We refer detaching the cell state as c-detach.

The goal of this experiment is to corroborate our hypothesis that the gradients through the cell state path carry information about long term dependencies.

Figure 5 shows the effect of c-detach (with probabilities shown) on copying task and pixel by pixel MNIST task.

We notice in the copying task for T = 100, learning becomes very slow (figure 5 (a)) and does not converge even after 500 epochs, whereas when not detaching the cell state, even the Vanilla LSTM converges in around 150 epochs for most cases for T=100 as shown in the experiments in section 3.1.

For pixel by pixel MNIST (which involves 784 time steps), there is a much larger detrimental effect on learning as we find that none of the seeds cross 60% accuracy at the end of training ( Figure 5 (b) ).

This experiment corroborates our hypothesis that gradients through the cell state contain important components of the gradient signal as blocking them worsens the performance of these models when compared to Vanilla LSTM.

Capturing long term dependencies in data using recurrent neural networks has been long known to be a hard problem BID8 BID1 .

Therefore, there has been a considerable amount of work on addressing this issue.

Prior to the invention of the LSTM architecture (Hochreiter & Schmidhuber, 1997), another class of architectures called NARX (nonlinear autoregressive models with exogenous) recurrent networks BID20 was popular for tasks involving long term dependencies.

More recently gated recurrent unit (GRU) networks BID3 was proposed that adapts some favorable properties of LSTM while requiring fewer parameters.

Other recent recurrent architecture designs that are aimed at preventing EVGP can be found in Zhang et al.(2018), BID11 and BID18 .

Work has also been done towards better optimization for such tasks BID22 BID14 .

Since vanishing and exploding gradient problems BID8 BID2 also hinder this goal, gradient clipping methods have been proposed to alleviate this problem BID27 BID24 ).

Yet another line of work focuses on making use of unitary transition matrices in order to avoid loss of information as hidden states evolve over time.

BID16 propose to initialize recurrent networks with unitary weights while BID0 propose a new network parameterization that ensures that the state transition matrix remains unitary.

Extensions of the unitary RNNs have been proposed in BID32 , BID23 and BID10 .

Very recently, propose to learn an attention mechanism over past hidden states and sparsely back-propagate through paths with high attention weights in order to capture long term dependencies.

BID28 propose to add an unsupervised auxiliary loss to the original objective that is designed to encourage the network to capture such dependencies.

We point out that our proposal in this paper is orthogonal to a number of the aforementioned papers and may even be applied in conjunction to some of them.

Further, our method is specific to LSTM optimization and reduces computation relative to the vanilla LSTM optimization which is in stark contrast to most of the aforementioned approaches which increase the amount of computation needed for training.

In section 3.5 we showed that LSTMs trained with h-detach are stable even without gradient clipping.

We caution that while this is true, in general the gradient magnitude depends on the value of detaching probability used in h-detach.

Hence for the general case, we do not recommend removing gradient clipping.

When training stacked LSTMs, there are two ways in which h-detach can be used: 1) detaching the hidden state of all LSTMs simultaneously for a given time step t depending on the stochastic variable ξ t ) stochastically detaching the hidden state of each LSTM separately.

We leave this for future work.h-detach stochastically blocks the gradient from flowing through the hidden states of LSTM.

In corollary 1, we showed that in expectation, this is equivalent to dampening the gradient components from paths other than the cell state path.

We especially chose this strategy because of its ease of implementation in current auto-differentiation libraries.

Another approach to dampen these gradient components would be to directly multiply these components with a dampening factor.

This feature is currently unavailable in these libraries but may be an interesting direction to look into.

A downside of using this strategy though is that it will not reduce the amount of computation similar to h-detach (although it will not increase the amount of computation compared with vanilla LSTM either).

Regularizing the recurrent weight matrices to have small norm can also potentially prevent the gradient components from the cell state path from being suppressed but it may also restrict the representational power of the model.

Given the superficial similarity of h-detach with dropout, we outline the difference between the two methods.

Dropout randomly masks the hidden units of a network during the forward pass (and can be seen as a variant of the stochastic delta rule BID6 ).

Therefore, a common view of dropout is training an ensemble of networks BID30 .

On the other hand, our method does not mask the hidden units during the forward pass.

It instead randomly blocks the gradient component through the h-states of the LSTM only during the backward pass and does not change the output of the network during forward pass.

More specifically, our theoretical analysis shows the precise behavior of our method: the effect of h-detach is that it changes the update direction used for descent which prevents the gradients through the cell state path from being suppressed.

We would also like to point out that even though we show improvements on the image captioning task, it does not fit the profile of a task involving long term dependencies that we focus on.

We believe the reason why our method leads to improvements on this task is that the gradient components from the cell state path are important for this task and our theoretical analysis shows that h-detach prevents these components from getting suppressed compared with the gradient components from the other paths.

On the same note, we also tried our method on language modeling tasks but did not notice any benefit.

We proposed a simple stochastic algorithm called h-detach aimed at improving LSTM performance on tasks that involve long term dependencies.

We provided a theoretical understanding of the method using a novel analysis of the back-propagation equations of the LSTM architecture.

We note that our method reduces the amount of computation needed during training compared to vanilla LSTM training.

Finally, we empirically showed that h-detach is robust to initialization, makes the convergence of LSTM faster, and/or improves generalization compared to vanilla LSTM (and other existing methods) on various benchmark datasets. .

The next T − 1 entries are set to a 8 , which constitutes a delay.

The next single entry is a 9 , which represents a delimiter, which should indicate to the algorithm that it is now required to reproduce the initial 10 input tokens as output.

The remaining 10 input entries are set to a 8 .

The target sequence consists of T + 10 repeated entries of a 8 , followed by the first 10 entries of the input sequence in exactly the same order.

DISPLAYFORM0 Here denotes the element-wise product, also called the Hadamard product.

σ denotes the sigmoid activation function.

DISPLAYFORM1 For any ∈ {f, g, o, i}, define E (w) to be a matrix of size dim(h t ) × dim([h t ; x t ]).

We set all the elements of this matrix to 0s if if w is not an element of W .

Further, if w = (W ) kl , then (E (w)) kl = 1 and (E (w)) k l = 0 for all (k , l ) = (k, l).

DISPLAYFORM2 Lemma 1 Let us assume w is an entry of the matrix DISPLAYFORM3 Proof By chain rule of total differentiation, DISPLAYFORM4 We note that, DISPLAYFORM5 DISPLAYFORM6 Recall that h t = o t tanh(c t ), and thus DISPLAYFORM7 Using the previous Lemma as well as the above notation, we get DISPLAYFORM8 DISPLAYFORM9 Then, z t = (A t + B t )z t−1In other words, where all the symbols used to define A t and B t are defined in notation 1.Proof By Corollary 2, we get DISPLAYFORM10 Similarly by Corollary 3, we get DISPLAYFORM11 Thus we have DISPLAYFORM12 Applying this formula recursively proves the claim.

Note: Since A t has 0 n 's in the second column of the block matrix representation, it ignores the contribution of z t coming from h t−1 , whereas B t (having non-zero block matrices only in the second column of the block matrix representation) only takes into account the contribution coming from h t−1 .

Hence A t captures the contribution of the gradient coming from the cell state c t−1 .

T andz t be the analogue of z t when applying h-detach with probability p during back-propagation.

Then, z t = (A t + ξ t B t )(A t−1 + ξ t−1 B t−1 ) . . . (A 2 + ξ 2 B 2 )z 1 where ξ t , ξ t−1 , . . . , ξ 2 are i.i.d.

Bernoulli random variables with probability p of being 1, A t and B t and are same as defined in theorem 1.Proof Replacing DISPLAYFORM13 Iterating this formula gives, z t = (A t + ξ t B t )(A t−1 + ξ t−1 B t−1 ) . . . (A 3 + ξ 3 B 3 )z 2Corollary 4 E[z t ] = (A t + pB t )(A t−1 + pB t−1 ) . . . (A 3 + pB 3 )z 2It suffices to take the expectation both sides, and use independence of ξ t 's.

@highlight

A simple algorithm to improve optimization and handling of long term dependencies in LSTM

@highlight

The paper introduces a simple stochastic algorithm called h-detach that is specific to LSTM optimization and targeted towards addressing this problem.

@highlight

Proposes a simple modification to the training process of the LSTM to facilitate gradient propogation along cell states, or the "linear temporal path"